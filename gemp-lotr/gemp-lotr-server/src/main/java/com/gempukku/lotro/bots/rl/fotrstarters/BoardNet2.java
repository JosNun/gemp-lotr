package com.gempukku.lotro.bots.rl.fotrstarters;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import com.alibaba.fastjson2.JSON;
import com.alibaba.fastjson2.JSONArray;
import com.alibaba.fastjson2.JSONObject;
import com.gempukku.lotro.bots.simulation.BoardDump;
import com.gempukku.lotro.game.state.LotroGame;
import com.gempukku.lotro.logic.decisions.AwaitingDecision;

import java.io.IOException;
import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

/**
 * The combined network (nn/board_net2.py, exported by nn/export_onnx2.py) run from the bot through ONNX Runtime: for
 * a card-action prompt of any phase, reconcile or "move again?", the log-probability of each option (pass last at a
 * card action); for an assignment, per minion, of each character and of leaving it unassigned; and the six ways the
 * game ends from the deciding player's side.
 * <p>
 * The board is read by {@link BoardDump#record} -- the code that wrote the training records -- and turned into arrays
 * here exactly as board_net2.encode turns a record into them. {@code BoardNet2 parity <model dir>} checks that against
 * PyTorch on 200 July records (board-net2-parity.jsonl): an encoder that differs feeds the model what it never saw.
 */
public final class BoardNet2 {
    static final int MAX_CARDS = 64, MAX_OPTS = 24, MAX_TOK = 20, MAX_BAG = 48, MAX_MIN = 10, MAX_FPC = 12;
    static final int DYN = 12, N_KW = 16;
    static final List<String> ZONES = List.of("FREE_CHARACTERS", "SHADOW_CHARACTERS", "SUPPORT", "ATTACHED", "HAND",
            "ADVENTURE_PATH", "STACKED", "DISCARD", "DECK", "DEAD", "ADVENTURE_DECK", "REMOVED");
    static final List<String> KINDS = List.of("other", "play", "use", "heal", "transfer", "keep", "discard", "yes", "no");
    static final List<String> CONTEXTS = List.of("fellowship", "shadow", "maneuver", "archery", "assignment",
            "skirmish", "regroup", "optional", "required", "other", "reconcile", "move", "assign fp", "assign shadow");
    static final List<String> BAGS = List.of("myDiscard", "oppDiscard", "myDead", "oppDead", "oppSeen", "myDeck");

    /** What the network makes of a board. */
    public record Result(String kind, String context, double[] options, double[][] assign, double[] outcome) {
        /** The deciding player's chance of winning: the sum of the three ways to win. */
        public double win() {
            return outcome[0] + outcome[1] + outcome[2];
        }
    }

    private final OrtEnvironment env = OrtEnvironment.getEnvironment();
    private final OrtSession session, judge;
    private final Map<String, Integer> rows = new HashMap<>();
    private final Set<Integer> seen = new HashSet<>();
    /** The trees' share when mixed, per context (board-net2-mix: "context weight" lines, "default" for the rest). */
    private final Map<String, Double> treeWeight = new HashMap<>();

    private BoardNet2(Path dir) throws IOException, OrtException {
        treeWeight.put("default", 0.6);
        Path mix = dir.resolve("board-net2-mix");
        if (Files.exists(mix))
            for (String line : Files.readAllLines(mix)) {
                String[] kv = line.trim().split("\\s+(?=[-0-9.]+$)");
                if (kv.length == 2)
                    treeWeight.put(kv[0].trim(), Double.parseDouble(kv[1]));
            }
        List<String> lines = Files.readAllLines(dir.resolve("board-net2-cards.tsv"));
        for (int i = 1; i < lines.size(); i++) // row 0 is unknown; the header is line 0
            rows.put(lines.get(i).substring(0, lines.get(i).indexOf('\t')), i);
        for (String s : Files.readAllLines(dir.resolve("board-net2-seen.txt")))
            if (!s.isBlank())
                seen.add(Integer.parseInt(s.trim()));
        var options = new OrtSession.SessionOptions();
        options.setIntraOpNumThreads(1); // the bot plays many games side by side
        session = env.createSession(dir.resolve("board-net2.onnx").toString(), options);
        Path j = dir.resolve("board-net2-judge.onnx");
        judge = Files.exists(j) ? env.createSession(j.toString(), options) : null;
    }

    /** The network in {@code dir}, or null if there is none. */
    public static BoardNet2 load(Path dir) throws IOException {
        if (!Files.exists(dir.resolve("board-net2.onnx")))
            return null;
        try {
            BoardDump.LIVE = true;
            BoardDump.TRUE_DECK = true; // a bot knows its own deck; replays read --decklists instead (BoardDump.REPLAY)
            return new BoardNet2(dir);
        } catch (OrtException e) {
            throw new IOException("board-net2.onnx in " + dir, e);
        }
    }

    /** The trees' share at {@code context} (a CONTEXTS name). */
    public double treeWeight(String context) {
        return treeWeight.getOrDefault(context, treeWeight.get("default"));
    }

    /** The set-2 state the network was trained with, whatever feature set the trees use. */
    public static double[] state(com.gempukku.lotro.game.state.GameState gs, AwaitingDecision d, String player) {
        int previous = CardFeatures.version();
        try {
            CardFeatures.setVersion(GeneralStateFeatures.VERSION);
            return new GeneralStateFeatures().extractFeatures(gs, d, player);
        } catch (RuntimeException e) {
            return null;
        } finally {
            CardFeatures.setVersion(previous);
        }
    }

    /** The network's reading of the decision the bot is answering, from its game state; null if none. */
    public Result read(com.gempukku.lotro.game.state.GameState gs, AwaitingDecision d, String player) {
        double[] state = state(gs, d, player);
        return state == null ? null : read(gs.getGame(), d, player, state);
    }

    /** log-softmax of {@code v} (-inf entries stay -inf). */
    public static double[] logSoftmax(double[] v) {
        double max = Double.NEGATIVE_INFINITY, sum = 0;
        for (double x : v)
            max = Math.max(max, x);
        for (double x : v)
            sum += x == Double.NEGATIVE_INFINITY ? 0 : Math.exp(x - max);
        double norm = max + Math.log(sum);
        double[] out = new double[v.length];
        for (int i = 0; i < v.length; i++)
            out[i] = v[i] - norm;
        return out;
    }

    /** (1 - w) * network + w * trees, both as log-probabilities over the same options. */
    public static double[] mix(double[] net, double[] trees, double w) {
        double[] a = logSoftmax(net), b = logSoftmax(trees), out = new double[a.length];
        for (int i = 0; i < a.length; i++)
            out[i] = (1 - w) * a[i] + w * b[i];
        return out;
    }

    public static int argmax(double[] v) {
        int best = -1;
        for (int i = 0; i < v.length; i++)
            if (best < 0 || v[i] > v[best])
                best = i;
        return best;
    }

    /** Whether a separate outcome-only judge was exported alongside. */
    public boolean hasJudge() {
        return judge != null;
    }

    /** The network's reading of {@code d}, unanswered; null if it does not take this decision or cannot read it. */
    public Result read(LotroGame game, AwaitingDecision d, String player, double[] state) {
        JSONObject rec = BoardDump.record(game, d, player, null, state);
        return rec == null ? null : run(rec, false);
    }

    /** The judge's (or, without one, the network's) six outcome probabilities for the board at {@code d}. */
    public double[] outcome(LotroGame game, AwaitingDecision d, String player, double[] state) {
        JSONObject rec = BoardDump.record(game, d, player, null, state);
        Result r = rec == null ? null : run(rec, judge != null);
        return r == null ? null : r.outcome();
    }

    private static String base(String bp) {
        return bp.replaceAll("[*T]+$", "");
    }

    private int row(String bp) {
        Integer r = rows.get(bp);
        if (r == null)
            r = rows.get(base(bp));
        return r == null ? 0 : r;
    }

    static String context(JSONObject r) {
        String k = r.getString("kind");
        if (k.equals("move") || k.equals("reconcile"))
            return k;
        if (k.equals("assign"))
            return r.getIntValue("fp") == 1 ? "assign fp" : "assign shadow";
        String t = r.getString("ctx") == null ? "" : r.getString("ctx");
        String[][] keys = {{"Fellowship action", "fellowship"}, {"Shadow action", "shadow"}, {"Maneuver", "maneuver"},
                {"Archery", "archery"}, {"Assignment action", "assignment"},
                {"Choose action to play or Pass", "skirmish"}, {"Regroup", "regroup"}, {"Optional", "optional"},
                {"Required", "required"}};
        for (String[] kv : keys)
            if (t.contains(kv[0]))
                return kv[1];
        return "other";
    }

    static int optionKind(String kind, JSONObject o) {
        String text = o.getString("text");
        if (kind.equals("move"))
            return KINDS.indexOf(text.equals("Yes") ? "yes" : "no");
        if (kind.equals("reconcile"))
            return KINDS.indexOf(text);
        String t = text.toLowerCase();
        for (int i = 1; i < KINDS.size(); i++)
            if (t.startsWith(KINDS.get(i)))
                return i;
        return 0;
    }

    private static long clamp(long v) {
        return Math.min(v, 127);
    }

    /** board_net2.encode for one record, then the network; {@code useJudge}: the outcome-only model instead. */
    Result run(JSONObject r, boolean useJudge) {
        String kind = r.getString("kind");
        JSONArray opts = r.getJSONArray("opts") != null ? r.getJSONArray("opts") : new JSONArray();
        if (opts.size() > MAX_OPTS)
            return null;
        JSONArray allCards = r.getJSONArray("cards");
        int nc = Math.min(MAX_CARDS, allCards.size());
        long[] cf = new long[MAX_CARDS], ci = new long[MAX_CARDS], cz = new long[MAX_CARDS];
        long[] cd = new long[MAX_CARDS * DYN], ck = new long[MAX_CARDS * N_KW];
        boolean[] cm = new boolean[MAX_CARDS];
        for (int i = 0; i < nc; i++) {
            JSONObject c = allCards.getJSONObject(i);
            int row = row(c.getString("bp"));
            cf[i] = row;
            ci[i] = seen.contains(row) ? row : 0;
            cm[i] = true;
            int z = ZONES.indexOf(c.getString("zone"));
            cz[i] = z >= 0 ? z + 1 : 0;
            int att = c.getIntValue("att");
            if (!(att >= 0 && att < MAX_CARDS))
                att = -1;
            long[] v = {c.getIntValue("own"), clamp(c.getIntValue("str")), clamp(c.getIntValue("vit")),
                    clamp(c.getIntValue("res")), clamp(c.getIntValue("wnd")), c.getIntValue("act"),
                    c.getIntValue("skir"), c.getIntValue("asg"), c.getIntValue("rb"), att >= 0 ? 1 : 0, att + 1,
                    clamp(c.getIntValue("cost"))};
            System.arraycopy(v, 0, cd, i * DYN, DYN);
            JSONArray kw = c.getJSONArray("kw");
            if (kw != null)
                for (int k = 0; k < kw.size(); k++) {
                    JSONArray e = kw.getJSONArray(k);
                    ck[i * N_KW + e.getIntValue(0)] = clamp(e.getIntValue(1));
                }
        }
        long[] os = new long[MAX_OPTS], ok = new long[MAX_OPTS], oc = new long[MAX_OPTS];
        long[] ti = new long[MAX_OPTS * MAX_TOK], tv = new long[MAX_OPTS * MAX_TOK];
        boolean[] om = new boolean[MAX_OPTS + 1];
        Arrays.fill(os, -1);
        for (int j = 0; j < opts.size(); j++) {
            JSONObject o = opts.getJSONObject(j);
            int src = o.getIntValue("src");
            os[j] = src >= 0 && src < nc ? src : -1;
            ok[j] = optionKind(kind, o);
            oc[j] = clamp(o.getIntValue("cost"));
            JSONArray tok = o.getJSONArray("tok");
            if (tok != null)
                for (int t = 0; t < Math.min(MAX_TOK, tok.size()); t++) {
                    JSONArray e = tok.getJSONArray(t);
                    ti[j * MAX_TOK + t] = (long) e.getDoubleValue(0) + 1;
                    tv[j * MAX_TOK + t] = clamp((long) e.getDoubleValue(1));
                }
            om[j] = true;
        }
        long[] am = new long[MAX_MIN], af = new long[MAX_FPC];
        Arrays.fill(am, -1);
        Arrays.fill(af, -1);
        int nMin = 0, nFp = 0;
        if (kind.equals("assign")) {
            JSONObject asg = r.getJSONObject("asg");
            JSONArray fp = asg.getJSONArray("fp"), mins = asg.getJSONArray("min");
            if (fp.size() > MAX_FPC || mins.size() > MAX_MIN)
                return null;
            for (int i = 0; i < fp.size(); i++) {
                int x = fp.getIntValue(i);
                if (x < 0 || x >= MAX_CARDS)
                    return null;
                af[i] = x;
            }
            for (int i = 0; i < mins.size(); i++) {
                int x = mins.getIntValue(i);
                if (x < 0 || x >= MAX_CARDS)
                    return null;
                am[i] = x;
            }
            nMin = mins.size();
            nFp = fp.size();
        } else if (kind.equals("action"))
            om[MAX_OPTS] = true; // pass
        long[] bg = new long[BAGS.size() * MAX_BAG], bc = new long[BAGS.size() * MAX_BAG];
        JSONObject bags = r.getJSONObject("bags") != null ? r.getJSONObject("bags") : new JSONObject();
        for (int k = 0; k < BAGS.size(); k++) {
            JSONArray bag = bags.getJSONArray(BAGS.get(k));
            if (bag == null)
                continue;
            // Counter.most_common: by count, ties in order of first appearance
            LinkedHashMap<String, Integer> counts = new LinkedHashMap<>();
            for (int i = 0; i < bag.size(); i++)
                counts.merge(base(bag.getString(i)), 1, Integer::sum);
            List<Map.Entry<String, Integer>> es = new ArrayList<>(counts.entrySet());
            es.sort((a, b) -> b.getValue() - a.getValue());
            for (int i = 0; i < Math.min(MAX_BAG, es.size()); i++) {
                Integer row = rows.get(es.get(i).getKey());
                bg[k * MAX_BAG + i] = row != null && row != 0 ? row : -1;
                bc[k * MAX_BAG + i] = clamp(es.get(i).getValue());
            }
        }
        JSONArray st = r.getJSONArray("state");
        float[] state = new float[st.size() + 3];
        for (int i = 0; i < st.size(); i++)
            state[i] = st.getFloatValue(i);
        JSONArray deck = bags.getJSONArray("myDeck");
        state[st.size()] = (deck != null ? deck.size() : 0) / 20f;
        state[st.size() + 1] = bags.getIntValue("myDeckUnknown") / 20f;
        state[st.size() + 2] = bags.containsKey("myDeck") ? 1f : 0f;
        String context = context(r);
        long cx = CONTEXTS.indexOf(context);
        OrtSession s = useJudge && judge != null ? judge : session;
        Map<String, OnnxTensor> in = new LinkedHashMap<>();
        try {
            in.put("state", OnnxTensor.createTensor(env, FloatBuffer.wrap(state), new long[]{1, state.length}));
            in.put("cx", OnnxTensor.createTensor(env, LongBuffer.wrap(new long[]{cx}), new long[]{1}));
            in.put("cf", t(cf, 1, MAX_CARDS));
            in.put("ci", t(ci, 1, MAX_CARDS));
            in.put("cm", OnnxTensor.createTensor(env, new boolean[][]{cm}));
            in.put("cd", t(cd, 1, MAX_CARDS, DYN));
            in.put("cz", t(cz, 1, MAX_CARDS));
            in.put("ck", t(ck, 1, MAX_CARDS, N_KW));
            in.put("os", t(os, 1, MAX_OPTS));
            in.put("ok", t(ok, 1, MAX_OPTS));
            in.put("oc", t(oc, 1, MAX_OPTS));
            in.put("ti", t(ti, 1, MAX_OPTS, MAX_TOK));
            in.put("tv", t(tv, 1, MAX_OPTS, MAX_TOK));
            in.put("om", OnnxTensor.createTensor(env, new boolean[][]{om}));
            in.put("am", t(am, 1, MAX_MIN));
            in.put("af", t(af, 1, MAX_FPC));
            in.put("bg", t(bg, 1, BAGS.size(), MAX_BAG));
            in.put("bc", t(bc, 1, BAGS.size(), MAX_BAG));
            try (OrtSession.Result res = s.run(in)) {
                float[] lp = ((float[][]) res.get(0).getValue())[0];
                float[][] la = ((float[][][]) res.get(1).getValue())[0];
                float[] w = ((float[][]) res.get(2).getValue())[0];
                double[] options;
                if (kind.equals("action")) {
                    options = new double[opts.size() + 1];
                    for (int j = 0; j < opts.size(); j++)
                        options[j] = lp[j];
                    options[opts.size()] = lp[MAX_OPTS];
                } else {
                    options = new double[opts.size()];
                    for (int j = 0; j < opts.size(); j++)
                        options[j] = lp[j];
                }
                double[][] assign = new double[nMin][nFp + 1];
                for (int m = 0; m < nMin; m++) {
                    for (int f = 0; f < nFp; f++)
                        assign[m][f] = la[m][f];
                    assign[m][nFp] = la[m][MAX_FPC];
                }
                double[] outcome = new double[w.length];
                for (int i = 0; i < w.length; i++)
                    outcome[i] = w[i];
                return new Result(kind, context, options, assign, outcome);
            }
        } catch (OrtException | RuntimeException e) {
            return null;
        } finally {
            for (OnnxTensor x : in.values())
                x.close();
        }
    }

    private OnnxTensor t(long[] v, long... shape) throws OrtException {
        return OnnxTensor.createTensor(env, LongBuffer.wrap(v), shape);
    }

    /** {@code parity <model dir>}: the Java encoder against PyTorch on the exported July sample. */
    public static void main(String[] args) throws Exception {
        if (args.length != 2 || !args[0].equals("parity"))
            throw new IllegalArgumentException("usage: BoardNet2 parity <model dir>");
        Path dir = Path.of(args[1]);
        BoardNet2 net = load(dir);
        int n = 0, same = 0, failed = 0;
        double worst = 0, worstWin = 0;
        for (String line : Files.readAllLines(dir.resolve("board-net2-parity.jsonl"))) {
            JSONObject p = JSON.parseObject(line);
            Result r = net.run(p.getJSONObject("record"), false);
            n++;
            if (r == null) {
                failed++;
                continue;
            }
            // the parity file holds the options then pass; Java's options end with pass only at a card action
            JSONArray want = p.getJSONArray("opt");
            int argGot = 0, argWant = 0;
            for (int j = 0; j < r.options().length; j++) {
                double w = want.getDoubleValue(j);
                worst = Math.max(worst, Math.abs(r.options()[j] - w));
                if (r.options()[j] > r.options()[argGot])
                    argGot = j;
                if (w > want.getDoubleValue(argWant))
                    argWant = j;
            }
            same += argGot == argWant ? 1 : 0;
            JSONArray win = p.getJSONArray("win");
            for (int i = 0; i < win.size(); i++)
                worstWin = Math.max(worstWin, Math.abs(r.outcome()[i] - win.getDoubleValue(i)));
        }
        System.out.printf("%d records, %d the encoder could not read; options max |diff| %.2e, same choice %d of %d; "
                + "outcome max |diff| %.2e%n", n, failed, worst, same, n - failed, worstWin);
        System.exit(0);
    }
}
