package com.gempukku.lotro.bots.rl.fotrstarters;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import com.gempukku.lotro.common.CardType;
import com.gempukku.lotro.common.Zone;
import com.gempukku.lotro.game.state.LotroGame;
import com.gempukku.lotro.game.PhysicalCard;
import com.gempukku.lotro.game.state.GameState;
import com.gempukku.lotro.logic.decisions.AwaitingDecision;

import java.io.IOException;
import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

/**
 * The board network (nn/board_net.py, exported by nn/export_onnx.py) run from the bot through ONNX Runtime: the
 * log-probability of each option of a Fellowship or Shadow card-action decision, pass last. Its inputs are built here
 * exactly as board_net.encode builds them from HumanReplay --boards records (BoardDump): every card in play, in the
 * deciding player's hand and behind an option as an item, the options with their card, kind, cost and ability tokens,
 * and both discard and dead piles. Any difference between the two encoders is a model fed what it never saw, so the
 * July agreement through the bot is the check that they agree.
 * <p>
 * On held-out July, people's choices: the network 66.0%, the trees 64.5%, 0.6 network + 0.4 trees 66.5%.
 */
public final class BoardNet {
    static final int MAX_CARDS = 48, MAX_OPTS = 24, MAX_TOK = 20, MAX_BAG = 40, TWILIGHT = 6;
    static final List<String> ZONES = List.of("FREE_CHARACTERS", "SHADOW_CHARACTERS", "SUPPORT", "ATTACHED", "HAND",
            "ADVENTURE_PATH", "STACKED", "DISCARD", "DECK", "DEAD");
    static final List<String> KINDS = List.of("play", "use", "heal", "transfer");
    public static final List<String> CONTEXTS = List.of("Play Fellowship action or Pass", "Play Shadow action or Pass");

    private final OrtEnvironment env = OrtEnvironment.getEnvironment();
    private final OrtSession session;
    private final Map<String, Integer> rows = new HashMap<>();
    private final Set<Integer> seen = new HashSet<>();
    private final int stateWidth;

    private BoardNet(Path dir, int stateWidth) throws IOException, OrtException {
        this.stateWidth = stateWidth;
        List<String> lines = Files.readAllLines(dir.resolve("board-net-cards.tsv"));
        for (int i = 1; i < lines.size(); i++) // row 0 is unknown; the header is line 0
            rows.put(lines.get(i).substring(0, lines.get(i).indexOf('\t')), i);
        for (String s : Files.readAllLines(dir.resolve("board-net-seen.txt")))
            if (!s.isBlank())
                seen.add(Integer.parseInt(s.trim()));
        session = env.createSession(dir.resolve("board-net.onnx").toString(), new OrtSession.SessionOptions());
    }

    /** The network in {@code dir}, or null if there is none. */
    public static BoardNet load(Path dir) throws IOException {
        if (!Files.exists(dir.resolve("board-net.onnx")))
            return null;
        try {
            return new BoardNet(dir, GeneralStateFeatures.length());
        } catch (OrtException e) {
            throw new IOException("board-net.onnx in " + dir, e);
        }
    }

    public static boolean applies(AwaitingDecision d) {
        return d.getText() != null && CONTEXTS.contains(d.getText());
    }

    private int row(String bp) {
        Integer r = rows.get(bp);
        if (r == null)
            r = rows.get(bp.replaceAll("[*T]+$", ""));
        return r == null ? 0 : r;
    }

    /**
     * Log-probabilities over the decision's options, then pass (length options + 1); null if the network does not
     * read this decision (another prompt, more than {@link #MAX_OPTS} options) or it fails.
     */
    public double[] logProbs(GameState gs, AwaitingDecision d, String player, double[] state, List<String> vocab) {
        String[] ids = d.getDecisionParameters().getOrDefault("actionId", new String[0]);
        if (!applies(d) || ids.length == 0 || ids.length > MAX_OPTS || state.length != stateWidth)
            return null;
        try {
            LotroGame game = gs.getGame();
            var mq = game.getModifiersQuerying();
            String[] cardIds = d.getDecisionParameters().getOrDefault("cardId", new String[0]);
            String[] texts = d.getDecisionParameters().getOrDefault("actionText", new String[0]);
            List<PhysicalCard> items = new ArrayList<>();
            Map<Integer, Integer> index = new HashMap<>();
            for (PhysicalCard c : gs.getInPlay())
                add(items, index, c);
            for (PhysicalCard c : gs.getHand(player))
                add(items, index, c);
            for (String cid : cardIds)
                try {
                    add(items, index, gs.findCardById(Integer.parseInt(cid)));
                } catch (RuntimeException ignored) {
                }
            Set<Integer> inSkirmish = new HashSet<>(), assigned = new HashSet<>(), bearers = new HashSet<>();
            var sk = gs.getSkirmish();
            if (sk != null) {
                if (sk.getFellowshipCharacter() != null)
                    inSkirmish.add(sk.getFellowshipCharacter().getCardId());
                for (PhysicalCard m : sk.getShadowCharacters())
                    inSkirmish.add(m.getCardId());
            }
            for (var a : gs.getAssignments()) {
                assigned.add(a.getFellowshipCharacter().getCardId());
                for (PhysicalCard m : a.getShadowCharacters())
                    assigned.add(m.getCardId());
            }
            for (String p : gs.getPlayerNames()) {
                PhysicalCard rb = gs.getRingBearer(p);
                if (rb != null)
                    bearers.add(rb.getCardId());
            }
            int cdw = 10 + ZONES.size();
            long[] cf = new long[MAX_CARDS], ci = new long[MAX_CARDS];
            float[] cd = new float[MAX_CARDS * cdw];
            boolean[] cm = new boolean[MAX_CARDS];
            for (int i = 0; i < Math.min(MAX_CARDS, items.size()); i++) {
                PhysicalCard c = items.get(i);
                int r = row(c.getBlueprintId());
                cf[i] = r;
                ci[i] = seen.contains(r) ? r : 0;
                cm[i] = true;
                PhysicalCard on = c.getAttachedTo() != null ? c.getAttachedTo() : c.getStackedOn();
                int att = on != null && index.containsKey(on.getCardId()) ? index.get(on.getCardId()) : -1;
                CardType type = c.getBlueprint().getCardType();
                boolean character = type == CardType.COMPANION || type == CardType.ALLY || type == CardType.MINION;
                boolean live = character && c.getZone() != null && c.getZone().isInPlay();
                boolean inPlay = c.getZone() != null && c.getZone().isInPlay();
                int o = i * cdw;
                cd[o] = c.getOwner().equals(player) ? 1 : 0;
                cd[o + 1] = live ? mq.getStrength(game, c) / 10f : 0;
                cd[o + 2] = live ? mq.getVitality(game, c) / 5f : 0;
                cd[o + 3] = inPlay ? gs.getWounds(c) / 5f : 0;
                cd[o + 4] = inSkirmish.contains(c.getCardId()) ? 1 : 0;
                cd[o + 5] = assigned.contains(c.getCardId()) ? 1 : 0;
                cd[o + 6] = bearers.contains(c.getCardId()) ? 1 : 0;
                cd[o + 7] = att >= 0 ? 1 : 0;
                cd[o + 8] = (att + 1) / (float) MAX_CARDS;
                cd[o + 9] = 1;
                int z = ZONES.indexOf(String.valueOf(c.getZone()));
                if (z >= 0)
                    cd[o + 10 + z] = 1;
            }
            long[] os = new long[MAX_OPTS], ok = new long[MAX_OPTS], ti = new long[MAX_OPTS * MAX_TOK];
            float[] oc = new float[MAX_OPTS], tv = new float[MAX_OPTS * MAX_TOK];
            boolean[] om = new boolean[MAX_OPTS + 1];
            Arrays.fill(os, -1);
            String context = com.gempukku.lotro.bots.rl.fotrstarters.models.cardaction.AnyPhaseCardActionTrainer
                    .contextName(d.getText());
            for (int j = 0; j < ids.length; j++) {
                PhysicalCard card = null;
                int src = -1;
                try {
                    card = gs.findCardById(Integer.parseInt(cardIds[j]));
                    src = index.getOrDefault(Integer.parseInt(cardIds[j]), -1);
                } catch (RuntimeException ignored) {
                }
                os[j] = src >= 0 && src < MAX_CARDS ? src : -1;
                String text = j < texts.length ? texts[j] : "";
                String lower = text.toLowerCase();
                for (int k = 0; k < KINDS.size(); k++)
                    if (lower.startsWith(KINDS.get(k))) {
                        ok[j] = k + 1;
                        break;
                    }
                int cost = 0;
                if (card != null && lower.startsWith("play"))
                    try {
                        cost = mq.getTwilightCostToPlay(game, card, null, 0, false);
                    } catch (RuntimeException e) {
                        cost = card.getBlueprint().getTwilightCost();
                    }
                oc[j] = cost;
                if (card != null && vocab != null) {
                    double[] t = CardTextFeatures.of(vocab, card.getBlueprintId(), text, context);
                    int n = 0;
                    for (int k = 0; k < t.length && n < MAX_TOK; k++)
                        if (t[k] != 0) {
                            ti[j * MAX_TOK + n] = k + 1;
                            tv[j * MAX_TOK + n] = (float) t[k];
                            n++;
                        }
                }
                om[j] = true;
            }
            om[MAX_OPTS] = true;
            long[] bg = new long[4 * MAX_BAG];
            String opp = gs.getPlayerNames().stream().filter(p -> !p.equals(player)).findFirst().orElse(null);
            List<List<? extends PhysicalCard>> bags = new ArrayList<>();
            bags.add(gs.getDiscard(player));
            bags.add(opp != null ? gs.getDiscard(opp) : List.of());
            bags.add(gs.getDeadPile(player));
            bags.add(opp != null ? gs.getDeadPile(opp) : List.of());
            for (int k = 0; k < 4; k++) {
                List<? extends PhysicalCard> bag = bags.get(k);
                for (int i = 0; bag != null && i < Math.min(MAX_BAG, bag.size()); i++) {
                    int r = row(bag.get(i).getBlueprintId());
                    bg[k * MAX_BAG + i] = r != 0 ? r : -1;
                }
            }
            float[] st = new float[state.length];
            for (int i = 0; i < state.length; i++)
                st[i] = (float) state[i];
            long cx = CONTEXTS.indexOf(d.getText());
            Map<String, OnnxTensor> in = new LinkedHashMap<>();
            try {
                in.put("state", OnnxTensor.createTensor(env, FloatBuffer.wrap(st), new long[]{1, st.length}));
                in.put("cx", OnnxTensor.createTensor(env, LongBuffer.wrap(new long[]{cx}), new long[]{1}));
                in.put("cf", OnnxTensor.createTensor(env, LongBuffer.wrap(cf), new long[]{1, MAX_CARDS}));
                in.put("cd", OnnxTensor.createTensor(env, FloatBuffer.wrap(cd), new long[]{1, MAX_CARDS, cdw}));
                in.put("ci", OnnxTensor.createTensor(env, LongBuffer.wrap(ci), new long[]{1, MAX_CARDS}));
                in.put("cm", OnnxTensor.createTensor(env, new boolean[][]{cm}));
                in.put("os", OnnxTensor.createTensor(env, LongBuffer.wrap(os), new long[]{1, MAX_OPTS}));
                in.put("ok", OnnxTensor.createTensor(env, LongBuffer.wrap(ok), new long[]{1, MAX_OPTS}));
                in.put("om", OnnxTensor.createTensor(env, new boolean[][]{om}));
                in.put("oc", OnnxTensor.createTensor(env, FloatBuffer.wrap(oc), new long[]{1, MAX_OPTS}));
                in.put("ti", OnnxTensor.createTensor(env, LongBuffer.wrap(ti), new long[]{1, MAX_OPTS, MAX_TOK}));
                in.put("tv", OnnxTensor.createTensor(env, FloatBuffer.wrap(tv), new long[]{1, MAX_OPTS, MAX_TOK}));
                in.put("bg", OnnxTensor.createTensor(env, LongBuffer.wrap(bg), new long[]{1, 4, MAX_BAG}));
                try (OrtSession.Result r = session.run(in)) {
                    float[] lp = ((float[][]) r.get(0).getValue())[0];
                    double[] out = new double[ids.length + 1];
                    for (int j = 0; j < ids.length; j++)
                        out[j] = lp[j];
                    out[ids.length] = lp[MAX_OPTS];
                    return out;
                }
            } finally {
                for (OnnxTensor t : in.values())
                    t.close();
            }
        } catch (OrtException | RuntimeException e) {
            return null;
        }
    }

    private static void add(List<PhysicalCard> items, Map<Integer, Integer> index, PhysicalCard c) {
        if (c == null || index.containsKey(c.getCardId()))
            return;
        index.put(c.getCardId(), items.size());
        items.add(c);
    }
}
