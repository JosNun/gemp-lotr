package com.gempukku.lotro.bots.simulation;

import com.gempukku.lotro.bots.BotDecisionFallback;
import com.gempukku.lotro.bots.BotPlayer;
import com.gempukku.lotro.bots.rl.fotrstarters.BoardNet2;
import com.gempukku.lotro.game.PhysicalCard;
import com.gempukku.lotro.game.state.GameState;
import com.gempukku.lotro.logic.decisions.AwaitingDecision;
import com.gempukku.lotro.logic.timing.DefaultLotroGame;

import java.util.*;
import java.util.function.Function;

/**
 * Search at play time, only where it pays (BOT-NOTES, 2026-11-01): the bot answers as usual unless the decision is of a
 * kind search helps (skirmish plays, "move again?", Fellowship and Shadow plays -- search.contexts) and the network is
 * unsure (its top option under search.gate). Then its top search.k options and the bot's own answer are each played
 * out search.rollouts times on copies of the game -- the opponent's hand dealt again from what this player has not
 * seen, every option into the same futures, the bot playing both sides -- for a phase (a turn for moves and Fellowship
 * plays), and scored by the judge. The answer is the option with the most log-probability (what people would do) plus
 * value / search.lambda (what wins): the human anchor of the training targets, at play time. search.maxPerGame caps the
 * cost.
 */
public final class Searching implements BotPlayer {
    static final int K = Integer.getInteger("search.k", 3), ROLLOUTS = Integer.getInteger("search.rollouts", 4),
            MAX_PER_GAME = Integer.getInteger("search.maxPerGame", 60);
    /** search.stage1: play-outs a candidate in a first round (0: all of them at once); search.spread: go on past it. */
    static final int STAGE1 = Integer.getInteger("search.stage1", 0);
    static final double SPREAD = Double.parseDouble(System.getProperty("search.spread", "0.1"));
    /**
     * search.stage1Mode=judge: the first round only applies each candidate and judges the next position of this
     * player's (no play-out), and the second round is then all search.rollouts play-outs; "playout" (the default): the
     * first round is the first play-out. search.keep: candidates carried into the second round, the best first.
     */
    static final boolean STAGE1_JUDGE = "judge".equals(System.getProperty("search.stage1Mode"));
    static final int KEEP = Integer.getInteger("search.keep", 99);
    static final double GATE = Double.parseDouble(System.getProperty("search.gate", "0.6")),
            LAMBDA = Double.parseDouble(System.getProperty("search.lambda", "0.05"));
    static final Set<String> CONTEXTS = new HashSet<>(List.of(System.getProperty("search.contexts",
            "skirmish,move,fellowship,shadow").split(",")));
    static final Set<String> TURN_LONG = Set.of("move", "fellowship");

    /**
     * -Dsearch.boards=&lt;file&gt;: every searched decision as a board record with the search's answer and each candidate's
     * value ("searched": [option index, -1 for pass; value]), the soft targets board_net2.py --teacher reads -- what the
     * search knows, taught to the network that plays without it. Games are named as BotVsBot names them for outcomes.
     */
    static final java.io.PrintWriter BOARDS = Recording.open(System.getProperty("search.boards"));

    /** Searches made, decisions seen, by the bots of this run. */
    static final java.util.concurrent.atomic.AtomicLong searched = new java.util.concurrent.atomic.AtomicLong(),
            changed = new java.util.concurrent.atomic.AtomicLong(), nanos = new java.util.concurrent.atomic.AtomicLong(),
            playOuts = new java.util.concurrent.atomic.AtomicLong(), steps = new java.util.concurrent.atomic.AtomicLong(),
            judged = new java.util.concurrent.atomic.AtomicLong(), stoppedEarly = new java.util.concurrent.atomic.AtomicLong();

    private final BotPlayer inner;
    private final BoardNet2 net;
    private final GameCopier copier;
    private final Function<String, BotPlayer> policy;
    /**
     * The bot that answers has no network (-Dsearch.root=&lt;a trees-only directory&gt;, 2026-11-09): its trees' scores
     * gate the search, name the candidates and anchor the answer, and the network is left the judge alone. The trees
     * by themselves are level with sp-m3 against bots (49.8 +- 1.4) at an eighth of its cost a side.
     */
    private final boolean treesRoot;
    private final Map<Object, int[]> perGame = new WeakHashMap<>();

    public Searching(BotPlayer inner, BoardNet2 net, GameCopier copier, Function<String, BotPlayer> policy) {
        this(inner, net, copier, policy, false);
    }

    public Searching(BotPlayer inner, BoardNet2 net, GameCopier copier, Function<String, BotPlayer> policy,
                     boolean treesRoot) {
        this.inner = inner;
        this.net = net;
        this.copier = copier;
        this.policy = policy;
        this.treesRoot = treesRoot;
    }

    /** The trees' scores of a card action (each option's probability, then pass's) as log-probabilities. */
    private static double[] treesLogProbs(double[] scores) {
        double[] t = new double[scores.length];
        for (int i = 0; i < t.length; i++) {
            double q = Math.min(1 - 1e-9, Math.max(1e-9, scores[i]));
            t[i] = scores[i] == Double.NEGATIVE_INFINITY ? Double.NEGATIVE_INFINITY : Math.log(q / (1 - q));
        }
        return BoardNet2.logSoftmax(t);
    }

    @Override
    public String getName() {
        return inner.getName();
    }

    @Override
    public String chooseAction(GameState gs, AwaitingDecision d) {
        var treeScores = com.gempukku.lotro.bots.rl.fotrstarters.models.cardaction.AnyPhaseCardActionTrainer.LAST_SCORES;
        if (treesRoot)
            treeScores.remove(); // only this decision's scores are read below
        String own = inner.chooseAction(gs, d);
        try {
            String kind = BoardDump.kind(d);
            if (own == null || net == null || !("action".equals(kind) || "move".equals(kind)))
                return own;
            String context;
            double[] lp;
            if (treesRoot) {
                // card actions only: "move again?" has no scores of the trees' here
                context = "action".equals(kind) ? BoardNet2.actionContext(d.getText()) : kind;
                double[] scores = treeScores.get();
                if (!"action".equals(kind) || scores == null || !CONTEXTS.contains(context))
                    return own;
                lp = treesLogProbs(scores);
            } else {
                BoardNet2.Result r = net.read(gs, d, getName());
                if (r == null || !CONTEXTS.contains(r.context()))
                    return own;
                context = r.context();
                lp = BoardNet2.logSoftmax(r.options());
            }
            int[] used = perGame.computeIfAbsent(gs.getGame(), k -> new int[1]);
            if (used[0] >= MAX_PER_GAME)
                return own;
            if (Math.exp(Arrays.stream(lp).max().orElse(0)) >= GATE)
                return own; // the network is sure: no search
            List<String> answers = answers(d, kind, lp.length);
            if (answers == null)
                return own;
            // the candidates: the network's top K, and the bot's own answer
            Integer[] order = new Integer[lp.length];
            for (int i = 0; i < order.length; i++)
                order[i] = i;
            Arrays.sort(order, (a, b) -> Double.compare(lp[b], lp[a]));
            LinkedHashSet<Integer> cands = new LinkedHashSet<>();
            int ownIndex = answers.indexOf(own);
            if (ownIndex >= 0)
                cands.add(ownIndex); // the bot's own answer, and the network's top K
            for (int i = 0; i < Math.min(K, order.length); i++)
                cands.add(order[i]);
            if (cands.size() < 2)
                return own;
            used[0]++;
            searched.incrementAndGet();
            long t0 = System.nanoTime();
            boolean turnLong = TURN_LONG.contains(context);
            // from the bots' Random, which a seeded game seeds apart from the engine's: the same game searches the same
            // futures again, and the redeals owe nothing to the real deal's draws
            long[] worlds = new SplittableRandom(com.gempukku.lotro.bots.BotRandom.current().nextLong()).longs(ROLLOUTS)
                    .toArray();
            int best = -1;
            double bestScore = Double.NEGATIVE_INFINITY;
            Map<Integer, Double> values = new LinkedHashMap<>();
            // two rounds (search.stage1 > 0): the first play-outs of every candidate, and the rest only where they
            // spread the candidates by search.spread or more -- the searches that pay are the ones whose options
            // come out far apart, and nothing known before searching tells which (scripts/search_gate.py)
            Map<Integer, double[]> sums = new LinkedHashMap<>(); // candidate -> {value sum, play-outs}
            int first = STAGE1 > 0 ? Math.min(STAGE1, worlds.length) : worlds.length;
            for (int c : cands) {
                double[] qn = new double[2];
                for (int i = 0; i < first; i++) {
                    Double v = playOut(gs, d, answers.get(c), worlds[i], turnLong, STAGE1_JUDGE);
                    if (v != null) {
                        qn[0] += v;
                        qn[1]++;
                    }
                }
                sums.put(c, qn);
            }
            if (first < worlds.length) {
                double lo = Double.POSITIVE_INFINITY, hi = Double.NEGATIVE_INFINITY;
                for (double[] qn : sums.values())
                    if (qn[1] > 0) {
                        lo = Math.min(lo, qn[0] / qn[1]);
                        hi = Math.max(hi, qn[0] / qn[1]);
                    }
                if (!(hi - lo >= SPREAD)) {
                    stoppedEarly.incrementAndGet();
                    nanos.addAndGet(System.nanoTime() - t0);
                    return own; // the options look alike: the bot's own answer
                }
                // the best KEEP by the first round's score go on; a first round of judged positions is not reused
                List<Integer> ranked = new ArrayList<>(sums.keySet());
                java.util.function.ToDoubleFunction<Integer> sc = c -> sums.get(c)[1] == 0 ? Double.NEGATIVE_INFINITY
                        : lp[c] + sums.get(c)[0] / sums.get(c)[1] / LAMBDA;
                ranked.sort((x, y) -> Double.compare(sc.applyAsDouble(y), sc.applyAsDouble(x)));
                if (STAGE1_JUDGE)
                    sums.replaceAll((c, qn) -> new double[2]);
                if (ranked.size() > KEEP)
                    for (int c : ranked.subList(KEEP, ranked.size()))
                        sums.remove(c);
                int from = STAGE1_JUDGE ? 0 : first;
                for (var e : sums.entrySet())
                    for (int i = from; i < worlds.length; i++) {
                        Double v = playOut(gs, d, answers.get(e.getKey()), worlds[i], turnLong);
                        if (v != null) {
                            e.getValue()[0] += v;
                            e.getValue()[1]++;
                        }
                    }
            }
            for (var e : sums.entrySet()) {
                int c = e.getKey();
                double q = e.getValue()[0], n = e.getValue()[1];
                if (n == 0)
                    continue;
                values.put(c, q / n);
                double score = lp[c] + (q / n) / LAMBDA;
                if (score > bestScore) {
                    bestScore = score;
                    best = c;
                }
            }
            nanos.addAndGet(System.nanoTime() - t0);
            if (best < 0)
                return own;
            String pick = answers.get(best);
            if (!pick.equals(own))
                changed.incrementAndGet();
            if (BOARDS != null && values.size() > 1)
                record(gs, d, kind, pick, values, answers.size(), ownIndex, Math.exp(lp[order[0]]));
            return pick;
        } catch (RuntimeException e) {
            return own;
        }
    }

    private void record(GameState gs, AwaitingDecision d, String kind, String pick, Map<Integer, Double> values, int n,
                        int own, double top) {
        try {
            double[] st = BoardNet2.state(gs, d, getName());
            var rec = st == null ? null : BoardDump.record(gs.getGame(), d, getName(), pick, st);
            if (rec == null)
                return;
            com.alibaba.fastjson2.JSONArray searched = new com.alibaba.fastjson2.JSONArray();
            // a card action's last option is pass, which the records index -1
            values.forEach((c, v) -> searched.add(new double[]{"action".equals(kind) && c == n - 1 ? -1 : c, v}));
            rec.put("searched", searched);
            rec.put("rollouts", ROLLOUTS);
            // what the bot would have answered without the search, and how sure the network was: what a gate on
            // where search pays is learned from (search's gain = its pick's value - the bot's own answer's)
            rec.put("own", own < 0 ? null : "action".equals(kind) && own == n - 1 ? -1 : own);
            rec.put("netTop", top);
            String line = rec.toString();
            synchronized (BOARDS) {
                BOARDS.println(line);
            }
        } catch (RuntimeException ignored) {
            // a board the dump cannot describe is left out
        }
    }

    /** Each option's answer, in the network's order: a card action's actions then pass, or move again's two results. */
    private static List<String> answers(AwaitingDecision d, String kind, int n) {
        List<String> out = new ArrayList<>();
        if ("action".equals(kind)) {
            String[] ids = d.getDecisionParameters().getOrDefault("actionId", new String[0]);
            if (ids.length + 1 != n)
                return null;
            out.addAll(Arrays.asList(ids));
            out.add(""); // pass
        } else {
            for (int i = 0; i < n; i++)
                out.add(String.valueOf(i));
        }
        return out;
    }

    /**
     * One play-out of {@code answer}: on a copy, the opponent's hand dealt again from the cards this player has not seen
     * and both decks shuffled from {@code world}, the answer given, the bot playing both sides; at the end of the phase
     * (or the turn) the judge reads the first decision of this player's it can, from this player's side. A game that
     * ends scores 1 or 0.
     */
    private Double playOut(GameState live, AwaitingDecision d, String answer, long world, boolean turnLong) {
        return playOut(live, d, answer, world, turnLong, false);
    }

    /** As above; {@code immediate}: judge the first position of this player's after the answer, without playing on. */
    private Double playOut(GameState live, AwaitingDecision d, String answer, long world, boolean turnLong,
                           boolean immediate) {
        String me = getName();
        try {
            DefaultLotroGame copy = copier.copy((DefaultLotroGame) live.getGame());
            GameState gs = copy.getGameState();
            gs.setDeckShuffler(null);
            gs.setRandomSeed(world);
            if (gs.isInit())
                for (String p : gs.getPlayerNames()) {
                    if (p.equals(me))
                        continue;
                    List<PhysicalCard> hand = new ArrayList<>(gs.getHand(p));
                    gs.removeCardsFromZone(null, hand);
                    gs.shuffleCardsIntoDeck(hand, p);
                    for (int i = 0; i < hand.size(); i++)
                        gs.playerDrawsCard(p);
                }
            java.util.Random previous = com.gempukku.lotro.bots.BotRandom.use(new java.util.Random(world ^ 0x5DEECE66DL));
            try {
                for (String p : gs.getPlayerNames())
                    gs.shuffleDeck(p);
                var feedback = GameCopier.feedback(copy);
                AwaitingDecision pending = feedback.getAwaitingDecision(me);
                if (pending == null)
                    return null;
                feedback.participantDecided(me);
                BotDecisionFallback.decide(copy, me, pending, answer);
                copy.carryOutPendingActionsUntilDecisionNeeded();
                Map<String, BotPlayer> bots = new HashMap<>();
                for (String p : gs.getPlayerNames())
                    bots.put(p, policy.apply(p));
                var phase = gs.getCurrentPhase();
                int turn = gs.getTurnNumber();
                playOuts.incrementAndGet();
                for (int step = 0; step < 600; step++) {
                    steps.incrementAndGet();
                    if (copy.isFinished())
                        return me.equals(copy.getWinnerPlayerId()) ? 1.0 : 0.0;
                    String who = null;
                    AwaitingDecision next = null;
                    for (String p : gs.getPlayerNames())
                        if ((next = feedback.getAwaitingDecision(p)) != null) {
                            who = p;
                            break;
                        }
                    if (next == null)
                        return null;
                    boolean far = immediate || (turnLong ? gs.getTurnNumber() > turn
                            : gs.getCurrentPhase() != phase || gs.getTurnNumber() > turn);
                    if (far && who.equals(me) && BoardDump.kind(next) != null) {
                        double[] st = BoardNet2.state(gs, next, me);
                        double[] o = st == null ? null : net.outcome(copy, next, me, st);
                        if (o != null) {
                            judged.incrementAndGet();
                            return o[0] + o[1] + o[2];
                        }
                    }
                    String a = bots.get(who).chooseAction(gs, next);
                    feedback.participantDecided(who);
                    BotDecisionFallback.decide(copy, who, next, a);
                    copy.carryOutPendingActionsUntilDecisionNeeded();
                }
                return null;
            } finally {
                com.gempukku.lotro.bots.BotRandom.use(previous);
            }
        } catch (Exception e) {
            return null;
        }
    }
}
