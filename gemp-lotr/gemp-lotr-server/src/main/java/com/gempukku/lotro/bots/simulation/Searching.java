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

    /** What the bot would have answered, where the last decision on this thread was searched (ReplayLog shows it). */
    static final ThreadLocal<String> LAST_OWN = new ThreadLocal<>();

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

    /**
     * Who fights whom, searched (2026-11-13; "assign" in search.contexts). The search had only ever branched card
     * plays; the assignment decides most games, and the old teacher gained most when it searched them (BOT-NOTES,
     * 2026-10-18/20). The candidates are the bot's own assignment and the assignment model's next search.assignK
     * (FotrStarterBot.assignmentCandidates, legal under the rules), each played out through the skirmishes it sets up
     * and judged after them. The model gives the candidates no probabilities to anchor on, so the bot's own stands
     * unless another beats it by search.assignMargin (a guess, not measured: 0.03 is what a card action 0.6 of a nat
     * less likely must make up at lambda 0.05) -- one play-out answering alone overturned a third of all card actions
     * and lost. Two rounds as for card actions: the first play-outs, the rest only where they spread the candidates.
     */
    static final int ASSIGN_K = Integer.getInteger("search.assignK", 3);
    static final double ASSIGN_MARGIN = Double.parseDouble(System.getProperty("search.assignMargin", "0.03"));
    static final java.util.concurrent.atomic.AtomicLong assignSearched = new java.util.concurrent.atomic.AtomicLong(),
            assignChanged = new java.util.concurrent.atomic.AtomicLong(),
            assignStopped = new java.util.concurrent.atomic.AtomicLong();
    private static final java.util.concurrent.atomic.AtomicLong[] ASSIGN_COUNTS = {assignSearched, assignChanged,
            assignStopped};

    private String searchAssignment(GameState gs, AwaitingDecision d, String own) {
        if (!(inner instanceof com.gempukku.lotro.bots.rl.fotrstarters.FotrStarterBot bot))
            return own;
        int[] used = perGame.computeIfAbsent(gs.getGame(), k -> new int[1]);
        if (used[0] >= MAX_PER_GAME)
            return own;
        LinkedHashSet<String> cands = new LinkedHashSet<>();
        cands.add(own);
        cands.addAll(bot.assignmentCandidates(gs, d, ASSIGN_K + 1));
        while (cands.size() > ASSIGN_K + 1) {
            String last = null;
            for (String c : cands)
                last = c;
            cands.remove(last);
        }
        if (cands.size() < 2)
            return own;
        if (decided(gs, d))
            return own;
        used[0]++;
        LAST_OWN.set(own);
        Played r = byPlayOuts(gs, d, own, cands, true, ASSIGN_MARGIN, ASSIGN_COUNTS);
        if (STEPS != null && r.bothRounds())
            assignmentSteps(gs, d, r.best(), r.sums(), own);
        return r.best();
    }

    /** -Dsearch.assignDepth=turn: an assignment's play-outs go on to this player's first decision of the next turn
     *  (through regroup and any further moves), not only through the skirmishes. */
    static final boolean ASSIGN_TURN = "turn".equals(System.getProperty("search.assignDepth"));

    /**
     * -Dsearch.decided=x (2026-11-17): no search where the judge already gives this player less than x or more than
     * 1 - x of winning -- a game decided either way has nothing for a search to change. The judge reads the position
     * at the decision; where it cannot (a pick, a choice), its last reading of this game stands in.
     */
    static final double DECIDED = Double.parseDouble(System.getProperty("search.decided", "0"));
    static final java.util.concurrent.atomic.AtomicLong decidedSkips = new java.util.concurrent.atomic.AtomicLong();
    private final Map<Object, double[]> lastJudged = Collections.synchronizedMap(new java.util.WeakHashMap<>());

    private boolean decided(GameState gs, AwaitingDecision d) {
        if (DECIDED <= 0 || net == null)
            return false;
        double[] last = null;
        try {
            if (BoardDump.kind(d) != null) {
                double[] st = BoardNet2.state(gs, d, getName());
                double[] o = st == null ? null : net.outcome(gs.getGame(), d, getName(), st);
                if (o != null)
                    lastJudged.put(gs.getGame(), last = new double[]{o[0] + o[1] + o[2]});
            }
        } catch (RuntimeException e) {
            // unread: the last reading stands in
        }
        if (last == null)
            last = lastJudged.get(gs.getGame());
        if (last == null || last[0] >= DECIDED && last[0] <= 1 - DECIDED)
            return false;
        decidedSkips.incrementAndGet();
        return true;
    }

    /** What a search of whole answers came to: the answer, each candidate's {value sum, play-outs}, and whether the
     *  second round was played (the first did not find them alike). */
    private record Played(String best, Map<String, double[]> sums, boolean bothRounds) {
    }

    /**
     * Candidates that are whole answers with no probabilities to anchor on (assignments, picks of one card, choices):
     * each played out in the same worlds -- through the skirmishes if {@code fights}, else to the end of the phase --
     * in two rounds as card actions are, and the bot's own answer kept unless another's mean value beats it by
     * {@code margin}. {@code counts}: {searched, answered differently, stopped after the first round}.
     */
    private Played byPlayOuts(GameState gs, AwaitingDecision d, String own, Collection<String> cands, boolean fights,
                              double margin, java.util.concurrent.atomic.AtomicLong[] counts) {
        return byPlayOuts(gs, d, own, cands, fights, fights && ASSIGN_TURN, margin, counts);
    }

    /** As above; {@code turnLong}: play-outs to this player's first decision of the next turn. */
    private Played byPlayOuts(GameState gs, AwaitingDecision d, String own, Collection<String> cands, boolean fights,
                              boolean turnLong, double margin, java.util.concurrent.atomic.AtomicLong[] counts) {
        counts[0].incrementAndGet();
        long t0 = System.nanoTime();
        long[] worlds = new SplittableRandom(com.gempukku.lotro.bots.BotRandom.current().nextLong()).longs(ROLLOUTS)
                .toArray();
        int first = STAGE1 > 0 ? Math.min(STAGE1, worlds.length) : worlds.length;
        Map<String, double[]> sums = new LinkedHashMap<>(); // candidate -> {value sum, play-outs}
        for (String c : cands) {
            double[] qn = new double[2];
            for (int i = 0; i < first; i++) {
                Double v = playOut(gs, d, c, worlds[i], turnLong, false, fights);
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
                counts[2].incrementAndGet();
                nanos.addAndGet(System.nanoTime() - t0);
                return new Played(own, sums, false); // the candidates look alike: the bot's own
            }
            for (var e : sums.entrySet())
                for (int i = first; i < worlds.length; i++) {
                    Double v = playOut(gs, d, e.getKey(), worlds[i], turnLong, false, fights);
                    if (v != null) {
                        e.getValue()[0] += v;
                        e.getValue()[1]++;
                    }
                }
        }
        String best = own;
        double bestScore = Double.NEGATIVE_INFINITY;
        for (var e : sums.entrySet()) {
            double q = e.getValue()[0], n = e.getValue()[1];
            if (n == 0)
                continue;
            double score = q / n + (e.getKey().equals(own) ? margin : 0);
            if (score > bestScore) {
                bestScore = score;
                best = e.getKey();
            }
        }
        nanos.addAndGet(System.nanoTime() - t0);
        if (!best.equals(own))
            counts[1].incrementAndGet();
        return new Played(best, sums, first < worlds.length);
    }

    /**
     * The picks and choices nothing searched (2026-11-14): which card a cost or an effect lands on, who exerts, which
     * skirmish next, a choice between a card's options. "select" (a pick of exactly one card in play), "arbitrary"
     * (one card from a list shown) and "choice" (a multiple choice of up to four answers, "move again?" apart) in
     * search.contexts: the bot's own answer against up to search.selectK others drawn at random -- there is no one
     * model that ranks them all -- played out to the end of the phase. The bot's own stands unless beaten by
     * search.selectMargin (0.03, the assignments' guess carried over, not measured).
     */
    static final int SELECT_K = Integer.getInteger("search.selectK", 3);
    static final double SELECT_MARGIN = Double.parseDouble(System.getProperty("search.selectMargin", "0.03"));
    static final Map<String, java.util.concurrent.atomic.AtomicLong[]> SELECT_COUNTS = new java.util.TreeMap<>();

    static {
        for (String k : List.of("select", "arbitrary", "choice", "movepick", "reconcile", "optional", "multi", "number", "respond"))
            SELECT_COUNTS.put(k, counts());
    }

    private static java.util.concurrent.atomic.AtomicLong[] counts() {
        return new java.util.concurrent.atomic.AtomicLong[]{new java.util.concurrent.atomic.AtomicLong(),
                new java.util.concurrent.atomic.AtomicLong(), new java.util.concurrent.atomic.AtomicLong()};
    }

    /** The context a pick or choice is searched under, or null if it is not one of them. */
    private static String selectContext(AwaitingDecision d) {
        var p = d.getDecisionParameters();
        switch (d.getDecisionType()) {
            case CARD_SELECTION, ARBITRARY_CARDS -> {
                String[] min = p.get("min"), max = p.get("max");
                if (min == null || max == null || !"1".equals(min[0]) || !"1".equals(max[0]))
                    return null;
                return d.getDecisionType() == com.gempukku.lotro.logic.decisions.AwaitingDecisionType.CARD_SELECTION
                        ? "select" : "arbitrary";
            }
            case MULTIPLE_CHOICE -> {
                String[] results = p.get("results");
                return results != null && results.length >= 2 && results.length <= 4 ? "choice" : null;
            }
            default -> {
                return null;
            }
        }
    }

    /** -Dsearch.dropNoEffect: see chooseAction. Counted whether or not the search then runs. */
    static final boolean DROP_NO_EFFECT = Boolean.getBoolean("search.dropNoEffect");
    static final java.util.concurrent.atomic.AtomicLong noEffectDropped = new java.util.concurrent.atomic.AtomicLong();

    /**
     * -Dsearch.selectFights: a pick's play-outs run on through the assignment and skirmish phases, as an assignment's
     * do, not to the end of the phase it was made in. For picks whose worth shows only in the fights: who takes an
     * archery wound, judged at the end of the archery phase, is judged before anyone has fought.
     */
    static final boolean SELECT_FIGHTS = Boolean.getBoolean("search.selectFights");

    /**
     * -Dsearch.actionFights=&lt;contexts&gt;: card actions in these contexts (shadow, maneuver, archery, assignment,
     * skirmish) are played out to the regroup phase, as picks are under search.selectFights, not to the end of the
     * phase they are played in.
     */
    static final Set<String> ACTION_FIGHTS = new HashSet<>(List.of(System.getProperty("search.actionFights", "").split(",")));

    /**
     * -Dsearch.selfCheck: a pick searched through both rounds is searched a second time in other worlds, and the two
     * answers compared, by prompt -- how much of a label from this search is the position and how much the draw
     * (2026-11-20). The second search's answer is not used; the game's later draws differ for it.
     */
    static final boolean SELF_CHECK = Boolean.getBoolean("search.selfCheck");
    static final Map<String, long[]> SELF = new java.util.concurrent.ConcurrentSkipListMap<>();

    static {
        if (SELF_CHECK)
            Runtime.getRuntime().addShutdownHook(new Thread(() -> {
                System.err.println("self-check: searched twice | same pick | first changed the bot's answer | second "
                        + "named the same | second went back to the bot's | second stopped after its first round");
                SELF.forEach((k, c) -> System.err.printf("%7d %7d %7d %7d %7d %7d  %s%n", c[0], c[1], c[2], c[3], c[4],
                        c[5], k));
            }));
    }

    private String searchSelection(GameState gs, AwaitingDecision d, String own, String context) {
        var p = d.getDecisionParameters();
        List<String> options = new ArrayList<>();
        if (context.equals("choice")) {
            for (int i = 0; i < p.get("results").length; i++)
                options.add(String.valueOf(i));
        } else {
            String[] ids = p.getOrDefault("cardId", new String[0]), ok = p.get("selectable");
            for (int i = 0; i < ids.length; i++)
                if (ok == null || i >= ok.length || "true".equals(ok[i]))
                    options.add(ids[i]);
        }
        if (!options.remove(own) || options.isEmpty())
            return own; // an answer of another shape than one of the options: not ours to search
        int[] used = perGame.computeIfAbsent(gs.getGame(), k -> new int[1]);
        if (used[0] >= MAX_PER_GAME)
            return own;
        Collections.shuffle(options, new java.util.Random(com.gempukku.lotro.bots.BotRandom.current().nextLong()));
        List<String> cands = new ArrayList<>();
        cands.add(own);
        cands.addAll(options.subList(0, Math.min(SELECT_K, options.size())));
        String prompt = prompt(context, d);
        if (SELECT_ONLY != null && !SELECT_ONLY.matcher(prompt).find()
                || SELECT_SKIP != null && SELECT_SKIP.matcher(prompt).find())
            return own;
        if (decided(gs, d))
            return own;
        used[0]++;
        LAST_OWN.set(own);
        var by = SELECT_PROMPTS.computeIfAbsent(prompt, k -> counts());
        var ph = gs.getCurrentPhase();
        boolean fights = SELECT_FIGHTS && ph != null && ph.ordinal() >= com.gempukku.lotro.common.Phase.SHADOW.ordinal()
                && ph.ordinal() < com.gempukku.lotro.common.Phase.REGROUP.ordinal();
        Played r = byPlayOuts(gs, d, own, cands, fights, false, SELECT_MARGIN, SELECT_COUNTS.get(context));
        by[0].incrementAndGet();
        if (!r.best().equals(own))
            by[1].incrementAndGet();
        if (!r.bothRounds())
            by[2].incrementAndGet();
        if (STEPS != null && r.bothRounds())
            selectionSteps(gs, d, r.best(), r.sums(), own);
        if (SELF_CHECK && r.bothRounds()) {
            // the same candidates searched again in other worlds: does the search name the same pick twice?
            Played again = byPlayOuts(gs, d, own, cands, fights, false, SELECT_MARGIN, counts());
            long[] c = SELF.computeIfAbsent(prompt, k -> new long[6]);
            synchronized (c) {
                c[0]++;
                if (again.best().equals(r.best()))
                    c[1]++;
                if (!r.best().equals(own)) {
                    c[2]++;
                    if (again.best().equals(r.best()))
                        c[3]++;
                    else if (again.best().equals(own))
                        c[4]++;
                    if (!again.bothRounds())
                        c[5]++;
                }
            }
        }
        return r.best();
    }

    /**
     * -Dsearch.steps: a pick or choice searched through both rounds, one step an option played out, shaped as
     * TrainingSteps writes them (a card pick in a skirmish is the "assigned" kind; reconcile, exert and character picks
     * carry their options). People's picks are written as the chosen card alone; here the options not chosen are
     * written too, reward 0, with their play-out values, so whoever fits from them decides what to keep.
     */
    private void selectionSteps(GameState gs, AwaitingDecision d, String pick, Map<String, double[]> sums, String own) {
        long id = STEP_IDS.incrementAndGet();
        try {
            double[] state = new com.gempukku.lotro.bots.rl.fotrstarters.GeneralStateFeatures()
                    .extractFeatures(gs, d, getName());
            boolean fp = getName().equals(gs.getCurrentPlayerId());
            StringBuilder lines = new StringBuilder();
            for (String answer : sums.keySet())
                try {
                    double[] qn = sums.get(answer);
                    if (qn[1] <= 0)
                        continue;
                    com.gempukku.lotro.bots.rl.semanticaction.SemanticAction action = switch (d.getDecisionType()) {
                        case MULTIPLE_CHOICE -> new com.gempukku.lotro.bots.rl.semanticaction.MultipleChoiceAction(answer, d);
                        case ARBITRARY_CARDS ->
                                new com.gempukku.lotro.bots.rl.semanticaction.ChooseFromArbitraryCardsAction(answer, d);
                        default -> TrainingSteps.withOptions(
                                gs.getCurrentPhase() == com.gempukku.lotro.common.Phase.SKIRMISH
                                        ? new com.gempukku.lotro.bots.rl.semanticaction.CardSelectionAssignedAction(answer, d, gs)
                                        : new com.gempukku.lotro.bots.rl.semanticaction.CardSelectionAction(answer, d, gs),
                                gs, d, answer, getName());
                    };
                    var step = new com.gempukku.lotro.bots.rl.LearningStep(state, action, fp, d);
                    step.reward = answer.equals(pick) ? 10.0 : 0.0;
                    lines.append(line(step, qn[0] / qn[1], answer.equals(own), id)).append('\n');
                } catch (RuntimeException ignored) {
                    // an answer the semantic action cannot describe
                }
            synchronized (STEPS) {
                STEPS.print(lines);
                STEPS.flush();
            }
        } catch (RuntimeException e) {
            // a label lost; the game goes on
        }
    }

    /**
     * Two more kinds (2026-11-17), as searchOther: "multi" (a pick of several cards, in play or from a list: the bot's
     * own cards against up to search.selectK other sets of as many, drawn at random) and "number" (a number from a
     * range of at most six: the bot's own against up to search.selectK others), played out to the end of the phase.
     */
    private String searchMore(GameState gs, AwaitingDecision d, String own, String context) {
        var p = d.getDecisionParameters();
        var rnd = new java.util.Random(com.gempukku.lotro.bots.BotRandom.current().nextLong());
        LinkedHashSet<String> cands = new LinkedHashSet<>();
        cands.add(own);
        if (context.equals("number")) {
            String[] min = p.get("min"), max = p.get("max");
            if (min == null || max == null)
                return own;
            int lo = Integer.parseInt(min[0]), hi = Integer.parseInt(max[0]);
            if (hi - lo < 1 || hi - lo > 5)
                return own;
            List<String> others = new ArrayList<>();
            for (int i = lo; i <= hi; i++)
                if (!String.valueOf(i).equals(own))
                    others.add(String.valueOf(i));
            if (others.size() != hi - lo)
                return own; // the bot's answer is not a number of the range
            Collections.shuffle(others, rnd);
            cands.addAll(others.subList(0, Math.min(SELECT_K, others.size())));
        } else {
            String[] ids = p.getOrDefault("cardId", new String[0]), ok = p.get("selectable");
            List<String> pool = new ArrayList<>();
            for (int i = 0; i < ids.length; i++)
                if (ok == null || i >= ok.length || "true".equals(ok[i]))
                    pool.add(ids[i]);
            List<String> mine = own.isEmpty() ? List.of() : List.of(own.split(","));
            if (mine.isEmpty() || mine.size() >= pool.size() || !pool.containsAll(mine))
                return own; // nothing picked, everything picked, or an answer of another shape
            String key = canonical(mine);
            for (int tries = 0; tries < 4 * SELECT_K && cands.size() <= SELECT_K; tries++) {
                List<String> draw = new ArrayList<>(pool);
                Collections.shuffle(draw, rnd);
                List<String> set = draw.subList(0, mine.size());
                if (!canonical(set).equals(key) && cands.stream().noneMatch(c -> canonical(List.of(c.split(","))).equals(canonical(set))))
                    cands.add(String.join(",", set));
            }
        }
        if (cands.size() < 2)
            return own;
        int[] used = perGame.computeIfAbsent(gs.getGame(), k -> new int[1]);
        if (used[0] >= MAX_PER_GAME)
            return own;
        if (decided(gs, d))
            return own;
        used[0]++;
        LAST_OWN.set(own);
        var by = SELECT_PROMPTS.computeIfAbsent(prompt(context, d), k -> counts());
        Played r = byPlayOuts(gs, d, own, cands, false, false, SELECT_MARGIN, SELECT_COUNTS.get(context));
        by[0].incrementAndGet();
        if (!r.best().equals(own))
            by[1].incrementAndGet();
        if (!r.bothRounds())
            by[2].incrementAndGet();
        return r.best();
    }

    private static String canonical(List<String> ids) {
        List<String> sorted = new ArrayList<>(ids);
        Collections.sort(sorted);
        return String.join(",", sorted);
    }

    /** A pick of at most one card in play, none allowed (min 0, max 1). */
    private static boolean optional(AwaitingDecision d) {
        if (d.getDecisionType() != com.gempukku.lotro.logic.decisions.AwaitingDecisionType.CARD_SELECTION)
            return false;
        String[] min = d.getDecisionParameters().get("min"), max = d.getDecisionParameters().get("max");
        return min != null && max != null && "0".equals(min[0]) && "1".equals(max[0]);
    }

    /**
     * Three kinds of decision nothing searched before (2026-11-16), each the bot's own answer against the others,
     * kept unless beaten by search.selectMargin: "movepick" ("move again?": yes against no, play-outs to this
     * player's first decision of the next turn -- the card-action search's "move" context anchored on the network and
     * gave nothing); "reconcile" (the reconcile discard: keep everything or one card, up to search.selectK others at
     * random, to the next turn as well -- the phase ends with the answer); "optional" (any other pick of at most one
     * card, none allowed, to the end of the phase).
     */
    private String searchOther(GameState gs, AwaitingDecision d, String own, String context) {
        var p = d.getDecisionParameters();
        List<String> options = new ArrayList<>();
        if (context.equals("movepick")) {
            String[] results = p.get("results");
            if (results == null || results.length != 2)
                return own;
            options.add("0");
            options.add("1");
        } else if (context.equals("respond")) {
            // an optional response (2026-11-17): each action offered, and passing; the card-action search does not
            // reach these (their context is "other", and the trees leave no scores for them)
            for (String id : p.getOrDefault("actionId", new String[0]))
                options.add(id);
            options.add("");
        } else {
            String[] ids = p.getOrDefault("cardId", new String[0]), min = p.get("min"), max = p.get("max");
            if (max == null || !"1".equals(max[0]) && !(context.equals("reconcile") && own.indexOf(',') < 0))
                return own;
            for (String id : ids)
                options.add(id);
            if (min != null && "0".equals(min[0]))
                options.add("");
        }
        if (!options.remove(own) || options.isEmpty())
            return own;
        int[] used = perGame.computeIfAbsent(gs.getGame(), k -> new int[1]);
        if (used[0] >= MAX_PER_GAME)
            return own;
        Collections.shuffle(options, new java.util.Random(com.gempukku.lotro.bots.BotRandom.current().nextLong()));
        List<String> cands = new ArrayList<>();
        cands.add(own);
        cands.addAll(options.subList(0, Math.min(SELECT_K, options.size())));
        if (decided(gs, d))
            return own;
        used[0]++;
        LAST_OWN.set(own);
        String prompt = prompt(context, d);
        var by = SELECT_PROMPTS.computeIfAbsent(prompt, k -> counts());
        Played r = byPlayOuts(gs, d, own, cands, false, context.equals("movepick") || context.equals("reconcile"), SELECT_MARGIN,
                SELECT_COUNTS.get(context));
        if (STEPS != null && r.bothRounds() && !context.equals("respond"))
            selectionSteps(gs, d, r.best(), r.sums(), own); // "move again?" and reconcile as labels too
        by[0].incrementAndGet();
        if (!r.best().equals(own))
            by[1].incrementAndGet();
        if (!r.bothRounds())
            by[2].incrementAndGet();
        return r.best();
    }

    /**
     * Which picks and choices the search overturns (2026-11-14): 86% of picks stop after the first round and 3% change
     * the answer, so most of their cost confirms the bot. Counts by prompt -- the context and the decision's text with
     * tags and numbers taken out, 50 letters of it -- are printed after a run ("picks by prompt"), and
     * -Dsearch.selectOnly=&lt;regex&gt; searches only the prompts it finds.
     */
    static final Map<String, java.util.concurrent.atomic.AtomicLong[]> SELECT_PROMPTS =
            new java.util.concurrent.ConcurrentHashMap<>();
    static final java.util.regex.Pattern SELECT_ONLY = System.getProperty("search.selectOnly") == null ? null
            : java.util.regex.Pattern.compile(System.getProperty("search.selectOnly"));

    /** -Dsearch.selectSkip=&lt;regex&gt;: the prompts not to search (write it without spaces: "." for one). */
    static final java.util.regex.Pattern SELECT_SKIP = System.getProperty("search.selectSkip") == null ? null
            : java.util.regex.Pattern.compile(System.getProperty("search.selectSkip"));

    private static String prompt(String context, AwaitingDecision d) {
        String t = d.getText() == null ? "" : d.getText().replaceAll("<[^>]*>", "").replaceAll("[0-9]+", "#")
                .replaceAll("\\s+", " ").trim();
        return context + ": " + (t.length() > 50 ? t.substring(0, 50) : t);
    }

    /** -Dsearch.steps: an assignment searched through both rounds, as BotTeach.teachAssignment writes its labels. */
    private void assignmentSteps(GameState gs, AwaitingDecision d, String pick, Map<String, double[]> sums, String own) {
        Collection<String> played = sums.keySet();
        long id = STEP_IDS.incrementAndGet();
        try {
            double[] state = new com.gempukku.lotro.bots.rl.fotrstarters.GeneralStateFeatures()
                    .extractFeatures(gs, d, getName());
            boolean fp = getName().equals(gs.getCurrentPlayerId());
            StringBuilder lines = new StringBuilder();
            for (String answer : played)
                try {
                    var probe = com.gempukku.lotro.bots.rl.fotrstarters.models.assignment.AssignmentProbe.features(gs, d,
                            com.gempukku.lotro.bots.rl.fotrstarters.models.assignment.AssignmentProbe.parse(answer));
                    var step = new com.gempukku.lotro.bots.rl.LearningStep(state,
                            new com.gempukku.lotro.bots.rl.semanticaction.AssignMinionsAction(answer, d, gs, fp)
                                    .withProbe(probe), fp, d);
                    step.reward = answer.equals(pick) ? 10.0 : 0.0;
                    double[] qn = sums.get(answer);
                    if (qn[1] > 0)
                        lines.append(line(step, qn[0] / qn[1], answer.equals(own), id)).append('\n');
                } catch (RuntimeException ignored) {
                    // an assignment the semantic action cannot describe
                }
            synchronized (STEPS) {
                STEPS.print(lines);
                STEPS.flush();
            }
        } catch (RuntimeException e) {
            // a label lost; the game goes on
        }
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
            if (own != null && net != null && "assign".equals(kind) && CONTEXTS.contains("assign"))
                return searchAssignment(gs, d, own);
            if (own != null && net != null && kind == null) {
                String sel = selectContext(d);
                if (sel != null && CONTEXTS.contains(sel))
                    return searchSelection(gs, d, own, sel);
            }
            if (own != null && net != null && "action".equals(kind) && CONTEXTS.contains("respond")
                    && d.getText() != null && d.getText().contains("Optional responses"))
                return searchOther(gs, d, own, "respond");
            if (own != null && net != null && "move".equals(kind) && CONTEXTS.contains("movepick"))
                return searchOther(gs, d, own, "movepick");
            if (own != null && net != null && "reconcile".equals(kind) && CONTEXTS.contains("reconcile"))
                return searchOther(gs, d, own, "reconcile");
            if (own != null && net != null && kind == null && CONTEXTS.contains("optional") && optional(d))
                return searchOther(gs, d, own, "optional");
            if (own != null && net != null && kind == null && selectContext(d) == null && !optional(d)) {
                var type = d.getDecisionType();
                boolean cards = type == com.gempukku.lotro.logic.decisions.AwaitingDecisionType.CARD_SELECTION
                        || type == com.gempukku.lotro.logic.decisions.AwaitingDecisionType.ARBITRARY_CARDS;
                if (cards && CONTEXTS.contains("multi"))
                    return searchMore(gs, d, own, "multi");
                if (type == com.gempukku.lotro.logic.decisions.AwaitingDecisionType.INTEGER && CONTEXTS.contains("number"))
                    return searchMore(gs, d, own, "number");
            }
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
            // search.dropNoEffect: an action upstream flags as doing nothing is not played out (the bot's own answer
            // still is); the next of the trees' answers takes its place
            String[] idle = DROP_NO_EFFECT && "action".equals(kind) ? d.getDecisionParameters().get("noEffect") : null;
            for (int i = 0, taken = 0; i < order.length && taken < K; i++) {
                int c = order[i];
                if (idle != null && c < idle.length && !"false".equals(idle[c])) {
                    noEffectDropped.incrementAndGet();
                    continue;
                }
                cands.add(c);
                taken++;
            }
            if (cands.size() < 2)
                return own;
            if (decided(gs, d))
                return own;
            used[0]++;
            LAST_OWN.set(own);
            searched.incrementAndGet();
            long t0 = System.nanoTime();
            boolean turnLong = TURN_LONG.contains(context);
            var phaseNow = gs.getCurrentPhase();
            boolean fights = ACTION_FIGHTS.contains(context) && phaseNow != null
                    && phaseNow.ordinal() >= com.gempukku.lotro.common.Phase.SHADOW.ordinal()
                    && phaseNow.ordinal() < com.gempukku.lotro.common.Phase.REGROUP.ordinal();
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
                    Double v = playOut(gs, d, answers.get(c), worlds[i], turnLong, STAGE1_JUDGE, fights);
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
                        Double v = playOut(gs, d, answers.get(e.getKey()), worlds[i], turnLong, false, fights);
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
            if (STEPS != null && values.size() > 1 && "action".equals(kind))
                steps(gs, d, pick, values, answers, own);
            return pick;
        } catch (RuntimeException e) {
            return own;
        }
    }

    /**
     * -Dsearch.steps=&lt;file&gt;: the search as a teacher for the trees it sits on (2026-11-11). A card action whose
     * search ran both rounds is written as BotTeach writes its labels -- one step an option played out, the search's
     * answer good (reward 10), the others bad (0) -- to refit the card-action slots from (BotTeach --train-only
     * --steps). The answer, not the best value: it is what the strong bot plays, the trees' own preference in it.
     * Searches that stopped after the first round say only "as the trees do" and are not written.
     */
    static final java.io.PrintWriter STEPS = Recording.open(System.getProperty("search.steps"));

    /** One number a searched decision, so a reader can put its options' steps back together. */
    static final java.util.concurrent.atomic.AtomicLong STEP_IDS = new java.util.concurrent.atomic.AtomicLong();

    /** A step's line with what the search knew: the option's mean play-out value, whether it was the bot's own
     *  answer, and the decision's number -- so labels can be cut again without playing (the first labels marked every
     *  option that was not the answer bad, near-ties and all; sp-m6a, -1.0). */
    private static String line(com.gempukku.lotro.bots.rl.LearningStep step, double value, boolean own, long id) {
        var o = step.toJson();
        o.put("searchValue", value);
        o.put("searchOwn", own);
        o.put("searchId", id);
        return o.toString();
    }

    private void steps(GameState gs, AwaitingDecision d, String pick, Map<Integer, Double> values, List<String> answers,
                       String own) {
        Collection<Integer> played = values.keySet();
        long id = STEP_IDS.incrementAndGet();
        try {
            double[] state = new com.gempukku.lotro.bots.rl.fotrstarters.GeneralStateFeatures()
                    .extractFeatures(gs, d, getName());
            boolean fp = getName().equals(gs.getCurrentPlayerId());
            StringBuilder lines = new StringBuilder();
            for (int c : played) {
                String answer = answers.get(c);
                try {
                    double[] probe = null;
                    if (inner instanceof com.gempukku.lotro.bots.rl.fotrstarters.FotrStarterBot bot) {
                        probe = answer.isEmpty() ? com.gempukku.lotro.bots.rl.fotrstarters.EffectProbe.pass(gs, getName())
                                : com.gempukku.lotro.bots.rl.fotrstarters.EffectProbe.probe(gs, getName(), answer, bot::as);
                        if (probe == null)
                            probe = com.gempukku.lotro.bots.rl.fotrstarters.EffectProbe.none();
                    }
                    var step = new com.gempukku.lotro.bots.rl.LearningStep(state,
                            new com.gempukku.lotro.bots.rl.semanticaction.CardActionChoiceAction(answer, d, gs)
                                    .withProbe(probe), fp, d);
                    step.reward = answer.equals(pick) ? 10.0 : 0.0; // trainers read reward > 0 as good
                    lines.append(line(step, values.get(c), answer.equals(own), id)).append('\n');
                } catch (RuntimeException ignored) {
                    // an answer the semantic action cannot describe
                }
            }
            synchronized (STEPS) {
                STEPS.print(lines);
                STEPS.flush();
            }
        } catch (RuntimeException e) {
            // a label lost; the game goes on
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
            // which game: boards of one game are alike, and are held out together when a model is checked on them
            rec.put("game", "g" + Integer.toHexString(System.identityHashCode(gs.getGame())) + "-" + ProcessHandle.current().pid());
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
    /**
     * -Dsearch.peek: a yardstick, not a bot -- the play-outs keep the opponent's true hand instead of dealing it again
     * (decks are still shuffled). How much stronger the search is for it bounds what hidden information is worth
     * here, and so what knowing the opponent's exact decklist (which the redeal draws from) can be worth.
     */
    static final boolean PEEK = Boolean.getBoolean("search.peek");

    private Double playOut(GameState live, AwaitingDecision d, String answer, long world, boolean turnLong) {
        return playOut(live, d, answer, world, turnLong, false);
    }

    /** As above; {@code immediate}: judge the first position of this player's after the answer, without playing on. */
    private Double playOut(GameState live, AwaitingDecision d, String answer, long world, boolean turnLong,
                           boolean immediate) {
        return playOut(live, d, answer, world, turnLong, immediate, false);
    }

    /** As above; {@code fights}: play on until the regroup phase (the fights are over) or the next turn. */
    private Double playOut(GameState live, AwaitingDecision d, String answer, long world, boolean turnLong,
                           boolean immediate, boolean fights) {
        String me = getName();
        try {
            DefaultLotroGame copy = copier.copy((DefaultLotroGame) live.getGame());
            GameState gs = copy.getGameState();
            gs.setDeckShuffler(null);
            gs.setRandomSeed(world);
            if (gs.isInit() && !PEEK)
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
            boolean wasIn = com.gempukku.lotro.bots.rl.fotrstarters.models.FlatTrees.IN_PLAYOUT.get();
            com.gempukku.lotro.bots.rl.fotrstarters.models.FlatTrees.IN_PLAYOUT.set(true);
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
                    var now = gs.getCurrentPhase();
                    boolean far = immediate || (turnLong ? gs.getTurnNumber() > turn
                            : fights ? gs.getTurnNumber() > turn || now.ordinal() >= com.gempukku.lotro.common.Phase.REGROUP.ordinal()
                            : now != phase || gs.getTurnNumber() > turn);
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
                com.gempukku.lotro.bots.rl.fotrstarters.models.FlatTrees.IN_PLAYOUT.set(wasIn);
                com.gempukku.lotro.bots.BotRandom.use(previous);
            }
        } catch (Exception e) {
            if (PLAYOUT_ERRORS.getAndIncrement() < 3)
                e.printStackTrace(System.out); // a play-out that cannot be played says so, the first few times
            return null;
        }
    }

    static final java.util.concurrent.atomic.AtomicInteger PLAYOUT_ERRORS = new java.util.concurrent.atomic.AtomicInteger();
}
