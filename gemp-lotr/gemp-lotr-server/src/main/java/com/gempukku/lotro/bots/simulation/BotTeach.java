package com.gempukku.lotro.bots.simulation;

import com.gempukku.lotro.bots.BotPlayer;
import com.gempukku.lotro.bots.BotService;
import com.gempukku.lotro.bots.rl.LearningStep;
import com.gempukku.lotro.bots.rl.RLGameStateFeatures;
import com.gempukku.lotro.bots.rl.fotrstarters.CardFeatures;
import com.gempukku.lotro.bots.rl.fotrstarters.FotrStartersRLGameStateFeatures;
import com.gempukku.lotro.bots.rl.fotrstarters.GeneralStateFeatures;
import com.gempukku.lotro.bots.rl.fotrstarters.models.ModelRegistry;
import com.gempukku.lotro.bots.rl.fotrstarters.models.Trainer;
import com.gempukku.lotro.bots.rl.semanticaction.CardActionChoiceAction;
import com.gempukku.lotro.game.LotroFormat;
import com.gempukku.lotro.game.state.GameState;
import com.gempukku.lotro.logic.decisions.AwaitingDecision;
import com.gempukku.lotro.logic.decisions.AwaitingDecisionType;
import com.gempukku.lotro.logic.vo.LotroDeck;
import smile.classification.SoftClassifier;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Lookahead as a teacher. Self-play outcome labels say a move was good if its game was won, which cannot tell a
 * good choice from a lucky game (see BOT-NOTES.md); this labels decisions by trying them.
 * <p>
 * In games between two bots, a share of the card-action decisions (play, use, or pass) are branched: for each option,
 * the game is replayed from its seed to that decision (BotReplay checks this is exact), the option is taken, and the
 * game is played out by the bot on both sides -- {@code --rollouts} times. At the branch the engine is reseeded and
 * both draw decks reshuffled, so a rollout does not know the real deck order; every option gets the same set of
 * reshuffles, so they are compared on the same futures. The option(s) that won most are labelled good, options that
 * won clearly less ({@code --margin}) bad, the rest left out. The main game goes on with the bot's own choice, so the
 * labels are for the states the bot actually reaches.
 * <p>
 * The labelled steps are appended to {@code <out>/teacher-steps.jsonl}; then every card-action model of the
 * {@code --base} set is refit on them alone and the set written to {@code --out}. {@code --train-only} skips the games
 * and refits from the file.
 * <pre>
 *   BOT_MAIN=com.gempukku.lotro.bots.simulation.BotTeach ./bot-vs-bot.sh --base bot-models-exp/v2 \
 *       --out bot-models-exp/taught --suite starters:train --games 300
 * </pre>
 */
public class BotTeach {
    record Options(String suite, Path base, Path out, int games, int rollouts, double teachRate, int maxCandidates,
                   double margin, int jobs, boolean trainOnly, boolean play, int minSteps, double lambda, Path steps,
                   int depth, boolean both, Path value, boolean peek, boolean fromCopies, boolean choices, boolean assign, Path recordValues) {
    }

    private static final AtomicInteger taught = new AtomicInteger(), labelled = new AtomicInteger(),
            rolloutsRun = new AtomicInteger(), diverged = new AtomicInteger(), cutOff = new AtomicInteger(),
            choicesSearched = new AtomicInteger(), assignmentsSearched = new AtomicInteger(),
            movesSearched = new AtomicInteger(), movesTaken = new AtomicInteger(), assignmentsTaught = new AtomicInteger(), rbChecked = new AtomicInteger();
    /** Scores a rollout's position when --depth stops it early: P(win) for the player whose state it is. */
    static volatile SoftClassifier<double[]> valueModel;
    /**
     * -Dteach.opponents=&lt;kind&gt;,...: a league. The learning bot plays seat ~a and is the one searched; ~b is each
     * opponent in turn (game by game), in the main game and in the play-outs -- not the learner against itself.
     */
    static final List<String> OPPONENTS = System.getProperty("teach.opponents") == null ? List.of()
            : List.of(System.getProperty("teach.opponents").split(","));
    /**
     * -Dsearch.judge=&lt;model dir&gt;: a cut-off rollout is scored by the combined network's outcome head (the
     * board-net2-judge.onnx there if exported, else board-net2.onnx's own) instead of the value model -- read at the
     * first decision after the cut that the network takes and that the searching player is asked.
     */
    static final com.gempukku.lotro.bots.rl.fotrstarters.BoardNet2 JUDGE = judge(System.getProperty("search.judge"));

    private static com.gempukku.lotro.bots.rl.fotrstarters.BoardNet2 judge(String dir) {
        if (dir == null)
            return null;
        try {
            return com.gempukku.lotro.bots.rl.fotrstarters.BoardNet2.load(java.nio.file.Path.of(dir));
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }
    /** 3 if the value model reads feature set 2 with KnowledgeFeatures appended (value-features.txt). */
    private static volatile int valueFeatures = 2;
    /** --corruption-weight: how much a cut-off position's value moves with corruption pressure; 0 = off. */
    static volatile double corruptionWeight = 0;
    /** --progress-weight: how much a cut-off position's value moves with how far each fellowship has got; 0 = off. */
    static volatile double progressWeight = 0;
    /** With --clone, rollouts play on a copy of the live game made by this, instead of replaying it from its seed. */
    private static volatile GameCopier copier;

    /** The main game's positions so far on this thread, for --record-values: {decider, set-2 vector}. */
    private static final ThreadLocal<List<Object[]>> POSITIONS = ThreadLocal.withInitial(ArrayList::new);
    /** The lowest resistance each Ring-bearer had in the main game on this thread (burdens already taken off). */
    private static final ThreadLocal<Map<String, Integer>> LOWEST_RESISTANCE = ThreadLocal.withInitial(HashMap::new);

    private static final long STALL_MINUTES = 15;
    /**
     * Decisions per bot per main game before it concedes, as on the live server ("Possible loop detected"). Real decks
     * play long games: at 1,000, about 5% of the user's decks' games against real Expanded decks ended there.
     */
    static volatile int maxDecisions = 1000;
    /** --card-picks: also search single-card picks from a list (see Teaching.chooseAction). */
    static volatile boolean cardPicks = false;
    /**
     * --move-rollouts / --move-depth: the "move again?" question searched with its own rollouts, played out to the end
     * (depth 0) by default. Cut-off values cannot judge a move: the value models learn the race backwards from bot
     * games, and a model fit on human positions, which has it the right way round, loses (see BOT-NOTES.md, "Should
     * the bot race?"). A playout asks the bot's own future play whether moving on wins.
     */
    static volatile int moveRollouts = 0, moveDepth = 0;
    /**
     * --no-card-search: with --play, card actions -- most of a game's decisions and nearly all of the search's cost --
     * are left to the models; assignments and choices (--assign, --choices) are still searched.
     */
    static volatile boolean cardSearch = true;
    /**
     * -Dsearch.phases=shadow,skirmish: search (or, teaching, label) only card actions whose prompt names one of these
     * (the prompt says "Play Shadow action or Pass" and so on); unset, every phase. Searching Fellowship-phase plays
     * loses (44.5% against the plain bot), so a teacher should leave them out.
     */
    static final List<String> SEARCH_PHASES = System.getProperty("search.phases") == null ? null
            : List.of(System.getProperty("search.phases").toLowerCase().split(","));

    /**
     * -Dsearch.phaseCut: rollouts stop when the phase they branched in ends (not --depth turns on) and are scored by the
     * value model there. A phase planner: the Shadow player's plays judged by the board they leave, sequences included
     * (the bot plays on to the end of the phase), at a fraction of a turn-long rollout.
     */
    static final boolean PHASE_CUT = Boolean.getBoolean("search.phaseCut");

    static boolean searchedPhase(AwaitingDecision decision) {
        if (SEARCH_PHASES == null)
            return true;
        String text = decision.getText() == null ? "" : decision.getText().toLowerCase();
        return SEARCH_PHASES.stream().anyMatch(text::contains);
    }
    /** --assign-teach-rate: share of assignment decisions branched in teacher mode (with --assign); default --teach-rate. */
    static volatile double assignTeachRate = -1;
    /**
     * --deck-order-teach-rate: share of "Choose card to put on top of deck" / "beneath draw deck" decisions branched
     * in teacher mode -- the bot's commonest random answer (3.8 a game between real decks), which people's replays do
     * not record. Each option's value goes to -Dteach.pickValues=&lt;file&gt; too: whether the order matters at all.
     */
    static volatile double deckOrderTeachRate = 0;
    /**
     * --trigger-order-teach-rate: share of ACTION_CHOICE decisions -- which of several simultaneous required triggers
     * resolves first, answered at random -- branched to measure it: values only (-Dteach.pickValues), no labels yet.
     */
    static volatile double triggerOrderTeachRate = 0;
    /**
     * --rb-assign-check: at a Free Peoples assignment where the bot's own answer puts a minion on its Ring-bearer while
     * another character is left idle, branch the answer against the Ring-bearer's minions moved onto each idle
     * character (where legal). Bot-against-bot Ring-bearer deaths in a skirmish follow the bot's own assignment in 88%
     * of cases, a third of them with a character idle. Values to -Dteach.pickValues, labels (with AssignmentProbe's
     * numbers) to the steps file.
     */
    static volatile boolean rbAssignCheck = false;
    static final java.util.regex.Pattern DECK_ORDER = java.util.regex.Pattern.compile("on top of deck|beneath draw deck");
    private static final PrintWriter PICK_VALUES;
    static {
        PrintWriter w = null;
        try {
            if (System.getProperty("teach.pickValues") != null)
                w = new PrintWriter(Files.newBufferedWriter(Path.of(System.getProperty("teach.pickValues"))), true);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        PICK_VALUES = w;
    }

    public static void main(String[] args) throws Exception {
        Options opts = parse(args);
        try {
            run(opts);
        } finally {
            System.exit(0); // ends any game thread still spinning in the engine (see generate)
        }
    }

    private static void run(Options opts) throws Exception {
        PrintStream out = System.out;
        System.setOut(new PrintStream(OutputStream.nullOutputStream(), true, StandardCharsets.UTF_8));

        BotVsBot.Context ctx = BotVsBot.context(Path.of("gemp-lotr-cards/src/main/resources"), opts.base());
        Files.createDirectories(opts.out());
        Path stepsFile = opts.steps() != null ? opts.steps() : opts.out().resolve("teacher-steps.jsonl");
        if (opts.play()) {
            generate(opts, ctx, stepsFile, out); // scores the search player; its labels are kept too
            return;
        }
        if (!opts.trainOnly())
            generate(opts, ctx, stepsFile, out);
        if (System.getProperty("teach.noTrain") == null) // the round loop needs only the boards (-Dteach.noTrain)
            train(opts, stepsFile, out);
    }

    private static void generate(Options opts, BotVsBot.Context ctx, Path stepsFile, PrintStream out) throws Exception {
        List<BotVsBot.Matchup> matchups = BotVsBot.loadSuite(opts.suite(), ctx);
        Map<String, LotroDeck> decks = new HashMap<>();
        for (BotVsBot.Matchup m : matchups)
            for (String spec : List.of(m.deck1(), m.deck2()))
                decks.computeIfAbsent(m.format() + "|" + spec, k -> BotVsBot.loadDeck(spec, m.format(), ctx));
        ModelRegistry models = ctx.models(opts.base());
        RLGameStateFeatures features = models.getFeatureVersion() >= GeneralStateFeatures.VERSION
                ? new GeneralStateFeatures() : new FotrStartersRLGameStateFeatures();
        // guarded: the models alone corrupt their own Ring-bearer with burden costs on real decks (see Guarded)
        String policy = (System.getenv("BOT_UNGUARDED") != null ? "general@" : "safe@") + opts.base();
        if (opts.depth() > 0) {
            Path file = opts.value().resolve("value.model");
            if (!Files.exists(file))
                BotVsBot.die("--depth needs " + file + " (BotProbe --save-value " + opts.value() + ")");
            try (ObjectInputStream in = new ObjectInputStream(new FileInputStream(file.toFile()))) {
                @SuppressWarnings("unchecked")
                SoftClassifier<double[]> model = (SoftClassifier<double[]>) in.readObject();
                valueModel = model;
            }
            Path vf = opts.value().resolve("value-features.txt");
            if (Files.exists(vf))
                valueFeatures = Integer.parseInt(Files.readString(vf).trim());
            out.printf("Rollouts stop %d turn%s after the branch and are scored by %s (feature set %d)%n", opts.depth(),
                    opts.depth() == 1 ? "" : "s", file, valueFeatures);
        }

        if (opts.fromCopies()) {
            long c0 = System.nanoTime();
            copier = new GameCopier(ctx.library(), ctx.formats(), decks);
            out.printf("Rollouts play on copies of the game (%,d shared objects, %.1fs)%n", copier.sharedCount(),
                    (System.nanoTime() - c0) / 1e9);
        }
        out.printf("Teaching on %d matchups from %s: %d games, %.0f%% of card-action decisions branched, %d rollouts "
                        + "per option, on %d threads; steps to %s%n", matchups.size(), opts.suite(), opts.games(),
                100 * opts.teachRate(), opts.rollouts(), opts.jobs(), stepsFile);
        if (opts.play())
            out.println("Play mode: A branches every card-action decision and plays the option that won most; "
                    + "B is the plain bot. A takes deck 1 on even games of each matchup.");
        long t0 = System.nanoTime();
        AtomicInteger done = new AtomicInteger(), aWins = new AtomicInteger(), finished = new AtomicInteger();
        // CPU time of the game threads, rollouts included (they run on the game's thread): wall time x threads
        // overstates it whenever stragglers leave threads idle
        java.util.concurrent.atomic.AtomicLong cpuNanos = new java.util.concurrent.atomic.AtomicLong();
        Map<String, int[]> record = new ConcurrentHashMap<>(); // deck -> {wins, games}
        Map<String, int[]> pairs = new ConcurrentHashMap<>(); // "deck1 vs deck2" -> {deck1 wins, games}
        Map<String, Map<String, Integer>> endings = new ConcurrentHashMap<>(); // deck -> "won: reason" -> games
        var threads = java.lang.management.ManagementFactory.getThreadMXBean();
        try (Writer sink = Files.newBufferedWriter(stepsFile, StandardCharsets.UTF_8,
                java.nio.file.StandardOpenOption.CREATE, java.nio.file.StandardOpenOption.APPEND)) {
            ExecutorService pool = Executors.newFixedThreadPool(opts.jobs());
            List<Future<?>> futures = new ArrayList<>();
            for (int g = 0; g < opts.games(); g++) {
                int game = g;
                futures.add(pool.submit(() -> {
                    long cpu0 = threads.getCurrentThreadCpuTime();
                    BotVsBot.Matchup m = matchups.get(game % matchups.size());
                    boolean swap = (game / matchups.size()) % 2 == 1;
                    String deckA = swap ? m.deck2() : m.deck1(), deckB = swap ? m.deck1() : m.deck2();
                    Map<String, LotroDeck> seats = Map.of(
                            "~a", decks.get(m.format() + "|" + deckA),
                            "~b", decks.get(m.format() + "|" + deckB));
                    LotroFormat format = ctx.formats().getFormat(m.format());
                    long seed = new SplittableRandom().nextLong();
                    List<BotGame.Step> steps = new ArrayList<>();
                    String opponent = OPPONENTS.isEmpty() ? policy : OPPONENTS.get(game % OPPONENTS.size());
                    Branch branch = new Branch(format, ctx, seats, seed, steps, policy, opponent, opts);
                    List<BotPlayer> bots = List.of(
                            new Teaching(BotVsBot.bot(policy, "~a", ctx), branch, features, sink, opts, opts.play()),
                            !OPPONENTS.isEmpty() || opts.play() && !opts.both() ? BotVsBot.bot(opponent, "~b", ctx)
                                    : new Teaching(BotVsBot.bot(policy, "~b", ctx), branch, features, sink, opts, opts.play()));
                    // its rollouts make this game slow, not stuck
                    POSITIONS.get().clear();
                    LOWEST_RESISTANCE.get().clear();
                    BotGame.Result r = BotGame.play(format, ctx.library(), bots, seats, maxDecisions, seed, steps, 24 * 3600);
                    if (r.finished() && opts.recordValues() != null) {
                        StringBuilder rows = new StringBuilder();
                        for (Object[] row : POSITIONS.get()) {
                            rows.append(row[0].equals(r.winner()) ? 1 : 0);
                            for (double x : (double[]) row[1])
                                rows.append(',').append(x == Math.rint(x) ? String.valueOf((long) x) : String.valueOf(x));
                            rows.append('\n');
                        }
                        synchronized (BotTeach.class) {
                            try {
                                Files.writeString(opts.recordValues(), rows, StandardCharsets.UTF_8,
                                        java.nio.file.StandardOpenOption.CREATE, java.nio.file.StandardOpenOption.APPEND);
                            } catch (IOException e) {
                                throw new UncheckedIOException(e);
                            }
                        }
                    }
                    if (r.finished()) {
                        finished.incrementAndGet();
                        if (r.winner().equals("~a"))
                            aWins.incrementAndGet();
                        String winnerDeck = r.winner().equals("~a") ? deckA : deckB;
                        // the reason names the loser's fate ("The Ring-Bearer is dead") or the winner's feat
                        // ("Surviving to Regroup phase on site 9"), so file it from both sides
                        String why = String.valueOf(r.reason()).replaceAll("\\d+", "N");
                        for (String deck : List.of(deckA, deckB))
                            endings.computeIfAbsent(deck, k -> new ConcurrentHashMap<>())
                                    .merge((deck.equals(winnerDeck) ? "won: " : "lost: ") + why, 1, Integer::sum);
                        // how close each Ring-bearer came to corruption: its lowest resistance less burdens
                        for (String seat : List.of("~a", "~b")) {
                            Integer low = LOWEST_RESISTANCE.get().get(seat);
                            if (low == null)
                                continue;
                            String bucket = low <= 0 ? "0 (corrupted)" : low >= 4 ? "4+" : String.valueOf(low);
                            endings.computeIfAbsent(seat.equals("~a") ? deckA : deckB, k -> new ConcurrentHashMap<>())
                                    .merge("~ own Ring-bearer's lowest resistance: " + bucket, 1, Integer::sum);
                        }
                        pairs.compute(m.deck1() + " vs " + m.deck2(), (k, v) -> {
                            int[] w = v == null ? new int[2] : v;
                            w[1]++;
                            if (winnerDeck.equals(m.deck1()))
                                w[0]++;
                            return w;
                        });
                        for (String deck : List.of(deckA, deckB))
                            record.compute(deck, (k, v) -> {
                                int[] w = v == null ? new int[2] : v;
                                w[1]++;
                                if (deck.equals(winnerDeck))
                                    w[0]++;
                                return w;
                            });
                    }
                    cpuNanos.addAndGet(threads.getCurrentThreadCpuTime() - cpu0);
                    int n = done.incrementAndGet();
                    if (n % 10 == 0 || n == opts.games()) {
                        double s = (System.nanoTime() - t0) / 1e9;
                        out.printf("  %d games, %d decisions taught, %d labelled steps, %d rollouts (%.0f/s)%s, %.0fs%s%n",
                                n, taught.get(), labelled.get(), rolloutsRun.get(), rolloutsRun.get() / s,
                                (diverged.get() > 0 ? ", " + diverged.get() + " REPLAYS DIVERGED" : "")
                                        + (cutOff.get() > 0 ? ", " + cutOff.get() + " cut off" : ""), s,
                                opts.play() ? String.format("; search won %d of %d", aWins.get(), finished.get()) : "");
                    }
                }));
            }
            pool.shutdown();
            // A game can spin inside one engine call for hours -- GameStats.updateGameStats evaluating strength
            // through modifier conditions that spot cards whose types are themselves modified -- where neither
            // the watchdog's winner nor anything else can reach it, and Java cannot stop the thread. So: once no
            // game has finished for STALL_MINUTES, give up on the rest; main() exits, which ends their threads.
            int lastDone = -1;
            long lastProgress = System.nanoTime();
            while (!pool.awaitTermination(10, TimeUnit.SECONDS)) {
                if (done.get() != lastDone) {
                    lastDone = done.get();
                    lastProgress = System.nanoTime();
                } else if (System.nanoTime() - lastProgress > STALL_MINUTES * 60_000_000_000L) {
                    out.printf("  %d games abandoned: no game finished for %d minutes%n", opts.games() - done.get(),
                            STALL_MINUTES);
                    break;
                }
            }
            for (Future<?> f : futures)
                if (f.isDone())
                    f.get(); // surface any exception
        }
        if (matchups.size() > 1 && !pairs.isEmpty()) {
            out.println("  head to head (first deck's wins / games, 95% interval):");
            new TreeMap<>(pairs).forEach((k, w) -> {
                double p = (double) w[0] / w[1];
                out.printf("    %5.1f%% +- %4.1f  %4d / %-4d  %s%n", 100 * p, 196 * Math.sqrt(p * (1 - p) / w[1]), w[0],
                        w[1], k);
            });
        }
        if (matchups.size() > 1 && !record.isEmpty()) {
            out.println("  standings (wins / games, 95% interval):");
            record.entrySet().stream()
                    .sorted(Comparator.comparingDouble(e -> -(double) e.getValue()[0] / e.getValue()[1]))
                    .forEach(e -> {
                        double p = (double) e.getValue()[0] / e.getValue()[1];
                        out.printf("    %5.1f%% +- %4.1f  %4d / %-4d  %s%n", 100 * p,
                                196 * Math.sqrt(p * (1 - p) / e.getValue()[1]), e.getValue()[0], e.getValue()[1],
                                e.getKey());
                    });
        }
        if (assignmentsSearched.get() > 0)
            out.printf("  %d assignment decisions searched%n", assignmentsSearched.get());
        if (choicesSearched.get() > 0)
            out.printf("  %d multiple-choice decisions searched%n", choicesSearched.get());
        if (assignmentsTaught.get() > 0)
            out.printf("  %d assignments taught%n", assignmentsTaught.get());
        if (movesSearched.get() > 0)
            out.printf("  %d \"move again?\" decisions searched with %d rollouts, moved on %d%n", movesSearched.get(),
                    moveRollouts, movesTaken.get());
        if (opts.fromCopies() && CloneRollout.copies.get() > 0)
            out.printf("  %d rollouts from copies, %.0f us a copy%n", CloneRollout.copies.get(),
                    CloneRollout.copyNanos.get() / 1e3 / CloneRollout.copies.get());
        if (matchups.size() > 1 && !endings.isEmpty()) {
            out.println("  how each deck's games ended:");
            new TreeMap<>(endings).forEach((deck, e) -> {
                out.println("    " + deck);
                e.entrySet().stream().sorted(Map.Entry.comparingByKey(Comparator.reverseOrder()))
                        .forEach(x -> out.printf("      %5d  %s%n", x.getValue(), x.getKey()));
            });
        }
        if (opts.play() && finished.get() > 0) {
            double p = aWins.get() / (double) finished.get();
            out.printf("search player won %d of %d: %.1f%% +- %.1f; %.1f CPU-seconds a game%n",
                    aWins.get(), finished.get(), 100 * p, 196 * Math.sqrt(p * (1 - p) / finished.get()),
                    cpuNanos.get() / 1e9 / Math.max(1, done.get()));
        }
    }

    private static void train(Options opts, Path stepsFile, PrintStream out) throws Exception {
        ModelRegistry base = BotVsBot.loadModels(opts.base());
        CardFeatures.setVersion(base.getFeatureVersion());
        com.gempukku.lotro.bots.rl.fotrstarters.models.AbstractTrainer.lambda = opts.lambda();
        // A decision's options are written together (TrainingSteps, teach()): consecutive lines with the same state and
        // question are one decision, which ListwiseTreeModel compares its options within
        List<LearningStep> steps = new ArrayList<>();
        String lastKey = null;
        int group = -1;
        for (String line : Files.readAllLines(stepsFile)) {
            if (line.isBlank())
                continue;
            int a = line.indexOf("\"action\""), d = line.indexOf("\"decision\"");
            String key = (a > 0 ? line.substring(0, a) : "") + (d > 0 ? line.substring(d) : "");
            key = key.replaceAll("\"reward\":[-0-9.]+,?", "");
            if (!key.equals(lastKey))
                group++;
            lastKey = key;
            LearningStep step = LearningStep.fromJson(line);
            step.group = group;
            steps.add(step);
        }
        out.printf("Refitting card-action models of %s on %d teacher-labelled steps (models with at least %d; "
                + "lambda %s)%n", opts.base(), steps.size(), opts.minSteps(), opts.lambda());

        // The models are fit side by side, biggest first: one at a time, a refit on human labels kept ~2.7 of 8
        // cores busy and took 17-60 minutes, most of it in the many small models after the big one.
        List<Map.Entry<Trainer, List<LearningStep>>> work = new ArrayList<>();
        // -Dtrain.only=Name,Name: refit just these; the rest stay the base's
        Set<String> only = System.getProperty("train.only") == null ? null
                : Set.of(System.getProperty("train.only").split(","));
        for (Trainer t : BotService.extendedTrainers()) {
            if (only != null && !only.contains(t.getClass().getSimpleName()))
                continue;
            List<LearningStep> relevant = steps.stream().filter(t::isStepRelevant).toList();
            if (relevant.size() >= opts.minSteps()) // fewer cannot beat the outcome-trained model it would replace
                work.add(Map.entry(t, relevant));
        }
        work.sort(Comparator.comparingInt(e -> -e.getValue().size()));
        int version = base.getFeatureVersion();
        // -Dtrain.threads: each fit holds its rows and a copy as a DataFrame; 3.8 M rows on all cores ran out of 28 GB
        ExecutorService pool = Executors.newFixedThreadPool(Integer.getInteger("train.threads",
                Math.max(1, Runtime.getRuntime().availableProcessors() - 1)));
        List<Future<SoftClassifier<double[]>>> fits = new ArrayList<>();
        for (var e : work)
            fits.add(pool.submit(() -> {
                CardFeatures.setVersion(version); // per thread
                return e.getKey().train(e.getValue());
            }));
        for (int i = 0; i < work.size(); i++) {
            Trainer t = work.get(i).getKey();
            try {
                base.registerModel(t.getClass(), fits.get(i).get());
            } catch (ExecutionException e) { // say so: System.out is muted, and the exit in main hid an OOM here
                out.println("  " + t.getClass().getSimpleName() + " failed: " + e.getCause());
                // and what is under it: a refit's own wrapper said only "anyPhase.refitContexts"
                for (Throwable c = e.getCause().getCause(); c != null; c = c.getCause())
                    out.println("    caused by " + c);
                pool.shutdownNow();
                throw e;
            }
            out.printf("  %-45s %6d steps%n", t.getClass().getSimpleName(), work.get(i).getValue().size());
        }
        pool.shutdown();
        for (Trainer t : BotService.extendedTrainers())
            if (base.hasModel(t.getClass()) && base.getModel(t.getClass()) != null)
                BotTrain.save(opts.out(), t, base.getModel(t.getClass()));
        Files.writeString(opts.out().resolve("features.txt"), base.getFeatureVersion() + "\n");
        if (com.gempukku.lotro.bots.rl.fotrstarters.models.cardaction.AnyPhaseCardActionTrainer.RICH_TRAINING)
            Files.writeString(opts.out().resolve("any-phase-rich"), "");
        if (com.gempukku.lotro.bots.rl.fotrstarters.models.cardaction.AnyPhaseCardActionTrainer.PROBE_TRAINING)
            Files.writeString(opts.out().resolve("skirmish-probe"), "");
        out.println("models in " + opts.out());
    }

    /** Everything needed to replay one game to a decision and play it out from there. */
    record Branch(LotroFormat format, BotVsBot.Context ctx, Map<String, LotroDeck> decks, long seed,
                  List<BotGame.Step> steps, String policy, String opponent, Options opts) {
        /** Wins for {@code player} out of the rollouts, with {@code answer} given at the decision after the steps. */
        double value(List<BotGame.Step> prefix, String player, String answer, long[] worlds, GameState live) {
            return value(prefix, player, answer, worlds, live, opts.depth());
        }

        double value(List<BotGame.Step> prefix, String player, String answer, long[] worlds, GameState live, int depth) {
            double wins = 0;
            for (long world : worlds) {
                Shared shared;
                BotGame.Result r;
                if (opts.fromCopies()) { // the same rollout from a copy of the live game; BotCloneCheck checks it is
                    shared = new Shared(List.of(), player, answer, world, depth, opts.peek());
                    r = CloneRollout.play(copier, (com.gempukku.lotro.logic.timing.DefaultLotroGame) live.getGame(),
                            prefix, shared, BotVsBot.bot(policy, "~a", ctx), BotVsBot.bot(opponent, "~b", ctx));
                } else {
                    shared = new Shared(prefix, player, answer, world, depth, opts.peek());
                    r = BotGame.play(format, ctx.library(),
                            List.of(new Branching("~a", shared, BotVsBot.bot(policy, "~a", ctx)),
                                    new Branching("~b", shared, BotVsBot.bot(opponent, "~b", ctx))),
                            decks, 1000, seed);
                }
                rolloutsRun.incrementAndGet();
                if (shared.diverged)
                    diverged.incrementAndGet();
                if (shared.value >= 0) {
                    cutOff.incrementAndGet();
                    wins += shared.value;
                } else
                    wins += !r.finished() ? 0.5 : r.winner().equals(player) ? 1 : 0;
            }
            return wins / worlds.length;
        }
    }

    static final class Shared {
        final List<BotGame.Step> prefix;
        final String player, answer;
        final long world;
        final int depth;
        final boolean peek;
        int next, startTurn;
        com.gempukku.lotro.common.Phase startPhase;
        boolean branched, diverged;
        /** P(win) for player from the value model, if the rollout was cut off; -1 if it was played out. */
        double value = -1;

        Shared(List<BotGame.Step> prefix, String player, String answer, long world, int depth, boolean peek) {
            this.prefix = prefix;
            this.player = player;
            this.answer = answer;
            this.world = world;
            this.depth = depth;
            this.peek = peek;
        }
    }

    /** How close a Ring-bearer with this much resistance left is to corruption, 0 (far) to 1 (corrupted). */
    static double pressure(double resistanceLeft) {
        return resistanceLeft <= 0 ? 1 : resistanceLeft <= 1 ? 0.6 : resistanceLeft <= 2 ? 0.35
                : resistanceLeft <= 3 ? 0.15 : 0;
    }

    /** Replays the recorded answers, gives the option being tried, then plays as the policy bot. */
    record Branching(String name, Shared shared, BotPlayer policy) implements BotPlayer {
        @Override
        public String chooseAction(GameState gameState, AwaitingDecision decision) {
            if (shared.next < shared.prefix.size()) {
                BotGame.Step step = shared.prefix.get(shared.next++);
                if (!step.player().equals(name))
                    shared.diverged = true;
                return step.answer();
            }
            if (!shared.branched) {
                shared.branched = true;
                if (!name.equals(shared.player))
                    shared.diverged = true;
                // from here the future is one of many: new draws, and decks in an order nobody has seen
                gameState.setRandomSeed(shared.world);
                if (!shared.peek && gameState.isInit()) { // before the deal there are no hands
                    // The searching player cannot see the opponent's hand: deal it again from the cards they have
                    // not seen, hand and deck together. Without this every rollout knew it, and a search that
                    // knows what tricks are coming plays a game nobody at a table can.
                    for (String p : gameState.getPlayerNames()) {
                        if (p.equals(shared.player))
                            continue;
                        List<com.gempukku.lotro.game.PhysicalCard> hand = new ArrayList<>(gameState.getHand(p));
                        gameState.removeCardsFromZone(null, hand);
                        gameState.shuffleCardsIntoDeck(hand, p);
                        for (int i = 0; i < hand.size(); i++)
                            gameState.playerDrawsCard(p);
                    }
                }
                com.gempukku.lotro.bots.BotRandom.current().setSeed(shared.world ^ 0x5DEECE66DL);
                for (String p : gameState.getPlayerNames())
                    gameState.shuffleDeck(p);
                shared.startTurn = gameState.getTurnNumber();
                shared.startPhase = gameState.getCurrentPhase();
                return shared.answer;
            }
            boolean far = shared.depth > 0 && shared.value < 0 && (PHASE_CUT ? gameState.getCurrentPhase() != shared.startPhase
                    || gameState.getTurnNumber() > shared.startTurn : gameState.getTurnNumber() >= shared.startTurn + shared.depth);
            if (far && JUDGE != null) {
                // the learned judge reads boards only at decisions it was trained on, from the decider's side
                if (name.equals(shared.player) && BoardDump.kind(decision) != null) {
                    double[] st = com.gempukku.lotro.bots.rl.fotrstarters.BoardNet2.state(gameState, decision, shared.player);
                    double[] o = st == null ? null : JUDGE.outcome(gameState.getGame(), decision, shared.player, st);
                    if (o != null) {
                        shared.value = o[0] + o[1] + o[2];
                        gameState.getGame().playerLost(name, "rollout cut off");
                    }
                }
                return policy.chooseAction(gameState, decision);
            }
            if (far) {
                // far enough: score the position for the player who branched, and end this throwaway game
                double[] p = new double[2];
                double[] v = new GeneralStateFeatures().extractFeatures(gameState, decision, shared.player);
                if (valueFeatures == 3)
                    v = com.gempukku.lotro.bots.rl.fotrstarters.KnowledgeFeatures.withKnowledge(v, gameState, shared.player);
                else if (valueFeatures == 4)
                    v = com.gempukku.lotro.bots.rl.fotrstarters.CorruptionFeatures.withCorruption(v, gameState, shared.player);
                valueModel.predict(v, p);
                shared.value = p[1];
                if (progressWeight > 0) {
                    // The race, which outcome-fit value models learn backwards: in bot games a fellowship further
                    // along is more likely to lose (later sites are deadlier, moving feeds the Shadow twilight), so
                    // value-real rates the player at site 8 at 42% against 59% at site 2 -- and search steers away
                    // from moving. Site 9 survived is a win; reward getting there.
                    double mine = v[GeneralStateFeatures.index("myPosition")],
                            theirs = v[GeneralStateFeatures.index("oppPosition")];
                    shared.value = Math.max(0, Math.min(1, shared.value
                            + progressWeight * (Math.pow(mine / 9, 2) - Math.pow(theirs / 9, 2))));
                }
                if (corruptionWeight > 0) {
                    // A rule the value model cannot learn from outcomes (see BOT-NOTES.md, corruption): a Ring-bearer
                    // a burden or two from its resistance is close to losing.
                    double mine = v[GeneralStateFeatures.index("myRbResistance")],
                            theirs = v[GeneralStateFeatures.index("oppRbResistance")];
                    shared.value = Math.max(0, Math.min(1, shared.value
                            + corruptionWeight * (pressure(theirs) - pressure(mine))));
                }
                gameState.getGame().playerLost(name, "rollout cut off");
            }
            return policy.chooseAction(gameState, decision);
        }

        @Override
        public String getName() {
            return name;
        }
    }

    /** The bot in the main game; now and then it stops to branch a decision before answering it as usual. */
    record Teaching(BotPlayer inner, Branch branch, RLGameStateFeatures record, Writer sink, Options opts,
                    boolean search) implements BotPlayer {
        @Override
        public String chooseAction(GameState gameState, AwaitingDecision decision) {
            try {
                var game = gameState.getGame();
                for (String p : gameState.getPlayerNames()) {
                    var bearer = gameState.getRingBearer(p);
                    if (bearer != null && gameState.isInit())
                        LOWEST_RESISTANCE.get().merge(p, game.getModifiersQuerying().getResistance(game, bearer), Math::min);
                }
            } catch (RuntimeException ignored) {
                // before the deal
            }
            if (opts.recordValues() != null && decision.getDecisionType() == AwaitingDecisionType.CARD_ACTION_CHOICE) {
                try {
                    POSITIONS.get().add(new Object[]{getName(),
                            new GeneralStateFeatures().extractFeatures(gameState, decision, getName())});
                } catch (RuntimeException ignored) {
                }
            }
            String[] actionIds = decision.getDecisionParameters().get("actionId");
            if (decision.getDecisionType() == AwaitingDecisionType.CARD_ACTION_CHOICE && actionIds != null
                    && actionIds.length > 0 && searchedPhase(decision) && (search ? cardSearch
                    : ThreadLocalRandom.current().nextDouble() < opts.teachRate())) {
                // searching, the bot's own choice is always a candidate (--max-candidates below the number of options
                // then searches the model's pick, pass, and random others); asked once, so its loop guard stays in step
                String mine = search ? inner.chooseAction(gameState, decision) : null;
                String best = teach(gameState, decision, actionIds, mine);
                if (search)
                    return best != null ? best : mine;
            }
            if (!search && rbAssignCheck && decision.getDecisionType() == AwaitingDecisionType.ASSIGN_MINIONS
                    && getName().equals(gameState.getCurrentPlayerId())) {
                String mine = inner.chooseAction(gameState, decision);
                List<String> options = rbAlternatives(gameState, decision, mine);
                if (options.size() > 1)
                    teachAssignment(gameState, decision, options);
                return mine;
            }
            if (!search && opts.assign() && decision.getDecisionType() == AwaitingDecisionType.ASSIGN_MINIONS
                    && starter() instanceof com.gempukku.lotro.bots.rl.fotrstarters.FotrStarterBot bot
                    && ThreadLocalRandom.current().nextDouble() < (assignTeachRate >= 0 ? assignTeachRate : opts.teachRate())) {
                // Teacher labels for assignments too: who skirmishes whom decides most games, and the human
                // assignments were the biggest single gain (human22-as, +6 points). Same candidates as the search.
                List<String> options = new ArrayList<>(bot.assignmentCandidates(gameState, decision,
                        Math.max(1, opts.maxCandidates() - 1)));
                if (getName().equals(gameState.getCurrentPlayerId()) && !options.contains(""))
                    options.add("");
                if (options.size() > 1)
                    teachAssignment(gameState, decision, options);
            }
            if (!search && deckOrderTeachRate > 0 && decision.getDecisionType() == AwaitingDecisionType.ARBITRARY_CARDS
                    && decision.getText() != null && DECK_ORDER.matcher(decision.getText()).find()
                    && ThreadLocalRandom.current().nextDouble() < deckOrderTeachRate) {
                var p = decision.getDecisionParameters();
                String[] ids = p.get("cardId"), blueprints = p.get("blueprintId"), selectable = p.get("selectable");
                if (ids != null && blueprints != null && selectable != null && p.get("max") != null
                        && "1".equals(p.get("max")[0])) {
                    List<String> options = new ArrayList<>();
                    Set<String> seen = new HashSet<>();
                    for (int i = 0; i < ids.length; i++)
                        if ("true".equals(selectable[i]) && seen.add(blueprints[i]))
                            options.add(ids[i]);
                    if (options.size() > 1)
                        teachPick(gameState, decision, options);
                }
            }
            if (!search && triggerOrderTeachRate > 0 && decision.getDecisionType() == AwaitingDecisionType.ACTION_CHOICE
                    && ThreadLocalRandom.current().nextDouble() < triggerOrderTeachRate) {
                String[] ids = decision.getDecisionParameters().get("actionId");
                if (ids != null && ids.length > 1)
                    teachPick(gameState, decision, Arrays.asList(ids));
            }
            if (search && opts.assign() && decision.getDecisionType() == AwaitingDecisionType.ASSIGN_MINIONS
                    && starter() instanceof com.gempukku.lotro.bots.rl.fotrstarters.FotrStarterBot bot) {
                // Who skirmishes whom decides most games here: 94% end with a Ring-bearer killed, usually in a
                // skirmish it was left to fight alone. The model's best few assignments, and (for the Free Peoples)
                // assigning nothing and leaving it to the Shadow player.
                List<String> options = new ArrayList<>(bot.assignmentCandidates(gameState, decision,
                        Math.max(1, opts.maxCandidates() - 1)));
                if (getName().equals(gameState.getCurrentPlayerId()) && !options.contains(""))
                    options.add("");
                if (options.size() > 1) {
                    assignmentsSearched.incrementAndGet();
                    String best = pick(gameState, options);
                    if (best != null) {
                        inner.chooseAction(gameState, decision);
                        return best;
                    }
                }
            }
            if (search && cardPicks && decision.getDecisionType() == AwaitingDecisionType.ARBITRARY_CARDS) {
                // "Choose cards from deck" and the like: which card a search-for-a-card effect fetches, answered at
                // random by the models (0.6 a game between real decks). One pick only; copies of a card are one option.
                // Measured as nothing (69.3% against 69.1% +- 2.3 on held-out real decks) at +30% CPU: opt-in.
                var p = decision.getDecisionParameters();
                String[] ids = p.get("cardId"), blueprints = p.get("blueprintId"), selectable = p.get("selectable");
                int min = p.get("min") != null ? Integer.parseInt(p.get("min")[0]) : 0;
                int max = p.get("max") != null ? Integer.parseInt(p.get("max")[0]) : 0;
                if (max == 1 && ids != null && blueprints != null && selectable != null) {
                    List<String> options = new ArrayList<>();
                    Set<String> seen = new HashSet<>();
                    for (int i = 0; i < ids.length; i++)
                        if ("true".equals(selectable[i]) && seen.add(blueprints[i]))
                            options.add(ids[i]);
                    Collections.shuffle(options);
                    options = new ArrayList<>(options.subList(0, Math.min(options.size(), opts.maxCandidates())));
                    if (min == 0)
                        options.add("");
                    if (options.size() > 1) {
                        choicesSearched.incrementAndGet();
                        String best = pick(gameState, options);
                        if (best != null) {
                            inner.chooseAction(gameState, decision);
                            return best;
                        }
                    }
                }
            }
            if (search && opts.choices() && decision.getDecisionType() == AwaitingDecisionType.INTEGER) {
                // The burden bid above all, which the models answer with a uniform 0-4 -- and which can corrupt a
                // low-resistance Ring-bearer before the game starts. Up to five values from the minimum.
                String[] min = decision.getDecisionParameters().get("min"), max = decision.getDecisionParameters().get("max");
                int lo = min != null ? Integer.parseInt(min[0]) : 0;
                int hi = Math.min(max != null ? Integer.parseInt(max[0]) : Integer.MAX_VALUE, lo + 4);
                if (hi > lo) {
                    List<String> options = new ArrayList<>();
                    for (int v = lo; v <= hi; v++)
                        options.add(String.valueOf(v));
                    choicesSearched.incrementAndGet();
                    String best = pick(gameState, options);
                    if (best != null) {
                        inner.chooseAction(gameState, decision);
                        return best;
                    }
                }
            }
            String[] results = decision.getDecisionParameters().get("results");
            if (search && opts.choices() && decision.getDecisionType() == AwaitingDecisionType.MULTIPLE_CHOICE
                    && results != null && results.length > 1) {
                // Yes/no and pick-one questions -- move again or stop, pay a cost to assign, keep or discard --
                // which the models answer from a fixed state ranking or not at all. Searched, not labelled.
                boolean move = moveRollouts > 0 && decision.getText() != null
                        && decision.getText().toLowerCase().contains("another move");
                String best = move ? choose(gameState, results.length, moveRollouts, moveDepth)
                        : choose(gameState, results.length, opts.rollouts(), opts.depth());
                if (move) {
                    movesSearched.incrementAndGet();
                    String answer = best != null ? best : inner.chooseAction(gameState, decision);
                    if (answer != null && answer.matches("\\d+") && Integer.parseInt(answer) < results.length
                            && results[Integer.parseInt(answer)].equalsIgnoreCase("yes"))
                        movesTaken.incrementAndGet();
                    if (best == null)
                        return answer;
                }
                if (best != null) {
                    inner.chooseAction(gameState, decision);
                    return best;
                }
            }
            return inner.chooseAction(gameState, decision);
        }

        /** The option that won most over the rollouts; null if they all scored the same. */
        private String pick(GameState gameState, List<String> options) {
            List<BotGame.Step> prefix = List.copyOf(branch.steps());
            long[] worlds = new SplittableRandom().longs(opts.rollouts()).toArray();
            String bestOption = null;
            double best = -1, worst = 2;
            for (String option : options) {
                double v = branch.value(prefix, getName(), option, worlds, gameState);
                if (v > best) {
                    best = v;
                    bestOption = option;
                }
                worst = Math.min(worst, v);
            }
            return best > worst ? bestOption : null;
        }

        /** The answer (an index into the results) that won most over the rollouts; null on a tie. */
        private String choose(GameState gameState, int n, int rollouts, int depth) {
            List<BotGame.Step> prefix = List.copyOf(branch.steps());
            long[] worlds = new SplittableRandom().longs(rollouts).toArray();
            String bestOption = null;
            double best = -1, worst = 2;
            for (int i = 0; i < n; i++) {
                double v = branch.value(prefix, getName(), String.valueOf(i), worlds, gameState, depth);
                if (v > best) {
                    best = v;
                    bestOption = String.valueOf(i);
                }
                worst = Math.min(worst, v);
            }
            choicesSearched.incrementAndGet();
            return best > worst ? bestOption : null;
        }

        /** Branches the decision and writes its labels; returns the option that won most, or null on a tie. */
        /**
         * The learned bot under the guard. Checking {@code inner} itself found a Guarded since rollouts and main games
         * were guarded by default, and turned assignment search off without a word: no run since had searched one.
         */
        private BotPlayer starter() {
            return inner instanceof Guarded g ? g.inner() : inner;
        }

        /** What card action {@code answer} did on a copy (EffectProbe), the models answering on the way. */
        private double[] probe(GameState gameState, AwaitingDecision decision, String answer) {
            if (decision.getDecisionType() != AwaitingDecisionType.CARD_ACTION_CHOICE
                    || !(starter() instanceof com.gempukku.lotro.bots.rl.fotrstarters.FotrStarterBot bot))
                return null;
            if (answer.isEmpty())
                return com.gempukku.lotro.bots.rl.fotrstarters.EffectProbe.pass(gameState, getName());
            double[] f = com.gempukku.lotro.bots.rl.fotrstarters.EffectProbe.probe(gameState, getName(), answer, bot::as);
            return f != null ? f : com.gempukku.lotro.bots.rl.fotrstarters.EffectProbe.none();
        }

        /** Branches an assignment over {@code options} and writes the labels, as {@link #teach} does for card actions. */
        /** Branches a one-card pick over {@code options} and writes the labels, as {@link #teach} does for card actions. */
        private void teachPick(GameState gameState, AwaitingDecision decision, List<String> options) {
            List<BotGame.Step> prefix = List.copyOf(branch.steps());
            long[] worlds = new SplittableRandom().longs(opts.rollouts()).toArray();
            double[] value = new double[options.size()];
            double best = 0, worst = 1;
            for (int i = 0; i < options.size(); i++) {
                value[i] = branch.value(prefix, getName(), options.get(i), worlds, gameState);
                best = Math.max(best, value[i]);
                worst = Math.min(worst, value[i]);
            }
            if (PICK_VALUES != null)
                synchronized (PICK_VALUES) {
                    PICK_VALUES.println(decision.getText().replace('\t', ' ') + "\t" + options.size() + "\t" + best + "\t"
                            + worst + "\t" + opts.rollouts());
                }
            if (Arrays.stream(value).allMatch(v -> v == value[0])
                    || decision.getDecisionType() != AwaitingDecisionType.ARBITRARY_CARDS)
                return; // trigger order: measured only, no semantic action to write it with yet
            double[] state = record.extractFeatures(gameState, decision, getName());
            boolean fp = getName().equals(gameState.getCurrentPlayerId());
            StringBuilder lines = new StringBuilder();
            for (int i = 0; i < options.size(); i++) {
                int label = value[i] >= best - 1e-9 ? 1 : value[i] <= best - opts.margin() ? 0 : -1;
                if (label < 0)
                    continue;
                LearningStep step = new LearningStep(state,
                        new com.gempukku.lotro.bots.rl.semanticaction.ChooseFromArbitraryCardsAction(options.get(i), decision),
                        fp, decision);
                step.reward = label == 1 ? 10.0 : 0.0;
                lines.append(step.toJson()).append('\n');
                labelled.incrementAndGet();
            }
            synchronized (sink) {
                try {
                    sink.write(lines.toString());
                    sink.flush();
                } catch (IOException e) {
                    throw new UncheckedIOException(e);
                }
            }
        }

        /**
         * The bot's answer first, then the same with the Ring-bearer's minions moved onto each idle character, where the
         * rules allow; just the answer if it puts nothing on the Ring-bearer or leaves no one idle.
         */
        private List<String> rbAlternatives(GameState gs, AwaitingDecision decision, String mine) {
            List<String> options = new ArrayList<>();
            options.add(mine == null ? "" : mine);
            try {
                var bearer = gs.getRingBearer(getName());
                String[] free = decision.getDecisionParameters().get("freeCharacters");
                String[] minions = decision.getDecisionParameters().get("minions");
                if (bearer == null || free == null)
                    return options;
                String rb = String.valueOf(bearer.getCardId());
                var answer = com.gempukku.lotro.bots.rl.fotrstarters.models.assignment.AssignmentProbe.parse(mine);
                List<String> onRb = answer.getOrDefault(rb, List.of());
                if (onRb.isEmpty())
                    return options;
                var legality = new com.gempukku.lotro.bots.AssignmentLegality(gs.getGame());
                for (String c : free) {
                    if (c.equals(rb) || !answer.getOrDefault(c, List.of()).isEmpty())
                        continue;
                    Map<String, List<String>> alt = new LinkedHashMap<>();
                    answer.forEach((k, v) -> alt.put(k, new ArrayList<>(v)));
                    alt.remove(rb);
                    alt.put(c, new ArrayList<>(onRb));
                    if (legality.isLegal(com.gempukku.lotro.common.Side.FREE_PEOPLE, free, minions, alt))
                        options.add(com.gempukku.lotro.bots.AssignmentLegality.format(alt));
                }
            } catch (RuntimeException ignored) {
            }
            if (options.size() > 1)
                rbChecked.incrementAndGet();
            return options;
        }

        private void teachAssignment(GameState gameState, AwaitingDecision decision, List<String> options) {
            List<BotGame.Step> prefix = List.copyOf(branch.steps());
            long[] worlds = new SplittableRandom().longs(opts.rollouts()).toArray();
            double[] value = new double[options.size()];
            double best = 0;
            for (int i = 0; i < options.size(); i++) {
                value[i] = branch.value(prefix, getName(), options.get(i), worlds, gameState);
                best = Math.max(best, value[i]);
            }
            if (PICK_VALUES != null && rbAssignCheck)
                synchronized (PICK_VALUES) {
                    StringBuilder v = new StringBuilder();
                    for (double x : value)
                        v.append(x).append(' ');
                    PICK_VALUES.println("rb-assign\t" + options.size() + "\t" + v.toString().trim() + "\t"
                            + gameState.getTurnNumber() + "\t" + opts.rollouts());
                }
            if (Arrays.stream(value).allMatch(v -> v == value[0]))
                return;
            assignmentsTaught.incrementAndGet();
            double[] state = record.extractFeatures(gameState, decision, getName());
            boolean fp = getName().equals(gameState.getCurrentPlayerId());
            StringBuilder lines = new StringBuilder();
            for (int i = 0; i < options.size(); i++) {
                int label = value[i] >= best - 1e-9 ? 1 : value[i] <= best - opts.margin() ? 0 : -1;
                if (label < 0)
                    continue;
                try {
                    LearningStep step = new LearningStep(state,
                            new com.gempukku.lotro.bots.rl.semanticaction.AssignMinionsAction(options.get(i), decision,
                                    gameState, fp).withProbe(com.gempukku.lotro.bots.rl.fotrstarters.models.assignment
                                    .AssignmentProbe.features(gameState, decision, com.gempukku.lotro.bots.rl.fotrstarters
                                    .models.assignment.AssignmentProbe.parse(options.get(i)))), fp, decision);
                    step.reward = label == 1 ? 10.0 : 0.0;
                    lines.append(step.toJson()).append('\n');
                    labelled.incrementAndGet();
                } catch (RuntimeException ignored) {
                    // an answer the semantic action cannot describe
                }
            }
            synchronized (sink) {
                try {
                    sink.write(lines.toString());
                    sink.flush();
                } catch (IOException e) {
                    throw new UncheckedIOException(e);
                }
            }
        }

        private String teach(GameState gameState, AwaitingDecision decision, String[] actionIds, String mine) {
            List<String> options = new ArrayList<>(Arrays.asList(actionIds));
            Collections.shuffle(options);
            if (mine != null && options.remove(mine))
                options.add(0, mine);
            options = new ArrayList<>(options.subList(0, Math.min(options.size(), opts.maxCandidates() - 1)));
            options.add(""); // pass
            List<BotGame.Step> prefix = List.copyOf(branch.steps());
            long[] worlds = new SplittableRandom().longs(opts.rollouts()).toArray();
            double[] value = new double[options.size()];
            double best = 0;
            for (int i = 0; i < options.size(); i++) {
                value[i] = branch.value(prefix, getName(), options.get(i), worlds, gameState);
                best = Math.max(best, value[i]);
            }
            if (Arrays.stream(value).allMatch(v -> v == value[0]))
                return null; // nothing to tell them apart
            taught.incrementAndGet();
            String bestOption = null;
            for (int i = 0; i < options.size(); i++)
                if (value[i] >= best - 1e-9 && bestOption == null)
                    bestOption = options.get(i);

            double[] state = record.extractFeatures(gameState, decision, getName());
            boolean fp = getName().equals(gameState.getCurrentPlayerId());
            double lowest = Arrays.stream(value).min().orElse(best);
            if (BoardDump.OUT != null) {
                // the board with every searched option's value -- soft targets for the combined network: what people
                // do, tilted toward what the search says wins (near-ties included: they say the options are equal).
                // "searched": [index into the decision's actions (-1 pass), value]; "game" is "teacher"
                BoardDump.GAME.set("teacher");
                double[] st = com.gempukku.lotro.bots.rl.fotrstarters.BoardNet2.state(gameState, decision, getName());
                var rec = st == null ? null : BoardDump.record(gameState.getGame(), decision, getName(), bestOption, st);
                if (rec != null) {
                    com.alibaba.fastjson2.JSONArray searched = new com.alibaba.fastjson2.JSONArray();
                    List<String> ids = Arrays.asList(actionIds);
                    for (int i = 0; i < options.size(); i++)
                        searched.add(new double[]{options.get(i).isEmpty() ? -1 : ids.indexOf(options.get(i)), value[i]});
                    rec.put("searched", searched);
                    rec.put("rollouts", opts.rollouts());
                    BoardDump.writeRecord(rec);
                }
            }
            StringBuilder lines = new StringBuilder();
            for (int i = 0; i < options.size(); i++) {
                int label = value[i] >= best - 1e-9 ? 1 : value[i] <= best - opts.margin() ? 0 : -1;
                if (label < 0)
                    continue;
                LearningStep step = new LearningStep(state,
                        new CardActionChoiceAction(options.get(i), decision, gameState)
                                .withProbe(probe(gameState, decision, options.get(i))), fp, decision);
                step.reward = label == 1 ? 10.0 : 0.0; // trainers read reward > 0 as good
                lines.append(step.toJson()).append('\n');
                labelled.incrementAndGet();
            }
            synchronized (sink) {
                try {
                    sink.write(lines.toString());
                    sink.flush();
                } catch (IOException e) {
                    throw new UncheckedIOException(e);
                }
            }
            return bestOption;
        }

        @Override
        public String getName() {
            return inner.getName();
        }
    }

    private static Options parse(String[] args) {
        String suite = "starters:train";
        Path base = Path.of("bot-models-exp/v2"), out = null;
        int games = 100, rollouts = 16, maxCandidates = 5;
        double teachRate = 0.1, margin = 0.125;
        int jobs = Math.max(1, Runtime.getRuntime().availableProcessors() - 2);
        boolean trainOnly = false, play = false, both = false, peek = false, clone = false, choices = false, assign = false;
        int minSteps = 50, depth = 0;
        double lambda = 0;
        Path steps = null, value = null, recordValues = null;
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--play" -> play = true;
                case "--both" -> both = true;
                case "--peek" -> peek = true;
                case "--clone" -> clone = true;
                case "--choices" -> choices = true;
                case "--assign" -> assign = true;
                case "--max-decisions" -> maxDecisions = Integer.parseInt(args[++i]);
                case "--card-picks" -> cardPicks = true;
                case "--move-rollouts" -> moveRollouts = Integer.parseInt(args[++i]);
                case "--boards" -> {
                    try {
                        BoardDump.OUT = new java.io.PrintWriter(java.nio.file.Files.newBufferedWriter(java.nio.file.Path.of(args[++i])), true);
                        BoardDump.TRUE_DECK = true;
                    } catch (IOException e) {
                        throw new UncheckedIOException(e);
                    }
                }
                case "--no-card-search" -> cardSearch = false;
                case "--assign-teach-rate" -> assignTeachRate = Double.parseDouble(args[++i]);
                case "--deck-order-teach-rate" -> deckOrderTeachRate = Double.parseDouble(args[++i]);
                case "--trigger-order-teach-rate" -> triggerOrderTeachRate = Double.parseDouble(args[++i]);
                case "--rb-assign-check" -> rbAssignCheck = true;
                case "--move-depth" -> moveDepth = Integer.parseInt(args[++i]);
                case "--corruption-weight" -> corruptionWeight = Double.parseDouble(args[++i]);
                case "--progress-weight" -> progressWeight = Double.parseDouble(args[++i]);
                case "--record-values" -> recordValues = Path.of(args[++i]);
                case "--model" -> {
                    String m = args[++i];
                    com.gempukku.lotro.bots.rl.fotrstarters.models.AbstractTrainer.useTrees = !m.equals("logistic");
                    com.gempukku.lotro.bots.rl.fotrstarters.models.AbstractTrainer.listwise = m.equals("listwise");
                }
                case "--min-steps" -> minSteps = Integer.parseInt(args[++i]);
                case "--depth" -> depth = Integer.parseInt(args[++i]);
                case "--lambda" -> lambda = Double.parseDouble(args[++i]);
                case "--steps" -> steps = Path.of(args[++i]);
                case "--value" -> value = Path.of(args[++i]);
                case "--suite" -> suite = args[++i];
                case "--base" -> base = Path.of(args[++i]);
                case "--out" -> out = Path.of(args[++i]);
                case "--games" -> games = Integer.parseInt(args[++i]);
                case "--rollouts" -> rollouts = Integer.parseInt(args[++i]);
                case "--teach-rate" -> teachRate = Double.parseDouble(args[++i]);
                case "--max-candidates" -> maxCandidates = Integer.parseInt(args[++i]);
                case "--margin" -> margin = Double.parseDouble(args[++i]);
                case "--jobs" -> jobs = Integer.parseInt(args[++i]);
                case "--train-only" -> trainOnly = true;
                default -> BotVsBot.die(usage());
            }
        }
        if (out == null)
            BotVsBot.die(usage());
        return new Options(suite, base, out, games, rollouts, teachRate, maxCandidates, margin, jobs, trainOnly, play,
                minSteps, lambda, steps, depth, both, value == null ? base : value, peek, clone, choices, assign, recordValues);
    }

    private static String usage() {
        return """
                usage: BotTeach --out <model dir> [options]
                  --base bot-models-exp/v2  models that play (main game and rollouts) and are refit
                  --suite starters:train    matchups to play
                  --games 100               main games
                  --teach-rate 0.1          share of card-action decisions branched
                  --rollouts 16             per option, on the same reshuffled futures
                  --max-candidates 5        options tried per decision, pass included
                  --margin 0.125            how much less an option must win to be labelled bad
                  --depth 0                 stop rollouts this many turns after the branch and score the
                                            position with <base>/value.model (BotProbe --save-value); 0 plays out
                  --value <dir>             take value.model (and value-features.txt) from here instead
                  --peek                    rollouts keep the opponent's real hand (by default it is redealt
                                            from their unseen cards)
                  --train-only              refit from <out>/teacher-steps.jsonl without playing
                  --steps <file>            labels to refit from, instead of <out>/teacher-steps.jsonl
                  --min-steps 50            only refit models with at least this many labels
                  --lambda 0                L2 penalty for the refit
                  --model logistic          or trees, for the refit
                  --play                    instead: a bot that searches every card-action decision (A)
                                            against the plain bot (B); reports A's win rate
                  --both                    with --play, B searches too
                  --clone                   rollouts copy the live game (GameCopier) instead of replaying it
                  --choices                 with --play, also search multiple-choice decisions (move again,
                                            optional costs) and small number choices (the burden bid)
                  --card-picks              with --play, also search single-card picks from a list (fetching
                                            a card from the deck); measured as no gain
                  --assign                  with --play, also search skirmish assignments: the model's best
                                            few, and assigning nothing
                  --corruption-weight 0     add w x (corruption pressure on the opponent's Ring-bearer less
                                            on mine) to cut-off values; pressure is 0.15/0.35/0.6/1 at 3/2/1/0
                                            resistance left
                  --max-decisions 1000      per bot per main game, then it concedes (the live server's rule)
                  --no-card-search          with --play, leave card actions to the models and search only
                                            assignments and choices
                  --move-rollouts 0         with --choices, search "move again?" with this many rollouts of its
                                            own, played out (or cut off after --move-depth turns)
                  --progress-weight 0       add w x ((my site / 9)^2 - (their site / 9)^2) to cut-off values
                  --record-values <file>    append the deciding player's set-2 state at each card-action
                                            decision of the main games, labelled with whether that player
                                            won, for a value model (BotProbe --fit)""";
    }
}
