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
                   int depth, boolean both, Path value, boolean peek, boolean fromCopies) {
    }

    private static final AtomicInteger taught = new AtomicInteger(), labelled = new AtomicInteger(),
            rolloutsRun = new AtomicInteger(), diverged = new AtomicInteger(), cutOff = new AtomicInteger();
    /** Scores a rollout's position when --depth stops it early: P(win) for the player whose state it is. */
    static volatile SoftClassifier<double[]> valueModel;
    /** 3 if the value model reads feature set 2 with KnowledgeFeatures appended (value-features.txt). */
    private static volatile int valueFeatures = 2;
    /** With --clone, rollouts play on a copy of the live game made by this, instead of replaying it from its seed. */
    private static volatile GameCopier copier;

    private static final long STALL_MINUTES = 15;

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
        train(opts, stepsFile, out);
    }

    private static void generate(Options opts, BotVsBot.Context ctx, Path stepsFile, PrintStream out) throws Exception {
        List<BotVsBot.Matchup> matchups = BotVsBot.loadSuite(opts.suite(), ctx);
        Map<String, LotroDeck> decks = new HashMap<>();
        for (BotVsBot.Matchup m : matchups)
            for (String spec : List.of(m.deck1(), m.deck2()))
                decks.computeIfAbsent(m.format() + "|" + spec, k -> BotVsBot.loadDeck(spec, m.format(), ctx));
        ModelRegistry models = ctx.models(opts.base());
        RLGameStateFeatures features = models.getFeatureVersion() == GeneralStateFeatures.VERSION
                ? new GeneralStateFeatures() : new FotrStartersRLGameStateFeatures();
        String policy = "general@" + opts.base();
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
                    Branch branch = new Branch(format, ctx, seats, seed, steps, policy, opts);
                    List<BotPlayer> bots = List.of(
                            new Teaching(BotVsBot.bot(policy, "~a", ctx), branch, features, sink, opts, opts.play()),
                            opts.play() && !opts.both() ? BotVsBot.bot(policy, "~b", ctx)
                                    : new Teaching(BotVsBot.bot(policy, "~b", ctx), branch, features, sink, opts, opts.play()));
                    // its rollouts make this game slow, not stuck
                    BotGame.Result r = BotGame.play(format, ctx.library(), bots, seats, 1000, seed, steps, 24 * 3600);
                    if (r.finished()) {
                        finished.incrementAndGet();
                        if (r.winner().equals("~a"))
                            aWins.incrementAndGet();
                        String winnerDeck = r.winner().equals("~a") ? deckA : deckB;
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
        if (opts.fromCopies() && CloneRollout.copies.get() > 0)
            out.printf("  %d rollouts from copies, %.0f us a copy%n", CloneRollout.copies.get(),
                    CloneRollout.copyNanos.get() / 1e3 / CloneRollout.copies.get());
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
        List<LearningStep> steps = new ArrayList<>();
        for (String line : Files.readAllLines(stepsFile))
            if (!line.isBlank())
                steps.add(LearningStep.fromJson(line));
        out.printf("Refitting card-action models of %s on %d teacher-labelled steps (models with at least %d; "
                + "lambda %s)%n", opts.base(), steps.size(), opts.minSteps(), opts.lambda());

        for (Trainer t : BotService.extendedTrainers()) {
            List<LearningStep> relevant = steps.stream().filter(t::isStepRelevant).toList();
            if (relevant.size() < opts.minSteps())
                continue; // too few to beat the outcome-trained model it would replace
            SoftClassifier<double[]> model = t.train(relevant);
            base.registerModel(t.getClass(), model);
            out.printf("  %-45s %6d steps%n", t.getClass().getSimpleName(), relevant.size());
        }
        for (Trainer t : BotService.extendedTrainers())
            if (base.hasModel(t.getClass()) && base.getModel(t.getClass()) != null)
                BotTrain.save(opts.out(), t, base.getModel(t.getClass()));
        Files.writeString(opts.out().resolve("features.txt"), base.getFeatureVersion() + "\n");
        out.println("models in " + opts.out());
    }

    /** Everything needed to replay one game to a decision and play it out from there. */
    record Branch(LotroFormat format, BotVsBot.Context ctx, Map<String, LotroDeck> decks, long seed,
                  List<BotGame.Step> steps, String policy, Options opts) {
        /** Wins for {@code player} out of the rollouts, with {@code answer} given at the decision after the steps. */
        double value(List<BotGame.Step> prefix, String player, String answer, long[] worlds, GameState live) {
            double wins = 0;
            for (long world : worlds) {
                Shared shared;
                BotGame.Result r;
                if (opts.fromCopies()) { // the same rollout from a copy of the live game; BotCloneCheck checks it is
                    shared = new Shared(List.of(), player, answer, world, opts.depth(), opts.peek());
                    r = CloneRollout.play(copier, (com.gempukku.lotro.logic.timing.DefaultLotroGame) live.getGame(),
                            prefix, shared, BotVsBot.bot(policy, "~a", ctx), BotVsBot.bot(policy, "~b", ctx));
                } else {
                    shared = new Shared(prefix, player, answer, world, opts.depth(), opts.peek());
                    r = BotGame.play(format, ctx.library(),
                            List.of(new Branching("~a", shared, BotVsBot.bot(policy, "~a", ctx)),
                                    new Branching("~b", shared, BotVsBot.bot(policy, "~b", ctx))),
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
                return shared.answer;
            }
            if (shared.depth > 0 && shared.value < 0 && gameState.getTurnNumber() >= shared.startTurn + shared.depth) {
                // far enough: score the position for the player who branched, and end this throwaway game
                double[] p = new double[2];
                double[] v = new GeneralStateFeatures().extractFeatures(gameState, decision, shared.player);
                if (valueFeatures == 3)
                    v = com.gempukku.lotro.bots.rl.fotrstarters.KnowledgeFeatures.withKnowledge(v, gameState, shared.player);
                valueModel.predict(v, p);
                shared.value = p[1];
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
            String[] actionIds = decision.getDecisionParameters().get("actionId");
            if (decision.getDecisionType() == AwaitingDecisionType.CARD_ACTION_CHOICE && actionIds != null
                    && actionIds.length > 0 && (search || ThreadLocalRandom.current().nextDouble() < opts.teachRate())) {
                String best = teach(gameState, decision, actionIds);
                if (search && best != null) {
                    inner.chooseAction(gameState, decision); // keep the bot's own state (loop guard) in step
                    return best;
                }
            }
            return inner.chooseAction(gameState, decision);
        }

        /** Branches the decision and writes its labels; returns the option that won most, or null on a tie. */
        private String teach(GameState gameState, AwaitingDecision decision, String[] actionIds) {
            List<String> options = new ArrayList<>(Arrays.asList(actionIds));
            Collections.shuffle(options);
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
            StringBuilder lines = new StringBuilder();
            for (int i = 0; i < options.size(); i++) {
                int label = value[i] >= best - 1e-9 ? 1 : value[i] <= best - opts.margin() ? 0 : -1;
                if (label < 0)
                    continue;
                LearningStep step = new LearningStep(state,
                        new CardActionChoiceAction(options.get(i), decision, gameState), fp, decision);
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
        boolean trainOnly = false, play = false, both = false, peek = false, clone = false;
        int minSteps = 50, depth = 0;
        double lambda = 0;
        Path steps = null, value = null;
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--play" -> play = true;
                case "--both" -> both = true;
                case "--peek" -> peek = true;
                case "--clone" -> clone = true;
                case "--model" -> com.gempukku.lotro.bots.rl.fotrstarters.models.AbstractTrainer.useTrees =
                        args[++i].equals("trees");
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
                minSteps, lambda, steps, depth, both, value == null ? base : value, peek, clone);
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
                  --both                    with --play, B searches too: for ranking decks (rr: suites)
                  --clone                   rollouts copy the live game (GameCopier) instead of replaying it""";
    }
}
