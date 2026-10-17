package com.gempukku.lotro.bots.simulation;

import com.gempukku.lotro.bots.BotPlayer;
import com.gempukku.lotro.bots.BotService;
import com.gempukku.lotro.bots.random.RandomLearningBot;
import com.gempukku.lotro.bots.rl.LearningBotPlayer;
import com.gempukku.lotro.bots.rl.LearningStep;
import com.gempukku.lotro.bots.rl.ReplayBuffer;
import com.gempukku.lotro.bots.rl.fotrstarters.FotrStarterBot;
import com.gempukku.lotro.bots.rl.fotrstarters.FotrStartersRLGameStateFeatures;
import com.gempukku.lotro.bots.rl.RLGameStateFeatures;
import com.gempukku.lotro.bots.rl.fotrstarters.CardFeatures;
import com.gempukku.lotro.bots.rl.fotrstarters.GeneralStateFeatures;
import com.gempukku.lotro.bots.rl.fotrstarters.models.ConstantModel;
import com.gempukku.lotro.bots.rl.fotrstarters.models.ModelRegistry;
import com.gempukku.lotro.bots.rl.fotrstarters.models.Trainer;
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
 * Self-play training from the command line, into a model directory of its own -- the server's own loop
 * ({@code BotService.START_SIMULATIONS_AT_STARTUP}) only plays the two FotR starters and overwrites bot-models/.
 * <p>
 * Each generation plays --games games over the matchups of a suite with the current models, both bots learning,
 * labels every decision in a game with its result (won or lost, as the server's loop does), then refits every model
 * on everything gathered so far and writes it to --out. A share of games (--vs-random) put a random bot on one side,
 * and --epsilon makes the model bots answer at random now and then, so the data holds moves the models would not
 * choose themselves.
 * <p>
 * The trained models are scored with BotVsBot, e.g. {@code --a fotr@bot-models-new --b fotr --suite starters:test}.
 * <pre>
 *   ./bot-train.sh --suite starters:train --out bot-models-all --generations 3 --games 3000
 *   ./bot-train.sh --suite starters --init random --out bot-models-scratch
 * </pre>
 */
public class BotTrain {
    private static final String BOT_A = "~a";
    private static final String BOT_B = "~b";

    record Options(String suite, Path init, boolean fromRandom, Path out, int generations, int games, double epsilon,
                   double vsRandom, int jobs, int maxDecisions, Path resources, boolean extended, int features) {
    }

    public static void main(String[] args) throws Exception {
        Options opts = parse(args);
        // Card loading and the bots' "Unknown ..." lines all go to System.out; none of it is wanted here.
        PrintStream out = System.out;
        System.setOut(new PrintStream(OutputStream.nullOutputStream(), true, StandardCharsets.UTF_8));

        BotVsBot.Context ctx = BotVsBot.context(opts.resources(), opts.init());
        List<BotVsBot.Matchup> matchups = BotVsBot.loadSuite(opts.suite(), ctx);
        Map<String, LotroDeck> decks = new HashMap<>();
        for (BotVsBot.Matchup m : matchups) {
            if (ctx.formats().getFormat(m.format()) == null)
                BotVsBot.die("Unknown format \"" + m.format() + "\"");
            for (String spec : List.of(m.deck1(), m.deck2()))
                decks.computeIfAbsent(m.format() + "|" + spec, k -> BotVsBot.loadDeck(spec, m.format(), ctx));
        }

        List<Trainer> trainers = opts.extended() ? BotService.extendedTrainers() : BotService.allTrainers();
        ModelRegistry models = opts.fromRandom() ? null : BotVsBot.loadModels(opts.init());
        // Starting from models of another feature set: they play generation 0, recording the new set's features,
        // and none of them can stand in for a model of the new set.
        boolean sameFeatures = models != null && models.getFeatureVersion() == opts.features();
        Files.createDirectories(opts.out());
        Files.writeString(opts.out().resolve("features.txt"), opts.features() + "\n");
        if (sameFeatures) // so --out is a complete model set even if some model never gets data
            for (Trainer t : trainers)
                if (models.hasModel(t.getClass()))
                    save(opts.out(), t, models.getModel(t.getClass()));
        CardFeatures.setVersion(opts.features()); // training happens on this thread
        RLGameStateFeatures recordFeatures = opts.features() == GeneralStateFeatures.VERSION
                ? new GeneralStateFeatures() : new FotrStartersRLGameStateFeatures();

        // Every relevant step gathered so far, by trainer. A step can serve several trainers.
        Map<Class<?>, List<LearningStep>> data = new ConcurrentHashMap<>();
        for (Trainer t : trainers)
            data.put(t.getClass(), Collections.synchronizedList(new ArrayList<>()));

        out.printf("%d matchups from %s; %d generations x %d games on %d threads; models from %s into %s%n",
                matchups.size(), opts.suite(), opts.generations(), opts.games(), opts.jobs(),
                opts.fromRandom() ? "random play" : opts.init(), opts.out());

        ExecutorService pool = Executors.newFixedThreadPool(opts.jobs());
        for (int gen = 0; gen < opts.generations(); gen++) {
            long t0 = System.nanoTime();
            ModelRegistry current = models;
            boolean randomOnly = current == null;
            boolean keepOld = gen > 0 || sameFeatures;
            AtomicInteger aWins = new AtomicInteger(), finished = new AtomicInteger(), dropped = new AtomicInteger();
            List<Future<?>> futures = new ArrayList<>();
            for (int g = 0; g < opts.games(); g++) {
                int game = g;
                futures.add(pool.submit(() -> {
                    BotVsBot.Matchup m = matchups.get(game % matchups.size());
                    boolean deckSwap = (game / matchups.size()) % 2 == 1;
                    // a share of games against a random bot, as the server's loop does, on alternating sides
                    boolean randomB = !randomOnly && ThreadLocalRandom.current().nextDouble() < opts.vsRandom();
                    Collector ca = new Collector(), cb = new Collector();
                    LearningBotPlayer a = randomOnly ? new RandomLearningBot(recordFeatures, BOT_A, ca)
                            : new FotrStarterBot(new FotrStartersRLGameStateFeatures(), BOT_A, current, ca, opts.epsilon(), opts.extended())
                            .recordWith(recordFeatures);
                    LearningBotPlayer b = randomOnly || randomB
                            ? new RandomLearningBot(recordFeatures, BOT_B, cb)
                            : new FotrStarterBot(new FotrStartersRLGameStateFeatures(), BOT_B, current, cb, opts.epsilon(), opts.extended())
                            .recordWith(recordFeatures);
                    Map<String, LotroDeck> seats = new HashMap<>();
                    seats.put(BOT_A, decks.get(m.format() + "|" + (deckSwap ? m.deck2() : m.deck1())));
                    seats.put(BOT_B, decks.get(m.format() + "|" + (deckSwap ? m.deck1() : m.deck2())));

                    BotGame.Result r = BotGame.play(ctx.formats().getFormat(m.format()), ctx.library(),
                            List.<BotPlayer>of(a, b), seats, opts.maxDecisions());
                    if (!r.finished())
                        return; // no result to label the decisions with
                    finished.incrementAndGet();
                    if (r.winner().equals(BOT_A))
                        aWins.incrementAndGet();
                    // the same reward scale as SimpleBatchSimulationRunner: trainers read reward > 0 as a win
                    a.endEpisode(r.winner().equals(BOT_A) ? 10.0 : 0.0);
                    b.endEpisode(r.winner().equals(BOT_B) ? 10.0 : 0.0);
                    for (Collector c : List.of(ca, cb))
                        for (LearningStep step : c.steps) {
                            LearningStep detached = detach(step);
                            if (detached == null) {
                                dropped.incrementAndGet();
                                continue;
                            }
                            for (Trainer t : trainers)
                                if (t.isStepRelevant(detached))
                                    data.get(t.getClass()).add(detached);
                        }
                }));
            }
            for (Future<?> f : futures)
                f.get();
            double playSeconds = (System.nanoTime() - t0) / 1e9;

            ModelRegistry next = new ModelRegistry();
            next.setFeatureVersion(opts.features());
            List<String> kept = new ArrayList<>();
            int totalSteps = 0;
            for (Trainer t : trainers) {
                List<LearningStep> steps;
                synchronized (data.get(t.getClass())) {
                    steps = new ArrayList<>(data.get(t.getClass()));
                }
                totalSteps += steps.size();
                SoftClassifier<double[]> model = null;
                try {
                    if (!steps.isEmpty())
                        model = t.train(steps);
                } catch (RuntimeException e) {
                    // e.g. every example labelled the same: logistic regression needs both classes
                }
                if (model == null) {
                    model = current == null || !keepOld ? null : current.getModel(t.getClass());
                    if (model == null)
                        model = new ConstantModel(0);
                    kept.add(t.getClass().getSimpleName() + " (" + steps.size() + " steps)");
                }
                if (model != null) {
                    next.registerModel(t.getClass(), model);
                    save(opts.out(), t, model);
                }
            }
            models = next;

            out.printf("generation %d: %d of %d games finished in %.0fs (A won %d), %d steps held, trained in %.0fs%n",
                    gen, finished.get(), opts.games(), playSeconds, aWins.get(), totalSteps,
                    (System.nanoTime() - t0) / 1e9 - playSeconds);
            if (dropped.get() > 0)
                out.printf("  %d steps could not be stored (no JSON form for their action)%n", dropped.get());
            if (!kept.isEmpty())
                out.println("  not refit, previous model kept: " + String.join(", ", kept));
            if (!BotGame.BOT_ERRORS.isEmpty()) {
                out.println("  bots threw while choosing (the fallback answered):");
                BotGame.BOT_ERRORS.forEach((k, n) -> out.printf("    %5d  %s%n", n, k));
                BotGame.BOT_ERRORS.clear();
            }
        }
        pool.shutdown();
        out.println("models in " + opts.out());
    }

    /**
     * A step through JSON and back. The live one holds the engine's decision object, which can keep its whole game
     * reachable; held for thousands of games that is the heap. Returns null for actions with no JSON form.
     */
    private static LearningStep detach(LearningStep step) {
        try {
            return LearningStep.fromJson(step.toJson().toString());
        } catch (RuntimeException e) {
            return null;
        }
    }

    private static void save(Path dir, Trainer trainer, SoftClassifier<double[]> model) throws IOException {
        if (model == null)
            return;
        Path file = dir.resolve(trainer.getClass().getSimpleName() + ".model");
        try (ObjectOutputStream o = new ObjectOutputStream(new FileOutputStream(file.toFile()))) {
            o.writeObject(model);
        }
    }

    /** A replay buffer that keeps every episode it is given, in order, for one game. */
    private static final class Collector extends ReplayBuffer {
        final List<LearningStep> steps = new ArrayList<>();

        Collector() {
            super(Integer.MAX_VALUE);
        }

        @Override
        public void addEpisode(List<LearningStep> episode) {
            if (episode != null)
                steps.addAll(episode);
        }
    }

    private static Options parse(String[] args) {
        String suite = "starters:train";
        String init = "bot-models";
        Path out = null;
        int generations = 3;
        int games = 2000;
        double epsilon = 0.05;
        double vsRandom = 0.2;
        int jobs = Math.max(1, Runtime.getRuntime().availableProcessors() - 2);
        int maxDecisions = 1000;
        Path resources = Path.of("gemp-lotr-cards/src/main/resources");
        boolean extended = false;
        int features = 1;
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--features" -> {
                    features = Integer.parseInt(args[++i]);
                    if (features != 1 && features != GeneralStateFeatures.VERSION)
                        BotVsBot.die("--features is 1 or " + GeneralStateFeatures.VERSION);
                }
                case "--bot" -> {
                    String bot = args[++i];
                    if (!List.of("fotr", "general").contains(bot))
                        BotVsBot.die("--bot is fotr or general, not " + bot);
                    extended = bot.equals("general");
                }
                case "--model" -> {
                    String model = args[++i];
                    if (!List.of("logistic", "trees").contains(model))
                        BotVsBot.die("--model is logistic or trees, not " + model);
                    com.gempukku.lotro.bots.rl.fotrstarters.models.AbstractTrainer.useTrees = model.equals("trees");
                }
                case "--suite" -> suite = args[++i];
                case "--init" -> init = args[++i];
                case "--out" -> out = Path.of(args[++i]);
                case "--generations" -> generations = Integer.parseInt(args[++i]);
                case "--games" -> games = Integer.parseInt(args[++i]);
                case "--epsilon" -> epsilon = Double.parseDouble(args[++i]);
                case "--vs-random" -> vsRandom = Double.parseDouble(args[++i]);
                case "--jobs" -> jobs = Integer.parseInt(args[++i]);
                case "--max-decisions" -> maxDecisions = Integer.parseInt(args[++i]);
                case "--resources" -> resources = Path.of(args[++i]);
                default -> BotVsBot.die(usage());
            }
        }
        if (out == null)
            BotVsBot.die(usage());
        if (out.toAbsolutePath().normalize().equals(Path.of("bot-models").toAbsolutePath().normalize()))
            BotVsBot.die("--out bot-models would overwrite the shipped models; pick another directory");
        boolean fromRandom = init.equals("random");
        return new Options(suite, fromRandom ? null : Path.of(init), fromRandom, out, generations, games, epsilon,
                vsRandom, jobs, maxDecisions, resources, extended, features);
    }

    private static String usage() {
        return """
                usage: BotTrain --out <model dir> [options]
                  --bot fotr              fotr, or general for the extended bot and its extra models
                  --model logistic        logistic, or trees (gradient-boosted; can weigh state against cards)
                  --features 1            feature set: 1 (the original), or 2 (GeneralStateFeatures, any deck)
                  --suite starters:train  matchups to self-play (as BotVsBot: starters[:train|test|all] or a file)
                  --init bot-models       models to start from, or "random" to start from random play
                  --generations 3         play, refit, repeat
                  --games 2000            per generation
                  --epsilon 0.05          chance a model bot answers at random
                  --vs-random 0.2         share of games against a random bot
                  --jobs N                threads (default cores - 2)""";
    }
}
