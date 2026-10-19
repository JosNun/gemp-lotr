package com.gempukku.lotro.bots.simulation;

import com.gempukku.lotro.bots.BotPlayer;
import com.gempukku.lotro.bots.rl.fotrstarters.FotrStartersRLGameStateFeatures;
import com.gempukku.lotro.bots.rl.fotrstarters.GeneralStateFeatures;
import com.gempukku.lotro.game.state.GameState;
import com.gempukku.lotro.logic.decisions.AwaitingDecision;
import com.gempukku.lotro.logic.decisions.AwaitingDecisionType;
import com.gempukku.lotro.logic.vo.LotroDeck;
import smile.classification.LogisticRegression;
import smile.classification.SoftClassifier;

import java.io.OutputStream;
import java.io.PrintStream;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.*;

/**
 * Do the feature sets carry information about who wins? Plays games between two bots, records feature set 1 and set 2
 * of the deciding player's state at every card-action decision, labels each row with whether that player went on to
 * win, fits P(win | state) on 80% of the games and reports the AUC on the other 20%.
 * <p>
 * This is the question underneath "set 2 tied set 1": if set 2 predicts the result much better, the information is
 * in it and the per-decision label is what fails to use it; if not, the features are the problem.
 * <pre>
 *   ./bot-vs-bot.sh ... (BOT_MAIN=com.gempukku.lotro.bots.simulation.BotProbe) --suite starters:test --games 20
 * </pre>
 */
public class BotProbe {
    record Row(double[] v1, double[] v2, double[] v3, String decider) {
        double[] set(int n) {
            return n == 1 ? v1 : n == 2 ? v2 : n == 3 ? v3
                    : com.gempukku.lotro.bots.rl.fotrstarters.CorruptionFeatures.withCorruption(v2, null, decider);
        }
    }

    record Game(List<Row> rows, String winner) {
    }

    public static void main(String[] args) throws Exception {
        String suite = "starters:test", kind = "general@bot-models-exp/v1";
        Path saveValue = null;
        int valueSet = 2;
        String trainSuite = null;
        boolean valueTrees = false;
        Path fit = null;
        int games = 20;
        int jobs = Math.max(1, Runtime.getRuntime().availableProcessors() - 2);
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--suite" -> suite = args[++i];
                case "--games" -> games = Integer.parseInt(args[++i]);
                case "--bot" -> kind = args[++i];
                case "--jobs" -> jobs = Integer.parseInt(args[++i]);
                case "--save-value" -> saveValue = Path.of(args[++i]);
                case "--value-features" -> valueSet = Integer.parseInt(args[++i]);
                case "--train-suite" -> trainSuite = args[++i];
                case "--value-model" -> valueTrees = args[++i].equals("trees");
                case "--fit" -> fit = Path.of(args[++i]);
                default -> BotVsBot.die("usage: BotProbe [--suite starters:test] [--games 20] [--bot general@dir] "
                        + "[--train-suite <suite>] [--save-value <model dir>] [--value-features 2|3] [--value-model logistic|trees]");
            }
        }
        PrintStream out = System.out;
        System.setOut(new PrintStream(OutputStream.nullOutputStream()));

        if (fit != null) {
            // positions recorded from other games (BotTeach --record-values): label, then the set-2 vector
            if (saveValue == null)
                BotVsBot.die("--fit needs --save-value <dir>");
            List<double[]> x = new ArrayList<>();
            List<Integer> y = new ArrayList<>();
            for (String line : java.nio.file.Files.readAllLines(fit)) {
                if (line.isBlank())
                    continue;
                String[] parts = line.split(",");
                double[] v = new double[parts.length - 1];
                for (int i = 1; i < parts.length; i++)
                    v[i - 1] = Double.parseDouble(parts[i]);
                x.add(v);
                y.add(Integer.parseInt(parts[0]));
            }
            double[][] xs = x.toArray(double[][]::new);
            int[] ys = y.stream().mapToInt(Integer::intValue).toArray();
            Object model = valueTrees ? com.gempukku.lotro.bots.rl.fotrstarters.models.TreeModel.fit(xs, ys)
                    : com.gempukku.lotro.bots.rl.fotrstarters.models.ScaledModel.logistic(xs, ys, 0.1);
            java.nio.file.Files.createDirectories(saveValue);
            try (var o = new java.io.ObjectOutputStream(new java.io.FileOutputStream(saveValue.resolve("value.model").toFile()))) {
                o.writeObject(model);
            }
            java.nio.file.Files.writeString(saveValue.resolve("value-features.txt"), "2\n");
            out.println("value model (set 2, " + (valueTrees ? "trees" : "logistic") + ", " + xs.length
                    + " positions from " + fit + ") in " + saveValue.resolve("value.model"));
            System.exit(0);
        }
        BotVsBot.Context ctx = BotVsBot.context(Path.of("gemp-lotr-cards/src/main/resources"), Path.of("bot-models"));
        ExecutorService pool = Executors.newFixedThreadPool(jobs);
        List<Game> all = play(suite, games, kind, ctx, pool);
        // decks the value model never saw: what scoring a new deck's games needs. Split by game instead, the
        // features that describe a deck's remaining cards let a model learn each deck's win rate and look better
        // than it is.
        List<Game> other = trainSuite == null ? null : play(trainSuite, games, kind, ctx, pool);
        pool.shutdown();
        if (System.getenv("PROBE_DUMP") != null) {
            // a few skirmish decisions from the first game, as the bot saw them
            all.get(0).rows().stream().filter(r -> r.v2()[GeneralStateFeatures.index("inSkirmish")] == 1).limit(3)
                    .forEach(r -> out.println(r.decider() + ": " + GeneralStateFeatures.describe(r.v2())));
        }
        Collections.shuffle(all, new Random(1));
        int split = all.size() * 4 / 5;
        List<Game> train = other != null ? other : all.subList(0, split), test = other != null ? all : all.subList(split, all.size());
        out.printf("%d games, %d decisions; fit on %d games%s, scored on %d%n", train.size() + test.size(),
                train.stream().mapToInt(g -> g.rows().size()).sum() + test.stream().mapToInt(g -> g.rows().size()).sum(),
                train.size(), other != null ? " of " + trainSuite : "", test.size());
        out.printf("  feature set 1, logistic: held-out AUC of P(win | state) = %.3f%n", auc(train, test, 1, false));
        out.printf("  feature set 2, logistic: held-out AUC of P(win | state) = %.3f%n", auc(train, test, 2, true));
        out.printf("  feature set 2, trees:    held-out AUC of P(win | state) = %.3f%n", aucTrees(train, test, 2));
        out.printf("  feature set 3, logistic: held-out AUC of P(win | state) = %.3f%n", auc(train, test, 3, true));
        out.printf("  feature set 3, trees:    held-out AUC of P(win | state) = %.3f%n", aucTrees(train, test, 3));
        out.printf("  feature set 4, logistic: held-out AUC of P(win | state) = %.3f%n", auc(train, test, 4, true));

        if (saveValue != null) {
            // the value model a depth-limited search scores its cut-off positions with, fit on every game of the
            // training suite (or of --suite, without one)
            List<double[]> x = new ArrayList<>();
            List<Integer> y = new ArrayList<>();
            for (Game g : other != null ? other : all)
                for (Row r : g.rows()) {
                    x.add(r.set(valueSet));
                    y.add(r.decider().equals(g.winner()) ? 1 : 0);
                }
            double[][] xs = x.toArray(double[][]::new);
            int[] ys = y.stream().mapToInt(Integer::intValue).toArray();
            Object model = valueTrees ? com.gempukku.lotro.bots.rl.fotrstarters.models.TreeModel.fit(xs, ys)
                    : com.gempukku.lotro.bots.rl.fotrstarters.models.ScaledModel.logistic(xs, ys, 0.1);
            java.nio.file.Files.createDirectories(saveValue);
            try (var o = new java.io.ObjectOutputStream(new java.io.FileOutputStream(saveValue.resolve("value.model").toFile()))) {
                o.writeObject(model);
            }
            java.nio.file.Files.writeString(saveValue.resolve("value-features.txt"), valueSet + "\n");
            out.println("value model (set " + valueSet + (valueTrees ? ", trees, " : ", logistic, ") + x.size() + " positions) in " + saveValue.resolve("value.model"));
        }
    }

    private static List<Game> play(String suite, int games, String botKind, BotVsBot.Context ctx,
                                   ExecutorService pool) throws Exception {
        List<BotVsBot.Matchup> matchups = BotVsBot.loadSuite(suite, ctx);
        List<Future<Game>> futures = new ArrayList<>();
        for (BotVsBot.Matchup m : matchups) {
            LotroDeck d1 = BotVsBot.loadDeck(m.deck1(), m.format(), ctx), d2 = BotVsBot.loadDeck(m.deck2(), m.format(), ctx);
            for (int g = 0; g < games; g++) {
                boolean swap = g % 2 == 1;
                futures.add(pool.submit(() -> {
                    List<Row> rows = new ArrayList<>();
                    BotPlayer a = new Recorder(BotVsBot.bot(botKind, "~a", ctx), rows);
                    BotPlayer b = new Recorder(BotVsBot.bot(botKind, "~b", ctx), rows);
                    Map<String, LotroDeck> decks = Map.of("~a", swap ? d2 : d1, "~b", swap ? d1 : d2);
                    BotGame.Result r = BotGame.play(ctx.formats().getFormat(m.format()), ctx.library(), List.of(a, b),
                            decks, 1000);
                    return r.finished() ? new Game(rows, r.winner()) : null;
                }));
            }
        }
        List<Game> all = new ArrayList<>();
        for (Future<Game> f : futures)
            if (f.get(15, TimeUnit.MINUTES) != null)
                all.add(f.get());
        return all;
    }

    private static double aucTrees(List<Game> train, List<Game> test, int set) {
        List<double[]> x = new ArrayList<>();
        List<Integer> y = new ArrayList<>();
        for (Game g : train)
            for (Row r : g.rows()) {
                x.add(r.set(set));
                y.add(r.decider().equals(g.winner()) ? 1 : 0);
            }
        var model = com.gempukku.lotro.bots.rl.fotrstarters.models.TreeModel.fit(x.toArray(double[][]::new),
                y.stream().mapToInt(Integer::intValue).toArray());
        List<double[]> scored = new ArrayList<>();
        for (Game g : test)
            for (Row r : g.rows()) {
                double[] p = new double[2];
                model.predict(r.set(set), p);
                scored.add(new double[]{p[1], r.decider().equals(g.winner()) ? 1 : 0});
            }
        return rankAuc(scored);
    }

    private static double rankAuc(List<double[]> scored) {
        scored.sort(Comparator.comparingDouble(s -> s[0]));
        double rankSum = 0;
        long pos = 0;
        for (int i = 0; i < scored.size(); i++)
            if (scored.get(i)[1] == 1) {
                rankSum += i + 1;
                pos++;
            }
        long neg = scored.size() - pos;
        return (rankSum - pos * (pos + 1) / 2.0) / ((double) pos * neg);
    }

    private static double auc(List<Game> train, List<Game> test, int set, boolean scaled) {
        List<double[]> x = new ArrayList<>();
        List<Integer> y = new ArrayList<>();
        for (Game g : train)
            for (Row r : g.rows()) {
                x.add(r.set(set));
                y.add(r.decider().equals(g.winner()) ? 1 : 0);
            }
        double[][] xs = x.toArray(double[][]::new);
        int dims = xs[0].length;
        double[] mean = new double[dims], sd = new double[dims];
        Arrays.fill(sd, 1);
        if (scaled) {
            for (double[] row : xs)
                for (int d = 0; d < dims; d++)
                    mean[d] += row[d] / xs.length;
            double[] var = new double[dims];
            for (double[] row : xs)
                for (int d = 0; d < dims; d++)
                    var[d] += (row[d] - mean[d]) * (row[d] - mean[d]) / xs.length;
            for (int d = 0; d < dims; d++)
                sd[d] = var[d] > 1e-12 ? Math.sqrt(var[d]) : 1;
        }
        java.util.function.UnaryOperator<double[]> z = row -> {
            double[] r = new double[dims];
            for (int d = 0; d < dims; d++)
                r[d] = (row[d] - mean[d]) / sd[d];
            return r;
        };
        SoftClassifier<double[]> model = LogisticRegression.fit(Arrays.stream(xs).map(z).toArray(double[][]::new),
                y.stream().mapToInt(Integer::intValue).toArray(), 0.1, 1e-5, 500);
        List<double[]> scored = new ArrayList<>(); // {score, label}
        for (Game g : test)
            for (Row r : g.rows()) {
                double[] p = new double[2];
                model.predict(z.apply(r.set(set)), p);
                scored.add(new double[]{p[1], r.decider().equals(g.winner()) ? 1 : 0});
            }
        // AUC as the chance a random winning row outscores a random losing one (rank-sum)
        scored.sort(Comparator.comparingDouble(s -> s[0]));
        double rankSum = 0;
        long pos = 0;
        for (int i = 0; i < scored.size(); i++)
            if (scored.get(i)[1] == 1) {
                rankSum += i + 1;
                pos++;
            }
        long neg = scored.size() - pos;
        return (rankSum - pos * (pos + 1) / 2.0) / ((double) pos * neg);
    }

    /** Passes decisions through, noting both feature sets for card-action decisions first. */
    private record Recorder(BotPlayer inner, List<Row> rows) implements BotPlayer {
        private static final FotrStartersRLGameStateFeatures V1 = new FotrStartersRLGameStateFeatures();
        private static final GeneralStateFeatures V2 = new GeneralStateFeatures();

        @Override
        public String chooseAction(GameState gameState, AwaitingDecision decision) {
            if (decision.getDecisionType() == AwaitingDecisionType.CARD_ACTION_CHOICE) {
                try {
                    // set 1 is 26 numbers, or 30 in skirmishes and responses; pad to one length
                    double[] v1 = Arrays.copyOf(V1.extractFeatures(gameState, decision, getName()), 30);
                    double[] v2 = V2.extractFeatures(gameState, decision, getName());
                    rows.add(new Row(v1, v2, com.gempukku.lotro.bots.rl.fotrstarters.KnowledgeFeatures.withKnowledge(v2,
                            gameState, getName()), getName()));
                } catch (RuntimeException ignored) {
                }
            }
            return inner.chooseAction(gameState, decision);
        }

        @Override
        public String getName() {
            return inner.getName();
        }
    }
}
