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
    record Row(double[] v1, double[] v2, String decider) {
    }

    record Game(List<Row> rows, String winner) {
    }

    public static void main(String[] args) throws Exception {
        String suite = "starters:test", kind = "general@bot-models-exp/v1";
        Path saveValue = null;
        int games = 20;
        int jobs = Math.max(1, Runtime.getRuntime().availableProcessors() - 2);
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--suite" -> suite = args[++i];
                case "--games" -> games = Integer.parseInt(args[++i]);
                case "--bot" -> kind = args[++i];
                case "--jobs" -> jobs = Integer.parseInt(args[++i]);
                case "--save-value" -> saveValue = Path.of(args[++i]);
                default -> BotVsBot.die("usage: BotProbe [--suite starters:test] [--games 20] [--bot general@dir] "
                        + "[--save-value <model dir>]");
            }
        }
        PrintStream out = System.out;
        System.setOut(new PrintStream(OutputStream.nullOutputStream()));

        BotVsBot.Context ctx = BotVsBot.context(Path.of("gemp-lotr-cards/src/main/resources"), Path.of("bot-models"));
        List<BotVsBot.Matchup> matchups = BotVsBot.loadSuite(suite, ctx);
        ExecutorService pool = Executors.newFixedThreadPool(jobs);
        List<Future<Game>> futures = new ArrayList<>();
        String botKind = kind;
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
            if (f.get() != null)
                all.add(f.get());
        pool.shutdown();
        if (System.getenv("PROBE_DUMP") != null) {
            // a few skirmish decisions from the first game, as the bot saw them
            all.get(0).rows().stream().filter(r -> r.v2()[GeneralStateFeatures.index("inSkirmish")] == 1).limit(3)
                    .forEach(r -> out.println(r.decider() + ": " + GeneralStateFeatures.describe(r.v2())));
        }
        Collections.shuffle(all, new Random(1));
        int split = all.size() * 4 / 5;
        out.printf("%d games, %d decisions; fit on %d games, scored on %d%n", all.size(),
                all.stream().mapToInt(g -> g.rows().size()).sum(), split, all.size() - split);
        List<Game> train = all.subList(0, split), test = all.subList(split, all.size());
        out.printf("  feature set 1, logistic: held-out AUC of P(win | state) = %.3f%n", auc(train, test, 1, false));
        out.printf("  feature set 2, logistic: held-out AUC of P(win | state) = %.3f%n", auc(train, test, 2, false));
        out.printf("  feature set 2, trees:    held-out AUC of P(win | state) = %.3f%n", aucTrees(train, test));

        if (saveValue != null) {
            // the value model a depth-limited search scores its cut-off positions with: set 2, trees, every game
            List<double[]> x = new ArrayList<>();
            List<Integer> y = new ArrayList<>();
            for (Game g : all)
                for (Row r : g.rows()) {
                    x.add(r.v2());
                    y.add(r.decider().equals(g.winner()) ? 1 : 0);
                }
            var model = com.gempukku.lotro.bots.rl.fotrstarters.models.TreeModel.fit(x.toArray(double[][]::new),
                    y.stream().mapToInt(Integer::intValue).toArray());
            java.nio.file.Files.createDirectories(saveValue);
            try (var o = new java.io.ObjectOutputStream(new java.io.FileOutputStream(saveValue.resolve("value.model").toFile()))) {
                o.writeObject(model);
            }
            out.println("value model (set 2, trees, " + x.size() + " positions) in " + saveValue.resolve("value.model"));
        }
    }

    private static double aucTrees(List<Game> train, List<Game> test) {
        List<double[]> x = new ArrayList<>();
        List<Integer> y = new ArrayList<>();
        for (Game g : train)
            for (Row r : g.rows()) {
                x.add(r.v2());
                y.add(r.decider().equals(g.winner()) ? 1 : 0);
            }
        var model = com.gempukku.lotro.bots.rl.fotrstarters.models.TreeModel.fit(x.toArray(double[][]::new),
                y.stream().mapToInt(Integer::intValue).toArray());
        List<double[]> scored = new ArrayList<>();
        for (Game g : test)
            for (Row r : g.rows()) {
                double[] p = new double[2];
                model.predict(r.v2(), p);
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
                x.add(set == 1 ? r.v1() : r.v2());
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
                model.predict(z.apply(set == 1 ? r.v1() : r.v2()), p);
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
                    rows.add(new Row(v1, V2.extractFeatures(gameState, decision, getName()), getName()));
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
