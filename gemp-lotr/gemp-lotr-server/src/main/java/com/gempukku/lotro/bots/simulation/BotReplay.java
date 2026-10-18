package com.gempukku.lotro.bots.simulation;

import com.gempukku.lotro.bots.BotPlayer;
import com.gempukku.lotro.game.state.GameState;
import com.gempukku.lotro.logic.decisions.AwaitingDecision;
import com.gempukku.lotro.logic.vo.LotroDeck;

import java.io.OutputStream;
import java.io.PrintStream;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.*;

/**
 * Is a seeded game replayable? Plays seeded games between two bots, then replays each from the same seed, feeding
 * the engine the recorded answers, and checks it asks the same questions -- type, text, every parameter, card ids
 * included -- in the same order and ends the same way.
 * <p>
 * Lookahead rests on this: branching a game at a decision means replaying it to that point and then answering
 * differently. Anything the engine decides without the seed -- iteration over identity-hashed sets, a stray clock or
 * ThreadLocalRandom -- shows up here as a divergence, with the step where it happened.
 * <pre>
 *   BOT_MAIN=com.gempukku.lotro.bots.simulation.BotReplay ./bot-vs-bot.sh --suite starters --games 10
 * </pre>
 */
public class BotReplay {
    record Check(String label, int steps, int divergedAt, String original, String replayed, boolean sameEnd,
                 boolean otherSeedDiffers) {
    }

    public static void main(String[] args) throws Exception {
        String suite = "starters", kind = "general@bot-models-exp/v1";
        int games = 4;
        int jobs = Math.max(1, Runtime.getRuntime().availableProcessors() - 2);
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--suite" -> suite = args[++i];
                case "--games" -> games = Integer.parseInt(args[++i]);
                case "--bot" -> kind = args[++i];
                case "--jobs" -> jobs = Integer.parseInt(args[++i]);
                default -> BotVsBot.die("usage: BotReplay [--suite starters] [--games 4] [--bot general@dir] [--jobs N]");
            }
        }
        PrintStream out = System.out;
        System.setOut(new PrintStream(OutputStream.nullOutputStream()));

        BotVsBot.Context ctx = BotVsBot.context(Path.of("gemp-lotr-cards/src/main/resources"), Path.of("bot-models"));
        List<BotVsBot.Matchup> matchups = BotVsBot.loadSuite(suite, ctx);
        ExecutorService pool = Executors.newFixedThreadPool(jobs);
        List<Future<Check>> futures = new ArrayList<>();
        String botKind = kind;
        for (BotVsBot.Matchup m : matchups) {
            LotroDeck d1 = BotVsBot.loadDeck(m.deck1(), m.format(), ctx), d2 = BotVsBot.loadDeck(m.deck2(), m.format(), ctx);
            for (int g = 0; g < games; g++) {
                long seed = new Random().nextLong();
                String label = m.label() + " seed " + seed;
                futures.add(pool.submit(() -> {
                    Map<String, LotroDeck> decks = Map.of("~a", d1, "~b", d2);
                    var format = ctx.formats().getFormat(m.format());
                    BotGame.Result original = BotGame.play(format, ctx.library(),
                            List.of(BotVsBot.bot(botKind, "~a", ctx), BotVsBot.bot(botKind, "~b", ctx)), decks, 1000, seed);
                    if (!original.finished())
                        return null;
                    BotGame.Result replay = replay(format, ctx, decks, seed, original.steps());
                    // a well-mixed other seed: java.util.Random's first draws from adjacent seeds are correlated,
                    // and seed + 1 left 5 of 412 games' shuffles unchanged
                    BotGame.Result otherSeed = replay(format, ctx, decks, new java.util.SplittableRandom(seed).nextLong(),
                            original.steps());

                    int at = firstDifference(original.steps(), replay.steps());
                    boolean sameEnd = Objects.equals(original.winner(), replay.winner())
                            && original.turns() == replay.turns() && Objects.equals(original.reason(), replay.reason());
                    boolean otherDiffers = firstDifference(original.steps(), otherSeed.steps()) >= 0;
                    return new Check(label, original.steps().size(), at,
                            at >= 0 && at < original.steps().size() ? original.steps().get(at).toString() : null,
                            at >= 0 && at < replay.steps().size() ? replay.steps().get(at).toString() : null,
                            sameEnd, otherDiffers);
                }));
            }
        }
        List<Check> checks = new ArrayList<>();
        for (Future<Check> f : futures)
            if (f.get() != null)
                checks.add(f.get());
        pool.shutdown();

        long identical = checks.stream().filter(c -> c.divergedAt() < 0 && c.sameEnd()).count();
        long seedMatters = checks.stream().filter(Check::otherSeedDiffers).count();
        out.printf("%d games replayed (%d decisions): %d identical, %d diverged; a different seed changed %d of them%n",
                checks.size(), checks.stream().mapToInt(Check::steps).sum(), identical, checks.size() - identical,
                seedMatters);
        checks.stream().filter(c -> c.divergedAt() >= 0 || !c.sameEnd()).limit(5).forEach(c -> {
            out.printf("  %s: diverged at decision %d of %d%n", c.label(), c.divergedAt(), c.steps());
            out.println("    original: " + abbreviate(c.original()));
            out.println("    replayed: " + abbreviate(c.replayed()));
        });
    }

    /** The same game again, every decision answered from the recording. */
    static BotGame.Result replay(com.gempukku.lotro.game.LotroFormat format, BotVsBot.Context ctx,
                                 Map<String, LotroDeck> decks, long seed, List<BotGame.Step> recorded) {
        Script script = new Script(recorded);
        return BotGame.play(format, ctx.library(), List.of(new Scripted("~a", script), new Scripted("~b", script)),
                decks, 1000, seed);
    }

    static int firstDifference(List<BotGame.Step> a, List<BotGame.Step> b) {
        for (int i = 0; i < Math.min(a.size(), b.size()); i++)
            if (!a.get(i).player().equals(b.get(i).player()) || !a.get(i).question().equals(b.get(i).question()))
                return i;
        return a.size() == b.size() ? -1 : Math.min(a.size(), b.size());
    }

    private static String abbreviate(String s) {
        return s == null ? "(none)" : s.length() > 300 ? s.substring(0, 300) + "..." : s;
    }

    /** The recorded answers, in order, shared by both seats. */
    static final class Script {
        private final List<BotGame.Step> steps;
        private int next;

        Script(List<BotGame.Step> steps) {
            this.steps = steps;
        }

        synchronized String answer(String player) {
            if (next >= steps.size())
                throw new IllegalStateException("replay asked for more decisions than were recorded");
            BotGame.Step step = steps.get(next++);
            if (!step.player().equals(player))
                throw new IllegalStateException("replay asked " + player + ", recording has " + step.player());
            return step.answer();
        }
    }

    record Scripted(String name, Script script) implements BotPlayer {
        @Override
        public String chooseAction(GameState gameState, AwaitingDecision decision) {
            return script.answer(name);
        }

        @Override
        public String getName() {
            return name;
        }
    }
}
