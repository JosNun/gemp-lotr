package com.gempukku.lotro.bots.simulation;

import com.gempukku.lotro.logic.vo.LotroDeck;

import java.io.OutputStream;
import java.io.PrintStream;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.*;

/**
 * Did an engine change change any game? Plays a fixed set of seeded games between two deterministic bots and prints
 * one line per game: its seed, decision count, a hash of every question and answer, the winner and the turn. Run it
 * before and after a change and diff the two outputs; a speed-up that only removes work must leave them identical.
 * <pre>
 *   BOT_MAIN=com.gempukku.lotro.bots.simulation.BotTrace ./bot-vs-bot.sh --games 4 > /tmp/before.txt
 * </pre>
 * Seeds come from {@code --seed}, so two runs play the same games. Also prints total engine CPU seconds, which is
 * the other half of the comparison.
 */
public class BotTrace {
    public static void main(String[] args) throws Exception {
        String suite = "starters:test", kind = "general@bot-models-exp/v2";
        int games = 4;
        long master = 1;
        int jobs = Math.max(1, Runtime.getRuntime().availableProcessors() - 2);
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--suite" -> suite = args[++i];
                case "--games" -> games = Integer.parseInt(args[++i]);
                case "--bot" -> kind = args[++i];
                case "--seed" -> master = Long.parseLong(args[++i]);
                case "--jobs" -> jobs = Integer.parseInt(args[++i]);
                default -> BotVsBot.die("usage: BotTrace [--suite starters:test] [--games 4] [--bot general@dir] "
                        + "[--seed 1] [--jobs N]");
            }
        }
        PrintStream out = System.out;
        System.setOut(new PrintStream(OutputStream.nullOutputStream()));

        BotVsBot.Context ctx = BotVsBot.context(Path.of("gemp-lotr-cards/src/main/resources"), Path.of("bot-models"));
        List<BotVsBot.Matchup> matchups = BotVsBot.loadSuite(suite, ctx);
        ExecutorService pool = Executors.newFixedThreadPool(jobs);
        var threads = java.lang.management.ManagementFactory.getThreadMXBean();
        java.util.concurrent.atomic.AtomicLong cpu = new java.util.concurrent.atomic.AtomicLong();
        List<Future<String>> futures = new ArrayList<>();
        String botKind = kind;
        for (BotVsBot.Matchup m : matchups) {
            LotroDeck d1 = BotVsBot.loadDeck(m.deck1(), m.format(), ctx), d2 = BotVsBot.loadDeck(m.deck2(), m.format(), ctx);
            for (int g = 0; g < games; g++) {
                // by matchup, not by position: the suite does not list its matchups in the same order every run
                long seed = new SplittableRandom(master * 1_000_003L + m.label().hashCode() * 101L + g).nextLong();
                futures.add(pool.submit(() -> {
                    long c0 = threads.getCurrentThreadCpuTime();
                    BotGame.Result r = BotGame.play(ctx.formats().getFormat(m.format()), ctx.library(),
                            List.of(BotVsBot.bot(botKind, "~a", ctx), BotVsBot.bot(botKind, "~b", ctx)),
                            Map.of("~a", d1, "~b", d2), 1000, seed);
                    cpu.addAndGet(threads.getCurrentThreadCpuTime() - c0);
                    if (String.valueOf(seed).equals(System.getenv("TRACE_DUMP")))
                        r.steps().forEach(s -> System.err.println(s));
                    long h = 1125899906842597L;
                    for (BotGame.Step s : r.steps())
                        h = 31 * h + (s.player() + "|" + s.question() + "|" + s.answer()).hashCode();
                    return String.format("%s seed %d: %d decisions, hash %016x, winner %s turn %d%s", m.label(), seed,
                            r.steps().size(), h, r.winner(), r.turns(), r.error() == null ? "" : ", " + r.error());
                }));
            }
        }
        for (Future<String> f : futures)
            out.println(f.get(15, TimeUnit.MINUTES));
        out.printf("# %d games, %.1f CPU-seconds%n", futures.size(), cpu.get() / 1e9);
        System.exit(0);
    }
}
