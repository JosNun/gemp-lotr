package com.gempukku.lotro.bots.simulation;

import com.gempukku.lotro.bots.BotPlayer;
import com.gempukku.lotro.bots.random.RandomDecisionBot;
import com.gempukku.lotro.game.LotroFormat;
import com.gempukku.lotro.game.state.GameState;
import com.gempukku.lotro.logic.decisions.AwaitingDecision;
import com.gempukku.lotro.logic.decisions.AwaitingDecisionType;
import com.gempukku.lotro.logic.timing.DefaultLotroGame;
import com.gempukku.lotro.logic.vo.LotroDeck;
import smile.classification.SoftClassifier;

import java.io.FileInputStream;
import java.io.ObjectInputStream;
import java.io.OutputStream;
import java.io.PrintStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Is a copied game the game? (GameCopier, for BotTeach --clone.) Plays seeded games between two bots, then replays
 * each from its seed and stops at a random decision k, where it checks three things:
 * <ul>
 * <li>(b) the original is not touched: a copy is made and played out to the end by random bots on a reseeded
 *     engine, before anything else -- and the original, replayed on from k afterwards, must still ask every recorded
 *     question and end the same way;</li>
 * <li>(c) a rollout from a copy is the replay rollout: for {@code --worlds} world seeds, BotTeach's rollout (replay to
 *     k, reseed, reshuffle, give an option, play on with the policy bots) and the same from a copy must give the same
 *     steps and the same result (and the same value-model score, with {@code --depth});</li>
 * <li>(a) a copy continued with the recorded answers asks the recorded questions and ends as the original did.</li>
 * </ul>
 * Anything that differs means the copy shares or misses state. Also times copying against replaying to k.
 * <pre>
 *   BOT_MAIN=com.gempukku.lotro.bots.simulation.BotCloneCheck ./bot-vs-bot.sh --suite starters:test --games 4
 * </pre>
 */
public class BotCloneCheck {
    private static final AtomicInteger games = new AtomicInteger(), aSame = new AtomicInteger(),
            bSame = new AtomicInteger(), cSame = new AtomicInteger(), cRuns = new AtomicInteger(),
            skipped = new AtomicInteger();
    private static final AtomicLong copyNanos = new AtomicLong(), replayNanos = new AtomicLong(),
            timedRollouts = new AtomicLong(), decisionsChecked = new AtomicLong();
    private static final List<String> failures = Collections.synchronizedList(new ArrayList<>());

    private static final java.lang.management.ThreadMXBean THREADS =
            java.lang.management.ManagementFactory.getThreadMXBean();

    /** This thread's CPU time: wall time means little on a loaded machine. */
    private static long cpu() {
        return THREADS.getCurrentThreadCpuTime();
    }

    public static void main(String[] args) throws Exception {
        String suite = "starters:test", policy = "general@bot-models-exp/v2";
        int perMatchup = 2, worlds = 2, depth = 0;
        long master = 1;
        int jobs = 2;
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--suite" -> suite = args[++i];
                case "--games" -> perMatchup = Integer.parseInt(args[++i]);
                case "--bot" -> policy = args[++i];
                case "--worlds" -> worlds = Integer.parseInt(args[++i]);
                case "--depth" -> depth = Integer.parseInt(args[++i]);
                case "--seed" -> master = Long.parseLong(args[++i]);
                case "--jobs" -> jobs = Integer.parseInt(args[++i]);
                default -> BotVsBot.die("usage: BotCloneCheck [--suite starters:test] [--games 2 (per matchup)] "
                        + "[--bot general@dir] [--worlds 2] [--depth 0] [--seed 1] [--jobs 2]");
            }
        }
        PrintStream out = System.out;
        System.setOut(new PrintStream(OutputStream.nullOutputStream()));
        try {
            run(out, suite, policy, perMatchup, worlds, depth, master, jobs);
        } finally {
            System.exit(0);
        }
    }

    private static void run(PrintStream out, String suite, String policy, int perMatchup, int worlds, int depth,
                            long master, int jobs) throws Exception {
        BotVsBot.Context ctx = BotVsBot.context(Path.of("gemp-lotr-cards/src/main/resources"), Path.of("bot-models"));
        List<BotVsBot.Matchup> matchups = BotVsBot.loadSuite(suite, ctx);
        Map<String, LotroDeck> decks = new HashMap<>();
        for (BotVsBot.Matchup m : matchups)
            for (String spec : List.of(m.deck1(), m.deck2()))
                decks.computeIfAbsent(m.format() + "|" + spec, k -> BotVsBot.loadDeck(spec, m.format(), ctx));
        if (depth > 0) {
            Path file = Path.of(policy.substring(policy.indexOf('@') + 1)).resolve("value.model");
            if (!Files.exists(file))
                BotVsBot.die("--depth needs " + file);
            try (ObjectInputStream in = new ObjectInputStream(new FileInputStream(file.toFile()))) {
                @SuppressWarnings("unchecked")
                SoftClassifier<double[]> model = (SoftClassifier<double[]>) in.readObject();
                BotTeach.valueModel = model;
            }
        }
        long t0 = System.nanoTime();
        GameCopier copier = new GameCopier(ctx.library(), ctx.formats(), decks);
        out.printf("Shared with every copy: %,d objects (library, formats, decks, engine statics), found in %.1fs%n",
                copier.sharedCount(), (System.nanoTime() - t0) / 1e9);

        if (System.getenv("CLONE_STATS") != null)
            copier.histogram = new ConcurrentHashMap<>();
        ExecutorService pool = Executors.newFixedThreadPool(jobs);
        List<Future<?>> futures = new ArrayList<>();
        String botKind = policy;
        int depthF = depth, worldsF = worlds;
        for (BotVsBot.Matchup m : matchups) {
            Map<String, LotroDeck> seats = Map.of("~a", decks.get(m.format() + "|" + m.deck1()),
                    "~b", decks.get(m.format() + "|" + m.deck2()));
            LotroFormat format = ctx.formats().getFormat(m.format());
            for (int g = 0; g < perMatchup; g++) {
                long seed = new SplittableRandom(master * 1_000_003L + m.label().hashCode() * 101L + g).nextLong();
                String label = m.label() + " seed " + seed;
                futures.add(pool.submit(() -> {
                    try {
                        checkGame(label, format, ctx, seats, seed, botKind, copier, worldsF, depthF);
                    } catch (Throwable e) {
                        failures.add(label + ": " + e);
                        e.printStackTrace(System.err);
                    }
                    int n = games.get();
                    if (n > 0 && n % 20 == 0)
                        System.err.printf("  %d games checked%n", n);
                }));
            }
        }
        for (Future<?> f : futures)
            f.get();
        pool.shutdown();

        int n = games.get();
        out.printf("%d games checked at a random decision (%d skipped: unfinished originals)%n", n, skipped.get());
        out.printf("  (a) copy continued with the recorded answers: %d identical, %d diverged%n", aSame.get(),
                n - aSame.get());
        out.printf("  (b) original replayed on after a copy was played out differently: %d identical, %d diverged%n",
                bSame.get(), n - bSame.get());
        out.printf("  (c) rollout from a copy vs replay rollout (%d worlds a game%s): %d identical, %d diverged%n",
                worldsF, depthF > 0 ? ", depth " + depthF : "", cSame.get(), cRuns.get() - cSame.get());
        if (timedRollouts.get() > 0)
            out.printf("  per rollout, thread CPU: copy %.0f us, replay to the branch %.0f us (mean over %d, at decision %.0f "
                            + "on average)%n", copyNanos.get() / 1e3 / timedRollouts.get(),
                    replayNanos.get() / 1e3 / timedRollouts.get(), timedRollouts.get(),
                    (double) decisionsChecked.get() / Math.max(1, n));
        out.printf("  shared objects now %,d; %d engine classes not yet initialised%n", copier.sharedCount(),
                copier.classesPending());
        if (copier.histogram != null) {
            long copiesMade = CloneRollout.copies.get() + 3L * n + timedRollouts.get();
            out.printf("  objects copied per copy, by class (%d copies):%n", copiesMade);
            copier.histogram.entrySet().stream().sorted(Map.Entry.comparingByValue(Comparator.reverseOrder()))
                    .limit(40).forEach(e -> out.printf("    %8.1f  %s%n", e.getValue() / (double) copiesMade,
                            e.getKey()));
            out.printf("    %8.1f  total%n", copier.histogram.values().stream().mapToLong(Integer::longValue).sum()
                    / (double) copiesMade);
        }
        failures.stream().limit(10).forEach(f -> out.println("  " + f));
    }

    private static boolean sameEnd(BotGame.Result x, BotGame.Result y) {
        return Objects.equals(x.winner(), y.winner()) && x.turns() == y.turns()
                && Objects.equals(x.reason(), y.reason()) && Objects.equals(x.error(), y.error())
                && x.decisions() == y.decisions();
    }

    /** Where two step lists first differ (player, question or answer), or -1. */
    private static int firstDifference(List<BotGame.Step> a, List<BotGame.Step> b) {
        for (int i = 0; i < Math.min(a.size(), b.size()); i++)
            if (!a.get(i).equals(b.get(i)))
                return i;
        return a.size() == b.size() ? -1 : Math.min(a.size(), b.size());
    }

    private static String describe(String what, List<BotGame.Step> want, List<BotGame.Step> got, int at) {
        return what + " differs at step " + at + ": want "
                + (at < want.size() ? abbreviate(want.get(at).toString()) : "(end)") + " got "
                + (at < got.size() ? abbreviate(got.get(at).toString()) : "(end)");
    }

    private static String abbreviate(String s) {
        return s.length() > 240 ? s.substring(0, 240) + "..." : s;
    }

    private static void checkGame(String label, LotroFormat format, BotVsBot.Context ctx, Map<String, LotroDeck> decks,
                                  long seed, String policy, GameCopier copier, int worlds, int depth) {
        BotGame.Result original = BotGame.play(format, ctx.library(),
                List.of(BotVsBot.bot(policy, "~a", ctx), BotVsBot.bot(policy, "~b", ctx)), decks, 1000, seed);
        if (!original.finished()) {
            skipped.incrementAndGet();
            return;
        }
        List<BotGame.Step> recorded = original.steps();
        Random pick = new Random(seed ^ 0x9E3779B97F4A7C15L);
        int k = pick.nextInt(recorded.size());
        decisionsChecked.addAndGet(k);
        List<BotGame.Step> prefix = List.copyOf(recorded.subList(0, k));
        boolean[] aOk = {false};

        BotReplay.Script script = new BotReplay.Script(recorded);
        int[] asked = {0};
        BotPlayer[] probes = new BotPlayer[2];
        for (int i = 0; i < 2; i++) {
            String name = i == 0 ? "~a" : "~b";
            probes[i] = new BotPlayer() {
                @Override
                public String chooseAction(GameState gameState, AwaitingDecision decision) {
                    if (asked[0]++ == k) {
                        try {
                            probe(gameState, decision);
                        } catch (RuntimeException e) {
                            // BotGame would swallow it and answer at random
                            failures.add(label + " at k=" + k + ": " + e);
                            if (System.getenv("CLONE_TRACE") != null)
                                e.printStackTrace(System.err);
                        }
                    }
                    return script.answer(name);
                }

                private void probe(GameState gameState, AwaitingDecision decision) {
                    {
                        DefaultLotroGame live = (DefaultLotroGame) gameState.getGame();
                        String before = BotGame.fingerprint(decision);
                        // (b) first: a copy played out by random bots on a reseeded engine
                        DefaultLotroGame wild = copier.copy(live);
                        wild.getGameState().setRandomSeed(seed * 31 + 7);
                        try {
                            for (String p : wild.getGameState().getPlayerNames())
                                wild.getGameState().shuffleDeck(p);
                        } catch (NullPointerException e) {
                            // no decks yet: the bid and seating come before them
                        }
                        BotGame.resume(wild, List.of(new RandomDecisionBot("~a"), new RandomDecisionBot("~b")), 1000,
                                new ArrayList<>(), Map.of(), k, new Random(seed));
                        // (c) rollouts by replay and by copy, on the same worlds
                        checkRollouts(label, format, ctx, decks, seed, policy, copier, live, prefix, recorded.get(k),
                                decision, worlds, depth, pick);
                        // (a) a copy continued with the recorded answers
                        DefaultLotroGame copy = copier.copy(live);
                        BotReplay.Script rest = new BotReplay.Script(recorded.subList(k, recorded.size()));
                        Map<String, Integer> counts = new HashMap<>();
                        prefix.forEach(s -> counts.merge(s.player(), 1, Integer::sum));
                        List<BotGame.Step> steps = new ArrayList<>();
                        BotGame.Result r = BotGame.resume(copy, List.of(new BotReplay.Scripted("~a", rest),
                                new BotReplay.Scripted("~b", rest)), 1000, steps, counts, k, new Random(0));
                        int at = firstDifference(recorded.subList(k, recorded.size()), steps);
                        if (at < 0 && sameEnd(original, r))
                            aOk[0] = true;
                        else
                            failures.add(label + " (a) at k=" + k + ": " + (at >= 0
                                    ? describe("copy", recorded.subList(k, recorded.size()), steps, at)
                                    : "same steps, end " + r.winner() + "/" + r.turns() + "/" + r.reason() + "/"
                                    + r.error() + " vs " + original.winner() + "/" + original.turns() + "/"
                                    + original.reason()));
                        if (!before.equals(BotGame.fingerprint(live.getUserFeedback() instanceof
                                com.gempukku.lotro.game.DefaultUserFeedback f ? f.getAwaitingDecision(name) : null)))
                            failures.add(label + " (b): the live decision changed while copies were played");
                    }
                }

                @Override
                public String getName() {
                    return name;
                }
            };
        }
        BotGame.Result replayed = BotGame.play(format, ctx.library(), List.of(probes), decks, 1000, seed);
        int at = firstDifference(recorded, replayed.steps());
        games.incrementAndGet();
        if (aOk[0])
            aSame.incrementAndGet();
        if (at < 0 && sameEnd(original, replayed))
            bSame.incrementAndGet();
        else
            failures.add(label + " (b) at k=" + k + ": " + (at >= 0 ? describe("original", recorded,
                    replayed.steps(), at) : "same steps, different end"));
    }

    private static void checkRollouts(String label, LotroFormat format, BotVsBot.Context ctx,
                                      Map<String, LotroDeck> decks, long seed, String policy, GameCopier copier,
                                      DefaultLotroGame live, List<BotGame.Step> prefix, BotGame.Step here,
                                      AwaitingDecision decision, int worlds, int depth, Random pick) {
        String[] actionIds = decision.getDecisionParameters().get("actionId");
        for (int w = 0; w < worlds; w++) {
            long world = pick.nextLong();
            // an option as BotTeach would try one: any card action or pass; otherwise the recorded answer
            String answer = here.answer();
            if (decision.getDecisionType() == AwaitingDecisionType.CARD_ACTION_CHOICE && actionIds != null
                    && actionIds.length > 0) {
                int i = pick.nextInt(actionIds.length + 1);
                answer = i == actionIds.length ? "" : actionIds[i];
            }
            // by replay, exactly as BotTeach.Branch.value
            BotTeach.Shared byReplay = new BotTeach.Shared(prefix, here.player(), answer, world, depth, false);
            long[] branchedAt = {0};
            long r0 = cpu();
            List<BotPlayer> replayBots = new ArrayList<>();
            for (String name : List.of("~a", "~b")) {
                BotTeach.Branching b = new BotTeach.Branching(name, byReplay, BotVsBot.bot(policy, name, ctx));
                replayBots.add(new BotPlayer() {
                    @Override
                    public String chooseAction(GameState gameState, AwaitingDecision d) {
                        if (!byReplay.branched && byReplay.next == byReplay.prefix.size() && branchedAt[0] == 0)
                            branchedAt[0] = cpu();
                        return b.chooseAction(gameState, d);
                    }

                    @Override
                    public String getName() {
                        return name;
                    }
                });
            }
            BotGame.Result r1 = BotGame.play(format, ctx.library(), replayBots, decks, 1000, seed);
            // by copy, as BotTeach --clone
            BotTeach.Shared byCopy = new BotTeach.Shared(List.of(), here.player(), answer, world, depth, false);
            BotGame.Result r2 = CloneRollout.play(copier, live, prefix, byCopy, BotVsBot.bot(policy, "~a", ctx),
                    BotVsBot.bot(policy, "~b", ctx));
            if (branchedAt[0] > 0) {
                replayNanos.addAndGet(branchedAt[0] - r0);
                int reps = 5; // the copy the rollout made, again, timed on its own
                long c0 = cpu();
                for (int i = 0; i < reps; i++)
                    copier.copy(live);
                copyNanos.addAndGet((cpu() - c0) / reps);
                timedRollouts.incrementAndGet();
            }
            cRuns.incrementAndGet();
            List<BotGame.Step> after = r1.steps().subList(Math.min(prefix.size(), r1.steps().size()), r1.steps().size());
            int at = firstDifference(after, r2.steps());
            boolean same = at < 0 && sameEnd(r1, r2) && byReplay.value == byCopy.value
                    && byReplay.diverged == byCopy.diverged;
            if (same)
                cSame.incrementAndGet();
            else
                failures.add(label + " (c) k=" + prefix.size() + " world " + world + ": " + (at >= 0
                        ? describe("rollout", after, r2.steps(), at)
                        : "same steps; end " + r2.winner() + "/" + r2.turns() + "/" + r2.reason() + "/" + r2.error()
                        + " vs " + r1.winner() + "/" + r1.turns() + "/" + r1.reason() + "/" + r1.error()
                        + ", value " + byCopy.value + " vs " + byReplay.value));
        }
    }
}
