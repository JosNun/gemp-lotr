package com.gempukku.lotro.bots.simulation;

import com.gempukku.lotro.bots.BotPlayer;
import com.gempukku.lotro.bots.ModelDirectory;
import com.gempukku.lotro.bots.random.RandomDecisionBot;
import com.gempukku.lotro.bots.rl.fotrstarters.FotrStarterBot;
import com.gempukku.lotro.bots.rl.fotrstarters.FotrStartersRLGameStateFeatures;
import com.gempukku.lotro.bots.rl.fotrstarters.models.ModelRegistry;
import com.gempukku.lotro.common.CardType;
import com.gempukku.lotro.common.SitesBlock;
import com.gempukku.lotro.db.DeckSerialization;
import com.gempukku.lotro.game.CardCollection;
import com.gempukku.lotro.game.CardNotFoundException;
import com.gempukku.lotro.game.DefaultAdventureLibrary;
import com.gempukku.lotro.game.LotroCardBlueprintLibrary;
import com.gempukku.lotro.game.LotroFormat;
import com.gempukku.lotro.game.formats.LotroFormatLibrary;
import com.gempukku.lotro.logic.vo.LotroDeck;
import com.gempukku.lotro.packs.ProductLibrary;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.*;
import java.util.function.BiFunction;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * Headless bot-vs-bot, for comparing bots and decks without a server or database.
 * <p>
 * Two bots, A and B, each named by kind (see {@link #KINDS}), play every matchup with seats swapped on alternate
 * games, so a strong deck cannot pass for a strong bot. With A and B of the same kind the A/B split is noise around
 * 50% and the interesting numbers are the per-deck ones.
 * <p>
 * Games are played by {@link BotGame}, which applies each answer as the live server does.
 * <p>
 * Run from gemp-lotr/ (where bot-models/ and gemp-lotr-cards/ live), through bot-vs-bot.sh:
 * <pre>
 *   ./bot-vs-bot.sh aragorn gandalf --games 200
 *   ./bot-vs-bot.sh "starter:TTT - Aragorn" "starter:TTT - Theoden" --format ttt_block --a fotr --b random
 *   ./bot-vs-bot.sh --suite starters --games 10          # every starter, paired within its site block
 *   ./bot-vs-bot.sh --suite my-suite.txt --a general --b fotr
 * </pre>
 * A deck is "aragorn", "gandalf", "starter:&lt;name&gt;" for any starter product in gemp-lotr-cards (--list-starters
 * names them), or a file holding GEMP's deck string, ringbearer|ring|sites|cards, comma-separated blueprint ids.
 * Lines starting with # are ignored and whitespace is dropped, so a deck file can be split over lines. A suite file
 * holds one matchup per line, "format | deck1 | deck2".
 */
public class BotVsBot {
    static final String ARAGORN_STARTER = "1_290|1_2|1_320,1_327,1_340,1_346,1_349,1_351,1_355,1_358,1_361|1_365,1_365,1_92,1_94,1_94,1_97," +
            "1_97,1_101,1_104,1_104,1_106,1_107,1_107,1_296,1_296,1_299,1_299,1_51,1_108,1_108,1_110,1_110,1_309," +
            "1_311,1_116,1_116,1_116,1_117,1_117,1_117,1_121,1_121,1_121,1_133,1_133,1_141,1_141,1_145,1_150," +
            "1_150,1_150,1_150,1_151,1_151,1_151,1_152,1_152,1_152,1_152,1_153,1_153,1_153,1_153,1_154,1_154," +
            "1_154,1_157,1_157,1_158,1_158";
    static final String GANDALF_STARTER = "1_290|1_2|1_326,1_331,1_337,1_345,1_349,1_350,1_353,1_359,1_360|1_70,1_97,1_286,1_286,1_286,1_37," +
            "1_37,1_364,1_364,1_12,1_12,1_296,1_296,1_299,1_76,1_76,1_76,1_51,1_51,1_78,1_78,1_78,1_78,1_304," +
            "1_304,1_84,1_312,1_26,1_26,1_86,1_168,1_168,1_168,1_176,1_176,1_176,1_176,1_177,1_178,1_178,1_178" +
            ",1_178,1_179,1_179,1_179,1_180,1_180,1_180,1_180,1_181,1_181,1_181,1_187,1_187,1_187,1_191," +
            "1_191,1_191,1_196,1_196";

    /**
     * The engine's names for the players: by the deck they hold, not by the bot holding it. The engine draws one Random
     * in player order and keeps players in maps keyed by name, so with the names fixed the two games of a pair deal each
     * deck the same cards and seat the decks the same way, and only which bot answers for which deck differs.
     */
    private static final String DECK_1 = "~1", DECK_2 = "~2";
    /** Which of those bot A is in the game this thread is playing, for the tally of its guesses. */
    private static final ThreadLocal<String> A_NAME = new ThreadLocal<>();

    /** The format a starter is played under when a suite pairs it up, by the block of its sites. */
    private static final Map<SitesBlock, String> FORMAT_FOR_SITES = Map.of(
            SitesBlock.FELLOWSHIP, "fotr_block",
            SitesBlock.TWO_TOWERS, "ttt_block",
            SitesBlock.KING, "king_block",
            SitesBlock.SHADOWS, "expanded",
            SitesBlock.MULTIPATH, "multipath");

    /** Lines a bot prints when it has nothing better than a random answer. */
    private static final Pattern UNKNOWN = Pattern.compile("^Unknown .*");
    /** Card ids and option lists differ every time; drop them so the same decision tallies as one. */
    private static final Pattern VOLATILE = Pattern.compile("\\[[^]]*]|\\([^)]*\\)|;.*$|\\d+");

    private static final ThreadLocal<int[]> randomAnswers = ThreadLocal.withInitial(() -> new int[2]);

    record Options(List<String> decks, String suite, String format, String kindA, String kindB, int games, int jobs,
                   int maxDecisions, Path resources, Path models, boolean verbose, boolean listStarters, Long seed) {
    }

    record Matchup(String format, String deck1, String deck2) {
        String label() {
            return deck1 + " vs " + deck2 + (format.equals("fotr_block") ? "" : " [" + format + "]");
        }
    }

    /**
     * One game. {@code aSeat} is the deck A played (1 or 2); winner and first are "a", "b" or null.
     * {@code random} counts answers each bot could only guess, [A, B].
     */
    record Outcome(int matchup, int game, int aSeat, String winner, String first, String reason, int turns,
                   int[] random, int decisions, String error, long cpuNanos) {
        boolean finished() {
            return error == null && winner != null;
        }

        int winningSeat() {
            return winner.equals("a") ? aSeat : 3 - aSeat;
        }
    }

    /** The bots this runner knows, by the name --a and --b take. */
    static final Map<String, BiFunction<String, Context, BotPlayer>> KINDS = new LinkedHashMap<>();

    static {
        KINDS.put("fotr", (name, ctx) ->
                new FotrStarterBot(new FotrStartersRLGameStateFeatures(), name, ctx.models(ctx.modelDir()), null));
        KINDS.put("general", (name, ctx) -> new FotrStarterBot(new FotrStartersRLGameStateFeatures(), name,
                ctx.models(ctx.modelDir()), null, 0.0, true));
        KINDS.put("random", (name, ctx) -> new RandomDecisionBot(name));
    }

    /**
     * A kind from --a/--b: a name in {@link #KINDS}, or "fotr@dir" / "general@dir" for that bot on the models in
     * dir. "general" is FotrStarterBot with its extensions on (see FotrStarterBot.extended).
     */
    /** Game copies for search@ bots' play-outs: one per run, shared by the game threads. */
    private static volatile GameCopier searchCopier;

    static BotPlayer bot(String kind, String name, Context ctx) {
        if (kind.startsWith("search@")) {
            // the general bot on that directory, searching where it is unsure (Searching; search.* properties)
            String dir = kind.substring("search@".length());
            if (searchCopier == null)
                synchronized (BotVsBot.class) {
                    if (searchCopier == null)
                        searchCopier = new GameCopier(ctx.library(), ctx.formats());
                }
            // -Dsearch.policy=<dir>: the bots that play the play-outs out (default: the searching bot's own) -- the
            // network was ~55% of a search game's time, most of it the play-outs' own decisions
            String policy = System.getProperty("search.policy", dir);
            // -Dsearch.root=<dir>: the bot that answers (default: dir's own); dir then only lends its judge. A root
            // with no network searches on its trees' scores (Searching.treesRoot)
            String root = System.getProperty("search.root", dir);
            return new Searching(bot("general@" + root, name, ctx), ctx.models(Path.of(dir)).getBoardNet2(),
                    searchCopier, n -> bot("general@" + policy, n, ctx),
                    ctx.models(Path.of(root)).getBoardNet2() == null);
        }
        int at = kind.indexOf('@');
        if (at > 0) {
            BotPlayer bot = new FotrStarterBot(new FotrStartersRLGameStateFeatures(), name,
                    ctx.models(Path.of(kind.substring(at + 1))), null, 0.0, !kind.startsWith("fotr@"));
            if (kind.startsWith("safe@") || kind.startsWith("careful@")
                    || System.getenv("BOT_GUARD") != null && kind.startsWith("general@")) {
                // see Guarded: never corrupt your own Ring-bearer with a card's burden cost; careful@ also declines
                // extra moves with a thin fellowship
                Guarded.init(ctx);
                return new Guarded(bot, kind.startsWith("careful@") ? Integer.getInteger("careful", 2) : -1);
            }
            return bot;
        }
        return KINDS.get(kind).apply(name, ctx);
    }

    static boolean knownKind(String kind) {
        int at = kind.indexOf('@');
        if (at > 0)
            return List.of("fotr", "general", "safe", "careful", "search").contains(kind.substring(0, at))
                    && Files.isDirectory(Path.of(kind.substring(at + 1)));
        return KINDS.containsKey(kind);
    }

    record Context(LotroCardBlueprintLibrary library, LotroFormatLibrary formats, Path resources, Path modelDir,
                   Map<String, Object> cache) {
        synchronized ModelRegistry models(Path dir) {
            return (ModelRegistry) cache.computeIfAbsent("models:" + dir.toAbsolutePath().normalize(), k -> {
                try {
                    return loadModels(dir);
                } catch (IOException | ClassNotFoundException e) {
                    throw new RuntimeException(e);
                }
            });
        }

        @SuppressWarnings("unchecked")
        synchronized Map<String, List<CardCollection.Item>> starters() {
            return (Map<String, List<CardCollection.Item>>) cache.computeIfAbsent("starters", k -> loadStarters(this));
        }
    }

    /** A serialised model file. */
    static smile.classification.SoftClassifier<double[]> readModel(Path file) throws IOException {
        return ModelDirectory.readModel(file);
    }

    /** The card and format libraries, read straight from gemp-lotr-cards with no server configuration. */
    static Context context(Path res, Path modelDir) {
        if (!Files.isDirectory(res.resolve("cards")))
            die("No cards under " + res.toAbsolutePath() + " -- run from gemp-lotr/, or pass --resources");
        LotroCardBlueprintLibrary library = new LotroCardBlueprintLibrary(res.resolve("cards").toFile(),
                res.resolve("blueprintMapping.txt").toFile(), res.resolve("setConfig.hjson").toFile(),
                res.resolve("rarities").toFile());
        LotroFormatLibrary formats = new LotroFormatLibrary(new DefaultAdventureLibrary(), library,
                res.resolve("lotrFormats.hjson").toFile(), res.resolve("sealed").toFile());
        ModelDirectory.prepare(library, formats, res.resolve("cards").toFile());
        return new Context(library, formats, res, modelDir, new ConcurrentHashMap<>());
    }

    public static void main(String[] args) throws Exception {
        try {
            run(args);
        } finally {
            System.exit(0); // also after an exception, which otherwise leaves the game pool's threads holding the JVM
        }
    }

    private static void run(String[] args) throws Exception {
        Options opts = parse(args);

        // The card library and the bots chat on System.out: one line per card file loaded, and one per decision a
        // bot could only guess at. Keep the second kind, tallied by bot; drop the rest.
        Map<String, Integer> unknown = new ConcurrentHashMap<>();
        PrintStream out = System.out;
        if (!opts.verbose())
            System.setOut(new PrintStream(new LineTally(unknown, opts), true, StandardCharsets.UTF_8));

        long t0 = System.nanoTime();
        Path res = opts.resources();
        Context ctx = context(res, opts.models());
        LotroCardBlueprintLibrary library = ctx.library();
        LotroFormatLibrary formats = ctx.formats();

        if (opts.listStarters()) {
            for (String name : new TreeSet<>(ctx.starters().keySet())) {
                try {
                    out.printf("%-58s %s%n", name, siteBlock(starterDeck(name, "fotr_block", ctx), library));
                } catch (IllegalArgumentException e) {
                    System.err.println("skipped " + e.getMessage());
                }
            }
            return;
        }

        List<Matchup> matchups = opts.suite() != null ? loadSuite(opts.suite(), ctx)
                : List.of(new Matchup(opts.format(), opts.decks().get(0), opts.decks().get(1)));
        // Load every deck once up front, so a typo fails before the first game rather than mid-run.
        Map<String, LotroDeck> decks = new HashMap<>();
        for (Matchup m : matchups) {
            if (formats.getFormat(m.format()) == null)
                die("Unknown format \"" + m.format() + "\". Codes are in " + res.resolve("lotrFormats.hjson"));
            for (String spec : List.of(m.deck1(), m.deck2())) {
                try {
                    decks.computeIfAbsent(m.format() + "|" + spec, k -> loadDeck(spec, m.format(), ctx));
                } catch (IllegalArgumentException e) {
                    die(e.getMessage());
                }
            }
        }
        for (String kind : List.of(opts.kindA(), opts.kindB()))
            if (!knownKind(kind))
                die("Unknown bot \"" + kind + "\"; known: " + String.join(", ", KINDS.keySet()) + ", fotr@<model dir>");

        System.err.printf("Loaded in %.1fs. A = %s, B = %s; %d matchup%s x %d games on %d threads; seed %s%n%n",
                (System.nanoTime() - t0) / 1e9, opts.kindA(), opts.kindB(), matchups.size(),
                matchups.size() == 1 ? "" : "s", opts.games(), opts.jobs(), opts.seed() == null ? "off" : opts.seed());

        ExecutorService pool = Executors.newFixedThreadPool(opts.jobs());
        List<List<Future<Outcome>>> futures = new ArrayList<>();
        FEED_TOTAL = matchups.size() * opts.games();
        if (FEED == null)
            FEED = ("bvb-" + opts.kindA().replaceAll(".*/", "") + "-vs-" + opts.kindB().replaceAll(".*/", "") + "-"
                    + new java.text.SimpleDateFormat("MMdd-HHmm").format(new Date())).replaceAll("[^A-Za-z0-9._-]", "_");
        else if (FEED.equals("off"))
            FEED = null;
        for (int mi = 0; mi < matchups.size(); mi++) {
            Matchup m = matchups.get(mi);
            LotroFormat format = formats.getFormat(m.format());
            LotroDeck deck1 = decks.get(m.format() + "|" + m.deck1());
            LotroDeck deck2 = decks.get(m.format() + "|" + m.deck2());
            List<Future<Outcome>> games = new ArrayList<>();
            for (int g = 0; g < opts.games(); g++) {
                int matchup = mi, game = g;
                games.add(pool.submit(() -> play(matchup, game, format, deck1, deck2, opts, ctx)));
            }
            futures.add(games);
        }

        List<Outcome> all = new ArrayList<>();
        boolean single = matchups.size() == 1;
        for (int mi = 0; mi < matchups.size(); mi++) {
            List<Outcome> outcomes = new ArrayList<>();
            List<Future<Outcome>> games = futures.get(mi);
            for (int g = 0; g < games.size(); g++) {
                Outcome o;
                try {
                    // see BotTeach.generate: a game can spin in one engine call where nothing reaches it
                    o = games.get(g).get(15, TimeUnit.MINUTES);
                } catch (TimeoutException e) {
                    o = new Outcome(mi, g, g % 2 == 0 ? 1 : 2, null, null, null, 0, new int[2], 0,
                            "abandoned: still running after 15 minutes", 0);
                }
                outcomes.add(o);
                if (single || o.error() != null)
                    out.println(describe(o, matchups.get(mi)));
            }
            if (!single)
                out.println(matchupLine(matchups.get(mi), outcomes));
            all.addAll(outcomes);
        }
        pool.shutdown();
        System.setOut(out);

        summarise(all, matchups, opts, unknown, (System.nanoTime() - t0) / 1e9);
        CardUse.report(System.out);
        standings(all, matchups);
        if (Searching.searched.get() > 0)
            System.out.printf("search: %d decisions searched, %d answered differently from the bot; %.1f s a search, "
                            + "%d play-outs, %.0f steps each, %d judged, %d stopped after the first round%n",
                    Searching.searched.get(), Searching.changed.get(),
                    Searching.nanos.get() / 1e9 / Searching.searched.get(), Searching.playOuts.get(),
                    (double) Searching.steps.get() / Math.max(1, Searching.playOuts.get()), Searching.judged.get(),
                    Searching.stoppedEarly.get());
        if (Searching.assignSearched.get() > 0)
            System.out.printf("assignments: %d searched, %d answered differently from the bot, %d stopped after the "
                            + "first round%n", Searching.assignSearched.get(), Searching.assignChanged.get(),
                    Searching.assignStopped.get());
        Searching.SELECT_COUNTS.forEach((k, c) -> {
            if (c[0].get() > 0)
                System.out.printf("%s: %d searched, %d answered differently from the bot, %d stopped after the first "
                        + "round%n", k, c[0].get(), c[1].get(), c[2].get());
        });
        System.exit(0); // ends any game thread still spinning in the engine
    }

    private static final boolean FIXED_DECKS = Boolean.getBoolean("bvb.fixedDecks");
    /**
     * -Dbvb.boards=&lt;file&gt;: board records of one seat's decisions with its answers (Recording) -- the trees' choices
     * for the network; -Dbvb.boardsSeat=A|B|both (A), -Dbvb.boardsRate (1.0: every decision), -Dbvb.outcomes=&lt;file&gt;
     * how each game ended, for the outcome head.
     */
    /**
     * A record per finished game for the dashboard's live panels (Feed): to the stream -Dfeed names, by default
     * bvb-&lt;A&gt;-vs-&lt;B&gt;-&lt;when it started&gt;; -Dfeed=off for none.
     */
    static String FEED = System.getProperty("feed");
    static final java.util.concurrent.atomic.AtomicInteger FEED_DONE = new java.util.concurrent.atomic.AtomicInteger(),
            FEED_FINISHED = new java.util.concurrent.atomic.AtomicInteger(),
            FEED_A_WINS = new java.util.concurrent.atomic.AtomicInteger();
    static volatile int FEED_TOTAL;

    private static final java.io.PrintWriter BOARDS = Recording.open(System.getProperty("bvb.boards")),
            OUTCOMES = Recording.open(System.getProperty("bvb.outcomes"));
    private static final String BOARDS_SEAT = System.getProperty("bvb.boardsSeat", "A");
    private static final double BOARDS_RATE = Double.parseDouble(System.getProperty("bvb.boardsRate", "1.0"));

    static {
        if (BOARDS != null) {
            BoardDump.TRUE_DECK = true;
            BoardDump.LIVE = true;
        }
    }

    /** -Dbvb.gameLog=&lt;file&gt;: one JSON line per game -- how it ended and how each side played (GameLog). */
    private static final java.io.PrintWriter GAME_LOG = gameLog(System.getProperty("bvb.gameLog"));

    private static java.io.PrintWriter gameLog(String path) {
        if (path == null)
            return null;
        try {
            return new java.io.PrintWriter(java.nio.file.Files.newBufferedWriter(java.nio.file.Path.of(path),
                    java.nio.charset.StandardCharsets.UTF_8, java.nio.file.StandardOpenOption.CREATE,
                    java.nio.file.StandardOpenOption.APPEND), true);
        } catch (java.io.IOException e) {
            throw new java.io.UncheckedIOException(e);
        }
    }

    private static Outcome play(int matchup, int game, LotroFormat format, LotroDeck deck1, LotroDeck deck2,
                                Options opts, Context ctx) {
        // A takes deck 1 on even games and deck 2 on odd ones -- or always deck 1 with -Dbvb.fixedDecks, so that an
        // opponent from the pool of styles plays the decks its style was played with. Fresh bots every game:
        // FotrStarterBot keeps loop-detection state and an episode log that only endEpisode() clears.
        int aSeat = FIXED_DECKS || game % 2 == 0 ? 1 : 2;
        String aName = aSeat == 1 ? DECK_1 : DECK_2, bName = aSeat == 1 ? DECK_2 : DECK_1;
        BotPlayer a = CardUse.wrap(bot(opts.kindA(), aName, ctx), "A");
        BotPlayer b = CardUse.wrap(bot(opts.kindB(), bName, ctx), "B");
        String gameId = null;
        if (BOARDS != null || OUTCOMES != null) {
            gameId = "bvb-" + Long.toHexString(java.util.concurrent.ThreadLocalRandom.current().nextLong()) + "-" + matchup
                    + "-" + game;
            BoardDump.GAME.set(gameId);
        }
        if (BOARDS != null && (BOARDS_SEAT.equals("A") || BOARDS_SEAT.equals("both")))
            a = new Recording(a, BOARDS, BOARDS_RATE);
        if (BOARDS != null && (BOARDS_SEAT.equals("B") || BOARDS_SEAT.equals("both")))
            b = new Recording(b, BOARDS, BOARDS_RATE);
        Map<String, LotroDeck> decks = new HashMap<>();
        decks.put(DECK_1, deck1);
        decks.put(DECK_2, deck2);
        // deck 1's player first: when both are asked at once (the bids) the game loop answers the first in the list
        // first, and the bots' Random is drawn in that order
        List<BotPlayer> players = aSeat == 1 ? List.of(a, b) : List.of(b, a);
        // with fixed decks the two games of a "pair" hold the same decks, and would be the same game twice
        int pair = FIXED_DECKS ? game : game / 2;
        Long seed = opts.seed() == null ? null : gameSeed(opts.seed(), deck1.getDeckName(), deck2.getDeckName(), pair);

        int[] random = randomAnswers.get();
        Arrays.fill(random, 0);
        var threads = java.lang.management.ManagementFactory.getThreadMXBean();
        long cpu0 = threads.getCurrentThreadCpuTime();
        GameLog log = GAME_LOG != null ? new GameLog() : null;
        if (log != null)
            GameLog.CURRENT.set(log);
        A_NAME.set(aName);
        BotGame.Result r;
        try {
            r = BotGame.play(format, ctx.library(), players, decks, opts.maxDecisions(), seed);
        } finally {
            GameLog.CURRENT.remove();
            A_NAME.remove();
        }
        long cpu = threads.getCurrentThreadCpuTime() - cpu0;
        if (r.finished())
            Recording.outcome(OUTCOMES, gameId, r.winner(), r.reason());
        if (log != null) {
            Map<String, String> names = new java.util.LinkedHashMap<>();
            names.put(aName, "A");
            names.put(bName, "B");
            var line = log.finish(r, names);
            line.put("matchup", matchup);
            line.put("game", game);
            line.put("pair", pair);
            line.put("seed", seed);
            line.put("botA", opts.kindA());
            line.put("botB", opts.kindB());
            line.put("deckA", decks.get(aName).getDeckName());
            line.put("deckB", decks.get(bName).getDeckName());
            line.put("cpu", cpu / 1e9);
            synchronized (GAME_LOG) {
                GAME_LOG.println(line);
            }
        }
        random[0] += r.rejected().getOrDefault(aName, 0);
        random[1] += r.rejected().getOrDefault(bName, 0);
        if (FEED != null) {
            // the dashboard's live panel: progress, A's win rate as games come in, and how each one ended
            int done = FEED_DONE.incrementAndGet();
            String w = seat(r.winner(), aName);
            int fin = w == null ? FEED_FINISHED.get() : FEED_FINISHED.incrementAndGet();
            int wins = "a".equals(w) ? FEED_A_WINS.incrementAndGet() : FEED_A_WINS.get();
            Map<String, Object> rec = new LinkedHashMap<>();
            rec.put("done", done);
            rec.put("total", FEED_TOTAL);
            rec.put("a", opts.kindA());
            rec.put("b", opts.kindB());
            rec.put("wins", wins);
            rec.put("games", fin);
            rec.put("route", w == null ? "unfinished" : GameLog.route(r.reason()));
            rec.put("turns", r.turns());
            Feed.emit(FEED, rec);
        }
        return new Outcome(matchup, game, aSeat, seat(r.winner(), aName), seat(r.first(), aName), r.reason(),
                r.turns(), random.clone(), r.decisions(), r.error(), cpu);
    }

    /** "a" or "b" for an engine player name, given which name bot A had; null for no player. */
    private static String seat(String player, String aName) {
        return player == null || !(player.equals(DECK_1) || player.equals(DECK_2)) ? null
                : player.equals(aName) ? "a" : "b";
    }

    /**
     * A game's seed: the run's, the matchup's two decks by name, and the pair, so both games of a pair get the same
     * deal and a matchup gets the same deals in any suite it is in -- two runs on one --seed are then matched game by
     * game (scripts/paired.py). Hashed, not added: java.util.Random's first draws from nearby seeds are alike.
     */
    static long gameSeed(long base, String deck1, String deck2, int pair) {
        long h = mix(base);
        for (String deck : List.of(deck1, deck2)) {
            for (byte c : deck.getBytes(StandardCharsets.UTF_8))
                h = (h ^ (c & 0xff)) * 0x100000001B3L; // FNV-1a
            h = mix(h); // between the names, so ("ab", "c") and ("a", "bc") differ
        }
        return mix(h ^ pair);
    }

    /** SplittableRandom's finaliser (Stafford's mix 13). */
    private static long mix(long z) {
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    /**
     * A's wins in each pair -- games 2j and 2j+1 of a matchup, both finished -- in which A held each deck once, so the
     * decks' part cancels. Empty where games do not pair: fixed decks, or an odd number of games a matchup.
     */
    private static List<Integer> pairWins(List<Outcome> done, Options opts) {
        if (FIXED_DECKS || opts.games() % 2 != 0)
            return List.of();
        Map<List<Integer>, int[]> pairs = new HashMap<>(); // (matchup, pair) -> {games, A's wins}
        for (Outcome o : done) {
            int[] p = pairs.computeIfAbsent(List.of(o.matchup(), o.game() / 2), k -> new int[2]);
            p[0]++;
            p[1] += o.winner().equals("a") ? 1 : 0;
        }
        return pairs.values().stream().filter(p -> p[0] == 2).map(p -> p[1]).toList();
    }

    /**
     * The 95% half-width, in points, of a mean of per-pair scores (each in [0, 1]): from their spread, NaN under two
     * pairs. On two 6,330-game logs (unseeded) it was +-1.06-1.08 where the binomial interval said +-1.23; seeded, so a
     * pair is one deal played both ways round, sp-m3 vs sp-m2 over 600 games gave +-1.9 against +-3.4 unseeded by pair
     * and +-4.0 binomial (2026-11-09, scripts/paired.py).
     */
    private static double pairInterval(List<Double> scores) {
        int n = scores.size();
        if (n < 2)
            return Double.NaN;
        double mean = scores.stream().mapToDouble(Double::doubleValue).sum() / n;
        double ss = scores.stream().mapToDouble(x -> (x - mean) * (x - mean)).sum();
        return 196 * Math.sqrt(ss / (n - 1) / n);
    }

    private static String describe(Outcome o, Matchup m) {
        if (o.error() != null)
            return String.format("game %3d of %s: ERROR %s", o.game() + 1, m.label(), o.error());
        String winDeck = o.winningSeat() == 1 ? m.deck1() : m.deck2();
        return String.format("game %3d: %s won with %-28s turn %2d, %s%s", o.game() + 1, o.winner().toUpperCase(),
                winDeck, o.turns(), o.reason(), o.winner().equals(o.first()) ? " (went first)" : "");
    }

    private static String matchupLine(Matchup m, List<Outcome> outcomes) {
        List<Outcome> done = outcomes.stream().filter(Outcome::finished).toList();
        long aWins = done.stream().filter(o -> o.winner().equals("a")).count();
        long deck1Wins = done.stream().filter(o -> o.winningSeat() == 1).count();
        long errors = outcomes.size() - done.size();
        double randA = outcomes.stream().mapToInt(o -> o.random()[0]).sum() / (double) outcomes.size();
        double randB = outcomes.stream().mapToInt(o -> o.random()[1]).sum() / (double) outcomes.size();
        return String.format("%-70s A %2d/%-2d  deck1 %2d/%-2d  guesses/game A %5.1f B %5.1f%s",
                m.label(), aWins, done.size(), deck1Wins, done.size(), randA, randB,
                errors > 0 ? "  " + errors + " ERRORS" : "");
    }

    private static void summarise(List<Outcome> outcomes, List<Matchup> matchups, Options opts,
                                  Map<String, Integer> unknown, double seconds) {
        List<Outcome> done = outcomes.stream().filter(Outcome::finished).toList();
        long errors = outcomes.size() - done.size();

        System.out.println();
        System.out.printf("%d games in %.0fs, %d finished; %.2f CPU-seconds a game%n", outcomes.size(), seconds,
                done.size(), outcomes.stream().mapToLong(Outcome::cpuNanos).sum() / 1e9 / outcomes.size());
        System.out.println(opts.seed() == null ? "  seed off: every game dealt afresh"
                : "  seed " + opts.seed() + " (--seed " + opts.seed() + " deals these games again)");
        if (!done.isEmpty()) {
            long aWins = done.stream().filter(o -> o.winner().equals("a")).count();
            double p = (double) aWins / done.size();
            double binomial = 196 * Math.sqrt(p * (1 - p) / done.size());
            // the decks decide most games, and within a pair they swap: scored by pair, their part of the noise goes
            List<Integer> pairs = pairWins(done, opts);
            double byPair = pairInterval(pairs.stream().map(w -> w / 2.0).toList());
            // scripts grep this line up to the +- figure: keep that part as it is
            System.out.printf("  A (%s) won %d, B (%s) won %d: A %.1f%% +- %.1f%s%n", opts.kindA(), aWins, opts.kindB(),
                    done.size() - aWins, 100 * p, Double.isNaN(byPair) ? binomial : byPair,
                    Double.isNaN(byPair) ? " (binomial)" : String.format(" (by pair; binomial +- %.1f)", binomial));
            if (!pairs.isEmpty()) {
                long both = pairs.stream().filter(w -> w == 2).count(), neither = pairs.stream().filter(w -> w == 0).count();
                System.out.printf("  pairs: A won both %d, split %d, B won both %d%n", both,
                        pairs.size() - both - neither, neither);
            }
            if (matchups.size() == 1) {
                Matchup m = matchups.get(0);
                long deck1Wins = done.stream().filter(o -> o.winningSeat() == 1).count();
                System.out.printf("  %s won %d, %s won %d%n", m.deck1(), deck1Wins, m.deck2(), done.size() - deck1Wins);
            }
            long firstWins = done.stream().filter(o -> o.winner().equals(o.first())).count();
            System.out.printf("  player going first won %d of %d; median length %d turns, %d decisions%n",
                    firstWins, done.size(), done.stream().mapToInt(Outcome::turns).sorted().toArray()[done.size() / 2],
                    done.stream().mapToInt(Outcome::decisions).sorted().toArray()[done.size() / 2]);
            for (int seat = 0; seat < 2; seat++) {
                int s = seat;
                long guesses = outcomes.stream().mapToLong(o -> o.random()[s]).sum();
                String kind = s == 0 ? opts.kindA() : opts.kindB();
                System.out.printf("  %s (%s) guessed %s answers per game%n", s == 0 ? "A" : "B", kind,
                        kind.equals("random") ? "all its" : String.format("%.1f", guesses / (double) outcomes.size()));
            }
            System.out.println("  how games ended:");
            done.stream().collect(Collectors.groupingBy(o -> o.reason().replaceAll("\\d+", "N"), TreeMap::new,
                            Collectors.counting()))
                    .forEach((r, n) -> System.out.printf("    %5d  %s%n", n, r));
            // going first wins 54% of people's Movie Block games and lost between the bots: how each seat wins
            System.out.println("  how games ended, by whether the winner went first (first / second):");
            done.stream().collect(Collectors.groupingBy(o -> o.reason().replaceAll("\\d+", "N"), TreeMap::new,
                            Collectors.partitioningBy(o -> o.winner().equals(o.first()), Collectors.counting())))
                    .forEach((r, n) -> System.out.printf("    %5d / %5d  %s%n", n.get(true), n.get(false), r));
        }
        if (errors > 0) {
            System.out.printf("  %d games crashed or stalled:%n", errors);
            outcomes.stream().filter(o -> o.error() != null)
                    .collect(Collectors.groupingBy(o -> o.error().replaceAll("\\d+", "N"), Collectors.counting()))
                    .entrySet().stream().sorted(Map.Entry.<String, Long>comparingByValue().reversed()).limit(10)
                    .forEach(e -> System.out.printf("    %5d  %s%n", e.getValue(), e.getKey()));
        }
        if (!BotGame.BOT_ERRORS.isEmpty()) {
            System.out.println("  bots threw while choosing (count, exception, decision) -- the fallback answered:");
            BotGame.BOT_ERRORS.entrySet().stream().sorted(Map.Entry.<String, Integer>comparingByValue().reversed())
                    .limit(10).forEach(e -> System.out.printf("    %5d  %s%n", e.getValue(), e.getKey()));
        }
        if (!BotGame.BOT_REJECTED.isEmpty()) {
            // by bot and kind of decision, whatever the answer was
            Map<String, Integer> byKind = new TreeMap<>();
            BotGame.BOT_REJECTED.forEach((k, n) -> byKind.merge(k.replaceAll(" answered '.*'$", ""), n, Integer::sum));
            System.out.println("  answers the engine refused, by bot and decision (count):");
            byKind.entrySet().stream().sorted(Map.Entry.<String, Integer>comparingByValue().reversed()).limit(12)
                    .forEach(e -> System.out.printf("    %5d  %s%n", e.getValue(), e.getKey()));
            System.out.println("  answers the engine refused (count, bot: decision) -- the fallback answered:");
            BotGame.BOT_REJECTED.entrySet().stream().sorted(Map.Entry.<String, Integer>comparingByValue().reversed())
                    .limit(15).forEach(e -> System.out.printf("    %5d  %s%n", e.getValue(), e.getKey()));
        }
        if (!unknown.isEmpty()) {
            System.out.println("  decisions answered at random, by bot (count, decision):");
            unknown.entrySet().stream().sorted(Map.Entry.<String, Integer>comparingByValue().reversed()).limit(25)
                    .forEach(e -> System.out.printf("    %5d  %s%n", e.getValue(), e.getKey()));
        }
    }

    /**
     * Each deck's record over every game it played, best first, when a suite has more than one matchup: for a
     * round robin ("rr:" suites) this is the ranking of the decks.
     */
    static void standings(List<Outcome> outcomes, List<Matchup> matchups) {
        if (matchups.size() < 2)
            return;
        Map<String, int[]> record = new TreeMap<>(); // deck -> {wins, games}
        // deck -> (matchup, pair) -> {games, wins}: seeded, a pair deals the deck the same cards twice, so its two
        // games are not two independent tries and a binomial interval would be too narrow
        Map<String, Map<List<Integer>, int[]>> byPair = new HashMap<>();
        for (Outcome o : outcomes) {
            if (!o.finished())
                continue;
            Matchup m = matchups.get(o.matchup());
            String winner = o.winningSeat() == 1 ? m.deck1() : m.deck2();
            for (String deck : List.of(m.deck1(), m.deck2())) {
                int[] r = record.computeIfAbsent(deck, k -> new int[2]);
                r[1]++;
                if (deck.equals(winner))
                    r[0]++;
                int[] q = byPair.computeIfAbsent(deck, k -> new HashMap<>())
                        .computeIfAbsent(List.of(o.matchup(), FIXED_DECKS ? o.game() : o.game() / 2), k -> new int[2]);
                q[0]++;
                q[1] += deck.equals(winner) ? 1 : 0;
            }
        }
        System.out.println("  standings (wins / games, 95% interval: by pair where a deck has pairs, else binomial):");
        record.entrySet().stream()
                .sorted(Comparator.comparingDouble(e -> -(double) e.getValue()[0] / e.getValue()[1]))
                .forEach(e -> {
                    double p = (double) e.getValue()[0] / e.getValue()[1];
                    double ci = pairInterval(byPair.get(e.getKey()).values().stream().filter(q -> q[0] == 2)
                            .map(q -> q[1] / 2.0).toList());
                    System.out.printf("    %5.1f%% +- %4.1f  %4d / %-4d  %s%n", 100 * p,
                            Double.isNaN(ci) ? 196 * Math.sqrt(p * (1 - p) / e.getValue()[1]) : ci, e.getValue()[0],
                            e.getValue()[1], e.getKey());
                });
    }

    // ---- decks and suites ----

    /**
     * "starters[:train|test|all]", "rr:&lt;format&gt;:&lt;file&gt;" -- every pair of the decks listed in the file, one
     * deck spec per line (# comments), all under one format -- or a suite file of "format | deck1 | deck2" lines.
     */
    static List<Matchup> loadSuite(String spec, Context ctx) throws IOException {
        if (spec.equals("starters") || spec.startsWith("starters:"))
            return starterSuite(ctx, spec.contains(":") ? spec.substring(spec.indexOf(':') + 1) : "all");
        if (spec.startsWith("rr:")) {
            String[] parts = spec.split(":", 3);
            if (parts.length != 3)
                die("Round robin suite is rr:<format>:<deck list file>");
            List<String> decks = Files.readAllLines(Path.of(parts[2])).stream().map(String::trim)
                    .filter(l -> !l.isEmpty() && !l.startsWith("#")).toList();
            List<Matchup> matchups = new ArrayList<>();
            for (int i = 0; i < decks.size(); i++)
                for (int j = i + 1; j < decks.size(); j++)
                    matchups.add(new Matchup(parts[1], decks.get(i), decks.get(j)));
            return matchups;
        }
        List<Matchup> matchups = new ArrayList<>();
        for (String line : Files.readAllLines(Path.of(spec))) {
            if (line.isBlank() || line.trim().startsWith("#"))
                continue;
            String[] parts = line.split("\\|");
            if (parts.length != 3)
                die("Suite line is not \"format | deck1 | deck2\": " + line);
            matchups.add(new Matchup(parts[0].trim(), parts[1].trim(), parts[2].trim()));
        }
        return matchups;
    }

    /**
     * Every starter whose sites put it in a format we know, each paired with the next starter of the same site
     * block (wrapping round), so every starter plays twice, once in each column.
     * <p>
     * {@code split} "train" keeps the 1st, 3rd, 5th... starter of each block by name, "test" the others, so a bot
     * trained on one half can be scored on decks it has never seen. "all" keeps everything.
     */
    private static List<Matchup> starterSuite(Context ctx, String split) {
        if (!List.of("all", "train", "test").contains(split))
            die("starters split must be all, train or test, not \"" + split + "\"");
        Map<String, List<String>> byFormat = new TreeMap<>();
        for (String name : new TreeSet<>(ctx.starters().keySet())) {
            String format;
            try {
                format = FORMAT_FOR_SITES.get(siteBlock(starterDeck(name, "fotr_block", ctx), ctx.library()));
            } catch (IllegalArgumentException e) {
                System.err.println("skipped " + e.getMessage());
                continue;
            }
            if (format != null)
                byFormat.computeIfAbsent(format, k -> new ArrayList<>()).add(name);
        }
        if (!split.equals("all")) {
            int keep = split.equals("train") ? 0 : 1;
            byFormat.replaceAll((format, names) -> java.util.stream.IntStream.range(0, names.size())
                    .filter(i -> i % 2 == keep).mapToObj(names::get).toList());
        }
        List<Matchup> matchups = new ArrayList<>();
        byFormat.forEach((format, names) -> {
            for (int i = 0; i < names.size() && names.size() > 1; i++)
                matchups.add(new Matchup(format, "starter:" + names.get(i),
                        "starter:" + names.get((i + 1) % names.size())));
        });
        return matchups;
    }

    private static SitesBlock siteBlock(LotroDeck deck, LotroCardBlueprintLibrary library) {
        Map<SitesBlock, Long> blocks = deck.getSites().stream().map(id -> {
            try {
                return library.getLotroCardBlueprint(id).getSiteBlock();
            } catch (CardNotFoundException e) {
                return null;
            }
        }).filter(Objects::nonNull).collect(Collectors.groupingBy(b -> b, Collectors.counting()));
        return blocks.entrySet().stream().max(Map.Entry.comparingByValue()).map(Map.Entry::getKey).orElse(null);
    }

    static LotroDeck loadDeck(String spec, String format, Context ctx) {
        if (spec.toLowerCase().startsWith("starter:"))
            return starterDeck(resolveStarter(spec.substring("starter:".length()), ctx), format, ctx);
        String name;
        String contents;
        switch (spec.toLowerCase()) {
            case "aragorn" -> { name = "Aragorn Starter"; contents = ARAGORN_STARTER; }
            case "gandalf" -> { name = "Gandalf Starter"; contents = GANDALF_STARTER; }
            default -> {
                Path path = Path.of(spec);
                if (!Files.exists(path))
                    die("No deck \"" + spec + "\": give aragorn, gandalf, starter:<name>, or a file path");
                name = path.getFileName().toString().replaceFirst("\\.[^.]*$", "");
                try {
                    contents = Files.readAllLines(path).stream()
                            .filter(l -> !l.trim().startsWith("#"))
                            .collect(Collectors.joining())
                            .replaceAll("\\s", "");
                } catch (IOException e) {
                    throw new UncheckedIOException(e);
                }
            }
        }
        return checked(DeckSerialization.buildDeckFromContents(name, contents, format, ""), ctx.library());
    }

    private static LotroDeck checked(LotroDeck deck, LotroCardBlueprintLibrary library) {
        String name = deck.getDeckName();
        // An unknown id otherwise surfaces mid-game as a stack trace from deep inside the engine.
        List<String> ids = new ArrayList<>(deck.getSites());
        ids.addAll(deck.getDrawDeckCards());
        if (deck.getRingBearer() != null) ids.add(deck.getRingBearer());
        if (deck.getRing() != null) ids.add(deck.getRing());
        List<String> missing = ids.stream().distinct().filter(id -> {
            try {
                library.getLotroCardBlueprint(id);
                return false;
            } catch (CardNotFoundException e) {
                return true;
            }
        }).toList();
        if (!missing.isEmpty())
            throw new IllegalArgumentException(name + ": no such card " + String.join(", ", missing));
        if (deck.getRingBearer() == null || deck.getRing() == null)
            throw new IllegalArgumentException(name + ": needs a ring-bearer and a ring (ringbearer|ring|sites|cards)");
        return deck;
    }

    /** Fixed-content products whose name says Starter, by name. Selection packs open to nothing useful. */
    private static Map<String, List<CardCollection.Item>> loadStarters(Context ctx) {
        Map<String, List<CardCollection.Item>> starters = new HashMap<>();
        new ProductLibrary(ctx.library(), ctx.resources().resolve("product").toFile()).GetAllProducts()
                .forEach((name, box) -> {
                    if (!name.toLowerCase().contains("starter"))
                        return;
                    try {
                        List<CardCollection.Item> items = box.openPack().stream()
                                .filter(i -> i.getType() == CardCollection.Item.Type.CARD).toList();
                        if (items.stream().mapToInt(CardCollection.Item::getCount).sum() >= 40)
                            starters.put(name, items);
                    } catch (RuntimeException ignored) {
                        // random or selection products: not a fixed deck
                    }
                });
        return starters;
    }

    private static String resolveStarter(String query, Context ctx) {
        List<String> matches = ctx.starters().keySet().stream()
                .filter(n -> n.toLowerCase().contains(query.toLowerCase())).sorted().toList();
        String exact = matches.stream().filter(n -> n.equalsIgnoreCase(query)).findFirst().orElse(null);
        if (exact == null && matches.size() != 1)
            die(matches.isEmpty() ? "No starter matching \"" + query + "\" -- see --list-starters"
                    : "\"" + query + "\" matches several starters:\n  " + String.join("\n  ", matches));
        return exact != null ? exact : matches.get(0);
    }

    /**
     * Builds a deck from a starter product. The product lists cards, not roles, so the roles come from the cards:
     * The One Ring is the ring, sites are the adventure deck, and the first copy of the first card that can bear
     * the ring is the ring-bearer -- the product files list the ring-bearer first.
     */
    private static LotroDeck starterDeck(String name, String format, Context ctx) {
        LotroDeck deck = new LotroDeck(name);
        deck.setTargetFormat(format);
        for (CardCollection.Item item : ctx.starters().get(name)) {
            for (int i = 0; i < item.getCount(); i++) {
                String id = item.getBlueprintId();
                try {
                    var bp = ctx.library().getLotroCardBlueprint(id);
                    if (bp.getCardType() == CardType.THE_ONE_RING && deck.getRing() == null)
                        deck.setRing(id);
                    else if (bp.getCardType() == CardType.SITE)
                        deck.addSite(id);
                    else if (deck.getRingBearer() == null && bp.canStartWithRing())
                        deck.setRingBearer(id);
                    else
                        deck.addCard(id);
                } catch (CardNotFoundException e) {
                    deck.addCard(id); // reported by checked()
                }
            }
        }
        return checked(deck, ctx.library());
    }

    /** A model directory, read as the server reads one (ModelDirectory); a directory missing a model ends the run. */
    static ModelRegistry loadModels(Path dir) throws IOException, ClassNotFoundException {
        try {
            return ModelDirectory.load(dir);
        } catch (ModelDirectory.MissingModels e) {
            die(e.getMessage());
            throw e; // not reached: die exits
        }
    }

    /** The feature set a model directory was trained on: what its features.txt says, or 1 without one. */
    static int featureVersion(Path dir) throws IOException {
        return ModelDirectory.featureVersion(dir);
    }

    // ---- command line ----

    private static Options parse(String[] args) {
        List<String> positional = new ArrayList<>();
        String suite = null;
        String format = "fotr_block";
        String kindA = "fotr";
        String kindB = "fotr";
        int games = 20;
        int jobs = Math.max(1, Runtime.getRuntime().availableProcessors() - 2);
        int maxDecisions = 1000;
        Path resources = Path.of("gemp-lotr-cards/src/main/resources");
        Path models = Path.of("bot-models");
        boolean verbose = false;
        boolean listStarters = false;
        Long seed = new SplittableRandom().nextLong();
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--suite" -> suite = args[++i];
                case "--format" -> format = args[++i];
                case "--a" -> kindA = args[++i];
                case "--b" -> kindB = args[++i];
                case "--games" -> games = Integer.parseInt(args[++i]);
                case "--jobs" -> jobs = Integer.parseInt(args[++i]);
                case "--max-decisions" -> maxDecisions = Integer.parseInt(args[++i]);
                case "--resources" -> resources = Path.of(args[++i]);
                case "--models" -> models = Path.of(args[++i]);
                case "--seed" -> {
                    String v = args[++i];
                    seed = v.equals("off") ? null : Long.valueOf(v);
                }
                case "--verbose" -> verbose = true;
                case "--list-starters" -> listStarters = true;
                case "-h", "--help" -> die(usage());
                default -> positional.add(args[i]);
            }
        }
        if (!listStarters && (suite == null) == (positional.size() != 2))
            die(usage());
        if (!Files.isDirectory(resources.resolve("cards")))
            die("No cards under " + resources.toAbsolutePath() + " -- run from gemp-lotr/, or pass --resources");
        return new Options(positional, suite, format, kindA, kindB, games, jobs, maxDecisions, resources, models,
                verbose, listStarters, seed);
    }

    private static String usage() {
        return """
                usage: BotVsBot <deck1> <deck2> [--format fotr_block] [options]
                       BotVsBot --suite starters|<file> [options]
                       BotVsBot --list-starters
                  deck: aragorn | gandalf | starter:<name> | path to a file holding ringbearer|ring|sites|cards
                  suite file: one "format | deck1 | deck2" per line; "starters" pairs every starter in its block,
                              "starters:train" / "starters:test" alternate halves of them;
                              "rr:<format>:<file>" plays every pair of the decks listed in the file
                options:
                  --a fotr --b fotr     the two bots: fotr, general, random, fotr@<dir>, general@<dir>;
                                        they swap decks every game
                  --games 20            per matchup
                  --jobs N              threads (default cores - 2)
                  --models bot-models   where the fotr bot's models are
                  --max-decisions 1000  per bot per game, then it concedes as on the live server
                  --seed N|off          the deals (default: drawn at random, printed in the summary); both games
                                        of a pair get the same one, so only the bots differ between them
                  --verbose             print the bots' own output instead of tallying it""";
    }

    static void die(String message) {
        System.err.println(message);
        System.exit(2);
    }

    /** Counts "Unknown ..." lines by bot and decision, dropping the volatile parts; swallows everything else. */
    private static final class LineTally extends OutputStream {
        private final Map<String, Integer> counts;
        private final Options opts;
        private final ThreadLocal<ByteArrayOutputStream> line = ThreadLocal.withInitial(ByteArrayOutputStream::new);

        LineTally(Map<String, Integer> counts, Options opts) {
            this.counts = counts;
            this.opts = opts;
        }

        @Override
        public void write(int b) {
            if (b != '\n') {
                line.get().write(b);
                return;
            }
            String text = line.get().toString(StandardCharsets.UTF_8);
            line.get().reset();
            String decider = BotGame.DECIDING.get();
            if (decider == null || !UNKNOWN.matcher(text).matches())
                return;
            int seat = decider.equals(A_NAME.get()) ? 0 : 1;
            randomAnswers.get()[seat]++;
            // with both seats the same kind, one tally for the kind; otherwise one per seat
            String bot = opts.kindA().equals(opts.kindB()) ? opts.kindA()
                    : seat == 0 ? "A " + opts.kindA() : "B " + opts.kindB();
            counts.merge(bot + ": " + VOLATILE.matcher(text).replaceAll("").replaceAll("\\s+", " ").trim(), 1,
                    Integer::sum);
        }
    }
}
