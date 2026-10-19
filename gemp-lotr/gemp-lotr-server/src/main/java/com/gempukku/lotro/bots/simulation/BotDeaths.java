package com.gempukku.lotro.bots.simulation;

import com.gempukku.lotro.communication.GameStateListener;
import com.gempukku.lotro.game.PhysicalCard;
import com.gempukku.lotro.logic.timing.DefaultLotroGame;
import com.gempukku.lotro.logic.vo.LotroDeck;

import java.io.OutputStream;
import java.io.PrintStream;
import java.lang.reflect.Proxy;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.*;

/**
 * How do Ring-bearers die? Plays games between two bots with the game log recorded -- the lines a player reads in
 * GEMP's chat, phases, assignments and skirmishes -- and for every game lost to a dead Ring-bearer reports the phase
 * it died in, whether it was in a skirmish and against how many minions, and how many companions its fellowship
 * still had. {@code --dump <file>} writes the last lines of the first games' logs, for reading.
 * <pre>
 *   BOT_MAIN=com.gempukku.lotro.bots.simulation.BotDeaths ./bot-vs-bot.sh --suite decks/mine-vs-shadows.txt --games 2
 * </pre>
 */
public class BotDeaths {
    /** What the log said, as it happened. */
    static final class Log {
        final List<String> lines = new ArrayList<>();
        String phase = "?";
        PhysicalCard skirmishFp;
        int skirmishMinions;
        final Map<Integer, Integer> assignedBy = new HashMap<>(); // fp card id -> minions assigned to it
        /** At each Shadow phase: "companions in play / companions in the Free Peoples player's hand". */
        final List<String> atShadow = new ArrayList<>();
        /** Companions in play each time the Free Peoples player chose to move again. */
        final List<Integer> movedAgainWith = new ArrayList<>();
    }

    static final List<String> ALL_AT_SHADOW = java.util.Collections.synchronizedList(new ArrayList<>());
    static final List<Integer> ALL_MOVED_AGAIN = java.util.Collections.synchronizedList(new ArrayList<>());

    record Death(String phase, boolean inSkirmish, int minions, int companionsLeft, int turn, int site,
                 List<String> tail) {
    }

    public static void main(String[] args) throws Exception {
        String suite = "decks/mine-vs-shadows.txt", a = "general@bot-models-exp/v2", b = a;
        int games = 2;
        Path dump = null;
        int jobs = Math.max(1, Runtime.getRuntime().availableProcessors() - 2);
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--suite" -> suite = args[++i];
                case "--games" -> games = Integer.parseInt(args[++i]);
                case "--a" -> a = args[++i];
                case "--b" -> b = args[++i];
                case "--dump" -> dump = Path.of(args[++i]);
                case "--jobs" -> jobs = Integer.parseInt(args[++i]);
                default -> BotVsBot.die("usage: BotDeaths [--suite file] [--games 2] [--a bot] [--b bot] [--dump file]");
            }
        }
        PrintStream out = System.out;
        System.setOut(new PrintStream(OutputStream.nullOutputStream()));
        BotVsBot.Context ctx = BotVsBot.context(Path.of("gemp-lotr-cards/src/main/resources"), Path.of("bot-models"));
        List<BotVsBot.Matchup> matchups = BotVsBot.loadSuite(suite, ctx);
        ExecutorService pool = Executors.newFixedThreadPool(jobs);
        List<Future<Death>> futures = new ArrayList<>();
        String ka = a, kb = b;
        int[] ended = new int[1];
        for (BotVsBot.Matchup m : matchups) {
            LotroDeck d1 = BotVsBot.loadDeck(m.deck1(), m.format(), ctx), d2 = BotVsBot.loadDeck(m.deck2(), m.format(), ctx);
            for (int g = 0; g < games; g++) {
                boolean swap = g % 2 == 1;
                futures.add(pool.submit(() -> {
                    Log log = new Log();
                    DefaultLotroGame[] holder = new DefaultLotroGame[1];
                    BotGame.ON_START.set(game -> {
                        BotGame.ON_START.remove(); // the first game only: not the rollouts inside it
                        holder[0] = game;
                        game.addGameStateListener("~log", listener(log, game));
                    });
                    BotGame.Result r = BotGame.play(ctx.formats().getFormat(m.format()), ctx.library(),
                            List.of(BotVsBot.bot(ka, "~a", ctx), BotVsBot.bot(kb, "~b", ctx)),
                            Map.of("~a", swap ? d2 : d1, "~b", swap ? d1 : d2), 1000);
                    BotGame.ON_START.remove();
                    ALL_AT_SHADOW.addAll(log.atShadow);
                    ALL_MOVED_AGAIN.addAll(log.movedAgainWith);
                    if (!r.finished() || r.reason() == null || !r.reason().contains(System.getenv().getOrDefault("DEATH_REASON", "Ring-Bearer is dead")))
                        return null;
                    var gs = holder[0].getGameState();
                    String loser = r.winner().equals("~a") ? "~b" : "~a";
                    long companions = gs.getInPlay().stream().filter(c -> c.getOwner().equals(loser)
                            && c.getBlueprint().getCardType() == com.gempukku.lotro.common.CardType.COMPANION).count();
                    List<String> tail = log.lines.subList(Math.max(0, log.lines.size() - (System.getenv("DEATH_FULL") != null ? 100000 : 40)), log.lines.size());
                    return new Death(log.phase, log.skirmishFp != null, log.skirmishMinions, (int) companions,
                            gs.getTurnNumber(), gs.getCurrentSiteNumber(), new ArrayList<>(tail));
                }));
            }
        }
        List<Death> deaths = new ArrayList<>();
        int total = futures.size();
        for (Future<Death> f : futures) {
            Death d = f.get(15, TimeUnit.MINUTES);
            if (d != null)
                deaths.add(d);
        }
        pool.shutdown();
        out.printf("%d games, %d lost to a dead Ring-bearer%n", total, deaths.size());
        tally(out, "phase it died in", deaths.stream().map(Death::phase).toList());
        tally(out, "in a skirmish, against", deaths.stream()
                .map(d -> d.inSkirmish() ? d.minions() + " minion(s)" : "not in a skirmish").toList());
        tally(out, "companions left in play", deaths.stream().map(d -> String.valueOf(d.companionsLeft())).toList());
        tally(out, "every game, at each Shadow phase: Free Peoples companions (4 = 4+, 3 = 3+ in hand)", ALL_AT_SHADOW);
        tally(out, "every game, companions in play when the Free Peoples player chose to move again",
                ALL_MOVED_AGAIN.stream().map(String::valueOf).toList());
        tally(out, "site", deaths.stream().map(d -> String.valueOf(d.site())).toList());
        if (dump != null) {
            StringBuilder sb = new StringBuilder();
            for (Death d : deaths.subList(0, Math.min(25, deaths.size()))) {
                sb.append("==== died in ").append(d.phase()).append(", site ").append(d.site()).append(", turn ")
                        .append(d.turn()).append(", ").append(d.companionsLeft()).append(" companions left\n");
                d.tail().forEach(l -> sb.append("  ").append(l).append('\n'));
            }
            Files.writeString(dump, sb.toString());
            out.println("logs in " + dump);
        }
        System.exit(0);
    }

    private static void tally(PrintStream out, String title, List<String> values) {
        out.println("  " + title + ":");
        values.stream().collect(java.util.stream.Collectors.groupingBy(v -> v, TreeMap::new,
                java.util.stream.Collectors.counting())).forEach((k, n) -> out.printf("    %5d  %s%n", n, k));
    }

    /** Records the log; every other notification is ignored. */
    private static GameStateListener listener(Log log, DefaultLotroGame game) {
        return (GameStateListener) Proxy.newProxyInstance(GameStateListener.class.getClassLoader(),
                new Class<?>[]{GameStateListener.class}, (proxy, method, args) -> {
                    switch (method.getName()) {
                        case "sendMessage" -> {
                            String line = strip((String) args[0]);
                            log.lines.add("[" + log.phase + "] " + line);
                            if (line.contains("decides to move again"))
                                log.movedAgainWith.add(companions(game, false));
                        }
                        case "setCurrentPhase" -> {
                            log.phase = (String) args[0];
                            if ("Shadow".equals(args[0]))
                                log.atShadow.add(Math.min(companions(game, false), 4) + " in play / "
                                        + Math.min(companions(game, true), 3) + " in hand");
                            log.lines.add("--- " + args[0] + " (turn " + game.getGameState().getTurnNumber()
                                    + ", " + game.getGameState().getCurrentPlayerId() + ")");
                        }
                        case "startSkirmish" -> {
                            log.skirmishFp = (PhysicalCard) args[0];
                            log.skirmishMinions = ((Set<?>) args[1]).size();
                            log.lines.add("    skirmish: " + name((PhysicalCard) args[0]) + " vs "
                                    + ((Set<?>) args[1]).stream().map(c -> name((PhysicalCard) c)).toList());
                        }
                        case "finishSkirmish" -> log.skirmishFp = null;
                        case "addAssignment" -> log.lines.add("    assigned: " + name((PhysicalCard) args[0]) + " <- "
                                + ((Set<?>) args[1]).stream().map(c -> name((PhysicalCard) c)).toList());
                        case "isLiveConnection" -> {
                            return false;
                        }
                        case "getAssignedPlayerId" -> {
                            return "~log";
                        }
                        case "hashCode" -> {
                            return System.identityHashCode(proxy);
                        }
                        case "equals" -> {
                            return proxy == args[0];
                        }
                        default -> {
                        }
                    }
                    return null;
                });
    }

    /** The current (Free Peoples) player's companions, in play or in hand; capped buckets are done by the caller. */
    private static int companions(DefaultLotroGame game, boolean hand) {
        var gs = game.getGameState();
        String fp = gs.getCurrentPlayerId();
        var cards = hand ? gs.getHand(fp) : gs.getInPlay();
        return (int) cards.stream().filter(c -> c.getOwner().equals(fp)
                && c.getBlueprint().getCardType() == com.gempukku.lotro.common.CardType.COMPANION).count();
    }

    private static String name(PhysicalCard c) {
        return c.getBlueprint().getTitle() + "(" + c.getOwner() + ")";
    }

    private static String strip(String s) {
        return s.replaceAll("<[^>]*>", "");
    }
}
