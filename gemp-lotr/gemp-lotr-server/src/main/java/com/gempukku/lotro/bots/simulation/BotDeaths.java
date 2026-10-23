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
 * <p>
 * For a death in a skirmish it also says who put the minions on the Ring-bearer -- the Free Peoples player's own
 * assignment, the Shadow player assigning what was left, or neither (a card effect) -- and, at the Free Peoples
 * assignment that turn, how many other characters it left without a minion and how many minions it left for the
 * Shadow player; and whether a card action that grants defender was offered to it that turn, and used.
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
        Set<PhysicalCard> skirmishMinionCards = Set.of();
        /** Every assignment decision answered, in order. */
        final List<Assign> assigns = new ArrayList<>();
        /** Card actions granting defender offered to the Free Peoples player: turn, and whether one was taken. */
        final List<int[]> defenderOffers = new ArrayList<>();
    }

    record Assign(int turn, boolean fp, List<String> free, List<String> minions, Map<String, List<String>> answer,
                  String bearer, String phase) {
    }

    /** Every game's Free Peoples assignments: [count, a minion on the Ring-bearer, ...while another was idle]. */
    static final long[] ALL_FP_ASSIGNS = new long[3];

    /** Records what a bot is asked and answers, for the death report. */
    record Recorder(com.gempukku.lotro.bots.BotPlayer inner, Log log) implements com.gempukku.lotro.bots.BotPlayer {
        @Override
        public String chooseAction(com.gempukku.lotro.game.state.GameState gs,
                                   com.gempukku.lotro.logic.decisions.AwaitingDecision d) {
            String answer = inner.chooseAction(gs, d);
            try {
                boolean fp = getName().equals(gs.getCurrentPlayerId());
                switch (d.getDecisionType()) {
                    case ASSIGN_MINIONS -> {
                        var p = d.getDecisionParameters();
                        var bearer = gs.getRingBearer(gs.getCurrentPlayerId());
                        log.assigns.add(new Assign(gs.getTurnNumber(), fp,
                                p.get("freeCharacters") == null ? List.of() : List.of(p.get("freeCharacters")),
                                p.get("minions") == null ? List.of() : List.of(p.get("minions")),
                                com.gempukku.lotro.bots.rl.fotrstarters.models.assignment.AssignmentProbe.parse(answer),
                                bearer == null ? "" : String.valueOf(bearer.getCardId()), String.valueOf(gs.getCurrentPhase())));
                    }
                    case CARD_ACTION_CHOICE -> {
                        String[] ids = d.getDecisionParameters().get("actionId"), cards = d.getDecisionParameters().get("cardId");
                        if (!fp || ids == null || cards == null)
                            break;
                        boolean offered = false, taken = false;
                        for (int i = 0; i < ids.length && i < cards.length; i++) {
                            var card = gs.findCardById(Integer.parseInt(cards[i]));
                            if (card == null || card.getBlueprint().getJsonDefinition() == null
                                    || !card.getBlueprint().getJsonDefinition().toString().toLowerCase().contains("defender"))
                                continue;
                            offered = true;
                            taken |= ids[i].equals(answer);
                        }
                        if (offered)
                            log.defenderOffers.add(new int[]{gs.getTurnNumber(), taken ? 1 : 0});
                    }
                    default -> {
                    }
                }
            } catch (RuntimeException ignored) {
                // a decision the record cannot read; the answer stands
            }
            return answer;
        }

        @Override
        public String getName() {
            return inner.getName();
        }
    }

    static final List<String> ALL_AT_SHADOW = java.util.Collections.synchronizedList(new ArrayList<>());
    static final List<Integer> ALL_MOVED_AGAIN = java.util.Collections.synchronizedList(new ArrayList<>());

    record Death(String phase, boolean inSkirmish, int minions, int companionsLeft, int turn, int site,
                 List<String> tail, String assignedBy, String leftOver, String defender) {
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
                            List.of(new Recorder(BotVsBot.bot(ka, "~a", ctx), log), new Recorder(BotVsBot.bot(kb, "~b", ctx), log)),
                            Map.of("~a", swap ? d2 : d1, "~b", swap ? d1 : d2), 1000);
                    BotGame.ON_START.remove();
                    ALL_AT_SHADOW.addAll(log.atShadow);
                    for (Assign as : log.assigns)
                        if (as.fp() && as.free().contains(as.bearer()) && as.free().size() > 1) {
                            boolean onBearer = !as.answer().getOrDefault(as.bearer(), List.of()).isEmpty();
                            boolean idle = as.free().stream().anyMatch(c -> !c.equals(as.bearer())
                                    && as.answer().getOrDefault(c, List.of()).isEmpty());
                            synchronized (ALL_FP_ASSIGNS) {
                                ALL_FP_ASSIGNS[0]++;
                                ALL_FP_ASSIGNS[1] += onBearer ? 1 : 0;
                                ALL_FP_ASSIGNS[2] += onBearer && idle ? 1 : 0;
                            }
                        }
                    ALL_MOVED_AGAIN.addAll(log.movedAgainWith);
                    if (!r.finished() || r.reason() == null || !r.reason().contains(System.getenv().getOrDefault("DEATH_REASON", "Ring-Bearer is dead")))
                        return null;
                    var gs = holder[0].getGameState();
                    String loser = r.winner().equals("~a") ? "~b" : "~a";
                    long companions = gs.getInPlay().stream().filter(c -> c.getOwner().equals(loser)
                            && c.getBlueprint().getCardType() == com.gempukku.lotro.common.CardType.COMPANION).count();
                    List<String> tail = log.lines.subList(Math.max(0, log.lines.size() - (System.getenv("DEATH_FULL") != null ? 100000 : 40)), log.lines.size());
                    String[] why = assignment(log, gs.getTurnNumber());
                    return new Death(log.phase, log.skirmishFp != null, log.skirmishMinions, (int) companions,
                            gs.getTurnNumber(), gs.getCurrentSiteNumber(), new ArrayList<>(tail), why[0], why[1], why[2]);
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
        out.printf("  every game's Free Peoples assignments with the Ring-bearer and another free: %d; a minion on the "
                        + "Ring-bearer %.1f%%, while another was idle %.1f%% (people at July's: 9.6%%, 6.6%%)%n",
                ALL_FP_ASSIGNS[0], 100.0 * ALL_FP_ASSIGNS[1] / Math.max(1, ALL_FP_ASSIGNS[0]),
                100.0 * ALL_FP_ASSIGNS[2] / Math.max(1, ALL_FP_ASSIGNS[0]));
        tally(out, "phase it died in", deaths.stream().map(Death::phase).toList());
        tally(out, "in a skirmish, against", deaths.stream()
                .map(d -> d.inSkirmish() ? d.minions() + " minion(s)" : "not in a skirmish").toList());
        tally(out, "companions left in play", deaths.stream().map(d -> String.valueOf(d.companionsLeft())).toList());
        tally(out, "in a skirmish: who put the minions on the Ring-bearer", deaths.stream()
                .filter(Death::inSkirmish).map(Death::assignedBy).toList());
        tally(out, "in a skirmish, at the Free Peoples assignment that turn: other characters left without a minion / "
                + "minions left for the Shadow player", deaths.stream().filter(Death::inSkirmish).map(Death::leftOver).toList());
        tally(out, "in a skirmish, put there by the Shadow player: characters idle / minions left", deaths.stream()
                .filter(d -> d.inSkirmish() && d.assignedBy().equals("Shadow player")).map(Death::leftOver).toList());
        tally(out, "a card action granting defender that turn (Free Peoples)", deaths.stream()
                .filter(Death::inSkirmish).map(Death::defender).toList());
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

    /**
     * Who put the minions the Ring-bearer died to on it, what the Free Peoples assignment that turn left over (other
     * characters with no minion / minions unassigned), and whether a defender action was offered and used that turn.
     */
    private static String[] assignment(Log log, int turn) {
        String bearer = log.skirmishFp == null ? null : String.valueOf(log.skirmishFp.getCardId());
        boolean byFp = false, byShadow = false;
        int idle = -1, left = -1;
        for (Assign a : log.assigns) {
            if (a.turn() != turn)
                continue;
            List<String> onBearer = bearer == null ? List.of() : a.answer().getOrDefault(bearer, List.of());
            for (PhysicalCard m : log.skirmishMinionCards)
                if (onBearer.contains(String.valueOf(m.getCardId()))) {
                    byFp |= a.fp();
                    byShadow |= !a.fp();
                }
            if (a.fp() && idle < 0) {
                idle = 0;
                for (String c : a.free())
                    if (!c.equals(bearer) && a.answer().getOrDefault(c, List.of()).isEmpty())
                        idle++;
                Set<String> assigned = new HashSet<>();
                a.answer().values().forEach(assigned::addAll);
                left = (int) a.minions().stream().filter(m -> !assigned.contains(m)).count();
            }
        }
        String by = byFp && byShadow ? "both" : byFp ? "Free Peoples player" : byShadow ? "Shadow player"
                : "neither (a card effect, or an earlier round)";
        String leftOver = idle < 0 ? "no Free Peoples assignment that turn"
                : Math.min(idle, 3) + (idle >= 3 ? "+" : "") + " idle / " + Math.min(left, 3) + (left >= 3 ? "+" : "") + " left";
        boolean offered = false, taken = false;
        for (int[] o : log.defenderOffers)
            if (o[0] == turn) {
                offered = true;
                taken |= o[1] == 1;
            }
        String defender = !offered ? "none offered" : taken ? "offered, used" : "offered, not used";
        return new String[]{by, leftOver, defender};
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
                            @SuppressWarnings("unchecked")
                            Set<PhysicalCard> minions = new HashSet<>((Set<PhysicalCard>) args[1]);
                            log.skirmishMinionCards = minions;
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
