package com.gempukku.lotro.bots.simulation;

import com.gempukku.lotro.bots.BotDecisionFallback;
import com.gempukku.lotro.bots.BotPlayer;
import com.gempukku.lotro.game.DefaultUserFeedback;
import com.gempukku.lotro.game.LotroCardBlueprintLibrary;
import com.gempukku.lotro.game.LotroFormat;
import com.gempukku.lotro.logic.decisions.AwaitingDecision;
import com.gempukku.lotro.logic.decisions.DecisionResultInvalidException;
import com.gempukku.lotro.logic.timing.DefaultLotroGame;
import com.gempukku.lotro.logic.timing.GameResultListener;
import com.gempukku.lotro.logic.vo.LotroDeck;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.Collectors;

/**
 * One headless game between two bots, with each decision applied the way {@code LotroGameMediator.botAnswered}
 * applies it on the live server: the same {@link BotDecisionFallback} for answers the engine rejects, and the same
 * concession after too many decisions. So a result here is what a bot table would have done.
 * <p>
 * {@link FotrStartersSimulation} re-asks a rejected decision instead, which a deterministic model answers the same
 * way forever once it is off the decks it was trained on.
 */
public final class BotGame {
    /** The name of the bot whose answer is being chosen on this thread, for anyone listening to what it prints. */
    public static final ThreadLocal<String> DECIDING = new ThreadLocal<>();

    /** Called with each game this thread starts, once it is dealt: where a diagnostic attaches a listener. */
    public static final ThreadLocal<java.util.function.Consumer<DefaultLotroGame>> ON_START = new ThreadLocal<>();

    /**
     * A game that has not finished in this long is stuck inside the engine, not slow: games take a fraction of a
     * second, and the first one found spun in TurnProcedure.carryOutPendingActionsUntilDecisionNeeded for over ten
     * minutes without asking anyone anything. That loop only stops once there is a winner, so the watchdog makes one.
     */
    private static final long STUCK_SECONDS = 60;

    /**
     * Exceptions bots threw while choosing, by exception and the bot frame it came from. The game goes on with the
     * fallback's answer, as on the server, so without this a bot that throws on some decision just looks weak.
     */
    public static final Map<String, Integer> BOT_ERRORS = new java.util.concurrent.ConcurrentHashMap<>();
    private static final ScheduledExecutorService WATCHDOG = Executors.newSingleThreadScheduledExecutor(r -> {
        Thread t = new Thread(r, "bot-game-watchdog");
        t.setDaemon(true);
        return t;
    });

    /**
     * {@code winner} and {@code first} are bot names; {@code rejected} counts, by bot, answers the engine refused
     * and the fallback replaced. On a crash or stall only {@code error} and the counts are set.
     */
    public record Result(String winner, String first, String reason, int turns, int decisions,
                         Map<String, Integer> rejected, String error, List<Step> steps) {
        public boolean finished() {
            return error == null && winner != null;
        }
    }

    /**
     * One decision as the engine put it and as it was answered: who was asked, a fingerprint of the question (type,
     * text and every parameter, card ids included), and the answer the engine accepted. A seeded game replayed with
     * the same answers should ask the same questions in the same order; {@link BotReplay} checks that it does.
     */
    public record Step(String player, String question, String answer) {
    }

    static String fingerprint(AwaitingDecision decision) {
        StringBuilder sb = new StringBuilder(decision.getDecisionType() + "|" + decision.getText());
        new java.util.TreeMap<>(decision.getDecisionParameters())
                .forEach((k, v) -> sb.append('|').append(k).append('=').append(java.util.Arrays.toString(v)));
        return sb.toString();
    }

    private BotGame() {
    }

    public static Result play(LotroFormat format, LotroCardBlueprintLibrary library, List<BotPlayer> bots,
                              Map<String, LotroDeck> decks, int maxDecisions) {
        return play(format, library, bots, decks, maxDecisions, null);
    }

    /** As above; with a seed, every random draw the engine makes is fixed, so the game can be replayed. */
    public static Result play(LotroFormat format, LotroCardBlueprintLibrary library, List<BotPlayer> bots,
                              Map<String, LotroDeck> decks, int maxDecisions, Long seed) {
        return play(format, library, bots, decks, maxDecisions, seed, new java.util.ArrayList<>(), STUCK_SECONDS);
    }

    /**
     * As above, recording into {@code steps} as the game goes -- a bot that branches the game (BotTeach) reads it to
     * replay to where it is -- and with its own stuck limit, since a game whose bot runs rollouts is slow, not stuck.
     */
    public static Result play(LotroFormat format, LotroCardBlueprintLibrary library, List<BotPlayer> bots,
                              Map<String, LotroDeck> decks, int maxDecisions, Long seed, List<Step> steps,
                              long stuckSeconds) {
        // the bots' own draws, seeded apart from the engine's; put back after, as a rollout is a game inside a game
        java.util.Random botRandom = seed == null ? new java.util.Random()
                : new java.util.Random(new java.util.SplittableRandom(seed).nextLong());
        return run(() -> {
            DefaultUserFeedback feedback = new DefaultUserFeedback();
            DefaultLotroGame game = new DefaultLotroGame(format, decks, feedback, library);
            if (seed != null)
                game.getGameState().setRandomSeed(seed);
            feedback.setGame(game);
            return game;
        }, true, bots, maxDecisions, steps, stuckSeconds, new HashMap<>(), 0, botRandom);
    }

    /**
     * Plays on a game that is already waiting on a decision -- a copy made by {@link GameCopier} -- exactly as
     * {@link #play} would have from that point. {@code decisionsSoFar} and {@code totalSoFar} carry the counts the
     * game had reached (the loop guard counts from the start of the game, so a copy must too), and the steps go on
     * {@code steps}. The bots get their own Random for the length of the game, like a played game's, and the
     * thread's is put back after.
     */
    public static Result resume(DefaultLotroGame game, List<BotPlayer> bots, int maxDecisions, List<Step> steps,
                                long stuckSeconds, Map<String, Integer> decisionsSoFar, int totalSoFar,
                                java.util.Random botRandom) {
        return run(() -> game, false, bots, maxDecisions, steps, stuckSeconds, new HashMap<>(decisionsSoFar),
                totalSoFar, botRandom);
    }

    /** As {@link #resume}, with the stuck limit of a played game. */
    public static Result resume(DefaultLotroGame game, List<BotPlayer> bots, int maxDecisions, List<Step> steps,
                                Map<String, Integer> decisionsSoFar, int totalSoFar, java.util.Random botRandom) {
        return resume(game, bots, maxDecisions, steps, STUCK_SECONDS, decisionsSoFar, totalSoFar, botRandom);
    }

    private static Result run(java.util.function.Supplier<DefaultLotroGame> setup, boolean start, List<BotPlayer> bots,
                              int maxDecisions, List<Step> steps, long stuckSeconds, Map<String, Integer> decisions,
                              int totalSoFar, java.util.Random gameBotRandom) {
        Map<String, BotPlayer> byName = bots.stream().collect(Collectors.toMap(BotPlayer::getName, b -> b));
        String[] reason = {null};
        Map<String, Integer> rejected = new HashMap<>();
        int total = totalSoFar;
        AtomicBoolean stuck = new AtomicBoolean();
        ScheduledFuture<?> watchdog = null;
        java.util.Random botRandom = com.gempukku.lotro.bots.BotRandom.use(gameBotRandom);
        try {
            DefaultLotroGame game = setup.get();
            DefaultUserFeedback feedback = (DefaultUserFeedback) game.getUserFeedback();
            GameLog log = GameLog.CURRENT.get();
            if (log != null && start)
                log.bind(game); // the first game started here; copies played inside a decision are not it
            watchdog = WATCHDOG.schedule(() -> {
                if (!game.isFinished()) {
                    stuck.set(true);
                    // not thread-safe, and need not be: this game's result is thrown away
                    game.playerLost(bots.get(0).getName(), "stuck");
                }
            }, stuckSeconds, TimeUnit.SECONDS);
            game.addGameResultListener(new GameResultListener() {
                @Override
                public void gameFinished(String winner, String winReason, Map<String, String> losers) {
                    // A win by surviving or by sites sets the loser's reason to "Other player won"; a loss sets the
                    // winner's to "Last remaining player in game". The other side of each is the one that says how.
                    reason[0] = "Last remaining player in game".equals(winReason) && !losers.isEmpty()
                            ? String.join("; ", losers.values()) : winReason;
                }

                @Override
                public void gameCancelled() {
                    reason[0] = "cancelled";
                }
            });
            if (start) {
                game.startGame();
                if (ON_START.get() != null)
                    ON_START.get().accept(game);
            }

            while (!game.isFinished()) {
                BotPlayer bot = bots.stream().filter(b -> feedback.getAwaitingDecision(b.getName()) != null)
                        .findFirst().orElse(null);
                if (bot == null)
                    return new Result(null, null, null, 0, total, rejected, "stalled: game unfinished and nobody to decide", steps);
                String name = bot.getName();

                if (decisions.merge(name, 1, Integer::sum) > maxDecisions) {
                    // the live server's rule (LotroGameMediator.botAnswered)
                    game.playerLost(name, "Possible loop detected");
                    break;
                }
                total++;

                AwaitingDecision decision = feedback.getAwaitingDecision(name);
                String question = fingerprint(decision);
                String answer;
                DECIDING.set(name);
                try {
                    answer = byName.get(name).chooseAction(game.getGameState(), decision);
                } catch (RuntimeException e) {
                    answer = null; // as BotGameStateListener: the fallback picks a valid default
                    String frame = java.util.Arrays.stream(e.getStackTrace())
                            .filter(f -> f.getClassName().contains(".bots.")).findFirst()
                            .map(f -> f.getClassName().replaceAll(".*\\.", "") + ":" + f.getLineNumber()).orElse("?");
                    String key = e.getClass().getSimpleName() + " at " + frame + " (" + decision.getText() + ")";
                    if (BOT_ERRORS.merge(key, 1, Integer::sum) == 1 && System.getenv("BOT_TRACE") != null) {
                        System.err.println("== " + key);
                        e.printStackTrace(System.err);
                    }
                } finally {
                    DECIDING.remove();
                }
                feedback.participantDecided(name);
                try {
                    String accepted = BotDecisionFallback.decide(game, name, decision, answer);
                    if (!accepted.equals(answer))
                        rejected.merge(name, 1, Integer::sum);
                    steps.add(new Step(name, question, accepted));
                    if (log != null)
                        log.decided(game, name, decision, accepted);
                } catch (DecisionResultInvalidException e) {
                    game.playerLost(name, "Invalid decision");
                    break;
                }
                game.carryOutPendingActionsUntilDecisionNeeded();
            }

            if (stuck.get())
                return new Result(null, null, null, 0, total, rejected, "engine stuck: no decision asked for "
                        + stuckSeconds + "s, turn " + game.getGameState().getTurnNumber() + ", "
                        + game.getGameState().getCurrentPhase(), steps);
            var state = game.getGameState();
            return new Result(game.getWinnerPlayerId(), state.getFirstPlayerId(), reason[0], state.getTurnNumber(),
                    total, rejected, null, steps);
        } catch (RuntimeException e) {
            StringWriter trace = new StringWriter();
            e.printStackTrace(new PrintWriter(trace));
            String where = trace.toString().lines().skip(1).limit(3).map(String::trim).collect(Collectors.joining(" < "));
            return new Result(null, null, null, 0, total, rejected, e + " at " + where, steps);
        } finally {
            if (watchdog != null)
                watchdog.cancel(false);
            com.gempukku.lotro.bots.BotRandom.use(botRandom);
        }
    }
}
