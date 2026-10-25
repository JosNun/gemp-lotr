package com.gempukku.lotro.bots.simulation;

import com.gempukku.lotro.bots.BotDecisionFallback;
import com.gempukku.lotro.bots.BotPlayer;
import com.gempukku.lotro.game.PhysicalCard;
import com.gempukku.lotro.game.state.GameState;
import com.gempukku.lotro.logic.decisions.AwaitingDecision;
import com.gempukku.lotro.logic.decisions.AwaitingDecisionType;
import com.gempukku.lotro.logic.timing.DefaultLotroGame;

import java.io.PrintWriter;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.atomic.AtomicLong;

/**
 * The bot against people, on people's own games: at a decision of a re-created human game (HumanReplay), the bot is
 * asked what it would do; where that differs from what the human did, both answers are played on copies of the game
 * with the bot finishing it for both players, {@link #ROLLOUTS} times each on the same reshuffled futures, and the
 * deciding player's wins compared. Over many decisions: which kinds of decision the bot gets wrong where people get
 * them right, and the reverse. The continuation is the bot's, so this credits answers that suit the bot's own play;
 * with that caveat it is the nearest thing to the bot against people with no person at the table.
 * <pre>
 *   ./replay.sh HumanReplay --manifests ~/dev/gemp-replays/manifests-2022-11/movie --jobs 8 \
 *       --counterfactual safe@bot-models-exp/d4a-lwrc-d5 --cf-out cf.tsv [--cf-rate 0.1] [--cf-rollouts 8]
 * </pre>
 */
final class Counterfactual {
    static volatile String KIND;
    static volatile double RATE = 0.1;
    static volatile int ROLLOUTS = 8;
    static volatile PrintWriter OUT;
    /** --cf-only: test only the decisions whose bucket matches (e.g. "play or Pass" for skirmishes); null for all. */
    static volatile java.util.regex.Pattern ONLY;
    /** --cf-offer: test only card-action decisions with an option whose text matches (e.g. "The One Ring"). */
    static volatile java.util.regex.Pattern OFFER;
    /** --cf-phases ASSIGNMENT,SKIRMISH: test only decisions asked in these phases (the fierce round is in them too). */
    static volatile Set<String> PHASES;
    /**
     * --cf-deaths: where the bot differs, play each answer on only to the end of the turn and count the Free Peoples
     * player's companions killed and whether its Ring-bearer died -- what a skirmish decision costs in companions, the
     * bots losing 30% more per skirmish than people. Rows: game, player, bucket, type, human, bot, companions killed
     * after the human's answer and after the bot's (summed over the play-outs), play-outs, side, Ring-bearers killed
     * after each, phase, turn.
     */
    static volatile boolean DEATHS;
    /**
     * --deaths-teach &lt;file&gt;: at every Shadow skirmish action prompt of the re-created games, each option played on
     * to the end of the turn {@link #DEATHS_ROLLOUTS} times by the bot ({@code --counterfactual} names it) and scored
     * by the Free Peoples companions killed, a dead Ring-bearer counted {@link #BEARER_WEIGHT} times; the best labelled
     * good, those a companion or more worse labelled bad, written as steps with the skirmish probe's numbers. The bots'
     * Shadow skirmish answers killed 0.069 fewer companions than people's where they differed: a teacher for that.
     */
    static volatile PrintWriter DEATHS_TEACH;
    static volatile int DEATHS_ROLLOUTS = 6;
    static final int BEARER_WEIGHT = 5;
    /**
     * --deaths-sides shadow,fp: whose skirmish prompts the deaths teacher labels. The Shadow side's (the default)
     * maximise the companions killed; the Free Peoples side's minimise the same score, their own companions lost.
     */
    static volatile java.util.Set<String> DEATHS_SIDES = java.util.Set.of("shadow");
    /**
     * --deaths-value &lt;weights&gt;: score each option by the win model ({@code --move-value}'s format) at the start of
     * the deciding player's next turn, through the opponent's turn, instead of by companions killed this turn; the
     * best labelled good, those {@link #DEATHS_VALUE_MARGIN} or more below it bad. Companions lost alone leave out what
     * a skirmish play costs -- exertions above all -- and labelled one way: Free Peoples labels by them lost (47.0%).
     */
    static volatile boolean DEATHS_BY_VALUE = false;
    /**
     * --deaths-prompts &lt;regex&gt;: the card-action prompts the deaths teacher labels, instead of the skirmish's
     * ("Choose action to play or Pass"). Against people's positions the bots' Free Peoples archery answers lost 0.08 of a
     * companion more than people's where they differed, the largest gap of any kind of decision.
     */
    static volatile java.util.regex.Pattern DEATHS_PROMPTS;
    static volatile double DEATHS_VALUE_MARGIN = 0.02;
    static final java.util.concurrent.atomic.AtomicLong deathsTaught = new java.util.concurrent.atomic.AtomicLong();
    static final AtomicLong asked = new AtomicLong(), agreed = new AtomicLong(), compared = new AtomicLong();

    private Counterfactual() {
    }

    /** Before the human's answer is adopted: the bot's own, and if it differs, the two played out. */
    static void consider(DefaultLotroGame live, GameCopier copier, BotVsBot.Context ctx, String gameName,
                         List<String> players, String player, AwaitingDecision decision, String humanAnswer,
                         Random sampler) {
        if (KIND == null)
            return;
        GameState gs = live.getGameState();
        if (ONLY != null && !ONLY.matcher(bucket(gs, decision, player)).find())
            return;
        if (OFFER != null && !offers(decision, OFFER))
            return;
        if (PHASES != null && !PHASES.contains(String.valueOf(gs.getCurrentPhase())))
            return;
        if (sampler.nextDouble() >= RATE)
            return;
        String botAnswer;
        try {
            BotPlayer bot = BotVsBot.bot(KIND, player, ctx);
            botAnswer = bot.chooseAction(gs, decision);
        } catch (RuntimeException e) {
            return;
        }
        if (botAnswer == null)
            return;
        asked.incrementAndGet();
        String bucket = bucket(gs, decision, player);
        if (same(gs, decision, humanAnswer, botAnswer)) {
            agreed.incrementAndGet();
            write(gs, gameName, player, bucket, decision, humanAnswer, botAnswer, -1, -1, 0);
            return;
        }
        if (DEATHS) {
            int[] h = new int[2], b = new int[2];
            int n = 0;
            long base = sampler.nextLong();
            for (int r = 0; r < ROLLOUTS; r++) {
                long seed = base + r * 0x9E3779B97F4A7C15L;
                int[] dh = turnDeaths(live, copier, ctx, players, player, humanAnswer, seed);
                int[] db = turnDeaths(live, copier, ctx, players, player, botAnswer, seed);
                if (dh == null || db == null)
                    continue;
                n++;
                h[0] += dh[0];
                h[1] += dh[1];
                b[0] += db[0];
                b[1] += db[1];
            }
            if (n == 0)
                return;
            compared.incrementAndGet();
            PrintWriter out = OUT;
            if (out != null)
                synchronized (Counterfactual.class) {
                    out.println(String.join("\t", gameName, player, bucket, decision.getDecisionType().name(),
                            humanAnswer.replace('\t', ' '), botAnswer.replace('\t', ' '), String.valueOf(h[0]),
                            String.valueOf(b[0]), String.valueOf(n),
                            player.equals(gs.getCurrentPlayerId()) ? "FP" : "Shadow", String.valueOf(h[1]),
                            String.valueOf(b[1]), String.valueOf(gs.getCurrentPhase()), String.valueOf(gs.getTurnNumber())));
                    out.flush();
                }
            return;
        }
        if (ROLLOUTS == 0) {
            // --cf-rollouts 0: only whether the bot chose as the human did -- how human-like it is, at the cost of
            // replaying the games
            write(gs, gameName, player, bucket, decision, humanAnswer, botAnswer, -1, -1, -1);
            return;
        }
        int humanWins = 0, botWins = 0, n = 0;
        long seedBase = sampler.nextLong();
        for (int r = 0; r < ROLLOUTS; r++) {
            long seed = seedBase + r * 0x9E3779B97F4A7C15L;
            Boolean h = playOut(live, copier, ctx, players, player, humanAnswer, seed);
            Boolean b = playOut(live, copier, ctx, players, player, botAnswer, seed);
            if (h == null || b == null)
                continue;
            n++;
            humanWins += h ? 1 : 0;
            botWins += b ? 1 : 0;
        }
        if (n == 0)
            return;
        compared.incrementAndGet();
        write(gs, gameName, player, bucket, decision, humanAnswer, botAnswer, humanWins, botWins, n);
    }

    private static boolean offers(AwaitingDecision d, java.util.regex.Pattern p) {
        String[] texts = d.getDecisionParameters().get("actionText");
        if (d.getDecisionType() != AwaitingDecisionType.CARD_ACTION_CHOICE || texts == null)
            return false;
        for (String t : texts)
            if (t != null && p.matcher(t).find())
                return true;
        return false;
    }

    /** At an assignment: the Ring-bearer's id, then the free characters' ids ("rb;id id id"); empty otherwise. */
    private static String assignmentContext(GameState gs, AwaitingDecision d) {
        if (d.getDecisionType() != AwaitingDecisionType.ASSIGN_MINIONS)
            return "";
        try {
            var bearer = gs.getRingBearer(gs.getCurrentPlayerId());
            String[] free = d.getDecisionParameters().get("freeCharacters");
            return (bearer == null ? "" : String.valueOf(bearer.getCardId())) + ";"
                    + (free == null ? "" : String.join(" ", free));
        } catch (RuntimeException e) {
            return "";
        }
    }

    /**
     * --move-teach &lt;file&gt;: at every "another move?" decision of the re-created games, each answer played on a copy
     * through the end of the opponent's next turn {@link #MOVE_ROLLOUTS} times by the bot, and the position scored by
     * a small win model fit on people's games ({@code --move-value}, scripts/race_value.py: "name weight" lines,
     * "cube9:name weight" for (name / 9)^3, and "intercept b", over set-2 state features from the deciding player's side); a finished game scores 1 or 0. Deaths and the race are weighed
     * as people's results weigh them, and the opponent's turn shows whether moving on forced a double move. The better
     * answer is labelled where it scores {@link #MOVE_MARGIN} more; every decision's scores go to &lt;file&gt;.tsv.
     */
    static volatile PrintWriter MOVE_TEACH, MOVE_LOG;
    static volatile int MOVE_ROLLOUTS = 8;
    static volatile double MOVE_MARGIN = 0.02;
    static volatile int[] MOVE_VALUE_INDEX;
    static volatile double[] MOVE_VALUE_WEIGHT;
    static volatile double MOVE_VALUE_INTERCEPT;
    static final java.util.concurrent.atomic.AtomicLong movesTaught = new java.util.concurrent.atomic.AtomicLong(),
            movesScored = new java.util.concurrent.atomic.AtomicLong();

    static void loadMoveValue(Path file) throws java.io.IOException {
        List<Integer> index = new ArrayList<>();
        List<Double> weight = new ArrayList<>();
        for (String line : java.nio.file.Files.readAllLines(file)) {
            String[] f = line.trim().split("\\s+");
            if (f.length != 2)
                continue;
            if (f[0].equals("intercept")) {
                MOVE_VALUE_INTERCEPT = Double.parseDouble(f[1]);
                continue;
            }
            // "cube9:name" is (name / 9)^3: a site's worth grows along the path (people's games: a site at 3 is worth 0.2
            // of a companion, at 8 2.6); stored as a negative index
            boolean cube = f[0].startsWith("cube9:");
            int i = com.gempukku.lotro.bots.rl.fotrstarters.GeneralStateFeatures.index(cube ? f[0].substring(6) : f[0]);
            index.add(cube ? -1 - i : i);
            weight.add(Double.parseDouble(f[1]));
        }
        MOVE_VALUE_INDEX = index.stream().mapToInt(Integer::intValue).toArray();
        MOVE_VALUE_WEIGHT = weight.stream().mapToDouble(Double::doubleValue).toArray();
    }

    static void teachMove(DefaultLotroGame live, GameCopier copier, BotVsBot.Context ctx, List<String> players,
                          String player, AwaitingDecision decision, double[] state, String humanAnswer, Random sampler) {
        PrintWriter out = MOVE_TEACH;
        if (out == null || state == null || decision.getDecisionType() != AwaitingDecisionType.MULTIPLE_CHOICE
                || decision.getText() == null || !decision.getText().toLowerCase().contains("another move"))
            return;
        String[] results = decision.getDecisionParameters().get("results");
        if (results == null || results.length < 2)
            return;
        double[] value = new double[results.length];
        long base = sampler.nextLong();
        for (int i = 0; i < results.length; i++) {
            int n = 0;
            for (int r = 0; r < MOVE_ROLLOUTS; r++) {
                Double v = roundValue(live, copier, ctx, players, player, String.valueOf(i), base + r * 0x9E3779B97F4A7C15L);
                if (v == null)
                    continue;
                n++;
                value[i] += v;
            }
            if (n == 0)
                return;
            value[i] /= n;
        }
        movesScored.incrementAndGet();
        int best = 0, worst = 0;
        for (int i = 1; i < value.length; i++) {
            if (value[i] > value[best])
                best = i;
            if (value[i] < value[worst])
                worst = i;
        }
        GameState gs = live.getGameState();
        StringBuilder log = new StringBuilder(String.valueOf(gs.getTurnNumber())).append('\t').append(humanAnswer);
        for (int i = 0; i < results.length; i++)
            log.append('\t').append(results[i]).append('=').append(String.format("%.4f", value[i]));
        String line = null;
        if (value[best] - value[worst] >= MOVE_MARGIN) {
            var step = new com.gempukku.lotro.bots.rl.LearningStep(state,
                    new com.gempukku.lotro.bots.rl.semanticaction.MultipleChoiceAction(results[best]), true, decision);
            step.reward = 10;
            line = step.toJson().toString();
            movesTaught.incrementAndGet();
        }
        synchronized (Counterfactual.class) {
            if (line != null) {
                out.println(line);
                out.flush();
            }
            if (MOVE_LOG != null) {
                MOVE_LOG.println(log);
                MOVE_LOG.flush();
            }
        }
    }

    /** {@code answer} given on a copy, played through the opponent's next turn; the win model's score for {@code player}. */
    private static Double roundValue(DefaultLotroGame live, GameCopier copier, BotVsBot.Context ctx, List<String> players,
                                     String player, String answer, long seed) {
        try {
            DefaultLotroGame copy = copier.copy(live);
            GameState gs = copy.getGameState();
            gs.setDeckShuffler(null);
            gs.setRandomSeed(seed);
            for (String p : players)
                gs.shuffleDeck(p);
            int turn = gs.getTurnNumber();
            var feedback = GameCopier.feedback(copy);
            AwaitingDecision pending = feedback.getAwaitingDecision(player);
            if (pending == null)
                return null;
            feedback.participantDecided(player);
            BotDecisionFallback.decide(copy, player, pending, answer);
            copy.carryOutPendingActionsUntilDecisionNeeded();
            Map<String, BotPlayer> bots = new HashMap<>();
            for (String p : players)
                bots.put(p, BotVsBot.bot(KIND, p, ctx));
            com.gempukku.lotro.bots.BotRandom.current().setSeed(seed ^ 0x5DEECE66DL);
            AwaitingDecision last = null;
            for (int step = 0; step < 6000 && !copy.isFinished() && copy.getGameState().getTurnNumber() <= turn + 1; step++) {
                String who = null;
                AwaitingDecision d = null;
                for (String p : players)
                    if ((d = feedback.getAwaitingDecision(p)) != null) {
                        who = p;
                        break;
                    }
                if (d == null)
                    return null;
                last = d;
                String a = bots.get(who).chooseAction(copy.getGameState(), d);
                feedback.participantDecided(who);
                BotDecisionFallback.decide(copy, who, d, a);
                copy.carryOutPendingActionsUntilDecisionNeeded();
            }
            if (copy.isFinished())
                return player.equals(copy.getWinnerPlayerId()) ? 1.0 : 0.0;
            AwaitingDecision now = null;
            for (String p : players)
                if ((now = feedback.getAwaitingDecision(p)) != null)
                    break;
            double[] x = HumanReplay.valueState(copy, now != null ? now : last, player);
            if (x == null)
                return null;
            double z = MOVE_VALUE_INTERCEPT;
            for (int k = 0; k < MOVE_VALUE_INDEX.length; k++) {
                int i = MOVE_VALUE_INDEX[k];
                z += MOVE_VALUE_WEIGHT[k] * (i >= 0 ? x[i] : Math.pow(x[-1 - i] / 9, 3));
            }
            return 1 / (1 + Math.exp(-z));
        } catch (Exception e) {
            return null;
        }
    }

    /**
     * --assign-teach &lt;file&gt; (with --deaths-value): at every Free Peoples assignment of the re-created games, the
     * human's answer, the bot's {@link #ASSIGN_CANDIDATES} best and assigning nothing, each played through the
     * opponent's next turn {@link #DEATHS_ROLLOUTS} times and scored by the win model; labelled as the deaths teacher
     * labels by value, with AssignmentProbe's numbers. In the bots' own games the Free Peoples bot put the killing
     * minion on the victim itself in 88% of its skirmish deaths.
     */
    static volatile PrintWriter ASSIGN_TEACH;
    static volatile int ASSIGN_CANDIDATES = 4;
    static final java.util.concurrent.atomic.AtomicLong assignsTaught = new java.util.concurrent.atomic.AtomicLong();

    static void teachAssignment(DefaultLotroGame live, GameCopier copier, BotVsBot.Context ctx, List<String> players,
                                String player, AwaitingDecision decision, double[] state, String humanAnswer,
                                Random sampler) {
        PrintWriter out = ASSIGN_TEACH;
        GameState gs = live.getGameState();
        if (out == null || state == null || decision.getDecisionType() != AwaitingDecisionType.ASSIGN_MINIONS
                || !player.equals(gs.getCurrentPlayerId()) || MOVE_VALUE_INDEX == null)
            return;
        List<String> options = new ArrayList<>();
        options.add(humanAnswer == null ? "" : humanAnswer);
        try {
            if (BotVsBot.bot(KIND, player, ctx) instanceof com.gempukku.lotro.bots.rl.fotrstarters.FotrStarterBot bot)
                for (String c : bot.assignmentCandidates(gs, decision, ASSIGN_CANDIDATES))
                    if (!options.contains(c))
                        options.add(c);
        } catch (RuntimeException ignored) {
            // no candidates: the human's answer and nothing
        }
        if (!options.contains(""))
            options.add("");
        if (options.size() < 2)
            return;
        double[] score = new double[options.size()];
        long base = sampler.nextLong();
        for (int i = 0; i < options.size(); i++) {
            int n = 0;
            for (int r = 0; r < DEATHS_ROLLOUTS; r++) {
                Double v = roundValue(live, copier, ctx, players, player, options.get(i), base + r * 0x9E3779B97F4A7C15L);
                if (v == null)
                    continue;
                n++;
                score[i] += v;
            }
            if (n == 0)
                return;
            score[i] /= n;
        }
        double best = Arrays.stream(score).max().orElse(0);
        if (Arrays.stream(score).allMatch(v -> v == score[0]))
            return;
        StringBuilder lines = new StringBuilder();
        for (int i = 0; i < options.size(); i++) {
            int label = score[i] >= best - 1e-9 ? 1 : score[i] <= best - DEATHS_VALUE_MARGIN ? 0 : -1;
            if (label < 0)
                continue;
            try {
                var action = new com.gempukku.lotro.bots.rl.semanticaction.AssignMinionsAction(options.get(i), decision, gs,
                        true).withProbe(com.gempukku.lotro.bots.rl.fotrstarters.models.assignment.AssignmentProbe
                        .features(gs, decision, com.gempukku.lotro.bots.rl.fotrstarters.models.assignment.AssignmentProbe
                                .parse(options.get(i))));
                var step = new com.gempukku.lotro.bots.rl.LearningStep(state, action, true, decision);
                step.reward = label == 1 ? 10 : 0;
                lines.append(step.toJson()).append('\n');
            } catch (RuntimeException ignored) {
                // an answer the semantic action cannot describe
            }
        }
        assignsTaught.incrementAndGet();
        synchronized (Counterfactual.class) {
            out.print(lines);
            out.flush();
        }
    }

    /** {@link #DEATHS_TEACH}'s labels for one decision; nothing if it is not a skirmish prompt of a side in {@link #DEATHS_SIDES}. */
    static void teachByDeaths(DefaultLotroGame live, GameCopier copier, BotVsBot.Context ctx, List<String> players,
                              String player, AwaitingDecision decision, double[] state, Random sampler) {
        PrintWriter out = DEATHS_TEACH;
        GameState gs = live.getGameState();
        boolean fp = player.equals(gs.getCurrentPlayerId());
        boolean prompt = DEATHS_PROMPTS != null
                ? decision.getDecisionType() == AwaitingDecisionType.CARD_ACTION_CHOICE && decision.getText() != null
                && DEATHS_PROMPTS.matcher(decision.getText()).find()
                : com.gempukku.lotro.bots.rl.fotrstarters.EffectProbe.applies(decision);
        if (out == null || state == null || !DEATHS_SIDES.contains(fp ? "fp" : "shadow") || !prompt)
            return;
        int sign = fp ? -1 : 1;
        String[] ids = decision.getDecisionParameters().get("actionId");
        if (ids == null || ids.length == 0)
            return;
        List<String> options = new ArrayList<>(List.of(ids));
        options.add("");
        double[] score = new double[options.size()];
        long base = sampler.nextLong();
        for (int i = 0; i < options.size(); i++) {
            int n = 0;
            for (int r = 0; r < DEATHS_ROLLOUTS; r++) {
                long seed = base + r * 0x9E3779B97F4A7C15L;
                if (DEATHS_BY_VALUE) {
                    Double v = roundValue(live, copier, ctx, players, player, options.get(i), seed);
                    if (v == null)
                        continue;
                    n++;
                    score[i] += v;
                    continue;
                }
                int[] d = turnDeaths(live, copier, ctx, players, player, options.get(i), seed);
                if (d == null)
                    continue;
                n++;
                score[i] += sign * (d[0] + BEARER_WEIGHT * d[1]);
            }
            if (n == 0)
                return;
            score[i] /= n;
        }
        double best = Arrays.stream(score).max().orElse(0);
        if (Arrays.stream(score).allMatch(v -> v == score[0]))
            return;
        StringBuilder lines = new StringBuilder();
        for (int i = 0; i < options.size(); i++) {
            double margin = DEATHS_BY_VALUE ? DEATHS_VALUE_MARGIN : 1;
            int label = score[i] >= best - 1e-9 ? 1 : score[i] <= best - margin ? 0 : -1;
            if (label < 0)
                continue;
            String o = options.get(i);
            var action = new com.gempukku.lotro.bots.rl.semanticaction.CardActionChoiceAction(o, decision, gs)
                    .withProbe(TrainingSteps.probe(gs, decision, o, player));
            var step = new com.gempukku.lotro.bots.rl.LearningStep(state, action, false, decision);
            step.reward = label == 1 ? 10 : 0;
            lines.append(step.toJson()).append('\n');
        }
        deathsTaught.incrementAndGet();
        synchronized (Counterfactual.class) {
            out.print(lines);
            out.flush();
        }
    }

    /**
     * {@code answer} given on a copy and the turn played out by the bot for both players: the Free Peoples player's
     * companions killed (not its Ring-bearer), and 1 if its Ring-bearer died; null if it cannot be told.
     */
    private static int[] turnDeaths(DefaultLotroGame live, GameCopier copier, BotVsBot.Context ctx, List<String> players,
                                    String player, String answer, long seed) {
        try {
            DefaultLotroGame copy = copier.copy(live);
            GameState gs = copy.getGameState();
            gs.setDeckShuffler(null);
            gs.setRandomSeed(seed);
            for (String p : players)
                gs.shuffleDeck(p);
            String fp = gs.getCurrentPlayerId();
            var bearer = gs.getRingBearer(fp);
            int before = companionsDead(gs, fp, bearer);
            int turn = gs.getTurnNumber();
            var feedback = GameCopier.feedback(copy);
            AwaitingDecision pending = feedback.getAwaitingDecision(player);
            if (pending == null)
                return null;
            feedback.participantDecided(player);
            BotDecisionFallback.decide(copy, player, pending, answer);
            copy.carryOutPendingActionsUntilDecisionNeeded();
            Map<String, BotPlayer> bots = new HashMap<>();
            for (String p : players)
                bots.put(p, BotVsBot.bot(KIND, p, ctx));
            com.gempukku.lotro.bots.BotRandom.current().setSeed(seed ^ 0x5DEECE66DL);
            for (int step = 0; step < 3000 && !copy.isFinished() && copy.getGameState().getTurnNumber() == turn; step++) {
                String who = null;
                AwaitingDecision d = null;
                for (String p : players)
                    if ((d = feedback.getAwaitingDecision(p)) != null) {
                        who = p;
                        break;
                    }
                if (d == null)
                    return null;
                String a = bots.get(who).chooseAction(copy.getGameState(), d);
                feedback.participantDecided(who);
                BotDecisionFallback.decide(copy, who, d, a);
                copy.carryOutPendingActionsUntilDecisionNeeded();
            }
            GameState after = copy.getGameState();
            boolean rbDead = bearer != null && after.getDeadPile(fp).stream().anyMatch(c -> c.getCardId() == bearer.getCardId());
            return new int[]{companionsDead(after, fp, bearer) - before, rbDead ? 1 : 0};
        } catch (Exception e) {
            return null;
        }
    }

    private static int companionsDead(GameState gs, String fp, PhysicalCard bearer) {
        int n = 0;
        for (PhysicalCard c : gs.getDeadPile(fp))
            if (c.getBlueprint().getCardType() == com.gempukku.lotro.common.CardType.COMPANION
                    && (bearer == null || c.getCardId() != bearer.getCardId()))
                n++;
        return n;
    }

    /** Whether {@code player} wins after giving {@code answer} on a copy, the bot playing on for both; null on error. */
    private static Boolean playOut(DefaultLotroGame live, GameCopier copier, BotVsBot.Context ctx, List<String> players,
                                   String player, String answer, long seed) {
        try {
            DefaultLotroGame copy = copier.copy(live);
            GameState gs = copy.getGameState();
            gs.setDeckShuffler(null); // the replay's forced draws end here: the futures are the shuffle's
            gs.setRandomSeed(seed);
            for (String p : players)
                gs.shuffleDeck(p);
            var feedback = GameCopier.feedback(copy);
            AwaitingDecision pending = feedback.getAwaitingDecision(player);
            if (pending == null)
                return null;
            feedback.participantDecided(player);
            BotDecisionFallback.decide(copy, player, pending, answer);
            copy.carryOutPendingActionsUntilDecisionNeeded();
            if (!copy.isFinished()) {
                List<BotPlayer> bots = new ArrayList<>();
                for (String p : players)
                    bots.add(BotVsBot.bot(KIND, p, ctx));
                BotGame.resume(copy, bots, 3000, new ArrayList<>(), new HashMap<>(), 0, new Random(seed ^ 0x5DEECE66DL));
            }
            if (!copy.isFinished() || copy.getWinnerPlayerId() == null)
                return null;
            return player.equals(copy.getWinnerPlayerId());
        } catch (Exception e) {
            return null;
        }
    }

    /** The same answer in effect: the same card and action for card actions, the same pairings for assignments. */
    static boolean same(GameState gs, AwaitingDecision d, String a, String b) {
        if (a.equals(b))
            return true;
        if (d.getDecisionType() == AwaitingDecisionType.CARD_ACTION_CHOICE)
            return describeAction(gs, d, a).equals(describeAction(gs, d, b));
        if (d.getDecisionType() == AwaitingDecisionType.ASSIGN_MINIONS)
            return TrainingSteps.canonical(a).equals(TrainingSteps.canonical(b));
        if (d.getDecisionType() == AwaitingDecisionType.CARD_SELECTION
                || d.getDecisionType() == AwaitingDecisionType.ARBITRARY_CARDS)
            return blueprints(gs, d, a).equals(blueprints(gs, d, b));
        return false;
    }

    private static String describeAction(GameState gs, AwaitingDecision d, String answer) {
        if (answer.isEmpty())
            return "pass";
        String[] ids = d.getDecisionParameters().get("actionId"), cards = d.getDecisionParameters().get("cardId"),
                texts = d.getDecisionParameters().get("actionText");
        for (int i = 0; ids != null && i < ids.length; i++)
            if (ids[i].equals(answer)) {
                String bp = cards != null && i < cards.length ? gs.getBlueprintId(Integer.parseInt(cards[i])) : "?";
                return bp + "|" + (texts != null && i < texts.length ? texts[i] : "");
            }
        return answer;
    }

    private static List<String> blueprints(GameState gs, AwaitingDecision d, String answer) {
        List<String> out = new ArrayList<>();
        String[] bps = d.getDecisionParameters().get("blueprintId"), ids = d.getDecisionParameters().get("cardId");
        for (String part : answer.split(",")) {
            if (part.isBlank())
                continue;
            String bp = part;
            for (int i = 0; ids != null && i < ids.length; i++)
                if (ids[i].equals(part))
                    bp = bps != null && i < bps.length && !bps[i].equals("inPlay") ? bps[i] : blueprintOf(gs, part);
            out.add(bp);
        }
        Collections.sort(out);
        return out;
    }

    private static String blueprintOf(GameState gs, String id) {
        try {
            PhysicalCard c = gs.findCardById(Integer.parseInt(id));
            return c != null ? c.getBlueprintId() : id;
        } catch (RuntimeException e) {
            return id;
        }
    }

    /** What kind of decision: its type, and for card actions the phase (the prompt), the side otherwise. */
    static String bucket(GameState gs, AwaitingDecision d, String player) {
        String side = player.equals(gs.getCurrentPlayerId()) ? "FP" : "Shadow";
        String text = d.getText() == null ? "" : d.getText().replaceAll("[^A-Za-z ]", "").trim();
        return switch (d.getDecisionType()) {
            case CARD_ACTION_CHOICE -> "card action: " + text;
            case ASSIGN_MINIONS -> "assignment (" + side + ")";
            case MULTIPLE_CHOICE -> "choice: " + text;
            case INTEGER -> "number: " + text;
            default -> d.getDecisionType().name().toLowerCase().replace('_', ' ') + ": "
                    + (text.length() > 40 ? text.substring(0, 40) : text);
        };
    }

    /**
     * One row: game, player, bucket, type, the two raw answers, the wins and rollouts; then the side, what each answer
     * did (card and action text for card actions, the cards for selections), and in a skirmish the deciding player's
     * strength margin; then the turn and phase, at an assignment the Ring-bearer and the free characters, and at a
     * card action every option offered. Agreed: n is 0; differed but not played out (--cf-rollouts 0): n is -1.
     */
    private static void write(GameState gs, String game, String player, String bucket, AwaitingDecision d, String human,
                              String bot, int humanWins, int botWins, int n) {
        PrintWriter out = OUT;
        if (out == null)
            return;
        String side = player.equals(gs.getCurrentPlayerId()) ? "FP" : "Shadow";
        boolean actions = d.getDecisionType() == AwaitingDecisionType.CARD_ACTION_CHOICE;
        boolean selections = d.getDecisionType() == AwaitingDecisionType.CARD_SELECTION
                || d.getDecisionType() == AwaitingDecisionType.ARBITRARY_CARDS;
        String humanDid = actions ? describeAction(gs, d, human) : selections ? String.join(" ", blueprints(gs, d, human)) : "";
        String botDid = actions ? describeAction(gs, d, bot) : selections ? String.join(" ", blueprints(gs, d, bot)) : "";
        String margin = "";
        var stats = gs.getMostRecentGameStats();
        if (gs.getSkirmish() != null && gs.getSkirmish().getFellowshipCharacter() != null && stats != null) {
            int m = stats.getFellowshipSkirmishStrength() - stats.getShadowSkirmishStrength();
            margin = String.valueOf(side.equals("FP") ? m : -m);
        }
        synchronized (Counterfactual.class) {
            out.println(String.join("\t", game, player, bucket, d.getDecisionType().name(), human.replace('\t', ' '),
                    bot.replace('\t', ' '), String.valueOf(humanWins), String.valueOf(botWins), String.valueOf(n),
                    side, humanDid.replace('\t', ' '), botDid.replace('\t', ' '), margin,
                    String.valueOf(gs.getTurnNumber()), String.valueOf(gs.getCurrentPhase()), assignmentContext(gs, d),
                    actions ? offered(gs, d) : ""));
            out.flush();
        }
    }

    /** At a card action: every option, as describeAction gives them, joined by " ;; " -- which cards were on offer. */
    private static String offered(GameState gs, AwaitingDecision d) {
        String[] ids = d.getDecisionParameters().get("actionId");
        List<String> out = new ArrayList<>();
        for (int i = 0; ids != null && i < ids.length; i++)
            out.add(describeAction(gs, d, ids[i]).replace('\t', ' '));
        return String.join(" ;; ", out);
    }
}
