package com.gempukku.lotro.bots.simulation;

import com.gempukku.lotro.bots.BotDecisionFallback;
import com.gempukku.lotro.bots.BotPlayer;
import com.gempukku.lotro.game.PhysicalCard;
import com.gempukku.lotro.game.state.GameState;
import com.gempukku.lotro.logic.decisions.AwaitingDecision;
import com.gempukku.lotro.logic.decisions.AwaitingDecisionType;
import com.gempukku.lotro.logic.timing.DefaultLotroGame;

import java.io.PrintWriter;
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
     * strength margin; then the turn and phase, and at an assignment the Ring-bearer and the free characters. Agreed: n is 0; differed but not played out (--cf-rollouts 0): n is -1.
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
                    String.valueOf(gs.getTurnNumber()), String.valueOf(gs.getCurrentPhase()), assignmentContext(gs, d)));
            out.flush();
        }
    }
}
