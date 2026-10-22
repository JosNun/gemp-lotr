package com.gempukku.lotro.bots.rl.fotrstarters;

import com.gempukku.lotro.bots.BotDecisionFallback;
import com.gempukku.lotro.bots.BotPlayer;
import com.gempukku.lotro.common.CardType;
import com.gempukku.lotro.game.PhysicalCard;
import com.gempukku.lotro.game.state.GameState;
import com.gempukku.lotro.game.state.Skirmish;
import com.gempukku.lotro.logic.timing.RuleUtils;
import com.gempukku.lotro.logic.decisions.AwaitingDecision;
import com.gempukku.lotro.logic.decisions.AwaitingDecisionType;
import com.gempukku.lotro.logic.effects.ResolveSkirmishEffect;
import com.gempukku.lotro.logic.timing.DefaultLotroGame;
import com.gempukku.lotro.game.DefaultUserFeedback;

import java.util.Arrays;
import java.util.List;
import java.util.function.Function;
import java.util.function.UnaryOperator;

/**
 * What a card action does, found by doing it: the option is given on a copy of the game, the questions it raises on
 * the way (a target, a cost) answered by the bots, until the next card-action prompt that is not a response window,
 * and the board read before and after. Built for skirmishes first. The card-action model saw a card and the skirmish as it stood, never what the card would change; in July
 * 2022's games, where it passed and the human played -- Glorfindel exerting to wound a minion, Cirdan, Smeagol --
 * it lost 3.6 points (Free Peoples) and 4.6 (Shadow) a time. One step, not a rollout: what the play does now, not how
 * the game ends, so it costs a copy (~170 us) and a few answers per option.
 * <p>
 * Every number is from the deciding player's side: a margin and a skirmish result they want high, burdens and threats
 * signed by whom they hurt.
 */
public final class EffectProbe {
    /**
     * The numbers {@link #probe} gives. The last three are the hand: one hand holds both sides' cards, so a Shadow card
     * played is a slot for a Free Peoples card at reconcile, and a card held may be the only one of its kind -- other
     * copies of the played card in hand, and my hand's Free Peoples and Shadow cards after the play. Then my draw deck's
     * Free Peoples and Shadow cards left: a player knows their own deck, not its order. A re-created human game knows
     * only the deck cards that were seen -- the ones drawn later -- so these are counted from the other side: every
     * format GEMP has makes a draw deck half and half (validateShadowFPCount), so each side's half less that side's
     * cards out of the deck.
     */
    public static final int LENGTH = 29;
    /** What models fit before the hand numbers read: a "probe-contexts" file with no length= line. */
    public static final int LENGTH_WITHOUT_HAND = 22;
    /** The first numbers, all a skirmish-only model ("skirmish-probe") was fit on: no strength totals. */
    public static final int SKIRMISH_LENGTH = 20;
    /** The response window prompt: passed on the copy, it is not where a play has finished. */
    private static final String RESPONSES_PROMPT = "Optional responses";
    /** The skirmish action prompt (SkirmishActionProcedureAction); no other card-action prompt reads so. */
    public static final String SKIRMISH_PROMPT = "Choose action to play or Pass";
    /**
     * The last two: an outcome-fit value model's chance that I win, after the play, and its change from now. People's
     * choices teach what people do, not what an extra burden or wound is worth; in July's games the bot passed where
     * Shadow players kept pushing a skirmish already won, and lost 3.7 points a time. Null: zeros.
     */
    public static volatile smile.classification.SoftClassifier<double[]> VALUE;

    private static double value(GameState gs, String me) {
        var model = VALUE;
        if (model == null)
            return 0;
        double[] p = new double[2];
        model.predict(new GeneralStateFeatures().extractFeatures(gs, null, me), p);
        return p[1];
    }

    /** Copies a live game (GameCopier::copy); set by what runs the bots. Null: nothing probed, and zeros. */
    public static volatile UnaryOperator<DefaultLotroGame> FORK;
    /** Questions answered on the copy before giving up on reaching the next card-action prompt. */
    private static final int MAX_STEPS = 12;

    private EffectProbe() {
    }

    public static boolean applies(AwaitingDecision decision) {
        return decision.getDecisionType() == AwaitingDecisionType.CARD_ACTION_CHOICE
                && SKIRMISH_PROMPT.equals(decision.getText());
    }

    /** Passing: nothing changes. */
    public static double[] pass(GameState gs, String me) {
        Board now = Board.read(gs, me);
        double[] f = new double[LENGTH];
        f[0] = 1;
        f[1] = now.margin;
        f[3] = now.result;
        f[19] = 1;
        f[23] = now.handFp;
        f[24] = now.handShadow;
        int[] deck = deckLeft(gs, me);
        f[25] = deck[0];
        f[26] = deck[1];
        f[28] = value(gs, me);
        return f;
    }

    /** {@code f} cut or padded to {@code n} numbers: stored probes are as long as the probe was when they were made. */
    public static double[] fit(double[] f, int n) {
        return f == null ? new double[n] : f.length == n ? f : Arrays.copyOf(f, n);
    }

    /**
     * {@code answer} given on a copy of the game {@code gs} belongs to, which must be waiting on {@code decision} from
     * {@code me}; {@code bots} answers what the copy asks on the way, for either player. Null if it cannot be told.
     */
    public static double[] probe(GameState gs, String me, String answer, Function<String, BotPlayer> bots) {
        UnaryOperator<DefaultLotroGame> fork = FORK;
        if (fork == null || !(gs.getGame() instanceof DefaultLotroGame live))
            return null;
        try {
            Board before = Board.read(gs, me);
            DefaultLotroGame copy = fork.apply(live);
            if (!(copy.getUserFeedback() instanceof DefaultUserFeedback feedback))
                return null;
            AwaitingDecision pending = feedback.getAwaitingDecision(me);
            if (pending == null)
                return null;
            int copies = copies(gs, pending, answer, me);
            feedback.participantDecided(me);
            BotDecisionFallback.decide(copy, me, pending, answer);
            copy.carryOutPendingActionsUntilDecisionNeeded();
            List<String> players = copy.getGameState().getPlayerNames();
            boolean inSkirmish = before.fpCharacter != -1;
            boolean resolved = false, ended = false;
            for (int step = 0; step < MAX_STEPS; step++) {
                GameState now = copy.getGameState();
                if (copy.isFinished()) {
                    resolved = true;
                    break;
                }
                Skirmish s = now.getSkirmish();
                if (inSkirmish && (s == null || s.getFellowshipCharacter() == null
                        || s.getFellowshipCharacter().getCardId() != before.fpCharacter)) {
                    resolved = ended = true;
                    break;
                }
                String who = null;
                AwaitingDecision d = null;
                for (String p : players)
                    if ((d = feedback.getAwaitingDecision(p)) != null) {
                        who = p;
                        break;
                    }
                if (d == null)
                    break;
                String reply;
                if (d.getDecisionType() == AwaitingDecisionType.CARD_ACTION_CHOICE) {
                    if (!RESPONSES_PROMPT.equals(d.getText())) {
                        resolved = true;
                        break;
                    }
                    reply = ""; // a response window: what the card does, not what can be done about it
                } else {
                    reply = bots.apply(who).chooseAction(now, d);
                }
                feedback.participantDecided(who);
                BotDecisionFallback.decide(copy, who, d, reply);
                copy.carryOutPendingActionsUntilDecisionNeeded();
            }
            Board after = Board.read(copy.getGameState(), me);
            boolean fp = before.fp;
            double[] f = new double[LENGTH];
            f[0] = 1;
            f[1] = ended ? 0 : after.margin;
            f[2] = ended ? 0 : after.margin - before.margin;
            f[3] = ended ? 0 : after.result;
            f[4] = ended ? 0 : after.result - before.result;
            f[5] = after.damage - before.damage;
            f[6] = (fp ? -1 : 1) * (after.burdens - before.burdens);
            f[7] = (fp ? -1 : 1) * (after.threats - before.threats);
            f[8] = (fp ? -1 : 1) * (after.twilight - before.twilight);
            f[9] = after.hand - before.hand;
            f[10] = after.myWounds - before.myWounds;
            f[11] = after.theirWounds - before.theirWounds;
            f[12] = after.myCharacters - before.myCharacters;
            f[13] = after.theirCharacters - before.theirCharacters;
            f[14] = after.myCards - before.myCards;
            f[15] = after.theirCards - before.theirCards;
            f[16] = (fp ? 1 : -1) * (after.resistance - before.resistance);
            f[17] = ended ? 1 : 0;
            f[18] = !copy.isFinished() || copy.getWinnerPlayerId() == null ? 0 : me.equals(copy.getWinnerPlayerId()) ? 1 : -1;
            f[19] = resolved ? 1 : 0;
            f[20] = after.myStrength - before.myStrength;
            f[21] = after.theirStrength - before.theirStrength;
            f[22] = copies;
            f[23] = after.handFp;
            f[24] = after.handShadow;
            int[] deck = deckLeft(gs, me);
            f[25] = deck[0];
            f[26] = deck[1];
            double valueAfter = value(copy.getGameState(), me);
            f[27] = valueAfter - value(gs, me);
            f[28] = valueAfter;
            return f;
        } catch (Exception e) {
            return null;
        }
    }

    /**
     * {@code me}'s draw deck's Free Peoples and Shadow cards left, from the cards out of it: half the draw deck each,
     * less that side's cards in hand, play, discard, dead pile, removed or stacked (the Ring-bearer never was in it).
     * PROBE_CHECK=1 checks it against the deck itself, where the deck is real.
     */
    public static int[] deckLeft(GameState gs, String me) {
        int outFp = 0, outShadow = 0;
        PhysicalCard bearer = null;
        try {
            bearer = gs.getRingBearer(me);
        } catch (RuntimeException ignored) {
        }
        List<List<? extends PhysicalCard>> zones = List.of(gs.getHand(me), gs.getVoidFromHand(me), gs.getRemoved(me),
                gs.getDiscard(me), gs.getDeadPile(me), gs.getStacked(me), gs.getInPlay());
        for (List<? extends PhysicalCard> zone : zones)
            for (PhysicalCard c : zone) {
                if (!c.getOwner().equals(me) || c == bearer)
                    continue;
                var side = c.getBlueprint().getSide();
                outFp += side == com.gempukku.lotro.common.Side.FREE_PEOPLE ? 1 : 0;
                outShadow += side == com.gempukku.lotro.common.Side.SHADOW ? 1 : 0;
            }
        int half = (gs.getDeck(me).size() + outFp + outShadow) / 2;
        int[] left = {half - outFp, half - outShadow};
        if (CHECK) {
            int fp = 0, shadow = 0;
            for (PhysicalCard c : gs.getDeck(me)) {
                fp += c.getBlueprint().getSide() == com.gempukku.lotro.common.Side.FREE_PEOPLE ? 1 : 0;
                shadow += c.getBlueprint().getSide() == com.gempukku.lotro.common.Side.SHADOW ? 1 : 0;
            }
            CHECKED.incrementAndGet();
            if (fp != left[0] || shadow != left[1])
                System.err.println("deckLeft: counted " + left[0] + "/" + left[1] + ", deck holds " + fp + "/" + shadow
                        + " (" + me + ", turn " + gs.getTurnNumber() + ", " + gs.getCurrentPhase() + ")");
        }
        return left;
    }

    private static final boolean CHECK = System.getenv("PROBE_CHECK") != null;
    public static final java.util.concurrent.atomic.AtomicLong CHECKED = new java.util.concurrent.atomic.AtomicLong();

    /** Other copies of the card {@code answer} plays (by title) in {@code me}'s hand. */
    private static int copies(GameState gs, AwaitingDecision decision, String answer, String me) {
        int n = 0;
        String[] ids = decision.getDecisionParameters().get("actionId"), cards = decision.getDecisionParameters().get("cardId");
        for (int i = 0; ids != null && cards != null && i < ids.length && i < cards.length; i++) {
            if (!ids[i].equals(answer))
                continue;
            int cardId = Integer.parseInt(cards[i]);
            PhysicalCard source = gs.findCardById(cardId);
            if (source == null)
                return n;
            String title = source.getBlueprint().getTitle();
            for (PhysicalCard c : gs.getHand(me))
                if (c.getCardId() != cardId && title.equals(c.getBlueprint().getTitle()))
                    n++;
            return n;
        }
        return n;
    }

    /** No probe: all zeros, "probed" among them. */
    public static double[] none() {
        return new double[LENGTH];
    }

    /** What the probe compares, read off one board from {@code me}'s side. */
    private record Board(boolean fp, int fpCharacter, int margin, int result, int damage, int burdens, int threats,
                         int twilight, int hand, int myWounds, int theirWounds, int myCharacters, int theirCharacters,
                         int myCards, int theirCards, int resistance, int myStrength, int theirStrength,
                         int handFp, int handShadow) {
        static Board read(GameState gs, String me) {
            var game = gs.getGame();
            boolean fp = me.equals(gs.getCurrentPlayerId());
            int sign = fp ? 1 : -1;
            Skirmish s = gs.getSkirmish();
            int fpCharacter = -1, margin = 0, result = 0, damage = 0;
            if (s != null && s.getFellowshipCharacter() != null && !s.getShadowCharacters().isEmpty()) {
                fpCharacter = s.getFellowshipCharacter().getCardId();
                margin = sign * (RuleUtils.getFellowshipSkirmishStrength(game) - RuleUtils.getShadowSkirmishStrength(game));
                result = sign * switch (new ResolveSkirmishEffect().getUpcomingResult(game)) {
                    case FELLOWSHIP_OVERWHELMED -> -2;
                    case FELLOWSHIP_LOSES -> -1;
                    case SHADOW_LOSES -> 1;
                    case SHADOW_OVERWHELMED -> 2;
                };
                damage = fp ? RuleUtils.getFellowshipSkirmishDamageBonus(game) : RuleUtils.getShadowSkirmishDamageBonus(game);
            } else if (s != null && s.getFellowshipCharacter() != null) {
                fpCharacter = s.getFellowshipCharacter().getCardId();
            }
            int myWounds = 0, theirWounds = 0, myCharacters = 0, theirCharacters = 0, myCards = 0, theirCards = 0;
            int myStrength = 0, theirStrength = 0;
            for (PhysicalCard c : gs.getInPlay()) {
                boolean mine = c.getOwner().equals(me);
                CardType type = c.getBlueprint().getCardType();
                boolean character = type == CardType.COMPANION || type == CardType.ALLY || type == CardType.MINION;
                if (mine) {
                    myCards++;
                    myWounds += gs.getWounds(c);
                    myCharacters += character ? 1 : 0;
                    myStrength += character ? game.getModifiersQuerying().getStrength(game, c) : 0;
                } else {
                    theirCards++;
                    theirWounds += gs.getWounds(c);
                    theirCharacters += character ? 1 : 0;
                    theirStrength += character ? game.getModifiersQuerying().getStrength(game, c) : 0;
                }
            }
            String fpPlayer = gs.getCurrentPlayerId();
            int resistance = 0;
            try {
                PhysicalCard bearer = gs.getRingBearer(fpPlayer);
                if (bearer != null)
                    resistance = game.getModifiersQuerying().getResistance(game, bearer);
            } catch (RuntimeException ignored) {
            }
            int handFp = 0, handShadow = 0;
            for (PhysicalCard c : gs.getHand(me)) {
                handFp += c.getBlueprint().getSide() == com.gempukku.lotro.common.Side.FREE_PEOPLE ? 1 : 0;
                handShadow += c.getBlueprint().getSide() == com.gempukku.lotro.common.Side.SHADOW ? 1 : 0;
            }
            return new Board(fp, fpCharacter, margin, result, damage, gs.getBurdens(), gs.getThreats(),
                    gs.getTwilightPool(), gs.getHand(me).size(), myWounds, theirWounds, myCharacters, theirCharacters,
                    myCards, theirCards, resistance, myStrength, theirStrength, handFp, handShadow);
        }
    }

    /** For tests and reports: the names of the numbers, in order. */
    public static final List<String> NAMES = Arrays.asList("probed", "marginAfter", "marginDelta", "resultAfter",
            "resultDelta", "damageDelta", "burdensForMe", "threatsForMe", "twilightForMe", "handDelta", "myWoundsDelta",
            "theirWoundsDelta", "myCharactersDelta", "theirCharactersDelta", "myCardsDelta", "theirCardsDelta",
            "resistanceForMe", "skirmishEnded", "gameOver", "resolved", "myStrengthDelta", "theirStrengthDelta",
            "copiesInHand", "myHandFpAfter", "myHandShadowAfter", "myDeckFpLeft", "myDeckShadowLeft",
            "valueDelta", "valueAfter");
}
