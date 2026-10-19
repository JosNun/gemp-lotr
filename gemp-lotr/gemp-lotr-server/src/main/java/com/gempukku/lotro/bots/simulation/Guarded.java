package com.gempukku.lotro.bots.simulation;

import com.gempukku.lotro.bots.BotDecisionFallback;
import com.gempukku.lotro.bots.BotPlayer;
import com.gempukku.lotro.game.state.GameState;
import com.gempukku.lotro.logic.decisions.AwaitingDecision;
import com.gempukku.lotro.logic.decisions.AwaitingDecisionType;
import com.gempukku.lotro.logic.timing.DefaultLotroGame;

import java.util.concurrent.atomic.AtomicInteger;

/**
 * A bot that will not corrupt its own Ring-bearer. Real decks have abilities that cost a burden -- an Elf deck on
 * Galadriel, Bearer of Wisdom (resistance 3) retrieves cards from its deck for a burden each -- and the models, which
 * score options without the state that would say "this one ends the game", use them until the Ring-bearer corrupts: 72%
 * of 4,800 games between the most-played Movie Block decks ended that way, most on turn 1.
 * <p>
 * So: The One Ring's abilities are not used in a skirmish the Ring-bearer is not in; with resistance at 2 x
 * {@link #MARGIN} or less, a card action is first tried on a copy of the game (played on to the next decision) and
 * passed on if its cost leaves the Ring-bearer within {@link #MARGIN} of corruption; and within MARGIN, a
 * multiple-choice answer that corrupts it is swapped for one that does not.
 */
public final class Guarded implements BotPlayer {
    static final int MARGIN = 3;
    /** Answers replaced, over the run. */
    public static final AtomicInteger saved = new AtomicInteger(), moves = new AtomicInteger();
    private static volatile GameCopier copier;

    private final BotPlayer inner;
    /** Decline an optional extra move with this many companions in play or fewer; -1 never. */
    private final int carefulMoves;

    public Guarded(BotPlayer inner) {
        this(inner, -1);
    }

    public Guarded(BotPlayer inner, int carefulMoves) {
        this.inner = inner;
        this.carefulMoves = carefulMoves;
    }

    static void init(BotVsBot.Context ctx) {
        if (copier == null)
            synchronized (Guarded.class) {
                if (copier == null)
                    copier = new GameCopier(ctx.library(), ctx.formats());
            }
    }

    @Override
    public String chooseAction(GameState gameState, AwaitingDecision decision) {
        String answer = inner.chooseAction(gameState, decision);
        AwaitingDecisionType type = decision.getDecisionType();
        if (answer == null || answer.isEmpty())
            return answer;
        if (type == AwaitingDecisionType.CARD_ACTION_CHOICE) {
            if (ringOutsideOwnSkirmish(gameState, decision, answer) || costsTooMuch(gameState, answer)) {
                saved.incrementAndGet();
                return "";
            }
            return answer;
        }
        if (type == AwaitingDecisionType.MULTIPLE_CHOICE && carefulMoves >= 0
                && decision.getText().toLowerCase().contains("another move") && companions(gameState) <= carefulMoves) {
            // Each move adds twilight, which buys the Shadow player minions; a thin fellowship cannot take them all,
            // and the rest go to the Ring-bearer. 50 of 85 extra moves by guarded v2-real were with 2 companions or
            // fewer, and Ring-bearers mostly die skirmishing alone.
            String[] results = decision.getDecisionParameters().get("results");
            for (int i = 0; results != null && i < results.length; i++)
                if (results[i].equalsIgnoreCase("no")) {
                    if (!String.valueOf(i).equals(answer))
                        moves.incrementAndGet();
                    return String.valueOf(i);
                }
        }
        if (type != AwaitingDecisionType.MULTIPLE_CHOICE || resistance(gameState) > MARGIN
                || resistanceAfter(gameState, answer) > 0)
            return answer;
        saved.incrementAndGet();
        String[] results = decision.getDecisionParameters().get("results");
        for (int i = 0; results != null && i < results.length; i++)
            if (!String.valueOf(i).equals(answer) && resistanceAfter(gameState, String.valueOf(i)) > 0)
                return String.valueOf(i);
        return answer;
    }

    /**
     * The One Ring's skirmish abilities (a burden for strength, say) used while the Ring-bearer is not the one
     * skirmishing: they help nobody. Seen: three burdens paid while a Huorn fought.
     */
    private boolean ringOutsideOwnSkirmish(GameState gs, AwaitingDecision decision, String answer) {
        try {
            String[] actions = decision.getDecisionParameters().get("actionId");
            String[] cards = decision.getDecisionParameters().get("cardId");
            if (actions == null || cards == null || gs.getSkirmish() == null)
                return false;
            for (int i = 0; i < actions.length && i < cards.length; i++) {
                if (!actions[i].equals(answer))
                    continue;
                var card = gs.findCardById(Integer.parseInt(cards[i]));
                if (card == null || card.getBlueprint().getCardType() != com.gempukku.lotro.common.CardType.THE_ONE_RING
                        || !card.getOwner().equals(getName()))
                    return false;
                var bearer = gs.getRingBearer(getName());
                return bearer == null || gs.getSkirmish().getFellowshipCharacter() != bearer;
            }
        } catch (RuntimeException ignored) {
        }
        return false;
    }

    /** With resistance already low, a card action whose cost takes this Ring-bearer to within MARGIN of corruption. */
    private boolean costsTooMuch(GameState gs, String answer) {
        int now = resistance(gs);
        if (now > 2 * MARGIN)
            return false;
        int after = resistanceAfter(gs, answer);
        return after < now && after <= MARGIN;
    }

    private int companions(GameState gs) {
        return (int) gs.getInPlay().stream().filter(c -> c.getOwner().equals(getName())
                && c.getBlueprint().getCardType() == com.gempukku.lotro.common.CardType.COMPANION).count();
    }

    private int resistance(GameState gs) {
        try {
            var bearer = gs.getRingBearer(getName());
            var game = gs.getGame();
            return bearer == null ? Integer.MAX_VALUE : game.getModifiersQuerying().getResistance(game, bearer);
        } catch (RuntimeException e) {
            return Integer.MAX_VALUE;
        }
    }

    /**
     * This bot's Ring-bearer's resistance after giving {@code answer} now, played on a copy to the next decision; 0 if
     * that loses the game by corruption, MAX_VALUE if it cannot be told.
     */
    private int resistanceAfter(GameState gs, String answer) {
        if (copier == null || !(gs.getGame() instanceof DefaultLotroGame live))
            return Integer.MAX_VALUE;
        try {
            DefaultLotroGame copy = copier.copy(live);
            var feedback = GameCopier.feedback(copy);
            AwaitingDecision pending = feedback.getAwaitingDecision(getName());
            if (pending == null)
                return Integer.MAX_VALUE;
            feedback.participantDecided(getName());
            BotDecisionFallback.decide(copy, getName(), pending, answer);
            copy.carryOutPendingActionsUntilDecisionNeeded();
            var bearer = copy.getGameState().getRingBearer(getName());
            if (bearer == null)
                return Integer.MAX_VALUE;
            int left = copy.getModifiersQuerying().getResistance(copy, bearer);
            return copy.isFinished() && !getName().equals(copy.getWinnerPlayerId()) ? Math.min(0, left) : left;
        } catch (Exception e) {
            return Integer.MAX_VALUE;
        }
    }

    @Override
    public String getName() {
        return inner.getName();
    }
}
