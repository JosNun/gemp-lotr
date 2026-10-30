package com.gempukku.lotro.bots.rl.fotrstarters.models.cardselection;

import com.gempukku.lotro.bots.rl.LearningStep;
import com.gempukku.lotro.bots.rl.RLGameStateFeatures;
import com.gempukku.lotro.bots.rl.fotrstarters.CardFeatures;
import com.gempukku.lotro.bots.rl.fotrstarters.EffectProbe;
import com.gempukku.lotro.bots.rl.fotrstarters.models.AbstractTrainer;
import com.gempukku.lotro.bots.rl.fotrstarters.models.LabeledPoint;
import com.gempukku.lotro.bots.rl.fotrstarters.models.ModelRegistry;
import com.gempukku.lotro.bots.rl.semanticaction.CardSelectionAction;
import com.gempukku.lotro.common.Side;
import com.gempukku.lotro.game.CardNotFoundException;
import com.gempukku.lotro.game.PhysicalCard;
import com.gempukku.lotro.game.state.GameState;
import com.gempukku.lotro.logic.decisions.AwaitingDecision;
import com.gempukku.lotro.logic.decisions.AwaitingDecisionType;
import smile.classification.SoftClassifier;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Reconcile: discard one card from hand, or none, then draw up to the hand size. {@link ReconcileTrainer} scored the
 * cards and always discarded the best-scored one; people keep their whole hand 48% of the time (54,045 of 112,648
 * reconciles in 2022's Movie Block games, from 13% discarding with one card in hand to 67% with eight), and in July's
 * games the bot chose as they did 18.9% of the time. Here "keep everything" is an option beside each card, compared
 * listwise (BotTeach --model listwise), with what the choice turns on: a card's other copies in hand, whether it is dead
 * (unique, and a copy of it already in play or in the dead pile), how many cards each answer draws, and what the
 * draw deck holds -- one hand holds both sides' cards, so a Shadow card cycled is a draw from a deck of both.
 * <p>
 * The per-option numbers need the live game, so they are written with the label (CardSelectionAction's options) and
 * only labels written so are fit on.
 */
public class ReconcileChoiceTrainer extends AbstractTrainer {
    static final String PROMPT = "Reconcile - choose card to discard";
    /** Per-option numbers: keep-everything, copies in hand, dead, side, draws, deck FP/Shadow left, hand FP/Shadow after. */
    public static final int EXTRA = 9;

    public static boolean applies(AwaitingDecision decision) {
        return decision.getDecisionType() == AwaitingDecisionType.CARD_SELECTION && decision.getText() != null
                && decision.getText().startsWith(PROMPT);
    }

    /** The options -- keep everything first, then each card in the decision's order -- with their numbers. */
    public record Options(String[] blueprints, double[][] extras) {
    }

    public static Options options(GameState gs, AwaitingDecision decision, String me) {
        String[] cardIds = decision.getDecisionParameters().get("cardId");
        List<? extends PhysicalCard> hand = gs.getHand(me);
        int handSize = gs.getGame().getFormat().getHandSize();
        int[] deck = EffectProbe.deckLeft(gs, me);
        int handFp = 0, handShadow = 0;
        for (PhysicalCard c : hand) {
            handFp += c.getBlueprint().getSide() == Side.FREE_PEOPLE ? 1 : 0;
            handShadow += c.getBlueprint().getSide() == Side.SHADOW ? 1 : 0;
        }
        String[] blueprints = new String[cardIds.length + 1];
        double[][] extras = new double[cardIds.length + 1][];
        extras[0] = new double[]{1, 0, 0, 0, Math.max(0, handSize - hand.size()), deck[0], deck[1], handFp, handShadow};
        for (int i = 0; i < cardIds.length; i++) {
            PhysicalCard card = gs.findCardById(Integer.parseInt(cardIds[i]));
            blueprints[i + 1] = card.getBlueprintId();
            var bp = card.getBlueprint();
            String title = bp.getTitle();
            int copies = 0;
            for (PhysicalCard c : hand)
                if (c != card && title.equals(c.getBlueprint().getTitle()))
                    copies++;
            boolean dead = false;
            if (bp.isUnique()) {
                for (PhysicalCard c : gs.getInPlay())
                    dead |= c.getOwner().equals(me) && title.equals(c.getBlueprint().getTitle());
                for (PhysicalCard c : gs.getDeadPile(me))
                    dead |= title.equals(c.getBlueprint().getTitle());
            }
            int side = bp.getSide() == Side.FREE_PEOPLE ? 1 : bp.getSide() == Side.SHADOW ? -1 : 0;
            extras[i + 1] = new double[]{0, copies, dead ? 1 : 0, side, Math.max(0, handSize - hand.size() + 1),
                    deck[0], deck[1], handFp - (side == 1 ? 1 : 0), handShadow - (side == -1 ? 1 : 0)};
        }
        return new Options(blueprints, extras);
    }

    /** Which option {@code answer} is: 0 for none, else the card's place in the decision plus one. */
    public static int chosen(AwaitingDecision decision, String answer) {
        if (answer == null || answer.isBlank())
            return 0;
        String[] cardIds = decision.getDecisionParameters().get("cardId");
        return Arrays.asList(cardIds).indexOf(answer.split(",")[0].trim()) + 1;
    }

    /**
     * -Dreconcile.plain (training) / a "reconcile-plain" file in the model directory (play): of the per-option numbers
     * only "keep everything" -- to tell what the option is worth from what the numbers are.
     */
    public static final boolean PLAIN_TRAINING = Boolean.getBoolean("reconcile.plain");

    private static double[] vector(double[] state, String blueprint, double[] extra, boolean plain, List<String> text)
            throws CardNotFoundException {
        if (plain) {
            extra = extra.clone();
            Arrays.fill(extra, 1, extra.length, 0);
        }
        String bp = blueprint != null ? blueprint : CardFeatures.PASS;
        double[] card = CardFeatures.getCardFeatures(bp, 0);
        double[] option = Arrays.copyOf(card, card.length + extra.length);
        System.arraycopy(extra, 0, option, card.length, extra.length);
        // card text at the very end, where a model fit without it never looks
        return com.gempukku.lotro.bots.rl.fotrstarters.CardTextFeatures.withText(CardFeatures.join(state, option, bp),
                text, bp);
    }

    @Override
    public boolean appliesTo(GameState gameState, AwaitingDecision decision, String playerName) {
        return applies(decision);
    }

    @Override
    public boolean isStepRelevant(LearningStep step) {
        return applies(step.decision) && step.action instanceof CardSelectionAction a && a.getOptionExtras() != null;
    }

    @Override
    public String getAnswer(GameState gameState, AwaitingDecision decision, String playerName,
                            RLGameStateFeatures features, ModelRegistry modelRegistry) {
        SoftClassifier<double[]> model = modelRegistry.getModel(getClass());
        if (model == null)
            return null;
        String[] cardIds = decision.getDecisionParameters().get("cardId");
        double[] state = features.extractFeatures(gameState, decision, playerName);
        Options o = options(gameState, decision, playerName);
        List<String> text = modelRegistry.selectionText(getClass());
        // Keep the hand only if keeping is likelier than discarding any card at all. Taking the likeliest single
        // option kept it 84% of the time on July's reconciles (people: 53%) -- keeping is one option against seven
        // cards that share the rest -- and lost 45.8% against the bot that always discarded.
        double[] score = new double[o.blueprints().length];
        Arrays.fill(score, Double.NEGATIVE_INFINITY);
        for (int i = 0; i < o.blueprints().length; i++) {
            try {
                double[] p = new double[2];
                model.predict(vector(state, o.blueprints()[i], o.extras()[i], modelRegistry.isReconcilePlain(), text), p);
                score[i] = logit(p[1]);
            } catch (CardNotFoundException ignored) {
                // a card the features cannot describe is not discarded
            }
        }
        double[] share = softmax(score);
        var net2 = modelRegistry.getBoardNet2();
        if (net2 != null) {
            // keep, then the cards in the decision's order -- the same options in the same order as the network's
            var r = net2.read(gameState, decision, playerName);
            if (r != null && r.options().length == share.length) {
                double[] m = com.gempukku.lotro.bots.rl.fotrstarters.BoardNet2.mix(r.options(), score,
                        net2.treeWeight("reconcile"));
                share = softmax(m);
            }
        }
        if (share[0] >= 0.5)
            return "";
        int best = 1;
        for (int i = 2; i < share.length; i++)
            if (share[i] > share[best])
                best = i;
        return share.length > 1 ? cardIds[best - 1] : "";
    }

    /** The score a probability came from: ListwiseTreeModel gives the logistic of an option's softmax score. */
    public static double logit(double p) {
        double q = Math.min(1 - 1e-12, Math.max(1e-12, p));
        return Math.log(q / (1 - q));
    }

    /** Each option's share of the decision. */
    public static double[] softmax(double[] score) {
        double max = Double.NEGATIVE_INFINITY, sum = 0;
        for (double v : score)
            max = Math.max(max, v);
        double[] share = new double[score.length];
        for (int i = 0; i < score.length; i++)
            sum += share[i] = score[i] == Double.NEGATIVE_INFINITY ? 0 : Math.exp(score[i] - max);
        for (int i = 0; i < score.length; i++)
            share[i] /= sum;
        return share;
    }

    @Override
    protected List<LabeledPoint> extractTrainingData(List<LearningStep> steps) {
        List<LabeledPoint> data = new ArrayList<>();
        int group = 0;
        List<String> text = com.gempukku.lotro.bots.rl.fotrstarters.CardTextFeatures.trainingSelectionText(getClass());
        for (LearningStep step : steps) {
            if (!isStepRelevant(step))
                continue;
            CardSelectionAction a = (CardSelectionAction) step.action;
            group++;
            for (int i = 0; i < a.getOptionBlueprints().length; i++) {
                try {
                    data.add(new LabeledPoint(i == a.getOptionChosen() ? 1 : 0,
                            vector(step.state, a.getOptionBlueprints()[i], a.getOptionExtras()[i], PLAIN_TRAINING, text),
                            group));
                } catch (CardNotFoundException ignored) {
                }
            }
        }
        return data;
    }
}
