package com.gempukku.lotro.bots.rl.fotrstarters.models.cardaction;

import com.gempukku.lotro.bots.rl.LearningStep;
import com.gempukku.lotro.bots.rl.RLGameStateFeatures;
import com.gempukku.lotro.bots.rl.fotrstarters.CardFeatures;
import com.gempukku.lotro.bots.rl.fotrstarters.models.AbstractTrainer;
import com.gempukku.lotro.bots.rl.fotrstarters.models.LabeledPoint;
import com.gempukku.lotro.bots.rl.fotrstarters.models.ModelRegistry;
import com.gempukku.lotro.bots.rl.semanticaction.CardActionChoiceAction;
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
 * Play/use/pass for any card-action decision the phase-specific answerers do not take: the Archery and Assignment
 * phases, Regroup actions that are "Use" rather than "Play", and any prompt worded differently from the FotR
 * starters'. Without it those were answered at random -- about 4,000 times over 1,030 starter games.
 * <p>
 * One model across phases, so the decision's phase and the action's kind are inputs: {@link #CONTEXTS} and
 * {@link #KINDS} one-hot, read from the prompt and the action text since a stored step keeps no phase. The state
 * vector is padded to the skirmish length, because the game-state features hand skirmish and response decisions
 * four more numbers than the rest.
 */
public class AnyPhaseCardActionTrainer extends AbstractTrainer {
    private static final int STATE_LENGTH = 30;
    private static final List<String> CONTEXTS = List.of("fellowship", "shadow", "maneuver", "archery", "assignment",
            "skirmish", "regroup", "response");
    private static final List<String> KINDS = List.of("play", "use", "heal", "transfer");

    @Override
    public boolean appliesTo(GameState gameState, AwaitingDecision decision, String playerName) {
        String[] actionIds = decision.getDecisionParameters().get("actionId");
        return decision.getDecisionType() == AwaitingDecisionType.CARD_ACTION_CHOICE
                && actionIds != null && actionIds.length > 0;
    }

    @Override
    public boolean isStepRelevant(LearningStep step) {
        if (step.decision.getDecisionType() != AwaitingDecisionType.CARD_ACTION_CHOICE)
            return false;
        String[] actionIds = step.decision.getDecisionParameters().get("actionId");
        return step.action instanceof CardActionChoiceAction && actionIds != null && actionIds.length > 0;
    }

    @Override
    public String getAnswer(GameState gameState, AwaitingDecision decision, String playerName,
                            RLGameStateFeatures features, ModelRegistry modelRegistry) {
        SoftClassifier<double[]> model = modelRegistry.getModel(getClass());
        if (model == null)
            return null;
        double[] state = features.extractFeatures(gameState, decision, playerName);
        double[] context = context(decision.getText());
        String[] cardIds = decision.getDecisionParameters().get("cardId");
        String[] actionTexts = decision.getDecisionParameters().get("actionText");

        int best = -1; // -1 is pass
        double bestScore;
        try {
            bestScore = score(model, state, context, null, CardFeatures.PASS, 0);
        } catch (CardNotFoundException e) {
            return null; // cannot happen: PASS is all zeros
        }
        for (int i = 0; i < cardIds.length; i++) {
            try {
                int cardId = Integer.parseInt(cardIds[i]);
                int wounds = 0;
                for (PhysicalCard card : gameState.getInPlay())
                    if (card.getCardId() == cardId)
                        wounds = gameState.getWounds(card);
                double s = score(model, state, context, actionTexts[i], gameState.getBlueprintId(cardId), wounds);
                if (s > bestScore) {
                    bestScore = s;
                    best = i;
                }
            } catch (CardNotFoundException | RuntimeException ignored) {
                // a card the features cannot describe is not offered
            }
        }
        return best < 0 ? "" : String.valueOf(best);
    }

    private double score(SoftClassifier<double[]> model, double[] state, double[] context, String actionText,
                         String blueprintId, int wounds) throws CardNotFoundException {
        double[] probs = new double[2];
        model.predict(vector(state, context, actionText, blueprintId, wounds), probs);
        return probs[1];
    }

    @Override
    protected List<LabeledPoint> extractTrainingData(List<LearningStep> steps) {
        List<LabeledPoint> data = new ArrayList<>();
        for (LearningStep step : steps) {
            if (!isStepRelevant(step))
                continue;
            CardActionChoiceAction action = (CardActionChoiceAction) step.action;
            String blueprint = action.getSourceBlueprint() != null ? action.getSourceBlueprint() : CardFeatures.PASS;
            try {
                data.add(new LabeledPoint(step.reward > 0 ? 1 : 0, vector(step.state, context(step.decision.getText()),
                        action.getSourceBlueprint() != null ? action.getActionText() : null, blueprint,
                        action.getWoundsOnSource())));
            } catch (CardNotFoundException ignored) {
            }
        }
        return data;
    }

    private static double[] vector(double[] state, double[] context, String actionText, String blueprintId, int wounds)
            throws CardNotFoundException {
        double[] card = CardFeatures.getCardFeatures(blueprintId, wounds);
        double[] kind = new double[KINDS.size()];
        if (actionText != null)
            for (int k = 0; k < KINDS.size(); k++)
                if (actionText.toLowerCase().startsWith(KINDS.get(k)))
                    kind[k] = 1;
        double[] option = new double[context.length + kind.length + card.length];
        System.arraycopy(context, 0, option, 0, context.length);
        System.arraycopy(kind, 0, option, context.length, kind.length);
        System.arraycopy(card, 0, option, context.length + kind.length, card.length);
        // set 1 states are 26 or 30 long by decision type; set 2's are one length already
        double[] fixedState = CardFeatures.version() >= 2 ? state : Arrays.copyOf(state, STATE_LENGTH);
        return CardFeatures.join(fixedState, option, blueprintId);
    }

    private static double[] context(String text) {
        String lower = text.toLowerCase();
        double[] c = new double[CONTEXTS.size()];
        for (int i = 0; i < CONTEXTS.size(); i++)
            if (lower.contains(CONTEXTS.get(i))) {
                c[i] = 1;
                break;
            }
        return c;
    }
}
