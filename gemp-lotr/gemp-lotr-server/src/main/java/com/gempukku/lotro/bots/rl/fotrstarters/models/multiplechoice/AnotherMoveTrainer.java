package com.gempukku.lotro.bots.rl.fotrstarters.models.multiplechoice;

import com.gempukku.lotro.bots.rl.RLGameStateFeatures;
import com.gempukku.lotro.bots.rl.fotrstarters.models.ModelRegistry;
import com.gempukku.lotro.bots.rl.semanticaction.MultipleChoiceAction;
import com.gempukku.lotro.game.state.GameState;
import com.gempukku.lotro.logic.decisions.AwaitingDecision;
import smile.classification.SoftClassifier;

public class AnotherMoveTrainer extends AbstractMultipleChoiceTrainer {
    @Override
    protected String getTextTrigger() {
        return "another move";
    }

    @Override
    protected String getPositiveOption() {
        return "Yes";
    }

    @Override
    protected String getNegativeOption() {
        return "No";
    }

    /**
     * Move again when the model's yes is above the model directory's "move-threshold" (0.5 without one). Fit on
     * people, it moves about as often as they do; on July 2022's games, wherever the bot and the human differed,
     * moving did 10-13 points better than stopping whichever of them chose it (turns 1-2: 535 played out).
     */
    @Override
    public String getAnswer(GameState gameState, AwaitingDecision decision, String playerName,
                            RLGameStateFeatures features, ModelRegistry modelRegistry) {
        SoftClassifier<double[]> model = modelRegistry.getModel(getClass());
        double[] probs = new double[2];
        model.predict(features.extractFeatures(gameState, decision, playerName), probs);
        String option = probs[1] > modelRegistry.getMoveThreshold() ? getPositiveOption() : getNegativeOption();
        return new MultipleChoiceAction(option).toDecisionString(decision, gameState);
    }
}
