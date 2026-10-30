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
    private static double logOdds(double p) {
        double q = Math.min(1 - 1e-9, Math.max(1e-9, p));
        return Math.log(q / (1 - q));
    }

    @Override
    public String getAnswer(GameState gameState, AwaitingDecision decision, String playerName,
                            RLGameStateFeatures features, ModelRegistry modelRegistry) {
        SoftClassifier<double[]> model = modelRegistry.getModel(getClass());
        double[] probs = new double[2];
        model.predict(features.extractFeatures(gameState, decision, playerName), probs);
        boolean yes = probs[1] > modelRegistry.getMoveThreshold();
        var net2 = modelRegistry.getBoardNet2();
        if (net2 != null && getPositiveOption().equals("Yes")) {
            // log-odds of moving: the network's, mixed with the trees' against their threshold
            var r = net2.read(gameState, decision, playerName);
            if (r != null && r.options().length == 2) {
                double w = net2.treeWeight("move");
                double net = r.options()[0] - r.options()[1];
                double trees = logOdds(probs[1]) - logOdds(modelRegistry.getMoveThreshold());
                yes = (1 - w) * net + w * trees > 0;
            }
        }
        String option = yes ? getPositiveOption() : getNegativeOption();
        return new MultipleChoiceAction(option).toDecisionString(decision, gameState);
    }
}
