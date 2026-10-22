package com.gempukku.lotro.bots.rl.fotrstarters.models.integerchoice;

import com.gempukku.lotro.bots.rl.LearningStep;
import com.gempukku.lotro.bots.rl.RLGameStateFeatures;
import com.gempukku.lotro.bots.rl.fotrstarters.models.AbstractTrainer;
import com.gempukku.lotro.bots.rl.fotrstarters.models.LabeledPoint;
import com.gempukku.lotro.bots.rl.fotrstarters.models.ModelRegistry;
import com.gempukku.lotro.bots.rl.semanticaction.IntegerChoiceAction;
import com.gempukku.lotro.game.state.GameState;
import com.gempukku.lotro.logic.decisions.AwaitingDecision;
import com.gempukku.lotro.logic.decisions.AwaitingDecisionType;
import smile.classification.SoftClassifier;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

/**
 * "How many?" -- the number prompts no other model answers: how many times to heal with Elrond, Herald to Gil-galad,
 * to exert Gandalf, X, tokens to remove, twilight to add. The bot answered them at random; on July 2022's games it
 * chose Elrond's count as people did 0% of the time (70 asked: people heal as often as allowed, 545 of 545 in November).
 * Each value from min to max (at most {@link #MAX_OPTIONS} of them) is an option, compared listwise (BotTeach --model
 * listwise), with the prompt's words and the value against the range. With a "number-bid" file in the model directory
 * it takes the burden bid too, which BurdenTrainer draws at people's overall rates without looking at the game.
 */
public class NumberChoiceTrainer extends AbstractTrainer {
    static final int MAX_OPTIONS = 10;
    private static final List<String> WORDS = List.of("heal", "exert", "discard", "token", "twilight", "threat",
            "burden", "bid", "remove", "add", "archery", " x", "spot", "wound", "draw");

    /** Whether this model answers {@code decision}: "how many to spot" is always the most, and the bid only on request. */
    public static boolean applies(AwaitingDecision decision, boolean bid) {
        if (decision.getDecisionType() != AwaitingDecisionType.INTEGER || decision.getText() == null)
            return false;
        String t = decision.getText().toLowerCase(Locale.ROOT);
        if (t.contains("to spot"))
            return false;
        return bid || !t.contains("burdens to bid");
    }

    private static int param(AwaitingDecision d, String name, int otherwise) {
        String[] v = d.getDecisionParameters().get(name);
        if (v == null || v.length == 0 || v[0] == null || v[0].isBlank())
            return otherwise;
        try {
            return Integer.parseInt(v[0]);
        } catch (NumberFormatException e) {
            return otherwise;
        }
    }

    static int min(AwaitingDecision d) {
        return param(d, "min", 0);
    }

    static int max(AwaitingDecision d) {
        int min = min(d);
        return Math.min(param(d, "max", min + MAX_OPTIONS - 1), min + MAX_OPTIONS - 1);
    }

    static double[] vector(double[] state, String text, int value, int min, int max) {
        String t = " " + text.toLowerCase(Locale.ROOT).replaceAll("<[^>]*>", " ");
        double[] option = new double[WORDS.size() + 7];
        for (int i = 0; i < WORDS.size(); i++)
            option[i] = t.contains(WORDS.get(i)) ? 1 : 0;
        int k = WORDS.size();
        option[k] = value;
        option[k + 1] = value - min;
        option[k + 2] = max - value;
        option[k + 3] = max > min ? (value - min) / (double) (max - min) : 1;
        option[k + 4] = value == min ? 1 : 0;
        option[k + 5] = value == max ? 1 : 0;
        option[k + 6] = max - min;
        double[] x = Arrays.copyOf(state, state.length + option.length);
        System.arraycopy(option, 0, x, state.length, option.length);
        return x;
    }

    @Override
    public boolean appliesTo(GameState gameState, AwaitingDecision decision, String playerName) {
        return applies(decision, true);
    }

    @Override
    public boolean isStepRelevant(LearningStep step) {
        return applies(step.decision, true) && step.action instanceof IntegerChoiceAction;
    }

    @Override
    public String getAnswer(GameState gameState, AwaitingDecision decision, String playerName,
                            RLGameStateFeatures features, ModelRegistry modelRegistry) {
        SoftClassifier<double[]> model = modelRegistry.getModel(getClass());
        if (model == null)
            return null;
        int min = min(decision), max = max(decision);
        if (max < min)
            return null;
        double[] state = features.extractFeatures(gameState, decision, playerName);
        int best = min;
        double top = Double.NEGATIVE_INFINITY;
        for (int v = min; v <= max; v++) {
            double[] p = new double[2];
            model.predict(vector(state, decision.getText(), v, min, max), p);
            if (p[1] > top) {
                top = p[1];
                best = v;
            }
        }
        return String.valueOf(best);
    }

    @Override
    protected List<LabeledPoint> extractTrainingData(List<LearningStep> steps) {
        List<LabeledPoint> data = new ArrayList<>();
        int group = 0;
        for (LearningStep step : steps) {
            if (!isStepRelevant(step))
                continue;
            int chosen = ((IntegerChoiceAction) step.action).getValue();
            int min = min(step.decision), max = Math.max(max(step.decision), chosen);
            if (max <= min || chosen < min)
                continue; // nothing to choose, or an answer outside the range written
            group++;
            for (int v = min; v <= max; v++)
                data.add(new LabeledPoint(v == chosen ? 1 : 0, vector(step.state, step.decision.getText(), v, min, max),
                        group));
        }
        return data;
    }
}
