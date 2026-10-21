package com.gempukku.lotro.bots.rl.fotrstarters.models.multiplechoice;

import com.gempukku.lotro.bots.rl.LearningStep;
import com.gempukku.lotro.bots.rl.RLGameStateFeatures;
import com.gempukku.lotro.bots.rl.fotrstarters.models.AbstractTrainer;
import com.gempukku.lotro.bots.rl.fotrstarters.models.LabeledPoint;
import com.gempukku.lotro.bots.rl.fotrstarters.models.ModelRegistry;
import com.gempukku.lotro.bots.rl.semanticaction.MultipleChoiceAction;
import com.gempukku.lotro.game.state.GameState;
import com.gempukku.lotro.logic.decisions.AwaitingDecision;
import com.gempukku.lotro.logic.decisions.AwaitingDecisionType;
import smile.classification.SoftClassifier;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * The multiple choices no other model answers -- a card's choice of costs or effects: "Discard your hand" or "Add 2
 * burdens", "Make Gollum strength +3" or "Add a burden", "Add 2 burdens" or "Add 2 threats". The bot answered them at
 * random; against people, in July 2022's games, it lost 4.0 points where it differed from them. Each option is the
 * state and words from its text (what it costs or does, and how many); fit listwise on what people chose
 * (BotTeach --model listwise), so the options of one question are compared with each other.
 */
public class GenericChoiceTrainer extends AbstractTrainer {
    private static final List<String> WORDS = List.of("burden", "threat", "discard", "hand", "exert", "wound", "heal",
            "remove", "add", "token", "condition", "possession", "strength", "draw", "deck", "twilight", "play",
            "companion", "minion", "site", "shadow", "bearer", "ring", "return", "spot", "prevent", "cancel",
            "skirmish", "control", "opponent", "this", "top", "bottom", "discard pile", "your", "ally");
    private static final Pattern NUMBER = Pattern.compile("(\\d+)");
    /** Questions the dedicated models answer. */
    private static final List<String> OTHERS = List.of("another move", "mulligan", "go first");

    static double[] optionFeatures(String option, int options) {
        String t = option.toLowerCase(Locale.ROOT).replaceAll("<[^>]*>", " ");
        double[] f = new double[WORDS.size() + 3];
        for (int i = 0; i < WORDS.size(); i++)
            if (t.contains(WORDS.get(i)))
                f[i] = 1;
        Matcher m = NUMBER.matcher(t);
        f[WORDS.size()] = m.find() ? Integer.parseInt(m.group(1)) : 1;
        f[WORDS.size() + 1] = t.contains("+") ? 1 : t.contains("-") ? -1 : 0;
        f[WORDS.size() + 2] = options;
        return f;
    }

    private static double[] vector(double[] state, String option, int options) {
        double[] o = optionFeatures(option, options);
        double[] x = Arrays.copyOf(state, state.length + o.length);
        System.arraycopy(o, 0, x, state.length, o.length);
        return x;
    }

    private static boolean generic(String text) {
        String t = text == null ? "" : text.toLowerCase(Locale.ROOT);
        return OTHERS.stream().noneMatch(t::contains);
    }

    @Override
    public boolean appliesTo(GameState gameState, AwaitingDecision decision, String playerName) {
        return decision.getDecisionType() == AwaitingDecisionType.MULTIPLE_CHOICE && generic(decision.getText());
    }

    @Override
    public boolean isStepRelevant(LearningStep step) {
        return step.decision.getDecisionType() == AwaitingDecisionType.MULTIPLE_CHOICE
                && step.action instanceof MultipleChoiceAction && generic(step.decision.getText());
    }

    @Override
    public String getAnswer(GameState gameState, AwaitingDecision decision, String playerName,
                            RLGameStateFeatures features, ModelRegistry modelRegistry) {
        SoftClassifier<double[]> model = modelRegistry.getModel(getClass());
        String[] results = decision.getDecisionParameters().get("results");
        if (model == null || results == null || results.length == 0)
            return null;
        double[] state = features.extractFeatures(gameState, decision, playerName);
        int best = 0;
        double top = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < results.length; i++) {
            double[] p = new double[2];
            model.predict(vector(state, results[i], results.length), p);
            if (p[1] > top) {
                top = p[1];
                best = i;
            }
        }
        return String.valueOf(best);
    }

    @Override
    protected List<LabeledPoint> extractTrainingData(List<LearningStep> steps) {
        List<LabeledPoint> data = new ArrayList<>();
        for (LearningStep step : steps) {
            if (!isStepRelevant(step))
                continue;
            String[] results = step.decision.getDecisionParameters().get("results");
            String option = ((MultipleChoiceAction) step.action).getChosenOption();
            if (option == null || results == null)
                continue;
            data.add(new LabeledPoint(step.reward > 0 ? 1 : 0, vector(step.state, option, results.length), step.group));
        }
        return data;
    }
}
