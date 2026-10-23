package com.gempukku.lotro.bots.rl.fotrstarters.models.cardselection;

import com.gempukku.lotro.bots.rl.LearningStep;
import com.gempukku.lotro.bots.rl.RLGameStateFeatures;
import com.gempukku.lotro.bots.rl.fotrstarters.models.AbstractTrainer;
import com.gempukku.lotro.bots.rl.fotrstarters.models.LabeledPoint;
import com.gempukku.lotro.bots.rl.fotrstarters.models.ModelRegistry;
import com.gempukku.lotro.bots.rl.semanticaction.CardSelectionAction;
import com.gempukku.lotro.game.CardNotFoundException;
import com.gempukku.lotro.game.PhysicalCard;
import com.gempukku.lotro.game.state.GameState;
import com.gempukku.lotro.logic.decisions.AwaitingDecision;
import com.gempukku.lotro.logic.decisions.AwaitingDecisionType;
import smile.classification.SoftClassifier;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

/**
 * Which one character takes an archery wound, a wound, or a heal -- sanctuary healing included. Their models scored a
 * card and its wounds; in a skirmish the bot did not ask them, and took whoever was fighting. Each candidate compared
 * listwise with {@link ExertChoiceTrainer#character}'s numbers (side, vitality, Ring-bearer, fighting now or later,
 * strength, wounds), which prompt it is, and its vitality after: one wound left on 1 is a character lost. Written with
 * the label (CardSelectionAction's options).
 */
public class CharacterChoiceTrainer extends AbstractTrainer {
    private static final List<String> KINDS = List.of("archery wound", "to wound", "to heal", "sanctuary");

    public static boolean applies(AwaitingDecision decision) {
        if (decision.getDecisionType() != AwaitingDecisionType.CARD_SELECTION || decision.getText() == null)
            return false;
        String t = decision.getText().toLowerCase(Locale.ROOT);
        String[] ids = decision.getDecisionParameters().get("cardId");
        return KINDS.stream().anyMatch(t::contains) && ids != null && ids.length > 1
                && "1".equals(decision.getDecisionParameters().get("min")[0])
                && "1".equals(decision.getDecisionParameters().get("max")[0]);
    }

    public static ReconcileChoiceTrainer.Options options(GameState gs, AwaitingDecision decision, String me) {
        String t = decision.getText().toLowerCase(Locale.ROOT);
        boolean heal = t.contains("heal");
        String[] cardIds = decision.getDecisionParameters().get("cardId");
        String[] blueprints = new String[cardIds.length];
        double[][] extras = new double[cardIds.length][];
        for (int i = 0; i < cardIds.length; i++) {
            PhysicalCard card = gs.findCardById(Integer.parseInt(cardIds[i]));
            blueprints[i] = card.getBlueprintId();
            double[] c = ExertChoiceTrainer.character(gs, card, me);
            double[] e = Arrays.copyOf(c, c.length + KINDS.size() + 2);
            for (int k = 0; k < KINDS.size(); k++)
                e[c.length + k] = t.contains(KINDS.get(k)) ? 1 : 0;
            double after = c[1] + (heal ? 1 : -1);
            e[c.length + KINDS.size()] = after;
            e[c.length + KINDS.size() + 1] = !heal && after <= 0 ? 1 : 0;
            extras[i] = e;
        }
        return new ReconcileChoiceTrainer.Options(blueprints, extras);
    }

    @Override
    public boolean appliesTo(GameState gameState, AwaitingDecision decision, String playerName) {
        return applies(decision);
    }

    @Override
    public boolean isStepRelevant(LearningStep step) {
        return applies(step.decision) && step.action instanceof CardSelectionAction a && a.getOptionExtras() != null
                && a.getOptionChosen() >= 0;
    }

    @Override
    public String getAnswer(GameState gameState, AwaitingDecision decision, String playerName,
                            RLGameStateFeatures features, ModelRegistry modelRegistry) {
        SoftClassifier<double[]> model = modelRegistry.getModel(getClass());
        if (model == null)
            return null;
        String[] cardIds = decision.getDecisionParameters().get("cardId");
        double[] state = features.extractFeatures(gameState, decision, playerName);
        var o = options(gameState, decision, playerName);
        int best = -1;
        double top = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < cardIds.length; i++) {
            try {
                double[] p = new double[2];
                model.predict(ExertChoiceTrainer.vector(state, o.blueprints()[i], (int) o.extras()[i][8], o.extras()[i]), p);
                if (p[1] > top) {
                    top = p[1];
                    best = i;
                }
            } catch (CardNotFoundException ignored) {
            }
        }
        return best < 0 ? null : cardIds[best];
    }

    @Override
    protected List<LabeledPoint> extractTrainingData(List<LearningStep> steps) {
        List<LabeledPoint> data = new ArrayList<>();
        int group = 0;
        for (LearningStep step : steps) {
            if (!isStepRelevant(step))
                continue;
            CardSelectionAction a = (CardSelectionAction) step.action;
            group++;
            for (int i = 0; i < a.getOptionBlueprints().length; i++) {
                try {
                    double[] e = a.getOptionExtras()[i];
                    data.add(new LabeledPoint(i == a.getOptionChosen() ? 1 : 0,
                            ExertChoiceTrainer.vector(step.state, a.getOptionBlueprints()[i], (int) e[8], e), group));
                } catch (CardNotFoundException ignored) {
                }
            }
        }
        return data;
    }
}
