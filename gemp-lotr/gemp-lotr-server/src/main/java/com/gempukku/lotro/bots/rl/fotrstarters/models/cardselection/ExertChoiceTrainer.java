package com.gempukku.lotro.bots.rl.fotrstarters.models.cardselection;

import com.gempukku.lotro.bots.rl.LearningStep;
import com.gempukku.lotro.bots.rl.RLGameStateFeatures;
import com.gempukku.lotro.bots.rl.fotrstarters.CardFeatures;
import com.gempukku.lotro.bots.rl.fotrstarters.models.AbstractTrainer;
import com.gempukku.lotro.bots.rl.fotrstarters.models.LabeledPoint;
import com.gempukku.lotro.bots.rl.fotrstarters.models.ModelRegistry;
import com.gempukku.lotro.bots.rl.semanticaction.CardSelectionAction;
import com.gempukku.lotro.game.CardNotFoundException;
import com.gempukku.lotro.game.PhysicalCard;
import com.gempukku.lotro.game.state.Assignment;
import com.gempukku.lotro.game.state.GameState;
import com.gempukku.lotro.game.state.Skirmish;
import com.gempukku.lotro.logic.decisions.AwaitingDecision;
import com.gempukku.lotro.logic.decisions.AwaitingDecisionType;
import smile.classification.SoftClassifier;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * "Choose cards to exert", one character of several -- 99% of them (33,569 of 33,895 in 2022's Movie Block games), a
 * cost more often than not. {@link ExertTrainer} scored each candidate by its card and wounds; in a skirmish the bot
 * did not ask it at all, and exerted whoever was fighting. In July's games, where the bot chose another character than
 * the human, it lost 3.2 points a time. An exert is a wound that cannot kill: a character on 1 vitality is never
 * offered, but one exerted down to 1 can pay no more exert costs and dies to the next wound. So each candidate is
 * compared listwise (BotTeach --model listwise) with what the choice turns on: whose it is, its vitality before and
 * after, whether this is its last exert, the Ring-bearer, whether it is fighting now or assigned to fight later this
 * turn, and its strength. Written with the label (CardSelectionAction's options), as only the live game knows them.
 */
public class ExertChoiceTrainer extends AbstractTrainer {
    static final String PROMPT = "Choose cards to exert";
    public static final int EXTRA = 9;

    public static boolean applies(AwaitingDecision decision) {
        if (decision.getDecisionType() != AwaitingDecisionType.CARD_SELECTION || decision.getText() == null
                || !decision.getText().startsWith(PROMPT))
            return false;
        String[] ids = decision.getDecisionParameters().get("cardId");
        return "1".equals(decision.getDecisionParameters().get("min")[0])
                && "1".equals(decision.getDecisionParameters().get("max")[0]) && ids != null && ids.length > 1;
    }

    /** Each candidate's numbers, in the decision's order. */
    public static ReconcileChoiceTrainer.Options options(GameState gs, AwaitingDecision decision, String me) {
        String[] cardIds = decision.getDecisionParameters().get("cardId");
        String[] blueprints = new String[cardIds.length];
        double[][] extras = new double[cardIds.length][];
        for (int i = 0; i < cardIds.length; i++) {
            PhysicalCard card = gs.findCardById(Integer.parseInt(cardIds[i]));
            blueprints[i] = card.getBlueprintId();
            extras[i] = character(gs, card, me);
        }
        return new ReconcileChoiceTrainer.Options(blueprints, extras);
    }

    /**
     * A character's numbers from {@code me}'s side: mine (1) or not (-1), vitality now and after one exert, whether
     * that leaves it on 1, the Ring-bearer, fighting now, assigned to fight later this turn, strength, wounds.
     */
    public static double[] character(GameState gs, PhysicalCard card, String me) {
        var game = gs.getGame();
        int vitality = game.getModifiersQuerying().getVitality(game, card);
        boolean bearer = false;
        try {
            bearer = gs.getRingBearer(card.getOwner()) == card;
        } catch (RuntimeException ignored) {
        }
        boolean fighting = false;
        Skirmish s = gs.getSkirmish();
        if (s != null)
            fighting = s.getFellowshipCharacter() == card || s.getShadowCharacters().contains(card);
        boolean assigned = false;
        for (Assignment a : gs.getAssignments())
            assigned |= a.getFellowshipCharacter() == card || a.getShadowCharacters().contains(card);
        return new double[]{card.getOwner().equals(me) ? 1 : -1, vitality, vitality - 1, vitality - 1 <= 1 ? 1 : 0,
                bearer ? 1 : 0, fighting ? 1 : 0, assigned && !fighting ? 1 : 0,
                game.getModifiersQuerying().getStrength(game, card), gs.getWounds(card)};
    }

    /** The candidate {@code answer} names, in the decision's order; -1 if none. */
    public static int chosen(AwaitingDecision decision, String answer) {
        if (answer == null || answer.isBlank())
            return -1;
        return Arrays.asList(decision.getDecisionParameters().get("cardId")).indexOf(answer.split(",")[0].trim());
    }

    static double[] vector(double[] state, String blueprint, int wounds, double[] extra) throws CardNotFoundException {
        double[] card = CardFeatures.getCardFeatures(blueprint, wounds);
        double[] option = Arrays.copyOf(card, card.length + extra.length);
        System.arraycopy(extra, 0, option, card.length, extra.length);
        return CardFeatures.join(state, option, blueprint);
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
                model.predict(vector(state, o.blueprints()[i], (int) o.extras()[i][8], o.extras()[i]), p);
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
                            vector(step.state, a.getOptionBlueprints()[i], (int) e[8], e), group));
                } catch (CardNotFoundException ignored) {
                }
            }
        }
        return data;
    }
}
