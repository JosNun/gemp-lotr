package com.gempukku.lotro.bots.rl.fotrstarters.models.cardselection;

import com.gempukku.lotro.bots.rl.LearningStep;
import com.gempukku.lotro.bots.rl.RLGameStateFeatures;
import com.gempukku.lotro.bots.rl.fotrstarters.models.AbstractTrainer;
import com.gempukku.lotro.bots.rl.fotrstarters.models.LabeledPoint;
import com.gempukku.lotro.bots.rl.fotrstarters.models.ModelRegistry;
import com.gempukku.lotro.bots.rl.semanticaction.CardSelectionAction;
import com.gempukku.lotro.common.Keyword;
import com.gempukku.lotro.game.CardNotFoundException;
import com.gempukku.lotro.game.PhysicalCard;
import com.gempukku.lotro.game.state.Assignment;
import com.gempukku.lotro.game.state.GameState;
import com.gempukku.lotro.logic.decisions.AwaitingDecision;
import com.gempukku.lotro.logic.decisions.AwaitingDecisionType;
import com.gempukku.lotro.logic.modifiers.CantBeOverwhelmedModifier;
import smile.classification.SoftClassifier;

import java.util.ArrayList;
import java.util.List;

/**
 * "Choose next skirmish to resolve": which assigned character fights next. {@link SkirmishOrderTrainer} saw printed
 * strengths, and learned from the chosen skirmish alone -- from people's decisions, all chosen, so nothing to tell the
 * options apart; on July 2022's games it agreed with people 47.7% of the time, below the self-play bot it replaced
 * (53.6%). Here every skirmish still to fight is an option, compared listwise, as the board stands: the character's
 * and the minions' strength, the margin, won, overwhelmed, whether the character would die, the minions that would,
 * the Ring-bearer, vitality, damage. Written with the label (CardSelectionAction's options).
 */
public class SkirmishOrderChoiceTrainer extends AbstractTrainer {
    static final String PROMPT = "next skirmish to resolve";
    public static final int EXTRA = 13;

    public static boolean applies(AwaitingDecision decision) {
        String[] ids = decision.getDecisionParameters().get("cardId");
        return decision.getDecisionType() == AwaitingDecisionType.CARD_SELECTION && decision.getText() != null
                && decision.getText().toLowerCase().contains(PROMPT) && ids != null && ids.length > 1
                && "1".equals(decision.getDecisionParameters().get("max")[0]);
    }

    public static ReconcileChoiceTrainer.Options options(GameState gs, AwaitingDecision decision, String me) {
        String[] cardIds = decision.getDecisionParameters().get("cardId");
        String[] blueprints = new String[cardIds.length];
        double[][] extras = new double[cardIds.length][];
        for (int i = 0; i < cardIds.length; i++) {
            PhysicalCard fp = gs.findCardById(Integer.parseInt(cardIds[i]));
            blueprints[i] = fp.getBlueprintId();
            extras[i] = fight(gs, fp, me);
        }
        return new ReconcileChoiceTrainer.Options(blueprints, extras);
    }

    /** The skirmish {@code fp} is assigned to, as the board stands, from {@code me}'s side (margin signed for me). */
    static double[] fight(GameState gs, PhysicalCard fp, String me) {
        var game = gs.getGame();
        var q = game.getModifiersQuerying();
        List<PhysicalCard> minions = new ArrayList<>();
        for (Assignment a : gs.getAssignments())
            if (a.getFellowshipCharacter() == fp)
                minions.addAll(a.getShadowCharacters());
        int fpStrength = q.getStrength(game, fp), minionStrength = 0, minionDamage = 0, minionsDie = 0;
        int fpDamage = q.getKeywordCount(game, fp, Keyword.DAMAGE);
        for (PhysicalCard m : minions) {
            minionStrength += q.getStrength(game, m);
            minionDamage += q.getKeywordCount(game, m, Keyword.DAMAGE);
        }
        int multiplier = q.getOverwhelmMultiplier(game, fp);
        boolean overwhelmed = !minions.isEmpty() && multiplier < CantBeOverwhelmedModifier.ImmuneToOverwhelmThreshold
                && fpStrength * multiplier <= minionStrength;
        boolean won = fpStrength > minionStrength;
        int vitality = q.getVitality(game, fp);
        boolean dies = overwhelmed || !won && vitality <= 1 + minionDamage;
        if (won)
            for (PhysicalCard m : minions)
                minionsDie += q.getVitality(game, m) <= 1 + fpDamage ? 1 : 0;
        boolean bearer = false;
        try {
            bearer = gs.getRingBearer(fp.getOwner()) == fp;
        } catch (RuntimeException ignored) {
        }
        int sign = me.equals(gs.getCurrentPlayerId()) ? 1 : -1;
        return new double[]{fpStrength, minionStrength, sign * (fpStrength - minionStrength), won ? 1 : 0,
                overwhelmed ? 1 : 0, dies ? 1 : 0, minionsDie, bearer ? 1 : 0, vitality, minions.size(), minionDamage,
                fpDamage, sign};
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
                model.predict(ExertChoiceTrainer.vector(state, o.blueprints()[i], 0, o.extras()[i]), p);
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
                    data.add(new LabeledPoint(i == a.getOptionChosen() ? 1 : 0,
                            ExertChoiceTrainer.vector(step.state, a.getOptionBlueprints()[i], 0, a.getOptionExtras()[i]),
                            group));
                } catch (CardNotFoundException ignored) {
                }
            }
        }
        return data;
    }

    /** The decision's option {@code answer} names. */
    public static int chosen(AwaitingDecision decision, String answer) {
        return ExertChoiceTrainer.chosen(decision, answer);
    }
}
