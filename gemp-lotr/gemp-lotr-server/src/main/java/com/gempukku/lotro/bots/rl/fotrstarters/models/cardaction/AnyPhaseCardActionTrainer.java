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

    /**
     * -DanyPhase.rich (training) / an "any-phase-rich" file in the model directory (play): append what the per-phase
     * models see and this one did not -- an item's class and its holder, and whether a card can touch anyone in the
     * current skirmish (EXTRA numbers). The per-phase models beat this one however it was fit (BOT-NOTES.md).
     */
    public static final boolean RICH_TRAINING = Boolean.getBoolean("anyPhase.rich");
    /** -DanyPhase.perContext (training): one listwise model per {@link #CONTEXTS} entry, chosen by the prompt. */
    public static final boolean PER_CONTEXT_TRAINING = Boolean.getBoolean("anyPhase.perContext");
    static final int EXTRA = 10;

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
        boolean rich = modelRegistry.isAnyPhaseRich();
        double[] state = features.extractFeatures(gameState, decision, playerName);
        double[] context = context(decision.getText());
        String[] cardIds = decision.getDecisionParameters().get("cardId");
        String[] actionTexts = decision.getDecisionParameters().get("actionText");

        int best = -1; // -1 is pass
        double bestScore;
        try {
            bestScore = score(model, state, context, null, CardFeatures.PASS, 0, rich ? new double[EXTRA] : null);
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
                String bp = gameState.getBlueprintId(cardId);
                double s = score(model, state, context, actionTexts[i], bp, wounds,
                        rich ? liveExtra(gameState, cardId, bp) : null);
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
                         String blueprintId, int wounds, double[] extra) throws CardNotFoundException {
        double[] probs = new double[2];
        model.predict(vector(state, context, actionText, blueprintId, wounds, extra), probs);
        return probs[1];
    }

    /** The EXTRA numbers from the game, as the per-phase answerers compute theirs. */
    private static double[] liveExtra(GameState gs, int cardId, String blueprintId) {
        String holder = null;
        for (PhysicalCard c : gs.getInPlay())
            if (c.getCardId() == cardId && c.getAttachedTo() != null)
                holder = c.getAttachedTo().getBlueprintId();
        var skirmish = gs.getSkirmish();
        String fp = null;
        boolean fpCanExert = false, sourceIn = false;
        List<String> minions = new ArrayList<>();
        if (skirmish != null && skirmish.getFellowshipCharacter() != null) {
            var fc = skirmish.getFellowshipCharacter();
            fp = fc.getBlueprintId();
            fpCanExert = gs.getWounds(fc) + 1 < fc.getBlueprint().getVitality();
            sourceIn = fc.getCardId() == cardId;
            for (PhysicalCard m : skirmish.getShadowCharacters())
                minions.add(m.getBlueprintId());
        }
        return extra(blueprintId, holder, fp, fpCanExert, minions, sourceIn);
    }

    /** An item's hand weapon / armor / shield class and its holder (Ring-bearer, cost, strength, vitality); in a skirmish,
     * whether the card can affect someone in it, whether it is the companion skirmishing, and whether that one can exert. */
    static double[] extra(String blueprintId, String holder, String fp, boolean fpCanExert, List<String> minions,
                          boolean sourceIn) {
        double[] e = new double[EXTRA];
        try {
            if (holder != null) {
                double[] item = CardFeatures.getItemCardFeatures(blueprintId, holder, 0);
                System.arraycopy(item, 3, e, 0, 7);
            }
            if (fp != null) {
                List<Boolean> no = new ArrayList<>();
                for (String ignored : minions)
                    no.add(false);
                e[7] = CardFeatures.getSkirmishPlayCardFeatures(blueprintId, fp, fpCanExert, minions, no, sourceIn)[3];
                e[8] = sourceIn ? 1 : 0;
                e[9] = fpCanExert ? 1 : 0;
            }
        } catch (CardNotFoundException | RuntimeException ignored) {
            // a card the item or skirmish features cannot describe: zeros, as for pass
        }
        return e;
    }

    /** The training vector of a stored step, as {@link #getAnswer} builds it in a game. */
    public static double[] stepVector(LearningStep step, boolean rich) throws CardNotFoundException {
        CardActionChoiceAction action = (CardActionChoiceAction) step.action;
        String blueprint = action.getSourceBlueprint() != null ? action.getSourceBlueprint() : CardFeatures.PASS;
        double[] extra = null;
        if (rich)
            extra = action.getSourceBlueprint() == null ? new double[EXTRA] : extra(blueprint, action.getHolderBlueprint(),
                    action.getSkirmishingFpBlueprintId(), action.getRemainingVitalityOnSkirmishFp() > 1,
                    action.getSkirmishingMinionBlueprintIds() != null ? action.getSkirmishingMinionBlueprintIds() : List.of(),
                    action.isSourceInSkirmish());
        return vector(step.state, context(step.decision.getText()),
                action.getSourceBlueprint() != null ? action.getActionText() : null, blueprint, action.getWoundsOnSource(),
                extra);
    }

    @Override
    protected List<LabeledPoint> extractTrainingData(List<LearningStep> steps) {
        List<LabeledPoint> data = new ArrayList<>();
        for (LearningStep step : steps) {
            if (!isStepRelevant(step))
                continue;
            try {
                data.add(new LabeledPoint(step.reward > 0 ? 1 : 0, stepVector(step, RICH_TRAINING), step.group));
            } catch (CardNotFoundException ignored) {
            }
        }
        return data;
    }

    private static double[] vector(double[] state, double[] context, String actionText, String blueprintId, int wounds)
            throws CardNotFoundException {
        return vector(state, context, actionText, blueprintId, wounds, null);
    }

    private static double[] vector(double[] state, double[] context, String actionText, String blueprintId, int wounds,
                                   double[] extra) throws CardNotFoundException {
        double[] card = CardFeatures.getCardFeatures(blueprintId, wounds);
        if (extra != null) {
            card = Arrays.copyOf(card, card.length + extra.length);
            System.arraycopy(extra, 0, card, card.length - extra.length, extra.length);
        }
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

    @Override
    public SoftClassifier<double[]> train(List<LearningStep> steps) {
        List<LabeledPoint> points = extractTrainingData(steps);
        if (!PER_CONTEXT_TRAINING || points.isEmpty())
            return trainWithPoints(points);
        // one model per phase context: the per-phase models' specialisation, with every option of a decision compared
        int offset = CardFeatures.version() >= 2 ? com.gempukku.lotro.bots.rl.fotrstarters.GeneralStateFeatures.length()
                : STATE_LENGTH;
        List<List<LabeledPoint>> parts = new ArrayList<>();
        for (int c = 0; c <= CONTEXTS.size(); c++)
            parts.add(new ArrayList<>());
        for (LabeledPoint p : points)
            parts.get(PerContextModel.contextOf(p.x(), offset, CONTEXTS.size())).add(p);
        List<SoftClassifier<double[]>> models = new ArrayList<>();
        for (List<LabeledPoint> part : parts)
            models.add(part.size() < 200 || part.stream().mapToInt(LabeledPoint::y).distinct().count() < 2 ? null
                    : trainWithPoints(part));
        SoftClassifier<double[]> all = trainWithPoints(points); // for a context with too few rows of its own
        return new PerContextModel(models, all, offset, CONTEXTS.size());
    }

    /** Dispatches each option to its decision's context model; the context is one-hot right after the state. */
    static final class PerContextModel implements SoftClassifier<double[]>, java.io.Serializable {
        @java.io.Serial
        private static final long serialVersionUID = 1L;
        private final List<SoftClassifier<double[]>> models;
        private final SoftClassifier<double[]> fallback;
        private final int offset, contexts;

        PerContextModel(List<SoftClassifier<double[]>> models, SoftClassifier<double[]> fallback, int offset, int contexts) {
            this.models = models;
            this.fallback = fallback;
            this.offset = offset;
            this.contexts = contexts;
        }

        static int contextOf(double[] x, int offset, int contexts) {
            for (int c = 0; c < contexts; c++)
                if (x[offset + c] == 1)
                    return c;
            return contexts;
        }

        private SoftClassifier<double[]> pick(double[] x) {
            SoftClassifier<double[]> m = models.get(contextOf(x, offset, contexts));
            return m != null ? m : fallback;
        }

        @Override
        public int predict(double[] x) {
            return pick(x).predict(x);
        }

        @Override
        public int predict(double[] x, double[] posteriori) {
            return pick(x).predict(x, posteriori);
        }
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
