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
        try {
            widen(gs, decision, cardIds, extras);
        } catch (RuntimeException e) {
            // a board the wider numbers cannot be read off: the first BASE stand, the rest zeros
            for (int i = 0; i < extras.length; i++)
                extras[i] = Arrays.copyOf(extras[i], WIDE);
        }
        return new ReconcileChoiceTrainer.Options(blueprints, extras);
    }

    /** The numbers a pick was described by until 2026-11-20, and with what {@link #widen} adds. */
    public static final int BASE = 15, WIDE = 49;
    /** -Dcharacter.wide: fit on the wider numbers (rows written before them are left out). */
    static final boolean FIT_WIDE = Boolean.getBoolean("character.wide");
    private static final java.util.regex.Pattern REMAINING = java.util.regex.Pattern.compile("remaining \\w+: *(\\d+)");

    /** {@code e} as a model reads it: the wider numbers where it was fit on them, else the first BASE. */
    public static double[] shaped(double[] e, boolean wide) {
        return wide ? (e.length == WIDE ? e : Arrays.copyOf(e, WIDE)) : (e.length == BASE ? e : Arrays.copyOf(e, BASE));
    }

    /**
     * What decides a wound or a heal that the first numbers do not say (2026-11-20: a model fit on the search's picks
     * alone named 67% of them held out, where the search names its own 95%). The candidate: resistance, printed cost,
     * cards it bears, companion / ally / minion, archer, defender, damage, fierce, enduring, the minions assigned to
     * it and their strength. The prompt: wounds still to place, how many candidates. The board from the candidate's
     * side: the other side's characters, their strength in all and at most, its own side's characters; its strength
     * against the strongest and the weakest of the other side, whether the strongest overwhelms it. And where it
     * stands among the candidates, which a model scoring one candidate at a time cannot see: its rank by vitality
     * left, by strength, by wounds and by cost, most and least of each, the share of candidates on its own side.
     */
    private static void widen(GameState gs, AwaitingDecision decision, String[] cardIds, double[][] extras) {
        var game = gs.getGame();
        var q = game.getModifiersQuerying();
        int n = cardIds.length;
        PhysicalCard[] cards = new PhysicalCard[n];
        for (int i = 0; i < n; i++)
            cards[i] = gs.findCardById(Integer.parseInt(cardIds[i]));
        double remaining = 0;
        if (decision.getText() != null) {
            var m = REMAINING.matcher(decision.getText().toLowerCase(Locale.ROOT));
            if (m.find())
                remaining = Integer.parseInt(m.group(1));
        }
        // the characters in play by side: Shadow's minions, the Free Peoples' companions and allies
        List<PhysicalCard> minions = new ArrayList<>(), free = new ArrayList<>();
        for (PhysicalCard card : gs.getInPlay()) {
            var type = card.getBlueprint().getCardType();
            if (type == com.gempukku.lotro.common.CardType.MINION)
                minions.add(card);
            else if (type == com.gempukku.lotro.common.CardType.COMPANION || type == com.gempukku.lotro.common.CardType.ALLY)
                free.add(card);
        }
        double[] cost = new double[n];
        double mine = 0;
        for (int i = 0; i < n; i++) {
            cost[i] = cards[i].getBlueprint().getTwilightCost();
            mine += extras[i][0] > 0 ? 1 : 0;
        }
        for (int i = 0; i < n; i++) {
            PhysicalCard card = cards[i];
            double[] e = Arrays.copyOf(extras[i], WIDE);
            var type = card.getBlueprint().getCardType();
            boolean minion = type == com.gempukku.lotro.common.CardType.MINION;
            int k = BASE;
            double resistance = 0;
            if (type == com.gempukku.lotro.common.CardType.COMPANION)
                try {
                    resistance = q.getResistance(game, card);
                } catch (RuntimeException ignored) {
                }
            e[k++] = resistance;
            e[k++] = cost[i];
            e[k++] = gs.getAttachedCards(card).size();
            e[k++] = type == com.gempukku.lotro.common.CardType.COMPANION ? 1 : type == com.gempukku.lotro.common.CardType.ALLY ? 2 : minion ? 3 : 0;
            for (var kw : new com.gempukku.lotro.common.Keyword[]{com.gempukku.lotro.common.Keyword.ARCHER,
                    com.gempukku.lotro.common.Keyword.DEFENDER, com.gempukku.lotro.common.Keyword.DAMAGE,
                    com.gempukku.lotro.common.Keyword.FIERCE, com.gempukku.lotro.common.Keyword.ENDURING})
                e[k++] = q.getKeywordCount(game, card, kw);
            double facing = 0, facingStrength = 0;
            for (var a : gs.getAssignments())
                if (a.getFellowshipCharacter() == card)
                    for (PhysicalCard m : a.getShadowCharacters()) {
                        facing++;
                        facingStrength += q.getStrength(game, m);
                    }
                else if (a.getShadowCharacters().contains(card)) {
                    facing++;
                    facingStrength += q.getStrength(game, a.getFellowshipCharacter());
                }
            e[k++] = facing;
            e[k++] = facingStrength;
            e[k++] = remaining;
            e[k++] = n;
            List<PhysicalCard> other = minion ? free : minions, own = minion ? minions : free;
            double total = 0, most = 0, least = Double.POSITIVE_INFINITY;
            for (PhysicalCard o : other) {
                double st = q.getStrength(game, o);
                total += st;
                most = Math.max(most, st);
                least = Math.min(least, st);
            }
            if (other.isEmpty())
                least = 0;
            e[k++] = other.size();
            e[k++] = total;
            e[k++] = most;
            e[k++] = own.size();
            e[k++] = e[7] - most;
            e[k++] = e[7] - least;
            e[k++] = most >= 2 * e[7] && most > 0 ? 1 : 0;
            // among the candidates: vitality left (index 9 + KINDS = "after"), strength, wounds, cost
            int after = 9 + KINDS.size();
            k = rank(extras, i, after, e, k, null);
            k = rank(extras, i, 7, e, k, null);
            k = rank(extras, i, 8, e, k, null);
            k = rank(extras, i, -1, e, k, cost);
            e[k++] = mine / n;
            e[k++] = e[after] >= 2 ? 1 : 0;
            extras[i] = e;
        }
    }

    /** Candidate {@code i}'s share of the others below it on column {@code col} (or {@code values}), whether it is
     * the most, whether the least: three numbers written from {@code k}. */
    private static int rank(double[][] extras, int i, int col, double[] e, int k, double[] values) {
        double v = values != null ? values[i] : extras[i][col];
        int below = 0, above = 0;
        for (int j = 0; j < extras.length; j++) {
            if (j == i)
                continue;
            double w = values != null ? values[j] : extras[j][col];
            below += w < v ? 1 : 0;
            above += w > v ? 1 : 0;
        }
        e[k++] = extras.length > 1 ? below / (double) (extras.length - 1) : 0;
        e[k++] = above == 0 ? 1 : 0;
        e[k++] = below == 0 ? 1 : 0;
        return k;
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
                model.predict(ExertChoiceTrainer.vector(state, o.blueprints()[i], (int) o.extras()[i][8],
                        shaped(o.extras()[i], modelRegistry.isCharacterWide())), p);
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
                    if (FIT_WIDE && e.length < WIDE)
                        continue; // written before the wider numbers
                    data.add(new LabeledPoint(i == a.getOptionChosen() ? 1 : 0,
                            ExertChoiceTrainer.vector(step.state, a.getOptionBlueprints()[i], (int) e[8], shaped(e, FIT_WIDE)), group));
                } catch (CardNotFoundException ignored) {
                }
            }
        }
        return data;
    }
}
