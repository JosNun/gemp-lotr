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
    /**
     * Opponent-adaptive racing (2026-11-07): moving on pays against a Shadow side that does not punish it and costs
     * against one that does (bots: a 0.25 threshold lost 1.5 points to sp-m2; people's positions: it won 9.5). The
     * minions the opponent sent against this player in each of its turns so far, read at its assignments; with a
     * "move-adaptive" file ("ref slope min max"), the threshold moves by slope per minion above ref (both sides average
     * ~1.8 minions in play as the Shadow phase ends), clamped to [min, max].
     */
    private static final java.util.Map<Object, java.util.Map<String, java.util.Map<Integer, Integer>>> MINIONS =
            java.util.Collections.synchronizedMap(new java.util.WeakHashMap<>());

    /** FotrStarterBot: {@code player} is assigning as the Free Peoples player; the minions it faces this turn. */
    public static void observeAssignment(GameState gs, String player, AwaitingDecision decision) {
        int n = decision.getDecisionParameters().getOrDefault("minions", new String[0]).length;
        MINIONS.computeIfAbsent(gs.getGame(), g -> new java.util.concurrent.ConcurrentHashMap<>())
                .computeIfAbsent(player, p -> new java.util.concurrent.ConcurrentHashMap<>())
                .merge(gs.getTurnNumber(), n, Math::max);
    }

    static double threshold(GameState gs, String player, ModelRegistry registry) {
        double base = registry.getMoveThreshold();
        double[] a = registry.getMoveAdaptive();
        if (a == null)
            return base;
        var byPlayer = MINIONS.get(gs.getGame());
        var turns = byPlayer == null ? null : byPlayer.get(player);
        if (turns == null || turns.isEmpty())
            return base;
        double avg = turns.values().stream().mapToInt(Integer::intValue).average().orElse(a[0]);
        return Math.max(a[2], Math.min(a[3], base + a[1] * (avg - a[0])));
    }

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
        double threshold = threshold(gameState, playerName, modelRegistry);
        boolean yes = probs[1] > threshold;
        var net2 = modelRegistry.getBoardNet2();
        if (net2 != null && getPositiveOption().equals("Yes")) {
            // log-odds of moving: the network's, mixed with the trees' against their threshold
            var r = net2.read(gameState, decision, playerName);
            if (r != null && r.options().length == 2) {
                double w = net2.treeWeight("move");
                double net = r.options()[0] - r.options()[1];
                double trees = logOdds(probs[1]) - logOdds(threshold);
                yes = net2.chooseYes((1 - w) * net + w * trees);
            }
        }
        String option = yes ? getPositiveOption() : getNegativeOption();
        return new MultipleChoiceAction(option).toDecisionString(decision, gameState);
    }
}
