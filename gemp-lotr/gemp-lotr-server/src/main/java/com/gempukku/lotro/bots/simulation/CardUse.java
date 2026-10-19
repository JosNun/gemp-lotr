package com.gempukku.lotro.bots.simulation;

import com.gempukku.lotro.bots.BotPlayer;
import com.gempukku.lotro.bots.rl.fotrstarters.GeneralStateFeatures;
import com.gempukku.lotro.game.state.GameState;
import com.gempukku.lotro.logic.decisions.AwaitingDecision;
import com.gempukku.lotro.logic.decisions.AwaitingDecisionType;

import java.io.PrintStream;
import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicIntegerArray;

/**
 * How a bot uses one card: each time a card action of that blueprint is offered, whether the bot took it, by the
 * Ring-bearer's resistance left and the skirmish margin. {@code -DcardUse=4_1} on BotVsBot wraps both bots. People
 * use The One Ring, Answer To All Riddles (4_1) in 2.1% of the 13,381 times it was offered in October-November 2022
 * Movie Block replays, mostly at resistance 1; a bot that wears it to win a skirmish turns every wound into a burden
 * and corrupts later.
 */
public final class CardUse implements BotPlayer {
    static final String BLUEPRINT = System.getProperty("cardUse");
    /** label -> bucket -> {offered, used} */
    private static final Map<String, Map<String, AtomicIntegerArray>> TALLY = new ConcurrentHashMap<>();

    private final BotPlayer inner;
    private final String label;

    CardUse(BotPlayer inner, String label) {
        this.inner = inner;
        this.label = label;
    }

    static BotPlayer wrap(BotPlayer bot, String label) {
        return BLUEPRINT == null ? bot : new CardUse(bot, label);
    }

    @Override
    public String chooseAction(GameState gameState, AwaitingDecision decision) {
        String answer = inner.chooseAction(gameState, decision);
        if (decision.getDecisionType() != AwaitingDecisionType.CARD_ACTION_CHOICE)
            return answer;
        try {
            String[] actions = decision.getDecisionParameters().get("actionId");
            String[] cards = decision.getDecisionParameters().get("cardId");
            if (actions == null || cards == null)
                return answer;
            if (BLUEPRINT.equals("all")) {
                // every card: offered/used per blueprint, for scripts/card_use.py to hold against the human rates
                Map<String, Boolean> seen = new java.util.HashMap<>();
                for (int i = 0; i < actions.length && i < cards.length; i++) {
                    var card = gameState.findCardById(Integer.parseInt(cards[i]));
                    if (card != null)
                        seen.merge(card.getBlueprintId(), actions[i].equals(answer), Boolean::logicalOr);
                }
                seen.forEach((bp, u) -> count(bp, u));
                return answer;
            }
            boolean offered = false, used = false;
            for (int i = 0; i < actions.length && i < cards.length; i++) {
                var card = gameState.findCardById(Integer.parseInt(cards[i]));
                if (card != null && BLUEPRINT.equals(card.getBlueprintId())) {
                    offered = true;
                    used |= actions[i].equals(answer);
                }
            }
            if (!offered)
                return answer;
            double[] f = new GeneralStateFeatures().extractFeatures(gameState, decision, getName());
            int res = (int) f[GeneralStateFeatures.index("myRbResistance")];
            int margin = (int) f[GeneralStateFeatures.index("mySkirmishMargin")];
            boolean skirmish = f[GeneralStateFeatures.index("inSkirmish")] > 0;
            count("all", used);
            count("resistance " + (res >= 8 ? "8+" : res <= 1 ? "<=1" : String.valueOf(res)), used);
            count(!skirmish ? "margin: not in a skirmish" : "margin " + (margin <= -3 ? "<=-3" : margin >= 3 ? ">=3"
                    : String.format("%+d", margin)), used);
        } catch (RuntimeException ignored) {
        }
        return answer;
    }

    private void count(String bucket, boolean used) {
        AtomicIntegerArray a = TALLY.computeIfAbsent(label, k -> new ConcurrentHashMap<>())
                .computeIfAbsent(bucket, k -> new AtomicIntegerArray(2));
        a.incrementAndGet(0);
        if (used)
            a.incrementAndGet(1);
    }

    static void report(PrintStream out) {
        if (BLUEPRINT == null)
            return;
        out.println("  use of " + BLUEPRINT + " when offered (used / offered):");
        new TreeMap<>(TALLY).forEach((label, buckets) -> new TreeMap<>(buckets).forEach((b, a) ->
                out.printf("    %-3s %-28s %5.1f%%  %6d / %d%n", label, b, 100.0 * a.get(1) / a.get(0), a.get(1), a.get(0))));
    }

    @Override
    public String getName() {
        return inner.getName();
    }
}
