package com.gempukku.lotro.bots.rl.fotrstarters.models.integerchoice;


import com.gempukku.lotro.bots.rl.RLGameStateFeatures;
import com.gempukku.lotro.bots.rl.fotrstarters.models.ModelRegistry;
import com.gempukku.lotro.game.state.GameState;
import com.gempukku.lotro.logic.decisions.AwaitingDecision;


public class BurdenTrainer extends AbstractIntegerTrainer {
    private static final int MAX_BURDENS_TO_BID = 9;

    @Override
    protected String getTextTrigger() {
        return "burdens to bid";
    }

    public int getMaxChoice() {
        return MAX_BURDENS_TO_BID;
    }

    @Override
    public String getAnswer(GameState gameState, AwaitingDecision decision, String playerName, RLGameStateFeatures features, ModelRegistry modelRegistry) {
        // Was a uniform 0-4, from when the ring-bearer could not be assumed to be Frodo. Against real decks that
        // corrupted low-resistance ring-bearers (Galadriel, Bearer of Wisdom, starts at 3) before the first turn:
        // 277 of 4,800 games between the most-played Movie Block decks. People bid low -- Movie Block, Feb 2025:
        // 0 67%, 1 24%, 2 8% -- and never so much that the ring-bearer starts within a few burdens of corruption.
        int resistance = 10;
        try {
            String ringBearer = gameState.getLotroDeck(playerName).getRingBearer();
            resistance = gameState.getGame().getLotroCardBlueprintLibrary().getLotroCardBlueprint(ringBearer).getResistance();
        } catch (Exception ignored) {
            // no deck or no such card: bid as for a resistance-10 ring-bearer
        }
        if (resistance < 8)
            return "0";
        double r = com.gempukku.lotro.bots.BotRandom.current().nextDouble();
        return r < 0.67 ? "0" : r < 0.92 ? "1" : "2";
    }
}
