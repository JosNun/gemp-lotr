package com.gempukku.lotro.bots.rl.fotrstarters.models.arbitrarycards;

/**
 * Which card a search of the draw deck fetches. Without a model the bot answered it at random: against people, in
 * October 2022's games, it agreed with them 14% of the time and lost 11 points where it differed.
 */
public class CardFromDeckTrainer extends AbstractArbitraryCardsTrainer {
    @Override
    protected String getTextTrigger() {
        return "choose cards from deck";
    }

    @Override
    protected boolean useNotChosen() {
        return true;
    }
}
