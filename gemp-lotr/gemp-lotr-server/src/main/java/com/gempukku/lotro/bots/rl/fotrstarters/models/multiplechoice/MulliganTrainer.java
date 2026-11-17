package com.gempukku.lotro.bots.rl.fotrstarters.models.multiplechoice;


public class MulliganTrainer extends AbstractMultipleChoiceTrainer {
    @Override
    protected String getTextTrigger() {
        return "mulligan";
    }

    @Override
    protected String getPositiveOption() {
        return "Yes";
    }

    @Override
    protected String getNegativeOption() {
        return "No";
    }

    /** The model directory's "mulligan-threshold" (0.5 without the file): the search mulligans where the model keeps
     *  far more often than the other way (3,925 against 343 of 12,099 searched fully, 2026-11-17). */
    @Override
    protected double threshold(com.gempukku.lotro.bots.rl.fotrstarters.models.ModelRegistry modelRegistry) {
        return modelRegistry.getMulliganThreshold();
    }
}
