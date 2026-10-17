package com.gempukku.lotro.bots.rl.fotrstarters.models;

import smile.classification.SoftClassifier;

/**
 * A model with nothing to go on: every input gets the same answer, so every option of a decision ties and the
 * first is taken. Stands in where a model could not be fit -- all its examples had one label, or there were none --
 * and no earlier model trained on the same feature set exists to keep.
 */
public final class ConstantModel implements SoftClassifier<double[]> {
    private final int label;

    public ConstantModel(int label) {
        this.label = label;
    }

    @Override
    public int predict(double[] x) {
        return label;
    }

    @Override
    public int predict(double[] x, double[] posteriori) {
        java.util.Arrays.fill(posteriori, 0);
        if (label < posteriori.length)
            posteriori[label] = 1;
        return label;
    }
}
