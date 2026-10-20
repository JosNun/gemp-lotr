package com.gempukku.lotro.bots.rl.fotrstarters.models;

/** {@code group}: the decision this row is an option of, for {@link ListwiseTreeModel}; -1 when not known. */
public record LabeledPoint(int y, double[] x, int group) {
    public LabeledPoint(int y, double[] x) {
        this(y, x, -1);
    }
}