package com.gempukku.lotro.bots.rl.fotrstarters.models;

import smile.classification.GradientTreeBoost;
import smile.classification.SoftClassifier;
import smile.data.DataFrame;
import smile.data.Tuple;
import smile.data.formula.Formula;
import smile.data.type.StructType;
import smile.data.vector.IntVector;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Gradient-boosted trees behind the same {@code SoftClassifier<double[]>} the trainers and bots already use, so a
 * model directory can hold either kind and nothing that reads it needs to know.
 * <p>
 * Why trees: every option of a decision is scored as state + option features, and in a logistic regression the
 * state's contribution is the same for every option, so it cannot change which one wins. The logistic models rank
 * cards the same way whatever the twilight, wounds or board. Trees split on state and card together.
 */
public final class TreeModel implements SoftClassifier<double[]> {
    // pinned to the value it had before trees.maxRows: saved models carry it, and a changed field initialiser
    // changes the computed one
    private static final long serialVersionUID = 5876436858692950002L;

    /**
     * Trees, depth, leaves, leaf size, shrinkage, subsample. Smile's defaults are 500 trees at 0.05; 150 at 0.1 is
     * a guess at the same fit for a third of the training time, which is paid for 34 models every generation.
     */
    private static final int TREES = 150, MAX_DEPTH = 20, MAX_NODES = 16, NODE_SIZE = 20;
    private static final double SHRINKAGE = 0.1, SUBSAMPLE = 0.7;
    /** Rows a model is fit on at most; the big card-action models have ~10x this. Sampled without replacement. */
    // -Dtrees.maxRows: the human-game labels run to millions of rows a month (BOT-NOTES.md, HumanReplay)
    private static final int MAX_ROWS = Integer.getInteger("trees.maxRows", 150_000);

    private final GradientTreeBoost model;
    private final StructType schema;

    private TreeModel(GradientTreeBoost model, StructType schema) {
        this.model = model;
        this.schema = schema;
    }

    public static TreeModel fit(double[][] x, int[] y) {
        if (x.length > MAX_ROWS) {
            List<Integer> rows = new ArrayList<>(x.length);
            for (int i = 0; i < x.length; i++)
                rows.add(i);
            Collections.shuffle(rows, new Random(x.length));
            double[][] sx = new double[MAX_ROWS][];
            int[] sy = new int[MAX_ROWS];
            for (int i = 0; i < MAX_ROWS; i++) {
                sx[i] = x[rows.get(i)];
                sy[i] = y[rows.get(i)];
            }
            x = sx;
            y = sy;
        }
        DataFrame features = DataFrame.of(x);
        DataFrame data = features.merge(IntVector.of("label", y));
        GradientTreeBoost model = GradientTreeBoost.fit(Formula.lhs("label"), data, TREES, MAX_DEPTH, MAX_NODES,
                NODE_SIZE, SHRINKAGE, SUBSAMPLE);
        return new TreeModel(model, features.schema());
    }

    @Override
    public int predict(double[] x) {
        return model.predict(Tuple.of(x, schema));
    }

    @Override
    public int predict(double[] x, double[] posteriori) {
        return model.predict(Tuple.of(x, schema), posteriori);
    }
}
