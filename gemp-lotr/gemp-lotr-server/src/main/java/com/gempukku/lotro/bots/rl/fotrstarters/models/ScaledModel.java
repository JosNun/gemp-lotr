package com.gempukku.lotro.bots.rl.fotrstarters.models;

import smile.classification.SoftClassifier;

/**
 * A model fit on standardised inputs, with the standardisation kept alongside so it can be handed raw vectors.
 * Logistic regression on raw counts -- deck sizes near 50 beside 0/1 flags -- converges badly in the iterations smile
 * gives it.
 */
public final class ScaledModel implements SoftClassifier<double[]>, java.io.Serializable {
    private final double[] mean, sd;
    private final SoftClassifier<double[]> inner;

    private ScaledModel(double[] mean, double[] sd, SoftClassifier<double[]> inner) {
        this.mean = mean;
        this.sd = sd;
        this.inner = inner;
    }

    public static ScaledModel logistic(double[][] x, int[] y, double lambda) {
        int dims = x[0].length;
        double[] mean = new double[dims], sd = new double[dims];
        for (double[] row : x)
            for (int d = 0; d < dims; d++)
                mean[d] += row[d] / x.length;
        for (double[] row : x)
            for (int d = 0; d < dims; d++)
                sd[d] += (row[d] - mean[d]) * (row[d] - mean[d]) / x.length;
        for (int d = 0; d < dims; d++)
            sd[d] = sd[d] > 1e-12 ? Math.sqrt(sd[d]) : 1;
        double[][] z = new double[x.length][];
        for (int i = 0; i < x.length; i++)
            z[i] = scale(x[i], mean, sd);
        return new ScaledModel(mean, sd, smile.classification.LogisticRegression.fit(z, y, lambda, 1e-5, 500));
    }

    private static double[] scale(double[] x, double[] mean, double[] sd) {
        double[] z = new double[x.length];
        for (int d = 0; d < x.length; d++)
            z[d] = (x[d] - mean[d]) / sd[d];
        return z;
    }

    @Override
    public int predict(double[] x) {
        return inner.predict(scale(x, mean, sd));
    }

    @Override
    public int predict(double[] x, double[] posteriori) {
        return inner.predict(scale(x, mean, sd), posteriori);
    }
}
