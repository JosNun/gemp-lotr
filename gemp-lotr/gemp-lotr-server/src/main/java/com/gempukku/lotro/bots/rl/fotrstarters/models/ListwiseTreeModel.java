package com.gempukku.lotro.bots.rl.fotrstarters.models;

import smile.base.cart.CART;
import smile.base.cart.Loss;
import smile.classification.SoftClassifier;
import smile.data.DataFrame;
import smile.data.Tuple;
import smile.data.type.DataTypes;
import smile.data.type.StructField;
import smile.data.type.StructType;
import smile.regression.RegressionTree;

import java.io.Serial;
import java.io.Serializable;
import java.util.*;

/**
 * Boosted trees fit to the choice within each decision (a conditional logit, the softmax over a decision's options):
 * each round fits a regression tree to {@code chosen - P(option | decision)} and adds it to every option's score. A
 * per-option yes/no model ({@link TreeModel}) is fit on rows pooled across decisions, so an option's score carries how
 * often options like it are taken anywhere -- passing, offered in every decision, crowds out a card people play when
 * it is offered (Double Shot: taken 54% by people, ranked first 25% of the time by the yes/no trees). Here only the
 * options on offer together are compared.
 * <p>
 * Scores are not probabilities: {@link #predict(double[], double[])} returns the logistic of the score, which keeps
 * the argmax, the one thing the answerers use.
 */
public final class ListwiseTreeModel implements SoftClassifier<double[]>, Serializable {
    @Serial
    private static final long serialVersionUID = 1L;

    static final int TREES = Integer.getInteger("trees.n", 150), MAX_DEPTH = 20,
            MAX_NODES = Integer.getInteger("trees.nodes", 16), NODE_SIZE = 20;
    static final double SHRINKAGE = 0.1, SUBSAMPLE = 0.7;

    private final RegressionTree[] trees;
    private final StructType schema;

    private ListwiseTreeModel(RegressionTree[] trees, StructType schema) {
        this.trees = trees;
        this.schema = schema;
    }

    /** {@code group[i]}: which decision row i is an option of; a decision's rows need not be adjacent. */
    public static ListwiseTreeModel fit(double[][] x, int[] y, int[] group) {
        int n = x.length;
        Map<Integer, List<Integer>> byGroup = new LinkedHashMap<>();
        for (int i = 0; i < n; i++)
            byGroup.computeIfAbsent(group[i], k -> new ArrayList<>()).add(i);
        // only decisions with a choice and something to compare it with teach anything
        List<int[]> groups = new ArrayList<>();
        for (List<Integer> g : byGroup.values()) {
            int pos = 0;
            for (int i : g)
                pos += y[i];
            if (pos >= 1 && g.size() >= 2)
                groups.add(g.stream().mapToInt(Integer::intValue).toArray());
        }
        DataFrame data = DataFrame.of(x);
        int[][] order = CART.order(data);
        StructField response = new StructField("residual", DataTypes.DoubleType);
        double[] score = new double[n];
        double[] residual = new double[n];
        RegressionTree[] trees = new RegressionTree[TREES];
        Random random = new Random(n);
        for (int t = 0; t < TREES; t++) {
            Arrays.fill(residual, 0);
            int[] samples = new int[n];
            for (int[] g : groups) {
                if (random.nextDouble() >= SUBSAMPLE)
                    continue; // whole decisions in or out: the softmax is over a decision
                double max = Double.NEGATIVE_INFINITY, sum = 0, pos = 0;
                for (int i : g) {
                    max = Math.max(max, score[i]);
                    pos += y[i];
                }
                for (int i : g)
                    sum += Math.exp(score[i] - max);
                for (int i : g) {
                    double p = Math.exp(score[i] - max) / sum;
                    residual[i] = y[i] / pos - p; // gradient of the log-likelihood of the choice
                    samples[i] = 1;
                }
            }
            RegressionTree tree = new RegressionTree(data, Loss.ls(residual), response, MAX_DEPTH, MAX_NODES,
                    NODE_SIZE, x[0].length, samples, order);
            trees[t] = tree;
            for (int i = 0; i < n; i++)
                score[i] += SHRINKAGE * tree.predict(data.get(i));
        }
        return new ListwiseTreeModel(trees, data.schema());
    }

    public double score(double[] x) {
        Tuple tuple = Tuple.of(x, schema);
        double s = 0;
        for (RegressionTree tree : trees)
            s += SHRINKAGE * tree.predict(tuple);
        return s;
    }

    @Override
    public int predict(double[] x) {
        return score(x) > 0 ? 1 : 0;
    }

    @Override
    public int predict(double[] x, double[] posteriori) {
        double p = 1 / (1 + Math.exp(-score(x)));
        posteriori[0] = 1 - p;
        posteriori[1] = p;
        return p > 0.5 ? 1 : 0;
    }

    /** A self-check on made-up decisions: the option with the larger first feature is the one chosen. */
    public static void main(String[] args) {
        Random r = new Random(1);
        int decisions = 2000, opts = 4;
        double[][] x = new double[decisions * opts][];
        int[] y = new int[x.length], g = new int[x.length];
        for (int d = 0; d < decisions; d++) {
            int best = -1;
            double bestV = -1;
            for (int o = 0; o < opts; o++) {
                int i = d * opts + o;
                // a common offset per decision: pooled, the first feature alone does not say which option wins
                double offset = r.nextGaussian() * 3;
                x[i] = new double[]{offset + r.nextDouble(), r.nextDouble()};
                g[i] = d;
            }
            for (int o = 0; o < opts; o++) {
                double v = x[d * opts + o][0];
                if (v > bestV) {
                    bestV = v;
                    best = o;
                }
            }
            y[d * opts + best] = 1;
        }
        ListwiseTreeModel m = fit(x, y, g);
        int right = 0;
        for (int d = 0; d < decisions; d++) {
            int arg = 0;
            double top = Double.NEGATIVE_INFINITY;
            for (int o = 0; o < opts; o++) {
                double s = m.score(x[d * opts + o]);
                if (s > top) {
                    top = s;
                    arg = o;
                }
            }
            right += y[d * opts + arg];
        }
        System.out.printf("picked the chosen option in %d of %d decisions%n", right, decisions);
    }
}
