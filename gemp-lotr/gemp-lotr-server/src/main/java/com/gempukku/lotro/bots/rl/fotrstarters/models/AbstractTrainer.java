package com.gempukku.lotro.bots.rl.fotrstarters.models;

import com.gempukku.lotro.bots.rl.DecisionAnswerer;
import com.gempukku.lotro.bots.rl.LearningStep;
import smile.classification.LogisticRegression;
import smile.classification.SoftClassifier;

import java.util.*;

public abstract class AbstractTrainer implements Trainer, DecisionAnswerer {
    /**
     * Fit logistic regressions (as the shipped bot-models/ are) or {@link TreeModel}s. Set once, before training,
     * by BotTrain --model; a model directory may then hold either kind.
     */
    public static volatile boolean useTrees = false;

    /**
     * L2 penalty for the logistic fits; 0, Smile's default, is what every model so far was fit with. Hundreds of
     * thousands of self-play rows did not need it; a few hundred teacher labels against ~300 inputs do.
     */
    public static volatile double lambda = 0;

    /** Fit {@link ListwiseTreeModel}s where the rows carry their decision (BotTeach --model listwise). */
    public static volatile boolean listwise = false;

    abstract protected List<LabeledPoint> extractTrainingData(List<LearningStep> steps);

    /**
     * Rows a model is fit on at most, sampled without replacement. Set-2 vectors are ~5x set 1's and the big
     * card-action models have over a million rows by the third generation; a guess at where more stops paying.
     */
    // -Dtrain.maxRows: this cap applies before TreeModel's own (trees.maxRows), so until it was configurable every
    // human-label refit -- 1.4-3.8 M rows -- was fit on 300,000 of them, whatever trees.maxRows said
    private static final int MAX_ROWS = Integer.getInteger("train.maxRows", 300_000);

    public SoftClassifier<double[]> trainWithPoints(List<LabeledPoint> points) {
        if (listwise && useTrees && !points.isEmpty() && points.stream().allMatch(p -> p.group() >= 0))
            return listwiseFit(points);
        if (points.size() > MAX_ROWS) {
            points = new java.util.ArrayList<>(points);
            java.util.Collections.shuffle(points, new java.util.Random(points.size()));
            points = points.subList(0, MAX_ROWS);
        }
        if (points.stream().mapToInt(LabeledPoint::y).distinct().count() < 2)
            return new ConstantModel(points.isEmpty() ? 0 : points.get(0).y());
        double[][] x = points.stream().map(LabeledPoint::x).toArray(double[][]::new);
        int[] y = points.stream().mapToInt(LabeledPoint::y).toArray();
        return useTrees ? TreeModel.fit(x, y) : lambda > 0 ? LogisticRegression.fit(x, y, lambda, 1e-5, 500)
                : LogisticRegression.fit(x, y);
    }

    /** The cap taken as whole decisions, at random: a softmax over half a decision's options is a different one. */
    private static SoftClassifier<double[]> listwiseFit(List<LabeledPoint> points) {
        Map<Integer, List<LabeledPoint>> byGroup = new LinkedHashMap<>();
        for (LabeledPoint p : points)
            byGroup.computeIfAbsent(p.group(), k -> new ArrayList<>()).add(p);
        List<List<LabeledPoint>> groups = new ArrayList<>(byGroup.values());
        Collections.shuffle(groups, new Random(points.size()));
        List<LabeledPoint> kept = new ArrayList<>();
        for (List<LabeledPoint> g : groups) {
            if (kept.size() + g.size() > MAX_ROWS)
                break;
            kept.addAll(g);
        }
        double[][] x = kept.stream().map(LabeledPoint::x).toArray(double[][]::new);
        int[] y = kept.stream().mapToInt(LabeledPoint::y).toArray();
        int[] group = kept.stream().mapToInt(LabeledPoint::group).toArray();
        return ListwiseTreeModel.fit(x, y, group);
    }

    @Override
    public SoftClassifier<double[]> train(List<LearningStep> steps) {
        return trainWithPoints(extractTrainingData(steps));
    }
}
