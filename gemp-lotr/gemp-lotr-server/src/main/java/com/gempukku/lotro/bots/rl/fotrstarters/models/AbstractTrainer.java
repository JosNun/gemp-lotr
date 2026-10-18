package com.gempukku.lotro.bots.rl.fotrstarters.models;

import com.gempukku.lotro.bots.rl.DecisionAnswerer;
import com.gempukku.lotro.bots.rl.LearningStep;
import smile.classification.LogisticRegression;
import smile.classification.SoftClassifier;

import java.util.List;

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

    abstract protected List<LabeledPoint> extractTrainingData(List<LearningStep> steps);

    /**
     * Rows a model is fit on at most, sampled without replacement. Set-2 vectors are ~5x set 1's and the big
     * card-action models have over a million rows by the third generation; a guess at where more stops paying.
     */
    private static final int MAX_ROWS = 300_000;

    public SoftClassifier<double[]> trainWithPoints(List<LabeledPoint> points) {
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

    @Override
    public SoftClassifier<double[]> train(List<LearningStep> steps) {
        return trainWithPoints(extractTrainingData(steps));
    }
}
