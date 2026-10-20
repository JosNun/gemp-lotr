package com.gempukku.lotro.bots.rl.fotrstarters.models;

import smile.classification.SoftClassifier;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

public class ModelRegistry {
    private final Map<Class<? extends Trainer>, SoftClassifier<double[]>> modelMap = new HashMap<>();
    /** The feature set (CardFeatures.version()) these models were trained on: 1 unless a model dir says 2. */
    private int featureVersion = 1;
    /**
     * Card actions go to AnyPhaseCardActionTrainer first, not the per-phase answerers: set by a file "any-phase-first"
     * in the model directory. The single model is the one ScoreProbe can hold against human decisions offline.
     */
    private boolean anyPhaseFirst;

    public boolean isAnyPhaseFirst() {
        return anyPhaseFirst;
    }

    public void setAnyPhaseFirst(boolean anyPhaseFirst) {
        this.anyPhaseFirst = anyPhaseFirst;
    }

    public int getFeatureVersion() {
        return featureVersion;
    }

    public void setFeatureVersion(int featureVersion) {
        this.featureVersion = featureVersion;
    }

    public <T extends Trainer> void registerModel(Class<T> trainerClass, SoftClassifier<double[]> model) {
        modelMap.put(trainerClass, model);
    }

    public <T extends Trainer> SoftClassifier<double[]> getModel(Class<T> trainerClass) {
        return modelMap.get(trainerClass);
    }

    public boolean hasModel(Class<? extends Trainer> trainerClass) {
        return modelMap.containsKey(trainerClass);
    }

    public Set<Class<? extends Trainer>> getRegisteredTrainerTypes() {
        return Collections.unmodifiableSet(modelMap.keySet());
    }
}
