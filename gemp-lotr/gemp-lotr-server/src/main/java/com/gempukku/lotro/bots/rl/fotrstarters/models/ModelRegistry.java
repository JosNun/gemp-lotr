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
    /** AnyPhaseCardActionTrainer's vectors carry its EXTRA numbers: an "any-phase-rich" file in the model directory. */
    private boolean anyPhaseRich;
    /** Skirmish options carry what they did on a copy (EffectProbe): a "skirmish-probe" file in the model directory. */
    private boolean skirmishProbe;

    /** Card-action contexts whose options carry all of EffectProbe's numbers: a "probe-contexts" file lists them. */
    private java.util.Set<String> probeContexts = java.util.Set.of();

    /** How many of EffectProbe's numbers those contexts were fit on: "length=N" in the file, else 22. */
    private int probeLength = 22;

    public int getProbeLength() {
        return probeLength;
    }

    public void setProbeLength(int probeLength) {
        this.probeLength = probeLength;
    }

    public java.util.Set<String> getProbeContexts() {
        return probeContexts;
    }

    public void setProbeContexts(java.util.Set<String> probeContexts) {
        this.probeContexts = probeContexts;
    }

    /** ReconcileChoiceTrainer fit with only its keep-everything number: a "reconcile-plain" file. */
    private boolean reconcilePlain;

    public boolean isReconcilePlain() {
        return reconcilePlain;
    }

    public void setReconcilePlain(boolean reconcilePlain) {
        this.reconcilePlain = reconcilePlain;
    }

    /** Card actions pass only if passing outweighs every play together: a "pass-by-mass" file. */
    private boolean passByMass;

    public boolean isPassByMass() {
        return passByMass;
    }

    public void setPassByMass(boolean passByMass) {
        this.passByMass = passByMass;
    }

    /** Skirmish card selections go to the models, not to "the one fighting": a "no-skirmish-rule" file. */
    private boolean noSkirmishRule;

    public boolean isNoSkirmishRule() {
        return noSkirmishRule;
    }

    public void setNoSkirmishRule(boolean noSkirmishRule) {
        this.noSkirmishRule = noSkirmishRule;
    }

    /** Sides ("fp", "shadow") whose assignment model reads AssignmentProbe's numbers: an "assign-probe" file. */
    private java.util.Set<String> assignProbe = java.util.Set.of();

    public java.util.Set<String> getAssignProbe() {
        return assignProbe;
    }

    public void setAssignProbe(java.util.Set<String> assignProbe) {
        this.assignProbe = assignProbe;
    }

    public boolean isSkirmishProbe() {
        return skirmishProbe;
    }

    public void setSkirmishProbe(boolean skirmishProbe) {
        this.skirmishProbe = skirmishProbe;
    }

    public boolean isAnyPhaseRich() {
        return anyPhaseRich;
    }

    public void setAnyPhaseRich(boolean anyPhaseRich) {
        this.anyPhaseRich = anyPhaseRich;
    }

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
