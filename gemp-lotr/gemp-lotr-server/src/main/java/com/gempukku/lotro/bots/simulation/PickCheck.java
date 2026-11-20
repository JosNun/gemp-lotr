package com.gempukku.lotro.bots.simulation;

import com.gempukku.lotro.bots.ModelDirectory;
import com.gempukku.lotro.bots.rl.LearningStep;
import com.gempukku.lotro.bots.rl.fotrstarters.models.ModelRegistry;
import com.gempukku.lotro.bots.rl.fotrstarters.models.cardselection.CharacterChoiceTrainer;
import com.gempukku.lotro.bots.rl.fotrstarters.models.cardselection.ExertChoiceTrainer;
import com.gempukku.lotro.bots.rl.semanticaction.CardSelectionAction;
import smile.classification.SoftClassifier;

import java.io.BufferedReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.TreeMap;

/**
 * How often a model directory's wound and heal pick model (CharacterChoiceTrainer) names the pick written in a steps
 * file, by prompt, beside chance (one in the number of options). The 2026-11-20 diagnostic: a model refit from the
 * search's picks that cannot name them on picks held out has not been shown what decides them, or they are noise.
 *
 * <pre>PickCheck &lt;model dir&gt; &lt;steps.jsonl&gt;</pre>
 */
public class PickCheck {
    public static void main(String[] args) throws Exception {
        BotVsBot.context(BotVsBot.defaultResources(), Path.of(args[0]));
        ModelRegistry registry = ModelDirectory.load(Path.of(args[0]));
        com.gempukku.lotro.bots.rl.fotrstarters.CardFeatures.setVersion(registry.getFeatureVersion());
        SoftClassifier<double[]> model = registry.getModel(CharacterChoiceTrainer.class);
        CharacterChoiceTrainer trainer = new CharacterChoiceTrainer();
        Map<String, double[]> by = new TreeMap<>(); // prompt -> {picks, named, chance}
        try (BufferedReader in = Files.newBufferedReader(Path.of(args[1]))) {
            for (String line; (line = in.readLine()) != null; ) {
                LearningStep step;
                try {
                    step = LearningStep.fromJson(line);
                } catch (RuntimeException e) {
                    continue;
                }
                if (!trainer.isStepRelevant(step))
                    continue;
                CardSelectionAction a = (CardSelectionAction) step.action;
                int best = -1;
                double top = Double.NEGATIVE_INFINITY;
                for (int i = 0; i < a.getOptionBlueprints().length; i++) {
                    double[] e = a.getOptionExtras()[i], p = new double[2];
                    model.predict(ExertChoiceTrainer.vector(step.state, a.getOptionBlueprints()[i], (int) e[8],
                            CharacterChoiceTrainer.shaped(e, registry.isCharacterWide())), p);
                    if (p[1] > top) {
                        top = p[1];
                        best = i;
                    }
                }
                String t = step.decision.getText().replaceAll("[0-9]+", "#");
                for (String key : new String[]{t.length() > 44 ? t.substring(0, 44) : t, "all"}) {
                    double[] c = by.computeIfAbsent(key, k -> new double[3]);
                    c[0]++;
                    c[1] += best == a.getOptionChosen() ? 1 : 0;
                    c[2] += 1.0 / a.getOptionBlueprints().length;
                }
            }
        }
        by.forEach((k, c) -> System.out.printf("%7.0f picks  named %5.1f%%  chance %5.1f%%  %s%n", c[0], 100 * c[1] / c[0],
                100 * c[2] / c[0], k));
        System.exit(0);
    }
}
