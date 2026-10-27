package com.gempukku.lotro.bots;

import com.gempukku.lotro.bots.rl.fotrstarters.AbilityFeatures;
import com.gempukku.lotro.bots.rl.fotrstarters.CardTextFeatures;
import com.gempukku.lotro.bots.rl.fotrstarters.CardFeatures;
import com.gempukku.lotro.bots.rl.fotrstarters.EffectProbe;
import com.gempukku.lotro.bots.rl.fotrstarters.models.ModelRegistry;
import com.gempukku.lotro.bots.rl.fotrstarters.models.Trainer;
import com.gempukku.lotro.bots.rl.fotrstarters.models.cardaction.AnyPhaseCardActionTrainer;
import com.gempukku.lotro.bots.simulation.GameCopier;
import com.gempukku.lotro.game.LotroCardBlueprintLibrary;
import com.gempukku.lotro.game.formats.LotroFormatLibrary;
import smile.classification.SoftClassifier;

import java.io.File;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * A model directory as the bots read it: one .model file per {@link BotService#extendedTrainers()} model, and the
 * marker files that say how those models were fit. The harness (BotVsBot and everything built on it) and the server
 * (BotService, given -Dbot.models) both load through here, so a directory plays the same bot in either.
 * <p>
 * Markers are read off the directory, not configured, because a model is only right when fed what it was fit on:
 * <ul>
 * <li>features.txt -- the card feature set (CardFeatures.version()); 1 without one.</li>
 * <li>any-phase-first, any-phase-rich, skirmish-probe, reconcile-plain, pass-by-mass, no-skirmish-rule, number-bid, character-outside-skirmish --
 *     empty files switching on the {@link ModelRegistry} flag of the same name.</li>
 * <li>move-threshold -- a number: AnotherMoveTrainer moves above it (0.5 without the file).</li>
 * <li>assign-probe -- the sides ("fp", "shadow") whose assignment model reads AssignmentProbe.</li>
 * <li>probe-contexts -- the card-action contexts fed all of EffectProbe's numbers, with an optional length=N.</li>
 * <li>probe-value.model -- the value model EffectProbe's last numbers come from; process-wide, see {@link #load}.</li>
 * </ul>
 * Some of what a directory needs is not in it: the card library behind the features, card abilities for feature set
 * 3, and a way to copy a live game for the probe. {@link #prepare} sets those, once per process.
 */
public final class ModelDirectory {
    private ModelDirectory() {
    }

    /** A directory without one of the models every bot needs. */
    public static final class MissingModels extends IOException {
        public MissingModels(String message) {
            super(message);
        }
    }

    /**
     * Every model and marker in {@code dir}. The extended bot's own models are optional: without one it answers those
     * decisions as the plain bot does. The rest are not, because ModelRegistry hands a missing one back as null and
     * the first decision it is asked turns into an NPE, answered by the fallback for the rest of the game.
     * <p>
     * A probe-value.model is installed as {@link EffectProbe#VALUE}, which every bot in the process reads: two
     * directories with different value models cannot play in one JVM.
     */
    public static ModelRegistry load(Path dir) throws IOException {
        ModelRegistry registry = new ModelRegistry();
        List<String> missing = new ArrayList<>();
        List<Trainer> required = BotService.allTrainers();
        for (Trainer trainer : BotService.extendedTrainers()) {
            File file = dir.resolve(trainer.getClass().getSimpleName() + ".model").toFile();
            if (!file.exists()) {
                if (required.stream().anyMatch(t -> t.getClass() == trainer.getClass()))
                    missing.add(file.getName());
                continue;
            }
            registry.registerModel(trainer.getClass(), readModel(file.toPath()));
        }
        if (!missing.isEmpty())
            throw new MissingModels("Missing from " + dir.toAbsolutePath() + ": " + String.join(", ", missing));
        registry.setFeatureVersion(featureVersion(dir));
        registry.setAnyPhaseFirst(Files.exists(dir.resolve("any-phase-first")));
        registry.setAnyPhaseRich(Files.exists(dir.resolve("any-phase-rich")));
        registry.setResponseForecast(Files.exists(dir.resolve("response-forecast")));
        if (Files.exists(dir.resolve("card-text")))
            registry.setCardText(CardTextFeatures.vocabulary(dir.resolve("card-text")));
        if (Files.exists(dir.resolve("card-text-selections")))
            registry.setCardTextSelections(Set.of(Files.readString(dir.resolve("card-text-selections")).trim()
                    .split("[,\\s]+")));
        registry.setSkirmishProbe(Files.exists(dir.resolve("skirmish-probe")));
        registry.setReconcilePlain(Files.exists(dir.resolve("reconcile-plain")));
        registry.setPassByMass(Files.exists(dir.resolve("pass-by-mass")));
        registry.setNoSkirmishRule(Files.exists(dir.resolve("no-skirmish-rule")));
        registry.setNumberBid(Files.exists(dir.resolve("number-bid")));
        registry.setCharacterOutsideSkirmish(Files.exists(dir.resolve("character-outside-skirmish")));
        if (Files.exists(dir.resolve("move-threshold")))
            registry.setMoveThreshold(Double.parseDouble(Files.readString(dir.resolve("move-threshold")).trim()));
        // the probe's value numbers: the model directory carries the value model it was fit with
        if (Files.exists(dir.resolve("probe-value.model")))
            EffectProbe.VALUE = readModel(dir.resolve("probe-value.model"));
        if (Files.exists(dir.resolve("assign-probe")))
            registry.setAssignProbe(Set.of(Files.readString(dir.resolve("assign-probe")).trim().toLowerCase()
                    .split("[,\\s]+")));
        if (Files.exists(dir.resolve("probe-contexts"))) {
            String list = Files.readString(dir.resolve("probe-contexts"));
            Matcher length = Pattern.compile("length=(\\d+)").matcher(list);
            registry.setProbeLength(length.find() ? Integer.parseInt(length.group(1))
                    : EffectProbe.LENGTH_WITHOUT_HAND);
            registry.setProbeContexts(AnyPhaseCardActionTrainer.names(list.replaceAll("length=\\d+", "")));
        }
        return registry;
    }

    /** The feature set a model directory was trained on: what its features.txt says, or 1 without one. */
    public static int featureVersion(Path dir) throws IOException {
        Path marker = dir.resolve("features.txt");
        return Files.exists(marker) ? Integer.parseInt(Files.readString(marker).trim()) : 1;
    }

    /** A serialised model file. */
    @SuppressWarnings("unchecked")
    public static SoftClassifier<double[]> readModel(Path file) throws IOException {
        try (ObjectInputStream in = new ObjectInputStream(Files.newInputStream(file))) {
            return (SoftClassifier<double[]>) in.readObject();
        } catch (ClassNotFoundException e) {
            throw new IOException(e);
        }
    }

    /**
     * What every model directory's bot reads besides its models: the card library behind its features, the card
     * definitions behind feature set 3's ability numbers, and a copier for EffectProbe. Each is process-wide and
     * fails silently when missing -- set 3 vectors carry zeros where the abilities go, and a probe without a copier
     * gives zeros for every option, so a skirmish-probe model plays as if every card did nothing. Call once the
     * library is loaded; later calls keep the copier the first one made.
     */
    public static synchronized void prepare(LotroCardBlueprintLibrary library, LotroFormatLibrary formats,
                                            File cardsDir) {
        CardFeatures.init(library);
        AbilityFeatures.init(cardsDir, library);
        CardTextFeatures.init(cardsDir, library);
        if (EffectProbe.FORK == null) {
            GameCopier copier = new GameCopier(library, formats);
            // An admin's card reload swaps in new blueprints, which the copier has not seen as shared: every copy of a
            // game holding one then fails (on the definitions' JSONObjects) and the probe gives zeros until a
            // restart. Shared on top of the old ones, which games started before the reload still hold.
            library.subscribeToRefreshes(() -> {
                copier.reshare(library);
                copier.reshare(formats);
            });
            EffectProbe.FORK = copier::copy;
        }
    }
}
