package com.gempukku.lotro.bots.simulation;

import com.gempukku.lotro.bots.rl.LearningStep;
import com.gempukku.lotro.bots.rl.fotrstarters.CardFeatures;
import com.gempukku.lotro.bots.rl.fotrstarters.EffectProbe;
import com.gempukku.lotro.bots.rl.fotrstarters.GeneralStateFeatures;
import com.gempukku.lotro.bots.rl.fotrstarters.models.cardselection.ExertChoiceTrainer;
import com.gempukku.lotro.bots.rl.fotrstarters.models.cardselection.ReconcileChoiceTrainer;
import com.gempukku.lotro.bots.rl.semanticaction.*;
import com.gempukku.lotro.common.Phase;
import com.gempukku.lotro.game.state.GameState;
import com.gempukku.lotro.logic.decisions.AwaitingDecision;
import com.gempukku.lotro.logic.timing.DefaultLotroGame;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

/**
 * The human's answers to reproduced decisions as teacher labels, in the form BotTeach writes and
 * {@code BotTeach --train-only --steps <file>} refits from: one {@link LearningStep} per option with the feature-set-2
 * state, reward 10 for the option the human took and 0 for the others. Card-action choices and multiple choices list
 * every option (the alternatives are the negatives, as in BotTeach); the other types get the human's answer only.
 * ACTION_CHOICE is skipped: it has no JSON form (BOT-NOTES, "Other things found").
 */
final class TrainingSteps implements HumanReplay.StepSink {
    private final Writer out;
    /**
     * --probe-bot: answers what a card-action option asks on its copy, for EffectProbe; with it, every option of a
     * card-action prompt is written with what it did. Null: no probe.
     */
    static volatile java.util.function.Function<String, com.gempukku.lotro.bots.BotPlayer> PROBE_BOTS;
    /** --steps-only: write only the decisions whose prompt matches; null for all. */
    static volatile java.util.regex.Pattern ONLY;
    final AtomicLong decisions = new AtomicLong(), steps = new AtomicLong();

    private TrainingSteps(Writer out) {
        this.out = out;
    }

    static TrainingSteps open(Path file) throws IOException {
        return new TrainingSteps(Files.newBufferedWriter(file, StandardCharsets.UTF_8));
    }

    @Override
    public double[] state(DefaultLotroGame game, AwaitingDecision decision, String player) {
        int previous = CardFeatures.version();
        try {
            CardFeatures.setVersion(GeneralStateFeatures.VERSION);
            return new GeneralStateFeatures().extractFeatures(game.getGameState(), decision, player);
        } catch (RuntimeException e) {
            return null; // e.g. hands read before the deal; the decision is simply not written
        } finally {
            CardFeatures.setVersion(previous);
        }
    }

    /** The state as {@link #state} writes it, for a decision no steps file is written for. */
    static double[] stateOf(DefaultLotroGame game, AwaitingDecision decision, String player) {
        int previous = CardFeatures.version();
        try {
            CardFeatures.setVersion(GeneralStateFeatures.VERSION);
            return new GeneralStateFeatures().extractFeatures(game.getGameState(), decision, player);
        } catch (RuntimeException e) {
            return null;
        } finally {
            CardFeatures.setVersion(previous);
        }
    }

    /** {@code before} is the game as the decision was asked: the semantic actions read the board as it was. */
    @Override
    public void write(double[] state, AwaitingDecision decision, String answer, String player, DefaultLotroGame before) {
        if (ONLY != null && (decision.getText() == null || !ONLY.matcher(decision.getText()).find()))
            return;
        int previous = CardFeatures.version();
        StringBuilder lines = new StringBuilder();
        try {
            CardFeatures.setVersion(GeneralStateFeatures.VERSION);
            GameState gs = before.getGameState();
            boolean fp = player.equals(gs.getCurrentPlayerId());
            List<String[]> labelled = new ArrayList<>(); // {answer, reward}
            switch (decision.getDecisionType()) {
                case CARD_ACTION_CHOICE -> {
                    String[] ids = decision.getDecisionParameters().getOrDefault("actionId", new String[0]);
                    if (ids.length == 0)
                        return; // nothing to choose: pass was the only answer
                    for (String id : ids)
                        labelled.add(new String[]{id, id.equals(answer) ? "10" : "0"});
                    labelled.add(new String[]{"", answer.isEmpty() ? "10" : "0"});
                }
                case MULTIPLE_CHOICE -> {
                    String[] results = decision.getDecisionParameters().getOrDefault("results", new String[0]);
                    for (int i = 0; i < results.length; i++)
                        labelled.add(new String[]{String.valueOf(i), String.valueOf(i).equals(answer) ? "10" : "0"});
                }
                case ACTION_CHOICE -> {
                    return;
                }
                case ASSIGN_MINIONS -> {
                    // The human's assignment and up to NEGATIVES others the rules allowed. Written alone, as it was
                    // first, the refit model said yes to everything: assigning every minion to the Ring-bearer.
                    var trainer = fp ? new com.gempukku.lotro.bots.rl.fotrstarters.models.assignment.FpAssignmentTrainer()
                            : new com.gempukku.lotro.bots.rl.fotrstarters.models.assignment.ShadowAssignmentTrainer();
                    List<String> others = new ArrayList<>(trainer.allAnswers(gs, decision,
                            new com.gempukku.lotro.bots.AssignmentLegality(before)));
                    String chosen = canonical(answer);
                    if (!others.removeIf(o -> canonical(o).equals(chosen)) || others.isEmpty())
                        return; // not an answer the enumeration makes, or the only one: nothing to compare it with
                    labelled.add(new String[]{answer, "10"});
                    java.util.Collections.shuffle(others, new java.util.Random(answer.hashCode() ^ gs.getTurnNumber()));
                    for (String o : others.subList(0, Math.min(NEGATIVES, others.size())))
                        labelled.add(new String[]{o, "0"});
                }
                default -> labelled.add(new String[]{answer, "10"});
            }
            for (String[] l : labelled) {
                SemanticAction action = switch (decision.getDecisionType()) {
                    case INTEGER -> new IntegerChoiceAction(Integer.parseInt(l[0]));
                    case MULTIPLE_CHOICE -> new MultipleChoiceAction(l[0], decision);
                    case ARBITRARY_CARDS -> new ChooseFromArbitraryCardsAction(l[0], decision);
                    case CARD_ACTION_CHOICE -> new CardActionChoiceAction(l[0], decision, gs).withProbe(probe(gs, decision, l[0], player));
                    case CARD_SELECTION -> gs.getCurrentPhase() == Phase.SKIRMISH
                            ? withOptions(new CardSelectionAssignedAction(l[0], decision, gs), gs, decision, l[0], player)
                            : withOptions(new CardSelectionAction(l[0], decision, gs), gs, decision, l[0], player);
                    case ASSIGN_MINIONS -> new AssignMinionsAction(l[0], decision, gs, fp).withProbe(
                            com.gempukku.lotro.bots.rl.fotrstarters.models.assignment.AssignmentProbe.features(gs,
                                    decision, com.gempukku.lotro.bots.rl.fotrstarters.models.assignment.AssignmentProbe.parse(l[0])));
                    case ACTION_CHOICE -> throw new IllegalStateException();
                };
                LearningStep step = new LearningStep(state, action, fp, decision);
                step.reward = Double.parseDouble(l[1]);
                lines.append(step.toJson()).append('\n');
            }
        } catch (RuntimeException e) {
            return; // a semantic action that cannot describe this decision: skip it, as self-play recording does
        } finally {
            CardFeatures.setVersion(previous);
        }
        synchronized (this) {
            try {
                out.write(lines.toString());
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        }
        decisions.incrementAndGet();
        steps.addAndGet(lines.toString().lines().count());
    }

    /** What {@code answer} did on a copy, at a card-action prompt with --probe-bot; null otherwise. */
    static double[] probe(GameState gs, AwaitingDecision decision, String answer, String player) {
        var bots = PROBE_BOTS;
        if (bots == null || decision.getDecisionType() != com.gempukku.lotro.logic.decisions.AwaitingDecisionType.CARD_ACTION_CHOICE)
            return null;
        if (answer.isEmpty())
            return EffectProbe.pass(gs, player);
        double[] f = EffectProbe.probe(gs, player, answer, bots);
        return f != null ? f : EffectProbe.none();
    }

    /**
     * The options with what only the live game knows: at reconcile keep everything and each card
     * (ReconcileChoiceTrainer), at a one-of exert each character (ExertChoiceTrainer).
     */
    private static CardSelectionAction withOptions(CardSelectionAction action, GameState gs, AwaitingDecision decision,
                                                   String answer, String player) {
        if (ReconcileChoiceTrainer.applies(decision)) {
            var o = ReconcileChoiceTrainer.options(gs, decision, player);
            return action.withOptions(o.blueprints(), o.extras(), ReconcileChoiceTrainer.chosen(decision, answer));
        }
        if (ExertChoiceTrainer.applies(decision)) {
            var o = ExertChoiceTrainer.options(gs, decision, player);
            return action.withOptions(o.blueprints(), o.extras(), ExertChoiceTrainer.chosen(decision, answer));
        }
        if (com.gempukku.lotro.bots.rl.fotrstarters.models.cardselection.SkirmishOrderChoiceTrainer.applies(decision)) {
            var o = com.gempukku.lotro.bots.rl.fotrstarters.models.cardselection.SkirmishOrderChoiceTrainer.options(gs, decision, player);
            return action.withOptions(o.blueprints(), o.extras(), ExertChoiceTrainer.chosen(decision, answer));
        }
        if (com.gempukku.lotro.bots.rl.fotrstarters.models.cardselection.CharacterChoiceTrainer.applies(decision)) {
            var o = com.gempukku.lotro.bots.rl.fotrstarters.models.cardselection.CharacterChoiceTrainer.options(gs, decision, player);
            return action.withOptions(o.blueprints(), o.extras(), ExertChoiceTrainer.chosen(decision, answer));
        }
        return action;
    }

    /** Rejected assignments written per human one. */
    static final int NEGATIVES = 12;

    /** An assignment answer ("fp minion minion,fp minion") with its order dropped, for comparing two. */
    static String canonical(String answer) {
        java.util.TreeMap<String, java.util.TreeSet<String>> m = new java.util.TreeMap<>();
        for (String part : answer.split(",")) {
            String[] ids = part.trim().split("\\s+");
            if (ids.length < 2)
                continue;
            m.computeIfAbsent(ids[0], k -> new java.util.TreeSet<>()).addAll(java.util.Arrays.asList(ids).subList(1, ids.length));
        }
        return m.toString();
    }

    void close() throws IOException {
        out.close();
    }
}
