package com.gempukku.lotro.bots.simulation;

import com.gempukku.lotro.bots.rl.LearningStep;
import com.gempukku.lotro.bots.rl.fotrstarters.CardFeatures;
import com.gempukku.lotro.bots.rl.fotrstarters.GeneralStateFeatures;
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

    /** {@code before} is the game as the decision was asked: the semantic actions read the board as it was. */
    @Override
    public void write(double[] state, AwaitingDecision decision, String answer, String player, DefaultLotroGame before) {
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
                default -> labelled.add(new String[]{answer, "10"});
            }
            for (String[] l : labelled) {
                SemanticAction action = switch (decision.getDecisionType()) {
                    case INTEGER -> new IntegerChoiceAction(Integer.parseInt(l[0]));
                    case MULTIPLE_CHOICE -> new MultipleChoiceAction(l[0], decision);
                    case ARBITRARY_CARDS -> new ChooseFromArbitraryCardsAction(l[0], decision);
                    case CARD_ACTION_CHOICE -> new CardActionChoiceAction(l[0], decision, gs);
                    case CARD_SELECTION -> gs.getCurrentPhase() == Phase.SKIRMISH
                            ? new CardSelectionAssignedAction(l[0], decision, gs)
                            : new CardSelectionAction(l[0], decision, gs);
                    case ASSIGN_MINIONS -> new AssignMinionsAction(l[0], decision, gs, fp);
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

    void close() throws IOException {
        out.close();
    }
}
