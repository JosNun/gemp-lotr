package com.gempukku.lotro.bots.rl.fotrstarters;

import com.gempukku.lotro.bots.AssignmentLegality;
import com.gempukku.lotro.bots.random.RandomDecisionBot;
import com.gempukku.lotro.bots.rl.*;
import com.gempukku.lotro.bots.rl.fotrstarters.models.ModelRegistry;
import com.gempukku.lotro.bots.rl.fotrstarters.models.arbitrarycards.CardFromDiscardTrainer;
import com.gempukku.lotro.bots.rl.fotrstarters.models.arbitrarycards.StartingFellowshipTrainer;
import com.gempukku.lotro.bots.rl.fotrstarters.models.assignment.AbstractAssignmentTrainer;
import com.gempukku.lotro.bots.rl.fotrstarters.models.assignment.FpAssignmentTrainer;
import com.gempukku.lotro.bots.rl.fotrstarters.models.assignment.ShadowAssignmentTrainer;
import com.gempukku.lotro.bots.rl.fotrstarters.models.cardaction.*;
import com.gempukku.lotro.bots.rl.fotrstarters.models.cardselection.*;
import com.gempukku.lotro.bots.rl.fotrstarters.models.integerchoice.BurdenTrainer;
import com.gempukku.lotro.bots.rl.fotrstarters.models.multiplechoice.AnotherMoveTrainer;
import com.gempukku.lotro.bots.rl.fotrstarters.models.multiplechoice.GoFirstTrainer;
import com.gempukku.lotro.bots.rl.fotrstarters.models.multiplechoice.MulliganTrainer;
import com.gempukku.lotro.bots.rl.semanticaction.*;
import com.gempukku.lotro.common.Phase;
import com.gempukku.lotro.game.PhysicalCard;
import com.gempukku.lotro.game.state.GameState;
import com.gempukku.lotro.game.state.Skirmish;
import com.gempukku.lotro.logic.decisions.AwaitingDecision;
import com.gempukku.lotro.logic.decisions.CardActionSelectionDecision;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

public class FotrStarterBot extends RandomDecisionBot implements LearningBotPlayer {
    private final List<DecisionAnswerer> cardSelectionTrainers = List.of(
            new ReconcileTrainer(),
            new SanctuaryTrainer(),
            new ArcheryWoundTrainer(),
            new AttachItemTrainer(),
            new SkirmishOrderTrainer(),
            new HealTrainer(),
            new DiscardFromHandTrainer(),
            new ExertTrainer(),
            new DiscardFromPlayTrainer(),
            new PlayFromHandTrainer()
    );

    private final List<DecisionAnswerer> fallBackTrainers = List.of(
            new FallBackCardSelectionTrainer() // card selection
    );

    private final List<DecisionAnswerer> integerTrainers = List.of(
            new BurdenTrainer()
    );

    private final List<DecisionAnswerer> multipleChoiceTrainers = List.of(
            new GoFirstTrainer(),
            new MulliganTrainer(),
            new AnotherMoveTrainer()
    );

    private final List<DecisionAnswerer> arbitraryCardsTrainers = List.of(
            new CardFromDiscardTrainer(),
            new StartingFellowshipTrainer(),
            new com.gempukku.lotro.bots.rl.fotrstarters.models.arbitrarycards.CardFromDeckTrainer()
    );

    private final List<DecisionAnswerer> cardActionTrainers = List.of(
            new FellowshipCardActionAnswerer(),
            new ShadowCardActionAnswerer(),
            new ManeuverCardActionAnswerer(),
            new SkirmishFpCardActionAnswerer(),
            new SkirmishShadowCardActionAnswerer(),
            new RegroupCardActionAnswerer(),
            new OptionalResponsesCardActionTrainer()
    );

    private final List<AbstractAssignmentTrainer> assignmentTrainers = List.of(
            new ShadowAssignmentTrainer(),
            new FpAssignmentTrainer()
    );

    private final RLGameStateFeatures features;
    /** Features to store in learning steps when they differ from the ones played with; see recordWith. */
    private RLGameStateFeatures recordFeatures;
    private final ModelRegistry modelRegistry;
    private final ReplayBuffer replayBuffer;

    private CardActionSelectionDecision lastDecision = null;
    private int decisionRepeat = 0;
    private String lastAction = null;
    private int answerRepeat = 0;

    private final List<LearningStep> episodeSteps = new ArrayList<>();
    /** Chance of answering at random instead of by the models, so self-play sees moves the models would not make. */
    private final double epsilon;
    /**
     * Off by default, which is the bot as trained on the FotR starters. On, decisions that bot answered at random get
     * an answer: card actions in any phase from {@link AnyPhaseCardActionTrainer} (when its model is loaded), sites
     * by shadow number, "how many X to spot" with the most.
     */
    private final boolean extended;
    private final AnyPhaseCardActionTrainer anyPhaseTrainer = new AnyPhaseCardActionTrainer();
    /** Reconcile with "keep everything" as an option, where the model directory has one. */
    private final ReconcileChoiceTrainer reconcileChoice = new ReconcileChoiceTrainer();
    /** One-of exerts, compared with each character's vitality and fights, where the model directory has the model. */
    private final ExertChoiceTrainer exertChoice = new ExertChoiceTrainer();

    public FotrStarterBot(RLGameStateFeatures features, String playerId, ModelRegistry modelRegistry, ReplayBuffer replayBuffer) {
        this(features, playerId, modelRegistry, replayBuffer, 0.0, false);
    }

    public FotrStarterBot(RLGameStateFeatures features, String playerId, ModelRegistry modelRegistry, ReplayBuffer replayBuffer,
                          double epsilon, boolean extended) {
        super(playerId);
        // models trained on feature set 2 are only ever fed set 2, whatever the caller passed
        this.features = modelRegistry != null && modelRegistry.getFeatureVersion() >= GeneralStateFeatures.VERSION
                ? new GeneralStateFeatures() : features;
        this.modelRegistry = modelRegistry;
        this.replayBuffer = replayBuffer;
        this.epsilon = epsilon;
        this.extended = extended;
    }

    @Override
    public String chooseAction(GameState gameState, AwaitingDecision decision) {
        return chooseAction(gameState, decision, AssignmentLegality.UNCHECKED);
    }

    /** The same models playing as {@code player}, recording nothing: for answering on a copy (EffectProbe). */
    public FotrStarterBot as(String player) {
        return new FotrStarterBot(features, player, modelRegistry, null, 0.0, extended);
    }

    /**
     * Store learning steps with these features instead of the ones the bot plays with. Training a new feature set
     * starts from models of the old one: they play the first generation, and what it records has to be in the set
     * being trained.
     */
    public FotrStarterBot recordWith(RLGameStateFeatures recordFeatures) {
        this.recordFeatures = recordFeatures;
        return this;
    }

    @Override
    public String chooseAction(GameState gameState, AwaitingDecision decision, AssignmentLegality legality) {
        int previousVersion = CardFeatures.version();
        try {
            CardFeatures.setVersion(modelRegistry == null ? 1 : modelRegistry.getFeatureVersion());
            SemanticAction action = chooseSemanticAction(gameState, decision, legality);

            if (replayBuffer != null) {
                // Store temporarily — reward comes later
                double[] stateVector = (recordFeatures != null ? recordFeatures : features)
                        .extractFeatures(gameState, decision, getName());
                episodeSteps.add(new LearningStep(stateVector, action, getName().equals(gameState.getCurrentPlayerId()), decision));
            }

            return action.toDecisionString(decision, gameState);
        } finally {
            CardFeatures.setVersion(previousVersion);
        }
    }

    /**
     * The assignments the models rank highest for this decision, best first and legal under the rules, for a search
     * to try rather than taking the top one. Empty if no assignment model applies.
     */
    public List<String> assignmentCandidates(GameState gameState, AwaitingDecision decision, int k) {
        int previousVersion = CardFeatures.version();
        try {
            CardFeatures.setVersion(modelRegistry == null ? 1 : modelRegistry.getFeatureVersion());
            for (AbstractAssignmentTrainer trainer : assignmentTrainers)
                if (trainer.appliesTo(gameState, decision, getName()))
                    return trainer.candidates(gameState, decision, getName(), features, modelRegistry,
                            new AssignmentLegality(gameState.getGame()), k);
            return List.of();
        } finally {
            CardFeatures.setVersion(previousVersion);
        }
    }

    private SemanticAction chooseSemanticAction(GameState gameState, AwaitingDecision decision, AssignmentLegality legality) {
        String action = epsilon > 0 && com.gempukku.lotro.bots.BotRandom.current().nextDouble() < epsilon
                ? super.chooseAction(gameState, decision, legality)
                : switch (decision.getDecisionType()) {
            case INTEGER -> chooseIntegerAction(gameState, decision);
            case MULTIPLE_CHOICE -> chooseMultipleChoiceAction(gameState, decision);
            case ARBITRARY_CARDS -> chooseArbitraryCardsAction(gameState, decision);
            case CARD_ACTION_CHOICE -> chooseCardActionChoice(gameState, decision);
            case ACTION_CHOICE -> chooseActionChoice(gameState, decision);
            case CARD_SELECTION -> chooseCardSelectionAction(gameState, decision);
            case ASSIGN_MINIONS -> chooseAssignmentAction(gameState, decision, legality);
        };

        return switch (decision.getDecisionType()) {
            case INTEGER -> new IntegerChoiceAction(Integer.parseInt(action));
            case MULTIPLE_CHOICE -> new MultipleChoiceAction(action, decision);
            case ARBITRARY_CARDS -> new ChooseFromArbitraryCardsAction(action, decision);
            case CARD_ACTION_CHOICE -> new CardActionChoiceAction(action, decision, gameState);
            case ACTION_CHOICE -> new ActionChoiceAction(action, decision);
            case CARD_SELECTION -> {
                if (gameState.getCurrentPhase().equals(Phase.SKIRMISH)) {
                    yield new CardSelectionAssignedAction(action, decision, gameState);
                }
                yield new CardSelectionAction(action, decision, gameState);
            }
            case ASSIGN_MINIONS -> new AssignMinionsAction(action, decision, gameState, gameState.getCurrentPlayerId().equals(getName()));
        };
    }

    private String chooseAssignmentAction(GameState gameState, AwaitingDecision decision, AssignmentLegality legality) {
        for (AbstractAssignmentTrainer trainer : assignmentTrainers) {
            if (trainer.appliesTo(gameState, decision, getName())) {
                return trainer.getAnswer(gameState, decision, getName(), features, modelRegistry, legality);
            }
        }

        System.out.println("Unknown assign minions decision: "
                + gameState.getCurrentPhase() + " - "
                + decision.getText()
                + "; freeCharacters=" + Arrays.toString(decision.getDecisionParameters().get("freeCharacters"))
                + "; minions=" + Arrays.toString(decision.getDecisionParameters().get("minions")));
        return super.chooseAction(gameState, decision, legality);
    }

    private String chooseActionChoice(GameState gameState, AwaitingDecision decision) {
        // Order does not matter with FotR starters
//        System.out.println("Unknown action decision: "
//                + gameState.getCurrentPhase() + " - "
//                + decision.getText()
//                + "; actionId=" + Arrays.toString(decision.getDecisionParameters().get("actionId"))
//                + "; blueprintId=" + Arrays.toString(decision.getDecisionParameters().get("blueprintId"))
//                + "; actionText=" + Arrays.toString(decision.getDecisionParameters().get("actionText")));
        return super.chooseAction(gameState, decision);
    }

    private String chooseCardActionChoice(GameState gameState, AwaitingDecision decision) {
        String[] actionIds = decision.getDecisionParameters().get("actionId");
        if (actionIds == null || actionIds.length == 0) {
            // No actions available: must pass
            return "";
        }

        String chosenAnswer = null;
        boolean anyFirst = extended && modelRegistry.isAnyPhaseFirst() && modelRegistry.hasModel(AnyPhaseCardActionTrainer.class);
        if (anyFirst)
            chosenAnswer = anyPhaseTrainer.getAnswer(gameState, decision, getName(), features, modelRegistry);
        for (DecisionAnswerer trainer : cardActionTrainers) {
            if (!anyFirst && trainer.appliesTo(gameState, decision, getName())) {
                chosenAnswer = trainer.getAnswer(gameState, decision, getName(), features, modelRegistry);
            }
        }
        if (chosenAnswer == null && extended && modelRegistry.hasModel(AnyPhaseCardActionTrainer.class))
            chosenAnswer = anyPhaseTrainer.getAnswer(gameState, decision, getName(), features, modelRegistry);
        // Loop prevention
        if (chosenAnswer != null) {
            if (decision.equals(lastDecision) && chosenAnswer.equals(lastAction)) {
                answerRepeat++;
                decisionRepeat++;
                if (answerRepeat >= 3) {
                    answerRepeat = 0;
                    lastAction = null;
                    return "";
                }
            } else if (decision.equals(lastDecision)) {
                decisionRepeat++;
                if (decisionRepeat >= 6) {
                    answerRepeat = 0;
                    lastAction = null;
                    return "";
                }
            } else {
                lastDecision = (CardActionSelectionDecision) decision;
                decisionRepeat = 0;
                answerRepeat = 0;
                lastAction = chosenAnswer;
            }
            return chosenAnswer;
        }

        System.out.println("Unknown card action decision: "
                + gameState.getCurrentPhase() + " - "
                + decision.getText()
                + "; actionId=" + Arrays.toString(decision.getDecisionParameters().get("actionId"))
                + "; cardId=" + Arrays.toString(decision.getDecisionParameters().get("cardId"))
                + "; blueprintId=" + Arrays.toString(decision.getDecisionParameters().get("blueprintId"))
                + "; actionText=" + Arrays.toString(decision.getDecisionParameters().get("actionText")));
        return super.chooseAction(gameState, decision);
    }

    private String chooseArbitraryCardsAction(GameState gameState, AwaitingDecision decision) {
        Map<String, String[]> params = decision.getDecisionParameters();
        String[] cardIds = params.get("cardId");
        String[] selectable = params.get("selectable");
        int min = params.containsKey("min") ? Integer.parseInt(params.get("min")[0]) : 0;
        int max = params.containsKey("max") ? Integer.parseInt(params.get("max")[0]) : cardIds.length;

        // Check if something is selectable
        if (cardIds == null || selectable == null || cardIds.length != selectable.length)
            return ""; // Invalid input
        // Collect all selectable indices
        List<Integer> selectableIndices = new ArrayList<>();
        for (int i = 0; i < selectable.length; i++) {
            if (Boolean.parseBoolean(selectable[i])) {
                selectableIndices.add(i);
            }
        }
        // If nothing is selectable or max == 0, just return empty
        if (selectableIndices.isEmpty() || max == 0)
            return "";

        for (DecisionAnswerer trainer : arbitraryCardsTrainers) {
            if (trainer instanceof com.gempukku.lotro.bots.rl.fotrstarters.models.Trainer t
                    && !modelRegistry.hasModel(t.getClass()))
                continue; // an optional model (CardFromDeckTrainer) this model set does not have
            if (trainer.appliesTo(gameState, decision, getName())) {
                return trainer.getAnswer(gameState, decision, getName(), features, modelRegistry);
            }
        }

        if (extended && decision.getText().startsWith("Choose site to play")) {
            String site = chooseSite(gameState, cardIds, selectable, params.get("blueprintId"));
            if (site != null)
                return site;
        }

        // Fallback to random decision
        System.out.println("Unknown arbitrary card selection action: "
                + gameState.getCurrentPhase() + " - "
                + decision.getText()
                + " (" + decision.getDecisionParameters().get("min")[0]
                + ";" + decision.getDecisionParameters().get("max")[0] + ") "
                + "cards=" + Arrays.toString(decision.getDecisionParameters().get("blueprintId"))
                + ", selectable=" + Arrays.toString(decision.getDecisionParameters().get("selectable")));
        return super.chooseAction(gameState, decision);
    }

    private String chooseCardSelectionAction(GameState gameState, AwaitingDecision decision) {
        int min = Integer.parseInt(decision.getDecisionParameters().get("min")[0]);
        int max = Integer.parseInt(decision.getDecisionParameters().get("max")[0]);
        List<String> cardIds = Arrays.stream(decision.getDecisionParameters().get("cardId")).toList();

        if (min == max && min == cardIds.size()) {
            // Need to choose all
            return String.join(",", cardIds);
        }

        if (extended && modelRegistry.hasModel(ExertChoiceTrainer.class) && ExertChoiceTrainer.applies(decision)) {
            String answer = exertChoice.getAnswer(gameState, decision, getName(), features, modelRegistry);
            if (answer != null)
                return answer;
        }

        // Choose the one skirmishing -- unless the model directory says to ask the models ("no-skirmish-rule"): an
        // exert cost paid by the companion fighting is often the wrong one
        if (!(extended && modelRegistry.isNoSkirmishRule()) && gameState.getSkirmish() != null &&
                gameState.getSkirmish().getFellowshipCharacter() != null &&
                gameState.getSkirmish().getShadowCharacters() != null &&
                !gameState.getSkirmish().getShadowCharacters().isEmpty()
                && min == 1 && max == 1) {
            Skirmish skirmish = gameState.getSkirmish();
            for (String cardId : cardIds) {
                if (skirmish.getFellowshipCharacter().getCardId() == Integer.parseInt(cardId)) {
                    return cardId;
                }
                for (PhysicalCard shadowCharacter : skirmish.getShadowCharacters()) {
                    if (shadowCharacter.getCardId() == Integer.parseInt(cardId)) {
                        return cardId;
                    }
                }
            }
            // Skirmishing character cannot be chosen, choose whatever
            return super.chooseAction(gameState, decision);
        }

        if (extended && modelRegistry.hasModel(ReconcileChoiceTrainer.class) && ReconcileChoiceTrainer.applies(decision)) {
            String answer = reconcileChoice.getAnswer(gameState, decision, getName(), features, modelRegistry);
            if (answer != null)
                return answer;
        }
        for (DecisionAnswerer trainer : cardSelectionTrainers) {
            if (trainer.appliesTo(gameState, decision, getName())) {
                return trainer.getAnswer(gameState, decision, getName(), features, modelRegistry);
            }
        }

        // Last fallback to trainer
        for (DecisionAnswerer trainer : fallBackTrainers) {
            if (trainer.appliesTo(gameState, decision, getName())) {
                return trainer.getAnswer(gameState, decision, getName(), features, modelRegistry);
            }
        }

        // Fallback to random decision
        System.out.println("Unknown card selection action: " + gameState.getCurrentPhase() + " - " + decision.getText() + " (" + decision.getDecisionParameters().get("min")[0] + ";" + decision.getDecisionParameters().get("max")[0] + ") " + Arrays.toString(decision.getDecisionParameters().get("cardId")));
        return super.chooseAction(gameState, decision);
    }

    private String chooseIntegerAction(GameState gameState, AwaitingDecision decision) {
        for (DecisionAnswerer trainer : integerTrainers) {
            if (trainer.appliesTo(gameState, decision, getName())) {
                return trainer.getAnswer(gameState, decision, getName(), features, modelRegistry);
            }
        }
        if (decision.getText().contains("how many to spot")
                || (extended && SPOT_HOW_MANY.matcher(decision.getText()).find())) {
            return decision.getDecisionParameters().get("max")[0];
        }

        System.out.println("Unknown integer action: " + decision.getText());
        return super.chooseAction(gameState, decision);
    }

    private String chooseMultipleChoiceAction(GameState gameState, AwaitingDecision decision) {
        String[] options = decision.getDecisionParameters().get("results");
        for (DecisionAnswerer trainer : multipleChoiceTrainers) {
            if (trainer.appliesTo(gameState, decision, getName())) {
                return trainer.getAnswer(gameState, decision, getName(), features, modelRegistry);
            }
        }
        if (extended && modelRegistry.hasModel(com.gempukku.lotro.bots.rl.fotrstarters.models.multiplechoice.GenericChoiceTrainer.class)) {
            // a card's choice of costs or effects, which no other model answers: otherwise answered at random below
            var generic = new com.gempukku.lotro.bots.rl.fotrstarters.models.multiplechoice.GenericChoiceTrainer();
            if (generic.appliesTo(gameState, decision, getName())) {
                String answer = generic.getAnswer(gameState, decision, getName(), features, modelRegistry);
                if (answer != null)
                    return answer;
            }
        }
        if (List.of(options).contains("Heal a companion") && List.of(options).contains("Remove a Shadow condition from a companion")) {
            return new MultipleChoiceAction("Heal a companion").toDecisionString(decision, gameState);
        }
        if (List.of(options).contains("Exert Frodo") && List.of(options).contains("Exert 2 other companions")) {
            return new MultipleChoiceAction("Exert Frodo").toDecisionString(decision, gameState);
        }

        System.out.println("Unknown multiple choice action: " + decision.getText() + " - " + Arrays.toString(options));
        return super.chooseAction(gameState, decision);
    }

    /** "Choose how many ROHAN man to spot", "...Ents to spot": spotting more makes the effect bigger, never smaller. */
    private static final java.util.regex.Pattern SPOT_HOW_MANY =
            java.util.regex.Pattern.compile("(?i)how many .* to spot");

    /**
     * From the Shadows sets on, sites carry no number and whoever plays the next one picks it from their adventure
     * deck. A site's shadow number is the twilight added when the fellowship moves there: the Shadow player wants
     * the most, the Free Peoples player the least. Decided by whose turn it is, so it holds whichever player the
     * engine asks.
     */
    private String chooseSite(GameState gameState, String[] cardIds, String[] selectable, String[] blueprintIds) {
        // the choices carry temporary ids ("temp0"), not card ids; the blueprints say which site each is
        if (blueprintIds == null || blueprintIds.length != cardIds.length)
            return null;
        boolean shadow = !getName().equals(gameState.getCurrentPlayerId());
        String best = null;
        int bestTwilight = 0;
        for (int i = 0; i < cardIds.length; i++) {
            if (!Boolean.parseBoolean(selectable[i]))
                continue;
            int twilight;
            try {
                twilight = CardFeatures.blueprint(blueprintIds[i]).getTwilightCost();
            } catch (com.gempukku.lotro.game.CardNotFoundException e) {
                continue;
            }
            if (best == null || (shadow ? twilight > bestTwilight : twilight < bestTwilight)) {
                best = cardIds[i];
                bestTwilight = twilight;
            }
        }
        return best;
    }

    @Override
    public void observe(GameState gameState, AwaitingDecision decision, String playerId, String chosenAction, double reward, boolean terminal) {
        // At this stage, we assume `chosenAction` was already used to resolve game state,
        // and we just want to record the final outcome of this decision if needed.
        // Right now, we store full episode and use reward only in `endEpisode`.
    }

    @Override
    public void endEpisode(double reward) {
        episodeSteps.forEach(learningStep -> learningStep.reward = reward);
        if (replayBuffer != null) {
            replayBuffer.addEpisode(new ArrayList<>(episodeSteps));
        }
        episodeSteps.clear();
    }
}
