package com.gempukku.lotro.bots.rl.fotrstarters.models.assignment;

import com.gempukku.lotro.bots.AssignmentLegality;
import com.gempukku.lotro.bots.rl.LearningStep;
import com.gempukku.lotro.bots.rl.RLGameStateFeatures;
import com.gempukku.lotro.bots.rl.fotrstarters.CardFeatures;
import com.gempukku.lotro.bots.rl.fotrstarters.models.AbstractTrainer;
import com.gempukku.lotro.bots.rl.fotrstarters.models.LabeledPoint;
import com.gempukku.lotro.bots.rl.fotrstarters.models.ModelRegistry;
import com.gempukku.lotro.bots.rl.semanticaction.AssignMinionsAction;
import com.gempukku.lotro.common.CardType;
import com.gempukku.lotro.common.Side;
import com.gempukku.lotro.game.CardNotFoundException;
import com.gempukku.lotro.game.state.GameState;
import com.gempukku.lotro.logic.decisions.AwaitingDecision;
import com.gempukku.lotro.logic.decisions.AwaitingDecisionType;
import smile.classification.SoftClassifier;

import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public abstract class AbstractAssignmentTrainer extends AbstractTrainer {
    protected abstract boolean isForFp();

    protected abstract void generateAssignmentsRecursive(Map<String, String> minionsToAssign,
                                                         List<String> minionKeys, int index,
                                                         Map<String, String> freeChars,
                                                         AssignmentInfo current,
                                                         List<AssignmentInfo> results,
                                                         GameState gameState);


    @Override
    public boolean appliesTo(GameState gameState, AwaitingDecision decision, String playerName) {
        boolean correctPlayer = (isForFp() && gameState.getCurrentPlayerId().equals(playerName)) ||
                (!isForFp() && !gameState.getCurrentPlayerId().equals(playerName));

        return decision.getDecisionType().equals(AwaitingDecisionType.ASSIGN_MINIONS) &&
                correctPlayer;
    }

    @Override
    public boolean isStepRelevant(LearningStep step) {
        return step.decision.getDecisionType().equals(AwaitingDecisionType.ASSIGN_MINIONS) &&
                step.action instanceof AssignMinionsAction ama &&
                ama.isFreePeoplesAssignment() == isForFp();
    }

    @Override
    public String getAnswer(GameState gameState, AwaitingDecision decision, String playerName, RLGameStateFeatures features, ModelRegistry modelRegistry) {
        return getAnswer(gameState, decision, playerName, features, modelRegistry, AssignmentLegality.UNCHECKED);
    }

    /**
     * Scores every candidate assignment with the model and answers with the best one the game will accept. If the
     * rules reject every candidate (only possible for Shadow, whose candidates assign every minion), the best-scored
     * one is trimmed down to its legal pairings instead.
     */
    public String getAnswer(GameState gameState, AwaitingDecision decision, String playerName, RLGameStateFeatures features,
                            ModelRegistry modelRegistry, AssignmentLegality legality) {
        List<String> ranked = ranked(gameState, decision, playerName, features, modelRegistry, legality, 1, true);
        return ranked.isEmpty() ? "" : ranked.get(0);
    }

    /**
     * The {@code k} legal candidate assignments the model scores highest, best first, as answers -- what a search
     * branches on instead of trusting the top one. Empty if there is nothing to assign.
     */
    public List<String> candidates(GameState gameState, AwaitingDecision decision, String playerName,
                                   RLGameStateFeatures features, ModelRegistry modelRegistry,
                                   AssignmentLegality legality, int k) {
        return ranked(gameState, decision, playerName, features, modelRegistry, legality, k, false);
    }

    /**
     * Every candidate assignment the rules accept, as answers, unscored; assigning nothing is among them when it is
     * legal. What a human's answer is labelled against: trained on the chosen one alone, the model learns that every
     * assignment is good.
     */
    public List<String> allAnswers(GameState gameState, AwaitingDecision decision, AssignmentLegality legality) {
        Map<String, String[]> params = decision.getDecisionParameters();
        String[] freeCharIds = params.get("freeCharacters"), minionIds = params.get("minions");
        Side side = isForFp() ? Side.FREE_PEOPLE : Side.SHADOW;
        List<String> answers = new ArrayList<>();
        for (AssignmentInfo a : enumerate(gameState, decision))
            if (legality.isLegal(side, freeCharIds, minionIds, a.physicalAssignment())) {
                String answer = format(a.physicalAssignment());
                if (!answers.contains(answer))
                    answers.add(answer);
            }
        return answers;
    }

    private List<AssignmentInfo> enumerate(GameState gameState, AwaitingDecision decision) {
        Map<String, String[]> params = decision.getDecisionParameters();
        String[] freeCharIds = params.get("freeCharacters");
        String[] minionIds = params.get("minions");

        if (freeCharIds == null || minionIds == null || freeCharIds.length == 0 || minionIds.length == 0)
            return List.of();
        Map<String, String> freeChars = Stream.of(freeCharIds)
                .collect(Collectors.toMap(Function.identity(), s -> gameState.getBlueprintId(Integer.parseInt(s))));
        Map<String, String> minions = Stream.of(minionIds)
                .collect(Collectors.toMap(Function.identity(), s -> gameState.getBlueprintId(Integer.parseInt(s))));
        HashMap<String, List<String>> alreadyAssignedMap = new HashMap<>();
        gameState.getAssignments().forEach(assignment -> {
            String fpBlueprint = assignment.getFellowshipCharacter().getBlueprintId();
            alreadyAssignedMap.put(fpBlueprint, new ArrayList<>());
            assignment.getShadowCharacters().forEach(shadow -> alreadyAssignedMap.get(fpBlueprint).add(shadow.getBlueprintId()));
        });
        gameState.getInPlay().forEach(fpCharacter -> {
            if ((fpCharacter.getBlueprint().getCardType().equals(CardType.COMPANION) ||
                    fpCharacter.getBlueprint().getCardType().equals(CardType.ALLY)) &&
                    fpCharacter.getOwner().equals(gameState.getCurrentPlayerId()))
                alreadyAssignedMap.putIfAbsent(fpCharacter.getBlueprintId(), new ArrayList<>());
        });
        return generateAllAssignments(alreadyAssignedMap, freeChars, minions, gameState);
    }

    private List<String> ranked(GameState gameState, AwaitingDecision decision, String playerName,
                                RLGameStateFeatures features, ModelRegistry modelRegistry,
                                AssignmentLegality legality, int k, boolean trimIllegal) {
        Map<String, String[]> params = decision.getDecisionParameters();
        String[] freeCharIds = params.get("freeCharacters");
        String[] minionIds = params.get("minions");

        if (freeCharIds == null || minionIds == null || freeCharIds.length == 0 || minionIds.length == 0)
            return List.of();


        Map<String, String> freeChars = Stream.of(freeCharIds)
                .collect(Collectors.toMap(
                        Function.identity(),
                        s -> gameState.getBlueprintId(Integer.parseInt(s))
                ));
        Map<String, String> minions = Stream.of(minionIds)
                .collect(Collectors.toMap(
                        Function.identity(),
                        s -> gameState.getBlueprintId(Integer.parseInt(s))
                ));
        HashMap<String, List<String>> alreadyAssignedMap = new HashMap<>();
        HashMap<String, Integer> woundsOnFp = new HashMap<>();
        gameState.getAssignments().forEach(assignment -> {
            String fpBlueprint = assignment.getFellowshipCharacter().getBlueprintId();
            alreadyAssignedMap.put(fpBlueprint, new ArrayList<>());
            assignment.getShadowCharacters().forEach(shadow -> alreadyAssignedMap.get(fpBlueprint).add(shadow.getBlueprintId()));
        });
        gameState.getInPlay().forEach(fpCharacter -> {
            if ((fpCharacter.getBlueprint().getCardType().equals(CardType.COMPANION) ||
                    fpCharacter.getBlueprint().getCardType().equals(CardType.ALLY)) &&
                    fpCharacter.getOwner().equals(gameState.getCurrentPlayerId())) {
                woundsOnFp.put(fpCharacter.getBlueprintId(), gameState.getWounds(fpCharacter));
                if (!alreadyAssignedMap.containsKey(fpCharacter.getBlueprintId())) {
                    alreadyAssignedMap.put(fpCharacter.getBlueprintId(), new ArrayList<>());
                }
            }
        });

        List<AssignmentInfo> allAssignments = generateAllAssignments(alreadyAssignedMap, freeChars, minions, gameState);
        double[] stateVector = features.extractFeatures(gameState, decision, playerName);
        SoftClassifier<double[]> model = modelRegistry.getModel(getClass());

        Side side = isForFp() ? Side.FREE_PEOPLE : Side.SHADOW;
        List<Map.Entry<Double, Map<String, List<String>>>> legal = new ArrayList<>();
        double bestIllegalScore = Double.NEGATIVE_INFINITY;
        Map<String, List<String>> bestIllegalAssignment = null;

        // the network deciding alone (trees' share 0): enumerate and check legality, but leave the scoring to it
        var alone = modelRegistry.getBoardNet2();
        boolean netAlone = alone != null && isForFp() && alone.treeWeight("assign fp") == 0;
        for (AssignmentInfo assignment : allAssignments) {
            if (netAlone) {
                if (legality.isLegal(side, freeCharIds, minionIds, assignment.physicalAssignment()))
                    legal.add(Map.entry(0.5, assignment.physicalAssignment()));
                else if (bestIllegalAssignment == null)
                    bestIllegalAssignment = assignment.physicalAssignment();
                continue;
            }
            try {
                double[] assignmentVector = CardFeatures.getAssignmentFeatures(assignment.blueprintAssignment(), woundsOnFp,
                        assignment.numberOfUnassignedMinions(), assignment.strengthOfUnassignedMinions());
                if (modelRegistry.getAssignProbe().contains(side()))
                    assignmentVector = append(assignmentVector,
                            AssignmentProbe.features(gameState, decision, assignment.physicalAssignment()));
                double[] extended = Arrays.copyOf(stateVector, stateVector.length + assignmentVector.length);
                System.arraycopy(assignmentVector, 0, extended, stateVector.length, assignmentVector.length);

                double[] probs = new double[2];
                model.predict(extended, probs);
                if (!legality.isLegal(side, freeCharIds, minionIds, assignment.physicalAssignment())) {
                    if (probs[1] > bestIllegalScore) {
                        bestIllegalScore = probs[1];
                        bestIllegalAssignment = assignment.physicalAssignment();
                    }
                } else
                    legal.add(Map.entry(probs[1], assignment.physicalAssignment()));
            } catch (CardNotFoundException ignore) {
            }
        }
        var net2 = modelRegistry.getBoardNet2();
        if (net2 != null && isForFp() && legal.size() > 1) {
            // the combined network: a candidate's log-likelihood is the sum over minions of where it sends each (or
            // leaves it), mixed with the trees' scores as log-probabilities over the candidates
            var r = net2.read(gameState, decision, playerName);
            if (r != null && r.assign().length == minionIds.length) {
                List<String> fps = Arrays.asList(freeCharIds), mins = Arrays.asList(minionIds);
                double[] net = new double[legal.size()], trees = new double[legal.size()];
                for (int c = 0; c < legal.size(); c++) {
                    double p = Math.min(1 - 1e-9, Math.max(1e-9, legal.get(c).getKey()));
                    trees[c] = Math.log(p / (1 - p));
                    int[] to = new int[mins.size()];
                    Arrays.fill(to, fps.size()); // unassigned
                    for (var e : legal.get(c).getValue().entrySet())
                        for (String m : e.getValue())
                            if (mins.indexOf(m) >= 0 && fps.indexOf(e.getKey()) >= 0)
                                to[mins.indexOf(m)] = fps.indexOf(e.getKey());
                    for (int m = 0; m < to.length; m++)
                        net[c] += r.assign()[m][to[m]];
                }
                double[] mixed = com.gempukku.lotro.bots.rl.fotrstarters.BoardNet2.mix(net, trees,
                        net2.treeWeight("assign fp"));
                List<Map.Entry<Double, Map<String, List<String>>>> rescored = new ArrayList<>();
                for (int c = 0; c < legal.size(); c++)
                    rescored.add(Map.entry(mixed[c], legal.get(c).getValue()));
                legal = rescored;
            }
        }
        // stable: among equal scores the first generated wins, as before
        legal.sort(Map.Entry.<Double, Map<String, List<String>>>comparingByKey().reversed());
        List<String> answers = new ArrayList<>();
        for (var entry : legal) {
            String answer = format(entry.getValue());
            // the best may be to assign nothing ("" -- getAnswer's answer then, as it always was); a search adds
            // that option itself, so candidates leave it out
            if (!answers.contains(answer) && (trimIllegal || !answer.isEmpty()))
                answers.add(answer);
            if (answers.size() >= k)
                break;
        }
        if (legal.isEmpty() && trimIllegal && bestIllegalAssignment != null)
            answers.add(format(legality.keepLegal(side, freeCharIds, minionIds, bestIllegalAssignment)));
        return answers;
    }

    private static String format(Map<String, List<String>> assignment) {
        if (assignment == null || assignment.isEmpty())
            return "";
        return assignment.entrySet().stream()
                .map(entry -> entry.getKey() + " " + String.join(" ", entry.getValue()))
                .collect(Collectors.joining(","));
    }


    private List<AssignmentInfo> generateAllAssignments(HashMap<String, List<String>> alreadyAssignedMap,
                                                        Map<String, String> freeChars,
                                                        Map<String, String> minions,
                                                        GameState gameState) {
        List<AssignmentInfo> results = new ArrayList<>();

        // Ensure every freeChar is initialized in the assignment map
        freeChars.forEach((physicalId, blueprintId) -> {
            if (!alreadyAssignedMap.containsKey(blueprintId)) {
                throw new IllegalArgumentException("Unknown fp card: " + blueprintId);
            }

        });

        // Deep copy to avoid mutating input map
        AssignmentInfo current = new AssignmentInfo(new HashMap<>(), new HashMap<>(), 0 , 0);
        for (String fp : alreadyAssignedMap.keySet()) {
            current.blueprintAssignment().put(fp, new ArrayList<>(alreadyAssignedMap.get(fp)));
        }

        generateAssignmentsRecursive(minions, new ArrayList<>(minions.keySet()), 0, freeChars, current, results, gameState);
        return results;
    }

    @Override
    protected List<LabeledPoint> extractTrainingData(List<LearningStep> steps) {
        List<LabeledPoint> data = new ArrayList<>();

        for (LearningStep step : steps) {
            if (!isStepRelevant(step)) continue;

            AssignMinionsAction action = (AssignMinionsAction) step.action;
            Map<String, List<String>> assignmentMap = action.getAssignmentMap();
            Map<String, List<String>> alreadyAssignedMap = action.getAlreadyAssignedMap();
            int numberOfUnassignedMinions = action.getNumberOfUnassignedMinions();
            int strengthOfUnassignedMinions = action.getStrengthOfUnassignedMinions();


            alreadyAssignedMap.forEach((fp, shadow) -> {
                if (!assignmentMap.containsKey(fp)) {
                    assignmentMap.put(fp, new ArrayList<>());
                }
                assignmentMap.get(fp).addAll(shadow);
            });


            double[] probe = PROBE_SIDES.contains(side())
                    ? (action.getProbe() != null ? action.getProbe() : new double[AssignmentProbe.LENGTH]) : null;
            if (step.reward > 0) {
                // Chosen: good
                addLabeledPoints(data, assignmentMap, numberOfUnassignedMinions, strengthOfUnassignedMinions, action.getWoundsOnFp(), step.state, 1, step.group, probe);
            } else {
                // Chosen: bad
                addLabeledPoints(data, assignmentMap, numberOfUnassignedMinions, strengthOfUnassignedMinions, action.getWoundsOnFp(), step.state, 0, step.group, probe);
            }
        }

        return data;
    }

    private void addLabeledPoints(List<LabeledPoint> data, Map<String, List<String>> assignmentMap,
                                  int numberOfUnassignedMinions, int strengthOfUnassignedMinions,
                                  Map<String, Integer> woundsMap, double[] state, int label, int group, double[] probe) {
        try {
            double[] assignmentVector = CardFeatures.getAssignmentFeatures(assignmentMap, woundsMap,
                    numberOfUnassignedMinions, strengthOfUnassignedMinions);
            if (probe != null)
                assignmentVector = append(assignmentVector, probe);
            double[] extended = Arrays.copyOf(state, state.length + assignmentVector.length);
            System.arraycopy(assignmentVector, 0, extended, state.length, assignmentVector.length);
            data.add(new LabeledPoint(label, extended, group)); // group: the decision, for --model listwise
        } catch (CardNotFoundException ignore) {
        }
    }

    /**
     * -Dassign.probe=fp[,shadow] (training) / an "assign-probe" file naming the sides (play): assignments end with the
     * skirmishes they set up as the board stands, AssignmentProbe's numbers, where the printed strengths were all.
     */
    static final Set<String> PROBE_SIDES = System.getProperty("assign.probe") == null ? Set.of()
            : Set.of(System.getProperty("assign.probe").toLowerCase().split("[,\\s]+"));

    private String side() {
        return isForFp() ? "fp" : "shadow";
    }

    private static double[] append(double[] x, double[] more) {
        double[] out = Arrays.copyOf(x, x.length + more.length);
        System.arraycopy(more, 0, out, x.length, more.length);
        return out;
    }

    protected record AssignmentInfo(Map<String, List<String>> blueprintAssignment,
                                    Map<String, List<String>> physicalAssignment,
                                    int numberOfUnassignedMinions,
                                    int strengthOfUnassignedMinions) {
    }
}
