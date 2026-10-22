package com.gempukku.lotro.bots.rl.semanticaction;

import com.alibaba.fastjson2.JSONObject;
import com.gempukku.lotro.game.PhysicalCard;
import com.gempukku.lotro.game.state.GameState;
import com.gempukku.lotro.logic.decisions.AwaitingDecision;
import com.gempukku.lotro.logic.decisions.AwaitingDecisionType;
import com.gempukku.lotro.logic.decisions.CardsSelectionDecision;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class CardSelectionAction implements SemanticAction {
    protected final List<String> chosenBlueprintIds = new ArrayList<>();
    protected final List<Integer> woundsOnChosen = new ArrayList<>();
    protected final List<String> notChosenBlueprintIds = new ArrayList<>();
    protected final List<Integer> woundsOnNotChosen = new ArrayList<>();
    protected final String zoneString;
    /**
     * A decision's options with numbers only the live game gives, for a model that compares them (reconcile: keep
     * everything, then each card); null where none was written.
     */
    private String[] optionBlueprints;
    private double[][] optionExtras;
    private int optionChosen = -1;

    public CardSelectionAction(String answer, AwaitingDecision decision, GameState gameState) {
        String[] individualCards = answer.split(",");

        for (String individualCard : individualCards) {
            if (!individualCard.isEmpty()) {
                chosenBlueprintIds.add(gameState.getBlueprintId(Integer.parseInt(individualCard)));
                int wounds = 0;
                for (PhysicalCard physicalCard : gameState.getInPlay()) {
                    if (physicalCard.getCardId() == Integer.parseInt(individualCard)) {
                        wounds = gameState.getWounds(physicalCard);
                    }
                }
                woundsOnChosen.add(wounds);
            }
        }

        String zoneOfAllCards = null;

        if (decision instanceof CardsSelectionDecision csd) {
            List<String> allChoices = Arrays.asList(csd.getDecisionParameters().get("cardId"));
            for (String choice : allChoices) {
                if (!chosenBlueprintIds.contains(gameState.getBlueprintId(Integer.parseInt(choice)))) {
                    notChosenBlueprintIds.add(gameState.getBlueprintId(Integer.parseInt(choice)));
                    int wounds = 0;
                    for (PhysicalCard physicalCard : gameState.getInPlay()) {
                        if (physicalCard.getCardId() == Integer.parseInt(choice)) {
                            wounds = gameState.getWounds(physicalCard);
                        }
                    }
                    woundsOnNotChosen.add(wounds);
                }
                for (PhysicalCard physicalCard : gameState.getAllCards()) {
                    if (physicalCard.getCardId() == Integer.parseInt(choice)) {
                        String zoneOfThisCard = physicalCard.getZone().getHumanReadable();
                        if (zoneOfAllCards == null) {
                            zoneOfAllCards = zoneOfThisCard;
                        }
                        if (!zoneOfAllCards.equals(zoneOfThisCard)) {
                            throw new IllegalArgumentException("Cards are not from the same zone");
                        }
                    }
                }

            }
        }

        this.zoneString = zoneOfAllCards;

        if (chosenBlueprintIds.size() != woundsOnChosen.size() || notChosenBlueprintIds.size() != woundsOnNotChosen.size()) {
            throw new IllegalArgumentException("Card and wound list have different sizes");
        }
    }

    public CardSelectionAction(String[] chosenBlueprintIds, String[] notChosenBlueprintIds, Integer[] woundsOnChosen, Integer[] woundsOnNotChosen, String zone) {
        this.chosenBlueprintIds.addAll(Arrays.asList(chosenBlueprintIds));
        this.notChosenBlueprintIds.addAll(Arrays.asList(notChosenBlueprintIds));
        this.woundsOnChosen.addAll(Arrays.asList(woundsOnChosen));
        this.woundsOnNotChosen.addAll(Arrays.asList(woundsOnNotChosen));
        this.zoneString = zone;
    }

    public List<String> getChosenBlueprintIds() {
        return chosenBlueprintIds;
    }

    public List<String> getNotChosenBlueprintIds() {
        return notChosenBlueprintIds;
    }

    public List<Integer> getWoundsOnChosen() {
        return woundsOnChosen;
    }

    public List<Integer> getWoundsOnNotChosen() {
        return woundsOnNotChosen;
    }

    public String getZoneString() {
        return zoneString;
    }

    public CardSelectionAction withOptions(String[] blueprints, double[][] extras, int chosen) {
        this.optionBlueprints = blueprints;
        this.optionExtras = extras;
        this.optionChosen = chosen;
        return this;
    }

    /** The options, if written, into a JSON form (subclasses write their own). */
    protected void putOptions(JSONObject obj) {
        if (optionExtras != null) {
            obj.put("options", optionBlueprints);
            obj.put("optionExtras", optionExtras);
            obj.put("optionChosen", optionChosen);
        }
    }

    public String[] getOptionBlueprints() {
        return optionBlueprints;
    }

    public double[][] getOptionExtras() {
        return optionExtras;
    }

    public int getOptionChosen() {
        return optionChosen;
    }

    @Override
    public String toDecisionString(AwaitingDecision decision, GameState gameState) {
        if (decision.getDecisionType() != AwaitingDecisionType.CARD_SELECTION) {
            throw new IllegalArgumentException("Wrong decision type.");
        }

        int min = Integer.parseInt(decision.getDecisionParameters().get("min")[0]);
        int max = Integer.parseInt(decision.getDecisionParameters().get("max")[0]);
        if (chosenBlueprintIds.size() < min || chosenBlueprintIds.size() > max) {
            throw new IllegalArgumentException("Chosen number out of bounds.");
        }

        if (chosenBlueprintIds.isEmpty()) {
            return "";
        }

        List<String> chosenBlueprintIdsCopy = new ArrayList<>(chosenBlueprintIds);

        List<String> chosenIds = new ArrayList<>();
        for (String cardId : decision.getDecisionParameters().get("cardId")) {
            if (chosenBlueprintIdsCopy.contains(gameState.getBlueprintId(Integer.parseInt(cardId)))) {
                chosenIds.add(cardId);
                chosenBlueprintIdsCopy.remove(gameState.getBlueprintId(Integer.parseInt(cardId)));
            }
        }

        if (chosenBlueprintIdsCopy.isEmpty()) {
            return String.join(",", chosenIds);
        } else {
            throw new IllegalArgumentException("Cards to pick are not present in the decision.");
        }
    }

    @Override
    public JSONObject toJson() {
        JSONObject obj = new JSONObject();
        obj.put("type", "CardSelectionAction");
        obj.put("chosenBlueprints", chosenBlueprintIds.toArray());
        obj.put("woundsOnChosen", woundsOnChosen.toArray());
        obj.put("notChosenBlueprints", notChosenBlueprintIds.toArray());
        obj.put("woundsOnNotChosen", woundsOnNotChosen.toArray());
        obj.put("zone", zoneString);
        putOptions(obj);
        return obj;
    }

    public static CardSelectionAction fromJson(JSONObject obj) {
        CardSelectionAction a = new CardSelectionAction(obj.getObject("chosenBlueprints", String[].class), obj.getObject("notChosenBlueprints", String[].class),
                obj.getObject("woundsOnChosen", Integer[].class), obj.getObject("woundsOnNotChosen", Integer[].class),
                obj.getString("zone"));
        if (obj.containsKey("optionExtras"))
            a.withOptions(obj.getObject("options", String[].class), obj.getObject("optionExtras", double[][].class),
                    obj.getIntValue("optionChosen"));
        return a;
    }
}
