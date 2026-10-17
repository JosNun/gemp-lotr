package com.gempukku.lotro.bots.rl.fotrstarters;

import com.gempukku.lotro.common.CardType;
import com.gempukku.lotro.common.Keyword;
import com.gempukku.lotro.common.PossessionClass;
import com.gempukku.lotro.common.Side;
import com.gempukku.lotro.game.CardNotFoundException;
import com.gempukku.lotro.game.LotroCardBlueprint;
import com.gempukku.lotro.game.LotroCardBlueprintLibrary;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import static com.gempukku.lotro.common.Timeword.*;

public class CardFeatures {
    private static LotroCardBlueprintLibrary library = null;
    public static final String PASS = "pass";

    /**
     * Which feature set this thread is computing: 1, the original, or 2 ({@link GeneralStateFeatures}). Set by the
     * bot around each decision and by BotTrain around training, so bots on either set can share a process -- a
     * model is only ever fed vectors of the set it was trained on.
     */
    private static final ThreadLocal<Integer> VERSION = ThreadLocal.withInitial(() -> 1);

    public static LotroCardBlueprint blueprint(String blueprintId) throws CardNotFoundException {
        return library.getLotroCardBlueprint(blueprintId);
    }

    public static int version() {
        return VERSION.get();
    }

    public static void setVersion(int version) {
        VERSION.set(version);
    }

    /** Keywords that change what a card does in a fight or a turn, as counts (damage +2 is 2) or flags. */
    private static final List<Keyword> KEYWORDS = List.of(Keyword.DAMAGE, Keyword.FIERCE, Keyword.ARCHER,
            Keyword.AMBUSH, Keyword.ROAMING, Keyword.ENDURING, Keyword.LURKER, Keyword.MUSTER, Keyword.TOIL,
            Keyword.HUNTER, Keyword.RANGER, Keyword.VALIANT, Keyword.UNHASTY, Keyword.RING_BOUND, Keyword.TWILIGHT,
            Keyword.TALE, Keyword.SPELL, Keyword.WEATHER, Keyword.SEARCH, Keyword.STEALTH, Keyword.KNIGHT,
            Keyword.BESIEGER);

    /** Length of a set-2 card vector; PASS is this many zeros. */
    private static int v2Length() {
        return 22 + 2 + KEYWORDS.size() + com.gempukku.lotro.common.Culture.values().length
                + com.gempukku.lotro.common.Race.values().length;
    }

    private CardFeatures() {

    }

    public static void init(LotroCardBlueprintLibrary library) {
        CardFeatures.library = library;
    }

    public static double[] getCardFeatures(String blueprintId, int woundsPlaced) throws CardNotFoundException {
        if (version() == 2)
            return getCardFeaturesV2(blueprintId, woundsPlaced);
        if (library != null) {
            try {
                LotroCardBlueprint blueprint = library.getLotroCardBlueprint(blueprintId);

                List<Double> features = new ArrayList<>();

                features.add(blueprint.getSide() == Side.SHADOW ? 1.0 : 0.0);
                features.add(blueprint.getSide() == Side.FREE_PEOPLE ? 1.0 : 0.0);
                features.add(blueprint.getCardType() == CardType.COMPANION ? 1.0 : 0.0);
                features.add(blueprint.canStartWithRing() ? 1.0 : 0.0);
                features.add(blueprint.getCardType() == CardType.ALLY ? 1.0 : 0.0);
                features.add(blueprint.getCardType() == CardType.MINION ? 1.0 : 0.0);
                features.add(blueprint.getCardType() == CardType.POSSESSION || blueprint.getCardType() == CardType.ARTIFACT ? 1.0 : 0.0);
                features.add(blueprint.getCardType() == CardType.CONDITION ? 1.0 : 0.0);
                features.add(blueprint.getCardType() == CardType.FOLLOWER ? 1.0 : 0.0);
                features.add(blueprint.getCardType() == CardType.EVENT && blueprint.hasTimeword(FELLOWSHIP) ? 1.0 : 0.0);
                features.add(blueprint.getCardType() == CardType.EVENT && blueprint.hasTimeword(SHADOW) ? 1.0 : 0.0);
                features.add(blueprint.getCardType() == CardType.EVENT && blueprint.hasTimeword(MANEUVER) ? 1.0 : 0.0);
                features.add(blueprint.getCardType() == CardType.EVENT && blueprint.hasTimeword(ARCHERY) ? 1.0 : 0.0);
                features.add(blueprint.getCardType() == CardType.EVENT && blueprint.hasTimeword(ASSIGNMENT) ? 1.0 : 0.0);
                features.add(blueprint.getCardType() == CardType.EVENT && blueprint.hasTimeword(SKIRMISH) ? 1.0 : 0.0);
                features.add(blueprint.getCardType() == CardType.EVENT && blueprint.hasTimeword(REGROUP) ? 1.0 : 0.0);
                features.add(blueprint.getCardType() == CardType.EVENT && blueprint.hasTimeword(RESPONSE) ? 1.0 : 0.0);

                features.add((double) blueprint.getTwilightCost());
                features.add((double) blueprint.getStrength());
                features.add((double) blueprint.getVitality());
                features.add((double) blueprint.getSiteNumber());

                features.add((double) woundsPlaced);

                return features.stream().mapToDouble(Double::doubleValue).toArray();
            } catch (CardNotFoundException e) {
                if (blueprintId.equals(PASS)) {
                    return new double[22];
                } else {
                    throw e;
                }
            }
        } else {
            throw new IllegalStateException("Blueprint library not initialized");
        }
    }

    /**
     * Set 2: set 1's 22 numbers, then resistance, unique, the {@link #KEYWORDS}, culture and race one-hot. Set 1
     * cannot tell a damage +1 fierce minion from a plain one of the same cost and strength, nor one culture's
     * cards from another's, which across decks is most of what distinguishes them.
     */
    private static double[] getCardFeaturesV2(String blueprintId, int woundsPlaced) throws CardNotFoundException {
        if (blueprintId.equals(PASS))
            return new double[v2Length()];
        LotroCardBlueprint blueprint = library.getLotroCardBlueprint(blueprintId);
        VERSION.set(1);
        double[] base;
        try {
            base = getCardFeatures(blueprintId, woundsPlaced);
        } finally {
            VERSION.set(2);
        }
        double[] f = Arrays.copyOf(base, v2Length());
        int i = base.length;
        f[i++] = blueprint.getResistance();
        f[i++] = blueprint.isUnique() ? 1 : 0;
        for (Keyword keyword : KEYWORDS)
            f[i++] = blueprint.getKeywordCount(keyword);
        if (blueprint.getCulture() != null)
            f[i + blueprint.getCulture().ordinal()] = 1;
        i += com.gempukku.lotro.common.Culture.values().length;
        if (blueprint.getRace() != null)
            f[i + blueprint.getRace().ordinal()] = 1;
        return f;
    }

    /** State features each option is crossed with in set 2, and the scale that brings each to about 0..1. */
    private static final String[] CROSS_STATE = {"iAmFp", "twilight", "inSkirmish", "mySkirmishMargin",
            "fpCompanions", "minions", "shadowAdvantage", "rbResistanceLeft", "site", "myHand", "fpMinVitalityLeft"};
    private static final double[] CROSS_STATE_SCALE = {1, 10, 1, 5, 5, 5, 10, 10, 9, 8, 4};
    private static final int CROSS_CARD = 12;

    /**
     * An option's vector after the state's. In set 1, just that. In set 2, followed by every product of a few state
     * features ({@link #CROSS_STATE}) with a summary of the option's card (type, cost, strength, vitality, damage):
     * a logistic model adds the state's own term equally to every option, so without these the state can never
     * change which option wins -- "play the skirmish event when losing the skirmish" is a product of the two.
     * {@code blueprintId} is the card the option is about, or PASS.
     */
    public static double[] join(double[] state, double[] option, String blueprintId) {
        int crossLength = version() == 2 ? CROSS_STATE.length * CROSS_CARD : 0;
        double[] x = new double[state.length + option.length + crossLength];
        System.arraycopy(state, 0, x, 0, state.length);
        System.arraycopy(option, 0, x, state.length, option.length);
        if (crossLength == 0)
            return x;
        if (state.length != GeneralStateFeatures.length())
            throw new IllegalStateException("Set-2 join of a " + state.length + "-long state; set 2 states are "
                    + GeneralStateFeatures.length());
        double[] card = crossSummary(blueprintId);
        int k = state.length + option.length;
        for (int s = 0; s < CROSS_STATE.length; s++) {
            double v = state[GeneralStateFeatures.index(CROSS_STATE[s])] / CROSS_STATE_SCALE[s];
            for (double c : card)
                x[k++] = v * c;
        }
        return x;
    }

    private static double[] crossSummary(String blueprintId) {
        double[] c = new double[CROSS_CARD];
        if (blueprintId == null || blueprintId.equals(PASS) || library == null)
            return c;
        try {
            LotroCardBlueprint bp = library.getLotroCardBlueprint(blueprintId);
            CardType type = bp.getCardType();
            c[0] = type == CardType.COMPANION ? 1 : 0;
            c[1] = type == CardType.ALLY ? 1 : 0;
            c[2] = type == CardType.MINION ? 1 : 0;
            c[3] = type == CardType.POSSESSION || type == CardType.ARTIFACT ? 1 : 0;
            c[4] = type == CardType.CONDITION ? 1 : 0;
            c[5] = type == CardType.EVENT && bp.hasTimeword(SKIRMISH) ? 1 : 0;
            c[6] = type == CardType.EVENT && !bp.hasTimeword(SKIRMISH) ? 1 : 0;
            c[7] = bp.getSide() == Side.FREE_PEOPLE ? 1 : 0;
            c[8] = bp.getTwilightCost() / 5.0;
            c[9] = bp.getStrength() / 10.0;
            c[10] = bp.getVitality() / 5.0;
            c[11] = bp.getKeywordCount(Keyword.DAMAGE) / 2.0;
        } catch (CardNotFoundException ignored) {
        }
        return c;
    }

    public static double[] getFpAssignedCardFeatures(String blueprintId, int woundsPlaced, int numberOfMinions, int minionStrength) throws CardNotFoundException {
        // TODO items
        if (library != null) {
            LotroCardBlueprint blueprint = library.getLotroCardBlueprint(blueprintId);

            List<Double> features = new ArrayList<>();

            features.add(blueprint.getCardType() == CardType.COMPANION ? 1.0 : 0.0);
            features.add(blueprint.canStartWithRing() ? 1.0 : 0.0);
            features.add((double) blueprint.getTwilightCost());
            features.add((double) blueprint.getStrength());
            features.add((double) blueprint.getVitality());
            features.add((double) woundsPlaced);

            features.add(((double) numberOfMinions));
            features.add(((double) minionStrength));

            return features.stream().mapToDouble(Double::doubleValue).toArray();
        } else {
            throw new IllegalStateException("Blueprint library not initialized");
        }
    }

    public static double[] getItemCardFeatures(String blueprintId, String holderBlueprintId, int woundsPlaced) throws CardNotFoundException {
        if (library != null) {
            try {
                LotroCardBlueprint blueprint = library.getLotroCardBlueprint(blueprintId);
                LotroCardBlueprint holderBlueprint = library.getLotroCardBlueprint(holderBlueprintId);

                List<Double> features = new ArrayList<>();

                features.add(blueprint.getSide() == Side.SHADOW ? 1.0 : 0.0);
                features.add(blueprint.getSide() == Side.FREE_PEOPLE ? 1.0 : 0.0);
                features.add((double) blueprint.getTwilightCost());
                features.add(blueprint.getPossessionClasses() != null && blueprint.getPossessionClasses().contains(PossessionClass.HAND_WEAPON) ? 1.0 : 0.0);
                features.add(blueprint.getPossessionClasses() != null && blueprint.getPossessionClasses().contains(PossessionClass.ARMOR) ? 1.0 : 0.0);
                features.add(blueprint.getPossessionClasses() != null && blueprint.getPossessionClasses().contains(PossessionClass.SHIELD) ? 1.0 : 0.0);

                features.add(holderBlueprint.canStartWithRing() ? 1.0 : 0.0);
                features.add((double) holderBlueprint.getTwilightCost());
                features.add((double) holderBlueprint.getStrength());
                features.add((double) holderBlueprint.getVitality());
                features.add((double) woundsPlaced);

                return features.stream().mapToDouble(Double::doubleValue).toArray();
            } catch (CardNotFoundException e) {
                if (blueprintId.equals(PASS)) {
                    return new double[11];
                } else {
                    throw e;
                }
            }
        } else {
            throw new IllegalStateException("Blueprint library not initialized");
        }
    }

    public static double[] getAssignmentFeatures(Map<String, List<String>> assignmentMap, Map<String, Integer> woundsMap,
                                                 int numberOfUnassignedMinions, int strengthOfUnassignedMinions) throws CardNotFoundException {
        List<Double> tbr = new ArrayList<>();

        List<String> companions = assignmentMap.keySet().stream().filter(s -> {
            try {
                return library.getLotroCardBlueprint(s).getCardType().equals(CardType.COMPANION);
            } catch (CardNotFoundException e) {
                return false;
            }
        }).collect(Collectors.toList());
        List<String> allies = assignmentMap.keySet().stream().filter(s -> {
            try {
                return library.getLotroCardBlueprint(s).getCardType().equals(CardType.ALLY);
            } catch (CardNotFoundException e) {
                return false;
            }
        }).collect(Collectors.toList());

        companions.sort((o1, o2) -> {
            try {
                return Integer.compare(library.getLotroCardBlueprint(o1).getStrength(), library.getLotroCardBlueprint(o2).getStrength());
            } catch (CardNotFoundException e) {
                return 0;
            }
        });
        allies.sort((o1, o2) -> {
            try {
                return Integer.compare(library.getLotroCardBlueprint(o1).getStrength(), library.getLotroCardBlueprint(o2).getStrength());
            } catch (CardNotFoundException e) {
                return 0;
            }
        });

        while (companions.size() < 9) {
            companions.add(PASS);
        }
        while (allies.size() < 4) {
            allies.add(PASS);
        }

        final int[] companionsAdded = {0};
        companions.forEach(companion -> {
            if (companionsAdded[0] >= 9) {
                return;
            }
            companionsAdded[0]++;
            if (companion.equals(PASS)) {
                double[] nullVector = new double[8];
                tbr.addAll(Arrays.stream(nullVector).boxed().collect(Collectors.toList()));
            } else {
                int strength = assignmentMap.get(companion).stream().mapToInt(value -> {
                    try {
                        return library.getLotroCardBlueprint(value).getStrength();
                    } catch (CardNotFoundException e) {
                        e.printStackTrace();
                        return 0;
                    }
                }).sum();

                try {
                    double[] thisAssignment = getFpAssignedCardFeatures(companion, woundsMap.get(companion), assignmentMap.get(companion).size(), strength);
                    tbr.addAll(Arrays.stream(thisAssignment).boxed().collect(Collectors.toList()));
                } catch (CardNotFoundException e) {
                    double[] nullVector = new double[8];
                    tbr.addAll(Arrays.stream(nullVector).boxed().collect(Collectors.toList()));
                }
            }
        });

        final int[] alliesAdded = {0};
        allies.forEach(ally -> {
            if (alliesAdded[0] >= 4) {
                return;
            }
            alliesAdded[0]++;
            if (ally.equals(PASS)) {
                double[] nullVector = new double[8];
                tbr.addAll(Arrays.stream(nullVector).boxed().collect(Collectors.toList()));
            } else {
                int strength = assignmentMap.get(ally).stream().mapToInt(value -> {
                    try {
                        return library.getLotroCardBlueprint(value).getStrength();
                    } catch (CardNotFoundException e) {
                        e.printStackTrace();
                        return 0;
                    }
                }).sum();

                try {
                    double[] thisAssignment = getFpAssignedCardFeatures(ally, woundsMap.get(ally), assignmentMap.get(ally).size(), strength);
                    tbr.addAll(Arrays.stream(thisAssignment).boxed().collect(Collectors.toList()));
                } catch (CardNotFoundException e) {
                    double[] nullVector = new double[8];
                    tbr.addAll(Arrays.stream(nullVector).boxed().collect(Collectors.toList()));
                }
            }
        });

        tbr.add((double) numberOfUnassignedMinions);
        tbr.add((double) strengthOfUnassignedMinions);

        return tbr.stream().mapToDouble(Double::doubleValue).toArray();
    }

    public static double[] getSkirmishPlayCardFeatures(String cardBlueprintId, String fpBlueprintId, boolean fpCanExert, List<String> minionBlueprintIds, List<Boolean> minionsCanExert, boolean sourceInSkirmish) throws CardNotFoundException {
        if (library != null) {
            try {
                LotroCardBlueprint cardBlueprint = library.getLotroCardBlueprint(cardBlueprintId);
                LotroCardBlueprint fpBlueprint = library.getLotroCardBlueprint(fpBlueprintId);

                List<Double> features = new ArrayList<>();

                features.add(cardBlueprint.getSide() == Side.SHADOW ? 1.0 : 0.0);
                features.add(cardBlueprint.getSide() == Side.FREE_PEOPLE ? 1.0 : 0.0);
                features.add((double)cardBlueprint.getTwilightCost());


                boolean shadowCanAffect = false;
                for (int i = 0; i < minionBlueprintIds.size(); i++) {
                    String minionBlueprintId = minionBlueprintIds.get(i);
                    boolean minionCanExert = minionsCanExert.get(i);
                    if (canCardAffectTarget(cardBlueprintId, minionBlueprintId, sourceInSkirmish, minionCanExert)) {
                        shadowCanAffect = true;
                        break;
                    }
                }
                features.add(canCardAffectTarget(cardBlueprintId, fpBlueprintId, sourceInSkirmish, fpCanExert) || shadowCanAffect ? 1.0 : 0.0);

                return features.stream().mapToDouble(Double::doubleValue).toArray();
            } catch (CardNotFoundException e) {
                if (cardBlueprintId.equals(PASS)) {
                    return new double[4];
                } else {
                    throw e;
                }
            }
        } else {
            throw new IllegalStateException("Blueprint library not initialized");
        }
    }

    private static boolean canCardAffectTarget(String sourceBlueprintId, String targetBlueprintId, boolean sourceInSkirmish, boolean targetCanExert) {
        if (library == null)
            throw new IllegalStateException("Blueprint library not initialized");

        try {
            LotroCardBlueprint sourceCard = library.getLotroCardBlueprint(sourceBlueprintId);
            LotroCardBlueprint targetCard = library.getLotroCardBlueprint(targetBlueprintId);

            Object effectsObj = sourceCard.getJsonDefinition().get("effects");
            if (effectsObj == null)
                return false;

            return containsMatchingSelect(effectsObj, targetCard, sourceInSkirmish && targetBlueprintId.equals(sourceBlueprintId), null, targetCanExert);

        } catch (CardNotFoundException e) {
            return false;
        }
    }

    private static boolean containsMatchingSelect(Object node, LotroCardBlueprint targetCard, boolean sourceInSkirmish, String costSelect, boolean targetCanExert) {
        if (node instanceof JSONObject obj) {
            for (Object keyObj : obj.keySet()) {
                String key = keyObj.toString();
                if (key.equals("cost")) {
                    Object costObj = obj.get(key);
                    if (targetCanExert && costObj instanceof JSONObject costJson &&
                            "exert".equals(costJson.get("type")) &&
                            costJson.containsKey("memorize")) {
                        costSelect = (String) costJson.get("select");
                    }
                    continue; // Skip cost blocks
                }

                Object value = obj.get(key);

                // Direct "select" check
                if (key.equals("select")) {
                    if (value instanceof String select) {
                        if (select.startsWith("choose(")) {
                            return matchesFilter(select, targetCard);
                        } else if ("self".equalsIgnoreCase(select) && sourceInSkirmish) {
                            return true;
                        } else if (select.startsWith("memory(") && costSelect != null) {
                            return matchesFilter(costSelect, targetCard);
                        }
                    }
                }

                // Recurse into other fields
                if (containsMatchingSelect(value, targetCard, sourceInSkirmish, costSelect, targetCanExert))
                    return true;
            }
        } else if (node instanceof JSONArray arr) {
            for (Object item : arr) {
                if (containsMatchingSelect(item, targetCard, sourceInSkirmish, costSelect, targetCanExert)) {
                    return true;
                }
            }
        }
        return false;
    }

    private static boolean matchesFilter(String select, LotroCardBlueprint targetCard) {
        if (!select.startsWith("choose(") || !select.endsWith(")"))
            return false;

        String filtersRaw = select.substring("choose(".length(), select.length() - 1);
        List<String> filters = splitTopLevelFilters(filtersRaw);

        for (String filter : filters) {
            filter = filter.trim().toLowerCase();

            if (filter.equals("your"))
                continue;

            if (filter.startsWith("or(") && filter.endsWith(")")) {
                String orArgs = filter.substring("or(".length(), filter.length() - 1);
                List<String> innerFilters = splitTopLevelFilters(orArgs);

                boolean anyMatched = false;
                for (String inner : innerFilters) {
                    if (matchesSingleFilter(inner.trim(), targetCard)) {
                        anyMatched = true;
                        break;
                    }
                }

                if (!anyMatched)
                    return false;
            } else if (!matchesSingleFilter(filter, targetCard)) {
                return false;
            }
        }

        return true;
    }

    private static List<String> splitTopLevelFilters(String raw) {
        List<String> parts = new ArrayList<>();
        StringBuilder current = new StringBuilder();
        int parens = 0;

        for (char c : raw.toCharArray()) {
            if (c == ',' && parens == 0) {
                parts.add(current.toString());
                current.setLength(0);
            } else {
                if (c == '(') parens++;
                else if (c == ')') parens--;
                current.append(c);
            }
        }

        if (current.length() > 0)
            parts.add(current.toString());

        return parts;
    }

    private static boolean matchesSingleFilter(String filter, LotroCardBlueprint targetCard) {
        if (filter.startsWith("culture(") && filter.endsWith(")")) {
            String cultureValue = filter.substring("culture(".length(), filter.length() - 1);
            return targetCard.getCulture().getHumanReadable().equalsIgnoreCase(cultureValue);
        } else if (filter.startsWith("name(") && filter.endsWith(")")) {
            String nameValue = filter.substring("name(".length(), filter.length() - 1);
            return targetCard.getSanitizedFullName().toLowerCase().contains(nameValue);
        } else if (targetCard.getRace() != null &&
                targetCard.getRace().getHumanReadable().equalsIgnoreCase(filter)) {
            return true; // Race match
        } else if (targetCard.hasKeyword(Keyword.parse(filter))) {
            return true; // Keyword match
        } else if (targetCard.getCardType().toString().equalsIgnoreCase(filter)) {
            return true; // Card type match
        }

        return false;
    }
}
