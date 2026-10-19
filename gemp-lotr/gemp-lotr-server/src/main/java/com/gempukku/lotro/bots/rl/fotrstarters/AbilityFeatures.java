package com.gempukku.lotro.bots.rl.fotrstarters;

import org.hjson.JsonArray;
import org.hjson.JsonObject;
import org.hjson.JsonValue;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.util.*;

/**
 * What a card's abilities cost and do, read from its card definition: for card feature set 3, after set 2's type,
 * cost, strength, keywords, culture and race. Set 2 cannot tell "Archery: discard this weapon to make the archery total
 * +1" (Hand Axe, which the bot used on 58% of the times it was offered against people's 15%) from "Archery: spot an
 * Elf archer to make the archery total +1" (Double Shot: 1% against 53%) -- the cost is what makes one a waste and the
 * other free. Each ability (activated, event, response, optional trigger) adds its cost types to the first half of the
 * vector and its effect types to the second, by amount where the definition gives one.
 */
public final class AbilityFeatures {
    /** Cost types, with "self" variants where paying with the card itself is what matters. */
    private static final List<String> COSTS = List.of("addBurdens", "discard:self", "discard", "exert:self", "exert",
            "removeTwilight", "addTwilight", "discardFromHand", "wound", "RemoveCultureTokens", "returnToHand",
            "exhaust", "discardTopCardsFromDeck", "removeBurdens", "addThreats", "removeThreats", "other");
    private static final List<String> EFFECTS = List.of("modifyStrength", "addKeyword", "wound", "heal", "drawCards",
            "addBurdens", "removeBurdens", "preventWound", "discard", "playCard", "fetchCard", "modifyArcheryTotal",
            "addThreats", "removeThreats", "addTwilight", "removeTwilight", "exert", "exhaust", "cancelSkirmish",
            "liberateSite", "takeControlOfSite", "putOnRing", "AddCultureTokens", "returnToHand", "modifyResistance",
            "cantTakeWounds", "addModifier", "other");
    private static final Map<String, String> EFFECT_ALIAS = Map.ofEntries(
            Map.entry("negateWound", "preventWound"), Map.entry("playCardFromDiscard", "playCard"),
            Map.entry("playCardFromDrawDeck", "playCard"), Map.entry("PlayCardFromHand", "playCard"),
            Map.entry("playCardFromHand", "playCard"), Map.entry("putCardsFromDiscardIntoHand", "fetchCard"),
            Map.entry("putCardsFromDeckIntoHand", "fetchCard"), Map.entry("addKeyword", "addKeyword"));
    /** Wrappers whose children are walked: they are not themselves costs or effects. */
    private static final Set<String> WRAPPERS = Set.of("choice", "Conditional", "If", "if", "optional", "repeat",
            "preventable", "forEachYouCanSpot", "costToEffect");
    private static final Set<String> ABILITIES = Set.of("activated", "event", "responseEvent", "activatedTrigger",
            "Response", "activatedInDrawDeck", "activatedFromStacked");

    public static final int LENGTH = COSTS.size() + EFFECTS.size() + 2;

    private static volatile Map<String, double[]> byId;
    private static com.gempukku.lotro.game.LotroCardBlueprintLibrary library;

    private AbilityFeatures() {
    }

    public static synchronized void init(File cardsDir, com.gempukku.lotro.game.LotroCardBlueprintLibrary lib) {
        if (byId != null)
            return;
        library = lib;
        Map<String, double[]> m = new HashMap<>();
        Deque<File> todo = new ArrayDeque<>(List.of(cardsDir));
        while (!todo.isEmpty()) {
            File f = todo.pop();
            if (f.isDirectory()) {
                File[] kids = f.listFiles();
                if (kids != null)
                    todo.addAll(Arrays.asList(kids));
            } else if (f.getName().endsWith(".hjson") || f.getName().endsWith(".json")) {
                try (Reader r = new FileReader(f, StandardCharsets.UTF_8)) {
                    JsonValue v = JsonValue.readHjson(r);
                    if (!v.isObject())
                        continue;
                    for (JsonObject.Member card : v.asObject())
                        if (card.getValue().isObject())
                            m.putIfAbsent(card.getName(), profile(card.getValue().asObject()));
                } catch (IOException | RuntimeException ignored) {
                    // a file the parser does not take: those cards get zeros
                }
            }
        }
        byId = m;
    }

    /** The vector for a blueprint; zeros for a card with no abilities or none found. */
    public static double[] of(String blueprintId) {
        Map<String, double[]> m = byId;
        if (m == null || blueprintId == null)
            return new double[LENGTH];
        double[] v = m.get(blueprintId);
        if (v == null && library != null) {
            String base = library.getBaseBlueprintId(blueprintId);
            if (base != null)
                v = m.get(base);
        }
        if (v == null)
            v = m.get(blueprintId.replaceAll("[*T]+$", ""));
        return v == null ? new double[LENGTH] : v;
    }

    private static double[] profile(JsonObject card) {
        double[] f = new double[LENGTH];
        JsonValue effects = card.get("effects");
        if (effects == null)
            return f;
        List<JsonObject> list = new ArrayList<>();
        if (effects.isArray()) {
            for (JsonValue e : effects.asArray())
                if (e.isObject())
                    list.add(e.asObject());
        } else if (effects.isObject())
            list.add(effects.asObject());
        for (JsonObject e : list) {
            String type = str(e, "type");
            boolean optionalTrigger = "Trigger".equalsIgnoreCase(type) && e.getBoolean("optional", false);
            if (type == null || !(ABILITIES.contains(type) || optionalTrigger))
                continue;
            f[COSTS.size() + EFFECTS.size()]++; // abilities
            if (e.get("requires") != null)
                f[COSTS.size() + EFFECTS.size() + 1]++;
            walk(e.get("cost"), f, true);
            walk(e.get("extraCost"), f, true);
            walk(e.get("effect"), f, false);
        }
        return f;
    }

    private static void walk(JsonValue v, double[] f, boolean cost) {
        if (v == null)
            return;
        if (v.isArray()) {
            for (JsonValue x : v.asArray())
                walk(x, f, cost);
            return;
        }
        if (!v.isObject())
            return;
        JsonObject o = v.asObject();
        String type = str(o, "type");
        if (type == null)
            return;
        if (WRAPPERS.contains(type)) {
            for (JsonObject.Member mem : o)
                if (mem.getValue().isObject() || mem.getValue().isArray())
                    walk(mem.getValue(), f, cost);
            return;
        }
        double amount = 1;
        JsonValue a = o.get("amount");
        if (a != null && a.isNumber())
            amount = Math.max(1, a.asDouble());
        if (cost) {
            String key = type;
            if (("discard".equals(type) || "exert".equals(type)) && "self".equals(str(o, "select")))
                key = type + ":self";
            int i = COSTS.indexOf(key);
            f[i >= 0 ? i : COSTS.indexOf("other")] += amount;
        } else {
            String key = EFFECT_ALIAS.getOrDefault(type, type);
            int i = EFFECTS.indexOf(key);
            f[COSTS.size() + (i >= 0 ? i : EFFECTS.indexOf("other"))] += amount;
        }
    }

    private static String str(JsonObject o, String key) {
        JsonValue v = o.get(key);
        return v != null && v.isString() ? v.asString() : null;
    }
}
