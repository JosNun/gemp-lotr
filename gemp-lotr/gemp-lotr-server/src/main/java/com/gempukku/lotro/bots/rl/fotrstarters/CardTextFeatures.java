package com.gempukku.lotro.bots.rl.fotrstarters;

import org.hjson.JsonObject;
import org.hjson.JsonValue;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

/**
 * What an option's card does, read from its card definition as a bag of tokens over a fixed vocabulary. Set 2 sees a
 * card's printed numbers and keywords and nothing of its game text; AbilityFeatures (set 3) counted cost and effect
 * types and added nothing in play, because counting flattens away who an effect lands on and when it fires. Over
 * held-out July the cards the bot misjudged most were abilities it cannot see at all: site abilities (Rohirrim Road,
 * played 15% against people's 24%; Dunharrow Plateau 16% against 36%), characters' "Use" abilities (Legolas,
 * Merry, Úlairë Enquëa 18% against 97%), and events whose cost is the point (Saved From the Fire, 0.6% against 38%).
 * <p>
 * Each token names a role and what fills it: {@code ab:activated@regroup} (an ability, and its phase),
 * {@code cost:exert}, {@code cost>self} (who pays), {@code effect:heal}, {@code effect>companion} (who it lands on),
 * {@code effect:modifystrength#} (summed amount), {@code trig:losesskirmish}, {@code req:canspot},
 * {@code static:modifycost}. Which abilities an option brings depends on the action: playing a card brings all of
 * them; using one brings the activated abilities of the current phase; an optional trigger brings the triggers.
 * <p>
 * The vocabulary is the tokens found on at least {@link #MIN_CARDS} cards, written beside the models ("card-text")
 * so a bot reads exactly the list it was fit on.
 */
public final class CardTextFeatures {
    /** A token must appear on this many cards' definitions to be in the vocabulary: rarer ones are a card's name. */
    public static final int MIN_CARDS = Integer.getInteger("cardText.minCards", 5);
    /**
     * The sets the vocabulary is counted over (-DcardText.sets, "1-10" by default: Movie Block's). Counted over all
     * 3,997 definitions, a 15-card floor kept 461 tokens and dropped the kill in Saved From the Fire's cost and the
     * cost modifier behind Rohirrim Road -- tokens common in the format and rare in the 20 sets around it.
     */
    private static final String SETS = System.getProperty("cardText.sets", "1-10");

    private static final Set<String> WRAPPERS = Set.of("choice", "conditional", "if", "optional", "repeat",
            "preventable", "foreachyoucanspot", "costtoeffect", "memorizenumber", "memorizeactive");
    private static final Set<String> ACTIVATED = Set.of("activated", "activatedindrawdeck", "activatedfromstacked");
    /** Members whose string value is a filter, read for the words in it. */
    private static final Set<String> FILTERS = Set.of("select", "filter", "against", "on", "target", "from", "where");
    /** Filter words that are not about who: plumbing, memory names. */
    private static final Set<String> NOT_WHO = Set.of("choose", "all", "memory", "and", "or", "random", "count",
            "frommemory", "self_", "any", "min", "max");

    private static volatile Map<String, JsonObject> defs;
    private static com.gempukku.lotro.game.LotroCardBlueprintLibrary library;
    private static final Map<String, Map<String, Double>> CACHE = new java.util.concurrent.ConcurrentHashMap<>();

    private CardTextFeatures() {
    }

    public static synchronized void init(File cardsDir, com.gempukku.lotro.game.LotroCardBlueprintLibrary lib) {
        if (defs != null)
            return;
        library = lib;
        Map<String, JsonObject> m = new HashMap<>();
        Deque<File> todo = new ArrayDeque<>(List.of(cardsDir));
        while (!todo.isEmpty()) {
            File f = todo.pop();
            if (f.isDirectory()) {
                File[] kids = f.listFiles();
                if (kids != null) {
                    Arrays.sort(kids); // official before unofficial: putIfAbsent keeps the first found
                    for (int i = kids.length - 1; i >= 0; i--)
                        todo.push(kids[i]);
                }
            } else if (f.getName().endsWith(".hjson") || f.getName().endsWith(".json")) {
                try (Reader r = new FileReader(f, StandardCharsets.UTF_8)) {
                    JsonValue v = JsonValue.readHjson(r);
                    if (!v.isObject())
                        continue;
                    for (JsonObject.Member card : v.asObject())
                        if (card.getValue().isObject() && card.getValue().asObject().get("type") != null)
                            m.putIfAbsent(card.getName(), card.getValue().asObject());
                } catch (IOException | RuntimeException ignored) {
                    // a file the parser does not take: its cards get zeros
                }
            }
        }
        defs = m;
    }

    /** The vocabulary file beside a model, or one built from every definition and written there. */
    public static List<String> vocabulary(Path file) throws IOException {
        if (Files.exists(file))
            return Files.readAllLines(file).stream().map(String::trim).filter(s -> !s.isEmpty()).toList();
        Map<String, Integer> df = new TreeMap<>();
        String[] range = SETS.split("-");
        int lo = Integer.parseInt(range[0]), hi = Integer.parseInt(range[range.length - 1]);
        for (Map.Entry<String, JsonObject> card : defs.entrySet()) {
            String set = card.getKey().replaceAll("_.*", "");
            if (!set.matches("\\d+") || Integer.parseInt(set) < lo || Integer.parseInt(set) > hi)
                continue;
            for (String t : tokens(card.getValue(), "play", null).keySet())
                df.merge(t, 1, Integer::sum);
        }
        List<String> vocab = df.entrySet().stream().filter(e -> e.getValue() >= MIN_CARDS).map(Map.Entry::getKey).toList();
        Files.write(file, vocab);
        return vocab;
    }

    /**
     * The option's vector over {@code vocab}: the tokens of the abilities {@code actionText} brings, in {@code phase}
     * (a prompt context name: "fellowship", "regroup"..., "none" for the skirmish's). Zeros for pass and for a card
     * with no definition.
     */
    public static double[] of(List<String> vocab, String blueprintId, String actionText, String phase) {
        double[] f = new double[vocab.size()];
        if (blueprintId == null || actionText == null || defs == null || blueprintId.equals(CardFeatures.PASS))
            return f;
        String kind = kind(actionText);
        String p = phase == null || phase.equals("none") ? "skirmish" : phase;
        Map<String, Double> bag = CACHE.computeIfAbsent(blueprintId + "|" + kind + "|" + p, k -> {
            JsonObject card = definition(blueprintId);
            return card == null ? Map.of() : tokens(card, kind, p);
        });
        for (int i = 0; i < f.length; i++)
            f[i] = bag.getOrDefault(vocab.get(i), 0.0);
        return f;
    }

    private static String kind(String actionText) {
        String t = actionText.toLowerCase();
        if (t.startsWith("use"))
            return "use";
        if (t.startsWith("optional trigger") || t.startsWith("required trigger"))
            return "trigger";
        return "play";
    }

    private static JsonObject definition(String blueprintId) {
        JsonObject card = defs.get(blueprintId);
        if (card == null && library != null) {
            String base = library.getBaseBlueprintId(blueprintId);
            if (base != null)
                card = defs.get(base);
        }
        if (card == null)
            card = defs.get(blueprintId.replaceAll("[*T]+$", ""));
        return card;
    }

    /** The tokens of a card's abilities that {@code kind} brings in {@code phase} (null: every phase). */
    static Map<String, Double> tokens(JsonObject card, String kind, String phase) {
        Map<String, Double> out = new HashMap<>();
        List<JsonObject> abilities = new ArrayList<>();
        JsonValue effects = card.get("effects");
        if (effects != null && effects.isArray()) {
            for (JsonValue e : effects.asArray())
                if (e.isObject())
                    abilities.add(e.asObject());
        } else if (effects != null && effects.isObject())
            abilities.add(effects.asObject());
        List<JsonObject> chosen = new ArrayList<>();
        for (JsonObject a : abilities) {
            String type = lower(str(a, "type"));
            if (type == null)
                continue;
            boolean activated = ACTIVATED.contains(type);
            switch (kind) {
                case "use" -> {
                    if (activated && (phase == null || phase.equalsIgnoreCase(str(a, "phase"))))
                        chosen.add(a);
                }
                case "trigger" -> {
                    if (type.contains("trigger"))
                        chosen.add(a);
                }
                default -> chosen.add(a);
            }
        }
        if (chosen.isEmpty() && kind.equals("use")) // an ability of another phase, or one the prompt does not name
            for (JsonObject a : abilities)
                if (ACTIVATED.contains(lower(str(a, "type"))))
                    chosen.add(a);
        if (kind.equals("play") && card.get("target") != null)
            filterWords(card.get("target"), "attach", out);
        for (JsonObject a : chosen) {
            String type = lower(str(a, "type"));
            String head = type.equals("trigger") ? (a.getBoolean("optional", false) ? "optionaltrigger" : "trigger") : type;
            add(out, "ab:" + head, 1);
            String abPhase = lower(str(a, "phase"));
            if (abPhase != null)
                add(out, "ab:" + head + "@" + abPhase, 1);
            if (a.get("limitPerTurn") != null || a.get("limitPerPhase") != null)
                add(out, "ab:limited", 1);
            for (JsonObject.Member m : a) {
                String key = m.getName();
                String role = switch (key) {
                    case "cost", "extraCost" -> "cost";
                    case "requires" -> "req";
                    case "trigger" -> "trig";
                    case "effect" -> "effect";
                    case "modifier" -> "static";
                    default -> null;
                };
                if (role != null)
                    walk(m.getValue(), role, out);
            }
        }
        return out;
    }

    private static void walk(JsonValue v, String role, Map<String, Double> out) {
        if (v == null)
            return;
        if (v.isArray()) {
            for (JsonValue x : v.asArray())
                walk(x, role, out);
            return;
        }
        if (!v.isObject())
            return;
        JsonObject o = v.asObject();
        String type = lower(str(o, "type"));
        if (type != null && !WRAPPERS.contains(type)) {
            add(out, role + ":" + type, 1);
            for (String k : List.of("amount", "count", "times")) {
                JsonValue a = o.get(k);
                if (a != null && a.isNumber())
                    add(out, role + ":" + type + "#", a.asDouble());
                else if (a != null && a.isString() && a.asString().matches("\\d+(-\\d+)?"))
                    add(out, role + ":" + type + "#", Double.parseDouble(a.asString().replaceAll("-.*", "")));
                else if (a != null)
                    add(out, role + ":" + type + "#var", 1);
            }
            String until = lower(str(o, "until"));
            if (until != null)
                add(out, role + ":until=" + until, 1);
        }
        for (JsonObject.Member m : o) {
            String key = m.getName();
            JsonValue value = m.getValue();
            if (FILTERS.contains(key) && value.isString())
                filterWords(value, role, out);
            else if (value.isObject() || value.isArray()) {
                String child = switch (key) {
                    case "requires", "check", "condition" -> "req";
                    case "modifier" -> role.equals("static") ? "static" : role + "mod";
                    case "cost" -> "cost";
                    default -> role;
                };
                walk(value, child, out);
            }
        }
    }

    /** "choose(culture(dwarven),companion,not(ringBearer))" -> role>culture, role>dwarven, role>companion, role>not... */
    private static void filterWords(JsonValue v, String role, Map<String, Double> out) {
        if (!v.isString())
            return;
        Set<String> seen = new HashSet<>();
        for (String w : v.asString().toLowerCase().split("[^a-z]+"))
            if (w.length() >= 3 && !NOT_WHO.contains(w) && seen.add(w))
                add(out, role + ">" + w, 1);
    }

    private static void add(Map<String, Double> out, String token, double amount) {
        out.merge(token, amount, Double::sum);
    }

    private static String str(JsonObject o, String key) {
        JsonValue v = o.get(key);
        return v != null && v.isString() ? v.asString() : null;
    }

    private static String lower(String s) {
        return s == null ? null : s.toLowerCase();
    }

    /**
     * The selection models given card text: -DcardText.selections=DiscardFromHandTrainer,... with the vocabulary from
     * -DanyPhase.cardText (training), or a "card-text-selections" file listing them beside a "card-text" (play).
     */
    private static final Set<String> TRAINING_SELECTIONS = new HashSet<>(Arrays.asList(
            System.getProperty("cardText.selections", "").split("[,\\s]+")));
    private static volatile List<String> trainingVocab;

    /** The vocabulary a selection model is being fit with, or null if it is fit without card text. */
    public static List<String> trainingSelectionText(Class<?> trainer) {
        String file = System.getProperty("anyPhase.cardText");
        if (file == null || !TRAINING_SELECTIONS.contains(trainer.getSimpleName()))
            return null;
        if (trainingVocab == null)
            synchronized (CardTextFeatures.class) {
                if (trainingVocab == null)
                    try {
                        trainingVocab = vocabulary(Path.of(file));
                    } catch (IOException e) {
                        throw new IllegalStateException("anyPhase.cardText " + file, e);
                    }
            }
        return trainingVocab;
    }

    /** A selection option's card vector with what the card brings when played appended; unchanged for null. */
    public static double[] withText(double[] card, List<String> vocab, String blueprintId) {
        if (vocab == null)
            return card;
        double[] text = of(vocab, blueprintId, "Play", "none");
        double[] out = Arrays.copyOf(card, card.length + text.length);
        System.arraycopy(text, 0, out, card.length, text.length);
        return out;
    }

    /** Prints the vocabulary's size and a few cards' tokens: {@code <cardsDir> <blueprint>|<action text>|<phase>...}. */
    public static void main(String[] args) throws IOException {
        init(new File(args[0]), null);
        Path tmp = Files.createTempFile("card-text", ".txt");
        Files.delete(tmp);
        List<String> vocab = vocabulary(tmp);
        System.out.println(defs.size() + " definitions, " + vocab.size() + " tokens on " + MIN_CARDS + "+ cards of sets " + SETS);
        for (int i = 1; i < args.length; i++) {
            String[] p = args[i].split("\\|");
            double[] f = of(vocab, p[0], p[1], p.length > 2 ? p[2] : "none");
            StringBuilder sb = new StringBuilder(args[i] + ":");
            for (int k = 0; k < f.length; k++)
                if (f[k] != 0)
                    sb.append(' ').append(vocab.get(k)).append(f[k] == 1 ? "" : "=" + f[k]);
            System.out.println(sb);
        }
        Files.deleteIfExists(tmp);
    }
}
