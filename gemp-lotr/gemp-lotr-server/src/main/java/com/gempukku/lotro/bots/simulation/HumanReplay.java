package com.gempukku.lotro.bots.simulation;

import com.alibaba.fastjson2.JSON;
import com.alibaba.fastjson2.JSONArray;
import com.alibaba.fastjson2.JSONObject;
import com.gempukku.lotro.bots.rl.fotrstarters.EffectProbe;
import com.gempukku.lotro.common.CardType;
import com.gempukku.lotro.game.DefaultUserFeedback;
import com.gempukku.lotro.game.LotroCardBlueprint;
import com.gempukku.lotro.game.LotroCardBlueprintLibrary;
import com.gempukku.lotro.game.LotroFormat;
import com.gempukku.lotro.game.PhysicalCard;
import com.gempukku.lotro.game.state.EventSerializer;
import com.gempukku.lotro.game.state.GameCommunicationChannel;
import com.gempukku.lotro.game.state.GameEvent;
import com.gempukku.lotro.game.state.GameState;
import com.gempukku.lotro.logic.decisions.AwaitingDecision;
import com.gempukku.lotro.logic.decisions.AwaitingDecisionType;
import com.gempukku.lotro.logic.decisions.DecisionResultInvalidException;
import com.gempukku.lotro.logic.timing.DefaultLotroGame;
import com.gempukku.lotro.logic.vo.LotroDeck;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NamedNodeMap;
import org.w3c.dom.NodeList;

import javax.xml.parsers.DocumentBuilderFactory;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * Re-creates recorded human games inside the engine: builds both decks so the engine hands out the card ids the 2020
 * server did, forces each deck shuffle to put the cards the players actually drew on top (GameState.DeckShuffler),
 * and answers every decision the way the human did. The answers are not in the recordings; they are found by trying
 * each plausible answer on a copy of the game (GameCopier) and keeping the one whose events -- cards placed, removed,
 * tokens, assignments, phases, twilight, the next questions -- are the recorded ones. A game is reproduced up to the
 * first point where no answer gives the recorded events, or the engine asks a question the human was not asked.
 * <pre>
 *   ./replay.sh HumanReplay --manifests ~/dev/gemp-replays/manifests-2020-03/fotr [--limit 50] [--jobs 2]
 *       [--steps out.jsonl] [--report report.tsv] [--verbose]
 * </pre>
 * Manifests come from scripts/replay_manifest.py. See BOT-NOTES.md, "Re-creating human games".
 */
public class HumanReplay {
    /**
     * Event types both the 2020 recorder and today's engine emit, compared one for one. Messages and stats are not, nor
     * CAC ("card affected by card", a highlight for the client): the 2020 engine also sent one for a card paying a
     * cost on itself.
     */
    static final Set<String> COMPARED = Set.of("PCIP", "RCFP", "MCIP", "AT", "RT", "AA", "RA", "SS", "ES", "GPC", "TP",
            "TC", "PP", "CA", "EP", "D");

    /** One replay event: its type code, attributes and (for decisions) parameters, in order. */
    record Ev(String type, Map<String, String> attrs, List<String[]> params) {
        String attr(String k) {
            return attrs.get(k);
        }

        List<String> param(String name) {
            List<String> out = new ArrayList<>();
            for (String[] p : params)
                if (p[0].equals(name))
                    out.add(p[1]);
            return out;
        }
    }

    // ---------------------------------------------------------------- manifests

    record DeckInfo(int idBase, List<String> sites, List<String> deck, String rb, String ring, List<String> fill) {
    }

    record Manifest(String name, String format, List<String> players, Map<String, Integer> bids, String first,
                    String winner, boolean conceded, Map<String, DeckInfo> decks, Map<String, List<Ev>> events,
                    Map<String, List<int[]>> departures) {
    }

    static Manifest load(Path file) throws IOException {
        JSONObject o = JSON.parseObject(Files.readString(file));
        List<String> players = o.getJSONArray("players").toJavaList(String.class);
        Map<String, Integer> bids = new HashMap<>();
        o.getJSONObject("bids").forEach((k, v) -> bids.put(k, ((Number) v).intValue()));
        Map<String, DeckInfo> decks = new HashMap<>();
        Map<String, List<Ev>> events = new HashMap<>();
        Map<String, List<int[]>> departures = new HashMap<>();
        for (String p : players) {
            JSONObject d = o.getJSONObject("decks").getJSONObject(p);
            decks.put(p, new DeckInfo(d.getIntValue("idBase"), nullableList(d.getJSONArray("sites")),
                    nullableList(d.getJSONArray("deck")), d.getString("rb"), d.getString("ring"),
                    d.getJSONArray("fill").toJavaList(String.class)));
            List<Ev> evs = new ArrayList<>();
            for (Object e : o.getJSONObject("events").getJSONArray(p)) {
                JSONArray a = (JSONArray) e;
                Map<String, String> attrs = new LinkedHashMap<>();
                a.getJSONObject(1).forEach((k, v) -> attrs.put(k, String.valueOf(v)));
                List<String[]> params = new ArrayList<>();
                for (Object pp : a.getJSONArray(2)) {
                    JSONArray x = (JSONArray) pp;
                    params.add(new String[]{x.getString(0), x.getString(1)});
                }
                evs.add(new Ev(a.getString(0), attrs, params));
            }
            events.put(p, evs);
            List<int[]> dep = new ArrayList<>();
            for (Object x : o.getJSONObject("departures").getJSONArray(p)) {
                JSONArray a = (JSONArray) x;
                dep.add(new int[]{a.getIntValue(0), a.getIntValue(1)});
            }
            departures.put(p, dep);
        }
        return new Manifest(file.getFileName().toString().replaceFirst("\\.json$", ""), o.getString("format"),
                players, bids, o.getString("first"), o.getString("winner"),
                Boolean.TRUE.equals(o.getBoolean("conceded")), decks, events, departures);
    }

    private static List<String> nullableList(JSONArray a) {
        List<String> out = new ArrayList<>();
        for (int i = 0; i < a.size(); i++)
            out.add(a.getString(i));
        return out;
    }

    // ---------------------------------------------------------------- comparing events

    private static final Pattern NON_ASCII = Pattern.compile("[^\\x20-\\x7e]");

    /** Text as both sides can agree on it: the 2020 files mangled every non-ASCII character (unique dots, accents). */
    static String norm(String s) {
        if (s == null)
            return "";
        s = s.replaceAll("<[^>]*>", "");
        return NON_ASCII.matcher(s.replace("&#183;", "")).replaceAll("").replace("??", "").trim();
    }

    private static String sortedIds(String csv) {
        if (csv == null || csv.isEmpty())
            return "";
        String[] ids = csv.split(",");
        Arrays.sort(ids);
        return String.join(",", ids);
    }

    /**
     * Whether an event takes part in the comparison. Today's engine also shows each player their own adventure deck
     * as it is dealt (PCIP zone ADVENTURE_DECK) and both draw decks when the game ends (zone DECK), which the 2020
     * server did not.
     */
    static boolean comparable(Ev e) {
        return COMPARED.contains(e.type()) && !(e.type().equals("PCIP")
                && ("ADVENTURE_DECK".equals(e.attr("zone")) || "DECK".equals(e.attr("zone"))));
    }

    /** The event of a decision {@link Game#engineOnlyAnswer} would answer by itself. */
    static boolean engineOnlyEvent(Ev e) {
        if (!e.param("deckInspection").isEmpty())
            return true;
        String type = e.attrs().getOrDefault("decisionType", "");
        int min = parseOr(e.param("min"), 0), max = parseOr(e.param("max"), 1);
        return switch (type) {
            case "CARD_ACTION_CHOICE" -> e.param("actionId").isEmpty();
            case "ACTION_CHOICE" -> !e.param("actionId").isEmpty();
            case "ARBITRARY_CARDS" -> max >= 1 && e.param("selectable").stream().filter("true"::equals).count() == 1;
            case "CARD_SELECTION" -> !e.param("cardId").isEmpty() && min == e.param("cardId").size();
            default -> false;
        };
    }

    private static int parseOr(List<String> v, int dflt) {
        try {
            return v.isEmpty() ? dflt : Integer.parseInt(v.get(0));
        } catch (NumberFormatException e) {
            return dflt;
        }
    }

    /** An RCFP of several cards as one per card; any other event as it is. */
    static List<Ev> explode(Ev e) {
        String ids = e.attr("otherCardIds");
        if (!e.type().equals("RCFP") || ids == null || !ids.contains(","))
            return List.of(e);
        List<Ev> out = new ArrayList<>();
        for (String id : ids.split(",")) {
            Map<String, String> a = new LinkedHashMap<>(e.attrs());
            a.put("otherCardIds", id);
            out.add(new Ev(e.type(), a, e.params()));
        }
        return out;
    }

    /** What two events must share to count as the same: the attributes the 2020 recorder also wrote. */
    static String key(Ev e) {
        StringBuilder sb = new StringBuilder(e.type());
        switch (e.type()) {
            case "PCIP" -> {
                for (String k : List.of("blueprintId", "cardId", "zone", "targetCardId", "participantId", "index"))
                    sb.append('|').append(e.attrs().getOrDefault(k, ""));
            }
            case "MCIP" -> {
                for (String k : List.of("cardId", "zone", "targetCardId"))
                    sb.append('|').append(e.attrs().getOrDefault(k, ""));
            }
            case "RCFP" -> sb.append('|').append(sortedIds(e.attr("otherCardIds")));
            case "AT", "RT" -> {
                for (String k : List.of("cardId", "token", "count"))
                    sb.append('|').append(e.attrs().getOrDefault(k, ""));
            }
            case "AA", "SS", "CAC" -> sb.append('|').append(e.attrs().getOrDefault("cardId", ""))
                    .append('|').append(sortedIds(e.attr("otherCardIds")));
            case "RA", "CA", "EP" -> sb.append('|').append(e.attrs().getOrDefault("cardId", ""));
            case "GPC" -> sb.append('|').append(e.attrs().getOrDefault("phase", ""));
            case "TP" -> sb.append('|').append(e.attrs().getOrDefault("count", ""));
            case "TC" -> sb.append('|').append(e.attrs().getOrDefault("participantId", ""));
            case "PP" -> sb.append('|').append(e.attrs().getOrDefault("participantId", ""))
                    .append('|').append(e.attrs().getOrDefault("index", ""));
            case "D" -> sb.append('|').append(question(e));
            default -> {
            }
        }
        return sb.toString();
    }

    /** A decision's question: type, text and its options as a sorted multiset (arbitrary cards list in deck order). */
    /** A decision key's type and text, without its options. */
    static String questionHead(String key) {
        String[] p = key.split("\\|", 4);
        return p.length >= 3 ? p[0] + "|" + p[1] + "|" + p[2] : key;
    }

    /** Two decision keys reduced to what differs: type and text, then the options only one of them has. */
    static String[] paramDiff(String a, String b) {
        String[] pa = a.split("\\|", 4), pb = b.split("\\|", 4);
        List<String> la = new ArrayList<>(Arrays.asList(pa.length > 3 ? pa[3].split(";") : new String[0]));
        List<String> lb = new ArrayList<>(Arrays.asList(pb.length > 3 ? pb[3].split(";") : new String[0]));
        List<String> onlyA = new ArrayList<>(la), onlyB = new ArrayList<>(lb);
        for (String x : lb)
            onlyA.remove(x);
        for (String x : la)
            onlyB.remove(x);
        return new String[]{"D|" + pa[1] + "|" + (pa.length > 2 ? pa[2] : "") + " only " + onlyA,
                "D|" + pb[1] + "|" + (pb.length > 2 ? pb[2] : "") + " only " + onlyB};
    }

    /** Decision texts reworded since 2020, today's text to the recordings'. */
    static final Map<String, String> TEXT_ALIASES = Map.of(
            "Choose cards from deck", "Choose card from deck",
            "Choose a card to play", "Choose card from deck",
            "Choose cards from hand to discard", "Choose cards from hand",
            "Choose number of pipes you wish to spot", "Choose how many to spot");

    static String question(Ev e) {
        String type = e.attrs().getOrDefault("decisionType", "");
        List<String> ps = new ArrayList<>();
        for (String[] p : e.params()) {
            // the temp ids of arbitrary-card choices only number the list; the blueprints say what is in it
            if (p[0].equals("cardId") && p[1].startsWith("temp"))
                continue;
            // searches and selections may now stop short by choice (min 0) where in 2020 they had to take cards
            if (p[0].equals("min") && (type.equals("ARBITRARY_CARDS") || type.equals("CARD_SELECTION")))
                continue;
            // action texts name the card, whose name lost its accents to the 2020 files' encoding, and were
            // reworded ("Optional trigger from X" / "Use X", "Use X" / "Add a burden..."): the card ids, one per
            // action, say what is on offer
            if (p[0].equals("actionText"))
                continue;
            ps.add(p[0] + "=" + norm(p[1]));
        }
        Collections.sort(ps);
        return type + "|" + looseText(type, e.attr("text")) + "|" + String.join(";", ps);
    }

    /**
     * A question's text as far as it survived since 2020. Card selections, multiple choices, arbitrary-card choices
     * and numbers (but the bid) lose it altogether ("How many times do you wish to heal it" / "Choose how many times
     * to heal Elrond...", "Choose card from deck" / "...from discard" for the same list)
     * (their options say what they are; "...to prevent exhaust <card>" became "...to prevent exhausting Aragorn"):
     * what a selection is for was
     * reworded freely ("Choose characters to wound" / "Choose cards to wound") and their options and what follows
     * say which question it is. Card-action prompts keep only the part after the card name ("Wound Goblin
     * Scavenger(s) - Optional \"is about to\" responses"). The rest: letters only, lower case, without articles
     * ("skip Archery phase" / "skip the Archery phase").
     */
    static String looseText(String type, String raw) {
        String text = norm(raw);
        text = TEXT_ALIASES.getOrDefault(text, text);
        if (type.equals("ARBITRARY_CARDS") && text.contains("inspect the contents"))
            return "deck inspection"; // today's engine only, never the question a human answered
        if (type.equals("CARD_SELECTION") || type.equals("MULTIPLE_CHOICE") || type.equals("ARBITRARY_CARDS")
                || type.equals("INTEGER") && !text.contains("burdens to bid"))
            return "";
        if (type.equals("CARD_ACTION_CHOICE") && text.contains(" - "))
            text = text.substring(text.lastIndexOf(" - ") + 3);
        return (" " + text.toLowerCase().replaceAll("[^a-z0-9]+", " ") + " ")
                .replaceAll(" (the|a|an) ", " ").replaceAll(" (the|a|an) ", " ").trim();
    }

    private static final ThreadLocal<Document> DOC = ThreadLocal.withInitial(() -> {
        try {
            return DocumentBuilderFactory.newInstance().newDocumentBuilder().newDocument();
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
    });

    /** An engine event in the recordings' form, through the server's own serializer. */
    static Ev fromEngine(GameEvent ge) {
        Element el = (Element) new EventSerializer().serializeEvent(DOC.get(), ge);
        Map<String, String> attrs = new LinkedHashMap<>();
        NamedNodeMap nm = el.getAttributes();
        for (int i = 0; i < nm.getLength(); i++)
            attrs.put(nm.item(i).getNodeName(), nm.item(i).getNodeValue());
        String type = attrs.remove("type");
        List<String[]> params = new ArrayList<>();
        NodeList kids = el.getElementsByTagName("parameter");
        for (int i = 0; i < kids.getLength(); i++) {
            Element p = (Element) kids.item(i);
            params.add(new String[]{p.getAttribute("name"), p.getAttribute("value")});
        }
        return new Ev(type, attrs, params);
    }

    // ---------------------------------------------------------------- one game

    /** Why a game stopped being reproduced, and where. */
    record Outcome(String game, int recordedDecisions, int reproduced, int questionsMatched, String bucket,
                   String detail, String lastQuestion, int ambiguous, int engineOnly, int reproducedSoft,
                   int reproducedTrivial, int recordedTrivial) {
    }

    static final class Game {
        final Manifest m;
        final BotVsBot.Context ctx;
        final GameCopier copier;
        final StepSink sink;
        final boolean verbose;
        final Map<String, Integer> ptr = new HashMap<>();
        /** The recorded decision each player's pending engine decision was matched to. */
        final Map<String, Integer> pendingRec = new HashMap<>();
        final Map<String, Integer> idBase = new HashMap<>();
        DefaultLotroGame game;
        Map<String, GameCommunicationChannel> channels;
        int reproduced, questionsMatched;
        String lastQuestion;
        /** Which decisions Counterfactual tests: the same ones on every run of the same game. */
        final Random cfSampler;
        /** --values: the deciding player and the set-2 state at each card-action choice reproduced, for a value model. */
        final List<String> valueRows = new ArrayList<>();
        final List<String> valueDeciders = new ArrayList<>();

        Game(Manifest m, BotVsBot.Context ctx, GameCopier copier, StepSink sink, boolean verbose) {
            this.m = m;
            this.ctx = ctx;
            this.copier = copier;
            this.sink = sink;
            this.verbose = verbose;
            this.cfSampler = new Random(m.name().hashCode());
            for (String p : m.players())
                ptr.put(p, 0);
            explodeRemovals();
            markRedundant();
        }

        /** Recorded decisions with something to choose (see {@link HumanReplay#engineOnlyEvent}). */
        int recordedTrivial() {
            int n = 0;
            for (String p : m.players()) {
                List<Ev> evs = m.events().get(p);
                for (int i = 0; i < evs.size(); i++)
                    if (evs.get(i).type().equals("D") && answered(evs, i) && engineOnlyEvent(evs.get(i)))
                        n++;
            }
            return n;
        }

        int recordedDecisions() {
            int n = 0;
            for (String p : m.players()) {
                List<Ev> evs = m.events().get(p);
                for (int i = 0; i < evs.size(); i++)
                    if (evs.get(i).type().equals("D") && answered(evs, i))
                        n++;
            }
            return n;
        }

        /** A decision with nothing after it but messages was never answered: the game ended on it. */
        static boolean answered(List<Ev> evs, int i) {
            for (int j = i + 1; j < evs.size(); j++)
                if (COMPARED.contains(evs.get(j).type()))
                    return true;
            return false;
        }

        // ---- decks

        LotroDeck deck(String p, LotroCardBlueprintLibrary library) {
            DeckInfo d = m.decks().get(p);
            LotroDeck deck = new LotroDeck(p);
            deck.setRingBearer(d.rb());
            deck.setRing(d.ring());
            // sites: the ones played are known; the rest are never seen, so any site of the missing numbers will do
            Set<Integer> have = new HashSet<>();
            for (String s : d.sites())
                if (s != null)
                    have.add(siteNumber(s, library));
            Deque<Integer> missing = new ArrayDeque<>();
            for (int n = 1; n <= 9; n++)
                if (!have.contains(n))
                    missing.add(n);
            for (String s : d.sites())
                deck.addSite(s != null ? s : fillerSite(missing.isEmpty() ? 9 : missing.poll(), library));
            // draw deck: unseen ids get the cards that went into play straight from the deck or showed in a deck
            // search and no seen id explains (the manifest's fill), then an inert filler
            Deque<String> fill = new ArrayDeque<>(d.fill());
            for (String b : d.deck())
                deck.addCard(b != null ? b : fill.isEmpty() ? FILLER : fill.poll());
            return deck;
        }

        // ---- the shuffle hook: the cards that leave the deck next, in the order they left it in the recording

        void shuffle(String player, List<? extends PhysicalCard> deck) {
            int from = ptr.get(player);
            Map<Integer, PhysicalCard> byId = new HashMap<>();
            for (PhysicalCard c : deck)
                byId.put(c.getCardId(), c);
            Map<PhysicalCard, Integer> rank = new HashMap<>();
            for (int[] dep : m.departures().get(player)) {
                if (dep[0] < from)
                    continue;
                PhysicalCard c = byId.get(dep[1]);
                if (c != null && !rank.containsKey(c))
                    rank.put(c, rank.size());
            }
            int top = rank.size();
            deck.sort(Comparator.comparingInt(c -> rank.getOrDefault(c, top)));
        }

        Set<Integer> futureDepartures(String player) {
            Set<Integer> out = new HashSet<>();
            int from = ptr.get(player);
            for (int[] dep : m.departures().get(player))
                if (dep[0] >= from)
                    out.add(dep[1]);
            return out;
        }

        // ---- running

        DefaultLotroGame newGame(long seed) {
            LotroCardBlueprintLibrary library = ctx.library();
            LotroFormat format = format(ctx, m.format());
            Map<String, LotroDeck> decks = new LinkedHashMap<>();
            for (String p : m.players())
                decks.put(p, deck(p, library));
            DefaultUserFeedback feedback = new DefaultUserFeedback();
            DefaultLotroGame g = new DefaultLotroGame(format, decks, feedback, library);
            feedback.setGame(g);
            g.getGameState().setRandomSeed(seed);
            return g;
        }

        void attach(DefaultLotroGame g) {
            channels = new LinkedHashMap<>();
            LotroFormat format = g.getFormat();
            for (String p : m.players()) {
                GameCommunicationChannel ch = new GameCommunicationChannel(p, 0, false, format);
                g.addGameStateListener(p, ch);
                ch.consumeGameEvents(); // the board as sent to someone joining: not news
                channels.put(p, ch);
            }
            g.getGameState().setDeckShuffler((player, deck, random) -> shuffle(player, deck));
        }

        Map<String, List<Ev>> drain(Map<String, GameCommunicationChannel> chs) {
            Map<String, List<Ev>> out = new LinkedHashMap<>();
            chs.forEach((p, ch) -> {
                List<Ev> evs = new ArrayList<>();
                for (GameEvent ge : ch.consumeGameEvents()) {
                    for (Ev e : explode(fromEngine(ge)))
                        if (comparable(e))
                            evs.add(e);
                }
                out.put(p, evs);
            });
            return out;
        }

        /** How far the engine's new events follow each player's recording from where it was. */
        /** {@code soft}: recorded questions matched by a question with other options. */
        record Match(int matched, boolean full, Map<String, Integer> newPtr, Map<String, Integer> decisions,
                     Set<String> soft, Map<String, String> last, String expected, String got, String channel) {
        }

        /**
         * Compares the engine's new events with the recording, per player: the events since the last question must be
         * the next ones recorded, as a multiset -- the 2020 server sent some batches in another order (minions
         * discarded at the end of a turn, the cards of a draw). A question the engine asks that the recording has
         * with other options (a card offered then that is not now, or the reverse) counts, but is marked: see
         * {@link Match#soft}. A question only today's engine asks ({@link #engineOnlyAnswer}) takes no recorded event
         * if the recording has none for it.
         */
        Match compare(Map<String, List<Ev>> fresh) {
            int matched = 0;
            boolean full = true;
            Map<String, Integer> newPtr = new HashMap<>(ptr);
            Map<String, Integer> decisions = new HashMap<>();
            Set<String> soft = new HashSet<>();
            Map<String, String> newLast = new HashMap<>(engineLast);
            String expected = null, got = null, channel = null;
            for (String p : m.players()) {
                List<Ev> mine = new ArrayList<>();
                for (Ev e : fresh.get(p))
                    if (!repeats(e, newLast, p))
                        mine.add(e);
                Side withOnly = side(p, mine, false, false), withoutOnly = side(p, mine, true, false);
                Side best = withOnly.full() || !withoutOnly.full() && withOnly.ok() >= withoutOnly.ok()
                        ? withOnly : withoutOnly;
                if (!best.full()) {
                    // a question the old server asked and today's engine settles by itself -- one card to pick,
                    // how many Elves to spot -- taken as answered, if everything after it then matches
                    for (boolean dropEngine : new boolean[]{false, true}) {
                        Side s = side(p, mine, dropEngine, true);
                        if (s.full()) {
                            best = s;
                            recordedOnly++;
                            break;
                        }
                    }
                }
                matched += best.ok();
                decisions.putAll(best.decisions());
                soft.addAll(best.soft());
                if (!best.full() && full) {
                    full = false;
                    expected = best.missing().isEmpty() ? "(end of recording)" : String.join(" ; ", best.missing());
                    got = String.join(" ; ", best.extra());
                    channel = p;
                    if (best.missing().size() == 1 && best.extra().size() == 1 && best.missing().get(0).startsWith("D|")
                            && best.extra().get(0).startsWith("D|")) {
                        String[] d = paramDiff(best.missing().get(0), best.extra().get(0));
                        expected = d[0];
                        got = d[1];
                    }
                }
                newPtr.put(p, best.end());
                if (trace)
                    System.err.println("  [" + p + "] engine " + mine.stream().map(HumanReplay::key).toList()
                            + "\n  [" + p + "] recorded " + best.window().stream().map(w -> key(m.events().get(p).get(w)))
                            .toList());
            }
            return new Match(matched, full, newPtr, decisions, soft, newLast, expected, got, channel);
        }

        /** One player's events held against their recording. */
        record Side(int ok, boolean full, List<String> missing, List<String> extra, Map<String, Integer> decisions,
                    Set<String> soft, List<Integer> window, int end) {
        }

        /**
         * A recorded question today's engine may not ask: one selectable card ("Choose a card to play" with one
         * option), a number of characters to spot, or where to play Gollum from. The 2022 server asked these; today's
         * engine plays the card, spots as many as it can, and picks the zone itself.
         */
        static boolean recordedOnlyEvent(Ev e) {
            if (!e.type().equals("D"))
                return false;
            String type = e.attrs().getOrDefault("decisionType", ""), text = e.attrs().getOrDefault("text", "");
            return type.equals("ARBITRARY_CARDS") && e.param("selectable").stream().filter("true"::equals).count() == 1
                    || type.equals("INTEGER") && text.toLowerCase().contains("spot")
                    || type.equals("MULTIPLE_CHOICE") && e.param("results").stream().anyMatch(r -> r.startsWith("Play Gollum from"));
        }

        int recordedOnly;

        /** A recorded-only question with nothing like it among the engine's new events. */
        static boolean notAsked(Ev recorded, List<Ev> engine) {
            if (!recordedOnlyEvent(recorded))
                return false;
            String head = questionHead(key(recorded));
            return engine.stream().noneMatch(x -> x.type().equals("D") && questionHead(key(x)).equals(head));
        }

        Side side(String p, List<Ev> mine, boolean dropEngineOnly, boolean dropRecordedOnly) {
            List<Ev> rec = m.events().get(p);
            boolean[] skip = redundant.get(p);
            List<Ev> ev = new ArrayList<>(), flashes = new ArrayList<>();
            for (Ev e : mine)
                if (e.type().equals("CA"))
                    flashes.add(e);
                else if (!(dropEngineOnly && e.type().equals("D") && engineOnlyEvent(e)))
                    ev.add(e);
            int i = ptr.get(p);
            List<Integer> window = new ArrayList<>();
            int j = i;
            for (int n = 0; n < ev.size(); n++) {
                while (j < rec.size() && (!comparable(rec.get(j)) || skip[j] || rec.get(j).type().equals("CA")
                        || dropRecordedOnly && notAsked(rec.get(j), ev)))
                    j++;
                if (j >= rec.size())
                    break;
                window.add(j);
                j++;
            }
            // CA (a card flashed as used) is the 2020 engine's and today's alike for most cards, but not all: a
            // required trigger that only resets a per-turn limit (Ford of Bruinen) flashed then and does not now.
            // So it counts towards how well an answer fits, and a missing or extra one does not break the match.
            int flashOk = 0;
            int spanEnd = window.isEmpty() ? i : window.get(window.size() - 1);
            Map<String, Integer> recFlashes = new HashMap<>();
            for (int w = i; w < spanEnd && w < rec.size(); w++)
                if (rec.get(w).type().equals("CA"))
                    recFlashes.merge(key(rec.get(w)), 1, Integer::sum);
            for (Ev e : flashes) {
                Integer n = recFlashes.get(key(e));
                if (n != null && n > 0) {
                    recFlashes.put(key(e), n - 1);
                    flashOk++;
                }
            }
            // exact matches first, as a multiset
            Map<String, Deque<Integer>> want = new LinkedHashMap<>();
            for (int w : window)
                want.computeIfAbsent(key(rec.get(w)), k -> new ArrayDeque<>()).add(w);
            int ok = flashOk;
            List<Ev> extra = new ArrayList<>();
            Map<String, Integer> decisions = new HashMap<>();
            for (Ev e : ev) {
                Deque<Integer> q = want.get(key(e));
                if (q != null && !q.isEmpty()) {
                    int w = q.poll();
                    if (e.type().equals("D"))
                        decisions.put(p, w);
                    ok++;
                } else
                    extra.add(e);
            }
            // then a question asked here with other options than the human had
            Set<String> soft = new HashSet<>();
            for (Iterator<Ev> it = extra.iterator(); it.hasNext(); ) {
                Ev e = it.next();
                if (!e.type().equals("D") || !e.param("deckInspection").isEmpty())
                    continue;
                String head = questionHead(key(e));
                for (Deque<Integer> q : want.values()) {
                    Integer w = q.peek();
                    if (w != null && rec.get(w).type().equals("D") && questionHead(key(rec.get(w))).equals(head)) {
                        q.poll();
                        decisions.put(p, w);
                        soft.add(p + "#" + w);
                        ok++;
                        it.remove();
                        break;
                    }
                }
            }
            List<String> missing = new ArrayList<>();
            want.forEach((k, q) -> q.forEach(w -> missing.add(k)));
            boolean full = extra.isEmpty() && missing.isEmpty() && window.size() == ev.size();
            return new Side(ok, full, missing, extra.stream().map(HumanReplay::key).toList(), decisions, soft, window,
                    window.isEmpty() ? i : window.get(window.size() - 1) + 1);
        }

        boolean trace = TRACE;

        /**
         * A phase change to the phase already showing, or a twilight update to the count already showing: today's
         * engine and the 2020 one send different numbers of these, and they change nothing. {@code last} holds what
         * the stream last showed, per player and type.
         */
        static boolean repeats(Ev e, Map<String, String> last, String player) {
            String value = switch (e.type()) {
                case "GPC" -> e.attr("phase");
                case "TP" -> e.attr("count");
                default -> null;
            };
            if (value == null)
                return false;
            return value.equals(last.put(player + "|" + e.type(), value));
        }

        /** The engine's streams' last phase and twilight shown, as committed. */
        final Map<String, String> engineLast = new HashMap<>();
        /**
         * One RCFP per card, in the recordings (here) and the engine's events ({@link #drain}): the 2020 server and
         * today's batch the cards a single effect removes differently. Departure indices are moved along.
         */
        void explodeRemovals() {
            for (String p : m.players()) {
                List<Ev> rec = m.events().get(p);
                List<Ev> out = new ArrayList<>();
                int[] moved = new int[rec.size() + 1];
                for (int i = 0; i < rec.size(); i++) {
                    moved[i] = out.size();
                    out.addAll(explode(rec.get(i)));
                }
                moved[rec.size()] = out.size();
                rec.clear();
                rec.addAll(out);
                for (int[] dep : m.departures().get(p))
                    dep[0] = moved[dep[0]];
            }
        }

        /** Recorded events that repeat the phase or twilight already showing. */
        final Map<String, boolean[]> redundant = new HashMap<>();

        void markRedundant() {
            for (String p : m.players()) {
                List<Ev> rec = m.events().get(p);
                boolean[] skip = new boolean[rec.size()];
                Map<String, String> last = new HashMap<>();
                for (int i = 0; i < rec.size(); i++)
                    skip[i] = repeats(rec.get(i), last, p);
                redundant.put(p, skip);
            }
        }

        /** Recorded questions ("player#index") the engine asked with other options. */
        final Set<String> softQuestions = new HashSet<>();
        int reproducedSoft, reproducedTrivial;

        void commit(Match mt) {
            ptr.putAll(mt.newPtr());
            pendingRec.putAll(mt.decisions());
            softQuestions.addAll(mt.soft());
            engineLast.clear();
            engineLast.putAll(mt.last());
        }

        Outcome stop(String bucket, String detail) {
            return new Outcome(m.name(), recordedDecisions(), reproduced, questionsMatched, bucket, detail, lastQuestion,
                    ambiguous, engineOnly, reproducedSoft, reproducedTrivial, recordedTrivial());
        }

        Outcome run() {
            // seating: with tied bids the engine's seating shuffle picks who chooses; find a seed that picks the
            // player who did. Deck shuffles do not use the Random at all here.
            String chooser = null;
            Pattern chose = Pattern.compile("(.+) has chosen to go (first|second)");
            for (Ev e : m.events().get(m.players().get(0))) {
                if (e.type().equals("M")) {
                    Matcher mm = chose.matcher(e.attrs().getOrDefault("message", ""));
                    if (mm.matches()) {
                        chooser = mm.group(1);
                        break;
                    }
                }
            }
            // well-mixed seeds: java.util.Random's first draw is nearly the same for adjacent small seeds
            long seed = 0;
            for (int i = 1; i < 64; i++) {
                seed = new SplittableRandom(i).nextLong();
                DefaultLotroGame g = newGame(seed);
                g.startGame();
                if (seatChooser(g, chooser))
                    break;
            }
            if (verbose)
                System.err.println(m.name() + ": seat chooser " + chooser + ", seed " + seed);
            game = newGame(seed);
            attach(game);
            game.startGame();
            Match first = compare(drain(channels));
            commit(first);
            if (!first.full())
                return stop("setup differs", first.channel() + ": expected " + first.expected() + " got " + first.got());

            while (true) {
                if (game.isFinished())
                    return stop(remainingDecisions() == 0 ? "completed" : "engine ended the game early",
                            "winner " + game.getWinnerPlayerId());
                DefaultUserFeedback fb = GameCopier.feedback(game);
                String p = null;
                for (String q : m.players())
                    if (fb.getAwaitingDecision(q) != null) {
                        p = q;
                        break;
                    }
                if (p == null)
                    return stop("engine stalled", "no decision pending");
                AwaitingDecision decision = fb.getAwaitingDecision(p);
                Integer rec = pendingRec.get(p);
                if (rec == null) {
                    // a question only today's engine asks (a deck inspection) or one the human's auto-pass setting
                    // skipped (nothing to play): answered for them, not counted
                    if (!engineOnly(decision))
                        return stop("internal", "no recorded decision matched for " + p + ": " + BotGame.fingerprint(decision));
                    List<String> auto = decision.getDecisionType() == AwaitingDecisionType.ACTION_CHOICE
                            ? Arrays.asList(decision.getDecisionParameters().get("actionId"))
                            : List.of(engineOnlyAnswer(decision));
                    Trial t = tryAnswers(p, auto);
                    if (t.match == null)
                        return stop(t.error != null ? "engine error" : "no valid answer", t.error);
                    adopt(p, t);
                    if (!t.match.full())
                        return stop("events differ", t.match.channel() + ": expected " + t.match.expected()
                                + " got " + t.match.got() + " after engine-only " + BotGame.fingerprint(decision));
                    engineOnly++;
                    continue;
                }
                List<Ev> recEvents = m.events().get(p);
                Ev recD = recEvents.get(rec);
                lastQuestion = p + ": " + recD.attr("decisionType") + " " + norm(recD.attr("text"));
                questionsMatched++;
                if (!answered(recEvents, rec))
                    return stop("completed", "game ended on " + lastQuestion + (m.conceded() ? " (conceded)" : ""));

                // what the recordings show next: the asked player's events up to their next question, then the
                // other player's up to theirs (some of an answer's effects only reached the opponent's recording)
                List<Ev> after = new ArrayList<>();
                for (int j = rec + 1; j < recEvents.size() && !recEvents.get(j).type().equals("D"); j++)
                    after.add(recEvents.get(j));
                for (String q : m.players()) {
                    if (q.equals(p))
                        continue;
                    List<Ev> other = m.events().get(q);
                    after.add(new Ev("SEP", Map.of(), List.of())); // where one recording's run ends
                    for (int j = ptr.get(q); j < other.size() && !other.get(j).type().equals("D"); j++)
                        after.add(other.get(j));
                }
                List<String> cands = candidates(decision, after, p);
                double[] state = sink != null ? sink.state(game, decision, p) : null;
                if (VALUES != null && decision.getDecisionType() == AwaitingDecisionType.CARD_ACTION_CHOICE) {
                    double[] v = state != null ? state : valueState(game, decision, p);
                    if (v != null) {
                        valueDeciders.add(p);
                        valueRows.add(Arrays.stream(v).mapToObj(x -> x == (long) x ? String.valueOf((long) x)
                                : String.valueOf(x)).collect(Collectors.joining(",")));
                    }
                }
                Trial t = tryAnswers(p, cands);
                if (t.match == null)
                    return stop(t.error != null ? "engine error" : "no valid answer",
                            t.error != null ? t.error : lastQuestion + " candidates " + cands);
                if (t.match.full() && !t.ambiguous && !engineOnly(decision))
                    Counterfactual.consider(game, copier, ctx, m.name(), m.players(), p, decision, t.answer, cfSampler);
                if (t.match.full() && !t.ambiguous && !engineOnly(decision) && Counterfactual.ASSIGN_TEACH != null)
                    Counterfactual.teachAssignment(game, copier, ctx, m.players(), p, decision,
                            state != null ? state : TrainingSteps.stateOf(game, decision, p), t.answer, cfSampler);
                if (t.match.full() && !t.ambiguous && !engineOnly(decision) && Counterfactual.MOVE_TEACH != null)
                    Counterfactual.teachMove(game, copier, ctx, m.players(), p, decision,
                            state != null ? state : TrainingSteps.stateOf(game, decision, p), t.answer, cfSampler);
                if (t.match.full() && !t.ambiguous && !engineOnly(decision) && Counterfactual.DEATHS_TEACH != null)
                    Counterfactual.teachByDeaths(game, copier, ctx, m.players(), p, decision,
                            state != null ? state : TrainingSteps.stateOf(game, decision, p), cfSampler);
                // the board as the human saw it; not when another answer fit the recording as well
                if (t.match.full() && !t.ambiguous && sink != null && state != null)
                    if (ONLY_PLAYERS == null || ONLY_PLAYERS.contains(m.name() + "\t" + p))
                        sink.write(state, decision, t.answer, p, game);
                adopt(p, t);
                if (!t.match.full())
                    return stop("events differ", t.match.channel() + ": expected " + t.match.expected()
                            + " got " + t.match.got() + " after " + lastQuestion + " answered '" + t.answer + "'");
                reproduced++;
                if (engineOnly(decision))
                    reproducedTrivial++; // nothing to choose: an empty action list or a forced selection
                if (softQuestions.contains(p + "#" + rec))
                    reproducedSoft++;
                if (t.ambiguous)
                    ambiguous++;
                if (verbose)
                    System.err.println(m.name() + " " + reproduced + " " + lastQuestion + " -> '" + t.answer + "'"
                            + (t.ambiguous ? " (ambiguous)" : ""));
            }
        }

        int engineOnly, ambiguous;

        /** The best answer found for a decision: the copy it was played on and how its events matched. */
        static final class Trial {
            DefaultLotroGame game;
            Match match;
            String answer;
            Map<String, GameCommunicationChannel> channels;
            String error;
            /** Another answer matched the recording just as well: the events cannot tell them apart. */
            boolean ambiguous;
        }

        /**
         * Plays each answer on a copy of the game and keeps the one whose events follow the recording furthest:
         * all of them (up to the next question) if one does, then the most. Ties go to the earlier answer, which is
         * the one the recording's events pointed at.
         * <p>
         * An answer can be right and still miss because something random came out differently (a card discarded or
         * revealed at random from a hand). So if some answer got further into the recording than the best complete
         * one before it missed, it is tried again with the engine's Random reseeded, up to {@link #RESEEDS} times,
         * and kept if a reseed makes it complete.
         */
        Trial tryAnswers(String p, List<String> cands) {
            Trial best = new Trial();
            List<Trial> partial = new ArrayList<>();
            int tried = 0;
            for (String c : cands) {
                if (tried++ >= MAX_CANDIDATES)
                    break;
                Trial t = attempt(p, c, null, best);
                if (t == null)
                    continue;
                if (!t.match.full())
                    partial.add(t);
                consider(best, t);
            }
            // a full match that only fits by taking a question with other options (a card revealed at random is
            // not the one the human saw): see whether a reseed makes it exact
            if (best.match != null && best.match.full() && !best.match.soft().isEmpty()) {
                String answer = best.answer;
                for (int r = 1; r <= RESEEDS; r++) {
                    Trial again = attempt(p, answer, new SplittableRandom(r * 7919L).nextLong(), best);
                    if (again != null && again.match.full() && again.match.soft().isEmpty()) {
                        RESEEDED.incrementAndGet();
                        best.game = again.game;
                        best.match = again.match;
                        best.channels = again.channels;
                        return best;
                    }
                }
            }
            int bar = best.match == null ? -1 : best.match.full() ? best.match.matched() : -1;
            partial.sort(Comparator.comparingInt((Trial t) -> -t.match.matched()));
            for (Trial t : partial.subList(0, Math.min(3, partial.size()))) {
                if (t.match.matched() <= bar)
                    break;
                for (int r = 1; r <= RESEEDS; r++) {
                    Trial again = attempt(p, t.answer, new SplittableRandom(r * 7919L).nextLong(), best);
                    if (again != null && again.match.full()) {
                        RESEEDED.incrementAndGet();
                        consider(best, again);
                        return best;
                    }
                }
            }
            return best;
        }


        private void consider(Trial best, Trial t) {
            if (best.match == null || t.match.full() && !best.match.full()
                    || t.match.full() == best.match.full() && t.match.matched() > best.match.matched()) {
                best.game = t.game;
                best.match = t.match;
                best.answer = t.answer;
                best.channels = t.channels;
                best.ambiguous = false;
            } else if (t.match.full() && best.match.full() && t.match.matched() == best.match.matched())
                best.ambiguous = true;
        }

        /** One answer played on a copy of the game, with the Random reseeded first if {@code seed} is given. */
        private Trial attempt(String p, String c, Long seed, Trial errors) {
            DefaultLotroGame copy = copier.copy(game);
            if (seed != null)
                copy.getGameState().setRandomSeed(seed);
            Map<String, GameCommunicationChannel> saved = channels;
            attach(copy);
            Map<String, GameCommunicationChannel> chs = channels;
            channels = saved;
            DefaultUserFeedback cfb = GameCopier.feedback(copy);
            AwaitingDecision d = cfb.getAwaitingDecision(p);
            cfb.participantDecided(p);
            try {
                d.decisionMade(c);
            } catch (DecisionResultInvalidException e) {
                if (trace)
                    System.err.println(" try " + p + " '" + c + "' rejected: " + e.getMessage());
                return null;
            }
            try {
                copy.carryOutPendingActionsUntilDecisionNeeded();
            } catch (RuntimeException e) {
                errors.error = e.toString();
                return null;
            }
            if (trace)
                System.err.println(" try " + p + " '" + c + "'" + (seed != null ? " reseeded" : ""));
            Trial t = new Trial();
            t.game = copy;
            t.match = compare(drain(chs));
            t.answer = c;
            t.channels = chs;
            return t;
        }

        void adopt(String p, Trial t) {
            game = t.game;
            channels = t.channels;
            game.getGameState().setDeckShuffler((player, deck, random) -> shuffle(player, deck));
            pendingRec.remove(p);
            commit(t.match);
        }

        /**
         * The answer to a decision the 2020 recordings have no counterpart for, or null if it is not one: a deck
         * inspection (today's engine shows the deck before a search), a card action choice with nothing to play (the
         * human's auto-pass setting skipped it then), or a choice with one possible answer (a search that finds one
         * card, which the 2020 server took without asking because it had to take one), or the order of required
         * triggers (ACTION_CHOICE, which the 2020 engine settled itself). See {@link #compare}.
         */
        static String engineOnlyAnswer(AwaitingDecision d) {
            Map<String, String[]> ps = d.getDecisionParameters();
            if (ps.containsKey("deckInspection"))
                return "";
            switch (d.getDecisionType()) {
                case CARD_ACTION_CHOICE -> {
                    return ps.getOrDefault("actionId", new String[0]).length == 0 ? "" : null;
                }
                case ACTION_CHOICE -> {
                    // the order of two required triggers: the 2020 engine picked it; the answers are all tried
                    String[] ids = ps.getOrDefault("actionId", new String[0]);
                    return ids.length > 0 ? ids[0] : null;
                }
                case ARBITRARY_CARDS -> {
                    String[] ids = ps.getOrDefault("cardId", new String[0]);
                    String[] sel = ps.getOrDefault("selectable", new String[0]);
                    List<String> selectable = new ArrayList<>();
                    for (int i = 0; i < ids.length; i++)
                        if (i >= sel.length || "true".equals(sel[i]))
                            selectable.add(ids[i]);
                    return selectable.size() == 1 && intParam(ps, "max", 1) >= 1 ? selectable.get(0) : null;
                }
                case CARD_SELECTION -> {
                    String[] ids = ps.getOrDefault("cardId", new String[0]);
                    return ids.length > 0 && intParam(ps, "min", 0) == ids.length ? String.join(",", ids) : null;
                }
                default -> {
                    return null;
                }
            }
        }

        static boolean engineOnly(AwaitingDecision d) {
            return engineOnlyAnswer(d) != null;
        }

        int remainingDecisions() {
            int n = 0;
            for (String p : m.players()) {
                List<Ev> evs = m.events().get(p);
                for (int i = ptr.get(p); i < evs.size(); i++)
                    if (evs.get(i).type().equals("D") && answered(evs, i))
                        n++;
            }
            return n;
        }

        /** Answers the bids and reports whether the engine then asks {@code chooser} to pick a seat. */
        boolean seatChooser(DefaultLotroGame g, String chooser) {
            DefaultUserFeedback fb = GameCopier.feedback(g);
            for (String p : m.players()) {
                AwaitingDecision d = fb.getAwaitingDecision(p);
                if (d == null || d.getDecisionType() != AwaitingDecisionType.INTEGER)
                    return true; // not a bidding format; nothing to arrange
                fb.participantDecided(p);
                try {
                    d.decisionMade(String.valueOf(m.bids().get(p)));
                } catch (DecisionResultInvalidException e) {
                    return true;
                }
            }
            g.carryOutPendingActionsUntilDecisionNeeded();
            return chooser == null || fb.getAwaitingDecision(chooser) != null;
        }

        // ---- the answers worth trying, best guess first

        List<String> candidates(AwaitingDecision d, List<Ev> after, String p) {
            Map<String, String[]> params = d.getDecisionParameters();
            List<String> mentioned = new ArrayList<>(); // card ids in the recorded events after the decision
            List<String> blueprints = new ArrayList<>();
            List<String> messages = new ArrayList<>();
            List<Ev> assignments = new ArrayList<>();
            // runs of consecutive AA events (messages between them aside): one player's assignment each
            List<List<Ev>> assignmentRuns = new ArrayList<>();
            boolean inRun = false;
            for (Ev e : after) {
                if (e.type().equals("AA")) {
                    if (!inRun)
                        assignmentRuns.add(new ArrayList<>());
                    assignmentRuns.get(assignmentRuns.size() - 1).add(e);
                    inRun = true;
                } else if (!e.type().equals("M") && !e.type().equals("GS"))
                    inRun = false;
            }
            Pattern hint = Pattern.compile("value='([^']+)'");
            for (Ev e : after) {
                for (String k : List.of("cardId", "targetCardId"))
                    if (e.attr(k) != null)
                        mentioned.add(e.attr(k));
                if (e.attr("otherCardIds") != null)
                    mentioned.addAll(Arrays.asList(e.attr("otherCardIds").split(",")));
                if (e.attr("blueprintId") != null && !e.type().equals("D"))
                    blueprints.add(e.attr("blueprintId"));
                if (e.type().equals("M")) {
                    String msg = e.attrs().getOrDefault("message", "");
                    messages.add(msg);
                    Matcher mm = hint.matcher(msg);
                    while (mm.find())
                        blueprints.add(mm.group(1));
                }
                if (e.type().equals("AA"))
                    assignments.add(e);
            }
            LinkedHashSet<String> out = new LinkedHashSet<>();
            switch (d.getDecisionType()) {
                case INTEGER -> {
                    int min = intParam(params, "min", 0), max = intParam(params, "max", Integer.MAX_VALUE);
                    if (d.getText().contains("burdens to bid") && m.bids().containsKey(p))
                        out.add(String.valueOf(m.bids().get(p)));
                    Pattern num = Pattern.compile("\\b(\\d+)\\b");
                    for (String msg : messages) {
                        String plain = msg.replaceAll("value='[^']*'", "");
                        for (String name : m.players())
                            plain = plain.replace(name, "");
                        Matcher mm = num.matcher(plain);
                        while (mm.find()) {
                            long v = Long.parseLong(mm.group(1));
                            if (v >= min && v <= max)
                                out.add(String.valueOf(v));
                        }
                    }
                    for (long v = min; v <= Math.min(max, (long) min + 12); v++)
                        out.add(String.valueOf(v));
                }
                case MULTIPLE_CHOICE -> {
                    String[] results = params.getOrDefault("results", new String[0]);
                    for (int i = 0; i < results.length; i++)
                        out.add(String.valueOf(i));
                }
                case CARD_ACTION_CHOICE, ACTION_CHOICE -> {
                    String[] ids = params.getOrDefault("actionId", new String[0]);
                    String[] cards = params.getOrDefault("cardId", new String[0]);
                    for (String id : mentioned)
                        for (int i = 0; i < cards.length && i < ids.length; i++)
                            if (cards[i].equals(id))
                                out.add(ids[i]);
                    if (d.getDecisionType() == AwaitingDecisionType.CARD_ACTION_CHOICE)
                        out.add("");
                    out.addAll(Arrays.asList(ids));
                    if (d.getDecisionType() == AwaitingDecisionType.ACTION_CHOICE)
                        out.add("");
                }
                case CARD_SELECTION -> {
                    List<String> opts = Arrays.asList(params.getOrDefault("cardId", new String[0]));
                    int min = intParam(params, "min", 0), max = intParam(params, "max", opts.size());
                    List<String> hinted = new ArrayList<>();
                    for (String id : mentioned)
                        if (opts.contains(id) && !hinted.contains(id))
                            hinted.add(id);
                    for (int k = Math.min(max, hinted.size()); k >= Math.max(1, min); k--)
                        out.add(String.join(",", hinted.subList(0, k)));
                    if (min == 0)
                        out.add("");
                    if (min <= 1)
                        out.addAll(opts);
                    if (max >= 2 && opts.size() <= 10)
                        for (int i = 0; i < opts.size(); i++)
                            for (int j = i + 1; j < opts.size(); j++)
                                out.add(opts.get(i) + "," + opts.get(j));
                    if (min > 2 && opts.size() >= min)
                        out.add(String.join(",", opts.subList(0, min)));
                }
                case ARBITRARY_CARDS -> {
                    String[] temp = params.getOrDefault("cardId", new String[0]);
                    String[] bps = params.getOrDefault("blueprintId", new String[0]);
                    String[] sel = params.getOrDefault("selectable", new String[0]);
                    int min = intParam(params, "min", 0), max = intParam(params, "max", temp.length);
                    List<PhysicalCard> physical = physicalCards(d);
                    Set<Integer> future = futureDepartures(p);
                    // one option per blueprint: the copy that is not drawn later in the recording, if there is one
                    Map<String, List<Integer>> byBlueprint = new LinkedHashMap<>();
                    for (int i = 0; i < temp.length; i++)
                        if (i >= sel.length || "true".equals(sel[i]))
                            byBlueprint.computeIfAbsent(bps[i], k -> new ArrayList<>()).add(i);
                    byBlueprint.values().forEach(list -> list.sort(Comparator.comparingInt(i ->
                            physical != null && i < physical.size() && future.contains(physical.get(i).getCardId()) ? 1 : 0)));
                    // cards whose own id shows up next (a card taken from the discard pile keeps its id): those
                    // copies exactly
                    List<String> byId = new ArrayList<>();
                    if (physical != null)
                        for (String id : mentioned)
                            for (int i = 0; i < physical.size() && i < temp.length; i++)
                                if (String.valueOf(physical.get(i).getCardId()).equals(id)
                                        && (i >= sel.length || "true".equals(sel[i])) && !byId.contains(temp[i])
                                        && byId.size() < max)
                                    byId.add(temp[i]);
                    for (int k = byId.size(); k >= Math.max(1, min); k--)
                        out.add(String.join(",", byId.subList(0, k)));
                    List<String> hinted = new ArrayList<>();
                    Map<String, Integer> used = new HashMap<>();
                    for (String b : blueprints) {
                        List<Integer> idx = byBlueprint.get(b);
                        int u = used.getOrDefault(b, 0);
                        if (idx != null && u < idx.size() && hinted.size() < max) {
                            hinted.add(temp[idx.get(u)]);
                            used.put(b, u + 1);
                        }
                    }
                    for (int k = hinted.size(); k >= Math.max(1, min); k--)
                        out.add(String.join(",", hinted.subList(0, k)));
                    if (min == 0)
                        out.add("");
                    for (List<Integer> idx : byBlueprint.values())
                        out.add(temp[idx.get(0)]);
                    for (List<Integer> idx : byBlueprint.values()) // every other copy too
                        for (int i : idx)
                            out.add(temp[i]);
                }
                case ASSIGN_MINIONS -> {
                    // the AA events that follow, kept to this decision's characters and minions: each run of them
                    // (both recordings show an assignment, and the next player's follows), then all together
                    Set<String> fps = new HashSet<>(Arrays.asList(params.getOrDefault("freeCharacters", new String[0])));
                    Set<String> minions = new HashSet<>(Arrays.asList(params.getOrDefault("minions", new String[0])));
                    List<List<Ev>> sources = new ArrayList<>(assignmentRuns);
                    sources.add(assignments);
                    for (List<Ev> source : sources) {
                        Map<String, LinkedHashSet<String>> byFp = new LinkedHashMap<>();
                        Set<String> used = new HashSet<>();
                        for (Ev e : source) {
                            String fp = e.attr("cardId");
                            if (fp == null)
                                continue;
                            if (fp.startsWith("-"))
                                fp = fp.substring(1);
                            if (!fps.contains(fp))
                                continue;
                            for (String mn : e.attrs().getOrDefault("otherCardIds", "").split(","))
                                if (minions.contains(mn) && (byFp.getOrDefault(fp, new LinkedHashSet<>()).contains(mn)
                                        || used.add(mn)))
                                    byFp.computeIfAbsent(fp, k -> new LinkedHashSet<>()).add(mn);
                        }
                        if (!byFp.isEmpty())
                            out.add(byFp.entrySet().stream().map(en -> en.getKey() + " " + String.join(" ", en.getValue()))
                                    .collect(Collectors.joining(",")));
                    }
                    out.add("");
                }
            }
            return new ArrayList<>(out);
        }

        static int intParam(Map<String, String[]> params, String name, int dflt) {
            String[] v = params.get(name);
            if (v == null || v.length == 0)
                return dflt;
            try {
                return Integer.parseInt(v[0]);
            } catch (NumberFormatException e) {
                return dflt;
            }
        }
    }

    static final int MAX_CANDIDATES = 60;
    /** Reseeds of the engine's Random tried on an answer that missed only, it seems, by chance. */
    static final int RESEEDS = 24;
    static final java.util.concurrent.atomic.AtomicInteger RESEEDED = new java.util.concurrent.atomic.AtomicInteger();
    /** Print every comparison (--trace): the engine's events and the recorded ones they were held against. */
    static volatile boolean TRACE;

    @SuppressWarnings("unchecked")
    static List<PhysicalCard> physicalCards(AwaitingDecision d) {
        try {
            Class<?> c = d.getClass();
            while (c != null && !c.getSimpleName().equals("ArbitraryCardsSelectionDecision"))
                c = c.getSuperclass();
            if (c == null)
                return null;
            var f = c.getDeclaredField("_physicalCards");
            f.setAccessible(true);
            return new ArrayList<>((Collection<? extends PhysicalCard>) f.get(d));
        } catch (ReflectiveOperationException e) {
            return null;
        }
    }

    /** The format a recording names; names were capitalised since ("Fellowship block" is "Fellowship Block"). */
    static LotroFormat format(BotVsBot.Context ctx, String name) {
        for (LotroFormat f : ctx.formats().getAllFormats().values())
            if (f.getName().equalsIgnoreCase(name))
                return f;
        throw new IllegalArgumentException("no format named " + name);
    }

    /** A card that does nothing from the draw deck, for ids the recording never showed. */
    static final String FILLER = "1_2";

    private static final Map<Integer, String> FILLER_SITES = new ConcurrentHashMap<>();

    static int siteNumber(String blueprint, LotroCardBlueprintLibrary library) {
        try {
            LotroCardBlueprint b = library.getLotroCardBlueprint(blueprint);
            return b.getCardType() == CardType.SITE ? b.getSiteNumber() : -1;
        } catch (Exception e) {
            return -1;
        }
    }

    static String fillerSite(int number, LotroCardBlueprintLibrary library) {
        return FILLER_SITES.computeIfAbsent(number, n -> {
            for (int set = 1; set <= 3; set++)
                for (int i = 1; i <= 370; i++) {
                    String id = set + "_" + i;
                    if (siteNumber(id, library) == n)
                        return id;
                }
            throw new IllegalStateException("no site " + n);
        });
    }

    // ---------------------------------------------------------------- training data

    /** Writes (state, human answer) pairs for the decisions reproduced, in BotTeach's teacher-label form. */
    interface StepSink {
        double[] state(DefaultLotroGame game, AwaitingDecision decision, String player);

        void write(double[] state, AwaitingDecision decision, String answer, String player, DefaultLotroGame before);
    }

    // ---------------------------------------------------------------- positions and who won

    /**
     * --values: where positions go, labelled with the recorded winner, for {@code BotProbe --fit}. Value models fit on
     * bot games learn the race backwards (bots that move on die), so this asks how positions turned out for people.
     */
    static volatile PrintWriter VALUES;

    /**
     * --steps-players &lt;file&gt;: write training steps only for these "game&lt;TAB&gt;player" pairs (the game id is the
     * manifest's file name without ".json") -- one style of player's decisions, for a bot that plays that style.
     */
    static volatile java.util.Set<String> ONLY_PLAYERS;

    static double[] valueState(DefaultLotroGame game, AwaitingDecision decision, String player) {
        int previous = com.gempukku.lotro.bots.rl.fotrstarters.CardFeatures.version();
        try {
            com.gempukku.lotro.bots.rl.fotrstarters.CardFeatures.setVersion(
                    com.gempukku.lotro.bots.rl.fotrstarters.GeneralStateFeatures.VERSION);
            return new com.gempukku.lotro.bots.rl.fotrstarters.GeneralStateFeatures()
                    .extractFeatures(game.getGameState(), decision, player);
        } catch (RuntimeException e) {
            return null;
        } finally {
            com.gempukku.lotro.bots.rl.fotrstarters.CardFeatures.setVersion(previous);
        }
    }

    /** A game's positions, once it is known who won it; games with no recorded winner are left out. */
    static void writeValues(Game g) {
        if (VALUES == null || g.m.winner() == null || !g.m.players().contains(g.m.winner()))
            return;
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < g.valueRows.size(); i++)
            sb.append(g.valueDeciders.get(i).equals(g.m.winner()) ? '1' : '0').append(',').append(g.valueRows.get(i))
                    .append('\n');
        synchronized (HumanReplay.class) {
            VALUES.print(sb);
        }
    }

    // ---------------------------------------------------------------- main

    public static void main(String[] args) throws Exception {
        Path manifests = null, report = null, steps = null;
        int limit = Integer.MAX_VALUE, jobs = 2;
        boolean verbose = false;
        String only = null, probeBot = null;
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--manifests" -> manifests = Path.of(args[++i]);
                case "--limit" -> limit = Integer.parseInt(args[++i]);
                case "--jobs" -> jobs = Integer.parseInt(args[++i]);
                case "--report" -> report = Path.of(args[++i]);
                case "--steps" -> steps = Path.of(args[++i]);
                case "--counterfactual" -> Counterfactual.KIND = args[++i];
                case "--cf-out" -> Counterfactual.OUT = new PrintWriter(Files.newBufferedWriter(Path.of(args[++i]), StandardCharsets.UTF_8));
                case "--cf-rate" -> Counterfactual.RATE = Double.parseDouble(args[++i]);
                case "--cf-rollouts" -> Counterfactual.ROLLOUTS = Integer.parseInt(args[++i]);
                case "--cf-only" -> Counterfactual.ONLY = java.util.regex.Pattern.compile(args[++i]);
                case "--cf-offer" -> Counterfactual.OFFER = java.util.regex.Pattern.compile(args[++i]);
                case "--cf-phases" -> Counterfactual.PHASES = java.util.Set.of(args[++i].toUpperCase().split(","));
                case "--cf-deaths" -> Counterfactual.DEATHS = true;
                case "--deaths-teach" -> Counterfactual.DEATHS_TEACH = new PrintWriter(Files.newBufferedWriter(Path.of(args[++i]), StandardCharsets.UTF_8));
                case "--deaths-rollouts" -> Counterfactual.DEATHS_ROLLOUTS = Integer.parseInt(args[++i]);
                case "--move-teach" -> {
                    Counterfactual.MOVE_TEACH = new PrintWriter(Files.newBufferedWriter(Path.of(args[++i]), StandardCharsets.UTF_8));
                    Counterfactual.MOVE_LOG = new PrintWriter(Files.newBufferedWriter(Path.of(args[i] + ".tsv"), StandardCharsets.UTF_8));
                }
                case "--move-value" -> Counterfactual.loadMoveValue(Path.of(args[++i]));
                case "--cf-opponents" -> Counterfactual.OPPONENTS = List.of(args[++i].split(","));
                case "--deaths-value" -> {
                    Counterfactual.loadMoveValue(Path.of(args[++i]));
                    Counterfactual.DEATHS_BY_VALUE = true;
                }
                case "--deaths-prompts" -> Counterfactual.DEATHS_PROMPTS = java.util.regex.Pattern.compile(args[++i]);
                case "--assign-teach" -> Counterfactual.ASSIGN_TEACH = new PrintWriter(Files.newBufferedWriter(Path.of(args[++i]), StandardCharsets.UTF_8));
                case "--move-rollouts" -> Counterfactual.MOVE_ROLLOUTS = Integer.parseInt(args[++i]);
                case "--deaths-sides" -> Counterfactual.DEATHS_SIDES = java.util.Set.of(args[++i].toLowerCase().split(","));
                case "--steps-only" -> TrainingSteps.ONLY = java.util.regex.Pattern.compile(args[++i]);
                case "--opp-block" -> TrainingSteps.OPP = true;
                case "--steps-players" -> ONLY_PLAYERS = new java.util.HashSet<>(Files.readAllLines(Path.of(args[++i])));
                case "--probe-bot" -> probeBot = args[++i];
                case "--probe-value" -> EffectProbe.VALUE = BotVsBot.readModel(Path.of(args[++i]).resolve("value.model"));
                case "--values" -> VALUES = new PrintWriter(Files.newBufferedWriter(Path.of(args[++i]), StandardCharsets.UTF_8));
                case "--verbose" -> verbose = true;
                case "--trace" -> TRACE = true;
                case "--only" -> only = args[++i];
                default -> BotVsBot.die("usage: HumanReplay --manifests <dir> [--limit N] [--jobs 2] [--report f.tsv] "
                        + "[--steps f.jsonl [--steps-only <prompt regex>] [--probe-bot <bot>]] [--values f.csv] "
                        + "[--only <name substring>] [--verbose]");
            }
        }
        PrintStream out = System.out;
        System.setOut(new PrintStream(OutputStream.nullOutputStream()));
        BotVsBot.Context ctx = BotVsBot.context(Path.of("gemp-lotr-cards/src/main/resources"), Path.of("bot-models"));
        GameCopier copier = new GameCopier(ctx.library(), ctx.formats());
        if (probeBot != null) {
            String kind = probeBot;
            EffectProbe.FORK = copier::copy;
            TrainingSteps.PROBE_BOTS = p -> BotVsBot.bot(kind, p, ctx);
        }
        List<Path> files;
        try (var s = Files.list(manifests)) {
            String o = only;
            files = s.filter(f -> f.toString().endsWith(".json")).filter(f -> o == null || f.toString().contains(o))
                    .sorted().limit(limit).toList();
        }
        StepSink sink = steps != null ? TrainingSteps.open(steps) : null;
        ExecutorService pool = Executors.newFixedThreadPool(jobs, r -> {
            Thread t = new Thread(r);
            t.setDaemon(true);
            return t;
        });
        boolean v = verbose;
        List<Future<Outcome>> futures = new ArrayList<>();
        for (Path f : files)
            futures.add(pool.submit(() -> {
                Manifest m = load(f);
                try {
                    Game g = new Game(m, ctx, copier, sink, v);
                    Outcome o = g.run();
                    writeValues(g);
                    return o;
                } catch (RuntimeException e) {
                    StringWriter sw = new StringWriter();
                    e.printStackTrace(new PrintWriter(sw));
                    return new Outcome(m.name(), 0, 0, 0, "harness error",
                            sw.toString().lines().limit(4).map(String::trim).collect(Collectors.joining(" < ")), null, 0, 0, 0, 0, 0);
                }
            }));
        List<Outcome> outcomes = new ArrayList<>();
        for (int i = 0; i < futures.size(); i++) {
            Outcome o;
            try {
                o = futures.get(i).get(5, TimeUnit.MINUTES);
            } catch (TimeoutException e) {
                futures.get(i).cancel(true);
                o = new Outcome(files.get(i).getFileName().toString(), 0, 0, 0, "engine stuck", "over 5 minutes", null, 0, 0, 0, 0, 0);
            }
            outcomes.add(o);
            if (verbose)
                out.println(o);
        }
        if (VALUES != null)
            VALUES.close();
        if (Counterfactual.ASSIGN_TEACH != null) {
            out.printf("assignment teacher: %d Free Peoples assignments labelled%n", Counterfactual.assignsTaught.get());
            Counterfactual.ASSIGN_TEACH.close();
        }
        if (Counterfactual.MOVE_TEACH != null) {
            out.printf("move teacher: %d move decisions scored, %d labelled%n", Counterfactual.movesScored.get(),
                    Counterfactual.movesTaught.get());
            Counterfactual.MOVE_TEACH.close();
            Counterfactual.MOVE_LOG.close();
        }
        if (Counterfactual.DEATHS_TEACH != null) {
            out.printf("deaths teacher: %d skirmish decisions labelled (%s)%n", Counterfactual.deathsTaught.get(), String.join(",", Counterfactual.DEATHS_SIDES));
            Counterfactual.DEATHS_TEACH.close();
        }
        if (Counterfactual.KIND != null) {
            out.printf("counterfactual: %d decisions asked, bot agreed with the human on %d, %d played out%n",
                    Counterfactual.asked.get(), Counterfactual.agreed.get(), Counterfactual.compared.get());
            if (Counterfactual.OUT != null)
                Counterfactual.OUT.close();
        }
        if (sink instanceof TrainingSteps ts) {
            ts.close();
            out.printf("wrote %d training rows for %d reproduced decisions to %s%n", ts.steps.get(),
                    ts.decisions.get(), steps);
        }
        summarise(outcomes, out);
        if (report != null) {
            try (PrintWriter w = new PrintWriter(Files.newBufferedWriter(report, StandardCharsets.UTF_8))) {
                w.println("game\trecorded\treproduced\tquestions\tambiguous\ttrivial\trecordedTrivial\tbucket\tdetail\tlast");
                for (Outcome o : outcomes)
                    w.println(String.join("\t", o.game(), String.valueOf(o.recordedDecisions()),
                            String.valueOf(o.reproduced()), String.valueOf(o.questionsMatched()),
                            String.valueOf(o.ambiguous()), String.valueOf(o.reproducedTrivial()),
                            String.valueOf(o.recordedTrivial()), o.bucket(),
                            String.valueOf(o.detail()).replace('\t', ' '), String.valueOf(o.lastQuestion())));
            }
        }
        System.exit(0);
    }

    static void summarise(List<Outcome> outcomes, PrintStream out) {
        long recorded = outcomes.stream().mapToLong(Outcome::recordedDecisions).sum();
        long reproduced = outcomes.stream().mapToLong(Outcome::reproduced).sum();
        long complete = outcomes.stream().filter(o -> o.bucket().equals("completed")).count();
        out.printf("%d games, %d recorded human decisions: %d reproduced (%.1f%%); %d games reproduced to the end%n",
                outcomes.size(), recorded, reproduced, 100.0 * reproduced / Math.max(1, recorded), complete);
        long recTrivial = outcomes.stream().mapToLong(Outcome::recordedTrivial).sum();
        long repTrivial = outcomes.stream().mapToLong(Outcome::reproducedTrivial).sum();
        out.printf("  of the reproduced, %d had nothing to choose (no action to play, or a forced selection); with "
                        + "something to choose: %d of %d (%.1f%%)%n", repTrivial, reproduced - repTrivial,
                recorded - recTrivial, 100.0 * (reproduced - repTrivial) / Math.max(1, recorded - recTrivial));
        out.printf("  of the reproduced, %d were asked with other options than the human had (a card offered then and "
                + "not now, or the reverse)%n", outcomes.stream().mapToLong(Outcome::reproducedSoft).sum());
        out.printf("  of the reproduced, %d had another answer matching the recording as well (ambiguous); %d questions "
                        + "only today's engine asks were answered for the players%n",
                outcomes.stream().mapToLong(Outcome::ambiguous).sum(), outcomes.stream().mapToLong(Outcome::engineOnly).sum());
        out.printf("  %d answers matched only once the engine's Random was reseeded (a random discard or reveal)%n",
                RESEEDED.get());
        Map<String, Integer> buckets = new TreeMap<>();
        for (Outcome o : outcomes)
            buckets.merge(o.bucket(), 1, Integer::sum);
        buckets.forEach((b, n) -> out.printf("  %5d  %s%n", n, b));
        int[] edges = {0, 1, 10, 25, 50, 100, 200, 400, Integer.MAX_VALUE};
        out.println("  decisions reproduced before the first divergence, games:");
        for (int i = 0; i + 1 < edges.length; i++) {
            int lo = edges[i], hi = edges[i + 1];
            long n = outcomes.stream().filter(o -> o.reproduced() >= lo && o.reproduced() < hi).count();
            out.printf("    %4d-%-5s %d%n", lo, hi == Integer.MAX_VALUE ? "" : String.valueOf(hi - 1), n);
        }
    }
}
