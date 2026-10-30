package com.gempukku.lotro.bots.simulation;

import com.alibaba.fastjson2.JSON;
import com.alibaba.fastjson2.JSONArray;
import com.alibaba.fastjson2.JSONObject;
import com.gempukku.lotro.bots.rl.fotrstarters.CardFeatures;
import com.gempukku.lotro.bots.rl.fotrstarters.CardTextFeatures;
import com.gempukku.lotro.bots.rl.fotrstarters.models.assignment.AssignmentProbe;
import com.gempukku.lotro.bots.rl.fotrstarters.models.cardselection.ReconcileChoiceTrainer;
import com.gempukku.lotro.common.CardType;
import com.gempukku.lotro.common.Keyword;
import com.gempukku.lotro.common.Zone;
import com.gempukku.lotro.game.PhysicalCard;
import com.gempukku.lotro.game.state.GameState;
import com.gempukku.lotro.game.state.LotroGame;
import com.gempukku.lotro.logic.decisions.AwaitingDecision;
import com.gempukku.lotro.logic.decisions.AwaitingDecisionType;

import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

/**
 * A decision as the whole board, for the network in nn/: every card that matters to it as an item -- its card, whose it
 * is, its zone, what it is attached to, its modified strength, vitality and resistance, wounds, its keywords as the
 * game has them now (a damage +1 from a sword, fierce from a condition), whether it is active, in the skirmish or
 * assigned, the Ring-bearer, and for a card in hand what it costs to play now -- the options with the item each comes
 * from, and the one chosen. The set-2 state goes along for the summary numbers. The trees see the board as ~75 totals
 * and fit their own training months barely better than held-out ones (Fellowship plays 67.4% against 66.1%): the limit
 * is what the inputs can say, not capacity.
 * <p>
 * Items, in order: the cards the options name (first, so a size cap never cuts the card an option plays -- with them
 * after the hand, 1,988 of 5,765 options on July's boards over 48 cards lost theirs, 586 of them the chosen one), the
 * rest in play, the hand, then the player's own adventure deck (a player knows their sites; never the opponent's
 * unplayed ones).
 * <p>
 * Kinds of decision ("kind"):
 * <ul>
 * <li>action -- a card-action prompt of any phase ("Play Skirmish action or Pass", responses): options as before;</li>
 * <li>assign -- assigning minions: the characters and minions offered (item indices) and, per minion, the character
 * it went to (-1: left unassigned);</li>
 * <li>move -- "Do you want to make another move?": chosen 0 yes, 1 no;</li>
 * <li>reconcile -- discard one card or keep everything: options keep, then the cards; chosen 0 keep.</li>
 * </ul>
 * Piles ("bags", card blueprints): both discard and dead piles; "oppSeen", every card of the opponent's seen in a
 * public zone this game (kept after it goes back to hand or deck: what a player remembers of a deck); "myDeck", what
 * is left in the player's own draw deck, sorted (its contents, never its order), with "myDeckUnknown" cards of it not
 * known. For people's games the decklist comes from {@code --decklists} (built from the same deck's other games, so it
 * never tells which cards this game is about to draw); for bot games it is the deck itself.
 * <p>
 * {@code HumanReplay --boards <file>} writes one JSON line per reproduced decision of those kinds;
 * {@code BoardDump cards <out.tsv>} writes the card table the items refer to (set-2 card numbers and card text).
 */
final class BoardDump {
    static volatile PrintWriter OUT;
    /** "game\tplayer" -> the decklist's known blueprints (HumanReplay --decklists). Null: use the deck itself. */
    static volatile Map<String, List<String>> DECKS;
    /**
     * Bot games (BotTeach): the remainder is the deck itself. Never in replays -- there the engine's deck is built from
     * the cards the game went on to show, so its contents are the future draws.
     */
    static volatile boolean TRUE_DECK = false;

    /** The keywords recorded per card, as counts (damage +2 is 2). */
    static final List<Keyword> KEYWORDS = List.of(Keyword.DAMAGE, Keyword.FIERCE, Keyword.ARCHER, Keyword.AMBUSH,
            Keyword.ROAMING, Keyword.ENDURING, Keyword.LURKER, Keyword.HUNTER, Keyword.DEFENDER, Keyword.TOIL,
            Keyword.MUSTER, Keyword.UNHASTY, Keyword.RING_BOUND, Keyword.RANGER, Keyword.TRACKER, Keyword.VALIANT);
    static final String MOVE = "Do you want to make another move?";

    private BoardDump() {
    }

    static void write(LotroGame game, AwaitingDecision d, String player, String answer, double[] state) {
        PrintWriter out = OUT;
        if (out == null)
            return;
        try {
            GameState gs = game.getGameState();
            observe(game);
            String kind = kind(d);
            if (kind == null)
                return;
            var mq = game.getModifiersQuerying();
            List<PhysicalCard> items = new ArrayList<>();
            Map<Integer, Integer> index = new HashMap<>();
            Map<String, String[]> params = d.getDecisionParameters();
            for (String key : List.of("cardId", "freeCharacters", "minions"))
                for (String cid : params.getOrDefault(key, new String[0]))
                    try {
                        add(items, index, gs.findCardById(Integer.parseInt(cid)));
                    } catch (RuntimeException ignored) {
                    }
            for (PhysicalCard c : gs.getInPlay())
                add(items, index, c);
            for (PhysicalCard c : gs.getHand(player))
                add(items, index, c);
            for (PhysicalCard c : gs.getAdventureDeck(player))
                add(items, index, c);
            Set<Integer> inSkirmish = new HashSet<>(), assigned = new HashSet<>();
            var sk = gs.getSkirmish();
            if (sk != null) {
                if (sk.getFellowshipCharacter() != null)
                    inSkirmish.add(sk.getFellowshipCharacter().getCardId());
                for (PhysicalCard m : sk.getShadowCharacters())
                    inSkirmish.add(m.getCardId());
            }
            for (var a : gs.getAssignments()) {
                assigned.add(a.getFellowshipCharacter().getCardId());
                for (PhysicalCard m : a.getShadowCharacters())
                    assigned.add(m.getCardId());
            }
            Set<Integer> bearers = new HashSet<>();
            for (String p : gs.getPlayerNames()) {
                PhysicalCard rb = gs.getRingBearer(p);
                if (rb != null)
                    bearers.add(rb.getCardId());
            }
            JSONArray cards = new JSONArray();
            for (PhysicalCard c : items) {
                JSONObject o = new JSONObject();
                o.put("bp", c.getBlueprintId());
                o.put("own", c.getOwner().equals(player) ? 1 : 0);
                o.put("zone", String.valueOf(c.getZone()));
                PhysicalCard on = c.getAttachedTo() != null ? c.getAttachedTo() : c.getStackedOn();
                o.put("att", on != null && index.containsKey(on.getCardId()) ? index.get(on.getCardId()) : -1);
                CardType type = c.getBlueprint().getCardType();
                boolean character = type == CardType.COMPANION || type == CardType.ALLY || type == CardType.MINION;
                boolean inPlay = c.getZone() != null && c.getZone().isInPlay();
                boolean live = character && inPlay;
                o.put("str", live ? mq.getStrength(game, c) : 0);
                o.put("vit", live ? mq.getVitality(game, c) : 0);
                o.put("res", live && type == CardType.COMPANION ? mq.getResistance(game, c) : 0);
                o.put("wnd", inPlay ? gs.getWounds(c) : 0);
                o.put("act", inPlay && gs.isCardInPlayActive(c) ? 1 : 0);
                o.put("skir", inSkirmish.contains(c.getCardId()) ? 1 : 0);
                o.put("asg", assigned.contains(c.getCardId()) ? 1 : 0);
                o.put("rb", bearers.contains(c.getCardId()) ? 1 : 0);
                if (inPlay) {
                    JSONArray kw = new JSONArray();
                    for (int k = 0; k < KEYWORDS.size(); k++) {
                        int n = mq.getKeywordCount(game, c, KEYWORDS.get(k));
                        if (n > 0)
                            kw.add(new int[]{k, n});
                    }
                    if (!kw.isEmpty())
                        o.put("kw", kw);
                }
                if (c.getZone() == Zone.HAND)
                    o.put("cost", cost(game, c));
                cards.add(o);
            }
            JSONObject rec = new JSONObject();
            rec.put("game", GAME.get());
            rec.put("player", player);
            rec.put("kind", kind);
            rec.put("ctx", d.getText());
            switch (kind) {
                case "action" -> {
                    if (!action(game, d, answer, index, rec))
                        return;
                    double[] tree = treeScores(game, d, player);
                    if (tree != null)
                        rec.put("tree", tree);
                }
                case "assign" -> {
                    if (!assign(d, answer, index, rec))
                        return;
                }
                case "move" -> {
                    JSONArray opts = new JSONArray();
                    for (String r : params.getOrDefault("results", new String[0]))
                        opts.add(Map.of("src", -1, "text", r));
                    rec.put("opts", opts);
                    rec.put("chosen", Integer.parseInt(answer.trim()));
                }
                case "reconcile" -> {
                    JSONArray opts = new JSONArray();
                    opts.add(Map.of("src", -1, "text", "keep"));
                    for (String cid : params.getOrDefault("cardId", new String[0]))
                        opts.add(Map.of("src", index.getOrDefault(Integer.parseInt(cid), -1), "text", "discard"));
                    rec.put("opts", opts);
                    rec.put("chosen", ReconcileChoiceTrainer.chosen(d, answer));
                }
                default -> {
                    return;
                }
            }
            rec.put("bags", bags(gs, player));
            rec.put("fp", player.equals(gs.getCurrentPlayerId()) ? 1 : 0);
            rec.put("state", state);
            rec.put("cards", cards);
            String line = rec.toString();
            synchronized (BoardDump.class) {
                out.println(line);
            }
        } catch (RuntimeException ignored) {
            // a board the dump cannot describe is left out
        }
    }

    /** The decision's kind, or null for one the network does not take. */
    static String kind(AwaitingDecision d) {
        var t = d.getDecisionType();
        if (t == AwaitingDecisionType.CARD_ACTION_CHOICE)
            return d.getDecisionParameters().getOrDefault("actionId", new String[0]).length > 0 ? "action" : null;
        if (t == AwaitingDecisionType.ASSIGN_MINIONS)
            return "assign";
        if (t == AwaitingDecisionType.MULTIPLE_CHOICE && MOVE.equals(d.getText()))
            return "move";
        if (ReconcileChoiceTrainer.applies(d))
            return "reconcile";
        return null;
    }

    private static boolean action(LotroGame game, AwaitingDecision d, String answer, Map<Integer, Integer> index,
                                  JSONObject rec) {
        GameState gs = game.getGameState();
        var params = d.getDecisionParameters();
        String[] ids = params.getOrDefault("actionId", new String[0]);
        String[] cardIds = params.getOrDefault("cardId", new String[0]);
        String[] texts = params.getOrDefault("actionText", new String[0]);
        JSONArray opts = new JSONArray();
        int chosen = -1;
        List<String> vocab = vocab();
        String context = com.gempukku.lotro.bots.rl.fotrstarters.models.cardaction.AnyPhaseCardActionTrainer
                .contextName(d.getText());
        for (int i = 0; i < ids.length; i++) {
            JSONObject o = new JSONObject();
            int src = -1;
            PhysicalCard card = null;
            try {
                card = gs.findCardById(Integer.parseInt(cardIds[i]));
                src = index.getOrDefault(Integer.parseInt(cardIds[i]), -1);
            } catch (RuntimeException ignored) {
            }
            String text = i < texts.length ? texts[i] : "";
            o.put("src", src);
            o.put("text", text);
            // what this option does: its twilight cost to play (with modifiers) and the card text of the ability
            // it fires in this phase -- the trees have both and the network did not
            o.put("cost", card != null && text.toLowerCase().startsWith("play") ? cost(game, card) : 0);
            if (card != null && vocab != null) {
                double[] t = CardTextFeatures.of(vocab, card.getBlueprintId(), text, context);
                JSONArray tok = new JSONArray();
                for (int k = 0; k < t.length; k++)
                    if (t[k] != 0)
                        tok.add(new double[]{k, t[k]});
                o.put("tok", tok);
            }
            opts.add(o);
            if (ids[i].equals(answer))
                chosen = i;
        }
        if (chosen < 0 && !answer.isEmpty())
            return false;
        rec.put("opts", opts);
        rec.put("chosen", chosen); // -1: pass
        return true;
    }

    /** Per minion offered, the character the answer sent it to (an index into "fp"), -1 if none. */
    private static boolean assign(AwaitingDecision d, String answer, Map<Integer, Integer> index, JSONObject rec) {
        var params = d.getDecisionParameters();
        String[] fps = params.getOrDefault("freeCharacters", new String[0]);
        String[] minions = params.getOrDefault("minions", new String[0]);
        if (fps.length == 0 || minions.length == 0)
            return false;
        Map<String, String> to = new HashMap<>();
        for (var e : AssignmentProbe.parse(answer).entrySet())
            for (String m : e.getValue())
                to.put(m, e.getKey());
        List<String> fpList = Arrays.asList(fps);
        JSONArray fp = new JSONArray(), min = new JSONArray(), chosen = new JSONArray();
        for (String f : fps)
            fp.add(index.getOrDefault(Integer.parseInt(f), -1));
        for (String m : minions) {
            min.add(index.getOrDefault(Integer.parseInt(m), -1));
            String f = to.get(m);
            chosen.add(f == null ? -1 : fpList.indexOf(f));
        }
        rec.put("asg", Map.of("fp", fp, "min", min));
        rec.put("chosen", chosen);
        return true;
    }

    private static int cost(LotroGame game, PhysicalCard card) {
        try {
            return game.getModifiersQuerying().getTwilightCostToPlay(game, card, null, 0, false);
        } catch (RuntimeException e) {
            return card.getBlueprint().getTwilightCost();
        }
    }

    // ---- piles

    /** What the deciding player has seen of the opponent's cards this game: game -> owner -> blueprints. */
    private static final Map<Object, Map<String, LinkedHashSet<String>>> SEEN = new WeakHashMap<>();

    /** Remember every card each player has shown in a public zone; HumanReplay calls this at every decision. */
    static void observe(LotroGame game) {
        if (OUT == null)
            return;
        try {
            see(game);
        } catch (RuntimeException ignored) {
            // before the deal GameState throws rather than answer an empty pile
        }
    }

    private static void see(LotroGame game) {
        GameState gs = game.getGameState();
        Map<String, LinkedHashSet<String>> seen;
        synchronized (SEEN) {
            seen = SEEN.computeIfAbsent(game, k -> new HashMap<>());
        }
        for (String p : gs.getPlayerNames()) {
            var s = seen.computeIfAbsent(p, k -> new LinkedHashSet<>());
            for (PhysicalCard c : gs.getInPlay())
                if (c.getOwner().equals(p) && c.getBlueprint().getCardType() != CardType.SITE)
                    s.add(base(c.getBlueprintId()));
            for (List<? extends PhysicalCard> pile : List.of(gs.getDiscard(p), gs.getDeadPile(p), gs.getStacked(p),
                    gs.getRemoved(p)))
                if (pile != null)
                    for (PhysicalCard c : pile)
                        s.add(base(c.getBlueprintId()));
        }
    }

    private static JSONObject bags(GameState gs, String player) {
        JSONObject bags = new JSONObject();
        String opp = gs.getPlayerNames().stream().filter(p -> !p.equals(player)).findFirst().orElse(null);
        bags.put("myDiscard", ids(gs.getDiscard(player)));
        bags.put("myDead", ids(gs.getDeadPile(player)));
        if (opp != null) {
            bags.put("oppDiscard", ids(gs.getDiscard(opp)));
            bags.put("oppDead", ids(gs.getDeadPile(opp)));
            Map<String, LinkedHashSet<String>> seen;
            synchronized (SEEN) {
                seen = SEEN.get(gs.getGame());
            }
            bags.put("oppSeen", seen != null && seen.get(opp) != null ? new ArrayList<>(seen.get(opp)) : List.of());
        }
        // the own deck's remainder: the decklist less every card of the player's that is out of the deck
        List<? extends PhysicalCard> deck = gs.getDeck(player);
        List<String> list;
        Map<String, List<String>> decks = DECKS;
        if (decks != null) {
            List<String> known = decks.get(GAME.get() + "\t" + player);
            list = known != null ? new ArrayList<>(known) : new ArrayList<>();
            Map<String, Integer> left = new HashMap<>();
            for (String b : list)
                left.merge(base(b), 1, Integer::sum);
            List<PhysicalCard> out = new ArrayList<>();
            for (PhysicalCard c : gs.getInPlay())
                if (c.getOwner().equals(player))
                    out.add(c);
            for (List<? extends PhysicalCard> pile : List.of(gs.getHand(player), gs.getDiscard(player),
                    gs.getDeadPile(player), gs.getStacked(player), gs.getRemoved(player)))
                if (pile != null)
                    out.addAll(pile);
            for (PhysicalCard c : out)
                left.computeIfPresent(base(c.getBlueprintId()), (k, v) -> v > 1 ? v - 1 : null);
            list = new ArrayList<>();
            for (var e : left.entrySet())
                for (int i = 0; i < e.getValue(); i++)
                    list.add(e.getKey());
        } else if (!TRUE_DECK) {
            return bags;
        } else {
            list = new ArrayList<>();
            for (PhysicalCard c : deck)
                list.add(base(c.getBlueprintId()));
        }
        Collections.sort(list); // the contents, never the order
        bags.put("myDeck", list);
        bags.put("myDeckUnknown", Math.max(0, deck.size() - list.size()));
        return bags;
    }

    static String base(String bp) {
        return bp.replaceAll("[*T]+$", "");
    }

    private static JSONArray ids(List<? extends PhysicalCard> cards) {
        JSONArray a = new JSONArray();
        if (cards != null)
            for (PhysicalCard c : cards)
                a.add(c.getBlueprintId());
        return a;
    }

    /** {@code --decklists <file>}: JSON lines {game, player, deck: [blueprints]} (scripts/decklists.py). */
    static Map<String, List<String>> loadDecklists(Path file) throws IOException {
        Map<String, List<String>> m = new HashMap<>();
        for (String line : Files.readAllLines(file)) {
            JSONObject o = JSON.parseObject(line);
            m.put(o.getString("game") + "\t" + o.getString("player"), o.getJSONArray("deck").toJavaList(String.class));
        }
        return m;
    }

    /** The game being replayed on this thread (HumanReplay sets it), for the records' "game". */
    static final ThreadLocal<String> GAME = new ThreadLocal<>();

    /** -Dboards.trees=&lt;model dir&gt;: also record that bot's card-action scores for every option, then pass. */
    private static final String TREES = System.getProperty("boards.trees");
    private static volatile com.gempukku.lotro.bots.rl.fotrstarters.models.ModelRegistry treeRegistry;

    private static double[] treeScores(LotroGame game, AwaitingDecision d, String player) {
        if (TREES == null)
            return null;
        try {
            if (treeRegistry == null)
                synchronized (BoardDump.class) {
                    if (treeRegistry == null)
                        treeRegistry = com.gempukku.lotro.bots.ModelDirectory.load(Path.of(TREES));
                }
            var trainer = new com.gempukku.lotro.bots.rl.fotrstarters.models.cardaction.AnyPhaseCardActionTrainer();
            int previous = CardFeatures.version();
            try {
                CardFeatures.setVersion(treeRegistry.getFeatureVersion());
                com.gempukku.lotro.bots.rl.fotrstarters.models.cardaction.AnyPhaseCardActionTrainer.LAST_SCORES.remove();
                trainer.getAnswer(game.getGameState(), d, player,
                        new com.gempukku.lotro.bots.rl.fotrstarters.GeneralStateFeatures(), treeRegistry);
                return com.gempukku.lotro.bots.rl.fotrstarters.models.cardaction.AnyPhaseCardActionTrainer.LAST_SCORES.get();
            } finally {
                CardFeatures.setVersion(previous);
            }
        } catch (Exception e) {
            return null;
        }
    }

    private static volatile List<String> vocab;

    /** The card-text vocabulary the current models use, for the options' ability tokens; null if it cannot be read. */
    private static List<String> vocab() {
        if (vocab == null)
            try {
                vocab = CardTextFeatures.vocabulary(Path.of("bot-models-exp/sel10-big5/card-text"));
            } catch (IOException e) {
                return null;
            }
        return vocab;
    }

    private static void add(List<PhysicalCard> items, Map<Integer, Integer> index, PhysicalCard c) {
        if (c == null || index.containsKey(c.getCardId()))
            return;
        index.put(c.getCardId(), items.size());
        items.add(c);
    }

    /** {@code cards <out.tsv>}: blueprint, then its set-2 card numbers, then its card-text tokens (vocabulary in the header). */
    public static void main(String[] args) throws Exception {
        if (args.length != 2 || !args[0].equals("cards"))
            throw new IllegalArgumentException("usage: BoardDump cards <out.tsv>");
        BotVsBot.context(Path.of("gemp-lotr-cards/src/main/resources"), null);
        CardFeatures.setVersion(2);
        List<String> vocab = CardTextFeatures.vocabulary(Path.of("bot-models-exp/sel10-big5/card-text"));
        try (PrintWriter out = new PrintWriter(Files.newBufferedWriter(Path.of(args[1]), StandardCharsets.UTF_8))) {
            out.println("bp\tnumbers\t" + String.join(" ", vocab));
            int n = 0;
            for (String bp : CardTextFeatures.ids()) {
                try {
                    double[] f = CardFeatures.getCardFeatures(bp, 0);
                    double[] t = CardTextFeatures.of(vocab, bp, "Play", "none");
                    StringBuilder sb = new StringBuilder(bp).append('\t');
                    for (int i = 0; i < f.length; i++)
                        sb.append(i > 0 ? " " : "").append((float) f[i]);
                    sb.append('\t');
                    for (int i = 0; i < t.length; i++)
                        sb.append(i > 0 ? " " : "").append((float) t[i]);
                    out.println(sb);
                    n++;
                } catch (Exception ignored) {
                    // a definition the library does not load
                }
            }
            System.out.println(n + " cards");
        }
        System.exit(0);
    }
}
