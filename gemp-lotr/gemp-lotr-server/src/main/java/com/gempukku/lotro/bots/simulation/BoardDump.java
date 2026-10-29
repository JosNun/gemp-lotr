package com.gempukku.lotro.bots.simulation;

import com.alibaba.fastjson2.JSONArray;
import com.alibaba.fastjson2.JSONObject;
import com.gempukku.lotro.bots.rl.fotrstarters.CardFeatures;
import com.gempukku.lotro.bots.rl.fotrstarters.CardTextFeatures;
import com.gempukku.lotro.common.CardType;
import com.gempukku.lotro.logic.timing.DefaultLotroGame;
import com.gempukku.lotro.game.PhysicalCard;
import com.gempukku.lotro.game.state.GameState;
import com.gempukku.lotro.logic.decisions.AwaitingDecision;

import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

/**
 * A card-action decision as the whole board, for the neural prototype in nn/: every card in play and in the deciding
 * player's hand as an item -- its card, whose it is, its zone, what it is attached to, its modified strength and
 * vitality, wounds, whether it is in the skirmish or assigned, whether it is a Ring-bearer -- the options offered with
 * the item each comes from, and the one chosen. The set-2 state goes along for the summary numbers. The trees see the
 * board as ~75 totals and fit their own training months barely better than held-out ones (Fellowship plays 67.4%
 * against 66.1%): the limit is what the inputs can say, not capacity.
 * <p>
 * {@code HumanReplay --boards <file>} writes one JSON line per reproduced card-action decision that --steps-only
 * admits; {@code BoardDump cards <out.tsv>} writes the card table the items refer to (set-2 card numbers and card text).
 */
final class BoardDump {
    static volatile PrintWriter OUT;

    private BoardDump() {
    }

    static void write(DefaultLotroGame game, AwaitingDecision d, String player, String answer, double[] state) {
        PrintWriter out = OUT;
        if (out == null)
            return;
        try {
            GameState gs = game.getGameState();
            var mq = game.getModifiersQuerying();
            List<PhysicalCard> items = new ArrayList<>();
            Map<Integer, Integer> index = new HashMap<>();
            for (PhysicalCard c : gs.getInPlay())
                add(items, index, c);
            for (PhysicalCard c : gs.getHand(player))
                add(items, index, c);
            String[] ids = d.getDecisionParameters().getOrDefault("actionId", new String[0]);
            String[] cardIds = d.getDecisionParameters().getOrDefault("cardId", new String[0]);
            String[] texts = d.getDecisionParameters().getOrDefault("actionText", new String[0]);
            for (String cid : cardIds) // an option's card from somewhere else: the discard pile, stacked, the deck
                try {
                    PhysicalCard c = gs.findCardById(Integer.parseInt(cid));
                    if (c != null)
                        add(items, index, c);
                } catch (RuntimeException ignored) {
                }
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
                boolean live = character && c.getZone() != null && c.getZone().isInPlay();
                o.put("str", live ? mq.getStrength(game, c) : 0);
                o.put("vit", live ? mq.getVitality(game, c) : 0);
                o.put("wnd", c.getZone() != null && c.getZone().isInPlay() ? gs.getWounds(c) : 0);
                o.put("skir", inSkirmish.contains(c.getCardId()) ? 1 : 0);
                o.put("asg", assigned.contains(c.getCardId()) ? 1 : 0);
                o.put("rb", bearers.contains(c.getCardId()) ? 1 : 0);
                cards.add(o);
            }
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
                int cost = 0;
                if (card != null && text.toLowerCase().startsWith("play"))
                    try {
                        cost = mq.getTwilightCostToPlay(game, card, null, 0, false);
                    } catch (RuntimeException e) {
                        cost = card.getBlueprint().getTwilightCost();
                    }
                o.put("cost", cost);
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
            // what is public but not on the table: both discard piles and dead piles, as cards
            JSONObject bags = new JSONObject();
            String opp = gs.getPlayerNames().stream().filter(p -> !p.equals(player)).findFirst().orElse(null);
            bags.put("myDiscard", ids(gs.getDiscard(player)));
            bags.put("myDead", ids(gs.getDeadPile(player)));
            if (opp != null) {
                bags.put("oppDiscard", ids(gs.getDiscard(opp)));
                bags.put("oppDead", ids(gs.getDeadPile(opp)));
            }
            JSONObject rec = new JSONObject();
            rec.put("game", GAME.get());
            rec.put("player", player);
            double[] tree = treeScores(game, d, player);
            if (tree != null)
                rec.put("tree", tree);
            rec.put("ctx", d.getText());
            rec.put("bags", bags);
            rec.put("fp", player.equals(gs.getCurrentPlayerId()) ? 1 : 0);
            rec.put("state", state);
            rec.put("cards", cards);
            rec.put("opts", opts);
            rec.put("chosen", chosen); // -1: pass
            String line = rec.toString();
            synchronized (BoardDump.class) {
                out.println(line);
            }
        } catch (RuntimeException ignored) {
            // a board the dump cannot describe is left out
        }
    }

    private static JSONArray ids(List<? extends PhysicalCard> cards) {
        JSONArray a = new JSONArray();
        if (cards != null)
            for (PhysicalCard c : cards)
                a.add(c.getBlueprintId());
        return a;
    }

    /** The game being replayed on this thread (HumanReplay sets it), for the records' "game". */
    static final ThreadLocal<String> GAME = new ThreadLocal<>();

    /** -Dboards.trees=&lt;model dir&gt;: also record that bot's card-action scores for every option, then pass. */
    private static final String TREES = System.getProperty("boards.trees");
    private static volatile com.gempukku.lotro.bots.rl.fotrstarters.models.ModelRegistry treeRegistry;

    private static double[] treeScores(DefaultLotroGame game, AwaitingDecision d, String player) {
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
            } catch (java.io.IOException e) {
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
