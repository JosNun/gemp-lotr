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
            for (int i = 0; i < ids.length; i++) {
                JSONObject o = new JSONObject();
                int src = -1;
                try {
                    src = index.getOrDefault(Integer.parseInt(cardIds[i]), -1);
                } catch (RuntimeException ignored) {
                }
                o.put("src", src);
                o.put("text", i < texts.length ? texts[i] : "");
                opts.add(o);
                if (ids[i].equals(answer))
                    chosen = i;
            }
            JSONObject rec = new JSONObject();
            rec.put("ctx", d.getText());
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
