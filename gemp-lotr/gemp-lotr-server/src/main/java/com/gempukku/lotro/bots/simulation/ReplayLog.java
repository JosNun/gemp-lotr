package com.gempukku.lotro.bots.simulation;

import com.alibaba.fastjson2.JSONArray;
import com.alibaba.fastjson2.JSONObject;
import com.gempukku.lotro.bots.BotPlayer;
import com.gempukku.lotro.common.CardType;
import com.gempukku.lotro.game.PhysicalCard;
import com.gempukku.lotro.game.state.GameState;
import com.gempukku.lotro.logic.GameUtils;
import com.gempukku.lotro.logic.decisions.AwaitingDecision;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

/**
 * A game written down to watch (2026-11-14, -Dbvb.replays=&lt;dir&gt;): the user wanted to show other developers what the
 * bots do, in one static page. GEMP's own replays are a player's event stream for its web client, which needs the
 * server; this is the board as it stood at every decision of either bot -- both hands, since the watcher is neither
 * player -- with what was asked, what could be answered and what was. One JSON file a game
 * (replay-&lt;matchup&gt;-&lt;game&gt;.json) for scripts/replay_page.py to put in a page. Cards are named once ("cards").
 */
final class ReplayLog {
    static final String DIR = System.getProperty("bvb.replays");
    private static final ThreadLocal<ReplayLog> CURRENT = new ThreadLocal<>();

    private final JSONArray steps = new JSONArray();
    private final Map<String, JSONObject> cards = new TreeMap<>();

    static void start() {
        if (DIR != null)
            CURRENT.set(new ReplayLog());
    }

    static BotPlayer wrap(BotPlayer bot) {
        return DIR == null ? bot : new Watching(bot);
    }

    /** Writes the game: {@code meta} is whatever the caller knows of it (bots, decks, winner, how it ended). */
    static void finish(JSONObject meta, int matchup, int game) {
        ReplayLog log = CURRENT.get();
        CURRENT.remove();
        if (log == null)
            return;
        try {
            JSONObject o = new JSONObject();
            o.put("meta", meta);
            o.put("cards", log.cards);
            o.put("steps", log.steps);
            Files.createDirectories(Path.of(DIR));
            Files.writeString(Path.of(DIR, "replay-" + matchup + "-" + game + ".json"), o.toString(),
                    StandardCharsets.UTF_8);
        } catch (Exception e) {
            System.err.println("replay not written: " + e);
        }
    }

    private record Watching(BotPlayer inner) implements BotPlayer {
        @Override
        public String getName() {
            return inner.getName();
        }

        @Override
        public String chooseAction(GameState gs, AwaitingDecision d) {
            ReplayLog log = CURRENT.get();
            JSONObject step = null;
            if (log != null)
                try {
                    step = log.board(gs, d, getName());
                } catch (RuntimeException e) {
                    step = null; // a board that could not be read: the game goes on
                }
            Searching.LAST_OWN.remove();
            String answer = inner.chooseAction(gs, d);
            if (step != null)
                try {
                    step.put("a", log.answer(gs, d, answer));
                    String own = Searching.LAST_OWN.get();
                    if (own != null && !own.equals(answer))
                        step.put("own", log.answer(gs, d, own)); // what the bot would have answered without the search
                    if (own != null)
                        step.put("searched", true);
                    log.steps.add(step);
                } catch (RuntimeException ignored) {
                }
            return answer;
        }
    }

    private String card(PhysicalCard c) {
        String bp = c.getBlueprintId();
        if (!cards.containsKey(bp)) {
            JSONObject o = new JSONObject();
            var b = c.getBlueprint();
            o.put("name", GameUtils.getFullName(b));
            o.put("type", String.valueOf(b.getCardType()));
            o.put("side", String.valueOf(b.getSide()));
            o.put("culture", String.valueOf(b.getCulture()));
            cards.put(bp, o);
        }
        return bp;
    }

    private JSONObject board(GameState gs, AwaitingDecision d, String player) {
        var game = gs.getGame();
        var mq = game.getModifiersQuerying();
        JSONObject s = new JSONObject();
        s.put("p", player);
        s.put("turn", gs.getTurnNumber());
        s.put("phase", String.valueOf(gs.getCurrentPhase()));
        s.put("fp", gs.getCurrentPlayerId());
        s.put("tw", gs.getTwilightPool());
        JSONObject per = new JSONObject();
        for (String p : gs.getPlayerNames()) {
            JSONObject o = new JSONObject();
            o.put("site", gs.getPlayerPosition(p));
            JSONArray hand = new JSONArray();
            for (PhysicalCard c : gs.getHand(p))
                hand.add(card(c));
            o.put("hand", hand);
            o.put("deck", gs.getDeck(p).size());
            o.put("discard", gs.getDiscard(p).size());
            JSONArray dead = new JSONArray();
            for (PhysicalCard c : gs.getDeadPile(p))
                dead.add(card(c));
            o.put("dead", dead);
            PhysicalCard rb = gs.getRingBearer(p);
            o.put("rb", rb != null ? rb.getCardId() : -1);
            o.put("burdens", rb != null ? gs.getPlayerBurdens(p) : 0);
            per.put(p, o);
        }
        s.put("players", per);
        Set<Integer> fighting = new HashSet<>();
        var sk = gs.getSkirmish();
        if (sk != null) {
            if (sk.getFellowshipCharacter() != null)
                fighting.add(sk.getFellowshipCharacter().getCardId());
            for (PhysicalCard m : sk.getShadowCharacters())
                fighting.add(m.getCardId());
        }
        Map<Integer, Integer> assignedTo = new HashMap<>();
        for (var a : gs.getAssignments())
            for (PhysicalCard m : a.getShadowCharacters())
                assignedTo.put(m.getCardId(), a.getFellowshipCharacter().getCardId());
        JSONArray inPlay = new JSONArray();
        for (PhysicalCard c : gs.getInPlay()) {
            JSONObject o = new JSONObject();
            o.put("id", c.getCardId());
            o.put("bp", card(c));
            o.put("o", c.getOwner());
            o.put("z", String.valueOf(c.getZone()));
            PhysicalCard on = c.getAttachedTo() != null ? c.getAttachedTo() : c.getStackedOn();
            if (on != null)
                o.put("on", on.getCardId());
            CardType t = c.getBlueprint().getCardType();
            if (t == CardType.COMPANION || t == CardType.ALLY || t == CardType.MINION) {
                o.put("str", mq.getStrength(game, c));
                o.put("vit", mq.getVitality(game, c));
                o.put("w", gs.getWounds(c));
            }
            if (t == CardType.SITE)
                o.put("n", c.getSiteNumber());
            if (fighting.contains(c.getCardId()))
                o.put("f", 1);
            if (assignedTo.containsKey(c.getCardId()))
                o.put("vs", assignedTo.get(c.getCardId()));
            inPlay.add(o);
        }
        s.put("cards", inPlay);
        s.put("q", d.getText() == null ? "" : d.getText().replaceAll("<[^>]*>", ""));
        s.put("kind", String.valueOf(d.getDecisionType()));
        s.put("opts", options(gs, d));
        return s;
    }

    /** What could be answered, in words: an action's text, a card's name, a choice's text, a number's range. */
    private JSONArray options(GameState gs, AwaitingDecision d) {
        var p = d.getDecisionParameters();
        JSONArray out = new JSONArray();
        switch (d.getDecisionType()) {
            case CARD_ACTION_CHOICE, ACTION_CHOICE -> {
                for (String t : p.getOrDefault("actionText", new String[0]))
                    out.add(t.replaceAll("<[^>]*>", ""));
            }
            case CARD_SELECTION -> {
                for (String id : p.getOrDefault("cardId", new String[0]))
                    out.add(name(gs, id));
            }
            case ARBITRARY_CARDS -> {
                String[] bps = p.getOrDefault("blueprintId", new String[0]), ok = p.get("selectable");
                for (int i = 0; i < bps.length; i++)
                    if (ok == null || i >= ok.length || "true".equals(ok[i]))
                        out.add(blueprintName(gs, bps[i]));
            }
            case MULTIPLE_CHOICE -> out.addAll(Arrays.asList(p.getOrDefault("results", new String[0])));
            case INTEGER -> out.add(p.getOrDefault("min", new String[]{"?"})[0] + " to "
                    + p.getOrDefault("max", new String[]{"?"})[0]);
            case ASSIGN_MINIONS -> {
                for (String id : p.getOrDefault("minions", new String[0]))
                    out.add(name(gs, id));
            }
        }
        return out;
    }

    private String answer(GameState gs, AwaitingDecision d, String answer) {
        var p = d.getDecisionParameters();
        if (answer == null)
            return "(no answer)";
        switch (d.getDecisionType()) {
            case CARD_ACTION_CHOICE, ACTION_CHOICE -> {
                if (answer.isEmpty())
                    return "Pass";
                String[] ids = p.getOrDefault("actionId", new String[0]), texts = p.getOrDefault("actionText", new String[0]);
                for (int i = 0; i < ids.length && i < texts.length; i++)
                    if (ids[i].equals(answer))
                        return texts[i].replaceAll("<[^>]*>", "");
                return answer;
            }
            case CARD_SELECTION -> {
                if (answer.isEmpty())
                    return "(none)";
                List<String> names = new ArrayList<>();
                for (String id : answer.split(","))
                    names.add(name(gs, id.trim()));
                return String.join(", ", names);
            }
            case ARBITRARY_CARDS -> {
                if (answer.isEmpty())
                    return "(none)";
                String[] ids = p.getOrDefault("cardId", new String[0]), bps = p.getOrDefault("blueprintId", new String[0]);
                List<String> names = new ArrayList<>();
                for (String a : answer.split(","))
                    for (int i = 0; i < ids.length && i < bps.length; i++)
                        if (ids[i].equals(a.trim()))
                            names.add(blueprintName(gs, bps[i]));
                return names.isEmpty() ? answer : String.join(", ", names);
            }
            case MULTIPLE_CHOICE -> {
                String[] r = p.getOrDefault("results", new String[0]);
                try {
                    return r[Integer.parseInt(answer.trim())];
                } catch (RuntimeException e) {
                    return answer;
                }
            }
            case ASSIGN_MINIONS -> {
                List<String> parts = new ArrayList<>();
                for (String part : answer.split(",")) {
                    String[] ids = part.trim().split("\\s+");
                    if (ids.length < 2)
                        continue;
                    List<String> minions = new ArrayList<>();
                    for (int i = 1; i < ids.length; i++)
                        minions.add(name(gs, ids[i]));
                    parts.add(name(gs, ids[0]) + " fights " + String.join(" and ", minions));
                }
                return parts.isEmpty() ? "(no one assigned)" : String.join("; ", parts);
            }
            default -> {
                return answer;
            }
        }
    }

    private String name(GameState gs, String cardId) {
        try {
            PhysicalCard c = gs.findCardById(Integer.parseInt(cardId));
            card(c);
            return GameUtils.getFullName(c);
        } catch (RuntimeException e) {
            return cardId;
        }
    }

    private String blueprintName(GameState gs, String blueprintId) {
        try {
            return GameUtils.getFullName(gs.getGame().getLotroCardBlueprintLibrary().getLotroCardBlueprint(blueprintId));
        } catch (Exception e) {
            return blueprintId;
        }
    }
}
