package com.gempukku.lotro.bots.simulation;

import com.alibaba.fastjson2.JSONObject;
import com.gempukku.lotro.common.CardType;
import com.gempukku.lotro.game.PhysicalCard;
import com.gempukku.lotro.game.state.GameState;
import com.gempukku.lotro.logic.decisions.AwaitingDecision;
import com.gempukku.lotro.logic.decisions.AwaitingDecisionType;
import com.gempukku.lotro.logic.timing.DefaultLotroGame;

import java.util.*;

/**
 * How one bot game was played, per player, for the dashboard's style fingerprints and the "does its style follow its
 * deck" question: how it ended (site 9, the Ring-bearer killed, corrupted), and per player the site reached, companions
 * lost, "move again?" asked and taken, card-action prompts passed per phase, the cards played, minions played per Shadow
 * turn, and the twilight left when the Shadow player stopped playing. {@code -Dbvb.gameLog=<file>} makes BotVsBot write
 * one JSON line per game.
 * <p>
 * It binds to the first game started on its thread and ignores every other: the skirmish probe and search play copies
 * of the game inside a decision, and those are not the game.
 */
public final class GameLog {
    static final ThreadLocal<GameLog> CURRENT = new ThreadLocal<>();

    private DefaultLotroGame game;
    private final Map<String, Player> players = new LinkedHashMap<>();

    static final class Player {
        int moveAsked, moveYes, shadowTurns, minions, unspentSum, unspentN;
        final Map<String, int[]> passes = new TreeMap<>(); // context -> {asked, passed}
        final Map<String, Integer> played = new TreeMap<>();
        final Set<Integer> shadowTurnSeen = new HashSet<>();
    }

    /** BotGame: the game this log follows, if it has none yet. */
    void bind(DefaultLotroGame g) {
        if (game == null)
            game = g;
    }

    /** BotGame: {@code player} answered {@code decision} with {@code answer} (as the engine accepted it). */
    void decided(DefaultLotroGame g, String player, AwaitingDecision decision, String answer) {
        if (g != game || answer == null)
            return;
        try {
            GameState gs = g.getGameState();
            Player p = players.computeIfAbsent(player, k -> new Player());
            boolean fp = player.equals(gs.getCurrentPlayerId());
            if (!fp && gs.isInit())
                if (p.shadowTurnSeen.add(gs.getTurnNumber()))
                    p.shadowTurns++;
            if (decision.getDecisionType() == AwaitingDecisionType.MULTIPLE_CHOICE
                    && BoardDump.MOVE.equals(decision.getText())) {
                p.moveAsked++;
                String[] results = decision.getDecisionParameters().getOrDefault("results", new String[0]);
                try {
                    if ("Yes".equals(results[Integer.parseInt(answer.trim())]))
                        p.moveYes++;
                } catch (RuntimeException ignored) {
                }
            }
            if (decision.getDecisionType() == AwaitingDecisionType.CARD_ACTION_CHOICE) {
                String ctx = com.gempukku.lotro.bots.rl.fotrstarters.models.cardaction.AnyPhaseCardActionTrainer
                        .contextName(decision.getText());
                if ("Choose action to play or Pass".equals(decision.getText()))
                    ctx = "skirmish";
                int[] c = p.passes.computeIfAbsent(ctx, k -> new int[2]);
                c[0]++;
                if (answer.isEmpty()) {
                    c[1]++;
                    if ("Play Shadow action or Pass".equals(decision.getText())) {
                        p.unspentSum += gs.getTwilightPool();
                        p.unspentN++;
                    }
                } else {
                    String[] ids = decision.getDecisionParameters().getOrDefault("actionId", new String[0]);
                    String[] cards = decision.getDecisionParameters().getOrDefault("cardId", new String[0]);
                    String[] texts = decision.getDecisionParameters().getOrDefault("actionText", new String[0]);
                    int i = Arrays.asList(ids).indexOf(answer);
                    if (i >= 0 && i < texts.length && texts[i].startsWith("Play")) {
                        PhysicalCard card = gs.findCardById(Integer.parseInt(cards[i]));
                        if (card != null) {
                            p.played.merge(BoardDump.base(card.getBlueprintId()), 1, Integer::sum);
                            if (!fp && card.getBlueprint().getCardType() == CardType.MINION)
                                p.minions++;
                        }
                    }
                }
            }
        } catch (RuntimeException ignored) {
            // a decision the log cannot read is left out of it
        }
    }

    static String route(String reason) {
        if (reason == null)
            return null;
        String r = reason.toLowerCase();
        if (r.contains("surviving to regroup") || r.contains("site 9") || r.contains("site 6"))
            return "site9";
        if (r.contains("corrupt"))
            return "corrupt";
        if (r.contains("ring-bearer is dead") || r.contains("ring-bearer dead"))
            return "kill";
        return "other";
    }

    /** The line for this game; {@code names} maps the engine's player names to what the log calls them. */
    JSONObject finish(BotGame.Result r, Map<String, String> names) {
        JSONObject o = new JSONObject();
        o.put("winner", r.winner() == null ? null : names.getOrDefault(r.winner(), r.winner()));
        o.put("first", r.first() == null ? null : names.getOrDefault(r.first(), r.first()));
        o.put("reason", r.reason());
        o.put("route", route(r.reason()));
        o.put("turns", r.turns());
        o.put("error", r.error());
        JSONObject ps = new JSONObject();
        if (game != null) {
            GameState gs = game.getGameState();
            for (String name : names.keySet()) {
                JSONObject q = new JSONObject();
                Player p = players.getOrDefault(name, new Player());
                try {
                    q.put("site", gs.getPlayerPosition(name));
                    int dead = 0;
                    for (PhysicalCard c : gs.getDeadPile(name))
                        dead += c.getBlueprint().getCardType() == CardType.COMPANION ? 1 : 0;
                    q.put("deadCompanions", dead);
                } catch (RuntimeException ignored) {
                }
                q.put("moveAsked", p.moveAsked);
                q.put("moveYes", p.moveYes);
                q.put("shadowTurns", p.shadowTurns);
                q.put("minions", p.minions);
                q.put("unspentTwilight", p.unspentN == 0 ? null : (double) p.unspentSum / p.unspentN);
                JSONObject pass = new JSONObject();
                p.passes.forEach((k, v) -> pass.put(k, v));
                q.put("passes", pass);
                q.put("played", p.played);
                ps.put(names.get(name), q);
            }
        }
        o.put("players", ps);
        return o;
    }
}
