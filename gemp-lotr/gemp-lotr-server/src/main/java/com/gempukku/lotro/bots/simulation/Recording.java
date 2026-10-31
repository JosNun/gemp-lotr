package com.gempukku.lotro.bots.simulation;

import com.alibaba.fastjson2.JSONObject;
import com.gempukku.lotro.bots.BotPlayer;
import com.gempukku.lotro.game.state.GameState;
import com.gempukku.lotro.logic.decisions.AwaitingDecision;

import java.io.IOException;
import java.io.PrintWriter;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.concurrent.ThreadLocalRandom;

/**
 * A bot whose decisions are written down as board records as it plays -- the decisions the network takes (card
 * actions of every phase, assignments, "move again?", reconcile), each with the answer the bot gave -- and, through
 * {@link #outcome}, how each game ended. Two uses (BOT-NOTES, 2026-10-31):
 * <ul>
 * <li>what the trees know, for the network: sel10-net6's own choices on the positions of its games, labelled like
 * people's (BotVsBot -Dbvb.boards);</li>
 * <li>the judge's horizon: every position of the round loop's league games with the eventual winner, so the outcome
 * head learns what positions are worth as these bots play them (BotTeach -Dteach.positions).</li>
 * </ul>
 * Each record's "game" is the id {@link BoardDump#GAME} holds on the thread; outcome lines share it, in the format of
 * game_routes.py's output (game, winner, route, how, conf).
 */
final class Recording implements BotPlayer {
    private final BotPlayer inner;
    private final PrintWriter out;
    private final double rate;

    Recording(BotPlayer inner, PrintWriter out, double rate) {
        this.inner = inner;
        this.out = out;
        this.rate = rate;
    }

    static PrintWriter open(String path) {
        if (path == null)
            return null;
        try {
            return new PrintWriter(Files.newBufferedWriter(Path.of(path), StandardCharsets.UTF_8,
                    StandardOpenOption.CREATE, StandardOpenOption.APPEND), true);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    @Override
    public String chooseAction(GameState gameState, AwaitingDecision decision) {
        String answer = inner.chooseAction(gameState, decision);
        if (out != null && answer != null && BoardDump.kind(decision) != null
                && ThreadLocalRandom.current().nextDouble() < rate)
            try {
                // (the callers set BoardDump.TRUE_DECK and LIVE once: a bot knows its own deck, and remembers what it saw)
                double[] st = com.gempukku.lotro.bots.rl.fotrstarters.BoardNet2.state(gameState, decision, getName());
                JSONObject rec = st == null ? null
                        : BoardDump.record(gameState.getGame(), decision, getName(), answer, st);
                if (rec != null) {
                    String line = rec.toString();
                    synchronized (out) {
                        out.println(line);
                    }
                }
            } catch (RuntimeException ignored) {
                // a board the dump cannot describe is left out
            }
        return answer;
    }

    @Override
    public String getName() {
        return inner.getName();
    }

    /** How a game ended, for the outcome head: winner by name, route from the engine's reason. */
    static void outcome(PrintWriter out, String game, String winner, String reason) {
        if (out == null || game == null || winner == null)
            return;
        JSONObject o = new JSONObject();
        o.put("game", game);
        o.put("winner", winner);
        String route = GameLog.route(reason);
        o.put("route", "other".equals(route) ? null : route);
        o.put("how", "other".equals(route) || route == null ? "unsure" : "board");
        o.put("conf", 1.0);
        synchronized (out) {
            out.println(o);
        }
    }
}
