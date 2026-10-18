package com.gempukku.lotro.bots.simulation;

import com.gempukku.lotro.bots.BotPlayer;
import com.gempukku.lotro.logic.timing.DefaultLotroGame;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;

/**
 * BotTeach's rollout, played on a copy of the live game instead of a replay of it. The live game is waiting on the
 * decision being branched; the copy is too, so the {@link BotTeach.Branching} bots are given no prefix and their first
 * question is the branch itself, where they reseed the engine, reseed the bots' Random, reshuffle both decks and give
 * the option -- the same steps, in the same order, as at the end of a replay. {@link BotCloneCheck} checks the two
 * give the same game.
 */
final class CloneRollout {
    /** Time spent copying, and copies made: BotTeach prints the average. */
    static final AtomicLong copyNanos = new AtomicLong(), copies = new AtomicLong();

    private CloneRollout() {
    }

    /**
     * Plays one rollout from a copy of {@code live}. {@code prefix} is the steps the live game has taken so far (only
     * counted: the loop guard counts from the start of the game); {@code shared} must have an empty prefix.
     */
    static BotGame.Result play(GameCopier copier, DefaultLotroGame live, List<BotGame.Step> prefix,
                               BotTeach.Shared shared, BotPlayer policyA, BotPlayer policyB) {
        if (!shared.prefix.isEmpty())
            throw new IllegalArgumentException("a rollout from a copy starts at the branch");
        long t0 = System.nanoTime();
        DefaultLotroGame copy = copier.copy(live);
        copyNanos.addAndGet(System.nanoTime() - t0);
        copies.incrementAndGet();
        Map<String, Integer> counts = new HashMap<>();
        for (BotGame.Step s : prefix)
            counts.merge(s.player(), 1, Integer::sum);
        // the Random is reseeded at the branch, before any bot draws from it
        return BotGame.resume(copy, List.of(new BotTeach.Branching(policyA.getName(), shared, policyA),
                        new BotTeach.Branching(policyB.getName(), shared, policyB)), 1000, new ArrayList<>(), counts,
                prefix.size(), new java.util.Random(0));
    }
}
