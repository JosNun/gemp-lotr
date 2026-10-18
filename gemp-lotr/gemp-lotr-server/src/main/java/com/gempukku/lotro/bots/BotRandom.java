package com.gempukku.lotro.bots;

import java.util.Random;

/**
 * Where bots draw their own random choices: the burden bid, exploration, and the fallback for decisions no model
 * answers. Unseeded by default, as before. A harness that wants a game it can repeat -- the same seed asking the same
 * questions and getting the same answers -- installs a seeded one for the game's thread ({@link #use}); the engine's
 * own draws come from GameState's Random, which bots must not touch, or replaying a game's answers without its bots
 * would draw differently.
 */
public final class BotRandom {
    private static final ThreadLocal<Random> CURRENT = ThreadLocal.withInitial(Random::new);

    private BotRandom() {
    }

    public static Random current() {
        return CURRENT.get();
    }

    /** Installs {@code random} for this thread and returns the one it replaces, to be put back with this again. */
    public static Random use(Random random) {
        Random previous = CURRENT.get();
        CURRENT.set(random);
        return previous;
    }
}
