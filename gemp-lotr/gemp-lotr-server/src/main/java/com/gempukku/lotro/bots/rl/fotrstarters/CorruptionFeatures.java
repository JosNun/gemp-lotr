package com.gempukku.lotro.bots.rl.fotrstarters;

import com.gempukku.lotro.game.state.GameState;

import java.util.Arrays;

/**
 * How close each Ring-bearer is to corruption, as steps a linear value model can use (set 4 = set 2 + this block).
 * Set 2 has resistance left as a number, and a logistic fit on it rated the opponent one burden from corruption at
 * 56% won against 50% at full resistance -- in plain-bot games corruption almost never decided anything, and a
 * straight line cannot say "one burden from losing is nearly lost" anyway. Resistance here is the engine's, burdens
 * already taken off.
 */
public final class CorruptionFeatures {
    public static final int LENGTH = 8;

    private CorruptionFeatures() {
    }

    /** Set 2 with this block appended: for each of me and the opponent, resistance left <= 1, <= 2, <= 3, and 1/resistance. */
    public static double[] withCorruption(double[] set2, GameState gs, String me) {
        double[] f = Arrays.copyOf(set2, set2.length + LENGTH);
        double mine = set2[GeneralStateFeatures.index("myRbResistance")];
        double theirs = set2[GeneralStateFeatures.index("oppRbResistance")];
        if (set2[GeneralStateFeatures.index("pregame")] == 1)
            return f;
        fill(f, set2.length, mine);
        fill(f, set2.length + 4, theirs);
        return f;
    }

    private static void fill(double[] f, int at, double resistance) {
        f[at] = resistance <= 1 ? 1 : 0;
        f[at + 1] = resistance <= 2 ? 1 : 0;
        f[at + 2] = resistance <= 3 ? 1 : 0;
        f[at + 3] = 1.0 / Math.max(1, resistance);
    }
}
