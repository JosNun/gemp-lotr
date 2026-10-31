package com.gempukku.lotro.bots.rl.fotrstarters.models.assignment;

import com.gempukku.lotro.common.Keyword;
import com.gempukku.lotro.game.PhysicalCard;
import com.gempukku.lotro.game.state.Assignment;
import com.gempukku.lotro.game.state.GameState;
import com.gempukku.lotro.logic.decisions.AwaitingDecision;
import com.gempukku.lotro.logic.modifiers.CantBeOverwhelmedModifier;

import java.util.*;

/**
 * The skirmishes an assignment sets up, as they stand now. The assignment models saw each card's printed strength --
 * not a companion's weapon, a minion's bonus from the site or a condition, vitality left or damage -- and the Free
 * Peoples model chose as people did 45% of the time on July 2022's games, losing 1.2 points a time where it did not.
 * Read from the board as it is (the engine's strength, vitality, damage keyword and overwhelm multiplier), so it is
 * what each skirmish would be if nothing more were played: won, lost, overwhelmed, who dies, the Ring-bearer, and what
 * is left for the Shadow player to assign -- leftover minions and the companions still free for them.
 */
public final class AssignmentProbe {
    public static final int LENGTH = 15;
    public static final List<String> NAMES = List.of("skirmishes", "won", "lost", "overwhelmed", "companionsDie",
            "bearerLoses", "bearerDies", "minMargin", "sumMargin", "minionsDie", "leftMinions", "leftStrength",
            "leftMaxStrength", "freeCharacters", "worstLeftMatch");

    private AssignmentProbe() {
    }

    /** An answer ("fp minion minion,fp minion") as character id to minion ids. */
    public static Map<String, List<String>> parse(String answer) {
        Map<String, List<String>> m = new LinkedHashMap<>();
        if (answer == null)
            return m;
        for (String part : answer.split(",")) {
            String[] ids = part.trim().split("\\s+");
            if (ids.length < 2)
                continue;
            m.computeIfAbsent(ids[0], k -> new ArrayList<>()).addAll(Arrays.asList(ids).subList(1, ids.length));
        }
        return m;
    }

    /**
     * Per decision, each card's strength, vitality, damage and overwhelm multiplier, asked of the engine once: every
     * candidate assignment of a decision asks about the same cards, and nothing is assigned while they are scored.
     * The probe was a quarter of a tree bot's time (JFR, 2026-10-30), most of it in these modifier queries.
     */
    private static final ThreadLocal<Object[]> CACHE = new ThreadLocal<>(); // {decision, game state, Map<id, int[4]>}

    @SuppressWarnings("unchecked")
    private static int[] stats(GameState gs, AwaitingDecision decision, PhysicalCard c) {
        Object[] cache = CACHE.get();
        if (cache == null || cache[0] != decision || cache[1] != gs) {
            cache = new Object[]{decision, gs, new HashMap<Integer, int[]>()};
            CACHE.set(cache);
        }
        return ((Map<Integer, int[]>) cache[2]).computeIfAbsent(c.getCardId(), id -> {
            var game = gs.getGame();
            var q = game.getModifiersQuerying();
            return new int[]{q.getStrength(game, c), q.getVitality(game, c), q.getKeywordCount(game, c, Keyword.DAMAGE),
                    q.getOverwhelmMultiplier(game, c)};
        });
    }

    /** The skirmishes {@code assignment} sets up at {@code decision}, with those already assigned. */
    public static double[] features(GameState gs, AwaitingDecision decision, Map<String, List<String>> assignment) {
        Map<PhysicalCard, List<PhysicalCard>> fights = new LinkedHashMap<>();
        for (Assignment a : gs.getAssignments())
            fights.computeIfAbsent(a.getFellowshipCharacter(), k -> new ArrayList<>()).addAll(a.getShadowCharacters());
        Set<String> assignedMinions = new HashSet<>();
        for (var e : assignment.entrySet()) {
            PhysicalCard fp = gs.findCardById(Integer.parseInt(e.getKey()));
            if (fp == null)
                continue;
            for (String m : e.getValue()) {
                PhysicalCard minion = gs.findCardById(Integer.parseInt(m));
                if (minion != null) {
                    fights.computeIfAbsent(fp, k -> new ArrayList<>()).add(minion);
                    assignedMinions.add(m);
                }
            }
        }
        PhysicalCard bearer = null;
        try {
            bearer = gs.getRingBearer(gs.getCurrentPlayerId());
        } catch (RuntimeException ignored) {
        }
        double[] f = new double[LENGTH];
        int minMargin = Integer.MAX_VALUE;
        for (var e : fights.entrySet()) {
            PhysicalCard fp = e.getKey();
            if (e.getValue().isEmpty())
                continue;
            int[] fs = stats(gs, decision, fp);
            int fpStrength = fs[0], minions = 0, minionDamage = 0;
            for (PhysicalCard m : e.getValue()) {
                int[] ms = stats(gs, decision, m);
                minions += ms[0];
                minionDamage += ms[2];
            }
            int multiplier = fs[3];
            boolean overwhelmed = multiplier < CantBeOverwhelmedModifier.ImmuneToOverwhelmThreshold
                    && fpStrength * multiplier <= minions;
            boolean won = fpStrength > minions;
            boolean dies = overwhelmed || !won && fs[1] <= 1 + minionDamage;
            f[0]++;
            f[won ? 1 : 2]++;
            f[3] += overwhelmed ? 1 : 0;
            f[4] += dies ? 1 : 0;
            f[5] += fp == bearer && !won ? 1 : 0;
            f[6] += fp == bearer && dies ? 1 : 0;
            minMargin = Math.min(minMargin, fpStrength - minions);
            f[8] += fpStrength - minions;
            if (won) {
                int fpDamage = fs[2];
                for (PhysicalCard m : e.getValue())
                    f[9] += stats(gs, decision, m)[1] <= 1 + fpDamage ? 1 : 0;
            }
        }
        f[7] = minMargin == Integer.MAX_VALUE ? 0 : minMargin;
        // left for the Shadow player: unassigned minions, and the characters still free to take them
        String[] minionIds = decision.getDecisionParameters().get("minions");
        String[] freeIds = decision.getDecisionParameters().get("freeCharacters");
        int leftMax = 0;
        for (int i = 0; minionIds != null && i < minionIds.length; i++) {
            if (assignedMinions.contains(minionIds[i]))
                continue;
            PhysicalCard m = gs.findCardById(Integer.parseInt(minionIds[i]));
            if (m == null)
                continue;
            int s = stats(gs, decision, m)[0];
            f[10]++;
            f[11] += s;
            leftMax = Math.max(leftMax, s);
        }
        f[12] = leftMax;
        int weakestFree = Integer.MAX_VALUE;
        for (int i = 0; freeIds != null && i < freeIds.length; i++) {
            if (!assignment.getOrDefault(freeIds[i], List.of()).isEmpty())
                continue;
            PhysicalCard c = gs.findCardById(Integer.parseInt(freeIds[i]));
            if (c == null)
                continue;
            f[13]++;
            weakestFree = Math.min(weakestFree, stats(gs, decision, c)[0]);
        }
        f[14] = f[10] > 0 && weakestFree != Integer.MAX_VALUE ? leftMax - weakestFree : 0;
        return f;
    }
}
