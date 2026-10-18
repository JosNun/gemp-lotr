package com.gempukku.lotro.bots.rl.fotrstarters;

import com.gempukku.lotro.common.*;
import com.gempukku.lotro.game.LotroCardBlueprint;
import com.gempukku.lotro.game.PhysicalCard;
import com.gempukku.lotro.game.state.GameState;

import java.util.*;

/**
 * What a player knows beyond the board, as a block appended to feature set 2 (set 3). Three things a person plays
 * with that set 2 leaves out:
 * <ul>
 *   <li><b>What is left in my deck.</b> My decklist minus what I have seen: the draw deck, by type and strength.
 *   Whether the next draws can still bring a companion or a big minion changes what a hand is worth.</li>
 *   <li><b>What the opponent is playing.</b> The cultures and types of every card of theirs that has been public --
 *   in play, discarded, dead -- which is how a player reads an archetype; and, since in deck testing both lists are
 *   known, the summary of their unseen cards (hand and deck together: which of them is in hand is not known).</li>
 *   <li><b>A clogged hand.</b> Cards in hand that cannot help soon: the wrong side for this turn, unaffordable, or
 *   unique characters whose title is already in play or dead (a dead unique can never be played again).</li>
 * </ul>
 * The names follow the GeneralStateFeatures style; {@link #describe} prints a vector by name.
 */
public final class KnowledgeFeatures {
    private static final List<String> NAMES = new ArrayList<>();
    private static final Map<String, Integer> INDEX = new HashMap<>();

    static {
        for (String who : List.of("myDeck", "oppUnseen"))
            for (String what : List.of("Size", "Companions", "CompanionStrength", "Allies", "Minions",
                    "MinionStrength", "MinionVitality", "SkirmishEvents", "Events", "Conditions", "Possessions",
                    "FpCards", "ShadowCards", "TwilightCost"))
                add(who + what);
        for (String what : List.of("Cards", "Minions", "MinionStrength", "Events", "Conditions", "Possessions",
                "Companions", "Allies"))
            add("oppSeen" + what);
        for (Culture culture : Culture.values())
            add("oppSeen_" + culture.name());
        for (Culture culture : Culture.values())
            add("mySeen_" + culture.name());
        for (String what : List.of("WrongSide", "Unaffordable", "DeadUniques", "MaxCost", "Characters", "Cost"))
            add("myHand" + what);
    }

    private KnowledgeFeatures() {
    }

    private static void add(String name) {
        INDEX.put(name, NAMES.size());
        NAMES.add(name);
    }

    public static int length() {
        return NAMES.size();
    }

    public static String describe(double[] f, int offset) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < NAMES.size(); i++)
            if (f[offset + i] != 0)
                sb.append(NAMES.get(i)).append('=').append(f[offset + i]).append(' ');
        return sb.toString().trim();
    }

    /** Set 2 with this block appended. */
    public static double[] withKnowledge(double[] set2, GameState gs, String me) {
        double[] f = Arrays.copyOf(set2, set2.length + NAMES.size());
        fill(f, set2.length, gs, me);
        return f;
    }

    static void fill(double[] f, int at, GameState gs, String me) {
        String opp = gs.getPlayerNames().stream().filter(p -> !p.equals(me)).findFirst().orElse(null);
        String fp = gs.getCurrentPlayerId();
        if (opp == null || fp == null || !gs.isInit())
            return;
        boolean iAmFp = me.equals(fp);

        for (PhysicalCard card : zone(() -> gs.getDeck(me)))
            pool(f, at, "myDeck", card.getBlueprint());
        for (PhysicalCard card : zone(() -> gs.getDeck(opp)))
            pool(f, at, "oppUnseen", card.getBlueprint());
        for (PhysicalCard card : zone(() -> gs.getHand(opp)))
            pool(f, at, "oppUnseen", card.getBlueprint());

        // public cards: in play, discard, dead pile (removed-from-game cards are public too)
        Set<String> titlesTaken = new HashSet<>();
        List<PhysicalCard> seen = new ArrayList<>();
        for (PhysicalCard card : gs.getInPlay())
            seen.add(card);
        for (String player : List.of(me, opp)) {
            seen.addAll(zone(() -> gs.getDiscard(player)));
            seen.addAll(zone(() -> gs.getDeadPile(player)));
            seen.addAll(zone(() -> gs.getRemoved(player)));
            for (PhysicalCard dead : zone(() -> gs.getDeadPile(player)))
                if (player.equals(me))
                    titlesTaken.add(dead.getBlueprint().getTitle());
        }
        for (PhysicalCard card : seen) {
            LotroCardBlueprint bp = card.getBlueprint();
            if (bp.getCardType() == CardType.SITE || bp.getCardType() == CardType.THE_ONE_RING)
                continue;
            boolean mine = card.getOwner().equals(me);
            if (mine && card.getZone() != null && card.getZone().isInPlay())
                titlesTaken.add(bp.getTitle());
            if (bp.getCulture() != null)
                inc(f, at, (mine ? "mySeen_" : "oppSeen_") + bp.getCulture().name(), 1);
            if (mine)
                continue;
            inc(f, at, "oppSeenCards", 1);
            switch (bp.getCardType()) {
                case MINION -> {
                    inc(f, at, "oppSeenMinions", 1);
                    inc(f, at, "oppSeenMinionStrength", bp.getStrength());
                }
                case EVENT -> inc(f, at, "oppSeenEvents", 1);
                case CONDITION -> inc(f, at, "oppSeenConditions", 1);
                case POSSESSION, ARTIFACT -> inc(f, at, "oppSeenPossessions", 1);
                case COMPANION -> inc(f, at, "oppSeenCompanions", 1);
                case ALLY -> inc(f, at, "oppSeenAllies", 1);
                default -> {
                }
            }
        }

        int twilight = gs.getTwilightPool();
        for (PhysicalCard card : zone(() -> gs.getHand(me))) {
            LotroCardBlueprint bp = card.getBlueprint();
            int cost = bp.getTwilightCost();
            inc(f, at, "myHandCost", cost);
            f[at + INDEX.get("myHandMaxCost")] = Math.max(f[at + INDEX.get("myHandMaxCost")], cost);
            boolean fpCard = bp.getSide() == Side.FREE_PEOPLE;
            if (fpCard != iAmFp)
                inc(f, at, "myHandWrongSide", 1);
            else if (!fpCard && cost > twilight)
                inc(f, at, "myHandUnaffordable", 1); // as Shadow, twilight in the pool is the budget
            CardType type = bp.getCardType();
            if (type == CardType.COMPANION || type == CardType.ALLY || type == CardType.MINION) {
                inc(f, at, "myHandCharacters", 1);
                if (bp.isUnique() && titlesTaken.contains(bp.getTitle()))
                    inc(f, at, "myHandDeadUniques", 1);
            }
        }
    }

    private static void pool(double[] f, int at, String who, LotroCardBlueprint bp) {
        inc(f, at, who + "Size", 1);
        inc(f, at, who + "TwilightCost", bp.getTwilightCost());
        inc(f, at, bp.getSide() == Side.FREE_PEOPLE ? who + "FpCards" : who + "ShadowCards", 1);
        switch (bp.getCardType()) {
            case COMPANION -> {
                inc(f, at, who + "Companions", 1);
                inc(f, at, who + "CompanionStrength", bp.getStrength());
            }
            case ALLY -> inc(f, at, who + "Allies", 1);
            case MINION -> {
                inc(f, at, who + "Minions", 1);
                inc(f, at, who + "MinionStrength", bp.getStrength());
                inc(f, at, who + "MinionVitality", bp.getVitality());
            }
            case EVENT -> {
                inc(f, at, who + "Events", 1);
                if (bp.hasTimeword(Timeword.SKIRMISH))
                    inc(f, at, who + "SkirmishEvents", 1);
            }
            case CONDITION -> inc(f, at, who + "Conditions", 1);
            case POSSESSION, ARTIFACT -> inc(f, at, who + "Possessions", 1);
            default -> {
            }
        }
    }

    private static void inc(double[] f, int at, String name, double v) {
        f[at + INDEX.get(name)] += v;
    }

    private static List<? extends PhysicalCard> zone(java.util.function.Supplier<List<? extends PhysicalCard>> read) {
        try {
            List<? extends PhysicalCard> cards = read.get();
            return cards == null ? List.of() : cards;
        } catch (NullPointerException e) {
            return List.of();
        }
    }
}
