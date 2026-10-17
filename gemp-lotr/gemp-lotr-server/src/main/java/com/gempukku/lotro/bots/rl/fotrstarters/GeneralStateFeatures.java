package com.gempukku.lotro.bots.rl.fotrstarters;

import com.gempukku.lotro.bots.rl.RLGameStateFeatures;
import com.gempukku.lotro.common.*;
import com.gempukku.lotro.game.PhysicalCard;
import com.gempukku.lotro.game.state.GameState;
import com.gempukku.lotro.game.state.LotroGame;
import com.gempukku.lotro.game.state.Skirmish;
import com.gempukku.lotro.logic.decisions.AwaitingDecision;
import com.gempukku.lotro.logic.modifiers.ModifiersQuerying;
import com.gempukku.lotro.logic.timing.GameStats;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Feature set 2: one fixed-length game-state vector for every decision, that means the same thing on any deck and
 * from either seat.
 * <p>
 * What it replaces ({@link FotrStartersRLGameStateFeatures}) counts copies of the 69 cards in the FotR starters,
 * identifies the deck by whether it holds Aragorn or Gandalf, counts only minions the *opponent* owns -- so as the
 * Shadow player the bot never saw its own -- and reads printed strength. Here the board is described by side (Free
 * Peoples, Shadow) rather than by owner, with {@code iAmFp} saying which side the bot is on, and strength, vitality
 * and resistance are the engine's modified values. Both of the last two are already net: ModifiersQuerying's
 * vitality has the wounds taken off and its resistance the burdens, so neither is subtracted again here.
 * <p>
 * Every feature has a name; {@link #index(String)} finds it, which is how {@link CardFeatures#join} picks the state
 * features it crosses with each option.
 */
public class GeneralStateFeatures implements RLGameStateFeatures {
    public static final int VERSION = 2;

    private static final List<String> NAMES = new ArrayList<>();
    private static final Map<String, Integer> INDEX = new HashMap<>();

    static {
        for (String name : List.of(
                "iAmFp", "pregame", "turn", "site", "myPosition", "oppPosition", "twilight", "moves", "moveLimit",
                "burdens", "threats", "wearingRing", "fpInitiative", "rbVitalityLeft", "rbResistanceLeft",
                // the Free Peoples side of the board
                "fpCompanions", "fpAllies", "fpCompanionStrength", "fpCompanionVitalityLeft", "fpMinVitalityLeft",
                "fpExhausted", "fpPossessions", "fpConditions", "fpDeadCompanions",
                // the Shadow side
                "minions", "minionStrength", "minionVitalityLeft", "maxMinionStrength", "fierceMinions",
                "archerMinions", "minionDamage", "shadowConditions", "shadowPossessions", "shadowAdvantage",
                // hands and piles: mine, and what is public of the opponent's
                "myHand", "myHandFp", "myHandShadow", "myHandMinions", "myHandSkirmishEvents", "myHandShadowCost",
                "oppHand", "myDeck", "oppDeck", "myDiscard", "oppDiscard",
                // the race, from the bot's side: both fellowships persist across turns, and the Shadow player's
                // own is inactive -- but it is half of whether that player wins. Set 2 first left it out and
                // predicted the winner worse than set 1 (held-out AUC 0.687 against 0.750, BotProbe).
                "myBurdens", "oppBurdens", "myRbResistance", "oppRbResistance", "myCompanions", "oppCompanions",
                "myCompanionVitalityLeft", "oppCompanionVitalityLeft", "myWounds", "oppWounds", "myDead", "oppDead",
                // the skirmish being fought, if any; the margin is from the bot's side
                "inSkirmish", "mySkirmishMargin", "fpSkirmishDamage", "shadowSkirmishDamage", "overwhelmed",
                "skirmishFpVitalityLeft", "skirmishMinions"))
            add(name);
        for (Phase phase : Phase.values())
            add("phase_" + phase.name());
    }

    private static void add(String name) {
        INDEX.put(name, NAMES.size());
        NAMES.add(name);
    }

    /** The vector with its feature names, non-zero entries only: for reading one by eye. */
    public static String describe(double[] f) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < f.length && i < NAMES.size(); i++)
            if (f[i] != 0)
                sb.append(NAMES.get(i)).append('=').append(f[i] == Math.rint(f[i]) ? String.valueOf((long) f[i]) : String.valueOf(f[i])).append(' ');
        return sb.toString().trim();
    }

    public static int length() {
        return NAMES.size();
    }

    public static int index(String name) {
        Integer i = INDEX.get(name);
        if (i == null)
            throw new IllegalArgumentException("No state feature " + name);
        return i;
    }

    @Override
    public double[] extractFeatures(GameState gs, AwaitingDecision decision, String me) {
        double[] f = new double[NAMES.size()];
        LotroGame game = gs.getGame();
        String fp = gs.getCurrentPlayerId();
        String opp = gs.getPlayerNames().stream().filter(p -> !p.equals(me)).findFirst().orElse(null);
        boolean iAmFp = me.equals(fp);
        set(f, "iAmFp", iAmFp ? 1 : 0);
        if (gs.getCurrentPhase() != null)
            set(f, "phase_" + gs.getCurrentPhase().name(), 1);

        // Hands and piles exist from the deal; before it, GameState throws rather than answer empty.
        List<? extends PhysicalCard> hand = zone(() -> gs.getHand(me));
        set(f, "myHand", hand.size());
        for (PhysicalCard card : hand) {
            var bp = card.getBlueprint();
            if (bp.getSide() == Side.FREE_PEOPLE)
                add(f, "myHandFp", 1);
            if (bp.getSide() == Side.SHADOW) {
                add(f, "myHandShadow", 1);
                add(f, "myHandShadowCost", bp.getTwilightCost());
            }
            if (bp.getCardType() == CardType.MINION)
                add(f, "myHandMinions", 1);
            if (bp.getCardType() == CardType.EVENT && bp.hasTimeword(Timeword.SKIRMISH))
                add(f, "myHandSkirmishEvents", 1);
        }
        if (opp != null) {
            set(f, "oppHand", zone(() -> gs.getHand(opp)).size());
            set(f, "oppDeck", zone(() -> gs.getDeck(opp)).size());
            set(f, "oppDiscard", zone(() -> gs.getDiscard(opp)).size());
        }
        set(f, "myDeck", zone(() -> gs.getDeck(me)).size());
        set(f, "myDiscard", zone(() -> gs.getDiscard(me)).size());

        if (fp == null || game == null || !gs.isInit()) {
            set(f, "pregame", 1);
            return f;
        }
        ModifiersQuerying mq = game.getModifiersQuerying();
        String shadow = gs.getPlayerNames().stream().filter(p -> !p.equals(fp)).findFirst().orElse(null);

        set(f, "turn", gs.getTurnNumber());
        set(f, "site", gs.getCurrentSiteNumber());
        set(f, "myPosition", gs.getPlayerPosition(me));
        if (opp != null)
            set(f, "oppPosition", gs.getPlayerPosition(opp));
        set(f, "twilight", gs.getTwilightPool());
        set(f, "moves", gs.getMoveCount());
        GameStats stats = gs.getMostRecentGameStats();
        if (stats != null)
            set(f, "moveLimit", stats.getMoveLimit());
        set(f, "burdens", gs.getBurdens());
        set(f, "threats", gs.getThreats());
        set(f, "wearingRing", gs.isWearingRing() ? 1 : 0);
        set(f, "fpInitiative", mq.hasInitiative(game) == Side.FREE_PEOPLE ? 1 : 0);
        PhysicalCard rb = gs.getRingBearer(fp);
        if (rb != null) {
            set(f, "rbVitalityLeft", mq.getVitality(game, rb));
            set(f, "rbResistanceLeft", mq.getResistance(game, rb));
        }

        int minVitality = Integer.MAX_VALUE;
        for (PhysicalCard card : gs.getInPlay()) {
            if (!gs.isCardInPlayActive(card))
                continue;
            var bp = card.getBlueprint();
            CardType type = bp.getCardType();
            boolean fpSide = card.getOwner().equals(fp);
            if (type == CardType.COMPANION && fpSide) {
                int left = mq.getVitality(game, card);
                add(f, "fpCompanions", 1);
                add(f, "fpCompanionStrength", mq.getStrength(game, card));
                add(f, "fpCompanionVitalityLeft", left);
                minVitality = Math.min(minVitality, left);
                if (left <= 1)
                    add(f, "fpExhausted", 1);
            } else if (type == CardType.ALLY && fpSide) {
                add(f, "fpAllies", 1);
            } else if (type == CardType.MINION) {
                int strength = mq.getStrength(game, card);
                add(f, "minions", 1);
                add(f, "minionStrength", strength);
                add(f, "minionVitalityLeft", mq.getVitality(game, card));
                f[index("maxMinionStrength")] = Math.max(f[index("maxMinionStrength")], strength);
                if (mq.hasKeyword(game, card, Keyword.FIERCE))
                    add(f, "fierceMinions", 1);
                if (mq.hasKeyword(game, card, Keyword.ARCHER))
                    add(f, "archerMinions", 1);
                add(f, "minionDamage", mq.getKeywordCount(game, card, Keyword.DAMAGE));
            } else if (type == CardType.CONDITION) {
                add(f, bp.getSide() == Side.SHADOW ? "shadowConditions" : "fpConditions", 1);
            } else if (type == CardType.POSSESSION || type == CardType.ARTIFACT) {
                add(f, bp.getSide() == Side.SHADOW ? "shadowPossessions" : "fpPossessions", 1);
            }
        }
        set(f, "fpMinVitalityLeft", minVitality == Integer.MAX_VALUE ? 0 : minVitality);

        if (opp != null) {
            for (String player : List.of(me, opp)) {
                String who = player.equals(me) ? "my" : "opp";
                set(f, who + "Burdens", gs.getPlayerBurdens(player));
                PhysicalCard bearer = gs.getRingBearer(player);
                if (bearer != null)
                    set(f, who + "RbResistance", mq.getResistance(game, bearer));
                set(f, who + "Dead", zone(() -> gs.getDeadPile(player)).size());
            }
            // active or not: the point is the fellowship each player will have on their own turn
            for (PhysicalCard card : gs.getInPlay()) {
                if (card.getBlueprint().getCardType() != CardType.COMPANION)
                    continue;
                String who = card.getOwner().equals(me) ? "my" : "opp";
                int wounds = gs.getWounds(card);
                add(f, who + "Companions", 1);
                add(f, who + "CompanionVitalityLeft", mq.getVitality(game, card));
                add(f, who + "Wounds", wounds);
            }
        }
        set(f, "fpDeadCompanions", gs.getDeadPile(fp).size());
        set(f, "shadowAdvantage", f[index("minionStrength")] - f[index("fpCompanionStrength")]);

        Skirmish skirmish = gs.getSkirmish();
        if (skirmish != null && skirmish.getFellowshipCharacter() != null && stats != null) {
            int margin = stats.getFellowshipSkirmishStrength() - stats.getShadowSkirmishStrength();
            PhysicalCard fpChar = skirmish.getFellowshipCharacter();
            set(f, "inSkirmish", 1);
            set(f, "mySkirmishMargin", iAmFp ? margin : -margin);
            set(f, "fpSkirmishDamage", stats.getFellowshipSkirmishDamageBonus());
            set(f, "shadowSkirmishDamage", stats.getShadowSkirmishDamageBonus());
            set(f, "overwhelmed", stats.isFpOverwhelmed() ? 1 : 0);
            set(f, "skirmishFpVitalityLeft", mq.getVitality(game, fpChar));
            set(f, "skirmishMinions", skirmish.getShadowCharacters().size());
        }
        return f;
    }

    private static List<? extends PhysicalCard> zone(java.util.function.Supplier<List<? extends PhysicalCard>> read) {
        try {
            List<? extends PhysicalCard> cards = read.get();
            return cards == null ? List.of() : cards;
        } catch (NullPointerException e) {
            return List.of();
        }
    }

    private static void set(double[] f, String name, double value) {
        f[index(name)] = value;
    }

    private static void add(double[] f, String name, double value) {
        f[index(name)] += value;
    }
}
