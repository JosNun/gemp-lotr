package com.gempukku.lotro.logic.modifiers;

import com.gempukku.lotro.common.*;
import com.gempukku.lotro.filters.Filters;
import com.gempukku.lotro.game.ExtraPlayCost;
import com.gempukku.lotro.game.LotroCardBlueprint;
import com.gempukku.lotro.game.PhysicalCard;
import com.gempukku.lotro.game.state.LotroGame;
import com.gempukku.lotro.logic.actions.AssignmentCostAction;
import com.gempukku.lotro.logic.actions.CostToEffectAction;
import com.gempukku.lotro.logic.modifiers.evaluator.Evaluator;
import com.gempukku.lotro.logic.timing.Action;

import java.util.*;

public class ModifiersLogic implements ModifiersEnvironment, ModifiersQuerying {
    private final Map<ModifierEffect, List<Modifier>> _modifiers = new HashMap<>();

    private final Map<Phase, List<Modifier>> _untilStartOfPhaseModifiers = new HashMap<>();
    private final Map<Phase, List<Modifier>> _untilEndOfPhaseModifiers = new HashMap<>();
    private final List<Modifier> _untilEndOfTurnModifiers = new LinkedList<>();
    // The modifiers an effect has set up for a time (until the end of a phase or turn, until the start of a phase):
    // the result of an effect already carried out, not any card's game text, so a card whose text is taken away
    // afterwards keeps them (a bonus its own ability gave it before the Star-glass, 10_13).
    private final Set<Modifier> _modifiersOfEffects = Collections.newSetFromMap(new IdentityHashMap<>());

    // The modifiers left out of the questions asked while one of them is worked out: a handful at most, asked
    // about for every modifier of every question -- an array compared by identity, where a HashSet's lookups and
    // inserts were 4% of a bot game's CPU (JFR, 2026-11-20). No modifier overrides equals.
    private final SkipSet _skipSet = new SkipSet();

    /*
     * Answers kept while the game stands still (2026-11-20).
     *
     * The game asks the same questions over and over with nothing changed between: in 792 bot games a character's
     * vitality was asked 323 million times, 92% of them a question already answered since the last change; "is
     * this card of that type" 5.8 billion times, 93%; whether a card has a keyword 726 million, 90%.  Each answer
     * walks every modifier of its kind and works out its condition and its filter.
     *
     * So the answers to the commonest questions -- strength, vitality, resistance, text removed, has-keyword,
     * keyword count, card type -- are kept by card, each stamped with the count of changes it was worked out at:
     * GameState's (every method there that changes the board, the cards and skirmishes it made, what an action
     * remembers, questions asked "as if") plus this class's own (modifiers added and removed, limits used, wounds
     * and draws counted).  A stamp that is not the count now is no answer.
     *
     * An answer may depend on what was being worked out around it: a modifier whose own condition is being worked
     * out is left out of the questions asked inside (the skip set).  So an answer is kept only if working it out
     * met no modifier that was left out, and it is remembered which modifiers it did ask about (a 256-bit mask by
     * identity hash); it is used again only where none of the modifiers now left out is among them -- then the
     * working out would have gone exactly as it did.  A false match in the mask only costs working it out again.
     *
     * Nothing is kept while a modifier whose condition remembers (SelfSpottingCondition) is in the game: working
     * that out is itself a change.
     *
     * -Dgemp.modifierMemo=false works everything out every time, as before.  -Dgemp.modifierMemo=verify works every
     * kept answer out again as it is used and throws if they differ: a change to the game that the counts miss
     * shows there, at the first stale answer.
     */
    private static final String MEMO_MODE = System.getProperty("gemp.modifierMemo", "true");
    private static final boolean MEMO = !"false".equals(MEMO_MODE), VERIFY = "verify".equals(MEMO_MODE);
    private static final int K_STRENGTH = 0, K_VITALITY = 1, K_RESISTANCE = 2, K_TEXT_REMOVED = 3, K_CARD_TYPE = 4,
            K_HAS_KEYWORD = 5, K_KEYWORD_COUNT = 6;
    private static final int SLOT_CARD_TYPE = 4, SLOT_HAS_KEYWORD = SLOT_CARD_TYPE + CardType.values().length,
            SLOT_KEYWORD_COUNT = SLOT_HAS_KEYWORD + Keyword.values().length,
            SLOTS = SLOT_KEYWORD_COUNT + Keyword.values().length;

    private static final class Memo {
        private long epoch = -1;
        private int value;
        private long met0, met1, met2, met3;
    }

    // This class's own count of changes, shared with the limit counters it hands out.
    private final long[] _modifierEpoch = new long[1];
    private int _selfSpotting;
    // Not part of the game: left behind when a game is copied, and built again as questions are asked.
    private transient Map<PhysicalCard, Memo[]> _memos;
    private transient long _met0, _met1, _met2, _met3;
    private transient long[] _frames;
    private transient int _frameTop;
    private transient boolean _verifying;

    private void countSelfSpotting() {
        int count = 0;
        for (List<Modifier> list : _modifiers.values())
            for (Modifier modifier : list)
                if (modifier.getCondition() instanceof SelfSpottingCondition)
                    count++;
        _selfSpotting = count;
    }

    private static int maskBit(Modifier modifier) {
        final int hash = System.identityHashCode(modifier);
        return (hash ^ (hash >>> 8) ^ (hash >>> 16)) & 255;
    }

    /** Whether a modifier is left out just now; notes that the answer being worked out asked. */
    private boolean skipped(Modifier modifier) {
        final int bit = maskBit(modifier);
        switch (bit >> 6) {
            case 0 -> _met0 |= 1L << bit;
            case 1 -> _met1 |= 1L << bit;
            case 2 -> _met2 |= 1L << bit;
            default -> _met3 |= 1L << bit;
        }
        return _skipSet.contains(modifier);
    }

    private Memo recall(LotroGame game, PhysicalCard card, int slot) {
        if (_memos == null)
            return null;
        final Memo[] ofCard = _memos.get(card);
        if (ofCard == null)
            return null;
        final Memo memo = ofCard[slot];
        if (memo == null || memo.epoch != game.getGameState().getQueryEpoch() + _modifierEpoch[0])
            return null;
        for (int i = 0; i < _skipSet._size; i++) {
            final int bit = maskBit(_skipSet._modifiers[i]);
            final long word = switch (bit >> 6) {
                case 0 -> memo.met0;
                case 1 -> memo.met1;
                case 2 -> memo.met2;
                default -> memo.met3;
            };
            if ((word & (1L << bit)) != 0)
                return null;
        }
        // what it asked about, the answer being worked out around it has now asked about too
        _met0 |= memo.met0;
        _met1 |= memo.met1;
        _met2 |= memo.met2;
        _met3 |= memo.met3;
        return memo;
    }

    private int workOut(LotroGame game, PhysicalCard card, int kind, Object about) {
        return switch (kind) {
            case K_STRENGTH -> workOutGetStrength(game, card);
            case K_VITALITY -> workOutGetVitality(game, card);
            case K_RESISTANCE -> workOutGetResistance(game, card);
            case K_TEXT_REMOVED -> workOutHasTextRemoved(game, card) ? 1 : 0;
            case K_CARD_TYPE -> workOutIsCardType(game, card, (CardType) about) ? 1 : 0;
            case K_HAS_KEYWORD -> workOutHasKeyword(game, card, (Keyword) about) ? 1 : 0;
            default -> workOutGetKeywordCount(game, card, (Keyword) about);
        };
    }

    private int ask(LotroGame game, PhysicalCard card, int slot, int kind, Object about) {
        if (!MEMO || _verifying || _selfSpotting > 0 || card == null)
            return workOut(game, card, kind, about);
        final Memo memo = recall(game, card, slot);
        if (memo != null) {
            if (VERIFY) {
                final int fresh;
                _verifying = true;
                try {
                    fresh = workOut(game, card, kind, about);
                } finally {
                    _verifying = false;
                }
                if (fresh != memo.value)
                    throw new IllegalStateException("A kept answer is stale: question " + kind
                            + (about == null ? "" : " " + about) + " of " + card.getBlueprintId() + " (card "
                            + card.getCardId() + ", " + card.getZone() + ") kept as " + memo.value + ", worked out as "
                            + fresh + " in " + game.getGameState().getCurrentPhase());
            }
            return memo.value;
        }
        // a frame: what the answer around this one had asked about so far, the left-out modifiers met, the count
        if (_frames == null)
            _frames = new long[6 * 16];
        else if (_frameTop + 6 > _frames.length)
            _frames = Arrays.copyOf(_frames, _frames.length * 2);
        final int frame = _frameTop;
        final long epoch = game.getGameState().getQueryEpoch() + _modifierEpoch[0];
        final long hits = _skipSet._hits;
        _frames[frame] = _met0;
        _frames[frame + 1] = _met1;
        _frames[frame + 2] = _met2;
        _frames[frame + 3] = _met3;
        _frameTop = frame + 6;
        _met0 = _met1 = _met2 = _met3 = 0;
        try {
            final int value = workOut(game, card, kind, about);
            if (_skipSet._hits == hits && _selfSpotting == 0
                    && epoch == game.getGameState().getQueryEpoch() + _modifierEpoch[0]) {
                if (_memos == null)
                    _memos = new IdentityHashMap<>();
                Memo[] ofCard = _memos.get(card);
                if (ofCard == null)
                    _memos.put(card, ofCard = new Memo[SLOTS]);
                Memo kept = ofCard[slot];
                if (kept == null)
                    ofCard[slot] = kept = new Memo();
                kept.epoch = epoch;
                kept.value = value;
                kept.met0 = _met0;
                kept.met1 = _met1;
                kept.met2 = _met2;
                kept.met3 = _met3;
            }
            return value;
        } finally {
            _met0 |= _frames[frame];
            _met1 |= _frames[frame + 1];
            _met2 |= _frames[frame + 2];
            _met3 |= _frames[frame + 3];
            _frameTop = frame;
        }
    }


    private static final class SkipSet {
        private Modifier[] _modifiers = new Modifier[8];
        private int _size;

        // How often a question met a modifier that was left out: an answer worked out without meeting one does not
        // depend on what was being worked out around it.
        private long _hits;

        boolean contains(Modifier modifier) {
            if (has(modifier)) {
                _hits++;
                return true;
            }
            return false;
        }

        private boolean has(Modifier modifier) {
            for (int i = _size - 1; i >= 0; i--)
                if (_modifiers[i] == modifier)
                    return true;
            return false;
        }

        void add(Modifier modifier) {
            if (has(modifier))
                return;
            if (_size == _modifiers.length)
                _modifiers = Arrays.copyOf(_modifiers, _size * 2);
            _modifiers[_size++] = modifier;
        }

        void remove(Modifier modifier) {
            for (int i = _size - 1; i >= 0; i--)
                if (_modifiers[i] == modifier) {
                    System.arraycopy(_modifiers, i + 1, _modifiers, i, _size - i - 1);
                    _modifiers[--_size] = null;
                    return;
                }
        }
    }
    // The modifiers isInForce is working out, innermost last; the first _workingOutDepth entries are in use.
    private final List<WorkingOut> _workingOut = new ArrayList<>();
    private int _workingOutDepth = 0;

    private final Map<Phase, Map<String, LimitCounter>> _endOfPhaseLimitCounters = new HashMap<>();
    private final Map<Phase, Map<String, LimitCounter>> _startOfPhaseLimitCounters = new HashMap<>();
    private final Map<String, LimitCounter> _turnLimitCounters = new HashMap<>();
    // "Limit once per site": cleared when the fellowship moves to another site, and with the turn.
    private final Map<String, LimitCounter> _siteLimitCounters = new HashMap<>();

    private int _drawnThisPhaseCount = 0;
    private final Map<Integer, Integer> _woundsPerPhaseMap = new HashMap<>();

    @Override
    public LimitCounter getUntilEndOfPhaseLimitCounter(PhysicalCard card, Phase phase) {
        return getUntilEndOfPhaseLimitCounter(card, "", phase);
    }

    @Override
    public LimitCounter getUntilStartOfPhaseLimitCounter(PhysicalCard card, Phase phase) {
        return getUntilStartOfPhaseLimitCounter(card, "", phase);
    }

    @Override
    public LimitCounter getUntilEndOfPhaseLimitCounter(PhysicalCard card, String prefix, Phase phase) {
        Map<String, LimitCounter> limitCounterMap = _endOfPhaseLimitCounters.get(phase);
        if (limitCounterMap == null) {
            limitCounterMap = new HashMap<>();
            _endOfPhaseLimitCounters.put(phase, limitCounterMap);
        }
        LimitCounter limitCounter = limitCounterMap.get(prefix + card.getCardId());
        if (limitCounter == null) {
            limitCounter = new DefaultLimitCounter(_modifierEpoch);
            limitCounterMap.put(prefix + card.getCardId(), limitCounter);
        }
        return limitCounter;
    }

    @Override
    public LimitCounter getUntilStartOfPhaseLimitCounter(PhysicalCard card, String prefix, Phase phase) {
        Map<String, LimitCounter> limitCounterMap = _startOfPhaseLimitCounters.get(phase);
        if (limitCounterMap == null) {
            limitCounterMap = new HashMap<>();
            _startOfPhaseLimitCounters.put(phase, limitCounterMap);
        }
        LimitCounter limitCounter = limitCounterMap.get(prefix + card.getCardId());
        if (limitCounter == null) {
            limitCounter = new DefaultLimitCounter(_modifierEpoch);
            limitCounterMap.put(prefix + card.getCardId(), limitCounter);
        }
        return limitCounter;
    }

    @Override
    public LimitCounter getUntilEndOfTurnLimitCounter(PhysicalCard card) {
        return getUntilEndOfTurnLimitCounter(card, "");
    }

    @Override
    public LimitCounter getUntilEndOfTurnLimitCounter(PhysicalCard card, String prefix) {
        LimitCounter limitCounter = _turnLimitCounters.get(prefix + card.getCardId());
        if (limitCounter == null) {
            limitCounter = new DefaultLimitCounter(_modifierEpoch);
            _turnLimitCounters.put(prefix + card.getCardId(), limitCounter);
        }
        return limitCounter;
    }

    @Override
    public LimitCounter getSiteLimitCounter(PhysicalCard card) {
        return getSiteLimitCounter(card, "");
    }

    @Override
    public LimitCounter getSiteLimitCounter(PhysicalCard card, String prefix) {
        LimitCounter limitCounter = _siteLimitCounters.get(prefix + card.getCardId());
        if (limitCounter == null) {
            limitCounter = new DefaultLimitCounter(_modifierEpoch);
            _siteLimitCounters.put(prefix + card.getCardId(), limitCounter);
        }
        return limitCounter;
    }

    /**
     * The fellowship has moved to another site: what was counted "per site" was counted at the site it left.
     * An exertion on arriving, which comes after this, belongs to the site arrived at (F-0146).
     */
    public void signalFellowshipMoved() {
        _modifierEpoch[0]++;
        _siteLimitCounters.clear();
    }

    @Override
    public void addedWound(PhysicalCard card) {
        _modifierEpoch[0]++;
        final int cardId = card.getCardId();
        final Integer previousWounds = _woundsPerPhaseMap.get(cardId);
        if (previousWounds == null)
            _woundsPerPhaseMap.put(cardId, 1);
        else
            _woundsPerPhaseMap.put(cardId, previousWounds + 1);
    }

    private List<Modifier> getEffectModifiers(ModifierEffect modifierEffect) {
        List<Modifier> modifiers = _modifiers.get(modifierEffect);
        if (modifiers == null) {
            modifiers = new LinkedList<>();
            _modifiers.put(modifierEffect, modifiers);
        }
        return modifiers;
    }

    private void removeModifiers(List<Modifier> modifiers) {
        _modifierEpoch[0]++;
        for (List<Modifier> list : _modifiers.values())
            list.removeAll(modifiers);
        modifiers.forEach(_modifiersOfEffects::remove);
        countSelfSpotting();
    }

    private void removeModifier(Modifier modifier) {
        _modifierEpoch[0]++;
        for (List<Modifier> list : _modifiers.values())
            list.remove(modifier);
        _modifiersOfEffects.remove(modifier);
        countSelfSpotting();
    }

    @Override
    public ModifierHook addAlwaysOnModifier(Modifier modifier) {
        addModifier(modifier);
        return new ModifierHookImpl(modifier);
    }

    private void addModifier(Modifier modifier) {
        _modifierEpoch[0]++;
        ModifierEffect modifierEffect = modifier.getModifierEffect();
        getEffectModifiers(modifierEffect).add(modifier);
        if (modifier.getCondition() instanceof SelfSpottingCondition)
            _selfSpotting++;
    }

    private List<Modifier> getModifiers(LotroGame game, ModifierEffect modifierEffect) {
        return getKeywordModifiersAffectingCard(game, modifierEffect, null, null);
    }

    private List<Modifier> getModifiersAffectingCard(LotroGame game, ModifierEffect modifierEffect, PhysicalCard card) {
        return getKeywordModifiersAffectingCard(game, modifierEffect, null, card);
    }

    /**
     * The modifiers that say what the SOURCE of an effect may not do: "burdens cannot be added", "threats cannot
     * be removed by Free Peoples cards", "Shadow cards may not discard cards from your hand".  They are asked
     * about the card whose text is being carried out, and they hold whether or not that card is hindered: a card
     * that hinders itself as a cost, or is hindered in answer to its ability, still carries the ability out
     * (CR 5.0 "hinder"), and "cannot" binds what it then does (Ketura on Q-0070, "can't > can"; F-0151).
     * getModifiersAffectingCard answers with nothing for a hindered card, which is right for what the card
     * itself has (its text, its keywords, the bonuses on it) and wrong for what its effect is forbidden.
     */
    private List<Modifier> getModifiersAffectingSource(LotroGame game, ModifierEffect modifierEffect, PhysicalCard source) {
        return getKeywordModifiersAffectingCard(game, modifierEffect, null, source, true);
    }

    /**
     * The modifiers that keep a card from being taken out of play: "cannot be discarded", "cannot be returned to
     * hand".  For a card that is not hindered these are all of them, as getModifiersAffectingCard gives them.  A
     * hindered card gets nothing from other cards, their protections included (CR 5.0 "hinder"), but what the
     * RULES forbid about it still holds: "The Ring-bearer cannot be discarded or returned to your hand" (CR 4.2
     * "Ring-bearer") is no card's text, and a discard that reaches hindered cards must not take a hindered
     * Ring-bearer and lose the game (F-0174).  A rule's modifier is one with no source card (RingBearerRule and
     * the other rule classes add theirs so).  This is the counterpart of getModifiersAffectingSource: there, what
     * a hindered source is forbidden to do; here, what may not be done to a hindered card by rule.
     */
    private List<Modifier> getModifiersProtectingCard(LotroGame game, ModifierEffect modifierEffect, PhysicalCard card) {
        if (card == null || !game.getGameState().isHindered(card))
            return getModifiersAffectingCard(game, modifierEffect, card);

        List<Modifier> rules = new LinkedList<>();
        for (Modifier modifier : getKeywordModifiersAffectingCard(game, modifierEffect, null, card, true)) {
            if (modifier.getSource() == null)
                rules.add(modifier);
        }
        return rules;
    }

    private List<Modifier> getKeywordModifiersAffectingCard(LotroGame game, ModifierEffect modifierEffect, Keyword keyword, PhysicalCard card) {
        return getKeywordModifiersAffectingCard(game, modifierEffect, keyword, card, false);
    }

    private List<Modifier> getKeywordModifiersAffectingCard(LotroGame game, ModifierEffect modifierEffect, Keyword keyword, PhysicalCard card, boolean evenIfHindered) {
        List<Modifier> modifiers = _modifiers.get(modifierEffect);
        if (modifiers == null || modifiers.isEmpty())
            return Collections.emptyList();

        //A hindered card has no text and nothing applies to it.
        //If we are checking that the card has been hindered, we don't want to abort early
        if(!evenIfHindered && card != null && keyword != Keyword.HINDERED && game.getGameState().isHindered(card))
            return Collections.emptyList();

        List<Modifier> liveModifiers = new ArrayList<>();
        for (Modifier modifier : modifiers) {
            boolean keywordMatches = false;

            if(modifier instanceof KeywordAffectingModifier keyModifier) {
                keywordMatches = keyModifier.getKeywords() == null || keyModifier.getKeywords().contains(keyword);
            }
            else if(modifier instanceof DelegateModifier delegateModifier) {
                if(delegateModifier.delegate instanceof KeywordAffectingModifier keyModifier) {
                    keywordMatches = keyModifier.getKeywords() == null || keyModifier.getKeywords().contains(keyword);
                }
            }

            if (keyword == null || keywordMatches) {
                if (!skipped(modifier)) {
                    if (isInForce(game, modifierEffect, modifier, card))
                        liveModifiers.add(modifier);
                } else if (isAssumedInForce(modifier, card)) {
                    // Worked out as if it were in force, to see whether its answer leans on itself (CR 4.2 "loops").
                    liveModifiers.add(modifier);
                } else if (isHeldWhileBeingWorkedOut(game, modifier, modifierEffect)
                        && (card == null || modifier.affectsCard(game, card))) {
                    // Asked about while its own condition is being worked out: in force if it held (Q-0276).
                    liveModifiers.add(modifier);
                }
            }
        }

        return liveModifiers;
    }

    /**
     * Whether a modifier is in force for a card (or, with no card, at all): its condition holds, the text of its
     * card has not been taken away, and it affects the card.  While this is worked out the modifier is left out of
     * every question asked inside it (the skip set), so that two texts that read each other do not ask each other
     * for ever.
     *
     * That alone gives the wrong answer where the two texts undo each other.  CR 4.2 "loops": "When an infinite
     * loop occurs that includes no voluntary actions, the loop ends immediately with no effect", with Frodo, Weary
     * From the Journey 15_144 skirmishing Black Land Observer 15_101 or Wandering Hillman 15_94: "Neither the
     * minions nor Frodo would get the strength bonuses."  Frodo's +2 makes him 10, which gives the Observer +3,
     * which takes the +2 away; worked out with his own +2 left out, each of the two saw the other without its bonus
     * and both applied (F-0536).
     *
     * So when the working out asked about this very modifier for this very card again, and the answer is that it is
     * in force, it is worked out a second time as if it were in force.  If it then is not, its answer undoes itself,
     * which is the loop going round, and it is not in force: no effect.  If it still is, its answer does not lean on
     * itself and stands.  A modifier that is not in force with itself left out stays so either way, so a text that
     * holds only because it holds is not in force, as before.  A loop of texts that answer the same either way
     * (Frodo at 10 before his bonus: the Observer is 11 whatever Frodo gets) is no loop, and the texts apply.  A
     * modifier asked about again only for ANOTHER card is not worked out twice: it is left out there, as before.
     * The second working out costs nothing where no text reads itself back.  A condition that counts its own grant
     * (SelfSpottingCondition, Q-0276) is held as it was and is not worked out twice.
     */
    private boolean isInForce(LotroGame game, ModifierEffect modifierEffect, Modifier modifier, PhysicalCard card) {
        final WorkingOut workingOut = pushWorkingOut(modifier, card);
        _skipSet.add(modifier);
        try {
            boolean inForce = worksOutInForce(game, modifierEffect, modifier, card);
            if (inForce && workingOut.askedAgain && !(modifier.getCondition() instanceof SelfSpottingCondition)) {
                workingOut.assumedInForce = true;
                inForce = worksOutInForce(game, modifierEffect, modifier, card);
            }
            return inForce;
        } finally {
            _skipSet.remove(modifier);
            popWorkingOut();
        }
    }

    private boolean worksOutInForce(LotroGame game, ModifierEffect modifierEffect, Modifier modifier, PhysicalCard card) {
        Condition condition = modifier.getCondition();
        if (condition != null && !condition.isFullfilled(game))
            return false;
        if (modifierEffect != ModifierEffect.TEXT_MODIFIER && modifier.getSource() != null
                && !modifier.isNonCardTextModifier() && !_modifiersOfEffects.contains(modifier)
                && hasTextRemoved(game, modifier.getSource()))
            return false;
        return card == null || modifier.affectsCard(game, card);
    }

    /**
     * A modifier that isInForce is working out, met again inside its own working out: notes that it was met again
     * if it is met for the same card, and answers whether it is being worked out as if in force.  A modifier left
     * out by any other part of this class (affectsCardWithSkipSet, hasKeywordAsGiven) has no entry and answers false.
     */
    private boolean isAssumedInForce(Modifier modifier, PhysicalCard card) {
        for (int i = _workingOutDepth - 1; i >= 0; i--) {
            final WorkingOut workingOut = _workingOut.get(i);
            if (workingOut.modifier == modifier) {
                if (workingOut.card != card)
                    return false;
                workingOut.askedAgain = true;
                return workingOut.assumedInForce;
            }
        }
        return false;
    }

    private WorkingOut pushWorkingOut(Modifier modifier, PhysicalCard card) {
        final WorkingOut workingOut;
        if (_workingOutDepth == _workingOut.size()) {
            workingOut = new WorkingOut();
            _workingOut.add(workingOut);
        } else {
            workingOut = _workingOut.get(_workingOutDepth);
        }
        _workingOutDepth++;
        workingOut.modifier = modifier;
        workingOut.card = card;
        workingOut.assumedInForce = false;
        workingOut.askedAgain = false;
        return workingOut;
    }

    private void popWorkingOut() {
        final WorkingOut workingOut = _workingOut.get(--_workingOutDepth);
        workingOut.modifier = null;
        workingOut.card = null;
    }

    /** One modifier that isInForce is working out; kept in a stack that is reused, as this is asked very often. */
    private static final class WorkingOut {
        private Modifier modifier;
        private PhysicalCard card;
        private boolean assumedInForce;
        private boolean askedAgain;
    }

    private static boolean namesKeyword(Modifier modifier, Keyword keyword) {
        Modifier named = modifier instanceof DelegateModifier delegateModifier ? delegateModifier.delegate : modifier;
        return named instanceof KeywordAffectingModifier keyModifier && keyModifier.getKeywords() != null
                && keyModifier.getKeywords().contains(keyword);
    }

    /**
     * A modifier whose condition counts its own grant once it has held (SelfSpottingCondition: Sentry Uruk 15_170,
     * Q-0276), asked about while that condition is being worked out: in force if the condition held when it was
     * last worked out and the text of its card has not been taken away.  Every other modifier is left out while
     * it is being worked out, as before.
     */
    private boolean isHeldWhileBeingWorkedOut(LotroGame game, Modifier modifier, ModifierEffect modifierEffect) {
        if (!(modifier.getCondition() instanceof SelfSpottingCondition selfSpotting) || !selfSpotting.hasHeld())
            return false;
        return modifierEffect == ModifierEffect.TEXT_MODIFIER || modifier.getSource() == null
                || modifier.isNonCardTextModifier() || _modifiersOfEffects.contains(modifier)
                || !hasTextRemoved(game, modifier.getSource());
    }

    /**
     * Works out, at each step of the game, the condition of every modifier that counts its own grant once it has
     * held (SelfSpottingCondition), so that what each remembers follows the board whether or not anything asked
     * about its card in between: a fierce minion that comes into play and is killed before anything asks whether
     * Sentry Uruk 15_170 is fierce has still made it fierce (Q-0276).  Called from TurnProcedure beside the other
     * state-based checks.
     */
    @Override
    public void checkSelfSpottingConditions(LotroGame game) {
        List<Modifier> modifiers = _modifiers.get(ModifierEffect.GIVE_KEYWORD_MODIFIER);
        if (modifiers == null)
            return;
        // Gathered first: working a condition out reads the modifiers, and in nearly every game there is none.
        List<Modifier> selfSpotting = null;
        for (Modifier modifier : modifiers) {
            if (modifier.getCondition() instanceof SelfSpottingCondition) {
                if (selfSpotting == null)
                    selfSpotting = new ArrayList<>();
                selfSpotting.add(modifier);
            }
        }
        if (selfSpotting == null)
            return;
        for (Modifier modifier : selfSpotting) {
            if (!skipped(modifier)) {
                _skipSet.add(modifier);
                try {
                    modifier.getCondition().isFullfilled(game);
                } finally {
                    _skipSet.remove(modifier);
                }
            }
        }
    }

    public void signalEndOfPhase(Phase phase) {
        _modifierEpoch[0]++;
        List<Modifier> list = _untilEndOfPhaseModifiers.get(phase);
        if (list != null) {
            removeModifiers(list);
            list.clear();
        }
        Map<String, LimitCounter> counterMap = _endOfPhaseLimitCounters.get(phase);
        if (counterMap != null)
            counterMap.clear();

        _drawnThisPhaseCount = 0;
        _woundsPerPhaseMap.clear();
    }

    public void signalStartOfPhase(Phase phase) {
        _modifierEpoch[0]++;
        List<Modifier> list = _untilStartOfPhaseModifiers.get(phase);
        if (list != null) {
            removeModifiers(list);
            list.clear();
        }

        Map<String, LimitCounter> counterMap = _startOfPhaseLimitCounters.get(phase);
        if (counterMap != null)
            counterMap.clear();
    }

    public void signalEndOfTurn() {
        _modifierEpoch[0]++;
        removeModifiers(_untilEndOfTurnModifiers);
        _untilEndOfTurnModifiers.clear();

        for (List<Modifier> modifiers : _untilStartOfPhaseModifiers.values())
            removeModifiers(modifiers);
        _untilStartOfPhaseModifiers.clear();

        for (List<Modifier> modifiers : _untilEndOfPhaseModifiers.values())
            removeModifiers(modifiers);
        _untilEndOfPhaseModifiers.clear();

        _turnLimitCounters.clear();
        _siteLimitCounters.clear();
        _startOfPhaseLimitCounters.clear();
        _endOfPhaseLimitCounters.clear();
    }

    @Override
    public void addUntilEndOfPhaseModifier(Modifier modifier, Phase phase) {
        _modifierEpoch[0]++;
        addModifier(modifier);
        _modifiersOfEffects.add(modifier);
        List<Modifier> list = _untilEndOfPhaseModifiers.get(phase);
        if (list == null) {
            list = new LinkedList<>();
            _untilEndOfPhaseModifiers.put(phase, list);
        }
        list.add(modifier);
    }

    @Override
    public void addUntilStartOfPhaseModifier(Modifier modifier, Phase phase) {
        _modifierEpoch[0]++;
        addModifier(modifier);
        _modifiersOfEffects.add(modifier);
        List<Modifier> list = _untilStartOfPhaseModifiers.get(phase);
        if (list == null) {
            list = new LinkedList<>();
            _untilStartOfPhaseModifiers.put(phase, list);
        }
        list.add(modifier);
    }

    @Override
    public void addUntilEndOfTurnModifier(Modifier modifier) {
        _modifierEpoch[0]++;
        addModifier(modifier);
        _modifiersOfEffects.add(modifier);
        _untilEndOfTurnModifiers.add(modifier);
    }

    @Override
    public Collection<Modifier> getModifiersAffecting(LotroGame game, PhysicalCard card) {
        Set<Modifier> result = new LinkedHashSet<>();
        for (List<Modifier> modifiers : _modifiers.values()) {
            for (Modifier modifier : modifiers) {
                Condition condition = modifier.getCondition();
                if (condition == null || condition.isFullfilled(game))
                    if (affectsCardWithSkipSet(game, card, modifier))
                        result.addAll(modifier.getShownAsAffecting(game, card));
            }
        }
        return result;
    }

    @Override
    public Evaluator getFPStrengthOverrideEvaluator(LotroGame game, PhysicalCard fpCharacter) {
        for (Modifier modifier : getModifiersAffectingCard(game, ModifierEffect.SKIRMISH_STRENGTH_EVALUATOR_MODIFIER, fpCharacter)) {
            Evaluator evaluator = modifier.getFpSkirmishStrengthOverrideEvaluator(game, fpCharacter);
            if (evaluator != null)
                return evaluator;
        }
        return null;
    }

    @Override
    public Evaluator getShadowStrengthOverrideEvaluator(LotroGame game, PhysicalCard shadowCharacter) {
        for (Modifier modifier : getModifiersAffectingCard(game, ModifierEffect.SKIRMISH_STRENGTH_EVALUATOR_MODIFIER, shadowCharacter)) {
            Evaluator evaluator = modifier.getShadowSkirmishStrengthOverrideEvaluator(game, shadowCharacter);
            if (evaluator != null)
                return evaluator;
        }
        return null;
    }

    private boolean affectsCardWithSkipSet(LotroGame game, PhysicalCard physicalCard, Modifier modifier) {
        if (!skipped(modifier) && physicalCard != null) {
            _skipSet.add(modifier);
            boolean result = modifier.affectsCard(game, physicalCard);
            _skipSet.remove(modifier);
            return result;
        } else {
            return false;
        }
    }

    @Override
    public boolean hasTextRemoved(LotroGame game, PhysicalCard card) {
        return ask(game, card, K_TEXT_REMOVED, K_TEXT_REMOVED, null) != 0;
    }

    private boolean workOutHasTextRemoved(LotroGame game, PhysicalCard card) {
        for (Modifier modifier : getModifiersAffectingCard(game, ModifierEffect.TEXT_MODIFIER, card)) {
            if (modifier.hasRemovedText(game, card))
                return true;
        }
        return false;
    }

    private boolean hasAllKeywordsRemoved(LotroGame game, PhysicalCard card) {
        for (Modifier modifier : getModifiersAffectingCard(game, ModifierEffect.LOSE_ALL_KEYWORDS_MODIFIER, card)) {
            if (modifier.lostAllKeywords(game, card))
                return true;
        }
        return false;
    }

    @Override
    public boolean hasKeyword(LotroGame game, PhysicalCard physicalCard, Keyword keyword) {
        return ask(game, physicalCard, SLOT_HAS_KEYWORD + keyword.ordinal(), K_HAS_KEYWORD, keyword) != 0;
    }

    private boolean workOutHasKeyword(LotroGame game, PhysicalCard physicalCard, Keyword keyword) {
        return hasKeywordAsGiven(game, physicalCard, keyword)
                && getKeywordGainLimit(game, physicalCard, keyword) > 0;
    }

    /**
     * The most of a keyword a card may have while it is barred from gaining keywords (CantGainKeywordsModifier,
     * Into the Caves 17_100; Q-0282): what it had as the bar was laid. Integer.MAX_VALUE with no bar, and for what
     * is no game text keyword (the keywords a lose-all removes: isCandidateForKeywordRemovalWithTextRemoval).
     */
    private int getKeywordGainLimit(LotroGame game, PhysicalCard physicalCard, Keyword keyword) {
        if (!isCandidateForKeywordRemovalWithTextRemoval(game, physicalCard, keyword))
            return Integer.MAX_VALUE;
        int limit = Integer.MAX_VALUE;
        for (Modifier modifier : getModifiersAffectingCard(game, ModifierEffect.CANT_GAIN_KEYWORDS_MODIFIER, physicalCard))
            limit = Math.min(limit, modifier.getKeywordGainLimit(game, physicalCard, keyword));
        return limit;
    }

    private boolean hasKeywordAsGiven(LotroGame game, PhysicalCard physicalCard, Keyword keyword) {
        LoggingThreadLocal.logMethodStart(physicalCard, "hasKeyword " + keyword.getHumanReadable());
        try {
            if (isCandidateForKeywordRemovalWithTextRemoval(game, physicalCard, keyword) &&
                    (hasTextRemoved(game, physicalCard) || hasAllKeywordsRemoved(game, physicalCard)))
                return false;

            for (Modifier modifier : getKeywordModifiersAffectingCard(game, ModifierEffect.REMOVE_KEYWORD_MODIFIER,
                    keyword, physicalCard)) {
                if (modifier.isKeywordRemoved(game, physicalCard, keyword))
                    return false;
            }

            if (physicalCard.getBlueprint().hasKeyword(keyword))
                return true;

            if(keyword == Keyword.HINDERED || !hasKeyword(game, physicalCard, Keyword.HINDERED)) {
                for (Modifier modifier : getKeywordModifiersAffectingCard(game, ModifierEffect.GIVE_KEYWORD_MODIFIER, keyword, physicalCard)) {
                    if (appliesKeywordModifier(game, physicalCard, modifier.getSource(), keyword)) {
                        if (!skipped(modifier)) {
                            _skipSet.add(modifier);
                            try {
                                if (modifier.hasKeyword(game, physicalCard, keyword))
                                    return true;
                            } finally {
                                _skipSet.remove(modifier);
                            }
                        } else if (modifier.getCondition() instanceof SelfSpottingCondition) {
                            // Only a modifier that counts its own grant reaches here while it is being worked
                            // out (isHeldWhileBeingWorkedOut): it gives what it held (Q-0276).
                            if (modifier.hasKeyword(game, physicalCard, keyword))
                                return true;
                        } else if (isAssumedInForce(modifier, physicalCard)) {
                            // Worked out as if in force (isInForce, CR 4.2 "loops"): it gives the keyword it names.
                            // One that names none (a copy of another card's keywords) is read as giving nothing,
                            // as it was before, and is not asked, as asking could come back here.
                            if (namesKeyword(modifier, keyword))
                                return true;
                        } else {
                            return false;
                        }
                    }
                }
            }

            return false;
        } finally {
            LoggingThreadLocal.logMethodEnd();
        }
    }

    @Override
    public int getKeywordCount(LotroGame game, PhysicalCard physicalCard, Keyword keyword) {
        return ask(game, physicalCard, SLOT_KEYWORD_COUNT + keyword.ordinal(), K_KEYWORD_COUNT, keyword);
    }

    private int workOutGetKeywordCount(LotroGame game, PhysicalCard physicalCard, Keyword keyword) {
        LoggingThreadLocal.logMethodStart(physicalCard, "getKeywordCount " + keyword.getHumanReadable());
        try {
            if (isCandidateForKeywordRemovalWithTextRemoval(game, physicalCard, keyword)
                    && (hasTextRemoved(game, physicalCard) || hasAllKeywordsRemoved(game, physicalCard)))
                return 0;

            for (Modifier modifier : getKeywordModifiersAffectingCard(game, ModifierEffect.REMOVE_KEYWORD_MODIFIER, keyword, physicalCard)) {
                if (modifier.isKeywordRemoved(game, physicalCard, keyword))
                    return 0;
            }

            int result = physicalCard.getBlueprint().getKeywordCount(keyword);
            for (Modifier modifier : getKeywordModifiersAffectingCard(game, ModifierEffect.GIVE_KEYWORD_MODIFIER, keyword, physicalCard)) {
                if (appliesKeywordModifier(game, physicalCard, modifier.getSource(), keyword))
                    result += modifier.getKeywordCountModifier(game, physicalCard, keyword);
            }
            return Math.min(Math.max(0, result), getKeywordGainLimit(game, physicalCard, keyword));
        } finally {
            LoggingThreadLocal.logMethodEnd();
        }
    }

    private boolean isCandidateForKeywordRemovalWithTextRemoval(LotroGame game, PhysicalCard physicalCard, Keyword keyword) {
        if (!keyword.isRealKeyword())
            return false;
        // Ring-bearer is ALWAYS Ring-bound and cannot lose that.
        // Frodo and Sam are deliberately NOT exempted: per ruling, Ring-bound is a keyword and keywords are game text,
        // so e.g. Helpless ("Sam's game text does not apply") leaves a non-Ring-bearer Sam unbound.
        if (keyword == Keyword.RING_BOUND && game.getGameState().getRingBearer(physicalCard.getOwner()) == physicalCard)
            return false;
        return true;
    }

    private boolean appliesKeywordModifier(LotroGame game, PhysicalCard affecting, PhysicalCard modifierSource, Keyword keyword) {
        if (modifierSource == null)
            return true;
        for (Modifier modifier : getKeywordModifiersAffectingCard(game, ModifierEffect.CANCEL_KEYWORD_BONUS_TARGET_MODIFIER, keyword, affecting)) {
            if (!modifier.appliesKeywordModifier(game, modifierSource, keyword))
                return false;
        }
        return true;
    }

    @Override
    public boolean hasSignet(LotroGame game, PhysicalCard physicalCard, Signet signet) {
        LoggingThreadLocal.logMethodStart(physicalCard, "hasSignet " + signet);
        try {
            if (physicalCard.getBlueprint().getSignet() == signet)
                return true;
            for (Modifier modifier : getModifiersAffectingCard(game, ModifierEffect.SIGNET_MODIFIER, physicalCard)) {
                if (modifier.hasSignet(game, physicalCard, signet))
                    return true;
            }
            return false;
        } finally {
            LoggingThreadLocal.logMethodEnd();
        }
    }

    @Override
    public int getArcheryTotal(LotroGame game, Side side, int baseArcheryTotal) {
        int result = baseArcheryTotal;
        for (Modifier modifier : getModifiers(game, ModifierEffect.ARCHERY_MODIFIER)) {
            if(addsToArcheryTotal(game, modifier.getSource())) {
                result += modifier.getArcheryTotalModifier(game, side);
            }
        }
        return Math.max(0, result);
    }

    @Override
    public int getHandSize(LotroGame game, String playerId) {
        int result = game.getFormat().getHandSize();
        for (Modifier modifier : getModifiers(game, ModifierEffect.HAND_SIZE_MODIFIER))
            result += modifier.getHandSizeModifier(game, playerId);
        return Math.max(0, result);
    }

    @Override
    public int getMoveLimit(LotroGame game, int baseMoveLimit) {
        int result = baseMoveLimit;
        for (Modifier modifier : getModifiers(game, ModifierEffect.MOVE_LIMIT_MODIFIER))
            result += modifier.getMoveLimitModifier(game);
        return Math.max(1, result);
    }

    @Override
    public int getThreatLimit(LotroGame game, int baseThreatLimit) {
        int result = baseThreatLimit;
        for (Modifier modifier : getModifiers(game, ModifierEffect.THREAT_LIMIT_MODIFIER))
            result += modifier.getThreatLimitModifier(game);
        return Math.max(1, result);
    }

    @Override
    public boolean addsTwilightForCompanionMove(LotroGame game, PhysicalCard companion) {
        for (Modifier modifier : getModifiersAffectingCard(game, ModifierEffect.MOVE_TWILIGHT_MODIFIER, companion)) {
            if (!modifier.addsTwilightForCompanionMove(game, companion))
                return false;
        }
        return true;
    }

    @Override
    public int getStrength(LotroGame game, PhysicalCard physicalCard) {
        return ask(game, physicalCard, K_STRENGTH, K_STRENGTH, null);
    }

    private int workOutGetStrength(LotroGame game, PhysicalCard physicalCard) {
        LoggingThreadLocal.logMethodStart(physicalCard, "getStrength");
        try {
            int result = physicalCard.getBlueprint().getStrength();
            for (Modifier modifier : getModifiersAffectingCard(game, ModifierEffect.STRENGTH_MODIFIER, physicalCard)) {
                final int strengthModifier = modifier.getStrengthModifier(game, physicalCard);
                if (strengthModifier <= 0 || appliesStrengthBonusModifier(game, modifier.getSource(), physicalCard))
                    result += strengthModifier;
            }
            return Math.max(0, result);
        } finally {
            LoggingThreadLocal.logMethodEnd();
        }
    }

    @Override
    public boolean appliesStrengthBonusModifier(LotroGame game, PhysicalCard modifierSource, PhysicalCard modifierTarget) {
        for (Modifier modifier : getModifiersAffectingCard(game, ModifierEffect.STRENGTH_BONUS_MODIFIER, modifierTarget)) {
            if (!modifier.appliesStrengthBonusModifier(game, modifierSource, modifierTarget))
                return false;
        }
        return true;
    }

    @Override
    public int getVitality(LotroGame game, PhysicalCard physicalCard) {
        return ask(game, physicalCard, K_VITALITY, K_VITALITY, null);
    }

    private int workOutGetVitality(LotroGame game, PhysicalCard physicalCard) {
        LoggingThreadLocal.logMethodStart(physicalCard, "getVitality");
        try {
            int result = physicalCard.getBlueprint().getVitality() - game.getGameState().getWounds(physicalCard);
            for (Modifier modifier : getModifiersAffectingCard(game, ModifierEffect.VITALITY_MODIFIER, physicalCard)) {
                result += modifier.getVitalityModifier(game, physicalCard);
            }
            return Math.max(0, result);
        } finally {
            LoggingThreadLocal.logMethodEnd();
        }
    }

    @Override
    public int getResistance(LotroGame game, PhysicalCard physicalCard) {
        return ask(game, physicalCard, K_RESISTANCE, K_RESISTANCE, null);
    }

    private int workOutGetResistance(LotroGame game, PhysicalCard physicalCard) {
        LoggingThreadLocal.logMethodStart(physicalCard, "getResistance");
        try {
            int result = physicalCard.getBlueprint().getResistance();
            // Companions resistance is reduced by the number of burdens
            if (physicalCard.getBlueprint().getCardType() == CardType.COMPANION)
                result -= game.getGameState().getBurdens();
            for (Modifier modifier : getModifiersAffectingCard(game, ModifierEffect.RESISTANCE_MODIFIER, physicalCard)) {
                result += modifier.getResistanceModifier(game, physicalCard);
            }
            return Math.max(0, result);
        } finally {
            LoggingThreadLocal.logMethodEnd();
        }
    }

    @Override
    public int getMinionSiteNumber(LotroGame game, PhysicalCard physicalCard) {
        LoggingThreadLocal.logMethodStart(physicalCard, "getMinionSiteNumber");
        try {
            int result = physicalCard.getBlueprint().getSiteNumber();
            for (Modifier modifier : getModifiersAffectingCard(game, ModifierEffect.SITE_NUMBER_MODIFIER, physicalCard)) {
                result += modifier.getMinionSiteNumberModifier(game, physicalCard);
            }
            return Math.max(0, result);
        } finally {
            LoggingThreadLocal.logMethodEnd();
        }
    }

    @Override
    public int getCurrentTwilightCost(LotroGame game, PhysicalCard physicalCard) {
        final LotroCardBlueprint blueprint = physicalCard.getBlueprint();
        // A site has a Shadow number and not a twilight cost.  What changes it ("Each site's Shadow number is -1")
        // changes it where the site lies, so a site is read with everything that applies.
        if (blueprint.getCardType() == CardType.SITE)
            return getTwilightCostToPlay(game, physicalCard, null, 0, true);
        // A card in play costs the number printed on it (CR 4.2 "twilight cost": "Cards already in play that
        // affect another card's twilight cost have that effect only when the card to be modified is coming into
        // play"; and by Ketura's ruling of 2026-10-06 the card's own text about its cost is read the same way).
        final Zone zone = physicalCard.getZone();
        if (zone != null && zone.isInPlay())
            return Math.max(0, blueprint.getTwilightCost());
        // Anywhere else the question is what the card would cost to play, the roaming penalty apart.
        return getTwilightCostToPlay(game, physicalCard, null, 0, true);
    }

    @Override
    public int getTwilightCostToTransfer(LotroGame game, PhysicalCard physicalCard) {
        // A card being transferred is in play, and is not coming into play: neither another card's text nor its
        // own changes the number printed on it.
        return Math.max(0, physicalCard.getBlueprint().getTwilightCost());
    }

    @Override
    public int getTwilightCostToPlay(LotroGame game, PhysicalCard physicalCard, PhysicalCard target, int twilightCostModifier, boolean ignoreRoamingPenalty) {
        LoggingThreadLocal.logMethodStart(physicalCard, "getTwilightCost");
        try {
            // What a card costs to play is what it costs to play now: an event printed with several phases is
            // priced as an event of this one (CR 4.2 "phase action"), before its play as it is during it.
            int result = game.getGameState().asIfBeingPlayed(physicalCard, () -> {
                int cost = physicalCard.getBlueprint().getTwilightCost() + twilightCostModifier;
                cost += physicalCard.getBlueprint().getTwilightCostModifier(game, physicalCard, target);
                for (Modifier modifier : getModifiersAffectingCard(game, ModifierEffect.TWILIGHT_COST_MODIFIER, physicalCard)) {
                    cost += modifier.getTwilightCostModifier(game, physicalCard, target, ignoreRoamingPenalty);
                }
                return cost;
            });
            result = Math.max(0, result);

            if (!ignoreRoamingPenalty && hasKeyword(game, physicalCard, Keyword.ROAMING)) {
                int roamingPenalty = getRoamingPenalty(game, physicalCard);
                result += Math.max(0, roamingPenalty);
            }
            return result;
        } finally {
            LoggingThreadLocal.logMethodEnd();
        }
    }

    @Override
    public int getRoamingPenalty(LotroGame game, PhysicalCard physicalCard) {
        LoggingThreadLocal.logMethodStart(physicalCard, "getRoamingPenalty");
        try {
            int result = 2;
            for (Modifier modifier : getModifiersAffectingCard(game, ModifierEffect.TWILIGHT_COST_MODIFIER, physicalCard)) {
                result += modifier.getRoamingPenaltyModifier(game, physicalCard);
            }
            return Math.max(0, result);
        } finally {
            LoggingThreadLocal.logMethodEnd();
        }
    }

    @Override
    public int getOverwhelmMultiplier(LotroGame game, PhysicalCard card) {
        LoggingThreadLocal.logMethodStart(card, "getOverwhelmMultiplier");
        try {
            int overwhelmMultiplier = 2;
            for (Modifier modifier : getModifiersAffectingCard(game, ModifierEffect.OVERWHELM_MODIFIER, card)) {
                overwhelmMultiplier = Math.max(overwhelmMultiplier, modifier.getOverwhelmMultiplier(game, card));
            }
            return overwhelmMultiplier;
        } finally {
            LoggingThreadLocal.logMethodEnd();
        }
    }

    @Override
    public List<CardType> getCardTypes(LotroGame game, PhysicalCard card) {
        var types = new ArrayList<CardType>();
        types.add(card.getBlueprint().getCardType());
        for (Modifier modifier : getModifiersAffectingCard(game, ModifierEffect.ADDITIONAL_CARD_TYPE, card)) {
            var cardTypeMod = (IsAdditionalCardTypeModifier)modifier;
            types.add(cardTypeMod.getType());
        }

        return types;
    }

    @Override
    public boolean isCardType(LotroGame game, PhysicalCard card, CardType cardType) {
        // Asked by every filter that names a type, for every card it looks at: with nothing in the game that gives a
        // card another type, the printed one is the answer.
        final List<Modifier> additional = _modifiers.get(ModifierEffect.ADDITIONAL_CARD_TYPE);
        if (additional == null || additional.isEmpty())
            return card.getBlueprint().getCardType() == cardType;
        return ask(game, card, SLOT_CARD_TYPE + cardType.ordinal(), K_CARD_TYPE, cardType) != 0;
    }

    private boolean workOutIsCardType(LotroGame game, PhysicalCard card, CardType cardType) {
        var types = getCardTypes(game, card);
        return types.contains(cardType);
    }



    @Override
    public List<Race> getRaces(LotroGame game, PhysicalCard card) {
        var races = new ArrayList<Race>();
        if(card.getBlueprint().getRace() != null) {
            races.add(card.getBlueprint().getRace());
        }

        for (Modifier modifier : getModifiersAffectingCard(game, ModifierEffect.ADDITIONAL_RACE, card)) {
            var cardTypeMod = (IsAdditionalRaceModifier)modifier;
            races.add(cardTypeMod.getRace());
        }

        return races;
    }

    @Override
    public boolean isRace(LotroGame game, PhysicalCard card, Race race) {
        var races = getRaces(game, card);
        return races.contains(race);
    }

    @Override
    public List<Culture> getCultures(LotroGame game, PhysicalCard card) {
        var cultures = new ArrayList<Culture>();
        if(card.getBlueprint().getCulture() != null) {
            cultures.add(card.getBlueprint().getCulture());
        }

        for (Modifier modifier : getModifiersAffectingCard(game, ModifierEffect.ADDITIONAL_CULTURE, card)) {
            var cultureMod = (IsAdditionalCultureModifier)modifier;
            cultures.add(cultureMod.getCulture());
        }

        return cultures;
    }

    @Override
    public boolean isCulture(LotroGame game, PhysicalCard card, Culture culture) {
        var cultures = getCultures(game, card);
        return cultures.contains(culture);
    }

    @Override
    public int getWoundsTakenInCurrentPhase(PhysicalCard card) {
        Integer wounds = _woundsPerPhaseMap.get(card.getCardId());
        if (wounds == null)
            return 0;
        return wounds;
    }

    @Override
    public boolean canTakeWounds(LotroGame game, Collection<PhysicalCard> woundSources, PhysicalCard card, int woundsToTake) {
        LoggingThreadLocal.logMethodStart(card, "canTakeWound");
        try {
            if(game.getGameState().isHindered(card))
                return false;

            for (Modifier modifier : getModifiersAffectingCard(game, ModifierEffect.WOUND_MODIFIER, card)) {
                Integer woundsTaken = _woundsPerPhaseMap.get(card.getCardId());
                if (woundsTaken == null)
                    woundsTaken = 0;
                if (!modifier.canTakeWounds(game, woundSources, card, woundsTaken, woundsToTake))
                    return false;
            }
            return true;
        } finally {
            LoggingThreadLocal.logMethodEnd();
        }
    }

    @Override
    public boolean canTakeWoundsFromLosingSkirmish(LotroGame game, PhysicalCard card) {
        LoggingThreadLocal.logMethodStart(card, "canTakeWound");
        try {
            if(game.getGameState().isHindered(card))
                return false;

            for (Modifier modifier : getModifiersAffectingCard(game, ModifierEffect.WOUND_MODIFIER, card)) {
                if (!modifier.canTakeWoundsFromLosingSkirmish(game, card))
                    return false;
            }
            return true;
        } finally {
            LoggingThreadLocal.logMethodEnd();
        }
    }

    @Override
    public boolean canTakeArcheryWound(LotroGame game, PhysicalCard card) {
        LoggingThreadLocal.logMethodStart(card, "canTakeArcheryWound");
        try {
            if(game.getGameState().isHindered(card))
                return false;
            for (Modifier modifier : getModifiersAffectingCard(game, ModifierEffect.WOUND_MODIFIER, card)) {
                if (!modifier.canTakeArcheryWound(game, card))
                    return false;
            }
            return true;
        } finally {
            LoggingThreadLocal.logMethodEnd();
        }
    }

    @Override
    public boolean canBeExerted(LotroGame game, PhysicalCard exertionSource, PhysicalCard exertedCard) {
        LoggingThreadLocal.logMethodStart(exertedCard, "canBeExerted");
        try {
            if(game.getGameState().isHindered(exertedCard))
                return false;

            for (Modifier modifier : getModifiersAffectingCard(game, ModifierEffect.WOUND_MODIFIER, exertedCard)) {
                if (!modifier.canBeExerted(game, exertionSource, exertedCard))
                    return false;
            }
            return true;
        } finally {
            LoggingThreadLocal.logMethodEnd();
        }
    }

    @Override
    public boolean isUnhastyCompanionAllowedToParticipateInSkirmishes(LotroGame game, PhysicalCard card) {
        LoggingThreadLocal.logMethodStart(card, "isAllyAllowedToParticipateInSkirmishes");
        try {
            for (Modifier modifier : getModifiersAffectingCard(game, ModifierEffect.PRESENCE_MODIFIER, card)) {
                if (modifier.isUnhastyCompanionAllowedToParticipateInSkirmishes(game, card))
                    return true;
            }
            return false;
        } finally {
            LoggingThreadLocal.logMethodEnd();
        }
    }

    @Override
    public boolean isAllyAllowedToParticipateInArcheryFire(LotroGame game, PhysicalCard card) {
        LoggingThreadLocal.logMethodStart(card, "isAllyAllowedToParticipateInArcheryFire");
        try {
            // "May not participate" beats every "may participate", whatever card says it.
            if (isAllyPreventedFromParticipatingInArcheryFire(game, card))
                return false;
            for (Modifier modifier : getModifiersAffectingCard(game, ModifierEffect.PRESENCE_MODIFIER, card)) {
                if (modifier.isAllyParticipateInArcheryFire(game, card))
                    return true;
            }
            return false;
        } finally {
            LoggingThreadLocal.logMethodEnd();
        }
    }

    @Override
    public boolean isAllyAllowedToParticipateInSkirmishes(LotroGame game, Side sidePlayer, PhysicalCard card) {
        LoggingThreadLocal.logMethodStart(card, "isAllyAllowedToParticipateInSkirmishes");
        try {
            // "May not participate" beats every "may participate", whatever card says it.
            if (isAllyPreventedFromParticipatingInSkirmishes(game, sidePlayer, card))
                return false;
            for (Modifier modifier : getModifiersAffectingCard(game, ModifierEffect.PRESENCE_MODIFIER, card)) {
                if (modifier.isAllyParticipateInSkirmishes(game, sidePlayer, card))
                    return true;
            }
            return false;
        } finally {
            LoggingThreadLocal.logMethodEnd();
        }
    }

    @Override
    public boolean isAllyPreventedFromParticipatingInArcheryFire(LotroGame game, PhysicalCard card) {
        LoggingThreadLocal.logMethodStart(card, "isAllyPreventedFromParticipatingInArcheryFire");
        try {
            for (Modifier modifier : getModifiersAffectingCard(game, ModifierEffect.PRESENCE_MODIFIER, card)) {
                if (modifier.isAllyPreventedFromParticipatingInArcheryFire(game, card))
                    return true;
            }
            return false;
        } finally {
            LoggingThreadLocal.logMethodEnd();
        }
    }

    @Override
    public boolean isAllyPreventedFromParticipatingInSkirmishes(LotroGame game, Side sidePlayer, PhysicalCard card) {
        LoggingThreadLocal.logMethodStart(card, "isAllyPreventedFromParticipatingInSkirmishes");
        try {
            for (Modifier modifier : getModifiersAffectingCard(game, ModifierEffect.PRESENCE_MODIFIER, card)) {
                if (modifier.isAllyPreventedFromParticipatingInSkirmishes(game, sidePlayer, card))
                    return true;
            }
            return false;
        } finally {
            LoggingThreadLocal.logMethodEnd();
        }
    }

    @Override
    public boolean addsToArcheryTotal(LotroGame game, PhysicalCard card) {
        LoggingThreadLocal.logMethodStart(card, "addsToArcheryTotal");
        try {
            for (Modifier modifier : getModifiersAffectingCard(game, ModifierEffect.ARCHERY_MODIFIER, card))
                if (!modifier.addsToArcheryTotal(game, card))
                    return false;

            return true;
        } finally {
            LoggingThreadLocal.logMethodEnd();
        }
    }

    @Override
    public boolean canPlayAction(LotroGame game, String performingPlayer, Action action) {
        for (Modifier modifier : getModifiers(game, ModifierEffect.ACTION_MODIFIER))
            if (!modifier.canPlayAction(game, performingPlayer, action))
                return false;
        return true;
    }

    @Override
    public boolean canPlayCard(LotroGame game, String performingPlayer, PhysicalCard card) {
        for (Modifier modifier : getModifiers(game, ModifierEffect.ACTION_MODIFIER))
            if (!modifier.canPlayCard(game, performingPlayer, card))
                return false;
        return true;
    }

    @Override
    public boolean canCancelSkirmish(LotroGame game, PhysicalCard card) {
        for (Modifier modifier : getModifiers(game, ModifierEffect.CANCEL_SKIRMISH_MODIFIER))
            if (!modifier.canCancelSkirmish(game, card))
                return false;
        return true;
    }

    @Override
    public boolean canHavePlayedOn(LotroGame game, PhysicalCard playedCard, PhysicalCard target) {
        for (Modifier modifier : getModifiersAffectingCard(game, ModifierEffect.TARGET_MODIFIER, target))
            if (!modifier.canHavePlayedOn(game, playedCard, target))
                return false;
        return true;
    }

    @Override
    public boolean canHaveTransferredOn(LotroGame game, PhysicalCard playedCard, PhysicalCard target) {
        for (Modifier modifier : getModifiersAffectingCard(game, ModifierEffect.TARGET_MODIFIER, target))
            if (!modifier.canHaveTransferredOn(game, playedCard, target))
                return false;
        return true;
    }

    @Override
    public boolean canBeTransferred(LotroGame game, PhysicalCard attachment) {
        for (Modifier modifier : getModifiersAffectingCard(game, ModifierEffect.TRANSFER_MODIFIER, attachment))
            if (!modifier.canBeTransferred(game, attachment))
                return false;
        return true;
    }

    @Override
    public int bearableItemsOfClass(LotroGame game, PhysicalCard attachedTo, PossessionClass itemClass) {
        int total = 1;
        for (Modifier mod : getModifiersAffectingCard(game, ModifierEffect.CAN_BEAR_EXTRA_ITEMS, attachedTo)) {
            if (mod instanceof final BearExtraItemsModifier modifier) {
                total += modifier.getItemClassBonus(game, itemClass);
            }
        }
        return total;
    }

    @Override
    public boolean shouldSkipPhase(LotroGame game, Phase phase, String playerId) {
        for (Modifier modifier : getModifiers(game, ModifierEffect.ACTION_MODIFIER))
            if (modifier.shouldSkipPhase(game, phase, playerId))
                return true;
        return false;
    }

    @Override
    public List<? extends Action> getExtraPhaseActions(LotroGame game, PhysicalCard target) {
        List<Action> activateCardActions = new LinkedList<>();

        for (Modifier modifier : getModifiersAffectingCard(game, ModifierEffect.EXTRA_ACTION_MODIFIER, target)) {
            List<? extends Action> actions = modifier.getExtraPhaseAction(game, target);
            if (actions != null)
                activateCardActions.addAll(actions);
        }

        return activateCardActions;
    }

    @Override
    public Collection<PhysicalCard> getGameTextCardsToDuplicate(LotroGame game, PhysicalCard target) {
        var cards = new LinkedList<PhysicalCard>();

        for (Modifier modifier : getModifiersAffectingCard(game, ModifierEffect.GAME_TEXT_DUPLICATE, target)) {
            var dupeMod = (DuplicateGameTextModifier) modifier;
            var dupes = dupeMod.getCardsToDuplicate(game);
            if(dupes != null) {
                cards.addAll(dupes);
            }
        }

        return cards;
    }

    @Override
    public boolean hasExtraPhaseActionsFromOtherPlayer(LotroGame game, PhysicalCard target) {
        for (Modifier modifier : getModifiersAffectingCard(game, ModifierEffect.EXTRA_ACTION_MODIFIER, target)) {
            var extraActionModifier = (AddActionToCardModifier) modifier;
            if(extraActionModifier.getTargetPlayer() != null && !extraActionModifier.getTargetPlayer().equals(target.getOwner()))
                return true;
        }

        return false;
    }

    @Override
    public List<? extends Action> getExtraPhaseActionsFromStacked(LotroGame game, PhysicalCard target) {
        List<Action> activateCardActions = new LinkedList<>();

        for (Modifier modifier : getModifiersAffectingCard(game, ModifierEffect.EXTRA_ACTION_MODIFIER, target)) {
            List<? extends Action> actions = modifier.getExtraPhaseActionFromStacked(game, target);
            if (actions != null)
                activateCardActions.addAll(actions);
        }

        return activateCardActions;
    }

    @Override
    public boolean canPayExtraCostsToPlay(LotroGame game, PhysicalCard target) {
        final List<? extends ExtraPlayCost> playCosts = target.getBlueprint().getExtraCostToPlay(game, target);
        if (playCosts != null)
            for (ExtraPlayCost playCost : playCosts) {
                final Condition condition = playCost.getCondition();
                if ((condition == null || condition.isFullfilled(game)) && !playCost.canPayExtraCostsToPlay(game, target))
                    return false;
            }

        for (Modifier modifier : getModifiersAffectingCard(game, ModifierEffect.EXTRA_COST_MODIFIER, target)) {
            if (!modifier.canPayExtraCostsToPlay(game, target))
                return false;
        }

        return true;
    }

    @Override
    public void appendExtraCosts(LotroGame game, CostToEffectAction action, PhysicalCard target) {
        final List<? extends ExtraPlayCost> playCosts = target.getBlueprint().getExtraCostToPlay(game, target);
        if (playCosts != null)
            for (ExtraPlayCost playCost : playCosts) {
                final Condition condition = playCost.getCondition();
                if (condition == null || condition.isFullfilled(game))
                    playCost.appendExtraCosts(game, action, target);
            }

        for (Modifier modifier : getModifiersAffectingCard(game, ModifierEffect.EXTRA_COST_MODIFIER, target)) {
            modifier.appendExtraCosts(game, action, target);
        }
    }

    @Override
    public boolean isValidAssignments(LotroGame game, Side side, Map<PhysicalCard, Set<PhysicalCard>> assignments) {
        for (Modifier modifier : getModifiers(game, ModifierEffect.ASSIGNMENT_MODIFIER)) {
            if (!modifier.isValidAssignments(game, side, assignments))
                return false;
            for (Map.Entry<PhysicalCard, Set<PhysicalCard>> assignment : assignments.entrySet()) {
                if (affectsCardWithSkipSet(game, assignment.getKey(), modifier))
                    if (!modifier.isValidAssignments(game, side, assignment.getKey(), assignment.getValue()))
                        return false;
            }
        }
        return true;
    }

    @Override
    public boolean canBeAssignedToSkirmish(LotroGame game, Side sidePlayer, PhysicalCard card) {
        LoggingThreadLocal.logMethodStart(card, "canBeAssignedToSkirmish");
        try {
            if(hasKeyword(game, card, Keyword.HINDERED))
                return false;

            for (Modifier modifier : getModifiersAffectingCard(game, ModifierEffect.ASSIGNMENT_MODIFIER, card))
                if (modifier.isPreventedFromBeingAssignedToSkirmish(game, sidePlayer, card))
                    return false;
            return true;
        } finally {
            LoggingThreadLocal.logMethodEnd();
        }
    }

    @Override
    public boolean canBeDiscardedFromPlay(LotroGame game, String performingPlayer, PhysicalCard card, PhysicalCard source) {
        LoggingThreadLocal.logMethodStart(card, "canBeDiscardedFromPlay");
        try {
            for (Modifier modifier : getModifiersProtectingCard(game, ModifierEffect.DISCARD_FROM_PLAY_MODIFIER, card))
                if (!modifier.canBeDiscardedFromPlay(game, performingPlayer, card, source))
                    return false;
            return true;
        } finally {
            LoggingThreadLocal.logMethodEnd();
        }
    }

    @Override
    public boolean canBeHinderedBy(LotroGame game, String performingPlayer, PhysicalCard card, PhysicalCard source) {
        LoggingThreadLocal.logMethodStart(card, "canBeHinderedBy");
        try {
            for (Modifier modifier : getModifiersAffectingCard(game, ModifierEffect.HINDERED_MODIFER, card)) {
                var hinderedMod = (CantHinderModifier) modifier;
                if (!hinderedMod.canBeHinderedBy(game, performingPlayer, source))
                    return false;
            }
            return true;
        } finally {
            LoggingThreadLocal.logMethodEnd();
        }
    }

    @Override
    public boolean canBeReturnedToHand(LotroGame game, PhysicalCard card, PhysicalCard source) {
        LoggingThreadLocal.logMethodStart(card, "canBeReturnedToHand");
        try {
            for (Modifier modifier : getModifiersProtectingCard(game, ModifierEffect.RETURN_TO_HAND_MODIFIER, card))
                if (!modifier.canBeReturnedToHand(game, card, source))
                    return false;
            return true;
        } finally {
            LoggingThreadLocal.logMethodEnd();
        }
    }

    @Override
    public boolean canBeHealed(LotroGame game, String performingPlayer, PhysicalCard card, PhysicalCard healSource) {
        LoggingThreadLocal.logMethodStart(card, "canBeHealed");
        try {
            if(game.getGameState().isHindered(card))
                return false;

            for (Modifier modifier : getModifiersAffectingCard(game, ModifierEffect.WOUND_MODIFIER, card))
                if (!modifier.canBeHealed(game, performingPlayer, card, healSource))
                    return false;
            return true;
        } finally {
            LoggingThreadLocal.logMethodEnd();
        }
    }

    @Override
    public boolean canAddBurden(LotroGame game, String performingPlayer, PhysicalCard source) {
        for (Modifier modifier : getModifiersAffectingSource(game, ModifierEffect.BURDEN_MODIFIER, source)) {
            if (!modifier.canAddBurden(game, performingPlayer, source))
                return false;
        }
        return true;
    }

    @Override
    public boolean canRemoveBurden(LotroGame game, PhysicalCard source) {
        for (Modifier modifier : getModifiersAffectingSource(game, ModifierEffect.BURDEN_MODIFIER, source)) {
            if (!modifier.canRemoveBurden(game, source))
                return false;
        }
        return true;
    }

    @Override
    public boolean canRemoveThreat(LotroGame game, PhysicalCard source) {
        for (Modifier modifier : getModifiersAffectingSource(game, ModifierEffect.THREAT_MODIFIER, source)) {
            if (!modifier.canRemoveThreat(game, source))
                return false;
        }
        return true;
    }

    /**
     * Rule of 4. "You cannot draw (or take into hand) more than 4 cards during your fellowship phase."
     *
     * @param game
     * @param playerId
     * @return
     */
    @Override
    public boolean canDrawCardNoIncrement(LotroGame game, String playerId) {
        if (game.getGameState().getCurrentPlayerId().equals(playerId)) {
            if (game.getGameState().getCurrentPhase() != Phase.FELLOWSHIP)
                return true;
            if (game.getGameState().getCurrentPhase() == Phase.FELLOWSHIP && _drawnThisPhaseCount < 4) {
                return true;
            } else {
                return false;
            }
        }
        return true;
    }

    @Override
    public int getFellowshipDrawnCards(LotroGame game) {
        if(game.getGameState().getCurrentPhase() != Phase.FELLOWSHIP)
            return 0;

        return _drawnThisPhaseCount;
    }

    /**
     * Rule of 4. "You cannot draw (or take into hand) more than 4 cards during your fellowship phase."
     *
     * @param game
     * @param playerId
     * @return
     */
    @Override
    public boolean canDrawCardAndIncrementForRuleOfFour(LotroGame game, String playerId) {
        _modifierEpoch[0]++;
        if (game.getGameState().getCurrentPlayerId().equals(playerId)) {
            if (game.getGameState().getCurrentPhase() != Phase.FELLOWSHIP)
                return true;
            if (game.getGameState().getCurrentPhase() == Phase.FELLOWSHIP && _drawnThisPhaseCount < 4) {
                _drawnThisPhaseCount++;
                return true;
            } else {
                return false;
            }
        }
        return true;
    }

    @Override
    public boolean canLookOrRevealCardsInHand(LotroGame game, String revealingPlayerId, String performingPlayerId) {
        for (Modifier modifier : getModifiers(game, ModifierEffect.LOOK_OR_REVEAL_MODIFIER))
            if (!modifier.canLookOrRevealCardsInHand(game, revealingPlayerId, performingPlayerId))
                return false;
        return true;
    }

    @Override
    public boolean canDiscardCardsFromHand(LotroGame game, String playerId, PhysicalCard source) {
        for (Modifier modifier : getModifiersAffectingSource(game, ModifierEffect.DISCARD_NOT_FROM_PLAY, source))
            if (!modifier.canDiscardCardsFromHand(game, playerId, source))
                return false;
        return true;
    }

    @Override
    public boolean canDiscardCardsFromTopOfDeck(LotroGame game, String playerId, PhysicalCard source) {
        for (Modifier modifier : getModifiersAffectingSource(game, ModifierEffect.DISCARD_NOT_FROM_PLAY, source))
            if (!modifier.canDiscardCardsFromTopOfDeck(game, playerId, source))
                return false;
        return true;
    }

    @Override
    public boolean canBeLiberated(LotroGame game, String playerId, PhysicalCard card, PhysicalCard source) {
        LoggingThreadLocal.logMethodStart(card, "canBeLiberated");
        try {
            for (Modifier modifier : getModifiersAffectingCard(game, ModifierEffect.LIBERATION_MODIFIER, card))
                if (!modifier.canBeLiberated(game, playerId, card, source))
                    return false;
            return true;
        } finally {
            LoggingThreadLocal.logMethodEnd();
        }
    }

    @Override
    public Side hasInitiative(LotroGame game) {
        return hasInitiative(game, 0);
    }

    @Override
    public Side hasInitiativeNotCounting(LotroGame game, PhysicalCard cardAsking) {
        // The card is one of those counted below only while it is the Free Peoples player's own and lies in his
        // hand or in the void from it.
        final boolean counted = cardAsking != null
                && (cardAsking.getZone() == Zone.HAND || cardAsking.getZone() == Zone.VOID_FROM_HAND)
                && cardAsking.getOwner().equals(game.getGameState().getCurrentPlayerId());
        return hasInitiative(game, counted ? 1 : 0);
    }

    private Side hasInitiative(LotroGame game, int cardsNotCounted) {
        for (Modifier modifier : getModifiers(game, ModifierEffect.INITIATIVE_MODIFIER)) {
            if (!modifier.shadowCanHaveInitiative(game))
                return Side.FREE_PEOPLE;
        }

        for (Modifier modifier : getModifiers(game, ModifierEffect.INITIATIVE_MODIFIER)) {
            Side initiative = modifier.hasInitiative(game);
            if (initiative != null)
                return initiative;
        }

        // A card being played from hand is still counted while it is in the void and, once placed, until its play
        // is announced: the loss of initiative is answered together with the play (CR 4.2 "playing a card").
        int freePeopleInitiativeHandSize = game.getGameState().getHand(game.getGameState().getCurrentPlayerId()).size()
                + game.getGameState().getVoidFromHand(game.getGameState().getCurrentPlayerId()).size()
                + game.getActionsEnvironment().countCardsEnteringPlayFromHand(game.getGameState().getCurrentPlayerId())
                - cardsNotCounted;

        int initiativeHandSize = 4;
        for (Modifier modifier : getModifiers(game, ModifierEffect.INITIATIVE_MODIFIER))
            initiativeHandSize += modifier.getInitiativeHandSizeModifier(game);

        if (freePeopleInitiativeHandSize < initiativeHandSize)
            return Side.SHADOW;
        else
            return Side.FREE_PEOPLE;
    }

    @Override
    public int getNumberOfSpottableFPCultures(LotroGame game, String playerId) {
        Set<Culture> spottableCulturesBasedOnCards = new HashSet<>();
        for (PhysicalCard spottableFPCard : Filters.filterActive(game, Side.FREE_PEOPLE, Filters.spottable)) {
            spottableCulturesBasedOnCards.addAll(game.getModifiersQuerying().getCultures(game, spottableFPCard));
        }

        int result = 0;
        for (Culture spottableCulturesBasedOnCardsOnCard : spottableCulturesBasedOnCards) {
            if (canPlayerSpotCulture(game, playerId, spottableCulturesBasedOnCardsOnCard))
                result++;
        }

        for (Modifier modifier : getModifiers(game, ModifierEffect.SPOT_MODIFIER)) {
            result += modifier.getFPCulturesSpotCountModifier(game, playerId);
        }

        return result;
    }

    @Override
    public int getNumberOfSpottableControlledSites(LotroGame game, String playerId) {
        int result = Filters.filterActive(game, Filters.siteControlledByPlayer(playerId)).size();

        for (Modifier modifier : getModifiers(game, ModifierEffect.SPOT_MODIFIER)) {
            result += modifier.getSiteControlledSpotCountModifier(game, playerId);
        }

        return result;
    }

    private boolean canPlayerSpotCulture(LotroGame game, String playerId, Culture culture) {
        for (Modifier modifier : getModifiers(game, ModifierEffect.SPOT_MODIFIER))
            if (!modifier.canSpotCulture(game, culture, playerId))
                return false;
        return true;
    }

    @Override
    public int getNumberOfSpottableShadowCultures(LotroGame game, String playerId) {
        Set<Culture> spottableCulturesBasedOnCards = new HashSet<>();
        for (PhysicalCard spottableShadowCard : Filters.filterActive(game, Side.SHADOW, Filters.spottable)) {
            spottableCulturesBasedOnCards.addAll(game.getModifiersQuerying().getCultures(game, spottableShadowCard));
        }

        int result = 0;
        for (Culture spottableCulturesBasedOnCardsOnCard : spottableCulturesBasedOnCards) {
            if (canPlayerSpotCulture(game, playerId, spottableCulturesBasedOnCardsOnCard))
                result++;
        }

        return result;
    }

    @Override
    public int getSpotBonus(LotroGame game, Filterable filter, String spottingPlayer) {
        int result = 0;
        for (Modifier modifier : getModifiers(game, ModifierEffect.SPOT_MODIFIER))
            result += modifier.getSpotCountModifier(game, filter, spottingPlayer);
        return Math.max(0, result);
    }

    @Override
    public boolean hasFlagActive(LotroGame game, ModifierFlag modifierFlag) {
        for (Modifier modifier : getModifiers(game, ModifierEffect.SPECIAL_FLAG_MODIFIER))
            if (modifier.hasFlagActive(game, modifierFlag))
                return true;

        return false;
    }

    @Override
    public boolean hasFlagActive(LotroGame game, ModifierFlag modifierFlag, String playerId) {
        for (Modifier modifier : getModifiers(game, ModifierEffect.SPECIAL_FLAG_MODIFIER))
            if (modifier.hasFlagActive(game, modifierFlag, playerId))
                return true;

        return false;
    }

    @Override
    public boolean canReplaceSite(LotroGame game, String playerId, PhysicalCard siteToReplace) {
        for (Modifier modifier : getModifiersAffectingCard(game, ModifierEffect.REPLACE_SITE_MODIFIER, siteToReplace))
            if (!modifier.isSiteReplaceable(game, playerId))
                return false;

        return true;
    }

    @Override
    public boolean canPlaySite(LotroGame game, String playerId) {
        for (Modifier modifier : getModifiers(game, ModifierEffect.PLAY_SITE_MODIFIER))
            if (!modifier.canPlaySite(game, playerId))
                return false;

        return true;
    }

    @Override
    public CostToEffectAction getAssignmentCosts(LotroGame game, String assigningPlayer, PhysicalCard fpCharacter, PhysicalCard minion) {
        // Only the Free Peoples player is ever charged to assign
        if (assigningPlayer == null || !assigningPlayer.equals(game.getGameState().getCurrentPlayerId()))
            return null;

        // One payment for the pair, every charge in it.  Its source is the card that charges (the first, should
        // two ever charge for one assignment).
        AssignmentCostAction payment = null;
        for (Modifier modifier : getModifiersAffectingCard(game, ModifierEffect.ASSIGNMENT_MODIFIER, minion)) {
            final AssignmentCostAction payingInto = (payment != null) ? payment : new AssignmentCostAction(modifier.getSource(), assigningPlayer);
            if (modifier.appendExtraAssignmentCosts(game, payingInto, fpCharacter, minion))
                payment = payingInto;
        }
        return payment;
    }

    @Override
    public boolean appendAssignmentCosts(LotroGame game, CostToEffectAction action, PhysicalCard fpCharacter, PhysicalCard minion) {
        boolean charged = false;
        for (Modifier modifier : getModifiersAffectingCard(game, ModifierEffect.ASSIGNMENT_MODIFIER, minion)) {
            if (modifier.appendExtraAssignmentCosts(game, action, fpCharacter, minion))
                charged = true;
        }
        return charged;
    }

    @Override
    public int getSanctuaryHealModifier(LotroGame game) {
        int result = 0;
        for (Modifier modifier : getModifiers(game, ModifierEffect.SANCTUARY_HEAL_MODIFIER))
            result += modifier.getSanctuaryHealModifier(game);

        return result;
    }

    @Override
    public int getStartingFellowshipCostModifier(LotroGame game, String playerId) {
        int result = 0;
        for (Modifier modifier : getModifiers(game, ModifierEffect.STARTING_FELLOWSHIP_COST_MODIFIER))
            result += modifier.getStartingFellowshipCostModifier(game, playerId);

        return result;
    }

    @Override
    public int getMinimumBid(LotroGame game, String playerId) {
        int result = 0;
        for (Modifier modifier : getModifiers(game, ModifierEffect.MINIMUM_BID_MODIFIER))
            result = Math.max(result, modifier.getMinimumBidModifier(game, playerId));

        return result;
    }

    @Override
    public boolean isHandRevealed(LotroGame game, String playerId) {
        for (Modifier modifier : getModifiers(game, ModifierEffect.HAND_REVEAL_MODIFIER))
            if (modifier.isHandRevealed(game, playerId))
                return true;
        return false;
    }

    @Override
    public int getUniqueness(LotroGame game, PhysicalCard card) {
        // Uniqueness is the number of copies of the card allowed in play at once: 1 is unique, 4 is non-unique.
        // Loosening overrides raise the blueprint's limit; restricting overrides then cap the result, so that
        // "is unique" always beats "is not unique" no matter which order the modifiers were added.
        int result = card.getBlueprint().getUniqueRestriction();
        int restriction = Integer.MAX_VALUE;
        for (Modifier modifier : getModifiersAffectingCard(game, ModifierEffect.UNIQUENESS_MODIFIER, card)) {
            int override = modifier.getOverrideUniqueness(game, card);
            if (override > 0) {
                if (modifier.loosensUniqueness(game, card))
                    result = Math.max(result, override);
                else
                    restriction = Math.min(restriction, override);
            }
        }
        if (restriction != Integer.MAX_VALUE)
            result = Math.min(result, restriction);
        return result;
    }

    @Override
    public int getPotentialDiscount(LotroGame game, PhysicalCard playedCard) {
        int result = playedCard.getBlueprint().getPotentialDiscount(game, playedCard.getOwner(), playedCard);

        for (Modifier modifier : getModifiersAffectingCard(game, ModifierEffect.POTENTIAL_DISCOUNT_MODIFIER, playedCard)) {
            result += modifier.getPotentialDiscount(game, playedCard);
        }

        return result;
    }

    @Override
    public void appendPotentialDiscounts(LotroGame game, CostToEffectAction action, PhysicalCard playedCard) {
        playedCard.getBlueprint().appendPotentialDiscountEffects(game, action, playedCard.getOwner(), playedCard);

        for (Modifier modifier : getModifiersAffectingCard(game, ModifierEffect.POTENTIAL_DISCOUNT_MODIFIER, playedCard)) {
            modifier.appendPotentialDiscounts(game, action, playedCard);
        }
    }

    @Override
    public boolean deadPileGoesToDiscard(LotroGame game, PhysicalCard card) {
        for (Modifier modifier : getModifiersAffectingCard(game, ModifierEffect.DEAD_PILE_MODIFIER, card))
            if (modifier.deadPileGoesToDiscard(game, card))
                return true;
        return false;
    }

    @Override
    public boolean comesIntoPlayExhausted(LotroGame game, PhysicalCard card) {
        return !getModifiersAffectingCard(game, ModifierEffect.COMES_INTO_PLAY_EXHAUSTED_MODIFIER, card).isEmpty();
    }

    @Override
    public boolean sanctuaryMayRemoveBurdens(LotroGame game, String playerId) {
        for (Modifier modifier : getModifiers(game, ModifierEffect.SANCTUARY_REMOVE_BURDEN_MODIFIER))
            if (modifier.sanctuaryMayRemoveBurdens(game, playerId))
                return true;
        return false;
    }

    private class ModifierHookImpl implements ModifierHook {
        private final Modifier _modifier;

        private ModifierHookImpl(Modifier modifier) {
            _modifier = modifier;
        }

        @Override
        public void stop() {
            removeModifier(_modifier);
        }
    }
}
