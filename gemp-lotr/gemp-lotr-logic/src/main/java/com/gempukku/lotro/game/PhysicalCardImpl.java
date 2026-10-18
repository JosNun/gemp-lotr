package com.gempukku.lotro.game;

import com.gempukku.lotro.common.Zone;
import com.gempukku.lotro.game.state.LotroGame;
import com.gempukku.lotro.logic.modifiers.Modifier;
import com.gempukku.lotro.logic.modifiers.ModifierHook;

import java.util.LinkedList;
import java.util.List;

public class PhysicalCardImpl implements PhysicalCard {
    private int _cardId;
    private boolean _flipped;
    private final String _blueprintId;
    private final String _owner;
    private String _cardController;
    private Zone _zone;
    private Zone _playedFromZone;
    private final LotroCardBlueprint _blueprint;

    private PhysicalCardImpl _attachedTo;
    private PhysicalCardImpl _stackedOn;

    private List<ModifierHook> _modifierHooks;
    private List<ModifierHook> _modifierHooksStacked;
    private List<ModifierHook> _modifierHooksInDiscard;
    private List<ModifierHook> _modifierHooksControlledSite;
    private List<ModifierHook> _modifierHooksPermanentSite;

    private WhileInZoneData _whileInZoneData;

    // The game's count of changes (GameState), for a card that is in a game: what is set on a card directly is
    // read by questions about the game too.
    private long[] _queryEpoch;

    public void shareQueryEpoch(long[] queryEpoch) {
        _queryEpoch = queryEpoch;
    }

    // Whether this card is active (GameState.isCardInPlayActive), and the count of changes plus one that was
    // worked out at; not part of the game.
    private transient long _activeAt;
    private transient boolean _active;

    public boolean hasActiveKept(long count) {
        return _activeAt == count + 1;
    }

    public boolean getActiveKept() {
        return _active;
    }

    public void keepActive(long count, boolean active) {
        _activeAt = count + 1;
        _active = active;
    }

    private void touched() {
        if (_queryEpoch != null)
            _queryEpoch[0]++;
    }

    private Integer _siteNumber;

    // Optional display name override (e.g., meta-site modifiers display the visual card's name)
    private String _displayName;

    /**
     * The id the card was created with. Card ids change mid-game (GameState.assignNewCardId hides a card's history),
     * so they cannot be a hash; this cannot change. Hashing on it rather than on identity makes every HashSet and
     * HashMap of cards iterate in the same order whenever a game is replayed -- ChooseArbitraryCardsEffect's choices,
     * for one, came out in a different order every run. Equality stays identity.
     */
    private final int _createdAs;

    public PhysicalCardImpl(int cardId, String blueprintId, String owner, LotroCardBlueprint blueprint) {
        _cardId = cardId;
        _createdAs = cardId;
        _blueprintId = blueprintId;
        _owner = owner;
        _blueprint = blueprint;
    }

    @Override
    public int hashCode() {
        return _createdAs;
    }

    @Override
    public boolean equals(Object o) {
        return this == o;
    }

    @Override
    public void setCardId(int cardId) {
        touched();
        _cardId = cardId;
    }

    @Override
    public String getBlueprintId() {
        return _blueprintId;
    }

    /**
     * Returns a display name override, or null if the card should use its blueprint name.
     * Used by meta-site modifiers to display the visual position card's name instead.
     */
    public String getDisplayName() {
        return _displayName;
    }

    public void setDisplayName(String displayName) {
        _displayName = displayName;
    }

    public void setZone(Zone zone) {
        touched();
        _zone = zone;
    }

    @Override
    public boolean isFlipped() {
        return _flipped;
    }

    @Override
    public void setFlipped(boolean flipped) {
        touched();
        _flipped = flipped;
    }

    @Override
    public Zone getZone() {
        return _zone;
    }

    @Override
    public void setPlayedFromZone(Zone zone) {
        touched();
        _playedFromZone = zone;
    }
    @Override
    public Zone getPlayedFromZone() {
        return _playedFromZone;
    }

    @Override
    public String getOwner() {
        return _owner;
    }

    public void setCardController(String siteController) {
        touched();
        _cardController = siteController;
    }

    @Override
    public String getCardController() {
        return _cardController;
    }

    public void startAffectingGame(LotroGame game) {
        List<? extends Modifier> modifiers = _blueprint.getInPlayModifiers(game, this);
        if (modifiers != null) {
            _modifierHooks = new LinkedList<>();
            for (Modifier modifier : modifiers)
                _modifierHooks.add(game.getModifiersEnvironment().addAlwaysOnModifier(modifier));
        }
    }

    public void stopAffectingGame() {
        if (_modifierHooks != null) {
            for (ModifierHook modifierHook : _modifierHooks)
                modifierHook.stop();
            _modifierHooks = null;
        }
    }

    public void startAffectingGameStacked(LotroGame game) {
        List<? extends Modifier> modifiers = _blueprint.getStackedOnModifiers(game, this);
        if (modifiers != null) {
            _modifierHooksStacked = new LinkedList<>();
            for (Modifier modifier : modifiers)
                _modifierHooksStacked.add(game.getModifiersEnvironment().addAlwaysOnModifier(modifier));
        }
    }

    public void stopAffectingGameStacked() {
        if (_modifierHooksStacked != null) {
            for (ModifierHook modifierHook : _modifierHooksStacked)
                modifierHook.stop();
            _modifierHooksStacked = null;
        }
    }

    public void startAffectingGameInDiscard(LotroGame game) {
        List<? extends Modifier> modifiers = _blueprint.getInDiscardModifiers(game, this);
        if (modifiers != null) {
            _modifierHooksInDiscard = new LinkedList<>();
            for (Modifier modifier : modifiers)
                _modifierHooksInDiscard.add(game.getModifiersEnvironment().addAlwaysOnModifier(modifier));
        }
    }

    public void stopAffectingGameInDiscard() {
        if (_modifierHooksInDiscard != null) {
            for (ModifierHook modifierHook : _modifierHooksInDiscard)
                modifierHook.stop();
            _modifierHooksInDiscard = null;
        }
    }

    public void startAffectingGamePermanentSite(LotroGame game) {
        List<? extends Modifier> modifiers = _blueprint.getPermanentSiteModifiers(game, this);
        if (modifiers != null) {
            _modifierHooksPermanentSite = new LinkedList<>();
            for (Modifier modifier : modifiers)
                _modifierHooksPermanentSite.add(game.getModifiersEnvironment().addAlwaysOnModifier(modifier));
        }
    }

    public void stopAffectingGameControlledSite() {
        if (_modifierHooksControlledSite != null) {
            for (ModifierHook modifierHook : _modifierHooksControlledSite)
                modifierHook.stop();
            _modifierHooksControlledSite = null;
        }
    }

    public void startAffectingGameControlledSite(LotroGame game) {
        List<? extends Modifier> modifiers = _blueprint.getControlledSiteModifiers(game, this);
        if (modifiers != null) {
            _modifierHooksControlledSite = new LinkedList<>();
            for (Modifier modifier : modifiers)
                _modifierHooksControlledSite.add(game.getModifiersEnvironment().addAlwaysOnModifier(modifier));
        }
    }

    @Override
    public int getCardId() {
        return _cardId;
    }

    @Override
    public LotroCardBlueprint getBlueprint() {
        return _blueprint;
    }

    public void attachTo(PhysicalCardImpl physicalCard) {
        touched();
        _attachedTo = physicalCard;
    }

    @Override
    public PhysicalCard getAttachedTo() {
        return _attachedTo;
    }

    public void stackOn(PhysicalCardImpl physicalCard) {
        touched();
        _stackedOn = physicalCard;
    }

    @Override
    public PhysicalCard getStackedOn() {
        return _stackedOn;
    }

    @Override
    public WhileInZoneData getWhileInZoneData() {
        return _whileInZoneData;
    }

    @Override
    public void setWhileInZoneData(WhileInZoneData whileInZoneData) {
        touched();
        _whileInZoneData = whileInZoneData;
    }

    @Override
    public Integer getSiteNumber() {
        return _siteNumber;
    }

    @Override
    public void setSiteNumber(Integer number) {
        touched();
        _siteNumber = number;
    }

    @Override
    public void copyCardStats(PhysicalCard other) {
        touched();
        this._flipped = other.isFlipped();
        this._cardController = other.getCardController();
        this._siteNumber = other.getSiteNumber();
    }
}
