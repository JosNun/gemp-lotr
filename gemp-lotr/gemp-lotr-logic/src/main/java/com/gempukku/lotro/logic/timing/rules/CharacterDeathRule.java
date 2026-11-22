package com.gempukku.lotro.logic.timing.rules;

import com.gempukku.lotro.common.CardType;
import com.gempukku.lotro.common.Phase;
import com.gempukku.lotro.filters.Filters;
import com.gempukku.lotro.game.AbstractActionProxy;
import com.gempukku.lotro.game.PhysicalCard;
import com.gempukku.lotro.game.state.LotroGame;
import com.gempukku.lotro.game.state.actions.DefaultActionsEnvironment;
import com.gempukku.lotro.logic.actions.RequiredTriggerAction;
import com.gempukku.lotro.logic.effects.KillEffect;
import com.gempukku.lotro.logic.modifiers.ModifiersLogic;
import com.gempukku.lotro.logic.timing.EffectResult;
import com.gempukku.lotro.logic.timing.UnrespondableEffect;
import com.gempukku.lotro.logic.timing.results.ZeroVitalityResult;

import java.util.*;

public class CharacterDeathRule {
    private final Set<PhysicalCard> _charactersAlreadyOnWayToDeath = new HashSet<>();
    private final DefaultActionsEnvironment _actionsEnvironment;
    // The count of changes this was last checked at.  Asked after every effect; with the count as it was, the
    // same characters are at no vitality as were, and they are already on their way.
    private long _checkedAt = -1;

    public CharacterDeathRule(DefaultActionsEnvironment actionsEnvironment) {
        _actionsEnvironment = actionsEnvironment;
    }

    public void applyRule() {
        _actionsEnvironment.addAlwaysOnActionProxy(
                new AbstractActionProxy() {
                    @Override
                    public List<? extends RequiredTriggerAction> getRequiredAfterTriggers(LotroGame game, EffectResult effectResult) {
                        if (effectResult.getType() == EffectResult.Type.ZERO_VITALITY) {
                            ZeroVitalityResult zeroVitalityResult = (ZeroVitalityResult) effectResult;
                            final Set<PhysicalCard> characters = zeroVitalityResult.getCharacters();
                            RequiredTriggerAction action = new RequiredTriggerAction(null);
                            action.setText("Character death");
                            action.appendEffect(
                                    new KillEffect(characters, (PhysicalCard) null, KillEffect.Cause.WOUNDS));
                            action.appendEffect(
                                    new UnrespondableEffect() {
                                        @Override
                                        protected void doPlayEffect(LotroGame game) {
                                            _charactersAlreadyOnWayToDeath.removeAll(characters);
                                            // one that outlived its death is to be found again
                                            _checkedAt = -1;
                                        }
                                    });

                            return Collections.singletonList(action);
                        }
                        return null;
                    }
                });
    }

    public void checkCharactersZeroVitality(LotroGame game) {
        if (game.getGameState().getCurrentPhase() != Phase.PUT_RING_BEARER && game.getGameState().getCurrentPhase() != Phase.BETWEEN_TURNS) {
            final long count = game.getModifiersQuerying().getChangeCount(game);
            final boolean settled = count != -1 && count == _checkedAt;
            if (settled && !ModifiersLogic.verifiesKeptAnswers())
                return;
            _checkedAt = count;
            Collection<PhysicalCard> characters = Filters.filterActive(game,
                    Filters.or(CardType.ALLY, CardType.COMPANION, CardType.MINION));

            Set<PhysicalCard> deadChars = new HashSet<>();
            for (PhysicalCard character : characters)
                if (!_charactersAlreadyOnWayToDeath.contains(character) && game.getModifiersQuerying().getVitality(game, character) <= 0)
                    deadChars.add(character);

            if (settled && !deadChars.isEmpty())
                throw new IllegalStateException("A character came to no vitality with no change counted: " + deadChars);
            if (!deadChars.isEmpty()) {
                _charactersAlreadyOnWayToDeath.addAll(deadChars);
                game.getActionsEnvironment().emitEffectResult(new ZeroVitalityResult(deadChars));
            }
        }
    }
}
