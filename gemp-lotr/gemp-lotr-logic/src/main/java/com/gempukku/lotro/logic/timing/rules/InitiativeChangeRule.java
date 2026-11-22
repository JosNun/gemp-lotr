package com.gempukku.lotro.logic.timing.rules;

import com.gempukku.lotro.common.Phase;
import com.gempukku.lotro.logic.modifiers.ModifiersLogic;
import com.gempukku.lotro.common.Side;
import com.gempukku.lotro.game.state.LotroGame;
import com.gempukku.lotro.logic.timing.results.InitiativeChangeResult;

public class InitiativeChangeRule {
    private Side _initiativeSide = null;

    // The count of changes this was last checked at with nothing found: asked after every effect, and most
    // effects change nothing that initiative reads.
    private long _checkedAt = -1;

    public void checkInitiativeChange(LotroGame game) {
        if (game.getGameState().getCurrentPhase() != Phase.PUT_RING_BEARER) {
            final long count = game.getModifiersQuerying().getChangeCount(game);
            final boolean settled = count != -1 && count == _checkedAt;
            if (settled && !ModifiersLogic.verifiesKeptAnswers())
                return;
            Side initiativeSide = game.getModifiersQuerying().hasInitiative(game);
            _checkedAt = count;
            if (initiativeSide != _initiativeSide) {
                if (settled)
                    throw new IllegalStateException("Initiative changed with no change counted");
                _initiativeSide = initiativeSide;
                game.getActionsEnvironment().emitEffectResult(new InitiativeChangeResult(_initiativeSide));
            }
        }
    }
}
