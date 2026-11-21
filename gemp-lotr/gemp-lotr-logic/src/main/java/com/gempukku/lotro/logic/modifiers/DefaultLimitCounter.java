package com.gempukku.lotro.logic.modifiers;

public class DefaultLimitCounter implements LimitCounter {
    private int _count;
    // ModifiersLogic's count of changes: a limit used up changes what a modifier gives
    private final long[] _epoch;

    public DefaultLimitCounter() {
        this(null);
    }

    public DefaultLimitCounter(long[] epoch) {
        _epoch = epoch;
    }

    @Override
    public int incrementToLimit(int limit, int incrementBy) {
        int maxIncrement = limit - _count;
        int finalIncrement = Math.min(maxIncrement, incrementBy);
        _count += finalIncrement;
        if (_epoch != null)
            _epoch[0]++;
        return finalIncrement;
    }

    @Override
    public int getUsedLimit() {
        return _count;
    }
}
