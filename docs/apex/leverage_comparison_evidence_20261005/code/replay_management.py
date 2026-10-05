"""Private, pure management acceleration for an owned offline replay history.

No simulation/engine globals are patched. Both ``advance`` and its management
helper execute their original code, at the actual clock, on every call. Only
immutable closed-history validation and confirmed pivots are memoized.

Prefix proof for the existing confirmed_zigzag implementation: Wilder ATR at
index i depends only on indices <= i. Each zigzag iteration reads that bar,
prior ATR, and extrema already visited; confirmation-lag extrema searches end
at i. Appended frozen Pivots (including their hashes) are never rewritten.
Consequently a complete valid history's pivots with available_at <= now equal
the pivots computed from its closed prefix. We match that *entire prefix*, not
its length/endpoints, before using this property. Tests compare every prefix,
altered futures, backwards clocks, and complete OPEN trade trajectories.

Full-history validation failures disable the optimization, not earlier replay:
the original functions then see only the supplied prefix and actual clock.
"""

from bisect import bisect_right
from types import FunctionType

from . import simulation
from .engine import ATR_PERIOD, DAY, H4
from .models import Candle


def _clone(function, **replacements):
    cloned = FunctionType(
        function.__code__,
        dict(function.__globals__, **replacements),
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )
    cloned.__kwdefaults__ = dict(function.__kwdefaults__ or {})
    return cloned


class _History:
    def __init__(self, bars, seconds):
        self.bars = tuple(bars)
        try:
            self.times = (
                [b.open_time / 1000 + seconds for b in self.bars]
                if all(
                    type(b) is Candle and type(b.open_time) is int and b.open_time >= 0
                    for b in self.bars
                ) else []
            )
        except OverflowError:
            self.times = []
        self.valid = None
        self.pivots = self.pivot_times = None

    def matches(self, bars, now):
        if type(bars) not in (list, tuple) or len(self.times) != len(self.bars):
            return False
        count = bisect_right(self.times, now)
        # Identity matters: numeric equality alone would let False equal 0,
        # bypassing the original validators on an altered candle timestamp.
        return len(bars) == count and all(
            actual is owned for actual, owned in zip(bars, self.bars)
        )


class _ManagementCache:
    """One instance per replay; arbitrary non-owned inputs take the slow path."""

    def __init__(self, daily, execution):
        self.histories = {
            DAY: _History(daily, DAY),
            H4: _History(execution, H4),
        }
        self.validation_hits = self.pivot_hits = self.pivot_builds = 0
        helper = _clone(
            simulation._management_inputs,
            _closed_bars=self._closed_bars,
            confirmed_zigzag=self._pivots,
        )
        self.advance = _clone(simulation.advance, _management_inputs=helper)

    def _closed_bars(self, bars, interval_ms, now):
        history = self.histories.get(interval_ms / 1000)
        if history is not None and history.matches(bars, now):
            if history.valid is None:
                try:
                    simulation._closed_bars(
                        history.bars, interval_ms, max(history.times, default=now)
                    )
                    history.valid = True
                except (ValueError, TypeError, OverflowError):
                    history.valid = False
            if history.valid:
                self.validation_hits += 1
                return list(bars)
        return simulation._closed_bars(bars, interval_ms, now)

    def _pivots(
        self, bars, timeframe_seconds=DAY, now=None,
        atr_period=ATR_PERIOD, atr_multiple=2.0,
    ):
        history = self.histories[H4]
        if (
            timeframe_seconds == H4 and atr_period == ATR_PERIOD
            and atr_multiple == 1 and now is not None
            and history.valid and history.matches(bars, now)
        ):
            if history.pivots is None:
                try:
                    history.pivots = tuple(simulation.confirmed_zigzag(
                        history.bars, H4,
                        history.times[-1] if history.times else now,
                        atr_period=ATR_PERIOD, atr_multiple=1,
                    ))
                    history.pivot_times = [p.available_at for p in history.pivots]
                    self.pivot_builds += 1
                except (ValueError, TypeError, OverflowError):
                    # A bad future or a gap must not contaminate a good prefix.
                    history.valid = False
            if history.valid:
                self.pivot_hits += 1
                return list(history.pivots[:bisect_right(history.pivot_times, now)])
        return simulation.confirmed_zigzag(
            bars, timeframe_seconds, now, atr_period, atr_multiple
        )
