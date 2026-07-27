"""Strategy plugin contract.

One interface, so every candidate is scored by the same execution engine, the same cost
model and the same statistics. The alternative — a bespoke script per strategy, which is
what the ~369 root files are — makes cross-strategy comparison meaningless because each
script has its own subtly different fill assumptions.

Two flavours, because not every candidate is per-symbol:

  PerSymbolStrategy      sees one symbol's dataframe. Most repo strategies.
  CrossSectionalStrategy sees a panel of all symbols at a timestamp. Needed for
                         cross-sectional momentum and funding carry, which rank the
                         universe and cannot be expressed one symbol at a time.

A strategy returns Signals; it does NOT decide fills, costs or position size. Those belong
to execution.py and portfolio.py so they stay identical across candidates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Protocol

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Signal:
    """One entry decision, expressed in BAR INDICES so execution can enforce timing.

    conf_idx  index of the last CLOSED bar that confirms the signal. Execution enters at
              conf_idx+1's open. A strategy must never look past conf_idx.
    side      'long' | 'short'
    sl_dist   stop distance in PRICE units. Execution derives sl/tp from this and rr, so
              the cost model (fee_r = round_trip * entry / sl_dist) applies uniformly.
    rr        reward:risk multiple for the fixed target.
    meta      free-form, carried through to the output for later grouping.
    """
    conf_idx: int
    side: str
    sl_dist: float
    rr: float
    meta: dict = field(default_factory=dict)


class PerSymbolStrategy(Protocol):
    name: str

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add whatever indicator columns detect() needs. Must be causal."""
        ...

    def detect(self, df: pd.DataFrame, **params) -> list[Signal]:
        """Return signals. MUST NOT reference any row after a Signal's conf_idx."""
        ...


class CrossSectionalStrategy(Protocol):
    name: str

    def detect_panel(self, panel: dict[str, pd.DataFrame], **params) -> list[tuple[str, Signal]]:
        """panel maps symbol -> prepared df, all sharing one time index.

        Returns (symbol, Signal) pairs. Same causality rule applies per symbol.
        """
        ...


# ---------------------------------------------------------------------------
# no-lookahead verification — every strategy must pass this before it is scored
# ---------------------------------------------------------------------------
def assert_causal(strategy, df: pd.DataFrame, params: dict | None = None,
                  n_checks: int = 5, seed: int = 0) -> dict:
    """Truncate the dataframe just after a signal and confirm the signal is unchanged.

    This is the check that would have caught the two lookahead bugs found by audit in the
    2026-07 work (a 23h leak in a daily-resampled BTC series, and an entry-candle-close
    leak). It is cheap and it is not optional: a strategy that fails here will look
    brilliant and be worthless.

    THE PAD MATTERS. Pivot-based detectors scan with a `range(start, n - PIVOT_RIGHT)`
    guard (mirroring the live detector). A signal's BOS bar can be as little as one bar
    after its originating scan bar, so truncating at conf_idx+2 puts that scan bar outside
    the truncated loop range and the signal cannot be re-emitted — even though it is
    perfectly causal. Verified empirically on BTCUSDT: a signal at conf_idx 7671 originates
    at scan bar 7670 and reappears at pad>=3, and its pivots are bit-identical using only
    bars 0..7670. So `confirm_lag` is the detector's structural confirmation delay, NOT a
    licence to peek: the pivots a signal uses are all confirmed at or before its scan bar.
    """
    params = params or {}
    prepared = strategy.prepare(df)
    full = strategy.detect(prepared, **params)
    if not full:
        return {"checked": 0, "mismatches": 0, "note": "no signals to check"}

    pad = 2 + int(getattr(strategy, "confirm_lag", 0))
    rng = np.random.default_rng(seed)
    idxs = sorted(rng.choice(len(full), size=min(n_checks, len(full)), replace=False))
    mismatches = []
    for i in idxs:
        sig = full[i]
        cut = sig.conf_idx + pad
        if cut >= len(df):
            continue
        trunc_prepared = strategy.prepare(df.iloc[:cut].copy())
        trunc = strategy.detect(trunc_prepared, **params)
        match = [s for s in trunc if s.conf_idx == sig.conf_idx and s.side == sig.side]
        if not match:
            mismatches.append((sig.conf_idx, "signal disappeared when future data removed"))
            continue
        # stop distance must not move either — that would mean the geometry peeked
        if not np.isclose(match[0].sl_dist, sig.sl_dist, rtol=1e-6):
            mismatches.append((sig.conf_idx, f"sl_dist changed "
                                             f"{sig.sl_dist:.8g} -> {match[0].sl_dist:.8g}"))
    return {"checked": len(idxs), "mismatches": len(mismatches), "detail": mismatches}


def assert_firewall(strategy, df: pd.DataFrame, cutoff: pd.Timestamp,
                    params: dict | None = None) -> dict:
    """Signals before `cutoff` must be identical with and without post-cutoff data.

    Guards the holdout: if a detector's internal state (a global dedup set, a rolling
    normalisation, a resample) depends on later bars, the design period is silently
    contaminated. Verified 6/6 on the incumbent detector during the 2026-07 work.
    """
    params = params or {}
    tcol = "start" if "start" in df.columns else df.index.name
    times = df[tcol] if tcol in df.columns else df.index
    cut_i = int(np.searchsorted(np.asarray(times), np.datetime64(cutoff)))
    if cut_i < 50 or cut_i >= len(df):
        return {"comparable": 0, "identical": True, "note": "cutoff outside data"}

    full = strategy.detect(strategy.prepare(df.copy()), **params)
    trunc = strategy.detect(strategy.prepare(df.iloc[:cut_i].copy()), **params)
    key = lambda s: (s.conf_idx, s.side, round(s.sl_dist, 10))
    a = sorted(key(s) for s in full if s.conf_idx < cut_i - 3)
    b = sorted(key(s) for s in trunc if s.conf_idx < cut_i - 3)
    return {"comparable": len(a), "identical": a == b,
            "only_full": len(set(a) - set(b)), "only_trunc": len(set(b) - set(a))}
