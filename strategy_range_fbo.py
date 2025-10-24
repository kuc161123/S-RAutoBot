"""
Range FBO (Failed Breakout) detector.

Detects when price breaks out of a recent range and then re-enters the range within a short window
— a common failed-breakout pattern suitable for mean-reverting back to the mid or opposite band.

This module is intentionally lightweight: it uses 15m data only and returns a simple Signal.
Execution is phantom-only initially; Trend BE/TP1 plumbing is reused later if enabled.
"""
from dataclasses import dataclass
from typing import Optional, Dict

import numpy as np


@dataclass
class Signal:
    side: str  # "long" or "short"
    entry: float
    sl: float
    tp: float
    reason: str
    meta: Dict


def _atr(series_high, series_low, series_close, length: int = 14) -> float:
    try:
        prev = series_close.shift()
        tr = np.maximum(series_high - series_low, np.maximum((series_high - prev).abs(), (series_low - prev).abs()))
        atr = float(tr.rolling(length).mean().iloc[-1])
        return atr
    except Exception:
        return 0.0


def detect_range_fbo_signal(df, settings: Dict, symbol: str) -> Optional[Signal]:
    """
    Detect a range failed-breakout re-entry and propose a range trade back to mid/opposite band.

    Heuristics (15m):
    - Range bands = rolling N-high/low; width_pct in sane bounds
    - FBO up: prior bar closed above high, current closes back below high → short
    - FBO down: prior bar closed below low, current closes back above low → long
    """
    try:
        if df is None or len(df) < 50:
            return None
        s = settings or {}
        lookback = int(s.get('lookback', 40))
        width_min = float(s.get('width_min_pct', 0.01))   # 1%
        width_max = float(s.get('width_max_pct', 0.08))   # 8%
        reentry_max_bars = int(s.get('reentry_max_bars', 4))
        atr_buf = float(s.get('sl_atr_buf', 0.6))

        high = df['high']
        low = df['low']
        close = df['close']
        price = float(close.iloc[-1])

        rng_high = float(high.rolling(lookback).max().iloc[-2])  # use previous bar range
        rng_low = float(low.rolling(lookback).min().iloc[-2])
        if rng_high <= 0 or rng_low <= 0 or rng_high <= rng_low:
            return None
        width_pct = (rng_high - rng_low) / max(1e-9, rng_low)
        if not (width_min <= width_pct <= width_max):
            return None

        prev_close = float(close.iloc[-2])
        cur_close = float(close.iloc[-1])
        # Count bars since last outside event (coarse):
        recent_closes = close.tail(reentry_max_bars + 1).values
        broke_up = any(c > rng_high for c in recent_closes[:-1])
        broke_dn = any(c < rng_low for c in recent_closes[:-1])

        atr = _atr(high, low, close, 14)
        mid = (rng_high + rng_low) / 2.0

        # Up-break fail → re-enter below high → short
        if broke_up and cur_close < rng_high <= prev_close:
            entry = cur_close
            sl = rng_high + atr_buf * atr
            tp = rng_low  # target opposite band for full-phantom simulation
            wick_ratio = 0.0
            try:
                rng = float(high.iloc[-1] - low.iloc[-1])
                wick_ratio = float((high.iloc[-1] - cur_close) / max(1e-9, rng))
            except Exception:
                pass
            meta = {
                'range_high': rng_high,
                'range_low': rng_low,
                'range_mid': mid,
                'range_width_pct': width_pct,
                'fbo_type': 'up_fail',
                'wick_ratio': wick_ratio,
                'retest_ok': bool(cur_close < rng_high and prev_close > rng_high)
            }
            return Signal('short', entry, sl, tp, f"FBO short: re-entered below range high {rng_high:.4f}", meta)

        # Down-break fail → re-enter above low → long
        if broke_dn and cur_close > rng_low >= prev_close:
            entry = cur_close
            sl = rng_low - atr_buf * atr
            tp = rng_high
            wick_ratio = 0.0
            try:
                rng = float(high.iloc[-1] - low.iloc[-1])
                wick_ratio = float((cur_close - low.iloc[-1]) / max(1e-9, rng))
            except Exception:
                pass
            meta = {
                'range_high': rng_high,
                'range_low': rng_low,
                'range_mid': mid,
                'range_width_pct': width_pct,
                'fbo_type': 'down_fail',
                'wick_ratio': wick_ratio,
                'retest_ok': bool(cur_close > rng_low and prev_close < rng_low)
            }
            return Signal('long', entry, sl, tp, f"FBO long: re-entered above range low {rng_low:.4f}", meta)

        return None
    except Exception:
        return None

