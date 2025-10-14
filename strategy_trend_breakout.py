from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
import logging
from strategy_pullback import _pivot_low, _pivot_high  # reuse pivot helpers


@dataclass
class TrendSettings:
    atr_len: int = 14
    rr: float = 2.5               # 1:2.5 R:R
    sl_atr_mult: float = 1.5      # ATR leg for hybrid SL
    confirm_candles: int = 2      # require 2 bullish/bearish candles
    pivot_l: int = 3              # pivot sensitivity for HL/LH
    pivot_r: int = 3
    breakout_buffer_atr: float = 0.1  # require break beyond S/R by this ATR
    pivot_buffer_atr: float = 0.05    # SL buffer beyond pivot (hybrid)


@dataclass
class Signal:
    side: str
    entry: float
    sl: float
    tp: float
    reason: str
    meta: dict


def _atr(df: pd.DataFrame, n: int = 14) -> float:
    prev = df['close'].shift()
    tr = np.maximum(df['high'] - df['low'], np.maximum((df['high'] - prev).abs(), (df['low'] - prev).abs()))
    atr = tr.rolling(n).mean().iloc[-1]
    return float(atr) if pd.notna(atr) else float(tr.iloc[-1])


def _last_break_levels(df: pd.DataFrame, L: int, R: int) -> tuple[float, float]:
    """Return most recent pivot-based resistance and support levels (zones simplified to last pivot)."""
    high, low = df['high'], df['low']
    ph = pd.Series(_pivot_high(high, L, R), index=df.index).dropna()
    pl = pd.Series(_pivot_low(low, L, R), index=df.index).dropna()
    last_res = float(ph.iloc[-1]) if len(ph) else float(high.rolling(L+R+1).max().iloc[-1])
    last_sup = float(pl.iloc[-1]) if len(pl) else float(low.rolling(L+R+1).min().iloc[-1])
    return last_res, last_sup


def _two_candle_confirmation(df: pd.DataFrame, side: str, n: int) -> bool:
    o = df['open'].iloc[-n:]
    c = df['close'].iloc[-n:]
    if len(o) < n:
        return False
    if side == 'long':
        return bool(((c > o).sum() == n))
    else:
        return bool(((c < o).sum() == n))


def detect_signal(df: pd.DataFrame, s: TrendSettings, symbol: str = "") -> Optional[Signal]:
    """Trend Pullback: Break S/R → HL/LH → 2 candles confirm → enter.

    - Works on 15m bars (assumes df is 15m window)
    - Hybrid SL: pivot-based with ATR buffer vs ATR leg; choose conservative
    - TP: rr=2.5
    """
    logger = logging.getLogger(__name__)
    if df is None or len(df) < 80:
        return None

    high, low, close = df['high'], df['low'], df['close']
    price = float(close.iloc[-1])
    atr = _atr(df, s.atr_len)
    if atr <= 0:
        return None

    res, sup = _last_break_levels(df.iloc[:-1], s.pivot_l, s.pivot_r)
    buf = float(s.breakout_buffer_atr) * float(atr)

    # Detect prior breakout context within recent window
    long_ctx = bool(close.iloc[-2] <= res and close.iloc[-1] > res + buf)
    short_ctx = bool(close.iloc[-2] >= sup and close.iloc[-1] < sup - buf)

    meta = {'atr': float(atr), 'resistance': float(res), 'support': float(sup)}

    # Try LONG: after resistance break, wait for HL above broken level then 2 green candles
    if long_ctx:
        # Find recent HL pivot above broken resistance within last ~10 bars
        pl = pd.Series(_pivot_low(low, s.pivot_l, s.pivot_r), index=df.index)
        hl_idx = None
        for i in range(len(df)-2, max(len(df)-12, 0), -1):
            try:
                if not np.isnan(pl.iloc[i]) and float(pl.iloc[i]) > (res - 1e-9):
                    hl_idx = i
                    break
            except Exception:
                continue
        if hl_idx is None:
            return None
        hl_price = float(pl.iloc[hl_idx])
        # Require 2 bullish confirms after HL
        if not _two_candle_confirmation(df, 'long', s.confirm_candles):
            return None
        entry = price
        # Hybrid SL: max of (pivot HL - buffer, entry - atr_mult*atr)
        sl_pivot = hl_price - (s.pivot_buffer_atr * atr)
        sl_atr = entry - (s.sl_atr_mult * atr)
        sl = min(sl_pivot, sl_atr)  # more conservative (farther)
        # Enforce 1% min stop distance
        min_stop = entry * 0.01
        if (entry - sl) < min_stop:
            sl = entry - min_stop
        if entry <= sl:
            return None
        R = entry - sl
        tp = entry + s.rr * R
        meta.update({'break_level': float(res), 'hl_price': float(hl_price), 'break_dist_atr': float((entry - res)/atr), 'retrace_depth_atr': float((entry - hl_price)/max(1e-9, atr)), 'confirm_candles': int(s.confirm_candles)})
        return Signal('long', float(entry), float(sl), float(tp), 'Trend Pullback LONG (break→HL→2 green)', meta)

    # Try SHORT: after support break, wait for LH below broken level then 2 red candles
    if short_ctx:
        ph = pd.Series(_pivot_high(high, s.pivot_l, s.pivot_r), index=df.index)
        lh_idx = None
        for i in range(len(df)-2, max(len(df)-12, 0), -1):
            try:
                if not np.isnan(ph.iloc[i]) and float(ph.iloc[i]) < (sup + 1e-9):
                    lh_idx = i
                    break
            except Exception:
                continue
        if lh_idx is None:
            return None
        lh_price = float(ph.iloc[lh_idx])
        if not _two_candle_confirmation(df, 'short', s.confirm_candles):
            return None
        entry = price
        sl_pivot = lh_price + (s.pivot_buffer_atr * atr)
        sl_atr = entry + (s.sl_atr_mult * atr)
        sl = max(sl_pivot, sl_atr)
        min_stop = entry * 0.01
        if (sl - entry) < min_stop:
            sl = entry + min_stop
        if sl <= entry:
            return None
        R = sl - entry
        tp = entry - s.rr * R
        meta.update({'break_level': float(sup), 'lh_price': float(lh_price), 'break_dist_atr': float((sup - entry)/atr), 'retrace_depth_atr': float((lh_price - entry)/max(1e-9, atr)), 'confirm_candles': int(s.confirm_candles)})
        return Signal('short', float(entry), float(sl), float(tp), 'Trend Pullback SHORT (break→LH→2 red)', meta)

    return None
