from dataclasses import dataclass
from typing import Optional, Dict
import numpy as np
import pandas as pd


@dataclass
class TrendSettings:
    channel_len: int = 20
    atr_len: int = 14
    breakout_k_atr: float = 0.3   # buffer over channel by ATR
    sl_atr_mult: float = 1.5      # SL distance in ATR from entry
    rr: float = 2.5               # fixed R:R
    use_ema_stack: bool = True


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


def _ema(series: pd.Series, n: int) -> float:
    if len(series) < n:
        return float(series.iloc[-1])
    return float(series.ewm(span=n, adjust=False).mean().iloc[-1])


def detect_signal(df: pd.DataFrame, s: TrendSettings, symbol: str = "") -> Optional[Signal]:
    if df is None or len(df) < max(60, s.channel_len + 2):
        return None

    close = df['close']
    high = df['high']
    low = df['low']

    # Donchian channel excluding current bar (to avoid lookahead)
    hh = high.shift(1).rolling(s.channel_len).max().iloc[-1]
    ll = low.shift(1).rolling(s.channel_len).min().iloc[-1]
    if not np.isfinite(hh) or not np.isfinite(ll):
        return None

    atr = _atr(df, s.atr_len)
    price = float(close.iloc[-1])
    k = s.breakout_k_atr

    # Optional EMA stack filter
    ema_ok = True
    if s.use_ema_stack:
        ema20 = _ema(close, 20)
        ema50 = _ema(close, 50)
        ema_ok = (price > ema20 > ema50) or (price < ema20 < ema50)

    meta = {
        'channel_len': s.channel_len,
        'atr': float(atr),
        'upper_channel': float(hh),
        'lower_channel': float(ll),
        'ema_ok': bool(ema_ok),
    }

    # Long breakout
    if price > hh + k * atr and ema_ok:
        entry = price
        sl = entry - s.sl_atr_mult * atr
        R = entry - sl
        if R <= 0:
            return None
        tp = entry + s.rr * R
        reason = f"Donchian breakout LONG > HH({s.channel_len}) + {k}*ATR"
        meta.update({'breakout_dist_atr': float((price - hh) / max(1e-9, atr))})
        return Signal('long', float(entry), float(sl), float(tp), reason, meta)

    # Short breakout
    if price < ll - k * atr and ema_ok:
        entry = price
        sl = entry + s.sl_atr_mult * atr
        R = sl - entry
        if R <= 0:
            return None
        tp = entry - s.rr * R
        reason = f"Donchian breakout SHORT < LL({s.channel_len}) - {k}*ATR"
        meta.update({'breakout_dist_atr': float((ll - price) / max(1e-9, atr))})
        return Signal('short', float(entry), float(sl), float(tp), reason, meta)

    return None

