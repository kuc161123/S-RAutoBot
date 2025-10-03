"""
Scalping Strategy (Phase 0 - Phantom-only pilot)

Design: VWAP pullback + ORB continuation on the current timeframe (recommended 1–3m in future).
For now, operates on the provided DataFrame with minimal dependencies.

Outputs a Signal-like object compatible with existing flow.
"""
from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np


@dataclass
class ScalpSettings:
    rr: float = 1.5
    atr_len: int = 7
    ema_fast: int = 8
    ema_slow: int = 21
    vwap_window: int = 100
    min_bb_width_pct: float = 0.7  # 70th percentile of BB width
    vol_ratio_min: float = 1.3     # 1.3x 20-bar avg
    wick_ratio_min: float = 0.3
    vwap_dist_atr_max: float = 0.6 # max normalized distance to VWAP


@dataclass
class ScalpSignal:
    side: str
    entry: float
    sl: float
    tp: float
    reason: str
    meta: dict


def _atr(df: pd.DataFrame, n: int) -> pd.Series:
    prev = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev).abs(),
        (df["low"] - prev).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _vwap(df: pd.DataFrame, window: int) -> pd.Series:
    # Simple rolling VWAP (not session anchored) for short windows
    tp = (df['high'] + df['low'] + df['close']) / 3
    vol = df['volume'].clip(lower=0.0)
    pv = tp * vol
    num = pv.rolling(window).sum()
    den = vol.rolling(window).sum().replace(0, np.nan)
    return num / den


def detect_scalp_signal(df: pd.DataFrame, s: ScalpSettings = ScalpSettings(), symbol: str = "") -> Optional[ScalpSignal]:
    if df is None or len(df) < max(s.vwap_window, 50):
        return None

    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']

    atr = _atr(df, s.atr_len)
    ema_f = _ema(close, s.ema_fast)
    ema_s = _ema(close, s.ema_slow)
    vwap = _vwap(df, s.vwap_window)

    # BB width percentile proxy (use rolling std percentage)
    std20 = close.rolling(20).std()
    bbw = (std20 / close).fillna(0)
    if len(bbw) < 30:
        return None
    last_bbw = float(bbw.iloc[-1])
    bbw_pct = (bbw <= last_bbw).mean()  # percentile of current width

    # Volume spike ratio
    vol20 = volume.rolling(20).mean()
    vol_ratio = (volume.iloc[-1] / vol20.iloc[-1]) if vol20.iloc[-1] > 0 else 1.0

    # Wick ratios (last candle)
    c = close.iloc[-1]
    o = df['open'].iloc[-1]
    h = high.iloc[-1]
    l = low.iloc[-1]
    rng = max(1e-9, h - l)
    upper_w = max(0.0, h - max(c, o)) / rng
    lower_w = max(0.0, min(c, o) - l) / rng

    # Trend check (continuation)
    ema_aligned_up = (c > ema_f.iloc[-1] > ema_s.iloc[-1])
    ema_aligned_dn = (c < ema_f.iloc[-1] < ema_s.iloc[-1])

    # Distance to VWAP (normalized by ATR)
    cur_vwap = float(vwap.iloc[-1]) if not np.isnan(vwap.iloc[-1]) else c
    cur_atr = float(atr.iloc[-1]) if atr.iloc[-1] > 0 else max(1e-9, rng)
    dist_vwap_atr = abs(c - cur_vwap) / cur_atr

    # ORB filter (first 20 bars): skip if not above/below initial range for continuation bias
    # Keep simple: only check if enough bars
    orb_ok = True
    if len(df) >= 40:
        first_high = high.iloc[:20].max()
        first_low = low.iloc[:20].min()
        if ema_aligned_up and c <= first_high:
            orb_ok = False
        if ema_aligned_dn and c >= first_low:
            orb_ok = False

    # Long scalp candidate
    if ema_aligned_up and bbw_pct >= s.min_bb_width_pct and vol_ratio >= s.vol_ratio_min and lower_w >= s.wick_ratio_min and dist_vwap_atr <= s.vwap_dist_atr_max and orb_ok:
        # Stop below VWAP/EMA band
        sl = min(cur_vwap, ema_s.iloc[-1]) - 0.8 * cur_atr
        entry = c
        if entry <= sl:
            return None
        tp = entry + s.rr * (entry - sl)
        return ScalpSignal(
            side='long', entry=entry, sl=sl, tp=tp,
            reason=f"SCALP: VWAP pullback + EMA band + vol spike",
            meta={
                'vwap': cur_vwap,
                'atr': cur_atr,
                'bbw_pct': bbw_pct,
                'vol_ratio': vol_ratio,
                'dist_vwap_atr': dist_vwap_atr
            }
        )

    # Short scalp candidate
    if ema_aligned_dn and bbw_pct >= s.min_bb_width_pct and vol_ratio >= s.vol_ratio_min and upper_w >= s.wick_ratio_min and dist_vwap_atr <= s.vwap_dist_atr_max and orb_ok:
        sl = max(cur_vwap, ema_s.iloc[-1]) + 0.8 * cur_atr
        entry = c
        if sl <= entry:
            return None
        tp = entry - s.rr * (sl - entry)
        return ScalpSignal(
            side='short', entry=entry, sl=sl, tp=tp,
            reason=f"SCALP: VWAP pullback + EMA band + vol spike",
            meta={
                'vwap': cur_vwap,
                'atr': cur_atr,
                'bbw_pct': bbw_pct,
                'vol_ratio': vol_ratio,
                'dist_vwap_atr': dist_vwap_atr
            }
        )

    return None
