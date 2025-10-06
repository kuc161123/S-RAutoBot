"""
Strategy-specific regime scoring utilities.

Lightweight, heuristic scores that rank how suitable the current market
context is for each strategy (0-100). These run fast and rely only on
per-symbol OHLCV frames, avoiding heavy recomputation or cross-module
dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import pandas as pd


def _ema(series: pd.Series, n: int) -> float:
    if len(series) < n:
        return float(series.iloc[-1])
    return float(series.ewm(span=n, adjust=False).mean().iloc[-1])


def _atr(df: pd.DataFrame, n: int = 14) -> float:
    if len(df) < 2:
        return float(max(1e-9, (df['high'].iloc[-1] - df['low'].iloc[-1])))
    prev_close = df['close'].shift()
    tr = np.maximum(df['high'] - df['low'], np.maximum((df['high'] - prev_close).abs(), (df['low'] - prev_close).abs()))
    atr = tr.rolling(n).mean().iloc[-1]
    return float(atr) if pd.notna(atr) else float(tr.iloc[-1])


def _slope_strength(series: pd.Series, window: int = 20) -> float:
    if len(series) < window:
        return 0.0
    y = series.tail(window).values
    x = np.arange(len(y))
    try:
        slope = np.polyfit(x, y, 1)[0]
    except Exception:
        slope = 0.0
    avg = float(np.mean(y)) if np.mean(y) else 1.0
    return float(np.clip((slope / avg) * 100.0, -100.0, 100.0))


def score_pullback_regime(df: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
    """Score trend-following (pullback) suitability.

    Components (weights in brackets):
    - Trend slope % over 20 bars (40)
    - EMA stack alignment 8>21>50 or 8<21<50 (35)
    - Volatility sanity via ATR% of price (15)
    - Last candle body/ATR as weak confirmation (10)
    """
    if df is None or df.empty:
        return 0.0, {}
    close = df['close']
    high = df['high']
    low = df['low']

    price = float(close.iloc[-1])
    atr = max(1e-9, _atr(df, 14))
    atr_pct = float(np.clip((atr / max(1e-9, price)) * 100.0, 0.0, 100.0))

    slope_pct = _slope_strength(close, 20)
    ema8 = _ema(close, 8)
    ema21 = _ema(close, 21)
    ema50 = _ema(close, 50)
    bullish_stack = (price > ema8 > ema21 > ema50)
    bearish_stack = (price < ema8 < ema21 < ema50)
    stack_score = 100.0 if (bullish_stack or bearish_stack) else 30.0 if ((ema8 > ema21) or (ema8 < ema21)) else 0.0

    rng = float(high.iloc[-1] - low.iloc[-1])
    body = float(abs(close.iloc[-1] - df['open'].iloc[-1]))
    body_atr = float(np.clip((body / max(atr, 1e-9)) * 100.0, 0.0, 100.0))

    # Penalize extreme vol a bit for continuation setups
    vol_score = 100.0 - min(40.0, max(0.0, atr_pct - 2.0) * 5.0)  # >2% ATR of price starts to reduce score
    trend_score = np.clip((abs(slope_pct) / 50.0) * 100.0, 0.0, 100.0)

    score = (0.40 * trend_score) + (0.35 * stack_score) + (0.15 * vol_score) + (0.10 * body_atr)
    return float(np.clip(score, 0.0, 100.0)), {
        'trend_strength': trend_score,
        'ema_stack': stack_score,
        'atr_pct': atr_pct,
        'body_atr': body_atr,
    }


def score_mr_regime(df: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
    """Score mean-reversion suitability using simple band heuristics over 30 bars.

    Components (weights in brackets):
    - Range width in ATR within [1.0, 4.0] ATR (40)
    - Touches near HH/LL in last 30 bars (30)
    - In-range ratio of closes between HH/LL (20)
    - Volatility regime comfort via ATR% of price (10)
    """
    if df is None or len(df) < 30:
        return 0.0, {}
    tail = df.tail(30)
    close = tail['close']
    high = tail['high']
    low = tail['low']

    price = float(close.iloc[-1])
    atr = max(1e-9, _atr(tail, 14))
    atr_pct = float(np.clip((atr / max(1e-9, price)) * 100.0, 0.0, 100.0))

    HH = float(high.max()); LL = float(low.min())
    width = HH - LL
    width_atr = width / atr if atr > 0 else 0.0
    # Score width best around 2 ATR (bell-like)
    width_score = float(np.clip(100.0 - (abs(width_atr - 2.0) * 35.0), 0.0, 100.0))

    tol = width * 0.02  # 2% tolerance
    upper_touches = int(((high >= (HH - tol)) & (high <= (HH + tol))).sum())
    lower_touches = int(((low <= (LL + tol)) & (low >= (LL - tol))).sum())
    touch_score = float(np.clip(((upper_touches + lower_touches) / 8.0) * 100.0, 0.0, 100.0))

    in_range_ratio = float(((close >= LL) & (close <= HH)).mean())
    in_range_score = float(in_range_ratio * 100.0)

    vol_score = 100.0 - min(40.0, max(0.0, atr_pct - 1.5) * 40.0)  # above ~1.5% ATR of price begins to hurt MR

    score = (0.40 * width_score) + (0.30 * touch_score) + (0.20 * in_range_score) + (0.10 * vol_score)
    return float(np.clip(score, 0.0, 100.0)), {
        'width_atr': width_atr,
        'touches': float(upper_touches + lower_touches),
        'in_range_pct': in_range_score,
        'atr_pct': atr_pct,
    }

