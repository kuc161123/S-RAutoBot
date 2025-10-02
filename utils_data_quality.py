"""
Data quality utilities for consistent feature generation and labeling.

Capabilities:
- Ensure only fully closed candles are used for features/labels.
- Sanitize OHLCV (drop invalid rows, clip obvious anomalies lightly).
- Apply spec-aware price rounding using config.yaml symbol_meta tick_size.
- Maintain lightweight QA counters (optionally persisted later).
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Dict

import pandas as pd
import yaml

_CONFIG_CACHE: Dict | None = None
_QA_COUNTERS: Dict[str, int] = {
    'closed_bars_used': 0,
    'dropped_zero_volume': 0,
    'rounded_price_rows': 0,
}


def load_config_cached() -> Dict:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        with open('config.yaml', 'r') as f:
            _CONFIG_CACHE = yaml.safe_load(f) or {}
    return _CONFIG_CACHE


def get_symbol_specs(symbol: str) -> Dict[str, float]:
    cfg = load_config_cached()
    meta = (cfg.get('symbol_meta') or {}).copy()
    default = meta.get('default', {})
    spec = default.copy()
    spec.update(meta.get(symbol, {}))
    return {
        'tick_size': float(spec.get('tick_size', 0.0001)),
        'qty_step': float(spec.get('qty_step', 0.001)),
        'min_qty': float(spec.get('min_qty', 0.001)),
    }


def _round_to_tick(x: float, tick: float) -> float:
    if tick <= 0:
        return float(x)
    # Round to nearest tick with correct decimals
    import decimal
    d_step = decimal.Decimal(str(tick))
    places = -d_step.as_tuple().exponent if d_step.as_tuple().exponent < 0 else 0
    return round(round(x / tick) * tick, places)


def ensure_closed_candles(df: pd.DataFrame, timeframe_minutes: int = 15) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    idx = df.index
    if idx.tz is None:
        # Assume UTC if naive
        last_ts = idx[-1].tz_localize('UTC')
    else:
        last_ts = idx[-1]
    now = datetime.now(timezone.utc)
    # If the last candle is not older than timeframe, drop it (likely in-progress)
    if (now - last_ts).total_seconds() < timeframe_minutes * 60 and len(df) > 1:
        _QA_COUNTERS['closed_bars_used'] += 1
        return df.iloc[:-1]
    _QA_COUNTERS['closed_bars_used'] += 1
    return df


def sanitize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    # Drop rows with non-positive or NaN volume
    before = len(df)
    clean = df.copy()
    clean = clean[clean['volume'].fillna(0) > 0]
    _QA_COUNTERS['dropped_zero_volume'] += max(0, before - len(clean))
    # Basic fill forward for any NaN price fields (rare)
    clean[['open', 'high', 'low', 'close']] = clean[['open', 'high', 'low', 'close']].ffill()
    return clean


def round_df_prices(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    tick = get_symbol_specs(symbol).get('tick_size', 0.0001)
    rounded = df.copy()
    for col in ('open', 'high', 'low', 'close'):
        rounded[col] = rounded[col].apply(lambda x: _round_to_tick(float(x), tick))
    _QA_COUNTERS['rounded_price_rows'] += len(rounded)
    return rounded


def prepare_df_for_features(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Pipeline used by strategies before computing features/signals."""
    cfg = load_config_cached()
    tf = 15
    try:
        tf = int(str(cfg.get('trade', {}).get('timeframe', '15')).replace('m', '').replace("'", ''))
    except Exception:
        tf = 15
    step1 = ensure_closed_candles(df, tf)
    step2 = sanitize_ohlcv(step1)
    step3 = round_df_prices(step2, symbol)
    return step3


def get_qa_counters() -> Dict[str, int]:
    return dict(_QA_COUNTERS)

