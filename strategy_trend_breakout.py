from dataclasses import dataclass
from typing import Optional, Dict
import numpy as np
import pandas as pd
import logging
from strategy_pullback import _pivot_low, _pivot_high  # reuse robust pivot helpers
try:
    # Lightweight HTF S/R filter utilities
    from multi_timeframe_sr import should_use_mtf_level
    _HTF_AVAILABLE = True
except Exception:
    _HTF_AVAILABLE = False


@dataclass
class TrendSettings:
    channel_len: int = 20
    atr_len: int = 14
    breakout_k_atr: float = 0.3   # buffer over channel by ATR
    sl_atr_mult: float = 1.5      # SL distance in ATR from entry
    rr: float = 2.5               # base R:R (will be adjusted by volatility schedule)
    use_ema_stack: bool = True
    use_htf_sr_filter: bool = True  # Avoid breakouts into nearby 1H/4H SR
    # Entry quality filters
    require_range_expansion: bool = True
    range_expansion_min: float = 1.2  # today's range vs 20-bar median
    require_retest: bool = True
    retest_max_dist_atr: float = 0.5  # breakout distance (ATR) upper bound for retest-style entries


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
    logger = logging.getLogger(__name__)
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

    # ATR and ATR series (for volatility percentile)
    prev = close.shift()
    trarr = np.maximum(high - low, np.maximum((high - prev).abs(), (low - prev).abs()))
    atr_series = trarr.rolling(s.atr_len).mean()
    atr = float(atr_series.iloc[-1]) if not pd.isna(atr_series.iloc[-1]) else float(trarr.iloc[-1])
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

    # Compute daily range expansion vs 20-bar median
    try:
        rng_today = float(high.iloc[-1] - low.iloc[-1])
        med_range = float((high - low).rolling(20).median().iloc[-1]) if len(df) >= 20 else rng_today
        range_expansion = rng_today / max(1e-9, med_range)
    except Exception:
        range_expansion = 1.0

    # Long breakout (with filters)
    if price > hh + k * atr and ema_ok:
        # Range expansion filter
        if s.require_range_expansion and not (range_expansion >= s.range_expansion_min):
            return None
        # Retest-style proximity filter (avoid late chase far from HH)
        bdist_atr = (price - float(hh)) / max(1e-9, float(atr))
        if s.require_retest and not (bdist_atr <= s.retest_max_dist_atr):
            return None
        entry = price
        # Volatility-aware SL buffer (hybrid, similar to MR)
        try:
            vol_pct = 0.5
            valid = atr_series.dropna()
            if len(valid) >= 30:
                current_atr = float(atr)
                vol_pct = float((valid < current_atr).sum() / len(valid))
            if vol_pct > 0.8:
                vol_mult = 1.4
            elif vol_pct > 0.6:
                vol_mult = 1.2
            elif vol_pct < 0.2:
                vol_mult = 0.8
            else:
                vol_mult = 1.0
        except Exception:
            vol_mult = 1.0
        adjusted = float(s.sl_atr_mult) * float(vol_mult)

        # Hybrid SL options
        # 1) Breakout level (HH) with small buffer
        sl_opt1 = float(hh) - (adjusted * 0.3 * float(atr))
        # 2) ATR-based from entry
        sl_opt2 = float(entry) - (adjusted * float(atr))
        # 3) Structural: recent pivot low with larger buffer
        try:
            pl = pd.Series(_pivot_low(low, 5, 5), index=df.index).dropna()
            pivot_low_val = float(pl.iloc[-1]) if len(pl) else float(low.rolling(10).min().iloc[-1])
        except Exception:
            pivot_low_val = float(low.rolling(10).min().iloc[-1])
        sl_opt3 = float(pivot_low_val) - (adjusted * float(atr))

        sl = min(sl_opt1, sl_opt2, sl_opt3)
        # Enforce minimum stop distance (1%)
        min_stop = float(entry) * 0.01
        if (entry - sl) < min_stop:
            sl = float(entry) - min_stop
            try:
                logger.info(f"[{symbol}] Trend SL adjusted to minimum distance (1% from entry)")
            except Exception:
                pass

        R = float(entry) - float(sl)
        if R <= 0:
            return None
        # Volatility-aware RR schedule
        fee_adjustment = 1.00165
        rr_eff = float(s.rr)
        try:
            # vol_pct ~ ATR percentile (higher means current ATR high vs history)
            vol_pct = 0.5
            valid = atr_series.dropna()
            if len(valid) >= 30:
                current_atr = float(atr)
                vol_pct = float((valid < current_atr).sum() / len(valid))
            # Map: low vol -> higher RR, high vol -> lower RR
            if vol_pct < 0.3:
                rr_eff = max(rr_eff, 3.0)
            elif vol_pct > 0.7:
                rr_eff = min(rr_eff, 2.0)
            else:
                rr_eff = float(s.rr)
        except Exception:
            rr_eff = float(s.rr)
        tp = float(entry) + rr_eff * R * fee_adjustment
        reason = f"Donchian breakout LONG > HH({s.channel_len}) + {k}*ATR"
        meta.update({'breakout_dist_atr': float(bdist_atr), 'pivot_low': float(low.rolling(10).min().iloc[-1]), 'range_expansion': float(range_expansion), 'rr_eff': float(rr_eff)})
        # HTF S/R guard: avoid breakouts into nearby resistance
        try:
            if s.use_htf_sr_filter and _HTF_AVAILABLE:
                use_mtf, level, why = should_use_mtf_level(symbol, entry, price, df)
                if use_mtf and ('resistance' in why.lower()):
                    logger.info(f"[{symbol}] Trend LONG blocked by HTF filter: {why}")
                    return None
                if use_mtf:
                    meta['htf_level'] = float(level)
                    meta['htf_reason'] = why
        except Exception:
            pass
        try:
            which = 'pivot' if sl == sl_opt3 else ('atr' if sl == sl_opt2 else 'breakout')
            logger.info(f"[{symbol}] Trend LONG SL method: {which} | entry={entry:.4f} sl={sl:.4f} tp={tp:.4f}")
        except Exception:
            pass
        return Signal('long', float(entry), float(sl), float(tp), reason, meta)

    # Short breakout (with filters)
    if price < ll - k * atr and ema_ok:
        # Range expansion filter
        if s.require_range_expansion and not (range_expansion >= s.range_expansion_min):
            return None
        # Retest-style proximity filter
        bdist_atr = (float(ll) - price) / max(1e-9, float(atr))
        if s.require_retest and not (bdist_atr <= s.retest_max_dist_atr):
            return None
        entry = price
        # Volatility-aware SL buffer (hybrid)
        try:
            vol_pct = 0.5
            valid = atr_series.dropna()
            if len(valid) >= 30:
                current_atr = float(atr)
                vol_pct = float((valid < current_atr).sum() / len(valid))
            if vol_pct > 0.8:
                vol_mult = 1.4
            elif vol_pct > 0.6:
                vol_mult = 1.2
            elif vol_pct < 0.2:
                vol_mult = 0.8
            else:
                vol_mult = 1.0
        except Exception:
            vol_mult = 1.0
        adjusted = float(s.sl_atr_mult) * float(vol_mult)

        # Hybrid SL options
        # 1) Breakout level (LL) with small buffer
        sl_opt1 = float(ll) + (adjusted * 0.3 * float(atr))
        # 2) ATR-based from entry
        sl_opt2 = float(entry) + (adjusted * float(atr))
        # 3) Structural: recent pivot high with larger buffer
        try:
            ph = pd.Series(_pivot_high(high, 5, 5), index=df.index).dropna()
            pivot_high_val = float(ph.iloc[-1]) if len(ph) else float(high.rolling(10).max().iloc[-1])
        except Exception:
            pivot_high_val = float(high.rolling(10).max().iloc[-1])
        sl_opt3 = float(pivot_high_val) + (adjusted * float(atr))

        sl = max(sl_opt1, sl_opt2, sl_opt3)
        # Enforce minimum stop distance (1%)
        min_stop = float(entry) * 0.01
        if (sl - entry) < min_stop:
            sl = float(entry) + min_stop
            try:
                logger.info(f"[{symbol}] Trend SL adjusted to minimum distance (1% from entry)")
            except Exception:
                pass

        R = float(sl) - float(entry)
        if R <= 0:
            return None
        fee_adjustment = 1.00165
        rr_eff = float(s.rr)
        try:
            vol_pct = 0.5
            valid = atr_series.dropna()
            if len(valid) >= 30:
                current_atr = float(atr)
                vol_pct = float((valid < current_atr).sum() / len(valid))
            if vol_pct < 0.3:
                rr_eff = max(rr_eff, 3.0)
            elif vol_pct > 0.7:
                rr_eff = min(rr_eff, 2.0)
            else:
                rr_eff = float(s.rr)
        except Exception:
            rr_eff = float(s.rr)
        tp = float(entry) - rr_eff * R * fee_adjustment
        reason = f"Donchian breakout SHORT < LL({s.channel_len}) - {k}*ATR"
        meta.update({'breakout_dist_atr': float(bdist_atr), 'pivot_high': float(high.rolling(10).max().iloc[-1]), 'range_expansion': float(range_expansion), 'rr_eff': float(rr_eff)})
        # HTF S/R guard: avoid breakouts into nearby support
        try:
            if s.use_htf_sr_filter and _HTF_AVAILABLE:
                use_mtf, level, why = should_use_mtf_level(symbol, entry, price, df)
                if use_mtf and ('support' in why.lower()):
                    logger.info(f"[{symbol}] Trend SHORT blocked by HTF filter: {why}")
                    return None
                if use_mtf:
                    meta['htf_level'] = float(level)
                    meta['htf_reason'] = why
        except Exception:
            pass
        try:
            which = 'pivot' if sl == sl_opt3 else ('atr' if sl == sl_opt2 else 'breakout')
            logger.info(f"[{symbol}] Trend SHORT SL method: {which} | entry={entry:.4f} sl={sl:.4f} tp={tp:.4f}")
        except Exception:
            pass
        return Signal('short', float(entry), float(sl), float(tp), reason, meta)

    return None
