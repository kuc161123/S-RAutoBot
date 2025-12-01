from __future__ import annotations
"""
Scalping Strategy (Phase 0 - Phantom-only pilot)

Design: EVWAP (exponential VWAP) pullback + ORB continuation on the current timeframe (recommended 1–3m in future).
For now, operates on the provided DataFrame with minimal dependencies.

Outputs a Signal-like object compatible with existing flow.
"""
from dataclasses import dataclass
from typing import Optional, Dict
import pandas as pd
import numpy as np


@dataclass
class ScalpSettings:
    rr: float = 2.0
    atr_len: int = 7
    ema_fast: int = 8
    ema_slow: int = 21
    vwap_window: int = 100
    min_bb_width_pct: float = 0.7  # 70th percentile of BB width
    vol_ratio_min: float = 1.3     # 1.3x 20-bar avg
    wick_ratio_min: float = 0.4    # Raised from 0.3 for stronger rejection wicks
    # New: body and wick alignment constraints for signal generation (so phantoms benefit)
    body_ratio_min: float = 0.30   # require a meaningful body in signal direction
    wick_delta_min: float = 0.10   # require dominant wick in trade direction by at least this delta
    vwap_dist_atr_max: float = 0.70 # Allow mid-band up to 0.7 ATR for detection
    orb_enabled: bool = False      # Optional ORB continuation filter
    # Enforce minimum 1R distance (as % of price) so fees don't erode R
    min_r_pct: float = 0.005
    # VWAP mode and session anchoring (signal-only behavior configured by live flow)
    vwap_mode: str = 'evwap'                  # 'evwap' | 'session_evwap'
    vwap_session_scheme: str = 'utc_day'      # 'utc_day' | 'sessions'
    vwap_session_warmup_bars: int = 30
    vwap_cap_warmup: float = 2.0
    vwap_session_windows: Optional[Dict[str, str]] = None  # e.g., {'asian': '00:00-08:00', ...}
    # VWAP pattern and options
    vwap_pattern: str = 'revert'              # 'bounce' | 'reject' | 'revert'
    vwap_bounce_band_atr_min: float = 0.10
    vwap_bounce_band_atr_max: float = 0.60
    vwap_bounce_lookback_bars: int = 3
    # VWAP-only detection mode
    vwap_only: bool = False                   # If true, use only VWAP proximity for acceptance (drop vol/bbw/wick gates)
    vwap_require_alignment: bool = True       # If true, require EMA alignment in vwap-only / patterns
    # Multi-anchor means OR (EVWAP OR EMA-band OR BB-mid) for signals
    means_enabled: bool = True
    ema_band_cap_atr: float = 1.5
    bb_mid_cap_atr: float = 1.5
    # K-of-N acceptance (phantoms): slope mandatory + K of [means, volume, wick_body, bbw]
    kofn_enabled: bool = True
    kofn_k: int = 2
    # ATR fallback (phantoms): accept when pullback size to last swing in [min,max] ATR
    atr_fallback_enabled: bool = True
    atr_fb_min: float = 0.5
    atr_fb_max: float = 1.5
    atr_fb_lookback_bars: int = 20
    # Near-miss phantom: accept when exactly one gate fails (informational learning)
    near_miss_enabled: bool = True


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
    """Exponential VWAP (not session-anchored): ema(pv) / ema(vol).

    Using ewm provides faster adaptation than rolling sums, so EVWAP "follows"
    price with less lag while retaining volume weighting.
    """
    tp = (df['high'] + df['low'] + df['close']) / 3
    vol = df['volume'].clip(lower=0.0)
    pv = tp * vol
    num = pv.ewm(span=window, adjust=False).mean()
    den = vol.ewm(span=window, adjust=False).mean().replace(0, np.nan)
    return num / den


def _session_start_for(ts: pd.Timestamp, scheme: str, windows: Optional[Dict[str, str]] = None) -> pd.Timestamp:
    """Return UTC session start timestamp for given ts based on scheme/windows."""
    tsu = ts.tz_convert('UTC') if ts.tzinfo else ts.tz_localize('UTC')
    day = tsu.normalize()
    if scheme == 'utc_day':
        return day
    # sessions scheme: choose from windows, default 00-08,08-16,16-24
    wins = windows or {'asian': '00:00-08:00', 'european': '08:00-16:00', 'us': '16:00-24:00'}
    hhmm = tsu.strftime('%H:%M')
    h = int(hhmm[:2]); m = int(hhmm[3:5])
    mins = h*60 + m
    def _parse(seg: str):
        a, b = seg.split('-'); ah, am = map(int, a.split(':')); bh, bm = map(int, b.split(':'))
        return ah*60+am, bh*60+bm
    start_min = 0
    for _, seg in wins.items():
        smin, emin = _parse(seg)
        if smin <= mins < emin:
            start_min = smin
            break
    # build start timestamp
    sh = start_min // 60; sm = start_min % 60
    return day + pd.Timedelta(hours=sh, minutes=sm)


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
    # VWAP/EVWAP (optionally session-anchored)
    if s.vwap_mode == 'session_evwap':
        try:
            last_ts = df.index[-1]
        except Exception:
            last_ts = None
        if last_ts is not None:
            start_ts = _session_start_for(last_ts, s.vwap_session_scheme, s.vwap_session_windows)
            df_sess = df[df.index >= start_ts]
            vwap_s = _vwap(df_sess, s.vwap_window)
            vwap = vwap_s
            sess_len = len(df_sess)
            warmup = sess_len < int(s.vwap_session_warmup_bars)
        else:
            vwap = _vwap(df, s.vwap_window)
            warmup = False
    else:
        vwap = _vwap(df, s.vwap_window)
        warmup = False

    # BB width percentile proxy (use rolling std percentage)
    std20 = close.rolling(20).std()
    bbw = (std20 / close).fillna(0)
    if len(bbw) < 30:
        return None
    last_bbw = float(bbw.iloc[-1])
    bbw_pct = (bbw <= last_bbw).mean()  # percentile of current width
    # BB midline (20 SMA)
    ma20 = close.rolling(20).mean()

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
    # Body ratio and direction
    body_ratio = abs(c - o) / rng if rng > 0 else 0.0
    body_up = (c > o)
    body_dn = (c < o)

    # Trend check (continuation)
    ema_aligned_up = (c > ema_f.iloc[-1] > ema_s.iloc[-1])
    ema_aligned_dn = (c < ema_f.iloc[-1] < ema_s.iloc[-1])

    # Distances to means (normalized by ATR)
    cur_vwap = float(vwap.iloc[-1]) if not np.isnan(vwap.iloc[-1]) else c
    cur_atr = float(atr.iloc[-1]) if atr.iloc[-1] > 0 else max(1e-9, rng)
    dist_vwap_atr = abs(c - cur_vwap) / cur_atr
    dist_ema_band = abs(c - float(ema_s.iloc[-1])) / cur_atr if len(ema_s) else 1e9
    dist_bb_mid = abs(c - float(ma20.iloc[-1])) / cur_atr if len(ma20) else 1e9
    # Effective VWAP cap (warmup may allow looser cap)
    eff_vwap_cap = max(float(s.vwap_dist_atr_max), float(s.vwap_cap_warmup)) if warmup else float(s.vwap_dist_atr_max)

    # Optional ORB filter: require price beyond recent 20-bar range for continuation bias
    orb_ok = True
    if s.orb_enabled and len(df) >= 40:
        recent_high = float(high.iloc[-20:].max())
        recent_low = float(low.iloc[-20:].min())
        if ema_aligned_up and c <= recent_high:
            orb_ok = False
        if ema_aligned_dn and c >= recent_low:
            orb_ok = False

    # Multi-anchor mean proximity and gate booleans
    means_evwap_ok = dist_vwap_atr <= eff_vwap_cap
    means_ema_ok = dist_ema_band <= float(s.ema_band_cap_atr)
    means_bb_ok = dist_bb_mid <= float(s.bb_mid_cap_atr)
    # Default gates
    vol_ok = (float(vol_ratio) >= float(s.vol_ratio_min))
    bbw_ok = (float(bbw_pct) >= float(s.min_bb_width_pct))
    wick_body_long = (lower_w >= max(float(s.wick_ratio_min), float(upper_w) + float(s.wick_delta_min))) and body_up and (body_ratio >= float(s.body_ratio_min))
    wick_body_short = (upper_w >= max(float(s.wick_ratio_min), float(lower_w) + float(s.wick_delta_min))) and body_dn and (body_ratio >= float(s.body_ratio_min))

    # Pattern selection
    pattern = (s.vwap_pattern or 'revert').lower()
    means_ok = False
    if bool(s.vwap_only):
        means_ok = means_evwap_ok
        vol_ok = True; bbw_ok = True; wick_body_long = True; wick_body_short = True
        if not bool(s.vwap_require_alignment):
            ema_aligned_up = True; ema_aligned_dn = True
    elif pattern == 'bounce':
        vwap_now = float(vwap.iloc[-1]) if not np.isnan(vwap.iloc[-1]) else c
        band_min = float(s.vwap_bounce_band_atr_min)
        band_max = float(s.vwap_bounce_band_atr_max)
        look = int(max(1, s.vwap_bounce_lookback_bars))
        recent_lows = low.iloc[-look:]
        recent_highs = high.iloc[-look:]
        touch_long = abs(float(recent_lows.min()) - vwap_now) / max(1e-9, cur_atr)
        touch_short = abs(float(recent_highs.max()) - vwap_now) / max(1e-9, cur_atr)
        open_now = float(o)
        bounce_long = (
            (ema_aligned_up or not s.vwap_require_alignment) and
            (c >= vwap_now) and (open_now >= vwap_now) and
            (band_min <= touch_long <= band_max)
        )
        bounce_short = (
            (ema_aligned_dn or not s.vwap_require_alignment) and
            (c <= vwap_now) and (open_now <= vwap_now) and
            (band_min <= touch_short <= band_max)
        )
        means_ok = bounce_long or bounce_short
    elif pattern == 'reject':
        vwap_now = float(vwap.iloc[-1]) if not np.isnan(vwap.iloc[-1]) else c
        reject_long = (l < vwap_now and c > vwap_now) and (ema_aligned_up or not s.vwap_require_alignment)
        reject_short = (h > vwap_now and c < vwap_now) and (ema_aligned_dn or not s.vwap_require_alignment)
        means_ok = reject_long or reject_short
    else:
        means_ok = means_evwap_ok or (bool(s.means_enabled) and (means_ema_ok or means_bb_ok))

    # Long scalp candidate (means OR)
    if (
        ema_aligned_up
        and bbw_ok
        and vol_ok
        and wick_body_long
        and means_ok
        and orb_ok
    ):
        # Stop below VWAP/EMA band with high-volatility widening
        buf_mult = 0.8
        if bbw_pct >= 0.85:
            buf_mult = 1.2
        elif bbw_pct >= 0.70:
            buf_mult = 1.0
        sl = min(cur_vwap, ema_s.iloc[-1]) - buf_mult * cur_atr
        entry = c
        # Enforce minimum stop distance to avoid micro-stops on 3m and ensure 1R ≥ min_r_pct
        min_dist = max(entry * s.min_r_pct, 0.6 * cur_atr, entry * 0.002)
        if entry - sl < min_dist:
            sl = entry - min_dist
        if entry <= sl:
            return None
        
        # Calculate TP and validate
        tp = entry + s.rr * (entry - sl)
        # Ensure SL is below entry (sanity check)
        if sl >= entry:
            return None
        # Ensure TP is above entry
        if tp <= entry:
            return None
            
        return ScalpSignal(
            side='long', entry=entry, sl=sl, tp=tp,
            reason=f"SCALP: accept={'vwap_bounce' if pattern=='bounce' else ('vwap_reject' if pattern=='reject' else 'means_or')}",
            meta={
                'vwap': cur_vwap,
                'atr': cur_atr,
                'bbw_pct': bbw_pct,
                'vol_ratio': vol_ratio,
                'dist_vwap_atr': dist_vwap_atr,
                'means': {
                    'evwap_ok': bool(means_evwap_ok), 'ema_ok': bool(means_ema_ok), 'bb_ok': bool(means_bb_ok),
                    'dist_evwap': float(dist_vwap_atr), 'dist_ema': float(dist_ema_band), 'dist_bb': float(dist_bb_mid),
                    'cap': float(eff_vwap_cap)
                },
                'acceptance_path': 'vwap_bounce' if pattern=='bounce' else ('vwap_reject' if pattern=='reject' else 'means_or')
            }
        )

    # Short scalp candidate
    if (
        ema_aligned_dn
        and bbw_ok
        and vol_ok
        and wick_body_short
        and means_ok
        and orb_ok
    ):
        buf_mult = 0.8
        if bbw_pct >= 0.85:
            buf_mult = 1.2
        elif bbw_pct >= 0.70:
            buf_mult = 1.0
        sl = max(cur_vwap, ema_s.iloc[-1]) + buf_mult * cur_atr
        entry = c
        # Enforce minimum stop distance and ensure 1R ≥ min_r_pct
        min_dist = max(entry * s.min_r_pct, 0.6 * cur_atr, entry * 0.002)
        if sl - entry < min_dist:
            sl = entry + min_dist
        if sl <= entry:
            return None
        
        # Calculate TP and ensure it's positive and below entry
        tp = entry - s.rr * (sl - entry)
        # Validate TP is positive and below entry
        if tp <= 0:
            # TP went negative - use safer calculation (minimum 20% profit from entry)
            tp = max(entry * 0.2, entry - (sl - entry) * 1.0)  # At least 1:1 R:R or 20% of entry
            if tp <= 0:
                return None  # Skip this signal if we can't get valid TP
        if tp >= entry:
            # TP is above entry for short - invalid
            return None
            
        return ScalpSignal(
            side='short', entry=entry, sl=sl, tp=tp,
            reason=f"SCALP: accept={'vwap_bounce' if pattern=='bounce' else ('vwap_reject' if pattern=='reject' else 'means_or')}",
            meta={
                'vwap': cur_vwap,
                'atr': cur_atr,
                'bbw_pct': bbw_pct,
                'vol_ratio': vol_ratio,
                'dist_vwap_atr': dist_vwap_atr,
                'means': {
                    'evwap_ok': bool(means_evwap_ok), 'ema_ok': bool(means_ema_ok), 'bb_ok': bool(means_bb_ok),
                    'dist_evwap': float(dist_vwap_atr), 'dist_ema': float(dist_ema_band), 'dist_bb': float(dist_bb_mid),
                    'cap': float(eff_vwap_cap)
                },
                'acceptance_path': 'vwap_bounce' if pattern=='bounce' else ('vwap_reject' if pattern=='reject' else 'means_or')
            }
        )
    # K-of-N (phantom) path — slope mandatory
    if (not bool(s.vwap_only)) and bool(s.kofn_enabled):
        if ema_aligned_up:
            gates = [bool(means_ok), bool(vol_ok), bool(wick_body_long), bool(bbw_ok)]
            if sum(1 for g in gates if g) >= int(s.kofn_k):
                buf_mult = 0.8
                if bbw_pct >= 0.85:
                    buf_mult = 1.2
                elif bbw_pct >= 0.70:
                    buf_mult = 1.0
                sl = min(cur_vwap, ema_s.iloc[-1]) - buf_mult * cur_atr
                entry = c
                min_dist = max(entry * s.min_r_pct, 0.6 * cur_atr, entry * 0.002)
                if entry - sl < min_dist:
                    sl = entry - min_dist
                if entry > sl:
                    tp = entry + s.rr * (entry - sl)
                    return ScalpSignal(
                        side='long', entry=entry, sl=sl, tp=tp,
                        reason=f"SCALP: accept=kofn",
                        meta={'acceptance_path': 'kofn'}
                    )
        if ema_aligned_dn:
            gates = [bool(means_ok), bool(vol_ok), bool(wick_body_short), bool(bbw_ok)]
            if sum(1 for g in gates if g) >= int(s.kofn_k):
                buf_mult = 0.8
                if bbw_pct >= 0.85:
                    buf_mult = 1.2
                elif bbw_pct >= 0.70:
                    buf_mult = 1.0
                sl = max(cur_vwap, ema_s.iloc[-1]) + buf_mult * cur_atr
                entry = c
                min_dist = max(entry * s.min_r_pct, 0.6 * cur_atr, entry * 0.002)
                if sl - entry < min_dist:
                    sl = entry + min_dist
                if sl > entry:
                    tp = entry - s.rr * (sl - entry)
                    return ScalpSignal(
                        side='short', entry=entry, sl=sl, tp=tp,
                        reason=f"SCALP: accept=kofn",
                        meta={'acceptance_path': 'kofn'}
                    )
    # ATR fallback (phantom)
    if (not bool(s.vwap_only)) and bool(s.atr_fallback_enabled):
        lb = int(max(2, s.atr_fb_lookback_bars))
        if ema_aligned_up and wick_body_long:
            try:
                swing_low = float(low.iloc[-lb:].min())
                size_atr = (c - swing_low) / cur_atr if cur_atr > 0 else 0.0
            except Exception:
                size_atr = 0.0
            if float(s.atr_fb_min) <= size_atr <= float(s.atr_fb_max):
                buf_mult = 0.8
                sl = min(cur_vwap, ema_s.iloc[-1]) - buf_mult * cur_atr
                entry = c
                min_dist = max(entry * s.min_r_pct, 0.6 * cur_atr, entry * 0.002)
                if entry - sl < min_dist:
                    sl = entry - min_dist
                if entry > sl:
                    tp = entry + s.rr * (entry - sl)
                    return ScalpSignal(
                        side='long', entry=entry, sl=sl, tp=tp,
                        reason=f"SCALP: accept=atr_fallback size={size_atr:.2f}ATR",
                        meta={'acceptance_path': 'atr_fallback', 'atr_fallback_size': float(size_atr)}
                    )
        if ema_aligned_dn and wick_body_short:
            try:
                swing_high = float(high.iloc[-lb:].max())
                size_atr = (swing_high - c) / cur_atr if cur_atr > 0 else 0.0
            except Exception:
                size_atr = 0.0
            if float(s.atr_fb_min) <= size_atr <= float(s.atr_fb_max):
                buf_mult = 0.8
                sl = max(cur_vwap, ema_s.iloc[-1]) + buf_mult * cur_atr
                entry = c
                min_dist = max(entry * s.min_r_pct, 0.6 * cur_atr, entry * 0.002)
                if sl - entry < min_dist:
                    sl = entry + min_dist
                if sl > entry:
                    tp = entry - s.rr * (sl - entry)
                    return ScalpSignal(
                        side='short', entry=entry, sl=sl, tp=tp,
                        reason=f"SCALP: accept=atr_fallback size={size_atr:.2f}ATR",
                        meta={'acceptance_path': 'atr_fallback', 'atr_fallback_size': float(size_atr)}
                    )
    # Near-miss phantom acceptance
    if (not bool(s.vwap_only)) and bool(s.near_miss_enabled):
        if ema_aligned_up:
            fails = [not means_ok, not vol_ok, not wick_body_long, not bbw_ok]
            if sum(1 for f in fails if f) == 1:
                buf_mult = 0.8
                sl = min(cur_vwap, ema_s.iloc[-1]) - buf_mult * cur_atr
                entry = c
                min_dist = max(entry * s.min_r_pct, 0.6 * cur_atr, entry * 0.002)
                if entry - sl < min_dist:
                    sl = entry - min_dist
                if entry > sl:
                    tp = entry + s.rr * (entry - sl)
                    return ScalpSignal(
                        side='long', entry=entry, sl=sl, tp=tp,
                        reason=f"SCALP: accept=near_miss",
                        meta={'acceptance_path': 'near_miss'}
                    )
        if ema_aligned_dn:
            fails = [not means_ok, not vol_ok, not wick_body_short, not bbw_ok]
            if sum(1 for f in fails if f) == 1:
                buf_mult = 0.8
                sl = max(cur_vwap, ema_s.iloc[-1]) + buf_mult * cur_atr
                entry = c
                min_dist = max(entry * s.min_r_pct, 0.6 * cur_atr, entry * 0.002)
                if sl - entry < min_dist:
                    sl = entry + min_dist
                if sl > entry:
                    tp = entry - s.rr * (sl - entry)
                    return ScalpSignal(
                        side='short', entry=entry, sl=sl, tp=tp,
                        reason=f"SCALP: accept=near_miss",
                        meta={'acceptance_path': 'near_miss'}
                    )

    return None
