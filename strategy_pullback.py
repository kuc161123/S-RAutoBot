from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# 3m microframe provider (injected from live_bot)
_get_df3m = None  # type: Optional[callable]

def set_trend_microframe_provider(fn):
    """Register a callback: fn(symbol) -> pd.DataFrame for 3m microframe."""
    global _get_df3m
    _get_df3m = fn

# --- Event notifier integration (for Telegram, etc.) ---
_event_notifier = None  # type: Optional[callable]
_redis_client = None    # type: Optional[object]
_entry_executor = None  # type: Optional[callable]
_phantom_recorder = None  # type: Optional[callable]
_invalidation_hook = None  # type: Optional[callable]

def set_trend_event_notifier(fn):
    """Register a notifier callback taking (symbol:str, text:str)."""
    global _event_notifier
    _event_notifier = fn

def _notify(symbol: str, text: str):
    try:
        if _event_notifier:
            _event_notifier(symbol, text)
    except Exception:
        pass

# --- Simple Redis-backed state persistence (optional) ---
def set_trend_state_store(redis_client):
    """Register a Redis client for persisting per-symbol pullback state."""
    global _redis_client
    _redis_client = redis_client

def set_trend_entry_executor(fn):
    """Register entry executor: fn(symbol, side, entry, sl, tp, meta_dict)"""
    global _entry_executor
    _entry_executor = fn

def set_trend_phantom_recorder(fn):
    """Register phantom recorder: fn(symbol, side, entry, sl, tp, meta_dict)"""
    global _phantom_recorder
    _phantom_recorder = fn

def set_trend_invalidation_hook(fn):
    """Register invalidation hook: fn(symbol, info: dict) called on breakout invalidation/reset."""
    global _invalidation_hook
    _invalidation_hook = fn

def revert_to_neutral(symbol: str):
    """Reset the breakout state to NEUTRAL after a non-executed outcome (gate block, ML reject, etc.)."""
    try:
        st = breakout_states.setdefault(symbol, BreakoutState())
        st.state = "NEUTRAL"
        st.breakout_level = 0.0
        st.pullback_extreme = 0.0
        st.pullback_time = None
        st.breakout_time = None
        st.confirmation_count = 0
        st.micro_state = ""
        st.last_counter_pivot = 0.0
        st.last_counter_time = None
        st.bos_cross_notified = False
        st.waiting_reason = ""
        st.divergence_ok = False
        st.divergence_type = "NONE"
        st.divergence_score = 0.0
        st.divergence_time = None
        st.divergence_timeout_notified = False
        st.last_executed = False
        _persist_state(symbol, st)
    except Exception:
        pass

def _persist_state(symbol: str, state: 'BreakoutState'):
    try:
        if not _redis_client:
            return
        import json, time
        payload = {
            'state': state.state,
            'breakout_level': float(state.breakout_level or 0.0),
            'pullback_extreme': float(state.pullback_extreme or 0.0),
            'pullback_time': state.pullback_time.isoformat() if state.pullback_time else None,
            'breakout_time': state.breakout_time.isoformat() if state.breakout_time else None,
            'confirmation_count': int(state.confirmation_count or 0),
            'last_resistance': float(state.last_resistance or 0.0),
            'last_support': float(state.last_support or 0.0),
            'previous_pivot_low': float(state.previous_pivot_low or 0.0),
            'previous_pivot_high': float(state.previous_pivot_high or 0.0),
            'last_update_ts': int(time.time())
        }
        _redis_client.set(f'trend:state:{symbol}', json.dumps(payload))
    except Exception:
        pass

def hydrate_trend_states(frames: Dict[str, pd.DataFrame], timeframe_min: int = 15, max_age_bars: int = 48) -> int:
    """Hydrate in-memory breakout_states from Redis. Returns number restored."""
    try:
        if not _redis_client:
            return 0
        import time, json
        restored = 0
        try:
            cur = 0
            keys = []
            while True:
                cur, batch = _redis_client.scan(cur, match='trend:state:*', count=200)
                keys.extend(batch or [])
                if cur == 0:
                    break
        except Exception:
            try:
                keys = _redis_client.keys('trend:state:*') or []
            except Exception:
                keys = []
        now = int(time.time()); max_age = int(max(1, max_age_bars) * timeframe_min * 60)
        for k in keys:
            try:
                raw = _redis_client.get(k)
                if not raw:
                    continue
                rec = json.loads(raw)
                ts = int(rec.get('last_update_ts', 0) or 0)
                if ts and (now - ts) > max_age:
                    continue
                sym = k.split(':', 2)[-1]
                if sym not in frames:
                    continue
                st = breakout_states.setdefault(sym, BreakoutState())
                st.state = str(rec.get('state', st.state))
                st.breakout_level = float(rec.get('breakout_level', 0.0))
                st.pullback_extreme = float(rec.get('pullback_extreme', 0.0))
                try:
                    pb = rec.get('pullback_time')
                    st.pullback_time = pd.to_datetime(pb) if pb else None
                except Exception:
                    st.pullback_time = None
                try:
                    bo = rec.get('breakout_time')
                    st.breakout_time = pd.to_datetime(bo) if bo else None
                except Exception:
                    st.breakout_time = None
                st.confirmation_count = int(rec.get('confirmation_count', 0))
                st.last_resistance = float(rec.get('last_resistance', 0.0))
                st.last_support = float(rec.get('last_support', 0.0))
                st.previous_pivot_low = float(rec.get('previous_pivot_low', 0.0))
                st.previous_pivot_high = float(rec.get('previous_pivot_high', 0.0))
                restored += 1
            except Exception:
                continue
        return restored
    except Exception:
        return 0

@dataclass
class Settings:
    left:int=2
    right:int=2
    atr_len:int=14
    sl_buf_atr:float=0.5
    rr:float=2.5
    use_ema:bool=False
    ema_len:int=200
    use_vol:bool=False
    vol_len:int=20
    vol_mult:float=1.2
    both_hit_rule:str="SL_FIRST"
    confirmation_candles:int=2  # Number of confirmation candles required
    use_mtf_sr:bool=True  # Enable multi-timeframe S/R
    mtf_weight:float=2.0  # Prefer major levels 2x over minor
    mtf_min_strength:float=3.0  # Minimum strength for major levels
    min_candles_between_signals:int=5 # Minimum number of candles between signals for the same symbol
    # Extra breathing room when a pivot-based stop is selected
    # Expressed as fraction of entry price (e.g., 0.01 = 1%)
    extra_pivot_breath_pct: float = 0.01
    # Maximum bars to wait after HL/LH for confirmations before forgetting the setup
    confirmation_timeout_bars: int = 6
    # Use 3m microframe for pullback and confirmation phases
    use_3m_pullback: bool = True
    use_3m_confirm: bool = True
    # Microstructure/BOS controls
    retest_enabled: bool = True
    retest_distance_mode: str = "atr"  # atr|percent
    retest_max_dist_atr: float = 0.50
    retest_max_dist_pct: float = 0.40
    require_protective_hl_for_long: bool = True
    require_protective_lh_for_short: bool = True
    bos_body_min_ratio: float = 0.30
    bos_confirm_closes: int = 1  # extra 3m closes beyond pivot
    breakout_to_pullback_bars_3m: int = 10
    pullback_to_bos_bars_3m: int = 10
    breakout_buffer_atr: float = 0.05  # invalidation buffer beyond level (15m ATR)
    # 15m breakout gating
    require_main_confirm_for_breakout: bool = True
    current_bar_confirmed: bool = False
    # Divergence confirmation (3m) before BOS
    div_enabled: bool = False
    div_mode: str = "optional"  # off|optional|strict
    div_require: str = "any"     # any|all
    div_use_rsi: bool = True
    div_use_tsi: bool = True
    div_rsi_len: int = 14
    div_tsi_long: int = 25
    div_tsi_short: int = 13
    div_window_bars_3m: int = 6
    div_min_strength_rsi: float = 2.0
    div_min_strength_tsi: float = 0.3
    div_notify: bool = True
    # BOS armed hold window (minutes) when BOS crosses first and waiting for protective HL/LH
    bos_armed_hold_minutes: int = 300  # 5 hours
    # SL policy
    sl_mode: str = "breakout"  # breakout | hybrid
    breakout_sl_buffer_atr: float = 0.30
    min_r_pct: float = 0.005
    # Exec-only S/R gate (phantoms free): require major HTF level + confluence + clearance
    sr_exec_enabled: bool = True
    sr_min_strength: float = 2.5
    sr_confluence_tolerance_pct: float = 0.0025  # 0.25%
    sr_min_break_clear_atr: float = 0.10

@dataclass
class Signal:
    side:str             # "long" or "short"
    entry:float
    sl:float
    tp:float
    reason:str
    meta:dict

@dataclass
class BreakoutState:
    """Track the state of a breakout for each symbol"""
    state:str = "NEUTRAL"  # NEUTRAL, RESISTANCE_BROKEN, SUPPORT_BROKEN, HL_FORMED, LH_FORMED, SIGNAL_SENT
    breakout_level:float = 0.0  # The S/R level that was broken
    breakout_time:Optional[datetime] = None
    pullback_extreme:float = 0.0  # Lowest point in pullback for long, highest for short
    pullback_time:Optional[datetime] = None
    confirmation_count:int = 0
    last_signal_time:Optional[datetime] = None
    last_signal_candle_time: Optional[datetime] = None
    last_resistance:float = 0.0  # Track the resistance level
    last_support:float = 0.0  # Track the support level
    previous_pivot_low:float = 0.0  # Track previous pivot low for structure-based stops
    previous_pivot_high:float = 0.0  # Track previous pivot high for structure-based stops
    # Microstructure fields
    micro_state: str = ""  # PULLBACK_3M, BOS_PENDING
    last_counter_pivot: float = 0.0  # last 3m LH (long) or HL (short)
    last_counter_time: Optional[datetime] = None
    retest_ok: bool = False
    bos_pending_confirms: int = 0
    micro_start_time: Optional[datetime] = None
    last_counter_pivot_notified: float = 0.0
    # Divergence tracking
    divergence_ok: bool = False
    divergence_type: str = "NONE"  # HBULL | HBEAR | NONE
    divergence_score: float = 0.0
    divergence_time: Optional[datetime] = None
    divergence_timeout_notified: bool = False
    div_rsi_delta: float = 0.0
    div_tsi_delta: float = 0.0
    # One-time micro confirmation note (HH/HL or LH/LL) just after breakout
    micro_note_sent: bool = False
    # Exec gate diagnostics
    last_sr_gate_ok: Optional[bool] = None
    last_sr_meta: dict = None
    last_htf_gate_ok: Optional[bool] = None
    last_htf_meta: dict = None
    # Execution marker (set true when an exchange order is placed)
    last_executed: bool = False

def update_sr_gate(symbol: str, ok: bool, meta: dict):
    try:
        st = breakout_states.setdefault(symbol, BreakoutState())
        st.last_sr_gate_ok = bool(ok)
        st.last_sr_meta = dict(meta or {})
    except Exception:
        pass

def update_htf_gate(symbol: str, ok: bool, meta: dict):
    try:
        st = breakout_states.setdefault(symbol, BreakoutState())
        st.last_htf_gate_ok = bool(ok)
        st.last_htf_meta = dict(meta or {})
    except Exception:
        pass

def mark_executed(symbol: str):
    try:
        st = breakout_states.setdefault(symbol, BreakoutState())
        st.last_executed = True
    except Exception:
        pass
    # BOS cross monitoring (when pivot broken but protective HL/LH not yet confirmed)
    bos_cross_notified: bool = False
    bos_cross_time: Optional[datetime] = None
    waiting_reason: str = ""  # e.g., WAIT_HL, WAIT_LH, ""

# Global state tracking for each symbol
breakout_states: Dict[str, BreakoutState] = {}

def _pivot_high(h:pd.Series, L:int, R:int):
    """Find pivot highs in price series"""
    win = h.rolling(L+R+1, center=True).max()
    mask = (h == win) & h.notna()
    return np.where(mask, h, np.nan)

def _pivot_low(l:pd.Series, L:int, R:int):
    """Find pivot lows in price series"""
    win = l.rolling(L+R+1, center=True).min()
    mask = (l == win) & l.notna()
    return np.where(mask, l, np.nan)

def _rsi(close: pd.Series, length: int = 14) -> pd.Series:
    try:
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        # Wilder's smoothing via EMA with alpha=1/length
        roll_up = gain.ewm(alpha=1.0/float(max(1, length)), adjust=False).mean()
        roll_down = loss.ewm(alpha=1.0/float(max(1, length)), adjust=False).mean()
        rs = roll_up / roll_down.replace(0.0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi.fillna(50.0)
    except Exception:
        return pd.Series(np.full(len(close), 50.0), index=close.index)

def _tsi(close: pd.Series, long_len: int = 25, short_len: int = 13) -> pd.Series:
    try:
        m = close.diff()
        m_abs = m.abs()
        # Two-stage EMA smoothing (short then long)
        ema_m_short = m.ewm(span=max(1, short_len), adjust=False).mean()
        ema_m_long = ema_m_short.ewm(span=max(1, long_len), adjust=False).mean()
        ema_a_short = m_abs.ewm(span=max(1, short_len), adjust=False).mean()
        ema_a_long = ema_a_short.ewm(span=max(1, long_len), adjust=False).mean()
        tsi = 100.0 * (ema_m_long / ema_a_long.replace(0.0, np.nan))
        return tsi.fillna(0.0)
    except Exception:
        return pd.Series(np.zeros(len(close)), index=close.index)

def _last_two_pivots(series: pd.Series, kind: str = 'low', L: int = 2, R: int = 2):
    try:
        piv = _pivot_low(series, L, R) if kind == 'low' else _pivot_high(series, L, R)
        s = pd.Series(piv, index=series.index)
        nz = s.dropna()
        if len(nz) < 2:
            return None
        # Return tuples: (idx1, val1), (idx2, val2)
        idx2 = nz.index[-1]; v2 = float(nz.iloc[-1])
        idx1 = nz.index[-2]; v1 = float(nz.iloc[-2])
        return (idx1, v1), (idx2, v2)
    except Exception:
        return None

def _hidden_divergence(df3: pd.DataFrame, path: str, s: Settings):
    """Evaluate hidden divergence on 3m pivots.
    path: 'long' (HL) or 'short' (LH)
    Returns: (ok:bool, div_type:str, score:float, details:str, dr:float, dt:float)
    """
    try:
        close = df3['close']
        lows = df3['low']; highs = df3['high']
        pivs = _last_two_pivots(lows if path=='long' else highs, 'low' if path=='long' else 'high', 2, 2)
        if not pivs:
            return False, 'NONE', 0.0, '', 0.0, 0.0
        (i1, p1), (i2, p2) = pivs
        # Price HL/LH check (should already be true from protective pivot detection)
        price_ok = (p2 > p1) if path=='long' else (p2 < p1)
        if not price_ok:
            return False, 'NONE', 0.0, '', 0.0, 0.0
        # Oscillators
        rsi = _rsi(close, s.div_rsi_len) if s.div_use_rsi else None
        tsi = _tsi(close, s.div_tsi_long, s.div_tsi_short) if s.div_use_tsi else None
        rsi_ok = tsi_ok = False
        dr = dt = 0.0
        if rsi is not None and i1 in rsi.index and i2 in rsi.index:
            r1 = float(rsi.loc[i1]); r2 = float(rsi.loc[i2]); dr = r2 - r1
            if path=='long':
                rsi_ok = (r2 < r1 - s.div_min_strength_rsi)
            else:
                rsi_ok = (r2 > r1 + s.div_min_strength_rsi)
        if tsi is not None and i1 in tsi.index and i2 in tsi.index:
            t1 = float(tsi.loc[i1]); t2 = float(tsi.loc[i2]); dt = t2 - t1
            if path=='long':
                tsi_ok = (t2 < t1 - s.div_min_strength_tsi)
            else:
                tsi_ok = (t2 > t1 + s.div_min_strength_tsi)
        any_ok = (rsi_ok or tsi_ok)
        all_ok = ( (not s.div_use_rsi or rsi_ok) and (not s.div_use_tsi or tsi_ok) )
        ok = any_ok if s.div_require == 'any' else all_ok
        if not ok:
            return False, 'NONE', 0.0, '', float(dr if 'dr' in locals() else 0.0), float(dt if 'dt' in locals() else 0.0)
        div_type = 'HBULL' if path=='long' else 'HBEAR'
        score = max(abs(dr), abs(dt))
        details = []
        if s.div_use_rsi:
            details.append(f"RSIΔ={dr:+.2f}")
        if s.div_use_tsi:
            details.append(f"TSIΔ={dt:+.2f}")
        return True, div_type, float(score), (' '.join(details)), float(dr), float(dt)
    except Exception:
        return False, 'NONE', 0.0, '', 0.0, 0.0

def _micro_confirms(df3: pd.DataFrame, path: str) -> bool:
    """Check if 3m shows HH/HL (long) or LH/LL (short) using last two pivots."""
    try:
        if path == 'long':
            hp = _last_two_pivots(df3['high'], 'high', 2, 2)
            lp = _last_two_pivots(df3['low'], 'low', 2, 2)
            if (not hp) or (not lp):
                return False
            (_, h1), (_, h2) = hp; (_, l1), (_, l2) = lp
            return (h2 > h1) and (l2 > l1)
        else:
            hp = _last_two_pivots(df3['high'], 'high', 2, 2)
            lp = _last_two_pivots(df3['low'], 'low', 2, 2)
            if (not hp) or (not lp):
                return False
            (_, h1), (_, h2) = hp; (_, l1), (_, l2) = lp
            return (h2 < h1) and (l2 < l1)
    except Exception:
        return False

def _atr(df:pd.DataFrame, n:int):
    """Calculate Average True Range"""
    prev_close = df["close"].shift()
    tr = np.maximum(df["high"]-df["low"],
         np.maximum(abs(df["high"]-prev_close), abs(df["low"]-prev_close)))
    return pd.Series(tr, index=df.index).rolling(n).mean().values

def _ema(s:pd.Series, n:int):
    """Calculate Exponential Moving Average"""
    return s.ewm(span=n, adjust=False).mean().values

def find_previous_pivot_low(df:pd.DataFrame, current_idx:int, left:int=2, right:int=2, lookback:int=50) -> float:
    """Find the most recent pivot low before current index"""
    start_idx = max(0, current_idx - lookback)
    
    # Get pivot lows
    pivot_lows = _pivot_low(df["low"], left, right)
    
    # Find most recent pivot before current index
    for i in range(current_idx - right - 1, start_idx, -1):
        if not np.isnan(pivot_lows[i]):
            return pivot_lows[i]
    
    # If no pivot found, use lowest low in lookback period
    return df["low"].iloc[start_idx:current_idx].min()

def find_previous_pivot_high(df:pd.DataFrame, current_idx:int, left:int=2, right:int=2, lookback:int=50) -> float:
    """Find the most recent pivot high before current index"""
    start_idx = max(0, current_idx - lookback)
    
    # Get pivot highs
    pivot_highs = _pivot_high(df["high"], left, right)
    
    # Find most recent pivot before current index
    for i in range(current_idx - right - 1, start_idx, -1):
        if not np.isnan(pivot_highs[i]):
            return pivot_highs[i]
    
    # If no pivot found, use highest high in lookback period
    return df["high"].iloc[start_idx:current_idx].max()

def detect_higher_low(df:pd.DataFrame, above_level:float, lookback:int=10) -> bool:
    """
    Detect if price has made a higher low above a certain level.
    Used after resistance break to confirm pullback completion.
    """
    recent_lows = df["low"].iloc[-lookback:]
    
    # Find the lowest point in recent candles
    min_idx = recent_lows.idxmin()
    min_low = recent_lows.loc[min_idx]
    
    # Check if we've made a higher low (price bounced)
    latest_low = df["low"].iloc[-1]
    prev_low = df["low"].iloc[-2]
    
    # Higher low conditions:
    # 1. The minimum low is above the breakout level
    # 2. Current price is moving up from the minimum
    # 3. We have at least 2 candles since the minimum
    if min_low > above_level:
        min_pos = recent_lows.index.get_loc(min_idx)
        current_pos = len(recent_lows) - 1
        if current_pos - min_pos >= 2:  # At least 2 candles since the low
            if df["close"].iloc[-1] > min_low and df["close"].iloc[-2] > min_low:
                return True
    
    return False

def detect_lower_high(df:pd.DataFrame, below_level:float, lookback:int=10) -> bool:
    """
    Detect if price has made a lower high below a certain level.
    Used after support break to confirm pullback completion.
    """
    recent_highs = df["high"].iloc[-lookback:]
    
    # Find the highest point in recent candles
    max_idx = recent_highs.idxmax()
    max_high = recent_highs.loc[max_idx]
    
    # Check if we've made a lower high (price rejected)
    latest_high = df["high"].iloc[-1]
    prev_high = df["high"].iloc[-2]
    
    # Lower high conditions:
    # 1. The maximum high is below the breakout level
    # 2. Current price is moving down from the maximum
    # 3. We have at least 2 candles since the maximum
    if max_high < below_level:
        max_pos = recent_highs.index.get_loc(max_idx)
        current_pos = len(recent_highs) - 1
        if current_pos - max_pos >= 2:  # At least 2 candles since the high
            if df["close"].iloc[-1] < max_high and df["close"].iloc[-2] < max_high:
                return True
    
    return False

def _last_pivot(series: pd.Series, kind: str = 'high', left: int = 2, right: int = 2) -> Optional[float]:
    try:
        if kind == 'high':
            piv = _pivot_high(series, left, right)
        else:
            piv = _pivot_low(series, left, right)
        s = pd.Series(piv, index=series.index).dropna()
        if len(s) == 0:
            return None
        return float(s.iloc[-1])
    except Exception:
        return None

def _body_ratio(df: pd.DataFrame) -> float:
    try:
        o = float(df['open'].iloc[-1]); c = float(df['close'].iloc[-1]);
        h = float(df['high'].iloc[-1]); l = float(df['low'].iloc[-1])
        rng = max(1e-9, h - l)
        body = abs(c - o)
        return body / rng
    except Exception:
        return 0.0

def _atr_val(df: pd.DataFrame, n: int) -> float:
    try:
        return float(_atr(df, n)[-1])
    except Exception:
        return 0.0

def count_confirmation_candles(df:pd.DataFrame, direction:str, count_needed:int=2) -> int:
    """
    Count consecutive confirmation candles in the specified direction.
    For long: bullish candles (close > open)
    For short: bearish candles (close < open)
    """
    confirmation_count = 0
    
    # Check the last 'count_needed' candles
    for i in range(-count_needed, 0):
        if i >= -len(df):
            candle_open = df["open"].iloc[i]
            candle_close = df["close"].iloc[i]
            
            if direction == "long" and candle_close > candle_open:
                confirmation_count += 1
            elif direction == "short" and candle_close < candle_open:
                confirmation_count += 1
            else:
                # Reset count if we get opposite candle
                confirmation_count = 0
                break
    
    return confirmation_count

def detect_signal_pullback(df:pd.DataFrame, s:Settings, symbol:str="") -> Optional[Signal]:
    """
    Enhanced signal detection with pullback strategy:
    1. Wait for S/R breakout
    2. Wait for pullback (HL for long, LH for short)
    3. Wait for confirmation candles
    4. Then generate signal
    """
    # Import MTF module if enabled
    if s.use_mtf_sr:
        try:
            from multi_timeframe_sr import mtf_sr, should_use_mtf_level
        except ImportError:
            logger.warning("MTF S/R module not available, using standard pivots")
            s.use_mtf_sr = False
    
    # Initialize state for new symbols
    if symbol not in breakout_states:
        breakout_states[symbol] = BreakoutState()
    
    state = breakout_states[symbol]

    # Utility: compact preconditions logger
    def _log_preconditions(reason: str = ""):  # reason: optional context tag
        try:
            # Basic distances to S/R in ATR units (positive means beyond the level)
            dist_res_atr = (c - float(state.last_resistance)) / max(1e-9, atr) if state.last_resistance else 0.0
            dist_sup_atr = (float(state.last_support) - c) / max(1e-9, atr) if state.last_support else 0.0
            # Confirmation progress based on current state
            conf_need = int(s.confirmation_candles)
            conf_have = 0
            st = state.state
            if st in ("HL_FORMED", "RESISTANCE_BROKEN"):
                conf_have = count_confirmation_candles(df, "long", conf_need)
            elif st in ("LH_FORMED", "SUPPORT_BROKEN"):
                conf_have = count_confirmation_candles(df, "short", conf_need)
            # Trend structure snapshot (best-effort from last pivots when available)
            # Recompute minimally to avoid holding external state
            try:
                ph2 = pd.Series(_pivot_high(df["high"], s.left, s.right)).dropna()
                pl2 = pd.Series(_pivot_low(df["low"], s.left, s.right)).dropna()
                trend_up = bool(len(ph2) >= 2 and len(pl2) >= 2 and float(ph2.iloc[-1]) > float(ph2.iloc[-2]) and float(pl2.iloc[-1]) > float(pl2.iloc[-2]))
                trend_dn = bool(len(ph2) >= 2 and len(pl2) >= 2 and float(ph2.iloc[-1]) < float(ph2.iloc[-2]) and float(pl2.iloc[-1]) < float(pl2.iloc[-2]))
            except Exception:
                trend_up = trend_dn = False
            logger.info(
                f"[{symbol}] Preconditions: state={state.state} res={state.last_resistance:.4f} sup={state.last_support:.4f} "
                f"dResATR={dist_res_atr:.2f} dSupATR={dist_sup_atr:.2f} conf={conf_have}/{conf_need} "
                f"trendUp={trend_up} trendDn={trend_dn}{(' reason='+reason) if reason else ''}"
            )
        except Exception:
            pass

    # Cooldown logic to prevent multiple signals in quick succession
    if state.last_signal_candle_time:
        time_since_last_signal = df.index[-1] - state.last_signal_candle_time
        # Assuming 15-minute candles, convert min_candles_between_signals to timedelta
        cooldown_duration = pd.Timedelta(minutes=s.min_candles_between_signals * 15)
        if time_since_last_signal < cooldown_duration:
            _log_preconditions("cooldown")
            return None # Still in cooldown period
    
    # Require minimum candles for reliable S/R detection
    min_candles = 200
    if len(df) < min_candles:
        # Not enough history for stable S/R
        try:
            logger.info(f"[{symbol}] Preconditions: insufficient_history ({len(df)}/{min_candles})")
        except Exception:
            pass
        return None
    
    # Calculate current S/R levels
    high, low, close, vol = df["high"], df["low"], df["close"], df["volume"]
    
    ph = pd.Series(_pivot_high(high, s.left, s.right), index=df.index)
    pl = pd.Series(_pivot_low(low, s.left, s.right), index=df.index)
    
    dh = ph.dropna()
    dl = pl.dropna()
    if len(dh) < 2 or len(dl) < 2:
        _log_preconditions("sr_insufficient")
        return None
    
    lastHigh, prevHigh = float(dh.iloc[-1]), float(dh.iloc[-2])
    lastLow, prevLow = float(dl.iloc[-1]), float(dl.iloc[-2])
    
    # Determine market structure
    trendUp = (lastHigh > prevHigh) and (lastLow > prevLow)
    trendDn = (lastHigh < prevHigh) and (lastLow < prevLow)
    
    nearestRes, nearestSup = lastHigh, lastLow
    
    # Update tracked S/R levels (will be updated after MTF check)
    state.last_resistance = nearestRes
    state.last_support = nearestSup
    
    # Calculate ATR for stop loss
    atr = float(_atr(df, s.atr_len)[-1])
    
    # EMA filter
    ema_ok_long = True
    ema_ok_short = True
    if s.use_ema and len(df) >= s.ema_len:
        ema_val = float(_ema(close, s.ema_len)[-1])
        ema_ok_long = close.iloc[-1] > ema_val
        ema_ok_short = close.iloc[-1] < ema_val
    
    # Volume filter
    vol_ok = True
    if s.use_vol:
        vol_ok = vol.iloc[-1] > vol.rolling(s.vol_len).mean().iloc[-1] * s.vol_mult
    
    c = float(close.iloc[-1])
    current_time = df.index[-1]
    
    # Check for MTF levels if enabled (moved here for current_time)
    if s.use_mtf_sr:
        try:
            # Update MTF levels periodically based on configured interval
            if mtf_sr.should_update(symbol):
                mtf_sr.update_sr_levels(symbol, df)
                logger.debug(f"[{symbol}] Updated HTF S/R levels")
            
            # Check if we should use MTF levels
            original_resistance = nearestRes
            original_support = nearestSup
            
            use_mtf_res, mtf_res, res_reason = should_use_mtf_level(symbol, nearestRes, c, df)
            if use_mtf_res and mtf_res > 0:
                # Validate resistance is above current price
                if mtf_res > c:
                    logger.info(f"[{symbol}] Using MTF resistance: {mtf_res:.4f} ({res_reason})")
                    nearestRes = mtf_res
                else:
                    logger.warning(f"[{symbol}] MTF resistance {mtf_res:.4f} is below price {c:.4f}, keeping original")
            
            use_mtf_sup, mtf_sup, sup_reason = should_use_mtf_level(symbol, nearestSup, c, df)
            if use_mtf_sup and mtf_sup > 0:
                # Validate support is below current price
                if mtf_sup < c:
                    logger.info(f"[{symbol}] Using MTF support: {mtf_sup:.4f} ({sup_reason})")
                    nearestSup = mtf_sup
                else:
                    logger.warning(f"[{symbol}] MTF support {mtf_sup:.4f} is above price {c:.4f}, keeping original")
                
        except Exception as e:
            logger.debug(f"[{symbol}] MTF S/R check failed: {e}")
    
    # Update tracked S/R levels after MTF check
    state.last_resistance = nearestRes
    state.last_support = nearestSup
    
    # State machine for pullback strategy
    if state.state == "NEUTRAL":
        # Check for initial breakout
        if trendUp and c > nearestRes and vol_ok and ema_ok_long:
            # Require 15m bar close if configured
            try:
                if s.require_main_confirm_for_breakout and not bool(getattr(s, 'current_bar_confirmed', False)):
                    return None
            except Exception:
                pass
            # Resistance broken - wait for pullback
            state.state = "RESISTANCE_BROKEN"
            state.breakout_level = nearestRes
            state.breakout_time = current_time
            state.confirmation_count = 0
            # Find and store previous pivot low for structure-based stop
            current_idx = len(df) - 1
            state.previous_pivot_low = find_previous_pivot_low(df, current_idx)
            msg = f"[{symbol}] Resistance broken at {nearestRes:.4f} — focusing 3m pullback → BOS"
            logger.info(msg)
            _notify(symbol, f"🔵 Trend: {msg}")
            _persist_state(symbol, state)
            
        elif trendDn and c < nearestSup and vol_ok and ema_ok_short:
            try:
                if s.require_main_confirm_for_breakout and not bool(getattr(s, 'current_bar_confirmed', False)):
                    return None
            except Exception:
                pass
            # Support broken - wait for pullback
            state.state = "SUPPORT_BROKEN"
            state.breakout_level = nearestSup
            state.breakout_time = current_time
            state.confirmation_count = 0
            # Find and store previous pivot high for structure-based stop
            current_idx = len(df) - 1
            state.previous_pivot_high = find_previous_pivot_high(df, current_idx)
            msg = f"[{symbol}] Support broken at {nearestSup:.4f} — focusing 3m pullback → BOS"
            logger.info(msg)
            _notify(symbol, f"🔵 Trend: {msg}")
            _persist_state(symbol, state)
    
    elif state.state == "RESISTANCE_BROKEN":
        # Long setup: 3m pullback → last LH pivot → BOS above LH
        df3 = df
        try:
            if s.use_3m_pullback and _get_df3m is not None:
                d3 = _get_df3m(symbol)
                if d3 is not None and len(d3) >= 30:
                    df3 = d3
        except Exception:
            df3 = df
        # Optional pre-pullback micro confirmation note (HH/HL since breakout)
        try:
            if (not state.micro_note_sent) and (not state.retest_ok) and state.breakout_time is not None and df3 is not df:
                # Use last ~20 bars after breakout
                seg = df3[df3.index >= state.breakout_time].tail(30)
                if len(seg) >= 6 and _micro_confirms(seg, 'long'):
                    _notify(symbol, f"🔹 Trend: [{symbol}] 3m micro confirms HH/HL since breakout")
                    state.micro_note_sent = True
        except Exception:
            pass
        # Invalidation: fell back below breakout buffer
        try:
            atr15 = _atr_val(df, s.atr_len)
            if c < (state.breakout_level - s.breakout_buffer_atr * atr15):
                msg = f"[{symbol}] Invalidation: fell below breakout buffer, reset"
                logger.info(msg); _notify(symbol, f"🛑 Trend: {msg}")
                try:
                    if _invalidation_hook is not None:
                        info = {
                            'breakout_level': float(state.breakout_level or 0.0),
                            'direction': 'long',
                            'breakout_time': str(state.breakout_time) if state.breakout_time else ''
                        }
                        _invalidation_hook(symbol, info)
                except Exception:
                    pass
                state.state = "NEUTRAL"; state.confirmation_count = 0; state.micro_state = ""; state.retest_ok=False; state.last_counter_pivot=0.0; state.bos_cross_notified=False; state.waiting_reason=""; state.bos_cross_time=None
                _persist_state(symbol, state)
                return None
        except Exception:
            pass
        # Retest proximity
        if s.retest_enabled and not state.retest_ok:
            try:
                if len(df3) >= 10:
                    recent = df3.tail(20)
                    min_dist = min(abs(float(x) - float(state.breakout_level)) for x in recent['low'][-20:])
                    ok = (min_dist <= (s.retest_max_dist_atr * max(1e-9, _atr_val(df, s.atr_len)))) if s.retest_distance_mode=='atr' else (min_dist <= float(state.breakout_level)*(s.retest_max_dist_pct/100.0))
                    if ok:
                        state.retest_ok = True; state.micro_state = "PULLBACK_3M"; state.micro_start_time = current_time
                        msg = f"[{symbol}] 3m retest near breakout ({min_dist:.4f}) — pullback started"
                        logger.info(msg); _notify(symbol, f"🔵 Trend: {msg}")
            except Exception:
                pass
        # Optional protective HL before BOS
        protective_hl_ok = True
        if s.require_protective_hl_for_long:
            try:
                protective_hl_ok = detect_higher_low(df3, state.breakout_level)
                if protective_hl_ok and state.state != "HL_FORMED":
                    state.state = "HL_FORMED"; state.pullback_time = current_time
                    try:
                        state.pullback_extreme = df3['low'].iloc[-10:].min()
                    except Exception:
                        state.pullback_extreme = df['low'].iloc[-10:].min()
                    msg = f"[{symbol}] 3m HL formed above breakout; BOS pending"
                    logger.info(msg); _notify(symbol, f"🔵 Trend: {msg}")
                    # If BOS was already broken earlier (armed) or is currently above pivot, send signal immediately
                    try:
                        bos_cross_now = (state.last_counter_pivot > 0) and (float(df3['close'].iloc[-1]) > state.last_counter_pivot)
                    except Exception:
                        bos_cross_now = False
                    if (state.bos_cross_notified or bos_cross_now) and state.retest_ok:
                        # In strict mode, require divergence before immediate send on protective HL
                        try:
                            if s.div_enabled and s.div_mode == 'strict' and not bool(state.divergence_ok):
                                state.waiting_reason = 'WAIT_DIV'
                                _notify(symbol, f"⏳ Trend: [{symbol}] Protective HL formed; waiting for divergence (strict)")
                                return None
                        except Exception:
                            pass
                        try:
                            entry = float(df3['close'].iloc[-1]); atr = _atr_val(df, s.atr_len)
                            sl_opt3 = (state.pullback_extreme - s.sl_buf_atr*atr) if state.pullback_extreme else (entry - s.sl_buf_atr*atr)
                            sl = min(state.previous_pivot_low - (s.sl_buf_atr*0.3*atr), state.breakout_level - (s.sl_buf_atr*1.6*atr), sl_opt3)
                            if s.extra_pivot_breath_pct > 0:
                                sl = float(sl) - float(entry)*float(s.extra_pivot_breath_pct)
                            R = abs(entry - sl); tp = entry + (s.rr * R * 1.00165)
                            state.state = "SIGNAL_SENT"; state.last_signal_time = current_time; state.last_signal_candle_time = df.index[-1]
                            state.waiting_reason = ""; state.bos_cross_notified = False
                            msg2 = f"[{symbol}] ✅ BOS confirmed (3m) → LONG @ {entry:.4f} (pivot {state.last_counter_pivot:.4f})"
                            logger.info(msg2); _notify(symbol, msg2); _persist_state(symbol, state)
                            if _entry_executor is not None:
                                _entry_executor(symbol, "long", float(entry), float(sl), float(tp), {"phase":"3m_bos","lh_pivot": state.last_counter_pivot, "div_ok": bool(state.divergence_ok), "div_type": state.divergence_type, "div_score": float(state.divergence_score), "div_rsi_delta": float(state.div_rsi_delta), "div_tsi_delta": float(state.div_tsi_delta)})
                            return Signal("long", entry, sl, tp, f"BOS long: break of LH {state.last_counter_pivot:.4f}", {"atr": _atr_val(df, s.atr_len), "breakout_level": state.breakout_level, "lh_pivot": state.last_counter_pivot})
                        except Exception:
                            pass
                    # Start divergence check (strict gating if enabled)
                    try:
                        if s.div_enabled and s.div_mode in ("optional","strict"):
                            ok, div_type, score, det, dr, dt = _hidden_divergence(df3, 'long', s)
                            if ok:
                                state.divergence_ok = True; state.divergence_type = div_type; state.divergence_score = score; state.divergence_time = current_time
                                state.div_rsi_delta = float(dr); state.div_tsi_delta = float(dt)
                                if s.div_notify:
                                    _notify(symbol, f"🔬 Trend: Hidden bullish divergence confirmed (3m) {(''+det) if det else ''}")
                            else:
                                state.divergence_ok = False; state.divergence_type = 'NONE'; state.divergence_score = 0.0; state.divergence_time=None; state.divergence_timeout_notified=False
                                if s.div_mode == 'strict' and s.div_notify:
                                    _notify(symbol, f"🧪 Trend: Divergence check started (3m, strict)")
                    except Exception:
                        pass
            except Exception:
                protective_hl_ok = False
        # Track last counter-trend pivot (LH)
        try:
            piv = _last_pivot(df3['high'], 'high', 2, 2)
            if piv is not None:
                prev = state.last_counter_pivot
                state.last_counter_pivot = float(piv); state.last_counter_time = current_time
                if state.retest_ok and prev == 0.0 and state.last_counter_pivot > 0:
                    m = f"[{symbol}] 3m counter‑trend LH marked @ {state.last_counter_pivot:.4f}"
                    logger.info(m); _notify(symbol, f"🔎 Trend: {m}")
        except Exception:
            pass
        # If BOS level already broken but HL not yet confirmed, notify once and set waiting reason
        try:
            bos_cross = (state.last_counter_pivot > 0) and (float(df3['close'].iloc[-1]) > state.last_counter_pivot)
        except Exception:
            bos_cross = False
        if bos_cross and not protective_hl_ok and not state.bos_cross_notified:
            try:
                _notify(symbol, f"⏳ Trend: [{symbol}] BOS level broken (3m) @ {state.last_counter_pivot:.4f} — waiting for protective HL above breakout")
                state.bos_cross_notified = True; state.bos_cross_time = current_time; state.waiting_reason = 'WAIT_HL'
            except Exception:
                pass
        # Divergence gating (strict): require divergence_ok before BOS is eligible
        if s.div_enabled and s.div_mode == 'strict':
            try:
                # Update divergence within confirm window bars after pivot
                if state.pullback_time is not None and df3 is not df:
                    bars_since_pb = int((df3.index >= state.pullback_time).sum())
                else:
                    bars_since_pb = 0
                if not state.divergence_ok:
                    ok, div_type, score, det, dr, dt = _hidden_divergence(df3, 'long', s)
                    if ok:
                        state.divergence_ok = True; state.divergence_type = div_type; state.divergence_score=score; state.divergence_time=current_time
                        state.div_rsi_delta = float(dr); state.div_tsi_delta = float(dt)
                        if s.div_notify:
                            _notify(symbol, f"🔬 Trend: [{symbol}] Hidden bullish divergence confirmed (3m) {(''+det) if det else ''}")
                    elif s.div_window_bars_3m > 0 and bars_since_pb >= int(s.div_window_bars_3m) and not state.divergence_timeout_notified:
                        state.divergence_timeout_notified = True
                        # If BOS armed hold is active, do not reset; just notify and continue waiting for HL
                        hold_active = False
                        try:
                            if state.bos_cross_time is not None and int(s.bos_armed_hold_minutes) > 0:
                                hold_active = (current_time - state.bos_cross_time) < pd.Timedelta(minutes=int(s.bos_armed_hold_minutes))
                        except Exception:
                            hold_active = False
                        if s.div_notify:
                            if hold_active:
                                _notify(symbol, f"🛑 Trend: [{symbol}] Divergence timeout (3m) — BOS armed hold active; waiting for protective HL")
                            else:
                                _notify(symbol, f"🛑 Trend: [{symbol}] Divergence timeout (3m) — resetting to NEUTRAL")
                        if not hold_active:
                            state.state = "NEUTRAL"; state.micro_state = ""; state.retest_ok=False; state.last_counter_pivot=0.0; state.confirmation_count=0; state.bos_cross_notified=False; state.waiting_reason=""; state.bos_cross_time=None
                            state.divergence_ok=False; state.divergence_type='NONE'; state.divergence_score=0.0; state.divergence_time=None
                            _persist_state(symbol, state)
                            return None
            except Exception:
                pass

        # BOS: close above last LH (ignore body filter) and optional extra closes
        try:
            br = _body_ratio(df3)
            bos_ready = (state.last_counter_pivot > 0) and (float(df3['close'].iloc[-1]) > state.last_counter_pivot)
            body_ok = bool(br >= float(s.bos_body_min_ratio or 0.0))
            # Eligibility (exec): require protective HL; divergence only matters if div_mode=strict; enforce body filter
            eligible_long = (
                state.retest_ok and bos_ready and protective_hl_ok and body_ok and (
                    (not s.div_enabled) or (s.div_mode != 'strict') or bool(state.divergence_ok)
                )
            )
            if eligible_long:
                # Exec-only SR gate: require strong HTF level + confluence + clearance; phantoms remain free
                try:
                    # Under rule_mode, SR gating becomes soft (scored by rule engine)
                    import os
                    rule_mode_env = os.getenv('TREND_RULE_MODE', '1')
                    sr_gate_active = s.sr_exec_enabled and s.use_mtf_sr and (rule_mode_env not in ('1','true','on','yes'))
                    if sr_gate_active:
                        sr_ok = True; sr_reason = ""
                        level = None; strength = 0.0; conf_ok = True; clearance_ok = True
                        try:
                            c15 = float(df['close'].iloc[-1])
                            # Fetch validated levels and pick nearest resistance above price
                            from multi_timeframe_sr import mtf_sr
                            vlevels = mtf_sr.get_price_validated_levels(symbol, c15)
                            res_levels = [(lv, st) for (lv, st, t) in vlevels if t == 'resistance']
                            if res_levels:
                                # nearest by absolute distance
                                level, strength = min(res_levels, key=lambda x: abs(x[0] - c15))
                                # Confluence with lastHigh
                                tol = float(s.sr_confluence_tolerance_pct or 0.0) * max(1e-9, float(level))
                                conf_ok = (abs(float(level) - float(lastHigh)) <= tol)
                                # Clearance beyond the HTF level in ATR(15m)
                                atr15 = _atr_val(df, s.atr_len)
                                clearance = (c15 - float(level)) / max(1e-9, atr15)
                                clearance_ok = (clearance >= float(s.sr_min_break_clear_atr or 0.0))
                                sr_ok = (float(strength) >= float(s.sr_min_strength or 0.0)) and conf_ok and clearance_ok
                            else:
                                sr_ok = False; sr_reason = "no_resistance_level"
                        except Exception as _sre:
                            sr_ok = True; sr_reason = f"sr_check_error:{_sre}"
                        # Persist SR gate status for states
                        try:
                            update_sr_gate(symbol, bool(sr_ok), {
                                'level': float(level or 0.0),
                                'strength': float(strength or 0.0),
                                'conf_ok': bool(conf_ok),
                                'clear_ok': bool(clearance_ok)
                            })
                        except Exception:
                            pass
                        if not sr_ok:
                            # Record phantom and skip execute
                            try:
                                if _phantom_recorder is not None and bos_ready:
                                    entry = float(df3['close'].iloc[-1])
                                    atr15_val = _atr_val(df, s.atr_len)
                                    sl = float(state.breakout_level) - float(s.breakout_sl_buffer_atr)*float(atr15_val)
                                    min_stop = float(entry) - float(entry)*float(s.min_r_pct)
                                    sl = min(sl, min_stop)
                                    if s.extra_pivot_breath_pct > 0:
                                        sl = float(sl) - float(entry)*float(s.extra_pivot_breath_pct)
                                    R = abs(entry - sl); tp = entry + (s.rr * R * 1.00165)
                                    meta = {"phase":"3m_bos_phantom","sr_exec_gate":"fail","sr_strength": float(strength),"sr_level": float(level or 0.0),"sr_conf": bool(conf_ok),"sr_clear": bool(clearance_ok)}
                                    _phantom_recorder(symbol, "long", float(entry), float(sl), float(tp), meta)
                            except Exception:
                                pass
                            try:
                                _notify(symbol, f"🛑 Trend: [{symbol}] SR exec gate blocked — routed to phantom")
                            except Exception:
                                pass
                            try:
                                revert_to_neutral(symbol)
                            except Exception:
                                pass
                            return None
                except Exception:
                    pass
                # Clear waiting reason if any
                state.waiting_reason = ""; state.bos_cross_notified = False
                if s.bos_confirm_closes <= 0:
                    entry = float(df3['close'].iloc[-1]); atr = _atr_val(df, s.atr_len)
                    if s.sl_mode == 'breakout':
                        atr15 = _atr_val(df, s.atr_len)
                        sl = float(state.breakout_level) - float(s.breakout_sl_buffer_atr)*float(atr15)
                        min_stop = float(entry) - float(entry)*float(s.min_r_pct)
                        sl = min(sl, min_stop)
                    else:
                        if s.sl_mode == 'breakout':
                            atr15 = _atr_val(df, s.atr_len)
                            sl = float(state.breakout_level) - float(s.breakout_sl_buffer_atr)*float(atr15)
                            min_stop = float(entry) - float(entry)*float(s.min_r_pct)
                            sl = min(sl, min_stop)
                        else:
                            sl_opt3 = (state.pullback_extreme - s.sl_buf_atr*atr) if state.pullback_extreme else (entry - s.sl_buf_atr*atr)
                            sl = min(state.previous_pivot_low - (s.sl_buf_atr*0.3*atr), state.breakout_level - (s.sl_buf_atr*1.6*atr), sl_opt3)
                    if s.extra_pivot_breath_pct > 0:
                        sl = float(sl) - float(entry)*float(s.extra_pivot_breath_pct)
                    R = abs(entry - sl); tp = entry + (s.rr * R * 1.00165)
                    state.state = "SIGNAL_SENT"; state.last_signal_time = current_time; state.last_signal_candle_time = df.index[-1]
                    msg = f"[{symbol}] ✅ BOS confirmed (3m) → LONG @ {entry:.4f} (pivot {state.last_counter_pivot:.4f})"
                    logger.info(msg); _notify(symbol, msg); _persist_state(symbol, state)
                    if _entry_executor is not None:
                        _entry_executor(symbol, "long", float(entry), float(sl), float(tp), {"phase":"3m_bos","lh_pivot": state.last_counter_pivot, "div_ok": bool(state.divergence_ok), "div_type": state.divergence_type, "div_score": float(state.divergence_score), "div_rsi_delta": float(state.div_rsi_delta), "div_tsi_delta": float(state.div_tsi_delta)})
                    return Signal("long", entry, sl, tp, f"BOS long: break of LH {state.last_counter_pivot:.4f}", {"atr": atr, "breakout_level": state.breakout_level, "lh_pivot": state.last_counter_pivot})
                else:
                    state.bos_pending_confirms += 1
                    try:
                        if state.bos_pending_confirms < int(s.bos_confirm_closes):
                            _notify(symbol, f"🔸 Trend: [{symbol}] BOS confirm {state.bos_pending_confirms}/{int(s.bos_confirm_closes)} (3m)")
                    except Exception:
                        pass
                    if state.bos_pending_confirms >= int(s.bos_confirm_closes):
                        entry = float(df3['close'].iloc[-1]); atr = _atr_val(df, s.atr_len)
                        if s.sl_mode == 'breakout':
                            atr15 = _atr_val(df, s.atr_len)
                            sl = float(state.breakout_level) - float(s.breakout_sl_buffer_atr)*float(atr15)
                            min_stop = float(entry) - float(entry)*float(s.min_r_pct)
                            sl = min(sl, min_stop)
                        else:
                            sl_opt3 = (state.pullback_extreme - s.sl_buf_atr*atr) if state.pullback_extreme else (entry - s.sl_buf_atr*atr)
                            sl = min(state.previous_pivot_low - (s.sl_buf_atr*0.3*atr), state.breakout_level - (s.sl_buf_atr*1.6*atr), sl_opt3)
                        if s.extra_pivot_breath_pct > 0:
                            sl = float(sl) - float(entry)*float(s.extra_pivot_breath_pct)
                        R = abs(entry - sl); tp = entry + (s.rr * R * 1.00165)
                        state.state = "SIGNAL_SENT"; state.last_signal_time = current_time; state.last_signal_candle_time = df.index[-1]
                        msg = f"[{symbol}] ✅ BOS confirmed (3m {state.bos_pending_confirms}/{s.bos_confirm_closes}) → LONG @ {entry:.4f}"
                        logger.info(msg); _notify(symbol, msg); _persist_state(symbol, state)
                        if _entry_executor is not None:
                            _entry_executor(symbol, "long", float(entry), float(sl), float(tp), {"phase":"3m_bos","lh_pivot": state.last_counter_pivot, "div_ok": bool(state.divergence_ok), "div_type": state.divergence_type, "div_score": float(state.divergence_score), "div_rsi_delta": float(state.div_rsi_delta), "div_tsi_delta": float(state.div_tsi_delta)})
                        return Signal("long", entry, sl, tp, f"BOS long: break of LH {state.last_counter_pivot:.4f}", {"atr": atr, "breakout_level": state.breakout_level, "lh_pivot": state.last_counter_pivot})
            else:
                # Not eligible for execution, but BOS crossed: record a phantom for learning
                try:
                    if bos_ready and _phantom_recorder is not None:
                        entry = float(df3['close'].iloc[-1])
                        atr15_val = _atr_val(df, s.atr_len)
                        if s.sl_mode == 'breakout':
                            sl = float(state.breakout_level) - float(s.breakout_sl_buffer_atr)*float(atr15_val)
                            min_stop = float(entry) - float(entry)*float(s.min_r_pct)
                            sl = min(sl, min_stop)
                        else:
                            atr_val = _atr_val(df, s.atr_len)
                            sl_opt3 = (state.pullback_extreme - s.sl_buf_atr*atr_val) if state.pullback_extreme else (entry - s.sl_buf_atr*atr_val)
                            sl = min(state.previous_pivot_low - (s.sl_buf_atr*0.3*atr_val), state.breakout_level - (s.sl_buf_atr*1.6*atr_val), sl_opt3)
                        if s.extra_pivot_breath_pct > 0:
                            sl = float(sl) - float(entry)*float(s.extra_pivot_breath_pct)
                        R = abs(entry - sl); tp = entry + (s.rr * R * 1.00165)
                        meta = {
                            "phase": "3m_bos_phantom",
                            "lh_pivot": float(state.last_counter_pivot or 0.0),
                            "div_ok": bool(state.divergence_ok),
                            "div_type": state.divergence_type,
                            "div_score": float(state.divergence_score or 0.0),
                            "div_rsi_delta": float(state.div_rsi_delta or 0.0),
                            "div_tsi_delta": float(state.div_tsi_delta or 0.0),
                            "protective_pivot_present": bool(protective_hl_ok),
                            "breakout_level": float(state.breakout_level or 0.0)
                        }
                        _phantom_recorder(symbol, "long", float(entry), float(sl), float(tp), meta)
                except Exception:
                    pass
        except Exception:
            pass
        # Timeouts
        try:
            if state.breakout_time is not None and s.breakout_to_pullback_bars_3m > 0 and df3 is not df:
                bars_since = int((df3.index >= state.breakout_time).sum())
                if (not state.retest_ok) and bars_since >= int(s.breakout_to_pullback_bars_3m):
                    msg = f"[{symbol}] Timeout: no 3m pullback near breakout within {s.breakout_to_pullback_bars_3m} bars"
                    logger.info(msg); _notify(symbol, f"🛑 Trend: {msg}")
                state.state = "NEUTRAL"; state.micro_state=""; state.retest_ok=False; state.last_counter_pivot=0.0; state.confirmation_count=0; state.bos_cross_notified=False; state.waiting_reason=""; state.bos_cross_time=None
                _persist_state(symbol, state)
            if state.pullback_time is not None and s.pullback_to_bos_bars_3m > 0 and df3 is not df:
                bars_since_pb = int((df3.index >= state.pullback_time).sum())
                if bars_since_pb >= int(s.pullback_to_bos_bars_3m):
                    msg = f"[{symbol}] Timeout: BOS not confirmed within {s.pullback_to_bos_bars_3m} 3m bars"
                    logger.info(msg); _notify(symbol, f"🛑 Trend: {msg}")
                state.state = "NEUTRAL"; state.micro_state=""; state.retest_ok=False; state.last_counter_pivot=0.0; state.confirmation_count=0; state.bos_cross_notified=False; state.waiting_reason=""; state.bos_cross_time=None
                _persist_state(symbol, state)
        except Exception:
            pass
    
    elif state.state == "SUPPORT_BROKEN":
        # Short setup: 3m pullback → last HL pivot → BOS below HL
        df3 = df
        try:
            if s.use_3m_pullback and _get_df3m is not None:
                d3 = _get_df3m(symbol)
                if d3 is not None and len(d3) >= 30:
                    df3 = d3
        except Exception:
            df3 = df
        # Optional pre-pullback micro confirmation note (LH/LL since breakout)
        try:
            if (not state.micro_note_sent) and (not state.retest_ok) and state.breakout_time is not None and df3 is not df:
                seg = df3[df3.index >= state.breakout_time].tail(30)
                if len(seg) >= 6 and _micro_confirms(seg, 'short'):
                    _notify(symbol, f"🔹 Trend: [{symbol}] 3m micro confirms LH/LL since breakout")
                    state.micro_note_sent = True
        except Exception:
            pass
        # Invalidation: rose back above breakout buffer
        try:
            atr15 = _atr_val(df, s.atr_len)
            if c > (state.breakout_level + s.breakout_buffer_atr * atr15):
                msg = f"[{symbol}] Invalidation: rose above breakout buffer, reset"
                logger.info(msg); _notify(symbol, f"🛑 Trend: {msg}")
                try:
                    if _invalidation_hook is not None:
                        info = {
                            'breakout_level': float(state.breakout_level or 0.0),
                            'direction': 'short',
                            'breakout_time': str(state.breakout_time) if state.breakout_time else ''
                        }
                        _invalidation_hook(symbol, info)
                except Exception:
                    pass
                state.state = "NEUTRAL"; state.confirmation_count = 0; state.micro_state = ""; state.retest_ok=False; state.last_counter_pivot=0.0; state.bos_cross_notified=False; state.waiting_reason=""; state.bos_cross_time=None
                _persist_state(symbol, state)
                return None
        except Exception:
            pass
        # Retest proximity
        if s.retest_enabled and not state.retest_ok:
            try:
                if len(df3) >= 10:
                    recent = df3.tail(20)
                    min_dist = min(abs(float(x) - float(state.breakout_level)) for x in recent['high'][-20:])
                    ok = (min_dist <= (s.retest_max_dist_atr * max(1e-9, _atr_val(df, s.atr_len)))) if s.retest_distance_mode=='atr' else (min_dist <= float(state.breakout_level)*(s.retest_max_dist_pct/100.0))
                    if ok:
                        state.retest_ok = True; state.micro_state = "PULLBACK_3M"; state.micro_start_time = current_time
                        msg = f"[{symbol}] 3m retest near breakout ({min_dist:.4f}) — pullback started"
                        logger.info(msg); _notify(symbol, f"🔵 Trend: {msg}")
            except Exception:
                pass
        # Optional protective LH before BOS
        protective_lh_ok = True
        if s.require_protective_lh_for_short:
            try:
                protective_lh_ok = detect_lower_high(df3, state.breakout_level)
                if protective_lh_ok and state.state != "LH_FORMED":
                    state.state = "LH_FORMED"; state.pullback_time = current_time
                    try:
                        state.pullback_extreme = df3['high'].iloc[-10:].max()
                    except Exception:
                        state.pullback_extreme = df['high'].iloc[-10:].max()
                    msg = f"[{symbol}] 3m LH formed below breakout; BOS pending"
                    logger.info(msg); _notify(symbol, f"🔵 Trend: {msg}")
                    # If BOS was already broken earlier (armed) or is currently below pivot, send signal immediately
                    try:
                        bos_cross_now = (state.last_counter_pivot > 0) and (float(df3['close'].iloc[-1]) < state.last_counter_pivot)
                    except Exception:
                        bos_cross_now = False
                    if (state.bos_cross_notified or bos_cross_now) and state.retest_ok:
                        # Respect divergence strict mode: require divergence_ok before sending
                        try:
                            if s.div_enabled and s.div_mode == 'strict' and not bool(state.divergence_ok):
                                state.waiting_reason = 'WAIT_DIV'
                                _notify(symbol, f"⏳ Trend: [{symbol}] Protective LH formed; waiting for divergence (strict)")
                                return None
                        except Exception:
                            pass
                        try:
                            entry = float(df3['close'].iloc[-1]); atr = _atr_val(df, s.atr_len)
                            sl_opt3 = (state.pullback_extreme + s.sl_buf_atr*atr) if state.pullback_extreme else (entry + s.sl_buf_atr*atr)
                            sl = max(state.previous_pivot_high + (s.sl_buf_atr*0.3*atr), state.breakout_level + (s.sl_buf_atr*1.6*atr), sl_opt3)
                            if s.extra_pivot_breath_pct > 0:
                                sl = float(sl) + float(entry)*float(s.extra_pivot_breath_pct)
                            R = abs(sl - entry); tp = entry - (s.rr * R * 1.00165)
                            state.state = "SIGNAL_SENT"; state.last_signal_time = current_time; state.last_signal_candle_time = df.index[-1]
                            state.waiting_reason = ""; state.bos_cross_notified = False
                            msg2 = f"[{symbol}] ✅ BOS confirmed (3m) → SHORT @ {entry:.4f} (pivot {state.last_counter_pivot:.4f})"
                            logger.info(msg2); _notify(symbol, msg2); _persist_state(symbol, state)
                            if _entry_executor is not None:
                                _entry_executor(symbol, "short", float(entry), float(sl), float(tp), {"phase":"3m_bos","hl_pivot": state.last_counter_pivot, "div_ok": bool(state.divergence_ok), "div_type": state.divergence_type, "div_score": float(state.divergence_score), "div_rsi_delta": float(state.div_rsi_delta), "div_tsi_delta": float(state.div_tsi_delta)})
                            return Signal("short", entry, sl, tp, f"BOS short: break of HL {state.last_counter_pivot:.4f}", {"atr": _atr_val(df, s.atr_len), "breakout_level": state.breakout_level, "hl_pivot": state.last_counter_pivot})
                        except Exception:
                            pass
                    # Start divergence check (strict gating if enabled)
                    try:
                        if s.div_enabled and s.div_mode in ("optional","strict"):
                            ok, div_type, score, det, dr, dt = _hidden_divergence(df3, 'short', s)
                            if ok:
                                state.divergence_ok = True; state.divergence_type = div_type; state.divergence_score = score; state.divergence_time = current_time
                                state.div_rsi_delta = float(dr); state.div_tsi_delta = float(dt)
                                if s.div_notify:
                                    _notify(symbol, f"🔬 Trend: Hidden bearish divergence confirmed (3m) {(''+det) if det else ''}")
                            else:
                                state.divergence_ok = False; state.divergence_type='NONE'; state.divergence_score=0.0; state.divergence_time=None; state.divergence_timeout_notified=False
                                if s.div_mode == 'strict' and s.div_notify:
                                    _notify(symbol, f"🧪 Trend: Divergence check started (3m, strict)")
                    except Exception:
                        pass
            except Exception:
                protective_lh_ok = False
        # Track last counter-trend pivot (HL)
        try:
            piv = _last_pivot(df3['low'], 'low', 2, 2)
            if piv is not None:
                prev = state.last_counter_pivot
                state.last_counter_pivot = float(piv); state.last_counter_time = current_time
                if state.retest_ok and prev == 0.0 and state.last_counter_pivot > 0:
                    m = f"[{symbol}] 3m counter‑trend HL marked @ {state.last_counter_pivot:.4f}"
                    logger.info(m); _notify(symbol, f"🔎 Trend: {m}")
        except Exception:
            pass
        # If BOS level already broken but LH not yet confirmed, notify once and set waiting reason
        try:
            bos_cross = (state.last_counter_pivot > 0) and (float(df3['close'].iloc[-1]) < state.last_counter_pivot)
        except Exception:
            bos_cross = False
        if bos_cross and not protective_lh_ok and not state.bos_cross_notified:
            try:
                _notify(symbol, f"⏳ Trend: [{symbol}] BOS level broken (3m) @ {state.last_counter_pivot:.4f} — waiting for protective LH below breakout")
                state.bos_cross_notified = True; state.bos_cross_time = current_time; state.waiting_reason = 'WAIT_LH'
            except Exception:
                pass
        # Divergence gating (strict): require divergence_ok before BOS is eligible
        if s.div_enabled and s.div_mode == 'strict':
            try:
                if state.pullback_time is not None and df3 is not df:
                    bars_since_pb = int((df3.index >= state.pullback_time).sum())
                else:
                    bars_since_pb = 0
                if not state.divergence_ok:
                    ok, div_type, score, det, dr, dt = _hidden_divergence(df3, 'short', s)
                    if ok:
                        state.divergence_ok = True; state.divergence_type = div_type; state.divergence_score=score; state.divergence_time=current_time
                        state.div_rsi_delta = float(dr); state.div_tsi_delta = float(dt)
                        if s.div_notify:
                            _notify(symbol, f"🔬 Trend: [{symbol}] Hidden bearish divergence confirmed (3m) {(''+det) if det else ''}")
                    elif s.div_window_bars_3m > 0 and bars_since_pb >= int(s.div_window_bars_3m) and not state.divergence_timeout_notified:
                        state.divergence_timeout_notified = True
                        hold_active = False
                        try:
                            if state.bos_cross_time is not None and int(s.bos_armed_hold_minutes) > 0:
                                hold_active = (current_time - state.bos_cross_time) < pd.Timedelta(minutes=int(s.bos_armed_hold_minutes))
                        except Exception:
                            hold_active = False
                        if s.div_notify:
                            if hold_active:
                                _notify(symbol, f"🛑 Trend: [{symbol}] Divergence timeout (3m) — BOS armed hold active; waiting for protective LH")
                            else:
                                _notify(symbol, f"🛑 Trend: [{symbol}] Divergence timeout (3m) — resetting to NEUTRAL")
                        if not hold_active:
                            state.state = "NEUTRAL"; state.micro_state = ""; state.retest_ok=False; state.last_counter_pivot=0.0; state.confirmation_count=0; state.bos_cross_notified=False; state.waiting_reason=""; state.bos_cross_time=None
                            state.divergence_ok=False; state.divergence_type='NONE'; state.divergence_score=0.0; state.divergence_time=None
                            _persist_state(symbol, state)
                            return None
            except Exception:
                pass

        # BOS: close below last HL (ignore body filter)
        try:
            br = _body_ratio(df3)
            bos_ready = (state.last_counter_pivot > 0) and (float(df3['close'].iloc[-1]) < state.last_counter_pivot)
            body_ok = bool(br >= float(s.bos_body_min_ratio or 0.0))
            # Eligibility (exec): require protective LH; divergence only matters if div_mode=strict; enforce body filter
            eligible_short = (
                state.retest_ok and bos_ready and protective_lh_ok and body_ok and (
                    (not s.div_enabled) or (s.div_mode != 'strict') or bool(state.divergence_ok)
                )
            )
            if eligible_short:
                # Exec-only SR gate on short path
                try:
                    import os
                    rule_mode_env = os.getenv('TREND_RULE_MODE', '1')
                    sr_gate_active = s.sr_exec_enabled and s.use_mtf_sr and (rule_mode_env not in ('1','true','on','yes'))
                    if sr_gate_active:
                        sr_ok = True; sr_reason = ""
                        level = None; strength = 0.0; conf_ok = True; clearance_ok = True
                        try:
                            c15 = float(df['close'].iloc[-1])
                            from multi_timeframe_sr import mtf_sr
                            vlevels = mtf_sr.get_price_validated_levels(symbol, c15)
                            sup_levels = [(lv, st) for (lv, st, t) in vlevels if t == 'support']
                            if sup_levels:
                                level, strength = min(sup_levels, key=lambda x: abs(x[0] - c15))
                                tol = float(s.sr_confluence_tolerance_pct or 0.0) * max(1e-9, float(level))
                                conf_ok = (abs(float(level) - float(lastLow)) <= tol)
                                atr15 = _atr_val(df, s.atr_len)
                                clearance = (float(level) - c15) / max(1e-9, atr15)
                                clearance_ok = (clearance >= float(s.sr_min_break_clear_atr or 0.0))
                                sr_ok = (float(strength) >= float(s.sr_min_strength or 0.0)) and conf_ok and clearance_ok
                            else:
                                sr_ok = False; sr_reason = "no_support_level"
                        except Exception as _sre:
                            sr_ok = True; sr_reason = f"sr_check_error:{_sre}"
                        # Persist SR gate status for states
                        try:
                            update_sr_gate(symbol, bool(sr_ok), {
                                'level': float(level or 0.0),
                                'strength': float(strength or 0.0),
                                'conf_ok': bool(conf_ok),
                                'clear_ok': bool(clearance_ok)
                            })
                        except Exception:
                            pass
                        if not sr_ok:
                            try:
                                if _phantom_recorder is not None and bos_ready:
                                    entry = float(df3['close'].iloc[-1])
                                    atr15_val = _atr_val(df, s.atr_len)
                                    sl = float(state.breakout_level) + float(s.breakout_sl_buffer_atr)*float(atr15_val)
                                    min_stop = float(entry) + float(entry)*float(s.min_r_pct)
                                    sl = max(sl, min_stop)
                                    if s.extra_pivot_breath_pct > 0:
                                        sl = float(sl) + float(entry)*float(s.extra_pivot_breath_pct)
                                    R = abs(entry - sl); tp = entry - (s.rr * R * 1.00165)
                                    meta = {"phase":"3m_bos_phantom","sr_exec_gate":"fail","sr_strength": float(strength),"sr_level": float(level or 0.0),"sr_conf": bool(conf_ok),"sr_clear": bool(clearance_ok)}
                                    _phantom_recorder(symbol, "short", float(entry), float(sl), float(tp), meta)
                            except Exception:
                                pass
                            try:
                                _notify(symbol, f"🛑 Trend: [{symbol}] SR exec gate blocked — routed to phantom")
                            except Exception:
                                pass
                            try:
                                revert_to_neutral(symbol)
                            except Exception:
                                pass
                            return None
                except Exception:
                    pass
                state.waiting_reason = ""; state.bos_cross_notified = False
                if s.bos_confirm_closes <= 0:
                    entry = float(df3['close'].iloc[-1]); atr = _atr_val(df, s.atr_len)
                    if s.sl_mode == 'breakout':
                        atr15 = _atr_val(df, s.atr_len)
                        sl = float(state.breakout_level) + float(s.breakout_sl_buffer_atr)*float(atr15)
                        min_stop = float(entry) + float(entry)*float(s.min_r_pct)
                        sl = max(sl, min_stop)
                    else:
                        sl_opt3 = (state.pullback_extreme + s.sl_buf_atr*atr) if state.pullback_extreme else (entry + s.sl_buf_atr*atr)
                        sl = max(state.previous_pivot_high + (s.sl_buf_atr*0.3*atr), state.breakout_level + (s.sl_buf_atr*1.6*atr), sl_opt3)
                    if s.extra_pivot_breath_pct > 0:
                        sl = float(sl) + float(entry)*float(s.extra_pivot_breath_pct)
                    R = abs(sl - entry); tp = entry - (s.rr * R * 1.00165)
                    state.state = "SIGNAL_SENT"; state.last_signal_time = current_time; state.last_signal_candle_time = df.index[-1]
                    msg = f"[{symbol}] ✅ BOS confirmed (3m) → SHORT @ {entry:.4f} (pivot {state.last_counter_pivot:.4f})"
                    logger.info(msg); _notify(symbol, msg); _persist_state(symbol, state)
                    if _entry_executor is not None:
                        _entry_executor(symbol, "short", float(entry), float(sl), float(tp), {"phase":"3m_bos","hl_pivot": state.last_counter_pivot, "div_ok": bool(state.divergence_ok), "div_type": state.divergence_type, "div_score": float(state.divergence_score), "div_rsi_delta": float(state.div_rsi_delta), "div_tsi_delta": float(state.div_tsi_delta)})
                    return Signal("short", entry, sl, tp, f"BOS short: break of HL {state.last_counter_pivot:.4f}", {"atr": atr, "breakout_level": state.breakout_level, "hl_pivot": state.last_counter_pivot})
                else:
                    state.bos_pending_confirms += 1
                    try:
                        if state.bos_pending_confirms < int(s.bos_confirm_closes):
                            _notify(symbol, f"🔸 Trend: BOS confirm {state.bos_pending_confirms}/{int(s.bos_confirm_closes)} (3m)")
                    except Exception:
                        pass
                    if state.bos_pending_confirms >= int(s.bos_confirm_closes):
                        entry = float(df3['close'].iloc[-1]); atr = _atr_val(df, s.atr_len)
                        if s.sl_mode == 'breakout':
                            atr15 = _atr_val(df, s.atr_len)
                            sl = float(state.breakout_level) + float(s.breakout_sl_buffer_atr)*float(atr15)
                            min_stop = float(entry) + float(entry)*float(s.min_r_pct)
                            sl = max(sl, min_stop)
                        else:
                            sl_opt3 = (state.pullback_extreme + s.sl_buf_atr*atr) if state.pullback_extreme else (entry + s.sl_buf_atr*atr)
                            sl = max(state.previous_pivot_high + (s.sl_buf_atr*0.3*atr), state.breakout_level + (s.sl_buf_atr*1.6*atr), sl_opt3)
                        if s.extra_pivot_breath_pct > 0:
                            sl = float(sl) + float(entry)*float(s.extra_pivot_breath_pct)
                        R = abs(sl - entry); tp = entry - (s.rr * R * 1.00165)
                        state.state = "SIGNAL_SENT"; state.last_signal_time = current_time; state.last_signal_candle_time = df.index[-1]
                        msg = f"[{symbol}] ✅ BOS confirmed (3m {state.bos_pending_confirms}/{s.bos_confirm_closes}) → SHORT @ {entry:.4f}"
                        logger.info(msg); _notify(symbol, msg); _persist_state(symbol, state)
                        if _entry_executor is not None:
                            _entry_executor(symbol, "short", float(entry), float(sl), float(tp), {"phase":"3m_bos","hl_pivot": state.last_counter_pivot, "div_ok": bool(state.divergence_ok), "div_type": state.divergence_type, "div_score": float(state.divergence_score), "div_rsi_delta": float(state.div_rsi_delta), "div_tsi_delta": float(state.div_tsi_delta)})
                        return Signal("short", entry, sl, tp, f"BOS short: break of HL {state.last_counter_pivot:.4f}", {"atr": atr, "breakout_level": state.breakout_level, "hl_pivot": state.last_counter_pivot})
            else:
                # Not eligible for execution, but BOS crossed: record a phantom (short)
                try:
                    if bos_ready and _phantom_recorder is not None:
                        entry = float(df3['close'].iloc[-1])
                        atr15_val = _atr_val(df, s.atr_len)
                        if s.sl_mode == 'breakout':
                            sl = float(state.breakout_level) + float(s.breakout_sl_buffer_atr)*float(atr15_val)
                            min_stop = float(entry) + float(entry)*float(s.min_r_pct)
                            sl = max(sl, min_stop)
                        else:
                            atr_val = _atr_val(df, s.atr_len)
                            sl_opt3 = (state.pullback_extreme + s.sl_buf_atr*atr_val) if state.pullback_extreme else (entry + s.sl_buf_atr*atr_val)
                            sl = max(state.previous_pivot_high + (s.sl_buf_atr*0.3*atr_val), state.breakout_level + (s.sl_buf_atr*1.6*atr_val), sl_opt3)
                        if s.extra_pivot_breath_pct > 0:
                            sl = float(sl) + float(entry)*float(s.extra_pivot_breath_pct)
                        R = abs(entry - sl); tp = entry - (s.rr * R * 1.00165)
                        meta = {
                            "phase": "3m_bos_phantom",
                            "hl_pivot": float(state.last_counter_pivot or 0.0),
                            "div_ok": bool(state.divergence_ok),
                            "div_type": state.divergence_type,
                            "div_score": float(state.divergence_score or 0.0),
                            "div_rsi_delta": float(state.div_rsi_delta or 0.0),
                            "div_tsi_delta": float(state.div_tsi_delta or 0.0),
                            "protective_pivot_present": bool(protective_lh_ok),
                            "breakout_level": float(state.breakout_level or 0.0)
                        }
                        _phantom_recorder(symbol, "short", float(entry), float(sl), float(tp), meta)
                except Exception:
                    pass
        except Exception:
            pass
        # Timeouts
        try:
            if state.breakout_time is not None and s.breakout_to_pullback_bars_3m > 0 and df3 is not df:
                bars_since = int((df3.index >= state.breakout_time).sum())
                if (not state.retest_ok) and bars_since >= int(s.breakout_to_pullback_bars_3m):
                    msg = f"[{symbol}] Timeout: no 3m pullback near breakout within {s.breakout_to_pullback_bars_3m} bars"
                    logger.info(msg); _notify(symbol, f"🛑 Trend: {msg}")
                    state.state = "NEUTRAL"; state.micro_state=""; state.retest_ok=False; state.last_counter_pivot=0.0; state.confirmation_count=0; state.bos_cross_notified=False; state.waiting_reason=""; state.bos_cross_time=None
                    _persist_state(symbol, state)
            if state.pullback_time is not None and s.pullback_to_bos_bars_3m > 0 and df3 is not df:
                bars_since_pb = int((df3.index >= state.pullback_time).sum())
                # When BOS is armed (pivot broken), hold waiting for protective pivot up to configured minutes
                hold_active = False
                try:
                    if state.bos_cross_time is not None and int(s.bos_armed_hold_minutes) > 0:
                        hold_active = (current_time - state.bos_cross_time) < pd.Timedelta(minutes=int(s.bos_armed_hold_minutes))
                except Exception:
                    hold_active = False
                if bars_since_pb >= int(s.pullback_to_bos_bars_3m) and not hold_active:
                    msg = f"[{symbol}] Timeout: BOS not confirmed within {s.pullback_to_bos_bars_3m} 3m bars"
                    logger.info(msg); _notify(symbol, f"🛑 Trend: {msg}")
                    state.state = "NEUTRAL"; state.micro_state=""; state.retest_ok=False; state.last_counter_pivot=0.0; state.confirmation_count=0; state.bos_cross_notified=False; state.waiting_reason=""; state.bos_cross_time=None
                    _persist_state(symbol, state)
        except Exception:
            pass
    
    elif state.state == "HL_FORMED":
        # New BOS-based flow supersedes legacy 2-candle confirms
        try:
            if getattr(s, 'retest_enabled', False):
                return None
        except Exception:
            pass
        # Long: BOS logic handled in RESISTANCE_BROKEN branch
        pass
    
    elif state.state == "LH_FORMED":
        # New BOS-based flow supersedes legacy 2-candle confirms
        try:
            if getattr(s, 'retest_enabled', False):
                return None
        except Exception:
            pass
        # Count confirmation candles for short
        df_conf = df
        try:
            if s.use_3m_confirm and _get_df3m is not None:
                d3 = _get_df3m(symbol)
                if d3 is not None and len(d3) >= max(10, s.confirmation_candles+2):
                    df_conf = d3
        except Exception:
            df_conf = df
        confirmations = count_confirmation_candles(df_conf, "short", s.confirmation_candles)
        # Verbose per-candle confirmation diagnostics
        try:
            if confirmations > 0 and confirmations < s.confirmation_candles:
                o = float(df["open"].iloc[-1]); cl = float(df["close"].iloc[-1])
                hi = float(df["high"].iloc[-1]); lo = float(df["low"].iloc[-1])
                body = abs(cl - o); rng = max(1e-9, hi - lo)
                body_ratio = body / rng * 100.0
                try:
                    atr_recent = float(_atr(df, s.atr_len)[-1])
                except Exception:
                    atr_recent = rng
                candle_range_atr = (rng / atr_recent * 100.0) if atr_recent > 0 else 0.0
                try:
                    vol_ratio = float(df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1])
                except Exception:
                    vol_ratio = 1.0
                logger.info(f"[{symbol}] LH confirmation {'(3m) ' if df_conf is not df else ''}{confirmations}/{s.confirmation_candles} | Body {body_ratio:.1f}% | Range/ATR {candle_range_atr:.1f}% | Vol× {vol_ratio:.2f}")
        except Exception:
            pass
        # Log each confirmation candle as it accrues (until signal fires)
        try:
            if confirmations > 0 and confirmations < s.confirmation_candles:
                logger.info(f"[{symbol}] LH confirmation {'(3m) ' if df_conf is not df else ''}{confirmations}/{s.confirmation_candles}")
        except Exception:
            pass
        # Notify on confirmation progress (only when it increases)
        try:
            if confirmations > 0 and confirmations > int(state.confirmation_count):
                state.confirmation_count = confirmations
                _notify(symbol, f"🔵 Trend: LH confirmation {confirmations}/{s.confirmation_candles}")
                _persist_state(symbol, state)
        except Exception:
            pass

        if confirmations >= s.confirmation_candles:
            # Generate SHORT signal
            entry = c
            
            # HYBRID STOP LOSS METHOD - use whichever gives more room
            # Option 1: Previous pivot high
            sl_option1 = state.previous_pivot_high + (s.sl_buf_atr * 0.3 * atr)  # Smaller buffer since further away
            
            # Option 2: Breakout level plus ATR
            sl_option2 = state.breakout_level + (s.sl_buf_atr * 1.6 * atr)  # Larger buffer from breakout
            
            # Option 3: Original method (pullback extreme)
            sl_option3 = state.pullback_extreme + (s.sl_buf_atr * atr)
            
            # Use the highest SL (gives most room)
            sl = max(sl_option1, sl_option2, sl_option3)

            # Breathing room: apply to any chosen stop method (move SL further by +extra% of entry)
            if s.extra_pivot_breath_pct > 0:
                old_sl = sl
                sl = float(sl) + float(entry) * float(s.extra_pivot_breath_pct)
                logger.info(f"[{symbol}] Stop breathing: +{s.extra_pivot_breath_pct*100:.1f}% of entry ({old_sl:.4f} -> {sl:.4f})")
            
            # Ensure minimum stop distance (at least 1% from entry)
            min_stop_distance = entry * 0.01
            if abs(sl - entry) < min_stop_distance:
                sl = entry + min_stop_distance
                logger.info(f"[{symbol}] Adjusted stop to minimum distance (1% from entry)")
            
            # Log which method was used
            if sl == sl_option1:
                logger.info(f"[{symbol}] Using previous pivot high for stop: {state.previous_pivot_high:.4f}")
            elif sl == sl_option2:
                logger.info(f"[{symbol}] Using breakout level for stop: {state.breakout_level:.4f}")
            else:
                logger.info(f"[{symbol}] Using pullback extreme for stop: {state.pullback_extreme:.4f}")
            
            if sl <= entry:
                logger.info(f"[{symbol}] Short signal rejected - invalid SL placement")
                state.state = "NEUTRAL"
                return None
            
            R = sl - entry
            # Account for fees and slippage in TP calculation
            # Bybit fees: 0.06% entry + 0.055% exit (limit) = 0.115% total
            # Add 0.05% for slippage = 0.165% total cost
            # To get 2.5:1 after fees, we need to target slightly higher
            fee_adjustment = 1.00165  # Compensate for 0.165% total costs
            tp = entry - (s.rr * R * fee_adjustment)
            
            state.state = "SIGNAL_SENT"
            state.last_signal_time = current_time
            state.last_signal_candle_time = df.index[-1]
            
            sig_msg = f"[{symbol}] 🔴 SHORT SIGNAL (Pullback) - Entry: {entry:.4f}, SL: {sl:.4f}, TP: {tp:.4f}"
            logger.info(sig_msg)
            _notify(symbol, sig_msg)
            _persist_state(symbol, state)
            # Stream-side entry execution (if registered)
            try:
                if _entry_executor is not None:
                    meta = {"phase":"3m_confirm","lh": True}
                    _entry_executor(symbol, "short", float(entry), float(sl), float(tp), meta)
            except Exception:
                pass
            
            return Signal("short", entry, sl, tp,
                         f"Pullback short: LH below {state.breakout_level:.4f} + {s.confirmation_candles} confirmations",
                         {"atr": atr, "res": nearestRes, "sup": nearestSup,
                          "breakout_level": state.breakout_level, "pullback_high": state.pullback_extreme})
        
        # Reset if pullback crosses back into the breakout level (invalidate setup)
        elif df["high"].iloc[-1] > state.breakout_level:
            msg = f"[{symbol}] Pullback crossed back into breakout level ({state.breakout_level:.4f}) — forgetting setup"
            logger.info(msg)
            _notify(symbol, f"🛑 Trend: {msg}")
            state.state = "NEUTRAL"
            state.confirmation_count = 0
            _persist_state(symbol, state)
        # Reset if price breaks above pullback high
        elif df["high"].iloc[-1] > state.pullback_extreme:
            msg = f"[{symbol}] Pullback high broken, resetting to neutral"
            logger.info(msg)
            _notify(symbol, f"🛑 Trend: {msg}")
            state.state = "NEUTRAL"
            state.confirmation_count = 0
        # Reset if confirmations do not arrive within timeout bars
        else:
            try:
                if state.pullback_time is not None:
                    bars_elapsed = int((current_time - state.pullback_time) / pd.Timedelta(minutes=15))
                    if bars_elapsed >= int(s.confirmation_timeout_bars) and confirmations < int(s.confirmation_candles):
                        logger.info(f"[{symbol}] Confirmation timeout ({bars_elapsed} bars) — forgetting setup")
                        state.state = "NEUTRAL"
                        state.confirmation_count = 0
            except Exception:
                pass
    
    elif state.state == "SIGNAL_SENT":
        # Don't send another signal until state is reset
        # State will be reset when position closes (handled in live_bot.py)
        pass
    
    # No signal this bar: emit snapshot of current preconditions
    # Also provide a concise next-step hint when neutral
    try:
        if state.state == "NEUTRAL":
            hints = []
            if state.last_resistance:
                try:
                    if c <= float(state.last_resistance):
                        hints.append(f"close > {float(state.last_resistance):.4f} (resistance)")
                except Exception:
                    pass
            if state.last_support:
                try:
                    if c >= float(state.last_support):
                        hints.append(f"close < {float(state.last_support):.4f} (support)")
                except Exception:
                    pass
            if hints:
                logger.info(f"[{symbol}] Next: need {' OR '.join(hints)} to start pullback sequence")
    except Exception:
        pass
    _log_preconditions()
    return None

def reset_symbol_state(symbol:str):
    """Reset the breakout state for a symbol (called when position closes)"""
    if symbol in breakout_states:
        logger.info(f"[{symbol}] Resetting breakout state to NEUTRAL")
        breakout_states[symbol] = BreakoutState()
        try:
            if _redis_client:
                _redis_client.delete(f'trend:state:{symbol}')
        except Exception:
            pass

def get_trend_states_snapshot() -> Dict[str, dict]:
    """Return a lightweight snapshot of current pullback states per symbol."""
    out: Dict[str, dict] = {}
    for sym, st in breakout_states.items():
        try:
            age_bars = None
            if st.pullback_time:
                try:
                    age_bars = int((datetime.utcnow() - st.pullback_time.to_pydatetime()).total_seconds() // (15*60))
                except Exception:
                    age_bars = None
            out[sym] = {
                'state': st.state,
                'breakout_level': float(st.breakout_level or 0.0),
                'pullback_extreme': float(st.pullback_extreme or 0.0),
                'confirm_progress': int(st.confirmation_count or 0),
                'pullback_age_bars': age_bars,
                # 3m microstructure fields (for transparent monitoring)
                'micro_state': st.micro_state or "",
                'last_counter_pivot': float(st.last_counter_pivot or 0.0),
                # Divergence
                'divergence_ok': bool(st.divergence_ok),
                'divergence_type': st.divergence_type or 'NONE',
                'divergence_score': float(st.divergence_score or 0.0),
                # BOS waiting details
                'bos_crossed': bool(st.bos_cross_notified or (st.bos_cross_time is not None)),
                'waiting_reason': st.waiting_reason or '',
                # Gates and execution
                'gates': {
                    'sr_ok': st.last_sr_gate_ok,
                    'htf_ok': st.last_htf_gate_ok,
                },
                'executed': bool(st.last_executed)
            }
        except Exception:
            out[sym] = {'state': st.state}
    return out
