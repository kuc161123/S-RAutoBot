# Standard library imports
import asyncio
import json
import logging
import os
import signal
import sys
import time
import yaml
import subprocess
from datetime import datetime
from typing import Dict, Optional, Tuple

# Third-party imports
import numpy as np
import pandas as pd
import websockets
from dotenv import load_dotenv
try:
    import redis as _redis
except Exception:
    _redis = None

# Core trading components
from broker_bybit import Bybit, BybitConfig
from candle_storage_postgres import CandleStorage
from multi_websocket_handler import MultiWebSocketHandler
from position_mgr import RiskConfig, Book, Position
from sizer import Sizer
from strategy_pullback import Settings  # Settings are the same for both strategies
from telegram_bot import TGBot

# Optional scalping modules (import granularly; ML scorer optional)
detect_scalp_signal = None
ScalpSettings = None
get_scalp_phantom_tracker = None
get_scalp_scorer = None
SCALP_AVAILABLE = False
try:
    from strategy_scalp import detect_scalp_signal, ScalpSettings
except Exception as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"Scalp import: strategy_scalp unavailable: {e}")
try:
    from scalp_phantom_tracker import get_scalp_phantom_tracker
except Exception as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"Scalp import: scalp_phantom_tracker unavailable: {e}")
try:
    # ML scorer is optional for phantom recording
    from ml_scorer_scalp import get_scalp_scorer
except Exception as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"Scalp import: ml_scorer_scalp unavailable: {e}")

# Consider Scalp available if signal detection and tracker are present
SCALP_AVAILABLE = bool(detect_scalp_signal is not None and get_scalp_phantom_tracker is not None)

# Safe accessor for Scalp Phantom Tracker to avoid local scoping issues
def _safe_get_scalp_phantom_tracker():
    try:
        # Prefer already-imported symbol
        if callable(get_scalp_phantom_tracker):
            return get_scalp_phantom_tracker()
    except Exception:
        pass
    try:
        from scalp_phantom_tracker import get_scalp_phantom_tracker as _g
        return _g()
    except Exception as e:
        try:
            logging.getLogger(__name__).debug(f"Scalp tracker import error: {e}")
        except Exception:
            pass
        return None

# Trade tracking with PostgreSQL fallback
try:
    from trade_tracker_postgres import TradeTrackerPostgres as TradeTracker, Trade
    USING_POSTGRES_TRACKER = True
except ImportError:
    from trade_tracker import TradeTracker, Trade
    USING_POSTGRES_TRACKER = False

# ML system (Trend + Enhanced Mean Reversion only)
try:
    from phantom_trade_tracker import get_phantom_tracker
    from enhanced_mr_scorer import get_enhanced_mr_scorer
    from mr_phantom_tracker import get_mr_phantom_tracker
    from ml_scorer_trend import get_trend_scorer
    from enhanced_market_regime import get_enhanced_market_regime, get_regime_summary
    logger = logging.getLogger(__name__)
    logger.info("Using Enhanced ML System (Trend + Enhanced MR)")
    ML_AVAILABLE = True
    ENHANCED_ML_AVAILABLE = True
except Exception as e:
    ML_AVAILABLE = False
    ENHANCED_ML_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"ML system not available: {e}")

# Disable legacy Mean Reversion (classic) path
def detect_signal_mean_reversion(*args, **kwargs):
    return None

# Symbol data collection for ML learning
try:
    from symbol_data_collector import get_symbol_collector
    SYMBOL_COLLECTOR_AVAILABLE = True
    logger.info("Symbol data collector initialized for future ML")
except ImportError as e:
    SYMBOL_COLLECTOR_AVAILABLE = False
    logger.warning(f"Symbol data collector not available: {e}")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Log which trade tracker we're using
if USING_POSTGRES_TRACKER:
    logger.info("Using PostgreSQL trade tracker for persistence")
else:
    logger.info("Using JSON trade tracker")

# Bot build/version tag (bump to trigger deploys and visible in logs)
VERSION = "2025.10.10.1"

# Load environment variables
load_dotenv()

def new_frame():
    return pd.DataFrame(columns=["open","high","low","close","volume"])

def meta_for(symbol:str, cfg_meta:dict):
    if symbol in cfg_meta: 
        return cfg_meta[symbol]
    return cfg_meta.get("default", {"qty_step":0.001,"min_qty":0.001,"tick_size":0.1})

def replace_env_vars(config:dict) -> dict:
    """Replace ${VAR} placeholders with environment variables"""
    def replace_in_value(value):
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]
            env_value = os.getenv(env_var)
            if env_value is None:
                logger.warning(f"Environment variable {env_var} not found")
                return value
            # Try to parse numbers
            if env_value.isdigit():
                return int(env_value)
            try:
                return float(env_value)
            except:
                return env_value
        elif isinstance(value, dict):
            return {k: replace_in_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [replace_in_value(item) for item in value]
        return value
    
    return {k: replace_in_value(v) for k, v in config.items()}

# --- Adaptive Phantom Flow Controller (phantom-only) ---
class FlowController:
    """Adaptive controller to hit daily phantom targets per strategy.

    Computes a relax ratio r in [0,1] based on pace vs. target and applies
    bounded adjustments to phantom exploration gates. Execution rules remain unchanged.
    """
    def __init__(self, cfg: dict, redis_client=None):
        self.cfg = cfg or {}
        self.enabled = bool(self.cfg.get('phantom_flow', {}).get('enabled', False))
        pf = self.cfg.get('phantom_flow', {})
        self.targets = pf.get('daily_target', {'trend': 40, 'mr': 40, 'scalp': 40})
        self.smoothing_hours = int(pf.get('smoothing_hours', 3))
        self.limits = pf.get('relax_limits', {})
        self.guards = pf.get('min_quality_guards', {})
        self.session_multipliers = pf.get('session_multipliers', {})
        self.burst_cfg = pf.get('burst', {'enabled': True, 'no_accept_minutes': 30, 'boost': 0.25, 'max_boost': 0.40})
        # Advanced relax behavior (deficit/boost/min_relax)
        rm = pf.get('relax_mode', {})
        self.relax_mode = {
            'enable_deficit': bool(rm.get('enable_deficit', False)),
            'deficit_scale': float(rm.get('deficit_scale', 0.30)),
            'weight_deficit': float(rm.get('weight_deficit', 0.6)),
            'catchup': {
                'enabled': bool(rm.get('catchup', {}).get('enabled', False)),
                'pace_gap_trades': int(rm.get('catchup', {}).get('pace_gap_trades', 8)),
                'boost_base': float(rm.get('catchup', {}).get('boost_base', 0.30)),
                'boost_slope': float(rm.get('catchup', {}).get('boost_slope', 0.02)),
                'max_boost': float(rm.get('catchup', {}).get('max_boost', 0.40)),
            },
            'min_relax': {
                'trend': float(rm.get('min_relax', {}).get('trend', 0.0)),
                'mr': float(rm.get('min_relax', {}).get('mr', 0.0)),
                'scalp': float(rm.get('min_relax', {}).get('scalp', 0.0)),
            },
            # Optional WR guard (disabled by default)
            'wr_guard': {
                'enabled': bool(rm.get('wr_guard', {}).get('enabled', False)),
                'window': int(rm.get('wr_guard', {}).get('window', 40)),
                'min_wr': float(rm.get('wr_guard', {}).get('min_wr', 20.0)),
                'cap': float(rm.get('wr_guard', {}).get('cap', 0.60)),
            }
        }
        self.redis = redis_client
        # In-memory fallback state to survive Redis outages
        # Structure: {'accepted': {day: {strategy: int}}, 'relax': {day: {strategy: float}}}
        self._mem = {'accepted': {}, 'relax': {}, 'accepted_ts': {}}
        if self.redis is None:
            # Best-effort local init; safe to fail if REDIS_URL not present
            try:
                import redis as _redis, os as _os
                url = _os.getenv('REDIS_URL')
                if url:
                    self.redis = _redis.from_url(url, decode_responses=True)
                    self.redis.ping()
            except Exception:
                self.redis = None

    def _day(self):
        from datetime import datetime as _dt
        return _dt.utcnow().strftime('%Y%m%d')

    def _get(self, key: str, default: float = 0.0) -> float:
        if not self.redis:
            return default
        try:
            v = self.redis.get(key)
            return float(v) if v is not None else default
        except Exception:
            return default

    def _set(self, key: str, value: float):
        if not self.redis:
            return
        try:
            self.redis.set(key, value)
        except Exception:
            pass

    def increment_accepted(self, strategy: str, amount: int = 1):
        # Always update in-memory state; best-effort Redis
        if not self.enabled:
            return
        day = self._day()
        try:
            day_map = self._mem['accepted'].setdefault(day, {})
            day_map[strategy] = int(day_map.get(strategy, 0)) + int(amount)
            # Track timestamps for inactivity burst
            tsmap = self._mem['accepted_ts'].setdefault(day, {}).setdefault(strategy, [])
            import time as _t
            tsmap.append(int(_t.time()))
            # Keep last 500 entries
            if len(tsmap) > 500:
                self._mem['accepted_ts'][day][strategy] = tsmap[-500:]
        except Exception:
            pass
        if self.redis:
            try:
                self.redis.incrby(f'phantom:flow:{day}:{strategy}:accepted', amount)
            except Exception:
                pass

    def _pace_relax(self, strategy: str, accepted: int) -> Tuple[float, float, float]:
        """Compute pace-based relax and supporting values.
        Returns (r_pace, pace_target, target).
        """
        day = self._day()
        # Compute pace target by hours elapsed
        from datetime import datetime as _dt
        hours_elapsed = max(1, _dt.utcnow().hour)
        target = float(self.targets.get(strategy, 40))
        pace_target = target * min(1.0, hours_elapsed / 24.0)
        deficit = max(0.0, pace_target - float(accepted))
        base_r = min(1.0, deficit / max(1.0, target * 0.5))
        return base_r, pace_target, target

    def _relax_ratio(self, strategy: str) -> float:
        if not self.enabled:
            return 0.0
        day = self._day()
        # Accepted count from Redis or memory fallback
        accepted = 0
        if self.redis:
            try:
                accepted = int(self.redis.get(f'phantom:flow:{day}:{strategy}:accepted') or 0)
            except Exception:
                accepted = 0
        if accepted == 0:
            try:
                accepted = int(self._mem['accepted'].get(day, {}).get(strategy, 0))
            except Exception:
                accepted = 0
        # Pace-based relax
        r_pace, pace_target, target = self._pace_relax(strategy, accepted)
        # Deficit relax (total-day perspective)
        rm = self.relax_mode
        r_def = 0.0
        if rm.get('enable_deficit', False):
            try:
                scale = max(0.05, float(rm.get('deficit_scale', 0.30)))
                r_def = (max(0.0, target - float(accepted)) / max(1.0, target * scale))
                r_def = float(max(0.0, min(1.0, r_def)))
            except Exception:
                r_def = 0.0
        # Catch-up boost when behind pace by X trades
        r_boost = 0.0
        try:
            cu = rm.get('catchup', {})
            if cu.get('enabled', False):
                gap = max(0.0, pace_target - float(accepted))
                if gap >= float(cu.get('pace_gap_trades', 8)):
                    base = float(cu.get('boost_base', 0.30))
                    slope = float(cu.get('boost_slope', 0.02))
                    maxb = float(cu.get('max_boost', 0.40))
                    r_boost = min(maxb, base + slope * (gap - float(cu.get('pace_gap_trades', 8))))
        except Exception:
            r_boost = 0.0
        # Blend deficit with pace
        try:
            w = float(rm.get('weight_deficit', 0.6)) if rm.get('enable_deficit', False) else 0.0
            r_blend = max(r_pace, (w * r_def) + ((1.0 - w) * r_pace))
        except Exception:
            r_blend = r_pace
        # Apply boost and clamp
        r_inst = float(max(0.0, min(1.0, r_blend + r_boost)))
        # Apply inactivity burst if no accepts in recent window
        try:
            if bool(self.burst_cfg.get('enabled', True)):
                window_min = int(self.burst_cfg.get('no_accept_minutes', 30))
                import time as _t
                cutoff = int(_t.time()) - window_min * 60
                recent = False
                try:
                    tsmap = self._mem.get('accepted_ts', {}).get(day, {}).get(strategy, [])
                    if any(t >= cutoff for t in tsmap):
                        recent = True
                except Exception:
                    recent = True
                if not recent:
                    burst = float(self.burst_cfg.get('boost', 0.25))
                    r_inst = min(1.0, r_inst + burst)
        except Exception:
            pass
        # Session multiplier
        try:
            mult = 1.0
            sess_map = self.session_multipliers.get(strategy, {}) if isinstance(self.session_multipliers, dict) else {}
            sess = self._session_key()
            mult = float(sess_map.get(sess, 1.0)) if isinstance(sess_map, dict) else 1.0
            r_inst = max(0.0, min(1.0, r_inst * mult))
        except Exception:
            pass
        # Apply per-strategy minimum relax
        try:
            min_map = rm.get('min_relax', {})
            min_r = float(min_map.get(strategy, 0.0))
            r_inst = max(min_r, r_inst)
        except Exception:
            pass
        # WR guard clamp (optional)
        try:
            wg = rm.get('wr_guard', {})
            if wg.get('enabled', False):
                if self.redis:
                    try:
                        window = max(5, int(wg.get('window', 40)))
                        min_wr = float(wg.get('min_wr', 20.0)) / 100.0
                        cap = float(wg.get('cap', 0.60))
                        # Map strategy to WR key
                        key = f"phantom:wr:{strategy}"
                        vals = self.redis.lrange(key, 0, window - 1) or []
                        wr = None
                        guard_active = False
                        if len(vals) >= max(5, window // 2):
                            try:
                                wins = sum(1 for v in vals if str(v) == '1')
                                wr = wins / len(vals) if len(vals) > 0 else None
                            except Exception:
                                wr = None
                        if wr is not None and wr < min_wr:
                            r_inst = min(r_inst, cap)
                            guard_active = True
                        # Stash WR and guard state into components for UI/debug
                        try:
                            comps = self._mem.setdefault('relax_components', {})
                            dmap = comps.setdefault(day, {})
                            d = dmap.setdefault(strategy, {})
                            d['wr'] = float(wr) if wr is not None else None
                            d['guard_cap'] = cap
                            d['guard_active'] = guard_active
                        except Exception:
                            pass
                        # After scanning all symbols, publish Range state snapshot to shared + Redis
                        try:
                            from datetime import datetime as _dt
                            import json as _json
                            in_range_cnt = sum(1 for st in (self._range_symbol_state or {}).values() if st.get('in_range'))
                            near_edge_cnt = sum(1 for st in (self._range_symbol_state or {}).values() if st.get('near_edge'))
                            # Phantom open count (from tracker, persisted in Redis)
                            ph_open = 0
                            try:
                                pt = self.shared.get('phantom_tracker')
                                if pt:
                                    for lst in (getattr(pt, 'active_phantoms', {}) or {}).values():
                                        for p in (lst or []):
                                            if (getattr(p, 'strategy_name','') or '').startswith('range') and not getattr(p, 'exit_time', None):
                                                ph_open += 1
                            except Exception:
                                pass
                            exec_today = 0
                            try:
                                if hasattr(self, '_range_exec_counter'):
                                    exec_today = int(self._range_exec_counter.get('count', 0))
                            except Exception:
                                pass
                            tp1_hits = 0
                            try:
                                if hasattr(self, '_redis') and self._redis is not None:
                                    day = _dt.utcnow().strftime('%Y%m%d')
                                    tp1_hits = int(self._redis.get(f'state:range:tp1_hits:{day}') or 0)
                            except Exception:
                                pass
                            snapshot = {
                                'ts': _dt.utcnow().isoformat()+'Z',
                                'in_range': in_range_cnt,
                                'near_edge': near_edge_cnt,
                                'exec_today': exec_today,
                                'phantom_open': ph_open,
                                'tp1_mid_hits_today': tp1_hits,
                                'reasons': dict(self._range_reasons or {})
                            }
                            self.shared['range_states'] = snapshot
                            try:
                                if hasattr(self, '_redis') and self._redis is not None:
                                    self._redis.set('state:range:summary', _json.dumps(snapshot))
                            except Exception:
                                pass
                        except Exception:
                            pass
                    except Exception:
                        pass
        except Exception:
            pass
        # Simple EMA smoothing using stored state
        prev = 0.0
        if self.redis:
            key = f'phantom:flow:{day}:{strategy}:relax'
            prev = self._get(key, 0.0)
        else:
            try:
                prev = float(self._mem['relax'].get(day, {}).get(strategy, 0.0))
            except Exception:
                prev = 0.0
        # alpha ~ 1/smoothing_hours (bounded)
        alpha = max(0.2, min(1.0, 1.0 / max(1, self.smoothing_hours)))
        r = prev * (1 - alpha) + r_inst * alpha
        if self.redis:
            try:
                self._set(f'phantom:flow:{day}:{strategy}:relax', r)
            except Exception:
                pass
        else:
            try:
                self._mem['relax'].setdefault(day, {})[strategy] = float(max(0.0, min(1.0, r)))
            except Exception:
                pass
        # Stash components for UI/debug (memory only)
        try:
            comps = self._mem.setdefault('relax_components', {})
            dmap = comps.setdefault(day, {})
            dmap[strategy] = {
                'pace': float(r_pace),
                'deficit': float(r_def),
                'boost': float(r_boost),
                'min': float(rm.get('min_relax', {}).get(strategy, 0.0)),
                'inst': float(r_inst),
                'smoothed': float(max(0.0, min(1.0, r)))
            }
        except Exception:
            pass
        return float(max(0.0, min(1.0, r)))

    def _session_key(self) -> str:
        try:
            from datetime import datetime as _dt
            h = _dt.utcnow().hour
            if 0 <= h < 8:
                return 'asian'
            if 8 <= h < 16:
                return 'european'
            if 16 <= h < 24:
                return 'us'
            return 'off_hours'
        except Exception:
            return 'off_hours'

    # --- Per-strategy gate adjustments ---
    def adjust_trend(self, slope_min: float, ema_min: float, breakout_min: float) -> Dict[str, float]:
        r = self._relax_ratio('trend')
        lim = self.limits.get('trend', {})
        g = self.guards.get('trend', {})
        slope_adj = slope_min - r * float(lim.get('slope', 0.0))
        ema_adj = ema_min - r * float(lim.get('ema_stack', 0.0))
        br_adj = breakout_min - r * float(lim.get('breakout', 0.0))
        # Apply guards
        slope_adj = max(float(g.get('slope_min', 0.0)), slope_adj)
        ema_adj = max(float(g.get('ema_stack_min', 0.0)), ema_adj)
        br_adj = max(float(g.get('breakout_min', 0.0)), br_adj)
        return {'slope_min': slope_adj, 'ema_min': ema_adj, 'breakout_min': br_adj, 'relax': r}

    # Backward-compat shim (no-op mapping)
    def adjust_pullback(self, trend_min: float, confirm_min: float, mtf_min: float) -> Dict[str, float]:
        return {'trend_min': trend_min, 'confirm_min': confirm_min, 'mtf_min': mtf_min, 'relax': 0.0}

    def adjust_mr(self, rc_min: float, touches_min: int, dist_mid_min: float, rev_min: float) -> Dict[str, float]:
        r = self._relax_ratio('mr')
        lim = self.limits.get('mr', {})
        g = self.guards.get('mr', {})
        rc_adj = rc_min - r * float(lim.get('rc', 0.0))
        touches_adj = touches_min - int(round(r * float(lim.get('touches', 0.0))))
        dist_adj = dist_mid_min - r * float(lim.get('dist_mid_atr', 0.0))
        rev_adj = rev_min - r * float(lim.get('rev_atr', 0.0))
        # Apply guards
        rc_adj = max(float(g.get('rc_min', 0.0)), rc_adj)
        touches_adj = max(int(g.get('touches_min', 0)), touches_adj)
        dist_adj = max(float(g.get('dist_mid_atr_min', 0.0)), dist_adj)
        rev_adj = max(float(g.get('rev_candle_atr_min', 0.0)), rev_adj)
        return {'rc_min': rc_adj, 'touches_min': touches_adj, 'dist_mid_min': dist_adj, 'rev_min': rev_adj, 'relax': r}

    def adjust_scalp(self, sc_settings):
        """Mutate a ScalpSettings instance using relax ratio."""
        r = self._relax_ratio('scalp')
        lim = self.limits.get('scalp', {})
        g = self.guards.get('scalp', {})
        try:
            sc_settings.vwap_dist_atr_max = min(float(g.get('vwap_dist_atr_max', sc_settings.vwap_dist_atr_max)),
                                                sc_settings.vwap_dist_atr_max + r * float(lim.get('vwap_dist_atr', 0.0)))
            sc_settings.min_bb_width_pct = max(float(g.get('min_bb_width_pct', 0.0)),
                                               sc_settings.min_bb_width_pct - r * float(lim.get('bb_width_pct', 0.0)))
            sc_settings.vol_ratio_min = max(float(g.get('vol_ratio_min', 0.0)),
                                            sc_settings.vol_ratio_min - r * float(lim.get('vol_ratio', 0.0)))
            # New: relax wick ratio
            if 'wick' in lim:
                sc_settings.wick_ratio_min = max(0.0, sc_settings.wick_ratio_min - r * float(lim.get('wick', 0.0)))
        except Exception:
            pass
        return sc_settings

    # --- Introspection helpers for debugging/QA ---
    def get_status(self) -> dict:
        """Return flow controller status snapshot for UI/debug.
        Includes enabled, targets, accepted counts and relax ratios per strategy.
        """
        day = self._day()
        strategies = ['trend', 'mr', 'scalp']
        out = {
            'enabled': self.enabled,
            'targets': self.targets,
            'smoothing_hours': self.smoothing_hours,
            'accepted': {},
            'relax': {},
            'components': {}
        }
        for s in strategies:
            # Accepted from Redis or memory
            acc = 0
            if self.redis:
                try:
                    acc = int(self.redis.get(f'phantom:flow:{day}:{s}:accepted') or 0)
                except Exception:
                    acc = 0
            if acc == 0:
                acc = int(self._mem['accepted'].get(day, {}).get(s, 0)) if day in self._mem['accepted'] else 0
            out['accepted'][s] = acc
            # Relax from Redis or memory
            rx = 0.0
            if self.redis:
                try:
                    rx = float(self.redis.get(f'phantom:flow:{day}:{s}:relax') or 0.0)
                except Exception:
                    rx = 0.0
            if rx == 0.0:
                rx = float(self._mem['relax'].get(day, {}).get(s, 0.0)) if day in self._mem['relax'] else 0.0
            out['relax'][s] = rx
            # Include components if available
            try:
                comps = self._mem.get('relax_components', {}).get(day, {}).get(s, {})
                out['components'][s] = comps
            except Exception:
                pass
        return out

class TradingBot:
    def __init__(self):
        self.running = False
        self.ws = None
        self.frames: Dict[str, pd.DataFrame] = {}
        self.frames_3m: Dict[str, pd.DataFrame] = {}
        # Scalp secondary stream state + health
        self._scalp_secondary_started: bool = False
        self._scalp_last_confirm: Dict[str, pd.Timestamp] = {}
        self._scalp_stream_tf: Optional[str] = None
        self._scalp_fallback_warned: Dict[str, bool] = {}
        self.tg: Optional[TGBot] = None
        self.bybit = None
        self.storage = CandleStorage()  # Will use DATABASE_URL from environment
        self.last_save_time = datetime.now()
        self.trade_tracker = TradeTracker()  # Initialize trade tracker
        self._tasks = set()
        # Background DB write queue for 3m candles to avoid blocking the event loop
        try:
            import asyncio as _asyncio
            self._db_queue: Optional[_asyncio.Queue] = _asyncio.Queue()
        except Exception:
            self._db_queue = None
        # Per-symbol cooldown for 3m scalp detections (timestamp of last recorded attempt)
        self._scalp_cooldown: Dict[str, pd.Timestamp] = {}
        # Store per-symbol ML features at execution time to record outcomes later
        self._last_signal_features: Dict[str, dict] = {}
        # Best-effort Redis client for lightweight runtime state (e.g., open position strategy hints)
        self._redis = None
        try:
            if _redis is not None:
                url = os.getenv('REDIS_URL')
                if url:
                    self._redis = _redis.from_url(url, decode_responses=True)
                    try:
                        self._redis.ping()
                    except Exception:
                        self._redis = None
        except Exception:
            self._redis = None

        # Scalp diagnostics counters (reset periodically in summaries)
        self._scalp_stats: Dict[str, int] = {
            'confirms': 0,
            'signals': 0,
            'dedup_skips': 0,
            'cooldown_skips': 0
        }
        # Per-symbol execution locks to avoid concurrent duplicate opens/TP/SL
        try:
            import asyncio as _asyncio
            self._exec_locks: Dict[str, _asyncio.Lock] = {}
        except Exception:
            self._exec_locks = {}
        # Track stream-side executions to suppress duplicate main-loop notifications
        self._stream_executed: Dict[str, float] = {}
        # Track if TP/SL has been applied for a symbol (best-effort idempotency)
        self._tpsl_applied: Dict[str, float] = {}
        # Last scalp execution block reason per symbol (for diagnostics)
        self._scalp_last_exec_reason: Dict[str, str] = {}
        # Cache of latest scalp detections per symbol (for promotion timing)
        self._scalp_last_signal: Dict[str, Dict[str, object]] = {}
        # HTF gating persistence (per-strategy per-symbol)
        self._htf_hold: Dict[str, Dict[str, Dict[str, object]]] = {
            'mr': {},
            'trend': {}
        }
        # Trend history readiness tracking (avoid log spam when 15m history is short)
        self._trend_hist_warned: Dict[str, bool] = {}
        self._trend_hist_min: int = 80  # minimum 15m bars required for pullback detection
        # Trend-only mode toggle (set in run() from config/env)
        self._trend_only: bool = False
        # Per-position metadata: breakout levels, policy flags, timestamps
        self._position_meta: Dict[str, dict] = {}
        # Per-symbol HTF exec metrics cache (refresh on 15m close)
        self._htf_exec_cache: Dict[str, Dict[str, object]] = {}
        # Phantom notification de-dup (open/tp1/close)
        self._phantom_open_notified: set[str] = set()
        self._phantom_tp1_notified: set[str] = set()
        self._phantom_close_notified: set[str] = set()

    def _btc_micro_trend(self) -> str:
        """Compute BTC 1–3m micro-trend direction: up/down/none."""
        try:
            for sym in ('BTCUSDT','BTCUSD','BTCUSDC'):
                df3 = self.frames_3m.get(sym)
                if df3 is not None and not df3.empty and len(df3['close']) >= 50:
                    e20 = df3['close'].ewm(span=20, adjust=False).mean()
                    e50 = df3['close'].ewm(span=50, adjust=False).mean()
                    if float(e20.iloc[-1]) > float(e50.iloc[-1]):
                        return 'up'
                    if float(e20.iloc[-1]) < float(e50.iloc[-1]):
                        return 'down'
                    return 'none'
        except Exception:
            pass
        return 'none'

    def _scalp_hard_gates_pass(self, symbol: str, side: str, feats: dict) -> tuple[bool, list[str]]:
        """Execution-only hard gates for Scalp. Returns (ok, reasons)."""
        reasons = []
        try:
            cfg = getattr(self, 'config', {}) or {}
            sc = (cfg.get('scalp', {}) or {})
            hg = (sc.get('hard_gates', {}) or {})
            if not bool(hg.get('apply_to_exec', True)):
                return True, reasons
            # HTF ts15 (check if enabled)
            if bool(hg.get('htf_enabled', False)):
                try:
                    ts15 = float(feats.get('ts15', 0.0) or 0.0)
                    thr_ts = float(hg.get('htf_min_ts15', 60.0))
                    if ts15 < thr_ts:
                        reasons.append(f"htf<{thr_ts:.0f}")
                except Exception:
                    reasons.append('htf:na')
            # Volume ratio (3m) (check if enabled)
            if bool(hg.get('vol_enabled', False)):
                try:
                    vr = float(feats.get('volume_ratio', 0.0) or 0.0)
                    vmin = float(hg.get('vol_ratio_min_3m', 1.3))
                    if vr < vmin:
                        reasons.append(f"vol<{vmin:.2f}")
                except Exception:
                    reasons.append('vol:na')
            # Body ratio with direction (check if enabled)
            if bool(hg.get('body_enabled', False)):
                try:
                    br = float(feats.get('body_ratio', 0.0) or 0.0)
                    bmin = float(hg.get('body_ratio_min_3m', 0.35))
                    bsgn = str(feats.get('body_sign', 'flat'))
                    if br < bmin:
                        reasons.append(f"body<{bmin:.2f}")
                    else:
                        if (side == 'long' and bsgn != 'up') or (side == 'short' and bsgn != 'down'):
                            reasons.append('body_dir')
                except Exception:
                    reasons.append('body:na')
            # 15m alignment (check if enabled - using new flag name)
            try:
                if bool(hg.get('align_15m_enabled', hg.get('align_15m', True))):
                    dir15 = str(feats.get('ema_dir_15m', 'none'))
                    if (side == 'long' and dir15 != 'up') or (side == 'short' and dir15 != 'down'):
                        reasons.append('15m_align')
            except Exception:
                reasons.append('15m:na')
            # Leader alignment with BTC 1–3m micro-trend (check if enabled - using new flag name)
            try:
                if bool(hg.get('leader_align_btc_enabled', hg.get('leader_align_btc', True))) and not symbol.upper().startswith('BTC'):
                    btcd = self._btc_micro_trend()
                    if (side == 'long' and btcd == 'down') or (side == 'short' and btcd == 'up'):
                        reasons.append('btc_opposes')
            except Exception:
                reasons.append('btc:na')
        except Exception:
            reasons.append('gates:error')
        return (len(reasons) == 0), reasons

    def _phantom_trend_regime_ok(self, sym: str, df: 'pd.DataFrame', regime_analysis) -> tuple[bool, str]:
        """Return whether Trend phantom should be recorded under current regime.

        Applies router.htf_bias thresholds if enabled; otherwise uses trend.regime settings.
        """
        try:
            if getattr(self, '_disable_gates', False):
                return True, 'gates_disabled'
        except Exception:
            pass
        # Config override: ungated Trend phantom flow (learn-first)
        try:
            if bool(((self.config.get('phantom', {}) or {}).get('ungated_trend', False))):
                return True, 'ungated_trend'
        except Exception:
            pass
        try:
            cfg = self.config
        except Exception:
            cfg = {}
        # Prefer HTF metrics when enabled
        try:
            htf_cfg = (cfg.get('router', {}) or {}).get('htf_bias', {})
            if bool(htf_cfg.get('enabled', False)):
                metrics = self._get_htf_metrics(sym, df)
                min_ts = float((htf_cfg.get('trend', {}) or {}).get('min_trend_strength', 60.0))
                ts15 = float(metrics.get('ts15', 0.0)); ts60 = float(metrics.get('ts60', 0.0))
                ok = (ts15 >= min_ts) and ((ts60 == 0.0) or (ts60 >= min_ts))
                reason = f"ts15/ts60 {ts15:.1f}/{ts60:.1f} {'>=' if ok else '<'} {min_ts:.1f}"
                return ok, reason
        except Exception:
            pass
        # Fallback to trend.regime config
        try:
            tr_reg = (cfg.get('trend', {}) or {}).get('regime', {})
            if bool(tr_reg.get('enabled', True)):
                prim = getattr(regime_analysis, 'primary_regime', 'unknown') if regime_analysis else 'unknown'
                conf = float(getattr(regime_analysis, 'regime_confidence', 0.0) or 0.0)
                vol = str(getattr(regime_analysis, 'volatility_level', 'normal') or 'normal')
                min_conf = float(tr_reg.get('min_conf', 0.60))
                allowed_vol = set(tr_reg.get('allowed_vol', ['low', 'normal']))
                ok = (prim == 'trending') and (conf >= min_conf) and (vol in allowed_vol)
                reason = f"prim={prim} conf={conf:.2f} vol={vol} min_conf={min_conf:.2f} allowed={','.join(sorted(allowed_vol))}"
                return ok, reason
        except Exception:
            pass
        # If no regime config, allow by default
        return True, 'regime_n/a'

    def _phantom_mr_regime_ok(self, sym: str, df: 'pd.DataFrame', regime_analysis) -> tuple[bool, str]:
        """Return whether MR phantom should be recorded under current regime.

        Uses router.htf_bias thresholds for range/ts if enabled; else basic MR regime checks.
        """
        try:
            if getattr(self, '_disable_gates', False):
                return True, 'gates_disabled'
        except Exception:
            pass
        try:
            cfg = self.config
        except Exception:
            cfg = {}
        try:
            htf_cfg = (cfg.get('router', {}) or {}).get('htf_bias', {})
            if bool(htf_cfg.get('enabled', False)):
                metrics = self._get_htf_metrics(sym, df)
                min_rq = float((htf_cfg.get('mr', {}) or {}).get('min_range_quality', 0.60))
                max_ts = float((htf_cfg.get('mr', {}) or {}).get('max_trend_strength', 40.0))
                rc15 = float(metrics.get('rc15', 0.0)); rc60 = float(metrics.get('rc60', 0.0)); ts15 = float(metrics.get('ts15', 0.0)); ts60 = float(metrics.get('ts60', 0.0))
                ok = (rc15 >= min_rq) and ((rc60 == 0.0) or (rc60 >= min_rq)) and (ts15 <= max_ts) and ((ts60 == 0.0) or (ts60 <= max_ts))
                reason = f"rc15/rc60 {rc15:.2f}/{rc60:.2f}≥{min_rq:.2f} & ts15/ts60 {ts15:.1f}/{ts60:.1f}≤{max_ts:.1f}"
                return ok, reason
        except Exception:
            pass
        try:
            mr_reg = (cfg.get('mr', {}) or {}).get('regime', {})
            if bool(mr_reg.get('enabled', True)):
                prim = getattr(regime_analysis, 'primary_regime', 'unknown') if regime_analysis else 'unknown'
                conf = float(getattr(regime_analysis, 'regime_confidence', 0.0) or 0.0)
                min_conf = float(mr_reg.get('min_conf', 0.60))
                ok = (prim == 'ranging') and (conf >= min_conf)
                reason = f"prim={prim} conf={conf:.2f} min_conf={min_conf:.2f}"
                return ok, reason
        except Exception:
            pass
        return True, 'regime_n/a'

    def _resample_ohlcv(self, df: pd.DataFrame, minutes: int) -> pd.DataFrame:
        try:
            rule = f"{int(minutes)}T"
            agg = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }
            rdf = df.resample(rule).agg(agg).dropna()
            return rdf
        except Exception:
            return df

    def _get_htf_metrics(self, symbol: str, df: pd.DataFrame) -> Dict[str, float]:
        """Return composite HTF metrics for gating: rc15, ts15, rc60, ts60."""
        out = {'rc15': 0.0, 'ts15': 0.0, 'rc60': 0.0, 'ts60': 0.0}
        try:
            htf15 = get_enhanced_market_regime(df, symbol)
            out['ts15'] = float(getattr(htf15, 'trend_strength', 0.0) or 0.0)
            try:
                snap = getattr(htf15, 'feature_snapshot', {}) or {}
                out['rc15'] = float(snap.get('range_confidence', 0.0) or 0.0)
            except Exception:
                out['rc15'] = 0.0
        except Exception:
            pass
        # 60m composite if enabled
        try:
            cfg = getattr(self, 'config', {}) or {}
            hcfg = (cfg.get('router', {}) or {}).get('htf_bias', {})
            comp = (hcfg.get('composite', {}) or {}).get('enabled', False)
            if comp:
                df60 = self._resample_ohlcv(df, 60)
                htf60 = get_enhanced_market_regime(df60, symbol)
                out['ts60'] = float(getattr(htf60, 'trend_strength', 0.0) or 0.0)
                try:
                    snap60 = getattr(htf60, 'feature_snapshot', {}) or {}
                    out['rc60'] = float(snap60.get('range_confidence', 0.0) or 0.0)
                except Exception:
                    out['rc60'] = 0.0
        except Exception:
            pass
        return out

    def _compute_symbol_htf_exec_metrics(self, symbol: str, df: pd.DataFrame) -> Dict[str, object]:
        """Compute and cache per-symbol HTF metrics used for exec gating.

        Returns a dict with keys:
          - ts1h, ts4h: trend strength on 1H/4H
          - ema_dir_1h/4h: 'up'/'down'/'none' stack direction
          - ema_ok_1h/4h: bool alignment with intended side (computed elsewhere)
          - ema_dist_1h/4h: |price-EMA50|/EMA50 in pct
          - adx_1h: ADX(14) on 1H
          - rsi_1h: RSI(14) on 1H
          - struct_dir_1h/4h: 'up'/'down'/'none' based on last two pivots
        """
        try:
            if df is None or len(df) < 20:
                return {
                    'ts1h': 0.0, 'ts4h': 0.0,
                    'ema_dir_1h': 'none', 'ema_dir_4h': 'none',
                    'ema_dist_1h': 0.0, 'ema_dist_4h': 0.0,
                    'adx_1h': 0.0, 'rsi_1h': 50.0,
                    'struct_dir_1h': 'none', 'struct_dir_4h': 'none'
                }
            # Cache on latest 15m bar index
            try:
                last_idx = df.index[-1]
            except Exception:
                last_idx = None
            cached = self._htf_exec_cache.get(symbol)
            if cached and cached.get('last_idx') == last_idx:
                return cached.get('metrics', {})

            def _resample(df_in: pd.DataFrame, minutes: int) -> pd.DataFrame:
                try:
                    rule = f"{int(minutes)}T"
                    agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
                    return df_in.resample(rule).agg(agg).dropna()
                except Exception:
                    return df_in

            def _ema(series: 'pd.Series', span: int) -> float:
                try:
                    return float(series.ewm(span=span, adjust=False).mean().iloc[-1])
                except Exception:
                    return 0.0

            def _rsi(series: 'pd.Series', period: int = 14) -> float:
                try:
                    delta = series.diff()
                    up = delta.clip(lower=0).rolling(period).mean()
                    down = (-delta.clip(upper=0)).rolling(period).mean()
                    rs = up / (down.replace(0, 1e-9))
                    r = 100 - (100 / (1 + rs))
                    return float(r.iloc[-1]) if len(r) else 50.0
                except Exception:
                    return 50.0

            def _adx(df_in: 'pd.DataFrame', period: int = 14) -> float:
                try:
                    high = df_in['high']; low = df_in['low']; close = df_in['close']
                    plus_dm = high.diff()
                    minus_dm = low.diff().mul(-1)
                    plus_dm[(plus_dm < 0) | (plus_dm < minus_dm)] = 0
                    minus_dm[(minus_dm < 0) | (minus_dm < plus_dm)] = 0
                    tr1 = high - low
                    tr2 = (high - close.shift(1)).abs()
                    tr3 = (low - close.shift(1)).abs()
                    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                    atr = tr.ewm(alpha=1/period, adjust=False).mean()
                    plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr.replace(0, 1e-9))
                    minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr.replace(0, 1e-9))
                    dx = (plus_di.subtract(minus_di).abs() / (plus_di + minus_di).replace(0, 1e-9)) * 100
                    adx = dx.ewm(alpha=1/period, adjust=False).mean()
                    return float(adx.iloc[-1]) if len(adx) else 0.0
                except Exception:
                    return 0.0

            def _ema_dir(df_in: 'pd.DataFrame') -> tuple[str, float]:
                try:
                    price = float(df_in['close'].iloc[-1])
                    e20 = _ema(df_in['close'], 20)
                    e50 = _ema(df_in['close'], 50)
                    e200 = _ema(df_in['close'], 200) if len(df_in) >= 200 else _ema(df_in['close'], 100)
                    if price > e20 > e50 > e200:
                        d = 'up'
                    elif price < e20 < e50 < e200:
                        d = 'down'
                    else:
                        d = 'none'
                    ema_dist = abs(price - (e50 if e50 else price)) / max(e50, 1e-9)
                    return d, float(ema_dist * 100.0)
                except Exception:
                    return 'none', 0.0

            def _structure_dir(df_in: 'pd.DataFrame') -> str:
                # Simple pivot-based HH/HL vs LH/LL
                try:
                    left = right = 2
                    highs = df_in['high']; lows = df_in['low']
                    piv_hi_idx = []
                    for i in range(left, len(highs) - right):
                        h = highs.iloc[i]
                        if all(h > highs.iloc[i - j - 1] for j in range(left)) and all(h >= highs.iloc[i + j + 1] for j in range(right)):
                            piv_hi_idx.append(i)
                    piv_lo_idx = []
                    for i in range(left, len(lows) - right):
                        l = lows.iloc[i]
                        if all(l < lows.iloc[i - j - 1] for j in range(left)) and all(l <= lows.iloc[i + j + 1] for j in range(right)):
                            piv_lo_idx.append(i)
                    if len(piv_hi_idx) < 2 or len(piv_lo_idx) < 2:
                        return 'none'
                    last_his = highs.iloc[piv_hi_idx[-2]:piv_hi_idx[-1]+1]
                    last_los = lows.iloc[piv_lo_idx[-2]:piv_lo_idx[-1]+1]
                    hh = highs.iloc[piv_hi_idx[-1]] > highs.iloc[piv_hi_idx[-2]]
                    hl = lows.iloc[piv_lo_idx[-1]] > lows.iloc[piv_lo_idx[-2]]
                    if hh and hl:
                        return 'up'
                    ll = highs.iloc[piv_hi_idx[-1]] < highs.iloc[piv_hi_idx[-2]]
                    lh = lows.iloc[piv_lo_idx[-1]] < lows.iloc[piv_lo_idx[-2]]
                    if ll and lh:
                        return 'down'
                    return 'none'
                except Exception:
                    return 'none'

            import pandas as pd  # type: ignore

            df1h = _resample(df, 60)
            df4h = _resample(df, 240)

            # Trend strength via enhanced regime
            try:
                ts1h = float(get_enhanced_market_regime(df1h, symbol).trend_strength) if len(df1h) >= 20 else 0.0
            except Exception:
                ts1h = 0.0
            try:
                ts4h = float(get_enhanced_market_regime(df4h, symbol).trend_strength) if len(df4h) >= 20 else 0.0
            except Exception:
                ts4h = 0.0

            # EMA alignment + distance
            ema_dir_1h, ema_dist_1h = _ema_dir(df1h)
            ema_dir_4h, ema_dist_4h = _ema_dir(df4h) if len(df4h) else ('none', 0.0)

            # Momentum
            adx_1h = _adx(df1h, 14) if len(df1h) >= 30 else 0.0
            rsi_1h = _rsi(df1h['close'], 14) if len(df1h) >= 20 else 50.0

            # Structure
            struct_dir_1h = _structure_dir(df1h) if len(df1h) >= 20 else 'none'
            struct_dir_4h = _structure_dir(df4h) if len(df4h) >= 20 else 'none'

            metrics = {
                'ts1h': ts1h, 'ts4h': ts4h,
                'ema_dir_1h': ema_dir_1h, 'ema_dir_4h': ema_dir_4h,
                'ema_dist_1h': ema_dist_1h, 'ema_dist_4h': ema_dist_4h,
                'adx_1h': adx_1h, 'rsi_1h': rsi_1h,
                'struct_dir_1h': struct_dir_1h, 'struct_dir_4h': struct_dir_4h
            }
            self._htf_exec_cache[symbol] = {'last_idx': last_idx, 'metrics': metrics}
            return metrics
        except Exception:
            return {'ts1h': 0.0, 'ts4h': 0.0, 'ema_dir_1h': 'none', 'ema_dir_4h': 'none', 'ema_dist_1h': 0.0, 'ema_dist_4h': 0.0, 'adx_1h': 0.0, 'rsi_1h': 50.0, 'struct_dir_1h': 'none', 'struct_dir_4h': 'none'}

    def _compute_qscore(self, symbol: str, side: str, df15: 'pd.DataFrame', df3: 'pd.DataFrame' = None) -> tuple[float, dict, list[str]]:
        """Compute rule-based quality score (0–100) with component breakdown and reasons.

        Components: SR (25), HTF (30), BOS/Confirm (15), Micro 3m (10), Risk geometry (10), Divergence (10).
        """
        total = 0.0
        comp = {}
        reasons: list[str] = []
        try:
            # --- SR quality (HTF S/R)
            try:
                from multi_timeframe_sr import mtf_sr
                price = float(df15['close'].iloc[-1]) if (df15 is not None and len(df15) > 0) else 0.0
                vlevels = mtf_sr.get_price_validated_levels(symbol, price)
                if side == 'long':
                    # nearest resistance above price (for clearance eval)
                    res = [(lv, st) for (lv, st, t) in vlevels if t == 'resistance']
                    if res:
                        level, strength = min(res, key=lambda x: abs(x[0] - price))
                        # stronger is better up to 5 touches proxy scaled to 100
                        sr_strength = max(0.0, min(100.0, float(strength) / 5.0 * 100.0))
                        # clearance if we are above that level (post-break), else low
                        atr15 = float(((df15['high'] - df15['low']).rolling(14).mean().iloc[-1]) if len(df15) >= 14 else (df15['high'].iloc[-1] - df15['low'].iloc[-1]))
                        clearance = ((price - level) / max(1e-9, atr15)) if atr15 else 0.0
                        sr_clear = max(0.0, min(100.0, (clearance / 0.25) * 100.0))  # 0.25 ATR maps to 100
                        comp['sr'] = 0.6 * sr_strength + 0.4 * sr_clear
                    else:
                        comp['sr'] = 20.0
                        reasons.append('SR: no_resistance_level')
                else:
                    sup = [(lv, st) for (lv, st, t) in vlevels if t == 'support']
                    if sup:
                        level, strength = min(sup, key=lambda x: abs(x[0] - price))
                        sr_strength = max(0.0, min(100.0, float(strength) / 5.0 * 100.0))
                        atr15 = float(((df15['high'] - df15['low']).rolling(14).mean().iloc[-1]) if len(df15) >= 14 else (df15['high'].iloc[-1] - df15['low'].iloc[-1]))
                        clearance = ((level - price) / max(1e-9, atr15)) if atr15 else 0.0
                        sr_clear = max(0.0, min(100.0, (clearance / 0.25) * 100.0))
                        comp['sr'] = 0.6 * sr_strength + 0.4 * sr_clear
                    else:
                        comp['sr'] = 20.0
                        reasons.append('SR: no_support_level')
            except Exception as _e:
                comp['sr'] = 40.0
                reasons.append(f'SR:error:{_e}')

            # --- HTF quality (1H/4H + composite)
            try:
                htf = self._compute_symbol_htf_exec_metrics(symbol, df15)
                # Alignment mapping relaxed: match=100, mismatch=50, none=40 (neutral)
                d_match = 'up' if side == 'long' else 'down'
                d1 = str(htf.get('ema_dir_1h', 'none'))
                d4 = str(htf.get('ema_dir_4h', 'none'))
                if d1 == d_match:
                    align1h = 100.0
                elif d1 == 'none':
                    align1h = 40.0
                else:
                    align1h = 50.0
                if d4 == d_match:
                    align4h = 100.0
                elif d4 == 'none':
                    align4h = 40.0
                else:
                    align4h = 50.0
                ts1h = max(0.0, min(100.0, float(htf.get('ts1h', 0.0))))
                ts4h = max(0.0, min(100.0, float(htf.get('ts4h', 0.0))))
                adx = max(0.0, min(100.0, float(htf.get('adx_1h', 0.0)) * 3.0))  # scale ADX ~ 0–33 → 0–100
                # Structure mapping relaxed: match=100, none=40, mismatch=0
                s1 = str(htf.get('struct_dir_1h', 'none'))
                s4 = str(htf.get('struct_dir_4h', 'none'))
                if s1 == d_match:
                    struct1h = 100.0
                elif s1 == 'none':
                    struct1h = 40.0
                else:
                    struct1h = 0.0
                if s4 == d_match:
                    struct4h = 100.0
                elif s4 == 'none':
                    struct4h = 40.0
                else:
                    struct4h = 0.0
                comp['htf'] = 0.25 * ts1h + 0.10 * ts4h + 0.20 * align1h + 0.10 * align4h + 0.20 * adx + 0.10 * struct1h + 0.05 * struct4h
            except Exception as _e:
                comp['htf'] = 40.0
                reasons.append(f'HTF:error:{_e}')

            # --- BOS / Confirmations
            try:
                # proxies from last features if available
                confirms = 0
                try:
                    confirms = int(getattr(self, '_last_signal_features', {}).get(symbol, {}).get('confirm_candles', 0))
                except Exception:
                    confirms = 0
                bos = min(100.0, confirms / 2.0 * 100.0)  # 0,50,100 for 0,1,≥2 confirms
                comp['bos'] = bos
            except Exception:
                comp['bos'] = 50.0

            # --- Micro 3m alignment
            try:
                ok3, why3 = self._micro_context_trend(symbol, side)
                comp['micro'] = 100.0 if ok3 else 30.0
                if not ok3:
                    reasons.append(f'micro:{why3}')
            except Exception as _e:
                comp['micro'] = 50.0
                reasons.append(f'micro:error:{_e}')

            # --- Risk geometry
            try:
                # prefer higher R; penalize extremely small ATR
                cl = df15['close']; price = float(cl.iloc[-1]) if len(cl) else 0.0
                atr15 = float(((df15['high'] - df15['low']).rolling(14).mean().iloc[-1]) if len(df15) >= 14 else (df15['high'].iloc[-1] - df15['low'].iloc[-1]))
                rng_med = float((df15['high'] - df15['low']).rolling(20).median().iloc[-1]) if len(df15) >= 20 else atr15
                bw = (rng_med / max(1e-9, price))
                # mid range width (~2%-6%) scores best
                if bw <= 0:
                    comp['risk'] = 40.0
                elif bw < 0.01:
                    comp['risk'] = 50.0
                    reasons.append('risk:range_too_tight')
                elif bw > 0.08:
                    comp['risk'] = 50.0
                    reasons.append('risk:range_too_wide')
                else:
                    comp['risk'] = 80.0
            except Exception:
                comp['risk'] = 50.0

            # --- Divergence (optional)
            try:
                lf = getattr(self, '_last_signal_features', {}).get(symbol, {}) if hasattr(self, '_last_signal_features') else {}
                div_ok = bool(lf.get('div_ok', False))
                comp['div'] = 80.0 if div_ok else 50.0
            except Exception:
                comp['div'] = 50.0

            # Weighted sum (normalized); weights from config with sane defaults
            try:
                rm = (getattr(self, 'config', {}) or {}).get('trend', {}).get('rule_mode', {})
                wcfg = (rm.get('weights', {}) or {})
                w_sr = float(wcfg.get('sr', 0.25)); w_htf = float(wcfg.get('htf', 0.30)); w_bos = float(wcfg.get('bos', 0.15))
                w_micro = float(wcfg.get('micro', 0.10)); w_risk = float(wcfg.get('risk', 0.10)); w_div = float(wcfg.get('div', 0.10))
                w_sum = max(1e-9, (w_sr + w_htf + w_bos + w_micro + w_risk + w_div))
                w_sr /= w_sum; w_htf /= w_sum; w_bos /= w_sum; w_micro /= w_sum; w_risk /= w_sum; w_div /= w_sum
            except Exception:
                w_sr, w_htf, w_bos, w_micro, w_risk, w_div = 0.25, 0.30, 0.15, 0.10, 0.10, 0.10
            q = (
                w_sr * comp.get('sr', 50.0)
                + w_htf * comp.get('htf', 50.0)
                + w_bos * comp.get('bos', 50.0)
                + w_micro * comp.get('micro', 50.0)
                + w_risk * comp.get('risk', 50.0)
                + w_div * comp.get('div', 50.0)
            )
            total = max(0.0, min(100.0, q))
        except Exception as _e:
            total = 50.0
            reasons.append(f'qscore:error:{_e}')
        return float(total), comp, reasons

    def _apply_htf_exec_gate(self, symbol: str, df: 'pd.DataFrame', side: str, threshold: float) -> tuple[bool, float, str, Dict[str, object]]:
        """Apply per-symbol HTF gate for Trend execution.

        Returns: (allowed, new_threshold, reason, metrics)
        - In gated mode, allowed=False skips execution; threshold unchanged
        - In soft mode, allowed may remain False but threshold is increased; caller can re-evaluate ML
        """
        try:
            cfg = getattr(self, 'config', {}) or {}
            gate = ((((cfg.get('trend', {}) or {}).get('exec', {}) or {}).get('htf_gate', {}) or {}))
            enabled = bool(gate.get('enabled', True))
            if not enabled:
                return True, threshold, 'disabled', {}
            mode = str(gate.get('mode', 'gated')).lower()
            min_ts1h = float(gate.get('min_trend_strength_1h', 60.0))
            min_ts4h = float(gate.get('min_trend_strength_4h', 55.0))
            ema_required = bool(gate.get('ema_alignment', True))
            adx_min_1h = float(gate.get('adx_min_1h', 0.0))
            struct_required = bool(gate.get('structure_confluence', False))
            soft_delta = float(gate.get('soft_delta', 5.0))

            m = self._compute_symbol_htf_exec_metrics(symbol, df)
            ts1h = float(m.get('ts1h', 0.0)); ts4h = float(m.get('ts4h', 0.0))
            ema_ok_1h = (m.get('ema_dir_1h') == ('up' if side == 'long' else 'down')) if ema_required else True
            ema_ok_4h = (m.get('ema_dir_4h') == ('up' if side == 'long' else 'down')) if (ema_required and min_ts4h > 0) else True
            adx_ok = (float(m.get('adx_1h', 0.0)) >= adx_min_1h) if adx_min_1h > 0 else True
            struct_ok_1h = (m.get('struct_dir_1h') == ('up' if side == 'long' else 'down')) if struct_required else True
            struct_ok_4h = (m.get('struct_dir_4h') == ('up' if side == 'long' else 'down')) if (struct_required and min_ts4h > 0) else True
            ts_ok = (ts1h >= min_ts1h) and ((min_ts4h <= 0) or (ts4h >= min_ts4h))
            all_ok = ts_ok and ema_ok_1h and ema_ok_4h and adx_ok and struct_ok_1h and struct_ok_4h
            if all_ok:
                try:
                    logger.info(f"[{symbol}] HTF Gate: PASS ts1h={ts1h:.1f}/{min_ts1h:.1f} ts4h={ts4h:.1f}/{min_ts4h:.1f} ema={ema_ok_1h}/{ema_ok_4h} struct={struct_ok_1h}/{struct_ok_4h} adx1h={m.get('adx_1h',0.0):.1f}/{adx_min_1h:.1f} mode={mode}")
                except Exception:
                    pass
                # Event
                try:
                    ev = self.shared.get('trend_events', [])
                    ev.append({'symbol': symbol, 'text': f"HTF PASS ({side}) ts1h={ts1h:.0f} ts4h={ts4h:.0f} ema={ema_ok_1h}/{ema_ok_4h} adx={m.get('adx_1h',0.0):.0f}"})
                    self.shared['trend_events'] = ev[-200:]
                except Exception:
                    pass
                return True, threshold, 'pass', m
            else:
                if mode == 'soft':
                    new_thr = float(threshold) + soft_delta
                    try:
                        logger.info(f"[{symbol}] HTF Gate: SOFT +thr={soft_delta:.0f} → new_thr={new_thr:.0f} (ts1h={ts1h:.1f}/{min_ts1h:.1f} ts4h={ts4h:.1f}/{min_ts4h:.1f} ema={ema_ok_1h}/{ema_ok_4h} struct={struct_ok_1h}/{struct_ok_4h} adx1h={m.get('adx_1h',0.0):.1f}/{adx_min_1h:.1f})")
                    except Exception:
                        pass
                    return False, new_thr, 'soft', m
                else:
                    try:
                        logger.info(f"[{symbol}] HTF Gate: FAIL ts1h={ts1h:.1f}/{min_ts1h:.1f} ts4h={ts4h:.1f}/{min_ts4h:.1f} ema={ema_ok_1h}/{ema_ok_4h} struct={struct_ok_1h}/{struct_ok_4h} adx1h={m.get('adx_1h',0.0):.1f}/{adx_min_1h:.1f} mode=gated")
                    except Exception:
                        pass
                    try:
                        ev = self.shared.get('trend_events', [])
                        ev.append({'symbol': symbol, 'text': f"HTF FAIL ({side}) ts1h={ts1h:.0f} ts4h={ts4h:.0f} ema={ema_ok_1h}/{ema_ok_4h} adx={m.get('adx_1h',0.0):.0f}"})
                        self.shared['trend_events'] = ev[-200:]
                    except Exception:
                        pass
                    return False, threshold, 'gated', m
        except Exception as _ge:
            try:
                logger.debug(f"[{symbol}] HTF gate error: {_ge}")
            except Exception:
                pass
            return True, threshold, 'error', {}


    # --- 3m micro-context checks (diagnostic) ---
    def _micro_context_trend(self, symbol: str, side: str) -> tuple[bool, str]:
        df3 = self.frames_3m.get(symbol)
        if df3 is None or len(df3) < 5:
            return True, 'no_3m'
        try:
            tail = df3['close'].tail(4)
            ema = df3['close'].ewm(span=20, adjust=False).mean().iloc[-1]
            up_seq = tail.iloc[-1] > tail.iloc[-2] >= tail.iloc[-3]
            dn_seq = tail.iloc[-1] < tail.iloc[-2] <= tail.iloc[-3]
            price = float(tail.iloc[-1])
            if side == 'long':
                ok = up_seq or (price > float(ema))
                return ok, ('up_seq' if ok else 'weak_momentum')
            else:
                ok = dn_seq or (price < float(ema))
                return ok, ('dn_seq' if ok else 'weak_momentum')
        except Exception:
            return True, 'err'

    def _micro_context_mr(self, symbol: str, side: str) -> tuple[bool, str]:
        df3 = self.frames_3m.get(symbol)
        if df3 is None or len(df3) < 4:
            return True, 'no_3m'
        try:
            tail = df3['close'].tail(3)
            up = tail.iloc[-1] > tail.iloc[-2]
            dn = tail.iloc[-1] < tail.iloc[-2]
            if side == 'long':
                return up, ('up_tick' if up else 'no_up_tick')
            else:
                return dn, ('dn_tick' if dn else 'no_dn_tick')
        except Exception:
            return True, 'err'

    def _micro_context_scalp(self, symbol: str, side: str) -> tuple[bool, str]:
        """Simple 3m micro context for Scalp: require immediate momentum with side."""
        df3 = self.frames_3m.get(symbol)
        if df3 is None or len(df3) < 3:
            return True, 'no_3m'
        try:
            tail = df3['close'].tail(3)
            up = tail.iloc[-1] > tail.iloc[-2]
            dn = tail.iloc[-1] < tail.iloc[-2]
            if side == 'long':
                return up, ('up_tick' if up else 'no_up_tick')
            else:
                return dn, ('dn_tick' if dn else 'no_dn_tick')
        except Exception:
            return True, 'err'
        
    def _create_task(self, coro):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()
        task = loop.create_task(coro)
        try:
            self._tasks.add(task)
            task.add_done_callback(lambda t: self._tasks.discard(t))
        except Exception:
            pass
        return task

    async def _execute_scalp_trade(self, sym: str, sig_obj, ml_score: float = 0.0, exec_id: str = None) -> bool:
        """Execute a Scalp trade immediately. Returns True if executed.

        Bypasses routing/regime/micro gates. Still subject to hard execution guards
        (existing position, invalid SL/TP, sizing, exchange errors).
        """
        try:
            # Ensure only one execution path runs per symbol to avoid duplicate TP/SL placements
            try:
                lock = self._exec_locks.get(sym)
            except Exception:
                lock = None
            if lock is None:
                import asyncio as _asyncio
                lock = _asyncio.Lock()
                self._exec_locks[sym] = lock
            async with lock:
                bybit = self.bybit
                book = self.book
                sizer = getattr(self, 'sizer', None)
                cfg = getattr(self, 'config', {}) or {}
                shared = getattr(self, 'shared', {}) or {}
            # One position per symbol
                if sym in book.positions:
                    try:
                        self._scalp_last_exec_reason[sym] = 'position_exists'
                    except Exception:
                        pass
                    return False
            # Determine execution id for tracing
                try:
                    if exec_id is None:
                        meta = getattr(sig_obj, 'meta', {}) if hasattr(sig_obj, 'meta') else {}
                        if isinstance(meta, dict):
                            exec_id = meta.get('exec_id')
                    if exec_id is None:
                        import uuid as _uuid
                        exec_id = _uuid.uuid4().hex[:8]
                except Exception:
                    exec_id = 'n/a'
            # Round TP/SL to tick size (directional for SL). Mirror mode can bypass extra adjustments.
                m = meta_for(sym, cfg.get('symbol_meta', {}))
                from position_mgr import round_step
                import math as _math
                tick_size = float(m.get("tick_size", 0.000001))
                # Price formatter based on tick size for consistent logs
                try:
                    import decimal as _dec
                    _d = _dec.Decimal(str(tick_size))
                    _decs = -_d.as_tuple().exponent if _d.as_tuple().exponent < 0 else 4
                except Exception:
                    _decs = 4
                fmt = f"{{:.{_decs}f}}"
                # Mirror mode flag: execute with exact signal SL/TP (only round to tick, ensure 1-tick separation)
                try:
                    use_mirror = bool((((cfg.get('scalp', {}) or {}).get('exec', {}) or {}).get('use_signal_sl_tp', True)))
                except Exception:
                    use_mirror = True
                # Directional rounding helpers
                def _floor_tick(x: float, step: float) -> float:
                    try:
                        return (_math.floor(x / step)) * step
                    except Exception:
                        return round_step(x, step)
                def _ceil_tick(x: float, step: float) -> float:
                    try:
                        return (_math.ceil(x / step)) * step
                    except Exception:
                        return round_step(x, step)
                # In non-mirror mode, enforce a minimum SL distance pre-sizing based on ATR and config
                if not use_mirror:
                    try:
                        df3_pre = self.frames_3m.get(sym)
                        atr14 = 0.0
                        if df3_pre is not None and len(df3_pre) >= 15:
                            pc = df3_pre['close'].shift()
                            tr = (df3_pre['high'] - df3_pre['low']).combine((df3_pre['high'] - pc).abs(), max).combine((df3_pre['low'] - pc).abs(), max)
                            atr14 = float(tr.rolling(14).mean().iloc[-1]) if tr.notna().sum() >= 14 else float((df3_pre['high'] - df3_pre['low']).iloc[-1])
                        try:
                            min_r_pct = float(((self.config.get('scalp', {}) or {}).get('min_r_pct', 0.005)))
                        except Exception:
                            min_r_pct = 0.005
                        entry_pre = float(getattr(sig_obj, 'entry', 0.0) or 0.0)
                        min_dist = max(entry_pre * min_r_pct, 0.60 * atr14, entry_pre * 0.002)
                        if str(getattr(sig_obj, 'side','short')).lower() == 'short':
                            if float(sig_obj.sl) - entry_pre < min_dist:
                                sig_obj.sl = entry_pre + min_dist
                        else:
                            if entry_pre - float(sig_obj.sl) < min_dist:
                                sig_obj.sl = entry_pre - min_dist
                    except Exception:
                        pass
                # Round SL away from entry strictly
                if str(getattr(sig_obj, 'side', 'short')).lower() == 'short':
                    sl_r = _ceil_tick(float(sig_obj.sl), tick_size)
                    if sl_r <= float(sig_obj.entry):
                        sl_r = float(sig_obj.entry) + tick_size
                    sig_obj.sl = float(sl_r)
                    # TP can be rounded toward price safely
                    sig_obj.tp = round_step(float(sig_obj.tp), tick_size)
                else:
                    sl_r = _floor_tick(float(sig_obj.sl), tick_size)
                    if sl_r >= float(sig_obj.entry):
                        sl_r = float(sig_obj.entry) - tick_size
                    sig_obj.sl = float(sl_r)
                    sig_obj.tp = round_step(float(sig_obj.tp), tick_size)
            # Update sizer balance
                try:
                    bal = bybit.get_balance()
                    if bal:
                        sizer.account_balance = bal
                        shared["last_balance"] = bal
                except Exception:
                    pass
            # Sizing
                qty = sizer.qty_for(float(sig_obj.entry), float(sig_obj.sl), m.get("qty_step",0.001), m.get("min_qty",0.001), ml_score=ml_score)
                if qty <= 0:
                    try:
                        self._scalp_last_exec_reason[sym] = 'size_zero'
                        if not hasattr(self, '_scalp_last_exec_detail'):
                            self._scalp_last_exec_detail = {}
                        self._scalp_last_exec_detail[sym] = f"qty={qty}"
                    except Exception:
                        pass
                    return False
            # SL sanity
                try:
                    df_main = self.frames.get(sym)
                    current_price = float(df_main['close'].iloc[-1]) if (df_main is not None and len(df_main)>0) else float(sig_obj.entry)
                except Exception:
                    current_price = float(sig_obj.entry)
                if (sig_obj.side == "long" and float(sig_obj.sl) >= current_price) or (sig_obj.side == "short" and float(sig_obj.sl) <= current_price):
                    try:
                        self._scalp_last_exec_reason[sym] = 'invalid_sl'
                        if not hasattr(self, '_scalp_last_exec_detail'):
                            self._scalp_last_exec_detail = {}
                        self._scalp_last_exec_detail[sym] = f"side={sig_obj.side} sl={float(sig_obj.sl)} price={current_price}"
                    except Exception:
                        pass
                    return False
            # Leverage and market order
                max_lev = int(m.get("max_leverage", 10))
                lev_resp = bybit.set_leverage(sym, max_lev)
                if lev_resp is None:
                    try:
                        self._scalp_last_exec_reason[sym] = 'leverage_error'
                        if not hasattr(self, '_scalp_last_exec_detail'):
                            self._scalp_last_exec_detail = {}
                        self._scalp_last_exec_detail[sym] = 'set_leverage returned None'
                    except Exception:
                        pass
                    return False
                side = "Buy" if sig_obj.side == "long" else "Sell"
                try:
                    m_resp = bybit.place_market(sym, side, qty, reduce_only=False)
                    try:
                        rc = str(m_resp.get('retCode')) if isinstance(m_resp, dict) else 'n/a'
                        rm = str(m_resp.get('retMsg')) if isinstance(m_resp, dict) else 'n/a'
                        oid = None
                        try:
                            oid = (m_resp.get('result') or {}).get('orderId')
                        except Exception:
                            oid = None
                    except Exception:
                        rc = rm = 'n/a'; oid = None
                    logger.info(f"[{sym}|id={exec_id}] Market order placed: side={side} qty={qty} retCode={rc} retMsg={rm} orderId={oid}")
                except Exception as _me:
                    logger.error(f"[{sym}|id={exec_id}] Market order error: {_me}")
                    try:
                        self._scalp_last_exec_reason[sym] = 'market_order_error'
                        if not hasattr(self, '_scalp_last_exec_detail'):
                            self._scalp_last_exec_detail = {}
                        self._scalp_last_exec_detail[sym] = f"{type(_me).__name__}: {_me}"
                    except Exception:
                        pass
                    return False
            # Read back position (avg entry and size) before setting TP/SL
                actual_entry = float(sig_obj.entry)
                pos_qty_for_tpsl = qty
                try:
                    await asyncio.sleep(1.0)
                    position = bybit.get_position(sym)
                    if isinstance(position, dict):
                        if position.get("avgPrice"):
                            actual_entry = float(position["avgPrice"]) or actual_entry
                        try:
                            if position.get("size") is not None:
                                pos_qty_for_tpsl = float(position.get("size")) or pos_qty_for_tpsl
                        except Exception:
                            pass
                except Exception:
                    position = None
                try:
                    logger.info(f"[{sym}|id={exec_id}] Fill readback: entry={fmt.format(actual_entry)} size={pos_qty_for_tpsl}")
                except Exception:
                    pass
            # Rebase SL/TP distances to actual fill to avoid zero-distance protections due to slippage
                try:
                    e_sig = float(getattr(sig_obj, 'entry', actual_entry))
                    sl_sig = float(getattr(sig_obj, 'sl', actual_entry))
                    tp_sig = float(getattr(sig_obj, 'tp', actual_entry))
                    if sig_obj.side == 'long':
                        R = max(tick_size, e_sig - sl_sig)
                        rr = ((tp_sig - e_sig) / R) if R > 0 else float(((self.config.get('scalp', {}) or {}).get('rr', 2.0)))
                        new_sl = actual_entry - R
                        new_tp = actual_entry + rr * R
                        new_sl = _floor_tick(new_sl, tick_size)
                        if new_sl >= actual_entry:
                            new_sl = _floor_tick(actual_entry - (2.0 * tick_size), tick_size)
                        new_tp = _ceil_tick(new_tp, tick_size)
                    else:
                        R = max(tick_size, sl_sig - e_sig)
                        rr = ((e_sig - tp_sig) / R) if R > 0 else float(((self.config.get('scalp', {}) or {}).get('rr', 2.0)))
                        new_sl = actual_entry + R
                        new_tp = actual_entry - rr * R
                        new_sl = _ceil_tick(new_sl, tick_size)
                        if new_sl <= actual_entry:
                            new_sl = _ceil_tick(actual_entry + (2.0 * tick_size), tick_size)
                        new_tp = _floor_tick(new_tp, tick_size)
                    sig_obj.sl = float(new_sl)
                    sig_obj.tp = float(new_tp)
                    try:
                        import decimal as _dec
                        _d = _dec.Decimal(str(tick_size))
                        _decs = -_d.as_tuple().exponent if _d.as_tuple().exponent < 0 else 4
                    except Exception:
                        _decs = 4
                    _fmt = f"{{:.{_decs}f}}"
                    try:
                        logger.info(
                            f"[{sym}] Rebased protections to fill: sig_entry={_fmt.format(e_sig)} fill={_fmt.format(actual_entry)} R={R:.8f} RR={rr:.2f} -> TP={_fmt.format(sig_obj.tp)} SL={_fmt.format(sig_obj.sl)}"
                        )
                    except Exception:
                        pass
                except Exception:
                    pass
            # Update dynamic slippage EWMA from fills (per-symbol)
                try:
                    inst_slip = 0.0
                    try:
                        inst_slip = abs(actual_entry - float(sig_obj.entry)) / max(1e-9, float(sig_obj.entry))
                    except Exception:
                        inst_slip = 0.0
                    if hasattr(self, '_redis') and self._redis is not None:
                        key_s = f'scalp:slip:ewma:{sym}'; key_n = f'scalp:slip:n:{sym}'
                        prev = float(self._redis.get(key_s) or 0.0)
                        n = int(self._redis.get(key_n) or 0)
                        alpha = 0.2  # EWMA weight
                        ewma = prev if n > 0 else inst_slip
                        ewma = (1 - alpha) * ewma + alpha * inst_slip
                        self._redis.set(key_s, str(ewma))
                        self._redis.set(key_n, str(min(n + 1, 100000)))
                except Exception:
                    pass
            # Adjust TP to achieve target R:R net of fees/slippage with optional bias
                try:
                    fee_total_pct = float((self.config.get('trade', {}) or {}).get('fee_total_pct', 0.00165)) if hasattr(self, 'config') else 0.00165
                    # Use the higher of configured slippage and dynamic EWMA
                    slip_pct = 0.0005
                    try:
                        slip_pct = float((((self.config.get('scalp', {}) or {}).get('exec', {}) or {}).get('slippage_pct', 0.0005)))
                    except Exception:
                        pass
                    try:
                        if hasattr(self, '_redis') and self._redis is not None:
                            ewma = float(self._redis.get(f'scalp:slip:ewma:{sym}') or 0.0)
                            slip_pct = max(slip_pct, min(0.003, ewma))
                    except Exception:
                        pass
                    # Optional RR bias to slightly extend TP to counter fee/slippage underfills
                    rr_bias_pct = 0.0
                    try:
                        rr_bias_pct = float((((self.config.get('scalp', {}) or {}).get('exec', {}) or {}).get('rr_bias_pct', 0.0)))
                    except Exception:
                        rr_bias_pct = 0.0
                    # Post-exec enforcement: mirror mode only ensures 1-tick separation; enhanced mode repeats min-distance
                    if use_mirror:
                        try:
                            if str(getattr(sig_obj, 'side','short')).lower() == 'short':
                                if float(sig_obj.sl) <= float(actual_entry):
                                    sig_obj.sl = _ceil_tick(float(actual_entry) + tick_size, tick_size)
                            else:
                                if float(sig_obj.sl) >= float(actual_entry):
                                    sig_obj.sl = _floor_tick(float(actual_entry) - tick_size, tick_size)
                        except Exception:
                            pass
                    else:
                        try:
                            df3_post = self.frames_3m.get(sym)
                            atr14_post = 0.0
                            if df3_post is not None and len(df3_post) >= 15:
                                pc = df3_post['close'].shift()
                                tr = (df3_post['high'] - df3_post['low']).combine((df3_post['high'] - pc).abs(), max).combine((df3_post['low'] - pc).abs(), max)
                                atr14_post = float(tr.rolling(14).mean().iloc[-1]) if df3_post['close'].notna().sum() >= 14 else float((df3_post['high'] - df3_post['low']).iloc[-1])
                            try:
                                min_r_pct = float(((self.config.get('scalp', {}) or {}).get('min_r_pct', 0.005)))
                            except Exception:
                                min_r_pct = 0.005
                            min_dist2 = max(float(actual_entry) * min_r_pct, 0.60 * atr14_post, float(actual_entry) * 0.002)
                            if str(getattr(sig_obj, 'side','short')).lower() == 'short':
                                if float(sig_obj.sl) - float(actual_entry) < min_dist2:
                                    sig_obj.sl = _ceil_tick(float(actual_entry) + min_dist2, tick_size)
                            else:
                                if float(actual_entry) - float(sig_obj.sl) < min_dist2:
                                    sig_obj.sl = _floor_tick(float(actual_entry) - min_dist2, tick_size)
                        except Exception:
                            pass
                    # Compute current RR and adjust TP distance (enhanced mode only). Mirror mode keeps signal TP.
                    rr = None
                    try:
                        if use_mirror:
                            raise RuntimeError('mirror_tp_keep')
                        if sig_obj.side == 'long':
                            R = max(1e-9, float(actual_entry) - float(sig_obj.sl))
                            rr = max(0.1, (float(sig_obj.tp) - float(actual_entry)) / R)
                            gross = rr * R * (1.0 + fee_total_pct + slip_pct) * (1.0 + max(-0.25, min(0.25, rr_bias_pct)))
                            new_tp = float(actual_entry) + gross
                        else:
                            R = max(1e-9, float(sig_obj.sl) - float(actual_entry))
                            rr = max(0.1, (float(actual_entry) - float(sig_obj.tp)) / R)
                            gross = rr * R * (1.0 + fee_total_pct + slip_pct) * (1.0 + max(-0.25, min(0.25, rr_bias_pct)))
                            new_tp = float(actual_entry) - gross
                        # Round TP to tick size
                        from position_mgr import round_step
                        new_tp = round_step(new_tp, tick_size)
                        sig_obj.tp = float(new_tp)
                        try:
                            logger.debug(f"[{sym}] Scalp TP adj: rr={rr:.2f} fee={fee_total_pct*100:.2f}bps slip={slip_pct*100:.2f}bps -> TP {new_tp:.4f}")
                        except Exception:
                            pass
                    except Exception:
                        pass
                except Exception:
                    pass
                # Set TP/SL exactly once: prefer Partial mode, fallback to Full, final fallback reduce-only limit + SL
                try:
                        # Pre-log the TP/SL request so we can audit even if confirmation fails
                        try:
                            logger.info(
                                f"[{sym}] Requesting TP/SL: preferred=Partial qty={pos_qty_for_tpsl} "
                                f"TP={fmt.format(float(sig_obj.tp))} SL={fmt.format(float(sig_obj.sl))}"
                            )
                        except Exception:
                            pass
                        applied_mode = None  # 'partial' | 'full' | 'reduce_only'
                        try:
                            # Prefer Partial (qty-aware, limit TP) using exchange-reported position size
                            tpsl_resp = bybit.set_tpsl(sym, take_profit=float(sig_obj.tp), stop_loss=float(sig_obj.sl), qty=pos_qty_for_tpsl)
                            applied_mode = 'partial'
                            try:
                                rc = str(tpsl_resp.get('retCode')) if isinstance(tpsl_resp, dict) else 'n/a'
                                rm = str(tpsl_resp.get('retMsg')) if isinstance(tpsl_resp, dict) else 'n/a'
                                logger.info(f"[{sym}] trading-stop (Partial) retCode={rc} retMsg={rm}")
                            except Exception:
                                pass
                        except Exception as _pe:
                            logger.warning(f"[{sym}] trading-stop Partial failed: {_pe}; trying Full")
                            try:
                                # Fallback to Full (no qty); does not place limit TP orders
                                tpsl_resp = bybit.set_tpsl(sym, take_profit=float(sig_obj.tp), stop_loss=float(sig_obj.sl))
                                applied_mode = 'full'
                                try:
                                    rc = str(tpsl_resp.get('retCode')) if isinstance(tpsl_resp, dict) else 'n/a'
                                    rm = str(tpsl_resp.get('retMsg')) if isinstance(tpsl_resp, dict) else 'n/a'
                                    logger.info(f"[{sym}] trading-stop (Full) retCode={rc} retMsg={rm}")
                                except Exception:
                                    pass
                            except Exception as _fe:
                                logger.warning(f"[{sym}] trading-stop Full failed: {_fe}; placing reduce-only TP and SL-only")
                                # Final fallback: place a single reduce-only limit TP and SL-only
                                tp_side = "Sell" if sig_obj.side == "long" else "Buy"
                                try:
                                    ro_resp = bybit.place_reduce_only_limit(sym, tp_side, pos_qty_for_tpsl, float(sig_obj.tp), post_only=True, reduce_only=True)
                                    logger.info(f"[{sym}] reduce-only TP placed: side={tp_side} qty={pos_qty_for_tpsl} price={fmt.format(float(sig_obj.tp))}")
                                except Exception as _re:
                                    logger.warning(f"[{sym}] reduce-only TP failed: {_re}")
                                try:
                                    sl_resp = bybit.set_sl_only(sym, stop_loss=float(sig_obj.sl), qty=pos_qty_for_tpsl)
                                    logger.info(f"[{sym}] SL-only placed via trading-stop (Partial)")
                                except Exception as _se:
                                    logger.warning(f"[{sym}] SL-only placement failed: {_se}")
                                applied_mode = 'reduce_only'
                except Exception:
                    pass

                # Scalp: scale-out disabled — single TP/SL only

                # Best-effort verification without reapplying (to avoid duplicate TP/SL sets)
                    try:
                        checks = 0
                        while checks < 5:
                            await asyncio.sleep(0.8)
                            pos = bybit.get_position(sym)
                            tp_ok = False; sl_ok = False
                            try:
                                tpv = pos.get('takeProfit') if pos else None
                                slv = pos.get('stopLoss') if pos else None
                                tp_ok = (tpv not in (None, '', '0')) and (float(tpv) > 0)
                                sl_ok = (slv not in (None, '', '0')) and (float(slv) > 0)
                            except Exception:
                                tp_ok = sl_ok = False
                            if tp_ok and sl_ok:
                                break
                            checks += 1

                        # If still missing and we used Full mode, one single re-apply try
                        if checks >= 5 and applied_mode == 'full':
                            try:
                                bybit.set_tpsl(sym, take_profit=float(sig_obj.tp), stop_loss=float(sig_obj.sl))
                            except Exception:
                                pass
                        elif checks >= 5 and applied_mode in ('partial', 'reduce_only'):
                            # Attempt a safe fallback path when Partial/ReduceOnly did not reflect on server
                            try:
                                if applied_mode == 'partial':
                                    # Try Full mode once
                                    bybit.set_tpsl(sym, take_profit=float(sig_obj.tp), stop_loss=float(sig_obj.sl))
                                else:
                                    # applied_mode == 'reduce_only': ensure SL-only is present, and add a backup TP limit
                                    tp_side = "Sell" if sig_obj.side == "long" else "Buy"
                                    bybit.place_reduce_only_limit(sym, tp_side, qty, float(sig_obj.tp), post_only=True, reduce_only=True)
                                    bybit.set_sl_only(sym, stop_loss=float(sig_obj.sl), qty=qty)
                            except Exception:
                                pass
                    except Exception:
                        pass

                    # Final guard: ensure protection exists; else emergency close the position
                    try:
                        pos_final = bybit.get_position(sym)
                        tpv = pos_final.get('takeProfit') if isinstance(pos_final, dict) else None
                        slv = pos_final.get('stopLoss') if isinstance(pos_final, dict) else None
                        tp_present = (tpv not in (None, '', '0')) and (float(tpv) > 0)
                        sl_present = (slv not in (None, '', '0')) and (float(slv) > 0)
                        # Also verify order-book protections
                        qty_step = float(m.get('qty_step', 0.001)) if 'm' in locals() else 0.001
                        from position_mgr import round_step as _rstep
                        qty_for_orders = _rstep(pos_qty_for_tpsl if 'pos_qty_for_tpsl' in locals() else qty, qty_step)
                        tp_order_ok = False; sl_cond_ok = False
                        try:
                            orders = bybit.get_open_orders(sym) or []
                            tp_side = "Sell" if str(getattr(sig_obj, 'side','long')).lower() == 'long' else "Buy"
                            # To close a long, we Sell; to close a short, we Buy
                            sl_side = "Sell" if str(getattr(sig_obj, 'side','long')).lower() == 'long' else "Buy"
                            tol = tick_size * 2.01
                            for od in orders:
                                try:
                                    ro = od.get('reduceOnly')
                                    if (ro is True) or (str(ro).lower() == 'true') or (str(ro) == '1'):
                                        p = None
                                        if od.get('orderType') == 'Limit' and od.get('side') == tp_side:
                                            p = float(od.get('price')) if od.get('price') not in (None, '', '0') else None
                                            if p is not None and abs(p - float(sig_obj.tp)) <= tol:
                                                tp_order_ok = True
                                        # Conditional stop orders may appear as Market/Stop/StopMarket with triggerPrice
                                        trig = od.get('triggerPrice')
                                        if od.get('orderType') in ('Market','Stop','StopMarket') and trig not in (None, '', '0'):
                                            if od.get('side') == sl_side:
                                                pp = float(trig)
                                                if abs(pp - float(sig_obj.sl)) <= tol:
                                                    sl_cond_ok = True
                                except Exception:
                                    continue
                        except Exception:
                            pass
                        # Attempt to add missing protections
                        if not tp_present and not tp_order_ok:
                            try:
                                bybit.place_reduce_only_limit(sym, tp_side, qty_for_orders, float(sig_obj.tp), post_only=True, reduce_only=True)
                                tp_order_ok = True
                            except Exception:
                                # PostOnly rejection fallback: nudge by 2 ticks away from price
                                try:
                                    adj = float(sig_obj.tp) + (2.0 * tick_size) if tp_side == 'Sell' else float(sig_obj.tp) - (2.0 * tick_size)
                                    bybit.place_reduce_only_limit(sym, tp_side, qty_for_orders, float(adj), post_only=True, reduce_only=True)
                                    tp_order_ok = True
                                except Exception:
                                    tp_order_ok = False
                        if not sl_present and not sl_cond_ok:
                            try:
                                bybit.set_sl_only(sym, stop_loss=float(sig_obj.sl), qty=qty_for_orders)
                                # place conditional stop as visible backup on order book
                                bybit.place_conditional_stop(sym, sl_side, float(sig_obj.sl), qty_for_orders, reduce_only=True)
                                sl_cond_ok = True
                            except Exception:
                                sl_cond_ok = False
                        # Evaluate final protection status
                        if not ((tp_present or tp_order_ok) and (sl_present or sl_cond_ok)):
                            # Emergency close to prevent unprotected exposure
                            try:
                                logger.critical(f"[{sym}|id={exec_id}] Protections not confirmed — closing. tp_present={tp_present} tp_order_ok={tp_order_ok} sl_present={sl_present} sl_cond_ok={sl_cond_ok}")
                            except Exception:
                                pass
                            try:
                                em_pos = bybit.get_position(sym)
                                em_qty = float(em_pos.get('size') or 0.0) if isinstance(em_pos, dict) else 0.0
                                if em_qty <= 0.0:
                                    em_qty = qty_for_orders
                                emergency_side = "Sell" if sig_obj.side == "long" else "Buy"
                                close_result = bybit.place_market(sym, emergency_side, em_qty, reduce_only=True)
                                try:
                                    bybit.cancel_all_orders(sym)
                                except Exception:
                                    pass
                                logger.critical(f"[{sym}|id={exec_id}] Emergency position closure executed due to missing TP/SL protections: {close_result}")
                                if self.tg:
                                    await self.tg.send_message(f"🚨 EMERGENCY CLOSURE: {sym} {sig_obj.side.upper()} (id={exec_id}) closed — TP/SL protections not confirmed")
                            except Exception as close_error:
                                logger.critical(f"[{sym}|id={exec_id}] CRITICAL: Failed to close unprotected Scalp position: {close_error}")
                                if self.tg:
                                    await self.tg.send_message(f"🆘 CRITICAL: {sym} (id={exec_id}) position UNPROTECTED — TP/SL failed and emergency close failed")
                            try:
                                self._scalp_last_exec_reason[sym] = 'tpsl_error'
                            except Exception:
                                pass
                            return False
                    except Exception:
                        pass

                    # TP/SL confirmation audit log (for faster ops review)
                    try:
                        # Refresh read-back to include any server-side rounding
                        pos_chk = bybit.get_position(sym)
                        tpc = pos_chk.get('takeProfit') if isinstance(pos_chk, dict) else None
                        slc = pos_chk.get('stopLoss') if isinstance(pos_chk, dict) else None
                        # Format using symbol tick precision
                        try:
                            import decimal as _dec
                            d_step = _dec.Decimal(str(tick_size))
                            decs = -d_step.as_tuple().exponent if d_step.as_tuple().exponent < 0 else 4
                        except Exception:
                            decs = 4
                        fmt = f"{{:.{decs}f}}"
                        # Order-level verification: ensure reduce-only TP limit and conditional SL exist
                        tp_order_ok = False; sl_cond_ok = False
                        try:
                            orders = bybit.get_open_orders(sym) or []
                            try:
                                logger.info(f"[{sym}|id={exec_id}] Open orders fetched: count={len(orders)}")
                            except Exception:
                                pass
                            # For long, TP is Sell; for short, TP is Buy
                            side_str = str(getattr(sig_obj, 'side','long')).lower()
                            tp_side = "Sell" if side_str == 'long' else "Buy"
                            # For SL conditional stop (to close), long=>Sell, short=>Buy
                            sl_side = "Sell" if side_str == 'long' else "Buy"
                            # Round qty to step
                            try:
                                from position_mgr import round_step as _rstep
                                qty_step = float(m.get('qty_step', 0.001)) if 'm' in locals() else 0.001
                                qty_orders = _rstep(pos_qty_for_tpsl if 'pos_qty_for_tpsl' in locals() else qty, qty_step)
                            except Exception:
                                qty_orders = pos_qty_for_tpsl if 'pos_qty_for_tpsl' in locals() else qty
                            # Consider price near our TP (within 2 ticks)
                            tol = tick_size * 2.01
                            for od in orders:
                                try:
                                    ro = od.get('reduceOnly')
                                    if (ro is True) or (str(ro).lower() == 'true') or (str(ro) == '1'):
                                        # TP limit order
                                        if od.get('orderType') == 'Limit' and od.get('side') == tp_side:
                                            p = float(od.get('price')) if od.get('price') not in (None, '', '0') else None
                                            if p is not None and abs(p - float(sig_obj.tp)) <= tol:
                                                tp_order_ok = True
                                        # Conditional stop-market (SL)
                                        trig = od.get('triggerPrice')
                                        if od.get('orderType') in ('Market','Stop','StopMarket') and trig not in (None, '', '0') and od.get('side') == sl_side:
                                            pp = float(trig)
                                            if abs(pp - float(sig_obj.sl)) <= tol:
                                                sl_cond_ok = True
                                except Exception:
                                    continue
                            # If position-level TP not present or order-level TP missing, place fallback reduce-only TP
                            if (tpc in (None, '', '0')) or (not tp_order_ok):
                                # Iteratively adjust TP price away from market until accepted (bounded attempts)
                                max_tp_attempts = 6
                                tp_price = float(sig_obj.tp)
                                tp_order_id = None
                                for i in range(max_tp_attempts):
                                    try:
                                        ro_resp = bybit.place_reduce_only_limit(sym, tp_side, qty_orders, float(tp_price), post_only=True, reduce_only=True)
                                        tp_order_ok = True
                                        try:
                                            tp_order_id = (ro_resp.get('result', {}) or {}).get('orderId') if isinstance(ro_resp, dict) else None
                                        except Exception:
                                            tp_order_id = None
                                        try:
                                            logger.info(f"[{sym}|id={exec_id}] Reduce-only TP placed (attempt {i+1}/{max_tp_attempts}) price={fmt.format(tp_price)} orderId={tp_order_id}")
                                        except Exception:
                                            pass
                                        break
                                    except Exception as _re:
                                        # Move price further away to satisfy PostOnly/precision constraints
                                        step = float((i+1) * 2.0 * tick_size)
                                        tp_price = float(tp_price + step) if tp_side == 'Sell' else float(tp_price - step)
                                        try:
                                            logger.warning(f"[{sym}|id={exec_id}] Reduce-only TP failed (attempt {i+1}): {_re}; adjust→{fmt.format(tp_price)}")
                                        except Exception:
                                            pass
                                # Persist last attempted TP
                                try:
                                    sig_obj.tp = float(tp_price)
                                except Exception:
                                    pass
                            # If position-level SL not present and no conditional stop, place fallback conditional stop
                            if (slc in (None, '', '0')) and (not sl_cond_ok):
                                try:
                                    bybit.set_sl_only(sym, stop_loss=float(sig_obj.sl), qty=qty_orders)
                                    try:
                                        logger.info(f"[{sym}|id={exec_id}] SL-only via trading-stop placed @ {fmt.format(float(sig_obj.sl))}")
                                    except Exception:
                                        pass
                                except Exception as _se1:
                                    try:
                                        logger.warning(f"[{sym}|id={exec_id}] SL-only via trading-stop failed: {_se1}")
                                    except Exception:
                                        pass
                                # Conditional stop fallback with iterative price backing off
                                max_sl_attempts = 6
                                sl_price = float(sig_obj.sl)
                                for i in range(max_sl_attempts):
                                    try:
                                        cs_resp = bybit.place_conditional_stop(sym, sl_side, float(sl_price), qty_orders, reduce_only=True)
                                        sl_cond_ok = True
                                        sl_order_id = None
                                        try:
                                            sl_order_id = (cs_resp.get('result', {}) or {}).get('orderId') if isinstance(cs_resp, dict) else None
                                        except Exception:
                                            sl_order_id = None
                                        try:
                                            logger.info(f"[{sym}|id={exec_id}] Conditional SL placed (attempt {i+1}/{max_sl_attempts}) trigger={fmt.format(sl_price)} orderId={sl_order_id}")
                                        except Exception:
                                            pass
                                        break
                                    except Exception as _se2:
                                        step = float((i+1) * 2.0 * tick_size)
                                        # For long (close Sell), push SL lower; for short (close Buy), push SL higher
                                        sl_price = float(sl_price - step) if sl_side == 'Sell' else float(sl_price + step)
                                        try:
                                            logger.warning(f"[{sym}|id={exec_id}] Conditional SL failed (attempt {i+1}): {_se2}; adjust→{fmt.format(sl_price)}")
                                        except Exception:
                                            pass
                                # Persist last attempted SL
                                try:
                                    sig_obj.sl = float(sl_price)
                                except Exception:
                                    pass
                            # Refresh state with a short wait for server-side propagation
                            try:
                                await asyncio.sleep(0.5)
                            except Exception:
                                pass
                            pos_chk = bybit.get_position(sym)
                            tpc = pos_chk.get('takeProfit') if isinstance(pos_chk, dict) else tpc
                            slc = pos_chk.get('stopLoss') if isinstance(pos_chk, dict) else slc
                        except Exception:
                            pass
                        # Confirm again after order-level fallback
                        if ((tpc not in (None, '', '0')) or tp_order_ok) and ((slc not in (None, '', '0')) or sl_cond_ok):
                            logger.info(f"[{sym}|id={exec_id}] TP/SL confirmed: TP={fmt.format(float(tpc))} SL={fmt.format(float(slc))} (mode={applied_mode})")
                            try:
                                if self.tg:
                                    await self.tg.send_message(
                                        f"✅ Scalp TP/SL confirmed: {sym} (id={exec_id})\nTP={fmt.format(float(tpc))} SL={fmt.format(float(slc))} (mode={applied_mode})"
                                    )
                            except Exception:
                                pass
                        else:
                            # If we created orderIds, include them in diagnostic detail even if position fields are empty
                            try:
                                tp_id = locals().get('tp_order_id', None)
                                sl_id = locals().get('sl_order_id', None)
                                logger.warning(f"[{sym}|id={exec_id}] Protections not yet confirmed at position-level: posTP={tpc not in (None,'','0')} posSL={slc not in (None,'','0')} tp_order_ok={tp_order_ok} sl_cond_ok={sl_cond_ok} tp_id={tp_id} sl_id={sl_id}")
                            except Exception:
                                pass
                        
                    except Exception:
                        pass

                    # Mark TP/SL applied to help future idempotency checks
                    try:
                        from time import time as _now
                        self._tpsl_applied[sym] = float(_now())
                    except Exception:
                        pass
                except Exception as _tpsle:
                    try:
                        self._scalp_last_exec_reason[sym] = 'tpsl_error'
                        if not hasattr(self, '_scalp_last_exec_detail'):
                            self._scalp_last_exec_detail = {}
                        self._scalp_last_exec_detail[sym] = f"{type(_tpsle).__name__}: {_tpsle}"
                    except Exception:
                        pass
                    return False
            # Update book
                # Capture last signal features for this symbol (to carry qscore into Position)
                try:
                    if not hasattr(self, '_last_signal_features'):
                        self._last_signal_features = {}
                    # Merge any meta into existing features to preserve qscore/components
                    prev_feats = dict(self._last_signal_features.get(sym, {}) or {})
                    meta_feats = dict(getattr(sig_obj, 'meta', {}) or {})
                    merged = {**prev_feats, **meta_feats}
                    self._last_signal_features[sym] = merged
                except Exception:
                    pass
                # Update book with qscore if available
                self.book.positions[sym] = Position(
                    sig_obj.side,
                    qty,
                    entry=actual_entry,
                    sl=float(sig_obj.sl),
                    tp=float(sig_obj.tp),
                    entry_time=datetime.now(),
                    strategy_name='scalp',
                    ml_score=float(ml_score or 0.0),
                    qscore=float(((getattr(self, '_last_signal_features', {}) or {}).get(sym, {}) or {}).get('qscore', 0.0) or 0.0)
                )
            # Telegram notify
                try:
                    if self.tg:
                        # Compute target RR for visibility
                        try:
                            Rv = (actual_entry - float(sig_obj.sl)) if sig_obj.side == 'long' else (float(sig_obj.sl) - actual_entry)
                            target_rr = max(0.1, (float(sig_obj.tp) - actual_entry)/Rv) if sig_obj.side=='long' else max(0.1, (actual_entry - float(sig_obj.tp))/Rv)
                        except Exception:
                            target_rr = 1.5
                        # Format prices with symbol-specific decimals
                        try:
                            import decimal as _dec
                            d_step = _dec.Decimal(str(tick_size))
                            decs = -d_step.as_tuple().exponent if d_step.as_tuple().exponent < 0 else 4
                        except Exception:
                            decs = 4
                        fmt = f"{{:.{decs}f}}"
                        # Include Qscore if available
                        try:
                            qv = float(((getattr(self, '_last_signal_features', {}) or {}).get(sym, {}) or {}).get('qscore', 0.0) or 0.0)
                        except Exception:
                            qv = 0.0
                        msg = (
                            f"🩳 Scalp: Executing {sym} {sig_obj.side.upper()} (id={exec_id})\n\n"
                            f"Entry: {fmt.format(actual_entry)}\n"
                            f"Stop Loss: {fmt.format(float(sig_obj.sl))}\n"
                            f"Take Profit: {fmt.format(float(sig_obj.tp))}\n"
                            f"Quantity: {qty}\n"
                            f"Target RR: {target_rr:.2f}\n"
                            f"Q: {qv:.1f}"
                        )
                        await self.tg.send_message(msg)
                except Exception:
                    pass
                # Successful execution
                return True
        except Exception as e:
            logger.info(f"[{sym}] Scalp stream execute error: {e}")
            return False


    # --- Scalp feature builder for ML/phantom ---
    def _build_scalp_features(self, df: pd.DataFrame, sc_meta: dict | None = None,
                              vol_level: str | None = None, cluster_id: int | None = None) -> dict:
        """Compute Scalp ML features from a given dataframe window.

        Returns a dict matching ml_scorer_scalp._prepare_features keys.
        """
        sc_meta = sc_meta or {}
        out = {}
        try:
            tail = df.tail(50).copy()
            close = tail['close']
            open_ = tail['open']
            high = tail['high']
            low = tail['low']

            # ATR(14) and ATR pct of price
            prev_close = close.shift()
            tr = np.maximum(high - low, np.maximum((high - prev_close).abs(), (low - prev_close).abs()))
            atr = float(tr.rolling(14).mean().iloc[-1]) if len(tr) >= 14 else float(tr.iloc[-1])
            price = float(close.iloc[-1]) if len(close) else 0.0
            atr_pct = float((atr / max(1e-9, price)) * 100.0) if price else 0.0
            out['atr_pct'] = max(0.0, atr_pct)

            # Bollinger bands width percent (20)
            if len(close) >= 20:
                ma = close.rolling(20).mean()
                sd = close.rolling(20).std()
                upper = ma + 2 * sd
                lower = ma - 2 * sd
                bb_width = float((upper.iloc[-1] - lower.iloc[-1]))
                bb_width_pct = float(bb_width / max(1e-9, price))
            else:
                bb_width_pct = 0.0
            out['bb_width_pct'] = max(0.0, bb_width_pct)

            # Impulse ratio: |close change| / ATR
            if len(close) >= 2 and atr > 0:
                impulse_ratio = float(abs(close.iloc[-1] - close.iloc[-2]) / atr)
            else:
                impulse_ratio = 0.0
            out['impulse_ratio'] = max(0.0, impulse_ratio)

            # EMA slopes (fast=20, slow=50) over 10 bars as percent of price
            def _ema(s: pd.Series, n: int) -> pd.Series:
                return s.ewm(span=n, adjust=False).mean()

            if len(close) >= 20:
                ema_fast = _ema(close, 20)
                if len(ema_fast) >= 11:
                    slope_fast = float((ema_fast.iloc[-1] - ema_fast.iloc[-11]) / 10.0)
                    out['ema_slope_fast'] = float((slope_fast / max(1e-9, price)) * 100.0)
                else:
                    out['ema_slope_fast'] = 0.0
            else:
                out['ema_slope_fast'] = 0.0
            if len(close) >= 50:
                ema_slow = _ema(close, 50)
                if len(ema_slow) >= 11:
                    slope_slow = float((ema_slow.iloc[-1] - ema_slow.iloc[-11]) / 10.0)
                    out['ema_slope_slow'] = float((slope_slow / max(1e-9, price)) * 100.0)
                else:
                    out['ema_slope_slow'] = 0.0
            else:
                out['ema_slope_slow'] = 0.0

            # Wick ratios (last candle)
            if len(tail) >= 1:
                o = float(open_.iloc[-1]); c = float(close.iloc[-1]); h = float(high.iloc[-1]); l = float(low.iloc[-1])
                rng = max(1e-9, h - l)
                upper_wick = h - max(o, c)
                lower_wick = min(o, c) - l
                out['upper_wick_ratio'] = float(max(0.0, upper_wick / rng))
                out['lower_wick_ratio'] = float(max(0.0, lower_wick / rng))
            else:
                out['upper_wick_ratio'] = 0.0
                out['lower_wick_ratio'] = 0.0

            # Volume ratio
            vol = tail['volume']
            if len(vol) >= 20 and vol.iloc[-20:-1].mean() > 0:
                out['volume_ratio'] = float(vol.iloc[-1] / max(1e-9, vol.rolling(20).mean().iloc[-1]))
            else:
                out['volume_ratio'] = float(sc_meta.get('vol_ratio', 1.0))

            # Body ratio and 15m EMA direction for gating
            try:
                if len(tail) >= 1:
                    o = float(open_.iloc[-1]); c = float(close.iloc[-1]); h = float(high.iloc[-1]); l = float(low.iloc[-1])
                    rng = max(1e-9, h - l)
                    out['body_ratio'] = float(abs(c - o) / rng)
                    out['body_sign'] = 'up' if (c - o) > 0 else 'down' if (c - o) < 0 else 'flat'
                else:
                    out['body_ratio'] = 0.0
                    out['body_sign'] = 'flat'
            except Exception:
                out['body_ratio'] = 0.0
                out['body_sign'] = 'flat'

            # 15m EMA alignment direction (up/down/none)
            try:
                sym = sc_meta.get('symbol') if isinstance(sc_meta, dict) else None
                df15 = self.frames.get(sym) if hasattr(self, 'frames') and sym in getattr(self, 'frames', {}) else None
                if df15 is not None and not df15.empty and len(df15['close']) >= 50:
                    e20 = df15['close'].ewm(span=20, adjust=False).mean()
                    e50 = df15['close'].ewm(span=50, adjust=False).mean()
                    out['ema_dir_15m'] = 'up' if float(e20.iloc[-1]) > float(e50.iloc[-1]) else ('down' if float(e20.iloc[-1]) < float(e50.iloc[-1]) else 'none')
                else:
                    out['ema_dir_15m'] = 'none'
            except Exception:
                out['ema_dir_15m'] = 'none'

            # VWAP distance in ATR
            try:
                if 'dist_vwap_atr' in sc_meta and sc_meta.get('dist_vwap_atr') is not None:
                    out['vwap_dist_atr'] = float(sc_meta.get('dist_vwap_atr'))
                else:
                    tp = (tail['high'] + tail['low'] + tail['close']) / 3.0
                    vwap = (tp * tail['volume']).rolling(20).sum() / tail['volume'].rolling(20).sum()
                    out['vwap_dist_atr'] = float(abs(close.iloc[-1] - vwap.iloc[-1]) / max(1e-9, atr)) if len(vwap.dropna()) else 0.0
            except Exception:
                out['vwap_dist_atr'] = float(sc_meta.get('dist_vwap_atr', 0.0) or 0.0)

            # Session label for mapping (scorer maps to 0..3)
            try:
                hour = pd.Timestamp.utcnow().hour
                if 0 <= hour < 8:
                    out['session'] = 'asian'
                elif 8 <= hour < 16:
                    out['session'] = 'european'
                elif 16 <= hour < 24:
                    out['session'] = 'us'
                else:
                    out['session'] = 'off_hours'
            except Exception:
                out['session'] = 'off_hours'

            # Volatility regime & cluster
            out['volatility_regime'] = vol_level if isinstance(vol_level, str) else 'normal'
            out['symbol_cluster'] = int(cluster_id) if cluster_id is not None else 3

            # --- Additional micro features (v2) ---
            # Short-horizon returns (%)
            try:
                if len(close) >= 2 and close.iloc[-2] != 0:
                    out['ret_1'] = float((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100.0)
                else:
                    out['ret_1'] = 0.0
                if len(close) >= 4 and close.iloc[-4] != 0:
                    out['ret_3'] = float((close.iloc[-1] - close.iloc[-4]) / close.iloc[-4] * 100.0)
                else:
                    out['ret_3'] = 0.0
                if len(close) >= 6 and close.iloc[-6] != 0:
                    out['ret_5'] = float((close.iloc[-1] - close.iloc[-6]) / close.iloc[-6] * 100.0)
                else:
                    out['ret_5'] = 0.0
            except Exception:
                out['ret_1'] = out.get('ret_1', 0.0)
                out['ret_3'] = out.get('ret_3', 0.0)
                out['ret_5'] = out.get('ret_5', 0.0)

            # ATR slope over last 10 bars, scaled to % of price
            try:
                if len(tr) >= 11 and price:
                    atr_prev = float(tr.rolling(14).mean().iloc[-11]) if len(tr) >= 14 else float(tr.iloc[-11])
                    atr_slope = (atr - atr_prev) / 10.0
                    out['atr_slope'] = float((atr_slope / max(1e-9, price)) * 100.0)
                else:
                    out['atr_slope'] = 0.0
            except Exception:
                out['atr_slope'] = out.get('atr_slope', 0.0)

            # Volatility-of-volatility: std of last 10 returns (%)
            try:
                if len(close) >= 11 and price:
                    rets = close.pct_change().tail(10).dropna().values
                    out['vol_of_vol'] = float(np.std(rets) * 100.0)
                else:
                    out['vol_of_vol'] = 0.0
            except Exception:
                out['vol_of_vol'] = out.get('vol_of_vol', 0.0)

            # Round tick distance proxy: distance to nearest .00/.25/.50/.75 (% of price)
            try:
                q = [0.00, 0.25, 0.50, 0.75]
                frac = (price * 100.0) % 1.0  # rough proxy at 0.01 scale
                nearest = min(q, key=lambda x: abs(frac - x)) if isinstance(frac, float) else 0.0
                out['round_tick_dist'] = float(abs(frac - nearest) * 100.0)
            except Exception:
                out['round_tick_dist'] = out.get('round_tick_dist', 0.0)
            return out
        except Exception:
            return {}

    def _compute_qscore_scalp(self, symbol: str, side: str, entry: float, sl: float, tp: float,
                               df3: 'pd.DataFrame' = None, df15: 'pd.DataFrame' = None,
                               sc_feats: dict | None = None) -> tuple[float, dict, list[str]]:
        """Compute Scalp quality score (0–100) with component breakdown.

        Components: mom, pull, micro, htf, sr, risk.
        """
        comp = {}; reasons = []
        try:
            f = sc_feats or {}
            # Ensure target RR is included for ML training quality
            try:
                R = abs(float(entry) - float(sl))
                f['rr_target'] = float(abs(float(tp) - float(entry)) / R) if R > 0 else 0.0
            except Exception:
                f['rr_target'] = f.get('rr_target', 0.0)
            # Momentum: impulse vs ATR + wick direction + volume
            imp = float(f.get('impulse_ratio', 0.0) or 0.0)
            volr = float(f.get('volume_ratio', 1.0) or 1.0)
            upw = float(f.get('upper_wick_ratio', 0.0) or 0.0)
            loww = float(f.get('lower_wick_ratio', 0.0) or 0.0)
            mom = min(100.0, max(0.0, 60.0 * imp + 15.0 * max(0.0, volr - 1.0)))
            # Directional wick bonus
            if side == 'long':
                mom += min(10.0, loww * 20.0)
                if upw > 0.6:
                    reasons.append('mom:long_selling_wick')
            else:
                mom += min(10.0, upw * 20.0)
                if loww > 0.6:
                    reasons.append('mom:short_buying_wick')
            comp['mom'] = max(0.0, min(100.0, mom))

            # Pullback fit: closer to VWAP is better; wider BB up to a point is better
            vwap_da = float(f.get('vwap_dist_atr', 0.0) or 0.0)
            bbw = float(f.get('bb_width_pct', 0.0) or 0.0)
            # Map VWAP distance: 0 ATR→100, 0.8 ATR→0
            vwap_score = max(0.0, min(100.0, (1.0 - (vwap_da / 0.8)) * 100.0))
            # Map BB width: 0%→0, 2%→100 (cap)
            bb_score = max(0.0, min(100.0, (bbw / 0.02) * 100.0)) if bbw >= 0 else 0.0
            comp['pull'] = 0.7 * vwap_score + 0.3 * bb_score

            # Micro structure: 3m sequence of closes
            try:
                micro = 50.0
                if df3 is not None and len(df3) >= 4:
                    tail = df3['close'].tail(4)
                    up_seq = tail.iloc[-1] > tail.iloc[-2] >= tail.iloc[-3]
                    dn_seq = tail.iloc[-1] < tail.iloc[-2] <= tail.iloc[-3]
                    ok = (up_seq if side == 'long' else dn_seq)
                    micro = 100.0 if ok else 30.0
                comp['micro'] = micro
            except Exception:
                comp['micro'] = 50.0

            # HTF nudge from 15m/60m composite
            try:
                comp_htf = 50.0
                compo = self._get_htf_metrics(symbol, df15 if df15 is not None else self.frames.get(symbol))
                ts15 = float(compo.get('ts15', 0.0) or 0.0)
                # Favor longs when ts15 high; shorts when ts15 low (invert)
                if side == 'long':
                    comp_htf = max(0.0, min(100.0, ts15))
                else:
                    comp_htf = max(0.0, min(100.0, 100.0 - ts15))
                comp['htf'] = comp_htf
            except Exception as _he:
                comp['htf'] = 50.0; reasons.append(f'htf:error:{_he}')

            # SR clearance ahead (15m HTF levels)
            try:
                from multi_timeframe_sr import mtf_sr
                dfref = df15 if df15 is not None else self.frames.get(symbol)
                price = float(dfref['close'].iloc[-1]) if dfref is not None and len(dfref) else float(df3['close'].iloc[-1]) if df3 is not None and len(df3) else 0.0
                prev = dfref['close'].shift() if dfref is not None else (df3['close'].shift() if df3 is not None else None)
                high = (dfref['high'] if dfref is not None else (df3['high'] if df3 is not None else None))
                low = (dfref['low'] if dfref is not None else (df3['low'] if df3 is not None else None))
                if prev is not None and high is not None and low is not None:
                    trarr = np.maximum(high - low, np.maximum((high - prev).abs(), (low - prev).abs()))
                    atr14 = float(trarr.rolling(14).mean().iloc[-1]) if len(trarr) >= 14 else float(trarr.iloc[-1])
                else:
                    atr14 = price * 0.005
                vlevels = mtf_sr.get_price_validated_levels(symbol, price)
                if side == 'long':
                    # distance to nearest resistance above price
                    res = [lv for (lv, st, t) in vlevels if t == 'resistance']
                    dist_atr = min((abs(lv - price)/max(1e-9, atr14) for lv in res), default=1.0)
                else:
                    sup = [lv for (lv, st, t) in vlevels if t == 'support']
                    dist_atr = min((abs(lv - price)/max(1e-9, atr14) for lv in sup), default=1.0)
                # Clearance mapping: larger distance to next level = higher score
                # 0.0 ATR -> 0, 0.6 ATR -> 100 (capped)
                comp['sr'] = max(0.0, min(100.0, (dist_atr / 0.60) * 100.0))
            except Exception as _se:
                comp['sr'] = 50.0; reasons.append(f'sr:error:{_se}')

            # Risk geometry: target RR and stop size sanity
            try:
                if side == 'long':
                    R = max(1e-9, entry - sl)
                    rr = max(0.0, (tp - entry) / R)
                else:
                    R = max(1e-9, sl - entry)
                    rr = max(0.0, (entry - tp) / R)
                # Map RR: 1.2→40, 1.5→60, 2.0→85, ≥3→100
                if rr <= 1.2:
                    risk = 40.0
                elif rr <= 1.5:
                    risk = 60.0
                elif rr <= 2.0:
                    risk = 85.0
                else:
                    risk = 100.0
                comp['risk'] = risk
            except Exception:
                comp['risk'] = 60.0

            # Weighted sum using config weights
            try:
                wcfg = (((self.config.get('scalp', {}) or {}).get('rule_mode', {}) or {}).get('weights', {}) or {})
                wm = float(wcfg.get('mom', 0.35)); wp = float(wcfg.get('pull', 0.20)); wmi = float(wcfg.get('micro', 0.15)); wh = float(wcfg.get('htf', 0.10)); ws = float(wcfg.get('sr', 0.10)); wrk = float(wcfg.get('risk', 0.10))
                s = max(1e-9, wm + wp + wmi + wh + ws + wrk); wm/=s; wp/=s; wmi/=s; wh/=s; ws/=s; wrk/=s
            except Exception:
                wm, wp, wmi, wh, ws, wrk = 0.35, 0.20, 0.15, 0.10, 0.10, 0.10
            q = wm*comp.get('mom',50.0) + wp*comp.get('pull',50.0) + wmi*comp.get('micro',50.0) + wh*comp.get('htf',50.0) + ws*comp.get('sr',50.0) + wrk*comp.get('risk',50.0)
            return max(0.0, min(100.0, float(q))), comp, reasons
        except Exception as e:
            return 50.0, {}, [f'scalp_q:error:{e}']

    async def _collect_secondary_stream(self, ws_url: str, timeframe: str, symbols: list[str]):
        """Collect a secondary timeframe (e.g., 3m) for scalp detection."""
        handler = MultiWebSocketHandler(ws_url, self.running)
        # Track the configured TF for diagnostics
        try:
            self._scalp_stream_tf = str(timeframe)
        except Exception:
            self._scalp_stream_tf = None
        topics = [f"{timeframe}.{s}" for s in symbols]
        async for sym, k in handler.multi_kline_stream(topics):
            try:
                ts = int(k.get("start")) if k.get("start") is not None else None
                row = pd.DataFrame(
                    [[float(k["open"]), float(k["high"]), float(k["low"]), float(k["close"]), float(k["volume"])]] ,
                    index=[pd.to_datetime(ts, unit="ms", utc=True) if ts else pd.Timestamp.utcnow()],
                    columns=["open","high","low","close","volume"]
                )
                df = self.frames_3m.get(sym)
                if df is None:
                    df = new_frame()
                # TZ consistency
                if df.index.tz is None and row.index.tz is not None:
                    df.index = df.index.tz_localize('UTC')
                elif df.index.tz is not None and row.index.tz is None:
                    row.index = row.index.tz_localize('UTC')
                # Gap fill if we missed bars while disconnected
                try:
                    if len(df) > 0:
                        last_ts = df.index[-1]
                        try:
                            exp_delta = pd.Timedelta(minutes=int(timeframe))
                        except Exception:
                            exp_delta = pd.Timedelta(minutes=3)
                        if (row.index[0] - last_ts) > (exp_delta + pd.Timedelta(seconds=5)) and self.bybit is not None:
                            try:
                                start_ms = int(((last_ts + pd.Timedelta(seconds=1)).tz_convert('UTC').timestamp() if last_ts.tzinfo else (last_ts + pd.Timedelta(seconds=1)).timestamp()) * 1000)
                            except Exception:
                                start_ms = None
                            end_ms = int((row.index[0].tz_convert('UTC').timestamp() if row.index[0].tzinfo else row.index[0].timestamp()) * 1000)
                            if start_ms is not None and end_ms:
                                try:
                                    kl = self.bybit.get_klines(sym, timeframe, limit=200, start=start_ms, end=end_ms) or []
                                    fill_rows = []
                                    for kr in kl:
                                        try:
                                            ts2 = int(kr[0]) if isinstance(kr, (list, tuple)) else int(kr.get('start'))
                                            idx2 = pd.to_datetime(ts2, unit='ms', utc=True)
                                            o2 = float(kr[1] if isinstance(kr, (list, tuple)) else kr.get('open'))
                                            h2 = float(kr[2] if isinstance(kr, (list, tuple)) else kr.get('high'))
                                            l2 = float(kr[3] if isinstance(kr, (list, tuple)) else kr.get('low'))
                                            c2 = float(kr[4] if isinstance(kr, (list, tuple)) else kr.get('close'))
                                            v2 = float(kr[5] if isinstance(kr, (list, tuple)) else kr.get('volume'))
                                            fill_rows.append((idx2, o2, h2, l2, c2, v2))
                                        except Exception:
                                            continue
                                    if fill_rows:
                                        fdf = pd.DataFrame([(o, h, l, c, v) for (_, o, h, l, c, v) in fill_rows],
                                                           index=[ix for (ix, *_r) in fill_rows],
                                                           columns=["open","high","low","close","volume"]).sort_index()
                                        df = pd.concat([df, fdf])
                                except Exception:
                                    pass
                except Exception:
                    pass
                # Append current row
                df.loc[row.index[0]] = row.iloc[0]
                df.sort_index(inplace=True)
                self.frames_3m[sym] = df.tail(1000)
                # Update shadow simulator with 3m close to improve shadow trade lifecycle
                try:
                    from shadow_trade_simulator import get_shadow_tracker
                    last_close_3m = float(row.iloc[0]['close'])
                    get_shadow_tracker().update_prices(sym, last_close_3m)
                except Exception:
                    pass
                # Persist the latest 3m candle row (offload to DB writer if available)
                try:
                    if getattr(self, '_db_queue', None) is not None:
                        await self._db_queue.put(('3m', sym, row))
                    else:
                        self.storage.save_candles_3m(sym, row)
                except Exception as e:
                    logger.debug(f"3m candle persist error for {sym}: {e}")

                # Update any active Scalp phantoms for this symbol using the latest bar extremes
                try:
                    if SCALP_AVAILABLE and get_scalp_phantom_tracker is not None:
                        scpt = get_scalp_phantom_tracker()
                        scpt.update_scalp_phantom_prices(sym, float(row['close'].iloc[0]), df=self.frames_3m.get(sym))
                except Exception:
                    pass

                # On 3m bar close, attempt phantom-only scalp detection for NONE regime
                try:
                    confirm = bool(k.get("confirm", False))
                except Exception:
                    confirm = False
                if not confirm:
                    continue
                # Count confirmed 3m bar for diagnostics
                try:
                    self._scalp_stats['confirms'] = self._scalp_stats.get('confirms', 0) + 1
                except Exception:
                    pass
                # Mark last confirm per-symbol for health/fallback
                try:
                    self._scalp_last_confirm[sym] = row.index[0]
                except Exception:
                    pass

                # Advance Trend microstructure on each confirmed 3m bar (no 15m wait)
                try:
                    df15 = self.frames.get(sym)
                    if df15 is not None and not df15.empty and getattr(self, '_detect_trend_signal', None) is not None and getattr(self, '_trend_settings', None) is not None:
                        # Run micro-step by invoking pullback detection (it consults 3m frames internally)
                        _ = self._detect_trend_signal(df15.copy(), self._trend_settings, sym)
                except Exception as _te:
                    logger.debug(f"[{sym}] Trend micro-step error on 3m: {_te}")

                # BE move monitoring on 3m close: detect TP1 hit and move SL→BE promptly
                try:
                    if hasattr(self, '_scaleout') and sym in getattr(self, '_scaleout', {}) and sym in self.book.positions:
                        so = self._scaleout.get(sym) or {}
                        if not bool(so.get('be_moved')) and bool(so.get('move_be', True)):
                            side = str(so.get('side',''))
                            tp1 = float(so.get('tp1', 0.0))
                            entry_px = float(so.get('entry', 0.0))
                            df3c = self.frames_3m.get(sym)
                            last_px = float(df3c['close'].iloc[-1]) if df3c is not None and not df3c.empty else None
                            hit = (last_px is not None) and ((side == 'long' and last_px >= tp1) or (side == 'short' and last_px <= tp1))
                            if hit:
                                # TP1 reached: notify once
                                if not bool(so.get('tp1_notified', False)):
                                    try:
                                        if self.tg:
                                            qty1 = so.get('qty1')
                                            msg_tp1 = f"🎯 TP1 reached: {sym} price {last_px:.4f} {'≥' if side=='long' else '≤'} TP1 {tp1:.4f}"
                                            if qty1:
                                                msg_tp1 += f" | qty1={float(qty1):.4f}"
                                            await self.tg.send_message(msg_tp1)
                                        # Append to event log
                                        try:
                                            evts = self.shared.get('trend_events')
                                            if isinstance(evts, list):
                                                from datetime import datetime as _dt
                                                evts.append({'ts': _dt.utcnow().isoformat()+'Z', 'symbol': sym, 'text': msg_tp1})
                                                if len(evts) > 60:
                                                    del evts[:len(evts)-60]
                                        except Exception:
                                            pass
                                    except Exception:
                                        pass
                                    # Mark tp1 hit
                                    self._scaleout[sym]['tp1_notified'] = True
                                    self._scaleout[sym]['tp1_hit'] = True
                                    try:
                                        from datetime import datetime as _dt
                                        self._scaleout[sym]['tp1_time'] = _dt.utcnow()
                                    except Exception:
                                        pass

                                # Move stop to break-even (entry price) — only for non-recovered OR if reconcile toggle is on
                                try:
                                    tr_exec = ((self.config.get('trend', {}) or {}).get('exec', {}) or {})
                                    reconcile = bool(tr_exec.get('recovery_reconcile_be', False))
                                    is_recovered = bool(so.get('recovered', False))
                                    if (not is_recovered) or reconcile:
                                        self.bybit.set_tpsl(sym, stop_loss=float(entry_px))
                                        self._scaleout[sym]['be_moved'] = True
                                        # Update in-memory book position SL
                                        try:
                                            if sym in self.book.positions:
                                                self.book.positions[sym].sl = float(entry_px)
                                        except Exception:
                                            pass
                                        # Notify BE move
                                        if self.tg:
                                            await self.tg.send_message(f"🛡️ BE Move: {sym} SL→BE at {entry_px:.4f} after TP1 hit")
                                        # Append to event log
                                        try:
                                            evts = self.shared.get('trend_events')
                                            if isinstance(evts, list):
                                                from datetime import datetime as _dt
                                                evts.append({'ts': _dt.utcnow().isoformat()+'Z', 'symbol': sym, 'text': f"BE Move: SL→BE at {entry_px:.4f} after TP1"})
                                                if len(evts) > 60:
                                                    del evts[:len(evts)-60]
                                        except Exception:
                                            pass
                                except Exception:
                                    pass
                except Exception as _e:
                    logger.debug(f"[{sym}] 3m BE monitor error: {_e}")

                # Trend-only mode: skip Scalp analysis/logging on 3m stream but keep Trend micro-step above
                try:
                    if getattr(self, '_trend_only', False):
                        continue
                except Exception:
                    pass

                # Analysis trace (start)
                try:
                    logger.debug(f"[{sym}] 🩳 Scalp(3m) analysis start: tf={timeframe}m")
                except Exception:
                    pass

                # Guard: scalper enabled and modules available
                try:
                    scalp_cfg = self.config.get('scalp', {}) if hasattr(self, 'config') else {}
                    use_scalp = bool(scalp_cfg.get('enabled', False) and SCALP_AVAILABLE)
                except Exception:
                    use_scalp = False
                if not use_scalp or detect_scalp_signal is None:
                    continue

                # Regime independence: if scalp.independent=true, do NOT gate by regime/volatility
                vol_level = 'normal'
                try:
                    scalp_cfg = self.config.get('scalp', {}) if hasattr(self, 'config') else {}
                    scalp_independent = bool(scalp_cfg.get('independent', False))
                except Exception:
                    scalp_independent = False
                if not scalp_independent:
                    # Legacy behavior: block extreme volatility
                    vol_level = 'unknown'
                    if 'frames' in dir(self) and sym in self.frames and not self.frames[sym].empty and ENHANCED_ML_AVAILABLE:
                        try:
                            analysis = get_enhanced_market_regime(self.frames[sym].tail(200), sym)
                            vol_level = analysis.volatility_level
                            if vol_level == 'extreme':
                                logger.debug(f"[{sym}] 🩳 Scalp(3m) skipped: volatility={vol_level}")
                                try:
                                    logger.info(f"[{sym}] 🧮 Scalp decision final: blocked (reason=volatility_extreme)")
                                except Exception:
                                    pass
                                # Do not continue; proceed to Trend analysis
                        except Exception:
                            pass

                # Per-symbol cooldown: configurable bars between scalp records
                last_ts = self._scalp_cooldown.get(sym)
                bar_ts = row.index[0]
                try:
                    s_cfg = self.config.get('scalp', {}) if hasattr(self, 'config') else {}
                    cooldown_bars = int(s_cfg.get('cooldown_bars', 8))
                except Exception:
                    cooldown_bars = 8
                # Convert timeframe string to seconds
                try:
                    bar_seconds = int(timeframe) * 60
                except Exception:
                    bar_seconds = 180
                # Cooldown disabled: do not block Scalp detections by cooldown between bars
                # (kept structure for future toggle; intentionally no early return here)

                # Run scalper on 3m df (independent of regime)
                # Build ScalpSettings, applying phantom-only relaxed thresholds if enabled
                try:
                    sc_settings = ScalpSettings()
                    try:
                        s_cfg = self.config.get('scalp', {}) if hasattr(self, 'config') else {}
                        exp = s_cfg.get('explore', {})
                        # Allow RR override via config (defaults to 2.0 in ScalpSettings)
                        try:
                            if 'rr' in s_cfg:
                                sc_settings.rr = float(s_cfg.get('rr'))
                        except Exception:
                            pass
                        # Apply minimum 1R distance if configured
                        try:
                            if 'min_r_pct' in s_cfg:
                                sc_settings.min_r_pct = float(s_cfg.get('min_r_pct'))
                        except Exception:
                            pass
                        if exp.get('relax_enabled', False):
                            # Apply relaxed thresholds for phantom exploration only
                            sc_settings.vwap_dist_atr_max = float(exp.get('vwap_dist_atr_max', sc_settings.vwap_dist_atr_max))
                            sc_settings.min_bb_width_pct = float(exp.get('min_bb_width_pct', sc_settings.min_bb_width_pct))
                            sc_settings.vol_ratio_min = float(exp.get('vol_ratio_min', sc_settings.vol_ratio_min))
                            # Allow tuning of additional params if provided
                            if 'wick_ratio_min' in exp:
                                sc_settings.wick_ratio_min = float(exp.get('wick_ratio_min', sc_settings.wick_ratio_min))
                            if 'vwap_window' in exp:
                                sc_settings.vwap_window = int(exp.get('vwap_window', sc_settings.vwap_window))
                            if 'ema_fast' in exp:
                                sc_settings.ema_fast = int(exp.get('ema_fast', sc_settings.ema_fast))
                            if 'ema_slow' in exp:
                                sc_settings.ema_slow = int(exp.get('ema_slow', sc_settings.ema_slow))
                            if 'atr_len' in exp:
                                sc_settings.atr_len = int(exp.get('atr_len', sc_settings.atr_len))
                            if 'orb_enabled' in exp:
                                sc_settings.orb_enabled = bool(exp.get('orb_enabled', sc_settings.orb_enabled))
                    except Exception:
                        pass
                    # Apply adaptive flow relax on top of config (phantom-only)
                    try:
                        if hasattr(self, 'flow_controller') and self.flow_controller and self.flow_controller.enabled:
                            sc_settings = self.flow_controller.adjust_scalp(sc_settings)
                    except Exception:
                        pass
                    # Trace adjusted settings
                    try:
                        logger.debug(
                            f"[{sym}] 🩳 Scalp settings: vwap_max={sc_settings.vwap_dist_atr_max:.2f}, "
                            f"bb_min={sc_settings.min_bb_width_pct:.2f}, vol_ratio_min={sc_settings.vol_ratio_min:.2f}"
                        )
                    except Exception:
                        pass
                    df3_for_sig = self.frames_3m[sym].copy()
                    sc_sig = detect_scalp_signal(df3_for_sig, sc_settings, sym)
                except Exception as e:
                    logger.debug(f"[{sym}] Scalp(3m) detection error: {e}")
                    sc_sig = None
                if not sc_sig:
                    # Heartbeat: emit compact no-signal line at INFO (configurable)
                    try:
                        log_cfg = (self.config.get('scalp', {}) or {}).get('logging', {}) or {}
                        hb_enabled = bool(log_cfg.get('heartbeat', False))
                        decision_only = bool(log_cfg.get('decision_only', False))
                    except Exception:
                        hb_enabled = False; decision_only = False
                    # Optional: compact gate probe to explain no-signal
                    try:
                        if hb_enabled and not decision_only and df3_for_sig is not None and len(df3_for_sig) >= max(sc_settings.vwap_window, 50):
                            import numpy as _np
                            _c = df3_for_sig['close']; _h = df3_for_sig['high']; _l = df3_for_sig['low']; _v = df3_for_sig['volume']
                            _atr = (_h - _l).rolling(sc_settings.atr_len).mean()
                            _ema_f = _c.ewm(span=sc_settings.ema_fast, adjust=False).mean()
                            _ema_s = _c.ewm(span=sc_settings.ema_slow, adjust=False).mean()
                            _tp = (_h + _l + _c) / 3
                            _pv = _tp * _v.clip(lower=0.0)
                            _vwap = _pv.rolling(sc_settings.vwap_window).sum() / _v.rolling(sc_settings.vwap_window).sum().replace(0, _np.nan)
                            _std20 = _c.rolling(20).std(); _bbw = (_std20 / _c).fillna(0)
                            _bbw_pct = float((_bbw <= float(_bbw.iloc[-1])).mean()) if len(_bbw) else 0.0
                            _vol20 = _v.rolling(20).mean(); _vol_ratio = float((_v.iloc[-1] / _vol20.iloc[-1]) if _vol20.iloc[-1] > 0 else 1.0)
                            _rng = max(1e-9, float(_h.iloc[-1] - _l.iloc[-1]))
                            _o = float(df3_for_sig['open'].iloc[-1]); _cl = float(_c.iloc[-1])
                            _upper_w = max(0.0, float(_h.iloc[-1]) - max(_cl, _o)) / _rng
                            _lower_w = max(0.0, min(_cl, _o) - float(_l.iloc[-1])) / _rng
                            _ema_up = bool(_cl > float(_ema_f.iloc[-1]) > float(_ema_s.iloc[-1]))
                            _ema_dn = bool(_cl < float(_ema_f.iloc[-1]) < float(_ema_s.iloc[-1]))
                            _cur_vwap = float(_vwap.iloc[-1]) if _vwap.notna().iloc[-1] else _cl
                            _cur_atr = float(_atr.iloc[-1]) if (_atr.iloc[-1] is not None and _atr.iloc[-1] > 0) else _rng
                            _dist_vwap_atr = abs(_cl - _cur_vwap) / max(1e-9, _cur_atr)
                            _orb_ok = True
                            if len(df3_for_sig) >= 40:
                                _first_high = float(_h.iloc[:20].max()); _first_low = float(_l.iloc[:20].min())
                                if _ema_up and _cl <= _first_high:
                                    _orb_ok = False
                                if _ema_dn and _cl >= _first_low:
                                    _orb_ok = False
                            logger.info(
                                f"[{sym}] 🩳 Scalp: no signal (up={_ema_up} dn={_ema_dn} bbw={_bbw_pct:.2f}/{sc_settings.min_bb_width_pct:.2f} "
                                f"vol={_vol_ratio:.2f}/{sc_settings.vol_ratio_min:.2f} wickL={_lower_w:.2f} wickU={_upper_w:.2f}/{sc_settings.wick_ratio_min:.2f} "
                                f"vwap={_dist_vwap_atr:.2f}/{sc_settings.vwap_dist_atr_max:.2f} orb={_orb_ok})"
                            )
                            try:
                                reasons = []
                                if not (_ema_up or _ema_dn):
                                    reasons.append('ema_misaligned')
                                if _bbw_pct < float(sc_settings.min_bb_width_pct):
                                    reasons.append('bbw_low')
                                if _vol_ratio < float(sc_settings.vol_ratio_min):
                                    reasons.append('vol_low')
                                if _dist_vwap_atr > float(sc_settings.vwap_dist_atr_max):
                                    reasons.append('vwap_far')
                                if not _orb_ok:
                                    reasons.append('orb_block')
                                if not reasons:
                                    reasons.append('filters_unmet')
                                # Update per-symbol Scalp state flags
                                try:
                                    self._scalp_symbol_state[sym] = {
                                        'mom': bool(_ema_up or _ema_dn),
                                        'pull': bool((_bbw_pct >= float(sc_settings.min_bb_width_pct)) and (max(_upper_w, _lower_w) >= float(sc_settings.wick_ratio_min))),
                                        'vwap': bool(_dist_vwap_atr <= float(sc_settings.vwap_dist_atr_max)),
                                        # Keep last q_ge_thr flag value until a new signal evaluates
                                        **({'q_ge_thr': self._scalp_symbol_state.get(sym, {}).get('q_ge_thr', False)} if hasattr(self, '_scalp_symbol_state') else {})
                                    }
                                except Exception:
                                    pass
                                # Reasons histogram
                                try:
                                    for r in reasons:
                                        self._scalp_reasons[r] = self._scalp_reasons.get(r, 0) + 1
                                except Exception:
                                    pass
                                logger.info(f"[{sym}] 🧮 Scalp heartbeat: decision=no_signal reasons={','.join(reasons)}")
                            except Exception:
                                pass
                    except Exception:
                        pass
                    continue
                # Suppress non-high-ML chatter; only log minimal detection
                try:
                    logger.debug(f"[{sym}] 🩳 Scalp signal det: {getattr(sc_sig, 'side','?').upper()} @ {float(getattr(sc_sig,'entry',0.0)):.4f}")
                except Exception:
                    pass

                # Ensure a decision-final line is always emitted for visibility
                _scalp_decision_logged = False

                # Cache latest scalp detection for promotion timing
                try:
                    self._scalp_last_signal[sym] = {
                        'ts': pd.Timestamp.utcnow(),
                        'side': getattr(sc_sig, 'side', ''),
                        'entry': float(getattr(sc_sig, 'entry', 0.0) or 0.0),
                        'sl': float(getattr(sc_sig, 'sl', 0.0) or 0.0),
                        'tp': float(getattr(sc_sig, 'tp', 0.0) or 0.0)
                    }
                except Exception:
                    pass

                # Immediate execution for ALL Scalp detections (disabled by default; use high-ML path instead)
                try:
                    exec_all = False
                    try:
                        exec_all = bool((((self.config.get('scalp', {}) or {}).get('exec', {}) or {}).get('execute_all', False)))
                    except Exception:
                        exec_all = False
                    if exec_all:
                        # Build features for record
                        _df_src_hi = None
                        try:
                            _df_src_hi = self.frames_3m.get(sym)
                            if _df_src_hi is None or getattr(_df_src_hi, 'empty', True):
                                _df_src_hi = self.frames.get(sym)
                            if _df_src_hi is None or getattr(_df_src_hi, 'empty', True):
                                _df_src_hi = df
                        except Exception:
                            _df_src_hi = df
                        sc_feats_hi = self._build_scalp_features(_df_src_hi, getattr(sc_sig, 'meta', {}) or {}, vol_level, None)
                        # Attach HTF composite metrics to features for ML training
                        try:
                            comp = self._get_htf_metrics(sym, self.frames.get(sym))
                            sc_feats_hi['ts15'] = float(comp.get('ts15', 0.0)); sc_feats_hi['ts60'] = float(comp.get('ts60', 0.0))
                            sc_feats_hi['rc15'] = float(comp.get('rc15', 0.0)); sc_feats_hi['rc60'] = float(comp.get('rc60', 0.0))
                        except Exception:
                            pass
                        # Score ML for learning even though we execute immediately
                        ml_s_immediate = 0.0
                        try:
                            from ml_scorer_scalp import get_scalp_scorer
                            _scorer = get_scalp_scorer()
                            ml_s_immediate, _ = _scorer.score_signal({'side': sc_sig.side, 'entry': sc_sig.entry, 'sl': sc_sig.sl, 'tp': sc_sig.tp}, sc_feats_hi)
                        except Exception:
                            ml_s_immediate = 0.0
                        if sym in self.book.positions:
                            logger.info(f"[{sym}] 🛑 Scalp execute blocked: reason=position_exists")
                        else:
                            executed = False
                            try:
                                executed = await self._execute_scalp_trade(sym, sc_sig, ml_score=float(ml_s_immediate or 0.0))
                            except Exception as _ee:
                                logger.info(f"[{sym}] Scalp execute error: {_ee}")
                                executed = False
                            # Record executed phantom for learning
                            try:
                                scpt = get_scalp_phantom_tracker()
                                # Optionally cancel any pre-existing active scalp phantom to avoid duplicate tracking
                                try:
                                    cancel_on_hi = bool(((self.config.get('scalp', {}) or {}).get('exec', {}) or {}).get('cancel_active_on_high_ml', True))
                                except Exception:
                                    cancel_on_hi = True
                                if bool(executed) and cancel_on_hi:
                                    try:
                                        scpt.cancel_active(sym)
                                    except Exception:
                                        pass
                                # Attach Qscore snapshot to features before recording
                                try:
                                    _qs = self._compute_qscore_scalp(sym, sc_sig.side, float(sc_sig.entry), float(sc_sig.sl), float(sc_sig.tp), df3=_df_src_hi, df15=self.frames.get(sym), sc_feats=sc_feats_hi)
                                    sc_feats_hi['qscore'] = float(_qs[0]); sc_feats_hi['qscore_components'] = dict(_qs[1]); sc_feats_hi['qscore_reasons'] = list(_qs[2])
                                except Exception:
                                    pass
                                scpt.record_scalp_signal(
                                    sym,
                                    {'side': sc_sig.side, 'entry': sc_sig.entry, 'sl': sc_sig.sl, 'tp': sc_sig.tp},
                                    float(ml_s_immediate or 0.0),
                                    bool(executed),
                                    sc_feats_hi
                                )
                            except Exception:
                                pass
                            if executed:
                                logger.info(f"[{sym}] 🧮 Scalp decision final: exec_scalp (reason=immediate)")
                                continue
                            else:
                                logger.info(f"[{sym}] 🛑 Scalp immediate execute blocked: reason=exec_guard")
                except Exception:
                    pass


                # EARLY debug force-accept: record immediately to prove acceptance path
                try:
                    fa_cfg = (self.config.get('scalp', {}).get('debug', {}) or {}) if hasattr(self, 'config') else {}
                    if bool(fa_cfg.get('force_accept', False)):
                        try:
                            scpt = get_scalp_phantom_tracker()
                            # Choose a source df safely without boolean evaluation on DataFrames
                            _df_src = None
                            try:
                                _df_src = self.frames_3m.get(sym)
                                if _df_src is None or getattr(_df_src, 'empty', True):
                                    _df_src = self.frames.get(sym)
                                if _df_src is None or getattr(_df_src, 'empty', True):
                                    _df_src = df
                            except Exception:
                                _df_src = df
                            sc_feats_early = self._build_scalp_features(_df_src, getattr(sc_sig, 'meta', {}) or {}, vol_level, None)
                            try:
                                comp = self._get_htf_metrics(sym, self.frames.get(sym))
                                sc_feats_early['ts15'] = float(comp.get('ts15', 0.0)); sc_feats_early['ts60'] = float(comp.get('ts60', 0.0))
                                sc_feats_early['rc15'] = float(comp.get('rc15', 0.0)); sc_feats_early['rc60'] = float(comp.get('rc60', 0.0))
                            except Exception:
                                pass
                            # Compute Scalp ML score for visibility (heuristic if model not ready)
                            ml_s_early = 0.0
                            try:
                                from ml_scorer_scalp import get_scalp_scorer
                                _scorer = get_scalp_scorer()
                                ml_s_early, _ = _scorer.score_signal({'side': sc_sig.side, 'entry': sc_sig.entry, 'sl': sc_sig.sl, 'tp': sc_sig.tp}, sc_feats_early)
                            except Exception:
                                ml_s_early = 0.0
                            scpt.record_scalp_signal(
                                sym,
                                {'side': sc_sig.side, 'entry': sc_sig.entry, 'sl': sc_sig.sl, 'tp': sc_sig.tp},
                                float(ml_s_early or 0.0),
                                False,
                                (lambda _f: (
                                    (lambda _qs: (_f.update({'qscore': float(_qs[0]), 'qscore_components': dict(_qs[1]), 'qscore_reasons': list(_qs[2])}), _f)[1])(
                                        self._compute_qscore_scalp(sym, sc_sig.side, float(sc_sig.entry), float(sc_sig.sl), float(sc_sig.tp), df3=_df_src, df15=self.frames.get(sym), sc_feats=_f)
                                    ),
                                    _f
                                )[1])(sc_feats_early)
                            )
                            try:
                                self._scalp_stats['signals'] = self._scalp_stats.get('signals', 0) + 1
                            except Exception:
                                pass
                            logger.info(f"[{sym}] 👻 Phantom-only (Scalp 3m none): {sc_sig.side.upper()} @ {sc_sig.entry:.4f}")
                            logger.info(f"[{sym}] 🧮 Scalp decision final: phantom (reason=debug_force_early)")
                            _scalp_decision_logged = True
                            # Skip the rest to avoid double-recording
                            continue
                        except Exception as _fae:
                            logger.warning(f"[{sym}] Scalp debug force-accept error: {_fae}")
                except Exception:
                    pass

                # Dedup via Redis (phantom dedup scope) — optional
                dedup_ok = True
                try:
                    s_cfg = self.config.get('scalp', {}) if hasattr(self, 'config') else {}
                    if bool(s_cfg.get('dedup_enabled', False)):
                        scpt = _safe_get_scalp_phantom_tracker()
                        r = scpt.redis_client
                        if r is not None:
                            key = f"{sym}:{sc_sig.side}:{round(float(sc_sig.entry),6)}:{round(float(sc_sig.sl),6)}:{round(float(sc_sig.tp),6)}"
                            # TTL configurable
                            try:
                                dedup_hours = int(s_cfg.get('dedup_hours', 8))
                            except Exception:
                                dedup_hours = 8
                            if r.exists(f"phantom:dedup:scalp:{key}"):
                                dedup_ok = False
                            else:
                                r.setex(f"phantom:dedup:scalp:{key}", max(1, dedup_hours) * 3600, '1')
                except Exception:
                    pass
                if not dedup_ok:
                    logger.debug(f"[{sym}] 🩳 Scalp(3m) dedup: duplicate signal skipped")
                    try:
                        self._scalp_stats['dedup_skips'] = self._scalp_stats.get('dedup_skips', 0) + 1
                    except Exception:
                        pass
                    try:
                        logger.info(f"[{sym}] 🧮 Scalp decision final: blocked (reason=dedup)")
                        _scalp_decision_logged = True
                        # Telegram: notify dedup skip for visibility (only when dedup is enabled)
                        try:
                            if self.tg and bool(s_cfg.get('dedup_enabled', False)):
                                await self.tg.send_message(f"🛑 Scalp: [{sym}] dedup skip — phantom suppressed")
                        except Exception:
                            pass
                    except Exception:
                        pass
                    continue

                # One active phantom per symbol guard
                # Allow multiple active phantoms per symbol for Scalp learning (no active_symbol block)
                try:
                    if scpt.has_active(sym):
                        logger.debug(f"[{sym}] 🧮 Scalp decision note: active phantom exists — allowing additional phantom for learning")
                except Exception:
                    pass

                # Record/Execute Scalp: build full feature set first
                # Choose a source df safely without boolean evaluation on DataFrames
                _df_src2 = None
                try:
                    _df_src2 = self.frames_3m.get(sym)
                    if _df_src2 is None or getattr(_df_src2, 'empty', True):
                        _df_src2 = self.frames.get(sym)
                    if _df_src2 is None or getattr(_df_src2, 'empty', True):
                        _df_src2 = df
                except Exception:
                    _df_src2 = df
                sc_feats = self._build_scalp_features(_df_src2, getattr(sc_sig, 'meta', {}) or {}, vol_level, None)
                try:
                    comp = self._get_htf_metrics(sym, self.frames.get(sym))
                    sc_feats['ts15'] = float(comp.get('ts15', 0.0)); sc_feats['ts60'] = float(comp.get('ts60', 0.0))
                    sc_feats['rc15'] = float(comp.get('rc15', 0.0)); sc_feats['rc60'] = float(comp.get('rc60', 0.0))
                except Exception:
                    pass
                try:
                    _qs = self._compute_qscore_scalp(sym, sc_sig.side, float(sc_sig.entry), float(sc_sig.sl), float(sc_sig.tp), df3=_df_src2, df15=self.frames.get(sym), sc_feats=sc_feats)
                    sc_feats['qscore'] = float(_qs[0]); sc_feats['qscore_components'] = dict(_qs[1]); sc_feats['qscore_reasons'] = list(_qs[2])
                except Exception:
                    pass
                sc_feats['routing'] = 'none'
                # Ensure hourly per-symbol pacing vars exist before any High-ML early exit uses them
                try:
                    hb = self.config.get('phantom', {}).get('hourly_symbol_budget', {}) or {}
                    sc_limit = int(hb.get('scalp', 4))
                except Exception:
                    sc_limit = 4
                if not hasattr(self, '_scalp_budget'):
                    self._scalp_budget = {}
                now_ts = pd.Timestamp.utcnow().timestamp()
                blist = [ts for ts in self._scalp_budget.get(sym, []) if (now_ts - ts) < 3600]
                self._scalp_budget[sym] = blist
                try:
                    scpt = _safe_get_scalp_phantom_tracker()
                    # Compute Scalp ML score if scorer is available
                    ml_s = 0.0
                    try:
                        from ml_scorer_scalp import get_scalp_scorer
                        _scorer = get_scalp_scorer()
                        ml_s, _ = _scorer.score_signal({'side': sc_sig.side, 'entry': sc_sig.entry, 'sl': sc_sig.sl, 'tp': sc_sig.tp}, sc_feats)
                    except Exception:
                        ml_s = 0.0
                    # Qscore-gated execution (primary) OR High-ML override (optional)
                    try:
                        e_cfg = (self.config.get('scalp', {}) or {}).get('exec', {})
                        allow_hi = bool(e_cfg.get('allow_stream_high_ml', False))
                        hi_thr = float(e_cfg.get('high_ml_force', 92))
                    except Exception:
                        allow_hi = False; hi_thr = 92.0
                    # Scalp Qscore execution gate
                    exec_enabled = True
                    exec_thr = 60.0
                    try:
                        # Default to enabled; allow config to explicitly disable
                        exec_enabled = bool(((self.config.get('scalp', {}) or {}).get('exec', {}) or {}).get('enabled', True))
                        exec_thr = float((((self.config.get('scalp', {}) or {}).get('rule_mode', {}) or {}).get('execute_q_min', 60)))
                    except Exception:
                        exec_enabled = True
                        exec_thr = 60.0
                    # Session gating for Scalp execution (skip when qscore_only=true)
                    try:
                        sc_cfg = (self.config.get('scalp', {}) or {})
                        q_only_sc = bool(((sc_cfg.get('exec') or {}).get('qscore_only', True)))
                    except Exception:
                        q_only_sc = True
                    # No session restrictions for Scalp (always allow)
                    session_ok = True
                    # EV-based threshold bump for Scalp
                    try:
                        from ml_scorer_scalp import get_scalp_scorer
                        sc_ev_scorer = get_scalp_scorer()
                        ev_thr_sc = float(sc_ev_scorer.get_ev_threshold(sc_feats))
                        hi_thr = max(hi_thr, ev_thr_sc)
                    except Exception:
                        pass
                    # Attempt Qscore execution first (if enabled)
                    did_exec = False
                    exec_reason = None
                    try:
                        # Update per-symbol state for Q≥thr
                        try:
                            qv = float(sc_feats.get('qscore', 0.0))
                            self._scalp_symbol_state[sym] = {**(self._scalp_symbol_state.get(sym, {}) or {}), 'q_ge_thr': bool(qv >= exec_thr)}
                        except Exception:
                            pass
                        # Use learned threshold if available
                        q_val = float(sc_feats.get('qscore', 0.0))
                        try:
                            ctx = {'session': self._session_label(), 'volatility_regime': sc_feats.get('volatility_regime', 'global')}
                            rm_sc = (((self.config.get('scalp', {}) or {}).get('rule_mode', {}) or {}))
                            use_adapter = bool(rm_sc.get('adapter_enabled', True))
                            if use_adapter and hasattr(self, '_qadapt_scalp') and self._qadapt_scalp:
                                exec_thr = float(self._qadapt_scalp.get_threshold(ctx, floor=60.0, ceiling=95.0, default=exec_thr))
                        except Exception:
                            pass
                        if exec_enabled and session_ok and q_val >= exec_thr:
                            # Execution-only hard gates for Scalp
                            try:
                                ok_g, reasons = self._scalp_hard_gates_pass(sym, sc_sig.side, sc_feats)
                            except Exception as _ge:
                                ok_g, reasons = False, [f"gates:error:{_ge}"]
                            if not ok_g:
                                # Block execution but record phantom for learning
                                exec_reason = 'hard_gates'
                                try:
                                    if self.tg:
                                        await self.tg.send_message(
                                            f"🛑 Scalp EXEC hard-gate blocked: {sym} reasons={','.join(reasons)} — phantom recorded\n"
                                            f"Q={float(sc_feats.get('qscore',0.0)):.1f} (≥ {exec_thr:.0f})"
                                        )
                                except Exception:
                                    pass
                                try:
                                    scpt.record_scalp_signal(
                                        sym,
                                        {'side': sc_sig.side, 'entry': sc_sig.entry, 'sl': sc_sig.sl, 'tp': sc_sig.tp},
                                        float(ml_s or 0.0),
                                        False,
                                        sc_feats
                                    )
                                    _scalp_decision_logged = True
                                except Exception:
                                    pass
                                continue
                            if sym in self.book.positions:
                                exec_reason = 'position_exists'
                            else:
                                # Daily exec cap
                                if not hasattr(self, '_scalp_exec_counter'):
                                    self._scalp_exec_counter = {'day': None, 'count': 0}
                                from datetime import datetime as _dt
                                day_str = _dt.utcnow().strftime('%Y%m%d')
                                if self._scalp_exec_counter['day'] != day_str:
                                    self._scalp_exec_counter = {'day': day_str, 'count': 0}
                                daily_cap = int(e_cfg.get('daily_cap', 0) or 0)
                                # Ignore daily_cap for Scalp (no execution gating by cap)
                                # Risk override
                                old_risk = None
                                try:
                                    old_risk = self.sizer.risk.risk_percent
                                    self.sizer.risk.risk_percent = float(e_cfg.get('risk_percent', old_risk or 1.0))
                                except Exception:
                                    pass
                                # Pre-exec notify
                                try:
                                    if self.tg:
                                        # Create an execution id and stash it on the signal/meta and features
                                        try:
                                            import uuid as _uuid
                                            exec_id = getattr(sc_sig, 'meta', {}).get('exec_id') if isinstance(getattr(sc_sig, 'meta', {}), dict) else None
                                            if not exec_id:
                                                exec_id = _uuid.uuid4().hex[:8]
                                                if isinstance(getattr(sc_sig, 'meta', {}), dict):
                                                    sc_sig.meta['exec_id'] = exec_id
                                            sc_feats['exec_id'] = exec_id
                                        except Exception:
                                            exec_id = 'unknown'
                                        comps = sc_feats.get('qscore_components', {}) or {}
                                        comp_line = f"MOM={comps.get('mom',0):.0f} PULL={comps.get('pull',0):.0f} Micro={comps.get('micro',0):.0f} HTF={comps.get('htf',0):.0f} SR={comps.get('sr',0):.0f} Risk={comps.get('risk',0):.0f}"
                                        # Stash features to carry Q into Position and close-notify
                                        try:
                                            if not hasattr(self, '_last_signal_features'):
                                                self._last_signal_features = {}
                                            self._last_signal_features[sym] = dict(sc_feats)
                                        except Exception:
                                            pass
                                        await self.tg.send_message(f"🟢 Scalp EXECUTE: {sym} {sc_sig.side.upper()} Q={float(sc_feats.get('qscore',0.0)):.1f} (≥ {exec_thr:.0f}) (id={exec_id})\n{comp_line}")
                                except Exception:
                                    pass
                                try:
                                    # Prefer new signature with exec_id; fallback to legacy signature if unavailable
                                    try:
                                        did_exec = await self._execute_scalp_trade(sym, sc_sig, ml_score=float(ml_s or 0.0), exec_id=exec_id)
                                    except TypeError:
                                        did_exec = await self._execute_scalp_trade(sym, sc_sig, ml_score=float(ml_s or 0.0))
                                finally:
                                    try:
                                        if old_risk is not None:
                                            self.sizer.risk.risk_percent = old_risk
                                    except Exception:
                                        pass
                                    if did_exec:
                                        self._scalp_exec_counter['count'] += 1
                                        exec_reason = 'qgate'
                                        # On success, record executed mirror and short-circuit the rest of the loop
                                        try:
                                            scpt.record_scalp_signal(
                                                sym,
                                                {'side': sc_sig.side, 'entry': sc_sig.entry, 'sl': sc_sig.sl, 'tp': sc_sig.tp},
                                                float(ml_s or 0.0),
                                                True,
                                                sc_feats
                                            )
                                            _scalp_decision_logged = True
                                            self._scalp_cooldown[sym] = bar_ts
                                            blist.append(now_ts)
                                            self._scalp_budget[sym] = blist
                                            try:
                                                logger.info(f"[{sym}|id={exec_id}] 🧮 Scalp decision final: exec_scalp (reason=qscore {float(sc_feats.get('qscore',0.0)):.1f}>={exec_thr:.0f})")
                                            except Exception:
                                                pass
                                            continue
                                        except Exception:
                                            pass
                                    else:
                                        # Pull detailed reason from executor if set
                                        try:
                                            exec_reason = (getattr(self, '_scalp_last_exec_reason', {}) or {}).get(sym, exec_reason)
                                        except Exception:
                                            pass
                    except Exception as _xe:
                        did_exec = False
                        try:
                            exec_id = (getattr(sc_sig, 'meta', {}) or {}).get('exec_id') if isinstance(getattr(sc_sig,'meta',{}), dict) else None
                        except Exception:
                            exec_id = None
                        try:
                            logger.warning(f"[{sym}|id={exec_id or 'n/a'}] Scalp execute exception: {_xe}")
                            if not hasattr(self, '_scalp_last_exec_reason'):
                                self._scalp_last_exec_reason = {}
                            self._scalp_last_exec_reason[sym] = 'exec_exception'
                            if not hasattr(self, '_scalp_last_exec_detail'):
                                self._scalp_last_exec_detail = {}
                            self._scalp_last_exec_detail[sym] = f"{type(_xe).__name__}: {_xe}"
                        except Exception:
                            pass
                    # If Q gate passed but blocked, notify
                    try:
                        if (not did_exec) and exec_enabled and session_ok and float(sc_feats.get('qscore', 0.0)) >= exec_thr:
                            if self.tg:
                                # Prefer explicit exec_reason, then last_exec_reason from executor; include detail if available
                                r = exec_reason or getattr(self, '_scalp_last_exec_reason', {}).get(sym) or 'exec_guard'
                                det = None
                                try:
                                    det = (getattr(self, '_scalp_last_exec_detail', {}) or {}).get(sym)
                                except Exception:
                                    det = None
                                comps = sc_feats.get('qscore_components', {}) or {}
                                comp_line = f"MOM={comps.get('mom',0):.0f} PULL={comps.get('pull',0):.0f} Micro={comps.get('micro',0):.0f} HTF={comps.get('htf',0):.0f} SR={comps.get('sr',0):.0f} Risk={comps.get('risk',0):.0f}"
                                try:
                                    ex_id = (getattr(sc_sig, 'meta', {}) or {}).get('exec_id') if isinstance(getattr(sc_sig,'meta',{}), dict) else None
                                except Exception:
                                    ex_id = None
                                extra = f"; detail={det}" if det else ""
                                await self.tg.send_message(f"🛑 Scalp: [{sym}] EXEC blocked (reason={r}{extra}) — phantom recorded (id={ex_id or 'n/a'})\nQ={float(sc_feats.get('qscore',0.0)):.1f} (≥ {exec_thr:.0f})\n{comp_line}")
                    except Exception:
                        pass

                    if allow_hi and float(ml_s or 0.0) >= hi_thr:
                        try:
                            if sym in self.book.positions:
                                logger.info(f"[{sym}] 🛑 Scalp High-ML blocked: reason=position_exists")
                            else:
                                # Regime gate for scalp execution (volatility + micro-trend alignment)
                                vol_ok = str(sc_feats.get('volatility_regime','normal')) in ('normal','high')
                                fast = float(sc_feats.get('ema_slope_fast', 0.0) or 0.0)
                                slow = float(sc_feats.get('ema_slope_slow', 0.0) or 0.0)
                                side = str(sc_sig.side)
                                micro_ok = (side == 'long' and fast >= 0.0 and slow >= 0.0) or (side == 'short' and fast <= 0.0 and slow <= 0.0)
                                if not (vol_ok and micro_ok):
                                    logger.info(f"[{sym}] 🛑 Scalp execution blocked by regime gate (vol={sc_feats.get('volatility_regime')} fast={fast:.2f} slow={slow:.2f} side={side})")
                                    executed = False
                                else:
                                    # Generate/propagate exec_id for high-ML path as well
                                    try:
                                        import uuid as _uuid
                                        exec_id_h = (getattr(sc_sig, 'meta', {}) or {}).get('exec_id') if isinstance(getattr(sc_sig,'meta',{}), dict) else None
                                        if not exec_id_h:
                                            exec_id_h = _uuid.uuid4().hex[:8]
                                            if isinstance(getattr(sc_sig, 'meta', {}), dict):
                                                sc_sig.meta['exec_id'] = exec_id_h
                                    except Exception:
                                        exec_id_h = None
                                    try:
                                        executed = await self._execute_scalp_trade(sym, sc_sig, ml_score=float(ml_s or 0.0), exec_id=exec_id_h)
                                    except TypeError:
                                        executed = await self._execute_scalp_trade(sym, sc_sig, ml_score=float(ml_s or 0.0))
                                # Optionally cancel any pre-existing active scalp phantom to avoid duplicate tracking
                                try:
                                    cancel_on_hi = bool(((self.config.get('scalp', {}) or {}).get('exec', {}) or {}).get('cancel_active_on_high_ml', True))
                                except Exception:
                                    cancel_on_hi = True
                                if bool(executed) and cancel_on_hi:
                                    try:
                                        scpt.cancel_active(sym)
                                    except Exception:
                                        pass
                                # Record as executed mirror for learning
                                scpt.record_scalp_signal(
                                    sym,
                                    {'side': sc_sig.side, 'entry': sc_sig.entry, 'sl': sc_sig.sl, 'tp': sc_sig.tp},
                                    float(ml_s or 0.0),
                                    bool(executed),
                                    sc_feats
                                )
                                if executed:
                                    logger.info(f"[{sym}] 🧮 Scalp decision final: exec_scalp (reason=ml_extreme {float(ml_s or 0.0):.1f}>={hi_thr:.0f})")
                                    # Skip rest of gating for this detection
                                    _scalp_decision_logged = True
                                    self._scalp_cooldown[sym] = bar_ts
                                    blist.append(now_ts)
                                    self._scalp_budget[sym] = blist
                                    continue
                        except Exception as _ee:
                            logger.info(f"[{sym}] Scalp High-ML override error: {_ee}")
                    # Enforce per-strategy hourly per-symbol budget for scalp
                    hb = self.config.get('phantom', {}).get('hourly_symbol_budget', {}) or {}
                    sc_limit = int(hb.get('scalp', sc_limit))
                    # now_ts and blist already initialized above; refresh list for safety
                    now_ts = pd.Timestamp.utcnow().timestamp()
                    blist = [ts for ts in self._scalp_budget.get(sym, []) if (now_ts - ts) < 3600]
                    self._scalp_budget[sym] = blist
                    sc_remaining = sc_limit - len(blist)
                    # Daily cap (none) per strategy for scalp
                    daily_ok = True
                    n_key = None
                    try:
                        if scpt.redis_client is not None:
                            day = pd.Timestamp.utcnow().strftime('%Y%m%d')
                            caps_cfg = (self.config.get('phantom', {}).get('caps', {}) or {}).get('scalp', {})
                            none_cap = int(caps_cfg.get('none', 200))
                            n_key = f"phantom:daily:none_count:{day}:scalp"
                            n_val = int(scpt.redis_client.get(n_key) or 0)
                            daily_ok = n_val < none_cap
                    except Exception:
                        pass
                    # Visibility into decision inputs
                    # Decision context (reduce to DEBUG to avoid log noise)
                    try:
                        logger.debug(f"[{sym}] 🩳 Scalp decision context: dedup_ok={dedup_ok} hourly_remaining={max(0, sc_remaining)} daily_ok={daily_ok}")
                    except Exception:
                        pass
                    # Force-accept disabled in high-ML-only mode
                    force_accept = False
                    try:
                        force_accept = bool(((self.config.get('scalp', {}).get('debug', {}) or {}).get('force_accept', False)))
                    except Exception:
                        force_accept = False
                    if force_accept:
                        # Check phantom Q-score threshold even for force_accept
                        ph_min = float(((self.config.get('scalp', {}) or {}).get('rule_mode', {}) or {}).get('phantom_q_min', 20))
                        q_score = float(sc_feats.get('qscore', 0.0))
                        if q_score < ph_min:
                            logger.info(f"[{sym}] 📦 Scalp phantom REJECTED (force_accept): Q={q_score:.1f} < {ph_min:.1f}")
                            continue
                        _rec = scpt.record_scalp_signal(
                            sym,
                            {'side': sc_sig.side, 'entry': sc_sig.entry, 'sl': sc_sig.sl, 'tp': sc_sig.tp},
                            float(ml_s or 0.0),
                            False,
                            sc_feats
                        )
                        if _rec is not None:
                            try:
                                self._scalp_stats['signals'] = self._scalp_stats.get('signals', 0) + 1
                            except Exception:
                                pass
                            logger.info(f"[{sym}] 👻 Phantom-only (Scalp 3m none): {sc_sig.side.upper()} @ {sc_sig.entry:.4f}")
                            try:
                                logger.info(f"[{sym}] 🧮 Scalp decision final: phantom (reason=debug_force)")
                                _scalp_decision_logged = True
                            except Exception:
                                pass
                            self._scalp_cooldown[sym] = bar_ts
                            blist.append(now_ts)
                            self._scalp_budget[sym] = blist
                        else:
                            logger.info(f"[{sym}] 🛑 Scalp phantom (none route debug) dropped by regime gate (tracker)")
                    elif sc_remaining > 0 and daily_ok:
                        # Check phantom Q-score threshold (cut extreme noise)
                        ph_min = float(((self.config.get('scalp', {}) or {}).get('rule_mode', {}) or {}).get('phantom_q_min', 20))
                        q_score = float(sc_feats.get('qscore', 0.0))
                        if q_score < ph_min:
                            logger.info(f"[{sym}] 📦 Scalp phantom REJECTED: Q={q_score:.1f} < {ph_min:.1f}")
                            continue
                        logger.info(f"[{sym}] 📝 Scalp phantom ACCEPTED: Q={q_score:.1f} ≥ {ph_min:.1f}")
                        _rec = scpt.record_scalp_signal(
                            sym,
                            {'side': sc_sig.side, 'entry': sc_sig.entry, 'sl': sc_sig.sl, 'tp': sc_sig.tp},
                            float(ml_s or 0.0),
                            False,
                            sc_feats
                        )
                        if _rec is not None:
                            try:
                                self._scalp_stats['signals'] = self._scalp_stats.get('signals', 0) + 1
                            except Exception:
                                pass
                            # Increment Flow Controller accepted counter for scalp (phantom-only pacing)
                            try:
                                if hasattr(self, 'flow_controller') and self.flow_controller and self.flow_controller.enabled:
                                    self.flow_controller.increment_accepted('scalp', 1)
                            except Exception:
                                pass
                            # Increment daily none count for scalp
                            try:
                                if scpt.redis_client is not None and n_key is not None:
                                    scpt.redis_client.incr(n_key)
                            except Exception:
                                pass
                            logger.info(f"[{sym}] 👻 Phantom-only (Scalp 3m none): {sc_sig.side.upper()} @ {sc_sig.entry:.4f}")
                            # Ensure Telegram receives an open notification immediately (dedup-safe)
                            try:
                                if self.tg and hasattr(self, '_notify_scalp_phantom'):
                                    await self._notify_scalp_phantom(_rec)
                            except Exception:
                                pass
                            try:
                                # Explain why not executed: budgets or below ML override (still phantom)
                                reason = 'unknown'
                                try:
                                    if not daily_ok:
                                        reason = 'daily_cap'
                                    elif sc_remaining <= 0:
                                        reason = 'hourly_budget'
                                    else:
                                        reason = f"ml<thr {float(ml_s or 0.0):.1f}<{hi_thr:.0f}"
                                except Exception:
                                    pass
                                logger.info(f"[{sym}] 🧮 Scalp decision final: phantom (reason={reason})")
                                _scalp_decision_logged = True
                                # Telegram notify for blocked reasons leading to phantom (non-q reasons), include Qscore
                                try:
                                    if self.tg and reason in ('daily_cap','hourly_budget'):
                                        comps = sc_feats.get('qscore_components', {}) or {}
                                        comp_line = f"MOM={comps.get('mom',0):.0f} PULL={comps.get('pull',0):.0f} Micro={comps.get('micro',0):.0f} HTF={comps.get('htf',0):.0f} SR={comps.get('sr',0):.0f} Risk={comps.get('risk',0):.0f}"
                                        await self.tg.send_message(f"🛑 Scalp: [{sym}] EXEC blocked (reason={reason}) — phantom recorded\nQ={float(sc_feats.get('qscore',0.0)):.1f}\n{comp_line}")
                                except Exception:
                                    pass
                            except Exception:
                                pass
                            self._scalp_cooldown[sym] = bar_ts
                            blist.append(now_ts)
                            self._scalp_budget[sym] = blist
                        else:
                            logger.debug(f"[{sym}] 🛑 Scalp phantom (none route) dropped by regime gate (tracker)")
                            _scalp_decision_logged = True
                        # Shadow execute Scalp if ML is trained and score ≥ threshold
                        try:
                            scorer = get_scalp_scorer() if get_scalp_scorer is not None else None
                            if getattr(scorer, 'is_ml_ready', False):
                                score, _ = scorer.score_signal({'side': sc_sig.side}, sc_feats)
                                thr_sc = getattr(scorer, 'min_score', 75)
                                if score >= thr_sc:
                                    from shadow_trade_simulator import get_shadow_tracker
                                    get_shadow_tracker().record_shadow_trade(
                                        strategy='scalp',
                                        symbol=sym,
                                        side=sc_sig.side,
                                        entry=float(sc_sig.entry),
                                        sl=float(sc_sig.sl),
                                        tp=float(sc_sig.tp),
                                        ml_score=float(score or 0.0),
                                        features=sc_feats or {}
                                    )
                                    logger.debug(f"[{sym}] 🩳 Scalp shadow trade recorded (score {score:.1f} ≥ {thr_sc})")
                        except Exception as se:
                            logger.debug(f"[{sym}] Scalp shadow error: {se}")
                    else:
                        # Blocked by hourly per-symbol budget or daily cap
                        try:
                            reason = 'hourly_budget' if sc_remaining <= 0 else ('daily_cap' if not daily_ok else 'unknown')
                            logger.info(f"[{sym}] 🧮 Scalp decision final: blocked (reason={reason})")
                            _scalp_decision_logged = True
                        except Exception:
                            pass
                except Exception as e:
                    # Elevate to WARNING so it is visible at default log level
                    logger.warning(f"[{sym}] Scalp(3m) record error: {e}")
                    try:
                        logger.info(f"[{sym}] 🧮 Scalp decision final: blocked (reason=record_error)")
                        _scalp_decision_logged = True
                    except Exception:
                        pass
                finally:
                    # Backstop: ensure a decision-final line is always emitted
                    try:
                        if not _scalp_decision_logged:
                            logger.debug(f"[{sym}] 🧮 Scalp decision final: blocked (reason=unknown_path)")
                    except Exception:
                        pass
                # Periodically publish Scalp state snapshot (approx every 30s)
                try:
                    import time as _t, json as _json
                    now_ts = _t.time()
                    last = getattr(self, '_scalp_state_flush_ts', 0.0)
                    if (now_ts - float(last)) >= 30.0:
                        mom = sum(1 for st in (self._scalp_symbol_state or {}).values() if st.get('mom'))
                        pull = sum(1 for st in (self._scalp_symbol_state or {}).values() if st.get('pull'))
                        vwap = sum(1 for st in (self._scalp_symbol_state or {}).values() if st.get('vwap'))
                        qthr = sum(1 for st in (self._scalp_symbol_state or {}).values() if st.get('q_ge_thr'))
                        exec_today = 0
                        try:
                            if hasattr(self, '_scalp_exec_counter'):
                                exec_today = int(self._scalp_exec_counter.get('count', 0))
                        except Exception:
                            pass
                        ph_open = 0
                        try:
                            from scalp_phantom_tracker import get_scalp_phantom_tracker
                            scpt2 = get_scalp_phantom_tracker()
                            act = getattr(scpt2, 'active', {}) or {}
                            ph_open = sum(len(lst) for lst in act.values())
                        except Exception:
                            pass
                        from datetime import datetime as _dt
                        snapshot = {
                            'ts': _dt.utcnow().isoformat()+'Z',
                            'mom': mom,
                            'pull': pull,
                            'vwap': vwap,
                            'q_ge_thr': qthr,
                            'exec_today': exec_today,
                            'phantom_open': ph_open,
                            'reasons': dict(self._scalp_reasons or {})
                        }
                        self.shared['scalp_states'] = snapshot
                        try:
                            if hasattr(self, '_redis') and self._redis is not None:
                                self._redis.set('state:scalp:summary', _json.dumps(snapshot))
                        except Exception:
                            pass
                        self._scalp_state_flush_ts = now_ts
                except Exception:
                    pass
            except Exception as e:
                logger.debug(f"Secondary stream update error for {sym}: {e}")

    def _scalp_secondary_stale(self, sym: str, stale_minutes: int = 30) -> bool:
        """Return True if the 3m scalp stream appears stale/missing for a symbol."""
        try:
            last = self._scalp_last_confirm.get(sym)
            if last is None:
                return True
            # If no confirm in the last 'stale_minutes', consider stale
            delta = (pd.Timestamp.utcnow().tz_localize('UTC') - last) if last.tzinfo else (pd.Timestamp.utcnow() - last)
            return delta.total_seconds() > stale_minutes * 60
        except Exception:
            return True

    async def _maybe_run_scalp_fallback(self, sym: str, df: pd.DataFrame, regime_analysis, cluster_id: Optional[int]):
        """Run Scalp detection on main/3m frames when the secondary stream is unavailable or stale.

        This preserves Scalp independence by only engaging when the 3m loop isn't producing confirms.
        """
        # Trend-only mode: completely skip Scalp fallback path
        try:
            if getattr(self, '_trend_only', False):
                return
        except Exception:
            pass
        # Guards
        try:
            s_cfg = self.config.get('scalp', {}) if hasattr(self, 'config') else {}
        except Exception:
            s_cfg = {}
        use_scalp = bool(s_cfg.get('enabled', False) and SCALP_AVAILABLE and detect_scalp_signal is not None)
        if not use_scalp:
            return

        scalp_independent = bool(s_cfg.get('independent', False))
        # Only fallback when 3m stream not started or appears stale
        if scalp_independent and self._scalp_secondary_started and not self._scalp_secondary_stale(sym):
            return

        # Prefer 3m frames if we have enough, else use main tf
        try:
            df3 = self.frames_3m.get(sym)
        except Exception:
            df3 = None
        if df3 is not None and not df3.empty and len(df3) >= 120:
            df_for_scalp = df3
            logger.info(f"[{sym}] 🩳 Fallback Scalp using 3m frames ({len(df3)} bars)")
        else:
            df_for_scalp = df
            if df3 is None or df3.empty:
                logger.info(f"[{sym}] 🩳 Fallback Scalp using main tf: 3m unavailable")
            else:
                logger.info(f"[{sym}] 🩳 Fallback Scalp using main tf: 3m sparse ({len(df3)} bars)")

        # Run detection
        try:
            sc_set = ScalpSettings()
            try:
                s_cfg = self.config.get('scalp', {}) if hasattr(self, 'config') else {}
                if 'min_r_pct' in s_cfg:
                    sc_set.min_r_pct = float(s_cfg.get('min_r_pct'))
            except Exception:
                pass
            sc_sig = detect_scalp_signal(df_for_scalp.copy(), sc_set, sym)
        except Exception as e:
            logger.debug(f"[{sym}] Scalp fallback detection error: {e}")
            sc_sig = None
        if not sc_sig:
            return

        # Immediate execution for ALL Scalp detections in fallback (disabled by default; use high-ML path instead)
        try:
            exec_all = False
            try:
                exec_all = bool((((self.config.get('scalp', {}) or {}).get('exec', {}) or {}).get('execute_all', False)))
            except Exception:
                exec_all = False
            # Build features
            sc_meta = getattr(sc_sig, 'meta', {}) or {}
            try:
                vol_level = getattr(regime_analysis, 'volatility_level', 'normal') if regime_analysis else 'normal'
            except Exception:
                vol_level = 'normal'
            sc_feats = self._build_scalp_features(df_for_scalp, sc_meta, vol_level, cluster_id)
            try:
                comp = self._get_htf_metrics(sym, self.frames.get(sym))
                sc_feats['ts15'] = float(comp.get('ts15', 0.0)); sc_feats['ts60'] = float(comp.get('ts60', 0.0))
                sc_feats['rc15'] = float(comp.get('rc15', 0.0)); sc_feats['rc60'] = float(comp.get('rc60', 0.0))
            except Exception:
                pass
            # Score Scalp ML for learning even on immediate executes
            ml_s_immediate = 0.0
            try:
                from ml_scorer_scalp import get_scalp_scorer
                _scorer = get_scalp_scorer()
                ml_s_immediate, _ = _scorer.score_signal({'side': sc_sig.side, 'entry': sc_sig.entry, 'sl': sc_sig.sl, 'tp': sc_sig.tp}, sc_feats)
            except Exception:
                ml_s_immediate = 0.0
            if exec_all:
                if sym in self.book.positions:
                    logger.info(f"[{sym}] 🛑 Scalp fallback execute blocked: reason=position_exists")
                    return
                executed = False
                try:
                    executed = await self._execute_scalp_trade(sym, sc_sig, ml_score=float(ml_s_immediate or 0.0))
                except Exception as _ee:
                    logger.info(f"[{sym}] Scalp fallback execute error: {_ee}")
                    executed = False
                try:
                    scpt = _safe_get_scalp_phantom_tracker()
                    # Cancel actives if configured
                    try:
                        cancel_on_hi = bool(((self.config.get('scalp', {}) or {}).get('exec', {}) or {}).get('cancel_active_on_high_ml', True))
                    except Exception:
                        cancel_on_hi = True
                    if bool(executed) and cancel_on_hi:
                        try:
                            scpt.cancel_active(sym)
                        except Exception:
                            pass
                    scpt.record_scalp_signal(
                        sym,
                        {'side': sc_sig.side, 'entry': sc_sig.entry, 'sl': sc_sig.sl, 'tp': sc_sig.tp},
                        float(ml_s_immediate or 0.0),
                        bool(executed),
                        sc_feats
                    )
                except Exception:
                    pass
                if executed:
                    logger.info(f"[{sym}] 🧮 Scalp fallback decision: exec_scalp (reason=immediate)")
                    return
        except Exception:
            pass

        # Redis dedup (fallback path) — optional
        dedup_ok = True
        try:
            s_cfg = self.config.get('scalp', {}) if hasattr(self, 'config') else {}
            if bool(s_cfg.get('dedup_enabled', False)):
                scpt = _safe_get_scalp_phantom_tracker()
                r = scpt.redis_client
                if r is not None:
                    key = f"{sym}:{sc_sig.side}:{round(float(sc_sig.entry),6)}:{round(float(sc_sig.sl),6)}:{round(float(sc_sig.tp),6)}"
                    # TTL configurable
                    try:
                        dedup_hours = int(s_cfg.get('dedup_hours', 8))
                    except Exception:
                        dedup_hours = 8
                    if r.exists(f"phantom:dedup:scalp:{key}"):
                        dedup_ok = False
                    else:
                        r.setex(f"phantom:dedup:scalp:{key}", max(1, dedup_hours) * 3600, '1')
        except Exception:
            pass
        if not dedup_ok:
            logger.debug(f"[{sym}] 🩳 Scalp fallback dedup: duplicate signal skipped")
            return

        # Allow multiple active phantoms per symbol for fallback learning (no active_symbol block)
        try:
            if scpt.has_active(sym):
                logger.debug(f"[{sym}] 🧮 Scalp fallback note: active phantom exists — allowing additional phantom for learning")
        except Exception:
            pass

        # Build features and record phantom with pacing
        sc_meta = getattr(sc_sig, 'meta', {}) or {}
        try:
            vol_level = getattr(regime_analysis, 'volatility_level', 'normal') if regime_analysis else 'normal'
        except Exception:
            vol_level = 'normal'
        sc_feats = self._build_scalp_features(df_for_scalp, sc_meta, vol_level, cluster_id)
        try:
            comp = self._get_htf_metrics(sym, self.frames.get(sym))
            sc_feats['ts15'] = float(comp.get('ts15', 0.0)); sc_feats['ts60'] = float(comp.get('ts60', 0.0))
            sc_feats['rc15'] = float(comp.get('rc15', 0.0)); sc_feats['rc60'] = float(comp.get('rc60', 0.0))
        except Exception:
            pass
        sc_feats['routing'] = 'fallback'

        try:
            scpt = _safe_get_scalp_phantom_tracker()
            hb = self.config.get('phantom', {}).get('hourly_symbol_budget', {}) or {}
            sc_limit = int(hb.get('scalp', 4))
            if not hasattr(self, '_scalp_budget'):
                self._scalp_budget = {}
            now_ts = pd.Timestamp.utcnow().timestamp()
            blist = [ts for ts in self._scalp_budget.get(sym, []) if (now_ts - ts) < 3600]
            self._scalp_budget[sym] = blist
            sc_remaining = sc_limit - len(blist)
            # Daily cap (none) per strategy for scalp
            daily_ok = True
            n_key = None
            try:
                if scpt.redis_client is not None:
                    day = pd.Timestamp.utcnow().strftime('%Y%m%d')
                    caps_cfg = (self.config.get('phantom', {}).get('caps', {}) or {}).get('scalp', {})
                    none_cap = int(caps_cfg.get('none', 200))
                    n_key = f"phantom:daily:none_count:{day}:scalp"
                    n_val = int(scpt.redis_client.get(n_key) or 0)
                    daily_ok = n_val < none_cap
            except Exception:
                pass
            # Debug: force accept path for fallback
            force_accept = False
            try:
                force_accept = bool(((self.config.get('scalp', {}).get('debug', {}) or {}).get('force_accept', False)))
            except Exception:
                force_accept = False
            if force_accept:
                scpt.record_scalp_signal(
                    sym,
                    {'side': sc_sig.side, 'entry': sc_sig.entry, 'sl': sc_sig.sl, 'tp': sc_sig.tp},
                    float(ml_s or 0.0) if 'ml_s' in locals() else 0.0,
                    False,
                    sc_feats
                )
                logger.info(f"[{sym}] 👻 Phantom-only (Scalp fallback): {sc_sig.side.upper()} @ {sc_sig.entry:.4f}")
                try:
                    logger.info(f"[{sym}] 🧮 Scalp decision final: phantom (reason=debug_force)")
                except Exception:
                    pass
                blist.append(now_ts)
                self._scalp_budget[sym] = blist
                return
            # High-ML immediate execution override (fallback path)
            try:
                e_cfg = (self.config.get('scalp', {}) or {}).get('exec', {})
                allow_hi = bool(e_cfg.get('allow_stream_high_ml', True))
                hi_thr = float(e_cfg.get('high_ml_force', 92))
            except Exception:
                allow_hi = True; hi_thr = 92.0
            if allow_hi and float(ml_s or 0.0) >= hi_thr:
                try:
                    if sym in self.book.positions:
                        logger.info(f"[{sym}] 🛑 Scalp High-ML (fallback) blocked: reason=position_exists")
                    else:
                        vol_ok = str(sc_feats.get('volatility_regime','normal')) in ('normal','high')
                        fast = float(sc_feats.get('ema_slope_fast', 0.0) or 0.0)
                        slow = float(sc_feats.get('ema_slope_slow', 0.0) or 0.0)
                        side = str(sc_sig.side)
                        micro_ok = (side == 'long' and fast >= 0.0 and slow >= 0.0) or (side == 'short' and fast <= 0.0 and slow <= 0.0)
                        if not (vol_ok and micro_ok):
                            logger.info(f"[{sym}] 🛑 Scalp execution (fallback) blocked by regime gate (vol={sc_feats.get('volatility_regime')} fast={fast:.2f} slow={slow:.2f} side={side})")
                            executed = False
                        else:
                            executed = await self._execute_scalp_trade(sym, sc_sig, ml_score=float(ml_s or 0.0))
                        scpt.record_scalp_signal(
                            sym,
                            {'side': sc_sig.side, 'entry': sc_sig.entry, 'sl': sc_sig.sl, 'tp': sc_sig.tp},
                            float(ml_s or 0.0),
                            bool(executed),
                            sc_feats
                        )
                        if executed:
                            logger.info(f"[{sym}] 🧮 Scalp decision final: exec_scalp (reason=ml_extreme {float(ml_s or 0.0):.1f}>={hi_thr:.0f})")
                            blist.append(now_ts)
                            self._scalp_budget[sym] = blist
                            return
                        else:
                            logger.info(f"[{sym}] 🛑 Scalp High-ML override blocked: reason=exec_guard")
                except Exception as _ee:
                    logger.info(f"[{sym}] Scalp High-ML (fallback) error: {_ee}")
            if sc_remaining > 0 and daily_ok:
                scpt.record_scalp_signal(
                    sym,
                    {'side': sc_sig.side, 'entry': sc_sig.entry, 'sl': sc_sig.sl, 'tp': sc_sig.tp},
                    float(ml_s or 0.0) if 'ml_s' in locals() else 0.0,
                    False,
                    sc_feats
                )
                try:
                    if hasattr(self, 'flow_controller') and self.flow_controller and self.flow_controller.enabled:
                        self.flow_controller.increment_accepted('scalp', 1)
                except Exception:
                    pass
                try:
                    if scpt.redis_client is not None and n_key is not None:
                        scpt.redis_client.incr(n_key)
                except Exception:
                    pass
                logger.info(f"[{sym}] 👻 Phantom-only (Scalp fallback): {sc_sig.side.upper()} @ {sc_sig.entry:.4f}")
                blist.append(now_ts)
                self._scalp_budget[sym] = blist
                try:
                            # Explain fallback phantom reason (usually ML below threshold vs budget/cap)
                            reason_fb = 'unknown'
                            try:
                                if not daily_ok:
                                    reason_fb = 'daily_cap'
                                elif sc_remaining <= 0:
                                    reason_fb = 'hourly_budget'
                                else:
                                    reason_fb = f"ml<thr {float(ml_s or 0.0):.1f}<{hi_thr:.0f}"
                            except Exception:
                                pass
                            logger.info(f"[{sym}] 🧮 Scalp decision final: phantom (reason={reason_fb})")
                except Exception:
                    pass
            else:
                # Blocked by hourly per-symbol budget or daily cap
                try:
                    reason = 'hourly_budget' if sc_remaining <= 0 else ('daily_cap' if not daily_ok else 'unknown')
                    logger.info(f"[{sym}] 🧮 Scalp decision final: blocked (reason={reason})")
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"[{sym}] Scalp fallback record error: {e}")
            try:
                logger.info(f"[{sym}] 🧮 Scalp decision final: blocked (reason=record_error)")
            except Exception:
                pass

    async def _notify_phantom_trade(self, label: str, phantom):
        if not self.tg:
            return

        try:
            outcome = getattr(phantom, 'outcome', None)
            was_exec = bool(getattr(phantom, 'was_executed', False))
            symbol = getattr(phantom, 'symbol', '?')
            side = getattr(phantom, 'side', '?').upper()
            entry_price = float(getattr(phantom, 'entry_price', 0.0) or 0.0)
            sl = float(getattr(phantom, 'stop_loss', 0.0) or 0.0)
            tp = float(getattr(phantom, 'take_profit', 0.0) or 0.0)
            ml_score = float(getattr(phantom, 'ml_score', 0.0) or 0.0)

            # Open notification (no outcome yet)
            if not outcome:
                prefix = "👻" if not was_exec else "🟢"
                pid = getattr(phantom, 'phantom_id', '')
                pid_suffix = f" [#{pid}]" if isinstance(pid, str) and pid else ""
                # Include Qscore if present in features
                try:
                    feats = getattr(phantom, 'features', {}) or {}
                    qv = feats.get('qscore', None)
                except Exception:
                    qv = None
                lines = [
                    f"{prefix} *{label} Phantom Opened*{pid_suffix}",
                    f"{symbol} {side} | ML {ml_score:.1f}{(' | Q '+format(float(qv),'.1f')) if isinstance(qv,(int,float)) else ''}",
                    f"Entry: {entry_price:.4f}",
                    f"TP / SL: {tp:.4f} / {sl:.4f}"
                ]
                if label == "Mean Reversion":
                    try:
                        ru = getattr(phantom, 'range_upper', None)
                        rl = getattr(phantom, 'range_lower', None)
                        if isinstance(ru, (int,float)) and isinstance(rl, (int,float)) and rl and ru:
                            width_pct = (float(ru) - float(rl)) / float(rl) * 100.0
                            lines.append(f"Range: {rl:.4f}-{ru:.4f} ({width_pct:.1f}% width)")
                        pos = getattr(phantom, 'range_position', None)
                        if isinstance(pos, (int,float)):
                            lines.append(f"Range pos: {float(pos):.2f}")
                    except Exception:
                        pass
                try:
                    # Append Qscore if available in features
                    feats_local = getattr(phantom, 'features', {}) or {}
                    qv = feats_local.get('qscore', None)
                    if isinstance(qv, (int,float)):
                        lines.append(f"Q={float(qv):.1f}")
                except Exception:
                    pass
                await self.tg.send_message("\n".join([l for l in lines if l]))
                # Log open event
                try:
                    logger.info(f"[{symbol}] 👻 Phantom opened ({label}): {side} @ {entry_price:.4f} TP/SL {tp:.4f}/{sl:.4f} ML {ml_score:.1f}")
                except Exception:
                    pass
                return

            # Close notification
            outcome_emoji = "✅" if outcome == "win" else "❌"
            prefix = "👻" if not was_exec else "🔁"
            exit_reason = getattr(phantom, 'exit_reason', 'unknown')
            exit_label = str(exit_reason).replace('_', ' ').title()
            pnl_percent = float(getattr(phantom, 'pnl_percent', 0.0) or 0.0)
            exit_price = float(getattr(phantom, 'exit_price', 0.0) or 0.0)
            realized_rr = getattr(phantom, 'realized_rr', None)

            pid = getattr(phantom, 'phantom_id', '')
            pid_suffix = f" [#{pid}]" if isinstance(pid, str) and pid else ""
            # Include Qscore if present in features
            try:
                feats = getattr(phantom, 'features', {}) or {}
                qv = feats.get('qscore', None)
            except Exception:
                qv = None
            lines = [
                f"{prefix} *{label} Phantom {outcome_emoji}*{pid_suffix}",
                f"{symbol} {side} | ML {ml_score:.1f}{(' | Q '+format(float(qv),'.1f')) if isinstance(qv,(int,float)) else ''}",
                f"Entry → Exit: {entry_price:.4f} → {exit_price:.4f}",
                f"P&L: {pnl_percent:+.2f}% ({exit_label})"
            ]
            try:
                if isinstance(realized_rr, (int, float)):
                    lines.append(f"Realized R: {float(realized_rr):.2f}R")
            except Exception:
                pass

            if label == "Mean Reversion":
                range_conf = getattr(phantom, 'range_confidence', None)
                if isinstance(range_conf, (int,float)):
                    lines.append(f"Range confidence: {float(range_conf):.2f}")
                range_pos = getattr(phantom, 'range_position', None)
                if isinstance(range_pos, (int,float)):
                    lines.append(f"Range position: {float(range_pos):.2f}")

            try:
                feats_local = getattr(phantom, 'features', {}) or {}
                qv = feats_local.get('qscore', None)
                if isinstance(qv, (int,float)):
                    lines.append(f"Q={float(qv):.1f}")
            except Exception:
                pass
            await self.tg.send_message("\n".join(lines))
            # Log close event
            try:
                logger.info(f"[{symbol}] 👻 Phantom closed ({label}): {side} {outcome.upper()} PnL {pnl_percent:+.2f}% exit {exit_price:.4f} ({exit_label})")
            except Exception:
                pass

            # Telemetry counters for phantom outcomes (non-executed only)
            try:
                if hasattr(self.tg, 'shared') and not getattr(phantom, 'was_executed', False):
                    tel = self.tg.shared.get('telemetry', {})
                    if outcome == 'win':
                        tel['phantom_wins'] = tel.get('phantom_wins', 0) + 1
                    elif outcome == 'loss':
                        tel['phantom_losses'] = tel.get('phantom_losses', 0) + 1
            except Exception:
                pass

        except Exception as notify_err:
            logger.debug(f"Failed phantom notification: {notify_err}")

    async def _notify_trend_phantom(self, phantom):
        """Generic phantom lifecycle notifier (Trend/Range) with reasons.

        Called by PhantomTradeTracker via set_notifier(). Labels derived from strategy_name.
        """
        if not self.tg:
            return
        try:
            symbol = getattr(phantom, 'symbol', '?')
            side = str(getattr(phantom, 'side', '?')).upper()
            entry = float(getattr(phantom, 'entry_price', 0.0) or 0.0)
            sl = float(getattr(phantom, 'stop_loss', 0.0) or 0.0)
            tp = float(getattr(phantom, 'take_profit', 0.0) or 0.0)
            ml = float(getattr(phantom, 'ml_score', 0.0) or 0.0)
            outcome = getattr(phantom, 'outcome', None)
            pid = getattr(phantom, 'phantom_id', '')
            pid_suffix = f" [#{pid}]" if isinstance(pid, str) and pid else ""
            feats = getattr(phantom, 'features', {}) or {}

            # Determine label by strategy
            strat = str(getattr(phantom, 'strategy_name', '') or '').lower()
            if strat.startswith('range'):
                label_title = 'Range Phantom'
            elif strat.startswith('scalp'):
                label_title = 'Scalp Phantom'
            else:
                label_title = 'Trend Phantom'

            # State counters: track Range TP1(mid) hits for dashboard (persisted per-day)
            try:
                evt = str(getattr(phantom, 'phantom_event', '') or '')
                if label_title.startswith('Range') and evt == 'tp1':
                    if hasattr(self, '_redis') and self._redis is not None:
                        from datetime import datetime as _dt
                        day = _dt.utcnow().strftime('%Y%m%d')
                        key = f'state:range:tp1_hits:{day}'
                        try:
                            cur = int(self._redis.get(key) or 0)
                        except Exception:
                            cur = 0
                        try:
                            self._redis.set(key, str(cur + 1))
                        except Exception:
                            pass
            except Exception:
                pass

            # Closure (dedup by phantom_id)
            if outcome in ('win','loss') or getattr(phantom, 'exit_time', None):
                emoji = '✅' if outcome == 'win' else '❌'
                exit_reason = str(getattr(phantom, 'exit_reason', 'unknown')).replace('_',' ').title()
                pnl = float(getattr(phantom, 'pnl_percent', 0.0) or 0.0)
                exit_px = float(getattr(phantom, 'exit_price', 0.0) or 0.0)
                rr = getattr(phantom, 'realized_rr', None)
                q_close = None
                try:
                    q_close = float((feats or {}).get('qscore')) if isinstance(feats, dict) and ('qscore' in feats) else None
                except Exception:
                    q_close = None
                lines = [
                    f"👻 *{label_title} {emoji}*{pid_suffix}",
                    f"{symbol} {side} | ML {ml:.1f}",
                    f"Entry → Exit: {entry:.4f} → {exit_px:.4f}",
                    f"P&L: {pnl:+.2f}% ({exit_reason})"
                ]
                try:
                    if isinstance(rr, (int,float)):
                        lines.append(f"Realized R: {float(rr):.2f}R")
                except Exception:
                    pass
                try:
                    if isinstance(q_close, (int,float)):
                        lines.append(f"Q={q_close:.1f}")
                except Exception:
                    pass
                if pid and pid in self._phantom_close_notified:
                    return
                await self.tg.send_message("\n".join(lines))
                if pid:
                    self._phantom_close_notified.add(pid)
                return

            # TP1 event (active)
            if bool(getattr(phantom, 'tp1_hit', False)) and str(getattr(phantom, 'phantom_event','')) == 'tp1':
                if not (pid and pid in self._phantom_tp1_notified):
                    # If Range midline is present, include it
                    try:
                        strat = str(getattr(phantom, 'strategy_name', '') or '').lower()
                        feats_tp1 = getattr(phantom, 'features', {}) or {}
                        if strat.startswith('range') and isinstance(feats_tp1.get('range_mid', None), (int,float)):
                            await self.tg.send_message(f"🎯 Phantom TP1: {symbol} {side}{pid_suffix} — SL→BE at {entry:.4f} (mid {float(feats_tp1['range_mid']):.4f})")
                        else:
                            await self.tg.send_message(f"🎯 Phantom TP1: {symbol} {side}{pid_suffix} — SL→BE at {entry:.4f}")
                    except Exception:
                        await self.tg.send_message(f"🎯 Phantom TP1: {symbol} {side}{pid_suffix} — SL→BE at {entry:.4f}")
                    if pid:
                        self._phantom_tp1_notified.add(pid)
                try:
                    setattr(phantom, 'phantom_event', '')
                except Exception:
                    pass
                return

            # Open (active) — include reason
            decision = str(feats.get('decision',''))
            reason = feats.get('diversion_reason', '')
            q = feats.get('qscore', None)
            comps = feats.get('qscore_components', {}) or {}
            reason_line = None
            if decision:
                reason_line = f"Reason: {decision}"
            elif reason:
                reason_line = f"Reason: {reason}"
            elif isinstance(q, (int,float)):
                try:
                    # Use per-strategy thresholds
                    if label_title.startswith('Range'):
                        exec_min = float((((self.config.get('range',{}) or {}).get('rule_mode',{}) or {}).get('execute_q_min', 78)))
                    elif label_title.startswith('Scalp'):
                        exec_min = float((((self.config.get('scalp',{}) or {}).get('rule_mode',{}) or {}).get('execute_q_min', 60)))
                    else:
                        exec_min = float(((self.config.get('trend',{}) or {}).get('rule_mode',{}) or {}).get('execute_q_min', 78))
                    lt = "<" if float(q) < exec_min else ">="
                    reason_line = f"Q={float(q):.1f} ({lt} {exec_min:.0f})"
                except Exception:
                    reason_line = f"Q={float(q):.1f}"
            # Compute simple regime label from features if available
            reg_label = None
            try:
                feats_local = getattr(phantom, 'features', {}) or {}
                t15 = float(feats_local.get('ts15', 0.0) or 0.0)
                t60 = float(feats_local.get('ts60', 0.0) or 0.0)
                rc15 = float(feats_local.get('rc15', 0.0) or 0.0)
                rc60 = float(feats_local.get('rc60', 0.0) or 0.0)
                if t15 >= 60 or t60 >= 60:
                    reg_label = 'Trending'
                elif rc15 >= 0.6 or rc60 >= 0.6:
                    reg_label = 'Ranging'
                else:
                    reg_label = 'Neutral'
            except Exception:
                reg_label = None

            lines = [
                f"👻 *{label_title} Opened*{pid_suffix}",
                f"{symbol} {side} | ML {ml:.1f}",
                f"Entry: {entry:.4f}",
            ]
            try:
                if label_title.startswith('Range'):
                    if isinstance(feats.get('range_mid', None), (int,float)):
                        lines.append(f"TP1(mid): {float(feats['range_mid']):.4f}")
                    lines.append(f"TP2 / SL: {tp:.4f} / {sl:.4f}")
                else:
                    lines.append(f"TP / SL: {tp:.4f} / {sl:.4f}")
            except Exception:
                lines.append(f"TP / SL: {tp:.4f} / {sl:.4f}")
            try:
                if reg_label:
                    lines.append(f"Regime: {reg_label}")
            except Exception:
                pass
            if reason_line:
                lines.append(reason_line)
            try:
                if comps:
                    # Display Qscore components per strategy to avoid confusion
                    if label_title.startswith('Range'):
                        # Range components: rng, fbo, prox, micro, risk, sr
                        lines.append(
                            f"RNG={comps.get('rng',0):.0f} FBO={comps.get('fbo',0):.0f} PROX={comps.get('prox',0):.0f} Micro={comps.get('micro',0):.0f} Risk={comps.get('risk',0):.0f} SR={comps.get('sr',0):.0f}"
                        )
                    elif label_title.startswith('Scalp'):
                        # Scalp components: mom, pull, micro, htf, sr, risk
                        lines.append(
                            f"MOM={comps.get('mom',0):.0f} PULL={comps.get('pull',0):.0f} Micro={comps.get('micro',0):.0f} HTF={comps.get('htf',0):.0f} SR={comps.get('sr',0):.0f} Risk={comps.get('risk',0):.0f}"
                        )
                    else:
                        # Trend components: sr, htf, bos, micro, risk, div
                        lines.append(
                            f"SR={comps.get('sr',0):.0f} HTF={comps.get('htf',0):.0f} BOS={comps.get('bos',0):.0f} Micro={comps.get('micro',0):.0f} Risk={comps.get('risk',0):.0f} Div={comps.get('div',0):.0f}"
                        )
            except Exception:
                pass
            # De-dup open by phantom_id
            if pid and pid in self._phantom_open_notified:
                return
            await self.tg.send_message("\n".join([l for l in lines if l]))
            if pid:
                self._phantom_open_notified.add(pid)
        except Exception as e:
            logger.debug(f"Trend phantom notify error: {e}")

    async def _notify_scalp_phantom(self, phantom):
        """Scalp phantom lifecycle notifier (open/TP/close) to Telegram.

        Called by ScalpPhantomTracker via set_notifier().
        """
        if not self.tg:
            return
        try:
            symbol = getattr(phantom, 'symbol', '?')
            side = str(getattr(phantom, 'side', '?')).upper()
            entry = float(getattr(phantom, 'entry_price', 0.0) or 0.0)
            sl = float(getattr(phantom, 'stop_loss', 0.0) or 0.0)
            tp = float(getattr(phantom, 'take_profit', 0.0) or 0.0)
            ml = float(getattr(phantom, 'ml_score', 0.0) or 0.0)
            outcome = getattr(phantom, 'outcome', None)
            pid = getattr(phantom, 'phantom_id', '')
            pid_suffix = f" [#{pid}]" if isinstance(pid, str) and pid else ""
            feats = getattr(phantom, 'features', {}) or {}

            # Close notification
            if outcome in ('win','loss') or getattr(phantom, 'exit_time', None):
                emoji = '✅' if outcome == 'win' else ('⏱️' if str(getattr(phantom,'exit_reason','')).lower()=='timeout' else '❌')
                exit_reason = str(getattr(phantom, 'exit_reason', 'unknown')).replace('_',' ').title()
                pnl = float(getattr(phantom, 'pnl_percent', 0.0) or 0.0)
                exit_px = float(getattr(phantom, 'exit_price', 0.0) or 0.0)
                rr = getattr(phantom, 'realized_rr', None)
                qv = None
                try:
                    feats = getattr(phantom, 'features', {}) or {}
                    qv = float(feats.get('qscore')) if 'qscore' in feats else None
                except Exception:
                    qv = None
                lines = [
                    f"👻 *Scalp Phantom {emoji}*{pid_suffix}",
                    f"{symbol} {side} | ML {ml:.1f}",
                    f"Entry → Exit: {entry:.4f} → {exit_px:.4f}",
                    f"P&L: {pnl:+.2f}% ({exit_reason})"
                ]
                try:
                    if isinstance(rr, (int,float)):
                        lines.append(f"Realized R: {float(rr):.2f}R")
                except Exception:
                    pass
                try:
                    if isinstance(qv, (int,float)):
                        lines.append(f"Q={qv:.1f}")
                except Exception:
                    pass
                if pid and pid in getattr(self, '_phantom_close_notified', set()):
                    return
                await self.tg.send_message("\n".join(lines))
                try:
                    self._phantom_close_notified.add(pid)
                except Exception:
                    pass
                return

            # Open notification with Qscore breakdown
            q = feats.get('qscore', None)
            comps = feats.get('qscore_components', {}) or {}
            reason_line = f"Q={float(q):.1f}" if isinstance(q, (int,float)) else None
            lines = [
                f"👻 *Scalp Phantom Opened*{pid_suffix}",
                f"{symbol} {side} | ML {ml:.1f}",
                f"Entry: {entry:.4f}",
                f"TP / SL: {tp:.4f} / {sl:.4f}"
            ]
            if reason_line:
                lines.append(reason_line)
            try:
                if comps:
                    lines.append(f"MOM={comps.get('mom',0):.0f} PULL={comps.get('pull',0):.0f} Micro={comps.get('micro',0):.0f} HTF={comps.get('htf',0):.0f} SR={comps.get('sr',0):.0f} Risk={comps.get('risk',0):.0f}")
            except Exception:
                pass
            if pid and pid in getattr(self, '_phantom_open_notified', set()):
                return
            await self.tg.send_message("\n".join([l for l in lines if l]))
            try:
                self._phantom_open_notified.add(pid)
            except Exception:
                pass
        except Exception as e:
            try:
                logger.debug(f"Scalp phantom notify error: {e}")
            except Exception:
                pass

    def _trend_invalidation_handoff(self, symbol: str, info: dict):
        """Soft handoff from Trend invalidation to Range phantom (one-shot, guarded)."""
        try:
            cfg = self.config if hasattr(self, 'config') else {}
            rcfg = (cfg.get('range', {}) or {})
            hcfg = (rcfg.get('handoff', {}) or {})
            if not bool(rcfg.get('enabled', True)) or not bool(hcfg.get('enabled', True)):
                return
            # Cooldown
            if not hasattr(self, '_range_handoff_last'):
                self._range_handoff_last = {}
            import time, asyncio as _asyncio
            now = time.time()
            cd_min = int(hcfg.get('cooldown_min', 30))
            last = float(self._range_handoff_last.get(symbol, 0))
            if last and (now - last) < cd_min * 60:
                if bool(hcfg.get('notify_suppress', True)) and self.tg:
                    try:
                        left = int(cd_min*60 - (now-last))
                        _asyncio.create_task(self.tg.send_message(f"📦 Range handoff suppressed: {symbol} cooldown {left//60}m"))
                    except Exception:
                        pass
                return
            df = self.frames.get(symbol)
            if df is None or df.empty:
                return
            # Guard strictly by bars since breakout_time
            try:
                max_bars = int(hcfg.get('max_bars', 2))
            except Exception:
                max_bars = 2
            try:
                bt_s = info.get('breakout_time') if isinstance(info, dict) else None
                bars_since = None
                if bt_s:
                    import pandas as pd
                    bt = pd.Timestamp(bt_s)
                    bars_since = int((df.index >= bt).sum())
                    if bars_since > max_bars:
                        return
            except Exception:
                pass
            # Run one-shot range detection
            try:
                from strategy_range_fbo import detect_range_fbo_signal
                sig = detect_range_fbo_signal(df, rcfg, symbol)
            except Exception:
                sig = None
            if not sig:
                return
            # Qscore_range
            try:
                q, qc, qr = self._compute_qscore_range(symbol, sig.side, df, self.frames_3m.get(symbol) if hasattr(self, 'frames_3m') else None)
            except Exception:
                q, qc, qr = 50.0, {}, []
            # Build robust features for Range handoff phantom
            feats = {
                'handoff': True,
                'handoff_reason': 'trend_invalidation',
                'breakout_level': float(info.get('breakout_level', 0.0) or 0.0),
                'qscore': float(q),
                'qscore_components': dict(qc),
                'qscore_reasons': list(qr)
            }
            # Stamp handoff timing and invalidation distance (ATR)
            try:
                if bt_s and bars_since is not None:
                    feats['bars_since_breakout'] = int(bars_since)
            except Exception:
                pass
            try:
                # Compute invalidation distance in ATR units using last close vs breakout_level
                prev = df['close'].shift()
                import numpy as np
                trarr = np.maximum(df['high'] - df['low'], np.maximum((df['high'] - prev).abs(), (df['low'] - prev).abs()))
                atr14 = float(trarr.rolling(14).mean().iloc[-1]) if len(trarr) >= 14 else float(trarr.iloc[-1])
                last_px = float(df['close'].iloc[-1])
                bl = float(info.get('breakout_level', 0.0) or 0.0)
                feats['invalidation_dist_atr'] = float(abs(last_px - bl) / max(1e-9, atr14))
            except Exception:
                pass
            # Range features
            try:
                feats.update({
                    'range_high': float(sig.meta.get('range_high', 0.0)),
                    'range_low': float(sig.meta.get('range_low', 0.0)),
                    'range_mid': float(sig.meta.get('range_mid', 0.0)),
                    'range_width_pct': float(sig.meta.get('range_width_pct', 0.0)),
                    'fbo_type': str(sig.meta.get('fbo_type','')),
                    'wick_ratio': float(sig.meta.get('wick_ratio', 0.0)),
                    'retest_ok': bool(sig.meta.get('retest_ok', False))
                })
            except Exception:
                pass
            # Session and cluster context
            try:
                from datetime import datetime as _dt
                hr = _dt.utcnow().hour
                feats['session'] = 'asian' if 0 <= hr < 8 else ('european' if hr < 16 else 'us')
            except Exception:
                feats['session'] = 'us'
            # Volatility regime and range position extras
            try:
                reg = get_enhanced_market_regime(df, symbol)
                feats['volatility_regime'] = str(getattr(reg, 'volatility_level', 'normal') or 'normal')
            except Exception:
                feats['volatility_regime'] = 'normal'
            try:
                clp = float(df['close'].iloc[-1]) if len(df) else 0.0
                rh = float(feats.get('range_high', 0.0)); rl = float(feats.get('range_low', 0.0))
                rng_w = max(1e-9, (rh - rl))
                feats['range_pos_pct'] = float(min(1.0, max(0.0, (clp - rl) / rng_w)))
                prev = df['close'].shift()
                trarr = np.maximum(df['high'] - df['low'], np.maximum((df['high'] - prev).abs(), (df['low'] - prev).abs()))
                atr14 = float(trarr.rolling(14).mean().iloc[-1]) if len(trarr) >= 14 else float(trarr.iloc[-1])
                edge_dist = min(abs(clp - rl), abs(rh - clp))
                feats['edge_distance_atr'] = float(edge_dist / max(1e-9, atr14))
            except Exception:
                pass
            try:
                from symbol_clustering import load_symbol_clusters
                clusters = load_symbol_clusters()
                feats['symbol_cluster'] = int(clusters.get(symbol, 3))
            except Exception:
                feats['symbol_cluster'] = 3
            # RR target
            try:
                entry = float(sig.entry); sl_v = float(sig.sl); tp_v = float(sig.tp)
                R = abs(entry - sl_v)
                feats['rr_target'] = float(abs(tp_v - entry) / R) if R > 0 else 0.0
            except Exception:
                feats['rr_target'] = 0.0
            try:
                comp = self._get_htf_metrics(symbol, df)
                feats['ts15'] = float(comp.get('ts15', 0.0)); feats['ts60'] = float(comp.get('ts60', 0.0))
                feats['rc15'] = float(comp.get('rc15', 0.0)); feats['rc60'] = float(comp.get('rc60', 0.0))
            except Exception:
                pass
            # Record phantom (prefer shared tracker instance)
            try:
                pt = None
                try:
                    pt = self.shared.get('phantom_tracker') if hasattr(self, 'shared') else None
                except Exception:
                    pt = None
                if pt is None:
                    from phantom_trade_tracker import get_phantom_tracker
                    pt = get_phantom_tracker()
                pt.record_signal(symbol, {'side': sig.side, 'entry': float(sig.entry), 'sl': float(sig.sl), 'tp': float(sig.tp)}, 0.0, False, feats, 'range_fbo')
            except Exception:
                pass
            # Notify + events
            try:
                comps = f"RNG={qc.get('rng',0):.0f} FBO={qc.get('fbo',0):.0f} PROX={qc.get('prox',0):.0f} Micro={qc.get('micro',0):.0f} Risk={qc.get('risk',0):.0f}"
                if self.tg:
                    # Append regime label
                    try:
                        reg = None
                        t15 = float(feats.get('ts15',0.0)); t60 = float(feats.get('ts60',0.0)); rc15 = float(feats.get('rc15',0.0)); rc60 = float(feats.get('rc60',0.0))
                        if t15 >= 60 or t60 >= 60:
                            reg = 'Trending'
                        elif rc15 >= 0.6 or rc60 >= 0.6:
                            reg = 'Ranging'
                        else:
                            reg = 'Neutral'
                        msg = f"📦 Range Handoff PHANTOM: [{symbol}] Q={q:.1f} (from Trend invalidation)\n{comps}\nRegime: {reg}"
                    except Exception:
                        msg = f"📦 Range Handoff PHANTOM: [{symbol}] Q={q:.1f} (from Trend invalidation)\n{comps}"
                    _asyncio.create_task(self.tg.send_message(msg))
                evts = self.shared.get('trend_events')
                if isinstance(evts, list):
                    from datetime import datetime as _dt
                    evts.append({'ts': _dt.utcnow().isoformat()+'Z', 'symbol': symbol, 'text': f"Range Handoff PHANTOM Q={q:.1f}"})
                    if len(evts) > 60:
                        del evts[:len(evts)-60]
            except Exception:
                pass
            self._range_handoff_last[symbol] = now
        except Exception as e:
            try:
                logger.debug(f"Range handoff error for {symbol}: {e}")
            except Exception:
                pass

    def _compute_qscore_range(self, symbol: str, side: str, df15: 'pd.DataFrame', df3: 'pd.DataFrame' = None) -> tuple[float, dict, list[str]]:
        """Compute range FBO quality score (0–100) with components.

        Components: rng (35), fbo (25), prox (15), micro (10), risk (10), sr (5)
        """
        comp = {}; reasons = []
        try:
            cl = df15['close']; price = float(cl.iloc[-1]) if len(cl) else 0.0
            high = df15['high']; low = df15['low']
            lookback = 40
            try:
                lookback = int(((self.config.get('range', {}) or {}).get('lookback', 40)))
            except Exception:
                pass
            rng_high = float(high.rolling(lookback).max().iloc[-2]) if len(high) >= lookback+2 else 0.0
            rng_low = float(low.rolling(lookback).min().iloc[-2]) if len(low) >= lookback+2 else 0.0
            width_pct = ((rng_high - rng_low) / max(1e-9, rng_low)) if (rng_high > rng_low > 0) else 0.0
            # Range quality: best in mid band of configured bounds
            try:
                width_min = float(((self.config.get('range', {}) or {}).get('width_min_pct', 0.008)))
                width_max = float(((self.config.get('range', {}) or {}).get('width_max_pct', 0.16)))
            except Exception:
                width_min, width_max = 0.008, 0.16  # Match config defaults
            if width_pct <= 0:
                comp['rng'] = 20.0; reasons.append('rng:none')
            elif width_pct < width_min:
                comp['rng'] = 50.0; reasons.append('rng:too_tight')
            elif width_pct > width_max:
                comp['rng'] = 50.0; reasons.append('rng:too_wide')
            else:
                comp['rng'] = 85.0
            # FBO strength proxy: wick ratio and volume zscore
            try:
                rng_bar = float(high.iloc[-1] - low.iloc[-1]);
                wick = (float(high.iloc[-1] - cl.iloc[-1]) if side=='short' else float(cl.iloc[-1] - low.iloc[-1]))
                wick_ratio = max(0.0, min(1.0, wick / max(1e-9, rng_bar)))
            except Exception:
                wick_ratio = 0.0
            vol_z = 0.0
            try:
                v = df15['volume']
                vol_z = float((v.iloc[-1] - v.rolling(20).mean().iloc[-1]) / max(1e-9, v.rolling(20).std().iloc[-1])) if len(v) >= 20 else 0.0
            except Exception:
                vol_z = 0.0
            comp['fbo'] = max(0.0, min(100.0, 60.0 * wick_ratio + 10.0 * max(0.0, vol_z)))
            # Proximity to band/mid preference
            try:
                mid = (rng_high + rng_low) / 2.0
                if side == 'short':
                    prox = abs(price - mid) / max(1e-9, (rng_high - rng_low))
                else:
                    prox = abs(price - mid) / max(1e-9, (rng_high - rng_low))
                comp['prox'] = max(0.0, min(100.0, (1.0 - prox) * 100.0))
            except Exception:
                comp['prox'] = 50.0
            # Micro 3m reversal alignment
            try:
                ok3 = True
                if df3 is not None and len(df3) >= 4:
                    tail = df3['close'].tail(4)
                    up_seq = tail.iloc[-1] > tail.iloc[-2] >= tail.iloc[-3]
                    dn_seq = tail.iloc[-1] < tail.iloc[-2] <= tail.iloc[-3]
                    ok3 = (dn_seq if side=='short' else up_seq)
                comp['micro'] = 100.0 if ok3 else 30.0
            except Exception:
                comp['micro'] = 50.0
            # Risk geometry: mid band widths are safer
            comp['risk'] = 80.0 if 0.01 <= width_pct <= 0.08 else 50.0
            # SR confluence (HTF S/R near band edge)
            try:
                from multi_timeframe_sr import mtf_sr
                # Compute ATR(14) for normalization
                prev = cl.shift()
                tr = np.maximum(high - low, np.maximum((high - prev).abs(), (low - prev).abs()))
                atr14 = float(tr.rolling(14).mean().iloc[-1]) if len(tr) >= 14 else float(tr.iloc[-1])
                sr_cfg = (((self.config.get('range', {}) or {}).get('sr_confluence', {}) or {}))
                max_da = float(sr_cfg.get('max_dist_atr', 0.3))
                vlevels = mtf_sr.get_price_validated_levels(symbol, price)
                # Choose relevant band edge and level type
                if side == 'short':
                    edge = rng_high
                    levels = [(lv, st) for (lv, st, t) in vlevels if t == 'resistance']
                else:
                    edge = rng_low
                    levels = [(lv, st) for (lv, st, t) in vlevels if t == 'support']
                sr_score = 50.0
                if levels and atr14 > 0:
                    level, strength = min(levels, key=lambda x: abs(x[0] - edge))
                    dist_atr = abs(level - edge) / atr14
                    # Map distance to score: 0 ATR -> 100, max_da ATR -> 0 (clipped)
                    sr_score = max(0.0, min(100.0, (1.0 - (dist_atr / max(1e-9, max_da))) * 100.0))
                    # mild boost by strength proxy (cap small)
                    try:
                        sr_score = min(100.0, sr_score + min(10.0, float(strength) * 2.0))
                    except Exception:
                        pass
                comp['sr'] = sr_score
            except Exception as _sre:
                comp['sr'] = 50.0
                reasons.append(f'sr:error:{_sre}')

            # Weighted sum
            try:
                wcfg = (((self.config.get('range', {}) or {}).get('rule_mode', {}) or {}).get('weights', {}) or {})
                wr = float(wcfg.get('rng', 0.35)); wf = float(wcfg.get('fbo', 0.25)); wp = float(wcfg.get('prox', 0.15)); wm = float(wcfg.get('micro', 0.10)); wk = float(wcfg.get('risk', 0.10)); ws = float(wcfg.get('sr', 0.05))
                s = max(1e-9, wr + wf + wp + wm + wk + ws); wr/=s; wf/=s; wp/=s; wm/=s; wk/=s; ws/=s
            except Exception:
                wr,wf,wp,wm,wk,ws = 0.35,0.25,0.15,0.10,0.10,0.05
            q = wr*comp.get('rng',50.0) + wf*comp.get('fbo',50.0) + wp*comp.get('prox',50.0) + wm*comp.get('micro',50.0) + wk*comp.get('risk',50.0) + ws*comp.get('sr',50.0)
            return max(0.0, min(100.0, float(q))), comp, reasons
        except Exception as e:
            return 50.0, {}, [f'qr:error:{e}']


    async def load_or_fetch_initial_data(self, symbols:list[str], timeframe:str):
        """Load candles from database or fetch from API if not available"""
        logger.info("Loading historical data from database...")
        
        # First, try to load from database
        stored_frames = self.storage.load_all_frames(symbols)
        
        # Add rate limiting for large symbol lists
        api_delay = 0.1 if len(symbols) > 100 else 0  # 100ms delay for >100 symbols
        
        for idx, symbol in enumerate(symbols):
            if symbol in stored_frames and len(stored_frames[symbol]) >= 200:  # Need 200 for reliable analysis
                # Use stored data if we have enough candles
                self.frames[symbol] = stored_frames[symbol]
                logger.info(f"[{symbol}] Loaded {len(stored_frames[symbol])} candles from database")
            else:
                # Fetch from API if not in database or insufficient data
                try:
                    # Rate limit API calls
                    if api_delay > 0 and idx > 0:
                        await asyncio.sleep(api_delay)
                    
                    logger.info(f"[{symbol}] Fetching from API (not enough in database)")
                    klines = self.bybit.get_klines(symbol, timeframe, limit=200)
                    
                    # Retry once if no data
                    if not klines:
                        logger.info(f"[{symbol}] Retrying API fetch...")
                        await asyncio.sleep(1)
                        klines = self.bybit.get_klines(symbol, timeframe, limit=200)
                
                    if klines:
                        # Convert to DataFrame
                        data = []
                        for k in klines:
                            # k = [timestamp, open, high, low, close, volume, turnover]
                            data.append({
                                'open': float(k[1]),
                                'high': float(k[2]),
                                'low': float(k[3]),
                                'close': float(k[4]),
                                'volume': float(k[5])
                            })
                        
                        df = pd.DataFrame(data)
                        # Set index to timestamp
                        df.index = pd.to_datetime([int(k[0]) for k in klines], unit='ms', utc=True)
                        df.sort_index(inplace=True)
                        
                        self.frames[symbol] = df
                        logger.info(f"[{symbol}] Fetched {len(df)} candles from API")
                        
                        # Save to database for next time
                        self.storage.save_candles(symbol, df)
                    else:
                        self.frames[symbol] = new_frame()
                        logger.warning(f"[{symbol}] No data available from API")
                        
                except Exception as e:
                    logger.error(f"[{symbol}] Failed to fetch data: {e}")
                    self.frames[symbol] = new_frame()
        
        # Save all fetched data immediately
        if self.frames:
            logger.info("Saving initial data to database...")
            self.storage.save_all_frames(self.frames)
        
        # Show database stats
        stats = self.storage.get_stats()
        logger.info(f"Database: {stats.get('total_candles', 0)} candles, {stats.get('symbols', 0)} symbols, {stats.get('db_size_mb', 0):.2f} MB")

    
    async def save_all_candles(self):
        """Save all candles to database"""
        try:
            if self.frames:
                loop = asyncio.get_running_loop()
                # Offload full save to a background thread to avoid blocking event loop
                await loop.run_in_executor(None, self.storage.save_all_frames, self.frames)
                logger.info("Auto-saved all candles to database (bg thread)")
        except Exception as e:
            logger.error(f"Failed to auto-save candles: {e}")

    async def backfill_frames_3m(self, symbols: list[str], use_api_fallback: bool = True, limit: int = 200):
        """Backfill 3m frames from DB; if missing and allowed, fetch from Bybit REST.

        - Loads up to `limit` recent 3m candles from the database for each symbol.
        - If no 3m data is present and API fallback is enabled, fetch recent 3m klines
          via REST and seed frames_3m, then persist to the 3m table.
        """
        loaded_db = 0
        fetched_api = 0
        for idx, symbol in enumerate(symbols):
            try:
                df3 = self.storage.load_candles_3m(symbol, limit=limit)
                if df3 is not None and not df3.empty:
                    self.frames_3m[symbol] = df3.tail(1000)
                    loaded_db += 1
                    continue
                if not use_api_fallback:
                    continue
                # Rate-limit API calls lightly for many symbols
                if idx > 0 and len(symbols) > 25:
                    await asyncio.sleep(0.05)
                # Fetch recent 3m klines from REST (max 200 per request)
                kl = self.bybit.get_klines(symbol, '3', limit=limit)
                if kl:
                    data = []
                    for k in kl:
                        data.append({
                            'open': float(k[1]),
                            'high': float(k[2]),
                            'low': float(k[3]),
                            'close': float(k[4]),
                            'volume': float(k[5])
                        })
                    import pandas as pd
                    df = pd.DataFrame(data)
                    df.index = pd.to_datetime([int(k[0]) for k in kl], unit='ms', utc=True)
                    df.sort_index(inplace=True)
                    self.frames_3m[symbol] = df.tail(1000)
                    # Persist to 3m table for future startups
                    self.storage.save_candles_3m(symbol, df)
                    fetched_api += 1
            except Exception as e:
                logger.debug(f"3m backfill error for {symbol}: {e}")
                continue
        if loaded_db or fetched_api:
            logger.info(f"🩳 3m backfill complete: DB={loaded_db} symbols, API={fetched_api} symbols")
        else:
            logger.info("🩳 3m backfill: no data found (DB empty, API disabled or unavailable)")
    
    def record_closed_trade(self, symbol: str, pos: Position, exit_price: float, exit_reason: str, leverage: float = 1.0):
        """Record a closed trade to history"""
        try:
            # Calculate PnL
            pnl_usd, pnl_percent = self.trade_tracker.calculate_pnl(
                symbol, pos.side, pos.entry, exit_price, pos.qty, leverage
            )
            
            # Create trade record
            trade = Trade(
                symbol=symbol,
                side=pos.side,
                entry_price=pos.entry,
                exit_price=exit_price,
                quantity=pos.qty,
                entry_time=pos.entry_time,  # Use entry time from position
                exit_time=datetime.now(),
                pnl_usd=pnl_usd,
                pnl_percent=pnl_percent,
                exit_reason=exit_reason,
                leverage=leverage,
                strategy_name=pos.strategy_name # Pass strategy name
            )
            
            # Add to tracker
            self.trade_tracker.add_trade(trade)
            logger.info(f"Trade recorded: {symbol} {exit_reason} PnL: ${pnl_usd:.2f}")

            if self.tg:
                try:
                    hold_minutes = 0
                    if pos.entry_time:
                        hold_minutes = int((datetime.now() - pos.entry_time).total_seconds() // 60)
                    outcome_emoji = "✅" if pnl_usd >= 0 else "❌"
                    exit_label_map = {
                        "tp": "Take Profit",
                        "sl": "Stop Loss",
                        "timeout": "Timeout",
                        "manual": "Manual",
                        "unknown": "Unknown"
                    }
                    exit_label = exit_label_map.get(exit_reason.lower(), exit_reason.upper()) if isinstance(exit_reason, str) else str(exit_reason)
                    ml_details = ""
                    if hasattr(pos, 'ml_score') and pos.ml_score:
                        ml_details = f"ML Score: {pos.ml_score:.1f}\n"

                    strategy_label = getattr(pos, 'strategy_name', 'unknown')
                    if isinstance(strategy_label, str):
                        if strategy_label == 'trend_breakout':
                            strategy_label = 'trend_pullback'
                        strategy_label = strategy_label.replace('_', ' ').title()
                    # Include realized R multiple for clarity vs 1% risk
                    realized_r = None
                    try:
                        R = (pos.entry - pos.sl) if pos.side == 'long' else (pos.sl - pos.entry)
                        if R and R != 0:
                            realized_r = (exit_price - pos.entry)/R if pos.side=='long' else (pos.entry - exit_price)/R
                    except Exception:
                        realized_r = None
                    rr_line = f"Realized R: {realized_r:.2f}R\n" if isinstance(realized_r, float) else ""
                    # Include Qscore if known
                    try:
                        qv = float(getattr(pos, 'qscore', 0.0) or 0.0)
                    except Exception:
                        try:
                            qv = float(((getattr(self, '_last_signal_features', {}) or {}).get(symbol, {}) or {}).get('qscore', 0.0) or 0.0)
                        except Exception:
                            qv = 0.0
                    q_line = f"Q: {qv:.1f}\n" if qv else ""
                    message = (
                        f"{outcome_emoji} *Trade Closed* {symbol} {pos.side.upper()}\n\n"
                        f"Exit Price: {exit_price:.4f}\n"
                        f"PnL: ${pnl_usd:.2f} ({pnl_percent:.2f}%)\n"
                        f"{rr_line}"
                        f"{q_line}"
                        f"Hold: {hold_minutes}m\n"
                        f"Exit: {exit_label}\n"
                        f"Strategy: {strategy_label}\n"
                        f"{ml_details}"
                    )
                    asyncio.create_task(self.tg.send_message(message.strip()))
                except Exception as notify_err:
                    logger.warning(f"Failed to send Telegram close notification: {notify_err}")

        except Exception as e:
            logger.error(f"Failed to record trade: {e}")
    
    async def check_closed_positions(self, book: Book, meta: dict = None, ml_scorer=None, reset_symbol_state=None, symbol_collector=None):
        """Check for positions that have been closed and record them"""
        try:
            # Get current positions from exchange
            current_positions = self.bybit.get_positions()
            if current_positions is None:
                logger.warning("Could not get positions from exchange, skipping closed position check")
                return
            
            # Build set of symbols with CONFIRMED open positions
            current_symbols = set()
            for p in current_positions:
                symbol = p.get('symbol')
                size = float(p.get('size', 0))
                # Only add if we're SURE there's an open position
                if symbol and size > 0:
                    current_symbols.add(symbol)
            
            # Find positions that might be closed
            potentially_closed = []
            for symbol, pos in list(book.positions.items()):
                if symbol not in current_symbols:
                    potentially_closed.append((symbol, pos))
            
            # Verify each potentially closed position
            confirmed_closed = []
            for symbol, pos in potentially_closed:
                try:
                    # Try to get recent order history to confirm close
                    resp = self.bybit._request("GET", "/v5/order/history", {
                        "category": "linear",
                        "symbol": symbol,
                        "limit": 20
                    })
                    orders = resp.get("result", {}).get("list", [])
                    
                    # Look for a FILLED reduce-only order (closing order)
                    found_close = False
                    exit_price = 0
                    exit_reason = "unknown"
                    
                    for order in orders:
                        # Check if this is a closing order
                        if (order.get("reduceOnly") == True and 
                            order.get("orderStatus") == "Filled"):
                            
                            found_close = True
                            # Handle empty strings and None values
                            avg_price_str = order.get("avgPrice", 0)
                            exit_price = float(avg_price_str) if avg_price_str and avg_price_str != "" else 0
                            
                            # Log order details for debugging
                            logger.debug(f"[{symbol}] Found filled reduceOnly order: avgPrice={avg_price_str}, triggerPrice={order.get('triggerPrice')}, orderType={order.get('orderType')}")
                            
                            # Determine if it was TP or SL based on trigger price
                            trigger_price_str = order.get("triggerPrice", 0)
                            trigger_price = float(trigger_price_str) if trigger_price_str and trigger_price_str != "" else 0
                            
                            # Also check orderType for better detection
                            order_type = order.get("orderType", "")
                            
                            if trigger_price > 0:
                                if pos.side == "long":
                                    if trigger_price >= pos.tp * 0.997:  # Within 0.3% of TP
                                        exit_reason = "tp"
                                    elif trigger_price <= pos.sl * 1.003:  # Within 0.3% of SL
                                        exit_reason = "sl"
                                else:  # short
                                    if trigger_price <= pos.tp * 1.003:
                                        exit_reason = "tp"
                                    elif trigger_price >= pos.sl * 0.997:
                                        exit_reason = "sl"
                            
                            # Check orderType as fallback
                            if exit_reason == "unknown":
                                if "TakeProfit" in order_type:
                                    exit_reason = "tp"
                                elif "StopLoss" in order_type:
                                    exit_reason = "sl"
                            
                            # If no trigger price, check exit price vs targets
                            if exit_reason == "unknown" and exit_price > 0:
                                # Log position targets for debugging
                                logger.debug(f"[{symbol}] Checking exit price {exit_price:.4f} vs targets - TP: {pos.tp:.4f}, SL: {pos.sl:.4f}, Side: {pos.side}")
                                
                                if pos.side == "long":
                                    if exit_price >= pos.tp * 0.997:
                                        exit_reason = "tp"
                                        logger.debug(f"[{symbol}] Long TP hit: exit {exit_price:.4f} >= TP {pos.tp:.4f} * 0.997")
                                    elif exit_price <= pos.sl * 1.003:
                                        exit_reason = "sl"
                                        logger.debug(f"[{symbol}] Long SL hit: exit {exit_price:.4f} <= SL {pos.sl:.4f} * 1.003")
                                    else:
                                        exit_reason = "manual"
                                        logger.debug(f"[{symbol}] Long manual close: exit {exit_price:.4f} between SL {pos.sl:.4f} and TP {pos.tp:.4f}")
                                else:  # short
                                    if exit_price <= pos.tp * 1.003:
                                        exit_reason = "tp"
                                        logger.debug(f"[{symbol}] Short TP hit: exit {exit_price:.4f} <= TP {pos.tp:.4f} * 1.003")
                                    elif exit_price >= pos.sl * 0.997:
                                        exit_reason = "sl"
                                        logger.debug(f"[{symbol}] Short SL hit: exit {exit_price:.4f} >= SL {pos.sl:.4f} * 0.997")
                                    else:
                                        exit_reason = "manual"
                                        logger.debug(f"[{symbol}] Short manual close: exit {exit_price:.4f} between TP {pos.tp:.4f} and SL {pos.sl:.4f}")
                            break
                    
                    # Re-confirm with live position size to avoid false positives due to API lag/partials
                    if found_close:
                        try:
                            pos_live = self.bybit.get_position(symbol)
                            live_size = 0.0
                            if isinstance(pos_live, dict):
                                live_size = float(pos_live.get('size', 0) or 0)
                            if live_size > 0:
                                logger.info(f"[{symbol}] Close evidence found but size still {live_size}; skipping record this cycle")
                                found_close = False
                        except Exception as _cp:
                            logger.debug(f"[{symbol}] Live size check failed: {_cp}")
                    # Only add to confirmed closed if we found evidence and have exit price
                    if found_close and exit_price > 0:
                        logger.info(f"[{symbol}] Position CONFIRMED closed at {exit_price:.4f} ({exit_reason})")
                        confirmed_closed.append((symbol, pos, exit_price, exit_reason))
                    else:
                        # Can't confirm it's closed - might be API lag
                        logger.debug(f"[{symbol}] Not in positions but can't confirm close - keeping in book")
                        
                except Exception as e:
                    logger.warning(f"Could not verify {symbol} close status: {e}")
                    # Don't process if we can't verify
            
            # Process only CONFIRMED closed positions
            for symbol, pos, exit_price, exit_reason in confirmed_closed:
                try:
                    # Get leverage
                    leverage = meta.get(symbol, {}).get("max_leverage", 1.0) if meta else 1.0
                    
                    # Normalize exit_reason for both ML and trade record consistency
                    try:
                        # Compute PnL pct and proximity to TP/SL
                        if pos.side == "long":
                            pnl_pct_tmp = ((exit_price - pos.entry) / pos.entry) * 100
                            tp_distance = abs(exit_price - pos.tp)
                            sl_distance = abs(exit_price - pos.sl)
                        else:
                            pnl_pct_tmp = ((pos.entry - exit_price) / pos.entry) * 100
                            tp_distance = abs(exit_price - pos.tp)
                            sl_distance = abs(exit_price - pos.sl)
                        # Reclassify manual closes near targets
                        if exit_reason == "manual":
                            if tp_distance < sl_distance and pnl_pct_tmp > 0:
                                exit_reason = "tp"
                                logger.info(f"[{symbol}] Manual close near TP with profit - treating as TP for records")
                            elif sl_distance < tp_distance and pnl_pct_tmp < 0:
                                exit_reason = "sl"
                                logger.info(f"[{symbol}] Manual close near SL with loss - treating as SL for records")
                        # Sanity-correct mismatched labels vs PnL
                        if exit_reason == "tp" and pnl_pct_tmp < 0:
                            exit_reason = "sl"
                        elif exit_reason == "sl" and pnl_pct_tmp > 0:
                            exit_reason = "tp"
                    except Exception:
                        pass

                    # If strategy_name is unknown (e.g., after restart), try recover from Redis hint
                    try:
                        if getattr(self, '_redis', None) is not None and getattr(pos, 'strategy_name', 'unknown') == 'unknown':
                            v = self._redis.get(f'openpos:strategy:{symbol}')
                            if isinstance(v, str) and v:
                                pos.strategy_name = v
                                logger.info(f"[{symbol}] Recovered strategy from Redis for close record: {v}")
                    except Exception:
                        pass

                    # Record the trade
                    self.record_closed_trade(symbol, pos, exit_price, exit_reason, leverage)
                    # Telegram close notify (all strategies)
                    try:
                        if self.tg:
                            emoji = '✅' if exit_reason == 'tp' else ('❌' if exit_reason == 'sl' else '🔚')
                            strat = str(getattr(pos, 'strategy_name', 'unknown')).replace('_',' ').title()
                            await self.tg.send_message(
                                f"{emoji} {symbol} CLOSED ({strat})\n"
                                f"Side: {pos.side.upper()}\n"
                                f"Entry: {float(pos.entry):.4f} → Exit: {float(exit_price):.4f}\n"
                                f"Reason: {exit_reason.upper()}"
                            )
                    except Exception:
                        pass
                    # Force-close executed phantom mirrors to align with exchange closure (Trend + MR)
                    try:
                        if pos.strategy_name in ['enhanced_mr', 'mean_reversion'] and mr_phantom_tracker is not None:
                            mr_phantom_tracker.force_close_executed(symbol, exit_price, exit_reason)
                        elif pos.strategy_name in ('trend_pullback','trend_breakout') and phantom_tracker is not None:
                            phantom_tracker.force_close_executed(symbol, exit_price, exit_reason)
                        elif pos.strategy_name == 'scalp':
                            try:
                                from scalp_phantom_tracker import get_scalp_phantom_tracker
                                scpt = get_scalp_phantom_tracker()
                                scpt.force_close_executed(symbol, exit_price, exit_reason)
                            except Exception as _sce:
                                logger.debug(f"[{symbol}] scalp force_close_executed failed: {_sce}")
                    except Exception as _fce:
                        logger.debug(f"[{symbol}] phantom force_close_executed failed: {_fce}")
                    
                    # Update symbol data collector with session performance
                    if symbol_collector:
                        try:
                            # Calculate hold time
                            hold_minutes = (datetime.now() - pos.entry_time).total_seconds() / 60
                            
                            # Calculate P&L for this position
                            if pos.side == "long":
                                pnl = (exit_price - pos.entry) * pos.qty
                            else:
                                pnl = (pos.entry - exit_price) * pos.qty
                            
                            # Determine session
                            session = symbol_collector.get_trading_session(datetime.now().hour)
                            
                            # Update session performance
                            symbol_collector.update_session_performance(
                                symbol=symbol,
                                session=session,
                                won=pnl > 0,
                                pnl=pnl,
                                hold_minutes=int(hold_minutes)
                            )
                            
                            logger.debug(f"[{symbol}] Updated session stats: {session}, PnL={pnl:.2f}, Hold={hold_minutes:.0f}min")
                            
                        except Exception as e:
                            logger.debug(f"Failed to update session performance: {e}")
                    
                    # Update ML scorer for all closed positions
                    if ml_scorer is not None:
                        # Always record outcome for ML learning (not just TP/SL)
                        # This ensures we learn from ALL trades including manual closes
                        try:
                            # Calculate P&L percentage
                            if pos.side == "long":
                                pnl_pct = ((exit_price - pos.entry) / pos.entry) * 100
                                # Debug logging for incorrect P&L
                                if exit_price > pos.entry and pnl_pct < 0:
                                    logger.error(f"[{symbol}] CALCULATION ERROR: Long position exit {exit_price:.4f} > entry {pos.entry:.4f} but P&L is {pnl_pct:.2f}%")
                            else:
                                pnl_pct = ((pos.entry - exit_price) / pos.entry) * 100
                                # Debug logging for incorrect P&L
                                if exit_price < pos.entry and pnl_pct < 0:
                                    logger.error(f"[{symbol}] CALCULATION ERROR: Short position exit {exit_price:.4f} < entry {pos.entry:.4f} but P&L is {pnl_pct:.2f}%")
                            
                            # CRITICAL FIX: Use actual P&L to determine win/loss, not exit_reason
                            # Negative P&L is ALWAYS a loss, positive is ALWAYS a win
                            outcome = "win" if pnl_pct > 0 else "loss"
                            
                            # Log warning if exit_reason doesn't match P&L for TP/SL
                            if exit_reason == "tp":
                                if pnl_pct < 0:
                                    logger.warning(f"[{symbol}] TP hit but negative P&L! Side: {pos.side}, Exit: {exit_price:.4f}, Entry: {pos.entry:.4f}, P&L: {pnl_pct:.2f}%")
                                    exit_reason = "sl"  # Correct the exit reason based on actual P&L
                            elif exit_reason == "sl":
                                if pnl_pct > 0:
                                    logger.warning(f"[{symbol}] SL hit but positive P&L! Exit: {exit_price:.4f}, Entry: {pos.entry:.4f}, P&L: {pnl_pct:.2f}%")
                                    exit_reason = "tp"  # Correct the exit reason based on actual P&L
                            
                            # For manual closes, try to determine if it was more like TP or SL based on P&L
                            if exit_reason == "manual":
                                # Check if the position's SL/TP values are valid
                                if pos.sl > 0 and pos.tp > 0:
                                    # For manual, still check proximity to targets
                                    if pos.side == "long":
                                        # If closer to TP than SL and profitable, likely a partial TP
                                        tp_distance = abs(exit_price - pos.tp)
                                        sl_distance = abs(exit_price - pos.sl)
                                        if tp_distance < sl_distance and pnl_pct > 0:
                                            logger.info(f"[{symbol}] Manual close near TP with profit - treating as TP for ML")
                                            exit_reason = "tp"
                                        elif sl_distance < tp_distance and pnl_pct < 0:
                                            logger.info(f"[{symbol}] Manual close near SL with loss - treating as SL for ML")
                                            exit_reason = "sl"
                                    else:  # short
                                        tp_distance = abs(exit_price - pos.tp)
                                        sl_distance = abs(exit_price - pos.sl)
                                        if tp_distance < sl_distance and pnl_pct > 0:
                                            logger.info(f"[{symbol}] Manual close near TP with profit - treating as TP for ML")
                                            exit_reason = "tp"
                                        elif sl_distance < tp_distance and pnl_pct < 0:
                                            logger.info(f"[{symbol}] Manual close near SL with loss - treating as SL for ML")
                                            exit_reason = "sl"
                            
                            # Create signal data for ML recording
                            try:
                                feat_ref = self._last_signal_features.pop(symbol, {})
                            except Exception:
                                feat_ref = {}
                            # Attach last captured features and version tags per strategy
                            # Default feature_version added when missing to help downstream schema tracking
                            try:
                                if isinstance(feat_ref, dict):
                                    if str(getattr(pos, 'strategy_name', '')).lower() in ('trend_pullback','trend_breakout'):
                                        feat_ref.setdefault('feature_version', 'trend_v1')
                                    elif str(getattr(pos, 'strategy_name', '')).lower() in ('enhanced_mr','mean_reversion'):
                                        feat_ref.setdefault('feature_version', 'mr_v1')
                            except Exception:
                                pass

                            signal_data = {
                                'symbol': symbol,
                                'features': feat_ref,
                                'score': 0,
                                'was_executed': True,
                                'meta': {
                                    'reason': getattr(pos, 'ml_reason', '')
                                }
                            }

                            # Enrich features with lifecycle flags from scale-out (if present)
                            try:
                                so = getattr(self, '_scaleout', {}).get(symbol) if hasattr(self, '_scaleout') else None
                                if isinstance(so, dict):
                                    feat_ref['tp1_hit'] = 1.0 if bool(so.get('tp1_hit')) else 0.0
                                    feat_ref['be_moved'] = 1.0 if bool(so.get('be_moved')) else 0.0
                                    # Runner hit is true if exit_reason is TP and be_moved was true
                                    try:
                                        feat_ref['runner_hit'] = 1.0 if (str(exit_reason).lower() == 'tp' and bool(so.get('be_moved'))) else 0.0
                                    except Exception:
                                        feat_ref['runner_hit'] = 0.0
                                    # Timing features
                                    try:
                                        if 'tp1_time' in so and pos.entry_time:
                                            feat_ref['time_to_tp1_sec'] = float((so['tp1_time'] - pos.entry_time).total_seconds())
                                        else:
                                            feat_ref['time_to_tp1_sec'] = 0.0
                                    except Exception:
                                        feat_ref['time_to_tp1_sec'] = 0.0
                                    try:
                                        if pos.entry_time:
                                            feat_ref['time_to_exit_sec'] = float((datetime.now() - pos.entry_time).total_seconds())
                                        else:
                                            feat_ref['time_to_exit_sec'] = 0.0
                                    except Exception:
                                        feat_ref['time_to_exit_sec'] = 0.0
                            except Exception:
                                pass
                            
                            # Debugging: Log strategy name and routing info for all closed trades
                            logger.info(f"[{symbol}] ML ROUTING DEBUG: strategy='{pos.strategy_name}', use_enhanced={shared.get('use_enhanced_parallel', False) if 'shared' in locals() else False}")

                            # Record outcome in appropriate ML scorer based on strategy (NO DUPLICATION)
                            # IMPORTANT: skip ML training updates for manual closes
                            skip_ml_update_manual = False
                            if str(exit_reason).lower() == 'manual':
                                logger.info(f"[{symbol}] ML SKIP: Manual close — not used for training (strategy={pos.strategy_name})")
                                # Still log and proceed without training updates
                                skip_ml_update_manual = True
                            # Get shared data components for ML scorers
                            shared_enhanced_mr = shared.get('enhanced_mr_scorer') if 'shared' in locals() else None
                            shared_mr_scorer = shared.get('mean_reversion_scorer') if 'shared' in locals() else None
                            use_enhanced = shared.get('use_enhanced_parallel', False) if 'shared' in locals() else False

                            logger.info(f"[{symbol}] ML COMPONENTS: enhanced_mr={shared_enhanced_mr is not None}, mr_scorer={shared_mr_scorer is not None}, use_enhanced={use_enhanced}")
                            logger.info(f"[{symbol}] ML ROUTING DECISION: Enhanced system active: {use_enhanced and shared_enhanced_mr}, Strategy: '{pos.strategy_name}'")

                            # MR Promotion flag from Redis for resilience across restarts
                            mr_promo_flag = False
                            try:
                                if getattr(self, '_redis', None) is not None and pos.strategy_name in ["enhanced_mr", "mean_reversion"]:
                                    mr_promo_flag = (self._redis.get(f'openpos:mr_promotion:{symbol}') == '1')
                            except Exception:
                                mr_promo_flag = False

                            if skip_ml_update_manual:
                                pass
                            elif use_enhanced and shared_enhanced_mr:
                                # Enhanced parallel system - STRICT routing guard
                                logger.info(f"[{symbol}] 🎯 ML ROUTING: strategy_name='{pos.strategy_name}', outcome='{outcome}'")
                                if pos.strategy_name == "enhanced_mr":
                                    if mr_promo_flag:
                                        logger.info(f"[{symbol}] MR Promotion was active — skipping executed MR ML update (phantom path will learn)")
                                        try:
                                            if mr_phantom_tracker is not None:
                                                mr_phantom_tracker.update_mr_phantom_prices(symbol, exit_price, df=self.frames.get(symbol))
                                        except Exception:
                                            pass
                                    else:
                                        try:
                                            signal_data['was_executed'] = True
                                        except Exception:
                                            pass
                                        shared_enhanced_mr.record_outcome(signal_data, outcome, pnl_pct)
                                        logger.info(f"[{symbol}] ✅ Enhanced MR ML updated with outcome.")
                                elif pos.strategy_name == "mean_reversion":
                                    # Guard: do NOT route 'mean_reversion' to Enhanced MR to avoid accidental increments
                                    if shared_mr_scorer is not None:
                                        try:
                                            signal_data['was_executed'] = True
                                        except Exception:
                                            pass
                                        shared_mr_scorer.record_outcome(signal_data, outcome, pnl_pct)
                                        logger.info(f"[{symbol}] ✅ Original MR ML updated with outcome (guarded).")
                                    else:
                                        logger.warning(f"[{symbol}] ⚠️ Mean Reversion outcome not recorded (no original MR scorer active; guard preventing Enhanced MR increment)")
                                elif str(getattr(pos, 'strategy_name', '')).lower() == "unknown":
                                    # Recovered position - try Redis hint first, else check reason text
                                    try:
                                        if getattr(self, '_redis', None) is not None:
                                            v = self._redis.get(f'openpos:strategy:{symbol}')
                                            if isinstance(v, str) and v:
                                                pos.strategy_name = v
                                                logger.info(f"[{symbol}] Inferred strategy from Redis: {v}")
                                    except Exception:
                                        pass
                                    reason = signal_data.get('meta', {}).get('reason', '')
                                    logger.info(f"[{symbol}] 🔍 UNKNOWN STRATEGY - Checking reason: '{reason}'")
                                    if 'Mean Reversion:' in reason or 'Rejection from resistance' in reason or 'Rejection from support' in reason:
                                        # Treat as MR, but guard against Enhanced MR increments if original scorer absent
                                        if shared_mr_scorer is not None:
                                            try:
                                                signal_data['was_executed'] = True
                                            except Exception:
                                                pass
                                            shared_mr_scorer.record_outcome(signal_data, outcome, pnl_pct)
                                            logger.info(f"[{symbol}] ✅ Original MR ML updated with outcome (recovered, inferred).")
                                        else:
                                            logger.warning(f"[{symbol}] ⚠️ Inferred MR outcome not recorded (no original MR scorer; guard active)")
                                    else:
                                        # Do not default unknown to any ML. Log and skip.
                                        logger.info(f"[{symbol}] 🛑 Unknown strategy — skipping ML update (no default routing)")
                                else:
                                    # Range route or Trend ML
                                    strat_l = str(getattr(pos, 'strategy_name','')).lower()
                                    if strat_l.startswith('range'):
                                        try:
                                            from ml_scorer_range import get_range_scorer
                                            rml = get_range_scorer()
                                            try:
                                                signal_data['was_executed'] = True
                                            except Exception:
                                                pass
                                            rml.record_outcome(signal_data, outcome, pnl_pct)
                                            logger.info(f"[{symbol}] Range ML updated with outcome.")
                                        except Exception as e:
                                            logger.debug(f"[{symbol}] Range ML update failed: {e}")
                                    else:
                                        # Route only if clearly Trend; never route Scalp or unknown/other to Trend ML
                                        strat_l = str(getattr(pos, 'strategy_name','')).lower()
                                        if strat_l == 'scalp':
                                            logger.info(f"[{symbol}] ⚙️ Scalp outcome handled by Scalp phantom tracker; skipping ML update")
                                        elif strat_l in ('trend_pullback','trend_breakout'):
                                            logger.info(f"[{symbol}] 🔵 TREND STRATEGY detected")
                                            if ml_scorer is not None:
                                                try:
                                                    signal_data['was_executed'] = True
                                                except Exception:
                                                    pass
                                                ml_scorer.record_outcome(signal_data, outcome, pnl_pct)
                                                logger.info(f"[{symbol}] Trend ML updated with outcome.")
                                        else:
                                            logger.info(f"[{symbol}] 🛑 Non-trend strategy '{strat_l}' — skipping Trend ML update")
                            else:
                                # Robust routing by strategy_name when enhanced path isn't active
                                if not skip_ml_update_manual:
                                    is_mr_trade = str(getattr(pos, 'strategy_name', '')).lower() in ("enhanced_mr", "mean_reversion")
                                    if is_mr_trade:
                                        if mr_promo_flag:
                                            logger.info(f"[{symbol}] MR Promotion was active — skipping executed MR ML update (phantom path will learn)")
                                            try:
                                                if mr_phantom_tracker is not None:
                                                    mr_phantom_tracker.update_mr_phantom_prices(symbol, exit_price, df=self.frames.get(symbol))
                                            except Exception:
                                                pass
                                        else:
                                            routed = False
                                            # Prefer Enhanced MR scorer if available even when use_enhanced flag is false
                                            if shared_enhanced_mr is not None:
                                                try:
                                                    try:
                                                        signal_data['was_executed'] = True
                                                    except Exception:
                                                        pass
                                                    shared_enhanced_mr.record_outcome(signal_data, outcome, pnl_pct)
                                                    logger.info(f"[{symbol}] Enhanced MR ML updated with outcome (fallback path).")
                                                    routed = True
                                                except Exception as e:
                                                    logger.warning(f"[{symbol}] Enhanced MR ML update failed: {e}")
                                            if (not routed) and (shared_mr_scorer is not None):
                                                try:
                                                    try:
                                                        signal_data['was_executed'] = True
                                                    except Exception:
                                                        pass
                                                    shared_mr_scorer.record_outcome(signal_data, outcome, pnl_pct)
                                                    logger.info(f"[{symbol}] Original MR ML updated with outcome (fallback path).")
                                                    routed = True
                                                except Exception as e:
                                                    logger.warning(f"[{symbol}] Original MR ML update failed: {e}")
                                            if not routed:
                                                try:
                                                    if getattr(self, '_redis', None) is not None:
                                                        import json
                                                        rec = {
                                                            'symbol': symbol,
                                                            'outcome': outcome,
                                                            'pnl_pct': float(pnl_pct),
                                                            'features': signal_data.get('features', {}),
                                                            'ts': datetime.utcnow().isoformat(),
                                                            'was_executed': True
                                                        }
                                                        self._redis.rpush('ml:deferred:mr_outcomes', json.dumps(rec))
                                                        self._redis.ltrim('ml:deferred:mr_outcomes', -1000, -1)
                                                        logger.info(f"[{symbol}] MR ML skipped: no scorer attached — deferred to Redis queue")
                                                    else:
                                                        logger.warning(f"[{symbol}] MR ML skipped: no scorer attached and no Redis available")
                                                except Exception as e:
                                                    logger.warning(f"[{symbol}] MR deferred enqueue failed: {e}")
                                    else:
                                        # Trend strategy
                                        logger.info(f"[{symbol}] 🔵 TREND STRATEGY detected")
                                        if ml_scorer is not None:
                                            try:
                                                ml_scorer.record_outcome(signal_data, outcome, pnl_pct)
                                                logger.info(f"[{symbol}] Trend ML updated with outcome.")
                                            except Exception as e:
                                                logger.warning(f"[{symbol}] Trend ML update failed: {e}")

                            
                            # Log with clear outcome based on actual P&L and corrected exit reason
                            actual_result = "WIN" if pnl_pct > 0 else "LOSS"
                            logger.info(f"[{symbol}] ML updated: {actual_result} ({pnl_pct:.2f}%) - Exit trigger: {exit_reason.upper()}")
                            
                            # Check if ML needs retraining after this trade completes
                            self._check_ml_retrain(ml_scorer)

                            # Also check mean reversion ML retraining (both enhanced and original systems)
                            if pos.strategy_name in ["enhanced_mr", "mean_reversion"]:
                                # Get the appropriate MR scorer references
                                enhanced_mr_scorer_ref = shared_enhanced_mr if use_enhanced and shared_enhanced_mr else None
                                original_mr_scorer_ref = shared_mr_scorer if not use_enhanced else None
                                
                                # Call the appropriate retrain check
                                self._check_mr_ml_retrain(original_mr_scorer_ref, enhanced_mr_scorer_ref)
                            
                        except Exception as e:
                            logger.error(f"Failed to update ML outcome: {e}")
                    
                    # Log position details for debugging
                    logger.debug(f"[{symbol}] Closed position details: side={pos.side}, entry={pos.entry:.4f}, exit={exit_price:.4f}, TP={pos.tp:.4f}, SL={pos.sl:.4f}, exit_reason={exit_reason}")
                    
                    # Remove from book
                    book.positions.pop(symbol)
                    # Cleanup runtime hints for this symbol
                    try:
                        if getattr(self, '_redis', None) is not None:
                            self._redis.delete(f'openpos:mr_promotion:{symbol}')
                            self._redis.delete(f'openpos:reason:{symbol}')
                    except Exception:
                        pass
                    logger.info(f"[{symbol}] Position removed from tracking")
                    # Cleanup runtime strategy hint for this symbol
                    try:
                        if getattr(self, '_redis', None) is not None:
                            self._redis.delete(f'openpos:strategy:{symbol}')
                    except Exception:
                        pass
                    
                    # Reset the strategy state
                    if reset_symbol_state:
                        reset_symbol_state(symbol)
                        logger.info(f"[{symbol}] Strategy state reset - ready for new signals")
                        
                except Exception as e:
                    logger.error(f"Error processing confirmed closed position {symbol}: {e}")
                    
        except Exception as e:
            logger.error(f"Error in check_closed_positions: {e}")
    
    def _check_ml_retrain(self, ml_scorer):
        """Check if ML needs retraining after a trade completes"""
        if not ml_scorer:
            return
            
        try:
            # Get retrain info
            retrain_info = ml_scorer.get_retrain_info()
            
            # Check if ready to retrain
            if retrain_info['can_train'] and retrain_info['trades_until_next_retrain'] == 0:
                total_trades = retrain_info.get('total_combined', retrain_info.get('total_records', retrain_info.get('total_trades', 'N/A')))
                logger.info(f"🔄 ML retrain triggered after real trade completion - {total_trades} total trades available")
                
                # Trigger retrain
                retrain_result = ml_scorer.startup_retrain()
                if retrain_result:
                    logger.info("✅ ML models successfully retrained after trade completion")
                else:
                    logger.warning("⚠️ ML retrain attempt failed")
            else:
                logger.debug(f"ML retrain check: {retrain_info['trades_until_next_retrain']} trades until next retrain")
                
        except Exception as e:
            logger.error(f"Error checking ML retrain: {e}")

    def _check_mr_ml_retrain(self, mean_reversion_scorer=None, enhanced_mr_scorer=None):
        """Check if Mean Reversion ML needs retraining after a trade completes"""
        # Use the appropriate MR scorer based on system configuration
        mr_scorer = enhanced_mr_scorer if enhanced_mr_scorer else mean_reversion_scorer
        
        if not mr_scorer:
            return

        try:
            # Get retrain info from the active MR scorer
            retrain_info = mr_scorer.get_retrain_info()
            
            scorer_type = "Enhanced MR" if enhanced_mr_scorer else "Original MR"

            # Check if ready to retrain
            if retrain_info['can_train'] and retrain_info['trades_until_next_retrain'] == 0:
                total_trades = retrain_info.get('total_combined', retrain_info.get('total_trades', 0))
                logger.info(f"🔄 {scorer_type} ML retrain triggered after trade completion - "
                           f"{total_trades} total trades available")

                # Trigger retrain
                retrain_result = mr_scorer.startup_retrain()
                if retrain_result:
                    logger.info(f"✅ {scorer_type} ML models successfully retrained after trade completion")
                else:
                    logger.warning(f"⚠️ {scorer_type} ML retrain attempt failed")
            else:
                logger.debug(f"{scorer_type} ML retrain check: {retrain_info['trades_until_next_retrain']} trades until next retrain")

        except Exception as e:
            logger.error(f"Error checking {scorer_type} ML retrain: {e}")

    async def recover_positions(self, book:Book, sizer:Sizer):
        """Recover existing positions from exchange - preserves all existing orders"""
        logger.info("Checking for existing positions to recover...")
        
        try:
            positions = self.bybit.get_positions()
            
            if positions:
                recovered = 0
                recovered_details = []
                for pos in positions:
                    # Skip if no position size
                    size_str = pos.get('size', '0')
                    if not size_str or size_str == '' or float(size_str) == 0:
                        continue
                        
                    symbol = pos['symbol']
                    side = "long" if pos['side'] == "Buy" else "short"
                    
                    # Safe conversion with empty string handling
                    qty = float(size_str)
                    entry = float(pos.get('avgPrice') or 0)
                    
                    # Get current TP/SL if set - PRESERVE THESE
                    # Handle empty strings by converting to 0
                    tp_str = pos.get('takeProfit', '0')
                    sl_str = pos.get('stopLoss', '0')
                    tp = float(tp_str) if tp_str and tp_str != '' else 0
                    sl = float(sl_str) if sl_str and sl_str != '' else 0
                    
                    # For recovered positions, validate TP/SL make sense
                    if side == "long":
                        # For long: TP should be > entry, SL should be < entry
                        if tp > 0 and sl > 0 and tp < sl:
                            logger.warning(f"[{symbol}] TP/SL appear swapped for long position! TP={tp:.4f} SL={sl:.4f} Entry={entry:.4f}")
                            # Swap them
                            tp, sl = sl, tp
                    else:  # short
                        # For short: TP should be < entry, SL should be > entry
                        if tp > 0 and sl > 0 and tp > sl:
                            logger.warning(f"[{symbol}] TP/SL appear swapped for short position! TP={tp:.4f} SL={sl:.4f} Entry={entry:.4f}")
                            # Swap them
                            tp, sl = sl, tp
                    
                    # Add to book
                    from position_mgr import Position
                    # Try to restore strategy from Redis; in trend-only mode default to trend_pullback
                    recovered_strategy = "unknown"
                    try:
                        if getattr(self, '_redis', None) is not None:
                            v = self._redis.get(f'openpos:strategy:{symbol}')
                            if isinstance(v, str) and v:
                                recovered_strategy = v
                        if recovered_strategy == "unknown":
                            try:
                                if getattr(self, '_trend_only', False):
                                    recovered_strategy = 'trend_pullback'
                            except Exception:
                                pass
                    except Exception:
                        recovered_strategy = "unknown"
                    book.positions[symbol] = Position(
                        side=side,
                        qty=qty,
                        entry=entry,
                        sl=sl if sl > 0 else (entry * 0.95 if side == "long" else entry * 1.05),
                        tp=tp if tp > 0 else (entry * 1.1 if side == "long" else entry * 0.9),
                        entry_time=datetime.now() - pd.Timedelta(hours=1),  # Approximate for recovered positions
                        strategy_name=recovered_strategy  # Restored if available; may remain 'unknown'
                    )
                    
                    # Hydrate per-position meta from Redis if available
                    try:
                        if getattr(self, '_redis', None) is not None:
                            import json as _json
                            meta_raw = self._redis.get(f'openpos:meta:{symbol}')
                            if meta_raw:
                                m = _json.loads(meta_raw)
                                if not hasattr(self, '_position_meta'):
                                    self._position_meta = {}
                                self._position_meta[symbol] = m
                    except Exception:
                        pass
                    # Hydrate scale-out state from Redis or reconstruct
                    so_hydrated = False
                    try:
                        if getattr(self, '_redis', None) is not None:
                            import json as _json
                            so_raw = self._redis.get(f'openpos:scaleout:{symbol}')
                            if so_raw:
                                so = _json.loads(so_raw)
                                if not hasattr(self, '_scaleout'):
                                    self._scaleout = {}
                                so['recovered'] = True
                                self._scaleout[symbol] = so
                                so_hydrated = True
                    except Exception:
                        so_hydrated = False
                    if not so_hydrated:
                        # Reconstruct TP1/TP2/qtys from config; do NOT modify exchange orders
                        try:
                            tr_exec = ((self.config.get('trend', {}) or {}).get('exec', {}) or {})
                            sc = (tr_exec.get('scaleout',{}) or {})
                            if bool(sc.get('enabled', False)):
                                frac = max(0.1, min(0.9, float(sc.get('fraction', 0.5))))
                                tp1_r = float(sc.get('tp1_r', 1.6))
                                tp2_r = float(sc.get('tp2_r', 3.0))
                                R = abs(entry - (sl if sl > 0 else entry))
                                if R > 0:
                                    if side == 'long':
                                        tp1 = float(entry) + tp1_r * R
                                        tp2 = float(entry) + tp2_r * R
                                    else:
                                        tp1 = float(entry) - tp1_r * R
                                        tp2 = float(entry) - tp2_r * R
                                    qty1 = float(qty) * frac
                                    qty2 = max(0.0, float(qty) - qty1)
                                    be_tolerance = float(entry) * 0.001  # 0.1%
                                    be_moved = abs(sl - entry) <= be_tolerance
                                    if not hasattr(self, '_scaleout'):
                                        self._scaleout = {}
                                    self._scaleout[symbol] = {
                                        'tp1': float(tp1), 'tp2': float(tp2), 'entry': float(entry),
                                        'side': side, 'be_moved': bool(be_moved), 'move_be': bool(sc.get('move_sl_to_be', True)),
                                        'qty1': float(qty1), 'qty2': float(qty2), 'tp1_order_id': None,
                                        'tp1_hit': bool(be_moved), 'tp1_notified': False,
                                        'recovered': True
                                    }
                        except Exception:
                            pass
                    recovered += 1
                    recovered_details.append({'symbol': symbol, 'strategy': recovered_strategy, 'scaleout': 'hydrated' if so_hydrated else 'reconstructed'})
                    if recovered_strategy != "unknown":
                        logger.info(f"Recovered {side} position: {symbol} qty={qty} entry={entry:.4f} TP={tp:.4f} SL={sl:.4f} strategy={recovered_strategy}")
                    else:
                        logger.info(f"Recovered {side} position: {symbol} qty={qty} entry={entry:.4f} TP={tp:.4f} SL={sl:.4f}")
                
                if recovered > 0:
                    logger.info(f"Successfully recovered {recovered} position(s) - WILL NOT MODIFY THEM")
                    logger.info("Existing positions and their TP/SL orders will run their course without interference")
                    
                    # Send Telegram notification
                    if self.tg:
                        msg = f"📊 *Recovered {recovered} existing position(s)*\n"
                        msg += "⚠️ *These positions will NOT be modified*\n"
                        msg += "✅ *TP/SL orders preserved as-is*\n\n"
                        for d in recovered_details:
                            sym = d['symbol']; pos = book.positions.get(sym)
                            emoji = "🟢" if pos and pos.side == "long" else "🔴"
                            msg += f"{emoji} {sym}: {pos.side if pos else '?'} qty={pos.qty if pos else 0}\n"
                            msg += f"   strategy={d['strategy']} scaleout={d['scaleout']}\n"
                        await self.tg.send_message(msg)
            else:
                logger.info("No existing positions to recover")
                
        except Exception as e:
            logger.error(f"Failed to recover positions: {e}")
    
    async def kline_stream(self, ws_url:str, topics:list[str]):
        """Stream klines from Bybit WebSocket (main stream) with timeframe-aware idle handling."""
        sub = {"op":"subscribe","args":[f"kline.{t}" for t in topics]}
        backoff = 3.0
        max_backoff = 30.0
        last_warn = 0.0

        def _compute_timeout() -> float:
            """Compute a sensible recv timeout based on the largest timeframe in topics."""
            try:
                tf_secs = 0
                for t in topics:
                    try:
                        tf = str(t).split(".")[0]
                        tf_i = int(tf)
                        tf_secs = max(tf_secs, tf_i * 60)
                    except Exception:
                        continue
                base = tf_secs if tf_secs > 0 else 60
                return max(120.0, float(base + 30))
            except Exception:
                return 120.0
        
        while self.running:
            try:
                logger.info(f"Connecting to WebSocket: {ws_url}")
                # Steady ping cadence; dynamic recv timeout per timeframe
                async with websockets.connect(ws_url, ping_interval=30, ping_timeout=40) as ws:
                    self.ws = ws
                    await ws.send(json.dumps(sub))
                    logger.info(f"Subscribed to topics: {topics}")
                    backoff = 3.0  # reset after successful connect
                    last_msg = time.monotonic()
                    recv_timeout = _compute_timeout()

                    while self.running:
                        try:
                            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=recv_timeout))
                            last_msg = time.monotonic()
                            if msg.get("success") == False:
                                if "already subscribed" in msg.get("ret_msg", ""):
                                    logger.debug("Already subscribed; continuing")
                                else:
                                    logger.error(f"Subscription failed: {msg}")
                                continue
                            topic = msg.get("topic","")
                            if topic.startswith("kline."):
                                sym = topic.split(".")[-1]
                                for k in msg.get("data", []):
                                    yield sym, k
                        except asyncio.TimeoutError:
                            # Idle: send ping. Reconnect only after prolonged idle (> 3x timeout) or ping failure.
                            try:
                                await ws.ping()
                                logger.debug(f"WebSocket idle ping sent (timeout={int(recv_timeout)}s)")
                            except Exception:
                                now = time.monotonic()
                                if now - last_warn > 120:
                                    logger.warning("WebSocket ping failed after timeout, reconnecting...")
                                    last_warn = now
                                break
                            # If prolonged idle without any message, refresh connection
                            now = time.monotonic()
                            if (now - last_msg) > (3 * recv_timeout):
                                logger.warning(f"WebSocket prolonged idle ({int(now - last_msg)}s), reconnecting...")
                                break
                        except websockets.exceptions.ConnectionClosed:
                            now = time.monotonic()
                            if now - last_warn > 120:
                                logger.warning("WebSocket connection closed, reconnecting...")
                                last_warn = now
                            else:
                                logger.info("WebSocket connection closed, reconnecting…")
                            break
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
            # Backoff between reconnects while running
            if self.running:
                import random
                sleep_s = min(max_backoff, backoff * (1.0 + random.uniform(-0.2, 0.2)))
                await asyncio.sleep(sleep_s)
                backoff = min(max_backoff, backoff * 1.6)

    async def auto_generate_enhanced_clusters(self):
        """Auto-generate or update enhanced clusters if needed"""
        try:
            # First try to load existing enhanced clusters
            try:
                from cluster_feature_enhancer import load_cluster_data
                simple_clusters, enhanced_clusters = load_cluster_data()
            except ImportError:
                logger.warning("cluster_feature_enhancer not available, using basic clustering only")
                simple_clusters, enhanced_clusters = {}, {}
            
            # Check if we need to generate or update
            needs_generation = False
            if not enhanced_clusters:
                logger.info("🆕 No enhanced clusters found, auto-generating...")
                needs_generation = True
            else:
                # Check for broken clustering (too many borderline symbols)
                try:
                    import json
                    from datetime import datetime
                    with open('symbol_clusters_enhanced.json', 'r') as f:
                        cluster_data = json.load(f)
                        
                        # Count borderline symbols
                        enhanced_data = cluster_data.get('enhanced_clusters', {})
                        total_symbols = len(enhanced_data)
                        borderline_count = sum(1 for data in enhanced_data.values() 
                                             if data.get('is_borderline', False))
                        
                        # If more than 50% are borderline, clustering is broken
                        if total_symbols > 0 and borderline_count / total_symbols > 0.5:
                            logger.warning(f"🚨 Broken clustering detected: {borderline_count}/{total_symbols} symbols are borderline!")
                            logger.info("Deleting and regenerating clusters...")
                            import os
                            os.remove('symbol_clusters_enhanced.json')
                            needs_generation = True
                        # Also check for obviously wrong assignments
                        elif enhanced_data.get('BTCUSDT', {}).get('primary_cluster') != 1:
                            logger.warning("🚨 BTCUSDT not in Blue Chip cluster - clustering is broken!")
                            os.remove('symbol_clusters_enhanced.json')
                            needs_generation = True
                        else:
                            # Check age
                            if 'generated_at' in cluster_data:
                                gen_time = datetime.fromisoformat(cluster_data['generated_at'])
                                days_old = (datetime.now() - gen_time).days
                                if days_old > 7:  # Update weekly
                                    logger.info(f"🔄 Enhanced clusters are {days_old} days old, updating...")
                                    needs_generation = True
                                else:
                                    logger.info(f"✅ Enhanced clusters are {days_old} days old, still fresh")
                except Exception as e:
                    logger.warning(f"Error checking clusters: {e}")
                    needs_generation = True
            
            # Generate if needed
            if needs_generation:
                logger.info("🎯 Generating enhanced clusters from historical data...")
                try:
                    from symbol_clustering import SymbolClusterer
                    from datetime import datetime
                    import json
                    import numpy as np
                    
                    # Use loaded frames data
                    if self.frames and len(self.frames) > 0:
                        # Only use symbols with enough data
                        valid_frames = {sym: df for sym, df in self.frames.items() 
                                      if len(df) >= 500}
                        
                        if len(valid_frames) >= 20:  # Need at least 20 symbols
                            clusterer = SymbolClusterer(valid_frames)
                            metrics = clusterer.calculate_metrics(min_candles=500)
                            clusters = clusterer.cluster_symbols()
                            
                            # Convert simple clusters to enhanced format for compatibility
                            enhanced_data = {}
                            for symbol, cluster_id in clusters.items():
                                enhanced_data[symbol] = {
                                    "primary_cluster": cluster_id,
                                    "confidence": 0.95,  # High confidence for rule-based clustering
                                    "is_borderline": False,  # No borderline in simple clustering
                                    "secondary_cluster": None,
                                    "secondary_confidence": 0.0
                                }
                            
                            # Create enhanced format output
                            output = {
                                "generated_at": datetime.now().isoformat(),
                                "cluster_descriptions": clusterer.get_cluster_descriptions(),
                                "symbol_clusters": clusters,  # Backward compatible key name
                                "enhanced_clusters": enhanced_data,
                                "metrics_summary": {}
                            }
                            
                            # Add metrics summary per cluster
                            for cluster_id in range(1, 6):
                                cluster_symbols = [s for s, c in clusters.items() if c == cluster_id]
                                if cluster_symbols:
                                    cluster_metrics = [metrics[s] for s in cluster_symbols if s in metrics]
                                    
                                    output["metrics_summary"][cluster_id] = {
                                        "count": len(cluster_symbols),
                                        "symbols": cluster_symbols[:5],  # Show first 5 as examples
                                        "avg_volatility": np.mean([m.avg_volatility for m in cluster_metrics]) if cluster_metrics else 0,
                                        "avg_btc_correlation": np.mean([m.btc_correlation for m in cluster_metrics]) if cluster_metrics else 0
                                    }
                            
                            # Save to enhanced clusters file for compatibility
                            with open('symbol_clusters_enhanced.json', 'w') as f:
                                json.dump(output, f, indent=2)
                            
                            logger.info(f"✅ Generated enhanced clusters for {len(clusters)} symbols")
                            
                            # Notify via Telegram if available
                            if hasattr(self, 'tg') and self.tg:
                                await self.tg.send_message(
                                    f"✅ *Auto-generated enhanced clusters*\n"
                                    f"Analyzed {len(clusters)} symbols\n"
                                    f"Use /clusters to view status"
                                )
                        else:
                            logger.warning(f"Only {len(valid_frames)} symbols have enough data, skipping generation")
                    else:
                        logger.warning("No frame data available for cluster generation")
                        
                except Exception as e:
                    logger.error(f"Failed to auto-generate clusters: {e}")
                    # Don't fail the bot startup
            
        except Exception as e:
            logger.error(f"Error in auto cluster generation: {e}")
            # Don't fail the bot startup

    async def run(self):
        """Main bot loop"""
        # Load config
        with open("config.yaml","r") as f:
            cfg = yaml.safe_load(f)
        
        # Replace environment variables
        cfg = replace_env_vars(cfg)
        
        # Store config as instance variable
        self.config = cfg
        # Global gating disable (per user request to allow phantom and executed trades through)
        try:
            self._disable_gates = True
            # Also reflect in config to minimize gated branches
            cfg.setdefault('trend', {}).setdefault('regime', {})['enabled'] = False
            cfg.setdefault('mr', {}).setdefault('regime', {})['enabled'] = False
            cfg.setdefault('router', {}).setdefault('htf_bias', {})['enabled'] = False
        except Exception:
            self._disable_gates = True
        # Trend-only mode toggle from config/env
        try:
            self._trend_only = bool(((cfg.get('modes', {}) or {}).get('trend_only', False)))
        except Exception:
            self._trend_only = False
        try:
            if not self._trend_only:
                env_flag = str(os.getenv('TREND_ONLY', '')).strip().lower()
                self._trend_only = env_flag in ('1','true','yes','on')
        except Exception:
            pass
        if self._trend_only:
            # Mutate runtime config to disable Scalp and MR execution paths
            try:
                cfg.setdefault('scalp', {})['enabled'] = False
            except Exception:
                pass
            try:
                cfg.setdefault('mr', {}).setdefault('exec', {})['enabled'] = False
            except Exception:
                pass
            # Prefer pure Trend Pullback path (no enhanced-parallel orchestration)
            try:
                cfg.setdefault('trade', {})['use_enhanced_parallel'] = False
            except Exception:
                pass
            # Silence non-trend logs via a lightweight filter
            class _TrendOnlyFilter(logging.Filter):
                def filter(self, record: logging.LogRecord) -> bool:
                    msg = str(record.getMessage())
                    # Drop common MR/Scalp lines
                    noisy = (
                        'Mean Reversion' in msg or ' phantom_mr' in msg or ' exec_mr' in msg or
                        '🩳' in msg or 'Scalp' in msg or 'scalp' in msg or 'MR execution' in msg or
                        'MR ' in msg or 'MR:' in msg or 'RANGE DEBUG' in msg or 'Range details' in msg or 'FALLBACK' in msg
                    )
                    return not noisy
            try:
                logging.getLogger().addFilter(_TrendOnlyFilter())
            except Exception:
                pass
            # Reduce verbosity of non-trend modules explicitly
            try:
                for name in (
                    'enhanced_market_regime', 'strategy_mean_reversion', 'ml_scorer_mean_reversion',
                    'enhanced_mr_scorer', 'mr_phantom_tracker', 'strategy_scalp', 'scalp_phantom_tracker',
                    'ml_scorer_scalp'
                ):
                    logging.getLogger(name).setLevel(logging.WARNING)
            except Exception:
                pass
            try:
                logger.info('🔕 Trend-only mode active: MR & Scalp disabled (exec + phantom); logs muted for non-trend')
            except Exception:
                pass
        # Startup build fingerprint (commit hash + timestamp)
        try:
            sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            try:
                sha = os.getenv('BUILD_SHA') or 'unknown'
            except Exception:
                sha = 'unknown'
        try:
            build_id = os.getenv('BUILD_ID', 'n/a')
            logger.info(f"🧬 Build: {sha} ver={VERSION} id={build_id} @ {datetime.utcnow().isoformat()}Z")
        except Exception:
            pass

        # Mute legacy/unused strategies globally so logs focus on Trend/Range/Scalp
        try:
            for name in (
                'enhanced_mr_scorer', 'mr_phantom_tracker',
                'strategy_regimes', 'strategy_mean_reversion', 'ml_signal_scorer_immediate', 'ml_scorer_mean_reversion',
                'backtester', 'enhanced_backtester',
                'regime_classifier', 'market_regime', 'enhanced_market_regime'
            ):
                try:
                    logging.getLogger(name).setLevel(logging.WARNING)
                except Exception:
                    pass
        except Exception:
            pass

        # Enforce MR disabled and disable enhanced-parallel router to keep only Trend Pullback, Range FBO and Scalp
        try:
            self._mr_disabled = True
        except Exception:
            pass
        try:
            cfg.setdefault('trade', {})['use_enhanced_parallel'] = False
        except Exception:
            pass
        
        # Extract configuration
        symbols = [s.upper() for s in cfg["trade"]["symbols"]]
        tf = cfg["trade"]["timeframe"]
        topics = [f"{tf}.{s}" for s in symbols]

        logger.info(f"Trading symbols: {symbols}")
        logger.info(f"Timeframe: {tf} minutes")
        logger.info("📌 Bot Policy: Existing positions and orders will NOT be modified - they will run their course")

        # Scalp configuration diagnostics (early visibility)
        try:
            scalp_cfg_diag = cfg.get('scalp', {})
            scalp_enabled = bool(scalp_cfg_diag.get('enabled', False))
            scalp_independent = bool(scalp_cfg_diag.get('independent', False))
            scalp_tf = str(scalp_cfg_diag.get('timeframe', '3'))
            logger.info(
                f"🩳 Scalp config: enabled={scalp_enabled}, independent={scalp_independent}, tf={scalp_tf}m, modules={'OK' if SCALP_AVAILABLE else 'MISSING'}"
            )
            # Startup fingerprint for scalp acceptance plumbing
            try:
                central_enabled = bool(((cfg.get('router', {}) or {}).get('central_enabled', False)))
            except Exception:
                central_enabled = False
            try:
                hb = (cfg.get('phantom', {}).get('hourly_symbol_budget', {}) or {}).get('scalp', 'n/a')
                cap_none = (cfg.get('phantom', {}).get('caps', {}).get('scalp', {}) or {}).get('none', 'n/a')
                sc_cfg = (cfg.get('scalp', {}) or {})
                dd_enabled = bool(sc_cfg.get('dedup_enabled', False))
                ddh = sc_cfg.get('dedup_hours', 'n/a')
                cdb = sc_cfg.get('cooldown_bars', 'n/a')
                dbg_force = bool(((sc_cfg.get('debug', {}) or {}).get('force_accept', False)) if isinstance(sc_cfg, dict) else False)
            except Exception:
                hb = cap_none = ddh = cdb = 'n/a'; dbg_force = False; dd_enabled = False
            logger.info(
                f"🔎 Startup fingerprint: central_router={central_enabled} scalp.hourly={hb} scalp.daily_none={cap_none} dedup_enabled={dd_enabled} dedup_hours={ddh} cooldown_bars={cdb} debug.force_accept={dbg_force} backstop=ON"
            )
        except Exception:
            pass
        
        # Import strategies for parallel system
        from strategy_mean_reversion import detect_signal as detect_signal_mean_reversion
        # Use pullback strategy detection with detailed logging
        from strategy_pullback import detect_signal_pullback as detect_trend_signal
        from strategy_pullback import reset_symbol_state as _reset_symbol_state
        # Cache Trend detect function for 3m micro-step usage
        try:
            self._detect_trend_signal = detect_trend_signal
        except Exception:
            self._detect_trend_signal = None
        # Strategy overrides: disable MR/Scalp detection functions at source
        if self._trend_only:
            try:
                detect_signal_mean_reversion = (lambda *args, **kwargs: None)
            except Exception:
                pass
            try:
                globals()['detect_scalp_signal'] = None
            except Exception:
                pass
        else:
            # Independent MR disable flag from config
            try:
                self._mr_disabled = bool(((cfg.get('modes', {}) or {}).get('disable_mr', True)))
            except Exception:
                self._mr_disabled = True
            if getattr(self, '_mr_disabled', False):
                try:
                    detect_signal_mean_reversion = (lambda *args, **kwargs: None)
                except Exception:
                    pass
        use_enhanced_parallel = cfg["trade"].get("use_enhanced_parallel", True) and ENHANCED_ML_AVAILABLE
        use_regime_switching = cfg["trade"].get("use_regime_switching", False)
        # Backward-compat: no reset function in Trend-only mode
        reset_symbol_state = None

        # Exploration flags (phantom-only loosening per strategy)
        exploration_enabled = bool(cfg.get('exploration', {}).get('enabled', True))
        mr_explore = cfg.get('mr', {}).get('explore', {})

        # Phantom config
        phantom_cfg = cfg.get('phantom', {})
        phantom_none_cap = int(phantom_cfg.get('none_cap', 50))
        phantom_cluster3_cap = int(phantom_cfg.get('cluster3_cap', 20))
        phantom_offhours_cap = int(phantom_cfg.get('offhours_cap', 15))
        phantom_enable_virtual = bool(phantom_cfg.get('enable_virtual_snapshots', False))
        phantom_virtual_delta = int(phantom_cfg.get('virtual_snapshots_delta', 5))

        if use_enhanced_parallel:
            strategy_type = "Enhanced Parallel (Trend Pullback + Mean Reversion with ML)"
            logger.info(f"📊 Strategy: {strategy_type}")
            logger.info("🧠 Using Enhanced Parallel ML System with regime-based strategy routing")
        else:
            strategy_type = "Trend Pullback"
            logger.info(f"📊 Strategy: {strategy_type}")
        
        # Initialize strategy settings
        settings = Settings(
            atr_len=cfg["trade"]["atr_len"],
            sl_buf_atr=cfg["trade"]["sl_buf_atr"],
            rr=cfg["trade"]["rr"],
            use_ema=cfg["trade"]["use_ema"],
            ema_len=cfg["trade"]["ema_len"],
            use_vol=cfg["trade"]["use_vol"],
            vol_len=cfg["trade"]["vol_len"],
            vol_mult=cfg["trade"]["vol_mult"],
            both_hit_rule=cfg["trade"]["both_hit_rule"],
            confirmation_candles=cfg["trade"].get("confirmation_candles", 2),
            # Trend-specific breathing room for pivot stops
            extra_pivot_breath_pct=float(((cfg.get('trend', {}) or {}).get('exec', {}) or {}).get('extra_pivot_breath_pct', 0.01)),
            confirmation_timeout_bars=int((cfg.get('trend', {}) or {}).get('confirmation_timeout_bars', 6)),
            use_3m_pullback=bool((((cfg.get('trend', {}) or {}).get('context', {}) or {}).get('use_3m_pullback', True))),
            use_3m_confirm=bool((((cfg.get('trend', {}) or {}).get('context', {}) or {}).get('use_3m_confirm', True))),
            # Microstructure + BOS config (with safe defaults)
            retest_enabled=bool((((cfg.get('trend', {}) or {}).get('exec', {}) or {}).get('retest', {}) or {}).get('enabled', True)),
            retest_distance_mode=str((((cfg.get('trend', {}) or {}).get('exec', {}) or {}).get('retest', {}) or {}).get('distance_mode', 'atr')),
            retest_max_dist_atr=float((((cfg.get('trend', {}) or {}).get('exec', {}) or {}).get('retest', {}) or {}).get('max_dist_atr', 0.50)),
            retest_max_dist_pct=float((((cfg.get('trend', {}) or {}).get('exec', {}) or {}).get('retest', {}) or {}).get('max_dist_pct', 0.40)),
            require_protective_hl_for_long=bool((((cfg.get('trend', {}) or {}).get('exec', {}) or {}).get('micro', {}) or {}).get('require_protective_hl_for_long', True)),
            require_protective_lh_for_short=bool((((cfg.get('trend', {}) or {}).get('exec', {}) or {}).get('micro', {}) or {}).get('require_protective_lh_for_short', True)),
            bos_body_min_ratio=float((((cfg.get('trend', {}) or {}).get('exec', {}) or {}).get('bos', {}) or {}).get('body_min_ratio', 0.25)),
            bos_confirm_closes=int((((cfg.get('trend', {}) or {}).get('exec', {}) or {}).get('bos', {}) or {}).get('confirm_closes', 1)),
            breakout_to_pullback_bars_3m=int((((cfg.get('trend', {}) or {}).get('exec', {}) or {}).get('timeouts', {}) or {}).get('breakout_to_pullback_bars_3m', 25)),
            pullback_to_bos_bars_3m=int((((cfg.get('trend', {}) or {}).get('exec', {}) or {}).get('timeouts', {}) or {}).get('pullback_to_bos_bars_3m', 25)),
            breakout_buffer_atr=float((((cfg.get('trend', {}) or {}).get('exec', {}) or {}).get('breakout_buffer_atr', 0.05))),
            # Divergence config (TSI/RSI) for 3m strict gating
            div_enabled=bool(((((cfg.get('trend', {}) or {}).get('exec', {}) or {}).get('divergence', {}) or {}).get('enabled', False))),
            div_mode=str(((((cfg.get('trend', {}) or {}).get('exec', {}) or {}).get('divergence', {}) or {}).get('mode', 'optional'))),
            div_require=str(((((cfg.get('trend', {}) or {}).get('exec', {}) or {}).get('divergence', {}) or {}).get('require', 'any'))),
            div_use_rsi=bool('rsi' in ((((cfg.get('trend', {}) or {}).get('exec', {}) or {}).get('divergence', {}) or {}).get('oscillators', ['rsi','tsi']))),
            div_use_tsi=bool('tsi' in ((((cfg.get('trend', {}) or {}).get('exec', {}) or {}).get('divergence', {}) or {}).get('oscillators', ['rsi','tsi']))),
            div_rsi_len=int(((((cfg.get('trend', {}) or {}).get('exec', {}) or {}).get('divergence', {}) or {}).get('rsi_len', 14))),
            div_tsi_long=int(((((cfg.get('trend', {}) or {}).get('exec', {}) or {}).get('divergence', {}) or {}).get('tsi_long', 25))),
            div_tsi_short=int(((((cfg.get('trend', {}) or {}).get('exec', {}) or {}).get('divergence', {}) or {}).get('tsi_short', 13))),
            div_window_bars_3m=int(((((cfg.get('trend', {}) or {}).get('exec', {}) or {}).get('divergence', {}) or {}).get('confirm_window_bars_3m', 6))),
            div_min_strength_rsi=float(((((cfg.get('trend', {}) or {}).get('exec', {}) or {}).get('divergence', {}) or {}).get('min_strength', {})).get('rsi', 2.0) if isinstance(((((cfg.get('trend', {}) or {}).get('exec', {}) or {}).get('divergence', {}) or {}).get('min_strength', {})), dict) else 2.0),
            div_min_strength_tsi=float(((((cfg.get('trend', {}) or {}).get('exec', {}) or {}).get('divergence', {}) or {}).get('min_strength', {})).get('tsi', 0.3) if isinstance(((((cfg.get('trend', {}) or {}).get('exec', {}) or {}).get('divergence', {}) or {}).get('min_strength', {})), dict) else 0.3),
            div_notify=bool(((((cfg.get('trend', {}) or {}).get('exec', {}) or {}).get('divergence', {}) or {}).get('notify', True)))
            ,
            bos_armed_hold_minutes=int((((cfg.get('trend', {}) or {}).get('exec', {}) or {}).get('bos_hold_minutes', 300)))
            ,
            sl_mode=str(((((cfg.get('trend', {}) or {}).get('exec', {}) or {}).get('sl_mode', 'breakout')))),
            breakout_sl_buffer_atr=float(((((cfg.get('trend', {}) or {}).get('exec', {}) or {}).get('breakout_sl_buffer_atr', 0.30)))),
            min_r_pct=float(((((cfg.get('trend', {}) or {}).get('exec', {}) or {}).get('min_r_pct', 0.005))))
        )
        # Pullback detection uses the generic Settings() already constructed above
        # Keep a separate alias to avoid refactoring call sites
        tr_cfg = cfg.get('trend', {}) or {}
        trend_settings = settings
        try:
            self._trend_settings = trend_settings
        except Exception:
            pass
        # Log effective Trend 3m timeouts for visibility
        try:
            logger.info(
                f"🔧 Trend 3m timeouts: breakout→pullback={int(getattr(trend_settings,'breakout_to_pullback_bars_3m',0))} bars, "
                f"pullback→BOS={int(getattr(trend_settings,'pullback_to_bos_bars_3m',0))} bars"
            )
        except Exception:
            pass
        # Propagate rule_mode to strategy (disable SR hard gate when enabled)
        try:
            rm = (cfg.get('trend', {}) or {}).get('rule_mode', {})
            import os as _os
            _os.environ['TREND_RULE_MODE'] = '1' if bool(rm.get('enabled', False)) else '0'
        except Exception:
            pass
        # Enable reset_symbol_state for pullback state machine
        reset_symbol_state = _reset_symbol_state

        # Initialize components
        risk = RiskConfig(
            risk_usd=cfg["trade"]["risk_usd"],
            risk_percent=cfg["trade"]["risk_percent"],
            use_percent_risk=cfg["trade"]["use_percent_risk"],
            use_ml_dynamic_risk=False,  # Disabled - using fixed 1% risk until ML models are consistent
            ml_risk_min_score=70.0,
            ml_risk_max_score=100.0,
            ml_risk_min_percent=1.0,
            ml_risk_max_percent=5.0
        )
        # Initialize sizer with fee-aware sizing to better match risk at SL
        fee_total_pct = 0.00165
        try:
            fee_total_pct = float(cfg.get('trade', {}).get('fee_total_pct', 0.00165))
        except Exception:
            pass
        sizer = Sizer(risk, fee_total_pct=fee_total_pct, include_fees=True)
        book = Book()
        # Expose for stream-side execution
        self.risk = risk
        self.sizer = sizer
        self.book = book
        panic_list:list[str] = []
        
        # Initialize ALL ML components (including enhanced ones)
        ml_scorer = None
        phantom_tracker = None
        mean_reversion_scorer = None # Initialize Mean Reversion Scorer
        enhanced_mr_scorer = None  # Initialize Enhanced MR Scorer
        mr_phantom_tracker = None  # Initialize MR Phantom Tracker
        use_ml = cfg["trade"].get("use_ml_scoring", True)  # Default to True for immediate learning
        
        
        # Initialize symbol data collector
        symbol_collector = None
        if SYMBOL_COLLECTOR_AVAILABLE:
            try:
                symbol_collector = get_symbol_collector()
                logger.info("📊 Symbol data collector active - tracking for future ML")
            except Exception as e:
                logger.warning(f"Could not initialize symbol collector: {e}")
        
        # Always initialize Trend Phantom Tracker so Telegram dashboard has stats
        try:
            from phantom_trade_tracker import get_phantom_tracker as _get_pt
            phantom_tracker = _get_pt()
            try:
                phantom_tracker.set_notifier(self._notify_trend_phantom)
            except Exception:
                pass
        except Exception:
            phantom_tracker = None

        # Initialize Qscore adapters for Trend/Range/Scalp and trigger startup retrain if enough data
        try:
            from ml_qscore_trend_adapter import get_trend_qadapter
            from ml_qscore_range_adapter import get_range_qadapter
            from ml_qscore_scalp_adapter import get_scalp_qadapter
            self._qadapt_trend = get_trend_qadapter()
            self._qadapt_range = get_range_qadapter()
            self._qadapt_scalp = get_scalp_qadapter()
            try:
                self._qadapt_trend.retrain_if_ready()
                self._qadapt_range.retrain_if_ready()
                self._qadapt_scalp.retrain_if_ready()
            except Exception:
                pass
        except Exception as _qe:
            logger.debug(f"QAdapters init failed: {_qe}")

        # Always wire Scalp phantom notifier for lifecycle messages (open/close)
        try:
            if SCALP_AVAILABLE and get_scalp_phantom_tracker is not None:
                scpt_always = get_scalp_phantom_tracker()
                scpt_always.set_notifier(self._notify_scalp_phantom)
        except Exception:
            pass

        if ML_AVAILABLE and use_ml:
            try:
                if use_enhanced_parallel and ENHANCED_ML_AVAILABLE:
                    # Initialize Enhanced Parallel ML System
                    ml_scorer = get_trend_scorer() if ('get_trend_scorer' in globals() and get_trend_scorer is not None) else None  # Trend ML
                    # Phantom tracker already initialized above (kept for clarity)
                    # Wire Scalp phantom notifier for Telegram lifecycle messages
                    try:
                        if SCALP_AVAILABLE and get_scalp_phantom_tracker is not None:
                            scpt = get_scalp_phantom_tracker()
                            scpt.set_notifier(self._notify_scalp_phantom)
                            # Backfill open notifications for already-active scalp phantoms
                            try:
                                import asyncio as _asyncio
                                for lst in (getattr(scpt, 'active', {}) or {}).values():
                                    for ph in (lst or []):
                                        res = self._notify_scalp_phantom(ph)
                                        if _asyncio.iscoroutine(res):
                                            _asyncio.create_task(res)
                            except Exception:
                                pass
                    except Exception:
                        pass
                    # Initialize MR components only if MR is enabled
                    if not getattr(self, '_mr_disabled', False):
                        enhanced_mr_scorer = get_enhanced_mr_scorer()  # Enhanced MR ML
                        mr_phantom_tracker = get_mr_phantom_tracker()  # MR phantom tracker
                    else:
                        enhanced_mr_scorer = None
                        mr_phantom_tracker = None
                    mean_reversion_scorer = None  # Not used in enhanced system

                    # Get and log stats for both systems
                    trend_stats = ml_scorer.get_stats() if ml_scorer is not None else {'status':'N/A','current_threshold':0,'completed_trades':0,'recent_win_rate':0}
                    mr_stats = enhanced_mr_scorer.get_enhanced_stats() if enhanced_mr_scorer is not None else {'status': 'disabled', 'current_threshold': 0, 'completed_trades': 0, 'recent_win_rate': 0}

                    logger.info(f"✅ Enhanced Parallel ML System initialized")
                    logger.info(f"   Trend ML: {trend_stats['status']} (threshold: {trend_stats['current_threshold']:.0f}, trades: {trend_stats['completed_trades']})")
                    if not getattr(self, '_mr_disabled', False):
                        logger.info(f"   Mean Reversion ML: {mr_stats['status']} (threshold: {mr_stats['current_threshold']:.0f}, trades: {mr_stats['completed_trades']})")

                    if trend_stats['recent_win_rate'] > 0:
                        logger.info(f"   Trend recent WR: {trend_stats['recent_win_rate']:.1f}%")
                    if (not getattr(self, '_mr_disabled', False)) and mr_stats.get('recent_win_rate', 0) > 0:
                        logger.info(f"   MR recent WR: {mr_stats.get('recent_win_rate', 0):.1f}%")

                    # Perform Enhanced MR startup retrain with phantom data
                    if not getattr(self, '_mr_disabled', False) and enhanced_mr_scorer is not None:
                        logger.info("🔄 Checking for Enhanced MR startup retrain...")
                        enhanced_mr_startup_result = enhanced_mr_scorer.startup_retrain()
                        if enhanced_mr_startup_result:
                            # Get updated stats after retrain
                            mr_stats = enhanced_mr_scorer.get_enhanced_stats()
                            logger.info(f"✅ Enhanced MR models retrained on startup")
                            logger.info(f"   Status: {mr_stats['status']}")
                            logger.info(f"   Threshold: {mr_stats['current_threshold']:.0f}")
                            if mr_stats.get('models_active'):
                                logger.info(f"   Active models: {', '.join(mr_stats['models_active'])}")
                        elif enhanced_mr_scorer.is_ml_ready:
                            logger.info("✅ Pre-trained Enhanced MR models loaded successfully.")
                        else:
                            logger.info("⚠️ No pre-trained Enhanced MR models found. Starting in online learning mode.")

                    # One-time Enhanced MR diagnostic dump and optional clear on start
                    if not getattr(self, '_mr_disabled', False) and enhanced_mr_scorer is not None:
                        try:
                            emr_cfg = cfg.get('enhanced_mr', {})
                            # Diagnostic dump: show first/last executed entries
                            if bool(emr_cfg.get('diag_dump_on_start', True)) and getattr(enhanced_mr_scorer, 'redis_client', None):
                                r = enhanced_mr_scorer.redis_client
                                total = int(r.llen('enhanced_mr:trades') or 0)
                                if total > 0:
                                    first = r.lrange('enhanced_mr:trades', 0, min(1, total-1)) or []
                                    last = r.lrange('enhanced_mr:trades', max(0, total-2), total-1) or []
                                    import json
                                    def _fmt(items):
                                        out = []
                                        for it in items:
                                            try:
                                                rec = json.loads(it)
                                                out.append(f"{rec.get('timestamp','?')} {rec.get('symbol','?')}")
                                            except Exception:
                                                out.append(str(it)[:80])
                                        return out
                                    # Clarify this buffer contains executed trade records only
                                    logger.info(f"🧪 Enhanced MR trade records (executed buffer): count={total}, first={_fmt(first)}, last={_fmt(last)}")
                            # Optional clearing (disabled by default)
                            if bool(emr_cfg.get('clear_on_start', False)) and getattr(enhanced_mr_scorer, 'redis_client', None):
                                r = enhanced_mr_scorer.redis_client
                                logger.warning("⚠️ Clearing Enhanced MR Redis namespace (clear_on_start=true)")
                                for key in ['enhanced_mr:trades', 'enhanced_mr:completed_trades', 'enhanced_mr:last_train_count']:
                                    try:
                                        r.delete(key)
                                    except Exception:
                                        pass
                                try:
                                    enhanced_mr_scorer.completed_trades = 0
                                    enhanced_mr_scorer.last_train_count = 0
                                except Exception:
                                    pass
                                logger.info("✅ Enhanced MR Redis keys cleared and counters reset")
                        except Exception as e:
                            logger.debug(f"Enhanced MR diagnostics/clear failed: {e}")
                    # Flush any deferred MR outcomes now that scorers are available
                    if not getattr(self, '_mr_disabled', False) and enhanced_mr_scorer is not None:
                        try:
                            self._flush_deferred_mr(enhanced_mr_scorer_ref=enhanced_mr_scorer, original_mr_scorer_ref=None)
                        except Exception:
                            pass

                else:
                    # Initialize original ML system (best-effort, don't fail phantom wiring if unavailable)
                    ml_scorer = None
                    try:
                        from ml_signal_scorer_immediate import get_immediate_scorer as _get_immediate_scorer
                        try:
                            ml_scorer = _get_immediate_scorer()
                        except Exception as _ml_e:
                            logger.debug(f"Immediate ML scorer init failed: {_ml_e}")
                    except Exception as _imp_e:
                        logger.debug(f"Immediate ML scorer import failed: {_imp_e}")
                    # Always wire phantom notifiers
                    phantom_tracker = get_phantom_tracker()
                    try:
                        phantom_tracker.set_notifier(self._notify_trend_phantom)
                    except Exception:
                        pass
                    try:
                        if SCALP_AVAILABLE and get_scalp_phantom_tracker is not None:
                            scpt = get_scalp_phantom_tracker()
                            scpt.set_notifier(self._notify_scalp_phantom)
                            # Backfill open notifications for already-active scalp phantoms
                            try:
                                import asyncio as _asyncio
                                for lst in (getattr(scpt, 'active', {}) or {}).values():
                                    for ph in (lst or []):
                                        res = self._notify_scalp_phantom(ph)
                                        if _asyncio.iscoroutine(res):
                                            _asyncio.create_task(res)
                            except Exception:
                                pass
                    except Exception:
                        pass
                    enhanced_mr_scorer = None
                    mr_phantom_tracker = None
                    mean_reversion_scorer = None  # MR disabled

                    # Get and log ML stats (if available)
                    if ml_scorer is not None:
                        try:
                            ml_stats = ml_scorer.get_stats()
                            logger.info(f"✅ Original ML Scorer initialized")
                            logger.info(f"   Status: {ml_stats['status']}")
                            logger.info(f"   Threshold: {ml_stats['current_threshold']:.0f}")
                            logger.info(f"   Completed trades: {ml_stats['completed_trades']}")
                            if ml_stats['recent_win_rate'] > 0:
                                logger.info(f"   Recent win rate: {ml_stats['recent_win_rate']:.1f}%")
                            if ml_stats.get('models_active'):
                                logger.info(f"   Active models: {', '.join(ml_stats['models_active'])}")
                        except Exception as _stats_e:
                            logger.debug(f"ML stats unavailable: {_stats_e}")
                    else:
                        logger.info("ℹ️ Original ML Scorer not available — running phantom-only tracking for Trend/Scalp.")
                # Phantom trades now expire naturally on TP/SL - no timeout needed
                
                # Perform startup retrain with all available data
                logger.info("🔄 Checking for ML startup retrain...")
                startup_result = ml_scorer.startup_retrain()
                if startup_result:
                    # Get updated stats after retrain
                    ml_stats = ml_scorer.get_stats()
                    logger.info(f"✅ ML models retrained on startup")
                    logger.info(f"   Status: {ml_stats['status']}")
                    logger.info(f"   Threshold: {ml_stats['current_threshold']:.0f}")
                    if ml_stats.get('models_active'):
                        logger.info(f"   Active models: {', '.join(ml_stats['models_active'])}")
                elif ml_scorer.is_ml_ready:
                    logger.info("✅ Pre-trained Trend ML models loaded successfully.")
                else:
                    logger.warning("⚠️ No pre-trained Trend model found. Starting in online learning mode.")
                
                
            except Exception as e:
                logger.warning(f"Failed to initialize ML/Phantom system: {e}. Running without ML.")
                ml_scorer = None
                phantom_tracker = None
        elif use_ml:
            logger.warning("ML scoring requested but ML module not available")
        
        # Initialize basic symbol clustering (enhanced will be done after data load)
        symbol_clusters = {}
        try:
            from symbol_clustering import load_symbol_clusters
            symbol_clusters = load_symbol_clusters()
            logger.info(f"Loaded basic symbol clusters for {len(symbol_clusters)} symbols")
        except Exception as e:
            logger.warning(f"Could not load symbol clusters: {e}. Features will use defaults.")
            symbol_clusters = {}
        
        # Initialize Bybit client
        self.bybit = bybit = Bybit(BybitConfig(
            cfg["bybit"]["base_url"], 
            cfg["bybit"]["api_key"], 
            cfg["bybit"]["api_secret"]
        ))

        # Initialize adaptive phantom flow controller (phantom-only)
        try:
            flow_ctrl = FlowController(cfg, getattr(phantom_tracker, 'redis_client', None) if 'phantom_tracker' in locals() else None)
        except Exception:
            flow_ctrl = FlowController(cfg, None)
        self.flow_controller = flow_ctrl
        
        # Test connection
        balance = bybit.get_balance()
        if balance:
            logger.info(f"Connected to Bybit. Balance: ${balance:.2f} USDT")
        else:
            logger.warning("Could not fetch balance, continuing anyway...")
        
        # Fetch historical data for all symbols
        await self.load_or_fetch_initial_data(symbols, tf)

        # Wire Trend pullback state persistence + hydrate from Redis
        try:
            from strategy_pullback import set_trend_state_store, hydrate_trend_states
            set_trend_state_store(self._redis)
            restored = hydrate_trend_states(self.frames, timeframe_min=int(tf), max_age_bars=int((cfg.get('trend',{}) or {}).get('state_max_age_bars', 48)))
            if restored:
                logger.info(f"Trend state: restored {restored} symbols from Redis")
        except Exception as e:
            logger.debug(f"Trend state hydrate failed: {e}")
        
        # Optional pre-run setup (scalp backfill, phantom timeouts)
        try:
            # Backfill 3m frames for scalper if enabled before analysis begins
            try:
                scalp_cfg = cfg.get('scalp', {})
                use_scalp = bool(scalp_cfg.get('enabled', False) and SCALP_AVAILABLE)
                if use_scalp:
                    await self.backfill_frames_3m(symbols, use_api_fallback=True, limit=200)
                else:
                    logger.debug("Scalp disabled in config; skipping 3m backfill")
            except Exception as e:
                logger.warning(f"3m backfill skipped due to error: {e}")
            # Apply MR phantom timeout override from config (exploration)
            try:
                if mr_explore and 'timeout_hours' in mr_explore and mr_phantom_tracker is not None:
                    mr_phantom_tracker.timeout_hours = int(mr_explore['timeout_hours'])
                    logger.info(f"MR phantom timeout set to {mr_phantom_tracker.timeout_hours}h (exploration)")
            except Exception as e:
                logger.debug(f"Could not set MR phantom timeout: {e}")
            # Apply Trend phantom timeout override from config (exploration)
            try:
                tr_explore = (cfg.get('trend', {}) or {}).get('explore', {})
                if tr_explore and 'timeout_hours' in tr_explore and phantom_tracker is not None:
                    phantom_tracker.timeout_hours = int(tr_explore['timeout_hours'])
                    logger.info(f"Trend phantom timeout set to {phantom_tracker.timeout_hours}h (exploration)")
            except Exception as e:
                logger.debug(f"Could not set Trend phantom timeout: {e}")
            # Apply Scalp phantom timeout override from config (exploration)
            try:
                s_cfg = cfg.get('scalp', {}).get('explore', {})
                if s_cfg and 'timeout_hours' in s_cfg and SCALP_AVAILABLE:
                    scpt = _safe_get_scalp_phantom_tracker()
                    scpt.timeout_hours = int(s_cfg['timeout_hours'])
                    logger.info(f"Scalp phantom timeout set to {scpt.timeout_hours}h (exploration)")
            except Exception as e:
                logger.debug(f"Could not set Scalp phantom timeout: {e}")

            # Note: main timeframe backfill intentionally not applied (per user request)
            
            # Send summary to Telegram if available
            if hasattr(self, 'tg') and self.tg and sr_results:
                symbols_with_levels = [sym for sym, count in sr_results.items() if count > 0]
                await self.tg.send_message(
                    f"📊 *HTF S/R Analysis Complete*\n"
                    f"Analyzed: {len(sr_results)} symbols\n" 
                    f"Found levels: {len(symbols_with_levels)} symbols\n"
                    f"Total levels: {sum(sr_results.values())}"
                )
        except Exception as e:
            logger.error(f"Failed to initialize HTF S/R levels: {e}")
        
        # Auto-generate enhanced clusters after data is loaded
        await self.auto_generate_enhanced_clusters()
        
        # Recover existing positions - PRESERVE THEIR ORDERS
        await self.recover_positions(book, sizer)
        
        # DISABLED: Not cancelling ANY orders to prevent accidental TP/SL removal
        # Orders will naturally cancel when positions close
        logger.info("Order cancellation DISABLED - all orders will be preserved")
        logger.info("TP/SL orders will close naturally with their positions")
        
        # Track analysis times
        last_analysis = {}
        
        # Setup shared data for Telegram - all ML components are now in scope
        shared = {
            "risk": risk,
            "book": book,
            "panic": panic_list,
            "meta": cfg.get("symbol_meta",{}),
            "config": cfg,
            "broker": bybit,
            "frames": self.frames,
            "last_analysis": last_analysis,
            "trade_tracker": self.trade_tracker,
            "ml_scorer": ml_scorer,
            "bot_instance": self,
            "last_balance": None,
            "timeframe": tf,
            "symbols_config": symbols,
            "risk_reward": settings.rr,
            # Expose live Trend Settings object for runtime adjustments (RR, timeouts)
            "trend_settings": settings,
            # Enhanced ML system components
            "enhanced_mr_scorer": enhanced_mr_scorer,
            "mr_phantom_tracker": mr_phantom_tracker,
            "mean_reversion_scorer": mean_reversion_scorer,
            "trend_scorer": get_trend_scorer() if 'get_trend_scorer' in globals() and get_trend_scorer is not None else None,
            "phantom_tracker": phantom_tracker,
            "use_enhanced_parallel": use_enhanced_parallel,
            # Phantom flow controller (adaptive phantom-only acceptance)
            "flow_controller": flow_ctrl,
            # MR promotion state (phantom→execute override when WR strong)
            "mr_promotion": {
                "active": False,
                "day": datetime.utcnow().strftime('%Y%m%d'),
                "count": 0
            },
            # Trend corking state (phantom→execute override for Trend when WR strong)
            "trend_promotion": {
                "active": False,
                "day": datetime.utcnow().strftime('%Y%m%d'),
                "count": 0
            },
            # Routing stickiness state per symbol
            "routing_state": {},
            # Simple telemetry counters
            "telemetry": {"ml_rejects": 0, "phantom_wins": 0, "phantom_losses": 0, "policy_rejects": 0}
        }
        # Initialize Range/Scalp states with Redis-backed continuity
        try:
            if not hasattr(self, '_range_symbol_state'):
                self._range_symbol_state = {}
            if not hasattr(self, '_scalp_symbol_state'):
                self._scalp_symbol_state = {}
            if not hasattr(self, '_range_reasons'):
                self._range_reasons = {}
            if not hasattr(self, '_scalp_reasons'):
                self._scalp_reasons = {}
            if hasattr(self, '_redis') and self._redis is not None:
                import json as _json
                rs = self._redis.get('state:range:summary')
                if rs:
                    shared['range_states'] = _json.loads(rs)
                ss = self._redis.get('state:scalp:summary')
                if ss:
                    shared['scalp_states'] = _json.loads(ss)
        except Exception:
            pass
        # Trend event ring buffer for Telegram dashboard/event log
        try:
            shared["trend_events"] = []
        except Exception:
            pass
        
        # Initialize Telegram bot with retry on conflict
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.tg = TGBot(cfg["telegram"]["token"], int(cfg["telegram"]["chat_id"]), shared)
                # Keep shared available to stream
                self.shared = shared
                await self.tg.start_polling()
                # Send shorter startup message for 20 symbols
                # Format risk display
                if risk.use_percent_risk:
                    risk_display = f"{risk.risk_percent}%"
                else:
                    risk_display = f"${risk.risk_usd}"
                
                await self.tg.send_message(
                    "🚀 *Trend Pullback Bot Started*\n\n"
                    f"📊 Monitoring: {len(symbols)} symbols | TF: {tf}m + 3m\n"
                    f"💰 Risk per trade: {risk_display} | R:R 1:{settings.rr}\n\n"
                    "15m break → 3m HL/LH → 3m 2/2 confirms → stream entry\n"
                    "Scale‑out: 50% at ~1.6R, SL→BE, runner to ~3.0R\n\n"
                    "Use /dashboard for buttons and status."
                )

                # Wire Trend event notifications to Telegram
                try:
                    from strategy_pullback import set_trend_event_notifier, set_trend_microframe_provider, set_trend_entry_executor, set_trend_phantom_recorder, set_trend_invalidation_hook
                    def _trend_notifier(symbol: str, text: str):
                        try:
                            # Fire-and-forget send to Telegram
                            asyncio.create_task(self.tg.send_message(text))
                            # Record into lightweight ring buffer for dashboard/event log
                            try:
                                evts = self.shared.get("trend_events") if hasattr(self, 'shared') else None
                                if isinstance(evts, list):
                                    from datetime import datetime as _dt
                                    evts.append({
                                        'ts': _dt.utcnow().isoformat() + 'Z',
                                        'symbol': symbol,
                                        'text': str(text)[:500]
                                    })
                                    # Cap buffer size
                                    if len(evts) > 60:
                                        del evts[:len(evts)-60]
                            except Exception:
                                pass
                        except Exception:
                            pass
                    set_trend_event_notifier(_trend_notifier)
                    # Provide 3m frames to pullback strategy for micro detection
                    try:
                        set_trend_microframe_provider(lambda sym: self.frames_3m.get(sym))
                    except Exception:
                        pass
                    # Provide phantom recorder callback for permissive BOS-cross phantoms
                    try:
                        def _trend_phantom_cb(sym: str, side: str, entry: float, sl: float, tp: float, meta: dict):
                            try:
                                # Respect phantom enabled toggle
                                cfg = self.config if hasattr(self, 'config') else {}
                                if not bool(((cfg.get('phantom', {}) or {}).get('enabled', True))):
                                    return
                                df = self.frames.get(sym)
                                feats = {}
                                if df is not None and not df.empty:
                                    cl = df['close']; price = float(cl.iloc[-1]) if len(cl) else 0.0
                                    ys = cl.tail(20).values if len(cl) >= 20 else cl.values
                                    try:
                                        slope = np.polyfit(np.arange(len(ys)), ys, 1)[0]
                                    except Exception:
                                        slope = 0.0
                                    trend_slope_pct = float((slope / price) * 100.0) if price else 0.0
                                    ema20 = float(cl.ewm(span=20, adjust=False).mean().iloc[-1]) if len(cl) >= 20 else price
                                    ema50 = float(cl.ewm(span=50, adjust=False).mean().iloc[-1]) if len(cl) >= 50 else ema20
                                    ema_stack_score = 100.0 if (price > ema20 > ema50 or price < ema20 < ema50) else (50.0 if ema20 != ema50 else 0.0)
                                    rng_today = float(df['high'].iloc[-1] - df['low'].iloc[-1])
                                    med_range = float((df['high'] - df['low']).rolling(20).median().iloc[-1]) if len(df) >= 20 else max(1e-9, rng_today)
                                    range_expansion = float(rng_today / max(1e-9, med_range))
                                    prev = cl.shift(); tr = np.maximum(df['high'] - df['low'], np.maximum((df['high'] - prev).abs(), (df['low'] - prev).abs()))
                                    atr = float(tr.rolling(14).mean().iloc[-1]) if len(tr) >= 14 else float(tr.iloc[-1])
                                    atr_pct = float((atr / max(1e-9, price)) * 100.0) if price else 0.0
                                    brk = float((meta or {}).get('breakout_level', 0.0) or 0.0)
                                    try:
                                        break_dist_atr = ((price - brk) / atr) if (side == 'long' and atr) else ((brk - price) / atr) if (atr) else 0.0
                                    except Exception:
                                        break_dist_atr = 0.0
                                    feats = {
                                        'trend_slope_pct': trend_slope_pct,
                                        'ema_stack_score': ema_stack_score,
                                        'atr_pct': atr_pct,
                                        'range_expansion': range_expansion,
                                        'breakout_dist_atr': float(break_dist_atr),
                                        # divergence + protective pivot flags
                                        'div_ok': 1 if bool((meta or {}).get('div_ok', False)) else 0,
                                        'div_score': float((meta or {}).get('div_score', 0.0) or 0.0),
                                        'div_rsi_delta': float((meta or {}).get('div_rsi_delta', 0.0) or 0.0),
                                        'div_tsi_delta': float((meta or {}).get('div_tsi_delta', 0.0) or 0.0),
                                        'protective_pivot_present': 1 if bool((meta or {}).get('protective_pivot_present', False)) else 0,
                                        'session': 'us',
                                        'symbol_cluster': 3,
                                        'volatility_regime': getattr(self, 'last_volatility_level', 'normal') if hasattr(self, 'last_volatility_level') else 'normal'
                                    }
                                # Score via ML if available
                                ml_score = 0.0
                                try:
                                    if 'get_trend_scorer' in globals() and get_trend_scorer is not None:
                                        tr_scorer = get_trend_scorer()
                                        dummy_sig = {'side': side, 'entry': entry, 'sl': sl, 'tp': tp}
                                        ml_score, _ = tr_scorer.score_signal(dummy_sig, feats)
                                except Exception:
                                    ml_score = 0.0
                                # Attach HTF snapshot to features
                                try:
                                    if df is not None and not df.empty:
                                        feats['htf'] = dict(self._compute_symbol_htf_exec_metrics(sym, df))
                                        comp0 = self._get_htf_metrics(sym, df)
                                        feats['htf_comp'] = dict(comp0)
                                        feats['ts15'] = float(comp0.get('ts15', 0.0))
                                        feats['ts60'] = float(comp0.get('ts60', 0.0))
                                        feats['rc15'] = float(comp0.get('rc15', 0.0))
                                        feats['rc60'] = float(comp0.get('rc60', 0.0))
                                except Exception:
                                    pass
                                # Attach Qscore snapshot for BOS phantom as well
                                try:
                                    rule_mode = (self.config.get('trend', {}) or {}).get('rule_mode', {}) if hasattr(self, 'config') else {}
                                    if bool(rule_mode.get('enabled', False)):
                                        qB, qcB, qrB = self._compute_qscore(sym, 'long' if side=='Buy' else 'short', df, self.frames_3m.get(sym) if hasattr(self, 'frames_3m') else None)
                                        feats['qscore'] = float(qB)
                                        feats['qscore_components'] = dict(qcB)
                                        feats['qscore_reasons'] = list(qrB)
                                except Exception:
                                    pass
                                # Attach rule-mode Qscore if enabled
                                try:
                                    rule_mode = (self.config.get('trend', {}) or {}).get('rule_mode', {}) if hasattr(self, 'config') else {}
                                    if bool(rule_mode.get('enabled', False)):
                                        q, qc, qr = self._compute_qscore(sym, 'long' if side=='Buy' else 'short', df, self.frames_3m.get(sym) if hasattr(self, 'frames_3m') else None)
                                        feats['qscore'] = float(q)
                                except Exception:
                                    pass
                                # Log and record phantom
                                try:
                                    try:
                                        logger.info(f"[{sym}] 👻 Trend PHANTOM (BOS) {side.upper()} @ {float(entry):.4f} SL {float(sl):.4f} TP {float(tp):.4f} | ML {ml_score:.1f}")
                                    except Exception:
                                        pass
                                    pt = get_phantom_tracker()
                                    pt.record_signal(sym, {'side': side, 'entry': float(entry), 'sl': float(sl), 'tp': float(tp)}, float(ml_score or 0.0), False, feats, 'trend_pullback')
                                except Exception:
                                    pass
                            except Exception:
                                pass
                        set_trend_phantom_recorder(_trend_phantom_cb)
                    except Exception:
                        pass
                    # Wire Trend invalidation handoff to Range phantom
                    try:
                        set_trend_invalidation_hook(self._trend_invalidation_handoff)
                    except Exception:
                        pass
                    # Provide stream-side entry executor to strategy (3m execution)
                    async def _stream_execute_trend(symbol: str, side: str, entry: float, sl: float, tp: float, meta: dict):
                        try:
                            exec_cfg = ((self.config.get('trend', {}) or {}).get('exec', {}) or {})
                            if not bool(exec_cfg.get('allow_stream_entry', True)):
                                return
                            # Guard: existing position
                            if symbol in self.book.positions:
                                return
                            # Ensure BOS confirmations reflected in Qscore BOS component for stream decisions
                            try:
                                if isinstance(meta, dict) and str(meta.get('phase','')) == '3m_bos':
                                    # Treat as fully confirmed (2/2) for BOS component visibility
                                    d = dict(getattr(self, '_last_signal_features', {}).get(symbol, {})) if hasattr(self, '_last_signal_features') else {}
                                    d['confirm_candles'] = max(2, int(d.get('confirm_candles', 0) or 0))
                                    if hasattr(self, '_last_signal_features'):
                                        self._last_signal_features[symbol] = d
                            except Exception:
                                pass
                            # Exec-only HTF gate (per-symbol)
                            try:
                                df_main = self.frames.get(symbol)
                                htf_ok, _, htf_mode, _ = self._apply_htf_exec_gate(symbol, df_main, side, 0.0)
                            except Exception:
                                htf_ok, htf_mode = True, 'error'
                            # Under rule_mode, HTF gating is scored, not blocked
                            rule_mode = (self.config.get('trend', {}) or {}).get('rule_mode', {}) if hasattr(self, 'config') else {}
                            rm_enabled = bool(rule_mode.get('enabled', False))
                            if (not rm_enabled) and (not htf_ok) and (htf_mode == 'gated'):
                                try:
                                    logger.info(f"[{symbol}] 🛑 Stream execute blocked by HTF gate (gated)")
                                except Exception:
                                    pass
                                # Persist HTF gate decision into trend state
                                try:
                                    from strategy_pullback import update_htf_gate
                                    m = {}
                                    try:
                                        m = self._compute_symbol_htf_exec_metrics(symbol, df_main)
                                    except Exception:
                                        m = {}
                                    meta_gate = {'mode': htf_mode}
                                    if isinstance(m, dict):
                                        for k in ('ts1h','ts4h','ema_dir_1h','ema_dir_4h','adx_1h','struct_dir_1h','struct_dir_4h'):
                                            if k in m:
                                                meta_gate[k] = m[k]
                                    update_htf_gate(symbol, False, meta_gate)
                                except Exception:
                                    pass
                                # Record phantom for learning with current stream proposal
                                try:
                                    from phantom_trade_tracker import get_phantom_tracker
                                    pt = get_phantom_tracker()
                                    # Build minimal features when available
                                    feats = {}
                                    try:
                                        feats = feats if isinstance(feats, dict) else {}
                                        feats['htf_gate'] = {'mode': htf_mode}
                                        feats['diversion_reason'] = 'htf_gate_stream'
                                        # Include full HTF snapshot + composite (flattened)
                                        try:
                                            feats['htf'] = dict(self._compute_symbol_htf_exec_metrics(symbol, df_main))
                                        except Exception:
                                            pass
                                        try:
                                            comp3 = self._get_htf_metrics(symbol, df_main)
                                            feats['htf_comp'] = dict(comp3)
                                            feats['ts15'] = float(comp3.get('ts15', 0.0))
                                            feats['ts60'] = float(comp3.get('ts60', 0.0))
                                            feats['rc15'] = float(comp3.get('rc15', 0.0))
                                            feats['rc60'] = float(comp3.get('rc60', 0.0))
                                        except Exception:
                                            pass
                                    except Exception:
                                        feats = {}
                                    pt.record_signal(symbol, {'side': side, 'entry': float(entry), 'sl': float(sl), 'tp': float(tp)}, float(ml_score_se or 0.0), False, feats, 'trend_pullback')
                                except Exception:
                                    pass
                                # Reset state to NEUTRAL since not executed
                                try:
                                    from strategy_pullback import revert_to_neutral
                                    revert_to_neutral(symbol)
                                except Exception:
                                    pass
                                # Notify
                                try:
                                    if self.tg:
                                        await self.tg.send_message(f"🛑 Trend: [{symbol}] HTF gate blocked (stream) — routed to phantom")
                                except Exception:
                                    pass
                                return
                            elif not htf_ok and htf_mode == 'soft':
                                try:
                                    logger.info(f"[{symbol}] 🔵 Stream execute: HTF soft mode — allowing with caution")
                                except Exception:
                                    pass
                            # Rule-mode: compute Qscore and block stream execute if below threshold
                            try:
                                rule_mode = (self.config.get('trend', {}) or {}).get('rule_mode', {}) if hasattr(self, 'config') else {}
                                rm_enabled = bool(rule_mode.get('enabled', False))
                            except Exception:
                                rm_enabled = False
                            if rm_enabled:
                                try:
                                    q, qc, qr = self._compute_qscore(symbol, 'long' if side=='Buy' else 'short', df_main, self.frames_3m.get(symbol) if hasattr(self, 'frames_3m') else None)
                                except Exception:
                                    q, qc, qr = 50.0, {}, []
                                exec_min = float(rule_mode.get('execute_q_min', 78))
                                ph_min = float(rule_mode.get('phantom_q_min', 65))
                                if q < exec_min:
                                    # Record phantom and return
                                    try:
                                        from phantom_trade_tracker import get_phantom_tracker
                                        pt = get_phantom_tracker()
                                        feats_q = {'qscore': float(q), 'qscore_components': dict(qc), 'qscore_reasons': list(qr), 'decision': 'rule_phantom_stream'}
                                        try:
                                            feats_q['htf'] = dict(self._compute_symbol_htf_exec_metrics(symbol, df_main))
                                            comp6 = self._get_htf_metrics(symbol, df_main)
                                            feats_q['htf_comp'] = dict(comp6)
                                            feats_q['ts15'] = float(comp6.get('ts15', 0.0))
                                            feats_q['ts60'] = float(comp6.get('ts60', 0.0))
                                            feats_q['rc15'] = float(comp6.get('rc15', 0.0))
                                            feats_q['rc60'] = float(comp6.get('rc60', 0.0))
                                        except Exception:
                                            pass
                                        pt.record_signal(symbol, {'side': 'long' if side=='Buy' else 'short', 'entry': float(entry), 'sl': float(sl), 'tp': float(tp)}, 0.0, False, feats_q, 'trend_pullback')
                                    except Exception:
                                        pass
                                    # Revert state & notify
                                    try:
                                        from strategy_pullback import revert_to_neutral
                                        revert_to_neutral(symbol)
                                    except Exception:
                                        pass
                                    try:
                                        if self.tg:
                                            comps = f"SR={qc.get('sr',0):.0f} HTF={qc.get('htf',0):.0f} BOS={qc.get('bos',0):.0f} Micro={qc.get('micro',0):.0f} Risk={qc.get('risk',0):.0f} Div={qc.get('div',0):.0f}"
                                            await self.tg.send_message(f"🟡 Rule-mode PHANTOM (stream): [{symbol}] Q={q:.1f} < {exec_min:.1f}\n{comps}")
                                    except Exception:
                                        pass
                                    # Append to events
                                    try:
                                        evts = self.shared.get('trend_events')
                                        if isinstance(evts, list):
                                            from datetime import datetime as _dt
                                            # Compute regime
                                            try:
                                                compm = self._get_htf_metrics(symbol, df_main)
                                                t15 = float(compm.get('ts15',0.0)); t60 = float(compm.get('ts60',0.0)); rc15 = float(compm.get('rc15',0.0)); rc60 = float(compm.get('rc60',0.0))
                                                reg = 'Trending' if (t15>=60 or t60>=60) else ('Ranging' if (rc15>=0.6 or rc60>=0.6) else 'Neutral')
                                            except Exception:
                                                reg = 'Unknown'
                                            evts.append({'ts': _dt.utcnow().isoformat()+'Z', 'symbol': symbol, 'text': f"Rule PHANTOM (stream) Q={q:.1f} comps: {comps} Regime:{reg}"})
                                            if len(evts) > 60:
                                                del evts[:len(evts)-60]
                                    except Exception:
                                        pass
                                    return
                            # Compute a quick ML score for logging (best-effort)
                            ml_score_se = 0.0
                            try:
                                if 'get_trend_scorer' in globals() and get_trend_scorer is not None:
                                    tr_scorer = get_trend_scorer()
                                    df = self.frames.get(symbol)
                                    feats = {}
                                    if df is not None and not df.empty:
                                        cl = df['close']; price = float(cl.iloc[-1]) if len(cl) else 0.0
                                        ys = cl.tail(20).values if len(cl) >= 20 else cl.values
                                        try:
                                            slope = np.polyfit(np.arange(len(ys)), ys, 1)[0]
                                        except Exception:
                                            slope = 0.0
                                        trend_slope_pct = float((slope / price) * 100.0) if price else 0.0
                                        ema20 = float(cl.ewm(span=20, adjust=False).mean().iloc[-1]) if len(cl) >= 20 else price
                                        ema50 = float(cl.ewm(span=50, adjust=False).mean().iloc[-1]) if len(cl) >= 50 else ema20
                                        ema_stack_score = 100.0 if (price > ema20 > ema50 or price < ema20 < ema50) else (50.0 if ema20 != ema50 else 0.0)
                                        rng_today = float(df['high'].iloc[-1] - df['low'].iloc[-1])
                                        med_range = float((df['high'] - df['low']).rolling(20).median().iloc[-1]) if len(df) >= 20 else max(1e-9, rng_today)
                                        range_expansion = float(rng_today / max(1e-9, med_range))
                                        prev = cl.shift(); tr = np.maximum(df['high'] - df['low'], np.maximum((df['high'] - prev).abs(), (df['low'] - prev).abs()))
                                        atr = float(tr.rolling(14).mean().iloc[-1]) if len(tr) >= 14 else float(tr.iloc[-1])
                                        atr_pct = float((atr / max(1e-9, price)) * 100.0) if price else 0.0
                                        feats = {
                                            'trend_slope_pct': trend_slope_pct,
                                            'ema_stack_score': ema_stack_score,
                                            'atr_pct': atr_pct,
                                            'range_expansion': range_expansion,
                                            'breakout_dist_atr': 0.0,
                                            'session': 'us', 'symbol_cluster': 3, 'volatility_regime': getattr(self, 'last_volatility_level', 'normal') if hasattr(self, 'last_volatility_level') else 'normal'
                                        }
                                    ml_score_se, _ = tr_scorer.score_signal({'side': side, 'entry': entry, 'sl': sl, 'tp': tp}, feats)
                            except Exception:
                                ml_score_se = 0.0
                            try:
                                # Include Qscore when available
                                msg = f"[{symbol}] ⚡ STREAM EXECUTE {side.upper()} @ {entry:.4f} SL {sl:.4f} TP {tp:.4f} | ML {ml_score_se:.1f}"
                                if rm_enabled:
                                    msg += f" | Q={q:.1f}"
                                logger.info(msg)
                            except Exception:
                                pass
                            # Acquire lock
                            import asyncio as _asyncio
                            if symbol not in self._exec_locks:
                                self._exec_locks[symbol] = _asyncio.Lock()
                            async with self._exec_locks[symbol]:
                                if symbol in self.book.positions:
                                    return
                                # Compute qty via sizer
                                meta_cfg = self.config.get('symbol_meta', {})
                                sm = meta_for(symbol, meta_cfg)
                                qty_step = float(sm.get('qty_step', 0.001)); min_qty = float(sm.get('min_qty', 0.001))
                                qty = self.sizer.qty_for(float(entry), float(sl), qty_step, min_qty)
                                if qty <= 0:
                                    return
                                # Place market
                                side_mkt = 'Buy' if side == 'long' else 'Sell'
                                self.bybit.place_market(symbol, side_mkt, qty, reduce_only=False)
                                # Set TP/SL with optional scale-out
                                try:
                                    sc_cfg = ((((self.config.get('trend', {}) or {}).get('exec', {}) or {}).get('scaleout', {}) or {}))
                                    if bool(sc_cfg.get('enabled', False)):
                                        frac = max(0.1, min(0.9, float(sc_cfg.get('fraction', 0.5))))
                                        tp1_r = float(sc_cfg.get('tp1_r', 1.6))
                                        tp2_r = float(sc_cfg.get('tp2_r', 3.0))
                                        R = abs(float(entry) - float(sl))
                                        if R > 0:
                                            if side == 'long':
                                                tp1 = float(entry) + tp1_r * R
                                                tp2 = float(entry) + tp2_r * R
                                                tp_side = "Sell"
                                            else:
                                                tp1 = float(entry) - tp1_r * R
                                                tp2 = float(entry) - tp2_r * R
                                                tp_side = "Buy"
                                            # Main TP to TP2
                                            try:
                                                self.bybit.set_tpsl(symbol, take_profit=float(tp2), stop_loss=float(sl))
                                            except Exception:
                                                pass
                                            # Partial reduce-only limit at TP1
                                            try:
                                                from position_mgr import round_step
                                                qty_step = float(meta_for(symbol, self.shared.get("meta", {})).get('qty_step', 0.001)) if hasattr(self, 'shared') else 0.001
                                                qty1 = round_step(float(qty) * frac, qty_step)
                                                tp1_resp = self.bybit.place_reduce_only_limit(symbol, tp_side, float(qty1), float(tp1), post_only=True, reduce_only=True)
                                            except Exception:
                                                tp1_resp = None
                                            # Track for BE move
                                            try:
                                                if not hasattr(self, '_scaleout'):
                                                    self._scaleout = {}
                                                self._scaleout[symbol] = {
                                                    'tp1': float(tp1), 'tp2': float(tp2), 'entry': float(entry),
                                                    'side': side, 'be_moved': False, 'move_be': bool(sc_cfg.get('move_sl_to_be', True)),
                                                    'qty1': float(qty1), 'qty2': max(0.0, float(qty) - float(qty1)),
                                                    'tp1_order_id': (tp1_resp.get('result', {}).get('orderId') if isinstance(tp1_resp, dict) else None)
                                                }
                                                if self.tg:
                                                    pct = int(round(frac*100))
                                                    oid = self._scaleout[symbol].get('tp1_order_id')
                                                    await self.tg.send_message(
                                                        f"📊 Scale-out armed: {symbol} TP1={tp1:.4f} qty1={qty1:.4f} ({pct}%) TP2={tp2:.4f} qty2={self._scaleout[symbol]['qty2']:.4f}"
                                                        + (f" ordId={oid}" if oid else "") + " | SL→BE after TP1"
                                                    )
                                            except Exception:
                                                pass
                                    else:
                                        # No scale-out: set TP/SL as provided
                                        try:
                                            self.bybit.set_tpsl(symbol, take_profit=float(tp), stop_loss=float(sl), qty=qty)
                                        except Exception:
                                            try:
                                                self.bybit.set_tpsl(symbol, take_profit=float(tp), stop_loss=float(sl))
                                            except Exception:
                                                pass
                                except Exception:
                                    pass
                                # Track locally
                                try:
                                    from position_mgr import Position
                                    # Attach qscore from last signal features if present
                                    try:
                                        qv = float(((getattr(self, '_last_signal_features', {}) or {}).get(symbol, {}) or {}).get('qscore', 0.0) or 0.0)
                                    except Exception:
                                        qv = 0.0
                                    self.book.positions[symbol] = Position(side=side, qty=float(qty), entry=float(entry), sl=float(sl), tp=float(tp), entry_time=datetime.utcnow(), strategy_name='trend_pullback', qscore=qv)
                                except Exception:
                                    pass
                                # Store per-position metadata for re-entry invalidation
                                try:
                                    if not hasattr(self, '_position_meta'):
                                        self._position_meta = {}
                                    blevel = float((meta or {}).get('breakout_level', 0.0)) if isinstance(meta, dict) else 0.0
                                    self._position_meta[symbol] = {
                                        'breakout_level': blevel,
                                        'side': side,
                                        'sl_mode': getattr(self._trend_settings, 'sl_mode', 'breakout'),
                                        'breakout_sl_buffer_atr': float(getattr(self._trend_settings, 'breakout_sl_buffer_atr', 0.30)),
                                        'min_r_pct': float(getattr(self._trend_settings, 'min_r_pct', 0.005)),
                                        'last_checked': None
                                    }
                                    # Persist hints for recovery
                                    try:
                                        if getattr(self, '_redis', None) is not None:
                                            import json as _json
                                            self._redis.set(f'openpos:strategy:{symbol}', 'trend_pullback')
                                            self._redis.set(f'openpos:entry:{symbol}', _json.dumps({'entry': float(entry), 'qty': float(qty), 'ts': datetime.utcnow().isoformat()}))
                                            self._redis.set(f'openpos:meta:{symbol}', _json.dumps(self._position_meta[symbol]))
                                    except Exception:
                                        pass
                                except Exception:
                                    pass
                                # Optionally cancel any active trend phantom for this symbol to avoid ML double count
                                try:
                                    ph_cfg = (self.config.get('phantom', {}) or {})
                                    if bool(ph_cfg.get('cancel_on_execute', True)):
                                        pt = self.shared.get('phantom_tracker') if hasattr(self, 'shared') else None
                                        if pt is not None and hasattr(pt, 'cancel_active'):
                                            pt.cancel_active(symbol)
                                except Exception:
                                    pass
                                # Mark stream-executed timestamp
                                try:
                                    from time import time as _now
                                    self._stream_executed[symbol] = float(_now())
                                except Exception:
                                    pass
                                # Telegram notify (include TP1/TP2 if scale-out armed)
                                try:
                                    if self.tg:
                                        msg = f"⚡ Stream Entry: {symbol} {side.upper()} qty={qty} entry={entry:.4f} sl={sl:.4f}"
                                        so = getattr(self, '_scaleout', {}).get(symbol) if hasattr(self, '_scaleout') else None
                                        if so and 'tp1' in so and 'tp2' in so:
                                            msg += f" TP1={so['tp1']:.4f} TP2={so['tp2']:.4f}"
                                        else:
                                            msg += f" tp={tp:.4f}"
                                        await self.tg.send_message(msg)
                                except Exception:
                                    pass
                                # Append to events feed
                                try:
                                    evts = self.shared.get('trend_events')
                                    if isinstance(evts, list):
                                        from datetime import datetime as _dt
                                        etxt = f"Stream EXECUTE qty={qty} entry={entry:.4f} sl={sl:.4f} tp={tp:.4f}"
                                        if rm_enabled:
                                            etxt += f" Q={q:.1f}"
                                        evts.append({'ts': _dt.utcnow().isoformat()+'Z', 'symbol': symbol, 'text': etxt})
                                        if len(evts) > 60:
                                            del evts[:len(evts)-60]
                                except Exception:
                                    pass
                                # Mark executed in Trend states snapshot
                                try:
                                    from strategy_pullback import mark_executed
                                    mark_executed(symbol)
                                except Exception:
                                    pass
                                # Record executed mirror for ML (was_executed=True) so retraining uses executed and phantom
                                try:
                                    from phantom_trade_tracker import get_phantom_tracker
                                    pt = get_phantom_tracker()
                                    # Build executed features similar to main path
                                    feats_exec = {}
                                    try:
                                        dfm = self.frames.get(symbol)
                                        if dfm is not None and not dfm.empty:
                                            clm = dfm['close']; price_m = float(clm.iloc[-1]) if len(clm) else 0.0
                                            ys_m = clm.tail(20).values if len(clm) >= 20 else clm.values
                                            try:
                                                slope_m = np.polyfit(np.arange(len(ys_m)), ys_m, 1)[0]
                                            except Exception:
                                                slope_m = 0.0
                                            ema20_m = float(clm.ewm(span=20, adjust=False).mean().iloc[-1]) if len(clm) >= 20 else price_m
                                            ema50_m = float(clm.ewm(span=50, adjust=False).mean().iloc[-1]) if len(clm) >= 50 else ema20_m
                                            ema_stack_m = 100.0 if (price_m > ema20_m > ema50_m or price_m < ema20_m < ema50_m) else (50.0 if ema20_m != ema50_m else 0.0)
                                            rng_today_m = float(dfm['high'].iloc[-1] - dfm['low'].iloc[-1])
                                            med_range_m = float((dfm['high'] - dfm['low']).rolling(20).median().iloc[-1]) if len(dfm) >= 20 else max(1e-9, rng_today_m)
                                            feats_exec = {
                                                'trend_slope_pct': float((slope_m / price_m) * 100.0) if price_m else 0.0,
                                                'ema_stack_score': float(ema_stack_m),
                                                'atr_pct': float(((np.maximum(dfm['high'] - dfm['low'], np.maximum((dfm['high'] - clm.shift()).abs(), (dfm['low'] - clm.shift()).abs()))).rolling(14).mean().iloc[-1]) / max(1e-9, price_m) * 100.0) if len(dfm) >= 14 else 0.0,
                                                'range_expansion': float(rng_today_m / max(1e-9, med_range_m)),
                                                'session': 'us',
                                                'symbol_cluster': 3,
                                                'volatility_regime': getattr(self, 'last_volatility_level', 'normal') if hasattr(self, 'last_volatility_level') else 'normal'
                                            }
                                            # Attach HTF snapshots + composite + qscore when available
                                            try:
                                                feats_exec['htf'] = dict(self._compute_symbol_htf_exec_metrics(symbol, dfm))
                                            except Exception:
                                                pass
                                            try:
                                                compE = self._get_htf_metrics(symbol, dfm)
                                                feats_exec['htf_comp'] = dict(compE)
                                                feats_exec['ts15'] = float(compE.get('ts15', 0.0))
                                                feats_exec['ts60'] = float(compE.get('ts60', 0.0))
                                                feats_exec['rc15'] = float(compE.get('rc15', 0.0))
                                                feats_exec['rc60'] = float(compE.get('rc60', 0.0))
                                            except Exception:
                                                pass
                                            try:
                                                if rm_enabled:
                                                    feats_exec['qscore'] = float(q)
                                            except Exception:
                                                pass
                                    except Exception:
                                        feats_exec = {}
                                    pt.record_signal(symbol, {'side': str(side), 'entry': float(entry), 'sl': float(sl), 'tp': float(tp)}, float(ml_score_se or 0.0), True, feats_exec, 'trend_pullback')
                                except Exception:
                                    pass
                        except Exception as e:
                            logger.debug(f"Stream execute trend error [{symbol}]: {e}")
                    def _entry_exec(symbol, side, entry, sl, tp, m):
                        try:
                            asyncio.create_task(_stream_execute_trend(symbol, side, entry, sl, tp, m or {}))
                        except Exception:
                            pass
                    set_trend_entry_executor(_entry_exec)
                except Exception:
                    logger.debug("Failed to wire trend notifier to Telegram")

                # Wire scalp phantom notifications to Telegram (ensure after Telegram init)
                try:
                    if SCALP_AVAILABLE and get_scalp_phantom_tracker is not None:
                        scpt = get_scalp_phantom_tracker()
                        scpt.set_notifier(self._notify_scalp_phantom)
                        # Backfill open notifications for already-active scalp phantoms
                        try:
                            import asyncio as _asyncio
                            for lst in (getattr(scpt, 'active', {}) or {}).values():
                                for ph in (lst or []):
                                    res = self._notify_scalp_phantom(ph)
                                    if _asyncio.iscoroutine(res):
                                        _asyncio.create_task(res)
                        except Exception:
                            pass
                except Exception:
                    logger.debug("Failed to set scalp notifier after Telegram init")

                # Phantom notifications are disabled (only executed high-ML opens + all closes sent elsewhere)

                # MR phantom notifier disabled
                # One-time backfill of MR phantom outcomes into Enhanced MR ML (guarded by config; default OFF)
                try:
                    mr_bf_enabled = False
                    try:
                        mr_bf_enabled = bool((((cfg.get('enhanced_mr', {}) or {}).get('backfill', {}) or {}).get('enabled', False)))
                    except Exception:
                        mr_bf_enabled = False
                    if enhanced_mr_scorer is not None and mr_bf_enabled:
                        backfilled_mr = False
                        client = getattr(enhanced_mr_scorer, 'redis_client', None)
                        if client is not None:
                            try:
                                if client.get('ml:backfill:enhanced_mr:done') == '1':
                                    backfilled_mr = True
                            except Exception:
                                pass
                        if not backfilled_mr:
                            fed_mr = 0
                            try:
                                mrpt = get_mr_phantom_tracker()
                                for rec in mrpt.get_learning_data():
                                    try:
                                        outcome = 'win' if int(rec.get('outcome', 0)) == 1 else 'loss'
                                        sig = {
                                            'features': rec.get('features', {}),
                                            'enhanced_features': rec.get('enhanced_features', {}),
                                            'score': float(rec.get('score', 0) or 0.0),
                                            'symbol': rec.get('symbol', 'UNKNOWN'),
                                            'was_executed': False
                                        }
                                        enhanced_mr_scorer.record_outcome(sig, outcome, float(rec.get('pnl_percent', 0.0) or 0.0))
                                        fed_mr += 1
                                    except Exception:
                                        pass
                                if client is not None:
                                    try:
                                        client.set('ml:backfill:enhanced_mr:done', '1')
                                    except Exception:
                                        pass
                                if fed_mr > 0:
                                    logger.info(f"🌀 MR ML backfill: fed {fed_mr} phantom outcomes into Enhanced MR ML store")
                                # Attempt a startup retrain after backfill if trainable
                                try:
                                    ok = enhanced_mr_scorer.startup_retrain()
                                    logger.info(f"🌀 MR ML startup retrain attempted: {'✅ success' if ok else '⚠️ skipped'}")
                                except Exception:
                                    pass
                                # Recalibrate executed-count to executed store length
                                try:
                                    if client is not None:
                                        exec_count = len(client.lrange('enhanced_mr:trades', 0, -1))
                                        enhanced_mr_scorer.completed_trades = exec_count
                                        client.set('enhanced_mr:completed_trades', str(exec_count))
                                        logger.info(f"🧮 MR executed count recalibrated to {exec_count} from executed store")
                                except Exception:
                                    pass
                                # Immediately evaluate MR promotion after backfill/retrain
                                try:
                                    mr_stats = enhanced_mr_scorer.get_enhanced_stats()
                                    prom_cfg = (self.config.get('mr', {}) or {}).get('promotion', {})
                                    if prom_cfg.get('enabled', False):
                                        mp = shared.get('mr_promotion', {})
                                        from datetime import datetime as _dt
                                        cur_day = _dt.utcnow().strftime('%Y%m%d')
                                        if mp.get('day') != cur_day:
                                            mp['day'] = cur_day
                                            mp['count'] = 0
                                        recent_wr = float(mr_stats.get('recent_win_rate', 0.0))
                                        recent_n = int(mr_stats.get('recent_trades', 0))
                                        total_exec = int(mr_stats.get('completed_trades', 0))
                                        promote_wr = float(prom_cfg.get('promote_wr', 50.0))
                                        demote_wr = float(prom_cfg.get('demote_wr', 30.0))
                                        min_recent = int(prom_cfg.get('min_recent', 20))
                                        min_total = int(prom_cfg.get('min_total_trades', 50))
                                        if not mp.get('active') and recent_n >= min_recent and total_exec >= min_total and recent_wr >= promote_wr:
                                            mp['active'] = True
                                            logger.info(f"🚀 MR Promotion activated (startup eval) WR {recent_wr:.1f}% (N={recent_n})")
                                            if self.tg:
                                                try:
                                                    await self.tg.send_message(f"🌀 MR Promotion: Activated (WR {recent_wr:.1f}% ≥ {promote_wr:.0f}%) [startup]")
                                                except Exception:
                                                    pass
                                        shared['mr_promotion'] = mp
                                except Exception:
                                    pass
                                # Immediately evaluate Trend promotion (corking) after startup
                                try:
                                    tr_cfg = (self.config.get('trend', {}) or {}).get('promotion', {})
                                    if not bool(tr_cfg.get('enabled', False)):
                                        raise Exception('Trend promotion disabled')
                                    if tr_cfg.get('enabled', False):
                                        tp = shared.get('trend_promotion', {})
                                        from datetime import datetime as _dt
                                        cur_day = _dt.utcnow().strftime('%Y%m%d')
                                        if tp.get('day') != cur_day:
                                            tp['day'] = cur_day
                                            tp['count'] = 0
                                        tr_scorer = shared.get('trend_scorer')
                                        tr_stats = tr_scorer.get_stats() if tr_scorer else {}
                                        recent_wr = float(tr_stats.get('recent_win_rate', 0.0))
                                        recent_n = int(tr_stats.get('recent_trades', 0))
                                        total_exec = int(tr_stats.get('executed_count', 0))
                                        promote_wr = float(tr_cfg.get('promote_wr', 55.0))
                                        demote_wr = float(tr_cfg.get('demote_wr', 35.0))
                                        min_recent = int(tr_cfg.get('min_recent', 30))
                                        min_total = int(tr_cfg.get('min_total_trades', 100))
                                        if (not tp.get('active')) and recent_n >= min_recent and total_exec >= min_total and recent_wr >= promote_wr:
                                            tp['active'] = True
                                            logger.info(f"🚀 Trend Promotion activated (startup eval) WR {recent_wr:.1f}% (N={recent_n})")
                                            if self.tg:
                                                try:
                                                    await self.tg.send_message(f"🚀 Trend Promotion: Activated (WR {recent_wr:.1f}% ≥ {promote_wr:.0f}%) [startup]")
                                                except Exception:
                                                    pass
                                        shared['trend_promotion'] = tp
                                except Exception:
                                    pass
                            except Exception as e:
                                logger.debug(f"MR ML backfill error: {e}")
                except Exception as e:
                    logger.debug(f"Failed to set MR notifier/backfill: {e}")
                # Wire scalp phantom notifications to Telegram (if scalp modules available)
                try:
                    if SCALP_AVAILABLE and get_scalp_phantom_tracker is not None:
                        scpt = get_scalp_phantom_tracker()
                        # Downtime backfill for Scalp phantoms using exchange klines (guarded by config; default OFF)
                        try:
                            scalp_bf_active = False
                            try:
                                scalp_bf_active = bool((((cfg.get('scalp', {}) or {}).get('backfill', {}) or {}).get('active_labeling', False)))
                            except Exception:
                                scalp_bf_active = False
                            if scalp_bf_active and hasattr(scpt, 'backfill_active') and self.bybit is not None:
                                def _fetch3(sym: str, start_ms: int, end_ms: Optional[int] = None):
                                    try:
                                        rows = self.bybit.get_klines(sym, '3', limit=200, start=start_ms, end=end_ms) or []
                                        out = []
                                        for r in rows:
                                            # Bybit v5 kline list: [start, open, high, low, close, volume, turnover]
                                            try:
                                                out.append({'high': float(r[2]), 'low': float(r[3])})
                                            except Exception:
                                                pass
                                        return out
                                    except Exception:
                                        return []
                                closed = scpt.backfill_active(_fetch3)
                                if closed:
                                    logger.info(f"🩳 Scalp phantom backfill: closed {closed} due to downtime recovery")
                        except Exception as _bf:
                            logger.debug(f"Scalp phantom backfill skipped: {_bf}")
                        # Periodic timeout sweep to ensure stale phantoms are cancelled
                        try:
                            async def _scalp_timeout_sweeper():
                                while self.running:
                                    try:
                                        if hasattr(scpt, 'sweep_timeouts'):
                                            cnt = scpt.sweep_timeouts()
                                            if cnt:
                                                logger.info(f"🩳 Scalp timeout sweep: cancelled {cnt} stale phantoms")
                                    except Exception:
                                        pass
                                    await asyncio.sleep(60)
                            self._create_task(_scalp_timeout_sweeper())
                        except Exception:
                            pass
                        # Periodic active-phantom updater to close on TP/SL using latest bars
                        try:
                            async def _scalp_active_updater():
                                while self.running:
                                    try:
                                        # Snapshot active symbols to avoid holding lock
                                        act = dict(scpt.active) if isinstance(getattr(scpt, 'active', None), dict) else {}
                                        for s in list(act.keys()):
                                            try:
                                                df3 = self.frames_3m.get(s) if hasattr(self, 'frames_3m') else None
                                                dfm = self.frames.get(s)
                                                src_df = df3 if (df3 is not None and not df3.empty) else dfm
                                                if src_df is None or getattr(src_df, 'empty', True):
                                                    continue
                                                # Use last close as current_price for labeling; pass full frame for extremes
                                                cur_price = float(src_df['close'].iloc[-1])
                                                if hasattr(scpt, 'update_scalp_phantom_prices'):
                                                    scpt.update_scalp_phantom_prices(s, cur_price, df=src_df)
                                            except Exception:
                                                pass
                                    except Exception:
                                        pass
                                    await asyncio.sleep(10)
                            self._create_task(_scalp_active_updater())
                        except Exception:
                            pass
                        # Scalp phantom notifier disabled
                        # Compute promotion readiness from config and stats
                        try:
                            scalp_cfg = cfg.get('scalp', {})
                            promote_enabled = bool(scalp_cfg.get('promote_enabled', False))
                            min_samples = int(scalp_cfg.get('promote_min_samples', 200))
                            target_wr = float(scalp_cfg.get('promote_min_wr', 55.0))
                            st = scpt.get_scalp_phantom_stats()
                            samples = int(st.get('total', 0))
                            wr = float(st.get('wr', 0.0))
                            shared['scalp_promoted'] = bool(promote_enabled and samples >= min_samples and wr >= target_wr)
                            logger.info(f"🩳 Scalp promotion readiness: enabled={promote_enabled}, samples={samples}, wr={wr:.1f}%, ready={shared['scalp_promoted']}")
                        except Exception as e:
                            logger.debug(f"Scalp promotion readiness check failed: {e}")
                        # One-time backfill of Scalp phantom outcomes into Scalp ML (guarded by config; default OFF)
                        try:
                            scalp_bf_ml = False
                            try:
                                scalp_bf_ml = bool((((cfg.get('scalp', {}) or {}).get('backfill', {}) or {}).get('ml_enabled', False)))
                            except Exception:
                                scalp_bf_ml = False
                            s_scorer = get_scalp_scorer() if (SCALP_AVAILABLE and get_scalp_scorer is not None) else None
                            backfilled = False
                            if scalp_bf_ml and s_scorer and getattr(s_scorer, 'redis_client', None):
                                try:
                                    if s_scorer.redis_client.get('ml:backfill:scalp:done') == '1':
                                        backfilled = True
                                except Exception:
                                    pass
                            if scalp_bf_ml and s_scorer and not backfilled:
                                fed = 0
                                for trades in scpt.completed.values():
                                    for t in trades:
                                        if getattr(t, 'outcome', None) in ('win','loss'):
                                            try:
                                                sig = {'features': t.features or {}, 'was_executed': False}
                                                s_scorer.record_outcome(sig, t.outcome, float(t.pnl_percent or 0.0))
                                                fed += 1
                                            except Exception:
                                                pass
                                # Set redis flag to avoid duplication on future starts
                                try:
                                    if scalp_bf_ml and s_scorer and getattr(s_scorer, 'redis_client', None):
                                        s_scorer.redis_client.set('ml:backfill:scalp:done', '1')
                                except Exception:
                                    pass
                                if fed > 0:
                                    logger.info(f"🩳 Scalp ML backfill: fed {fed} phantom outcomes into ML store")
                                # Attempt a startup retrain after backfill if trainable
                                try:
                                    if scalp_bf_ml:
                                        ri = s_scorer.get_retrain_info()
                                        if ri.get('trainable_size', 0) >= getattr(s_scorer, 'MIN_TRADES_FOR_ML', 50):
                                            ok = s_scorer.startup_retrain()
                                            logger.info(f"🩳 Scalp ML startup retrain attempted: {'✅ success' if ok else '⚠️ skipped'}")
                                except Exception:
                                    pass
                        except Exception as e:
                            logger.debug(f"Scalp ML backfill error: {e}")
                except Exception as e:
                    logger.debug(f"Failed to set scalp notifier: {e}")
                break  # Success
            except Exception as e:
                if "Conflict" in str(e) and attempt < max_retries - 1:
                    logger.warning(f"Telegram conflict detected, retrying in 5 seconds...")
                    await asyncio.sleep(5)
                    continue
                else:
                    logger.error(f"Telegram bot failed to start: {e}")
                    self.tg = None
                    break
        
        # Background training disabled - bot runs normally
        # Uncomment the lines below to enable background ML training:
        # try:
        #     from background_initial_trainer import get_background_trainer
        #     background_trainer = get_background_trainer(self.tg)
        #     training_started = await background_trainer.start_if_needed()
        #     if training_started:
        #         logger.info("🎯 Background ML training started - will run while bot trades")
        # except Exception as e:
        #     logger.error(f"Failed to start background trainer: {e}")
        logger.info("🚀 Bot starting in normal mode - background training disabled")
        
        # Signal tracking
        last_signal_time = {}
        signal_cooldown = 60  # Seconds between signals per symbol
        last_position_check = datetime.now()
        position_check_interval = 30  # Check for closed positions every 30 seconds
        
        # Add periodic logging summary to reduce log spam
        last_summary_log = datetime.now()
        summary_log_interval = 300  # Log summary every 5 minutes
        candles_processed = 0
        signals_detected = 0
        
        # Initialize ML breakout states dictionary
        ml_breakout_states = {}
        
        # Schedule weekly cluster updates
        last_cluster_update = datetime.now()
        cluster_update_interval = 7 * 24 * 60 * 60  # 7 days in seconds
        
        # Create background task for cluster updates
        async def weekly_cluster_updater():
            """Background task to update clusters weekly"""
            while self.running:
                try:
                    # Wait for next update time
                    await asyncio.sleep(cluster_update_interval)
                    
                    if self.running:  # Check if still running
                        logger.info("🔄 Running scheduled weekly cluster update...")
                        await self.auto_generate_enhanced_clusters()
                        
                except Exception as e:
                    logger.error(f"Error in weekly cluster updater: {e}")
                    # Continue running even if update fails
        
        # Start the weekly updater task
        self._create_task(weekly_cluster_updater())
        logger.info("📅 Started weekly cluster update scheduler")

        # Start DB writer task if a queue is available (offloads blocking DB writes)
        try:
            if getattr(self, '_db_queue', None) is not None:
                async def _db_writer():
                    loop = asyncio.get_running_loop()
                    while self.running:
                        try:
                            item = await self._db_queue.get()
                            if not item:
                                continue
                            kind, sym, row = item
                            if kind == '3m' and self.storage is not None:
                                await loop.run_in_executor(None, self.storage.save_candles_3m, sym, row)
                        except Exception:
                            await asyncio.sleep(0.05)
                self._create_task(_db_writer())
        except Exception:
            pass

        # Regime summary reporter (daily/weekly)
        try:
            rep_cfg = ((cfg.get('reporting', {}) or {}).get('regime_summary', {}) or {})
            if bool(rep_cfg.get('enabled', True)):
                async def _regime_summary_reporter():
                    import datetime as _dt
                    import asyncio as _asyncio
                    last_daily = None
                    last_weekly = None
                    while self.running:
                        now = _dt.datetime.utcnow()
                        hour_target = int(rep_cfg.get('hour_utc', 0))
                        do_daily = bool(rep_cfg.get('daily', True)) and (now.hour == hour_target) and (last_daily is None or (now - last_daily).total_seconds() > 23*3600)
                        do_weekly = bool(rep_cfg.get('weekly', True)) and (now.weekday() == 0) and (now.hour == hour_target) and (last_weekly is None or (now - last_weekly).total_seconds() > 6*24*3600)
                        if do_daily or do_weekly:
                            try:
                                from phantom_trade_tracker import get_phantom_tracker
                                pt = get_phantom_tracker()
                                cutoff = now - _dt.timedelta(days=(7 if do_weekly else 1))
                                buckets = {}
                                for trades in getattr(pt, 'phantom_trades', {}).values():
                                    for p in trades:
                                        try:
                                            if not getattr(p, 'exit_time', None):
                                                continue
                                            if getattr(p, 'exit_time') < cutoff:
                                                continue
                                            strat = str(getattr(p, 'strategy_name','') or '')
                                            feats = getattr(p, 'features', {}) or {}
                                            t15 = float(feats.get('ts15', 0.0)); t60 = float(feats.get('ts60', 0.0)); rc15 = float(feats.get('rc15', 0.0)); rc60 = float(feats.get('rc60', 0.0))
                                            reg = 'Trending' if (t15 >= 60 or t60 >= 60) else ('Ranging' if (rc15 >= 0.6 or rc60 >= 0.6) else 'Neutral')
                                            key = (('Trend' if strat.startswith('trend') else ('Range' if strat.startswith('range') else 'Other')), reg)
                                            b = buckets.get(key, {'n': 0, 'w': 0})
                                            b['n'] += 1
                                            b['w'] += 1 if getattr(p, 'outcome', '') == 'win' else 0
                                            buckets[key] = b
                                        except Exception:
                                            continue
                                title = '📊 Regime Summary (7d)' if do_weekly else '📊 Regime Summary (24h)'
                                lines = [title, '']
                                for (sg, reg), b in sorted(buckets.items()):
                                    wr = (b['w'] / b['n'] * 100.0) if b['n'] else 0.0
                                    lines.append(f'• {sg} — {reg}: WR {wr:.1f}% (N={b["n"]})')
                                msg = '\n'.join(lines) if len(lines) > 2 else (title + '\nNo data')
                                try:
                                    if self.tg:
                                        await self.tg.send_message(msg)
                                except Exception:
                                    pass
                                try:
                                    evts = self.shared.get('trend_events')
                                    if isinstance(evts, list):
                                        evts.append({'ts': now.isoformat() + 'Z', 'symbol': '-', 'text': title})
                                        if len(evts) > 60:
                                            del evts[:len(evts) - 60]
                                except Exception:
                                    pass
                                if do_daily:
                                    last_daily = now
                                if do_weekly:
                                    last_weekly = now
                            except Exception:
                                pass
                        await _asyncio.sleep(300)
                self._create_task(_regime_summary_reporter())
        except Exception:
            pass

        # Start Range FBO scanner (phantom-first; can execute when enabled)
        try:
            if bool(((cfg.get('range', {}) or {}).get('enabled', True))):
                async def _range_fbo_scanner():
                    from strategy_range_fbo import detect_range_fbo_signal
                    settings = cfg.get('range', {}) or {}
                    # Heartbeat control (lightweight "no FBO" per 15m bar)
                    log_cfg = (settings.get('logging') or {})
                    hb_enabled = bool(log_cfg.get('heartbeat', False))
                    decision_only = bool(log_cfg.get('decision_only', False))
                    last_bar_ts = {}
                    while self.running:
                        try:
                            for sym in list(symbols):
                                try:
                                    df = self.frames.get(sym)
                                    if df is None or df.empty:
                                        continue
                                    # Track once-per-bar heartbeat per symbol
                                    bar_ts = None
                                    try:
                                        bar_ts = df.index[-1]
                                    except Exception:
                                        bar_ts = None
                                    sig = detect_range_fbo_signal(df, settings, sym)
                                    if not sig:
                                        # Optional per-bar heartbeat when no signal
                                        try:
                                            if hb_enabled and not decision_only and bar_ts is not None:
                                                prev_ts = last_bar_ts.get(sym)
                                                if prev_ts != bar_ts:
                                                    # Compute a couple of light diagnostics for context
                                                    try:
                                                        lookback = int((settings.get('lookback') or 40))
                                                    except Exception:
                                                        lookback = 40
                                                    try:
                                                        high = df['high']; low = df['low']; close = df['close']
                                                        rng_high = float(high.rolling(lookback).max().iloc[-2]) if len(high) >= lookback+2 else 0.0
                                                        rng_low = float(low.rolling(lookback).min().iloc[-2]) if len(low) >= lookback+2 else 0.0
                                                        width_pct = ((rng_high - rng_low) / max(1e-9, rng_low)) if (rng_high > rng_low > 0) else 0.0
                                                        logger.info(f"[{sym}] 📦 Range: no FBO (width={width_pct*100.0:.1f}% bounds={rng_low:.4f}-{rng_high:.4f})")
                                                        # Emit a heartbeat decision line with lightweight reasons
                                                        try:
                                                            width_min = float(settings.get('width_min_pct', 0.01) or 0.01)
                                                            width_max = float(settings.get('width_max_pct', 0.10) or 0.10)
                                                            reasons = []
                                                            if width_pct < width_min:
                                                                reasons.append('width_too_narrow')
                                                            if width_pct > width_max:
                                                                reasons.append('width_too_wide')
                                                            rng_w = max(1e-9, (rng_high - rng_low))
                                                            cl = float(close.iloc[-1]) if len(close) else 0.0
                                                            # Consider near-edge if within 15% of band width
                                                            near_edge = min(abs(cl - rng_low), abs(rng_high - cl)) <= (0.15 * rng_w)
                                                            # Update per-symbol Range state
                                                            try:
                                                                self._range_symbol_state[sym] = {
                                                                    'in_range': bool((rng_low < cl < rng_high)),
                                                                    'near_edge': bool(near_edge)
                                                                }
                                                            except Exception:
                                                                pass
                                                            if not near_edge:
                                                                reasons.append('not_near_edge')
                                                            if not reasons:
                                                                reasons.append('no_fbo')
                                                            # Reasons histogram
                                                            try:
                                                                for r in reasons:
                                                                    self._range_reasons[r] = self._range_reasons.get(r, 0) + 1
                                                            except Exception:
                                                                pass
                                                            logger.info(f"[{sym}] 🧮 Range heartbeat: decision=no_signal reasons={','.join(reasons)}")
                                                        except Exception:
                                                            pass
                                                    except Exception:
                                                        logger.info(f"[{sym}] 📦 Range: no FBO")
                                                    last_bar_ts[sym] = bar_ts
                                        except Exception:
                                            pass
                                        continue
                                    # Phantom-only routing
                                    try:
                                        q, qc, qr = self._compute_qscore_range(sym, sig.side, df, self.frames_3m.get(sym) if hasattr(self, 'frames_3m') else None)
                                    except Exception:
                                        q, qc, qr = 50.0, {}, []
                                    # Build features
                                    # Build features for Range FBO ML training
                                    feats = {
                                        'range_high': float(sig.meta.get('range_high', 0.0)),
                                        'range_low': float(sig.meta.get('range_low', 0.0)),
                                        'range_mid': float(sig.meta.get('range_mid', 0.0)),
                                        'range_width_pct': float(sig.meta.get('range_width_pct', 0.0)),
                                        'fbo_type': str(sig.meta.get('fbo_type', '')),
                                        'wick_ratio': float(sig.meta.get('wick_ratio', 0.0)),
                                        'retest_ok': bool(sig.meta.get('retest_ok', False)),
                                        'qscore': float(q),
                                        'qscore_components': dict(qc),
                                        'qscore_reasons': list(qr),
                                        'volatility_regime': 'normal',
                                    }
                                    # Volatility regime, range position, and edge distance
                                    try:
                                        reg = get_enhanced_market_regime(df, sym)
                                        feats['volatility_regime'] = str(getattr(reg, 'volatility_level', 'normal') or 'normal')
                                    except Exception:
                                        pass
                                    try:
                                        clp = float(df['close'].iloc[-1]) if len(df) else 0.0
                                        rh = float(feats['range_high']); rl = float(feats['range_low'])
                                        rng_w = max(1e-9, (rh - rl))
                                        feats['range_pos_pct'] = float(min(1.0, max(0.0, (clp - rl) / rng_w)))
                                        prev = df['close'].shift()
                                        trarr = np.maximum(df['high'] - df['low'], np.maximum((df['high'] - prev).abs(), (df['low'] - prev).abs()))
                                        atr14 = float(trarr.rolling(14).mean().iloc[-1]) if len(trarr) >= 14 else float(trarr.iloc[-1])
                                        edge_dist = min(abs(clp - rl), abs(rh - clp))
                                        feats['edge_distance_atr'] = float(edge_dist / max(1e-9, atr14))
                                    except Exception:
                                        pass
                                    # Add session and symbol cluster context
                                    try:
                                        from datetime import datetime as _dt
                                        hr = _dt.utcnow().hour
                                        feats['session'] = 'asian' if 0 <= hr < 8 else ('european' if hr < 16 else 'us')
                                    except Exception:
                                        feats['session'] = 'us'
                                    try:
                                        from symbol_clustering import load_symbol_clusters
                                        clusters = load_symbol_clusters()
                                        feats['symbol_cluster'] = int(clusters.get(sym, 3))
                                    except Exception:
                                        feats['symbol_cluster'] = 3
                                    # Compute target RR for training quality
                                    try:
                                        entry = float(sig.entry); sl_v = float(sig.sl); tp_v = float(sig.tp)
                                        R = abs(entry - sl_v)
                                        feats['rr_target'] = float(abs(tp_v - entry) / R) if R > 0 else 0.0
                                    except Exception:
                                        feats['rr_target'] = 0.0
                                    # Attach HTF composite for context
                                    try:
                                        comp = self._get_htf_metrics(sym, df)
                                        feats['ts15'] = float(comp.get('ts15', 0.0)); feats['ts60'] = float(comp.get('ts60', 0.0))
                                        feats['rc15'] = float(comp.get('rc15', 0.0)); feats['rc60'] = float(comp.get('rc60', 0.0))
                                    except Exception:
                                        pass
                                    # SR confluence details and optional TP2 snapping
                                    try:
                                        sr_cfg = ((settings.get('sr_confluence') or {}))
                                        if bool(sr_cfg.get('enabled', True)):
                                            from multi_timeframe_sr import mtf_sr
                                            # ATR(14)
                                            prev = df['close'].shift()
                                            trarr = np.maximum(df['high'] - df['low'], np.maximum((df['high'] - prev).abs(), (df['low'] - prev).abs()))
                                            atr14 = float(trarr.rolling(14).mean().iloc[-1]) if len(trarr) >= 14 else float(trarr.iloc[-1])
                                            levels = mtf_sr.get_price_validated_levels(sym, float(df['close'].iloc[-1]))
                                            edge = float(feats.get('range_high') if sig.side=='short' else feats.get('range_low'))
                                            # Filter appropriate type
                                            lvlist = [(lv, st, t) for (lv, st, t) in levels if (t=='resistance' if sig.side=='short' else t=='support')]
                                            if lvlist and atr14>0:
                                                level, strength, ltype = min(lvlist, key=lambda x: abs(x[0] - edge))
                                                dist_atr = abs(level - edge) / atr14
                                                feats['sr_confluence'] = {'edge': 'high' if sig.side=='short' else 'low', 'level': float(level), 'strength': float(strength), 'type': ltype, 'dist_atr': float(dist_atr)}
                                                # Optional TP2 snapping
                                                if bool(sr_cfg.get('snap_tp2', True)) and dist_atr <= float(sr_cfg.get('max_dist_atr', 0.30)):
                                                    try:
                                                        sig.tp = float(level)
                                                    except Exception:
                                                        pass
                                    except Exception:
                                        pass
                                    # Verbose range analysis log (optional)
                                    try:
                                        log_cfg = (settings.get('logging') or {})
                                        if bool(log_cfg.get('verbose', False)) and not bool(log_cfg.get('decision_only', False)):
                                            rng_high = feats.get('range_high'); rng_low = feats.get('range_low'); rng_mid = feats.get('range_mid')
                                            width_pct = feats.get('range_width_pct'); fbo_t = feats.get('fbo_type'); wick = feats.get('wick_ratio')
                                            t15v = float(feats.get('ts15',0.0)); t60v = float(feats.get('ts60',0.0)); rc15v = float(feats.get('rc15',0.0)); rc60v = float(feats.get('rc60',0.0))
                                            reg = 'Trending' if (t15v>=60 or t60v>=60) else ('Ranging' if (rc15v>=0.6 or rc60v>=0.6) else 'Neutral')
                                            comps_s = f"RNG={qc.get('rng',0):.0f} FBO={qc.get('fbo',0):.0f} PROX={qc.get('prox',0):.0f} Micro={qc.get('micro',0):.0f} Risk={qc.get('risk',0):.0f}"
                                            logger.info(
                                                f"[{sym}] 📦 Range analysis: high={float(rng_high):.4f} low={float(rng_low):.4f} mid={float(rng_mid):.4f} "
                                                f"width={float(width_pct)*100.0:.1f}% fbo={fbo_t} wick={float(wick):.2f} q={q:.1f} comps: {comps_s} Regime:{reg}"
                                            )
                                    except Exception:
                                        pass
                                    exec_cfg = (settings.get('exec') or {})
                                    # Qscore-only gating for Range (default true)
                                    try:
                                        q_only_rg = bool(exec_cfg.get('qscore_only', True))
                                    except Exception:
                                        q_only_rg = True
                                    exec_enabled = bool(exec_cfg.get('enabled', False)) and not bool(settings.get('phantom_only', True))
                                    did_execute = False
                                    # Range exec gating
                                    if exec_enabled:
                                        try:
                                            # Existing position guard
                                            if sym not in self.book.positions:
                                                if q_only_rg:
                                                    # Execute solely on Qscore threshold
                                                    # Use learned threshold if available
                                                    thr_q = float((settings.get('rule_mode') or {}).get('execute_q_min', 78))
                                                    try:
                                                        ctx = {'session': self._session_label(), 'volatility_regime': getattr(htf, 'volatility_level', 'global') if 'htf' in locals() and htf else 'global'}
                                                        rm_rg = (((settings.get('rule_mode') or {})))
                                                        use_adapter = bool(rm_rg.get('adapter_enabled', True))
                                                        if use_adapter and hasattr(self, '_qadapt_range') and self._qadapt_range:
                                                            thr_q = float(self._qadapt_range.get_threshold(ctx, floor=60.0, ceiling=95.0, default=thr_q))
                                                    except Exception:
                                                        pass
                                                    if q >= thr_q and hasattr(self, '_range_exec_runner'):
                                                        try:
                                                            # Temporarily adjust risk percent if provided
                                                            old_risk = None
                                                            try:
                                                                old_risk = self.sizer.risk.risk_percent
                                                                self.sizer.risk.risk_percent = float(exec_cfg.get('risk_percent', old_risk or 1.0))
                                                            except Exception:
                                                                pass
                                                            # Pre-exec decision notification
                                                            try:
                                                                if self.tg:
                                                                    msg = f"🟢 Range EXECUTE: {sym} {sig.side.upper()} Q={q:.1f} (≥ {float((settings.get('rule_mode') or {}).get('execute_q_min', 78)):.1f})\n{comps}"
                                                                    if regime_label:
                                                                        msg += f"\nRegime: {regime_label}"
                                                                    await self.tg.send_message(msg)
                                                            except Exception:
                                                                pass
                                                            await self._range_exec_runner(sig, q)
                                                            did_execute = True
                                                        finally:
                                                            try:
                                                                if old_risk is not None:
                                                                    self.sizer.risk.risk_percent = old_risk
                                                            except Exception:
                                                                pass
                                                else:
                                                    # Legacy neutral and SR checks retained only when qscore_only disabled
                                                    # Daily cap
                                                    if not hasattr(self, '_range_exec_counter'):
                                                        self._range_exec_counter = {'day': None, 'count': 0}
                                                    from datetime import datetime as _dt
                                                    day_str = _dt.utcnow().strftime('%Y%m%d')
                                                    if self._range_exec_counter['day'] != day_str:
                                                        self._range_exec_counter = {'day': day_str, 'count': 0}
                                                    if self._range_exec_counter['count'] < int(exec_cfg.get('daily_cap', 3)):
                                                        # HTF neutrality gate: avoid strong trend
                                                        t15 = float(feats.get('ts15', 0.0)); t60 = float(feats.get('ts60', 0.0))
                                                        sr_ok = True
                                                        try:
                                                            src = feats.get('sr_confluence', {}) or {}
                                                            if src and float(src.get('dist_atr', 1.0)) > float((settings.get('sr_confluence') or {}).get('max_dist_atr', 0.30)) and bool((settings.get('sr_confluence') or {}).get('required', False)):
                                                                sr_ok = False
                                                        except Exception:
                                                            sr_ok = True
                                                    if max(t15, t60) < 60 and sr_ok and q >= float((settings.get('rule_mode') or {}).get('execute_q_min', 78)):
                                                        # Execute using shared runner when ready
                                                        if hasattr(self, '_range_exec_runner'):
                                                            try:
                                                                # Temporarily adjust risk percent if provided
                                                                old_risk = None
                                                                try:
                                                                    old_risk = self.sizer.risk.risk_percent
                                                                    self.sizer.risk.risk_percent = float(exec_cfg.get('risk_percent', old_risk or 1.0))
                                                                except Exception:
                                                                    pass
                                                                # Pre-exec decision notification
                                                                try:
                                                                    if self.tg:
                                                                        msg = f"🟢 Range EXECUTE: {sym} {sig.side.upper()} Q={q:.1f} (≥ {float((settings.get('rule_mode') or {}).get('execute_q_min', 78)):.1f})\n{comps}"
                                                                        if regime_label:
                                                                            msg += f"\nRegime: {regime_label}"
                                                                        await self.tg.send_message(msg)
                                                                except Exception:
                                                                    pass
                                                                await self._range_exec_runner(sig, q)
                                                                did_execute = True
                                                            finally:
                                                                try:
                                                                    if old_risk is not None:
                                                                        self.sizer.risk.risk_percent = old_risk
                                                                except Exception:
                                                                    pass
                                                        if did_execute:
                                                            self._range_exec_counter['count'] += 1
                                                            # Record executed phantom mirror for Range with full features snapshot
                                                            try:
                                                                pt = self.shared.get('phantom_tracker') if hasattr(self, 'shared') else None
                                                                if pt is not None:
                                                                    feats_exec = dict(feats) if isinstance(feats, dict) else {}
                                                                    feats_exec['qscore'] = float(q)
                                                                    try:
                                                                        feats_exec['htf'] = dict(self._compute_symbol_htf_exec_metrics(sym, df))
                                                                    except Exception:
                                                                        pass
                                                                    pt.record_signal(sym, {'side': sig.side, 'entry': float(sig.entry), 'sl': float(sig.sl), 'tp': float(sig.tp)}, 0.0, True, feats_exec, 'range_fbo')
                                                            except Exception:
                                                                pass
                                                            try:
                                                                logger.info(f"[{sym}] 🧮 Range decision final: execute (reason=q>={float((settings.get('rule_mode') or {}).get('execute_q_min', 78)):.1f} & htf_neutral)")
                                                            except Exception:
                                                                pass
                                        except Exception:
                                            did_execute = False
                                    if not did_execute:
                                        # Check phantom Q-score threshold
                                        ph_min = float((settings.get('rule_mode') or {}).get('phantom_q_min', 20))
                                        if q < ph_min:
                                            logger.info(f"[{sym}] 📦 Range phantom REJECTED: Q={q:.1f} < {ph_min:.1f}")
                                            continue

                                        # Phantom accepted - log acceptance
                                        exec_thr = float((settings.get('rule_mode') or {}).get('execute_q_min', 78))
                                        logger.info(f"[{sym}] 📝 Range phantom ACCEPTED: Q={q:.1f} ≥ {ph_min:.1f} (exec_q={exec_thr:.1f})")

                                        # Verbose decision reason
                                        try:
                                            log_cfg = (settings.get('logging') or {})
                                            if bool(log_cfg.get('verbose', False)) and not bool(log_cfg.get('decision_only', False)):
                                                thr = float((settings.get('rule_mode') or {}).get('execute_q_min', 78))
                                                reason = None
                                                if q < thr:
                                                    reason = f"q<thr {q:.1f}<{thr:.1f}"
                                                else:
                                                    t15v = float(feats.get('ts15',0.0)); t60v = float(feats.get('ts60',0.0))
                                                    if max(t15v, t60v) >= 60:
                                                        reason = 'htf_strong'
                                                    elif sym in self.book.positions:
                                                        reason = 'position_exists'
                                                    elif self._range_exec_counter['count'] >= int(exec_cfg.get('daily_cap', 3)):
                                                        reason = 'daily_cap'
                                                if reason:
                                                    logger.info(f"[{sym}] 🧮 Range decision final: phantom (reason={reason})")
                                                    # Telegram notify for blocked reasons (non-q) to mirror Trend behavior
                                                    try:
                                                        if self.tg and reason != 'q<thr':
                                                            msg = f"🛑 Range: [{sym}] EXEC blocked (reason={reason}) — phantom recorded"
                                                            if regime_label:
                                                                msg += f"\nRegime: {regime_label}"
                                                            await self.tg.send_message(msg)
                                                    except Exception:
                                                        pass
                                        except Exception:
                                            pass
                                        # Record phantom (prefer shared tracker instance)
                                        try:
                                            pt = None
                                            try:
                                                pt = self.shared.get('phantom_tracker') if hasattr(self, 'shared') else None
                                            except Exception:
                                                pt = None
                                            if pt is None:
                                                from phantom_trade_tracker import get_phantom_tracker
                                                pt = get_phantom_tracker()
                                            try:
                                                feats = dict(feats) if isinstance(feats, dict) else {}
                                                feats['qscore'] = float(q)
                                            except Exception:
                                                pass
                                            pt.record_signal(sym, {'side': sig.side, 'entry': float(sig.entry), 'sl': float(sig.sl), 'tp': float(sig.tp)}, 0.0, False, feats, 'range_fbo')
                                        except Exception as e:
                                            logger.error(f"[{sym}] ❌ Range phantom recording FAILED: {e}")
                                    # Update per-symbol Range state using latest band info
                                    try:
                                        cl = float(df['close'].iloc[-1])
                                        rng_high = float(feats.get('range_high', 0.0)); rng_low = float(feats.get('range_low', 0.0))
                                        rng_w = max(1e-9, (rng_high - rng_low))
                                        near_edge = min(abs(cl - rng_low), abs(rng_high - cl)) <= (0.15 * rng_w)
                                        self._range_symbol_state[sym] = {'in_range': bool((rng_low < cl < rng_high)), 'near_edge': bool(near_edge)}
                                    except Exception:
                                        pass
                                    # Decision message
                                    try:
                                        comps = f"RNG={qc.get('rng',0):.0f} FBO={qc.get('fbo',0):.0f} PROX={qc.get('prox',0):.0f} Micro={qc.get('micro',0):.0f} Risk={qc.get('risk',0):.0f} SR={qc.get('sr',0):.0f}"
                                        if self.tg:
                                            # Regime label from comps
                                            try:
                                                t15 = float(feats.get('ts15',0.0)); t60 = float(feats.get('ts60',0.0)); rc15 = float(feats.get('rc15',0.0)); rc60 = float(feats.get('rc60',0.0))
                                                if t15 >= 60 or t60 >= 60:
                                                    reg = 'Trending'
                                                elif rc15 >= 0.6 or rc60 >= 0.6:
                                                    reg = 'Ranging'
                                                else:
                                                    reg = 'Neutral'
                                                msg = f"🟡 Range PHANTOM: [{sym}] Q={q:.1f} < {float(((settings.get('rule_mode') or {}).get('execute_q_min', 78))):.1f}\n{comps}\nRegime: {reg}"
                                            except Exception:
                                                msg = f"🟡 Range PHANTOM: [{sym}] Q={q:.1f} < {float(((settings.get('rule_mode') or {}).get('execute_q_min', 78))):.1f}\n{comps}"
                                            await self.tg.send_message(msg)
                                    except Exception:
                                        pass
                                    # Mirror to events
                                    try:
                                        evts = self.shared.get('trend_events')
                                        if isinstance(evts, list):
                                            from datetime import datetime as _dt
                                            evts.append({'ts': _dt.utcnow().isoformat()+'Z', 'symbol': sym, 'text': f"Range PHANTOM Q={q:.1f} comps: {comps}"})
                                            if len(evts) > 60:
                                                del evts[:len(evts)-60]
                                    except Exception:
                                        pass
                                except Exception:
                                    continue
                        except Exception:
                            pass
                        # Sleep a short cadence
                        try:
                            await asyncio.sleep(10)
                        except Exception:
                            break
                self._create_task(_range_fbo_scanner())
                try:
                    # Reflect current exec/phantom settings in startup log for clarity
                    settings = cfg.get('range', {}) or {}
                    exec_cfg = (settings.get('exec') or {})
                    exec_enabled = bool(exec_cfg.get('enabled', False)) and not bool(settings.get('phantom_only', True))
                    logger.info(f"📦 Range FBO scanner started (exec={'ON' if exec_enabled else 'OFF'}, phantom_only={bool(settings.get('phantom_only', True))})")
                except Exception:
                    logger.info("📦 Range FBO scanner started")
        except Exception as e:
            logger.debug(f"Range scanner start failed: {e}")
        
        # Use multi-websocket handler if >190 topics
        if len(topics) > 190:
            logger.info(f"Using multi-websocket handler for {len(topics)} topics")
            ws_handler = MultiWebSocketHandler(cfg["bybit"]["ws_public"], self)
            stream = ws_handler.multi_kline_stream(topics)
        else:
            logger.info(f"Using single websocket for {len(topics)} topics")
            stream = self.kline_stream(cfg["bybit"]["ws_public"], topics)
        # Start secondary stream for scalps if configured
        scalp_cfg = cfg.get('scalp', {})
        use_scalp = bool(scalp_cfg.get('enabled', False) and SCALP_AVAILABLE)
        scalp_stream_tf = str(scalp_cfg.get('timeframe', '3'))
        scalp_use_frames_3m = bool(scalp_cfg.get('use_frames_3m_for_detection', True))
        # Start 3m secondary stream if Scalp needs it OR if Trend/MR 3m context is requested
        use_3m_for_context = False
        try:
            use_3m_for_context = bool((cfg.get('trend', {}).get('context', {}) or {}).get('use_3m_context', False) or
                                      (cfg.get('mr', {}).get('context', {}) or {}).get('use_3m_context', False))
        except Exception:
            use_3m_for_context = False
        if (use_scalp and SCALP_AVAILABLE and scalp_stream_tf) or (use_3m_for_context and scalp_stream_tf):
            try:
                self._create_task(self._collect_secondary_stream(cfg['bybit']['ws_public'], scalp_stream_tf, symbols))
                self._scalp_secondary_started = True
                logger.info(f"🧪 Secondary 3m stream started (tf={scalp_stream_tf}m) — sources: {'Scalp' if use_scalp else ''}{' + ' if use_scalp and use_3m_for_context else ''}{'Context' if use_3m_for_context else ''}")
            except Exception as e:
                logger.warning(f"Failed to start scalp secondary stream: {e}")

        # Range execute runner wrapper (available to background scanner after this point)
        async def _range_exec_runner(sig_obj, qscore: float = 0.0):
            thr = float((((cfg.get('range', {}) or {}).get('rule_mode', {}) or {}).get('execute_q_min', 78)))
            return await _try_execute('range_fbo', sig_obj, ml_score=0.0, threshold=thr)
        try:
            self._range_exec_runner = _range_exec_runner
        except Exception:
            pass

        # Start streaming
        async for sym, k in stream:
                if not self.running:
                    break
                
                # Skip symbols not in our configured list
                if sym not in symbols:
                    continue
                
                try:
                    # Parse kline
                    ts = int(k["start"])
                    row = pd.DataFrame(
                        [[float(k["open"]), float(k["high"]), float(k["low"]), float(k["close"]), float(k["volume"])]],
                        index=[pd.to_datetime(ts, unit="ms", utc=True)],
                        columns=["open","high","low","close","volume"]
                    )
                    
                    # Capture 15m confirm flag for breakout gating
                    try:
                        main_confirm = bool(k.get('confirm', False))
                    except Exception:
                        main_confirm = False
                    try:
                        if getattr(self, '_trend_settings', None) is not None:
                            self._trend_settings.current_bar_confirmed = main_confirm
                        # Also expose in shared for diagnostics if needed
                        try:
                            self.shared['last_main_confirm'] = {sym: main_confirm}
                        except Exception:
                            pass
                    except Exception:
                        pass

                    # Get existing frame or create new if not exists
                    if sym in self.frames:
                        df = self.frames[sym]
                    else:
                        df = new_frame()
                    
                    # Ensure both dataframes have consistent timezone handling
                    if df.index.tz is None and row.index.tz is not None:
                        # Convert existing df to UTC if it's timezone-naive
                        df.index = df.index.tz_localize('UTC')
                    elif df.index.tz is not None and row.index.tz is None:
                        # Convert new row to UTC if it's timezone-naive
                        row.index = row.index.tz_localize('UTC')
                        
                    df.loc[row.index[0]] = row.iloc[0]
                    df.sort_index(inplace=True)
                    # Limit candles per symbol based on total symbols to control memory
                    max_candles_per_symbol = max(2000, 100000 // len(self.config['trade']['symbols']))
                    df = df.tail(max_candles_per_symbol)  # Dynamic limit based on symbol count
                    self.frames[sym] = df
                    
                    # Update phantom trades with current price
                    if phantom_tracker is not None:
                        current_price = df['close'].iloc[-1]
                        # Get BTC price for context
                        btc_price = None
                        if 'BTCUSDT' in self.frames and not self.frames['BTCUSDT'].empty:
                            btc_price = self.frames['BTCUSDT']['close'].iloc[-1]
                        # Update phantom prices for both systems
                        if use_enhanced_parallel and ENHANCED_ML_AVAILABLE:
                            # Update Trend phantom tracker in parallel system
                            try:
                                phantom_tracker.update_phantom_prices(
                                    sym, current_price, df=df, btc_price=btc_price, symbol_collector=symbol_collector
                                )
                            except Exception:
                                pass
                            # Update MR phantom tracker only when MR is enabled and tracker exists
                            try:
                                if (not getattr(self, '_mr_disabled', False)) and (mr_phantom_tracker is not None):
                                    mr_phantom_tracker.update_mr_phantom_prices(sym, current_price, df=df)
                            except Exception:
                                pass
                            # Scalp phantom tracker price updates (if available)
                            try:
                                scpt = get_scalp_phantom_tracker()
                                df3u = self.frames_3m.get(sym) if hasattr(self, 'frames_3m') else None
                                scpt.update_scalp_phantom_prices(sym, current_price, df=df3u if df3u is not None and not df3u.empty else df)
                            except Exception:
                                pass
                        else:
                            # Original system
                            phantom_tracker.update_phantom_prices(
                                sym, current_price, df=df, btc_price=btc_price, symbol_collector=symbol_collector
                            )
                    # Update shadow simulations with current price
                    try:
                        get_shadow_tracker().update_prices(sym, float(df['close'].iloc[-1]))
                    except Exception:
                        pass

                    # Scale-out BE move monitoring: notify TP1 reached and move SL to BE
                    try:
                        if hasattr(self, '_scaleout') and sym in getattr(self, '_scaleout', {}) and sym in self.book.positions:
                            so = self._scaleout.get(sym) or {}
                            if not so.get('be_moved') and bool(so.get('move_be', True)):
                                side = str(so.get('side',''))
                                tp1 = float(so.get('tp1', 0.0))
                                entry_px = float(so.get('entry', 0.0))
                                last_px = float(df['close'].iloc[-1])
                                hit = (side == 'long' and last_px >= tp1) or (side == 'short' and last_px <= tp1)
                                if hit:
                                    try:
                                        # First, notify TP1 reached
                                        try:
                                            if self.tg:
                                                qty1 = None
                                                try:
                                                    qty1 = so.get('qty1')
                                                except Exception:
                                                    qty1 = None
                                                msg_tp1 = f"🎯 TP1 reached: {sym} price {last_px:.4f} {'≥' if side=='long' else '≤'} TP1 {tp1:.4f}"
                                                if qty1:
                                                    msg_tp1 += f" | qty1={float(qty1):.4f}"
                                                await self.tg.send_message(msg_tp1)
                                        except Exception:
                                            pass
                                        # Move stop to break-even (entry price)
                                        self.bybit.set_tpsl(sym, stop_loss=float(entry_px))
                                        self._scaleout[sym]['be_moved'] = True
                                        if self.tg:
                                            await self.tg.send_message(f"🛡️ BE Move: {sym} SL→BE at {entry_px:.4f} after TP1 hit")
                                    except Exception:
                                        pass
                    except Exception:
                        pass

                    # Re-entry invalidation: close if price re-enters breakout zone
                    try:
                        exec_cfg = ((self.config.get('trend', {}) or {}).get('exec', {}) or {})
                        if bool(exec_cfg.get('cancel_on_reentry', True)) and sym in self.book.positions:
                            pm = getattr(self, '_position_meta', {}).get(sym, {})
                            breakout_level = float(pm.get('breakout_level', 0.0) or 0.0)
                            if breakout_level > 0:
                                inv_mode = str(exec_cfg.get('invalidation_mode', 'on_close')).lower()
                                inv_tf = str(exec_cfg.get('invalidation_timeframe', '3m')).lower()
                                pre_tp1_only = bool(exec_cfg.get('cancel_on_reentry_pre_tp1_only', True))
                                # Pre-TP1 guard using BE moved flag
                                if pre_tp1_only:
                                    so = getattr(self, '_scaleout', {}) if hasattr(self, '_scaleout') else {}
                                    if isinstance(so, dict) and sym in so and so.get('be_moved'):
                                        raise Exception('skip_reentry_post_tp1')
                                # Determine price per timeframe
                                price_ok = False
                                last_price = None
                                if inv_tf == '3m' and hasattr(self, 'frames_3m'):
                                    df3c = self.frames_3m.get(sym)
                                    if df3c is not None and len(df3c) > 0:
                                        last_price = float(df3c['close'].iloc[-1])
                                        price_ok = True
                                if not price_ok:
                                    # fallback to main timeframe
                                    dfm = self.frames.get(sym)
                                    if dfm is not None and len(dfm) > 0:
                                        last_price = float(dfm['close'].iloc[-1])
                                        price_ok = True
                                if price_ok and last_price is not None:
                                    pos = self.book.positions.get(sym)
                                    if pos:
                                        side = str(pos.side)
                                        reenter = False
                                        if side == 'long':
                                            if inv_mode == 'on_close':
                                                reenter = last_price <= breakout_level
                                            else:  # on_touch uses current last price likewise
                                                reenter = last_price <= breakout_level
                                        else:  # short
                                            if inv_mode == 'on_close':
                                                reenter = last_price >= breakout_level
                                            else:
                                                reenter = last_price >= breakout_level
                                        if reenter:
                                            # Close position and cancel orders
                                            try:
                                                self.bybit.cancel_all_orders(sym)
                                            except Exception:
                                                pass
                                            try:
                                                close_side = 'Sell' if side == 'long' else 'Buy'
                                                self.bybit.place_market(sym, close_side, float(pos.qty), reduce_only=True)
                                                logger.info(f"[{sym}] 🛑 Re-entry invalidation: closed at {last_price:.4f} (breakout {breakout_level:.4f})")
                                                if self.tg:
                                                    await self.tg.send_message(f"🛑 Re-entry invalidation: {sym} closed at {last_price:.4f} (breakout {breakout_level:.4f})")
                                            except Exception as e:
                                                logger.warning(f"[{sym}] Re-entry invalidation close failed: {e}")
                    except Exception:
                        pass

                    # Auto-save to database every 2 minutes (reduce data loss on disconnects)
                    if (datetime.now() - self.last_save_time).total_seconds() > 120:
                        await self.save_all_candles()
                        self.last_save_time = datetime.now()
                
                    # Check for closed positions periodically
                    if (datetime.now() - last_position_check).total_seconds() > position_check_interval:
                        await self.check_closed_positions(book, shared.get("meta"), ml_scorer, reset_symbol_state, symbol_collector)
                        last_position_check = datetime.now()
                
                    # Handle panic close requests
                    if sym in panic_list and sym in book.positions:
                        logger.warning(f"Executing panic close for {sym}")
                        pos = book.positions.pop(sym)
                        side = "Sell" if pos.side == "long" else "Buy"
                        try:
                            bybit.place_market(sym, side, pos.qty, reduce_only=True)
                            if self.tg:
                                await self.tg.send_message(f"✅ Panic closed {sym}")
                        except Exception as e:
                            logger.error(f"Panic close error: {e}")
                            if self.tg:
                                await self.tg.send_message(f"❌ Failed to panic close {sym}: {e}")
                        panic_list.remove(sym)
                
                    # Only act on bar close
                    if not k.get("confirm", False):
                        continue
                
                    # Increment candle counter for summary
                    candles_processed += 1
                    
                    # Track analysis time
                    last_analysis[sym] = datetime.now()
                
                    # Strategy-specific regime routing (central router) — DISABLED by default
                    # When central router is disabled, we rely solely on independent MR/Trend flows
                    router_choice = 'none'
                    tr_score = mr_score = 0.0
                    # Disable central router (legacy regime selector)
                    central_router_enabled = False
                    # Legacy regime selector disabled — no routing scores calculated

                    # Higher-timeframe gating/bias using enhanced regime (HTF)
                    htf = None
                    try:
                        htf = get_enhanced_market_regime(df, sym)
                    except Exception:
                        htf = None

                    # Thresholds and hysteresis
                    sreg = self.config.get('strategy_regimes', {}) if hasattr(self, 'config') else {}
                    pb_cfg = sreg.get('trend', {}) if isinstance(sreg, dict) else {}
                    mr_cfg = sreg.get('mr', {}) if isinstance(sreg, dict) else {}
                    tie_margin = int(sreg.get('tie_breaker_margin', 5) if isinstance(sreg, dict) else 5)
                    pb_thr = int(pb_cfg.get('threshold', 60))
                    mr_thr = int(mr_cfg.get('threshold', 70))
                    pb_hold = int(pb_cfg.get('min_hold', 3))
                    mr_hold = int(mr_cfg.get('min_hold', 3))

                    prev_state = shared.get('routing_state', {}).get(sym)
                    prev_route = prev_state.get('strategy') if isinstance(prev_state, dict) else None
                    prev_idx = prev_state.get('last_idx') if isinstance(prev_state, dict) else None

                    # Trend score already computed via score_trend_regime

                    # Eligible candidates above threshold (central router only)
                    candidates = []
                    if central_router_enabled:
                        if tr_score >= pb_thr:
                            candidates.append(('trend_pullback', tr_score))
                        if mr_score >= mr_thr:
                            candidates.append(('enhanced_mr', mr_score))

                    # Apply HTF gating to candidates when enabled (drop misaligned). Use composite RC/TS.
                    # When execute_only flags are enabled, do NOT drop here — defer to execution stage
                    if central_router_enabled:
                        try:
                            htf_cfg = (self.config.get('router', {}) or {}).get('htf_bias', {})
                            mode = str(htf_cfg.get('mode', 'gated')).lower()
                            if bool(htf_cfg.get('enabled', False)):
                                # Exec-only regime flags
                                try:
                                    tr_exec_only = bool((((self.config.get('trend', {}) or {}).get('regime', {}) or {}).get('execute_only', True)))
                                except Exception:
                                    tr_exec_only = True
                                try:
                                    mr_exec_only = bool((((self.config.get('mr', {}) or {}).get('regime', {}) or {}).get('execute_only', True)))
                                except Exception:
                                    mr_exec_only = True
                                metrics = self._get_htf_metrics(sym, df)
                                min_ts = float((htf_cfg.get('trend', {}) or {}).get('min_trend_strength', 60.0))
                                min_rq = float((htf_cfg.get('mr', {}) or {}).get('min_range_quality', 0.60))
                                max_ts = float((htf_cfg.get('mr', {}) or {}).get('max_trend_strength', 40.0))
                                gated = []
                                for name, sc in candidates:
                                    if name == 'trend_pullback':
                                        if tr_exec_only:
                                            # Defer gating to execution; keep candidate
                                            gated.append((name, sc))
                                            continue
                                        ts_ok = (metrics['ts15'] >= min_ts) and ((metrics['ts60'] == 0.0) or (metrics['ts60'] >= min_ts))
                                        if ts_ok or mode == 'soft':
                                            gated.append((name, sc))
                                        else:
                                            logger.info(f"[{sym}] 🧮 HTF gate: drop TREND ts15={metrics['ts15']:.1f} ts60={metrics['ts60']:.1f} < {min_ts:.1f}")
                                    elif name == 'enhanced_mr':
                                        if mr_exec_only:
                                            gated.append((name, sc))
                                            continue
                                        rc_ok = (metrics['rc15'] >= min_rq) and ((metrics['rc60'] == 0.0) or (metrics['rc60'] >= min_rq))
                                        ts_ok = (metrics['ts15'] <= max_ts) and ((metrics['ts60'] == 0.0) or (metrics['ts60'] <= max_ts))
                                        if (rc_ok and ts_ok) or mode == 'soft':
                                            gated.append((name, sc))
                                        else:
                                            logger.info(f"[{sym}] 🧮 HTF gate: drop MR rc15={metrics['rc15']:.2f}/rc60={metrics['rc60']:.2f} ts15={metrics['ts15']:.1f}/ts60={metrics['ts60']:.1f} (need rc≥{min_rq:.2f} & ts≤{max_ts:.1f})")
                                candidates = gated
                        except Exception:
                            pass

                    if central_router_enabled and candidates:
                        # Choose best by score
                        candidates.sort(key=lambda x: x[1], reverse=True)
                        top, top_score = candidates[0]
                        # Tie-breaker & hysteresis
                        if prev_route in ('trend_pullback', 'enhanced_mr'):
                            # Hold previous for min_hold
                            hold = pb_hold if prev_route == 'trend_pullback' else mr_hold
                            try:
                                if prev_idx is not None and (candles_processed - int(prev_idx)) < hold:
                                    router_choice = prev_route
                                elif len(candidates) > 1 and abs(candidates[0][1] - candidates[1][1]) <= tie_margin:
                                    router_choice = prev_route
                                else:
                                    router_choice = top
                            except Exception:
                                router_choice = top
                        else:
                            router_choice = top
                    else:
                        router_choice = 'none'

                    if central_router_enabled:
                        try:
                            # One-line decision record snapshot (router + signal presence + ML margins)
                            tr_ml = '—'; mr_ml = '—'; tr_marg = '—'; mr_marg = '—'
                            try:
                                if 'ml_score_tr' in locals() and 'thr_tr' in locals():
                                    tr_ml = f"{float(ml_score_tr):.0f}/{int(thr_tr)}"; tr_marg = f"{float(ml_score_tr - thr_tr):+}" 
                            except Exception:
                                pass
                            try:
                                if 'ml_score_mr' in locals() and 'thr_mr' in locals():
                                    mr_ml = f"{float(ml_score_mr):.0f}/{int(thr_mr)}"; mr_marg = f"{float(ml_score_mr - thr_mr):+}"
                            except Exception:
                                pass
                            has_tr = 'Y' if ('soft_sig_tr' in locals() and soft_sig_tr) else 'N'
                            has_mr = 'Y' if ('soft_sig_mr' in locals() and soft_sig_mr) else 'N'
                            logger.info(
                                f"[{sym}] 🧮 Decision: router={str(router_choice).upper()} TR {tr_score:.0f}/{pb_thr} MR {mr_score:.0f}/{mr_thr} | "
                                f"sig TR={has_tr} MR={has_mr} | ML TR={tr_ml}{'' if tr_marg=='—' else f'({tr_marg})'} MR={mr_ml}{'' if mr_marg=='—' else f'({mr_marg})'}"
                            )
                        except Exception:
                            pass

                    # Log summary periodically instead of every candle
                    if (datetime.now() - last_summary_log).total_seconds() > summary_log_interval:
                        logger.info(f"📊 5-min Summary: {candles_processed} candles processed, {signals_detected} signals, {len(book.positions)} positions open")
                        # Telemetry counters for ML/phantom flows
                        try:
                            tel = shared.get('telemetry', {}) if 'shared' in locals() else {}
                            mr = tel.get('phantom_wins', 0)
                            ml = tel.get('phantom_losses', 0)
                            rej = tel.get('ml_rejects', 0)
                            logger.info(f"📈 Telemetry: ML rejects={rej}, Phantom wins={mr}, Phantom losses={ml}")
                        except Exception:
                            pass
                        # Scalp health snapshot
                        if not getattr(self, '_trend_only', False):
                            try:
                                st = getattr(self, '_scalp_stats', {}) or {}
                                logger.info(f"🩳 Scalp health: confirms={st.get('confirms',0)}, signals={st.get('signals',0)}, dedup={st.get('dedup_skips',0)}, cooldown={st.get('cooldown_skips',0)}")
                                # Reset counters for next interval
                                self._scalp_stats = {'confirms': 0, 'signals': 0, 'dedup_skips': 0, 'cooldown_skips': 0}
                            except Exception:
                                pass
                        # Add ML stats to summary
                        if use_enhanced_parallel and ENHANCED_ML_AVAILABLE:
                            # Enhanced parallel system stats
                            try:
                                tr_scorer = shared.get('trend_scorer')
                                if tr_scorer:
                                    tr_info = tr_scorer.get_retrain_info()
                                    logger.info(f"🧠 Trend ML: {tr_info.get('total_records', 0)} records, {tr_info.get('trades_until_next_retrain', 'N/A')} to retrain")
                            except Exception:
                                pass
                            # Trend Promotion toggling based on recent WR
                            try:
                                tr_cfg = (self.config.get('trend', {}) or {}).get('promotion', {})
                                if tr_cfg.get('enabled', False):
                                    tp = shared.get('trend_promotion', {})
                                    # Reset daily counter on UTC day change
                                    from datetime import datetime as _dt
                                    cur_day = _dt.utcnow().strftime('%Y%m%d')
                                    if tp.get('day') != cur_day:
                                        tp['day'] = cur_day
                                        tp['count'] = 0
                                    tr_scorer = shared.get('trend_scorer')
                                    tr_stats = tr_scorer.get_stats() if tr_scorer else {}
                                    recent_wr = float(tr_stats.get('recent_win_rate', 0.0))
                                    recent_n = int(tr_stats.get('recent_trades', 0))
                                    total_exec = int(tr_stats.get('executed_count', 0))
                                    promote_wr = float(tr_cfg.get('promote_wr', 55.0))
                                    demote_wr = float(tr_cfg.get('demote_wr', 35.0))
                                    min_recent = int(tr_cfg.get('min_recent', 30))
                                    min_total = int(tr_cfg.get('min_total_trades', 100))
                                    logger.info(f"   Trend recent WR: {recent_wr:.1f}% (N={recent_n}) | active={tp.get('active')} cap_used={tp.get('count',0)}")
                                    if (not tp.get('active')) and recent_n >= min_recent and total_exec >= min_total and recent_wr >= promote_wr:
                                        tp['active'] = True
                                        logger.info(f"🚀 Trend Promotion activated (WR {recent_wr:.1f}% ≥ {promote_wr:.1f}%, N={recent_n})")
                                        if self.tg:
                                            try:
                                                await self.tg.send_message(f"🚀 Trend Promotion: Activated (WR {recent_wr:.1f}% ≥ {promote_wr:.0f}%)")
                                            except Exception:
                                                pass
                                    elif tp.get('active') and recent_wr < demote_wr:
                                        tp['active'] = False
                                        logger.info(f"🛑 Trend Promotion deactivated (WR {recent_wr:.1f}% < {demote_wr:.1f}%)")
                                        if self.tg:
                                            try:
                                                await self.tg.send_message(f"🚦 Trend Promotion: Deactivated (WR {recent_wr:.1f}% < {demote_wr:.0f}%)")
                                            except Exception:
                                                pass
                                    shared['trend_promotion'] = tp
                            except Exception:
                                pass

                            if enhanced_mr_scorer:
                                mr_stats = enhanced_mr_scorer.get_enhanced_stats()
                                logger.info(f"🧠 Enhanced MR ML: {mr_stats.get('completed_trades', 0)} trades, "
                                           f"threshold: {mr_stats.get('current_threshold', 'N/A')}, "
                                           f"{mr_stats.get('trades_until_retrain', 'N/A')} to retrain")
                                # MR Promotion toggling based on recent WR
                                try:
                                    prom_cfg = (self.config.get('mr', {}) or {}).get('promotion', {})
                                    if prom_cfg.get('enabled', False):
                                        mp = shared.get('mr_promotion', {})
                                        # Reset daily counter on UTC day change
                                        from datetime import datetime as _dt
                                        cur_day = _dt.utcnow().strftime('%Y%m%d')
                                        if mp.get('day') != cur_day:
                                            mp['day'] = cur_day
                                            mp['count'] = 0

                                        recent_wr = float(mr_stats.get('recent_win_rate', 0.0))
                                        recent_n = int(mr_stats.get('recent_trades', 0))
                                        total_exec = int(mr_stats.get('completed_trades', 0))
                                        promote_wr = float(prom_cfg.get('promote_wr', 50.0))
                                        demote_wr = float(prom_cfg.get('demote_wr', 30.0))
                                        min_recent = int(prom_cfg.get('min_recent', 20))
                                        min_total = int(prom_cfg.get('min_total_trades', 50))

                                        logger.info(f"   MR recent WR: {recent_wr:.1f}% (N={recent_n}) | active={mp.get('active')} cap_used={mp.get('count',0)}")

                                        # Hysteresis: promote at ≥ promote_wr, demote at < demote_wr
                                        if not mp.get('active') and recent_n >= min_recent and total_exec >= min_total and recent_wr >= promote_wr:
                                            mp['active'] = True
                                            logger.info(f"🚀 MR Promotion activated (WR {recent_wr:.1f}% ≥ {promote_wr:.1f}%, N={recent_n})")
                                            if self.tg:
                                                try:
                                                    await self.tg.send_message(f"🌀 MR Promotion: Activated (WR {recent_wr:.1f}% ≥ {promote_wr:.0f}%)")
                                                except Exception:
                                                    pass
                                        elif mp.get('active') and recent_wr < demote_wr:
                                            mp['active'] = False
                                            logger.info(f"🛑 MR Promotion deactivated (WR {recent_wr:.1f}% < {demote_wr:.1f}%)")
                                            if self.tg:
                                                try:
                                                    await self.tg.send_message(f"🌀 MR Promotion: Deactivated (WR {recent_wr:.1f}% < {demote_wr:.0f}%)")
                                                except Exception:
                                                    pass
                                        shared['mr_promotion'] = mp
                                except Exception:
                                    pass
                            # Scalp ML (phantom-only visibility)
                            if SCALP_AVAILABLE and get_scalp_scorer is not None:
                                try:
                                    sc_scorer = get_scalp_scorer()
                                    # Sync scorer threshold with config for accurate dashboard display
                                    try:
                                        cfg_thr = None
                                        try:
                                            cfg_thr = float((self.config.get('scalp', {}) or {}).get('threshold', None))
                                        except Exception:
                                            cfg_thr = None
                                        if cfg_thr is not None and abs(float(sc_scorer.min_score) - float(cfg_thr)) > 1e-6:
                                            sc_scorer.min_score = float(cfg_thr)
                                            try:
                                                sc_scorer._save_state()
                                            except Exception:
                                                pass
                                    except Exception:
                                        pass
                                    info = {}
                                    try:
                                        info = sc_scorer.get_retrain_info()
                                    except Exception:
                                        info = {}
                                    nxt = info.get('trades_until_next_retrain', 'N/A')
                                    logger.info(
                                        f"🩳 Scalp ML: {getattr(sc_scorer,'completed_trades',0)} samples, threshold: {sc_scorer.min_score:.0f}, "
                                        f"ready: {'yes' if sc_scorer.is_ml_ready else 'no'}, next retrain in: {nxt} trades"
                                    )
                                except Exception:
                                    pass
                        else:
                            # Original system stats
                            if ml_scorer:
                                ml_stats = ml_scorer.get_stats()
                                retrain_info = ml_scorer.get_retrain_info()
                                logger.info(f"🧠 Trend ML: {ml_stats.get('completed_trades', 0)} trades, {retrain_info.get('trades_until_next_retrain', 'N/A')} to retrain")
                            # Get mean reversion scorer from shared data for logging
                            shared_mr_scorer_log = shared.get('mean_reversion_scorer') if 'shared' in locals() else None
                            if shared_mr_scorer_log and not getattr(self, '_trend_only', False):
                                mr_ml_stats = shared_mr_scorer_log.get_stats()
                                logger.info(f"🧠 Mean Reversion ML: {mr_ml_stats.get('completed_trades', 0)} trades")
                        last_summary_log = datetime.now()
                        candles_processed = 0
                        signals_detected = 0
                    
                    # Check signal cooldown
                    now = time.time()
                    if sym in last_signal_time:
                        if now - last_signal_time[sym] < signal_cooldown:
                            # Skip silently - no need to log every cooldown
                            continue
                
                    # Strategy independence: run Trend and MR fully independently (single-per-symbol concurrency)
                    try:
                        indep_cfg = cfg.get('strategy_independence', {}) if 'cfg' in locals() else {}
                        # Default to enabled so Trend/Scalp independent flows always run
                        independence_enabled = bool(indep_cfg.get('enabled', True))
                    except Exception:
                        independence_enabled = True

                    # In trend-only mode, always run independence block to ensure Trend logging,
                    # regardless of enhanced ML availability.
                    if independence_enabled or getattr(self, '_trend_only', False):
                        # Take a regime snapshot once for per-strategy filters
                        try:
                            regime_analysis = get_enhanced_market_regime(df, sym)
                        except Exception:
                            regime_analysis = None

                        # Helper to attempt execution for a given signal; returns True if executed
                        async def _try_execute(strategy_name: str, sig_obj, ml_score: float = 0.0, threshold: float = 75.0):
                            nonlocal book, bybit, sizer
                            dbg_logger = logging.getLogger(__name__)
                            # Route Scalp executions to the dedicated stream-side executor for robust TP/SL handling
                            try:
                                if strategy_name == 'scalp':
                                    return await self._execute_scalp_trade(sym, sig_obj, ml_score=float(ml_score or 0.0))
                            except Exception:
                                # Fall through to generic path if executor is unavailable
                                pass
                            # Optional 3m context enforcement (applies to both Trend and MR execution paths)
                            try:
                                # Trend micro-context (bypass for HIGH-ML)
                                if strategy_name in ('trend_pullback','trend_breakout'):
                                    ctx_cfg = (cfg.get('trend', {}).get('context', {}) or {}) if 'cfg' in locals() else {}
                                    is_high_ml = False
                                    try:
                                        is_high_ml = isinstance(getattr(sig_obj, 'meta', {}), dict) and bool(sig_obj.meta.get('high_ml'))
                                    except Exception:
                                        is_high_ml = False
                                    if bool(ctx_cfg.get('enforce', False)) and not is_high_ml:
                                        ok3, why3 = self._micro_context_trend(sym, sig_obj.side)
                                        if not ok3:
                                            dbg_logger.debug(f"[{sym}] Trend: skip — 3m_ctx_enforce ({why3})")
                                            return False
                                    elif bool(ctx_cfg.get('enforce', False)) and is_high_ml:
                                        try:
                                            logger.info(f"[{sym}] 🧮 Trend 3m.ctx bypass: HIGH-ML execution")
                                        except Exception:
                                            pass

                                # MR micro-context (bypass for HIGH-ML)
                                if strategy_name == 'enhanced_mr':
                                    ctx_cfg = (cfg.get('mr', {}).get('context', {}) or {}) if 'cfg' in locals() else {}
                                    is_high_ml = False
                                    try:
                                        is_high_ml = isinstance(getattr(sig_obj, 'meta', {}), dict) and bool(sig_obj.meta.get('high_ml'))
                                    except Exception:
                                        is_high_ml = False
                                    if bool(ctx_cfg.get('enforce', False)) and not is_high_ml:
                                        ok3, why3 = self._micro_context_mr(sym, sig_obj.side)
                                        if not ok3:
                                            dbg_logger.debug(f"[{sym}] MR: skip — 3m_ctx_enforce ({why3})")
                                            return False
                                    elif bool(ctx_cfg.get('enforce', False)) and is_high_ml:
                                        try:
                                            logger.info(f"[{sym}] 🧮 MR 3m.ctx bypass: HIGH-ML execution")
                                        except Exception:
                                            pass

                                # Scalp micro-context (if enabled)
                                if strategy_name == 'scalp':
                                    sctx = (cfg.get('scalp', {}).get('context', {}) or {}) if 'cfg' in locals() else {}
                                    if bool(sctx.get('enforce', False)):
                                        ok3, why3 = self._micro_context_scalp(sym, sig_obj.side)
                                        if not ok3:
                                            dbg_logger.debug(f"[{sym}] Scalp: skip — 3m_ctx_enforce ({why3})")
                                            return False
                            except Exception:
                                pass
                            # Enforce one-way mode: only one position per symbol at a time
                            if sym in book.positions:
                                try:
                                    dbg_logger.debug(f"[{sym}] {strategy_name}: skip — position_exists")
                                except Exception:
                                    pass
                                # Record Trend phantom with explicit reason if applicable
                                try:
                                    if strategy_name in ('trend_pullback','trend_breakout') and 'phantom_tracker' in locals() and phantom_tracker is not None:
                                        feats_pe = {}
                                        try:
                                            feats_pe = getattr(self, '_last_signal_features', {}).get(sym, {}) if hasattr(self, '_last_signal_features') else {}
                                        except Exception:
                                            feats_pe = {}
                                        try:
                                            if isinstance(feats_pe, dict):
                                                feats_pe = feats_pe.copy(); feats_pe['diversion_reason'] = 'position_exists'
                                        except Exception:
                                            pass
                                        ph_sig = {'side': sig_obj.side, 'entry': float(sig_obj.entry), 'sl': float(sig_obj.sl), 'tp': float(sig_obj.tp)}
                                        phantom_tracker.record_signal(sym, ph_sig, float(ml_score or 0.0), False, feats_pe, 'trend_pullback')
                                except Exception:
                                    pass
                                return False
                            # Get symbol metadata and round TP/SL
                            m = meta_for(sym, shared["meta"])
                            from position_mgr import round_step
                            tick_size = m.get("tick_size", 0.000001)
                            original_tp = sig_obj.tp; original_sl = sig_obj.sl
                            sig_obj.tp = round_step(sig_obj.tp, tick_size)
                            sig_obj.sl = round_step(sig_obj.sl, tick_size)
                            if original_tp != sig_obj.tp or original_sl != sig_obj.sl:
                                logger.info(f"[{sym}] Rounded TP/SL to tick size {tick_size}. TP: {original_tp:.6f} -> {sig_obj.tp:.6f}, SL: {original_sl:.6f} -> {sig_obj.sl:.6f}")
                            # Balance and sizing
                            current_balance = bybit.get_balance()
                            if current_balance:
                                sizer.account_balance = current_balance
                                shared["last_balance"] = current_balance
                            risk_amount = sizer.account_balance * (risk.risk_percent / 100.0) if risk.use_percent_risk and sizer.account_balance else risk.risk_usd
                            qty = sizer.qty_for(sig_obj.entry, sig_obj.sl, m.get("qty_step",0.001), m.get("min_qty",0.001), ml_score=ml_score)
                            if qty <= 0:
                                logger.info(f"[{sym}] {strategy_name} qty calc invalid -> skip execution")
                                # Record as phantom if this was a valid Trend signal but balance/sizing blocked execution
                                try:
                                    if strategy_name in ('trend_pullback','trend_breakout') and 'phantom_tracker' in locals() and phantom_tracker is not None:
                                        # Use last cached trend features if available
                                        feats = {}
                                        try:
                                            feats = getattr(self, '_last_signal_features', {}).get(sym, {}) if hasattr(self, '_last_signal_features') else {}
                                        except Exception:
                                            feats = {}
                                        try:
                                            if isinstance(feats, dict):
                                                feats = feats.copy(); feats['diversion_reason'] = 'sizing_invalid'
                                        except Exception:
                                            pass
                                        ph_sig = {'side': sig_obj.side, 'entry': float(sig_obj.entry), 'sl': float(sig_obj.sl), 'tp': float(sig_obj.tp)}
                                        phantom_tracker.record_signal(sym, ph_sig, float(ml_score or 0.0), False, feats, 'trend_pullback')
                                        try:
                                            logger.info(f"[{sym}] 👻 Phantom recorded due to insufficient balance/sizing (ML {ml_score:.1f})")
                                        except Exception:
                                            pass
                                except Exception:
                                    pass
                                try:
                                    R = abs(float(sig_obj.entry) - float(sig_obj.sl))
                                    risk_val = (risk.risk_percent if getattr(risk, 'use_percent_risk', False) else getattr(risk, 'risk_usd', 0.0))
                                    dbg_logger.debug(f"[{sym}] {strategy_name}: skip — sizing_invalid (risk={risk_val}, R={R:.6f})")
                                except Exception:
                                    pass
                                return False
                            # SL sanity
                            current_price = df['close'].iloc[-1]
                            if (sig_obj.side == "long" and sig_obj.sl >= current_price) or (sig_obj.side == "short" and sig_obj.sl <= current_price):
                                logger.warning(f"[{sym}] {strategy_name} SL invalid relative to current price -> skip execution")
                                try:
                                    dbg_logger.debug(f"[{sym}] {strategy_name}: skip — sl_invalid (price={float(current_price):.6f}, sl={float(sig_obj.sl):.6f})")
                                except Exception:
                                    pass
                                return False
                            # Set leverage and place order
                            max_lev = int(m.get("max_leverage", 10))
                            bybit.set_leverage(sym, max_lev)
                            side = "Buy" if sig_obj.side == "long" else "Sell"
                            try:
                                logger.info(f"[{sym}] {strategy_name.upper()} EXECUTE {sig_obj.side.upper()} @ {float(sig_obj.entry):.4f} SL {float(sig_obj.sl):.4f} TP {float(sig_obj.tp):.4f} | ML {ml_score:.1f} thr {threshold:.1f}")
                            except Exception:
                                logger.info(f"[{sym}] {strategy_name.upper()} placing {side} order for {qty} units")
                            _ = bybit.place_market(sym, side, qty, reduce_only=False)
                            # Adjust TP to actual entry
                            actual_entry = sig_obj.entry
                            try:
                                await asyncio.sleep(0.5)
                                position = bybit.get_position(sym)
                                if position and position.get("avgPrice"):
                                    actual_entry = float(position["avgPrice"])
                                    if actual_entry != sig_obj.entry:
                                        fee_adjustment = 1.00165
                                        risk_distance = abs(actual_entry - sig_obj.sl)
                                        rr = settings.rr if strategy_name == 'enhanced_mr' else trend_settings.rr
                                        if sig_obj.side == "long":
                                            sig_obj.tp = actual_entry + (rr * risk_distance * fee_adjustment)
                                        else:
                                            sig_obj.tp = actual_entry - (rr * risk_distance * fee_adjustment)
                                        logger.info(f"[{sym}] {strategy_name.upper()} TP adjusted for actual entry: {sig_obj.tp:.4f}")
                                        # Recalc SL for slippage to preserve risk (MR & Trend)
                                        try:
                                            if strategy_name == 'trend_pullback':
                                                sl_cfg = (cfg.get('trend', {}) or {}).get('exec', {}).get('slippage_recalc', {}) if 'cfg' in locals() else {}
                                                enabled = bool(sl_cfg.get('enabled', True))
                                                min_pct = float(sl_cfg.get('min_pct', 0.001))
                                                if enabled and qty > 0:
                                                    slip_pct = abs(actual_entry - sig_obj.entry) / max(1e-9, sig_obj.entry)
                                                    if slip_pct >= min_pct:
                                                        target_dist = float(risk_amount) / float(qty)
                                                        if sig_obj.side == 'long':
                                                            new_sl = actual_entry - target_dist
                                                            pivot = None
                                                            try:
                                                                if isinstance(sig_obj.meta, dict):
                                                                    pivot = float(sig_obj.meta.get('pivot_low'))
                                                            except Exception:
                                                                pivot = None
                                                            if pivot is not None:
                                                                atr = float(sig_obj.meta.get('atr', 0.0)) if isinstance(sig_obj.meta, dict) else 0.0
                                                                buffer = 0.05 * atr
                                                                new_sl = min(new_sl, pivot - buffer)
                                                            min_stop = actual_entry * 0.01
                                                            if (actual_entry - new_sl) < min_stop:
                                                                new_sl = actual_entry - min_stop
                                                            if new_sl < actual_entry:
                                                                sig_obj.sl = new_sl
                                                        else:
                                                            new_sl = actual_entry + target_dist
                                                            pivot = None
                                                            try:
                                                                if isinstance(sig_obj.meta, dict):
                                                                    pivot = float(sig_obj.meta.get('pivot_high'))
                                                            except Exception:
                                                                pivot = None
                                                            if pivot is not None:
                                                                atr = float(sig_obj.meta.get('atr', 0.0)) if isinstance(sig_obj.meta, dict) else 0.0
                                                                buffer = 0.05 * atr
                                                                new_sl = max(new_sl, pivot + buffer)
                                                            min_stop = actual_entry * 0.01
                                                            if (new_sl - actual_entry) < min_stop:
                                                                new_sl = actual_entry + min_stop
                                                            if new_sl > actual_entry:
                                                                sig_obj.sl = new_sl
                                                        logger.info(f"[{sym}] TREND SL recalc for slippage: SL -> {sig_obj.sl:.4f}")
                                            elif strategy_name == 'enhanced_mr':
                                                sl_cfg = (cfg.get('mr', {}) or {}).get('exec', {}).get('slippage_recalc', {}) if 'cfg' in locals() else {}
                                                enabled = bool(sl_cfg.get('enabled', True))
                                                min_pct = float(sl_cfg.get('min_pct', 0.001))
                                                pivot_buf_atr = float(sl_cfg.get('pivot_buffer_atr', 0.05))
                                                if enabled and qty > 0:
                                                    slip_pct = abs(actual_entry - sig_obj.entry) / max(1e-9, sig_obj.entry)
                                                    if slip_pct >= min_pct:
                                                        target_dist = float(risk_amount) / float(qty)
                                                        # Compute ATR quickly for buffer
                                                        try:
                                                            prev = df['close'].shift()
                                                            tr = np.maximum(df['high'] - df['low'], np.maximum((df['high'] - prev).abs(), (df['low'] - prev).abs()))
                                                            atr_len = int(getattr(settings, 'atr_len', 14) or 14)
                                                            atr_val = float(tr.rolling(atr_len).mean().iloc[-1]) if len(tr) >= atr_len else float(tr.iloc[-1])
                                                        except Exception:
                                                            atr_val = 0.0
                                                        if sig_obj.side == 'long':
                                                            new_sl = actual_entry - target_dist
                                                            pivot = None
                                                            try:
                                                                if isinstance(sig_obj.meta, dict):
                                                                    pivot = float(sig_obj.meta.get('range_lower'))
                                                            except Exception:
                                                                pivot = None
                                                            if pivot is not None and atr_val > 0:
                                                                new_sl = min(new_sl, pivot - (pivot_buf_atr * atr_val))
                                                            min_stop = actual_entry * 0.01
                                                            if (actual_entry - new_sl) < min_stop:
                                                                new_sl = actual_entry - min_stop
                                                            if new_sl < actual_entry:
                                                                sig_obj.sl = new_sl
                                                        else:
                                                            new_sl = actual_entry + target_dist
                                                            pivot = None
                                                            try:
                                                                if isinstance(sig_obj.meta, dict):
                                                                    pivot = float(sig_obj.meta.get('range_upper'))
                                                            except Exception:
                                                                pivot = None
                                                            if pivot is not None and atr_val > 0:
                                                                new_sl = max(new_sl, pivot + (pivot_buf_atr * atr_val))
                                                            min_stop = actual_entry * 0.01
                                                            if (new_sl - actual_entry) < min_stop:
                                                                new_sl = actual_entry + min_stop
                                                            if new_sl > actual_entry:
                                                                sig_obj.sl = new_sl
                                                        logger.info(f"[{sym}] MR SL recalc for slippage: SL -> {sig_obj.sl:.4f}")
                                        except Exception as _slerr:
                                            logger.debug(f"[{sym}] SL recalc skipped: {_slerr}")
                                        # Optional: Recalc SL to preserve risk when slippage exceeds threshold (Trend only)
                                        try:
                                            if strategy_name == 'trend_pullback':
                                                sl_cfg = (cfg.get('trend', {}) or {}).get('exec', {}).get('slippage_recalc', {}) if 'cfg' in locals() else {}
                                                enabled = bool(sl_cfg.get('enabled', True))
                                                min_pct = float(sl_cfg.get('min_pct', 0.001))  # 0.1%
                                                if enabled and qty > 0:
                                                    slip_pct = abs(actual_entry - sig_obj.entry) / max(1e-9, sig_obj.entry)
                                                    if slip_pct >= min_pct:
                                                        target_dist = float(risk_amount) / float(qty)
                                                        if sig_obj.side == 'long':
                                                            new_sl = actual_entry - target_dist
                                                            pivot = None
                                                            try:
                                                                if isinstance(sig_obj.meta, dict):
                                                                    pivot = float(sig_obj.meta.get('pivot_low'))
                                                            except Exception:
                                                                pivot = None
                                                            if pivot is not None:
                                                                atr = float(sig_obj.meta.get('atr', 0.0)) if isinstance(sig_obj.meta, dict) else 0.0
                                                                buffer = 0.05 * atr
                                                                new_sl = min(new_sl, pivot - buffer)
                                                            min_stop = actual_entry * 0.01
                                                            if (actual_entry - new_sl) < min_stop:
                                                                new_sl = actual_entry - min_stop
                                                            if new_sl < actual_entry:
                                                                sig_obj.sl = new_sl
                                                        else:
                                                            new_sl = actual_entry + target_dist
                                                            pivot = None
                                                            try:
                                                                if isinstance(sig_obj.meta, dict):
                                                                    pivot = float(sig_obj.meta.get('pivot_high'))
                                                            except Exception:
                                                                pivot = None
                                                            if pivot is not None:
                                                                atr = float(sig_obj.meta.get('atr', 0.0)) if isinstance(sig_obj.meta, dict) else 0.0
                                                                buffer = 0.05 * atr
                                                                new_sl = max(new_sl, pivot + buffer)
                                                            min_stop = actual_entry * 0.01
                                                            if (new_sl - actual_entry) < min_stop:
                                                                new_sl = actual_entry + min_stop
                                                            if new_sl > actual_entry:
                                                                sig_obj.sl = new_sl
                                                        logger.info(f"[{sym}] TREND SL recalc for slippage: SL -> {sig_obj.sl:.4f}")
                                        except Exception as _slerr:
                                            logger.debug(f"[{sym}] SL recalc skipped: {_slerr}")
                            except Exception:
                                pass
                            # Set TP/SL (with optional Trend scale-out when enabled)
                            main_tp_applied = None
                            try:
                                sc_cfg = ((((self.config.get('trend', {}) or {}).get('exec', {}) or {}).get('scaleout', {}) or {}))
                                if strategy_name in ('trend_pullback','trend_breakout') and bool(sc_cfg.get('enabled', False)) and qty > 0:
                                    from position_mgr import round_step
                                    qty_step = float(m.get('qty_step', 0.001))
                                    frac = max(0.1, min(0.9, float(sc_cfg.get('fraction', 0.5))))
                                    tp1_r = float(sc_cfg.get('tp1_r', 1.6))
                                    tp2_r = float(sc_cfg.get('tp2_r', 3.0))
                                    R = abs(float(actual_entry) - float(sig_obj.sl))
                                    if R > 0:
                                        if sig_obj.side == 'long':
                                            tp1 = float(actual_entry) + tp1_r * R
                                            tp2 = float(actual_entry) + tp2_r * R
                                            tp_side = "Sell"
                                        else:
                                            tp1 = float(actual_entry) - tp1_r * R
                                            tp2 = float(actual_entry) - tp2_r * R
                                            tp_side = "Buy"
                                        # Set main TP to TP2
                                        try:
                                            bybit.set_tpsl(sym, take_profit=float(tp2), stop_loss=float(sig_obj.sl))
                                            main_tp_applied = float(tp2)
                                        except Exception:
                                            pass
                                        # Place reduce-only limit for partial TP1
                                        try:
                                            qty1 = round_step(float(qty) * frac, qty_step)
                                            tp1_resp = bybit.place_reduce_only_limit(sym, tp_side, float(qty1), float(tp1), post_only=True, reduce_only=True)
                                        except Exception:
                                            tp1_resp = None
                                        # Track and notify
                                        try:
                                            if not hasattr(self, '_scaleout'):
                                                self._scaleout = {}
                                            self._scaleout[sym] = {
                                                'tp1': float(tp1), 'tp2': float(tp2), 'entry': float(actual_entry),
                                                'side': sig_obj.side, 'be_moved': False, 'move_be': bool(sc_cfg.get('move_sl_to_be', True)),
                                                'qty1': float(qty1), 'qty2': max(0.0, float(qty) - float(qty1)),
                                                'tp1_order_id': (tp1_resp.get('result', {}).get('orderId') if isinstance(tp1_resp, dict) else None)
                                            }
                                            if self.tg:
                                                pct = int(round(frac*100))
                                                oid = self._scaleout[sym].get('tp1_order_id')
                                                await self.tg.send_message(
                                                    f"📊 Scale-out armed: {sym} TP1={tp1:.4f} qty1={qty1:.4f} ({pct}%) TP2={tp2:.4f} qty2={self._scaleout[sym]['qty2']:.4f}"
                                                    + (f" ordId={oid}" if oid else "") + " | SL→BE after TP1"
                                                )
                                        except Exception:
                                            pass
                                else:
                                    # Single TP/SL
                                    bybit.set_tpsl(sym, take_profit=sig_obj.tp, stop_loss=sig_obj.sl, qty=qty)
                                    main_tp_applied = float(sig_obj.tp)
                            except Exception:
                                # Fallback: ensure at least SL is set
                                try:
                                    bybit.set_tpsl(sym, stop_loss=float(sig_obj.sl))
                                except Exception:
                                    pass
                            # Read back TP/SL from exchange to reflect any server-side rounding
                            try:
                                pos_tp_sl = bybit.get_position(sym)
                                if isinstance(pos_tp_sl, dict):
                                    tp_val = pos_tp_sl.get('takeProfit')
                                    sl_val = pos_tp_sl.get('stopLoss')
                                    if tp_val not in (None, '', '0'):
                                        sig_obj.tp = float(tp_val)
                                    if sl_val not in (None, '', '0'):
                                        sig_obj.sl = float(sl_val)
                            except Exception:
                                pass
                            # Update book
                            # Attach qscore from last signal features if present
                            try:
                                qv = float(((getattr(self, '_last_signal_features', {}) or {}).get(sym, {}) or {}).get('qscore', 0.0) or 0.0)
                            except Exception:
                                qv = 0.0
                            book.positions[sym] = Position(
                                sig_obj.side,
                                qty,
                                entry=actual_entry,
                                sl=sig_obj.sl,
                                tp=(main_tp_applied if main_tp_applied is not None else sig_obj.tp),
                                entry_time=datetime.now(),
                                strategy_name=strategy_name,
                                ml_score=float(ml_score or 0.0),
                                qscore=qv
                            )
                            # Optionally cancel active phantoms on execution (config)
                            try:
                                ph_cfg = cfg.get('phantom', {}) if 'cfg' in locals() else {}
                                if bool(ph_cfg.get('cancel_on_execute', False)):
                                    if strategy_name == 'enhanced_mr' and mr_phantom_tracker:
                                        mr_phantom_tracker.cancel_active(sym)
                                    elif strategy_name in ('trend_pullback','trend_breakout') and phantom_tracker:
                                        phantom_tracker.cancel_active(sym)
                            except Exception:
                                pass
                            # Notify only for high-ML executions (Trend/MR)
                            try:
                                # Suppress dup if stream-side already executed
                                if sym in getattr(self, '_stream_executed', {}):
                                    raise Exception('skip_high_ml_notify_stream_executed')
                                is_high_ml = bool(isinstance(getattr(sig_obj, 'meta', {}), dict) and sig_obj.meta.get('high_ml'))
                                if self.tg and is_high_ml:
                                    emoji = '📈'
                                    strategy_label = 'Enhanced Mr' if strategy_name=='enhanced_mr' else ('Range Fbo' if strategy_name=='range_fbo' else 'Trend Pullback')
                                    # Include scale-out info if available
                                    so = getattr(self, '_scaleout', {}).get(sym) if hasattr(self, '_scaleout') else None
                                    if so and 'tp1' in so and 'tp2' in so:
                                        msg = (
                                            f"🔥 HIGH-ML EXECUTE: {emoji} *{sym} {sig_obj.side.upper()}* ({strategy_label})\n\n"
                                            f"Entry: {actual_entry:.4f}\n"
                                            f"Stop Loss: {sig_obj.sl:.4f}\n"
                                            f"TP1: {so['tp1']:.4f} qty1={so.get('qty1',0):.4f}"
                                            + (f" ordId={so.get('tp1_order_id')}" if so.get('tp1_order_id') else "") + "\n"
                                            f"TP2: {so['tp2']:.4f} qty2={so.get('qty2',0):.4f}\n"
                                            f"Quantity: {qty}"
                                        )
                                    else:
                                        msg = (
                                            f"🔥 HIGH-ML EXECUTE: {emoji} *{sym} {sig_obj.side.upper()}* ({strategy_label})\n\n"
                                            f"Entry: {actual_entry:.4f}\n"
                                            f"Stop Loss: {sig_obj.sl:.4f}\n"
                                            f"Take Profit: {sig_obj.tp:.4f}\n"
                                            f"Quantity: {qty}"
                                        )
                                    await self.tg.send_message(msg)
                            except Exception:
                                pass
                            # Decision final (avoid duplicate logs for high-ML — specific log already printed)
                            try:
                                is_high_ml = bool(isinstance(getattr(sig_obj, 'meta', {}), dict) and sig_obj.meta.get('high_ml'))
                                if not is_high_ml:
                                    logger.info(f"[{sym}] 🧮 Decision final: exec_{'mr' if strategy_name=='enhanced_mr' else 'trend'} (reason=ok)")
                                # Mark Trend executed in state snapshot
                                try:
                                    if strategy_name in ('trend_pullback','trend_breakout'):
                                        from strategy_pullback import mark_executed
                                        mark_executed(sym)
                                except Exception:
                                    pass
                            except Exception:
                                pass
                            return True

                        # (Reverted) Stream-side Scalp promotion execution queue removed; main loop handles promotion only

                        # 1) Mean Reversion independent (disabled in trend-only mode)
                        if not getattr(self, '_trend_only', False):
                            try:
                                logger.debug(f"[{sym}] MR: analysis start")
                                sig_mr_ind = detect_signal_mean_reversion(df.copy(), settings, sym)
                            except Exception:
                                sig_mr_ind = None
                            if sig_mr_ind is None:
                                logger.debug(f"[{sym}] MR: skip — no_signal")
                        else:
                            sig_mr_ind = None
                        if sig_mr_ind is not None:
                            ef = sig_mr_ind.meta.get('mr_features', {}) if sig_mr_ind.meta else {}
                            # Enrich MR features minimally for EV gating
                            try:
                                if 'session' not in ef:
                                    hr = datetime.utcnow().hour
                                    ef['session'] = 'asian' if 0 <= hr < 8 else ('european' if hr < 16 else 'us')
                            except Exception:
                                pass
                            try:
                                if 'volatility_regime' not in ef:
                                    vol_reg = 'normal'
                                    try:
                                        mrk = get_enhanced_market_regime(df, sym)
                                        vol_reg = str(getattr(mrk, 'volatility', 'normal') or 'normal')
                                    except Exception:
                                        pass
                                    ef['volatility_regime'] = vol_reg
                            except Exception:
                                pass
                            ml_score_mr = 0.0; thr_mr = 75.0; mr_should = True
                            try:
                                if enhanced_mr_scorer:
                                    ml_score_mr, _ = enhanced_mr_scorer.score_signal(sig_mr_ind.__dict__, ef, df)
                                    thr_mr = getattr(enhanced_mr_scorer, 'min_score', 75)
                                    # EV-aware base threshold bump for MR
                                    try:
                                        ev_thr_mr = float(enhanced_mr_scorer.get_ev_threshold(ef))
                                        thr_mr = max(thr_mr, ev_thr_mr)
                                    except Exception:
                                        pass
                                    mr_should = ml_score_mr >= thr_mr
                                try:
                                    logger.debug(f"[{sym}] MR: ml_score={float(ml_score_mr or 0.0):.1f} thr={thr_mr:.0f} should={mr_should}")
                                except Exception:
                                    pass
                                # Extreme ML override (force execution bypassing router/regime/micro gates)
                                try:
                                    mr_hi_force = float((((cfg.get('mr', {}) or {}).get('exec', {}) or {}).get('high_ml_force', 90.0)))
                                except Exception:
                                    mr_hi_force = 90.0
                                # EV-based threshold bump for MR
                                try:
                                    if enhanced_mr_scorer:
                                        ev_thr_mr = float(enhanced_mr_scorer.get_ev_threshold(ef))
                                        mr_hi_force = max(mr_hi_force, ev_thr_mr)
                                except Exception:
                                    pass
                                if ml_score_mr >= mr_hi_force:
                                    try:
                                        if not sig_mr_ind.meta:
                                            sig_mr_ind.meta = {}
                                    except Exception:
                                        sig_mr_ind.meta = {}
                                    # Mark as high-ML for telemetry/notifications and bypass gates
                                    sig_mr_ind.meta['high_ml'] = True
                                    # Allow/disallow MR execution via config (default: disabled)
                                    try:
                                        allow_mr_exec = bool((((cfg.get('mr', {}) or {}).get('exec', {}) or {}).get('enabled', False)))
                                    except Exception:
                                        allow_mr_exec = False
                                    executed = False
                                    if allow_mr_exec:
                                        # Regime gate execution same as phantom
                                        try:
                                            ok_mr_reg, why_mr_reg = self._phantom_mr_regime_ok(sym, df, regime_analysis)
                                        except Exception:
                                            ok_mr_reg, why_mr_reg = True, 'n/a'
                                        if not ok_mr_reg:
                                            logger.info(f"[{sym}] 🛑 MR execution blocked by regime gate ({why_mr_reg})")
                                        else:
                                            try:
                                                # Store MR features for executed outcome logging
                                                self._last_signal_features[sym] = dict(ef)
                                            except Exception:
                                                pass
                                            executed = await _try_execute('enhanced_mr', sig_mr_ind, ml_score=ml_score_mr, threshold=thr_mr)
                                    if executed:
                                        try:
                                            logger.info(f"[{sym}] 🧮 Decision final: exec_mr (reason=ml_extreme {ml_score_mr:.1f}>={mr_hi_force:.0f})")
                                        except Exception:
                                            pass
                                        continue
                                    else:
                                        try:
                                            if not allow_mr_exec:
                                                if mr_phantom_tracker:
                                                    mr_phantom_tracker.record_mr_signal(sym, sig_mr_ind.__dict__, float(ml_score_mr or 0.0), False, {}, ef)
                                                logger.info(f"[{sym}] 🧮 Decision final: phantom_mr (reason=exec_disabled)")
                                                continue
                                            else:
                                                logger.info(f"[{sym}] 🛑 MR High-ML override blocked: reason=exec_guard")
                                        except Exception:
                                            pass
                                # Promotion override regardless of earlier guards
                                prom_cfg = (self.config.get('mr', {}) or {}).get('promotion', {})
                                promote_enabled = bool(prom_cfg.get('enabled', False))
                                promote_wr = float(prom_cfg.get('promote_wr', 50.0))
                                mr_stats = enhanced_mr_scorer.get_enhanced_stats() if enhanced_mr_scorer else {}
                                recent_wr = float(mr_stats.get('recent_win_rate', 0.0))
                                if promote_enabled and recent_wr >= promote_wr:
                                    # Force execute immediately — bypass HTF, regime, micro gates. Only hard exec guards apply.
                                    try:
                                        if not sig_mr_ind.meta:
                                            sig_mr_ind.meta = {}
                                    except Exception:
                                        sig_mr_ind.meta = {}
                                    sig_mr_ind.meta['promotion_forced'] = True
                                    try:
                                        if self.tg:
                                            await self.tg.send_message(f"🌀 MR Promotion: Force executing {sym} {sig_mr_ind.side.upper()} (WR ≥ {promote_wr:.0f}%)")
                                    except Exception:
                                        pass
                                    try:
                                        self._last_signal_features[sym] = dict(ef)
                                    except Exception:
                                        pass
                                    executed = await _try_execute('enhanced_mr', sig_mr_ind, ml_score=ml_score_mr, threshold=thr_mr)
                                    if executed:
                                        try:
                                            logger.info(f"[{sym}] 🧮 Decision final: exec_mr (reason=promotion)")
                                        except Exception:
                                            pass
                                        # Skip further MR gates/phantom for this symbol this loop
                                        continue
                                    else:
                                        try:
                                            logger.info(f"[{sym}] 🛑 MR Promotion blocked: reason=exec_guard")
                                        except Exception:
                                            pass
                                        # Fall through to normal handling if exchange/risk guard blocked
                            except Exception:
                                pass
                            # HTF gating for MR (gated/soft), composite + persistence + promotion bypass
                            try:
                                htf_cfg = (cfg.get('router', {}) or {}).get('htf_bias', {})
                                if bool(htf_cfg.get('enabled', False)):
                                    mode = str(htf_cfg.get('mode', 'gated')).lower()
                                    comp = (htf_cfg.get('composite', {}) or {})
                                    metrics = self._get_htf_metrics(sym, df)
                                    min_rq = float((htf_cfg.get('mr', {}) or {}).get('min_range_quality', 0.60))
                                    max_ts = float((htf_cfg.get('mr', {}) or {}).get('max_trend_strength', 40.0))
                                    rc_ok = (metrics['rc15'] >= min_rq) and ((metrics['rc60'] == 0.0) or (metrics['rc60'] >= min_rq))
                                    ts_ok = (metrics['ts15'] <= max_ts) and ((metrics['ts60'] == 0.0) or (metrics['ts60'] <= max_ts))
                                    allowed = rc_ok and ts_ok
                                    mild = False
                                    # Soft tolerance window
                                    try:
                                        tol = (comp.get('soft_tolerance', {}) or {}).get('mr', {})
                                        rq_tol = float(tol.get('range_quality', 0.05))
                                        ts_tol = float(tol.get('trend_strength', 5.0))
                                        mild = ((metrics['rc15'] >= (min_rq - rq_tol)) and (metrics['ts15'] <= (max_ts + ts_tol)))
                                    except Exception:
                                        mild = False
                                    # Promotion bypass tolerance
                                    try:
                                        pbx = (comp.get('promotion_bypass', {}) or {}).get('mr', {})
                                        pbx_en = bool(pbx.get('enabled', True))
                                        rq_bt = float(pbx.get('range_quality_tol', 0.05))
                                        ts_bt = float(pbx.get('trend_strength_tol', 5.0))
                                    except Exception:
                                        pbx_en = True; rq_bt = 0.05; ts_bt = 5.0
                                    promo_bypass_ok = pbx_en and (metrics['rc15'] >= (min_rq - rq_bt)) and (metrics['ts15'] <= (max_ts + ts_bt))

                                    # New: Unconditional bypass of MR HTF gate when promotion is active
                                    # or when ML is in the high-ML band. This ensures promotions and
                                    # high-ML signals reach execution even if HTF metrics disagree.
                                    try:
                                        is_promo_active = bool((shared.get('mr_promotion', {}) or {}).get('active'))
                                    except Exception:
                                        is_promo_active = False
                                    try:
                                        hi_force = float((((cfg.get('mr', {}) or {}).get('exec', {}) or {}).get('high_ml_force', 90.0)))
                                    except Exception:
                                        hi_force = 90.0
                                    if not allowed and (is_promo_active or float(ml_score_mr or 0.0) >= hi_force):
                                        allowed = True
                                        try:
                                            reason = 'promotion' if is_promo_active else f"ml_extreme {float(ml_score_mr or 0.0):.1f}>={hi_force:.0f}"
                                            logger.info(f"[{sym}] 🧮 HTF gate bypass (MR): allow due to {reason}")
                                        except Exception:
                                            pass
                                    # Persistence hold before flipping to allow
                                    try:
                                        min_hold = int((comp.get('min_hold_bars', 0)) or 0)
                                        if min_hold > 0:
                                            state = self._htf_hold.setdefault('mr', {}).setdefault(sym, {'last_allow': False, 'last_change_at': None})
                                            now_idx = df.index[-1]
                                            prev_allow = bool(state.get('last_allow', False))
                                            last_change = state.get('last_change_at')
                                            if allowed and not prev_allow:
                                                # Require hold window
                                                if last_change is None or (now_idx - last_change).total_seconds() < (min_hold * int(cfg['trade']['timeframe']) * 60):
                                                    allowed = False
                                                    logger.info(f"[{sym}] 🧮 HTF hold: delaying MR allow (hold={min_hold} bars)")
                                                else:
                                                    state['last_allow'] = True
                                            elif not allowed and prev_allow:
                                                state['last_allow'] = False
                                                state['last_change_at'] = now_idx
                                            elif (last_change is None):
                                                state['last_change_at'] = now_idx
                                    except Exception:
                                        pass
                                    if not allowed:
                                        if mode == 'gated' and not (mr_should and promo_bypass_ok):
                                            logger.info(f"[{sym}] 🧮 Decision final: phantom_mr (reason=htf_gate rc15={metrics['rc15']:.2f}/rc60={metrics['rc60']:.2f} ts15={metrics['ts15']:.1f}/ts60={metrics['ts60']:.1f})")
                                            if mr_phantom_tracker:
                                                mr_phantom_tracker.record_mr_signal(sym, sig_mr_ind.__dict__, float(ml_score_mr or 0.0), False, {}, ef)
                                            sig_mr_ind = None
                                        elif mode == 'soft' and mild:
                                            # Apply soft penalty to ML threshold
                                            try:
                                                pen = int((comp.get('soft_penalty', {}) or {}).get('mr', 5))
                                                thr_mr = float(thr_mr) + float(pen)
                                                logger.info(f"[{sym}] 🧮 HTF soft: MR thr +{pen} → {thr_mr:.0f} (mild misalignment)")
                                            except Exception:
                                                pass
                            except Exception:
                                pass
                            # MR regime filter: require ranging regime unless promotion_bypass is active
                            try:
                                mr_reg = (cfg.get('mr', {}) or {}).get('regime', {})
                                mr_reg_enabled = bool(mr_reg.get('enabled', True))
                                prim = getattr(regime_analysis, 'primary_regime', 'unknown') if regime_analysis else 'unknown'
                                conf = float(getattr(regime_analysis, 'regime_confidence', 1.0)) if regime_analysis else 1.0
                                persist = float(getattr(regime_analysis, 'regime_persistence', 1.0)) if regime_analysis else 1.0
                                mr_conf_req = float(mr_reg.get('min_conf', 0.60))
                                mr_persist_req = float(mr_reg.get('min_persist', 0.0))
                                promotion_bypass = bool(mr_reg.get('promotion_bypass', True))
                                mr_pass_regime = (not mr_reg_enabled) or ((prim == 'ranging') and (conf >= mr_conf_req) and (persist >= mr_persist_req))
                                # Regime gate disabled for MR phantom flow; allow ML to decide execution (high-ML only)
                                if False and (not mr_pass_regime and not (promotion_bypass and recent_wr >= promote_wr)):
                                    logger.debug(f"[{sym}] MR: skip — regime gate (prim={prim}, conf={conf:.2f}, persist={persist:.2f})")
                                    # Do NOT record phantom when regime gate fails (high-quality phantom policy)
                                    sig_mr_ind = None
                            except Exception:
                                pass
                            if sig_mr_ind is not None and float(ml_score_mr or 0.0) < float((((cfg.get('mr', {}) or {}).get('exec', {}) or {}).get('high_ml_force', 90.0))):
                                # Below high-ML execution threshold: record phantom with explicit reason
                                if mr_phantom_tracker:
                                    ok_ph_mr, why_mr = self._phantom_mr_regime_ok(sym, df, regime_analysis)
                                    if ok_ph_mr:
                                        mr_phantom_tracker.record_mr_signal(sym, sig_mr_ind.__dict__, float(ml_score_mr or 0.0), False, {}, ef)
                                    else:
                                        try:
                                            logger.info(f"[{sym}] 🛑 MR phantom dropped by regime gate ({why_mr})")
                                        except Exception:
                                            pass
                                    try:
                                        try:
                                            hi_force_dbg = float((((cfg.get('mr', {}) or {}).get('exec', {}) or {}).get('high_ml_force', 90.0)))
                                        except Exception:
                                            hi_force_dbg = 90.0
                                        logger.info(f"[{sym}] 🧮 Decision final: phantom_mr (reason=ml<thr {float(ml_score_mr or 0.0):.1f}<{hi_force_dbg:.0f})")
                                    except Exception:
                                        pass
                                continue
                            else:
                            # Optional 3m context (enforce if configured)
                                try:
                                    mctx = (cfg.get('mr', {}).get('context', {}) or {})
                                    if bool(mctx.get('use_3m_context', False)):
                                        ok3, why3 = self._micro_context_mr(sym, sig_mr_ind.side)
                                        logger.debug(f"[{sym}] MR 3m.ctx: {'ok' if ok3 else 'weak'} ({why3})")
                                        enforce3 = bool(((cfg.get('router', {}) or {}).get('htf_bias', {}).get('micro_context', {}) or {}).get('mr_enforce', False) or mctx.get('enforce', False))
                                        if enforce3 and not ok3:
                                            logger.info(f"[{sym}] 🧮 Decision final: phantom_mr (reason=micro_ctx {why3})")
                                            if mr_phantom_tracker:
                                                mr_phantom_tracker.record_mr_signal(sym, sig_mr_ind.__dict__, float(ml_score_mr or 0.0), False, {}, ef)
                                            sig_mr_ind = None
                                            try:
                                                if self.flow_controller and self.flow_controller.enabled:
                                                    self.flow_controller.increment_accepted('mr', 1)
                                            except Exception:
                                                pass
                                            continue
                                except Exception:
                                    pass
                                # Phantom record when below threshold
                                if mr_phantom_tracker and sig_mr_ind is not None:
                                    # ML below threshold
                                    try:
                                        logger.debug(f"[{sym}] MR: skip — ml<{thr_mr:.0f} (score {float(ml_score_mr or 0.0):.1f})")
                                    except Exception:
                                        pass
                                    mr_phantom_tracker.record_mr_signal(sym, sig_mr_ind.__dict__, float(ml_score_mr or 0.0), False, {}, ef)
                                    try:
                                        logger.info(f"[{sym}] 🧮 Decision final: phantom_mr (reason=ml<thr)")
                                    except Exception:
                                        pass
                                    try:
                                        if self.flow_controller and self.flow_controller.enabled:
                                            self.flow_controller.increment_accepted('mr', 1)
                                    except Exception:
                                        pass

                        # 2) Trend independent
                        try:
                            # Verbose heartbeat for Trend analysis
                            try:
                                verbose_tr = bool((((cfg.get('trend', {}) or {}).get('logging', {}) or {}).get('verbose', False)))
                            except Exception:
                                verbose_tr = False
                            if getattr(self, '_trend_only', False):
                                verbose_tr = True
                            if verbose_tr:
                                logger.info(f"[{sym}] 🔵 Trend: analysis start")
                            else:
                                logger.debug(f"[{sym}] Trend: analysis start")
                            # Ensure we have enough 15m history for detection
                            try:
                                if len(df) < getattr(self, '_trend_hist_min', 80):
                                    if not self._trend_hist_warned.get(sym, False):
                                        logger.info(f"[{sym}] 🔵 Trend waiting for history: have {len(df)}/{getattr(self, '_trend_hist_min', 80)} 15m bars")
                                        try:
                                            logger.info(f"[{sym}] 🧮 Trend heartbeat: decision=no_signal reasons=insufficient_history")
                                        except Exception:
                                            pass
                                        self._trend_hist_warned[sym] = True
                                    raise Exception('insufficient_history')
                            except Exception as _e:
                                if str(_e) != 'insufficient_history':
                                    logger.debug(f"[{sym}] Trend history check: {_e}")
                                raise
                            sig_tr_ind = detect_trend_signal(df.copy(), trend_settings, sym)
                        except Exception:
                            sig_tr_ind = None
                        if sig_tr_ind is None:
                            logger.debug(f"[{sym}] Trend: skip — no_signal")
                            # Optional INFO-level log for visibility when no signal is found
                            try:
                                log_no_sig = bool((((cfg.get('trend', {}) or {}).get('logging', {}) or {}).get('no_signal_info', False)))
                            except Exception:
                                log_no_sig = False
                            if getattr(self, '_trend_only', False):
                                log_no_sig = True
                            if log_no_sig:
                                logger.info(f"[{sym}] ❌ No Trend Signal: Pullback conditions not met")
                                try:
                                    # Always emit a heartbeat decision line for transparency
                                    logger.info(f"[{sym}] 🧮 Trend heartbeat: decision=no_signal reasons=pullback_unmet")
                                except Exception:
                                    pass
                        if sig_tr_ind is not None:
                            # Build trend features (regime-independent)
                            try:
                                cl = df['close']; price = float(cl.iloc[-1])
                                ys = cl.tail(20).values if len(cl) >= 20 else cl.values
                                try:
                                    slope = np.polyfit(np.arange(len(ys)), ys, 1)[0]
                                except Exception:
                                    slope = 0.0
                                trend_slope_pct = float((slope / price) * 100.0) if price else 0.0
                                ema20 = cl.ewm(span=20, adjust=False).mean().iloc[-1]
                                ema50 = cl.ewm(span=50, adjust=False).mean().iloc[-1] if len(cl) >= 50 else ema20
                                ema_stack_score = 100.0 if (price > ema20 > ema50 or price < ema20 < ema50) else 50.0 if (ema20 != ema50) else 0.0
                                rng_today = float(df['high'].iloc[-1] - df['low'].iloc[-1])
                                med_range = float((df['high'] - df['low']).rolling(20).median().iloc[-1]) if len(df) >= 20 else rng_today
                                range_expansion = float(rng_today / max(1e-9, med_range))
                                prev = cl.shift(); trarr = np.maximum(df['high'] - df['low'], np.maximum((df['high'] - prev).abs(), (df['low'] - prev).abs()))
                                atr = float(trarr.rolling(14).mean().iloc[-1]) if len(trarr) >= 14 else float(trarr.iloc[-1])
                                atr_pct = float((atr / max(1e-9, price)) * 100.0) if price else 0.0
                                close_vs_ema20_pct = float(((price - ema20) / max(1e-9, ema20)) * 100.0) if ema20 else 0.0
                                # Bollinger width pct (20)
                                try:
                                    ma20 = cl.rolling(20).mean()
                                    sd20 = cl.rolling(20).std()
                                    bb_upper = ma20 + 2*sd20
                                    bb_lower = ma20 - 2*sd20
                                    bb_width = float((bb_upper.iloc[-1] - bb_lower.iloc[-1])) if len(bb_upper) > 0 else 0.0
                                    bb_width_pct = float(bb_width / max(1e-9, float(price))) if price else 0.0
                                except Exception:
                                    bb_width_pct = 0.0
                                # Volatility regime and session
                                vol_reg = 'normal'
                                try:
                                    mrk = get_enhanced_market_regime(df, sym)
                                    vol_reg = str(getattr(mrk, 'volatility_level', 'normal') or 'normal')
                                except Exception:
                                    pass
                                try:
                                    hr = datetime.utcnow().hour
                                    sess = 'asian' if 0 <= hr < 8 else ('european' if hr < 16 else 'us')
                                except Exception:
                                    sess = 'us'
                                # Symbol cluster
                                sym_cluster = 3
                                try:
                                    from symbol_clustering import load_symbol_clusters
                                    clusters = load_symbol_clusters()
                                    sym_cluster = int(clusters.get(sym, 3))
                                except Exception:
                                    sym_cluster = 3
                                # Trend Pullback features
                                break_dist = float(getattr(sig_tr_ind, 'meta', {}).get('break_dist_atr', 0.0) if getattr(sig_tr_ind, 'meta', None) else 0.0)
                                retrace_depth = float(getattr(sig_tr_ind, 'meta', {}).get('retrace_depth_atr', 0.0) if getattr(sig_tr_ind, 'meta', None) else 0.0)
                                confirms = int(getattr(sig_tr_ind, 'meta', {}).get('confirm_candles', 0) if getattr(sig_tr_ind, 'meta', None) else 0)
                                trend_features = {
                                    'atr_pct': atr_pct,
                                    'trend_slope_pct': trend_slope_pct,
                                    'break_dist_atr': break_dist,
                                    'retrace_depth_atr': retrace_depth,
                                    'confirm_candles': confirms,
                                    'ema_stack_score': ema_stack_score,
                                    'range_expansion': range_expansion,
                                    'close_vs_ema20_pct': close_vs_ema20_pct,
                                    'bb_width_pct': bb_width_pct,
                                    'session': sess,
                                    'symbol_cluster': sym_cluster,
                                    'volatility_regime': vol_reg,
                                    # Divergence features (if provided by strategy)
                                    'div_ok': bool(getattr(sig_tr_ind, 'meta', {}).get('div_ok', False) if getattr(sig_tr_ind, 'meta', None) else False),
                                    'div_type': str(getattr(sig_tr_ind, 'meta', {}).get('div_type', 'NONE') if getattr(sig_tr_ind, 'meta', None) else 'NONE'),
                                    'div_score': float(getattr(sig_tr_ind, 'meta', {}).get('div_score', 0.0) if getattr(sig_tr_ind, 'meta', None) else 0.0),
                                    'div_rsi_delta': float(getattr(sig_tr_ind, 'meta', {}).get('div_rsi_delta', 0.0) if getattr(sig_tr_ind, 'meta', None) else 0.0),
                                    'div_tsi_delta': float(getattr(sig_tr_ind, 'meta', {}).get('div_tsi_delta', 0.0) if getattr(sig_tr_ind, 'meta', None) else 0.0),
                                }
                                # Include last counter-pivot if provided (HL/LH)
                                try:
                                    cp = None
                                    if isinstance(getattr(sig_tr_ind, 'meta', None), dict):
                                        cp = sig_tr_ind.meta.get('lh_pivot') if sig_tr_ind.side == 'long' else sig_tr_ind.meta.get('hl_pivot')
                                    if isinstance(cp, (int, float)):
                                        trend_features['counter_pivot'] = float(cp)
                                except Exception:
                                    pass
                                # Add RR target for executed/phantom quality learning
                                try:
                                    e = float(getattr(sig_tr_ind, 'entry', 0.0) or 0.0)
                                    sl_v = float(getattr(sig_tr_ind, 'sl', 0.0) or 0.0)
                                    tp_v = float(getattr(sig_tr_ind, 'tp', 0.0) or 0.0)
                                    R = abs(e - sl_v)
                                    trend_features['rr_target'] = float(abs(tp_v - e) / R) if R > 0 else 0.0
                                except Exception:
                                    trend_features['rr_target'] = 0.0
                                # Attach full HTF snapshots and composite (flattened)
                                try:
                                    htfm = self._compute_symbol_htf_exec_metrics(sym, df)
                                    trend_features['htf'] = dict(htfm)
                                except Exception:
                                    pass
                                try:
                                    comp = self._get_htf_metrics(sym, df)
                                    trend_features['htf_comp'] = dict(comp)
                                    trend_features['ts15'] = float(comp.get('ts15', 0.0))
                                    trend_features['ts60'] = float(comp.get('ts60', 0.0))
                                    trend_features['rc15'] = float(comp.get('rc15', 0.0))
                                    trend_features['rc60'] = float(comp.get('rc60', 0.0))
                                except Exception:
                                    pass
                                # If rule-mode, attach Qscore details
                                try:
                                    rule_mode = (cfg.get('trend', {}) or {}).get('rule_mode', {}) if 'cfg' in locals() else {}
                                    if bool(rule_mode.get('enabled', False)):
                                        q, qc, qr = self._compute_qscore(sym, soft_sig_tr.side, df, self.frames_3m.get(sym) if hasattr(self, 'frames_3m') else None)
                                        trend_features['qscore'] = float(q)
                                        trend_features['qscore_components'] = dict(qc)
                                        trend_features['qscore_reasons'] = list(qr)
                                except Exception:
                                    pass
                                # Ensure Qscore is present for training even when rule-mode disabled
                                try:
                                    if 'qscore' not in trend_features:
                                        q, qc, qr = self._compute_qscore(sym, sig_tr_ind.side, df, self.frames_3m.get(sym) if hasattr(self, 'frames_3m') else None)
                                        trend_features['qscore'] = float(q)
                                        trend_features['qscore_components'] = dict(qc)
                                        trend_features['qscore_reasons'] = list(qr)
                                except Exception:
                                    pass
                                # Log Trend Pullback signal meta and EV threshold snapshot
                                try:
                                    ev_thr_log = None
                                    _ts = get_trend_scorer() if 'get_trend_scorer' in globals() and get_trend_scorer is not None else None
                                    if _ts is not None:
                                        ev_thr_log = float(_ts.get_ev_threshold(trend_features))
                                except Exception:
                                    ev_thr_log = None
                                try:
                                    logger.info(
                                        f"[{sym}] Trend Pullback signal: {sig_tr_ind.side.upper()} "
                                        f"entry={float(sig_tr_ind.entry):.4f} sl={float(sig_tr_ind.sl):.4f} tp={float(sig_tr_ind.tp):.4f} | "
                                        f"break={getattr(sig_tr_ind,'meta',{}).get('break_level','n/a')} confirm={int(confirms)} "
                                        f"break_d_atr={float(break_dist):.2f} retrace_atr={float(retrace_depth):.2f} "
                                        f"ev_thr={ev_thr_log if ev_thr_log is not None else 'n/a'}"
                                    )
                                except Exception:
                                    pass
                            except Exception:
                                trend_features = {}
                            tr_should = True; ml_score_tr = 0.0; thr_tr = 70.0
                            try:
                                tr_scorer = get_trend_scorer() if 'get_trend_scorer' in globals() and get_trend_scorer is not None else None
                                if tr_scorer is not None:
                                    ml_score_tr, _ = tr_scorer.score_signal(sig_tr_ind.__dict__, trend_features)
                                    thr_tr = getattr(tr_scorer, 'min_score', 70)
                                    # Apply execution ML floor from config
                                    try:
                                        exec_floor = float(((cfg.get('trend', {}) or {}).get('exec', {}) or {}).get('min_ml', 0))
                                        if thr_tr < exec_floor:
                                            thr_tr = exec_floor
                                    except Exception:
                                        pass
                                    # EV-aware base threshold bump for Trend Pullback
                                    try:
                                        ev_thr_tr = float(tr_scorer.get_ev_threshold(trend_features))
                                        thr_tr = max(thr_tr, ev_thr_tr)
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                            # Rule-mode / Qscore-only: execute based on Qscore threshold when enabled
                            rule_mode = (cfg.get('trend', {}) or {}).get('rule_mode', {}) if 'cfg' in locals() else {}
                            rm_enabled = bool(rule_mode.get('enabled', False))
                            qscore = None; qcomp = {}; qreasons = []
                            if rm_enabled:
                                try:
                                    qscore, qcomp, qreasons = self._compute_qscore(sym, sig_tr_ind.side, df, self.frames_3m.get(sym) if hasattr(self, 'frames_3m') else None)
                                except Exception:
                                    qscore, qcomp, qreasons = 50.0, {}, []
                                # Attach to features for ML training
                                try:
                                    trend_features['qscore'] = float(qscore)
                                except Exception:
                                    pass
                                exec_min = float(rule_mode.get('execute_q_min', 78))
                                ph_min = float(rule_mode.get('phantom_q_min', 65))
                                # Safety: extreme volatility block
                                try:
                                    extreme_block = bool(rule_mode.get('safety', {}).get('extreme_vol_block', True)) and str(getattr(regime_analysis, 'volatility_level', 'normal')) == 'extreme'
                                except Exception:
                                    extreme_block = False
                                if extreme_block:
                                    try:
                                        # Tag diversion reason for phantom record downstream
                                        if isinstance(trend_features, dict):
                                            trend_features['diversion_reason'] = 'extreme_vol'
                                    except Exception:
                                        pass
                                    tr_should = False
                                    try:
                                        if self.tg:
                                            await self.tg.send_message(f"🛑 Trend: [{sym}] Extreme volatility — rule-mode blocked; phantom recorded")
                                    except Exception:
                                        pass
                                elif qscore >= exec_min:
                                    tr_should = True
                                elif qscore >= ph_min:
                                    tr_should = False
                                else:
                                    tr_should = False
                                # ML influence when mature: tie-break near threshold
                                try:
                                    if bool(rule_mode.get('ml_influence', {}).get('enabled', True)) and tr_scorer is not None:
                                        info = tr_scorer.get_retrain_info()
                                        total_recs = int(info.get('total_records', 0))
                                        exec_cnt = int(info.get('executed_count', 0))
                                        min_recs = int(rule_mode.get('ml_influence', {}).get('min_records', 2000))
                                        min_exec = int(rule_mode.get('ml_influence', {}).get('min_executed', 400))
                                        margin = float(rule_mode.get('ml_influence', {}).get('margin_points', 3))
                                        if total_recs >= min_recs and exec_cnt >= min_exec:
                                            # If near threshold and ML strong, allow exec
                                            if (exec_min - margin) <= float(qscore or 0.0) < exec_min and ml_score_tr >= thr_tr:
                                                tr_should = True
                                                try:
                                                    logger.info(f"[{sym}] Rule-mode ML tie-break: q={qscore:.1f}~{exec_min:.1f}, ml={ml_score_tr:.1f}≥{thr_tr:.1f}")
                                                except Exception:
                                                    pass
                                except Exception:
                                    pass
                            else:
                                tr_should = (ml_score_tr >= thr_tr)

                            # Qscore-only override: if enabled (default true), ignore ML gating and execute purely on Qscore
                            try:
                                q_only = bool((((cfg.get('trend', {}) or {}).get('exec', {}) or {}).get('qscore_only', True)))
                            except Exception:
                                q_only = True
                            if q_only:
                                try:
                                    # Ensure qscore computed
                                    if qscore is None:
                                        qscore, qcomp, qreasons = self._compute_qscore(sym, sig_tr_ind.side, df, self.frames_3m.get(sym) if hasattr(self, 'frames_3m') else None)
                                except Exception:
                                    qscore = 50.0
                                exec_min = float(rule_mode.get('execute_q_min', 78.0)) if isinstance(rule_mode, dict) else 78.0
                                # Use learned threshold if available
                                try:
                                    ctx = {'session': self._session_label(), 'volatility_regime': getattr(htf, 'volatility_level', 'global') if 'htf' in locals() and htf else 'global'}
                                    if hasattr(self, '_qadapt_trend') and self._qadapt_trend:
                                        exec_min = float(self._qadapt_trend.get_threshold(ctx, floor=60.0, ceiling=95.0, default=exec_min))
                                except Exception:
                                    pass
                                if float(qscore or 0.0) >= exec_min:
                                    try:
                                        logger.info(f"[{sym}] 🧮 Trend decision final: execute (Qscore {float(qscore):.1f} ≥ {exec_min:.1f})")
                                    except Exception:
                                        pass
                                    # Execute immediately via stream executor
                                    try:
                                        tf_copy = dict(trend_features) if isinstance(trend_features, dict) else {}
                                        tf_copy['qscore'] = float(qscore or 0.0)
                                        self._last_signal_features[sym] = tf_copy
                                    except Exception:
                                        pass
                                    executed = await _try_execute('trend_pullback', sig_tr_ind, ml_score=0.0, threshold=exec_min)
                                    if executed:
                                        # Telegram execution notify
                                        try:
                                            if self.tg:
                                                await self.tg.send_message(
                                                    f"🟢 Trend EXECUTE: {sym} {sig_tr_ind.side.upper()} Q={float(qscore or 0.0):.1f} (≥ {exec_min:.1f})\n"
                                                    f"Entry: {float(sig_tr_ind.entry):.4f} SL: {float(sig_tr_ind.sl):.4f} TP: {float(sig_tr_ind.tp):.4f}"
                                                )
                                        except Exception:
                                            pass
                                        # Record executed phantom mirror for WR by Qscore
                                        try:
                                            pt = self.shared.get('phantom_tracker') if hasattr(self, 'shared') else None
                                            if pt is not None:
                                                sigd = {'side': sig_tr_ind.side, 'entry': float(sig_tr_ind.entry), 'sl': float(sig_tr_ind.sl), 'tp': float(sig_tr_ind.tp)}
                                                pt.record_signal(sym, sigd, 0.0, True, tf_copy, 'trend_pullback')
                                        except Exception:
                                            pass
                                        continue
                                else:
                                    # Below threshold → record phantom only when allowed
                                    try:
                                        if phantom_tracker and sig_tr_ind is not None:
                                            tf_copy = dict(trend_features) if isinstance(trend_features, dict) else {}
                                            tf_copy['qscore'] = float(qscore or 0.0)
                                            phantom_tracker.record_signal(
                                                sym,
                                                {'side': sig_tr_ind.side, 'entry': sig_tr_ind.entry, 'sl': sig_tr_ind.sl, 'tp': sig_tr_ind.tp},
                                                0.0,
                                                False,
                                                tf_copy,
                                                'trend_pullback'
                                            )
                                            logger.info(f"[{sym}] 🧮 Trend decision final: phantom (reason=qscore<thr {float(qscore or 0.0):.1f}<{exec_min:.1f})")
                                    except Exception:
                                        pass
                                    continue
                            try:
                                logger.info(f"[{sym}] Trend Pullback gating: ml={float(ml_score_tr):.1f} thr={float(thr_tr):.1f} should={tr_should}")
                            except Exception:
                                pass
                            # Extreme ML override (legacy high-ML only used when rule_mode disabled)
                            try:
                                tr_hi_force = float((((cfg.get('trend', {}) or {}).get('exec', {}) or {}).get('high_ml_force', 92.0)))
                            except Exception:
                                tr_hi_force = 92.0
                            # EV-based threshold bump for Trend high-ML override
                            try:
                                if tr_scorer is not None:
                                    ev_thr_tr = float(tr_scorer.get_ev_threshold(trend_features))
                                    tr_hi_force = max(tr_hi_force, ev_thr_tr)
                            except Exception:
                                pass
                            if (not rm_enabled) and (ml_score_tr >= tr_hi_force):
                                try:
                                    logger.info(f"[{sym}] Trend Pullback HIGH-ML override: ml={float(ml_score_tr):.1f} ≥ {float(tr_hi_force):.1f}")
                                except Exception:
                                    pass
                                try:
                                    if not sig_tr_ind.meta:
                                        sig_tr_ind.meta = {}
                                except Exception:
                                    sig_tr_ind.meta = {}
                                # High-ML execution marker only
                                sig_tr_ind.meta['high_ml'] = True
                                # Allow/disallow Trend execution via config (default: disabled)
                                try:
                                    allow_tr_exec = bool((((cfg.get('trend', {}) or {}).get('exec', {}) or {}).get('enabled', True)))
                                except Exception:
                                    allow_tr_exec = True
                                executed = False
                                if allow_tr_exec:
                                    # Exec-only HTF gate (per-symbol) — independent of router.htf_bias
                                    try:
                                        ok_gate, thr_adj, mode, _m = self._apply_htf_exec_gate(sym, df, sig_tr_ind.side, thr_tr)
                                    except Exception:
                                        ok_gate, thr_adj, mode = True, thr_tr, 'error'
                                    # Persist HTF gate decision into trend state
                                    try:
                                        from strategy_pullback import update_htf_gate
                                        meta_gate = {'mode': mode}
                                        if isinstance(_m, dict):
                                            for k in ('ts1h','ts4h','ema_dir_1h','ema_dir_4h','adx_1h','struct_dir_1h','struct_dir_4h'):
                                                if k in _m:
                                                    meta_gate[k] = _m[k]
                                        update_htf_gate(sym, bool(ok_gate), meta_gate)
                                    except Exception:
                                        pass
                                    if not ok_gate and mode == 'gated' and not rm_enabled:
                                        try:
                                            logger.info(f"[{sym}] 🧮 Trend decision final: phantom (blocked) (reason=htf_gate)")
                                        except Exception:
                                            pass
                                        # Record phantom so ML learns gate failures too
                                        try:
                                            if phantom_tracker and sig_tr_ind is not None:
                                                # Attach minimal gate metrics in features under 'htf_gate'
                                                gf = dict(trend_features) if 'trend_features' in locals() and isinstance(trend_features, dict) else {}
                                                gf['htf_gate'] = {'mode': mode}
                                                phantom_tracker.record_signal(
                                                    sym,
                                                    {'side': sig_tr_ind.side, 'entry': sig_tr_ind.entry, 'sl': sig_tr_ind.sl, 'tp': sig_tr_ind.tp},
                                                    float(ml_score_tr or 0.0),
                                                    False,
                                                    gf,
                                                    'trend_pullback'
                                                )
                                        except Exception:
                                            pass
                                        # Reset state to NEUTRAL since not executed
                                        try:
                                            from strategy_pullback import revert_to_neutral
                                            revert_to_neutral(sym)
                                        except Exception:
                                            pass
                                        # Notify
                                        try:
                                            if self.tg:
                                                await self.tg.send_message(f"🛑 Trend: [{sym}] HTF gate blocked — routed to phantom")
                                        except Exception:
                                            pass
                                        continue
                                    elif not ok_gate and mode == 'soft' and not rm_enabled:
                                        try:
                                            thr_tr = float(thr_adj)
                                            logger.info(f"[{sym}] 🔵 Trend decision context: HTF soft mode → new_thr={thr_tr:.0f} high_ml={ml_score_tr:.1f}/{tr_hi_force:.0f}")
                                        except Exception:
                                            pass
                                    else:
                                        try:
                                            logger.info(f"[{sym}] 🧮 Trend decision final: execute (reason=high_ml {ml_score_tr:.1f}≥{tr_hi_force:.0f} & htf_ok)")
                                        except Exception:
                                            pass
                                    try:
                                        self._last_signal_features[sym] = dict(trend_features)
                                    except Exception:
                                        pass
                                    executed = await _try_execute('trend_pullback', sig_tr_ind, ml_score=ml_score_tr, threshold=thr_tr)
                                if executed:
                                    try:
                                        logger.info(f"[{sym}] 🧮 Decision final: exec_trend (reason=ml_extreme {ml_score_tr:.1f}>={tr_hi_force:.0f})")
                                    except Exception:
                                        pass
                                    # Telegram execution notify
                                    try:
                                        if self.tg:
                                            await self.tg.send_message(
                                                f"🟢 Trend EXECUTE: {sym} {sig_tr_ind.side.upper()} (HIGH-ML)\n"
                                                f"Entry: {float(sig_tr_ind.entry):.4f} SL: {float(sig_tr_ind.sl):.4f} TP: {float(sig_tr_ind.tp):.4f}"
                                            )
                                    except Exception:
                                        pass
                                    # Record executed phantom mirror for WR by Qscore/ML
                                    try:
                                        pt = self.shared.get('phantom_tracker') if hasattr(self, 'shared') else None
                                        if pt is not None:
                                            tf_copy2 = dict(trend_features) if isinstance(trend_features, dict) else {}
                                            if 'qscore' not in tf_copy2:
                                                try:
                                                    q_tmp, _, _ = self._compute_qscore(sym, sig_tr_ind.side, df, self.frames_3m.get(sym) if hasattr(self, 'frames_3m') else None)
                                                    tf_copy2['qscore'] = float(q_tmp)
                                                except Exception:
                                                    pass
                                            pt.record_signal(sym, {'side': sig_tr_ind.side, 'entry': float(sig_tr_ind.entry), 'sl': float(sig_tr_ind.sl), 'tp': float(sig_tr_ind.tp)}, float(ml_score_tr or 0.0), True, tf_copy2, 'trend_pullback')
                                    except Exception:
                                        pass
                                    continue
                                else:
                                    try:
                                        if not allow_tr_exec:
                                            # Only accept phantom if regime passes
                                            ok_reg, why_reg = self._phantom_trend_regime_ok(sym, df, regime_analysis)
                                            if phantom_tracker and sig_tr_ind is not None and ok_reg:
                                                _rec = phantom_tracker.record_signal(sym, {'side': sig_tr_ind.side, 'entry': sig_tr_ind.entry, 'sl': sig_tr_ind.sl, 'tp': sig_tr_ind.tp}, float(ml_score_tr or 0.0), False, trend_features, 'trend_pullback')
                                                if _rec is not None:
                                                    try:
                                                        log_flag = bool((((cfg.get('trend', {}) or {}).get('logging', {}) or {}).get('phantom_info', False)))
                                                    except Exception:
                                                        log_flag = False
                                                    (logger.info if log_flag else logger.debug)(f"[{sym}] 🧮 Decision final: phantom_trend (reason=exec_disabled)")
                                            else:
                                                logger.info(f"[{sym}] 🛑 Trend phantom dropped by regime gate ({why_reg})")
                                            continue
                                        else:
                                            logger.info(f"[{sym}] 🛑 Trend High-ML override blocked: reason=exec_guard")
                                    except Exception:
                                        pass
                                try:
                                    logger.debug(f"[{sym}] Trend: ml_score={float(ml_score_tr or 0.0):.1f} thr={thr_tr:.0f} should={tr_should}")
                                except Exception:
                                    pass
                            # Optional 3m context (diagnostic only)
                            try:
                                if bool(((cfg.get('trend', {}).get('context', {}) or {}).get('use_3m_context', False))):
                                    ok3, why3 = self._micro_context_trend(sym, sig_tr_ind.side)
                                    logger.debug(f"[{sym}] Trend 3m.ctx: {'ok' if ok3 else 'weak'} ({why3})")
                            except Exception:
                                pass
                            # HTF gating for Trend (gated/soft), composite + persistence + promotion bypass
                            try:
                                htf_cfg = (cfg.get('router', {}) or {}).get('htf_bias', {})
                                if bool(htf_cfg.get('enabled', False)):
                                    mode = str(htf_cfg.get('mode', 'gated')).lower()
                                    comp = (htf_cfg.get('composite', {}) or {})
                                    metrics = self._get_htf_metrics(sym, df)
                                    min_ts = float((htf_cfg.get('trend', {}) or {}).get('min_trend_strength', 60.0))
                                    ts_ok = (metrics['ts15'] >= min_ts) and ((metrics['ts60'] == 0.0) or (metrics['ts60'] >= min_ts))
                                    allowed = ts_ok
                                    mild = False
                                    # Soft tolerance window
                                    try:
                                        tol = (comp.get('soft_tolerance', {}) or {}).get('trend', {})
                                        ts_tol = float(tol.get('trend_strength', 5.0))
                                        mild = (metrics['ts15'] >= (min_ts - ts_tol))
                                    except Exception:
                                        mild = False
                                    # Promotion bypass tolerance
                                    try:
                                        pbx = (comp.get('promotion_bypass', {}) or {}).get('trend', {})
                                        pbx_en = bool(pbx.get('enabled', True))
                                        ts_bt = float(pbx.get('trend_strength_tol', 5.0))
                                    except Exception:
                                        pbx_en = True; ts_bt = 5.0
                                    # Is trend promotion active? (corking)
                                    tr_promo_active = False
                                    try:
                                        tp = shared.get('trend_promotion', {}) if 'shared' in locals() else {}
                                        tr_promo_active = bool(tp.get('active'))
                                    except Exception:
                                        tr_promo_active = False
                                    promo_bypass_ok = pbx_en and tr_promo_active and (metrics['ts15'] >= (min_ts - ts_bt))
                                    # Persistence hold before flipping to allow
                                    try:
                                        min_hold = int((comp.get('min_hold_bars', 0)) or 0)
                                        if min_hold > 0:
                                            state = self._htf_hold.setdefault('trend', {}).setdefault(sym, {'last_allow': False, 'last_change_at': None})
                                            now_idx = df.index[-1]
                                            prev_allow = bool(state.get('last_allow', False))
                                            last_change = state.get('last_change_at')
                                            if allowed and not prev_allow:
                                                if last_change is None or (now_idx - last_change).total_seconds() < (min_hold * int(cfg['trade']['timeframe']) * 60):
                                                    allowed = False
                                                    logger.info(f"[{sym}] 🧮 HTF hold: delaying TREND allow (hold={min_hold} bars)")
                                                else:
                                                    state['last_allow'] = True
                                            elif not allowed and prev_allow:
                                                state['last_allow'] = False
                                                state['last_change_at'] = now_idx
                                            elif (last_change is None):
                                                state['last_change_at'] = now_idx
                                    except Exception:
                                        pass
                                    if not allowed:
                                        if mode == 'gated' and not promo_bypass_ok:
                                            try:
                                                log_flag = bool((((cfg.get('trend', {}) or {}).get('logging', {}) or {}).get('phantom_info', False)))
                                            except Exception:
                                                log_flag = False
                                            (logger.info if log_flag else logger.debug)(f"[{sym}] 🧮 Decision final: phantom_trend (reason=htf_gate ts15={metrics['ts15']:.1f} ts60={metrics['ts60']:.1f} < {min_ts:.1f})")
                                            if phantom_tracker:
                                                phantom_tracker.record_signal(sym, {'side': sig_tr_ind.side, 'entry': sig_tr_ind.entry, 'sl': sig_tr_ind.sl, 'tp': sig_tr_ind.tp}, float(ml_score_tr or 0.0), False, trend_features, 'trend_pullback')
                                            sig_tr_ind = None
                                            continue
                                        elif mode == 'soft' and mild:
                                            # Apply soft penalty to Trend ML threshold
                                            try:
                                                pen = int((comp.get('soft_penalty', {}) or {}).get('trend', 5))
                                                thr_tr = float(thr_tr) + float(pen) if 'thr_tr' in locals() else float(pen)
                                                logger.info(f"[{sym}] 🧮 HTF soft: TR thr +{pen} → {thr_tr:.0f} (mild misalignment)")
                                            except Exception:
                                                pass
                            except Exception:
                                pass
                            # Optional 3m context enforce for Trend
                            try:
                                tctx = (cfg.get('trend', {}).get('context', {}) or {})
                                if bool(tctx.get('use_3m_context', False)):
                                    ok3, why3 = self._micro_context_trend(sym, sig_tr_ind.side)
                                    logger.debug(f"[{sym}] Trend 3m.ctx: {'ok' if ok3 else 'weak'} ({why3})")
                                    enforce3 = bool(((cfg.get('router', {}) or {}).get('htf_bias', {}).get('micro_context', {}) or {}).get('trend_enforce', False) or tctx.get('enforce', False))
                                    if enforce3 and not ok3:
                                        try:
                                            log_flag = bool((((cfg.get('trend', {}) or {}).get('logging', {}) or {}).get('phantom_info', False)))
                                        except Exception:
                                            log_flag = False
                                        (logger.info if log_flag else logger.debug)(f"[{sym}] 🧮 Decision final: phantom_trend (reason=micro_ctx {why3})")
                                        if phantom_tracker:
                                            phantom_tracker.record_signal(sym, {'side': sig_tr_ind.side, 'entry': sig_tr_ind.entry, 'sl': sig_tr_ind.sl, 'tp': sig_tr_ind.tp}, float(ml_score_tr or 0.0), False, trend_features, 'trend_pullback')
                                        sig_tr_ind = None
                                        continue
                            except Exception:
                                pass
                            # Trend regime filter and exec bootstrapping (phantom-only until ready)
                            try:
                                tr_reg = (cfg.get('trend', {}) or {}).get('regime', {})
                                tr_exec = (cfg.get('trend', {}) or {}).get('exec', {})
                                tr_reg_enabled = bool(tr_reg.get('enabled', True))
                                prim = getattr(regime_analysis, 'primary_regime', 'unknown') if regime_analysis else 'unknown'
                                conf = float(getattr(regime_analysis, 'regime_confidence', 1.0)) if regime_analysis else 1.0
                                vol = getattr(regime_analysis, 'volatility_level', 'normal') if regime_analysis else 'normal'
                                tr_conf_req = float(tr_reg.get('min_conf', 0.60))
                                tr_allowed_vol = set(tr_reg.get('allowed_vol', ['low','normal']))
                                tr_pass_regime = (not tr_reg_enabled) or ((prim == 'trending') and (conf >= tr_conf_req) and (vol in tr_allowed_vol))
                                # Exec readiness gates
                                trend_exec_enabled = False
                                if tr_exec.get('bootstrap_phantom_only', True):
                                    if tr_scorer is not None and ((not tr_exec.get('require_ready', True)) or getattr(tr_scorer, 'is_ml_ready', False)):
                                        info = tr_scorer.get_retrain_info() if hasattr(tr_scorer, 'get_retrain_info') else {}
                                        total = int(info.get('total_records', 0))
                                        if total >= int(tr_exec.get('min_train_records', 300)):
                                            min_wr = float(tr_exec.get('min_recent_wr', 0.0))
                                            trend_exec_enabled = (min_wr <= 0.0) or (((tr_scorer.get_stats() or {}).get('recent_win_rate', 0.0)) >= min_wr)
                                else:
                                    trend_exec_enabled = True
                            except Exception:
                                tr_pass_regime = True; trend_exec_enabled = False

                            if False and ((not tr_pass_regime) or (not trend_exec_enabled)):
                                # Try Trend promotion (corking) override if active
                                try:
                                    tr_cfg = (self.config.get('trend', {}) or {}).get('promotion', {})
                                    tp = shared.get('trend_promotion', {})
                                    cap = int(tr_cfg.get('daily_exec_cap', 20))
                                    allow_tr = True
                                    if bool(tr_cfg.get('block_extreme_vol', True)):
                                        allow_tr = getattr(regime_analysis, 'volatility_level', 'normal') != 'extreme'
                                    if tp.get('active') and int(tp.get('count', 0)) < cap and allow_tr and sym not in book.positions:
                                        # Respect allow_cork_override for bootstrap phantom mode
                                        try:
                                            exec_cfg = (cfg.get('trend', {}) or {}).get('exec', {})
                                            if bool(exec_cfg.get('bootstrap_phantom_only', False)) and not bool(exec_cfg.get('allow_cork_override', True)):
                                                raise Exception('Trend bootstrap phantom mode without cork override')
                                        except Exception:
                                            pass
                                        executed = await _try_execute('trend_pullback', sig_tr_ind, ml_score=ml_score_tr or 0.0, threshold=thr_tr if 'thr_tr' in locals() else 70)
                                        if executed:
                                            try:
                                                if not sig_tr_ind.meta:
                                                    sig_tr_ind.meta = {}
                                                sig_tr_ind.meta['promotion_forced'] = True
                                                tp['count'] = int(tp.get('count', 0)) + 1
                                                shared['trend_promotion'] = tp
                                                if self.tg:
                                                    await self.tg.send_message(f"🚀 Trend Promotion: Force executing {sym} {sig_tr_ind.side.upper()} (cap {tp['count']}/{cap})")
                                            except Exception:
                                                pass
                                            # skip phantom record since executed
                                            continue
                                except Exception:
                                    pass
                                # Fall back to phantom record only if regime is OK
                                if phantom_tracker and sig_tr_ind is not None:
                                    # Log regime/exec gate block
                                    try:
                                        logger.debug(f"[{sym}] Trend: skip — regime/exec gate (reg={tr_pass_regime}, exec={trend_exec_enabled})")
                                    except Exception:
                                        pass
                                    if tr_pass_regime:
                                        try:
                                            if isinstance(trend_features, dict):
                                                trend_features = trend_features.copy(); trend_features['diversion_reason'] = 'exec_gate'
                                        except Exception:
                                            pass
                                        _rec = phantom_tracker.record_signal(sym, {'side': sig_tr_ind.side, 'entry': sig_tr_ind.entry, 'sl': sig_tr_ind.sl, 'tp': sig_tr_ind.tp}, float(ml_score_tr or 0.0), False, trend_features, 'trend_pullback')
                                        if _rec is not None:
                                            try:
                                                try:
                                                    log_flag = bool((((cfg.get('trend', {}) or {}).get('logging', {}) or {}).get('phantom_info', False)))
                                                except Exception:
                                                    log_flag = False
                                                (logger.info if log_flag else logger.debug)(f"[{sym}] 🧮 Decision final: phantom_trend (reason=exec_gate)")
                                            except Exception:
                                                pass
                                            try:
                                                if self.flow_controller and self.flow_controller.enabled:
                                                    self.flow_controller.increment_accepted('trend', 1)
                                            except Exception:
                                                pass
                                    else:
                                        logger.info(f"[{sym}] 🛑 Trend phantom dropped by regime gate (regime/exec_gate)")
                            if tr_should:
                                # Disable normal Trend execution – only high-ML executes
                                if phantom_tracker and sig_tr_ind is not None:
                                    ok_reg, why_reg = self._phantom_trend_regime_ok(sym, df, regime_analysis)
                                    if ok_reg:
                                        try:
                                            if isinstance(trend_features, dict):
                                                trend_features = trend_features.copy(); trend_features['diversion_reason'] = 'ml_below_high_ml'
                                        except Exception:
                                            pass
                                        _rec = phantom_tracker.record_signal(sym, {'side': sig_tr_ind.side, 'entry': sig_tr_ind.entry, 'sl': sig_tr_ind.sl, 'tp': sig_tr_ind.tp}, float(ml_score_tr or 0.0), False, trend_features, 'trend_pullback')
                                        if _rec is not None:
                                            try:
                                                try:
                                                    log_flag = bool((((cfg.get('trend', {}) or {}).get('logging', {}) or {}).get('phantom_info', False)))
                                                except Exception:
                                                    log_flag = False
                                                (logger.info if log_flag else logger.debug)(f"[{sym}] 🧮 Decision final: phantom_trend (reason=ml_below_high_ml)")
                                            except Exception:
                                                pass
                                    else:
                                        logger.info(f"[{sym}] 🛑 Trend phantom dropped by regime gate ({why_reg})")
                                continue
                            # Trend normal ML below high-ML: record phantom only if regime passes
                            if phantom_tracker:
                                try:
                                    logger.debug(f"[{sym}] Trend: below exec threshold (score {float(ml_score_tr or 0.0):.1f}) — consider phantom")
                                except Exception:
                                    pass
                                ok_reg, why_reg = self._phantom_trend_regime_ok(sym, df, regime_analysis)
                                if ok_reg:
                                    _rec = phantom_tracker.record_signal(
                                        sym,
                                        {'side': sig_tr_ind.side, 'entry': sig_tr_ind.entry, 'sl': sig_tr_ind.sl, 'tp': sig_tr_ind.tp},
                                        float(ml_score_tr or 0.0),
                                        False,
                                        trend_features,
                                        'trend_pullback'
                                    )
                                    if _rec is not None:
                                        try:
                                            try:
                                                log_flag = bool((((cfg.get('trend', {}) or {}).get('logging', {}) or {}).get('phantom_info', False)))
                                            except Exception:
                                                log_flag = False
                                            (logger.info if log_flag else logger.debug)(f"[{sym}] 🧮 Decision final: phantom_trend (reason=ml<thr)")
                                        except Exception:
                                            pass
                                        try:
                                            if self.flow_controller and self.flow_controller.enabled:
                                                self.flow_controller.increment_accepted('trend', 1)
                                        except Exception:
                                            pass
                                else:
                                    logger.info(f"[{sym}] 🛑 Trend phantom dropped by regime gate ({why_reg})")

                        # 3) Scalp independent (promotion-only execution)
                        try:
                            s_cfg = cfg.get('scalp', {}) if 'cfg' in locals() else {}
                            promote_en = bool(s_cfg.get('promote_enabled', False))
                            if promote_en and SCALP_AVAILABLE and detect_scalp_signal is not None:
                                # Promotion readiness from phantom stats (supports recent WR)
                                try:
                                    scpt = get_scalp_phantom_tracker()
                                    st = scpt.get_scalp_phantom_stats() or {}
                                    samples = int(st.get('total', 0))
                                    overall_wr = float(st.get('wr', 0.0))
                                except Exception:
                                    samples = 0; overall_wr = 0.0
                                # Compute metric
                                metric = str(s_cfg.get('promote_metric', 'recent')).lower()
                                window = int(s_cfg.get('promote_window', 50))
                                wr = overall_wr
                                if metric == 'recent':
                                    try:
                                        recents = []
                                        for trades in getattr(scpt, 'completed', {}).values():
                                            for p in trades:
                                                if getattr(p, 'outcome', None) in ('win','loss'):
                                                    recents.append(p)
                                        recents.sort(key=lambda x: getattr(x, 'exit_time', None) or getattr(x, 'signal_time', None))
                                        recents = recents[-window:]
                                        if recents:
                                            rw = sum(1 for p in recents if getattr(p, 'outcome', None) == 'win')
                                            wr = (rw / len(recents)) * 100.0
                                    except Exception:
                                        wr = overall_wr
                                # Only WR matters for Scalp promotion
                                min_wr = float(s_cfg.get('promote_min_wr', 50.0))
                                cap = int(s_cfg.get('daily_exec_cap', 20))
                                sp = shared.get('scalp_promotion', {}) if 'shared' in locals() else {}
                                if not isinstance(sp, dict):
                                    sp = {}
                                # Daily cap check
                                used = int(sp.get('count', 0))
                                ready = (wr >= min_wr)
                                # Try detect scalp signal on 3m frames; fallback to main tf
                                sc_sig = None
                                try:
                                    df3 = self.frames_3m.get(sym)
                                    src_df = df3 if (df3 is not None and len(df3) >= 120) else df
                                    sc_sig = detect_scalp_signal(src_df.copy(), ScalpSettings(), sym)
                                except Exception:
                                    sc_sig = None
                                # If no signal, allow reuse of latest 3m detection within recent_secs
                                if sc_sig is None and ready:
                                    try:
                                        recent_secs = int(s_cfg.get('promote_recent_secs', 10))
                                        last = self._scalp_last_signal.get(sym)
                                        if last and (pd.Timestamp.utcnow() - last.get('ts', pd.Timestamp.utcnow())).total_seconds() <= recent_secs:
                                            class _Tmp:
                                                side = last.get('side'); entry = last.get('entry'); sl = last.get('sl'); tp = last.get('tp'); meta = {}
                                            sc_sig = _Tmp()
                                        else:
                                            logger.info(f"[{sym}] Scalp Promotion: ready but no valid signal this bar")
                                    except Exception:
                                        pass
                                if ready and used < cap and sc_sig and sym not in book.positions:
                                    try:
                                        if self.tg:
                                            try:
                                                await self.tg.send_message(f"🩳 Scalp Promotion: Force executing {sym} {sc_sig.side.upper()} (WR ≥ {min_wr:.0f}%)")
                                            except Exception:
                                                pass
                                        executed = await _try_execute('scalp', sc_sig, ml_score=0.0, threshold=0.0)
                                        if executed:
                                            try:
                                                sp['count'] = int(sp.get('count', 0)) + 1
                                                sp['active'] = True
                                                shared['scalp_promotion'] = sp
                                            except Exception:
                                                pass
                                            # Skip phantom path for this symbol this loop
                                            continue
                                        else:
                                            logger.info(f"[{sym}] 🛑 Scalp Promotion blocked: reason=exec_guard")
                                    except Exception as se:
                                        logger.info(f"[{sym}] Scalp Promotion failed: {se}")
                                elif ready and used < cap and sc_sig is None:
                                    # Already logged above; ensure visibility
                                    logger.info(f"[{sym}] 🛑 Scalp Promotion blocked: reason=signal_absent")
                        except Exception:
                            pass

                        # Done with independence for this symbol
                        continue

                    # --- ENHANCED PARALLEL STRATEGY ROUTING ---
                    sig = None
                    selected_strategy = "trend_pullback"  # Default
                    selected_ml_scorer = ml_scorer
                    selected_phantom_tracker = phantom_tracker

                    if use_enhanced_parallel and ENHANCED_ML_AVAILABLE:
                        # Use enhanced regime detection for strategy routing
                        regime_analysis = get_enhanced_market_regime(df, sym)

                        # Enhanced regime analysis logging
                        logger.debug(f"🔍 [{sym}] MARKET ANALYSIS:")
                        logger.info(f"   📊 Regime: {regime_analysis.primary_regime.upper()} (confidence: {regime_analysis.regime_confidence:.1%})")
                        logger.info(f"   📈 Trend Strength: {regime_analysis.trend_strength:.1f} | Volatility: {regime_analysis.volatility_level}")
                        if regime_analysis.primary_regime == "ranging":
                            logger.info(f"   📦 Range Quality: {regime_analysis.range_quality} | Persistence: {regime_analysis.regime_persistence:.1%}")
                        # Suppressed global recommendation logging in favor of per-strategy router
                        # logger.info(f"   🎯 Recommended Strategy: {regime_analysis.recommended_strategy.upper().replace('_', ' ')}")

                        # Soft routing + hysteresis
                        try:
                            routing_state = shared.get('routing_state', {})
                            prev_state = routing_state.get(sym)
                        except Exception:
                            prev_state = None

                        uncertain = (
                            str(regime_analysis.recommended_strategy or 'none') == 'none' or
                            float(getattr(regime_analysis, 'regime_confidence', 0.0)) < 0.6 or
                            float(getattr(regime_analysis, 'regime_persistence', 0.0)) < 0.6
                        )
                        # Strategy isolation: always evaluate both strategies and arbitrate execution
                        try:
                            iso_cfg = cfg.get('strategy_isolation', {}) if 'cfg' in locals() else {}
                            if bool(iso_cfg.get('enabled', False)):
                                uncertain = True
                        except Exception:
                            pass

                        # Hysteresis: keep previous route for 3 candles unless high confidence shift
                        keep_prev = False
                        # Disable hysteresis when isolation is enabled to avoid cross-strategy stickiness
                        try:
                            if bool(cfg.get('strategy_isolation', {}).get('enabled', False)):
                                keep_prev = False
                        except Exception:
                            pass
                        if prev_state is not None:
                            try:
                                if (candles_processed - int(prev_state.get('last_idx', 0)) < 3):
                                    if float(getattr(regime_analysis, 'regime_confidence', 0.0)) < 0.7 or float(getattr(regime_analysis, 'regime_persistence', 0.0)) < 0.7:
                                        keep_prev = True
                            except Exception:
                                pass

                        handled = False
                        if uncertain:
                            logger.info(f"[{sym}] 🧭 SOFT ROUTING: Uncertain regime (conf={regime_analysis.regime_confidence:.1%}, persist={regime_analysis.regime_persistence:.1%}) → evaluate both strategies")
                            # Detect MR and Trend both
                            soft_sig_mr = None
                            soft_sig_tr = None
                            try:
                                soft_sig_mr = detect_signal_mean_reversion(df.copy(), settings, sym)
                            except Exception:
                                soft_sig_mr = None
                            try:
                                if len(df) < getattr(self, '_trend_hist_min', 80):
                                    if not self._trend_hist_warned.get(sym, False):
                                        logger.info(f"[{sym}] 🔵 Trend waiting for history: have {len(df)}/{getattr(self, '_trend_hist_min', 80)} 15m bars")
                                        self._trend_hist_warned[sym] = True
                                    soft_sig_tr = None
                                else:
                                    soft_sig_tr = detect_trend_signal(df.copy(), trend_settings, sym)
                            except Exception:
                                soft_sig_tr = None

                            # Score both with ML (if available)
                            tr_margin = -999; mr_margin = -999
                            tr_score = mr_score = 0.0
                            if soft_sig_tr and ('get_trend_scorer' in globals()) and get_trend_scorer is not None:
                                try:
                                    tr_scorer = get_trend_scorer()
                                    # build basic trend features similar to earlier
                                    cl = df['close']; price = float(cl.iloc[-1])
                                    ys = cl.tail(20).values if len(cl) >= 20 else cl.values
                                    try:
                                        slope = np.polyfit(np.arange(len(ys)), ys, 1)[0]
                                    except Exception:
                                        slope = 0.0
                                    trend_slope_pct = float((slope / price) * 100.0) if price else 0.0
                                    ema20 = cl.ewm(span=20, adjust=False).mean().iloc[-1]
                                    ema50 = cl.ewm(span=50, adjust=False).mean().iloc[-1] if len(cl) >= 50 else ema20
                                    ema_stack_score = 100.0 if (price > ema20 > ema50 or price < ema20 < ema50) else 50.0 if (ema20 != ema50) else 0.0
                                    rng_today = float(df['high'].iloc[-1] - df['low'].iloc[-1])
                                    med_range = float((df['high'] - df['low']).rolling(20).median().iloc[-1]) if len(df) >= 20 else rng_today
                                    range_expansion = float(rng_today / max(1e-9, med_range))
                                    prev = cl.shift(); trarr = np.maximum(df['high'] - df['low'], np.maximum((df['high'] - prev).abs(), (df['low'] - prev).abs()))
                                    atr = float(trarr.rolling(14).mean().iloc[-1]) if len(trarr) >= 14 else float(trarr.iloc[-1])
                                    atr_pct = float((atr / max(1e-9, price)) * 100.0) if price else 0.0
                                    close_vs_ema20_pct = float(((price - ema20) / max(1e-9, ema20)) * 100.0) if ema20 else 0.0
                                    trend_features = {
                                        'trend_slope_pct': trend_slope_pct,
                                        'ema_stack_score': ema_stack_score,
                                        'atr_pct': atr_pct,
                                        'range_expansion': range_expansion,
                                        'breakout_dist_atr': float(getattr(soft_sig_tr, 'meta', {}).get('breakout_dist_atr', 0.0)),
                                        'close_vs_ema20_pct': close_vs_ema20_pct,
                                        'bb_width_pct': 0.0,
                                        'session': 'us',
                                        'symbol_cluster': 3,
                                        'volatility_regime': getattr(regime_analysis, 'volatility_level', 'normal'),
                                    }
                                    # Attach HTF snapshots + comp (flattened)
                                    try:
                                        trend_features['htf'] = dict(self._compute_symbol_htf_exec_metrics(sym, df))
                                    except Exception:
                                        pass
                                    try:
                                        comp2 = self._get_htf_metrics(sym, df)
                                        trend_features['htf_comp'] = dict(comp2)
                                        trend_features['ts15'] = float(comp2.get('ts15', 0.0))
                                        trend_features['ts60'] = float(comp2.get('ts60', 0.0))
                                        trend_features['rc15'] = float(comp2.get('rc15', 0.0))
                                        trend_features['rc60'] = float(comp2.get('rc60', 0.0))
                                    except Exception:
                                        pass
                                    tr_score, _ = tr_scorer.score_signal(soft_sig_tr.__dict__, trend_features)
                                    tr_thr = getattr(tr_scorer, 'min_score', 70)
                                    tr_margin = tr_score - tr_thr
                                except Exception:
                                    pass
                            if soft_sig_mr and enhanced_mr_scorer:
                                try:
                                    ef = soft_sig_mr.meta.get('mr_features', {}) if soft_sig_mr.meta else {}
                                    mr_score, _ = enhanced_mr_scorer.score_signal(soft_sig_mr.__dict__, ef, df)
                                    mr_thr = getattr(enhanced_mr_scorer, 'min_score', 75)
                                    mr_margin = mr_score - mr_thr
                                except Exception:
                                    pass

                            # Choose by margin if available
                            chosen = None
                            if soft_sig_tr and tr_margin >= 0 and (tr_margin >= (mr_margin if mr_margin is not None else -999)):
                                chosen = ('trend_pullback', soft_sig_tr)
                            if soft_sig_mr and mr_margin >= 0 and (mr_margin > (tr_margin if tr_margin is not None else -999)):
                                chosen = ('enhanced_mr', soft_sig_mr)

                            if chosen is None:
                                # Trend Promotion: force execute Trend when corking is active and caps allow
                                try:
                                    tr_cfg = (self.config.get('trend', {}) or {}).get('promotion', {})
                                    tp = shared.get('trend_promotion', {})
                                    cap = int(tr_cfg.get('daily_exec_cap', 20))
                                    # Volatility block if configured
                                    allow_tr = True
                                    if bool(tr_cfg.get('block_extreme_vol', True)):
                                        allow_tr = getattr(regime_analysis, 'volatility_level', 'normal') != 'extreme'
                                    if False and soft_sig_tr and tp.get('active') and int(tp.get('count', 0)) < cap and allow_tr:
                                        # One-way per symbol guard
                                        if sym not in book.positions:
                                            # Respect allow_cork_override for bootstrap phantom mode
                                            try:
                                                exec_cfg = (cfg.get('trend', {}) or {}).get('exec', {})
                                                if bool(exec_cfg.get('bootstrap_phantom_only', False)) and not bool(exec_cfg.get('allow_cork_override', True)):
                                                    raise Exception('Trend bootstrap phantom mode without cork override')
                                            except Exception:
                                                pass
                                            # Execute trend now (bypass ML/regime gates)
                                            executed = await _try_execute('trend_pullback', soft_sig_tr, ml_score=tr_score if 'tr_score' in locals() and tr_score is not None else 0.0, threshold=tr_thr if 'tr_thr' in locals() else 70)
                                            if executed:
                                                # Mark promotion for this symbol and increment cap
                                                try:
                                                    # Tag signal meta
                                                    if not soft_sig_tr.meta:
                                                        soft_sig_tr.meta = {}
                                                    soft_sig_tr.meta['promotion_forced'] = True
                                                    # Increment cap count
                                                    tp['count'] = int(tp.get('count', 0)) + 1
                                                    shared['trend_promotion'] = tp
                                                    # Telegram announce
                                                    if self.tg:
                                                        await self.tg.send_message(f"🚀 Trend Promotion: Force executing {sym} {soft_sig_tr.side.upper()} (cap {tp['count']}/{cap})")
                                                except Exception:
                                                    pass
                                                # Move on to next symbol after executing
                                                continue
                                except Exception:
                                    pass
                                # MR Promotion: force execute MR even if ML thresholds not met (bypass guards) when recent WR ≥ promote_wr
                                try:
                                    prom_cfg = (self.config.get('mr', {}) or {}).get('promotion', {})
                                    promote_wr = float(prom_cfg.get('promote_wr', 50.0))
                                    mr_stats = enhanced_mr_scorer.get_enhanced_stats() if enhanced_mr_scorer else {}
                                    recent_wr = float(mr_stats.get('recent_win_rate', 0.0))
                                    if soft_sig_mr and recent_wr >= promote_wr:
                                        # Execute MR now (ignore ML thresholds/caps)
                                        try:
                                            if self.tg:
                                                try:
                                                    await self.tg.send_message(f"🌀 MR Promotion: Force executing {sym} {soft_sig_mr.side.upper()} (WR ≥ {promote_wr:.0f}%)")
                                                except Exception:
                                                    pass
                                            # Prevent duplicate
                                            if sym in book.positions:
                                                logger.info(f"[{sym}] Promotion forced (soft routing), but position already open. Skipping duplicate.")
                                                raise Exception("position_exists")
                                            # Meta and TP/SL rounding
                                            m = meta_for(sym, shared["meta"])
                                            from position_mgr import round_step
                                            tick_size = m.get("tick_size", 0.000001)
                                            original_tp = soft_sig_mr.tp; original_sl = soft_sig_mr.sl
                                            soft_sig_mr.tp = round_step(soft_sig_mr.tp, tick_size)
                                            soft_sig_mr.sl = round_step(soft_sig_mr.sl, tick_size)
                                            if original_tp != soft_sig_mr.tp or original_sl != soft_sig_mr.sl:
                                                logger.info(f"[{sym}] Rounded TP/SL to tick size {tick_size}. TP: {original_tp:.6f} -> {soft_sig_mr.tp:.6f}, SL: {original_sl:.6f} -> {soft_sig_mr.sl:.6f}")
                                            # Balance and sizing
                                            current_balance = bybit.get_balance()
                                            if current_balance:
                                                sizer.account_balance = current_balance
                                                shared["last_balance"] = current_balance
                                            risk_amount = sizer.account_balance * (risk.risk_percent / 100.0) if risk.use_percent_risk and sizer.account_balance else risk.risk_usd
                                            qty = sizer.qty_for(soft_sig_mr.entry, soft_sig_mr.sl, m.get("qty_step",0.001), m.get("min_qty",0.001), ml_score=0.0)
                                            if qty <= 0:
                                                logger.info(f"[{sym}] Promotion forced (soft routing), but qty calc invalid -> skip")
                                                raise Exception("invalid_qty")
                                            # SL sanity
                                            current_price = df['close'].iloc[-1]
                                            if (soft_sig_mr.side == "long" and soft_sig_mr.sl >= current_price) or (soft_sig_mr.side == "short" and soft_sig_mr.sl <= current_price):
                                                logger.warning(f"[{sym}] Promotion forced (soft routing), but SL invalid relative to current price -> skip")
                                                raise Exception("invalid_sl")
                                            # Set leverage and place order
                                            max_lev = int(m.get("max_leverage", 10))
                                            bybit.set_leverage(sym, max_lev)
                                            side = "Buy" if soft_sig_mr.side == "long" else "Sell"
                                            logger.info(f"[{sym}] MR Promotion (soft) placing {side} order for {qty} units")
                                            order_result = bybit.place_market(sym, side, qty, reduce_only=False)
                                            # Adjust TP to actual entry
                                            actual_entry = soft_sig_mr.entry
                                            try:
                                                await asyncio.sleep(0.5)
                                                position = bybit.get_position(sym)
                                                if position and position.get("avgPrice"):
                                                    actual_entry = float(position["avgPrice"])
                                                    if actual_entry != soft_sig_mr.entry:
                                                        fee_adjustment = 1.00165
                                                        risk_distance = abs(actual_entry - soft_sig_mr.sl)
                                                        if soft_sig_mr.side == "long":
                                                            soft_sig_mr.tp = actual_entry + (settings.rr * risk_distance * fee_adjustment)
                                                        else:
                                                            soft_sig_mr.tp = actual_entry - (settings.rr * risk_distance * fee_adjustment)
                                                        logger.info(f"[{sym}] MR Promotion (soft) TP adjusted for actual entry: {soft_sig_mr.tp:.4f}")
                                            except Exception:
                                                pass
                                            # Set TP/SL
                                            bybit.set_tpsl(sym, take_profit=soft_sig_mr.tp, stop_loss=soft_sig_mr.sl, qty=qty)
                                            # Update book
                                            # Attach qscore if known from last features
                                            try:
                                                qv = float(((getattr(self, '_last_signal_features', {}) or {}).get(sym, {}) or {}).get('qscore', 0.0) or 0.0)
                                            except Exception:
                                                qv = 0.0
                                            book.positions[sym] = Position(
                                                soft_sig_mr.side,
                                                qty,
                                                entry=actual_entry,
                                                sl=soft_sig_mr.sl,
                                                tp=soft_sig_mr.tp,
                                                entry_time=datetime.now(),
                                                strategy_name='enhanced_mr',
                                                qscore=qv
                                            )
                                            # Mark promotion flag
                                            try:
                                                if not soft_sig_mr.meta:
                                                    soft_sig_mr.meta = {}
                                                soft_sig_mr.meta['promotion_forced'] = True
                                            except Exception:
                                                pass
                                            # Standard open message
                                            if self.tg:
                                                try:
                                                    emoji = '📈'
                                                    strategy_label = 'Enhanced Mr'
                                                    msg = (
                                                        f"{emoji} *{sym} {soft_sig_mr.side.upper()}* ({strategy_label})\n\n"
                                                        f"Entry: {actual_entry:.4f}\n"
                                                        f"Stop Loss: {soft_sig_mr.sl:.4f}\n"
                                                        f"Take Profit: {soft_sig_mr.tp:.4f}\n"
                                                        f"Quantity: {qty}\n"
                                                        f"Risk: {risk.risk_percent if risk.use_percent_risk else risk.risk_usd}{'%' if risk.use_percent_risk else ''} (${risk_amount:.2f})\n"
                                                        f"Promotion: FORCED (WR ≥ {promote_wr:.0f}%)\n"
                                                        f"Reason: Mean Reversion (Promotion)"
                                                    )
                                                    await self.tg.send_message(msg)
                                                except Exception:
                                                    pass
                                            # Done with this symbol
                                            continue
                                        except Exception as e:
                                            logger.warning(f"[{sym}] MR Promotion forced execution (soft routing) failed: {e}")
                                            if self.tg:
                                                try:
                                                    await self.tg.send_message(f"🛑 MR Promotion: Failed to execute {sym} {soft_sig_mr.side.upper()} — {str(e)[:120]}")
                                                except Exception:
                                                    pass
                                            # Fallthrough to phantom recording if execution fails
                                            pass
                                except Exception:
                                    pass
                                # Record phantoms (if signals exist) and continue to next symbol
                                try:
                                    # Enforce per-strategy hourly budget (trend)
                                    hb = cfg.get('phantom', {}).get('hourly_symbol_budget', {}) or {}
                                    pb_limit = int(hb.get('trend', hb.get('pullback', 3)))
                                    if 'phantom_budget' not in shared or not isinstance(shared['phantom_budget'], dict):
                                        shared['phantom_budget'] = {}
                                    pb_map = shared['phantom_budget'].setdefault('trend', {})
                                    now_ts = time.time(); one_hour_ago = now_ts - 3600
                                    pb_list = [ts for ts in pb_map.get(sym, []) if ts > one_hour_ago]
                                    pb_map[sym] = pb_list
                                    shared['phantom_budget']['trend'] = pb_map
                                    pb_remaining = pb_limit - len(pb_list)
                                    if soft_sig_tr and phantom_tracker and pb_remaining > 0:
                                        phantom_tracker.record_signal(sym, {'side': soft_sig_tr.side, 'entry': soft_sig_tr.entry, 'sl': soft_sig_tr.sl, 'tp': soft_sig_tr.tp}, float(tr_score or 0.0), False, trend_features, 'trend_pullback')
                                        try:
                                            if hasattr(self, 'flow_controller') and self.flow_controller and self.flow_controller.enabled:
                                                self.flow_controller.increment_accepted('trend', 1)
                                        except Exception:
                                            pass
                                        pb_list.append(now_ts)
                                        pb_map[sym] = pb_list
                                except Exception:
                                    pass
                                try:
                                    # Enforce per-strategy hourly budget (mr)
                                    hb = cfg.get('phantom', {}).get('hourly_symbol_budget', {}) or {}
                                    mr_limit = int(hb.get('mr', 3))
                                    if 'phantom_budget' not in shared or not isinstance(shared['phantom_budget'], dict):
                                        shared['phantom_budget'] = {}
                                    mr_map = shared['phantom_budget'].setdefault('mr', {})
                                    now_ts = time.time(); one_hour_ago = now_ts - 3600
                                    mr_list = [ts for ts in mr_map.get(sym, []) if ts > one_hour_ago]
                                    mr_map[sym] = mr_list
                                    shared['phantom_budget']['mr'] = mr_map
                                    mr_remaining = mr_limit - len(mr_list)
                                    if soft_sig_mr and mr_phantom_tracker and mr_remaining > 0:
                                        ef2 = soft_sig_mr.meta.get('mr_features', {}) if soft_sig_mr.meta else {}
                                        mr_phantom_tracker.record_mr_signal(sym, soft_sig_mr.__dict__, float(mr_score or 0.0), False, {}, ef2)
                                        try:
                                            if hasattr(self, 'flow_controller') and self.flow_controller and self.flow_controller.enabled:
                                                self.flow_controller.increment_accepted('mr', 1)
                                        except Exception:
                                            pass
                                        mr_list.append(now_ts)
                                        mr_map[sym] = mr_list
                                except Exception:
                                    pass
                                logger.info(f"[{sym}] 🧭 SOFT ROUTING: No strategy exceeded ML threshold — recorded phantoms, skipping execution")
                                # Skip further processing for this symbol; no selected strategy/signal set
                                continue
                            else:
                                # Strategy isolation arbitration: pick chosen and proceed to execution
                                try:
                                    selected_strategy = chosen[0]
                                    if selected_strategy == 'enhanced_mr':
                                        selected_ml_scorer = enhanced_mr_scorer
                                        sig = chosen[1]
                                    else:
                                        selected_ml_scorer = ml_scorer
                                        sig = chosen[1]
                                    # mark handled so we skip recommended strategy branches
                                    handled = True
                                    # Update routing state stickiness record
                                    try:
                                        routing_state[sym] = {'strategy': selected_strategy, 'last_idx': candles_processed}
                                        shared['routing_state'] = routing_state
                                    except Exception:
                                        pass
                                except Exception:
                                    pass
                                continue

                            # Apply hysteresis if requested
                            if keep_prev and prev_state and chosen[0] != prev_state.get('strategy'):
                                logger.info(f"[{sym}] 🧭 Hysteresis: keeping previous route {prev_state.get('strategy')} over {chosen[0]}")
                                selected_strategy = prev_state.get('strategy')
                                if selected_strategy == 'enhanced_mr':
                                    selected_ml_scorer = enhanced_mr_scorer
                                    selected_phantom_tracker = mr_phantom_tracker
                                    sig = soft_sig_mr
                                else:
                                    selected_ml_scorer = ml_scorer
                                    selected_phantom_tracker = phantom_tracker
                                    sig = soft_sig_pb
                            else:
                                selected_strategy = chosen[0]
                                if selected_strategy == 'enhanced_mr':
                                    selected_ml_scorer = enhanced_mr_scorer
                                    selected_phantom_tracker = mr_phantom_tracker
                                    sig = soft_sig_mr
                                else:
                                    selected_ml_scorer = ml_scorer
                                    selected_phantom_tracker = phantom_tracker
                                    sig = soft_sig_pb

                            # Update routing state stickiness
                            try:
                                routing_state[sym] = {'strategy': selected_strategy, 'last_idx': candles_processed}
                                shared['routing_state'] = routing_state
                            except Exception:
                                pass
                            handled = True

                        # If not uncertain, proceed with recommended strategy with optional override check
                        if not uncertain:
                            # Check hysteresis preference
                            if keep_prev and prev_state and prev_state.get('strategy') in ('trend_pullback','enhanced_mr'):
                                logger.info(f"[{sym}] 🧭 Hysteresis: keeping previous route {prev_state.get('strategy')}")
                                if prev_state.get('strategy') == 'enhanced_mr':
                                    selected_strategy = 'enhanced_mr'
                                    selected_ml_scorer = enhanced_mr_scorer
                                    selected_phantom_tracker = mr_phantom_tracker
                                else:
                                    selected_strategy = 'trend_pullback'
                                    selected_ml_scorer = get_trend_scorer() if 'get_trend_scorer' in globals() and get_trend_scorer is not None else None
                                    selected_phantom_tracker = phantom_tracker
                            # Else start with recommended and optionally override later after signal detection

                        # Router override using per-strategy regime scores
                        if (not handled) and router_choice in ("enhanced_mr", "pullback", "trend_pullback"):
                            if router_choice == "enhanced_mr":
                                logger.debug(f"🟢 [{sym}] ROUTER OVERRIDE → ENHANCED MR ANALYSIS:")
                                sig = detect_signal_mean_reversion(df.copy(), settings, sym)
                                selected_strategy = "enhanced_mr"
                                selected_ml_scorer = enhanced_mr_scorer
                                selected_phantom_tracker = mr_phantom_tracker
                                # Update routing state stickiness
                                try:
                                    routing_state[sym] = {'strategy': selected_strategy, 'last_idx': candles_processed}
                                    shared['routing_state'] = routing_state
                                except Exception:
                                    pass
                                handled = True
                            elif router_choice == "pullback":
                                # Backward-compat: treat legacy 'pullback' route as Trend Pullback
                                logger.debug(f"🔵 [{sym}] ROUTER OVERRIDE → TREND PULLBACK ANALYSIS (legacy pullback route)")
                                # Ensure sufficient 15m history
                                if len(df) < getattr(self, '_trend_hist_min', 80):
                                    if not self._trend_hist_warned.get(sym, False):
                                        logger.info(f"[{sym}] 🔵 Trend waiting for history: have {len(df)}/{getattr(self, '_trend_hist_min', 80)} 15m bars")
                                        self._trend_hist_warned[sym] = True
                                    sig = None
                                else:
                                    sig = detect_trend_signal(df.copy(), trend_settings, sym)
                                selected_strategy = "trend_pullback"
                                selected_ml_scorer = get_trend_scorer() if 'get_trend_scorer' in globals() and get_trend_scorer is not None else None
                                selected_phantom_tracker = phantom_tracker
                                try:
                                    routing_state[sym] = {'strategy': selected_strategy, 'last_idx': candles_processed}
                                    shared['routing_state'] = routing_state
                                except Exception:
                                    pass
                                handled = True
                            else:
                                logger.debug(f"🟣 [{sym}] ROUTER OVERRIDE → TREND PULLBACK ANALYSIS:")
                                if len(df) < getattr(self, '_trend_hist_min', 80):
                                    if not self._trend_hist_warned.get(sym, False):
                                        logger.info(f"[{sym}] 🔵 Trend waiting for history: have {len(df)}/{getattr(self, '_trend_hist_min', 80)} 15m bars")
                                        self._trend_hist_warned[sym] = True
                                    sig = None
                                else:
                                    sig = detect_trend_signal(df.copy(), trend_settings, sym)
                                selected_strategy = "trend_pullback"
                                selected_ml_scorer = get_trend_scorer() if 'get_trend_scorer' in globals() and get_trend_scorer is not None else None
                                selected_phantom_tracker = phantom_tracker
                                try:
                                    routing_state[sym] = {'strategy': selected_strategy, 'last_idx': candles_processed}
                                    shared['routing_state'] = routing_state
                                except Exception:
                                    pass
                                handled = True

                        if (not handled) and selected_strategy == "trend_pullback":
                            # Trend pullback strategy scoring and gating
                            if sig is None:
                                # Guard on 15m history length
                                if len(df) < getattr(self, '_trend_hist_min', 80):
                                    if not self._trend_hist_warned.get(sym, False):
                                        logger.info(f"[{sym}] 🔵 Trend waiting for history: have {len(df)}/{getattr(self, '_trend_hist_min', 80)} 15m bars")
                                        self._trend_hist_warned[sym] = True
                                    sig = None
                                else:
                                    sig = detect_trend_signal(df.copy(), trend_settings, sym)
                            if sig:
                                # Build trend features for ML
                                try:
                                    # trend features
                                    cl = df['close']
                                    price = float(cl.iloc[-1])
                                    x = np.arange(min(20, len(cl)))
                                    ys = cl.tail(20).values if len(cl) >= 20 else cl.values
                                    slope = 0.0
                                    try:
                                        slope = np.polyfit(np.arange(len(ys)), ys, 1)[0]
                                    except Exception:
                                        pass
                                    trend_slope_pct = float((slope / price) * 100.0) if price else 0.0
                                    ema20 = cl.ewm(span=20, adjust=False).mean().iloc[-1]
                                    ema50 = cl.ewm(span=50, adjust=False).mean().iloc[-1] if len(cl) >= 50 else ema20
                                    ema_stack_score = 100.0 if (price > ema20 > ema50 or price < ema20 < ema50) else 50.0 if (ema20 != ema50) else 0.0
                                    rng_today = float(df['high'].iloc[-1] - df['low'].iloc[-1])
                                    med_range = float((df['high'] - df['low']).rolling(20).median().iloc[-1]) if len(df) >= 20 else rng_today
                                    range_expansion = float(rng_today / max(1e-9, med_range))
                                    prev = cl.shift(); tr = np.maximum(df['high'] - df['low'], np.maximum((df['high'] - prev).abs(), (df['low'] - prev).abs()))
                                    atr = float(tr.rolling(14).mean().iloc[-1]) if len(tr) >= 14 else float(tr.iloc[-1])
                                    atr_pct = float((atr / max(1e-9, price)) * 100.0) if price else 0.0
                                    close_vs_ema20_pct = float(((price - ema20) / max(1e-9, ema20)) * 100.0) if ema20 else 0.0
                                    bb_width_pct = 0.0
                                    if len(cl) >= 20:
                                        ma = cl.rolling(20).mean(); sd = cl.rolling(20).std()
                                        upper = float(ma.iloc[-1] + 2*sd.iloc[-1]); lower = float(ma.iloc[-1] - 2*sd.iloc[-1])
                                        bb_width_pct = float((upper - lower) / max(1e-9, price))
                                    # cluster id not readily available here; default 3
                                    trend_features = {
                                        'trend_slope_pct': trend_slope_pct,
                                        'ema_stack_score': ema_stack_score,
                                        'atr_pct': atr_pct,
                                        'range_expansion': range_expansion,
                                        'breakout_dist_atr': float(sig.meta.get('breakout_dist_atr', 0.0)) if getattr(sig, 'meta', None) else 0.0,
                                        'close_vs_ema20_pct': close_vs_ema20_pct,
                                        'bb_width_pct': bb_width_pct,
                                        'session': 'us',
                                        'symbol_cluster': 3,
                                        'volatility_regime': regime_analysis.volatility_level if hasattr(regime_analysis, 'volatility_level') else 'normal'
                                    }
                                    # Attach full HTF snapshot for ML/phantom
                                    try:
                                        htfm = self._compute_symbol_htf_exec_metrics(sym, df)
                                        trend_features['htf'] = dict(htfm)
                                    except Exception:
                                        pass
                                except Exception:
                                    trend_features = {}

                                # Score
                                trend_scorer = get_trend_scorer() if 'get_trend_scorer' in globals() and get_trend_scorer is not None else None
                                ml_score = 0.0; should_take_trade = True; ml_reason = 'Trend ML disabled'
                        if trend_scorer is not None:
                            try:
                                ml_score, ml_reason = trend_scorer.score_signal(sig.__dict__, trend_features)
                                threshold = getattr(trend_scorer, 'min_score', 70)
                                should_take_trade = ml_score >= threshold
                                # Apply per-symbol HTF exec gate
                                try:
                                    if should_take_trade:
                                        ok_gate, new_thr, mode, _m = self._apply_htf_exec_gate(sym, df, sig.side, threshold)
                                        if not ok_gate and mode == 'gated':
                                            should_take_trade = False
                                        elif not ok_gate and mode == 'soft':
                                            threshold = float(new_thr)
                                            should_take_trade = ml_score >= threshold
                                except Exception:
                                    pass
                                # Trend decision context (execution path)
                                try:
                                    pb_limit = int((self.config.get('phantom', {}).get('hourly_symbol_budget', {}) or {}).get('trend', (self.config.get('phantom', {}).get('hourly_symbol_budget', {}) or {}).get('pullback', 3)))
                                except Exception:
                                    pb_limit = 3
                                try:
                                    pb_map = (self.shared.get('phantom_budget', {}) or {}).get('trend', {}) if hasattr(self, 'shared') else {}
                                    pb_remaining = max(0, pb_limit - len(pb_map.get(sym, [])))
                                except Exception:
                                    pb_remaining = 'n/a'
                                # EV threshold (if available)
                                try:
                                    ev_thr = float(trend_scorer.get_ev_threshold(trend_features))
                                except Exception:
                                    ev_thr = None
                                try:
                                    ctx = f"dedup=True hourly_remaining={pb_remaining} daily_ok=True"
                                    if ev_thr is not None:
                                        ctx += f" ev_thr={ev_thr:.0f}"
                                    logger.info(f"[{sym}] 🔵 Trend decision context: {ctx}")
                                except Exception:
                                    pass
                                rule_mode = (cfg.get('trend', {}) or {}).get('rule_mode', {}) if 'cfg' in locals() else {}
                                rm_enabled = bool(rule_mode.get('enabled', False))
                                if rm_enabled:
                                    try:
                                        q, qc, qr = self._compute_qscore(sym, sig.side, df, self.frames_3m.get(sym) if hasattr(self, 'frames_3m') else None)
                                    except Exception:
                                        q, qc, qr = 50.0, {}, []
                                    # Attach Qscore to features for training
                                    try:
                                        trend_features['qscore'] = float(q)
                                        trend_features['qscore_components'] = dict(qc)
                                        trend_features['qscore_reasons'] = list(qr)
                                    except Exception:
                                        pass
                                    exec_min = float(rule_mode.get('execute_q_min', 78))
                                    ph_min = float(rule_mode.get('phantom_q_min', 65))
                                    # Apply rule decision irrespective of ML
                                    should_take_trade = (q >= exec_min)
                                    comps = f"SR={qc.get('sr',0):.0f} HTF={qc.get('htf',0):.0f} BOS={qc.get('bos',0):.0f} Micro={qc.get('micro',0):.0f} Risk={qc.get('risk',0):.0f} Div={qc.get('div',0):.0f}"
                                    if should_take_trade:
                                        logger.info(f"[{sym}] 🧮 Rule-mode: EXECUTE (Q={q:.1f} ≥ {exec_min:.1f}) comps: {comps}")
                                        try:
                                            if self.tg:
                                                # Compute regime label if possible
                                                try:
                                                    t15 = float(trend_features.get('ts15',0.0)); t60 = float(trend_features.get('ts60',0.0)); rc15 = float(trend_features.get('rc15',0.0)); rc60 = float(trend_features.get('rc60',0.0))
                                                    reg = 'Trending' if (t15>=60 or t60>=60) else ('Ranging' if (rc15>=0.6 or rc60>=0.6) else 'Neutral')
                                                except Exception:
                                                    reg = None
                                                # Stash features for qscore carry-over
                                                try:
                                                    if not hasattr(self, '_last_signal_features'):
                                                        self._last_signal_features = {}
                                                    self._last_signal_features[sym] = dict(feats)
                                                except Exception:
                                                    pass
                                                msg = f"🟢 Rule-mode EXECUTE: {sym} {sig.side.upper()} Q={q:.1f} (≥ {exec_min:.1f})\n{comps}"
                                                if reg:
                                                    msg += f"\nRegime: {reg}"
                                                await self.tg.send_message(msg)
                                        except Exception:
                                            pass
                                        try:
                                            trend_features['decision'] = 'rule_execute'
                                        except Exception:
                                            pass
                                        # Append to events feed
                                        try:
                                            evts = self.shared.get('trend_events')
                                            if isinstance(evts, list):
                                                from datetime import datetime as _dt
                                                evts.append({'ts': _dt.utcnow().isoformat()+'Z', 'symbol': sym, 'text': f"Rule EXECUTE Q={q:.1f} comps: {comps}"})
                                                if len(evts) > 60:
                                                    del evts[:len(evts)-60]
                                        except Exception:
                                            pass
                                    elif q >= ph_min:
                                        logger.info(f"[{sym}] 🧮 Rule-mode: PHANTOM (Q={q:.1f} < {exec_min:.1f}) comps: {comps}")
                                        try:
                                            if self.tg:
                                                try:
                                                    t15 = float(trend_features.get('ts15',0.0)); t60 = float(trend_features.get('ts60',0.0)); rc15 = float(trend_features.get('rc15',0.0)); rc60 = float(trend_features.get('rc60',0.0))
                                                    reg = 'Trending' if (t15>=60 or t60>=60) else ('Ranging' if (rc15>=0.6 or rc60>=0.6) else 'Neutral')
                                                    await self.tg.send_message(f"🟡 Rule-mode PHANTOM: [{sym}] Q={q:.1f} < {exec_min:.1f}\n{comps}\nRegime: {reg}")
                                                except Exception:
                                                    await self.tg.send_message(f"🟡 Rule-mode PHANTOM: [{sym}] Q={q:.1f} < {exec_min:.1f}\n{comps}")
                                        except Exception:
                                            pass
                                        try:
                                            trend_features['decision'] = 'rule_phantom'
                                        except Exception:
                                            pass
                                        try:
                                            evts = self.shared.get('trend_events')
                                            if isinstance(evts, list):
                                                from datetime import datetime as _dt
                                                evts.append({'ts': _dt.utcnow().isoformat()+'Z', 'symbol': sym, 'text': f"Rule PHANTOM Q={q:.1f} comps: {comps}"})
                                                if len(evts) > 60:
                                                    del evts[:len(evts)-60]
                                        except Exception:
                                            pass
                                    else:
                                        logger.info(f"[{sym}] 🧮 Rule-mode: PHANTOM (low-quality Q={q:.1f} < {ph_min:.1f}) comps: {comps}")
                                        try:
                                            if self.tg:
                                                try:
                                                    t15 = float(trend_features.get('ts15',0.0)); t60 = float(trend_features.get('ts60',0.0)); rc15 = float(trend_features.get('rc15',0.0)); rc60 = float(trend_features.get('rc60',0.0))
                                                    reg = 'Trending' if (t15>=60 or t60>=60) else ('Ranging' if (rc15>=0.6 or rc60>=0.6) else 'Neutral')
                                                    await self.tg.send_message(f"🟠 Rule-mode LOW-QUALITY: [{sym}] Q={q:.1f} < {ph_min:.1f} — phantom (low-weight)\n{comps}\nRegime: {reg}")
                                                except Exception:
                                                    await self.tg.send_message(f"🟠 Rule-mode LOW-QUALITY: [{sym}] Q={q:.1f} < {ph_min:.1f} — phantom (low-weight)\n{comps}")
                                        except Exception:
                                            pass
                                        try:
                                            trend_features['decision'] = 'rule_low_quality'
                                        except Exception:
                                            pass
                                        try:
                                            evts = self.shared.get('trend_events')
                                            if isinstance(evts, list):
                                                from datetime import datetime as _dt
                                                evts.append({'ts': _dt.utcnow().isoformat()+'Z', 'symbol': sym, 'text': f"Rule LOW-Q Q={q:.1f} comps: {comps}"})
                                                if len(evts) > 60:
                                                    del evts[:len(evts)-60]
                                        except Exception:
                                            pass
                                else:
                                    if should_take_trade:
                                        logger.info(f"[{sym}] 🧮 Trend decision final: execute (ML {ml_score:.1f} ≥ thr {threshold:.1f})")
                                    else:
                                        logger.info(f"[{sym}] 🧮 Trend decision final: phantom (ML {ml_score:.1f} < thr {threshold:.1f})")
                            except Exception as e:
                                logger.warning(f"Trend ML scoring error: {e}")
                                should_take_trade = False
                                ml_score = 0.0
                                # Persist features to attach on close for executed outcomes
                                try:
                                    if should_take_trade:
                                        self._last_signal_features[sym] = dict(trend_features)
                                except Exception:
                                    pass
                                # Regime gate for execution and phantom symmetry
                                try:
                                    ok_regime, why_reg = self._phantom_trend_regime_ok(sym, df, regime_analysis)
                                except Exception:
                                    ok_regime, why_reg = True, 'n/a'
                                # Record Trend signal into phantom tracker with regime gating for phantoms
                                try:
                                    if should_take_trade:
                                        if ok_regime:
                                            logger.info(f"[{sym}] 📊 PHANTOM ROUTING: Trend phantom tracker recording (executed=True, ML {ml_score:.1f})")
                                            phantom_tracker.record_signal(
                                                symbol=sym,
                                                signal={'side': sig.side, 'entry': sig.entry, 'sl': sig.sl, 'tp': sig.tp},
                                                ml_score=float(ml_score or 0.0),
                                                was_executed=True,
                                                features=trend_features,
                                                strategy_name='trend_pullback'
                                            )
                                        else:
                                            logger.info(f"[{sym}] 🛑 Trend execution blocked by regime gate ({why_reg})")
                                            # Treat as non-execution path
                                            should_take_trade = False
                                    else:
                                        if ok_regime:
                                            logger.info(f"[{sym}] 📊 PHANTOM ROUTING: Trend phantom tracker recording (executed=False, ML {ml_score:.1f})")
                                            phantom_tracker.record_signal(
                                                symbol=sym,
                                                signal={'side': sig.side, 'entry': sig.entry, 'sl': sig.sl, 'tp': sig.tp},
                                                ml_score=float(ml_score or 0.0),
                                                was_executed=False,
                                                features=trend_features,
                                                strategy_name='trend_pullback'
                                            )
                                        else:
                                            logger.info(f"[{sym}] 🛑 Trend phantom dropped by regime gate ({why_reg})")
                                except Exception:
                                    pass
                                if not should_take_trade:
                                    # Notify ML reject diverted to phantom
                                    try:
                                        if self.tg:
                                            await self.tg.send_message(f"🛑 Trend: [{sym}] ML reject — ML {ml_score:.1f} < thr {threshold:.1f} (phantom recorded)")
                                    except Exception:
                                        pass
                                    # Reset state to NEUTRAL since not executed
                                    try:
                                        from strategy_pullback import revert_to_neutral
                                        revert_to_neutral(sym)
                                    except Exception:
                                        pass
                                    # Skip execution, continue to next symbol
                                    continue
                            else:
                                logger.info(f"   ❌ No Trend Signal: Pullback conditions not met")

                        if (not handled) and regime_analysis.recommended_strategy == "enhanced_mr":
                            # Use Enhanced Mean Reversion System
                            logger.debug(f"🟢 [{sym}] ENHANCED MEAN REVERSION ANALYSIS:")
                            sig = detect_signal_mean_reversion(df.copy(), settings, sym)
                            selected_strategy = "enhanced_mr"
                            selected_ml_scorer = enhanced_mr_scorer
                            selected_phantom_tracker = mr_phantom_tracker

                            if sig:
                                logger.info(f"   ✅ Range Signal Detected: {sig.side.upper()} at {sig.entry:.4f}")
                                logger.info(f"   🎯 SL: {sig.sl:.4f} | TP: {sig.tp:.4f} | R:R: {((sig.tp-sig.entry)/(sig.entry-sig.sl) if sig.side=='long' else (sig.entry-sig.tp)/(sig.sl-sig.entry)):.2f}")
                                logger.info(f"   📝 Reason: {sig.reason}")
                                # No override towards legacy Pullback; Trend arbitration handled separately
                            else:
                                logger.info(f"   ❌ No Mean Reversion Signal: Range conditions not met")
                                logger.info(f"   💡 Range quality: {regime_analysis.range_quality}, confidence: {regime_analysis.regime_confidence:.1%}")

                        elif (not handled) and regime_analysis.recommended_strategy == "pullback":
                            # Treat legacy pullback recommendation as Trend Pullback
                            logger.debug(f"🔵 [{sym}] TREND PULLBACK ANALYSIS (legacy pullback route):")
                            sig = detect_trend_signal(df.copy(), trend_settings, sym)
                            selected_strategy = "trend_pullback"
                            selected_ml_scorer = get_trend_scorer() if 'get_trend_scorer' in globals() and get_trend_scorer is not None else None
                            selected_phantom_tracker = phantom_tracker

                            if sig:
                                logger.info(f"   ✅ Trend Signal Detected: {sig.side.upper()} at {sig.entry:.4f}")
                                logger.info(f"   🎯 SL: {sig.sl:.4f} | TP: {sig.tp:.4f} | R:R: {((sig.tp-sig.entry)/(sig.entry-sig.sl) if sig.side=='long' else (sig.entry-sig.tp)/(sig.sl-sig.entry)):.2f}")
                                logger.info(f"   📝 Reason: {sig.reason}")
                                # Log pullback-specific meta
                                try:
                                    _m = getattr(sig, 'meta', {}) or {}
                                    logger.info(
                                        f"   🧩 Pullback Meta: break={_m.get('break_level','n/a')} "
                                        f"confirm={int(_m.get('confirm_candles',0))} "
                                        f"break_d_atr={float(_m.get('break_dist_atr',0.0)):.2f} "
                                        f"retrace_atr={float(_m.get('retrace_depth_atr',0.0)):.2f}"
                                    )
                                except Exception:
                                    pass
                                # Override: if MR ML shows very strong signal, prefer it
                                try:
                                    alt_mr_sig = detect_signal_mean_reversion(df.copy(), settings, sym)
                                    if alt_mr_sig and enhanced_mr_scorer:
                                        alt_ef = alt_mr_sig.meta.get('mr_features', {}) if alt_mr_sig.meta else {}
                                        mr_score, _ = enhanced_mr_scorer.score_signal(alt_mr_sig.__dict__, alt_ef, df)
                                        mr_thr = getattr(enhanced_mr_scorer, 'min_score', 75)
                                        if mr_score >= mr_thr + 5:
                                            logger.info(f"[{sym}] 🔀 OVERRIDE: MR ML {mr_score:.1f} ≥ {mr_thr+5:.0f} → prefer MR over Trend")
                                            selected_strategy = 'enhanced_mr'
                                            selected_ml_scorer = enhanced_mr_scorer
                                            selected_phantom_tracker = mr_phantom_tracker
                                            sig = alt_mr_sig
                                            routing_state[sym] = {'strategy': selected_strategy, 'last_idx': candles_processed}
                                            shared['routing_state'] = routing_state
                                except Exception:
                                    pass
                            else:
                                logger.info(f"   ❌ No Trend Signal: Breakout structure insufficient")
                                logger.info(f"   💡 Trend strength: {regime_analysis.trend_strength:.1f}, volatility: {regime_analysis.volatility_level}")

                        elif (not handled):
                            # Router recommends no trading (none). Do phantom-only sampling with caps/dedup.
                            logger.debug(f"⏭️ [{sym}] STRATEGY SELECTION:")
                            logger.info(f"   ❌ SKIPPING EXECUTION - {regime_analysis.primary_regime.upper()} regime not suitable")
                            logger.info(f"   💡 Volatility: {regime_analysis.volatility_level}, confidence: {regime_analysis.regime_confidence:.1%}")
                            logger.info(f"   📊 Market needs: trending (>25) OR high-quality range")

                            # Initialize shared caches
                            if 'phantom_budget' not in shared:
                                shared['phantom_budget'] = {}
                            if 'phantom_cooldown' not in shared:
                                shared['phantom_cooldown'] = {}
                            # Try to use phantom tracker Redis for dedup (7d TTL)
                            phantom_dedup_redis = None
                            try:
                                if phantom_tracker and getattr(phantom_tracker, 'redis_client', None):
                                    phantom_dedup_redis = phantom_tracker.redis_client
                            except Exception:
                                pass

                            def _dedup_key(symbol, sig):
                                try:
                                    e = round(float(sig.entry), 4)
                                    s = round(float(sig.sl), 4)
                                    t = round(float(sig.tp), 4)
                                    return f"{symbol}:{sig.side}:{e}:{s}:{t}"
                                except Exception:
                                    return f"{symbol}:{sig.side}:{time.time()}"

                            def _not_duplicate(strategy_tag: str, symbol, sig):
                                key = _dedup_key(symbol, sig)
                                if phantom_dedup_redis:
                                    try:
                                        if phantom_dedup_redis.exists(f"phantom:dedup:{strategy_tag}:{key}"):
                                            # Track dedup hits (per day)
                                            try:
                                                from datetime import datetime as _dt
                                                day = _dt.utcnow().strftime('%Y%m%d')
                                                phantom_dedup_redis.incr(f"phantom:dedup_hits:{day}")
                                            except Exception:
                                                pass
                                            return False
                                    except Exception:
                                        pass
                                if phantom_dedup_redis:
                                    try:
                                        phantom_dedup_redis.setex(f"phantom:dedup:{strategy_tag}:{key}", 7*24*3600, '1')
                                    except Exception:
                                        pass
                                return True

                            # Daily caps via Redis (per strategy)
                            def _daily_capped(strategy: str, symbol, cluster_id: int) -> bool:
                                if not phantom_dedup_redis:
                                    return False
                                try:
                                    from datetime import datetime as _dt
                                    day = _dt.utcnow().strftime('%Y%m%d')
                                    caps_cfg = (cfg.get('phantom', {}).get('caps', {}) or {}).get(strategy, {})
                                    none_cap = int(caps_cfg.get('none', phantom_none_cap))
                                    cluster3_cap = int(caps_cfg.get('cluster3', phantom_cluster3_cap))
                                    offhours_cap = int(caps_cfg.get('offhours', phantom_offhours_cap))
                                    # None routing cap (per strategy)
                                    none_key = f"phantom:daily:none_count:{day}:{strategy}"
                                    none_val = int(phantom_dedup_redis.get(none_key) or 0)
                                    if none_val >= none_cap:
                                        return True
                                    # Cluster 3 cap
                                    if cluster_id == 3:
                                        cl_key = f"phantom:daily:cluster3_count:{day}:{strategy}"
                                        cl_val = int(phantom_dedup_redis.get(cl_key) or 0)
                                        if cl_val >= cluster3_cap:
                                            return True
                                    # Off-hours cap
                                    hour = _dt.utcnow().hour
                                    is_off = (hour >= 22 or hour < 2)
                                    if is_off:
                                        off_key = f"phantom:daily:offhours_count:{day}:{strategy}"
                                        off_val = int(phantom_dedup_redis.get(off_key) or 0)
                                        if off_val >= offhours_cap:
                                            return True
                                except Exception:
                                    return False
                                return False

                            def _increment_daily(strategy: str, symbol, cluster_id: int, routing: str):
                                if not phantom_dedup_redis:
                                    return
                                from datetime import datetime as _dt
                                day = _dt.utcnow().strftime('%Y%m%d')
                                try:
                                    phantom_dedup_redis.incr(f"phantom:daily:none_count:{day}:{strategy}")
                                    if cluster_id == 3:
                                        phantom_dedup_redis.incr(f"phantom:daily:cluster3_count:{day}:{strategy}")
                                    hour = _dt.utcnow().hour
                                    if (hour >= 22 or hour < 2):
                                        phantom_dedup_redis.incr(f"phantom:daily:offhours_count:{day}:{strategy}")
                                except Exception:
                                    pass

                            # Ensure per-strategy hourly budget maps
                            if 'phantom_budget' not in shared or not isinstance(shared['phantom_budget'], dict):
                                shared['phantom_budget'] = {}
                            pb_budget_map = shared['phantom_budget'].setdefault('trend', {})
                            mr_budget_map = shared['phantom_budget'].setdefault('mr', {})
                            # Configured per-strategy hourly budgets (defaults)
                            hb = cfg.get('phantom', {}).get('hourly_symbol_budget', {}) or {}
                            pb_limit = int(hb.get('trend', hb.get('pullback', 3)))
                            mr_limit = int(hb.get('mr', 3))
                            now_ts = time.time()
                            one_hour_ago = now_ts - 3600
                            # Clean old entries
                            pb_budget = [ts for ts in pb_budget_map.get(sym, []) if ts > one_hour_ago]
                            mr_budget = [ts for ts in mr_budget_map.get(sym, []) if ts > one_hour_ago]
                            pb_budget_map[sym] = pb_budget
                            mr_budget_map[sym] = mr_budget
                            shared['phantom_budget'] = {'trend': pb_budget_map, 'mr': mr_budget_map}

                            # Per-symbol phantom cooldown (8 candles)
                            phantom_cd_map = shared['phantom_cooldown']
                            last_idx = phantom_cd_map.get(sym, {}).get('last_idx', -999999)
                            if candles_processed - last_idx < 8:
                                continue

                            # Load cluster id for quotas
                            try:
                                from symbol_clustering import load_symbol_clusters
                                clusters_map = load_symbol_clusters()
                                cluster_id = int(clusters_map.get(sym, 3))
                            except Exception:
                                cluster_id = 3
                            # Per-strategy daily caps
                            mr_caps_reached = _daily_capped('mr', sym, cluster_id)
                            pb_caps_reached = _daily_capped('trend', sym, cluster_id)
                            if mr_caps_reached and pb_caps_reached:
                                logger.debug(f"[{sym}] Daily caps reached for MR and Trend; skipping both")
                            # Try MR phantom (exploration gating) or force-execute via MR Promotion
                            try:
                                sig_mr = detect_signal_mean_reversion(df.copy(), settings, sym)
                                # Force execution when MR recent WR ≥ promote threshold (bypass all guards)
                                promotion_forced = False
                                try:
                                    prom_cfg = (self.config.get('mr', {}) or {}).get('promotion', {})
                                    promote_wr = float(prom_cfg.get('promote_wr', 50.0))
                                    mr_stats = enhanced_mr_scorer.get_enhanced_stats() if enhanced_mr_scorer else {}
                                    recent_wr = float(mr_stats.get('recent_win_rate', 0.0))
                                    if sig_mr and recent_wr >= promote_wr:
                                        promotion_forced = True
                                except Exception:
                                    promotion_forced = False

                                # If promotion is forced and we have a signal, execute immediately
                                if sig_mr and promotion_forced:
                                    promotion_attempted = True
                                    try:
                                        if self.tg:
                                            try:
                                                await self.tg.send_message(f"🌀 MR Promotion: Force executing {sym} {sig_mr.side.upper()} (WR ≥ {promote_wr:.0f}%)")
                                            except Exception:
                                                pass
                                        # Prevent duplicate position on same symbol
                                        if sym in book.positions:
                                            logger.info(f"[{sym}] Promotion forced, but position already open. Skipping duplicate execution.")
                                            raise Exception("position_exists")

                                        # Get symbol metadata and round TP/SL
                                        m = meta_for(sym, shared["meta"])
                                        from position_mgr import round_step
                                        tick_size = m.get("tick_size", 0.000001)
                                        original_tp = sig_mr.tp; original_sl = sig_mr.sl
                                        sig_mr.tp = round_step(sig_mr.tp, tick_size)
                                        sig_mr.sl = round_step(sig_mr.sl, tick_size)
                                        if original_tp != sig_mr.tp or original_sl != sig_mr.sl:
                                            logger.info(f"[{sym}] Rounded TP/SL to tick size {tick_size}. TP: {original_tp:.6f} -> {sig_mr.tp:.6f}, SL: {original_sl:.6f} -> {sig_mr.sl:.6f}")

                                        # Balance and sizing
                                        current_balance = bybit.get_balance()
                                        if current_balance:
                                            sizer.account_balance = current_balance
                                            shared["last_balance"] = current_balance
                                        risk_amount = sizer.account_balance * (risk.risk_percent / 100.0) if risk.use_percent_risk and sizer.account_balance else risk.risk_usd
                                        qty = sizer.qty_for(sig_mr.entry, sig_mr.sl, m.get("qty_step",0.001), m.get("min_qty",0.001), ml_score=0.0)
                                        if qty <= 0:
                                            logger.info(f"[{sym}] Promotion forced, but qty calc invalid -> skip")
                                            raise Exception("invalid_qty")

                                        # Stop-loss sanity (prevent exchange error)
                                        current_price = df['close'].iloc[-1]
                                        if (sig_mr.side == "long" and sig_mr.sl >= current_price) or (sig_mr.side == "short" and sig_mr.sl <= current_price):
                                            logger.warning(f"[{sym}] Promotion forced, but SL invalid relative to current price -> skip")
                                            raise Exception("invalid_sl")

                                        # Set leverage and place market order
                                        max_lev = int(m.get("max_leverage", 10))
                                        bybit.set_leverage(sym, max_lev)
                                        side = "Buy" if sig_mr.side == "long" else "Sell"
                                        logger.info(f"[{sym}] MR Promotion placing {side} order for {qty} units")
                                        order_result = bybit.place_market(sym, side, qty, reduce_only=False)

                                        # Adjust TP to actual entry
                                        actual_entry = sig_mr.entry
                                        try:
                                            await asyncio.sleep(0.5)
                                            position = bybit.get_position(sym)
                                            if position and position.get("avgPrice"):
                                                actual_entry = float(position["avgPrice"])
                                                if actual_entry != sig_mr.entry:
                                                    fee_adjustment = 1.00165
                                                    risk_distance = abs(actual_entry - sig_mr.sl)
                                                    if sig_mr.side == "long":
                                                        sig_mr.tp = actual_entry + (settings.rr * risk_distance * fee_adjustment)
                                                    else:
                                                        sig_mr.tp = actual_entry - (settings.rr * risk_distance * fee_adjustment)
                                                    logger.info(f"[{sym}] MR Promotion TP adjusted for actual entry: {sig_mr.tp:.4f}")
                                        except Exception:
                                            pass

                                        # Set TP/SL with exchange read-back (parity with general exec path)
                                        try:
                                            bybit.set_tpsl(sym, take_profit=sig_mr.tp, stop_loss=sig_mr.sl, qty=qty)
                                            # Give the exchange a brief moment to register TP/SL, then read back
                                            try:
                                                await asyncio.sleep(0.4)
                                                pos_back = bybit.get_position(sym)
                                                if isinstance(pos_back, dict):
                                                    tp_val = pos_back.get('takeProfit')
                                                    sl_val = pos_back.get('stopLoss')
                                                    old_tp, old_sl = sig_mr.tp, sig_mr.sl
                                                    if tp_val not in (None, '', '0'):
                                                        sig_mr.tp = float(tp_val)
                                                    if sl_val not in (None, '', '0'):
                                                        sig_mr.sl = float(sl_val)
                                                    if (old_tp != sig_mr.tp) or (old_sl != sig_mr.sl):
                                                        logger.info(f"[{sym}] MR Promotion TP/SL read-back: TP {old_tp:.4f}→{sig_mr.tp:.4f}, SL {old_sl:.4f}→{sig_mr.sl:.4f}")
                                            except Exception:
                                                pass
                                            logger.info(f"[{sym}] MR Promotion TP/SL set successfully")
                                        except Exception as tpsl_error:
                                            logger.critical(f"[{sym}] CRITICAL: MR Promotion failed to set TP/SL: {tpsl_error}")
                                            logger.critical(f"[{sym}] Attempting emergency position closure to prevent unprotected position")
                                            # Attempt to close the unprotected position immediately
                                            try:
                                                emergency_side = "Sell" if sig_mr.side == "long" else "Buy"
                                                close_result = bybit.place_market(sym, emergency_side, qty, reduce_only=True)
                                                logger.warning(f"[{sym}] Emergency position closure executed: {close_result}")
                                                if self.tg:
                                                    await self.tg.send_message(f"🚨 EMERGENCY CLOSURE: {sym} position closed (MR Promotion) due to TP/SL failure: {str(tpsl_error)[:100]}")
                                            except Exception as close_error:
                                                logger.critical(f"[{sym}] FAILED TO CLOSE UNPROTECTED POSITION (MR Promotion): {close_error}")
                                                if self.tg:
                                                    await self.tg.send_message(f"🆘 CRITICAL: {sym} position UNPROTECTED! MR Promotion SL/TP failed: {str(tpsl_error)[:100]}")
                                                    await self.tg.send_message(f"🛑 Bot halted due to unprotected position. Use /resume to restart after manual review.")
                                            # Prevent adding to book if unprotected
                                            raise Exception(f"Failed to set TP/SL for {sym} (MR Promotion): {tpsl_error}")

                                        # Update book and notify standard open message
                                        book.positions[sym] = Position(
                                            sig_mr.side,
                                            qty,
                                            entry=actual_entry,
                                            sl=sig_mr.sl,
                                            tp=sig_mr.tp,
                                            entry_time=datetime.now(),
                                            strategy_name='enhanced_mr'
                                        )
                                        # Mark promotion on meta for accounting
                                        try:
                                            if not sig_mr.meta:
                                                sig_mr.meta = {}
                                            sig_mr.meta['promotion_forced'] = True
                                        except Exception:
                                            pass

                                        # Standard execution notification
                                        if self.tg:
                                            try:
                                                emoji = '📈'
                                                strategy_label = 'Enhanced Mr'
                                                msg = (
                                                    f"{emoji} *{sym} {sig_mr.side.upper()}* ({strategy_label})\n\n"
                                                    f"Entry: {actual_entry:.4f}\n"
                                                    f"Stop Loss: {sig_mr.sl:.4f}\n"
                                                    f"Take Profit: {sig_mr.tp:.4f}\n"
                                                    f"Quantity: {qty}\n"
                                                    f"Risk: {risk.risk_percent if risk.use_percent_risk else risk.risk_usd}{'%' if risk.use_percent_risk else ''} (${risk_amount:.2f})\n"
                                                    f"Promotion: FORCED (WR ≥ {promote_wr:.0f}%)\n"
                                                    f"Reason: Mean Reversion (Promotion)"
                                                )
                                                await self.tg.send_message(msg)
                                            except Exception:
                                                pass
                                        # Skip the rest of phantom sampling for this symbol
                                        continue
                                    except Exception as e:
                                        # Announce forced execution failure and skip phantom record to avoid confusion
                                        logger.warning(f"[{sym}] MR Promotion forced execution failed: {e}")
                                        if self.tg:
                                            try:
                                                await self.tg.send_message(f"🛑 MR Promotion: Failed to execute {sym} {sig_mr.side.upper()} — {str(e)[:120]}")
                                            except Exception:
                                                pass
                                        continue

                                # Regular phantom-only sampling path (when not promotion-forced)
                                mr_remaining = mr_limit - len(mr_budget)
                                if (not mr_caps_reached) and mr_remaining > 0 and sig_mr and _not_duplicate('mr', sym, sig_mr):
                                    ef = sig_mr.meta.get('mr_features', {}).copy() if sig_mr.meta else {}
                                    ef['routing'] = 'none'
                                    # Exploration gate (phantom-only) — high-vol allowed if configured
                                    meets_gate = True
                                    reasons = []
                                    try:
                                        if exploration_enabled and mr_explore.get('enabled', True):
                                            base_rc_min = float(mr_explore.get('rc_min', 0.70))
                                            base_touches_min = int(mr_explore.get('touches_min', 6))
                                            base_dist_mid_min = float(mr_explore.get('dist_mid_atr_min', 0.60))
                                            base_rev_min = float(mr_explore.get('rev_candle_atr_min', 1.0))
                                            allow_high = bool(mr_explore.get('allow_volatility_high', True))
                                            # Apply adaptive relax (phantom-only)
                                            rc_min = base_rc_min; touches_min = base_touches_min
                                            dist_mid_min = base_dist_mid_min; rev_min = base_rev_min
                                            try:
                                                if self.flow_controller and self.flow_controller.enabled:
                                                    adj = self.flow_controller.adjust_mr(rc_min, touches_min, dist_mid_min, rev_min)
                                                    rc_min = adj['rc_min']; touches_min = adj['touches_min']
                                                    dist_mid_min = adj['dist_mid_min']; rev_min = adj['rev_min']
                                            except Exception:
                                                pass
                                            rc = float(ef.get('range_confidence', 0))
                                            touches = float(ef.get('touch_count_sr', 0))
                                            dist_mid = float(ef.get('distance_from_midpoint_atr', 0))
                                            rev_atr = float(ef.get('reversal_candle_size_atr', 0))
                                            vol_reg = getattr(regime_analysis, 'volatility_level', 'normal')
                                            if rc < rc_min:
                                                meets_gate = False; reasons.append(f"rc {rc:.2f}<{rc_min}")
                                            if touches < touches_min:
                                                meets_gate = False; reasons.append(f"touches {touches:.0f}<{touches_min}")
                                            if dist_mid < dist_mid_min:
                                                meets_gate = False; reasons.append(f"edge {dist_mid:.2f}<{dist_mid_min}")
                                            if rev_atr < rev_min:
                                                meets_gate = False; reasons.append(f"rev {rev_atr:.2f}<{rev_min}")
                                            if vol_reg == 'high' and not allow_high:
                                                meets_gate = False; reasons.append("vol=high disallowed")
                                            # Stronger reversal confirmation for phantom-only MR
                                            try:
                                                o_last = df['open'].iloc[-1]
                                                c_last = df['close'].iloc[-1]
                                                h_last = df['high'].iloc[-1]
                                                l_last = df['low'].iloc[-1]
                                                rng = max(1e-9, h_last - l_last)
                                                body_ratio = abs(c_last - o_last) / rng
                                                if sig_mr.side == 'long':
                                                    if not (c_last > o_last and body_ratio >= 0.4):
                                                        meets_gate = False; reasons.append(f"weak bull reversal body={body_ratio:.2f}")
                                                else:
                                                    if not (c_last < o_last and body_ratio >= 0.4):
                                                        meets_gate = False; reasons.append(f"weak bear reversal body={body_ratio:.2f}")
                                            except Exception:
                                                pass
                                    except Exception:
                                        pass
                                    if not meets_gate:
                                        logger.info(f"[{sym}] 👻 MR explore skip: {', '.join(reasons) if reasons else 'gate fail'}")
                                    else:
                                        logger.info(f"[{sym}] 👻 Phantom-only (MR none): {sig_mr.side.upper()} @ {sig_mr.entry:.4f}")
                                        if mr_phantom_tracker:
                                            ok_ph_mr2, why_mr2 = self._phantom_mr_regime_ok(sym, df, regime_analysis)
                                            if ok_ph_mr2:
                                                mr_phantom_tracker.record_mr_signal(sym, sig_mr.__dict__, 0.0, False, {}, ef)
                                            else:
                                                logger.info(f"[{sym}] 🛑 MR phantom (none route) dropped by regime gate ({why_mr2})")
                                            # Count accepted MR phantom for flow pacing
                                            try:
                                                if self.flow_controller:
                                                    self.flow_controller.increment_accepted('mr', 1)
                                            except Exception:
                                                pass
                                            _increment_daily('mr', sym, cluster_id, 'none')
                                        # Consume MR hourly budget
                                        mr_budget.append(now_ts)
                                        mr_budget_map[sym] = mr_budget
                            except Exception:
                                pass

                            # Try Trend phantom (exploration gating)
                            if not pb_caps_reached:
                                try:
                                    # Gate on minimum history
                                    if len(df) < getattr(self, '_trend_hist_min', 80):
                                        if not self._trend_hist_warned.get(sym, False):
                                            logger.info(f"[{sym}] 🔵 Trend waiting for history: have {len(df)}/{getattr(self, '_trend_hist_min', 80)} 15m bars")
                                            self._trend_hist_warned[sym] = True
                                        sig_tr = None
                                    else:
                                        sig_tr = detect_trend_signal(df.copy(), trend_settings, sym)
                                    pb_remaining = pb_limit - len(pb_budget)
                                    if pb_remaining > 0 and sig_tr and _not_duplicate('trend', sym, sig_tr):
                                        # Build basic trend features
                                        cl = df['close']; price = float(cl.iloc[-1])
                                        ys = cl.tail(20).values if len(cl) >= 20 else cl.values
                                        try:
                                            slope = np.polyfit(np.arange(len(ys)), ys, 1)[0]
                                        except Exception:
                                            slope = 0.0
                                        trend_slope_pct = float((slope / price) * 100.0) if price else 0.0
                                        ema20 = cl.ewm(span=20, adjust=False).mean().iloc[-1]
                                        ema50 = cl.ewm(span=50, adjust=False).mean().iloc[-1] if len(cl) >= 50 else ema20
                                        ema_stack_score = 100.0 if (price > ema20 > ema50 or price < ema20 < ema50) else 50.0 if (ema20 != ema50) else 0.0
                                        rng_today = float(df['high'].iloc[-1] - df['low'].iloc[-1])
                                        med_range = float((df['high'] - df['low']).rolling(20).median().iloc[-1]) if len(df) >= 20 else rng_today
                                        range_expansion = float(rng_today / max(1e-9, med_range))
                                        prev = cl.shift(); trarr = np.maximum(df['high'] - df['low'], np.maximum((df['high'] - prev).abs(), (df['low'] - prev).abs()))
                                        atr = float(trarr.rolling(14).mean().iloc[-1]) if len(trarr) >= 14 else float(trarr.iloc[-1])
                                        atr_pct = float((atr / max(1e-9, price)) * 100.0) if price else 0.0
                                        trend_features = {
                                            'trend_slope_pct': trend_slope_pct,
                                            'ema_stack_score': ema_stack_score,
                                            'atr_pct': atr_pct,
                                            'range_expansion': range_expansion,
                                            'breakout_dist_atr': float(getattr(sig_tr, 'meta', {}).get('breakout_dist_atr', 0.0) if getattr(sig_tr, 'meta', None) else 0.0),
                                            'close_vs_ema20_pct': float(((price - ema20) / max(1e-9, ema20)) * 100.0) if ema20 else 0.0,
                                            'bb_width_pct': 0.0,
                                            'session': 'us',
                                            'symbol_cluster': 3,
                                            'volatility_regime': getattr(regime_analysis, 'volatility_level', 'normal')
                                        }
                                        # Trend exploration gate (apply adaptive relax)
                                        meets_gate = True
                                        reasons = []
                                        try:
                                            tr_explore = (cfg.get('trend', {}) or {}).get('explore', {})
                                            if exploration_enabled and tr_explore.get('enabled', True):
                                                slope_min = float(tr_explore.get('slope_min', 3.0))
                                                ema_min = float(tr_explore.get('ema_stack_min', 40.0))
                                                br_min = float(tr_explore.get('breakout_dist_atr_min', 0.1))
                                                # Apply FlowController relax if enabled
                                                try:
                                                    if self.flow_controller and self.flow_controller.enabled:
                                                        adj = self.flow_controller.adjust_trend(slope_min, ema_min, br_min)
                                                        slope_min = adj.get('slope_min', slope_min)
                                                        ema_min = adj.get('ema_min', ema_min)
                                                        br_min = adj.get('breakout_min', br_min)
                                                except Exception:
                                                    pass
                                                allow_high = bool(tr_explore.get('allow_volatility_high', True))
                                                vol_reg = getattr(regime_analysis, 'volatility_level', 'normal')
                                                if abs(trend_features['trend_slope_pct']) < slope_min:
                                                    meets_gate = False; reasons.append(f"slope {trend_features['trend_slope_pct']:.1f}<{slope_min}")
                                                if trend_features['ema_stack_score'] < ema_min:
                                                    meets_gate = False; reasons.append(f"ema {trend_features['ema_stack_score']:.0f}<{ema_min}")
                                                if trend_features['breakout_dist_atr'] < br_min:
                                                    meets_gate = False; reasons.append(f"br {trend_features['breakout_dist_atr']:.2f}<{br_min}")
                                                if vol_reg == 'high' and not allow_high:
                                                    meets_gate = False; reasons.append("vol=high disallowed")
                                        except Exception:
                                            pass
                                        if not meets_gate:
                                            # Log unified decision lines for Trend in NONE routing
                                            try:
                                                logger.info(f"[{sym}] 🔵 Trend decision context: dedup=True hourly_remaining={pb_remaining} daily_ok={not pb_caps_reached}")
                                                logger.info(f"[{sym}] 🧮 Trend decision final: phantom (reason=explore_gate {', '.join(reasons) if reasons else 'gate fail'})")
                                            except Exception:
                                                pass
                                            logger.info(f"[{sym}] 👻 Trend explore skip: {', '.join(reasons) if reasons else 'gate fail'}")
                                        else:
                                            try:
                                                logger.info(f"[{sym}] 🔵 Trend decision context: dedup=True hourly_remaining={pb_remaining} daily_ok={not pb_caps_reached}")
                                            except Exception:
                                                pass
                                            ok_ph, why = self._phantom_trend_regime_ok(sym, df, regime_analysis)
                                            if ok_ph:
                                                try:
                                                    logger.info(f"[{sym}] 🧮 Trend decision final: phantom (reason=routing=none explore)")
                                                except Exception:
                                                    pass
                                                logger.info(f"[{sym}] 👻 Phantom-only (Trend none): {sig_tr.side.upper()} @ {sig_tr.entry:.4f}")
                                                if phantom_tracker:
                                                    phantom_tracker.record_signal(
                                                        symbol=sym,
                                                        signal={'side': sig_tr.side, 'entry': sig_tr.entry, 'sl': sig_tr.sl, 'tp': sig_tr.tp},
                                                        ml_score=0.0,
                                                        was_executed=False,
                                                        features=trend_features,
                                                        strategy_name='trend_pullback'
                                                    )
                                            else:
                                                logger.info(f"[{sym}] 🛑 Trend phantom (none route) dropped by regime gate ({why})")
                                                try:
                                                    if self.flow_controller:
                                                        self.flow_controller.increment_accepted('trend', 1)
                                                except Exception:
                                                    pass
                                                _increment_daily('trend', sym, cluster_id, 'none')
                                            pb_budget.append(now_ts)
                                            pb_budget_map[sym] = pb_budget
                                except Exception:
                                    pass
                            # Before continuing, ensure Scalp fallback runs if 3m stream is unavailable/stale
                            try:
                                # Skip legacy Scalp fallback when disabled (prefer independent 3m stream only)
                                if not bool(((self.config.get('modes', {}) or {}).get('disable_scalp_fallback', True))):
                                    await self._maybe_run_scalp_fallback(sym, df, regime_analysis, cluster_id)
                            except Exception:
                                pass
                            # After phantom-only sampling + optional Scalp fallback, continue to next symbol
                            continue

                            # Scalp phantom (Phase 0)
                            # If scalp.independent=true, this router-driven scalp is disabled to avoid duplication
                            if use_scalp and SCALP_AVAILABLE and detect_scalp_signal is not None:
                                try:
                                    s_cfg = self.config.get('scalp', {}) if hasattr(self, 'config') else {}
                                    if bool(s_cfg.get('independent', False)):
                                        # Skip here; independent 3m stream handles Scalp
                                        pass
                                    else:
                                        # Prefer 3m frames when available
                                        df3 = self.frames_3m.get(sym)
                                        if df3 is not None and not df3.empty and len(df3) >= 120:
                                            df_for_scalp = df3
                                            logger.info(f"[{sym}] 🩳 Using 3m frames for scalp ({len(df3)} bars)")
                                        else:
                                            df_for_scalp = df
                                            if df3 is None or df3.empty:
                                                logger.info(f"[{sym}] 🩳 Scalp using main tf: 3m unavailable")
                                            else:
                                                logger.info(f"[{sym}] 🩳 Scalp using main tf: 3m sparse ({len(df3)} bars)")

                                        sc_sig = detect_scalp_signal(df_for_scalp.copy(), ScalpSettings(), sym)
                                        if sc_sig and _not_duplicate(sym, sc_sig):
                                            sc_meta = getattr(sc_sig, 'meta', {}) or {}
                                            sc_feats = self._build_scalp_features(df_for_scalp, sc_meta, regime_analysis.volatility_level, cluster_id)
                                            try:
                                                comp = self._get_htf_metrics(sym, self.frames.get(sym))
                                                sc_feats['ts15'] = float(comp.get('ts15', 0.0)); sc_feats['ts60'] = float(comp.get('ts60', 0.0))
                                                sc_feats['rc15'] = float(comp.get('rc15', 0.0)); sc_feats['rc60'] = float(comp.get('rc60', 0.0))
                                            except Exception:
                                                pass
                                            sc_feats['routing'] = 'none'
                                            logger.info(f"[{sym}] 👻 Phantom-only (Scalp none): {sc_sig.side.upper()} @ {sc_sig.entry:.4f}")
                                            try:
                                                from scalp_phantom_tracker import get_scalp_phantom_tracker
                                                scpt = get_scalp_phantom_tracker()
                                                # Compute ML for router-driven scalp phantom
                                                ml_s = 0.0
                                                try:
                                                    _scorer = get_scalp_scorer() if get_scalp_scorer is not None else None
                                                    if _scorer:
                                                        ml_s, _ = _scorer.score_signal(
                                                            {'side': sc_sig.side, 'entry': sc_sig.entry, 'sl': sc_sig.sl, 'tp': sc_sig.tp},
                                                            sc_feats
                                                        )
                                                    else:
                                                        ml_s = 0.0
                                                except Exception:
                                                    ml_s = 0.0
                                                scpt.record_scalp_signal(
                                                    sym,
                                                    {'side': sc_sig.side, 'entry': sc_sig.entry, 'sl': sc_sig.sl, 'tp': sc_sig.tp},
                                                    float(ml_s or 0.0),
                                                    False,
                                                    sc_feats
                                                )
                                                # Count accepted Scalp phantom for flow pacing
                                                try:
                                                    if hasattr(self, 'flow_controller') and self.flow_controller:
                                                        self.flow_controller.increment_accepted('scalp', 1)
                                                except Exception:
                                                    pass
                                            except Exception:
                                                pass
                                            # Scalp is exempt from daily caps; do not increment daily counters
                                        else:
                                            logger.info(f"[{sym}] 🩳 No Scalp Signal (filters not met)")
                                except Exception as e:
                                    logger.debug(f"[{sym}] Scalp detection error: {e}")
                            else:
                                if not use_scalp:
                                    logger.info("🩳 Scalp disabled by config")
                                elif not SCALP_AVAILABLE or detect_scalp_signal is None:
                                    logger.info("🩳 Scalp modules not available")

                            # Virtual snapshots around threshold if enabled
                            # Strategy-specific virtual snapshots (exploration)
                            if recorded > 0:
                                try:
                                    # Use available scorer thresholds; fallback to 75
                                    pb_thr = getattr(ml_scorer, 'min_score', 75) if ml_scorer else 75
                                    mr_thr = getattr(enhanced_mr_scorer, 'min_score', 75) if enhanced_mr_scorer else 75
                                    # Trend snapshots (none)
                                    # MR snapshots
                                    if exploration_enabled and mr_explore.get('enable_virtual_snapshots', False):
                                        delta_mr = int(mr_explore.get('snapshots_delta', phantom_virtual_delta))
                                        for delta in (-delta_mr, +delta_mr):
                                            vthr_mr = max(60, min(90, mr_thr + delta))
                                            if sig_mr and mr_phantom_tracker and not _daily_capped(sym, cluster_id):
                                                ef2 = ef.copy()
                                                ef2['routing'] = 'none'
                                                ef2['virtual_threshold'] = vthr_mr
                                                mr_phantom_tracker.record_mr_signal(sym, sig_mr.__dict__, 0.0, False, {}, ef2)
                                                try:
                                                    if hasattr(self, 'flow_controller') and self.flow_controller and self.flow_controller.enabled:
                                                        self.flow_controller.increment_accepted('mr', 1)
                                                except Exception:
                                                    pass
                                                _increment_daily(sym, cluster_id, 'none')
                                except Exception:
                                    pass

                            shared['phantom_budget'][sym] = budget
                            phantom_cd_map[sym] = { 'last_idx': candles_processed }
                            shared['phantom_cooldown'] = phantom_cd_map
                            # Done with phantom-only; continue to next symbol
                            continue

                    elif use_regime_switching:
                        # Original regime switching logic (fallback)
                        from market_regime import get_market_regime
                        current_regime = get_market_regime(df)
                        logger.debug(f"[{sym}] Basic regime: {current_regime}")

                        if current_regime == "Ranging":
                            sig = detect_signal_mean_reversion(df.copy(), settings, sym)
                            selected_strategy = "mean_reversion"
                        else:
                            sig = detect_trend_signal(df.copy(), trend_settings, sym)
                            selected_strategy = "trend_pullback"

                    else:
                        # Default trend strategy
                        sig = detect_trend_signal(df.copy(), trend_settings, sym)
                        selected_strategy = "trend_pullback"
                    # --- END PARALLEL STRATEGY ROUTING ---

                    if sig is None:
                        # Don't log every non-signal to reduce log spam
                        continue
                
                    logger.info(f"[{sym}] Signal detected: {sig.side} @ {sig.entry:.4f}")
                    signals_detected += 1
                
                    # Apply ML scoring and phantom tracking using selected system
                    ml_score = 0
                    ml_reason = "No ML Scoring"
                    should_take_trade = True

                    if selected_ml_scorer is not None and selected_phantom_tracker is not None:
                        try:
                            # Different feature extraction based on strategy
                            if selected_strategy == "enhanced_mr":
                                # Use enhanced MR features
                                # Normalize meta and extract MR features safely
                                try:
                                    sig.meta = getattr(sig, 'meta', {}) or {}
                                except Exception:
                                    sig.meta = {}
                                enhanced_features = (sig.meta or {}).get('mr_features', {}) or {}
                                logger.info(f"🧠 [{sym}] ENHANCED MR ML ANALYSIS:")
                                logger.info(f"   📊 Features: {len(enhanced_features)} basic MR features")

                                # Pre-ML guardrails for MR to boost live WR (phantom will still track rejects)
                                policy_reject_reason = None
                                try:
                                    rc = float(enhanced_features.get('range_confidence', 0))
                                    touches = float(enhanced_features.get('touch_count_sr', 0))
                                    dist_mid = float(enhanced_features.get('distance_from_midpoint_atr', 0))
                                    rev_atr = float(enhanced_features.get('reversal_candle_size_atr', 0))
                                    vol_reg = enhanced_features.get('volatility_regime', 'normal')

                                    if rc < 0.70:
                                        policy_reject_reason = f"Low range_confidence {rc:.2f}"
                                    elif touches < 4:
                                        policy_reject_reason = f"Insufficient S/R touches {touches:.0f}"
                                    elif dist_mid < 0.80:
                                        policy_reject_reason = f"Edge distance too small {dist_mid:.2f} ATR"
                                    elif isinstance(vol_reg, str) and vol_reg == 'high':
                                        policy_reject_reason = "High volatility regime"
                                    elif rev_atr < 1.10:
                                        policy_reject_reason = f"Weak reversal candle {rev_atr:.2f} ATR"
                                except Exception:
                                    # If any parsing issues, do not enforce guardrail
                                    policy_reject_reason = None

                                if policy_reject_reason is not None:
                                    logger.info(f"   ⛔ Policy Reject (MR): {policy_reject_reason}")
                                    # Record as phantom so learning continues
                                    try:
                                        if self.tg:
                                            await self.tg.send_message(
                                                f"🚫 *Policy Reject* {sym} {sig.side.upper()} (MR)\n{policy_reject_reason} — tracking via phantom"
                                            )
                                    except Exception:
                                        pass
                                    try:
                                        shared.get("telemetry", {})["policy_rejects"] += 1
                                    except Exception:
                                        pass
                                    policy_features = {'policy_reject': 1, 'policy_reason': policy_reject_reason}
                                    selected_phantom_tracker.record_mr_signal(
                                        sym, sig.__dict__, 0.0, False, policy_features, enhanced_features
                                    )
                                    continue

                                # Score using Enhanced MR ML system
                                ml_score, ml_reason = selected_ml_scorer.score_signal(sig.__dict__, enhanced_features, df)

                                # Detailed ML decision logging
                                threshold = selected_ml_scorer.min_score
                                # Cluster-aware threshold bump for high-vol symbols if cluster id is present
                                try:
                                    cl = int(enhanced_features.get('symbol_cluster', 0))
                                    if cl == 3:  # meme/high-vol cluster
                                        threshold = min(85, threshold + 3)
                                except Exception:
                                    pass
                                should_take_trade = ml_score >= threshold
                                promotion_forced = False

                                logger.info(f"   🎯 ML Score: {ml_score:.1f} / {threshold:.0f} threshold")
                                logger.info(f"   🔍 Analysis: {ml_reason}")
                                logger.info(f"   📈 Key Factors:")

                                # Log top contributing factors if available
                                try:
                                    top_features = sorted(enhanced_features.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
                                    for i, (feature, value) in enumerate(top_features):
                                        logger.info(f"      {i+1}. {feature}: {value:.3f}")
                                except:
                                    logger.info(f"      Range quality, oscillator signals, market microstructure")

                                if should_take_trade:
                                    logger.info(f"   ✅ DECISION: EXECUTE TRADE - ML confidence above {threshold}")
                                else:
                                    logger.info(f"   ❌ DECISION: REJECT TRADE - ML score {ml_score:.1f} below threshold {threshold}")
                                    logger.info(f"   💡 Rejection reason: {ml_reason}")

                                # MR Promotion override: execute despite ML below threshold (bypass all guards when WR ≥ promote_wr)
                                try:
                                    prom_cfg = (self.config.get('mr', {}) or {}).get('promotion', {})
                                    if not should_take_trade and bool(prom_cfg.get('enabled', False)):
                                        promote_wr = float(prom_cfg.get('promote_wr', 50.0))
                                        mr_stats = enhanced_mr_scorer.get_enhanced_stats() if enhanced_mr_scorer else {}
                                        recent_wr = float(mr_stats.get('recent_win_rate', 0.0))
                                        if recent_wr >= promote_wr:
                                            should_take_trade = True
                                            promotion_forced = True
                                            # Mark on signal meta for post-exec accounting
                                            try:
                                                if not sig.meta:
                                                    sig.meta = {}
                                                sig.meta['promotion_forced'] = True
                                            except Exception:
                                                pass
                                            logger.info(f"   🚀 MR Promotion override: executing despite ML {ml_score:.1f} < {threshold:.0f} (WR {recent_wr:.1f}% ≥ {promote_wr:.0f}%)")
                                            if self.tg:
                                                try:
                                                    await self.tg.send_message(f"🌀 MR Promotion: Executing {sym} {sig.side.upper()} despite ML {ml_score:.1f} < {threshold:.0f}")
                                                except Exception:
                                                    pass
                                except Exception:
                                    pass

                                # Record in MR phantom tracker BEFORE continue (for both executed and rejected)
                                logger.info(f"[{sym}] 📊 PHANTOM ROUTING: MR phantom tracker recording (executed={should_take_trade})")
                                selected_phantom_tracker.record_mr_signal(
                                    sym, sig.__dict__, ml_score, should_take_trade, {}, enhanced_features
                                )

                                # Skip trade execution if ML score below threshold (but phantom is recorded)
                                if not should_take_trade:
                                    try:
                                        shared.get("telemetry", {})["ml_rejects"] += 1
                                        logger.debug(f"Telemetry: ML rejects so far = {shared['telemetry']['ml_rejects']}")
                                    except Exception:
                                        pass
                                    # Suppress Telegram notifications for MR ML rejects (phantom-only)
                                    continue

                            else:
                                # Use trend features (fallback system)
                                from strategy_pullback_ml_learning import calculate_ml_features, BreakoutState  # legacy import; not used if enhanced

                                logger.info(f"🧠 [{sym}] TREND ML ANALYSIS:")

                                # Get or create state for this symbol
                                if sym not in ml_breakout_states:
                                    ml_breakout_states[sym] = BreakoutState()
                                state = ml_breakout_states[sym]

                                # Calculate retracement from entry price
                                if sig.side == "long":
                                    retracement = sig.entry  # Use entry as proxy for retracement level
                                else:
                                    retracement = sig.entry

                                # Calculate Trend features (fallback)
                                cl = df['close']; price = float(cl.iloc[-1])
                                ys = cl.tail(20).values if len(cl) >= 20 else cl.values
                                try:
                                    slope = np.polyfit(np.arange(len(ys)), ys, 1)[0]
                                except Exception:
                                    slope = 0.0
                                trend_slope_pct = float((slope / price) * 100.0) if price else 0.0
                                ema20 = cl.ewm(span=20, adjust=False).mean().iloc[-1]
                                ema50 = cl.ewm(span=50, adjust=False).mean().iloc[-1] if len(cl) >= 50 else ema20
                                ema_stack_score = 100.0 if (price > ema20 > ema50 or price < ema20 < ema50) else 50.0 if (ema20 != ema50) else 0.0
                                rng_today = float(df['high'].iloc[-1] - df['low'].iloc[-1])
                                med_range = float((df['high'] - df['low']).rolling(20).median().iloc[-1]) if len(df) >= 20 else rng_today
                                range_expansion = float(rng_today / max(1e-9, med_range))
                                prev = cl.shift(); trarr = np.maximum(df['high'] - df['low'], np.maximum((df['high'] - prev).abs(), (df['low'] - prev).abs()))
                                atr = float(trarr.rolling(14).mean().iloc[-1]) if len(trarr) >= 14 else float(trarr.iloc[-1])
                                atr_pct = float((atr / max(1e-9, price)) * 100.0) if price else 0.0
                                close_vs_ema20_pct = float(((price - ema20) / max(1e-9, ema20)) * 100.0) if ema20 else 0.0
                                basic_features = {
                                    'trend_slope_pct': trend_slope_pct,
                                    'ema_stack_score': ema_stack_score,
                                    'atr_pct': atr_pct,
                                    'range_expansion': range_expansion,
                                    'breakout_dist_atr': float(getattr(sig, 'meta', {}).get('breakout_dist_atr', 0.0) if getattr(sig, 'meta', None) else 0.0),
                                    'close_vs_ema20_pct': close_vs_ema20_pct,
                                    'bb_width_pct': 0.0,
                                    'session': 'us',
                                    'symbol_cluster': 3,
                                    'volatility_regime': getattr(regime_analysis, 'volatility_level', 'normal')
                                }

                                # Score using Trend ML system
                                # Pre-ML guardrails for Trend to boost live WR (phantom will still track rejects)
                                pb_policy_reason = None
                                try:
                                    slope_min = 3.0
                                    ema_min = 40.0
                                    br_min = 0.10
                                    vol_reg = getattr(regime_analysis, 'volatility_level', 'normal')
                                    if abs(basic_features.get('trend_slope_pct', 0.0)) < slope_min:
                                        pb_policy_reason = f"Weak slope {basic_features.get('trend_slope_pct',0.0):.1f}% < {slope_min}%"
                                    elif basic_features.get('ema_stack_score', 0.0) < ema_min:
                                        pb_policy_reason = f"Weak EMA stack {basic_features.get('ema_stack_score',0.0):.0f} < {ema_min}"
                                    elif basic_features.get('breakout_dist_atr', 0.0) < br_min:
                                        pb_policy_reason = f"Breakout too small {basic_features.get('breakout_dist_atr',0.0):.2f} ATR < {br_min}"
                                    elif vol_reg == 'high' and getattr((cfg.get('trend', {}) or {}).get('explore', {}), 'allow_volatility_high', True) is False:
                                        pb_policy_reason = "High volatility disallowed"
                                except Exception:
                                    pb_policy_reason = None

                                if pb_policy_reason is not None:
                                    logger.info(f"   ⛔ Policy Reject (Trend): {pb_policy_reason}")
                                    # Record as phantom so learning continues
                                    try:
                                        if self.tg:
                                            await self.tg.send_message(
                                                f"🚫 *Policy Reject* {sym} {sig.side.upper()} (Trend)\n{pb_policy_reason} — tracking via phantom"
                                            )
                                    except Exception:
                                        pass
                                    try:
                                        shared.get("telemetry", {})["policy_rejects"] += 1
                                    except Exception:
                                        pass
                                    policy_features = basic_features.copy()
                                    policy_features['policy_reject'] = 1
                                    policy_features['policy_reason'] = pb_policy_reason
                                    selected_phantom_tracker.record_signal(
                                        symbol=sym,
                                        signal={'side': sig.side, 'entry': sig.entry, 'sl': sig.sl, 'tp': sig.tp},
                                        ml_score=0.0,
                                        was_executed=False,
                                        features=policy_features,
                                        strategy_name=selected_strategy
                                    )
                                    continue

                                ml_score, ml_reason = selected_ml_scorer.score_signal(
                                    {'side': sig.side, 'entry': sig.entry, 'sl': sig.sl, 'tp': sig.tp},
                                    basic_features
                                )

                                # Detailed ML decision logging
                                threshold = selected_ml_scorer.min_score
                                # Cluster-aware threshold bump
                                try:
                                    cl = int(basic_features.get('symbol_cluster', 0))
                                    if cl == 3:
                                        threshold = min(85, threshold + 3)
                                except Exception:
                                    pass
                                should_take_trade = ml_score >= threshold

                                logger.info(f"   🎯 ML Score: {ml_score:.1f} / {threshold:.0f} threshold")
                                logger.info(f"   🔍 Analysis: {ml_reason}")
                                logger.info(f"   📈 Key Factors:")

                                # Log important technical factors (null-safe)
                                try:
                                    key_factors = []
                                    ts_val = (basic_features or {}).get('trend_strength')
                                    if ts_val is not None:
                                        key_factors.append(f"Trend: {float(ts_val):.2f}")
                                    atrp_val = (basic_features or {}).get('atr_percentile')
                                    if atrp_val is not None:
                                        key_factors.append(f"Volatility: {float(atrp_val):.2f}")
                                    volr_val = (basic_features or {}).get('volume_ratio')
                                    if volr_val is not None:
                                        key_factors.append(f"Volume: {float(volr_val):.2f}")
                                    rsi_val = (basic_features or {}).get('rsi')
                                    if rsi_val is not None:
                                        key_factors.append(f"RSI: {float(rsi_val):.1f}")
                                    bbp_val = (basic_features or {}).get('bb_position')
                                    if bbp_val is not None:
                                        key_factors.append(f"BB Pos: {float(bbp_val):.2f}")

                                    for factor in key_factors[:4]:  # Top 4 factors
                                        logger.info(f"      • {factor}")
                                except Exception:
                                    logger.info(f"      Trend strength, volume, volatility, momentum indicators")

                                if should_take_trade:
                                    logger.info(f"   ✅ DECISION: EXECUTE TRADE - ML confidence above {threshold}")
                                else:
                                    logger.info(f"   ❌ DECISION: REJECT TRADE - ML score {ml_score:.1f} below threshold {threshold}")
                                    logger.info(f"   💡 Rejection reason: {ml_reason}")

                                # Record in phantom tracker BEFORE continue (for both executed and rejected)
                                logger.info(f"[{sym}] 📊 PHANTOM ROUTING: Trend phantom tracker recording (executed={should_take_trade})")
                                selected_phantom_tracker.record_signal(
                                    symbol=sym,
                                    signal={'side': sig.side, 'entry': sig.entry, 'sl': sig.sl, 'tp': sig.tp},
                                    ml_score=ml_score,
                                    was_executed=should_take_trade,
                                    features=basic_features,
                                    strategy_name=selected_strategy
                                )

                                # Skip trade execution if ML score below threshold (but phantom is recorded)
                                if not should_take_trade:
                                    try:
                                        shared.get("telemetry", {})["ml_rejects"] += 1
                                        logger.debug(f"Telemetry: ML rejects so far = {shared['telemetry']['ml_rejects']}")
                                    except Exception:
                                        pass
                                    # Suppress Telegram notifications for Trend ML rejects (phantom-only)
                                    continue

                        except Exception as e:
                            logger.warning(f"🚨 [{sym}] ML SCORING ERROR: {e}")
                            # Telemetry: count errors
                            try:
                                tel = shared.get("telemetry", {})
                                tel['ml_errors'] = tel.get('ml_errors', 0) + 1
                            except Exception:
                                pass
                            # Policy: fail-open or fail-closed
                            try:
                                fail_open = bool(self.config.get('ml', {}).get('fail_open_on_error', False))
                            except Exception:
                                fail_open = False
                            if fail_open:
                                logger.warning(f"   🛡️ FALLBACK: Allowing trade for safety (score: 75)")
                                ml_score = 75.0
                                ml_reason = "ML Error - Using Default Safety Score"
                                should_take_trade = True
                            else:
                                logger.warning(f"   🛡️ FAIL-CLOSED: Skipping execution, recording phantom")
                                ml_score = 0.0
                                ml_reason = "ML Error - Skipping execution"
                                should_take_trade = False

                    # Update strategy name for position tracking
                    strategy_name = selected_strategy

                    # Store ML data in signal for later use
                    if hasattr(sig, '__dict__'):
                        sig.__dict__['ml_score'] = ml_score
                        sig.__dict__['features'] = basic_features if 'basic_features' in locals() else {}
                        sig.__dict__['enhanced_features'] = enhanced_features if 'enhanced_features' in locals() else {}

                except Exception as e:
                    # Safety policy for unexpected scoring issues
                    try:
                        import traceback
                        logger.exception(f"[{sym}] ML scoring error: {e}")
                        logger.debug(traceback.format_exc())
                    except Exception:
                        logger.warning(f"[{sym}] ML scoring error: {e}.")
                    try:
                        tel = shared.get("telemetry", {})
                        tel['ml_errors'] = tel.get('ml_errors', 0) + 1
                    except Exception:
                        pass
                    try:
                        fail_open = bool(self.config.get('ml', {}).get('fail_open_on_error', False))
                    except Exception:
                        fail_open = False
                    if fail_open:
                        logger.warning(f"   🛡️ FALLBACK: Allowing signal for safety")
                        should_take_trade = True
                    else:
                        logger.warning(f"   🛡️ FAIL-CLOSED: Skipping execution and recording phantom")
                        should_take_trade = False

                # Ensure we have a valid signal before proceeding
                if sig is None:
                    logger.warning(f"[{sym}] No valid signal found after processing, skipping")
                    continue

                # Stale-feed guard: skip execution if data is too old (phantom is still recorded above)
                try:
                    guard_cfg = self.config.get('execution', {}).get('stale_feed_guard', {}) if hasattr(self, 'config') else {}
                    if bool(guard_cfg.get('enabled', True)) and should_take_trade:
                        max_lag = int(guard_cfg.get('max_lag_sec', 180))
                        last_ts = df.index[-1]
                        last_sec = int(pd.Timestamp(last_ts).tz_convert('UTC').timestamp() if getattr(last_ts, 'tzinfo', None) else pd.Timestamp(last_ts, tz='UTC').timestamp())
                        now_sec = int(pd.Timestamp.utcnow().timestamp())
                        if now_sec - last_sec > max_lag:
                            logger.warning(f"[{sym}] STALE FEED: last={now_sec-last_sec}s ago (> {max_lag}s). Skipping execution; recording phantom instead.")
                            try:
                                if selected_strategy == 'enhanced_mr' and selected_phantom_tracker:
                                    # MR phantom record
                                    selected_phantom_tracker.record_mr_signal(
                                        sym, sig.__dict__, float(ml_score or 0.0), False, {}, enhanced_features if 'enhanced_features' in locals() else {}
                                    )
                                elif selected_phantom_tracker:
                                    # Trend phantom record — include full HTF snapshot
                                    try:
                                        feats_sf = {}
                                        if selected_strategy in ('trend_pullback','pullback'):
                                            feats_sf = dict(trend_features) if 'trend_features' in locals() and isinstance(trend_features, dict) else {}
                                            if 'htf' not in feats_sf:
                                                feats_sf['htf'] = dict(self._compute_symbol_htf_exec_metrics(sym, df))
                                            # Add composite + flatten if missing
                                            try:
                                                comp4 = self._get_htf_metrics(sym, df)
                                                feats_sf.setdefault('htf_comp', dict(comp4))
                                                feats_sf.setdefault('ts15', float(comp4.get('ts15', 0.0)))
                                                feats_sf.setdefault('ts60', float(comp4.get('ts60', 0.0)))
                                                feats_sf.setdefault('rc15', float(comp4.get('rc15', 0.0)))
                                                feats_sf.setdefault('rc60', float(comp4.get('rc60', 0.0)))
                                            except Exception:
                                                pass
                                            try:
                                                # Tag diversion reason and stale metrics
                                                feats_sf['diversion_reason'] = 'stale_feed'
                                                feats_sf['stale_age_sec'] = int(now_sec - last_sec)
                                                feats_sf['stale_max_lag_sec'] = int(max_lag)
                                            except Exception:
                                                pass
                                        else:
                                            feats_sf = basic_features if 'basic_features' in locals() else {}
                                    except Exception:
                                        feats_sf = basic_features if 'basic_features' in locals() else {}
                                    selected_phantom_tracker.record_signal(
                                        symbol=sym,
                                        signal={'side': sig.side, 'entry': sig.entry, 'sl': sig.sl, 'tp': sig.tp},
                                        ml_score=float(ml_score or 0.0),
                                        was_executed=False,
                                        features=feats_sf,
                                        strategy_name=selected_strategy
                                    )
                            except Exception:
                                pass
                            try:
                                from strategy_pullback import revert_to_neutral
                                revert_to_neutral(sym)
                            except Exception:
                                pass
                            # Notify stale diversion
                            try:
                                if self.tg:
                                    await self.tg.send_message(f"🛑 Trend: [{sym}] Stale feed (> {max_lag}s) — routed to phantom")
                            except Exception:
                                pass
                            continue
                except Exception:
                    pass

                # One position per symbol rule - wait for current position to close
                # Final trade execution decision logging
                logger.info(f"💯 [{sym}] FINAL TRADE DECISION:")

                # Check for existing positions
                if sym in book.positions:
                    logger.info(f"   ❌ POSITION CONFLICT: Existing position already open")
                    logger.info(f"   📊 Current positions: {list(book.positions.keys())}")
                    logger.info(f"   💡 One position per symbol rule prevents duplicate entries")
                    continue

                # Get symbol metadata
                m = meta_for(sym, shared["meta"])

                # Round TP/SL to symbol's tick size to prevent API errors
                from position_mgr import round_step
                tick_size = m.get("tick_size", 0.000001) # Default to a small tick size
                
                original_tp = sig.tp
                original_sl = sig.sl

                sig.tp = round_step(sig.tp, tick_size)
                sig.sl = round_step(sig.sl, tick_size)

                if original_tp != sig.tp or original_sl != sig.sl:
                    logger.info(f"[{sym}] Rounded TP/SL to tick size {tick_size}. TP: {original_tp:.6f} -> {sig.tp:.6f}, SL: {original_sl:.6f} -> {sig.sl:.6f}")

                # Check account balance and update risk calculation
                risk_amount = risk.risk_usd
                if risk.use_percent_risk and sizer.account_balance and sizer.account_balance > 0:
                    risk_amount = sizer.account_balance * (risk.risk_percent / 100.0)

                current_balance = bybit.get_balance()
                if current_balance:
                    balance = current_balance
                    # Update sizer with current balance for percentage-based risk
                    sizer.account_balance = balance
                    shared["last_balance"] = balance

                    # Calculate actual risk amount for this trade
                    if risk.use_percent_risk:
                        risk_amount = balance * (risk.risk_percent / 100.0)
                    else:
                        risk_amount = risk.risk_usd

                    # Calculate required margin with max leverage
                    required_margin = (risk_amount * 100) / m.get("max_leverage", 50)  # Rough estimate
                
                    # Check if we have enough for margin + buffer
                    if balance < required_margin * 1.5:  # 1.5x for safety
                        logger.info(f"   ❌ INSUFFICIENT BALANCE:")
                        logger.info(f"      💰 Available: ${balance:.2f}")
                        logger.info(f"      📊 Required Margin: ≈${required_margin:.2f}")
                        logger.info(f"      ⚠️ Safety Buffer: {1.5}x margin = ${required_margin * 1.5:.2f}")
                        logger.info(f"      💡 Need ${(required_margin * 1.5) - balance:.2f} more to safely execute")
                        continue

                    logger.info(f"   ✅ BALANCE CHECK PASSED:")
                    logger.info(f"      💰 Available: ${balance:.2f}")
                    logger.info(f"      💸 Risk Amount: ${risk_amount:.2f}")
                    logger.info(f"      🛡️ Margin Required: ≈${required_margin:.2f}")
                else:
                    if risk.use_percent_risk:
                        logger.warning("Balance unavailable; using fallback USD risk amount")
                    shared["last_balance"] = None

                # Calculate position size
                qty = sizer.qty_for(sig.entry, sig.sl, m.get("qty_step",0.001), m.get("min_qty",0.001), ml_score=ml_score)

                if qty <= 0:
                    logger.info(f"   ❌ POSITION SIZE ERROR:")
                    logger.info(f"      📊 Calculated Quantity: {qty}")
                    logger.info(f"      💡 Check: Risk amount, entry price, stop loss distance")
                    logger.info(f"      🔧 Symbol specs: min_qty={m.get('min_qty',0.001)}, qty_step={m.get('qty_step',0.001)}")
                    continue
                
                # Get current market price for stop loss validation
                current_price = df['close'].iloc[-1]
                
                # Validate stop loss is on correct side of market price
                logger.info(f"   🔍 STOP LOSS VALIDATION:")
                logger.info(f"      📍 Current Price: {current_price:.4f}")
                logger.info(f"      🛑 Stop Loss: {sig.sl:.4f}")
                logger.info(f"      📊 Entry: {sig.entry:.4f}")

                sl_valid = True
                if sig.side == "long":
                    if sig.sl >= current_price:
                        logger.info(f"      ❌ INVALID: Long SL ({sig.sl:.4f}) must be BELOW current price ({current_price:.4f})")
                        logger.info(f"      💡 Long stops protect against downward moves")
                        sl_valid = False
                    else:
                        logger.info(f"      ✅ VALID: Long SL ({sig.sl:.4f}) is below current price")
                else:  # short
                    if sig.sl <= current_price:
                        logger.info(f"      ❌ INVALID: Short SL ({sig.sl:.4f}) must be ABOVE current price ({current_price:.4f})")
                        logger.info(f"      💡 Short stops protect against upward moves")
                        sl_valid = False
                    else:
                        logger.info(f"      ✅ VALID: Short SL ({sig.sl:.4f}) is above current price")

                if not sl_valid:
                    continue
                
                # Final execution summary
                logger.info(f"   🚀 EXECUTING TRADE:")
                logger.info(f"      📊 Strategy: {selected_strategy.upper()}")
                logger.info(f"      🎯 Signal: {sig.side.upper()} @ {sig.entry:.4f}")
                logger.info(f"      🛑 Stop Loss: {sig.sl:.4f}")
                logger.info(f"      💰 Take Profit: {sig.tp:.4f}")
                logger.info(f"      📈 Risk:Reward: {((sig.tp-sig.entry)/(sig.entry-sig.sl) if sig.side=='long' else (sig.entry-sig.tp)/(sig.sl-sig.entry)):.2f}")
                logger.info(f"      🔢 Quantity: {qty}")
                logger.info(f"      🧠 ML Score: {ml_score:.1f}")
                logger.info(f"      💸 Risk Amount: ${risk_amount:.2f}")

                # IMPORTANT: Set leverage BEFORE opening position to prevent TP/SL cancellation
                max_lev = int(m.get("max_leverage", 10))
                logger.info(f"   ⚙️ Setting leverage to {max_lev}x (before position to preserve TP/SL)")
                bybit.set_leverage(sym, max_lev)
                
                # Validate SL price before executing trade to prevent Bybit API errors
                current_price = df['close'].iloc[-1]
                if sig.side == "short" and sig.sl <= current_price:
                    logger.error(f"[{sym}] Invalid SL for SHORT: {sig.sl:.4f} must be > current price {current_price:.4f}")
                    # Record phantom with diversion reason and suppress execute
                    try:
                        if selected_strategy in ('trend_pullback','pullback') and 'phantom_tracker' in locals() and phantom_tracker is not None:
                            feats_is = {}
                            try:
                                feats_is = getattr(self, '_last_signal_features', {}).get(sym, {}) if hasattr(self, '_last_signal_features') else {}
                            except Exception:
                                feats_is = {}
                            try:
                                if isinstance(feats_is, dict):
                                    feats_is = feats_is.copy(); feats_is['diversion_reason'] = 'invalid_sl'
                            except Exception:
                                pass
                            phantom_tracker.record_signal(sym, {'side': sig.side, 'entry': float(sig.entry), 'sl': float(sig.sl), 'tp': float(sig.tp)}, float(ml_score or 0.0), False, feats_is, 'trend_pullback')
                    except Exception:
                        pass
                    continue
                elif sig.side == "long" and sig.sl >= current_price:
                    logger.error(f"[{sym}] Invalid SL for LONG: {sig.sl:.4f} must be < current price {current_price:.4f}")
                    # Record phantom with diversion reason and suppress execute
                    try:
                        if selected_strategy in ('trend_pullback','pullback') and 'phantom_tracker' in locals() and phantom_tracker is not None:
                            feats_is = {}
                            try:
                                feats_is = getattr(self, '_last_signal_features', {}).get(sym, {}) if hasattr(self, '_last_signal_features') else {}
                            except Exception:
                                feats_is = {}
                            try:
                                if isinstance(feats_is, dict):
                                    feats_is = feats_is.copy(); feats_is['diversion_reason'] = 'invalid_sl'
                            except Exception:
                                pass
                            phantom_tracker.record_signal(sym, {'side': sig.side, 'entry': float(sig.entry), 'sl': float(sig.sl), 'tp': float(sig.tp)}, float(ml_score or 0.0), False, feats_is, 'trend_pullback')
                    except Exception:
                        pass
                    continue

                # Place market order AFTER leverage is set
                side = "Buy" if sig.side == "long" else "Sell"
                try:
                    logger.info(f"[{sym}] Placing {side} order for {qty} units")
                    logger.debug(f"[{sym}] Order details: current_price={current_price:.4f}, sig.entry={sig.entry:.4f}, sig.sl={sig.sl:.4f}, sig.tp={sig.tp:.4f}")
                    order_result = bybit.place_market(sym, side, qty, reduce_only=False)
                    
                    # Get actual entry price from position
                    actual_entry = sig.entry  # Default to signal entry
                    try:
                        # Small delay to ensure position is updated
                        await asyncio.sleep(0.5)
                        
                        position = bybit.get_position(sym)
                        if position and position.get("avgPrice"):
                            actual_entry = float(position["avgPrice"])
                            logger.info(f"[{sym}] Actual entry price: {actual_entry:.4f} (signal was {sig.entry:.4f})")
                            
                            # Recalculate TP based on actual entry to maintain R:R ratio
                            if actual_entry != sig.entry:
                                risk_distance = abs(actual_entry - sig.sl)
                                # Apply same R:R ratio and fee adjustment
                                fee_adjustment = 1.00165  # Same as in strategy
                                if sig.side == "long":
                                    new_tp = actual_entry + (risk_distance * settings.rr * fee_adjustment)
                                else:
                                    new_tp = actual_entry - (risk_distance * settings.rr * fee_adjustment)

                                # Log the adjustment
                                tp_adjustment_pct = ((new_tp - sig.tp) / sig.tp) * 100
                                logger.info(f"[{sym}] Adjusting TP from {sig.tp:.4f} to {new_tp:.4f} ({tp_adjustment_pct:+.2f}%) to maintain {settings.rr}:1 R:R")
                                sig.tp = new_tp
                                # Trend: Recalc SL to preserve risk if slippage is material (bounded by pivots)
                                try:
                                    if selected_strategy == 'trend_pullback':
                                        sl_cfg = (cfg.get('trend', {}) or {}).get('exec', {}).get('slippage_recalc', {}) if 'cfg' in locals() else {}
                                        enabled = bool(sl_cfg.get('enabled', True))
                                        min_pct = float(sl_cfg.get('min_pct', 0.001))
                                        if enabled and qty > 0:
                                            slip_pct = abs(actual_entry - sig.entry) / max(1e-9, sig.entry)
                                            if slip_pct >= min_pct:
                                                target_dist = float(risk_amount) / float(qty)
                                                if sig.side == 'long':
                                                    new_sl = actual_entry - target_dist
                                                    pivot = None
                                                    try:
                                                        if isinstance(sig.meta, dict):
                                                            pivot = float(sig.meta.get('pivot_low'))
                                                    except Exception:
                                                        pivot = None
                                                    if pivot is not None:
                                                        atr = float(sig.meta.get('atr', 0.0)) if isinstance(sig.meta, dict) else 0.0
                                                        new_sl = min(new_sl, pivot - 0.05 * atr)
                                                    min_stop = actual_entry * 0.01
                                                    if (actual_entry - new_sl) < min_stop:
                                                        new_sl = actual_entry - min_stop
                                                    if new_sl < actual_entry:
                                                        sig.sl = new_sl
                                                else:
                                                    new_sl = actual_entry + target_dist
                                                    pivot = None
                                                    try:
                                                        if isinstance(sig.meta, dict):
                                                            pivot = float(sig.meta.get('pivot_high'))
                                                    except Exception:
                                                        pivot = None
                                                    if pivot is not None:
                                                        atr = float(sig.meta.get('atr', 0.0)) if isinstance(sig.meta, dict) else 0.0
                                                        new_sl = max(new_sl, pivot + 0.05 * atr)
                                                    min_stop = actual_entry * 0.01
                                                    if (new_sl - actual_entry) < min_stop:
                                                        new_sl = actual_entry + min_stop
                                                    if new_sl > actual_entry:
                                                        sig.sl = new_sl
                                                logger.info(f"[{sym}] TREND SL recalc for slippage: SL -> {sig.sl:.4f}")
                                except Exception as _tre:
                                    logger.debug(f"[{sym}] Trend SL recalc skipped: {_tre}")
                                # MR: Recalc SL for slippage to preserve risk, bounded by range pivots
                                try:
                                    if selected_strategy == 'enhanced_mr':
                                        sl_cfg = (cfg.get('mr', {}) or {}).get('exec', {}).get('slippage_recalc', {}) if 'cfg' in locals() else {}
                                        enabled = bool(sl_cfg.get('enabled', True))
                                        min_pct = float(sl_cfg.get('min_pct', 0.001))
                                        pivot_buf_atr = float(sl_cfg.get('pivot_buffer_atr', 0.05))
                                        if enabled and qty > 0:
                                            slip_pct = abs(actual_entry - sig.entry) / max(1e-9, sig.entry)
                                            if slip_pct >= min_pct:
                                                target_dist = float(risk_amount) / float(qty)
                                                # Compute ATR quickly for buffer
                                                try:
                                                    prev = df['close'].shift()
                                                    tr = np.maximum(df['high'] - df['low'], np.maximum((df['high'] - prev).abs(), (df['low'] - prev).abs()))
                                                    atr_len = int(getattr(settings, 'atr_len', 14) or 14)
                                                    atr_val = float(tr.rolling(atr_len).mean().iloc[-1]) if len(tr) >= atr_len else float(tr.iloc[-1])
                                                except Exception:
                                                    atr_val = 0.0
                                                if sig.side == 'long':
                                                    new_sl = actual_entry - target_dist
                                                    pivot = None
                                                    try:
                                                        if isinstance(sig.meta, dict):
                                                            pivot = float(sig.meta.get('range_lower'))
                                                    except Exception:
                                                        pivot = None
                                                    if pivot is not None and atr_val > 0:
                                                        new_sl = min(new_sl, pivot - (pivot_buf_atr * atr_val))
                                                    min_stop = actual_entry * 0.01
                                                    if (actual_entry - new_sl) < min_stop:
                                                        new_sl = actual_entry - min_stop
                                                    if new_sl < actual_entry:
                                                        sig.sl = new_sl
                                                else:
                                                    new_sl = actual_entry + target_dist
                                                    pivot = None
                                                    try:
                                                        if isinstance(sig.meta, dict):
                                                            pivot = float(sig.meta.get('range_upper'))
                                                    except Exception:
                                                        pivot = None
                                                    if pivot is not None and atr_val > 0:
                                                        new_sl = max(new_sl, pivot + (pivot_buf_atr * atr_val))
                                                    min_stop = actual_entry * 0.01
                                                    if (new_sl - actual_entry) < min_stop:
                                                        new_sl = actual_entry + min_stop
                                                    if new_sl > actual_entry:
                                                        sig.sl = new_sl
                                                logger.info(f"[{sym}] MR SL recalc for slippage: SL -> {sig.sl:.4f}")
                                except Exception as _mre:
                                    logger.debug(f"[{sym}] MR SL recalc skipped: {_mre}")
                    except Exception as e:
                        logger.warning(f"[{sym}] Could not get actual entry price: {e}. Using signal entry.")
                    
                    # Set TP/SL - CRITICAL: This must succeed or position becomes unprotected
                    logger.info(f"[{sym}] Setting TP={sig.tp:.4f} (Limit), SL={sig.sl:.4f} (Market)")
                    try:
                        bybit.set_tpsl(sym, take_profit=sig.tp, stop_loss=sig.sl, qty=qty)
                        # Read back TP/SL from exchange to reflect any server-side rounding
                        try:
                            pos_back = bybit.get_position(sym)
                            if isinstance(pos_back, dict):
                                tp_val = pos_back.get('takeProfit')
                                sl_val = pos_back.get('stopLoss')
                                if tp_val not in (None, '', '0'):
                                    sig.tp = float(tp_val)
                                if sl_val not in (None, '', '0'):
                                    sig.sl = float(sl_val)
                        except Exception:
                            pass
                        logger.info(f"[{sym}] TP/SL set successfully")
                    except Exception as tpsl_error:
                        logger.critical(f"[{sym}] CRITICAL: Failed to set TP/SL: {tpsl_error}")
                        logger.critical(f"[{sym}] Attempting emergency position closure to prevent unprotected position")
                        
                        # Attempt to close the unprotected position immediately
                        try:
                            emergency_side = "Sell" if sig.side == "long" else "Buy"
                            close_result = bybit.place_market(sym, emergency_side, qty, reduce_only=True)
                            logger.warning(f"[{sym}] Emergency position closure executed: {close_result}")
                            if self.tg:
                                await self.tg.send_message(f"🚨 EMERGENCY CLOSURE: {sym} position closed due to TP/SL failure: {str(tpsl_error)[:100]}")
                        except Exception as close_error:
                            logger.critical(f"[{sym}] FAILED TO CLOSE UNPROTECTED POSITION: {close_error}")
                            if self.tg:
                                await self.tg.send_message(f"🆘 CRITICAL: {sym} position UNPROTECTED! Manual intervention required. SL/TP failed: {str(tpsl_error)[:100]}")
                                # Stop bot from taking new trades until manual review
                                await self.tg.send_message(f"🛑 Bot halted due to unprotected position. Use /resume to restart after manual review.")
                        
                        # Re-raise the original error to prevent position being added to book
                        raise Exception(f"Failed to set TP/SL for {sym}: {tpsl_error}")
                
                    # Only update book if BOTH market order AND TP/SL succeeded
                    book.positions[sym] = Position(
                        sig.side,
                        qty,
                        actual_entry,
                        sig.sl,
                        sig.tp,
                        datetime.now(),
                        strategy_name=(strategy_name if strategy_name in ('range_fbo','trend_pullback','enhanced_mr','scalp','mean_reversion') else selected_strategy),
                        ml_score=float(ml_score),
                        ml_reason=ml_reason if isinstance(ml_reason, str) else ""
                    )
                    # For Trend: if slippage caused a SL recalc earlier, ensure TP/SL have been read back and adjusted; otherwise, adjust here too
                    try:
                        if selected_strategy == 'trend_pullback':
                            # Already set, but re-apply set_tpsl if needed to reflect any SL recalc
                            bybit.set_tpsl(sym, take_profit=sig.tp, stop_loss=sig.sl, qty=qty)
                            pos_back = bybit.get_position(sym)
                            if isinstance(pos_back, dict):
                                tp_val = pos_back.get('takeProfit')
                                sl_val = pos_back.get('stopLoss')
                                if tp_val not in (None, '', '0'):
                                    sig.tp = float(tp_val)
                                if sl_val not in (None, '', '0'):
                                    sig.sl = float(sl_val)
                    except Exception:
                        pass
                    # Record MR phantom mirror (executed or promotion-forced phantom) with exchange-aligned values
                    try:
                        if selected_strategy == 'enhanced_mr' and mr_phantom_tracker is not None:
                            is_promo = False
                            try:
                                is_promo = bool(getattr(sig, 'meta', {}) and sig.meta.get('promotion_forced'))
                            except Exception:
                                is_promo = False
                            mr_phantom_tracker.record_mr_signal(
                                sym,
                                {'side': sig.side, 'entry': float(actual_entry), 'sl': float(sig.sl), 'tp': float(sig.tp), 'meta': getattr(sig, 'meta', {}) or {}},
                                float(ml_score or 0.0),
                                False if is_promo else True,
                                {},
                                enhanced_features if 'enhanced_features' in locals() else {}
                            )
                            # Persist promotion flag and reason for restart resilience
                            try:
                                if getattr(self, '_redis', None) is not None:
                                    if is_promo:
                                        self._redis.set(f'openpos:mr_promotion:{sym}', '1')
                                        self._redis.set(f'openpos:reason:{sym}', 'Mean Reversion (Promotion)')
                                    else:
                                        self._redis.delete(f'openpos:mr_promotion:{sym}')
                                        self._redis.set(f'openpos:reason:{sym}', 'Mean Reversion (ML)')
                            except Exception:
                                pass
                    except Exception as e:
                        logger.debug(f"[{sym}] MR executed/promotion phantom mirror record failed: {e}")
                    # Record Trend phantom mirror (executed or promotion-forced) with exchange-aligned values
                    try:
                        if selected_strategy == 'trend_pullback' and phantom_tracker is not None:
                            is_promo = False
                            try:
                                is_promo = bool(getattr(sig, 'meta', {}) and sig.meta.get('promotion_forced'))
                            except Exception:
                                is_promo = False
                            # Build basic trend features if available
                            trend_features = {}
                            try:
                                cl = df['close']; price = float(cl.iloc[-1])
                                ys = cl.tail(20).values if len(cl) >= 20 else cl.values
                                try:
                                    slope = np.polyfit(np.arange(len(ys)), ys, 1)[0]
                                except Exception:
                                    slope = 0.0
                                trend_slope_pct = float((slope / price) * 100.0) if price else 0.0
                                ema20 = cl.ewm(span=20, adjust=False).mean().iloc[-1]
                                ema50 = cl.ewm(span=50, adjust=False).mean().iloc[-1] if len(cl) >= 50 else ema20
                                ema_stack_score = 100.0 if (price > ema20 > ema50 or price < ema20 < ema50) else 50.0 if (ema20 != ema50) else 0.0
                                rng_today = float(df['high'].iloc[-1] - df['low'].iloc[-1])
                                med_range = float((df['high'] - df['low']).rolling(20).median().iloc[-1]) if len(df) >= 20 else rng_today
                                range_expansion = float(rng_today / max(1e-9, med_range))
                                prev = cl.shift(); trarr = np.maximum(df['high'] - df['low'], np.maximum((df['high'] - prev).abs(), (df['low'] - prev).abs()))
                                atr = float(trarr.rolling(14).mean().iloc[-1]) if len(trarr) >= 14 else float(trarr.iloc[-1])
                                atr_pct = float((atr / max(1e-9, price)) * 100.0) if price else 0.0
                                close_vs_ema20_pct = float(((price - ema20) / max(1e-9, ema20)) * 100.0) if ema20 else 0.0
                                trend_features = {
                                    'trend_slope_pct': trend_slope_pct,
                                    'ema_stack_score': ema_stack_score,
                                    'atr_pct': atr_pct,
                                    'range_expansion': range_expansion,
                                    'breakout_dist_atr': float(getattr(sig, 'meta', {}).get('breakout_dist_atr', 0.0)) if getattr(sig, 'meta', None) else 0.0,
                                    'close_vs_ema20_pct': close_vs_ema20_pct,
                                    'bb_width_pct': 0.0,
                                    'session': 'us',
                                    'symbol_cluster': 3,
                                    'volatility_regime': getattr(regime_analysis, 'volatility_level', 'normal') if 'regime_analysis' in locals() else 'normal',
                                }
                                # Add Qscore if rule_mode
                                try:
                                    rule_mode = (cfg.get('trend', {}) or {}).get('rule_mode', {}) if 'cfg' in locals() else {}
                                    if bool(rule_mode.get('enabled', False)):
                                        qE, _, _ = self._compute_qscore(sym, sig.side, df, self.frames_3m.get(sym) if hasattr(self, 'frames_3m') else None)
                                        trend_features['qscore'] = float(qE)
                                except Exception:
                                    pass
                                # Attach HTF snapshots + composite (flattened)
                                try:
                                    trend_features['htf'] = dict(self._compute_symbol_htf_exec_metrics(sym, df))
                                except Exception:
                                    pass
                                try:
                                    comp5 = self._get_htf_metrics(sym, df)
                                    trend_features['htf_comp'] = dict(comp5)
                                    trend_features['ts15'] = float(comp5.get('ts15', 0.0))
                                    trend_features['ts60'] = float(comp5.get('ts60', 0.0))
                                    trend_features['rc15'] = float(comp5.get('rc15', 0.0))
                                    trend_features['rc60'] = float(comp5.get('rc60', 0.0))
                                except Exception:
                                    pass
                                # Attach full HTF snapshot
                                try:
                                    trend_features['htf'] = dict(self._compute_symbol_htf_exec_metrics(sym, df))
                                except Exception:
                                    pass
                            except Exception:
                                trend_features = {}
                            phantom_tracker.record_signal(
                                sym,
                                {'side': sig.side, 'entry': float(actual_entry), 'sl': float(sig.sl), 'tp': float(sig.tp), 'meta': getattr(sig, 'meta', {}) or {}},
                                float(ml_score or 0.0),
                                False if is_promo else True,
                                trend_features,
                                'trend_pullback'
                            )
                            # Persist promotion flag and reason for restart resilience
                            try:
                                if getattr(self, '_redis', None) is not None:
                                    if is_promo:
                                        self._redis.set(f'openpos:trend_promotion:{sym}', '1')
                                        self._redis.set(f'openpos:reason:{sym}', 'Trend (Promotion)')
                                    else:
                                        self._redis.delete(f'openpos:trend_promotion:{sym}')
                                        self._redis.set(f'openpos:reason:{sym}', 'Trend (ML)')
                            except Exception:
                                pass
                    except Exception as e:
                        logger.debug(f"[{sym}] Trend executed/promotion phantom mirror record failed: {e}")
                    # Optionally cancel phantoms on exec (config-controlled)
                    try:
                        ph_cfg = cfg.get('phantom', {}) if 'cfg' in locals() else {}
                        if bool(ph_cfg.get('cancel_on_execute', False)):
                            if selected_strategy == 'enhanced_mr' and mr_phantom_tracker:
                                mr_phantom_tracker.cancel_active(sym)
                            elif selected_strategy in ('trend_pullback','pullback') and phantom_tracker:
                                phantom_tracker.cancel_active(sym)
                    except Exception:
                        pass
                    # Persist runtime hint of strategy for recovery after restarts
                    try:
                        if getattr(self, '_redis', None) is not None:
                            self._redis.set(f'openpos:strategy:{sym}', selected_strategy)
                    except Exception:
                        pass
                    # Record shadow simulation (phantom-only) for ML-informed adjustments
                    try:
                        base_features = {}
                        if selected_strategy == 'trend_pullback':
                            base_features = basic_features if 'basic_features' in locals() else {}
                        elif selected_strategy == 'enhanced_mr':
                            base_features = enhanced_features if 'enhanced_features' in locals() else {}
                        get_shadow_tracker().record_shadow_trade(
                            strategy=selected_strategy,
                            symbol=sym,
                            side=sig.side,
                            entry=float(actual_entry),
                            sl=float(sig.sl),
                            tp=float(sig.tp),
                            ml_score=float(ml_score or 0.0),
                            features=base_features or {}
                        )
                    except Exception:
                        pass
                    last_signal_time[sym] = now
                    
                    # Debug log the position details
                    logger.debug(f"[{sym}] Stored position: side={sig.side}, entry={actual_entry:.4f}, TP={sig.tp:.4f}, SL={sig.sl:.4f}")
                    
                    # Record comprehensive trade context for future ML
                    if symbol_collector:
                        try:
                            # Get current BTC price for market context
                            btc_price = None
                            if 'BTCUSDT' in self.frames:
                                btc_price = self.frames['BTCUSDT']['close'].iloc[-1]
                            
                            # Record the context
                            trade_context = symbol_collector.record_trade_context(
                                symbol=sym,
                                df=df,
                                btc_price=btc_price
                            )
                            logger.debug(f"[{sym}] Recorded trade context: session={trade_context.session}, vol_ratio={trade_context.volume_vs_avg:.2f}")
                            
                            # Update symbol profile
                            symbol_collector.update_symbol_profile(sym)
                            
                        except Exception as e:
                            logger.debug(f"Failed to record trade context: {e}")
                
                    # Send notification
                    if self.tg:
                        emoji = "🟢" if sig.side == "long" else "🔴"
                        logger.debug(f"[{sym}] Notification: side='{sig.side}', emoji='{emoji}', order_side='{side}'")

                        if risk.use_ml_dynamic_risk:
                            score_range = risk.ml_risk_max_score - risk.ml_risk_min_score
                            risk_range = risk.ml_risk_max_percent - risk.ml_risk_min_percent
                            clamped_score = max(risk.ml_risk_min_score, min(risk.ml_risk_max_score, ml_score))
                            if score_range > 0:
                                score_position = (clamped_score - risk.ml_risk_min_score) / score_range
                                actual_risk_pct = risk.ml_risk_min_percent + (score_position * risk_range)
                            else:
                                actual_risk_pct = risk.ml_risk_min_percent
                            risk_display = f"{actual_risk_pct:.2f}% (ML: {ml_score:.1f})"
                        else:
                            actual_risk_pct = risk.risk_percent
                            risk_display = f"{actual_risk_pct}%"

                        entry_info = f"Entry: {actual_entry:.4f}"
                        if actual_entry != sig.entry:
                            price_diff_pct = ((actual_entry - sig.entry) / sig.entry) * 100
                            entry_info += f" (signal: {sig.entry:.4f}, {price_diff_pct:+.2f}%)"

                        # Prefer runtime strategy_name when provided (range_fbo), else selected_strategy
                        try:
                            base_label = strategy_name if strategy_name in ('range_fbo','trend_pullback','enhanced_mr','scalp','mean_reversion') else selected_strategy
                        except Exception:
                            base_label = selected_strategy
                        strategy_label = base_label.replace('_', ' ').title()
                        threshold_text = "N/A"
                        if selected_ml_scorer is not None:
                            threshold_text = f"{selected_ml_scorer.min_score:.0f}"
                        elif selected_strategy == "enhanced_mr" and enhanced_mr_scorer is not None:
                            threshold_text = f"{enhanced_mr_scorer.min_score:.0f}"

                        msg = (
                            f"{emoji} *{sym} {sig.side.upper()}* ({strategy_label})\n\n"
                            f"{entry_info}\n"
                            f"Stop Loss: {sig.sl:.4f}\n"
                            f"Take Profit: {sig.tp:.4f}\n"
                            f"Quantity: {qty}\n"
                            f"Risk: {risk_display} (${risk_amount:.2f})\n"
                            f"ML Score: {ml_score:.1f} (≥ {threshold_text})\n"
                            f"Reason: {sig.reason}"
                        )
                        await self.tg.send_message(msg)
                
                    logger.info(f"[{sym}] {sig.side} position opened successfully")
                    # Account for MR promotion daily cap if override was used
                    try:
                        if selected_strategy == 'enhanced_mr' and isinstance(getattr(sig, 'meta', {}), dict) and sig.meta.get('promotion_forced'):
                            mp = shared.get('mr_promotion', {})
                            mp['count'] = int(mp.get('count', 0)) + 1
                            shared['mr_promotion'] = mp
                    except Exception:
                        pass
                    
                    # MR phantom tracking: Status already recorded correctly with initial phantom record
                
                except KeyError as e:
                    # KeyError likely means symbol not in config or metadata
                    if str(e).strip("'") == sym:
                        logger.debug(f"[{sym}] Not found in metadata/config - skipping")
                    else:
                        logger.error(f"[{sym}] KeyError accessing: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Order error: {e}")
                    if self.tg:
                        await self.tg.send_message(f"❌ Failed to open {sym} {sig.side}: {str(e)[:100]}")
                except Exception as e:
                    # Other errors - log with more detail
                    import traceback
                    logger.error(f"[{sym}] Processing error: {type(e).__name__}: {e}")
                    logger.debug(f"[{sym}] Traceback: {traceback.format_exc()}")
                    # Don't crash the whole bot for one symbol's error
                    continue
        

    async def start(self):
        """Start the bot"""
        self.running = True
        logger.info("Starting trading bot...")
        await self.run()
    
    async def stop(self):
        """Stop the bot"""
        logger.info("Stopping trading bot...")
        self.running = False
        # Cancel background tasks
        try:
            for task in list(self._tasks):
                task.cancel()
            if self._tasks:
                await asyncio.gather(*list(self._tasks), return_exceptions=True)
        except Exception:
            pass
        if self.ws:
            await self.ws.close()
        if self.tg:
            await self.tg.stop()

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}, shutting down...")
    asyncio.create_task(bot.stop())
    sys.exit(0)

# Global bot instance
bot = TradingBot()

if __name__ == "__main__":
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        asyncio.run(bot.start())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        # Save candles before shutdown
        if bot.frames:
            logger.info("Saving candles to database before shutdown...")
            bot.storage.save_all_frames(bot.frames)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        # Final cleanup
        if hasattr(bot, 'storage'):
            bot.storage.close()
        logger.info("Bot terminated")
        # Decision-only log filter: keep INFO logs focused on decisions (phantom/execution) for core strategies
        try:
            log_cfg = (cfg.get('logging', {}) or {})
            decision_only = bool(log_cfg.get('decision_only', True))
        except Exception:
            decision_only = True
        if decision_only:
            class _DecisionOnlyFilter(logging.Filter):
                def filter(self, record: logging.LogRecord) -> bool:
                    try:
                        # Always allow warnings and errors
                        if record.levelno >= logging.WARNING:
                            return True
                        msg = str(record.getMessage())
                        # Allow core decision/execution lines for Trend/Range/Scalp and TP/SL confirmations
                        allow_tokens = (
                            'Decision final:', 'decision final:',
                            'Trend decision final', 'Range decision final',
                            'exec_scalp', 'Scalp: Executing', 'TP/SL confirmed',
                            'Range PHANTOM', '👻', 'phantom', 'Phantom'
                        )
                        return any(t in msg for t in allow_tokens)
                    except Exception:
                        return True
            try:
                logging.getLogger().addFilter(_DecisionOnlyFilter())
            except Exception:
                pass
        # Helper: session label used by adapters/telemetry
        def _sess_label() -> str:
            try:
                hr = datetime.utcnow().hour
                return 'asian' if 0 <= hr < 8 else ('european' if hr < 16 else 'us')
            except Exception:
                return 'us'
        try:
            self._session_label = _sess_label
        except Exception:
            pass
