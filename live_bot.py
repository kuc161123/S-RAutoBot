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

# Trade tracking with PostgreSQL fallback
try:
    from trade_tracker_postgres import TradeTrackerPostgres as TradeTracker, Trade
    USING_POSTGRES_TRACKER = True
except ImportError:
    from trade_tracker import TradeTracker, Trade
    USING_POSTGRES_TRACKER = False

# ML scoring system - Enhanced parallel system (Trend + Mean Reversion)
try:
    from ml_signal_scorer_immediate import get_immediate_scorer
    from phantom_trade_tracker import get_phantom_tracker
    from enhanced_mr_scorer import get_enhanced_mr_scorer
    from mr_phantom_tracker import get_mr_phantom_tracker
    from ml_scorer_trend import get_trend_scorer
    from enhanced_market_regime import get_enhanced_market_regime, get_regime_summary
    from ml_scorer_mean_reversion import get_mean_reversion_scorer
    logger = logging.getLogger(__name__)
    logger.info("Using Enhanced Parallel ML System (Trend + Mean Reversion)")
    ML_AVAILABLE = True
    ENHANCED_ML_AVAILABLE = True
except ImportError as e:
    ML_AVAILABLE = False
    ENHANCED_ML_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"Enhanced ML not available: {e}")
    # Fallback to original ML if available
    try:
        from ml_signal_scorer_immediate import get_immediate_scorer
        from phantom_trade_tracker import get_phantom_tracker
        from ml_scorer_mean_reversion import get_mean_reversion_scorer
        try:
            from ml_scorer_trend import get_trend_scorer
        except Exception:
            get_trend_scorer = None
        ML_AVAILABLE = True
        logger.info("Using Original ML Scorer only")
    except ImportError as e2:
        logger.warning(f"No ML available: {e2}")

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
VERSION = "2025.10.09.3"

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
        # Cache of latest scalp detections per symbol (for promotion timing)
        self._scalp_last_signal: Dict[str, Dict[str, object]] = {}
        # HTF gating persistence (per-strategy per-symbol)
        self._htf_hold: Dict[str, Dict[str, Dict[str, object]]] = {
            'mr': {},
            'trend': {}
        }

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

    async def _execute_scalp_trade(self, sym: str, sig_obj, ml_score: float = 0.0) -> bool:
        """Execute a Scalp trade immediately. Returns True if executed.

        Bypasses routing/regime/micro gates. Still subject to hard execution guards
        (existing position, invalid SL/TP, sizing, exchange errors).
        """
        try:
            bybit = self.bybit
            book = self.book
            sizer = getattr(self, 'sizer', None)
            cfg = getattr(self, 'config', {}) or {}
            shared = getattr(self, 'shared', {}) or {}
            # One position per symbol
            if sym in book.positions:
                return False
            # Round TP/SL to tick size
            m = meta_for(sym, cfg.get('symbol_meta', {}))
            from position_mgr import round_step
            tick_size = m.get("tick_size", 0.000001)
            sig_obj.tp = round_step(float(sig_obj.tp), tick_size)
            sig_obj.sl = round_step(float(sig_obj.sl), tick_size)
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
                return False
            # SL sanity
            try:
                df_main = self.frames.get(sym)
                current_price = float(df_main['close'].iloc[-1]) if (df_main is not None and len(df_main)>0) else float(sig_obj.entry)
            except Exception:
                current_price = float(sig_obj.entry)
            if (sig_obj.side == "long" and float(sig_obj.sl) >= current_price) or (sig_obj.side == "short" and float(sig_obj.sl) <= current_price):
                return False
            # Leverage and market order
            max_lev = int(m.get("max_leverage", 10))
            bybit.set_leverage(sym, max_lev)
            side = "Buy" if sig_obj.side == "long" else "Sell"
            _ = bybit.place_market(sym, side, qty, reduce_only=False)
            # Read back avg entry and set TP/SL
            actual_entry = float(sig_obj.entry)
            try:
                await asyncio.sleep(0.4)
                position = bybit.get_position(sym)
                if position and position.get("avgPrice"):
                    actual_entry = float(position["avgPrice"]) or actual_entry
            except Exception:
                pass
            # Set TP/SL
            try:
                bybit.set_tpsl(sym, take_profit=float(sig_obj.tp), stop_loss=float(sig_obj.sl), qty=qty)
            except Exception:
                # try once more without qty (Full mode)
                try:
                    bybit.set_tpsl(sym, take_profit=float(sig_obj.tp), stop_loss=float(sig_obj.sl))
                except Exception:
                    return False
            # Update book
            self.book.positions[sym] = Position(
                sig_obj.side,
                qty,
                entry=actual_entry,
                sl=float(sig_obj.sl),
                tp=float(sig_obj.tp),
                entry_time=datetime.now(),
                strategy_name='scalp'
            )
            # Telegram notify
            try:
                if self.tg:
                    msg = (
                        f"🩳 Scalp High-ML: Executing {sym} {sig_obj.side.upper()}\n\n"
                        f"Entry: {actual_entry:.4f}\n"
                        f"Stop Loss: {float(sig_obj.sl):.4f}\n"
                        f"Take Profit: {float(sig_obj.tp):.4f}\n"
                        f"Quantity: {qty}"
                    )
                    await self.tg.send_message(msg)
            except Exception:
                pass
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

        except Exception:
            # Return whatever computed; scorer will default missing
            pass
        return out

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
                # Persist the latest 3m candle row
                try:
                    self.storage.save_candles_3m(sym, row)
                except Exception as e:
                    logger.debug(f"3m candle persist error for {sym}: {e}")

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
                                continue
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
                if last_ts is not None:
                    try:
                        if (bar_ts - last_ts).total_seconds() < (cooldown_bars * bar_seconds):
                            try:
                                self._scalp_stats['cooldown_skips'] = self._scalp_stats.get('cooldown_skips', 0) + 1
                            except Exception:
                                pass
                            try:
                                logger.info(f"[{sym}] 🧮 Scalp decision final: blocked (reason=cooldown)")
                            except Exception:
                                pass
                            continue
                    except Exception:
                        pass

                # Run scalper on 3m df (independent of regime)
                # Build ScalpSettings, applying phantom-only relaxed thresholds if enabled
                try:
                    sc_settings = ScalpSettings()
                    try:
                        s_cfg = self.config.get('scalp', {}) if hasattr(self, 'config') else {}
                        exp = s_cfg.get('explore', {})
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
                    # Optional: compact gate probe at debug to help tune thresholds
                    try:
                        if df3_for_sig is not None and len(df3_for_sig) >= max(sc_settings.vwap_window, 50):
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
                            _cur_atr = float(_atr.iloc[-1]) if _atr.iloc[-1] and _atr.iloc[-1] > 0 else _rng
                            _dist_vwap_atr = abs(_cl - _cur_vwap) / max(1e-9, _cur_atr)
                            _orb_ok = True
                            if len(df3_for_sig) >= 40:
                                _first_high = float(_h.iloc[:20].max()); _first_low = float(_l.iloc[:20].min())
                                if _ema_up and _cl <= _first_high:
                                    _orb_ok = False
                                if _ema_dn and _cl >= _first_low:
                                    _orb_ok = False
                            logger.debug(
                                f"[{sym}] 🩳 Scalp gates: up={_ema_up} dn={_ema_dn} bbw={_bbw_pct:.2f}>={sc_settings.min_bb_width_pct:.2f} "
                                f"vol={_vol_ratio:.2f}>={sc_settings.vol_ratio_min:.2f} wickL={_lower_w:.2f}/wickU={_upper_w:.2f}>={sc_settings.wick_ratio_min:.2f} "
                                f"distVWAP={_dist_vwap_atr:.2f}<={sc_settings.vwap_dist_atr_max:.2f} orb={_orb_ok} -> no_signal"
                            )
                    except Exception:
                        pass
                    # Do not spam logs at 3m otherwise; only log positives and regime skips
                    continue
                try:
                    logger.info(f"[{sym}] 🩳 Scalp signal: {getattr(sc_sig, 'side','?').upper()} @ {float(getattr(sc_sig,'entry',0.0)):.4f}")
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


                # EARLY debug force-accept: record immediately to prove acceptance path
                try:
                    fa_cfg = (self.config.get('scalp', {}).get('debug', {}) or {}) if hasattr(self, 'config') else {}
                    if bool(fa_cfg.get('force_accept', False)):
                        try:
                            from scalp_phantom_tracker import get_scalp_phantom_tracker
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
                                sc_feats_early
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

                # Dedup via Redis (phantom dedup scope)
                dedup_ok = True
                try:
                    from scalp_phantom_tracker import get_scalp_phantom_tracker
                    scpt = get_scalp_phantom_tracker()
                    r = scpt.redis_client
                    if r is not None:
                        key = f"{sym}:{sc_sig.side}:{round(float(sc_sig.entry),6)}:{round(float(sc_sig.sl),6)}:{round(float(sc_sig.tp),6)}"
                        # TTL configurable
                        try:
                            s_cfg = self.config.get('scalp', {}) if hasattr(self, 'config') else {}
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
                    except Exception:
                        pass
                    continue

                # One active phantom per symbol guard
                try:
                    if scpt.has_active(sym):
                        logger.info(f"[{sym}] 🧮 Scalp decision final: blocked (reason=active_symbol)")
                        _scalp_decision_logged = True
                        continue
                except Exception:
                    pass

                # Record phantom to scalp tracker; build full feature set
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
                sc_feats['routing'] = 'none'
                try:
                    scpt = get_scalp_phantom_tracker()
                    # Compute Scalp ML score if scorer is available
                    ml_s = 0.0
                    try:
                        from ml_scorer_scalp import get_scalp_scorer
                        _scorer = get_scalp_scorer()
                        ml_s, _ = _scorer.score_signal({'side': sc_sig.side, 'entry': sc_sig.entry, 'sl': sc_sig.sl, 'tp': sc_sig.tp}, sc_feats)
                    except Exception:
                        ml_s = 0.0
                    # High-ML immediate execution override (bypass gating)
                    try:
                        e_cfg = (self.config.get('scalp', {}) or {}).get('exec', {})
                        allow_hi = bool(e_cfg.get('allow_stream_high_ml', True))
                        hi_thr = float(e_cfg.get('high_ml_force', 92))
                    except Exception:
                        allow_hi = True; hi_thr = 92.0
                    if allow_hi and float(ml_s or 0.0) >= hi_thr:
                        try:
                            if sym in self.book.positions:
                                logger.info(f"[{sym}] 🛑 Scalp High-ML blocked: reason=position_exists")
                            else:
                                executed = await self._execute_scalp_trade(sym, sc_sig, ml_score=float(ml_s or 0.0))
                                # Record phantom as executed for learning
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
                                    self._scalp_cooldown[sym] = bar_ts
                                    blist.append(now_ts)
                                    self._scalp_budget[sym] = blist
                                    continue
                                else:
                                    logger.info(f"[{sym}] 🛑 Scalp High-ML override blocked: reason=exec_guard")
                        except Exception as _ee:
                            logger.info(f"[{sym}] Scalp High-ML override error: {_ee}")
                    # Enforce per-strategy hourly per-symbol budget for scalp
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
                    # Visibility into decision inputs
                    try:
                        logger.info(f"[{sym}] 🩳 Scalp decision context: dedup={dedup_ok} hourly_remaining={max(0, sc_remaining)} daily_ok={daily_ok}")
                    except Exception:
                        pass
                    # Debug: force-accept path for diagnostics
                    force_accept = False
                    try:
                        force_accept = bool(((self.config.get('scalp', {}).get('debug', {}) or {}).get('force_accept', False)))
                    except Exception:
                        force_accept = False
                    if force_accept:
                        scpt.record_scalp_signal(
                            sym,
                            {'side': sc_sig.side, 'entry': sc_sig.entry, 'sl': sc_sig.sl, 'tp': sc_sig.tp},
                            float(ml_s or 0.0),
                            False,
                            sc_feats
                        )
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
                    elif sc_remaining > 0 and daily_ok:
                        scpt.record_scalp_signal(
                            sym,
                            {'side': sc_sig.side, 'entry': sc_sig.entry, 'sl': sc_sig.sl, 'tp': sc_sig.tp},
                            float(ml_s or 0.0),
                            False,
                            sc_feats
                        )
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
                        try:
                            logger.info(f"[{sym}] 🧮 Scalp decision final: phantom (reason=ok)")
                            _scalp_decision_logged = True
                        except Exception:
                            pass
                        self._scalp_cooldown[sym] = bar_ts
                        blist.append(now_ts)
                        self._scalp_budget[sym] = blist
                        # Shadow execute Scalp if ML is trained and score ≥ threshold
                        try:
                            from ml_scorer_scalp import get_scalp_scorer
                            scorer = get_scalp_scorer()
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
                            logger.info(f"[{sym}] 🧮 Scalp decision final: blocked (reason=unknown_path)")
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

        # Redis dedup
        dedup_ok = True
        try:
            from scalp_phantom_tracker import get_scalp_phantom_tracker
            scpt = get_scalp_phantom_tracker()
            r = scpt.redis_client
            if r is not None:
                key = f"{sym}:{sc_sig.side}:{round(float(sc_sig.entry),6)}:{round(float(sc_sig.sl),6)}:{round(float(sc_sig.tp),6)}"
                # TTL configurable
                try:
                    s_cfg = self.config.get('scalp', {}) if hasattr(self, 'config') else {}
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

        # One active phantom per symbol guard (fallback)
        try:
            if scpt.has_active(sym):
                logger.info(f"[{sym}] 🧮 Scalp decision final: blocked (reason=active_symbol)")
                return
        except Exception:
            pass

        # Build features and record phantom with pacing
        sc_meta = getattr(sc_sig, 'meta', {}) or {}
        try:
            vol_level = getattr(regime_analysis, 'volatility_level', 'normal') if regime_analysis else 'normal'
        except Exception:
            vol_level = 'normal'
        sc_feats = self._build_scalp_features(df_for_scalp, sc_meta, vol_level, cluster_id)
        sc_feats['routing'] = 'fallback'

        try:
            scpt = get_scalp_phantom_tracker()
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
                            logger.info(f"[{sym}] 🧮 Scalp decision final: phantom (reason=ok_fallback)")
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
                lines = [
                    f"{prefix} *{label} Phantom Opened*{pid_suffix}",
                    f"{symbol} {side} | ML {ml_score:.1f}",
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
            lines = [
                f"{prefix} *{label} Phantom {outcome_emoji}*{pid_suffix}",
                f"{symbol} {side} | ML {ml_score:.1f}",
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
                self.storage.save_all_frames(self.frames)
                logger.info("Auto-saved all candles to database")
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
                    message = (
                        f"{outcome_emoji} *Trade Closed* {symbol} {pos.side.upper()}\n\n"
                        f"Exit Price: {exit_price:.4f}\n"
                        f"PnL: ${pnl_usd:.2f} ({pnl_percent:.2f}%)\n"
                        f"{rr_line}"
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
                    # Force-close executed phantom mirrors to align with exchange closure (Trend + MR)
                    try:
                        if pos.strategy_name in ['enhanced_mr', 'mean_reversion'] and mr_phantom_tracker is not None:
                            mr_phantom_tracker.force_close_executed(symbol, exit_price, exit_reason)
                        elif pos.strategy_name == 'trend_breakout' and phantom_tracker is not None:
                            phantom_tracker.force_close_executed(symbol, exit_price, exit_reason)
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
                            signal_data = {
                                'symbol': symbol,
                                'features': feat_ref,
                                'score': 0,
                                'was_executed': True,
                                'meta': {
                                    'reason': getattr(pos, 'ml_reason', '')
                                }
                            }
                            
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
                                        shared_mr_scorer.record_outcome(signal_data, outcome, pnl_pct)
                                        logger.info(f"[{symbol}] ✅ Original MR ML updated with outcome (guarded).")
                                    else:
                                        logger.warning(f"[{symbol}] ⚠️ Mean Reversion outcome not recorded (no original MR scorer active; guard preventing Enhanced MR increment)")
                                elif pos.strategy_name == "unknown":
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
                                            shared_mr_scorer.record_outcome(signal_data, outcome, pnl_pct)
                                            logger.info(f"[{symbol}] ✅ Original MR ML updated with outcome (recovered, inferred).")
                                        else:
                                            logger.warning(f"[{symbol}] ⚠️ Inferred MR outcome not recorded (no original MR scorer; guard active)")
                                    else:
                                        # Default to Trend ML
                                        if ml_scorer is not None:
                                            ml_scorer.record_outcome(signal_data, outcome, pnl_pct)
                                            logger.info(f"[{symbol}] ⚠️ Trend ML updated with outcome (recovered position, defaulted).")
                                else:
                                    # Trend strategy
                                    logger.info(f"[{symbol}] 🔵 TREND STRATEGY detected")
                                    if ml_scorer is not None:
                                        ml_scorer.record_outcome(signal_data, outcome, pnl_pct)
                                        logger.info(f"[{symbol}] Trend ML updated with outcome.")
                            else:
                                # Original system - record in appropriate ML scorer
                                if 'skip_ml_update_manual' not in locals():
                                    if pos.strategy_name == "mean_reversion" and shared_mr_scorer:
                                        if mr_promo_flag:
                                            logger.info(f"[{symbol}] MR Promotion was active — skipping executed MR ML update (phantom path will learn)")
                                            try:
                                                if mr_phantom_tracker is not None:
                                                    mr_phantom_tracker.update_mr_phantom_prices(symbol, exit_price, df=self.frames.get(symbol))
                                            except Exception:
                                                pass
                                        else:
                                            shared_mr_scorer.record_outcome(signal_data, outcome, pnl_pct)
                                            logger.info(f"[{symbol}] Mean Reversion ML updated with outcome.")
                                    elif ml_scorer is not None:
                                        ml_scorer.record_outcome(signal_data, outcome, pnl_pct)
                                        logger.info(f"[{symbol}] Trend ML updated with outcome.")

                            
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
                    # Try to restore strategy from Redis if available
                    recovered_strategy = "unknown"
                    try:
                        if getattr(self, '_redis', None) is not None:
                            v = self._redis.get(f'openpos:strategy:{symbol}')
                            if isinstance(v, str) and v:
                                recovered_strategy = v
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
                    
                    recovered += 1
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
                        for sym, pos in book.positions.items():
                            emoji = "🟢" if pos.side == "long" else "🔴"
                            msg += f"{emoji} {sym}: {pos.side} qty={pos.qty:.4f}\n"
                        await self.tg.send_message(msg)
            else:
                logger.info("No existing positions to recover")
                
        except Exception as e:
            logger.error(f"Failed to recover positions: {e}")
    
    async def kline_stream(self, ws_url:str, topics:list[str]):
        """Stream klines from Bybit WebSocket"""
        sub = {"op":"subscribe","args":[f"kline.{t}" for t in topics]}
        
        while self.running:
            try:
                logger.info(f"Connecting to WebSocket: {ws_url}")
                async with websockets.connect(ws_url, ping_interval=20, ping_timeout=10) as ws:
                    self.ws = ws
                    await ws.send(json.dumps(sub))
                    logger.info(f"Subscribed to topics: {topics}")
                    
                    while self.running:
                        try:
                            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
                            
                            if msg.get("success") == False:
                                logger.error(f"Subscription failed: {msg}")
                                continue
                                
                            topic = msg.get("topic","")
                            if topic.startswith("kline."):
                                sym = topic.split(".")[-1]
                                for k in msg.get("data", []):
                                    yield sym, k
                                    
                        except asyncio.TimeoutError:
                            logger.debug("WebSocket timeout, sending ping")
                            await ws.ping()
                        except websockets.exceptions.ConnectionClosed:
                            logger.warning("WebSocket connection closed, reconnecting...")
                            break
                            
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                if self.running:
                    await asyncio.sleep(5)  # Wait before reconnecting

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
                ddh = (cfg.get('scalp', {}) or {}).get('dedup_hours', 'n/a')
                cdb = (cfg.get('scalp', {}) or {}).get('cooldown_bars', 'n/a')
                dbg_force = bool(((cfg.get('scalp', {}).get('debug', {}) or {}).get('force_accept', False)) if isinstance(cfg.get('scalp', {}), dict) else False)
            except Exception:
                hb = cap_none = ddh = cdb = 'n/a'; dbg_force = False
            logger.info(
                f"🔎 Startup fingerprint: central_router={central_enabled} scalp.hourly={hb} scalp.daily_none={cap_none} dedup_hours={ddh} cooldown_bars={cdb} debug.force_accept={dbg_force} backstop=ON"
            )
        except Exception:
            pass
        
        # Import strategies for parallel system
        from strategy_mean_reversion import detect_signal as detect_signal_mean_reversion
        from strategy_trend_breakout import detect_signal as detect_trend_signal, TrendSettings as TrendSettingsTB
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
            strategy_type = "Enhanced Parallel (Trend + Mean Reversion with ML)"
            logger.info(f"📊 Strategy: {strategy_type}")
            logger.info("🧠 Using Enhanced Parallel ML System with regime-based strategy routing")
        else:
            strategy_type = "Trend Breakout"
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
            confirmation_candles=cfg["trade"].get("confirmation_candles", 2)
        )
        # Trend breakout settings
        tr_cfg = cfg.get('trend', {}) or {}
        trend_settings = TrendSettingsTB(
            channel_len=int(tr_cfg.get('channel_len', 20)),
            atr_len=int(tr_cfg.get('atr_len', 14)),
            breakout_k_atr=float(tr_cfg.get('breakout_k_atr', 0.3)),
            sl_atr_mult=float(tr_cfg.get('sl_atr_mult', 1.5)),
            rr=float(tr_cfg.get('rr', 2.5)),
            use_ema_stack=bool(tr_cfg.get('use_ema_stack', True)),
            require_range_expansion=bool(tr_cfg.get('explore', {}).get('require_range_expansion', True)),
            range_expansion_min=float(tr_cfg.get('explore', {}).get('range_expansion_min', 1.2)),
            require_retest=bool(tr_cfg.get('explore', {}).get('require_retest', True)),
            retest_max_dist_atr=float(tr_cfg.get('explore', {}).get('retest_max_dist_atr', 0.5)),
        )

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
        
        if ML_AVAILABLE and use_ml:
            try:
                if use_enhanced_parallel and ENHANCED_ML_AVAILABLE:
                    # Initialize Enhanced Parallel ML System
                    ml_scorer = get_trend_scorer() if ('get_trend_scorer' in globals() and get_trend_scorer is not None) else None  # Trend ML
                    phantom_tracker = get_phantom_tracker()  # Phantom tracker
                    enhanced_mr_scorer = get_enhanced_mr_scorer()  # Enhanced MR ML
                    mr_phantom_tracker = get_mr_phantom_tracker()  # MR phantom tracker
                    mean_reversion_scorer = None  # Not used in enhanced system

                    # Get and log stats for both systems
                    trend_stats = ml_scorer.get_stats() if ml_scorer is not None else {'status':'N/A','current_threshold':0,'completed_trades':0,'recent_win_rate':0}
                    mr_stats = enhanced_mr_scorer.get_enhanced_stats()

                    logger.info(f"✅ Enhanced Parallel ML System initialized")
                    logger.info(f"   Trend ML: {trend_stats['status']} (threshold: {trend_stats['current_threshold']:.0f}, trades: {trend_stats['completed_trades']})")
                    logger.info(f"   Mean Reversion ML: {mr_stats['status']} (threshold: {mr_stats['current_threshold']:.0f}, trades: {mr_stats['completed_trades']})")

                    if trend_stats['recent_win_rate'] > 0:
                        logger.info(f"   Trend recent WR: {trend_stats['recent_win_rate']:.1f}%")
                    if mr_stats['recent_win_rate'] > 0:
                        logger.info(f"   MR recent WR: {mr_stats['recent_win_rate']:.1f}%")

                    # Perform Enhanced MR startup retrain with phantom data
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

                else:
                    # Initialize original ML system
                    ml_scorer = get_immediate_scorer()
                    phantom_tracker = get_phantom_tracker()
                    enhanced_mr_scorer = None
                    mr_phantom_tracker = None
                    mean_reversion_scorer = get_mean_reversion_scorer() # Original MR scorer

                    # Get and log ML stats
                    ml_stats = ml_scorer.get_stats()
                    logger.info(f"✅ Original ML Scorer initialized")
                    logger.info(f"   Status: {ml_stats['status']}")
                    logger.info(f"   Threshold: {ml_stats['current_threshold']:.0f}")
                    logger.info(f"   Completed trades: {ml_stats['completed_trades']}")
                    if ml_stats['recent_win_rate'] > 0:
                        logger.info(f"   Recent win rate: {ml_stats['recent_win_rate']:.1f}%")
                    if ml_stats['models_active']:
                        logger.info(f"   Active models: {', '.join(ml_stats['models_active'])}")
                    
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
                    from scalp_phantom_tracker import get_scalp_phantom_tracker
                    scpt = get_scalp_phantom_tracker()
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
                    "🚀 *Trading Bot Started*\n\n"
                    f"📊 Monitoring: {len(symbols)} symbols\n"
                    f"⏰ Timeframe: {tf} minutes\n"
                    f"💰 Risk per trade: {risk_display}\n"
                    f"📈 R:R Ratio: 1:{settings.rr}\n\n"
                    "_Use /risk to manage risk settings_\n"
                    "_Use /dashboard for full status_"
                )

                if phantom_tracker and hasattr(phantom_tracker, 'set_notifier'):
                    def trend_notifier(trade, scope='Trend'):
                        self._create_task(self._notify_phantom_trade(scope, trade))
                    phantom_tracker.set_notifier(trend_notifier)

                if mr_phantom_tracker and hasattr(mr_phantom_tracker, 'set_notifier'):
                    def mr_notifier(trade, scope='Mean Reversion'):
                        self._create_task(self._notify_phantom_trade(scope, trade))
                    mr_phantom_tracker.set_notifier(mr_notifier)
                # One-time backfill of MR phantom outcomes into Enhanced MR ML (avoid duplicates via Redis flag)
                try:
                    if enhanced_mr_scorer is not None:
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
                        if hasattr(scpt, 'set_notifier'):
                            def scalp_notifier(trade, scope='Scalp'):
                                self._create_task(self._notify_phantom_trade(scope, trade))
                            scpt.set_notifier(scalp_notifier)
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
                        # One-time backfill of Scalp phantom outcomes into Scalp ML (avoid duplicates via Redis flag)
                        try:
                            s_scorer = get_scalp_scorer() if (SCALP_AVAILABLE and get_scalp_scorer is not None) else None
                            backfilled = False
                            if s_scorer and getattr(s_scorer, 'redis_client', None):
                                try:
                                    if s_scorer.redis_client.get('ml:backfill:scalp:done') == '1':
                                        backfilled = True
                                except Exception:
                                    pass
                            if s_scorer and not backfilled:
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
                                    if s_scorer and getattr(s_scorer, 'redis_client', None):
                                        s_scorer.redis_client.set('ml:backfill:scalp:done', '1')
                                except Exception:
                                    pass
                                if fed > 0:
                                    logger.info(f"🩳 Scalp ML backfill: fed {fed} phantom outcomes into ML store")
                                # Attempt a startup retrain after backfill if trainable
                                try:
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
                            # Update both phantom trackers in parallel system
                            phantom_tracker.update_phantom_prices(
                                sym, current_price, df=df, btc_price=btc_price, symbol_collector=symbol_collector
                            )
                            mr_phantom_tracker.update_mr_phantom_prices(sym, current_price, df=df)
                            # Scalp phantom tracker price updates (if available)
                            try:
                                from scalp_phantom_tracker import get_scalp_phantom_tracker
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
                
                    # Auto-save to database every 15 minutes
                    if (datetime.now() - self.last_save_time).total_seconds() > 900:
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
                    central_router_enabled = False
                    try:
                        cr = (self.config.get('router', {}) or {})
                        central_router_enabled = bool(cr.get('central_enabled', False))
                    except Exception:
                        central_router_enabled = False
                    if central_router_enabled:
                        try:
                            from strategy_regimes import score_trend_regime, score_mr_regime
                            tr_score, _ = score_trend_regime(df)
                            mr_score, _ = score_mr_regime(df)
                        except Exception:
                            pass

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
                            candidates.append(('trend_breakout', tr_score))
                        if mr_score >= mr_thr:
                            candidates.append(('enhanced_mr', mr_score))

                    # Apply HTF gating to candidates when enabled (drop misaligned). Use composite RC/TS.
                    if central_router_enabled:
                        try:
                            htf_cfg = (self.config.get('router', {}) or {}).get('htf_bias', {})
                            mode = str(htf_cfg.get('mode', 'gated')).lower()
                            if bool(htf_cfg.get('enabled', False)):
                                metrics = self._get_htf_metrics(sym, df)
                                min_ts = float((htf_cfg.get('trend', {}) or {}).get('min_trend_strength', 60.0))
                                min_rq = float((htf_cfg.get('mr', {}) or {}).get('min_range_quality', 0.60))
                                max_ts = float((htf_cfg.get('mr', {}) or {}).get('max_trend_strength', 40.0))
                                gated = []
                                for name, sc in candidates:
                                    if name == 'trend_breakout':
                                        ts_ok = (metrics['ts15'] >= min_ts) and ((metrics['ts60'] == 0.0) or (metrics['ts60'] >= min_ts))
                                        if ts_ok or mode == 'soft':
                                            gated.append((name, sc))
                                        else:
                                            logger.info(f"[{sym}] 🧮 HTF gate: drop TREND ts15={metrics['ts15']:.1f} ts60={metrics['ts60']:.1f} < {min_ts:.1f}")
                                    elif name == 'enhanced_mr':
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
                        if prev_route in ('trend_breakout', 'enhanced_mr'):
                            # Hold previous for min_hold
                            hold = pb_hold if prev_route == 'trend_breakout' else mr_hold
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
                            if shared_mr_scorer_log:
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
                        independence_enabled = bool(indep_cfg.get('enabled', False))
                    except Exception:
                        independence_enabled = False

                    if independence_enabled and ENHANCED_ML_AVAILABLE:
                        # Take a regime snapshot once for per-strategy filters
                        try:
                            regime_analysis = get_enhanced_market_regime(df, sym)
                        except Exception:
                            regime_analysis = None

                        # Helper to attempt execution for a given signal; returns True if executed
                        async def _try_execute(strategy_name: str, sig_obj, ml_score: float = 0.0, threshold: float = 75.0):
                            nonlocal book, bybit, sizer
                            dbg_logger = logging.getLogger(__name__)
                            # Optional 3m context enforcement (applies to both Trend and MR execution paths)
                            try:
                                if strategy_name == 'trend_breakout':
                                    ctx_cfg = (cfg.get('trend', {}).get('context', {}) or {}) if 'cfg' in locals() else {}
                                    if bool(ctx_cfg.get('enforce', False)):
                                        ok3, why3 = self._micro_context_trend(sym, sig_obj.side)
                                        if not ok3:
                                            dbg_logger.debug(f"[{sym}] Trend: skip — 3m_ctx_enforce ({why3})")
                                            return False
                                elif strategy_name == 'enhanced_mr':
                                    ctx_cfg = (cfg.get('mr', {}).get('context', {}) or {}) if 'cfg' in locals() else {}
                                    if bool(ctx_cfg.get('enforce', False)):
                                        ok3, why3 = self._micro_context_mr(sym, sig_obj.side)
                                        if not ok3:
                                            dbg_logger.debug(f"[{sym}] MR: skip — 3m_ctx_enforce ({why3})")
                                            return False
                                elif strategy_name == 'scalp':
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
                                            if strategy_name == 'trend_breakout':
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
                                            if strategy_name == 'trend_breakout':
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
                            # Set TP/SL
                            bybit.set_tpsl(sym, take_profit=sig_obj.tp, stop_loss=sig_obj.sl, qty=qty)
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
                            book.positions[sym] = Position(
                                sig_obj.side,
                                qty,
                                entry=actual_entry,
                                sl=sig_obj.sl,
                                tp=sig_obj.tp,
                                entry_time=datetime.now(),
                                strategy_name=strategy_name
                            )
                            # Optionally cancel active phantoms on execution (config)
                            try:
                                ph_cfg = cfg.get('phantom', {}) if 'cfg' in locals() else {}
                                if bool(ph_cfg.get('cancel_on_execute', False)):
                                    if strategy_name == 'enhanced_mr' and mr_phantom_tracker:
                                        mr_phantom_tracker.cancel_active(sym)
                                    elif strategy_name == 'trend_breakout' and phantom_tracker:
                                        phantom_tracker.cancel_active(sym)
                            except Exception:
                                pass
                            # Notify
                            try:
                                if self.tg:
                                    emoji = '📈'
                                    strategy_label = 'Enhanced Mr' if strategy_name=='enhanced_mr' else 'Trend Breakout'
                                    msg = (
                                        f"{emoji} *{sym} {sig_obj.side.upper()}* ({strategy_label})\n\n"
                                        f"Entry: {actual_entry:.4f}\n"
                                        f"Stop Loss: {sig_obj.sl:.4f}\n"
                                        f"Take Profit: {sig_obj.tp:.4f}\n"
                                        f"Quantity: {qty}\n"
                                        f"Risk: {risk.risk_percent if risk.use_percent_risk else risk.risk_usd}{'%' if risk.use_percent_risk else ''} (${risk_amount:.2f})\n"
                                    )
                                    await self.tg.send_message(msg)
                            except Exception:
                                pass
                            # Decision final (independence branch)
                            try:
                                promo = bool(getattr(sig_obj, 'meta', {}) and sig_obj.meta.get('promotion_forced')) if isinstance(getattr(sig_obj, 'meta', {}), dict) else False
                                logger.info(f"[{sym}] 🧮 Decision final: exec_{'mr' if strategy_name=='enhanced_mr' else 'trend'} (reason={'promotion' if promo else 'ok'})")
                            except Exception:
                                pass
                            return True

                        # (Reverted) Stream-side Scalp promotion execution queue removed; main loop handles promotion only

                        # 1) Mean Reversion independent
                        try:
                            logger.debug(f"[{sym}] MR: analysis start")
                            sig_mr_ind = detect_signal_mean_reversion(df.copy(), settings, sym)
                        except Exception:
                            sig_mr_ind = None
                        if sig_mr_ind is None:
                            logger.debug(f"[{sym}] MR: skip — no_signal")
                        if sig_mr_ind is not None:
                            ef = sig_mr_ind.meta.get('mr_features', {}) if sig_mr_ind.meta else {}
                            ml_score_mr = 0.0; thr_mr = 75.0; mr_should = True
                            try:
                                if enhanced_mr_scorer:
                                    ml_score_mr, _ = enhanced_mr_scorer.score_signal(sig_mr_ind.__dict__, ef, df)
                                    thr_mr = getattr(enhanced_mr_scorer, 'min_score', 75)
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
                                if ml_score_mr >= mr_hi_force:
                                    try:
                                        if not sig_mr_ind.meta:
                                            sig_mr_ind.meta = {}
                                    except Exception:
                                        sig_mr_ind.meta = {}
                                    sig_mr_ind.meta['promotion_forced'] = True
                                    executed = await _try_execute('enhanced_mr', sig_mr_ind, ml_score=ml_score_mr, threshold=thr_mr)
                                    if executed:
                                        try:
                                            logger.info(f"[{sym}] 🧮 Decision final: exec_mr (reason=ml_extreme {ml_score_mr:.1f}>={mr_hi_force:.0f})")
                                        except Exception:
                                            pass
                                        continue
                                    else:
                                        try:
                                            logger.info(f"[{sym}] 🛑 MR High-ML override blocked: reason=exec_guard")
                                        except Exception:
                                            pass
                                # Promotion override regardless of earlier guards
                                prom_cfg = (self.config.get('mr', {}) or {}).get('promotion', {})
                                promote_wr = float(prom_cfg.get('promote_wr', 50.0))
                                mr_stats = enhanced_mr_scorer.get_enhanced_stats() if enhanced_mr_scorer else {}
                                recent_wr = float(mr_stats.get('recent_win_rate', 0.0))
                                if recent_wr >= promote_wr:
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
                                            continue
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
                                if not mr_pass_regime and not (promotion_bypass and recent_wr >= promote_wr):
                                    logger.debug(f"[{sym}] MR: skip — regime gate (prim={prim}, conf={conf:.2f}, persist={persist:.2f})")
                                    if mr_phantom_tracker:
                                        mr_phantom_tracker.record_mr_signal(sym, sig_mr_ind.__dict__, float(ml_score_mr or 0.0), False, {}, ef)
                                    sig_mr_ind = None
                            except Exception:
                                pass
                            if mr_should and sig_mr_ind is not None:
                                executed = await _try_execute('enhanced_mr', sig_mr_ind, ml_score=ml_score_mr, threshold=thr_mr)
                                if not executed and mr_phantom_tracker and sig_mr_ind is not None:
                                    logger.debug(f"[{sym}] MR: skip — execution guard (see prior logs)")
                                    # Record phantom when position exists or execution not possible
                                    mr_phantom_tracker.record_mr_signal(sym, sig_mr_ind.__dict__, float(ml_score_mr or 0.0), False, {}, ef)
                                    try:
                                        logger.info(f"[{sym}] 🧮 Decision final: phantom_mr (reason=exec_guard)")
                                    except Exception:
                                        pass
                                    try:
                                        if self.flow_controller and self.flow_controller.enabled:
                                            self.flow_controller.increment_accepted('mr', 1)
                                    except Exception:
                                        pass
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
                            logger.debug(f"[{sym}] Trend: analysis start")
                            sig_tr_ind = detect_trend_signal(df.copy(), trend_settings, sym)
                        except Exception:
                            sig_tr_ind = None
                        if sig_tr_ind is None:
                            logger.debug(f"[{sym}] Trend: skip — no_signal")
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
                                trend_features = {
                                    'trend_slope_pct': trend_slope_pct,
                                    'ema_stack_score': ema_stack_score,
                                    'atr_pct': atr_pct,
                                    'range_expansion': range_expansion,
                                    'breakout_dist_atr': float(getattr(sig_tr_ind, 'meta', {}).get('breakout_dist_atr', 0.0) if getattr(sig_tr_ind, 'meta', None) else 0.0),
                                    'close_vs_ema20_pct': close_vs_ema20_pct,
                                    'bb_width_pct': 0.0,
                                    'session': 'us',
                                    'symbol_cluster': 3,
                                    'volatility_regime': 'normal'
                                }
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
                                    tr_should = ml_score_tr >= thr_tr
                                    # Extreme ML override (force execution bypassing router/regime/micro gates)
                                    try:
                                        tr_hi_force = float((((cfg.get('trend', {}) or {}).get('exec', {}) or {}).get('high_ml_force', 92.0)))
                                    except Exception:
                                        tr_hi_force = 92.0
                                    if ml_score_tr >= tr_hi_force:
                                        try:
                                            if not sig_tr_ind.meta:
                                                sig_tr_ind.meta = {}
                                        except Exception:
                                            sig_tr_ind.meta = {}
                                        sig_tr_ind.meta['promotion_forced'] = True
                                        executed = await _try_execute('trend_breakout', sig_tr_ind, ml_score=ml_score_tr, threshold=thr_tr)
                                        if executed:
                                            try:
                                                logger.info(f"[{sym}] 🧮 Decision final: exec_trend (reason=ml_extreme {ml_score_tr:.1f}>={tr_hi_force:.0f})")
                                            except Exception:
                                                pass
                                            continue
                                        else:
                                            try:
                                                logger.info(f"[{sym}] 🛑 Trend High-ML override blocked: reason=exec_guard")
                                            except Exception:
                                                pass
                                try:
                                    logger.debug(f"[{sym}] Trend: ml_score={float(ml_score_tr or 0.0):.1f} thr={thr_tr:.0f} should={tr_should}")
                                except Exception:
                                    pass
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
                                            logger.info(f"[{sym}] 🧮 Decision final: phantom_trend (reason=htf_gate ts15={metrics['ts15']:.1f} ts60={metrics['ts60']:.1f} < {min_ts:.1f})")
                                            if phantom_tracker:
                                                phantom_tracker.record_signal(sym, {'side': sig_tr_ind.side, 'entry': sig_tr_ind.entry, 'sl': sig_tr_ind.sl, 'tp': sig_tr_ind.tp}, float(ml_score_tr or 0.0), False, trend_features, 'trend_breakout')
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
                                        logger.info(f"[{sym}] 🧮 Decision final: phantom_trend (reason=micro_ctx {why3})")
                                        if phantom_tracker:
                                            phantom_tracker.record_signal(sym, {'side': sig_tr_ind.side, 'entry': sig_tr_ind.entry, 'sl': sig_tr_ind.sl, 'tp': sig_tr_ind.tp}, float(ml_score_tr or 0.0), False, trend_features, 'trend_breakout')
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

                            if (not tr_pass_regime) or (not trend_exec_enabled):
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
                                        executed = await _try_execute('trend_breakout', sig_tr_ind, ml_score=ml_score_tr or 0.0, threshold=thr_tr if 'thr_tr' in locals() else 70)
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
                                # Fall back to phantom record
                                if phantom_tracker and sig_tr_ind is not None:
                                    # Log regime/exec gate block
                                    try:
                                        logger.debug(f"[{sym}] Trend: skip — regime/exec gate (reg={tr_pass_regime}, exec={trend_exec_enabled})")
                                    except Exception:
                                        pass
                                    phantom_tracker.record_signal(sym, {'side': sig_tr_ind.side, 'entry': sig_tr_ind.entry, 'sl': sig_tr_ind.sl, 'tp': sig_tr_ind.tp}, float(ml_score_tr or 0.0), False, trend_features, 'trend_breakout')
                                    try:
                                        logger.info(f"[{sym}] 🧮 Decision final: phantom_trend (reason=regime/exec_gate)")
                                    except Exception:
                                        pass
                                    try:
                                        if self.flow_controller and self.flow_controller.enabled:
                                            self.flow_controller.increment_accepted('trend', 1)
                                    except Exception:
                                        pass
                            else:
                                if tr_should:
                                    executed = await _try_execute('trend_breakout', sig_tr_ind, ml_score=ml_score_tr, threshold=thr_tr)
                                    if not executed and phantom_tracker and sig_tr_ind is not None:
                                        logger.debug(f"[{sym}] Trend: skip — execution guard (see prior logs)")
                                        phantom_tracker.record_signal(sym, {'side': sig_tr_ind.side, 'entry': sig_tr_ind.entry, 'sl': sig_tr_ind.sl, 'tp': sig_tr_ind.tp}, float(ml_score_tr or 0.0), False, trend_features, 'trend_breakout')
                                        try:
                                            logger.info(f"[{sym}] 🧮 Decision final: phantom_trend (reason=exec_guard)")
                                        except Exception:
                                            pass
                                        try:
                                            if self.flow_controller and self.flow_controller.enabled:
                                                self.flow_controller.increment_accepted('trend', 1)
                                        except Exception:
                                            pass
                                else:
                                    if phantom_tracker:
                                        try:
                                            logger.debug(f"[{sym}] Trend: skip — ml<{thr_tr:.0f} (score {float(ml_score_tr or 0.0):.1f})")
                                        except Exception:
                                            pass
                                        # Only record phantom if ML ≥ trend.phantom.min_ml
                                        try:
                                            ph_min = float(((cfg.get('trend', {}) or {}).get('phantom', {}) or {}).get('min_ml', 0))
                                        except Exception:
                                            ph_min = 0.0
                                        if ml_score_tr >= ph_min:
                                            phantom_tracker.record_signal(sym, {'side': sig_tr_ind.side, 'entry': sig_tr_ind.entry, 'sl': sig_tr_ind.sl, 'tp': sig_tr_ind.tp}, float(ml_score_tr or 0.0), False, trend_features, 'trend_breakout')
                                            try:
                                                logger.info(f"[{sym}] 🧮 Decision final: phantom_trend (reason=ml<thr)")
                                            except Exception:
                                                pass
                                            try:
                                                if self.flow_controller and self.flow_controller.enabled:
                                                    self.flow_controller.increment_accepted('trend', 1)
                                            except Exception:
                                                pass
                                        else:
                                            try:
                                                logger.info(f"[{sym}] 🧮 Decision final: skip_trend_phantom (reason=ml<trend_phantom_min {ml_score_tr:.1f}<{ph_min:.0f})")
                                            except Exception:
                                                pass

                        # 3) Scalp independent (promotion-only execution)
                        try:
                            s_cfg = cfg.get('scalp', {}) if 'cfg' in locals() else {}
                            promote_en = bool(s_cfg.get('promote_enabled', False))
                            if promote_en and SCALP_AVAILABLE and detect_scalp_signal is not None:
                                # Promotion readiness from phantom stats (supports recent WR)
                                try:
                                    from scalp_phantom_tracker import get_scalp_phantom_tracker
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
                    selected_strategy = "trend_breakout"  # Default
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
                                chosen = ('trend_breakout', soft_sig_tr)
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
                                    if soft_sig_tr and tp.get('active') and int(tp.get('count', 0)) < cap and allow_tr:
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
                                            executed = await _try_execute('trend_breakout', soft_sig_tr, ml_score=tr_score if 'tr_score' in locals() and tr_score is not None else 0.0, threshold=tr_thr if 'tr_thr' in locals() else 70)
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
                                            book.positions[sym] = Position(
                                                soft_sig_mr.side,
                                                qty,
                                                entry=actual_entry,
                                                sl=soft_sig_mr.sl,
                                                tp=soft_sig_mr.tp,
                                                entry_time=datetime.now(),
                                                strategy_name='enhanced_mr'
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
                                        phantom_tracker.record_signal(sym, {'side': soft_sig_tr.side, 'entry': soft_sig_tr.entry, 'sl': soft_sig_tr.sl, 'tp': soft_sig_tr.tp}, float(tr_score or 0.0), False, trend_features, 'trend_breakout')
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
                            if keep_prev and prev_state and prev_state.get('strategy') in ('trend_breakout','enhanced_mr'):
                                logger.info(f"[{sym}] 🧭 Hysteresis: keeping previous route {prev_state.get('strategy')}")
                                if prev_state.get('strategy') == 'enhanced_mr':
                                    selected_strategy = 'enhanced_mr'
                                    selected_ml_scorer = enhanced_mr_scorer
                                    selected_phantom_tracker = mr_phantom_tracker
                                else:
                                    selected_strategy = 'trend_breakout'
                                    selected_ml_scorer = get_trend_scorer() if 'get_trend_scorer' in globals() and get_trend_scorer is not None else None
                                    selected_phantom_tracker = phantom_tracker
                            # Else start with recommended and optionally override later after signal detection

                        # Router override using per-strategy regime scores
                        if (not handled) and router_choice in ("enhanced_mr", "pullback", "trend_breakout"):
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
                                # Backward-compat: treat legacy 'pullback' route as Trend Breakout
                                logger.debug(f"🔵 [{sym}] ROUTER OVERRIDE → TREND BREAKOUT ANALYSIS (legacy pullback route)")
                                sig = detect_trend_signal(df.copy(), trend_settings, sym)
                                selected_strategy = "trend_breakout"
                                selected_ml_scorer = get_trend_scorer() if 'get_trend_scorer' in globals() and get_trend_scorer is not None else None
                                selected_phantom_tracker = phantom_tracker
                                try:
                                    routing_state[sym] = {'strategy': selected_strategy, 'last_idx': candles_processed}
                                    shared['routing_state'] = routing_state
                                except Exception:
                                    pass
                                handled = True
                            else:
                                logger.debug(f"🟣 [{sym}] ROUTER OVERRIDE → TREND BREAKOUT ANALYSIS:")
                                sig = detect_trend_signal(df.copy(), trend_settings, sym)
                                selected_strategy = "trend_breakout"
                                selected_ml_scorer = get_trend_scorer() if 'get_trend_scorer' in globals() and get_trend_scorer is not None else None
                                selected_phantom_tracker = phantom_tracker
                                try:
                                    routing_state[sym] = {'strategy': selected_strategy, 'last_idx': candles_processed}
                                    shared['routing_state'] = routing_state
                                except Exception:
                                    pass
                                handled = True

                        if (not handled) and selected_strategy == "trend_breakout":
                            # Trend breakout strategy scoring and gating
                            if sig is None:
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
                                        if should_take_trade:
                                            logger.info(f"   ✅ DECISION: EXECUTE TRADE - Trend ML {ml_score:.1f} ≥ {threshold}")
                                        else:
                                            logger.info(f"   ❌ DECISION: REJECT TRADE - Trend ML {ml_score:.1f} < {threshold}")
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
                                # Record phantom for trend before continue if not executing
                                logger.info(f"[{sym}] 📊 PHANTOM ROUTING: Trend phantom tracker recording (executed={should_take_trade})")
                                phantom_tracker.record_signal(
                                    symbol=sym,
                                    signal={'side': sig.side, 'entry': sig.entry, 'sl': sig.sl, 'tp': sig.tp},
                                    ml_score=float(ml_score or 0.0),
                                    was_executed=should_take_trade,
                                    features=trend_features,
                                    strategy_name='trend_breakout'
                                )
                                if not should_take_trade:
                                    # Skip execution, continue to next symbol
                                    continue
                            else:
                                logger.info(f"   ❌ No Trend Signal: Breakout conditions not met")

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
                            # Treat legacy pullback recommendation as Trend Breakout
                            logger.debug(f"🔵 [{sym}] TREND BREAKOUT ANALYSIS (legacy pullback route):")
                            sig = detect_trend_signal(df.copy(), trend_settings, sym)
                            selected_strategy = "trend_breakout"
                            selected_ml_scorer = get_trend_scorer() if 'get_trend_scorer' in globals() and get_trend_scorer is not None else None
                            selected_phantom_tracker = phantom_tracker

                            if sig:
                                logger.info(f"   ✅ Trend Signal Detected: {sig.side.upper()} at {sig.entry:.4f}")
                                logger.info(f"   🎯 SL: {sig.sl:.4f} | TP: {sig.tp:.4f} | R:R: {((sig.tp-sig.entry)/(sig.entry-sig.sl) if sig.side=='long' else (sig.entry-sig.tp)/(sig.sl-sig.entry)):.2f}")
                                logger.info(f"   📝 Reason: {sig.reason}")
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
                                            mr_phantom_tracker.record_mr_signal(sym, sig_mr.__dict__, 0.0, False, {}, ef)
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
                                            logger.info(f"[{sym}] 👻 Trend explore skip: {', '.join(reasons) if reasons else 'gate fail'}")
                                        else:
                                            logger.info(f"[{sym}] 👻 Phantom-only (Trend none): {sig_tr.side.upper()} @ {sig_tr.entry:.4f}")
                                            if phantom_tracker:
                                                phantom_tracker.record_signal(
                                                    symbol=sym,
                                                    signal={'side': sig_tr.side, 'entry': sig_tr.entry, 'sl': sig_tr.sl, 'tp': sig_tr.tp},
                                                    ml_score=0.0,
                                                    was_executed=False,
                                                    features=trend_features,
                                                    strategy_name='trend_breakout'
                                                )
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
                                            sc_feats['routing'] = 'none'
                                            logger.info(f"[{sym}] 👻 Phantom-only (Scalp none): {sc_sig.side.upper()} @ {sc_sig.entry:.4f}")
                                            try:
                                                from scalp_phantom_tracker import get_scalp_phantom_tracker
                                                scpt = get_scalp_phantom_tracker()
                                                # Compute ML for router-driven scalp phantom
                                                ml_s = 0.0
                                                try:
                                                    from ml_scorer_scalp import get_scalp_scorer
                                                    _scorer = get_scalp_scorer()
                                                    ml_s, _ = _scorer.score_signal({'side': sc_sig.side, 'entry': sc_sig.entry, 'sl': sc_sig.sl, 'tp': sc_sig.tp}, sc_feats)
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
                            selected_strategy = "trend_breakout"

                    else:
                        # Default trend strategy
                        sig = detect_trend_signal(df.copy(), trend_settings, sym)
                        selected_strategy = "trend_breakout"
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
                                    if not should_take_trade:
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
                                    if self.tg:
                                        reject_msg = (
                                            f"🤖 *ML Reject* {sym} {sig.side.upper()} (Enhanced MR)\n"
                                            f"Score {ml_score:.1f} < {threshold:.0f}. Tracking via phantom."
                                        )
                                        await self.tg.send_message(reject_msg)
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
                                    if self.tg:
                                        reject_msg = (
                                            f"🤖 *ML Reject* {sym} {sig.side.upper()} (Trend)\n"
                                            f"Score {ml_score:.1f} < {threshold:.0f}. Tracking via phantom."
                                        )
                                        await self.tg.send_message(reject_msg)
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
                                    # Trend phantom record
                                    selected_phantom_tracker.record_signal(
                                        symbol=sym,
                                        signal={'side': sig.side, 'entry': sig.entry, 'sl': sig.sl, 'tp': sig.tp},
                                        ml_score=float(ml_score or 0.0),
                                        was_executed=False,
                                        features=basic_features if 'basic_features' in locals() else {},
                                        strategy_name=selected_strategy
                                    )
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
                    if self.tg:
                        await self.tg.send_message(f"❌ {sym} SHORT rejected: SL {sig.sl:.4f} too low (current: {current_price:.4f})")
                    continue
                elif sig.side == "long" and sig.sl >= current_price:
                    logger.error(f"[{sym}] Invalid SL for LONG: {sig.sl:.4f} must be < current price {current_price:.4f}")
                    if self.tg:
                        await self.tg.send_message(f"❌ {sym} LONG rejected: SL {sig.sl:.4f} too high (current: {current_price:.4f})")
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
                                    if selected_strategy == 'trend_breakout':
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
                        strategy_name=selected_strategy,
                        ml_score=float(ml_score),
                        ml_reason=ml_reason if isinstance(ml_reason, str) else ""
                    )
                    # For Trend: if slippage caused a SL recalc earlier, ensure TP/SL have been read back and adjusted; otherwise, adjust here too
                    try:
                        if selected_strategy == 'trend_breakout':
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
                        if selected_strategy == 'trend_breakout' and phantom_tracker is not None:
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
                            except Exception:
                                trend_features = {}
                            phantom_tracker.record_signal(
                                sym,
                                {'side': sig.side, 'entry': float(actual_entry), 'sl': float(sig.sl), 'tp': float(sig.tp), 'meta': getattr(sig, 'meta', {}) or {}},
                                float(ml_score or 0.0),
                                False if is_promo else True,
                                trend_features,
                                'trend_breakout'
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
                            elif selected_strategy in ('trend_breakout','pullback') and phantom_tracker:
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
                        if selected_strategy == 'trend_breakout':
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

                        strategy_label = selected_strategy.replace('_', ' ').title()
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
