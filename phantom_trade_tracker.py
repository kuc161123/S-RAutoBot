"""
Phantom Trade Tracker for ML Learning
Tracks ALL signals (including rejected ones) to see their hypothetical outcomes
This allows ML to learn from the full spectrum of trading opportunities
"""
import json
import logging
import os
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import uuid
from typing import Callable, Dict, List, Optional, Tuple

import asyncio
import numpy as np
import redis
import yaml
from position_mgr import round_step

logger = logging.getLogger(__name__)

class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types"""
    def default(self, obj):
        if isinstance(obj, (np.bool_, np.bool8)):
            return bool(obj)
        elif isinstance(obj, (np.integer, np.int_, np.intc, np.intp, 
                            np.int8, np.int16, np.int32, np.int64,
                            np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float_, np.float16, 
                            np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

@dataclass
class PhantomTrade:
    """A potential trade that was analyzed but not necessarily executed"""
    symbol: str
    side: str  # "long" or "short"
    entry_price: float
    stop_loss: float
    take_profit: float
    signal_time: datetime
    ml_score: float  # ML score assigned to this signal
    was_executed: bool  # Whether this became a real trade
    features: Dict  # All ML features for this signal
    strategy_name: str = "unknown" # The strategy that generated the signal
    phantom_id: str = ""  # Unique ID for concurrent phantom distinction
    
    # Outcome tracking (filled in later)
    outcome: Optional[str] = None  # "win", "loss", or "active"
    exit_price: Optional[float] = None
    pnl_percent: Optional[float] = None
    exit_time: Optional[datetime] = None
    max_favorable: Optional[float] = None  # Best price reached
    max_adverse: Optional[float] = None  # Worst price reached
    # Enriched labels
    one_r_hit: Optional[bool] = None
    two_r_hit: Optional[bool] = None
    realized_rr: Optional[float] = None
    exit_reason: Optional[str] = None
    # Lifecycle flags
    be_moved: bool = False
    tp1_hit: bool = False
    
    def to_dict(self):
        """Convert to dictionary for storage"""
        import numpy as np
        d = asdict(self)
        
        # Convert numpy types to Python native types for JSON serialization
        for key, value in d.items():
            if isinstance(value, (np.bool_, np.bool8)):
                d[key] = bool(value)
            elif isinstance(value, (np.integer, np.floating)):
                d[key] = float(value)
            elif isinstance(value, dict):
                # Also check nested dictionaries (like features)
                cleaned_dict = {}
                for k, v in value.items():
                    if isinstance(v, (np.bool_, np.bool8)):
                        cleaned_dict[k] = bool(v)
                    elif isinstance(v, (np.integer, np.floating)):
                        cleaned_dict[k] = float(v)
                    else:
                        cleaned_dict[k] = v
                d[key] = cleaned_dict
        
        # Convert datetime fields
        d['signal_time'] = self.signal_time.isoformat()
        if self.exit_time:
            d['exit_time'] = self.exit_time.isoformat()
        return d
    
    @classmethod
    def from_dict(cls, d):
        """Create from dictionary"""
        d = d.copy()
        d['signal_time'] = datetime.fromisoformat(d['signal_time'])
        if d.get('exit_time'):
            d['exit_time'] = datetime.fromisoformat(d['exit_time'])
        return cls(**d)

class PhantomTradeTracker:
    """
    Tracks all trading signals to monitor their hypothetical outcomes
    Feeds this comprehensive data back to ML for improved learning
    """
    
    def __init__(self):
        self.redis_client = None
        self.phantom_trades: Dict[str, List[PhantomTrade]] = {}  # symbol -> list of completed phantoms
        # Allow multiple active phantoms per symbol
        self.active_phantoms: Dict[str, List[PhantomTrade]] = {}
        self.notifier: Optional[Callable] = None
        # Local blocked counters fallback: day (YYYYMMDD) -> counts
        self._blocked_counts: Dict[str, Dict[str, int]] = {}
        self._symbol_meta: Dict[str, Dict] = {}
        
        # Initialize Redis
        self._init_redis()
        self._load_from_redis()
        # Default timeout for trend phantom (can be overridden via config)
        self.timeout_hours: int = 36
        self._load_symbol_meta()

    def _load_symbol_meta(self):
        try:
            with open('config.yaml','r') as f:
                cfg = yaml.safe_load(f)
            sm = (cfg or {}).get('symbol_meta', {}) or {}
            if isinstance(sm, dict):
                self._symbol_meta = sm
        except Exception as e:
            logger.debug(f"PhantomTracker: failed to load symbol meta: {e}")

    def _tick_size_for(self, symbol: str) -> float:
        try:
            if symbol in self._symbol_meta and isinstance(self._symbol_meta[symbol], dict):
                return float(self._symbol_meta[symbol].get('tick_size', 0.000001) or 0.000001)
            default = self._symbol_meta.get('default', {}) if isinstance(self._symbol_meta, dict) else {}
            return float(default.get('tick_size', 0.000001) or 0.000001)
        except Exception:
            return 0.000001
    
    def _init_redis(self):
        """Initialize Redis connection"""
        try:
            redis_url = os.getenv('REDIS_URL')
            if redis_url:
                self.redis_client = redis.from_url(redis_url, decode_responses=True)
                self.redis_client.ping()
                logger.info("Phantom Tracker connected to Redis")
            else:
                logger.warning("No REDIS_URL for Phantom Tracker, using memory only")
        except Exception as e:
            logger.warning(f"Redis connection failed for Phantom Tracker: {e}")
            self.redis_client = None
    
    def _load_from_redis(self):
        """Load phantom trades from Redis"""
        if not self.redis_client:
            return
            
        try:
            # Load active phantoms (list per symbol)
            active_data = self.redis_client.get('phantom:active')
            if active_data:
                active_dict = json.loads(active_data)
                cnt = 0
                for symbol, rec in active_dict.items():
                    items: List[PhantomTrade] = []
                    if isinstance(rec, list):
                        for t in rec:
                            try:
                                items.append(PhantomTrade.from_dict(t))
                            except Exception:
                                pass
                    elif isinstance(rec, dict):
                        try:
                            items.append(PhantomTrade.from_dict(rec))
                        except Exception:
                            pass
                    if items:
                        self.active_phantoms[symbol] = items
                        cnt += len(items)
                logger.info(f"Loaded {cnt} active phantom trades across {len(self.active_phantoms)} symbols")
            
            # Load completed phantoms (last 1000)
            completed_data = self.redis_client.get('phantom:completed')
            if completed_data:
                completed_list = json.loads(completed_data)
                for trade_dict in completed_list:
                    phantom = PhantomTrade.from_dict(trade_dict)
                    if phantom.symbol not in self.phantom_trades:
                        self.phantom_trades[phantom.symbol] = []
                    self.phantom_trades[phantom.symbol].append(phantom)
                total = sum(len(trades) for trades in self.phantom_trades.values())
                logger.info(f"Loaded {total} completed phantom trades")
                
        except Exception as e:
            logger.error(f"Error loading phantom trades from Redis: {e}")
    
    def _save_to_redis(self):
        """Save phantom trades to Redis"""
        if not self.redis_client:
            return
            
        try:
            # Save active phantoms as lists per symbol
            active_dict = {}
            for symbol, trades in self.active_phantoms.items():
                active_dict[symbol] = [t.to_dict() for t in trades]
            self.redis_client.set('phantom:active', json.dumps(active_dict, cls=NumpyJSONEncoder))
            
            # Save completed phantoms (keep last 1000, and drop >30d old)
            from datetime import datetime, timedelta
            cutoff = datetime.now() - timedelta(days=30)
            all_completed = []
            for trades in self.phantom_trades.values():
                for trade in trades:
                    if trade.outcome in ['win', 'loss']:
                        try:
                            if trade.exit_time and trade.exit_time < cutoff:
                                continue
                        except Exception:
                            pass
                        all_completed.append(trade.to_dict())
            
            # Keep only last 1000 for storage efficiency
            all_completed = all_completed[-1000:]
            self.redis_client.set('phantom:completed', json.dumps(all_completed, cls=NumpyJSONEncoder))
            
            logger.debug(f"Saved {len(self.active_phantoms)} active and {len(all_completed)} completed phantoms")
            
        except Exception as e:
            logger.error(f"Error saving phantom trades to Redis: {e}")

    def cancel_active(self, symbol: str):
        """Cancel and remove any active phantom for a symbol (e.g., when a real trade is executed)."""
        try:
            if symbol in self.active_phantoms:
                del self.active_phantoms[symbol]
                self._save_to_redis()
                logger.info(f"[{symbol}] Phantom canceled due to executed trade (trend)")
        except Exception:
            pass

    def set_notifier(self, notifier: Optional[Callable]):
        """Register a callback to receive phantom trade events."""
        self.notifier = notifier

    def _incr_blocked(self, strategy: str = 'trend'):
        """Increment daily blocked-counter in Redis for visibility."""
        # Always track locally
        from datetime import datetime as _dt
        day = _dt.utcnow().strftime('%Y%m%d')
        day_map = self._blocked_counts.setdefault(day, {'total': 0, 'trend': 0, 'mr': 0, 'scalp': 0})
        day_map['total'] += 1
        day_map[strategy] = day_map.get(strategy, 0) + 1
        # Best-effort Redis
        if self.redis_client:
            try:
                self.redis_client.incr(f'phantom:blocked:{day}')
                self.redis_client.incr(f'phantom:blocked:{day}:{strategy}')
            except Exception:
                pass

    def get_blocked_counts(self, day: Optional[str] = None) -> Dict[str, int]:
        """Return local blocked counters for a given day (YYYYMMDD)."""
        from datetime import datetime as _dt
        if day is None:
            day = _dt.utcnow().strftime('%Y%m%d')
        # Ensure trend key present for callers
        d = self._blocked_counts.get(day, {'total': 0, 'trend': 0, 'mr': 0, 'scalp': 0})
        if 'trend' not in d:
            d['trend'] = 0
        return d

    def record_signal(self, symbol: str, signal: dict, ml_score: float, 
                     was_executed: bool, features: dict, strategy_name: str = "unknown") -> Optional[PhantomTrade]:
        """
        Record a new signal (whether executed or not)
        
        Args:
            symbol: Trading symbol
            signal: Signal dictionary with entry, sl, tp
            ml_score: ML score assigned to this signal
            was_executed: Whether this signal was actually traded
            features: ML features for this signal
        """
        # Annotate features with feature_version/count for reproducibility
        try:
            if isinstance(features, dict):
                from ml_scorer_trend import get_trend_scorer
                scorer = get_trend_scorer()
                features = features.copy()
                features.setdefault('feature_version', 'trend_v1')
                try:
                    expected = len(scorer._prepare_features({}))  # type: ignore[attr-defined]
                except Exception:
                    expected = 0
                features.setdefault('feature_count', expected)
        except Exception:
            pass

        # Regime gating for Trend-like phantoms (non-executed)
        try:
            if not was_executed and strategy_name in ('trend_pullback','trend_breakout','trend','unknown'):
                vol = str((features or {}).get('volatility_regime', 'normal'))
                if vol == 'extreme':
                    logger.info(f"[{symbol}] 🛑 Trend phantom dropped by regime (volatility=extreme)")
                    return None
                slope = 0.0
                try:
                    slope = float((features or {}).get('trend_slope_pct', 0.0))
                except Exception:
                    slope = 0.0
                ema_stack = 0.0
                try:
                    ema_stack = float((features or {}).get('ema_stack_score', 0.0))
                except Exception:
                    ema_stack = 0.0
                side = str(signal.get('side', ''))
                # Require EMA alignment and slope in the trade direction
                if side == 'long' and (slope < 0.0 or ema_stack < 40.0):
                    logger.info(f"[{symbol}] 🛑 Trend phantom dropped by micro-trend (need up; slope={slope:.2f} ema={ema_stack:.0f})")
                    return None
                if side == 'short' and (slope > 0.0 or ema_stack < 40.0):
                    logger.info(f"[{symbol}] 🛑 Trend phantom dropped by micro-trend (need down; slope={slope:.2f} ema={ema_stack:.0f})")
                    return None
        except Exception:
            pass

        # Round TP/SL to tick size for non-executed phantoms to align with exchange
        try:
            if not was_executed:
                ts = self._tick_size_for(symbol)
                raw_tp = float(signal.get('tp'))
                raw_sl = float(signal.get('sl'))
                r_tp = round_step(raw_tp, ts)
                r_sl = round_step(raw_sl, ts)
                if r_tp != raw_tp or r_sl != raw_sl:
                    logger.debug(f"[{symbol}] Phantom TP/SL rounded to tick {ts}: TP {raw_tp}→{r_tp}, SL {raw_sl}→{r_sl}")
                signal = signal.copy()
                signal['tp'] = r_tp
                signal['sl'] = r_sl
        except Exception:
            pass

        # Allow multiple active phantoms per symbol

        phantom = PhantomTrade(
            symbol=symbol,
            side=signal['side'],
            entry_price=signal['entry'],
            stop_loss=signal['sl'],
            take_profit=signal['tp'],
            signal_time=datetime.now(),
            ml_score=ml_score,
            was_executed=was_executed,
            features=features,
            strategy_name=strategy_name,
            phantom_id=(uuid.uuid4().hex[:8])
        )
        
        # Only set as active if this is a phantom (non-executed) record
        if not was_executed:
            lst = self.active_phantoms.setdefault(symbol, [])
            lst.append(phantom)
        
        # Initialize list if needed
        if symbol not in self.phantom_trades:
            self.phantom_trades[symbol] = []
        
        # Log the recording
        status = "EXECUTED" if was_executed else f"REJECTED (score: {ml_score:.1f})"
        logger.info(f"[{symbol}] Phantom trade recorded: {status} - "
                   f"Entry: {signal['entry']:.4f}, TP: {signal['tp']:.4f}, SL: {signal['sl']:.4f}")
        
        # Save to Redis (only active set changed for phantom; still OK to write unified snapshot)
        self._save_to_redis()

        # Notify on open immediately (phantom-only)
        if self.notifier and not was_executed:
            try:
                res = self.notifier(phantom)
                if asyncio.iscoroutine(res):
                    asyncio.create_task(res)
            except Exception:
                pass
        
        return phantom

    def force_close_executed(self, symbol: str, exit_price: float, exit_reason: str = 'manual'):
        """Force-close any executed trend phantom mirrors to align with exchange closure."""
        try:
            if symbol not in self.active_phantoms:
                return
            keep: List[PhantomTrade] = []
            for ph in list(self.active_phantoms.get(symbol, [])):
                if getattr(ph, 'was_executed', False):
                    if ph.side == 'long':
                        outcome = 'win' if exit_price >= ph.take_profit else ('loss' if exit_price <= ph.stop_loss else ('win' if exit_price > ph.entry_price else 'loss'))
                    else:
                        outcome = 'win' if exit_price <= ph.take_profit else ('loss' if exit_price >= ph.stop_loss else ('win' if exit_price < ph.entry_price else 'loss'))
                    self._close_phantom(symbol, ph, exit_price, outcome, df=None, btc_price=None, symbol_collector=None, exit_reason=exit_reason)
                else:
                    keep.append(ph)
            if keep:
                self.active_phantoms[symbol] = keep
            else:
                try:
                    del self.active_phantoms[symbol]
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"[{symbol}] trend force_close_executed error: {e}")
    
    def update_phantom_prices(self, symbol: str, current_price: float, 
                             df=None, btc_price: float = None, symbol_collector=None):
        """
        Update phantom trade tracking with current price
        Check if phantom would have hit TP or SL
        """
        if symbol not in self.active_phantoms:
            return
        act_list = list(self.active_phantoms.get(symbol, []))
        remaining: List[PhantomTrade] = []
        # Use intrabar extremes if df provided
        try:
            cur_high = float(df['high'].iloc[-1]) if df is not None else current_price
            cur_low = float(df['low'].iloc[-1]) if df is not None else current_price
        except Exception:
            cur_high = current_price
            cur_low = current_price
        for ph in act_list:
            try:
                # Extremes
                if ph.side == 'long':
                    ph.max_favorable = (ph.max_favorable if ph.max_favorable is not None else ph.entry_price)
                    ph.max_adverse = (ph.max_adverse if ph.max_adverse is not None else ph.entry_price)
                    if current_price > ph.max_favorable:
                        ph.max_favorable = current_price
                    if current_price < ph.max_adverse:
                        ph.max_adverse = current_price
                    # Range override: TP1 at range midline if available
                    try:
                        if str(getattr(ph, 'strategy_name', '') or '').startswith('range'):
                            mid = None
                            try:
                                mid = float((ph.features or {}).get('range_mid', None))
                            except Exception:
                                mid = None
                            if isinstance(mid, (int,float)) and (not ph.tp1_hit) and cur_high >= float(mid):
                                ph.tp1_hit = True
                                ph.be_moved = True
                                ph.stop_loss = float(ph.entry_price)
                                try:
                                    if self.notifier:
                                        setattr(ph, 'phantom_event', 'tp1')
                                        res = self.notifier(ph)
                                        if asyncio.iscoroutine(res):
                                            asyncio.create_task(res)
                                except Exception:
                                    pass
                    except Exception:
                        pass
                    # Simulate TP1 → move SL to BE once price crosses tp1 level (R-multiple default)
                    try:
                        import yaml
                        tp1_r = 1.6
                        with open('config.yaml','r') as _f:
                            _cfg = yaml.safe_load(_f)
                            tp1_r = float(((((_cfg.get('trend', {}) or {}).get('exec', {}) or {}).get('scaleout', {}) or {}).get('tp1_r', 1.6)))
                    except Exception:
                        tp1_r = 1.6
                    try:
                        R = abs(float(ph.entry_price) - float(ph.stop_loss))
                        tp1_lvl = float(ph.entry_price) + tp1_r * R
                        if (not ph.tp1_hit) and cur_high >= tp1_lvl:
                            ph.tp1_hit = True
                            ph.be_moved = True
                            # Simulate SL moved to BE after TP1
                            ph.stop_loss = float(ph.entry_price)
                            # Notify TP1 event
                            try:
                                if self.notifier:
                                    setattr(ph, 'phantom_event', 'tp1')
                                    res = self.notifier(ph)
                                    if asyncio.iscoroutine(res):
                                        asyncio.create_task(res)
                            except Exception:
                                pass
                    except Exception:
                        pass
                    if cur_high >= ph.take_profit:
                        self._close_phantom(symbol, ph, current_price, 'win', df, btc_price, symbol_collector, exit_reason='tp')
                        continue
                    if cur_low <= ph.stop_loss:
                        self._close_phantom(symbol, ph, current_price, 'loss', df, btc_price, symbol_collector, exit_reason='sl')
                        continue
                else:
                    ph.max_favorable = (ph.max_favorable if ph.max_favorable is not None else ph.entry_price)
                    ph.max_adverse = (ph.max_adverse if ph.max_adverse is not None else ph.entry_price)
                    if current_price < ph.max_favorable:
                        ph.max_favorable = current_price
                    if current_price > ph.max_adverse:
                        ph.max_adverse = current_price
                    # Range override: TP1 at range midline if available (short)
                    try:
                        if str(getattr(ph, 'strategy_name', '') or '').startswith('range'):
                            mid = None
                            try:
                                mid = float((ph.features or {}).get('range_mid', None))
                            except Exception:
                                mid = None
                            if isinstance(mid, (int,float)) and (not ph.tp1_hit) and cur_low <= float(mid):
                                ph.tp1_hit = True
                                ph.be_moved = True
                                ph.stop_loss = float(ph.entry_price)
                                try:
                                    if self.notifier:
                                        setattr(ph, 'phantom_event', 'tp1')
                                        res = self.notifier(ph)
                                        if asyncio.iscoroutine(res):
                                            asyncio.create_task(res)
                                except Exception:
                                    pass
                    except Exception:
                        pass
                    try:
                        import yaml
                        tp1_r = 1.6
                        with open('config.yaml','r') as _f:
                            _cfg = yaml.safe_load(_f)
                            tp1_r = float(((((_cfg.get('trend', {}) or {}).get('exec', {}) or {}).get('scaleout', {}) or {}).get('tp1_r', 1.6)))
                    except Exception:
                        tp1_r = 1.6
                    try:
                        R = abs(float(ph.entry_price) - float(ph.stop_loss))
                        tp1_lvl = float(ph.entry_price) - tp1_r * R
                        if (not ph.tp1_hit) and cur_low <= tp1_lvl:
                            ph.tp1_hit = True
                            ph.be_moved = True
                            ph.stop_loss = float(ph.entry_price)
                            # Notify TP1 event
                            try:
                                if self.notifier:
                                    setattr(ph, 'phantom_event', 'tp1')
                                    res = self.notifier(ph)
                                    if asyncio.iscoroutine(res):
                                        asyncio.create_task(res)
                            except Exception:
                                pass
                    except Exception:
                        pass
                    if cur_low <= ph.take_profit:
                        self._close_phantom(symbol, ph, current_price, 'win', df, btc_price, symbol_collector, exit_reason='tp')
                        continue
                    if cur_high >= ph.stop_loss:
                        self._close_phantom(symbol, ph, current_price, 'loss', df, btc_price, symbol_collector, exit_reason='sl')
                        continue

                # Timeout
                try:
                    if self.timeout_hours and ph.signal_time:
                        from datetime import datetime as _dt, timedelta as _td
                        if _dt.now() - ph.signal_time > _td(hours=int(self.timeout_hours)):
                            try:
                                if ph.side == 'long':
                                    pnl_pct_now = ((current_price - ph.entry_price) / ph.entry_price) * 100
                                else:
                                    pnl_pct_now = ((ph.entry_price - current_price) / ph.entry_price) * 100
                                outcome_timeout = 'win' if pnl_pct_now >= 0 else 'loss'
                            except Exception:
                                outcome_timeout = 'loss'
                            self._close_phantom(symbol, ph, current_price, outcome_timeout, df, btc_price, symbol_collector, exit_reason='timeout')
                            continue
                except Exception:
                    pass
                remaining.append(ph)
            except Exception:
                remaining.append(ph)
        if remaining:
            self.active_phantoms[symbol] = remaining
        else:
            try:
                del self.active_phantoms[symbol]
            except Exception:
                pass
        self._save_to_redis()
    
    def _close_phantom(self, symbol: str, phantom: PhantomTrade, exit_price: float, outcome: str, 
                       df=None, btc_price: float = None, symbol_collector=None, exit_reason: Optional[str] = None):
        """Close a specific phantom trade and record its outcome"""
        phantom.outcome = outcome
        # Align exit to exact TP/SL for clearer R:R accounting
        try:
            if str(exit_reason).lower() == 'tp':
                exit_price = float(phantom.take_profit)
            elif str(exit_reason).lower() == 'sl':
                exit_price = float(phantom.stop_loss)
        except Exception:
            pass
        phantom.exit_price = exit_price
        phantom.exit_time = datetime.now()
        
        # Calculate P&L
        if phantom.side == "long":
            phantom.pnl_percent = ((exit_price - phantom.entry_price) / phantom.entry_price) * 100
        else:
            phantom.pnl_percent = ((phantom.entry_price - exit_price) / phantom.entry_price) * 100
        if exit_reason:
            phantom.exit_reason = exit_reason

        # Enrich labels: 1R/2R hits and realized RR
        try:
            if phantom.side == 'long':
                R = phantom.entry_price - phantom.stop_loss
                one_r_lvl = phantom.entry_price + R
                two_r_lvl = phantom.entry_price + 2 * R
                max_fav = phantom.max_favorable if phantom.max_favorable is not None else phantom.entry_price
                phantom.one_r_hit = bool(max_fav >= one_r_lvl)
                phantom.two_r_hit = bool(max_fav >= two_r_lvl)
                phantom.realized_rr = (exit_price - phantom.entry_price) / R if R > 0 else 0.0
            else:
                R = phantom.stop_loss - phantom.entry_price
                one_r_lvl = phantom.entry_price - R
                two_r_lvl = phantom.entry_price - 2 * R
                min_fav = phantom.max_favorable if phantom.max_favorable is not None else phantom.entry_price
                phantom.one_r_hit = bool(min_fav <= one_r_lvl)
                phantom.two_r_hit = bool(min_fav <= two_r_lvl)
                phantom.realized_rr = (phantom.entry_price - exit_price) / R if R > 0 else 0.0
        except Exception:
            phantom.one_r_hit = None
            phantom.two_r_hit = None
            phantom.realized_rr = None

        # If TP1 was hit earlier, classify as a win regardless of later BE/SL (non-timeout)
        try:
            if getattr(phantom, 'tp1_hit', False) and str(exit_reason).lower() != 'timeout':
                phantom.outcome = 'win'
        except Exception:
            pass
        
        # Move to completed list
        if symbol not in self.phantom_trades:
            self.phantom_trades[symbol] = []
        self.phantom_trades[symbol].append(phantom)
        # Remove only this phantom from active list
        try:
            self.active_phantoms[symbol] = [p for p in self.active_phantoms.get(symbol, []) if getattr(p, 'phantom_id', '') != getattr(phantom, 'phantom_id', '')]
            if not self.active_phantoms[symbol]:
                del self.active_phantoms[symbol]
        except Exception:
            pass
        
        # Log the outcome
        status = "EXECUTED" if phantom.was_executed else f"PHANTOM (score: {phantom.ml_score:.1f})"
        result = "✅ WIN" if outcome == "win" else "❌ LOSS"
        logger.info(f"[{symbol}] {status} trade closed: {result} - P&L: {phantom.pnl_percent:.2f}%"
                    + (f" (exit: {phantom.exit_reason})" if phantom.exit_reason else ""))
        
        # Important: Log if ML was wrong
        if not phantom.was_executed:
            if phantom.ml_score < 65 and outcome == "win":
                logger.warning(f"[{symbol}] ML rejected a winning trade! Score was {phantom.ml_score:.1f}")
            elif phantom.ml_score >= 65 and outcome == "loss":
                logger.warning(f"[{symbol}] ML approved a losing trade! Score was {phantom.ml_score:.1f}")
        
        # Record phantom trade data for symbol-specific learning
        if symbol_collector and not phantom.was_executed:
            try:
                # Record phantom trade context
                symbol_collector.record_phantom_trade(
                    symbol=symbol,
                    df=df,
                    btc_price=btc_price,
                    ml_score=phantom.ml_score,
                    features=phantom.features,
                    outcome=outcome,
                    pnl_percent=phantom.pnl_percent,
                    signal_time=phantom.signal_time,
                    exit_time=phantom.exit_time
                )
                logger.debug(f"[{symbol}] Recorded phantom trade data for future ML")
            except Exception as e:
                logger.warning(f"[{symbol}] Failed to record phantom data: {e}")
        
        # Feed phantom trade outcome to ML (Trend/Range) when applicable (non-timeout)
        if not phantom.was_executed and str(getattr(phantom, 'exit_reason', '')) != 'timeout':
            try:
                strat = str(getattr(phantom, 'strategy_name', '') or '')
                if strat in ('trend_breakout','trend_pullback'):
                    self._feed_phantom_to_trend_ml(phantom)
                elif strat.startswith('range'):
                    try:
                        from ml_scorer_range import get_range_scorer
                        rml = get_range_scorer()
                        signal_data = {
                            'side': phantom.side,
                            'entry': phantom.entry_price,
                            'sl': phantom.stop_loss,
                            'tp': phantom.take_profit,
                            'symbol': phantom.symbol,
                            'features': {**(phantom.features or {})},
                            'score': phantom.ml_score,
                            'timestamp': phantom.signal_time,
                            'was_executed': False,
                            'exit_reason': getattr(phantom, 'exit_reason', None)
                        }
                        rml.record_outcome(signal_data, phantom.outcome, phantom.pnl_percent)
                    except Exception as e:
                        logger.debug(f"Range phantom ML feed error: {e}")
            except Exception:
                pass

        # Save to Redis
        self._save_to_redis()

        # Persist to Postgres for audit
        try:
            from phantom_persistence import PhantomPersistence
            import json as _json
            rec = {
                'symbol': symbol,
                'side': phantom.side,
                'entry': phantom.entry_price,
                'sl': phantom.stop_loss,
                'tp': phantom.take_profit,
                'signal_time': phantom.signal_time,
                'exit_time': phantom.exit_time,
                'outcome': phantom.outcome,
                'realized_rr': phantom.realized_rr if hasattr(phantom, 'realized_rr') else None,
                'pnl_percent': phantom.pnl_percent,
                'exit_reason': phantom.exit_reason,
                'strategy_name': getattr(phantom, 'strategy_name', 'trend_pullback'),
                'was_executed': phantom.was_executed,
                'ml_score': phantom.ml_score,
                'features_json': _json.dumps(phantom.features, cls=NumpyJSONEncoder) if isinstance(phantom.features, dict) else '{}'
            }
            try:
                PhantomPersistence().add_trade(rec)
            except Exception:
                pass
        except Exception:
            pass

        # Update rolling WR list for WR guard (skip timeouts)
        try:
            if self.redis_client and getattr(phantom, 'exit_reason', None) != 'timeout':
                strat = getattr(phantom, 'strategy_name', 'trend') or 'trend'
                key = f"phantom:wr:{'trend' if strat == 'unknown' else strat}"
                val = '1' if outcome == 'win' else '0'
                self.redis_client.lpush(key, val)
                self.redis_client.ltrim(key, 0, 199)
        except Exception:
            pass

        # Check if ML needs retraining after this trade completes
        self._check_ml_retrain()

        if self.notifier:
            try:
                result = self.notifier(phantom)
                if asyncio.iscoroutine(result):
                    asyncio.create_task(result)
            except Exception as notify_err:
                logger.debug(f"Phantom notifier error: {notify_err}")
    
    def get_phantom_stats(self, symbol: Optional[str] = None) -> dict:
        """
        Get statistics on phantom trades
        
        Args:
            symbol: Optional symbol to filter by
        """
        all_phantoms = []
        # Include completed phantoms
        if symbol:
            all_phantoms = self.phantom_trades.get(symbol, [])
        else:
            for trades in self.phantom_trades.values():
                all_phantoms.extend(trades)

        # Include active (in-flight) phantom trades so the dashboard reflects
        # freshly tracked rejections immediately (not only after close)
        try:
            if symbol:
                if symbol in self.active_phantoms:
                    all_phantoms.append(self.active_phantoms[symbol])
            else:
                for active in self.active_phantoms.values():
                    all_phantoms.append(active)
        except Exception:
            # Best-effort; stats still work with completed-only
            pass
        
        if not all_phantoms:
            return {
                'total': 0,
                'executed': 0,
                'rejected': 0,
                'rejection_stats': {}
            }
        
        executed = [p for p in all_phantoms if getattr(p, 'was_executed', False)]
        rejected = [p for p in all_phantoms if not getattr(p, 'was_executed', False)]
        
        # Calculate what would have happened with rejected trades
        rejected_wins = [p for p in rejected if p.outcome == "win"]
        rejected_losses = [p for p in rejected if p.outcome == "loss"]
        
        # ML accuracy analysis
        ml_correct_rejections = [p for p in rejected if p.ml_score < 65 and p.outcome == "loss"]
        ml_wrong_rejections = [p for p in rejected if p.ml_score < 65 and p.outcome == "win"]
        ml_correct_approvals = [p for p in executed if p.ml_score >= 65 and p.outcome == "win"]
        ml_wrong_approvals = [p for p in executed if p.ml_score >= 65 and p.outcome == "loss"]
        
        stats = {
            'total': len(all_phantoms),
            'executed': len(executed),
            'rejected': len(rejected),
            
            'rejection_stats': {
                'total_rejected': len(rejected),
                'would_have_won': len(rejected_wins),
                'would_have_lost': len(rejected_losses),
                'missed_profit_pct': sum((p.pnl_percent or 0) for p in rejected_wins) if rejected_wins else 0,
                'avoided_loss_pct': sum(abs(p.pnl_percent or 0) for p in rejected_losses) if rejected_losses else 0
            },
            
            'ml_accuracy': {
                'correct_rejections': len(ml_correct_rejections),
                'wrong_rejections': len(ml_wrong_rejections),
                'correct_approvals': len(ml_correct_approvals),
                'wrong_approvals': len(ml_wrong_approvals),
                'accuracy_pct': ((len(ml_correct_rejections) + len(ml_correct_approvals)) / 
                               len(all_phantoms) * 100) if all_phantoms else 0
            }
        }
        
        return stats
    
    def get_learning_data(self) -> List[Dict]:
        """
        Get all phantom trade data formatted for ML learning
        Returns both executed and phantom trades for comprehensive learning
        """
        learning_data = []
        
        for trades in self.phantom_trades.values():
            for trade in trades:
                if trade.outcome in ['win', 'loss']:
                    # Create learning record
                    # Time to outcome (seconds)
                    t_sec = None
                    try:
                        t_sec = int((trade.exit_time - trade.signal_time).total_seconds()) if trade.exit_time else None
                    except Exception:
                        t_sec = None

                    record = {
                        'features': trade.features,
                        'score': trade.ml_score,  # ML training expects 'score' field
                        'was_executed': trade.was_executed,
                        'outcome': 1 if trade.outcome == 'win' else 0,
                        'pnl_percent': trade.pnl_percent,
                        'symbol': trade.symbol,
                        'side': trade.side,
                        'max_favorable_move': abs(trade.max_favorable - trade.entry_price) / trade.entry_price * 100 if trade.max_favorable else 0,
                        'max_adverse_move': abs(trade.max_adverse - trade.entry_price) / trade.entry_price * 100 if trade.max_adverse else 0,
                        'time_to_outcome_sec': t_sec,
                        'one_r_hit': trade.one_r_hit,
                        'two_r_hit': trade.two_r_hit,
                        'realized_rr': trade.realized_rr
                    }
                    learning_data.append(record)
        
        return learning_data
    
    def cleanup_old_phantoms(self, hours: int = 24):
        """Remove phantom trades older than specified hours"""
        # This method is now deprecated - phantom trades will only close on TP/SL
        # Keeping the method signature for backwards compatibility
        pass
    
    def _feed_phantom_to_trend_ml(self, phantom):
        """Feed completed phantom trade outcome to Trend ML for training"""
        try:
            from ml_scorer_trend import get_trend_scorer
            ml_scorer = get_trend_scorer()

            signal_data = {
                'side': phantom.side,
                'entry': phantom.entry_price,
                'sl': phantom.stop_loss,
                'tp': phantom.take_profit,
                'symbol': phantom.symbol,
                'features': {**(phantom.features or {}),
                             'tp1_hit': 1.0 if getattr(phantom, 'tp1_hit', False) else 0.0,
                             'be_moved': 1.0 if getattr(phantom, 'be_moved', False) else 0.0},
                'score': phantom.ml_score,
                'timestamp': phantom.signal_time,
                'was_executed': False,
                'exit_reason': getattr(phantom, 'exit_reason', None)
            }

            outcome = phantom.outcome
            pnl_pct = phantom.pnl_percent if hasattr(phantom, 'pnl_percent') else 0
            ml_scorer.record_outcome(signal_data, outcome, pnl_pct)
            logger.debug(f"[{phantom.symbol}] Fed trend phantom trade outcome to ML: {outcome} "
                         f"(P&L: {pnl_pct:.2f}%, Score: {phantom.ml_score:.1f})")

        except Exception as e:
            logger.error(f"Error feeding trend phantom trade to ML: {e}")

    def _check_ml_retrain(self):
        """Check if ML needs retraining after a trade completes"""
        try:
            # Prefer Trend scorer
            try:
                from ml_scorer_trend import get_trend_scorer
                tml = get_trend_scorer()
                info = tml.get_retrain_info()
                if info.get('can_train') and int(info.get('trades_until_next_retrain', 1)) == 0:
                    # Best-effort internal retrain
                    retrain_ok = False
                    try:
                        retrain_ok = bool(tml._retrain())  # type: ignore[attr-defined]
                    except Exception:
                        retrain_ok = False
                    if retrain_ok:
                        logger.info("✅ Trend ML models retrained after trade completion")
                    else:
                        logger.debug("Trend ML retrain skipped or failed (insufficient data or no-op)")
                else:
                    logger.debug(f"Trend ML retrain check: {info.get('trades_until_next_retrain', '?')} trades until next retrain")
                return
            except Exception:
                pass
            # Fallback to legacy immediate scorer (kept for backward compatibility)
            try:
                from ml_signal_scorer_immediate import get_immediate_scorer
                iml = get_immediate_scorer()
                retrain_info = iml.get_retrain_info()
                if retrain_info.get('can_train') and int(retrain_info.get('trades_until_next_retrain', 1)) == 0:
                    try:
                        ok = iml.startup_retrain()
                        if ok:
                            logger.info("✅ Legacy ML retrained after trade completion")
                    except Exception:
                        pass
            except Exception:
                pass
        except Exception as e:
            logger.debug(f"ML retrain check failed: {e}")

# Global instance
_phantom_tracker = None

def get_phantom_tracker() -> PhantomTradeTracker:
    """Get or create the global phantom tracker instance"""
    global _phantom_tracker
    if _phantom_tracker is None:
        _phantom_tracker = PhantomTradeTracker()
    return _phantom_tracker
