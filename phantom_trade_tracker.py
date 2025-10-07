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
from typing import Callable, Dict, List, Optional, Tuple

import asyncio
import numpy as np
import redis

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
        self.phantom_trades = {}  # symbol -> list of phantom trades
        self.active_phantoms = {}  # symbol -> PhantomTrade
        self.notifier: Optional[Callable] = None
        # Local blocked counters fallback: day (YYYYMMDD) -> counts
        self._blocked_counts: Dict[str, Dict[str, int]] = {}
        
        # Initialize Redis
        self._init_redis()
        self._load_from_redis()
        # Default timeout for trend phantom (can be overridden via config)
        self.timeout_hours: int = 36
    
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
            # Load active phantoms
            active_data = self.redis_client.get('phantom:active')
            if active_data:
                active_dict = json.loads(active_data)
                for symbol, trade_dict in active_dict.items():
                    self.active_phantoms[symbol] = PhantomTrade.from_dict(trade_dict)
                logger.info(f"Loaded {len(self.active_phantoms)} active phantom trades")
            
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
            # Save active phantoms
            active_dict = {
                symbol: trade.to_dict() 
                for symbol, trade in self.active_phantoms.items()
            }
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
                     was_executed: bool, features: dict, strategy_name: str = "unknown") -> PhantomTrade:
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

        # Enforce single-active-per-symbol for phantom (non-executed) trades
        if symbol in self.active_phantoms and not was_executed:
            logger.info(f"[{symbol}] Phantom blocked: active trade in progress (strategy={strategy_name})")
            self._incr_blocked(strategy_name if strategy_name else 'trend')
            # Do not overwrite the active phantom; return the existing one
            return self.active_phantoms[symbol]

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
            strategy_name=strategy_name
        )
        
        # Only set as active if this is a phantom (non-executed) record
        if not was_executed:
            self.active_phantoms[symbol] = phantom
        
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
    
    def update_phantom_prices(self, symbol: str, current_price: float, 
                             df=None, btc_price: float = None, symbol_collector=None):
        """
        Update phantom trade tracking with current price
        Check if phantom would have hit TP or SL
        """
        if symbol not in self.active_phantoms:
            return
            
        phantom = self.active_phantoms[symbol]
        
        # Track maximum favorable and adverse excursions
        if phantom.side == "long":
            if phantom.max_favorable is None or current_price > phantom.max_favorable:
                phantom.max_favorable = current_price
            if phantom.max_adverse is None or current_price < phantom.max_adverse:
                phantom.max_adverse = current_price
                
            # Check if hit TP or SL
            if current_price >= phantom.take_profit:
                self._close_phantom(symbol, current_price, "win", df, btc_price, symbol_collector, exit_reason='tp')
            elif current_price <= phantom.stop_loss:
                self._close_phantom(symbol, current_price, "loss", df, btc_price, symbol_collector, exit_reason='sl')
                
        else:  # short
            if phantom.max_favorable is None or current_price < phantom.max_favorable:
                phantom.max_favorable = current_price
            if phantom.max_adverse is None or current_price > phantom.max_adverse:
                phantom.max_adverse = current_price
                
            # Check if hit TP or SL
            if current_price <= phantom.take_profit:
                self._close_phantom(symbol, current_price, "win", df, btc_price, symbol_collector, exit_reason='tp')
            elif current_price >= phantom.stop_loss:
                self._close_phantom(symbol, current_price, "loss", df, btc_price, symbol_collector, exit_reason='sl')

        # Timeout handling: close after configured hours
        try:
            if self.timeout_hours and phantom.signal_time:
                from datetime import datetime as _dt, timedelta as _td
                if _dt.now() - phantom.signal_time > _td(hours=int(self.timeout_hours)):
                    try:
                        if phantom.side == 'long':
                            pnl_pct_now = ((current_price - phantom.entry_price) / phantom.entry_price) * 100
                        else:
                            pnl_pct_now = ((phantom.entry_price - current_price) / phantom.entry_price) * 100
                        outcome_timeout = 'win' if pnl_pct_now >= 0 else 'loss'
                    except Exception:
                        outcome_timeout = 'loss'
                    self._close_phantom(symbol, current_price, outcome_timeout, df, btc_price, symbol_collector, exit_reason='timeout')
                    return
        except Exception:
            pass
    
    def _close_phantom(self, symbol: str, exit_price: float, outcome: str, 
                       df=None, btc_price: float = None, symbol_collector=None, exit_reason: Optional[str] = None):
        """Close a phantom trade and record its outcome"""
        if symbol not in self.active_phantoms:
            return
            
        phantom = self.active_phantoms[symbol]
        phantom.outcome = outcome
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
        
        # Move to completed list
        if symbol not in self.phantom_trades:
            self.phantom_trades[symbol] = []
        self.phantom_trades[symbol].append(phantom)
        del self.active_phantoms[symbol]
        
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
        
        # Feed phantom trade outcome to Trend ML for training when applicable
        if not phantom.was_executed:
            try:
                if getattr(phantom, 'strategy_name', '') == 'trend_breakout':
                    self._feed_phantom_to_trend_ml(phantom)
            except Exception:
                pass

        # Save to Redis
        self._save_to_redis()

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
                'features': phantom.features,
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
