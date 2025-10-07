"""
Mean Reversion Specific Phantom Trade Tracker
Specialized tracking for ranging market signals with enhanced learning data
"""
import json
import logging
import os
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional, Tuple
import uuid

import asyncio
import numpy as np
import redis

logger = logging.getLogger(__name__)

class MRNumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for mean reversion data"""
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
class MRPhantomTrade:
    """A mean reversion phantom trade with range-specific tracking"""
    symbol: str
    side: str  # "long" or "short"
    entry_price: float
    stop_loss: float
    take_profit: float
    signal_time: datetime
    ml_score: float
    was_executed: bool
    features: Dict  # Enhanced MR features
    enhanced_features: Dict  # Additional enhanced features
    strategy_name: str = "enhanced_mr"
    phantom_id: str = ""

    # Range-specific data
    range_upper: Optional[float] = None
    range_lower: Optional[float] = None
    range_confidence: Optional[float] = None
    range_position: Optional[float] = None  # 0=bottom, 1=top

    # Outcome tracking
    outcome: Optional[str] = None  # "win", "loss", or "active"
    exit_price: Optional[float] = None
    pnl_percent: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_reason: Optional[str] = None  # "tp", "sl", "timeout"

    # Range-specific tracking
    max_favorable: Optional[float] = None
    max_adverse: Optional[float] = None
    time_at_extremes: Optional[int] = None  # Candles spent at range boundaries
    range_breakout_occurred: Optional[bool] = None  # Did range break during trade
    # Enriched labels
    one_r_hit: Optional[bool] = None
    two_r_hit: Optional[bool] = None
    realized_rr: Optional[float] = None

    def to_dict(self):
        """Convert to dictionary with proper type handling"""
        d = asdict(self)

        # Handle numpy types in nested dictionaries
        for key in ['features', 'enhanced_features']:
            if key in d and d[key]:
                cleaned_dict = {}
                for k, v in d[key].items():
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

class MRPhantomTracker:
    """
    Specialized phantom tracker for mean reversion strategy
    Tracks range-specific behaviors and outcomes
    """

    def __init__(self):
        self.redis_client = None
        self.mr_phantom_trades: Dict[str, List[MRPhantomTrade]] = {}
        self.active_mr_phantoms: Dict[str, List[MRPhantomTrade]] = {}
        self.notifier: Optional[Callable] = None
        # Local blocked counters fallback: day (YYYYMMDD) -> counts
        self._blocked_counts: Dict[str, Dict[str, int]] = {}

        # Range tracking data
        self.range_performance = {}  # Track performance by range characteristics
        self.timeout_hours = 48  # Close phantom trades after 48 hours in ranging markets

        # Initialize Redis with MR-specific keys
        self._init_redis()
        self._load_from_redis()

    def _init_redis(self):
        """Initialize Redis connection with MR namespace"""
        try:
            redis_url = os.getenv('REDIS_URL')
            if redis_url:
                self.redis_client = redis.from_url(redis_url, decode_responses=True)
                self.redis_client.ping()
                logger.info("MR Phantom Tracker connected to Redis")
            else:
                logger.warning("No REDIS_URL for MR Phantom Tracker, using memory only")
        except Exception as e:
            logger.warning(f"Redis connection failed for MR Phantom Tracker: {e}")
            self.redis_client = None

    def _load_from_redis(self):
        """Load MR phantom trades from Redis"""
        if not self.redis_client:
            return

        try:
            # Load active MR phantoms
            active_data = self.redis_client.get('mr_phantom:active')
            if active_data:
                active_dict = json.loads(active_data)
                cnt = 0
                for symbol, rec in active_dict.items():
                    items: List[MRPhantomTrade] = []
                    if isinstance(rec, list):
                        for t in rec:
                            try:
                                items.append(MRPhantomTrade.from_dict(t))
                            except Exception:
                                pass
                    elif isinstance(rec, dict):
                        try:
                            items.append(MRPhantomTrade.from_dict(rec))
                        except Exception:
                            pass
                    if items:
                        self.active_mr_phantoms[symbol] = items
                        cnt += len(items)
                logger.info(f"Loaded {cnt} active MR phantoms across {len(self.active_mr_phantoms)} symbols")

            # Load completed MR phantoms
            completed_data = self.redis_client.get('mr_phantom:completed')
            if completed_data:
                completed_list = json.loads(completed_data)
                for trade_dict in completed_list:
                    phantom = MRPhantomTrade.from_dict(trade_dict)
                    if phantom.symbol not in self.mr_phantom_trades:
                        self.mr_phantom_trades[phantom.symbol] = []
                    self.mr_phantom_trades[phantom.symbol].append(phantom)

                total = sum(len(trades) for trades in self.mr_phantom_trades.values())
                logger.info(f"Loaded {total} completed MR phantom trades")

        except Exception as e:
            logger.error(f"Error loading MR phantom trades from Redis: {e}")

    def _save_to_redis(self):
        """Save MR phantom trades to Redis"""
        if not self.redis_client:
            return

        try:
            # Save active MR phantoms
            active_dict = {}
            for symbol, trades in self.active_mr_phantoms.items():
                active_dict[symbol] = [t.to_dict() for t in trades]
            self.redis_client.set('mr_phantom:active',
                                json.dumps(active_dict, cls=MRNumpyJSONEncoder))

            # Save completed MR phantoms (keep last 1500, and drop >30d old)
            from datetime import datetime, timedelta
            cutoff = datetime.now() - timedelta(days=30)
            all_completed = []
            for trades in self.mr_phantom_trades.values():
                for trade in trades:
                    if trade.outcome in ['win', 'loss']:
                        try:
                            if trade.exit_time and trade.exit_time < cutoff:
                                continue
                        except Exception:
                            pass
                        all_completed.append(trade.to_dict())

            # Keep only last 1500 for storage efficiency
            all_completed = all_completed[-1500:]
            self.redis_client.set('mr_phantom:completed',
                                json.dumps(all_completed, cls=MRNumpyJSONEncoder))

            logger.debug(f"Saved {len(self.active_mr_phantoms)} active and {len(all_completed)} completed MR phantoms")

        except Exception as e:
            logger.error(f"Error saving MR phantom trades to Redis: {e}")

    def cancel_active(self, symbol: str):
        """Cancel and remove any active MR phantom for a symbol (e.g., when an executed trade opens)."""
        try:
            if symbol in self.active_mr_phantoms:
                del self.active_mr_phantoms[symbol]
                self._save_to_redis()
                logger.info(f"[{symbol}] MR phantom canceled due to executed trade")
        except Exception:
            pass

    def set_notifier(self, notifier: Optional[Callable]):
        self.notifier = notifier

    def _incr_blocked(self):
        from datetime import datetime as _dt
        day = _dt.utcnow().strftime('%Y%m%d')
        # Always track locally
        day_map = self._blocked_counts.setdefault(day, {'total': 0, 'trend': 0, 'mr': 0, 'scalp': 0})
        day_map['total'] += 1
        day_map['mr'] = day_map.get('mr', 0) + 1
        # Best-effort Redis
        if self.redis_client:
            try:
                self.redis_client.incr(f'phantom:blocked:{day}')
                self.redis_client.incr(f'phantom:blocked:{day}:mr')
            except Exception:
                pass

    def get_blocked_counts(self, day: Optional[str] = None) -> Dict[str, int]:
        """Return local blocked counters for a given day (YYYYMMDD)."""
        from datetime import datetime as _dt
        if day is None:
            day = _dt.utcnow().strftime('%Y%m%d')
        return self._blocked_counts.get(day, {'total': 0, 'trend': 0, 'mr': 0, 'scalp': 0})

    def record_mr_signal(self, symbol: str, signal: dict, ml_score: float,
                        was_executed: bool, features: dict, enhanced_features: dict = None) -> MRPhantomTrade:
        """
        Record a mean reversion signal with range-specific data

        Args:
            symbol: Trading symbol
            signal: Signal dict with entry, sl, tp, meta
            ml_score: Enhanced MR ML score
            was_executed: Whether signal was actually traded
            features: Basic MR features
            enhanced_features: Enhanced feature set from enhanced_mr_features.py
        """
        # Allow multiple active MR phantoms per symbol

        # Extract range information from signal meta
        meta = signal.get('meta', {})
        range_upper = meta.get('range_upper')
        range_lower = meta.get('range_lower')

        # Calculate range-specific metrics
        range_confidence = None
        range_position = None
        if range_upper and range_lower and range_upper > range_lower:
            entry_price = signal['entry']
            range_height = range_upper - range_lower
            range_position = (entry_price - range_lower) / range_height

            # Get confidence from enhanced features if available
            if enhanced_features:
                range_confidence = enhanced_features.get('range_confidence', 0.5)

        # Calculate range width percentage for analysis (matches strategy filter)
        range_width_pct = 0.0
        if range_upper > 0 and range_lower > 0:
            range_width_pct = (range_upper - range_lower) / range_lower

        # Annotate features with MR feature version/count for reproducibility
        try:
            if isinstance(features, dict):
                # Import MR scorer to detect current feature layout
                from ml_scorer_mean_reversion import get_mean_reversion_scorer
                mr = get_mean_reversion_scorer()
                features = features.copy()
                # Convention: label as mr_simplified_v1 for current minimal set
                features.setdefault('feature_version', 'mr_simplified_v1')
                try:
                    expected = len(mr._prepare_features({}))
                except Exception:
                    expected = 0
                features.setdefault('feature_count', expected)
        except Exception:
            pass

        phantom = MRPhantomTrade(
            symbol=symbol,
            side=signal['side'],
            entry_price=signal['entry'],
            stop_loss=signal['sl'],
            take_profit=signal['tp'],
            signal_time=datetime.now(),
            ml_score=ml_score,
            was_executed=was_executed,
            features=features or {},
            enhanced_features=enhanced_features or {},
            range_upper=range_upper,
            range_lower=range_lower,
            range_confidence=range_confidence,
            range_position=range_position,
            strategy_name="enhanced_mr",
            phantom_id=(uuid.uuid4().hex[:8])
        )

        # Only set active for phantom (non-executed) records
        if not was_executed:
            lst = self.active_mr_phantoms.setdefault(symbol, [])
            lst.append(phantom)

        # Initialize list if needed
        if symbol not in self.mr_phantom_trades:
            self.mr_phantom_trades[symbol] = []

        # Log the recording with MR-specific info
        status = "EXECUTED" if was_executed else f"REJECTED (score: {ml_score:.1f})"
        range_info = ""
        if range_upper and range_lower:
            range_info = f", Range: {range_lower:.4f}-{range_upper:.4f} ({range_width_pct:.1%} width)"
            if range_position:
                range_info += f" (pos: {range_position:.2f})"

        logger.info(f"[{symbol}] MR phantom recorded: {status} - "
                   f"Entry: {signal['entry']:.4f}, Side: {signal['side']}{range_info}")

        # Save to Redis
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

    def update_mr_phantom_prices(self, symbol: str, current_price: float, df=None):
        """Update MR phantom trade with range-specific tracking"""
        if symbol not in self.active_mr_phantoms:
            return
        act_list = list(self.active_mr_phantoms.get(symbol, []))
        remaining: List[MRPhantomTrade] = []
        for ph in act_list:
            try:
                if ph.side == "long":
                    ph.max_favorable = max(ph.max_favorable or current_price, current_price)
                    ph.max_adverse = min(ph.max_adverse or current_price, current_price)
                    if current_price >= ph.take_profit:
                        self._close_mr_phantom(symbol, ph, current_price, "win", "tp")
                        continue
                    elif current_price <= ph.stop_loss:
                        self._close_mr_phantom(symbol, ph, current_price, "loss", "sl")
                        continue
                else:
                    ph.max_favorable = min(ph.max_favorable or current_price, current_price)
                    ph.max_adverse = max(ph.max_adverse or current_price, current_price)
                    if current_price <= ph.take_profit:
                        self._close_mr_phantom(symbol, ph, current_price, "win", "tp")
                        continue
                    elif current_price >= ph.stop_loss:
                        self._close_mr_phantom(symbol, ph, current_price, "loss", "sl")
                        continue
                # Timeout
                time_elapsed = datetime.now() - ph.signal_time
                if time_elapsed > timedelta(hours=self.timeout_hours):
                    try:
                        pnl_pct_now = ((current_price - ph.entry_price) / ph.entry_price) * 100 if ph.side=='long' else ((ph.entry_price - current_price) / ph.entry_price) * 100
                        outc = 'win' if pnl_pct_now >= 0 else 'loss'
                    except Exception:
                        outc = 'loss'
                    self._close_mr_phantom(symbol, ph, current_price, outc, "timeout")
                    continue
                remaining.append(ph)
            except Exception:
                remaining.append(ph)
        if remaining:
            self.active_mr_phantoms[symbol] = remaining
        else:
            try:
                del self.active_mr_phantoms[symbol]
            except Exception:
                pass

        # Legacy single-phantom block removed; handled above per phantom instance

    def _close_mr_phantom(self, symbol: str, phantom: MRPhantomTrade, exit_price: float, outcome: str, exit_reason: str):
        """Close specific MR phantom trade with range-specific analysis"""
        phantom.outcome = outcome
        phantom.exit_price = exit_price
        phantom.exit_time = datetime.now()
        phantom.exit_reason = exit_reason

        # Calculate P&L
        if phantom.side == "long":
            phantom.pnl_percent = ((exit_price - phantom.entry_price) / phantom.entry_price) * 100
        else:
            phantom.pnl_percent = ((phantom.entry_price - exit_price) / phantom.entry_price) * 100

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
        if symbol not in self.mr_phantom_trades:
            self.mr_phantom_trades[symbol] = []
        self.mr_phantom_trades[symbol].append(phantom)
        try:
            self.active_mr_phantoms[symbol] = [p for p in self.active_mr_phantoms.get(symbol, []) if getattr(p, 'phantom_id','') != getattr(phantom,'phantom_id','')]
            if not self.active_mr_phantoms[symbol]:
                del self.active_mr_phantoms[symbol]
        except Exception:
            pass

        # Log outcome with MR-specific details
        status = "EXECUTED" if phantom.was_executed else f"PHANTOM (score: {phantom.ml_score:.1f})"
        result = "✅ WIN" if outcome == "win" else "❌ LOSS"

        additional_info = []
        if phantom.range_breakout_occurred:
            additional_info.append("BREAKOUT")
        if phantom.range_position:
            additional_info.append(f"pos:{phantom.range_position:.2f}")
        if exit_reason == "timeout":
            additional_info.append("TIMEOUT")

        info_str = f" [{', '.join(additional_info)}]" if additional_info else ""

        logger.info(f"[{symbol}] MR {status} trade closed: {result} - "
                   f"P&L: {phantom.pnl_percent:.2f}%, Exit: {exit_reason}{info_str}")

        # Analyze ML accuracy for MR trades
        if not phantom.was_executed:
            self._analyze_mr_ml_accuracy(phantom, outcome)

        # Update range performance tracking
        self._update_range_performance(phantom)

        # Update rolling WR list for WR guard (skip timeouts)
        try:
            if self.redis_client and exit_reason != 'timeout':
                key = 'phantom:wr:mr'
                val = '1' if outcome == 'win' else '0'
                self.redis_client.lpush(key, val)
                self.redis_client.ltrim(key, 0, 199)
        except Exception:
            pass

        # Save to Redis
        self._save_to_redis()

        # Feed phantom trade outcome to MR ML for training
        if not phantom.was_executed:
            self._feed_phantom_to_mr_ml(phantom)

        # Check if ML needs retraining
        self._check_mr_ml_retrain()

        if self.notifier:
            try:
                result = self.notifier(phantom)
                if asyncio.iscoroutine(result):
                    asyncio.create_task(result)
            except Exception as notify_err:
                logger.debug(f"MR phantom notifier error: {notify_err}")

    def _analyze_mr_ml_accuracy(self, phantom: MRPhantomTrade, outcome: str):
        """Analyze ML accuracy for mean reversion trades"""
        ml_threshold = 72  # Default MR threshold

        if phantom.ml_score < ml_threshold and outcome == "win":
            range_info = ""
            if phantom.range_confidence:
                range_info = f" (range conf: {phantom.range_confidence:.2f})"
            logger.warning(f"[{phantom.symbol}] MR ML rejected a winning trade! "
                         f"Score: {phantom.ml_score:.1f}, Outcome: WIN{range_info}")

        elif phantom.ml_score >= ml_threshold and outcome == "loss":
            range_info = ""
            if phantom.range_confidence:
                range_info = f" (range conf: {phantom.range_confidence:.2f})"
            logger.warning(f"[{phantom.symbol}] MR ML approved a losing trade! "
                         f"Score: {phantom.ml_score:.1f}, Outcome: LOSS{range_info}")

    def _update_range_performance(self, phantom: MRPhantomTrade):
        """Update performance tracking by range characteristics"""
        if not phantom.range_confidence:
            return

        # Categorize ranges by confidence
        if phantom.range_confidence >= 0.8:
            range_category = "high_confidence"
        elif phantom.range_confidence >= 0.6:
            range_category = "medium_confidence"
        else:
            range_category = "low_confidence"

        if range_category not in self.range_performance:
            self.range_performance[range_category] = {'wins': 0, 'losses': 0, 'total_pnl': 0.0}

        if phantom.outcome == 'win':
            self.range_performance[range_category]['wins'] += 1
        else:
            self.range_performance[range_category]['losses'] += 1

        self.range_performance[range_category]['total_pnl'] += phantom.pnl_percent or 0.0

    def _feed_phantom_to_mr_ml(self, phantom: MRPhantomTrade):
        """Feed completed phantom trade outcome to MR ML for training"""
        try:
            from enhanced_mr_scorer import get_enhanced_mr_scorer
            mr_scorer = get_enhanced_mr_scorer()

            # Create signal data format expected by ML scorer
            signal_data = {
                'side': phantom.side,
                'entry': phantom.entry_price,
                'sl': phantom.stop_loss,
                'tp': phantom.take_profit,
                'symbol': phantom.symbol,
                'features': phantom.features,
                'enhanced_features': phantom.enhanced_features,
                'score': phantom.ml_score,  # ML scorer expects 'score' field
                'timestamp': phantom.signal_time,
                'exit_reason': phantom.exit_reason
            }

            # Calculate P&L percentage for outcome
            pnl_pct = phantom.pnl_percent if hasattr(phantom, 'pnl_percent') else 0

            # Determine outcome (win/loss)
            outcome = phantom.outcome

            # Record phantom trade outcome for ML learning
            mr_scorer.record_outcome(signal_data, outcome, pnl_pct)

            logger.debug(f"[{phantom.symbol}] Fed MR phantom trade outcome to ML: {outcome} "
                        f"(P&L: {pnl_pct:.2f}%, Score: {phantom.ml_score:.1f})")

        except Exception as e:
            logger.error(f"Error feeding phantom trade to MR ML: {e}")

    def _check_mr_ml_retrain(self):
        """Check if Enhanced MR ML needs retraining"""
        try:
            from enhanced_mr_scorer import get_enhanced_mr_scorer
            mr_scorer = get_enhanced_mr_scorer()

            # Get combined trade count
            executed_count = mr_scorer.completed_trades
            phantom_count = sum(len(trades) for trades in self.mr_phantom_trades.values())
            total_combined = executed_count + phantom_count

            # Check if ready to retrain
            trades_since_last = total_combined - mr_scorer.last_train_count
            if trades_since_last >= mr_scorer.RETRAIN_INTERVAL and executed_count >= mr_scorer.MIN_TRADES_FOR_ML:
                logger.info(f"🔄 Enhanced MR ML retrain triggered after phantom trade completion - "
                           f"{total_combined} total trades available")

                # Could trigger retrain here or let the main system handle it
                # For now, just log the opportunity

        except Exception as e:
            logger.error(f"Error checking Enhanced MR ML retrain: {e}")

    def get_learning_data(self) -> List[Dict]:
        """
        Get all MR phantom trade data formatted for ML learning
        Returns both executed and phantom trades for comprehensive learning
        """
        learning_data = []

        for trades in self.mr_phantom_trades.values():
            for trade in trades:
                if hasattr(trade, 'outcome') and trade.outcome in ['win', 'loss']:
                    # Create learning record
                    record = {
                        'features': trade.features,
                        'enhanced_features': trade.enhanced_features,
                        'score': trade.ml_score,  # ML training expects 'score' field
                        'was_executed': trade.was_executed,
                        'outcome': 1 if trade.outcome == 'win' else 0,
                        'pnl_percent': getattr(trade, 'pnl_percent', 0),
                        'symbol': trade.symbol,
                        'side': trade.side,
                        'strategy': 'enhanced_mr',
                        'range_upper': trade.range_upper,
                        'range_lower': trade.range_lower,
                        'range_position': getattr(trade, 'range_position', None),
                        'range_breakout': getattr(trade, 'range_breakout_occurred', False)
                    }
                    learning_data.append(record)

        return learning_data

    def get_mr_phantom_stats(self, symbol: Optional[str] = None) -> dict:
        """Get MR-specific phantom trade statistics"""
        all_mr_phantoms = []
        if symbol:
            all_mr_phantoms = self.mr_phantom_trades.get(symbol, [])
        else:
            for trades in self.mr_phantom_trades.values():
                all_mr_phantoms.extend(trades)
            
            # Also include active phantoms in the count
            for active_phantom in self.active_mr_phantoms.values():
                all_mr_phantoms.append(active_phantom)

        if not all_mr_phantoms:
            return {
                'total_mr_trades': 0,
                'executed': 0,
                'rejected': 0,
                'range_performance': {},
                'mr_specific_metrics': {},
                'outcome_analysis': {
                    'executed_win_rate': 0,
                    'rejected_would_win_rate': 0
                }
            }

        # Properly separate executed vs rejected trades
        executed = [p for p in all_mr_phantoms if hasattr(p, 'was_executed') and p.was_executed]
        rejected = [p for p in all_mr_phantoms if hasattr(p, 'was_executed') and not p.was_executed]

        # MR-specific analysis (only count completed trades for these metrics)
        completed_phantoms = [p for p in all_mr_phantoms if hasattr(p, 'outcome') and p.outcome in ['win', 'loss']]
        range_breakout_trades = [p for p in completed_phantoms if hasattr(p, 'range_breakout_occurred') and p.range_breakout_occurred]
        timeout_trades = [p for p in completed_phantoms if hasattr(p, 'exit_reason') and p.exit_reason == "timeout"]

        # Range confidence analysis
        high_conf_trades = [p for p in completed_phantoms if hasattr(p, 'range_confidence') and p.range_confidence and p.range_confidence >= 0.8]
        low_conf_trades = [p for p in completed_phantoms if hasattr(p, 'range_confidence') and p.range_confidence and p.range_confidence < 0.5]

        # Range position analysis
        boundary_trades = [p for p in completed_phantoms
                          if hasattr(p, 'range_position') and p.range_position and (p.range_position <= 0.2 or p.range_position >= 0.8)]

        stats = {
            'total_mr_trades': len(all_mr_phantoms),
            'executed': len(executed),
            'rejected': len(rejected),

            'mr_specific_metrics': {
                'range_breakout_during_trade': len(range_breakout_trades),
                'timeout_closures': len(timeout_trades),
                'high_confidence_ranges': len(high_conf_trades),
                'low_confidence_ranges': len(low_conf_trades),
                'boundary_entries': len(boundary_trades),
            },

            'range_performance': self.range_performance.copy(),

            'outcome_analysis': {
                'executed_win_rate': (sum(1 for p in executed if hasattr(p, 'outcome') and p.outcome == 'win') / len(executed) * 100) if executed else 0,
                'rejected_would_win_rate': (sum(1 for p in rejected if hasattr(p, 'outcome') and p.outcome == 'win') / len(rejected) * 100) if rejected else 0,
            }
        }

        # Add range-specific insights
        if high_conf_trades:
            high_conf_wins = sum(1 for p in high_conf_trades if hasattr(p, 'outcome') and p.outcome == 'win')
            stats['range_performance']['high_confidence_win_rate'] = high_conf_wins / len(high_conf_trades) * 100

        if boundary_trades:
            boundary_wins = sum(1 for p in boundary_trades if hasattr(p, 'outcome') and p.outcome == 'win')
            stats['mr_specific_metrics']['boundary_entry_win_rate'] = boundary_wins / len(boundary_trades) * 100

        return stats

    def get_mr_learning_data(self) -> List[Dict]:
        """Get MR phantom data formatted for Enhanced ML learning"""
        learning_data = []

        for trades in self.mr_phantom_trades.values():
            for trade in trades:
                if trade.outcome in ['win', 'loss']:
                    # Create enhanced learning record with MR-specific features
                    record = {
                        'features': trade.features,
                        'enhanced_features': trade.enhanced_features,
                        'ml_score': trade.ml_score,
                        'was_executed': trade.was_executed,
                        'outcome': 1 if trade.outcome == 'win' else 0,
                        'pnl_percent': trade.pnl_percent,
                        'symbol': trade.symbol,
                        'side': trade.side,
                        'strategy': trade.strategy_name,

                        # MR-specific data
                        'range_confidence': trade.range_confidence,
                        'range_position': trade.range_position,
                        'range_breakout_occurred': trade.range_breakout_occurred or False,
                        'exit_reason': trade.exit_reason,

                        # Performance metrics
                        'max_favorable_move': (abs(trade.max_favorable - trade.entry_price) /
                                            trade.entry_price * 100) if trade.max_favorable else 0,
                        'max_adverse_move': (abs(trade.max_adverse - trade.entry_price) /
                                           trade.entry_price * 100) if trade.max_adverse else 0,
                    }
                    learning_data.append(record)

        return learning_data

    def cleanup_old_mr_phantoms(self, hours: int = None):
        """Clean up old MR phantom trades (now mainly timeout-based)"""
        if hours is None:
            hours = self.timeout_hours

        cutoff_time = datetime.now() - timedelta(hours=hours)
        removed_count = 0

        for symbol in list(self.active_mr_phantoms.keys()):
            phantom = self.active_mr_phantoms[symbol]
            if phantom.signal_time < cutoff_time:
                logger.info(f"[{symbol}] Cleaning up old MR phantom trade (age: {hours}h)")
                self._close_mr_phantom(symbol, phantom.entry_price, "loss", "cleanup")
                removed_count += 1

        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} old MR phantom trades")

# Global instance
_mr_phantom_tracker = None

def get_mr_phantom_tracker() -> MRPhantomTracker:
    """Get or create the global MR phantom tracker instance"""
    global _mr_phantom_tracker
    if _mr_phantom_tracker is None:
        _mr_phantom_tracker = MRPhantomTracker()
        logger.info("Initialized MR Phantom Tracker")
    return _mr_phantom_tracker
