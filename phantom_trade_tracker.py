"""
Phantom Trade Tracker for ML Learning
Tracks ALL signals (including rejected ones) to see their hypothetical outcomes
This allows ML to learn from the full spectrum of trading opportunities
"""
import json
import logging
import redis
import os
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import time
import numpy as np

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
    
    # Outcome tracking (filled in later)
    outcome: Optional[str] = None  # "win", "loss", or "active"
    exit_price: Optional[float] = None
    pnl_percent: Optional[float] = None
    exit_time: Optional[datetime] = None
    max_favorable: Optional[float] = None  # Best price reached
    max_adverse: Optional[float] = None  # Worst price reached
    
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
        
        # Initialize Redis
        self._init_redis()
        self._load_from_redis()
    
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
            
            # Save completed phantoms (keep last 1000)
            all_completed = []
            for trades in self.phantom_trades.values():
                for trade in trades:
                    if trade.outcome in ['win', 'loss']:
                        all_completed.append(trade.to_dict())
            
            # Keep only last 1000 for storage efficiency
            all_completed = all_completed[-1000:]
            self.redis_client.set('phantom:completed', json.dumps(all_completed, cls=NumpyJSONEncoder))
            
            logger.debug(f"Saved {len(self.active_phantoms)} active and {len(all_completed)} completed phantoms")
            
        except Exception as e:
            logger.error(f"Error saving phantom trades to Redis: {e}")
    
    def record_signal(self, symbol: str, signal: dict, ml_score: float, 
                     was_executed: bool, features: dict) -> PhantomTrade:
        """
        Record a new signal (whether executed or not)
        
        Args:
            symbol: Trading symbol
            signal: Signal dictionary with entry, sl, tp
            ml_score: ML score assigned to this signal
            was_executed: Whether this signal was actually traded
            features: ML features for this signal
        """
        phantom = PhantomTrade(
            symbol=symbol,
            side=signal['side'],
            entry_price=signal['entry'],
            stop_loss=signal['sl'],
            take_profit=signal['tp'],
            signal_time=datetime.now(),
            ml_score=ml_score,
            was_executed=was_executed,
            features=features
        )
        
        # Store as active phantom
        self.active_phantoms[symbol] = phantom
        
        # Initialize list if needed
        if symbol not in self.phantom_trades:
            self.phantom_trades[symbol] = []
        
        # Log the recording
        status = "EXECUTED" if was_executed else f"REJECTED (score: {ml_score:.1f})"
        logger.info(f"[{symbol}] Phantom trade recorded: {status} - "
                   f"Entry: {signal['entry']:.4f}, TP: {signal['tp']:.4f}, SL: {signal['sl']:.4f}")
        
        # Save to Redis
        self._save_to_redis()
        
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
                self._close_phantom(symbol, current_price, "win", df, btc_price, symbol_collector)
            elif current_price <= phantom.stop_loss:
                self._close_phantom(symbol, current_price, "loss", df, btc_price, symbol_collector)
                
        else:  # short
            if phantom.max_favorable is None or current_price < phantom.max_favorable:
                phantom.max_favorable = current_price
            if phantom.max_adverse is None or current_price > phantom.max_adverse:
                phantom.max_adverse = current_price
                
            # Check if hit TP or SL
            if current_price <= phantom.take_profit:
                self._close_phantom(symbol, current_price, "win", df, btc_price, symbol_collector)
            elif current_price >= phantom.stop_loss:
                self._close_phantom(symbol, current_price, "loss", df, btc_price, symbol_collector)
    
    def _close_phantom(self, symbol: str, exit_price: float, outcome: str, 
                       df=None, btc_price: float = None, symbol_collector=None):
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
        
        # Move to completed list
        if symbol not in self.phantom_trades:
            self.phantom_trades[symbol] = []
        self.phantom_trades[symbol].append(phantom)
        del self.active_phantoms[symbol]
        
        # Log the outcome
        status = "EXECUTED" if phantom.was_executed else f"PHANTOM (score: {phantom.ml_score:.1f})"
        result = "✅ WIN" if outcome == "win" else "❌ LOSS"
        logger.info(f"[{symbol}] {status} trade closed: {result} - P&L: {phantom.pnl_percent:.2f}%")
        
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
        
        # Save to Redis
        self._save_to_redis()
    
    def get_phantom_stats(self, symbol: Optional[str] = None) -> dict:
        """
        Get statistics on phantom trades
        
        Args:
            symbol: Optional symbol to filter by
        """
        all_phantoms = []
        if symbol:
            all_phantoms = self.phantom_trades.get(symbol, [])
        else:
            for trades in self.phantom_trades.values():
                all_phantoms.extend(trades)
        
        if not all_phantoms:
            return {
                'total': 0,
                'executed': 0,
                'rejected': 0,
                'rejection_stats': {}
            }
        
        executed = [p for p in all_phantoms if p.was_executed]
        rejected = [p for p in all_phantoms if not p.was_executed]
        
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
                'missed_profit_pct': sum(p.pnl_percent for p in rejected_wins) if rejected_wins else 0,
                'avoided_loss_pct': sum(abs(p.pnl_percent) for p in rejected_losses) if rejected_losses else 0
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
                    record = {
                        'features': trade.features,
                        'ml_score': trade.ml_score,
                        'was_executed': trade.was_executed,
                        'outcome': 1 if trade.outcome == 'win' else 0,
                        'pnl_percent': trade.pnl_percent,
                        'symbol': trade.symbol,
                        'side': trade.side,
                        'max_favorable_move': abs(trade.max_favorable - trade.entry_price) / trade.entry_price * 100 if trade.max_favorable else 0,
                        'max_adverse_move': abs(trade.max_adverse - trade.entry_price) / trade.entry_price * 100 if trade.max_adverse else 0
                    }
                    learning_data.append(record)
        
        return learning_data
    
    def cleanup_old_phantoms(self, hours: int = 24):
        """Remove phantom trades older than specified hours"""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        for symbol in list(self.active_phantoms.keys()):
            phantom = self.active_phantoms[symbol]
            if phantom.signal_time < cutoff:
                # Force close as timeout
                logger.info(f"[{symbol}] Phantom trade timed out after {hours} hours")
                del self.active_phantoms[symbol]
        
        self._save_to_redis()

# Global instance
_phantom_tracker = None

def get_phantom_tracker() -> PhantomTradeTracker:
    """Get or create the global phantom tracker instance"""
    global _phantom_tracker
    if _phantom_tracker is None:
        _phantom_tracker = PhantomTradeTracker()
    return _phantom_tracker