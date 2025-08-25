"""
Multi-Timeframe Strategy Learner
Learns optimal HTF/LTF combinations and patterns for each symbol
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import redis.asyncio as redis
import structlog
from collections import defaultdict, deque

logger = structlog.get_logger(__name__)

@dataclass
class PatternSuccess:
    """Track success rate of specific patterns"""
    pattern_type: str  # 'rejection', 'break', 'continuation'
    htf_timeframe: str
    ltf_timeframe: str
    zone_type: str  # 'supply' or 'demand'
    structure_pattern: List[str]  # ['HH', 'HL'] etc
    success_count: int = 0
    failure_count: int = 0
    total_profit: float = 0
    avg_risk_reward: float = 0
    
    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0
    
    @property
    def score(self) -> float:
        """Combined score for pattern quality"""
        if self.success_count + self.failure_count < 5:
            return 0  # Not enough data
        
        base_score = self.success_rate * 100
        
        # Boost for good risk/reward
        if self.avg_risk_reward > 2:
            base_score *= 1.2
        elif self.avg_risk_reward > 1.5:
            base_score *= 1.1
        
        # Boost for consistency (more trades)
        trade_count = self.success_count + self.failure_count
        if trade_count > 20:
            base_score *= 1.15
        elif trade_count > 10:
            base_score *= 1.05
        
        return min(100, base_score)

@dataclass
class SymbolLearningProfile:
    """Learning profile for a specific symbol"""
    symbol: str
    best_htf: str = "240"  # Default 4H
    best_ltf: str = "15"   # Default 15m
    patterns: Dict[str, PatternSuccess] = field(default_factory=dict)
    trade_history: deque = field(default_factory=lambda: deque(maxlen=100))
    
    # Performance metrics
    total_trades: int = 0
    winning_trades: int = 0
    total_pnl: float = 0
    best_pattern_key: Optional[str] = None
    
    # Optimal parameters learned
    optimal_zone_distance: float = 0.02  # 2% from zone
    optimal_structure_confidence: float = 70
    optimal_volume_threshold: float = 1.5
    
    # Time-based patterns
    best_trading_hours: List[int] = field(default_factory=list)
    worst_trading_hours: List[int] = field(default_factory=list)
    
    @property
    def win_rate(self) -> float:
        return self.winning_trades / self.total_trades if self.total_trades > 0 else 0
    
    def update_best_pattern(self):
        """Update the best performing pattern"""
        if not self.patterns:
            return
        
        best_score = 0
        best_key = None
        
        for key, pattern in self.patterns.items():
            if pattern.score > best_score:
                best_score = pattern.score
                best_key = key
        
        self.best_pattern_key = best_key

class MTFStrategyLearner:
    """
    Learns and adapts the multi-timeframe strategy for each symbol
    Stores learned parameters in Redis for fast access
    """
    
    def __init__(self):
        self.redis_client = None
        self.symbol_profiles: Dict[str, SymbolLearningProfile] = {}
        self.symbol_parameters: Dict[str, Dict[str, Any]] = {}  # Store learned parameters
        
        # Learning configuration
        self.min_trades_for_learning = 10
        self.pattern_expiry_days = 30
        self.confidence_threshold = 0.6
        
        # Available timeframe combinations to test
        self.htf_options = ["240", "60"]  # 4H, 1H
        self.ltf_options = ["15", "5"]    # 15m, 5m
        
    async def initialize(self, redis_url: str):
        """Initialize Redis connection and load existing profiles"""
        try:
            self.redis_client = redis.from_url(
                redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            await self.redis_client.ping()
            
            # Load existing profiles
            await self._load_profiles()
            
            logger.info("MTF Strategy Learner initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize learner: {e}")
            self.redis_client = None
    
    def should_retrain(self, symbol: str) -> bool:
        """
        Check if the model for this symbol should be retrained
        
        Returns:
            bool: True if retraining is needed
        """
        if symbol not in self.symbol_profiles:
            return True  # New symbol needs training
        
        profile = self.symbol_profiles[symbol]
        
        # Retrain if no recent trades
        if profile.total_trades < self.min_trades_for_learning:
            return False  # Not enough data yet
        
        # Retrain every 50 trades
        if profile.total_trades % 50 == 0:
            return True
        
        # Retrain if last update was too long ago
        if profile.last_update:
            time_since_update = datetime.now() - profile.last_update
            if time_since_update.days > 7:  # Retrain weekly
                return True
        
        # Retrain if performance is poor
        recent_performance = profile.pattern_performance.get(profile.best_pattern_key, {})
        if recent_performance.get('win_rate', 0) < 0.4:  # Below 40% win rate
            return True
        
        return False
    
    async def train_for_symbol(self, symbol: str):
        """
        Train the MTF learner for a specific symbol
        Analyzes historical patterns and updates optimal parameters
        
        Args:
            symbol: Trading symbol to train
        """
        try:
            # Initialize profile if new symbol
            if symbol not in self.symbol_profiles:
                self.symbol_profiles[symbol] = SymbolLearningProfile(symbol=symbol)
                logger.info(f"Created new learning profile for {symbol}")
            
            profile = self.symbol_profiles[symbol]
            
            # Skip if not enough data
            if profile.total_trades < self.min_trades_for_learning:
                logger.debug(f"Not enough trades for {symbol}: {profile.total_trades}/{self.min_trades_for_learning}")
                return
            
            # Update best pattern based on performance
            profile.update_best_pattern()
            
            # Analyze patterns and update parameters
            best_patterns = sorted(
                profile.pattern_performance.items(),
                key=lambda x: x[1].get('win_rate', 0) * x[1].get('total_trades', 0),
                reverse=True
            )[:3]  # Top 3 patterns
            
            if best_patterns:
                # Extract optimal parameters from best patterns
                optimal_htf = None
                optimal_ltf = None
                optimal_strategy = None
                
                for pattern_key, performance in best_patterns:
                    parts = pattern_key.split('_')
                    if len(parts) >= 3:
                        if performance.get('win_rate', 0) > 0.5:
                            optimal_htf = parts[0]
                            optimal_ltf = parts[1]
                            optimal_strategy = parts[2] if len(parts) > 2 else 'rejection'
                            break
                
                # Update profile with optimal parameters
                if optimal_htf and optimal_ltf:
                    profile.optimal_htf = optimal_htf
                    profile.optimal_ltf = optimal_ltf
                    profile.preferred_patterns = [optimal_strategy]
                    profile.last_update = datetime.now()
                    
                    logger.info(
                        f"Updated {symbol} parameters: HTF={optimal_htf}, LTF={optimal_ltf}, "
                        f"Strategy={optimal_strategy}, Win Rate={best_patterns[0][1].get('win_rate', 0):.2%}"
                    )
                    
                    # Update symbol_parameters for easy access
                    self.symbol_parameters[symbol] = {
                        'htf': optimal_htf,
                        'ltf': optimal_ltf,
                        'strategy': optimal_strategy,
                        'confidence': profile.confidence_score,
                        'win_rate': profile.win_rate(),
                        'total_trades': profile.total_trades,
                        'last_update': datetime.now().isoformat()
                    }
                    
                    # Save to Redis if available
                    await self._save_profile(symbol)
            
            # Clean up old pattern data (keep last 100 patterns)
            if len(profile.recent_patterns) > 100:
                profile.recent_patterns = profile.recent_patterns[-100:]
            
            # Remove expired patterns
            current_time = datetime.now()
            expired_keys = [
                key for key in profile.pattern_performance.keys()
                if (current_time - profile.pattern_performance[key].get('last_seen', current_time)).days > self.pattern_expiry_days
            ]
            
            for key in expired_keys:
                del profile.pattern_performance[key]
            
            logger.info(f"Training completed for {symbol}: {profile.total_trades} trades analyzed")
            
        except Exception as e:
            logger.error(f"Error training MTF learner for {symbol}: {e}")
    
    def get_symbol_parameters(self, symbol: str) -> Dict[str, Any]:
        """
        Get the learned parameters for a symbol
        
        Returns:
            Dict containing optimal HTF, LTF, and strategy parameters
        """
        if symbol not in self.symbol_profiles:
            # Return defaults for unknown symbols
            return {
                'htf': '60',  # 1H
                'ltf': '15',  # 15m
                'strategy': 'rejection',
                'confidence': 0.5
            }
        
        profile = self.symbol_profiles[symbol]
        return {
            'htf': profile.optimal_htf,
            'ltf': profile.optimal_ltf,
            'strategy': profile.preferred_patterns[0] if profile.preferred_patterns else 'rejection',
            'confidence': profile.confidence_score,
            'win_rate': profile.win_rate(),
            'total_trades': profile.total_trades
        }
    
    async def record_trade_result(
        self,
        symbol: str,
        signal: Dict[str, Any],
        result: Dict[str, Any]
    ):
        """
        Record the result of a trade to learn from it
        
        Args:
            symbol: Trading symbol
            signal: The signal that triggered the trade
            result: Trade result including PnL, duration, etc.
        """
        
        # Get or create profile
        if symbol not in self.symbol_profiles:
            self.symbol_profiles[symbol] = SymbolLearningProfile(symbol)
        
        profile = self.symbol_profiles[symbol]
        
        # Extract pattern information
        pattern_key = self._create_pattern_key(signal)
        
        if pattern_key not in profile.patterns:
            profile.patterns[pattern_key] = PatternSuccess(
                pattern_type=signal.get('type', 'unknown'),
                htf_timeframe=signal['timeframes']['htf'],
                ltf_timeframe=signal['timeframes']['ltf'],
                zone_type=signal['htf_zone']['type'],
                structure_pattern=signal.get('structure_pattern', [])
            )
        
        pattern = profile.patterns[pattern_key]
        
        # Update pattern statistics
        if result['profit'] > 0:
            pattern.success_count += 1
            profile.winning_trades += 1
        else:
            pattern.failure_count += 1
        
        pattern.total_profit += result['profit']
        
        # Update risk/reward
        if result.get('risk_reward'):
            pattern.avg_risk_reward = (
                (pattern.avg_risk_reward * (pattern.success_count + pattern.failure_count - 1) + 
                 result['risk_reward']) / 
                (pattern.success_count + pattern.failure_count)
            )
        
        # Update profile statistics
        profile.total_trades += 1
        profile.total_pnl += result['profit']
        
        # Add to trade history
        profile.trade_history.append({
            'timestamp': datetime.now(),
            'signal': signal,
            'result': result,
            'pattern_key': pattern_key
        })
        
        # Learn from the trade
        await self._learn_from_trade(symbol, signal, result)
        
        # Update best pattern
        profile.update_best_pattern()
        
        # Save to Redis
        await self._save_profile(symbol)
        
        logger.info(f"Recorded trade for {symbol}: Profit={result['profit']:.2f}, "
                   f"Pattern={pattern_key}, Success Rate={pattern.success_rate:.2%}")
    
    async def _learn_from_trade(
        self,
        symbol: str,
        signal: Dict[str, Any],
        result: Dict[str, Any]
    ):
        """Learn and adapt strategy parameters from trade results"""
        
        profile = self.symbol_profiles[symbol]
        
        # Learn optimal timeframe combinations
        if result['profit'] > 0:
            # This combination worked, increase its weight
            htf = signal['timeframes']['htf']
            ltf = signal['timeframes']['ltf']
            
            # Simple learning: if this combo has >60% win rate after 10 trades, make it default
            pattern_key = f"{htf}_{ltf}"
            wins = sum(1 for t in profile.trade_history 
                      if t['signal']['timeframes']['htf'] == htf 
                      and t['signal']['timeframes']['ltf'] == ltf
                      and t['result']['profit'] > 0)
            total = sum(1 for t in profile.trade_history 
                       if t['signal']['timeframes']['htf'] == htf 
                       and t['signal']['timeframes']['ltf'] == ltf)
            
            if total >= 10 and wins / total > 0.6:
                profile.best_htf = htf
                profile.best_ltf = ltf
                logger.info(f"Updated best timeframes for {symbol}: HTF={htf}, LTF={ltf}")
        
        # Learn optimal zone distance
        zone_distance = abs(signal['entry_price'] - signal['htf_zone']['lower']) / signal['entry_price']
        if result['profit'] > 0:
            # Weighted average towards successful distances
            profile.optimal_zone_distance = (
                profile.optimal_zone_distance * 0.9 + zone_distance * 0.1
            )
        
        # Learn optimal confidence threshold
        if 'confidence' in signal:
            if result['profit'] > 0 and signal['confidence'] < profile.optimal_structure_confidence:
                # Lower threshold if profitable trades happen at lower confidence
                profile.optimal_structure_confidence *= 0.98
            elif result['profit'] < 0 and signal['confidence'] > profile.optimal_structure_confidence:
                # Raise threshold if losses happen at current confidence
                profile.optimal_structure_confidence *= 1.02
            
            # Keep within reasonable bounds
            profile.optimal_structure_confidence = max(60, min(90, profile.optimal_structure_confidence))
        
        # Learn time-based patterns
        trade_hour = datetime.now().hour
        if result['profit'] > 0:
            if trade_hour not in profile.best_trading_hours:
                profile.best_trading_hours.append(trade_hour)
        else:
            if trade_hour not in profile.worst_trading_hours:
                profile.worst_trading_hours.append(trade_hour)
    
    def get_optimal_parameters(self, symbol: str) -> Dict[str, Any]:
        """
        Get the learned optimal parameters for a symbol
        
        Returns:
            Dictionary with optimal trading parameters
        """
        
        if symbol not in self.symbol_profiles:
            # Return defaults
            return {
                'htf': '240',
                'ltf': '15',
                'zone_distance': 0.02,
                'confidence_threshold': 70,
                'volume_threshold': 1.5,
                'best_pattern': None,
                'win_rate': 0,
                'avoid_hours': []
            }
        
        profile = self.symbol_profiles[symbol]
        
        # Get best pattern details
        best_pattern = None
        if profile.best_pattern_key and profile.best_pattern_key in profile.patterns:
            pattern = profile.patterns[profile.best_pattern_key]
            best_pattern = {
                'type': pattern.pattern_type,
                'success_rate': pattern.success_rate,
                'avg_rr': pattern.avg_risk_reward,
                'structure': pattern.structure_pattern
            }
        
        return {
            'htf': profile.best_htf,
            'ltf': profile.best_ltf,
            'zone_distance': profile.optimal_zone_distance,
            'confidence_threshold': profile.optimal_structure_confidence,
            'volume_threshold': profile.optimal_volume_threshold,
            'best_pattern': best_pattern,
            'win_rate': profile.win_rate,
            'avoid_hours': profile.worst_trading_hours[-3:] if len(profile.worst_trading_hours) > 5 else []
        }
    
    def should_take_trade(self, symbol: str, signal: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Decide if a trade should be taken based on learned patterns
        
        Returns:
            (should_trade, reason)
        """
        
        if symbol not in self.symbol_profiles:
            # No history, take the trade to learn
            return True, "No history - learning mode"
        
        profile = self.symbol_profiles[symbol]
        
        # Check if we have enough data
        if profile.total_trades < self.min_trades_for_learning:
            return True, f"Learning phase ({profile.total_trades}/{self.min_trades_for_learning} trades)"
        
        # Check time-based filters
        current_hour = datetime.now().hour
        if current_hour in profile.worst_trading_hours[-3:] and len(profile.worst_trading_hours) > 5:
            return False, f"Poor performance at hour {current_hour}"
        
        # Check pattern success rate
        pattern_key = self._create_pattern_key(signal)
        if pattern_key in profile.patterns:
            pattern = profile.patterns[pattern_key]
            
            if pattern.success_count + pattern.failure_count >= 5:
                if pattern.success_rate < 0.4:
                    return False, f"Low success rate for pattern: {pattern.success_rate:.1%}"
                
                if pattern.avg_risk_reward < 1.0 and pattern.success_rate < 0.6:
                    return False, f"Poor risk/reward: {pattern.avg_risk_reward:.2f}"
        
        # Check confidence against learned threshold
        if signal.get('confidence', 0) < profile.optimal_structure_confidence:
            return False, f"Low confidence: {signal['confidence']:.1f} < {profile.optimal_structure_confidence:.1f}"
        
        # All checks passed
        return True, "Signal meets learned criteria"
    
    def _create_pattern_key(self, signal: Dict[str, Any]) -> str:
        """Create a unique key for a pattern"""
        
        htf = signal['timeframes']['htf']
        ltf = signal['timeframes']['ltf']
        zone_type = signal['htf_zone']['type']
        structure = '_'.join(signal.get('structure_pattern', [])[-2:])  # Last 2 patterns
        
        return f"{htf}_{ltf}_{zone_type}_{structure}"
    
    async def _save_profile(self, symbol: str):
        """Save symbol profile to Redis"""
        
        if not self.redis_client or symbol not in self.symbol_profiles:
            return
        
        try:
            profile = self.symbol_profiles[symbol]
            
            # Convert to JSON-serializable format
            profile_data = {
                'symbol': profile.symbol,
                'best_htf': profile.best_htf,
                'best_ltf': profile.best_ltf,
                'total_trades': profile.total_trades,
                'winning_trades': profile.winning_trades,
                'total_pnl': profile.total_pnl,
                'optimal_zone_distance': profile.optimal_zone_distance,
                'optimal_structure_confidence': profile.optimal_structure_confidence,
                'optimal_volume_threshold': profile.optimal_volume_threshold,
                'best_trading_hours': profile.best_trading_hours,
                'worst_trading_hours': profile.worst_trading_hours,
                'patterns': {
                    k: {
                        'success_count': p.success_count,
                        'failure_count': p.failure_count,
                        'total_profit': p.total_profit,
                        'avg_risk_reward': p.avg_risk_reward
                    }
                    for k, p in profile.patterns.items()
                }
            }
            
            key = f"mtf_profile:{symbol}"
            await self.redis_client.setex(
                key,
                86400 * 7,  # 7 days expiry
                json.dumps(profile_data)
            )
            
        except Exception as e:
            logger.error(f"Failed to save profile for {symbol}: {e}")
    
    async def _load_profiles(self):
        """Load existing profiles from Redis"""
        
        if not self.redis_client:
            return
        
        try:
            # Get all profile keys
            keys = await self.redis_client.keys("mtf_profile:*")
            
            for key in keys:
                data = await self.redis_client.get(key)
                if data:
                    profile_data = json.loads(data)
                    symbol = profile_data['symbol']
                    
                    # Reconstruct profile
                    profile = SymbolLearningProfile(
                        symbol=symbol,
                        best_htf=profile_data['best_htf'],
                        best_ltf=profile_data['best_ltf'],
                        total_trades=profile_data['total_trades'],
                        winning_trades=profile_data['winning_trades'],
                        total_pnl=profile_data['total_pnl'],
                        optimal_zone_distance=profile_data['optimal_zone_distance'],
                        optimal_structure_confidence=profile_data['optimal_structure_confidence'],
                        optimal_volume_threshold=profile_data['optimal_volume_threshold'],
                        best_trading_hours=profile_data['best_trading_hours'],
                        worst_trading_hours=profile_data['worst_trading_hours']
                    )
                    
                    self.symbol_profiles[symbol] = profile
            
            logger.info(f"Loaded {len(self.symbol_profiles)} symbol profiles")
            
        except Exception as e:
            logger.error(f"Failed to load profiles: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get overall learning statistics"""
        
        total_symbols = len(self.symbol_profiles)
        total_trades = sum(p.total_trades for p in self.symbol_profiles.values())
        total_pnl = sum(p.total_pnl for p in self.symbol_profiles.values())
        
        best_symbols = sorted(
            self.symbol_profiles.items(),
            key=lambda x: x[1].win_rate,
            reverse=True
        )[:5]
        
        return {
            'total_symbols_learned': total_symbols,
            'total_trades_analyzed': total_trades,
            'total_pnl': total_pnl,
            'avg_win_rate': sum(p.win_rate for p in self.symbol_profiles.values()) / total_symbols if total_symbols > 0 else 0,
            'best_performing_symbols': [
                {
                    'symbol': s,
                    'win_rate': p.win_rate,
                    'trades': p.total_trades,
                    'pnl': p.total_pnl
                }
                for s, p in best_symbols
            ]
        }