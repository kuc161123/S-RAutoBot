"""
Performance Tracking System for ML Enhancement
Tracks all trades and learns from outcomes without affecting live trading
"""
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
import structlog

logger = structlog.get_logger(__name__)

@dataclass
class TradeRecord:
    """Record of a single trade for learning"""
    # Required fields first
    timestamp: datetime
    symbol: str
    action: str  # BUY or SELL
    entry_price: float
    
    # Market conditions at entry (required)
    rsi: float
    macd: float
    macd_signal: float
    stoch_rsi_k: float
    volume_ratio: float
    distance_to_support: float
    distance_to_resistance: float
    trend: str  # BULLISH, BEARISH, RANGING
    volatility: float
    
    # Which confirmations triggered (required)
    confirmations: List[str]
    confirmation_count: int
    signal_score: float
    confidence: float
    
    # Strategy parameters used (required)
    rsi_threshold: float
    min_confirmations: int
    min_score: float
    
    # Optional fields (with defaults) at the end
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    pnl_percentage: Optional[float] = None
    win: Optional[bool] = None
    
    def to_dict(self):
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data):
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

class PerformanceTracker:
    """Tracks and analyzes trading performance for ML optimization"""
    
    def __init__(self, data_dir: str = "ml_data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        self.trades_file = os.path.join(data_dir, "trades.json")
        self.analysis_file = os.path.join(data_dir, "analysis.json")
        self.patterns_file = os.path.join(data_dir, "patterns.json")
        
        self.trades: List[TradeRecord] = self._load_trades()
        self.analysis_cache = {}
        
        logger.info(f"Performance tracker initialized with {len(self.trades)} historical trades")
    
    def _load_trades(self) -> List[TradeRecord]:
        """Load historical trades from file"""
        if os.path.exists(self.trades_file):
            try:
                with open(self.trades_file, 'r') as f:
                    data = json.load(f)
                    return [TradeRecord.from_dict(t) for t in data]
            except Exception as e:
                logger.error(f"Error loading trades: {e}")
        return []
    
    def _save_trades(self):
        """Save trades to file"""
        try:
            with open(self.trades_file, 'w') as f:
                json.dump([t.to_dict() for t in self.trades], f, indent=2)
        except Exception as e:
            logger.error(f"Error saving trades: {e}")
    
    def record_entry(self, symbol: str, action: str, price: float, 
                    market_data: Dict, confirmations: List[str],
                    score: float, confidence: float, params: Dict) -> str:
        """Record a trade entry"""
        trade = TradeRecord(
            timestamp=datetime.now(),
            symbol=symbol,
            action=action,
            entry_price=price,
            
            # Market conditions
            rsi=market_data.get('rsi', 50),
            macd=market_data.get('macd', 0),
            macd_signal=market_data.get('macd_signal', 0),
            stoch_rsi_k=market_data.get('stoch_rsi_k', 50),
            volume_ratio=market_data.get('volume_ratio', 1),
            distance_to_support=market_data.get('dist_to_support', 100),
            distance_to_resistance=market_data.get('dist_to_resistance', 100),
            trend=market_data.get('trend', 'RANGING'),
            volatility=market_data.get('volatility', 1),
            
            # Signal details
            confirmations=confirmations,
            confirmation_count=len(confirmations),
            signal_score=score,
            confidence=confidence,
            
            # Parameters
            rsi_threshold=params.get('rsi_oversold' if action == 'BUY' else 'rsi_overbought', 30),
            min_confirmations=params.get('min_confirmations', 3),
            min_score=params.get('min_score', 5)
        )
        
        self.trades.append(trade)
        self._save_trades()
        
        trade_id = f"{symbol}_{datetime.now().timestamp()}"
        logger.info(f"Recorded trade entry: {trade_id}")
        return trade_id
    
    def record_exit(self, trade_id: str, exit_price: float, pnl: float):
        """Record a trade exit"""
        # Find the trade by ID pattern
        for trade in reversed(self.trades):
            if f"{trade.symbol}_{trade.timestamp.timestamp()}" == trade_id:
                trade.exit_price = exit_price
                trade.pnl = pnl
                trade.pnl_percentage = (pnl / trade.entry_price) * 100
                trade.win = pnl > 0
                self._save_trades()
                logger.info(f"Recorded trade exit: {trade_id} PnL: ${pnl:.2f}")
                break
    
    def get_performance_stats(self, symbol: Optional[str] = None, 
                             days: int = 30) -> Dict:
        """Get performance statistics"""
        cutoff = datetime.now() - timedelta(days=days)
        
        # Filter trades
        trades = [t for t in self.trades if t.timestamp > cutoff and t.exit_price]
        if symbol:
            trades = [t for t in trades if t.symbol == symbol]
        
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'best_confirmations': [],
                'optimal_thresholds': {}
            }
        
        wins = [t for t in trades if t.win]
        losses = [t for t in trades if not t.win]
        
        stats = {
            'total_trades': len(trades),
            'win_rate': len(wins) / len(trades) * 100,
            'avg_win': np.mean([t.pnl_percentage for t in wins]) if wins else 0,
            'avg_loss': np.mean([t.pnl_percentage for t in losses]) if losses else 0,
            'profit_factor': abs(sum(t.pnl for t in wins) / sum(t.pnl for t in losses)) if losses else 0,
            'best_confirmations': self._analyze_confirmations(trades),
            'optimal_thresholds': self._find_optimal_thresholds(trades)
        }
        
        return stats
    
    def _analyze_confirmations(self, trades: List[TradeRecord]) -> List[Dict]:
        """Analyze which confirmations lead to wins"""
        confirmation_stats = {}
        
        for trade in trades:
            for conf in trade.confirmations:
                if conf not in confirmation_stats:
                    confirmation_stats[conf] = {'wins': 0, 'losses': 0, 'total': 0}
                
                confirmation_stats[conf]['total'] += 1
                if trade.win:
                    confirmation_stats[conf]['wins'] += 1
                else:
                    confirmation_stats[conf]['losses'] += 1
        
        # Calculate win rates
        results = []
        for conf, stats in confirmation_stats.items():
            win_rate = stats['wins'] / stats['total'] * 100 if stats['total'] > 0 else 0
            results.append({
                'confirmation': conf,
                'win_rate': win_rate,
                'occurrences': stats['total']
            })
        
        # Sort by win rate
        results.sort(key=lambda x: x['win_rate'], reverse=True)
        return results[:10]  # Top 10
    
    def _find_optimal_thresholds(self, trades: List[TradeRecord]) -> Dict:
        """Find optimal parameter thresholds based on performance"""
        if len(trades) < 20:
            return {}  # Not enough data
        
        # Group by RSI ranges and calculate win rates
        rsi_ranges = [(20, 30), (25, 35), (30, 40), (65, 75), (70, 80), (75, 85)]
        best_oversold = 30
        best_overbought = 70
        best_wr = 0
        
        for low, high in rsi_ranges[:3]:  # Oversold ranges
            range_trades = [t for t in trades if t.action == 'BUY' and low <= t.rsi <= high]
            if range_trades:
                wr = len([t for t in range_trades if t.win]) / len(range_trades)
                if wr > best_wr:
                    best_wr = wr
                    best_oversold = (low + high) / 2
        
        best_wr = 0
        for low, high in rsi_ranges[3:]:  # Overbought ranges
            range_trades = [t for t in trades if t.action == 'SELL' and low <= t.rsi <= high]
            if range_trades:
                wr = len([t for t in range_trades if t.win]) / len(range_trades)
                if wr > best_wr:
                    best_wr = wr
                    best_overbought = (low + high) / 2
        
        # Find optimal confirmation count
        conf_counts = [3, 4, 5, 6]
        best_conf = 3
        best_wr = 0
        
        for count in conf_counts:
            count_trades = [t for t in trades if t.confirmation_count >= count]
            if count_trades:
                wr = len([t for t in count_trades if t.win]) / len(count_trades)
                if wr > best_wr:
                    best_wr = wr
                    best_conf = count
        
        return {
            'rsi_oversold': best_oversold,
            'rsi_overbought': best_overbought,
            'min_confirmations': best_conf
        }
    
    def get_market_regime(self, symbol: str) -> Dict:
        """Identify current market regime based on recent performance"""
        recent_trades = [t for t in self.trades 
                        if t.symbol == symbol 
                        and t.timestamp > datetime.now() - timedelta(days=7)]
        
        if len(recent_trades) < 5:
            return {'regime': 'UNKNOWN', 'confidence': 0}
        
        # Analyze trend distribution
        trends = [t.trend for t in recent_trades]
        bullish = trends.count('BULLISH') / len(trends)
        bearish = trends.count('BEARISH') / len(trends)
        
        # Analyze volatility
        avg_volatility = np.mean([t.volatility for t in recent_trades])
        
        if bullish > 0.6:
            regime = 'TRENDING_UP'
        elif bearish > 0.6:
            regime = 'TRENDING_DOWN'
        elif avg_volatility > 2:
            regime = 'HIGH_VOLATILITY'
        elif avg_volatility < 1:
            regime = 'LOW_VOLATILITY'
        else:
            regime = 'RANGING'
        
        return {
            'regime': regime,
            'confidence': max(bullish, bearish),
            'volatility': avg_volatility
        }
    
    def get_symbol_profile(self, symbol: str) -> Dict:
        """Get ML profile for a specific symbol"""
        symbol_trades = [t for t in self.trades if t.symbol == symbol and t.exit_price]
        
        if len(symbol_trades) < 10:
            return {'status': 'insufficient_data', 'trades': len(symbol_trades)}
        
        return {
            'status': 'ready',
            'trades': len(symbol_trades),
            'performance': self.get_performance_stats(symbol),
            'regime': self.get_market_regime(symbol),
            'best_conditions': self._get_best_conditions(symbol_trades)
        }
    
    def _get_best_conditions(self, trades: List[TradeRecord]) -> Dict:
        """Find the best market conditions for a symbol"""
        winning_trades = [t for t in trades if t.win]
        
        if not winning_trades:
            return {}
        
        return {
            'avg_rsi': np.mean([t.rsi for t in winning_trades]),
            'avg_volume_ratio': np.mean([t.volume_ratio for t in winning_trades]),
            'preferred_trend': max(set([t.trend for t in winning_trades]), 
                                  key=[t.trend for t in winning_trades].count),
            'optimal_volatility': np.mean([t.volatility for t in winning_trades])
        }