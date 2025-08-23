"""
Symbol Rotator for Testing Mode
Efficiently rotates through all symbols to find trading opportunities
while managing API rate limits
"""
import asyncio
from typing import List, Dict, Set, Optional
from datetime import datetime, timedelta
import random
import structlog
from collections import deque, defaultdict

logger = structlog.get_logger(__name__)

class SymbolRotator:
    """
    Manages rotation through all available symbols
    Prioritizes based on liquidity, volatility, and past performance
    """
    
    def __init__(self, all_symbols: List[str], max_concurrent: int = 20):
        self.all_symbols = all_symbols
        self.max_concurrent = max_concurrent
        
        # Symbol queues by priority
        self.high_priority = deque()  # Best performing symbols
        self.medium_priority = deque()  # Average symbols
        self.low_priority = deque()  # Poor performing or new symbols
        
        # Performance tracking
        self.symbol_stats = defaultdict(lambda: {
            'scans': 0,
            'signals': 0,
            'trades': 0,
            'wins': 0,
            'last_scan': None,
            'last_trade': None,
            'score': 50  # Start with neutral score
        })
        
        # Current scanning batch
        self.current_batch = []
        self.last_rotation = datetime.now()
        
        # Initialize queues
        self._initialize_queues()
        
    def _initialize_queues(self):
        """Initialize symbol queues with all symbols"""
        
        # Shuffle for initial randomization
        shuffled = self.all_symbols.copy()
        random.shuffle(shuffled)
        
        # Put most liquid symbols in high priority
        liquid_symbols = [
            "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT",
            "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "MATICUSDT",
            "LINKUSDT", "LTCUSDT", "UNIUSDT", "ATOMUSDT", "ETCUSDT"
        ]
        
        for symbol in shuffled:
            if symbol in liquid_symbols:
                self.high_priority.append(symbol)
            else:
                self.medium_priority.append(symbol)
        
        logger.info(f"Initialized rotator with {len(self.all_symbols)} symbols")
        logger.info(f"High priority: {len(self.high_priority)}, Medium: {len(self.medium_priority)}")
    
    def get_next_batch(self) -> List[str]:
        """Get next batch of symbols to scan"""
        
        batch = []
        
        # Take symbols from queues based on priority
        # 50% from high priority, 35% from medium, 15% from low
        high_count = min(len(self.high_priority), self.max_concurrent // 2)
        medium_count = min(len(self.medium_priority), int(self.max_concurrent * 0.35))
        low_count = min(len(self.low_priority), self.max_concurrent - high_count - medium_count)
        
        # Add from high priority
        for _ in range(high_count):
            if self.high_priority:
                symbol = self.high_priority.popleft()
                batch.append(symbol)
                self.high_priority.append(symbol)  # Add back to end
        
        # Add from medium priority
        for _ in range(medium_count):
            if self.medium_priority:
                symbol = self.medium_priority.popleft()
                batch.append(symbol)
                self.medium_priority.append(symbol)  # Add back to end
        
        # Add from low priority
        for _ in range(low_count):
            if self.low_priority:
                symbol = self.low_priority.popleft()
                batch.append(symbol)
                self.low_priority.append(symbol)  # Add back to end
        
        # Fill remaining slots from medium if needed
        while len(batch) < self.max_concurrent and self.medium_priority:
            symbol = self.medium_priority.popleft()
            batch.append(symbol)
            self.medium_priority.append(symbol)
        
        self.current_batch = batch
        self.last_rotation = datetime.now()
        
        # Update scan counts
        for symbol in batch:
            self.symbol_stats[symbol]['scans'] += 1
            self.symbol_stats[symbol]['last_scan'] = datetime.now()
        
        logger.debug(f"Next batch: {len(batch)} symbols")
        return batch
    
    def record_signal(self, symbol: str):
        """Record that a signal was generated for a symbol"""
        
        self.symbol_stats[symbol]['signals'] += 1
        self._update_symbol_score(symbol, 5)  # Boost score for generating signals
    
    def record_trade(self, symbol: str, won: bool, pnl: float):
        """Record trade result for a symbol"""
        
        stats = self.symbol_stats[symbol]
        stats['trades'] += 1
        stats['last_trade'] = datetime.now()
        
        if won:
            stats['wins'] += 1
            self._update_symbol_score(symbol, 10)  # Big boost for wins
        else:
            self._update_symbol_score(symbol, -5)  # Small penalty for losses
        
        # Additional score adjustment based on PnL magnitude
        if abs(pnl) > 2:  # Big win or loss
            score_change = 5 if pnl > 0 else -3
            self._update_symbol_score(symbol, score_change)
    
    def _update_symbol_score(self, symbol: str, change: float):
        """Update symbol score and rebalance queues if needed"""
        
        old_score = self.symbol_stats[symbol]['score']
        new_score = max(0, min(100, old_score + change))
        self.symbol_stats[symbol]['score'] = new_score
        
        # Rebalance queues based on new score
        if new_score >= 70 and old_score < 70:
            # Move to high priority
            self._move_symbol_to_priority(symbol, 'high')
        elif new_score <= 30 and old_score > 30:
            # Move to low priority
            self._move_symbol_to_priority(symbol, 'low')
        elif 30 < new_score < 70:
            # Should be in medium priority
            if new_score >= 70 or new_score <= 30:
                self._move_symbol_to_priority(symbol, 'medium')
    
    def _move_symbol_to_priority(self, symbol: str, priority: str):
        """Move symbol between priority queues"""
        
        # Remove from all queues
        for queue in [self.high_priority, self.medium_priority, self.low_priority]:
            if symbol in queue:
                queue.remove(symbol)
        
        # Add to appropriate queue
        if priority == 'high':
            self.high_priority.append(symbol)
        elif priority == 'medium':
            self.medium_priority.append(symbol)
        else:
            self.low_priority.append(symbol)
        
        logger.debug(f"Moved {symbol} to {priority} priority (score: {self.symbol_stats[symbol]['score']})")
    
    def get_symbol_recommendations(self, top_n: int = 10) -> List[Dict]:
        """Get top performing symbols"""
        
        recommendations = []
        
        for symbol, stats in self.symbol_stats.items():
            if stats['trades'] > 0:
                win_rate = stats['wins'] / stats['trades']
                signal_rate = stats['signals'] / max(1, stats['scans'])
                
                recommendations.append({
                    'symbol': symbol,
                    'score': stats['score'],
                    'win_rate': win_rate,
                    'signal_rate': signal_rate,
                    'trades': stats['trades']
                })
        
        # Sort by score
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        return recommendations[:top_n]
    
    def should_skip_symbol(self, symbol: str) -> bool:
        """Check if symbol should be skipped"""
        
        stats = self.symbol_stats[symbol]
        
        # Skip if recently scanned (within 1 minute)
        if stats['last_scan']:
            if (datetime.now() - stats['last_scan']).total_seconds() < 60:
                return True
        
        # Skip if poor performer with enough data
        if stats['trades'] >= 10 and stats['score'] < 20:
            # Only scan every 10 minutes for poor performers
            if stats['last_scan']:
                if (datetime.now() - stats['last_scan']).total_seconds() < 600:
                    return True
        
        return False
    
    def get_statistics(self) -> Dict:
        """Get rotation statistics"""
        
        total_scanned = sum(1 for s in self.symbol_stats.values() if s['scans'] > 0)
        total_signals = sum(s['signals'] for s in self.symbol_stats.values())
        total_trades = sum(s['trades'] for s in self.symbol_stats.values())
        total_wins = sum(s['wins'] for s in self.symbol_stats.values())
        
        return {
            'total_symbols': len(self.all_symbols),
            'symbols_scanned': total_scanned,
            'total_signals': total_signals,
            'total_trades': total_trades,
            'win_rate': (total_wins / total_trades * 100) if total_trades > 0 else 0,
            'high_priority_count': len(self.high_priority),
            'medium_priority_count': len(self.medium_priority),
            'low_priority_count': len(self.low_priority),
            'current_batch_size': len(self.current_batch),
            'top_performers': self.get_symbol_recommendations(5)
        }