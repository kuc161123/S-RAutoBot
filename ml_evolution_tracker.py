"""
ML Evolution Performance Tracker
Tracks how evolution system would have performed vs general model
Helps make data-driven decision about when to enable
"""
import json
import logging
from datetime import datetime
from typing import Dict, List
import os
import redis
import numpy as np

logger = logging.getLogger(__name__)

class EvolutionPerformanceTracker:
    """Track shadow performance of ML Evolution system"""
    
    def __init__(self):
        self.performance_log = []  # List of comparisons
        self.symbol_stats = {}  # Per-symbol performance
        self.redis_client = None
        self._init_redis()
        self._load_state()
    
    def _init_redis(self):
        """Initialize Redis for persistence"""
        try:
            redis_url = os.getenv('REDIS_URL')
            if redis_url:
                self.redis_client = redis.from_url(redis_url, decode_responses=True)
                logger.info("Evolution tracker connected to Redis")
        except Exception as e:
            logger.warning(f"Evolution tracker Redis failed: {e}")
    
    def record_decision_comparison(
        self, 
        symbol: str,
        general_score: float,
        evolution_score: float,
        threshold: float,
        actual_outcome: str = None  # 'win', 'loss', or None if unknown yet
    ):
        """Record how each system would have decided"""
        general_decision = general_score >= threshold
        evolution_decision = evolution_score >= threshold
        
        comparison = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'general_score': general_score,
            'evolution_score': evolution_score,
            'threshold': threshold,
            'general_would_take': general_decision,
            'evolution_would_take': evolution_decision,
            'agreed': general_decision == evolution_decision,
            'outcome': actual_outcome
        }
        
        self.performance_log.append(comparison)
        
        # Update symbol stats
        if symbol not in self.symbol_stats:
            self.symbol_stats[symbol] = {
                'total_signals': 0,
                'agreements': 0,
                'general_better': 0,
                'evolution_better': 0,
                'both_right': 0,
                'both_wrong': 0
            }
        
        stats = self.symbol_stats[symbol]
        stats['total_signals'] += 1
        if comparison['agreed']:
            stats['agreements'] += 1
        
        # Log interesting divergences
        if not comparison['agreed']:
            logger.info(f"[{symbol}] DIVERGENCE: General {general_score:.1f} → {'TAKE' if general_decision else 'SKIP'}, "
                       f"Evolution {evolution_score:.1f} → {'TAKE' if evolution_decision else 'SKIP'}")
        
        self._save_state()
    
    def update_outcome(self, symbol: str, outcome: str, timestamp: str = None):
        """Update the outcome when trade completes"""
        # Find recent comparison for this symbol
        for comp in reversed(self.performance_log[-100:]):  # Check last 100
            if comp['symbol'] == symbol and comp['outcome'] is None:
                comp['outcome'] = outcome
                
                # Update performance stats
                stats = self.symbol_stats[symbol]
                general_correct = (comp['general_would_take'] and outcome == 'win') or \
                                 (not comp['general_would_take'] and outcome == 'loss')
                evolution_correct = (comp['evolution_would_take'] and outcome == 'win') or \
                                   (not comp['evolution_would_take'] and outcome == 'loss')
                
                if general_correct and evolution_correct:
                    stats['both_right'] += 1
                elif not general_correct and not evolution_correct:
                    stats['both_wrong'] += 1
                elif general_correct and not evolution_correct:
                    stats['general_better'] += 1
                elif evolution_correct and not general_correct:
                    stats['evolution_better'] += 1
                    logger.info(f"[{symbol}] Evolution would have been RIGHT where general was WRONG!")
                
                break
        
        self._save_state()
    
    def get_performance_summary(self) -> Dict:
        """Get overall performance comparison"""
        if not self.performance_log:
            return {'status': 'No data yet'}
        
        # Overall stats
        total = len(self.performance_log)
        agreements = sum(1 for c in self.performance_log if c['agreed'])
        
        # Performance comparison (only completed trades)
        completed = [c for c in self.performance_log if c['outcome'] is not None]
        
        if not completed:
            return {
                'total_signals': total,
                'agreement_rate': (agreements / total) * 100,
                'performance': 'Waiting for outcomes...'
            }
        
        general_wins = evolution_wins = 0
        for comp in completed:
            if comp['outcome'] == 'win':
                if comp['general_would_take']:
                    general_wins += 1
                if comp['evolution_would_take']:
                    evolution_wins += 1
        
        # Symbol-specific insights
        symbol_insights = {}
        for symbol, stats in self.symbol_stats.items():
            total_decided = stats['general_better'] + stats['evolution_better'] + \
                          stats['both_right'] + stats['both_wrong']
            if total_decided > 0:
                evolution_advantage = stats['evolution_better'] - stats['general_better']
                symbol_insights[symbol] = {
                    'signals': stats['total_signals'],
                    'evolution_advantage': evolution_advantage,
                    'agreement_rate': (stats['agreements'] / stats['total_signals']) * 100
                }
        
        return {
            'total_signals': total,
            'agreement_rate': (agreements / total) * 100,
            'completed_comparisons': len(completed),
            'general_win_rate': (general_wins / len(completed)) * 100 if completed else 0,
            'evolution_win_rate': (evolution_wins / len(completed)) * 100 if completed else 0,
            'symbol_insights': symbol_insights,
            'recommendation': self._get_recommendation()
        }
    
    def _get_recommendation(self) -> str:
        """Recommend whether to enable evolution based on performance"""
        if len(self.performance_log) < 100:
            return "Too early - need 100+ signals"
        
        completed = [c for c in self.performance_log if c['outcome'] is not None]
        if len(completed) < 50:
            return "Need 50+ completed trades for reliable assessment"
        
        # Count where evolution was better
        evolution_better_count = sum(stats['evolution_better'] for stats in self.symbol_stats.values())
        general_better_count = sum(stats['general_better'] for stats in self.symbol_stats.values())
        
        if evolution_better_count > general_better_count * 1.2:  # 20% better
            return "RECOMMENDED: Evolution showing 20%+ improvement"
        elif evolution_better_count > general_better_count:
            return "Promising: Evolution slightly better, monitor longer"
        else:
            return "Not yet: General model still performing better"
    
    def _save_state(self):
        """Save to Redis"""
        if not self.redis_client:
            return
        
        try:
            # Keep last 1000 comparisons
            recent_log = self.performance_log[-1000:]
            self.redis_client.set('evolution:performance_log', json.dumps(recent_log))
            self.redis_client.set('evolution:symbol_stats', json.dumps(self.symbol_stats))
        except Exception as e:
            logger.error(f"Failed to save evolution tracker: {e}")
    
    def _load_state(self):
        """Load from Redis"""
        if not self.redis_client:
            return
        
        try:
            log_data = self.redis_client.get('evolution:performance_log')
            if log_data:
                self.performance_log = json.loads(log_data)
            
            stats_data = self.redis_client.get('evolution:symbol_stats')
            if stats_data:
                self.symbol_stats = json.loads(stats_data)
                
            logger.info(f"Loaded {len(self.performance_log)} evolution comparisons")
        except Exception as e:
            logger.error(f"Failed to load evolution tracker: {e}")

# Global instance
_evolution_tracker = None

def get_evolution_tracker() -> EvolutionPerformanceTracker:
    """Get or create global tracker"""
    global _evolution_tracker
    if _evolution_tracker is None:
        _evolution_tracker = EvolutionPerformanceTracker()
    return _evolution_tracker