"""
ML Manager - Central control for ML features
"""
import structlog
from typing import Optional, Dict
from .performance_tracker import PerformanceTracker
from .adaptive_optimizer import AdaptiveOptimizer

logger = structlog.get_logger(__name__)

class MLManager:
    """Manages all ML components and provides status reports"""
    
    def __init__(self, strategy, config: dict):
        self.strategy = strategy
        self.config = config
        self.enabled = config.get('ml_enabled', True)
        
        if self.enabled and hasattr(strategy, 'performance_tracker'):
            self.tracker = strategy.performance_tracker
            self.optimizer = strategy.optimizer
        else:
            self.tracker = None
            self.optimizer = None
            self.enabled = False
    
    def get_status(self) -> Dict:
        """Get comprehensive ML status"""
        if not self.enabled:
            return {
                'enabled': False,
                'reason': 'ML disabled in config or not initialized'
            }
        
        status = {
            'enabled': True,
            'mode': 'shadow' if self.optimizer.shadow_mode else 'live',
            'total_trades_tracked': len(self.tracker.trades),
            'trades_with_results': len([t for t in self.tracker.trades if t.exit_price]),
            'optimization_report': self.optimizer.get_adaptation_report(),
            'shadow_evaluation': self.optimizer.evaluate_shadow_performance(),
            'top_performing_symbols': self._get_top_symbols(),
            'best_confirmations': self._get_best_confirmations(),
            'recommendations': self._get_recommendations()
        }
        
        return status
    
    def _get_top_symbols(self, limit: int = 5) -> list:
        """Get top performing symbols"""
        if not self.tracker:
            return []
        
        symbol_stats = {}
        for trade in self.tracker.trades:
            if trade.exit_price and trade.symbol:
                if trade.symbol not in symbol_stats:
                    symbol_stats[trade.symbol] = {'wins': 0, 'total': 0}
                
                symbol_stats[trade.symbol]['total'] += 1
                if trade.win:
                    symbol_stats[trade.symbol]['wins'] += 1
        
        # Calculate win rates
        results = []
        for symbol, stats in symbol_stats.items():
            if stats['total'] >= 5:  # Need at least 5 trades
                win_rate = stats['wins'] / stats['total'] * 100
                results.append({
                    'symbol': symbol,
                    'win_rate': win_rate,
                    'trades': stats['total']
                })
        
        results.sort(key=lambda x: x['win_rate'], reverse=True)
        return results[:limit]
    
    def _get_best_confirmations(self) -> list:
        """Get best performing confirmations"""
        if not self.tracker:
            return []
        
        all_trades = [t for t in self.tracker.trades if t.exit_price]
        if not all_trades:
            return []
        
        stats = self.tracker._analyze_confirmations(all_trades)
        return stats[:5]  # Top 5
    
    def _get_recommendations(self) -> list:
        """Get ML recommendations"""
        recommendations = []
        
        if not self.optimizer:
            return recommendations
        
        # Check shadow performance
        shadow_eval = self.optimizer.evaluate_shadow_performance()
        
        if shadow_eval['status'] == 'ready':
            if shadow_eval['recommendation'] == 'enable':
                recommendations.append({
                    'type': 'action',
                    'priority': 'high',
                    'message': f"Enable live ML mode - Shadow win rate {shadow_eval['shadow_win_rate']:.1f}% vs baseline {shadow_eval['baseline_win_rate']:.1f}%"
                })
            else:
                recommendations.append({
                    'type': 'info',
                    'priority': 'medium',
                    'message': f"Continue shadow testing - Current win rate {shadow_eval['shadow_win_rate']:.1f}%"
                })
        else:
            recommendations.append({
                'type': 'info',
                'priority': 'low',
                'message': f"Need {20 - shadow_eval.get('trades', 0)} more trades for shadow evaluation"
            })
        
        # Check if we have enough data
        total_trades = len(self.tracker.trades) if self.tracker else 0
        if total_trades < 50:
            recommendations.append({
                'type': 'info',
                'priority': 'low',
                'message': f"Collecting data - {50 - total_trades} more trades needed for optimization"
            })
        
        return recommendations
    
    def enable_live_mode(self) -> bool:
        """Enable live ML mode if ready"""
        if not self.optimizer:
            logger.error("ML optimizer not available")
            return False
        
        result = self.optimizer.enable_live_mode()
        if result:
            logger.info("ML live mode enabled successfully")
        else:
            logger.warning("ML live mode not ready yet")
        
        return result
    
    def get_symbol_analysis(self, symbol: str) -> Dict:
        """Get detailed ML analysis for a symbol"""
        if not self.tracker:
            return {'status': 'ML not available'}
        
        return self.tracker.get_symbol_profile(symbol)