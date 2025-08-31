"""
Adaptive Parameter Optimizer
Gradually adjusts strategy parameters based on performance
"""
import numpy as np
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
import json
import os
import structlog

logger = structlog.get_logger(__name__)

class AdaptiveOptimizer:
    """Safely optimizes strategy parameters based on ML insights"""
    
    def __init__(self, performance_tracker, config: dict):
        self.tracker = performance_tracker
        self.base_config = config.copy()
        
        # Safety limits - max deviation from base config
        self.max_deviation = 0.15  # 15% max change
        self.min_trades_required = 50  # Need 50 trades before adapting
        self.adaptation_rate = 0.1  # How fast to adapt (0.1 = 10% per update)
        
        # Adaptation state
        self.adaptations_file = "ml_data/adaptations.json"
        self.current_adaptations = self._load_adaptations()
        self.last_adaptation = datetime.now() - timedelta(hours=1)
        self.adaptation_interval = timedelta(hours=4)  # Adapt every 4 hours
        
        # Performance tracking
        self.baseline_win_rate = 0
        self.current_win_rate = 0
        self.improvement_threshold = 0.02  # Need 2% improvement to keep changes
        
        # Shadow mode - test without affecting real trades
        self.shadow_mode = True
        self.shadow_performance = []
        
        logger.info("Adaptive optimizer initialized in shadow mode")
    
    def _load_adaptations(self) -> Dict:
        """Load saved adaptations"""
        if os.path.exists(self.adaptations_file):
            try:
                with open(self.adaptations_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {}
    
    def _save_adaptations(self):
        """Save current adaptations"""
        os.makedirs(os.path.dirname(self.adaptations_file), exist_ok=True)
        with open(self.adaptations_file, 'w') as f:
            json.dump(self.current_adaptations, f, indent=2)
    
    def get_optimized_params(self, symbol: str, current_market: Dict) -> Dict:
        """Get optimized parameters for current conditions"""
        # Start with base config
        params = self.base_config.copy()
        
        # Check if we should adapt
        if not self._should_adapt():
            return params
        
        # Get performance data
        stats = self.tracker.get_performance_stats(symbol, days=7)
        
        if stats['total_trades'] < self.min_trades_required:
            logger.debug(f"Not enough trades for {symbol}: {stats['total_trades']}")
            return params
        
        # Get optimal thresholds from ML analysis
        optimal = stats.get('optimal_thresholds', {})
        
        # Apply gradual adaptations with safety limits
        if 'rsi_oversold' in optimal:
            adapted_value = self._safe_adapt(
                params['rsi_oversold'],
                optimal['rsi_oversold'],
                20, 40  # Valid range for RSI oversold
            )
            params['rsi_oversold'] = adapted_value
            params['scalp_rsi_oversold'] = adapted_value - 5
        
        if 'rsi_overbought' in optimal:
            adapted_value = self._safe_adapt(
                params['rsi_overbought'],
                optimal['rsi_overbought'],
                60, 80  # Valid range for RSI overbought
            )
            params['rsi_overbought'] = adapted_value
            params['scalp_rsi_overbought'] = adapted_value + 5
        
        if 'min_confirmations' in optimal:
            adapted_value = self._safe_adapt(
                params.get('min_confirmations', 3),
                optimal['min_confirmations'],
                2, 5  # Valid range for confirmations
            )
            params['min_confirmations'] = int(adapted_value)
        
        # Adapt based on market regime
        regime = self.tracker.get_market_regime(symbol)
        params = self._adapt_for_regime(params, regime)
        
        # Track adaptations
        self.current_adaptations[symbol] = {
            'timestamp': datetime.now().isoformat(),
            'adaptations': {
                'rsi_oversold': params['rsi_oversold'],
                'rsi_overbought': params['rsi_overbought'],
                'min_confirmations': params.get('min_confirmations', 3)
            },
            'regime': regime['regime'],
            'win_rate': stats['win_rate']
        }
        
        self._save_adaptations()
        
        if self.shadow_mode:
            logger.info(f"Shadow adaptations for {symbol}: RSI {params['rsi_oversold']}/{params['rsi_overbought']}, Conf: {params.get('min_confirmations', 3)}")
        
        return params
    
    def _should_adapt(self) -> bool:
        """Check if enough time has passed for adaptation"""
        if self.shadow_mode:
            return True  # Always adapt in shadow mode for testing
        
        if datetime.now() - self.last_adaptation < self.adaptation_interval:
            return False
        
        self.last_adaptation = datetime.now()
        return True
    
    def _safe_adapt(self, current: float, target: float, 
                   min_val: float, max_val: float) -> float:
        """Safely adapt a parameter with limits"""
        # Calculate maximum allowed change
        max_change = current * self.max_deviation
        
        # Calculate desired change with adaptation rate
        desired_change = (target - current) * self.adaptation_rate
        
        # Limit the change
        actual_change = max(-max_change, min(max_change, desired_change))
        
        # Apply change with bounds
        new_value = current + actual_change
        new_value = max(min_val, min(max_val, new_value))
        
        return round(new_value, 2)
    
    def _adapt_for_regime(self, params: Dict, regime: Dict) -> Dict:
        """Adapt parameters based on market regime"""
        regime_type = regime.get('regime', 'UNKNOWN')
        
        if regime_type == 'TRENDING_UP':
            # In uptrend, be more aggressive with buys
            params['min_confirmations'] = max(2, params.get('min_confirmations', 3) - 1)
            params['min_risk_reward'] = max(1.0, params.get('min_risk_reward', 1.2) - 0.1)
            
        elif regime_type == 'TRENDING_DOWN':
            # In downtrend, be more conservative
            params['min_confirmations'] = min(5, params.get('min_confirmations', 3) + 1)
            params['min_risk_reward'] = min(1.5, params.get('min_risk_reward', 1.2) + 0.1)
            
        elif regime_type == 'HIGH_VOLATILITY':
            # In high volatility, use wider stops
            params['scalp_rr_sl_multiplier'] = min(1.5, params.get('scalp_rr_sl_multiplier', 1.0) + 0.2)
            params['min_confirmations'] = min(5, params.get('min_confirmations', 3) + 1)
            
        elif regime_type == 'LOW_VOLATILITY':
            # In low volatility, use tighter stops
            params['scalp_rr_sl_multiplier'] = max(0.8, params.get('scalp_rr_sl_multiplier', 1.0) - 0.1)
            
        return params
    
    def evaluate_shadow_performance(self) -> Dict:
        """Evaluate shadow mode performance"""
        if len(self.shadow_performance) < 20:
            return {'status': 'insufficient_data', 'trades': len(self.shadow_performance)}
        
        wins = len([t for t in self.shadow_performance if t['win']])
        total = len(self.shadow_performance)
        
        shadow_win_rate = wins / total * 100 if total > 0 else 0
        
        return {
            'status': 'ready',
            'shadow_trades': total,
            'shadow_win_rate': shadow_win_rate,
            'baseline_win_rate': self.baseline_win_rate,
            'improvement': shadow_win_rate - self.baseline_win_rate,
            'recommendation': 'enable' if shadow_win_rate > self.baseline_win_rate + 2 else 'wait'
        }
    
    def record_shadow_trade(self, symbol: str, win: bool, params_used: Dict):
        """Record shadow trade result"""
        self.shadow_performance.append({
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'win': win,
            'params': params_used
        })
        
        # Keep only recent trades
        if len(self.shadow_performance) > 100:
            self.shadow_performance = self.shadow_performance[-100:]
    
    def enable_live_mode(self) -> bool:
        """Enable live adaptations if shadow performance is good"""
        eval_result = self.evaluate_shadow_performance()
        
        if eval_result['status'] != 'ready':
            logger.warning("Not enough shadow trades to enable live mode")
            return False
        
        if eval_result['recommendation'] != 'enable':
            logger.warning(f"Shadow performance not sufficient: {eval_result['shadow_win_rate']:.1f}%")
            return False
        
        self.shadow_mode = False
        logger.info(f"Live mode enabled! Shadow win rate: {eval_result['shadow_win_rate']:.1f}%")
        return True
    
    def get_adaptation_report(self) -> Dict:
        """Get detailed adaptation report"""
        report = {
            'mode': 'shadow' if self.shadow_mode else 'live',
            'total_adaptations': len(self.current_adaptations),
            'symbols_adapted': list(self.current_adaptations.keys()),
            'shadow_evaluation': self.evaluate_shadow_performance() if self.shadow_mode else None,
            'recent_adaptations': []
        }
        
        # Get recent adaptations
        for symbol, data in self.current_adaptations.items():
            if 'timestamp' in data:
                ts = datetime.fromisoformat(data['timestamp'])
                if datetime.now() - ts < timedelta(days=1):
                    report['recent_adaptations'].append({
                        'symbol': symbol,
                        'time_ago': str(datetime.now() - ts),
                        'adaptations': data['adaptations'],
                        'win_rate': data.get('win_rate', 0)
                    })
        
        return report