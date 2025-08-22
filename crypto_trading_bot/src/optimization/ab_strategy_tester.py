"""
A/B Strategy Testing System
Compare multiple strategies in real-time to find the best performer
"""
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import numpy as np
from scipy import stats
import structlog
import json

logger = structlog.get_logger(__name__)


@dataclass
class StrategyVariant:
    """Single strategy variant for A/B testing"""
    name: str
    config: Dict[str, Any]
    
    # Performance metrics
    trades_count: int = 0
    winning_trades: int = 0
    total_pnl: float = 0.0
    total_pnl_percent: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    
    # Statistical significance
    confidence_level: float = 0.0
    p_value: float = 1.0
    is_significant: bool = False
    
    # Trade history
    trade_returns: List[float] = field(default_factory=list)
    daily_returns: List[float] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    
    # Allocation
    allocation_percent: float = 0.0
    capital_allocated: float = 0.0


class ABStrategyTester:
    """
    A/B testing system for trading strategies
    Uses statistical significance to determine best strategies
    """
    
    def __init__(self, intelligent_engine):
        self.engine = intelligent_engine
        self.variants: Dict[str, StrategyVariant] = {}
        self.control_variant: Optional[StrategyVariant] = None
        self.test_duration = timedelta(days=30)
        self.test_start_time = datetime.now()
        self.minimum_trades = 30  # Min trades for statistical significance
        self.confidence_threshold = 0.95
        
    async def create_strategy_variants(self) -> Dict[str, StrategyVariant]:
        """Create multiple strategy variants for testing"""
        
        variants = {
            # Control - Conservative baseline
            "control_conservative": StrategyVariant(
                name="Conservative Control",
                config={
                    "risk_per_trade": 0.01,
                    "max_positions": 3,
                    "ml_confidence_threshold": 0.70,
                    "zone_score_threshold": 0.75,
                    "take_profit_ratio": 2.0,
                    "stop_loss_ratio": 1.0,
                    "use_trailing_stop": False,
                    "regime_filter": ["trending_strong", "trending_weak"]
                }
            ),
            
            # Variant A - Aggressive
            "variant_aggressive": StrategyVariant(
                name="Aggressive Trading",
                config={
                    "risk_per_trade": 0.02,
                    "max_positions": 5,
                    "ml_confidence_threshold": 0.60,
                    "zone_score_threshold": 0.65,
                    "take_profit_ratio": 3.0,
                    "stop_loss_ratio": 1.5,
                    "use_trailing_stop": True,
                    "regime_filter": ["trending_strong", "breakout", "volatile"]
                }
            ),
            
            # Variant B - ML-Heavy
            "variant_ml_focused": StrategyVariant(
                name="ML-Focused",
                config={
                    "risk_per_trade": 0.015,
                    "max_positions": 4,
                    "ml_confidence_threshold": 0.75,
                    "zone_score_threshold": 0.60,
                    "take_profit_ratio": 2.5,
                    "stop_loss_ratio": 1.2,
                    "use_trailing_stop": True,
                    "ml_weight": 0.7,  # Higher weight on ML signals
                    "technical_weight": 0.3
                }
            ),
            
            # Variant C - Scalping
            "variant_scalping": StrategyVariant(
                name="Scalping Strategy",
                config={
                    "risk_per_trade": 0.005,
                    "max_positions": 10,
                    "ml_confidence_threshold": 0.65,
                    "zone_score_threshold": 0.70,
                    "take_profit_ratio": 1.5,
                    "stop_loss_ratio": 0.75,
                    "use_trailing_stop": False,
                    "timeframe": "5m",
                    "quick_exit": True
                }
            ),
            
            # Variant D - Swing Trading
            "variant_swing": StrategyVariant(
                name="Swing Trading",
                config={
                    "risk_per_trade": 0.025,
                    "max_positions": 2,
                    "ml_confidence_threshold": 0.80,
                    "zone_score_threshold": 0.80,
                    "take_profit_ratio": 5.0,
                    "stop_loss_ratio": 2.0,
                    "use_trailing_stop": True,
                    "timeframe": "4h",
                    "hold_duration_min": 24  # Hours
                }
            ),
            
            # Variant E - Mean Reversion
            "variant_mean_reversion": StrategyVariant(
                name="Mean Reversion",
                config={
                    "risk_per_trade": 0.015,
                    "max_positions": 4,
                    "ml_confidence_threshold": 0.65,
                    "zone_score_threshold": 0.70,
                    "take_profit_ratio": 1.8,
                    "stop_loss_ratio": 1.0,
                    "use_trailing_stop": False,
                    "regime_filter": ["ranging_tight", "ranging_wide"],
                    "use_bollinger_bands": True,
                    "use_rsi_extremes": True
                }
            ),
            
            # Variant F - Momentum
            "variant_momentum": StrategyVariant(
                name="Momentum Trading",
                config={
                    "risk_per_trade": 0.02,
                    "max_positions": 4,
                    "ml_confidence_threshold": 0.65,
                    "zone_score_threshold": 0.65,
                    "take_profit_ratio": 3.5,
                    "stop_loss_ratio": 1.5,
                    "use_trailing_stop": True,
                    "regime_filter": ["trending_strong", "breakout"],
                    "volume_threshold": 1.5,
                    "momentum_period": 20
                }
            ),
            
            # Variant G - Hybrid Adaptive
            "variant_adaptive": StrategyVariant(
                name="Adaptive Hybrid",
                config={
                    "risk_per_trade": 0.015,
                    "max_positions": 4,
                    "ml_confidence_threshold": 0.68,
                    "zone_score_threshold": 0.68,
                    "take_profit_ratio": "adaptive",  # Changes based on market
                    "stop_loss_ratio": "adaptive",
                    "use_trailing_stop": True,
                    "adaptive_mode": True,
                    "regime_adaptive": True  # Changes strategy based on regime
                }
            )
        }
        
        # Set control variant
        self.control_variant = variants["control_conservative"]
        self.variants = variants
        
        # Initial equal allocation
        num_variants = len(variants)
        for variant in variants.values():
            variant.allocation_percent = 1.0 / num_variants
            variant.equity_curve = [10000 / num_variants]  # Starting capital
        
        logger.info(f"Created {num_variants} strategy variants for A/B testing")
        
        return variants
    
    async def execute_with_variant(
        self,
        variant_name: str,
        symbol: str,
        market_data: Dict
    ) -> Optional[Dict]:
        """Execute trading decision with specific variant"""
        
        if variant_name not in self.variants:
            return None
        
        variant = self.variants[variant_name]
        config = variant.config
        
        try:
            # Apply variant configuration
            signal = await self._generate_variant_signal(
                variant=variant,
                symbol=symbol,
                market_data=market_data,
                config=config
            )
            
            if signal:
                # Track execution
                await self._track_variant_execution(variant, signal)
            
            return signal
            
        except Exception as e:
            logger.error(f"Error executing variant {variant_name}: {e}")
            return None
    
    async def _generate_variant_signal(
        self,
        variant: StrategyVariant,
        symbol: str,
        market_data: Dict,
        config: Dict
    ) -> Optional[Dict]:
        """Generate signal based on variant configuration"""
        
        # Get ML prediction
        ml_confidence = market_data.get('ml_prediction', {}).get('confidence', 0)
        ml_direction = market_data.get('ml_prediction', {}).get('direction', 'neutral')
        
        # Check ML confidence threshold
        if ml_confidence < config['ml_confidence_threshold']:
            return None
        
        # Get zone score
        zone_score = market_data.get('zone', {}).get('composite_score', 0)
        
        # Check zone threshold
        if zone_score < config['zone_score_threshold']:
            return None
        
        # Check market regime filter
        if 'regime_filter' in config:
            current_regime = market_data.get('market_regime', 'unknown')
            if current_regime not in config['regime_filter']:
                return None
        
        # Volume check for momentum strategies
        if 'volume_threshold' in config:
            volume_ratio = market_data.get('volume_ratio', 1.0)
            if volume_ratio < config['volume_threshold']:
                return None
        
        # Generate signal
        signal = {
            'symbol': symbol,
            'variant': variant.name,
            'direction': ml_direction,
            'confidence': ml_confidence,
            'zone_score': zone_score,
            'entry_price': market_data.get('price', 0),
            'risk_per_trade': config['risk_per_trade'],
            'config': config
        }
        
        # Calculate stop loss and take profit
        atr = market_data.get('atr', 0)
        
        if config.get('adaptive_mode'):
            # Adaptive SL/TP based on market conditions
            volatility = market_data.get('volatility', 'normal')
            if volatility == 'high':
                sl_ratio = 2.0
                tp_ratio = 4.0
            elif volatility == 'low':
                sl_ratio = 1.0
                tp_ratio = 2.0
            else:
                sl_ratio = 1.5
                tp_ratio = 3.0
        else:
            sl_ratio = config['stop_loss_ratio']
            tp_ratio = config['take_profit_ratio']
        
        if ml_direction == 'long':
            signal['stop_loss'] = signal['entry_price'] - (atr * sl_ratio)
            signal['take_profit'] = signal['entry_price'] + (atr * tp_ratio)
        else:
            signal['stop_loss'] = signal['entry_price'] + (atr * sl_ratio)
            signal['take_profit'] = signal['entry_price'] - (atr * tp_ratio)
        
        signal['use_trailing_stop'] = config.get('use_trailing_stop', False)
        
        return signal
    
    async def _track_variant_execution(self, variant: StrategyVariant, signal: Dict):
        """Track variant execution for performance measurement"""
        
        variant.trades_count += 1
        
        # Simulate trade result (in production, track actual results)
        # This would be updated when trade closes
        simulated_return = np.random.normal(0.002, 0.02)  # Placeholder
        
        variant.trade_returns.append(simulated_return)
        
        # Update equity curve
        latest_equity = variant.equity_curve[-1]
        new_equity = latest_equity * (1 + simulated_return)
        variant.equity_curve.append(new_equity)
        
        # Update PnL
        variant.total_pnl += latest_equity * simulated_return
        variant.total_pnl_percent = (new_equity - variant.equity_curve[0]) / variant.equity_curve[0]
        
        logger.debug(f"Variant {variant.name} executed trade #{variant.trades_count}")
    
    async def update_variant_performance(
        self,
        variant_name: str,
        trade_result: Dict
    ):
        """Update variant performance with actual trade result"""
        
        if variant_name not in self.variants:
            return
        
        variant = self.variants[variant_name]
        
        # Update metrics
        pnl_percent = trade_result.get('pnl_percent', 0)
        variant.trade_returns.append(pnl_percent)
        
        if pnl_percent > 0:
            variant.winning_trades += 1
        
        # Update equity
        latest_equity = variant.equity_curve[-1]
        new_equity = latest_equity * (1 + pnl_percent)
        variant.equity_curve.append(new_equity)
        
        # Calculate updated metrics
        await self._calculate_variant_metrics(variant)
    
    async def _calculate_variant_metrics(self, variant: StrategyVariant):
        """Calculate performance metrics for variant"""
        
        if not variant.trade_returns:
            return
        
        returns = np.array(variant.trade_returns)
        
        # Win rate
        win_rate = variant.winning_trades / variant.trades_count if variant.trades_count > 0 else 0
        
        # Sharpe ratio
        if returns.std() > 0:
            variant.sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
        
        # Max drawdown
        equity = np.array(variant.equity_curve)
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max
        variant.max_drawdown = drawdown.min()
    
    async def run_statistical_tests(self) -> Dict[str, Any]:
        """Run statistical significance tests between variants"""
        
        results = {}
        
        if not self.control_variant:
            return results
        
        control_returns = np.array(self.control_variant.trade_returns)
        
        if len(control_returns) < self.minimum_trades:
            logger.info(f"Not enough trades for statistical significance "
                       f"({len(control_returns)}/{self.minimum_trades})")
            return results
        
        for name, variant in self.variants.items():
            if name == "control_conservative":
                continue
            
            variant_returns = np.array(variant.trade_returns)
            
            if len(variant_returns) < self.minimum_trades:
                continue
            
            # T-test for difference in means
            t_stat, p_value = stats.ttest_ind(variant_returns, control_returns)
            
            # Calculate confidence level
            confidence = 1 - p_value
            
            # Check if significant
            is_significant = p_value < (1 - self.confidence_threshold)
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt((variant_returns.std()**2 + control_returns.std()**2) / 2)
            effect_size = (variant_returns.mean() - control_returns.mean()) / pooled_std if pooled_std > 0 else 0
            
            # Update variant
            variant.p_value = p_value
            variant.confidence_level = confidence
            variant.is_significant = is_significant
            
            results[name] = {
                'variant_name': variant.name,
                'p_value': p_value,
                'confidence': confidence,
                'is_significant': is_significant,
                'effect_size': effect_size,
                'variant_mean_return': variant_returns.mean(),
                'control_mean_return': control_returns.mean(),
                'improvement': (variant_returns.mean() - control_returns.mean()) / abs(control_returns.mean()) 
                              if control_returns.mean() != 0 else 0
            }
            
            if is_significant:
                if variant_returns.mean() > control_returns.mean():
                    logger.info(f"✅ Variant {variant.name} significantly BETTER than control "
                               f"(p={p_value:.4f}, improvement={results[name]['improvement']:.2%})")
                else:
                    logger.info(f"❌ Variant {variant.name} significantly WORSE than control "
                               f"(p={p_value:.4f}, decline={results[name]['improvement']:.2%})")
        
        return results
    
    async def adaptive_allocation(self) -> Dict[str, float]:
        """Adaptively allocate capital based on performance"""
        
        allocations = {}
        
        # Calculate performance scores
        scores = {}
        for name, variant in self.variants.items():
            if variant.trades_count < 10:
                # Not enough data, use equal allocation
                scores[name] = 1.0
            else:
                # Score based on Sharpe ratio and win rate
                win_rate = variant.winning_trades / variant.trades_count
                sharpe_score = max(0, variant.sharpe_ratio)
                
                # Weighted score
                scores[name] = (sharpe_score * 0.6) + (win_rate * 0.4)
        
        # Normalize scores to allocations
        total_score = sum(scores.values())
        
        if total_score > 0:
            for name, score in scores.items():
                allocation = score / total_score
                
                # Apply bounds (min 5%, max 40%)
                allocation = max(0.05, min(0.40, allocation))
                
                allocations[name] = allocation
                self.variants[name].allocation_percent = allocation
        else:
            # Equal allocation if no scores
            num_variants = len(self.variants)
            for name in self.variants:
                allocations[name] = 1.0 / num_variants
        
        # Ensure allocations sum to 1
        total_alloc = sum(allocations.values())
        if total_alloc > 0:
            for name in allocations:
                allocations[name] /= total_alloc
        
        logger.info(f"Updated allocations: {allocations}")
        
        return allocations
    
    async def select_best_variant(self) -> Optional[str]:
        """Select best performing variant based on statistical significance"""
        
        best_variant = None
        best_score = float('-inf')
        
        for name, variant in self.variants.items():
            if variant.trades_count < self.minimum_trades:
                continue
            
            # Calculate composite score
            score = (
                variant.sharpe_ratio * 0.4 +
                (variant.winning_trades / variant.trades_count) * 0.3 +
                variant.total_pnl_percent * 0.2 +
                (1 / max(abs(variant.max_drawdown), 0.01)) * 0.1
            )
            
            # Bonus for statistical significance
            if variant.is_significant and variant.confidence_level > self.confidence_threshold:
                score *= 1.2
            
            if score > best_score:
                best_score = score
                best_variant = name
        
        if best_variant:
            logger.info(f"Best variant: {self.variants[best_variant].name} "
                       f"(Score: {best_score:.3f})")
        
        return best_variant
    
    async def generate_ab_test_report(self) -> Dict:
        """Generate comprehensive A/B test report"""
        
        report = {
            'test_duration': (datetime.now() - self.test_start_time).days,
            'total_trades': sum(v.trades_count for v in self.variants.values()),
            'variants': {}
        }
        
        for name, variant in self.variants.items():
            variant_report = {
                'name': variant.name,
                'config': variant.config,
                'trades_count': variant.trades_count,
                'win_rate': variant.winning_trades / variant.trades_count if variant.trades_count > 0 else 0,
                'total_pnl': variant.total_pnl,
                'total_pnl_percent': variant.total_pnl_percent,
                'sharpe_ratio': variant.sharpe_ratio,
                'max_drawdown': variant.max_drawdown,
                'allocation_percent': variant.allocation_percent,
                'is_significant': variant.is_significant,
                'confidence_level': variant.confidence_level,
                'p_value': variant.p_value
            }
            
            report['variants'][name] = variant_report
        
        # Run statistical tests
        statistical_results = await self.run_statistical_tests()
        report['statistical_tests'] = statistical_results
        
        # Best variant
        best = await self.select_best_variant()
        report['recommended_variant'] = best
        
        # Allocations
        allocations = await self.adaptive_allocation()
        report['recommended_allocations'] = allocations
        
        return report
    
    async def should_stop_variant(self, variant_name: str) -> bool:
        """Determine if a variant should be stopped due to poor performance"""
        
        if variant_name not in self.variants:
            return False
        
        variant = self.variants[variant_name]
        
        # Stop conditions
        stop_conditions = [
            variant.max_drawdown < -0.30,  # 30% drawdown
            variant.trades_count > 50 and variant.winning_trades / variant.trades_count < 0.30,  # 30% win rate after 50 trades
            variant.is_significant and variant.p_value < 0.05 and variant.total_pnl_percent < -0.10,  # Significantly worse
            variant.trades_count > 100 and variant.sharpe_ratio < 0  # Negative Sharpe after 100 trades
        ]
        
        if any(stop_conditions):
            logger.warning(f"Stopping variant {variant.name} due to poor performance")
            return True
        
        return False
    
    async def promote_to_production(self, variant_name: str) -> bool:
        """Promote winning variant to production"""
        
        if variant_name not in self.variants:
            return False
        
        variant = self.variants[variant_name]
        
        # Check promotion criteria
        criteria = [
            variant.trades_count >= self.minimum_trades * 2,  # Enough data
            variant.is_significant,  # Statistically significant
            variant.confidence_level >= self.confidence_threshold,
            variant.sharpe_ratio > 1.5,  # Good risk-adjusted returns
            variant.max_drawdown > -0.15,  # Acceptable drawdown
            variant.winning_trades / variant.trades_count > 0.45  # Decent win rate
        ]
        
        if all(criteria):
            logger.info(f"🎉 Promoting {variant.name} to production!")
            
            # Save winning configuration
            await self._save_winning_config(variant)
            
            return True
        
        return False
    
    async def _save_winning_config(self, variant: StrategyVariant):
        """Save winning configuration for production use"""
        
        config = {
            'variant_name': variant.name,
            'config': variant.config,
            'performance': {
                'trades_count': variant.trades_count,
                'win_rate': variant.winning_trades / variant.trades_count,
                'sharpe_ratio': variant.sharpe_ratio,
                'max_drawdown': variant.max_drawdown,
                'total_pnl_percent': variant.total_pnl_percent
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Save to file
        filename = f"winning_strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(f"strategies/{filename}", 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Winning configuration saved to {filename}")