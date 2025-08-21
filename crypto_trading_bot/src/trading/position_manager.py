"""
Enhanced Position Manager with ML-based sizing and leverage optimization
"""
import asyncio
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import structlog
import numpy as np
from dataclasses import dataclass

from ..config import settings
from ..api.bybit_client import BybitClient
from ..strategy.ml_predictor import ml_predictor
from ..utils.reliability import retry_with_backoff, data_validator

logger = structlog.get_logger(__name__)

@dataclass
class PositionParameters:
    """Dynamic position parameters"""
    symbol: str
    base_leverage: int
    ml_leverage: int
    final_leverage: int
    base_size: float
    ml_size_multiplier: float
    final_size: float
    margin_mode: str
    confidence_score: float
    risk_score: float
    market_conditions: Dict

class EnhancedPositionManager:
    """Manages positions with ML-based optimization"""
    
    def __init__(self, bybit_client: BybitClient):
        self.client = bybit_client
        self.positions = {}
        self.position_parameters = {}
        self.ml_learning_data = []
        
        # Global settings that can be applied to all symbols
        self.global_leverage = settings.default_leverage
        self.global_margin_mode = settings.default_margin_mode.value
        self.leverage_limits = {
            'low_risk': (1, 5),
            'medium_risk': (3, 10),
            'high_risk': (5, 20),
            'extreme': (10, 50)
        }
        
    @retry_with_backoff(max_attempts=3)
    async def set_leverage_all_symbols(self, leverage: int, symbols: List[str] = None) -> Dict[str, bool]:
        """
        Set leverage for all monitored symbols at once
        """
        if symbols is None:
            symbols = settings.default_symbols[:300]  # Use top 300
        
        results = {}
        tasks = []
        
        logger.info(f"Setting leverage {leverage}x for {len(symbols)} symbols")
        
        # Create tasks for parallel execution
        for symbol in symbols:
            task = self._set_symbol_leverage(symbol, leverage)
            tasks.append(task)
        
        # Execute in batches to avoid overwhelming the API
        batch_size = 20
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i+batch_size]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            
            # Process results
            for j, result in enumerate(batch_results):
                symbol = symbols[i+j]
                if isinstance(result, Exception):
                    logger.error(f"Failed to set leverage for {symbol}: {result}")
                    results[symbol] = False
                else:
                    results[symbol] = result
            
            # Small delay between batches
            await asyncio.sleep(0.5)
        
        # Update global setting
        self.global_leverage = leverage
        
        success_count = sum(1 for v in results.values() if v)
        logger.info(f"Leverage set successfully for {success_count}/{len(symbols)} symbols")
        
        return results
    
    async def _set_symbol_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for a single symbol"""
        try:
            result = await self.client.set_leverage(symbol, leverage)
            return result is not None
        except Exception as e:
            logger.error(f"Error setting leverage for {symbol}: {e}")
            return False
    
    @retry_with_backoff(max_attempts=3)
    async def set_margin_mode_all_symbols(self, mode: str, symbols: List[str] = None) -> Dict[str, bool]:
        """
        Set margin mode (cross/isolated) for all monitored symbols
        """
        if mode not in ['cross', 'isolated']:
            raise ValueError(f"Invalid margin mode: {mode}")
        
        if symbols is None:
            symbols = settings.default_symbols[:300]  # Use top 300
        
        results = {}
        tasks = []
        
        logger.info(f"Setting {mode} margin for {len(symbols)} symbols")
        
        # Create tasks for parallel execution
        for symbol in symbols:
            task = self._set_symbol_margin_mode(symbol, mode)
            tasks.append(task)
        
        # Execute in batches
        batch_size = 20
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i+batch_size]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            
            # Process results
            for j, result in enumerate(batch_results):
                symbol = symbols[i+j]
                if isinstance(result, Exception):
                    logger.error(f"Failed to set margin mode for {symbol}: {result}")
                    results[symbol] = False
                else:
                    results[symbol] = result
            
            # Small delay between batches
            await asyncio.sleep(0.5)
        
        # Update global setting
        self.global_margin_mode = mode
        
        success_count = sum(1 for v in results.values() if v)
        logger.info(f"Margin mode set successfully for {success_count}/{len(symbols)} symbols")
        
        return results
    
    async def _set_symbol_margin_mode(self, symbol: str, mode: str) -> bool:
        """Set margin mode for a single symbol"""
        try:
            result = await self.client.set_margin_mode(symbol, mode)
            return result is not None
        except Exception as e:
            logger.error(f"Error setting margin mode for {symbol}: {e}")
            return False
    
    def calculate_ml_optimized_parameters(
        self,
        symbol: str,
        signal: Dict,
        market_data: Dict,
        account_balance: float
    ) -> PositionParameters:
        """
        Calculate ML-optimized position parameters
        """
        
        # Get base parameters from settings
        base_leverage = self.global_leverage
        base_risk_percent = settings.default_risk_percent
        
        # Get ML predictions
        ml_confidence = signal.get('ml_confidence', 0.5)
        expected_profit = signal.get('expected_profit', 1.0)
        risk_score = self._calculate_risk_score(market_data)
        
        # Calculate ML-optimized leverage
        ml_leverage = self._calculate_optimal_leverage(
            ml_confidence,
            risk_score,
            market_data.get('volatility', 1.0),
            base_leverage
        )
        
        # Calculate ML-optimized position size
        ml_size_multiplier = self._calculate_size_multiplier(
            ml_confidence,
            expected_profit,
            risk_score
        )
        
        # Calculate base position size
        base_size = (account_balance * base_risk_percent / 100) / signal.get('stop_loss_percent', 1.0)
        
        # Apply ML adjustments
        final_leverage = self._apply_safety_limits(ml_leverage, risk_score)
        final_size = base_size * ml_size_multiplier
        
        # Determine optimal margin mode
        optimal_margin_mode = self._determine_margin_mode(
            risk_score,
            ml_confidence,
            market_data
        )
        
        parameters = PositionParameters(
            symbol=symbol,
            base_leverage=base_leverage,
            ml_leverage=ml_leverage,
            final_leverage=final_leverage,
            base_size=base_size,
            ml_size_multiplier=ml_size_multiplier,
            final_size=final_size,
            margin_mode=optimal_margin_mode,
            confidence_score=ml_confidence,
            risk_score=risk_score,
            market_conditions=market_data
        )
        
        # Store for learning
        self.position_parameters[symbol] = parameters
        
        logger.info(
            f"ML Position Parameters for {symbol}: "
            f"Leverage {base_leverage}x → {final_leverage}x, "
            f"Size multiplier: {ml_size_multiplier:.2f}, "
            f"Confidence: {ml_confidence:.2%}"
        )
        
        return parameters
    
    def _calculate_risk_score(self, market_data: Dict) -> float:
        """Calculate current market risk score (0-1, higher = riskier)"""
        
        risk_score = 0.5  # Base risk
        
        # Volatility component
        volatility = market_data.get('volatility', 1.0)
        if volatility > 2.0:
            risk_score += 0.2
        elif volatility > 1.5:
            risk_score += 0.1
        elif volatility < 0.5:
            risk_score -= 0.1
        
        # Market structure component
        market_structure = market_data.get('market_structure', 'ranging')
        if market_structure == 'transitioning':
            risk_score += 0.15
        elif market_structure == 'ranging':
            risk_score += 0.1
        elif market_structure in ['bullish', 'bearish']:
            risk_score -= 0.05
        
        # Order flow component
        order_flow = market_data.get('order_flow', 'neutral')
        if order_flow == 'neutral':
            risk_score += 0.1
        elif order_flow in ['strong_buying', 'strong_selling']:
            risk_score -= 0.05
        
        # Time-based risk (weekends, news events)
        current_hour = datetime.now().hour
        if current_hour < 6 or current_hour > 22:  # Off-peak hours
            risk_score += 0.05
        
        day_of_week = datetime.now().weekday()
        if day_of_week in [5, 6]:  # Weekend
            risk_score += 0.1
        
        return max(0, min(1, risk_score))
    
    def _calculate_optimal_leverage(
        self,
        confidence: float,
        risk_score: float,
        volatility: float,
        base_leverage: int
    ) -> int:
        """
        Calculate optimal leverage based on ML predictions and risk
        """
        
        # Determine risk category
        if risk_score < 0.3:
            risk_category = 'low_risk'
        elif risk_score < 0.5:
            risk_category = 'medium_risk'
        elif risk_score < 0.7:
            risk_category = 'high_risk'
        else:
            risk_category = 'extreme'
        
        min_lev, max_lev = self.leverage_limits[risk_category]
        
        # Adjust based on confidence
        if confidence > 0.8:
            leverage_multiplier = 1.5
        elif confidence > 0.7:
            leverage_multiplier = 1.2
        elif confidence > 0.6:
            leverage_multiplier = 1.0
        elif confidence > 0.5:
            leverage_multiplier = 0.8
        else:
            leverage_multiplier = 0.5
        
        # Calculate ML leverage
        ml_leverage = base_leverage * leverage_multiplier
        
        # Adjust for volatility
        if volatility > 2.0:
            ml_leverage *= 0.5
        elif volatility > 1.5:
            ml_leverage *= 0.75
        elif volatility < 0.5:
            ml_leverage *= 1.2
        
        # Apply limits
        ml_leverage = max(min_lev, min(max_lev, int(ml_leverage)))
        
        return ml_leverage
    
    def _calculate_size_multiplier(
        self,
        confidence: float,
        expected_profit: float,
        risk_score: float
    ) -> float:
        """
        Calculate position size multiplier based on ML predictions
        """
        
        # Base multiplier from confidence
        if confidence > 0.8:
            base_multiplier = 1.5
        elif confidence > 0.7:
            base_multiplier = 1.25
        elif confidence > 0.6:
            base_multiplier = 1.0
        elif confidence > 0.5:
            base_multiplier = 0.75
        else:
            base_multiplier = 0.5
        
        # Adjust for expected profit
        if expected_profit > 2.0:
            base_multiplier *= 1.2
        elif expected_profit > 1.5:
            base_multiplier *= 1.1
        elif expected_profit < 1.0:
            base_multiplier *= 0.8
        
        # Risk adjustment
        risk_multiplier = 1.0 - (risk_score * 0.5)
        
        final_multiplier = base_multiplier * risk_multiplier
        
        # Ensure reasonable limits
        return max(0.25, min(2.0, final_multiplier))
    
    def _apply_safety_limits(self, leverage: int, risk_score: float) -> int:
        """Apply safety limits to leverage based on risk"""
        
        # Maximum leverage based on risk
        if risk_score > 0.8:
            max_leverage = 5
        elif risk_score > 0.6:
            max_leverage = 10
        elif risk_score > 0.4:
            max_leverage = 20
        else:
            max_leverage = 50
        
        return min(leverage, max_leverage)
    
    def _determine_margin_mode(
        self,
        risk_score: float,
        confidence: float,
        market_data: Dict
    ) -> str:
        """
        Determine optimal margin mode based on conditions
        """
        
        # High confidence + low risk = isolated (limit risk to position)
        if confidence > 0.7 and risk_score < 0.4:
            return 'isolated'
        
        # Low confidence or high risk = cross (use full account as buffer)
        if confidence < 0.5 or risk_score > 0.6:
            return 'cross'
        
        # Volatile markets = cross (need more margin buffer)
        if market_data.get('volatility', 1.0) > 1.5:
            return 'cross'
        
        # Default to global setting
        return self.global_margin_mode
    
    async def record_position_outcome(
        self,
        symbol: str,
        outcome: Dict
    ):
        """
        Record position outcome for ML learning
        """
        
        if symbol not in self.position_parameters:
            return
        
        parameters = self.position_parameters[symbol]
        
        # Calculate performance metrics
        actual_profit = outcome.get('profit_percent', 0)
        was_successful = actual_profit > 0
        risk_reward_achieved = actual_profit / abs(outcome.get('risk_percent', 1))
        
        # Store learning data
        learning_record = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'ml_leverage': parameters.ml_leverage,
            'final_leverage': parameters.final_leverage,
            'size_multiplier': parameters.ml_size_multiplier,
            'confidence': parameters.confidence_score,
            'risk_score': parameters.risk_score,
            'outcome': was_successful,
            'profit': actual_profit,
            'risk_reward': risk_reward_achieved,
            'market_conditions': parameters.market_conditions
        }
        
        self.ml_learning_data.append(learning_record)
        
        # Update ML model with outcome
        ml_predictor.add_training_sample(
            zone=outcome.get('zone'),
            market_data=parameters.market_conditions,
            outcome=was_successful,
            profit_ratio=1 + (actual_profit / 100)
        )
        
        # Retrain if we have enough new data
        if len(self.ml_learning_data) % 20 == 0:
            await self._update_ml_models()
        
        logger.info(
            f"Recorded outcome for {symbol}: "
            f"Success={was_successful}, Profit={actual_profit:.2f}%, "
            f"ML Leverage={parameters.ml_leverage}x"
        )
    
    async def _update_ml_models(self):
        """Update ML models based on accumulated learning data"""
        
        if len(self.ml_learning_data) < 50:
            return
        
        try:
            # Analyze leverage performance
            leverage_performance = {}
            for record in self.ml_learning_data[-100:]:  # Last 100 trades
                lev_bucket = (record['final_leverage'] // 5) * 5  # Round to nearest 5
                if lev_bucket not in leverage_performance:
                    leverage_performance[lev_bucket] = []
                leverage_performance[lev_bucket].append(record['profit'])
            
            # Update leverage limits based on performance
            for leverage, profits in leverage_performance.items():
                avg_profit = np.mean(profits)
                win_rate = sum(1 for p in profits if p > 0) / len(profits)
                
                if win_rate < 0.4 or avg_profit < -1.0:
                    # Poor performance, reduce this leverage level
                    logger.info(f"Reducing usage of {leverage}x leverage (WR: {win_rate:.1%})")
                elif win_rate > 0.65 and avg_profit > 1.5:
                    # Good performance, can use more often
                    logger.info(f"Increasing usage of {leverage}x leverage (WR: {win_rate:.1%})")
            
            # Train ML predictor
            ml_predictor.train_models()
            
        except Exception as e:
            logger.error(f"Error updating ML models: {e}")

# Global position manager instance
position_manager = None

def initialize_position_manager(bybit_client: BybitClient):
    """Initialize the global position manager"""
    global position_manager
    position_manager = EnhancedPositionManager(bybit_client)
    return position_manager