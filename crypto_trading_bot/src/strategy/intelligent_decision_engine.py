"""
Intelligent Decision Engine with ML Integration
This module makes the bot truly intelligent by using ML predictions,
market regime detection, and adaptive strategies
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import structlog
from enum import Enum

from .ml_predictor import ml_predictor, ZoneFeatures
from .enhanced_ml_predictor import enhanced_ml_predictor, PredictionResult
from .advanced_supply_demand import EnhancedZone, MarketStructure, OrderFlowImbalance, VolumeProfile

logger = structlog.get_logger(__name__)


class MarketRegime(Enum):
    """Market regime types"""
    TRENDING_STRONG = "trending_strong"
    TRENDING_WEAK = "trending_weak"
    RANGING_TIGHT = "ranging_tight"
    RANGING_WIDE = "ranging_wide"
    VOLATILE = "volatile"
    BREAKOUT = "breakout"
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"


class TradingMode(Enum):
    """Adaptive trading modes"""
    AGGRESSIVE = "aggressive"      # High risk, high reward
    MODERATE = "moderate"          # Balanced approach
    CONSERVATIVE = "conservative"  # Low risk, steady gains
    SCALPING = "scalping"         # Quick in/out trades
    SWING = "swing"               # Hold for larger moves
    DEFENSIVE = "defensive"       # Capital preservation


@dataclass
class IntelligentSignal:
    """Enhanced signal with ML predictions and adaptive parameters"""
    symbol: str
    side: str  # Buy/Sell
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    position_size: float
    
    # ML predictions
    ml_success_probability: float
    ml_expected_profit_ratio: float
    ml_confidence: float
    ml_confidence_lower: float  # Lower bound of confidence interval
    ml_confidence_upper: float  # Upper bound of confidence interval
    ml_features: Dict[str, float]
    ml_warnings: List[str]  # Any ML warnings
    
    # Adaptive parameters
    market_regime: MarketRegime
    trading_mode: TradingMode
    risk_adjustment: float  # Risk multiplier based on confidence
    position_scaling: float  # Position size adjustment
    
    # Decision factors
    zone_score: float
    confluence_score: float
    sentiment_score: float
    momentum_score: float
    volume_score: float
    
    # Meta information
    signal_strength: float  # Overall signal strength 0-100
    expected_duration: int  # Expected holding time in minutes
    max_holding_time: int  # Maximum holding time
    trailing_stop_activation: float  # Price level to activate trailing
    
    # Reasoning
    entry_reasons: List[str]
    risk_factors: List[str]
    confidence_factors: List[str]


class IntelligentDecisionEngine:
    """
    Core intelligence module that:
    1. Uses ML predictions for every decision
    2. Adapts to market regimes
    3. Adjusts risk dynamically
    4. Learns from outcomes
    5. Integrates multiple data sources
    """
    
    def __init__(self):
        self.current_regime = {}  # Per symbol
        self.trading_mode = TradingMode.MODERATE
        self.performance_tracker = {}
        self.regime_detector = MarketRegimeDetector()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.risk_manager = AdaptiveRiskManager()
        
    def make_intelligent_decision(
        self,
        symbol: str,
        zone: Any,  # Can be EnhancedZone or dict
        market_data: Dict,
        account_balance: float,
        existing_positions: List[str]
    ) -> Optional[IntelligentSignal]:
        """
        Make an intelligent trading decision using all available data
        """
        try:
            # Ensure zone is an EnhancedZone object
            if not isinstance(zone, EnhancedZone):
                # If zone is a dict or other type, create EnhancedZone
                if isinstance(zone, dict):
                    zone = self._create_zone_from_dict(symbol, zone)
                elif isinstance(zone, str):
                    logger.error(f"Zone is a string for {symbol}: {zone}")
                    return None
                else:
                    logger.error(f"Unknown zone type for {symbol}: {type(zone)}")
                    return None
            # 1. Detect market regime
            regime = self.regime_detector.detect_regime(market_data)
            self.current_regime[symbol] = regime
            
            # 2. Get enhanced ML predictions with confidence intervals
            # Try enhanced predictor first, fallback to original if needed
            try:
                ml_prediction = enhanced_ml_predictor.predict_with_confidence(
                    zone, market_data
                )
                ml_success_prob = ml_prediction.success_probability
                ml_profit_ratio = ml_prediction.expected_profit
                ml_confidence = ml_prediction.prediction_confidence
                ml_confidence_lower = ml_prediction.confidence_lower
                ml_confidence_upper = ml_prediction.confidence_upper
                ml_warnings = ml_prediction.warnings
                
                # Log any ML warnings
                if ml_warnings:
                    logger.info(f"ML warnings for {symbol}: {', '.join(ml_warnings)}")
                    
            except Exception as e:
                logger.warning(f"Enhanced ML predictor failed, using original: {e}")
                # Fallback to original predictor
                ml_success_prob, ml_profit_ratio = ml_predictor.predict_zone_success(
                    zone, market_data
                )
                ml_confidence = self._calculate_ml_confidence(
                    ml_success_prob, 
                    ml_profit_ratio,
                    market_data
                )
                ml_confidence_lower = max(0, ml_success_prob - 0.15)
                ml_confidence_upper = min(1, ml_success_prob + 0.15)
                ml_warnings = []
            
            # 4. Determine trading mode based on market conditions
            trading_mode = self._determine_trading_mode(
                regime, 
                ml_confidence,
                len(existing_positions)
            )
            
            # 5. Check if signal meets minimum requirements
            if not self._should_take_trade(
                ml_success_prob,
                ml_confidence,
                zone.composite_score,
                trading_mode
            ):
                logger.debug(f"Signal rejected for {symbol}: ML prob={ml_success_prob:.2f}, confidence={ml_confidence:.2f}")
                return None
            
            # 6. Get sentiment analysis
            sentiment_score = self.sentiment_analyzer.analyze(symbol, market_data)
            
            # 7. Calculate adaptive risk parameters
            risk_params = self.risk_manager.calculate_adaptive_parameters(
                ml_confidence=ml_confidence,
                ml_success_prob=ml_success_prob,
                market_regime=regime,
                trading_mode=trading_mode,
                account_balance=account_balance,
                existing_positions=existing_positions
            )
            
            # 8. Calculate entry and exit prices with ML optimization
            entry_price, stop_loss, tp1, tp2 = self._calculate_ml_optimized_levels(
                zone=zone,
                market_data=market_data,
                ml_profit_ratio=ml_profit_ratio,
                risk_params=risk_params
            )
            
            # 9. Calculate intelligent position size
            position_size = self._calculate_intelligent_position_size(
                account_balance=account_balance,
                risk_amount=risk_params['risk_amount'],
                ml_confidence=ml_confidence,
                ml_success_prob=ml_success_prob,
                stop_distance=abs(entry_price - stop_loss),
                symbol=symbol,
                entry_price=entry_price
            )
            
            # 10. Extract ML features for tracking
            ml_features = self._extract_ml_features(zone, market_data)
            
            # 11. Calculate additional scores
            momentum_score = self._calculate_momentum_score(market_data)
            volume_score = self._calculate_volume_score(market_data)
            confluence_score = zone.confluence_score if hasattr(zone, 'confluence_score') else 60
            
            # 12. Determine expected holding time based on timeframe and regime
            expected_duration, max_holding = self._estimate_holding_time(
                regime, trading_mode, market_data
            )
            
            # 13. Calculate trailing stop activation
            trailing_activation = self._calculate_trailing_stop_activation(
                entry_price, tp1, ml_confidence
            )
            
            # 14. Build reasoning
            entry_reasons, risk_factors, confidence_factors = self._build_reasoning(
                zone, ml_success_prob, ml_confidence, regime, sentiment_score
            )
            
            # 15. Calculate overall signal strength
            signal_strength = self._calculate_signal_strength(
                ml_success_prob=ml_success_prob,
                ml_confidence=ml_confidence,
                zone_score=zone.composite_score,
                confluence_score=confluence_score,
                sentiment_score=sentiment_score,
                momentum_score=momentum_score,
                volume_score=volume_score
            )
            
            # Create intelligent signal with enhanced ML data
            signal = IntelligentSignal(
                symbol=symbol,
                side='Buy' if zone.zone_type == 'demand' else 'Sell',
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit_1=tp1,
                take_profit_2=tp2,
                position_size=position_size,
                ml_success_probability=ml_success_prob,
                ml_expected_profit_ratio=ml_profit_ratio,
                ml_confidence=ml_confidence,
                ml_confidence_lower=ml_confidence_lower,
                ml_confidence_upper=ml_confidence_upper,
                ml_features=ml_features,
                ml_warnings=ml_warnings,
                market_regime=regime,
                trading_mode=trading_mode,
                risk_adjustment=risk_params['risk_adjustment'],
                position_scaling=risk_params['position_scaling'],
                zone_score=zone.composite_score,
                confluence_score=confluence_score,
                sentiment_score=sentiment_score,
                momentum_score=momentum_score,
                volume_score=volume_score,
                signal_strength=signal_strength,
                expected_duration=expected_duration,
                max_holding_time=max_holding,
                trailing_stop_activation=trailing_activation,
                entry_reasons=entry_reasons,
                risk_factors=risk_factors,
                confidence_factors=confidence_factors
            )
            
            logger.info(
                f"🤖 Intelligent signal generated for {symbol}: "
                f"ML Prob={ml_success_prob:.2%}, ML Confidence={ml_confidence:.2%}, "
                f"Signal Strength={signal_strength:.1f}/100, Mode={trading_mode.value}"
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error in intelligent decision for {symbol}: {e}")
            return None
    
    def _calculate_ml_confidence(
        self, 
        success_prob: float, 
        profit_ratio: float,
        market_data: Dict
    ) -> float:
        """Calculate confidence in ML prediction"""
        
        # Base confidence from probability
        confidence = success_prob
        
        # Adjust based on expected profit
        if profit_ratio > 3:
            confidence *= 1.2
        elif profit_ratio > 2:
            confidence *= 1.1
        elif profit_ratio < 1.5:
            confidence *= 0.9
        
        # Adjust based on data quality
        data_quality = market_data.get('data_quality', 0.8)
        confidence *= data_quality
        
        # Adjust based on model training samples
        if len(ml_predictor.training_data) < 500:
            confidence *= 0.8
        elif len(ml_predictor.training_data) > 1000:
            confidence *= 1.1
        
        return min(1.0, max(0.0, confidence))
    
    def _determine_trading_mode(
        self,
        regime: MarketRegime,
        ml_confidence: float,
        num_positions: int
    ) -> TradingMode:
        """Determine appropriate trading mode"""
        
        # High confidence and trending market = aggressive
        if ml_confidence > 0.8 and regime in [MarketRegime.TRENDING_STRONG, MarketRegime.BREAKOUT]:
            return TradingMode.AGGRESSIVE
        
        # Low confidence or volatile market = conservative
        if ml_confidence < 0.6 or regime == MarketRegime.VOLATILE:
            return TradingMode.CONSERVATIVE
        
        # Many positions = defensive
        if num_positions > 10:
            return TradingMode.DEFENSIVE
        
        # Ranging market = scalping
        if regime in [MarketRegime.RANGING_TIGHT, MarketRegime.RANGING_WIDE]:
            return TradingMode.SCALPING
        
        # Default to moderate
        return TradingMode.MODERATE
    
    def _should_take_trade(
        self,
        ml_prob: float,
        ml_confidence: float,
        zone_score: float,
        mode: TradingMode
    ) -> bool:
        """Determine if trade should be taken based on mode and scores"""
        
        thresholds = {
            TradingMode.AGGRESSIVE: (0.55, 0.5, 40),
            TradingMode.MODERATE: (0.60, 0.6, 50),
            TradingMode.CONSERVATIVE: (0.70, 0.7, 60),
            TradingMode.SCALPING: (0.65, 0.6, 45),
            TradingMode.SWING: (0.65, 0.65, 55),
            TradingMode.DEFENSIVE: (0.75, 0.75, 65)
        }
        
        min_prob, min_conf, min_zone = thresholds.get(mode, (0.6, 0.6, 50))
        
        return (
            ml_prob >= min_prob and
            ml_confidence >= min_conf and
            zone_score >= min_zone
        )
    
    def _calculate_ml_optimized_levels(
        self,
        zone: EnhancedZone,
        market_data: Dict,
        ml_profit_ratio: float,
        risk_params: Dict
    ) -> Tuple[float, float, float, float]:
        """Calculate entry and exit levels optimized by ML predictions"""
        
        current_price = market_data['current_price']
        atr = market_data.get('atr', (zone.upper_bound - zone.lower_bound) * 0.5)
        
        if zone.zone_type == 'demand':
            # Long entry
            entry_price = min(current_price, zone.upper_bound)
            
            # Dynamic stop based on ML confidence
            stop_buffer = atr * (0.5 if risk_params['risk_adjustment'] > 1 else 0.3)
            stop_loss = zone.lower_bound - stop_buffer
            
            # ML-optimized targets
            risk = entry_price - stop_loss
            tp1 = entry_price + (risk * max(2.0, ml_profit_ratio * 0.7))
            tp2 = entry_price + (risk * max(3.0, ml_profit_ratio))
            
        else:  # supply zone
            # Short entry
            entry_price = max(current_price, zone.lower_bound)
            
            # Dynamic stop based on ML confidence
            stop_buffer = atr * (0.5 if risk_params['risk_adjustment'] > 1 else 0.3)
            stop_loss = zone.upper_bound + stop_buffer
            
            # ML-optimized targets
            risk = stop_loss - entry_price
            tp1 = entry_price - (risk * max(2.0, ml_profit_ratio * 0.7))
            tp2 = entry_price - (risk * max(3.0, ml_profit_ratio))
        
        return entry_price, stop_loss, tp1, tp2
    
    def _calculate_intelligent_position_size(
        self,
        account_balance: float,
        risk_amount: float,
        ml_confidence: float,
        ml_success_prob: float,
        stop_distance: float,
        symbol: str,
        entry_price: float
    ) -> float:
        """Calculate position size using hybrid risk formula"""
        
        # Get the risk percent from the parent context
        # This will be between 0.75% and 2.0% based on ML confidence
        # For now, we'll use a moderate approach based on ML confidence
        
        # Determine risk based on ML confidence (bounded)
        if ml_confidence >= 0.85:
            risk_percent = 0.02  # 2% for very high confidence
        elif ml_confidence >= 0.75:
            # Scale between 1% and 2%
            risk_percent = 0.01 + (ml_confidence - 0.75) * 0.1
        elif ml_confidence >= 0.65:
            risk_percent = 0.01  # 1% for medium confidence
        elif ml_confidence >= 0.55:
            # Scale between 0.75% and 1%
            risk_percent = 0.0075 + (ml_confidence - 0.55) * 0.025
        else:
            risk_percent = 0.0075  # 0.75% for low confidence
        
        # Calculate risk amount
        risk_amount = account_balance * risk_percent
        
        # Calculate position size based on stop distance
        # Position size = Risk Amount / Stop Distance
        position_size = risk_amount / stop_distance if stop_distance > 0 else 0
        
        # Further limit based on available balance (use max 30% per position)
        max_position_value = account_balance * 0.3
        max_position_by_value = max_position_value / entry_price if entry_price > 0 else 0
        position_size = min(position_size, max_position_by_value)
        
        # Ensure minimum notional value (5 USDT minimum for Bybit)
        min_notional = 5.5  # 5 USDT + 10% buffer
        
        if position_size * entry_price < min_notional:
            # Adjust to meet minimum
            min_position_size = min_notional / entry_price
            logger.info(f"Adjusting position size for {symbol} from {position_size:.4f} to {min_position_size:.4f} to meet ${min_notional} minimum notional")
            position_size = min_position_size
        
        return position_size
    
    def _extract_ml_features(self, zone: EnhancedZone, market_data: Dict) -> Dict[str, float]:
        """Extract ML features for tracking"""
        
        features = ml_predictor.extract_features(zone, market_data)
        
        return {
            'zone_strength': features.zone_strength,
            'volume_ratio': features.volume_ratio,
            'rejection_speed': features.rejection_speed,
            'institutional_interest': features.institutional_interest,
            'confluence_count': features.confluence_count,
            'timeframes_visible': features.timeframes_visible,
            'volatility_ratio': features.volatility_ratio,
            'liquidity_score': features.liquidity_score
        }
    
    def _calculate_momentum_score(self, market_data: Dict) -> float:
        """Calculate momentum score from market data"""
        
        df = market_data.get('dataframe')
        if df is None or df.empty:
            return 50.0
        
        # RSI momentum
        rsi = market_data.get('rsi', 50)
        rsi_score = 100 - abs(rsi - 50) * 2  # Neutral RSI = high score
        
        # Price momentum (rate of change)
        if len(df) > 20:
            roc = (df['close'].iloc[-1] / df['close'].iloc[-20] - 1) * 100
            roc_score = min(100, abs(roc) * 10)
        else:
            roc_score = 50
        
        # Volume momentum
        if 'volume' in df.columns and len(df) > 5:
            vol_ratio = df['volume'].iloc[-5:].mean() / df['volume'].iloc[-20:].mean()
            vol_score = min(100, vol_ratio * 50)
        else:
            vol_score = 50
        
        return (rsi_score + roc_score + vol_score) / 3
    
    def _calculate_volume_score(self, market_data: Dict) -> float:
        """Calculate volume score"""
        
        df = market_data.get('dataframe')
        if df is None or df.empty or 'volume' not in df.columns:
            return 50.0
        
        # Recent volume vs average
        recent_vol = df['volume'].iloc[-5:].mean()
        avg_vol = df['volume'].mean()
        
        if avg_vol > 0:
            vol_ratio = recent_vol / avg_vol
            return min(100, vol_ratio * 50)
        
        return 50.0
    
    def _estimate_holding_time(
        self,
        regime: MarketRegime,
        mode: TradingMode,
        market_data: Dict
    ) -> Tuple[int, int]:
        """Estimate expected and maximum holding time in minutes"""
        
        base_time = {
            TradingMode.SCALPING: (15, 60),
            TradingMode.AGGRESSIVE: (30, 120),
            TradingMode.MODERATE: (60, 240),
            TradingMode.CONSERVATIVE: (120, 480),
            TradingMode.SWING: (240, 1440),
            TradingMode.DEFENSIVE: (30, 120)
        }
        
        expected, maximum = base_time.get(mode, (60, 240))
        
        # Adjust for market regime
        if regime == MarketRegime.VOLATILE:
            expected *= 0.5
            maximum *= 0.5
        elif regime == MarketRegime.TRENDING_STRONG:
            expected *= 1.5
            maximum *= 2
        
        return int(expected), int(maximum)
    
    def _calculate_trailing_stop_activation(
        self,
        entry: float,
        tp1: float,
        ml_confidence: float
    ) -> float:
        """Calculate price level to activate trailing stop"""
        
        # Activate trailing stop at 50-80% of TP1 based on confidence
        activation_percent = 0.5 + (ml_confidence * 0.3)
        distance_to_tp1 = abs(tp1 - entry)
        
        if tp1 > entry:  # Long position
            return entry + (distance_to_tp1 * activation_percent)
        else:  # Short position
            return entry - (distance_to_tp1 * activation_percent)
    
    def _build_reasoning(
        self,
        zone: EnhancedZone,
        ml_prob: float,
        ml_conf: float,
        regime: MarketRegime,
        sentiment: float
    ) -> Tuple[List[str], List[str], List[str]]:
        """Build reasoning for the trade"""
        
        entry_reasons = []
        risk_factors = []
        confidence_factors = []
        
        # Entry reasons
        if ml_prob > 0.7:
            entry_reasons.append(f"High ML success probability: {ml_prob:.1%}")
        if zone.is_fresh:
            entry_reasons.append("Fresh untested zone")
        if zone.institutional_interest > 70:
            entry_reasons.append(f"High institutional interest: {zone.institutional_interest:.0f}%")
        if regime == MarketRegime.TRENDING_STRONG:
            entry_reasons.append("Strong trending market")
        
        # Risk factors
        if ml_conf < 0.7:
            risk_factors.append(f"Moderate ML confidence: {ml_conf:.1%}")
        if zone.test_count > 0:
            risk_factors.append(f"Zone tested {zone.test_count} times")
        if regime == MarketRegime.VOLATILE:
            risk_factors.append("High market volatility")
        if sentiment < 40:
            risk_factors.append("Negative market sentiment")
        
        # Confidence factors
        if len(zone.confluence_factors) > 3:
            confidence_factors.append(f"{len(zone.confluence_factors)} confluence factors")
        if zone.rejection_strength > 2.5:
            confidence_factors.append(f"Strong rejection: {zone.rejection_strength:.1f}")
        if ml_predictor.model_trained:
            confidence_factors.append(f"ML model trained on {len(ml_predictor.training_data)} samples")
        
        return entry_reasons, risk_factors, confidence_factors
    
    def _calculate_signal_strength(self, **scores) -> float:
        """Calculate overall signal strength"""
        
        weights = {
            'ml_success_prob': 0.25,
            'ml_confidence': 0.20,
            'zone_score': 0.15,
            'confluence_score': 0.15,
            'sentiment_score': 0.10,
            'momentum_score': 0.10,
            'volume_score': 0.05
        }
        
        total = 0
        for key, weight in weights.items():
            score = scores.get(key, 50)
            # Normalize to 0-100 if needed
            if key in ['ml_success_prob', 'ml_confidence']:
                score *= 100
            total += score * weight
        
        return min(100, max(0, total))
    
    def _create_zone_from_dict(self, symbol: str, zone_data: Dict) -> EnhancedZone:
        """Create EnhancedZone from dictionary data"""
        try:
            # Create a basic zone with available data
            zone = EnhancedZone(
                zone_type=zone_data.get('zone_type', zone_data.get('type', 'demand')),
                upper_bound=float(zone_data.get('upper_bound', zone_data.get('upper', 0))),
                lower_bound=float(zone_data.get('lower_bound', zone_data.get('lower', 0))),
                midpoint=float(zone_data.get('midpoint', 0)),
                strength_score=float(zone_data.get('strength_score', zone_data.get('score', 60))),
                formation_time=zone_data.get('formation_time', datetime.now()),
                timeframe=zone_data.get('timeframe', '15'),
                symbol=symbol
            )
            
            # Set additional attributes if available
            zone.composite_score = float(zone_data.get('composite_score', zone_data.get('score', 60)))
            zone.test_count = int(zone_data.get('test_count', zone_data.get('touches', 0)))
            zone.rejection_strength = float(zone_data.get('rejection_strength', zone_data.get('departure_strength', 2.0)))
            zone.institutional_interest = float(zone_data.get('institutional_interest', 50))
            zone.liquidity_pool = float(zone_data.get('liquidity_pool', 1000000))
            zone.is_fresh = zone_data.get('is_fresh', zone.test_count == 0)
            zone.zone_age_hours = float(zone_data.get('zone_age_hours', zone_data.get('age_hours', 0)))
            
            # Set volume profile if not present
            if not hasattr(zone, 'volume_profile') or zone.volume_profile is None:
                zone.volume_profile = VolumeProfile(
                    price_levels=[],
                    volumes=[],
                    poc=(zone.upper_bound + zone.lower_bound) / 2,
                    vah=zone.upper_bound,
                    val=zone.lower_bound,
                    total_volume=1000000,
                    buying_pressure=0.5,
                    selling_pressure=0.5
                )
            
            # Set confluence factors
            zone.confluence_factors = zone_data.get('confluence_factors', [])
            zone.timeframes_visible = zone_data.get('timeframes_visible', ['15'])
            
            return zone
            
        except Exception as e:
            logger.error(f"Error creating zone from dict for {symbol}: {e}")
            # Return a minimal zone as fallback
            return EnhancedZone(
                zone_type='demand',
                upper_bound=100,
                lower_bound=99,
                midpoint=99.5,
                strength_score=50,
                formation_time=datetime.now(),
                timeframe='15',
                symbol=symbol
            )
    
    def update_from_outcome(self, signal: IntelligentSignal, outcome: Dict):
        """Learn from trade outcomes to improve future decisions"""
        
        # Track performance by regime
        regime = signal.market_regime
        if regime not in self.performance_tracker:
            self.performance_tracker[regime] = {
                'trades': 0,
                'wins': 0,
                'total_profit': 0,
                'ml_accuracy': []
            }
        
        tracker = self.performance_tracker[regime]
        tracker['trades'] += 1
        
        if outcome['profit'] > 0:
            tracker['wins'] += 1
        tracker['total_profit'] += outcome['profit']
        
        # Track ML prediction accuracy
        predicted_success = signal.ml_success_probability > 0.5
        actual_success = outcome['profit'] > 0
        tracker['ml_accuracy'].append(predicted_success == actual_success)
        
        # Update both ML predictors with outcome
        try:
            # Get zone data for the signal (if available)
            zone = getattr(signal, '_zone', None)
            if zone:
                # Update enhanced ML predictor
                enhanced_ml_predictor.add_training_sample(
                    zone=zone,
                    market_data={'regime': regime.value, 'symbol': signal.symbol},
                    outcome=actual_success,
                    profit_ratio=outcome.get('profit_ratio', 0)
                )
                
                # Update prediction outcome for performance tracking
                if hasattr(signal, 'prediction_id'):
                    enhanced_ml_predictor.update_prediction_outcome(
                        signal.prediction_id,
                        actual_success,
                        outcome.get('profit_ratio', 0)
                    )
        except Exception as e:
            logger.warning(f"Failed to update enhanced ML: {e}")
        
        # Also update original ML predictor for compatibility
        ml_predictor.training_data.append({
            'features': signal.ml_features,
            'success': actual_success,
            'profit_ratio': outcome.get('profit_ratio', 0),
            'regime': regime.value,
            'mode': signal.trading_mode.value
        })
        
        # Retrain if enough new data
        if len(ml_predictor.training_data) % 50 == 0:
            logger.info(f"Retraining ML models with {len(ml_predictor.training_data)} samples")
            ml_predictor.train_models()


class MarketRegimeDetector:
    """Detect current market regime using multiple indicators"""
    
    def detect_regime(self, market_data: Dict) -> MarketRegime:
        """Detect the current market regime"""
        
        df = market_data.get('dataframe')
        if df is None or len(df) < 50:
            return MarketRegime.RANGING_WIDE
        
        # Calculate indicators
        atr = self._calculate_atr(df)
        trend_strength = self._calculate_trend_strength(df)
        volatility = self._calculate_volatility(df)
        volume_profile = self._analyze_volume_profile(df)
        
        # Determine regime
        if volatility > 2.0:
            return MarketRegime.VOLATILE
        
        if trend_strength > 0.7:
            return MarketRegime.TRENDING_STRONG
        elif trend_strength > 0.4:
            return MarketRegime.TRENDING_WEAK
        
        if self._is_breakout(df, atr):
            return MarketRegime.BREAKOUT
        
        if volume_profile == 'accumulation':
            return MarketRegime.ACCUMULATION
        elif volume_profile == 'distribution':
            return MarketRegime.DISTRIBUTION
        
        if volatility < 0.5:
            return MarketRegime.RANGING_TIGHT
        else:
            return MarketRegime.RANGING_WIDE
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        
        tr = pd.concat([
            high - low,
            (high - close).abs(),
            (low - close).abs()
        ], axis=1).max(axis=1)
        
        return tr.rolling(window=period).mean().iloc[-1]
    
    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """Calculate trend strength using ADX"""
        period = 14
        
        # Calculate +DM and -DM
        high_diff = df['high'].diff()
        low_diff = -df['low'].diff()
        
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        
        # Calculate ATR
        atr = self._calculate_atr(df, period)
        
        # Calculate +DI and -DI
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        # Calculate ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean().iloc[-1]
        
        return min(1.0, adx / 100)
    
    def _calculate_volatility(self, df: pd.DataFrame) -> float:
        """Calculate normalized volatility"""
        returns = df['close'].pct_change()
        volatility = returns.std()
        
        # Normalize by average price
        avg_price = df['close'].mean()
        normalized_vol = volatility * np.sqrt(252) / avg_price * 100
        
        return normalized_vol
    
    def _analyze_volume_profile(self, df: pd.DataFrame) -> str:
        """Analyze volume profile for accumulation/distribution"""
        if 'volume' not in df.columns:
            return 'neutral'
        
        # Recent vs historical volume
        recent_vol = df['volume'].iloc[-10:].mean()
        hist_vol = df['volume'].iloc[-50:-10].mean()
        
        # Price trend
        price_trend = df['close'].iloc[-1] > df['close'].iloc[-20]
        
        if recent_vol > hist_vol * 1.5:
            if price_trend:
                return 'accumulation'
            else:
                return 'distribution'
        
        return 'neutral'
    
    def _is_breakout(self, df: pd.DataFrame, atr: float) -> bool:
        """Detect if market is in breakout phase"""
        
        # Check if price broke recent highs/lows
        recent_high = df['high'].iloc[-20:-1].max()
        recent_low = df['low'].iloc[-20:-1].min()
        current_price = df['close'].iloc[-1]
        
        breakout_up = current_price > recent_high + (atr * 0.5)
        breakout_down = current_price < recent_low - (atr * 0.5)
        
        return breakout_up or breakout_down


class SentimentAnalyzer:
    """Analyze market sentiment from various sources"""
    
    def analyze(self, symbol: str, market_data: Dict) -> float:
        """
        Analyze sentiment and return score 0-100
        Higher score = more bullish
        """
        
        scores = []
        
        # Price action sentiment
        price_sentiment = self._analyze_price_action(market_data)
        scores.append(price_sentiment)
        
        # Volume sentiment
        volume_sentiment = self._analyze_volume(market_data)
        scores.append(volume_sentiment)
        
        # Order flow sentiment
        flow_sentiment = self._analyze_order_flow(market_data)
        scores.append(flow_sentiment)
        
        # Momentum sentiment
        momentum_sentiment = self._analyze_momentum(market_data)
        scores.append(momentum_sentiment)
        
        return np.mean(scores)
    
    def _analyze_price_action(self, market_data: Dict) -> float:
        """Analyze sentiment from price action"""
        
        df = market_data.get('dataframe')
        if df is None or len(df) < 20:
            return 50.0
        
        # Count bullish vs bearish candles
        bullish = (df['close'] > df['open']).sum()
        bearish = (df['close'] <= df['open']).sum()
        
        if bullish + bearish > 0:
            return (bullish / (bullish + bearish)) * 100
        
        return 50.0
    
    def _analyze_volume(self, market_data: Dict) -> float:
        """Analyze sentiment from volume patterns"""
        
        df = market_data.get('dataframe')
        if df is None or 'volume' not in df.columns:
            return 50.0
        
        # Volume on up days vs down days
        up_volume = df[df['close'] > df['open']]['volume'].sum()
        down_volume = df[df['close'] <= df['open']]['volume'].sum()
        
        total_volume = up_volume + down_volume
        if total_volume > 0:
            return (up_volume / total_volume) * 100
        
        return 50.0
    
    def _analyze_order_flow(self, market_data: Dict) -> float:
        """Analyze order flow sentiment"""
        
        order_flow = market_data.get('order_flow')
        
        if order_flow == OrderFlowImbalance.STRONG_BUYING:
            return 90
        elif order_flow == OrderFlowImbalance.BUYING:
            return 70
        elif order_flow == OrderFlowImbalance.STRONG_SELLING:
            return 10
        elif order_flow == OrderFlowImbalance.SELLING:
            return 30
        else:
            return 50
    
    def _analyze_momentum(self, market_data: Dict) -> float:
        """Analyze momentum indicators"""
        
        rsi = market_data.get('rsi', 50)
        
        # Convert RSI to sentiment score
        if rsi > 70:
            return 80  # Overbought but bullish
        elif rsi > 50:
            return 50 + (rsi - 50)  # Bullish
        elif rsi > 30:
            return 30 + (rsi - 30) * 0.5  # Bearish
        else:
            return 20  # Oversold but bearish
        
        return 50


class AdaptiveRiskManager:
    """Manage risk adaptively based on conditions"""
    
    def calculate_adaptive_parameters(
        self,
        ml_confidence: float,
        ml_success_prob: float,
        market_regime: MarketRegime,
        trading_mode: TradingMode,
        account_balance: float,
        existing_positions: List[str]
    ) -> Dict[str, float]:
        """Calculate adaptive risk parameters"""
        
        # Moderate Hybrid Risk Management
        # Base risk with bounded ML adjustments (0.75% - 2.0%)
        MIN_RISK_PERCENT = 0.75
        BASE_RISK_PERCENT = 1.0
        MAX_RISK_PERCENT = 2.0
        
        # Start with base risk
        risk_percent = BASE_RISK_PERCENT
        
        # ML Confidence adjustment (bounded)
        if ml_confidence >= 0.85:
            # Very high confidence: Use max risk
            risk_percent = MAX_RISK_PERCENT
        elif ml_confidence >= 0.75:
            # High confidence: Scale between base and max
            # Linear interpolation: 0.75->1.0, 0.85->2.0
            risk_percent = BASE_RISK_PERCENT + (ml_confidence - 0.75) * 10 * (MAX_RISK_PERCENT - BASE_RISK_PERCENT)
        elif ml_confidence >= 0.65:
            # Medium confidence: Use base risk
            risk_percent = BASE_RISK_PERCENT
        elif ml_confidence >= 0.55:
            # Lower confidence: Scale between min and base
            # Linear interpolation: 0.55->0.75, 0.65->1.0
            risk_percent = MIN_RISK_PERCENT + (ml_confidence - 0.55) * 10 * (BASE_RISK_PERCENT - MIN_RISK_PERCENT) / 2.5
        else:
            # Low confidence: Use minimum risk
            risk_percent = MIN_RISK_PERCENT
        
        # Market regime adjustment (small, bounded)
        regime_multiplier = 1.0
        if market_regime in [MarketRegime.TRENDING_STRONG, MarketRegime.BREAKOUT]:
            regime_multiplier = 1.1  # Slight increase for strong trends
        elif market_regime in [MarketRegime.VOLATILE, MarketRegime.DISTRIBUTION]:
            regime_multiplier = 0.9  # Slight decrease for volatile markets
        
        # Apply regime adjustment with bounds
        risk_percent = risk_percent * regime_multiplier
        
        # Final safety bounds - ensure we never exceed limits
        final_risk_percent = max(MIN_RISK_PERCENT, min(MAX_RISK_PERCENT, risk_percent))
        
        # For logging
        risk_adjustment = risk_percent / BASE_RISK_PERCENT
        regime_adj = regime_multiplier
        portfolio_adjustment = 1.0
        
        # Calculate position scaling
        position_scaling = risk_adjustment * regime_adj
        
        # Log the risk decision
        logger.info(
            f"Hybrid Risk Decision: ML Confidence={ml_confidence:.1%}, "
            f"Market Regime={market_regime.value}, "
            f"Risk={final_risk_percent:.2f}% "
            f"(Range: {MIN_RISK_PERCENT}%-{MAX_RISK_PERCENT}%)"
        )
        
        return {
            'risk_percent': final_risk_percent,
            'risk_amount': account_balance * final_risk_percent / 100,
            'risk_adjustment': risk_adjustment,
            'position_scaling': position_scaling,
            'max_positions': self._calculate_max_positions(trading_mode, market_regime)
        }
    
    def _calculate_max_positions(
        self,
        mode: TradingMode,
        regime: MarketRegime
    ) -> int:
        """Calculate maximum allowed positions"""
        
        base_positions = {
            TradingMode.AGGRESSIVE: 15,
            TradingMode.MODERATE: 10,
            TradingMode.CONSERVATIVE: 5,
            TradingMode.SCALPING: 20,
            TradingMode.SWING: 8,
            TradingMode.DEFENSIVE: 3
        }
        
        max_pos = base_positions.get(mode, 10)
        
        # Adjust for regime
        if regime == MarketRegime.VOLATILE:
            max_pos = int(max_pos * 0.5)
        elif regime == MarketRegime.TRENDING_STRONG:
            max_pos = int(max_pos * 1.2)
        
        return max(1, min(20, max_pos))


# Export the main engine
decision_engine = IntelligentDecisionEngine()