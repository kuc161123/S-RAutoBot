"""
ML Signal Scorer for Hybrid Smart Money Strategy
Scores and filters signals using machine learning
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import structlog
from datetime import datetime, timedelta

from .enhanced_ml_predictor import enhanced_ml_predictor, EnhancedZoneFeatures
from .hybrid_smart_money_strategy import HybridSignal, SignalType, MarketRegime

logger = structlog.get_logger(__name__)

class MLSignalScorer:
    """
    Scores signals from the hybrid strategy using ML models
    Enhances confidence scores and filters low-probability setups
    """
    
    def __init__(self):
        self.ml_predictor = enhanced_ml_predictor
        self.signal_history: Dict[str, List[Dict]] = {}
        self.performance_stats: Dict[SignalType, Dict] = {
            SignalType.ORDER_BLOCK: {"count": 0, "wins": 0, "total_pnl": 0},
            SignalType.FAIR_VALUE_GAP: {"count": 0, "wins": 0, "total_pnl": 0},
            SignalType.MEAN_REVERSION: {"count": 0, "wins": 0, "total_pnl": 0},
            SignalType.VWAP_BREAKOUT: {"count": 0, "wins": 0, "total_pnl": 0},
            SignalType.LIQUIDITY_SWEEP: {"count": 0, "wins": 0, "total_pnl": 0},
        }
        
    def score_signals(self, signals: List[HybridSignal], 
                     market_data: Dict[str, pd.DataFrame]) -> List[HybridSignal]:
        """
        Score and enhance signals using ML
        Returns filtered and scored signals
        """
        scored_signals = []
        
        for signal in signals:
            try:
                # Extract features for ML scoring
                features = self._extract_ml_features(signal, market_data.get(signal.symbol))
                
                # Get ML prediction if model is trained
                if self.ml_predictor.is_model_trained():
                    ml_result = self.ml_predictor.predict(features)
                    
                    # Update signal with ML scores
                    signal.ml_score = ml_result.success_probability * 100
                    signal.ml_features = {
                        "success_prob": ml_result.success_probability,
                        "expected_profit": ml_result.expected_profit,
                        "confidence_lower": ml_result.confidence_lower,
                        "confidence_upper": ml_result.confidence_upper,
                        "prediction_confidence": ml_result.prediction_confidence
                    }
                    
                    # Combine original confidence with ML score
                    original_confidence = signal.confidence
                    ml_weight = 0.4  # 40% weight to ML, 60% to strategy logic
                    signal.confidence = (original_confidence * 0.6 + signal.ml_score * ml_weight)
                    
                    # Add ML insights to confluences
                    if signal.ml_score >= 70:
                        signal.confluences.append(f"ML High Confidence: {signal.ml_score:.0f}%")
                    elif signal.ml_score >= 50:
                        signal.confluences.append(f"ML Medium Confidence: {signal.ml_score:.0f}%")
                    
                    # Adjust position size based on ML confidence
                    if ml_result.prediction_confidence >= 0.8:
                        signal.position_size_multiplier *= 1.2
                    elif ml_result.prediction_confidence <= 0.4:
                        signal.position_size_multiplier *= 0.8
                    
                    # Log ML enhancement
                    logger.debug(f"ML enhanced {signal.symbol} {signal.signal_type.value} signal: "
                               f"Original: {original_confidence:.1f}%, ML: {signal.ml_score:.1f}%, "
                               f"Final: {signal.confidence:.1f}%")
                else:
                    # No ML model available, use strategy confidence as-is
                    signal.ml_score = signal.confidence
                    logger.debug(f"ML model not trained, using original confidence for {signal.symbol}")
                
                # Apply signal type specific enhancements
                signal = self._enhance_by_signal_type(signal)
                
                # Filter based on final confidence
                if signal.confidence >= 50:  # Minimum 50% confidence after ML scoring
                    scored_signals.append(signal)
                    
            except Exception as e:
                logger.error(f"Error scoring signal for {signal.symbol}: {e}")
                # Keep signal with original confidence if ML scoring fails
                scored_signals.append(signal)
        
        # Rank signals by confidence
        scored_signals.sort(key=lambda x: x.confidence, reverse=True)
        
        # Log summary
        if scored_signals:
            logger.info(f"✅ ML Scorer: {len(scored_signals)}/{len(signals)} signals passed filtering")
            for sig in scored_signals[:3]:  # Top 3
                logger.info(f"  Top signal: {sig.symbol} {sig.signal_type.value} "
                          f"{sig.direction} @ {sig.entry_price:.4f} "
                          f"(confidence: {sig.confidence:.1f}%)")
        
        return scored_signals
    
    def _extract_ml_features(self, signal: HybridSignal, df: Optional[pd.DataFrame]) -> EnhancedZoneFeatures:
        """Extract features for ML prediction"""
        # Default features if no market data
        if df is None or len(df) < 50:
            return self._get_default_features(signal)
        
        # Calculate technical features
        current_price = df['close'].iloc[-1]
        volume_ratio = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
        
        # Calculate momentum
        returns_5 = (df['close'].iloc[-1] / df['close'].iloc[-6] - 1) * 100
        returns_20 = (df['close'].iloc[-1] / df['close'].iloc[-21] - 1) * 100
        momentum_score = (returns_5 * 0.7 + returns_20 * 0.3)
        
        # Volume momentum
        vol_change = (df['volume'].iloc[-5:].mean() / df['volume'].iloc[-10:-5].mean() - 1) * 100
        
        # Volatility percentile
        current_volatility = df['close'].pct_change().rolling(20).std().iloc[-1]
        hist_volatility = df['close'].pct_change().rolling(20).std().iloc[-100:]
        vol_percentile = (hist_volatility < current_volatility).sum() / len(hist_volatility) * 100
        
        # Market efficiency (how directional vs choppy)
        high_low_range = df['high'].iloc[-20:].max() - df['low'].iloc[-20:].min()
        close_to_close_range = abs(df['close'].iloc[-20] - df['close'].iloc[-1])
        efficiency = (close_to_close_range / high_low_range * 100) if high_low_range > 0 else 50
        
        # Signal type specific features
        zone_strength = 70  # Default
        if signal.signal_type == SignalType.ORDER_BLOCK and signal.order_block:
            zone_strength = signal.order_block.strength
        elif signal.signal_type == SignalType.FAIR_VALUE_GAP:
            zone_strength = 75  # FVGs typically have good success
        elif signal.signal_type == SignalType.LIQUIDITY_SWEEP:
            zone_strength = 80  # Liquidity sweeps are strong signals
        elif signal.signal_type == SignalType.MEAN_REVERSION:
            zone_strength = 65  # Mean reversion moderate confidence
        elif signal.signal_type == SignalType.VWAP_BREAKOUT:
            zone_strength = 70  # VWAP breakouts good confidence
        
        # Create feature object
        features = EnhancedZoneFeatures(
            zone_strength=zone_strength,
            volume_ratio=volume_ratio,
            rejection_speed=abs(momentum_score),
            institutional_interest=min(100, volume_ratio * 30),
            confluence_count=len(signal.confluences),
            timeframes_visible=len(signal.timeframes_aligned),
            market_structure=1 if signal.direction == "buy" else -1,
            order_flow_imbalance=1 if volume_ratio > 1.5 else 0,
            zone_age_hours=0,  # Fresh signal
            test_count=0,  # New signal
            distance_from_poc=0,  # Would need volume profile
            in_value_area=True,  # Assume in value area
            trend_alignment=True if signal.risk_reward_ratio > 2 else False,
            volatility_ratio=current_volatility * 100,
            liquidity_score=volume_ratio * 50,
            time_of_day=datetime.now().hour,
            day_of_week=datetime.now().weekday(),
            momentum_score=momentum_score,
            volume_momentum=vol_change,
            volatility_percentile=vol_percentile,
            spread_ratio=0.5,  # Would need bid/ask data
            market_efficiency=efficiency,
            cross_market_correlation=0.5,  # Would need BTC correlation
            zone_quality_score=signal.confidence,
            risk_on_off_score=50  # Neutral risk sentiment
        )
        
        return features
    
    def _get_default_features(self, signal: HybridSignal) -> EnhancedZoneFeatures:
        """Get default features when market data is not available"""
        return EnhancedZoneFeatures(
            zone_strength=signal.confidence,
            volume_ratio=1.0,
            rejection_speed=50,
            institutional_interest=50,
            confluence_count=len(signal.confluences),
            timeframes_visible=len(signal.timeframes_aligned),
            market_structure=1 if signal.direction == "buy" else -1,
            order_flow_imbalance=0,
            zone_age_hours=0,
            test_count=0,
            distance_from_poc=0,
            in_value_area=True,
            trend_alignment=True,
            volatility_ratio=2.0,
            liquidity_score=50,
            time_of_day=datetime.now().hour,
            day_of_week=datetime.now().weekday(),
            momentum_score=0,
            volume_momentum=0,
            volatility_percentile=50,
            spread_ratio=0.5,
            market_efficiency=50,
            cross_market_correlation=0.5,
            zone_quality_score=signal.confidence,
            risk_on_off_score=50
        )
    
    def _enhance_by_signal_type(self, signal: HybridSignal) -> HybridSignal:
        """Apply signal type specific enhancements based on historical performance"""
        
        # Get win rate for this signal type
        stats = self.performance_stats.get(signal.signal_type)
        if stats and stats["count"] > 10:
            win_rate = (stats["wins"] / stats["count"]) * 100
            
            # Boost confidence if this signal type is performing well
            if win_rate > 70:
                signal.confidence = min(100, signal.confidence * 1.1)
                signal.confluences.append(f"{signal.signal_type.value} Win Rate: {win_rate:.0f}%")
            elif win_rate < 40:
                signal.confidence *= 0.9
                signal.confluences.append(f"⚠️ {signal.signal_type.value} Win Rate: {win_rate:.0f}%")
        
        # Signal type specific adjustments
        if signal.signal_type == SignalType.LIQUIDITY_SWEEP:
            # Liquidity sweeps are typically high probability
            signal.confidence = min(100, signal.confidence * 1.15)
            
        elif signal.signal_type == SignalType.ORDER_BLOCK:
            # Order blocks need volume confirmation
            if "High Volume" in str(signal.confluences):
                signal.confidence = min(100, signal.confidence * 1.1)
                
        elif signal.signal_type == SignalType.MEAN_REVERSION:
            # Mean reversion works better in ranging markets
            if any("RSI" in c for c in signal.confluences):
                signal.confidence = min(100, signal.confidence * 1.05)
                
        elif signal.signal_type == SignalType.VWAP_BREAKOUT:
            # VWAP breakouts need momentum confirmation
            if "MACD" in str(signal.confluences):
                signal.confidence = min(100, signal.confidence * 1.1)
        
        return signal
    
    def update_performance(self, signal_type: SignalType, won: bool, pnl: float):
        """Update performance statistics for a signal type"""
        if signal_type in self.performance_stats:
            stats = self.performance_stats[signal_type]
            stats["count"] += 1
            if won:
                stats["wins"] += 1
            stats["total_pnl"] += pnl
            
            # Log performance update
            win_rate = (stats["wins"] / stats["count"]) * 100
            avg_pnl = stats["total_pnl"] / stats["count"]
            logger.info(f"📊 {signal_type.value} Performance: "
                       f"Win Rate: {win_rate:.1f}%, Avg PnL: {avg_pnl:.2f}")
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary for all signal types"""
        summary = {}
        for signal_type, stats in self.performance_stats.items():
            if stats["count"] > 0:
                summary[signal_type.value] = {
                    "trades": stats["count"],
                    "wins": stats["wins"],
                    "win_rate": (stats["wins"] / stats["count"]) * 100,
                    "total_pnl": stats["total_pnl"],
                    "avg_pnl": stats["total_pnl"] / stats["count"]
                }
        return summary

# Create global instance
ml_signal_scorer = MLSignalScorer()