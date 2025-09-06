"""
ML Evolution System - Gradual transition from general to symbol-specific models
Runs in parallel with existing ML system for safe, non-breaking implementation
"""
import numpy as np
import logging
import json
import os
import pickle
import redis
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import asyncio
import psycopg2
from psycopg2.extras import RealDictCursor

# Import existing systems
from ml_signal_scorer_immediate import get_immediate_scorer
from phantom_trade_tracker import get_phantom_tracker

logger = logging.getLogger(__name__)

@dataclass
class SymbolMLStats:
    """Track performance stats for each symbol"""
    symbol: str
    total_signals: int = 0
    executed_trades: int = 0
    phantom_trades: int = 0
    last_train_time: Optional[datetime] = None
    model_version: int = 0
    confidence_score: float = 0.0
    recent_performance: List[float] = None
    
    def __post_init__(self):
        if self.recent_performance is None:
            self.recent_performance = []

class MLEvolutionSystem:
    """
    Manages the evolution from general to symbol-specific ML models
    - Uses general model as baseline
    - Gradually introduces symbol-specific models
    - Automatically adjusts confidence based on performance
    - Falls back to general model if issues arise
    """
    
    # Configuration
    MIN_TRADES_FOR_SYMBOL_ML = 50  # Need at least 50 trades to start symbol ML
    RETRAIN_INTERVAL_HOURS = 4  # Check for retraining every 4 hours
    MAX_SYMBOL_CONFIDENCE = 0.8  # Never trust symbol model more than 80%
    PERFORMANCE_WINDOW = 20  # Last 20 trades for performance tracking
    
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.general_scorer = get_immediate_scorer()  # Existing system
        self.symbol_models = {}  # symbol -> trained model
        self.symbol_stats = {}  # symbol -> SymbolMLStats
        self.symbol_scalers = {}  # symbol -> fitted scaler
        
        # Redis connection for state persistence
        self.redis_client = None
        self._init_redis()
        
        # PostgreSQL connection
        self.db_conn = None
        self._init_postgres()
        
        # Load saved state
        if self.enabled:
            self._load_state()
            logger.info("ML Evolution System initialized (ENABLED)")
        else:
            logger.info("ML Evolution System initialized (SHADOW MODE)")
    
    def _init_redis(self):
        """Initialize Redis connection"""
        try:
            redis_url = os.getenv('REDIS_URL')
            if redis_url:
                self.redis_client = redis.from_url(redis_url, decode_responses=True)
                self.redis_client.ping()
                logger.info("ML Evolution connected to Redis")
        except Exception as e:
            logger.warning(f"ML Evolution Redis connection failed: {e}")
            self.redis_client = None
    
    def _init_postgres(self):
        """Initialize PostgreSQL connection"""
        try:
            db_url = os.getenv('DATABASE_URL')
            if db_url:
                self.db_conn = psycopg2.connect(db_url, cursor_factory=RealDictCursor)
                logger.info("ML Evolution connected to PostgreSQL")
        except Exception as e:
            logger.warning(f"ML Evolution PostgreSQL connection failed: {e}")
            self.db_conn = None
    
    def score_signal(self, symbol: str, signal: dict, features: dict) -> Tuple[float, str, dict]:
        """
        Score a signal using blend of general and symbol-specific models
        
        Returns:
            - score (0-100)
            - reasoning string
            - detailed breakdown dict
        """
        # Always get general score first (fallback)
        general_score, general_reasoning = self.general_scorer.score_signal(signal, features)
        
        # Initialize stats if needed
        if symbol not in self.symbol_stats:
            self.symbol_stats[symbol] = SymbolMLStats(symbol=symbol)
        
        # Check if we should use symbol-specific model
        stats = self.symbol_stats[symbol]
        total_data = stats.executed_trades + stats.phantom_trades
        
        breakdown = {
            'general_score': general_score,
            'symbol_score': None,
            'confidence': 0.0,
            'final_score': general_score,
            'data_count': total_data
        }
        
        if total_data >= self.MIN_TRADES_FOR_SYMBOL_ML and symbol in self.symbol_models:
            try:
                # Get symbol-specific score
                symbol_score = self._get_symbol_score(symbol, features)
                confidence = self._calculate_confidence(symbol)
                
                # Weighted combination
                final_score = (general_score * (1 - confidence) + symbol_score * confidence)
                
                # Update breakdown
                breakdown['symbol_score'] = symbol_score
                breakdown['confidence'] = confidence
                breakdown['final_score'] = final_score
                
                # Build reasoning
                reasoning = (f"General: {general_score:.1f}, Symbol: {symbol_score:.1f}, "
                           f"Confidence: {confidence*100:.0f}%, Final: {final_score:.1f}")
                
                # Log in shadow mode
                if not self.enabled:
                    logger.info(f"[{symbol}] SHADOW MODE - {reasoning}")
                    return general_score, general_reasoning, breakdown  # Return general score
                
                return final_score, reasoning, breakdown
                
            except Exception as e:
                logger.error(f"Error in symbol scoring for {symbol}: {e}")
                # Fall back to general
                return general_score, general_reasoning, breakdown
        
        # Not enough data for symbol model
        return general_score, general_reasoning, breakdown
    
    def _get_symbol_score(self, symbol: str, features: dict) -> float:
        """Get symbol-specific ML score"""
        if symbol not in self.symbol_models:
            raise ValueError(f"No model for {symbol}")
        
        # Enhance features with historical data
        enhanced_features = self._enhance_features_with_history(symbol, features)
        
        # Prepare feature vector (same as general model)
        feature_vector = self._prepare_features(enhanced_features)
        X = np.array([feature_vector]).reshape(1, -1)
        
        # Scale if scaler exists
        if symbol in self.symbol_scalers:
            X_scaled = self.symbol_scalers[symbol].transform(X)
        else:
            X_scaled = X
        
        # Get predictions from symbol model
        model = self.symbol_models[symbol]
        
        # Handle different model types
        predictions = []
        if 'rf' in model:
            pred = model['rf'].predict_proba(X_scaled)[0][1]
            predictions.append(pred)
        if 'gb' in model:
            pred = model['gb'].predict_proba(X_scaled)[0][1]
            predictions.append(pred)
        if 'nn' in model:
            pred = model['nn'].predict_proba(X_scaled)[0][1]
            predictions.append(pred)
        
        # Average predictions
        if predictions:
            return np.mean(predictions) * 100
        else:
            raise ValueError("No predictions from symbol model")
    
    def _calculate_confidence(self, symbol: str) -> float:
        """Calculate confidence in symbol-specific model"""
        stats = self.symbol_stats.get(symbol)
        if not stats:
            return 0.0
        
        # Base confidence on amount of data
        total_data = stats.executed_trades + stats.phantom_trades
        data_confidence = min(total_data / 500, 0.8)  # Max 80% from data
        
        # Adjust based on recent performance
        if len(stats.recent_performance) >= 10:
            recent_wr = np.mean(stats.recent_performance)
            if recent_wr > 0.6:  # Winning 60%+
                performance_adjustment = 0.1
            elif recent_wr < 0.4:  # Winning less than 40%
                performance_adjustment = -0.2
            else:
                performance_adjustment = 0.0
        else:
            performance_adjustment = 0.0
        
        # Final confidence
        confidence = max(0.0, min(self.MAX_SYMBOL_CONFIDENCE, 
                                 data_confidence + performance_adjustment))
        
        stats.confidence_score = confidence
        return confidence
    
    def _enhance_features_with_history(self, symbol: str, features: dict) -> dict:
        """Add historical context from PostgreSQL"""
        enhanced = features.copy()
        
        if not self.db_conn:
            return enhanced
        
        try:
            with self.db_conn.cursor() as cur:
                # Get average volume for this hour
                hour = datetime.now().hour
                cur.execute("""
                    SELECT AVG(volume) as avg_vol
                    FROM candles
                    WHERE symbol = %s 
                    AND EXTRACT(hour FROM timestamp) = %s
                    AND timestamp > NOW() - INTERVAL '30 days'
                """, (symbol, hour))
                
                result = cur.fetchone()
                if result and result['avg_vol']:
                    enhanced['historical_hour_volume'] = float(result['avg_vol'])
                    enhanced['volume_vs_historical'] = features.get('volume', 0) / float(result['avg_vol'])
                
                # Get win rate at similar RSI levels
                current_rsi = features.get('rsi', 50)
                rsi_range = 5
                cur.execute("""
                    SELECT 
                        COUNT(CASE WHEN pnl_percent > 0 THEN 1 END)::float / 
                        NULLIF(COUNT(*), 0) as win_rate
                    FROM trades
                    WHERE symbol = %s
                    AND features->>'rsi' IS NOT NULL
                    AND ABS((features->>'rsi')::float - %s) < %s
                """, (symbol, current_rsi, rsi_range))
                
                result = cur.fetchone()
                if result and result['win_rate'] is not None:
                    enhanced['historical_rsi_win_rate'] = float(result['win_rate'])
                
        except Exception as e:
            logger.warning(f"Error enhancing features for {symbol}: {e}")
        
        return enhanced
    
    def _prepare_features(self, features: dict) -> list:
        """Convert features to vector (same format as general model)"""
        # This should match the feature order in ml_signal_scorer_immediate.py
        feature_order = [
            'trend_strength', 'higher_tf_alignment', 'ema_distance_ratio',
            'volume_ratio', 'volume_trend', 'breakout_volume',
            'support_resistance_strength', 'pullback_depth', 'confirmation_candle_strength',
            'atr_percentile', 'risk_reward_ratio', 'atr_stop_distance',
            'hour_of_day', 'day_of_week', 'candle_body_ratio', 'upper_wick_ratio',
            'lower_wick_ratio', 'candle_range_atr', 'volume_ma_ratio',
            'rsi', 'bb_position', 'volume_percentile',
            # New historical features
            'historical_hour_volume', 'volume_vs_historical', 'historical_rsi_win_rate'
        ]
        
        vector = []
        for feat in feature_order:
            if feat in features:
                val = features[feat]
                # Convert categorical to numeric
                if feat == 'volatility_regime':
                    val = {'low': 0, 'normal': 1, 'high': 2}.get(val, 1)
                elif feat == 'session':
                    val = {'asian': 0, 'european': 1, 'us': 2, 'off_hours': 3}.get(val, 3)
                vector.append(float(val) if val is not None else 0)
            else:
                vector.append(0)
        
        return vector
    
    def update_stats(self, symbol: str, is_phantom: bool = False):
        """Update symbol statistics"""
        if symbol not in self.symbol_stats:
            self.symbol_stats[symbol] = SymbolMLStats(symbol=symbol)
        
        stats = self.symbol_stats[symbol]
        stats.total_signals += 1
        
        if is_phantom:
            stats.phantom_trades += 1
        else:
            stats.executed_trades += 1
    
    def record_outcome(self, symbol: str, won: bool, pnl_percent: float):
        """Record trade outcome for confidence tracking"""
        if symbol not in self.symbol_stats:
            self.symbol_stats[symbol] = SymbolMLStats(symbol=symbol)
        
        stats = self.symbol_stats[symbol]
        stats.recent_performance.append(1.0 if won else 0.0)
        
        # Keep only recent trades
        if len(stats.recent_performance) > self.PERFORMANCE_WINDOW:
            stats.recent_performance.pop(0)
        
        # Log performance
        if len(stats.recent_performance) >= 10:
            recent_wr = np.mean(stats.recent_performance) * 100
            logger.info(f"[{symbol}] Recent performance: {recent_wr:.1f}% win rate")
        
        # Automatic safety check
        self._check_symbol_model_performance(symbol)
    
    def _check_symbol_model_performance(self, symbol: str):
        """
        Automatic safety monitoring - adjusts confidence if model underperforms
        This ensures the system self-corrects without manual intervention
        """
        stats = self.symbol_stats.get(symbol)
        if not stats or len(stats.recent_performance) < 10:
            return
        
        # Only check if we have a symbol model
        if symbol not in self.symbol_models:
            return
        
        recent_wr = np.mean(stats.recent_performance)
        
        # Critical underperformance check (< 30% win rate)
        if recent_wr < 0.30:
            logger.warning(f"[{symbol}] CRITICAL: Win rate {recent_wr*100:.1f}% - reducing confidence")
            stats.confidence_score = max(0.1, stats.confidence_score * 0.5)
            
            # If still underperforming after reduction, disable symbol model
            if stats.confidence_score <= 0.2:
                logger.error(f"[{symbol}] Disabling symbol model due to poor performance")
                stats.confidence_score = 0.0
        
        # Good performance check (> 60% win rate)
        elif recent_wr > 0.60 and stats.confidence_score < 0.6:
            logger.info(f"[{symbol}] Good performance {recent_wr*100:.1f}% - increasing confidence")
            stats.confidence_score = min(0.8, stats.confidence_score * 1.2)
        
        # Log if confidence was adjusted
        if hasattr(stats, '_last_confidence') and stats._last_confidence != stats.confidence_score:
            logger.info(f"[{symbol}] Confidence adjusted: {stats._last_confidence:.2f} → {stats.confidence_score:.2f}")
        stats._last_confidence = stats.confidence_score
    
    def _save_state(self):
        """Save current state to Redis"""
        if not self.redis_client:
            return
        
        try:
            # Save symbol stats
            stats_dict = {}
            for symbol, stats in self.symbol_stats.items():
                stats_dict[symbol] = {
                    'total_signals': stats.total_signals,
                    'executed_trades': stats.executed_trades,
                    'phantom_trades': stats.phantom_trades,
                    'last_train_time': stats.last_train_time.isoformat() if stats.last_train_time else None,
                    'model_version': stats.model_version,
                    'confidence_score': stats.confidence_score,
                    'recent_performance': stats.recent_performance
                }
            
            self.redis_client.set('evolution:stats', json.dumps(stats_dict))
            
            # Save models and scalers
            import base64
            for symbol in self.symbol_models:
                if symbol in self.symbol_models:
                    model_data = base64.b64encode(pickle.dumps(self.symbol_models[symbol])).decode()
                    self.redis_client.set(f'evolution:model:{symbol}', model_data)
                
                if symbol in self.symbol_scalers:
                    scaler_data = base64.b64encode(pickle.dumps(self.symbol_scalers[symbol])).decode()
                    self.redis_client.set(f'evolution:scaler:{symbol}', scaler_data)
            
        except Exception as e:
            logger.error(f"Error saving evolution state: {e}")
    
    def _load_state(self):
        """Load saved state from Redis"""
        if not self.redis_client:
            return
        
        try:
            # Load symbol stats
            stats_data = self.redis_client.get('evolution:stats')
            if stats_data:
                stats_dict = json.loads(stats_data)
                for symbol, data in stats_dict.items():
                    stats = SymbolMLStats(
                        symbol=symbol,
                        total_signals=data['total_signals'],
                        executed_trades=data['executed_trades'],
                        phantom_trades=data['phantom_trades'],
                        last_train_time=datetime.fromisoformat(data['last_train_time']) if data['last_train_time'] else None,
                        model_version=data['model_version'],
                        confidence_score=data['confidence_score'],
                        recent_performance=data['recent_performance']
                    )
                    self.symbol_stats[symbol] = stats
                
                logger.info(f"Loaded evolution stats for {len(self.symbol_stats)} symbols")
            
            # Load models and scalers
            import base64
            for symbol in self.symbol_stats:
                # Load model
                model_data = self.redis_client.get(f'evolution:model:{symbol}')
                if model_data:
                    self.symbol_models[symbol] = pickle.loads(base64.b64decode(model_data))
                
                # Load scaler
                scaler_data = self.redis_client.get(f'evolution:scaler:{symbol}')
                if scaler_data:
                    self.symbol_scalers[symbol] = pickle.loads(base64.b64decode(scaler_data))
            
            logger.info(f"Loaded {len(self.symbol_models)} symbol-specific models")
            
        except Exception as e:
            logger.error(f"Error loading evolution state: {e}")

# Global instance
_evolution_system = None

def get_evolution_system(enabled: bool = False) -> MLEvolutionSystem:
    """Get or create the global ML evolution system"""
    global _evolution_system
    if _evolution_system is None:
        _evolution_system = MLEvolutionSystem(enabled=enabled)
    return _evolution_system