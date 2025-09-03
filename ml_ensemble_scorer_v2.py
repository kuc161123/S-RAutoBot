"""
Enhanced ML Ensemble Scorer V2 with New Pullback Features
Includes Fibonacci, 1H trend, and volatility regime features
Backward compatible drop-in replacement
"""
import numpy as np
import pandas as pd
import logging
import json
import os
import redis
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, asdict
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    
logger = logging.getLogger(__name__)

@dataclass
class EnhancedSignalFeatures:
    """Enhanced features including new pullback strategy improvements"""
    
    # Original features
    trend_strength: float
    higher_tf_alignment: float
    ema_distance_ratio: float
    volume_ratio: float
    volume_trend: float
    breakout_volume: float
    support_resistance_strength: float
    pullback_depth: float
    confirmation_candle_strength: float
    atr_percentile: float
    volatility_regime: str
    hour_of_day: int
    day_of_week: int
    session: str
    risk_reward_ratio: float
    atr_stop_distance: float
    
    # NEW Enhanced features
    fibonacci_retracement: float = 50.0  # Actual retracement %
    is_golden_zone: float = 0.0  # 1.0 if 38.2-61.8%
    is_shallow_pullback: float = 0.0  # 1.0 if <38.2%
    is_deep_pullback: float = 0.0  # 1.0 if >78.6%
    fib_50_distance: float = 0.0  # Distance from 50%
    
    higher_tf_trend_bullish: float = 0.0  # 1H trend indicators
    higher_tf_trend_bearish: float = 0.0
    higher_tf_trend_neutral: float = 1.0
    
    volatility_low: float = 0.0  # One-hot encoded volatility
    volatility_normal: float = 1.0
    volatility_high: float = 0.0
    
    adjusted_confirmation_candles: int = 2  # Dynamic params
    adjusted_sl_buffer: float = 0.5
    adjusted_rr_ratio: float = 2.0
    
    pullback_quality_score: float = 50.0  # Overall quality 0-100

class MarketRegime:
    """Market regime classification"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"

class EnsembleMLScorerV2:
    """
    Enhanced Ensemble ML Scorer with new pullback features
    """
    
    MIN_TRADES_TO_LEARN = 200
    RETRAIN_INTERVAL = 50
    
    def __init__(self, min_score: float = 70.0, enabled: bool = True):
        self.min_score = min_score
        self.enabled = enabled
        self.is_trained = False
        self.completed_trades_count = 0
        self.last_train_count = 0
        
        # Ensemble models
        self.models = {
            'rf': None,
            'gb': None,  # GradientBoosting instead of XGB if not available
            'nn': None,
        }
        
        # Separate long/short models
        self.long_models = {'rf': None, 'gb': None, 'nn': None}
        self.short_models = {'rf': None, 'gb': None, 'nn': None}
        
        # Scalers
        self.scalers = {
            'ensemble': StandardScaler(),
            'long': StandardScaler(),
            'short': StandardScaler()
        }
        
        # Feature importance tracking
        self.feature_importance = {}
        self.enhanced_feature_importance = {}  # Track new features separately
        
        # Performance tracking
        self.model_performance = {
            'rf': {'correct': 0, 'total': 0},
            'gb': {'correct': 0, 'total': 0},
            'nn': {'correct': 0, 'total': 0},
            'ensemble': {'correct': 0, 'total': 0}
        }
        
        # Redis storage
        self.redis_client = None
        if enabled:
            self._init_redis()
            self._load_from_redis()
    
    def _init_redis(self):
        """Initialize Redis connection"""
        try:
            redis_url = os.getenv('REDIS_URL')
            if redis_url:
                self.redis_client = redis.from_url(redis_url, decode_responses=True)
                self.redis_client.ping()
                logger.info("Connected to Redis for ML V2 storage")
            else:
                self.local_storage = {'signals': [], 'completed_trades': []}
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            self.local_storage = {'signals': [], 'completed_trades': []}
    
    def _load_from_redis(self):
        """Load models from Redis"""
        if not self.redis_client:
            return
        
        try:
            # Load ensemble models
            for model_name in ['rf', 'gb', 'nn']:
                model_data = self.redis_client.get(f'ml_v2_model_{model_name}')
                if model_data:
                    import base64
                    self.models[model_name] = pickle.loads(base64.b64decode(model_data))
            
            # Load scalers
            scaler_data = self.redis_client.get('ml_v2_scaler_ensemble')
            if scaler_data:
                import base64
                self.scalers['ensemble'] = pickle.loads(base64.b64decode(scaler_data))
            
            # Check if models loaded
            if any(m is not None for m in self.models.values()):
                self.is_trained = True
                logger.info("Loaded ML V2 ensemble models from Redis")
            
            # Load trade count
            count = self.redis_client.get('ml_v2_trades_count')
            self.completed_trades_count = int(count) if count else 0
            
        except Exception as e:
            logger.error(f"Error loading from Redis: {e}")
    
    def extract_enhanced_features(self, signal_meta: Dict, df: pd.DataFrame,
                                 df_1h: Optional[pd.DataFrame] = None) -> EnhancedSignalFeatures:
        """Extract all features including new enhancements"""
        
        # Basic calculations
        close_prices = df['close'].values[-20:]
        trend_strength = min(100, abs(np.polyfit(range(len(close_prices)), close_prices, 1)[0]) * 100)
        
        avg_volume = df['volume'].rolling(20).mean().iloc[-1]
        current_volume = df['volume'].iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Extract enhanced features from metadata
        # Fibonacci features
        fib_retracement = 50.0  # Default
        if 'fib_retracement' in signal_meta:
            try:
                fib_str = signal_meta['fib_retracement'].rstrip('%')
                fib_retracement = float(fib_str)
            except:
                fib_retracement = 50.0
        
        fib_decimal = fib_retracement / 100
        is_golden_zone = 1.0 if 38.2 <= fib_retracement <= 61.8 else 0.0
        is_shallow = 1.0 if fib_retracement < 38.2 else 0.0
        is_deep = 1.0 if fib_retracement > 78.6 else 0.0
        fib_50_distance = abs(fib_retracement - 50.0)
        
        # 1H trend features
        ht_bullish = 0.0
        ht_bearish = 0.0
        ht_neutral = 1.0
        
        if 'higher_tf_trend' in signal_meta:
            trend = signal_meta['higher_tf_trend']
            if trend == "BULLISH":
                ht_bullish, ht_neutral = 1.0, 0.0
            elif trend == "BEARISH":
                ht_bearish, ht_neutral = 1.0, 0.0
        
        # Volatility features
        vol_regime = signal_meta.get('volatility', 'NORMAL')
        vol_low = 1.0 if vol_regime == "LOW" else 0.0
        vol_normal = 1.0 if vol_regime == "NORMAL" else 0.0
        vol_high = 1.0 if vol_regime == "HIGH" else 0.0
        
        # Dynamic parameters
        adj_confirmation = signal_meta.get('confirmation_candles', 2)
        
        # Adjust parameters based on volatility
        if vol_regime == "LOW":
            adj_sl_buffer, adj_rr = 0.3, 2.5
        elif vol_regime == "HIGH":
            adj_sl_buffer, adj_rr = 0.7, 1.8
        else:
            adj_sl_buffer, adj_rr = 0.5, 2.0
        
        # Calculate pullback quality score
        quality_score = self._calculate_quality_score(
            fib_retracement, is_golden_zone, trend_strength,
            volume_ratio, vol_regime, ht_bullish if signal_meta.get('side') == 'long' else ht_bearish
        )
        
        # Time features
        now = datetime.now()
        hour = now.hour
        dow = now.weekday()
        
        if 0 <= hour < 8:
            session = "asian"
        elif 8 <= hour < 16:
            session = "european"
        elif 16 <= hour < 22:
            session = "us"
        else:
            session = "off_hours"
        
        # ATR calculations
        atr = self._calculate_atr(df)
        atr_values = [self._calculate_atr(df.iloc[i-14:i]) for i in range(100, min(len(df), 200))]
        atr_percentile = 50.0
        if atr_values:
            atr_percentile = (np.searchsorted(np.sort(atr_values), atr) * 100 / len(atr_values))
        
        return EnhancedSignalFeatures(
            # Original features
            trend_strength=trend_strength,
            higher_tf_alignment=ht_bullish * 100 if signal_meta.get('side') == 'long' else ht_bearish * 100,
            ema_distance_ratio=0.0,
            volume_ratio=volume_ratio,
            volume_trend=0.0,
            breakout_volume=volume_ratio,
            support_resistance_strength=75.0,
            pullback_depth=fib_retracement,
            confirmation_candle_strength=50.0,
            atr_percentile=atr_percentile,
            volatility_regime=vol_regime,
            hour_of_day=hour,
            day_of_week=dow,
            session=session,
            risk_reward_ratio=adj_rr,
            atr_stop_distance=adj_sl_buffer,
            
            # Enhanced features
            fibonacci_retracement=fib_retracement,
            is_golden_zone=is_golden_zone,
            is_shallow_pullback=is_shallow,
            is_deep_pullback=is_deep,
            fib_50_distance=fib_50_distance,
            higher_tf_trend_bullish=ht_bullish,
            higher_tf_trend_bearish=ht_bearish,
            higher_tf_trend_neutral=ht_neutral,
            volatility_low=vol_low,
            volatility_normal=vol_normal,
            volatility_high=vol_high,
            adjusted_confirmation_candles=adj_confirmation,
            adjusted_sl_buffer=adj_sl_buffer,
            adjusted_rr_ratio=adj_rr,
            pullback_quality_score=quality_score
        )
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR"""
        if len(df) < period:
            return 0.0
        
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        tr = np.maximum(high[1:] - low[1:],
                       np.maximum(np.abs(high[1:] - close[:-1]),
                                 np.abs(low[1:] - close[:-1])))
        return np.mean(tr[-period:]) if len(tr) >= period else 0.0
    
    def _calculate_quality_score(self, fib: float, is_golden: float,
                                trend: float, volume: float, volatility: str,
                                trend_alignment: float) -> float:
        """Calculate comprehensive quality score"""
        score = 0.0
        
        # Fibonacci (30 points)
        if is_golden:
            score += 30
        elif fib < 38.2:
            score += 5
        elif fib > 78.6:
            score += 3
        else:
            score += 15
        
        # Trend strength (20 points)
        score += min(20, trend * 0.2)
        
        # Volume (20 points)
        if volume > 1.5:
            score += 20
        elif volume > 1.2:
            score += 15
        elif volume > 1.0:
            score += 10
        else:
            score += 5
        
        # Trend alignment (20 points)
        score += trend_alignment * 0.2
        
        # Volatility (10 points)
        if volatility == "NORMAL":
            score += 10
        elif volatility == "LOW":
            score += 7
        else:
            score += 5
        
        return min(100, score)
    
    def score_signal(self, signal_meta: Dict, df: pd.DataFrame,
                    df_1h: Optional[pd.DataFrame] = None) -> Tuple[bool, float, str]:
        """Score signal with ensemble voting"""
        
        if not self.enabled:
            return True, 75.0, "ML scoring disabled"
        
        try:
            # Extract enhanced features
            features = self.extract_enhanced_features(signal_meta, df, df_1h)
            
            # If not trained, use quality score
            if not self.is_trained:
                score = features.pullback_quality_score
                should_take = score >= 60
                
                reason = f"Pre-ML Score {score:.0f} - "
                reason += f"Fib: {features.fibonacci_retracement:.1f}%, "
                reason += f"1H: {signal_meta.get('higher_tf_trend', 'N/A')}, "
                reason += f"Vol: {features.volatility_regime}"
                
                return should_take, score, reason
            
            # Use ensemble voting
            feature_dict = asdict(features)
            
            # Remove non-numeric
            feature_dict.pop('volatility_regime', None)
            feature_dict.pop('session', None)
            
            X = np.array(list(feature_dict.values())).reshape(1, -1)
            X_scaled = self.scalers['ensemble'].transform(X)
            
            # Get predictions from each model
            predictions = []
            scores = []
            
            for model_name, model in self.models.items():
                if model is not None:
                    prob = model.predict_proba(X_scaled)[0, 1]
                    predictions.append(prob > 0.5)
                    scores.append(prob)
            
            if not scores:
                return True, 75.0, "No models available"
            
            # Ensemble score (average)
            ensemble_score = np.mean(scores) * 100
            ensemble_prediction = sum(predictions) >= len(predictions) / 2
            
            should_take = ensemble_score >= self.min_score
            
            # Generate reason
            if should_take:
                top_features = []
                if features.is_golden_zone > 0.5:
                    top_features.append("Golden Zone")
                if features.higher_tf_trend_bullish > 0.5 and signal_meta.get('side') == 'long':
                    top_features.append("1H Aligned")
                if features.volume_ratio > 1.5:
                    top_features.append("High Vol")
                if features.pullback_quality_score > 70:
                    top_features.append(f"Quality {features.pullback_quality_score:.0f}")
                
                reason = f"Ensemble Score {ensemble_score:.1f} - " + ", ".join(top_features[:3])
            else:
                weak_features = []
                if features.is_deep_pullback > 0.5:
                    weak_features.append("Too Deep")
                if features.is_shallow_pullback > 0.5:
                    weak_features.append("Too Shallow")
                if features.higher_tf_trend_bearish > 0.5 and signal_meta.get('side') == 'long':
                    weak_features.append("1H Against")
                
                reason = f"Ensemble Score {ensemble_score:.1f} - " + ", ".join(weak_features[:3])
            
            # Log details
            logger.info(f"ML V2 Score: {ensemble_score:.1f} | Models agree: {sum(predictions)}/{len(predictions)} | "
                       f"Fib: {features.fibonacci_retracement:.1f}% | "
                       f"1H: {signal_meta.get('higher_tf_trend', 'N/A')} | "
                       f"Vol: {features.volatility_regime}")
            
            return should_take, ensemble_score, reason
            
        except Exception as e:
            logger.error(f"Error scoring signal: {e}")
            return True, 75.0, "Scoring error - allowing signal"
    
    def _train_models(self, X: np.array, y: np.array):
        """Train ensemble models"""
        
        # Random Forest
        self.models['rf'] = RandomForestClassifier(
            n_estimators=100, max_depth=10,
            min_samples_split=10, random_state=42
        )
        self.models['rf'].fit(X, y)
        
        # Gradient Boosting
        if XGB_AVAILABLE:
            self.models['gb'] = xgb.XGBClassifier(
                n_estimators=100, max_depth=5,
                learning_rate=0.1, random_state=42
            )
        else:
            self.models['gb'] = GradientBoostingClassifier(
                n_estimators=100, max_depth=5,
                learning_rate=0.1, random_state=42
            )
        self.models['gb'].fit(X, y)
        
        # Neural Network
        self.models['nn'] = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu', solver='adam',
            max_iter=500, random_state=42
        )
        self.models['nn'].fit(X, y)
        
        # Calculate feature importance
        if hasattr(self.models['rf'], 'feature_importances_'):
            self.feature_importance = dict(enumerate(self.models['rf'].feature_importances_))
    
    def record_trade_result(self, signal_id: str, features: Dict,
                           profit_loss: float, success: bool):
        """Record completed trade"""
        
        trade_data = {
            'signal_id': signal_id,
            'timestamp': datetime.now().isoformat(),
            'features': features if isinstance(features, dict) else asdict(features),
            'profit_loss': profit_loss,
            'success': success
        }
        
        # Store
        if self.redis_client:
            self.redis_client.rpush('ml_v2_completed_trades', json.dumps(trade_data))
            self.completed_trades_count += 1
            self.redis_client.set('ml_v2_trades_count', self.completed_trades_count)
        else:
            if hasattr(self, 'local_storage'):
                self.local_storage['completed_trades'].append(trade_data)
            self.completed_trades_count += 1
        
        # Check for retraining
        if self.completed_trades_count >= self.MIN_TRADES_TO_LEARN:
            if self.completed_trades_count - self.last_train_count >= self.RETRAIN_INTERVAL:
                self.retrain()
    
    def retrain(self):
        """Retrain all models"""
        try:
            logger.info("Training ML V2 ensemble models...")
            
            # Get trades
            if self.redis_client:
                trades_data = self.redis_client.lrange('ml_v2_completed_trades', 0, -1)
                trades = [json.loads(t) for t in trades_data]
            else:
                trades = getattr(self, 'local_storage', {}).get('completed_trades', [])
            
            if len(trades) < self.MIN_TRADES_TO_LEARN:
                return
            
            # Prepare data
            X = []
            y = []
            
            for trade in trades:
                features = trade['features']
                features.pop('volatility_regime', None)
                features.pop('session', None)
                
                X.append(list(features.values()))
                y.append(1 if trade['success'] else 0)
            
            X = np.array(X)
            y = np.array(y)
            
            # Scale and train
            X_scaled = self.scalers['ensemble'].fit_transform(X)
            self._train_models(X_scaled, y)
            
            # Save to Redis
            if self.redis_client:
                import base64
                for name, model in self.models.items():
                    if model:
                        model_data = base64.b64encode(pickle.dumps(model)).decode()
                        self.redis_client.set(f'ml_v2_model_{name}', model_data)
                
                scaler_data = base64.b64encode(pickle.dumps(self.scalers['ensemble'])).decode()
                self.redis_client.set('ml_v2_scaler_ensemble', scaler_data)
            
            self.is_trained = True
            self.last_train_count = self.completed_trades_count
            
            logger.info(f"✅ ML V2 ensemble trained on {len(trades)} trades")
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
    
    def get_stats(self) -> Dict:
        """Get ML statistics"""
        return {
            'enabled': self.enabled,
            'is_trained': self.is_trained,
            'completed_trades': self.completed_trades_count,
            'trades_needed': max(0, self.MIN_TRADES_TO_LEARN - self.completed_trades_count),
            'model_types': list(self.models.keys()),
            'min_score_threshold': self.min_score,
            'ensemble_performance': self.model_performance.get('ensemble', {}),
            'enhanced_features': True
        }

# Singleton instance
_ml_scorer_instance = None

def get_ensemble_scorer(min_score: float = 70.0, enabled: bool = True) -> EnsembleMLScorerV2:
    """Get or create singleton instance (backward compatible name)"""
    global _ml_scorer_instance
    if _ml_scorer_instance is None:
        _ml_scorer_instance = EnsembleMLScorerV2(min_score, enabled)
    return _ml_scorer_instance