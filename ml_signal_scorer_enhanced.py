"""
Enhanced Machine Learning Signal Scorer with New Pullback Features
Includes Fibonacci levels, 1H trend, and volatility regime learning
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class SignalFeatures:
    """Enhanced features including new pullback strategy improvements"""
    
    # Original trend strength features
    trend_strength: float  # 0-100, based on ADX or price momentum
    higher_tf_alignment: float  # 0-100, alignment with 1H trend
    ema_distance_ratio: float  # Distance from EMA as % of ATR
    
    # Enhanced with 1H trend classification
    higher_tf_trend_bullish: float  # 1.0 if 1H bullish, 0.0 if not
    higher_tf_trend_bearish: float  # 1.0 if 1H bearish, 0.0 if not
    higher_tf_trend_neutral: float  # 1.0 if 1H neutral, 0.0 if not
    
    # Volume features
    volume_ratio: float  # Current vs 20-period average
    volume_trend: float  # Volume momentum
    breakout_volume: float  # Volume at breakout vs average
    
    # Market structure features
    support_resistance_strength: float  # Based on pivot touches
    pullback_depth: float  # Depth of pullback as % of breakout move
    confirmation_candle_strength: float  # Size and volume of confirmation
    
    # NEW: Fibonacci features
    fibonacci_retracement: float  # Actual retracement percentage (0-100)
    is_golden_zone: float  # 1.0 if in 38.2-61.8%, 0.0 if not
    is_shallow_pullback: float  # 1.0 if <38.2%, 0.0 if not
    is_deep_pullback: float  # 1.0 if >78.6%, 0.0 if not
    fib_50_distance: float  # Distance from perfect 50% level
    
    # Enhanced volatility features
    atr_percentile: float  # Current ATR vs 100-period percentile
    volatility_regime: str  # "low", "normal", "high"
    volatility_low: float  # 1.0 if low vol, 0.0 if not
    volatility_normal: float  # 1.0 if normal vol, 0.0 if not
    volatility_high: float  # 1.0 if high vol, 0.0 if not
    
    # NEW: Dynamic parameter features
    adjusted_confirmation_candles: int  # Dynamic based on volatility
    adjusted_sl_buffer: float  # Dynamic stop loss buffer
    adjusted_rr_ratio: float  # Dynamic risk/reward ratio
    
    # Time features
    hour_of_day: int  # 0-23 UTC
    day_of_week: int  # 0-6
    session: str  # "asian", "european", "us", "off_hours"
    
    # Risk/Reward features
    risk_reward_ratio: float  # Actual RR based on levels
    atr_stop_distance: float  # Stop distance in ATR multiples
    
    # NEW: Pullback quality score
    pullback_quality_score: float  # Combined score 0-100

class MLSignalScorerEnhanced:
    """
    Enhanced ML scorer that learns from new pullback features
    """
    
    MIN_TRADES_TO_LEARN = 200  # Start ML after 200 trades
    RETRAIN_INTERVAL = 50  # Retrain every 50 new trades
    
    def __init__(self, min_score: float = 70.0, enabled: bool = True):
        self.min_score = min_score
        self.enabled = enabled
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.completed_trades_count = 0
        self.last_train_count = 0
        self.feature_importance = {}
        
        # Initialize Redis connection
        self.redis_client = None
        if enabled:
            self._init_redis()
            self._load_from_redis()
            
        # Check if ready for ML
        self._check_ready_for_ml()
    
    def _init_redis(self):
        """Initialize Redis connection"""
        try:
            redis_url = os.getenv('REDIS_URL')
            if redis_url:
                self.redis_client = redis.from_url(redis_url, decode_responses=True)
                self.redis_client.ping()
                logger.info("Connected to Redis for enhanced ML storage")
            else:
                logger.warning("No REDIS_URL found, ML data won't persist")
                self.local_storage = {'signals': [], 'completed_trades': []}
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            self.local_storage = {'signals': [], 'completed_trades': []}
    
    def _load_from_redis(self):
        """Load model and data from Redis"""
        if not self.redis_client:
            return
            
        try:
            # Load model
            model_data = self.redis_client.get('ml_enhanced_model')
            if model_data:
                import base64
                self.model = pickle.loads(base64.b64decode(model_data))
                self.is_trained = True
                logger.info("Loaded enhanced ML model from Redis")
            
            # Load scaler
            scaler_data = self.redis_client.get('ml_enhanced_scaler')
            if scaler_data:
                import base64
                self.scaler = pickle.loads(base64.b64decode(scaler_data))
            
            # Load completed trades count
            count = self.redis_client.get('ml_enhanced_trades_count')
            self.completed_trades_count = int(count) if count else 0
            
            # Load feature importance
            importance_data = self.redis_client.get('ml_enhanced_feature_importance')
            if importance_data:
                self.feature_importance = json.loads(importance_data)
                
        except Exception as e:
            logger.error(f"Error loading from Redis: {e}")
    
    def extract_enhanced_features(self, signal_meta: Dict, df: pd.DataFrame, 
                                 df_1h: Optional[pd.DataFrame] = None) -> SignalFeatures:
        """
        Extract enhanced features including Fibonacci and multi-timeframe
        """
        
        # Calculate basic trend strength (simplified)
        close_prices = df['close'].values[-20:]
        trend_strength = abs(np.polyfit(range(len(close_prices)), close_prices, 1)[0]) * 100
        trend_strength = min(100, trend_strength)
        
        # Extract 1H trend if available
        higher_tf_trend_bullish = 0.0
        higher_tf_trend_bearish = 0.0
        higher_tf_trend_neutral = 1.0
        
        if 'higher_tf_trend' in signal_meta:
            trend = signal_meta['higher_tf_trend']
            if trend == "BULLISH":
                higher_tf_trend_bullish = 1.0
                higher_tf_trend_neutral = 0.0
            elif trend == "BEARISH":
                higher_tf_trend_bearish = 1.0
                higher_tf_trend_neutral = 0.0
        
        # Volume features
        avg_volume = df['volume'].rolling(20).mean().iloc[-1]
        current_volume = df['volume'].iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Fibonacci features from signal metadata
        fibonacci_retracement = 0.5  # Default 50%
        if 'fib_retracement' in signal_meta:
            # Parse percentage string like "38.2%"
            fib_str = signal_meta['fib_retracement'].rstrip('%')
            try:
                fibonacci_retracement = float(fib_str) / 100
            except:
                fibonacci_retracement = 0.5
        
        # Determine Fibonacci zones
        is_golden_zone = 1.0 if 0.382 <= fibonacci_retracement <= 0.618 else 0.0
        is_shallow_pullback = 1.0 if fibonacci_retracement < 0.382 else 0.0
        is_deep_pullback = 1.0 if fibonacci_retracement > 0.786 else 0.0
        fib_50_distance = abs(fibonacci_retracement - 0.5)
        
        # Volatility features from metadata
        volatility_regime = signal_meta.get('volatility', 'NORMAL')
        volatility_low = 1.0 if volatility_regime == "LOW" else 0.0
        volatility_normal = 1.0 if volatility_regime == "NORMAL" else 0.0
        volatility_high = 1.0 if volatility_regime == "HIGH" else 0.0
        
        # Dynamic parameters (from enhanced strategy)
        adjusted_confirmation = signal_meta.get('confirmation_candles', 2)
        
        # Adjust stop buffer based on volatility
        base_sl_buffer = 0.5
        if volatility_regime == "LOW":
            adjusted_sl_buffer = 0.3
        elif volatility_regime == "HIGH":
            adjusted_sl_buffer = 0.7
        else:
            adjusted_sl_buffer = 0.5
        
        # Adjust R:R based on volatility
        if volatility_regime == "LOW":
            adjusted_rr = 2.5
        elif volatility_regime == "HIGH":
            adjusted_rr = 1.8
        else:
            adjusted_rr = 2.0
        
        # Calculate pullback quality score
        pullback_quality_score = self._calculate_pullback_quality(
            fibonacci_retracement, 
            is_golden_zone,
            trend_strength,
            volume_ratio,
            volatility_regime
        )
        
        # Time features
        now = datetime.now()
        hour = now.hour
        day_of_week = now.weekday()
        
        # Determine trading session
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
        atr_values = [self._calculate_atr(df.iloc[i-14:i]) for i in range(100, len(df))]
        if atr_values:
            atr_percentile = np.percentile(atr_values, 
                                          np.searchsorted(np.sort(atr_values), atr) * 100 / len(atr_values))
        else:
            atr_percentile = 50.0
        
        return SignalFeatures(
            # Trend features
            trend_strength=trend_strength,
            higher_tf_alignment=higher_tf_trend_bullish * 100 if signal_meta.get('side') == 'long' else higher_tf_trend_bearish * 100,
            ema_distance_ratio=0.0,  # Could calculate if EMA provided
            
            # 1H trend features
            higher_tf_trend_bullish=higher_tf_trend_bullish,
            higher_tf_trend_bearish=higher_tf_trend_bearish,
            higher_tf_trend_neutral=higher_tf_trend_neutral,
            
            # Volume features
            volume_ratio=volume_ratio,
            volume_trend=0.0,  # Simplified
            breakout_volume=volume_ratio,  # Use current as proxy
            
            # Structure features
            support_resistance_strength=75.0,  # Default medium strength
            pullback_depth=fibonacci_retracement * 100,
            confirmation_candle_strength=50.0,  # Default medium
            
            # Fibonacci features
            fibonacci_retracement=fibonacci_retracement * 100,
            is_golden_zone=is_golden_zone,
            is_shallow_pullback=is_shallow_pullback,
            is_deep_pullback=is_deep_pullback,
            fib_50_distance=fib_50_distance * 100,
            
            # Volatility features
            atr_percentile=atr_percentile,
            volatility_regime=volatility_regime,
            volatility_low=volatility_low,
            volatility_normal=volatility_normal,
            volatility_high=volatility_high,
            
            # Dynamic parameters
            adjusted_confirmation_candles=adjusted_confirmation,
            adjusted_sl_buffer=adjusted_sl_buffer,
            adjusted_rr_ratio=adjusted_rr,
            
            # Time features
            hour_of_day=hour,
            day_of_week=day_of_week,
            session=session,
            
            # Risk features
            risk_reward_ratio=adjusted_rr,
            atr_stop_distance=adjusted_sl_buffer,
            
            # Quality score
            pullback_quality_score=pullback_quality_score
        )
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR"""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        if len(df) < period:
            return 0.0
        
        tr = np.maximum(high[1:] - low[1:], 
                       np.maximum(np.abs(high[1:] - close[:-1]),
                                 np.abs(low[1:] - close[:-1])))
        return np.mean(tr[-period:])
    
    def _calculate_pullback_quality(self, fib_level: float, is_golden: float,
                                   trend_strength: float, volume_ratio: float,
                                   volatility: str) -> float:
        """
        Calculate overall pullback quality score (0-100)
        Higher score = better quality pullback
        """
        score = 0.0
        
        # Fibonacci contribution (40 points max)
        if is_golden:
            score += 40  # Perfect zone
        elif fib_level < 0.382:
            score += 10  # Too shallow
        elif fib_level > 0.786:
            score += 5   # Too deep
        else:
            score += 25  # Acceptable but not ideal
        
        # Trend strength contribution (30 points max)
        score += min(30, trend_strength * 0.3)
        
        # Volume contribution (20 points max)
        if volume_ratio > 1.5:
            score += 20
        elif volume_ratio > 1.2:
            score += 15
        elif volume_ratio > 1.0:
            score += 10
        else:
            score += 5
        
        # Volatility adjustment (10 points max)
        if volatility == "NORMAL":
            score += 10
        elif volatility == "LOW":
            score += 7
        else:  # HIGH
            score += 5
        
        return min(100, score)
    
    def score_signal(self, signal_meta: Dict, df: pd.DataFrame,
                    df_1h: Optional[pd.DataFrame] = None) -> Tuple[bool, float, str]:
        """
        Score a signal with enhanced features
        Returns: (should_take, score, reason)
        """
        
        if not self.enabled:
            return True, 75.0, "ML scoring disabled"
        
        try:
            # Extract enhanced features
            features = self.extract_enhanced_features(signal_meta, df, df_1h)
            
            # If model is trained, use it
            if self.is_trained and self.model:
                # Prepare features for model
                feature_dict = asdict(features)
                
                # Remove non-numeric features
                feature_dict.pop('volatility_regime', None)
                feature_dict.pop('session', None)
                
                # Convert to array
                X = np.array(list(feature_dict.values())).reshape(1, -1)
                
                # Scale features
                X_scaled = self.scaler.transform(X)
                
                # Get prediction probability
                prob = self.model.predict_proba(X_scaled)[0, 1]
                score = prob * 100
                
                # Check if meets minimum score
                should_take = score >= self.min_score
                
                # Generate detailed reason
                if should_take:
                    top_features = self._get_top_features(feature_dict)
                    reason = f"Score {score:.1f} - Strong: {', '.join(top_features)}"
                else:
                    weak_features = self._get_weak_features(feature_dict)
                    reason = f"Score {score:.1f} - Weak: {', '.join(weak_features)}"
                
                # Log important features
                logger.info(f"ML Score: {score:.1f} | Fib: {features.fibonacci_retracement:.1f}% | "
                           f"1H: {signal_meta.get('higher_tf_trend', 'N/A')} | "
                           f"Vol: {features.volatility_regime} | "
                           f"Quality: {features.pullback_quality_score:.0f}")
                
                return should_take, score, reason
            
            else:
                # Not trained yet, use pullback quality score
                score = features.pullback_quality_score
                should_take = score >= 60  # Lower threshold before ML kicks in
                
                reason = f"Pre-ML Score {score:.0f} - Fib: {features.fibonacci_retracement:.1f}%, "
                reason += f"1H: {signal_meta.get('higher_tf_trend', 'N/A')}, "
                reason += f"Vol: {features.volatility_regime}"
                
                return should_take, score, reason
                
        except Exception as e:
            logger.error(f"Error scoring signal: {e}")
            return True, 75.0, "Scoring error - allowing signal"
    
    def _get_top_features(self, feature_dict: Dict) -> List[str]:
        """Get top positive contributing features"""
        top_features = []
        
        if feature_dict.get('is_golden_zone', 0) > 0.5:
            top_features.append("Golden Zone")
        if feature_dict.get('higher_tf_trend_bullish', 0) > 0.5:
            top_features.append("1H Bullish")
        if feature_dict.get('volume_ratio', 1.0) > 1.5:
            top_features.append("High Volume")
        if feature_dict.get('pullback_quality_score', 0) > 70:
            top_features.append("Quality Pullback")
        
        return top_features[:3] if top_features else ["Good Setup"]
    
    def _get_weak_features(self, feature_dict: Dict) -> List[str]:
        """Get weak features causing low score"""
        weak_features = []
        
        if feature_dict.get('is_deep_pullback', 0) > 0.5:
            weak_features.append("Too Deep")
        if feature_dict.get('is_shallow_pullback', 0) > 0.5:
            weak_features.append("Too Shallow")
        if feature_dict.get('volume_ratio', 1.0) < 0.8:
            weak_features.append("Low Volume")
        if feature_dict.get('higher_tf_trend_bearish', 0) > 0.5:
            weak_features.append("1H Against")
        
        return weak_features[:3] if weak_features else ["Weak Setup"]
    
    def record_trade_result(self, signal_id: str, features: SignalFeatures,
                           profit_loss: float, success: bool):
        """Record completed trade with enhanced features"""
        
        trade_data = {
            'signal_id': signal_id,
            'timestamp': datetime.now().isoformat(),
            'features': asdict(features),
            'profit_loss': profit_loss,
            'success': success
        }
        
        # Store in Redis or local
        if self.redis_client:
            # Add to completed trades list
            self.redis_client.rpush('ml_enhanced_completed_trades', json.dumps(trade_data))
            
            # Update count
            self.completed_trades_count += 1
            self.redis_client.set('ml_enhanced_trades_count', self.completed_trades_count)
            
        else:
            self.local_storage['completed_trades'].append(trade_data)
            self.completed_trades_count += 1
        
        # Check if time to retrain
        if self.completed_trades_count >= self.MIN_TRADES_TO_LEARN:
            if self.completed_trades_count - self.last_train_count >= self.RETRAIN_INTERVAL:
                self._train_model()
    
    def _train_model(self):
        """Train the enhanced ML model"""
        try:
            logger.info("Training enhanced ML model...")
            
            # Get all completed trades
            if self.redis_client:
                trades_data = self.redis_client.lrange('ml_enhanced_completed_trades', 0, -1)
                trades = [json.loads(t) for t in trades_data]
            else:
                trades = self.local_storage['completed_trades']
            
            if len(trades) < self.MIN_TRADES_TO_LEARN:
                return
            
            # Prepare training data
            X = []
            y = []
            
            for trade in trades:
                features = trade['features']
                # Remove non-numeric features
                features.pop('volatility_regime', None)
                features.pop('session', None)
                
                X.append(list(features.values()))
                y.append(1 if trade['success'] else 0)
            
            X = np.array(X)
            y = np.array(y)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                random_state=42
            )
            self.model.fit(X_scaled, y)
            
            # Calculate feature importance
            feature_names = list(trades[0]['features'].keys())
            feature_names = [f for f in feature_names if f not in ['volatility_regime', 'session']]
            
            importances = self.model.feature_importances_
            self.feature_importance = dict(zip(feature_names, importances))
            
            # Log top features
            top_features = sorted(self.feature_importance.items(), 
                                key=lambda x: x[1], reverse=True)[:5]
            logger.info("Top 5 important features:")
            for feat, imp in top_features:
                logger.info(f"  {feat}: {imp:.3f}")
            
            # Save to Redis
            if self.redis_client:
                import base64
                model_data = base64.b64encode(pickle.dumps(self.model)).decode()
                scaler_data = base64.b64encode(pickle.dumps(self.scaler)).decode()
                
                self.redis_client.set('ml_enhanced_model', model_data)
                self.redis_client.set('ml_enhanced_scaler', scaler_data)
                self.redis_client.set('ml_enhanced_feature_importance', 
                                    json.dumps(self.feature_importance))
            
            self.is_trained = True
            self.last_train_count = self.completed_trades_count
            
            # Calculate model performance
            accuracy = self.model.score(X_scaled, y)
            logger.info(f"✅ Enhanced ML model trained! Accuracy: {accuracy:.2%}")
            logger.info(f"Total trades learned from: {len(trades)}")
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
    
    def _check_ready_for_ml(self):
        """Check if we have enough data to use ML"""
        if self.completed_trades_count >= self.MIN_TRADES_TO_LEARN and not self.is_trained:
            logger.info(f"Have {self.completed_trades_count} trades, training initial model...")
            self._train_model()
    
    def get_stats(self) -> Dict:
        """Get current ML statistics"""
        return {
            'enabled': self.enabled,
            'is_trained': self.is_trained,
            'completed_trades': self.completed_trades_count,
            'trades_needed': max(0, self.MIN_TRADES_TO_LEARN - self.completed_trades_count),
            'last_train_count': self.last_train_count,
            'model_type': 'RandomForestClassifier' if self.model else 'None',
            'min_score_threshold': self.min_score,
            'top_features': dict(sorted(self.feature_importance.items(), 
                                       key=lambda x: x[1], reverse=True)[:5])
        }

# Singleton instance
_ml_scorer_instance = None

def get_ml_scorer(min_score: float = 70.0, enabled: bool = True) -> MLSignalScorerEnhanced:
    """Get or create the singleton ML scorer instance"""
    global _ml_scorer_instance
    if _ml_scorer_instance is None:
        _ml_scorer_instance = MLSignalScorerEnhanced(min_score, enabled)
    return _ml_scorer_instance