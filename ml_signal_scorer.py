"""
Machine Learning Signal Scorer with Redis Storage
Enhances pullback strategy signals with quality scores (0-100)
Learns after 200 trades, then continuously improves
Designed to be non-breaking - falls back to allowing all signals if any issues
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
    """Features extracted from a potential signal"""
    # Trend strength features
    trend_strength: float  # 0-100, based on ADX or price momentum
    higher_tf_alignment: float  # 0-100, alignment with 1H trend
    ema_distance_ratio: float  # Distance from EMA as % of ATR
    
    # Volume features
    volume_ratio: float  # Current vs 20-period average
    volume_trend: float  # Volume momentum (increasing/decreasing)
    breakout_volume: float  # Volume at breakout vs average
    
    # Market structure features
    support_resistance_strength: float  # Based on pivot touches
    pullback_depth: float  # Depth of pullback as % of breakout move
    confirmation_candle_strength: float  # Size and volume of confirmation candles
    
    # Volatility features
    atr_percentile: float  # Current ATR vs 100-period percentile
    volatility_regime: str  # "low", "normal", "high"
    
    # Time features
    hour_of_day: int  # 0-23 UTC
    day_of_week: int  # 0-6
    session: str  # "asian", "european", "us", "off_hours"
    
    # Risk/Reward features
    risk_reward_ratio: float  # Actual RR based on levels
    atr_stop_distance: float  # Stop distance in ATR multiples

class MLSignalScorer:
    """
    Scores trading signals from 0-100 based on historical performance patterns
    Uses Redis for persistent storage across deployments
    Learns after 200 trades, then continuously improves
    Falls back gracefully to not interfere with existing strategy
    """
    
    MIN_TRADES_TO_LEARN = 200  # Start ML scoring after 200 completed trades
    RETRAIN_INTERVAL = 50  # Retrain model every 50 new completed trades
    
    def __init__(self, min_score: float = 70.0, enabled: bool = True):
        self.min_score = min_score
        self.enabled = enabled
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.completed_trades_count = 0
        self.last_train_count = 0
        
        # Initialize Redis connection
        self.redis_client = None
        if enabled:
            self._init_redis()
            self._load_from_redis()
            
        # Check if we can start using ML
        self._check_ready_for_ml()
    
    def _init_redis(self):
        """Initialize Redis connection using Railway Redis URL"""
        try:
            redis_url = os.getenv('REDIS_URL')
            if redis_url:
                self.redis_client = redis.from_url(redis_url, decode_responses=True)
                self.redis_client.ping()
                logger.info("Connected to Redis for ML data storage")
            else:
                logger.warning("No REDIS_URL found, ML data will not persist across restarts")
                # Use local in-memory fallback
                self.local_storage = {'signals': [], 'completed_trades': []}
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Using local storage.")
            self.local_storage = {'signals': [], 'completed_trades': []}
    
    def _load_from_redis(self):
        """Load model and historical data from Redis"""
        try:
            if self.redis_client:
                # Load completed trades count
                count = self.redis_client.get('ml:completed_trades_count')
                self.completed_trades_count = int(count) if count else 0
                
                # Load last train count
                last_train = self.redis_client.get('ml:last_train_count')
                self.last_train_count = int(last_train) if last_train else 0
                
                # Load trained model if exists
                model_data = self.redis_client.get('ml:model')
                if model_data:
                    model_dict = pickle.loads(model_data.encode('latin-1'))
                    self.model = model_dict['model']
                    self.scaler = model_dict['scaler']
                    self.is_trained = True
                    logger.info(f"Loaded ML model from Redis (trained on {self.last_train_count} trades)")
                
                logger.info(f"ML Status: {self.completed_trades_count} completed trades, "
                          f"{'Model Active' if self.is_trained else f'Collecting data ({self.MIN_TRADES_TO_LEARN - self.completed_trades_count} more needed)'}")
        except Exception as e:
            logger.error(f"Failed to load from Redis: {e}")
    
    def _check_ready_for_ml(self):
        """Check if we have enough data to start using ML"""
        if self.completed_trades_count >= self.MIN_TRADES_TO_LEARN and not self.is_trained:
            logger.info(f"Ready to train ML model with {self.completed_trades_count} trades")
            self._train_model()
        elif self.is_trained and (self.completed_trades_count - self.last_train_count) >= self.RETRAIN_INTERVAL:
            logger.info(f"Retraining model with {self.completed_trades_count} trades")
            self._train_model()
    
    def extract_features(self, df: pd.DataFrame, signal_type: str, 
                         entry: float, sl: float, tp: float,
                         symbol: str, meta: dict) -> Optional[SignalFeatures]:
        """
        Extract ML features from the current market state
        Returns None if feature extraction fails (safety fallback)
        """
        try:
            # Calculate trend strength (simplified ADX alternative)
            high, low, close = df["high"], df["low"], df["close"]
            
            # Price momentum for trend strength (0-100)
            momentum_period = 20
            if len(df) > momentum_period:
                price_change = (close.iloc[-1] - close.iloc[-momentum_period]) / close.iloc[-momentum_period]
                trend_strength = min(abs(price_change) * 1000, 100)  # Scale to 0-100
            else:
                trend_strength = 50  # Neutral if not enough data
            
            # EMA distance
            atr = meta.get('atr', 0.001)
            if 'ema_len' in meta and len(df) >= 200:
                ema_200 = df['close'].ewm(span=200, adjust=False).mean().iloc[-1]
                ema_distance = abs(close.iloc[-1] - ema_200) / atr
                ema_distance_ratio = min(ema_distance, 10.0)  # Cap at 10 ATRs
            else:
                ema_distance_ratio = 1.0
            
            # Volume analysis
            volume = df["volume"]
            vol_ma = volume.rolling(20).mean().iloc[-1]
            volume_ratio = volume.iloc[-1] / vol_ma if vol_ma > 0 else 1.0
            
            # Volume trend (increasing or decreasing)
            if len(volume) > 5:
                recent_vol_ma = volume.iloc[-5:].mean()
                older_vol_ma = volume.iloc[-10:-5].mean()
                volume_trend = (recent_vol_ma - older_vol_ma) / older_vol_ma if older_vol_ma > 0 else 0
            else:
                volume_trend = 0
            
            # Breakout volume (from meta if available)
            breakout_volume = meta.get('breakout_volume_ratio', volume_ratio)
            
            # Support/Resistance strength (simplified - count recent touches)
            res_level = meta.get('res', entry)
            sup_level = meta.get('sup', entry)
            touches = 0
            for i in range(max(-50, -len(df)), 0):
                if signal_type == "long" and abs(df['high'].iloc[i] - res_level) / atr < 0.5:
                    touches += 1
                elif signal_type == "short" and abs(df['low'].iloc[i] - sup_level) / atr < 0.5:
                    touches += 1
            support_resistance_strength = min(touches * 10, 100)
            
            # Pullback depth
            if 'breakout_level' in meta and 'pullback_extreme' in meta:
                breakout_level = meta['breakout_level']
                pullback_extreme = meta.get('pullback_low' if signal_type == 'long' else 'pullback_high', entry)
                breakout_move = abs(entry - breakout_level)
                pullback_move = abs(pullback_extreme - breakout_level)
                pullback_depth = (pullback_move / breakout_move * 100) if breakout_move > 0 else 50
            else:
                pullback_depth = 38.2  # Default to common Fibonacci level
            
            # Confirmation candle strength
            confirmation_candle_strength = 50.0  # Default
            if len(df) >= 2:
                last_candle_size = abs(df['close'].iloc[-1] - df['open'].iloc[-1])
                last_candle_vol = df['volume'].iloc[-1]
                avg_candle_size = abs(df['close'] - df['open']).rolling(20).mean().iloc[-1]
                if avg_candle_size > 0:
                    confirmation_candle_strength = min((last_candle_size / avg_candle_size) * 50, 100)
            
            # ATR percentile
            if len(df) >= 100:
                atr_series = pd.Series([atr])
                atr_100 = self._calculate_atr_series(df.tail(100))
                atr_percentile = (atr_series.iloc[0] > atr_100).sum() / len(atr_100) * 100
            else:
                atr_percentile = 50
            
            # Volatility regime
            if atr_percentile < 30:
                volatility_regime = "low"
            elif atr_percentile < 70:
                volatility_regime = "normal"
            else:
                volatility_regime = "high"
            
            # Time features
            current_time = datetime.now()
            hour_of_day = current_time.hour
            day_of_week = current_time.weekday()
            
            # Trading session
            if 0 <= hour_of_day < 8:
                session = "asian"
            elif 8 <= hour_of_day < 16:
                session = "european"
            elif 16 <= hour_of_day < 23:
                session = "us"
            else:
                session = "off_hours"
            
            # Risk/Reward
            risk = abs(entry - sl)
            reward = abs(tp - entry)
            risk_reward_ratio = reward / risk if risk > 0 else 2.0
            atr_stop_distance = risk / atr if atr > 0 else 1.0
            
            # Higher timeframe alignment (simplified - check 1H trend)
            if len(df) >= 12:  # 12 * 15min = 3 hours
                older_price = df['close'].iloc[-12]
                recent_price = df['close'].iloc[-1]
                ht_trend = "up" if recent_price > older_price else "down"
                if (signal_type == "long" and ht_trend == "up") or (signal_type == "short" and ht_trend == "down"):
                    higher_tf_alignment = 80.0
                else:
                    higher_tf_alignment = 20.0
            else:
                higher_tf_alignment = 50.0
            
            return SignalFeatures(
                trend_strength=trend_strength,
                higher_tf_alignment=higher_tf_alignment,
                ema_distance_ratio=ema_distance_ratio,
                volume_ratio=volume_ratio,
                volume_trend=volume_trend,
                breakout_volume=breakout_volume,
                support_resistance_strength=support_resistance_strength,
                pullback_depth=pullback_depth,
                confirmation_candle_strength=confirmation_candle_strength,
                atr_percentile=atr_percentile,
                volatility_regime=volatility_regime,
                hour_of_day=hour_of_day,
                day_of_week=day_of_week,
                session=session,
                risk_reward_ratio=risk_reward_ratio,
                atr_stop_distance=atr_stop_distance
            )
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return None
    
    def _calculate_atr_series(self, df: pd.DataFrame) -> pd.Series:
        """Calculate ATR for a dataframe"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        return atr.dropna()
    
    def score_signal(self, features: Optional[SignalFeatures]) -> Tuple[float, str]:
        """
        Score a signal from 0-100 based on features
        Returns (score, reason)
        Falls back to 75.0 (pass) if ML is disabled or fails
        """
        # Safety fallback - if no features or ML disabled, give neutral passing score
        if not self.enabled or features is None:
            return 75.0, "ML scoring disabled or unavailable"
        
        # Check if we're still collecting data
        if not self.is_trained:
            trades_needed = self.MIN_TRADES_TO_LEARN - self.completed_trades_count
            if trades_needed > 0:
                return 75.0, f"Collecting data: {self.completed_trades_count}/{self.MIN_TRADES_TO_LEARN} trades"
        
        # If we have a trained model, use it
        if self.model is not None and self.is_trained:
            try:
                # Convert features to array for model
                feature_array = self._features_to_array(features)
                
                # Scale features
                feature_array_scaled = self.scaler.transform(feature_array.reshape(1, -1))
                
                # Get prediction probability
                prob = self.model.predict_proba(feature_array_scaled)[0, 1]
                score = float(prob * 100)
                
                # Add confidence context
                if score > 80:
                    reason = f"ML: High confidence ({score:.0f}%)"
                elif score > 60:
                    reason = f"ML: Moderate confidence ({score:.0f}%)"
                else:
                    reason = f"ML: Low confidence ({score:.0f}%)"
                
                return score, reason
                
            except Exception as e:
                logger.error(f"Model prediction failed: {e}")
                return 75.0, "ML prediction failed, using default"
        
        # No model yet - use rule-based scoring
        return self._rule_based_scoring(features)
    
    def _train_model(self):
        """Train the ML model on completed trades"""
        try:
            # Get completed trades from Redis
            trades = self._get_completed_trades()
            
            if len(trades) < self.MIN_TRADES_TO_LEARN:
                logger.warning(f"Not enough trades for training: {len(trades)}")
                return
            
            # Prepare training data
            X, y = self._prepare_training_data(trades)
            
            if len(X) < 50:  # Safety check
                logger.warning("Not enough valid training samples")
                return
            
            # Train model
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                class_weight='balanced'  # Handle imbalanced win/loss
            )
            
            # Fit scaler and model
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            
            # Calculate training accuracy
            train_score = self.model.score(X_scaled, y)
            
            # Save to Redis
            self._save_model_to_redis()
            
            self.is_trained = True
            self.last_train_count = self.completed_trades_count
            
            logger.info(f"✅ ML Model trained successfully on {len(X)} trades. "
                       f"Training accuracy: {train_score:.2%}")
            
            # Log feature importance
            self._log_feature_importance()
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            self.is_trained = False
    
    def _prepare_training_data(self, trades: List[dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from completed trades"""
        X = []
        y = []
        
        for trade in trades:
            if 'features' in trade and 'outcome' in trade:
                features = trade['features']
                feature_array = self._features_dict_to_array(features)
                X.append(feature_array)
                
                # Binary classification: win/breakeven = 1, loss = 0
                y.append(1 if trade['outcome'] in ['win', 'breakeven'] else 0)
        
        return np.array(X), np.array(y)
    
    def _features_dict_to_array(self, features: dict) -> np.ndarray:
        """Convert feature dictionary to numpy array"""
        volatility_map = {"low": 0, "normal": 1, "high": 2}
        session_map = {"asian": 0, "european": 1, "us": 2, "off_hours": 3}
        
        return np.array([
            features.get('trend_strength', 50),
            features.get('higher_tf_alignment', 50),
            features.get('ema_distance_ratio', 1),
            features.get('volume_ratio', 1),
            features.get('volume_trend', 0),
            features.get('breakout_volume', 1),
            features.get('support_resistance_strength', 50),
            features.get('pullback_depth', 50),
            features.get('confirmation_candle_strength', 50),
            features.get('atr_percentile', 50),
            volatility_map.get(features.get('volatility_regime', 'normal'), 1),
            features.get('hour_of_day', 12),
            features.get('day_of_week', 3),
            session_map.get(features.get('session', 'us'), 2),
            features.get('risk_reward_ratio', 2),
            features.get('atr_stop_distance', 1)
        ])
    
    def _save_model_to_redis(self):
        """Save trained model to Redis"""
        try:
            if self.redis_client and self.model:
                model_data = {
                    'model': self.model,
                    'scaler': self.scaler,
                    'train_count': self.completed_trades_count,
                    'timestamp': datetime.now().isoformat()
                }
                model_bytes = pickle.dumps(model_data).decode('latin-1')
                self.redis_client.set('ml:model', model_bytes)
                self.redis_client.set('ml:last_train_count', str(self.last_train_count))
                logger.info("Model saved to Redis")
        except Exception as e:
            logger.error(f"Failed to save model to Redis: {e}")
    
    def _log_feature_importance(self):
        """Log feature importance from trained model"""
        if self.model and hasattr(self.model, 'feature_importances_'):
            feature_names = [
                'trend_strength', 'higher_tf_alignment', 'ema_distance',
                'volume_ratio', 'volume_trend', 'breakout_volume',
                'support_resistance', 'pullback_depth', 'confirmation_strength',
                'atr_percentile', 'volatility_regime', 'hour', 'day_of_week',
                'session', 'risk_reward', 'atr_stop_distance'
            ]
            
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1][:5]  # Top 5
            
            logger.info("Top 5 most important features:")
            for i, idx in enumerate(indices, 1):
                logger.info(f"  {i}. {feature_names[idx]}: {importances[idx]:.3f}")
    
    def _rule_based_scoring(self, features: SignalFeatures) -> Tuple[float, str]:
        """
        Rule-based scoring when no ML model is available
        This provides immediate value while collecting data for ML training
        """
        score = 50.0  # Base score
        reasons = []
        
        # Trend strength bonus (up to +15)
        if features.trend_strength > 70:
            score += 15
            reasons.append("strong trend")
        elif features.trend_strength > 40:
            score += 8
        
        # Higher timeframe alignment (up to +15)
        if features.higher_tf_alignment > 70:
            score += 15
            reasons.append("HTF aligned")
        elif features.higher_tf_alignment < 30:
            score -= 10
            reasons.append("HTF conflict")
        
        # Volume confirmation (up to +10)
        if features.volume_ratio > 1.5:
            score += 10
            reasons.append("high volume")
        elif features.volume_ratio < 0.7:
            score -= 5
            reasons.append("low volume")
        
        # Pullback quality (up to +10)
        if 38.2 <= features.pullback_depth <= 61.8:  # Fibonacci zone
            score += 10
            reasons.append("good pullback")
        elif features.pullback_depth > 78.6:
            score -= 10
            reasons.append("deep pullback")
        
        # Support/Resistance strength (up to +10)
        if features.support_resistance_strength > 50:
            score += 10
            reasons.append("strong S/R")
        
        # Volatility regime (up to +/-5)
        if features.volatility_regime == "normal":
            score += 5
        elif features.volatility_regime == "high":
            score -= 5
            reasons.append("high volatility")
        
        # Time-based adjustments
        if features.session == "off_hours":
            score -= 5
        elif features.session == "us":
            score += 3
        
        # Risk/Reward bonus
        if features.risk_reward_ratio > 2.5:
            score += 5
            reasons.append("good RR")
        
        # Cap score between 0 and 100
        score = max(0, min(100, score))
        
        reason = f"Rule-based: {', '.join(reasons)}" if reasons else f"Rule-based score: {score:.0f}"
        return score, reason
    
    def _features_to_array(self, features: SignalFeatures) -> np.ndarray:
        """Convert features to numpy array for model input"""
        # Map categorical to numeric
        volatility_map = {"low": 0, "normal": 1, "high": 2}
        session_map = {"asian": 0, "european": 1, "us": 2, "off_hours": 3}
        
        return np.array([
            features.trend_strength,
            features.higher_tf_alignment,
            features.ema_distance_ratio,
            features.volume_ratio,
            features.volume_trend,
            features.breakout_volume,
            features.support_resistance_strength,
            features.pullback_depth,
            features.confirmation_candle_strength,
            features.atr_percentile,
            volatility_map.get(features.volatility_regime, 1),
            features.hour_of_day,
            features.day_of_week,
            session_map.get(features.session, 3),
            features.risk_reward_ratio,
            features.atr_stop_distance
        ])
    
    def get_ml_stats(self) -> dict:
        """Get current ML system statistics"""
        stats = {
            'enabled': self.enabled,
            'is_trained': self.is_trained,
            'completed_trades': self.completed_trades_count,
            'trades_needed': max(0, self.MIN_TRADES_TO_LEARN - self.completed_trades_count),
            'last_train_count': self.last_train_count,
            'model_type': type(self.model).__name__ if self.model else 'None',
            'min_score_threshold': self.min_score
        }
        
        # Add performance metrics if available
        if self.is_trained and self.redis_client:
            try:
                # Get recent predictions accuracy
                recent_trades = self._get_completed_trades()[-50:]  # Last 50 trades
                if recent_trades:
                    correct = sum(1 for t in recent_trades 
                                if t.get('score', 0) > 70 and t.get('outcome') == 'win')
                    stats['recent_accuracy'] = correct / len(recent_trades)
            except:
                pass
        
        return stats
    
    def should_take_signal(self, df: pd.DataFrame, signal_type: str,
                          entry: float, sl: float, tp: float,
                          symbol: str, meta: dict) -> Tuple[bool, float, str]:
        """
        Main method to check if a signal should be taken
        Returns (should_take, score, reason)
        ALWAYS returns True if ML is disabled or errors occur (safety fallback)
        """
        try:
            # Extract features
            features = self.extract_features(df, signal_type, entry, sl, tp, symbol, meta)
            
            # Score the signal
            score, reason = self.score_signal(features)
            
            # Collect data for future training (non-blocking)
            self._collect_signal_data(symbol, signal_type, features, score, entry, sl, tp)
            
            # Decision
            should_take = score >= self.min_score
            
            if not should_take:
                reason = f"Score {score:.1f} below threshold {self.min_score}. {reason}"
            else:
                reason = f"Score {score:.1f} - {reason}"
            
            return should_take, score, reason
            
        except Exception as e:
            logger.error(f"Signal scoring error: {e}")
            # SAFETY: Always return True if anything goes wrong
            return True, 75.0, "ML scoring error - allowing signal"
    
    def _collect_signal_data(self, symbol: str, signal_type: str, 
                            features: Optional[SignalFeatures], score: float,
                            entry: float, sl: float, tp: float):
        """Collect signal data for future model training"""
        try:
            if features is None:
                return
            
            signal_data = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'type': signal_type,
                'entry': entry,
                'sl': sl,
                'tp': tp,
                'score': score,
                'features': {
                    'trend_strength': features.trend_strength,
                    'higher_tf_alignment': features.higher_tf_alignment,
                    'ema_distance_ratio': features.ema_distance_ratio,
                    'volume_ratio': features.volume_ratio,
                    'volume_trend': features.volume_trend,
                    'breakout_volume': features.breakout_volume,
                    'support_resistance_strength': features.support_resistance_strength,
                    'pullback_depth': features.pullback_depth,
                    'confirmation_candle_strength': features.confirmation_candle_strength,
                    'atr_percentile': features.atr_percentile,
                    'volatility_regime': features.volatility_regime,
                    'hour_of_day': features.hour_of_day,
                    'day_of_week': features.day_of_week,
                    'session': features.session,
                    'risk_reward_ratio': features.risk_reward_ratio,
                    'atr_stop_distance': features.atr_stop_distance
                },
                'outcome': None  # Will be updated when position closes
            }
            
            # Save to Redis
            self._save_signal_to_redis(signal_data)
                
        except Exception as e:
            logger.error(f"Failed to collect signal data: {e}")
    
    def _save_signal_to_redis(self, signal_data: dict):
        """Save signal to Redis for persistence"""
        try:
            if self.redis_client:
                # Generate unique key
                key = f"ml:signal:{signal_data['symbol']}:{signal_data['timestamp']}"
                
                # Store with expiry (keep for 30 days)
                self.redis_client.setex(key, 30*24*3600, json.dumps(signal_data))
                
                # Add to pending signals list
                self.redis_client.lpush('ml:pending_signals', key)
                
                # Trim list to keep only recent 1000 pending signals
                self.redis_client.ltrim('ml:pending_signals', 0, 999)
            else:
                # Fallback to local storage
                if hasattr(self, 'local_storage'):
                    self.local_storage['signals'].append(signal_data)
        except Exception as e:
            logger.error(f"Failed to save signal to Redis: {e}")
    
    def update_signal_outcome(self, symbol: str, entry_time: datetime, outcome: str, pnl_r: float):
        """
        Update the outcome of a historical signal for training
        Called when a position closes
        """
        try:
            updated = False
            
            if self.redis_client:
                # Search in pending signals
                pending_keys = self.redis_client.lrange('ml:pending_signals', 0, -1)
                
                for key in pending_keys:
                    signal_data = self.redis_client.get(key)
                    if signal_data:
                        signal = json.loads(signal_data)
                        if (signal['symbol'] == symbol and 
                            abs((datetime.fromisoformat(signal['timestamp']) - entry_time).total_seconds()) < 300):
                            
                            # Update signal
                            signal['outcome'] = outcome  # 'win', 'loss', 'breakeven'
                            signal['pnl_r'] = pnl_r  # PnL in R-multiples
                            
                            # Move to completed trades
                            self._add_completed_trade(signal)
                            
                            # Remove from pending
                            self.redis_client.lrem('ml:pending_signals', 1, key)
                            
                            updated = True
                            break
            else:
                # Fallback to local storage
                if hasattr(self, 'local_storage'):
                    for signal in reversed(self.local_storage['signals']):
                        if (signal['symbol'] == symbol and 
                            abs((datetime.fromisoformat(signal['timestamp']) - entry_time).total_seconds()) < 300):
                            signal['outcome'] = outcome
                            signal['pnl_r'] = pnl_r
                            self.local_storage['completed_trades'].append(signal)
                            updated = True
                            break
            
            if updated:
                self.completed_trades_count += 1
                
                # Update count in Redis
                if self.redis_client:
                    self.redis_client.set('ml:completed_trades_count', str(self.completed_trades_count))
                
                logger.info(f"Trade outcome updated: {symbol} - {outcome} ({pnl_r:.2f}R). "
                          f"Total completed: {self.completed_trades_count}")
                
                # Check if we should train or retrain
                self._check_ready_for_ml()
                
        except Exception as e:
            logger.error(f"Failed to update signal outcome: {e}")
    
    def _add_completed_trade(self, trade_data: dict):
        """Add a completed trade to Redis storage"""
        try:
            if self.redis_client:
                # Store completed trade
                key = f"ml:completed:{trade_data['symbol']}:{trade_data['timestamp']}"
                self.redis_client.setex(key, 90*24*3600, json.dumps(trade_data))  # Keep for 90 days
                
                # Add to completed list
                self.redis_client.lpush('ml:completed_trades', key)
                
                # Keep only recent 1000 completed trades
                self.redis_client.ltrim('ml:completed_trades', 0, 999)
        except Exception as e:
            logger.error(f"Failed to add completed trade: {e}")
    
    def _get_completed_trades(self) -> List[dict]:
        """Get all completed trades from Redis"""
        trades = []
        
        try:
            if self.redis_client:
                # Get completed trade keys
                trade_keys = self.redis_client.lrange('ml:completed_trades', 0, -1)
                
                for key in trade_keys:
                    trade_data = self.redis_client.get(key)
                    if trade_data:
                        trades.append(json.loads(trade_data))
            else:
                # Fallback to local storage
                if hasattr(self, 'local_storage'):
                    trades = self.local_storage.get('completed_trades', [])
        except Exception as e:
            logger.error(f"Failed to get completed trades: {e}")
        
        return trades

# Singleton instance
_scorer_instance = None

def get_scorer(enabled: bool = True, min_score: float = 70.0) -> MLSignalScorer:
    """Get or create the singleton ML scorer instance"""
    global _scorer_instance
    if _scorer_instance is None:
        _scorer_instance = MLSignalScorer(enabled=enabled, min_score=min_score)
    return _scorer_instance