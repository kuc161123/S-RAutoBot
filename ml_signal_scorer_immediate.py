"""
Enhanced ML Signal Scorer with Immediate Activation
Works from day 1 with continuous learning from both real and phantom trades
Progressively improves as it gathers more data
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
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ImmediateMLScorer:
    """
    ML Scorer that works immediately and learns continuously
    - Starts with theory-based scoring on day 1
    - Learns from every trade outcome (real and phantom)
    - Progressively improves accuracy
    - Adapts threshold based on performance
    """
    
    # Learning parameters
    MIN_TRADES_FOR_ML = 10  # Start using ML models after just 10 trades
    RETRAIN_INTERVAL = 5   # Retrain every 5 new trades for rapid learning
    INITIAL_THRESHOLD = 55  # Start lenient, will adapt based on performance
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.min_score = self.INITIAL_THRESHOLD  # Start lenient
        self.completed_trades = 0
        self.models = {}  # Will hold ensemble models
        self.scaler = StandardScaler()
        self.is_ml_ready = False
        
        # Performance tracking for threshold adaptation
        self.recent_performance = []  # Track last 20 trade outcomes
        
        # Initialize Redis
        self.redis_client = None
        if enabled:
            self._init_redis()
            self._load_state()
            
    def _init_redis(self):
        """Initialize Redis connection"""
        try:
            redis_url = os.getenv('REDIS_URL')
            if redis_url:
                self.redis_client = redis.from_url(redis_url, decode_responses=True)
                self.redis_client.ping()
                logger.info("ImmediateML connected to Redis")
            else:
                logger.warning("No Redis, using memory only")
                self.memory_storage = {'trades': [], 'phantoms': []}
        except Exception as e:
            logger.warning(f"Redis failed: {e}")
            self.redis_client = None
            self.memory_storage = {'trades': [], 'phantoms': []}
    
    def _load_state(self):
        """Load saved state from Redis"""
        if not self.redis_client:
            return
            
        try:
            # Load completed trades count
            count = self.redis_client.get('iml:completed_trades')
            self.completed_trades = int(count) if count else 0
            
            # Load threshold
            threshold = self.redis_client.get('iml:threshold')
            if threshold:
                self.min_score = float(threshold)
                
            # Load recent performance
            perf = self.redis_client.get('iml:recent_performance')
            if perf:
                self.recent_performance = json.loads(perf)
                
            # Load models if exist
            if self.completed_trades >= self.MIN_TRADES_FOR_ML:
                model_data = self.redis_client.get('iml:models')
                if model_data:
                    import base64
                    self.models = pickle.loads(base64.b64decode(model_data))
                    self.is_ml_ready = True
                    logger.info(f"Loaded ML models (trained on {self.completed_trades} trades)")
                    
        except Exception as e:
            logger.error(f"Error loading state: {e}")
    
    def score_signal(self, signal: dict, features: dict) -> Tuple[float, str]:
        """
        Score a signal from 0-100
        Returns (score, reasoning)
        """
        # Start with base score
        score = 50.0
        reasoning = []
        
        # If we have ML models, use them
        if self.is_ml_ready and self.models:
            try:
                # Prepare feature vector
                feature_vector = self._prepare_features(features)
                X = np.array([feature_vector]).reshape(1, -1)
                
                # Scale features
                X_scaled = self.scaler.transform(X)
                
                # Get predictions from ensemble
                predictions = []
                if 'rf' in self.models:
                    pred = self.models['rf'].predict_proba(X_scaled)[0][1]
                    predictions.append(pred)
                    reasoning.append(f"RF: {pred*100:.1f}")
                    
                if 'gb' in self.models:
                    pred = self.models['gb'].predict_proba(X_scaled)[0][1]
                    predictions.append(pred)
                    reasoning.append(f"GB: {pred*100:.1f}")
                    
                if 'nn' in self.models:
                    pred = self.models['nn'].predict_proba(X_scaled)[0][1]
                    predictions.append(pred)
                    reasoning.append(f"NN: {pred*100:.1f}")
                
                # Average predictions
                if predictions:
                    ml_score = np.mean(predictions) * 100
                    score = ml_score
                    reasoning.insert(0, f"ML Ensemble: {score:.1f}")
                    
            except Exception as e:
                logger.warning(f"ML scoring failed, using rules: {e}")
                
        # If no ML or ML failed, use rule-based scoring
        if score == 50.0:
            score, rules_reasoning = self._rule_based_scoring(signal, features)
            reasoning = rules_reasoning
            
        # Apply confidence adjustment based on data availability
        if self.completed_trades < 10:
            # Very early stage - be more conservative
            confidence = 0.7 + (self.completed_trades * 0.03)  # 70% to 100% over 10 trades
            score = score * confidence
            reasoning.append(f"Early-stage confidence: {confidence*100:.0f}%")
        
        return score, " | ".join(reasoning)
    
    def _rule_based_scoring(self, signal: dict, features: dict) -> Tuple[float, List[str]]:
        """
        Theory-based scoring for when ML isn't ready
        Uses market principles to score signals
        """
        score = 50.0  # Neutral start
        reasoning = []
        
        # Volume check (strong volume = +20)
        vol_ratio = features.get('volume_ratio', 1.0)
        if vol_ratio > 1.5:
            score += 20
            reasoning.append("Strong volume")
        elif vol_ratio > 1.2:
            score += 10
            reasoning.append("Good volume")
        elif vol_ratio < 0.8:
            score -= 10
            reasoning.append("Weak volume")
            
        # Trend alignment (with trend = +15)
        trend_strength = features.get('trend_strength', 50)
        if trend_strength > 65:
            score += 15
            reasoning.append("Strong trend")
        elif trend_strength > 55:
            score += 7
            reasoning.append("Moderate trend")
        elif trend_strength < 35:
            score -= 10
            reasoning.append("Against trend")
            
        # Support/Resistance strength (+15 for strong levels)
        sr_strength = features.get('support_resistance_strength', 0)
        if sr_strength >= 3:
            score += 15
            reasoning.append("Strong S/R")
        elif sr_strength >= 2:
            score += 7
            reasoning.append("Moderate S/R")
            
        # Volatility regime (-10 for extreme volatility)
        volatility = features.get('volatility_regime', 'normal')
        if volatility == 'high':
            score -= 10
            reasoning.append("High volatility")
        elif volatility == 'low':
            score += 5
            reasoning.append("Low volatility")
            
        # Time of day bonus
        hour = features.get('hour_of_day', 12)
        if 14 <= hour <= 20:  # US market hours
            score += 5
            reasoning.append("Prime hours")
        elif 2 <= hour <= 6:  # Dead zone
            score -= 5
            reasoning.append("Off hours")
            
        # Risk/Reward check
        rr = features.get('risk_reward_ratio', 2.0)
        if rr >= 3.0:
            score += 10
            reasoning.append(f"R:R {rr:.1f}")
        elif rr >= 2.0:
            score += 5
            reasoning.append(f"R:R {rr:.1f}")
        elif rr < 1.5:
            score -= 15
            reasoning.append(f"Poor R:R {rr:.1f}")
            
        # Ensure score stays in bounds
        score = max(0, min(100, score))
        
        reasoning.insert(0, f"Rule-based: {score:.1f}")
        return score, reasoning
    
    def _prepare_features(self, features: dict) -> list:
        """Convert feature dict to vector for ML"""
        # Standard feature order
        feature_order = [
            'trend_strength', 'higher_tf_alignment', 'ema_distance_ratio',
            'volume_ratio', 'volume_trend', 'breakout_volume',
            'support_resistance_strength', 'pullback_depth', 'confirmation_candle_strength',
            'atr_percentile', 'risk_reward_ratio', 'atr_stop_distance',
            'hour_of_day', 'day_of_week', 'candle_body_ratio', 'upper_wick_ratio',
            'lower_wick_ratio', 'candle_range_atr', 'volume_ma_ratio',
            'rsi', 'bb_position', 'volume_percentile'
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
    
    def record_outcome(self, signal_data: dict, outcome: str, pnl_percent: float):
        """
        Record trade outcome for learning
        
        Args:
            signal_data: Dict with features, score, symbol, etc
            outcome: 'win' or 'loss'
            pnl_percent: Actual P&L percentage
        """
        self.completed_trades += 1
        
        # Track recent performance for threshold adaptation
        self.recent_performance.append(1 if outcome == 'win' else 0)
        if len(self.recent_performance) > 20:
            self.recent_performance.pop(0)
            
        # Adapt threshold based on performance
        self._adapt_threshold()
        
        # Store trade data
        trade_record = {
            'features': signal_data['features'],
            'score': signal_data['score'],
            'outcome': outcome,
            'pnl_percent': pnl_percent,
            'timestamp': datetime.now().isoformat()
        }
        
        self._store_trade(trade_record)
        
        # Retrain if interval reached
        if self.completed_trades >= self.MIN_TRADES_FOR_ML:
            if self.completed_trades % self.RETRAIN_INTERVAL == 0:
                self._retrain_models()
                
    def _adapt_threshold(self):
        """Adapt scoring threshold based on recent performance"""
        if len(self.recent_performance) >= 10:
            win_rate = sum(self.recent_performance) / len(self.recent_performance)
            
            # If winning too much (>70%), raise threshold to be more selective
            if win_rate > 0.70:
                self.min_score = min(75, self.min_score + 2)
                logger.info(f"Raising threshold to {self.min_score} (WR: {win_rate*100:.1f}%)")
                
            # If losing too much (<30%), lower threshold to be less selective
            elif win_rate < 0.30:
                self.min_score = max(45, self.min_score - 2)
                logger.info(f"Lowering threshold to {self.min_score} (WR: {win_rate*100:.1f}%)")
                
            # Save new threshold
            if self.redis_client:
                self.redis_client.set('iml:threshold', str(self.min_score))
    
    def _store_trade(self, trade_record: dict):
        """Store trade record for training"""
        try:
            if self.redis_client:
                # Add to list in Redis
                self.redis_client.rpush('iml:trades', json.dumps(trade_record))
                # Keep only last 1000 trades
                self.redis_client.ltrim('iml:trades', -1000, -1)
                # Update count
                self.redis_client.set('iml:completed_trades', str(self.completed_trades))
                # Update recent performance
                self.redis_client.set('iml:recent_performance', json.dumps(self.recent_performance))
            else:
                # Use memory storage
                if 'trades' not in self.memory_storage:
                    self.memory_storage['trades'] = []
                self.memory_storage['trades'].append(trade_record)
                # Keep only last 1000
                self.memory_storage['trades'] = self.memory_storage['trades'][-1000:]
                
        except Exception as e:
            logger.error(f"Error storing trade: {e}")
    
    def _retrain_models(self):
        """Retrain ML models with available data"""
        try:
            logger.info(f"Retraining ML models with {self.completed_trades} trades...")
            
            # Get trade data
            trades = []
            if self.redis_client:
                trade_data = self.redis_client.lrange('iml:trades', 0, -1)
                trades = [json.loads(t) for t in trade_data]
            else:
                trades = self.memory_storage.get('trades', [])
                
            if len(trades) < self.MIN_TRADES_FOR_ML:
                return
                
            # Prepare training data
            X = []
            y = []
            
            for trade in trades:
                features = trade['features']
                outcome = 1 if trade.get('outcome') == 'win' else 0
                
                feature_vector = self._prepare_features(features)
                X.append(feature_vector)
                y.append(outcome)
                
            X = np.array(X)
            y = np.array(y)
            
            # Fit scaler
            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)
            
            # Train ensemble models
            self.models = {}
            
            # Random Forest
            rf = RandomForestClassifier(
                n_estimators=50,
                max_depth=5,
                min_samples_split=5,
                random_state=42
            )
            rf.fit(X_scaled, y)
            self.models['rf'] = rf
            
            # Gradient Boosting
            gb = GradientBoostingClassifier(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.1,
                random_state=42
            )
            gb.fit(X_scaled, y)
            self.models['gb'] = gb
            
            # Neural Network
            if len(trades) >= 30:  # Need more data for NN
                nn = MLPClassifier(
                    hidden_layer_sizes=(10, 5),
                    activation='relu',
                    learning_rate_init=0.001,
                    max_iter=500,
                    random_state=42
                )
                nn.fit(X_scaled, y)
                self.models['nn'] = nn
            
            self.is_ml_ready = True
            
            # Save models
            if self.redis_client:
                import base64
                model_data = base64.b64encode(pickle.dumps(self.models)).decode('ascii')
                self.redis_client.set('iml:models', model_data)
                
            logger.info(f"ML models trained successfully! Win rate: {np.mean(y)*100:.1f}%")
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            
    def get_stats(self) -> dict:
        """Get current ML statistics"""
        win_rate = 0
        if self.recent_performance:
            win_rate = sum(self.recent_performance) / len(self.recent_performance) * 100
            
        return {
            'enabled': self.enabled,
            'completed_trades': self.completed_trades,
            'is_ml_ready': self.is_ml_ready,
            'current_threshold': self.min_score,
            'recent_win_rate': win_rate,
            'models_active': list(self.models.keys()) if self.models else [],
            'status': 'ML Active' if self.is_ml_ready else f'Learning ({self.completed_trades}/{self.MIN_TRADES_FOR_ML})'
        }

# Global instance
_immediate_scorer = None

def get_immediate_scorer() -> ImmediateMLScorer:
    """Get or create the global immediate ML scorer"""
    global _immediate_scorer
    if _immediate_scorer is None:
        _immediate_scorer = ImmediateMLScorer()
    return _immediate_scorer