"""
Enhanced ML Signal Scorer with Immediate Activation
Works from day 1 with continuous learning from both real and phantom trades
Progressively improves as it gathers more data

IMPORTANT: This scorer ALWAYS uses only the original 22 features for stability.
The enhanced 48-feature set is reserved for the ML Evolution system in shadow mode.
This separation ensures the live trading system remains stable while evolution learns.
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
    RETRAIN_INTERVAL = 100  # Retrain every 100 combined trades for stability
    INITIAL_THRESHOLD = 70  # Start at 70, minimum threshold for quality control
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.min_score = self.INITIAL_THRESHOLD  # Start lenient
        self.completed_trades = 0
        self.last_train_count = 0  # Track total trades at last training
        self.models = {}  # Will hold ensemble models
        self.scaler = StandardScaler()
        self.is_ml_ready = False
        
        # Performance tracking for threshold adaptation
        self.recent_performance = []  # Track last 20 trade outcomes
        
        # Data cache for enhanced features (used by evolution system)
        self.last_data_cache = {}  # symbol -> {df, btc_price}
        
        # Track which feature set the models were trained with
        self.model_feature_version = 'original'  # 'original' or 'enhanced'
        self.feature_count = 31  # 22 original + 5 basic cluster + 4 enhanced cluster features
        
        # Flag to force retrain
        self.force_retrain = False
        
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
            
            # Load last train count
            last_train = self.redis_client.get('iml:last_train_count')
            self.last_train_count = int(last_train) if last_train else 0
            
            # Load threshold
            threshold = self.redis_client.get('iml:threshold')
            if threshold:
                self.min_score = max(70.0, float(threshold))  # Enforce minimum of 70
                
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
                    
                # Also load scaler if exists
                scaler_data = self.redis_client.get('iml:scaler')
                if scaler_data:
                    self.scaler = pickle.loads(base64.b64decode(scaler_data))
                    logger.info("Loaded fitted scaler")
                
                # Load feature version info
                feature_version = self.redis_client.get('iml:feature_version')
                if feature_version:
                    self.model_feature_version = feature_version
                    self.feature_count = 48 if feature_version == 'enhanced' else 22
                    logger.info(f"Models use {self.model_feature_version} features ({self.feature_count} total)")
                    
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
                
                # Check if feature count matches what models expect
                if hasattr(self.scaler, 'n_features_in_') and self.scaler.n_features_in_ != len(feature_vector):
                    logger.warning(f"Feature count mismatch: models expect {self.scaler.n_features_in_}, got {len(feature_vector)}")
                    logger.warning("Forcing retrain on next trade completion")
                    self.force_retrain = True
                    raise ValueError("Feature count mismatch - need to retrain")
                
                # Scale features - check if scaler is fitted
                if hasattr(self.scaler, 'mean_') and self.scaler.mean_ is not None:
                    X_scaled = self.scaler.transform(X)
                else:
                    # If scaler not fitted, use unscaled features
                    logger.warning("Scaler not fitted, using unscaled features")
                    X_scaled = X
                
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
        """Convert feature dict to vector for ML - includes cluster features"""
        # Core original features
        original_features = [
            'trend_strength', 'higher_tf_alignment', 'ema_distance_ratio',
            'volume_ratio', 'volume_trend', 'breakout_volume',
            'support_resistance_strength', 'pullback_depth', 'confirmation_candle_strength',
            'atr_percentile', 'risk_reward_ratio', 'atr_stop_distance',
            'hour_of_day', 'day_of_week', 'candle_body_ratio', 'upper_wick_ratio',
            'lower_wick_ratio', 'candle_range_atr', 'volume_ma_ratio',
            'rsi', 'bb_position', 'volume_percentile'
        ]
        
        # Basic cluster features
        basic_cluster_features = [
            'symbol_cluster', 'cluster_volatility_norm', 'cluster_volume_norm',
            'btc_correlation_bucket', 'price_tier'
        ]
        
        # Enhanced cluster features (added by cluster_feature_enhancer)
        enhanced_cluster_features = [
            'cluster_confidence', 'cluster_secondary', 'cluster_mixed', 'cluster_conf_ratio'
        ]
        
        # Use all features including enhanced clusters
        feature_order = original_features + basic_cluster_features + enhanced_cluster_features
        
        vector = []
        for feat in feature_order:
            if feat in features:
                val = features[feat]
                # Convert categorical to numeric
                if feat == 'volatility_regime':
                    val = {'low': 0, 'normal': 1, 'high': 2}.get(val, 1)
                elif feat == 'session':
                    val = {'asian': 0, 'european': 1, 'us': 2, 'off_hours': 3}.get(val, 3)
                elif isinstance(val, bool):
                    val = 1.0 if val else 0.0
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
        
        # Check if we should retrain based on combined trades
        if self.completed_trades >= self.MIN_TRADES_FOR_ML:
            # Get total combined trade count (executed + phantom)
            total_combined = self.completed_trades
            try:
                from phantom_trade_tracker import get_phantom_tracker
                phantom_tracker = get_phantom_tracker()
                phantom_count = sum(len(trades) for trades in phantom_tracker.phantom_trades.values())
                total_combined = self.completed_trades + phantom_count
            except:
                pass  # Use executed trades only if phantom tracker not available
            
            # Retrain if we've had RETRAIN_INTERVAL new trades since last training OR force retrain
            if (total_combined - self.last_train_count >= self.RETRAIN_INTERVAL) or self.force_retrain:
                if self.force_retrain:
                    logger.info(f"Force retraining ML models to reset feature expectations")
                    self.force_retrain = False
                else:
                    logger.info(f"Retraining triggered: {total_combined - self.last_train_count} new trades since last training")
                self._retrain_models()
                self.last_train_count = total_combined
                
    def _adapt_threshold(self):
        """Adapt scoring threshold based on recent performance"""
        if len(self.recent_performance) >= 10:
            win_rate = sum(self.recent_performance) / len(self.recent_performance)
            
            # If winning too much (>70%), raise threshold to be more selective
            if win_rate > 0.70:
                self.min_score = min(85, self.min_score + 2)  # Can go up to 85
                logger.info(f"Raising threshold to {self.min_score} (WR: {win_rate*100:.1f}%)")
                
            # If losing too much (<30%), lower threshold but never below 70
            elif win_rate < 0.30:
                self.min_score = max(70, self.min_score - 2)  # Never go below 70
                logger.info(f"Threshold would lower but keeping at {self.min_score} minimum (WR: {win_rate*100:.1f}%)")
                
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
                # Save last train count when it changes
                if hasattr(self, 'last_train_count'):
                    self.redis_client.set('iml:last_train_count', str(self.last_train_count))
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
            
            # Get executed trade data
            trades = []
            if self.redis_client:
                trade_data = self.redis_client.lrange('iml:trades', 0, -1)
                trades = [json.loads(t) for t in trade_data]
            else:
                trades = self.memory_storage.get('trades', [])
            
            # Get phantom trade data
            phantom_data = []
            try:
                from phantom_trade_tracker import get_phantom_tracker
                phantom_tracker = get_phantom_tracker()
                phantom_data = phantom_tracker.get_learning_data()
                logger.info(f"Including {len(phantom_data)} phantom trades in training")
            except Exception as e:
                logger.warning(f"Could not get phantom data: {e}")
            
            # Combine all data
            all_training_data = []
            
            # Add executed trades
            for trade in trades:
                all_training_data.append({
                    'features': trade['features'],
                    'outcome': 1 if trade.get('outcome') == 'win' else 0,
                    'was_executed': True
                })
            
            # Add phantom trades
            for phantom in phantom_data:
                all_training_data.append({
                    'features': phantom['features'],
                    'outcome': phantom['outcome'],
                    'was_executed': phantom['was_executed']
                })
            
            total_data = len(all_training_data)
            if total_data < self.MIN_TRADES_FOR_ML:
                logger.info(f"Not enough data yet: {total_data}/{self.MIN_TRADES_FOR_ML}")
                return
                
            # ALWAYS use original features for the main ML system
            # This ensures stability and prevents feature mismatches
            self.model_feature_version = 'original'
            self.feature_count = 27
            logger.info("Training original ML with 22 core features (enhanced features reserved for evolution)")
            
            # Prepare training data
            X = []
            y = []
            
            for data in all_training_data:
                features = data['features']
                outcome = data['outcome']
                
                feature_vector = self._prepare_features(features)
                X.append(feature_vector)
                y.append(outcome)
                
            X = np.array(X)
            y = np.array(y)
            
            # Log data composition
            executed_count = sum(1 for d in all_training_data if d['was_executed'])
            phantom_count = total_data - executed_count
            logger.info(f"Training on {total_data} total trades: {executed_count} executed, {phantom_count} phantom")
            
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
            
            # Save models and scaler
            if self.redis_client:
                import base64
                model_data = base64.b64encode(pickle.dumps(self.models)).decode('ascii')
                self.redis_client.set('iml:models', model_data)
                
                # Save fitted scaler
                scaler_data = base64.b64encode(pickle.dumps(self.scaler)).decode('ascii')
                self.redis_client.set('iml:scaler', scaler_data)
                
                # Save feature version info and count
                self.redis_client.set('iml:feature_version', self.model_feature_version)
                self.redis_client.set('iml:feature_count', str(X.shape[1]))
                logger.info(f"Saved models trained with {self.model_feature_version} features (count: {X.shape[1]})")
                
            # Calculate win rates
            overall_wr = np.mean(y) * 100
            executed_wr = np.mean([all_training_data[i]['outcome'] for i in range(len(all_training_data)) if all_training_data[i]['was_executed']])
            phantom_wr = np.mean([all_training_data[i]['outcome'] for i in range(len(all_training_data)) if not all_training_data[i]['was_executed']])
            
            logger.info(f"ML models trained successfully! Overall WR: {overall_wr:.1f}% | Executed: {executed_wr*100:.1f}% | Phantom: {phantom_wr*100:.1f}%")
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            
    def get_stats(self) -> dict:
        """Get current ML statistics"""
        win_rate = 0
        if self.recent_performance:
            win_rate = sum(self.recent_performance) / len(self.recent_performance) * 100
            
        stats = {
            'enabled': self.enabled,
            'completed_trades': self.completed_trades,
            'is_ml_ready': self.is_ml_ready,
            'current_threshold': self.min_score,
            'recent_win_rate': win_rate,
            'models_active': list(self.models.keys()) if self.models else [],
            'status': 'ML Active' if self.is_ml_ready else f'Learning ({self.completed_trades}/{self.MIN_TRADES_FOR_ML})',
            'model_feature_version': self.model_feature_version,
            'feature_count': self.feature_count
        }
        
        # Add patterns if ML is ready
        if self.is_ml_ready:
            patterns = self.get_learned_patterns()
            stats['patterns'] = patterns
            
        return stats
    
    def get_retrain_info(self) -> dict:
        """Get information about next retrain"""
        info = {
            'is_ml_ready': self.is_ml_ready,
            'completed_trades': self.completed_trades,
            'last_train_count': self.last_train_count,
            'phantom_count': 0,
            'total_combined': self.completed_trades,
            'trades_until_next_retrain': 0,
            'next_retrain_at': 0,
            'can_train': False
        }
        
        # Get phantom trade count
        try:
            from phantom_trade_tracker import get_phantom_tracker
            phantom_tracker = get_phantom_tracker()
            phantom_count = sum(len(trades) for trades in phantom_tracker.phantom_trades.values())
            info['phantom_count'] = phantom_count
            info['total_combined'] = self.completed_trades + phantom_count
        except:
            pass
        
        # Calculate retrain info
        if not self.is_ml_ready:
            # Not trained yet
            info['can_train'] = info['total_combined'] >= self.MIN_TRADES_FOR_ML
            info['trades_until_next_retrain'] = max(0, self.MIN_TRADES_FOR_ML - info['total_combined'])
            info['next_retrain_at'] = self.MIN_TRADES_FOR_ML
        else:
            # Already trained, calculate next retrain
            trades_since_last = info['total_combined'] - self.last_train_count
            info['trades_until_next_retrain'] = max(0, self.RETRAIN_INTERVAL - trades_since_last)
            info['next_retrain_at'] = self.last_train_count + self.RETRAIN_INTERVAL
            info['can_train'] = True  # We can always retrain if models exist
        
        return info
    
    def force_retrain_models(self):
        """Force clear and retrain models to reset feature expectations"""
        logger.info("Force clearing ML models and scaler for clean retrain")
        
        # Clear existing models
        self.models = {}
        self.scaler = StandardScaler()
        self.is_ml_ready = False
        self.model_feature_version = 'original'  # Reset to original features
        self.feature_count = 22
        
        # Clear from Redis
        if self.redis_client:
            try:
                self.redis_client.delete('iml:models')
                self.redis_client.delete('iml:scaler')
                self.redis_client.delete('iml:feature_version')
                logger.info("Cleared ML models from Redis")
            except Exception as e:
                logger.error(f"Error clearing Redis: {e}")
        
        # Trigger retrain on next trade
        self.force_retrain = True
        logger.info("ML models cleared. Will retrain with available features on next trade outcome.")
    
    def startup_retrain(self) -> bool:
        """Retrain models on startup with all available data"""
        logger.info("Checking if startup retrain is needed...")
        
        # Check for feature update marker
        feature_version_key = 'iml:feature_calculations_version'
        current_feature_version = 'v2_complete'  # Version with all 22 features properly calculated
        
        stored_version = None
        if self.redis_client:
            stored_version = self.redis_client.get(feature_version_key)
        
        # Force retrain if feature calculations have been updated
        force_due_to_features = False
        if stored_version != current_feature_version:
            logger.info(f"Feature calculations updated from {stored_version} to {current_feature_version} - forcing retrain")
            force_due_to_features = True
        
        # Count available data
        executed_count = 0
        phantom_count = 0
        
        # Get executed trades from Redis/memory
        try:
            if self.redis_client:
                trade_data = self.redis_client.lrange('iml:trades', 0, -1)
                executed_count = len(trade_data)
            else:
                executed_count = len(self.memory_storage.get('trades', []))
        except Exception as e:
            logger.warning(f"Error counting executed trades: {e}")
        
        # Get phantom trades
        try:
            from phantom_trade_tracker import get_phantom_tracker
            phantom_tracker = get_phantom_tracker()
            phantom_count = sum(len(trades) for trades in phantom_tracker.phantom_trades.values())
        except Exception as e:
            logger.warning(f"Error counting phantom trades: {e}")
        
        total_available = executed_count + phantom_count
        logger.info(f"Startup data available: {executed_count} executed, {phantom_count} phantom, {total_available} total")
        
        # Decide if we should retrain
        should_retrain = force_due_to_features
        
        if not self.is_ml_ready and total_available >= self.MIN_TRADES_FOR_ML:
            # No models but enough data - definitely train
            logger.info("No models loaded but sufficient data available - will train")
            should_retrain = True
        elif self.is_ml_ready:
            # Models exist - check if we have significantly more data
            new_trades = total_available - self.last_train_count
            if new_trades >= self.RETRAIN_INTERVAL:
                logger.info(f"Found {new_trades} new trades since last training - will retrain")
                should_retrain = True
            elif not self.models:
                # Models flagged as ready but not actually loaded
                logger.warning("ML marked as ready but no models found - will retrain")
                should_retrain = True
            else:
                logger.info(f"Models are up to date ({new_trades} new trades, need {self.RETRAIN_INTERVAL})")
        else:
            logger.info(f"Insufficient data for training ({total_available} trades, need {self.MIN_TRADES_FOR_ML})")
        
        # Perform retrain if needed
        if should_retrain:
            logger.info("Starting startup model retraining...")
            try:
                # Update counts to match actual data
                self.completed_trades = executed_count
                if self.redis_client:
                    self.redis_client.set('iml:completed_trades', str(self.completed_trades))
                
                # Force retrain with all available data
                self._retrain_models()
                
                # Update last train count to current total
                self.last_train_count = total_available
                if self.redis_client:
                    self.redis_client.set('iml:last_train_count', str(self.last_train_count))
                
                # Save feature version marker
                if self.redis_client:
                    self.redis_client.set(feature_version_key, current_feature_version)
                
                logger.info("✅ Startup retrain completed successfully")
                return True
            except Exception as e:
                logger.error(f"Startup retrain failed: {e}")
                return False
        
        return False
    
    def get_learned_patterns(self) -> dict:
        """Extract learned patterns and insights from trained models"""
        patterns = {
            'feature_importance': {},
            'winning_patterns': [],
            'losing_patterns': [],
            'time_patterns': {},
            'market_conditions': {}
        }
        
        if not self.is_ml_ready or not self.models:
            return patterns
        
        try:
            # Get feature importance from Random Forest
            if 'rf' in self.models:
                feature_names = [
                    'trend_strength', 'higher_tf_alignment', 'ema_distance_ratio',
                    'volume_ratio', 'volume_trend', 'breakout_volume',
                    'support_resistance_strength', 'pullback_depth', 'confirmation_candle_strength',
                    'atr_percentile', 'risk_reward_ratio', 'atr_stop_distance',
                    'hour_of_day', 'day_of_week', 'candle_body_ratio', 'upper_wick_ratio',
                    'lower_wick_ratio', 'candle_range_atr', 'volume_ma_ratio',
                    'rsi', 'bb_position', 'volume_percentile',
                    # Cluster features
                    'symbol_cluster', 'cluster_volatility_norm', 'cluster_volume_norm',
                    'btc_correlation_bucket', 'price_tier',
                    # Enhanced cluster features
                    'cluster_confidence', 'cluster_secondary', 'cluster_mixed', 'cluster_conf_ratio'
                ]
                
                importances = self.models['rf'].feature_importances_
                
                # Handle feature count mismatch gracefully
                if len(importances) != len(feature_names):
                    logger.warning(f"Feature count mismatch in patterns: {len(importances)} importances vs {len(feature_names)} names")
                    # Use only the features we have importances for
                    feature_names = feature_names[:len(importances)]
                
                # Get top 10 most important features
                feature_importance = list(zip(feature_names, importances))
                feature_importance.sort(key=lambda x: x[1], reverse=True)
                
                for feat, imp in feature_importance[:10]:
                    patterns['feature_importance'][feat] = round(imp * 100, 1)
            
            # Analyze recent trades for patterns
            trades = self._get_recent_trades()
            if trades:
                wins = [t for t in trades if t.get('outcome') == 'win']
                losses = [t for t in trades if t.get('outcome') == 'loss']
                
                # Find common patterns in winning trades
                if wins:
                    win_patterns = self._analyze_trade_patterns(wins, 'winning')
                    patterns['winning_patterns'] = win_patterns
                
                # Find common patterns in losing trades
                if losses:
                    loss_patterns = self._analyze_trade_patterns(losses, 'losing')
                    patterns['losing_patterns'] = loss_patterns
                
                # Time-based patterns
                patterns['time_patterns'] = self._analyze_time_patterns(trades)
                
                # Market condition patterns
                patterns['market_conditions'] = self._analyze_market_conditions(trades)
                
        except Exception as e:
            logger.error(f"Error extracting patterns: {e}")
            
        return patterns
    
    def _get_recent_trades(self) -> list:
        """Get recent trade data for analysis"""
        trades = []
        try:
            if self.redis_client:
                trade_data = self.redis_client.lrange('iml:trades', -50, -1)  # Last 50 trades
                trades = [json.loads(t) for t in trade_data]
            else:
                trades = self.memory_storage.get('trades', [])[-50:]
        except Exception as e:
            logger.error(f"Error getting recent trades: {e}")
        return trades
    
    def _analyze_trade_patterns(self, trades: list, trade_type: str) -> list:
        """Analyze common patterns in winning or losing trades"""
        patterns = []
        
        if not trades:
            return patterns
        
        # Aggregate feature values
        feature_sums = {}
        feature_counts = {}
        
        for trade in trades:
            features = trade.get('features', {})
            for feat, val in features.items():
                if isinstance(val, (int, float)):
                    if feat not in feature_sums:
                        feature_sums[feat] = 0
                        feature_counts[feat] = 0
                    feature_sums[feat] += val
                    feature_counts[feat] += 1
        
        # Calculate averages and identify significant patterns
        for feat, sum_val in feature_sums.items():
            if feature_counts[feat] > 0:
                avg = sum_val / feature_counts[feat]
                
                # Identify significant patterns based on feature type
                if feat == 'volume_ratio' and avg > 1.5:
                    patterns.append(f"High volume ({avg:.1f}x average)")
                elif feat == 'trend_strength' and avg > 70:
                    patterns.append(f"Strong trend alignment ({avg:.0f}%)")
                elif feat == 'support_resistance_strength' and avg >= 3:
                    patterns.append(f"Strong S/R levels ({avg:.1f} touches)")
                elif feat == 'risk_reward_ratio':
                    patterns.append(f"Average R:R ratio: {avg:.2f}")
                elif feat == 'rsi':
                    if trade_type == 'winning':
                        if avg < 40:
                            patterns.append(f"Oversold conditions (RSI ~{avg:.0f})")
                        elif avg > 60:
                            patterns.append(f"Overbought conditions (RSI ~{avg:.0f})")
                    else:  # losing
                        if avg < 30 or avg > 70:
                            patterns.append(f"Extreme RSI levels (~{avg:.0f})")
                elif feat == 'hour_of_day':
                    patterns.append(f"Most active hour: {int(avg)}:00 UTC")
        
        # Limit to top 5 patterns
        return patterns[:5]
    
    def _analyze_time_patterns(self, trades: list) -> dict:
        """Analyze time-based trading patterns"""
        time_stats = {
            'best_hours': {},
            'worst_hours': {},
            'best_days': {},
            'session_performance': {}
        }
        
        # Group by hour
        hour_performance = {}
        for trade in trades:
            features = trade.get('features', {})
            hour = int(features.get('hour_of_day', 0))
            outcome = trade.get('outcome')
            
            if hour not in hour_performance:
                hour_performance[hour] = {'wins': 0, 'losses': 0}
            
            if outcome == 'win':
                hour_performance[hour]['wins'] += 1
            else:
                hour_performance[hour]['losses'] += 1
        
        # Calculate win rates by hour
        for hour, stats in hour_performance.items():
            total = stats['wins'] + stats['losses']
            if total >= 3:  # Minimum trades to be significant
                win_rate = (stats['wins'] / total) * 100
                if win_rate >= 60:
                    time_stats['best_hours'][f"{hour}:00 UTC"] = f"{win_rate:.0f}% WR ({total} trades)"
                elif win_rate <= 30:
                    time_stats['worst_hours'][f"{hour}:00 UTC"] = f"{win_rate:.0f}% WR ({total} trades)"
        
        # Trading sessions
        session_map = {
            'Asian': list(range(0, 8)),
            'European': list(range(8, 16)),
            'US': list(range(16, 24))
        }
        
        for session, hours in session_map.items():
            wins = sum(hour_performance.get(h, {}).get('wins', 0) for h in hours)
            losses = sum(hour_performance.get(h, {}).get('losses', 0) for h in hours)
            total = wins + losses
            if total > 0:
                win_rate = (wins / total) * 100
                time_stats['session_performance'][session] = f"{win_rate:.0f}% WR ({total} trades)"
        
        return time_stats
    
    def _analyze_market_conditions(self, trades: list) -> dict:
        """Analyze market condition patterns"""
        conditions = {
            'volatility_impact': {},
            'volume_impact': {},
            'trend_impact': {}
        }
        
        # Volatility analysis
        vol_groups = {'low': [], 'normal': [], 'high': []}
        for trade in trades:
            features = trade.get('features', {})
            vol = features.get('volatility_regime', 'normal')
            outcome = 1 if trade.get('outcome') == 'win' else 0
            if vol in vol_groups:
                vol_groups[vol].append(outcome)
        
        for vol_type, outcomes in vol_groups.items():
            if outcomes:
                win_rate = (sum(outcomes) / len(outcomes)) * 100
                conditions['volatility_impact'][vol_type] = f"{win_rate:.0f}% WR ({len(outcomes)} trades)"
        
        # Volume analysis
        high_vol_trades = []
        low_vol_trades = []
        for trade in trades:
            features = trade.get('features', {})
            vol_ratio = features.get('volume_ratio', 1.0)
            outcome = 1 if trade.get('outcome') == 'win' else 0
            
            if vol_ratio > 1.5:
                high_vol_trades.append(outcome)
            elif vol_ratio < 0.7:
                low_vol_trades.append(outcome)
        
        if high_vol_trades:
            win_rate = (sum(high_vol_trades) / len(high_vol_trades)) * 100
            conditions['volume_impact']['high_volume'] = f"{win_rate:.0f}% WR ({len(high_vol_trades)} trades)"
        
        if low_vol_trades:
            win_rate = (sum(low_vol_trades) / len(low_vol_trades)) * 100
            conditions['volume_impact']['low_volume'] = f"{win_rate:.0f}% WR ({len(low_vol_trades)} trades)"
        
        # Trend analysis
        strong_trend = []
        weak_trend = []
        for trade in trades:
            features = trade.get('features', {})
            trend = features.get('trend_strength', 50)
            outcome = 1 if trade.get('outcome') == 'win' else 0
            
            if trend > 65:
                strong_trend.append(outcome)
            elif trend < 35:
                weak_trend.append(outcome)
        
        if strong_trend:
            win_rate = (sum(strong_trend) / len(strong_trend)) * 100
            conditions['trend_impact']['strong_trend'] = f"{win_rate:.0f}% WR ({len(strong_trend)} trades)"
        
        if weak_trend:
            win_rate = (sum(weak_trend) / len(weak_trend)) * 100
            conditions['trend_impact']['counter_trend'] = f"{win_rate:.0f}% WR ({len(weak_trend)} trades)"
        
        return conditions

# Global instance
_immediate_scorer = None

def get_immediate_scorer() -> ImmediateMLScorer:
    """Get or create the global immediate ML scorer"""
    global _immediate_scorer
    if _immediate_scorer is None:
        _immediate_scorer = ImmediateMLScorer()
    return _immediate_scorer