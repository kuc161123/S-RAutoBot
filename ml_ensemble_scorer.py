"""
Enhanced ML Ensemble Scorer with Multiple Models
Implements ensemble voting, separate long/short models, and market regime detection
Designed as a drop-in replacement for ml_signal_scorer.py
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

# Try to import XGBoost, but make it optional
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("XGBoost not available, using GradientBoosting instead")

# Import base classes from original scorer
from ml_signal_scorer import SignalFeatures, MLSignalScorer

logger = logging.getLogger(__name__)

class MarketRegime:
    """Classify current market regime"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"

class EnsembleMLScorer(MLSignalScorer):
    """
    Advanced ML Scorer with ensemble voting and market regime awareness
    Extends the base MLSignalScorer for backward compatibility
    """
    
    MIN_TRADES_TO_LEARN = 200  # Same as base
    RETRAIN_INTERVAL = 50
    
    def __init__(self, min_score: float = 70.0, enabled: bool = True):
        # Initialize base class
        super().__init__(min_score, enabled)
        
        # Ensemble models
        self.models = {
            'rf': None,  # Random Forest
            'xgb': None,  # XGBoost
            'nn': None,  # Neural Network
        }
        
        # Separate models for long/short
        self.long_models = {
            'rf': None,
            'xgb': None,
            'nn': None,
        }
        self.short_models = {
            'rf': None,
            'xgb': None,
            'nn': None,
        }
        
        # Market regime classifier
        self.regime_classifier = None
        self.regime_models = {}  # Models for each regime
        
        # Scalers for each model type
        self.scalers = {
            'ensemble': StandardScaler(),
            'long': StandardScaler(),
            'short': StandardScaler(),
            'regime': StandardScaler()
        }
        
        # Online learning buffer
        self.online_buffer = []
        self.online_learning_enabled = True
        self.online_update_frequency = 10  # Update after every 10 trades
        
        # Performance tracking
        self.model_performance = {
            'rf': {'correct': 0, 'total': 0},
            'xgb': {'correct': 0, 'total': 0},
            'nn': {'correct': 0, 'total': 0},
            'ensemble': {'correct': 0, 'total': 0}
        }
    
    def detect_market_regime(self, df: pd.DataFrame) -> str:
        """
        Detect current market regime based on price action and volatility
        """
        try:
            if len(df) < 100:
                return MarketRegime.RANGING
            
            # Calculate indicators
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            # Trend detection using linear regression slope
            x = np.arange(len(close[-50:]))
            y = close[-50:]
            slope = np.polyfit(x, y, 1)[0]
            
            # Volatility using ATR
            tr = np.maximum(high[-20:] - low[-20:], 
                          np.abs(high[-20:] - np.roll(close[-20:], 1)),
                          np.abs(low[-20:] - np.roll(close[-20:], 1)))
            atr = np.mean(tr[1:])
            atr_ratio = atr / np.mean(close[-20:])
            
            # Range detection using efficiency ratio
            net_change = abs(close[-1] - close[-20])
            total_change = np.sum(np.abs(np.diff(close[-20:])))
            efficiency_ratio = net_change / total_change if total_change > 0 else 0
            
            # Classify regime
            if atr_ratio > 0.03:  # High volatility (3% ATR)
                return MarketRegime.VOLATILE
            elif efficiency_ratio < 0.3:  # Low efficiency = ranging
                return MarketRegime.RANGING
            elif slope > 0 and efficiency_ratio > 0.5:
                return MarketRegime.TRENDING_UP
            elif slope < 0 and efficiency_ratio > 0.5:
                return MarketRegime.TRENDING_DOWN
            else:
                return MarketRegime.RANGING
                
        except Exception as e:
            logger.error(f"Regime detection failed: {e}")
            return MarketRegime.RANGING
    
    def _create_ensemble_models(self) -> Dict:
        """Create ensemble of different model types"""
        models = {
            'rf': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            )
        }
        
        # Use XGBoost if available, otherwise GradientBoosting
        if XGB_AVAILABLE:
            models['xgb'] = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        else:
            models['gb'] = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            )
        
        models['nn'] = MLPClassifier(
            hidden_layer_sizes=(50, 30, 10),
            activation='relu',
            solver='adam',
            learning_rate='adaptive',
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        
        return models
    
    def _train_ensemble(self, X: np.ndarray, y: np.ndarray, model_set: str = 'ensemble'):
        """Train ensemble of models"""
        try:
            # Determine which models to train
            if model_set == 'long':
                models = self.long_models = self._create_ensemble_models()
                scaler = self.scalers['long']
            elif model_set == 'short':
                models = self.short_models = self._create_ensemble_models()
                scaler = self.scalers['short']
            else:
                models = self.models = self._create_ensemble_models()
                scaler = self.scalers['ensemble']
            
            # Scale features
            X_scaled = scaler.fit_transform(X)
            
            # Train each model
            for model_name, model in models.items():
                try:
                    model.fit(X_scaled, y)
                    
                    # Calculate cross-validation score
                    cv_score = cross_val_score(model, X_scaled, y, cv=3, scoring='accuracy')
                    logger.info(f"{model_set.upper()} {model_name.upper()} CV accuracy: {cv_score.mean():.3f}")
                    
                except Exception as e:
                    logger.error(f"Failed to train {model_name}: {e}")
                    models[model_name] = None
            
            return models
            
        except Exception as e:
            logger.error(f"Ensemble training failed: {e}")
            return None
    
    def _train_market_regime_models(self, trades: List[dict]):
        """Train separate models for each market regime"""
        try:
            # Group trades by regime
            regime_trades = {
                MarketRegime.TRENDING_UP: [],
                MarketRegime.TRENDING_DOWN: [],
                MarketRegime.RANGING: [],
                MarketRegime.VOLATILE: []
            }
            
            for trade in trades:
                if 'market_regime' in trade.get('features', {}):
                    regime = trade['features']['market_regime']
                    regime_trades[regime].append(trade)
            
            # Train model for each regime with sufficient data
            for regime, trades_in_regime in regime_trades.items():
                if len(trades_in_regime) >= 30:  # Need minimum trades
                    logger.info(f"Training model for {regime} regime ({len(trades_in_regime)} trades)")
                    
                    X, y = self._prepare_training_data(trades_in_regime)
                    if len(X) > 20:
                        regime_model = RandomForestClassifier(
                            n_estimators=50,
                            max_depth=8,
                            min_samples_split=5,
                            random_state=42,
                            class_weight='balanced'
                        )
                        
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X)
                        regime_model.fit(X_scaled, y)
                        
                        self.regime_models[regime] = {
                            'model': regime_model,
                            'scaler': scaler,
                            'accuracy': regime_model.score(X_scaled, y)
                        }
                        
                        logger.info(f"{regime} model accuracy: {self.regime_models[regime]['accuracy']:.3f}")
            
        except Exception as e:
            logger.error(f"Regime model training failed: {e}")
    
    def _train_model(self):
        """Enhanced training with ensemble and separate long/short models"""
        try:
            # Get completed trades from Redis
            trades = self._get_completed_trades()
            
            if len(trades) < self.MIN_TRADES_TO_LEARN:
                logger.warning(f"Not enough trades for training: {len(trades)}")
                return
            
            # Separate long and short trades
            long_trades = [t for t in trades if t.get('type') == 'long']
            short_trades = [t for t in trades if t.get('type') == 'short']
            
            logger.info(f"Training ensemble models: {len(trades)} total, {len(long_trades)} long, {len(short_trades)} short")
            
            # Train combined ensemble
            X_all, y_all = self._prepare_training_data(trades)
            if len(X_all) >= 50:
                self._train_ensemble(X_all, y_all, 'ensemble')
            
            # Train long-specific models if enough data
            if len(long_trades) >= 30:
                X_long, y_long = self._prepare_training_data(long_trades)
                self._train_ensemble(X_long, y_long, 'long')
                logger.info(f"Trained long-specific models on {len(X_long)} samples")
            
            # Train short-specific models if enough data
            if len(short_trades) >= 30:
                X_short, y_short = self._prepare_training_data(short_trades)
                self._train_ensemble(X_short, y_short, 'short')
                logger.info(f"Trained short-specific models on {len(X_short)} samples")
            
            # Train market regime models
            self._train_market_regime_models(trades)
            
            # Update training status
            self.is_trained = any(m is not None for m in self.models.values())
            self.last_train_count = self.completed_trades_count
            
            # Save to Redis
            self._save_ensemble_to_redis()
            
            # Log performance metrics
            self._log_ensemble_performance()
            
            logger.info(f"✅ Ensemble training complete. Models active: {sum(1 for m in self.models.values() if m)}/3")
            
        except Exception as e:
            logger.error(f"Ensemble training failed: {e}")
            self.is_trained = False
    
    def _ensemble_predict(self, features: np.ndarray, signal_type: str, market_regime: str) -> Tuple[float, Dict[str, float]]:
        """Get ensemble prediction with individual model scores"""
        predictions = {}
        weights = {}
        
        # Choose appropriate model set
        if signal_type == 'long' and all(m is not None for m in self.long_models.values()):
            models = self.long_models
            scaler = self.scalers['long']
            logger.debug("Using long-specific models")
        elif signal_type == 'short' and all(m is not None for m in self.short_models.values()):
            models = self.short_models
            scaler = self.scalers['short']
            logger.debug("Using short-specific models")
        else:
            models = self.models
            scaler = self.scalers['ensemble']
            logger.debug("Using general ensemble models")
        
        # Check for regime-specific model
        if market_regime in self.regime_models:
            regime_data = self.regime_models[market_regime]
            try:
                regime_scaled = regime_data['scaler'].transform(features.reshape(1, -1))
                regime_prob = regime_data['model'].predict_proba(regime_scaled)[0, 1]
                predictions['regime'] = regime_prob
                weights['regime'] = regime_data['accuracy'] * 1.5  # Give regime models more weight
                logger.debug(f"Regime model ({market_regime}) prediction: {regime_prob:.3f}")
            except:
                pass
        
        # Get predictions from each model
        try:
            features_scaled = scaler.transform(features.reshape(1, -1))
            
            # Random Forest
            if models.get('rf') is not None:
                rf_prob = models['rf'].predict_proba(features_scaled)[0, 1]
                predictions['rf'] = rf_prob
                weights['rf'] = self.model_performance['rf']['correct'] / max(self.model_performance['rf']['total'], 1)
            
            # XGBoost or GradientBoosting
            if models.get('xgb') is not None:
                xgb_prob = models['xgb'].predict_proba(features_scaled)[0, 1]
                predictions['xgb'] = xgb_prob
                weights['xgb'] = self.model_performance.get('xgb', {'correct': 0, 'total': 1})['correct'] / max(self.model_performance.get('xgb', {'total': 1})['total'], 1)
            elif models.get('gb') is not None:
                gb_prob = models['gb'].predict_proba(features_scaled)[0, 1]
                predictions['gb'] = gb_prob
                weights['gb'] = self.model_performance.get('gb', {'correct': 0, 'total': 1})['correct'] / max(self.model_performance.get('gb', {'total': 1})['total'], 1)
            
            # Neural Network
            if models.get('nn') is not None:
                nn_prob = models['nn'].predict_proba(features_scaled)[0, 1]
                predictions['nn'] = nn_prob
                weights['nn'] = self.model_performance['nn']['correct'] / max(self.model_performance['nn']['total'], 1)
        
        except Exception as e:
            logger.error(f"Ensemble prediction failed: {e}")
        
        # Calculate weighted ensemble score
        if predictions:
            # Normalize weights
            if sum(weights.values()) > 0:
                total_weight = sum(weights.values())
                weights = {k: v/total_weight for k, v in weights.items()}
            else:
                # Equal weights if no performance data
                weights = {k: 1/len(predictions) for k in predictions.keys()}
            
            # Weighted average
            ensemble_score = sum(predictions[k] * weights.get(k, 1/len(predictions)) 
                               for k in predictions.keys())
            
            return ensemble_score, predictions
        else:
            return 0.5, {}  # Neutral if no models available
    
    def score_signal(self, features: Optional[SignalFeatures]) -> Tuple[float, str]:
        """
        Enhanced scoring using ensemble voting
        Returns (score, reason)
        """
        # Safety fallback
        if not self.enabled or features is None:
            return 75.0, "ML scoring disabled or unavailable"
        
        # Check if still collecting initial data
        if not self.is_trained:
            trades_needed = self.MIN_TRADES_TO_LEARN - self.completed_trades_count
            if trades_needed > 0:
                # Still use rule-based scoring during collection
                return self._rule_based_scoring(features)
        
        # Detect market regime
        market_regime = features.volatility_regime  # Simple regime from features
        
        # If models are trained, use ensemble
        if any(m is not None for m in self.models.values()):
            try:
                # Convert features to array
                feature_array = self._features_to_array(features)
                
                # Determine signal type from features (simplified)
                signal_type = 'long' if features.trend_strength > 50 else 'short'
                
                # Get ensemble prediction
                ensemble_score, individual_scores = self._ensemble_predict(
                    feature_array, signal_type, market_regime
                )
                
                # Convert to 0-100 scale
                score = float(ensemble_score * 100)
                
                # Build detailed reason
                model_details = ", ".join([f"{k}:{v*100:.0f}%" for k, v in individual_scores.items()])
                
                if score > 80:
                    reason = f"Ensemble HIGH confidence ({score:.0f}%) [{model_details}]"
                elif score > 60:
                    reason = f"Ensemble MODERATE ({score:.0f}%) [{model_details}]"
                else:
                    reason = f"Ensemble LOW ({score:.0f}%) [{model_details}]"
                
                # Update for online learning
                if self.online_learning_enabled:
                    self.online_buffer.append({
                        'features': feature_array,
                        'prediction': ensemble_score,
                        'timestamp': datetime.now()
                    })
                
                return score, reason
                
            except Exception as e:
                logger.error(f"Ensemble scoring failed: {e}")
                return 75.0, "Ensemble error, using default"
        
        # Fallback to rule-based
        return self._rule_based_scoring(features)
    
    def _online_learning_update(self):
        """Perform online learning update with recent trades"""
        try:
            if len(self.online_buffer) < self.online_update_frequency:
                return
            
            # Get recent outcomes
            recent_trades = self.online_buffer[-self.online_update_frequency:]
            
            # Prepare mini-batch for update
            X = np.array([t['features'] for t in recent_trades])
            
            # Get actual outcomes (would need to match with closed positions)
            # For now, this is a placeholder - would need integration with position tracking
            
            # Partial fit for online learning (if models support it)
            # This is where incremental learning would happen
            
            logger.info(f"Online learning update with {len(recent_trades)} recent trades")
            
            # Clear buffer
            self.online_buffer = []
            
        except Exception as e:
            logger.error(f"Online learning update failed: {e}")
    
    def _save_ensemble_to_redis(self):
        """Save ensemble models to Redis"""
        try:
            if self.redis_client:
                ensemble_data = {
                    'models': self.models,
                    'long_models': self.long_models,
                    'short_models': self.short_models,
                    'regime_models': self.regime_models,
                    'scalers': self.scalers,
                    'model_performance': self.model_performance,
                    'train_count': self.completed_trades_count,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Serialize with pickle
                model_bytes = pickle.dumps(ensemble_data).decode('latin-1')
                self.redis_client.set('ml:ensemble_model', model_bytes)
                self.redis_client.set('ml:last_train_count', str(self.last_train_count))
                
                logger.info("Ensemble models saved to Redis")
                
        except Exception as e:
            logger.error(f"Failed to save ensemble to Redis: {e}")
    
    def _load_from_redis(self):
        """Load ensemble models from Redis"""
        try:
            if self.redis_client:
                # Load base class data first
                super()._load_from_redis()
                
                # Load ensemble data
                ensemble_data = self.redis_client.get('ml:ensemble_model')
                if ensemble_data:
                    data = pickle.loads(ensemble_data.encode('latin-1'))
                    self.models = data.get('models', self.models)
                    self.long_models = data.get('long_models', self.long_models)
                    self.short_models = data.get('short_models', self.short_models)
                    self.regime_models = data.get('regime_models', self.regime_models)
                    self.scalers = data.get('scalers', self.scalers)
                    self.model_performance = data.get('model_performance', self.model_performance)
                    
                    # Check if any models are trained
                    self.is_trained = any(m is not None for m in self.models.values())
                    
                    logger.info(f"Loaded ensemble models from Redis")
                    logger.info(f"Active models - General: {sum(1 for m in self.models.values() if m)}, "
                              f"Long: {sum(1 for m in self.long_models.values() if m)}, "
                              f"Short: {sum(1 for m in self.short_models.values() if m)}")
                    
        except Exception as e:
            logger.error(f"Failed to load ensemble from Redis: {e}")
    
    def _log_ensemble_performance(self):
        """Log detailed performance metrics for ensemble"""
        try:
            logger.info("=" * 50)
            logger.info("ENSEMBLE PERFORMANCE METRICS")
            logger.info("=" * 50)
            
            for model_name, perf in self.model_performance.items():
                if perf['total'] > 0:
                    accuracy = perf['correct'] / perf['total'] * 100
                    logger.info(f"{model_name.upper()}: {accuracy:.1f}% ({perf['correct']}/{perf['total']})")
            
            # Log feature importance from RF
            if self.models.get('rf') is not None and hasattr(self.models['rf'], 'feature_importances_'):
                self._log_feature_importance()
            
            logger.info("=" * 50)
            
        except Exception as e:
            logger.error(f"Failed to log performance: {e}")
    
    def update_signal_outcome(self, symbol: str, entry_time: datetime, outcome: str, pnl_r: float):
        """
        Enhanced outcome tracking with model performance updates
        """
        # Call parent method first
        super().update_signal_outcome(symbol, entry_time, outcome, pnl_r)
        
        try:
            # Update individual model performance
            # This would need the original predictions stored
            # For now, update ensemble performance
            if outcome in ['win', 'breakeven']:
                self.model_performance['ensemble']['correct'] += 1
            self.model_performance['ensemble']['total'] += 1
            
            # Trigger online learning if buffer is full
            if self.online_learning_enabled and len(self.online_buffer) >= self.online_update_frequency:
                self._online_learning_update()
                
        except Exception as e:
            logger.error(f"Failed to update ensemble performance: {e}")

# Singleton instance
_ml_scorer_instance = None

# Factory function for backward compatibility
def get_ensemble_scorer(enabled: bool = True, min_score: float = 70.0, force_reset: bool = False) -> EnsembleMLScorer:
    """Get or create the singleton ensemble scorer instance"""
    global _ml_scorer_instance
    
    # Only reset if explicitly requested
    if force_reset:
        _ml_scorer_instance = None
        logger.info("ML Scorer force reset requested - clearing all data")
    
    # Create instance if doesn't exist
    if _ml_scorer_instance is None:
        _ml_scorer_instance = EnsembleMLScorer(min_score=min_score, enabled=enabled)
        logger.info(f"ML Scorer created - starting with {_ml_scorer_instance.completed_trades_count} completed trades")
    
    return _ml_scorer_instance