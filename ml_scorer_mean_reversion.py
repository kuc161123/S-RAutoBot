"""
ML Scorer for Mean Reversion Strategy

This is a dedicated ensemble ML scorer for the Mean Reversion strategy,
separate from the Pullback strategy's scorer. It will be trained on
features specific to ranging markets.
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

@dataclass
class MeanReversionFeatures:
    """Features specific to Mean Reversion strategy"""
    # Range Characteristics
    range_width_atr: float
    time_in_range_candles: float
    touch_count_sr: float

    # Reversal Strength
    reversal_candle_size_atr: float
    volume_at_reversal_ratio: float

    # Oscillator Confirmation
    rsi_at_edge: float
    stochastic_at_edge: float

    # Mean Reversion Specific
    distance_from_midpoint_atr: float

    # General Context (from existing features, if applicable)
    volatility_regime: str # "low", "normal", "high"
    hour_of_day: int
    day_of_week: int
    session: str
    symbol_cluster: int

class MLScorerMeanReversion:
    """
    Dedicated ML Scorer for the Mean Reversion strategy.
    Uses an ensemble of models.
    """
    MIN_TRADES_FOR_ML = 50  # Start using ML models after 50 trades for this strategy
    RETRAIN_INTERVAL = 25   # Retrain every 25 new trades
    INITIAL_THRESHOLD = 70  # Default score threshold

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.min_score = self.INITIAL_THRESHOLD
        self.completed_trades = 0
        self.last_train_count = 0
        self.models = {}  # Ensemble models
        self.scaler = StandardScaler()
        self.is_ml_ready = False
        self.recent_performance = []

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
                logger.info("MeanReversionML connected to Redis")
            else:
                logger.warning("No Redis, using memory only for MeanReversionML")
                self.memory_storage = {'trades': []}
        except Exception as e:
            logger.warning(f"Redis failed for MeanReversionML: {e}")
            self.redis_client = None
            self.memory_storage = {'trades': []}

    def _load_state(self):
        """Load saved state from Redis"""
        if not self.redis_client:
            return
        try:
            count = self.redis_client.get('ml:completed_trades:mean_reversion')
            self.completed_trades = int(count) if count else 0
            last_train = self.redis_client.get('ml:last_train_count:mean_reversion')
            self.last_train_count = int(last_train) if last_train else 0
            threshold = self.redis_client.get('ml:threshold:mean_reversion')
            if threshold:
                self.min_score = float(threshold)
            
            model_data = self.redis_client.get('ml:model:mean_reversion')
            if model_data:
                import base64
                self.models = pickle.loads(base64.b64decode(model_data))
                scaler_data = self.redis_client.get('ml:scaler:mean_reversion')
                self.scaler = pickle.loads(base64.b64decode(scaler_data))
                self.is_ml_ready = True
                logger.info(f"Loaded Mean Reversion ML models (trained on {self.completed_trades} trades)")
        except Exception as e:
            logger.error(f"Error loading Mean Reversion ML state: {e}")

    def _save_state(self):
        """Save current state to Redis"""
        if not self.redis_client:
            return
        try:
            self.redis_client.set('ml:completed_trades:mean_reversion', str(self.completed_trades))
            self.redis_client.set('ml:last_train_count:mean_reversion', str(self.last_train_count))
            self.redis_client.set('ml:threshold:mean_reversion', str(self.min_score))
            if self.is_ml_ready:
                import base64
                self.redis_client.set('ml:model:mean_reversion', base64.b64encode(pickle.dumps(self.models)).decode('ascii'))
                self.redis_client.set('ml:scaler:mean_reversion', base64.b64encode(pickle.dumps(self.scaler)).decode('ascii'))
        except Exception as e:
            logger.error(f"Error saving Mean Reversion ML state: {e}")

    def _prepare_features(self, features: Dict) -> List[float]:
        """Converts feature dictionary to a list for the model."""
        # Define the order of features for consistency
        feature_order = [
            'range_width_atr',
            'time_in_range_candles',
            'touch_count_sr',
            'reversal_candle_size_atr',
            'volume_at_reversal_ratio',
            'rsi_at_edge',
            'stochastic_at_edge',
            'distance_from_midpoint_atr',
            # Categorical features need mapping
            'volatility_regime', # Map to 0, 1, 2
            'hour_of_day',
            'day_of_week',
            'session', # Map to 0, 1, 2, 3
            'symbol_cluster'
        ]
        vector = []
        for feat in feature_order:
            val = features.get(feat)
            if feat == 'volatility_regime':
                val = {"low": 0, "normal": 1, "high": 2}.get(val, 1)
            elif feat == 'session':
                val = {"asian": 0, "european": 1, "us": 2, "off_hours": 3}.get(val, 3)
            vector.append(float(val) if val is not None else 0)
        return vector

    def _calibrate_probability(self, prob: float) -> float:
        """Calibrate raw probability to better spread scores."""
        if prob < 0.5:
            calibrated = prob ** 0.7
        else:
            calibrated = 1 - ((1 - prob) ** 0.7)
        calibrated = calibrated * 1.3
        return min(1.0, calibrated)

    def score_signal(self, features: Dict) -> Tuple[float, str]:
        """Scores a mean reversion signal."""
        if not self.enabled:
            return 75.0, "ML scoring disabled"

        if not self.is_ml_ready:
            return 75.0, "ML not ready, using default"

        try:
            feature_vector = self._prepare_features(features)
            X = np.array([feature_vector]).reshape(1, -1)
            X_scaled = self.scaler.transform(X)

            predictions = []
            for model_name, model in self.models.items():
                if model:
                    prob = model.predict_proba(X_scaled)[0][1]
                    predictions.append(self._calibrate_probability(prob) * 100)
            
            if predictions:
                score = float(np.mean(predictions))
                reason = f"Ensemble Score: {score:.1f}"
                return score, reason
            return 75.0, "No models available"
        except Exception as e:
            logger.error(f"Mean Reversion ML scoring failed: {e}")
            return 75.0, "Scoring error, using default"

    def record_outcome(self, features: Dict, outcome: str):
        """Records a trade outcome for training."""
        self.completed_trades += 1
        # Store features and outcome for retraining
        # (Simplified for now, actual storage would be in Redis list)
        if hasattr(self, 'memory_storage'):
            self.memory_storage['trades'].append({'features': features, 'outcome': 1 if outcome == 'win' else 0})
        self._save_state()

        if self.completed_trades >= self.MIN_TRADES_FOR_ML and \
           (self.completed_trades - self.last_train_count >= self.RETRAIN_INTERVAL):
            self._retrain_models()

    def _retrain_models(self):
        """Retrains the ensemble models."""
        logger.info("Retraining Mean Reversion ML models...")
        # In a real scenario, this would load data from Redis
        training_data = getattr(self, 'memory_storage', {}).get('trades', [])
        if len(training_data) < self.MIN_TRADES_FOR_ML:
            logger.warning("Not enough data for Mean Reversion ML retraining.")
            return

        X = np.array([self._prepare_features(d['features']) for d in training_data])
        y = np.array([d['outcome'] for d in training_data])

        if len(X) == 0 or len(y) == 0:
            logger.warning("No valid data for Mean Reversion ML retraining.")
            return

        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)

        self.models = {
            'rf': RandomForestClassifier(n_estimators=50, random_state=42),
            'gb': GradientBoostingClassifier(n_estimators=50, random_state=42),
            'nn': MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=200, random_state=42)
        }

        for name, model in self.models.items():
            model.fit(X_scaled, y)
        
        self.is_ml_ready = True
        self.last_train_count = self.completed_trades
        self._save_state()
        logger.info("✅ Mean Reversion ML models trained and saved.")

    def get_stats(self) -> Dict:
        """Returns current ML stats."""
        return {
            'enabled': self.enabled,
            'is_ml_ready': self.is_ml_ready,
            'completed_trades': self.completed_trades,
            'min_score': self.min_score,
            'last_train_count': self.last_train_count,
            'models_active': list(self.models.keys()) if self.models else []
        }

_mean_reversion_scorer_instance = None

def get_mean_reversion_scorer(enabled: bool = True) -> MLScorerMeanReversion:
    global _mean_reversion_scorer_instance
    if _mean_reversion_scorer_instance is None:
        _mean_reversion_scorer_instance = MLScorerMeanReversion(enabled=enabled)
    return _mean_reversion_scorer_instance
