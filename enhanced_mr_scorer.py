"""
Enhanced ML Scorer for Mean Reversion Strategy
Multi-model ensemble specialized for ranging market conditions
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
from sklearn.svm import SVC
try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except Exception:
    xgb = None
    _XGB_AVAILABLE = False
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Enhanced MR features removed for simplification - using basic MR features

logger = logging.getLogger(__name__)

class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types"""
    def default(self, obj):
        if isinstance(obj, (np.bool_, np.bool8)):
            return bool(obj)
        elif isinstance(obj, (np.integer, np.int_, np.intc, np.intp,
                            np.int8, np.int16, np.int32, np.int64,
                            np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float_, np.float16,
                            np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

@dataclass
class MRModelPerformance:
    """Track performance of individual models in ensemble"""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    predictions_count: int
    last_updated: datetime

class EnhancedMeanReversionScorer:
    """
    Advanced ML Scorer for Mean Reversion Strategy
    Uses specialized ensemble: RangeDetector + ReversalPredictor + ExitOptimizer + RegimeClassifier
    """

    # Learning parameters optimized for mean reversion
    MIN_TRADES_FOR_ML = 30      # Start ML after 30 MR trades
    RETRAIN_INTERVAL = 50       # Retrain every 50 new trades
    INITIAL_THRESHOLD = 72      # Start at 72% for ranging markets
    MAX_THRESHOLD = 88          # Maximum threshold
    MIN_THRESHOLD = 65          # Minimum threshold

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.min_score = self.INITIAL_THRESHOLD
        self.completed_trades = 0
        self.last_train_count = 0

        # Multi-model ensemble for mean reversion
        self.ensemble_models = {
            'range_detector': None,      # RF: Identifies quality ranging conditions
            'reversal_predictor': None,  # XGBoost: Predicts reversal probability
            'exit_optimizer': None,      # NN: Optimizes exit strategies
            'regime_classifier': None    # SVM: Micro-regime within ranges
        }

        self.scalers = {
            'range_detector': StandardScaler(),
            'reversal_predictor': StandardScaler(),
            'exit_optimizer': StandardScaler(),
            'regime_classifier': StandardScaler()
        }

        self.is_ml_ready = False
        self.model_performance = {}
        self.recent_performance = []  # Track recent outcomes for threshold adaptation

        # Feature importance tracking
        self.feature_importance = {}
        self.prediction_confidence_history = []

        # Enhanced learning from multiple timeframes
        self.timeframe_performance = {'15m': [], '1h': [], '4h': []}

        # Initialize Redis with MR-specific keys
        self.redis_client = None
        if enabled:
            self._init_redis()
            self._load_state()

    def _init_redis(self):
        """Initialize Redis connection with MR-specific namespace"""
        try:
            redis_url = os.getenv('REDIS_URL')
            if redis_url:
                self.redis_client = redis.from_url(redis_url, decode_responses=True)
                self.redis_client.ping()
                logger.info("Enhanced MR ML connected to Redis")
            else:
                logger.warning("No Redis for Enhanced MR ML, using memory only")
                self.memory_storage = {'trades': [], 'phantoms': []}
        except Exception as e:
            logger.warning(f"Redis failed for Enhanced MR ML: {e}")
            self.redis_client = None
            self.memory_storage = {'trades': [], 'phantoms': []}

    def _load_state(self):
        """Load saved state from Redis"""
        if not self.redis_client:
            return

        try:
            # Load completed trades count
            count = self.redis_client.get('enhanced_mr:completed_trades')
            self.completed_trades = int(count) if count else 0

            # Load last train count
            last_train = self.redis_client.get('enhanced_mr:last_train_count')
            self.last_train_count = int(last_train) if last_train else 0

            # Load threshold
            threshold = self.redis_client.get('enhanced_mr:threshold')
            if threshold:
                self.min_score = float(threshold)
                logger.info(f"Loaded Enhanced MR threshold: {self.min_score}")

            # Load recent performance
            perf = self.redis_client.get('enhanced_mr:recent_performance')
            if perf:
                self.recent_performance = json.loads(perf)

            # Load models if exist
            if self.completed_trades >= self.MIN_TRADES_FOR_ML:
                self._load_ensemble_models()

            logger.info(f"Enhanced MR ML state loaded: {self.completed_trades} trades, "
                       f"ML ready: {self.is_ml_ready}")

        except Exception as e:
            logger.error(f"Error loading Enhanced MR ML state: {e}")

    def _load_ensemble_models(self):
        """Load all ensemble models from Redis"""
        try:
            models_data = self.redis_client.get('enhanced_mr:ensemble_models')
            scalers_data = self.redis_client.get('enhanced_mr:scalers')

            if models_data and scalers_data:
                import base64
                self.ensemble_models = pickle.loads(base64.b64decode(models_data))
                self.scalers = pickle.loads(base64.b64decode(scalers_data))

                # Load performance data
                perf_data = self.redis_client.get('enhanced_mr:model_performance')
                if perf_data:
                    perf_dict = json.loads(perf_data)
                    self.model_performance = {
                        name: MRModelPerformance(**data) for name, data in perf_dict.items()
                    }

                self.is_ml_ready = True
                active_models = [name for name, model in self.ensemble_models.items() if model is not None]
                logger.info(f"Loaded Enhanced MR ensemble models: {active_models}")

        except Exception as e:
            logger.error(f"Error loading Enhanced MR ensemble models: {e}")

    def _save_state(self):
        """Save current state to Redis"""
        if not self.redis_client:
            return

        try:
            self.redis_client.set('enhanced_mr:completed_trades', str(self.completed_trades))
            self.redis_client.set('enhanced_mr:last_train_count', str(self.last_train_count))
            self.redis_client.set('enhanced_mr:threshold', str(self.min_score))
            self.redis_client.set('enhanced_mr:recent_performance',
                                json.dumps(self.recent_performance, cls=NumpyJSONEncoder))

            # Save models and scalers
            if self.is_ml_ready:
                import base64
                models_data = base64.b64encode(pickle.dumps(self.ensemble_models)).decode('ascii')
                scalers_data = base64.b64encode(pickle.dumps(self.scalers)).decode('ascii')

                self.redis_client.set('enhanced_mr:ensemble_models', models_data)
                self.redis_client.set('enhanced_mr:scalers', scalers_data)

                # Save performance data
                perf_dict = {name: asdict(perf) for name, perf in self.model_performance.items()}
                self.redis_client.set('enhanced_mr:model_performance',
                                    json.dumps(perf_dict, cls=NumpyJSONEncoder))

        except Exception as e:
            logger.error(f"Error saving Enhanced MR ML state: {e}")

    def score_signal(self, signal_data: dict, features: dict, df: pd.DataFrame = None) -> Tuple[float, str]:
        """
        Score a mean reversion signal using specialized ensemble

        Args:
            signal_data: Signal information (side, entry, sl, tp, meta)
            features: Enhanced MR features from enhanced_mr_features.py
            df: Price data for additional context

        Returns:
            Tuple of (score 0-100, reasoning string)
        """
        if not self.enabled:
            return 75.0, "Enhanced MR ML disabled"

        # Calculate enhanced features if not provided
        if not features or len(features) < 20:
            try:
                # Use basic MR features from signal meta
                features = signal_data.get('meta', {}).get('mr_features', {})
            except Exception as e:
                logger.warning(f"Failed to calculate enhanced features: {e}")
                return self._fallback_theory_score(signal_data, features)

        # Use ML ensemble if ready
        if self.is_ml_ready and any(model is not None for model in self.ensemble_models.values()):
            try:
                return self._ensemble_ml_score(features, signal_data)
            except Exception as e:
                logger.warning(f"Enhanced MR ML scoring failed: {e}")

        # Fallback to enhanced theory-based scoring
        return self._enhanced_theory_score(features, signal_data)

    def _ensemble_ml_score(self, features: dict, signal_data: dict) -> Tuple[float, str]:
        """Score using specialized ML ensemble"""
        feature_vector = self._prepare_feature_vector(features)
        X = np.array([feature_vector]).reshape(1, -1)

        model_predictions = {}
        model_confidences = {}

        # Get predictions from each specialized model
        for model_name, model in self.ensemble_models.items():
            if model is None:
                continue

            try:
                scaler = self.scalers[model_name]
                if hasattr(scaler, 'mean_') and scaler.mean_ is not None:
                    X_scaled = scaler.transform(X)
                else:
                    X_scaled = X

                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(X_scaled)[0][1]  # Probability of positive class
                    confidence = max(prob, 1 - prob)  # Distance from 0.5
                else:
                    # For models without predict_proba, use decision function
                    decision = model.decision_function(X_scaled)[0] if hasattr(model, 'decision_function') else 0.5
                    prob = 1 / (1 + np.exp(-decision))  # Sigmoid transform
                    confidence = abs(decision)

                model_predictions[model_name] = prob
                model_confidences[model_name] = confidence

            except Exception as e:
                logger.warning(f"Error in {model_name} prediction: {e}")
                continue

        if not model_predictions:
            return self._enhanced_theory_score(features, signal_data)

        # Weighted ensemble scoring based on model specialization
        ensemble_score = 0.0
        reasoning_parts = []
        total_weight = 0.0

        # Model-specific weights for mean reversion
        model_weights = {
            'range_detector': 0.30,      # Range quality is crucial
            'reversal_predictor': 0.40,  # Reversal probability most important
            'exit_optimizer': 0.20,      # Exit timing valuable
            'regime_classifier': 0.10    # Regime context helpful
        }

        for model_name, prob in model_predictions.items():
            weight = model_weights.get(model_name, 0.25)

            # Calibrate probability for mean reversion characteristics
            calibrated_prob = self._calibrate_mr_probability(prob, model_name)
            model_score = calibrated_prob * 100

            ensemble_score += model_score * weight
            total_weight += weight

            confidence = model_confidences.get(model_name, 0.5)
            reasoning_parts.append(f"{model_name[:6]}: {model_score:.1f} (conf: {confidence:.2f})")

        # Normalize by total weight
        if total_weight > 0:
            ensemble_score /= total_weight

        # Apply confidence adjustment based on model agreement
        model_scores = [pred * 100 for pred in model_predictions.values()]
        score_std = np.std(model_scores) if len(model_scores) > 1 else 0

        # Lower confidence adjustment if models disagree significantly
        confidence_adj = max(0.85, 1.0 - (score_std / 100))  # Reduce by up to 15% for disagreement
        ensemble_score *= confidence_adj

        # Ensure score stays within bounds
        ensemble_score = max(0, min(100, ensemble_score))

        # Build reasoning string
        reasoning = f"Enhanced MR Ensemble ({len(model_predictions)} models): {ensemble_score:.1f} | " + " | ".join(reasoning_parts[:3])
        if confidence_adj < 0.95:
            reasoning += f" | Agreement: {confidence_adj:.2f}"

        return float(ensemble_score), reasoning

    def _calibrate_mr_probability(self, prob: float, model_name: str) -> float:
        """Calibrate probabilities for mean reversion characteristics"""
        # Different calibration for different model types
        if model_name == 'range_detector':
            # Range detector: More conservative, favor higher probabilities
            return prob ** 0.8
        elif model_name == 'reversal_predictor':
            # Reversal predictor: Apply sigmoid enhancement for extremes
            if prob < 0.3:
                return prob ** 1.2  # Make low probs even lower
            elif prob > 0.7:
                return 1 - ((1 - prob) ** 1.2)  # Make high probs higher
            else:
                return prob
        elif model_name == 'exit_optimizer':
            # Exit optimizer: Linear calibration with slight boost
            return min(1.0, prob * 1.1)
        else:
            # Default calibration
            return prob

    def _enhanced_theory_score(self, features: dict, signal_data: dict) -> Tuple[float, str]:
        """Enhanced theory-based scoring using comprehensive feature set"""
        score = 50.0  # Base score
        reasoning = []

        try:
            # ===== RANGE QUALITY ANALYSIS (30 points) =====
            range_confidence = features.get('range_confidence', 0.5)
            range_width_atr = features.get('range_width_atr', 2.0)
            range_strength = features.get('range_strength', 0.8)

            # Range quality bonus
            if range_confidence >= 0.8:
                score += 15
                reasoning.append(f"Strong range (conf: {range_confidence:.2f})")
            elif range_confidence >= 0.6:
                score += 8
                reasoning.append("Good range confidence")

            # Optimal range width (not too tight, not too wide)
            if 1.5 <= range_width_atr <= 4.0:
                score += 10
                reasoning.append(f"Optimal range width ({range_width_atr:.1f} ATR)")
            elif range_width_atr > 4.0:
                score -= 5  # Too wide, less reliable

            # Range strength (fewer breakout attempts = stronger range)
            if range_strength >= 0.8:
                score += 5
                reasoning.append("Strong range boundaries")

            # ===== REVERSAL SIGNAL STRENGTH (25 points) =====
            rsi_oversold = features.get('rsi_oversold_strength', 0.0)
            rsi_overbought = features.get('rsi_overbought_strength', 0.0)
            williams_r = features.get('williams_r', -50.0)
            stochastic = features.get('stochastic_current', 50.0)

            # RSI extremes
            if rsi_oversold >= 10:  # RSI below 20
                score += 15
                reasoning.append(f"Strong oversold (RSI: {30-rsi_oversold:.0f})")
            elif rsi_oversold >= 5:  # RSI below 25
                score += 8
                reasoning.append("Moderate oversold")

            if rsi_overbought >= 10:  # RSI above 80
                score += 15
                reasoning.append(f"Strong overbought (RSI: {70+rsi_overbought:.0f})")
            elif rsi_overbought >= 5:  # RSI above 75
                score += 8
                reasoning.append("Moderate overbought")

            # Williams %R confirmation
            if williams_r <= -80:  # Very oversold
                score += 5
                reasoning.append("Williams %R oversold")
            elif williams_r >= -20:  # Very overbought
                score += 5
                reasoning.append("Williams %R overbought")

            # Stochastic confirmation
            if stochastic <= 20 or stochastic >= 80:
                score += 5
                reasoning.append("Stochastic extreme")

            # ===== VOLUME AND MOMENTUM (15 points) =====
            volume_ratio = features.get('volume_ratio', 1.0)
            price_momentum_5 = abs(features.get('price_momentum_5', 0.0))
            buy_sell_ratio = features.get('buy_sell_ratio', 0.5)

            # Volume confirmation
            if volume_ratio >= 1.5:
                score += 8
                reasoning.append(f"Strong volume ({volume_ratio:.1f}x)")
            elif volume_ratio >= 1.2:
                score += 4
                reasoning.append("Good volume")

            # Momentum exhaustion (good for mean reversion)
            if price_momentum_5 >= 0.03:  # 3% momentum
                score += 4
                reasoning.append("Momentum exhaustion")

            # Order flow balance
            if abs(buy_sell_ratio - 0.5) >= 0.2:  # Imbalanced flow
                score += 3
                reasoning.append("Order flow imbalance")

            # ===== POSITION AND TIMING (15 points) =====
            range_position = features.get('range_position', 0.5)
            distance_to_upper = features.get('distance_to_upper_atr', 1.0)
            distance_to_lower = features.get('distance_to_lower_atr', 1.0)
            session = features.get('session', 'us')

            # Position within range (closer to boundaries = better)
            if range_position <= 0.2 or range_position >= 0.8:  # Near boundaries
                score += 10
                reasoning.append("Near range boundary")
            elif range_position <= 0.3 or range_position >= 0.7:
                score += 5
                reasoning.append("Good range position")

            # Distance to boundaries
            min_distance = min(distance_to_upper, distance_to_lower)
            if min_distance <= 0.5:  # Very close to boundary
                score += 5
                reasoning.append("Close to S/R")

            # Session timing
            if session in ['european', 'us']:  # Better liquidity
                score += 3
                reasoning.append(f"Good session ({session})")

            # ===== MARKET MICROSTRUCTURE (10 points) =====
            volatility_regime = features.get('volatility_regime', 'normal')
            avg_spread = features.get('avg_spread_atr', 1.0)
            price_clustering = features.get('price_clustering', 0.1)

            # Volatility regime
            if volatility_regime == 'low':
                score += 5  # Mean reversion works better in low vol
                reasoning.append("Low volatility")
            elif volatility_regime == 'high':
                score -= 3  # Reduce confidence in high vol
                reasoning.append("High volatility penalty")

            # Price clustering (support/resistance strength)
            if price_clustering >= 0.15:
                score += 3
                reasoning.append("Strong price clustering")

            # Spread analysis (tighter spreads = better conditions)
            if avg_spread <= 0.8:
                score += 2
                reasoning.append("Tight spreads")

            # ===== RISK-REWARD CONTEXT (5 points) =====
            signal_rr = features.get('signal_risk_reward', 2.0)
            if signal_rr >= 2.5:
                score += 5
                reasoning.append(f"Good R:R ({signal_rr:.1f})")
            elif signal_rr >= 2.0:
                score += 2
                reasoning.append(f"R:R {signal_rr:.1f}")
            elif signal_rr < 1.5:
                score -= 5
                reasoning.append(f"Poor R:R ({signal_rr:.1f})")

            # Apply bounds
            score = max(35, min(95, score))  # Clamp between 35-95

            # Limit reasoning to top factors
            top_reasoning = reasoning[:4]
            reason_str = "; ".join(top_reasoning) if top_reasoning else "Theory-based analysis"

            return float(score), f"Enhanced Theory: {score:.0f} ({reason_str})"

        except Exception as e:
            logger.error(f"Error in enhanced theory scoring: {e}")
            return 75.0, "Theory scoring error - using default"

    def _fallback_theory_score(self, signal_data: dict, features: dict) -> Tuple[float, str]:
        """Fallback scoring when feature calculation fails"""
        # Basic scoring based on signal data
        score = 70.0  # Conservative default for mean reversion

        if signal_data and 'meta' in signal_data:
            meta = signal_data['meta']
            if 'range_upper' in meta and 'range_lower' in meta:
                score += 5  # Bonus for detected range

        return float(score), "Fallback theory score"

    def _prepare_feature_vector(self, features: dict) -> List[float]:
        """Convert feature dictionary to vector for ML models.

        The feature names here must match the output of
        `ml_scorer_mean_reversion.calculate_mean_reversion_features`.
        """

        feature_names = [
            'range_width_atr',         # Width of the detected range in ATR terms
            'touch_count_sr',          # Support/resistance touches in the range
            'volume_at_reversal_ratio',# Volume confirmation at the reversal
            'volatility_regime',       # Low/normal/high
        ]

        vector: List[float] = []

        for name in feature_names:
            val = features.get(name, 0.0)

            if name == 'volatility_regime':
                # Encode categorical volatility regime deterministically
                val = {'low': 0.0, 'normal': 1.0, 'high': 2.0}.get(str(val).lower(), 1.0)
            elif pd.isna(val) or val is None:
                val = 0.0
            else:
                try:
                    val = float(val)
                except (TypeError, ValueError):
                    val = 0.0

            vector.append(val)

        return vector

    def record_outcome(self, signal_data: dict, outcome: str, pnl_percent: float):
        """Record trade outcome for enhanced MR learning.
        Accepts optional signal_data['was_executed'] (default True). Phantom outcomes should
        be stored via MR phantom tracker and NOT increment executed counters here.
        """
        try:
            was_executed = bool(signal_data.get('was_executed', True))
            if was_executed:
                self.completed_trades += 1

            # Track recent performance
            outcome_binary = 1 if outcome == 'win' else 0
            self.recent_performance.append(outcome_binary)
            if len(self.recent_performance) > 30:  # Keep last 30 trades
                self.recent_performance = self.recent_performance[-30:]

            # Adapt threshold based on performance
            self._adapt_mr_threshold()

            # Store enhanced trade record
            trade_record = {
                'features': signal_data.get('features', {}),
                'enhanced_features': signal_data.get('enhanced_features', {}),
                'score': signal_data.get('score', 0),
                'outcome': outcome_binary,
                'pnl_percent': float(pnl_percent),
                'timestamp': datetime.now().isoformat(),
                'symbol': signal_data.get('symbol', 'UNKNOWN'),
                'strategy': 'enhanced_mr',
                'exit_reason': signal_data.get('exit_reason'),
                'was_executed': was_executed
            }
            # Only store executed trades in executed store; phantom data comes from MR phantom tracker
            if was_executed:
                self._store_trade_record(trade_record)

            # Save state
            self._save_state()

            # Check for retraining (including phantom trades)
            total_combined = self._get_total_combined_trades()
            if (total_combined >= self.MIN_TRADES_FOR_ML and
                total_combined - self.last_train_count >= self.RETRAIN_INTERVAL):
                logger.info(f"Enhanced MR ML retrain triggered: {total_combined - self.last_train_count} new trades (executed + phantom)")
                self._retrain_ensemble()

        except Exception as e:
            logger.error(f"Error recording Enhanced MR outcome: {e}")

    def _get_total_combined_trades(self) -> int:
        """Get total trade count including phantom trades from MR phantom tracker"""
        total = self.completed_trades
        try:
            from mr_phantom_tracker import get_mr_phantom_tracker
            mr_phantom_tracker = get_mr_phantom_tracker()
            phantom_count = sum(
                len([p for p in trades if not getattr(p, 'was_executed', False)])
                for trades in mr_phantom_tracker.mr_phantom_trades.values()
            )
            total += phantom_count
        except Exception as e:
            logger.debug(f"Could not get MR phantom trade count: {e}")
        return total

    def get_retrain_info(self) -> dict:
        """Get information about next retrain - includes phantom trades"""
        phantom_count = 0
        try:
            from mr_phantom_tracker import get_mr_phantom_tracker
            mr_phantom_tracker = get_mr_phantom_tracker()
            phantom_count = sum(
                len([p for p in trades if not getattr(p, 'was_executed', False)])
                for trades in mr_phantom_tracker.mr_phantom_trades.values()
            )
        except Exception as e:
            logger.debug(f"Could not get MR phantom trade count: {e}")
        # Compute executed via Redis store length when possible to avoid confusion
        executed_count = self.completed_trades
        try:
            if self.redis_client:
                executed_count = len(self.redis_client.lrange('enhanced_mr:trades', 0, -1))
        except Exception:
            pass
        total_combined = executed_count + phantom_count
        
        info = {
            'is_ml_ready': self.is_ml_ready,
            'completed_trades': executed_count,
            'phantom_count': phantom_count,
            'total_combined': total_combined,
            'last_train_count': self.last_train_count,
            'trades_until_next_retrain': 0,
            'next_retrain_at': 0,
            'can_train': False
        }
        
        # Calculate retrain info
        if not self.is_ml_ready:
            # Not trained yet
            info['can_train'] = total_combined >= self.MIN_TRADES_FOR_ML
            info['trades_until_next_retrain'] = max(0, self.MIN_TRADES_FOR_ML - total_combined)
            info['next_retrain_at'] = self.MIN_TRADES_FOR_ML
        else:
            # Already trained, calculate next retrain
            trades_since_last = total_combined - self.last_train_count
            info['trades_until_next_retrain'] = max(0, self.RETRAIN_INTERVAL - trades_since_last)
            info['next_retrain_at'] = self.last_train_count + self.RETRAIN_INTERVAL
            info['can_train'] = True  # We can always retrain if models exist
        
        return info

    def startup_retrain(self) -> bool:
        """Retrain models on startup with all available data including phantom trades"""
        logger.info("Checking Enhanced MR startup retrain...")
        
        # Count available data
        executed_count = 0
        phantom_count = 0
        
        # Get executed trades
        try:
            if self.redis_client:
                trade_data = self.redis_client.lrange('enhanced_mr:trades', 0, -1)
                executed_count = len(trade_data)
            else:
                executed_count = len(self.memory_storage.get('trades', []))
        except Exception as e:
            logger.warning(f"Error counting Enhanced MR executed trades: {e}")
        
        # Get phantom trades
        try:
            from mr_phantom_tracker import get_mr_phantom_tracker
            mr_phantom_tracker = get_mr_phantom_tracker()
            phantom_count = sum(
                len([p for p in trades if not getattr(p, 'was_executed', False)])
                for trades in mr_phantom_tracker.mr_phantom_trades.values()
            )
        except Exception as e:
            logger.warning(f"Error counting Enhanced MR phantom trades: {e}")
        
        total_available = executed_count + phantom_count
        logger.info(f"Enhanced MR startup data: {executed_count} executed, {phantom_count} phantom, {total_available} total")
        
        # Decide if we should retrain
        should_retrain = False
        
        if not self.is_ml_ready and total_available >= self.MIN_TRADES_FOR_ML:
            logger.info("No Enhanced MR models but sufficient data - will train")
            should_retrain = True
        elif self.is_ml_ready:
            new_trades = total_available - self.last_train_count
            if new_trades >= self.RETRAIN_INTERVAL:
                logger.info(f"Found {new_trades} new Enhanced MR trades since last training - will retrain")
                should_retrain = True
            elif not any(model is not None for model in self.ensemble_models.values()):
                logger.warning("Enhanced MR marked as ready but no models found - will retrain")
                should_retrain = True
            else:
                logger.info(f"Enhanced MR models up to date ({new_trades} new trades, need {self.RETRAIN_INTERVAL})")
        else:
            logger.info(f"Insufficient Enhanced MR data for training ({total_available} trades, need {self.MIN_TRADES_FOR_ML})")
        
        # Perform retrain if needed
        if should_retrain:
            logger.info("Starting Enhanced MR startup model retraining...")
            try:
                # Update counts to match actual data
                self.completed_trades = executed_count
                if self.redis_client:
                    self.redis_client.set('enhanced_mr:completed_trades', str(self.completed_trades))
                
                # Force retrain with all available data (including phantoms)
                self._retrain_ensemble_with_phantoms()
                
                # Update last train count to current total
                self.last_train_count = total_available
                if self.redis_client:
                    self.redis_client.set('enhanced_mr:last_train_count', str(self.last_train_count))
                
                logger.info("✅ Enhanced MR startup retrain completed successfully")
                return True
            except Exception as e:
                logger.error(f"Enhanced MR startup retrain failed: {e}")
                return False
        
        return False

    def _adapt_mr_threshold(self):
        """Adapt threshold based on recent mean reversion performance"""
        if len(self.recent_performance) >= 15:
            win_rate = sum(self.recent_performance) / len(self.recent_performance)

            # Target win rate for mean reversion: 70-75%
            if win_rate > 0.78:  # Too high - raise threshold
                new_threshold = min(self.MAX_THRESHOLD, self.min_score + 2)
                if new_threshold != self.min_score:
                    self.min_score = new_threshold
                    logger.info(f"Enhanced MR threshold raised to {self.min_score} (WR: {win_rate*100:.1f}%)")

            elif win_rate < 0.65:  # Too low - lower threshold
                new_threshold = max(self.MIN_THRESHOLD, self.min_score - 1.5)
                if new_threshold != self.min_score:
                    self.min_score = new_threshold
                    logger.info(f"Enhanced MR threshold lowered to {self.min_score} (WR: {win_rate*100:.1f}%)")

    def _store_trade_record(self, trade_record: dict):
        """Store trade record for training"""
        try:
            if self.redis_client:
                self.redis_client.rpush('enhanced_mr:trades',
                                      json.dumps(trade_record, cls=NumpyJSONEncoder))
                # Keep only last 2000 trades
                self.redis_client.ltrim('enhanced_mr:trades', -2000, -1)
            else:
                if 'trades' not in self.memory_storage:
                    self.memory_storage['trades'] = []
                self.memory_storage['trades'].append(trade_record)
                self.memory_storage['trades'] = self.memory_storage['trades'][-1000:]

        except Exception as e:
            logger.error(f"Error storing Enhanced MR trade record: {e}")

    def _retrain_ensemble(self):
        """Retrain the specialized ensemble models (executed trades only)"""
        self._retrain_ensemble_with_phantoms(include_phantoms=False)

    def _retrain_ensemble_with_phantoms(self, include_phantoms=True):
        """Retrain the specialized ensemble models with optional phantom data"""
        try:
            total_combined = self._get_total_combined_trades() if include_phantoms else self.completed_trades
            logger.info(f"🔄 Retraining Enhanced MR ensemble models... ({total_combined} total trades, phantoms: {include_phantoms})")

            # Load training data
            training_data = self._load_training_data(include_phantoms=include_phantoms)

            # Cap training set size (keep most recent)
            MAX_TRAINING_SAMPLES = 5000
            if len(training_data) > MAX_TRAINING_SAMPLES:
                discarded = len(training_data) - MAX_TRAINING_SAMPLES
                training_data = training_data[-MAX_TRAINING_SAMPLES:]
                logger.info(f"Capped Enhanced MR training set to {MAX_TRAINING_SAMPLES} (discarded {discarded})")

            if len(training_data) < self.MIN_TRADES_FOR_ML:
                logger.warning(f"Not enough Enhanced MR data for training: {len(training_data)} < {self.MIN_TRADES_FOR_ML}")
                return

            # Prepare features and targets
            X_list, y_list = self._prepare_training_data(training_data)

            if len(X_list) < self.MIN_TRADES_FOR_ML:
                logger.warning(f"Not enough valid Enhanced MR features: {len(X_list)}")
                return

            X = np.array(X_list)
            y = np.array(y_list)

            # Log training info
            wins = np.sum(y)
            losses = len(y) - wins
            win_rate = wins / len(y) if len(y) > 0 else 0.0
            
            # Count executed vs phantom trades
            executed_count = sum(1 for trade in training_data if trade.get('was_executed', True))
            phantom_count = len(training_data) - executed_count
            
            logger.info(f"Enhanced MR training: {len(y)} total trades ({executed_count} executed, {phantom_count} phantom)")
            logger.info(f"Enhanced MR results: {wins} wins, {losses} losses ({win_rate:.1%})")

            # Train specialized models
            self._train_specialized_ensemble(X, y)

            self.is_ml_ready = True
            self.last_train_count = total_combined if include_phantoms else self.completed_trades
            self._save_state()

            active_models = [name for name, model in self.ensemble_models.items() if model is not None]
            logger.info(f"✅ Enhanced MR ensemble trained! Active models: {active_models}")

        except Exception as e:
            logger.error(f"Enhanced MR ensemble training failed: {e}")

    def _load_training_data(self, include_phantoms=True) -> List[dict]:
        """Load training data from storage, optionally including phantom trades"""
        training_data = []

        try:
            # Load executed trades
            if self.redis_client:
                trade_data = self.redis_client.lrange('enhanced_mr:trades', 0, -1)
                for trade_json in trade_data:
                    try:
                        trade = json.loads(trade_json)
                        trade['was_executed'] = True  # Mark as executed trade
                        training_data.append(trade)
                    except json.JSONDecodeError:
                        continue
            else:
                executed_trades = self.memory_storage.get('trades', [])
                for trade in executed_trades:
                    trade['was_executed'] = True
                    training_data.append(trade)

            # Load phantom trades if requested
            if include_phantoms:
                try:
                    from mr_phantom_tracker import get_mr_phantom_tracker
                    mr_phantom_tracker = get_mr_phantom_tracker()
                    phantom_data = mr_phantom_tracker.get_mr_learning_data()

                    # Convert phantom data format to match executed trades
                    for phantom in phantom_data:
                        if phantom.get('was_executed'):
                            continue  # executed trades already captured in executed dataset
                        phantom_record = {
                            'enhanced_features': phantom.get('enhanced_features', {}),
                            'features': phantom.get('features', {}),
                            'score': phantom.get('score', 0),
                            'outcome': phantom.get('outcome', 0),
                            'pnl_percent': phantom.get('pnl_percent', 0.0),
                            'timestamp': phantom.get('signal_time', datetime.now().isoformat()),
                            'symbol': phantom.get('symbol', 'UNKNOWN'),
                            'strategy': 'enhanced_mr',
                            'was_executed': phantom.get('was_executed', False)
                        }
                        training_data.append(phantom_record)
                    
                    logger.info(f"Enhanced MR training data: {len(phantom_data)} phantom trades added")
                    
                except Exception as e:
                    logger.warning(f"Could not load Enhanced MR phantom data: {e}")

        except Exception as e:
            logger.error(f"Error loading Enhanced MR training data: {e}")

        return training_data

    def _prepare_training_data(self, training_data: List[dict]) -> Tuple[List, List]:
        """Prepare features and targets for model training"""
        X_list = []
        y_list = []

        for trade in training_data:
            try:
                # Use enhanced features if available, fallback to regular features
                features = trade.get('enhanced_features') or trade.get('features', {})
                outcome = trade.get('outcome', 0)

                feature_vector = self._prepare_feature_vector(features)
                if len(feature_vector) == 4:  # 4 basic MR features
                    X_list.append(feature_vector)
                    y_list.append(outcome)

            except Exception as e:
                logger.debug(f"Skipping invalid Enhanced MR training record: {e}")
                continue

        return X_list, y_list

    def _train_specialized_ensemble(self, X: np.ndarray, y: np.ndarray):
        """Train the specialized ensemble models"""

        # 1. Range Detector (Random Forest) - Identifies quality ranging conditions
        try:
            rf_scaler = StandardScaler()
            X_rf = rf_scaler.fit_transform(X)

            rf_model = RandomForestClassifier(
                n_estimators=150,
                max_depth=10,
                min_samples_split=8,
                min_samples_leaf=4,
                class_weight='balanced',
                random_state=42
            )
            rf_model.fit(X_rf, y)

            self.ensemble_models['range_detector'] = rf_model
            self.scalers['range_detector'] = rf_scaler

            logger.info("✅ Range Detector (RF) trained")

        except Exception as e:
            logger.warning(f"Range Detector training failed: {e}")
            self.ensemble_models['range_detector'] = None

        # 2. Reversal Predictor (XGBoost) - Predicts reversal probability (optional)
        try:
            if _XGB_AVAILABLE:
                xgb_scaler = StandardScaler()
                X_xgb = xgb_scaler.fit_transform(X)

                xgb_model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42
                )
                xgb_model.fit(X_xgb, y)

                self.ensemble_models['reversal_predictor'] = xgb_model
                self.scalers['reversal_predictor'] = xgb_scaler

                logger.info("✅ Reversal Predictor (XGB) trained")
            else:
                self.ensemble_models['reversal_predictor'] = None
                logger.info("ℹ️ XGBoost not available; skipping Reversal Predictor")

        except Exception as e:
            logger.warning(f"Reversal Predictor training failed: {e}")
            self.ensemble_models['reversal_predictor'] = None

        # 3. Exit Optimizer (Neural Network) - Optimizes exit strategies
        try:
            if len(X) >= 50:  # Need more data for NN
                nn_scaler = StandardScaler()
                X_nn = nn_scaler.fit_transform(X)

                nn_model = MLPClassifier(
                    hidden_layer_sizes=(64, 32, 16),
                    activation='relu',
                    learning_rate_init=0.001,
                    max_iter=500,
                    early_stopping=True,
                    validation_fraction=0.2,
                    random_state=42
                )
                nn_model.fit(X_nn, y)

                self.ensemble_models['exit_optimizer'] = nn_model
                self.scalers['exit_optimizer'] = nn_scaler

                logger.info("✅ Exit Optimizer (NN) trained")

            else:
                logger.info("Not enough data for Exit Optimizer (NN)")

        except Exception as e:
            logger.warning(f"Exit Optimizer training failed: {e}")
            self.ensemble_models['exit_optimizer'] = None

        # 4. Regime Classifier (SVM) - Micro-regime within ranges
        try:
            if len(X) >= 40:  # SVM needs reasonable amount of data
                svm_scaler = StandardScaler()
                X_svm = svm_scaler.fit_transform(X)

                svm_model = SVC(
                    kernel='rbf',
                    C=1.0,
                    gamma='scale',
                    probability=True,  # Enable probability estimates
                    random_state=42
                )
                svm_model.fit(X_svm, y)

                self.ensemble_models['regime_classifier'] = svm_model
                self.scalers['regime_classifier'] = svm_scaler

                logger.info("✅ Regime Classifier (SVM) trained")

            else:
                logger.info("Not enough data for Regime Classifier (SVM)")

        except Exception as e:
            logger.warning(f"Regime Classifier training failed: {e}")
            self.ensemble_models['regime_classifier'] = None

    def get_enhanced_stats(self) -> dict:
        """Get comprehensive Enhanced MR statistics"""
        try:
            recent_win_rate = 0.0
            if self.recent_performance:
                recent_win_rate = sum(self.recent_performance) / len(self.recent_performance) * 100

            active_models = [name for name, model in self.ensemble_models.items() if model is not None]

            # Include phantom count for combined progress visibility
            phantom_count = 0
            try:
                from mr_phantom_tracker import get_mr_phantom_tracker
                mr_phantom_tracker = get_mr_phantom_tracker()
                phantom_count = sum(
                    len([p for p in trades if not getattr(p, 'was_executed', False)])
                    for trades in mr_phantom_tracker.mr_phantom_trades.values()
                )
            except Exception:
                pass

            # Compute executed via Redis list length when available
            executed_count = self.completed_trades
            try:
                if self.redis_client:
                    executed_count = len(self.redis_client.lrange('enhanced_mr:trades', 0, -1))
            except Exception:
                pass
            total_combined = executed_count + phantom_count

            # Compute trades_until_retrain consistently with combined counts
            if not self.is_ml_ready:
                trades_until_retrain = max(0, self.MIN_TRADES_FOR_ML - total_combined)
                status_text = f'Learning ({total_combined}/{self.MIN_TRADES_FOR_ML})'
            else:
                trades_since_last = total_combined - self.last_train_count
                trades_until_retrain = max(0, self.RETRAIN_INTERVAL - trades_since_last)
                status_text = 'Advanced ML Active'

            stats = {
                'strategy': 'Enhanced Mean Reversion',
                'enabled': self.enabled,
                'is_ml_ready': self.is_ml_ready,
                'status': status_text,
                'completed_trades': executed_count,
                'phantom_count': phantom_count,
                'total_combined': total_combined,
                'current_threshold': self.min_score,
                'min_threshold': self.MIN_THRESHOLD,
                'max_threshold': self.MAX_THRESHOLD,
                'last_train_count': self.last_train_count,
                'models_active': active_models,
                'model_count': len(active_models),
                'recent_win_rate': recent_win_rate,
                'recent_trades': len(self.recent_performance),
                'retrain_interval': self.RETRAIN_INTERVAL,
                'trades_until_retrain': trades_until_retrain,
                'feature_count': 4,  # Basic MR features
            }

            # Add model performance if available
            if self.model_performance:
                stats['model_performance'] = {
                    name: {
                        'accuracy': perf.accuracy,
                        'predictions': perf.predictions_count
                    } for name, perf in self.model_performance.items()
                }

            return stats

        except Exception as e:
            logger.error(f"Error getting Enhanced MR stats: {e}")
            return {
                'strategy': 'Enhanced Mean Reversion',
                'enabled': self.enabled,
                'error': str(e)
            }

    def get_enhanced_patterns(self) -> dict:
        """Extract learned patterns and insights for Enhanced MR.

        Returns structure compatible with UI renderer used for Pullback patterns:
        {
          'feature_importance': {feat: importance%},
          'time_patterns': {
             'best_hours': {hour: 'WR xx% (N=yy)'},
             'worst_hours': {...},
             'session_performance': {session: 'WR xx% (N=yy)'}
          },
          'market_conditions': { ... },
          'winning_patterns': [str,...],
          'losing_patterns': [str,...]
        }
        """
        patterns = {
            'feature_importance': {},
            'time_patterns': {},
            'market_conditions': {},
            'winning_patterns': [],
            'losing_patterns': []
        }

        try:
            # 1) Feature importance from RangeDetector (RF) if present
            rf = self.ensemble_models.get('range_detector')
            if rf is not None and hasattr(rf, 'feature_importances_'):
                feature_names = [
                    'range_width_atr',
                    'touch_count_sr',
                    'volume_at_reversal_ratio',
                    'volatility_regime'
                ]
                imps = getattr(rf, 'feature_importances_', None)
                if imps is not None and len(imps) > 0:
                    # Guard mismatch
                    if len(imps) != len(feature_names):
                        feature_names = feature_names[:len(imps)]
                    pairs = list(zip(feature_names, imps))
                    pairs.sort(key=lambda x: x[1], reverse=True)
                    for feat, imp in pairs:
                        patterns['feature_importance'][feat] = round(float(imp) * 100, 1)

            # 2) Load recent executed + phantom MR trades for pattern mining
            data = []
            try:
                training_data = self._load_training_data(include_phantoms=True)
                # Keep last 500 by timestamp if available
                def _ts(rec):
                    try:
                        return datetime.fromisoformat(str(rec.get('timestamp')))
                    except Exception:
                        return datetime.utcnow()
                training_data.sort(key=_ts)
                data = training_data[-500:]
            except Exception:
                data = []

            if not data:
                return patterns

            # Normalize helpers
            def _outcome(rec):
                o = rec.get('outcome', 0)
                try:
                    return int(o)
                except Exception:
                    return 1 if str(o).lower() == 'win' else 0

            def _feat(rec, key, default=None):
                f = rec.get('enhanced_features') or rec.get('features') or {}
                return f.get(key, default)

            # 3) Time-based patterns (hours + sessions)
            from collections import defaultdict
            by_hour = defaultdict(lambda: {'w': 0, 'n': 0})
            by_sess = defaultdict(lambda: {'w': 0, 'n': 0})
            for rec in data:
                try:
                    ts = datetime.fromisoformat(str(rec.get('timestamp')))
                except Exception:
                    ts = datetime.utcnow()
                hr = ts.hour
                by_hour[hr]['n'] += 1
                by_hour[hr]['w'] += _outcome(rec)
                # Simple UTC session map
                if 0 <= hr < 8:
                    sess = 'Asia'
                elif 8 <= hr < 14:
                    sess = 'Europe'
                elif 14 <= hr < 22:
                    sess = 'US'
                else:
                    sess = 'Off'
                by_sess[sess]['n'] += 1
                by_sess[sess]['w'] += _outcome(rec)

            def _wr_fmt(w, n):
                wr = (w / n * 100.0) if n > 0 else 0.0
                return f"WR {wr:.0f}% (N={n})"

            # Best/Worst hours by WR with min Ns
            hour_stats = []
            for h, c in by_hour.items():
                if c['n'] >= 3:
                    hour_stats.append((h, c['w'], c['n']))
            hour_stats.sort(key=lambda x: (x[1]/x[2]) if x[2] else 0.0, reverse=True)
            best_hours = {str(h): _wr_fmt(w, n) for h, w, n in hour_stats[:5]}
            worst_hours = {str(h): _wr_fmt(w, n) for h, w, n in hour_stats[-5:][::-1]}

            session_perf = {s: _wr_fmt(c['w'], c['n']) for s, c in by_sess.items() if c['n'] > 0}
            patterns['time_patterns'] = {
                'best_hours': best_hours,
                'worst_hours': worst_hours,
                'session_performance': session_perf
            }

            # 4) Market condition patterns
            # Buckets for key features
            def _bucket_width(x):
                try:
                    x = float(x)
                except Exception:
                    return 'unknown'
                if x < 0.8:
                    return 'narrow(<0.8 ATR)'
                if x < 1.5:
                    return 'medium(0.8-1.5 ATR)'
                return 'wide(>1.5 ATR)'

            def _bucket_volrev(x):
                try:
                    x = float(x)
                except Exception:
                    return 'unknown'
                if x < 0.8:
                    return 'low(<0.8)'
                if x < 1.2:
                    return 'normal(0.8-1.2)'
                return 'high(>1.2)'

            def _vol_reg(x):
                if x is None:
                    return 'unknown'
                s = str(x).lower()
                if s in ('0', 'low'):
                    return 'low'
                if s in ('2', 'high'):
                    return 'high'
                return 'normal'

            from collections import Counter
            cond_counts = {}
            # Volatility regime
            vr = Counter()
            # Width buckets
            wb = Counter()
            # Touches buckets
            tb = Counter()
            # Volume at reversal buckets
            vb = Counter()
            for rec in data:
                o = _outcome(rec)
                ef = rec.get('enhanced_features') or rec.get('features') or {}
                vr[_vol_reg(ef.get('volatility_regime'))] += (o, 1)
                wb[_bucket_width(ef.get('range_width_atr'))] += (o, 1)
                try:
                    t = float(ef.get('touch_count_sr', 0))
                    touch_key = 'touches≥4' if t >= 4 else 'touches≤3'
                except Exception:
                    touch_key = 'touches≤3'
                tb[touch_key] += (o, 1)
                vb[_bucket_volrev(ef.get('volume_at_reversal_ratio'))] += (o, 1)

            def _wr_map(counter_obj):
                out = {}
                for k, (w, n) in counter_obj.items():
                    try:
                        wr = (w / n) * 100.0 if n else 0.0
                        out[k] = f"WR {wr:.0f}% (N={n})"
                    except Exception:
                        out[k] = "WR 0% (N=0)"
                return out

            # Convert Counters of tuples to dicts
            patterns['market_conditions'] = {
                'volatility_regime': _wr_map(vr),
                'range_width_atr': _wr_map(wb),
                'touch_count_sr': _wr_map(tb),
                'volume_at_reversal': _wr_map(vb)
            }

            # 5) Summarize winning/losing patterns (heuristic statements)
            # Use thresholds to highlight notable deltas
            try:
                def _top_k(d, k=2):
                    # d values like "WR xx% (N=yy)" -> sort by WR
                    def _wr(v):
                        try:
                            return float(v.split('%')[0].split()[-1])
                        except Exception:
                            return 0.0
                    return sorted(d.items(), key=lambda x: _wr(x[1]), reverse=True)[:k]

                vr_top = _top_k(patterns['market_conditions'].get('volatility_regime', {}))
                wb_top = _top_k(patterns['market_conditions'].get('range_width_atr', {}))
                tb_top = _top_k(patterns['market_conditions'].get('touch_count_sr', {}))
                vb_top = _top_k(patterns['market_conditions'].get('volume_at_reversal', {}))

                for k, v in vr_top:
                    patterns['winning_patterns'].append(f"Better in {k} volatility: {v}")
                for k, v in wb_top:
                    patterns['winning_patterns'].append(f"Range width {k}: {v}")
                for k, v in tb_top:
                    patterns['winning_patterns'].append(f"S/R {k}: {v}")
                for k, v in vb_top:
                    patterns['winning_patterns'].append(f"Reversal volume {k}: {v}")

                # Losing: pick bottom 2 for volatility and width
                def _bottom_k(d, k=2):
                    def _wr(v):
                        try:
                            return float(v.split('%')[0].split()[-1])
                        except Exception:
                            return 0.0
                    return sorted(d.items(), key=lambda x: _wr(x[1]))[:k]

                vr_bot = _bottom_k(patterns['market_conditions'].get('volatility_regime', {}))
                wb_bot = _bottom_k(patterns['market_conditions'].get('range_width_atr', {}))
                for k, v in vr_bot:
                    patterns['losing_patterns'].append(f"Worse in {k} volatility: {v}")
                for k, v in wb_bot:
                    patterns['losing_patterns'].append(f"Range width {k}: {v}")
            except Exception:
                pass

        except Exception as e:
            logger.error(f"Error extracting Enhanced MR patterns: {e}")

        return patterns

# Global instance
_enhanced_mr_scorer = None

def get_enhanced_mr_scorer(enabled: bool = True) -> EnhancedMeanReversionScorer:
    """Get or create the global enhanced mean reversion scorer"""
    global _enhanced_mr_scorer
    if _enhanced_mr_scorer is None:
        _enhanced_mr_scorer = EnhancedMeanReversionScorer(enabled=enabled)
        logger.info("Initialized Enhanced Mean Reversion ML Scorer")
    return _enhanced_mr_scorer
