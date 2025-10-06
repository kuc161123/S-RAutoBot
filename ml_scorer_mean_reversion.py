"""
ML Scorer for Mean Reversion Strategy

This is a dedicated ensemble ML scorer for the Mean Reversion strategy,
separate from the Trend strategy's scorer. It will be trained on
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

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def _stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3):
    """Calculate Stochastic Oscillator"""
    low_min = df['low'].rolling(window=k_period).min()
    high_max = df['high'].rolling(window=k_period).max()
    k_percent = 100 * ((df['close'] - low_min) / (high_max - low_min))
    d_percent = k_percent.rolling(window=d_period).mean()
    return k_percent, d_percent

def _atr(df: pd.DataFrame, n: int):
    """Calculate ATR"""
    prev_close = df["close"].shift()
    tr = np.maximum(df["high"]-df["low"],
         np.maximum(abs(df["high"]-prev_close), abs(df["low"]-prev_close)))
    return pd.Series(tr, index=df.index).rolling(n).mean().values

def calculate_mean_reversion_features(df: pd.DataFrame, signal_data: dict, symbol: str = "UNKNOWN") -> dict:
    """
    Calculate features specific to mean reversion strategy

    Args:
        df: Price DataFrame with OHLCV data
        signal_data: Dictionary containing signal information (side, entry, sl, tp, etc.)
        symbol: Trading symbol for context

    Returns:
        Dictionary of mean reversion specific features
    """
    if len(df) < 50:
        # Return default features if not enough data
        return _get_default_mr_features()

    try:
        # Extract signal information
        side = signal_data.get('side', 'long')
        entry_price = signal_data.get('entry', df['close'].iloc[-1])
        meta = signal_data.get('meta', {})
        range_upper = meta.get('range_upper', 0)
        range_lower = meta.get('range_lower', 0)
        range_confidence_meta = meta.get('range_confidence')

        # Basic OHLCV data
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        current_price = close.iloc[-1]
        current_atr = _atr(df, 14)[-1]

        # --- RANGE CHARACTERISTICS ---
        # Range width in ATR terms
        if range_upper > range_lower and current_atr > 0:
            range_width_atr = (range_upper - range_lower) / current_atr
        else:
            range_width_atr = 2.0  # Default assumption

        # Time spent in current range (estimate based on recent volatility)
        recent_highs = high.rolling(20).max()
        recent_lows = low.rolling(20).min()
        range_stability = 1.0 - ((recent_highs.iloc[-1] - recent_lows.iloc[-1]) /
                               (recent_highs.iloc[-20] - recent_lows.iloc[-20])) if len(df) >= 20 else 0.5
        time_in_range_candles = max(10, range_stability * 50)  # 10-50 candle estimate

        # Support/Resistance touch count (simplified)
        if range_upper > 0 and range_lower > 0:
            upper_touches = ((high.tail(30) >= range_upper * 0.995) &
                           (high.tail(30) <= range_upper * 1.005)).sum()
            lower_touches = ((low.tail(30) <= range_lower * 1.005) &
                           (low.tail(30) >= range_lower * 0.995)).sum()
            touch_count_sr = float(upper_touches + lower_touches)
        else:
            touch_count_sr = 2.0  # Default

        # Estimate range confidence if not provided
        if range_confidence_meta is not None:
            range_confidence_metric = float(range_confidence_meta)
        else:
            if range_upper > range_lower and range_upper > 0:
                range_pct = (range_upper - range_lower) / range_lower
                touch_score = min(1.0, touch_count_sr / 6)
                respect_rate = 0.0
                if range_upper > range_lower:
                    candles_in_range = ((close.tail(30) >= range_lower) & (close.tail(30) <= range_upper)).mean()
                    respect_rate = float(candles_in_range)
                range_confidence_metric = float(np.clip((touch_score * 0.5) + (respect_rate * 0.3) + (max(0.0, min(1.0, 0.08 / max(range_pct, 1e-6))) * 0.2), 0.0, 1.0))
            else:
                range_confidence_metric = 0.5

        # --- REVERSAL STRENGTH ---
        # Current candle size relative to ATR
        candle_range = high.iloc[-1] - low.iloc[-1]
        reversal_candle_size_atr = candle_range / current_atr if current_atr > 0 else 1.0

        # Volume at reversal point
        avg_volume_20 = volume.rolling(20).mean().iloc[-1] if len(df) >= 20 else volume.iloc[-1]
        volume_at_reversal_ratio = volume.iloc[-1] / avg_volume_20 if avg_volume_20 > 0 else 1.0

        # --- OSCILLATOR READINGS ---
        # RSI at the edge (should be extreme for good mean reversion)
        rsi_values = _rsi(close, 14)
        rsi_current = rsi_values.iloc[-1] if len(rsi_values) > 0 else 50.0

        # For mean reversion, we want RSI to be extreme
        if side == 'long':
            rsi_at_edge = max(0, 30 - rsi_current)  # How oversold (0-30 range)
        else:
            rsi_at_edge = max(0, rsi_current - 70)  # How overbought (0-30 range)

        # Stochastic at edge
        try:
            stoch_k, stoch_d = _stochastic(df, 14, 3)
            stoch_current = stoch_k.iloc[-1] if len(stoch_k) > 0 else 50.0

            if side == 'long':
                stochastic_at_edge = max(0, 20 - stoch_current)  # How oversold
            else:
                stochastic_at_edge = max(0, stoch_current - 80)  # How overbought
        except:
            stochastic_at_edge = 10.0  # Default moderate reading

        # --- MEAN REVERSION SPECIFIC ---
        # Distance from range midpoint
        if range_upper > range_lower:
            range_midpoint = (range_upper + range_lower) / 2
            distance_from_midpoint = abs(current_price - range_midpoint)
            distance_from_midpoint_atr = distance_from_midpoint / current_atr if current_atr > 0 else 0.5
        else:
            distance_from_midpoint_atr = 0.5

        # --- CONTEXT FEATURES ---
        # Volatility regime
        if len(df) >= 20:
            recent_atr_avg = np.mean(_atr(df, 14)[-20:])
            if current_atr > recent_atr_avg * 1.5:
                volatility_regime = "high"
            elif current_atr < recent_atr_avg * 0.7:
                volatility_regime = "low"
            else:
                volatility_regime = "normal"
        else:
            volatility_regime = "normal"

        # Time features
        now = datetime.now()
        hour_of_day = now.hour
        day_of_week = now.weekday()

        # Trading session
        if 0 <= hour_of_day < 8:
            session = "asian"
        elif 8 <= hour_of_day < 16:
            session = "european"
        elif 16 <= hour_of_day < 24:
            session = "us"
        else:
            session = "off_hours"

        # Symbol cluster (try to get from existing clustering)
        try:
            from symbol_clustering import load_symbol_clusters
            clusters = load_symbol_clusters()
            symbol_cluster = clusters.get(symbol, 3)  # Default to cluster 3
        except:
            symbol_cluster = 3

        return {
            # Range characteristics
            'range_width_atr': float(range_width_atr),
            'time_in_range_candles': float(time_in_range_candles),
            'touch_count_sr': float(touch_count_sr),
            'range_confidence': range_confidence_metric,

            # Reversal strength
            'reversal_candle_size_atr': float(reversal_candle_size_atr),
            'volume_at_reversal_ratio': float(volume_at_reversal_ratio),

            # Oscillator confirmation
            'rsi_at_edge': float(rsi_at_edge),
            'stochastic_at_edge': float(stochastic_at_edge),

            # Mean reversion specific
            'distance_from_midpoint_atr': float(distance_from_midpoint_atr),

            # Context
            'volatility_regime': volatility_regime,
            'hour_of_day': int(hour_of_day),
            'day_of_week': int(day_of_week),
            'session': session,
            'symbol_cluster': int(symbol_cluster)
        }

    except Exception as e:
        logger.error(f"Error calculating mean reversion features: {e}")
        return _get_default_mr_features()

def _get_default_mr_features() -> dict:
    """Return default features when calculation fails"""
    return {
        'range_width_atr': 2.0,
        'time_in_range_candles': 20.0,
        'touch_count_sr': 2.0,
        'range_confidence': 0.5,
        'reversal_candle_size_atr': 1.0,
        'volume_at_reversal_ratio': 1.0,
        'rsi_at_edge': 10.0,
        'stochastic_at_edge': 10.0,
        'distance_from_midpoint_atr': 0.5,
        'volatility_regime': "normal",
        'hour_of_day': 12,
        'day_of_week': 2,
        'session': "us",
        'symbol_cluster': 3
    }

class MLScorerMeanReversion:
    """
    Dedicated ML Scorer for the Mean Reversion strategy.
    Uses an ensemble of models trained on mean reversion specific features.
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
                # Validate feature count against current simplified spec
                expected = len(self._prepare_features({}))
                scaler_feats = getattr(self.scaler, 'n_features_in_', None)
                if scaler_feats is not None and scaler_feats != expected:
                    logger.warning(
                        f"Mean Reversion scaler feature mismatch: expected {expected}, found {scaler_feats}. Disabling ML until retrain."
                    )
                    self.models = {}
                    self.is_ml_ready = False
                    try:
                        self.redis_client.delete('ml:model:mean_reversion')
                        self.redis_client.delete('ml:scaler:mean_reversion')
                    except Exception:
                        pass
                else:
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
        """Converts feature dictionary to a list for the model (simplified set)."""
        # Minimal high-signal MR features
        feature_order = [
            'range_width_atr',
            'range_confidence',
            'touch_count_sr',
            'distance_from_midpoint_atr',
            'rsi_at_edge',
            'reversal_candle_size_atr',
            'volume_at_reversal_ratio',
            'volatility_regime',  # map to numeric
            'symbol_cluster'
        ]
        vector = []
        for feat in feature_order:
            val = features.get(feat)
            if feat == 'volatility_regime':
                val = {"low": 0, "normal": 1, "high": 2}.get(val, 1)
            vector.append(float(val) if val is not None else 0.0)
        return vector

    def _calibrate_probability(self, prob: float) -> float:
        """Calibrate raw probability to better spread scores."""
        if prob < 0.5:
            calibrated = prob ** 0.7
        else:
            calibrated = 1 - ((1 - prob) ** 0.7)
        calibrated = calibrated * 1.3
        return min(1.0, calibrated)

    def score_signal(self, signal_data: dict, features: Dict) -> Tuple[float, str]:
        """Scores a mean reversion signal using ensemble of specialized models.

        Args:
            signal_data: Signal information (side, entry, sl, tp, meta)
            features: Mean reversion specific features

        Returns:
            Tuple of (score, reason)
        """
        if not self.enabled:
            return 75.0, "ML scoring disabled"

        if not self.is_ml_ready:
            # Use theory-based scoring for mean reversion when ML not ready
            return self._theory_based_score(features)

        try:
            feature_vector = self._prepare_features(features)
            X = np.array([feature_vector]).reshape(1, -1)

            # Guard against feature count mismatch with saved scaler/models
            try:
                if hasattr(self.scaler, 'n_features_in_') and self.scaler.n_features_in_ != X.shape[1]:
                    logger.warning(
                        f"Mean Reversion ML feature mismatch: scaler expects {getattr(self.scaler, 'n_features_in_', '?')}, got {X.shape[1]}. Falling back to theory-based scoring until retrain."
                    )
                    # Disable ML usage until retrain occurs
                    self.is_ml_ready = False
                    return self._theory_based_score(features)
                X_scaled = self.scaler.transform(X)
            except Exception as e:
                logger.warning(f"Scaler transform failed ({e}); using theory-based scoring")
                self.is_ml_ready = False
                return self._theory_based_score(features)

            predictions = []
            for model_name, model in self.models.items():
                if model:
                    prob = model.predict_proba(X_scaled)[0][1]
                    predictions.append(self._calibrate_probability(prob) * 100)

            if predictions:
                score = float(np.mean(predictions))
                reason = f"MR Ensemble: {score:.1f} ({len(predictions)} models)"
                return score, reason
            return 75.0, "No models available"
        except Exception as e:
            logger.error(f"Mean Reversion ML scoring failed: {e}")
            # Fall back to theory-based scoring
            return self._theory_based_score(features)

    def _theory_based_score(self, features: Dict) -> Tuple[float, str]:
        """Theory-based scoring for mean reversion when ML not ready"""
        try:
            score = 50.0  # Base score
            reasons = []

            # Range characteristics (up to +20 points)
            range_width = features.get('range_width_atr', 2.0)
            if 1.5 <= range_width <= 4.0:  # Good range width
                score += 10
                reasons.append(f"Good range width ({range_width:.1f} ATR)")

            touches = features.get('touch_count_sr', 2.0)
            if touches >= 3:
                score += 10
                reasons.append(f"Strong S/R ({touches:.0f} touches)")

            # Oscillator extremes (up to +15 points)
            rsi_extreme = features.get('rsi_at_edge', 10.0)
            if rsi_extreme >= 15:  # Very extreme RSI
                score += 15
                reasons.append(f"Extreme RSI ({rsi_extreme:.0f})")
            elif rsi_extreme >= 8:
                score += 8
                reasons.append(f"Moderate RSI extreme")

            # Volume confirmation (up to +10 points)
            volume_ratio = features.get('volume_at_reversal_ratio', 1.0)
            if volume_ratio >= 1.3:
                score += 10
                reasons.append(f"Volume confirmation ({volume_ratio:.1f}x)")
            elif volume_ratio >= 1.1:
                score += 5

            # Distance from midpoint (up to +10 points)
            distance = features.get('distance_from_midpoint_atr', 0.5)
            if distance >= 0.8:  # Far from midpoint (good for reversion)
                score += 10
                reasons.append("Far from range midpoint")
            elif distance >= 0.5:
                score += 5

            # Reversal candle strength (up to +5 points)
            candle_size = features.get('reversal_candle_size_atr', 1.0)
            if candle_size >= 1.2:
                score += 5
                reasons.append("Strong reversal candle")

            # Time-based adjustments
            session = features.get('session', 'us')
            if session in ['us', 'european']:  # Better liquidity sessions
                score += 5
                reasons.append(f"Good session ({session})")

            # Volatility regime adjustment
            vol_regime = features.get('volatility_regime', 'normal')
            if vol_regime == 'low':
                score += 5  # Mean reversion works better in low vol
                reasons.append("Low volatility")
            elif vol_regime == 'high':
                score -= 5  # Reduce confidence in high vol

            score = max(40.0, min(95.0, score))  # Clamp between 40-95
            reason_str = "; ".join(reasons[:3]) if reasons else "Theory-based"

            return score, f"Theory: {score:.0f} ({reason_str})"

        except Exception as e:
            logger.error(f"Theory-based scoring failed: {e}")
            return 75.0, "Theory scoring error"

    def record_outcome(self, signal_data: Dict, outcome: str, pnl_percent: float = 0.0):
        """Records a trade outcome for training.

        Args:
            signal_data: Original signal data with features
            outcome: 'win' or 'loss'
            pnl_percent: P&L percentage for this trade
        """
        try:
            self.completed_trades += 1

            # Extract features; if missing, try to recover from MR phantom tracker (executed trades)
            features = signal_data.get('features', {})
            if not features:
                try:
                    from mr_phantom_tracker import MRPhantomTrade, MRPhantomTracker
                    # Create an instance (singleton pattern not used here)
                    tracker = MRPhantomTracker()
                    sym = signal_data.get('symbol')
                    if sym and sym in tracker.mr_phantom_trades:
                        recent = [p for p in reversed(tracker.mr_phantom_trades[sym]) if getattr(p, 'was_executed', False)]
                        if recent:
                            features = recent[0].features or {}
                except Exception:
                    pass
            outcome_binary = 1 if outcome == 'win' else 0

            # Store in Redis for persistence
            if self.redis_client:
                trade_record = {
                    'features': features,
                    'outcome': outcome_binary,
                    'pnl_percent': pnl_percent,
                    'timestamp': datetime.now().isoformat(),
                    'was_executed': 1
                }

                # Store in Redis list with strategy-specific key
                self.redis_client.lpush('ml:trades:mean_reversion', json.dumps(trade_record))

                # Keep only last 1000 trades to prevent memory bloat
                self.redis_client.ltrim('ml:trades:mean_reversion', 0, 999)

            else:
                # Memory fallback
                if not hasattr(self, 'memory_storage'):
                    self.memory_storage = {'trades': []}
                self.memory_storage['trades'].append({
                    'features': features,
                    'outcome': outcome_binary,
                    'pnl_percent': pnl_percent
                })
                # Keep only last 500 trades in memory
                self.memory_storage['trades'] = self.memory_storage['trades'][-500:]

            # Track recent performance for threshold adaptation
            self.recent_performance.append(outcome_binary)
            if len(self.recent_performance) > 50:
                self.recent_performance = self.recent_performance[-50:]

            # Save state
            self._save_state()

            logger.debug(f"Mean Reversion ML recorded: {outcome} (P&L: {pnl_percent:.1f}%), total trades: {self.completed_trades}")

            # Check for retraining
            if self.completed_trades >= self.MIN_TRADES_FOR_ML and \
               (self.completed_trades - self.last_train_count >= self.RETRAIN_INTERVAL):
                self._retrain_models()

        except Exception as e:
            logger.error(f"Error recording Mean Reversion outcome: {e}")

    def _retrain_models(self):
        """Retrains the ensemble models using stored trade data."""
        try:
            logger.info(f"🔄 Retraining Mean Reversion ML models... ({self.completed_trades} total trades)")

            # Load training data from Redis or memory
            training_data = self._load_training_data()

            if len(training_data) < self.MIN_TRADES_FOR_ML:
                logger.warning(f"Not enough data for Mean Reversion ML retraining: {len(training_data)} < {self.MIN_TRADES_FOR_ML}")
                return

            # Prepare features and targets
            X_list = []
            y_list = []
            weights_list = []
            pnl_list = []

            for trade in training_data:
                try:
                    features = trade.get('features', {})
                    outcome = trade.get('outcome', 0)
                    pnl = trade.get('pnl_percent', 0.0)

                    feature_vector = self._prepare_features(features)
                    expected_len = len(self._prepare_features({}))
                    if len(feature_vector) == expected_len:
                        X_list.append(feature_vector)
                        y_list.append(outcome)
                        pnl_list.append(pnl)
                        # Compute sample weight
                        w = 0.9  # default for MR executed (no explicit flag)
                        try:
                            if isinstance(features, dict) and features.get('routing') == 'none':
                                w = 0.5
                            cl = int(features.get('symbol_cluster', 0)) if isinstance(features, dict) else 0
                            if cl == 3:
                                w *= 0.7
                        except Exception:
                            pass
                        weights_list.append(w)
                except Exception as e:
                    logger.debug(f"Skipping invalid trade data: {e}")
                    continue

            if len(X_list) < self.MIN_TRADES_FOR_ML:
                logger.warning(f"Not enough valid features for training: {len(X_list)}")
                return

            X = np.array(X_list)
            y = np.array(y_list)

            # Check class balance
            wins = np.sum(y)
            losses = len(y) - wins
            win_rate = wins / len(y) if len(y) > 0 else 0.0

            logger.info(f"Training data: {len(y)} trades, {wins} wins, {losses} losses ({win_rate:.1%} win rate)")

            # Fit scaler and transform data
            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)

            # Create ensemble models optimized for mean reversion
            self.models = {
                'range_rf': RandomForestClassifier(
                    n_estimators=100,
                    max_depth=8,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    class_weight='balanced',
                    random_state=42
                ),
                'reversal_gb': GradientBoostingClassifier(
                    n_estimators=75,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    random_state=42
                ),
                'oscillator_nn': MLPClassifier(
                    hidden_layer_sizes=(20, 10),
                    max_iter=300,
                    alpha=0.01,
                    random_state=42
                )
            }

            # Train each model
            trained_models = 0
            for name, model in self.models.items():
                try:
                    if name in ('range_rf', 'reversal_gb'):
                        model.fit(X_scaled, y, sample_weight=np.array(weights_list))
                    else:
                        model.fit(X_scaled, y)
                    trained_models += 1
                    logger.debug(f"Trained {name} model successfully")
                except Exception as e:
                    logger.warning(f"Failed to train {name}: {e}")
                    self.models[name] = None

            if trained_models > 0:
                self.is_ml_ready = True
                self.last_train_count = self.completed_trades
                self._save_state()

                # Calculate training performance
                try:
                    train_score = self._evaluate_ensemble(X_scaled, y)
                    logger.info(f"✅ Mean Reversion ML models trained! {trained_models}/3 models, training accuracy: {train_score:.1%}")
                except:
                    logger.info(f"✅ Mean Reversion ML models trained! {trained_models}/3 models active")
            else:
                logger.error("❌ No models were trained successfully")

        except Exception as e:
            logger.error(f"Error during Mean Reversion ML retraining: {e}")

    def _load_training_data(self) -> List[dict]:
        """Load training data from Redis or memory storage"""
        training_data = []

        try:
            if self.redis_client:
                # Load from Redis
                redis_data = self.redis_client.lrange('ml:trades:mean_reversion', 0, -1)
                for trade_json in redis_data:
                    try:
                        trade = json.loads(trade_json)
                        training_data.append(trade)
                    except json.JSONDecodeError:
                        continue
            else:
                # Load from memory
                training_data = getattr(self, 'memory_storage', {}).get('trades', [])

        except Exception as e:
            logger.error(f"Error loading training data: {e}")

        return training_data

    def _evaluate_ensemble(self, X_scaled: np.ndarray, y: np.ndarray) -> float:
        """Evaluate ensemble performance on training data"""
        predictions = []
        for model_name, model in self.models.items():
            if model:
                try:
                    pred_proba = model.predict_proba(X_scaled)[:, 1]
                    predictions.append(pred_proba)
                except:
                    continue

        if predictions:
            ensemble_proba = np.mean(predictions, axis=0)
            ensemble_pred = (ensemble_proba > 0.5).astype(int)
            accuracy = np.mean(ensemble_pred == y)
            return accuracy
        return 0.0

    def get_stats(self) -> Dict:
        """Returns current ML stats for Mean Reversion strategy."""
        try:
            # Calculate recent performance if available
            recent_win_rate = 0.0
            if len(self.recent_performance) > 0:
                recent_win_rate = (sum(self.recent_performance) / len(self.recent_performance)) * 100

            # Count active models
            active_models = [name for name, model in self.models.items() if model is not None] if self.models else []

            # Get training info
            retrain_info = self.get_retrain_info()

            return {
                'strategy': 'Mean Reversion',
                'enabled': self.enabled,
                'is_ml_ready': self.is_ml_ready,
                'status': 'Active' if self.is_ml_ready else 'Learning',
                'completed_trades': self.completed_trades,
                'min_score': self.min_score,
                'current_threshold': self.min_score,
                'last_train_count': self.last_train_count,
                'models_active': active_models,
                'recent_win_rate': recent_win_rate,
                'recent_trades': len(self.recent_performance),
                'min_trades_for_ml': self.MIN_TRADES_FOR_ML,
                'retrain_interval': self.RETRAIN_INTERVAL,
                'next_retrain_in': retrain_info.get('trades_until_next_retrain', 0)
            }
        except Exception as e:
            logger.error(f"Error getting Mean Reversion stats: {e}")
            return {
                'strategy': 'Mean Reversion',
                'enabled': self.enabled,
                'error': str(e)
            }

    def get_retrain_info(self) -> Dict:
        """Get information about retraining status"""
        trades_until_next = max(0, self.RETRAIN_INTERVAL - (self.completed_trades - self.last_train_count))
        can_train = self.completed_trades >= self.MIN_TRADES_FOR_ML

        return {
            'can_train': can_train,
            'trades_until_next_retrain': trades_until_next,
            'total_trades': self.completed_trades,
            'last_train_at': self.last_train_count
        }

    def startup_retrain(self) -> bool:
        """Force retrain on startup if conditions are met"""
        try:
            if self.completed_trades >= self.MIN_TRADES_FOR_ML:
                logger.info(f"Mean Reversion startup retrain: {self.completed_trades} trades available")
                self._retrain_models()
                return self.is_ml_ready
            else:
                logger.info(f"Mean Reversion startup: Not enough trades for retrain ({self.completed_trades} < {self.MIN_TRADES_FOR_ML})")
                return False
        except Exception as e:
            logger.error(f"Mean Reversion startup retrain failed: {e}")
            return False

_mean_reversion_scorer_instance = None

def get_mean_reversion_scorer(enabled: bool = True) -> MLScorerMeanReversion:
    """Get singleton instance of Mean Reversion ML scorer"""
    global _mean_reversion_scorer_instance
    if _mean_reversion_scorer_instance is None:
        _mean_reversion_scorer_instance = MLScorerMeanReversion(enabled=enabled)
        logger.info("Initialized Mean Reversion ML Scorer")
    return _mean_reversion_scorer_instance
