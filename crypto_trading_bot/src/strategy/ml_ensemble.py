"""
ML Ensemble Model System
Combines multiple models for robust predictions
Implements online learning and model drift detection
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import joblib
import structlog
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

logger = structlog.get_logger(__name__)

@dataclass
class ModelPerformance:
    """Track model performance metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    samples_trained: int
    last_training: datetime
    drift_score: float
    feature_importance: Dict[str, float]

@dataclass
class PredictionResult:
    """Ensemble prediction result"""
    prediction: float  # 0-1 for probability
    confidence: float  # Confidence in prediction
    model_agreements: Dict[str, float]  # Individual model predictions
    expected_return: float  # Expected profit/loss
    risk_score: float  # Risk assessment
    recommendation: str  # 'strong_buy', 'buy', 'hold', 'sell', 'strong_sell'

class MLEnsemble:
    """
    Production-grade ML ensemble system with:
    - Multiple model types (RF, XGBoost, MLP, GB)
    - Feature engineering pipeline
    - Online learning capability
    - Model drift detection
    - Feature importance tracking
    - Cross-validation
    """
    
    def __init__(self):
        # Models
        self.models = {
            'random_forest': None,
            'xgboost': None,
            'gradient_boost': None,
            'neural_network': None
        }
        
        # Scalers
        self.feature_scaler = RobustScaler()  # Robust to outliers
        self.target_scaler = StandardScaler()
        
        # Feature configuration
        self.feature_columns = [
            # Price features
            'returns_1', 'returns_5', 'returns_10',
            'volatility_10', 'volatility_20',
            'rsi', 'macd', 'macd_signal',
            'bb_upper', 'bb_lower', 'bb_position',
            
            # Volume features
            'volume_ratio', 'volume_ma_ratio',
            'volume_delta', 'cumulative_delta',
            
            # Market structure
            'trend_strength', 'support_distance', 'resistance_distance',
            'higher_high', 'lower_low',
            
            # Order flow
            'order_flow_imbalance', 'absorption_score',
            'delta_divergence', 'liquidity_score',
            
            # Wyckoff features
            'wyckoff_phase', 'spring_detected',
            'institutional_activity',
            
            # Time features
            'hour_sin', 'hour_cos',
            'day_sin', 'day_cos',
            
            # Interaction features
            'volume_price_correlation',
            'delta_price_correlation',
            'trend_volume_interaction'
        ]
        
        # Training configuration
        self.min_training_samples = 500
        self.validation_split = 0.2
        self.online_learning_batch = 50
        
        # Performance tracking
        self.model_performance = {}
        self.training_history = []
        self.prediction_history = []
        
        # Drift detection
        self.baseline_performance = None
        self.drift_threshold = 0.15  # 15% performance drop
        
        # State
        self.is_trained = False
        self.training_data = []
        self.feature_importance_history = []
        
    def engineer_features(self, df: pd.DataFrame, zone_data: Dict = None) -> pd.DataFrame:
        """
        Comprehensive feature engineering
        Creates 30+ features from price, volume, and pattern data
        """
        
        features = pd.DataFrame(index=df.index)
        
        # Price-based features
        features['returns_1'] = df['close'].pct_change(1)
        features['returns_5'] = df['close'].pct_change(5)
        features['returns_10'] = df['close'].pct_change(10)
        
        # Volatility
        features['volatility_10'] = df['returns_1'].rolling(10).std()
        features['volatility_20'] = df['returns_1'].rolling(20).std()
        
        # RSI
        features['rsi'] = self._calculate_rsi(df['close'])
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        features['macd'] = exp1 - exp2
        features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        ma20 = df['close'].rolling(20).mean()
        std20 = df['close'].rolling(20).std()
        features['bb_upper'] = ma20 + (std20 * 2)
        features['bb_lower'] = ma20 - (std20 * 2)
        features['bb_position'] = (df['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'] + 0.0001)
        
        # Volume features
        features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        features['volume_ma_ratio'] = df['volume'].rolling(5).mean() / df['volume'].rolling(20).mean()
        
        # Volume delta (if available)
        if 'volume_delta' in df.columns:
            features['volume_delta'] = df['volume_delta']
            features['cumulative_delta'] = df['volume_delta'].cumsum()
        else:
            features['volume_delta'] = 0
            features['cumulative_delta'] = 0
        
        # Market structure
        features['trend_strength'] = self._calculate_trend_strength(df)
        features['support_distance'] = self._calculate_support_distance(df)
        features['resistance_distance'] = self._calculate_resistance_distance(df)
        
        # Higher highs and lower lows
        features['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
        features['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
        
        # Order flow features (simplified if not available)
        if 'order_flow_imbalance' in df.columns:
            features['order_flow_imbalance'] = df['order_flow_imbalance']
        else:
            features['order_flow_imbalance'] = features['volume_delta'] / df['volume']
        
        features['absorption_score'] = self._calculate_absorption_score(df)
        features['delta_divergence'] = self._calculate_delta_divergence(df, features)
        features['liquidity_score'] = self._calculate_liquidity_score(df)
        
        # Wyckoff features (from zone_data if available)
        if zone_data:
            features['wyckoff_phase'] = zone_data.get('wyckoff_phase', 0)
            features['spring_detected'] = zone_data.get('spring_detected', 0)
            features['institutional_activity'] = zone_data.get('institutional_activity', 0)
        else:
            features['wyckoff_phase'] = 0
            features['spring_detected'] = 0
            features['institutional_activity'] = 0
        
        # Time features (cyclical encoding)
        if hasattr(df.index, 'hour'):
            features['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
            features['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
            features['day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
            features['day_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        else:
            features['hour_sin'] = 0
            features['hour_cos'] = 1
            features['day_sin'] = 0
            features['day_cos'] = 1
        
        # Interaction features
        features['volume_price_correlation'] = df['close'].rolling(20).corr(df['volume'])
        features['delta_price_correlation'] = features['cumulative_delta'].rolling(20).corr(df['close'])
        features['trend_volume_interaction'] = features['trend_strength'] * features['volume_ratio']
        
        # Fill NaN values
        features = features.fillna(method='ffill').fillna(0)
        
        return features
    
    def train(self, X: pd.DataFrame, y: np.ndarray, online: bool = False):
        """
        Train ensemble models
        
        Args:
            X: Feature DataFrame
            y: Target values (0 or 1 for classification)
            online: Whether this is online learning update
        """
        
        if len(X) < self.min_training_samples and not online:
            logger.warning(f"Insufficient training samples: {len(X)} < {self.min_training_samples}")
            return
        
        try:
            # Ensure we have the right features
            X_train = X[self.feature_columns] if all(col in X.columns for col in self.feature_columns) else X
            
            # Scale features
            X_scaled = self.feature_scaler.fit_transform(X_train)
            
            # Split data
            if not online:
                X_train_split, X_val_split, y_train, y_val = train_test_split(
                    X_scaled, y, test_size=self.validation_split, random_state=42
                )
            else:
                X_train_split, y_train = X_scaled, y
                X_val_split, y_val = None, None
            
            # Train Random Forest
            self.models['random_forest'] = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                random_state=42,
                n_jobs=-1
            )
            self.models['random_forest'].fit(X_train_split, y_train)
            
            # Train XGBoost
            self.models['xgboost'] = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
            self.models['xgboost'].fit(X_train_split, y_train)
            
            # Train Gradient Boosting
            self.models['gradient_boost'] = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
            self.models['gradient_boost'].fit(X_train_split, y_train)
            
            # Train Neural Network
            self.models['neural_network'] = MLPClassifier(
                hidden_layer_sizes=(100, 50, 25),
                activation='relu',
                solver='adam',
                alpha=0.001,
                max_iter=500,
                random_state=42
            )
            self.models['neural_network'].fit(X_train_split, y_train)
            
            # Evaluate if validation set available
            if X_val_split is not None:
                self._evaluate_models(X_val_split, y_val)
            
            # Update feature importance
            self._update_feature_importance(X_train)
            
            # Mark as trained
            self.is_trained = True
            
            # Record training
            self.training_history.append({
                'timestamp': datetime.now(),
                'samples': len(X),
                'online': online
            })
            
            logger.info(f"Ensemble training complete with {len(X)} samples")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def predict(self, X: pd.DataFrame) -> PredictionResult:
        """
        Make ensemble prediction
        
        Args:
            X: Feature DataFrame
            
        Returns:
            PredictionResult with ensemble prediction
        """
        
        if not self.is_trained:
            raise ValueError("Models not trained")
        
        try:
            # Prepare features
            X_pred = X[self.feature_columns] if all(col in X.columns for col in self.feature_columns) else X
            X_scaled = self.feature_scaler.transform(X_pred)
            
            # Get predictions from each model
            predictions = {}
            
            # Random Forest
            rf_pred = self.models['random_forest'].predict_proba(X_scaled)[:, 1]
            predictions['random_forest'] = rf_pred.mean()
            
            # XGBoost
            xgb_pred = self.models['xgboost'].predict_proba(X_scaled)[:, 1]
            predictions['xgboost'] = xgb_pred.mean()
            
            # Gradient Boosting (regression, so we clip to [0, 1])
            gb_pred = np.clip(self.models['gradient_boost'].predict(X_scaled), 0, 1)
            predictions['gradient_boost'] = gb_pred.mean()
            
            # Neural Network
            nn_pred = self.models['neural_network'].predict_proba(X_scaled)[:, 1]
            predictions['neural_network'] = nn_pred.mean()
            
            # Weighted ensemble (can be optimized)
            weights = {
                'random_forest': 0.25,
                'xgboost': 0.35,
                'gradient_boost': 0.20,
                'neural_network': 0.20
            }
            
            ensemble_pred = sum(predictions[model] * weight for model, weight in weights.items())
            
            # Calculate confidence (based on model agreement)
            pred_std = np.std(list(predictions.values()))
            confidence = 1.0 - (pred_std * 2)  # Higher agreement = higher confidence
            confidence = max(0, min(1, confidence))
            
            # Calculate expected return (simplified)
            expected_return = (ensemble_pred - 0.5) * 4  # Scale to roughly -2 to +2
            
            # Calculate risk score
            risk_score = self._calculate_risk_score(X_pred, pred_std)
            
            # Determine recommendation
            if ensemble_pred > 0.7 and confidence > 0.7:
                recommendation = 'strong_buy'
            elif ensemble_pred > 0.6:
                recommendation = 'buy'
            elif ensemble_pred < 0.3 and confidence > 0.7:
                recommendation = 'strong_sell'
            elif ensemble_pred < 0.4:
                recommendation = 'sell'
            else:
                recommendation = 'hold'
            
            result = PredictionResult(
                prediction=ensemble_pred,
                confidence=confidence,
                model_agreements=predictions,
                expected_return=expected_return,
                risk_score=risk_score,
                recommendation=recommendation
            )
            
            # Store prediction for drift detection
            self.prediction_history.append({
                'timestamp': datetime.now(),
                'prediction': ensemble_pred,
                'confidence': confidence
            })
            
            # Check for model drift
            if len(self.prediction_history) > 100:
                self._check_model_drift()
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def online_update(self, X: pd.DataFrame, y: np.ndarray):
        """
        Online learning update with new data
        
        Args:
            X: New feature data
            y: New target values
        """
        
        if len(X) < self.online_learning_batch:
            # Store for batch update
            self.training_data.append((X, y))
            
            if sum(len(x) for x, _ in self.training_data) >= self.online_learning_batch:
                # Combine stored data
                X_combined = pd.concat([x for x, _ in self.training_data])
                y_combined = np.concatenate([y for _, y in self.training_data])
                
                # Train with online flag
                self.train(X_combined, y_combined, online=True)
                
                # Clear stored data
                self.training_data = []
        else:
            # Train immediately
            self.train(X, y, online=True)
    
    def _evaluate_models(self, X_val: np.ndarray, y_val: np.ndarray):
        """Evaluate individual model performance"""
        
        for name, model in self.models.items():
            if model is None:
                continue
            
            if name == 'gradient_boost':
                # Regression model, threshold at 0.5
                y_pred = (model.predict(X_val) > 0.5).astype(int)
            else:
                y_pred = model.predict(X_val)
            
            performance = ModelPerformance(
                accuracy=accuracy_score(y_val, y_pred),
                precision=precision_score(y_val, y_pred, zero_division=0),
                recall=recall_score(y_val, y_pred, zero_division=0),
                f1_score=f1_score(y_val, y_pred, zero_division=0),
                samples_trained=len(y_val),
                last_training=datetime.now(),
                drift_score=0.0,
                feature_importance={}
            )
            
            self.model_performance[name] = performance
            
            logger.info(f"{name} performance - Accuracy: {performance.accuracy:.3f}, F1: {performance.f1_score:.3f}")
    
    def _update_feature_importance(self, X: pd.DataFrame):
        """Update and track feature importance"""
        
        importance_dict = {}
        
        # Get feature importance from tree-based models
        if self.models['random_forest']:
            rf_importance = self.models['random_forest'].feature_importances_
            for i, col in enumerate(X.columns):
                importance_dict[col] = importance_dict.get(col, 0) + rf_importance[i]
        
        if self.models['xgboost']:
            xgb_importance = self.models['xgboost'].feature_importances_
            for i, col in enumerate(X.columns):
                importance_dict[col] = importance_dict.get(col, 0) + xgb_importance[i]
        
        # Normalize
        total = sum(importance_dict.values())
        if total > 0:
            importance_dict = {k: v/total for k, v in importance_dict.items()}
        
        # Store
        self.feature_importance_history.append({
            'timestamp': datetime.now(),
            'importance': importance_dict
        })
        
        # Log top features
        top_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:5]
        logger.info(f"Top features: {top_features}")
    
    def _check_model_drift(self):
        """Detect model drift based on prediction distribution"""
        
        recent_predictions = [p['prediction'] for p in self.prediction_history[-50:]]
        older_predictions = [p['prediction'] for p in self.prediction_history[-100:-50]]
        
        if len(older_predictions) < 20:
            return
        
        # Compare distributions
        recent_mean = np.mean(recent_predictions)
        older_mean = np.mean(older_predictions)
        
        drift = abs(recent_mean - older_mean)
        
        if drift > self.drift_threshold:
            logger.warning(f"Model drift detected: {drift:.3f} > {self.drift_threshold}")
            
            # Update drift scores
            for name in self.model_performance:
                self.model_performance[name].drift_score = drift
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_trend_strength(self, df: pd.DataFrame) -> pd.Series:
        """Calculate trend strength using ADX logic"""
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(14).mean()
        
        # Simplified trend strength
        price_change = df['close'].diff(14)
        trend_strength = abs(price_change) / atr
        
        return trend_strength.fillna(0)
    
    def _calculate_support_distance(self, df: pd.DataFrame) -> pd.Series:
        """Calculate distance to nearest support"""
        support = df['low'].rolling(20).min()
        distance = (df['close'] - support) / df['close']
        return distance.fillna(0)
    
    def _calculate_resistance_distance(self, df: pd.DataFrame) -> pd.Series:
        """Calculate distance to nearest resistance"""
        resistance = df['high'].rolling(20).max()
        distance = (resistance - df['close']) / df['close']
        return distance.fillna(0)
    
    def _calculate_absorption_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate absorption pattern score"""
        volume_ratio = df['volume'] / df['volume'].rolling(20).mean()
        price_range = (df['high'] - df['low']) / df['close']
        avg_range = price_range.rolling(20).mean()
        
        # High volume with small range = absorption
        absorption = volume_ratio * (1 - price_range / avg_range.where(avg_range > 0, 1))
        
        return absorption.fillna(0)
    
    def _calculate_delta_divergence(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
        """Calculate delta divergence score"""
        if 'cumulative_delta' in features.columns:
            price_change = df['close'].pct_change(10)
            delta_change = features['cumulative_delta'].pct_change(10)
            
            # Divergence when signs don't match
            divergence = np.where(
                (price_change * delta_change) < 0,
                abs(price_change - delta_change),
                0
            )
            
            return pd.Series(divergence, index=df.index).fillna(0)
        
        return pd.Series(0, index=df.index)
    
    def _calculate_liquidity_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate liquidity score based on volume profile"""
        # Volume at key levels
        volume_at_highs = df[df['high'] == df['high'].rolling(20).max()]['volume']
        volume_at_lows = df[df['low'] == df['low'].rolling(20).min()]['volume']
        
        # Create score series
        liquidity = pd.Series(0, index=df.index)
        
        # Mark high liquidity areas
        for idx in volume_at_highs.index:
            if idx in liquidity.index:
                liquidity.loc[idx] = volume_at_highs.loc[idx] / df['volume'].mean()
        
        for idx in volume_at_lows.index:
            if idx in liquidity.index:
                liquidity.loc[idx] = max(liquidity.loc[idx], volume_at_lows.loc[idx] / df['volume'].mean())
        
        return liquidity.fillna(0)
    
    def _calculate_risk_score(self, features: pd.DataFrame, prediction_std: float) -> float:
        """Calculate risk score for prediction"""
        
        risk_factors = []
        
        # Volatility risk
        if 'volatility_20' in features.columns:
            vol_risk = features['volatility_20'].iloc[-1] / features['volatility_20'].mean()
            risk_factors.append(min(vol_risk, 2))
        
        # Prediction uncertainty
        risk_factors.append(prediction_std * 2)
        
        # Market conditions
        if 'order_flow_imbalance' in features.columns:
            imbalance_risk = abs(features['order_flow_imbalance'].iloc[-1])
            risk_factors.append(min(imbalance_risk, 1))
        
        # Average risk
        risk_score = np.mean(risk_factors) if risk_factors else 0.5
        
        return min(1, max(0, risk_score))
    
    def get_model_stats(self) -> Dict:
        """Get comprehensive model statistics"""
        return {
            'is_trained': self.is_trained,
            'models': list(self.models.keys()),
            'performance': {
                name: {
                    'accuracy': perf.accuracy,
                    'f1_score': perf.f1_score,
                    'drift_score': perf.drift_score
                }
                for name, perf in self.model_performance.items()
            },
            'training_history': len(self.training_history),
            'prediction_history': len(self.prediction_history),
            'feature_count': len(self.feature_columns)
        }
    
    def save_models(self, path: str):
        """Save trained models to disk"""
        import os
        os.makedirs(path, exist_ok=True)
        
        for name, model in self.models.items():
            if model:
                joblib.dump(model, f"{path}/{name}.joblib")
        
        # Save scalers
        joblib.dump(self.feature_scaler, f"{path}/feature_scaler.joblib")
        joblib.dump(self.target_scaler, f"{path}/target_scaler.joblib")
        
        logger.info(f"Models saved to {path}")
    
    def load_models(self, path: str):
        """Load trained models from disk"""
        import os
        
        for name in self.models.keys():
            model_path = f"{path}/{name}.joblib"
            if os.path.exists(model_path):
                self.models[name] = joblib.load(model_path)
        
        # Load scalers
        scaler_path = f"{path}/feature_scaler.joblib"
        if os.path.exists(scaler_path):
            self.feature_scaler = joblib.load(scaler_path)
        
        target_scaler_path = f"{path}/target_scaler.joblib"
        if os.path.exists(target_scaler_path):
            self.target_scaler = joblib.load(target_scaler_path)
        
        self.is_trained = any(model is not None for model in self.models.values())
        
        logger.info(f"Models loaded from {path}")