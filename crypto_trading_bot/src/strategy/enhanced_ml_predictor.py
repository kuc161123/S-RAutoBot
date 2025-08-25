"""
Enhanced Machine Learning Predictor with Improved Accuracy and Reliability
Implements data validation, cross-validation, confidence intervals, and performance tracking
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import joblib
import structlog
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, IsolationForest
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

logger = structlog.get_logger(__name__)

@dataclass
class EnhancedZoneFeatures:
    """Enhanced features for ML prediction with dynamic market features"""
    # Original features
    zone_strength: float
    volume_ratio: float
    rejection_speed: float
    institutional_interest: float
    confluence_count: int
    timeframes_visible: int
    market_structure: int
    order_flow_imbalance: int
    zone_age_hours: float
    test_count: int
    distance_from_poc: float
    in_value_area: bool
    trend_alignment: bool
    volatility_ratio: float
    liquidity_score: float
    time_of_day: int
    day_of_week: int
    
    # New dynamic features
    momentum_score: float = 0.0  # Rate of price change
    volume_momentum: float = 0.0  # Volume acceleration
    volatility_percentile: float = 0.0  # Current vol vs historical
    spread_ratio: float = 0.0  # Bid-ask spread relative to ATR
    market_efficiency: float = 0.0  # Price efficiency score
    cross_market_correlation: float = 0.0  # Correlation with BTC
    zone_quality_score: float = 0.0  # Composite zone quality
    risk_on_off_score: float = 0.0  # Market risk sentiment
    
    def to_array(self) -> np.array:
        """Convert to numpy array for ML model"""
        return np.array([
            self.zone_strength,
            self.volume_ratio,
            self.rejection_speed,
            self.institutional_interest,
            self.confluence_count,
            self.timeframes_visible,
            self.market_structure,
            self.order_flow_imbalance,
            self.zone_age_hours,
            self.test_count,
            self.distance_from_poc,
            int(self.in_value_area),
            int(self.trend_alignment),
            self.volatility_ratio,
            self.liquidity_score,
            self.time_of_day,
            self.day_of_week,
            # New features
            self.momentum_score,
            self.volume_momentum,
            self.volatility_percentile,
            self.spread_ratio,
            self.market_efficiency,
            self.cross_market_correlation,
            self.zone_quality_score,
            self.risk_on_off_score
        ])

@dataclass
class PredictionResult:
    """ML prediction with confidence intervals"""
    success_probability: float
    confidence_lower: float  # Lower bound of confidence interval
    confidence_upper: float  # Upper bound of confidence interval
    expected_profit: float
    profit_lower: float
    profit_upper: float
    prediction_confidence: float  # Overall confidence in prediction
    feature_contributions: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

@dataclass
class ModelPerformance:
    """Track model performance metrics"""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    sharpe_improvement: float = 0.0
    total_predictions: int = 0
    correct_predictions: int = 0
    profit_correlation: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    performance_by_regime: Dict[str, float] = field(default_factory=dict)

class EnhancedMLPredictor:
    """Enhanced ML predictor with improved accuracy and reliability"""
    
    def __init__(self):
        # Models
        self.success_classifier = None
        self.profit_regressor = None
        self.anomaly_detector = None  # For outlier detection
        
        # Scalers
        self.scaler = RobustScaler()  # More robust to outliers
        self.feature_selector = None
        
        # Training data and metadata
        self.training_data = []
        self.feature_importance = {}
        self.selected_features = []
        self.model_trained = False
        self.min_training_samples = 100
        self.max_training_samples = 10000  # Prevent memory issues
        
        # Performance tracking
        self.performance = ModelPerformance()
        self.prediction_history = []
        self.model_version = "1.0.0"
        self.last_training_time = None
        
        # Validation thresholds
        self.min_confidence = 0.3  # Minimum acceptable confidence
        self.max_confidence = 0.95  # Cap unrealistic confidence
        self.outlier_threshold = 0.05  # 5% contamination for outlier detection
        
    def extract_enhanced_features(
        self, 
        zone: 'EnhancedZone', 
        market_data: Dict
    ) -> EnhancedZoneFeatures:
        """Extract enhanced ML features with dynamic market indicators"""
        
        # Market structure encoding
        market_structure_map = {
            'bullish': 2,
            'bearish': -2,
            'ranging': 0,
            'transitioning': 1
        }
        
        # Order flow encoding
        order_flow_map = {
            'strong_buying': 2,
            'buying': 1,
            'neutral': 0,
            'selling': -1,
            'strong_selling': -2
        }
        
        # Calculate dynamic features
        momentum_score = self._calculate_momentum(market_data)
        volume_momentum = self._calculate_volume_momentum(market_data)
        volatility_percentile = self._calculate_volatility_percentile(market_data)
        spread_ratio = self._calculate_spread_ratio(market_data)
        market_efficiency = self._calculate_market_efficiency(market_data)
        cross_market_correlation = market_data.get('btc_correlation', 0.5)
        zone_quality_score = self._calculate_zone_quality(zone)
        risk_on_off_score = self._calculate_risk_sentiment(market_data)
        
        features = EnhancedZoneFeatures(
            zone_strength=zone.strength_score / 100,
            volume_ratio=zone.volume_profile.total_volume / max(market_data.get('avg_volume', 1), 1),
            rejection_speed=zone.rejection_strength,
            institutional_interest=zone.institutional_interest / 100,
            confluence_count=len(zone.confluence_factors),
            timeframes_visible=len(zone.timeframes_visible),
            market_structure=market_structure_map.get(
                market_data.get('market_structure', 'ranging'), 0
            ),
            order_flow_imbalance=order_flow_map.get(
                str(zone.order_flow_imbalance).split('.')[-1].lower(), 0
            ),
            zone_age_hours=min(zone.zone_age_hours, 168),  # Cap at 1 week
            test_count=min(zone.test_count, 10),  # Cap at 10
            distance_from_poc=abs(zone.volume_profile.poc - 
                                (zone.upper_bound + zone.lower_bound) / 2) / max(zone.volume_profile.poc, 1),
            in_value_area=(zone.lower_bound <= zone.volume_profile.vah and 
                          zone.upper_bound >= zone.volume_profile.val),
            trend_alignment='trend_aligned' in zone.confluence_factors,
            volatility_ratio=min(market_data.get('volatility', 1.0), 3.0),  # Cap at 3x
            liquidity_score=min(zone.liquidity_pool / max(market_data.get('avg_liquidity', zone.liquidity_pool), 1), 2.0),
            time_of_day=zone.formation_time.hour,
            day_of_week=zone.formation_time.weekday(),
            # New features
            momentum_score=momentum_score,
            volume_momentum=volume_momentum,
            volatility_percentile=volatility_percentile,
            spread_ratio=spread_ratio,
            market_efficiency=market_efficiency,
            cross_market_correlation=cross_market_correlation,
            zone_quality_score=zone_quality_score,
            risk_on_off_score=risk_on_off_score
        )
        
        return features
    
    def _calculate_momentum(self, market_data: Dict) -> float:
        """Calculate price momentum score"""
        try:
            returns = market_data.get('returns_5m', [])
            if not returns:
                return 0.0
            
            # Calculate momentum as recent returns vs longer-term
            recent = np.mean(returns[-5:]) if len(returns) >= 5 else 0
            longer = np.mean(returns[-20:]) if len(returns) >= 20 else 0
            
            momentum = (recent - longer) / max(abs(longer), 0.0001)
            return np.clip(momentum, -2, 2)  # Clip to reasonable range
            
        except Exception:
            return 0.0
    
    def _calculate_volume_momentum(self, market_data: Dict) -> float:
        """Calculate volume momentum"""
        try:
            volumes = market_data.get('volumes_5m', [])
            if not volumes or len(volumes) < 2:
                return 0.0
            
            recent_vol = np.mean(volumes[-5:]) if len(volumes) >= 5 else volumes[-1]
            avg_vol = np.mean(volumes)
            
            return np.clip((recent_vol - avg_vol) / max(avg_vol, 1), -2, 2)
            
        except Exception:
            return 0.0
    
    def _calculate_volatility_percentile(self, market_data: Dict) -> float:
        """Calculate current volatility percentile"""
        try:
            current_vol = market_data.get('volatility', 1.0)
            vol_history = market_data.get('volatility_history', [current_vol])
            
            if not vol_history:
                return 0.5
            
            percentile = sum(1 for v in vol_history if v < current_vol) / len(vol_history)
            return percentile
            
        except Exception:
            return 0.5
    
    def _calculate_spread_ratio(self, market_data: Dict) -> float:
        """Calculate bid-ask spread relative to ATR"""
        try:
            spread = market_data.get('spread', 0.001)
            atr = market_data.get('atr', 1.0)
            
            return min(spread / max(atr, 0.0001), 0.1)  # Cap at 10% of ATR
            
        except Exception:
            return 0.01
    
    def _calculate_market_efficiency(self, market_data: Dict) -> float:
        """Calculate market efficiency (0 = random walk, 1 = trending)"""
        try:
            prices = market_data.get('prices_5m', [])
            if len(prices) < 10:
                return 0.5
            
            # Calculate efficiency ratio (directional movement / total movement)
            net_change = abs(prices[-1] - prices[0])
            total_movement = sum(abs(prices[i] - prices[i-1]) for i in range(1, len(prices)))
            
            if total_movement == 0:
                return 0.5
            
            efficiency = net_change / total_movement
            return np.clip(efficiency, 0, 1)
            
        except Exception:
            return 0.5
    
    def _calculate_zone_quality(self, zone: 'EnhancedZone') -> float:
        """Calculate composite zone quality score"""
        try:
            quality = 0.0
            
            # Strong rejection
            if zone.rejection_strength > 2:
                quality += 0.25
            
            # Multiple timeframes
            if len(zone.timeframes_visible) >= 3:
                quality += 0.25
            
            # High institutional interest
            if zone.institutional_interest > 70:
                quality += 0.25
            
            # Multiple confluence factors
            if len(zone.confluence_factors) >= 3:
                quality += 0.25
            
            return quality
            
        except Exception:
            return 0.5
    
    def _calculate_risk_sentiment(self, market_data: Dict) -> float:
        """Calculate market risk-on/risk-off sentiment"""
        try:
            # Use VIX proxy or volatility
            vix_proxy = market_data.get('vix_proxy', 20)
            
            if vix_proxy < 15:
                return 1.0  # Risk-on
            elif vix_proxy < 25:
                return 0.5  # Neutral
            else:
                return 0.0  # Risk-off
                
        except Exception:
            return 0.5
    
    def validate_training_data(self, features: np.array, outcome: bool, profit: float) -> bool:
        """Validate training data before adding to dataset"""
        
        # Check for NaN or infinite values
        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            logger.warning("Invalid features detected (NaN or Inf)")
            return False
        
        # Check profit bounds (realistic profit/loss)
        if profit < -1.0 or profit > 10.0:  # -100% to +1000%
            logger.warning(f"Unrealistic profit ratio: {profit}")
            return False
        
        # Check feature ranges
        if np.any(features < -10) or np.any(features > 10):
            logger.warning("Features outside reasonable range")
            return False
        
        # Detect outliers using Isolation Forest if trained
        if self.anomaly_detector is not None:
            try:
                features_scaled = self.scaler.transform(features.reshape(1, -1))
                is_outlier = self.anomaly_detector.predict(features_scaled)[0] == -1
                
                if is_outlier:
                    logger.info("Outlier detected in training data")
                    # Still accept but flag it
                    
            except Exception as e:
                logger.debug(f"Outlier detection failed: {e}")
        
        return True
    
    def predict_with_confidence(
        self,
        zone: 'EnhancedZone',
        market_data: Dict
    ) -> PredictionResult:
        """
        Make prediction with confidence intervals and validation
        """
        
        result = PredictionResult(
            success_probability=0.5,
            confidence_lower=0.3,
            confidence_upper=0.7,
            expected_profit=1.0,
            profit_lower=0.5,
            profit_upper=1.5,
            prediction_confidence=0.5
        )
        
        if not self.model_trained:
            # Return baseline predictions with low confidence
            base_prob = zone.composite_score / 100
            result.success_probability = base_prob
            result.confidence_lower = max(0, base_prob - 0.2)
            result.confidence_upper = min(1, base_prob + 0.2)
            result.prediction_confidence = 0.3  # Low confidence
            result.warnings.append("Model not trained - using rule-based prediction")
            return result
        
        try:
            # Extract features
            features = self.extract_enhanced_features(zone, market_data)
            X = features.to_array().reshape(1, -1)
            
            # Validate features
            if np.any(np.isnan(X)) or np.any(np.isinf(X)):
                result.warnings.append("Invalid features detected")
                return result
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Select features if selector is trained
            if self.feature_selector is not None:
                X_scaled = self.feature_selector.transform(X_scaled)
            
            # Get predictions from classifier
            success_proba = self.success_classifier.predict_proba(X_scaled)[0]
            success_prob = success_proba[1]
            
            # Calculate confidence intervals using ensemble variance
            if hasattr(self.success_classifier, 'estimators_'):
                # Get predictions from each tree
                tree_predictions = np.array([
                    tree.predict_proba(X_scaled)[0, 1] 
                    for tree in self.success_classifier.estimators_
                ])
                
                # Calculate confidence interval
                std_dev = np.std(tree_predictions)
                confidence_lower = max(0, success_prob - 1.96 * std_dev)
                confidence_upper = min(1, success_prob + 1.96 * std_dev)
                
                # Overall prediction confidence based on agreement
                prediction_confidence = 1 - (std_dev * 2)  # Higher std = lower confidence
            else:
                confidence_lower = max(0, success_prob - 0.15)
                confidence_upper = min(1, success_prob + 0.15)
                prediction_confidence = 0.7
            
            # Predict expected profit
            expected_profit = self.profit_regressor.predict(X_scaled)[0]
            
            # Calculate profit confidence intervals
            if hasattr(self.profit_regressor, 'estimators_'):
                profit_predictions = np.array([
                    estimator.predict(X_scaled)[0]
                    for estimator in self.profit_regressor.estimators_
                ])
                profit_std = np.std(profit_predictions)
                profit_lower = max(0.1, expected_profit - 1.96 * profit_std)
                profit_upper = min(5.0, expected_profit + 1.96 * profit_std)
            else:
                profit_lower = max(0.1, expected_profit * 0.7)
                profit_upper = min(5.0, expected_profit * 1.3)
            
            # Combine with rule-based confidence (weighted average)
            ml_weight = min(0.8, prediction_confidence)  # Cap ML weight at 80%
            rule_weight = 1 - ml_weight
            
            combined_prob = (success_prob * ml_weight + zone.composite_score / 100 * rule_weight)
            
            # Sanity checks
            if combined_prob < self.min_confidence:
                result.warnings.append(f"Low confidence: {combined_prob:.2f}")
            if combined_prob > self.max_confidence:
                combined_prob = self.max_confidence
                result.warnings.append("Confidence capped at maximum")
            
            # Check for anomalies
            if self.anomaly_detector is not None:
                is_anomaly = self.anomaly_detector.predict(X_scaled)[0] == -1
                if is_anomaly:
                    prediction_confidence *= 0.7  # Reduce confidence for anomalies
                    result.warnings.append("Unusual market conditions detected")
            
            # Update result
            result.success_probability = combined_prob
            result.confidence_lower = confidence_lower
            result.confidence_upper = confidence_upper
            result.expected_profit = max(0.5, expected_profit)
            result.profit_lower = profit_lower
            result.profit_upper = profit_upper
            result.prediction_confidence = prediction_confidence
            
            # Add feature contributions if available
            if self.feature_importance:
                top_features = sorted(
                    self.feature_importance.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:5]
                result.feature_contributions = dict(top_features)
            
            # Track prediction for performance monitoring
            self._track_prediction(result, zone, market_data)
            
            return result
            
        except Exception as e:
            logger.error(f"ML prediction failed: {e}")
            result.warnings.append(f"Prediction error: {str(e)}")
            return result
    
    def add_training_sample(
        self,
        zone: 'EnhancedZone',
        market_data: Dict,
        outcome: bool,
        profit_ratio: float
    ):
        """Add validated training sample"""
        
        features = self.extract_enhanced_features(zone, market_data)
        features_array = features.to_array()
        
        # Validate before adding
        if not self.validate_training_data(features_array, outcome, profit_ratio):
            logger.warning("Training sample rejected due to validation failure")
            return
        
        # Add to training data
        self.training_data.append({
            'features': features_array,
            'success': outcome,
            'profit_ratio': profit_ratio,
            'timestamp': datetime.now(),
            'market_regime': market_data.get('regime', 'unknown')
        })
        
        # Limit dataset size
        if len(self.training_data) > self.max_training_samples:
            # Keep most recent samples
            self.training_data = self.training_data[-self.max_training_samples:]
        
        # Retrain based on performance or schedule
        should_retrain = self._should_retrain()
        
        if should_retrain and len(self.training_data) >= self.min_training_samples:
            self.train_models()
    
    def _should_retrain(self) -> bool:
        """Determine if models should be retrained"""
        
        # Always train if not trained
        if not self.model_trained:
            return True
        
        # Retrain every 100 new samples
        if len(self.training_data) % 100 == 0:
            return True
        
        # Retrain if performance degrades
        if self.performance.accuracy < 0.55:  # Below 55% accuracy
            return True
        
        # Retrain if not trained recently (24 hours)
        if self.last_training_time:
            time_since_training = datetime.now() - self.last_training_time
            if time_since_training > timedelta(hours=24):
                return True
        
        return False
    
    def train_models(self):
        """Train ML models with cross-validation and feature selection"""
        
        if len(self.training_data) < self.min_training_samples:
            logger.warning(f"Not enough training data: {len(self.training_data)}")
            return
        
        try:
            logger.info(f"Training models with {len(self.training_data)} samples")
            
            # Prepare training data
            X = np.array([d['features'] for d in self.training_data])
            y_success = np.array([d['success'] for d in self.training_data])
            y_profit = np.array([d['profit_ratio'] for d in self.training_data])
            
            # Remove any NaN or Inf values
            valid_mask = ~(np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1))
            X = X[valid_mask]
            y_success = y_success[valid_mask]
            y_profit = y_profit[valid_mask]
            
            if len(X) < self.min_training_samples:
                logger.warning("Too many invalid samples removed")
                return
            
            # Split data for testing
            X_train, X_test, y_success_train, y_success_test = train_test_split(
                X, y_success, test_size=0.2, random_state=42, stratify=y_success
            )
            _, _, y_profit_train, y_profit_test = train_test_split(
                X, y_profit, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Feature selection
            self.feature_selector = SelectKBest(f_classif, k=min(15, X.shape[1]))
            X_train_selected = self.feature_selector.fit_transform(X_train_scaled, y_success_train)
            X_test_selected = self.feature_selector.transform(X_test_scaled)
            
            # Train anomaly detector
            self.anomaly_detector = IsolationForest(
                contamination=self.outlier_threshold,
                random_state=42
            )
            self.anomaly_detector.fit(X_train_scaled)
            
            # Train success classifier with cross-validation
            self.success_classifier = RandomForestClassifier(
                n_estimators=200,  # More trees for better confidence intervals
                max_depth=12,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            )
            
            # Cross-validation
            cv_scores = cross_val_score(
                self.success_classifier, 
                X_train_selected, 
                y_success_train,
                cv=KFold(n_splits=5, shuffle=True, random_state=42),
                scoring='accuracy'
            )
            
            logger.info(f"Cross-validation scores: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            
            # Train final model
            self.success_classifier.fit(X_train_selected, y_success_train)
            
            # Train profit regressor
            self.profit_regressor = GradientBoostingRegressor(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.05,
                min_samples_split=10,
                min_samples_leaf=5,
                subsample=0.8,
                random_state=42
            )
            self.profit_regressor.fit(X_train_selected, y_profit_train)
            
            # Calculate feature importance
            feature_mask = self.feature_selector.get_support()
            selected_feature_indices = np.where(feature_mask)[0]
            
            feature_names = [
                'zone_strength', 'volume_ratio', 'rejection_speed', 'institutional_interest',
                'confluence_count', 'timeframes_visible', 'market_structure', 'order_flow',
                'zone_age', 'test_count', 'distance_poc', 'in_value_area', 'trend_aligned',
                'volatility_ratio', 'liquidity_score', 'time_of_day', 'day_of_week',
                'momentum', 'volume_momentum', 'vol_percentile', 'spread_ratio',
                'market_efficiency', 'btc_correlation', 'zone_quality', 'risk_sentiment'
            ]
            
            self.feature_importance = {}
            for idx, importance in zip(selected_feature_indices, self.success_classifier.feature_importances_):
                if idx < len(feature_names):
                    self.feature_importance[feature_names[idx]] = importance
            
            # Evaluate models
            y_pred = self.success_classifier.predict(X_test_selected)
            
            # Update performance metrics
            self.performance.accuracy = accuracy_score(y_success_test, y_pred)
            self.performance.precision = precision_score(y_success_test, y_pred, zero_division=0)
            self.performance.recall = recall_score(y_success_test, y_pred, zero_division=0)
            self.performance.f1_score = f1_score(y_success_test, y_pred, zero_division=0)
            self.performance.last_updated = datetime.now()
            
            # Log results
            logger.info(f"Model trained successfully:")
            logger.info(f"  Accuracy: {self.performance.accuracy:.3f}")
            logger.info(f"  Precision: {self.performance.precision:.3f}")
            logger.info(f"  Recall: {self.performance.recall:.3f}")
            logger.info(f"  F1 Score: {self.performance.f1_score:.3f}")
            logger.info(f"  Top features: {list(sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:5])}")
            
            self.model_trained = True
            self.last_training_time = datetime.now()
            self.model_version = f"2.{len(self.training_data)}.{int(self.performance.accuracy * 100)}"
            
        except Exception as e:
            logger.error(f"Model training failed: {e}", exc_info=True)
    
    def _track_prediction(self, result: PredictionResult, zone: 'EnhancedZone', market_data: Dict):
        """Track prediction for performance monitoring"""
        
        prediction_record = {
            'timestamp': datetime.now(),
            'symbol': market_data.get('symbol', 'unknown'),
            'success_prob': result.success_probability,
            'confidence': result.prediction_confidence,
            'expected_profit': result.expected_profit,
            'zone_score': zone.composite_score,
            'market_regime': market_data.get('regime', 'unknown')
        }
        
        self.prediction_history.append(prediction_record)
        
        # Keep only recent predictions (last 1000)
        if len(self.prediction_history) > 1000:
            self.prediction_history = self.prediction_history[-1000:]
        
        self.performance.total_predictions += 1
    
    def update_prediction_outcome(self, prediction_id: str, actual_outcome: bool, actual_profit: float):
        """Update prediction with actual outcome for performance tracking"""
        
        # Match prediction and update performance
        if actual_outcome:
            self.performance.correct_predictions += 1
        
        # Update accuracy
        if self.performance.total_predictions > 0:
            self.performance.accuracy = (
                self.performance.correct_predictions / self.performance.total_predictions
            )
    
    def get_model_diagnostics(self) -> Dict:
        """Get comprehensive model diagnostics"""
        
        return {
            'model_version': self.model_version,
            'model_trained': self.model_trained,
            'training_samples': len(self.training_data),
            'last_training': self.last_training_time.isoformat() if self.last_training_time else None,
            'performance': {
                'accuracy': self.performance.accuracy,
                'precision': self.performance.precision,
                'recall': self.performance.recall,
                'f1_score': self.performance.f1_score,
                'total_predictions': self.performance.total_predictions,
                'correct_predictions': self.performance.correct_predictions
            },
            'feature_importance': dict(sorted(
                self.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]) if self.feature_importance else {},
            'selected_features': len(self.selected_features) if self.feature_selector else 0,
            'warnings': []
        }
    
    def save_models(self, path: str):
        """Save all models and metadata"""
        if self.model_trained:
            import os
            os.makedirs(path, exist_ok=True)
            
            # Save models
            joblib.dump(self.success_classifier, f"{path}/success_classifier_v2.pkl")
            joblib.dump(self.profit_regressor, f"{path}/profit_regressor_v2.pkl")
            joblib.dump(self.anomaly_detector, f"{path}/anomaly_detector.pkl")
            joblib.dump(self.scaler, f"{path}/scaler_v2.pkl")
            joblib.dump(self.feature_selector, f"{path}/feature_selector.pkl")
            
            # Save metadata
            metadata = {
                'version': self.model_version,
                'training_data': self.training_data,
                'feature_importance': self.feature_importance,
                'performance': self.performance.__dict__,
                'last_training': self.last_training_time
            }
            joblib.dump(metadata, f"{path}/metadata_v2.pkl")
            
            logger.info(f"Enhanced models saved to {path}")
    
    def load_models(self, path: str):
        """Load models and metadata"""
        import os
        
        try:
            # Check for v2 models first
            v2_files = [
                f"{path}/success_classifier_v2.pkl",
                f"{path}/profit_regressor_v2.pkl",
                f"{path}/anomaly_detector.pkl",
                f"{path}/scaler_v2.pkl",
                f"{path}/metadata_v2.pkl"
            ]
            
            if all(os.path.exists(f) for f in v2_files):
                self.success_classifier = joblib.load(v2_files[0])
                self.profit_regressor = joblib.load(v2_files[1])
                self.anomaly_detector = joblib.load(v2_files[2])
                self.scaler = joblib.load(v2_files[3])
                
                # Load metadata
                metadata = joblib.load(v2_files[4])
                self.training_data = metadata.get('training_data', [])
                self.feature_importance = metadata.get('feature_importance', {})
                self.model_version = metadata.get('version', '2.0.0')
                self.last_training_time = metadata.get('last_training')
                
                # Load performance
                if 'performance' in metadata:
                    for key, value in metadata['performance'].items():
                        if hasattr(self.performance, key):
                            setattr(self.performance, key, value)
                
                # Try to load feature selector
                selector_path = f"{path}/feature_selector.pkl"
                if os.path.exists(selector_path):
                    self.feature_selector = joblib.load(selector_path)
                
                self.model_trained = True
                logger.info(f"Enhanced models v{self.model_version} loaded from {path}")
                
            else:
                # Try to load v1 models for backward compatibility
                from .ml_predictor import MLPredictor
                v1_predictor = MLPredictor()
                v1_predictor.load_models(path)
                
                if v1_predictor.model_trained:
                    # Migrate v1 models
                    self.success_classifier = v1_predictor.success_classifier
                    self.profit_regressor = v1_predictor.profit_regressor
                    self.scaler = v1_predictor.scaler
                    self.training_data = v1_predictor.training_data
                    self.model_trained = True
                    self.model_version = "1.0.0-migrated"
                    logger.info("Migrated v1 models to enhanced predictor")
                else:
                    logger.info("No models found, will train new enhanced models")
                    self.model_trained = False
                    
        except Exception as e:
            logger.warning(f"Could not load models: {e}")
            self.model_trained = False

# Global instance
enhanced_ml_predictor = EnhancedMLPredictor()