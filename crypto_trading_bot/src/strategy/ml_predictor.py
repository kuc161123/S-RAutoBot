"""
Machine Learning Predictor for Supply & Demand Strategy Enhancement
Uses historical zone performance to predict future success probability
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import joblib
import structlog
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

logger = structlog.get_logger(__name__)

@dataclass
class ZoneFeatures:
    """Features for ML prediction"""
    zone_strength: float
    volume_ratio: float
    rejection_speed: float
    institutional_interest: float
    confluence_count: int
    timeframes_visible: int
    market_structure: int  # Encoded
    order_flow_imbalance: int  # Encoded
    zone_age_hours: float
    test_count: int
    distance_from_poc: float
    in_value_area: bool
    trend_alignment: bool
    volatility_ratio: float
    liquidity_score: float
    time_of_day: int  # Hour
    day_of_week: int
    
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
            self.day_of_week
        ])

class MLPredictor:
    """Machine Learning predictor for zone success probability"""
    
    def __init__(self):
        self.success_classifier = None
        self.profit_regressor = None
        self.scaler = StandardScaler()
        self.feature_importance = {}
        self.model_trained = False
        self.training_data = []
        self.min_training_samples = 100
        
    def extract_features(self, zone: 'EnhancedZone', market_data: Dict) -> ZoneFeatures:
        """Extract ML features from zone and market data"""
        
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
        
        features = ZoneFeatures(
            zone_strength=zone.strength_score / 100,
            volume_ratio=zone.volume_profile.total_volume / market_data.get('avg_volume', 1),
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
            zone_age_hours=zone.zone_age_hours,
            test_count=zone.test_count,
            distance_from_poc=abs(zone.volume_profile.poc - 
                                (zone.upper_bound + zone.lower_bound) / 2) / zone.volume_profile.poc,
            in_value_area=(zone.lower_bound <= zone.volume_profile.vah and 
                          zone.upper_bound >= zone.volume_profile.val),
            trend_alignment='trend_aligned' in zone.confluence_factors,
            volatility_ratio=market_data.get('volatility', 1.0),
            liquidity_score=zone.liquidity_pool / market_data.get('avg_liquidity', zone.liquidity_pool),
            time_of_day=zone.formation_time.hour,
            day_of_week=zone.formation_time.weekday()
        )
        
        return features
    
    def predict_zone_success(
        self,
        zone: 'EnhancedZone',
        market_data: Dict
    ) -> Tuple[float, float]:
        """
        Predict zone success probability and expected profit
        Returns: (success_probability, expected_profit_ratio)
        """
        
        if not self.model_trained:
            # Return baseline predictions
            base_prob = zone.composite_score / 100
            expected_profit = 1.5 if base_prob > 0.6 else 1.0
            return base_prob, expected_profit
        
        try:
            # Extract features
            features = self.extract_features(zone, market_data)
            X = features.to_array().reshape(1, -1)
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Predict success probability
            success_prob = self.success_classifier.predict_proba(X_scaled)[0, 1]
            
            # Predict expected profit ratio
            expected_profit = self.profit_regressor.predict(X_scaled)[0]
            
            # Combine with rule-based confidence
            combined_prob = (success_prob * 0.7 + zone.composite_score / 100 * 0.3)
            
            return combined_prob, max(0.5, expected_profit)
            
        except Exception as e:
            logger.error(f"ML prediction failed: {e}")
            # Fallback to rule-based
            return zone.composite_score / 100, 1.5
    
    def add_training_sample(
        self,
        zone: 'EnhancedZone',
        market_data: Dict,
        outcome: bool,
        profit_ratio: float
    ):
        """Add a training sample from actual trade outcome"""
        
        features = self.extract_features(zone, market_data)
        
        self.training_data.append({
            'features': features.to_array(),
            'success': outcome,
            'profit_ratio': profit_ratio,
            'timestamp': datetime.now()
        })
        
        # Retrain if we have enough new samples
        if len(self.training_data) >= self.min_training_samples:
            if not self.model_trained or len(self.training_data) % 50 == 0:
                self.train_models()
    
    def train_models(self):
        """Train ML models on collected data"""
        
        if len(self.training_data) < self.min_training_samples:
            logger.warning(f"Not enough training data: {len(self.training_data)}")
            return
        
        try:
            # Prepare training data
            X = np.array([d['features'] for d in self.training_data])
            y_success = np.array([d['success'] for d in self.training_data])
            y_profit = np.array([d['profit_ratio'] for d in self.training_data])
            
            # Split data
            X_train, X_test, y_success_train, y_success_test = train_test_split(
                X, y_success, test_size=0.2, random_state=42
            )
            _, _, y_profit_train, y_profit_test = train_test_split(
                X, y_profit, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train success classifier
            self.success_classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42
            )
            self.success_classifier.fit(X_train_scaled, y_success_train)
            
            # Train profit regressor
            self.profit_regressor = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
            self.profit_regressor.fit(X_train_scaled, y_profit_train)
            
            # Calculate feature importance
            self.feature_importance = {
                'zone_strength': self.success_classifier.feature_importances_[0],
                'volume_ratio': self.success_classifier.feature_importances_[1],
                'rejection_speed': self.success_classifier.feature_importances_[2],
                'institutional_interest': self.success_classifier.feature_importances_[3],
                'confluence_count': self.success_classifier.feature_importances_[4],
                'timeframes_visible': self.success_classifier.feature_importances_[5],
                'market_structure': self.success_classifier.feature_importances_[6],
                'order_flow': self.success_classifier.feature_importances_[7]
            }
            
            # Evaluate models
            train_score = self.success_classifier.score(X_train_scaled, y_success_train)
            test_score = self.success_classifier.score(X_test_scaled, y_success_test)
            
            logger.info(f"ML models trained - Train accuracy: {train_score:.2f}, Test accuracy: {test_score:.2f}")
            logger.info(f"Top features: {sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]}")
            
            self.model_trained = True
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
    
    def save_models(self, path: str):
        """Save trained models to disk"""
        if self.model_trained:
            joblib.dump(self.success_classifier, f"{path}/success_classifier.pkl")
            joblib.dump(self.profit_regressor, f"{path}/profit_regressor.pkl")
            joblib.dump(self.scaler, f"{path}/scaler.pkl")
            joblib.dump(self.training_data, f"{path}/training_data.pkl")
            logger.info(f"Models saved to {path}")
    
    def load_models(self, path: str):
        """Load trained models from disk"""
        import os
        
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        
        try:
            # Check if model files exist
            model_files = [
                f"{path}/success_classifier.pkl",
                f"{path}/profit_regressor.pkl",
                f"{path}/scaler.pkl",
                f"{path}/training_data.pkl"
            ]
            
            if all(os.path.exists(f) for f in model_files):
                self.success_classifier = joblib.load(model_files[0])
                self.profit_regressor = joblib.load(model_files[1])
                self.scaler = joblib.load(model_files[2])
                self.training_data = joblib.load(model_files[3])
                self.model_trained = True
                logger.info(f"Models loaded successfully from {path}")
            else:
                logger.info("No existing models found. Will train new models when sufficient data is collected")
                self.model_trained = False
        except Exception as e:
            logger.warning(f"Could not load models (will train new ones): {e}")
            self.model_trained = False
    
    def get_confidence_adjusted_signal(
        self,
        base_signal: Dict,
        zone: 'EnhancedZone',
        market_data: Dict
    ) -> Dict:
        """Adjust trading signal based on ML predictions"""
        
        # Get ML predictions
        success_prob, expected_profit = self.predict_zone_success(zone, market_data)
        
        # Adjust position size based on confidence
        if success_prob > 0.7:
            position_multiplier = 1.2
        elif success_prob > 0.6:
            position_multiplier = 1.0
        elif success_prob > 0.5:
            position_multiplier = 0.8
        else:
            position_multiplier = 0.5
        
        # Adjust take profit based on expected profit
        tp_multiplier = min(2.0, expected_profit)
        
        # Create adjusted signal
        adjusted_signal = base_signal.copy()
        adjusted_signal['ml_confidence'] = success_prob
        adjusted_signal['expected_profit'] = expected_profit
        adjusted_signal['position_multiplier'] = position_multiplier
        
        # Adjust take profit levels
        if base_signal['type'] == 'BUY':
            adjusted_signal['take_profit_1'] = base_signal['entry_price'] * (1 + 0.01 * tp_multiplier)
            adjusted_signal['take_profit_2'] = base_signal['entry_price'] * (1 + 0.02 * tp_multiplier)
        else:  # SELL
            adjusted_signal['take_profit_1'] = base_signal['entry_price'] * (1 - 0.01 * tp_multiplier)
            adjusted_signal['take_profit_2'] = base_signal['entry_price'] * (1 - 0.02 * tp_multiplier)
        
        return adjusted_signal

class PatternRecognition:
    """Advanced pattern recognition for zone validation"""
    
    @staticmethod
    def detect_accumulation_distribution(df: pd.DataFrame, zone: 'EnhancedZone') -> str:
        """Detect Wyckoff accumulation/distribution patterns"""
        
        # Get price action around zone
        zone_mask = (df['low'] <= zone.upper_bound) & (df['high'] >= zone.lower_bound)
        zone_data = df[zone_mask]
        
        if len(zone_data) < 5:
            return "insufficient_data"
        
        # Check for accumulation signs (for demand zones)
        if zone.zone_type == 'demand':
            # Spring pattern - false breakdown below support
            if any(zone_data['low'] < zone.lower_bound * 0.99) and \
               zone_data.iloc[-1]['close'] > zone.upper_bound:
                return "spring_pattern"
            
            # Absorption - high volume with little price movement
            avg_volume = df['volume'].mean()
            if zone_data['volume'].mean() > avg_volume * 1.5 and \
               (zone_data['high'].max() - zone_data['low'].min()) < zone_data['atr'].mean():
                return "accumulation"
        
        # Check for distribution signs (for supply zones)
        elif zone.zone_type == 'supply':
            # Upthrust pattern - false breakout above resistance
            if any(zone_data['high'] > zone.upper_bound * 1.01) and \
               zone_data.iloc[-1]['close'] < zone.lower_bound:
                return "upthrust_pattern"
            
            # Distribution - high volume with limited upside
            avg_volume = df['volume'].mean()
            if zone_data['volume'].mean() > avg_volume * 1.5 and \
               (zone_data['high'].max() - zone_data['low'].min()) < zone_data['atr'].mean():
                return "distribution"
        
        return "neutral"
    
    @staticmethod
    def detect_order_block(df: pd.DataFrame, i: int) -> bool:
        """Detect institutional order blocks"""
        
        if i < 3 or i > len(df) - 3:
            return False
        
        # Check for imbalance (gap or strong move)
        prev_high = df.iloc[i-1]['high']
        prev_low = df.iloc[i-1]['low']
        curr_open = df.iloc[i]['open']
        curr_close = df.iloc[i]['close']
        next_low = df.iloc[i+1]['low']
        next_high = df.iloc[i+1]['high']
        
        # Bullish order block
        if curr_close > curr_open:  # Bullish candle
            # Check for imbalance after
            if next_low > prev_high:  # Gap up
                return True
            # Check for strong move
            if (curr_close - curr_open) > df.iloc[i]['atr'] * 2:
                return True
        
        # Bearish order block
        elif curr_close < curr_open:  # Bearish candle
            # Check for imbalance after
            if next_high < prev_low:  # Gap down
                return True
            # Check for strong move
            if (curr_open - curr_close) > df.iloc[i]['atr'] * 2:
                return True
        
        return False

# Global ML predictor instance
ml_predictor = MLPredictor()