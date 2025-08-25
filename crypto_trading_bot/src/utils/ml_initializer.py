"""
ML Model Initializer
Helps bootstrap ML training with synthetic historical data or accelerated learning
"""
import numpy as np
import structlog
from datetime import datetime, timedelta
from typing import Dict, List

logger = structlog.get_logger(__name__)

class MLInitializer:
    """Initialize ML models with bootstrap data"""
    
    @staticmethod
    async def bootstrap_ml_training():
        """
        Bootstrap ML training with synthetic data to get models started
        This helps the bot start making ML-enhanced decisions faster
        """
        from ..strategy.enhanced_ml_predictor import enhanced_ml_predictor
        from ..strategy.ml_predictor import ml_predictor
        from ..strategy.advanced_supply_demand import EnhancedZone, ZoneType
        
        logger.info("Starting ML bootstrap process...")
        
        # Generate synthetic training data based on common patterns
        synthetic_samples = []
        
        # High confidence winning patterns
        for i in range(30):
            zone = MLInitializer._create_synthetic_zone(
                zone_type=ZoneType.DEMAND if i % 2 == 0 else ZoneType.SUPPLY,
                strength=80 + np.random.uniform(-10, 10),
                touches=np.random.randint(1, 3),
                age_hours=np.random.uniform(1, 48)
            )
            
            market_data = {
                'avg_volume': 1000000,
                'volatility': 1.0 + np.random.uniform(-0.3, 0.3),
                'market_structure': np.random.choice(['bullish', 'bearish', 'ranging']),
                'regime': 'trending_strong',
                'symbol': f'BTCUSDT',
                'btc_correlation': 0.8
            }
            
            # High confidence patterns typically win
            outcome = np.random.random() < 0.75  # 75% win rate for strong zones
            profit = np.random.uniform(1.2, 2.5) if outcome else np.random.uniform(-0.8, -0.3)
            
            try:
                enhanced_ml_predictor.add_training_sample(
                    zone=zone,
                    market_data=market_data,
                    outcome=outcome,
                    profit_ratio=profit
                )
            except Exception as e:
                logger.warning(f"Failed to add synthetic sample: {e}")
        
        # Medium confidence patterns
        for i in range(40):
            zone = MLInitializer._create_synthetic_zone(
                zone_type=ZoneType.DEMAND if i % 2 == 0 else ZoneType.SUPPLY,
                strength=60 + np.random.uniform(-10, 10),
                touches=np.random.randint(2, 5),
                age_hours=np.random.uniform(24, 96)
            )
            
            market_data = {
                'avg_volume': 800000,
                'volatility': 1.2 + np.random.uniform(-0.3, 0.3),
                'market_structure': np.random.choice(['ranging', 'transitioning']),
                'regime': 'ranging_wide',
                'symbol': 'ETHUSDT',
                'btc_correlation': 0.6
            }
            
            # Medium confidence patterns have mixed results
            outcome = np.random.random() < 0.55  # 55% win rate
            profit = np.random.uniform(0.8, 1.8) if outcome else np.random.uniform(-0.9, -0.4)
            
            try:
                enhanced_ml_predictor.add_training_sample(
                    zone=zone,
                    market_data=market_data,
                    outcome=outcome,
                    profit_ratio=profit
                )
            except Exception as e:
                logger.warning(f"Failed to add synthetic sample: {e}")
        
        # Low confidence losing patterns
        for i in range(30):
            zone = MLInitializer._create_synthetic_zone(
                zone_type=ZoneType.DEMAND if i % 2 == 0 else ZoneType.SUPPLY,
                strength=40 + np.random.uniform(-10, 10),
                touches=np.random.randint(4, 8),
                age_hours=np.random.uniform(96, 168)
            )
            
            market_data = {
                'avg_volume': 500000,
                'volatility': 1.5 + np.random.uniform(-0.3, 0.5),
                'market_structure': 'volatile',
                'regime': 'volatile',
                'symbol': 'SOLUSDT',
                'btc_correlation': 0.4
            }
            
            # Low confidence patterns typically lose
            outcome = np.random.random() < 0.35  # 35% win rate
            profit = np.random.uniform(0.5, 1.2) if outcome else np.random.uniform(-1.2, -0.5)
            
            try:
                enhanced_ml_predictor.add_training_sample(
                    zone=zone,
                    market_data=market_data,
                    outcome=outcome,
                    profit_ratio=profit
                )
            except Exception as e:
                logger.warning(f"Failed to add synthetic sample: {e}")
        
        # Trigger training
        logger.info(f"Added {len(enhanced_ml_predictor.training_data)} synthetic samples")
        
        if len(enhanced_ml_predictor.training_data) >= enhanced_ml_predictor.min_training_samples:
            logger.info("Training enhanced ML models with bootstrap data...")
            enhanced_ml_predictor.train_models()
            
            # Also train original predictor for compatibility
            if len(ml_predictor.training_data) >= ml_predictor.min_training_samples:
                ml_predictor.train_models()
            
            logger.info("ML models trained successfully with bootstrap data")
            return True
        else:
            logger.warning("Not enough samples for training")
            return False
    
    @staticmethod
    def _create_synthetic_zone(zone_type, strength, touches, age_hours):
        """Create a synthetic zone for training"""
        from ..strategy.advanced_supply_demand import EnhancedZone, ZoneType, VolumeProfile
        
        # Create a mock volume profile
        volume_profile = VolumeProfile()
        volume_profile.total_volume = np.random.uniform(100000, 1000000)
        volume_profile.poc = 50000  # Price point of control
        volume_profile.val = 49000  # Value area low
        volume_profile.vah = 51000  # Value area high
        
        # Create the zone
        zone = EnhancedZone(
            zone_type=zone_type,
            upper_bound=51000 if zone_type == ZoneType.SUPPLY else 49500,
            lower_bound=50500 if zone_type == ZoneType.SUPPLY else 49000,
            strength_score=strength,
            formation_time=datetime.now() - timedelta(hours=age_hours),
            timeframe="15m"
        )
        
        # Set additional properties
        zone.volume_profile = volume_profile
        zone.test_count = touches
        zone.zone_age_hours = age_hours
        zone.rejection_strength = strength / 30
        zone.institutional_interest = strength * 0.8
        zone.confluence_factors = ['ma_support'] if strength > 60 else []
        zone.timeframes_visible = ['5m', '15m'] if strength > 70 else ['15m']
        zone.liquidity_pool = np.random.uniform(10000, 100000)
        zone.composite_score = strength
        zone.touches = touches
        
        return zone
    
    @staticmethod
    def get_ml_training_status() -> Dict:
        """Get current ML training status"""
        from ..strategy.enhanced_ml_predictor import enhanced_ml_predictor
        from ..strategy.ml_predictor import ml_predictor
        
        status = {
            'enhanced_ml': {
                'trained': enhanced_ml_predictor.model_trained,
                'samples': len(enhanced_ml_predictor.training_data),
                'min_required': enhanced_ml_predictor.min_training_samples,
                'version': enhanced_ml_predictor.model_version,
                'last_training': enhanced_ml_predictor.last_training_time.isoformat() if enhanced_ml_predictor.last_training_time else None,
                'performance': {
                    'accuracy': enhanced_ml_predictor.performance.accuracy,
                    'precision': enhanced_ml_predictor.performance.precision,
                    'recall': enhanced_ml_predictor.performance.recall,
                    'f1_score': enhanced_ml_predictor.performance.f1_score
                } if enhanced_ml_predictor.model_trained else None
            },
            'original_ml': {
                'trained': ml_predictor.model_trained,
                'samples': len(ml_predictor.training_data),
                'min_required': ml_predictor.min_training_samples
            },
            'ready': enhanced_ml_predictor.model_trained or ml_predictor.model_trained,
            'needs_bootstrap': not enhanced_ml_predictor.model_trained and len(enhanced_ml_predictor.training_data) < 50
        }
        
        return status

# Global instance
ml_initializer = MLInitializer()