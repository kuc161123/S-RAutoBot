"""
Enhanced Feature Engineering for ML Evolution
Adds rich market context without breaking existing systems
"""
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple
import psycopg2
from psycopg2.extras import RealDictCursor
import os
import redis
import json

logger = logging.getLogger(__name__)

class EnhancedFeatureEngine:
    """
    Calculates advanced features for ML Evolution
    Designed to supplement existing features, not replace them
    """
    
    def __init__(self):
        self.db_conn = None
        self.redis_client = None
        self.btc_cache = {}  # Cache BTC data for efficiency
        self.volume_profiles = {}  # Cache volume profiles
        self.failed_signals = {}  # Track near-miss signals
        
        self._init_connections()
    
    def _init_connections(self):
        """Initialize database connections"""
        # PostgreSQL
        try:
            db_url = os.getenv('DATABASE_URL')
            if db_url:
                self.db_conn = psycopg2.connect(db_url, cursor_factory=RealDictCursor)
                logger.info("Enhanced features connected to PostgreSQL")
        except Exception as e:
            logger.warning(f"Enhanced features PostgreSQL failed: {e}")
        
        # Redis for caching
        try:
            redis_url = os.getenv('REDIS_URL')
            if redis_url:
                self.redis_client = redis.from_url(redis_url, decode_responses=True)
                logger.info("Enhanced features connected to Redis")
        except Exception as e:
            logger.warning(f"Enhanced features Redis failed: {e}")
    
    def enhance_features(self, symbol: str, df: pd.DataFrame, current_features: Dict, 
                        btc_price: Optional[float] = None) -> Dict:
        """
        Add enhanced features to existing feature set
        
        Args:
            symbol: Trading symbol
            df: Recent candle data
            current_features: Existing features from strategy
            btc_price: Current BTC price if available
        
        Returns:
            Enhanced feature dictionary (includes original features)
        """
        try:
            enhanced = current_features.copy()
            
            # 1. BTC Correlation Features
            if btc_price and symbol != 'BTCUSDT':
                btc_features = self._calculate_btc_correlation(symbol, df, btc_price)
                enhanced.update(btc_features)
            
            # 2. Volume Profile Features
            volume_features = self._calculate_volume_profile(symbol, df)
            enhanced.update(volume_features)
            
            # 3. Market Microstructure
            micro_features = self._calculate_microstructure(df)
            enhanced.update(micro_features)
            
            # 4. Temporal Patterns
            temporal_features = self._calculate_temporal_patterns(symbol, datetime.now())
            enhanced.update(temporal_features)
            
            # 5. Volatility Regime
            volatility_features = self._calculate_volatility_regime(df)
            enhanced.update(volatility_features)
            
            # 6. Failed Signal Context
            failed_signal_features = self._get_failed_signal_context(symbol, df['close'].iloc[-1])
            enhanced.update(failed_signal_features)
            
            return enhanced
            
        except Exception as e:
            logger.warning(f"Error enhancing features for {symbol}: {e}")
            # Return original features if enhancement fails
            return current_features
    
    def _calculate_btc_correlation(self, symbol: str, df: pd.DataFrame, btc_price: float) -> Dict:
        """Calculate BTC correlation and relative strength"""
        features = {}
        
        try:
            # Get BTC data from cache or fetch
            btc_data = self._get_btc_data()
            if not btc_data or len(btc_data) < 20:
                return features
            
            # Current symbol returns
            symbol_returns = df['close'].pct_change().fillna(0)
            
            # BTC returns for same period
            btc_df = pd.DataFrame(btc_data)
            btc_returns = btc_df['close'].pct_change().fillna(0)
            
            # Correlation over different periods
            if len(symbol_returns) >= 20 and len(btc_returns) >= 20:
                features['btc_corr_20'] = symbol_returns.tail(20).corr(btc_returns.tail(20))
            
            if len(symbol_returns) >= 60 and len(btc_returns) >= 60:
                features['btc_corr_60'] = symbol_returns.tail(60).corr(btc_returns.tail(60))
            
            # Relative strength
            if len(symbol_returns) >= 10 and len(btc_returns) >= 10:
                symbol_perf_10 = (df['close'].iloc[-1] / df['close'].iloc[-10] - 1) * 100
                btc_perf_10 = (btc_df['close'].iloc[-1] / btc_df['close'].iloc[-10] - 1) * 100
                features['relative_strength_btc'] = symbol_perf_10 - btc_perf_10
            
            # BTC trend alignment
            btc_sma_20 = btc_df['close'].rolling(20).mean().iloc[-1]
            features['btc_trend_up'] = 1 if btc_price > btc_sma_20 else 0
            
            # Recent BTC momentum
            if len(btc_df) >= 5:
                btc_momentum_5 = (btc_df['close'].iloc[-1] / btc_df['close'].iloc[-5] - 1) * 100
                features['btc_momentum_5'] = btc_momentum_5
            
        except Exception as e:
            logger.debug(f"BTC correlation calculation error: {e}")
        
        return features
    
    def _calculate_volume_profile(self, symbol: str, df: pd.DataFrame) -> Dict:
        """Calculate volume profile features"""
        features = {}
        
        try:
            # Volume nodes - where most trading happened
            if len(df) >= 24:  # Need at least 24 candles
                # Discretize price into bins
                price_bins = pd.cut(df['close'].tail(24), bins=10)
                volume_by_price = df.groupby(price_bins)['volume'].sum()
                
                # Find high volume node (HVN)
                hvn_idx = volume_by_price.idxmax()
                if pd.notna(hvn_idx):
                    hvn_price = hvn_idx.mid  # Middle of the bin
                    current_price = df['close'].iloc[-1]
                    
                    # Distance from HVN as percentage
                    features['distance_from_hvn'] = ((current_price - hvn_price) / hvn_price) * 100
                    
                    # Are we above or below HVN?
                    features['above_hvn'] = 1 if current_price > hvn_price else 0
                
                # Volume concentration
                total_volume = df['volume'].tail(24).sum()
                if total_volume > 0:
                    top_2_volume = volume_by_price.nlargest(2).sum()
                    features['volume_concentration'] = top_2_volume / total_volume
            
            # VWAP calculation
            if len(df) >= 20:
                typical_price = (df['high'] + df['low'] + df['close']) / 3
                vwap = (typical_price * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
                current_price = df['close'].iloc[-1]
                features['vwap_distance'] = ((current_price - vwap.iloc[-1]) / vwap.iloc[-1]) * 100
                features['above_vwap'] = 1 if current_price > vwap.iloc[-1] else 0
            
        except Exception as e:
            logger.debug(f"Volume profile calculation error: {e}")
        
        return features
    
    def _calculate_microstructure(self, df: pd.DataFrame) -> Dict:
        """Calculate market microstructure features"""
        features = {}
        
        try:
            # Order flow imbalance proxy (using candle data)
            if len(df) >= 5:
                recent = df.tail(5)
                
                # Buy pressure: close near high
                buy_pressure = ((recent['close'] - recent['low']) / 
                               (recent['high'] - recent['low'])).fillna(0.5).mean()
                features['buy_pressure'] = buy_pressure
                
                # Volume-weighted price movement
                price_changes = recent['close'].pct_change().fillna(0)
                volume_weighted_move = (price_changes * recent['volume']).sum() / recent['volume'].sum()
                features['volume_weighted_direction'] = volume_weighted_move * 100
                
                # Acceleration (is momentum increasing?)
                if len(df) >= 10:
                    returns = df['close'].pct_change().fillna(0)
                    accel = returns.tail(5).mean() - returns.tail(10).head(5).mean()
                    features['price_acceleration'] = accel * 100
            
            # Spread proxy (high-low range)
            if len(df) >= 20:
                avg_range = ((df['high'] - df['low']) / df['close']).tail(20).mean()
                current_range = (df['high'].iloc[-1] - df['low'].iloc[-1]) / df['close'].iloc[-1]
                features['spread_ratio'] = current_range / avg_range if avg_range > 0 else 1
            
        except Exception as e:
            logger.debug(f"Microstructure calculation error: {e}")
        
        return features
    
    def _calculate_temporal_patterns(self, symbol: str, current_time: datetime) -> Dict:
        """Calculate time-based pattern features"""
        features = {}
        
        try:
            # Day of week (0 = Monday)
            features['day_of_week'] = current_time.weekday()
            features['is_weekend'] = 1 if current_time.weekday() >= 5 else 0
            
            # Month-end effects (last 3 days of month)
            days_in_month = pd.Timestamp(current_time).days_in_month
            day_of_month = current_time.day
            features['month_end'] = 1 if day_of_month >= days_in_month - 2 else 0
            
            # Options expiry (typically 3rd Friday)
            # Simplified: check if it's Friday and week 3
            features['near_expiry'] = 1 if (current_time.weekday() == 4 and 
                                           15 <= current_time.day <= 21) else 0
            
            # Query historical performance for this hour/day if PostgreSQL available
            if self.db_conn:
                try:
                    with self.db_conn.cursor() as cur:
                        # Win rate for this hour and day of week
                        cur.execute("""
                            SELECT 
                                COUNT(CASE WHEN pnl_percent > 0 THEN 1 END)::float / 
                                NULLIF(COUNT(*), 0) as win_rate
                            FROM trades
                            WHERE symbol = %s
                            AND EXTRACT(hour FROM entry_time) = %s
                            AND EXTRACT(dow FROM entry_time) = %s
                            AND exit_time IS NOT NULL
                        """, (symbol, current_time.hour, current_time.weekday()))
                        
                        result = cur.fetchone()
                        if result and result['win_rate'] is not None:
                            features['historical_hour_day_winrate'] = float(result['win_rate'])
                        
                except Exception as e:
                    logger.debug(f"Temporal pattern DB query error: {e}")
            
        except Exception as e:
            logger.debug(f"Temporal pattern calculation error: {e}")
        
        return features
    
    def _calculate_volatility_regime(self, df: pd.DataFrame) -> Dict:
        """Calculate volatility regime features"""
        features = {}
        
        try:
            # ATR-based volatility
            if len(df) >= 100:
                # Current vs historical volatility
                atr = self._calculate_atr(df, 14)
                current_vol = atr.iloc[-1]
                
                # Volatility percentiles
                vol_pct_30 = (current_vol > atr.tail(30)).mean()
                vol_pct_100 = (current_vol > atr.tail(100)).mean()
                
                features['volatility_percentile_30'] = vol_pct_30
                features['volatility_percentile_100'] = vol_pct_100
                
                # Volatility trend (expanding or contracting)
                if len(atr) >= 20:
                    vol_sma_5 = atr.rolling(5).mean().iloc[-1]
                    vol_sma_20 = atr.rolling(20).mean().iloc[-1]
                    features['volatility_expanding'] = 1 if vol_sma_5 > vol_sma_20 else 0
                    features['volatility_ratio'] = vol_sma_5 / vol_sma_20 if vol_sma_20 > 0 else 1
                
                # Volatility spike detection
                if len(atr) >= 10:
                    vol_mean = atr.tail(10).mean()
                    vol_std = atr.tail(10).std()
                    if vol_std > 0:
                        vol_zscore = (current_vol - vol_mean) / vol_std
                        features['volatility_spike'] = 1 if vol_zscore > 2 else 0
                        features['volatility_zscore'] = vol_zscore
            
        except Exception as e:
            logger.debug(f"Volatility regime calculation error: {e}")
        
        return features
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def _get_failed_signal_context(self, symbol: str, current_price: float) -> Dict:
        """Get context about failed signals near current price"""
        features = {}
        
        try:
            # Check recent failed signals from cache
            cache_key = f"failed_signals:{symbol}"
            if self.redis_client:
                failed_data = self.redis_client.get(cache_key)
                if failed_data:
                    failed_signals = json.loads(failed_data)
                    
                    # Count near misses in last hour
                    one_hour_ago = (datetime.now() - timedelta(hours=1)).isoformat()
                    recent_fails = [s for s in failed_signals 
                                   if s['timestamp'] > one_hour_ago]
                    
                    features['failed_signals_1h'] = len(recent_fails)
                    
                    # Multiple rejections at same level?
                    price_threshold = current_price * 0.002  # 0.2% range
                    same_level_fails = [s for s in recent_fails 
                                       if abs(s['price'] - current_price) < price_threshold]
                    
                    features['rejections_at_level'] = len(same_level_fails)
            
        except Exception as e:
            logger.debug(f"Failed signal context error: {e}")
        
        return features
    
    def _get_btc_data(self) -> List[Dict]:
        """Get recent BTC data from cache or database"""
        try:
            # Check cache first
            if 'btc_data' in self.btc_cache:
                cache_time = self.btc_cache.get('timestamp', 0)
                if time.time() - cache_time < 300:  # 5 min cache
                    return self.btc_cache['btc_data']
            
            # Fetch from database
            if self.db_conn:
                with self.db_conn.cursor() as cur:
                    cur.execute("""
                        SELECT timestamp, open, high, low, close, volume
                        FROM candles
                        WHERE symbol = 'BTCUSDT'
                        AND timestamp > NOW() - INTERVAL '1 day'
                        ORDER BY timestamp DESC
                        LIMIT 100
                    """)
                    
                    data = cur.fetchall()
                    if data:
                        # Cache it
                        self.btc_cache = {
                            'btc_data': data,
                            'timestamp': time.time()
                        }
                        return data
            
        except Exception as e:
            logger.debug(f"Error fetching BTC data: {e}")
        
        return []
    
    def record_failed_signal(self, symbol: str, price: float, reason: str):
        """Record a signal that almost triggered but didn't"""
        try:
            cache_key = f"failed_signals:{symbol}"
            
            failed_signal = {
                'timestamp': datetime.now().isoformat(),
                'price': price,
                'reason': reason
            }
            
            if self.redis_client:
                # Get existing
                existing = self.redis_client.get(cache_key)
                if existing:
                    signals = json.loads(existing)
                else:
                    signals = []
                
                # Add new
                signals.append(failed_signal)
                
                # Keep only last 100
                signals = signals[-100:]
                
                # Save with 24h expiry
                self.redis_client.setex(cache_key, 86400, json.dumps(signals))
                
        except Exception as e:
            logger.debug(f"Error recording failed signal: {e}")

# Global instance
_feature_engine = None

def get_feature_engine() -> EnhancedFeatureEngine:
    """Get or create global feature engine"""
    global _feature_engine
    if _feature_engine is None:
        _feature_engine = EnhancedFeatureEngine()
    return _feature_engine

import time  # Add this import at the top