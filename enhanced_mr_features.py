"""
Enhanced Feature Engineering for Mean Reversion Strategy
Provides 30+ specialized features for ranging market analysis
"""
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, Optional, Tuple
from scipy import stats
from scipy.signal import find_peaks
import ta

logger = logging.getLogger(__name__)

def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Calculate Average True Range"""
    high = df['high']
    low = df['low']
    close = df['close']
    prev_close = close.shift()

    tr = np.maximum(high - low,
         np.maximum(abs(high - prev_close), abs(low - prev_close)))
    return tr.rolling(n).mean()

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def _stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
    """Calculate Stochastic Oscillator"""
    low_min = df['low'].rolling(window=k_period).min()
    high_max = df['high'].rolling(window=k_period).max()
    k_percent = 100 * ((df['close'] - low_min) / (high_max - low_min))
    d_percent = k_percent.rolling(window=d_period).mean()
    return k_percent, d_percent

def _williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Williams %R"""
    high_max = df['high'].rolling(window=period).max()
    low_min = df['low'].rolling(window=period).min()
    return -100 * ((high_max - df['close']) / (high_max - low_min))

def _cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Calculate Commodity Channel Index"""
    tp = (df['high'] + df['low'] + df['close']) / 3
    ma = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
    return (tp - ma) / (0.015 * mad)

def _mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Money Flow Index"""
    tp = (df['high'] + df['low'] + df['close']) / 3
    raw_money_flow = tp * df['volume']

    positive_flow = np.where(tp > tp.shift(1), raw_money_flow, 0)
    negative_flow = np.where(tp < tp.shift(1), raw_money_flow, 0)

    positive_flow_sum = pd.Series(positive_flow).rolling(window=period).sum()
    negative_flow_sum = pd.Series(negative_flow).rolling(window=period).sum()

    money_ratio = positive_flow_sum / negative_flow_sum
    return 100 - (100 / (1 + money_ratio))

def _detect_range_boundaries(df: pd.DataFrame, lookback: int = 50) -> Dict:
    """Advanced range boundary detection using multiple methods"""
    if len(df) < lookback:
        return {'upper': None, 'lower': None, 'confidence': 0.0, 'age': 0}

    recent_df = df.tail(lookback)
    high_prices = recent_df['high']
    low_prices = recent_df['low']

    # Method 1: Pivot-based levels
    high_peaks, _ = find_peaks(high_prices, distance=5, prominence=high_prices.std() * 0.5)
    low_peaks, _ = find_peaks(-low_prices, distance=5, prominence=low_prices.std() * 0.5)

    if len(high_peaks) >= 2 and len(low_peaks) >= 2:
        # Get most recent significant levels
        recent_highs = high_prices.iloc[high_peaks[-2:]]
        recent_lows = low_prices.iloc[low_peaks[-2:]]

        upper_bound = recent_highs.mean()
        lower_bound = recent_lows.mean()

        # Calculate confidence based on level consistency
        upper_std = recent_highs.std()
        lower_std = recent_lows.std()
        avg_range = upper_bound - lower_bound

        # High confidence if levels are consistent (low std relative to range)
        if avg_range > 0:
            confidence = 1.0 - ((upper_std + lower_std) / avg_range)
            confidence = max(0.0, min(1.0, confidence))
        else:
            confidence = 0.0

        # Estimate range age (candles since range formation)
        range_age = min(len(high_peaks), len(low_peaks)) * 5  # Approximate

        return {
            'upper': float(upper_bound),
            'lower': float(lower_bound),
            'confidence': float(confidence),
            'age': int(range_age)
        }

    return {'upper': None, 'lower': None, 'confidence': 0.0, 'age': 0}

def calculate_enhanced_mr_features(df: pd.DataFrame, signal_data: dict, symbol: str = "UNKNOWN") -> Dict:
    """
    Calculate comprehensive mean reversion features

    Returns 30+ features optimized for ranging market analysis
    """
    if len(df) < 50:
        return _get_default_enhanced_features()

    try:
        # Basic data
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        current_price = close.iloc[-1]
        atr = _atr(df)
        current_atr = atr.iloc[-1] if len(atr) > 0 else 0.1

        features = {}

        # ========== RANGE CHARACTERISTICS (12 features) ==========
        range_info = _detect_range_boundaries(df)

        # Range structure
        if range_info['upper'] and range_info['lower']:
            range_width = range_info['upper'] - range_info['lower']
            features['range_width_atr'] = float(range_width / current_atr) if current_atr > 0 else 2.0
            features['range_confidence'] = float(range_info['confidence'])
            features['range_age_candles'] = float(range_info['age'])

            # Range position
            range_position = (current_price - range_info['lower']) / range_width if range_width > 0 else 0.5
            features['range_position'] = float(range_position)  # 0=bottom, 1=top

            # Distance from boundaries
            dist_to_upper = abs(current_price - range_info['upper'])
            dist_to_lower = abs(current_price - range_info['lower'])
            features['distance_to_upper_atr'] = float(dist_to_upper / current_atr) if current_atr > 0 else 1.0
            features['distance_to_lower_atr'] = float(dist_to_lower / current_atr) if current_atr > 0 else 1.0

        else:
            # Default values when range not detected
            features.update({
                'range_width_atr': 2.0,
                'range_confidence': 0.3,
                'range_age_candles': 10.0,
                'range_position': 0.5,
                'distance_to_upper_atr': 1.0,
                'distance_to_lower_atr': 1.0
            })

        # Range dynamics
        price_changes = close.pct_change().dropna()
        features['range_volatility'] = float(price_changes.tail(20).std() * 100) if len(price_changes) > 20 else 2.0

        # Volume profile in range
        volume_profile = volume.tail(20)
        features['range_avg_volume'] = float(volume_profile.mean()) if len(volume_profile) > 0 else 1000
        features['volume_concentration'] = float(volume_profile.std() / volume_profile.mean()) if volume_profile.mean() > 0 else 1.0

        # Range breakout history (simplified)
        breakout_attempts = 0
        if len(df) >= 50:
            recent_high = high.tail(50).max()
            recent_low = low.tail(50).min()
            range_height = recent_high - recent_low

            # Count times price approached boundaries
            upper_approaches = ((high > recent_high - range_height * 0.05).tail(20)).sum()
            lower_approaches = ((low < recent_low + range_height * 0.05).tail(20)).sum()
            breakout_attempts = upper_approaches + lower_approaches

        features['breakout_attempts'] = float(breakout_attempts)
        features['range_strength'] = float(1.0 / (1.0 + breakout_attempts * 0.1))  # Fewer attempts = stronger range

        # ========== OSCILLATOR ENSEMBLE (8 features) ==========

        # RSI analysis
        rsi = _rsi(close, 14)
        current_rsi = rsi.iloc[-1] if len(rsi) > 0 else 50.0
        features['rsi_current'] = float(current_rsi)

        # RSI extremes for mean reversion
        if current_rsi <= 30:
            features['rsi_oversold_strength'] = float(30 - current_rsi)  # 0-30
        else:
            features['rsi_oversold_strength'] = 0.0

        if current_rsi >= 70:
            features['rsi_overbought_strength'] = float(current_rsi - 70)  # 0-30
        else:
            features['rsi_overbought_strength'] = 0.0

        # RSI divergence (simplified)
        if len(close) >= 10 and len(rsi) >= 10:
            price_trend = (close.iloc[-1] - close.iloc[-10]) / close.iloc[-10]
            rsi_trend = (rsi.iloc[-1] - rsi.iloc[-10]) / rsi.iloc[-10]
            features['rsi_divergence'] = float(abs(price_trend - rsi_trend))
        else:
            features['rsi_divergence'] = 0.0

        # Stochastic
        stoch_k, stoch_d = _stochastic(df, 14, 3)
        current_stoch = stoch_k.iloc[-1] if len(stoch_k) > 0 else 50.0
        features['stochastic_current'] = float(current_stoch)

        # Williams %R
        williams_r = _williams_r(df, 14)
        current_williams = williams_r.iloc[-1] if len(williams_r) > 0 else -50.0
        features['williams_r'] = float(current_williams)

        # CCI
        cci = _cci(df, 20)
        current_cci = cci.iloc[-1] if len(cci) > 0 else 0.0
        features['cci_current'] = float(current_cci)

        # MFI
        mfi = _mfi(df, 14)
        current_mfi = mfi.iloc[-1] if len(mfi) > 0 else 50.0
        features['mfi_current'] = float(current_mfi)

        # ========== MARKET MICROSTRUCTURE (10 features) ==========

        # Price rejection analysis
        recent_candles = df.tail(5)
        upper_wicks = (recent_candles['high'] - np.maximum(recent_candles['open'], recent_candles['close']))
        lower_wicks = (np.minimum(recent_candles['open'], recent_candles['close']) - recent_candles['low'])
        candle_bodies = abs(recent_candles['close'] - recent_candles['open'])

        features['avg_upper_wick_ratio'] = float((upper_wicks / candle_bodies).mean()) if candle_bodies.mean() > 0 else 0.5
        features['avg_lower_wick_ratio'] = float((lower_wicks / candle_bodies).mean()) if candle_bodies.mean() > 0 else 0.5

        # Volume analysis
        volume_ma_20 = volume.rolling(20).mean()
        current_volume_ratio = volume.iloc[-1] / volume_ma_20.iloc[-1] if volume_ma_20.iloc[-1] > 0 else 1.0
        features['volume_ratio'] = float(current_volume_ratio)

        # Volume trend
        volume_trend = volume.tail(5).pct_change().mean() if len(volume) >= 5 else 0.0
        features['volume_trend'] = float(volume_trend)

        # Price momentum decay (mean reversion signal)
        price_momentum = close.pct_change(5).iloc[-1] if len(close) >= 5 else 0.0
        features['price_momentum_5'] = float(price_momentum)

        # Volatility analysis
        volatility_20 = close.pct_change().rolling(20).std()
        current_volatility = volatility_20.iloc[-1] if len(volatility_20) > 0 else 0.02
        volatility_percentile = (volatility_20 < current_volatility).mean() if len(volatility_20) > 0 else 0.5
        features['volatility_percentile'] = float(volatility_percentile)

        # Order flow approximation (using volume and price)
        if len(df) >= 3:
            recent_3 = df.tail(3)
            buy_volume_approx = recent_3.loc[recent_3['close'] > recent_3['open'], 'volume'].sum()
            sell_volume_approx = recent_3.loc[recent_3['close'] < recent_3['open'], 'volume'].sum()
            total_volume = recent_3['volume'].sum()

            if total_volume > 0:
                features['buy_sell_ratio'] = float(buy_volume_approx / total_volume)
            else:
                features['buy_sell_ratio'] = 0.5
        else:
            features['buy_sell_ratio'] = 0.5

        # Spread approximation (using high-low)
        avg_spread = (high - low).tail(10).mean()
        features['avg_spread_atr'] = float(avg_spread / current_atr) if current_atr > 0 else 1.0

        # Market maker activity approximation (doji candles)
        doji_count = (abs(close - df['open']) < (high - low) * 0.1).tail(10).sum()
        features['doji_frequency'] = float(doji_count / 10)

        # Price clustering (support for mean reversion)
        price_clusters = close.tail(20).round(2).value_counts()
        max_cluster_size = price_clusters.max() if len(price_clusters) > 0 else 1
        features['price_clustering'] = float(max_cluster_size / 20)

        # ========== CONTEXT FEATURES (6 features) ==========

        # Time-based features
        now = datetime.now()
        features['hour_of_day'] = int(now.hour)
        features['day_of_week'] = int(now.weekday())

        # Trading session (encoded as numbers)
        hour = now.hour
        if 0 <= hour < 8:
            session = 0.0  # asian
        elif 8 <= hour < 16:
            session = 1.0  # european
        elif 16 <= hour < 24:
            session = 2.0  # us
        else:
            session = 3.0  # off_hours
        features['session'] = session

        # Market cap tier (simplified based on symbol)
        # Use proper hardcoded clustering instead of primitive name length logic
        try:
            from hardcoded_clusters import get_symbol_cluster
            cluster_id = get_symbol_cluster(symbol)
            features['symbol_cluster'] = cluster_id
            features['market_cap_tier'] = cluster_id  # Keep compatibility with existing features
            
            # Add cluster-specific volatility norms for MR strategy
            cluster_volatility_norms = {
                1: 2.5,  # Blue chip - lower volatility expectation
                2: 0.1,  # Stable - very low volatility
                3: 8.5,  # Meme/volatile - high volatility expectation  
                4: 4.2,  # Mid-cap alts - moderate volatility
                5: 6.5   # Small cap - higher volatility
            }
            features['cluster_volatility_norm'] = cluster_volatility_norms.get(cluster_id, 5.0)
            
        except ImportError:
            # Fallback if clustering not available
            features['symbol_cluster'] = 3
            features['market_cap_tier'] = 2
            features['cluster_volatility_norm'] = 5.0

        # Volatility regime classification (encoded as numbers)
        vol_low_threshold = volatility_20.quantile(0.33) if len(volatility_20) > 0 else 0.01
        vol_high_threshold = volatility_20.quantile(0.67) if len(volatility_20) > 0 else 0.03
        if current_volatility < vol_low_threshold:
            features['volatility_regime'] = 0.0  # low
        elif current_volatility > vol_high_threshold:
            features['volatility_regime'] = 2.0  # high
        else:
            features['volatility_regime'] = 1.0  # normal

        # Risk-reward context from signal
        if signal_data and 'entry' in signal_data and 'sl' in signal_data and 'tp' in signal_data:
            entry = signal_data['entry']
            sl = signal_data['sl']
            tp = signal_data['tp']

            risk = abs(entry - sl)
            reward = abs(tp - entry)
            features['signal_risk_reward'] = float(reward / risk) if risk > 0 else 2.0
        else:
            features['signal_risk_reward'] = 2.0

        # Add cluster enhancement features for improved MR strategy performance
        try:
            from cluster_feature_enhancer import enhance_ml_features
            features = enhance_ml_features(features, symbol)
            logger.debug(f"[{symbol}] Added cluster enhancement features")
        except ImportError:
            logger.debug(f"[{symbol}] Cluster feature enhancer not available")

        # Convert numpy types to Python native types for JSON serialization
        for key, value in features.items():
            if isinstance(value, np.floating):
                features[key] = float(value)
            elif isinstance(value, np.integer):
                features[key] = int(value)
            elif pd.isna(value):
                features[key] = 0.0

        logger.debug(f"[{symbol}] Calculated {len(features)} enhanced MR features")
        return features

    except Exception as e:
        logger.error(f"[{symbol}] Error calculating enhanced MR features: {e}")
        return _get_default_enhanced_features()

def _get_default_enhanced_features() -> Dict:
    """Return default feature values when calculation fails"""
    return {
        # Range characteristics
        'range_width_atr': 2.0,
        'range_confidence': 0.3,
        'range_age_candles': 20.0,
        'range_position': 0.5,
        'distance_to_upper_atr': 1.0,
        'distance_to_lower_atr': 1.0,
        'range_volatility': 2.0,
        'range_avg_volume': 1000.0,
        'volume_concentration': 1.0,
        'breakout_attempts': 2.0,
        'range_strength': 0.8,

        # Oscillator ensemble
        'rsi_current': 50.0,
        'rsi_oversold_strength': 0.0,
        'rsi_overbought_strength': 0.0,
        'rsi_divergence': 0.0,
        'stochastic_current': 50.0,
        'williams_r': -50.0,
        'cci_current': 0.0,
        'mfi_current': 50.0,

        # Market microstructure
        'avg_upper_wick_ratio': 0.3,
        'avg_lower_wick_ratio': 0.3,
        'volume_ratio': 1.0,
        'volume_trend': 0.0,
        'price_momentum_5': 0.0,
        'volatility_percentile': 0.5,
        'buy_sell_ratio': 0.5,
        'avg_spread_atr': 1.0,
        'doji_frequency': 0.2,
        'price_clustering': 0.1,

        # Context features
        'hour_of_day': 12,
        'day_of_week': 2,
        'session': "us",
        'market_cap_tier': 2,
        'volatility_regime': "normal",
        'signal_risk_reward': 2.0,
        
        # Cluster features (defaults)
        'symbol_cluster': 3,
        'cluster_volatility_norm': 5.0,
        'cluster_confidence': 1.0
    }

def get_feature_count() -> int:
    """Return the number of features for model sizing"""
    return len(_get_default_enhanced_features())

def get_feature_names() -> list:
    """Return ordered list of feature names for model training"""
    return list(_get_default_enhanced_features().keys())