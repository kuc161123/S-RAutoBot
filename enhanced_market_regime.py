"""
Enhanced Market Regime Detection
Sophisticated classification for parallel strategy routing
Determines: HighQualityRanging, LowQualityRanging, Trending, Volatile
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from scipy.signal import find_peaks
import ta

logger = logging.getLogger(__name__)

@dataclass
class RegimeAnalysis:
    """Complete regime analysis result"""
    primary_regime: str  # "ranging", "trending", "volatile"
    regime_confidence: float  # 0-1 confidence in classification
    range_quality: str  # "high", "medium", "low" (if ranging)
    trend_strength: float  # 0-100 trend strength (if trending)
    volatility_level: str  # "low", "normal", "high"
    regime_persistence: float  # How long this regime has been active
    recommended_strategy: str  # "enhanced_mr", "pullback", "none"

def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Calculate Average True Range"""
    high = df['high']
    low = df['low']
    close = df['close']
    prev_close = close.shift()

    tr = np.maximum(high - low,
         np.maximum(abs(high - prev_close), abs(low - prev_close)))
    return tr.rolling(n).mean()

def _adx(df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate ADX, +DI, -DI"""
    high = df['high']
    low = df['low']
    close = df['close']

    # Calculate True Range and Directional Movement
    plus_dm = high.diff()
    minus_dm = low.diff().mul(-1)

    # Only keep positive movements
    plus_dm[plus_dm < 0] = 0
    plus_dm[plus_dm < minus_dm] = 0
    minus_dm[minus_dm < 0] = 0
    minus_dm[minus_dm < plus_dm] = 0

    # True Range calculation
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Smooth the values
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)

    # Calculate DX and ADX
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.ewm(alpha=1/period, adjust=False).mean()

    return adx, plus_di, minus_di

def _bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands"""
    close = df['close']
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()

    upper_bb = sma + (std * std_dev)
    lower_bb = sma - (std * std_dev)

    return upper_bb, sma, lower_bb

def _detect_range_quality(df: pd.DataFrame, lookback: int = 100) -> Dict:
    """Advanced range quality detection"""
    if len(df) < lookback:
        return {'quality': 'low', 'confidence': 0.2, 'upper': None, 'lower': None, 'age': 0}

    recent_df = df.tail(lookback)
    high_prices = recent_df['high']
    low_prices = recent_df['low']
    close_prices = recent_df['close']
    volume_prices = recent_df['volume']

    # Find significant peaks and troughs with more relaxed criteria
    # Use adaptive prominence based on price level (percentage-based)
    avg_price = close_prices.mean()
    prominence_threshold = avg_price * 0.001  # 0.1% of average price (was 0.5%)
    min_prominence = avg_price * 0.0005  # 0.05% minimum (was 0.2%)

    # Use the higher of the two thresholds
    prominence = max(prominence_threshold, min_prominence)

    high_peaks, high_properties = find_peaks(high_prices,
                                           distance=5,
                                           prominence=prominence)
    low_peaks, low_properties = find_peaks(-low_prices,
                                         distance=5,
                                         prominence=prominence)

    if len(high_peaks) < 2 or len(low_peaks) < 2:
        # Fallback: use simple max/min approach
        upper_level = high_prices.max()
        lower_level = low_prices.min()
        
        # Basic range validation
        if upper_level <= lower_level:
            return {'quality': 'low', 'confidence': 0.2, 'upper': None, 'lower': None, 'age': 0}
        
        range_height = upper_level - lower_level
        range_pct = range_height / lower_level
        
        # Simple quality assessment for fallback - add missing fields
        tolerance = range_height * 0.02
        upper_touches = ((high_prices >= (upper_level - tolerance)) &
                        (high_prices <= (upper_level + tolerance))).sum()
        lower_touches = ((low_prices >= (lower_level - tolerance)) &
                        (low_prices <= (lower_level + tolerance))).sum()
        total_touches = upper_touches + lower_touches
        in_range_count = ((close_prices >= lower_level) & (close_prices <= upper_level)).sum()
        range_respect = in_range_count / len(close_prices)
        
        if range_pct < 0.015:  # Less than 1.5%
            return {'quality': 'low', 'confidence': 0.3, 'upper': float(upper_level), 'lower': float(lower_level), 'age': 0,
                   'touches': int(total_touches), 'respect_rate': float(range_respect), 'level_consistency': 0.3}
        elif range_pct > 0.12:  # More than 12%
            return {'quality': 'low', 'confidence': 0.3, 'upper': float(upper_level), 'lower': float(lower_level), 'age': 0,
                   'touches': int(total_touches), 'respect_rate': float(range_respect), 'level_consistency': 0.3}
        else:
            return {'quality': 'medium', 'confidence': 0.5, 'upper': float(upper_level), 'lower': float(lower_level), 'age': lookback//2,
                   'touches': int(total_touches), 'respect_rate': float(range_respect), 'level_consistency': 0.6}

    # Get recent significant levels - prefer more recent peaks
    # Take the highest highs and lowest lows from recent peaks
    recent_high_count = min(6, len(high_peaks))
    recent_low_count = min(6, len(low_peaks))
    
    recent_highs = high_prices.iloc[high_peaks[-recent_high_count:]]
    recent_lows = low_prices.iloc[low_peaks[-recent_low_count:]]

    # Calculate potential range boundaries - use percentile approach for robustness
    upper_level = recent_highs.quantile(0.8)  # 80th percentile of recent highs
    lower_level = recent_lows.quantile(0.2)   # 20th percentile of recent lows

    if upper_level <= lower_level:
        return {'quality': 'low', 'confidence': 0.2, 'upper': None, 'lower': None, 'age': 0}

    range_height = upper_level - lower_level

    # Quality factors
    quality_score = 0.0

    # 1. Level consistency (how tight are the peaks/troughs around the levels)
    upper_consistency = 1.0 - (recent_highs.std() / range_height) if range_height > 0 else 0
    lower_consistency = 1.0 - (recent_lows.std() / range_height) if range_height > 0 else 0
    level_consistency = (upper_consistency + lower_consistency) / 2
    quality_score += level_consistency * 0.30

    # 2. Touch frequency (how often price visits the levels)
    tolerance = range_height * 0.05  # 5% tolerance (increased from 2% for crypto volatility)
    upper_touches = ((high_prices >= (upper_level - tolerance)) &
                    (high_prices <= (upper_level + tolerance))).sum()
    lower_touches = ((low_prices >= (lower_level - tolerance)) &
                    (low_prices <= (lower_level + tolerance))).sum()

    total_touches = upper_touches + lower_touches
    touch_score = min(1.0, total_touches / 10)  # Normalize to max 10 touches
    quality_score += touch_score * 0.25

    # 3. Range respect (price stays within range most of the time)
    in_range_count = ((close_prices >= lower_level) & (close_prices <= upper_level)).sum()
    range_respect = in_range_count / len(close_prices)
    quality_score += range_respect * 0.20

    # 4. Volume confirmation at levels
    volume_mean = volume_prices.mean()
    upper_volume_avg = volume_prices[(high_prices >= (upper_level - tolerance)) &
                                   (high_prices <= (upper_level + tolerance))].mean()
    lower_volume_avg = volume_prices[(low_prices >= (lower_level - tolerance)) &
                                   (low_prices <= (lower_level + tolerance))].mean()

    volume_confirmation = 0
    if not pd.isna(upper_volume_avg) and upper_volume_avg > volume_mean:
        volume_confirmation += 0.5
    if not pd.isna(lower_volume_avg) and lower_volume_avg > volume_mean:
        volume_confirmation += 0.5

    quality_score += volume_confirmation * 0.15

    # 5. Range age/persistence
    # Estimate how long the range has been active
    range_age_estimate = min(len(high_peaks), len(low_peaks)) * 5  # Rough estimate
    age_score = min(1.0, range_age_estimate / 50)  # Normalize to 50 candles
    quality_score += age_score * 0.10

    # Determine quality level
    if quality_score >= 0.75:
        quality = 'high'
    elif quality_score >= 0.50:
        quality = 'medium'
    else:
        quality = 'low'

    return {
        'quality': quality,
        'confidence': float(quality_score),
        'upper': float(upper_level),
        'lower': float(lower_level),
        'age': int(range_age_estimate),
        'touches': int(total_touches),
        'respect_rate': float(range_respect),
        'level_consistency': float(level_consistency)
    }

def get_enhanced_market_regime(df: pd.DataFrame, symbol: str = "UNKNOWN") -> RegimeAnalysis:
    """
    Enhanced market regime detection for parallel strategy routing

    Returns RegimeAnalysis with recommended strategy
    """
    if len(df) < 50:
        return RegimeAnalysis(
            primary_regime="volatile",
            regime_confidence=0.3,
            range_quality="low",
            trend_strength=50.0,
            volatility_level="normal",
            regime_persistence=0.0,
            recommended_strategy="none"
        )

    try:
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']

        # ===== TREND ANALYSIS =====
        adx_series, plus_di, minus_di = _adx(df, 14)
        current_adx = adx_series.iloc[-1] if len(adx_series) > 0 else 25

        # Enhanced trend strength calculation
        # Consider multiple timeframe EMAs
        ema_short = close.ewm(span=8).mean()
        ema_medium = close.ewm(span=21).mean()
        ema_long = close.ewm(span=50).mean()

        # EMA alignment score
        if ema_short.iloc[-1] > ema_medium.iloc[-1] > ema_long.iloc[-1]:
            ema_alignment = 100  # Strong uptrend
        elif ema_short.iloc[-1] < ema_medium.iloc[-1] < ema_long.iloc[-1]:
            ema_alignment = 100  # Strong downtrend
        else:
            # Calculate partial alignment
            short_medium = abs(ema_short.iloc[-1] - ema_medium.iloc[-1]) / ema_medium.iloc[-1]
            medium_long = abs(ema_medium.iloc[-1] - ema_long.iloc[-1]) / ema_long.iloc[-1]
            ema_alignment = max(0, 100 - (short_medium + medium_long) * 5000)

        # Combined trend strength
        trend_strength = (current_adx * 0.6 + ema_alignment * 0.4)

        # ===== VOLATILITY ANALYSIS =====
        atr_series = _atr(df, 14)
        current_atr = atr_series.iloc[-1] if len(atr_series) > 0 else close.iloc[-1] * 0.02

        # Bollinger Band analysis
        bb_upper, bb_middle, bb_lower = _bollinger_bands(df, 20, 2)
        bb_width = (bb_upper - bb_lower) / bb_middle
        current_bb_width = bb_width.iloc[-1] if len(bb_width) > 0 else 0.04

        # Volatility percentile
        atr_percentile = (atr_series < current_atr).sum() / len(atr_series) if len(atr_series) > 20 else 0.5
        bb_width_percentile = (bb_width < current_bb_width).sum() / len(bb_width) if len(bb_width) > 20 else 0.5

        volatility_score = (atr_percentile + bb_width_percentile) / 2

        if volatility_score < 0.33:
            volatility_level = "low"
        elif volatility_score > 0.67:
            volatility_level = "high"
        else:
            volatility_level = "normal"

        # ===== RANGE ANALYSIS =====
        range_analysis = _detect_range_quality(df, lookback=80)
        range_quality = range_analysis['quality']
        range_confidence = range_analysis['confidence']

        # ===== REGIME CLASSIFICATION =====

        # Regime scores
        trend_score = 0
        range_score = 0
        volatile_score = 0

        # Trend scoring
        if trend_strength >= 30:  # Strong trend
            trend_score += trend_strength / 100 * 0.4
            if volatility_level == "normal":
                trend_score += 0.2  # Bonus for normal volatility in trends
            if range_confidence < 0.5:  # Low range confidence supports trending
                trend_score += 0.2

        # Range scoring
        range_score = range_confidence * 0.6
        if trend_strength < 25:  # Low trend strength supports ranging
            range_score += 0.2
        if volatility_level == "low":  # Low volatility supports ranging
            range_score += 0.15

        # Volatile scoring (high volatility, low trend, low range quality)
        if volatility_level == "high":
            volatile_score += 0.4
        if trend_strength < 20 and range_confidence < 0.4:
            volatile_score += 0.3
        bb_threshold = bb_width.quantile(0.8) if len(bb_width) > 20 else 0.06
        if current_bb_width > bb_threshold:
            volatile_score += 0.2

        # Determine primary regime
        regime_scores = {
            'trending': trend_score,
            'ranging': range_score,
            'volatile': volatile_score
        }

        primary_regime = max(regime_scores, key=regime_scores.get)
        regime_confidence = regime_scores[primary_regime]

        # ===== STRATEGY RECOMMENDATION =====

        if primary_regime == "ranging":
            # Apply range width filter: Only recommend enhanced MR for optimal range sizes
            if range_analysis['upper'] and range_analysis['lower']:
                range_width = (range_analysis['upper'] - range_analysis['lower']) / range_analysis['lower']
                min_range_width = 0.006  # 0.6% minimum (was 0.8%)
                max_range_width = 0.08   # 8.0% maximum (was 6.0%)

                if min_range_width <= range_width <= max_range_width:
                    if range_quality in ["high", "medium"] and regime_confidence >= 0.35:  # Lowered from 0.4
                        recommended_strategy = "enhanced_mr"
                    elif range_quality == "medium" and regime_confidence >= 0.25:  # Lowered from 0.3
                        recommended_strategy = "enhanced_mr"  # Still try MR but with caution
                    elif range_quality == "low" and regime_confidence >= 0.35 and range_width >= 0.02:  # Lowered from 0.025
                        recommended_strategy = "enhanced_mr"  # Give low quality ranges a chance if wide enough
                    else:
                        recommended_strategy = "none"  # Range too poor quality
                else:
                    recommended_strategy = "none"  # Range width outside optimal bounds
                    logger.debug(f"[{symbol}] Range width {range_width:.1%} outside optimal bounds ({min_range_width:.1%}-{max_range_width:.1%})")
            else:
                recommended_strategy = "none"  # No valid range detected

        elif primary_regime == "trending":
            if trend_strength >= 25 and volatility_level != "high":
                recommended_strategy = "pullback"
            else:
                recommended_strategy = "none"  # Trend too weak or too volatile

        else:  # volatile
            recommended_strategy = "none"  # Skip volatile markets

        # ===== REGIME PERSISTENCE =====
        # Estimate how long current regime has been active
        persistence_lookback = min(30, len(df) // 2)
        if persistence_lookback > 5:
            recent_regimes = []
            for i in range(persistence_lookback):
                subset = df.iloc[-(i+20):-(i) if i > 0 else len(df)]
                if len(subset) >= 20:
                    mini_analysis = _detect_range_quality(subset, 20)
                    mini_adx = current_adx  # Simplified

                    if mini_analysis['confidence'] >= 0.5:
                        recent_regimes.append('ranging')
                    elif mini_adx >= 25:
                        recent_regimes.append('trending')
                    else:
                        recent_regimes.append('volatile')

            # Calculate persistence
            if recent_regimes:
                current_regime_count = sum(1 for r in recent_regimes if r == primary_regime)
                regime_persistence = current_regime_count / len(recent_regimes)
            else:
                regime_persistence = 0.5
        else:
            regime_persistence = 0.5

        logger.debug(f"[{symbol}] Regime: {primary_regime} ({regime_confidence:.2f}), "
                    f"Strategy: {recommended_strategy}, Range: {range_quality}, "
                    f"Trend: {trend_strength:.1f}, Vol: {volatility_level}")

        return RegimeAnalysis(
            primary_regime=primary_regime,
            regime_confidence=float(regime_confidence),
            range_quality=range_quality,
            trend_strength=float(trend_strength),
            volatility_level=volatility_level,
            regime_persistence=float(regime_persistence),
            recommended_strategy=recommended_strategy
        )

    except Exception as e:
        logger.error(f"[{symbol}] Error in enhanced regime detection: {e}")
        # Return safe default
        return RegimeAnalysis(
            primary_regime="volatile",
            regime_confidence=0.3,
            range_quality="low",
            trend_strength=50.0,
            volatility_level="normal",
            regime_persistence=0.5,
            recommended_strategy="none"
        )

def get_regime_summary(df: pd.DataFrame, symbol: str = "UNKNOWN") -> str:
    """Get a human-readable regime summary"""
    analysis = get_enhanced_market_regime(df, symbol)

    confidence_desc = "High" if analysis.regime_confidence >= 0.7 else "Medium" if analysis.regime_confidence >= 0.5 else "Low"

    summary = f"{analysis.primary_regime.title()} ({confidence_desc} confidence)"

    if analysis.primary_regime == "ranging":
        summary += f", {analysis.range_quality} quality range"
    elif analysis.primary_regime == "trending":
        summary += f", {analysis.trend_strength:.0f}% trend strength"

    summary += f", {analysis.volatility_level} volatility"
    summary += f" → {analysis.recommended_strategy.replace('_', ' ').title()}"

    return summary

# Backward compatibility with existing code
def get_market_regime(df: pd.DataFrame, adx_period: int = 14, bb_period: int = 20) -> str:
    """
    Backward compatible function that returns simple regime classification
    Maps enhanced regime to original format
    """
    analysis = get_enhanced_market_regime(df)

    # Map enhanced classification to original format
    if analysis.primary_regime == "ranging":
        return "Ranging"
    elif analysis.primary_regime == "trending":
        return "Trending"
    else:
        return "Volatile"