"""
Enhanced Market Regime Detection
Sophisticated classification for parallel strategy routing
Determines: HighQualityRanging, LowQualityRanging, Trending, Volatile
"""
import logging
import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
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
    trend_probability: float = 0.0
    range_probability: float = 0.0
    volatile_probability: float = 0.0
    feature_snapshot: Dict[str, float] = field(default_factory=dict)


def _normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    adjusted = {label: max(0.0, float(value)) for label, value in scores.items()}
    total = sum(adjusted.values())
    if total <= 0:
        uniform = 1.0 / max(len(scores), 1)
        return {label: uniform for label in scores}
    return {label: value / total for label, value in adjusted.items()}


def _safe_resample(df: pd.DataFrame, rule: str, min_length: int = 5) -> Optional[pd.DataFrame]:
    if len(df) < min_length or not isinstance(df.index, pd.DatetimeIndex):
        return None
    if df.index.tz is None:
        df = df.tz_localize("UTC")
    try:
        resampled = df.resample(rule).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()
        if len(resampled) >= min_length:
            return resampled
    except Exception:
        pass
    return None


def _linear_regression_stats(series: pd.Series, window: int) -> Tuple[float, float]:
    if len(series) < window or window <= 1:
        return 0.0, 0.0
    window_series = series.tail(window).values
    if np.allclose(window_series, window_series[0]):
        return 0.0, 0.0
    x = np.arange(window, dtype=float)
    slope, intercept = np.polyfit(x, window_series, 1)
    fitted = slope * x + intercept
    resid = window_series - fitted
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((window_series - window_series.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(slope), float(max(min(r2, 1.0), 0.0))


def _hurst_exponent(series: pd.Series) -> float:
    if len(series) < 64:
        return 0.5
    lags = range(2, min(100, len(series) // 2))
    tau = [np.sqrt(((series[lag:] - series[:-lag]) ** 2).mean()) for lag in lags]
    with np.errstate(divide="ignore"):
        log_lags = np.log(lags)
        log_tau = np.log(np.maximum(tau, 1e-9))
    slope, _ = np.polyfit(log_lags, log_tau, 1)
    return float(max(0.0, min(1.0, slope * 2)))


def _auto_corr(series: pd.Series, lag: int) -> float:
    if len(series) <= lag or lag <= 0:
        return 0.0
    return float(series.autocorr(lag=lag))


def _compute_regime_features(df: pd.DataFrame) -> Dict[str, float]:
    features: Dict[str, float] = {}
    close = df["close"]
    volume = df["volume"]

    slope_15, r2_15 = _linear_regression_stats(close, min(len(close), 30))
    slope_60, r2_60 = _linear_regression_stats(close, min(len(close), 60))
    slope_240, r2_240 = _linear_regression_stats(close, min(len(close), 120))

    features["trend_slope_15"] = slope_15
    features["trend_r2_15"] = r2_15
    features["trend_slope_60"] = slope_60
    features["trend_r2_60"] = r2_60
    features["trend_slope_240"] = slope_240
    features["trend_r2_240"] = r2_240

    ema_short = close.ewm(span=8).mean().iloc[-1]
    ema_long = close.ewm(span=55).mean().iloc[-1]
    features["ema_alignment"] = float((ema_short - ema_long) / ema_long) if ema_long else 0.0

    atr_series = _atr(df, 14)
    current_atr = atr_series.iloc[-1] if len(atr_series) else close.iloc[-1] * 0.02
    features["atr_to_price"] = float(current_atr / close.iloc[-1]) if close.iloc[-1] else 0.0

    rolling_std = close.pct_change().rolling(20).std().iloc[-1]
    if rolling_std and not math.isnan(rolling_std):
        features["atr_volatility_ratio"] = float(current_atr / (rolling_std * close.iloc[-1]))
    else:
        features["atr_volatility_ratio"] = 0.0

    bb_upper, bb_middle, bb_lower = _bollinger_bands(df, 20, 2)
    bb_width = ((bb_upper - bb_lower) / bb_middle).fillna(0)
    current_bb_width = bb_width.iloc[-1] if len(bb_width) else 0.0
    if len(bb_width) > 20:
        features["bb_width_percentile"] = float((bb_width < current_bb_width).sum() / len(bb_width))
    else:
        features["bb_width_percentile"] = 0.5

    returns = close.pct_change().dropna()
    features["return_volatility"] = float(returns.std()) if len(returns) else 0.0
    features["hurst_exponent"] = _hurst_exponent(close)
    features["auto_corr_1"] = _auto_corr(returns, 1)
    features["auto_corr_2"] = _auto_corr(returns, 2)

    vol_window = volume.rolling(30)
    vol_mean = vol_window.mean().iloc[-1]
    vol_std = vol_window.std().iloc[-1]
    features["volume_zscore"] = float((volume.iloc[-1] - vol_mean) / vol_std) if vol_std else 0.0

    price_range = close.rolling(20).apply(lambda x: x.max() - x.min()).iloc[-1]
    features["price_chop_ratio"] = float(price_range / close.iloc[-1]) if close.iloc[-1] else 0.0

    for rule, prefix in (("1H", "60"), ("4H", "240")):
        resampled = _safe_resample(df, rule)
        if resampled is not None:
            slope, r2 = _linear_regression_stats(resampled["close"], min(len(resampled), 30))
            features[f"trend_slope_{prefix}"] = slope
            features[f"trend_r2_{prefix}"] = r2

    return features


_CLASSIFIER = None


def _get_classifier():
    global _CLASSIFIER
    if _CLASSIFIER is None:
        try:
            from regime_classifier import get_regime_classifier

            _CLASSIFIER = get_regime_classifier()
        except Exception as exc:
            logger.debug(f"Regime classifier unavailable: {exc}")
            _CLASSIFIER = False
    return _CLASSIFIER if _CLASSIFIER not in (None, False) else None

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
    touch_score = min(1.0, total_touches / 6)  # More forgiving (was /10)
    quality_score += touch_score * 0.25

    # 3. Range respect (price stays within range most of the time) - more lenient
    in_range_count = ((close_prices >= lower_level) & (close_prices <= upper_level)).sum()
    range_respect = in_range_count / len(close_prices)
    if range_respect >= 0.4:  # More lenient threshold
        quality_score += range_respect * 0.20
    else:
        quality_score += range_respect * 0.10  # Partial credit

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

    quality_score += volume_confirmation * 0.05  # Make volume less critical (was 0.15)

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

        feature_snapshot = _compute_regime_features(df)
        classifier = _get_classifier()
        ml_probabilities = None
        if classifier:
            ml_probabilities = classifier.predict_probabilities(feature_snapshot)

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

        # More aggressive volatility thresholds to filter chaotic markets
        if volatility_score < 0.25:
            volatility_level = "low"
        elif volatility_score > 0.85:
            volatility_level = "extreme"  # New extreme level for very volatile markets
        elif volatility_score > 0.65:
            volatility_level = "high"
        else:
            volatility_level = "normal"

        # ===== RANGE ANALYSIS =====
        range_analysis = _detect_range_quality(df, lookback=80)
        range_quality = range_analysis['quality']
        range_confidence = range_analysis['confidence']
        try:
            # Expose range_confidence via feature_snapshot for downstream gates
            if isinstance(feature_snapshot, dict):
                feature_snapshot['range_confidence'] = float(range_confidence)
        except Exception:
            pass

        # ===== REGIME CLASSIFICATION =====

        # Regime scores
        trend_score = 0
        range_score = 0
        volatile_score = 0

        # Trend scoring - MUCH MORE STRICT for truly trending markets
        if trend_strength >= 40:  # Strong trend threshold raised significantly
            trend_score += trend_strength / 100 * 0.5  # Higher weight for strong trends
            if volatility_level in ["low", "normal"]:  # Require low/normal volatility
                trend_score += 0.25  # Bonus for stable volatility in trends
            if range_confidence < 0.4:  # Range confidence must be low for trending
                trend_score += 0.2
            
            # Additional trend quality checks
            # EMA alignment bonus - require proper trend structure
            if len(close) >= 50:
                ema_8 = close.ewm(span=8).mean().iloc[-1]
                ema_21 = close.ewm(span=21).mean().iloc[-1] 
                ema_50 = close.ewm(span=50).mean().iloc[-1]
                current_price = close.iloc[-1]
                
                # Strong uptrend alignment
                if current_price > ema_8 > ema_21 > ema_50:
                    trend_score += 0.15
                # Strong downtrend alignment  
                elif current_price < ema_8 < ema_21 < ema_50:
                    trend_score += 0.15

        # Range scoring - MUCH MORE STRICT for truly ranging markets
        # Only consider ranges with high confidence (≥50%)
        if range_confidence >= 0.5:
            range_score = range_confidence * 0.7  # Higher weight for confident ranges
            
            if trend_strength < 20:  # Very low trend strength required for ranging
                range_score += 0.2
            if volatility_level == "low":  # Low volatility strongly supports ranging
                range_score += 0.2
                
            # Additional range quality requirements
            if range_analysis.get('touches', 0) >= 4:  # Minimum 4 touches for quality
                range_score += 0.1
            if range_analysis.get('respect_rate', 0) >= 0.6:  # 60% respect rate required
                range_score += 0.1
        else:
            # Low confidence ranges get minimal score
            range_score = range_confidence * 0.3

        # Volatile scoring (high volatility, low trend, low range quality)
        if volatility_level == "extreme":
            volatile_score += 0.7  # Very high score for extreme volatility
        elif volatility_level == "high":
            volatile_score += 0.4
        if trend_strength < 20 and range_confidence < 0.4:
            volatile_score += 0.3
        bb_threshold = bb_width.quantile(0.8) if len(bb_width) > 20 else 0.06
        if current_bb_width > bb_threshold:
            volatile_score += 0.2
        
        # Additional volatility penalty - check for extreme ATR spikes
        if len(atr_series) > 10:
            recent_atr_mean = atr_series.tail(10).mean()
            atr_spike_ratio = current_atr / recent_atr_mean if recent_atr_mean > 0 else 1
            if atr_spike_ratio > 2.0:  # ATR is 2x higher than recent average
                volatile_score += 0.3
                logger.info(f"[{symbol}] 🚨 ATR SPIKE: {atr_spike_ratio:.1f}x recent average - adding volatility penalty")

        # Determine primary regime
        regime_scores = {
            'trending': trend_score,
            'ranging': range_score,
            'volatile': volatile_score
        }

        probability_map = _normalize_scores(regime_scores)
        primary_regime = max(probability_map, key=probability_map.get)
        regime_confidence = probability_map[primary_regime]

        if ml_probabilities:
            # Allow weight override via environment variable REGIME_BLEND_WEIGHT (default 0.4)
            import os
            try:
                blend_weight = float(os.getenv('REGIME_BLEND_WEIGHT', '0.4'))
            except Exception:
                blend_weight = 0.4
            blended = {}
            for label in regime_scores:
                blended[label] = (
                    probability_map.get(label, 0.0) * (1 - blend_weight)
                    + ml_probabilities.get(label, 0.0) * blend_weight
                )
            probability_map = _normalize_scores(blended)
            ml_primary = max(probability_map, key=probability_map.get)
            if ml_primary != primary_regime:
                logger.info(
                    f"[{symbol}] 🤖 ML override: {primary_regime} → {ml_primary} "
                    f"(heur={regime_confidence:.1%}, ml={probability_map[ml_primary]:.1%})"
                )
            primary_regime = ml_primary
            regime_confidence = probability_map[primary_regime]

        # ===== STRATEGY RECOMMENDATION =====

        if primary_regime == "ranging":
            # STRICT range requirements for truly ranging markets
            if range_analysis['upper'] and range_analysis['lower']:
                range_width = (range_analysis['upper'] - range_analysis['lower']) / range_analysis['lower']
                min_range_width = 0.015  # 1.5% minimum - tighter than before
                max_range_width = 0.08   # 8.0% maximum - tighter than before
                
                touches = range_analysis.get('touches', 0)
                respect_rate = range_analysis.get('respect_rate', 0)

                logger.info(f"[{symbol}] 🔍 RANGE DEBUG: width={range_width:.1%}, quality={range_quality}, confidence={regime_confidence:.1%}")
                logger.info(f"[{symbol}] 📊 Range details: touches={touches}, respect={respect_rate:.1%}, bounds={min_range_width:.1%}-{max_range_width:.1%}")

                if min_range_width <= range_width <= max_range_width:
                    # STRICT quality requirements - only high quality ranges
                    if (range_quality == "high" and 
                        regime_confidence >= 0.6 and  # 60% minimum confidence
                        touches >= 4 and              # Minimum 4 touches
                        respect_rate >= 0.6):         # 60% respect rate
                        recommended_strategy = "enhanced_mr"
                        logger.info(f"[{symbol}] ✅ MR: HIGH QUALITY range (conf={regime_confidence:.1%}, touches={touches}, respect={respect_rate:.1%})")
                    
                    elif (range_quality == "medium" and 
                          regime_confidence >= 0.7 and  # Higher confidence for medium quality
                          touches >= 5 and              # More touches required  
                          respect_rate >= 0.7):         # Higher respect rate
                        recommended_strategy = "enhanced_mr"
                        logger.info(f"[{symbol}] ✅ MR: MEDIUM quality range with strong metrics (conf={regime_confidence:.1%}, touches={touches})")
                    
                    else:
                        recommended_strategy = "none"
                        logger.info(f"[{symbol}] ❌ MR: Range quality insufficient (quality={range_quality}, conf={regime_confidence:.1%}, touches={touches}, respect={respect_rate:.1%})")
                else:
                    recommended_strategy = "none"
                    logger.info(f"[{symbol}] ❌ MR: Range width {range_width:.1%} outside strict bounds ({min_range_width:.1%}-{max_range_width:.1%})")
            else:
                recommended_strategy = "none"
                logger.info(f"[{symbol}] ❌ MR: No valid range detected")

        elif primary_regime == "trending":
            logger.info(f"[{symbol}] 🔍 TREND DEBUG: strength={trend_strength:.1f}, volatility={volatility_level}")
            
            # STRICT trend requirements for truly trending markets
            if (trend_strength >= 35 and                    # Much higher threshold
                regime_confidence >= 0.6 and               # High confidence required
                volatility_level in ["low", "normal"] and  # Stable volatility only
                range_confidence < 0.4):                   # Low range confidence
                
                # Additional EMA alignment check for trend quality
                trend_aligned = False
                if len(close) >= 50:
                    ema_8 = close.ewm(span=8).mean().iloc[-1]
                    ema_21 = close.ewm(span=21).mean().iloc[-1] 
                    ema_50 = close.ewm(span=50).mean().iloc[-1]
                    current_price = close.iloc[-1]
                    
                    # Check for proper trend alignment
                    if (current_price > ema_8 > ema_21 > ema_50 or  # Strong uptrend
                        current_price < ema_8 < ema_21 < ema_50):   # Strong downtrend
                        trend_aligned = True
                
                if trend_aligned:
                    recommended_strategy = "trend"
                    logger.info(f"[{symbol}] ✅ TREND: Strong trending market (strength={trend_strength:.1f}, conf={regime_confidence:.1%}, EMA aligned)")
                else:
                    recommended_strategy = "none"
                    logger.info(f"[{symbol}] ❌ TREND: Lacks EMA alignment (strength={trend_strength:.1f})")
            else:
                recommended_strategy = "none"
                logger.info(f"[{symbol}] ❌ TREND: Requirements not met (strength={trend_strength:.1f}, conf={regime_confidence:.1%}, vol={volatility_level})")

        elif primary_regime == "volatile":
            logger.info(f"[{symbol}] 🔍 VOLATILE DEBUG: strength={trend_strength:.1f}, volatility={volatility_level}")
            # STRICT: No trading in volatile markets regardless of trend strength
            recommended_strategy = "none"
            logger.info(f"[{symbol}] ❌ VOLATILE: Market too chaotic for any strategy (volatility={volatility_level})")

        else:
            recommended_strategy = "none"
            logger.info(f"[{symbol}] ❌ Unknown regime: {primary_regime}")

        # ===== VOLATILITY SAFETY CHECK =====
        # CRITICAL: No trading in extreme volatility regardless of other factors
        if volatility_level == "extreme":
            recommended_strategy = "none"
            logger.info(f"[{symbol}] 🚫 VOLATILITY OVERRIDE: Extreme volatility detected - no trading allowed")
        
        # ===== COMPREHENSIVE FALLBACK SYSTEM =====
        elif recommended_strategy == "none":
            logger.info(f"[{symbol}] 🛡️ FALLBACK ANALYSIS:")
            
            # STRICT fallback - no permissive defaults
            if volatility_level in ["low", "normal"]:
                # Only very high confidence signals allowed in fallback
                if (range_analysis.get('confidence', 0) >= 0.7 and  # 70% confidence minimum
                    range_analysis.get('quality') == "high" and
                    range_analysis.get('touches', 0) >= 5):
                    recommended_strategy = "enhanced_mr"
                    logger.info(f"[{symbol}] ✅ FALLBACK: Exceptional range detected ({range_analysis.get('confidence', 0):.1%}) -> MR")
                elif (trend_strength >= 40 and                     # Very strong trend required
                      volatility_level == "low"):                  # Only in low volatility
                    recommended_strategy = "trend"
                    logger.info(f"[{symbol}] ✅ FALLBACK: Exceptional trend detected ({trend_strength:.1f}) -> Trend")
                else:
                    # No weak signals allowed - prefer safety
                    recommended_strategy = "none"
                    logger.info(f"[{symbol}] ❌ FALLBACK: No exceptional signals - safer to avoid trading")
            else:
                # High volatility markets get no fallback
                recommended_strategy = "none"
                logger.info(f"[{symbol}] ❌ FALLBACK: High/extreme volatility - no trading allowed")

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
            recommended_strategy=recommended_strategy,
            trend_probability=float(probability_map.get('trending', 0.0)),
            range_probability=float(probability_map.get('ranging', 0.0)),
            volatile_probability=float(probability_map.get('volatile', 0.0)),
            feature_snapshot=feature_snapshot
        )

    except Exception as e:
        logger.error(f"[{symbol}] Error in enhanced regime detection: {e}")
        # Return safe default - assume extreme volatility in error cases
        return RegimeAnalysis(
            primary_regime="volatile",
            regime_confidence=0.3,
            range_quality="low",
            trend_strength=50.0,
            volatility_level="extreme",  # Assume extreme volatility for safety
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
