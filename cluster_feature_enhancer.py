"""
Cluster Feature Enhancer
Safely adds enhanced cluster features to ML without breaking existing code
"""
import logging
from typing import Dict, Optional, Tuple
from symbol_clustering_enhanced import load_enhanced_clusters

logger = logging.getLogger(__name__)

# Cache for enhanced clusters
_enhanced_clusters_cache = None
_simple_clusters_cache = None
_cache_loaded = False

def load_cluster_data(force_reload=False):
    """Load enhanced cluster data once and cache it"""
    global _enhanced_clusters_cache, _simple_clusters_cache, _cache_loaded
    
    if not _cache_loaded or force_reload:
        try:
            simple, enhanced = load_enhanced_clusters()
            _simple_clusters_cache = simple
            _enhanced_clusters_cache = enhanced
            _cache_loaded = True
            logger.info(f"Loaded enhanced clusters for {len(enhanced)} symbols")
        except Exception as e:
            logger.warning(f"Failed to load enhanced clusters: {e}. Using defaults.")
            _simple_clusters_cache = {}
            _enhanced_clusters_cache = {}
            _cache_loaded = True
    
    return _simple_clusters_cache, _enhanced_clusters_cache

def reload_cluster_cache():
    """Force reload the cluster cache"""
    global _cache_loaded
    _cache_loaded = False
    return load_cluster_data(force_reload=True)

def enhance_ml_features(features: Dict, symbol: str) -> Dict:
    """
    Enhance ML features with cluster confidence data
    This is backward compatible - adds new features without breaking existing ones
    """
    # Load cluster data
    simple_clusters, enhanced_clusters = load_cluster_data()
    
    # Get simple cluster (backward compatible)
    if symbol in simple_clusters:
        features['symbol_cluster'] = simple_clusters[symbol]
    else:
        features['symbol_cluster'] = 3  # Default to volatile cluster
    
    # Add enhanced features if available
    if symbol in enhanced_clusters:
        enhanced = enhanced_clusters[symbol]
        
        # Primary cluster confidence
        features['cluster_confidence'] = enhanced.get('primary_confidence', 1.0)
        
        # Secondary cluster (0 if none)
        features['cluster_secondary'] = enhanced.get('secondary_cluster', 0) or 0
        
        # Mixed behavior flag
        features['cluster_mixed'] = 1 if enhanced.get('is_borderline', False) else 0
        
        # Confidence ratio (primary vs secondary)
        if enhanced.get('secondary_confidence'):
            features['cluster_conf_ratio'] = enhanced['primary_confidence'] / (enhanced['secondary_confidence'] + 0.001)
        else:
            features['cluster_conf_ratio'] = 10.0  # High ratio = very confident
            
        logger.debug(f"{symbol}: Cluster {features['symbol_cluster']} "
                    f"(conf: {features['cluster_confidence']:.2f}, "
                    f"mixed: {features['cluster_mixed']})")
    else:
        # Default enhanced features
        features['cluster_confidence'] = 1.0
        features['cluster_secondary'] = 0
        features['cluster_mixed'] = 0
        features['cluster_conf_ratio'] = 10.0
    
    # Normalize cluster features based on primary cluster
    # This helps ML understand relative behavior within cluster
    primary_cluster = features['symbol_cluster']
    
    # Cluster-based normalization factors
    cluster_volatility_norms = {
        1: 2.5,   # Blue chip - low volatility baseline
        2: 0.1,   # Stable - very low volatility
        3: 8.5,   # Meme - high volatility
        4: 4.2,   # Mid-cap - moderate volatility
        5: 6.5    # Small cap - higher volatility
    }
    
    cluster_volume_norms = {
        1: 5.0,   # Blue chip - high volume
        2: 1.0,   # Stable - low volume
        3: 3.0,   # Meme - moderate volume
        4: 2.5,   # Mid-cap - moderate volume
        5: 1.5    # Small cap - lower volume
    }
    
    # Apply normalization
    features['cluster_volatility_norm'] = cluster_volatility_norms.get(primary_cluster, 5.0)
    features['cluster_volume_norm'] = cluster_volume_norms.get(primary_cluster, 2.0)
    
    return features

def get_cluster_description(symbol: str) -> str:
    """Get human-readable cluster description for a symbol"""
    simple_clusters, enhanced_clusters = load_cluster_data()
    
    cluster_names = {
        1: "Blue Chip",
        2: "Stable",
        3: "Meme/Volatile",
        4: "Mid-Cap Alt",
        5: "Small Cap"
    }
    
    if symbol in enhanced_clusters:
        enhanced = enhanced_clusters[symbol]
        primary = enhanced.get('primary_cluster', 3)
        confidence = enhanced.get('primary_confidence', 1.0)
        secondary = enhanced.get('secondary_cluster')
        
        desc = f"{cluster_names.get(primary, 'Unknown')} ({confidence:.0%})"
        
        if secondary and enhanced.get('secondary_confidence', 0) > 0.2:
            sec_conf = enhanced.get('secondary_confidence', 0)
            desc += f" / {cluster_names.get(secondary, 'Unknown')} ({sec_conf:.0%})"
            
        if enhanced.get('is_borderline'):
            desc += " [Borderline]"
            
        return desc
    elif symbol in simple_clusters:
        cluster = simple_clusters[symbol]
        return cluster_names.get(cluster, 'Unknown')
    else:
        return "Unclustered"

def should_adjust_risk_for_cluster(symbol: str) -> Tuple[bool, float, str]:
    """
    Determine if risk should be adjusted based on cluster confidence
    Returns: (should_adjust, multiplier, reason)
    """
    _, enhanced_clusters = load_cluster_data()
    
    if symbol not in enhanced_clusters:
        return False, 1.0, "No cluster data"
    
    enhanced = enhanced_clusters[symbol]
    confidence = enhanced.get('primary_confidence', 1.0)
    is_borderline = enhanced.get('is_borderline', False)
    
    # Borderline symbols = more uncertainty = lower risk
    if is_borderline:
        return True, 0.8, f"Borderline cluster (conf: {confidence:.0%})"
    
    # Very low confidence = high uncertainty
    if confidence < 0.6:
        return True, 0.7, f"Low cluster confidence ({confidence:.0%})"
    
    # High confidence in volatile cluster = normal risk
    # High confidence in stable cluster = can increase risk slightly
    primary_cluster = enhanced.get('primary_cluster', 3)
    if primary_cluster in [1, 2] and confidence > 0.9:  # Blue chip or stable
        return True, 1.2, f"High confidence stable asset ({confidence:.0%})"
    
    return False, 1.0, "Normal confidence"