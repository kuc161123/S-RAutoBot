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
    Enhance ML features with cluster data using hardcoded clusters
    This is backward compatible - adds new features without breaking existing ones
    """
    # Use hardcoded clusters
    try:
        from hardcoded_clusters import get_symbol_cluster
        features['symbol_cluster'] = get_symbol_cluster(symbol)
    except ImportError:
        # Fallback to old method if hardcoded_clusters not available
        simple_clusters, enhanced_clusters = load_cluster_data()
        if symbol in simple_clusters:
            features['symbol_cluster'] = simple_clusters[symbol]
        else:
            features['symbol_cluster'] = 3  # Default to volatile cluster
    
    # Keep features for compatibility but set to neutral values
    # This prevents the broken borderline detection from confusing ML
    features['cluster_confidence'] = 1.0  # Always confident
    features['cluster_secondary'] = 0     # No secondary cluster
    features['cluster_mixed'] = 0         # Never mixed/borderline
    features['cluster_conf_ratio'] = 10.0 # High confidence ratio
    
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
    
    # Add MTF features if available
    try:
        from multi_timeframe_sr import mtf_sr
        
        # Get current price from features if available
        current_price = features.get('entry_price', 0)
        if current_price > 0:
            # Get nearest MTF levels
            nearest = mtf_sr.get_nearest_levels(symbol, current_price, above_count=1, below_count=1)
            
            # Check if near resistance
            if nearest['resistance']:
                res_distance = abs(nearest['resistance'][0] - current_price) / current_price
                if res_distance < 0.005:  # Within 0.5%
                    features['near_major_resistance'] = 1
                    features['mtf_level_strength'] = min(10.0, mtf_sr.get_level_strength(symbol, nearest['resistance'][0]))
                else:
                    features['near_major_resistance'] = 0
            
            # Check if near support
            if nearest['support']:
                sup_distance = abs(current_price - nearest['support'][0]) / current_price
                if sup_distance < 0.005:  # Within 0.5%
                    features['near_major_support'] = 1
                    if features['mtf_level_strength'] == 0:  # Don't override resistance strength
                        features['mtf_level_strength'] = min(10.0, mtf_sr.get_level_strength(symbol, nearest['support'][0]))
                else:
                    features['near_major_support'] = 0
                    
    except Exception as e:
        logger.debug(f"MTF features not available: {e}")
        # Keep default values
    
    return features

def get_cluster_description(symbol: str) -> str:
    """Get human-readable cluster description for a symbol using hardcoded clusters"""
    try:
        from hardcoded_clusters import get_symbol_cluster, get_cluster_name
        cluster_id = get_symbol_cluster(symbol)
        return get_cluster_name(cluster_id)
    except ImportError:
        # Fallback to old method
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
            return cluster_names.get(primary, 'Unknown')
        elif symbol in simple_clusters:
            cluster = simple_clusters[symbol]
            return cluster_names.get(cluster, 'Unknown')
        else:
            return "Small Cap"  # Default for unknown symbols

def should_adjust_risk_for_cluster(symbol: str) -> Tuple[bool, float, str]:
    """
    Determine if risk should be adjusted based on hardcoded cluster
    Returns: (should_adjust, multiplier, reason)
    """
    try:
        from hardcoded_clusters import get_symbol_cluster, get_cluster_name
        cluster_id = get_symbol_cluster(symbol)
        cluster_name = get_cluster_name(cluster_id)
        
        # Risk adjustments based on cluster type
        if cluster_id == 1:  # Blue Chip
            return True, 1.2, f"Blue Chip asset - stable and liquid"
        elif cluster_id == 2:  # Stable
            return True, 0.5, f"Stablecoin - minimal volatility"
        elif cluster_id == 3:  # Meme/Volatile
            return True, 0.8, f"Meme/Volatile - high risk"
        elif cluster_id == 4:  # Mid-Cap
            return False, 1.0, f"Mid-Cap Alt - standard risk"
        else:  # Small Cap
            return True, 0.7, f"Small Cap - higher uncertainty"
            
    except ImportError:
        # Fallback to no adjustment if hardcoded clusters not available
        return False, 1.0, "Using default risk"