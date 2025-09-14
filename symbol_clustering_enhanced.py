"""
Enhanced Symbol Clustering with Confidence Scores
Extends the original clustering with confidence scoring and dynamic updates
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances

logger = logging.getLogger(__name__)

@dataclass
class ClusterAssignment:
    """Enhanced cluster assignment with confidence scores"""
    primary_cluster: int
    primary_confidence: float
    secondary_cluster: Optional[int] = None
    secondary_confidence: Optional[float] = None
    distances: List[float] = None  # Distance to each cluster center
    is_borderline: bool = False  # True if confidence < 0.8

@dataclass
class EnhancedSymbolMetrics:
    """Extended metrics for better clustering"""
    symbol: str
    # Original metrics
    avg_volatility: float
    trend_persistence: float
    volume_profile: float
    btc_correlation: float
    avg_spread_pct: float
    price_level: int
    # New metrics for better clustering
    volatility_30d: float  # Recent volatility
    volume_trend_30d: float  # Volume increasing/decreasing
    price_momentum_30d: float  # Price trend strength
    correlation_stability: float  # How stable is BTC correlation
    
class EnhancedSymbolClusterer:
    """Enhanced clustering with confidence scores and dynamic updates"""
    
    CONFIDENCE_THRESHOLD = 0.8  # Below this = borderline
    MIGRATION_THRESHOLD = 0.4  # 40% behavior change triggers migration
    
    def __init__(self, frames: Dict[str, pd.DataFrame]):
        self.frames = frames
        self.metrics = {}
        self.cluster_centers = {}  # Store cluster centers for confidence calculation
        self.scaler = StandardScaler()
        
    def calculate_enhanced_metrics(self, min_candles: int = 500) -> Dict[str, EnhancedSymbolMetrics]:
        """Calculate enhanced clustering metrics for each symbol"""
        
        # Get BTC data for correlation
        btc_df = self.frames.get('BTCUSDT')
        if btc_df is None or len(btc_df) < min_candles:
            logger.error("BTC data not available for correlation")
            return {}
            
        btc_returns = btc_df['close'].pct_change().dropna()
        
        for symbol, df in self.frames.items():
            if len(df) < min_candles:
                continue
                
            try:
                # Calculate enhanced metrics
                metrics = self._calculate_enhanced_symbol_metrics(symbol, df, btc_returns)
                if metrics:
                    self.metrics[symbol] = metrics
                    
            except Exception as e:
                logger.warning(f"Failed to calculate enhanced metrics for {symbol}: {e}")
                continue
                
        return self.metrics
    
    def _calculate_enhanced_symbol_metrics(self, symbol: str, df: pd.DataFrame, btc_returns: pd.Series) -> EnhancedSymbolMetrics:
        """Calculate enhanced metrics for a single symbol"""
        
        # Original metrics calculation (same as before)
        high = df['high']
        low = df['low'] 
        close = df['close']
        volume = df['volume']
        
        # ATR calculation
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        avg_volatility = (atr / close).mean() * 100
        
        # Trend persistence
        close_diff = close.diff()
        trend_changes = (close_diff.shift() * close_diff < 0).sum()
        avg_trend_length = len(df) / (trend_changes + 1)
        trend_persistence = min(avg_trend_length, 20)
        
        # Volume profile
        volume_ma = volume.rolling(20).mean()
        volume_ratio = (volume / volume_ma).mean()
        
        # BTC correlation
        btc_corr = 0.0
        corr_stability = 1.0
        if symbol != 'BTCUSDT':
            sym_returns = close.pct_change().dropna()
            aligned_returns = pd.concat([btc_returns, sym_returns], axis=1, join='inner')
            if len(aligned_returns) > 100:
                btc_corr = aligned_returns.corr().iloc[0, 1]
                
                # Calculate correlation stability (rolling correlation std)
                rolling_corr = aligned_returns.iloc[:, 0].rolling(30).corr(aligned_returns.iloc[:, 1])
                corr_stability = 1.0 - rolling_corr.std() if not rolling_corr.std() != rolling_corr.std() else 1.0
        
        # Average spread
        avg_spread_pct = ((high - low) / close).mean() * 100
        
        # Price level
        avg_price = close.mean()
        if avg_price < 0.01:
            price_level = 1
        elif avg_price < 0.1:
            price_level = 2
        elif avg_price < 1:
            price_level = 3
        elif avg_price < 10:
            price_level = 4
        else:
            price_level = 5
            
        # NEW: 30-day metrics for dynamic updates
        recent_data = df.tail(30 * 24 * 4)  # Assuming 15min candles, ~30 days
        if len(recent_data) > 100:
            # Recent volatility
            recent_atr = tr.tail(30 * 24 * 4).rolling(14).mean()
            volatility_30d = (recent_atr / close.tail(len(recent_atr))).mean() * 100
            
            # Volume trend
            recent_vol = volume.tail(30 * 24 * 4)
            vol_start = recent_vol.head(len(recent_vol)//3).mean()
            vol_end = recent_vol.tail(len(recent_vol)//3).mean()
            volume_trend_30d = (vol_end - vol_start) / vol_start if vol_start > 0 else 0
            
            # Price momentum
            price_start = close.tail(30 * 24 * 4).iloc[0]
            price_end = close.iloc[-1]
            price_momentum_30d = (price_end - price_start) / price_start
        else:
            volatility_30d = avg_volatility
            volume_trend_30d = 0
            price_momentum_30d = 0
            
        return EnhancedSymbolMetrics(
            symbol=symbol,
            avg_volatility=float(avg_volatility),
            trend_persistence=float(trend_persistence),
            volume_profile=float(volume_ratio),
            btc_correlation=float(btc_corr),
            avg_spread_pct=float(avg_spread_pct),
            price_level=price_level,
            volatility_30d=float(volatility_30d),
            volume_trend_30d=float(volume_trend_30d),
            price_momentum_30d=float(price_momentum_30d),
            correlation_stability=float(corr_stability)
        )
    
    def cluster_with_confidence(self, n_clusters: int = 5) -> Dict[str, ClusterAssignment]:
        """Perform clustering with confidence scores"""
        
        if not self.metrics:
            logger.error("No metrics calculated for clustering")
            return {}
            
        # First, get basic clusters using original rules
        basic_clusters = self._get_basic_clusters()
        
        # Calculate cluster centers based on assigned symbols
        self._calculate_cluster_centers(basic_clusters)
        
        # Now calculate confidence scores for each symbol
        confidence_clusters = {}
        
        for symbol, metrics in self.metrics.items():
            # Calculate distances to all cluster centers
            distances = self._calculate_distances_to_clusters(metrics)
            
            # Sort clusters by distance (closest first)
            sorted_clusters = sorted(enumerate(distances, 1), key=lambda x: x[1])
            
            # Primary cluster (closest)
            primary_cluster = sorted_clusters[0][0]
            primary_distance = sorted_clusters[0][1]
            
            # Secondary cluster (second closest)
            secondary_cluster = sorted_clusters[1][0] if len(sorted_clusters) > 1 else None
            secondary_distance = sorted_clusters[1][1] if len(sorted_clusters) > 1 else None
            
            # Calculate confidence based on distance ratios
            if secondary_distance and secondary_distance > 0:
                # Confidence = how much closer we are to primary vs secondary
                confidence_ratio = primary_distance / secondary_distance
                primary_confidence = 1.0 - confidence_ratio
                primary_confidence = max(0.3, min(1.0, primary_confidence))  # Clamp between 0.3 and 1.0
                
                secondary_confidence = 1.0 - primary_confidence
            else:
                primary_confidence = 1.0
                secondary_confidence = 0.0
            
            # Determine if borderline
            is_borderline = primary_confidence < self.CONFIDENCE_THRESHOLD
            
            confidence_clusters[symbol] = ClusterAssignment(
                primary_cluster=primary_cluster,
                primary_confidence=primary_confidence,
                secondary_cluster=secondary_cluster if secondary_confidence > 0.2 else None,
                secondary_confidence=secondary_confidence if secondary_confidence > 0.2 else None,
                distances=distances,
                is_borderline=is_borderline
            )
            
        return confidence_clusters
    
    def _get_basic_clusters(self) -> Dict[str, int]:
        """Get basic clusters using original rules"""
        clusters = {}
        
        for symbol, m in self.metrics.items():
            # Same rules as original
            if m.price_level >= 4 and m.avg_volatility < 3 and abs(m.btc_correlation) > 0.6:
                clusters[symbol] = 1
            elif m.avg_volatility < 1:
                clusters[symbol] = 2
            elif m.avg_volatility > 5 and abs(m.btc_correlation) < 0.3:
                clusters[symbol] = 3
            elif m.price_level >= 3 and abs(m.btc_correlation) > 0.5:
                clusters[symbol] = 4
            else:
                clusters[symbol] = 5
                
        return clusters
    
    def _calculate_cluster_centers(self, basic_clusters: Dict[str, int]):
        """Calculate the center point of each cluster"""
        # Group symbols by cluster
        cluster_groups = {}
        for symbol, cluster in basic_clusters.items():
            if cluster not in cluster_groups:
                cluster_groups[cluster] = []
            cluster_groups[cluster].append(symbol)
        
        # Calculate average metrics for each cluster
        self.cluster_centers = {}
        for cluster_id, symbols in cluster_groups.items():
            cluster_metrics = [self.metrics[s] for s in symbols if s in self.metrics]
            if cluster_metrics:
                self.cluster_centers[cluster_id] = {
                    'avg_volatility': np.mean([m.avg_volatility for m in cluster_metrics]),
                    'trend_persistence': np.mean([m.trend_persistence for m in cluster_metrics]),
                    'volume_profile': np.mean([m.volume_profile for m in cluster_metrics]),
                    'btc_correlation': np.mean([m.btc_correlation for m in cluster_metrics]),
                    'price_level': np.mean([m.price_level for m in cluster_metrics])
                }
    
    def _calculate_distances_to_clusters(self, metrics: EnhancedSymbolMetrics) -> List[float]:
        """Calculate normalized distance to each cluster center"""
        distances = []
        
        # Create feature vector for this symbol
        symbol_features = np.array([
            metrics.avg_volatility,
            metrics.trend_persistence,
            metrics.volume_profile,
            metrics.btc_correlation,
            metrics.price_level
        ])
        
        # Calculate distance to each cluster
        for cluster_id in range(1, 6):
            if cluster_id in self.cluster_centers:
                center = self.cluster_centers[cluster_id]
                center_features = np.array([
                    center['avg_volatility'],
                    center['trend_persistence'],
                    center['volume_profile'],
                    center['btc_correlation'],
                    center['price_level']
                ])
                
                # Normalize features before distance calculation
                # Simple min-max normalization
                distance = np.sqrt(
                    ((symbol_features[0] - center_features[0]) / 10) ** 2 +  # Volatility scale ~10
                    ((symbol_features[1] - center_features[1]) / 20) ** 2 +  # Persistence scale ~20
                    ((symbol_features[2] - center_features[2]) / 2) ** 2 +   # Volume ratio scale ~2
                    ((symbol_features[3] - center_features[3])) ** 2 +       # Correlation already -1 to 1
                    ((symbol_features[4] - center_features[4]) / 5) ** 2     # Price level scale 1-5
                )
                distances.append(float(distance))
            else:
                distances.append(float('inf'))
                
        return distances
    
    def save_enhanced_clusters(self, filepath: str = "symbol_clusters_enhanced.json"):
        """Save enhanced clustering results to file"""
        
        confidence_clusters = self.cluster_with_confidence()
        
        # Create backward-compatible output
        output = {
            "generated_at": datetime.now().isoformat(),
            "version": "enhanced_v1",
            "cluster_descriptions": {
                1: "Blue Chip (BTC, ETH, major coins)",
                2: "Stable/Low Volatility",
                3: "High Volatility/Meme Coins", 
                4: "Mid-Cap Alts (BTC followers)",
                5: "Small Caps/Others"
            },
            # Backward compatible simple mapping
            "symbol_clusters": {
                symbol: assignment.primary_cluster 
                for symbol, assignment in confidence_clusters.items()
            },
            # New enhanced data
            "enhanced_clusters": {
                symbol: {
                    "primary_cluster": assignment.primary_cluster,
                    "primary_confidence": round(assignment.primary_confidence, 3),
                    "secondary_cluster": assignment.secondary_cluster,
                    "secondary_confidence": round(assignment.secondary_confidence, 3) if assignment.secondary_confidence else None,
                    "is_borderline": assignment.is_borderline,
                    "distances": [round(d, 3) for d in assignment.distances] if assignment.distances else None
                }
                for symbol, assignment in confidence_clusters.items()
            },
            "metrics_summary": {}
        }
        
        # Add metrics summary per cluster
        for cluster_id in range(1, 6):
            cluster_symbols = [s for s, a in confidence_clusters.items() if a.primary_cluster == cluster_id]
            if cluster_symbols:
                cluster_metrics = [self.metrics[s] for s in cluster_symbols if s in self.metrics]
                
                # Count borderline symbols
                borderline_count = sum(1 for s in cluster_symbols 
                                     if confidence_clusters[s].is_borderline)
                
                output["metrics_summary"][cluster_id] = {
                    "count": len(cluster_symbols),
                    "borderline_count": borderline_count,
                    "symbols": cluster_symbols[:5],
                    "avg_volatility": round(np.mean([m.avg_volatility for m in cluster_metrics]), 2),
                    "avg_btc_correlation": round(np.mean([m.btc_correlation for m in cluster_metrics]), 3),
                    "avg_confidence": round(np.mean([confidence_clusters[s].primary_confidence 
                                                   for s in cluster_symbols]), 3)
                }
        
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
            
        logger.info(f"Saved enhanced clustering results to {filepath}")
        return confidence_clusters


def load_enhanced_clusters(filepath: str = "symbol_clusters_enhanced.json") -> Tuple[Dict[str, int], Dict[str, Dict]]:
    """
    Load enhanced clusters from file
    Returns: (simple_clusters, enhanced_data)
    - simple_clusters: backward compatible dict of symbol -> cluster_id
    - enhanced_data: dict of symbol -> enhanced cluster info
    """
    try:
        # Try current directory first
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
        # Try script directory
        elif os.path.exists(os.path.join(os.path.dirname(__file__), filepath)):
            filepath = os.path.join(os.path.dirname(__file__), filepath)
            with open(filepath, 'r') as f:
                data = json.load(f)
        else:
            raise FileNotFoundError(f"Could not find {filepath}")
            
        simple_clusters = data.get("symbol_clusters", {})
        enhanced_data = data.get("enhanced_clusters", {})
        
        logger.info(f"Successfully loaded enhanced clusters from {filepath}")
        logger.info(f"Found {len(simple_clusters)} simple clusters and {len(enhanced_data)} enhanced entries")
        
        return simple_clusters, enhanced_data
        
    except FileNotFoundError:
        logger.warning(f"Enhanced cluster file {filepath} not found. Using original clusters.")
        # Fall back to original clusters
        try:
            from symbol_clustering import load_symbol_clusters
            simple_clusters = load_symbol_clusters()
            # Create dummy enhanced data
            enhanced_data = {
                symbol: {
                    "primary_cluster": cluster,
                    "primary_confidence": 1.0,
                    "secondary_cluster": None,
                    "secondary_confidence": None,
                    "is_borderline": False
                }
                for symbol, cluster in simple_clusters.items()
            }
            return simple_clusters, enhanced_data
        except:
            return {}, {}
    except Exception as e:
        logger.error(f"Error loading enhanced clusters: {e}")
        return {}, {}