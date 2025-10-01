import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Tuple
from dataclasses import dataclass
import asyncio

logger = logging.getLogger(__name__)

@dataclass
class SymbolMetrics:
    """Metrics for clustering analysis"""
    symbol: str
    avg_volatility: float  # Average ATR as % of price
    trend_persistence: float  # Average bars per trend
    volume_profile: float  # Average volume ratio
    btc_correlation: float  # Correlation with BTC
    avg_spread_pct: float  # Average spread as %
    price_level: int  # 1=<$0.01, 2=$0.01-0.1, 3=$0.1-1, 4=$1-10, 5=$10+

class SymbolClusterer:
    """Analyzes and clusters symbols based on behavior patterns"""
    
    def __init__(self, frames: Dict[str, pd.DataFrame]):
        self.frames = frames
        self.metrics = {}
        
    def calculate_metrics(self, min_candles: int = 500) -> Dict[str, SymbolMetrics]:
        """Calculate clustering metrics for each symbol"""
        
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
                # Calculate metrics
                metrics = self._calculate_symbol_metrics(symbol, df, btc_returns)
                if metrics:
                    self.metrics[symbol] = metrics
                    
            except Exception as e:
                logger.warning(f"Failed to calculate metrics for {symbol}: {e}")
                continue
                
        return self.metrics
    
    def _calculate_symbol_metrics(self, symbol: str, df: pd.DataFrame, btc_returns: pd.Series) -> SymbolMetrics:
        """Calculate metrics for a single symbol"""
        
        # ATR volatility
        high = df['high']
        low = df['low']
        close = df['close']
        
        # ATR calculation
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        avg_volatility = (atr / close).mean() * 100  # As percentage
        
        # Trend persistence (average trend length)
        # Simple approach: count consecutive up/down closes
        close_diff = close.diff()
        trend_changes = (close_diff.shift() * close_diff < 0).sum()
        avg_trend_length = len(df) / (trend_changes + 1)
        trend_persistence = min(avg_trend_length, 20)  # Cap at 20
        
        # Volume profile
        volume = df['volume']
        volume_ma = volume.rolling(20).mean()
        volume_ratio = (volume / volume_ma).mean()
        
        # BTC correlation (if not BTC itself)
        btc_corr = 0.0
        if symbol != 'BTCUSDT':
            sym_returns = close.pct_change().dropna()
            # Align indices
            aligned_returns = pd.concat([btc_returns, sym_returns], axis=1, join='inner')
            if len(aligned_returns) > 100:
                btc_corr = aligned_returns.corr().iloc[0, 1]
        
        # Average spread estimate (using high-low as proxy)
        avg_spread_pct = ((high - low) / close).mean() * 100
        
        # Price level categorization
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
            
        return SymbolMetrics(
            symbol=symbol,
            avg_volatility=float(avg_volatility),
            trend_persistence=float(trend_persistence),
            volume_profile=float(volume_ratio),
            btc_correlation=float(btc_corr),
            avg_spread_pct=float(avg_spread_pct),
            price_level=price_level
        )
    
    def cluster_symbols(self, n_clusters: int = 5) -> Dict[str, int]:
        """Perform clustering using simple rule-based approach"""
        
        if not self.metrics:
            logger.error("No metrics calculated for clustering")
            return {}
            
        clusters = {}
        
        # Updated for top 50 market cap symbols
        major_cryptos = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'AVAXUSDT', 
                        'LINKUSDT', 'DOTUSDT', 'BCHUSDT', 'LTCUSDT', 'NEARUSDT',
                        'ICPUSDT', 'ARBUSDT', 'OPUSDT', 'INJUSDT', 'ATOMUSDT']
        
        # No stablecoins in top 50 crypto market cap list
        stablecoins = []
        
        # Meme/gaming/speculative coins from top 50
        meme_coins = ['DOGEUSDT', '1000PEPEUSDT', 'FLOKIUSDT', 'GMTUSDT', 'APEUSDT', 
                     'GALAUSDT', 'SANDUSDT', 'MANAUSDT', 'AXSUSDT', 'CHZUSDT']
        
        for symbol, m in self.metrics.items():
            # Rule-based clustering for interpretability
            
            # Cluster 1: Major cryptocurrencies
            if symbol in major_cryptos:
                clusters[symbol] = 1
                
            # Cluster 2: Stablecoins
            elif symbol in stablecoins or m.avg_volatility < 1:
                clusters[symbol] = 2
                
            # Cluster 3: Meme/High volatility coins
            elif symbol in meme_coins or (m.avg_volatility > 5 and abs(m.btc_correlation) < 0.3):
                clusters[symbol] = 3
                
            # Cluster 4: Mid-cap alts (moderate everything, follows BTC)
            elif m.price_level >= 3 and abs(m.btc_correlation) > 0.5:
                clusters[symbol] = 4
                
            # Cluster 5: Small caps and others
            else:
                clusters[symbol] = 5
                
        return clusters
    
    def get_cluster_descriptions(self) -> Dict[int, str]:
        """Get human-readable cluster descriptions"""
        return {
            1: "Blue Chip (BTC, ETH, major coins)",
            2: "Stable/Low Volatility",
            3: "High Volatility/Meme Coins",
            4: "Mid-Cap Alts (BTC followers)",
            5: "Small Caps/Others"
        }
    
    def save_clusters(self, filepath: str = "symbol_clusters.json"):
        """Save clustering results to file"""
        
        clusters = self.cluster_symbols()
        
        # Create output with metadata
        output = {
            "generated_at": datetime.now().isoformat(),
            "cluster_descriptions": self.get_cluster_descriptions(),
            "symbol_clusters": clusters,
            "metrics_summary": {}
        }
        
        # Add metrics summary per cluster
        for cluster_id in range(1, 6):
            cluster_symbols = [s for s, c in clusters.items() if c == cluster_id]
            if cluster_symbols:
                cluster_metrics = [self.metrics[s] for s in cluster_symbols if s in self.metrics]
                
                output["metrics_summary"][cluster_id] = {
                    "count": len(cluster_symbols),
                    "symbols": cluster_symbols[:5],  # Show first 5 as examples
                    "avg_volatility": np.mean([m.avg_volatility for m in cluster_metrics]),
                    "avg_btc_correlation": np.mean([m.btc_correlation for m in cluster_metrics])
                }
        
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
            
        logger.info(f"Saved clustering results to {filepath}")
        return clusters


async def update_symbol_clusters(frames: Dict[str, pd.DataFrame]):
    """Main function to update symbol clusters"""
    
    logger.info("Starting symbol clustering analysis...")
    
    clusterer = SymbolClusterer(frames)
    
    # Calculate metrics
    metrics = clusterer.calculate_metrics(min_candles=300)  # Require at least 300 candles
    logger.info(f"Calculated metrics for {len(metrics)} symbols")
    
    # Perform clustering and save
    clusters = clusterer.save_clusters()
    logger.info(f"Clustering complete. Assigned {len(clusters)} symbols to clusters")
    
    # Print summary
    descriptions = clusterer.get_cluster_descriptions()
    for cluster_id, desc in descriptions.items():
        count = sum(1 for c in clusters.values() if c == cluster_id)
        logger.info(f"Cluster {cluster_id} ({desc}): {count} symbols")
    
    return clusters


# Standalone cluster loading function for strategy use
def load_symbol_clusters(filepath: str = "symbol_clusters.json") -> Dict[str, int]:
    """Load symbol clusters from file"""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            return data.get("symbol_clusters", {})
    except FileNotFoundError:
        logger.warning(f"Cluster file {filepath} not found. Using default cluster 3 for all symbols.")
        return {}
    except Exception as e:
        logger.error(f"Error loading clusters: {e}")
        return {}