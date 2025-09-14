"""
Dynamic Cluster Updater
Monitors symbol behavior changes and updates clusters weekly
"""
import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict

from symbol_clustering_enhanced import EnhancedSymbolClusterer, load_enhanced_clusters
from candle_storage_postgres import CandleStorage
from telegram_bot import TGBot

logger = logging.getLogger(__name__)

@dataclass 
class ClusterMigration:
    """Record of a symbol migrating between clusters"""
    symbol: str
    old_cluster: int
    new_cluster: int
    old_confidence: float
    new_confidence: float
    reason: str
    timestamp: datetime
    metrics_change: Dict[str, float]

class DynamicClusterUpdater:
    """Updates clusters based on recent behavior changes"""
    
    BEHAVIOR_CHANGE_THRESHOLD = 0.4  # 40% change triggers review
    MIN_DAYS_BETWEEN_UPDATES = 7  # Don't update more than weekly
    MIN_CANDLES_FOR_UPDATE = 1000  # Need enough recent data
    
    def __init__(self, storage: CandleStorage = None):
        self.storage = storage or CandleStorage()
        self.last_update_time = None
        self.migrations = []  # Track all migrations
        
    async def check_and_update_clusters(self, symbols: List[str], force: bool = False) -> Dict[str, ClusterMigration]:
        """
        Check if clusters need updating and perform update if needed
        Returns dict of symbol -> migration info for changed symbols
        """
        
        # Check if enough time has passed
        if not force and self.last_update_time:
            days_since_update = (datetime.now() - self.last_update_time).days
            if days_since_update < self.MIN_DAYS_BETWEEN_UPDATES:
                logger.info(f"Skipping cluster update - only {days_since_update} days since last update")
                return {}
        
        logger.info("Starting dynamic cluster update check...")
        
        # Load current clusters
        current_simple, current_enhanced = load_enhanced_clusters()
        if not current_enhanced:
            logger.warning("No enhanced clusters found, cannot perform dynamic update")
            return {}
        
        # Load recent data for all symbols
        frames = {}
        for symbol in symbols:
            try:
                df = self.storage.load_candles(symbol, limit=self.MIN_CANDLES_FOR_UPDATE)
                if df is not None and len(df) >= self.MIN_CANDLES_FOR_UPDATE:
                    frames[symbol] = df
                    logger.debug(f"Loaded {len(df)} candles for {symbol}")
            except Exception as e:
                logger.warning(f"Failed to load data for {symbol}: {e}")
        
        if not frames:
            logger.error("No data loaded for cluster update")
            return {}
        
        # Calculate new clusters based on recent behavior
        logger.info(f"Analyzing {len(frames)} symbols for behavior changes...")
        clusterer = EnhancedSymbolClusterer(frames)
        new_metrics = clusterer.calculate_enhanced_metrics(min_candles=500)
        new_assignments = clusterer.cluster_with_confidence()
        
        # Compare and identify migrations
        migrations = {}
        
        for symbol, new_assign in new_assignments.items():
            if symbol in current_enhanced:
                old_data = current_enhanced[symbol]
                old_cluster = old_data['primary_cluster']
                old_confidence = old_data['primary_confidence']
                
                # Check if primary cluster changed
                if new_assign.primary_cluster != old_cluster:
                    # Calculate what changed
                    metrics_change = self._calculate_metrics_change(
                        symbol, new_metrics.get(symbol), clusterer.metrics
                    )
                    
                    # Determine migration reason
                    reason = self._determine_migration_reason(
                        old_cluster, new_assign.primary_cluster, metrics_change
                    )
                    
                    migration = ClusterMigration(
                        symbol=symbol,
                        old_cluster=old_cluster,
                        new_cluster=new_assign.primary_cluster,
                        old_confidence=old_confidence,
                        new_confidence=new_assign.primary_confidence,
                        reason=reason,
                        timestamp=datetime.now(),
                        metrics_change=metrics_change
                    )
                    
                    migrations[symbol] = migration
                    self.migrations.append(migration)
                    
                    logger.info(f"CLUSTER MIGRATION: {symbol} from Cluster {old_cluster} to {new_assign.primary_cluster}")
                    logger.info(f"  Reason: {reason}")
                    logger.info(f"  Confidence: {old_confidence:.2f} -> {new_assign.primary_confidence:.2f}")
                
                # Also check if confidence dropped significantly (became more borderline)
                elif old_confidence - new_assign.primary_confidence > 0.2:
                    logger.warning(f"{symbol} confidence dropped from {old_confidence:.2f} to {new_assign.primary_confidence:.2f}")
        
        # Save updated clusters if there were changes
        if migrations:
            logger.info(f"Found {len(migrations)} cluster migrations, saving updated clusters...")
            clusterer.save_enhanced_clusters()
            self.last_update_time = datetime.now()
            
            # Save migration history
            self._save_migration_history()
        else:
            logger.info("No cluster migrations detected")
        
        return migrations
    
    def _calculate_metrics_change(self, symbol: str, new_metrics, all_metrics) -> Dict[str, float]:
        """Calculate percentage change in key metrics"""
        changes = {}
        
        if not new_metrics or symbol not in all_metrics:
            return changes
        
        old_metrics = all_metrics[symbol]
        
        # Compare 30-day metrics vs overall metrics
        if old_metrics.avg_volatility > 0:
            changes['volatility_change'] = (new_metrics.volatility_30d - old_metrics.avg_volatility) / old_metrics.avg_volatility
        
        changes['volume_trend'] = new_metrics.volume_trend_30d
        changes['price_momentum'] = new_metrics.price_momentum_30d
        changes['correlation_change'] = new_metrics.btc_correlation - old_metrics.btc_correlation
        
        return changes
    
    def _determine_migration_reason(self, old_cluster: int, new_cluster: int, metrics_change: Dict[str, float]) -> str:
        """Determine the primary reason for cluster migration"""
        reasons = []
        
        # Check volatility changes
        if 'volatility_change' in metrics_change:
            vol_change = metrics_change['volatility_change']
            if abs(vol_change) > 0.3:
                direction = "increased" if vol_change > 0 else "decreased"
                reasons.append(f"Volatility {direction} by {abs(vol_change)*100:.0f}%")
        
        # Check volume trends
        if 'volume_trend' in metrics_change:
            vol_trend = metrics_change['volume_trend']
            if abs(vol_trend) > 0.5:
                direction = "surged" if vol_trend > 0 else "declined"
                reasons.append(f"Volume {direction} by {abs(vol_trend)*100:.0f}%")
        
        # Check correlation changes
        if 'correlation_change' in metrics_change:
            corr_change = metrics_change['correlation_change']
            if abs(corr_change) > 0.3:
                direction = "increased" if corr_change > 0 else "decreased"
                reasons.append(f"BTC correlation {direction} to {corr_change:.2f}")
        
        # Cluster-specific reasons
        if old_cluster == 3 and new_cluster == 4:  # Meme to Mid-cap
            reasons.append("Matured from meme to established")
        elif old_cluster == 4 and new_cluster == 3:  # Mid-cap to Meme
            reasons.append("Increased speculative behavior")
        elif old_cluster == 5 and new_cluster in [3, 4]:  # Small cap promoted
            reasons.append("Gained market significance")
        
        return " | ".join(reasons) if reasons else "Behavior pattern change"
    
    def _save_migration_history(self, filepath: str = "cluster_migration_history.json"):
        """Save migration history to file"""
        try:
            history = {
                "last_update": self.last_update_time.isoformat() if self.last_update_time else None,
                "migrations": [
                    {
                        "symbol": m.symbol,
                        "old_cluster": m.old_cluster,
                        "new_cluster": m.new_cluster,
                        "old_confidence": m.old_confidence,
                        "new_confidence": m.new_confidence,
                        "reason": m.reason,
                        "timestamp": m.timestamp.isoformat(),
                        "metrics_change": m.metrics_change
                    }
                    for m in self.migrations[-100:]  # Keep last 100 migrations
                ]
            }
            
            with open(filepath, 'w') as f:
                json.dump(history, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save migration history: {e}")
    
    def get_migration_summary(self) -> str:
        """Get a summary of recent migrations for reporting"""
        if not self.migrations:
            return "No cluster migrations recorded"
        
        recent_migrations = self.migrations[-10:]  # Last 10 migrations
        
        summary = "Recent Cluster Migrations:\n"
        for m in recent_migrations:
            summary += f"\n{m.symbol}: Cluster {m.old_cluster} → {m.new_cluster}\n"
            summary += f"  Confidence: {m.old_confidence:.2%} → {m.new_confidence:.2%}\n"
            summary += f"  Reason: {m.reason}\n"
            summary += f"  Time: {m.timestamp.strftime('%Y-%m-%d %H:%M')}\n"
        
        return summary


async def run_cluster_update(symbols: List[str], telegram_bot: Optional[TGBot] = None):
    """
    Run cluster update and notify via Telegram
    This can be scheduled to run weekly
    """
    updater = DynamicClusterUpdater()
    
    try:
        # Check and update clusters
        migrations = await updater.check_and_update_clusters(symbols)
        
        if migrations and telegram_bot:
            # Send notification about migrations
            message = "🔄 *Cluster Update Complete*\n\n"
            
            if migrations:
                message += f"Found {len(migrations)} symbol migrations:\n\n"
                
                for symbol, migration in list(migrations.items())[:5]:  # Show first 5
                    cluster_names = {
                        1: "Blue Chip",
                        2: "Stable",
                        3: "Meme/Volatile", 
                        4: "Mid-Cap",
                        5: "Small Cap"
                    }
                    
                    old_name = cluster_names.get(migration.old_cluster, "Unknown")
                    new_name = cluster_names.get(migration.new_cluster, "Unknown")
                    
                    message += f"• *{symbol}*: {old_name} → {new_name}\n"
                    message += f"  Reason: {migration.reason}\n\n"
                
                if len(migrations) > 5:
                    message += f"... and {len(migrations) - 5} more\n"
            else:
                message += "No migrations needed - all symbols stable"
            
            await telegram_bot.send_message(message)
            
    except Exception as e:
        logger.error(f"Cluster update failed: {e}")
        if telegram_bot:
            await telegram_bot.send_message(f"❌ Cluster update failed: {str(e)[:100]}")


if __name__ == "__main__":
    # Test the updater
    import yaml
    
    async def test():
        # Load config for symbols
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
            symbols = config['trade']['symbols']
        
        updater = DynamicClusterUpdater()
        migrations = await updater.check_and_update_clusters(symbols[:20], force=True)  # Test with first 20
        
        print(f"\nFound {len(migrations)} migrations")
        print(updater.get_migration_summary())
    
    asyncio.run(test())