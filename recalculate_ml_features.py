#!/usr/bin/env python3
"""
Safe feature recalculation for historical trades
Recalculates all 22 ML features for stored trades using historical candle data
"""
import os
import sys
import json
import redis
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pickle
import base64

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strategy_pullback_ml_learning import calculate_ml_features, BreakoutState
from candle_storage_postgres import CandleStorage
from phantom_trade_tracker import PhantomTrade

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FeatureRecalculator:
    """Safely recalculates features for historical trades"""
    
    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.redis_client = None
        self.candle_storage = CandleStorage()
        self.backup_data = {}
        self.stats = {
            'executed_trades_processed': 0,
            'phantom_trades_processed': 0,
            'features_updated': 0,
            'errors': 0
        }
        
        # Initialize Redis
        self._init_redis()
        
    def _init_redis(self):
        """Initialize Redis connection"""
        try:
            redis_url = os.getenv('REDIS_URL')
            if redis_url:
                self.redis_client = redis.from_url(redis_url, decode_responses=True)
                self.redis_client.ping()
                logger.info("Connected to Redis")
            else:
                logger.error("No REDIS_URL found - cannot proceed")
                sys.exit(1)
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            sys.exit(1)
    
    def backup_data(self):
        """Create backup of all trade data before modification"""
        logger.info("Creating backup of all trade data...")
        
        try:
            # Backup executed trades
            executed_trades = self.redis_client.lrange('iml:trades', 0, -1)
            self.backup_data['executed_trades'] = executed_trades.copy()
            
            # Backup phantom trades
            phantom_data = self.redis_client.get('phantom:completed')
            if phantom_data:
                self.backup_data['phantom_trades'] = phantom_data
            
            # Save backup to file
            backup_file = f"ml_features_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(backup_file, 'w') as f:
                json.dump(self.backup_data, f, indent=2)
            
            logger.info(f"Backup saved to {backup_file}")
            logger.info(f"Backed up {len(executed_trades)} executed trades")
            
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            sys.exit(1)
    
    def get_candle_data_for_trade(self, symbol: str, trade_time: datetime) -> Optional[pd.DataFrame]:
        """Get historical candle data around trade time"""
        try:
            # Load recent candles from storage
            df = self.candle_storage.load_candles(symbol, limit=500)
            
            if df is None or df.empty:
                logger.warning(f"No candle data for {symbol}")
                return None
            
            # Filter candles before trade time
            if isinstance(trade_time, str):
                trade_time = datetime.fromisoformat(trade_time.replace('Z', '+00:00'))
            
            # Get candles up to trade time
            mask = df.index <= trade_time
            historical_df = df[mask]
            
            if len(historical_df) < 200:  # Need enough history for indicators
                logger.warning(f"Insufficient history for {symbol} at {trade_time}")
                return None
                
            return historical_df.tail(200)  # Return last 200 candles before trade
            
        except Exception as e:
            logger.error(f"Error getting candles for {symbol}: {e}")
            return None
    
    def recalculate_features(self, trade_data: dict, df: pd.DataFrame) -> dict:
        """Recalculate all 22 features using complete calculation"""
        try:
            # Create a simple state object
            state = BreakoutState()
            
            # Extract basic info from stored features
            side = trade_data['features'].get('side', 'long')
            
            # Use stored retracement or calculate a default
            retracement = trade_data['features'].get('retracement_pct', 50.0)
            if retracement == 0:
                retracement = 50.0  # Default to middle
            
            # Recalculate all features with new complete function
            new_features = calculate_ml_features(df, state, side, retracement)
            
            # Log what changed
            old_features = trade_data['features']
            changed_features = []
            for key, new_val in new_features.items():
                old_val = old_features.get(key, 'missing')
                if old_val == 'missing' or abs(float(old_val or 0) - float(new_val or 0)) > 0.01:
                    changed_features.append(f"{key}: {old_val} → {new_val:.2f}")
            
            if changed_features and len(changed_features) <= 5:
                logger.debug(f"Updated features: {', '.join(changed_features[:5])}")
            
            return new_features
            
        except Exception as e:
            logger.error(f"Error recalculating features: {e}")
            return trade_data['features']  # Return original on error
    
    def process_executed_trades(self):
        """Process and update executed trades"""
        logger.info("Processing executed trades...")
        
        trades = self.redis_client.lrange('iml:trades', 0, -1)
        updated_trades = []
        
        for i, trade_json in enumerate(trades):
            try:
                trade = json.loads(trade_json)
                
                # Get symbol from features or try to parse from data
                symbol = None
                if 'symbol' in trade:
                    symbol = trade['symbol']
                elif 'features' in trade and 'symbol' in trade['features']:
                    symbol = trade['features']['symbol']
                
                if not symbol:
                    logger.warning(f"Trade {i} has no symbol, skipping")
                    updated_trades.append(trade_json)
                    continue
                
                # Get candle data
                trade_time = trade.get('timestamp', datetime.now().isoformat())
                df = self.get_candle_data_for_trade(symbol, trade_time)
                
                if df is not None:
                    # Recalculate features
                    old_features = trade['features'].copy()
                    new_features = self.recalculate_features(trade, df)
                    
                    # Update trade
                    trade['features'] = new_features
                    trade['features_version'] = 'v2_complete'
                    
                    # Check if features actually changed
                    if old_features != new_features:
                        self.stats['features_updated'] += 1
                        logger.info(f"Updated features for {symbol} trade {i+1}/{len(trades)}")
                    
                    updated_trades.append(json.dumps(trade))
                    self.stats['executed_trades_processed'] += 1
                else:
                    # Keep original if no candle data
                    updated_trades.append(trade_json)
                    
            except Exception as e:
                logger.error(f"Error processing trade {i}: {e}")
                updated_trades.append(trade_json)  # Keep original on error
                self.stats['errors'] += 1
        
        # Update Redis if not dry run
        if not self.dry_run:
            logger.info("Updating executed trades in Redis...")
            pipe = self.redis_client.pipeline()
            pipe.delete('iml:trades')
            for trade in updated_trades:
                pipe.rpush('iml:trades', trade)
            pipe.execute()
            logger.info(f"Updated {len(updated_trades)} executed trades")
        else:
            logger.info(f"DRY RUN: Would update {len(updated_trades)} executed trades")
    
    def process_phantom_trades(self):
        """Process and update phantom trades"""
        logger.info("Processing phantom trades...")
        
        phantom_data = self.redis_client.get('phantom:completed')
        if not phantom_data:
            logger.info("No phantom trades to process")
            return
        
        try:
            phantom_dict = json.loads(phantom_data)
            updated_phantom_dict = {}
            
            for symbol, trades in phantom_dict.items():
                updated_trades = []
                
                for trade in trades:
                    try:
                        # Get candle data
                        trade_time = trade.get('signal_time', datetime.now().isoformat())
                        df = self.get_candle_data_for_trade(symbol, trade_time)
                        
                        if df is not None:
                            # Create a trade data structure compatible with recalculate_features
                            trade_data = {'features': trade.get('features', {})}
                            
                            # Recalculate features
                            old_features = trade['features'].copy()
                            new_features = self.recalculate_features(trade_data, df)
                            
                            # Update trade
                            trade['features'] = new_features
                            trade['features_version'] = 'v2_complete'
                            
                            # Check if features actually changed
                            if old_features != new_features:
                                self.stats['features_updated'] += 1
                                logger.info(f"Updated phantom features for {symbol}")
                            
                            self.stats['phantom_trades_processed'] += 1
                        
                        updated_trades.append(trade)
                        
                    except Exception as e:
                        logger.error(f"Error processing phantom trade for {symbol}: {e}")
                        updated_trades.append(trade)  # Keep original on error
                        self.stats['errors'] += 1
                
                updated_phantom_dict[symbol] = updated_trades
            
            # Update Redis if not dry run
            if not self.dry_run:
                logger.info("Updating phantom trades in Redis...")
                self.redis_client.set('phantom:completed', json.dumps(updated_phantom_dict))
                logger.info(f"Updated phantom trades for {len(updated_phantom_dict)} symbols")
            else:
                logger.info(f"DRY RUN: Would update phantom trades for {len(updated_phantom_dict)} symbols")
                
        except Exception as e:
            logger.error(f"Error processing phantom trades: {e}")
            self.stats['errors'] += 1
    
    def trigger_retrain(self):
        """Trigger ML model retrain after feature update"""
        if not self.dry_run:
            logger.info("Triggering ML retrain with updated features...")
            # Clear the feature version to force retrain on next startup
            self.redis_client.delete('iml:feature_calculations_version')
            logger.info("Cleared feature version - ML will retrain on next startup")
        else:
            logger.info("DRY RUN: Would trigger ML retrain")
    
    def print_summary(self):
        """Print summary of changes"""
        logger.info("\n" + "="*50)
        logger.info("FEATURE RECALCULATION SUMMARY")
        logger.info("="*50)
        logger.info(f"Executed trades processed: {self.stats['executed_trades_processed']}")
        logger.info(f"Phantom trades processed: {self.stats['phantom_trades_processed']}")
        logger.info(f"Features updated: {self.stats['features_updated']}")
        logger.info(f"Errors: {self.stats['errors']}")
        logger.info(f"Mode: {'DRY RUN' if self.dry_run else 'LIVE UPDATE'}")
        logger.info("="*50)
    
    def run(self):
        """Run the feature recalculation process"""
        logger.info("Starting ML feature recalculation...")
        logger.info(f"Mode: {'DRY RUN' if self.dry_run else 'LIVE UPDATE'}")
        
        # Always backup first
        self.backup_data()
        
        # Process trades
        self.process_executed_trades()
        self.process_phantom_trades()
        
        # Trigger retrain
        self.trigger_retrain()
        
        # Print summary
        self.print_summary()

def main():
    """Main entry point"""
    # Check command line arguments
    dry_run = True
    if len(sys.argv) > 1 and sys.argv[1] == '--live':
        dry_run = False
        response = input("⚠️  WARNING: This will modify stored ML data. Type 'yes' to continue: ")
        if response.lower() != 'yes':
            logger.info("Aborted.")
            return
    
    # Run recalculator
    recalculator = FeatureRecalculator(dry_run=dry_run)
    recalculator.run()
    
    if dry_run:
        logger.info("\nTo apply changes, run: python recalculate_ml_features.py --live")

if __name__ == "__main__":
    main()