"""
Automatic Symbol-Specific ML Trainer
Runs in background to train symbol models when sufficient data is available
"""
import numpy as np
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import psycopg2
from psycopg2.extras import RealDictCursor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import os
import json

from phantom_trade_tracker import get_phantom_tracker
from ml_evolution_system import get_evolution_system

logger = logging.getLogger(__name__)

class SymbolMLTrainer:
    """
    Automatically trains symbol-specific ML models in the background
    - Monitors data availability
    - Trains when sufficient data exists
    - Updates models periodically
    - Handles all data sources (executed, phantom, historical)
    """
    
    # Configuration
    MIN_TRADES_FOR_TRAINING = 50
    RETRAIN_INTERVAL_HOURS = 4
    BACKGROUND_CHECK_INTERVAL = 3600  # Check hourly
    
    def __init__(self):
        self.evolution_system = get_evolution_system()
        self.phantom_tracker = get_phantom_tracker()
        self.is_running = False
        
        # PostgreSQL connection
        self.db_conn = None
        self._init_postgres()
    
    def _init_postgres(self):
        """Initialize PostgreSQL connection"""
        try:
            db_url = os.getenv('DATABASE_URL')
            if db_url:
                self.db_conn = psycopg2.connect(db_url, cursor_factory=RealDictCursor)
                logger.info("Symbol trainer connected to PostgreSQL")
        except Exception as e:
            logger.warning(f"Symbol trainer PostgreSQL connection failed: {e}")
    
    async def start_background_training(self):
        """Start the automatic background training loop"""
        if self.is_running:
            logger.warning("Background training already running")
            return
        
        self.is_running = True
        logger.info("Starting automatic symbol ML training loop")
        
        while self.is_running:
            try:
                # Get all active symbols
                symbols = self._get_active_symbols()
                logger.info(f"Checking {len(symbols)} symbols for training needs")
                
                # Check each symbol
                for symbol in symbols:
                    try:
                        if self._should_train_symbol(symbol):
                            logger.info(f"[{symbol}] Training symbol-specific model")
                            success = await self._train_symbol_model(symbol)
                            if success:
                                logger.info(f"[{symbol}] Successfully trained model")
                            else:
                                logger.warning(f"[{symbol}] Training failed")
                    except Exception as e:
                        logger.error(f"[{symbol}] Training error: {e}")
                
                # Save state after training
                self.evolution_system._save_state()
                
                # Wait before next check
                await asyncio.sleep(self.BACKGROUND_CHECK_INTERVAL)
                
            except Exception as e:
                logger.error(f"Background training loop error: {e}")
                await asyncio.sleep(300)  # Retry in 5 minutes
    
    def stop_background_training(self):
        """Stop the background training loop"""
        self.is_running = False
        logger.info("Stopping background training loop")
    
    def _get_active_symbols(self) -> List[str]:
        """Get list of symbols with recent activity"""
        symbols = set()
        
        # From evolution system stats
        symbols.update(self.evolution_system.symbol_stats.keys())
        
        # From phantom tracker
        symbols.update(self.phantom_tracker.phantom_trades.keys())
        
        # From recent database trades
        if self.db_conn:
            try:
                with self.db_conn.cursor() as cur:
                    cur.execute("""
                        SELECT DISTINCT symbol 
                        FROM trades 
                        WHERE exit_time > NOW() - INTERVAL '7 days'
                    """)
                    for row in cur.fetchall():
                        symbols.add(row['symbol'])
            except Exception as e:
                logger.warning(f"Error getting symbols from DB: {e}")
        
        return list(symbols)
    
    def _should_train_symbol(self, symbol: str) -> bool:
        """Check if symbol needs training"""
        stats = self.evolution_system.symbol_stats.get(symbol)
        
        # Get total data count
        total_data = self._get_symbol_data_count(symbol)
        
        # Need minimum data
        if total_data < self.MIN_TRADES_FOR_TRAINING:
            return False
        
        # Check if never trained
        if symbol not in self.evolution_system.symbol_models:
            logger.info(f"[{symbol}] Never trained, has {total_data} data points")
            return True
        
        # Check if retrain interval passed
        if stats and stats.last_train_time:
            hours_since = (datetime.now() - stats.last_train_time).total_seconds() / 3600
            if hours_since > self.RETRAIN_INTERVAL_HOURS:
                logger.info(f"[{symbol}] Due for retraining ({hours_since:.1f} hours)")
                return True
        
        return False
    
    def _get_symbol_data_count(self, symbol: str) -> int:
        """Get total available data for symbol"""
        count = 0
        
        # Executed trades from DB
        if self.db_conn:
            try:
                with self.db_conn.cursor() as cur:
                    cur.execute("""
                        SELECT COUNT(*) as count 
                        FROM trades 
                        WHERE symbol = %s AND exit_time IS NOT NULL
                    """, (symbol,))
                    result = cur.fetchone()
                    if result:
                        count += result['count']
            except Exception as e:
                logger.warning(f"Error counting DB trades: {e}")
        
        # Phantom trades
        phantom_data = self.phantom_tracker.get_phantom_stats(symbol)
        count += phantom_data.get('total', 0)
        
        return count
    
    async def _train_symbol_model(self, symbol: str) -> bool:
        """Train a symbol-specific model"""
        try:
            # Gather all training data
            all_data = self._gather_training_data(symbol)
            
            if len(all_data) < self.MIN_TRADES_FOR_TRAINING:
                logger.warning(f"[{symbol}] Not enough data: {len(all_data)}")
                return False
            
            # Prepare features and labels
            X = []
            y = []
            
            for data in all_data:
                features = data['features']
                outcome = data['outcome']
                
                # Use evolution system's feature preparation
                feature_vector = self.evolution_system._prepare_features(features)
                X.append(feature_vector)
                y.append(outcome)
            
            X = np.array(X)
            y = np.array(y)
            
            # Split for validation (80/20)
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Train ensemble models
            models = {}
            
            # Random Forest
            rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=8,  # Deeper for symbol-specific
                min_samples_split=5,
                random_state=42
            )
            rf.fit(X_train_scaled, y_train)
            val_score_rf = rf.score(X_val_scaled, y_val)
            models['rf'] = rf
            
            # Gradient Boosting
            gb = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.05,
                random_state=42
            )
            gb.fit(X_train_scaled, y_train)
            val_score_gb = gb.score(X_val_scaled, y_val)
            models['gb'] = gb
            
            # Neural Network if enough data
            if len(all_data) >= 100:
                nn = MLPClassifier(
                    hidden_layer_sizes=(20, 10),  # Larger for symbol-specific
                    activation='relu',
                    learning_rate_init=0.001,
                    max_iter=1000,
                    random_state=42
                )
                nn.fit(X_train_scaled, y_train)
                val_score_nn = nn.score(X_val_scaled, y_val)
                models['nn'] = nn
            
            # Calculate performance metrics
            train_wr = np.mean(y_train) * 100
            val_wr = np.mean(y_val) * 100
            executed_data = [d for d in all_data if d.get('was_executed', False)]
            phantom_data = [d for d in all_data if not d.get('was_executed', False)]
            
            logger.info(f"[{symbol}] Model trained on {len(all_data)} samples")
            logger.info(f"[{symbol}] Data composition: {len(executed_data)} executed, {len(phantom_data)} phantom")
            logger.info(f"[{symbol}] Train WR: {train_wr:.1f}%, Val WR: {val_wr:.1f}%")
            logger.info(f"[{symbol}] Model accuracy - RF: {val_score_rf:.2f}, GB: {val_score_gb:.2f}")
            
            # Update evolution system
            self.evolution_system.symbol_models[symbol] = models
            self.evolution_system.symbol_scalers[symbol] = scaler
            
            # Update stats
            if symbol not in self.evolution_system.symbol_stats:
                from ml_evolution_system import SymbolMLStats
                self.evolution_system.symbol_stats[symbol] = SymbolMLStats(symbol=symbol)
            
            stats = self.evolution_system.symbol_stats[symbol]
            stats.last_train_time = datetime.now()
            stats.model_version += 1
            stats.executed_trades = len(executed_data)
            stats.phantom_trades = len(phantom_data)
            
            return True
            
        except Exception as e:
            logger.error(f"[{symbol}] Training failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def _gather_training_data(self, symbol: str) -> List[Dict]:
        """Gather all available data for training"""
        all_data = []
        
        # Initialize enhanced feature engine
        enhanced_features_available = False
        feature_engine = None
        try:
            from enhanced_features import get_feature_engine
            feature_engine = get_feature_engine()
            enhanced_features_available = True
            logger.info(f"[{symbol}] Enhanced features available for training")
        except Exception as e:
            logger.debug(f"Enhanced features not available: {e}")
        
        # 1. Get executed trades from database
        if self.db_conn:
            try:
                with self.db_conn.cursor() as cur:
                    cur.execute("""
                        SELECT 
                            symbol, side, entry_price, exit_price,
                            pnl_percent, features, exit_time,
                            CASE WHEN pnl_percent > 0 THEN 1 ELSE 0 END as outcome
                        FROM trades
                        WHERE symbol = %s 
                        AND exit_time IS NOT NULL
                        AND features IS NOT NULL
                        ORDER BY exit_time DESC
                        LIMIT 1000
                    """, (symbol,))
                    
                    for row in cur.fetchall():
                        if row['features']:
                            # Parse features if stored as JSON string
                            features = row['features']
                            if isinstance(features, str):
                                features = json.loads(features)
                            
                            # Enhance features if available
                            if enhanced_features_available and feature_engine:
                                try:
                                    # Get historical candles around trade time
                                    trade_time = row['exit_time']
                                    df = self._get_candles_for_time(symbol, trade_time)
                                    if df is not None and len(df) > 0:
                                        btc_price = self._get_btc_price_at_time(trade_time)
                                        features = feature_engine.enhance_features(
                                            symbol=symbol,
                                            df=df,
                                            current_features=features,
                                            btc_price=btc_price
                                        )
                                except Exception as e:
                                    logger.debug(f"Could not enhance historical features: {e}")
                            
                            all_data.append({
                                'features': features,
                                'outcome': row['outcome'],
                                'pnl_percent': float(row['pnl_percent']),
                                'was_executed': True,
                                'timestamp': row['exit_time']
                            })
            except Exception as e:
                logger.error(f"Error getting executed trades for {symbol}: {e}")
        
        # 2. Get phantom trades
        phantom_learning_data = self.phantom_tracker.get_learning_data()
        for phantom in phantom_learning_data:
            if phantom['symbol'] == symbol:
                # Phantom features may already be enhanced if recorded recently
                all_data.append({
                    'features': phantom['features'],
                    'outcome': phantom['outcome'],
                    'pnl_percent': phantom['pnl_percent'],
                    'was_executed': phantom['was_executed'],
                    'ml_score': phantom.get('ml_score', 0)
                })
        
        # 3. Enhance recent data with current market context if needed
        # This ensures we're training on the most relevant patterns
        
        logger.info(f"[{symbol}] Gathered {len(all_data)} total training samples")
        
        # Sort by timestamp if available (most recent first for time-series aware training)
        all_data.sort(key=lambda x: x.get('timestamp', datetime.min), reverse=True)
        
        return all_data
    
    def _get_candles_for_time(self, symbol: str, trade_time: datetime, candles_before: int = 200):
        """Get candle data around a specific time for feature enhancement"""
        if not self.db_conn:
            return None
        
        try:
            with self.db_conn.cursor() as cur:
                # Get candles before the trade time
                cur.execute("""
                    SELECT timestamp, open, high, low, close, volume
                    FROM candles
                    WHERE symbol = %s
                    AND timestamp <= %s
                    ORDER BY timestamp DESC
                    LIMIT %s
                """, (symbol, trade_time, candles_before))
                
                rows = cur.fetchall()
                if rows:
                    # Convert to DataFrame
                    import pandas as pd
                    df = pd.DataFrame(rows)
                    df.set_index('timestamp', inplace=True)
                    df = df.sort_index()
                    return df
        except Exception as e:
            logger.debug(f"Error getting historical candles: {e}")
        
        return None
    
    def _get_btc_price_at_time(self, trade_time: datetime):
        """Get BTC price at a specific time"""
        if not self.db_conn:
            return None
        
        try:
            with self.db_conn.cursor() as cur:
                cur.execute("""
                    SELECT close
                    FROM candles
                    WHERE symbol = 'BTCUSDT'
                    AND timestamp <= %s
                    ORDER BY timestamp DESC
                    LIMIT 1
                """, (trade_time,))
                
                result = cur.fetchone()
                if result:
                    return float(result['close'])
        except Exception as e:
            logger.debug(f"Error getting BTC price: {e}")
        
        return None

# Create trainer instance
_symbol_trainer = None

def get_symbol_trainer() -> SymbolMLTrainer:
    """Get or create the global symbol trainer"""
    global _symbol_trainer
    if _symbol_trainer is None:
        _symbol_trainer = SymbolMLTrainer()
    return _symbol_trainer