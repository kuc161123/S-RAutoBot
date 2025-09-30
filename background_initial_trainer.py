#!/usr/bin/env python3
"""
Background Initial ML Trainer

Runs alongside the live bot to perform initial ML model training from historical data.
Once complete, the live bot's existing retraining system takes over for incremental learning.

This trainer:
1. Runs in a separate process to avoid interfering with live trading
2. Backtests both Pullback and Enhanced Mean Reversion strategies
3. Trains initial ML models using historical data
4. Reports progress via Telegram (if available)
5. Exits once training is complete
"""

import asyncio
import multiprocessing as mp
import logging
import yaml
import os
import time
import redis
from datetime import datetime
from typing import List, Dict, Optional
from dotenv import load_dotenv

# Import existing components
from backtester import Backtester
from strategy_pullback_ml_learning import get_ml_learning_signals, MinimalSettings as PullbackSettings, reset_symbol_state as reset_pullback_state

logger = logging.getLogger(__name__)

class BackgroundInitialTrainer:
    """
    Background trainer that runs initial ML model training while live bot operates
    """
    
    def __init__(self, telegram_bot=None):
        self.telegram_bot = telegram_bot
        self.process = None
        self.running = False
        self.redis_client = None
        self.progress_key = "background_trainer:progress"
        
        # Initialize Redis for progress tracking
        self._init_redis()
    
    def _init_redis(self):
        """Initialize Redis connection for progress tracking"""
        try:
            redis_url = os.getenv('REDIS_URL')
            if redis_url:
                self.redis_client = redis.from_url(redis_url, decode_responses=True)
                self.redis_client.ping()
                logger.info("Background trainer connected to Redis")
        except Exception as e:
            logger.warning(f"Redis connection failed for background trainer: {e}")
    
    def should_run_initial_training(self) -> bool:
        """Check if initial training is needed (no existing ML models)"""
        if not self.redis_client:
            return True  # Safe default - run training
        
        try:
            # Check if pullback ML model exists
            pullback_model = self.redis_client.get('ml_scorer:model_data')
            
            # Check if enhanced MR model exists  
            mr_model = self.redis_client.get('enhanced_mr:model_data')
            
            # If either model is missing, run training
            if not pullback_model or not mr_model:
                logger.info("Missing ML models detected - initial training needed")
                return True
            
            logger.info("Existing ML models found - skipping initial training")
            return False
            
        except Exception as e:
            logger.error(f"Error checking existing models: {e}")
            return True  # Safe default
    
    async def start_if_needed(self):
        """Start background training if needed"""
        if not self.should_run_initial_training():
            await self._notify_telegram("✅ *ML models already exist* - skipping initial training")
            return False
        
        if self.running:
            logger.warning("Background trainer already running")
            return False
        
        logger.info("Starting background initial ML training...")
        await self._notify_telegram(
            "🚀 *Starting Background ML Training*\n\n"
            "Training initial models from historical data:\n"
            "• Pullback Strategy\n"
            "• Enhanced Mean Reversion Strategy\n\n"
            "This will run in background while live trading continues.\n"
            "Use /training_status to check progress."
        )
        
        # Start training process
        self.process = mp.Process(target=self._run_training_process, daemon=True)
        self.process.start()
        self.running = True
        
        # Start progress monitoring
        asyncio.create_task(self._monitor_progress())
        
        return True
    
    def _run_training_process(self):
        """Main training process (runs in separate process)"""
        try:
            # Setup logging for subprocess
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - [TRAINER] %(name)s - %(levelname)s - %(message)s'
            )
            logger = logging.getLogger(__name__)
            
            # Load environment variables
            load_dotenv()
            
            # Initialize Redis for progress updates
            redis_client = None
            try:
                redis_url = os.getenv('REDIS_URL')
                if redis_url:
                    redis_client = redis.from_url(redis_url, decode_responses=True)
            except:
                pass
            
            def update_progress(stage: str, symbol: str = "", progress: int = 0, total: int = 0):
                """Update training progress"""
                progress_data = {
                    'stage': stage,
                    'symbol': symbol,
                    'progress': progress,
                    'total': total,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'running'
                }
                
                if redis_client:
                    try:
                        redis_client.setex(self.progress_key, 3600, str(progress_data))  # 1 hour TTL
                    except:
                        pass
                
                logger.info(f"[{stage}] {symbol} ({progress}/{total})")
            
            logger.info("🎯 Starting Background ML Training Process")
            
            # Load configuration
            with open("config.yaml", "r") as f:
                cfg = yaml.safe_load(f)
            symbols = cfg["trade"]["symbols"]
            total_symbols = len(symbols)
            
            logger.info(f"📊 Will train on {total_symbols} symbols")
            
            # Phase 1: Pullback Strategy Training
            logger.info("\n🎯 Phase 1: Pullback Strategy Training")
            update_progress("Initializing Pullback Backtester", "", 0, total_symbols)
            
            pullback_backtester = Backtester(
                get_ml_learning_signals, 
                PullbackSettings(), 
                reset_state_func=reset_pullback_state
            )
            
            all_pullback_data = []
            for i, symbol in enumerate(symbols):
                update_progress("Backtesting Pullback", symbol, i+1, total_symbols)
                try:
                    results = pullback_backtester.run(symbol)
                    all_pullback_data.extend(results)
                    logger.info(f"  ✅ {symbol}: {len(results)} signals")
                except Exception as e:
                    logger.error(f"  ❌ {symbol}: {e}")
            
            logger.info(f"📈 Pullback backtesting complete: {len(all_pullback_data)} total signals")
            
            # Train pullback model
            update_progress("Training Pullback Model", "", total_symbols, total_symbols)
            self._train_pullback_model(all_pullback_data)
            
            # Phase 2: Enhanced Mean Reversion Strategy Training
            logger.info("\n🎯 Phase 2: Enhanced Mean Reversion Strategy Training")
            update_progress("Initializing Enhanced MR Backtester", "", 0, total_symbols)
            
            # Import Enhanced MR components
            try:
                from enhanced_mr_strategy import detect_enhanced_mr_signal, EnhancedMRSettings
                from enhanced_mr_strategy import reset_enhanced_mr_state
                
                mr_backtester = Backtester(
                    detect_enhanced_mr_signal,
                    EnhancedMRSettings(),
                    reset_state_func=reset_enhanced_mr_state
                )
                
                all_mr_data = []
                for i, symbol in enumerate(symbols):
                    update_progress("Backtesting Enhanced MR", symbol, i+1, total_symbols)
                    try:
                        results = mr_backtester.run(symbol)
                        all_mr_data.extend(results)
                        logger.info(f"  ✅ {symbol}: {len(results)} signals")
                    except Exception as e:
                        logger.error(f"  ❌ {symbol}: {e}")
                
                logger.info(f"📈 Enhanced MR backtesting complete: {len(all_mr_data)} total signals")
                
                # Train Enhanced MR model
                update_progress("Training Enhanced MR Model", "", total_symbols, total_symbols)
                self._train_enhanced_mr_model(all_mr_data)
                
            except ImportError as e:
                logger.warning(f"Enhanced MR strategy not available: {e}")
                logger.info("Falling back to original Mean Reversion strategy")
                
                # Fallback to original MR strategy
                from strategy_mean_reversion import detect_signal as detect_signal_mean_reversion
                from strategy_mean_reversion import reset_symbol_state as reset_mean_reversion_state
                from strategy_pullback import Settings as ReversionSettings
                
                mr_backtester = Backtester(
                    detect_signal_mean_reversion,
                    ReversionSettings(),
                    reset_state_func=reset_mean_reversion_state
                )
                
                all_mr_data = []
                for i, symbol in enumerate(symbols):
                    update_progress("Backtesting Mean Reversion", symbol, i+1, total_symbols)
                    try:
                        results = mr_backtester.run(symbol)
                        all_mr_data.extend(results)
                        logger.info(f"  ✅ {symbol}: {len(results)} signals")
                    except Exception as e:
                        logger.error(f"  ❌ {symbol}: {e}")
                
                logger.info(f"📈 Mean Reversion backtesting complete: {len(all_mr_data)} total signals")
                
                # Train original MR model
                update_progress("Training MR Model", "", total_symbols, total_symbols)
                self._train_mean_reversion_model(all_mr_data)
            
            # Mark completion
            completion_data = {
                'stage': 'completed',
                'timestamp': datetime.now().isoformat(),
                'status': 'completed',
                'pullback_signals': len(all_pullback_data),
                'mr_signals': len(all_mr_data),
                'total_symbols': total_symbols
            }
            
            if redis_client:
                try:
                    redis_client.setex(self.progress_key, 3600, str(completion_data))
                except:
                    pass
            
            logger.info("🎉 Background ML Training Complete!")
            logger.info(f"   📊 Pullback signals: {len(all_pullback_data)}")
            logger.info(f"   📊 MR signals: {len(all_mr_data)}")
            logger.info("   ✅ Models trained and saved to Redis")
            logger.info("   🔄 Live bot will now handle incremental retraining")
            
        except Exception as e:
            logger.error(f"Background training failed: {e}", exc_info=True)
            
            # Mark error
            error_data = {
                'stage': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'status': 'error'
            }
            
            if redis_client:
                try:
                    redis_client.setex(self.progress_key, 3600, str(error_data))
                except:
                    pass
    
    def _train_pullback_model(self, training_data: List[Dict]):
        """Train pullback ML model"""
        if not training_data:
            logger.warning("No training data for pullback model")
            return
        
        logger.info(f"🧠 Training Pullback ML model on {len(training_data)} signals...")
        
        try:
            from ml_signal_scorer_immediate import get_immediate_scorer
            
            scorer = get_immediate_scorer(enabled=True)
            
            # Load data for training
            scorer.memory_storage = {'trades': training_data, 'phantoms': []}
            scorer.completed_trades = len(training_data)
            
            # Train model
            success = scorer.startup_retrain()
            
            if success and scorer.is_ml_ready:
                logger.info("✅ Pullback ML model trained successfully")
            else:
                logger.error("❌ Pullback ML model training failed")
                
        except Exception as e:
            logger.error(f"Error training pullback model: {e}")
    
    def _train_enhanced_mr_model(self, training_data: List[Dict]):
        """Train Enhanced MR ML model"""
        if not training_data:
            logger.warning("No training data for Enhanced MR model")
            return
        
        logger.info(f"🧠 Training Enhanced MR ML model on {len(training_data)} signals...")
        
        try:
            from enhanced_mr_scorer import get_enhanced_mr_scorer
            
            scorer = get_enhanced_mr_scorer(enabled=True)
            
            # Load data for training
            scorer.memory_storage = {'trades': training_data, 'phantoms': []}
            scorer.completed_trades = len(training_data)
            
            # Train model
            success = scorer.startup_retrain()
            
            if success and scorer.is_ml_ready:
                logger.info("✅ Enhanced MR ML model trained successfully")
            else:
                logger.error("❌ Enhanced MR ML model training failed")
                
        except Exception as e:
            logger.error(f"Error training Enhanced MR model: {e}")
    
    def _train_mean_reversion_model(self, training_data: List[Dict]):
        """Train original MR ML model (fallback)"""
        if not training_data:
            logger.warning("No training data for MR model")
            return
        
        logger.info(f"🧠 Training Mean Reversion ML model on {len(training_data)} signals...")
        
        try:
            from ml_scorer_mean_reversion import get_mean_reversion_scorer
            
            scorer = get_mean_reversion_scorer(enabled=True)
            
            # Load data for training
            scorer.memory_storage = {'trades': training_data}
            scorer.completed_trades = len(training_data)
            
            # Train model
            scorer._retrain_models()
            
            if scorer.is_ml_ready:
                logger.info("✅ Mean Reversion ML model trained successfully")
            else:
                logger.error("❌ Mean Reversion ML model training failed")
                
        except Exception as e:
            logger.error(f"Error training MR model: {e}")
    
    async def _monitor_progress(self):
        """Monitor training progress and send updates"""
        last_stage = ""
        last_symbol = ""
        
        while self.running:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                if not self.redis_client:
                    continue
                
                progress_str = self.redis_client.get(self.progress_key)
                if not progress_str:
                    continue
                
                # Parse progress (simple string parsing since we stored as string)
                progress_data = eval(progress_str)  # Note: In production, use json.loads instead
                
                stage = progress_data.get('stage', '')
                symbol = progress_data.get('symbol', '')
                progress = progress_data.get('progress', 0)
                total = progress_data.get('total', 0)
                status = progress_data.get('status', '')
                
                # Send update if significant progress
                if stage != last_stage or (symbol and symbol != last_symbol):
                    if stage in ['Backtesting Pullback', 'Backtesting Enhanced MR', 'Backtesting Mean Reversion']:
                        if progress % 20 == 0 or progress == total:  # Update every 20 symbols or at completion
                            await self._notify_telegram(
                                f"📊 *Training Progress*\n\n"
                                f"Stage: {stage}\n"
                                f"Progress: {progress}/{total} symbols\n"
                                f"Current: {symbol}"
                            )
                    elif stage in ['Training Pullback Model', 'Training Enhanced MR Model', 'Training MR Model']:
                        await self._notify_telegram(f"🧠 *{stage}*\nAnalyzing signals and building ML models...")
                    
                    last_stage = stage
                    last_symbol = symbol
                
                # Check for completion or error
                if status == 'completed':
                    await self._notify_telegram(
                        "🎉 *Background ML Training Complete!*\n\n"
                        f"✅ Pullback signals: {progress_data.get('pullback_signals', 0)}\n"
                        f"✅ MR signals: {progress_data.get('mr_signals', 0)}\n"
                        f"✅ Total symbols: {progress_data.get('total_symbols', 0)}\n\n"
                        "🔄 Live bot will now handle automatic retraining as new trades accumulate.\n"
                        "Use /ml and /enhanced_mr to check model status."
                    )
                    self.running = False
                    break
                
                elif status == 'error':
                    await self._notify_telegram(
                        f"❌ *Background Training Error*\n\n"
                        f"Error: {progress_data.get('error', 'Unknown error')}\n\n"
                        "Training will retry on next bot restart."
                    )
                    self.running = False
                    break
                    
            except Exception as e:
                logger.error(f"Error monitoring training progress: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _notify_telegram(self, message: str):
        """Send notification via Telegram if available"""
        if self.telegram_bot:
            try:
                await self.telegram_bot.send_message(message)
            except Exception as e:
                logger.error(f"Failed to send Telegram notification: {e}")
    
    def get_status(self) -> Dict:
        """Get current training status"""
        if not self.redis_client:
            return {'status': 'unknown', 'message': 'Redis not available'}
        
        try:
            progress_str = self.redis_client.get(self.progress_key)
            if not progress_str:
                if self.running:
                    return {'status': 'running', 'message': 'Training in progress...'}
                else:
                    return {'status': 'not_started', 'message': 'Training not started'}
            
            progress_data = eval(progress_str)  # Note: In production, use json.loads
            return progress_data
            
        except Exception as e:
            return {'status': 'error', 'message': f'Error getting status: {e}'}
    
    def stop(self):
        """Stop the background trainer"""
        if self.process and self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=10)
            if self.process.is_alive():
                self.process.kill()
        
        self.running = False
        logger.info("Background trainer stopped")

# Global instance
_background_trainer = None

def get_background_trainer(telegram_bot=None) -> BackgroundInitialTrainer:
    """Get or create the global background trainer instance"""
    global _background_trainer
    if _background_trainer is None:
        _background_trainer = BackgroundInitialTrainer(telegram_bot)
    return _background_trainer

if __name__ == "__main__":
    # Direct execution for testing
    load_dotenv()
    trainer = BackgroundInitialTrainer()
    
    # Run synchronously for testing
    import asyncio
    async def test_run():
        await trainer.start_if_needed()
        
        # Wait for completion
        while trainer.running:
            status = trainer.get_status()
            print(f"Status: {status}")
            await asyncio.sleep(10)
    
    asyncio.run(test_run())