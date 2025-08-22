"""
Fixed Integrated Trading Engine with all critical improvements
"""
import asyncio
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
import structlog
import json
import redis.asyncio as redis
import traceback

from ..api.enhanced_bybit_client import EnhancedBybitClient
from ..strategy.advanced_supply_demand import AdvancedSupplyDemandStrategy
from ..strategy.ml_predictor import ml_predictor
from .position_manager import EnhancedPositionManager
from .multi_timeframe_scanner import MultiTimeframeScanner
from .order_manager import OrderManager
from ..telegram.bot import TradingBot
from ..config import settings
from ..db.database import DatabaseManager
from ..utils.bot_fixes import (
    position_safety,
    ml_validator,
    db_pool,
    health_monitor
)

logger = structlog.get_logger(__name__)

class FixedIntegratedEngine:
    """
    Fixed trading engine with:
    - Proper rate limiting for 300 symbols
    - One position per symbol enforcement
    - ML validation before predictions
    - Database connection pooling
    - Error recovery and health monitoring
    - Batch operations for efficiency
    """
    
    def __init__(self, bybit_client: EnhancedBybitClient, telegram_bot: TradingBot):
        self.client = bybit_client
        self.telegram_bot = telegram_bot
        self.db_manager = DatabaseManager()
        
        # Components
        self.strategy = AdvancedSupplyDemandStrategy()
        self.position_manager = EnhancedPositionManager(bybit_client)
        self.scanner = MultiTimeframeScanner(bybit_client, self.strategy)
        self.order_manager = OrderManager(bybit_client)
        
        # Redis
        self.redis_client = None
        
        # State
        self.is_running = False
        self.positions_per_symbol = {}  # Enforces one position per symbol
        self.symbol_locks = {}  # Prevent concurrent operations per symbol
        
        # Configuration - reduced for testing and stability
        self.monitored_symbols = settings.default_symbols[:20]  # Start with 20 symbols
        self.batch_size = 5  # Smaller batches to avoid rate limits
        
        # Performance tracking
        self.stats = {
            'trades_today': 0,
            'pnl_today': 0,
            'errors_today': 0,
            'last_health_check': None
        }
        
    async def initialize(self):
        """Initialize with proper error handling"""
        
        try:
            logger.info("Initializing Fixed Integrated Engine...")
            
            # Initialize Redis with retry
            await self._init_redis()
            
            # Initialize scanner
            await self.scanner.initialize()
            
            # Initialize leverage and margin in batches
            await self._init_symbol_settings_batch()
            
            # Load ML models if available
            await self._init_ml_models()
            
            # Sync existing positions
            await self._sync_positions()
            
            # Start health monitoring
            asyncio.create_task(self._health_monitor())
            
            logger.info("Fixed Integrated Engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            logger.error(traceback.format_exc())
            raise
            
    async def _init_redis(self):
        """Initialize Redis with retry logic"""
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.redis_client = redis.from_url(
                    settings.redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5
                )
                await self.redis_client.ping()
                logger.info("Redis connected successfully")
                return
                
            except Exception as e:
                logger.warning(f"Redis connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    logger.error("Redis connection failed - continuing without caching")
                    self.redis_client = None
                    
    async def _init_symbol_settings_batch(self):
        """Initialize leverage and margin for all symbols in batches"""
        
        logger.info(f"Initializing settings for {len(self.monitored_symbols)} symbols...")
        
        leverage = settings.default_leverage
        margin_mode = settings.default_margin_mode.value
        
        # Process in batches to avoid overwhelming the API
        successful = 0
        failed = 0
        
        for i in range(0, len(self.monitored_symbols), self.batch_size):
            batch = self.monitored_symbols[i:i + self.batch_size]
            
            # Set leverage for batch
            tasks = []
            for symbol in batch:
                task = self.client.set_leverage(symbol, leverage)
                tasks.append(task)
                
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for symbol, result in zip(batch, results):
                if isinstance(result, Exception):
                    logger.warning(f"Failed to set leverage for {symbol}: {result}")
                    failed += 1
                elif result:
                    successful += 1
                else:
                    failed += 1
                    
            # Small delay between batches
            await asyncio.sleep(1)
            
        logger.info(f"Leverage set: {successful} successful, {failed} failed")
        
        # Set margin mode in batches
        successful = 0
        failed = 0
        
        for i in range(0, len(self.monitored_symbols), self.batch_size):
            batch = self.monitored_symbols[i:i + self.batch_size]
            
            tasks = []
            for symbol in batch:
                task = self.client.set_margin_mode(symbol, margin_mode)
                tasks.append(task)
                
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for symbol, result in zip(batch, results):
                if isinstance(result, Exception):
                    logger.warning(f"Failed to set margin mode for {symbol}: {result}")
                    failed += 1
                elif result:
                    successful += 1
                else:
                    failed += 1
                    
            await asyncio.sleep(1)
            
        logger.info(f"Margin mode set: {successful} successful, {failed} failed")
        
    async def _init_ml_models(self):
        """Initialize ML models with validation"""
        
        try:
            # Try to load existing models
            ml_predictor.load_models("/tmp/ml_models")
            logger.info("ML models loaded successfully")
            
            # Validate training data if any
            if ml_predictor.training_data:
                if ml_validator.validate_training_data(ml_predictor.training_data):
                    logger.info("ML training data validated")
                else:
                    logger.warning("ML training data validation failed - clearing")
                    ml_predictor.training_data = []
                    
        except Exception as e:
            logger.info(f"No pre-trained ML models found: {e}")
            logger.info("Will train from scratch with new data")
            
    async def _sync_positions(self):
        """Sync existing positions from exchange"""
        
        try:
            positions = await self.client.get_positions()
            
            for position in positions:
                if float(position.get('size', 0)) > 0:
                    symbol = position['symbol']
                    side = position['side']
                    
                    # Determine position type
                    position_type = 'long' if side == 'Buy' else 'short'
                    
                    # Track position
                    self.positions_per_symbol[symbol] = {
                        'type': position_type,
                        'data': position,
                        'entry_price': float(position.get('avgPrice', 0)),
                        'size': float(position.get('size', 0)),
                        'unrealized_pnl': float(position.get('unrealisedPnl', 0)),
                        'sync_time': datetime.now()
                    }
                    
                    # Register with position safety
                    position_safety.register_position(symbol, {
                        'side': side,
                        'size': float(position.get('size', 0)),
                        'entry_price': float(position.get('avgPrice', 0))
                    })
                    
                    # Update scanner
                    self.scanner.update_position_status(symbol, position_type)
                    
            logger.info(f"Synced {len(self.positions_per_symbol)} existing positions")
            
        except Exception as e:
            logger.error(f"Error syncing positions: {e}")
            health_monitor.metrics['position_sync_errors'] += 1
            
    async def start(self):
        """Start the trading engine"""
        
        if self.is_running:
            logger.warning("Engine already running")
            return
            
        self.is_running = True
        logger.info("Starting Fixed Integrated Engine...")
        
        # Start scanner
        await self.scanner.start_scanning()
        
        # Start main tasks with error recovery
        asyncio.create_task(self._run_with_recovery(self._main_loop()))
        asyncio.create_task(self._run_with_recovery(self._signal_processor()))
        asyncio.create_task(self._run_with_recovery(self._position_monitor()))
        asyncio.create_task(self._run_with_recovery(self._ml_trainer()))
        
        logger.info("Engine started successfully")
        
    async def stop(self):
        """Stop the trading engine"""
        
        if not self.is_running:
            return
            
        self.is_running = False
        
        # Stop scanner
        await self.scanner.stop_scanning()
        
        # Close positions if configured
        if settings.get('close_positions_on_stop', False):
            await self._close_all_positions()
            
        logger.info("Engine stopped")
        
    async def _run_with_recovery(self, coro):
        """Run coroutine with automatic recovery"""
        
        max_failures = 5
        failure_count = 0
        
        while self.is_running:
            try:
                await coro
                failure_count = 0  # Reset on success
                
            except Exception as e:
                failure_count += 1
                logger.error(f"Task failed ({failure_count}/{max_failures}): {e}")
                logger.error(traceback.format_exc())
                
                if failure_count >= max_failures:
                    logger.critical(f"Task failed {max_failures} times, stopping")
                    break
                    
                # Exponential backoff
                await asyncio.sleep(2 ** failure_count)
                
    async def _main_loop(self):
        """Main trading loop with error handling"""
        
        while self.is_running:
            try:
                # Update market data for active symbols
                active_symbols = list(self.positions_per_symbol.keys())
                if active_symbols:
                    await self._update_market_data(active_symbols[:10])  # Limit to 10 at a time
                    
                # Train ML if enough data
                if len(ml_predictor.training_data) >= 100:
                    if ml_validator.validate_training_data(ml_predictor.training_data):
                        ml_predictor.train_models()
                        logger.info("ML models retrained")
                        
                # Update stats
                await self._update_stats()
                
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                self.stats['errors_today'] += 1
                await asyncio.sleep(10)
                
    async def _signal_processor(self):
        """Process signals with position safety"""
        
        while self.is_running:
            try:
                if not self.redis_client:
                    await asyncio.sleep(5)
                    continue
                    
                # Get signal from queue
                signal_data = await self.redis_client.rpop("signal_queue")
                
                if signal_data:
                    signal_info = json.loads(signal_data)
                    symbol = signal_info['symbol']
                    
                    # Get lock for symbol
                    if symbol not in self.symbol_locks:
                        self.symbol_locks[symbol] = asyncio.Lock()
                        
                    async with self.symbol_locks[symbol]:
                        # Check if we can take position
                        if symbol in self.positions_per_symbol:
                            logger.info(f"Position already exists for {symbol}, skipping")
                            continue
                            
                        # Get full signal
                        signal = await self.redis_client.get(f"signal:{symbol}")
                        if signal:
                            signal = json.loads(signal)
                            
                            # Validate ML prediction if present
                            if 'ml_confidence' in signal:
                                if not ml_validator.validate_prediction(signal):
                                    logger.warning(f"Invalid ML prediction for {symbol}, skipping")
                                    continue
                                    
                            await self._execute_signal_safe(symbol, signal)
                            
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Signal processor error: {e}")
                await asyncio.sleep(5)
                
    async def _execute_signal_safe(self, symbol: str, signal: Dict):
        """Execute signal with all safety checks"""
        
        try:
            # Double-check position
            if symbol in self.positions_per_symbol:
                return
            
            # Calculate risk-reward ratio (must be 1:2 or better)
            risk_reward_ratio = self._calculate_risk_reward(signal)
            
            if risk_reward_ratio < 2.0:
                logger.info(f"❌ Signal rejected for {symbol}: Risk-reward ratio 1:{risk_reward_ratio:.1f} < 1:2 (minimum)")
                return
            
            logger.info(f"✅ Signal accepted for {symbol}: Risk-reward ratio 1:{risk_reward_ratio:.1f}")
            
            # Add risk-reward to signal for tracking
            signal['risk_reward_ratio'] = risk_reward_ratio
                
            # Check with position safety manager
            side = 'Buy' if signal['type'] == 'BUY' else 'Sell'
            if not await position_safety.can_open_position(symbol, side):
                logger.warning(f"Position safety check failed for {symbol}")
                return
                
            # Get account balance
            account_info = await self.client.get_account_info()
            balance = float(account_info.get('totalAvailableBalance', 0))
            
            if balance <= 0:
                logger.error("Insufficient balance")
                return
                
            # Get ML parameters
            market_data = {
                'volatility': signal.get('volatility', 1.0),
                'market_structure': signal.get('market_structure', 'ranging'),
                'order_flow': signal.get('order_flow', 'neutral')
            }
            
            parameters = self.position_manager.calculate_ml_optimized_parameters(
                symbol=symbol,
                signal=signal,
                market_data=market_data,
                account_balance=balance
            )
            
            # Validate position size
            instrument = self.client.get_instrument(symbol)
            if instrument:
                min_qty = float(instrument['min_qty'])
                max_qty = float(instrument['max_qty'])
                qty_step = float(instrument['qty_step'])
                
                # Round to valid quantity
                final_size = parameters.final_size
                final_size = max(min_qty, min(max_qty, final_size))
                final_size = round(final_size / qty_step) * qty_step
                
                if final_size < min_qty:
                    logger.warning(f"Position size too small for {symbol}")
                    return
                    
            else:
                logger.error(f"No instrument data for {symbol}")
                return
                
            # Set leverage
            await self.client.set_leverage(symbol, parameters.final_leverage)
            
            # Set margin mode
            await self.client.set_margin_mode(symbol, parameters.margin_mode)
            
            # Place order
            order_data = {
                'category': 'linear',
                'symbol': symbol,
                'side': side,
                'orderType': 'Market',
                'qty': str(final_size),
                'timeInForce': 'IOC',
                'reduceOnly': False,
                'closeOnTrigger': False
            }
            
            # Add stop loss
            if 'stop_loss' in signal:
                order_data['stopLoss'] = str(signal['stop_loss'])
                
            # Add take profit
            if 'take_profit_1' in signal:
                order_data['takeProfit'] = str(signal['take_profit_1'])
                
            order_id = await self.client.place_order(**order_data)
            
            if order_id:
                # Track position
                position_type = 'long' if signal['type'] == 'BUY' else 'short'
                self.positions_per_symbol[symbol] = {
                    'type': position_type,
                    'order_id': order_id,
                    'entry_price': signal.get('entry_price'),
                    'stop_loss': signal.get('stop_loss'),
                    'take_profit_1': signal.get('take_profit_1'),
                    'size': final_size,
                    'leverage': parameters.final_leverage,
                    'entry_time': datetime.now()
                }
                
                # Register with position safety
                position_safety.register_position(symbol, {
                    'side': side,
                    'size': final_size,
                    'entry_price': signal.get('entry_price', 0)
                })
                
                # Update scanner
                self.scanner.update_position_status(symbol, position_type)
                
                # Track for ML learning first
                await self._track_trade_for_ml(symbol, signal)
                
                # Log to database with risk-reward
                await db_pool.execute_with_retry(
                    self._log_trade_to_db,
                    symbol, signal, parameters, order_id
                )
                
                # Send notification
                await self._send_notification(symbol, signal, parameters)
                
                # Update stats
                self.stats['trades_today'] += 1
                
                logger.info(f"✅ Position opened for {symbol}: {position_type}, size={final_size}, R:R=1:{signal.get('risk_reward_ratio', 0):.1f}")
                
                # Monitor trade outcome for ML learning
                asyncio.create_task(self._monitor_trade_outcome(symbol, signal, order_id))
                
        except Exception as e:
            logger.error(f"Error executing signal for {symbol}: {e}")
            logger.error(traceback.format_exc())
            
            # Clean up on error
            if symbol in self.positions_per_symbol:
                del self.positions_per_symbol[symbol]
            position_safety.remove_position(symbol)
            self.scanner.update_position_status(symbol, 'closed')
            
    async def _position_monitor(self):
        """Monitor positions with sync validation"""
        
        sync_interval = 300  # Sync every 5 minutes (not 30 seconds)
        last_sync = datetime.now()
        
        while self.is_running:
            try:
                # Update position data periodically (don't re-register)
                if (datetime.now() - last_sync).total_seconds() > sync_interval:
                    await self._update_position_data()
                    last_sync = datetime.now()
                    
                # Get current positions
                positions = await self.client.get_positions()
                
                # Check each position
                for position in positions:
                    symbol = position.get('symbol')
                    size = float(position.get('size', 0))
                    
                    if size > 0:
                        # Update tracking
                        if symbol in self.positions_per_symbol:
                            self.positions_per_symbol[symbol]['data'] = position
                            self.positions_per_symbol[symbol]['unrealized_pnl'] = float(
                                position.get('unrealisedPnl', 0)
                            )
                    else:
                        # Position closed
                        if symbol in self.positions_per_symbol:
                            await self._handle_position_closed(symbol, position)
                            
                # Check for positions that should be closed
                for symbol in list(self.positions_per_symbol.keys()):
                    if symbol not in [p['symbol'] for p in positions if float(p.get('size', 0)) > 0]:
                        await self._handle_position_closed(symbol, None)
                        
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Position monitor error: {e}")
                health_monitor.metrics['position_sync_errors'] += 1
                await asyncio.sleep(10)
                
    async def _handle_position_closed(self, symbol: str, position_data: Optional[Dict]):
        """Handle closed position with ML feedback"""
        
        if symbol not in self.positions_per_symbol:
            return
            
        try:
            position_info = self.positions_per_symbol[symbol]
            
            # Calculate PnL
            pnl = 0
            pnl_percent = 0
            
            if position_data:
                pnl = float(position_data.get('realisedPnl', 0))
                if position_info.get('entry_price') and position_info.get('size'):
                    entry_value = position_info['entry_price'] * position_info['size']
                    if entry_value > 0:
                        pnl_percent = (pnl / entry_value) * 100
                        
            # Update ML with outcome
            if pnl_percent != 0:
                await self.position_manager.record_position_outcome(
                    symbol=symbol,
                    outcome={
                        'profit_percent': pnl_percent,
                        'risk_percent': 2.0,  # Default risk
                        'zone': position_info.get('zone')
                    }
                )
                
            # Update stats
            self.stats['pnl_today'] += pnl
            
            # Clean up
            del self.positions_per_symbol[symbol]
            position_safety.remove_position(symbol)
            self.scanner.update_position_status(symbol, 'closed')
            
            logger.info(f"Position closed for {symbol}: PnL={pnl:.2f} ({pnl_percent:.2f}%)")
            
        except Exception as e:
            logger.error(f"Error handling closed position for {symbol}: {e}")
            
    async def _ml_trainer(self):
        """Periodic ML model training"""
        
        while self.is_running:
            try:
                # Train every hour if we have enough data
                await asyncio.sleep(3600)
                
                if len(ml_predictor.training_data) >= 100:
                    if ml_validator.validate_training_data(ml_predictor.training_data):
                        ml_predictor.train_models()
                        
                        # Save models
                        ml_predictor.save_models("/tmp/ml_models")
                        logger.info("ML models retrained and saved")
                        
            except Exception as e:
                logger.error(f"ML trainer error: {e}")
                
    async def _health_monitor(self):
        """Monitor system health"""
        
        while self.is_running:
            try:
                # Check health
                health_status = health_monitor.check_system_health()
                
                if not health_status['healthy']:
                    logger.warning(f"System unhealthy: {health_status}")
                    
                    # Send alert
                    message = f"⚠️ System Health Alert\n"
                    for error in health_status['errors']:
                        message += f"❌ {error}\n"
                    for warning in health_status['warnings']:
                        message += f"⚡ {warning}\n"
                        
                    for chat_id in settings.telegram_allowed_chat_ids:
                        await self.telegram_bot.send_notification(chat_id, message)
                        
                self.stats['last_health_check'] = datetime.now()
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(300)
                
    async def _update_market_data(self, symbols: List[str]):
        """Update market data for symbols"""
        
        # This would fetch latest prices, volumes, etc.
        # Simplified for now
        pass
        
    async def _update_stats(self):
        """Update daily statistics"""
        
        # Save to database
        await self.db_manager.update_daily_stats(self.stats)
        
    async def _log_trade_to_db(self, symbol: str, signal: Dict, parameters, order_id: str):
        """Log trade to database"""
        
        trade_data = {
            'chat_id': settings.telegram_allowed_chat_ids[0] if settings.telegram_allowed_chat_ids else 0,
            'symbol': symbol,
            'order_id': order_id,
            'side': 'Buy' if signal['type'] == 'BUY' else 'Sell',
            'entry_price': signal.get('entry_price'),
            'stop_loss': signal.get('stop_loss'),
            'take_profit_1': signal.get('take_profit_1'),
            'take_profit_2': signal.get('take_profit_2'),
            'position_size': parameters.final_size,
            'leverage': parameters.final_leverage
        }
        
        self.db_manager.log_trade(trade_data)
        
    async def _update_position_data(self):
        """Update position data without re-registering (for periodic sync)"""
        try:
            positions = await self.client.get_positions()
            updated_count = 0
            
            for position in positions:
                if float(position.get('size', 0)) > 0:
                    symbol = position['symbol']
                    
                    # Only update if already tracking
                    if symbol in self.positions_per_symbol:
                        self.positions_per_symbol[symbol]['data'] = position
                        self.positions_per_symbol[symbol]['unrealized_pnl'] = float(position.get('unrealisedPnl', 0))
                        self.positions_per_symbol[symbol]['sync_time'] = datetime.now()
                        updated_count += 1
            
            if updated_count > 0:
                logger.debug(f"Updated data for {updated_count} positions")
                
        except Exception as e:
            logger.error(f"Error updating position data: {e}")
    
    async def _send_notification(self, symbol: str, signal: Dict, parameters):
        """Send trade notification"""
        
        message = f"""
🎯 New {signal['type']} Position

Symbol: {symbol}
Entry: {signal.get('entry_price', 'Market')}
Stop Loss: {signal.get('stop_loss', 'None')}
Take Profit: {signal.get('take_profit_1', 'None')}

Size: {parameters.final_size:.4f}
Leverage: {parameters.final_leverage}x
Confidence: {parameters.confidence_score:.1%}
"""
        
        for chat_id in settings.telegram_allowed_chat_ids:
            await self.telegram_bot.send_notification(chat_id, message)
            
    async def _close_all_positions(self):
        """Close all open positions"""
        
        for symbol in list(self.positions_per_symbol.keys()):
            try:
                # Close position
                position = self.positions_per_symbol[symbol]
                side = 'Sell' if position['type'] == 'long' else 'Buy'
                
                await self.client.place_order(
                    category='linear',
                    symbol=symbol,
                    side=side,
                    orderType='Market',
                    qty=str(position['size']),
                    reduceOnly=True
                )
                
                logger.info(f"Closed position for {symbol}")
                
            except Exception as e:
                logger.error(f"Error closing position for {symbol}: {e}")
                
    def get_status(self) -> Dict:
        """Get engine status"""
        
        return {
            'is_running': self.is_running,
            'monitored_symbols': len(self.monitored_symbols),
            'active_positions': len(self.positions_per_symbol),
            'positions': {
                symbol: {
                    'type': info['type'],
                    'pnl': info.get('unrealized_pnl', 0),
                    'leverage': info.get('leverage', 1)
                }
                for symbol, info in self.positions_per_symbol.items()
            },
            'stats': self.stats,
            'ml_trained': ml_predictor.model_trained,
            'ml_samples': len(ml_predictor.training_data),
            'health': health_monitor.check_system_health()
        }
    
    def _calculate_risk_reward(self, signal: Dict) -> float:
        """Calculate risk-reward ratio from signal"""
        try:
            entry_price = float(signal.get('entry_price', 0))
            stop_loss = float(signal.get('stop_loss', 0))
            take_profit = float(signal.get('take_profit', signal.get('take_profit_1', 0)))
            
            if entry_price == 0 or stop_loss == 0 or take_profit == 0:
                return 0
            
            # Calculate risk and reward
            if signal['type'] == 'BUY':
                risk = entry_price - stop_loss
                reward = take_profit - entry_price
            else:  # SELL
                risk = stop_loss - entry_price
                reward = entry_price - take_profit
            
            if risk <= 0:
                return 0
            
            return reward / risk
            
        except Exception as e:
            logger.error(f"Error calculating risk-reward: {e}")
            return 0
    
    async def _track_trade_for_ml(self, symbol: str, signal: Dict):
        """Track trade data for ML model learning and continuous improvement"""
        try:
            # Prepare comprehensive trade data for ML tracking
            trade_data = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'signal_type': signal['type'],
                'entry_price': signal.get('entry_price'),
                'stop_loss': signal.get('stop_loss'),
                'take_profit': signal.get('take_profit', signal.get('take_profit_1')),
                'risk_reward_ratio': signal.get('risk_reward_ratio', 0),
                'zone_score': signal.get('score', 0),
                'confluence_score': signal.get('confluence_score', 0),
                'market_structure': signal.get('market_structure'),
                'order_flow': signal.get('order_flow'),
                'volume_analysis': signal.get('volume_analysis'),
                'timeframes': signal.get('confirming_timeframes', []),
                'ml_prediction': signal.get('ml_prediction'),
                'ml_confidence': signal.get('ml_confidence', 0),
                'zone_type': signal.get('zone_type'),
                'departure_strength': signal.get('departure_strength', 0),
                'base_candles': signal.get('base_candles', 0)
            }
            
            # Add to ML training data for continuous learning
            ml_predictor.training_data.append(trade_data)
            
            # Store in Redis for persistence and later analysis
            if self.redis_client:
                key = f"ml_trade_data:{symbol}:{datetime.now().timestamp()}"
                await self.redis_client.setex(
                    key,
                    86400 * 30,  # Keep for 30 days
                    json.dumps(trade_data)
                )
                
                # Also add to learning queue
                await self.redis_client.lpush(
                    "ml_learning_queue",
                    json.dumps({'symbol': symbol, 'trade_data': trade_data})
                )
            
            logger.info(f"✅ Trade tracked for ML learning: {symbol} R:R=1:{trade_data['risk_reward_ratio']:.1f}")
            
        except Exception as e:
            logger.error(f"Error tracking trade for ML: {e}")
    
    async def _monitor_trade_outcome(self, symbol: str, signal: Dict, order_id: str):
        """Monitor trade outcome for ML model improvement"""
        try:
            # Initial wait for trade to develop
            await asyncio.sleep(60)  # Wait 1 minute
            
            max_monitoring_time = 3600 * 4  # Monitor for up to 4 hours
            check_interval = 60  # Check every minute
            monitoring_start = datetime.now()
            
            while (datetime.now() - monitoring_start).total_seconds() < max_monitoring_time:
                try:
                    # Check position status
                    positions = await self.client.get_positions(symbol)
                    
                    position_found = False
                    for position in positions:
                        if float(position.get('size', 0)) > 0:
                            position_found = True
                            
                            # Calculate current performance
                            pnl = float(position.get('unrealisedPnl', 0))
                            pnl_percent = float(position.get('unrealisedPnlPcnt', 0)) * 100
                            current_price = float(position.get('markPrice', 0))
                            entry_price = float(signal.get('entry_price', 0))
                            
                            # Check if target or stop hit
                            take_profit = float(signal.get('take_profit', signal.get('take_profit_1', 0)))
                            stop_loss = float(signal.get('stop_loss', 0))
                            
                            trade_closed = False
                            outcome = 'ongoing'
                            
                            if signal['type'] == 'BUY':
                                if current_price >= take_profit:
                                    outcome = 'target_hit'
                                    trade_closed = True
                                elif current_price <= stop_loss:
                                    outcome = 'stop_hit'
                                    trade_closed = True
                            else:  # SELL
                                if current_price <= take_profit:
                                    outcome = 'target_hit'
                                    trade_closed = True
                                elif current_price >= stop_loss:
                                    outcome = 'stop_hit'
                                    trade_closed = True
                            
                            # Store outcome data for ML learning
                            outcome_data = {
                                'symbol': symbol,
                                'order_id': order_id,
                                'timestamp': datetime.now().isoformat(),
                                'outcome': outcome,
                                'pnl': pnl,
                                'pnl_percent': pnl_percent,
                                'current_price': current_price,
                                'entry_price': entry_price,
                                'risk_reward_achieved': abs(pnl_percent / 100) if pnl > 0 else 0,
                                'trade_duration': (datetime.now() - monitoring_start).total_seconds(),
                                'successful': pnl > 0,
                                'signal_data': signal
                            }
                            
                            # Store in Redis for ML training
                            if self.redis_client:
                                key = f"ml_trade_outcome:{symbol}:{order_id}"
                                await self.redis_client.setex(
                                    key,
                                    86400 * 30,  # Keep for 30 days
                                    json.dumps(outcome_data, default=str)
                                )
                            
                            # Update ML training data with outcome
                            if trade_closed or pnl_percent > 100 or pnl_percent < -50:
                                # Significant move or trade closed
                                ml_predictor.training_data.append(outcome_data)
                                
                                # Trigger ML model update if enough new data
                                if len(ml_predictor.training_data) % 10 == 0:
                                    asyncio.create_task(self._retrain_ml_models())
                                
                                logger.info(
                                    f"📊 Trade outcome recorded: {symbol} "
                                    f"Outcome={outcome}, PnL={pnl:.2f} ({pnl_percent:.2f}%), "
                                    f"R:R achieved={outcome_data['risk_reward_achieved']:.2f}"
                                )
                                
                                if trade_closed:
                                    return  # Stop monitoring
                    
                    if not position_found:
                        # Position closed externally
                        logger.info(f"Position for {symbol} no longer exists")
                        return
                    
                except Exception as e:
                    logger.error(f"Error checking position for {symbol}: {e}")
                
                await asyncio.sleep(check_interval)
            
            logger.info(f"Stopped monitoring {symbol} after {max_monitoring_time/3600:.1f} hours")
            
        except Exception as e:
            logger.error(f"Error monitoring trade outcome: {e}")
    
    async def _retrain_ml_models(self):
        """Retrain ML models with new outcome data"""
        try:
            # Only retrain if we have enough quality data
            quality_trades = [
                t for t in ml_predictor.training_data 
                if isinstance(t, dict) and t.get('risk_reward_ratio', 0) >= 2.0
            ]
            
            if len(quality_trades) >= 20:
                logger.info(f"Retraining ML models with {len(quality_trades)} quality trades...")
                
                # Validate and train
                if ml_validator.validate_training_data(quality_trades):
                    ml_predictor.train_models(quality_trades)
                    
                    # Save updated models
                    ml_predictor.save_models("/tmp/ml_models")
                    
                    logger.info(f"✅ ML models retrained and saved successfully")
                    
                    # Send notification about improved accuracy
                    if self.telegram_bot and ml_predictor.get_accuracy() > 0:
                        message = (
                            f"🤖 ML Model Updated\n"
                            f"Training samples: {len(quality_trades)}\n"
                            f"Model accuracy: {ml_predictor.get_accuracy():.2%}\n"
                            f"The bot is learning and improving!"
                        )
                        for chat_id in settings.telegram_allowed_chat_ids:
                            try:
                                await self.telegram_bot.send_notification(chat_id, message)
                            except:
                                pass
                                
        except Exception as e:
            logger.error(f"Error retraining ML models: {e}")