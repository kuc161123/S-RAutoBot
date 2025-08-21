"""
Integrated Trading Engine with all enhancements
Combines multi-timeframe scanning, ML optimization, and position management
"""
import asyncio
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
import structlog
import redis.asyncio as redis

from ..api.bybit_client import BybitClient
from ..strategy.advanced_supply_demand import AdvancedSupplyDemandStrategy
from ..strategy.ml_predictor import ml_predictor
from .position_manager import EnhancedPositionManager
from .multi_timeframe_scanner import MultiTimeframeScanner
from .order_manager import OrderManager
from ..telegram.bot import TradingBot
from ..config import settings
from ..db.database import DatabaseManager
from ..utils.reliability import safe_executor, error_recovery
from ..utils.logging import trading_logger

logger = structlog.get_logger(__name__)

class IntegratedTradingEngine:
    """
    Main trading engine that integrates all components:
    - Multi-timeframe scanning (5m, 15m, 1h, 4h)
    - Advanced Supply & Demand with ML
    - Position management (one per symbol, long/short)
    - Leverage and margin optimization
    - Redis caching
    - PostgreSQL persistence
    """
    
    def __init__(
        self,
        bybit_client: BybitClient,
        telegram_bot: TradingBot
    ):
        self.client = bybit_client
        self.telegram_bot = telegram_bot
        self.db_manager = DatabaseManager()
        
        # Initialize components
        self.strategy = AdvancedSupplyDemandStrategy()
        self.position_manager = EnhancedPositionManager(bybit_client)
        self.scanner = MultiTimeframeScanner(bybit_client, self.strategy)
        self.order_manager = OrderManager(bybit_client)
        
        # Redis for real-time data
        self.redis_client = None
        
        # Trading state
        self.is_running = False
        self.positions_per_symbol = {}  # symbol -> {type: 'long'/'short', data: {...}}
        self.pending_orders = {}
        
        # Top 300 symbols
        self.monitored_symbols = settings.default_symbols[:300]
        
        # Performance tracking
        self.daily_stats = {
            'trades': 0,
            'pnl': 0,
            'fees': 0,
            'winning_trades': 0,
            'losing_trades': 0
        }
        
    async def initialize(self):
        """Initialize all components"""
        try:
            # Initialize Redis
            self.redis_client = redis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            await self.redis_client.ping()
            logger.info("Redis initialized for integrated engine")
            
            # Initialize scanner
            await self.scanner.initialize()
            
            # Set leverage and margin for all symbols
            logger.info("Setting leverage and margin for 300 symbols...")
            await self._initialize_symbol_settings()
            
            # Load ML models if available
            try:
                ml_predictor.load_models("/tmp/ml_models")
                logger.info("ML models loaded")
            except:
                logger.info("No pre-trained ML models, will train from scratch")
            
            # Load existing positions
            await self._load_existing_positions()
            
            logger.info("Integrated trading engine initialized")
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise
    
    async def _initialize_symbol_settings(self):
        """Set leverage and margin mode for all symbols"""
        
        # Set leverage for all symbols (using ML-optimized or default)
        leverage = settings.default_leverage
        margin_mode = settings.default_margin_mode.value
        
        # Apply to all 300 symbols in batches
        symbols = self.monitored_symbols
        
        # Set leverage
        leverage_results = await self.position_manager.set_leverage_all_symbols(
            leverage, symbols
        )
        
        success_count = sum(1 for v in leverage_results.values() if v)
        logger.info(f"Leverage set for {success_count}/{len(symbols)} symbols")
        
        # Set margin mode
        margin_results = await self.position_manager.set_margin_mode_all_symbols(
            margin_mode, symbols
        )
        
        success_count = sum(1 for v in margin_results.values() if v)
        logger.info(f"Margin mode set for {success_count}/{len(symbols)} symbols")
    
    async def _load_existing_positions(self):
        """Load existing positions from exchange"""
        
        try:
            positions = await self.client.get_positions()
            
            for position in positions:
                if float(position['size']) > 0:
                    symbol = position['symbol']
                    side = position['side']
                    
                    # Determine position type
                    position_type = 'long' if side == 'Buy' else 'short'
                    
                    # Track position
                    self.positions_per_symbol[symbol] = {
                        'type': position_type,
                        'data': position,
                        'entry_price': float(position['avgPrice']),
                        'size': float(position['size']),
                        'unrealized_pnl': float(position['unrealisedPnl'])
                    }
                    
                    # Update scanner
                    self.scanner.update_position_status(symbol, position_type)
            
            logger.info(f"Loaded {len(self.positions_per_symbol)} existing positions")
            
        except Exception as e:
            logger.error(f"Error loading positions: {e}")
    
    async def start(self):
        """Start the trading engine"""
        
        if self.is_running:
            logger.warning("Trading engine already running")
            return
        
        self.is_running = True
        logger.info("Starting integrated trading engine...")
        
        # Start scanner
        await self.scanner.start_scanning()
        
        # Start main trading loop
        asyncio.create_task(self._main_loop())
        
        # Start signal processor
        asyncio.create_task(self._process_signals())
        
        # Start position monitor
        asyncio.create_task(self._monitor_positions())
        
        logger.info("Trading engine started")
    
    async def stop(self):
        """Stop the trading engine"""
        
        if not self.is_running:
            logger.warning("Trading engine not running")
            return
        
        self.is_running = False
        
        # Stop scanner
        await self.scanner.stop_scanning()
        
        logger.info("Trading engine stopped")
    
    async def _main_loop(self):
        """Main trading loop"""
        
        while self.is_running:
            try:
                # Update market data
                await self._update_market_data()
                
                # Check for ML model updates
                if len(ml_predictor.training_data) >= 100:
                    ml_predictor.train_models()
                
                # Update daily stats
                await self._update_daily_stats()
                
                await asyncio.sleep(30)  # Run every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await error_recovery.handle_error(e, {'engine': self})
                await asyncio.sleep(10)
    
    async def _process_signals(self):
        """Process trading signals from scanner"""
        
        while self.is_running:
            try:
                # Get signals from Redis queue
                signal_data = await self.redis_client.rpop("signal_queue")
                
                if signal_data:
                    import json
                    signal_info = json.loads(signal_data)
                    symbol = signal_info['symbol']
                    
                    # Get full signal data
                    signal = await self.redis_client.get(f"signal:{symbol}")
                    if signal:
                        signal = json.loads(signal)
                        await self._execute_signal(symbol, signal)
                
                await asyncio.sleep(1)  # Check every second
                
            except Exception as e:
                logger.error(f"Error processing signals: {e}")
                await asyncio.sleep(5)
    
    async def _execute_signal(self, symbol: str, signal: Dict):
        """Execute a trading signal"""
        
        try:
            # Check if we can take position (one per symbol)
            if symbol in self.positions_per_symbol:
                logger.info(f"Already have position for {symbol}, skipping signal")
                return
            
            # Get account info
            account_info = await self.client.get_account_info()
            balance = float(account_info.get('totalWalletBalance', 0))
            
            if balance <= 0:
                logger.error("Insufficient balance")
                return
            
            # Get ML-optimized parameters
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
            
            # Prepare order
            side = 'Buy' if signal['type'] == 'BUY' else 'Sell'
            position_type = 'long' if signal['type'] == 'BUY' else 'short'
            
            # Set leverage for this specific trade
            await self.client.set_leverage(symbol, parameters.final_leverage)
            
            # Set margin mode
            await self.client.set_margin_mode(symbol, parameters.margin_mode)
            
            # Place order
            order_data = {
                'symbol': symbol,
                'side': side,
                'qty': parameters.final_size,
                'order_type': 'Market',
                'stop_loss': signal['stop_loss'],
                'take_profit': signal['take_profit_1'],
                'reduce_only': False
            }
            
            order_id = await self.order_manager.place_order(**order_data)
            
            if order_id:
                # Track position
                self.positions_per_symbol[symbol] = {
                    'type': position_type,
                    'order_id': order_id,
                    'entry_price': signal['entry_price'],
                    'stop_loss': signal['stop_loss'],
                    'take_profit_1': signal['take_profit_1'],
                    'take_profit_2': signal['take_profit_2'],
                    'size': parameters.final_size,
                    'leverage': parameters.final_leverage,
                    'ml_confidence': parameters.confidence_score,
                    'entry_time': datetime.now()
                }
                
                # Update scanner
                self.scanner.update_position_status(symbol, position_type)
                
                # Log to database
                trade_id = self.db_manager.log_trade({
                    'chat_id': settings.telegram_allowed_chat_ids[0] if settings.telegram_allowed_chat_ids else 0,
                    'symbol': symbol,
                    'order_id': order_id,
                    'side': side,
                    'position_type': position_type,
                    'entry_price': signal['entry_price'],
                    'stop_loss': signal['stop_loss'],
                    'take_profit_1': signal['take_profit_1'],
                    'take_profit_2': signal['take_profit_2'],
                    'position_size': parameters.final_size,
                    'leverage': parameters.final_leverage,
                    'ml_leverage': parameters.ml_leverage,
                    'margin_mode': parameters.margin_mode,
                    'zone_type': signal.get('zone', {}).get('zone_type'),
                    'zone_score': signal.get('zone', {}).get('composite_score'),
                    'ml_confidence': parameters.confidence_score,
                    'confluence_score': signal.get('confluence_score'),
                    'timeframes': signal.get('confirming_timeframes')
                })
                
                # Send notification
                await self._send_trade_notification(symbol, signal, parameters)
                
                # Update daily stats
                self.daily_stats['trades'] += 1
                
                logger.info(
                    f"Executed {position_type} position for {symbol}: "
                    f"Size={parameters.final_size:.4f}, "
                    f"Leverage={parameters.final_leverage}x, "
                    f"ML Confidence={parameters.confidence_score:.2%}"
                )
                
        except Exception as e:
            logger.error(f"Error executing signal for {symbol}: {e}")
            # Clean up position tracking on error
            if symbol in self.positions_per_symbol:
                del self.positions_per_symbol[symbol]
            self.scanner.update_position_status(symbol, 'closed')
    
    async def _monitor_positions(self):
        """Monitor and manage open positions"""
        
        while self.is_running:
            try:
                # Get current positions from exchange
                positions = await self.client.get_positions()
                
                for position in positions:
                    if float(position['size']) > 0:
                        symbol = position['symbol']
                        
                        # Check if position closed
                        if symbol in self.positions_per_symbol and float(position['size']) == 0:
                            await self._handle_position_closed(symbol, position)
                        
                        # Update position data
                        elif symbol in self.positions_per_symbol:
                            self.positions_per_symbol[symbol]['data'] = position
                            self.positions_per_symbol[symbol]['unrealized_pnl'] = float(position['unrealisedPnl'])
                
                # Check for positions that should be closed
                for symbol in list(self.positions_per_symbol.keys()):
                    if symbol not in [p['symbol'] for p in positions if float(p['size']) > 0]:
                        # Position was closed
                        await self._handle_position_closed(symbol, None)
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Error monitoring positions: {e}")
                await asyncio.sleep(10)
    
    async def _handle_position_closed(self, symbol: str, position_data: Optional[Dict]):
        """Handle a closed position"""
        
        if symbol not in self.positions_per_symbol:
            return
        
        position_info = self.positions_per_symbol[symbol]
        
        # Calculate PnL
        if position_data:
            pnl = float(position_data.get('realisedPnl', 0))
            pnl_percent = (pnl / (position_info['entry_price'] * position_info['size'])) * 100
        else:
            pnl = 0
            pnl_percent = 0
        
        # Update ML with outcome
        await self.position_manager.record_position_outcome(
            symbol=symbol,
            outcome={
                'profit_percent': pnl_percent,
                'risk_percent': abs(position_info['entry_price'] - position_info['stop_loss']) / position_info['entry_price'] * 100,
                'zone': position_info.get('zone')
            }
        )
        
        # Update database
        # ... database update code ...
        
        # Update daily stats
        if pnl > 0:
            self.daily_stats['winning_trades'] += 1
        else:
            self.daily_stats['losing_trades'] += 1
        self.daily_stats['pnl'] += pnl
        
        # Clear position tracking
        del self.positions_per_symbol[symbol]
        
        # Update scanner - can trade this symbol again
        self.scanner.update_position_status(symbol, 'closed')
        
        logger.info(
            f"Position closed for {symbol}: "
            f"PnL={pnl:.2f} ({pnl_percent:.2f}%)"
        )
    
    async def _send_trade_notification(self, symbol: str, signal: Dict, parameters):
        """Send trade notification via Telegram"""
        
        message = f"""
🎯 **New {signal['type']} Position**

Symbol: {symbol}
Type: {'Long 📈' if signal['type'] == 'BUY' else 'Short 📉'}
Entry: {signal['entry_price']:.4f}
Stop Loss: {signal['stop_loss']:.4f}
TP1: {signal['take_profit_1']:.4f}
TP2: {signal['take_profit_2']:.4f}

Position Size: {parameters.final_size:.4f}
Leverage: {parameters.final_leverage}x (ML: {parameters.ml_leverage}x)
Margin Mode: {parameters.margin_mode}

ML Confidence: {parameters.confidence_score:.1%}
Confluence Score: {signal.get('confluence_score', 0)}
Timeframes: {', '.join(signal.get('confirming_timeframes', []))}

Risk Score: {parameters.risk_score:.2f}
"""
        
        for chat_id in settings.telegram_allowed_chat_ids:
            await self.telegram_bot.send_notification(chat_id, message)
    
    async def _update_market_data(self):
        """Update market data in Redis"""
        
        # This would update general market conditions
        # For now, simplified implementation
        pass
    
    async def _update_daily_stats(self):
        """Update daily statistics"""
        
        # Save to database
        await self.db_manager.update_daily_stats(self.daily_stats)
    
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
            'daily_stats': self.daily_stats,
            'ml_trained': ml_predictor.model_trained,
            'ml_samples': len(ml_predictor.training_data)
        }