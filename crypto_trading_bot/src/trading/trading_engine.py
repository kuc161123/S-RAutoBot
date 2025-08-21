import asyncio
from typing import Dict, Set, List, Optional
from datetime import datetime, timedelta
import structlog

from ..api.bybit_client import BybitClient
from ..strategy.supply_demand import SupplyDemandStrategy
from .order_manager import OrderManager
from ..telegram.bot import TradingBot
from ..config import settings
from ..db.database import DatabaseManager
from ..utils.logging import trading_logger

logger = structlog.get_logger(__name__)

class TradingEngine:
    """Main trading engine that coordinates all components"""
    
    def __init__(
        self,
        bybit_client: BybitClient,
        strategy: SupplyDemandStrategy,
        order_manager: OrderManager,
        telegram_bot: TradingBot
    ):
        self.client = bybit_client
        self.strategy = strategy
        self.order_manager = order_manager
        self.telegram_bot = telegram_bot
        self.db_manager = DatabaseManager()
        
        self.is_running = False
        self.monitored_symbols: Set[str] = set()
        self.scan_tasks: Dict[str, asyncio.Task] = {}
        self.market_data: Dict[str, Dict] = {}
        
        # Default symbols to monitor
        self.default_symbols = [
            "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT",
            "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "MATICUSDT", "DOTUSDT"
        ]
        
    async def run(self):
        """Main trading loop"""
        logger.info("Trading engine starting...")
        
        # Load monitored symbols
        await self.load_monitored_symbols()
        
        # Main loop
        while True:
            try:
                if self.is_running:
                    await self._trading_cycle()
                
                await asyncio.sleep(settings.symbol_scan_interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(10)
    
    async def start(self):
        """Start trading"""
        if self.is_running:
            logger.warning("Trading already running")
            return
        
        self.is_running = True
        logger.info("Trading started")
        
        # Start symbol scanners
        await self._start_symbol_scanners()
    
    async def stop(self):
        """Stop trading"""
        if not self.is_running:
            logger.warning("Trading already stopped")
            return
        
        self.is_running = False
        logger.info("Trading stopped")
        
        # Stop symbol scanners
        await self._stop_symbol_scanners()
    
    async def load_monitored_symbols(self):
        """Load symbols to monitor"""
        # For now, use default symbols
        # In production, load from database or user preferences
        self.monitored_symbols = set(self.default_symbols)
        logger.info(f"Loaded {len(self.monitored_symbols)} symbols")
    
    async def add_symbol(self, symbol: str):
        """Add symbol to monitoring"""
        symbol = symbol.upper()
        
        # Check if symbol exists
        instrument = self.client.get_instrument(symbol)
        if not instrument:
            logger.error(f"Unknown symbol: {symbol}")
            return False
        
        self.monitored_symbols.add(symbol)
        
        # Start scanner if trading is active
        if self.is_running and symbol not in self.scan_tasks:
            task = asyncio.create_task(self._symbol_scanner(symbol))
            self.scan_tasks[symbol] = task
        
        logger.info(f"Added {symbol} to monitoring")
        return True
    
    async def remove_symbol(self, symbol: str):
        """Remove symbol from monitoring"""
        symbol = symbol.upper()
        
        if symbol in self.monitored_symbols:
            self.monitored_symbols.remove(symbol)
            
            # Stop scanner
            if symbol in self.scan_tasks:
                self.scan_tasks[symbol].cancel()
                del self.scan_tasks[symbol]
            
            logger.info(f"Removed {symbol} from monitoring")
            return True
        
        return False
    
    async def _start_symbol_scanners(self):
        """Start scanners for all monitored symbols"""
        for symbol in self.monitored_symbols:
            if symbol not in self.scan_tasks:
                task = asyncio.create_task(self._symbol_scanner(symbol))
                self.scan_tasks[symbol] = task
        
        logger.info(f"Started {len(self.scan_tasks)} symbol scanners")
    
    async def _stop_symbol_scanners(self):
        """Stop all symbol scanners"""
        for task in self.scan_tasks.values():
            task.cancel()
        
        self.scan_tasks.clear()
        logger.info("Stopped all symbol scanners")
    
    async def _symbol_scanner(self, symbol: str):
        """Scanner for a specific symbol"""
        logger.info(f"Scanner started for {symbol}")
        
        # Subscribe to WebSocket for this symbol
        self.client.subscribe_klines(
            symbol, 
            "15",  # 15 minute candles
            lambda data: asyncio.create_task(self._handle_kline_update(symbol, data))
        )
        
        while symbol in self.monitored_symbols and self.is_running:
            try:
                # Fetch latest data
                df = await self.client.get_klines(symbol, "15", limit=100)
                
                if not df.empty:
                    # Detect zones
                    zones = self.strategy.detect_zones(df, symbol, "15m")
                    
                    # Update strategy zones
                    self.strategy.zones[symbol] = zones
                    
                    # Log zone detection
                    for zone in zones:
                        trading_logger.log_zone_detected({
                            'symbol': symbol,
                            'type': zone.zone_type.value,
                            'score': zone.score
                        })
                    
                    # Check for entry signals
                    await self._check_entry_signals(symbol, df.iloc[-1]['close'])
                
                # Update positions
                await self._update_positions(symbol)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in scanner for {symbol}: {e}")
                await asyncio.sleep(10)
        
        logger.info(f"Scanner stopped for {symbol}")
    
    async def _handle_kline_update(self, symbol: str, data: Dict):
        """Handle real-time kline updates"""
        try:
            # Update market data
            self.market_data[symbol] = {
                'price': float(data.get('close', 0)),
                'volume': float(data.get('volume', 0)),
                'timestamp': datetime.fromtimestamp(data.get('timestamp', 0) / 1000)
            }
            
            # Update zones with current price
            self.strategy.update_zones(symbol, self.market_data[symbol]['price'])
            
        except Exception as e:
            logger.error(f"Error handling kline update: {e}")
    
    async def _check_entry_signals(self, symbol: str, current_price: float):
        """Check for entry signals"""
        try:
            # Skip if already have position
            if symbol in self.order_manager.active_positions:
                return
            
            # Get account info
            account_info = await self.client.get_account_info()
            balance = float(account_info.get('totalWalletBalance', 0))
            
            # Get instrument info
            instrument = self.client.get_instrument(symbol)
            
            # Check for signal
            signal = self.strategy.check_entry_signal(
                symbol,
                current_price,
                balance,
                settings.default_risk_percent,
                instrument
            )
            
            if signal:
                logger.info(f"Entry signal detected for {symbol}")
                
                # Execute the signal
                order_id = await self.order_manager.execute_signal(signal, balance)
                
                if order_id:
                    # Log trade
                    trade_id = self.db_manager.log_trade({
                        'chat_id': settings.telegram_allowed_chat_ids[0] if settings.telegram_allowed_chat_ids else 0,
                        'symbol': symbol,
                        'order_id': order_id,
                        'side': signal.side,
                        'entry_price': signal.entry_price,
                        'stop_loss': signal.stop_loss,
                        'take_profit_1': signal.take_profit_1,
                        'take_profit_2': signal.take_profit_2,
                        'position_size': signal.position_size,
                        'zone_type': signal.zone.zone_type.value,
                        'zone_score': signal.zone.score
                    })
                    
                    # Log to monitoring
                    trading_logger.log_trade_opened({
                        'symbol': symbol,
                        'side': signal.side,
                        'entry_price': signal.entry_price,
                        'position_size': signal.position_size,
                        'zone_score': signal.zone.score
                    })
                    
                    # Send notification
                    from ..telegram.formatters import format_trade_signal
                    message = format_trade_signal({
                        'symbol': symbol,
                        'side': signal.side,
                        'entry_price': signal.entry_price,
                        'stop_loss': signal.stop_loss,
                        'take_profit_1': signal.take_profit_1,
                        'take_profit_2': signal.take_profit_2,
                        'position_size': signal.position_size,
                        'zone_type': signal.zone.zone_type.value,
                        'zone_score': signal.zone.score,
                        'confidence': signal.confidence * 100
                    })
                    
                    for chat_id in settings.telegram_allowed_chat_ids:
                        await self.telegram_bot.send_notification(chat_id, message)
                
        except Exception as e:
            logger.error(f"Error checking entry signals: {e}")
    
    async def _update_positions(self, symbol: str):
        """Update positions for a symbol"""
        try:
            if symbol not in self.market_data:
                return
            
            current_price = self.market_data[symbol]['price']
            
            # Update position management
            await self.order_manager.update_positions({symbol: current_price})
            
        except Exception as e:
            logger.error(f"Error updating positions: {e}")
    
    async def _trading_cycle(self):
        """Main trading cycle"""
        try:
            # Update account metrics
            account_info = await self.client.get_account_info()
            balance = float(account_info.get('totalWalletBalance', 0))
            
            # Calculate win rate
            daily_stats = self.order_manager.get_daily_stats()
            total_trades = daily_stats['trades']
            
            if total_trades > 0:
                # Get trades from database for win rate calculation
                trades = self.db_manager.get_closed_trades_today()
                winning_trades = len([t for t in trades if t.pnl > 0])
                current_win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
            else:
                current_win_rate = 0
            
            # Update metrics
            trading_logger.update_account_metrics(balance, current_win_rate)
            
            # Check for daily loss limit
            if abs(daily_stats['pnl']) >= balance * (settings.max_daily_loss_percent / 100):
                logger.warning("Daily loss limit reached, stopping trading")
                await self.stop()
                
                # Notify users
                for chat_id in settings.telegram_allowed_chat_ids:
                    await self.telegram_bot.send_notification(
                        chat_id,
                        "⚠️ Daily loss limit reached. Trading stopped automatically."
                    )
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
    
    def get_status(self) -> Dict:
        """Get current engine status"""
        return {
            'is_running': self.is_running,
            'monitored_symbols': list(self.monitored_symbols),
            'active_scanners': list(self.scan_tasks.keys()),
            'active_positions': len(self.order_manager.active_positions),
            'daily_stats': self.order_manager.get_daily_stats()
        }