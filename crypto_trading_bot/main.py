#!/usr/bin/env python3
"""
Crypto Trading Bot - Main Entry Point
Simple, clean, and efficient trading bot for Bybit
"""
import asyncio
import signal
import sys
from typing import Optional
import structlog

# Import our modules
from config import settings
from exchange.bybit_client import BybitClient
from strategy.simple_strategy import SimpleStrategy
from trading.position_manager import PositionManager
from trading.order_executor import OrderExecutor
from trading.signal_generator import SignalGenerator
from telegram.bot import TelegramBot
from utils.logger import setup_logger

# Setup logger
logger = setup_logger(settings.log_level)

class TradingBot:
    """Main trading bot application"""
    
    def __init__(self):
        self.exchange: Optional[BybitClient] = None
        self.strategy: Optional[SimpleStrategy] = None
        self.position_manager: Optional[PositionManager] = None
        self.order_executor: Optional[OrderExecutor] = None
        self.signal_generator: Optional[SignalGenerator] = None
        self.telegram_bot: Optional[TelegramBot] = None
        self.is_running = False
        
        logger.info("Trading bot initializing...")
    
    async def initialize(self):
        """Initialize all components"""
        try:
            # Initialize exchange client
            self.exchange = BybitClient(
                api_key=settings.bybit_api_key,
                api_secret=settings.bybit_api_secret,
                testnet=settings.bybit_testnet
            )
            
            # Initialize strategy
            self.strategy = SimpleStrategy(vars(settings))
            
            # Initialize position manager
            self.position_manager = PositionManager(
                max_positions=settings.max_positions,
                risk_per_trade=settings.risk_per_trade
            )
            
            # Initialize Telegram bot if configured
            if settings.telegram_enabled:
                self.telegram_bot = TelegramBot(
                    token=settings.telegram_bot_token,
                    chat_ids=settings.telegram_chat_ids
                )
            
            # Initialize order executor
            self.order_executor = OrderExecutor(
                exchange_client=self.exchange,
                position_manager=self.position_manager,
                telegram_notifier=self.telegram_bot
            )
            
            # Initialize signal generator
            self.signal_generator = SignalGenerator(
                exchange_client=self.exchange,
                strategy=self.strategy,
                order_executor=self.order_executor,
                config=settings
            )
            
            # Link telegram bot to signal generator
            if self.telegram_bot:
                self.telegram_bot.signal_generator = self.signal_generator
            
            logger.info("All components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            return False
    
    async def start(self):
        """Start the trading bot"""
        try:
            # Initialize components
            if not await self.initialize():
                logger.error("Failed to initialize bot")
                return
            
            self.is_running = True
            
            # Start Telegram bot
            if self.telegram_bot:
                await self.telegram_bot.start()
            
            # Log startup info
            balance = self.exchange.get_account_balance()
            logger.info(f"Starting trading bot...")
            logger.info(f"Mode: {'TESTNET' if settings.is_testnet else 'MAINNET'}")
            logger.info(f"Balance: ${balance:.2f}")
            logger.info(f"Symbols: {', '.join(settings.initial_symbols)}")
            logger.info(f"Risk per trade: {settings.risk_per_trade*100:.1f}%")
            logger.info(f"Max positions: {settings.max_positions}")
            logger.info(f"Leverage: {settings.leverage}x")
            
            # Delay before starting
            logger.info(f"Starting in {settings.startup_delay} seconds...")
            await asyncio.sleep(settings.startup_delay)
            
            # Start signal generation
            await self.signal_generator.start(settings.initial_symbols)
            
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the trading bot"""
        if not self.is_running:
            return
        
        self.is_running = False
        logger.info("Shutting down trading bot...")
        
        try:
            # Stop signal generator (closes positions)
            if self.signal_generator:
                await self.signal_generator.stop()
            
            # Stop Telegram bot
            if self.telegram_bot:
                await self.telegram_bot.stop()
            
            # Clean up exchange
            if self.exchange:
                await self.exchange.cleanup()
            
            # Show final statistics
            if self.position_manager:
                stats = self.position_manager.get_statistics()
                logger.info("Final Statistics:")
                logger.info(f"  Total trades: {stats['total_trades']}")
                logger.info(f"  Win rate: {stats['win_rate']:.1f}%")
                logger.info(f"  Total P&L: ${stats['total_pnl']:.2f}")
            
            logger.info("Trading bot stopped successfully")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

async def main():
    """Main entry point"""
    bot = TradingBot()
    
    # Setup signal handlers
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}")
        asyncio.create_task(bot.stop())
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start bot
    await bot.start()

if __name__ == "__main__":
    # Print startup banner
    print("=" * 50)
    print("       CRYPTO TRADING BOT v2.0")
    print("       Simple. Clean. Efficient.")
    print("=" * 50)
    print()
    
    try:
        # Check if .env file exists
        import os
        if not os.path.exists('.env'):
            print("❌ ERROR: .env file not found!")
            print("Please create a .env file with the following variables:")
            print("  BYBIT_API_KEY=your_api_key")
            print("  BYBIT_API_SECRET=your_api_secret")
            print("  BYBIT_TESTNET=true")
            print("  TELEGRAM_BOT_TOKEN=your_bot_token")
            print("  TELEGRAM_CHAT_IDS=[your_chat_id]")
            sys.exit(1)
        
        # Run the bot
        asyncio.run(main())
        
    except KeyboardInterrupt:
        print("\n👋 Bot stopped by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)