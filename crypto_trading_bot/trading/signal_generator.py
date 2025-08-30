"""
Signal generation and coordination
"""
import asyncio
from typing import Dict, List
import pandas as pd
import structlog
from datetime import datetime, timedelta

logger = structlog.get_logger(__name__)

class SignalGenerator:
    """Generate and manage trading signals"""
    
    def __init__(self, exchange_client, strategy, order_executor, config):
        self.exchange = exchange_client
        self.strategy = strategy
        self.executor = order_executor
        self.config = config
        
        self.market_data: Dict[str, pd.DataFrame] = {}
        self.last_signal_time: Dict[str, datetime] = {}
        self.signal_cooldown = timedelta(minutes=15)  # Avoid duplicate signals
        
        logger.info("Signal generator initialized")
    
    async def start(self, symbols: List[str]):
        """Start signal generation"""
        try:
            logger.info(f"Starting signal generator for {len(symbols)} symbols")
            
            # Initialize exchange with symbols
            await self.exchange.initialize(symbols)
            
            # Set up WebSocket callbacks
            self.exchange.on_kline_update = self.on_kline_update
            
            # Main scanning loop
            while True:
                try:
                    await self.scan_for_signals()
                    await self.executor.check_positions()
                    await asyncio.sleep(self.config.scan_interval)
                    
                except Exception as e:
                    logger.error(f"Error in scan loop: {e}")
                    await asyncio.sleep(5)
                    
        except Exception as e:
            logger.error(f"Failed to start signal generator: {e}")
    
    async def on_kline_update(self, symbol: str, df: pd.DataFrame):
        """Handle real-time kline updates"""
        try:
            # Update market data
            self.market_data[symbol] = df
            
            # Add indicators
            from utils.indicators import add_all_indicators
            df_with_indicators = add_all_indicators(df, vars(self.config))
            self.market_data[symbol] = df_with_indicators
            
        except Exception as e:
            logger.error(f"Error handling kline update for {symbol}: {e}")
    
    async def scan_for_signals(self):
        """Scan all symbols for trading signals"""
        try:
            # Update market data with indicators
            for symbol in list(self.exchange.kline_data.keys()):
                df = self.exchange.kline_data.get(symbol)
                if df is not None and len(df) > 0:
                    from utils.indicators import add_all_indicators
                    df_with_indicators = add_all_indicators(df, vars(self.config))
                    self.market_data[symbol] = df_with_indicators
            
            # Get signals from strategy
            signals = self.strategy.scan_symbols(self.market_data)
            
            # Process signals
            for signal in signals:
                await self.process_signal(signal)
                
        except Exception as e:
            logger.error(f"Error scanning for signals: {e}")
    
    async def process_signal(self, signal):
        """Process a trading signal"""
        try:
            symbol = signal.symbol
            
            # Check cooldown
            if symbol in self.last_signal_time:
                time_since_last = datetime.now() - self.last_signal_time[symbol]
                if time_since_last < self.signal_cooldown:
                    logger.debug(f"Signal for {symbol} still in cooldown")
                    return
            
            # Check confidence threshold
            if signal.confidence < 0.5:  # Minimum 50% confidence
                logger.debug(f"Signal confidence too low for {symbol}: {signal.confidence}")
                return
            
            # Execute signal
            if await self.executor.execute_signal(signal):
                self.last_signal_time[symbol] = datetime.now()
                logger.info(f"Signal executed for {symbol}")
            
        except Exception as e:
            logger.error(f"Error processing signal: {e}")
    
    async def stop(self):
        """Stop signal generator"""
        try:
            # Close all positions
            await self.executor.close_all_positions()
            
            # Clean up exchange connections
            await self.exchange.cleanup()
            
            logger.info("Signal generator stopped")
            
        except Exception as e:
            logger.error(f"Error stopping signal generator: {e}")