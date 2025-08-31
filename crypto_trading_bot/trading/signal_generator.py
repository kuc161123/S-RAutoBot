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
        
        # Configurable cooldown from environment
        cooldown_minutes = config.signal_cooldown_minutes if hasattr(config, 'signal_cooldown_minutes') else 5
        self.signal_cooldown = timedelta(minutes=cooldown_minutes)
        self.scalp_cooldown = timedelta(minutes=max(cooldown_minutes // 2, 1))  # Half for scalps, minimum 1 minute
        
        logger.info("Signal generator initialized")
    
    async def start(self, symbols: List[str]):
        """Start signal generation"""
        try:
            logger.info(f"Starting signal generator for {len(symbols)} symbols")
            
            # Initialize exchange with symbols
            await self.exchange.initialize(symbols)
            
            # Recover existing positions from exchange
            await self._recover_positions()
            
            # Set up WebSocket callbacks
            self.exchange.on_kline_update = self.on_kline_update
            
            # Main scanning loop - CONTINUOUS AGGRESSIVE SCANNING
            while True:
                try:
                    # Scan for signals on ALL symbols continuously
                    await self.scan_for_signals()
                    
                    # Check existing positions
                    await self.executor.check_positions()
                    
                    # Log scanning status
                    open_positions = len(self.executor.position_manager.positions)
                    max_positions = self.config.max_positions
                    logger.info(f"Scanning complete - Positions: {open_positions}/{max_positions}")
                    
                    # Shorter interval for more aggressive scanning
                    scan_interval = min(self.config.scan_interval, 30)  # Max 30 seconds
                    await asyncio.sleep(scan_interval)
                    
                except Exception as e:
                    logger.error(f"Error in scan loop: {e}")
                    await asyncio.sleep(5)
                    
        except Exception as e:
            logger.error(f"Failed to start signal generator: {e}")
    
    async def _recover_positions(self):
        """Recover existing positions from exchange on startup"""
        try:
            logger.info("Checking for existing positions...")
            positions = self.exchange.get_positions()
            
            if positions:
                logger.info(f"Found {len(positions)} existing positions to recover")
                for pos in positions:
                    # Add to position manager
                    self.executor.position_manager.add_position(
                        symbol=pos['symbol'],
                        side="BUY" if pos['side'] == "Buy" else "SELL",
                        entry_price=pos['entry_price'],
                        size=pos['size'],
                        stop_loss=0,  # Will be managed by exchange
                        take_profit=0  # Will be managed by exchange
                    )
                    logger.info(f"Recovered position: {pos['symbol']} {pos['side']} Size: {pos['size']}")
            else:
                logger.info("No existing positions to recover")
                
        except Exception as e:
            logger.error(f"Error recovering positions: {e}")
    
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
            # Batch symbols to avoid rate limiting (5 symbols at a time)
            symbols = list(self.exchange.kline_data.keys())
            batch_size = 5
            
            for i in range(0, len(symbols), batch_size):
                batch = symbols[i:i+batch_size]
                
                # Update market data with indicators for this batch
                for symbol in batch:
                    df = self.exchange.kline_data.get(symbol)
                    if df is not None and len(df) > 0:
                        from utils.indicators import add_all_indicators
                        df_with_indicators = add_all_indicators(df, vars(self.config))
                        self.market_data[symbol] = df_with_indicators
                
                # Small delay between batches to avoid rate limiting
                if i + batch_size < len(symbols):
                    await asyncio.sleep(0.5)
            
            # Get signals from strategy
            signals = self.strategy.scan_symbols(self.market_data)
            
            # Process signals with rate limiting
            for i, signal in enumerate(signals):
                await self.process_signal(signal)
                # Small delay between signal processing to avoid order rate limits
                if i < len(signals) - 1:
                    await asyncio.sleep(0.2)
                
        except Exception as e:
            logger.error(f"Error scanning for signals: {e}")
    
    async def process_signal(self, signal):
        """Process a trading signal"""
        try:
            symbol = signal.symbol
            
            # IMPORTANT: Only apply cooldown if we HAVE a position
            # This allows continuous scanning for symbols without positions
            if self.executor.position_manager.has_position(symbol):
                logger.debug(f"Already have position in {symbol}, skipping signal")
                return
            
            # Check cooldown ONLY for failed attempts (not successful trades)
            if symbol in self.last_signal_time and symbol not in self.executor.position_manager.positions:
                time_since_last = datetime.now() - self.last_signal_time[symbol]
                cooldown = timedelta(minutes=2)  # Short cooldown for retry after failed attempt
                if time_since_last < cooldown:
                    logger.debug(f"Signal for {symbol} in retry cooldown")
                    return
            
            # Check confidence threshold
            if signal.confidence < 0.5:  # Minimum 50% confidence
                logger.debug(f"Signal confidence too low for {symbol}: {signal.confidence}")
                return
            
            # Execute signal
            success = await self.executor.execute_signal(signal)
            
            # Only track failed attempts for cooldown
            if not success:
                self.last_signal_time[symbol] = datetime.now()
                logger.info(f"Signal failed for {symbol}, will retry after cooldown")
            else:
                # Clear cooldown on success
                if symbol in self.last_signal_time:
                    del self.last_signal_time[symbol]
                logger.info(f"Signal executed successfully for {symbol}")
            
        except Exception as e:
            logger.error(f"Error processing signal: {e}")
    
    async def stop(self):
        """Stop signal generator"""
        try:
            # DO NOT close positions on shutdown - keep them running
            # await self.executor.close_all_positions()  # REMOVED - keeps trades active
            
            logger.info("Keeping all positions open during shutdown")
            
            # Show current positions before stopping
            positions = self.executor.position_manager.get_open_positions()
            if positions:
                logger.info(f"Active positions that will remain open: {len(positions)}")
                for pos in positions:
                    logger.info(f"  - {pos.symbol}: {pos.side} ${pos.pnl:.2f}")
            
            # Clean up exchange connections
            await self.exchange.cleanup()
            
            logger.info("Signal generator stopped - positions remain active")
            
        except Exception as e:
            logger.error(f"Error stopping signal generator: {e}")