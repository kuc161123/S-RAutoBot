"""
Multi-timeframe market scanner with HTF/LTF synchronization
Implements the multi-timeframe supply/demand strategy with market structure
"""
import asyncio
from typing import Dict, List, Set, Tuple, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import structlog
import redis.asyncio as redis
import json

from ..config import settings
from ..api.enhanced_bybit_client import EnhancedBybitClient as BybitClient
from ..strategy.advanced_supply_demand import AdvancedSupplyDemandStrategy
from ..strategy.market_structure_analyzer import MarketStructureAnalyzer, MarketStructure
from ..utils.reliability import rate_limiter, retry_with_backoff
from ..utils.symbol_rotator import SymbolRotator
from ..utils.signal_debugger import signal_debugger

logger = structlog.get_logger(__name__)

class MultiTimeframeScanner:
    """
    Enhanced scanner that synchronizes HTF supply/demand zones with LTF market structure
    for precise entry signals across all symbols
    """
    
    def __init__(self, bybit_client: BybitClient, strategy: AdvancedSupplyDemandStrategy, symbols: List[str] = None):
        self.client = bybit_client
        self.strategy = strategy
        self.structure_analyzer = MarketStructureAnalyzer()
        self.redis_client = None
        self.scanning_tasks = {}
        
        # Multi-timeframe data storage
        self.timeframe_data = {}  # symbol -> timeframe -> data
        self.htf_zones = {}  # symbol -> supply/demand zones from HTF
        self.ltf_structure = {}  # symbol -> market structure from LTF
        self.zone_confluence = {}  # symbol -> zones across timeframes
        
        # Timeframe configuration for strategy
        # Use consistent timeframe format (minutes as string)
        self.htf_timeframes = ["240", "60"]  # Higher timeframes for zones (4H, 1H)  
        self.ltf_timeframes = ["15", "5"]  # Lower timeframes for structure (15m, 5m)
        self.primary_htf = "240"  # Primary HTF for zone identification
        self.primary_ltf = "15"  # Primary LTF for entry timing
        
        # Use provided symbols or defaults
        if symbols:
            self.symbols = symbols
            logger.info(f"Scanner initialized with {len(symbols)} provided symbols")
        else:
            self.symbols = settings.default_symbols[:50]  # Fallback
            logger.warning("No symbols provided to scanner, using defaults")
        
        # Symbol rotation - use scaling config
        try:
            try:
                from ..config_modules import scaling_config
            except ImportError:
                from ..config_modules.scaling_config import scaling_config
            batch_size = scaling_config.get_batch_size()
        except Exception as e:
            # Fallback if config not available
            batch_size = min(10, max(5, len(self.symbols) // 2))
        
        self.symbol_rotator = SymbolRotator(self.symbols, max_concurrent=batch_size)
        logger.info(f"Symbol rotator configured with batch size: {batch_size} for {len(self.symbols)} symbols")
        self.current_scan_batch = []
        
        # Position tracking - one position per symbol
        self.active_positions = {}  # symbol -> position_type (long/short)
        
        # Performance tracking per symbol
        self.symbol_performance = {}  # symbol -> {success_rate, best_tf_combo}
        
        # Health monitoring
        self.scanning_task = None
        self.is_scanning = False
        self.last_scan_time = None
        self.last_batch_time = None
        self.scan_count = 0
        self.error_count = 0
        self.consecutive_errors = 0
        self.max_consecutive_errors = 10
        self.scan_metrics = {
            'total_scans': 0,
            'successful_scans': 0,
            'failed_scans': 0,
            'signals_generated': 0,
            'last_signal_time': None,
            'symbols_per_minute': 0,
            'average_scan_time': 0
        }
        
    async def initialize(self):
        """Initialize Redis connection for caching"""
        try:
            # Clear any leftover position tracking from previous runs
            self.active_positions.clear()
            logger.info("Scanner initialized, position tracking cleared")
            
            # Try to connect with retries
            max_retries = 3
            retry_delay = 2
            
            for attempt in range(max_retries):
                try:
                    self.redis_client = redis.from_url(
                        settings.redis_url,
                        encoding="utf-8",
                        decode_responses=True,
                        socket_connect_timeout=5,
                        socket_timeout=5,
                        retry_on_timeout=True
                    )
                    await self.redis_client.ping()
                    logger.info("Redis connected for multi-timeframe caching")
                    
                    # Signal queue will be connected by engine - don't connect here
                    # This prevents multiple connection attempts
                    from ..utils.signal_queue import signal_queue
                    logger.debug(f"Signal queue status: Redis={signal_queue.redis_client is not None}, Memory={signal_queue.in_memory_queue is not None}")
                    
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    logger.warning(f"Redis connection attempt {attempt + 1}/{max_retries} failed: {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
                    else:
                        logger.error("Redis connection failed after all retries")
                        self.redis_client = None
                        
        except Exception as e:
            logger.error(f"Redis initialization error: {e}")
            # Continue without Redis (degraded mode)
            self.redis_client = None
            
        # Check signal queue status (should be initialized by engine)
        try:
            from ..utils.signal_queue import signal_queue
            if not signal_queue.redis_client and not signal_queue.in_memory_queue:
                logger.warning("Signal queue not yet initialized by engine - scanner will wait")
                # Don't connect here - let engine handle it
        except Exception as e:
            logger.error(f"Failed to initialize signal queue: {e}")
    
    def update_symbols(self, new_symbols: List[str]):
        """Update the symbol list dynamically"""
        if new_symbols != self.symbols:
            logger.info(f"Updating scanner symbols from {len(self.symbols)} to {len(new_symbols)}")
            self.symbols = new_symbols
            batch_size = min(10, max(5, len(self.symbols) // 2))
            self.symbol_rotator = SymbolRotator(self.symbols, max_concurrent=batch_size)
            logger.info(f"Rotator updated with batch size: {batch_size} for {len(new_symbols)} symbols")
    
    async def start_scanning(self):
        """Start scanning with symbol rotation for efficiency and auto-recovery"""
        logger.info("\n" + "="*80)
        logger.info("🎆 MULTI-TIMEFRAME SCANNER STARTING UP")
        logger.info("="*80)
        logger.info(f"📊 Total symbols to scan: {len(self.symbols)}")
        logger.info(f"📋 Symbol list sample: {self.symbols[:10] if self.symbols else 'NO SYMBOLS!'}")
        logger.info(f"🕒 HTF timeframes (for zone detection): {self.htf_timeframes}")
        logger.info(f"🕓 LTF timeframes (for entry signals): {self.ltf_timeframes}")
        logger.info(f"🎯 Batch size: {self.batch_size} symbols per batch")
        logger.info(f"⚡ ZONE DETECTION WILL START IN SECONDS!")
        logger.info("="*80 + "\n")
        
        if self.is_scanning:
            logger.warning("Scanner already running, restarting...")
            await self.stop_scanning()
            await asyncio.sleep(2)
        
        self.is_scanning = True
        self.last_scan_time = datetime.now()
        self.consecutive_errors = 0
        
        # Log scanner configuration
        logger.info(f"Scanner config: batch_size={self.symbol_rotator.max_concurrent}, symbols={len(self.symbols)}")
        
        # Start main scanning loop with rotation and recovery wrapper
        self.scanning_task = asyncio.create_task(self._scanning_loop_with_recovery())
        logger.info("Created scanning task")
        
        # Start health monitor
        asyncio.create_task(self._health_monitor())
        logger.info("Created health monitor task")
        
        logger.info("✅ Scanner started successfully!")
    
    async def _scanning_loop_with_recovery(self):
        """Scanning loop with automatic recovery on failure"""
        max_retries = 5
        retry_count = 0
        
        while self.is_scanning:
            try:
                await self._scanning_loop()
                retry_count = 0  # Reset on successful run
                
            except Exception as e:
                retry_count += 1
                self.consecutive_errors += 1
                logger.error(f"Scanner crashed (attempt {retry_count}/{max_retries}): {e}")
                
                if retry_count >= max_retries:
                    logger.critical("Scanner failed too many times, stopping")
                    self.is_scanning = False
                    break
                
                # Exponential backoff
                wait_time = min(60, 2 ** retry_count)
                logger.info(f"Restarting scanner in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
    
    async def _scanning_loop(self):
        """Main scanning loop that rotates through symbols"""
        batch_count = 0
        logger.info(f"🚀 Scanner loop started with {len(self.symbols)} total symbols")
        
        while self.is_scanning:
            try:
                batch_count += 1
                scan_start_time = datetime.now()
                
                # Get next batch of symbols to scan
                batch = self.symbol_rotator.get_next_batch()
                self.current_scan_batch = batch
                self.last_batch_time = datetime.now()
                
                # Check if batch is empty and handle it
                if not batch:
                    logger.warning(f"Empty batch returned! Using fallback symbols")
                    # Use first few symbols as fallback
                    batch = self.symbols[:self.symbol_rotator.max_concurrent] if self.symbols else []
                    if not batch:
                        logger.error("No symbols available to scan!")
                        await asyncio.sleep(10)
                        continue
                
                # Log full batch for debugging signal generation issues
                logger.info(f"📊 Batch #{batch_count}: Scanning {len(batch)} symbols: {', '.join(batch)}")
                
                # Create tasks for batch
                tasks = []
                for symbol in batch:
                    # Always try to scan, don't skip based on positions yet
                    task = asyncio.create_task(self._scan_symbol_once(symbol))
                    tasks.append(task)
                    logger.debug(f"Created scan task for {symbol}")
                
                # Wait for batch to complete with timeout
                if tasks:
                    try:
                        results = await asyncio.wait_for(
                            asyncio.gather(*tasks, return_exceptions=True),
                            timeout=60  # 60 second timeout for batch
                        )
                        
                        # Count successes/failures
                        for result in results:
                            if isinstance(result, asyncio.CancelledError):
                                # Don't count cancellations as failures
                                logger.debug("Task cancelled during batch processing")
                            elif isinstance(result, Exception):
                                self.scan_metrics['failed_scans'] += 1
                            else:
                                self.scan_metrics['successful_scans'] += 1
                    
                    except asyncio.TimeoutError:
                        logger.warning(f"Batch scan timeout for {len(tasks)} symbols")
                        self.scan_metrics['failed_scans'] += len(tasks)
                        # Cancel remaining tasks
                        for task in tasks:
                            if not task.done():
                                task.cancel()
                
                # Update metrics
                self.scan_metrics['total_scans'] += len(batch)
                self.last_scan_time = datetime.now()
                self.scan_count += len(batch)
                self.consecutive_errors = 0  # Reset on successful batch
                
                # Calculate scan rate
                scan_duration = (datetime.now() - scan_start_time).total_seconds()
                if scan_duration > 0:
                    self.scan_metrics['symbols_per_minute'] = (len(batch) / scan_duration) * 60
                
                # Pause between batches - use scaling config
                try:
                    try:
                        from ..config_modules import scaling_config
                    except ImportError:
                        from ..config_modules.scaling_config import scaling_config
                    batch_delay = scaling_config.get_scan_delay()
                except Exception as e:
                    # Fallback if config not available
                    batch_delay = 10 if len(self.symbols) <= 20 else 30
                
                await asyncio.sleep(batch_delay)
                
                # Clean up old data more frequently (every 50 scans) for memory management
                if self.scan_count % 50 == 0:
                    await self._cleanup_old_data()
                
                # Sync positions every 10 scans to prevent stale tracking
                if self.scan_count % 10 == 0:
                    await self.sync_positions_with_exchange()
                
            except asyncio.TimeoutError:
                logger.warning("Batch scan timeout, continuing with next batch")
                self.consecutive_errors += 1
                await asyncio.sleep(2)
                
            except Exception as e:
                self.consecutive_errors += 1
                logger.error(f"Scanning loop error: {e}")
                
                if self.consecutive_errors >= self.max_consecutive_errors:
                    logger.error("Too many consecutive errors, scanner needs restart")
                    raise
                
                await asyncio.sleep(10)
    
    async def stop_scanning(self):
        """Stop all scanning tasks gracefully"""
        self.is_scanning = False
        
        # Cancel main scanning task
        if self.scanning_task and not self.scanning_task.done():
            self.scanning_task.cancel()
            try:
                await self.scanning_task
            except asyncio.CancelledError:
                logger.debug("Main scanning task cancelled successfully")
            except Exception as e:
                logger.error(f"Error stopping scanning task: {e}")
        
        # Cancel other tasks
        for task_name, task in self.scanning_tasks.items():
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    logger.debug(f"Task {task_name} cancelled successfully")
                except Exception as e:
                    logger.error(f"Error cancelling task {task_name}: {e}")
        
        self.scanning_tasks.clear()
        logger.info("Stopped all scanning tasks")
    
    async def _health_monitor(self):
        """Monitor scanner health and restart if needed"""
        check_interval = 60  # Check every minute (adjusted for slower scanning)
        stuck_threshold = 300  # Consider stuck if no scan in 5 minutes (increased for slower pace)
        restart_count = 0
        max_restarts = 10  # Allow up to 10 restarts
        status_report_interval = 0  # Counter for status reports
        
        while self.is_scanning:
            try:
                await asyncio.sleep(check_interval)
                status_report_interval += 1
                
                # Every 5 minutes, show a status report
                if status_report_interval >= 5:
                    status_report_interval = 0
                    logger.info("\n" + "="*60)
                    logger.info("📊 SCANNER STATUS REPORT")
                    logger.info("="*60)
                    logger.info(f"✅ Active positions: {len(self.active_positions)}")
                    logger.info(f"📈 Signals generated: {self.scan_metrics.get('signals_generated', 0)}")
                    logger.info(f"🔍 Symbols being monitored: {len(self.symbol_rotator.active_symbols)}")
                    
                    if self.scan_metrics.get('last_signal_time'):
                        time_since = (datetime.now() - self.scan_metrics['last_signal_time']).total_seconds() / 60
                        logger.info(f"⏰ Last signal: {time_since:.1f} minutes ago")
                    
                    # Show some active zones if available
                    if hasattr(self.strategy, 'zones'):
                        total_zones = sum(len(zones) for zones in self.strategy.zones.values())
                        logger.info(f"🎯 Total zones tracked: {total_zones}")
                    
                    logger.info("="*60 + "\n")
                
                # Check if scanner is stuck
                if self.last_scan_time:
                    time_since_last_scan = (datetime.now() - self.last_scan_time).total_seconds()
                    
                    if time_since_last_scan > stuck_threshold:
                        restart_count += 1
                        logger.error(f"⚠️ Scanner stuck! No scan for {time_since_last_scan:.0f}s (restart #{restart_count})")
                        
                        # Log current state
                        logger.info(f"Scanner state: scanning={self.is_scanning}, "
                                  f"batch_size={len(self.current_scan_batch)}, "
                                  f"errors={self.consecutive_errors}, "
                                  f"total_scans={self.scan_count}")
                        
                        if restart_count <= max_restarts:
                            # Attempt restart
                            logger.info("🔄 Restarting scanner...")
                            await self._restart_scanner()
                            
                            # Reset consecutive errors after restart
                            self.consecutive_errors = 0
                        else:
                            logger.critical(f"Scanner failed after {max_restarts} restarts!")
                            self.is_scanning = False
                            break
                else:
                    # No scans yet, update last scan time to prevent false positives
                    self.last_scan_time = datetime.now()
                
                # Reset restart count if scanner is healthy
                if self.last_scan_time and (datetime.now() - self.last_scan_time).total_seconds() < 60:
                    if restart_count > 0:
                        logger.info(f"✅ Scanner recovered, resetting restart counter")
                        restart_count = 0
                
                # Log health metrics periodically
                if self.scan_count > 0 and self.scan_count % 50 == 0:  # More frequent logging
                    logger.info(f"📊 Scanner health: scans={self.scan_metrics['total_scans']}, "
                              f"success_rate={self._get_success_rate():.1f}%, "
                              f"scan_rate={self.scan_metrics['symbols_per_minute']:.1f}/min, "
                              f"signals={self.scan_metrics['signals_generated']}, "
                              f"errors={self.error_count}")
                
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
    
    async def _restart_scanner(self):
        """Restart the scanner"""
        logger.warning("Restarting scanner...")
        
        # Stop current scanning
        self.is_scanning = False
        if self.scanning_task and not self.scanning_task.done():
            self.scanning_task.cancel()
            try:
                await self.scanning_task
            except asyncio.CancelledError:
                pass
        
        # Wait a bit
        await asyncio.sleep(5)
        
        # Reset state
        self.consecutive_errors = 0
        self.last_scan_time = datetime.now()
        
        # Restart
        self.is_scanning = True
        self.scanning_task = asyncio.create_task(self._scanning_loop_with_recovery())
        
        logger.info("Scanner restarted successfully")
    
    async def _cleanup_old_data(self):
        """Aggressive cleanup for managing 558 symbols in memory"""
        try:
            current_time = datetime.now()
            symbols_to_clean = []
            
            # More aggressive cleanup thresholds for 558 symbols
            max_cached_symbols = 100  # Only keep data for 100 symbols max
            stale_threshold = 1800  # 30 minutes (reduced from 1 hour)
            
            # Count current cached symbols
            cached_count = len(self.timeframe_data)
            
            if cached_count > max_cached_symbols:
                # Sort symbols by last scan time and clean oldest
                symbol_times = []
                for symbol in self.timeframe_data.keys():
                    if symbol in self.active_positions:
                        continue  # Never clean active positions
                    
                    last_scan = None
                    if symbol in self.symbol_rotator.symbol_stats:
                        last_scan = self.symbol_rotator.symbol_stats[symbol].get('last_scan')
                    
                    if last_scan:
                        symbol_times.append((symbol, last_scan))
                    else:
                        symbols_to_clean.append(symbol)  # Clean if no scan time
                
                # Sort by last scan time (oldest first)
                symbol_times.sort(key=lambda x: x[1])
                
                # Clean oldest symbols to get under limit
                num_to_clean = cached_count - max_cached_symbols
                for symbol, _ in symbol_times[:num_to_clean]:
                    symbols_to_clean.append(symbol)
            
            # Also clean stale symbols
            for symbol in list(self.timeframe_data.keys()):
                if symbol in self.active_positions or symbol in symbols_to_clean:
                    continue
                
                if symbol in self.symbol_rotator.symbol_stats:
                    last_scan = self.symbol_rotator.symbol_stats[symbol].get('last_scan')
                    if last_scan and (current_time - last_scan).total_seconds() > stale_threshold:
                        symbols_to_clean.append(symbol)
            
            # Clean up old data
            cleaned = 0
            for symbol in set(symbols_to_clean):  # Use set to avoid duplicates
                if symbol in self.timeframe_data:
                    del self.timeframe_data[symbol]
                    cleaned += 1
                if symbol in self.htf_zones:
                    del self.htf_zones[symbol]
                if symbol in self.ltf_structure:
                    del self.ltf_structure[symbol]
            
            if cleaned > 0:
                logger.info(f"Memory cleanup: removed data for {cleaned} symbols, {len(self.timeframe_data)} remain cached")
                
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
    
    def _get_success_rate(self) -> float:
        """Calculate scanner success rate"""
        total = self.scan_metrics['successful_scans'] + self.scan_metrics['failed_scans']
        if total == 0:
            return 100.0
        return (self.scan_metrics['successful_scans'] / total) * 100
    
    def get_scanner_status(self) -> Dict:
        """Get current scanner status for monitoring"""
        time_since_last_scan = None
        if self.last_scan_time:
            time_since_last_scan = (datetime.now() - self.last_scan_time).total_seconds()
        
        return {
            'is_scanning': self.is_scanning,
            'last_scan_seconds_ago': time_since_last_scan,
            'current_batch_size': len(self.current_scan_batch),
            'active_positions': len(self.active_positions),
            'consecutive_errors': self.consecutive_errors,
            'metrics': self.scan_metrics,
            'success_rate': self._get_success_rate(),
            'healthy': self.is_scanning and time_since_last_scan and time_since_last_scan < 600  # Adjusted for slower scanning pace
        }
    
    async def _scan_symbol_once(self, symbol: str):
        """Scan a symbol once with timeout protection"""
        scan_timeout = 45  # 45 second timeout for entire scan (increased for stability)
        
        try:
            # Wrap entire scan in timeout
            await asyncio.wait_for(
                self._scan_symbol_inner(symbol),
                timeout=scan_timeout
            )
            
            # Success - update last scan time
            self.last_scan_time = datetime.now()
            logger.debug(f"✅ Scan completed for {symbol}")
            
        except asyncio.TimeoutError:
            logger.error(f"Scan timeout for {symbol} after {scan_timeout}s")
            self.error_count += 1
            self.consecutive_errors += 1
            # Don't re-raise timeout, let scanner continue
            
        except asyncio.CancelledError:
            # Task was cancelled, this is normal during shutdown
            logger.debug(f"Scan cancelled for {symbol}")
            raise  # Re-raise to properly handle cancellation
            
        except Exception as e:
            logger.error(f"Error scanning {symbol}: {e}")
            self.error_count += 1
            raise  # Re-raise to be counted as failed scan
    
    async def _scan_symbol_inner(self, symbol: str):
        """Inner scan logic without timeout wrapper"""
        # Apply rate limiting
        await rate_limiter.acquire()
        
        # Additional small delay for stability
        await asyncio.sleep(1)  # 1 second delay between symbols
        
        # Skip if should skip
        if self.symbol_rotator.should_skip_symbol(symbol):
            return
        
        # Update timeframe data
        await self._update_timeframe_data(symbol)
        
        # Check if we already have a position (double-check with exchange)
        if symbol in self.active_positions:
            # Verify position still exists on exchange (with timeout)
            try:
                actual_positions = await asyncio.wait_for(
                    self.client.get_positions(),
                    timeout=10  # 10 second timeout (increased for stability)
                )
                has_position = any(
                    pos.get('symbol') == symbol and float(pos.get('size', 0)) > 0
                    for pos in actual_positions
                )
                
                if has_position:
                    logger.debug(f"Skipping {symbol} - position already exists")
                    return
                else:
                    # Position was closed, remove from tracking
                    logger.info(f"Position for {symbol} was closed, removing from tracking")
                    del self.active_positions[symbol]
                    
                    # Import position_safety here to avoid circular import
                    from ..utils.bot_fixes import position_safety
                    position_safety.remove_position(symbol)
                    
            except asyncio.TimeoutError:
                logger.warning(f"Timeout checking positions for {symbol}")
                return  # Skip this symbol
        
        # Look for opportunities
        logger.info(f"\n{'='*60}")
        logger.info(f"🔎 ANALYZING {symbol.upper()} FOR ZONES AND SIGNALS")
        logger.info(f"{'='*60}")
        signal = await self._analyze_symbol(symbol)
        
        if signal:
            logger.info(f"✨ SIGNAL FOUND for {symbol}!")
        else:
            logger.debug(f"No signal for {symbol}")
        
        if signal:
            logger.info(f"🚨 SIGNAL DETAILS for {symbol}: direction={signal.get('direction')}, "
                       f"entry={signal.get('entry_price')}, sl={signal.get('stop_loss')}, "
                       f"tp={signal.get('take_profit_1')}")
        
        if signal:
            # Update metrics
            self.scan_metrics['signals_generated'] += 1
            self.scan_metrics['last_signal_time'] = datetime.now()
            
            logger.info(f"🎯 SIGNAL GENERATED for {symbol}! Direction={signal.get('direction')}, Entry={signal.get('entry_price')}")
            
            # Record signal for rotator
            self.symbol_rotator.record_signal(symbol)
            
            # Process if no position
            if symbol not in self.active_positions:
                logger.info(f"📤 Processing signal for {symbol} (no existing position)")
                try:
                    await self._process_signal(symbol, signal)
                    logger.info(f"✅ Signal processed for {symbol}")
                except Exception as e:
                    logger.error(f"❌ Failed to process signal for {symbol}: {e}", exc_info=True)
            else:
                logger.warning(f"⚠️ Skipping signal for {symbol} - position already tracked in scanner: {self.active_positions.get(symbol)}")
        else:
            logger.debug(f"No signal for {symbol}")
    
    async def _update_timeframe_data(self, symbol: str):
        """Update HTF and LTF data with timeout protection"""
        
        logger.info(f"📊 Fetching data for {symbol}...")
        self.timeframe_data[symbol] = {}
        api_timeout = 20  # 20 second timeout for API calls (increased for stability)
        
        # Update HTF data for zone identification
        logger.info(f"🎯 Starting zone detection data fetch for {symbol}")
        logger.info(f"📒 HTF timeframes for zones: {self.htf_timeframes}")
        for timeframe in self.htf_timeframes:
            logger.info(f"📊 Fetching {timeframe}-minute data for {symbol} (500 candles for zone detection)")
            try:
                # Try to get from cache first (with timeout)
                try:
                    cached_data = await asyncio.wait_for(
                        self._get_cached_data(symbol, timeframe),
                        timeout=5  # 5 second timeout for cache
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"Cache timeout for {symbol} {timeframe}")
                    cached_data = None
                
                if cached_data and self._is_cache_valid(cached_data):
                    # Convert JSON string back to DataFrame
                    if isinstance(cached_data['data'], str):
                        df_cached = pd.read_json(cached_data['data'], orient='split')
                    else:
                        df_cached = cached_data['data']
                    self.timeframe_data[symbol][timeframe] = df_cached
                else:
                    # Fetch fresh data with timeout
                    try:
                        df = await asyncio.wait_for(
                            self.client.get_klines(symbol, timeframe, limit=500),
                            timeout=api_timeout
                        )
                        
                        if df is not None and not df.empty:
                            # Store in memory
                            self.timeframe_data[symbol][timeframe] = df
                            logger.info(f"✅ HTF DATA READY: Got {len(df)} candles for {symbol} {timeframe}min, last price: {df['close'].iloc[-1]:.2f}")
                            logger.info(f"🔬 This data will be used for ZONE DETECTION")
                            
                            # Cache in Redis (don't wait if it fails)
                            asyncio.create_task(self._cache_data_async(symbol, timeframe, df))
                        else:
                            logger.warning(f"❌ No HTF data returned for {symbol} {timeframe}")
                    
                    except asyncio.TimeoutError:
                        logger.error(f"❌ API timeout fetching HTF {symbol} {timeframe}")
                        continue
                    except Exception as e:
                        logger.error(f"❌ API error fetching {symbol} {timeframe}: {str(e)}")
                        continue
                
            except Exception as e:
                logger.error(f"Error updating HTF {symbol} {timeframe}: {e}")
        
        # Update LTF data for market structure
        for timeframe in self.ltf_timeframes:
            try:
                # Fetch fresh LTF data with timeout
                df = await asyncio.wait_for(
                    self.client.get_klines(symbol, timeframe, limit=200),
                    timeout=api_timeout
                )
                
                if df is not None and not df.empty:
                    self.timeframe_data[symbol][timeframe] = df
                    # Cache async (don't wait)
                    asyncio.create_task(self._cache_data_async(symbol, timeframe, df, ttl=60))
                else:
                    logger.warning(f"No LTF data returned for {symbol} {timeframe}")
            
            except asyncio.TimeoutError:
                logger.error(f"API timeout fetching LTF {symbol} {timeframe}")
                continue
                    
            except Exception as e:
                logger.error(f"Error updating LTF {symbol} {timeframe}: {e}")
    
    async def _cache_data_async(self, symbol: str, timeframe: str, df: pd.DataFrame, ttl: Optional[int] = None):
        """Cache data asynchronously to prevent blocking"""
        try:
            await self._cache_data(symbol, timeframe, df, ttl)
        except Exception as e:
            logger.debug(f"Failed to cache data for {symbol} {timeframe}: {e}")
    
    async def _analyze_symbol(self, symbol: str) -> Optional[Dict]:
        """
        Analyze symbol using HTF zones and LTF market structure
        This is the core of the multi-timeframe strategy
        """
        
        signal_debugger.log_scan_start(symbol)
        logger.info(f"🔬 Phase 1: Fetching multi-timeframe data for {symbol}")
        
        # Update data for all timeframes
        await self._update_timeframe_data(symbol)
        logger.info(f"🔬 Phase 2: Analyzing zones and signals for {symbol}")
        
        if symbol not in self.timeframe_data or not self.timeframe_data[symbol]:
            logger.warning(f"⚠️ No data available for {symbol}, trying with single timeframe")
            # Try to get at least one timeframe
            try:
                df = await self.client.get_klines(symbol, "15", limit=100)
                if df is not None and not df.empty:
                    self.timeframe_data[symbol] = {"15": df}
                    logger.info(f"✅ Got fallback 15m data for {symbol}")
                else:
                    logger.error(f"❌ Failed to get any data for {symbol}")
                    signal_debugger.log_no_signal(symbol, "No data available")
                    return None
            except Exception as e:
                logger.error(f"❌ Fallback data fetch failed for {symbol}: {e}")
                return None
        
        # FIRST: Check if strategy has any signals directly
        # This is what generates the "BUY signal generated" logs
        logger.info(f"🔍 Starting zone analysis for {symbol}")
        available_timeframes = list(self.timeframe_data.get(symbol, {}).keys())
        logger.info(f"🔍 Timeframes with data for {symbol}: {available_timeframes}")
        
        if not available_timeframes:
            logger.warning(f"⚠️ NO DATA AVAILABLE for {symbol}! Cannot detect zones. Check if HTF data fetch succeeded.")
            return None
        
        # Try HTF timeframes first, but also try any available timeframe
        timeframes_to_check = self.htf_timeframes + available_timeframes
        timeframes_to_check = list(dict.fromkeys(timeframes_to_check))  # Remove duplicates
        
        for timeframe in timeframes_to_check:
            logger.info(f"📊 Checking timeframe {timeframe} for {symbol}")
            if timeframe not in self.timeframe_data.get(symbol, {}):
                logger.debug(f"⚠️ No data for {symbol} on {timeframe}, skipping")
                continue
            
            df = self.timeframe_data[symbol][timeframe]
            if df is None or not isinstance(df, pd.DataFrame) or df.empty:
                logger.warning(f"❌ Empty dataframe for {symbol} on {timeframe}")
                continue
            
            logger.info(f"📊 Calling strategy.analyze_market for {symbol} on {timeframe}")
            logger.info(f"📊 DataFrame shape: {df.shape}, Last price: {df['close'].iloc[-1]:.4f}")
            
            # Get analysis from strategy - includes both zones AND signals
            analysis = self.strategy.analyze_market(
                symbol=symbol,
                df=df,
                timeframes=[timeframe]
            )
            
            # Log zone details
            zones = analysis.get('zones', [])
            if zones:
                logger.info(f"📊 Found {len(zones)} zones for {symbol}:")
                for i, zone in enumerate(zones[:3]):  # Log first 3 zones
                    logger.info(f"  Zone {i+1}: Type={zone.zone_type}, Score={zone.composite_score:.1f}, "
                              f"Range=[{zone.lower_bound:.4f}, {zone.upper_bound:.4f}]")
            
            logger.info(f"📊 Strategy returned: zones={len(analysis.get('zones', []))}, signals={len(analysis.get('signals', []))}")
            
            # Check for signals from strategy
            if analysis.get('signals') and len(analysis['signals']) > 0:
                logger.info(f"🎯 Found {len(analysis['signals'])} signals from strategy for {symbol}")
                # Return the first valid signal
                for signal in analysis['signals']:
                    # Add required fields if missing
                    signal['symbol'] = symbol
                    signal['timeframe'] = timeframe
                    signal['confidence'] = signal.get('confidence', 80)
                    logger.info(f"🔥 Using strategy signal for {symbol}: {signal.get('direction')} @ {signal.get('entry_price')}")
                    return signal
            else:
                logger.debug(f"No signals from strategy for {symbol} on {timeframe}")
        
        # If no direct signals from strategy, continue with HTF/LTF confluence method
        # Step 1: Get HTF supply/demand zones
        htf_zones = await self._get_htf_zones(symbol)
        signal_debugger.log_htf_zones(symbol, htf_zones)
        
        if not htf_zones:
            logger.debug(f"No HTF zones found for {symbol}")
            signal_debugger.log_no_signal(symbol, "No HTF zones")
            return None
        
        # Step 2: Analyze LTF market structure
        ltf_structure = await self._get_ltf_structure(symbol)
        signal_debugger.log_ltf_structure(symbol, ltf_structure)
        
        if not ltf_structure:
            logger.debug(f"No LTF structure available for {symbol}")
            signal_debugger.log_no_signal(symbol, "No LTF structure")
            return None
        
        # Step 3: Find confluence between HTF zones and LTF structure
        signal = await self._find_htf_ltf_confluence(symbol, htf_zones, ltf_structure)
        
        if signal:
            signal_debugger.log_signal_generated(symbol, signal)
            logger.info(f"Strong signal found for {symbol}: {signal['direction']} "
                       f"at HTF zone with LTF {signal['structure_pattern']}")
            
            # Store performance data for ML learning
            await self._update_symbol_performance(symbol, signal)
        else:
            signal_debugger.log_no_signal(symbol, "No HTF/LTF confluence")
        
        return signal
    
    async def _get_htf_zones(self, symbol: str) -> List[Dict]:
        """Get supply/demand zones from higher timeframes"""
        zones = []
        
        for timeframe in self.htf_timeframes:
            if timeframe not in self.timeframe_data.get(symbol, {}):
                continue
                
            df = self.timeframe_data[symbol][timeframe]
            if df is None or not isinstance(df, pd.DataFrame) or df.empty:
                continue
            
            # Analyze HTF for zones
            analysis = self.strategy.analyze_market(
                symbol=symbol,
                df=df,
                timeframes=[timeframe]
            )
            
            if analysis.get('zones'):
                # Add timeframe info to zones
                for zone in analysis['zones'][:5]:  # Top 5 zones
                    zone_dict = {
                        'type': zone.zone_type,
                        'upper': zone.upper_bound,
                        'lower': zone.lower_bound,
                        'strength': zone.strength_score,
                        'timeframe': timeframe,
                        'touches': zone.test_count
                    }
                    zones.append(zone_dict)
        
        # Sort by strength
        zones.sort(key=lambda x: x['strength'], reverse=True)
        return zones[:10]  # Return top 10 zones
    
    async def _get_ltf_structure(self, symbol: str) -> Optional[MarketStructure]:
        """Analyze market structure on lower timeframes"""
        
        # Use primary LTF for structure analysis
        if self.primary_ltf not in self.timeframe_data.get(symbol, {}):
            return None
            
        df = self.timeframe_data[symbol][self.primary_ltf]
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return None
        
        # Analyze market structure
        structure = self.structure_analyzer.analyze_structure(df, self.primary_ltf)
        
        # Store for later use
        self.ltf_structure[symbol] = structure
        
        return structure
    
    async def _find_htf_ltf_confluence(
        self,
        symbol: str,
        htf_zones: List[Dict],
        ltf_structure: MarketStructure
    ) -> Optional[Dict]:
        """
        Find trading opportunities where HTF zones align with LTF structure
        This is where the magic happens!
        """
        
        if not htf_zones or not ltf_structure:
            return None
        
        # Get current price
        current_price = self.timeframe_data[symbol][self.primary_ltf]['close'].iloc[-1]
        
        # Find nearby zones
        nearby_zones = self._find_nearby_zones(current_price, htf_zones)
        
        if not nearby_zones:
            return None
        
        # Check each nearby zone for LTF structure confluence
        for zone in nearby_zones:
            # Determine zone bounds
            supply_zone = (zone['lower'], zone['upper']) if zone['type'] == 'supply' else None
            demand_zone = (zone['lower'], zone['upper']) if zone['type'] == 'demand' else None
            
            # Check for structure signal at zone
            structure_signal = self.structure_analyzer.detect_entry_signal(
                structure=ltf_structure,
                current_price=current_price,
                supply_zone=supply_zone,
                demand_zone=demand_zone,
                timeframe=self.primary_ltf
            )
            
            if structure_signal and structure_signal.confidence > 45:  # Lowered from 70 for testing
                # We have confluence! HTF zone + LTF structure
                return {
                    'symbol': symbol,
                    'direction': structure_signal.direction,
                    'entry_price': current_price,
                    'entry_zone': structure_signal.entry_zone,
                    'stop_loss': structure_signal.stop_loss,
                    'htf_zone': zone,
                    'ltf_structure': ltf_structure.trend.value,
                    'structure_pattern': [p.value for p in structure_signal.pattern[-3:]],
                    'confidence': (zone['strength'] + structure_signal.confidence) / 2,
                    'timeframes': {
                        'htf': zone['timeframe'],
                        'ltf': self.primary_ltf
                    }
                }
        
        return None
    
    def _find_nearby_zones(self, current_price: float, zones: List[Dict], threshold: float = 0.03) -> List[Dict]:  # Increased from 0.02 for testing
        """Find zones within threshold distance of current price"""
        nearby = []
        
        for zone in zones:
            distance_to_zone = 0
            
            if current_price > zone['upper']:
                # Price above zone
                distance_to_zone = (current_price - zone['upper']) / current_price
            elif current_price < zone['lower']:
                # Price below zone
                distance_to_zone = (zone['lower'] - current_price) / current_price
            else:
                # Price within zone
                distance_to_zone = 0
            
            if distance_to_zone <= threshold:  # Within 3% of zone (increased for testing)
                nearby.append(zone)
        
        return nearby
    
    async def _update_symbol_performance(self, symbol: str, signal: Dict):
        """Track performance data for ML learning"""
        if symbol not in self.symbol_performance:
            self.symbol_performance[symbol] = {
                'signals': [],
                'success_rate': 0,
                'best_htf': self.primary_htf,
                'best_ltf': self.primary_ltf
            }
        
        # Add signal to history
        self.symbol_performance[symbol]['signals'].append({
            'timestamp': datetime.now(),
            'signal': signal,
            'result': None  # Will be updated after trade completes
        })
    
    async def _process_signal(self, symbol: str, signal: Dict):
        """Process a trading signal"""
        
        try:
            # FINAL CHECK: Ensure no position exists before processing
            actual_positions = await self.client.get_positions()
            has_position = any(
                pos.get('symbol') == symbol and float(pos.get('size', 0)) > 0
                for pos in actual_positions
            )
            
            if has_position:
                logger.warning(f"Aborting signal for {symbol} - position already exists on exchange")
                self.active_positions[symbol] = 'existing'
                return
            
            # Ensure signal has correct format
            signal = self._ensure_signal_format(signal)
            
            # Skip if signal was marked invalid
            if signal.get('invalid', False):
                logger.error(f"❌ Signal validation failed for {symbol} - check prices: entry={signal.get('entry_price'):.8f}, sl={signal.get('stop_loss'):.8f}, tp={signal.get('take_profit_1'):.8f}")
                return
            
            # Record that we're taking a position
            signal_type = self._determine_signal_type(signal)
            self.active_positions[symbol] = signal_type
            
            # Store signal in Redis for execution
            logger.info(f"📤 Storing signal for {symbol}: entry={signal.get('entry_price'):.8f}, sl={signal.get('stop_loss'):.8f}, tp={signal.get('take_profit_1'):.8f}")
            
            # Try to store the signal
            try:
                await self._store_signal(symbol, signal)
            except Exception as e:
                logger.error(f"Failed to store signal for {symbol}: {e}")
                # Clear position tracking on storage failure
                if symbol in self.active_positions:
                    del self.active_positions[symbol]
                    logger.info(f"Cleared position tracking for {symbol} after storage failure")
                return
            
            logger.info(
                f"Signal generated for {symbol}: "
                f"Type={signal_type}, "
                f"Confluence={signal.get('confluence_score', 0)}, "
                f"Timeframes={signal.get('confirming_timeframes', [])}"
            )
            
        except Exception as e:
            logger.error(f"Error processing signal for {symbol}: {e}")
            # Remove position tracking on error
            if symbol in self.active_positions:
                del self.active_positions[symbol]
    
    def update_position_status(self, symbol: str, status: str):
        """Update position status (called by position manager)"""
        
        if status == 'closed':
            # Position closed, can look for new opportunities
            if symbol in self.active_positions:
                del self.active_positions[symbol]
                logger.info(f"Position closed for {symbol}, can trade again")
        elif status in ['long', 'short']:
            # Position opened
            self.active_positions[symbol] = status
            logger.info(f"Position opened for {symbol}: {status}")
    
    async def sync_positions_with_exchange(self):
        """Sync position tracking with actual exchange positions"""
        try:
            actual_positions = await self.client.get_positions()
            active_symbols = set()
            
            for pos in actual_positions:
                symbol = pos.get('symbol')
                size = float(pos.get('size', 0))
                if symbol and size > 0:
                    active_symbols.add(symbol)
                    # Add to tracking if not already tracked
                    if symbol not in self.active_positions:
                        side = 'long' if pos.get('side') == 'Buy' else 'short'
                        self.active_positions[symbol] = side
                        logger.info(f"Added {symbol} to position tracking from exchange sync")
            
            # Remove symbols that no longer have positions
            symbols_to_remove = []
            for symbol in self.active_positions:
                if symbol not in active_symbols:
                    symbols_to_remove.append(symbol)
            
            for symbol in symbols_to_remove:
                del self.active_positions[symbol]
                logger.info(f"Removed {symbol} from position tracking (no position on exchange)")
            
            if symbols_to_remove or active_symbols:
                logger.info(f"Position sync complete: {len(self.active_positions)} active positions")
                
        except Exception as e:
            logger.error(f"Failed to sync positions: {e}")
    
    def _determine_signal_type(self, signal: Dict) -> str:
        """Determine signal type from direction"""
        direction = signal.get('direction', '').lower()
        if 'long' in direction or 'buy' in direction:
            return 'long'
        elif 'short' in direction or 'sell' in direction:
            return 'short'
        return 'unknown'
    
    def _ensure_signal_format(self, signal: Dict) -> Dict:
        """Ensure signal has all required fields for execution"""
        
        # First validate critical fields exist
        entry_price = signal.get('entry_price', 0)
        stop_loss = signal.get('stop_loss', 0) 
        take_profit = signal.get('take_profit_1', 0)
        
        # Validate prices are reasonable
        if entry_price <= 0:
            logger.error(f"Invalid signal - entry price is zero or negative: {entry_price}")
            signal['invalid'] = True
            return signal
            
        # Check stop loss makes sense for the direction
        direction = signal.get('direction', '').upper()
        if 'BUY' in direction or 'LONG' in direction:
            if stop_loss >= entry_price:
                logger.error(f"Invalid BUY signal - stop loss {stop_loss:.8f} >= entry {entry_price:.8f}")
                signal['invalid'] = True
                return signal
            if take_profit <= entry_price:
                logger.error(f"Invalid BUY signal - take profit {take_profit:.8f} <= entry {entry_price:.8f}")
                signal['invalid'] = True
                return signal
        elif 'SELL' in direction or 'SHORT' in direction:
            if stop_loss <= entry_price:
                logger.error(f"Invalid SELL signal - stop loss {stop_loss:.8f} <= entry {entry_price:.8f}")
                signal['invalid'] = True
                return signal
            if take_profit >= entry_price:
                logger.error(f"Invalid SELL signal - take profit {take_profit:.8f} >= entry {entry_price:.8f}")
                signal['invalid'] = True
                return signal
        
        # Check risk/reward ratio
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        if risk > 0 and reward / risk < 0.5:  # Less than 0.5:1 RR is terrible
            logger.warning(f"Poor risk/reward ratio: {reward/risk:.2f}:1 for {signal.get('symbol')}")
        
        # Add action field if missing
        if 'action' not in signal:
            if 'LONG' in direction or 'BUY' in direction:
                signal['action'] = 'BUY'
            elif 'SHORT' in direction or 'SELL' in direction:
                signal['action'] = 'SELL'
            else:
                signal['action'] = 'BUY'  # Default
        
        # Ensure position size (use a reasonable default)
        if 'position_size' not in signal or not signal['position_size']:
            # Calculate default position size with minimum notional value
            entry_price = signal.get('entry_price', 1.0)
            if entry_price > 0:
                # Ensure minimum $20 notional value (Bybit requires $5-10 minimum)
                min_notional = 20.0  # $20 minimum for safety
                min_size = min_notional / entry_price
                
                # Target $100 worth, but ensure minimum
                target_size = 100.0 / entry_price
                signal['position_size'] = max(min_size, target_size)
                
                logger.info(f"Calculated position size: {signal['position_size']:.6f} (notional: ${signal['position_size'] * entry_price:.2f})")
            else:
                signal['position_size'] = 0.01  # Safer default
        
        # CRITICAL: Add risk_amount for portfolio heat calculation
        if 'risk_amount' not in signal:
            # Calculate risk based on stop loss distance
            entry = signal.get('entry_price', 0)
            stop = signal.get('stop_loss', 0)
            size = signal['position_size']
            
            if entry > 0 and stop > 0:
                risk_per_unit = abs(entry - stop)
                signal['risk_amount'] = risk_per_unit * size * entry  # Dollar risk
            else:
                # Default to 1% of $10000 = $100 risk
                signal['risk_amount'] = 100
        
        # Ensure confidence score
        if 'confidence' not in signal:
            signal['confidence'] = signal.get('zone_score', 50.0)
        
        # Ensure all required price fields
        signal.setdefault('order_type', 'MARKET')
        signal.setdefault('time_in_force', 'GTC')
        signal.setdefault('reduce_only', False)
        signal.setdefault('close_on_trigger', False)
        
        # Add ML fields if missing
        signal.setdefault('ml_confidence', 0.5)
        signal.setdefault('ml_success_probability', 0.5)
        
        logger.info(f"Signal formatted for {signal.get('symbol')}: action={signal['action']}, size={signal['position_size']:.4f}, risk=${signal['risk_amount']:.2f}")
        
        return signal
    
    async def _get_cached_data(self, symbol: str, timeframe: str) -> Optional[Dict]:
        """Get cached data from Redis with proper error handling"""
        
        if not self.redis_client:
            return None
        
        try:
            key = f"klines:{symbol}:{timeframe}"
            data = await self.redis_client.get(key)
            
            if data:
                return json.loads(data)
        
        except asyncio.CancelledError:
            # Task cancelled, don't log as error
            logger.debug(f"Redis operation cancelled for {symbol}:{timeframe}")
            raise  # Re-raise to handle properly
            
        except (ConnectionError, TimeoutError) as e:
            logger.warning(f"Redis connection issue for {symbol}:{timeframe}: {e}")
            return None
            
        except Exception as e:
            logger.error(f"Redis get error for {symbol}:{timeframe}: {e}")
            return None
        
        return None
    
    async def _cache_data(self, symbol: str, timeframe: str, df: pd.DataFrame, ttl: Optional[int] = None):
        """Cache data in Redis with proper error handling"""
        
        if not self.redis_client:
            return
        
        try:
            key = f"klines:{symbol}:{timeframe}"
            
            # Convert DataFrame to JSON-serializable format
            data = {
                'timestamp': datetime.now().isoformat(),
                'data': df.to_json(orient='split'),
                'timeframe': timeframe,
                'symbol': symbol
            }
            
            # Cache with expiration based on timeframe or custom TTL
            expiry = ttl if ttl else self._get_cache_expiry(timeframe)
            await self.redis_client.setex(
                key,
                expiry,
                json.dumps(data)
            )
        
        except asyncio.CancelledError:
            # Task cancelled, don't log as error
            logger.debug(f"Redis cache operation cancelled for {symbol}:{timeframe}")
            raise  # Re-raise to handle properly
            
        except (ConnectionError, TimeoutError) as e:
            logger.warning(f"Redis connection issue during cache for {symbol}:{timeframe}: {e}")
            # Continue without caching
            
        except Exception as e:
            logger.error(f"Redis cache error for {symbol}:{timeframe}: {e}")
    
    def _get_cache_expiry(self, timeframe: str) -> int:
        """Get cache expiry in seconds based on timeframe"""
        
        timeframe_map = {
            "5": 300,      # 5 minutes
            "15": 900,     # 15 minutes
            "60": 3600,    # 1 hour
            "240": 14400   # 4 hours
        }
        
        return timeframe_map.get(timeframe, 900)
    
    def _is_cache_valid(self, cached_data: Dict) -> bool:
        """Check if cached data is still valid"""
        
        try:
            timestamp = datetime.fromisoformat(cached_data['timestamp'])
            age = (datetime.now() - timestamp).total_seconds()
            
            # Consider cache valid if less than timeframe duration
            timeframe = cached_data.get('timeframe', '15')
            max_age = int(timeframe) * 60  # Convert to seconds
            
            return age < max_age
            
        except Exception:
            return False
    
    async def _store_signal(self, symbol: str, signal: Dict):
        """Store signal in Redis for execution"""
        
        logger.info(f"📝 Starting signal storage for {symbol}")
        
        try:
            # Add metadata
            signal['timestamp'] = datetime.now().isoformat()
            signal['symbol'] = symbol
            
            logger.info(f"📝 Signal metadata added for {symbol}")
            
            # Store in Redis if available
            if self.redis_client:
                try:
                    key = f"signal:{symbol}"
                    # Store with short expiry (signals are time-sensitive)
                    await self.redis_client.setex(
                        key,
                        60,  # 1 minute expiry
                        json.dumps(signal, default=str)
                    )
                    logger.info(f"📝 Signal stored in Redis for {symbol}")
                except Exception as e:
                    logger.warning(f"Failed to store in Redis: {e}")
            else:
                logger.info(f"📝 No Redis client, skipping Redis storage")
            
            # ALWAYS push to signal queue (it handles Redis/memory fallback)
            logger.info(f"📝 Importing signal queue for {symbol}")
            from ..utils.signal_queue import signal_queue
            
            # Check signal queue is connected (by engine)
            logger.info(f"📝 Checking signal queue connection - Redis: {signal_queue.redis_client is not None}, Memory: {signal_queue.in_memory_queue is not None}")
            
            if not signal_queue.redis_client and not signal_queue.in_memory_queue:
                logger.warning(f"📝 Signal queue not connected! Initializing in-memory fallback...")
                # Emergency fallback - connect with in-memory queue
                await signal_queue.connect(None)  # Will create in-memory queue
                logger.info(f"📝 Emergency fallback: Memory queue created")
            
            # Push signal
            logger.info(f"📝 Pushing signal to queue for {symbol}...")
            success = await signal_queue.push(signal)
            if success:
                logger.info(f"\n" + "="*60)
                logger.info(f"🎆 SIGNAL QUEUED FOR EXECUTION!")
                logger.info(f"  Symbol: {symbol}")
                logger.info(f"  Direction: {signal.get('direction', 'UNKNOWN').upper()}")
                logger.info(f"  Entry: ${signal.get('entry_price', 0):.2f}")
                logger.info(f"  Stop Loss: ${signal.get('stop_loss', 0):.2f}")
                logger.info(f"  Target: ${signal.get('take_profit_1', 0):.2f}")
                logger.info(f"  Confidence: {signal.get('confidence', 0):.1f}%")
                logger.info("="*60 + "\n")
                self.scan_metrics['signals_queued'] = self.scan_metrics.get('signals_queued', 0) + 1
            else:
                logger.error(f"❌ Failed to queue signal for {symbol}")
            
        except Exception as e:
            logger.error(f"Error storing signal for {symbol}: {e}", exc_info=True)

# Global scanner instance
multi_timeframe_scanner = None

def initialize_scanner(bybit_client: BybitClient, strategy: AdvancedSupplyDemandStrategy):
    """Initialize the global scanner"""
    global multi_timeframe_scanner
    multi_timeframe_scanner = MultiTimeframeScanner(bybit_client, strategy)
    return multi_timeframe_scanner