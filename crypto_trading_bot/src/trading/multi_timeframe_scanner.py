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
from ..api.bybit_client import BybitClient
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
    
    def __init__(self, bybit_client: BybitClient, strategy: AdvancedSupplyDemandStrategy):
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
        self.htf_timeframes = ["240", "60"]  # Higher timeframes for zones (4H, 1H)
        self.ltf_timeframes = ["15", "5"]  # Lower timeframes for structure (15m, 5m)
        self.primary_htf = "240"  # Primary HTF for zone identification
        self.primary_ltf = "15"  # Primary LTF for entry timing
        
        # ALL available symbols for maximum learning opportunities
        self.symbols = settings.default_symbols[:300]  # All top 300 symbols
        
        # Symbol rotation for efficient scanning
        self.symbol_rotator = SymbolRotator(self.symbols, max_concurrent=20)
        self.current_scan_batch = []
        
        # Position tracking - one position per symbol
        self.active_positions = {}  # symbol -> position_type (long/short)
        
        # Performance tracking per symbol
        self.symbol_performance = {}  # symbol -> {success_rate, best_tf_combo}
        
    async def initialize(self):
        """Initialize Redis connection for caching"""
        try:
            self.redis_client = redis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            await self.redis_client.ping()
            logger.info("Redis connected for multi-timeframe caching")
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            # Continue without Redis (degraded mode)
            self.redis_client = None
    
    async def start_scanning(self):
        """Start scanning with symbol rotation for efficiency"""
        logger.info(f"Starting multi-timeframe scanner for {len(self.symbols)} symbols with rotation")
        
        # Start main scanning loop with rotation
        self.scanning_task = asyncio.create_task(self._scanning_loop())
        
        logger.info("Scanner started with symbol rotation enabled")
    
    async def _scanning_loop(self):
        """Main scanning loop that rotates through symbols"""
        while True:
            try:
                # Get next batch of symbols to scan
                batch = self.symbol_rotator.get_next_batch()
                self.current_scan_batch = batch
                
                logger.debug(f"Scanning batch of {len(batch)} symbols")
                
                # Create tasks for batch
                tasks = []
                for symbol in batch:
                    if symbol not in self.active_positions:  # Skip if we have position
                        task = asyncio.create_task(self._scan_symbol_once(symbol))
                        tasks.append(task)
                
                # Wait for batch to complete
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                
                # Brief pause between batches
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Scanning loop error: {e}")
                await asyncio.sleep(10)
    
    async def stop_scanning(self):
        """Stop all scanning tasks"""
        for task in self.scanning_tasks.values():
            task.cancel()
        
        self.scanning_tasks.clear()
        logger.info("Stopped all scanning tasks")
    
    async def _scan_symbol_once(self, symbol: str):
        """Scan a symbol once (for rotation mode)"""
        try:
            # Apply rate limiting
            await rate_limiter.acquire()
            
            # Skip if should skip
            if self.symbol_rotator.should_skip_symbol(symbol):
                return
            
            # Update timeframe data
            await self._update_timeframe_data(symbol)
            
            # Check if we already have a position (double-check with exchange)
            if symbol in self.active_positions:
                # Verify position still exists on exchange
                actual_positions = await self.client.get_positions()
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
                    position_safety.remove_position(symbol)
            
            # Look for opportunities
            signal = await self._analyze_symbol(symbol)
            
            if signal:
                # Record signal for rotator
                self.symbol_rotator.record_signal(symbol)
                
                # Process if no position
                if symbol not in self.active_positions:
                    await self._process_signal(symbol, signal)
                    
        except Exception as e:
            logger.error(f"Error scanning {symbol}: {e}")
    
    async def _update_timeframe_data(self, symbol: str):
        """Update HTF and LTF data for synchronized analysis"""
        
        self.timeframe_data[symbol] = {}
        
        # Update HTF data for zone identification
        for timeframe in self.htf_timeframes:
            try:
                # Try to get from cache first
                cached_data = await self._get_cached_data(symbol, timeframe)
                
                if cached_data and self._is_cache_valid(cached_data):
                    # Convert JSON string back to DataFrame
                    if isinstance(cached_data['data'], str):
                        df_cached = pd.read_json(cached_data['data'], orient='split')
                    else:
                        df_cached = cached_data['data']
                    self.timeframe_data[symbol][timeframe] = df_cached
                else:
                    # Fetch fresh data - more candles for HTF
                    df = await self.client.get_klines(symbol, timeframe, limit=500)
                    
                    if df is not None and not df.empty:
                        # Store in memory
                        self.timeframe_data[symbol][timeframe] = df
                        
                        # Cache in Redis
                        await self._cache_data(symbol, timeframe, df)
                    else:
                        logger.warning(f"No HTF data returned for {symbol} {timeframe}")
                
            except Exception as e:
                logger.error(f"Error updating HTF {symbol} {timeframe}: {e}")
        
        # Update LTF data for market structure
        for timeframe in self.ltf_timeframes:
            try:
                # Fetch fresh LTF data (more frequent updates needed)
                df = await self.client.get_klines(symbol, timeframe, limit=200)
                
                if df is not None and not df.empty:
                    self.timeframe_data[symbol][timeframe] = df
                    # Cache with shorter TTL for LTF
                    await self._cache_data(symbol, timeframe, df, ttl=60)
                else:
                    logger.warning(f"No LTF data returned for {symbol} {timeframe}")
                    
            except Exception as e:
                logger.error(f"Error updating LTF {symbol} {timeframe}: {e}")
    
    async def _analyze_symbol(self, symbol: str) -> Optional[Dict]:
        """
        Analyze symbol using HTF zones and LTF market structure
        This is the core of the multi-timeframe strategy
        """
        
        signal_debugger.log_scan_start(symbol)
        logger.debug(f"Analyzing {symbol} with HTF/LTF strategy...")
        
        # Update data for all timeframes
        await self._update_timeframe_data(symbol)
        
        if symbol not in self.timeframe_data:
            logger.debug(f"No data available for {symbol}")
            signal_debugger.log_no_signal(symbol, "No data available")
            return None
        
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
            
            # Record that we're taking a position
            signal_type = self._determine_signal_type(signal)
            self.active_positions[symbol] = signal_type
            
            # Store signal in Redis for execution
            await self._store_signal(symbol, signal)
            
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
    
    async def _get_cached_data(self, symbol: str, timeframe: str) -> Optional[Dict]:
        """Get cached data from Redis"""
        
        if not self.redis_client:
            return None
        
        try:
            key = f"klines:{symbol}:{timeframe}"
            data = await self.redis_client.get(key)
            
            if data:
                return json.loads(data)
            
        except Exception as e:
            logger.error(f"Redis get error: {e}")
        
        return None
    
    async def _cache_data(self, symbol: str, timeframe: str, df: pd.DataFrame, ttl: Optional[int] = None):
        """Cache data in Redis with optional TTL override"""
        
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
            
        except Exception as e:
            logger.error(f"Redis cache error: {e}")
    
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
        
        if not self.redis_client:
            return
        
        try:
            key = f"signal:{symbol}"
            
            # Add metadata
            signal['timestamp'] = datetime.now().isoformat()
            signal['symbol'] = symbol
            
            # Store with short expiry (signals are time-sensitive)
            await self.redis_client.setex(
                key,
                60,  # 1 minute expiry
                json.dumps(signal, default=str)
            )
            
            # Add to the shared signal queue used by UltraIntelligentEngine
            from ..utils.signal_queue import signal_queue
            await signal_queue.push(signal)
            
        except Exception as e:
            logger.error(f"Error storing signal: {e}")

# Global scanner instance
multi_timeframe_scanner = None

def initialize_scanner(bybit_client: BybitClient, strategy: AdvancedSupplyDemandStrategy):
    """Initialize the global scanner"""
    global multi_timeframe_scanner
    multi_timeframe_scanner = MultiTimeframeScanner(bybit_client, strategy)
    return multi_timeframe_scanner