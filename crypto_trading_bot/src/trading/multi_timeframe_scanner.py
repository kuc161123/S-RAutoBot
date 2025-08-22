"""
Multi-timeframe market scanner with Redis caching
"""
import asyncio
from typing import Dict, List, Set, Tuple, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import structlog
import redis.asyncio as redis
import json

from ..config import settings
from ..api.bybit_client import BybitClient
from ..strategy.advanced_supply_demand import AdvancedSupplyDemandStrategy
from ..utils.reliability import rate_limiter, retry_with_backoff

logger = structlog.get_logger(__name__)

class MultiTimeframeScanner:
    """Scans multiple timeframes for trading opportunities"""
    
    def __init__(self, bybit_client: BybitClient, strategy: AdvancedSupplyDemandStrategy):
        self.client = bybit_client
        self.strategy = strategy
        self.redis_client = None
        self.scanning_tasks = {}
        self.timeframe_data = {}  # symbol -> timeframe -> data
        self.zone_confluence = {}  # symbol -> zones across timeframes
        
        # Configure timeframes to monitor
        self.timeframes = settings.monitored_timeframes  # ["5", "15", "60", "240"]
        self.primary_timeframe = settings.default_timeframe  # "15"
        
        # Top 300 symbols
        self.symbols = settings.default_symbols[:300]
        
        # Position tracking - one position per symbol
        self.active_positions = {}  # symbol -> position_type (long/short)
        
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
        """Start scanning all symbols on all timeframes"""
        logger.info(f"Starting multi-timeframe scanner for {len(self.symbols)} symbols")
        
        # Create scanning tasks for each symbol
        for symbol in self.symbols:
            if symbol not in self.scanning_tasks:
                task = asyncio.create_task(self._scan_symbol(symbol))
                self.scanning_tasks[symbol] = task
        
        logger.info(f"Started {len(self.scanning_tasks)} scanning tasks")
    
    async def stop_scanning(self):
        """Stop all scanning tasks"""
        for task in self.scanning_tasks.values():
            task.cancel()
        
        self.scanning_tasks.clear()
        logger.info("Stopped all scanning tasks")
    
    @retry_with_backoff(max_attempts=3)
    async def _scan_symbol(self, symbol: str):
        """Scan a single symbol across all timeframes"""
        while True:
            try:
                # Apply rate limiting
                await rate_limiter.acquire()
                
                # Update timeframe data first
                await self._update_timeframe_data(symbol)
                
                # Check if we already have a position for this symbol
                if symbol in self.active_positions:
                    # Already have a position, skip analysis
                    pass
                else:
                    # No position, look for opportunities
                    signal = await self._analyze_symbol(symbol)
                    
                    if signal:
                        # Ensure only one position per symbol
                        if symbol not in self.active_positions:
                            await self._process_signal(symbol, signal)
                
                # Cache update interval based on timeframe
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
                await asyncio.sleep(10)
    
    async def _update_timeframe_data(self, symbol: str):
        """Update data for all timeframes of a symbol"""
        
        self.timeframe_data[symbol] = {}
        
        for timeframe in self.timeframes:
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
                    # Fetch fresh data
                    df = await self.client.get_klines(symbol, timeframe, limit=200)
                    
                    if df is not None and not df.empty:
                        # Store in memory
                        self.timeframe_data[symbol][timeframe] = df
                        
                        # Cache in Redis
                        await self._cache_data(symbol, timeframe, df)
                    else:
                        logger.warning(f"No data returned for {symbol} {timeframe}")
                
            except Exception as e:
                logger.error(f"Error updating {symbol} {timeframe}: {e}")
    
    async def _analyze_symbol(self, symbol: str) -> Optional[Dict]:
        """Analyze symbol across all timeframes for signals"""
        
        logger.debug(f"Analyzing {symbol} for trading opportunities...")
        
        # Update data for all timeframes
        await self._update_timeframe_data(symbol)
        
        if symbol not in self.timeframe_data:
            logger.debug(f"No data available for {symbol}")
            return None
        
        # Analyze each timeframe
        timeframe_analyses = {}
        zone_alignments = []
        
        for timeframe in self.timeframes:
            if timeframe not in self.timeframe_data[symbol]:
                continue
            
            df = self.timeframe_data[symbol][timeframe]
            
            # Skip if dataframe is invalid or not a DataFrame
            if df is None or not isinstance(df, pd.DataFrame) or df.empty:
                logger.debug(f"Skipping {symbol} {timeframe} - invalid or no data")
                continue
            
            # Run advanced supply/demand analysis
            analysis = self.strategy.analyze_market(
                symbol=symbol,
                df=df,
                timeframes=[timeframe]
            )
            
            timeframe_analyses[timeframe] = analysis
            
            # Collect zones for confluence
            if analysis['zones']:
                zone_alignments.extend([
                    (zone, timeframe) for zone in analysis['zones'][:3]
                ])
        
        # Log analysis summary
        if timeframe_analyses:
            signals_found = sum(1 for a in timeframe_analyses.values() if a.get('signals'))
            logger.debug(f"{symbol}: Analyzed {len(timeframe_analyses)} timeframes, {signals_found} have signals")
        
        # Check for multi-timeframe confluence
        confluence_signal = self._check_confluence(
            symbol,
            timeframe_analyses,
            zone_alignments
        )
        
        if confluence_signal:
            logger.info(f"✅ Confluence signal found for {symbol}: {confluence_signal.get('type', 'UNKNOWN')}")
            # Determine if we should go long or short
            signal_type = self._determine_signal_type(confluence_signal)
            
            # Ensure we can take this position
            if self._can_take_position(symbol, signal_type):
                logger.info(f"📊 Signal ready for {symbol}: Type={signal_type}")
                return confluence_signal
            else:
                logger.debug(f"Cannot take position for {symbol} - already have position")
        else:
            logger.debug(f"No confluence found for {symbol}")
        
        return None
    
    def _check_confluence(
        self,
        symbol: str,
        analyses: Dict,
        zone_alignments: List
    ) -> Optional[Dict]:
        """Check for confluence across timeframes"""
        
        if not analyses:
            return None
        
        # Get primary timeframe analysis
        primary_analysis = analyses.get(self.primary_timeframe)
        if not primary_analysis or not primary_analysis['signals']:
            return None
        
        primary_signal = primary_analysis['signals'][0]
        confluence_score = 0
        confirming_timeframes = [self.primary_timeframe]
        
        # Check alignment with other timeframes
        for timeframe, analysis in analyses.items():
            if timeframe == self.primary_timeframe:
                continue
            
            # Check market structure alignment
            if analysis['market_structure'] == primary_analysis['market_structure']:
                confluence_score += 10
            
            # Check order flow alignment
            if analysis['order_flow'] == primary_analysis['order_flow']:
                confluence_score += 10
            
            # Check for supporting signals
            if analysis['signals']:
                for signal in analysis['signals']:
                    if signal['type'] == primary_signal['type']:
                        confluence_score += 20
                        confirming_timeframes.append(timeframe)
                        break
        
        # Check zone alignment
        aligned_zones = self._find_aligned_zones(zone_alignments)
        if aligned_zones:
            confluence_score += 15 * len(aligned_zones)
        
        # Require minimum confluence
        if confluence_score >= 40 and len(confirming_timeframes) >= 2:
            # Enhance the signal with multi-timeframe data
            enhanced_signal = primary_signal.copy()
            enhanced_signal['confluence_score'] = confluence_score
            enhanced_signal['confirming_timeframes'] = confirming_timeframes
            enhanced_signal['aligned_zones'] = aligned_zones
            enhanced_signal['multi_timeframe'] = True
            
            return enhanced_signal
        
        return None
    
    def _find_aligned_zones(self, zone_alignments: List) -> List:
        """Find zones that align across timeframes"""
        aligned = []
        
        for i, (zone1, tf1) in enumerate(zone_alignments):
            for j, (zone2, tf2) in enumerate(zone_alignments[i+1:], i+1):
                if tf1 != tf2:
                    # Check if zones overlap
                    overlap = self._zones_overlap(zone1, zone2)
                    if overlap > 0.5:  # 50% overlap
                        aligned.append({
                            'zones': [zone1, zone2],
                            'timeframes': [tf1, tf2],
                            'overlap': overlap
                        })
        
        return aligned
    
    def _zones_overlap(self, zone1, zone2) -> float:
        """Calculate overlap percentage between two zones"""
        
        # Calculate intersection
        intersection_high = min(zone1.upper_bound, zone2.upper_bound)
        intersection_low = max(zone1.lower_bound, zone2.lower_bound)
        
        if intersection_high < intersection_low:
            return 0  # No overlap
        
        intersection_range = intersection_high - intersection_low
        
        # Calculate union
        union_high = max(zone1.upper_bound, zone2.upper_bound)
        union_low = min(zone1.lower_bound, zone2.lower_bound)
        union_range = union_high - union_low
        
        if union_range == 0:
            return 0
        
        return intersection_range / union_range
    
    def _determine_signal_type(self, signal: Dict) -> str:
        """Determine if signal is for long or short position"""
        return 'long' if signal['type'] == 'BUY' else 'short'
    
    def _can_take_position(self, symbol: str, signal_type: str) -> bool:
        """Check if we can take a position (one per symbol)"""
        
        # No existing position for this symbol
        if symbol not in self.active_positions:
            return True
        
        # Already have a position of the same type
        if self.active_positions[symbol] == signal_type:
            return False
        
        # Have opposite position - would need to close first
        # This should be handled by position manager
        return False
    
    async def _process_signal(self, symbol: str, signal: Dict):
        """Process a trading signal"""
        
        try:
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
    
    async def _cache_data(self, symbol: str, timeframe: str, df: pd.DataFrame):
        """Cache data in Redis"""
        
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
            
            # Cache with expiration based on timeframe
            expiry = self._get_cache_expiry(timeframe)
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
            
            # Also add to signal queue
            await self.redis_client.lpush(
                "signal_queue",
                json.dumps({'symbol': symbol, 'timestamp': signal['timestamp']})
            )
            
        except Exception as e:
            logger.error(f"Error storing signal: {e}")

# Global scanner instance
multi_timeframe_scanner = None

def initialize_scanner(bybit_client: BybitClient, strategy: AdvancedSupplyDemandStrategy):
    """Initialize the global scanner"""
    global multi_timeframe_scanner
    multi_timeframe_scanner = MultiTimeframeScanner(bybit_client, strategy)
    return multi_timeframe_scanner