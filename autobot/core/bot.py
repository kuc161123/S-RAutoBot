"""
1H Multi-Divergence Trading Bot - SAFE PRECISION (235 SYMBOLS)
==============================================================
235 Symbols (Safe/Liquid) | Pivot-Based Deduplication | +9,543R (90-Day Projection)
Expected Performance: ~3,180R/Month (Based on Deduped Blind Test)

Divergence Types:
- REG_BULL: Regular Bullish (Reversal)
- REG_BEAR: Regular Bearish (Reversal)
- HID_BULL: Hidden Bullish (Continuation)
- HID_BEAR: Hidden Bearish (Continuation)

Main Features:
- 1H divergence detection with EMA 200 trend filter
- Break of Structure (BOS) confirmation (12 candles max)
- Per-symbol divergence type and R:R configuration
- Fixed TP/SL (no trailing, no partial)
- 1% risk per trade
"""

import asyncio
import logging
import yaml
import os
import time
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from autobot.brokers.bybit import Bybit, BybitConfig
from autobot.core.divergence_detector import (
    detect_divergences,
    prepare_dataframe,
    check_bos,
    is_trend_aligned,
    DivergenceSignal,
    get_signal_emoji,
    get_signal_description
)
from autobot.config.symbol_rr_mapping import SymbolRRConfig
from autobot.utils.telegram_handler import TelegramHandler
from autobot.core.storage import StorageHandler

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class PendingSignal:
    """Tracks a divergence waiting for BOS confirmation"""
    signal: DivergenceSignal
    detected_at: datetime
    candles_waited: int = 0
    max_wait_candles: int = 12  # 12 hours on 1H - increased for better BOS rate
    
    def is_expired(self) -> bool:
        return self.candles_waited >= self.max_wait_candles
    
    def increment_wait(self):
        self.candles_waited += 1


@dataclass
class ActiveTrade:
    """Represents an active trade position"""
    symbol: str
    side: str
    entry_price: float
    stop_loss: float
    take_profit: float
    rr_ratio: float
    position_size: float
    entry_time: datetime
    order_id: str = ""


class Bot4H:
    """1H Multi-Divergence Trading Bot - SAFE PRECISION (235 SYMBOLS)"""
    
    def __init__(self):
        """Initialize bot"""
        logger.info("="*60)
        logger.info("1H MULTI-DIVERGENCE BOT INITIALIZING")
        logger.info("="*60)
        
        self.load_config()
        self.setup_broker()
        self.setup_telegram()
        self.symbol_config = SymbolRRConfig()
        
        # Trading state
        self.pending_signals: Dict[str, List[PendingSignal]] = {}  # {symbol: [signals]}
        self.active_trades: Dict[str, ActiveTrade] = {}  # {symbol: trade}
        
        # [STARTUP PROTECTION] Track which symbols have been seen since startup
        # First candle for each symbol is used for initialization only (no trades)
        self.symbols_initialized: set = set()
        
        # Stats
        self.stats = {
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'total_r': 0.0,
            'win_rate': 0.0,
            'avg_r': 0.0
        }
        self.stats_file = 'stats.json'
        
        # Load stats if exists
        self.load_stats()
        
        # Lifetime Stats (persistent across all restarts)
        self.lifetime_stats_file = 'lifetime_stats.json'
        self.lifetime_stats = {
            'start_date': None,  # ISO format string
            'starting_balance': 0.0,
            'total_r': 0.0,
            'total_pnl': 0.0,
            'total_trades': 0,
            'wins': 0,
            'best_day_r': 0.0,
            'best_day_date': None,
            'worst_day_r': 0.0,
            'worst_day_date': None,
            'daily_r': {}  # {date_str: r_value}
        }
        self.load_lifetime_stats()
        
        # Per-symbol stats
        self.symbol_stats: Dict[str, dict] = {}  # {symbol: {trades, wins, total_r}}
        
        # Bot control
        self.trading_enabled = True
        
        # Start Time for Uptime
        self.start_time = datetime.now()
        
        # [BULLETPROOF] Startup Grace Period - NO trades for first 65 minutes
        # This CANNOT be bypassed regardless of candle timing
        self.startup_grace_period_minutes = 65
        logger.info(f"[STARTUP] Grace period: {self.startup_grace_period_minutes} minutes (no trades until {(self.start_time + timedelta(minutes=self.startup_grace_period_minutes)).strftime('%H:%M')})")
        
        # Candle tracking
        self.last_candle_close: Dict[str, datetime] = {}
        
        # Scan state for dashboard visibility
        self.scan_state = {
            'last_scan_time': None,
            'symbols_scanned': 0,
            'fresh_divergences': 0,
            'last_scan_signals': []  # List of symbols with fresh divergences
        }
        
        # RSI cache for dashboard
        self.rsi_cache: Dict[str, float] = {}  # {symbol: last_rsi}
        
        # Radar cache for developing setups
        
        # New startup logic: Immediate trading allowed (24h staleness filter protects us)
        logger.info("[STARTUP] Precision Mode Active. Immediate signal processing enabled.")
        
        # Signal deduplication - Persistence (DB or File)
        self.storage = StorageHandler()
        # self.seen_signals is no longer used directly, use self.storage.is_seen()
        
        # BOS Performance Tracking (for monitoring stale filter removal impact)
        self.bos_tracking = {
            'divergences_detected_today': 0,
            'bos_confirmed_today': 0,
            'divergences_detected_total': 0,
            'bos_confirmed_total': 0,
            'last_reset': datetime.now().date()
        }
        
        logger.info(f"Loaded {len(self.symbol_config.get_enabled_symbols())} enabled symbols")
        logger.info(f"Loaded {len(self.symbol_config.get_enabled_symbols())} enabled symbols")
        # Logger info for signals is now handled by StorageHandler

    def load_stats(self):
        """Load internal stats from JSON file"""
        if os.path.exists(self.stats_file):
            try:
                import json
                with open(self.stats_file, 'r') as f:
                    data = json.load(f)
                    self.stats.update(data)
                logger.info(f"Loaded stats: {self.stats['total_trades']} trades, {self.stats['total_r']:.2f}R")
            except Exception as e:
                logger.error(f"Failed to load stats: {e}")

    def save_stats(self):
        """Save internal stats to JSON file"""
        try:
            import json
            with open(self.stats_file, 'w') as f:
                json.dump(self.stats, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save stats: {e}")
    
    def load_lifetime_stats(self):
        """Load lifetime stats from JSON file"""
        import json
        if os.path.exists(self.lifetime_stats_file):
            try:
                with open(self.lifetime_stats_file, 'r') as f:
                    data = json.load(f)
                    self.lifetime_stats.update(data)
                logger.info(f"Loaded lifetime stats: {self.lifetime_stats['total_r']:.1f}R since {self.lifetime_stats.get('start_date', 'unknown')}")
            except Exception as e:
                logger.error(f"Failed to load lifetime stats: {e}")
    
    def save_lifetime_stats(self):
        """Save lifetime stats to JSON file"""
        import json
        try:
            with open(self.lifetime_stats_file, 'w') as f:
                json.dump(self.lifetime_stats, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save lifetime stats: {e}")
    
    async def initialize_lifetime_stats(self):
        """Initialize lifetime stats if this is the first run"""
        if self.lifetime_stats.get('start_date') is None:
            self.lifetime_stats['start_date'] = datetime.now().strftime('%Y-%m-%d')
            try:
                balance = await self.broker.get_balance()
                self.lifetime_stats['starting_balance'] = balance or 0.0
            except:
                self.lifetime_stats['starting_balance'] = 0.0
            self.save_lifetime_stats()
            logger.info(f"Initialized lifetime stats: Start {self.lifetime_stats['start_date']}, Balance ${self.lifetime_stats['starting_balance']:.2f}")
    
    def update_lifetime_stats(self, r_value: float, pnl: float, is_win: bool):
        """Update lifetime stats after a trade closes"""
        today = datetime.now().strftime('%Y-%m-%d')
        
        logger.info(f"[LIFETIME] Updating stats: R={r_value:+.2f}, PnL=${pnl:.2f}, Win={is_win}")
        
        # Update totals
        self.lifetime_stats['total_r'] += r_value
        self.lifetime_stats['total_pnl'] += pnl
        self.lifetime_stats['total_trades'] += 1
        if is_win:
            self.lifetime_stats['wins'] += 1
        
        # Update daily R
        if today not in self.lifetime_stats['daily_r']:
            self.lifetime_stats['daily_r'][today] = 0.0
        self.lifetime_stats['daily_r'][today] += r_value
        
        daily_r = self.lifetime_stats['daily_r'][today]
        
        # Update best/worst day
        if daily_r > self.lifetime_stats.get('best_day_r', 0):
            self.lifetime_stats['best_day_r'] = daily_r
            self.lifetime_stats['best_day_date'] = today
        if daily_r < self.lifetime_stats.get('worst_day_r', 0):
            self.lifetime_stats['worst_day_r'] = daily_r
            self.lifetime_stats['worst_day_date'] = today
        
        logger.info(f"[LIFETIME] New totals: R={self.lifetime_stats['total_r']:+.2f}, Trades={self.lifetime_stats['total_trades']}")
        
        self.save_lifetime_stats()
    
    # Legacy load/save methods removed in favor of StorageHandler
    
    def _track_divergence_detected(self):
        """Track a divergence detection"""
        # Reset daily counters if new day
        today = datetime.now().date()
        if self.bos_tracking['last_reset'] != today:
            self.bos_tracking['divergences_detected_today'] = 0
            self.bos_tracking['bos_confirmed_today'] = 0
            self.bos_tracking['last_reset'] = today
        
        self.bos_tracking['divergences_detected_today'] += 1
        self.bos_tracking['divergences_detected_total'] += 1
    
    def _track_bos_confirmed(self):
        """Track a BOS confirmation"""
        # Reset daily counters if new day
        today = datetime.now().date()
        if self.bos_tracking['last_reset'] != today:
            self.bos_tracking['divergences_detected_today'] = 0
            self.bos_tracking['bos_confirmed_today'] = 0
            self.bos_tracking['last_reset'] = today
        
        self.bos_tracking['bos_confirmed_today'] += 1
        self.bos_tracking['bos_confirmed_total'] += 1
    
    def get_signal_id(self, signal) -> str:
        """Generate unique ID using Pivot Timestamp (Backtest-Aligned Deduplication).
        
        Uses symbol + side + divergence_code + PIVOT timestamp to ensure
        multiple detections of the same setup are treated as one.
        """
        # Use pivot_timestamp for robust deduplication matching backtest
        # Fallback to timestamp if pivot_timestamp not available (legacy safety)
        ts = getattr(signal, 'pivot_timestamp', signal.timestamp)
        return f"{signal.symbol}_{signal.side}_{signal.divergence_code}_{ts.strftime('%Y%m%d_%H%M')}"
    
    def load_config(self):
        """Load configuration from config.yaml"""
        with open('config.yaml', 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.strategy_config = self.config.get('strategy', {})
        self.risk_config = self.config.get('risk', {})
        self.timeframe = self.strategy_config.get('timeframe', '240')
        
        logger.info(f"Loaded config: Timeframe={self.timeframe}, Risk={self.risk_config.get('risk_per_trade', 0.01)*100}%")
    
    def setup_broker(self):
        """Initialize Bybit broker connection"""
        bybit_config_dict = self.config.get('bybit', {})
        
        # Expand environment variables
        api_key = os.path.expandvars(bybit_config_dict.get('api_key', ''))
        api_secret = os.path.expandvars(bybit_config_dict.get('api_secret', ''))
        
        bybit_config = BybitConfig(
            api_key=api_key,
            api_secret=api_secret,
            base_url=bybit_config_dict.get('base_url', 'https://api.bybit.com')
        )
        
        self.broker = Bybit(bybit_config)
        logger.info("Broker connection initialized")
    
    async def _apply_max_leverage(self):
        """
        Set maximum allowed leverage for all enabled symbols.
        This ensures optimal margin usage.
        """
        logger.info("🔧 CONFIGURING MAX LEVERAGE for all symbols...")
        enabled_symbols = self.symbol_config.get_enabled_symbols()
        
        count = 0
        for symbol in enabled_symbols:
            try:
                # set_leverage(None) defaults to max leverage in broker
                await self.broker.set_leverage(symbol)
                count += 1
                await asyncio.sleep(0.1)  # Avoid rate limits
            except Exception as e:
                logger.warning(f"[{symbol}] Could not set leverage: {e}")
        
        logger.info(f"✅ Max leverage configured for {count} symbols")

    
    def set_risk_per_trade(self, risk_pct: float):
        """
        Dynamically update risk per trade (percentage)
        
        Args:
            risk_pct: Risk percentage (e.g., 0.005 for 0.5%)
        """
        if 0.001 <= risk_pct <= 0.05:
            self.risk_config['risk_per_trade'] = risk_pct
            # Clear USD risk so percentage takes effect
            self.risk_config['risk_amount_usd'] = None
            logger.info(f"Risk updated to {risk_pct*100:.2f}%")
            return True, f"Risk updated to {risk_pct*100:.2f}%"
        else:
            return False, "Risk must be between 0.1% and 5.0%"
    
    def set_risk_usd(self, amount_usd: float):
        """
        Dynamically update risk per trade (fixed USD)
        
        Args:
            amount_usd: Fixed USD amount to risk per trade
        """
        if 0.1 <= amount_usd <= 1000:
            self.risk_config['risk_amount_usd'] = amount_usd
            logger.info(f"Risk updated to ${amount_usd:.2f} per trade")
            return True, f"Risk updated to ${amount_usd:.2f} per trade"
        else:
            return False, "USD risk must be between $0.1 and $1000"
    
    async def sync_with_exchange(self):
        """
        Sync internal active_trades with actual exchange positions.
        Removes stale trades that no longer exist on Bybit.
        """
        try:
            # Get actual positions from Bybit
            positions = await self.broker.get_positions()
            logger.info(f"[SYNC] Fetched {len(positions) if positions else 0} positions from Bybit")
            
            # Log details of what we got
            if positions:
                open_count = sum(1 for pos in positions if float(pos.get('size', 0)) > 0)
                logger.info(f"[SYNC] {open_count} positions with size > 0")
                for pos in positions[:5]:  # Log first 5
                    if float(pos.get('size', 0)) > 0:
                        logger.info(f"[SYNC] Found: {pos.get('symbol')} size={pos.get('size')} side={pos.get('side')}")
            else:
                logger.warning("[SYNC] get_positions() returned None or empty list!")
            
            # Build set of symbols with actual open positions
            actual_open = set()
            for pos in positions:
                if float(pos.get('size', 0)) > 0:
                    actual_open.add(pos.get('symbol'))
            
            # Find stale trades (in our tracking but not on exchange)
            stale_trades = []
            for symbol in list(self.active_trades.keys()):
                if symbol not in actual_open:
                    stale_trades.append(symbol)
            
            # Remove stale trades and update stats with REAL PnL from exchange
            for symbol in stale_trades:
                trade = self.active_trades.pop(symbol, None)
                if trade:
                    logger.warning(f"[{symbol}] Removed stale trade from tracking (closed externally)")
                    
                    # Get ACTUAL closed PnL from Bybit instead of assuming -1R
                    try:
                        closed_pnl_records = await self.broker.get_closed_pnl(symbol, limit=1)
                        if closed_pnl_records:
                            latest = closed_pnl_records[0]
                            actual_pnl = float(latest.get('closedPnl', 0))
                            
                            # Calculate R based on risk amount
                            risk_usd = self.risk_config.get('risk_amount_usd', None)
                            if risk_usd:
                                risk_amount = float(risk_usd)
                            else:
                                balance = await self.broker.get_balance() or 1000
                                risk_amount = balance * self.risk_config.get('risk_per_trade', 0.005)
                            
                            actual_r = actual_pnl / risk_amount if risk_amount > 0 else 0
                            
                            # Update stats with real values
                            self.stats['total_trades'] += 1
                            self.stats['total_r'] += actual_r
                            
                            if actual_pnl >= 0:
                                self.stats['wins'] += 1
                                logger.info(f"[{symbol}] Closed with WIN: ${actual_pnl:.2f} ({actual_r:+.2f}R)")
                            else:
                                self.stats['losses'] += 1
                                logger.info(f"[{symbol}] Closed with LOSS: ${actual_pnl:.2f} ({actual_r:+.2f}R)")
                            
                            if symbol in self.symbol_stats:
                                self.symbol_stats[symbol]['trades'] += 1
                                self.symbol_stats[symbol]['total_r'] += actual_r
                                if actual_pnl >= 0:
                                    self.symbol_stats[symbol]['wins'] += 1
                        else:
                            # No closed PnL found - skip stats update
                            logger.warning(f"[{symbol}] No closed PnL found - skipping stats update")
                    except Exception as e:
                        logger.error(f"[{symbol}] Failed to get closed PnL: {e}")
                    
                    self.save_stats()
            
            # ADOPT EXISTING POSITIONS (On startup or manual intervention)
            adopted_count = 0
            for pos in positions:
                symbol = pos.get('symbol')
                size = float(pos.get('size', 0))
                
                if size > 0 and symbol not in self.active_trades:
                    # Found an active position on exchange that bot doesn't know about
                    side = "long" if pos.get('side') == "Buy" else "short"
                    entry_price = float(pos.get('avgPrice', 0))
                    stop_loss = float(pos.get('stopLoss', 0))
                    take_profit = float(pos.get('takeProfit', 0))
                    entry_time = datetime.now()  # We don't know exact start, assume now
                    
                    # Estimate R:R
                    rr_ratio = 0.0
                    if stop_loss > 0 and abs(entry_price - stop_loss) > 0:
                         risk = abs(entry_price - stop_loss)
                         reward = abs(take_profit - entry_price) if take_profit > 0 else 0
                         rr_ratio = round(reward / risk, 1)
                    
                    # Create trade object
                    new_trade = ActiveTrade(
                        symbol=symbol,
                        side=side,
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        rr_ratio=rr_ratio,
                        position_size=size,
                        entry_time=entry_time
                    )
                    
                    self.active_trades[symbol] = new_trade
                    adopted_count += 1
                    logger.info(f"[{symbol}] Adopted existing position: {side} {size} @ {entry_price}")
            
            if stale_trades or adopted_count > 0:
                logger.info(f"Synced with exchange: removed {len(stale_trades)} stale, adopted {adopted_count} existing trades")
            
            return len(stale_trades) + adopted_count
        except Exception as e:
            logger.error(f"Failed to sync with exchange: {e}")
            return 0
    
    def setup_telegram(self):
        """Initialize Telegram handler with commands"""
        telegram_config = self.config.get('telegram', {})
        
        bot_token = os.path.expandvars(telegram_config.get('token', ''))
        chat_id = os.path.expandvars(telegram_config.get('chat_id', ''))
        
        if bot_token and chat_id:
            self.telegram = TelegramHandler(bot_token, chat_id, self)
            logger.info("Telegram handler enabled (with commands)")
        else:
            self.telegram = None
            logger.warning("Telegram not configured - notifications disabled")
    
    async def fetch_4h_data(self, symbol: str, limit: int = 500) -> Optional[pd.DataFrame]:
        """
        Fetch candle data from Bybit (uses configured timeframe)
        
        Args:
            symbol: Trading pair
            limit: Number of candles to fetch
            
        Returns:
            DataFrame with OHLCV data or None
        """
        try:
            # Bybit klines API
            end_time = int(datetime.now().timestamp() * 1000)
            
            params = {
                'category': 'linear',
                'symbol': symbol,
                'interval': self.timeframe,
                'limit': limit,
                'end': end_time
            }
            
            response = await self.broker._request('GET', '/v5/market/kline', params)
            
            if not response or 'result' not in response:
                logger.warning(f"[{symbol}] No data returned from API")
                return None
            
            klines = response['result'].get('list', [])
            
            if not klines:
                return None
            
            # Parse into DataFrame
            df = pd.DataFrame(klines, columns=['start', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
            df['start'] = pd.to_datetime(df['start'].astype(int), unit='ms')
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            df.set_index('start', inplace=True)
            df.sort_index(inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"[{symbol}] Error fetching data: {e}")
            return None
    
    async def check_new_candle_close(self, symbol: str) -> bool:
        """
        Check if a new 1H candle has closed for a symbol
        
        Returns:
            True if new candle closed, False otherwise
        """
        now = datetime.now()
        
        # 1H candle closes at the top of every hour (00:00, 01:00, 02:00, etc.)
        current_hour = now.hour
        current_candle_close = now.replace(minute=0, second=0, microsecond=0)
        
        last_close = self.last_candle_close.get(symbol)
        
        if last_close is None or current_candle_close > last_close:
            self.last_candle_close[symbol] = current_candle_close
            return True
        
        return False
    
    async def process_symbol(self, symbol: str) -> int:
        """
        Process a single symbol for signals and trade execution
        
        Args:
            symbol: Trading pair to process
            
        Returns:
            int: Number of new signals found, or -1 if no new candle
        """
        # [REMOVED] 65-minute startup grace period
        # The new 24h signal filter makes this brute-force delay unnecessary.
        # checks passed.
        
        # Check if new candle closed
        if not await self.check_new_candle_close(symbol):
            return -1
            
        # [CRITICAL FIX] STALENESS CHECK
        # At startup, check_new_candle_close will return True for the last closed candle
        # even if it closed 59 minutes ago. We must SKIP processing if it's stale.
        now = datetime.now()
        current_candle_close = now.replace(minute=0, second=0, microsecond=0)
        minutes_since_close = (now - current_candle_close).total_seconds() / 60
        
        if minutes_since_close > 5:
            # If we are more than 5 minutes past the hour, this is likely a startup
            # catch-up event. Skip processing to avoid "Startup Shock" mass entry.
            logger.info(f"[{symbol}] Skipping stale candle (Closed {minutes_since_close:.0f}m ago) to prevent startup shock 🛡️")
            return 0
        
        # [REMOVED] First-candle skip
        # We now trust the 24h stale filter to prevent processing historical signals.
        # Immediate trading is now allowed.
        self.symbols_initialized.add(symbol)
        
        # [BOT-BACKTEST ALIGNMENT - CHANGE 1] Skip divergence detection if already in trade
        # This matches backtest behavior: one trade per symbol at a time
        if symbol in self.active_trades:
            logger.debug(f"[{symbol}] Already in trade - skipping divergence detection")
            # Still check pending BOS for existing signals (rare edge case)
            df = await self.fetch_4h_data(symbol)
            if df is not None and len(df) >= 100:
                df = prepare_dataframe(df)
                await self.check_pending_bos(symbol, df)
            return 0
        
        logger.info(f"[{symbol}] New 1H candle closed - processing...")
        
        # Fetch data
        df = await self.fetch_4h_data(symbol)
        if df is None or len(df) < 100:
            logger.warning(f"[{symbol}] Insufficient data")
            return 0
        
        # Prepare indicators
        df = prepare_dataframe(df)
        
        # Cache latest RSI
        if 'rsi' in df.columns and len(df) > 0:
            last_rsi = df['rsi'].iloc[-1]
            self.rsi_cache[symbol] = last_rsi
            
            # === RADAR DETECTION (Developing Patterns) ===
            # specialized logic to see if we are 'close' to a divergence
            try:
                # 1. Get recent price extremes (last 20 candles)
                last_20 = df.iloc[-20:]
                low_20 = last_20['low'].min()
                high_20 = last_20['high'].max()
                current_close = df['close'].iloc[-1]
                
                # 2. Get trend
                ema = df['daily_ema'].iloc[-1] if 'daily_ema' in df.columns else 0
                ema_distance_pct = ((current_close - ema) / ema * 100) if ema > 0 else 0
                
                # 3. Look for previous pivot RSI to calculate divergence strength
                from autobot.core.divergence_detector import find_pivots
                close_arr = df['close'].values
                rsi_arr = df['rsi'].values
                _, price_lows = find_pivots(close_arr, 3, 3)
                price_highs, _ = find_pivots(close_arr, 3, 3)
                
                prev_pivot_rsi = None
                pivot_distance = 0
                
                # 4. Check Bullish Setup Forming
                # Price is near recent low (within 1%) BUT RSI is rising/higher
                if current_close < low_20 * 1.01 and current_close > ema:
                    # Find last pivot low RSI
                    # Match the main divergence detector logic: look back from -4 (after PIVOT_RIGHT=3)
                    for i in range(len(df) - 4, max(0, len(df) - 50), -1):
                        if not pd.isna(price_lows[i]):
                            prev_pivot_rsi = rsi_arr[i]
                            pivot_distance = len(df) - 1 - i
                            break
                    
                    rsi_divergence = last_rsi - prev_pivot_rsi if prev_pivot_rsi else 0
                    
                    # CRITICAL FIX: Only show as bullish if RSI is HIGHER than previous pivot
                    # Negative divergence = NOT a bullish divergence, don't show
                    if rsi_divergence > 0:
                        candles_since_potential = 0
                        pivot_progress = min(6, candles_since_potential + 3)
                        
                        self.radar_items[symbol] = {
                            'type': 'bullish_setup',
                            'price': current_close,
                            'rsi': last_rsi,
                            'prev_pivot_rsi': prev_pivot_rsi or 0,
                            'rsi_div': rsi_divergence,
                            'ema_dist': ema_distance_pct,
                            'pivot_progress': pivot_progress,
                            'pivot_distance': pivot_distance
                        }
                    elif symbol in self.radar_items and self.radar_items[symbol].get('type') == 'bullish_setup':
                        # Remove if no longer valid
                        del self.radar_items[symbol]
                
                # 5. Check Bearish Setup Forming
                # Price is near recent high (within 1%) BUT RSI is falling/lower
                elif current_close > high_20 * 0.99 and current_close < ema:
                    # Find last pivot high RSI
                    for i in range(len(df) - 4, max(0, len(df) - 50), -1):
                        if not pd.isna(price_highs[i]):
                            prev_pivot_rsi = rsi_arr[i]
                            pivot_distance = len(df) - 1 - i
                            break
                    
                    rsi_divergence = prev_pivot_rsi - last_rsi if prev_pivot_rsi else 0
                    
                    # CRITICAL FIX: Only show as bearish if RSI is LOWER than previous pivot
                    # Negative divergence = NOT a bearish divergence, don't show
                    if rsi_divergence > 0:
                        candles_since_potential = 0
                        pivot_progress = min(6, candles_since_potential + 3)
                        
                        self.radar_items[symbol] = {
                            'type': 'bearish_setup',
                            'price': current_close,
                            'rsi': last_rsi,
                            'prev_pivot_rsi': prev_pivot_rsi or 0,
                            'rsi_div': rsi_divergence,
                            'ema_dist': ema_distance_pct,
                            'pivot_progress': pivot_progress,
                            'pivot_distance': pivot_distance
                        }
                    elif symbol in self.radar_items and self.radar_items[symbol].get('type') == 'bearish_setup':
                        # Remove if no longer valid
                        del self.radar_items[symbol]
                
                # 6. RSI Extremes (Hot) - Track time in zone
                elif last_rsi <= 25:
                    # Track when it entered extreme zone
                    if symbol not in self.extreme_zone_tracker:
                        self.extreme_zone_tracker[symbol] = datetime.now()
                    
                    hours_in_zone = (datetime.now() - self.extreme_zone_tracker[symbol]).total_seconds() / 3600
                    
                    self.radar_items[symbol] = {
                        'type': 'extreme_oversold',
                        'price': current_close,
                        'rsi': last_rsi,
                        'hours_in_zone': hours_in_zone,
                        'ema_dist': ema_distance_pct
                    }
                    
                elif last_rsi >= 75:
                    # Track when it entered extreme zone
                    if symbol not in self.extreme_zone_tracker:
                        self.extreme_zone_tracker[symbol] = datetime.now()
                    
                    hours_in_zone = (datetime.now() - self.extreme_zone_tracker[symbol]).total_seconds() / 3600
                    
                    self.radar_items[symbol] = {
                        'type': 'extreme_overbought',
                        'price': current_close,
                        'rsi': last_rsi,
                        'hours_in_zone': hours_in_zone,
                        'ema_dist': ema_distance_pct
                    }
                
                # Remove if normal and clear tracker
                else:
                    if symbol in self.radar_items:
                        del self.radar_items[symbol]
                    if symbol in self.extreme_zone_tracker:
                        del self.extreme_zone_tracker[symbol]
                    
            except Exception as e:
                pass
        
        # 1. Detect new divergences
        new_signals = detect_divergences(df, symbol)
        valid_signals_count = 0
        duplicate_count = 0
        
        for signal in new_signals:
            # Only accept trend-aligned signals
            if not signal.daily_trend_aligned:
                logger.debug(f"[{symbol}] {signal.divergence_code} divergence detected but NOT trend-aligned - SKIP")
                continue

            # [CRITICAL FIX] IGNORE HISTORY ON RESTART
            # Filter out signals that are too old to be relevant
            # 24 hours (time needed for valid signal) + buffer
            hours_since_signal = (df.index[-1] - signal.timestamp).total_seconds() / 3600
            if hours_since_signal > 24:
                # Silently skip old signals to avoid log spam on restart
                continue
            
            # DIVERGENCE TYPE FILTER: Only accept if this divergence type is allowed for this symbol
            allowed_div = self.symbol_config.get_divergence_for_symbol(symbol)
            if allowed_div and signal.divergence_code != allowed_div:
                logger.debug(f"[{symbol}] {signal.divergence_code} detected but only {allowed_div} allowed - SKIP")
                continue
            
            # DEDUPLICATION: Check if we've already seen this signal
            signal_id = self.get_signal_id(signal)
            if self.storage.is_seen(signal_id):
                duplicate_count += 1
                continue  # Already processed
            
            # Mark signal as seen and save
            self.storage.add_signal(signal_id)
            
            # Only allow ONE pending signal per symbol at a time
            if symbol in self.pending_signals and len(self.pending_signals[symbol]) > 0:
                logger.info(f"[{symbol}] Already has pending signal - skipping additional divergence")
                continue
            
            # Add to pending signals
            pending = PendingSignal(signal=signal, detected_at=datetime.now())
            
            if symbol not in self.pending_signals:
                self.pending_signals[symbol] = []
            
            self.pending_signals[symbol].append(pending)
            valid_signals_count += 1
            
            # Track divergence detection
            self._track_divergence_detected()
            
            logger.info(f"[{symbol}] 🔔 NEW {signal.divergence_code} divergence detected! Waiting for BOS...")
            logger.info(f"[{symbol}]   Price: ${signal.price:.2f}, RSI: {signal.rsi_value:.1f}, Swing: ${signal.swing_level:.2f}")
            logger.info(f"[{symbol}]   Signal ID: {signal_id}")
            
            # Send Telegram notification for divergence detected
            if self.telegram:
                div_emoji = get_signal_emoji(signal.divergence_code)
                div_name = signal.get_display_name()
                side_text = 'LONG' if signal.side == 'long' else 'SHORT'
                div_msg = f"""
🔔 **DIVERGENCE DETECTED**
━━━━━━━━━━━━━━━━━━━━

{div_emoji} **{symbol}** | {div_name}

**SIGNAL DETAILS**
├ Type: `{signal.divergence_code}`
├ Side: {side_text}
├ Price: ${signal.price:,.4f}
├ RSI: {signal.rsi_value:.1f}
├ Swing Level: ${signal.swing_level:,.4f}
└ Status: Waiting for BOS (0/12)

**ETA**
⏳ 0-12 hours to potential entry

━━━━━━━━━━━━━━━━━━━━
📊 [Chart](https://www.tradingview.com/chart/?symbol=BYBIT:{symbol})
"""
                try:
                    await self.telegram.send_message(div_msg)
                except Exception as e:
                    logger.error(f"Failed to send divergence notification: {e}")
        
        if valid_signals_count == 0:
            logger.info(f"[{symbol}] Scan complete - No new divergences (skipped {duplicate_count} duplicates)")
            
        # 2. Check pending signals for BOS
        await self.check_pending_bos(symbol, df)
        
        # 3. Update active trades
        await self.monitor_active_trades(symbol)
        
        return valid_signals_count
    
    async def check_pending_bos(self, symbol: str, df: pd.DataFrame):
        """
        Check if any pending signals have BOS confirmation
        
        Args:
            symbol: Trading pair
            df: Latest OHLCV data
        """
        if symbol not in self.pending_signals:
            return
        
        current_idx = len(df) - 1
        signals_to_remove = []
        
        for pending in self.pending_signals[symbol]:
            # Check if expired
            if pending.is_expired():
                logger.info(f"[{symbol}] Signal expired after {pending.candles_waited} candles - removing")
                signals_to_remove.append(pending)
                continue
            
            # Check if still trend-aligned
            if not is_trend_aligned(df, pending.signal, current_idx):
                logger.info(f"[{symbol}] Signal no longer trend-aligned - removing")
                signals_to_remove.append(pending)
                continue
            
            # Check for BOS
            if check_bos(df, pending.signal, current_idx):
                logger.info(f"[{symbol}] ✅ BOS CONFIRMED! Executing trade...")
                
                # Send BOS confirmed notification
                if self.telegram:
                    div_emoji = get_signal_emoji(pending.signal.divergence_code)
                    div_name = pending.signal.get_display_name()
                    side_text = 'LONG' if pending.signal.side == 'long' else 'SHORT'
                    bos_msg = f"""
✅ **BOS CONFIRMED!**
━━━━━━━━━━━━━━━━━━━━

{div_emoji} **{symbol}** | {div_name}

🔓 Break of Structure confirmed after {pending.candles_waited} candles
├ Type: `{pending.signal.divergence_code}`
├ Side: {side_text}
⚡ Executing trade now...

━━━━━━━━━━━━━━━━━━━━
"""
                    try:
                        await self.telegram.send_message(bos_msg)
                    except Exception as e:
                         logger.error(f"Failed to send BOS notification: {e}")
                
                # Track BOS confirmation
                self._track_bos_confirmed()
                
                await self.execute_trade(symbol, pending.signal, df)
                signals_to_remove.append(pending)
            else:
                # Increment wait counter
                pending.increment_wait()
                logger.debug(f"[{symbol}] Waiting for BOS ({pending.candles_waited}/{pending.max_wait_candles})")
        
        # Remove signals
        for sig in signals_to_remove:
            self.pending_signals[symbol].remove(sig)
    
    async def execute_trade(self, symbol: str, signal: DivergenceSignal, df: pd.DataFrame):
        """
        Execute trade entry with TP/SL orders
        
        Args:
            symbol: Trading pair
            signal: Divergence signal
            df: Latest OHLCV data
        """
        # Skip if already in a trade for this symbol (Internal Check)
        if symbol in self.active_trades:
            logger.warning(f"[{symbol}] Already in a trade (internal) - skipping")
            return
            
        # [CRITICAL] Skip if already in a trade on EXCHANGE (prevents pyramiding after restart)
        try:
            # We fetch all positions to be sure. 
            # Optimization: could cache this or check specific symbol if API supports it,
            # but get_positions() is robust now.
            start_check = time.time()
            existing_positions = await self.broker.get_positions()
            for p in existing_positions:
                if p.get('symbol') == symbol and float(p.get('size', 0)) > 0:
                    logger.warning(f"[{symbol}] Position already exists on EXCHANGE! Skipping to prevent sizingup/pyramiding.")
                    # Sync internal state while we are at it
                    self.active_trades.add(symbol) 
                    return
            logger.info(f"[{symbol}] No existing position found (took {time.time()-start_check:.2f}s). Proceeding...")
        except Exception as e:
            logger.error(f"[{symbol}] Failed to check existing positions: {e}")
            return # Safety fail-close
        
        # Get symbol-specific R:R
        rr = self.symbol_config.get_rr_for_symbol(symbol)
        if rr is None:
            logger.error(f"[{symbol}] No R:R configured - skipping")
            return
        
        # Get CURRENT market price from ticker for accurate SL/TP calculation
        # This prevents "StopLoss should lower than base_price" errors when market moves
        try:
            ticker = await self.broker.get_ticker(symbol)
            entry_price = float(ticker.get('lastPrice', df.iloc[-1]['close']))
            logger.info(f"[{symbol}] Using current ticker price: ${entry_price:.6f}")
        except Exception as e:
            logger.warning(f"[{symbol}] Failed to get ticker, using candle close: {e}")
            entry_price = df.iloc[-1]['close']
        
        atr = df.iloc[-1]['atr']
        
        # SL distance - use PER-SYMBOL atr_mult from config (critical for performance)
        symbol_cfg = self.symbol_config.get_symbol_config(symbol)
        sl_mult = symbol_cfg.get('atr_mult', self.strategy_config.get('exit_params', {}).get('sl_atr_mult', 1.0))
        sl_distance = atr * sl_mult
        logger.info(f"[{symbol}] Using ATR mult: {sl_mult}x → SL distance: {sl_distance:.6f}")
        
        # Calculate position size based on RISK (Safety First)
        # Goal: If SL hits, loss = risk_amount
        try:
            # 1. Determine Risk Amount
            risk_amount_usd = self.risk_config.get('risk_amount_usd', None)
            account_balance = await self.broker.get_balance()
            
            if risk_amount_usd:
                risk_amount = float(risk_amount_usd)
            else:
                margin_pct = self.risk_config.get('risk_per_trade', 0.002)
                risk_amount = account_balance * margin_pct
            
            # 2. Calculate Qty based on Risk
            # Qty = Risk / SL_Distance
            if sl_distance <= 0:
                logger.error(f"[{symbol}] Invalid SL distance: {sl_distance}")
                return

            raw_qty = risk_amount / sl_distance
            
            # 3. Apply Max Leverage to minimize Margin Usage
            leverage = await self.broker.get_max_leverage(symbol)
            await self.broker.set_leverage(symbol, leverage)
            
            # 4. Check Margin Requirements
            position_value = raw_qty * entry_price
            required_margin = position_value / leverage
            
            # Get available balance
            available_balance = account_balance
            try:
                positions = await self.broker.get_positions()
                if positions:
                    total_margin_used = sum(float(p.get('positionIM', 0)) for p in positions if float(p.get('size', 0)) > 0)
                    available_balance = account_balance - total_margin_used
            except:
                pass

            # Skip if insufficient margin
            if required_margin > available_balance:
                logger.warning(f"[{symbol}] Insufficient margin for trade. Need ${required_margin:.2f}, Have ${available_balance:.2f} (Risk-Based Sizing)")
                return

            # 5. Round to Precision
            try:
                _, qty_step = await self.broker._get_precisions(symbol)
                from decimal import Decimal, ROUND_DOWN
                qty_step_dec = Decimal(qty_step)
                raw_qty_dec = Decimal(str(raw_qty))
                position_size_qty = float(raw_qty_dec.quantize(qty_step_dec, rounding=ROUND_DOWN))
                logger.info(f"[{symbol}] Risk-Based Qty: {raw_qty:.6f} → {position_size_qty} (Risk: ${risk_amount:.2f}, SL Dist: {sl_distance:.4f})")
            except Exception as e:
                logger.warning(f"[{symbol}] Precision fetch failed, using int(): {e}")
                position_size_qty = int(raw_qty)
            
            # Sanity check
            if position_size_qty <= 0:
                logger.error(f"[{symbol}] Invalid calculated qty: {position_size_qty} (Risk: ${risk_amount:.2f} too small for ATR {sl_distance:.4f})")
                return
            
        except Exception as e:
            logger.error(f"[{symbol}] Error calculating position size: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return
        
        # Calculate TP/SL prices
        if signal.side == 'long':
            sl_price = entry_price - sl_distance
            tp_price = entry_price + (sl_distance * rr)
        else:
            sl_price = entry_price + sl_distance
            tp_price = entry_price - (sl_distance * rr)
        
        # === VALIDATE SL IS ON CORRECT SIDE OF ENTRY ===
        # This catches edge cases where rapid price movement makes SL invalid
        if signal.side == 'long' and sl_price >= entry_price:
            logger.error(f"[{symbol}] Invalid SL for LONG: SL ${sl_price:.6f} >= Entry ${entry_price:.6f} (ATR too small or price moved)")
            return
        if signal.side == 'short' and sl_price <= entry_price:
            logger.error(f"[{symbol}] Invalid SL for SHORT: SL ${sl_price:.6f} <= Entry ${entry_price:.6f} (ATR too small or price moved)")
            return
        
        # === ATOMIC BRACKET ORDER: Market Entry + TP/SL ===
        # TP/SL set atomically with entry - position NEVER unprotected
        try:
            response = await self.broker.place_market(
                symbol=symbol,
                side=signal.side,
                qty=position_size_qty,
                take_profit=tp_price,  # Atomic TP protection
                stop_loss=sl_price     # Atomic SL protection
            )
            
            ret_code = response.get('retCode', -1) if isinstance(response, dict) else -1
            ret_msg = response.get('retMsg', 'Unknown') if isinstance(response, dict) else str(response)
            result = response.get('result', {}) if isinstance(response, dict) else {}
            order_id = result.get('orderId')
            
            if ret_code != 0 or not order_id:
                logger.error(f"[{symbol}] Failed to place bracket order: {ret_msg}")
                if self.telegram:
                    await self.telegram.send_message(f"❌ **TRADE FAILED**\\n\\n{symbol} | {ret_msg}\\nQty: {position_size_qty}\\nTP: ${tp_price:.4f}\\nSL: ${sl_price:.4f}")
                return
            
            actual_entry = float(result.get('avgPrice') or entry_price)
            logger.info(f"[{symbol}] ✅ BRACKET ORDER FILLED: {order_id}")
            logger.info(f"[{symbol}]   Entry: ${actual_entry:.4f} | TP: ${tp_price:.4f} | SL: ${sl_price:.4f}")
            
        except Exception as e:
            logger.error(f"[{symbol}] Error placing bracket order: {e}")
            if self.telegram:
                await self.telegram.send_message(f"❌ **TRADE FAILED**\n\n{symbol} | Error: {e}\nQty: {position_size_qty}")
            return
        
        # Track active trade
        trade = ActiveTrade(
            symbol=symbol,
            side=signal.side,
            entry_price=actual_entry,
            stop_loss=sl_price,
            take_profit=tp_price,
            rr_ratio=rr,
            position_size=position_size_qty,
            entry_time=datetime.now(),
            order_id=order_id
        )
        
        self.active_trades[symbol] = trade
        
        # [BOT-BACKTEST ALIGNMENT - CHANGE 2] Clear pending signals when trade opens
        # This prevents stale signals from accumulating
        if symbol in self.pending_signals:
            self.pending_signals[symbol] = []
        
        # Update stats
        if symbol not in self.symbol_stats:
            self.symbol_stats[symbol] = {'trades': 0, 'wins': 0, 'total_r': 0.0}
        
        # Send enhanced notification
        await self.send_entry_notification(trade, signal)
        
        logger.info(f"[{symbol}] Trade opened: Entry=${actual_entry:.4f}, SL=${sl_price:.4f}, TP=${tp_price:.4f} ({rr}:1 R:R)")

    
    async def monitor_active_trades(self, symbol: str):
        """
        Monitor active trade for exits
        
        Args:
            symbol: Trading pair
        """
        if symbol not in self.active_trades:
            return
        
        trade = self.active_trades[symbol]
        
        # Check if position still open on exchange
        try:
            position = await self.broker.get_position(symbol)
            
            if position is None or position.get('size', 0) == 0:
                # Position closed
                await self.handle_trade_exit(symbol, trade)
                
        except Exception as e:
            logger.error(f"[{symbol}] Error checking position: {e}")
    
    async def handle_trade_exit(self, symbol: str, trade: ActiveTrade):
        """
        Handle trade exit and update stats
        
        Args:
            symbol: Trading pair
            trade: Active trade object
        """
        logger.info(f"[{symbol}] Trade closed - processing exit...")
        
        # Get actual exit details from exchange
        try:
            # Fetch closed P&L for this position
            closed_pnl = await self.broker.get_closed_pnl(symbol, limit=1)
            
            if closed_pnl:
                pnl_usd = float(closed_pnl[0].get('closedPnl', 0))
                exit_price = float(closed_pnl[0].get('avgExitPrice', 0))
                
                # Calculate R value
                sl_distance = abs(trade.entry_price - trade.stop_loss)
                if sl_distance > 0:
                    if trade.side == 'long':
                        price_diff = exit_price - trade.entry_price
                    else:
                        price_diff = trade.entry_price - exit_price
                    
                    r_value = price_diff / sl_distance
                else:
                    r_value = 0
                
                # Determine result
                result = 'WIN' if r_value > 0 else 'LOSS'
                
            else:
                # Fallback if no closed P&L found
                logger.warning(f"[{symbol}] No closed P&L found - using estimates")
                exit_price = trade.take_profit if trade.side == 'long' else trade.stop_loss
                r_value = trade.rr_ratio if exit_price == trade.take_profit else -1.0
                result = 'WIN' if r_value > 0 else 'LOSS'
                pnl_usd = 0
                
        except Exception as e:
            logger.error(f"[{symbol}] Error getting exit details: {e}")
            exit_price = 0
            r_value = 0
            result = 'UNKNOWN'
            pnl_usd = 0
        
        # Update stats
        self.stats['total_trades'] += 1
        self.stats['total_r'] += r_value
        
        if result == 'WIN':
            self.stats['wins'] += 1
            self.symbol_stats[symbol]['wins'] += 1
        else:
            self.stats['losses'] += 1
        
        self.symbol_stats[symbol]['trades'] += 1
        self.symbol_stats[symbol]['total_r'] += r_value
        
        # Calculate new stats
        total = self.stats['wins'] + self.stats['losses']
        if total > 0:
            self.stats['win_rate'] = (self.stats['wins'] / total) * 100
            self.stats['avg_r'] = self.stats['total_r'] / total
        
        # Save session stats
        self.save_stats()
        
        # Update lifetime stats (persistent across restarts)
        is_win = result == 'WIN'
        self.update_lifetime_stats(r_value, pnl_usd, is_win)
        
        # Calculate time held
        time_held = datetime.now() - trade.entry_time
        hours_held = time_held.total_seconds() / 3600
        
        # Send exit notification
        await self.send_exit_notification(
            symbol=symbol,
            trade=trade,
            exit_price=exit_price,
            result=result,
            r_value=r_value,
            pnl_usd=pnl_usd,
            hours_held=hours_held
        )
        
        # Remove from active
        del self.active_trades[symbol]
        
        logger.info(f"[{symbol}] Trade processed: {result}, {r_value:+.2f}R, held {hours_held:.1f}h")
    
    async def send_entry_notification(self, trade: ActiveTrade, signal: DivergenceSignal):
        """Send enhanced Telegram notification for trade entry"""
        if self.telegram:
            div_emoji = get_signal_emoji(signal.divergence_code)
            div_name = signal.get_display_name()
            direction = '🟢 LONG' if trade.side == 'long' else '🔴 SHORT'
            entry_time = trade.entry_time.strftime('%H:%M')
            
            # Calculate actual risk amount
            risk_usd = self.risk_config.get('risk_amount_usd')
            if risk_usd:
                risk_display = f"${float(risk_usd):.2f}"
            else:
                balance = await self.broker.get_balance() or 1000
                risk_pct = self.risk_config.get('risk_per_trade', 0.001)
                risk_display = f"${balance * risk_pct:.2f} ({risk_pct*100:.1f}%)"
            
            # Current active count
            active_count = len(self.active_trades)
            
            msg = f"""
🔔 **NEW TRADE OPENED**
━━━━━━━━━━━━━━━━━━━━

{div_emoji} **{trade.symbol}** | {direction}

⏰ {entry_time} | {div_name}

**ENTRY DETAILS**
├ 💵 Price: ${trade.entry_price:,.4f}
├ 📏 Size: {trade.position_size:.4f}
└ 💰 Risk: {risk_display}

**EXIT LEVELS**
├ 🎯 TP: ${trade.take_profit:,.4f} (+{trade.rr_ratio}R)
└ ⛔ SL: ${trade.stop_loss:,.4f} (-1R)

📊 Active Positions: {active_count}

━━━━━━━━━━━━━━━━━━━━
"""
            await self.telegram.send_message(msg)
        else:
            logger.info(f"[{trade.symbol}] Entry: ${trade.entry_price:.4f}, SL: ${trade.stop_loss:.4f}, TP: ${trade.take_profit:.4f}")



    def _update_stats_on_exit(self, r_value: float, is_win: bool):
        """Update and persist stats after trade exit"""
        self.stats['total_trades'] += 1
        self.stats['total_r'] += r_value
        
        if is_win:
            self.stats['wins'] += 1
        else:
            self.stats['losses'] += 1
        
        if self.stats['total_trades'] > 0:
            self.stats['win_rate'] = (self.stats['wins'] / self.stats['total_trades']) * 100
            self.stats['avg_r'] = self.stats['total_r'] / self.stats['total_trades']
            
        # Save to disk
        self.save_stats()

    async def send_exit_notification(self, symbol: str, trade: ActiveTrade, exit_price: float, 
                                     result: str, r_value: float, pnl_usd: float, hours_held: float):
        """Send enhanced exit notification"""
        if self.telegram:
            emoji = "✅" if result == "WIN" else "❌"
            direction = '🟢 LONG' if trade.side == 'long' else '🔴 SHORT'
            
            # Lifetime stats
            lifetime = self.lifetime_stats
            lifetime_r = lifetime.get('total_r', 0)
            lifetime_trades = lifetime.get('total_trades', 0)
            lifetime_wins = lifetime.get('wins', 0)
            lifetime_wr = (lifetime_wins / lifetime_trades * 100) if lifetime_trades > 0 else 0
            
            # Remaining positions
            remaining = len(self.active_trades) - 1  # -1 because this one is closing
            
            msg = f"""
{emoji} **TRADE CLOSED - {result}**
━━━━━━━━━━━━━━━━━━━━

📊 **{symbol}** | {direction}
⏱️ Held: {hours_held:.1f}h

**RESULT**
├ 💵 Entry: ${trade.entry_price:,.4f}
├ 💵 Exit: ${exit_price:,.4f}
└ {'🟢' if r_value > 0 else '🔴'} P&L: **{r_value:+.2f}R** (${pnl_usd:+.2f})

**LIFETIME STATS**
├ Total: {lifetime_r:+.1f}R ({lifetime_trades} trades)
├ Win Rate: {lifetime_wr:.1f}%
└ This Trade: {r_value:+.2f}R

📊 Remaining Positions: {max(0, remaining)}

━━━━━━━━━━━━━━━━━━━━
"""
            await self.telegram.send_message(msg)

    async def sync_positions_at_startup(self):
        """Fetch current open positions and sync internal state"""
        try:
            logger.info("[SYNC] Checking for existing positions on exchange...")
            positions = await self.broker.get_positions()
            count = 0
            for p in positions:
                if float(p.get('size', 0)) > 0:
                    sym = p.get('symbol')
                    # active_trades is a Dict, not a Set - use dict key assignment
                    # We don't have full trade info, so just mark as "synced position"
                    self.active_trades[sym] = None  # Placeholder to mark position exists
                    count += 1
                    logger.info(f"[SYNC] Found existing position: {sym}")
            logger.info(f"[SYNC] Synced {count} active positions.")
        except Exception as e:
            logger.error(f"[SYNC] Failed to sync positions: {e}")

    
    async def run(self):
        """Main bot loop"""
        if not self.broker.session:
             self.broker.session = aiohttp.ClientSession()

        # [NEW] Configure leverage at startup (SERIAL FETCH & SET)
        # This iterates enabled symbols one-by-one, gets max leverage, and SETS it on exchange
        # Ensures account is fully prepped and "Invalid Leverage" errors are handled early
        try:
            enabled_symbols = self.symbol_config.get_enabled_symbols()
            await self.broker.configure_active_symbols(enabled_symbols)
        except Exception as e:
            logger.error(f"Failed to configure leverage: {e}")

        # Start Telegram bot
        if self.telegram:
            await self.telegram.start()
            
        logger.info("============================================================")
        logger.info(f"🤖 1H TREND-DIVERGENCE BOT STARTED")
        logger.info("============================================================")
        logger.info(f"Timeframe: {self.timeframe}")
        logger.info(f"Enabled Symbols: {len(self.symbol_config.get_enabled_symbols())}")
        
        # Log risk setting clearly
        if self.risk_config.get('risk_amount_usd'):
             logger.info(f"Risk per trade: ${self.risk_config.get('risk_amount_usd')} (Fixed USD)")
        else:
             logger.info(f"Risk per trade: {self.risk_config.get('risk_per_trade', 0.01)*100}%")
             
        logger.info("============================================================")
        
        enabled_symbols = self.symbol_config.get_enabled_symbols()
        
        # Initialize Telegram handler (start command polling)
        if self.telegram:
            try:
                await self.telegram.start()
                
                # Send startup notification
                # Include lifetime stats if available
                lifetime = self.lifetime_stats
                lifetime_r = lifetime.get('total_r', 0)
                start_date = lifetime.get('start_date', 'Today')
                
                # Calculate risk display
                risk_pct = self.risk_config.get('risk_per_trade', 0.001)
                
                msg = f"""
🤖 **BOT STARTED**
━━━━━━━━━━━━━━━━━━━━

📊 **Strategy**: 1H Precision Divergence (Safe 235)
📈 **Symbols**: {len(enabled_symbols)}
💰 **Risk**: {risk_pct*100:.1f}% per trade

**LIFETIME STATS**
├ Total R: {lifetime_r:+.1f}R
└ Since: {start_date}

**EXPECTED**: ~3,180R/month

━━━━━━━━━━━━━━━━━━━━
💡 /dashboard /help
"""
                await self.telegram.send_message(msg)
            except Exception as e:
                logger.error(f"Telegram initialization failed: {e}")

        # [NEW] Sync existing positions to prevent pyramiding
        await self.sync_positions_at_startup()
        
        # Initialize lifetime stats (first run only)
        await self.initialize_lifetime_stats()
        
        # Start main loop
        last_check = 0
        
        while True:
            try:
                # Track cycle stats
                symbols_processed = 0
                total_signals_found = 0
                
                # Process each enabled symbol
                for symbol in enabled_symbols:
                    try:
                        signals = await self.process_symbol(symbol)
                        if signals >= 0:
                            symbols_processed += 1
                            total_signals_found += signals
                    except Exception as e:
                        logger.error(f"[{symbol}] Error processing: {e}")
                
                # If we completed a scan cycle (processed symbols implies candle close)
                if symbols_processed > 0:
                    # Update scan state for dashboard
                    self.scan_state['last_scan_time'] = datetime.now()
                    self.scan_state['symbols_scanned'] = symbols_processed
                    self.scan_state['fresh_divergences'] = total_signals_found
                    
                    logger.info(f"✅ Hourly Scan Complete. Processed: {symbols_processed}, New Signals: {total_signals_found}")
                    
                    if total_signals_found == 0 and self.telegram:
                        # Send 'No Divergence' summary to reassure user bot is working
                        pending_count = sum(len(sigs) for sigs in self.pending_signals.values())
                        active_count = len(self.active_trades)
                        
                        msg = f"""
🕵️ **HOURLY SCAN COMPLETE**

📊 Checked: {symbols_processed} Symbols
🔍 Result: **No New Divergences Found**

**Detection Criteria:**
• RSI Divergence (Pivot within 10 candles)
• Trend Aligned (EMA 200)
• BOS Confirmation (12 candle max wait)

**Current Status:**
├ ⏳ Pending BOS: {pending_count}
└ 📈 Active Trades: {active_count}

Next scan in ~60 mins ⏳
"""
                        await self.telegram.send_message(msg)
                
                # Sleep for 1 minute before next check
                await asyncio.sleep(60)
                
            except KeyboardInterrupt:
                logger.info("Bot stopped by user")
                break
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                await asyncio.sleep(60)


async def main():
    """Entry point"""
    bot = Bot4H()
    await bot.run()


if __name__ == "__main__":
    asyncio.run(main())
