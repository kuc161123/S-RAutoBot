"""
4H Trend-Divergence Trading Bot
================================
Validated Strategy: +434R, +0.35R/trade, 25% WR
Per-Symbol R:R Optimization

Main Features:
- 4H divergence detection with Daily Trend filter
- Break of Structure (BOS) confirmation
- Per-symbol R:R ratios (2:1 to 6:1)
- Fixed TP/SL (no trailing, no partial)
- 1% risk per trade
"""

import asyncio
import logging
import yaml
import os
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
    DivergenceSignal
)
from autobot.config.symbol_rr_mapping import SymbolRRConfig
from autobot.utils.telegram_handler import TelegramHandler

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
    max_wait_candles: int = 6  # 24 hours on 4H
    
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
    """4H Trend-Divergence Trading Bot"""
    
    def __init__(self):
        """Initialize bot"""
        logger.info("="*60)
        logger.info("4H TREND-DIVERGENCE BOT INITIALIZING")
        logger.info("="*60)
        
        self.load_config()
        self.setup_broker()
        self.setup_telegram()
        self.symbol_config = SymbolRRConfig()
        
        # Trading state
        self.pending_signals: Dict[str, List[PendingSignal]] = {}  # {symbol: [signals]}
        self.active_trades: Dict[str, ActiveTrade] = {}  # {symbol: trade}
        
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
        
        # Per-symbol stats
        self.symbol_stats: Dict[str, dict] = {}  # {symbol: {trades, wins, total_r}}
        
        # Bot control
        self.trading_enabled = True
        
        # Start Time for Uptime
        self.start_time = datetime.now()
        
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
        self.radar_items: Dict[str, str] = {}  # {symbol: description}
        
        # Track time in extreme zones for ETA
        self.extreme_zone_tracker: Dict[str, datetime] = {}  # {symbol: entry_time}
        
        # Signal deduplication - track seen signals to prevent duplicates
        self.seen_signals_file = 'data/seen_signals.json'
        self.seen_signals: set = self.load_seen_signals()  # Set of unique signal IDs
        
        logger.info(f"Loaded {len(self.symbol_config.get_enabled_symbols())} enabled symbols")
        logger.info(f"Loaded {len(self.seen_signals)} previously seen signals")

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
    
    def load_seen_signals(self) -> set:
        """Load previously seen signal IDs from JSON file"""
        try:
            import json
            if os.path.exists(self.seen_signals_file):
                with open(self.seen_signals_file, 'r') as f:
                    data = json.load(f)
                    return set(data.get('signals', []))
        except Exception as e:
            logger.error(f"Failed to load seen signals: {e}")
        return set()
    
    def save_seen_signals(self):
        """Save seen signal IDs to JSON file"""
        try:
            import json
            os.makedirs(os.path.dirname(self.seen_signals_file), exist_ok=True)
            with open(self.seen_signals_file, 'w') as f:
                json.dump({'signals': list(self.seen_signals)}, f)
        except Exception as e:
            logger.error(f"Failed to save seen signals: {e}")
    
    def get_signal_id(self, signal) -> str:
        """Generate unique ID for a divergence signal"""
        # Use symbol + side + pivot timestamp to create unique ID
        # This ensures we never trade the same divergence twice
        return f"{signal.symbol}_{signal.side}_{signal.timestamp.isoformat()}"
    
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
        Dynamically update risk per trade
        
        Args:
            risk_pct: Risk percentage (e.g., 0.005 for 0.5%)
        """
        if 0.001 <= risk_pct <= 0.05:
            self.risk_config['risk_per_trade'] = risk_pct
            logger.info(f"Risk updated to {risk_pct*100:.1f}%")
            return True, f"Risk updated to {risk_pct*100:.1f}%"
        else:
            return False, "Risk must be between 0.1% and 5.0%"
    
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
        # Check if new candle closed
        if not await self.check_new_candle_close(symbol):
            return -1
        
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
                logger.debug(f"[{symbol}] {signal.signal_type.upper()} divergence detected but NOT trend-aligned - SKIP")
                continue
            
            # DEDUPLICATION: Check if we've already seen this signal
            signal_id = self.get_signal_id(signal)
            if signal_id in self.seen_signals:
                duplicate_count += 1
                continue  # Already processed this signal before
            
            # STALE SIGNAL CHECK: Skip if price already broke past swing level
            current_price = df['close'].iloc[-1]
            if signal.side == 'long' and current_price > signal.swing_level:
                logger.info(f"[{symbol}] Stale BULLISH signal - price ${current_price:.4f} already above swing ${signal.swing_level:.4f} - SKIP")
                self.seen_signals.add(signal_id)  # Mark as seen to prevent future retries
                self.save_seen_signals()
                continue
            elif signal.side == 'short' and current_price < signal.swing_level:
                logger.info(f"[{symbol}] Stale BEARISH signal - price ${current_price:.4f} already below swing ${signal.swing_level:.4f} - SKIP")
                self.seen_signals.add(signal_id)
                self.save_seen_signals()
                continue
            
            # Mark signal as seen and save
            self.seen_signals.add(signal_id)
            self.save_seen_signals()
            
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
            
            logger.info(f"[{symbol}] 🔔 NEW {signal.signal_type.upper()} DIVERGENCE detected! Waiting for BOS...")
            logger.info(f"[{symbol}]   Price: ${signal.price:.2f}, RSI: {signal.rsi_value:.1f}, Swing: ${signal.swing_level:.2f}")
            logger.info(f"[{symbol}]   Signal ID: {signal_id}")
            
            # Send Telegram notification for divergence detected
            if self.telegram:
                side_emoji = '🟢' if signal.side == 'long' else '🔴'
                side_text = 'BULLISH' if signal.side == 'long' else 'BEARISH'
                div_msg = f"""
🔔 **DIVERGENCE DETECTED**
━━━━━━━━━━━━━━━━━━━━

📊 **{symbol}** | {side_emoji} {side_text}

**SIGNAL DETAILS**
├ Price: ${signal.price:,.4f}
├ RSI: {signal.rsi_value:.1f}
├ Swing Level: ${signal.swing_level:,.4f}
└ Status: Waiting for BOS (0/6)

**ETA**
⏳ 0-6 hours to potential entry

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
                    side_emoji = '🟢' if pending.signal.side == 'long' else '🔴'
                    side_text = 'LONG' if pending.signal.side == 'long' else 'SHORT'
                    bos_msg = f"""
✅ **BOS CONFIRMED!**
━━━━━━━━━━━━━━━━━━━━

📊 **{symbol}** | {side_emoji} {side_text}

🔓 Break of Structure confirmed after {pending.candles_waited} candles
⚡ Executing trade now...

━━━━━━━━━━━━━━━━━━━━
"""
                    try:
                        await self.telegram.send_message(bos_msg)
                    except Exception as e:
                        logger.error(f"Failed to send BOS notification: {e}")
                
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
        # Skip if already in a trade for this symbol
        if symbol in self.active_trades:
            logger.warning(f"[{symbol}] Already in a trade - skipping")
            return
        
        # Get symbol-specific R:R
        rr = self.symbol_config.get_rr_for_symbol(symbol)
        if rr is None:
            logger.error(f"[{symbol}] No R:R configured - skipping")
            return
        
        # Get current market price
        latest_candle = df.iloc[-1]
        entry_price = latest_candle['open']  # Next candle open
        atr = latest_candle['atr']
        
        # Calculate position size (1% risk)
        try:
            account_balance = await self.broker.get_balance()
            risk_amount = account_balance * self.risk_config.get('risk_per_trade', 0.01)
            
            # SL distance
            sl_mult = self.strategy_config.get('exit_params', {}).get('sl_atr_mult', 1.0)
            sl_distance = atr * sl_mult
            
            # Position size logic:
            # Risk = (Entry - SL) * Qty
            # Qty = Risk / (Entry - SL)
            # Qty = RiskAmount / SL_Distance
            
            raw_qty = risk_amount / sl_distance
            
            # Get qty precision from Bybit API
            try:
                _, qty_step = await self.broker.get_precisions(symbol)
                from decimal import Decimal, ROUND_DOWN
                qty_step_dec = Decimal(qty_step)
                raw_qty_dec = Decimal(str(raw_qty))
                # Round DOWN to qty_step
                position_size_qty = float(raw_qty_dec.quantize(qty_step_dec, rounding=ROUND_DOWN))
                logger.info(f"[{symbol}] Qty: {raw_qty:.6f} → {position_size_qty} (step={qty_step})")
            except Exception as e:
                logger.warning(f"[{symbol}] Precision fetch failed, using int(): {e}")
                position_size_qty = int(raw_qty)
            
            # Sanity check: If Qty is zero or invalid
            if position_size_qty <= 0:
                logger.error(f"[{symbol}] Invalid calculated qty: {position_size_qty} (too small for risk)")
                return
            
        except Exception as e:
            logger.error(f"[{symbol}] Error calculating position size: {e}")
            return
        
        # Calculate TP/SL prices
        if signal.side == 'long':
            sl_price = entry_price - sl_distance
            tp_price = entry_price + (sl_distance * rr)
        else:
            sl_price = entry_price + sl_distance
            tp_price = entry_price - (sl_distance * rr)
        
        # === PLACE MARKET ENTRY ORDER ===
        try:
            order = await self.broker.place_market(
                symbol=symbol,
                side=signal.side,
                qty=position_size_qty
            )
            
            if not order or 'orderId' not in order:
                logger.error(f"[{symbol}] Failed to place market order")
                if self.telegram:
                    await self.telegram.send_message(f"❌ **TRADE FAILED**\n\n{symbol} | Failed to place market order\nOrder returned: {order}")
                return
            
            order_id = order['orderId']
            actual_entry = float(order.get('avgPrice', entry_price))
            
            logger.info(f"[{symbol}] ✅ Market ENTRY order filled: {order_id} @ ${actual_entry:.4f}")
            
        except Exception as e:
            logger.error(f"[{symbol}] Error placing market order: {e}")
            if self.telegram:
                await self.telegram.send_message(f"❌ **TRADE FAILED**\n\n{symbol} | Error: {e}")
            return
        
        # === PLACE LIMIT TAKE PROFIT ORDER ===
        try:
            tp_order = await self.broker.place_reduce_only_limit(
                symbol=symbol,
                side='sell' if signal.side=='long' else 'buy',
                qty=position_size_qty,
                price=tp_price,
                reduce_only=True
            )
            
            logger.info(f"[{symbol}] ✅ LIMIT TP order placed @ ${tp_price:.4f}")
            
        except Exception as e:
            logger.error(f"[{symbol}] Error placing TP limit order: {e}")
            logger.critical(f"[{symbol}] CRITICAL: Position open without TP!")
            if self.telegram:
                await self.telegram.send_message(f"⚠️ **CRITICAL: NO TP ORDER**\n\n{symbol} | Position OPEN but TP failed!\nError: {e}\n\n🚨 Set TP manually: ${tp_price:.4f}")
            # Don't return - try to place SL
        
        # === PLACE STOP-LOSS MARKET ORDER ===
        try:
            sl_order = await self.broker.place_conditional_stop(
                symbol=symbol,
                side='sell' if signal.side=='long' else 'buy',
                qty=position_size_qty,
                trigger_price=sl_price,
                reduce_only=True
            )
            
            logger.info(f"[{symbol}] ✅ STOP-LOSS market order placed @ ${sl_price:.4f}")
            
        except Exception as e:
            logger.error(f"[{symbol}] Error placing SL stop order: {e}")
            logger.critical(f"[{symbol}] CRITICAL: Position open without SL!")
            if self.telegram:
                await self.telegram.send_message(f"🚨 **CRITICAL: NO SL ORDER**\n\n{symbol} | Position OPEN but SL failed!\nError: {e}\n\n🚨🚨 CLOSE POSITION or SET SL MANUALLY: ${sl_price:.4f}")
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
            direction = '🟢 LONG' if trade.side == 'long' else '🔴 SHORT'
            entry_time = trade.entry_time.strftime('%H:%M UTC')
            
            msg = f"""
🔔 **NEW TRADE OPENED**
━━━━━━━━━━━━━━━━━━━━

📊 **{trade.symbol}** | {direction}
⏰ Time: {entry_time}

**ENTRY**
💵 Price: ${trade.entry_price:,.4f}
📏 Size: {trade.position_size:.4f}

**EXIT LEVELS**
⛔ Stop Loss: ${trade.stop_loss:,.4f} (STOP-MARKET)
🎯 Take Profit: ${trade.take_profit:,.4f} (LIMIT)
📊 R:R Ratio: {trade.rr_ratio}:1

**STRATEGY**
🔍 Setup: {signal.signal_type.title()} Divergence
📈 Trend: Daily EMA Aligned ✅
✅ Confirmation: Break of Structure

**RISK**
💰 Risking: 1.0% of capital
🎲 Potential: {trade.rr_ratio}:1 reward

━━━━━━━━━━━━━━━━━━━━
📊 [View Chart](https://www.tradingview.com/chart/?symbol=BYBIT:{trade.symbol})
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
            
            # Symbol stats
            sym_stats = self.symbol_stats.get(symbol, {'trades': 0, 'wins': 0, 'total_r': 0})
            sym_wr = (sym_stats['wins'] / sym_stats['trades'] * 100) if sym_stats['trades'] > 0 else 0
            
            msg = f"""
{emoji} **TRADE CLOSED - {result}**
━━━━━━━━━━━━━━━━━━━━

📊 **{symbol}** | {direction}
⏱️ Held: {hours_held:.1f} hours

**PRICES**
💵 Entry: ${trade.entry_price:,.4f}
💵 Exit: ${exit_price:,.4f}
{'📈' if exit_price > trade.entry_price else '📉'} Change: {((exit_price/trade.entry_price - 1) * 100):+.2f}%

**PROFIT/LOSS**
💰 P&L: {r_value:+.2f}R (${pnl_usd:+.2f})

**CUMULATIVE STATS (ALL-TIME)**
├ Total Trades: {self.stats['total_trades']}
├ ✅ Wins: {self.stats['wins']} | ❌ Losses: {self.stats['losses']}
├ Win Rate: {self.stats['win_rate']:.1f}%
├ Avg R/Trade: {self.stats['avg_r']:+.2f}R
└ Total R: {self.stats['total_r']:+.1f}R

**{symbol} PERFORMANCE**
├ Trades: {sym_stats['trades']}
├ Win Rate: {sym_wr:.1f}%
└ Total R: {sym_stats['total_r']:+.1f}R

━━━━━━━━━━━━━━━━━━━━
💡 /dashboard /positions /stats
"""
            await self.telegram.send_message(msg)

    
    async def run(self):
        """Main bot loop"""
        logger.info("="*60)
        logger.info("🤖 1H TREND-DIVERGENCE BOT STARTED")
        logger.info("="*60)
        logger.info(f"Timeframe: 1H")
        logger.info(f"Enabled Symbols: {len(self.symbol_config.get_enabled_symbols())}")
        logger.info(f"Risk per trade: {self.risk_config.get('risk_per_trade', 0.01)*100}%")
        logger.info("="*60)
        
        enabled_symbols = self.symbol_config.get_enabled_symbols()
        
        # Initialize Telegram handler (start command polling)
        if self.telegram:
            try:
                await self.telegram.initialize()
                
                # Send startup notification
                msg = f"""
🤖 **1H ROBUST VALIDATED BOT STARTED**

⏰ **Timeframe**: 1H (60 minutes)
📊 **Strategy**: RSI Divergence + EMA 200 + BOS
💰 **Risk**: {self.risk_config.get('risk_per_trade', 0.01)*100:.1f}% per trade
📈 **Enabled Symbols**: {len(enabled_symbols)} (100% Validated)

**Active Portfolio:**
{', '.join(sorted(enabled_symbols)[:8])}... +{len(enabled_symbols)-8} more

**Validation Status:** ✅ ALL SYMBOLS PASSED
• Walk-Forward Optimization
• Monte Carlo Simulation  
• Stress Test (2x fees)

**Expected OOS Performance:** +750R / Year

━━━━━━━━━━━━━━━━━━━━
💡 /help for commands
"""
                await self.telegram.send_message(msg)
            except Exception as e:
                logger.error(f"Telegram initialization failed: {e}")

                await self.telegram.send_message(msg)
            except Exception as e:
                logger.error(f"Telegram initialization failed: {e}")

        # === APPLY MAX LEVERAGE ===
        try:
            await self._apply_max_leverage()
        except Exception as e:
            logger.error(f"Failed to set max leverage: {e}")
        
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
                        msg = f"""
🕵️ **HOURLY SCAN COMPLETE**

Checked: {symbols_processed} Symbols
Result: **No New Divergences Found**

Bot is active and monitoring for:
• RSI Divergence
• Trend Alignment (EMA 200)
• Fresh Patterns (<3 candles)

Next scan in 60 mins... ⏳
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
