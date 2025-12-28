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
        
        logger.info(f"Loaded {len(self.symbol_config.get_enabled_symbols())} enabled symbols")
        logger.info("Initialization complete")

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
        
        # 1. Detect new divergences
        new_signals = detect_divergences(df, symbol)
        valid_signals_count = 0
        
        for signal in new_signals:
            # Only accept trend-aligned signals
            if not signal.daily_trend_aligned:
                logger.debug(f"[{symbol}] {signal.signal_type.upper()} divergence detected but NOT trend-aligned - SKIP")
                continue
            
            # CRITICAL: Only accept FRESH signals (detected within last 3 candles)
            # This prevents processing historical signals on startup
            if (len(df) - signal.divergence_idx) > 3:
                continue
            
            # Add to pending signals
            pending = PendingSignal(signal=signal, detected_at=datetime.now())
            
            if symbol not in self.pending_signals:
                self.pending_signals[symbol] = []
            
            self.pending_signals[symbol].append(pending)
            valid_signals_count += 1
            
            logger.info(f"[{symbol}] 🔔 {signal.signal_type.upper()} DIVERGENCE detected! Waiting for BOS...")
            logger.info(f"[{symbol}]   Price: ${signal.price:.2f}, RSI: {signal.rsi_value:.1f}, Swing: ${signal.swing_level:.2f}")
        
        if valid_signals_count == 0:
            logger.info(f"[{symbol}] Scan complete - No fresh divergences found")
            
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
            
            position_size_qty = risk_amount / sl_distance
            
            # Sanity check: If Qty is infinite or NaN
            if position_size_qty <= 0 or not isinstance(position_size_qty, (int, float)):
                logger.error(f"[{symbol}] Invalid calculated qty: {position_size_qty}")
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
                return
            
            order_id = order['orderId']
            actual_entry = order.get('avgPrice', entry_price)
            
            logger.info(f"[{symbol}] ✅ Market ENTRY order filled: {order_id} @ ${actual_entry:.4f}")
            
        except Exception as e:
            logger.error(f"[{symbol}] Error placing market order: {e}")
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
🤖 **1H HYBRID STRATEGY BOT STARTED**

⏰ **Timeframe**: 1H (60 minutes)
📊 **Strategy**: Hybrid (15 Robust + Top 20 Performers)
💰 **Risk**: {self.risk_config.get('risk_per_trade', 0.01)*100:.1f}% per trade
📈 **Enabled Symbols**: {len(enabled_symbols)} (High Growth Portfolio)

**Active Portfolio (33 Symbols):**
{', '.join(sorted(enabled_symbols)[:10])}... and {len(enabled_symbols)-10} more

**Expected Performance:**
• Combined Potential: >1,000R / Year
• Robust Core: SUI, AAVE, KAITO, ICP (+155R OOS)
• Top Growth: WIF, LPT, JASMY, BANANA (+50R each)

🚀 Aggressive growth with verified robust base

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
