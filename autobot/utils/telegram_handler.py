"""
Enhanced Telegram Handler with Commands
========================================
Full command support for 4H bot:
- /dashboard - Main dashboard
- /help - Command list
- /positions - All active positions
- /stats - Performance statistics
- /stop - Emergency stop
"""

import asyncio
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
import logging

logger = logging.getLogger(__name__)


class TelegramHandler:
    """Handles Telegram bot commands and responses"""
    
    def __init__(self, bot_token: str, chat_id: str, bot_instance):
        """
        Initialize Telegram handler
        
        Args:
            bot_token: Telegram bot token
            chat_id: Chat ID for notifications
            bot_instance: Reference to main Bot4H instance
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.bot = bot_instance  # Reference to main bot
        self.app = None
        
    async def initialize(self):
        """Initialize Telegram application with increased timeouts"""
        from telegram.request import HTTPXRequest
        
        # Increased timeouts to prevent ConnectTimeout errors
        request = HTTPXRequest(
            connection_pool_size=8,
            connect_timeout=30.0,   # Default is 5
            read_timeout=30.0,      # Default is 5
            write_timeout=30.0,     # Default is 5
            pool_timeout=30.0       # Default is 1
        )
        
        self.app = (
            Application.builder()
            .token(self.bot_token)
            .request(request)
            .get_updates_request(request)
            .build()
        )
        
        # Register command handlers
        self.app.add_handler(CommandHandler("help", self.cmd_help))
        self.app.add_handler(CommandHandler("dashboard", self.cmd_dashboard))
        self.app.add_handler(CommandHandler("positions", self.cmd_positions))
        self.app.add_handler(CommandHandler("stats", self.cmd_stats))
        self.app.add_handler(CommandHandler("radar", self.cmd_radar))
        self.app.add_handler(CommandHandler("stop", self.cmd_stop))
        self.app.add_handler(CommandHandler("start", self.cmd_start))
        self.app.add_handler(CommandHandler("risk", self.cmd_risk))
        self.app.add_handler(CommandHandler("performance", self.cmd_performance))
        
        # Start polling with longer interval to avoid rate limits
        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling(
            poll_interval=2.0,      # Check every 2 seconds (default 0.0)
            timeout=20,             # Long polling timeout
            drop_pending_updates=True  # Don't process old commands on restart
        )
        
        logger.info("Telegram command handler started")
    
    async def send_message(self, message: str):
        """Send a message"""
        if self.app:
            await self.app.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode='Markdown',
                disable_web_page_preview=True
            )
    
    # === COMMAND HANDLERS ===
    
    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Help command"""
        msg = """
🤖 **4H TREND-DIVERGENCE BOT**

📊 **MONITORING**
/dashboard - Live trading dashboard
/positions - All active positions
/stats - Performance statistics
/performance - Symbol leaderboard (R values)
/radar - Full radar watch (all symbols)

⚙️ **CONTROL**
/stop -Emergency stop (halt trading)
/start - Resume trading
/help - Show this message

💡 **Strategy**: 1H RSI Divergence (Validated)
**Portfolio**: 79 Symbols, ~+2400R OOS/Year
"""
        await update.message.reply_text(msg, parse_mode='Markdown')
    
    async def cmd_dashboard(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comprehensive dashboard with Bybit-verified data"""
        try:
            import time
            from datetime import datetime
            
            # === SYNC WITH EXCHANGE FIRST ===
            # This ensures active_trades matches actual Bybit positions
            await self.bot.sync_with_exchange()
            
            # === SYSTEM INFO ===
            uptime_hrs = (datetime.now() - self.bot.start_time).total_seconds() / 3600
            if uptime_hrs < 0: uptime_hrs = 0
            enabled = len(self.bot.symbol_config.get_enabled_symbols())
            pending = sum(len(sigs) for sigs in self.bot.pending_signals.values())
            active = len(self.bot.active_trades)
            
            # === GET EXCHANGE-VERIFIED P&L ===
            try:
                balance = await self.bot.broker.get_balance()
                
                # Get closed P&L from exchange (last 100 trades)
                closed_records = await self.bot.broker.get_all_closed_pnl(limit=100)
                
                total_closed_pnl = 0
                wins_exchange = 0
                losses_exchange = 0
                win_pnl = 0
                loss_pnl = 0
                
                if closed_records:
                    for record in closed_records:
                        pnl = float(record.get('closedPnl', 0))
                        total_closed_pnl += pnl
                        
                        if pnl > 0:
                            wins_exchange += 1
                            win_pnl += pnl
                        else:
                            losses_exchange += 1
                            loss_pnl += pnl
                
                total_exchange = wins_exchange + losses_exchange
                exchange_wr = (wins_exchange / total_exchange * 100) if total_exchange > 0 else 0
                
            except Exception as e:
                logger.error(f"Error fetching exchange data: {e}")
                balance = 0
                total_closed_pnl = 0
                exchange_wr = 0
                total_exchange = 0
                wins_exchange = 0
                losses_exchange = 0
            
            # === GET UNREALIZED P&L FOR ACTIVE POSITIONS ===
            unrealized_pnl_usd = 0
            unrealized_r_total = 0
            
            if active > 0:
                try:
                    positions = await self.bot.broker.get_positions()
                    for pos in positions:
                        if float(pos.get('size', 0)) > 0:
                            unrealized = float(pos.get('unrealisedPnl', 0))
                            unrealized_pnl_usd += unrealized
                    
                    # Convert to R (approximate)
                    avg_risk_usd = balance * self.bot.risk_config.get('risk_per_trade', 0.01) if balance > 0 else 10
                    if avg_risk_usd > 0:
                        unrealized_r_total = unrealized_pnl_usd / avg_risk_usd
                except Exception as e:
                    logger.error(f"Error fetching unrealized P&L: {e}")
            
            # === INTERNAL STATS (for tracking) ===
            stats = self.bot.stats
            
            # === SCAN STATE ===
            scan = self.bot.scan_state
            last_scan = scan.get('last_scan_time')
            if last_scan:
                mins_ago = int((datetime.now() - last_scan).total_seconds() / 60)
                last_scan_str = f"{mins_ago} mins ago"
                next_scan_mins = max(0, 60 - mins_ago)
            else:
                last_scan_str = "Not yet"
                next_scan_mins = "~60"
            
            # === PENDING SIGNALS (Awaiting BOS) ===
            pending_list = []
            for sym, sigs in self.bot.pending_signals.items():
                for sig in sigs:
                    side_icon = "🟢" if sig.signal.signal_type == 'bullish' else "🔴"
                    pending_list.append(f"{side_icon} {sym} ({sig.candles_waited}/6)")
            pending_str = "\n│   ".join(pending_list[:3]) if pending_list else "None"
            
            # === RADAR (Categorized with ETA) ===
            pending_radar = []
            developing_radar = []
            extreme_radar = []
            
            # 1. Pending BOS signals (most accurate ETA)
            for sym, sigs in self.bot.pending_signals.items():
                for sig in sigs:
                    side_icon = "🟢" if sig.signal.signal_type == 'bullish' else "🔴"
                    candles_left = 6 - sig.candles_waited
                    hours_max = candles_left
                    pending_radar.append(f"│   {side_icon} {sym}: {sig.candles_waited}/6 candles → Max {hours_max}h to entry")
            
            # 2. Developing patterns and extreme zones (with rich multi-line format)
            if getattr(self.bot, 'radar_items', None):
                for sym, data in self.bot.radar_items.items():
                    if isinstance(data, dict):
                        if data['type'] == 'bullish_setup':
                            # Only warn if stretched beyond ±3% from EMA
                            ema_sign = "⚠️ stretched" if abs(data['ema_dist']) > 3 else "✓"
                            progress_bar = "▓" * data['pivot_progress'] + "░" * (6 - data['pivot_progress'])
                            rsi_trend = "⬆️" if data['rsi_div'] > 0 else "→"
                            
                            item = f"""│   {sym}: 🟢 Bullish Divergence Forming
│   ├─ Price: ${data['price']:g} (Testing 20-bar low, {data['ema_dist']:+.1f}% from EMA200 {ema_sign})
│   ├─ RSI: {data['rsi']:.0f} {rsi_trend} (Previous pivot: {data['prev_pivot_rsi']:.0f}) → {data['rsi_div']:+.0f} point divergence
│   ├─ Progress: {progress_bar} {data['pivot_progress']}/6 candles to pivot confirmation
│   └─ ETA: 3-9h to confirmed signal, then 0-6h to BOS trigger"""
                            developing_radar.append(item)
                            
                        elif data['type'] == 'bearish_setup':
                            # Only warn if stretched beyond ±3% from EMA
                            ema_sign = "⚠️ stretched" if abs(data['ema_dist']) > 3 else "✓"
                            progress_bar = "▓" * data['pivot_progress'] + "░" * (6 - data['pivot_progress'])
                            rsi_trend = "⬇️" if data['rsi_div'] > 0 else "→"
                            
                            item = f"""│   {sym}: 🔴 Bearish Divergence Forming
│   ├─ Price: ${data['price']:g} (Testing 20-bar high, {data['ema_dist']:+.1f}% from EMA200 {ema_sign})
│   ├─ RSI: {data['rsi']:.0f} {rsi_trend} (Previous pivot: {data['prev_pivot_rsi']:.0f}) → {data['rsi_div']:+.0f} point divergence
│   ├─ Progress: {progress_bar} {data['pivot_progress']}/6 candles to pivot confirmation
│   └─ ETA: 3-9h to confirmed signal, then 0-6h to BOS trigger"""
                            developing_radar.append(item)
                            
                        elif data['type'] == 'extreme_oversold':
                            ema_warn = "⚠️ stretched" if abs(data['ema_dist']) > 3 else ""
                            item = f"""│   {sym}: ❄️ Extreme Oversold Zone
│   ├─ RSI: {data['rsi']:.0f}⬇️ ({data['hours_in_zone']:.0f}h in extreme zone)
│   ├─ Price: ${data['price']:g} ({data['ema_dist']:+.1f}% from EMA {ema_warn})
│   └─ ETA: Reversal likely within 2-8h"""
                            extreme_radar.append(item)
                            
                        elif data['type'] == 'extreme_overbought':
                            ema_warn = "⚠️ stretched" if abs(data['ema_dist']) > 3 else ""
                            item = f"""│   {sym}: 🔥 Extreme Overbought Zone
│   ├─ RSI: {data['rsi']:.0f}⬇️ ({data['hours_in_zone']:.0f}h in extreme zone)
│   ├─ Price: ${data['price']:g} ({data['ema_dist']:+.1f}% from EMA {ema_warn})
│   └─ ETA: Reversal likely within 2-8h"""
                            extreme_radar.append(item)

            
            # Build strings (limit items to prevent Telegram message too long)
            pending_count = len(pending_radar)
            pending_radar_str = "\n".join(pending_radar[:5]) if pending_radar else "│   None"
            if pending_count > 5:
                pending_radar_str += f"\n│   ... and {pending_count - 5} more"
            
            developing_count = len(developing_radar)
            developing_radar_str = "\n".join(developing_radar[:3]) if developing_radar else "│   Scanning..."
            if developing_count > 3:
                developing_radar_str += f"\n│   ... and {developing_count - 3} more"
            
            extreme_count = len(extreme_radar)
            extreme_radar_str = "\n".join(extreme_radar[:3]) if extreme_radar else "│   None"
            if extreme_count > 3:
                extreme_radar_str += f"\n│   ... and {extreme_count - 3} more"
            
            # === BUILD COMPREHENSIVE MESSAGE ===
            msg = f"""
📊 **1H VALIDATED DASHBOARD**
━━━━━━━━━━━━━━━━━━━━

⏰ **SYSTEM**
├ Uptime: {uptime_hrs:.1f}h
├ Timeframe: 1H (60m)
├ Risk/Trade: {self.bot.risk_config.get('risk_per_trade', 0.01)*100}%
└ Enabled: {enabled} Symbols (Validated)

🎯 **STRATEGY**
├ Setup: RSI Divergence + EMA 200
├ Confidence: 100% Anti-Overfit
├ Risk/Reward: 4:1 to 10:1
└ Expected OOS: ~+2400R/Yr

🔍 **SCANNING STATUS**
├ Last Scan: {last_scan_str}
├ Next Scan: ~{next_scan_mins} mins
└ Seen Signals: {len(self.bot.seen_signals)} (deduped)

📡 **RADAR WATCH**
┌─ Pending BOS (Confirmed Signals):
{pending_radar_str}
├─ Developing Setups (3-9h):
{developing_radar_str}
└─ Extreme Zones (2-8h):
{extreme_radar_str}



💼 **WALLET (BYBIT)**
├ Balance: ${balance:,.2f} USDT
└ Realized P&L: ${total_closed_pnl:+,.2f}

📊 **EXCHANGE STATS**
├ Trades: {total_exchange} | WR: {exchange_wr:.1f}%
└ P&L: ${total_closed_pnl:+,.2f}

📈 **INTERNAL TRACKING**
├ Trades: {stats['total_trades']} | WR: {stats['win_rate']:.1f}%
├ Avg R: {stats['avg_r']:+.2f}R
└ Total R: {stats['total_r']:+.1f}R

🔔 **POSITIONS**
├ Pending: {pending} | Active: {active}
└ Unrealized: ${unrealized_pnl_usd:+,.2f} ({unrealized_r_total:+.1f}R)
"""
            
            # === SHOW ACTIVE POSITIONS (if any) ===
            if self.bot.active_trades:
                msg += "\n📍 **ACTIVE POSITIONS**\n\n"
                
                for symbol, trade in list(self.bot.active_trades.items())[:5]:  # Max 5
                    try:
                        # Get current price for accurate R
                        ticker = await self.bot.broker.get_ticker(symbol)
                        current_price = float(ticker.get('lastPrice', 0)) if ticker else 0
                        
                        # Calculate current R
                        sl_distance = abs(trade.entry_price - trade.stop_loss)
                        if current_price > 0 and sl_distance > 0:
                            if trade.side == 'long':
                                current_r = (current_price - trade.entry_price) / sl_distance
                            else:
                                current_r = (trade.entry_price - current_price) / sl_distance
                        else:
                            current_r = 0
                        
                        side_icon = "🟢" if trade.side == 'long' else "🔴"
                        r_status = "📈" if current_r > 0 else "📉"
                        
                        msg += f"""
├ {side_icon} `{symbol}` {trade.side.upper()}
├ Entry: ${trade.entry_price:.4f} → ${current_price:.4f}
├ {r_status} Current: {current_r:+.2f}R | Target: {trade.rr_ratio}R
└ SL: ${trade.stop_loss:.4f} | TP: ${trade.take_profit:.4f}

"""
                    except Exception as e:
                        logger.error(f"Error displaying {symbol}: {e}")
                        continue
                
                if len(self.bot.active_trades) > 5:
                    msg += f"... and {len(self.bot.active_trades) - 5} more\n\n"
            
            # === TOP PERFORMING SYMBOLS ===
            if self.bot.symbol_stats:
                top_symbols = sorted(
                    [(sym, stats) for sym, stats in self.bot.symbol_stats.items() if stats['trades'] > 0],
                    key=lambda x: x[1]['total_r'],
                    reverse=True
                )[:3]
                
                if top_symbols:
                    msg += "\n🏆 **TOP SYMBOLS (by Total R)**\n"
                    for sym, sym_stats in top_symbols:
                        sym_wr = (sym_stats['wins'] / sym_stats['trades'] * 100) if sym_stats['trades'] > 0 else 0
                        msg += f"├ `{sym}`: {sym_stats['total_r']:+.1f}R ({sym_stats['trades']} trades, {sym_wr:.0f}% WR)\n"
                    msg += "\n"
            
            msg += """
━━━━━━━━━━━━━━━━━━━━
💡 /positions /stats /help
"""
            
            await update.message.reply_text(msg, parse_mode='Markdown')
            
        except Exception as e:
            await update.message.reply_text(f"❌ Dashboard error: {e}")
            logger.error(f"Dashboard error: {e}")
            import traceback
            logger.error(traceback.format_exc())

    
    async def cmd_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show all active positions"""
        try:
            if not self.bot.active_trades:
                await update.message.reply_text("📊 No active positions.")
                return
            
            msg = f"📊 **ACTIVE POSITIONS** ({len(self.bot.active_trades)} open)\\n\\n"
            
            for symbol, trade in self.bot.active_trades.items():
                side_icon = "🟢" if trade.side == 'long' else "🔴"
                
                msg += f"""
┌─ {side_icon} {trade.side.upper()} `{symbol}`
├ Entry: ${trade.entry_price:,.2f}
├ Stop Loss: ${trade.stop_loss:,.2f}
├ Take Profit: ${trade.take_profit:,.2f}
├ R:R: {trade.rr_ratio}:1
└ Size: {trade.position_size:.4f}

"""
            
            msg += "━━━━━━━━━━━━━━━━━━━━\\n"
            msg += "💡 /dashboard /stats"
            
            await update.message.reply_text(msg, parse_mode='Markdown')
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")
            logger.error(f"Positions error: {e}")
    
    async def cmd_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Performance statistics"""
        try:
            stats = self.bot.stats
            
            # Calculate per-symbol performance
            symbol_performance = {}
            # This would need to be tracked in the bot
            
            msg = f"""
📊 **PERFORMANCE STATISTICS**
━━━━━━━━━━━━━━━━━━━━

📈 **OVERALL**
├ Total Trades: {stats['total_trades']}
├ Wins: {stats['wins']} (✅)
├ Losses: {stats['losses']} (❌)
├ Win Rate: {stats['win_rate']:.1f}%
├ Avg R/Trade: {stats['avg_r']:+.2f}R
└ Total R: {stats['total_r']:+.1f}R

🎯 **VS BACKTEST**
├ Expected WR: 25%
├ Actual WR: {stats['win_rate']:.1f}%
├ Expected R/Trade: +0.35R
├ Actual R/Trade: {stats['avg_r']:+.2f}R
└ Delta: {stats['avg_r'] - 0.35:+.2f}R

━━━━━━━━━━━━━━━━━━━━
💡 /dashboard /positions
"""
            await update.message.reply_text(msg, parse_mode='Markdown')
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")
    
    async def cmd_performance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Symbol performance leaderboard"""
        try:
            symbol_stats = self.bot.symbol_stats
            
            if not symbol_stats:
                await update.message.reply_text("📊 No trades recorded yet.")
                return
            
            sorted_symbols = sorted(
                [(sym, data) for sym, data in symbol_stats.items() if data.get('trades', 0) > 0],
                key=lambda x: x[1].get('total_r', 0), reverse=True
            )
            
            if not sorted_symbols:
                await update.message.reply_text("📊 No completed trades yet.")
                return
            
            # Top 5
            top5_str = ""
            for i, (sym, data) in enumerate(sorted_symbols[:5]):
                emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "📈"
                wr = (data.get('wins', 0) / max(data.get('trades', 1), 1)) * 100
                top5_str += f"{emoji} {sym}: {data.get('total_r', 0):+.1f}R ({data.get('trades', 0)}T, {wr:.0f}%)\n"
            
            # Bottom 5
            bottom5_str = ""
            for sym, data in sorted_symbols[-5:][::-1]:
                wr = (data.get('wins', 0) / max(data.get('trades', 1), 1)) * 100
                bottom5_str += f"📉 {sym}: {data.get('total_r', 0):+.1f}R ({data.get('trades', 0)}T, {wr:.0f}%)\n"
            
            total_r = sum(d.get('total_r', 0) for d in symbol_stats.values())
            active = len([s for s, d in symbol_stats.items() if d.get('trades', 0) > 0])
            profitable = len([s for s, d in symbol_stats.items() if d.get('total_r', 0) > 0])
            
            msg = f"""
📊 **SYMBOL LEADERBOARD**
━━━━━━━━━━━━━━━━━━━━

🏆 **TOP 5**
{top5_str}
⚠️ **BOTTOM 5**
{bottom5_str}
📈 Active: {active} | Profitable: {profitable} | Total R: {total_r:+.1f}R

💡 /dashboard /stats
"""
            await update.message.reply_text(msg, parse_mode='Markdown')
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")
    async def cmd_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Emergency stop"""
        self.bot.trading_enabled = False
        msg = """
⛔ **EMERGENCY STOP EXECUTED**

Trading has been halted.
Pending signals will be ignored.
Active positions will remain open but no new trades will be taken.

To resume: `/start`
"""
        await update.message.reply_text(msg, parse_mode='Markdown')
        logger.warning(f"⛔ EMERGENCY STOP triggered by user {update.effective_user.name}")
    
    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Start/resume trading"""
        self.bot.trading_enabled = True
        msg = "✅ **TRADING RESUMED**\n\nThe bot will process the next available signals."
        await update.message.reply_text(msg, parse_mode='Markdown')
        logger.info(f"✅ Trading resumed by user {update.effective_user.name}")
    
    async def cmd_risk(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """View or update risk per trade"""
        try:
            if not context.args:
                # View current risk
                risk_pct = self.bot.risk_config.get('risk_per_trade', 0.01) * 100
                msg = f"💰 **CURRENT RISK**: {risk_pct:.1f}% per trade\n\nTo update: `/risk 0.5` (for 0.5%)"
                await update.message.reply_text(msg, parse_mode='Markdown')
                return
            
            # Update risk
            try:
                val_str = context.args[0].replace('%', '')
                new_risk = float(val_str)
                
                # If user enters 1, assume 1%. If 0.01, assume 1%
                if new_risk >= 1:
                    new_risk = new_risk / 100
                
                success, msg = self.bot.set_risk_per_trade(new_risk)
                if success:
                    await update.message.reply_text(f"✅ {msg}")
                else:
                    await update.message.reply_text(f"❌ {msg}")
                    
            except ValueError:
                await update.message.reply_text("❌ Invalid format. Use: `/risk 0.5`")
                
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")
            logger.error(f"Risk command error: {e}")
    async def cmd_radar(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show full radar watch for all symbols"""
        try:
            # Build comprehensive radar view with defensive checks
            pending_count = sum(len(sigs) for sigs in self.bot.pending_signals.values()) if self.bot.pending_signals else 0
            developing_count = 0
            extreme_count = 0
            
            if self.bot.radar_items:
                for data in self.bot.radar_items.values():
                    if isinstance(data, dict):
                        data_type = data.get('type', '')
                        if data_type in ['bullish_setup', 'bearish_setup']:
                            developing_count += 1
                        elif data_type in ['extreme_oversold', 'extreme_overbought']:
                            extreme_count += 1
            
            msg = f"""
📡 **FULL RADAR WATCH**
━━━━━━━━━━━━━━━━━━━━

Total Active: {pending_count + developing_count + extreme_count} signals

"""
            
            # 1. Pending BOS
            if self.bot.pending_signals:
                msg += "🎯 **PENDING BOS (Confirmed)**\n\n"
                for sym, sigs in self.bot.pending_signals.items():
                    for sig in sigs:
                        side_icon = "🟢" if sig.signal.signal_type == 'bullish' else "🔴"
                        candles_left = 6 - sig.candles_waited
                        msg += f"{side_icon} **{sym}**: {sig.candles_waited}/6 candles → Max {candles_left}h to entry\n"
                msg += "\n"
            
            # 2. Developing Setups
            developing = []
            extreme = []
            
            if self.bot.radar_items:
                for sym, data in self.bot.radar_items.items():
                    if isinstance(data, dict):
                        data_type = data.get('type', '')
                        if data_type in ['bullish_setup', 'bearish_setup']:
                            developing.append((sym, data))
                        elif data_type in ['extreme_oversold', 'extreme_overbought']:
                            extreme.append((sym, data))
            
            if developing:
                msg += "🔮 **DEVELOPING PATTERNS**\n\n"
                for sym, data in developing:
                    try:
                        data_type = data.get('type', '')
                        progress = int(data.get('pivot_progress', 3) or 3)
                        progress = max(0, min(6, progress))  # Clamp to 0-6
                        progress_bar = "▓" * progress + "░" * (6 - progress)
                        
                        price = float(data.get('price', 0) or 0)
                        ema_dist = float(data.get('ema_dist', 0) or 0)
                        rsi = float(data.get('rsi', 0) or 0)
                        prev_rsi = float(data.get('prev_pivot_rsi', 0) or 0)
                        rsi_div = float(data.get('rsi_div', 0) or 0)
                        
                        if data_type == 'bullish_setup':
                            msg += f"""🟢 **{sym}**: Bullish Divergence Forming
├─ Price: ${price:,.4f} ({ema_dist:+.1f}% from EMA)
├─ RSI: {rsi:.0f} ⬆️ (was {prev_rsi:.0f}, +{rsi_div:.0f}pts)
├─ Progress: {progress_bar} {progress}/6
└─ ETA: 3-9h to signal\n\n"""
                        else:
                            msg += f"""🔴 **{sym}**: Bearish Divergence Forming
├─ Price: ${price:,.4f} ({ema_dist:+.1f}% from EMA)
├─ RSI: {rsi:.0f} ⬇️ (was {prev_rsi:.0f}, +{rsi_div:.0f}pts)
├─ Progress: {progress_bar} {progress}/6
└─ ETA: 3-9h to signal\n\n"""
                    except Exception as item_err:
                        msg += f"⚠️ {sym}: Error displaying\n"
                        logger.error(f"Radar item error for {sym}: {item_err}")
            
            if extreme:
                msg += "⚡ **EXTREME ZONES**\n\n"
                for sym, data in extreme:
                    try:
                        data_type = data.get('type', '')
                        rsi = float(data.get('rsi', 0) or 0)
                        hours = float(data.get('hours_in_zone', 0) or 0)
                        price = float(data.get('price', 0) or 0)
                        
                        if data_type == 'extreme_oversold':
                            msg += f"""❄️ **{sym}**: Extreme Oversold
├─ RSI: {rsi:.0f} ({hours:.0f}h in zone)
├─ Price: ${price:,.4f}
└─ ETA: 2-8h to reversal\n\n"""
                        else:
                            msg += f"""🔥 **{sym}**: Extreme Overbought
├─ RSI: {rsi:.0f} ({hours:.0f}h in zone)
├─ Price: ${price:,.4f}
└─ ETA: 2-8h to reversal\n\n"""
                    except Exception as item_err:
                        msg += f"⚠️ {sym}: Error displaying\n"
                        logger.error(f"Radar extreme item error for {sym}: {item_err}")
            
            if not self.bot.pending_signals and not developing and not extreme:
                msg += "All clear - no active radar signals\n"
            
            await update.message.reply_text(msg, parse_mode='Markdown')
            
        except Exception as e:
            import traceback
            logger.error(f"Error in cmd_radar: {e}")
            logger.error(traceback.format_exc())
            await update.message.reply_text(f"❌ Radar error: {str(e)[:100]}")
