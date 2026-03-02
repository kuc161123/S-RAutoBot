"""
Enhanced Telegram Handler with Commands
========================================
Full command support for Multi-Divergence Bot:
- /dashboard - Main dashboard with divergence breakdown
- /help - Command list
- /positions - All active positions
- /stats - Performance statistics
- /stop - Emergency stop

Divergence Types:
- REG_BULL (🟢): Regular Bullish (Reversal)
- REG_BEAR (🔴): Regular Bearish (Reversal)
- HID_BULL (🟠): Hidden Bullish (Continuation)
- HID_BEAR (🟡): Hidden Bearish (Continuation)
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
        
    async def start(self):
        """Initialize Telegram application and start polling"""
        if self.app:
            logger.warning("Telegram bot already started")
            return

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
        self.app.add_handler(CommandHandler("resetstats", self.cmd_resetstats))
        self.app.add_handler(CommandHandler("resetlifetime", self.cmd_resetlifetime))
        self.app.add_handler(CommandHandler("debug", self.cmd_debug))
        self.app.add_handler(CommandHandler("setbalance", self.cmd_setbalance))
        
        # Start polling with longer interval to avoid rate limits
        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling(
            poll_interval=2.0,      # Check every 2 seconds (default 0.0)
            timeout=20,             # Long polling timeout
            drop_pending_updates=True  # Don't process old commands on restart
        )
        
        logger.info("Telegram command handler started")
        
        # Rate limiting
        self._last_message_time = 0
    
    async def send_message(self, message: str, retries: int = 3):
        """Send a message with rate limiting and retry logic"""
        import time
        import asyncio
        
        if not self.app:
            logger.warning("Telegram app not initialized - message not sent")
            return
        
        # Rate limiting - wait at least 0.5s between messages
        now = time.time()
        time_since_last = now - getattr(self, '_last_message_time', 0)
        if time_since_last < 0.5:
            await asyncio.sleep(0.5 - time_since_last)
        
        for attempt in range(retries):
            try:
                await self.app.bot.send_message(
                    chat_id=self.chat_id,
                    text=message,
                    parse_mode='Markdown',
                    disable_web_page_preview=True
                )
                self._last_message_time = time.time()
                return  # Success
            except Exception as e:
                logger.error(f"Telegram send failed (attempt {attempt+1}/{retries}): {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(1)  # Wait before retry
                else:
                    logger.error(f"Failed to send message after {retries} attempts")

    
    # === COMMAND HANDLERS ===
    
    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Help command with all available commands"""
        # Get current symbol count from config
        try:
            enabled_count = len(self.bot.symbol_config.get_enabled_symbols())
        except:
            enabled_count = 0
        
        msg = f"""🤖 **1H MULTI-DIVERGENCE BOT**
━━━━━━━━━━━━━━━━━━━━

📊 **MONITORING**
├ /dashboard - Live trading dashboard
├ /positions - All active positions
├ /stats - Performance statistics (all-time)
├ /performance - Symbol leaderboard (R values)
└ /radar - Developing setups & RSI extremes

💰 **RISK MANAGEMENT**
├ /risk - View current risk settings
├ /risk 0.2 - Set 0.2% risk per trade
├ /risk $5 - Set fixed $5 per trade
└ /setbalance - Set P&L baseline after deposits

🔄 **RESET OPTIONS**
├ /resetstats - Reset session stats only
└ /resetlifetime - Full reset (ALL tracking + baseline)

⚙️ **BOT CONTROL**
├ /stop - Emergency halt (stops new trades)
├ /start - Resume trading
└ /debug - Technical debugging info

❓ /help - Show this message

━━━━━━━━━━━━━━━━━━━━
**Strategy**: 1H Precision Divergence
**Portfolio**: {enabled_count} Symbols | 0.1% Risk
"""
        await update.message.reply_text(msg, parse_mode='Markdown')
    
    async def cmd_dashboard(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Clean, focused trading dashboard"""
        try:
            from datetime import datetime, timedelta

            # === SYNC WITH EXCHANGE ===
            await self.bot.sync_with_exchange()

            # === GATHER ALL DATA ===
            uptime_hrs = max(0, (datetime.now() - self.bot.start_time).total_seconds() / 3600)
            pending = sum(len(sigs) for sigs in self.bot.pending_signals.values())

            # Fetch positions ONCE and reuse
            positions = []
            try:
                positions = await self.bot.broker.get_positions() or []
            except Exception as e:
                logger.error(f"[DASHBOARD] Failed to get positions from Bybit: {e}")

            active_positions = [p for p in positions if float(p.get('size', 0)) > 0]
            active = len(active_positions)
            if active == 0:
                active = len(self.bot.active_trades)  # Fallback to internal tracking
            logger.info(f"[DASHBOARD] Bybit reports {active} active positions")

            enabled = len(self.bot.symbol_config.get_enabled_symbols())

            # Scan status
            scan = self.bot.scan_state
            last_scan = scan.get('last_scan_time')
            mins_ago = int((datetime.now() - last_scan).total_seconds() / 60) if last_scan else 0

            # Balance and P&L
            balance = await self.bot.broker.get_balance() or 0

            # Risk amount for R calculation
            risk_usd = self.bot.risk_config.get('risk_amount_usd')
            if risk_usd:
                risk_amount = float(risk_usd)
                risk_pct = (risk_amount / balance * 100) if balance > 0 else 0
                risk_display = f"${risk_amount:.2f} ({risk_pct:.2f}%)"
            else:
                risk_pct = self.bot.risk_config.get('risk_per_trade', 0.005)
                risk_amount = balance * risk_pct if balance > 0 else 10
                risk_display = f"${risk_amount:.2f} ({risk_pct*100:.1f}%)"

            # Realized P&L (today + weekly)
            realized_pnl = 0
            weekly_pnl = 0
            today_trades = 0
            today_wins = 0
            weekly_trades = 0
            weekly_wins = 0
            try:
                closed_records = await self.bot.broker.get_all_closed_pnl(limit=100)
                if closed_records:
                    today = datetime.now().date()
                    week_ago = today - timedelta(days=7)
                    for record in closed_records:
                        try:
                            close_time = int(record.get('updatedTime', record.get('createdTime', 0)))
                            trade_date = datetime.fromtimestamp(close_time / 1000).date()
                            pnl = float(record.get('closedPnl', 0))
                            if trade_date == today:
                                realized_pnl += pnl
                                today_trades += 1
                                if pnl > 0:
                                    today_wins += 1
                            if trade_date >= week_ago:
                                weekly_pnl += pnl
                                weekly_trades += 1
                                if pnl > 0:
                                    weekly_wins += 1
                        except:
                            continue
            except Exception as e:
                logger.error(f"Error getting realized P&L: {e}")

            realized_r = realized_pnl / risk_amount if risk_amount > 0 else 0
            weekly_r = weekly_pnl / risk_amount if risk_amount > 0 else 0
            today_losses = today_trades - today_wins
            weekly_losses = weekly_trades - weekly_wins

            # Unrealized P&L (from cached positions)
            unrealized_pnl = 0
            positions_up = 0
            positions_down = 0
            position_lines = []
            for pos in active_positions:
                unrealized = float(pos.get('unrealisedPnl', 0))
                unrealized_pnl += unrealized
                symbol = pos.get('symbol', '???')
                side = pos.get('side', '?')
                side_icon = "📈" if side == 'Buy' else "📉"
                pnl_icon = "🟢" if unrealized >= 0 else "🔴"
                pos_r = unrealized / risk_amount if risk_amount > 0 else 0
                position_lines.append(f"├ {side_icon} {symbol}: {pnl_icon} ${unrealized:+,.2f} ({pos_r:+.1f}R)")
                if unrealized >= 0:
                    positions_up += 1
                else:
                    positions_down += 1

            unrealized_r = unrealized_pnl / risk_amount if risk_amount > 0 else 0

            # Net P&L
            net_pnl = realized_pnl + unrealized_pnl
            net_r = net_pnl / risk_amount if risk_amount > 0 else 0
            net_emoji = "🟢" if net_pnl >= 0 else "🔴"

            # === ALL-TIME PERFORMANCE FROM LOCAL LIFETIME_STATS ===
            lifetime = self.bot.lifetime_stats
            start_date = lifetime.get('start_date', 'Unknown')
            starting_balance = lifetime.get('starting_balance', 0) or balance

            lifetime_r = lifetime.get('total_r', 0.0)
            lifetime_pnl = lifetime.get('total_pnl', 0.0)
            lifetime_trades = lifetime.get('total_trades', 0)
            lifetime_wins = lifetime.get('wins', 0)
            lifetime_losses = lifetime_trades - lifetime_wins
            lifetime_wr = (lifetime_wins / lifetime_trades * 100) if lifetime_trades > 0 else 0

            # Best/worst from local storage
            best_trade_r = lifetime.get('best_trade_r', 0.0)
            best_trade_symbol = lifetime.get('best_trade_symbol', 'N/A')
            worst_trade_r = lifetime.get('worst_trade_r', 0.0)
            worst_trade_symbol = lifetime.get('worst_trade_symbol', 'N/A')
            best_day_r = lifetime.get('best_day_r', 0.0)
            best_day_date = lifetime.get('best_day_date', 'N/A')
            worst_day_r = lifetime.get('worst_day_r', 0.0)
            worst_day_date = lifetime.get('worst_day_date', 'N/A')

            # Drawdown and streaks from local storage
            max_dd = lifetime.get('max_drawdown_r', 0.0)
            current_streak = lifetime.get('current_streak', 0)
            streak_type = "W" if current_streak > 0 else ("L" if current_streak < 0 else "")
            longest_win_streak = lifetime.get('longest_win_streak', 0)
            longest_loss_streak = lifetime.get('longest_loss_streak', 0)

            # Calculate profit factor from stored gross data
            gross_profit = lifetime.get('gross_profit_r', 0.0)
            gross_loss = abs(lifetime.get('gross_loss_r', 0.0))
            if gross_profit > 0 and gross_loss > 0:
                profit_factor = gross_profit / gross_loss
                profit_factor_display = f"{profit_factor:.2f}"
            elif gross_profit > 0 and gross_loss == 0:
                profit_factor_display = "∞"
            else:
                # Fallback estimation from aggregate stats
                if lifetime_losses > 0 and lifetime_wins > 0:
                    avg_r = lifetime_r / lifetime_trades if lifetime_trades > 0 else 0
                    total_win_r = lifetime_wins * avg_r if lifetime_wins > 0 else 0
                    total_loss_r = abs(lifetime_r - total_win_r) if lifetime_r < total_win_r else abs(lifetime_losses)
                    profit_factor = total_win_r / total_loss_r if total_loss_r > 0 else 0
                    profit_factor_display = f"~{profit_factor:.1f}" if profit_factor > 0 else "N/A"
                else:
                    profit_factor_display = "N/A"

            # Calculate days since start
            try:
                if start_date and start_date != 'Unknown':
                    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                    days_running = (datetime.now() - start_dt).days
                else:
                    days_running = 0
                    start_date = datetime.now().strftime('%Y-%m-%d')
            except:
                days_running = 0

            # Available balance (from cached positions)
            total_margin = sum(float(p.get('positionIM', 0)) for p in active_positions)
            available_balance = balance - total_margin
            available_pct = (available_balance / balance * 100) if balance > 0 else 0

            # API Key expiry
            api_info = await self.bot.broker.get_api_key_info()
            if api_info:
                days_left = api_info.get('days_left')
                if days_left is not None:
                    if days_left <= 7:
                        key_status = f"⚠️ {days_left}d left!"
                    elif days_left <= 30:
                        key_status = f"🟡 {days_left}d left"
                    else:
                        key_status = f"✅ {days_left}d left"
                else:
                    key_status = "✅ Connected"
            else:
                key_status = "⚠️ API Check Failed"

            next_scan_mins = max(0, 60 - mins_ago)

            # Get today's signal counts
            divs_today = self.bot.bos_tracking.get('divergences_detected_today', 0)
            bos_today = self.bot.bos_tracking.get('bos_confirmed_today', 0)

            # P&L Return
            pnl_return_pct = (lifetime_pnl / starting_balance * 100) if starting_balance > 0 else 0
            pnl_emoji = "↑" if pnl_return_pct >= 0 else "↓"

            # Average R
            avg_r = lifetime_r / lifetime_trades if lifetime_trades > 0 else 0

            # Expectancy (avg R per trade)
            expectancy_display = f"{avg_r:+.2f}R" if lifetime_trades > 0 else "N/A"

            # === BUILD POSITIONS SECTION ===
            if position_lines:
                pos_detail = "\n".join(position_lines)
                pos_section = f"""📡 **POSITIONS ({active} Active)**
├ {positions_up} 🟢 Profit | {positions_down} 🔴 Loss
{pos_detail}
└ Awaiting BOS: {pending} symbols"""
            else:
                pos_section = f"""📡 **POSITIONS (0 Active)**
├ No open trades
└ Awaiting BOS: {pending} symbols"""

            # === BUILD ENHANCED DASHBOARD ===
            msg = f"""💰 **TRADING DASHBOARD**
━━━━━━━━━━━━━━━━━━━━

📈 **TODAY'S P&L**
├ 💵 Realized: ${realized_pnl:+,.2f} ({realized_r:+.1f}R) | {today_wins}W/{today_losses}L
├ 📊 Unrealized: ${unrealized_pnl:+,.2f} ({unrealized_r:+.1f}R) | {active} open
└ {net_emoji} **Net: ${net_pnl:+,.2f} ({net_r:+.1f}R)**

📅 **7-DAY P&L**
├ Realized: ${weekly_pnl:+,.2f} ({weekly_r:+.1f}R) | {weekly_trades} trades
└ Record: {weekly_wins}W/{weekly_losses}L

🏆 **ALL-TIME** ({days_running} days)
├ {lifetime_r:+.1f}R (${lifetime_pnl:+,.2f}) | {lifetime_trades} trades
├ WR: {lifetime_wr:.1f}% ({lifetime_wins}W/{lifetime_losses}L) | PF: {profit_factor_display}
├ Avg: {expectancy_display} | Max DD: {max_dd:-.1f}R
└ Streak: {abs(current_streak)}{streak_type} | Best: {longest_win_streak}W / {longest_loss_streak}L

🥇 **BEST/WORST**
├ Best Trade: {best_trade_r:+.1f}R ({best_trade_symbol})
├ Worst Trade: {worst_trade_r:+.1f}R ({worst_trade_symbol})
├ Best Day: {best_day_r:+.1f}R ({best_day_date})
└ Worst Day: {worst_day_r:+.1f}R ({worst_day_date})

💼 **ACCOUNT**
├ Balance: ${balance:,.2f}
├ Available: ${available_balance:,.2f} ({available_pct:.0f}%)
├ Risk/Trade: {risk_display}
└ Return: {pnl_emoji}{abs(pnl_return_pct):.1f}% (Base: ${starting_balance:,.0f})

{pos_section}

⏰ **SYSTEM HEALTH**
├ Uptime: {uptime_hrs:.1f}h ✅
├ Next Scan: ~{next_scan_mins}m
├ Symbols: {enabled} | Signals: {divs_today}D/{bos_today}BOS
└ 🔑 API: {key_status}

━━━━━━━━━━━━━━━━━━━━
/positions | /radar | /stats | /performance
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
        """Performance statistics with all-time tracking"""
        try:
            stats = self.bot.stats  # Session stats
            lifetime = self.bot.lifetime_stats  # All-time stats
            
            # Session stats
            session_trades = stats['total_trades']
            session_wr = stats['win_rate']
            session_avg_r = stats['avg_r']
            session_total_r = stats['total_r']
            
            # All-time stats (source of truth)
            lt_trades = lifetime.get('total_trades', 0)
            lt_wins = lifetime.get('wins', 0)
            lt_losses = lt_trades - lt_wins
            lt_wr = (lt_wins / lt_trades * 100) if lt_trades > 0 else 0
            lt_total_r = lifetime.get('total_r', 0.0)
            lt_avg_r = lt_total_r / lt_trades if lt_trades > 0 else 0
            lt_pnl = lifetime.get('total_pnl', 0.0)
            
            # Best/worst metrics
            best_trade_r = lifetime.get('best_trade_r', 0.0)
            best_trade_sym = lifetime.get('best_trade_symbol', 'N/A')
            worst_trade_r = lifetime.get('worst_trade_r', 0.0)
            worst_trade_sym = lifetime.get('worst_trade_symbol', 'N/A')
            best_day_r = lifetime.get('best_day_r', 0.0)
            best_day_date = lifetime.get('best_day_date', 'N/A')
            worst_day_r = lifetime.get('worst_day_r', 0.0)
            worst_day_date = lifetime.get('worst_day_date', 'N/A')
            
            # Streaks and drawdown
            max_dd = lifetime.get('max_drawdown_r', 0.0)
            current_streak = lifetime.get('current_streak', 0)
            streak_type = "W" if current_streak > 0 else ("L" if current_streak < 0 else "")
            longest_win = lifetime.get('longest_win_streak', 0)
            longest_loss = lifetime.get('longest_loss_streak', 0)
            
            # Get enabled symbols count from config
            enabled_symbols = len(self.bot.symbol_config.get_enabled_symbols())
            
            msg = f"""
📊 **PERFORMANCE STATISTICS**
━━━━━━━━━━━━━━━━━━━━

🏆 **ALL-TIME (Source of Truth)**
├ Total Trades: {lt_trades}
├ Wins: {lt_wins} (✅) | Losses: {lt_losses} (❌)
├ Win Rate: {lt_wr:.1f}%
├ Avg R/Trade: {lt_avg_r:+.2f}R
├ Total R: {lt_total_r:+.1f}R
└ Total P&L: ${lt_pnl:+,.2f}

📈 **EXTREMES**
├ 🥇 Best Trade: {best_trade_r:+.1f}R ({best_trade_sym})
├ 🥉 Worst Trade: {worst_trade_r:+.1f}R ({worst_trade_sym})
├ ⬆️ Best Day: {best_day_r:+.1f}R ({best_day_date})
└ ⬇️ Worst Day: {worst_day_r:+.1f}R ({worst_day_date})

📉 **RISK METRICS**
├ Max Drawdown: {max_dd:.1f}R
├ Current Streak: {abs(current_streak)}{streak_type}
└ Longest Streaks: {longest_win}W / {longest_loss}L

📝 **SESSION STATS (Since Restart)**
├ Trades: {session_trades} | WR: {session_wr:.1f}%
└ Total R: {session_total_r:+.1f}R

🎯 **VS EXPECTED ({enabled_symbols} symbols)**
├ Expected WR: ~23%
├ Actual WR: {lt_wr:.1f}%
├ Expected R/Trade: +0.50R
├ Actual R/Trade: {lt_avg_r:+.2f}R
└ Delta: {lt_avg_r - 0.50:+.2f}R

━━━━━━━━━━━━━━━━━━━━
💡 /dashboard /positions /performance
"""
            await update.message.reply_text(msg, parse_mode='Markdown')
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
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
        """View or update risk per trade (supports % and USD)"""
        try:
            # Get current balance for conversions
            try:
                balance = await self.bot.broker.get_balance()
            except:
                balance = 1000  # Fallback
            
            current_risk_pct = self.bot.risk_config.get('risk_per_trade', 0.005)
            current_risk_usd = balance * current_risk_pct
            
            if not context.args:
                # View current risk
                msg = f"""💰 **CURRENT RISK SETTINGS**
━━━━━━━━━━━━━━━━━━━━

📊 **Per Trade Risk**
├ Percentage: {current_risk_pct*100:.2f}%
├ USD Amount: ${current_risk_usd:.2f}
└ Balance: ${balance:,.2f}

⚙️ **To Update**:
├ `/risk 0.5` or `/risk 0.5%` → Set to 0.5%
├ `/risk $50` or `/risk 50usd` → Set to $50 per trade
└ `/risk 1%` → Set to 1%

💡 Current: ${current_risk_usd:.2f} per trade ({current_risk_pct*100:.2f}%)
"""
                await update.message.reply_text(msg, parse_mode='Markdown')
                return
            
            # Parse input - support multiple formats
            input_val = context.args[0].lower().strip()
            
            # Check for USD formats: $50, 50usd, 50$
            is_usd = False
            if input_val.startswith('$'):
                is_usd = True
                input_val = input_val[1:]  # Remove $
            elif input_val.endswith('usd'):
                is_usd = True
                input_val = input_val[:-3]  # Remove usd
            elif input_val.endswith('$'):
                is_usd = True
                input_val = input_val[:-1]  # Remove trailing $
            
            # Check for percentage format: 0.5%, 1%
            if input_val.endswith('%'):
                input_val = input_val[:-1]  # Remove %
            
            try:
                amount = float(input_val)
            except ValueError:
                await update.message.reply_text("❌ Invalid format. Examples:\n`/risk 0.5` (0.5%)\n`/risk $50` ($50 per trade)\n`/risk 1%` (1%)")
                return
            
            if is_usd:
                # Convert USD to percentage
                if balance <= 0:
                    await update.message.reply_text("❌ Cannot calculate percentage - balance unavailable")
                    return
                
                new_risk_pct = amount / balance
                new_risk_usd = amount
                
                # Validate
                if new_risk_pct > 0.05:  # Max 5%
                    await update.message.reply_text(f"⚠️ ${amount:.2f} is {new_risk_pct*100:.1f}% of your balance - too high! Max is 5%.")
                    return
                if new_risk_pct < 0.001:  # Min 0.1%
                    await update.message.reply_text(f"⚠️ ${amount:.2f} is only {new_risk_pct*100:.3f}% - too low! Min is 0.1%.")
                    return
            else:
                # Treat as percentage
                # Handle both 0.5 and 50 input styles
                if amount >= 1:
                    new_risk_pct = amount / 100  # User entered 1 for 1%
                else:
                    new_risk_pct = amount  # User entered 0.01 for 1%
                
                new_risk_usd = balance * new_risk_pct
                
                # Validate
                if new_risk_pct > 0.05:
                    await update.message.reply_text(f"⚠️ {new_risk_pct*100:.1f}% is too high! Max is 5%.")
                    return
                if new_risk_pct < 0.001:
                    await update.message.reply_text(f"⚠️ {new_risk_pct*100:.3f}% is too low! Min is 0.1%.")
                    return
            
            # Apply the new risk
            if is_usd:
                # Use fixed USD amount
                success, result_msg = self.bot.set_risk_usd(new_risk_usd)
                if success:
                    msg = f"""✅ **RISK UPDATED (Fixed USD)**
━━━━━━━━━━━━━━━━━━━━

📊 **New Risk Per Trade**
├ Fixed Amount: ${new_risk_usd:.2f}
├ Equivalent: ~{new_risk_pct*100:.2f}% of balance
└ Balance: ${balance:,.2f}

💡 Each trade will now risk exactly ${new_risk_usd:.2f}
"""
                    await update.message.reply_text(msg, parse_mode='Markdown')
                else:
                    await update.message.reply_text(f"❌ {result_msg}")
            else:
                # Use percentage
                success, result_msg = self.bot.set_risk_per_trade(new_risk_pct)
                if success:
                    msg = f"""✅ **RISK UPDATED (Percentage)**
━━━━━━━━━━━━━━━━━━━━

📊 **New Risk Per Trade**
├ Percentage: {new_risk_pct*100:.2f}%
├ USD Amount: ~${new_risk_usd:.2f}
└ Balance: ${balance:,.2f}

💡 Each trade will now risk {new_risk_pct*100:.2f}% of balance
"""
                    await update.message.reply_text(msg, parse_mode='Markdown')
                else:
                    await update.message.reply_text(f"❌ {result_msg}")
                
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")
            logger.error(f"Risk command error: {e}")
    
    async def cmd_resetstats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Reset all internal stats to zero"""
        try:
            # Reset all stats
            self.bot.stats = {
                'total_trades': 0,
                'wins': 0,
                'losses': 0,
                'total_r': 0.0,
                'win_rate': 0.0
            }
            self.bot.symbol_stats = {}
            self.bot.save_stats()
            
            msg = """✅ **STATS RESET**
━━━━━━━━━━━━━━━━━━━━

All internal tracking stats have been reset to zero.

📊 New Stats:
├ Trades: 0
├ Wins: 0  
├ Losses: 0
├ Total R: 0.0
└ Win Rate: 0.0%

💡 Fresh tracking starts now!
"""
            await update.message.reply_text(msg, parse_mode='Markdown')
            logger.info("Stats reset by user command")
            
        except Exception as e:
            await update.message.reply_text(f"❌ Error resetting stats: {e}")
            logger.error(f"Reset stats error: {e}")
    
    async def cmd_resetlifetime(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Full reset of ALL tracking - lifetime stats, session stats, and baseline balance"""
        try:
            # Require confirmation
            if not context.args or context.args[0].lower() != 'confirm':
                current_balance = await self.bot.broker.get_balance() or 0
                lifetime = self.bot.lifetime_stats
                
                msg = f"""⚠️ **FULL RESET WARNING**
━━━━━━━━━━━━━━━━━━━━

This will reset ALL tracking data:
├ 📊 Session stats → 0
├ 🏆 Lifetime stats → 0  
├ 💰 Baseline → ${current_balance:,.2f} (current balance)
├ 📈 Best/Worst trade → cleared
├ 📉 Max drawdown → cleared
└ 🔥 Streaks → cleared

**Current Lifetime Stats (will be lost):**
├ Total Trades: {lifetime.get('total_trades', 0)}
├ Total R: {lifetime.get('total_r', 0):+.1f}R
├ Total P&L: ${lifetime.get('total_pnl', 0):+,.2f}
└ Since: {lifetime.get('start_date', 'Unknown')}

⚡ To confirm, type:
`/resetlifetime confirm`
"""
                await update.message.reply_text(msg, parse_mode='Markdown')
                return
            
            # Get current balance as new baseline
            new_balance = await self.bot.broker.get_balance() or 0
            today = __import__('datetime').datetime.now().strftime('%Y-%m-%d')
            
            # Reset session stats
            self.bot.stats = {
                'total_trades': 0,
                'wins': 0,
                'losses': 0,
                'total_r': 0.0,
                'win_rate': 0.0,
                'avg_r': 0.0
            }
            self.bot.symbol_stats = {}
            self.bot.save_stats()
            
            # Reset lifetime stats with new baseline
            self.bot.lifetime_stats = {
                'start_date': today,
                'starting_balance': new_balance,
                'total_r': 0.0,
                'total_pnl': 0.0,
                'total_trades': 0,
                'wins': 0,
                'best_day_r': 0.0,
                'best_day_date': None,
                'worst_day_r': 0.0,
                'worst_day_date': None,
                'daily_r': {},
                'best_trade_r': 0.0,
                'best_trade_symbol': '',
                'best_trade_date': None,
                'worst_trade_r': 0.0,
                'worst_trade_symbol': '',
                'worst_trade_date': None,
                'max_drawdown_r': 0.0,
                'peak_equity_r': 0.0,
                'current_streak': 0,
                'longest_win_streak': 0,
                'longest_loss_streak': 0
            }
            self.bot.save_lifetime_stats()
            
            msg = f"""✅ **FULL RESET COMPLETE**
━━━━━━━━━━━━━━━━━━━━

All tracking has been reset:
├ 📅 New Start Date: {today}
├ 💰 New Baseline: ${new_balance:,.2f}
├ 📊 Session Stats: 0
├ 🏆 Lifetime Stats: 0
└ 📈 All extremes: cleared

💡 Fresh tracking starts from this moment!
Your P&L Return will now be calculated from ${new_balance:,.0f}.
"""
            await update.message.reply_text(msg, parse_mode='Markdown')
            logger.warning(f"[RESET] Full lifetime reset performed by user. New baseline: ${new_balance:.2f}")
            
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")
            import traceback
            logger.error(f"Reset lifetime error: {traceback.format_exc()}")
    
    async def cmd_debug(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Debug command to show raw Bybit API responses"""
        try:
            msg_parts = ["🔧 **DEBUG INFO**\n━━━━━━━━━━━━━━━━━━━━\n"]
            
            # Get raw positions from Bybit
            msg_parts.append("📊 **POSITIONS API**\n")
            try:
                positions = await self.bot.broker.get_positions()
                if positions is None:
                    msg_parts.append("❌ API returned None\n")
                elif len(positions) == 0:
                    msg_parts.append("⚠️ API returned empty list (0 positions)\n")
                else:
                    # Count positions with size > 0
                    open_count = sum(1 for p in positions if float(p.get('size', 0)) > 0)
                    msg_parts.append(f"✅ Total: {len(positions)} | Open (size>0): {open_count}\n")
                    
                    # Show first 5 open positions
                    msg_parts.append("\n📍 **Sample Positions:**\n")
                    shown = 0
                    for pos in positions:
                        if float(pos.get('size', 0)) > 0:
                            sym = pos.get('symbol', '?')[:12]
                            size = pos.get('size', 0)
                            side = pos.get('side', '?')
                            msg_parts.append(f"├ {sym}: {size} ({side})\n")
                            shown += 1
                            if shown >= 5:
                                break
                    
                    if open_count > 5:
                        msg_parts.append(f"└ ...and {open_count - 5} more\n")
            except Exception as e:
                msg_parts.append(f"❌ Error: {e}\n")
            
            # Get balance
            msg_parts.append("\n💰 **BALANCE API**\n")
            try:
                balance = await self.bot.broker.get_balance()
                msg_parts.append(f"✅ Balance: ${balance:.2f}\n" if balance else "❌ Balance: None\n")
            except Exception as e:
                msg_parts.append(f"❌ Error: {e}\n")
            
            # Internal tracking
            msg_parts.append("\n🔄 **INTERNAL TRACKING**\n")
            msg_parts.append(f"├ active_trades: {len(self.bot.active_trades)}\n")
            msg_parts.append(f"├ pending_signals: {sum(len(s) for s in self.bot.pending_signals.values())}\n")
            msg_parts.append(f"└ enabled_symbols: {len(self.bot.symbol_config.get_enabled_symbols())}\n")
            
            await update.message.reply_text(''.join(msg_parts), parse_mode='Markdown')
            
        except Exception as e:
            await update.message.reply_text(f"❌ Debug error: {e}")
            logger.error(f"Debug command error: {e}")
            
    async def cmd_radar(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show full radar watch for all symbols - handles long messages"""
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
            
            # Get divergence type summary
            div_summary = self.bot.symbol_config.get_divergence_summary()
            
            msg = f"""📡 **FULL RADAR WATCH**
━━━━━━━━━━━━━━━━━━━━

📊 **Portfolio**: {self.bot.symbol_config.get_total_enabled()} symbols
├ 🟢 `REG_BULL`: {div_summary.get('REG_BULL', 0)}
├ 🔴 `REG_BEAR`: {div_summary.get('REG_BEAR', 0)}
├ 🟠 `HID_BULL`: {div_summary.get('HID_BULL', 0)}
└ 🟡 `HID_BEAR`: {div_summary.get('HID_BEAR', 0)}

🎯 **Active Signals**: {pending_count + developing_count + extreme_count}

"""
            
            # 1. Pending BOS - show with divergence code
            if self.bot.pending_signals:
                msg += "⏳ **PENDING BOS (Waiting for Breakout)**\n\n"
                for sym, sigs in list(self.bot.pending_signals.items())[:10]:  # Limit to 10
                    for sig in sigs:
                        div_code = getattr(sig.signal, 'divergence_code', sig.signal.signal_type.upper()[:3])
                        side_icon = "🟢" if sig.signal.side == 'long' else "🔴"
                        candles_left = 12 - sig.candles_waited
                        msg += f"{side_icon} **{sym}** `{div_code}`: {sig.candles_waited}/12 → {candles_left}h max\n"
                
                if pending_count > 10:
                    msg += f"_...and {pending_count - 10} more_\n"
                msg += "\n"
            
            # 2. Developing Setups (limit to 5)
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
                for sym, data in developing[:5]:  # Limit to 5
                    try:
                        data_type = data.get('type', '')
                        progress = int(data.get('pivot_progress', 3) or 3)
                        progress = max(0, min(6, progress))
                        rsi = float(data.get('rsi', 0) or 0)
                        side_icon = "🟢" if data_type == 'bullish_setup' else "🔴"
                        side_name = "Bull" if data_type == 'bullish_setup' else "Bear"
                        
                        msg += f"{side_icon} **{sym}**: {side_name} forming ({progress}/6) RSI:{rsi:.0f}\n"
                    except:
                        pass
                
                if len(developing) > 5:
                    msg += f"_...and {len(developing) - 5} more_\n"
                msg += "\n"
            
            if extreme:
                msg += "⚡ **EXTREME ZONES**\n\n"
                for sym, data in extreme[:5]:  # Limit to 5
                    try:
                        data_type = data.get('type', '')
                        rsi = float(data.get('rsi', 0) or 0)
                        hours = float(data.get('hours_in_zone', 0) or 0)
                        
                        if data_type == 'extreme_oversold':
                            msg += f"❄️ **{sym}**: RSI {rsi:.0f} ({hours:.0f}h oversold)\n"
                        else:
                            msg += f"🔥 **{sym}**: RSI {rsi:.0f} ({hours:.0f}h overbought)\n"
                    except:
                        pass
                
                if len(extreme) > 5:
                    msg += f"_...and {len(extreme) - 5} more_\n"
            
            if not self.bot.pending_signals and not developing and not extreme:
                msg += "✨ All clear - no active radar signals\n"
            
            msg += "\n💡 /dashboard /positions"
            
            # Handle long messages - Telegram limit is 4096 chars
            if len(msg) > 4000:
                # Split into chunks
                chunks = [msg[i:i+4000] for i in range(0, len(msg), 4000)]
                for i, chunk in enumerate(chunks):
                    if i == 0:
                        await update.message.reply_text(chunk, parse_mode='Markdown')
                    else:
                        await update.message.reply_text(f"...(continued)\n{chunk}", parse_mode='Markdown')
            else:
                await update.message.reply_text(msg, parse_mode='Markdown')
            
        except Exception as e:
            import traceback
            logger.error(f"Error in cmd_radar: {e}")
            logger.error(traceback.format_exc())
            await update.message.reply_text(f"❌ Radar error: {str(e)[:100]}")

    async def cmd_setbalance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Set the starting balance baseline for P&L calculations (accounts for deposits)"""
        try:
            if not context.args:
                # Show current baseline
                starting_balance = self.bot.lifetime_stats.get('starting_balance', 0)
                start_date = self.bot.lifetime_stats.get('start_date', 'Unknown')
                current_balance = await self.bot.broker.get_balance() or 0
                
                msg = f"""💰 **BASELINE BALANCE**
━━━━━━━━━━━━━━━━━━━━

📊 **Current Baseline**: ${starting_balance:,.2f}
├ Set on: {start_date}
├ Current Balance: ${current_balance:,.2f}
└ Difference: ${current_balance - starting_balance:+,.2f}

⚙️ **To Update:**
`/setbalance 700` → Set baseline to $700
`/setbalance now` → Set baseline to current balance

💡 Use this after making deposits to reset P&L tracking.
"""
                await update.message.reply_text(msg, parse_mode='Markdown')
                return
            
            # Parse the new balance
            input_val = context.args[0].lower().strip()
            
            if input_val == 'now':
                # Set to current balance
                new_balance = await self.bot.broker.get_balance() or 0
            else:
                try:
                    new_balance = float(input_val.replace('$', '').replace(',', ''))
                except ValueError:
                    await update.message.reply_text("❌ Invalid format. Use `/setbalance 700` or `/setbalance now`")
                    return
            
            if new_balance <= 0:
                await update.message.reply_text("❌ Balance must be positive.")
                return
            
            # Update lifetime stats
            old_balance = self.bot.lifetime_stats.get('starting_balance', 0)
            self.bot.lifetime_stats['starting_balance'] = new_balance
            self.bot.save_lifetime_stats()
            
            msg = f"""✅ **BASELINE UPDATED**
━━━━━━━━━━━━━━━━━━━━

📊 **New Baseline**: ${new_balance:,.2f}
├ Previous: ${old_balance:,.2f}
└ Change: ${new_balance - old_balance:+,.2f}

💡 P&L Return % will now be calculated relative to ${new_balance:,.0f}.
"""
            await update.message.reply_text(msg, parse_mode='Markdown')
            logger.info(f"[SETBALANCE] Updated baseline from ${old_balance:.2f} to ${new_balance:.2f} by user command")
            
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")
            logger.error(f"Setbalance command error: {e}")

