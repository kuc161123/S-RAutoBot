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
        """Initialize Telegram application"""
        self.app = Application.builder().token(self.bot_token).build()
        
        # Register command handlers
        self.app.add_handler(CommandHandler("help", self.cmd_help))
        self.app.add_handler(CommandHandler("dashboard", self.cmd_dashboard))
        self.app.add_handler(CommandHandler("positions", self.cmd_positions))
        self.app.add_handler(CommandHandler("stats", self.cmd_stats))
        self.app.add_handler(CommandHandler("stop", self.cmd_stop))
        self.app.add_handler(CommandHandler("start", self.cmd_start))
        self.app.add_handler(CommandHandler("risk", self.cmd_risk))
        
        # Start polling in background
        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling()
        
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

⚙️ **CONTROL**
/stop - Emergency stop (halt trading)
/start - Resume trading
/help - Show this message

💡 **Strategy**: 1H RSI Divergence (Validated)
**Portfolio**: 27 Symbols, +375R OOS Performance
"""
        await update.message.reply_text(msg, parse_mode='Markdown')
    
    async def cmd_dashboard(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comprehensive dashboard with Bybit-verified data"""
        try:
            import time
            from datetime import datetime
            
            # === SYSTEM INFO ===
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
            
            # === BUILD COMPREHENSIVE MESSAGE ===
            msg = f"""
📊 **1H VALIDATED DASHBOARD**
━━━━━━━━━━━━━━━━━━━━

⏰ **SYSTEM**
├ Uptime: {uptime_hrs:.1f}h
├ Timeframe: 1H (60m)
├ Risk/Trade: {self.bot.risk_config.get('risk_per_trade', 0.01)*100}%
└ Enabled Symbols: {enabled} (100% Validated)

🎯 **STRATEGY**
├ Setup: RSI Divergence + EMA 200
├ Confirmation: Break of Structure
├ Exit: Per-Symbol R:R (4.5:1 to 8:1)
└ Expected OOS: +375R/Yr

💼 **WALLET (BYBIT)**
├ Balance: ${balance:,.2f} USDT
└ Realized P&L: ${total_closed_pnl:+,.2f}

📊 **EXCHANGE-VERIFIED STATS**
├ Closed Trades: {total_exchange}
├ ✅ Wins: {wins_exchange} | ❌ Losses: {losses_exchange}
├ Win Rate: {exchange_wr:.1f}%
└ Total P&L: ${total_closed_pnl:+,.2f}

📈 **INTERNAL TRACKING**
├ Tracked Trades: {stats['total_trades']}
├ Win Rate: {stats['win_rate']:.1f}%
├ Avg R/Trade: {stats['avg_r']:+.2f}R
└ Total R: {stats['total_r']:+.1f}R

🔔 **CURRENT STATUS**
├ Pending Signals: {pending}
├ Active Positions: {active}
└ Unrealized P&L: ${unrealized_pnl_usd:+,.2f} ({unrealized_r_total:+.1f}R)
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
