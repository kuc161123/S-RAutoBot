from __future__ import annotations
from datetime import datetime
import logging
import asyncio
from typing import Optional

from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes, CallbackQueryHandler

# Use HTTPXRequest if available
try:
    from telegram.request import HTTPXRequest
except Exception:
    HTTPXRequest = None

logger = logging.getLogger(__name__)

class TGBot:
    def __init__(self, token: str, chat_id: int, shared: dict):
        """
        shared dict contains:
        - 'risk': RiskConfig object
        - 'book': Book object (active trades)
        - 'meta': dict for stats (daily_pnl, win_rate, etc.)
        """
        self.chat_id = chat_id
        self.shared = shared
        self.running = False
        
        # Build Application
        builder = Application.builder().token(token)
        if HTTPXRequest:
            request = HTTPXRequest(
                connect_timeout=10.0,
                read_timeout=120.0,
                pool_timeout=60.0,
                connection_pool_size=50
            )
            builder.request(request)
        self.app = builder.build()
        
        # Add Handlers
        self.app.add_handler(CommandHandler("start", self.start))
        self.app.add_handler(CommandHandler("help", self.help))
        self.app.add_handler(CommandHandler("dashboard", self.status))  # Alias for status
        self.app.add_handler(CommandHandler("status", self.status))
        self.app.add_handler(CommandHandler("health", self.health))
        self.app.add_handler(CommandHandler("symbols", self.symbols_info))
        self.app.add_handler(CommandHandler("backtest", self.backtest_info))
        self.app.add_handler(CommandHandler("risk", self.show_risk))
        self.app.add_handler(CommandHandler("set_risk", self.set_risk))
        self.app.add_handler(CommandHandler("risk_pct", self.set_risk_pct))
        self.app.add_handler(CommandHandler("risk_usd", self.set_risk_usd))
        
        # Error Handler
        self.app.add_error_handler(self.error_handler)

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "🚀 **AutoTrading Bot Online**\n"
            "Strategy: Adaptive Combo (Volatility Breakout)\n"
            "Use /help to see commands.",
            parse_mode=ParseMode.MARKDOWN
        )

    async def help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        msg = (
            "🛠 **Bot Commands**\n\n"
            "📊 **Dashboard**\n"
            "/status - Show active trades and daily stats\n"
            "/health - Bot uptime and connection status\n"
            "/symbols - Show active symbols count\n\n"
            "⚠️ **Risk Management**\n"
            "/risk - Show current risk settings\n"
            "/risk_pct <value> - Set risk % (0.1-5.0, e.g., /risk_pct 1.0)\n"
            "/risk_usd <value> - Set risk USD (5-1000, e.g., /risk_usd 50)\n\n"
            "📈 **Backtest Info**\n"
            "/backtest - Show backtest validation stats\n\n"
            "ℹ️ **Other**\n"
            "/help - Show this help message\n"
        )
        await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)

    async def status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        book = self.shared.get('book')
        meta = self.shared.get('meta', {})
        
        # Active Trades
        active_msg = "🚫 No active trades."
        if book and book.positions:
            lines = []
            for p in book.positions:
                pnl = p.unrealized_pnl if hasattr(p, 'unrealized_pnl') else 0.0
                icon = "🟢" if pnl >= 0 else "🔴"
                lines.append(f"{icon} {p.symbol} {p.side.upper()} | PnL: ${pnl:.2f}")
            active_msg = "\n".join(lines)
            
        # Daily Stats
        daily_pnl = meta.get('daily_pnl', 0.0)
        daily_wr = meta.get('daily_wr', 0.0)
        trades_today = meta.get('trades_today', 0)
        phantom_count = self.shared.get('phantom_count', 0)
        
        msg = (
            f"📊 **Live Status**\n\n"
            f"💰 **Daily PnL**: ${daily_pnl:.2f}\n"
            f"🎯 **Win Rate**: {daily_wr:.1f}% ({trades_today} trades)\n"
            f"👻 **Phantoms**: {phantom_count}\n\n"
            f"**Active Positions**:\n"
            f"{active_msg}"
        )
        await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)

    async def show_risk(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        risk = self.shared.get('risk')
        if not risk:
            await update.message.reply_text("❌ Risk config not available.")
            return
            
        mode = "Percentage" if risk.use_percent_risk else "Fixed USD"
        val = f"{risk.risk_percent}%" if risk.use_percent_risk else f"${risk.risk_usd}"
        
        await update.message.reply_text(
            f"⚠️ **Risk Configuration**\n"
            f"Mode: {mode}\n"
            f"Value: {val}",
            parse_mode=ParseMode.MARKDOWN
        )

    async def set_risk(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("Use /risk_pct <val> or /risk_usd <val>")

    async def set_risk_pct(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            val = float(context.args[0])
            if not (0.1 <= val <= 5.0):
                await update.message.reply_text("❌ Value must be between 0.1 and 5.0")
                return
            
            risk = self.shared.get('risk')
            if risk:
                risk.use_percent_risk = True
                risk.risk_percent = val
                await update.message.reply_text(f"✅ Risk set to **{val}%** of equity.", parse_mode=ParseMode.MARKDOWN)
        except (IndexError, ValueError):
            await update.message.reply_text("❌ Usage: /risk_pct 1.0")

    async def set_risk_usd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            val = float(context.args[0])
            if not (5 <= val <= 1000):
                await update.message.reply_text("❌ Value must be between 5 and 1000")
                return
            
            risk = self.shared.get('risk')
            if risk:
                risk.use_percent_risk = False
                risk.risk_usd = val
                await update.message.reply_text(f"✅ Risk set to **${val}** fixed.", parse_mode=ParseMode.MARKDOWN)
        except (IndexError, ValueError):
            await update.message.reply_text("❌ Usage: /risk_usd 50")

    async def error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE):
        logger.error(f"Telegram error: {context.error}", exc_info=context.error)

    async def send_message(self, text: str):
        if not self.running:
            return
        try:
            await self.app.bot.send_message(chat_id=self.chat_id, text=text, parse_mode=ParseMode.MARKDOWN)
        except Exception as e:
            logger.error(f"Failed to send message: {e}")

    async def start_polling(self):
        """Start the bot"""
        self.running = True
        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling(drop_pending_updates=True)
        logger.info("Telegram bot started")

    async def send_startup_notification(self):
        """Send startup notification with bot configuration"""
        risk = self.shared.get('risk')
        risk_mode = "Percentage" if (risk and risk.use_percent_risk) else "Fixed USD"
        risk_val = f"{risk.risk_percent}%" if (risk and risk.use_percent_risk) else f"${risk.risk_usd if risk else 'N/A'}"
        
        # Get symbol count
        overrides = self.shared.get('symbol_overrides', {})
        combo_count = len([s for s in overrides.values() if s.get('long') or s.get('short')])
        
        msg = (
            "🚀 **AutoTrading Bot Online**\n\n"
            "📊 **Strategy**: Adaptive Combo (Volatility Breakout)\n"
            f"✅ **Backtest Validated**: {combo_count} symbols with high-WR combos\n\n"
            "⚙️ **Configuration**:\n"
            f"💰 Risk Mode: {risk_mode} ({risk_val})\n"
            "🎯 Target: 1.0R (High WR Scalping)\n"
            "📉 Entry: Next Candle Open (realistic)\n\n"
            "Use /help to see all commands."
        )
        await self.send_message(msg)

    async def health(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show bot health status"""
        import time
        uptime = time.time() - self.shared.get('start_time', time.time())
        uptime_str = f"{int(uptime//3600)}h {int((uptime%3600)//60)}m"
        
        msg = (
            f"🏥 **Bot Health**\n\n"
            f"⏱ Uptime: {uptime_str}\n"
            f"🔗 Connection: {'✅ Active' if self.running else '❌ Inactive'}\n"
            f"📡 Telegram: {'✅ Online' if self.running else '❌ Offline'}\n"
        )
        await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)

    async def symbols_info(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show symbol/combo information"""
        overrides = self.shared.get('symbol_overrides', {})
        total_symbols = len(overrides)
        
        long_count = len([s for s in overrides.values() if s.get('long')])
        short_count = len([s for s in overrides.values() if s.get('short')])
        
        msg = (
            f"📊 **Symbols & Combos**\n\n"
            f"🎯 Validated Symbols: {total_symbols}\n"
            f"📈 Long Combos: {long_count}\n"
            f"📉 Short Combos: {short_count}\n\n"
            f"All combos are backtest-validated with WR > 60%"
        )
        await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)

    async def backtest_info(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show backtest validation information"""
        msg = (
            "📈 **Backtest Validation**\n\n"
            "✅ **Methodology**:\n"
            "• Entry: Next Candle Open (realistic)\n"
            "• Costs: 0.16% per trade (slippage + fees)\n"
            "• Sample Size: ~10k candles per symbol\n"
            "• Filter: WR > 60%, N >= 30\n\n"
            "🎯 **Strategy**:\n"
            "• Trigger: BBW > 0.45 + Vol > 0.8\n"
            "• Filter: RSI/MACD/VWAP/Fib combo classification\n"
            "• Target: 2.1R fixed\n\n"
            "All active combos passed strict validation."
        )
        await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)

    async def stop(self):
        self.running = False
        await self.app.updater.stop()
        await self.app.stop()
        await self.app.shutdown()
