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
        self.app.add_handler(CommandHandler("status", self.status))
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
            "/status - Show active trades and daily stats\n\n"
            "⚠️ **Risk Management**\n"
            "/risk - Show current risk settings\n"
            "/risk_pct <value> - Set risk as % of equity (e.g., 1.0)\n"
            "/risk_usd <value> - Set risk as fixed USD (e.g., 50)\n"
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
        
        msg = (
            f"📊 **Live Status**\n\n"
            f"💰 **Daily PnL**: ${daily_pnl:.2f}\n"
            f"🎯 **Win Rate**: {daily_wr:.1f}% ({trades_today} trades)\n\n"
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

    async def stop(self):
        self.running = False
        await self.app.updater.stop()
        await self.app.stop()
        await self.app.shutdown()
