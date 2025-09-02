from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from telegram.constants import UpdateType
import asyncio
import logging

logger = logging.getLogger(__name__)

class TGBot:
    def __init__(self, token:str, chat_id:int, shared:dict):
        # shared contains {"risk": RiskConfig, "book": Book, "panic": list, "meta": dict}
        self.app = Application.builder().token(token).build()
        self.chat_id = chat_id
        self.shared = shared
        
        # Add command handlers
        self.app.add_handler(CommandHandler("start", self.start))
        self.app.add_handler(CommandHandler("help", self.help))
        self.app.add_handler(CommandHandler("set_risk", self.set_risk))
        self.app.add_handler(CommandHandler("status", self.status))
        self.app.add_handler(CommandHandler("panic_close", self.panic_close))
        self.app.add_handler(CommandHandler("balance", self.balance))
        self.app.add_handler(CommandHandler("health", self.health))
        
        self.running = False

    async def start_polling(self):
        """Start the bot polling"""
        if not self.running:
            await self.app.initialize()
            await self.app.start()
            self.running = True
            logger.info("Telegram bot started polling")
            # Start polling in background, drop any pending updates to avoid conflicts
            await self.app.updater.start_polling(
                drop_pending_updates=True,
                allowed_updates=list(UpdateType)
            )

    async def stop(self):
        """Stop the bot"""
        if self.running:
            await self.app.updater.stop()
            await self.app.stop()
            await self.app.shutdown()
            self.running = False
            logger.info("Telegram bot stopped")

    async def send_message(self, text:str):
        """Send message to configured chat"""
        try:
            await self.app.bot.send_message(chat_id=self.chat_id, text=text, parse_mode='Markdown')
        except Exception as e:
            logger.error(f"Failed to send message: {e}")

    async def start(self, update:Update, ctx:ContextTypes.DEFAULT_TYPE):
        """Start command handler"""
        await update.message.reply_text(
            "🤖 *Trading Bot Active*\n\n"
            "Use /help to see available commands.",
            parse_mode='Markdown'
        )

    async def help(self, update:Update, ctx:ContextTypes.DEFAULT_TYPE):
        """Help command handler"""
        help_text = """
*Available Commands:*

/status - Show open positions
/balance - Show account balance
/set\_risk [amount] - Set risk per trade
/panic\_close [symbol] - Emergency close position
/help - Show this message

*Current Settings:*
Risk per trade: ${}
Max leverage: {}x
""".format(
            self.shared["risk"].risk_usd,
            self.shared["risk"].max_leverage
        )
        await update.message.reply_text(help_text, parse_mode='Markdown')

    async def set_risk(self, update:Update, ctx:ContextTypes.DEFAULT_TYPE):
        """Set risk amount per trade"""
        try:
            if not ctx.args:
                await update.message.reply_text("Usage: /set_risk 50")
                return
                
            v = float(ctx.args[0])
            if v <= 0 or v > 1000:
                await update.message.reply_text("Risk must be between $0 and $1000")
                return
                
            self.shared["risk"].risk_usd = v
            await update.message.reply_text(f"✅ Risk per trade set to ${v}")
            logger.info(f"Risk updated to ${v}")
        except ValueError:
            await update.message.reply_text("Invalid amount. Usage: /set_risk 50")
        except Exception as e:
            logger.error(f"Error in set_risk: {e}")
            await update.message.reply_text("Error updating risk")

    async def status(self, update:Update, ctx:ContextTypes.DEFAULT_TYPE):
        """Show current positions"""
        try:
            bk = self.shared["book"].positions
            if not bk:
                await update.message.reply_text("📊 No open positions")
                return
                
            lines = ["*Open Positions:*\n"]
            total_qty = 0
            
            for sym, p in bk.items():
                emoji = "🟢" if p.side == "long" else "🔴"
                lines.append(
                    f"{emoji} *{sym}*\n"
                    f"  Side: {p.side.upper()}\n"
                    f"  Qty: {p.qty}\n"
                    f"  Entry: {p.entry:.4f}\n"
                    f"  SL: {p.sl:.4f}\n"
                    f"  TP: {p.tp:.4f}\n"
                )
                total_qty += 1
                
            lines.append(f"\n*Total positions:* {total_qty}")
            await update.message.reply_text("\n".join(lines), parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error in status: {e}")
            await update.message.reply_text("Error getting status")

    async def balance(self, update:Update, ctx:ContextTypes.DEFAULT_TYPE):
        """Show account balance"""
        try:
            broker = self.shared.get("broker")
            if broker:
                balance = broker.get_balance()
                if balance:
                    await update.message.reply_text(f"💰 *Balance:* ${balance:.2f} USDT", parse_mode='Markdown')
                else:
                    await update.message.reply_text("Unable to fetch balance")
            else:
                await update.message.reply_text("Broker not initialized")
        except Exception as e:
            logger.error(f"Error in balance: {e}")
            await update.message.reply_text("Error getting balance")

    async def panic_close(self, update:Update, ctx:ContextTypes.DEFAULT_TYPE):
        """Emergency close position"""
        try:
            if not ctx.args:
                await update.message.reply_text("Usage: /panic_close BTCUSDT")
                return
                
            sym = ctx.args[0].upper()
            
            if sym not in self.shared["book"].positions:
                await update.message.reply_text(f"No position found for {sym}")
                return
                
            self.shared["panic"].append(sym)
            await update.message.reply_text(
                f"⚠️ *Panic close requested for {sym}*\n"
                f"Position will be closed at next tick.",
                parse_mode='Markdown'
            )
            logger.warning(f"Panic close requested for {sym}")
            
        except Exception as e:
            logger.error(f"Error in panic_close: {e}")
            await update.message.reply_text("Error processing panic close")
    
    async def health(self, update:Update, ctx:ContextTypes.DEFAULT_TYPE):
        """Show bot health and analysis status"""
        try:
            import datetime
            
            # Get shared data
            frames = self.shared.get("frames", {})
            last_analysis = self.shared.get("last_analysis", {})
            
            msg = "*🤖 Bot Health Status*\n\n"
            
            # Check if bot is receiving data
            if frames:
                msg += "✅ *Data Reception:* Active\n"
                msg += f"📊 *Symbols Tracked:* {len(frames)}\n\n"
                
                msg += "*Candle Data:*\n"
                for symbol, df in list(frames.items())[:5]:  # Show first 5
                    if df is not None and len(df) > 0:
                        last_time = df.index[-1]
                        candle_count = len(df)
                        msg += f"• {symbol}: {candle_count} candles, last: {last_time.strftime('%H:%M:%S')}\n"
                
                # Show last analysis times
                if last_analysis:
                    msg += "\n*Last Analysis:*\n"
                    now = datetime.datetime.now()
                    for sym, timestamp in list(last_analysis.items())[:5]:
                        time_ago = (now - timestamp).total_seconds()
                        if time_ago < 60:
                            msg += f"• {sym}: {int(time_ago)}s ago\n"
                        else:
                            msg += f"• {sym}: {int(time_ago/60)}m ago\n"
                else:
                    msg += "\n⏳ *Waiting for first candle close to analyze*"
            else:
                msg += "⚠️ *Data Reception:* No data yet\n"
                msg += "Bot is starting up..."
            
            await update.message.reply_text(msg, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error in health: {e}")
            await update.message.reply_text("Error getting health status")