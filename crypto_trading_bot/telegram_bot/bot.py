"""
Telegram bot for notifications and control
"""
import asyncio
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
import structlog
from typing import List, Optional

logger = structlog.get_logger(__name__)

class TelegramBot:
    """Simple Telegram bot for trading notifications"""
    
    def __init__(self, token: str, chat_ids: List[int], signal_generator=None):
        self.token = token
        self.chat_ids = chat_ids
        self.signal_generator = signal_generator
        self.app = None
        self.is_running = False
        
        logger.info(f"Telegram bot initialized with {len(chat_ids)} authorized chats")
    
    async def start(self):
        """Start the Telegram bot"""
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Create application with conflict handling
                self.app = Application.builder().token(self.token).build()
                
                # Add command handlers
                self.app.add_handler(CommandHandler("start", self.cmd_start))
                self.app.add_handler(CommandHandler("stop", self.cmd_stop))
                self.app.add_handler(CommandHandler("status", self.cmd_status))
                self.app.add_handler(CommandHandler("positions", self.cmd_positions))
                self.app.add_handler(CommandHandler("balance", self.cmd_balance))
                self.app.add_handler(CommandHandler("stats", self.cmd_stats))
                self.app.add_handler(CommandHandler("help", self.cmd_help))
                
                # Start bot
                await self.app.initialize()
                await self.app.start()
                
                # Try to stop any existing polling first
                try:
                    await self.app.updater.stop()
                except:
                    pass
                
                # Start polling with drop_pending_updates to clear old sessions
                await self.app.updater.start_polling(drop_pending_updates=True)
                
                self.is_running = True
                logger.info("Telegram bot started successfully")
                
                # Send startup message
                await self.send_message("🚀 Trading bot started successfully!")
                break  # Success, exit retry loop
                
            except Exception as e:
                retry_count += 1
                if "Conflict" in str(e):
                    logger.warning(f"Telegram conflict detected (attempt {retry_count}/{max_retries}). Another instance may be running.")
                    if retry_count < max_retries:
                        logger.info("Waiting 10 seconds before retry...")
                        await asyncio.sleep(10)
                    else:
                        logger.error("Failed to start Telegram bot after retries. Continuing without Telegram.")
                        self.app = None  # Disable Telegram functionality
                else:
                    logger.error(f"Failed to start Telegram bot: {e}")
                    break
    
    async def stop(self):
        """Stop the Telegram bot"""
        try:
            if self.app:
                await self.send_message("🛑 Trading bot shutting down...")
                await self.app.updater.stop()
                await self.app.stop()
                await self.app.shutdown()
            
            self.is_running = False
            logger.info("Telegram bot stopped")
            
        except Exception as e:
            logger.error(f"Error stopping Telegram bot: {e}")
    
    async def send_message(self, text: str):
        """Send message to all authorized chats"""
        if not self.app:
            return
        
        for chat_id in self.chat_ids:
            try:
                await self.app.bot.send_message(
                    chat_id=chat_id,
                    text=text,
                    parse_mode='Markdown'
                )
            except Exception as e:
                logger.error(f"Failed to send message to {chat_id}: {e}")
    
    # Command handlers
    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        if update.effective_chat.id not in self.chat_ids:
            await update.message.reply_text("❌ Unauthorized")
            return
        
        await update.message.reply_text(
            "✅ Trading bot is active!\n"
            "Use /help to see available commands."
        )
    
    async def cmd_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stop command"""
        if update.effective_chat.id not in self.chat_ids:
            await update.message.reply_text("❌ Unauthorized")
            return
        
        if self.signal_generator:
            await update.message.reply_text("⏸️ Stopping trading...")
            # Signal generator stop would be called from main
        else:
            await update.message.reply_text("❌ Bot not fully initialized")
    
    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command"""
        if update.effective_chat.id not in self.chat_ids:
            await update.message.reply_text("❌ Unauthorized")
            return
        
        if self.signal_generator and self.signal_generator.executor:
            positions = self.signal_generator.executor.position_manager.get_open_positions()
            total_pnl = self.signal_generator.executor.position_manager.get_total_pnl()
            
            status = f"📊 **Bot Status**\n"
            status += f"Status: {'🟢 Active' if self.is_running else '🔴 Inactive'}\n"
            status += f"Open Positions: {len(positions)}\n"
            status += f"Total P&L: ${total_pnl:.2f}\n"
            
            await update.message.reply_text(status, parse_mode='Markdown')
        else:
            await update.message.reply_text("❌ Bot not fully initialized")
    
    async def cmd_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /positions command"""
        if update.effective_chat.id not in self.chat_ids:
            await update.message.reply_text("❌ Unauthorized")
            return
        
        if self.signal_generator and self.signal_generator.executor:
            positions = self.signal_generator.executor.position_manager.get_open_positions()
            
            if not positions:
                await update.message.reply_text("📭 No open positions")
                return
            
            message = "📈 **Open Positions**\n\n"
            
            for pos in positions:
                pnl_emoji = "🟢" if pos.pnl > 0 else "🔴"
                message += (
                    f"{pnl_emoji} **{pos.symbol}**\n"
                    f"  Side: {pos.side}\n"
                    f"  Entry: ${pos.entry_price:.4f}\n"
                    f"  Size: {pos.size:.4f}\n"
                    f"  P&L: ${pos.pnl:.2f} ({pos.pnl_percent:.2f}%)\n\n"
                )
            
            await update.message.reply_text(message, parse_mode='Markdown')
        else:
            await update.message.reply_text("❌ Bot not fully initialized")
    
    async def cmd_balance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /balance command"""
        if update.effective_chat.id not in self.chat_ids:
            await update.message.reply_text("❌ Unauthorized")
            return
        
        if self.signal_generator and self.signal_generator.exchange:
            balance = self.signal_generator.exchange.get_account_balance()
            
            if balance:
                await update.message.reply_text(f"💰 **Balance**: ${balance:.2f}", parse_mode='Markdown')
            else:
                await update.message.reply_text("❌ Failed to get balance")
        else:
            await update.message.reply_text("❌ Bot not fully initialized")
    
    async def cmd_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stats command"""
        if update.effective_chat.id not in self.chat_ids:
            await update.message.reply_text("❌ Unauthorized")
            return
        
        if self.signal_generator and self.signal_generator.executor:
            stats = self.signal_generator.executor.position_manager.get_statistics()
            
            message = (
                f"📊 **Trading Statistics**\n\n"
                f"Total Trades: {stats['total_trades']}\n"
                f"Winning Trades: {stats['winning_trades']}\n"
                f"Losing Trades: {stats['losing_trades']}\n"
                f"Win Rate: {stats['win_rate']:.1f}%\n"
                f"Total P&L: ${stats['total_pnl']:.2f}\n"
                f"Average P&L: ${stats['average_pnl']:.2f}\n"
                f"Open Positions: {stats['open_positions']}"
            )
            
            await update.message.reply_text(message, parse_mode='Markdown')
        else:
            await update.message.reply_text("❌ Bot not fully initialized")
    
    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        if update.effective_chat.id not in self.chat_ids:
            await update.message.reply_text("❌ Unauthorized")
            return
        
        help_text = (
            "🤖 **Trading Bot Commands**\n\n"
            "/start - Start the bot\n"
            "/stop - Stop trading\n"
            "/status - Bot status\n"
            "/positions - Show open positions\n"
            "/balance - Account balance\n"
            "/stats - Trading statistics\n"
            "/help - This help message"
        )
        
        await update.message.reply_text(help_text, parse_mode='Markdown')