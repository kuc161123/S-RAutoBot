import asyncio
from typing import Optional, List, Dict, Any
from datetime import datetime

from telegram import (
    Update, 
    InlineKeyboardButton, 
    InlineKeyboardMarkup,
    ReplyKeyboardMarkup,
    KeyboardButton
)
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    filters,
    ContextTypes
)
import structlog

from ..config import settings
from typing import Union
from ..api.bybit_client import BybitClient
from ..api.enhanced_bybit_client import EnhancedBybitClient
from ..strategy.supply_demand import SupplyDemandStrategy
from ..strategy.advanced_supply_demand import AdvancedSupplyDemandStrategy
from ..db.models import User, Trade, BacktestResult
from ..backtesting.backtest_engine import BacktestEngine
from .formatters import format_status, format_positions, format_backtest_result

logger = structlog.get_logger(__name__)

class TradingBot:
    """Telegram bot for crypto trading"""
    
    def __init__(self, bybit_client: Union[BybitClient, EnhancedBybitClient], 
                 strategy: Union[SupplyDemandStrategy, AdvancedSupplyDemandStrategy]):
        self.bybit_client = bybit_client
        self.strategy = strategy
        self.application = None
        self.trading_enabled = True  # Auto-start enabled
        self.monitored_symbols = set()
        self.trading_engine = None  # Will be set by engine after initialization
        
    async def initialize(self):
        """Initialize the bot"""
        # Build application with timeout settings
        self.application = (
            Application.builder()
            .token(settings.telegram_bot_token)
            .connect_timeout(30.0)  # 30 seconds connect timeout
            .read_timeout(30.0)     # 30 seconds read timeout
            .write_timeout(30.0)    # 30 seconds write timeout
            .build()
        )
        
        # Add command handlers
        # MONITORING ONLY - No manual controls
        # Only basic status and monitoring commands
        self.application.add_handler(CommandHandler("start", self.cmd_start))
        self.application.add_handler(CommandHandler("status", self.cmd_status))
        self.application.add_handler(CommandHandler("positions", self.cmd_positions))
        self.application.add_handler(CommandHandler("logs", self.cmd_logs))
        self.application.add_handler(CommandHandler("zones", self.cmd_zones))
        
        # No keyboard buttons or callback handlers - monitoring only
        
        logger.info("Telegram bot initialized")
        
        # Send test notification to confirm bot is working
        try:
            await self.application.initialize()
            if settings.telegram_allowed_chat_ids:
                test_message = "🤖 **AUTO-TRADING BOT STARTED**\n\n"
                test_message += f"✅ Bot is now FULLY AUTONOMOUS\n"
                test_message += f"✅ Trading is ENABLED automatically\n"
                test_message += f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                test_message += f"🔧 Mode: Ultra Intelligent Engine\n"
                test_message += f"📊 Monitoring ALL AVAILABLE SYMBOLS\n"
                test_message += f"🔄 Continuous batch scanning (20 symbols/batch)\n\n"
                test_message += f"Bot will trade automatically - no action needed"
                
                for chat_id in settings.telegram_allowed_chat_ids:
                    try:
                        await self.application.bot.send_message(
                            chat_id=chat_id,
                            text=test_message,
                            parse_mode='Markdown'
                        )
                        logger.info(f"✅ Test notification sent successfully to chat_id {chat_id}")
                    except Exception as e:
                        logger.error(f"Failed to send test notification to chat_id {chat_id}: {e}")
        except Exception as e:
            logger.error(f"Failed to send startup notification: {e}")
    
    async def check_authorization(self, update: Update) -> bool:
        """Check if user is authorized"""
        chat_id = update.effective_chat.id
        
        if not settings.telegram_allowed_chat_ids:
            # No restrictions if list is empty
            return True
        
        if chat_id in settings.telegram_allowed_chat_ids:
            return True
        
        await update.message.reply_text(
            "⛔ Unauthorized access. This bot is private.\n"
            "Your chat ID: " + str(chat_id)
        )
        logger.warning(f"Unauthorized access attempt from chat_id: {chat_id}")
        return False
    
    async def handle_keyboard_button(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle keyboard button presses"""
        if not await self.check_authorization(update):
            return
        
        text = update.message.text
        
        # Map button text to commands
        if text == "📊 Status":
            await self.cmd_status(update, context)
        elif text == "💹 Positions":
            await self.cmd_positions(update, context)
        elif text == "🎯 Enable":
            await self.cmd_enable(update, context)
        elif text == "🛑 Disable":
            await self.cmd_disable(update, context)
        elif text == "⚙️ Settings":
            await self.show_settings_menu(update, context)
        elif text == "📈 Backtest":
            await self.show_backtest_help(update, context)
        elif text == "🤖 Auto-Trading":
            await self.show_autotrading_status(update, context)
        elif text == "🚨 Emergency":
            await self.cmd_emergency(update, context)
        else:
            await update.message.reply_text(
                "Unknown command. Use /help to see available commands."
            )
    
    async def show_autotrading_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show detailed autotrading status"""
        status_text = "🤖 **Auto-Trading Status**\n\n"
        
        # Check bot status
        if self.trading_enabled:
            status_text += "✅ **AUTO-TRADING IS ACTIVE**\n\n"
            status_text += "The bot is automatically:\n"
            status_text += "• Scanning for supply/demand zones\n"
            status_text += "• Analyzing price action patterns\n"
            status_text += "• Generating trading signals\n"
            status_text += "• Executing trades automatically\n"
            status_text += "• Managing stop losses and take profits\n\n"
        else:
            status_text += "🔴 **AUTO-TRADING IS DISABLED**\n\n"
            status_text += "The bot is in monitoring mode only.\n"
            status_text += "No trades will be executed.\n\n"
        
        # Check engine sync
        if hasattr(self, 'trading_engine') and self.trading_engine:
            engine_enabled = self.trading_engine.trading_enabled
            if engine_enabled == self.trading_enabled:
                status_text += "✅ Engine synchronized\n"
            else:
                status_text += "⚠️ Engine out of sync - restart recommended\n"
            
            # Show active settings
            status_text += f"\n**Active Settings:**\n"
            status_text += f"• Risk per trade: {settings.default_risk_percent}%\n"
            status_text += f"• Max positions: {settings.max_concurrent_positions}\n"
            status_text += f"• Symbols monitored: {len(self.monitored_symbols)}\n"
        
        # Add control buttons
        if self.trading_enabled:
            keyboard = [
                [InlineKeyboardButton("🛑 Disable Auto-Trading", callback_data="disable_trading")],
                [InlineKeyboardButton("📊 View Positions", callback_data="positions_refresh")]
            ]
        else:
            keyboard = [
                [InlineKeyboardButton("🎯 Enable Auto-Trading", callback_data="enable_confirm")],
                [InlineKeyboardButton("⚙️ Configure Settings", callback_data="settings_risk")]
            ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text(status_text, parse_mode='Markdown', reply_markup=reply_markup)
    
    async def show_settings_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show settings menu"""
        settings_text = (
            "⚙️ *Settings Menu*\n\n"
            "Choose what to configure:\n\n"
            "/risk - Risk management settings\n"
            "/symbols - Symbol selection\n"
            "/strategy - Strategy parameters\n"
            "/leverage - Leverage settings\n"
            "/margin - Margin mode\n\n"
            f"*Current Settings:*\n"
            f"Risk per trade: {settings.default_risk_percent}%\n"
            f"Max positions: {settings.max_concurrent_positions}\n"
            f"Leverage: {settings.default_leverage}x\n"
            f"Margin mode: {settings.default_margin_mode.value}"
        )
        
        keyboard = [
            [
                InlineKeyboardButton("💰 Risk", callback_data="settings_risk"),
                InlineKeyboardButton("📊 Symbols", callback_data="settings_symbols")
            ],
            [
                InlineKeyboardButton("🎯 Strategy", callback_data="settings_strategy"),
                InlineKeyboardButton("⚖️ Leverage", callback_data="settings_leverage")
            ],
            [
                InlineKeyboardButton("🔒 Margin Mode", callback_data="settings_margin")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            settings_text,
            parse_mode='Markdown',
            reply_markup=reply_markup
        )
    
    async def show_backtest_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show backtest help and quick options"""
        backtest_text = (
            "📈 *Backtest Module*\n\n"
            "Run historical simulations to test the strategy.\n\n"
            "*Usage:* `/backtest <symbol> <timeframe> <days>`\n\n"
            "*Examples:*\n"
            "• `/backtest BTCUSDT 15 30` - BTC 15min last 30 days\n"
            "• `/backtest ETHUSDT 1h 7` - ETH 1hour last 7 days\n\n"
            "*Quick Backtests:*"
        )
        
        keyboard = [
            [
                InlineKeyboardButton("BTC 15m 30d", callback_data="bt_BTCUSDT_15_30"),
                InlineKeyboardButton("ETH 15m 30d", callback_data="bt_ETHUSDT_15_30")
            ],
            [
                InlineKeyboardButton("BTC 1h 7d", callback_data="bt_BTCUSDT_60_7"),
                InlineKeyboardButton("SOL 15m 14d", callback_data="bt_SOLUSDT_15_14")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            backtest_text,
            parse_mode='Markdown',
            reply_markup=reply_markup
        )
    
    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        if not await self.check_authorization(update):
            return
        
        welcome_text = (
            "🤖 **AUTO-TRADING BOT ACTIVE**\n\n"
            "✅ Bot is running in FULL AUTOMATIC mode\n\n"
            "**Current Operation:**\n"
            "• Scanning markets continuously\n"
            "• Detecting supply/demand zones\n"  
            "• Executing trades automatically\n"
            "• Managing risk and positions\n\n"
            "**Monitoring Commands:**\n"
            "/status - Current bot status\n"
            "/positions - Open positions\n"
            "/zones - Active zones\n"
            "/logs - Recent activity\n\n"
            "⚠️ Manual controls disabled - Bot operates autonomously"
        )
        
        # No keyboard buttons - monitoring only
        
        await update.message.reply_text(
            welcome_text,
            parse_mode='Markdown'
        )
    
    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        if not await self.check_authorization(update):
            return
        
        help_text = (
            "📚 *Available Commands:*\n\n"
            "🎯 *Trading Control:*\n"
            "/enable - Start automated trading\n"
            "/disable - Stop automated trading\n"
            "/status - Show bot status\n"
            "/positions - View open positions\n\n"
            "🚨 *Emergency Controls:*\n"
            "/emergency - Emergency stop (disable + close all)\n"
            "/panic - Panic mode (stop all, preserve positions)\n"
            "/closeall - Close all positions immediately\n\n"
            "🤖 *ML & Intelligence:*\n"
            "/ml - ML model status & control\n"
            "/retrain - Retrain ML models\n\n"
            "📊 *Monitoring:*\n"
            "/zones - View active S/D zones\n"
            "/signals - View recent signals\n"
            "/scanner - Scanner status & control\n"
            "/phase - Scaling phase management\n"
            "/logs - View recent activity logs\n\n"
            "⚙️ *Configuration:*\n"
            "/symbols - Manage traded symbols\n"
            "/margin <symbol> <cross|isolated> - Set margin mode\n"
            "/leverage <symbol> <value> - Set leverage (1-125x)\n"
            "/risk - Configure risk parameters\n"
            "/strategy - Configure strategy settings\n\n"
            "📈 *Analysis:*\n"
            "/backtest <symbol> <timeframe> <days> - Run backtest\n\n"
            "💡 *Examples:*\n"
            "`/margin BTCUSDT isolated`\n"
            "`/leverage BTCUSDT 5`\n"
            "`/backtest BTCUSDT 15m 30`"
        )
        
        await update.message.reply_text(help_text, parse_mode='Markdown')
    
    async def cmd_enable(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /enable command"""
        if not await self.check_authorization(update):
            return
        
        if self.trading_enabled:
            await update.message.reply_text("✅ Trading is already enabled")
            return
        
        # Show confirmation with current settings
        account_info = await self.bybit_client.get_account_info()
        balance = float(account_info.get('totalWalletBalance', 0))
        
        confirmation_text = (
            "⚠️ *Enable Trading Confirmation*\n\n"
            f"Account Balance: ${balance:.2f}\n"
            f"Risk per trade: {settings.default_risk_percent}%\n"
            f"Max concurrent: {settings.max_concurrent_positions}\n"
            f"Max daily loss: {settings.max_daily_loss_percent}%\n"
            f"Active symbols: {len(self.monitored_symbols)}\n\n"
            "Are you sure you want to enable automated trading?"
        )
        
        keyboard = [
            [
                InlineKeyboardButton("✅ Confirm", callback_data="enable_confirm"),
                InlineKeyboardButton("❌ Cancel", callback_data="enable_cancel")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            confirmation_text,
            parse_mode='Markdown',
            reply_markup=reply_markup
        )
    
    async def cmd_disable(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /disable command"""
        if not await self.check_authorization(update):
            return
        
        if not self.trading_enabled:
            await update.message.reply_text("🛑 Trading is already disabled")
            return
        
        # Disable trading in both bot and engine
        self.trading_enabled = True  # Auto-start enabled
        if hasattr(self, 'trading_engine') and self.trading_engine:
            self.trading_engine.trading_enabled = False
            logger.info("Trading disabled in both bot and engine")
        
        await update.message.reply_text(
            "🛑 *Trading Disabled*\n\n"
            "Automated trading has been stopped.\n"
            "Scanner will continue monitoring but won't execute trades.\n"
            "Existing positions will remain open.",
            parse_mode='Markdown'
        )
        logger.info("Trading disabled by user")
    
    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command with comprehensive error handling"""
        if not await self.check_authorization(update):
            return
        
        try:
            import traceback
            logger.info(f"Status command requested by user {update.effective_user.id}")
            
            # Get account info with error handling
            try:
                account_info = await self.bybit_client.get_account_info()
                logger.debug(f"Account info fetched: Balance={account_info.get('totalWalletBalance', 'N/A')}")
            except Exception as e:
                logger.error(f"Error fetching account info: {e}")
                account_info = {'totalWalletBalance': '0', 'totalAvailableBalance': '0'}
            
            # Get positions with error handling
            try:
                positions = await self.bybit_client.get_positions()
                open_positions = []
                if positions:
                    for p in positions:
                        try:
                            size = float(p.get('size', p.get('qty', 0)))
                            if size > 0:
                                open_positions.append(p)
                        except (TypeError, ValueError):
                            continue
                logger.debug(f"Found {len(open_positions)} open positions")
            except Exception as e:
                logger.error(f"Error fetching positions: {e}")
                open_positions = []
            
            # Get active zones count with error handling
            total_zones = 0
            try:
                if hasattr(self.strategy, 'zones'):
                    # If strategy has zones dictionary
                    for symbol in self.monitored_symbols:
                        if symbol in self.strategy.zones:
                            total_zones += len(self.strategy.zones[symbol])
                elif hasattr(self.strategy, 'get_active_zones'):
                    # If strategy has get_active_zones method
                    for symbol in self.monitored_symbols:
                        try:
                            zones = self.strategy.get_active_zones(symbol)
                            if zones:
                                total_zones += len(zones)
                        except:
                            continue
                logger.debug(f"Total active zones: {total_zones}")
            except Exception as e:
                logger.warning(f"Error counting zones: {e}")
                total_zones = 0
            
            # Import formatter
            from .formatters import format_status
            
            # Format status message
            status_text = format_status(
                self.trading_enabled,
                account_info,
                open_positions,
                self.monitored_symbols,
                total_zones
            )
            
            # Add diagnostic information
            diagnostic_text = "\n\n📊 **Diagnostics:**\n"
            
            # Check trading engine status if available
            if hasattr(self, 'trading_engine') and self.trading_engine:
                try:
                    # Get real-time engine status
                    engine_trading = self.trading_engine.trading_enabled
                    engine_emergency = getattr(self.trading_engine, 'emergency_stop', False)
                    engine_positions = len(getattr(self.trading_engine, 'active_positions', {}))
                    engine_heat = getattr(self.trading_engine, 'portfolio_heat', 0)
                    
                    # Check if engine status matches bot status
                    sync_status = "✅ Synced" if engine_trading == self.trading_enabled else "⚠️ Out of sync"
                    
                    diagnostic_text += f"• Engine: 🟢 Connected {sync_status}\n"
                    diagnostic_text += f"• Bot Trading: {'✅ Enabled' if self.trading_enabled else '❌ Disabled'}\n"
                    diagnostic_text += f"• Engine Trading: {'✅ Enabled' if engine_trading else '❌ Disabled'}\n"
                    diagnostic_text += f"• Emergency Stop: {'🚨 ACTIVE' if engine_emergency else '✅ Clear'}\n"
                    diagnostic_text += f"• Active Positions: {engine_positions}\n"
                    diagnostic_text += f"• Portfolio Heat: {engine_heat:.1%}\n"
                    
                    # Check scanner status
                    if hasattr(self.trading_engine, 'scanner_task'):
                        scanner_running = self.trading_engine.scanner_task and not self.trading_engine.scanner_task.done()
                        diagnostic_text += f"• Scanner: {'🟢 Running' if scanner_running else '🔴 Stopped'}\n"
                except Exception as e:
                    diagnostic_text += f"• Engine: ⚠️ Error getting status: {str(e)[:30]}\n"
            else:
                diagnostic_text += "• Engine: ⚠️ Not connected\n"
                diagnostic_text += f"• Bot Trading: {'✅ Enabled' if self.trading_enabled else '❌ Disabled'}\n"
            
            # Check scanner status
            try:
                from ..utils.signal_queue import signal_queue
                queue_stats = await signal_queue.get_stats()
                diagnostic_text += f"• Signal Queue: {queue_stats.get('queue_size', 0)} pending\n"
                diagnostic_text += f"• Processing: {queue_stats.get('processing', 0)} signals\n"
            except:
                diagnostic_text += "• Signal Queue: ⚠️ Unavailable\n"
            
            # Check telegram connection
            diagnostic_text += f"• Telegram: {'🟢 Connected' if self.application and self.application.bot else '🔴 Disconnected'}\n"
            
            await update.message.reply_text(status_text + diagnostic_text, parse_mode='Markdown')
            logger.info("Status command completed successfully")
            
        except Exception as e:
            logger.error(f"Error getting status: {e}\n{traceback.format_exc()}")
            await update.message.reply_text(f"❌ Error fetching status: {str(e)[:100]}")
    
    async def cmd_symbols(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /symbols command"""
        if not await self.check_authorization(update):
            return
        
        # Show symbol selection menu with top 50 option
        keyboard = [
            [
                InlineKeyboardButton("🔥 Top 50 Coins", callback_data="symbols_top50"),
                InlineKeyboardButton("🏆 Top 20 Coins", callback_data="symbols_top20")
            ],
            [
                InlineKeyboardButton("💎 Top 10 Major", callback_data="symbols_major"),
                InlineKeyboardButton("📋 View Current", callback_data="symbols_view")
            ],
            [
                InlineKeyboardButton("🔄 Reset to Default (50)", callback_data="symbols_default")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        text = (
            "📊 *Symbol Management*\n\n"
            f"Currently monitoring: {len(self.monitored_symbols)} symbols\n\n"
            "The bot can monitor up to 50 cryptocurrencies simultaneously.\n\n"
            "Select a preset configuration:"
        )
        
        await update.message.reply_text(
            text,
            parse_mode='Markdown',
            reply_markup=reply_markup
        )
    
    async def cmd_margin(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /margin command"""
        if not await self.check_authorization(update):
            return
        
        if len(context.args) != 2:
            await update.message.reply_text(
                "❌ Usage: /margin <symbol> <cross|isolated>\n"
                "Example: `/margin BTCUSDT isolated`",
                parse_mode='Markdown'
            )
            return
        
        symbol = context.args[0].upper()
        mode = context.args[1].lower()
        
        if mode not in ['cross', 'isolated']:
            await update.message.reply_text("❌ Mode must be 'cross' or 'isolated'")
            return
        
        try:
            success = await self.bybit_client.set_margin_mode(symbol, mode)
            
            if success:
                await update.message.reply_text(
                    f"✅ Set {symbol} to {mode} margin mode"
                )
            else:
                await update.message.reply_text(
                    f"❌ Failed to set margin mode for {symbol}"
                )
                
        except Exception as e:
            logger.error(f"Error setting margin mode: {e}")
            await update.message.reply_text(f"❌ Error: {str(e)}")
    
    async def cmd_leverage(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /leverage command"""
        if not await self.check_authorization(update):
            return
        
        if len(context.args) != 2:
            await update.message.reply_text(
                "❌ Usage: /leverage <symbol> <value>\n"
                "Example: `/leverage BTCUSDT 5`",
                parse_mode='Markdown'
            )
            return
        
        symbol = context.args[0].upper()
        try:
            leverage = int(context.args[1])
        except ValueError:
            await update.message.reply_text("❌ Leverage must be a number")
            return
        
        try:
            success = await self.bybit_client.set_leverage(symbol, leverage)
            
            if success:
                await update.message.reply_text(
                    f"✅ Set {symbol} leverage to {leverage}x"
                )
            else:
                await update.message.reply_text(
                    f"❌ Failed to set leverage for {symbol}"
                )
                
        except Exception as e:
            logger.error(f"Error setting leverage: {e}")
            await update.message.reply_text(f"❌ Error: {str(e)}")
    
    async def cmd_risk(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /risk command"""
        if not await self.check_authorization(update):
            return
        
        text = (
            "⚠️ *Risk Management Settings*\n\n"
            f"Risk per trade: {settings.default_risk_percent}%\n"
            f"Max concurrent positions: {settings.max_concurrent_positions}\n"
            f"Max daily loss: {settings.max_daily_loss_percent}%\n"
            f"Default leverage: {settings.default_leverage}x\n"
            f"Use trailing stop: {'Yes' if settings.use_trailing_stop else 'No'}\n"
            f"Move to BE at TP1: {'Yes' if settings.move_stop_to_breakeven_at_tp1 else 'No'}\n\n"
            "Adjust settings:"
        )
        
        keyboard = [
            [
                InlineKeyboardButton("Risk: 0.5%", callback_data="risk_0.5"),
                InlineKeyboardButton("Risk: 1%", callback_data="risk_1"),
                InlineKeyboardButton("Risk: 2%", callback_data="risk_2")
            ],
            [
                InlineKeyboardButton("Max Pos: 3", callback_data="maxpos_3"),
                InlineKeyboardButton("Max Pos: 5", callback_data="maxpos_5"),
                InlineKeyboardButton("Max Pos: 10", callback_data="maxpos_10")
            ],
            [
                InlineKeyboardButton("Trailing: ON", callback_data="trailing_on"),
                InlineKeyboardButton("Trailing: OFF", callback_data="trailing_off")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            text,
            parse_mode='Markdown',
            reply_markup=reply_markup
        )
    
    async def cmd_strategy(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /strategy command"""
        if not await self.check_authorization(update):
            return
        
        text = (
            "📈 *Strategy Settings*\n\n"
            f"Min base candles: {settings.sd_min_base_candles}\n"
            f"Max base candles: {settings.sd_max_base_candles}\n"
            f"Departure ATR multiplier: {settings.sd_departure_atr_multiplier}\n"
            f"Zone buffer: {settings.sd_zone_buffer_percent}%\n"
            f"Max zone touches: {settings.sd_max_zone_touches}\n"
            f"Zone max age: {settings.sd_zone_max_age_hours} hours\n"
            f"Min zone score: {settings.sd_min_zone_score}\n"
            f"TP1 ratio: {settings.tp1_risk_ratio}R\n"
            f"TP2 ratio: {settings.tp2_risk_ratio}R\n"
            f"Partial at TP1: {settings.partial_tp1_percent}%"
        )
        
        await update.message.reply_text(text, parse_mode='Markdown')
    
    async def cmd_backtest(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /backtest command"""
        if not await self.check_authorization(update):
            return
        
        # Parse arguments
        if len(context.args) < 1:
            await update.message.reply_text(
                "❌ Usage: /backtest <symbol> [timeframe] [days]\n"
                "Example: `/backtest BTCUSDT 15m 30`",
                parse_mode='Markdown'
            )
            return
        
        symbol = context.args[0].upper()
        # Handle timeframe - support both '15' and '15m' formats
        timeframe = context.args[1] if len(context.args) > 1 else "15"
        # Clean up timeframe for display but keep original for processing
        timeframe_display = timeframe.upper() if 'm' in timeframe.lower() else f"{timeframe}M"
        days = int(context.args[2]) if len(context.args) > 2 else 30
        
        await update.message.reply_text(
            f"🔄 Running backtest for {symbol} on {timeframe_display} timeframe "
            f"for last {days} days...\n"
            "This may take a few minutes."
        )
        
        try:
            # Run backtest in background
            engine = BacktestEngine(self.bybit_client, self.strategy)
            result = await engine.run_backtest(symbol, timeframe, days)
            
            # Format and send results
            result_text = format_backtest_result(result)
            await update.message.reply_text(result_text, parse_mode='Markdown')
            
            # Send chart if available
            if result.get('chart_path'):
                with open(result['chart_path'], 'rb') as f:
                    await update.message.reply_photo(
                        photo=f,
                        caption="📊 Equity Curve"
                    )
            
        except Exception as e:
            import traceback
            logger.error(f"Backtest error: {e}\n{traceback.format_exc()}")
            error_msg = str(e)
            if "'bool' object is not callable" in error_msg:
                error_msg = "Strategy or engine initialization error. Please restart the bot."
            await update.message.reply_text(f"❌ Backtest failed: {error_msg[:200]}")
    
    async def cmd_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /positions command with interactive management"""
        if not await self.check_authorization(update):
            return
        
        try:
            positions = await self.bybit_client.get_positions()
            open_positions = [p for p in positions if float(p.get('size', 0)) > 0]
            
            if not open_positions:
                await update.message.reply_text("📊 No open positions")
                return
            
            text = format_positions(open_positions)
            
            # Add interactive buttons for position management
            keyboard = []
            
            # Add close buttons for first 3 positions
            for i, pos in enumerate(open_positions[:3]):
                symbol = pos['symbol']
                side = pos['side']
                keyboard.append([
                    InlineKeyboardButton(
                        f"Close {symbol} ({side})",
                        callback_data=f"close_{symbol}"
                    )
                ])
            
            # Add general actions
            keyboard.append([
                InlineKeyboardButton("🔄 Refresh", callback_data="positions_refresh"),
                InlineKeyboardButton("🚨 Close All", callback_data="closeall_confirm")
            ])
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            await update.message.reply_text(text, parse_mode='Markdown', reply_markup=reply_markup)
            
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            await update.message.reply_text("❌ Error fetching positions")
    
    async def cmd_logs(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /logs command"""
        if not await self.check_authorization(update):
            return
        
        # This would fetch from your logging system
        # For now, send placeholder
        await update.message.reply_text(
            "📝 *Recent Activity:*\n\n"
            "• Bot started\n"
            "• Instruments loaded: 523\n"
            "• WebSocket connected\n"
            "• Strategy initialized\n\n"
            "_Full logs available in server console_",
            parse_mode='Markdown'
        )
    
    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle inline keyboard callbacks"""
        query = update.callback_query
        await query.answer()
        
        data = query.data
        
        if data == "enable_confirm":
            self.trading_enabled = True
            
            # Also enable trading in the engine if connected
            if hasattr(self, 'trading_engine') and self.trading_engine:
                self.trading_engine.trading_enabled = True
                self.trading_engine.emergency_stop = False  # Clear emergency stop
                
                # Restart scanner if it was stopped
                if hasattr(self.trading_engine, 'start_scanner'):
                    asyncio.create_task(self.trading_engine.start_scanner())
                    logger.info("Scanner restarted with trading enabled")
                
                logger.info("Trading enabled in both bot and engine, emergency stop cleared")
            
            await query.edit_message_text(
                "✅ *Trading Enabled*\n\n"
                "• Automated trading is active\n"
                "• Scanner is monitoring symbols\n"
                "• Signals will be executed automatically\n"
                "• Emergency stop cleared\n\n"
                "The bot will now take trades based on supply/demand zones.",
                parse_mode='Markdown'
            )
            logger.info("Trading enabled by user - autotrading is active")
            
        elif data == "enable_cancel":
            await query.edit_message_text("❌ Trading enable cancelled")
        
        elif data == "disable_trading":
            # Disable trading in both bot and engine
            self.trading_enabled = True  # Auto-start enabled
            if hasattr(self, 'trading_engine') and self.trading_engine:
                self.trading_engine.trading_enabled = False
                logger.info("Trading disabled via callback in both bot and engine")
            
            await query.edit_message_text(
                "🛑 **Auto-Trading Disabled**\n\n"
                "• Automated trading stopped\n"
                "• Scanner continues monitoring\n"
                "• No new trades will be executed\n"
                "• Existing positions remain open\n\n"
                "Use /enable or press Enable button to resume.",
                parse_mode='Markdown'
            )
            
        elif data.startswith("risk_"):
            risk = float(data.split("_")[1])
            settings.default_risk_percent = risk
            await query.edit_message_text(f"✅ Risk per trade set to {risk}%")
            
        elif data.startswith("maxpos_"):
            max_pos = int(data.split("_")[1])
            settings.max_concurrent_positions = max_pos
            await query.edit_message_text(f"✅ Max concurrent positions set to {max_pos}")
            
        elif data.startswith("trailing_"):
            enabled = data.split("_")[1] == "on"
            settings.use_trailing_stop = enabled
            await query.edit_message_text(
                f"✅ Trailing stop {'enabled' if enabled else 'disabled'}"
            )
            
        elif data.startswith("symbols_"):
            await self.handle_symbol_selection(query, data)
            
        elif data.startswith("settings_"):
            await self.handle_settings_callback(query, data)
            
        elif data.startswith("bt_"):
            # Quick backtest buttons
            parts = data.split("_")
            if len(parts) == 4:
                symbol = parts[1]
                timeframe = parts[2]
                days = parts[3]
                
                await query.edit_message_text(
                    f"🔄 Running backtest for {symbol} on {timeframe}M timeframe for {days} days...\n"
                    "This may take a few minutes."
                )
                
                try:
                    from ..backtesting.backtest_engine import BacktestEngine
                    engine = BacktestEngine(self.bybit_client, self.strategy)
                    result = await engine.run_backtest(symbol, timeframe, int(days))
                    
                    from .formatters import format_backtest_result
                    result_text = format_backtest_result(result)
                    await query.message.reply_text(result_text, parse_mode='Markdown')
                    
                    if result.get('chart_path'):
                        with open(result['chart_path'], 'rb') as f:
                            await query.message.reply_photo(
                                photo=f,
                                caption="📊 Equity Curve"
                            )
                            
                except Exception as e:
                    import traceback
                    logger.error(f"Backtest error: {e}\n{traceback.format_exc()}")
                    error_msg = str(e)
                    if "'bool' object is not callable" in error_msg:
                        error_msg = "Strategy or engine initialization error. Please restart the bot."
                    await query.message.reply_text(f"❌ Backtest failed: {error_msg[:200]}")
        
        # Emergency control callbacks
        elif data == "emergency_confirm":
            # Disable trading in both bot and engine
            self.trading_enabled = True  # Auto-start enabled
            
            if hasattr(self, 'trading_engine') and self.trading_engine:
                self.trading_engine.trading_enabled = False
                self.trading_engine.emergency_stop = True
                
                # Cancel scanner task
                if hasattr(self.trading_engine, 'scanner_task'):
                    self.trading_engine.scanner_task.cancel()
            
            # Close all positions
            try:
                positions = await self.bybit_client.get_positions()
                open_positions = [p for p in positions if float(p.get('size', 0)) > 0]
                closed_count = 0
                
                for pos in open_positions:
                    try:
                        await self.bybit_client.close_position(pos['symbol'])
                        closed_count += 1
                    except:
                        pass
                
                await query.edit_message_text(
                    f"🚨 **EMERGENCY STOP COMPLETE**\n\n"
                    f"✅ Trading disabled\n"
                    f"✅ Emergency stop engaged\n"
                    f"✅ Scanner stopped\n"
                    f"✅ Closed {closed_count}/{len(open_positions)} positions\n\n"
                    f"System is now safe. Use /enable to resume.",
                    parse_mode='Markdown'
                )
                logger.warning(f"Emergency stop executed - closed {closed_count} positions")
            except Exception as e:
                await query.edit_message_text(f"❌ Emergency stop error: {str(e)}")
        
        elif data == "emergency_cancel":
            await query.edit_message_text("❌ Emergency stop cancelled")
        
        elif data == "closeall_confirm":
            try:
                positions = await self.bybit_client.get_positions()
                open_positions = [p for p in positions if float(p.get('size', 0)) > 0]
                closed = []
                failed = []
                
                for pos in open_positions:
                    try:
                        await self.bybit_client.close_position(pos['symbol'])
                        closed.append(pos['symbol'])
                    except Exception as e:
                        failed.append(pos['symbol'])
                
                text = "📊 **Position Closure Results**\n\n"
                if closed:
                    text += f"✅ Closed: {', '.join(closed)}\n"
                if failed:
                    text += f"❌ Failed: {', '.join(failed)}\n"
                
                await query.edit_message_text(text, parse_mode='Markdown')
            except Exception as e:
                await query.edit_message_text(f"❌ Error: {str(e)}")
        
        elif data == "closeall_cancel":
            await query.edit_message_text("❌ Close all cancelled")
        
        # ML control callbacks
        elif data == "ml_retrain":
            await query.edit_message_text("🔄 Starting ML retraining...")
            if hasattr(self, 'trading_engine') and self.trading_engine:
                if hasattr(self.trading_engine, 'ml_trainer'):
                    asyncio.create_task(self.trading_engine.ml_trainer.train_models())
                    await query.edit_message_text("✅ ML retraining started")
                else:
                    await query.edit_message_text("❌ ML trainer not available")
            else:
                await query.edit_message_text("❌ Engine not connected")
        
        elif data == "ml_stats":
            text = "📊 **ML Model Statistics**\n\n"
            text += "Model performance metrics coming soon..."
            await query.edit_message_text(text, parse_mode='Markdown')
        
        elif data == "ml_pause":
            if hasattr(self, 'trading_engine') and self.trading_engine:
                if hasattr(self.trading_engine, 'ml_enabled'):
                    self.trading_engine.ml_enabled = False
                    await query.edit_message_text("⏸️ ML predictions paused")
                else:
                    await query.edit_message_text("❌ ML control not available")
            else:
                await query.edit_message_text("❌ Engine not connected")
        
        elif data == "ml_resume":
            if hasattr(self, 'trading_engine') and self.trading_engine:
                if hasattr(self.trading_engine, 'ml_enabled'):
                    self.trading_engine.ml_enabled = True
                    await query.edit_message_text("▶️ ML predictions resumed")
                else:
                    await query.edit_message_text("❌ ML control not available")
            else:
                await query.edit_message_text("❌ Engine not connected")
        
        # Scanner control callbacks
        elif data == "scanner_start":
            if hasattr(self, 'trading_engine') and self.trading_engine:
                if hasattr(self.trading_engine, 'start_scanner'):
                    asyncio.create_task(self.trading_engine.start_scanner())
                    await query.edit_message_text("▶️ Scanner started")
                else:
                    await query.edit_message_text("❌ Scanner control not available")
            else:
                await query.edit_message_text("❌ Engine not connected")
        
        elif data == "scanner_stop":
            if hasattr(self, 'trading_engine') and self.trading_engine:
                if hasattr(self.trading_engine, 'scanner_task'):
                    self.trading_engine.scanner_task.cancel()
                    await query.edit_message_text("⏸️ Scanner stopped")
                else:
                    await query.edit_message_text("❌ Scanner not running")
            else:
                await query.edit_message_text("❌ Engine not connected")
        
        elif data == "scanner_restart":
            if hasattr(self, 'trading_engine') and self.trading_engine:
                if hasattr(self.trading_engine, 'scanner_task'):
                    self.trading_engine.scanner_task.cancel()
                    await asyncio.sleep(1)
                if hasattr(self.trading_engine, 'start_scanner'):
                    asyncio.create_task(self.trading_engine.start_scanner())
                    await query.edit_message_text("🔄 Scanner restarted")
                else:
                    await query.edit_message_text("❌ Scanner control not available")
            else:
                await query.edit_message_text("❌ Engine not connected")
        
        # Phase control callbacks
        elif data.startswith("phase_"):
            from ..config_modules.scaling_config import scaling_config
            
            if data == "phase_prev":
                current = scaling_config.CURRENT_PHASE
                if current > 0:
                    scaling_config.set_phase(current - 1)
                    await query.edit_message_text(f"✅ Switched to Phase {current - 1}")
                else:
                    await query.edit_message_text("Already at Phase 0")
            
            elif data == "phase_next":
                if scaling_config.next_phase():
                    await query.edit_message_text(f"✅ Advanced to Phase {scaling_config.CURRENT_PHASE}")
                else:
                    await query.edit_message_text("Already at maximum phase")
            
            else:
                # Direct phase selection (phase_0, phase_1, etc.)
                try:
                    phase_num = int(data.split("_")[1])
                    if scaling_config.set_phase(phase_num):
                        config = scaling_config.get_current_config()
                        await query.edit_message_text(
                            f"✅ **Phase {phase_num} Activated**\n\n"
                            f"{config['description']}\n"
                            f"Symbols: {config['symbol_count']}\n"
                            f"Batch: {config['batch_size']}\n"
                            f"Delay: {config['scan_delay']}s",
                            parse_mode='Markdown'
                        )
                    else:
                        await query.edit_message_text("❌ Invalid phase number")
                except:
                    await query.edit_message_text("❌ Error setting phase")
        
        # Position management callbacks
        elif data == "positions_refresh":
            try:
                positions = await self.bybit_client.get_positions()
                open_positions = [p for p in positions if float(p.get('size', 0)) > 0]
                
                if not open_positions:
                    await query.edit_message_text("📊 No open positions")
                else:
                    text = format_positions(open_positions)
                    
                    # Recreate keyboard with updated positions
                    keyboard = []
                    for i, pos in enumerate(open_positions[:3]):
                        symbol = pos['symbol']
                        side = pos['side']
                        keyboard.append([
                            InlineKeyboardButton(
                                f"Close {symbol} ({side})",
                                callback_data=f"close_{symbol}"
                            )
                        ])
                    
                    keyboard.append([
                        InlineKeyboardButton("🔄 Refresh", callback_data="positions_refresh"),
                        InlineKeyboardButton("🚨 Close All", callback_data="closeall_confirm")
                    ])
                    
                    reply_markup = InlineKeyboardMarkup(keyboard)
                    await query.edit_message_text(text, parse_mode='Markdown', reply_markup=reply_markup)
            except Exception as e:
                await query.edit_message_text(f"❌ Error: {str(e)}")
        
        elif data.startswith("close_"):
            symbol = data.replace("close_", "")
            try:
                await self.bybit_client.close_position(symbol)
                await query.edit_message_text(f"✅ Closed position: {symbol}")
            except Exception as e:
                await query.edit_message_text(f"❌ Failed to close {symbol}: {str(e)}")
    
    async def handle_settings_callback(self, query, data):
        """Handle settings callbacks"""
        if data == "settings_risk":
            # Show risk settings directly
            text = (
                "💰 *Risk Management Settings*\n\n"
                f"Risk per trade: {settings.default_risk_percent}%\n"
                f"Max concurrent: {settings.max_concurrent_positions}\n"
                f"Max daily loss: {settings.max_daily_loss_percent}%\n"
                f"Trailing stop: {'Enabled' if settings.use_trailing_stop else 'Disabled'}\n\n"
                "Select option to modify:"
            )
            
            keyboard = [
                [
                    InlineKeyboardButton("0.5%", callback_data="risk_0.5"),
                    InlineKeyboardButton("1%", callback_data="risk_1"),
                    InlineKeyboardButton("2%", callback_data="risk_2")
                ],
                [
                    InlineKeyboardButton("Max Pos: 3", callback_data="maxpos_3"),
                    InlineKeyboardButton("Max Pos: 5", callback_data="maxpos_5"),
                    InlineKeyboardButton("Max Pos: 10", callback_data="maxpos_10")
                ],
                [
                    InlineKeyboardButton("Trailing ON", callback_data="trailing_on"),
                    InlineKeyboardButton("Trailing OFF", callback_data="trailing_off")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await query.edit_message_text(text, parse_mode='Markdown', reply_markup=reply_markup)
            
        elif data == "settings_symbols":
            # Show symbol settings directly
            text = (
                "📊 *Symbol Selection*\n\n"
                f"Currently monitoring: {len(self.monitored_symbols)} symbols\n\n"
                "Choose preset or manage individually:"
            )
            
            keyboard = [
                [
                    InlineKeyboardButton("🔥 Top 50 Coins", callback_data="symbols_top50"),
                    InlineKeyboardButton("🏆 Top 20 Coins", callback_data="symbols_top20")
                ],
                [
                    InlineKeyboardButton("💎 Top 10 Major", callback_data="symbols_major"),
                    InlineKeyboardButton("📋 View Current", callback_data="symbols_view")
                ],
                [
                    InlineKeyboardButton("🔄 Reset to Default", callback_data="symbols_default")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await query.edit_message_text(text, parse_mode='Markdown', reply_markup=reply_markup)
            
        elif data == "settings_strategy":
            # Show strategy settings directly
            text = (
                "🎯 *Strategy Settings*\n\n"
                f"Min zone score: {settings.sd_min_zone_score}\n"
                f"Zone max age: {settings.sd_zone_max_age_hours}h\n"
                f"Max zone touches: {settings.sd_max_zone_touches}\n"
                f"TP1 ratio: {settings.tp1_risk_ratio}:1\n"
                f"TP2 ratio: {settings.tp2_risk_ratio}:1\n"
                f"Partial at TP1: {settings.partial_tp1_percent}%\n\n"
                "_Use /strategy command to modify_"
            )
            await query.edit_message_text(text, parse_mode='Markdown')
        elif data == "settings_leverage":
            text = (
                "⚖️ *Leverage Settings*\n\n"
                f"Current default: {settings.default_leverage}x\n\n"
                "To change leverage for a symbol:\n"
                "`/leverage <symbol> <value>`\n\n"
                "Example: `/leverage BTCUSDT 5`"
            )
            await query.edit_message_text(text, parse_mode='Markdown')
        elif data == "settings_margin":
            text = (
                "🔒 *Margin Mode Settings*\n\n"
                f"Current default: {settings.default_margin_mode.value}\n\n"
                "To change margin mode:\n"
                "`/margin <symbol> <cross|isolated>`\n\n"
                "Example: `/margin BTCUSDT isolated`"
            )
            await query.edit_message_text(text, parse_mode='Markdown')
    
    async def handle_symbol_selection(self, query, data):
        """Handle symbol selection callbacks"""
        if data == "symbols_top50":
            # Use the configured top 50 symbols from settings
            self.monitored_symbols = set(settings.default_symbols)
            await query.edit_message_text(
                f"✅ *Monitoring Top 50 Cryptocurrencies*\n\n"
                f"Total symbols: {len(self.monitored_symbols)}\n\n"
                f"Including: {', '.join(sorted(self.monitored_symbols)[:10])}...\n\n"
                "_All major cryptocurrencies are now being monitored_",
                parse_mode='Markdown'
            )
            
        elif data == "symbols_top20":
            # Get top 20 from the default symbols
            self.monitored_symbols = set(settings.default_symbols[:20])
            await query.edit_message_text(
                f"✅ *Monitoring Top 20 Cryptocurrencies*\n\n"
                f"Symbols: {', '.join(sorted(self.monitored_symbols)[:10])}...\n\n"
                f"Total: {len(self.monitored_symbols)} symbols",
                parse_mode='Markdown'
            )
            
        elif data == "symbols_major":
            # Top 10 major pairs
            self.monitored_symbols = set(settings.default_symbols[:10])
            await query.edit_message_text(
                f"✅ *Monitoring Top 10 Major Pairs*\n\n"
                f"Symbols: {', '.join(sorted(self.monitored_symbols))}",
                parse_mode='Markdown'
            )
            
        elif data == "symbols_default":
            # Reset to default configuration
            self.monitored_symbols = set(settings.default_symbols)
            await query.edit_message_text(
                f"🔄 *Reset to Default Configuration*\n\n"
                f"Monitoring all {len(self.monitored_symbols)} default symbols\n\n"
                "_Trading strategy will monitor top 50 cryptocurrencies_",
                parse_mode='Markdown'
            )
            
        elif data == "symbols_view":
            if self.monitored_symbols:
                sorted_symbols = sorted(self.monitored_symbols)
                text = f"📊 *Currently Monitored Symbols ({len(self.monitored_symbols)}):*\n\n"
                
                # Show in groups of 10 for better readability
                for i in range(0, len(sorted_symbols), 10):
                    group = sorted_symbols[i:i+10]
                    text += f"• {', '.join(group)}\n"
                    
                text += f"\n_Total: {len(self.monitored_symbols)} symbols_"
            else:
                text = "📊 No symbols currently monitored"
            await query.edit_message_text(text, parse_mode='Markdown')
    
    async def send_notification(self, chat_id: int, message: str, parse_mode: str = 'Markdown', reply_markup=None):
        """Send notification to user with optional buttons"""
        try:
            if not self.application:
                logger.error("Telegram bot not initialized - cannot send notification")
                return False
            
            if not self.application.bot:
                logger.error("Telegram bot instance not available - cannot send notification")
                return False
            
            logger.debug(f"Sending notification to chat_id {chat_id}")
            await self.application.bot.send_message(
                chat_id=chat_id,
                text=message,
                parse_mode=parse_mode,
                reply_markup=reply_markup
            )
            logger.debug(f"✅ Notification sent successfully to chat_id {chat_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to send notification to chat_id {chat_id}: {e}", exc_info=True)
            return False
    
    async def send_trade_signal_notification(self, signal: Dict):
        """Send interactive trade signal notification"""
        from .formatters import format_trade_signal
        
        message = format_trade_signal(signal)
        
        # Add interactive buttons
        keyboard = [
            [
                InlineKeyboardButton("✅ Take Trade", callback_data=f"take_{signal['symbol']}_{signal['side']}"),
                InlineKeyboardButton("❌ Skip", callback_data="signal_skip")
            ],
            [
                InlineKeyboardButton("📊 View Chart", callback_data=f"chart_{signal['symbol']}")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        for chat_id in settings.telegram_allowed_chat_ids:
            await self.send_notification(chat_id, message, reply_markup=reply_markup)
    
    async def send_position_closed_notification(self, trade: Dict):
        """Send position closed notification with stats"""
        from .formatters import format_trade_closed
        
        message = format_trade_closed(trade)
        
        # Add buttons for follow-up actions
        keyboard = [
            [
                InlineKeyboardButton("📊 View Summary", callback_data="daily_summary"),
                InlineKeyboardButton("💹 All Positions", callback_data="positions_refresh")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        for chat_id in settings.telegram_allowed_chat_ids:
            await self.send_notification(chat_id, message, reply_markup=reply_markup)
    
    async def send_error_notification(self, error_type: str, error_message: str, critical: bool = False):
        """Send error notification with recovery options"""
        from .formatters import format_error
        
        message = format_error(error_type, error_message)
        
        # Add recovery buttons for critical errors
        if critical:
            keyboard = [
                [
                    InlineKeyboardButton("🚨 Emergency Stop", callback_data="emergency_confirm"),
                    InlineKeyboardButton("🔄 Restart Scanner", callback_data="scanner_restart")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
        else:
            reply_markup = None
        
        for chat_id in settings.telegram_allowed_chat_ids:
            await self.send_notification(chat_id, message, reply_markup=reply_markup)
    
    async def send_daily_summary(self):
        """Send daily trading summary"""
        from .formatters import format_daily_summary
        from datetime import datetime, timedelta
        
        # Calculate daily stats (this would come from your database)
        summary = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0,
            'total_fees': 0,
            'best_symbol': 'N/A',
            'best_pnl': 0,
            'worst_symbol': 'N/A',
            'worst_pnl': 0
        }
        
        # Get actual data if engine is available
        if hasattr(self, 'trading_engine') and self.trading_engine:
            if hasattr(self.trading_engine, 'get_daily_stats'):
                summary = await self.trading_engine.get_daily_stats()
        
        message = format_daily_summary(summary)
        
        # Add action buttons
        keyboard = [
            [
                InlineKeyboardButton("📊 Full Report", callback_data="report_full"),
                InlineKeyboardButton("📈 Backtest", callback_data="bt_BTCUSDT_15_30")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        for chat_id in settings.telegram_allowed_chat_ids:
            await self.send_notification(chat_id, message, reply_markup=reply_markup)
    
    async def run_webhook(self, webhook_url: str, secret_token: str):
        """Run bot with webhook"""
        await self.application.initialize()
        await self.application.start()
        await self.application.updater.start_webhook(
            listen="0.0.0.0",
            port=8443,
            url_path=settings.telegram_bot_token,
            webhook_url=webhook_url,
            secret_token=secret_token
        )
        logger.info(f"Bot running with webhook at {webhook_url}")
    
    async def run_polling(self):
        """Run bot with long polling"""
        max_retries = 3
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                await self.application.initialize()
                await self.application.start()
                await self.application.updater.start_polling(
                    drop_pending_updates=True,
                    allowed_updates=["message", "callback_query"]
                )
                logger.info("Bot running with polling")
                return  # Success, exit the retry loop
            except Exception as e:
                logger.error(f"Failed to start bot polling (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error("Failed to start bot after all retries")
                    raise
    
    async def cmd_emergency(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Emergency stop - disable trading and close all positions"""
        if not await self.check_authorization(update):
            return
        
        await update.message.reply_text(
            "🚨 **EMERGENCY STOP INITIATED**\n\n"
            "This will:\n"
            "1. Disable automated trading\n"
            "2. Close all open positions\n"
            "3. Cancel all pending orders\n\n"
            "Are you sure?",
            parse_mode='Markdown'
        )
        
        keyboard = [
            [
                InlineKeyboardButton("🚨 CONFIRM EMERGENCY", callback_data="emergency_confirm"),
                InlineKeyboardButton("❌ Cancel", callback_data="emergency_cancel")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text("Confirm emergency stop:", reply_markup=reply_markup)
    
    async def cmd_panic(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Panic mode - stop all activity but preserve positions"""
        if not await self.check_authorization(update):
            return
        
        # Immediately disable trading in both bot and engine
        self.trading_enabled = True  # Auto-start enabled
        
        if hasattr(self, 'trading_engine') and self.trading_engine:
            # Disable trading in engine
            self.trading_engine.trading_enabled = False
            self.trading_engine.emergency_stop = True
            
            # Stop scanner if available
            if hasattr(self.trading_engine, 'scanner_task'):
                self.trading_engine.scanner_task.cancel()
        
        await update.message.reply_text(
            "😱 **PANIC MODE ACTIVATED**\n\n"
            "✅ Trading disabled\n"
            "✅ Scanner stopped\n"
            "✅ New signals blocked\n"
            "✅ Emergency stop engaged\n"
            "ℹ️ Positions preserved\n\n"
            "Use /enable to resume trading",
            parse_mode='Markdown'
        )
        logger.warning("Panic mode activated by user")
    
    async def cmd_closeall(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Close all open positions immediately"""
        if not await self.check_authorization(update):
            return
        
        positions = await self.bybit_client.get_positions()
        open_positions = [p for p in positions if float(p.get('size', 0)) > 0]
        
        if not open_positions:
            await update.message.reply_text("📊 No open positions to close")
            return
        
        text = f"⚠️ **Close All Positions**\n\n"
        text += f"Found {len(open_positions)} open positions\n\n"
        for pos in open_positions[:5]:  # Show first 5
            text += f"• {pos['symbol']}: {pos['side']} {pos['size']}\n"
        if len(open_positions) > 5:
            text += f"• ... and {len(open_positions) - 5} more\n"
        
        keyboard = [
            [
                InlineKeyboardButton("✅ Close All", callback_data="closeall_confirm"),
                InlineKeyboardButton("❌ Cancel", callback_data="closeall_cancel")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text(text, parse_mode='Markdown', reply_markup=reply_markup)
    
    async def cmd_ml(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """ML model status and control"""
        if not await self.check_authorization(update):
            return
        
        text = "🤖 **ML Model Control**\n\n"
        
        # Check if ML trainer exists
        if hasattr(self, 'trading_engine') and self.trading_engine:
            if hasattr(self.trading_engine, 'ml_trainer'):
                ml_trainer = self.trading_engine.ml_trainer
                text += "✅ ML System Active\n\n"
                text += f"Models: RF, GB, XGB, MLP\n"
                text += f"Features: 27 technical indicators\n"
                text += f"Training: Continuous learning\n\n"
                
                # Get last training info if available
                if hasattr(ml_trainer, 'last_training_time'):
                    text += f"Last trained: {ml_trainer.last_training_time}\n"
                if hasattr(ml_trainer, 'training_accuracy'):
                    text += f"Accuracy: {ml_trainer.training_accuracy:.2%}\n"
            else:
                text += "⚠️ ML System Not Found\n"
        else:
            text += "⚠️ Engine Not Connected\n"
        
        keyboard = [
            [
                InlineKeyboardButton("🔄 Retrain Models", callback_data="ml_retrain"),
                InlineKeyboardButton("📊 Model Stats", callback_data="ml_stats")
            ],
            [
                InlineKeyboardButton("⏸️ Pause ML", callback_data="ml_pause"),
                InlineKeyboardButton("▶️ Resume ML", callback_data="ml_resume")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text(text, parse_mode='Markdown', reply_markup=reply_markup)
    
    async def cmd_retrain(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Retrain ML models"""
        if not await self.check_authorization(update):
            return
        
        await update.message.reply_text("🔄 Starting ML model retraining...")
        
        # Trigger retraining
        if hasattr(self, 'trading_engine') and self.trading_engine:
            if hasattr(self.trading_engine, 'ml_trainer'):
                try:
                    asyncio.create_task(self.trading_engine.ml_trainer.train_models())
                    await update.message.reply_text(
                        "✅ ML retraining started\n"
                        "This may take several minutes..."
                    )
                except Exception as e:
                    await update.message.reply_text(f"❌ Retraining failed: {str(e)}")
            else:
                await update.message.reply_text("❌ ML trainer not available")
        else:
            await update.message.reply_text("❌ Trading engine not connected")
    
    async def cmd_zones(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """View active supply/demand zones"""
        if not await self.check_authorization(update):
            return
        
        text = "📊 **Active Supply/Demand Zones**\n\n"
        
        # Get zones from strategy
        total_zones = 0
        zone_summary = {}
        
        if hasattr(self.strategy, 'zones'):
            for symbol in list(self.monitored_symbols)[:5]:  # Show top 5 symbols
                if symbol in self.strategy.zones:
                    zones = self.strategy.zones[symbol]
                    if zones:
                        supply_zones = [z for z in zones if z.get('type') == 'supply']
                        demand_zones = [z for z in zones if z.get('type') == 'demand']
                        zone_summary[symbol] = {
                            'supply': len(supply_zones),
                            'demand': len(demand_zones)
                        }
                        total_zones += len(zones)
        
        if zone_summary:
            for symbol, counts in zone_summary.items():
                text += f"**{symbol}**\n"
                text += f"  🔴 Supply: {counts['supply']} zones\n"
                text += f"  🟢 Demand: {counts['demand']} zones\n\n"
            text += f"_Total: {total_zones} zones across {len(zone_summary)} symbols_"
        else:
            text += "No active zones detected\n\n"
            text += "_Zones are updated every scan cycle_"
        
        await update.message.reply_text(text, parse_mode='Markdown')
    
    async def cmd_signals(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """View recent signals"""
        if not await self.check_authorization(update):
            return
        
        text = "📡 **Recent Trading Signals**\n\n"
        
        # Get signal queue stats
        try:
            from ..utils.signal_queue import signal_queue
            stats = await signal_queue.get_stats()
            
            text += f"📊 Queue Status:\n"
            text += f"• Pending: {stats.get('queue_size', 0)}\n"
            text += f"• Processing: {stats.get('processing', 0)}\n"
            text += f"• Processed: {stats.get('processed_total', 0)}\n\n"
            
            # Get recent signals if available
            recent = stats.get('recent_signals', [])
            if recent:
                text += "🕐 Recent Signals:\n"
                for signal in recent[:5]:
                    text += f"• {signal.get('symbol', 'N/A')}: {signal.get('side', 'N/A')} @ {signal.get('time', 'N/A')}\n"
            else:
                text += "_No recent signals_"
        except:
            text += "⚠️ Signal queue unavailable"
        
        await update.message.reply_text(text, parse_mode='Markdown')
    
    async def cmd_scanner(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Scanner status and control"""
        if not await self.check_authorization(update):
            return
        
        text = "🔍 **Symbol Scanner Status**\n\n"
        
        # Get scanner status
        if hasattr(self, 'trading_engine') and self.trading_engine:
            if hasattr(self.trading_engine, 'scanner'):
                text += "✅ Scanner Active\n\n"
                text += f"Symbols: {len(self.monitored_symbols)}\n"
                text += f"Batch Size: 5\n"
                text += f"Scan Interval: 5 seconds\n\n"
                
                # Check if scanner task is running
                if hasattr(self.trading_engine, 'scanner_task'):
                    if self.trading_engine.scanner_task and not self.trading_engine.scanner_task.done():
                        text += "Status: 🟢 Running\n"
                    else:
                        text += "Status: 🔴 Stopped\n"
            else:
                text += "⚠️ Scanner Not Found\n"
        else:
            text += "⚠️ Engine Not Connected\n"
        
        keyboard = [
            [
                InlineKeyboardButton("▶️ Start Scanner", callback_data="scanner_start"),
                InlineKeyboardButton("⏸️ Stop Scanner", callback_data="scanner_stop")
            ],
            [
                InlineKeyboardButton("🔄 Restart Scanner", callback_data="scanner_restart")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text(text, parse_mode='Markdown', reply_markup=reply_markup)
    
    async def cmd_phase(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Scaling phase management"""
        if not await self.check_authorization(update):
            return
        
        from ..config_modules.scaling_config import scaling_config
        
        current_phase = scaling_config.CURRENT_PHASE
        current_config = scaling_config.get_current_config()
        
        text = f"📈 **Scaling Phase Management**\n\n"
        text += f"Current Phase: {current_phase}\n"
        text += f"Description: {current_config['description']}\n"
        text += f"Symbol Count: {current_config['symbol_count']}\n"
        text += f"Batch Size: {current_config['batch_size']}\n"
        text += f"Scan Delay: {current_config['scan_delay']}s\n\n"
        
        text += "**All Phases:**\n"
        for phase_num, phase_config in scaling_config.PHASES.items():
            emoji = "👉" if phase_num == current_phase else "  "
            text += f"{emoji} Phase {phase_num}: {phase_config['symbol_count']} symbols\n"
        
        keyboard = [
            [
                InlineKeyboardButton("⬅️ Previous", callback_data="phase_prev"),
                InlineKeyboardButton("➡️ Next", callback_data="phase_next")
            ],
            [
                InlineKeyboardButton(f"Phase 0 (5)", callback_data="phase_0"),
                InlineKeyboardButton(f"Phase 1 (20)", callback_data="phase_1"),
                InlineKeyboardButton(f"Phase 2 (50)", callback_data="phase_2")
            ],
            [
                InlineKeyboardButton(f"Phase 3 (100)", callback_data="phase_3"),
                InlineKeyboardButton(f"Phase 4 (200)", callback_data="phase_4"),
                InlineKeyboardButton(f"Phase 5 (300)", callback_data="phase_5")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text(text, parse_mode='Markdown', reply_markup=reply_markup)
    
    async def cmd_logs(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """View recent activity logs"""
        if not await self.check_authorization(update):
            return
        
        import os
        from datetime import datetime, timedelta
        
        text = "📋 **Recent Activity Logs**\n\n"
        
        # Try to read last 20 lines from log file
        log_file = "/tmp/trading_bot.log"
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    recent_lines = lines[-20:] if len(lines) > 20 else lines
                    
                    # Filter for important messages
                    important_keywords = ['SIGNAL', 'TRADE', 'ERROR', 'Zone', 'Position', 'ML']
                    filtered_lines = []
                    
                    for line in recent_lines:
                        if any(keyword in line for keyword in important_keywords):
                            # Simplify the line
                            parts = line.split(' - ', 2)
                            if len(parts) >= 3:
                                time_part = parts[0].split()[-1] if parts[0] else ''
                                message = parts[2].strip() if len(parts) > 2 else line.strip()
                                filtered_lines.append(f"{time_part}: {message[:50]}...")
                    
                    if filtered_lines:
                        text += "\n".join(filtered_lines[-10:])  # Show last 10 important messages
                    else:
                        text += "No significant events in recent logs"
            except Exception as e:
                text += f"Error reading logs: {str(e)}"
        else:
            text += "Log file not found\n\n"
            text += "_Logs are stored temporarily and reset on restart_"
        
        await update.message.reply_text(text, parse_mode='Markdown')