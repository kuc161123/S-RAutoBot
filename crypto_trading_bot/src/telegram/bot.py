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
from ..api.bybit_client import BybitClient
from ..strategy.supply_demand import SupplyDemandStrategy
from ..db.models import User, Trade, BacktestResult
from ..backtesting.backtest_engine import BacktestEngine
from .formatters import format_status, format_positions, format_backtest_result

logger = structlog.get_logger(__name__)

class TradingBot:
    """Telegram bot for crypto trading"""
    
    def __init__(self, bybit_client: BybitClient, strategy: SupplyDemandStrategy):
        self.bybit_client = bybit_client
        self.strategy = strategy
        self.application = None
        self.trading_enabled = False
        self.monitored_symbols = set()
        
    async def initialize(self):
        """Initialize the bot"""
        # Build application
        self.application = Application.builder().token(settings.telegram_bot_token).build()
        
        # Add command handlers
        self.application.add_handler(CommandHandler("start", self.cmd_start))
        self.application.add_handler(CommandHandler("help", self.cmd_help))
        self.application.add_handler(CommandHandler("enable", self.cmd_enable))
        self.application.add_handler(CommandHandler("disable", self.cmd_disable))
        self.application.add_handler(CommandHandler("status", self.cmd_status))
        self.application.add_handler(CommandHandler("symbols", self.cmd_symbols))
        self.application.add_handler(CommandHandler("margin", self.cmd_margin))
        self.application.add_handler(CommandHandler("leverage", self.cmd_leverage))
        self.application.add_handler(CommandHandler("risk", self.cmd_risk))
        self.application.add_handler(CommandHandler("strategy", self.cmd_strategy))
        self.application.add_handler(CommandHandler("backtest", self.cmd_backtest))
        self.application.add_handler(CommandHandler("positions", self.cmd_positions))
        self.application.add_handler(CommandHandler("logs", self.cmd_logs))
        
        # Add callback query handler for inline keyboards
        self.application.add_handler(CallbackQueryHandler(self.handle_callback))
        
        # Add message handler for keyboard buttons
        self.application.add_handler(MessageHandler(
            filters.TEXT & ~filters.COMMAND, 
            self.handle_keyboard_button
        ))
        
        logger.info("Telegram bot initialized")
    
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
        else:
            await update.message.reply_text(
                "Unknown command. Use /help to see available commands."
            )
    
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
            "🤖 *Welcome to Crypto Trading Bot!*\n\n"
            "This bot trades crypto futures on Bybit using Supply & Demand strategy.\n\n"
            "⚠️ *Risk Warning:* Futures trading involves significant risk. "
            "Only trade with funds you can afford to lose.\n\n"
            "📋 *Quick Start:*\n"
            "1. Configure risk with /risk\n"
            "2. Select symbols with /symbols\n"
            "3. Enable trading with /enable\n\n"
            "Use /help to see all commands."
        )
        
        # Create main menu keyboard
        keyboard = [
            [KeyboardButton("📊 Status"), KeyboardButton("💹 Positions")],
            [KeyboardButton("🎯 Enable"), KeyboardButton("🛑 Disable")],
            [KeyboardButton("⚙️ Settings"), KeyboardButton("📈 Backtest")]
        ]
        reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
        
        await update.message.reply_text(
            welcome_text,
            parse_mode='Markdown',
            reply_markup=reply_markup
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
            "⚙️ *Configuration:*\n"
            "/symbols - Manage traded symbols\n"
            "/margin <symbol> <cross|isolated> - Set margin mode\n"
            "/leverage <symbol> <value> - Set leverage (1-125x)\n"
            "/risk - Configure risk parameters\n"
            "/strategy - Configure strategy settings\n\n"
            "📊 *Analysis:*\n"
            "/backtest <symbol> <timeframe> <days> - Run backtest\n"
            "/logs - View recent activity logs\n\n"
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
        
        self.trading_enabled = False
        await update.message.reply_text(
            "🛑 *Trading Disabled*\n\n"
            "Automated trading has been stopped.\n"
            "Existing positions will remain open.",
            parse_mode='Markdown'
        )
        logger.info("Trading disabled by user")
    
    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command"""
        if not await self.check_authorization(update):
            return
        
        try:
            # Get account info
            account_info = await self.bybit_client.get_account_info()
            
            # Get positions
            positions = await self.bybit_client.get_positions()
            open_positions = [p for p in positions if float(p['size']) > 0]
            
            # Get active zones count
            total_zones = sum(len(self.strategy.get_active_zones(s)) 
                            for s in self.monitored_symbols)
            
            # Format status message
            status_text = format_status(
                self.trading_enabled,
                account_info,
                open_positions,
                self.monitored_symbols,
                total_zones
            )
            
            await update.message.reply_text(status_text, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            await update.message.reply_text("❌ Error fetching status")
    
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
            logger.error(f"Backtest error: {e}")
            await update.message.reply_text(f"❌ Backtest failed: {str(e)}")
    
    async def cmd_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /positions command"""
        if not await self.check_authorization(update):
            return
        
        try:
            positions = await self.bybit_client.get_positions()
            open_positions = [p for p in positions if float(p['size']) > 0]
            
            if not open_positions:
                await update.message.reply_text("📊 No open positions")
                return
            
            text = format_positions(open_positions)
            await update.message.reply_text(text, parse_mode='Markdown')
            
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
            await query.edit_message_text(
                "✅ *Trading Enabled*\n\n"
                "The bot is now actively monitoring and trading.",
                parse_mode='Markdown'
            )
            logger.info("Trading enabled by user")
            
        elif data == "enable_cancel":
            await query.edit_message_text("❌ Trading enable cancelled")
            
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
                    logger.error(f"Backtest error: {e}")
                    await query.message.reply_text(f"❌ Backtest failed: {str(e)}")
    
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
    
    async def send_notification(self, chat_id: int, message: str, parse_mode: str = 'Markdown'):
        """Send notification to user"""
        try:
            await self.application.bot.send_message(
                chat_id=chat_id,
                text=message,
                parse_mode=parse_mode
            )
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
    
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
        await self.application.initialize()
        await self.application.start()
        await self.application.updater.start_polling()
        logger.info("Bot running with polling")