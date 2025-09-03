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
        self.app.add_handler(CommandHandler("risk", self.show_risk))
        self.app.add_handler(CommandHandler("set_risk", self.set_risk))
        self.app.add_handler(CommandHandler("risk_percent", self.set_risk_percent))
        self.app.add_handler(CommandHandler("risk_usd", self.set_risk_usd))
        self.app.add_handler(CommandHandler("status", self.status))
        self.app.add_handler(CommandHandler("panic_close", self.panic_close))
        self.app.add_handler(CommandHandler("balance", self.balance))
        self.app.add_handler(CommandHandler("health", self.health))
        self.app.add_handler(CommandHandler("symbols", self.symbols))
        self.app.add_handler(CommandHandler("dashboard", self.dashboard))
        self.app.add_handler(CommandHandler("analysis", self.analysis))
        self.app.add_handler(CommandHandler("stats", self.stats))
        self.app.add_handler(CommandHandler("recent", self.recent_trades))
        self.app.add_handler(CommandHandler("ml", self.ml_stats))
        self.app.add_handler(CommandHandler("ml_stats", self.ml_stats))
        
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
📚 *Bot Commands*

📊 *Monitoring:*
/dashboard - Complete bot overview
/health - Bot status & analysis activity
/status - Open positions
/balance - Account balance
/symbols - List active trading pairs
/analysis [symbol] - Show analysis details

📈 *Statistics:*
/stats [days] - Trading statistics
/recent [limit] - Recent trades history
/ml or /ml\_stats - ML system status

⚙️ *Risk Management:*
/risk - Show current risk settings
/risk\_percent [value] - Set % risk (e.g., 2.5)
/risk\_usd [value] - Set USD risk (e.g., 100)
/set\_risk [amount] - Flexible (3% or 50)

⚙️ *Controls:*
/panic\_close [symbol] - Emergency close position

ℹ️ *Info:*
/start - Welcome message
/help - This help menu

📈 *Current Settings:*
• Risk per trade: {}
• Max leverage: {}x
• Timeframe: 15 minutes
• Strategy: S/R + Market Structure
""".format(
            f"{self.shared['risk'].risk_percent}%" if self.shared["risk"].use_percent_risk else f"${self.shared['risk'].risk_usd}",
            self.shared["risk"].max_leverage
        )
        await update.message.reply_text(help_text, parse_mode='Markdown')

    async def show_risk(self, update:Update, ctx:ContextTypes.DEFAULT_TYPE):
        """Show current risk settings"""
        try:
            risk = self.shared["risk"]
            
            # Get current balance if available
            balance_text = ""
            balance = None
            if "broker" in self.shared and hasattr(self.shared["broker"], "get_balance"):
                balance = self.shared["broker"].get_balance()
                if balance:
                    balance_text = f"💰 *Account Balance:* ${balance:.2f}\n"
            
            if risk.use_percent_risk:
                risk_amount = f"{risk.risk_percent}%"
                if balance_text and balance:
                    usd_amount = balance * (risk.risk_percent / 100)
                    risk_amount += f" (≈${usd_amount:.2f})"
                mode = "Percentage"
            else:
                risk_amount = f"${risk.risk_usd}"
                if balance_text and balance:
                    percent = (risk.risk_usd / balance) * 100
                    risk_amount += f" (≈{percent:.2f}%)"
                mode = "Fixed USD"
            
            msg = f"""📊 *Risk Management Settings*
            
{balance_text}⚙️ *Mode:* {mode}
💸 *Risk per trade:* {risk_amount}
📈 *Risk/Reward Ratio:* 1:{risk.rr if hasattr(risk, 'rr') else 2}

*Commands:*
`/risk_percent 2.5` - Set to 2.5%
`/risk_usd 100` - Set to $100
`/set_risk 3%` or `/set_risk 50` - Flexible"""
            
            await update.message.reply_text(msg, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error in show_risk: {e}")
            await update.message.reply_text("Error fetching risk settings")
    
    async def set_risk_percent(self, update:Update, ctx:ContextTypes.DEFAULT_TYPE):
        """Set risk as percentage of account"""
        try:
            if not ctx.args:
                await update.message.reply_text("Usage: /risk_percent 2.5")
                return
            
            value = float(ctx.args[0])
            
            # Validate
            if value <= 0:
                await update.message.reply_text("❌ Risk must be greater than 0%")
                return
            elif value > 10:
                await update.message.reply_text("⚠️ Risk cannot exceed 10% per trade")
                return
            elif value > 5:
                # Warning for high risk
                await update.message.reply_text(
                    f"⚠️ *High Risk Warning*\n\n"
                    f"Setting risk to {value}% is aggressive.\n"
                    f"Confirm with: `/set_risk {value}%`",
                    parse_mode='Markdown'
                )
                return
            
            # Apply the change
            self.shared["risk"].risk_percent = value
            self.shared["risk"].use_percent_risk = True
            
            # Calculate USD amount if balance available
            usd_info = ""
            if "broker" in self.shared and hasattr(self.shared["broker"], "get_balance"):
                balance = self.shared["broker"].get_balance()
                if balance:
                    usd_amount = balance * (value / 100)
                    usd_info = f" (≈${usd_amount:.2f} per trade)"
            
            await update.message.reply_text(
                f"✅ Risk updated to {value}% of account{usd_info}\n"
                f"Use `/risk` to view full settings"
            )
            logger.info(f"Risk updated to {value}% via Telegram")
            
        except ValueError:
            await update.message.reply_text("❌ Invalid number. Example: /risk_percent 2.5")
        except Exception as e:
            logger.error(f"Error in set_risk_percent: {e}")
            await update.message.reply_text("Error updating risk percentage")
    
    async def set_risk_usd(self, update:Update, ctx:ContextTypes.DEFAULT_TYPE):
        """Set risk as fixed USD amount"""
        try:
            if not ctx.args:
                await update.message.reply_text("Usage: /risk_usd 100")
                return
            
            value = float(ctx.args[0])
            
            # Validate
            if value <= 0:
                await update.message.reply_text("❌ Risk must be greater than $0")
                return
            elif value > 1000:
                await update.message.reply_text("⚠️ Risk cannot exceed $1000 per trade")
                return
            
            # Check if this is too high relative to balance
            percent_info = ""
            if "broker" in self.shared and hasattr(self.shared["broker"], "get_balance"):
                balance = self.shared["broker"].get_balance()
                if balance:
                    percent = (value / balance) * 100
                    percent_info = f" (≈{percent:.2f}% of account)"
                    
                    if percent > 5:
                        await update.message.reply_text(
                            f"⚠️ *High Risk Warning*\n\n"
                            f"${value} is {percent:.1f}% of your ${balance:.0f} account.\n"
                            f"Confirm with: `/set_risk {value}`",
                            parse_mode='Markdown'
                        )
                        return
            
            # Apply the change
            self.shared["risk"].risk_usd = value
            self.shared["risk"].use_percent_risk = False
            
            await update.message.reply_text(
                f"✅ Risk updated to ${value} per trade{percent_info}\n"
                f"Use `/risk` to view full settings"
            )
            logger.info(f"Risk updated to ${value} fixed via Telegram")
            
        except ValueError:
            await update.message.reply_text("❌ Invalid number. Example: /risk_usd 100")
        except Exception as e:
            logger.error(f"Error in set_risk_usd: {e}")
            await update.message.reply_text("Error updating risk amount")
    
    async def set_risk(self, update:Update, ctx:ContextTypes.DEFAULT_TYPE):
        """Set risk amount per trade - percentage (1%) or fixed USD (50)"""
        try:
            if not ctx.args:
                # Show current settings if no args
                await self.show_risk(update, ctx)
                return
            
            risk_str = ctx.args[0]
            
            if risk_str.endswith('%'):
                # Percentage-based risk
                v = float(risk_str.rstrip('%'))
                if v <= 0 or v > 10:
                    await update.message.reply_text("Risk percentage must be between 0% and 10%")
                    return
                
                self.shared["risk"].risk_percent = v
                self.shared["risk"].use_percent_risk = True
                await update.message.reply_text(f"✅ Risk set to {v}% of account per trade")
                logger.info(f"Risk updated to {v}%")
            else:
                # Fixed USD risk
                v = float(risk_str)
                if v <= 0 or v > 1000:
                    await update.message.reply_text("Risk must be between $0 and $1000")
                    return
                
                self.shared["risk"].risk_usd = v
                self.shared["risk"].use_percent_risk = False
                await update.message.reply_text(f"✅ Risk set to ${v} per trade")
                logger.info(f"Risk updated to ${v} fixed")
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
                msg = "📊 *Position Status*\n\n"
                msg += "No open positions\n\n"
                msg += f"Risk per trade: ${self.shared['risk'].risk_usd}\n"
                msg += f"Max positions: Unlimited\n"
                msg += f"\n_Bot is actively scanning {len(self.shared.get('frames', {}))} symbols_"
                await update.message.reply_text(msg, parse_mode='Markdown')
                return
                
            msg = "📊 *Open Positions*\n"
            msg += "━" * 20 + "\n\n"
            
            for sym, p in bk.items():
                emoji = "🟢" if p.side == "long" else "🔴"
                
                # Calculate current P&L if we have price data
                frames = self.shared.get("frames", {})
                pnl_str = ""
                if sym in frames and len(frames[sym]) > 0:
                    current_price = frames[sym]['close'].iloc[-1]
                    if p.side == "long":
                        pnl = (current_price - p.entry) * p.qty
                        pnl_pct = ((current_price - p.entry) / p.entry) * 100
                    else:
                        pnl = (p.entry - current_price) * p.qty
                        pnl_pct = ((p.entry - current_price) / p.entry) * 100
                    
                    pnl_emoji = "🟢" if pnl > 0 else "🔴"
                    pnl_str = f"\n  P&L: {pnl_emoji} ${pnl:.2f} ({pnl_pct:+.2f}%)"
                
                msg += f"{emoji} *{sym}* ({p.side.upper()})\n"
                msg += f"  Entry: {p.entry:.4f}\n"
                msg += f"  Size: {p.qty}\n"
                msg += f"  SL: {p.sl:.4f} | TP: {p.tp:.4f}"
                msg += pnl_str + "\n\n"
                
            msg += f"*Total positions:* {len(bk)}\n"
            msg += f"*Risk exposure:* ${len(bk) * self.shared['risk'].risk_usd}"
            
            await update.message.reply_text(msg, parse_mode='Markdown')
            
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
    
    async def symbols(self, update:Update, ctx:ContextTypes.DEFAULT_TYPE):
        """Show list of active trading symbols"""
        try:
            frames = self.shared.get("frames", {})
            
            if not frames:
                await update.message.reply_text("No symbols loaded yet")
                return
            
            msg = "📈 *Active Trading Pairs*\n\n"
            
            # Group symbols for better display
            symbols_list = list(frames.keys())
            
            # Show in groups of 5
            for i in range(0, len(symbols_list), 5):
                group = symbols_list[i:i+5]
                msg += " • ".join(group) + "\n"
            
            msg += f"\n*Total:* {len(symbols_list)} symbols"
            msg += "\n*Timeframe:* 15 minutes"
            msg += "\n*Strategy:* Support/Resistance Breakout"
            
            await update.message.reply_text(msg, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error in symbols: {e}")
            await update.message.reply_text("Error getting symbols")
    
    async def dashboard(self, update:Update, ctx:ContextTypes.DEFAULT_TYPE):
        """Show complete bot dashboard"""
        try:
            import datetime
            
            # Gather all data
            frames = self.shared.get("frames", {})
            book = self.shared.get("book")
            risk = self.shared.get("risk")
            last_analysis = self.shared.get("last_analysis", {})
            
            msg = "🎯 *Trading Bot Dashboard*\n"
            msg += "━" * 25 + "\n\n"
            
            # System Status
            msg += "⚡ *System Status*\n"
            if frames:
                msg += "• Status: ✅ Online\n"
                msg += f"• Symbols: {len(frames)} active\n"
            else:
                msg += "• Status: ⏳ Starting up\n"
            
            # Get balance
            broker = self.shared.get("broker")
            if broker:
                balance = broker.get_balance()
                if balance:
                    msg += f"• Balance: ${balance:.2f} USDT\n"
            
            msg += "\n"
            
            # Trading Settings
            msg += "⚙️ *Trading Settings*\n"
            if risk.use_percent_risk:
                msg += f"• Risk per trade: {risk.risk_percent}%\n"
            else:
                msg += f"• Risk per trade: ${risk.risk_usd}\n"
            msg += f"• Max leverage: {risk.max_leverage}x\n"
            msg += f"• Timeframe: 15 minutes\n"
            msg += "\n"
            
            # ML Status (Brief)
            ml_scorer = self.shared.get("ml_scorer")
            if ml_scorer:
                try:
                    ml_stats = ml_scorer.get_ml_stats()
                    msg += "🤖 *ML System*\n"
                    if ml_stats['is_trained']:
                        msg += f"• Status: ✅ Active\n"
                        msg += f"• Trained on: {ml_stats['completed_trades']} trades\n"
                    else:
                        progress = (ml_stats['completed_trades'] / 200) * 100
                        msg += f"• Status: 📊 Learning ({progress:.0f}%)\n"
                        msg += f"• Trades: {ml_stats['completed_trades']}/200\n"
                    msg += "• Use /ml for details\n"
                except:
                    pass
            msg += "\n"
            
            # Positions
            msg += "📊 *Positions*\n"
            if book and book.positions:
                for sym, pos in book.positions.items():
                    emoji = "🟢" if pos.side == "long" else "🔴"
                    msg += f"{emoji} {sym}: {pos.side.upper()}\n"
                    msg += f"  Entry: {pos.entry:.4f}\n"
                    msg += f"  Qty: {pos.qty}\n"
            else:
                msg += "• No open positions\n"
            msg += "\n"
            
            # Recent Analysis
            msg += "🔍 *Recent Analysis*\n"
            if last_analysis:
                now = datetime.datetime.now()
                recent = sorted(last_analysis.items(), key=lambda x: x[1], reverse=True)[:3]
                for sym, timestamp in recent:
                    time_ago = (now - timestamp).total_seconds()
                    if time_ago < 60:
                        msg += f"• {sym}: {int(time_ago)}s ago\n"
                    else:
                        msg += f"• {sym}: {int(time_ago/60)}m ago\n"
            else:
                msg += "• Waiting for data...\n"
            
            msg += "\n"
            
            # Next Analysis
            now = datetime.datetime.now()
            next_15 = (now.minute // 15 + 1) * 15
            if next_15 >= 60:
                next_time = now.replace(minute=0, second=0) + datetime.timedelta(hours=1)
            else:
                next_time = now.replace(minute=next_15, second=0)
            
            time_until = (next_time - now).total_seconds()
            msg += f"⏰ *Next Analysis:* {int(time_until/60)}m {int(time_until%60)}s\n"
            
            msg += "\n_Use /help for all commands_"
            
            await update.message.reply_text(msg, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error in dashboard: {e}")
            await update.message.reply_text("Error generating dashboard")
    
    async def analysis(self, update:Update, ctx:ContextTypes.DEFAULT_TYPE):
        """Show recent analysis details for symbols"""
        try:
            frames = self.shared.get("frames", {})
            analysis_log = self.shared.get("analysis_log", {})
            
            if not frames:
                await update.message.reply_text("No data available yet")
                return
            
            msg = "🔍 *Recent Analysis Details*\n"
            msg += "━" * 20 + "\n\n"
            
            # Get symbol from args or show first 5
            if ctx.args:
                symbols = [ctx.args[0].upper()]
            else:
                symbols = list(frames.keys())[:5]
            
            for symbol in symbols:
                if symbol not in frames:
                    continue
                    
                df = frames[symbol]
                if df is None or len(df) < 50:
                    msg += f"*{symbol}*: Insufficient data\n\n"
                    continue
                
                # Get last price and check for recent highs/lows
                last_price = df['close'].iloc[-1]
                recent_high = df['high'].iloc[-20:].max()
                recent_low = df['low'].iloc[-20:].min()
                
                msg += f"*{symbol}*\n"
                msg += f"• Price: {last_price:.4f}\n"
                msg += f"• 20-bar High: {recent_high:.4f}\n"
                msg += f"• 20-bar Low: {recent_low:.4f}\n"
                
                # Check if we have analysis logs
                if symbol in analysis_log:
                    log_entry = analysis_log[symbol]
                    msg += f"• Structure: {log_entry.get('structure', 'Unknown')}\n"
                    msg += f"• Signal: {log_entry.get('signal', 'None')}\n"
                else:
                    # Simple trend detection
                    sma20 = df['close'].iloc[-20:].mean()
                    if last_price > sma20 * 1.01:
                        msg += f"• Trend: Bullish (above SMA20: {sma20:.4f})\n"
                    elif last_price < sma20 * 0.99:
                        msg += f"• Trend: Bearish (below SMA20: {sma20:.4f})\n"
                    else:
                        msg += f"• Trend: Neutral (near SMA20: {sma20:.4f})\n"
                
                msg += "\n"
            
            msg += "_Use /analysis SYMBOL for specific pair_"
            
            await update.message.reply_text(msg, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error in analysis: {e}")
            await update.message.reply_text("Error getting analysis")
    
    async def stats(self, update:Update, ctx:ContextTypes.DEFAULT_TYPE):
        """Show trading statistics"""
        try:
            # Get trade tracker from shared
            tracker = self.shared.get("trade_tracker")
            if not tracker:
                await update.message.reply_text("Statistics tracking not initialized yet")
                return
            
            # Parse arguments for time period
            days = None
            if ctx.args:
                try:
                    days = int(ctx.args[0])
                except ValueError:
                    pass
            
            # Get formatted statistics
            msg = tracker.format_stats_message(days)
            await update.message.reply_text(msg, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error in stats: {e}")
            await update.message.reply_text("Error getting statistics")
    
    async def recent_trades(self, update:Update, ctx:ContextTypes.DEFAULT_TYPE):
        """Show recent trades"""
        try:
            # Get trade tracker from shared
            tracker = self.shared.get("trade_tracker")
            if not tracker:
                await update.message.reply_text("Trade tracking not initialized yet")
                return
            
            # Parse arguments for limit
            limit = 5
            if ctx.args:
                try:
                    limit = min(20, int(ctx.args[0]))  # Max 20 recent trades
                except ValueError:
                    pass
            
            # Get formatted recent trades
            msg = tracker.format_recent_trades(limit)
            await update.message.reply_text(msg, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error in recent_trades: {e}")
            await update.message.reply_text("Error getting recent trades")
    
    async def ml_stats(self, update:Update, ctx:ContextTypes.DEFAULT_TYPE):
        """Show ML system statistics and status"""
        try:
            msg = "🤖 *ML Signal Scoring System*\n"
            msg += "━" * 25 + "\n\n"
            
            # Check if ML scorer is available
            ml_scorer = self.shared.get("ml_scorer")
            
            if not ml_scorer:
                msg += "❌ *ML System: Not Available*\n\n"
                msg += "ML scoring is either:\n"
                msg += "• Disabled in config\n"
                msg += "• Not initialized yet\n\n"
                msg += "To enable: Set `use_ml_scoring: true` in config"
            else:
                # Get ML stats
                try:
                    stats = ml_scorer.get_ml_stats()
                    
                    # Status
                    if stats['is_trained']:
                        msg += "✅ *Status: Active & Learning*\n\n"
                    else:
                        msg += "📊 *Status: Collecting Data*\n\n"
                    
                    # Progress
                    msg += "📈 *Learning Progress*\n"
                    msg += f"• Completed trades: {stats['completed_trades']}\n"
                    
                    if not stats['is_trained']:
                        trades_needed = stats.get('trades_needed', 200)
                        progress_pct = (stats['completed_trades'] / 200) * 100
                        msg += f"• Progress: {progress_pct:.1f}%\n"
                        msg += f"• Trades needed: {trades_needed}\n"
                        msg += "\n⏳ ML will activate after 200 trades\n\n"
                    else:
                        msg += f"• Model trained on: {stats['last_train_count']} trades\n"
                        msg += f"• Model type: {stats['model_type']}\n"
                        if 'recent_accuracy' in stats:
                            msg += f"• Recent accuracy: {stats['recent_accuracy']*100:.1f}%\n"
                        msg += "\n"
                    
                    # Settings
                    msg += "⚙️ *Configuration*\n"
                    msg += f"• Enabled: {'Yes' if stats['enabled'] else 'No'}\n"
                    msg += f"• Min score threshold: {stats['min_score_threshold']}/100\n"
                    msg += f"• Ensemble models: 3 (RF, GB, NN)\n"
                    msg += "\n"
                    
                    # Features analyzed
                    msg += "🔍 *Features Analyzed*\n"
                    msg += "• Trend strength & alignment\n"
                    msg += "• Volume patterns\n"
                    msg += "• Support/Resistance strength\n"
                    msg += "• Pullback quality\n"
                    msg += "• Market volatility\n"
                    msg += "• Time of day patterns\n"
                    msg += "\n"
                    
                    # How it works
                    if stats['is_trained']:
                        msg += "💡 *How It's Working*\n"
                        msg += "• Scoring every signal 0-100\n"
                        msg += f"• Filtering signals below {stats['min_score_threshold']}\n"
                        msg += "• Learning from trade outcomes\n"
                        msg += "• Adapting to market changes\n"
                    else:
                        msg += "💡 *What's Happening*\n"
                        msg += "• Collecting data from all signals\n"
                        msg += "• Recording trade outcomes\n"
                        msg += "• Building pattern database\n"
                        msg += f"• {200 - stats['completed_trades']} more trades to activate\n"
                    
                except Exception as e:
                    logger.error(f"Error getting ML stats: {e}")
                    msg += "⚠️ Error retrieving ML statistics\n"
            
            await update.message.reply_text(msg, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error in ml_stats: {e}")
            await update.message.reply_text("Error getting ML statistics")