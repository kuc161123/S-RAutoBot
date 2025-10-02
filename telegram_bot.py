from datetime import datetime, timezone

from telegram import Update
from telegram.constants import UpdateType, ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes
import telegram.error
import asyncio
import logging

from ml_signal_scorer_immediate import get_immediate_scorer
from ml_scorer_mean_reversion import get_mean_reversion_scorer
from enhanced_market_regime import get_enhanced_market_regime, get_regime_summary

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
        self.app.add_handler(CommandHandler("setrisk", self.set_risk))  # Alternative command name
        self.app.add_handler(CommandHandler("risk_percent", self.set_risk_percent))
        self.app.add_handler(CommandHandler("riskpercent", self.set_risk_percent))  # Alternative command name
        self.app.add_handler(CommandHandler("risk_usd", self.set_risk_usd))
        self.app.add_handler(CommandHandler("riskusd", self.set_risk_usd))  # Alternative command name
        self.app.add_handler(CommandHandler("ml_risk", self.ml_risk))
        self.app.add_handler(CommandHandler("ml_risk_range", self.ml_risk_range))
        self.app.add_handler(CommandHandler("mlriskrange", self.ml_risk_range))  # Alternative command name
        self.app.add_handler(CommandHandler("mlriskrank", self.ml_risk_range))  # Alternative command name
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
        self.app.add_handler(CommandHandler("mlrankings", self.ml_rankings))
        self.app.add_handler(CommandHandler("mlpatterns", self.ml_patterns))
        self.app.add_handler(CommandHandler("mlretrain", self.ml_retrain_info))
        self.app.add_handler(CommandHandler("reset_stats", self.reset_stats))
        self.app.add_handler(CommandHandler("phantom", self.phantom_stats))
        self.app.add_handler(CommandHandler("phantom_detail", self.phantom_detail))
        self.app.add_handler(CommandHandler("evolution", self.evolution_performance))
        self.app.add_handler(CommandHandler("force_retrain", self.force_retrain_ml))
        self.app.add_handler(CommandHandler("clusters", self.cluster_status))
        self.app.add_handler(CommandHandler("update_clusters", self.update_clusters))
        self.app.add_handler(CommandHandler("set_ml_threshold", self.set_ml_threshold))
        self.app.add_handler(CommandHandler("htf_sr", self.htf_sr_status))
        self.app.add_handler(CommandHandler("update_htf_sr", self.update_htf_sr))
        self.app.add_handler(CommandHandler("mr_ml", self.mr_ml_stats))
        self.app.add_handler(CommandHandler("mr_retrain", self.mr_retrain))
        self.app.add_handler(CommandHandler("enhanced_mr", self.enhanced_mr_stats))
        self.app.add_handler(CommandHandler("enhancedmr", self.enhanced_mr_stats))  # Alternative command name
        self.app.add_handler(CommandHandler("mr_phantom", self.mr_phantom_stats))
        self.app.add_handler(CommandHandler("mrphantom", self.mr_phantom_stats))  # Alternative command name
        self.app.add_handler(CommandHandler("parallel_performance", self.parallel_performance))
        self.app.add_handler(CommandHandler("parallelperformance", self.parallel_performance))  # Alternative command name
        self.app.add_handler(CommandHandler("regime_analysis", self.regime_analysis))
        self.app.add_handler(CommandHandler("regimeanalysis", self.regime_analysis))  # Alternative command name
        self.app.add_handler(CommandHandler("regime", self.regime_single))
        self.app.add_handler(CommandHandler("strategy_comparison", self.strategy_comparison))
        self.app.add_handler(CommandHandler("strategycomparison", self.strategy_comparison))  # Alternative command name
        self.app.add_handler(CommandHandler("system", self.system_status))
        self.app.add_handler(CommandHandler("training_status", self.training_status))
        self.app.add_handler(CommandHandler("trainingstatus", self.training_status))  # Alternative command name
        self.app.add_handler(CommandHandler("mlstatus", self.ml_stats))
        self.app.add_handler(CommandHandler("panicclose", self.panic_close))
        self.app.add_handler(CommandHandler("forceretrain", self.force_retrain_ml))

        self.running = False

    def _compute_risk_snapshot(self):
        """Return (per_trade_risk_usd, label) based on current configuration."""
        risk = self.shared.get("risk")
        last_balance = self.shared.get("last_balance")

        per_trade = float(getattr(risk, 'risk_usd', 0.0))
        label: str

        if getattr(risk, 'use_percent_risk', False) and last_balance:
            per_trade = last_balance * (getattr(risk, 'risk_percent', 0.0) / 100.0)
            label = f"{risk.risk_percent}% (~${per_trade:.2f})"
        elif getattr(risk, 'use_percent_risk', False):
            per_trade = float(getattr(risk, 'risk_usd', 0.0))
            label = f"{risk.risk_percent}% (fallback ${per_trade:.2f})"
        else:
            label = f"${per_trade:.2f}"

        if getattr(risk, 'use_ml_dynamic_risk', False):
            label += " (ML dynamic)"

        return per_trade, label

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
        """Send message to configured chat with retry on network errors"""
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                # Try with Markdown first
                await self.app.bot.send_message(chat_id=self.chat_id, text=text, parse_mode='Markdown')
                return  # Success, exit
            except telegram.error.BadRequest as e:
                if "can't parse entities" in str(e).lower():
                    # Markdown parsing failed, try with better escaping
                    logger.warning("Markdown parsing failed, trying with escaped text")
                    try:
                        # Escape common problematic characters
                        escaped_text = text.replace('_', '\\_').replace('*', '\\*').replace('[', '\\[').replace(']', '\\]').replace('`', '\\`')
                        await self.app.bot.send_message(chat_id=self.chat_id, text=escaped_text, parse_mode='Markdown')
                        return  # Success, exit
                    except:
                        # If still fails, send as plain text
                        logger.warning("Escaped markdown also failed, sending as plain text")
                        try:
                            await self.app.bot.send_message(chat_id=self.chat_id, text=text)
                            return  # Success, exit
                        except Exception as plain_e:
                            if attempt < max_retries - 1:
                                logger.warning(f"Plain text send failed (attempt {attempt + 1}/{max_retries}): {plain_e}")
                                await asyncio.sleep(retry_delay)
                                continue
                            else:
                                logger.error(f"Failed to send message after {max_retries} attempts: {plain_e}")
                else:
                    logger.error(f"Failed to send message: {e}")
                    return  # Don't retry on non-network errors
            except (telegram.error.NetworkError, telegram.error.TimedOut) as e:
                # Network-related errors, retry
                if attempt < max_retries - 1:
                    logger.warning(f"Network error (attempt {attempt + 1}/{max_retries}): {e}")
                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    logger.error(f"Failed to send message after {max_retries} attempts: {e}")
            except Exception as e:
                # Check if it's a network-related error
                error_str = str(e).lower()
                if any(x in error_str for x in ['httpx.readerror', 'network', 'timeout', 'connection']):
                    if attempt < max_retries - 1:
                        logger.warning(f"Network error (attempt {attempt + 1}/{max_retries}): {e}")
                        await asyncio.sleep(retry_delay)
                        continue
                    else:
                        logger.error(f"Failed to send message after {max_retries} attempts: {e}")
                else:
                    logger.error(f"Failed to send message: {e}")
                    return  # Don't retry on non-network errors
    
    async def safe_reply(self, update: Update, text: str, parse_mode: str = 'Markdown'):
        """Safely reply to a message with automatic fallback and retry"""
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                await update.message.reply_text(text, parse_mode=parse_mode)
                return  # Success, exit
            except telegram.error.BadRequest as e:
                if "can't parse entities" in str(e).lower():
                    logger.warning(f"Markdown parsing failed in reply, trying escaped")
                    try:
                        # More comprehensive escaping
                        escaped_text = text
                        # Escape special markdown characters
                        for char in ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']:
                            escaped_text = escaped_text.replace(char, f'\\{char}')
                        await update.message.reply_text(escaped_text, parse_mode=parse_mode)
                        return  # Success, exit
                    except Exception as e2:
                        # Final fallback to plain text
                        logger.warning(f"Escaped markdown also failed ({e2}), replying as plain text")
                        # Remove all markdown formatting
                        plain_text = text
                        for char in ['*', '_', '`', '~']:
                            plain_text = plain_text.replace(char, '')
                        try:
                            await update.message.reply_text(plain_text, parse_mode=None)
                            return  # Success, exit
                        except Exception as plain_e:
                            if attempt < max_retries - 1:
                                logger.warning(f"Plain text reply failed (attempt {attempt + 1}/{max_retries}): {plain_e}")
                                await asyncio.sleep(retry_delay)
                                continue
                            else:
                                logger.error(f"Failed to reply after {max_retries} attempts: {plain_e}")
                else:
                    # Re-raise other BadRequest errors
                    logger.error(f"BadRequest error: {e}")
                    return
            except (telegram.error.NetworkError, telegram.error.TimedOut) as e:
                # Network-related errors, retry
                if attempt < max_retries - 1:
                    logger.warning(f"Network error in reply (attempt {attempt + 1}/{max_retries}): {e}")
                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    logger.error(f"Failed to reply after {max_retries} attempts: {e}")
            except Exception as e:
                # Check if it's a network-related error
                error_str = str(e).lower()
                if any(x in error_str for x in ['httpx.readerror', 'network', 'timeout', 'connection']):
                    if attempt < max_retries - 1:
                        logger.warning(f"Network error in reply (attempt {attempt + 1}/{max_retries}): {e}")
                        await asyncio.sleep(retry_delay)
                        continue
                    else:
                        logger.error(f"Failed to reply after {max_retries} attempts: {e}")
                else:
                    logger.error(f"Failed to reply: {e}")
                    return  # Don't retry on non-network errors

    async def start(self, update:Update, ctx:ContextTypes.DEFAULT_TYPE):
        """Start command handler"""
        await self.safe_reply(update,
            "🤖 *Trading Bot Active*\n\n"
            "Use /help to see available commands."
        )

    async def help(self, update:Update, ctx:ContextTypes.DEFAULT_TYPE):
        """Help command handler"""
        per_trade, risk_label = self._compute_risk_snapshot()
        risk_cfg = self.shared.get("risk")
        timeframe = self.shared.get("timeframe", "15")

        help_text = f"""📚 *Bot Commands*

📊 *Monitoring*
/dashboard – Full bot overview
/health – Data & heartbeat summary
/status – Open positions snapshot
/balance – Latest account balance
/symbols – Active trading universe
/regime SYMBOL – Market regime for a symbol

📈 *Performance & Analytics*
/stats [all|pullback|reversion] – Trade statistics
/recent [limit] – Recent trade log
/ml or /mlstatus – Pullback ML status
/mlpatterns [strategy] – Learned ML patterns
/phantom [strategy] – Phantom trade outcomes
/evolution – ML evolution shadow book
/analysis [symbol] – Recent analysis timestamps

🚀 *Enhanced Parallel System*
/system – Parallel routing status
/enhancedmr – Enhanced MR ML summary
/mrphantom – MR phantom trades
/parallel_performance – Compare strategies
/regimeanalysis – Top regime signals
/strategycomparison – Strategy performance table

⚙️ *Risk Management*
/risk – Current risk settings
/riskpercent value – Set % risk (e.g. 2.5)
/riskusd value – Set USD risk (e.g. 100)
/setrisk amount – Flexible input ("3%" or "50")
/mlrisk – Toggle ML dynamic risk
/mlriskrange min max – Dynamic risk bounds

🛠 *Controls*
/panic_close or /panicclose SYMBOL – Emergency exit
/forceretrain – Force ML retrain cycle
/update_clusters – Refresh symbol clusters

🎯 *ML Training*
/trainingstatus – Background training progress
/mr_ml – MR ML statistics

ℹ️ *Info*
/start – Welcome
/help – This menu

📈 *Current Settings*
• Risk per trade: {risk_label}
• Max leverage: {risk_cfg.max_leverage}x
• Timeframe: {timeframe} minutes
• Strategies: Pullback ML & Mean Reversion
"""
        await self.safe_reply(update, help_text)

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
📈 *Risk/Reward Ratio:* 1:{risk.rr if hasattr(risk, 'rr') else 2.5}

*Commands:*
`/risk_percent 2.5` - Set to 2.5%
`/risk_usd 100` - Set to $100
`/set_risk 3%` or `/set_risk 50` - Flexible"""
            
            await self.safe_reply(update, msg)
            
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

    async def ml_risk(self, update:Update, ctx:ContextTypes.DEFAULT_TYPE):
        """Enable/disable ML dynamic risk"""
        try:
            risk = self.shared["risk"]
            
            if not ctx.args:
                # Show current ML risk status
                msg = "🤖 *ML Dynamic Risk Status*\n"
                msg += "━" * 20 + "\n\n"
                
                if risk.use_ml_dynamic_risk:
                    msg += "• Status: ✅ *ENABLED*\n"
                    msg += f"• ML Score Range: {risk.ml_risk_min_score} - {risk.ml_risk_max_score}\n"
                    msg += f"• Risk Range: {risk.ml_risk_min_percent}% - {risk.ml_risk_max_percent}%\n\n"
                    
                    # Show example scaling
                    msg += "📊 *Risk Scaling Examples:*\n"
                    for score in [70, 75, 80, 85, 90, 95, 100]:
                        if score < risk.ml_risk_min_score:
                            continue
                        if score > risk.ml_risk_max_score:
                            score = risk.ml_risk_max_score
                        
                        # Calculate risk using same logic as sizer
                        score_range = risk.ml_risk_max_score - risk.ml_risk_min_score
                        risk_range = risk.ml_risk_max_percent - risk.ml_risk_min_percent
                        if score_range > 0:
                            score_position = (score - risk.ml_risk_min_score) / score_range
                            risk_pct = risk.ml_risk_min_percent + (score_position * risk_range)
                        else:
                            risk_pct = risk.ml_risk_min_percent
                            
                        msg += f"• ML Score {score}: {risk_pct:.2f}% risk\n"
                else:
                    msg += "• Status: ❌ *DISABLED*\n"
                    msg += f"• Fixed Risk: {risk.risk_percent}%\n"
                
                msg += "\nUsage:\n"
                msg += "`/ml_risk on` - Enable ML dynamic risk\n"
                msg += "`/ml_risk off` - Disable ML dynamic risk\n"
                msg += "`/ml_risk_range 1 5` - Set risk range"
                
                await self.safe_reply(update, msg)
                return
            
            action = ctx.args[0].lower()
            
            if action == "on":
                risk.use_ml_dynamic_risk = True
                msg = "✅ ML Dynamic Risk *ENABLED*\n\n"
                msg += f"Risk will scale from {risk.ml_risk_min_percent}% to {risk.ml_risk_max_percent}%\n"
                msg += f"Based on ML scores {risk.ml_risk_min_score} to {risk.ml_risk_max_score}\n\n"
                msg += "_Higher ML confidence = Higher position size_"
                await self.safe_reply(update, msg)
                logger.info("ML dynamic risk enabled")
                
            elif action == "off":
                risk.use_ml_dynamic_risk = False
                msg = "❌ ML Dynamic Risk *DISABLED*\n\n"
                msg += f"All trades will use fixed {risk.risk_percent}% risk"
                await self.safe_reply(update, msg)
                logger.info("ML dynamic risk disabled")
                
            else:
                await update.message.reply_text("Usage: /ml_risk on|off")
                
        except Exception as e:
            logger.error(f"Error in ml_risk: {e}")
            await update.message.reply_text("Error updating ML risk settings")

    async def ml_risk_range(self, update:Update, ctx:ContextTypes.DEFAULT_TYPE):
        """Set ML dynamic risk range"""
        try:
            if len(ctx.args) != 2:
                await update.message.reply_text(
                    "Usage: `/ml_risk_range 1 5`\n"
                    "Sets risk range from 1% to 5%",
                    parse_mode='Markdown'
                )
                return
            
            min_risk = float(ctx.args[0])
            max_risk = float(ctx.args[1])
            
            # Validate
            if min_risk <= 0 or min_risk > 10:
                await update.message.reply_text("Minimum risk must be between 0% and 10%")
                return
            
            if max_risk <= min_risk:
                await update.message.reply_text("Maximum risk must be greater than minimum risk")
                return
                
            if max_risk > 10:
                await update.message.reply_text("Maximum risk must not exceed 10%")
                return
            
            # Update
            risk = self.shared["risk"]
            risk.ml_risk_min_percent = min_risk
            risk.ml_risk_max_percent = max_risk
            
            msg = f"✅ ML Risk Range Updated\n\n"
            msg += f"• Minimum Risk: {min_risk}%\n"
            msg += f"• Maximum Risk: {max_risk}%\n\n"
            
            if risk.use_ml_dynamic_risk:
                msg += "_ML Dynamic Risk is currently ENABLED_"
            else:
                msg += "_ML Dynamic Risk is currently DISABLED_\n"
                msg += "Use `/ml_risk on` to enable"
            
            await self.safe_reply(update, msg)
            logger.info(f"ML risk range updated: {min_risk}% - {max_risk}%")
            
        except ValueError:
            await update.message.reply_text("Invalid values. Use numbers only.")
        except Exception as e:
            logger.error(f"Error in ml_risk_range: {e}")
            await update.message.reply_text("Error updating ML risk range")

    async def status(self, update:Update, ctx:ContextTypes.DEFAULT_TYPE):
        """Show current positions"""
        try:
            book = self.shared.get("book")
            positions = book.positions if book else {}
            frames = self.shared.get("frames", {})
            per_trade_risk, risk_label = self._compute_risk_snapshot()

            header = ["📊 *Open Positions*", "━━━━━━━━━━━━━━━━━━━━━━"]

            if not positions:
                header.append("")
                header.append("No open positions.")
                header.append(f"*Risk per trade:* {risk_label}")
                header.append(f"*Symbols scanning:* {len(frames)}")
                await self.safe_reply(update, "\n".join(header))
                return

            now = datetime.now(timezone.utc)
            total_pnl = 0.0
            lines = header + [""]

            for idx, (sym, pos) in enumerate(positions.items()):
                if idx >= 10:
                    lines.append(f"…and {len(positions) - idx} more symbols")
                    break

                emoji = "🟢" if pos.side == "long" else "🔴"
                lines.append(f"{emoji} *{sym}* ({pos.side.upper()})")

                strategy = getattr(pos, 'strategy_name', 'unknown')
                lines.append(f"  Strategy: {strategy}")
                lines.append(f"  Entry: {pos.entry:.4f} | Size: {pos.qty}")
                lines.append(f"  SL: {pos.sl:.4f} | TP: {pos.tp:.4f}")

                # Hold duration
                if getattr(pos, 'entry_time', None):
                    entry_time = pos.entry_time
                    if entry_time.tzinfo is None:
                        entry_time = entry_time.replace(tzinfo=timezone.utc)
                    held_minutes = max(0, int((now - entry_time).total_seconds() // 60))
                    lines.append(f"  Held: {held_minutes}m")

                # Live PnL snapshot
                if sym in frames and len(frames[sym]) > 0:
                    current_price = frames[sym]['close'].iloc[-1]
                    if pos.side == "long":
                        pnl = (current_price - pos.entry) * pos.qty
                        pnl_pct = ((current_price - pos.entry) / pos.entry) * 100
                    else:
                        pnl = (pos.entry - current_price) * pos.qty
                        pnl_pct = ((pos.entry - current_price) / pos.entry) * 100
                    total_pnl += pnl
                    pnl_emoji = "🟢" if pnl >= 0 else "🔴"
                    lines.append(f"  P&L: {pnl_emoji} ${pnl:.2f} ({pnl_pct:+.2f}%)")

                lines.append("")

            total_positions = len(positions)
            estimated_risk = per_trade_risk * total_positions
            lines.append(f"*Positions:* {total_positions} | *Estimated risk:* ${estimated_risk:.2f}")
            lines.append(f"*Risk per trade:* {risk_label}")
            lines.append(f"*Unrealised P&L:* ${total_pnl:.2f}")

            await self.safe_reply(update, "\n".join(lines))

        except Exception as e:
            logger.exception("Error in status: %s", e)
            await update.message.reply_text("Error getting status")

    async def balance(self, update:Update, ctx:ContextTypes.DEFAULT_TYPE):
        """Show account balance"""
        try:
            broker = self.shared.get("broker")
            if broker:
                balance = broker.get_balance()
                if balance:
                    self.shared["last_balance"] = balance
                    await self.safe_reply(update, f"💰 *Balance:* ${balance:.2f} USDT")
                else:
                    await update.message.reply_text("Unable to fetch balance")
            else:
                await update.message.reply_text("Broker not initialized")
        except Exception as e:
            logger.exception("Error in balance: %s", e)
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
            
            await self.safe_reply(update, msg)
            
        except Exception as e:
            logger.error(f"Error in health: {e}")
            await update.message.reply_text("Error getting health status")
    
    async def symbols(self, update:Update, ctx:ContextTypes.DEFAULT_TYPE):
        """Show list of active trading symbols"""
        try:
            frames = self.shared.get("frames", {})
            configured = self.shared.get("symbols_config")

            if configured:
                symbols_list = list(configured)
            elif frames:
                symbols_list = list(frames.keys())
            else:
                await update.message.reply_text("No symbols loaded yet")
                return

            msg = "📈 *Active Trading Pairs*\n\n"

            # Show in groups of 5
            for i in range(0, len(symbols_list), 5):
                group = symbols_list[i:i+5]
                msg += " • ".join(group) + "\n"
            
            msg += f"\n*Total:* {len(symbols_list)} symbols"
            timeframe = self.shared.get("timeframe")
            if timeframe:
                msg += f"\n*Timeframe:* {timeframe} minutes"
            msg += "\n*Strategies:* Pullback ML / Mean Reversion"
            
            await self.safe_reply(update, msg)
            
        except Exception as e:
            logger.exception("Error in symbols: %s", e)
            await update.message.reply_text("Error getting symbols")
    
    async def dashboard(self, update:Update, ctx:ContextTypes.DEFAULT_TYPE):
        """Show complete bot dashboard"""
        try:
            frames = self.shared.get("frames", {})
            book = self.shared.get("book")
            last_analysis = self.shared.get("last_analysis", {})
            per_trade_risk, risk_label = self._compute_risk_snapshot()

            lines = ["🎯 *Trading Bot Dashboard*", "━━━━━━━━━━━━━━━━━━━━━", ""]

            lines.append("⚡ *System Status*")
            if frames:
                lines.append("• Status: ✅ Online")
                lines.append(f"• Symbols streaming: {len(frames)}")
            else:
                lines.append("• Status: ⏳ Starting up")

            timeframe = self.shared.get("timeframe")
            if timeframe:
                lines.append(f"• Timeframe: {timeframe}m")

            symbols_cfg = self.shared.get("symbols_config")
            if symbols_cfg:
                lines.append(f"• Universe: {len(symbols_cfg)} symbols")

            if last_analysis:
                try:
                    latest_symbol, latest_time = max(last_analysis.items(), key=lambda kv: kv[1])
                    if isinstance(latest_time, datetime):
                        ref_now = datetime.now(latest_time.tzinfo) if latest_time.tzinfo else datetime.now()
                        age_minutes = max(0, int((ref_now - latest_time).total_seconds() // 60))
                        lines.append(f"• Last scan: {latest_symbol} ({age_minutes}m ago)")
                except Exception as exc:
                    logger.debug(f"Unable to compute last analysis recency: {exc}")

            broker = self.shared.get("broker")
            balance = self.shared.get("last_balance")
            if broker:
                if balance is None:
                    try:
                        balance = broker.get_balance()
                        if balance:
                            self.shared["last_balance"] = balance
                    except Exception as exc:
                        logger.warning(f"Error refreshing balance: {exc}")
                if balance is not None:
                    lines.append(f"• Balance: ${balance:.2f} USDT")
                try:
                    api_info = broker.get_api_key_info()
                    if api_info and api_info.get("expiredAt"):
                        expiry_timestamp = int(api_info["expiredAt"]) / 1000
                        expiry_date = datetime.fromtimestamp(expiry_timestamp)
                        days_remaining = (expiry_date - datetime.now()).days
                        if days_remaining < 14:
                            lines.append(f"⚠️ *API Key:* expires in {days_remaining} days")
                        else:
                            lines.append(f"🔑 API Key: {days_remaining} days remaining")
                except Exception as exc:
                    logger.warning(f"Could not fetch API key expiry: {exc}")

            lines.append("")

            risk = self.shared.get("risk")
            lines.append("⚙️ *Trading Settings*")
            lines.append(f"• Risk per trade: {risk_label}")
            lines.append(f"• Max leverage: {risk.max_leverage}x")
            if getattr(risk, 'use_ml_dynamic_risk', False):
                lines.append("• ML Dynamic Risk: Enabled")

            ml_scorer = self.shared.get("ml_scorer")
            if ml_scorer:
                try:
                    ml_stats = ml_scorer.get_stats()
                    lines.append("")
                    lines.append("🤖 *Pullback ML*")
                    lines.append(f"• Status: {ml_stats['status']}")
                    lines.append(f"• Trades used: {ml_stats['completed_trades']}")
                    if ml_stats.get('recent_win_rate'):
                        lines.append(f"• Recent win rate: {ml_stats['recent_win_rate']:.1f}%")
                except Exception as exc:
                    logger.debug(f"Unable to fetch pullback ML stats: {exc}")

            enhanced_mr = self.shared.get("enhanced_mr_scorer")
            if enhanced_mr:
                try:
                    mr_info = enhanced_mr.get_retrain_info()
                    lines.append("")
                    lines.append("🧠 *Mean Reversion ML*")
                    status = "✅ Ready" if mr_info.get('is_ml_ready') else "⏳ Training"
                    lines.append(f"• Status: {status}")
                    lines.append(f"• Trades (exec + phantom): {mr_info.get('total_combined', 0)}")
                    lines.append(f"• Next retrain in: {mr_info.get('trades_until_next_retrain', 0)} trades")
                except Exception as exc:
                    logger.debug(f"Unable to fetch MR ML stats: {exc}")

            positions = book.positions if book else {}
            lines.append("")
            lines.append("📊 *Positions*")
            if positions:
                estimated_risk = per_trade_risk * len(positions)
                lines.append(f"• Open positions: {len(positions)}")
                lines.append(f"• Estimated risk: ${estimated_risk:.2f}")
                if self.shared.get('use_enhanced_parallel', False):
                    lines.append("• Routing: Enhanced parallel (Pullback + MR)")
            else:
                lines.append("• No open positions")

            phantom_tracker = self.shared.get("phantom_tracker")
            if phantom_tracker:
                try:
                    stats = phantom_tracker.get_phantom_stats()
                    lines.append("")
                    lines.append("👻 *Pullback Phantom*")
                    lines.append(f"• Rejections tracked: {stats.get('rejected', 0)}")
                except Exception as exc:
                    logger.debug(f"Unable to fetch pullback phantom stats: {exc}")

            mr_phantom = self.shared.get("mr_phantom_tracker")
            if mr_phantom:
                try:
                    mr_stats = mr_phantom.get_mr_phantom_stats()
                    lines.append("")
                    lines.append("🌀 *MR Phantom*")
                    lines.append(f"• Tracked: {mr_stats.get('total_mr_trades', 0)}")
                except Exception as exc:
                    logger.debug(f"Unable to fetch MR phantom stats: {exc}")

            lines.append("")
            lines.append("_Use /status for position details and /ml for full analytics._")

            await self.safe_reply(update, "\n".join(lines))

        except Exception as e:
            logger.exception("Error in dashboard: %s", e)
            await update.message.reply_text("Error getting dashboard")
    


    async def regime_single(self, update:Update, ctx:ContextTypes.DEFAULT_TYPE):
        """Show enhanced regime snapshot for one or more symbols."""
        try:
            frames = self.shared.get("frames", {})
            if not frames:
                await update.message.reply_text("No market data available yet")
                return

            if ctx.args:
                symbols = [ctx.args[0].upper()]
            else:
                last_analysis = self.shared.get("last_analysis", {})
                if last_analysis:
                    symbols = [sym for sym, _ in sorted(last_analysis.items(), key=lambda kv: kv[1], reverse=True)[:5]]
                else:
                    symbols = list(frames.keys())[:5]

            if not symbols:
                await update.message.reply_text("No symbols to analyse")
                return

            lines = ["🔍 *Market Regime Snapshot*", ""]

            for sym in symbols:
                df = frames.get(sym)
                if df is None or len(df) < 50:
                    lines.append(f"• {sym}: insufficient data")
                    continue

                try:
                    analysis = get_enhanced_market_regime(df.tail(200), sym)
                    confidence = analysis.regime_confidence * 100
                    lines.append(f"• *{sym}*: {analysis.primary_regime.title()} ({confidence:.0f}% conf)")

                    detail = [f"Vol: {analysis.volatility_level}"]
                    if analysis.primary_regime == "ranging":
                        detail.append(f"Range: {analysis.range_quality}")
                    if analysis.recommended_strategy and analysis.recommended_strategy != "none":
                        detail.append(f"Strat: {analysis.recommended_strategy.replace('_', ' ').title()}")
                    lines.append("  " + " | ".join(detail))
                except Exception as exc:
                    logger.debug(f"Regime analysis error for {sym}: {exc}")
                    lines.append(f"• {sym}: unable to analyse")

            await self.safe_reply(update, "\n".join(lines))

        except Exception as e:
            logger.exception("Error in regime command: %s", e)
            await update.message.reply_text("Error retrieving regime information")


    async def analysis(self, update:Update, ctx:ContextTypes.DEFAULT_TYPE):
        """Show recent analysis details for symbols."""
        try:
            frames = self.shared.get("frames", {})
            last_analysis = self.shared.get("last_analysis", {})

            if not frames:
                await update.message.reply_text("No market data available yet")
                return

            if ctx.args:
                symbols = [ctx.args[0].upper()]
            elif last_analysis:
                symbols = [sym for sym, _ in sorted(last_analysis.items(), key=lambda kv: kv[1], reverse=True)[:5]]
            else:
                symbols = list(frames.keys())[:5]

            msg = "🔍 *Recent Analysis*\n"
            msg += "━━━━━━━━━━━━━━━━━━━━\n\n"

            for symbol in symbols:
                df = frames.get(symbol)
                if df is None or len(df) < 20:
                    msg += f"*{symbol}*: insufficient data\n\n"
                    continue

                msg += f"*{symbol}*\n"

                if symbol in last_analysis:
                    analysed_at = last_analysis[symbol]
                    if analysed_at.tzinfo is None:
                        analysed_at = analysed_at.replace(tzinfo=timezone.utc)
                    age_minutes = max(0, int((datetime.now(timezone.utc) - analysed_at).total_seconds() // 60))
                    msg += f"• Last analysed: {age_minutes}m ago\n"
                else:
                    msg += "• Last analysed: n/a\n"

                last_price = df['close'].iloc[-1]
                msg += f"• Last price: {last_price:.4f}\n"
                msg += f"• Candles loaded: {len(df)}\n\n"

            msg += "_Use /analysis SYMBOL for a specific market_"

            await self.safe_reply(update, msg)

        except Exception as e:
            logger.exception("Error in analysis: %s", e)
            await update.message.reply_text("Error getting analysis details")
    
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
            await self.safe_reply(update, msg)
            
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
            if hasattr(tracker, 'format_recent_trades'):
                msg = tracker.format_recent_trades(limit)
            else:
                # Fallback for TradeTrackerPostgres without format_recent_trades
                msg = "📜 *Recent Trades*\n"
                msg += "━" * 20 + "\n\n"
                
                # Get trades
                trades = []
                if hasattr(tracker, 'trades'):
                    trades = tracker.trades[-limit:] if tracker.trades else []
                elif hasattr(tracker, 'get_recent_trades'):
                    trades = tracker.get_recent_trades(limit)
                
                if not trades:
                    msg += "_No trades recorded yet_"
                else:
                    for i, trade in enumerate(reversed(trades[-limit:]), 1):
                        # Format each trade
                        symbol = getattr(trade, 'symbol', 'N/A')
                        side = getattr(trade, 'side', 'N/A').upper()
                        pnl = float(getattr(trade, 'pnl_usd', 0))
                        pnl_pct = float(getattr(trade, 'pnl_percent', 0))
                        exit_time = getattr(trade, 'exit_time', None)
                        
                        # Format time
                        time_str = ""
                        if exit_time:
                            if isinstance(exit_time, str):
                                time_str = exit_time[:16]  # Keep YYYY-MM-DD HH:MM
                            else:
                                time_str = exit_time.strftime("%Y-%m-%d %H:%M")
                        
                        # Build trade line
                        result_emoji = "✅" if pnl > 0 else "❌"
                        msg += f"{i}. {result_emoji} *{symbol}* {side}\n"
                        msg += f"   P&L: ${pnl:.2f} ({pnl_pct:+.2f}%)\n"
                        if time_str:
                            msg += f"   Time: {time_str}\n"
                        msg += "\n"
                
                msg += f"\n_Showing last {min(limit, len(trades))} trades_"
                
            await self.safe_reply(update, msg)
            
        except Exception as e:
            logger.error(f"Error in recent_trades: {e}")
            await update.message.reply_text("Error getting recent trades")
    
    async def ml_stats(self, update:Update, ctx:ContextTypes.DEFAULT_TYPE):
        """Show ML system statistics and status for a specific strategy."""
        try:
            strategy_arg = 'pullback' # Default strategy
            if ctx.args:
                strategy_arg = ctx.args[0].lower()
                if strategy_arg not in ['pullback', 'reversion']:
                    await self.safe_reply(update, "Invalid strategy. Choose `pullback` or `reversion`.")
                    return

            msg = f"🤖 *ML Status: {strategy_arg.title()} Strategy*\n"
            msg += "━" * 25 + "\n\n"

            if strategy_arg == 'pullback':
                ml_scorer = self.shared.get("ml_scorer")
            else:
                # Placeholder for the future mean reversion scorer
                ml_scorer = self.shared.get("ml_scorer_reversion") 

            if not ml_scorer:
                msg += f"❌ *ML System Not Available for {strategy_arg.title()} Strategy*\n"
                if strategy_arg == 'reversion':
                    msg += "This model will be trained after enough data is collected from the rule-based strategy."
                await self.safe_reply(update, msg)
                return

            # Get and display stats from the selected scorer
            stats = ml_scorer.get_stats()
            if stats.get('is_ml_ready'):
                msg += "✅ *Status: Active & Learning*\n"
                msg += f"• Model trained on: {stats.get('last_train_count', stats.get('completed_trades', 0))} trades\n"
            else:
                msg += "📊 *Status: Collecting Data*\n"
                trades_needed = stats.get('trades_needed', 200)
                msg += f"• Trades needed for training: {trades_needed}\n"

            msg += f"• Completed trades (live): {stats.get('completed_trades', 0)}\n"
            msg += f"• Current threshold: {stats.get('current_threshold', 70):.0f}\n"
            msg += f"• Active models: {len(stats.get('models_active', []))}\n"

            # Show retrain info for mean reversion
            if strategy_arg == 'reversion' and 'next_retrain_in' in stats:
                msg += f"• Next retrain in: {stats['next_retrain_in']} trades\n"

            # Add strategy-specific notes
            if strategy_arg == 'reversion':
                msg += "\n📝 *Mean Reversion Features:*\n"
                msg += "• Range characteristics\n"
                msg += "• Oscillator extremes\n"
                msg += "• Volume confirmation\n"
                msg += "• Reversal strength\n"

            await self.safe_reply(update, msg)

        except Exception as e:
            logger.error(f"Error in ml_stats: {e}")
            await update.message.reply_text("Error getting ML statistics")
    
    async def reset_stats(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Reset trade statistics"""
        try:
            import os
            import json
            from datetime import datetime
            
            # Check if user is authorized
            if update.effective_user.id != self.chat_id:
                await update.message.reply_text("❌ Unauthorized")
                return
            
            reset_count = 0
            backups = []
            
            # 1. Reset TradeTracker history
            if os.path.exists("trade_history.json"):
                try:
                    # Read existing data
                    with open("trade_history.json", 'r') as f:
                        data = json.load(f)
                        trade_count = len(data)
                    
                    if trade_count > 0:
                        # Create backup
                        backup_name = f"trade_history_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                        with open(backup_name, 'w') as f:
                            json.dump(data, f, indent=2)
                        backups.append(f"• {trade_count} trades → {backup_name}")
                        reset_count += trade_count
                        
                        # Create empty file
                        with open("trade_history.json", 'w') as f:
                            json.dump([], f)
                except Exception as e:
                    logger.error(f"Error backing up trade history: {e}")
            
            # 2. Reset any cached stats in shared tracker
            if "tracker" in self.shared and self.shared["tracker"]:
                try:
                    self.shared["tracker"].trades = []
                    self.shared["tracker"].save_trades()
                    backups.append("• Tracker cache cleared")
                except:
                    pass
            
            # 3. Reset ML trade count (keep model but reset counter)
            ml_reset_info = ""
            if "ml_scorer" in self.shared and self.shared["ml_scorer"]:
                try:
                    ml_scorer = self.shared["ml_scorer"]
                    old_count = ml_scorer.completed_trades_count if hasattr(ml_scorer, 'completed_trades_count') else 0
                    
                    # Reset counters
                    if hasattr(ml_scorer, 'completed_trades_count'):
                        ml_scorer.completed_trades_count = 0
                    if hasattr(ml_scorer, 'last_train_count'):
                        ml_scorer.last_train_count = 0
                    
                    # Clear Redis data if available
                    if hasattr(ml_scorer, 'redis_client') and ml_scorer.redis_client:
                        try:
                            ml_scorer.redis_client.delete('ml_completed_trades')
                            ml_scorer.redis_client.delete('ml_enhanced_completed_trades')
                            ml_scorer.redis_client.delete('ml_v2_completed_trades')
                            ml_scorer.redis_client.set('ml_trades_count', 0)
                            ml_scorer.redis_client.set('ml_enhanced_trades_count', 0)
                            ml_scorer.redis_client.set('ml_v2_trades_count', 0)
                        except:
                            pass
                    
                    ml_reset_info = f"\n🤖 **ML Status:**\n"
                    ml_reset_info += f"• Reset {old_count} trade counter\n"
                    ml_reset_info += f"• Model kept (if trained)\n"
                    ml_reset_info += f"• Will retrain after 200 new trades"
                except Exception as e:
                    logger.error(f"Error resetting ML stats: {e}")
            
            # Build response
            if reset_count > 0 or backups:
                response = "✅ **Statistics Reset Complete!**\n\n"
                
                if backups:
                    response += "**Backed up:**\n"
                    response += "\n".join(backups) + "\n"
                
                response += ml_reset_info + "\n"
                
                response += "\n**What happens now:**\n"
                response += "• Trade history: Starting fresh at 0\n"
                response += "• Win rate: Will recalculate from new trades\n"
                response += "• P&L: Reset to $0.00\n"
                response += "• New trades will build fresh statistics\n\n"
                response += "📊 Use /stats to see fresh statistics\n"
                response += "🤖 Use /ml to check ML status"
            else:
                response = "ℹ️ No statistics to reset - already clean\n\n"
                response += "📊 /stats - View statistics\n"
                response += "🤖 /ml - Check ML status"
            
            await self.safe_reply(update, response)
            
        except Exception as e:
            logger.error(f"Error resetting stats: {e}")
            await update.message.reply_text(f"❌ Error resetting stats: {str(e)[:200]}")
    
    async def ml_rankings(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Show ML rankings for all symbols"""
        try:
            # Get trade tracker for historical data
            trade_tracker = self.shared.get("trade_tracker")
            if not trade_tracker:
                await update.message.reply_text("Trade tracker not available")
                return
            
            # Get all trades
            all_trades = trade_tracker.trades if hasattr(trade_tracker, 'trades') else []
            
            if not all_trades:
                await update.message.reply_text(
                    "📊 *ML Symbol Rankings*\n\n"
                    "No completed trades yet.\n"
                    "Rankings will appear after trades complete.",
                    parse_mode='Markdown'
                )
                return
            
            # Calculate stats per symbol
            symbol_stats = {}
            
            for trade in all_trades:
                symbol = trade.symbol
                if symbol not in symbol_stats:
                    symbol_stats[symbol] = {
                        'trades': 0,
                        'wins': 0,
                        'losses': 0,
                        'total_pnl': 0,
                        'best_trade': 0,
                        'worst_trade': 0,
                        'last_5_trades': []
                    }
                
                stats = symbol_stats[symbol]
                stats['trades'] += 1
                
                # Convert Decimal to float for calculations
                pnl = float(trade.pnl_usd)
                
                if pnl > 0:
                    stats['wins'] += 1
                else:
                    stats['losses'] += 1
                
                stats['total_pnl'] += pnl
                stats['best_trade'] = max(stats['best_trade'], pnl)
                stats['worst_trade'] = min(stats['worst_trade'], pnl)
                
                # Track last 5 trades for trend
                stats['last_5_trades'].append(1 if pnl > 0 else 0)
                if len(stats['last_5_trades']) > 5:
                    stats['last_5_trades'].pop(0)
            
            # Calculate rankings
            rankings = []
            for symbol, stats in symbol_stats.items():
                win_rate = (stats['wins'] / stats['trades']) * 100 if stats['trades'] > 0 else 0
                avg_pnl = stats['total_pnl'] / stats['trades'] if stats['trades'] > 0 else 0
                
                # Recent performance (last 5 trades)
                if len(stats['last_5_trades']) >= 3:
                    recent_wr = sum(stats['last_5_trades']) / len(stats['last_5_trades']) * 100
                else:
                    recent_wr = win_rate
                
                # Score combines win rate, profitability, and recent performance
                # 50% win rate, 30% profitability, 20% recent performance
                normalized_pnl = max(-100, min(100, avg_pnl * 10))  # Normalize P&L
                score = (win_rate * 0.5) + (normalized_pnl * 0.3) + (recent_wr * 0.2)
                
                # Training data indicator
                data_quality = "🟢" if stats['trades'] >= 10 else "🟡" if stats['trades'] >= 5 else "🔴"
                
                rankings.append({
                    'symbol': symbol,
                    'win_rate': win_rate,
                    'trades': stats['trades'],
                    'wins': stats['wins'],
                    'losses': stats['losses'],
                    'total_pnl': stats['total_pnl'],
                    'avg_pnl': avg_pnl,
                    'score': score,
                    'recent_wr': recent_wr,
                    'data_quality': data_quality
                })
            
            # Sort by score
            rankings.sort(key=lambda x: x['score'], reverse=True)
            
            # Format message
            msg = "🏆 *ML Symbol Performance Rankings*\n"
            msg += "=" * 30 + "\n\n"
            
            # Summary
            total_symbols = len(rankings)
            profitable_symbols = sum(1 for r in rankings if r['total_pnl'] > 0)
            high_wr_symbols = sum(1 for r in rankings if r['win_rate'] >= 50)
            well_tested = sum(1 for r in rankings if r['trades'] >= 10)
            
            msg += f"📊 *Overview*\n"
            msg += f"Total Symbols: {total_symbols}\n"
            msg += f"Profitable: {profitable_symbols} ({profitable_symbols/total_symbols*100:.0f}%)\n" if total_symbols > 0 else ""
            msg += f"Win Rate ≥50%: {high_wr_symbols}\n"
            msg += f"Well Tested (10+ trades): {well_tested}\n\n"
            
            # Data quality legend
            msg += "📈 *Data Quality*\n"
            msg += "🟢 10+ trades (reliable)\n"
            msg += "🟡 5-9 trades (moderate)\n"
            msg += "🔴 <5 trades (limited)\n\n"
            
            # Top performers
            msg += "✅ *Top 10 Performers*\n"
            msg += "```\n"
            msg += f"{'#':<3} {'Symbol':<10} {'WR%':>6} {'Trades':>7} {'PnL':>8} {'Q'}\n"
            msg += "-" * 40 + "\n"
            
            for i, r in enumerate(rankings[:10], 1):
                msg += f"{i:<3} {r['symbol']:<10} {r['win_rate']:>5.1f}% {r['trades']:>7} ${r['total_pnl']:>7.2f} {r['data_quality']}\n"
            msg += "```\n\n"
            
            # Bottom performers (if more than 10)
            if len(rankings) > 10:
                msg += "❌ *Bottom 5 Performers*\n"
                msg += "```\n"
                msg += f"{'Symbol':<10} {'WR%':>6} {'Trades':>7} {'PnL':>8}\n"
                msg += "-" * 35 + "\n"
                
                bottom = rankings[-5:] if len(rankings) > 5 else []
                for r in bottom:
                    msg += f"{r['symbol']:<10} {r['win_rate']:>5.1f}% {r['trades']:>7} ${r['total_pnl']:>7.2f}\n"
                msg += "```\n\n"
            
            # Trending symbols
            trending_up = [r for r in rankings if r['recent_wr'] > r['win_rate'] + 10 and r['trades'] >= 5]
            trending_down = [r for r in rankings if r['recent_wr'] < r['win_rate'] - 10 and r['trades'] >= 5]
            
            if trending_up or trending_down:
                msg += "📈 *Trending*\n"
                if trending_up:
                    msg += "⬆️ Improving: " + ", ".join([r['symbol'] for r in trending_up[:3]]) + "\n"
                if trending_down:
                    msg += "⬇️ Declining: " + ", ".join([r['symbol'] for r in trending_down[:3]]) + "\n"
                msg += "\n"
            
            # ML recommendations
            msg += "🎯 *ML Recommendations*\n"
            
            # Find best reliable performer
            reliable = [r for r in rankings if r['trades'] >= 10]
            if reliable:
                best_reliable = reliable[0]
                msg += f"Most Reliable: {best_reliable['symbol']} "
                msg += f"({best_reliable['win_rate']:.1f}% in {best_reliable['trades']} trades)\n"
            
            # Find most profitable
            if rankings:
                most_profitable = max(rankings, key=lambda x: x['total_pnl'])
                if most_profitable['total_pnl'] > 0:
                    msg += f"Most Profitable: {most_profitable['symbol']} "
                    msg += f"(${most_profitable['total_pnl']:.2f})\n"
            
            # Symbols to watch (good WR but limited data)
            watch_list = [r for r in rankings if r['win_rate'] >= 60 and 3 <= r['trades'] < 10]
            if watch_list:
                msg += f"Watch List: " + ", ".join([r['symbol'] for r in watch_list[:5]]) + "\n"
            
            # Symbols to avoid
            avoid = [r for r in rankings if r['win_rate'] < 30 and r['trades'] >= 5]
            if avoid:
                msg += f"Consider Avoiding: " + ", ".join([r['symbol'] for r in avoid[:3]]) + "\n"
            
            msg += "\n_Refresh with /mlrankings_"
            
            await self.safe_reply(update, msg)
            
        except Exception as e:
            logger.error(f"Error in ml_rankings: {e}")
            import traceback
            logger.error(traceback.format_exc())
            await update.message.reply_text(f"Error generating rankings: {str(e)[:100]}")
    
    async def phantom_stats(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Show phantom trade statistics, with an optional strategy filter."""
        try:
            strategy_filter = None
            if ctx.args:
                strategy_filter = ctx.args[0].lower()
                if strategy_filter not in ['pullback', 'reversion']:
                    await update.message.reply_text("Invalid strategy. Choose `pullback` or `reversion`.")
                    return

            msg = "👻 *Phantom Trade Statistics*\n"
            msg += f"_Strategy: {strategy_filter.title() if strategy_filter else 'All'}_\n"
            msg += "━" * 25 + "\n\n"

            from phantom_trade_tracker import get_phantom_tracker
            phantom_tracker = get_phantom_tracker()

            all_phantoms = []
            for trades in phantom_tracker.phantom_trades.values():
                all_phantoms.extend(trades)

            if strategy_filter:
                all_phantoms = [p for p in all_phantoms if hasattr(p, 'strategy_name') and p.strategy_name.lower() == strategy_filter]

            if not all_phantoms:
                msg += "No phantom trades recorded for this filter."
                await self.safe_reply(update, msg)
                return

            # Calculate stats for the filtered set
            executed = [p for p in all_phantoms if p.was_executed]
            rejected = [p for p in all_phantoms if not p.was_executed]
            rejected_wins = [p for p in rejected if p.outcome == 'win']
            rejected_losses = [p for p in rejected if p.outcome == 'loss']
            missed_profit = sum(p.pnl_percent for p in rejected_wins if p.pnl_percent)
            avoided_loss = sum(abs(p.pnl_percent) for p in rejected_losses if p.pnl_percent)

            msg += "📊 *Overview*\n"
            msg += f"• Total signals tracked: {len(all_phantoms)}\n"
            msg += f"• Executed trades: {len(executed)}\n"
            msg += f"• Phantom trades: {len(rejected)}\n"
            
            # Verify counts add up
            if len(all_phantoms) != (len(executed) + len(rejected)):
                msg += f"⚠️ *Count mismatch: {len(all_phantoms)} ≠ {len(executed) + len(rejected)}*\n"
            msg += "\n"

            if rejected:
                msg += "🚫 *Rejected Trade Analysis*\n"
                rejected_wr = (len(rejected_wins) / len(rejected)) * 100 if rejected else 0
                msg += f"• Rejected Win Rate: {rejected_wr:.1f}% ({len(rejected_wins)}/{len(rejected)})\n"
                msg += f"• Missed Profit: +{missed_profit:.2f}%\n"
                msg += f"• Avoided Loss: -{avoided_loss:.2f}%\n"
                net_impact = missed_profit - avoided_loss
                msg += f"• *Net Impact: {net_impact:+.2f}%*\n"
            
            # Add executed trade analysis if available
            if executed:
                executed_wins = [p for p in executed if p.outcome == 'win']
                executed_losses = [p for p in executed if p.outcome == 'loss']
                if executed_wins or executed_losses:
                    msg += "\n✅ *Executed Trade Analysis*\n"
                    executed_wr = (len(executed_wins) / len(executed)) * 100 if executed else 0
                    msg += f"• Executed Win Rate: {executed_wr:.1f}% ({len(executed_wins)}/{len(executed)})\n"

            await self.safe_reply(update, msg)

        except Exception as e:
            logger.error(f"Error in phantom_stats: {e}")
            await update.message.reply_text("Error getting phantom statistics")
    
    async def phantom_detail(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Show detailed phantom statistics for a symbol"""
        try:
            # Get symbol from command
            if not ctx.args:
                await update.message.reply_text(
                    "Please specify a symbol\n"
                    "Usage: `/phantom\\_detail BTCUSDT`",
                    parse_mode='Markdown'
                )
                return
            
            symbol = ctx.args[0].upper()
            
            msg = f"👻 *Phantom Stats: {symbol}*\n"
            msg += "━" * 25 + "\n\n"
            
            # Get phantom tracker
            try:
                from phantom_trade_tracker import get_phantom_tracker
                phantom_tracker = get_phantom_tracker()
            except Exception as e:
                logger.error(f"Error importing phantom tracker: {e}")
                await update.message.reply_text("⚠️ Phantom tracker not available")
                return
            
            # Get symbol-specific statistics
            stats = phantom_tracker.get_phantom_stats(symbol)
            
            if stats['total'] == 0:
                msg += f"No phantom trades recorded for {symbol}\n"
                msg += "\n_Try another symbol or wait for more signals_"
                await self.safe_reply(update, msg)
                return
            
            # Overview for this symbol
            msg += "📊 *Overview*\n"
            msg += f"• Total signals: {stats['total']}\n"
            msg += f"• Executed: {stats['executed']}\n"
            msg += f"• Phantoms: {stats['rejected']}\n"
            if stats['total'] > 0:
                execution_rate = (stats['executed'] / stats['total']) * 100
                msg += f"• Execution rate: {execution_rate:.1f}%\n"
            msg += "\n"
            
            # Rejection analysis
            rejection_stats = stats['rejection_stats']
            if rejection_stats['total_rejected'] > 0:
                msg += "🚫 *Rejection Analysis*\n"
                msg += f"• Rejected trades: {rejection_stats['total_rejected']}\n"
                msg += f"• Would have won: {rejection_stats['would_have_won']}\n"
                msg += f"• Would have lost: {rejection_stats['would_have_lost']}\n"
                
                # Win rate of rejected trades
                if rejection_stats['total_rejected'] > 0:
                    rejected_wr = (rejection_stats['would_have_won'] / rejection_stats['total_rejected']) * 100
                    msg += f"• Rejected win rate: {rejected_wr:.1f}%\n"
                
                # Financial impact
                if rejection_stats['missed_profit_pct'] > 0:
                    msg += f"• Missed profit: +{rejection_stats['missed_profit_pct']:.2f}%\n"
                if rejection_stats['avoided_loss_pct'] > 0:
                    msg += f"• Avoided loss: -{rejection_stats['avoided_loss_pct']:.2f}%\n"
                
                # Net impact
                net_impact = rejection_stats['missed_profit_pct'] - rejection_stats['avoided_loss_pct']
                if net_impact != 0:
                    msg += f"• Net impact: {net_impact:+.2f}%\n"
                msg += "\n"
            
            # Recent phantom trades for this symbol
            if symbol in phantom_tracker.phantom_trades:
                recent_phantoms = phantom_tracker.phantom_trades[symbol][-5:]
                if recent_phantoms:
                    msg += "📜 *Recent Phantoms*\n"
                    for phantom in recent_phantoms:
                        if phantom.outcome:
                            outcome_emoji = "✅" if phantom.outcome == "win" else "❌"
                            msg += f"• Score {phantom.ml_score:.0f}: {outcome_emoji} "
                            msg += f"{phantom.side.upper()} {phantom.pnl_percent:+.2f}%\n"
                    msg += "\n"
            
            # Active phantom for this symbol
            if symbol in phantom_tracker.active_phantoms:
                phantom = phantom_tracker.active_phantoms[symbol]
                msg += "👀 *Currently Tracking*\n"
                msg += f"• {phantom.side.upper()} position\n"
                msg += f"• Entry: {phantom.entry_price:.4f}\n"
                msg += f"• ML Score: {phantom.ml_score:.1f}\n"
                msg += f"• Target: {phantom.take_profit:.4f}\n"
                msg += f"• Stop: {phantom.stop_loss:.4f}\n"
                msg += "\n"
            
            # ML insights
            msg += "💡 *ML Insights*\n"
            if rejection_stats['total_rejected'] > 0 and rejection_stats['would_have_won'] > rejection_stats['would_have_lost']:
                msg += "• ML may be too conservative\n"
                msg += "• Consider threshold adjustment\n"
            elif rejection_stats['total_rejected'] > 0 and rejection_stats['would_have_lost'] > rejection_stats['would_have_won']:
                msg += "• ML filtering effectively\n"
                msg += "• Avoiding losing trades\n"
            else:
                msg += "• Gathering more data\n"
                msg += "• Patterns emerging\n"
            
            await self.safe_reply(update, msg)
            
        except Exception as e:
            logger.error(f"Error in phantom_detail: {e}")
            import traceback
            logger.error(traceback.format_exc())
            await update.message.reply_text(f"Error getting phantom details: {str(e)[:100]}")
    
    async def evolution_performance(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Show ML Evolution shadow performance"""
        try:
            msg = "🧬 *ML Evolution Performance*\n"
            msg += "━" * 25 + "\n\n"
            
            try:
                from ml_evolution_tracker import get_evolution_tracker
                tracker = get_evolution_tracker()
                summary = tracker.get_performance_summary()
            except Exception as e:
                logger.error(f"Error getting evolution tracker: {e}")
                await update.message.reply_text("Evolution tracking not available")
                return
            
            if 'status' in summary:
                msg += summary['status']
            else:
                # Overview
                msg += "📊 *Shadow Mode Performance*\n"
                msg += f"• Total signals analyzed: {summary['total_signals']}\n"
                msg += f"• Agreement rate: {summary['agreement_rate']:.1f}%\n"
                msg += f"• Completed comparisons: {summary['completed_comparisons']}\n"
                msg += "\n"
                
                # Performance comparison
                if summary['completed_comparisons'] > 0:
                    msg += "🎯 *Win Rate Comparison*\n"
                    msg += f"• General model: {summary['general_win_rate']:.1f}%\n"
                    msg += f"• Evolution model: {summary['evolution_win_rate']:.1f}%\n"
                    
                    diff = summary['evolution_win_rate'] - summary['general_win_rate']
                    if diff > 0:
                        msg += f"• Evolution advantage: +{diff:.1f}%\n"
                    else:
                        msg += f"• General advantage: {abs(diff):.1f}%\n"
                    msg += "\n"
                
                # Symbol insights
                insights = summary.get('symbol_insights', {})
                if insights:
                    msg += "🔍 *Top Symbol Benefits*\n"
                    sorted_symbols = sorted(insights.items(), 
                                          key=lambda x: x[1]['evolution_advantage'], 
                                          reverse=True)[:5]
                    
                    for symbol, data in sorted_symbols:
                        advantage = data['evolution_advantage']
                        if advantage != 0:
                            msg += f"• {symbol}: "
                            if advantage > 0:
                                msg += f"+{advantage} better decisions\n"
                            else:
                                msg += f"{advantage} worse decisions\n"
                    msg += "\n"
                
                # Recommendation
                msg += "💡 *Recommendation*\n"
                msg += f"{summary['recommendation']}\n\n"
                
                msg += "_Shadow mode continues learning..._"
            
            await self.safe_reply(update, msg)
            
        except Exception as e:
            logger.error(f"Error in evolution_performance: {e}")
            import traceback
            logger.error(traceback.format_exc())
            await update.message.reply_text(f"Error getting evolution performance: {str(e)[:100]}")
    
    async def force_retrain_ml(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Force retrain ML models to reset feature expectations"""
        try:
            msg = "🔧 *ML Force Retrain*\n"
            msg += "━" * 25 + "\n\n"
            
            # Get ML scorer
            ml_scorer = self.shared.get("ml_scorer")
            if not ml_scorer:
                await update.message.reply_text("⚠️ ML scorer not available")
                return
            
            # Get current status before reset
            stats_before = ml_scorer.get_stats()
            
            msg += "📊 *Current Status*\n"
            msg += f"• Models: {', '.join(stats_before['models_active']) if stats_before['models_active'] else 'None'}\n"
            msg += f"• Feature version: {stats_before.get('model_feature_version', 'unknown')}\n"
            msg += f"• Feature count: {stats_before.get('feature_count', 'unknown')}\n"
            msg += f"• Completed trades: {stats_before['completed_trades']}\n\n"
            
            # Force retrain
            ml_scorer.force_retrain_models()
            
            msg += "✅ *Actions Taken*\n"
            msg += "• Cleared existing models\n"
            msg += "• Reset scaler\n"
            msg += "• Cleared Redis cache\n"
            msg += "• Reset to original features (22)\n\n"
            
            msg += "📝 *What Happens Next*\n"
            msg += "• Models will use rule-based scoring\n"
            msg += "• Will retrain on next trade completion\n"
            msg += "• Will detect available features automatically\n"
            msg += "• No interruption to trading\n\n"
            
            msg += "⚡ *Commands*\n"
            msg += "• `/ml` - Check ML status\n"
            msg += "• `/stats` - View trading stats"
            
            await self.safe_reply(update, msg)
            
        except Exception as e:
            logger.error(f"Error in force_retrain_ml: {e}")
            await update.message.reply_text(f"Error forcing ML retrain: {str(e)[:100]}")
    
    async def ml_patterns(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Show detailed ML patterns and insights for both strategies"""
        try:
            response_text = "🧠 *ML Pattern Analysis*\n"
            response_text += "━" * 25 + "\n\n"

            # Get ML scorers
            pullback_scorer = get_immediate_scorer()
            mean_reversion_scorer = get_mean_reversion_scorer()

            scorers = {
                "Pullback Strategy": pullback_scorer,
                "Mean Reversion Strategy": mean_reversion_scorer
            }

            for strategy_name, scorer in scorers.items():
                response_text += f"*{strategy_name}:*\n"
                if not scorer or not hasattr(scorer, 'get_learned_patterns'):
                    response_text += "  ❌ *Pattern Analysis Not Available*\n"
                    response_text += "  _ML system not yet initialized or trained._\n\n"
                    continue

                patterns = scorer.get_learned_patterns()

                if not patterns or all(not v for v in patterns.values()):
                    stats = scorer.get_stats()
                    response_text += f"  📊 *Collecting Data...*\n"
                    response_text += f"  • Completed trades: {stats.get('completed_trades', 0)}\n"
                    response_text += f"  • Status: {stats.get('status', 'Learning')}\n"
                    response_text += "  _Patterns will emerge after more trades._\n\n"
                    continue

                # Feature Importance (Top 10)
                if patterns.get('feature_importance'):
                    response_text += "  📊 *Feature Importance (Top 10)*\n"
                    response_text += "  _What drives winning trades_\n"
                    
                    for i, (feat, imp) in enumerate(list(patterns['feature_importance'].items())[:10], 1):
                        feat_name = feat.replace('_', ' ').title()
                        bar_length = int(imp / 10)
                        bar = '█' * bar_length + '░' * (10 - bar_length)
                        response_text += f"  {i}. {feat_name}\n"
                        response_text += f"     {bar} {imp:.1f}%\n"
                    response_text += "\n"
                
                # Time Analysis
                time_patterns = patterns.get('time_patterns', {})
                if time_patterns:
                    response_text += "  ⏰ *Time-Based Insights*\n"
                    
                    if time_patterns.get('best_hours'):
                        response_text += "  🌟 *Golden Hours*\n"
                        for hour, stats in list(time_patterns['best_hours'].items())[:5]:
                            response_text += f"  • {hour} → {stats}\n"
                        response_text += "\n"
                    
                    if time_patterns.get('worst_hours'):
                        response_text += "  ⚠️ *Danger Hours*\n"
                        for hour, stats in list(time_patterns['worst_hours'].items())[:5]:
                            response_text += f"  • {hour} → {stats}\n"
                        response_text += "\n"
                    
                    if time_patterns.get('session_performance'):
                        response_text += "  🌍 *Market Sessions*\n"
                        for session, perf in time_patterns['session_performance'].items():
                            if 'WR' in perf:
                                wr = float(perf.split('%')[0].split()[-1])
                                emoji = '🟢' if wr >= 50 else '🔴'
                            else:
                                emoji = '⚪'
                            response_text += f"  {emoji} {session}: {perf}\n"
                        response_text += "\n"
                
                # Market Conditions
                market_conditions = patterns.get('market_conditions', {})
                if market_conditions:
                    response_text += "  📈 *Market Condition Analysis*\n"
                    
                    if market_conditions.get('volatility_impact'):
                        response_text += "  🌊 *Volatility Performance*\n"
                        for vol_type, stats in market_conditions['volatility_impact'].items():
                            if 'WR' in stats:
                                wr = float(stats.split('%')[0].split()[-1])
                                emoji = '✅' if wr >= 50 else '❌'
                            else:
                                emoji = '➖'
                            response_text += f"  {emoji} {vol_type.title()}: {stats}\n"
                        response_text += "\n"
                    
                    if market_conditions.get('volume_impact'):
                        response_text += "  📊 *Volume Analysis*\n"
                        for vol_type, stats in market_conditions['volume_impact'].items():
                            vol_name = vol_type.replace('_', ' ').title()
                            if 'WR' in stats:
                                wr = float(stats.split('%')[0].split()[-1])
                                emoji = '✅' if wr >= 50 else '❌'
                            else:
                                emoji = '➖'
                            response_text += f"  {emoji} {vol_name}: {stats}\n"
                        response_text += "\n"
                    
                    if market_conditions.get('trend_impact'):
                        response_text += "  📉 *Trend Analysis*\n"
                        for trend_type, stats in market_conditions['trend_impact'].items():
                            trend_name = trend_type.replace('_', ' ').title()
                            if 'WR' in stats:
                                wr = float(stats.split('%')[0].split()[-1])
                                emoji = '✅' if wr >= 50 else '❌'
                            else:
                                emoji = '➖'
                            response_text += f"  {emoji} {trend_name}: {stats}\n"
                        response_text += "\n"
                
                # Winning vs Losing Patterns
                if patterns.get('winning_patterns') or patterns.get('losing_patterns'):
                    response_text += "  🎯 *Trade Outcome Patterns*\n"
                    
                    if patterns.get('winning_patterns'):
                        response_text += "  ✅ *Common in Winners*\n"
                        for pattern in patterns['winning_patterns']:
                            response_text += f"  • {pattern}\n"
                        response_text += "\n"
                    
                    if patterns.get('losing_patterns'):
                        response_text += "  ❌ *Common in Losers*\n"
                        for pattern in patterns['losing_patterns']:
                            response_text += f"  • {pattern}\n"
                        response_text += "\n"
                
                # Summary insights
                response_text += "  💡 *Key Takeaways*\n"
                response_text += "  • Focus on high-importance features\n"
                response_text += "  • Trade during golden hours\n"
                response_text += "  • Adapt to market conditions\n"
                response_text += "  • Avoid danger patterns\n\n"
            
            response_text += "--- End of " + strategy_name + " ---\n\n"

            await update.message.reply_text(response_text, parse_mode=ParseMode.MARKDOWN)
            
        except Exception as e:
            logger.error(f"Error in ml_patterns: {e}")
            await update.message.reply_text("Error getting ML patterns")
    
    async def ml_retrain_info(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Show ML retrain countdown information"""
        try:
            msg = "🔄 *ML Retrain Status*\n"
            msg += "━" * 25 + "\n\n"
            
            # Check if ML scorer is available
            ml_scorer = self.shared.get("ml_scorer")
            
            if not ml_scorer or not hasattr(ml_scorer, 'get_retrain_info'):
                msg += "❌ *ML System Not Available*\n\n"
                msg += "ML retraining info requires:\n"
                msg += "• ML system enabled\n"
                msg += "• Bot running with ML\n"
                await self.safe_reply(update, msg)
                return
            
            # Get retrain info
            info = ml_scorer.get_retrain_info()
            
            # Current status
            msg += "📊 *Current Status*\n"
            msg += f"• ML Ready: {'✅ Yes' if info['is_ml_ready'] else '❌ No'}\n"
            msg += f"• Executed trades: {info['completed_trades']}\n"
            msg += f"• Phantom trades: {info['phantom_count']}\n"
            msg += f"• Combined total: {info['total_combined']}\n"
            msg += "\n"
            
            # Training history
            if info['is_ml_ready']:
                msg += "📈 *Training History*\n"
                msg += f"• Last trained at: {info['last_train_count']} trades\n"
                trades_since = info['total_combined'] - info['last_train_count']
                msg += f"• Trades since last: {trades_since}\n"
                msg += "\n"
            
            # Next retrain countdown
            msg += "⏳ *Next Retrain*\n"
            if info['trades_until_next_retrain'] == 0:
                if info['is_ml_ready']:
                    msg += "🟢 **Ready to retrain!**\n"
                    msg += "Will retrain on next trade completion\n"
                else:
                    msg += "🟢 **Ready for initial training!**\n"
                    msg += "Will train on next trade completion\n"
            else:
                msg += f"• Trades needed: **{info['trades_until_next_retrain']}**\n"
                msg += f"• Will retrain at: {info['next_retrain_at']} total trades\n"
                
                # Progress bar - calculate based on actual retrain interval
                if info['is_ml_ready']:
                    # For retrain: how far through the 100-trade cycle
                    trades_in_cycle = ml_scorer.RETRAIN_INTERVAL - info['trades_until_next_retrain']
                    progress = (trades_in_cycle / ml_scorer.RETRAIN_INTERVAL) * 100
                else:
                    # For initial training: progress toward MIN_TRADES_FOR_ML
                    trades_so_far = ml_scorer.MIN_TRADES_FOR_ML - info['trades_until_next_retrain']
                    progress = (trades_so_far / ml_scorer.MIN_TRADES_FOR_ML) * 100
                
                progress = max(0, min(100, progress))
                filled = int(progress / 10)
                bar = '█' * filled + '░' * (10 - filled)
                msg += f"• Progress: {bar} {progress:.0f}%\n"
            
            msg += "\n"
            
            # Info about retraining
            msg += "ℹ️ *Retrain Info*\n"
            if not info['is_ml_ready']:
                msg += f"• Initial training after {ml_scorer.MIN_TRADES_FOR_ML} trades\n"
            msg += f"• Retrain interval: Every {ml_scorer.RETRAIN_INTERVAL} trades\n"
            msg += "• Both executed and phantom trades count\n"
            msg += "• Models improve with each retrain\n"
            msg += "\n"
            
            # Commands
            msg += "⚡ *Commands*\n"
            msg += "• `/force_retrain` - Force immediate retrain\n"
            msg += "• `/ml` - View ML status\n"
            msg += "• `/phantom` - View phantom trades"
            
            await self.safe_reply(update, msg)
            
        except Exception as e:
            logger.error(f"Error in ml_retrain_info: {e}")
            await update.message.reply_text("Error getting ML retrain info")
    
    async def cluster_status(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Show symbol cluster status using hardcoded clusters"""
        try:
            msg = "🎯 *Symbol Cluster Status*\n"
            msg += "━" * 25 + "\n\n"
            
            # Use hardcoded clusters
            try:
                from hardcoded_clusters import (get_hardcoded_clusters, get_cluster_name, 
                                               get_symbol_cluster, get_cluster_description,
                                               BLUE_CHIP, STABLE, MEME_VOLATILE, MID_CAP)
                
                # Get all hardcoded clusters
                all_clusters = get_hardcoded_clusters()
                
                # Count hardcoded symbols
                cluster_counts = {
                    1: len(BLUE_CHIP),
                    2: len(STABLE),
                    3: len(MEME_VOLATILE),
                    4: len(MID_CAP),
                    5: 0  # Will be calculated from active symbols
                }
                
                # Count small cap symbols from trading symbols
                trading_symbols = self.shared.get("frames", {}).keys()
                known_symbols = set(all_clusters.keys())
                small_cap_count = len([s for s in trading_symbols if s not in known_symbols])
                cluster_counts[5] = small_cap_count
                
                msg += "📊 *Hardcoded Cluster Distribution*\n"
                for i in range(1, 6):
                    name = get_cluster_name(i)
                    count = cluster_counts[i]
                    if count > 0:
                        msg += f"• {name}: {count} symbols\n"
                
                msg += "\n🔍 *Sample Symbols by Cluster*\n"
                
                # Show examples from each cluster
                msg += f"\n*{get_cluster_name(1)}:*\n"
                for symbol in BLUE_CHIP[:5]:
                    msg += f"• {symbol}\n"
                    
                msg += f"\n*{get_cluster_name(3)}:*\n"
                for symbol in MEME_VOLATILE[:5]:
                    msg += f"• {symbol}\n"
                    
                msg += f"\n*{get_cluster_name(4)}:*\n"  
                for symbol in MID_CAP[:5]:
                    msg += f"• {symbol}\n"
                
                # Show current positions with clusters
                positions = self.shared.get("book", {}).positions
                if positions:
                    msg += "\n📈 *Your Open Positions:*\n"
                    for symbol in list(positions.keys())[:10]:
                        cluster_id = get_symbol_cluster(symbol)
                        cluster_name = get_cluster_name(cluster_id)
                        msg += f"• {symbol}: {cluster_name}\n"
                
                msg += "\n✅ *Using hardcoded clusters*\n"
                msg += "_No generation needed - always available_"
                    
            except Exception as e:
                logger.error(f"Error loading clusters: {e}")
                msg += "❌ Error loading cluster data\n"
                msg += "Basic clustering may still be active\n"
            
            msg += "\n💡 *Commands*\n"
            msg += "• `/update_clusters` - Update clusters\n"
            msg += "• `/ml` - View ML status"
            
            await self.safe_reply(update, msg)
            
        except Exception as e:
            logger.error(f"Error in cluster_status: {e}")
            await update.message.reply_text("Error getting cluster status")
    
    async def update_clusters(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Inform user that hardcoded clusters are always up to date"""
        try:
            from hardcoded_clusters import get_hardcoded_clusters, BLUE_CHIP, STABLE, MEME_VOLATILE, MID_CAP
            
            msg = "✅ *Cluster Update Status*\n"
            msg += "━" * 25 + "\n\n"
            msg += "📌 *Using Hardcoded Clusters*\n"
            msg += "No update needed - clusters are hardcoded!\n\n"
            
            msg += "📊 *Current Distribution:*\n"
            msg += f"• Blue Chip: {len(BLUE_CHIP)} symbols\n"
            msg += f"• Stable: {len(STABLE)} symbols\n"
            msg += f"• Meme/Volatile: {len(MEME_VOLATILE)} symbols\n"
            msg += f"• Mid-Cap: {len(MID_CAP)} symbols\n"
            msg += f"• Small Cap: All others\n\n"
            
            msg += "💡 *Benefits:*\n"
            msg += "• Always available\n"
            msg += "• No generation needed\n"
            msg += "• Consistent classification\n"
            msg += "• Based on market research\n\n"
            
            msg += "Use `/clusters` to view full details"
            
            await update.message.reply_text(msg)
                
        except Exception as e:
            logger.error(f"Error in update_clusters: {e}")
            await update.message.reply_text("Error triggering cluster update")
    
    async def set_ml_threshold(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Set ML threshold for signal acceptance"""
        try:
            # Get ML scorer
            ml_scorer = self.shared.get("ml_scorer")
            if not ml_scorer:
                await update.message.reply_text("❌ ML system not available")
                return
            
            # Check if arguments provided
            if not ctx.args:
                # Show current threshold
                msg = "🤖 *ML Threshold Settings*\n"
                msg += "━" * 25 + "\n\n"
                msg += f"Current threshold: {ml_scorer.min_score}\n\n"
                msg += "📊 *Threshold Guide:*\n"
                msg += "• 50-60: Very lenient (more signals)\n"
                msg += "• 60-70: Moderate\n"
                msg += "• 70-80: Standard (default)\n" 
                msg += "• 80-90: Conservative\n"
                msg += "• 90-100: Very strict (fewer signals)\n\n"
                msg += "Usage: `/set_ml_threshold 75`"
                await self.safe_reply(update, msg)
                return
            
            try:
                new_threshold = float(ctx.args[0])
                
                # Validate threshold
                if new_threshold < 0 or new_threshold > 100:
                    await update.message.reply_text("❌ Threshold must be between 0 and 100")
                    return
                
                # Update threshold
                old_threshold = ml_scorer.min_score
                ml_scorer.min_score = new_threshold
                
                # Save to Redis if available
                if ml_scorer.redis_client:
                    try:
                        ml_scorer.redis_client.set('iml:threshold', new_threshold)
                    except:
                        pass
                
                # Prepare response
                msg = f"✅ *ML Threshold Updated*\n\n"
                msg += f"• Old threshold: {old_threshold}\n"
                msg += f"• New threshold: {new_threshold}\n\n"
                
                # Add interpretation
                if new_threshold < 60:
                    msg += "⚡ Very lenient - Expect more signals with higher risk\n"
                elif new_threshold < 70:
                    msg += "📊 Moderate - Balanced approach\n"
                elif new_threshold < 80:
                    msg += "✅ Standard - Good balance of quality and quantity\n"
                elif new_threshold < 90:
                    msg += "🛡️ Conservative - Higher quality signals only\n"
                else:
                    msg += "🏆 Very strict - Only the best signals\n"
                
                # Add stats if available
                stats = ml_scorer.get_stats()
                if stats.get('completed_trades', 0) > 0:
                    msg += f"\nBased on {stats['completed_trades']} completed trades"
                
                await self.safe_reply(update, msg)
                logger.info(f"ML threshold updated from {old_threshold} to {new_threshold}")
                
            except ValueError:
                await update.message.reply_text("❌ Invalid threshold. Please provide a number between 0 and 100")
            
        except Exception as e:
            logger.error(f"Error in set_ml_threshold: {e}")
            await update.message.reply_text("Error updating ML threshold")
    
    async def htf_sr_status(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Show HTF support/resistance status"""
        try:
            from multi_timeframe_sr import mtf_sr
            
            msg = "📊 *HTF Support/Resistance Status*\n"
            msg += "━" * 30 + "\n\n"
            
            # Count symbols with levels
            symbols_with_levels = 0
            total_levels = 0
            
            for symbol, levels in mtf_sr.sr_levels.items():
                if levels:
                    symbols_with_levels += 1
                    total_levels += len(levels)
            
            msg += f"📈 *Overview:*\n"
            msg += f"• Symbols analyzed: {len(mtf_sr.sr_levels)}\n"
            msg += f"• Symbols with levels: {symbols_with_levels}\n"
            msg += f"• Total levels tracked: {total_levels}\n"
            msg += f"• Update interval: {mtf_sr.update_interval} candles\n\n"
            
            # Show specific symbol if provided
            if ctx.args:
                symbol = ctx.args[0].upper()
                if symbol in mtf_sr.sr_levels:
                    # Get current price from frames
                    frames = self.shared.get("frames", {})
                    current_price = None
                    if symbol in frames and not frames[symbol].empty:
                        current_price = frames[symbol]['close'].iloc[-1]
                    
                    if current_price:
                        msg += f"📍 *{symbol} Levels:*\n"
                        msg += f"Current Price: {current_price:.4f}\n"
                        
                        # Get price-validated levels
                        validated_levels = mtf_sr.get_price_validated_levels(symbol, current_price)
                        
                        # Group by type
                        resistance_levels = [(l, s) for l, s, t in validated_levels if t == 'resistance']
                        support_levels = [(l, s) for l, s, t in validated_levels if t == 'support']
                        
                        # Show top 5 of each
                        if resistance_levels:
                            msg += "\n🔴 *Resistance (above price):*\n"
                            for level, strength in resistance_levels[:5]:
                                distance_pct = ((level - current_price) / current_price) * 100
                                msg += f"• {level:.4f} (strength: {strength:.1f}, +{distance_pct:.2f}%)\n"
                        else:
                            msg += "\n🔴 *Resistance:* None above current price\n"
                        
                        if support_levels:
                            msg += "\n🟢 *Support (below price):*\n"
                            for level, strength in support_levels[:5]:
                                distance_pct = ((current_price - level) / level) * 100
                                msg += f"• {level:.4f} (strength: {strength:.1f}, -{distance_pct:.2f}%)\n"
                        else:
                            msg += "\n🟢 *Support:* None below current price\n"
                    else:
                        msg += f"❌ No price data available for {symbol}"
                else:
                    msg += f"❌ No HTF levels found for {symbol}"
            else:
                # Show example usage
                msg += "💡 *Usage:*\n"
                msg += "`/htf_sr BTCUSDT` - Show levels for specific symbol\n"
                msg += "`/update_htf_sr` - Force update all HTF levels"
            
            await self.safe_reply(update, msg)
            
        except Exception as e:
            logger.error(f"Error in htf_sr_status: {e}")
            await update.message.reply_text("Error fetching HTF S/R status")
    
    async def update_htf_sr(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Force update HTF support/resistance levels"""
        try:
            # Get frames data
            frames = self.shared.get("frames", {})
            if not frames:
                await update.message.reply_text("❌ No candle data available")
                return
            
            # Send initial message
            msg = await update.message.reply_text("🔄 Updating HTF S/R levels for all symbols...")
            
            # Update HTF levels
            from multi_timeframe_sr import initialize_all_sr_levels
            results = initialize_all_sr_levels(frames)
            
            # Update message with results
            symbols_with_levels = [sym for sym, count in results.items() if count > 0]
            
            result_msg = "✅ *HTF S/R Update Complete*\n"
            result_msg += "━" * 25 + "\n\n"
            result_msg += f"📊 *Results:*\n"
            result_msg += f"• Symbols analyzed: {len(results)}\n"
            result_msg += f"• Found levels: {len(symbols_with_levels)} symbols\n"
            result_msg += f"• Total levels: {sum(results.values())}\n\n"
            
            # Show top 5 symbols by level count
            if symbols_with_levels:
                top_symbols = sorted(results.items(), key=lambda x: x[1], reverse=True)[:5]
                result_msg += "🏆 *Top Symbols by Level Count:*\n"
                for sym, count in top_symbols:
                    if count > 0:
                        result_msg += f"• {sym}: {count} levels\n"
            
            result_msg += "\nUse `/htf_sr [symbol]` to view specific levels"
            
            await msg.edit_text(result_msg)
            
        except Exception as e:
            logger.error(f"Error in update_htf_sr: {e}")
            await update.message.reply_text("Error updating HTF S/R levels")

    async def mr_ml_stats(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Show detailed Mean Reversion ML statistics"""
        try:
            msg = "🔄 *Mean Reversion ML Status*\n"
            msg += "━" * 30 + "\n\n"

            # Get mean reversion scorer
            try:
                mean_reversion_scorer = get_mean_reversion_scorer()
            except Exception as e:
                msg += f"❌ *Error getting Mean Reversion ML:* {e}\n"
                await self.safe_reply(update, msg)
                return

            # Get comprehensive stats
            stats = mean_reversion_scorer.get_stats()

            # Status section
            if stats.get('is_ml_ready'):
                msg += "✅ *Status: ACTIVE & LEARNING*\n"
                msg += f"• Models trained: {len(stats.get('models_active', []))}/3\n"
                msg += f"• Model types: {', '.join(stats.get('models_active', []))}\n"
            else:
                msg += "📊 *Status: COLLECTING DATA*\n"
                remaining = max(0, stats.get('min_trades_for_ml', 50) - stats.get('completed_trades', 0))
                msg += f"• Trades needed: {remaining} more\n"

            msg += "\n📊 *Trade Statistics:*\n"
            msg += f"• Total trades: {stats.get('completed_trades', 0)}\n"
            msg += f"• Last training: {stats.get('last_train_count', 0)} trades\n"

            if 'recent_win_rate' in stats and stats['recent_trades'] > 0:
                msg += f"• Recent win rate: {stats['recent_win_rate']:.1f}% ({stats['recent_trades']} trades)\n"

            # Retrain info
            msg += "\n🔄 *Retrain Schedule:*\n"
            msg += f"• Retrain interval: {stats.get('retrain_interval', 25)} trades\n"
            next_retrain = stats.get('next_retrain_in', 0)
            if next_retrain > 0:
                msg += f"• Next retrain in: {next_retrain} trades\n"
            else:
                msg += "• Ready for retrain! \ud83c\udf86\n"

            # Scoring configuration
            msg += "\n⚙️ *ML Configuration:*\n"
            msg += f"• Score threshold: {stats.get('current_threshold', 70):.0f}\n"
            msg += f"• Min trades for ML: {stats.get('min_trades_for_ml', 50)}\n"

            # Feature info
            msg += "\n🧪 *Features Used:*\n"
            msg += "• Range width & strength\n"
            msg += "• RSI & Stochastic extremes\n"
            msg += "• Volume confirmation\n"
            msg += "• Reversal candle quality\n"
            msg += "• Session & time context\n"

            await self.safe_reply(update, msg)

        except Exception as e:
            logger.error(f"Error in mr_ml_stats: {e}")
            await update.message.reply_text("Error getting Mean Reversion ML statistics")

    async def mr_retrain(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Force retrain Mean Reversion ML models"""
        try:
            msg = "🔄 *Mean Reversion ML Retrain*\n"
            msg += "━" * 25 + "\n\n"

            # Get mean reversion scorer
            try:
                mean_reversion_scorer = get_mean_reversion_scorer()
            except Exception as e:
                msg += f"❌ *Error:* {e}\n"
                await self.safe_reply(update, msg)
                return

            # Check if retrain is possible
            retrain_info = mean_reversion_scorer.get_retrain_info()

            if not retrain_info['can_train']:
                msg += "⚠️ *Cannot Retrain Yet*\n"
                remaining = max(0, 50 - retrain_info['total_trades'])
                msg += f"Need {remaining} more trades before first training.\n"
                await self.safe_reply(update, msg)
                return

            # Show pre-retrain status
            msg += "📊 *Pre-Retrain Status:*\n"
            msg += f"• Available trades: {retrain_info['total_trades']}\n"
            msg += f"• Last training: {retrain_info['last_train_at']} trades\n"

            # Attempt retrain
            msg += "\n🔄 *Starting Retrain...*\n"
            temp_msg = await update.message.reply_text(msg)

            try:
                success = mean_reversion_scorer.startup_retrain()

                if success:
                    msg += "✅ *Retrain Successful!*\n"
                    msg += "\n�\udf86 *Post-Retrain Status:*\n"

                    # Get updated stats
                    updated_stats = mean_reversion_scorer.get_stats()
                    msg += f"• Models active: {len(updated_stats.get('models_active', []))}\n"
                    msg += f"• Trained on: {updated_stats.get('last_train_count', 0)} trades\n"
                    msg += f"• Status: {updated_stats.get('status', 'Unknown')}\n"

                else:
                    msg += "❌ *Retrain Failed*\n"
                    msg += "Check logs for details.\n"

            except Exception as retrain_error:
                msg += f"❌ *Retrain Error:* {retrain_error}\n"

            await temp_msg.edit_text(msg)

        except Exception as e:
            logger.error(f"Error in mr_retrain: {e}")
            await update.message.reply_text("Error during Mean Reversion ML retrain")

    async def enhanced_mr_stats(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Show Enhanced Mean Reversion ML statistics"""
        try:
            msg = "🧠 *Enhanced Mean Reversion ML Status*\n"
            msg += "━" * 35 + "\n\n"

            # Check if enhanced system is available
            try:
                from enhanced_mr_scorer import get_enhanced_mr_scorer
                enhanced_mr_scorer = get_enhanced_mr_scorer()
            except ImportError:
                msg += "❌ *Enhanced MR ML not available*\n"
                msg += "Please check if enhanced_mr_scorer.py is installed.\n"
                await self.safe_reply(update, msg)
                return

            # Get enhanced stats
            stats = enhanced_mr_scorer.get_enhanced_stats()

            # Status section
            if stats.get('is_ml_ready'):
                msg += "✅ *Status: ADVANCED ML ACTIVE*\n"
                msg += f"• Strategy: {stats.get('strategy', 'Enhanced Mean Reversion')}\n"
                msg += f"• Models: {stats.get('model_count', 0)}/4 active\n"
                msg += f"• Features: {stats.get('feature_count', 30)}+ enhanced features\n"
            else:
                msg += f"📚 *Status: {stats.get('status', 'Learning')}*\n"
                msg += f"• Trades needed: {stats.get('min_trades_for_ml', 30)}\n"
                msg += f"• Progress: {stats.get('completed_trades', 0)}/{stats.get('min_trades_for_ml', 30)}\n"

            msg += "\n📊 *Performance Metrics:*\n"
            msg += f"• Completed trades: {stats.get('completed_trades', 0)}\n"
            msg += f"• Current threshold: {stats.get('current_threshold', 72):.0f}%\n"
            msg += f"• Threshold range: {stats.get('min_threshold', 65)}-{stats.get('max_threshold', 88)}%\n"

            if stats.get('recent_win_rate', 0) > 0:
                msg += f"• Recent win rate: {stats.get('recent_win_rate', 0):.1f}%\n"
                msg += f"• Sample size: {stats.get('recent_trades', 0)} trades\n"

            # Model details
            if stats.get('models_active'):
                msg += "\n🤖 *Active Models:*\n"
                for model in stats.get('models_active', []):
                    msg += f"• {model.replace('_', ' ').title()}\n"

            # Training info
            msg += "\n🔄 *Training Info:*\n"
            msg += f"• Last trained: {stats.get('last_train_count', 0)} trades\n"
            msg += f"• Retrain interval: {stats.get('retrain_interval', 50)} trades\n"
            msg += f"• Next retrain in: {stats.get('trades_until_retrain', 'N/A')} trades\n"

            await self.safe_reply(update, msg)

        except Exception as e:
            logger.error(f"Error in enhanced_mr_stats: {e}")
            await update.message.reply_text("Error getting Enhanced MR ML stats")

    async def mr_phantom_stats(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Show Mean Reversion phantom trade statistics"""
        try:
            msg = "👻 *Mean Reversion Phantom Trades*\n"
            msg += "━" * 30 + "\n\n"

            # Get MR phantom tracker
            try:
                from mr_phantom_tracker import get_mr_phantom_tracker
                mr_phantom_tracker = get_mr_phantom_tracker()
            except ImportError:
                msg += "❌ *MR Phantom Tracker not available*\n"
                await self.safe_reply(update, msg)
                return

            # Get phantom stats
            phantom_stats = mr_phantom_tracker.get_mr_phantom_stats()

            msg += f"📈 *Overall Statistics:*\n"
            total_trades = phantom_stats.get('total_mr_trades', 0)
            executed_trades = phantom_stats.get('executed', 0)
            rejected_trades = phantom_stats.get('rejected', 0)
            
            msg += f"• Total MR signals: {total_trades}\n"
            msg += f"• Executed: {executed_trades}\n"
            msg += f"• Rejected: {rejected_trades}\n"
            
            # Add verification that counts add up
            if total_trades != (executed_trades + rejected_trades):
                msg += f"⚠️ *Count mismatch detected: {total_trades} ≠ {executed_trades + rejected_trades}*\n"

            # Outcome analysis - show all rates, not just non-zero
            outcome = phantom_stats.get('outcome_analysis', {})
            if phantom_stats.get('executed', 0) > 0 or phantom_stats.get('rejected', 0) > 0:
                msg += f"\n📊 *Performance Analysis:*\n"
                
                # Show executed trades performance
                executed_count = phantom_stats.get('executed', 0)
                if executed_count > 0:
                    executed_wr = outcome.get('executed_win_rate', 0)
                    msg += f"• Executed trades: {executed_count} (Win rate: {executed_wr:.1f}%)\n"
                
                # Show rejected trades performance
                rejected_count = phantom_stats.get('rejected', 0)
                if rejected_count > 0:
                    rejected_wr = outcome.get('rejected_would_win_rate', 0)
                    msg += f"• Rejected trades: {rejected_count} (Would-be win rate: {rejected_wr:.1f}%)\n"

            # MR-specific metrics
            mr_metrics = phantom_stats.get('mr_specific_metrics', {})
            if mr_metrics:
                msg += f"\n📉 *Mean Reversion Specific:*\n"
                msg += f"• Range breakouts during trade: {mr_metrics.get('range_breakout_during_trade', 0)}\n"
                msg += f"• Timeout closures: {mr_metrics.get('timeout_closures', 0)}\n"
                msg += f"• High confidence ranges: {mr_metrics.get('high_confidence_ranges', 0)}\n"
                msg += f"• Boundary entries: {mr_metrics.get('boundary_entries', 0)}\n"

                if mr_metrics.get('boundary_entry_win_rate'):
                    msg += f"• Boundary entry win rate: {mr_metrics.get('boundary_entry_win_rate', 0):.1f}%\n"

            # Range performance breakdown
            range_perf = phantom_stats.get('range_performance', {})
            if range_perf:
                msg += f"\n🎯 *Range Quality Performance:*\n"
                for quality, data in range_perf.items():
                    if isinstance(data, dict) and data.get('wins') is not None:
                        total = data.get('wins', 0) + data.get('losses', 0)
                        if total > 0:
                            wr = (data.get('wins', 0) / total) * 100
                            msg += f"• {quality.replace('_', ' ').title()}: {wr:.1f}% ({total} trades)\n"

            await self.safe_reply(update, msg)

        except Exception as e:
            logger.error(f"Error in mr_phantom_stats: {e}")
            await update.message.reply_text("Error getting MR phantom stats")

    async def parallel_performance(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Show parallel strategy system performance comparison"""
        try:
            msg = "⚡ *Parallel Strategy Performance*\n"
            msg += "━" * 35 + "\n\n"

            # Check if enhanced parallel system is available
            try:
                from enhanced_mr_scorer import get_enhanced_mr_scorer
                from ml_signal_scorer_immediate import get_immediate_scorer
                enhanced_mr = get_enhanced_mr_scorer()
                pullback_ml = get_immediate_scorer()
            except ImportError:
                msg += "❌ *Enhanced parallel system not available*\n"
                await self.safe_reply(update, msg)
                return

            # Get stats from both systems
            pullback_stats = pullback_ml.get_stats()
            mr_stats = enhanced_mr.get_enhanced_stats()

            msg += "🎯 *Pullback Strategy (Trending Markets):*\n"
            msg += f"• Status: {pullback_stats.get('status', 'Unknown')}\n"
            msg += f"• Trades: {pullback_stats.get('completed_trades', 0)}\n"
            msg += f"• Threshold: {pullback_stats.get('current_threshold', 70):.0f}%\n"
            if pullback_stats.get('recent_win_rate', 0) > 0:
                msg += f"• Recent WR: {pullback_stats.get('recent_win_rate', 0):.1f}%\n"

            msg += "\n📉 *Mean Reversion Strategy (Ranging Markets):*\n"
            msg += f"• Status: {mr_stats.get('status', 'Unknown')}\n"
            msg += f"• Trades: {mr_stats.get('completed_trades', 0)}\n"
            msg += f"• Threshold: {mr_stats.get('current_threshold', 72):.0f}%\n"
            if mr_stats.get('recent_win_rate', 0) > 0:
                msg += f"• Recent WR: {mr_stats.get('recent_win_rate', 0):.1f}%\n"

            # Combined performance
            total_trades = pullback_stats.get('completed_trades', 0) + mr_stats.get('completed_trades', 0)
            msg += f"\n📊 *Combined System:*\n"
            msg += f"• Total trades: {total_trades}\n"
            msg += f"• Strategy coverage: Full market conditions\n"
            msg += f"• Adaptive routing: Regime-based selection\n"

            # Active models summary
            pullback_models = len(pullback_stats.get('models_active', []))
            mr_models = mr_stats.get('model_count', 0)
            msg += f"• Active ML models: {pullback_models + mr_models} total\n"
            msg += f"  - Pullback: {pullback_models}/3 models\n"
            msg += f"  - Mean Reversion: {mr_models}/4 models\n"

            await self.safe_reply(update, msg)

        except Exception as e:
            logger.error(f"Error in parallel_performance: {e}")
            await update.message.reply_text("Error getting parallel performance stats")

    async def regime_analysis(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Show current market regime analysis for top symbols"""
        try:
            msg = "🌐 *Market Regime Analysis*\n"
            msg += "━" * 30 + "\n\n"

            # Check if enhanced regime detection is available
            try:
                from enhanced_market_regime import get_enhanced_market_regime, get_regime_summary
            except ImportError:
                msg += "❌ *Enhanced regime detection not available*\n"
                await self.safe_reply(update, msg)
                return

            # Get current frames from shared data (if available)
            book = self.shared.get("book")
            if not book or not hasattr(book, 'positions'):
                msg += "❌ *No market data available*\n"
                await self.safe_reply(update, msg)
                return

            # Analyze regime for symbols with positions or top symbols
            frames = self.shared.get("frames", {})
            symbols_to_analyze = list(book.positions.keys()) if book.positions else ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']

            regime_summary = {}
            for symbol in symbols_to_analyze[:8]:  # Limit to 8 symbols
                if symbol in frames and not frames[symbol].empty:
                    try:
                        regime_analysis = get_enhanced_market_regime(frames[symbol], symbol)
                        regime_summary[symbol] = regime_analysis
                    except Exception as e:
                        logger.debug(f"Regime analysis failed for {symbol}: {e}")

            if regime_summary:
                msg += "📊 *Current Regime Analysis:*\n"
                for symbol, analysis in regime_summary.items():
                    msg += f"\n**{symbol}:**\n"
                    msg += f"• Regime: {analysis.primary_regime.title()}\n"
                    msg += f"• Confidence: {analysis.regime_confidence:.0%}\n"
                    msg += f"• Strategy: {analysis.recommended_strategy.replace('_', ' ').title()}\n"

                    if analysis.primary_regime == "ranging":
                        msg += f"• Range quality: {analysis.range_quality}\n"
                    elif analysis.primary_regime == "trending":
                        msg += f"• Trend strength: {analysis.trend_strength:.0f}%\n"

                    msg += f"• Volatility: {analysis.volatility_level}\n"

                # Overall summary
                regimes = [analysis.primary_regime for analysis in regime_summary.values()]
                trending_count = regimes.count('trending')
                ranging_count = regimes.count('ranging')
                volatile_count = regimes.count('volatile')

                msg += f"\n🔍 *Market Summary:*\n"
                msg += f"• Trending: {trending_count} symbols\n"
                msg += f"• Ranging: {ranging_count} symbols\n"
                msg += f"• Volatile: {volatile_count} symbols\n"

            else:
                msg += "❌ *No regime data available*\n"

            await self.safe_reply(update, msg)

        except Exception as e:
            logger.error(f"Error in regime_analysis: {e}")
            await update.message.reply_text("Error getting regime analysis")

    async def strategy_comparison(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Compare strategy performance and show regime accuracy"""
        try:
            msg = "⚖️ *Strategy Comparison Analysis*\n"
            msg += "━" * 35 + "\n\n"

            # Get trade tracker for historical performance
            trade_tracker = self.shared.get("trade_tracker")
            if not trade_tracker:
                msg += "❌ *No trade history available*\n"
                await self.safe_reply(update, msg)
                return

            try:
                # Get recent trades by strategy
                all_trades = trade_tracker.get_all_trades()

                # Filter recent trades (last 50)
                recent_trades = all_trades[-50:] if len(all_trades) > 50 else all_trades

                # Group by strategy
                pullback_trades = [t for t in recent_trades if t.strategy_name == "pullback"]
                mr_trades = [t for t in recent_trades if t.strategy_name in ["mean_reversion", "enhanced_mr"]]

                msg += f"📈 *Recent Performance (Last 50 trades):*\n"

                if pullback_trades:
                    pullback_wins = sum(1 for t in pullback_trades if t.pnl_usd > 0)
                    pullback_wr = (pullback_wins / len(pullback_trades)) * 100
                    pullback_pnl = sum(t.pnl_usd for t in pullback_trades)

                    msg += f"\n🎯 *Pullback Strategy:*\n"
                    msg += f"• Trades: {len(pullback_trades)}\n"
                    msg += f"• Win rate: {pullback_wr:.1f}%\n"
                    msg += f"• Total P&L: ${pullback_pnl:.2f}\n"
                    if len(pullback_trades) > 0:
                        avg_pnl = pullback_pnl / len(pullback_trades)
                        msg += f"• Avg P&L: ${avg_pnl:.2f}\n"

                if mr_trades:
                    mr_wins = sum(1 for t in mr_trades if t.pnl_usd > 0)
                    mr_wr = (mr_wins / len(mr_trades)) * 100
                    mr_pnl = sum(t.pnl_usd for t in mr_trades)

                    msg += f"\n📉 *Mean Reversion Strategy:*\n"
                    msg += f"• Trades: {len(mr_trades)}\n"
                    msg += f"• Win rate: {mr_wr:.1f}%\n"
                    msg += f"• Total P&L: ${mr_pnl:.2f}\n"
                    if len(mr_trades) > 0:
                        avg_pnl = mr_pnl / len(mr_trades)
                        msg += f"• Avg P&L: ${avg_pnl:.2f}\n"

                # Combined stats
                if pullback_trades or mr_trades:
                    total_trades = len(pullback_trades) + len(mr_trades)
                    total_wins = (len([t for t in pullback_trades if t.pnl_usd > 0]) +
                                 len([t for t in mr_trades if t.pnl_usd > 0]))
                    total_pnl = (sum(t.pnl_usd for t in pullback_trades) +
                                sum(t.pnl_usd for t in mr_trades))

                    msg += f"\n📊 *Combined Performance:*\n"
                    msg += f"• Total trades: {total_trades}\n"
                    if total_trades > 0:
                        combined_wr = (total_wins / total_trades) * 100
                        msg += f"• Combined win rate: {combined_wr:.1f}%\n"
                        msg += f"• Combined P&L: ${total_pnl:.2f}\n"
                        msg += f"• Avg per trade: ${total_pnl/total_trades:.2f}\n"

                    # Strategy distribution
                    pullback_pct = (len(pullback_trades) / total_trades) * 100 if total_trades > 0 else 0
                    mr_pct = (len(mr_trades) / total_trades) * 100 if total_trades > 0 else 0

                    msg += f"\n📋 *Strategy Distribution:*\n"
                    msg += f"• Pullback: {pullback_pct:.1f}% of trades\n"
                    msg += f"• Mean Reversion: {mr_pct:.1f}% of trades\n"

                else:
                    msg += "❌ *No recent strategy trades found*\n"

            except Exception as e:
                logger.error(f"Error analyzing trade history: {e}")
                msg += f"❌ *Error analyzing trades: {e}*\n"

            await self.safe_reply(update, msg)

        except Exception as e:
            logger.error(f"Error in strategy_comparison: {e}")
            await update.message.reply_text("Error comparing strategies")

    async def system_status(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Show enhanced parallel system status and configuration"""
        try:
            msg = "🤖 *Enhanced Parallel System Status*\n"
            msg += "━" * 40 + "\n\n"

            # System Architecture
            msg += "🏗️ *System Architecture:*\n"
            msg += "• 🔄 Parallel Strategy Routing\n"
            msg += "• 🧠 Enhanced ML Scorers (Pullback + MR)\n"
            msg += "• 👻 Independent Phantom Tracking\n"
            msg += "• 🎯 Regime-Based Strategy Selection\n\n"

            # Check system availability
            bot_instance = self.shared.get("bot_instance")

            msg += "⚡ *Component Status:*\n"

            # Enhanced MR System
            try:
                if bot_instance and hasattr(bot_instance, 'enhanced_mr_scorer') and bot_instance.enhanced_mr_scorer:
                    msg += "• ✅ Enhanced Mean Reversion ML\n"
                else:
                    msg += "• ⏳ Enhanced Mean Reversion ML (Loading)\n"
            except:
                msg += "• ❓ Enhanced Mean Reversion ML (Unknown)\n"

            # Pullback System
            ml_scorer = self.shared.get("ml_scorer")
            if ml_scorer:
                msg += "• ✅ Pullback ML System\n"
            else:
                msg += "• ⏳ Pullback ML System (Loading)\n"

            # Market Regime Detection
            try:
                from enhanced_market_regime import get_enhanced_market_regime
                msg += "• ✅ Enhanced Regime Detection\n"
            except:
                msg += "• ❌ Enhanced Regime Detection (Error)\n"

            # Phantom Trackers
            phantom_tracker = self.shared.get("phantom_tracker")
            if phantom_tracker:
                msg += "• ✅ Pullback Phantom Tracker\n"
            else:
                msg += "• ⏳ Pullback Phantom Tracker\n"

            try:
                if bot_instance and hasattr(bot_instance, 'mr_phantom_tracker') and bot_instance.mr_phantom_tracker:
                    msg += "• ✅ MR Phantom Tracker\n"
                else:
                    msg += "• ⏳ MR Phantom Tracker\n"
            except:
                msg += "• ❓ MR Phantom Tracker\n"

            msg += "\n🎯 *Strategy Selection Logic:*\n"
            msg += "• 📊 Trending Markets → Pullback Strategy\n"
            msg += "• 📦 High-Quality Ranges → Enhanced MR\n"
            msg += "• 🌪️ Volatile Markets → Skip Trading\n"
            msg += "• ⚖️ Independent ML Scoring Per Strategy\n\n"

            msg += "📈 *Performance Features:*\n"
            msg += "• 🎯 Consistent 2.5:1 Risk:Reward\n"
            msg += "• 💰 Fee-Adjusted Take Profits\n"
            msg += "• 🛡️ Hybrid Stop Loss Calculation\n"
            msg += "• 🔄 Volatility-Adaptive Buffers\n"
            msg += "• 📊 Real-Time Regime Analysis\n\n"

            msg += "⚙️ *Quick Access Commands:*\n"
            msg += "`/enhanced_mr` - MR ML status\n"
            msg += "`/mr_phantom` - MR phantom trades\n"
            msg += "`/parallel_performance` - Strategy comparison\n"
            msg += "`/regime_analysis` - Current market regimes\n"
            msg += "`/strategy_comparison` - Detailed performance\n\n"

            msg += "_Enhanced system provides complete market coverage with specialized strategies_"

            await self.safe_reply(update, msg)

        except Exception as e:
            logger.error(f"Error in system_status: {e}")
            await update.message.reply_text("Error getting system status")

    async def training_status(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Show background ML training status"""
        try:
            msg = "🎯 *Background ML Training Status*\n"
            msg += "━" * 35 + "\n\n"

            # Get background trainer status
            try:
                from background_initial_trainer import get_background_trainer
                trainer = get_background_trainer()
                status = trainer.get_status()
                
                current_status = status.get('status', 'unknown')
                
                if current_status == 'not_started':
                    msg += "⏳ *Status: Not Started*\n"
                    msg += "Training will begin if no existing ML models are detected.\n\n"
                    
                    # Check if models already exist
                    try:
                        import redis
                        import os
                        redis_client = redis.from_url(os.getenv('REDIS_URL'), decode_responses=True)
                        
                        pullback_model = redis_client.get('ml_scorer:model_data')
                        mr_model = redis_client.get('enhanced_mr:model_data')
                        
                        if pullback_model and mr_model:
                            msg += "✅ *Existing ML Models Found:*\n"
                            msg += "• Pullback ML Model: ✅ Trained\n"
                            msg += "• Enhanced MR Model: ✅ Trained\n\n"
                            msg += "🔄 Live bot handles automatic retraining as trades accumulate.\n"
                        else:
                            msg += "❌ *Missing ML Models:*\n"
                            if not pullback_model:
                                msg += "• Pullback ML Model: ⏳ Missing\n"
                            if not mr_model:
                                msg += "• Enhanced MR Model: ⏳ Missing\n"
                            msg += "\n📝 Training should start automatically on next bot restart.\n"
                    except:
                        msg += "❓ Unable to check existing models.\n"
                
                elif current_status == 'running':
                    msg += "🚀 *Status: Training In Progress*\n\n"
                    
                    stage = status.get('stage', 'Unknown')
                    symbol = status.get('symbol', '')
                    progress = status.get('progress', 0)
                    total = status.get('total', 0)
                    
                    msg += f"📊 *Current Stage:* {stage}\n"
                    if symbol:
                        msg += f"🔍 *Current Symbol:* {symbol}\n"
                    if total > 0:
                        percentage = (progress / total) * 100
                        msg += f"📈 *Progress:* {progress}/{total} ({percentage:.1f}%)\n"
                    
                    msg += f"\n⏰ *Last Updated:* {status.get('timestamp', 'Unknown')}\n\n"
                    msg += "💡 Training runs in background - live trading continues normally.\n"
                
                elif current_status == 'completed':
                    msg += "🎉 *Status: Training Complete!*\n\n"
                    
                    pullback_signals = status.get('pullback_signals', 0)
                    mr_signals = status.get('mr_signals', 0)
                    total_symbols = status.get('total_symbols', 0)
                    
                    msg += f"✅ *Results:*\n"
                    msg += f"• Pullback Signals: {pullback_signals:,}\n"
                    msg += f"• MR Signals: {mr_signals:,}\n"
                    msg += f"• Total Symbols: {total_symbols}\n\n"
                    
                    msg += f"⏰ *Completed:* {status.get('timestamp', 'Unknown')}\n\n"
                    msg += "🔄 *Next Steps:*\n"
                    msg += "• Live bot now handles automatic retraining\n"
                    msg += "• Use `/ml` and `/enhanced_mr` to check model status\n"
                    msg += "• Models retrain automatically as trades accumulate\n"
                
                elif current_status == 'error':
                    msg += "❌ *Status: Training Error*\n\n"
                    
                    error = status.get('error', 'Unknown error')
                    msg += f"🚨 *Error:* {error}\n\n"
                    msg += f"⏰ *Error Time:* {status.get('timestamp', 'Unknown')}\n\n"
                    msg += "🔄 *Recovery:*\n"
                    msg += "• Training will retry on next bot restart\n"
                    msg += "• Check logs for detailed error information\n"
                    msg += "• Ensure sufficient disk space and memory\n"
                
                else:
                    msg += f"❓ *Status: {current_status}*\n"
                    msg += "Unknown training status.\n"
                    
            except ImportError:
                msg += "❌ *Background Trainer Not Available*\n"
                msg += "Background training module not found.\n\n"
                msg += "💡 Use the existing `/ml` commands to check model status.\n"
            except Exception as e:
                msg += f"❌ *Error Getting Status*\n"
                msg += f"Error: {str(e)[:100]}...\n\n"
                msg += "Try again in a few moments.\n"

            msg += "\n" + "━" * 35 + "\n"
            msg += "📋 *Available Commands:*\n"
            msg += "`/ml` - Pullback ML status\n"
            msg += "`/enhanced_mr` - Enhanced MR status\n"
            msg += "`/phantom` - Phantom tracking stats\n"
            msg += "`/mr_phantom` - MR phantom stats\n"

            await self.safe_reply(update, msg)

        except Exception as e:
            logger.error(f"Error in training_status: {e}")
            await update.message.reply_text("Error getting training status")
