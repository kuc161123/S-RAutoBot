"""
Telegram Message Formatter
Provides consistent, beautiful formatting for all Telegram alerts
"""
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

class TelegramFormatter:
    """Format messages for Telegram notifications"""
    
    @staticmethod
    def format_position_opened(signal: Dict[str, Any], account_balance: float) -> str:
        """Format position opening alert with enhanced information"""
        
        # Determine direction emoji
        direction_emoji = "📈" if signal.get('action') == "BUY" else "📉"
        direction_text = "LONG" if signal.get('action') == "BUY" else "SHORT"
        
        # Calculate percentages
        entry = signal.get('entry_price', 0)
        sl = signal.get('stop_loss', 0)
        tp1 = signal.get('take_profit_1', 0)
        tp2 = signal.get('take_profit_2', 0)
        
        sl_percent = ((sl - entry) / entry * 100) if entry > 0 else 0
        tp1_percent = ((tp1 - entry) / entry * 100) if entry > 0 else 0
        tp2_percent = ((tp2 - entry) / entry * 100) if entry > 0 else 0
        
        # Get ML confidence level
        ml_confidence = signal.get('ml_confidence', 0)
        if ml_confidence >= 0.85:
            confidence_emoji = "🔥"
            confidence_text = "Very High"
        elif ml_confidence >= 0.75:
            confidence_emoji = "💪"
            confidence_text = "High"
        elif ml_confidence >= 0.65:
            confidence_emoji = "👍"
            confidence_text = "Medium"
        else:
            confidence_emoji = "⚠️"
            confidence_text = "Low"
        
        # Get risk amount and percentage
        risk_amount = signal.get('risk_amount', 0)
        risk_percent = (risk_amount / account_balance * 100) if account_balance > 0 else 0
        
        # Market regime
        market_regime = signal.get('market_regime', 'Unknown')
        zone_score = signal.get('zone_score', 0)
        
        # Risk:Reward ratio
        rr_ratio = abs(tp1_percent / sl_percent) if sl_percent != 0 else 0
        
        # Create TradingView link
        symbol = signal.get('symbol', '')
        tv_link = f"https://www.tradingview.com/chart/?symbol=BYBIT:{symbol}.P"
        
        message = f"""
{direction_emoji} **{direction_text}: {symbol}**
━━━━━━━━━━━━━━━━━━━━━

**Entry Points:**
💰 Entry: ${entry:,.4f}
🎯 TP1: ${tp1:,.4f} ({tp1_percent:+.1f}%)
🎯 TP2: ${tp2:,.4f} ({tp2_percent:+.1f}%)
🛡️ SL: ${sl:,.4f} ({sl_percent:+.1f}%)

**Risk Management:**
📊 Risk: {risk_percent:.1f}% (${risk_amount:.2f})
📐 R:R Ratio: 1:{rr_ratio:.1f}
💎 Position Size: {signal.get('position_size', 0):.4f}

**ML Analysis:**
{confidence_emoji} Confidence: {ml_confidence:.1%} ({confidence_text})
📈 Market: {market_regime}
💪 Zone Strength: {zone_score:.0f}/100
🎰 Win Probability: {signal.get('ml_success_probability', 0):.1%}

📱 [View Chart]({tv_link})
⏰ {datetime.now().strftime('%H:%M:%S UTC')}
"""
        return message.strip()
    
    @staticmethod
    def format_position_closed(
        symbol: str,
        side: str,
        pnl: float,
        pnl_percent: float,
        duration_seconds: int,
        exit_reason: str,
        daily_pnl: float,
        win_rate: float,
        total_trades: int
    ) -> str:
        """Format position closing alert"""
        
        # PnL emoji
        if pnl > 0:
            pnl_emoji = "✅"
            result_text = "PROFIT"
        else:
            pnl_emoji = "❌"
            result_text = "LOSS"
        
        # Format duration
        hours = duration_seconds // 3600
        minutes = (duration_seconds % 3600) // 60
        duration_text = f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m"
        
        # Exit reason formatting
        exit_emoji_map = {
            "TP1": "🎯",
            "TP2": "🎯",
            "SL": "🛡️",
            "MANUAL": "👤",
            "EMERGENCY": "🚨",
            "TIME": "⏰"
        }
        exit_emoji = exit_emoji_map.get(exit_reason.split("_")[0], "📤")
        
        # Daily PnL emoji
        daily_emoji = "📈" if daily_pnl > 0 else "📉"
        
        message = f"""
{pnl_emoji} **{result_text}: {symbol}**
━━━━━━━━━━━━━━━━━━━━━

**Trade Result:**
💵 P&L: ${pnl:+.2f} ({pnl_percent:+.1f}%)
⏱️ Duration: {duration_text}
{exit_emoji} Exit: {exit_reason.replace('_', ' ').title()}

**Session Statistics:**
{daily_emoji} Daily P&L: ${daily_pnl:+.2f}
📊 Win Rate: {win_rate:.1f}% ({total_trades} trades)
🎯 Side: {side}

⏰ {datetime.now().strftime('%H:%M:%S UTC')}
"""
        return message.strip()
    
    @staticmethod
    def format_risk_alert(alert_type: str, details: Dict[str, Any]) -> str:
        """Format risk management alerts"""
        
        alert_messages = {
            "DAILY_LOSS_APPROACHING": f"""
⚠️ **RISK ALERT: Approaching Daily Loss Limit**
━━━━━━━━━━━━━━━━━━━━━

Current Loss: ${details.get('current_loss', 0):.2f}
Daily Limit: ${details.get('limit', 0):.2f}
Remaining: ${details.get('remaining', 0):.2f}

🛡️ Trading will pause at limit
""",
            "HIGH_PORTFOLIO_HEAT": f"""
🔥 **RISK ALERT: High Portfolio Heat**
━━━━━━━━━━━━━━━━━━━━━

Portfolio Heat: {details.get('heat', 0):.1%}
Open Positions: {details.get('positions', 0)}
Total Risk: ${details.get('total_risk', 0):.2f}

⚠️ New positions restricted
""",
            "EMERGENCY_STOP": f"""
🚨 **EMERGENCY STOP ACTIVATED**
━━━━━━━━━━━━━━━━━━━━━

Reason: {details.get('reason', 'Unknown')}
Positions Closed: {details.get('positions_closed', 0)}
Total Loss: ${details.get('loss', 0):.2f}

🛑 Trading halted - manual review required
""",
            "VOLATILE_MARKET": f"""
⚡ **MARKET ALERT: High Volatility**
━━━━━━━━━━━━━━━━━━━━━

Volatility: {details.get('volatility', 0):.1f}x normal
Affected Symbols: {details.get('symbols', 'Multiple')}

📊 Risk reduced to {details.get('risk_percent', 0.75):.2f}%
"""
        }
        
        return alert_messages.get(alert_type, f"⚠️ Alert: {alert_type}")
    
    @staticmethod
    def format_daily_summary(stats: Dict[str, Any]) -> str:
        """Format daily trading summary"""
        
        # Determine overall day emoji
        daily_pnl = stats.get('daily_pnl', 0)
        if daily_pnl > 100:
            day_emoji = "🔥"
            day_text = "EXCELLENT DAY"
        elif daily_pnl > 0:
            day_emoji = "✅"
            day_text = "PROFITABLE DAY"
        elif daily_pnl > -50:
            day_emoji = "⚠️"
            day_text = "SMALL LOSS DAY"
        else:
            day_emoji = "❌"
            day_text = "DIFFICULT DAY"
        
        message = f"""
{day_emoji} **{day_text}**
━━━━━━━━━━━━━━━━━━━━━

**Daily Performance:**
💰 P&L: ${daily_pnl:+.2f} ({stats.get('daily_return', 0):+.1f}%)
📊 Trades: {stats.get('total_trades', 0)}
✅ Winners: {stats.get('winning_trades', 0)}
❌ Losers: {stats.get('losing_trades', 0)}
🎯 Win Rate: {stats.get('win_rate', 0):.1f}%

**Best Trade:** {stats.get('best_trade_symbol', 'N/A')} (+${stats.get('best_trade_pnl', 0):.2f})
**Worst Trade:** {stats.get('worst_trade_symbol', 'N/A')} (${stats.get('worst_trade_pnl', 0):.2f})

**Risk Metrics:**
📈 Max Drawdown: {stats.get('max_drawdown', 0):.1f}%
🔥 Avg Risk Used: {stats.get('avg_risk', 1.0):.2f}%
🤖 ML Accuracy: {stats.get('ml_accuracy', 0):.1f}%

**Account Status:**
💼 Balance: ${stats.get('current_balance', 0):,.2f}
📊 Total Return: {stats.get('total_return', 0):+.1f}%

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}
"""
        return message.strip()
    
    @staticmethod
    def format_ml_milestone(milestone_type: str, details: Dict[str, Any]) -> str:
        """Format ML learning milestones"""
        
        milestones = {
            "ACCURACY_IMPROVED": f"""
🤖 **ML MILESTONE: Accuracy Improved!**
━━━━━━━━━━━━━━━━━━━━━

New Accuracy: {details.get('new_accuracy', 0):.1%}
Previous: {details.get('old_accuracy', 0):.1%}
Improvement: +{details.get('improvement', 0):.1%}

Training Samples: {details.get('samples', 0):,}
""",
            "PATTERN_LEARNED": f"""
🧠 **ML MILESTONE: New Pattern Learned**
━━━━━━━━━━━━━━━━━━━━━

Pattern: {details.get('pattern_name', 'Unknown')}
Success Rate: {details.get('success_rate', 0):.1%}
Occurrences: {details.get('occurrences', 0)}
"""
        }
        
        return milestones.get(milestone_type, f"🤖 ML Update: {milestone_type}")
    
    @staticmethod
    def format_error_alert(error_type: str, details: str) -> str:
        """Format error alerts"""
        
        return f"""
⚠️ **SYSTEM ALERT: {error_type}**
━━━━━━━━━━━━━━━━━━━━━

{details}

⏰ {datetime.now().strftime('%H:%M:%S UTC')}
"""

# Global instance
telegram_formatter = TelegramFormatter()