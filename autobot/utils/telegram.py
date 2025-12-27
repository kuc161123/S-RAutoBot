"""
Simple Telegram Notification Module
====================================
Sends notifications via Telegram Bot API
"""

import aiohttp
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Simple Telegram bot for notifications"""
    
    def __init__(self, bot_token: str, chat_id: str):
        """
        Initialize Telegram notifier
        
        Args:
            bot_token: Telegram bot token
            chat_id: Chat ID to send messages to
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        
    async def send_message(self, message: str, parse_mode: str = "Markdown") -> bool:
        """
        Send a message via Telegram
        
        Args:
            message: Message text
            parse_mode: Parse mode (Markdown or HTML)
            
        Returns:
            True if sent successfully, False otherwise
        """
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/sendMessage"
                
                data = {
                    'chat_id': self.chat_id,
                    'text': message,
                    'parse_mode': parse_mode,
                    'disable_web_page_preview': True
                }
                
                async with session.post(url, json=data) as response:
                    if response.status == 200:
                        logger.debug("Telegram message sent successfully")
                        return True
                    else:
                        logger.error(f"Telegram send failed: {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Telegram error: {e}")
            return False
    
    async def send_startup(self, enabled_symbols: int, risk_pct: float):
        """Send bot startup notification"""
        msg = f"""
🤖 **4H TREND-DIVERGENCE BOT STARTED**

⏰ **Timeframe**: 4H (240 minutes)
📊 **Strategy**: RSI Divergence + BOS + Daily Trend Filter
💰 **Risk**: {risk_pct*100:.1f}% per trade
📈 **Enabled Symbols**: {enabled_symbols}

**Expected Performance:**
• Win Rate: ~25%
• Avg R/Trade: +0.35R
• Trades/Year: ~400

🔍 Monitoring 4H candles...
"""
        await self.send_message(msg)
    
    async def send_entry(self, symbol: str, side: str, entry: float, sl: float, tp: float, rr: float):
        """Send trade entry notification"""
        direction = '🟢 LONG' if side == 'long' else '🔴 SHORT'
        
        msg = f"""
🔔 **NEW TRADE OPENED**

📊 **Symbol**: {symbol}
📈 **Direction**: {direction}
💵 **Entry**: ${entry:,.2f}
⛔ **Stop Loss**: ${sl:,.2f}
🎯 **Take Profit**: ${tp:,.2f} ({rr:.1f}:1 R:R)

**Strategy**: 4H Divergence + Daily Trend
**Risk**: 1% of capital

📊 [View Chart](https://www.tradingview.com/chart/?symbol=BYBIT:{symbol})
"""
        await self.send_message(msg)
    
    async def send_exit(self, symbol: str, side: str, entry: float, exit_price: float, result: str, r_value: float, win_rate: float, total_r: float):
        """Send trade exit notification"""
        emoji = "✅" if result == "WIN" else "❌"
        direction = '🟢 LONG' if side == 'long' else '🔴 SHORT'
        
        msg = f"""
{emoji} **TRADE CLOSED - {result}**

📊 **Symbol**: {symbol}
📈 **Direction**: {direction}
💵 **Entry**: ${entry:,.2f}
💵 **Exit**: ${exit_price:,.2f}
💰 **Profit/Loss**: {r_value:+.2f}R

**Cumulative Performance:**
• Total R: {total_r:+.1f}R
• Win Rate: {win_rate:.1f}%
"""
        await self.send_message(msg)
    
    async def send_dashboard(self, stats: dict, active_positions: int, pending_signals: int):
        """Send dashboard info"""
        msg = f"""
📊 **4H DIVERGENCE BOT DASHBOARD**

**⏰ Timeframe**: 4H (240 minutes)
**📈 Active Symbols**: {stats.get('enabled_symbols', 0)}

**📊 Performance (All-Time)**
• Total R: {stats.get('total_r', 0):+. 1f}R
• Win Rate: {stats.get('win_rate', 0):.1f}%
• Avg R/Trade: {stats.get('avg_r', 0):+.2f}R
• Total Trades: {stats.get('total_trades', 0)}

**🔍 Current Status**
• Pending Signals: {pending_signals}
• Active Positions: {active_positions}

**🎯 Expected Target**: +0.35R/trade, 25% WR
"""
        await self.send_message(msg)
