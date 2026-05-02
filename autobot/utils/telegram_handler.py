"""
Enhanced Telegram Handler with Commands
========================================
Full command support for Multi-Divergence Bot:
- /dashboard - Main dashboard with divergence breakdown
- /help - Command list
- /positions - All active positions
- /stats - Performance statistics
- /stop - Emergency stop

Divergence Types:
- REG_BULL (🟢): Regular Bullish (Reversal)
- REG_BEAR (🔴): Regular Bearish (Reversal)
- HID_BULL (🟠): Hidden Bullish (Continuation)
- HID_BEAR (🟡): Hidden Bearish (Continuation)
"""

import asyncio
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
import logging

logger = logging.getLogger(__name__)


class TelegramHandler:
    """Handles Telegram bot commands and responses"""
    
    def __init__(self, bot_token: str, chat_id: str, bot_instance):
        """
        Initialize Telegram handler
        
        Args:
            bot_token: Telegram bot token
            chat_id: Chat ID for notifications
            bot_instance: Reference to main Bot4H instance
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.bot = bot_instance  # Reference to main bot
        self.app = None
        
    async def start(self):
        """Initialize Telegram application and start polling"""
        if self.app:
            logger.warning("Telegram bot already started")
            return

        from telegram.request import HTTPXRequest
        
        # Increased timeouts to prevent ConnectTimeout errors
        request = HTTPXRequest(
            connection_pool_size=8,
            connect_timeout=30.0,   # Default is 5
            read_timeout=30.0,      # Default is 5
            write_timeout=30.0,     # Default is 5
            pool_timeout=30.0       # Default is 1
        )
        
        self.app = (
            Application.builder()
            .token(self.bot_token)
            .request(request)
            .get_updates_request(request)
            .build()
        )
        
        # Register command handlers
        self.app.add_handler(CommandHandler("help", self.cmd_help))
        self.app.add_handler(CommandHandler("dashboard", self.cmd_dashboard))
        self.app.add_handler(CommandHandler("positions", self.cmd_positions))
        self.app.add_handler(CommandHandler("stats", self.cmd_stats))
        self.app.add_handler(CommandHandler("radar", self.cmd_radar))
        self.app.add_handler(CommandHandler("stop", self.cmd_stop))
        self.app.add_handler(CommandHandler("start", self.cmd_start))
        self.app.add_handler(CommandHandler("risk", self.cmd_risk))
        self.app.add_handler(CommandHandler("performance", self.cmd_performance))
        self.app.add_handler(CommandHandler("resetstats", self.cmd_resetstats))
        self.app.add_handler(CommandHandler("resetlifetime", self.cmd_resetlifetime))
        self.app.add_handler(CommandHandler("debug", self.cmd_debug))
        self.app.add_handler(CommandHandler("setbalance", self.cmd_setbalance))
        self.app.add_handler(CommandHandler("setregime", self.cmd_setregime))

        # Start polling with longer interval to avoid rate limits
        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling(
            poll_interval=2.0,      # Check every 2 seconds (default 0.0)
            timeout=20,             # Long polling timeout
            drop_pending_updates=True  # Don't process old commands on restart
        )
        
        logger.info("Telegram command handler started")
        
        # Rate limiting
        self._last_message_time = 0
    
    async def send_message(self, message: str, retries: int = 3):
        """Send a message with rate limiting and retry logic"""
        import time
        import asyncio

        if not self.app:
            logger.warning("Telegram app not initialized - message not sent")
            return

        # Split long messages for Telegram's 4096 char limit
        chunks = []
        msg = message
        while msg:
            if len(msg) <= 4000:
                chunks.append(msg)
                break
            split_at = msg.rfind('\n\n', 0, 4000)
            if split_at == -1:
                split_at = msg.rfind('\n', 0, 4000)
            if split_at == -1:
                split_at = 4000
            chunks.append(msg[:split_at])
            msg = msg[split_at:].lstrip('\n')

        for chunk in chunks:
            # Rate limiting - wait at least 0.5s between messages
            now = time.time()
            time_since_last = now - getattr(self, '_last_message_time', 0)
            if time_since_last < 0.5:
                await asyncio.sleep(0.5 - time_since_last)

            for attempt in range(retries):
                try:
                    await self.app.bot.send_message(
                        chat_id=self.chat_id,
                        text=chunk,
                        parse_mode='Markdown',
                        disable_web_page_preview=True
                    )
                    self._last_message_time = time.time()
                    break  # Success, move to next chunk
                except Exception as e:
                    logger.error(f"Telegram send failed (attempt {attempt+1}/{retries}): {e}")
                    if attempt < retries - 1:
                        await asyncio.sleep(1)  # Wait before retry
                    else:
                        logger.error(f"Failed to send message after {retries} attempts")

    
    # === COMMAND HANDLERS ===
    
    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Help command with all available commands"""
        # Get current symbol count from config
        try:
            enabled_count = len(self.bot.symbol_config.get_enabled_symbols())
        except:
            enabled_count = 0
        
        msg = f"""🤖 **1H MULTI-DIVERGENCE BOT**
━━━━━━━━━━━━━━━━━━━━

📊 **MONITORING**
├ /dashboard - Live trading dashboard
├ /positions - All active positions
├ /stats - Performance statistics (all-time)
├ /performance - Symbol leaderboard (R values)
└ /radar - Developing setups & RSI extremes

💰 **RISK MANAGEMENT**
├ /risk - View current risk settings
├ /risk 0.2 - Set 0.2% risk per trade
├ /risk $5 - Set fixed $5 per trade
├ /setregime - Override regime (cautious/adverse)
└ /setbalance - Set P&L baseline after deposits

🔄 **RESET OPTIONS**
├ /resetstats - Reset session stats only
└ /resetlifetime - Full reset (ALL tracking + baseline)

⚙️ **BOT CONTROL**
├ /stop - Emergency halt (stops new trades)
├ /start - Resume trading
└ /debug - Technical debugging info

❓ /help - Show this message

━━━━━━━━━━━━━━━━━━━━
**Strategy**: 1H Precision Divergence
**Portfolio**: {enabled_count} Symbols | 0.1% Risk
"""
        await update.message.reply_text(msg, parse_mode='Markdown')
    
    async def build_dashboard_message(self) -> str:
        """Build the full dashboard message string (reusable for auto-send and /dashboard command)"""
        from datetime import datetime, timedelta

        # === SYNC WITH EXCHANGE ===
        await self.bot.sync_with_exchange()

        # === GATHER ALL DATA ===
        uptime_hrs = max(0, (datetime.now() - self.bot.start_time).total_seconds() / 3600)
        pending = sum(len(sigs) for sigs in self.bot.pending_signals.values())

        # Fetch positions ONCE and reuse
        positions = []
        try:
            positions = await self.bot.broker.get_positions() or []
        except Exception as e:
            logger.error(f"[DASHBOARD] Failed to get positions from Bybit: {e}")

        active_positions = [p for p in positions if float(p.get('size', 0)) > 0]
        active = len(active_positions)
        if active == 0:
            active = len(self.bot.active_trades)  # Fallback to internal tracking
        logger.info(f"[DASHBOARD] Bybit reports {active} active positions")

        enabled = len(self.bot.symbol_config.get_enabled_symbols())

        # Scan status
        scan = self.bot.scan_state
        last_scan = scan.get('last_scan_time')
        mins_ago = int((datetime.now() - last_scan).total_seconds() / 60) if last_scan else 0

        # Balance and P&L
        balance = await self.bot.broker.get_balance() or 0

        # Risk amount for R calculation (use tapered base, not config starting value)
        config_risk_pct = self.bot.risk_config.get('risk_per_trade', 0.012)
        taper = self.bot.risk_config.get('taper_schedule')
        tapered_risk_pct = config_risk_pct
        if taper and balance > 0:
            for threshold, risk in taper:
                if balance >= threshold:
                    tapered_risk_pct = risk
        base_risk_amount = balance * tapered_risk_pct if balance > 0 else 10

        risk_usd = self.bot.risk_config.get('risk_amount_usd')
        if risk_usd:
            risk_amount = float(risk_usd)
            base_risk_amount = risk_amount  # Fixed USD overrides base too
            risk_pct = (risk_amount / balance * 100) if balance > 0 else 0
            risk_display = f"${risk_amount:.2f} ({risk_pct:.2f}%)"
        else:
            risk_pct = self.bot.get_adaptive_risk(balance=balance)
            risk_amount = balance * risk_pct if balance > 0 else 10
            taper_pct = tapered_risk_pct * 100
            adaptive_pct = risk_pct * 100
            if abs(taper_pct - adaptive_pct) > 0.001:
                risk_display = f"${risk_amount:.2f} ({adaptive_pct:.2f}% | taper {taper_pct:.2f}%)"
            else:
                risk_display = f"${risk_amount:.2f} ({adaptive_pct:.2f}%)"

        # Realized P&L (today + weekly)
        realized_pnl = 0
        weekly_pnl = 0
        today_trades = 0
        today_wins = 0
        weekly_trades = 0
        weekly_wins = 0
        try:
            closed_records = await self.bot.broker.get_all_closed_pnl(limit=200)
            if closed_records:
                today = datetime.now().date()
                week_ago = today - timedelta(days=6)
                for record in closed_records:
                    try:
                        close_time = int(record.get('updatedTime', record.get('createdTime', 0)))
                        trade_date = datetime.fromtimestamp(close_time / 1000).date()
                        pnl = float(record.get('closedPnl', 0))
                        if trade_date == today:
                            realized_pnl += pnl
                            today_trades += 1
                            if pnl > 0:
                                today_wins += 1
                        if trade_date >= week_ago:
                            weekly_pnl += pnl
                            weekly_trades += 1
                            if pnl > 0:
                                weekly_wins += 1
                    except:
                        continue
        except Exception as e:
            logger.error(f"Error getting realized P&L: {e}")

        realized_r = realized_pnl / base_risk_amount if base_risk_amount > 0 else 0
        weekly_r = weekly_pnl / base_risk_amount if base_risk_amount > 0 else 0
        today_losses = today_trades - today_wins
        weekly_losses = weekly_trades - weekly_wins

        # Unrealized P&L (from cached positions)
        unrealized_pnl = 0
        positions_up = 0
        positions_down = 0
        long_count = 0
        short_count = 0

        # Decorate positions for sorting
        decorated_positions = []
        for pos in active_positions:
            unrealized = float(pos.get('unrealisedPnl', 0))
            unrealized_pnl += unrealized
            symbol = pos.get('symbol', '???')
            side = pos.get('side', '?')

            if side == 'Buy':
                long_count += 1
            else:
                short_count += 1

            if unrealized >= 0:
                positions_up += 1
            else:
                positions_down += 1

            pos_r = unrealized / base_risk_amount if base_risk_amount > 0 else 0
            decorated_positions.append({
                'symbol': symbol,
                'side': side,
                'pnl': unrealized,
                'r_value': pos_r
            })

        # Sort positions by PnL descending
        decorated_positions.sort(reverse=True, key=lambda x: x['pnl'])

        unrealized_r = unrealized_pnl / base_risk_amount if base_risk_amount > 0 else 0

        # Net P&L
        net_pnl = realized_pnl + unrealized_pnl
        net_r = net_pnl / base_risk_amount if base_risk_amount > 0 else 0
        net_emoji = "🟢" if net_pnl >= 0 else "🔴"

        # === ALL-TIME PERFORMANCE FROM LOCAL LIFETIME_STATS ===
        lifetime = self.bot.lifetime_stats
        start_date = lifetime.get('start_date', 'Unknown')
        starting_balance = lifetime.get('starting_balance', 0) or balance

        lifetime_r = lifetime.get('total_r', 0.0)
        weighted_r = lifetime.get('weighted_total_r', 0.0)
        lifetime_pnl = lifetime.get('total_pnl', 0.0)
        lifetime_trades = lifetime.get('total_trades', 0)
        lifetime_wins = lifetime.get('wins', 0)
        lifetime_losses = lifetime_trades - lifetime_wins
        lifetime_wr = (lifetime_wins / lifetime_trades * 100) if lifetime_trades > 0 else 0

        # Best/worst from local storage
        best_trade_r = lifetime.get('best_trade_r', 0.0)
        best_trade_symbol = lifetime.get('best_trade_symbol') or 'N/A'
        worst_trade_r = lifetime.get('worst_trade_r', 0.0)
        worst_trade_symbol = lifetime.get('worst_trade_symbol') or 'N/A'
        best_day_r = lifetime.get('best_day_r', 0.0)
        best_day_date = lifetime.get('best_day_date') or 'N/A'
        worst_day_r = lifetime.get('worst_day_r', 0.0)
        worst_day_date = lifetime.get('worst_day_date') or 'N/A'

        # Drawdown and streaks from local storage
        max_dd = lifetime.get('max_drawdown_r', 0.0)
        current_streak = lifetime.get('current_streak', 0)
        streak_type = "W" if current_streak > 0 else ("L" if current_streak < 0 else "")
        longest_win_streak = lifetime.get('longest_win_streak', 0)
        longest_loss_streak = lifetime.get('longest_loss_streak', 0)

        # Calculate profit factor from stored gross data
        gross_wins = lifetime.get('gross_profit_r', 0.0)
        gross_losses = abs(lifetime.get('gross_loss_r', 0.0))
        if gross_losses > 0:
            profit_factor = gross_wins / gross_losses
            profit_factor_display = f"{profit_factor:.2f}"
        elif gross_wins > 0:
            profit_factor_display = "∞"
        else:
            profit_factor_display = "N/A"

        # Calculate days since start
        try:
            if start_date and start_date != 'Unknown':
                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                days_running = (datetime.now() - start_dt).days
            else:
                days_running = 0
                start_date = datetime.now().strftime('%Y-%m-%d')
        except:
            days_running = 0

        # Available balance (from cached positions)
        total_margin = sum(float(p.get('positionIM', 0)) for p in active_positions)
        available_balance = balance - total_margin
        available_pct = (available_balance / balance * 100) if balance > 0 else 0

        # API Key expiry
        api_info = await self.bot.broker.get_api_key_info()
        if api_info:
            days_left = api_info.get('days_left')
            if days_left is not None:
                if days_left <= 7:
                    key_status = f"⚠️ {days_left}d left!"
                elif days_left <= 30:
                    key_status = f"🟡 {days_left}d left"
                else:
                    key_status = f"✅ {days_left}d left"
            else:
                key_status = "✅ Connected"
        else:
            key_status = "⚠️ API Check Failed"

        next_scan_mins = max(0, 60 - mins_ago)

        # Get today's signal counts
        divs_today = self.bot.bos_tracking.get('divergences_detected_today', 0)
        bos_today = self.bot.bos_tracking.get('bos_confirmed_today', 0)

        # P&L Return (Fixed calculation)
        if starting_balance > 0:
            pnl_return_pct = ((balance - starting_balance) / starting_balance) * 100
        else:
            pnl_return_pct = 0
        pnl_emoji = "🟢" if pnl_return_pct >= 0 else "🔴"

        # Average R
        avg_r = lifetime_r / lifetime_trades if lifetime_trades > 0 else 0

        # Expectancy (avg R per trade)
        expectancy_display = f"{avg_r:+.2f}R" if lifetime_trades > 0 else "N/A"

        # === BUILD POSITIONS SECTION ===
        if decorated_positions:
            lines = []
            lines.append(f"├ Direction: {short_count} Shorts 🔴 | {long_count} Longs 🟢")
            lines.append(f"├ Status: {positions_up} Profit | {positions_down} Loss")

            # Top 3 (if we have more than 6, otherwise just show all)
            if len(decorated_positions) > 6:
                lines.append("├ ⭐ **Top 3:**")
                for p in decorated_positions[:3]:
                    icon = "📈" if p['side'] == 'Buy' else "📉"
                    lines.append(f"│ ├ {p['symbol']} {icon}: ${p['pnl']:+,.2f} ({p['r_value']:+.1f}R)")

                lines.append("└ ⚠️ **Bottom 3:**")
                for i, p in enumerate(decorated_positions[-3:]):
                    icon = "📈" if p['side'] == 'Buy' else "📉"
                    prefix = "  └" if i == 2 else "  ├"
                    lines.append(f"{prefix} {p['symbol']} {icon}: ${p['pnl']:+,.2f} ({p['r_value']:+.1f}R)")
            else:
                for i, p in enumerate(decorated_positions):
                    icon = "📈" if p['side'] == 'Buy' else "📉"
                    prefix = "└" if i == len(decorated_positions) - 1 else "├"
                    lines.append(f"{prefix} {p['symbol']} {icon}: ${p['pnl']:+,.2f} ({p['r_value']:+.1f}R)")

            pos_detail = "\n".join(lines)
            pos_section = f"""📡 **POSITIONS ({active} Active)**
{pos_detail}"""
        else:
            pos_section = f"""📡 **POSITIONS (0 Active)**
├ No open trades
└ Awaiting BOS: {pending} symbols"""

        # === REGIME STATUS (V2: multi-signal safety system) ===
        regime_display = "Building data..."
        regime_detail = ""
        try:
            regime_label, regime_mult, regime_diag = self.bot.get_regime_status()
            n_trades = len(self.bot.recent_trades)
            regime_icons = {
                'favorable': '🟢', 'cautious': '🟡', 'adverse': '🟠',
                'critical': '🔴', 'halted': '🛑', 'unknown': '⏳',
            }
            icon = regime_icons.get(regime_label, '❓')
            if n_trades < 10:
                regime_display = f"Critical 🔴 (10% risk) | SAFE START {n_trades}/10t"
            else:
                manual_tag = " [MANUAL]" if self.bot.regime_override is not None else ""
                remaining = 10 - self.bot.regime_override_trades if self.bot.regime_override else 0
                override_info = f" ({remaining}t left)" if manual_tag else ""
                regime_display = f"{regime_label.title()} {icon} ({regime_mult:.0%} risk){manual_tag}{override_info}"
                # Build detail lines
                details = []
                q = regime_diag['quality']
                details.append(f"├ 20t: {q['wr']:.0%} WR, {q['avg_r']:+.2f}R → {q['mult']:.2f}x")
                dd = regime_diag['drawdown']['dd_from_peak']
                if dd > 0:
                    details.append(f"├ DD from peak: {dd:.1f}R")
                dr = regime_diag['daily']['daily_r']
                if dr != 0:
                    details.append(f"├ Daily R: {dr:+.1f}R")
                ls = regime_diag['streak']['loss_streak']
                if ls >= 3:
                    details.append(f"├ Loss streak: {ls}")
                regime_detail = "\n" + "\n".join(details) if details else ""

                # Regime duration info
                regime_since = self.bot.lifetime_stats.get('current_regime_since')
                if regime_since and regime_label != 'unknown':
                    try:
                        since_dt = datetime.fromisoformat(regime_since)
                        hours = (datetime.now() - since_dt).total_seconds() / 3600
                        dur = f"{hours/24:.1f}d" if hours >= 24 else f"{hours:.0f}h"
                        dur_line = f"\n├ Duration: {dur}"
                        # Historical comparison for chop regimes
                        history = self.bot.lifetime_stats.get('regime_history', [])
                        chop = [h for h in history if h['label'] in ('critical', 'adverse')]
                        if chop and regime_label in ('critical', 'adverse'):
                            avg_h = sum(h['duration_hours'] for h in chop) / len(chop)
                            max_h = max(h['duration_hours'] for h in chop)
                            dur_line += f" (avg {avg_h/24:.1f}d / max {max_h/24:.1f}d)"
                        regime_detail += dur_line
                    except:
                        pass
        except Exception as e:
            logger.error(f"[DASHBOARD] Regime calc failed: {e}")

        # BTC market data (shared by ADX and CHOP display)
        btc_adx_display = ""
        btc_chop_display = ""
        try:
            params = {'category':'linear','symbol':'BTCUSDT','interval':'60','limit':'60'}
            kline_resp = await self.bot.broker._request("GET", "/v5/market/kline", params)
            data = kline_resp.get('result', {}).get('list', []) if kline_resp else []
            if len(data) >= 30:
                import pandas as pd
                import numpy as np
                from ta.trend import ADXIndicator
                df = pd.DataFrame(data, columns=['start','open','high','low','close','volume','turnover'])
                for c in ['high','low','close']: df[c] = df[c].astype(float)
                df = df.iloc[::-1].reset_index(drop=True)

                # ADX
                adx_indicator = ADXIndicator(df['high'], df['low'], df['close'], window=14)
                current_adx = adx_indicator.adx().iloc[-1]
                if not pd.isna(current_adx):
                    btc_adx_display = f"\n├ BTC ADX: {current_adx:.1f}"

                # CHOP Index
                tr = pd.concat([
                    df['high'] - df['low'],
                    (df['high'] - df['close'].shift(1)).abs(),
                    (df['low'] - df['close'].shift(1)).abs()
                ], axis=1).max(axis=1)
                period = 14
                atr_sum = tr.rolling(period).sum()
                hh = df['high'].rolling(period).max()
                ll = df['low'].rolling(period).min()
                hl_range = (hh - ll).replace(0, pd.NA)
                chop_series = 100 * np.log10(atr_sum / hl_range) / np.log10(period)
                btc_chop = chop_series.iloc[-1]
                if not pd.isna(btc_chop):
                    chop_label = "CHOPPY" if btc_chop > 55 else ("TREND" if btc_chop < 45 else "MIXED")
                    btc_chop_display = f"\n├ BTC CHOP: {btc_chop:.1f} ({chop_label})"
        except Exception as e:
            logger.error(f"[DASHBOARD] BTC indicators calc failed: {e}")

        # === BUILD EDGE CHECK SECTION ===
        regime_stats = lifetime.get('regime_stats', {})
        edge_lines = []
        regime_icons = {'favorable': '\U0001f7e2', 'cautious': '\U0001f7e1', 'adverse': '\U0001f7e0', 'critical': '\U0001f534'}
        for rk in ['favorable', 'cautious', 'adverse', 'critical']:
            rs = regime_stats.get(rk, {})
            t = rs.get('trades', 0)
            if t == 0:
                continue
            w = rs.get('wins', 0)
            wr = w / t * 100
            gp = rs.get('gross_profit_r', 0.0)
            gl = abs(rs.get('gross_loss_r', 0.0))
            pf = f"{gp/gl:.1f}" if gl > 0 else ("inf" if gp > 0 else "N/A")
            icon = regime_icons.get(rk, '\u26aa')
            edge_lines.append(f"\u251c {icon} {rk.title()}: {wr:.0f}% WR | PF {pf} | {t}t")
        if edge_lines:
            edge_lines[-1] = "\u2514" + edge_lines[-1][1:]
            edge_section = "\U0001f3af **EDGE CHECK**\n" + "\n".join(edge_lines)
        else:
            edge_section = ""

        # === BUILD ENHANCED DASHBOARD ===
        msg = f"""💰 **TRADING DASHBOARD**
━━━━━━━━━━━━━━━━━━━━

📈 **TODAY'S P&L**
├ 💵 Realized: ${realized_pnl:+,.2f} ({realized_r:+.1f}R) | {today_wins}W/{today_losses}L
├ 📊 Unrealized: ${unrealized_pnl:+,.2f} ({unrealized_r:+.1f}R) | {active} open
└ {net_emoji} **Net: ${net_pnl:+,.2f} ({net_r:+.1f}R)**

📅 **7-DAY P&L**
├ Realized: ${weekly_pnl:+,.2f} ({weekly_r:+.1f}R) | {weekly_trades} trades
└ Record: {weekly_wins}W/{weekly_losses}L

🏆 **ALL-TIME** ({days_running} days)
├ {lifetime_r:+.1f}R raw | {weighted_r:+.1f}R wtd | {lifetime_trades} trades
├ P&L: ${lifetime_pnl:+,.2f} | Balance: ${balance:,.2f}
├ WR: {lifetime_wr:.1f}% ({lifetime_wins}W/{lifetime_losses}L) | PF: {profit_factor_display}
├ Avg: {expectancy_display} | Max DD: {abs(max_dd):.1f}R
└ Streak: {abs(current_streak)}{streak_type} | Best: {longest_win_streak}W / {longest_loss_streak}L
{"" if not edge_section else chr(10) + edge_section + chr(10)}
💼 **ACCOUNT**
├ Balance: ${balance:,.2f}
├ Available: ${available_balance:,.2f} ({available_pct:.0f}%)
├ Risk/Trade: {risk_display}
└ Return: {pnl_emoji}{abs(pnl_return_pct):.1f}% (Base: ${starting_balance:,.0f})

{pos_section}

⏰ **SYSTEM HEALTH**
├ Uptime: {uptime_hrs:.1f}h ✅
├ Regime: {regime_display}{regime_detail}{btc_adx_display}{btc_chop_display}
├ Next Scan: ~{next_scan_mins}m
├ Symbols: {enabled} | Signals: {divs_today}D/{bos_today}BOS
├ ⏳ Pending BOS: {pending} signals
└ 🔑 API: {key_status}

━━━━━━━━━━━━━━━━━━━━
/positions | /radar | /stats | /performance
"""
        return msg

    async def cmd_dashboard(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Clean, focused trading dashboard"""
        try:
            msg = await self.build_dashboard_message()
            if len(msg) <= 4000:
                await update.message.reply_text(msg, parse_mode='Markdown')
            else:
                # Split at last newline before 4000 chars for Telegram's 4096 limit
                chunks = []
                while msg:
                    if len(msg) <= 4000:
                        chunks.append(msg)
                        break
                    split_at = msg.rfind('\n\n', 0, 4000)
                    if split_at == -1:
                        split_at = msg.rfind('\n', 0, 4000)
                    if split_at == -1:
                        split_at = 4000
                    chunks.append(msg[:split_at])
                    msg = msg[split_at:].lstrip('\n')
                for chunk in chunks:
                    await update.message.reply_text(chunk, parse_mode='Markdown')
        except Exception as e:
            await update.message.reply_text(f"❌ Dashboard error: {e}")
            logger.error(f"Dashboard error: {e}")
            import traceback
            logger.error(traceback.format_exc())

    
    async def cmd_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show all active positions"""
        try:
            if not self.bot.active_trades:
                await update.message.reply_text("📊 No active positions.")
                return
            
            msg = f"📊 **ACTIVE POSITIONS** ({len(self.bot.active_trades)} open)\n\n"

            for trade_key, trade in self.bot.active_trades.items():
                if trade is None:
                    symbol = trade_key.rsplit('_', 1)[0]
                    msg += f"┌─ ⚪ `{symbol}` (synced, no details)\n\n"
                    continue
                side_icon = "🟢" if trade.side == 'long' else "🔴"

                msg += f"""
┌─ {side_icon} {trade.side.upper()} `{trade.symbol}`
├ Entry: ${trade.entry_price:,.2f}
├ Stop Loss: ${trade.stop_loss:,.2f}
├ Take Profit: ${trade.take_profit:,.2f}
├ R:R: {trade.rr_ratio}:1
└ Size: {trade.position_size:.4f}

"""
            
            msg += "━━━━━━━━━━━━━━━━━━━━\n"
            msg += "💡 /dashboard /stats"
            
            await update.message.reply_text(msg, parse_mode='Markdown')
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")
            logger.error(f"Positions error: {e}")
    
    async def cmd_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Performance statistics with all-time tracking"""
        try:
            stats = self.bot.stats  # Session stats
            lifetime = self.bot.lifetime_stats  # All-time stats
            
            # Session stats
            session_trades = stats['total_trades']
            session_wr = stats['win_rate']
            session_avg_r = stats['avg_r']
            session_total_r = stats['total_r']
            
            # All-time stats (source of truth)
            lt_trades = lifetime.get('total_trades', 0)
            lt_wins = lifetime.get('wins', 0)
            lt_losses = lt_trades - lt_wins
            lt_wr = (lt_wins / lt_trades * 100) if lt_trades > 0 else 0
            lt_total_r = lifetime.get('total_r', 0.0)
            lt_avg_r = lt_total_r / lt_trades if lt_trades > 0 else 0
            lt_pnl = lifetime.get('total_pnl', 0.0)
            
            # Best/worst metrics
            best_trade_r = lifetime.get('best_trade_r', 0.0)
            best_trade_sym = lifetime.get('best_trade_symbol') or 'N/A'
            worst_trade_r = lifetime.get('worst_trade_r', 0.0)
            worst_trade_sym = lifetime.get('worst_trade_symbol') or 'N/A'
            best_day_r = lifetime.get('best_day_r', 0.0)
            best_day_date = lifetime.get('best_day_date') or 'N/A'
            worst_day_r = lifetime.get('worst_day_r', 0.0)
            worst_day_date = lifetime.get('worst_day_date') or 'N/A'
            
            # Streaks and drawdown
            max_dd = lifetime.get('max_drawdown_r', 0.0)
            current_streak = lifetime.get('current_streak', 0)
            streak_type = "W" if current_streak > 0 else ("L" if current_streak < 0 else "")
            longest_win = lifetime.get('longest_win_streak', 0)
            longest_loss = lifetime.get('longest_loss_streak', 0)
            
            # Get enabled symbols count from config
            enabled_symbols = len(self.bot.symbol_config.get_enabled_symbols())
            
            msg = f"""
📊 **PERFORMANCE STATISTICS**
━━━━━━━━━━━━━━━━━━━━

🏆 **ALL-TIME (Source of Truth)**
├ Total Trades: {lt_trades}
├ Wins: {lt_wins} (✅) | Losses: {lt_losses} (❌)
├ Win Rate: {lt_wr:.1f}%
├ Avg R/Trade: {lt_avg_r:+.2f}R
├ Total R: {lt_total_r:+.1f}R
└ Total P&L: ${lt_pnl:+,.2f}

📈 **EXTREMES**
├ 🥇 Best Trade: {best_trade_r:+.1f}R ({best_trade_sym})
├ 🥉 Worst Trade: {worst_trade_r:+.1f}R ({worst_trade_sym})
├ ⬆️ Best Day: {best_day_r:+.1f}R ({best_day_date})
└ ⬇️ Worst Day: {worst_day_r:+.1f}R ({worst_day_date})

📉 **RISK METRICS**
├ Max Drawdown: {max_dd:.1f}R
├ Current Streak: {abs(current_streak)}{streak_type}
└ Longest Streaks: {longest_win}W / {longest_loss}L

📝 **SESSION STATS (Since Restart)**
├ Trades: {session_trades} | WR: {session_wr:.1f}%
└ Total R: {session_total_r:+.1f}R

🎯 **VS EXPECTED ({enabled_symbols} symbols)**
├ Expected WR: ~23%
├ Actual WR: {lt_wr:.1f}%
├ Expected R/Trade: +0.50R
├ Actual R/Trade: {lt_avg_r:+.2f}R
└ Delta: {lt_avg_r - 0.50:+.2f}R

━━━━━━━━━━━━━━━━━━━━
💡 /dashboard /positions /performance
"""
            await update.message.reply_text(msg, parse_mode='Markdown')
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    async def cmd_performance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Symbol performance leaderboard"""
        try:
            symbol_stats = self.bot.symbol_stats
            
            if not symbol_stats:
                await update.message.reply_text("📊 No trades recorded yet.")
                return
            
            sorted_symbols = sorted(
                [(sym, data) for sym, data in symbol_stats.items() if data.get('trades', 0) > 0],
                key=lambda x: x[1].get('total_r', 0), reverse=True
            )
            
            if not sorted_symbols:
                await update.message.reply_text("📊 No completed trades yet.")
                return
            
            # Top 5
            top5_str = ""
            for i, (sym, data) in enumerate(sorted_symbols[:5]):
                emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "📈"
                wr = (data.get('wins', 0) / max(data.get('trades', 1), 1)) * 100
                top5_str += f"{emoji} {sym}: {data.get('total_r', 0):+.1f}R ({data.get('trades', 0)}T, {wr:.0f}%)\n"
            
            # Bottom 5
            bottom5_str = ""
            for sym, data in sorted_symbols[-5:][::-1]:
                wr = (data.get('wins', 0) / max(data.get('trades', 1), 1)) * 100
                bottom5_str += f"📉 {sym}: {data.get('total_r', 0):+.1f}R ({data.get('trades', 0)}T, {wr:.0f}%)\n"
            
            total_r = sum(d.get('total_r', 0) for d in symbol_stats.values())
            active = len([s for s, d in symbol_stats.items() if d.get('trades', 0) > 0])
            profitable = len([s for s, d in symbol_stats.items() if d.get('total_r', 0) > 0])
            
            msg = f"""
📊 **SYMBOL LEADERBOARD**
━━━━━━━━━━━━━━━━━━━━

🏆 **TOP 5**
{top5_str}
⚠️ **BOTTOM 5**
{bottom5_str}
📈 Active: {active} | Profitable: {profitable} | Total R: {total_r:+.1f}R

💡 /dashboard /stats
"""
            await update.message.reply_text(msg, parse_mode='Markdown')
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")
    async def cmd_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Emergency stop"""
        self.bot.trading_enabled = False
        msg = """
⛔ **EMERGENCY STOP EXECUTED**

Trading has been halted.
Pending signals will be ignored.
Active positions will remain open but no new trades will be taken.

To resume: `/start`
"""
        await update.message.reply_text(msg, parse_mode='Markdown')
        logger.warning(f"⛔ EMERGENCY STOP triggered by user {update.effective_user.name}")
    
    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Start/resume trading"""
        self.bot.trading_enabled = True
        msg = "✅ **TRADING RESUMED**\n\nThe bot will process the next available signals."
        await update.message.reply_text(msg, parse_mode='Markdown')
        logger.info(f"✅ Trading resumed by user {update.effective_user.name}")
    
    async def cmd_risk(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """View or update risk per trade (supports % and USD)"""
        try:
            # Get current balance for conversions
            try:
                balance = await self.bot.broker.get_balance()
            except:
                balance = 1000  # Fallback
            
            current_risk_pct = self.bot.risk_config.get('risk_per_trade', 0.005)
            current_risk_usd = balance * current_risk_pct
            
            if not context.args:
                # View current risk
                msg = f"""💰 **CURRENT RISK SETTINGS**
━━━━━━━━━━━━━━━━━━━━

📊 **Per Trade Risk**
├ Percentage: {current_risk_pct*100:.2f}%
├ USD Amount: ${current_risk_usd:.2f}
└ Balance: ${balance:,.2f}

⚙️ **To Update**:
├ `/risk 0.5` or `/risk 0.5%` → Set to 0.5%
├ `/risk $50` or `/risk 50usd` → Set to $50 per trade
└ `/risk 1%` → Set to 1%

💡 Current: ${current_risk_usd:.2f} per trade ({current_risk_pct*100:.2f}%)
"""
                await update.message.reply_text(msg, parse_mode='Markdown')
                return
            
            # Parse input - support multiple formats
            input_val = context.args[0].lower().strip()
            
            # Check for USD formats: $50, 50usd, 50$
            is_usd = False
            if input_val.startswith('$'):
                is_usd = True
                input_val = input_val[1:]  # Remove $
            elif input_val.endswith('usd'):
                is_usd = True
                input_val = input_val[:-3]  # Remove usd
            elif input_val.endswith('$'):
                is_usd = True
                input_val = input_val[:-1]  # Remove trailing $
            
            # Check for percentage format: 0.5%, 1%
            if input_val.endswith('%'):
                input_val = input_val[:-1]  # Remove %
            
            try:
                amount = float(input_val)
            except ValueError:
                await update.message.reply_text("❌ Invalid format. Examples:\n`/risk 0.5` (0.5%)\n`/risk $50` ($50 per trade)\n`/risk 1%` (1%)")
                return
            
            if is_usd:
                # Convert USD to percentage
                if balance <= 0:
                    await update.message.reply_text("❌ Cannot calculate percentage - balance unavailable")
                    return
                
                new_risk_pct = amount / balance
                new_risk_usd = amount
                
                # Validate
                if new_risk_pct > 0.05:  # Max 5%
                    await update.message.reply_text(f"⚠️ ${amount:.2f} is {new_risk_pct*100:.1f}% of your balance - too high! Max is 5%.")
                    return
                if new_risk_pct < 0.001:  # Min 0.1%
                    await update.message.reply_text(f"⚠️ ${amount:.2f} is only {new_risk_pct*100:.3f}% - too low! Min is 0.1%.")
                    return
            else:
                # Treat as percentage
                # Handle both 0.5 and 50 input styles
                if amount >= 1:
                    new_risk_pct = amount / 100  # User entered 1 for 1%
                else:
                    new_risk_pct = amount  # User entered 0.01 for 1%
                
                new_risk_usd = balance * new_risk_pct
                
                # Validate
                if new_risk_pct > 0.05:
                    await update.message.reply_text(f"⚠️ {new_risk_pct*100:.1f}% is too high! Max is 5%.")
                    return
                if new_risk_pct < 0.001:
                    await update.message.reply_text(f"⚠️ {new_risk_pct*100:.3f}% is too low! Min is 0.1%.")
                    return
            
            # Apply the new risk
            if is_usd:
                # Use fixed USD amount
                success, result_msg = self.bot.set_risk_usd(new_risk_usd)
                if success:
                    msg = f"""✅ **RISK UPDATED (Fixed USD)**
━━━━━━━━━━━━━━━━━━━━

📊 **New Risk Per Trade**
├ Fixed Amount: ${new_risk_usd:.2f}
├ Equivalent: ~{new_risk_pct*100:.2f}% of balance
└ Balance: ${balance:,.2f}

💡 Each trade will now risk exactly ${new_risk_usd:.2f}
"""
                    await update.message.reply_text(msg, parse_mode='Markdown')
                else:
                    await update.message.reply_text(f"❌ {result_msg}")
            else:
                # Use percentage
                success, result_msg = self.bot.set_risk_per_trade(new_risk_pct)
                if success:
                    msg = f"""✅ **RISK UPDATED (Percentage)**
━━━━━━━━━━━━━━━━━━━━

📊 **New Risk Per Trade**
├ Percentage: {new_risk_pct*100:.2f}%
├ USD Amount: ~${new_risk_usd:.2f}
└ Balance: ${balance:,.2f}

💡 Each trade will now risk {new_risk_pct*100:.2f}% of balance
"""
                    await update.message.reply_text(msg, parse_mode='Markdown')
                else:
                    await update.message.reply_text(f"❌ {result_msg}")
                
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")
            logger.error(f"Risk command error: {e}")
    
    async def cmd_resetstats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Reset all internal stats to zero"""
        try:
            # Reset all stats
            self.bot.stats = {
                'total_trades': 0,
                'wins': 0,
                'losses': 0,
                'total_r': 0.0,
                'win_rate': 0.0
            }
            self.bot.symbol_stats = {}
            self.bot.save_stats()
            
            msg = """✅ **STATS RESET**
━━━━━━━━━━━━━━━━━━━━

All internal tracking stats have been reset to zero.

📊 New Stats:
├ Trades: 0
├ Wins: 0  
├ Losses: 0
├ Total R: 0.0
└ Win Rate: 0.0%

💡 Fresh tracking starts now!
"""
            await update.message.reply_text(msg, parse_mode='Markdown')
            logger.info("Stats reset by user command")
            
        except Exception as e:
            await update.message.reply_text(f"❌ Error resetting stats: {e}")
            logger.error(f"Reset stats error: {e}")
    
    async def cmd_resetlifetime(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Full reset of ALL tracking - lifetime stats, session stats, and baseline balance"""
        try:
            # Require confirmation
            if not context.args or context.args[0].lower() != 'confirm':
                current_balance = await self.bot.broker.get_balance() or 0
                lifetime = self.bot.lifetime_stats
                
                msg = f"""⚠️ **FULL RESET WARNING**
━━━━━━━━━━━━━━━━━━━━

This will reset ALL tracking data:
├ 📊 Session stats → 0
├ 🏆 Lifetime stats → 0  
├ 💰 Baseline → ${current_balance:,.2f} (current balance)
├ 📈 Best/Worst trade → cleared
├ 📉 Max drawdown → cleared
└ 🔥 Streaks → cleared

**Current Lifetime Stats (will be lost):**
├ Total Trades: {lifetime.get('total_trades', 0)}
├ Total R: {lifetime.get('total_r', 0):+.1f}R
├ Total P&L: ${lifetime.get('total_pnl', 0):+,.2f}
└ Since: {lifetime.get('start_date', 'Unknown')}

⚡ To confirm, type:
`/resetlifetime confirm`
"""
                await update.message.reply_text(msg, parse_mode='Markdown')
                return
            
            # Get current balance as new baseline
            new_balance = await self.bot.broker.get_balance() or 0
            today = __import__('datetime').datetime.now().strftime('%Y-%m-%d')
            
            # Reset session stats
            self.bot.stats = {
                'total_trades': 0,
                'wins': 0,
                'losses': 0,
                'total_r': 0.0,
                'win_rate': 0.0,
                'avg_r': 0.0
            }
            self.bot.symbol_stats = {}
            self.bot.save_stats()
            
            # Reset lifetime stats with new baseline (uses StorageHandler for canonical defaults)
            self.bot.lifetime_stats = self.bot.storage.reset_lifetime_stats(new_balance)
            self.bot.recent_trades.clear()

            # Mark all currently open trades so their results don't pollute fresh stats
            pre_reset_count = 0
            for trade in self.bot.active_trades.values():
                if trade is not None:
                    trade.pre_reset = True
                    pre_reset_count += 1

            self.bot.save_lifetime_stats()
            
            open_note = f"\n├ 📡 {pre_reset_count} open trades excluded from new stats" if pre_reset_count > 0 else ""
            msg = f"""✅ **FULL RESET COMPLETE**
━━━━━━━━━━━━━━━━━━━━

All tracking has been reset:
├ 📅 New Start Date: {today}
├ 💰 New Baseline: ${new_balance:,.2f}
├ 📊 Session Stats: 0
├ 🏆 Lifetime Stats: 0
├ 📈 All extremes: cleared{open_note}
└ ⚙️ Regime: Critical 🔴 (SAFE START until 10 trades)

💡 Fresh tracking starts from this moment!
Your P&L Return will now be calculated from ${new_balance:,.0f}.
"""
            await update.message.reply_text(msg, parse_mode='Markdown')
            logger.warning(f"[RESET] Full lifetime reset performed by user. New baseline: ${new_balance:.2f}")
            
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")
            import traceback
            logger.error(f"Reset lifetime error: {traceback.format_exc()}")
    
    async def cmd_debug(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Debug command to show raw Bybit API responses"""
        try:
            msg_parts = ["🔧 **DEBUG INFO**\n━━━━━━━━━━━━━━━━━━━━\n"]
            
            # Get raw positions from Bybit
            msg_parts.append("📊 **POSITIONS API**\n")
            try:
                positions = await self.bot.broker.get_positions()
                if positions is None:
                    msg_parts.append("❌ API returned None\n")
                elif len(positions) == 0:
                    msg_parts.append("⚠️ API returned empty list (0 positions)\n")
                else:
                    # Count positions with size > 0
                    open_count = sum(1 for p in positions if float(p.get('size', 0)) > 0)
                    msg_parts.append(f"✅ Total: {len(positions)} | Open (size>0): {open_count}\n")
                    
                    # Show first 5 open positions
                    msg_parts.append("\n📍 **Sample Positions:**\n")
                    shown = 0
                    for pos in positions:
                        if float(pos.get('size', 0)) > 0:
                            sym = pos.get('symbol', '?')[:12]
                            size = pos.get('size', 0)
                            side = pos.get('side', '?')
                            msg_parts.append(f"├ {sym}: {size} ({side})\n")
                            shown += 1
                            if shown >= 5:
                                break
                    
                    if open_count > 5:
                        msg_parts.append(f"└ ...and {open_count - 5} more\n")
            except Exception as e:
                msg_parts.append(f"❌ Error: {e}\n")
            
            # Get balance
            msg_parts.append("\n💰 **BALANCE API**\n")
            try:
                balance = await self.bot.broker.get_balance()
                msg_parts.append(f"✅ Balance: ${balance:.2f}\n" if balance else "❌ Balance: None\n")
            except Exception as e:
                msg_parts.append(f"❌ Error: {e}\n")
            
            # Internal tracking
            msg_parts.append("\n🔄 **INTERNAL TRACKING**\n")
            msg_parts.append(f"├ active_trades: {len(self.bot.active_trades)}\n")
            msg_parts.append(f"├ pending_signals: {sum(len(s) for s in self.bot.pending_signals.values())}\n")
            msg_parts.append(f"└ enabled_symbols: {len(self.bot.symbol_config.get_enabled_symbols())}\n")
            
            await update.message.reply_text(''.join(msg_parts), parse_mode='Markdown')
            
        except Exception as e:
            await update.message.reply_text(f"❌ Debug error: {e}")
            logger.error(f"Debug command error: {e}")
            
    async def cmd_radar(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show full radar watch for all symbols - handles long messages"""
        try:
            # Build comprehensive radar view with defensive checks
            pending_count = sum(len(sigs) for sigs in self.bot.pending_signals.values()) if self.bot.pending_signals else 0
            developing_count = 0
            extreme_count = 0
            
            if self.bot.radar_items:
                for data in self.bot.radar_items.values():
                    if isinstance(data, dict):
                        data_type = data.get('type', '')
                        if data_type in ['bullish_setup', 'bearish_setup']:
                            developing_count += 1
                        elif data_type in ['extreme_oversold', 'extreme_overbought']:
                            extreme_count += 1
            
            # Get divergence type summary
            div_summary = self.bot.symbol_config.get_divergence_summary()
            
            msg = f"""📡 **FULL RADAR WATCH**
━━━━━━━━━━━━━━━━━━━━

📊 **Portfolio**: {self.bot.symbol_config.get_total_enabled()} symbols
├ 🟢 `REG_BULL`: {div_summary.get('REG_BULL', 0)}
├ 🔴 `REG_BEAR`: {div_summary.get('REG_BEAR', 0)}
├ 🟠 `HID_BULL`: {div_summary.get('HID_BULL', 0)}
└ 🟡 `HID_BEAR`: {div_summary.get('HID_BEAR', 0)}

🎯 **Active Signals**: {pending_count + developing_count + extreme_count}

"""
            
            # 1. Pending BOS - show with divergence code
            if self.bot.pending_signals:
                msg += "⏳ **PENDING BOS (Waiting for Breakout)**\n\n"
                for sym, sigs in list(self.bot.pending_signals.items())[:10]:  # Limit to 10
                    for sig in sigs:
                        div_code = getattr(sig.signal, 'divergence_code', sig.signal.signal_type.upper()[:3])
                        side_icon = "🟢" if sig.signal.side == 'long' else "🔴"
                        candles_left = 12 - sig.candles_waited
                        msg += f"{side_icon} **{sym}** `{div_code}`: {sig.candles_waited}/12 → {candles_left}h max\n"
                
                if pending_count > 10:
                    msg += f"_...and {pending_count - 10} more_\n"
                msg += "\n"
            
            # 2. Developing Setups (limit to 5)
            developing = []
            extreme = []
            
            if self.bot.radar_items:
                for sym, data in self.bot.radar_items.items():
                    if isinstance(data, dict):
                        data_type = data.get('type', '')
                        if data_type in ['bullish_setup', 'bearish_setup']:
                            developing.append((sym, data))
                        elif data_type in ['extreme_oversold', 'extreme_overbought']:
                            extreme.append((sym, data))
            
            if developing:
                msg += "🔮 **DEVELOPING PATTERNS**\n\n"
                for sym, data in developing[:5]:  # Limit to 5
                    try:
                        data_type = data.get('type', '')
                        progress = int(data.get('pivot_progress', 3) or 3)
                        progress = max(0, min(6, progress))
                        rsi = float(data.get('rsi', 0) or 0)
                        side_icon = "🟢" if data_type == 'bullish_setup' else "🔴"
                        side_name = "Bull" if data_type == 'bullish_setup' else "Bear"
                        
                        msg += f"{side_icon} **{sym}**: {side_name} forming ({progress}/6) RSI:{rsi:.0f}\n"
                    except:
                        pass
                
                if len(developing) > 5:
                    msg += f"_...and {len(developing) - 5} more_\n"
                msg += "\n"
            
            if extreme:
                msg += "⚡ **EXTREME ZONES**\n\n"
                for sym, data in extreme[:5]:  # Limit to 5
                    try:
                        data_type = data.get('type', '')
                        rsi = float(data.get('rsi', 0) or 0)
                        hours = float(data.get('hours_in_zone', 0) or 0)
                        
                        if data_type == 'extreme_oversold':
                            msg += f"❄️ **{sym}**: RSI {rsi:.0f} ({hours:.0f}h oversold)\n"
                        else:
                            msg += f"🔥 **{sym}**: RSI {rsi:.0f} ({hours:.0f}h overbought)\n"
                    except:
                        pass
                
                if len(extreme) > 5:
                    msg += f"_...and {len(extreme) - 5} more_\n"
            
            if not self.bot.pending_signals and not developing and not extreme:
                msg += "✨ All clear - no active radar signals\n"
            
            msg += "\n💡 /dashboard /positions"
            
            # Handle long messages - Telegram limit is 4096 chars
            if len(msg) > 4000:
                # Split into chunks
                chunks = [msg[i:i+4000] for i in range(0, len(msg), 4000)]
                for i, chunk in enumerate(chunks):
                    if i == 0:
                        await update.message.reply_text(chunk, parse_mode='Markdown')
                    else:
                        await update.message.reply_text(f"...(continued)\n{chunk}", parse_mode='Markdown')
            else:
                await update.message.reply_text(msg, parse_mode='Markdown')
            
        except Exception as e:
            import traceback
            logger.error(f"Error in cmd_radar: {e}")
            logger.error(traceback.format_exc())
            await update.message.reply_text(f"❌ Radar error: {str(e)[:100]}")

    async def cmd_setbalance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Set the starting balance baseline for P&L calculations (accounts for deposits)"""
        try:
            if not context.args:
                # Show current baseline
                starting_balance = self.bot.lifetime_stats.get('starting_balance', 0)
                start_date = self.bot.lifetime_stats.get('start_date', 'Unknown')
                current_balance = await self.bot.broker.get_balance() or 0
                
                msg = f"""💰 **BASELINE BALANCE**
━━━━━━━━━━━━━━━━━━━━

📊 **Current Baseline**: ${starting_balance:,.2f}
├ Set on: {start_date}
├ Current Balance: ${current_balance:,.2f}
└ Difference: ${current_balance - starting_balance:+,.2f}

⚙️ **To Update:**
`/setbalance 700` → Set baseline to $700
`/setbalance now` → Set baseline to current balance

💡 Use this after making deposits to reset P&L tracking.
"""
                await update.message.reply_text(msg, parse_mode='Markdown')
                return
            
            # Parse the new balance
            input_val = context.args[0].lower().strip()
            
            if input_val == 'now':
                # Set to current balance
                new_balance = await self.bot.broker.get_balance() or 0
            else:
                try:
                    new_balance = float(input_val.replace('$', '').replace(',', ''))
                except ValueError:
                    await update.message.reply_text("❌ Invalid format. Use `/setbalance 700` or `/setbalance now`")
                    return
            
            if new_balance <= 0:
                await update.message.reply_text("❌ Balance must be positive.")
                return
            
            # Update lifetime stats
            old_balance = self.bot.lifetime_stats.get('starting_balance', 0)
            self.bot.lifetime_stats['starting_balance'] = new_balance
            self.bot.save_lifetime_stats()
            
            msg = f"""✅ **BASELINE UPDATED**
━━━━━━━━━━━━━━━━━━━━

📊 **New Baseline**: ${new_balance:,.2f}
├ Previous: ${old_balance:,.2f}
└ Change: ${new_balance - old_balance:+,.2f}

💡 P&L Return % will now be calculated relative to ${new_balance:,.0f}.
"""
            await update.message.reply_text(msg, parse_mode='Markdown')
            logger.info(f"[SETBALANCE] Updated baseline from ${old_balance:.2f} to ${new_balance:.2f} by user command")
            
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")
            logger.error(f"Setbalance command error: {e}")

    async def cmd_setregime(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Manually override the regime risk tier"""
        try:
            valid_regimes = ['favorable', 'cautious', 'adverse', 'critical']
            regime_icons = {
                'favorable': '🟢', 'cautious': '🟡', 'adverse': '🟠', 'critical': '🔴',
            }
            regime_mults = {
                'favorable': '100%', 'cautious': '50%', 'adverse': '25%', 'critical': '10%',
            }

            if not context.args:
                # Show current status + usage
                current = self.bot.regime_override
                if current:
                    remaining = 10 - self.bot.regime_override_trades
                    status = f"**Active override**: {current.title()} {regime_icons[current]} ({regime_mults[current]} risk)\n├ Auto-clears in {remaining} trades"
                else:
                    label, mult, _ = self.bot.get_regime_status()
                    icon = regime_icons.get(label, '❓')
                    status = f"**No override active** — auto-detected: {label.title()} {icon} ({mult:.0%} risk)"

                msg = f"""⚙️ **REGIME OVERRIDE**
━━━━━━━━━━━━━━━━━━━━

{status}

**Usage:**
`/setregime cautious` → Force 50% risk
`/setregime adverse` → Force 25% risk
`/setregime critical` → Force 10% risk
`/setregime favorable` → Force 100% risk
`/setregime clear` → Return to auto-detection

💡 Override auto-clears after 10 new trades.
"""
                await update.message.reply_text(msg, parse_mode='Markdown')
                return

            arg = context.args[0].lower().strip()

            if arg == 'clear':
                self.bot.set_regime_override(None)
                label, mult, _ = self.bot.get_regime_status()
                icon = regime_icons.get(label, '❓')
                await update.message.reply_text(
                    f"✅ **Regime override cleared**\n\nAuto-detected: {label.title()} {icon} ({mult:.0%} risk)",
                    parse_mode='Markdown'
                )
                logger.info(f"[REGIME] Manual override cleared by user command")
                return

            if arg not in valid_regimes:
                await update.message.reply_text(
                    f"❌ Invalid regime `{arg}`\n\nValid: `favorable`, `cautious`, `adverse`, `critical`, `clear`",
                    parse_mode='Markdown'
                )
                return

            self.bot.set_regime_override(arg)
            icon = regime_icons[arg]
            mult = regime_mults[arg]
            await update.message.reply_text(
                f"✅ **Regime override set**: {arg.title()} {icon} ({mult} risk)\n\n"
                f"Will auto-clear after 10 new trades.\nUse `/setregime clear` to remove manually.",
                parse_mode='Markdown'
            )
            logger.info(f"[REGIME] Manual override set to '{arg}' by user command")

        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")
            logger.error(f"Setregime command error: {e}")

