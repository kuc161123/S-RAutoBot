from typing import List, Dict, Any
from datetime import datetime

def format_number(value: float, decimals: int = 2) -> str:
    """Format number with thousands separator"""
    if abs(value) >= 1000000:
        return f"{value/1000000:.{decimals}f}M"
    elif abs(value) >= 1000:
        return f"{value/1000:.{decimals}f}K"
    else:
        return f"{value:.{decimals}f}"

def format_status(
    trading_enabled: bool,
    account_info: Dict,
    positions: List[Dict],
    monitored_symbols: set,
    total_zones: int
) -> str:
    """Format bot status message"""
    
    # Extract account details
    balance = float(account_info.get('totalWalletBalance', 0))
    available = float(account_info.get('availableBalance', 0))
    unrealized_pnl = float(account_info.get('totalPerpUPL', 0))
    
    # Calculate metrics
    num_positions = len(positions)
    total_position_value = sum(float(p.get('positionValue', 0)) for p in positions)
    
    # Status emoji
    status_emoji = "🟢" if trading_enabled else "🔴"
    status_text = "Active" if trading_enabled else "Inactive"
    
    # PnL formatting
    pnl_emoji = "📈" if unrealized_pnl >= 0 else "📉"
    pnl_color = "+" if unrealized_pnl >= 0 else ""
    
    text = f"""
{status_emoji} *Bot Status: {status_text}*

💰 *Account:*
Balance: ${format_number(balance)}
Available: ${format_number(available)}
Unrealized PnL: {pnl_emoji} {pnl_color}${format_number(unrealized_pnl)}

📊 *Positions:*
Open: {num_positions}
Total Value: ${format_number(total_position_value)}

🎯 *Monitoring:*
Symbols: {len(monitored_symbols)}
Active Zones: {total_zones}

⏰ Last Update: {datetime.now().strftime('%H:%M:%S')}
"""
    
    return text.strip()

def format_positions(positions: List[Dict]) -> str:
    """Format open positions"""
    if not positions:
        return "📊 No open positions"
    
    text = "📊 *Open Positions:*\n\n"
    
    for pos in positions:
        symbol = pos['symbol']
        side = pos['side']
        size = float(pos['size'])
        entry_price = float(pos['avgPrice'])
        mark_price = float(pos['markPrice'])
        unrealized_pnl = float(pos['unrealizedPnl'])
        pnl_percent = (unrealized_pnl / float(pos['positionValue'])) * 100 if float(pos['positionValue']) > 0 else 0
        
        # Emoji based on side
        side_emoji = "🟢" if side == "Buy" else "🔴"
        
        # PnL formatting
        pnl_emoji = "📈" if unrealized_pnl >= 0 else "📉"
        pnl_color = "+" if unrealized_pnl >= 0 else ""
        
        text += f"""
{side_emoji} *{symbol}*
Side: {side} | Size: {size}
Entry: ${entry_price:.4f}
Mark: ${mark_price:.4f}
PnL: {pnl_emoji} {pnl_color}${unrealized_pnl:.2f} ({pnl_color}{pnl_percent:.2f}%)
"""
    
    return text.strip()

def format_backtest_result(result: Dict[str, Any]) -> str:
    """Format backtest results"""
    
    # Extract metrics
    total_trades = result.get('total_trades', 0)
    win_rate = result.get('win_rate', 0)
    profit_factor = result.get('profit_factor', 0)
    total_return = result.get('total_return', 0)
    max_drawdown = result.get('max_drawdown', 0)
    sharpe_ratio = result.get('sharpe_ratio', 0)
    avg_win = result.get('avg_win', 0)
    avg_loss = result.get('avg_loss', 0)
    best_trade = result.get('best_trade', 0)
    worst_trade = result.get('worst_trade', 0)
    
    # Zone statistics
    zone_stats = result.get('zone_stats', {})
    demand_win_rate = zone_stats.get('demand_win_rate', 0)
    supply_win_rate = zone_stats.get('supply_win_rate', 0)
    fresh_zone_win_rate = zone_stats.get('fresh_zone_win_rate', 0)
    
    # Probability scores
    probabilities = result.get('probability_scores', {})
    
    text = f"""
📈 *Backtest Results*

📊 *Performance Metrics:*
Total Trades: {total_trades}
Win Rate: {win_rate:.1f}%
Profit Factor: {profit_factor:.2f}
Total Return: {'+' if total_return >= 0 else ''}{total_return:.2f}%
Max Drawdown: -{max_drawdown:.2f}%
Sharpe Ratio: {sharpe_ratio:.2f}

💰 *Trade Statistics:*
Avg Win: +{avg_win:.2f}%
Avg Loss: -{abs(avg_loss):.2f}%
Best Trade: +{best_trade:.2f}%
Worst Trade: -{abs(worst_trade):.2f}%

🎯 *Zone Performance:*
Demand Zones: {demand_win_rate:.1f}% win rate
Supply Zones: {supply_win_rate:.1f}% win rate
Fresh Zones: {fresh_zone_win_rate:.1f}% win rate

🔮 *Probability Scores:*
Score 70-80: {probabilities.get('70_80', {}).get('win_rate', 0):.1f}% ({probabilities.get('70_80', {}).get('count', 0)} trades)
Score 80-90: {probabilities.get('80_90', {}).get('win_rate', 0):.1f}% ({probabilities.get('80_90', {}).get('count', 0)} trades)
Score 90+: {probabilities.get('90_100', {}).get('win_rate', 0):.1f}% ({probabilities.get('90_100', {}).get('count', 0)} trades)

📅 Period: {result.get('start_date', 'N/A')} to {result.get('end_date', 'N/A')}
"""
    
    return text.strip()

def format_trade_signal(signal: Dict[str, Any]) -> str:
    """Format trade signal notification"""
    
    symbol = signal['symbol']
    side = signal['side']
    entry = signal['entry_price']
    stop = signal['stop_loss']
    tp1 = signal['take_profit_1']
    tp2 = signal['take_profit_2']
    size = signal['position_size']
    zone_type = signal['zone_type']
    score = signal['zone_score']
    confidence = signal['confidence']
    
    # Emoji based on side
    side_emoji = "🟢 LONG" if side == "Buy" else "🔴 SHORT"
    
    # Risk calculation
    risk_percent = abs(entry - stop) / entry * 100
    reward1 = abs(tp1 - entry) / abs(entry - stop)
    reward2 = abs(tp2 - entry) / abs(entry - stop)
    
    text = f"""
🚨 *NEW TRADE SIGNAL*

{side_emoji} *{symbol}*

📍 *Entry:* ${entry:.4f}
🛑 *Stop Loss:* ${stop:.4f} (-{risk_percent:.1f}%)
🎯 *TP1:* ${tp1:.4f} (1:{reward1:.1f}R)
🎯 *TP2:* ${tp2:.4f} (1:{reward2:.1f}R)

📊 *Details:*
Position Size: {size}
Zone Type: {zone_type.title()} Zone
Zone Score: {score:.1f}/100
Confidence: {confidence:.1f}%

⚠️ Risk Warning: This is an automated signal. Trade at your own risk.
"""
    
    return text.strip()

def format_trade_closed(trade: Dict[str, Any]) -> str:
    """Format trade closed notification"""
    
    symbol = trade['symbol']
    side = trade['side']
    entry = trade['entry_price']
    exit = trade['exit_price']
    pnl = trade['pnl']
    pnl_percent = trade['pnl_percent']
    duration = trade['duration']
    reason = trade['close_reason']
    
    # Result emoji
    if pnl > 0:
        result_emoji = "✅ WIN"
        pnl_sign = "+"
    elif pnl < 0:
        result_emoji = "❌ LOSS"
        pnl_sign = ""
    else:
        result_emoji = "➖ BREAKEVEN"
        pnl_sign = ""
    
    text = f"""
{result_emoji} *TRADE CLOSED*

📊 *{symbol}* ({side})

Entry: ${entry:.4f}
Exit: ${exit:.4f}
PnL: {pnl_sign}${pnl:.2f} ({pnl_sign}{pnl_percent:.2f}%)

Duration: {duration}
Reason: {reason}
"""
    
    return text.strip()

def format_error(error_type: str, error_message: str) -> str:
    """Format error notification"""
    
    text = f"""
⚠️ *ERROR ALERT*

Type: {error_type}
Message: {error_message}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Please check the logs for more details.
"""
    
    return text.strip()

def format_daily_summary(summary: Dict[str, Any]) -> str:
    """Format daily trading summary"""
    
    date = summary.get('date', datetime.now().strftime('%Y-%m-%d'))
    total_trades = summary.get('total_trades', 0)
    winning_trades = summary.get('winning_trades', 0)
    losing_trades = summary.get('losing_trades', 0)
    total_pnl = summary.get('total_pnl', 0)
    total_fees = summary.get('total_fees', 0)
    net_pnl = total_pnl - total_fees
    
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    # PnL formatting
    pnl_emoji = "📈" if net_pnl >= 0 else "📉"
    pnl_sign = "+" if net_pnl >= 0 else ""
    
    text = f"""
📅 *Daily Summary - {date}*

📊 *Trading Activity:*
Total Trades: {total_trades}
Winners: {winning_trades} ({win_rate:.1f}%)
Losers: {losing_trades}

💰 *Profit & Loss:*
Gross PnL: {'+' if total_pnl >= 0 else ''}${total_pnl:.2f}
Fees: -${total_fees:.2f}
Net PnL: {pnl_emoji} {pnl_sign}${net_pnl:.2f}

🏆 *Best Performers:*
{summary.get('best_symbol', 'N/A')}: +${summary.get('best_pnl', 0):.2f}

💔 *Worst Performers:*
{summary.get('worst_symbol', 'N/A')}: -${abs(summary.get('worst_pnl', 0)):.2f}
"""
    
    return text.strip()