"""
ML Rankings Extension for Telegram Bot
Add this to your telegram_bot.py file
"""

# Add this command handler in __init__ after the other handlers:
# self.app.add_handler(CommandHandler("mlrankings", self.ml_rankings))

# Add this to the help text:
# /mlrankings - Symbol performance rankings

async def ml_stats(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Show ML scorer status"""
    try:
        ml_scorer = self.shared.get("ml_scorer")
        if not ml_scorer:
            await update.message.reply_text("ML Scorer not initialized")
            return
        
        msg = "🤖 *ML Signal Scorer Status*\n"
        msg += "=" * 25 + "\n\n"
        
        # Get counts
        completed = ml_scorer.completed_trades_count if hasattr(ml_scorer, 'completed_trades_count') else 0
        is_trained = ml_scorer.is_trained if hasattr(ml_scorer, 'is_trained') else False
        min_trades = ml_scorer.MIN_TRADES_TO_LEARN if hasattr(ml_scorer, 'MIN_TRADES_TO_LEARN') else 200
        
        # Status
        if is_trained:
            msg += "✅ *Status: ACTIVE*\n"
            msg += "ML is scoring signals\n\n"
        else:
            msg += "📈 *Status: LEARNING*\n"
            msg += f"Collecting data ({completed}/{min_trades})\n\n"
        
        # Progress bar
        progress = min(completed / min_trades * 100, 100)
        filled = int(progress / 10)
        bar = '■' * filled + '□' * (10 - filled)
        msg += f"*Progress:* {bar} {progress:.1f}%\n\n"
        
        # Stats
        msg += f"*Completed Trades:* {completed}\n"
        msg += f"*Training Threshold:* {min_trades}\n"
        
        if is_trained:
            msg += f"*Model Status:* Trained ✅\n"
            if hasattr(ml_scorer, 'min_score'):
                msg += f"*Min Score Required:* {ml_scorer.min_score}%\n"
        else:
            trades_needed = max(0, min_trades - completed)
            msg += f"*Trades Until Training:* {trades_needed}\n"
        
        msg += "\n\nUse /mlrankings to see symbol performance"
        
        await update.message.reply_text(msg, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Error in ml_stats: {e}")
        await update.message.reply_text("Error getting ML status")

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
            
            if trade.pnl_usd > 0:
                stats['wins'] += 1
            else:
                stats['losses'] += 1
            
            stats['total_pnl'] += trade.pnl_usd
            stats['best_trade'] = max(stats['best_trade'], trade.pnl_usd)
            stats['worst_trade'] = min(stats['worst_trade'], trade.pnl_usd)
            
            # Track last 5 trades for trend
            stats['last_5_trades'].append(1 if trade.pnl_usd > 0 else 0)
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
        msg += f"Profitable: {profitable_symbols} ({profitable_symbols/total_symbols*100:.0f}%)\n"
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
        
        await update.message.reply_text(msg, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Error in ml_rankings: {e}")
        import traceback
        logger.error(traceback.format_exc())
        await update.message.reply_text(f"Error generating rankings: {str(e)[:100]}")