#!/usr/bin/env python3
"""
Add /reset_stats command to Telegram bot
This allows resetting trade statistics directly from Telegram
"""

# Add this method to the TGBot class in telegram_bot.py:

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
            # Create backup
            with open("trade_history.json", 'r') as f:
                data = json.load(f)
                trade_count = len(data)
            
            if trade_count > 0:
                backup_name = f"trade_history_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                os.rename("trade_history.json", backup_name)
                backups.append(f"• {trade_count} trades → {backup_name}")
                reset_count += trade_count
                
                # Create empty file
                with open("trade_history.json", 'w') as f:
                    json.dump([], f)
        
        # 2. Reset any cached stats in TradeTracker
        if hasattr(self.shared.get("tracker"), "trades"):
            self.shared["tracker"].trades = []
            self.shared["tracker"].save_trades()
        
        # 3. Reset ML trade count (keep model but reset counter)
        if "ml_scorer" in self.shared and self.shared["ml_scorer"]:
            ml_scorer = self.shared["ml_scorer"]
            old_count = ml_scorer.completed_trades_count
            ml_scorer.completed_trades_count = 0
            ml_scorer.last_train_count = 0
            
            # Clear completed trades but keep model
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
            
            backups.append(f"• ML: Reset {old_count} completed trades counter")
        
        # Build response
        if reset_count > 0 or backups:
            response = "✅ **Statistics Reset Complete!**\n\n"
            
            if backups:
                response += "**Backed up:**\n"
                response += "\n".join(backups) + "\n\n"
            
            response += "**What happens now:**\n"
            response += "• Trade history: Starting fresh at 0\n"
            response += "• Win rate: Will recalculate from new trades\n"
            response += "• ML: Keeps learned model but resets counter\n"
            response += "• New trades will build fresh statistics\n\n"
            response += "📊 Use /stats to see fresh statistics\n"
            response += "🤖 Use /ml to check ML status"
        else:
            response = "ℹ️ No statistics to reset - already clean"
        
        await update.message.reply_text(response, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Error resetting stats: {e}")
        await update.message.reply_text(f"❌ Error resetting stats: {str(e)[:200]}")

# Also add this line to __init__ method where other handlers are registered:
# self.app.add_handler(CommandHandler("reset_stats", self.reset_stats))