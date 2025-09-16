#!/bin/bash
cd "/Users/lualakol/AutoTrading Bot"
git add live_bot.py telegram_bot.py
git commit -m "Fix /update_clusters command to generate clusters properly

- Modified telegram_bot.py to call bot's auto_generate_enhanced_clusters
- Added bot_instance to shared dict for telegram access
- Now uses simple clustering instead of broken enhanced algorithm
- Ensures BTCUSDT is correctly classified as Blue Chip

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"
git push origin main