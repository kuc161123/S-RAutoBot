#!/usr/bin/env python3
import subprocess
import os

os.chdir("/Users/lualakol/AutoTrading Bot")

# Stage changes
subprocess.run(["git", "add", "live_bot.py", "telegram_bot.py"])

# Commit
commit_msg = """Fix /update_clusters command to generate clusters properly

- Modified telegram_bot.py to call bot's auto_generate_enhanced_clusters
- Added bot_instance to shared dict for telegram access  
- Now uses simple clustering instead of broken enhanced algorithm
- Ensures BTCUSDT is correctly classified as Blue Chip

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"""

subprocess.run(["git", "commit", "-m", commit_msg])

# Push
subprocess.run(["git", "push", "origin", "main"])

print("Changes committed and pushed successfully!")