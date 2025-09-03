#!/usr/bin/env python3
"""
QUICK FIX: Stop ML from learning false data
Run this to apply a temporary fix
"""

print("""
=====================================
🚨 IMMEDIATE ACTION REQUIRED
=====================================

THE PROBLEM:
ML is recording OPEN trades as "completed" with fake outcomes.
This ruins the ML model with false data.

QUICK FIX (Do this NOW):
========================

1. STOP THE BOT
   Ctrl+C to stop live_bot.py

2. RESET ML DATA
   Run: python3 reset_all_stats.py
   Type: RESET
   
3. DISABLE ML TEMPORARILY
   Edit config.yaml:
   
   trade:
     use_ml_scoring: false  # <- Change to false
   
4. RESTART BOT
   python3 live_bot.py
   
This will:
- Stop ML from learning false data
- Clear the corrupted data
- Let bot trade WITHOUT ML scoring
- Use only the pullback strategy rules

BETTER SOLUTION (After stopping bleeding):
==========================================
We need to fix check_closed_positions() in live_bot.py
to only record trades that ACTUALLY hit TP or SL.

Current bug: It's checking if symbol is in exchange positions,
but this check is failing and marking open trades as closed.

Would you like me to:
A) Apply the quick fix above first (recommended)
B) Fix the code properly in live_bot.py
C) Both - quick fix now, proper fix after

Type your choice: 
""")

# Save this emergency config
import yaml
import shutil
from datetime import datetime

def emergency_disable_ml():
    """Emergency disable ML to stop false learning"""
    
    # Backup config
    shutil.copy('config.yaml', f'config_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.yaml')
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Disable ML
    config['trade']['use_ml_scoring'] = False
    
    # Save
    with open('config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print("✅ ML scoring DISABLED in config.yaml")
    print("Now restart your bot!")

if __name__ == "__main__":
    response = input("Apply emergency ML disable? (y/n): ")
    if response.lower() == 'y':
        emergency_disable_ml()