#!/usr/bin/env python3
"""
LIVE vs BACKTEST DISCREPANCY INVESTIGATION
===========================================
Analyzes why live BOS confirmation rate is much lower than backtest predictions.

User reports:
- 79 symbols configured
- Only ~3 BOS confirmations per day
- Backtest predicted ~4.6 BOS/day for just 28 symbols

Potential causes:
1. Signal deduplication too aggressive
2. Stale signal filtering
3. Trend filter (EMA 200) rejecting signals
4. BOS timeout (6 candles)
5. Recent market conditions
6. Bot logic differences from backtest
"""

import pandas as pd
import json
import os
from datetime import datetime, timedelta

print("="*80)
print("LIVE vs BACKTEST DISCREPANCY INVESTIGATION")
print("="*80)

print(f"\n🔍 USER REPORT:")
print(f"├─ Configured Symbols: 79")
print(f"├─ Live BOS/Day: ~3")
print(f"└─ Backtest Prediction (28 symbols): ~4.6 BOS/day")

print(f"\n⚠️  DISCREPANCY: 79 symbols live < 28 symbols backtest!")
print(f"    Expected ratio: 79/28 = 2.8x MORE trades")
print(f"    Actual: FEWER trades despite more symbols!")

# Try to load bot state files to investigate
print("\n" + "="*80)
print("CHECKING BOT STATE FILES")
print("="*80)

issues_found = []

# Check seen_signals.json
seen_signals_path = 'data/seen_signals.json'
if os.path.exists(seen_signals_path):
    try:
        with open(seen_signals_path, 'r') as f:
            seen_signals = json.load(f)
        
        print(f"\n✅ Found seen_signals.json")
        print(f"├─ Total unique signals seen: {len(seen_signals)}")
        
        # Check for old signals (signal deduplication issue)
        # Signal IDs format: symbol_side_YYYYMMDD_HH
        recent_count = 0
        old_count = 0
        now = datetime.now()
        
        for sig_id in seen_signals:
            try:
                # Extract date from signal ID
                parts = sig_id.split('_')
                if len(parts) >= 3:
                    date_str = parts[-2]  # YYYYMMDD
                    sig_date = datetime.strptime(date_str, '%Y%m%d')
                    age_days = (now - sig_date).days
                    
                    if age_days <= 7:
                        recent_count += 1
                    else:
                        old_count += 1
            except:
                continue
        
        print(f"├─ Recent (<7 days): {recent_count}")
        print(f"└─ Old (>7 days): {old_count}")
        
        if old_count > 1000:
            issues_found.append({
                'issue': 'EXCESSIVE SIGNAL HISTORY',
                'severity': 'HIGH',
                'description': f'{old_count} old signals stored, may be blocking new detections',
                'fix': 'Clear old signals from seen_signals.json (keep only last 7 days)'
            })
            
    except Exception as e:
        print(f"\n⚠️  Error reading seen_signals.json: {e}")
else:
    print(f"\n⚪ seen_signals.json not found at {seen_signals_path}")

# Check stats.json for actual live performance
stats_path = 'stats.json'
if os.path.exists(stats_path):
    try:
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        
        print(f"\n✅ Found stats.json")
        print(f"├─ Total Trades: {stats.get('total_trades', 0)}")
        print(f"├─ Win Rate: {stats.get('win_rate', 0):.1f}%")
        print(f"└─ Total R: {stats.get('total_r', 0):+.1f}R")
        
    except Exception as e:
        print(f"\n⚠️  Error reading stats.json: {e}")
else:
    print(f"\n⚪ stats.json not found")

# Analyze potential causes
print("\n" + "="*80)
print("POTENTIAL CAUSES OF LOW BOS RATE")
print("="*80)

causes = [
    {
        'cause': '1. SIGNAL DEDUPLICATION TOO AGGRESSIVE',
        'likelihood': 'HIGH',
        'description': 'seen_signals.json may have accumulated too many old signals',
        'symptoms': [
            'Bot detects divergence but marks it as "already seen"',
            'Telegram shows divergence detected but no BOS wait messages',
            'seen_signals.json file is very large'
        ],
        'fix': 'Clear old signals (>7 days) from seen_signals.json'
    },
    {
        'cause': '2. STALE SIGNAL FILTERING',
        'likelihood': 'MEDIUM-HIGH',
        'description': 'Divergences detected but price already passed swing level',
        'symptoms': [
            'Bot logs show divergences detected',
            'But immediately marked as "stale" and skipped',
            'No "waiting for BOS" messages'
        ],
        'fix': 'Review stale signal detection logic - may be too strict'
    },
    {
        'cause': '3. TREND FILTER (EMA 200) REJECTING SIGNALS',
        'likelihood': 'MEDIUM',
        'description': 'Price keeps crossing EMA 200, invalidating pending signals',
        'symptoms': [
            'Pending signals removed due to trend invalidation',
            'Many "trend no longer aligned" messages',
            'Market in choppy/sideways conditions'
        ],
        'fix': 'Consider looser trend filter or remove it temporarily'
    },
    {
        'cause': '4. BOS TIMEOUT (6 CANDLES)',
        'likelihood': 'MEDIUM',
        'description': 'Signals expire before BOS occurs',
        'symptoms': [
            'Many "signal expired after 6 candles" messages',
            'Divergences detected but never execute',
            'Low BOS confirmation rate'
        ],
        'fix': 'Increase MAX_WAIT_CANDLES from 6 to 10-12'
    },
    {
        'cause': '5. MARKET CONDITIONS (LOW VOLATILITY)',
        'likelihood': 'LOW-MEDIUM',
        'description': 'Recent market is consolidating with fewer divergences',
        'symptoms': [
            'Overall low divergence detection rate',
            'Applies to all symbols equally',
            'Market in tight ranges'
        ],
        'fix': 'Wait for more volatile conditions or adjust strategy'
    },
    {
        'cause': '6. BOT RESTART CLEARING PENDING SIGNALS',
        'likelihood': 'LOW',
        'description': 'Bot restarts lose pending signals waiting for BOS',
        'symptoms': [
            'Frequent bot restarts',
            'Pending signals not persisted to disk',
            'Signals detected but lost before BOS'
        ],
        'fix': 'Persist pending_signals to disk'
    }
]

for i, cause in enumerate(causes, 1):
    print(f"\n{cause['cause']}")
    print(f"Likelihood: {cause['likelihood']}")
    print(f"Description: {cause['description']}")
    print(f"\nSymptoms:")
    for symptom in cause['symptoms']:
        print(f"  • {symptom}")
    print(f"Fix: {cause['fix']}")
    print("-" * 80)

# Immediate recommendations
print("\n" + "="*80)
print("IMMEDIATE ACTION ITEMS")
print("="*80)

print("""
🔧 RECOMMENDED DEBUGGING STEPS:

1. CHECK BOT LOGS (TONIGHT):
   • Look for "DIVERGENCE DETECTED" messages
   • Count how many per hour/day
   • Check if they're marked as "already seen" or "stale"
   • Look for "BOS CONFIRMED" vs "signal expired" ratio

2. CHECK seen_signals.json:
   • How many signals stored?
   • How old are they?
   • Try clearing signals older than 7 days

3. MONITOR PENDING SIGNALS:
   • Check /radar command on Telegram
   • Are divergences being detected but not reaching BOS?
   • How long do they wait before expiring?

4. TEST WITH FEWER SYMBOLS:
   • Temporarily disable 50 symbols
   • Keep only the 28 validated ones
   • Compare BOS rate over 24 hours

5. ENABLE VERBOSE LOGGING:
   • Add detailed logs for:
     - Divergence detection
     - Stale signal checks
     - Trend alignment checks
     - BOS confirmation checks
     - Signal expiration

📊 EXPECTED vs ACTUAL COMPARISON:

With 79 symbols:
├─ Backtest Expectation: ~13 BOS/day (79/28 * 4.6)
├─ Your Reality: ~3 BOS/day
└─ Discrepancy: 4.3x LOWER than expected!

With 28 validated symbols:
├─ Backtest Expectation: ~4.6 BOS/day
└─ Test this to verify backtest accuracy

🎯 MOST LIKELY CULPRITS:

1. Signal deduplication (seen_signals.json bloated)
2. Stale signal detection too aggressive
3. Combination of both

💡 QUICK TEST:

Backup and clear seen_signals.json:
```bash
cp data/seen_signals.json data/seen_signals.json.backup
echo '[]' > data/seen_signals.json
```

Then monitor for 24 hours. If BOS rate increases significantly,
signal deduplication was the issue.
""")

if issues_found:
    print("\n" + "="*80)
    print("ISSUES DETECTED")
    print("="*80)
    for issue in issues_found:
        print(f"\n⚠️  {issue['issue']} (Severity: {issue['severity']})")
        print(f"   {issue['description']}")
        print(f"   FIX: {issue['fix']}")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)
print("""
1. Share your bot logs from the last 24 hours
2. Check seen_signals.json file size/age
3. Try clearing old signals as a test
4. Enable more detailed logging
5. Monitor /radar command output

I can help analyze logs and implement fixes once we identify the root cause.
""")
