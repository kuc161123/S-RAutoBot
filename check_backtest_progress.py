#!/usr/bin/env python3
import os
import re
from datetime import datetime

log_file = 'backtest_400.log'

if not os.path.exists(log_file):
    print("❌ Backtest log not found. Has it started?")
    exit(1)

with open(log_file, 'r') as f:
    lines = f.readlines()

# Parse log
processed = 0
passed = 0
failed = 0
start_time = None
latest_symbol = None

for line in lines:
    # Check start time
    if 'Started:' in line:
        try:
            start_time = datetime.strptime(line.split('Started: ')[1].strip(), '%Y-%m-%d %H:%M:%S')
        except:
            pass
    
    # Count processed
    if re.search(r'\[\d+/400\]', line):
        match = re.search(r'\[(\d+)/400\] (\w+)', line)
        if match:
            processed = int(match.group(1))
            latest_symbol = match.group(2)
    
    # Count passed
    if '✅' in line:
        passed += 1
    elif '⚠️' in line or 'Insufficient data' in line:
        failed += 1

# Calculate progress
total = 400
remaining = total - processed
progress_pct = (processed / total) * 100 if total > 0 else 0
pass_rate = (passed / processed * 100) if processed > 0 else 0

# Estimate completion
if start_time and processed > 0:
    elapsed = (datetime.now() - start_time).total_seconds()
    avg_per_symbol = elapsed / processed
    remaining_seconds = avg_per_symbol * remaining
    eta_minutes = remaining_seconds / 60
    eta_hours = eta_minutes / 60
    
    print(f"\n{'='*60}")
    print(f"🔬 BACKTEST PROGRESS")
    print(f"{'='*60}")
    print(f"Started:      {start_time.strftime('%H:%M:%S')}")
    print(f"Elapsed:      {int(elapsed/60)} minutes")
    print(f"Latest:       {latest_symbol or 'N/A'}")
    print(f"")
    print(f"Progress:     {processed}/{total} symbols ({progress_pct:.1f}%)")
    print(f"Passed:       {passed} symbols ({pass_rate:.1f}% pass rate)")
    print(f"Failed:       {failed} symbols")
    print(f"Remaining:    {remaining} symbols")
    print(f"")
    print(f"Speed:        {60/avg_per_symbol:.1f} symbols/min")
    print(f"ETA:          {int(eta_minutes)} minutes ({eta_hours:.1f} hours)")
    print(f"Estimated:    {(datetime.now().hour + int(eta_hours)) % 24:02d}:{int((datetime.now().minute + (eta_minutes % 60)) % 60):02d}")
    print(f"{'='*60}\n")
else:
    print(f"\n{'='*60}")
    print(f"🔬 BACKTEST PROGRESS")
    print(f"{'='*60}")
    print(f"Progress:     {processed}/{total} symbols ({progress_pct:.1f}%)")
    print(f"Passed:       {passed} symbols")
    print(f"Failed:       {failed} symbols")
    print(f"{'='*60}\n")

# Check if complete
if processed >= total:
    print("✅ BACKTEST COMPLETE!")
    print(f"   Results: {passed} symbols passed validation")
    print(f"   Output: symbol_overrides_400.yaml")
