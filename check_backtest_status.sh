#!/bin/bash
# Quick status check for backtest

echo "=== Backtest Status Check ==="
echo ""

# Check if process is running
echo "1. Process Status:"
if ps aux | grep -i "backtest_pro_rules" | grep -v grep > /dev/null; then
    echo "   ✓ Script is RUNNING"
    ps aux | grep -i "backtest_pro_rules" | grep -v grep | awk '{print "   PID:", $2, "| CPU:", $3"%", "| Time:", $10}'
else
    echo "   ✗ Script is NOT running"
fi
echo ""

# Check results file
echo "2. Results File:"
if [ -f backtest_pro_rules_results.json ]; then
    SIZE=$(stat -f%z backtest_pro_rules_results.json 2>/dev/null || stat -c%s backtest_pro_rules_results.json 2>/dev/null)
    MODIFIED=$(stat -f%Sm backtest_pro_rules_results.json 2>/dev/null || stat -c%y backtest_pro_rules_results.json 2>/dev/null | cut -d' ' -f1-2)
    echo "   ✓ File exists: $SIZE bytes"
    echo "   Last modified: $MODIFIED"
    
    # Count completed symbols
    COUNT=$(python3 -c "import json; f=open('backtest_pro_rules_results.json'); d=json.load(f); print(len(d))" 2>/dev/null || echo "0")
    echo "   Symbols completed: $COUNT"
    
    # Show recent symbols
    echo "   Recent symbols:"
    python3 -c "import json; f=open('backtest_pro_rules_results.json'); d=json.load(f); [print(f\"     - {k}\") for k in list(d.keys())[-5:]]" 2>/dev/null || echo "     (could not parse)"
else
    echo "   ✗ No results file yet (still processing first symbol)"
fi
echo ""

# Check progress file
echo "3. Progress File:"
if [ -f backtest_pro_rules_progress.json ]; then
    echo "   ✓ Progress file exists:"
    cat backtest_pro_rules_progress.json | python3 -m json.tool 2>/dev/null || cat backtest_pro_rules_progress.json
else
    echo "   ✗ No progress file yet"
fi
echo ""

# Check for any log files
echo "4. Log Files:"
if [ -f backtest_output.log ]; then
    echo "   ✓ Log file exists"
    echo "   Last 3 lines:"
    tail -3 backtest_output.log 2>/dev/null | sed 's/^/     /'
else
    echo "   ✗ No log file found"
fi
echo ""

echo "=== End Status ==="

