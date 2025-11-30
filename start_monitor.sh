#!/bin/bash
# Start the real-time monitor in a new terminal window

cd "$(dirname "$0")"

# Check if running on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    # Open in new Terminal window
    osascript -e "tell application \"Terminal\" to do script \"cd '$(pwd)' && python3 monitor_backtest.py\""
else
    # For Linux, run in current terminal
    python3 monitor_backtest.py
fi

