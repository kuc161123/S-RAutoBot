#!/usr/bin/env python3
"""
Startup script that ensures only one bot instance runs
"""
import os
import sys
import subprocess
import time
import signal

def kill_existing_bots():
    """Kill any existing bot processes"""
    try:
        # Find python processes running live_bot.py
        result = subprocess.run(
            ["pgrep", "-f", "live_bot.py"],
            capture_output=True,
            text=True
        )
        
        if result.stdout:
            pids = result.stdout.strip().split('\n')
            current_pid = str(os.getpid())
            
            for pid in pids:
                if pid and pid != current_pid:
                    try:
                        os.kill(int(pid), signal.SIGTERM)
                        print(f"Killed existing bot process: {pid}")
                        time.sleep(1)
                    except:
                        pass
    except:
        pass

if __name__ == "__main__":
    print("Starting trading bot...")
    print("Checking for existing instances...")
    
    # Kill any existing instances
    kill_existing_bots()
    
    # Wait a moment
    time.sleep(2)
    
    # Start the bot
    print("Launching bot...")
    os.execvp(sys.executable, [sys.executable, "live_bot.py"])