import asyncio
import logging
import threading
import os
import sys
import signal
import subprocess
from autobot.core.bot import VWAPBot
from dashboard import app

# PID file location
PID_FILE = "/tmp/vwap_bot.pid"

def kill_existing_instances():
    """Kill any other running instances of this bot"""
    current_pid = os.getpid()
    killed_count = 0
    
    print("🔍 Checking for existing bot instances...")
    
    # Method 1: Check PID file
    if os.path.exists(PID_FILE):
        try:
            with open(PID_FILE, 'r') as f:
                old_pid = int(f.read().strip())
            
            if old_pid != current_pid:
                try:
                    os.kill(old_pid, signal.SIGTERM)
                    print(f"⚠️ Killed previous instance (PID: {old_pid})")
                    killed_count += 1
                    # Wait a moment for it to die
                    import time
                    time.sleep(2)
                except ProcessLookupError:
                    print(f"📝 Old PID {old_pid} already dead, cleaning up")
                except PermissionError:
                    print(f"⚠️ Cannot kill PID {old_pid} - permission denied")
        except (ValueError, FileNotFoundError):
            pass
    
    # Method 2: Kill any other python processes running main.py (Linux/Mac)
    try:
        # Find all python processes running main.py
        result = subprocess.run(
            ["pgrep", "-f", "python.*main.py"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            pids = result.stdout.strip().split('\n')
            for pid_str in pids:
                try:
                    pid = int(pid_str)
                    if pid != current_pid:
                        os.kill(pid, signal.SIGTERM)
                        print(f"⚠️ Killed duplicate instance (PID: {pid})")
                        killed_count += 1
                except (ValueError, ProcessLookupError, PermissionError):
                    pass
    except FileNotFoundError:
        # pgrep not available (Windows)
        pass
    
    # Write current PID to file
    with open(PID_FILE, 'w') as f:
        f.write(str(current_pid))
    
    if killed_count > 0:
        print(f"✅ Killed {killed_count} existing instance(s)")
        import time
        time.sleep(1)  # Wait for cleanup
    else:
        print("✅ No existing instances found")
    
    print(f"📝 Current PID: {current_pid}")

def cleanup_on_exit():
    """Clean up PID file on exit"""
    try:
        if os.path.exists(PID_FILE):
            os.remove(PID_FILE)
    except:
        pass

def run_dashboard():
    """Run the web dashboard in a separate thread"""
    try:
        print("📊 Dashboard starting at http://localhost:8888")
        app.run(host='0.0.0.0', port=8888, debug=False, use_reloader=False)
    except Exception as e:
        logging.error(f"Dashboard failed to start: {e}")

if __name__ == "__main__":
    try:
        # FIRST: Kill any existing instances
        kill_existing_instances()
        
        # Register cleanup handler
        import atexit
        atexit.register(cleanup_on_exit)
        
        # Start dashboard in background thread
        dash_thread = threading.Thread(target=run_dashboard, daemon=True)
        dash_thread.start()
        
        # Start main bot
        bot = VWAPBot()
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        print("Bot stopped by user")
        cleanup_on_exit()
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        cleanup_on_exit()

