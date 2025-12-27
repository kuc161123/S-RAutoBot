import asyncio
import logging
import threading
import os
import sys
import signal
import subprocess
from autobot.core.bot import Bot4H
from dashboard import app

# PID file location
PID_FILE = "/tmp/divergence_bot.pid"

def kill_existing_instances():
    """Kill any other running instances of this bot - AGGRESSIVE"""
    import time
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
                    # First try SIGTERM
                    os.kill(old_pid, signal.SIGTERM)
                    print(f"⚠️ Sent SIGTERM to previous instance (PID: {old_pid})")
                    time.sleep(2)
                    
                    # Check if still alive and force kill
                    try:
                        os.kill(old_pid, 0)  # Check if process exists
                        os.kill(old_pid, signal.SIGKILL)  # Force kill
                        print(f"💀 Force killed (SIGKILL) PID: {old_pid}")
                    except ProcessLookupError:
                        pass  # Already dead
                    
                    killed_count += 1
                except ProcessLookupError:
                    print(f"📝 Old PID {old_pid} already dead, cleaning up")
                except PermissionError:
                    print(f"⚠️ Cannot kill PID {old_pid} - permission denied")
        except (ValueError, FileNotFoundError):
            pass
    
    # Method 2: Kill any other python processes running main.py or VWAPBot
    kill_patterns = [
        "python.*main.py",
        "python3.*main.py",
        "DivergenceBot",
    ]
    
    for pattern in kill_patterns:
        try:
            result = subprocess.run(
                ["pgrep", "-f", pattern],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                pids = result.stdout.strip().split('\n')
                for pid_str in pids:
                    try:
                        pid = int(pid_str)
                        if pid != current_pid:
                            # Try SIGTERM first
                            os.kill(pid, signal.SIGTERM)
                            print(f"⚠️ Sent SIGTERM to (PID: {pid}) [{pattern}]")
                            time.sleep(1)
                            
                            # Force kill if still alive
                            try:
                                os.kill(pid, 0)
                                os.kill(pid, signal.SIGKILL)
                                print(f"💀 Force killed (SIGKILL) PID: {pid}")
                            except ProcessLookupError:
                                pass
                            
                            killed_count += 1
                    except (ValueError, ProcessLookupError, PermissionError):
                        pass
        except FileNotFoundError:
            pass  # pgrep not available
    
    # Write current PID to file
    with open(PID_FILE, 'w') as f:
        f.write(str(current_pid))
    
    if killed_count > 0:
        print(f"✅ Killed {killed_count} existing instance(s)")
        time.sleep(3)  # Wait for cleanup and Telegram to reset
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
        bot = DivergenceBot()
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        print("Bot stopped by user")
        cleanup_on_exit()
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        cleanup_on_exit()

