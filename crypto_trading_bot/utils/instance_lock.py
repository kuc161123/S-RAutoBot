"""
Instance lock to prevent multiple bot instances
"""
import os
import time
import psutil
import structlog

logger = structlog.get_logger(__name__)

class InstanceLock:
    """Manages single instance enforcement"""
    
    @staticmethod
    def kill_other_instances():
        """Kill other Python processes running the bot"""
        current_pid = os.getpid()
        killed_count = 0
        
        try:
            # Look for other Python processes
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    # Skip current process
                    if proc.info['pid'] == current_pid:
                        continue
                    
                    # Check if it's a Python process running our bot
                    if proc.info['name'] and 'python' in proc.info['name'].lower():
                        cmdline = proc.info.get('cmdline', [])
                        if cmdline and any('main.py' in str(arg) or 'crypto_trading_bot' in str(arg) for arg in cmdline):
                            logger.warning(f"Killing conflicting bot instance: PID {proc.info['pid']}")
                            proc.terminate()
                            killed_count += 1
                            time.sleep(1)  # Give it time to terminate
                            
                            # Force kill if still running
                            if proc.is_running():
                                proc.kill()
                                
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                    
            if killed_count > 0:
                logger.info(f"Terminated {killed_count} conflicting bot instances")
                time.sleep(3)  # Give time for ports to be released
                
        except Exception as e:
            logger.error(f"Error checking for other instances: {e}")
    
    @staticmethod
    def create_lock_file():
        """Create a lock file with current PID"""
        lock_file = '/tmp/crypto_bot.lock'
        
        try:
            # Check if lock file exists
            if os.path.exists(lock_file):
                with open(lock_file, 'r') as f:
                    old_pid = int(f.read().strip())
                
                # Check if old process is still running
                try:
                    old_proc = psutil.Process(old_pid)
                    if old_proc.is_running():
                        logger.warning(f"Another instance is running (PID: {old_pid}). Terminating it...")
                        old_proc.terminate()
                        time.sleep(2)
                        if old_proc.is_running():
                            old_proc.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            # Write our PID
            with open(lock_file, 'w') as f:
                f.write(str(os.getpid()))
            
            logger.info(f"Lock file created: {lock_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create lock file: {e}")
            return False
    
    @staticmethod
    def remove_lock_file():
        """Remove lock file on exit"""
        lock_file = '/tmp/crypto_bot.lock'
        try:
            if os.path.exists(lock_file):
                os.remove(lock_file)
                logger.info("Lock file removed")
        except Exception as e:
            logger.error(f"Failed to remove lock file: {e}")