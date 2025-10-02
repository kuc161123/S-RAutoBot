#!/usr/bin/env python3
"""
Startup script that ensures only one bot instance runs
"""
import os
import sys
import subprocess
import time
import signal
import redis # New import

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

def check_for_trained_models(redis_client) -> bool:
    """Checks if pre-trained models exist in Redis."""
    try:
        # Check for Pullback model
        if not redis_client.exists('iml:models'):
            return False
        # Check for Mean Reversion model (future, but prepare for it)
        if not redis_client.exists('ml:model:mean_reversion'): # This key will be used by the new scorer
            return False
        return True
    except Exception as e:
        print(f"Warning: Could not check Redis for models: {e}")
        return False

if __name__ == "__main__":
    print("Starting trading bot...")
    print("Checking for existing instances...")

    # Kill any existing instances
    kill_existing_bots()

    # Wait a moment
    time.sleep(2)

    # Connect to Redis
    redis_url = os.getenv('REDIS_URL')
    r = None
    if redis_url:
        try:
            r = redis.from_url(redis_url)
            r.ping()
            print("Connected to Redis.")
        except Exception as e:
            print(f"Warning: Could not connect to Redis: {e}. Models will not be checked/saved.")
            r = None

    # Check if models need training
    should_train = False
    if r and not check_for_trained_models(r):
        print("Pre-trained models not found in Redis.")
        should_train = True
    elif os.getenv('FORCE_ML_TRAIN', 'false').lower() == 'true':
        print("FORCE_ML_TRAIN environment variable is true. Retraining models.")
        should_train = True
    else:
        print("Pre-trained models found in Redis. Skipping offline training.")

    if should_train:
        print("Running bootstrap pretrainer (candles-only) to build ML models...")
        try:
            # Use the candle-only bootstrap to generate samples and train scalers/models
            subprocess.run([sys.executable, "bootstrap_pretrain.py"], check=True)
            print("Bootstrap pretraining complete. Models saved if Redis available.")
        except subprocess.CalledProcessError as e:
            print(f"Error: Bootstrap pretraining failed with exit code {e.returncode}. Live bot will start in online learning mode.")
        except Exception as e:
            print(f"Error running bootstrap pretrainer: {e}. Live bot will start in online learning mode.")

    print("Launching live bot...")
    os.execvp(sys.executable, [sys.executable, "live_bot.py"])
