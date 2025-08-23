#!/usr/bin/env python3
"""
Testing Mode Startup Script
Runs the bot in safe testing mode with $100 capital
Focuses on ML training with minimal risk
"""
import asyncio
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from crypto_trading_bot.src.utils.testing_monitor import testing_monitor
from crypto_trading_bot.src.config import settings

async def main():
    print("=" * 60)
    print("CRYPTO TRADING BOT - TESTING MODE (ALL SYMBOLS)")
    print("=" * 60)
    print(f"Initial Balance: $100")
    print(f"Risk Per Trade: {settings.default_risk_percent}% ($0.30 per trade)")
    print(f"Max Leverage: {settings.default_leverage}x")
    print(f"Max Positions: {settings.max_concurrent_positions}")
    print(f"Symbols: ALL 300+ BYBIT FUTURES PAIRS")
    print(f"Strategy: Multi-timeframe Supply/Demand with ML Learning")
    print("=" * 60)
    print()
    
    # Confirmation
    confirm = input("⚠️  TESTING MODE: This will trade with REAL money ($100). Continue? (yes/no): ")
    if confirm.lower() != 'yes':
        print("Testing cancelled.")
        return
    
    print("\n✅ Starting bot in TESTING MODE...")
    print("📊 Focus: Collecting data and training ML model")
    print("🛡️  Safety: Minimal risk, tight stops, low leverage")
    print()
    
    # Import after confirmation
    from crypto_trading_bot.src.main import main as run_bot
    
    # Set testing environment variables
    os.environ['TRADING_MODE'] = 'TESTING'
    os.environ['MAX_RISK_USD'] = '10'  # Maximum $10 loss
    os.environ['ENABLE_ML_LEARNING'] = 'true'
    
    try:
        # Start monitoring task
        async def monitor_testing():
            while True:
                await asyncio.sleep(300)  # Every 5 minutes
                
                # Check if we should continue
                can_trade, reason = testing_monitor.should_continue_trading()
                if not can_trade:
                    print(f"\n⛔ STOPPING: {reason}")
                    print(await testing_monitor.generate_report())
                    sys.exit(0)
                
                # Print summary every 5 minutes
                summary = testing_monitor.get_summary()
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Testing Update:")
                print(f"  Phase: {summary['phase']}")
                print(f"  Balance: {summary['balance']}")
                print(f"  Trades: {summary['total_trades']}")
                print(f"  Win Rate: {summary['win_rate']}")
                print(f"  PnL: {summary['total_pnl_usd']} ({summary['total_pnl_percent']})")
                print(f"  Risk Level: {summary['risk_level']}x")
                
        # Start monitor
        monitor_task = asyncio.create_task(monitor_testing())
        
        # Run bot
        await run_bot()
        
    except KeyboardInterrupt:
        print("\n\nShutting down testing mode...")
        print(await testing_monitor.generate_report())
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print(await testing_monitor.generate_report())
    finally:
        # Save testing results
        with open(f"testing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", "w") as f:
            f.write(await testing_monitor.generate_report())
        print("\n📄 Testing report saved.")

if __name__ == "__main__":
    asyncio.run(main())