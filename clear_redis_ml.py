#!/usr/bin/env python3
"""
Clear Redis ML data storage to remove the 7 false trades
"""
import redis
import json
import os

def clear_redis_ml_data():
    """Clear all ML-related data from Redis"""
    try:
        # Connect to Redis using REDIS_URL from environment
        redis_url = os.getenv('REDIS_URL')
        if redis_url:
            r = redis.from_url(redis_url, decode_responses=True)
        else:
            # Try local Redis
            r = redis.Redis(host='localhost', port=6379, decode_responses=True, db=0)
        
        # Check if Redis is running
        r.ping()
        print("✅ Connected to Redis")
        
        # Get current data before clearing
        ml_data = r.get('ml_scorer_data')
        if ml_data:
            data = json.loads(ml_data)
            print(f"\n📊 Current Redis ML data:")
            print(f"   Completed trades: {len(data.get('completed_trades', []))}")
            print(f"   Signals: {len(data.get('signals', []))}")
            
            # Show the false trades
            if data.get('completed_trades'):
                print(f"\n🔍 Found {len(data['completed_trades'])} trades in Redis:")
                for i, trade in enumerate(data['completed_trades'], 1):
                    print(f"   {i}. {trade.get('symbol', 'Unknown')} - {trade.get('outcome', 'Unknown')}")
        
        # Clear ML scorer data
        print("\n🗑️  Clearing Redis ML data...")
        r.delete('ml_scorer_data')
        r.delete('ml_scorer_trades')  # In case it uses different keys
        r.delete('ml_completed_trades')
        r.delete('ml_signals')
        
        # Verify it's cleared
        ml_data = r.get('ml_scorer_data')
        if ml_data is None:
            print("✅ Redis ML data successfully cleared!")
        else:
            print("⚠️  Redis data may not be fully cleared")
            
    except redis.ConnectionError:
        print("❌ Redis is not running or not accessible")
        print("   The ML data might be stored in memory only")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("="*60)
    print("CLEARING REDIS ML DATA")
    print("="*60)
    
    clear_redis_ml_data()
    
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("1. Stop your bot (Ctrl+C)")
    print("2. Run: python3 clear_redis_ml.py")
    print("3. Run: python3 force_reset_ml.py")
    print("4. Start your bot: python3 live_bot.py")
    print("5. Check /ml - should now show 0 completed trades")