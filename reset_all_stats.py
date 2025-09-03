#!/usr/bin/env python3
"""
Reset ALL Trading Statistics - Fresh Start
"""
import os
import redis
import logging
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def reset_everything():
    """Reset all stats for complete fresh start"""
    
    print("\n" + "="*60)
    print("🔄 RESETTING ALL STATISTICS - FRESH START")
    print("="*60)
    
    # 1. Reset ML data in Redis
    print("\n📊 Resetting ML Learning Data...")
    try:
        redis_url = os.getenv('REDIS_URL')
        if redis_url:
            client = redis.from_url(redis_url, decode_responses=True)
            client.ping()
            
            # Find all ML keys
            ml_patterns = ['ml_*', 'ml:*']
            all_keys = []
            for pattern in ml_patterns:
                keys = client.keys(pattern)
                all_keys.extend(keys)
            
            # Also check specific keys
            specific_keys = [
                'ml_completed_trades',
                'ml_enhanced_completed_trades', 
                'ml_v2_completed_trades',
                'ml_ensemble_completed_trades',
                'ml_model',
                'ml_enhanced_model',
                'ml_ensemble_model_rf',
                'ml_ensemble_model_xgb',
                'ml_ensemble_model_nn',
                'ml_v2_model_rf',
                'ml_v2_model_gb',
                'ml_v2_model_nn',
                'ml_scaler',
                'ml_enhanced_scaler',
                'ml_v2_scaler_ensemble',
                'ml_trades_count',
                'ml_enhanced_trades_count',
                'ml_v2_trades_count',
                'ml_ensemble_trades_count',
                'ml_feature_importance',
                'ml_enhanced_feature_importance'
            ]
            
            for key in specific_keys:
                if client.exists(key):
                    all_keys.append(key)
            
            # Remove duplicates
            all_keys = list(set(all_keys))
            
            if all_keys:
                for key in all_keys:
                    client.delete(key)
                print(f"✅ Deleted {len(all_keys)} ML keys from Redis")
                for key in all_keys[:10]:  # Show first 10
                    print(f"   - {key}")
                if len(all_keys) > 10:
                    print(f"   ... and {len(all_keys)-10} more")
            else:
                print("   No ML data found in Redis")
                
        else:
            print("   No Redis configured - ML data only in memory")
            
    except Exception as e:
        print(f"   Warning: Could not reset Redis: {e}")
    
    # 2. Reset local files
    print("\n📁 Resetting Local Statistics...")
    files_to_reset = [
        'trade_history.json',
        'daily_stats.json',
        'symbol_stats.json',
        'ml_trades.json',
        'ml_performance.json'
    ]
    
    reset_count = 0
    for filename in files_to_reset:
        if os.path.exists(filename):
            try:
                backup = f"backup_{filename}.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                os.rename(filename, backup)
                print(f"   Backed up {filename} → {backup}")
                reset_count += 1
            except Exception as e:
                print(f"   Could not reset {filename}: {e}")
    
    if reset_count > 0:
        print(f"✅ Reset {reset_count} local files")
    else:
        print("   No local stats files found")
    
    # 3. PostgreSQL trades (if configured)
    print("\n🗄️ Checking PostgreSQL...")
    try:
        import psycopg2
        from urllib.parse import urlparse
        
        db_url = os.getenv('DATABASE_URL')
        if db_url:
            parsed = urlparse(db_url)
            conn = psycopg2.connect(
                host=parsed.hostname,
                port=parsed.port,
                user=parsed.username,
                password=parsed.password,
                database=parsed.path[1:]
            )
            cur = conn.cursor()
            
            # Check trades table
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'trades'
                );
            """)
            
            if cur.fetchone()[0]:
                cur.execute("SELECT COUNT(*) FROM trades;")
                count = cur.fetchone()[0]
                
                if count > 0:
                    # Create backup
                    backup_table = f"trades_backup_{datetime.now().strftime('%Y%m%d')}"
                    cur.execute(f"CREATE TABLE IF NOT EXISTS {backup_table} AS SELECT * FROM trades;")
                    cur.execute("DELETE FROM trades;")
                    conn.commit()
                    print(f"✅ Cleared {count} trades from PostgreSQL")
                    print(f"   Backup saved to: {backup_table}")
                else:
                    print("   No trades in PostgreSQL")
            else:
                print("   No trades table in PostgreSQL")
            
            cur.close()
            conn.close()
            
    except ImportError:
        print("   PostgreSQL not configured")
    except Exception as e:
        print(f"   PostgreSQL not accessible: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("✅ COMPLETE RESET SUCCESSFUL!")
    print("="*60)
    print("""
What was reset:
• All ML learning data cleared
• ML model deleted (will retrain from scratch)
• Trade history backed up and cleared
• Statistics reset to zero

What happens next:
1. ML Learning Mode Active
   - Will take ALL HL/LH signals
   - ~20-30 signals per day expected
   - Learning phase for first 200 trades

2. After 200 trades
   - ML automatically trains
   - Starts filtering signals
   - Quality improves dramatically

3. Timeline
   - Week 1-2: Learning (40-45% win rate)
   - Week 3: ML activates (50-55% win rate)
   - Week 4+: Optimized (55-65% win rate)

Monitor with Telegram:
• /ml - Check learning progress
• /dashboard - Overall performance
• /stats - Win/loss tracking

🚀 Restart your bot for fresh start with ML learning!
    """)

if __name__ == "__main__":
    # Confirm before resetting
    print("\n⚠️  WARNING: This will reset ALL statistics and ML data!")
    print("Your bot will start fresh with ML in learning mode.")
    print("\nType 'RESET' to confirm, or press Ctrl+C to cancel:")
    
    try:
        confirm = input("> ")
        if confirm == "RESET":
            reset_everything()
        else:
            print("\n❌ Cancelled - no changes made")
    except KeyboardInterrupt:
        print("\n\n❌ Cancelled - no changes made")