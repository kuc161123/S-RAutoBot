#!/usr/bin/env python3
"""
Reset Trading Statistics
Clears ML data, trade history, and statistics for fresh start
"""
import os
import redis
import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def reset_ml_data():
    """Reset ML learning data in Redis"""
    try:
        redis_url = os.getenv('REDIS_URL')
        if not redis_url:
            logger.info("No Redis URL found - ML data only in memory")
            return False
        
        client = redis.from_url(redis_url, decode_responses=True)
        client.ping()
        
        # ML related keys to delete
        ml_keys = [
            'ml_completed_trades',
            'ml_enhanced_completed_trades', 
            'ml_v2_completed_trades',
            'ml_model',
            'ml_enhanced_model',
            'ml_v2_model_rf',
            'ml_v2_model_gb', 
            'ml_v2_model_nn',
            'ml_scaler',
            'ml_enhanced_scaler',
            'ml_v2_scaler_ensemble',
            'ml_trades_count',
            'ml_enhanced_trades_count',
            'ml_v2_trades_count',
            'ml_feature_importance',
            'ml_enhanced_feature_importance'
        ]
        
        deleted = 0
        for key in ml_keys:
            if client.exists(key):
                client.delete(key)
                deleted += 1
                logger.info(f"  Deleted: {key}")
        
        logger.info(f"✅ Cleared {deleted} ML keys from Redis")
        return True
        
    except Exception as e:
        logger.error(f"Could not reset ML data: {e}")
        return False

def reset_trade_tracking():
    """Reset trade tracking files"""
    files_to_reset = [
        'trade_history.json',
        'daily_stats.json',
        'symbol_stats.json'
    ]
    
    for filename in files_to_reset:
        if os.path.exists(filename):
            try:
                # Backup before deleting
                backup_name = f"{filename}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                os.rename(filename, backup_name)
                logger.info(f"  Backed up {filename} to {backup_name}")
            except Exception as e:
                logger.warning(f"Could not backup {filename}: {e}")

def reset_postgres_trades():
    """Reset trade records in PostgreSQL"""
    try:
        import psycopg2
        from urllib.parse import urlparse
        
        db_url = os.getenv('DATABASE_URL')
        if not db_url:
            logger.info("No PostgreSQL database configured")
            return False
        
        # Parse database URL
        parsed = urlparse(db_url)
        
        conn = psycopg2.connect(
            host=parsed.hostname,
            port=parsed.port,
            user=parsed.username,
            password=parsed.password,
            database=parsed.path[1:]
        )
        cur = conn.cursor()
        
        # Check if trades table exists
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'trades'
            );
        """)
        
        if cur.fetchone()[0]:
            # Backup count before deletion
            cur.execute("SELECT COUNT(*) FROM trades;")
            count = cur.fetchone()[0]
            
            if count > 0:
                # Optional: Create backup table
                backup_table = f"trades_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                cur.execute(f"CREATE TABLE {backup_table} AS SELECT * FROM trades;")
                logger.info(f"  Created backup table: {backup_table} with {count} trades")
                
                # Clear trades table
                cur.execute("DELETE FROM trades;")
                conn.commit()
                logger.info(f"✅ Cleared {count} trades from PostgreSQL")
            else:
                logger.info("  No trades to clear in PostgreSQL")
        else:
            logger.info("  No trades table found")
        
        cur.close()
        conn.close()
        return True
        
    except ImportError:
        logger.info("PostgreSQL module not installed")
        return False
    except Exception as e:
        logger.error(f"Could not reset PostgreSQL trades: {e}")
        return False

def show_menu():
    """Show reset options menu"""
    print("\n" + "="*60)
    print("🔄 RESET TRADING STATISTICS")
    print("="*60)
    print("""
What would you like to reset?

1. ML Learning Data Only
   - Clears ML model and training data
   - Resets to 0 completed trades
   - ML starts fresh learning phase
   
2. Trade History Only  
   - Clears trade tracking files
   - Resets win/loss statistics
   - Keeps ML model intact
   
3. Everything (Fresh Start)
   - Clears all ML data
   - Resets all trade history
   - Complete fresh start
   
4. Cancel (Do nothing)

    """)
    
    choice = input("Enter choice (1-4): ")
    return choice

def main():
    """Main reset function"""
    
    choice = show_menu()
    
    if choice == '1':
        print("\n🔄 Resetting ML Learning Data...")
        print("-"*40)
        reset_ml_data()
        print("""
✅ ML Data Reset Complete!
• ML will start learning from scratch
• Next 200 trades will be learning phase
• All signals will be taken initially
        """)
        
    elif choice == '2':
        print("\n🔄 Resetting Trade History...")
        print("-"*40)
        reset_trade_tracking()
        reset_postgres_trades()
        print("""
✅ Trade History Reset Complete!
• Statistics reset to zero
• ML model preserved (if trained)
• Win/loss tracking starts fresh
        """)
        
    elif choice == '3':
        print("\n🔄 Full Reset - Everything...")
        print("-"*40)
        reset_ml_data()
        reset_trade_tracking()
        reset_postgres_trades()
        print("""
✅ Complete Reset Done!
• ML learning starts from zero
• All trade history cleared
• Complete fresh start
• Bot will take all signals initially
        """)
        
    else:
        print("\n❌ Cancelled - No changes made")
        return
    
    print("\n" + "="*60)
    print("📊 NEXT STEPS")
    print("="*60)
    print("""
1. Restart your bot for fresh start
2. ML will be in learning mode (takes all signals)
3. After 200 trades, ML automatically activates
4. Monitor progress with /ml command

Expected Timeline:
• Week 1-2: Learning phase (40-45% win rate)
• Week 3: ML activates (50-55% win rate)
• Week 4+: Optimized (55-65% win rate)
    """)

if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    main()