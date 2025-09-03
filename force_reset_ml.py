#!/usr/bin/env python3
"""
Force reset ML scorer singleton and clear the 7 false trades
"""
import os
import redis
import json

def force_reset_ml_scorer():
    """Force reset all ML scorer singletons"""
    
    # First, clear Redis storage if available
    try:
        redis_url = os.getenv('REDIS_URL')
        if redis_url:
            r = redis.from_url(redis_url, decode_responses=True)
            r.ping()
            
            # Get current data before clearing
            ml_data = r.get('ml_scorer_data')
            if ml_data:
                data = json.loads(ml_data)
                print(f"📊 Found {len(data.get('completed_trades', []))} trades in Redis")
            
            # Clear all ML-related keys
            r.delete('ml_scorer_data')
            r.delete('ml_scorer_trades')
            r.delete('ml_completed_trades')
            r.delete('ml_signals')
            print("✅ Cleared Redis ML storage")
        else:
            print("⚠️  No REDIS_URL found, skipping Redis clear")
    except Exception as e:
        print(f"⚠️  Redis clear failed: {e}")
    
    # Reset ml_ensemble_scorer singleton
    try:
        import ml_ensemble_scorer
        ml_ensemble_scorer._ml_scorer_instance = None
        print("✅ Reset ml_ensemble_scorer singleton")
    except:
        pass
    
    # Reset ml_signal_scorer singleton
    try:
        import ml_signal_scorer
        ml_signal_scorer._ml_scorer_instance = None
        print("✅ Reset ml_signal_scorer singleton")
    except:
        pass
    
    # Reset ml_signal_scorer_enhanced singleton
    try:
        import ml_signal_scorer_enhanced
        ml_signal_scorer_enhanced._ml_scorer_instance = None
        print("✅ Reset ml_signal_scorer_enhanced singleton")
    except:
        pass
    
    # Reset ml_ensemble_scorer_v2 singleton
    try:
        import ml_ensemble_scorer_v2
        ml_ensemble_scorer_v2._ml_scorer_instance = None
        print("✅ Reset ml_ensemble_scorer_v2 singleton")
    except:
        pass
    
    # Now create fresh instance and verify
    try:
        from ml_ensemble_scorer import get_ensemble_scorer
        scorer = get_ensemble_scorer(enabled=True, min_score=70.0)
        
        # Force reset the counter
        scorer.completed_trades_count = 0
        scorer.last_train_count = 0
        scorer.is_trained = False
        
        # Clear local storage
        if hasattr(scorer, 'local_storage'):
            scorer.local_storage = {'signals': [], 'completed_trades': []}
        
        # Clear the trades list if it exists
        if hasattr(scorer, 'trades'):
            scorer.trades = []
        
        print(f"\n✅ ML Scorer forcefully reset!")
        print(f"   Completed trades: {scorer.completed_trades_count}")
        print(f"   Is trained: {scorer.is_trained}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n✅ All ML scorer singletons cleared!")
    print("Now restart your bot and it should show 0 trades")

if __name__ == "__main__":
    force_reset_ml_scorer()
    
    print("\n" + "="*60)
    print("IMPORTANT: After running this script:")
    print("="*60)
    print("1. Stop your bot (Ctrl+C)")
    print("2. Run: python3 force_reset_ml.py (this script)")
    print("3. Start your bot: python3 live_bot.py")
    print("4. Check /ml - should now show 0 completed trades")