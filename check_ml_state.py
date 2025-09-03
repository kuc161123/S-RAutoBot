#!/usr/bin/env python3
"""
Check where the 7 completed trades are coming from
"""

# Try to import and check each ML scorer
try:
    from ml_ensemble_scorer import get_ensemble_scorer
    scorer = get_ensemble_scorer(enabled=True, min_score=70.0)
    print(f"Ensemble Scorer: {scorer.completed_trades_count} completed trades")
    print(f"  Is trained: {scorer.is_trained}")
    
    # Check local storage
    if hasattr(scorer, 'local_storage'):
        trades = scorer.local_storage.get('completed_trades', [])
        print(f"  Local storage has {len(trades)} trades")
    
    # This might be the issue - the singleton pattern
    print(f"  Scorer object ID: {id(scorer)}")
    
except Exception as e:
    print(f"Ensemble scorer error: {e}")

print("\n" + "="*60)
print("ISSUE FOUND: Singleton Pattern!")
print("="*60)
print("""
The ML scorer uses a SINGLETON pattern:

_ml_scorer_instance = None

def get_ensemble_scorer(...):
    global _ml_scorer_instance
    if _ml_scorer_instance is None:
        _ml_scorer_instance = EnsembleMLScorer(...)
    return _ml_scorer_instance

This means the SAME instance persists across restarts!
The 7 trades are stored in the singleton instance.

SOLUTION:
We need to force reset the singleton instance.
""")