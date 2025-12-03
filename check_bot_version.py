#!/usr/bin/env python3
"""
Diagnostic script to verify the bot code version
Run this on your server to confirm the notification logic is updated
"""

import sys
import inspect

# Add the parent directory to path
sys.path.insert(0, '/app')

try:
    from autobot.core.bot import PhantomTracker
    
    # Get the signature of record_phantom
    sig = inspect.signature(PhantomTracker.record_phantom)
    params = list(sig.parameters.keys())
    
    print("=" * 60)
    print("PHANTOM TRACKER DIAGNOSTIC")
    print("=" * 60)
    
    print(f"\nrecord_phantom parameters: {params}")
    
    if 'allowed_combos_long' in params and 'allowed_combos_short' in params:
        print("\n✅ CORRECT: Bot has the NEW notification code")
        print("   (Shows allowed combos for both LONG and SHORT)")
    elif 'allowed_combos' in params:
        print("\n⚠️  PARTIAL: Bot has the intermediate code")
        print("   (Shows allowed combos but only for rejected side)")
    else:
        print("\n❌ OLD: Bot has the OLD notification code")
        print("   (No allowed combos shown)")
        
    print("\n" + "=" * 60)
    
except Exception as e:
    print(f"Error: {e}")
    print("\nMake sure you're running this from the bot directory!")
