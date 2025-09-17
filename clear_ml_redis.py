#!/usr/bin/env python3
"""
Safely clear ML-related data from Redis while preserving other data.
Only clears: iml:*, phantom:*, ml:trades:*, symbol:collector:*
Preserves: positions, bot state, and other non-ML data
"""

import os
import sys
import redis
from typing import List, Dict
from collections import defaultdict

def connect_redis():
    """Connect to Redis using REDIS_URL from environment."""
    redis_url = os.environ.get('REDIS_URL')
    if not redis_url:
        print("❌ REDIS_URL environment variable not set")
        print("Please set REDIS_URL or create a .env file with REDIS_URL=redis://...")
        sys.exit(1)
    
    try:
        r = redis.from_url(redis_url, decode_responses=True)
        r.ping()  # Test connection
        print(f"✅ Connected to Redis")
        return r
    except Exception as e:
        print(f"❌ Failed to connect to Redis: {e}")
        sys.exit(1)

def scan_ml_keys(r: redis.Redis) -> Dict[str, List[str]]:
    """Scan for ML-related keys and categorize them."""
    ml_patterns = [
        'iml:*',           # Immediate ML scorer data
        'phantom:*',       # Phantom trades
        'ml:trades:*',     # ML trade data
        'symbol:collector:*'  # Symbol collection data
    ]
    
    found_keys = defaultdict(list)
    
    print("\n🔍 Scanning for ML-related keys...")
    
    for pattern in ml_patterns:
        # Use SCAN to find matching keys (safer than KEYS for production)
        cursor = 0
        pattern_clean = pattern.rstrip('*')
        
        while True:
            cursor, keys = r.scan(cursor, match=pattern, count=100)
            for key in keys:
                found_keys[pattern_clean].append(key)
            
            if cursor == 0:
                break
    
    return dict(found_keys)

def display_summary(found_keys: Dict[str, List[str]]):
    """Display summary of found keys."""
    print("\n📊 Summary of ML-related keys found:")
    print("-" * 50)
    
    total_keys = 0
    
    for pattern, keys in found_keys.items():
        count = len(keys)
        total_keys += count
        print(f"{pattern}: {count} keys")
        
        # Show a few examples if any exist
        if count > 0:
            examples = keys[:3]
            for example in examples:
                print(f"  └─ {example}")
            if count > 3:
                print(f"  └─ ... and {count - 3} more")
    
    print("-" * 50)
    print(f"Total ML keys to delete: {total_keys}")
    
    return total_keys

def show_preserved_keys(r: redis.Redis):
    """Show examples of keys that will be preserved."""
    print("\n🔒 Examples of keys that will be PRESERVED:")
    
    preserved_patterns = [
        'position:*',
        'bot:*',
        'user:*',
        'config:*',
        'state:*'
    ]
    
    for pattern in preserved_patterns:
        # Check if any keys match this pattern
        cursor, sample_keys = r.scan(0, match=pattern, count=5)
        if sample_keys:
            print(f"  ✓ {pattern} (found {len(sample_keys)} keys)")

def delete_ml_keys(r: redis.Redis, found_keys: Dict[str, List[str]]):
    """Delete all ML-related keys."""
    total_deleted = 0
    
    print("\n🗑️  Deleting ML-related keys...")
    
    for pattern, keys in found_keys.items():
        if not keys:
            continue
            
        print(f"\nDeleting {pattern} keys...")
        
        # Delete in batches for efficiency
        batch_size = 100
        for i in range(0, len(keys), batch_size):
            batch = keys[i:i + batch_size]
            deleted = r.delete(*batch)
            total_deleted += deleted
            
            # Show progress
            progress = min(i + batch_size, len(keys))
            print(f"  Deleted {progress}/{len(keys)} keys...", end='\r')
        
        print(f"  ✓ Deleted all {len(keys)} {pattern} keys")
    
    return total_deleted

def main():
    """Main function to clear ML Redis data."""
    print("🤖 Redis ML Data Cleaner")
    print("=" * 50)
    
    # Load .env if available
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # dotenv not installed, rely on environment variables
    
    # Connect to Redis
    r = connect_redis()
    
    # Scan for ML keys
    found_keys = scan_ml_keys(r)
    
    # Display summary
    total_keys = display_summary(found_keys)
    
    if total_keys == 0:
        print("\n✨ No ML-related keys found. Redis is already clean!")
        return
    
    # Show what will be preserved
    show_preserved_keys(r)
    
    # Ask for confirmation
    print("\n⚠️  WARNING: This will permanently delete all ML-related data!")
    print("Position data and bot state will be preserved.")
    
    confirmation = input("\nType 'DELETE' to confirm deletion: ")
    
    if confirmation != 'DELETE':
        print("❌ Deletion cancelled.")
        return
    
    # Delete the keys
    total_deleted = delete_ml_keys(r, found_keys)
    
    print(f"\n✅ Successfully deleted {total_deleted} ML-related keys")
    print("📊 Position data and bot state have been preserved")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Operation cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)