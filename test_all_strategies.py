#!/usr/bin/env python3
"""
Comprehensive test for all strategy components
Tests each strategy type with multiple symbols
"""
import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crypto_trading_bot.src.strategy.hybrid_smart_money_strategy import HybridSmartMoneyStrategy, SignalType
from crypto_trading_bot.src.strategy.ml_signal_scorer import ml_signal_scorer
from crypto_trading_bot.src.api.enhanced_bybit_client import EnhancedBybitClient
import pandas as pd
import numpy as np

async def test_all_strategies():
    """Test all strategy components comprehensively"""
    
    print("=" * 80)
    print("COMPREHENSIVE STRATEGY TEST - ALL COMPONENTS")
    print("=" * 80)
    
    # Initialize
    strategy = HybridSmartMoneyStrategy()
    client = EnhancedBybitClient()
    
    try:
        await client.initialize()
        print("✅ Client initialized\n")
        
        # Extended symbol list for better coverage
        test_symbols = [
            "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
            "ADAUSDT", "AVAXUSDT", "DOGEUSDT", "MATICUSDT", "LINKUSDT",
            "DOTUSDT", "UNIUSDT", "LTCUSDT", "ATOMUSDT", "NEARUSDT"
        ]
        
        # Track signals by type
        signals_by_type = {
            SignalType.ORDER_BLOCK: [],
            SignalType.FAIR_VALUE_GAP: [],
            SignalType.MEAN_REVERSION: [],
            SignalType.VWAP_BREAKOUT: [],
            SignalType.LIQUIDITY_SWEEP: []
        }
        
        print(f"Testing {len(test_symbols)} symbols...")
        print("-" * 80)
        
        for symbol in test_symbols:
            print(f"\n📊 {symbol}:", end=" ")
            
            # Get data
            df = await client.get_klines(symbol, '15', limit=200)
            
            if df is not None and len(df) > 100:
                # Generate signals
                signals = strategy.analyze(symbol, df)
                
                if signals:
                    # Score with ML
                    scored = ml_signal_scorer.score_signals(signals, {symbol: df})
                    
                    if scored:
                        print(f"✅ {len(scored)} signals")
                        for sig in scored:
                            signals_by_type[sig.signal_type].append(sig)
                            print(f"   • {sig.signal_type.value}: {sig.direction} "
                                  f"@ {sig.entry_price:.4f} (conf: {sig.confidence:.0f}%)")
                    else:
                        print("⚠️  Signals filtered")
                else:
                    print("📉 No signals")
            else:
                print("❌ Insufficient data")
        
        # Summary
        print("\n" + "=" * 80)
        print("SIGNAL TYPE BREAKDOWN")
        print("=" * 80)
        
        total_signals = 0
        for sig_type, sigs in signals_by_type.items():
            if sigs:
                avg_conf = sum(s.confidence for s in sigs) / len(sigs)
                print(f"\n{sig_type.value.upper()}:")
                print(f"  • Count: {len(sigs)}")
                print(f"  • Avg Confidence: {avg_conf:.1f}%")
                print(f"  • Symbols: {', '.join(set(s.symbol for s in sigs))}")
                total_signals += len(sigs)
        
        print(f"\n{'='*80}")
        print(f"TOTAL SIGNALS GENERATED: {total_signals}")
        
        if total_signals == 0:
            print("\n⚠️  No signals generated. Possible reasons:")
            print("  1. Market conditions don't match strategy criteria")
            print("  2. Parameters may be too restrictive")
            print("  3. Need more volatile market conditions")
        else:
            print(f"\n✅ Success Rate: {(total_signals/len(test_symbols)*100):.0f}% of symbols generated signals")
            
            # Find best signal
            all_signals = []
            for sigs in signals_by_type.values():
                all_signals.extend(sigs)
            
            if all_signals:
                best = max(all_signals, key=lambda x: x.confidence)
                print(f"\n🏆 BEST SIGNAL:")
                print(f"  • {best.symbol} {best.signal_type.value}")
                print(f"  • Direction: {best.direction}")
                print(f"  • Entry: {best.entry_price:.4f}")
                print(f"  • Stop Loss: {best.stop_loss:.4f}")
                print(f"  • Take Profit: {best.take_profit_1:.4f}")
                print(f"  • Confidence: {best.confidence:.1f}%")
                print(f"  • Risk/Reward: {best.risk_reward_ratio:.2f}")
        
        # Test mean reversion specifically (should work in ranging markets)
        print(f"\n{'='*80}")
        print("TESTING MEAN REVERSION SPECIFICALLY")
        print("=" * 80)
        
        # Find a ranging market
        for symbol in test_symbols[:5]:
            df = await client.get_klines(symbol, '15', limit=200)
            if df is not None and len(df) > 100:
                # Check if market is ranging
                regime = strategy._detect_market_regime(df)
                print(f"{symbol}: Market regime = {regime.value}")
        
        print("\n✅ All tests completed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await client.close()
        print("\n🏁 Test finished")

if __name__ == "__main__":
    asyncio.run(test_all_strategies())