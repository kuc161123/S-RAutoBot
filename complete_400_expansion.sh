#!/bin/bash
# Auto-complete script - runs when backtest finishes

echo "🎯 Completing 400-Symbol Expansion..."
echo ""

# 1. Check if backtest completed
if [ ! -f "symbol_overrides_400.yaml" ]; then
    echo "❌ Backtest not complete yet (symbol_overrides_400.yaml not found)"
    exit 1
fi

# 2. Show results summary
echo "✅ Backtest complete! Results:"
passed=$(grep -c "^[A-Z0-9].*:$" symbol_overrides_400.yaml)
echo "   - Symbols passed: $passed"
echo ""

# 3. Update bot with new results
echo "📝 Updating bot configuration..."
cp symbol_overrides_400.yaml symbol_overrides.yaml
echo "   - Updated symbol_overrides.yaml"
echo ""

# 4. Commit changes
echo "💾 Committing changes..."
git add config.yaml symbol_overrides.yaml symbols_400.yaml backtest_400_symbols.py fetch_symbols.py update_config_symbols.py 400_SYMBOL_EXPANSION.md
git commit -m "Expand to 400 tradeable symbols with deep backtest validation

- Fetched 546 tradeable USDT perpetuals from Bybit
- Selected 400 symbols for trading
- Updated config.yaml with new symbol list
- Ran ultra-deep backtest (10k candles, WR>60%, N>=30)
- $passed symbols passed validation with side-specific combos
- Bot configured for backtest-only execution mode

Backtest Parameters:
- Data: ~10,000 candles per symbol
- Filter: WR > 60%, N >= 30 (side-specific)
- Analysis: Separate LONG/SHORT strategies
- Quality: High-probability setups only"

echo ""

# 5. Push to remote
echo "🚀 Pushing to GitHub..."
git push

echo ""
echo "✅ COMPLETE! Bot is now configured with $passed validated trading symbols."
echo ""
echo "Next steps:"
echo "1. Restart the bot to load new symbols"
echo "2. Monitor Telegram for execution notifications"
echo "3. Review 400_SYMBOL_EXPANSION.md for details"
