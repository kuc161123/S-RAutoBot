#!/bin/bash
# Generate enhanced clusters for the first time

echo "🎯 Generating enhanced symbol clusters..."
echo "This will analyze historical data for all symbols"
echo ""

# Run the enhanced cluster generation
python generate_enhanced_clusters.py

echo ""
echo "✅ Enhanced clustering complete!"
echo "You can now use /clusters command in Telegram to view status"
echo "Use /update_clusters to refresh clusters weekly"