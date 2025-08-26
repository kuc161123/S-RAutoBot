#!/bin/bash

echo "Redis Setup Options:"
echo "===================="
echo ""
echo "Option 1: Install Redis locally (macOS with Homebrew):"
echo "  brew install redis"
echo "  brew services start redis"
echo ""
echo "Option 2: Use Redis Cloud (Free tier):"
echo "  1. Go to https://redis.com/try-free/"
echo "  2. Create free account (30MB free)"
echo "  3. Get your connection URL"
echo "  4. Update .env with: REDIS_URL=redis://default:password@host:port"
echo ""
echo "Option 3: Use Docker:"
echo "  docker run -d -p 6379:6379 --name redis redis:alpine"
echo ""
echo "Option 4: Continue without Redis (using in-memory queue):"
echo "  Set REDIS_URL= (empty) in .env"
echo ""

# Check if Redis is running locally
if command -v redis-cli &> /dev/null; then
    echo "Checking local Redis..."
    if redis-cli ping &> /dev/null; then
        echo "✅ Redis is running locally on port 6379"
        echo "Your .env should have: REDIS_URL=redis://localhost:6379/0"
    else
        echo "❌ Redis is installed but not running"
        echo "Run: brew services start redis (on macOS)"
    fi
else
    echo "❌ Redis not installed locally"
fi