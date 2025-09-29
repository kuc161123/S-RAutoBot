# 🤖 AutoTrading Bot - Complete Architecture & Flow

## 📊 Overview

This is a sophisticated automated trading bot for Bybit perpetual futures that uses ML-enhanced pullback strategies with pivot-based support/resistance detection. The bot monitors 263 trading pairs simultaneously and executes trades based on market structure analysis with machine learning scoring.

## 🏗️ Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              LIVE BOT MAIN LOOP                          │
│                                (live_bot.py)                             │
└─────────────────┬───────────────────────────────────┬───────────────────┘
                  │                                   │
                  ▼                                   ▼
┌─────────────────────────────┐     ┌─────────────────────────────────────┐
│    BYBIT WEBSOCKET STREAM   │     │         TELEGRAM BOT                 │
│   (Multi-WebSocket Handler) │     │        (telegram_bot.py)             │
│  • Live price data (263 sym)│     │  • User commands & monitoring        │
│  • 15-minute candles        │     │  • Real-time notifications           │
│  • Auto-reconnection        │     │  • Risk management interface         │
└─────────────────┬───────────┘     └─────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         SIGNAL DETECTION ENGINE                          │
│                   (strategy_pullback_ml_learning.py)                    │
│                                                                          │
│  1. Market Structure Analysis (HH/HL, LH/LL patterns)                   │
│  2. S/R Breakout Detection (0.3% zones)                                 │
│  3. Pullback Confirmation (2+ candles)                                  │
│  4. Multi-Timeframe S/R Integration                                     │
│  5. Feature Extraction (22 technical + cluster features)                │
└─────────────────┬───────────────────────────────────┬───────────────────┘
                  │                                   │
                  ▼                                   ▼
┌─────────────────────────────┐     ┌─────────────────────────────────────┐
│      ML SCORING ENGINE      │     │       PHANTOM TRADE TRACKER          │
│ (ml_signal_scorer_immediate)│     │    (phantom_trade_tracker.py)        │
│                             │     │                                      │
│ • Signal quality scoring    │     │ • Track all signals (taken/rejected)│
│ • Adaptive threshold (70+)  │     │ • Monitor hypothetical outcomes      │
│ • Continuous learning       │     │ • Provide learning data to ML        │
│ • Symbol clustering         │     │ • 30-day retention                   │
└─────────────────┬───────────┘     └──────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          RISK & POSITION MANAGER                         │
│                    (position_mgr.py + sizer.py)                         │
│                                                                          │
│  • One position per symbol rule                                         │
│  • 3% default risk (configurable)                                       │
│  • ML dynamic risk scaling (1-5%)                                       │
│  • Balance verification                                                  │
│  • Stop loss validation                                                  │
└─────────────────┬───────────────────────────────────┬───────────────────┘
                  │                                   │
                  ▼                                   ▼
┌─────────────────────────────┐     ┌─────────────────────────────────────┐
│     BYBIT API BROKER        │     │         DATA PERSISTENCE             │
│     (broker_bybit.py)       │     │                                      │
│                             │     │ ┌─────────────────────────────┐     │
│ • Market order execution    │     │ │   Candle Storage (Postgres) │     │
│ • TP/SL management          │     │ │ • 30-day retention          │     │
│ • Leverage setting          │     │ │ • Auto-save every 15 min    │     │
│ • Position tracking         │     │ └─────────────────────────────┘     │
└─────────────────────────────┘     │ ┌─────────────────────────────┐     │
                                    │ │  Trade Tracker (Postgres)   │     │
                                    │ │ • Performance statistics     │     │
                                    │ │ • Trade history             │     │
                                    │ └─────────────────────────────┘     │
                                    └─────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                         ML EVOLUTION SYSTEM                              │
│                        (Shadow Mode - Learning)                          │
│                                                                          │
│  • Symbol-specific models (50+ trades required)                         │
│  • Confidence-based scoring (0-80%)                                     │
│  • 4-hour retrain cycle                                                 │
│  • Performance tracking & comparison                                     │
└─────────────────────────────────────────────────────────────────────────┘
```

## 🔄 Complete Trading Flow

### 1. **Initialization Phase**
```
1. Load configuration (config.yaml)
2. Initialize ML scorer & phantom tracker
3. Connect to Bybit API
4. Load/fetch historical candles (200+ per symbol)
5. Initialize HTF support/resistance levels
6. Recover existing positions
7. Start Telegram bot
8. Begin WebSocket streaming
```

### 2. **Signal Detection Flow**
```
For each completed 15-minute candle:
1. Update price data frame
2. Check market structure (trending/ranging)
3. Detect S/R breakouts with zones
4. Wait for pullback formation
5. Confirm with 2+ candles
6. Extract 22+ ML features
7. Score signal with ML (0-100)
8. Record in phantom tracker
9. Execute if score ≥ 70
```

### 3. **Order Execution Flow**
```
1. Validate stop loss placement
2. Calculate position size (risk-based)
3. Set leverage BEFORE entry
4. Place market order
5. Get actual entry price
6. Recalculate TP for R:R ratio
7. Set TP (limit) and SL (market)
8. Update position book
9. Send Telegram notification
```

### 4. **Position Management Flow**
```
Every 30 seconds:
1. Check exchange positions
2. Compare with local book
3. Verify closed positions via orders
4. Calculate exit price & reason
5. Record trade outcome
6. Update ML with results
7. Reset strategy state
8. Check for ML retrain trigger
```

## 🧠 ML System Components

### **Immediate ML Scorer**
- **Theory-based start**: Works from day 1 with 70% threshold
- **Feature set**: 34 features (technical + cluster + MTF)
- **Learning sources**: Real trades + phantom trades
- **Retrain cycle**: Every 100 trades
- **Models**: XGBoost ensemble

### **Phantom Trade System**
- **Purpose**: Learn from all signals, not just executed ones
- **Tracking**: Entry, SL, TP, current price, max moves
- **Storage**: Redis with 30-day retention
- **Benefits**: 5-10x more learning data

### **ML Evolution (Shadow Mode)**
- **Symbol-specific models**: Custom for each trading pair
- **Activation**: After 50 trades per symbol
- **Confidence range**: 0-80% per symbol
- **Currently**: Learning mode only (not affecting trades)

## 📈 Key Features

### **Risk Management**
- Fixed percentage or USD risk per trade
- ML-based dynamic risk scaling
- One position per symbol limit
- Minimum balance verification
- Stop loss validation

### **Data Management**
- PostgreSQL for production (Railway)
- SQLite fallback for development
- 30-day data retention
- Auto-save every 15 minutes
- Connection pooling optimization

### **Monitoring & Control**
- 40+ Telegram commands
- Real-time position tracking
- Performance statistics
- ML system insights
- Emergency controls

### **Safety Features**
- WebSocket auto-reconnection
- Order preservation on restart
- Graceful shutdown handling
- Comprehensive error logging
- Position recovery system

## 🎯 Trading Strategy Details

### **ML Pullback Strategy (Active)**
The bot uses an advanced pullback strategy that:

1. **Identifies Trend**: Uses pivot highs/lows to determine market structure
2. **Detects Breakout**: Waits for clean break of S/R with 0.3% zone
3. **Confirms Pullback**: Requires retracement that respects breakout level
4. **Waits for Confirmation**: Needs 2+ candles in trend direction
5. **Scores with ML**: Evaluates signal quality based on 34 features
6. **Manages Risk**: Places stops using hybrid method (lowest of 3 calculations)
7. **Targets Profit**: Sets TP at 2.5x risk after fees (actual 2.67x gross)

### **Stop Loss Calculation**
Uses the most conservative of:
- Previous pivot ± buffer
- Breakout level ± buffer
- Pullback extreme ± buffer

Buffer increases 30-50% in high volatility conditions.

## 🚀 Performance Optimization

- **Multi-WebSocket**: Handles 263 symbols across multiple connections
- **Batch Processing**: Groups similar operations
- **Selective Logging**: Reduces log spam with periodic summaries
- **Efficient Storage**: Indexed database queries
- **Smart Caching**: Reuses calculated values

## 📊 Current Configuration

- **Symbols**: 263 trading pairs
- **Timeframe**: 15-minute candles
- **Risk**: 3% per trade (configurable)
- **R:R Ratio**: 2.5:1 after fees
- **ML Threshold**: 70% minimum score
- **ML Evolution**: Shadow mode (learning only)

---

This bot represents a sophisticated trading system that combines traditional technical analysis with modern machine learning, all while maintaining strict risk management and comprehensive monitoring capabilities.