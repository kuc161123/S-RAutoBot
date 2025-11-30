# Trading Bot - Complete Architecture Analysis

## Table of Contents
1. [Overview](#overview)
2. [Startup Flow](#startup-flow)
3. [Core Architecture](#core-architecture)
4. [Data Flow](#data-flow)
5. [Strategy System](#strategy-system)
6. [Signal Detection](#signal-detection)
7. [Execution Pipeline](#execution-pipeline)
8. [Risk Management](#risk-management)
9. [ML & Learning System](#ml--learning-system)
10. [Persistence & Tracking](#persistence--tracking)
11. [Key Components](#key-components)

---

## Overview

This is a **scalping-focused cryptocurrency trading bot** for Bybit that:
- Trades multiple symbols simultaneously (50+ USDT pairs)
- Uses 3-minute timeframe for scalping strategy
- Implements adaptive combo-based execution with Pro Rules fallback
- Records phantom trades for ML learning
- Manages risk with dynamic position sizing
- Persists data to PostgreSQL

**Primary Strategy**: Scalp (VWAP pullback + ORB continuation)
**Secondary**: Pro Rules (MTF + RSI + MACD + VWAP + Fib + Volume + Wick + ATR + BBW)

---

## Startup Flow

### 1. Entry Point: `start.py`

```
start.py
  ├─ Kill existing bot instances (ensure single instance)
  ├─ Connect to Redis (for ML models)
  ├─ Optional: Run pretraining (if models missing)
  └─ Launch: python -m autobot.core.bot
```

**Key Steps:**
1. **Instance Management**: Kills any existing bot processes to prevent duplicates
2. **Redis Connection**: Connects to Redis for ML model storage/retrieval
3. **Model Check**: Optionally runs pretraining if models are missing
4. **Bot Launch**: Executes the main bot module

---

## Core Architecture

### Main Bot Class: `TradingBot` (autobot/core/bot.py)

The bot is an async event-driven system with these core components:

```python
TradingBot
├─ Initialization
│  ├─ Load config.yaml
│  ├─ Initialize Bybit broker
│  ├─ Initialize WebSocket handler
│  ├─ Initialize position book
│  ├─ Initialize trade tracker (PostgreSQL)
│  ├─ Initialize Telegram bot
│  └─ Load ML models (if available)
│
├─ Data Management
│  ├─ frames: Dict[symbol, DataFrame]  # 15m candles
│  ├─ frames_3m: Dict[symbol, DataFrame]  # 3m candles (for scalping)
│  └─ CandleStorage: PostgreSQL persistence
│
├─ Strategy System
│  └─ Scalp Strategy (only active strategy)
│
└─ Execution System
   ├─ Adaptive Combo Manager
   ├─ Pro Rules Fallback
   └─ Position Management
```

### Key Initialization Steps:

1. **Config Loading**: Reads `config.yaml`, replaces environment variables
2. **Broker Setup**: Creates Bybit client with API credentials
3. **WebSocket Handler**: `MultiWebSocketHandler` for streaming candle data
4. **Position Book**: Tracks open positions per symbol
5. **Trade Tracker**: PostgreSQL-based trade history
6. **Telegram Bot**: Notifications and commands
7. **ML Components**: Loads scalp scorer if available

---

## Data Flow

### 1. WebSocket Connection

**Component**: `MultiWebSocketHandler` (autobot/data/websocket.py)

```
Bybit WebSocket
  ├─ Splits symbols across multiple connections (max 190 per connection)
  ├─ Handles reconnection automatically
  ├─ Supports primary + ALT endpoints
  └─ Streams kline updates in real-time
```

**Process:**
1. Symbols are split into chunks of 190 (Bybit limit is 200)
2. Each chunk gets its own WebSocket connection
3. Messages are merged into a single async generator
4. Handles connection failures with automatic retry

### 2. Candle Processing

```
WebSocket Message (kline update)
  ├─ Parse JSON: {symbol, interval, data}
  ├─ Convert to DataFrame row
  ├─ Update frames[symbol] (15m) or frames_3m[symbol] (3m)
  ├─ Persist to PostgreSQL (CandleStorage)
  └─ Trigger strategy evaluation
```

**Data Structures:**
- `frames[symbol]`: 15-minute candles (main timeframe)
- `frames_3m[symbol]`: 3-minute candles (scalp strategy)
- Both stored in memory as pandas DataFrames
- Persisted to PostgreSQL for historical analysis

### 3. Indicator Calculation

Indicators are calculated on-demand when strategies need them:
- **RSI** (14-period)
- **MACD** (12, 26, 9)
- **Bollinger Bands** (20, 2)
- **ATR** (14-period)
- **VWAP** (session-anchored or EVWAP)
- **Volume Ratio** (vs 20-bar MA)
- **Fibonacci Retracements**
- **EMA** (fast: 8, slow: 21)
- **Wick Analysis** (upper/lower wick ratios)

---

## Strategy System

### Active Strategy: Scalp

**Location**: `autobot/strategies/scalp/detector.py`

**Strategy Type**: VWAP Pullback + ORB Continuation

**Key Components:**

1. **Signal Detector** (`detect_scalp_signal`)
   - Analyzes 3m candles
   - Detects VWAP pullbacks
   - Checks volume, wick, BB width conditions
   - Returns `ScalpSignal` object

2. **Settings** (`ScalpSettings`)
   - Configurable thresholds
   - R:R ratio (default 2.1:1)
   - VWAP distance limits
   - Volume/BB/Wick requirements

3. **Signal Object** (`ScalpSignal`)
   ```python
   ScalpSignal:
     - side: 'long' | 'short'
     - entry: float
     - sl: float (stop loss)
     - tp: float (take profit)
     - reason: str
     - meta: dict (features for ML)
   ```

### Signal Detection Flow

```
New 3m Candle Arrives
  ├─ Calculate indicators (RSI, MACD, VWAP, etc.)
  ├─ Check VWAP pullback conditions
  │  ├─ Price distance from VWAP (in ATRs)
  │  ├─ VWAP pattern (bounce/reject/revert)
  │  └─ Session anchoring (if enabled)
  ├─ Check volume expansion (vol_ratio >= threshold)
  ├─ Check Bollinger Band width (BBW >= percentile)
  ├─ Check wick alignment (dominant wick in trade direction)
  ├─ Check body ratio (meaningful body in signal direction)
  ├─ Check EMA alignment (fast/slow EMA trend)
  └─ If all pass → Generate ScalpSignal
```

**Detection Criteria (from config):**
- VWAP distance: ≤ 1.5 ATR (configurable)
- Volume ratio: ≥ 0.8x (signal) / ≥ 1.5x (execution)
- BB width: ≥ 0.45% (signal) / ≥ 0.60% (execution)
- Wick delta: ≥ 0.10 (dominant wick)
- Wick ratio: ≥ 0.25 (minimum wick size)
- Body ratio: ≥ 0.30 (meaningful body)
- ATR %: ≥ 0.05% (minimum volatility)

---

## Execution Pipeline

### Execution Flow

```
ScalpSignal Detected
  ├─ Build Features (for ML scoring)
  │  ├─ RSI, MACD, VWAP, Fib, Volume, Wick, ATR, BBW
  │  ├─ MTF alignment (15m trend)
  │  └─ QScore components
  │
  ├─ ML Scoring (optional)
  │  └─ Score signal (0-100)
  │
  ├─ Execution Gate Check
  │  ├─ PATH 1: Adaptive Combo Manager
  │  │  ├─ Check if combo is enabled for this side
  │  │  ├─ Check if signal matches active combo
  │  │  └─ If yes → Execute
  │  │
  │  └─ PATH 2: Pro Rules Fallback
  │     ├─ Check MTF alignment
  │     ├─ Check RSI range (40-60 long, 35-55 short)
  │     ├─ Check MACD (bull for long, bear for short)
  │     ├─ Check VWAP distance (≤ 1.3 ATR)
  │     ├─ Check Fibonacci zone (23-38, 38-50, 50-61)
  │     ├─ Check Volume (≥ 1.5x)
  │     ├─ Check Wick (dominant + minimum size)
  │     ├─ Check ATR % (≥ 0.05%)
  │     └─ Check BB Width (≥ 0.010)
  │
  ├─ If Gate Passes:
  │  ├─ Calculate Position Size (Sizer)
  │  ├─ Check Position Limits (no existing position)
  │  ├─ Place Market Order (Bybit API)
  │  ├─ Set Stop Loss (Bybit API)
  │  ├─ Set Take Profit (Bybit API)
  │  ├─ Record Trade (PostgreSQL)
  │  └─ Send Telegram Notification
  │
  └─ If Gate Fails:
     └─ Record Phantom Trade (for learning)
```

### Adaptive Combo Manager

**Purpose**: Dynamically enable/disable trading combos based on performance

**How it works:**
1. Tracks combo performance (win rate, EV_R) over rolling 30-day window
2. Uses Wilson lower-bound confidence interval for WR
3. Enables combos with WR ≥ 45% (configurable)
4. Disables combos with WR < 45% (with hysteresis to avoid flip-flop)
5. Tracks long/short combos separately

**Combo Structure:**
```python
Combo = {
  'combo_id': 'rsi_40-60_macd_bull_vwap_<1.0_fib_38-50',
  'side': 'long',
  'enabled': bool,
  'wr_lb': float,  # Wilson lower-bound WR
  'ev_r': float,   # Expected value in R
  'samples': int
}
```

### Pro Rules Fallback

**Purpose**: Execute trades when Adaptive Combos aren't ready or combo not enabled

**Rules (all must pass):**

1. **MTF Alignment**: 15m timeframe agrees with trade direction
2. **RSI**: 
   - Long: 40-60 (or per-symbol optimized)
   - Short: 35-55 (or per-symbol optimized)
3. **MACD**: 
   - Long: Bullish (hist > 0) with |hist| ≥ 0.0003
   - Short: Bearish (hist < 0) with |hist| ≥ 0.0003
4. **VWAP Distance**: ≤ 1.3 ATR (or per-symbol optimized)
5. **Fibonacci Zone**: Must be in 23-38, 38-50, or 50-61
6. **Volume Ratio**: ≥ 1.5x (or per-symbol optimized)
7. **Wick**: 
   - Dominant wick in trade direction (delta ≥ 0.10)
   - Minimum wick ratio ≥ 0.25
8. **ATR %**: ≥ 0.05%
9. **BB Width**: ≥ 0.010 (or per-symbol optimized)

**Per-Symbol Optimization**: Symbols can have custom Pro Rules parameters stored in `PER_SYMBOL_PRO_RULES` dict.

---

## Risk Management

### Position Sizing: `Sizer` (autobot/utils/sizer.py)

**Calculation:**
```
R = |entry - sl|  # Risk per unit
risk_amount = account_balance × (risk_percent / 100)
quantity = risk_amount / R
```

**Features:**
- **Percentage-based risk**: Uses % of account balance (default 1%)
- **Fixed USD risk**: Fallback to fixed USD amount
- **Fee-aware sizing**: Adjusts for estimated fees (0.11% total)
- **ML dynamic risk**: Scales risk based on ML score (optional)
- **Minimum order value**: Ensures orders meet exchange minimums (5 USDT)

**ML Dynamic Risk** (if enabled):
- Low ML score → Lower risk (0.5%)
- High ML score → Higher risk (up to 2.0%)
- Linear interpolation between min/max

### Position Management: `Book` (autobot/utils/position.py)

**Features:**
- Tracks open positions per symbol
- Prevents duplicate positions (one position per symbol)
- Manages position lifecycle (open → monitoring → close)
- Handles TP/SL execution

### Risk Config: `RiskConfig`

```python
RiskConfig:
  - risk_usd: 10.0          # Fixed USD risk
  - risk_percent: 1.0       # % of account
  - use_percent_risk: True  # Use % mode
  - use_ml_dynamic_risk: False  # ML-based scaling
  - ml_risk_min_score: 70.0
  - ml_risk_max_score: 85.0
  - ml_risk_min_percent: 2.0
  - ml_risk_max_percent: 4.0
```

---

## ML & Learning System

### Components

1. **Scalp Scorer** (`autobot/strategies/scalp/scorer.py`)
   - Scores signals 0-100
   - Used for execution gating (if enabled)
   - Trained on historical phantom outcomes

2. **Phantom Tracker** (`autobot/strategies/scalp/phantom.py`)
   - Records signals that didn't execute
   - Tracks outcomes (TP/SL hit)
   - Used for ML training data

3. **QScore Adapter** (`autobot/strategies/scalp/qscore_adapter.py`)
   - Quality score calculation
   - Combines multiple factors
   - Used for signal prioritization

### Phantom Trade System

**Purpose**: Learn from signals that didn't execute

**Flow:**
```
Signal Detected but Blocked
  ├─ Record as Phantom Trade
  ├─ Track entry, SL, TP
  ├─ Monitor price movement
  ├─ Record outcome (TP hit, SL hit, timeout)
  └─ Use for ML training
```

**Phantom Flow Controller:**
- Adaptive daily targets (200 phantoms/day for scalp)
- Relaxes thresholds when behind pace
- Caps relaxation based on recent WR
- Session-based multipliers

---

## Persistence & Tracking

### Trade Tracker: PostgreSQL

**Component**: `TradeTrackerPostgres` (autobot/data/tracker.py)

**Stores:**
- Trade history (entry, exit, P&L)
- Open positions
- Trade metadata (strategy, ML score, features)
- Performance metrics

### Candle Storage: PostgreSQL

**Component**: `CandleStorage` (autobot/data/storage.py)

**Stores:**
- Historical candle data
- Multiple timeframes (3m, 15m)
- Used for backtesting and analysis

### Redis (Optional)

**Stores:**
- ML models (serialized)
- Phantom trade data
- Flow controller state
- Adaptive combo state

---

## Key Components

### 1. Bybit Broker (`autobot/brokers/bybit.py`)

**Responsibilities:**
- REST API calls (orders, positions, account)
- WebSocket connection management
- Order placement (market, limit)
- Position management (TP/SL)
- Error handling and retries

### 2. Telegram Bot (`autobot/core/telegram.py`)

**Features:**
- Trade notifications
- Performance reports
- Manual commands (/status, /positions, /watchlist)
- Interactive buttons
- System messages

### 3. Market Regime (`autobot/utils/regime.py`)

**Purpose**: Classify market conditions
- Normal, High, Extreme volatility
- Used for context (not execution gating in scalp)

### 4. Symbol Clusters (`autobot/utils/symbol_clusters.json`)

**Purpose**: Group symbols by volatility characteristics
- Used for regime-aware behavior
- Helps with symbol-specific optimizations

---

## Execution Summary

### Complete Flow (Signal to Trade)

```
1. WebSocket receives 3m candle update
2. Update frames_3m[symbol] DataFrame
3. Persist candle to PostgreSQL
4. Scalp detector analyzes new candle
5. If signal detected:
   a. Build features (RSI, MACD, VWAP, etc.)
   b. Score with ML (optional)
   c. Check execution gates:
      - Adaptive Combo enabled? → Execute if combo matches
      - Else: Pro Rules fallback → Execute if all rules pass
   d. If execute:
      - Calculate position size (Sizer)
      - Place market order (Bybit)
      - Set TP/SL (Bybit)
      - Record trade (PostgreSQL)
      - Send notification (Telegram)
   e. If blocked:
      - Record phantom trade
      - Monitor outcome
      - Use for ML training
6. Monitor open positions for TP/SL
7. Close positions when TP/SL hit
8. Record final trade outcome
```

---

## Configuration

### Main Config: `config.yaml`

**Key Sections:**
- `bybit`: API credentials and endpoints
- `trade`: Symbols, risk settings, timeframes
- `scalp`: Strategy-specific settings
  - `exec`: Execution settings (combos, fallback)
  - `signal`: Signal detection thresholds
  - `hard_gates`: Execution gates
- `phantom`: Phantom trade settings
- `phantom_flow`: Adaptive flow control
- `telegram`: Bot token and chat ID

---

## Performance & Optimization

### Current Optimizations:
1. **Multi-WebSocket**: Handles 200+ symbols efficiently
2. **PostgreSQL Persistence**: Fast historical data access
3. **In-Memory Frames**: Fast indicator calculation
4. **Async Architecture**: Non-blocking I/O
5. **Adaptive Combos**: Dynamic strategy optimization
6. **Per-Symbol Rules**: Optimized parameters per symbol

### Learning System:
- Phantom trades provide training data
- ML models improve over time
- Adaptive combos adjust based on performance
- Per-symbol optimization from backtesting

---

## Summary

This is a **sophisticated, production-ready trading bot** with:
- ✅ Multi-symbol scalping strategy
- ✅ Adaptive execution system (Combos + Pro Rules)
- ✅ Comprehensive risk management
- ✅ ML-based learning system
- ✅ Robust data persistence
- ✅ Real-time monitoring (Telegram)
- ✅ Per-symbol optimization
- ✅ Phantom trade learning

The bot is designed to continuously learn and adapt, using both rule-based (Pro Rules) and ML-based (Adaptive Combos) approaches for optimal performance.

