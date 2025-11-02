# Risk and Position Management - Complete Analysis

This document provides a comprehensive analysis of the AutoTrading Bot's risk management and position tracking system.

## Quick Links
- **One-Per-Symbol Rule**: Section 1
- **Risk Calculation**: Section 2
- **Leverage Management**: Section 3
- **Position Sizing**: Section 4
- **Order Execution**: Section 5
- **Partial vs Full Mode**: Section 6
- **Position Recovery**: Section 7
- **Closed Position Detection**: Section 8

---

## 1. ONE-PER-SYMBOL RULE ENFORCEMENT

### Architecture
The bot maintains a single position maximum per symbol through the `Book` dataclass:

```python
@dataclass
class Book:
    positions: Dict[str, Position] = field(default_factory=dict)
```

**Key files:**
- `/Users/lualakol/AutoTrading Bot/position_mgr.py` (Lines 21-52): Data structures
- `/Users/lualakol/AutoTrading Bot/live_bot.py`: Enforcement logic

### Enforcement Points

**PRE-EXECUTION CHECK** (Before every order):
```python
if sym in book.positions:
    logger.info(f"[{sym}] Already have open position, skip")
    continue
```
- Line 1365-1370: Scalp execution
- Line 7724: Trend execution
- Line 10284: Mean reversion execution

**POST-EXECUTION** (After successful fill):
```python
self.book.positions[sym] = Position(
    side=side, qty=qty, entry=entry, sl=sl, tp=tp,
    entry_time=datetime.now(), strategy_name=strategy_name,
    ml_score=ml_score, qscore=qv
)
```

**ON CLOSE** (When position closed):
```python
book.positions.pop(symbol)
logger.info(f"[{symbol}] Position removed from tracking")
```

---

## 2. RISK CALCULATION SYSTEM

### Configuration (config.yaml)
```yaml
trade:
  risk_percent: 1.0              # Risk as % of balance
  use_percent_risk: true         # Use percentage mode (ACTIVE)
  risk_usd: 10                   # Fallback if balance unknown
  fee_total_pct: 0.00110         # Include 0.11% fees
  use_ml_dynamic_risk: false     # Dynamic risk by ML score (OFF)
```

### Risk Calculation Flow (sizer.py, Lines 38-59)

```
STEP 1: Determine Risk Percentage
├─ If ML dynamic risk: Interpolate between min/max by ML score
└─ Otherwise: Use fixed risk_percent (1%)

STEP 2: Calculate Risk Amount in USD
├─ Formula: risk_amount = balance × (risk_percent / 100)
├─ Example: $1000 × 1% = $10 per trade
└─ Fallback to risk_usd if balance unavailable

STEP 3: Calculate Position Quantity
├─ R = |entry_price - stop_loss_price|
├─ Base qty = risk_amount / R
├─ qty = round_step(base_qty, qty_step)
└─ Validate: min_qty and min_order_value

STEP 4: Include Fees (Optional)
├─ When include_fees=true
├─ Denom = R + (fee_pct × avg_price)
└─ Results in wider SL to account for costs
```

### Example Calculation
```
Account: $1000
Entry: $50,000 (BTCUSDT)
SL: $49,500
R = $500

Risk amount = $1000 × 1% = $10
Qty = $10 / $500 = 0.02 BTC
After rounding: 0.02 BTC (qty_step: 0.001)
Position value = 0.02 × $50,000 = $1000
Max loss = $10 = 1% of account ✓
```

---

## 3. LEVERAGE DETERMINATION

### Symbol-Specific Caps (config.yaml)
```yaml
symbol_meta:
  BTCUSDT:
    max_leverage: 100.0   # Major coins
  BNBUSDT:
    max_leverage: 50.0    # Mid-tier altcoins
  FILUSDT:
    max_leverage: 25.0    # Smaller altcoins
  default:
    max_leverage: 10      # Fallback
```

### Setting Leverage (live_bot.py, Lines 8663-8664)
```python
max_lev = int(m.get("max_leverage", 10))  # From config
bybit.set_leverage(sym, max_lev)          # Set BEFORE market order
_ = bybit.place_market(sym, side, qty)    # Then place order
```

### Fallback Chain (broker_bybit.py, Lines 158-198)
```
Attempts: [requested_leverage, 20x, 10x]

Example:
  Requested: 75x (PEPE cap)
  Bybit rejects: "risk limit exceeded"
  Bot retries: 20x → Success
  
Final: 20x used instead of 75x
```

---

## 4. POSITION SIZING

### Quantity Rounding (position_mgr.py)
```python
def round_step(x: float, step: float) -> float:
    """Rounds to step respecting decimal precision"""
    d_step = decimal.Decimal(str(step))
    decimal_places = -d_step.as_tuple().exponent if d_step.as_tuple().exponent < 0 else 0
    rounded = round(x / step) * step
    return round(rounded, decimal_places)
```

### Validations
- **min_qty**: Minimum allowed quantity per symbol
- **min_order_value**: Minimum notional (e.g., $5 for Bybit)
- **qty_step**: Rounding increment per symbol
- If any validation fails: Position rejected (returns 0)

---

## 5. ORDER EXECUTION FLOW

### Complete Sequence
```
1. PRE-FLIGHT CHECKS
   ├─ Position doesn't exist
   ├─ ML score >= threshold
   ├─ Market regime OK
   └─ HTF support/resistance aligned

2. SIZING
   └─ Sizer.qty_for() → Position quantity

3. LEVERAGE
   └─ Set leverage (with fallback chain)

4. MARKET ORDER
   └─ place_market(symbol, side, qty)
      • IOC time-in-force (immediate or cancel)
      • reduceOnly=False (opening order)
      • One-way mode (single direction per symbol)

5. READBACK
   └─ Confirm fill and get actual entry price

6. TP/SL SETUP
   └─ Preferred: Partial mode with quantities
      • tpSize = qty, slSize = qty
      • tpOrderType = Limit, slOrderType = Market
   └─ Fallback: Full mode if qty unavailable

7. UPDATE BOOK
   └─ book.positions[symbol] = Position(...)

8. NOTIFICATIONS
   └─ Telegram alert + logging
```

---

## 6. PARTIAL VS FULL MODE

### Partial Mode (PREFERRED)
```python
data = {
    "tpslMode": "Partial",
    "tpSize": qty,              # Full position
    "slSize": qty,              # Full position
    "tpOrderType": "Limit",     # Better fills
    "slOrderType": "Market",    # Guaranteed exit
}
```

**Advantages:**
- Limit TP for better fills
- Market SL for guaranteed exit
- Explicit quantities prevent rounding errors
- Enables scale-out (TP1, TP2)

### Full Mode (FALLBACK)
```python
data = {
    "tpslMode": "Full",         # Entire position closes
    "tpOrderType": "Limit",
    "slOrderType": "Market",
    # No sizes needed
}
```

**Advantages:**
- Works with API lag (no need to read position size)
- Simpler API call

**Disadvantages:**
- Can't use Limit TP in Full mode (actual behavior: can use)
- Less control for complex strategies

---

## 7. POSITION RECOVERY ON RESTART

### Recovery Entry Point (live_bot.py, Line 6436)
```python
await self.recover_positions(book, sizer)  # Called before main loop
```

### Recovery Process (Lines 5410-5549)
```
1. FETCH: Get all positions with size > 0 from Bybit

2. POPULATE BOOK:
   ├─ Create Position objects
   ├─ Store symbol, side, qty, entry, TP, SL
   ├─ Validate TP/SL (swap if inverted)
   └─ Add to book.positions[symbol]

3. RESTORE METADATA:
   ├─ Try Redis: openpos:strategy:{symbol}
   ├─ Fallback: 'trend_pullback' or 'unknown'
   └─ Store strategy_name

4. RESTORE POSITION META:
   ├─ Redis key: openpos:meta:{symbol}
   └─ Execution parameters

5. RESTORE SCALE-OUT:
   ├─ Redis key: openpos:scaleout:{symbol}
   ├─ If available: Hydrate _scaleout[symbol]
   └─ If missing: Reconstruct from config

6. NOTIFY:
   └─ Telegram: "Recovered N position(s) - WILL NOT MODIFY THEM"
```

### Key Principle
**ALL existing TP/SL orders on Bybit are LEFT UNCHANGED**
- No modifications to entry, TP, or SL
- Bot only monitors for closes and records results
- Zero interference with pre-existing positions

---

## 8. CLOSED POSITION DETECTION

### Detection Loop (Every 30 seconds, Line 7403)
```
1. Fetch live positions from Bybit
2. Build set of symbols with open positions
3. For each symbol in book but NOT live:
   ├─ Fetch order history (last 20 orders)
   ├─ Search for FILLED reduceOnly order
   ├─ Read exit price and determine reason
   └─ Re-confirm with live size (avoid false positives)
4. For each confirmed close:
   ├─ Calculate PnL
   ├─ Record in trade tracker
   └─ Remove from book
```

### Exit Reason Classification (Lines 4830-4880)
```
LONG:
  ├─ exit_price >= TP × 0.997 → "tp"
  ├─ exit_price <= SL × 1.003 → "sl"
  └─ Otherwise → "manual"

SHORT: (Opposite logic)
  ├─ exit_price <= TP × 1.003 → "tp"
  ├─ exit_price >= SL × 0.997 → "sl"
  └─ Otherwise → "manual"
```

---

## Key Safety Features

✓ **One-Per-Symbol**: Enforced by Book dictionary structure
✓ **Leverage Safety**: Symbol-specific caps with fallback chain
✓ **Position Recovery**: Preserves ALL existing orders
✓ **Position Sizing**: Fixed 1% risk per trade (scalable by ML score)
✓ **Fee-Aware**: Can include 0.11% fees in sizing
✓ **Decimal Precision**: Custom round_step() function

---

## Configuration Quick Reference

**Risk Settings:**
```yaml
risk_percent: 1.0              # % of balance
use_percent_risk: true         # Percentage mode (ACTIVE)
fee_total_pct: 0.00110         # 0.11% fees
```

**Leverage (symbol_meta):**
```yaml
BTCUSDT: 100x
ETHUSDT: 100x
Most altcoins: 25-50x
Default: 10x
```

**Position Settings:**
```yaml
confirmation_candles: 2
ml_min_score: 70.0
rr: 2.5
```

---

## File Locations
- **position_mgr.py**: Core risk structures and Position/Book classes
- **sizer.py**: Position sizing calculations
- **broker_bybit.py**: Exchange API wrapper with leverage/order methods
- **live_bot.py**: Main orchestrator with recovery, execution, detection
- **config.yaml**: All risk and symbol configuration

