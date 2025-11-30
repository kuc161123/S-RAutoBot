# Combo System - Improvement Recommendations

## Current Implementation Analysis

### Strengths ✅
- Wilson Lower Bound for statistical confidence
- Hysteresis to prevent flip-flopping
- Side separation (long/short)
- Comprehensive tracking (exec vs phantom, 24h)
- State persistence to Redis

### Areas for Improvement 🔧

---

## 1. Adaptive Sample Size Requirements

### Current Issue
- Fixed `min_sample_size: 30` for all combos
- Rare combos may never reach 30 samples
- Common combos may need more samples for confidence

### Improvement: Adaptive Sample Size
```python
def _adaptive_sample_size(self, combo_frequency: float) -> int:
    """
    Adjust sample size based on combo frequency
    - Rare combos (< 1% of signals): Lower threshold (15-20)
    - Common combos (> 5% of signals): Higher threshold (40-50)
    - Normal combos: Standard threshold (30)
    """
    if combo_frequency < 0.01:  # Rare
        return 15
    elif combo_frequency > 0.05:  # Common
        return 45
    else:
        return 30
```

**Benefits**:
- Rare combos can enable faster
- Common combos require more evidence
- Better utilization of data

---

## 2. Recency Weighting

### Current Issue
- All trades weighted equally (30-day window)
- Recent performance may be more relevant
- Market conditions change over time

### Improvement: Exponential Decay Weighting
```python
def _weighted_metrics(self, trades: List[Trade], half_life_days: float = 7.0):
    """
    Weight recent trades more heavily using exponential decay
    - Half-life: 7 days (trades older than 7 days have 50% weight)
    - More recent = higher weight
    """
    now = datetime.utcnow()
    total_weight = 0.0
    weighted_wins = 0.0
    weighted_rr = 0.0
    
    for trade in trades:
        age_days = (now - trade.exit_time).total_seconds() / 86400
        weight = 2 ** (-age_days / half_life_days)  # Exponential decay
        total_weight += weight
        if trade.outcome == 'win':
            weighted_wins += weight
        weighted_rr += trade.realized_rr * weight
    
    wr = (weighted_wins / total_weight * 100) if total_weight > 0 else 0.0
    ev_r = (weighted_rr / total_weight) if total_weight > 0 else 0.0
    return wr, ev_r, total_weight
```

**Benefits**:
- Adapts faster to changing market conditions
- Recent performance has more influence
- Still considers historical data

---

## 3. Confidence-Based Thresholds

### Current Issue
- Fixed WR threshold (45%) regardless of sample size
- Small samples need higher thresholds
- Large samples can use lower thresholds

### Improvement: Dynamic Thresholds
```python
def _confidence_adjusted_threshold(self, n: int, base_threshold: float) -> float:
    """
    Adjust threshold based on sample size confidence
    - Small samples (n < 20): Higher threshold (50-55%)
    - Medium samples (20-50): Standard threshold (45%)
    - Large samples (n > 100): Lower threshold (40-42%)
    """
    if n < 20:
        return base_threshold + 5.0  # Require higher WR for small samples
    elif n > 100:
        return base_threshold - 3.0  # Can accept lower WR with high confidence
    else:
        return base_threshold
```

**Benefits**:
- More statistically sound
- Prevents premature enablement
- Better use of large datasets

---

## 4. Symbol-Specific Combo Performance

### Current Issue
- Combos tracked globally across all symbols
- Some combos may work well for BTC but not for altcoins
- No symbol-specific optimization

### Improvement: Per-Symbol Combo Tracking
```python
def _analyze_combo_performance(self, side: Optional[str] = None, symbol: Optional[str] = None):
    """
    Analyze combo performance with optional symbol filtering
    - Track combos per symbol
    - Enable combos only for symbols where they perform well
    """
    # Group by combo + symbol
    combos_by_symbol = {}
    for trade in trades:
        combo_key = self._get_combo_key(trade)
        symbol = trade.symbol
        key = f"{combo_key}:{symbol}"
        # Track performance per symbol
        ...
    
    # Enable combo only for symbols where it meets threshold
    return combos_by_symbol
```

**Benefits**:
- Better optimization per symbol
- Avoids bad combos on specific symbols
- More granular control

---

## 5. Time-Based Performance Analysis

### Current Issue
- No time-of-day or session awareness
- Some combos may work better during certain hours
- Market sessions have different characteristics

### Improvement: Session-Aware Combos
```python
def _session_performance(self, trades: List[Trade]) -> Dict[str, ComboPerformance]:
    """
    Track combo performance by trading session
    - Asian: 00:00-08:00 UTC
    - European: 08:00-16:00 UTC
    - US: 16:00-24:00 UTC
    """
    sessions = {'asian': [], 'european': [], 'us': []}
    for trade in trades:
        hour = trade.exit_time.hour
        if 0 <= hour < 8:
            sessions['asian'].append(trade)
        elif 8 <= hour < 16:
            sessions['european'].append(trade)
        else:
            sessions['us'].append(trade)
    
    # Calculate performance per session
    # Enable combo only during profitable sessions
    return session_performance
```

**Benefits**:
- Optimize for market sessions
- Disable combos during unprofitable hours
- Better adaptation to market cycles

---

## 6. Combo Key Parsing & Reverse Lookup

### Current Issue
- Combo keys are strings: "RSI:40-60 MACD:bull VWAP:1.2+ Fib:0-23 noMTF"
- Can't easily parse back to numeric ranges
- Hard to match signals to combos efficiently

### Improvement: Structured Combo Keys
```python
@dataclass
class ComboKey:
    """Structured combo identifier"""
    rsi_bin: str
    macd_bin: str
    vwap_bin: str
    fib_zone: str
    mtf: bool
    
    def to_string(self) -> str:
        mtf_str = 'MTF' if self.mtf else 'noMTF'
        return f"RSI:{self.rsi_bin} MACD:{self.macd_bin} VWAP:{self.vwap_bin} Fib:{self.fib_zone} {mtf_str}"
    
    @classmethod
    def from_string(cls, key_str: str) -> 'ComboKey':
        """Parse combo key string back to structured format"""
        # Parse "RSI:40-60 MACD:bull VWAP:1.2+ Fib:0-23 noMTF"
        parts = key_str.split()
        rsi = parts[0].split(':')[1]
        macd = parts[1].split(':')[1]
        vwap = parts[2].split(':')[1]
        fib = parts[3].split(':')[1]
        mtf = 'MTF' in parts[4]
        return cls(rsi_bin=rsi, macd_bin=macd, vwap_bin=vwap, fib_zone=fib, mtf=mtf)
    
    def matches_features(self, feats: dict) -> bool:
        """Check if features match this combo"""
        rsi = feats.get('rsi_14', 0)
        # Convert numeric RSI to bin and check match
        ...
```

**Benefits**:
- Efficient matching
- Easy to extend
- Better type safety

---

## 7. Performance Decay Detection

### Current Issue
- No detection of performance degradation
- Combos may start failing but stay enabled
- No early warning system

### Improvement: Performance Trend Analysis
```python
def _detect_performance_decay(self, combo: ComboPerformance) -> bool:
    """
    Detect if combo performance is declining
    - Compare recent 7-day WR vs 30-day WR
    - If recent WR < (30-day WR - 10%), flag as decaying
    """
    recent_wr = combo.wr_7d  # Last 7 days
    overall_wr = combo.wr    # Last 30 days
    
    if recent_wr < (overall_wr - 10.0):
        return True  # Performance declining
    return False

def update_combo_filters(self):
    # ... existing code ...
    
    # Check for performance decay
    if perf.enabled and self._detect_performance_decay(perf):
        # Disable decaying combos even if above threshold
        curr_enabled = False
        changes.append(f"DECAY: {key} - Recent WR {recent_wr:.1f}% < Overall {overall_wr:.1f}%")
```

**Benefits**:
- Early detection of failing combos
- Proactive disabling
- Better risk management

---

## 8. Combo Hierarchies & Grouping

### Current Issue
- All combos treated independently
- No grouping of related combos
- Can't enable/disable related patterns together

### Improvement: Combo Groups
```python
class ComboGroup:
    """Group related combos together"""
    name: str
    combos: List[str]  # Combo IDs in this group
    group_wr: float
    group_ev_r: float
    
    def should_enable_group(self) -> bool:
        """Enable group if majority of combos are profitable"""
        profitable = sum(1 for c in self.combos if c.enabled)
        return profitable >= (len(self.combos) * 0.6)  # 60% threshold

# Example groups:
groups = {
    'vwap_pullback': [
        'RSI:40-60 MACD:bull VWAP:<0.6 Fib:38-50 MTF',
        'RSI:40-60 MACD:bull VWAP:0.6-1.2 Fib:38-50 MTF',
    ],
    'fib_retracement': [
        'RSI:40-60 MACD:bull VWAP:<0.6 Fib:23-38 MTF',
        'RSI:40-60 MACD:bull VWAP:<0.6 Fib:38-50 MTF',
    ],
}
```

**Benefits**:
- Better organization
- Group-level decisions
- Easier management

---

## 9. Minimum Trade Frequency Check

### Current Issue
- Combos may have good WR but very low frequency
- Enabling rare combos may not be useful
- No minimum frequency requirement

### Improvement: Frequency-Based Filtering
```python
def _check_minimum_frequency(self, combo: ComboPerformance, total_signals: int) -> bool:
    """
    Require minimum frequency for combo to be useful
    - Combo must occur at least 0.5% of signals (1 in 200)
    - Prevents enabling combos that rarely occur
    """
    frequency = combo.n / total_signals if total_signals > 0 else 0.0
    min_frequency = 0.005  # 0.5%
    return frequency >= min_frequency
```

**Benefits**:
- Only enable useful combos
- Avoid rare patterns that don't matter
- Better resource utilization

---

## 10. Enhanced Metrics

### Current Issue
- Only tracks WR and EV_R
- Missing other useful metrics
- No risk-adjusted returns

### Improvement: Additional Metrics
```python
@dataclass
class ComboPerformance:
    # ... existing fields ...
    
    # New metrics
    sharpe_ratio: float = 0.0      # Risk-adjusted returns
    max_drawdown: float = 0.0       # Maximum losing streak
    avg_win_r: float = 0.0         # Average win in R
    avg_loss_r: float = 0.0        # Average loss in R
    profit_factor: float = 0.0      # Gross profit / Gross loss
    expectancy: float = 0.0          # Expected value per trade
    kelly_percent: float = 0.0      # Optimal position sizing %
    
    def calculate_advanced_metrics(self, trades: List[Trade]):
        """Calculate comprehensive performance metrics"""
        wins = [t.realized_rr for t in trades if t.outcome == 'win']
        losses = [abs(t.realized_rr) for t in trades if t.outcome == 'loss']
        
        self.avg_win_r = np.mean(wins) if wins else 0.0
        self.avg_loss_r = np.mean(losses) if losses else 0.0
        self.profit_factor = sum(wins) / sum(losses) if losses else 0.0
        
        # Sharpe ratio (simplified)
        returns = [t.realized_rr for t in trades]
        if len(returns) > 1:
            self.sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0.0
```

**Benefits**:
- Better performance assessment
- Risk-adjusted metrics
- More informed decisions

---

## 11. Machine Learning Integration

### Current Issue
- Rule-based enable/disable only
- No predictive capability
- Can't anticipate combo performance

### Improvement: ML-Based Combo Prediction
```python
class ComboMLPredictor:
    """Use ML to predict combo performance"""
    
    def predict_combo_performance(self, combo_features: dict, market_features: dict) -> float:
        """
        Predict combo WR using ML model
        - Input: Combo features + current market conditions
        - Output: Predicted WR
        """
        # Use trained model to predict
        predicted_wr = model.predict([combo_features, market_features])
        return predicted_wr
    
    def should_enable_combo(self, predicted_wr: float, historical_wr: float) -> bool:
        """
        Enable combo if:
        - Historical WR meets threshold, OR
        - Predicted WR is high and combo is new (no history)
        """
        if historical_wr >= 45.0:
            return True  # Proven performance
        elif predicted_wr >= 50.0 and historical_wr == 0:
            return True  # High prediction for new combo
        return False
```

**Benefits**:
- Predictive capability
- Enable promising new combos faster
- Better adaptation

---

## 12. Update Frequency Optimization

### Current Issue
- Fixed update frequency (50 trades or 1 hour)
- May be too frequent for small datasets
- May be too slow for large datasets

### Improvement: Adaptive Update Frequency
```python
def _calculate_update_frequency(self, total_trades: int) -> int:
    """
    Adjust update frequency based on data volume
    - Low volume (< 100 trades): Update every 100 trades or 4 hours
    - Medium volume (100-500): Update every 50 trades or 1 hour
    - High volume (> 500): Update every 25 trades or 30 minutes
    """
    if total_trades < 100:
        return 100, 4  # trades, hours
    elif total_trades > 500:
        return 25, 0.5
    else:
        return 50, 1
```

**Benefits**:
- Optimal update frequency
- Better resource usage
- Faster adaptation when needed

---

## Priority Recommendations

### High Priority (Implement First)
1. **Recency Weighting** - Adapts faster to market changes
2. **Confidence-Based Thresholds** - More statistically sound
3. **Performance Decay Detection** - Early warning system
4. **Combo Key Parsing** - Better efficiency and maintainability

### Medium Priority
5. **Adaptive Sample Size** - Better data utilization
6. **Symbol-Specific Tracking** - Better optimization
7. **Enhanced Metrics** - Better decision making
8. **Minimum Frequency Check** - Avoid useless combos

### Low Priority (Nice to Have)
9. **Session-Aware Performance** - Advanced optimization
10. **Combo Hierarchies** - Better organization
11. **ML Integration** - Predictive capability
12. **Adaptive Update Frequency** - Optimization

---

## Implementation Example

Here's how to implement recency weighting (highest priority):

```python
def _analyze_combo_performance(self, side: Optional[str] = None) -> Dict[str, ComboPerformance]:
    # ... existing code to collect trades ...
    
    # Apply exponential decay weighting
    now = datetime.utcnow()
    half_life_days = 7.0  # Configurable
    
    for key, trades in combos.items():
        total_weight = 0.0
        weighted_wins = 0.0
        weighted_rr = 0.0
        
        for trade in trades:
            age_days = (now - trade.exit_time).total_seconds() / 86400
            weight = 2 ** (-age_days / half_life_days)
            total_weight += weight
            if trade.outcome == 'win':
                weighted_wins += weight
            weighted_rr += trade.realized_rr * weight
        
        # Calculate weighted metrics
        wr = (weighted_wins / total_weight * 100) if total_weight > 0 else 0.0
        ev_r = (weighted_rr / total_weight) if total_weight > 0 else 0.0
        
        # Use weighted metrics for enable/disable decision
        ...
```

---

## Summary

The combo system is **well-implemented** but can be improved with:

1. **Statistical improvements**: Recency weighting, confidence-based thresholds
2. **Better tracking**: Symbol-specific, session-aware, performance decay
3. **Enhanced metrics**: Sharpe ratio, profit factor, risk-adjusted returns
4. **Efficiency**: Better combo key parsing, adaptive update frequency
5. **Advanced features**: ML prediction, combo hierarchies

**Recommended order**: Start with recency weighting and confidence-based thresholds for immediate impact, then add performance decay detection and enhanced metrics.

