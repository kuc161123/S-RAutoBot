# Boom Hunter Pro Strategy Analysis

## Overview
Boom Hunter Pro uses Ehlers Early Onset Trend (EOT) oscillators with:
- Highpass filter for cyclic component removal
- SuperSmoother filter for noise reduction
- Fast Attack/Slow Decay for normalization

## Core Components

### EOT 1 (Main Blue Oscillator)
- LPPeriod = 6, K1 = 0, K2 = 0.3
- Creates Quotient1 and Quotient2
- Trigger = SMA(Quotient1, 2)
- Range: -1 to +1, mapped to 0-100 for display

### EOT 2 (Red Wave)
- LPPeriod2 = 27, K12 = 0.8, K22 = 0.3
- Creates Quotient3 and Quotient4
- Used for overbought/exit warnings

### EOT 3 (Yellow Line)
- LPPeriod3 = 11, K13 = 0.99
- Creates Quotient5 and Quotient6
- Used for "Boom" detection (sudden moves)

### LSMA Wave Trend
- Combines: TCI, CSI, MFI, Willy Williams
- wt1 = main wave, wt2 = SMA(wt1, 6)
- wt2 < 20 = oversold pressure
- wt2 > 80 = overbought pressure

## Signal Conditions

### LONG Entries (4 quality levels):

1. **Lime (Best)** - `enter3`:
   - Quotient3 <= -0.9
   - crossover(q1, trigger)
   - barssince(warn2) <= 7
   - q1 <= 20
   - barssince(crossover(q1, 20)) <= 21

2. **Blue** - `enter5`:
   - barssince(q1 <= 0 AND crossunder(q1, trigger)) <= 5
   - crossover(q1, trigger)

3. **Gray** - `enter6`:
   - barssince(q1 <= 20 AND crossunder(q1, trigger)) <= 11
   - crossover(q1, trigger)
   - q1 <= 60

4. **Yellow** - `enter7`:
   - Quotient3 <= -0.9
   - crossover(q1, trigger)

### SHORT Entry:
- **Red** - `senter3`:
  - Quotient3 >= -0.9
  - crossunder(q1, trigger)
  - barssince(warn3) <= 7
  - q1 >= 99
  - barssince(crossover(q1, 80)) <= 21

### Exit Warnings:
- Orange: cross(Quotient5, Quotient6) AND Quotient5 > 0.5
- Red "Overbought": cross(Quotient3, Quotient4) AND Quotient3 > 0

## Key Thresholds
- Quotient <= -0.9: Oversold zone (entry zone)
- Quotient >= 0.9: Overbought zone
- q1 <= 20: Deep oversold
- q1 >= 80: Deep overbought
- wt2 <= 20: Pressure for longs
- wt2 >= 80: Pressure for shorts
