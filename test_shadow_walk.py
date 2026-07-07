"""Unit tests for shadow_logger.walk_outcome (pure candle walk).

Run: python3 test_shadow_walk.py   (or pytest test_shadow_walk.py)
"""
from autobot.core.shadow_logger import walk_outcome

# candle = (ts, open, high, low, close); entry=100, sl=95, tp=110 (rr=2) for longs
E, SL, TP, RR = 100.0, 95.0, 110.0, 2.0


def bar(h, l, ts=0):
    return (ts, (h + l) / 2, h, l, (h + l) / 2)


def test_long_clean_win():
    candles = [bar(105, 99), bar(111, 100), bar(120, 105)]
    st, r, mfe, mae, m2s, bars = walk_outcome(candles, E, SL, TP, 'long', RR)
    assert st == 'win' and r == RR
    assert abs(mfe - (111 - E) / 5) < 1e-9          # frozen at outcome bar
    assert abs(m2s - (120 - E) / 5) < 1e-9          # keeps growing past TP
    assert bars == 2


def test_long_clean_loss():
    candles = [bar(103, 98), bar(102, 94)]
    st, r, mfe, mae, m2s, bars = walk_outcome(candles, E, SL, TP, 'long', RR)
    assert st == 'loss' and r == -1.0
    assert abs(mfe - (103 - E) / 5) < 1e-9
    assert abs(mae - (94 - E) / 5) < 1e-9           # SL bar included in mae (legacy)
    assert abs(m2s - (103 - E) / 5) < 1e-9          # SL bar EXCLUDED from mfe_to_sl
    assert bars == 2


def test_ambiguous_bar_is_loss_and_excluded_from_m2s():
    candles = [bar(112, 94)]                        # hits TP and SL same bar
    st, r, mfe, mae, m2s, bars = walk_outcome(candles, E, SL, TP, 'long', RR)
    assert st == 'loss' and r == -1.0               # SL wins ties
    assert m2s == 0.0                                # ambiguous bar excluded
    assert bars == 1


def test_win_then_sl_stops_m2s():
    candles = [bar(111, 100), bar(115, 100), bar(118, 94), bar(200, 150)]
    st, r, mfe, mae, m2s, bars = walk_outcome(candles, E, SL, TP, 'long', RR)
    assert st == 'win' and bars == 1
    assert abs(m2s - (115 - E) / 5) < 1e-9          # SL bar (118 high) excluded; walk stops
    assert abs(mfe - (111 - E) / 5) < 1e-9          # frozen at outcome bar


def test_expired_no_hits():
    candles = [bar(105, 98)] * 10
    st, r, mfe, mae, m2s, bars = walk_outcome(candles, E, SL, TP, 'long', RR)
    assert st == 'expired' and r == 0.0
    assert bars == 10
    assert abs(m2s - (105 - E) / 5) < 1e-9


def test_short_win():
    # short: entry=100, sl=105, tp=90 (rr=2)
    candles = [bar(102, 96), bar(101, 89), bar(99, 80)]
    st, r, mfe, mae, m2s, bars = walk_outcome(candles, 100.0, 105.0, 90.0, 'short', 2.0)
    assert st == 'win' and r == 2.0 and bars == 2
    assert abs(m2s - (100 - 80) / 5) < 1e-9


def test_short_loss_ambiguous():
    candles = [bar(106, 89)]                        # both hit -> SL wins
    st, r, *_ = walk_outcome(candles, 100.0, 105.0, 90.0, 'short', 2.0)
    assert st == 'loss' and r == -1.0


def test_bad_risk_returns_expired():
    st, r, mfe, mae, m2s, bars = walk_outcome([bar(105, 99)], 100.0, 100.0, 110.0, 'long', 2.0)
    assert st == 'expired' and bars == 0


def test_horizon_cap():
    candles = [bar(105, 98)] * 300
    st, r, mfe, mae, m2s, bars = walk_outcome(candles, E, SL, TP, 'long', RR, horizon=200)
    assert bars == 200


def test_legacy_parity_loss_path():
    """mfe/mae must match the legacy inline loop for a mixed path ending in SL."""
    candles = [bar(108, 99), bar(109, 97), bar(104, 94)]
    st, r, mfe, mae, m2s, bars = walk_outcome(candles, E, SL, TP, 'long', RR)
    # legacy loop reproduction
    lmfe = lmae = 0.0
    for _, o, h, l, c in candles:
        lmfe = max(lmfe, (h - E) / 5)
        lmae = min(lmae, (l - E) / 5)
        if l <= SL:
            break
    assert st == 'loss'
    assert abs(mfe - lmfe) < 1e-12 and abs(mae - lmae) < 1e-12


if __name__ == '__main__':
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL {fn.__name__}: {e}")
    print(f"{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
