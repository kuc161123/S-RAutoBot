"""Statistics, including the multiple-testing correction.

Reuses the maths already validated in verify_overfit.py (binomial survival, bootstrap CI,
Deflated Sharpe per Bailey & Lopez de Prado) rather than reimplementing it. The Deflated
Sharpe is the part that matters most here: with N = 147 pre-registered trials, the expected
maximum Sharpe under the null is substantial, and an uncorrected "best strategy" is close to
meaningless.
"""
from __future__ import annotations

import math

import numpy as np


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _phi_inv(p: float) -> float:
    """Acklam's inverse normal CDF — adequate precision, no scipy dependency."""
    if p <= 0.0:
        return -np.inf
    if p >= 1.0:
        return np.inf
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    pl, ph = 0.02425, 1 - 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > ph:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def stats(r: np.ndarray) -> dict:
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return dict(n=0, wr=0.0, avg=0.0, tot=0.0, pf=0.0, sharpe=0.0, max_dd=0.0, se=0.0)
    gp = r[r > 0].sum()
    gl = abs(r[r < 0].sum())
    cum = np.cumsum(r)
    peak = np.maximum.accumulate(cum)
    sd = r.std(ddof=1) if len(r) > 1 else 0.0
    return dict(
        n=int(len(r)), wr=float((r > 0).mean()), avg=float(r.mean()),
        tot=float(cum[-1]), pf=float(gp / gl) if gl > 0 else float("inf"),
        sharpe=float(r.mean() / sd) if sd > 0 else 0.0,
        max_dd=float(abs((cum - peak).min())),
        se=float(sd / math.sqrt(len(r))) if len(r) > 0 else 0.0,
    )


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample t for strategy-vs-placebo. Unequal variance, unequal n."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return 0.0
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    denom = math.sqrt(va + vb)
    return float((a.mean() - b.mean()) / denom) if denom > 0 else 0.0


def bootstrap_ci(r: np.ndarray, n_boot: int = 5000, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float, float, float]:
    """(lo, hi, P(mean>=0)) on the mean, by resampling with replacement."""
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    if len(r) < 20:
        return (0.0, 0.0, 0.5)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(r), size=(n_boot, len(r)))
    means = r[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi), float((means >= 0).mean())


def expected_max_sharpe(n_trials: int, sd_trials: float = 1.0) -> float:
    """Expected MAXIMUM Sharpe from luck alone as the best of n_trials.

    Bailey & Lopez de Prado's SR0. The `sd_trials` scaling is essential and easy to get
    wrong: the expected maximum of n draws is in units of the CROSS-SECTIONAL standard
    deviation of the trial Sharpes, not units of 1. Passing sd_trials=1 (as a naive
    implementation does) makes the bar absurdly high and would reject everything.
    """
    if n_trials <= 1:
        return 0.0
    e = 0.5772156649
    z1 = _phi_inv(1 - 1.0 / n_trials)
    z2 = _phi_inv(1 - 1.0 / (n_trials * math.e))
    return float(sd_trials * ((1 - e) * z1 + e * z2))


def deflated_sharpe(sharpe: float, n_obs: int, n_trials: int, sd_trials: float,
                    skew: float = 0.0, kurt: float = 3.0) -> float:
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014).

    Returns P(true Sharpe > 0) after correcting for the fact that the observed Sharpe is
    the MAXIMUM over n_trials searches. `sd_trials` is the cross-sectional std of all
    trial Sharpes and can only be computed once every trial has run — so this is called at
    scoring time, not per-strategy.

    This is the guard against exactly the failure that produced the current live config:
    a +15.7 sigma in-sample result that was the best of ~60,000 uncorrected tests and
    transferred as nothing.
    """
    if n_obs < 2 or n_trials < 1 or not np.isfinite(sharpe):
        return 0.0
    sr0 = expected_max_sharpe(n_trials, sd_trials)
    denom = math.sqrt(max(1e-12, 1 - skew * sharpe + ((kurt - 1) / 4) * sharpe ** 2))
    z = (sharpe - sr0) * math.sqrt(max(1, n_obs - 1)) / denom
    return float(_phi(z))


def breakeven_wr(rr: float, cost_r: float = 0.0) -> float:
    """Win rate needed to break even at a given reward:risk, including cost drag."""
    return (1.0 + cost_r) / (1.0 + rr)
