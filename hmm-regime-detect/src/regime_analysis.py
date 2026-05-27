"""
Regime Analysis Module
======================
Computes regime-conditional statistics and compares regime-aware
vs regime-blind strategy performance.

Two analyses:
  1. Regime-conditional VaR / CVaR
     — shows risk is dramatically underestimated by ignoring regimes
     — bear regime VaR is 3-5x higher than bull regime VaR

  2. Regime-conditional strategy
     — simple momentum strategy, only active in bull regime
     — compare Sharpe with vs without regime filter
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.stats import norm


@dataclass
class RegimeStats:
    """Statistics for a single regime."""
    label: str
    n_days: int
    pct_time: float
    mean_daily_return: float
    daily_vol: float
    annualised_return: float
    annualised_vol: float
    sharpe: float
    skewness: float
    kurtosis: float
    var_95: float   # 1-day 95% VaR (negative = loss)
    cvar_95: float  # 1-day 95% CVaR (Expected Shortfall)
    var_99: float
    avg_persistence: float  # avg consecutive days in regime


def compute_regime_stats(
    returns: np.ndarray,
    states: np.ndarray,
    n_states: int = 3,
    rf_daily: float = 0.065 / 252,
) -> list[RegimeStats]:
    """
    Compute descriptive statistics for each detected regime.

    Parameters
    ----------
    returns  : daily log returns
    states   : Viterbi-decoded state sequence (same length as returns)
    n_states : number of HMM states
    rf_daily : daily risk-free rate (default: 6.5% annual / 252)
    """
    labels = {0: "Bear", 1: "Sideways", 2: "Bull"}
    if n_states == 2:
        labels = {0: "Bear/High-Vol", 1: "Bull/Low-Vol"}

    all_stats = []

    for k in range(n_states):
        mask = states == k
        r = returns[mask]

        if len(r) < 5:
            continue

        n_days = mask.sum()
        pct_time = n_days / len(returns) * 100

        mean_d = r.mean()
        vol_d = r.std()
        ann_ret = mean_d * 252
        ann_vol = vol_d * np.sqrt(252)
        sharpe = (mean_d - rf_daily) / vol_d * np.sqrt(252) if vol_d > 0 else 0

        # Higher moments
        from scipy.stats import skew, kurtosis
        skewness = skew(r)
        kurt = kurtosis(r)  # excess kurtosis

        # Historical VaR and CVaR
        var_95 = np.percentile(r, 5)    # 5th percentile = 95% VaR
        var_99 = np.percentile(r, 1)
        cvar_95 = r[r <= var_95].mean() if (r <= var_95).sum() > 0 else var_95

        # Average persistence: mean run length in this regime
        runs = []
        count = 0
        for s in states:
            if s == k:
                count += 1
            elif count > 0:
                runs.append(count)
                count = 0
        if count > 0:
            runs.append(count)
        avg_persistence = np.mean(runs) if runs else 0

        all_stats.append(RegimeStats(
            label=labels.get(k, f"State {k}"),
            n_days=int(n_days),
            pct_time=round(pct_time, 1),
            mean_daily_return=round(mean_d * 100, 4),
            daily_vol=round(vol_d * 100, 4),
            annualised_return=round(ann_ret * 100, 2),
            annualised_vol=round(ann_vol * 100, 2),
            sharpe=round(sharpe, 3),
            skewness=round(skewness, 3),
            kurtosis=round(kurt, 3),
            var_95=round(var_95 * 100, 4),
            cvar_95=round(cvar_95 * 100, 4),
            var_99=round(var_99 * 100, 4),
            avg_persistence=round(avg_persistence, 1),
        ))

    return all_stats


def print_regime_stats(stats: list[RegimeStats]):
    """Pretty-print regime statistics table."""
    print("\n" + "═" * 80)
    print("  REGIME-CONDITIONAL STATISTICS")
    print("═" * 80)
    header = f"{'Metric':<30}" + "".join(f"{s.label:>16}" for s in stats)
    print(header)
    print("─" * 80)

    rows = [
        ("Days in regime", "n_days", "d"),
        ("% of time", "pct_time", "f"),
        ("Mean daily return (%)", "mean_daily_return", "f"),
        ("Daily volatility (%)", "daily_vol", "f"),
        ("Annualised return (%)", "annualised_return", "f"),
        ("Annualised vol (%)", "annualised_vol", "f"),
        ("Sharpe ratio", "sharpe", "f"),
        ("Skewness", "skewness", "f"),
        ("Excess kurtosis", "kurtosis", "f"),
        ("95% VaR (1-day %)", "var_95", "f"),
        ("95% CVaR / ES (1-day %)", "cvar_95", "f"),
        ("99% VaR (1-day %)", "var_99", "f"),
        ("Avg persistence (days)", "avg_persistence", "f"),
    ]

    for label, attr, fmt in rows:
        vals = "".join(
            f"{getattr(s, attr):>16d}" if fmt == "d"
            else f"{getattr(s, attr):>16.3f}"
            for s in stats
        )
        print(f"{label:<30}{vals}")

    print("═" * 80)


def regime_conditional_var(
    returns: np.ndarray,
    states: np.ndarray,
    current_state: int,
    confidence: float = 0.95,
) -> dict:
    """
    Compute VaR and CVaR for the CURRENT regime.
    This is what a risk desk would use for real-time risk monitoring.

    Returns
    -------
    dict with var, cvar, and comparison to unconditional VaR
    """
    alpha = 1 - confidence

    # Unconditional (regime-blind) VaR
    unconditional_var = np.percentile(returns, alpha * 100)
    unconditional_cvar = returns[returns <= unconditional_var].mean()

    # Regime-conditional VaR
    r_regime = returns[states == current_state]
    if len(r_regime) < 10:
        conditional_var = unconditional_var
        conditional_cvar = unconditional_cvar
    else:
        conditional_var = np.percentile(r_regime, alpha * 100)
        conditional_cvar = r_regime[r_regime <= conditional_var].mean()

    labels = {0: "Bear", 1: "Sideways", 2: "Bull"}

    return {
        "current_regime": labels.get(current_state, f"State {current_state}"),
        "confidence": confidence,
        "conditional_var": round(conditional_var * 100, 4),
        "conditional_cvar": round(conditional_cvar * 100, 4),
        "unconditional_var": round(unconditional_var * 100, 4),
        "unconditional_cvar": round(unconditional_cvar * 100, 4),
        "var_ratio": round(conditional_var / unconditional_var, 3),
    }


def regime_conditional_strategy(
    returns: pd.Series,
    states: np.ndarray,
    bull_state: int = 2,
    bear_state: int = 0,
) -> dict:
    """
    Compare three strategies:
      1. Buy and hold (always long)
      2. Regime-blind momentum (always follow 20d momentum)
      3. Regime-aware (momentum only in bull regime, flat in bear)

    Returns
    -------
    dict with cumulative returns and Sharpe for each strategy
    """
    r = returns.values
    n = len(r)

    # ── Strategy 1: Buy and Hold ─────────────────────────────────────────
    bh_cumret = np.cumprod(1 + r) - 1

    # ── Strategy 2: Regime-blind momentum ───────────────────────────────
    window = 20
    momentum_signal = np.zeros(n)
    for i in range(window, n):
        cum_ret = r[i-window:i].sum()
        momentum_signal[i] = 1 if cum_ret > 0 else -1

    momentum_ret = momentum_signal * r
    momentum_cumret = np.cumprod(1 + momentum_ret) - 1

    # ── Strategy 3: Regime-aware momentum ───────────────────────────────
    regime_signal = np.zeros(n)
    for i in range(window, n):
        if states[i] == bull_state:
            # In bull regime: follow momentum
            cum_ret = r[i-window:i].sum()
            regime_signal[i] = 1 if cum_ret > 0 else 0
        elif states[i] == bear_state:
            # In bear regime: go flat or slight short
            regime_signal[i] = 0
        else:
            # Sideways: reduced position
            cum_ret = r[i-window:i].sum()
            regime_signal[i] = 0.5 if cum_ret > 0 else 0

    regime_ret = regime_signal * r
    regime_cumret = np.cumprod(1 + regime_ret) - 1

    def sharpe(rets, rf=0.065/252):
        excess = rets - rf
        return (excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0

    def max_dd(cumrets):
        wealth = 1 + cumrets
        rolling_max = np.maximum.accumulate(wealth)
        dd = (wealth - rolling_max) / rolling_max
        return dd.min() * 100

    rf_daily = 0.065 / 252

    results = {
        "buy_and_hold": {
            "label": "Buy & Hold",
            "final_return_pct": round(bh_cumret[-1] * 100, 2),
            "sharpe": round(sharpe(r), 3),
            "max_drawdown_pct": round(max_dd(bh_cumret), 2),
            "cumulative_returns": bh_cumret,
        },
        "momentum_blind": {
            "label": "Momentum (Regime-Blind)",
            "final_return_pct": round(momentum_cumret[-1] * 100, 2),
            "sharpe": round(sharpe(momentum_ret), 3),
            "max_drawdown_pct": round(max_dd(momentum_cumret), 2),
            "cumulative_returns": momentum_cumret,
        },
        "momentum_regime": {
            "label": "Momentum (Regime-Aware)",
            "final_return_pct": round(regime_cumret[-1] * 100, 2),
            "sharpe": round(sharpe(regime_ret), 3),
            "max_drawdown_pct": round(max_dd(regime_cumret), 2),
            "cumulative_returns": regime_cumret,
        },
    }

    return results


def print_strategy_comparison(strategy_results: dict):
    """Print strategy comparison table."""
    print("\n" + "═" * 65)
    print("  REGIME-AWARE vs REGIME-BLIND STRATEGY COMPARISON")
    print("═" * 65)
    print(f"  {'Strategy':<30} {'Return%':>9} {'Sharpe':>8} {'MaxDD%':>9}")
    print("─" * 65)
    for key, s in strategy_results.items():
        print(f"  {s['label']:<30} {s['final_return_pct']:>9.2f} "
              f"{s['sharpe']:>8.3f} {s['max_drawdown_pct']:>9.2f}")
    print("═" * 65)
