"""
Cointegration Testing
=====================
Uses Engle-Granger two-step method to find cointegrated pairs.

Step 1: Regress P_A on P_B, get residuals
Step 2: Run ADF test on residuals — if stationary, pair is cointegrated

We use p-value < 0.05 as the cointegration threshold.
"""

import numpy as np
import pandas as pd
from itertools import combinations
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant


def test_cointegration(series_a: pd.Series, series_b: pd.Series) -> dict:
    """
    Run Engle-Granger cointegration test on two price series.

    Returns
    -------
    dict with keys: pvalue, score, is_cointegrated, hedge_ratio, half_life
    """
    # Engle-Granger test (statsmodels implementation)
    score, pvalue, _ = coint(series_a, series_b)

    # OLS hedge ratio (static — will be replaced by Kalman dynamically)
    X = add_constant(series_b.values)
    model = OLS(series_a.values, X).fit()
    hedge_ratio = model.params[1]
    intercept = model.params[0]

    # Compute static spread
    spread = series_a.values - hedge_ratio * series_b.values - intercept

    # Half-life of mean reversion via AR(1) fit on spread
    half_life = compute_half_life(spread)

    return {
        "pvalue": pvalue,
        "score": score,
        "is_cointegrated": pvalue < 0.05,
        "hedge_ratio": hedge_ratio,
        "intercept": intercept,
        "half_life": half_life,
    }


def compute_half_life(spread: np.ndarray) -> float:
    """
    Estimate half-life of mean reversion using OLS on AR(1) model:
        delta_spread = lambda * spread_{t-1} + epsilon
    half_life = -log(2) / lambda
    """
    spread_lag = spread[:-1]
    delta_spread = np.diff(spread)

    X = add_constant(spread_lag)
    model = OLS(delta_spread, X).fit()
    lam = model.params[1]

    if lam >= 0:
        return np.inf  # Not mean reverting

    half_life = -np.log(2) / lam
    return round(half_life, 1)


def find_cointegrated_pairs(
    prices: pd.DataFrame,
    pvalue_threshold: float = 0.05,
    min_half_life: float = 5.0,
    max_half_life: float = 120.0,
) -> pd.DataFrame:
    """
    Screen all pairs in a price DataFrame for cointegration.

    Parameters
    ----------
    prices           : DataFrame where each column is a stock's price series
    pvalue_threshold : cointegration p-value cutoff
    min_half_life    : minimum mean reversion half-life in days (filter noise)
    max_half_life    : maximum half-life in days (filter non-reverting pairs)

    Returns
    -------
    DataFrame of valid pairs sorted by p-value, with columns:
    asset_a, asset_b, pvalue, hedge_ratio, half_life
    """
    tickers = prices.columns.tolist()
    results = []

    print(f"Testing {len(list(combinations(tickers, 2)))} pairs...")

    for a, b in combinations(tickers, 2):
        series_a = prices[a].dropna()
        series_b = prices[b].dropna()

        # Align on common dates
        common_idx = series_a.index.intersection(series_b.index)
        if len(common_idx) < 252:  # Need at least 1 year of data
            continue

        result = test_cointegration(series_a[common_idx], series_b[common_idx])

        if (
            result["is_cointegrated"]
            and min_half_life <= result["half_life"] <= max_half_life
        ):
            results.append(
                {
                    "asset_a": a,
                    "asset_b": b,
                    "pvalue": round(result["pvalue"], 4),
                    "hedge_ratio": round(result["hedge_ratio"], 4),
                    "half_life_days": result["half_life"],
                }
            )

    if not results:
        print("No cointegrated pairs found. Try relaxing thresholds.")
        return pd.DataFrame()

    df = pd.DataFrame(results).sort_values("pvalue").reset_index(drop=True)
    print(f"\nFound {len(df)} cointegrated pairs:")
    print(df.to_string(index=False))
    return df


def adf_test(series: np.ndarray, label: str = "Series") -> dict:
    """
    Run Augmented Dickey-Fuller test.
    H0: series has a unit root (non-stationary)
    Reject H0 if p-value < 0.05 → series is stationary
    """
    result = adfuller(series, autolag="AIC")
    output = {
        "label": label,
        "adf_statistic": round(result[0], 4),
        "pvalue": round(result[1], 4),
        "is_stationary": result[1] < 0.05,
        "critical_values": result[4],
    }
    print(
        f"ADF Test [{label}]: stat={output['adf_statistic']}, "
        f"p={output['pvalue']}, stationary={output['is_stationary']}"
    )
    return output