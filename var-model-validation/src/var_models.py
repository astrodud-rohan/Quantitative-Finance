"""
Implements four VaR methods:
    1. Historical Simulation (HS) VaR -- THE CHAMPION MODEL (most common VaR approach)
    2. Parametric (Variance-Covariance) VaR 
    3. EWMA VaR (RiskMetrics-style) -- CHALLENGER 1
    4. Filtered Historical Simulation (FHS) VaR -- CHALLENGER 2
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

CONFIDENCE_LEVELS = [0.95, 0.99]

def historical_simulation_var(returns: pd.Series, window=250, confidence=0.99):
    var_estimates = pd.Series(index=returns.index, dtype=float)
    alpha = 1 - confidence
    for t in range(window, len(returns)):
        window_returns = returns.iloc[t - window:t]
        var_estimates.iloc[t] = -np.percentile(window_returns, alpha * 100)
    return var_estimates

def parametric_var(returns: pd.Series, window=250, confidence=0.99):
    var_estimates = pd.Series(index=returns.index, dtype=float)
    z = norm.ppf(1- confidence)
    for t in range(window, len(returns)):
        window_returns = returns.iloc[t - window:t]
        mu, sigma = window_returns.mean(), window_returns.std()
        var_estimates.iloc[t] = -(mu + z * sigma)
    return var_estimates

def ewma_var(returns: pd.Series, lam=0.94, confidence=0.99, min_periods=30):
    var_estimates = pd.Series(index=returns.index, dtype=float)
    z = norm.ppf(1 - confidence)

    r = returns.values
    n = len(r)
    sigma2 = np.zeros(n)
    sigma2[min_periods] = np.var(r[:min_periods])

    for t in range(min_periods + 1, n):
        sigma2[t] = lam * sigma2[t - 1] + (1 - lam) * r[t - 1] ** 2

    for t in range(min_periods, n):
        sigma = np.sqrt(sigma2[t])
        var_estimates.iloc[t] = -(z * sigma)
    return var_estimates

def filtered_historical_simulation_var(returns: pd.Series, window=250, lam=0.94, confidence=0.99, min_periods=30):
    r = returns.values
    n = len(r)
    sigma2 = np.zeros(n)
    sigma2[min_periods] = np.var(r[:min_periods])
    for t in range(min_periods + 1, n):
        sigma2[t] = lam * sigma2[t - 1] + (1 - lam) * r[t - 1] ** 2
    sigma = np.sqrt(sigma2)
    sigma[sigma == 0] = np.nan

    standardized = r / sigma

    var_estimates = pd.Series(index=returns.index, dtype=float)
    alpha = 1 - confidence
    start = max(window, min_periods + 1)
    for t in range(start, n):
        z_window = standardized[t - window:t]
        z_window = z_window[~np.isnan(z_window)]
        pct = -np.percentile(z_window, alpha * 100)
        var_estimates.iloc[t] = pct * sigma[t]
    return var_estimates

def compute_all_var_models(df, window=250, confidence=0.99):
    out = df.copy()
    out["hs_var"] = historical_simulation_var(df["log_return"], window=window, confidence=confidence)
    out["parametric_var"] = parametric_var(df["log_return"], window=window, confidence=confidence)
    out["ewma_var"] = ewma_var(df["log_return"], confidence=confidence)
    out["fhs_var"] = filtered_historical_simulation_var(df["log_return"], window=window, confidence=confidence)

    out["realized_loss"] = -df["log_return"]
    return out

if __name__ == "__main__":
    df = pd.read_csv("../data/raw/index_prices.csv", parse_dates=["date"])
    scored = compute_all_var_models(df)
    scored.to_csv("../data/processed/var_estimates.csv", index=False)
    print(f"Computed VaR estimates for {scored['hs_var'].notna().sum()} days.")
    print(scored[["date", "log_return", "hs_var", "parametric_var", "ewma_var", "fhs_var"]].dropna().tail()) 