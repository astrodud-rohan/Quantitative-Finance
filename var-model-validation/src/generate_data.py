"""
Generates a synthetic daily equity index price series with:
    - GARCH(1,1)-like volatility clustering
    - Student-t innovations
    - injection of ~6-week Volatility Stress Regime partway through the sample, imitating market shock.
"""

import numpy as np
import pandas as pd
from datetime import date, timedelta

RANDOM_SEED = 7
N_DAYS = 1500 # ~6 years of trading days
START_PRICE = 4000.0

def generate_synthetic_index(n=N_DAYS, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)

    omega, alpha, beta = 0.000003, 0.08, 0.90
    long_run_var = omega / (1 - alpha - beta)

    sigma2 = np.zeros(n)
    eps = np.zeros(n)
    sigma2[0] = long_run_var

    t_dof = 5
    raw_t = rng.standard_t(t_dof, size=n)
    raw_t = raw_t / np.sqrt(t_dof / (t_dof - 2))

    daily_drift = 0.0003

    stress_start, stress_end = 900, 940
    vol_multiplier = np.ones(n)
    vol_multiplier[stress_start:stress_end] = 3.5

    decay_len = 30

    for i, d in enumerate(range(stress_end, stress_end + decay_len)):
        if d < n:
            vol_multiplier[d] = 3.5 - (3.5 -1.0) * (i / decay_len)
    
    returns = np.zeros(n)
    base_eps = np.zeros(n)
    for t in range(n):
        if t > 0:
            sigma2[t] = omega + alpha * base_eps[t - 1] ** 2 + beta * sigma2[t - 1]
        sigma_t = np.sqrt(sigma2[t])
        base_eps[t] = sigma_t * raw_t[t]

        eps[t] = base_eps[t] * vol_multiplier[t]
        returns[t] = daily_drift + eps[t]
    
    prices = START_PRICE * np.exp(np.cumsum(returns))

    start_date = date(2020, 1, 2)
    dates = []
    d = start_date
    while len(dates) < n:
        if d.weekday() < 5:
            dates.append(d)
        d += timedelta(days=1)
    
    df = pd.DataFrame({
        "date": dates,
        "price": prices,
        "log_return": returns,
    })
    return df, (stress_start, stress_end + decay_len)

if __name__ == "__main__":
    df, stress_window = generate_synthetic_index()
    out_path = "../data/raw/index_prices.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df):,} trading days -> {out_path}")
    print(f"Data range: {df['date'].min()} to {df['date'].max()}")
    print(f"Annualized vol (full sample): {df['log_return'].std() * np.sqrt(252):.4f}")
    print(f"Stress window (row indices): {stress_window}"
          f"-> dates {df['date'].iloc[stress_window[0]]} to {df['date'].iloc[min(stress_window[1], len(df) - 1)]}")
