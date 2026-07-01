"""
The Core Deliverable of the Project
Implements three pillars of regulatory VaR backtesting:
    1. Kupiec's Proportion of Failures (POF) test
    2. Christoffersen's Independence Test
    3. Basel Traffic-Light Approach
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2

def kupiec_pof_test(n_obs, n_breaches, confidence=0.99):
    p = 1 - confidence
    x = n_breaches
    n = n_obs
    pi_hat = x / n if n > 0 else 0

    if x == 0:
        log_lik_null = n * np.log(1 - p)
        log_lik_alt = 0.0
    elif x == n:
        log_lik_null = n * np.log(p)
        log_lik_alt = 0.0
    else:
        log_lik_null = (n - x) * np.log(1 - p) + x * np.log(p)
        log_lik_alt = (n - x) * np.log(1 - pi_hat) + x * np.log(pi_hat)
    
    lr_pof = -2 * (log_lik_null - log_lik_alt)
    p_value = 1 - chi2.cdf(lr_pof, df=1)

    expected_breaches = n * p
    return{
        "n_obs": n,
        "n_breaches": x,
        "expected_breaches": round(expected_breaches, 2),
        "breach_rate": round(pi_hat, 4),
        "expected_rate": p,
        "lr_stat": round(lr_pof, 4),
        "p_value": round(p_value, 4),
        "reject_h0": p_value < 0.05,
    }

def christoffersen_independence_test(breach_series: pd.Series):
    b = breach_series.astype(int).values
    n00 = n01 = n10 = n11 = 0
    for i in range(1, len(b)):
        prev, curr = b[i - 1], b[i]
        if prev == 0 and curr == 0:
            n00 += 1
        elif prev == 0 and curr == 1:
            n01 += 1
        elif prev == 1 and curr == 0:
            n10 += 1
        else:
            n11 += 1
    
    n0, n1 = n00 + n01, n10 + n11
    pi01 = n01 / n0 if n0 > 0 else 0
    pi11 = n11 / n1 if n1 > 0 else 0
    pi = (n01 + n11) / (n0 + n1) if (n0 + n1) > 0 else 0

    def safe_log_lik(n_a, n_b, prob):
        ll = 0.0
        if n_a > 0:
            ll += n_a * np.log(1 - prob) if prob < 1 else 0
        if n_b > 0:
            ll += n_b * np.log(prob) if prob > 0 else 0
        return ll
    
    log_lik_null = safe_log_lik(n00 + n10, n01 + n11, pi)
    log_lik_alt = safe_log_lik(n00, n01, pi01) + safe_log_lik(n10, n11, pi11)

    lr_ind = -2 * (log_lik_null - log_lik_alt)
    lr_ind = max(lr_ind, 0.0)
    p_value = 1 - chi2.cdf(lr_ind, df=1)

    return{
        "n00": n00, "n01": n01, "n10": n10, "n11": n11,
        "pi01": round(pi01, 4), "pi11": round(pi11, 4),
        "lr_stat": round(lr_ind, 4),
        "p_value": round(p_value, 4),
        "reject_h0": p_value < 0.05,
    }

def christoffersen_joint_test(kupiec_result, independence_result):
    lr_cc = kupiec_result["lr_stat"] + independence_result["lr_stat"]
    p_value = 1 - chi2.cdf(lr_cc, df=2)
    return{"lr_stat": round(lr_cc, 4), "p_value": round(p_value, 4), "reject_h0": p_value < 0.05}

def basel_traffic_light(n_breaches_250d, confidence=0.99):
    if confidence != 0.99:
        return{"zone": "N/A", "note": "Traffic-light test is defined for 99% VaR only."}
    
    amber_addons = {5: 0.40, 6: 0.50, 7: 0.65, 8: 0.75, 9: 0.85}

    if n_breaches_250d <= 4:
        zone, addon = "Green", 0.00
    elif n_breaches_250d <= 9:
        zone, addon = "Amber", amber_addons[n_breaches_250d]
    else:
        zone, addon = "Red", 1.00
    
    return{"n_breaches_250d": n_breaches_250d, "zone": zone, "capital_multiplier_addon": addon}

def rolling_250d_breach_count(breach_series: pd.Series):
    return breach_series.astype(int).rolling(250).sum()

if __name__ == "__main__":
    df = pd.read_csv("../data/processed/var_estimates.csv", parse_dates=["date"])
    df = df.dropna(subset=["hs_var"]).reset_index(drop=True)

    models = ["hs_var", "parametric_var", "ewma_var", "fhs_var"]
    print("=== KUPIEC POF TEST (full sample, 99% VaR) ===")
    for m in models:
        breach = df["realized_loss"] > df[m]
        result = kupiec_pof_test(len(df), breach.sum(), confidence=0.99)
        print(f"{m:16s} | breaches={result['n_breaches']:3d} expected={result['expected_breaches']:.1f}"
              f" LR={result['lr_stat']:.2f} p={result['p_value']:.4f} reject={result['reject_h0']}")
    
    print("\n=== CHRISTOFFERSEN INDEPENDENCE TEST ===")
    for m in models:
        breach = df["realized_loss"] > df[m]
        result = christoffersen_independence_test(breach)
        print(f"{m:16s} | LR={result['lr_stat']:.2f} p={result['p_value']:.4f} reject={result['reject_h0']}"
              f" (pi01={result['pi01']}, pi11={result['pi11']})")
    
    print("\n=== BASEL TRAFFIC-LIGHT (most recent 250-day window) ===")
    for m in models:
        breach = df["realized_loss"] > df[m]
        rolling_count = rolling_250d_breach_count(breach)
        latest_count = int(rolling_count.iloc[-1])
        tl = basel_traffic_light(latest_count)
        print(f"{m:16s} | {tl}")
