"""
Validation Metrics - Core Deliverable
- plays the role of MODEL RISK MANAGEMENT
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import chi2, norm

def gini_coefficient(y_true, y_pred_proba):
    auc = roc_auc_score(y_true, y_pred_proba)
    return 2 * auc - 1, auc

def ks_statistic(y_true, y_pred_proba):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    ks = np.max(np.abs(tpr - fpr))
    return ks

def decile_calibration_table(y_true, y_pred_proba, n_bins=10):
    df = pd.DataFrame({"y": y_true, "pred": y_pred_proba})
    df["decile"] = pd.qcut(df["pred"], q=n_bins, duplicates="drop")

    table = df.groupby("decile").agg(
        n=("y", "count"),
        actual_default_rate=("y", "mean"),
        avg_predicted_pd=("pred", "mean"),
    ).reset_index()
    table["abs_gap"] = (table["actual_default_rate"] - table["avg_predicted_pd"]).abs()
    return table

def hosmer_lemeshow_test(y_true, y_pred_proba, n_bins=10):
    df = pd.DataFrame({"y": y_true, "pred": y_pred_proba})
    df["decile"] = pd.qcut(df["pred"], q=n_bins, duplicates="drop")

    hl_stat = 0.0
    for _, group in df.groupby("decile"):
        n = len(group)
        observed = group["y"].sum()
        expected = group["pred"].sum()
        var = group["pred"].mean() * (1 - group["pred"].mean()) * n
        if var > 0:
            hl_stat += (observed - expected) ** 2 / (expected * (1 - group["pred"].mean()) + 1e-9)
    
    dof = n_bins -2
    p_value = 1 - chi2.cdf(hl_stat, dof)
    return hl_stat, p_value, dof

def population_stability_index(expected, actual, n_bins=10):
    breakpoints = np.quantile(expected, np.linspace(0, 1, n_bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    expected_counts, _ = np.histogram(expected, bins=breakpoints)
    actual_counts, _ = np.histogram(actual, bins=breakpoints)

    expected_pct = expected_counts / len(expected) + 1e-6
    actual_pct = actual_counts / len(actual) + 1e-6

    psi_per_bin = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    return np.sum(psi_per_bin)

def characteristic_stability_index(train_df, oot_df, features, n_bins=10):
    results = []
    for feat in features:
        psi = population_stability_index(
            train_df[feat].dropna().values,
            oot_df[feat].dropna().values,
            n_bins=n_bins,
        )
        results.append({"feature": feat, "csi": psi})
    return pd.DataFrame(results).sort_values("csi", ascending=False)

def binomial_backtest(n_obligors, observed_defaults, predicted_pd, confidence=0.95):
    expected_defaults = n_obligors * predicted_pd
    std_dev = np.sqrt(n_obligors * predicted_pd * (1 - predicted_pd))
    if std_dev == 0:
        z_score = 0
    else:
        z_score = (observed_defaults - expected_defaults) / std_dev
    
    p_value = 1 - norm.cdf(z_score)

    if p_value < 0.01:
        flag = "Red"
    elif p_value < 0.05:
        flag = "Amber"
    else:
        flag = "Green"
    
    return{
        "expected_defaults": expected_defaults,
        "observed_defaults": observed_defaults,
        "z_score": z_score,
        "p_value": p_value,
        "flag": flag, 
    }

def grade_level_backtest(df, grade_col, target_col, pred_col):
    results = []
    for grade, group in df.groupby(grade_col):
        n = len(group)
        observed = group[target_col].sum()
        avg_pred_pd = group[pred_col].mean()
        test = binomial_backtest(n, observed, avg_pred_pd)
        test["grade"] = grade
        test["n_obligors"] = n
        results.append(test)
    return pd.DataFrame(results)

if __name__ == "__main__":
    train = pd.read_csv("../data/processed/train_scored.csv")
    test = pd.read_csv("../data/processed/test_scored.csv")
    oot = pd.read_csv("../data/processed/oot_scored.csv")

    print("=== DISCRIMINATORY POWER ===")
    for name, d in [("Train", train), ("Test", test), ("OOT", oot)]:
        gini, auc = gini_coefficient(d["SeriousDlqin2yrs"], d["predicted_pd"])
        ks = ks_statistic(d["SeriousDlqin2yrs"], d["predicted_pd"])
        print(f"{name:6s} | AUC: {auc:.4f} | Gini: {gini:.4f} | KS: {ks:.4f}")

    print("\n=== CALIBRATION (OOT decile table) ===")
    cal_table = decile_calibration_table(oot["SeriousDlqin2yrs"], oot["predicted_pd"])
    print(cal_table.to_string(index=False))

    hl_stat, p_val, dof = hosmer_lemeshow_test(oot["SeriousDlqin2yrs"], oot["predicted_pd"])
    print(f"\nHosmer-Lemeshow: stat={hl_stat:.2f}, dof={dof}, p-value={p_val:.4f}")

    print("\n=== STABILITY: PSI (Train vs OOT predicted PD) ===")
    psi = population_stability_index(train["predicted_pd"].values, oot["predicted_pd"].values)
    print(f"PSI (score-level): {psi:.4f}")

    raw_features = [
        "RevolvingUtilizationOfUnsecuredLines", "age", "DebtRatio",
        "MonthlyIncome", "NumberOfOpenCreditLinesAndLoans",
    ]
    csi_table = characteristic_stability_index(train, oot, raw_features)
    print("\n=== STABILITY: CSI per feature ===")
    print(csi_table.to_string(index=False))
