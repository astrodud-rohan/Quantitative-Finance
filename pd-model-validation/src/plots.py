"""
Generates Plots - 
    - ROC curve (train/test/OOT overlay)
    - Calibration chart (predicted vs actual by decile)
    - Score distribution shift (train vs OOT) - visual companion to PSI
    - Coeeficient sign chart
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score

plt.rcParams["figure.dpi"] = 110

def plot_roc_curves(datasets: dict, target_col="SeriousDlqin2yrs", pred_col="predicted_pd", save_path="../reports/roc_curves.png"):
    plt.figure(figsize=(6, 6))
    for name, df in datasets.items():
        fpr, tpr, _ = roc_curve(df[target_col], df[pred_col])
        auc = roc_auc_score(df[target_col], df[pred_col])
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Train / Test / OOT")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_calibration_chart(cal_table: pd.DataFrame, save_path="../reports/calibration_chart.png"):
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(cal_table))
    ax.plot(x, cal_table["actual_default_rate"], marker="o", label="Actual default rate")
    ax.plot(x, cal_table["avg_predicted_pd"], marker="s", label="Avg predicted pd")
    ax.set_xticks(x)
    ax.set_xticklabels([f"D{i+1}" for i in range(len(cal_table))])
    ax.set_xlabel("PD Decile (low risk -> high risk)")
    ax.set_ylabel("Default Rate")
    ax.set_title("Calibration: Predicted vs Actual (OOT sample)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_score_shift(train_scores, oot_scores, save_path="../reports/score_distribution_shift.png"):
    plt.figure(figsize=(7, 5))
    plt.hist(train_scores, bins=30, alpha=0.5, density=True, label="Train (development)")
    plt.hist(oot_scores, bins=30, alpha=0.5, density=True, label="OOT (current)")
    plt.xlabel("Predicted PD")
    plt.ylabel("Density")
    plt.title("Predicted PD Distribution: Train vs OOT")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_coefficient_signs(coef_df: pd.DataFrame, save_path="../reports/coefficient_signs.png"):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#d62728" if not ok else "#2ca02c" for ok in coef_df["sign_ok"]]
    bars = ax.barh(coef_df["feature"], coef_df["coefficient"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Coefficient (on WoE-transformed feature)")
    ax.set_title("Coefficient Sign Check — Red = Unexpected / Zero")

    for bar, coef, ok in zip(bars, coef_df["coefficient"], coef_df["sign_ok"]):
        if abs(coef) < 1e-6:
            ax.annotate("ZERO — feature has no effect\n(flagged finding)",
                        xy=(0, bar.get_y() + bar.get_height() / 2),
                        xytext=(0.15, bar.get_y() + bar.get_height() / 2),
                        va="center", color="#d62728", fontsize=9,
                        arrowprops=dict(arrowstyle="->", color="#d62728"))
 
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    train = pd.read_csv("../data/processed/train_scored.csv")
    test = pd.read_csv("../data/processed/test_scored.csv")
    oot = pd.read_csv("../data/processed/oot_scored.csv")

    plot_roc_curves({"Train": train, "Test": test, "OOT": oot})

    import sys
    sys.path.append("src")
    from validation_metrics import decile_calibration_table

    cal_table = decile_calibration_table(oot["SeriousDlqin2yrs"], oot["predicted_pd"])
    plot_calibration_chart(cal_table)

    plot_score_shift(train["predicted_pd"], oot["predicted_pd"])

    import joblib
    from model_training import coefficient_report, WOE_FEATURES

    model = joblib.load("../data/processed/pd_model.pkl")
    coef_df = coefficient_report(model, WOE_FEATURES)
    plot_coefficient_signs(coef_df)

    print("Saved 4 exhibit charts to reports/")

