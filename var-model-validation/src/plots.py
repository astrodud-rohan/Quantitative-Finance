"""
Generates the standard exhibits for VaR backtesting report:
    1. VaR vs realized loss with breaches highlighted (champion model)
    2. Rolling 250-day breach count with Basel traffic-light zones
    3. Model comparison bar chart (Kupiec LR stats across all 4 models)
    4. Breach clustering visualization around the stress event
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import sys
sys.path.append("src")
from backtesting import rolling_250d_breach_count, basel_traffic_light

plt.rcParams["figure.dpi"] = 110

def plot_var_vs_losses(df, var_col="hs_var", model_name="Historical Simulation", save_path="../reports/var_vs_losses.png"):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(df["date"], df["realized_loss"], color="#4472C4", linewidth=0.8, label="Realized daily loss")
    ax.plot(df["date"], df[var_col], color="#C00000", linewidth=1.2, label=f"{model_name} VaR (99%)")

    breach = df["realized_loss"] > df[var_col]
    ax.scatter(df.loc[breach, "date"], df.loc[breach, "realized_loss"], color="#C00000", s=28, zorder=5, label="Breach")

    ax.axvspan(pd.Timestamp("2023-06-15"), pd.Timestamp("2023-09-21"), color="orange", alpha=0.12, label="Stress regime")
    ax.set_ylabel("Daily loss (log-return scale)")
    ax.set_title(f"{model_name} VaR vs Realized Losses - 99% Confidence")
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_rolling_breach_count(df, var_col="hs_var", model_name="Historical Simulation", save_path="../reports/rolling_breach_count.png"):
    breach = df["realized_loss"] > df[var_col]
    rolling_count = rolling_250d_breach_count(breach)
 
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.axhspan(0, 4, color="#C6E0B4", alpha=0.5, zorder=0)
    ax.axhspan(4, 9, color="#FFE699", alpha=0.5, zorder=0)
    ax.axhspan(9, max(rolling_count.max() + 2, 15), color="#F8CBAD", alpha=0.5, zorder=0)
 
    ax.plot(df["date"], rolling_count, color="#1F3864", linewidth=1.5)
    ax.set_ylabel("Breaches in trailing 250 days")
    ax.set_title(f"Basel Traffic-Light Test — {model_name} (99% VaR)")
 
    green_patch = mpatches.Patch(color="#C6E0B4", label="Green zone (0-4)")
    amber_patch = mpatches.Patch(color="#FFE699", label="Amber zone (5-9)")
    red_patch = mpatches.Patch(color="#F8CBAD", label="Red zone (10+)")
    ax.legend(handles=[green_patch, amber_patch, red_patch], loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_model_comparison(results: dict, save_path="../reports/model_comparison.png"):
    models = list(results.keys())
    kupiec_p = [results[m]["kupiec_p"] for m in models]
    christoffersen_p = [results[m]["christoffersen_p"] for m in models]
 
    x = np.arange(len(models))
    width = 0.35
 
    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width / 2, kupiec_p, width, label="Kupiec POF p-value", color="#4472C4")
    bars2 = ax.bar(x + width / 2, christoffersen_p, width, label="Christoffersen Independence p-value", color="#ED7D31")
    ax.axhline(0.05, color="black", linestyle="--", linewidth=0.8, label="5% significance threshold")
 
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("p-value")
    ax.set_title("Backtest p-values by Model (above dashed line = pass)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    df = pd.read_csv("../data/processed/var_estimates.csv", parse_dates=["date"])
    df = df.dropna(subset=["hs_var"]).reset_index(drop=True)

    plot_var_vs_losses(df, "hs_var", "Historical Simulation (Champion)")
    plot_rolling_breach_count(df, "hs_var", "Historical Simulation (Champion)")

    from backtesting import kupiec_pof_test, christoffersen_independence_test

    results = {}
    labels = {"hs_var": "HS (Champion)", "parametric_var": "Parametric", "ewma_var": "EWMA", "fhs_var": "FHS (Challenger)"}
    for col, label in labels.items():
        breach = df["realized_loss"] > df[col]
        kp = kupiec_pof_test(len(df), breach.sum())
        ci = christoffersen_independence_test(breach)
        results[label] = {"breaches": int(breach.sum()), "kupiec_p": kp["p_value"], "christoffersen_p": ci["p_value"]}

    plot_model_comparison(results)
    print("Saved 3 exhibit charts to reports/")    