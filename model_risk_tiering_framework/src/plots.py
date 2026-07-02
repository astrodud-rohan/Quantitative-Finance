"""
Generates exhibits for the governance report:
    1. Tier distribution pie/bar chart
    2. Models by business line, colored by tier
    3. Validation workload over the next 3 years
"""

import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append("src")
sys.path.append("scripts")
from build_inventory import build_inventory_df
from tiering_engine import score_inventory

plt.rcParams["figure.dpi"] = 110
 
TIER_COLORS = {"Tier 1 (High)": "#C00000", "Tier 2 (Medium)": "#BF8F00", "Tier 3 (Low)": "#548235"}
 
 
def plot_tier_distribution(df, save_path="reports/tier_distribution.png"):
    counts = df["tier"].value_counts().reindex(["Tier 1 (High)", "Tier 2 (Medium)", "Tier 3 (Low)"])
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(counts.index, counts.values, color=[TIER_COLORS[t] for t in counts.index])
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15, str(val),
                ha="center", fontweight="bold")
    ax.set_ylabel("Number of Models")
    ax.set_title("Model Inventory — Tier Distribution (n=18)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
 
 
def plot_business_line_breakdown(df, save_path="reports/business_line_breakdown.png"):
    pivot = df.groupby(["business_line", "tier"]).size().unstack(fill_value=0)
    pivot = pivot.reindex(columns=["Tier 1 (High)", "Tier 2 (Medium)", "Tier 3 (Low)"], fill_value=0)
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=True).index]
 
    fig, ax = plt.subplots(figsize=(9, 6))
    left = pd.Series(0, index=pivot.index)
    for tier in pivot.columns:
        ax.barh(pivot.index, pivot[tier], left=left, color=TIER_COLORS[tier], label=tier)
        left += pivot[tier]
    ax.set_xlabel("Number of Models")
    ax.set_title("Models by Business Line and Tier")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
 
 
def plot_validation_workload(save_path="reports/validation_workload.png"):
    # Illustrative 3-year validation workload based on cycle lengths per tier
    years = ["FY2026", "FY2027", "FY2028"]
    tier1 = [7, 7, 7]            # annual cycle -> all 7 every year
    tier2 = [5, 6, 5]            # 18-month cycle, staggered
    tier3 = [1, 1, 1]            # 36-month cycle, staggered across 3 models
 
    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(years))
    ax.bar(x, tier1, label="Tier 1 (Annual)", color=TIER_COLORS["Tier 1 (High)"])
    ax.bar(x, tier2, bottom=tier1, label="Tier 2 (18-month)", color=TIER_COLORS["Tier 2 (Medium)"])
    bottom2 = [a + b for a, b in zip(tier1, tier2)]
    ax.bar(x, tier3, bottom=bottom2, label="Tier 3 (36-month)", color=TIER_COLORS["Tier 3 (Low)"])
    ax.set_xticks(list(x))
    ax.set_xticklabels(years)
    ax.set_ylabel("Validations Due")
    ax.set_title("Projected Validation Workload, FY2026-FY2028")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
 
 
if __name__ == "__main__":
    df = build_inventory_df()
    df = score_inventory(df)
    plot_tier_distribution(df)
    plot_business_line_breakdown(df)
    plot_validation_workload()
    print("Saved 3 exhibit charts to reports/")