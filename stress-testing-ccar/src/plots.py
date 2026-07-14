"""
Generates the standard exhibits for a CCAR-style stress testing
validation report:
  1. Historical macro variables with the downturn period highlighted
  2. Scenario trajectories (unemployment) vs. historical training range
  3. Projected cumulative loss rate by scenario
  4. Sensitivity analysis: linear vs quadratic specification
"""

import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append("src")
from scenario_engine import build_scenarios

plt.rcParams["figure.dpi"] = 110

SCENARIO_COLORS = {"Baseline": "#4472C4", "Adverse": "#ED7D31", "Severely Adverse": "#C00000"}


def plot_historical_unemployment_range(history_df, training_max, save_path="../reports/historical_training_range.png"):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(len(history_df)), history_df["unemployment_rate"], color="#1F3864", linewidth=1.5)
    ax.axhline(training_max, color="#C00000", linestyle="--", linewidth=1,
                label=f"Historical training max ({training_max:.2f}%)")
    ax.set_xlabel("Quarter (historical)")
    ax.set_ylabel("Unemployment Rate (%)")
    ax.set_title("Historical Unemployment Rate Used to Train the Satellite Model")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_scenario_vs_training_range(scenarios, training_min, training_max,
                                     save_path="../reports/scenario_vs_training_range.png"):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axhspan(training_min, training_max, color="#C6E0B4", alpha=0.4, label="Historical training range")
    for name, df in scenarios.items():
        ax.plot(range(1, len(df) + 1), df["unemployment_rate"], marker="o",
                color=SCENARIO_COLORS[name], label=name)
    ax.set_xlabel("Stress Horizon Quarter")
    ax.set_ylabel("Unemployment Rate (%)")
    ax.set_title("CCAR Scenario Unemployment Paths vs. Satellite Model Training Range")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_cumulative_losses(projection_df, save_path="../reports/cumulative_losses_by_scenario.png"):
    fig, ax = plt.subplots(figsize=(9, 5))
    for name in ["Baseline", "Adverse", "Severely Adverse"]:
        d = projection_df[projection_df["scenario"] == name].reset_index(drop=True)
        x_vals = range(1, len(d) + 1)
        ax.plot(x_vals, d["cumulative_loss_rate"], marker="o",
                color=SCENARIO_COLORS[name], label=name)
        extrap_mask = d["any_extrapolation"]
        if extrap_mask.any():
            ax.scatter([x for x, flag in zip(x_vals, extrap_mask) if flag],
                       d.loc[extrap_mask, "cumulative_loss_rate"],
                       facecolors="none", edgecolors="black", s=120, linewidths=1.5, zorder=5)
    ax.set_xlabel("Stress Horizon Quarter")
    ax.set_ylabel("Cumulative Portfolio Loss Rate (%)")
    ax.set_title("Projected 9-Quarter Cumulative Loss Rate (circled = extrapolation quarters)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_specification_sensitivity(sensitivity_df, save_path="../reports/specification_sensitivity.png"):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(sensitivity_df))
    width = 0.35
    ax.bar([i - width/2 for i in x], sensitivity_df["cumulative_loss_lag1_linear"], width,
           label="Linear specification (current model)", color="#4472C4")
    ax.bar([i + width/2 for i in x], sensitivity_df["cumulative_loss_lag1_quadratic"], width,
           label="Quadratic specification (alternative)", color="#C00000")
    ax.set_xticks(list(x))
    ax.set_xticklabels(sensitivity_df["scenario"])
    ax.set_ylabel("9-Quarter Cumulative Loss Rate (%)")
    ax.set_title("Specification Sensitivity: Linear vs. Quadratic Unemployment Term")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    history = pd.read_csv("../data/raw/macro_and_loss_history.csv")
    projection = pd.read_csv("../data/processed/stress_projection_results.csv")
    sensitivity = pd.read_csv("../data/processed/sensitivity_analysis_results.csv")
    scenarios = build_scenarios()

    training_max = history["unemployment_rate"].max()
    training_min = history["unemployment_rate"].min()

    plot_historical_unemployment_range(history, training_max)
    plot_scenario_vs_training_range(scenarios, training_min, training_max)
    plot_cumulative_losses(projection)
    plot_specification_sensitivity(sensitivity)
    print("Saved 4 exhibit charts to reports/")
