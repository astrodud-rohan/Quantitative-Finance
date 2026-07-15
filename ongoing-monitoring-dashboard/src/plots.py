"""
Generates static exhibit charts for the Month 12 monitoring report
(the interactive dashboard in dashboard/monitoring_dashboard.html is the
primary day-to-day monitoring tool; these are the supporting exhibits
for the formal escalation report).
"""

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["figure.dpi"] = 110

RAG_COLORS = {"Green": "#548235", "Amber": "#BF8F00", "Red": "#C00000"}


def plot_metric_trend(df, metric_col, rag_col, title, ylabel, thresholds, save_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["month"], df[metric_col], color="#1F3864", linewidth=1.5, zorder=3)
    for _, row in df.iterrows():
        ax.scatter(row["month"], row[metric_col], color=RAG_COLORS[row[rag_col]], s=80, zorder=5,
                   edgecolors="black", linewidths=0.5)
    for val, label, color in thresholds:
        ax.axhline(val, color=color, linestyle="--", linewidth=0.8, alpha=0.7, label=label)
    ax.set_xlabel("Month")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(range(1, 13))
    ax.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    df = pd.read_csv("../data/processed/monthly_monitoring_results.csv")

    plot_metric_trend(
        df, "psi", "psi_rag", "Population Stability Index (PSI) by Month", "PSI",
        [(0.10, "Amber threshold (0.10)", "#BF8F00"), (0.25, "Red threshold (0.25)", "#C00000")],
        "../reports/psi_trend.png",
    )

    plot_metric_trend(
        df, "gini_pct_of_baseline", "gini_rag", "Discriminatory Power Retained vs. Baseline", "% of Baseline Gini",
        [(0.90, "Amber threshold (90%)", "#BF8F00"), (0.80, "Red threshold (80%)", "#C00000")],
        "../reports/gini_decay_trend.png",
    )

    plot_metric_trend(
        df, "override_rate", "override_rag", "Underwriter Override Rate by Month", "Override Rate",
        [(0.05, "Amber threshold (5%)", "#BF8F00"), (0.10, "Red threshold (10%)", "#C00000")],
        "../reports/override_rate_trend.png",
    )

    print("Saved 3 exhibit charts to reports/")
