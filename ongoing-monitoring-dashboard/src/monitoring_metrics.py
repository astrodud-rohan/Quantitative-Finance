"""
THE CORE DELIVERABLE OF THIS PROJECT.

Computes the standard ongoing monitoring metrics for a deployed credit
scorecard, on a monthly basis, and applies RAG (Red/Amber/Green)
threshold logic consistent with the Ongoing Monitoring Framework
(see docs/Ongoing_Monitoring_Framework.docx).

Three metrics are tracked:
  1. PSI (Population Stability Index) vs. the development-sample baseline
  2. Gini coefficient (discriminatory power) vs. the development-sample Gini
  3. Override rate (% of decline recommendations manually overridden)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PSI_THRESHOLDS = {"green_max": 0.10, "amber_max": 0.25}
GINI_DECAY_THRESHOLDS = {"green_min_pct": 0.90, "amber_min_pct": 0.80}
OVERRIDE_RATE_THRESHOLDS = {"green_max": 0.05, "amber_max": 0.10}

RAG_ORDER = {"Green": 0, "Amber": 1, "Red": 2}
RANDOM_SEED = 23


def population_stability_index(baseline: np.ndarray, current: np.ndarray, n_bins=10) -> float:
    breakpoints = np.quantile(baseline, np.linspace(0, 1, n_bins + 1))
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf
    baseline_counts, _ = np.histogram(baseline, bins=breakpoints)
    current_counts, _ = np.histogram(current, bins=breakpoints)
    baseline_pct = baseline_counts / len(baseline) + 1e-6
    current_pct = current_counts / len(current) + 1e-6
    return float(np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct)))


def gini_coefficient(y_true, y_score) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    auc = roc_auc_score(y_true, y_score)
    return 2 * auc - 1


def rag_for_psi(psi: float) -> str:
    if psi <= PSI_THRESHOLDS["green_max"]:
        return "Green"
    elif psi <= PSI_THRESHOLDS["amber_max"]:
        return "Amber"
    return "Red"


def rag_for_gini_decay(current_gini: float, baseline_gini: float) -> str:
    pct_of_baseline = current_gini / baseline_gini if baseline_gini else np.nan
    if pct_of_baseline >= GINI_DECAY_THRESHOLDS["green_min_pct"]:
        return "Green"
    elif pct_of_baseline >= GINI_DECAY_THRESHOLDS["amber_min_pct"]:
        return "Amber"
    return "Red"


def rag_for_override_rate(rate: float) -> str:
    if rate <= OVERRIDE_RATE_THRESHOLDS["green_max"]:
        return "Green"
    elif rate <= OVERRIDE_RATE_THRESHOLDS["amber_max"]:
        return "Amber"
    return "Red"


def worst_rag(rags: list) -> str:
    return max(rags, key=lambda r: RAG_ORDER[r])


def compute_baseline_gini(seed=RANDOM_SEED, n=20000) -> float:
    """
    Computes a clean, large-sample baseline Gini using the SAME generative
    process as month 0 (signal_weight=1.0, no decay), decoupled from the
    small-sample noise inherent in any single month's ~2,000 observations.
    """
    rng = np.random.default_rng(seed + 999)
    scores = rng.beta(2, 12, n) * 0.6
    scores = np.clip(scores, 0.001, 0.95)
    score_logit = np.log(scores / (1 - scores))
    true_prob_default = 1 / (1 + np.exp(-score_logit))
    outcome = rng.binomial(1, true_prob_default)
    return gini_coefficient(outcome, scores)


def compute_monthly_monitoring(production_df: pd.DataFrame, baseline_scores: np.ndarray) -> pd.DataFrame:
    baseline_gini = compute_baseline_gini()

    rows = []
    for month in sorted(production_df["month"].unique()):
        month_df = production_df[production_df["month"] == month]

        psi = population_stability_index(baseline_scores, month_df["predicted_pd"].values)
        gini = gini_coefficient(month_df["true_outcome"], month_df["predicted_pd"])

        n_declines = month_df["recommend_decline"].sum()
        n_overridden = month_df["overridden"].sum()
        override_rate = n_overridden / n_declines if n_declines > 0 else 0.0

        psi_rag = rag_for_psi(psi)
        gini_rag = rag_for_gini_decay(gini, baseline_gini)
        override_rag = rag_for_override_rate(override_rate)
        overall_rag = worst_rag([psi_rag, gini_rag, override_rag])

        rows.append({
            "month": month,
            "psi": round(psi, 4),
            "psi_rag": psi_rag,
            "gini": round(gini, 4),
            "gini_pct_of_baseline": round(gini / baseline_gini, 4) if baseline_gini else np.nan,
            "gini_rag": gini_rag,
            "override_rate": round(override_rate, 4),
            "override_rag": override_rag,
            "overall_rag": overall_rag,
        })

    return pd.DataFrame(rows), baseline_gini


if __name__ == "__main__":
    production_df = pd.read_csv("../data/processed/production_scores_12mo.csv")
    baseline_scores = np.load("../data/processed/baseline_score_distribution.npy")

    monitoring_df, baseline_gini = compute_monthly_monitoring(production_df, baseline_scores)
    print(f"Baseline (Month 1) Gini: {baseline_gini:.4f}\n")
    print(monitoring_df.to_string(index=False))

    monitoring_df.to_csv("../data/processed/monthly_monitoring_results.csv", index=False)

    escalations = monitoring_df[monitoring_df["overall_rag"] == "Red"]
    if len(escalations) > 0:
        print(f"\n{len(escalations)} month(s) breached Red status: months {escalations['month'].tolist()}")
