"""
Simulates 12 months of "production" data for a deployed retail credit PD
scorecard (conceptually the same model validated in Project 1), to support
an ongoing monitoring dashboard.
Three realistic degradation patterns are deliberately built in:

  1. POPULATION DRIFT: the distribution of model scores gradually shifts
     over the year (the applicant population becomes somewhat riskier),
     which should show up in a rising Population Stability Index (PSI)
     relative to the development-sample baseline.

  2. DISCRIMINATION DECAY: the model's ability to rank-order risk
     (measured monthly via Gini) erodes slowly over the year, as is
     common when a model is not refreshed and the world moves on.

  3. RISING OVERRIDE RATE: the rate at which underwriters manually
     override the model's decline recommendation increases over the
     year -- a classic early-warning sign of eroding business trust in
     a model, which itself is a standard ongoing monitoring metric.
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 23
N_MONTHS = 12
N_PER_MONTH = 2000

DECLINE_CUTOFF = 0.15  # applicants with predicted PD above this are recommended for decline


def simulate_baseline_distribution(seed=RANDOM_SEED):
    """The development-sample score distribution, used as the PSI baseline."""
    rng = np.random.default_rng(seed)
    n = 10000
    scores = rng.beta(2, 12, n) * 0.6  # right-skewed, mostly low PD, matches a healthy scorecard
    return scores


def simulate_production_month(month_idx, seed=RANDOM_SEED):
    """
    Simulates one month of production data.
    month_idx: 0-11 (Jan = 0 ... Dec = 11)
    """
    rng = np.random.default_rng(seed + 100 + month_idx)
    n = N_PER_MONTH

    # --- Population drift: mean of the score distribution creeps up over the year ---
    drift_factor = 1.0 + 0.018 * month_idx  # gentler drift than before
    scores = rng.beta(2, 12 / drift_factor, n) * 0.6
    scores = np.clip(scores, 0.001, 0.95)

    # --- Discrimination decay: correlation between score and true outcome weakens over time ---
    signal_weight = max(0.55, 1.0 - 0.028 * month_idx)   # decays from 1.0 toward 0.55
    noise_weight = 1.0 - signal_weight
    score_logit = np.log(scores / (1 - scores))
    noise = rng.normal(0, score_logit.std() if score_logit.std() > 0 else 1.0, n)
    blended_logit = signal_weight * score_logit + noise_weight * noise
    true_prob_default = 1 / (1 + np.exp(-blended_logit))
    true_prob_default = np.clip(true_prob_default, 0.01, 0.95)
    outcome = rng.binomial(1, true_prob_default)

    # --- Override rate: rises over the year as business pressure / model distrust grows ---
    recommend_decline = scores > DECLINE_CUTOFF
    base_override_rate = 0.02 + 0.0075 * month_idx  # rises from ~2% to ~10.25% by month 12
    override_draw = rng.uniform(0, 1, n)
    overridden = recommend_decline & (override_draw < base_override_rate)

    df = pd.DataFrame({
        "month": month_idx + 1,
        "predicted_pd": scores,
        "true_outcome": outcome,
        "recommend_decline": recommend_decline,
        "overridden": overridden,
    })
    return df


if __name__ == "__main__":
    baseline = simulate_baseline_distribution()
    np.save("../data/processed/baseline_score_distribution.npy", baseline)

    all_months = [simulate_production_month(m) for m in range(N_MONTHS)]
    full_df = pd.concat(all_months, ignore_index=True)
    full_df.to_csv("../data/processed/production_scores_12mo.csv", index=False)

    print(f"Baseline distribution: n={len(baseline)}, mean PD={baseline.mean():.4f}")
    print(f"Production data: {len(full_df)} scored applicants across {N_MONTHS} months\n")

    summary = full_df.groupby("month").agg(
        mean_pd=("predicted_pd", "mean"),
        default_rate=("true_outcome", "mean"),
        decline_rate=("recommend_decline", "mean"),
    )
    print(summary.round(4))
