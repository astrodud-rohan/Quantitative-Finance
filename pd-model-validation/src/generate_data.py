"""
Generates a synthetic credit dataset that mirrors the structure of the
'Give Me Some Credit' (GMSC) Kaggle dataset, with:
    - same feature schema
    - realistic correlations between features and defaults (SeriousDlqin2yrs)
    - a synthetic 'snapshot_date' field spanning 24 months, to carve out
      an out-of-time (OOT) sample later, like a real validator.
"""

import numpy as np
import pandas as pd
from datetime import date
from dateutil.relativedelta import relativedelta

RANDOM_SEED = 42
N_SAMPLES = 40000

def generate_synthetic_credit_data(n=N_SAMPLES, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)

    age = rng.integers(21, 75, n)
    monthly_income = np.round(rng.lognormal(mean=8.4, sigma=0.6, size=n), 2)
    monthly_income = np.clip(monthly_income, 800, 50000)

    debt_ratio = np.clip(rng.beta(2, 5, n) * 1.5, 0.0, 2.0)

    revolving_util = np.clip(rng.beta(2, 4, n), 0.0, 1.3)

    num_open_credit_lines = rng.integers(0, 25, n)
    num_real_estate_loans = rng.integers(0, 6, n)
    num_dependents = rng.poisson(0.9, n)
    num_dependents = np.clip(num_dependents, 0, 10)

    times_30_59_late = rng.poisson(0.3, n)
    times_60_89_late = rng.poisson(0.15, n)
    times_90_late = rng.poisson(0.1, n)

    start = date(2023, 1, 1)
    month_offsets = rng.integers(0, 24, n)
    snapshot_date = [start + relativedelta(months=int(m)) for m in month_offsets]

    z = (
        -0.00004 * monthly_income
        + 1.8 * revolving_util
        + 0.9 * debt_ratio
        + 0.55 * times_30_59_late
        + 0.85 * times_60_89_late
        + 1.3 * times_90_late
        - 0.012 * age
        + 0.04 * num_dependents
        - 0.01 * num_open_credit_lines
        - 3.6
    )

    z = z + 0.015 * np.array(month_offsets)

    prob_default = 1 / (1 + np.exp(-z))
    default = rng.binomial(1, prob_default)

    df = pd.DataFrame({
        "SeriousDlqin2yrs": default,
        "RevolvingUtilizationOfUnsecuredLines": np.round(revolving_util, 4),
        "age": age,
        "NumberOfTime30-59DaysPastDueNotWorse": times_30_59_late,
        "DebtRatio": np.round(debt_ratio, 4),
        "MonthlyIncome": monthly_income,
        "NumberOfOpenCreditLinesAndLoans": num_open_credit_lines,
        "NumberOfTimes90DaysLate": times_90_late,
        "NumberRealEstateLoansOrLines": num_real_estate_loans,
        "NumberOfTime60-89DaysPastDueNotWorse": times_60_89_late,
        "NumberOfDependents": num_dependents,
        "snapshot_date": snapshot_date
    })

    income_missing_idx = rng.choice(n, size=int(0.05 * n), replace=False)
    df.loc[income_missing_idx, "MonthlyIncome"] = np.nan

    dep_missing_idx = rng.choice(n, size=int(0.03 * n), replace=False)
    df.loc[dep_missing_idx, "NumberOfDependents"] = np.nan

    return df

if __name__ == "__main__":
    df = generate_synthetic_credit_data()
    out_path = "../data/raw/credit_data.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df):,} rows -> {out_path}")
    print(f"Default rate: {df['SeriousDlqin2yrs'].mean():.4f}")
    print(df.head())
    