"""
Data Processing handles:
    1. Loading raw data
    2. Missing value treatment (industry-realistic, not just drop/mean-fill)
    3. Train / Test / Out-of-Time (OOT) split based on snapshot_date
    4. Weight of Evidence (WoE) binning - standard transformation for credit scorecard development
        - handles non-linear relationships in a logistic regression
        - monotonic and interpretable to credit committee/regulator
        - handles missing values as their own bin
"""

import pandas as pd
import numpy as np

TARGET = "SeriousDlqin2yrs"

RAW_FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]

def load_raw_data(path="../data/raw/credit_data.csv"):
    df = pd.read_csv(path, parse_dates=["snapshot_date"])
    return df

def treat_missing_values(df):
    df = df.copy()
    df["MonthlyIncome_missing_flag"] = df["MonthlyIncome"].isna().astype(int)
    df["NumberOfDependents_missing_flag"] = df["NumberOfDependents"].isna().astype(int)
    return df

def train_test_oot_split(df, oot_months=4, test_frac=0.2, random_state=42):
    df = df.sort_values("snapshot_date")
    cutoff_date = df["snapshot_date"].max() - pd.DateOffset(months=oot_months)

    oot = df[df["snapshot_date"] > cutoff_date].copy()
    dev_pool = df[df["snapshot_date"] <= cutoff_date].copy()

    test = dev_pool.sample(frac=test_frac, random_state=random_state)
    train = dev_pool.drop(test.index)

    return (
        train.reset_index(drop=True),
        test.reset_index(drop=True),
        oot.reset_index(drop=True)
    )

def woe_bin_fit(df, feature, target=TARGET, n_bins=5):
    data = df[[feature, target]].copy()
    missing_mask = data[feature].isna()

    non_missing = data[~missing_mask]
    try:
        non_missing["bin"] = pd.qcut(non_missing[feature], q=n_bins, duplicates="drop")
    except ValueError:
        non_missing["bin"] = pd.cut(non_missing[feature], bins=n_bins)
    
    total_good = (data[target] == 0).sum()
    total_bad = (data[target] == 1).sum()

    bin_map = []
    grouped = non_missing.groupby("bin")[target].agg(["count", "sum"])
    for interval, row in grouped.iterrows():
        bad = row["sum"]
        good = row["count"] - bad

        pct_good = (good + 0.5) / (total_good + 0.5)
        pct_bad = (bad + 0.5) / (total_bad + 0.5)
        woe = np.log(pct_good / pct_bad)
        bin_map.append({
            "lower": interval.left,
            "upper": interval.right,
            "woe": woe,
            "count": int(row["count"]),
            "bad_rate": bad / row["count"] if row["count"] > 0 else np.nan
        })
    
    if missing_mask.sum() > 0:
        miss_bad = data.loc[missing_mask, target].sum()
        miss_good = missing_mask.sum() - miss_bad
        pct_good = (miss_good + 0.5) / (total_good + 0.5)
        pct_bad = (miss_bad + 0.5) / (total_bad + 0.5)
        missing_woe = np.log(pct_good / pct_bad)
    else:
        missing_woe = 0.0

    return {"bins": bin_map, "missing_woe": missing_woe}

def woe_bin_transform(series, bin_map):
    def lookup(val):
        if pd.isna(val):
            return bin_map["missing_woe"]
        for b in bin_map["bins"]:
            lower_ok = (b["lower"] is None) or (val > b["lower"])
            upper_ok = (b["upper"] is None) or (val <= b["upper"])
            if lower_ok and upper_ok:
                return b["woe"]
        return bin_map["bins"][0]["woe"] if val < bin_map["bins"][0]["lower"] else bin_map["bins"][-1]["woe"]
    return series.apply(lookup)

def fit_woe_all_features(train_df, features=RAW_FEATURES, n_bins=5):
    woe_maps = {}
    for feat in features:
        woe_maps[feat] = woe_bin_fit(train_df, feat, n_bins=n_bins)
    return woe_maps

def apply_woe_all_features(df, woe_maps):
    df_woe = df.copy()
    for feat, bin_map in woe_maps.items():
        df_woe[f"{feat}_woe"] = woe_bin_transform(df[feat], bin_map)
    return df_woe

if __name__ == "__main__":
    df = load_raw_data()
    df = treat_missing_values(df)
    train, test, oot = train_test_oot_split(df)
    print(f"Train: {len(train):,} | Test: {len(test):,} | OOT: {len(oot):,}")
    print(f"Train default rate: {train[TARGET].mean():.4f}")
    print(f"Test default rate:  {test[TARGET].mean():.4f}")
    print(f"OOT default rate:   {oot[TARGET].mean():.4f}")

    woe_maps = fit_woe_all_features(train)
    train_woe = apply_woe_all_features(train, woe_maps)
    test_woe = apply_woe_all_features(test, woe_maps)
    oot_woe = apply_woe_all_features(oot, woe_maps)

    train_woe.to_csv("../data/processed/train_woe.csv", index=False)
    test_woe.to_csv("../data/processed/test_woe.csv", index=False)
    oot_woe.to_csv("../data/processed/oot_woe.csv", index=False)
    print("Saved WoE-transformed train/test/OOT sets to ../data/processed/")