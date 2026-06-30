"""
Trains a logistic regression scorecard on WoE-transformed features
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib

WOE_FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines_woe",
    "age_woe",
    "NumberOfTime30-59DaysPastDueNotWorse_woe",
    "DebtRatio_woe",
    "MonthlyIncome_woe",
    "NumberOfOpenCreditLinesAndLoans_woe",
    "NumberOfTimes90DaysLate_woe",
    "NumberRealEstateLoansOrLines_woe",
    "NumberOfTime60-89DaysPastDueNotWorse_woe",
    "NumberOfDependents_woe",
]

TARGET = "SeriousDlqin2yrs"

def train_pd_model(train_df, features=WOE_FEATURES, target=TARGET):
    X = train_df[features]
    y = train_df[target]

    model = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=1000,
    )
    model.fit(X, y)
    return model

def score_dataset(model, df, features=WOE_FEATURES):
    return model.predict_proba(df[features])[:, 1]

def coefficient_report(model, features=WOE_FEATURES):
    coefs = pd.DataFrame({
        "feature": features,
        "coefficient": model.coef_[0],
    }).sort_values("coefficient")
    coefs["sign_ok"] = coefs["coefficient"] < 0
    return coefs

if __name__ == "__main__":
    train = pd.read_csv("../data/processed/train_woe.csv")
    test = pd.read_csv("../data/processed/test_woe.csv")
    oot = pd.read_csv("../data/processed/oot_woe.csv")

    model = train_pd_model(train)

    print("=== Coefficient sanity check (dev team self-review) ===")
    print(coefficient_report(model).to_string(index=False))

    train["predicted_pd"] = score_dataset(model, train)
    test["predicted_pd"] = score_dataset(model, test)
    oot["predicted_pd"] = score_dataset(model, oot)

    train.to_csv("../data/processed/train_scored.csv", index=False)
    test.to_csv("../data/processed/test_scored.csv", index=False)
    oot.to_csv("../data/processed/oot_scored.csv", index=False)

    joblib.dump(model, "../data/processed/pd_model.pkl")
    print("\nModel trained and scored datasets saved to ../data/processed/")
