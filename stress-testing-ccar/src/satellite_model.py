"""
Implements the "satellite model" — the regression linking macroeconomic
variables to portfolio loss rates, which is the centerpiece of any
CCAR-style stress testing framework. The Fed's own CCAR models work the
same way: a relatively simple regression (often linear) trained on
historical data, then applied to hypothetical future macro scenarios
that may go beyond anything in the historical record.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

FEATURES = ["unemployment_rate_lag1", "gdp_growth", "hpi_growth"]
TARGET = "portfolio_loss_rate"


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a 1-quarter-lagged unemployment rate feature, consistent with
    the lag structure used to GENERATE the synthetic loss data (see
    generate_macro_history.py) — i.e., the development team correctly
    identified the right lag structure, so this is not itself a finding.
    """
    out = df.copy()
    out["unemployment_rate_lag1"] = out["unemployment_rate"].shift(1)
    out = out.dropna().reset_index(drop=True)
    return out


def fit_satellite_model(df: pd.DataFrame):
    """Fits a linear regression: loss_rate ~ unemployment_rate_lag1 + gdp_growth + hpi_growth."""
    prepared = prepare_features(df)
    X = prepared[FEATURES]
    y = prepared[TARGET]
    model = LinearRegression()
    model.fit(X, y)
    return model, prepared


def predict_loss_rate(model, macro_inputs: pd.DataFrame) -> np.ndarray:
    """macro_inputs must already contain the FEATURES columns."""
    return model.predict(macro_inputs[FEATURES])


def model_summary(model, prepared_df: pd.DataFrame):
    """Returns coefficients, R-squared, and the TRAINING RANGE for each feature.

    The training range is critical: it is what we will later compare
    Fed stress scenario values against to detect extrapolation.
    """
    X = prepared_df[FEATURES]
    y = prepared_df[TARGET]
    r2 = model.score(X, y)
    coefs = dict(zip(FEATURES, model.coef_))
    ranges = {f: (X[f].min(), X[f].max()) for f in FEATURES}
    return {
        "intercept": model.intercept_,
        "coefficients": coefs,
        "r_squared": r2,
        "training_ranges": ranges,
        "n_obs": len(prepared_df),
    }


if __name__ == "__main__":
    df = pd.read_csv("../data/raw/macro_and_loss_history.csv")
    model, prepared = fit_satellite_model(df)
    summary = model_summary(model, prepared)

    print(f"Satellite model fit on {summary['n_obs']} quarters")
    print(f"R-squared: {summary['r_squared']:.4f}")
    print(f"Intercept: {summary['intercept']:.4f}")
    print("\nCoefficients:")
    for feat, coef in summary["coefficients"].items():
        print(f"  {feat:28s}: {coef:+.4f}")
    print("\nTraining data ranges (for later extrapolation check):")
    for feat, (lo, hi) in summary["training_ranges"].items():
        print(f"  {feat:28s}: [{lo:.2f}, {hi:.2f}]")
