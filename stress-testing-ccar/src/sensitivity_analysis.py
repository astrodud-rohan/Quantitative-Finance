"""
Tests the satellite model's SPECIFICATION RISK: how much does the
projected 9-quarter cumulative loss rate change under reasonable
alternative modeling choices the development team could just as easily
have made?

Two sensitivities are tested:
  1. LAG STRUCTURE: 1-quarter vs. 2-quarter lag on unemployment.
  2. FUNCTIONAL FORM: linear vs. a quadratic (convex) unemployment term,
     since the data-generating process (see generate_macro_history.py)
     has genuine convexity that a linear model only partially captures.
"""

import sys
sys.path.append("src")
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scenario_engine import build_scenarios


def fit_with_lag(df: pd.DataFrame, lag: int):
    out = df.copy()
    out["unemployment_rate_lag"] = out["unemployment_rate"].shift(lag)
    out = out.dropna().reset_index(drop=True)
    X = out[["unemployment_rate_lag", "gdp_growth", "hpi_growth"]]
    y = out["portfolio_loss_rate"]
    model = LinearRegression()
    model.fit(X, y)
    return model


def fit_with_quadratic_term(df: pd.DataFrame, lag: int = 1):
    out = df.copy()
    out["unemployment_rate_lag"] = out["unemployment_rate"].shift(lag)
    out["unemployment_rate_lag_sq"] = out["unemployment_rate_lag"] ** 2
    out = out.dropna().reset_index(drop=True)
    X = out[["unemployment_rate_lag", "unemployment_rate_lag_sq", "gdp_growth", "hpi_growth"]]
    y = out["portfolio_loss_rate"]
    model = LinearRegression()
    model.fit(X, y)
    return model


def project_with_lag(model, scenario_df: pd.DataFrame, last_n_historical: list, lag: int):
    df = scenario_df.copy().reset_index(drop=True)
    full_unemployment = last_n_historical + df["unemployment_rate"].tolist()
    lagged = full_unemployment[len(last_n_historical) - lag: len(last_n_historical) - lag + len(df)]
    X = pd.DataFrame({
        "unemployment_rate_lag": lagged,
        "gdp_growth": df["gdp_growth"],
        "hpi_growth": df["hpi_growth"],
    })
    return model.predict(X)


def project_with_quadratic(model, scenario_df: pd.DataFrame, last_n_historical: list, lag: int = 1):
    df = scenario_df.copy().reset_index(drop=True)
    full_unemployment = last_n_historical + df["unemployment_rate"].tolist()
    lagged = full_unemployment[len(last_n_historical) - lag: len(last_n_historical) - lag + len(df)]
    lagged_sq = [v ** 2 for v in lagged]
    X = pd.DataFrame({
        "unemployment_rate_lag": lagged,
        "unemployment_rate_lag_sq": lagged_sq,
        "gdp_growth": df["gdp_growth"],
        "hpi_growth": df["hpi_growth"],
    })
    return model.predict(X)


def run_sensitivity_analysis():
    history = pd.read_csv("../data/raw/macro_and_loss_history.csv")
    scenarios = build_scenarios()
    last_2_unemployment = history["unemployment_rate"].iloc[-2:].tolist()

    model_lag1 = fit_with_lag(history, lag=1)
    model_lag2 = fit_with_lag(history, lag=2)
    model_quad = fit_with_quadratic_term(history, lag=1)

    results = []
    for name, scenario_df in scenarios.items():
        loss_lag1 = project_with_lag(model_lag1, scenario_df, last_2_unemployment, lag=1)
        loss_lag2 = project_with_lag(model_lag2, scenario_df, last_2_unemployment, lag=2)
        loss_quad = project_with_quadratic(model_quad, scenario_df, last_2_unemployment, lag=1)

        results.append({
            "scenario": name,
            "cumulative_loss_lag1_linear": round(loss_lag1.sum(), 2),
            "cumulative_loss_lag2_linear": round(loss_lag2.sum(), 2),
            "cumulative_loss_lag1_quadratic": round(loss_quad.sum(), 2),
        })

    results_df = pd.DataFrame(results)
    results_df["pct_diff_lag2_vs_lag1"] = round(
        (results_df["cumulative_loss_lag2_linear"] / results_df["cumulative_loss_lag1_linear"] - 1) * 100, 1
    )
    results_df["pct_diff_quadratic_vs_linear"] = round(
        (results_df["cumulative_loss_lag1_quadratic"] / results_df["cumulative_loss_lag1_linear"] - 1) * 100, 1
    )
    return results_df


if __name__ == "__main__":
    results_df = run_sensitivity_analysis()
    print(results_df.to_string(index=False))
    results_df.to_csv("../data/processed/sensitivity_analysis_results.csv", index=False)
