"""
THE CORE DELIVERABLE OF THIS PROJECT.

Applies the fitted satellite model to each of the three CCAR-style
scenarios, projects quarterly and cumulative 9-quarter portfolio losses,
and -- critically -- checks each quarter's scenario inputs against the
satellite model's HISTORICAL TRAINING RANGE to flag extrapolation.
"""

import sys
sys.path.append("src")
import numpy as np
import pandas as pd
from satellite_model import fit_satellite_model, model_summary, FEATURES


def prepare_scenario_features(scenario_df: pd.DataFrame, last_historical_unemployment: float) -> pd.DataFrame:
    """
    Builds the lagged unemployment feature for a scenario projection.
    Quarter 1 of the scenario uses the LAST historical quarter's
    unemployment rate as the lag (continuity with history); subsequent
    quarters use the scenario's own prior-quarter unemployment.
    """
    df = scenario_df.copy().reset_index(drop=True)
    lagged = [last_historical_unemployment] + df["unemployment_rate"].iloc[:-1].tolist()
    df["unemployment_rate_lag1"] = lagged
    return df


def flag_extrapolation(scenario_df: pd.DataFrame, training_ranges: dict) -> pd.DataFrame:
    """For each quarter and each feature, flags whether the scenario value falls outside the training range."""
    df = scenario_df.copy()
    for feat, (lo, hi) in training_ranges.items():
        df[f"{feat}_outside_range"] = (df[feat] < lo) | (df[feat] > hi)
    flag_cols = [f"{feat}_outside_range" for feat in training_ranges]
    df["any_extrapolation"] = df[flag_cols].any(axis=1)
    return df


def project_scenario_losses(model, scenario_df: pd.DataFrame, training_ranges: dict,
                             last_historical_unemployment: float) -> pd.DataFrame:
    df = prepare_scenario_features(scenario_df, last_historical_unemployment)
    df["projected_loss_rate"] = model.predict(df[FEATURES])
    df = flag_extrapolation(df, training_ranges)
    df["cumulative_loss_rate"] = df["projected_loss_rate"].cumsum()
    return df


if __name__ == "__main__":
    from scenario_engine import build_scenarios

    history = pd.read_csv("../data/raw/macro_and_loss_history.csv")
    model, prepared = fit_satellite_model(history)
    summary = model_summary(model, prepared)
    training_ranges = summary["training_ranges"]
    last_historical_unemployment = history["unemployment_rate"].iloc[-1]

    scenarios = build_scenarios()
    all_results = []
    for name, scenario_df in scenarios.items():
        result = project_scenario_losses(model, scenario_df, training_ranges, last_historical_unemployment)
        result["scenario"] = name
        all_results.append(result)
        print(f"\n=== {name} ===")
        print(result[["quarter", "unemployment_rate", "projected_loss_rate",
                       "cumulative_loss_rate", "any_extrapolation"]].to_string(index=False))
        n_quarters_extrapolated = result["any_extrapolation"].sum()
        if n_quarters_extrapolated > 0:
            print(f"  -> WARNING: {n_quarters_extrapolated} of 9 quarters require extrapolation "
                  f"beyond the model's historical training range.")

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv("../data/processed/stress_projection_results.csv", index=False)

    print("\n=== SUMMARY: 9-Quarter Cumulative Loss Rate by Scenario ===")
    for name in scenarios:
        cum_loss = combined[combined["scenario"] == name]["cumulative_loss_rate"].iloc[-1]
        print(f"  {name:20s}: {cum_loss:.2f}%")
