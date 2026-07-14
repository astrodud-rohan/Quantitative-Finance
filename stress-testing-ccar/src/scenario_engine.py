"""
Defines three macroeconomic scenarios over a 9-quarter stress horizon,
styled after the structure of the Federal Reserve's annual CCAR/DFAST
scenarios (Baseline, Adverse, Severely Adverse). The actual trajectories
here are illustrative approximations, not the Fed's published numbers,
but follow the same general shape: a sharp initial shock to unemployment
and home prices, peaking around quarters 4-6, followed by a gradual
recovery.
"""

import numpy as np
import pandas as pd

N_QUARTERS = 9


def _path(start, peak, end, peak_quarter=5, n=N_QUARTERS):
    """Builds a smooth rise-then-recovery path: start -> peak at peak_quarter -> end by quarter n."""
    path = np.zeros(n)
    for q in range(1, n + 1):
        if q <= peak_quarter:
            frac = q / peak_quarter
            path[q - 1] = start + (peak - start) * frac
        else:
            frac = (q - peak_quarter) / (n - peak_quarter)
            path[q - 1] = peak + (end - peak) * frac
    return path


def build_scenarios():
    quarters = [f"Q{i}" for i in range(1, N_QUARTERS + 1)]

    scenarios = {}

    scenarios["Baseline"] = pd.DataFrame({
        "quarter": quarters,
        "unemployment_rate": _path(4.4, 4.6, 4.3, peak_quarter=4),
        "gdp_growth": np.full(N_QUARTERS, 2.2) + np.linspace(0, -0.1, N_QUARTERS),
        "hpi_growth": np.full(N_QUARTERS, 3.0) + np.linspace(0, -0.3, N_QUARTERS),
    })

    scenarios["Adverse"] = pd.DataFrame({
        "quarter": quarters,
        "unemployment_rate": _path(4.4, 7.5, 5.5, peak_quarter=5),
        "gdp_growth": _path(2.0, -1.5, 2.0, peak_quarter=4),
        "hpi_growth": _path(2.5, -5.0, 1.0, peak_quarter=5),
    })

    scenarios["Severely Adverse"] = pd.DataFrame({
        "quarter": quarters,
        "unemployment_rate": _path(4.4, 10.0, 6.5, peak_quarter=6),
        "gdp_growth": _path(1.5, -4.0, 1.8, peak_quarter=4),
        "hpi_growth": _path(2.0, -12.0, -1.0, peak_quarter=6),
    })

    for name, df in scenarios.items():
        df["unemployment_rate"] = np.round(df["unemployment_rate"], 2)
        df["gdp_growth"] = np.round(df["gdp_growth"], 2)
        df["hpi_growth"] = np.round(df["hpi_growth"], 2)
        df["scenario"] = name

    return scenarios


if __name__ == "__main__":
    scenarios = build_scenarios()
    for name, df in scenarios.items():
        print(f"\n=== {name} ===")
        print(df[["quarter", "unemployment_rate", "gdp_growth", "hpi_growth"]].to_string(index=False))
        df.to_csv(f"../data/processed/scenario_{name.lower().replace(' ', '_')}.csv", index=False)
