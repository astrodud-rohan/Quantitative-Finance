"""
Generates 10 years (40 quarters) of synthetic macroeconomic history and a correlated retail loan portfolio
loss rate series.

Design choices (deliberate, not arbitrary):
  - Unemployment rate ranges roughly 3.5% - 8.0% over the 40 quarters,
    including ONE moderate downturn (peak ~8.0%) around quarter 18-22.
  - This means the historical training data NEVER observes unemployment
    above 8.0%. When we later apply Fed-style "Severely Adverse" CCAR
    scenarios (which typically assume unemployment peaking around 10%),
    the satellite model will be asked to EXTRAPOLATE beyond its training
    range - this is the central, realistic validation finding this
    project is built around.
  - Portfolio loss rate is generated with a lagged, nonlinear-ish
    sensitivity to unemployment (loss rates react with a 1-2 quarter lag
    and accelerate disproportionately at higher unemployment), which is
    realistic credit risk behavior and also means a simple linear model
    fit only on the moderate downturn will understate convexity in a
    more severe one - a second, related validation finding.
"""
import numpy as np
import pandas as pd

RANDOM_SEED = 11
N_QUARTERS = 40


def generate_macro_history(n=N_QUARTERS, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)

    quarters = pd.period_range("2015Q1", periods=n, freq="Q")

    # --- Unemployment rate path: stable, then a moderate downturn, then recovery ---
    unemployment = np.zeros(n)
    unemployment[0] = 4.5
    long_run_mean = 4.3
    downturn_start, downturn_peak, downturn_end = 16, 22, 30
    for t in range(1, n):
        if t < downturn_start:
            drift = 0.15 * (long_run_mean - unemployment[t - 1])  # gentle mean reversion
        elif t < downturn_peak:
            drift = 0.62  # sharp rise into downturn
        elif t < downturn_end:
            drift = -0.38  # recovery
        else:
            drift = 0.15 * (long_run_mean - unemployment[t - 1])  # mean-revert post-recovery
        noise = rng.normal(0, 0.08)
        unemployment[t] = unemployment[t - 1] + drift + noise
    unemployment = np.clip(unemployment, 3.3, 8.5)

    # --- Real GDP growth (annualized %, quarterly): inversely related to unemployment changes ---
    unemployment_delta = np.diff(unemployment, prepend=unemployment[0])
    gdp_growth = 2.2 - 1.8 * unemployment_delta + rng.normal(0, 0.5, n)

    # --- National HPI growth (YoY %): also weakens during the downturn, with a lag ---
    hpi_growth = np.zeros(n)
    hpi_growth[0] = 4.0
    for t in range(1, n):
        lagged_unemployment_delta = unemployment_delta[max(0, t - 1)]
        hpi_growth[t] = 0.92 * hpi_growth[t - 1] - 2.5 * lagged_unemployment_delta + rng.normal(0, 0.3)
    hpi_growth = np.clip(hpi_growth, -8.0, 8.0)

    df = pd.DataFrame({
        "quarter": quarters.astype(str),
        "unemployment_rate": np.round(unemployment, 2),
        "gdp_growth": np.round(gdp_growth, 2),
        "hpi_growth": np.round(hpi_growth, 2),
    })
    return df


def generate_portfolio_loss_rate(macro_df, seed=RANDOM_SEED):
    """
    Generates a quarterly net charge-off rate (%) for a retail loan
    portfolio, with:
      - A 1-quarter lag to unemployment LEVEL (not just delta) -- loss
        rates respond to how bad the labor market IS, with a lag, not
        just how fast it's changing.
      - Mild convexity: losses accelerate disproportionately as
        unemployment rises above ~6%, which a LINEAR satellite model
        fit on this data will only partially capture -- by design, this
        creates a genuine specification risk finding later.
      - HPI growth has a smaller, contemporaneous negative effect
        (weaker collateral values -> somewhat higher losses).
    """
    rng = np.random.default_rng(seed + 1)
    n = len(macro_df)
    unemployment = macro_df["unemployment_rate"].values
    hpi = macro_df["hpi_growth"].values

    loss_rate = np.zeros(n)
    base = 0.55  # baseline NCO rate (%) at "normal" unemployment ~4.5%
    for t in range(n):
        lagged_unemployment = unemployment[t - 1] if t > 0 else unemployment[0]
        excess_unemployment = max(0, lagged_unemployment - 4.5)
        # convex term: excess unemployment squared, scaled down
        convex_term = 0.06 * excess_unemployment ** 2
        linear_term = 0.28 * excess_unemployment
        hpi_term = -0.02 * hpi[t]
        loss_rate[t] = base + linear_term + convex_term + hpi_term + rng.normal(0, 0.04)

    loss_rate = np.clip(loss_rate, 0.15, None)
    macro_df = macro_df.copy()
    macro_df["portfolio_loss_rate"] = np.round(loss_rate, 3)
    return macro_df


if __name__ == "__main__":
    macro_df = generate_macro_history()
    full_df = generate_portfolio_loss_rate(macro_df)
    full_df.to_csv("../data/raw/macro_and_loss_history.csv", index=False)
    print(f"Generated {len(full_df)} quarters of history -> data/raw/macro_and_loss_history.csv")
    print(f"Unemployment range: {full_df['unemployment_rate'].min():.2f}% - {full_df['unemployment_rate'].max():.2f}%")
    print(f"Loss rate range: {full_df['portfolio_loss_rate'].min():.2f}% - {full_df['portfolio_loss_rate'].max():.2f}%")
    print(full_df.tail(10))
