"""
main.py — HMM Regime Detection Pipeline
=========================================
Full pipeline:
  1. Download / generate market return data
  2. Fit 3-state Gaussian HMM via Baum-Welch
  3. Decode regimes via Viterbi
  4. Compute regime-conditional statistics and VaR
  5. Compare regime-aware vs blind strategy
  6. Generate all plots

Usage:
    python main.py                        # NIFTY 50 (synthetic if offline)
    python main.py --ticker "^GSPC"       # S&P 500
    python main.py --ticker "^NSEI"       # NIFTY 50 live
    python main.py --states 2             # 2-state model (bear/bull)
    python main.py --synthetic            # Always use synthetic data
"""

import argparse
import numpy as np
import pandas as pd
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.hmm import GaussianHMM
from src.regime_analysis import (
    compute_regime_stats,
    print_regime_stats,
    regime_conditional_var,
    regime_conditional_strategy,
    print_strategy_comparison,
)
from src.visualization import (
    plot_regime_timeline,
    plot_emission_distributions,
    plot_regime_probabilities,
    plot_strategy_comparison,
    plot_var_comparison,
)

os.makedirs("results", exist_ok=True)


def generate_synthetic_data(n_days: int = 2520, seed: int = 42) -> pd.Series:
    """
    Generate synthetic return series with known regime structure.
    3 regimes with realistic parameters matching empirical equity indices.

    True parameters (annualised):
      Bull:     mu = +18%,  sigma = 12%
      Sideways: mu = +4%,   sigma = 18%
      Bear:     mu = -25%,  sigma = 32%
    """
    rng = np.random.default_rng(seed)

    # Daily parameters
    regimes = {
        0: {"mu": -0.25/252, "sigma": 0.32/np.sqrt(252), "label": "Bear"},
        1: {"mu":  0.04/252, "sigma": 0.18/np.sqrt(252), "label": "Sideways"},
        2: {"mu":  0.18/252, "sigma": 0.12/np.sqrt(252), "label": "Bull"},
    }

    # Transition matrix: persistent regimes
    A = np.array([
        [0.975, 0.015, 0.010],  # Bear → {Bear, Sideways, Bull}
        [0.015, 0.960, 0.025],  # Sideways → {Bear, Sideways, Bull}
        [0.008, 0.022, 0.970],  # Bull → {Bear, Sideways, Bull}
    ])

    # Generate state sequence via Markov chain
    states_true = np.zeros(n_days, dtype=int)
    states_true[0] = 2  # Start in bull regime
    for t in range(1, n_days):
        states_true[t] = rng.choice(3, p=A[states_true[t-1]])

    # Generate returns given states
    returns = np.zeros(n_days)
    for t in range(n_days):
        k = states_true[t]
        returns[t] = rng.normal(regimes[k]["mu"], regimes[k]["sigma"])

    # Build price series from returns
    prices = 10000 * np.cumprod(1 + returns)
    dates = pd.date_range("2015-01-02", periods=n_days, freq="B")

    print(f"Synthetic data: {n_days} days, {n_days/252:.1f} years")
    print(f"True regime distribution:")
    for k in range(3):
        n = (states_true == k).sum()
        print(f"  {regimes[k]['label']}: {n} days ({n/n_days*100:.1f}%)")

    return_series = pd.Series(returns, index=dates, name="Returns")
    price_series = pd.Series(prices, index=dates, name="Price")
    return return_series, price_series, states_true


def download_data(ticker: str, years: int = 10) -> tuple:
    """Download real market data via yfinance."""
    try:
        import yfinance as yf
        from datetime import datetime, timedelta
        end = datetime.today().strftime("%Y-%m-%d")
        start = (datetime.today() - timedelta(days=years*365)).strftime("%Y-%m-%d")

        print(f"Downloading {ticker} from {start} to {end}...")
        df = yf.download(ticker, start=start, end=end,
                         auto_adjust=True, progress=False)

        if df.empty or len(df) < 500:
            raise ValueError("Insufficient data downloaded")

        prices = df["Close"].squeeze()
        returns = np.log(prices / prices.shift(1)).dropna()
        prices = prices.loc[returns.index]

        print(f"Downloaded {len(returns)} trading days")
        return returns, prices, None

    except Exception as e:
        print(f"Download failed ({e}). Using synthetic data.")
        return None, None, None


def run_pipeline(
    ticker: str = "^NSEI",
    n_states: int = 3,
    use_synthetic: bool = False,
    years: int = 10,
):
    print(f"\n{'═'*65}")
    print(f"  HMM REGIME DETECTION")
    print(f"  Ticker: {ticker}  |  States: {n_states}")
    print(f"{'═'*65}\n")

    # ── Step 1: Data ─────────────────────────────────────────────────────
    print("► Step 1: Loading market data...")
    states_true = None

    if not use_synthetic:
        returns, prices, states_true = download_data(ticker, years=years)

    if use_synthetic or returns is None:
        print("  Using synthetic data with known regime structure...")
        returns, prices, states_true = generate_synthetic_data(n_days=2520)
        ticker = "NIFTY 50 (Synthetic)"

    print(f"  Return stats: mean={returns.mean()*100:.4f}%/day, "
          f"std={returns.std()*100:.4f}%/day")
    print(f"  Skewness: {returns.skew():.3f}, Kurtosis: {returns.kurtosis():.3f}")

    # ── Step 2: Fit HMM ──────────────────────────────────────────────────
    print(f"\n► Step 2: Fitting {n_states}-state Gaussian HMM via Baum-Welch...")
    model = GaussianHMM(n_states=n_states, n_iter=300, tol=1e-8, random_seed=42)
    model.fit(returns.values)

    print(f"\n  Converged: {model.converged}")
    print(f"  Final log-likelihood: {model.log_likelihoods[-1]:.4f}")

    # ── Step 3: Decode Regimes ───────────────────────────────────────────
    print(f"\n► Step 3: Decoding regime sequence via Viterbi...")
    states = model.decode(returns.values)
    gamma = model.predict_proba(returns.values)

    labels = {0: "Bear", 1: "Sideways", 2: "Bull"}
    for k in range(n_states):
        n_k = (states == k).sum()
        print(f"  {labels.get(k, f'State {k}')}: "
              f"{n_k} days ({n_k/len(states)*100:.1f}%)")

    # ── Step 4: Regime Statistics ────────────────────────────────────────
    print(f"\n► Step 4: Computing regime-conditional statistics...")
    stats = compute_regime_stats(returns.values, states, n_states=n_states)
    print_regime_stats(stats)

    # VaR comparison
    current_state = states[-1]
    var_result = regime_conditional_var(returns.values, states, current_state)
    print(f"\n  Current regime: {var_result['current_regime']}")
    print(f"  Regime-conditional 95% VaR: {var_result['conditional_var']:.4f}%")
    print(f"  Unconditional 95% VaR:      {var_result['unconditional_var']:.4f}%")
    print(f"  VaR ratio (conditional/unconditional): {var_result['var_ratio']:.3f}x")

    # ── Step 5: Strategy Comparison ──────────────────────────────────────
    print(f"\n► Step 5: Strategy comparison (regime-aware vs blind)...")
    strategy_results = regime_conditional_strategy(
        returns, states,
        bull_state=n_states-1,
        bear_state=0,
    )
    print_strategy_comparison(strategy_results)

    # ── Step 6: Plots ─────────────────────────────────────────────────────
    print(f"\n► Step 6: Generating plots...")

    plot_regime_timeline(
        prices, states, n_states=n_states, ticker=ticker,
        save_path="results/regime_timeline.png"
    )

    plot_emission_distributions(
        returns.values, model.params, n_states=n_states,
        save_path="results/emission_distributions.png"
    )

    plot_regime_probabilities(
        returns.index, gamma, n_states=n_states,
        save_path="results/regime_probabilities.png"
    )

    plot_strategy_comparison(
        returns.index, strategy_results, states, n_states=n_states,
        save_path="results/strategy_comparison.png"
    )

    plot_var_comparison(
        returns.values, states, n_states=n_states,
        save_path="results/var_comparison.png"
    )

    print(f"\n✓ All plots saved to results/")
    print(f"✓ Done.\n")

    return model, states, strategy_results


def main():
    parser = argparse.ArgumentParser(description="HMM Market Regime Detection")
    parser.add_argument("--ticker", type=str, default="^NSEI",
                        help="Yahoo Finance ticker (default: ^NSEI = NIFTY 50)")
    parser.add_argument("--states", type=int, default=3,
                        help="Number of HMM states (default: 3)")
    parser.add_argument("--years", type=int, default=10,
                        help="Years of history (default: 10)")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data (for testing/demo)")
    args = parser.parse_args()

    run_pipeline(
        ticker=args.ticker,
        n_states=args.states,
        use_synthetic=args.synthetic,
        years=args.years,
    )


if __name__ == "__main__":
    main()
