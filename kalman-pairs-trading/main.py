"""
main.py — Full Pipeline Runner
================================
Orchestrates the complete pairs trading workflow:

  1. Download price data
  2. Screen for cointegrated pairs
  3. Run Kalman Filter for dynamic hedge ratio
  4. Generate z-score signals
  5. Backtest with transaction costs
  6. Plot and report metrics

Usage:
    python main.py                         # Run default pair (HDFCBANK / ICICIBANK)
    python main.py --sector banking        # Screen full banking universe
    python main.py --pair INFY.NS TCS.NS  # Run specific pair
    python main.py --us                    # Use US stocks (more reliable data)
"""

import argparse
import numpy as np
import pandas as pd
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.data import get_pair_data, get_sector_universe, CANDIDATE_PAIRS, US_CANDIDATE_PAIRS
from src.cointegration import find_cointegrated_pairs, test_cointegration, adf_test
from src.kalman_filter import KalmanFilterHedge
from src.backtest import BacktestConfig, compute_zscore, generate_signals, run_backtest, compute_metrics
from src.visualization import (
    plot_price_comparison,
    plot_kalman_hedge_ratio,
    plot_spread_and_signals,
    plot_performance_summary,
)

os.makedirs("results", exist_ok=True)


def run_single_pair(ticker_a: str, ticker_b: str, years: int = 4):
    """Run full pipeline on a single pair."""

    label_a = ticker_a.replace(".NS", "")
    label_b = ticker_b.replace(".NS", "")

    print(f"\n{'═'*60}")
    print(f"  PAIRS TRADING: {label_a} / {label_b}")
    print(f"{'═'*60}\n")

    # ── Step 1: Download Data ────────────────────────────────────────────
    print("► Step 1: Downloading price data...")
    prices_a, prices_b = get_pair_data(ticker_a, ticker_b, years=years)

    # ── Step 2: Cointegration Test ───────────────────────────────────────
    print("\n► Step 2: Testing cointegration...")
    coint_result = test_cointegration(prices_a, prices_b)

    print(f"  Cointegration p-value : {coint_result['pvalue']:.4f}")
    print(f"  Cointegrated?         : {'YES ✓' if coint_result['is_cointegrated'] else 'NO ✗'}")
    print(f"  Static hedge ratio    : {coint_result['hedge_ratio']:.4f}")
    print(f"  Mean reversion HL     : {coint_result['half_life']} days")

    if not coint_result["is_cointegrated"]:
        print("\n  WARNING: Pair is not cointegrated (p > 0.05).")
        print("  Proceeding anyway — Kalman Filter may still find tradeable signals.")
        print("  Consider screening a different pair.\n")

    # ADF test on raw price series (should be non-stationary — unit root)
    print("\n  ADF test on raw prices (expect non-stationary):")
    adf_test(prices_a.values, label=label_a)
    adf_test(prices_b.values, label=label_b)

    # ── Step 3: Kalman Filter ────────────────────────────────────────────
    print("\n► Step 3: Running Kalman Filter for dynamic hedge ratio...")

    # delta controls adaptation speed:
    # Lower delta (1e-5) = slower, smoother beta changes
    # Higher delta (1e-3) = faster adaptation, noisier
    kf = KalmanFilterHedge(delta=1e-4, R=1.0)
    kf_results = kf.run(prices_a.values, prices_b.values)

    betas = kf_results["betas"]
    spreads = kf_results["spreads"]

    print(f"  Beta range: [{betas.min():.3f}, {betas.max():.3f}]")
    print(f"  Beta mean : {betas.mean():.3f} ± {betas.std():.3f}")
    print(f"  Spread std: {spreads.std():.4f}")

    # ADF test on Kalman spread (should be stationary)
    print("\n  ADF test on Kalman spread (expect stationary):")
    adf_test(spreads, label="Kalman Spread")

    # ── Step 4: Signals ──────────────────────────────────────────────────
    print("\n► Step 4: Generating trading signals...")
    config = BacktestConfig(
        entry_zscore=1.5,
        exit_zscore=0.3,
        stop_zscore=3.5,
        transaction_cost_bps=10.0,
        capital=1_000_000.0,
        max_position_pct=0.3,
    )

    zscore = compute_zscore(spreads, window=30)
    signals = generate_signals(zscore, config)

    n_long = (signals == 1).sum()
    n_short = (signals == -1).sum()
    n_flat = (signals == 0).sum()
    print(f"  Days long : {n_long} ({n_long/len(signals)*100:.1f}%)")
    print(f"  Days short: {n_short} ({n_short/len(signals)*100:.1f}%)")
    print(f"  Days flat : {n_flat} ({n_flat/len(signals)*100:.1f}%)")

    # ── Step 5: Backtest ─────────────────────────────────────────────────
    print("\n► Step 5: Running backtest...")
    results = run_backtest(prices_a, prices_b, betas, spreads, signals, config)
    results["zscore"] = zscore

    # ── Step 6: Metrics ──────────────────────────────────────────────────
    print("\n► Step 6: Computing performance metrics...")
    metrics = compute_metrics(results, config)
    plot_performance_summary(metrics)

    # ── Step 7: Plots ────────────────────────────────────────────────────
    print("\n► Step 7: Generating plots...")

    plot_price_comparison(
        prices_a, prices_b, label_a, label_b,
        save_path=f"results/{label_a}_{label_b}_prices.png"
    )

    plot_kalman_hedge_ratio(
        prices_a, betas, label_a, label_b,
        save_path=f"results/{label_a}_{label_b}_hedge_ratio.png"
    )

    plot_spread_and_signals(
        results, config, label_a, label_b,
        save_path=f"results/{label_a}_{label_b}_strategy.png"
    )

    print(f"\n✓ All plots saved to results/")
    print(f"✓ Done.\n")

    return results, metrics


def run_sector_screen(sector: str = "banking", years: int = 4):
    """Screen a full sector universe for cointegrated pairs, then trade the best one."""

    print(f"\n{'═'*60}")
    print(f"  SECTOR SCREENING: {sector.upper()}")
    print(f"{'═'*60}\n")

    print("► Downloading sector universe...")
    prices = get_sector_universe(sector, years=years)

    print("\n► Screening for cointegrated pairs...")
    pairs_df = find_cointegrated_pairs(
        prices,
        pvalue_threshold=0.05,
        min_half_life=5.0,
        max_half_life=120.0,
    )

    if pairs_df.empty:
        print("No suitable pairs found. Try a different sector or time period.")
        return

    # Trade the best pair
    best = pairs_df.iloc[0]
    print(f"\n► Best pair: {best['asset_a']} / {best['asset_b']}")
    print(f"  p-value: {best['pvalue']}  |  half-life: {best['half_life_days']} days")

    run_single_pair(best["asset_a"], best["asset_b"], years=years)


def main():
    parser = argparse.ArgumentParser(
        description="Kalman Filter Pairs Trading Strategy"
    )
    parser.add_argument("--pair", nargs=2, metavar=("A", "B"),
                        help="Specific pair tickers, e.g. INFY.NS TCS.NS")
    parser.add_argument("--sector", type=str, default=None,
                        help="Screen a sector: banking, it, pharma, fmcg, energy")
    parser.add_argument("--us", action="store_true",
                        help="Use US stock pairs (more reliable data for testing)")
    parser.add_argument("--years", type=int, default=4,
                        help="Years of historical data to use (default: 4)")
    args = parser.parse_args()

    if args.sector:
        run_sector_screen(args.sector, years=args.years)

    elif args.pair:
        run_single_pair(args.pair[0], args.pair[1], years=args.years)

    elif args.us:
        # Default US pair: Gold vs Silver ETFs — highly cointegrated
        run_single_pair("GLD", "SLV", years=args.years)

    else:
        # Default: HDFC Bank vs ICICI Bank
        run_single_pair("HDFCBANK.NS", "ICICIBANK.NS", years=args.years)


if __name__ == "__main__":
    main()