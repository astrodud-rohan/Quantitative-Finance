"""
Visualization Module
====================
Produces publication-quality plots suitable for a GitHub README.
All plots use a dark theme with a clean quant-research aesthetic.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import warnings
warnings.filterwarnings("ignore")

# ── Style ──────────────────────────────────────────────────────────────────
DARK_BG = "#0d1117"
PANEL_BG = "#161b22"
GRID_COLOR = "#21262d"
TEXT_COLOR = "#e6edf3"
ACCENT_BLUE = "#58a6ff"
ACCENT_GREEN = "#3fb950"
ACCENT_RED = "#f85149"
ACCENT_AMBER = "#d29922"
ACCENT_PURPLE = "#bc8cff"

plt.rcParams.update({
    "figure.facecolor": DARK_BG,
    "axes.facecolor": PANEL_BG,
    "axes.edgecolor": GRID_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "axes.titlecolor": TEXT_COLOR,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "grid.color": GRID_COLOR,
    "grid.alpha": 0.6,
    "text.color": TEXT_COLOR,
    "legend.facecolor": PANEL_BG,
    "legend.edgecolor": GRID_COLOR,
    "font.family": "monospace",
    "lines.linewidth": 1.2,
})


def plot_price_comparison(
    prices_a: pd.Series,
    prices_b: pd.Series,
    label_a: str,
    label_b: str,
    save_path: str = None,
):
    """Plot normalized price series of the two assets."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), facecolor=DARK_BG)
    fig.suptitle(f"Price Series: {label_a} vs {label_b}", fontsize=14,
                 color=TEXT_COLOR, fontweight="bold", y=0.98)

    # Normalized prices (base 100)
    norm_a = prices_a / prices_a.iloc[0] * 100
    norm_b = prices_b / prices_b.iloc[0] * 100

    ax1.plot(norm_a.index, norm_a.values, color=ACCENT_BLUE, label=label_a, linewidth=1.3)
    ax1.plot(norm_b.index, norm_b.values, color=ACCENT_AMBER, label=label_b, linewidth=1.3)
    ax1.set_title("Normalized Prices (Base = 100)", fontsize=11)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylabel("Normalized Price")

    # Price ratio
    ratio = prices_a / prices_b
    ax2.plot(ratio.index, ratio.values, color=ACCENT_PURPLE, linewidth=1.2)
    ax2.axhline(ratio.mean(), color=ACCENT_GREEN, linestyle="--",
                alpha=0.7, label=f"Mean ratio: {ratio.mean():.3f}")
    ax2.fill_between(ratio.index,
                     ratio.mean() - ratio.std(),
                     ratio.mean() + ratio.std(),
                     alpha=0.15, color=ACCENT_PURPLE, label="±1 std")
    ax2.set_title("Price Ratio (A/B)", fontsize=11)
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylabel("Ratio")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
        print(f"Saved: {save_path}")
    plt.show()


def plot_kalman_hedge_ratio(
    prices_a: pd.Series,
    betas: np.ndarray,
    label_a: str,
    label_b: str,
    save_path: str = None,
):
    """Plot how the dynamic hedge ratio evolves vs static OLS ratio."""
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant

    # Static OLS beta for comparison
    X = add_constant(np.arange(len(prices_a)))
    static_beta = OLS(
        np.ones(len(prices_a)) * betas.mean(), X
    ).fit().params[0]

    fig, ax = plt.subplots(figsize=(14, 5), facecolor=DARK_BG)

    ax.plot(prices_a.index, betas, color=ACCENT_BLUE,
            linewidth=1.3, label="Kalman Dynamic β (hedge ratio)")
    ax.axhline(betas.mean(), color=ACCENT_AMBER, linestyle="--",
               alpha=0.8, linewidth=1.0, label=f"Mean β = {betas.mean():.3f}")
    ax.fill_between(prices_a.index,
                    betas - betas.std(),
                    betas + betas.std(),
                    alpha=0.12, color=ACCENT_BLUE)

    ax.set_title(f"Kalman Filter: Dynamic Hedge Ratio  ({label_a} / {label_b})",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Hedge Ratio (β)", fontsize=11)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
        print(f"Saved: {save_path}")
    plt.show()


def plot_spread_and_signals(
    results: pd.DataFrame,
    config,
    label_a: str,
    label_b: str,
    save_path: str = None,
):
    """
    Main strategy plot: spread, z-score, and trading signals.
    """
    fig = plt.figure(figsize=(16, 12), facecolor=DARK_BG)
    gs = gridspec.GridSpec(3, 1, hspace=0.12, figure=fig)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)

    fig.suptitle(
        f"Pairs Trading Strategy: {label_a} / {label_b}",
        fontsize=15, fontweight="bold", color=TEXT_COLOR, y=0.99
    )

    idx = results.index
    spread = results["spread"]
    signal = results["signal"]

    # Compute z-score
    roll_mean = spread.rolling(30).mean()
    roll_std = spread.rolling(30).std()
    zscore = (spread - roll_mean) / roll_std

    # ── Panel 1: Spread ─────────────────────────────────────────────────
    ax1.plot(idx, spread, color=ACCENT_BLUE, linewidth=0.9, alpha=0.9)
    ax1.plot(idx, roll_mean, color=ACCENT_AMBER, linewidth=1.0,
             linestyle="--", alpha=0.7, label="30d Rolling Mean")
    ax1.fill_between(idx, roll_mean - roll_std, roll_mean + roll_std,
                     alpha=0.1, color=ACCENT_BLUE)
    ax1.set_ylabel("Spread (₹)", fontsize=10)
    ax1.set_title("Kalman-Filtered Spread", fontsize=11, pad=4)
    ax1.legend(fontsize=9, loc="upper left")
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: Z-Score with signal bands ──────────────────────────────
    ax2.plot(idx, zscore, color=ACCENT_PURPLE, linewidth=0.9, alpha=0.9, label="Z-Score")
    ax2.axhline(config.entry_zscore, color=ACCENT_RED, linewidth=1.0,
                linestyle="--", alpha=0.8, label=f"Entry ±{config.entry_zscore}")
    ax2.axhline(-config.entry_zscore, color=ACCENT_GREEN, linewidth=1.0,
                linestyle="--", alpha=0.8)
    ax2.axhline(config.stop_zscore, color=ACCENT_RED, linewidth=0.7,
                linestyle=":", alpha=0.6, label=f"Stop ±{config.stop_zscore}")
    ax2.axhline(-config.stop_zscore, color=ACCENT_RED, linewidth=0.7,
                linestyle=":", alpha=0.6)
    ax2.axhline(0, color=TEXT_COLOR, linewidth=0.5, alpha=0.4)
    ax2.fill_between(idx, -config.exit_zscore, config.exit_zscore,
                     alpha=0.08, color=ACCENT_GREEN, label=f"Exit zone ±{config.exit_zscore}")
    ax2.set_ylabel("Z-Score", fontsize=10)
    ax2.set_title("Spread Z-Score & Trading Bands", fontsize=11, pad=4)
    ax2.legend(fontsize=9, loc="upper left", ncol=2)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-5, 5)

    # Shade long/short regions
    ax2.fill_between(idx, zscore, 0,
                     where=signal == 1, alpha=0.15, color=ACCENT_GREEN,
                     label="Long spread")
    ax2.fill_between(idx, zscore, 0,
                     where=signal == -1, alpha=0.15, color=ACCENT_RED,
                     label="Short spread")

    # ── Panel 3: Cumulative P&L ──────────────────────────────────────────
    cum_pnl = results["cum_pnl"]
    ax3.fill_between(idx, cum_pnl, 0,
                     where=cum_pnl >= 0, alpha=0.3, color=ACCENT_GREEN)
    ax3.fill_between(idx, cum_pnl, 0,
                     where=cum_pnl < 0, alpha=0.3, color=ACCENT_RED)
    ax3.plot(idx, cum_pnl, color=ACCENT_GREEN, linewidth=1.2, label="Cumulative P&L")
    ax3.axhline(0, color=TEXT_COLOR, linewidth=0.5, alpha=0.4)

    # Annotate final P&L
    final_pnl = cum_pnl.iloc[-1]
    ax3.annotate(
        f"Final: ₹{final_pnl:,.0f}",
        xy=(idx[-1], final_pnl),
        xytext=(-100, 15 if final_pnl >= 0 else -25),
        textcoords="offset points",
        color=ACCENT_GREEN if final_pnl >= 0 else ACCENT_RED,
        fontsize=10, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=TEXT_COLOR, lw=0.8)
    )

    ax3.set_ylabel("Cumulative P&L (₹)", fontsize=10)
    ax3.set_title("Strategy P&L (after transaction costs)", fontsize=11, pad=4)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=9, loc="upper left")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
        print(f"Saved: {save_path}")
    plt.show()


def plot_performance_summary(metrics: dict, save_path: str = None):
    """Print and optionally save a clean metrics summary table."""
    print("\n" + "═" * 50)
    print("  STRATEGY PERFORMANCE SUMMARY")
    print("═" * 50)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<35} {v:>10.2f}")
        else:
            print(f"  {k:<35} {v:>10,}")
    print("═" * 50)