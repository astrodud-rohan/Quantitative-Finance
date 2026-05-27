"""
Visualization Module — HMM Regime Detection
============================================
Produces 4 publication-quality plots:
  1. Price history with regime shading
  2. Return distribution per regime (overlapping Gaussians)
  3. Regime probability heatmap over time
  4. Strategy comparison (regime-aware vs blind)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings("ignore")

# ── Style ──────────────────────────────────────────────────────────────────
DARK_BG    = "#0d1117"
PANEL_BG   = "#161b22"
GRID_COLOR = "#21262d"
TEXT_COLOR = "#e6edf3"

REGIME_COLORS = {
    0: "#f85149",   # Bear  — red
    1: "#d29922",   # Sideways — amber
    2: "#3fb950",   # Bull  — green
}
REGIME_LABELS = {0: "Bear", 1: "Sideways", 2: "Bull"}

ACCENT_BLUE   = "#58a6ff"
ACCENT_GREEN  = "#3fb950"
ACCENT_RED    = "#f85149"
ACCENT_AMBER  = "#d29922"
ACCENT_PURPLE = "#bc8cff"

plt.rcParams.update({
    "figure.facecolor": DARK_BG,
    "axes.facecolor":   PANEL_BG,
    "axes.edgecolor":   GRID_COLOR,
    "axes.labelcolor":  TEXT_COLOR,
    "axes.titlecolor":  TEXT_COLOR,
    "xtick.color":      TEXT_COLOR,
    "ytick.color":      TEXT_COLOR,
    "grid.color":       GRID_COLOR,
    "grid.alpha":       0.5,
    "text.color":       TEXT_COLOR,
    "legend.facecolor": PANEL_BG,
    "legend.edgecolor": GRID_COLOR,
    "font.family":      "monospace",
    "lines.linewidth":  1.2,
})


def plot_regime_timeline(
    prices: pd.Series,
    states: np.ndarray,
    n_states: int = 3,
    ticker: str = "NIFTY 50",
    save_path: str = None,
):
    """
    Main regime plot: price history with background shading per regime.
    Each regime gets a distinct color band behind the price series.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9),
                                    facecolor=DARK_BG,
                                    gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08})

    fig.suptitle(f"HMM Regime Detection — {ticker}",
                 fontsize=15, fontweight="bold", color=TEXT_COLOR, y=0.99)

    idx = prices.index

    # ── Panel 1: Price + regime shading ──────────────────────────────────
    ax1.plot(idx, prices.values, color=TEXT_COLOR, linewidth=1.1,
             alpha=0.9, zorder=5, label=ticker)

    # Shade regime backgrounds
    i = 0
    while i < len(states):
        current_state = states[i]
        j = i
        while j < len(states) and states[j] == current_state:
            j += 1
        ax1.axvspan(idx[i], idx[min(j, len(idx)-1)],
                    alpha=0.18,
                    color=REGIME_COLORS.get(current_state, ACCENT_BLUE),
                    zorder=1)
        i = j

    ax1.set_ylabel("Price", fontsize=11)
    ax1.set_title("Price History with Detected Market Regimes", fontsize=12, pad=6)
    ax1.grid(True, alpha=0.25)
    ax1.tick_params(labelbottom=False)

    # Legend patches
    patches = [
        mpatches.Patch(color=REGIME_COLORS[k], alpha=0.6,
                       label=REGIME_LABELS.get(k, f"State {k}"))
        for k in range(n_states)
    ]
    ax1.legend(handles=patches + [plt.Line2D([0], [0], color=TEXT_COLOR, lw=1.2,
               label=ticker)], loc="upper left", fontsize=10)

    # ── Panel 2: Regime state sequence ───────────────────────────────────
    for k in range(n_states):
        mask = states == k
        ax2.scatter(idx[mask],
                    np.full(mask.sum(), k),
                    c=REGIME_COLORS[k], s=2.5, alpha=0.8, zorder=3)

    ax2.set_yticks(range(n_states))
    ax2.set_yticklabels([REGIME_LABELS.get(k, f"State {k}") for k in range(n_states)],
                         fontsize=9)
    ax2.set_ylabel("Regime", fontsize=10)
    ax2.set_title("Viterbi-Decoded State Sequence", fontsize=11, pad=4)
    ax2.grid(True, alpha=0.2)
    ax2.set_ylim(-0.5, n_states - 0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
        print(f"Saved: {save_path}")
    plt.show()


def plot_emission_distributions(
    returns: np.ndarray,
    params,
    n_states: int = 3,
    save_path: str = None,
):
    """
    Plot the fitted Gaussian emission distributions per regime
    overlaid on the empirical return histogram.
    """
    from scipy.stats import norm as sp_norm

    fig, ax = plt.subplots(figsize=(12, 6), facecolor=DARK_BG)
    ax.set_facecolor(PANEL_BG)

    # Histogram of all returns
    ax.hist(returns * 100, bins=80, density=True,
            color=ACCENT_BLUE, alpha=0.25, label="Empirical returns", zorder=2)

    # Fitted Gaussian per regime
    x = np.linspace(returns.min() * 100 * 1.3, returns.max() * 100 * 1.3, 500)
    for k in range(n_states):
        mu_pct = params.mu[k] * 100
        sigma_pct = params.sigma[k] * 100
        y = sp_norm.pdf(x, loc=mu_pct, scale=sigma_pct)
        # Scale by regime weight (approximate)
        ax.plot(x, y, color=REGIME_COLORS[k], linewidth=2.2,
                label=f"{REGIME_LABELS.get(k, f'State {k}')}: "
                      f"μ={mu_pct:.3f}%, σ={sigma_pct:.3f}%",
                zorder=5)
        ax.fill_between(x, y, alpha=0.08, color=REGIME_COLORS[k])
        ax.axvline(mu_pct, color=REGIME_COLORS[k],
                   linewidth=0.9, linestyle="--", alpha=0.7)

    ax.set_xlabel("Daily Return (%)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Fitted Emission Distributions per Regime", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
        print(f"Saved: {save_path}")
    plt.show()


def plot_regime_probabilities(
    dates: pd.DatetimeIndex,
    gamma: np.ndarray,
    n_states: int = 3,
    save_path: str = None,
):
    """
    Stacked area chart of smoothed posterior regime probabilities over time.
    Shows uncertainty: when probabilities are close, the model is unsure.
    """
    fig, ax = plt.subplots(figsize=(16, 5), facecolor=DARK_BG)
    ax.set_facecolor(PANEL_BG)

    # Stacked area
    bottoms = np.zeros(len(dates))
    for k in range(n_states):
        ax.fill_between(dates, bottoms, bottoms + gamma[:, k],
                        color=REGIME_COLORS[k], alpha=0.75,
                        label=REGIME_LABELS.get(k, f"State {k}"))
        bottoms += gamma[:, k]

    ax.set_ylim(0, 1)
    ax.set_ylabel("Posterior Probability P(S_t | r_{1:T})", fontsize=10)
    ax.set_title("Smoothed Regime Posterior Probabilities (Forward-Backward)", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
        print(f"Saved: {save_path}")
    plt.show()


def plot_strategy_comparison(
    dates: pd.DatetimeIndex,
    strategy_results: dict,
    states: np.ndarray,
    n_states: int = 3,
    save_path: str = None,
):
    """
    Compare cumulative returns of regime-aware vs regime-blind strategies,
    with regime shading in the background.
    """
    fig, ax = plt.subplots(figsize=(16, 7), facecolor=DARK_BG)
    ax.set_facecolor(PANEL_BG)

    # Regime background shading
    i = 0
    while i < len(states):
        s = states[i]
        j = i
        while j < len(states) and states[j] == s:
            j += 1
        ax.axvspan(dates[i], dates[min(j, len(dates)-1)],
                   alpha=0.1, color=REGIME_COLORS.get(s, ACCENT_BLUE))
        i = j

    # Strategy lines
    colors = [ACCENT_BLUE, ACCENT_AMBER, ACCENT_GREEN]
    styles = ["--", "-.", "-"]
    for (key, strat), color, style in zip(strategy_results.items(), colors, styles):
        cr = strat["cumulative_returns"] * 100
        ax.plot(dates, cr, color=color, linewidth=1.6,
                linestyle=style,
                label=f"{strat['label']}  "
                      f"[Ret={strat['final_return_pct']:.1f}%, "
                      f"Sharpe={strat['sharpe']:.2f}, "
                      f"MDD={strat['max_drawdown_pct']:.1f}%]")

    ax.axhline(0, color=TEXT_COLOR, linewidth=0.5, alpha=0.4)
    ax.set_ylabel("Cumulative Return (%)", fontsize=11)
    ax.set_title("Regime-Aware vs Regime-Blind Strategy Performance", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.25)

    # Regime legend
    patches = [
        mpatches.Patch(color=REGIME_COLORS[k], alpha=0.5,
                       label=REGIME_LABELS.get(k, f"State {k}"))
        for k in range(n_states)
    ]
    ax.legend(loc="upper left", fontsize=9,
              handles=ax.get_legend_handles_labels()[0] + patches)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
        print(f"Saved: {save_path}")
    plt.show()


def plot_var_comparison(
    returns: np.ndarray,
    states: np.ndarray,
    n_states: int = 3,
    save_path: str = None,
):
    """
    Box plots of daily returns per regime + VaR lines.
    Visually shows how risk differs across regimes.
    """
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=DARK_BG)
    ax.set_facecolor(PANEL_BG)

    data = [returns[states == k] * 100 for k in range(n_states)]
    labels = [REGIME_LABELS.get(k, f"State {k}") for k in range(n_states)]
    colors_list = [REGIME_COLORS[k] for k in range(n_states)]

    bp = ax.boxplot(data, labels=labels, patch_artist=True,
                    medianprops=dict(color=TEXT_COLOR, linewidth=2),
                    whiskerprops=dict(color=TEXT_COLOR, alpha=0.7),
                    capprops=dict(color=TEXT_COLOR),
                    flierprops=dict(marker="o", markersize=2, alpha=0.4))

    for patch, color in zip(bp["boxes"], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.4)

    # Add 95% VaR line per regime
    for k in range(n_states):
        r_k = returns[states == k]
        if len(r_k) > 5:
            var95 = np.percentile(r_k, 5) * 100
            ax.hlines(var95, k + 0.6, k + 1.4,
                      colors=colors_list[k], linewidths=2,
                      linestyles="--", label=f"95% VaR ({labels[k]}): {var95:.2f}%")

    ax.set_xlabel("Regime", fontsize=11)
    ax.set_ylabel("Daily Return (%)", fontsize=11)
    ax.set_title("Return Distribution & 95% VaR by Regime", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.25, axis="y")
    ax.axhline(0, color=TEXT_COLOR, linewidth=0.5, alpha=0.4)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
        print(f"Saved: {save_path}")
    plt.show()
