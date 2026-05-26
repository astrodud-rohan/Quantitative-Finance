"""
Signal Generation & Backtesting Engine
=======================================
Converts Kalman-filtered spread into z-score signals, then simulates
a realistic backtest with transaction costs and position sizing.

Trading logic:
  z > +entry_threshold  → SHORT spread  (sell A, buy B)
  z < -entry_threshold  → LONG spread   (buy A, sell B)
  |z| < exit_threshold  → CLOSE position
  |z| > stop_threshold  → STOP LOSS (cointegration may have broken)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class BacktestConfig:
    entry_zscore: float = 1.5      # Open trade when |z| crosses this
    exit_zscore: float = 0.3       # Close trade when |z| falls below this
    stop_zscore: float = 3.5       # Stop loss — cointegration breakdown
    transaction_cost_bps: float = 10.0  # Round-trip cost in basis points
    capital: float = 1_000_000.0  # Starting capital in INR
    max_position_pct: float = 0.3  # Max 30% of capital per trade


def compute_zscore(
    spread: np.ndarray, window: int = 30
) -> np.ndarray:
    """
    Rolling z-score of the spread.
    Uses a rolling window to capture local mean and std.

    z_t = (spread_t - mean(spread_{t-window:t})) / std(spread_{t-window:t})
    """
    spread_series = pd.Series(spread)
    rolling_mean = spread_series.rolling(window=window, min_periods=window).mean()
    rolling_std = spread_series.rolling(window=window, min_periods=window).std()

    zscore = (spread_series - rolling_mean) / rolling_std
    return zscore.values


def generate_signals(zscore: np.ndarray, config: BacktestConfig) -> np.ndarray:
    """
    Convert z-scores into position signals.

    Returns
    -------
    signals : array of {-1, 0, 1}
        +1 = long spread (buy A, sell B)
        -1 = short spread (sell A, buy B)
         0 = flat
    """
    signals = np.zeros(len(zscore))
    position = 0  # Current position state

    for i in range(1, len(zscore)):
        z = zscore[i]

        if np.isnan(z):
            signals[i] = 0
            continue

        if position == 0:
            # Entry logic
            if z < -config.entry_zscore:
                position = 1   # Long spread
            elif z > config.entry_zscore:
                position = -1  # Short spread

        elif position == 1:
            # Exit or stop long spread
            if z > -config.exit_zscore:
                position = 0
            elif z > config.stop_zscore:
                position = 0  # Stop loss

        elif position == -1:
            # Exit or stop short spread
            if z < config.exit_zscore:
                position = 0
            elif z < -config.stop_zscore:
                position = 0  # Stop loss

        signals[i] = position

    return signals


def run_backtest(
    prices_a: pd.Series,
    prices_b: pd.Series,
    betas: np.ndarray,
    spreads: np.ndarray,
    signals: np.ndarray,
    config: BacktestConfig,
) -> pd.DataFrame:
    """
    Simulate P&L from signals with realistic transaction costs.

    Strategy:
    - When signal = +1: BUY 1 unit of A, SELL beta units of B
    - When signal = -1: SELL 1 unit of A, BUY beta units of B
    - P&L = change in spread value * position size

    Returns
    -------
    DataFrame with columns: date, signal, spread, zscore, pnl, cum_pnl,
                            position_value, trade_flag
    """
    n = len(prices_a)
    pnl = np.zeros(n)
    trade_costs = np.zeros(n)
    prev_signal = 0

    # Position size: fixed notional per trade
    position_value = config.capital * config.max_position_pct

    for i in range(1, n):
        curr_signal = signals[i]
        prev_signal_val = signals[i - 1]

        # P&L from holding position: spread change * position
        spread_change = spreads[i] - spreads[i - 1]
        pnl[i] = curr_signal * spread_change * (position_value / prices_a.iloc[i])

        # Transaction cost on trade entry/exit
        if curr_signal != prev_signal_val:
            cost = position_value * (config.transaction_cost_bps / 10000.0)
            trade_costs[i] = cost
            pnl[i] -= cost

    # Build results DataFrame
    results = pd.DataFrame(
        {
            "price_a": prices_a.values,
            "price_b": prices_b.values,
            "beta": betas,
            "spread": spreads,
            "signal": signals,
            "pnl": pnl,
            "trade_cost": trade_costs,
            "cum_pnl": np.cumsum(pnl),
            "trade_flag": (np.diff(signals, prepend=0) != 0).astype(int),
        },
        index=prices_a.index,
    )

    return results


def compute_metrics(results: pd.DataFrame, config: BacktestConfig) -> dict:
    """
    Compute standard quant performance metrics.
    """
    pnl = results["pnl"]
    cum_pnl = results["cum_pnl"]

    # Annualised return
    total_return = cum_pnl.iloc[-1] / config.capital
    n_years = len(pnl) / 252
    annual_return = (1 + total_return) ** (1 / n_years) - 1

    # Sharpe ratio (annualised, assuming risk-free rate = 6.5% for India)
    rf_daily = 0.065 / 252
    excess_returns = pnl / config.capital - rf_daily
    sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)

    # Maximum drawdown
    rolling_max = cum_pnl.cummax()
    drawdown = (cum_pnl - rolling_max) / config.capital
    max_drawdown = drawdown.min()

    # Trade statistics
    trades = results[results["trade_flag"] == 1]
    n_trades = len(trades) // 2  # Entry + exit = 1 trade

    # Win rate (per-day P&L when in position)
    in_position = results[results["signal"] != 0]["pnl"]
    hit_rate = (in_position > 0).mean()

    # Profit factor
    gross_profit = in_position[in_position > 0].sum()
    gross_loss = abs(in_position[in_position < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    # Calmar ratio
    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else np.inf

    metrics = {
        "Total Return (%)": round(total_return * 100, 2),
        "Annual Return (%)": round(annual_return * 100, 2),
        "Sharpe Ratio": round(sharpe, 3),
        "Max Drawdown (%)": round(max_drawdown * 100, 2),
        "Calmar Ratio": round(calmar, 3),
        "Hit Rate (%)": round(hit_rate * 100, 2),
        "Profit Factor": round(profit_factor, 3),
        "Number of Trades": n_trades,
        "Total P&L (INR)": round(cum_pnl.iloc[-1], 0),
        "Total Transaction Costs (INR)": round(results["trade_cost"].sum(), 0),
    }

    return metrics