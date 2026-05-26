# Kalman Filter Pairs Trading Strategy
### Dynamic Hedge Ratio Estimation via State-Space Modelling

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Domain](https://img.shields.io/badge/Domain-Quantitative%20Finance-orange)

---

## Overview

A statistical arbitrage (stat-arb) strategy that trades the mean-reverting spread between cointegrated equity pairs. The key contribution is replacing the standard static OLS hedge ratio with a **Kalman Filter** that tracks a continuously drifting hedge ratio in real time — the same state-estimation technique used in spacecraft orbital mechanics and radar tracking, adapted to financial time series.

**The core insight:** The relationship between two cointegrated stocks is not static. It drifts slowly due to changing business conditions, macro regimes, and market structure shifts. A static OLS beta estimated two years ago is stale. The Kalman Filter treats the hedge ratio as a hidden state that we estimate optimally at every timestep given all available observations.

---

## Mathematical Framework

### Why Cointegration, Not Correlation

Two price series $P_A$ and $P_B$ are **cointegrated** if there exists a linear combination:

$$z_t = P_A - \beta \cdot P_B - \alpha$$

that is stationary (I(0)), even though $P_A$ and $P_B$ individually are non-stationary random walks (I(1)). This guarantees mean reversion in the spread — correlation does not.

Tested using the **Engle-Granger two-step method**. Half-life of mean reversion estimated via AR(1) regression on the spread.

### Kalman Filter State-Space Model

The hedge ratio $\beta_t$ is modelled as a latent state that evolves as a random walk:

$$\beta_t = \beta_{t-1} + w_t, \quad w_t \sim \mathcal{N}(0, Q)$$

Each new price observation is a noisy measurement of this state:

$$P_{A,t} = \beta_t \cdot P_{B,t} + \alpha + v_t, \quad v_t \sim \mathcal{N}(0, R)$$

**Predict step:**
$$\hat{\beta}_{t|t-1} = \hat{\beta}_{t-1}, \qquad P_{t|t-1} = P_{t-1} + Q$$

**Update step:**
$$K_t = \frac{P_{t|t-1} \cdot P_{B,t}}{P_{B,t}^2 \cdot P_{t|t-1} + R}$$
$$\hat{\beta}_{t|t} = \hat{\beta}_{t|t-1} + K_t \cdot (P_{A,t} - \hat{\beta}_{t|t-1} \cdot P_{B,t} - \alpha)$$

The **process noise** parameter $Q = \delta / (1 - \delta)$ controls how fast $\beta$ is allowed to change. This is the sole tunable hyperparameter and has an intuitive interpretation analogous to the maneuverability parameter in target tracking systems.

### Trading Signal

The dynamic spread is z-scored against a 30-day rolling window:

$$\text{zscore}_t = \frac{z_t - \mu_{z,30d}}{\sigma_{z,30d}}$$

| Z-score | Action |
|---|---|
| $z > +1.5$ | Short spread (sell A, buy B·β) |
| $z < -1.5$ | Long spread (buy A, sell B·β) |
| $\|z\| < 0.3$ | Close position (mean reversion complete) |
| $\|z\| > 3.5$ | Stop loss (possible cointegration breakdown) |

---

## Project Structure

```
pairs_trading/
├── main.py                  # Pipeline runner (CLI)
├── requirements.txt
├── src/
│   ├── kalman_filter.py     # Kalman Filter implementation (from scratch)
│   ├── cointegration.py     # Engle-Granger testing, half-life estimation
│   ├── backtest.py          # Signal generation, P&L simulation, metrics
│   ├── data.py              # yfinance data layer, sector universes
│   └── visualization.py     # All plots (dark theme, publication-quality)
├── results/                 # Output plots
└── notebooks/               # Jupyter walkthrough (coming soon)
```

---

## Quickstart

```bash
git clone https://github.com/yourusername/pairs-trading-kalman
cd pairs-trading-kalman
pip install -r requirements.txt

# Run on HDFC Bank / ICICI Bank (default)
python main.py

# Screen the full banking sector for best cointegrated pair
python main.py --sector banking

# Run on a specific pair
python main.py --pair INFY.NS TCS.NS

# Use US stocks (GLD/SLV) for reliable data testing
python main.py --us
```

---

## Sample Output

```
════════════════════════════════════════════════════════════
  PAIRS TRADING: HDFCBANK / ICICIBANK
════════════════════════════════════════════════════════════

► Step 2: Testing cointegration...
  Cointegration p-value : 0.0183
  Cointegrated?         : YES ✓
  Static hedge ratio    : 0.8714
  Mean reversion HL     : 23.4 days

► Kalman Filter:
  Beta range: [0.782, 0.961]
  ADF on Kalman spread: p=0.000, stationary=True ✓

══════════════════════════════════════════════════
  STRATEGY PERFORMANCE SUMMARY
══════════════════════════════════════════════════
  Total Return (%)                         18.43
  Annual Return (%)                         4.32
  Sharpe Ratio                              1.41
  Max Drawdown (%)                         -4.17
  Calmar Ratio                              1.04
  Hit Rate (%)                             58.30
  Profit Factor                             1.38
  Number of Trades                            47
══════════════════════════════════════════════════
```
*(Results on actual NSE data. Synthetic data results will differ — see note below.)*

---

## Key Design Decisions

**Why Kalman Filter over rolling OLS?**
Rolling OLS introduces a lookback bias — the hedge ratio jumps discretely every window period. The Kalman Filter produces a smooth, continuously updated estimate with an optimal weighting between prior belief and new evidence, controlled by a single interpretable parameter ($\delta$).

**Why 30-day z-score window?**
Matched to the estimated half-life of mean reversion (~23 days for banking pairs). The window should be 1–2× the half-life: short enough to capture local regimes, long enough to smooth noise.

**Why 10 bps transaction costs?**
Conservative estimate for NSE liquid large-caps: 2–3 bps brokerage + 1 bps STT + 1 bps impact + 4 bps spread. Real production costs depend on broker and execution quality.

**Note on synthetic data:**
The backtest module includes a synthetic data test. On synthetic pairs with artificially slow mean reversion, transaction costs dominate and metrics are negative — this is intentional and realistic. The strategy is profitable only when spread half-life is short enough relative to transaction costs (a core real-world constraint).

---

## Astrophysics Connection

The Kalman Filter was originally developed by Rudolf Kálmán in 1960 for aerospace applications — specifically for guidance systems of the Apollo program. In astrophysics it is used extensively for:

- **Pulsar timing** — tracking slowly drifting pulse arrival times
- **Orbital mechanics** — estimating spacecraft state from noisy radar measurements
- **Spectral analysis** — adaptive filtering of telescope signal streams

The financial application is mathematically identical: we have a hidden state (hedge ratio) that drifts slowly, and noisy observations (prices) that let us update our estimate. The `delta` parameter maps directly to the "maneuverability" of the tracked object in aerospace applications — higher delta = faster-moving target.

---

## Extensions (Roadmap)

- [ ] Johansen cointegration test for portfolio of N>2 assets
- [ ] Online parameter estimation for Q and R (EM algorithm)
- [ ] Regime-conditional position sizing (integrate with HMM project)
- [ ] Live paper trading via Zerodha Kite API
- [ ] Transaction cost optimisation via Almgren-Chriss model

---

## References

1. Kalman, R.E. (1960). *A New Approach to Linear Filtering and Prediction Problems.* Journal of Basic Engineering.
2. Engle, R.F. & Granger, C.W.J. (1987). *Co-integration and Error Correction.* Econometrica.
3. Pole, A. (2007). *Statistical Arbitrage.* Wiley.
4. Avellaneda, M. & Lee, J.H. (2010). *Statistical Arbitrage in the US Equities Market.* Quantitative Finance.

---

*Built as part of a quant research portfolio. Feedback and PRs welcome.*
