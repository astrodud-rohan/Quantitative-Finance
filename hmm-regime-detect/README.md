# Market Regime Detection via Hidden Markov Models
### Unsupervised Learning of Latent Market States from Price Returns

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Domain](https://img.shields.io/badge/Domain-Quantitative%20Finance%20%7C%20Risk-orange)

---

## Overview

Markets don't behave the same way all the time. Bull markets, bear markets, and high-volatility sideways regimes each have distinct return distributions — but the regime itself is **never directly observable**. This project treats market regimes as hidden latent states and learns them unsupervised from return data using a **3-state Gaussian Hidden Markov Model**.

The same mathematical framework used in astrophysics for **pulsar timing analysis** (detecting spin-down state transitions in neutron stars) and **variable star classification** is applied here to financial return series. The math is identical — we have a hidden state evolving via a Markov chain, and noisy observations drawn from state-conditional distributions.

**Key outputs:**
- Detected market regimes with posterior probabilities at every timestep
- Regime-conditional risk metrics (VaR, CVaR per regime)
- Demonstrated Sharpe improvement and 3× reduction in max drawdown from regime-aware strategy vs regime-blind

---

## Mathematical Framework

### Hidden Markov Model

Let $S_t \in \{0, 1, 2\}$ denote the hidden regime (Bear / Sideways / Bull) and $r_t$ denote the observed daily log return.

**Transition model** — regimes are persistent (Markov property):
$$P(S_t = j \mid S_{t-1} = i) = A_{ij}, \qquad \sum_j A_{ij} = 1$$

**Emission model** — Gaussian returns within each regime:
$$r_t \mid S_t = k \sim \mathcal{N}(\mu_k, \sigma_k^2)$$

### The Three Algorithms

**Baum-Welch (EM)** — learns $\theta = \{\pi, A, \mu, \sigma\}$ from unlabelled returns:

*E-step*: forward-backward algorithm computes
$$\gamma_t(k) = P(S_t = k \mid r_{1:T}, \theta), \qquad \xi_t(i,j) = P(S_t=i, S_{t+1}=j \mid r_{1:T}, \theta)$$

*M-step*: closed-form parameter updates
$$\hat{A}_{ij} = \frac{\sum_t \xi_t(i,j)}{\sum_t \gamma_t(i)}, \qquad \hat{\mu}_k = \frac{\sum_t \gamma_t(k) r_t}{\sum_t \gamma_t(k)}$$

**Viterbi** — finds the single most likely state sequence via log-domain dynamic programming:
$$\delta_t(j) = \max_i \left[\delta_{t-1}(i) + \log A_{ij}\right] + \log \mathcal{N}(r_t; \mu_j, \sigma_j)$$

**Forward algorithm** — evaluates model fit via total log-likelihood $\log P(r_{1:T} \mid \theta)$.

All three algorithms implemented **from scratch** in NumPy with log-domain scaling to prevent numerical underflow on long sequences.

---

## Results (10 years NIFTY 50 data)

### Fitted Regime Parameters

| Regime | Daily μ | Daily σ | Annual μ | Annual σ | Persistence | % Time |
|---|---|---|---|---|---|---|
| **Bear** | -0.13%  | 1.98% | -32.5% | 31.5% | 0.975 | ~18% |
| **Sideways** | +0.06% | 0.81% | +16.2% | 12.8% | 0.960 | ~62% |
| **Bull** | +0.08% | 1.20% | +18.4% | 19.0% | 0.970 | ~20% |

### Risk by Regime (VaR is not constant — this is the point)

| | Bear | Sideways | Bull |
|---|---|---|---|
| **95% VaR (1-day)** | -3.44% | -1.28% | -1.87% |
| **95% CVaR / ES** | -4.04% | -1.63% | -2.31% |
| **99% VaR (1-day)** | -4.39% | -1.86% | -2.54% |

Bear regime 95% VaR is **2.7× larger** than sideways regime. A single unconditional VaR (-1.83%) would massively underestimate risk during bear regimes.

### Strategy Comparison

| Strategy | Return | Sharpe | Max Drawdown |
|---|---|---|---|
| Buy & Hold | +158.9% | 0.256 | -32.6% |
| Momentum (Regime-Blind) | +73.5% | 0.037 | -32.1% |
| **Momentum (Regime-Aware)** | **+103.1%** | **0.127** | **-9.5%** |

Regime-aware strategy reduces max drawdown by **3.4×** by going flat during bear regimes.

---

## Project Structure

```
hmm_regime/
├── main.py                   # Full pipeline runner (CLI)
├── requirements.txt
├── README.md
├── src/
│   ├── hmm.py                # GaussianHMM — Baum-Welch + Viterbi from scratch
│   ├── regime_analysis.py    # Regime stats, VaR, strategy comparison
│   └── visualization.py     # 5 plots (dark theme)
└── results/                  # Output plots
    ├── regime_timeline.png
    ├── emission_distributions.png
    ├── regime_probabilities.png
    ├── strategy_comparison.png
    └── var_comparison.png
```

---

## Quickstart

```bash
git clone https://github.com/yourusername/hmm-regime-detection
cd hmm-regime-detection
pip install -r requirements.txt

# Run on NIFTY 50 (live data)
python main.py --ticker ^NSEI --years 10

# Run on S&P 500
python main.py --ticker ^GSPC

# 2-state model (bull/bear only)
python main.py --states 2

# Offline demo with synthetic data (known ground truth)
python main.py --synthetic
```

---

## Astrophysics Connection

HMMs were independently developed in the signal processing and physics communities for exactly this kind of problem:

**Pulsar timing** — a pulsar switches between "spin-up" and "spin-down" states (due to glitches or magnetospheric changes). The state is not directly observable; only the pulse arrival times are. Baum-Welch recovers the hidden state sequence from the noisy timing residuals.

**Variable star classification** — Cepheid variables, RR Lyrae, eclipsing binaries. Each stellar class has a characteristic light-curve "emission distribution." HMMs cluster them unsupervised.

The transition to finance is exact: replace "spin state" with "market regime" and "pulse arrival time" with "daily return." Same forward-backward recursion, same Viterbi backtracking, same EM parameter estimation.

---

## Key Design Decisions

**Why 3 states, not 2?**
A 2-state model conflates low-volatility bull markets with high-volatility recoveries. The 3-state model separates: trending up (bull), directionless high-noise (sideways), and trending down (bear). BIC model selection empirically favors 3 states on major equity indices.

**Why log-domain arithmetic?**
Forward/backward probabilities at $T=2520$ timesteps multiply hundreds of small numbers — direct computation causes underflow to machine zero. Log-domain implementation prevents this while remaining numerically exact.

**Why Viterbi and not MAP per timestep?**
Per-timestep MAP (argmax of $\gamma_t$) can produce regime sequences with single-day oscillations that don't respect the Markov structure. Viterbi finds the globally most probable *sequence*, producing stable, realistic regime transitions.

---

## Extensions (Roadmap)

- [ ] Multivariate HMM (joint regime across equity, rates, FX)
- [ ] Student-t emissions for fat-tailed regimes
- [ ] Online Baum-Welch for real-time regime updating
- [ ] Regime-conditional options pricing (Heston model per regime)
- [ ] Integration with Pairs Trading project: regime-conditional stat-arb activation

---

## References

1. Baum, L.E. et al. (1970). *A Maximisation Technique Occurring in the Statistical Analysis of Probabilistic Functions of Markov Chains.* Ann. Math. Stat.
2. Viterbi, A. (1967). *Error Bounds for Convolutional Codes.* IEEE Trans. Inf. Theory.
3. Hamilton, J.D. (1989). *A New Approach to the Economic Analysis of Nonstationary Time Series.* Econometrica.
4. Rabiner, L.R. (1989). *A Tutorial on Hidden Markov Models.* Proc. IEEE.

---

*Part of a quant research portfolio. See also: [Kalman Filter Pairs Trading](../pairs_trading_kalman).*
