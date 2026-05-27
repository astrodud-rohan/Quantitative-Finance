"""
Hidden Markov Model — From Scratch Implementation
==================================================
Implements the three core HMM algorithms:

  1. Forward-Backward  — computes state probabilities at each timestep
  2. Baum-Welch (EM)   — learns model parameters from observations
  3. Viterbi           — decodes most likely hidden state sequence

Emission model: Gaussian (univariate) per state
  r_t | S_t = k  ~  N(mu_k, sigma_k^2)

This is mathematically identical to HMMs used in:
  - Pulsar timing analysis (detecting spin-down state changes)
  - Spectral classification of variable stars
  - Radio signal state detection
...just applied to financial return series instead.
"""

import numpy as np
from scipy.stats import norm
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HMMParams:
    """
    Container for HMM parameters.

    Attributes
    ----------
    n_states : int       — number of hidden states (regimes)
    pi       : (K,)      — initial state distribution
    A        : (K, K)    — transition matrix  A[i,j] = P(S_t=j | S_{t-1}=i)
    mu       : (K,)      — emission means (expected return per regime)
    sigma    : (K,)      — emission std devs (volatility per regime)
    """
    n_states: int
    pi: np.ndarray
    A: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray

    def emission_prob(self, r: float) -> np.ndarray:
        """P(r_t = r | S_t = k) for all k. Returns shape (K,)."""
        probs = np.array([
            norm.pdf(r, loc=self.mu[k], scale=self.sigma[k])
            for k in range(self.n_states)
        ])
        # Clip to avoid numerical underflow
        return np.clip(probs, 1e-300, None)

    def emission_matrix(self, returns: np.ndarray) -> np.ndarray:
        """
        Compute full emission probability matrix.
        B[t, k] = P(r_t | S_t = k)
        Returns shape (T, K)
        """
        T = len(returns)
        B = np.zeros((T, self.n_states))
        for t in range(T):
            B[t] = self.emission_prob(returns[t])
        return B


class GaussianHMM:
    """
    Gaussian-emission Hidden Markov Model with Baum-Welch learning
    and Viterbi decoding.

    Parameters
    ----------
    n_states    : number of hidden states (2 or 3 recommended for markets)
    n_iter      : maximum EM iterations
    tol         : convergence tolerance on log-likelihood
    random_seed : for reproducible initialisation
    """

    def __init__(
        self,
        n_states: int = 3,
        n_iter: int = 200,
        tol: float = 1e-6,
        random_seed: int = 42,
    ):
        self.n_states = n_states
        self.n_iter = n_iter
        self.tol = tol
        self.random_seed = random_seed
        self.params: Optional[HMMParams] = None
        self.log_likelihoods: list[float] = []
        self.converged: bool = False

    # ── Initialisation ────────────────────────────────────────────────────

    def _initialise_params(self, returns: np.ndarray) -> HMMParams:
        """
        Initialise parameters using k-means-like quantile assignment.
        Better than random init: puts states near actual return clusters.
        """
        rng = np.random.default_rng(self.random_seed)
        K = self.n_states

        # Initial state distribution: uniform
        pi = np.ones(K) / K

        # Transition matrix: sticky (high self-transition = persistent regimes)
        # Off-diagonal small and equal
        A = np.full((K, K), 0.05 / (K - 1))
        np.fill_diagonal(A, 0.95)

        # Emission means: spread across return quantiles
        quantiles = np.linspace(5, 95, K)
        mu = np.percentile(returns, quantiles)

        # Sort so state 0 = lowest return (bear), state K-1 = highest (bull)
        mu = np.sort(mu)

        # Emission stds: estimate from return distribution
        global_std = returns.std()
        sigma = np.full(K, global_std * 0.8)
        # Bear state gets higher volatility
        sigma[0] = global_std * 1.4
        sigma[-1] = global_std * 0.6

        return HMMParams(n_states=K, pi=pi, A=A, mu=mu, sigma=sigma)

    # ── Forward Algorithm ─────────────────────────────────────────────────

    def _forward(self, returns: np.ndarray, B: np.ndarray, params: HMMParams):
        """
        Forward algorithm with log-scaling to prevent underflow.

        alpha[t, k] = P(r_1:t, S_t = k)

        Returns
        -------
        alpha      : (T, K) scaled forward variables
        log_scales : (T,) log scaling factors
        log_prob   : total log P(observations | model)
        """
        T = len(returns)
        K = params.n_states
        alpha = np.zeros((T, K))
        log_scales = np.zeros(T)

        # t = 0: initialise
        alpha[0] = params.pi * B[0]
        scale = alpha[0].sum()
        if scale < 1e-300:
            scale = 1e-300
        alpha[0] /= scale
        log_scales[0] = np.log(scale)

        # t = 1..T-1: recurse
        for t in range(1, T):
            # alpha[t, j] = sum_i alpha[t-1, i] * A[i, j] * B[t, j]
            alpha[t] = (alpha[t-1] @ params.A) * B[t]
            scale = alpha[t].sum()
            if scale < 1e-300:
                scale = 1e-300
            alpha[t] /= scale
            log_scales[t] = np.log(scale)

        log_prob = log_scales.sum()
        return alpha, log_scales, log_prob

    # ── Backward Algorithm ────────────────────────────────────────────────

    def _backward(self, returns: np.ndarray, B: np.ndarray, log_scales: np.ndarray, params: HMMParams):
        """
        Backward algorithm (scaled with same scales as forward pass).

        beta[t, k] = P(r_{t+1:T} | S_t = k)

        Returns
        -------
        beta : (T, K) scaled backward variables
        """
        T = len(returns)
        K = params.n_states
        beta = np.zeros((T, K))

        # t = T-1: initialise
        beta[T-1] = 1.0

        # t = T-2 .. 0: recurse backwards
        for t in range(T-2, -1, -1):
            # beta[t, i] = sum_j A[i, j] * B[t+1, j] * beta[t+1, j]
            beta[t] = params.A @ (B[t+1] * beta[t+1])
            # Scale with same factor used in forward pass
            scale = np.exp(log_scales[t+1])
            if scale < 1e-300:
                scale = 1e-300
            beta[t] /= scale

        return beta

    # ── E-Step ────────────────────────────────────────────────────────────

    def _e_step(self, returns: np.ndarray, B: np.ndarray, params: HMMParams):
        """
        E-step: compute gamma (state occupancy) and xi (transition counts).

        gamma[t, k] = P(S_t = k | r_1:T, theta)
        xi[t, i, j] = P(S_t=i, S_{t+1}=j | r_1:T, theta)

        Returns
        -------
        gamma    : (T, K)
        xi       : (T-1, K, K)
        log_prob : float
        """
        T = len(returns)
        K = params.n_states

        alpha, log_scales, log_prob = self._forward(returns, B, params)
        beta = self._backward(returns, B, log_scales, params)

        # gamma[t, k] proportional to alpha[t,k] * beta[t,k]
        gamma = alpha * beta
        gamma_sum = gamma.sum(axis=1, keepdims=True)
        gamma_sum = np.where(gamma_sum < 1e-300, 1e-300, gamma_sum)
        gamma /= gamma_sum

        # xi[t, i, j] = alpha[t,i] * A[i,j] * B[t+1,j] * beta[t+1,j]
        xi = np.zeros((T-1, K, K))
        for t in range(T-1):
            xi[t] = (
                alpha[t][:, None]
                * params.A
                * (B[t+1] * beta[t+1])[None, :]
            )
            xi_sum = xi[t].sum()
            if xi_sum > 1e-300:
                xi[t] /= xi_sum

        return gamma, xi, log_prob

    # ── M-Step ────────────────────────────────────────────────────────────

    def _m_step(self, returns: np.ndarray, gamma: np.ndarray, xi: np.ndarray) -> HMMParams:
        """
        M-step: re-estimate all parameters to maximise expected log-likelihood.
        """
        T = len(returns)
        K = self.n_states

        # Initial distribution
        pi = gamma[0] / gamma[0].sum()

        # Transition matrix: A[i,j] = sum_t xi[t,i,j] / sum_t gamma[t,i]
        A = xi.sum(axis=0)
        A_row_sums = A.sum(axis=1, keepdims=True)
        A_row_sums = np.where(A_row_sums < 1e-300, 1e-300, A_row_sums)
        A /= A_row_sums

        # Emission parameters: weighted mean and variance
        gamma_sum = gamma.sum(axis=0)  # (K,)
        gamma_sum = np.where(gamma_sum < 1e-300, 1e-300, gamma_sum)

        mu = (gamma * returns[:, None]).sum(axis=0) / gamma_sum

        sigma = np.sqrt(
            (gamma * (returns[:, None] - mu[None, :]) ** 2).sum(axis=0)
            / gamma_sum
        )
        sigma = np.clip(sigma, 1e-6, None)  # Prevent degenerate states

        return HMMParams(n_states=K, pi=pi, A=A, mu=mu, sigma=sigma)

    # ── Baum-Welch ────────────────────────────────────────────────────────

    def fit(self, returns: np.ndarray) -> "GaussianHMM":
        """
        Fit HMM to return series using Baum-Welch (EM) algorithm.

        Parameters
        ----------
        returns : 1D array of log returns (or simple returns)

        Returns
        -------
        self (fitted model)
        """
        params = self._initialise_params(returns)
        prev_log_prob = -np.inf
        self.log_likelihoods = []

        print(f"  Fitting {self.n_states}-state HMM via Baum-Welch...")
        print(f"  Initial mu: {np.round(params.mu * 100, 4)}%")
        print(f"  Initial sigma: {np.round(params.sigma * 100, 4)}%")

        for iteration in range(self.n_iter):
            # Compute emission matrix
            B = params.emission_matrix(returns)

            # E-step
            gamma, xi, log_prob = self._e_step(returns, B, params)

            # M-step
            params = self._m_step(returns, gamma, xi)

            self.log_likelihoods.append(log_prob)

            # Check convergence
            delta = log_prob - prev_log_prob
            if iteration % 20 == 0:
                print(f"  Iter {iteration:3d}: log-likelihood = {log_prob:.4f}  (Δ={delta:.6f})")

            if abs(delta) < self.tol and iteration > 5:
                print(f"  Converged at iteration {iteration}")
                self.converged = True
                break

            prev_log_prob = log_prob

        self.params = params
        self._sort_states_by_return()

        print(f"\n  Fitted parameters:")
        for k in range(self.n_states):
            label = ["Bear", "Sideways", "Bull"][k] if self.n_states == 3 else f"State {k}"
            print(f"    {label}: mu={params.mu[k]*100:.4f}%/day, "
                  f"sigma={params.sigma[k]*100:.4f}%/day, "
                  f"persistence={params.A[k,k]:.3f}")

        return self

    def _sort_states_by_return(self):
        """Sort states so state 0 = bear (lowest mu), state K-1 = bull (highest mu)."""
        order = np.argsort(self.params.mu)
        self.params.mu = self.params.mu[order]
        self.params.sigma = self.params.sigma[order]
        self.params.pi = self.params.pi[order]
        self.params.A = self.params.A[np.ix_(order, order)]

    # ── Viterbi ───────────────────────────────────────────────────────────

    def decode(self, returns: np.ndarray) -> np.ndarray:
        """
        Viterbi algorithm: find the most likely hidden state sequence.

        Uses log-domain to prevent underflow (same as Viterbi in
        communications / signal processing applications).

        Returns
        -------
        states : (T,) array of most likely state indices
        """
        if self.params is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        T = len(returns)
        K = self.n_states
        params = self.params
        B = params.emission_matrix(returns)

        # Log-domain Viterbi
        log_A = np.log(np.clip(params.A, 1e-300, None))
        log_B = np.log(np.clip(B, 1e-300, None))
        log_pi = np.log(np.clip(params.pi, 1e-300, None))

        # delta[t, k] = max log P(s_1:t, S_t=k, r_1:t)
        delta = np.zeros((T, K))
        psi = np.zeros((T, K), dtype=int)  # backpointer

        delta[0] = log_pi + log_B[0]

        for t in range(1, T):
            for j in range(K):
                candidates = delta[t-1] + log_A[:, j]
                psi[t, j] = np.argmax(candidates)
                delta[t, j] = candidates[psi[t, j]] + log_B[t, j]

        # Backtrack
        states = np.zeros(T, dtype=int)
        states[T-1] = np.argmax(delta[T-1])
        for t in range(T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]

        return states

    def predict_proba(self, returns: np.ndarray) -> np.ndarray:
        """
        Return posterior state probabilities at each timestep.
        Uses the gamma from the E-step (smoothed probabilities).

        Returns
        -------
        gamma : (T, K) — P(S_t = k | all observations)
        """
        if self.params is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        B = self.params.emission_matrix(returns)
        gamma, _, _ = self._e_step(returns, B, self.params)
        return gamma

    def score(self, returns: np.ndarray) -> float:
        """Log-likelihood of the data under the fitted model."""
        if self.params is None:
            raise RuntimeError("Model not fitted.")
        B = self.params.emission_matrix(returns)
        _, _, log_prob = self._forward(returns, B, self.params)
        return log_prob
