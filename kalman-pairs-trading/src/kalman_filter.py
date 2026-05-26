"""
Kalman Filter for Dynamic Hedge Ratio Estimation
=================================================
Treats the hedge ratio (beta) as a hidden state that evolves over time.
Each price observation is a noisy measurement of that state.

State equation:    beta_t = beta_{t-1} + w_t,   w_t ~ N(0, Q)
Observation eq:    P_A,t = beta_t * P_B,t + alpha + v_t,  v_t ~ N(0, R)

Analogous to tracking a slowly-drifting parameter in a physical system —
same math used in spacecraft state estimation and pulsar timing.
"""

import numpy as np


class KalmanFilterHedge:
    """
    1D Kalman Filter for online estimation of a dynamic hedge ratio.

    Parameters
    ----------
    delta : float
        Controls how quickly beta is allowed to drift (process noise).
        Smaller delta = slower adaptation. Typical range: 1e-5 to 1e-3.
    R : float
        Observation noise variance. Usually set to variance of price residuals.
    """

    def __init__(self, delta: float = 1e-4, R: float = 1.0):
        self.delta = delta
        self.R = R

        # Process noise covariance: Q = delta / (1 - delta)
        self.Q = delta / (1.0 - delta)

        # Initial state: beta and alpha both start at 0
        self.beta = 0.0
        self.alpha = 0.0

        # Initial state covariance (high uncertainty at start)
        self.P = 1.0

        # Storage for diagnostics
        self.betas = []
        self.alphas = []
        self.Ps = []
        self.gains = []

    def update(self, price_a: float, price_b: float) -> tuple[float, float, float]:
        """
        Process one new observation and return updated estimates.

        Parameters
        ----------
        price_a : float  — price of asset A (the "dependent" asset)
        price_b : float  — price of asset B (the "independent" asset / regressor)

        Returns
        -------
        beta  : current hedge ratio estimate
        alpha : current intercept estimate
        spread: current spread = price_a - beta*price_b - alpha
        """

        # --- PREDICT STEP ---
        # State propagates as a random walk; covariance grows by Q
        P_pred = self.P + self.Q

        # --- UPDATE STEP ---
        # Innovation (prediction error)
        predicted_a = self.beta * price_b + self.alpha
        innovation = price_a - predicted_a

        # Innovation covariance
        S = price_b * P_pred * price_b + self.R

        # Kalman Gain — how much to trust the new observation
        K = P_pred * price_b / S

        # Update beta
        self.beta = self.beta + K * innovation

        # Update intercept (alpha) with a separate simpler tracker
        # We use an exponential moving average for alpha for stability
        self.alpha = 0.99 * self.alpha + 0.01 * (price_a - self.beta * price_b)

        # Update state covariance
        self.P = (1 - K * price_b) * P_pred

        # Recompute spread with updated beta
        spread = price_a - self.beta * price_b - self.alpha

        # Store history for diagnostics
        self.betas.append(self.beta)
        self.alphas.append(self.alpha)
        self.Ps.append(self.P)
        self.gains.append(K)

        return self.beta, self.alpha, spread

    def run(self, prices_a: np.ndarray, prices_b: np.ndarray) -> dict:
        """
        Run filter over full price series.

        Returns a dict with arrays: betas, alphas, spreads, gains, covariances
        """
        spreads = []

        for pa, pb in zip(prices_a, prices_b):
            beta, alpha, spread = self.update(pa, pb)
            spreads.append(spread)

        return {
            "betas": np.array(self.betas),
            "alphas": np.array(self.alphas),
            "spreads": np.array(spreads),
            "gains": np.array(self.gains),
            "covariances": np.array(self.Ps),
        }