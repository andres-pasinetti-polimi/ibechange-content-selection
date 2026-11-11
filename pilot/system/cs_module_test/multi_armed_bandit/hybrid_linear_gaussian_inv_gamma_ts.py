import numpy as np
from scipy.stats import invgamma
import logging


# https://gregorygundersen.com/blog/2020/02/04/bayesian-linear-regression/


class HybridLinearGaussianInverseGammaTS:
    def __init__(self, global_dim, specific_dim, alpha_0=1.0, beta_0=1.0):
        self.feature_dim = feature_dim  # Dimension of feature_vector + action vectors

        self.timestamp = timestamp

        # Initialize prior parameters
        # THETA
        self.mu = np.full(feature_dim, 0)  # expected rating is zero
        self.P = np.eye(feature_dim)  # Precision matrix (prior precision)

        # SIGMA
        self.alpha = alpha_0  # Shape parameter for Gamma prior on precision
        self.beta = beta_0  # Scale parameter for Gamma prior on precision

        # Tracking history
        self.history = []
        self.theta_history = []

        # Save initial state
        self._log_state()

    def _log_state(self, reward=None):
        self.history.append(
            {
                "timestamp": self.timestamp,
                "mu": self.mu.copy(),
                "alpha": self.alpha,
                "beta": self.beta,
                "reward": reward,
            }
        )

    def select_action(self, actions, feature_vectors, timestamp=None):
        feature_vectors = np.array(feature_vectors)

        if timestamp is not None:
            self.timestamp = timestamp

        sigma_sample = invgamma.rvs(self.alpha, scale=self.beta, size=1)
        theta_sample = np.random.multivariate_normal(self.mu, sigma_sample * np.linalg.inv(self.P))  # Sample theta

        self.theta_history.append({"timestamp": self.timestamp, "theta_sample": theta_sample.copy()})

        # logging.info(f"feature_vectors: {feature_vectors}, theta_sample: {theta_sample}")
        # Compute expected reward for each action
        rewards = feature_vectors @ theta_sample

        # Select action with the highest sampled reward
        best_idx = np.argmax(rewards)
        return actions[best_idx], rewards[best_idx], theta_sample

    def update(self, feature_vector, reward, timestamp=None):
        """Update posterior parameters based on observed reward"""
        feature_vector = np.asarray(feature_vector)

        if timestamp is not None:
            self.timestamp = timestamp

        P_old = self.P
        P_new = P_old + np.outer(feature_vector, feature_vector)
        self.P = P_new
        # Update mean using old P
        mu_old = self.mu
        mu_new = np.linalg.inv(P_new) @ (P_old @ mu_old + feature_vector * reward)
        self.mu = mu_new

        # Update precision parameters correctly
        self.alpha += 0.5
        beta_old = self.beta
        self.beta = beta_old + 1 / 2 * (reward**2 + mu_old.T @ P_old @ mu_old - mu_new.T @ P_new @ mu_new)

        self._log_state(reward)

    def get_history(self):
        return {
            "bandit_type": "GlobalLinearGaussianInverseGammaTS",
            "feature_dim": self.feature_dim,
            "history": self.history,
            "theta_history": self.theta_history,
        }
