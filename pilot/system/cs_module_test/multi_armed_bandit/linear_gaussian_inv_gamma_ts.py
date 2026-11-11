import numpy as np
from scipy.stats import invgamma

# https://gregorygundersen.com/blog/2020/02/04/bayesian-linear-regression/


class LinearGaussianInverseGammaTS:
    def __init__(self, feature_dim, alpha_0=1.0, beta_0=1.0):
        self.feature_dim = feature_dim  # Dimension of feature_vector
        self.alpha = alpha_0  # Shape parameter for Gamma prior on precision
        self.beta = beta_0  # Scale parameter for Gamma prior on precision
        self.arms = {}  # Dictionary to store parameters for each action

    def _initialize_arm(self, action):
        """Initialize parameters for a new action."""
        self.arms[action] = {
            # THETA
            "mu": np.zeros(self.feature_dim),  # Mean parameter estimate
            "P": np.eye(self.feature_dim),  # Precision matrix (prior precision)
            # SIGMA
            "alpha": self.alpha,
            "beta": self.beta,
        }

    def select_action(self, actions, feature_vectors):
        expected_rewards = {}
        for action, feature_vector in zip(actions, feature_vectors):
            if action not in self.arms:
                self._initialize_arm(action)

            alpha = self.arms[action]["alpha"]
            beta = self.arms[action]["beta"]
            mu = self.arms[action]["mu"]
            P = self.arms[action]["P"]

            # Sample theta
            sigma_sample = invgamma.rvs(alpha, scale=beta, size=1)
            theta_sample = np.random.multivariate_normal(
                mu, sigma_sample * np.linalg.inv(P)
            )

            # Compute expected reward
            feature_vector = np.asarray(feature_vector)
            expected_rewards[action] = feature_vector.dot(theta_sample)

        # Return action with the highest sampled reward
        best_action = max(actions, key=lambda x: expected_rewards[x])
        return best_action

    def update(self, action, feature_vector, reward):
        """Update posterior parameters based on observed reward"""

        P = self.arms[action]["P"]
        P_new = P + np.outer(feature_vector, feature_vector)
        self.arms[action]["P"] = P_new

        mu = self.arms[action]["mu"]
        mu_new = np.linalg.inv(P_new) @ (P @ mu + feature_vector * reward)
        self.arms[action]["mu"] = mu_new

        alpha = self.arms[action]["alpha"]
        alpha_new = alpha + 1 / 2
        self.arms[action]["alpha"] = alpha_new

        beta = self.arms[action]["beta"]
        beta_new = beta + 1 / 2 * (
            reward**2 + mu.T @ P @ mu - mu_new.T @ P_new @ mu_new
        )
        self.arms[action]["beta"] = beta_new
