import numpy as np
from scipy.stats import invgamma
import logging


# https://gregorygundersen.com/blog/2020/02/04/bayesian-linear-regression/


class GlobalLinearGaussianInverseGammaTS:
    def __init__(self, feature_dim, lambda_0=1.0, alpha_0=2.0, beta_0=1.0):
        self.feature_dim = feature_dim  # Dimension of feature_vector + action vectors

        # Initialize prior parameters
        # THETA
        self.mu = np.full(feature_dim, 0)  # expected rating is zero
        self.P = lambda_0 * np.eye(feature_dim)  # Precision matrix (prior precision)

        # SIGMA
        self.alpha = alpha_0  # Shape parameter for Gamma prior on precision
        self.beta = beta_0  # Scale parameter for Gamma prior on precision

        self.initial_parameters = self._initial_parameters()

    def _initial_parameters(self):
        return {
            "mu": self.mu.copy(),
            "P": self.P.copy(),
            "alpha": self.alpha,
            "beta": self.beta,
        }

    def select_action(self, actions, feature_vectors):
        feature_vectors = np.array(feature_vectors)

        sigma_sample = invgamma.rvs(self.alpha, scale=self.beta, size=1)
        theta_sample = np.random.multivariate_normal(self.mu, sigma_sample * np.linalg.inv(self.P))  # Sample theta

        rewards = feature_vectors @ theta_sample

        best_idx = np.argmax(rewards)
        sampled = {
            "sigma": sigma_sample,
            "theta": theta_sample,
            "estimated_reward": rewards[best_idx],
        }
        return actions[best_idx], sampled

    def update(self, feature_vector, reward):
        """Update posterior parameters based on observed reward"""
        feature_vector = np.asarray(feature_vector)

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

        params = {
            "mu": self.mu.copy(),
            "P": self.P.copy(),
            "alpha": self.alpha,
            "beta": self.beta,
        }
        return params
