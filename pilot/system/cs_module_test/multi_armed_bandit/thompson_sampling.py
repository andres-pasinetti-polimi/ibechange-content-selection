import numpy as np


class ThompsonSampling:
    def __init__(self, prior_mean=0.5, prior_variance=1.0, noise_variance=0.25):
        self.means = {}  # Estimated mean reward for each action
        self.precisions = {}  # Precision (1/variance) for each action
        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        self.noise_variance = noise_variance  # Assumed variance of rewards

    def select_action(self, available_actions):
        sampled_values = {}

        for action in available_actions:
            if action not in self.means:
                self.means[action] = self.prior_mean
                self.precisions[action] = (
                    1 / self.prior_variance
                )  # Precision is 1/variance

            variance = 1 / self.precisions[action]
            sampled_values[action] = np.random.normal(
                self.means[action], np.sqrt(variance)
            )

        # Select action with the highest sampled value
        return max(sampled_values, key=sampled_values.get)

    def update(self, action, reward):
        if action not in self.means:
            self.means[action] = self.prior_mean
            self.precisions[action] = 1 / self.prior_variance

        # Bayesian update for normal distribution
        prior_precision = self.precisions[action]
        likelihood_precision = 1 / self.noise_variance  # Precision of observed data

        new_precision = prior_precision + likelihood_precision
        new_mean = (
            self.means[action] * prior_precision + reward * likelihood_precision
        ) / new_precision

        self.means[action] = new_mean
        self.precisions[action] = new_precision  # Update precision (inverse variance)
