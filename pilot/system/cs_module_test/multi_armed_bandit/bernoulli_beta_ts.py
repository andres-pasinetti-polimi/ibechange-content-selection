# https://papers.nips.cc/paper_files/paper/2011/file/e53a0a2978c28872a4505bdb51db06dc-Paper.pdf

import numpy as np


class BernoulliBetaTS:
    def __init__(self, alpha_0=1.0, beta_0=1.0):
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0
        self.initial_parameters = self._initial_parameters()

        # Store parameters dynamically as actions are encountered
        self.alpha = {}
        self.beta = {}

    def _initial_parameters(self):
        """Store initial parameters for the first action."""
        return {
            "alpha_0": self.alpha_0,
            "beta_0": self.beta_0,
        }

    def _initialize_action(self, action):
        """Initialize parameters for a new action if not already present."""
        if action not in self.alpha:
            self.alpha[action] = self.alpha_0
            self.beta[action] = self.beta_0

    def select_action(self, actions):
        """Sample from posterior and select the action with the highest sampled value"""

        sampled = {}

        for action in actions:
            self._initialize_action(action)  # Ensure action is registered

            # Sample theta from Beta(α, β)
            sampled_theta = np.random.beta(self.alpha[action], self.beta[action])

            sampled[action] = {
                "sampled_theta": sampled_theta,
            }

        # Pick the best action by sampled reward
        best_action = max(sampled, key=lambda a: sampled[a]["sampled_theta"])
        return best_action, sampled

    def update(self, action, reward):
        self._initialize_action(action)  # Ensure the action is registered
        if reward == 1:
            self.alpha[action] += 1  # Increment successes
        else:
            self.beta[action] += 1

        params = {
            "action": action,
            "alpha": self.alpha[action],
            "beta": self.beta[action],
        }
        return params
