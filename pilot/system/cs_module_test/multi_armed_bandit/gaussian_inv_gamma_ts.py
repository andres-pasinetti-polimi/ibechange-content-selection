import numpy as np


class GaussianInverseGammaTS:
    def __init__(self, mu_0=0.0, kappa_0=1.0, alpha_0=2.0, beta_0=1.0):
        self.mu_0 = mu_0
        self.kappa_0 = kappa_0
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0
        self.initial_parameters = self._initial_parameters()

        # Store parameters dynamically as actions are encountered
        self.mu = {}
        self.kappa = {}
        self.alpha = {}
        self.beta = {}

    def _initial_parameters(self):
        """Store initial parameters for the first action."""
        return {
            "mu_0": self.mu_0,
            "kappa_0": self.kappa_0,
            "alpha_0": self.alpha_0,
            "beta_0": self.beta_0,
        }

    def _initialize_action(self, action):
        """Initialize parameters for a new action if not already present."""
        if action not in self.mu:
            self.mu[action] = self.mu_0
            self.kappa[action] = self.kappa_0
            self.alpha[action] = self.alpha_0
            self.beta[action] = self.beta_0

    def select_action(self, actions):
        """Sample from posterior and select the action with the highest sampled value"""

        sampled = {}

        for action in actions:
            self._initialize_action(action)  # Ensure action is registered

            # Sample variance from Inv-Gamma(α, β)
            sampled_variance = 1 / np.random.gamma(self.alpha[action], 1 / self.beta[action])

            # Sample reward from Normal(μ, sampled_variance / κ)
            sampled_reward = np.random.normal(self.mu[action], np.sqrt(sampled_variance / self.kappa[action]))
            sampled[action] = {
                "estimated_reward": sampled_reward,
                "variance": sampled_variance,
            }

        # Pick the best action by sampled reward
        best_action = max(sampled, key=lambda a: sampled[a]["estimated_reward"])
        return best_action, sampled

    def update(self, action, reward):
        """Update posterior parameters after observing a reward"""

        # Store previous values before updating
        mu_old = self.mu[action]
        kappa_old = self.kappa[action]

        # Step 1: Update kappa (confidence in reward estimate)
        self.kappa[action] += 1
        kappa_new = self.kappa[action]  # Store new kappa

        # Step 2: Update reward estimate using weighted average formula
        self.mu[action] = (kappa_old * mu_old + reward) / kappa_new

        # Step 3: Update α (shape parameter for variance)
        self.alpha[action] += 0.5

        # Step 4: Update β (scale parameter for variance)
        self.beta[action] += (0.5 * (reward - mu_old) ** 2) * (kappa_old / kappa_new)

        params = {
            "action": action,
            "mu": self.mu[action],
            "kappa": self.kappa[action],
            "alpha": self.alpha[action],
            "beta": self.beta[action],
        }
        return params
