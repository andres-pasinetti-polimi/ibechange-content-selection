import numpy as np


class LinearTS:
    def __init__(self, dim_context, alpha=0.5, noise_variance=0.25):
        """
        Initialize Linear Thompson Sampling.

        :param dim_context: Dimension of the context vector
        :param alpha: Scaling factor for exploration
        :param noise_variance: Assumed variance of noise in rewards
        """
        self.dim_context = dim_context
        self.alpha = alpha  # Controls exploration-exploitation tradeoff
        self.noise_variance = noise_variance  # Observation noise variance
        self.arms = {}  # Dictionary to store parameters for each action

    def _initialize_arm(self, action):
        """Initialize parameters for a new action."""
        self.arms[action] = {
            "B": np.eye(self.dim_context),  # Precision matrix (d x d identity matrix)
            "f": np.zeros(self.dim_context),  # Weighted sum of rewards
            "mu": np.zeros(self.dim_context),  # Mean parameter estimate
        }

    def select_action(self, context_ids, contexts):
        """
        Select an action using Thompson Sampling.

        :param context_ids: List of action IDs
        :param contexts: List of corresponding context vectors
        :return: Selected action ID
        """
        theta_samples = {}
        expected_rewards = {}

        for action, context in zip(context_ids, contexts):
            if action not in self.arms:
                self._initialize_arm(action)

            context = np.asarray(context)  # Ensure input is a NumPy array

            # Compute posterior variance
            B_inv = np.linalg.inv(self.arms[action]["B"])

            # Sample theta from the posterior
            theta_sample = np.random.multivariate_normal(
                self.arms[action]["mu"], self.alpha**2 * B_inv
            )

            # Compute expected reward
            theta_samples[action] = theta_sample
            expected_rewards[action] = np.dot(context, theta_sample)

        # Select action with highest expected reward
        return max(expected_rewards, key=expected_rewards.get)

    def update(self, action, context, reward):
        """
        Update parameters for the given action based on observed reward.

        :param action: Selected action ID
        :param context: Context vector for the action
        :param reward: Observed reward
        """
        if action not in self.arms:
            self._initialize_arm(action)

        context = np.asarray(context)  # Convert to NumPy array

        if context.shape[0] != self.dim_context:
            raise ValueError(
                f"Shape mismatch: Expected context of size {self.dim_context}, got {context.shape}"
            )

        # Bayesian update step with noise variance
        self.arms[action]["B"] += (1 / self.noise_variance) * np.outer(context, context)
        self.arms[action]["f"] += (1 / self.noise_variance) * reward * context
        self.arms[action]["mu"] = (
            np.linalg.inv(self.arms[action]["B"]) @ self.arms[action]["f"]
        )

        # Debugging log
        # print(f"Updated parameters for action {action}: mu shape {self.arms[action]['mu'].shape}")
