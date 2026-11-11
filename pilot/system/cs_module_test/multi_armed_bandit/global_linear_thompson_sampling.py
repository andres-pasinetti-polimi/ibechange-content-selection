import numpy as np


class GlobalLinearThompsonSampling:
    def __init__(self, dim_context, alpha=0.5, noise_variance=0.25):
        """
        Initializes the Global Linear Thompson Sampling model.

        Parameters:
        - dim_context (int): The number of features in the context vector.
        - alpha (float): Exploration-exploitation parameter, reduced for [0,1] rewards.
        - noise_variance (float): Variance of the reward noise, adjusted for [0,1] rewards.
        """
        self.dim_context = dim_context
        self.alpha = alpha
        self.noise_variance = noise_variance

        # Initialize posterior parameters
        self.mu = np.zeros(self.dim_context)  # Mean estimate
        self.B = np.eye(self.dim_context)  # Precision matrix (B = A^T A)
        self.f = np.zeros(self.dim_context)  # Accumulated reward vector

    def select_action(self, context_ids, contexts):
        """
        Selects an action using Thompson Sampling.

        Parameters:
        - context_ids (list): List of action IDs.
        - contexts (ndarray): Matrix where each row is the context vector for an action.

        Returns:
        - (int, float): The selected action ID and its sampled reward estimate.
        """
        inv_B = np.linalg.inv(self.B)
        theta_sample = np.random.multivariate_normal(self.mu, (self.alpha**2) * inv_B)

        # Compute expected reward for each action
        rewards = contexts @ theta_sample

        # Select action with the highest sampled reward
        best_idx = np.argmax(rewards)
        return context_ids[best_idx], rewards[best_idx]

    def update(self, context, reward):
        """
        Updates the model parameters using Bayesian linear regression.

        Parameters:
        - context (ndarray): The context vector of the selected action.
        - reward (float): The observed reward from the environment.
        """
        b_a = np.array(context)

        self.B += (1 / self.noise_variance) * np.outer(
            b_a, b_a
        )  # Update precision matrix
        self.f += (
            (1 / self.noise_variance) * b_a * reward
        )  # Update accumulated reward vector

        # Compute new posterior mean
        inv_B = np.linalg.inv(self.B)
        self.mu = inv_B @ self.f
