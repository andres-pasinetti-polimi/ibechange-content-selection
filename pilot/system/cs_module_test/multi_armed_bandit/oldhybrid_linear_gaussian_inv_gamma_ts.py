import numpy as np
from scipy.stats import invgamma


class HybridLinearGaussianInverseGammaTS:
    """
    A Thompson Sampling agent with shared (global) and per-action (specific) linear Gaussian models.
    Rewards are modeled as:
        r = x_g^T theta_g + x_s^T theta_s(action) + noise
    where theta_g is shared across all actions and theta_s(action) is specific to each action.
    The noise precision has a common inverse-gamma prior (alpha, beta).
    """

    def __init__(self, global_dim, specific_dim, alpha_0=1.0, beta_0=1.0):
        # Dimensions
        self.global_dim = global_dim
        self.specific_dim = specific_dim
        self.feature_dim = global_dim + specific_dim

        # Global (shared) posterior parameters
        self.mu_g = np.zeros(global_dim)
        self.P_g = np.eye(global_dim)  # Precision

        # Common noise prior parameters
        self.alpha = alpha_0
        self.beta = beta_0

        # Storage for per-action specific parameters
        self.arms = {}

    def _initialize_arm(self, action):
        """Initialize specific posterior for a new action."""
        self.arms[action] = {
            # Specific (per-arm) posterior parameters
            "mu_s": np.zeros(self.specific_dim),
            "P_s": np.eye(self.specific_dim),
        }

    def select_action(self, actions, feature_vectors):
        """
        Select an action by sampling global and specific parameters and picking the action
        with highest sampled reward.

        Parameters:
        - actions: list of action identifiers
        - feature_vectors: list of length-(global_dim+specific_dim) arrays

        Returns:
        - best_action: action with highest sampled reward
        """
        # Sample shared noise variance
        sigma = invgamma.rvs(self.alpha, scale=self.beta)
        # Sample global theta
        theta_g = np.random.multivariate_normal(self.mu_g, sigma * np.linalg.inv(self.P_g))

        # Evaluate each action
        est_rewards = {}
        for action, fv in zip(actions, feature_vectors):
            if action not in self.arms:
                self._initialize_arm(action)

            mu_s = self.arms[action]["mu_s"]
            P_s = self.arms[action]["P_s"]
            # Sample specific theta
            theta_s = np.random.multivariate_normal(mu_s, sigma * np.linalg.inv(P_s))

            # Split features
            xg = np.asarray(fv[: self.global_dim])
            xs = np.asarray(fv[self.global_dim :])

            # Combined predicted reward
            est_rewards[action] = xg.dot(theta_g) + xs.dot(theta_s)

        # Choose best
        best_action = max(est_rewards, key=est_rewards.get)
        return best_action, est_rewards[best_action], theta_s

    def update(self, action, feature_vector, reward):
        """
        Update posterior parameters given observed (action, feature_vector, reward).

        Parameters:
        - action: identifier of the chosen action
        - feature_vector: array of length global_dim+specific_dim
        - reward: observed scalar reward
        """

        x = np.asarray(feature_vector)
        xg = x[: self.global_dim]
        xs = x[self.global_dim :]

        if action not in self.arms:
            self._initialize_arm(action)

        # Store old states
        mu_g_old = self.mu_g.copy()
        P_g_old = self.P_g.copy()
        mu_s_old = self.arms[action]["mu_s"].copy()
        P_s_old = self.arms[action]["P_s"].copy()

        # Compute residual using old means
        resid = reward - xg.dot(mu_g_old) - xs.dot(mu_s_old)

        # Update global posterior
        self.P_g = P_g_old + np.outer(xg, xg)
        self.mu_g = np.linalg.inv(self.P_g) @ (P_g_old @ mu_g_old + xg * (reward - xs.dot(mu_s_old)))

        # Update specific posterior for this action
        self.arms[action]["P_s"] = P_s_old + np.outer(xs, xs)
        self.arms[action]["mu_s"] = np.linalg.inv(self.arms[action]["P_s"]) @ (
            P_s_old @ mu_s_old + xs * (reward - xg.dot(mu_g_old))
        )

        # Update shared noise inverse-gamma hyperparameters
        self.alpha += 0.5

        # —— global conditional step ——
        resid_g = reward - xg.dot(mu_g_old) - xs.dot(mu_s_old)
        quad_g_old = mu_g_old.dot(P_g_old.dot(mu_g_old))
        # update P_g, mu_g exactly as you have…
        quad_g_new = self.mu_g.dot(self.P_g.dot(self.mu_g))
        self.beta += 0.5 * (resid_g**2 + quad_g_old - quad_g_new)

        # —— specific conditional step ——
        resid_s = reward - xg.dot(self.mu_g) - xs.dot(mu_s_old)
        quad_s_old = mu_s_old.dot(P_s_old.dot(mu_s_old))
        # update P_s, mu_s exactly as you have…
        quad_s_new = self.arms[action]["mu_s"].dot(self.arms[action]["P_s"].dot(self.arms[action]["mu_s"]))
        self.beta += 0.5 * (resid_s**2 + quad_s_old - quad_s_new)

    def get_params(self):
        """Return current posterior parameters."""
        return {"mu_global": self.mu_g, "P_global": self.P_g, "alpha": self.alpha, "beta": self.beta, "arms": self.arms}
