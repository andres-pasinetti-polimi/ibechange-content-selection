import numpy as np


class HybridLinUCB:
    """
    A LinUCB agent with shared (global) and per-action (specific) linear models.
    Model: r = x_g^T * theta_g + x_s^T * theta_s(action) + noise
    Maintains design matrices and response vectors for global and per-action regressions.
    """

    def __init__(self, global_dim, specific_dim, alpha=1.0):
        # Dimensions
        self.global_dim = global_dim
        self.specific_dim = specific_dim
        self.alpha = alpha  # UCB exploration parameter

        # Global (shared) ridge regression parameters: A_g * theta_g = b_g
        self.A_g = np.eye(global_dim)
        self.b_g = np.zeros(global_dim)

        # Per-action ridge parameters: A_s[action] * theta_s = b_s[action]
        self.A_s = {}  # action -> design matrix (specific_dim x specific_dim)
        self.b_s = {}  # action -> response vector (specific_dim,)

    def _initialize_arm(self, action):
        self.A_s[action] = np.eye(self.specific_dim)
        self.b_s[action] = np.zeros(self.specific_dim)

    def select_action(self, actions, feature_vectors):
        """
        For each action, compute UCB: estimated reward + alpha * uncertainty
        Returns the action maximizing this criterion.

        actions: list of action ids
        feature_vectors: list of length-(global_dim + specific_dim) arrays
        """
        # Precompute global estimates
        A_g_inv = np.linalg.inv(self.A_g)
        theta_g = A_g_inv.dot(self.b_g)

        best_action = None
        best_ucb = -np.inf
        for action, fv in zip(actions, feature_vectors):
            if action not in self.A_s:
                self._initialize_arm(action)

            # Specific estimates
            A_s_inv = np.linalg.inv(self.A_s[action])
            theta_s = A_s_inv.dot(self.b_s[action])

            # Split features
            xg = np.asarray(fv[: self.global_dim])
            xs = np.asarray(fv[self.global_dim :])

            # Predicted mean
            mean = xg.dot(theta_g) + xs.dot(theta_s)
            # Uncertainty: variance from global + specific
            var_g = xg.dot(A_g_inv).dot(xg)
            var_s = xs.dot(A_s_inv).dot(xs)
            ucb = mean + self.alpha * np.sqrt(var_g + var_s)

            if ucb > best_ucb:
                best_ucb = ucb
                best_action = action

        return best_action

    def update(self, action, feature_vector, reward):
        """
        Update global and per-action ridge parameters with new observation.

        action: chosen action id
        feature_vector: array of length global_dim + specific_dim
        reward: observed scalar reward
        """
        fv = np.asarray(feature_vector)
        xg = fv[: self.global_dim]
        xs = fv[self.global_dim :]

        if action not in self.A_s:
            self._initialize_arm(action)

        # Update global design and response
        self.A_g += np.outer(xg, xg)
        self.b_g += reward * xg

        # Update per-action design and response
        self.A_s[action] += np.outer(xs, xs)
        self.b_s[action] += reward * xs

    def get_params(self):
        """Return current ridge parameters."""
        return {
            "A_global": self.A_g,
            "b_global": self.b_g,
            "A_specific": self.A_s,
            "b_specific": self.b_s,
            "alpha": self.alpha,
        }
