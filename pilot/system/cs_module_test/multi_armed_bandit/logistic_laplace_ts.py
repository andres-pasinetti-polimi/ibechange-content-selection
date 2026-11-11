import numpy as np


# Logistic Thompson Sampling with Laplace approximation for Bayesian logistic regression.
# Based on Chapelle & Li (NIPS 2011), Algorithm 3.

# Reward is converted to binary (0 or 1).


class LogisticLaplaceTS:
    def __init__(self, feature_dim):
        self.feature_dim = feature_dim

        self.mu = np.zeros(feature_dim)  # Mean of the Gaussian prior
        self.P = np.eye(feature_dim)  # Precision matrix of the Gaussian prior

        self.initial_parameters = self._initial_parameters()

    def _initial_parameters(self):
        return {
            "mu": self.mu.copy(),
            "S": self.P.copy(),
        }

    def select_action(self, actions, feature_vectors):
        feature_vectors = np.array(feature_vectors)

        # Sample theta from the Gaussian prior
        theta_sample = np.random.multivariate_normal(self.mu, np.linalg.inv(self.P))

        # Compute the probabilities using the logistic function
        logits = feature_vectors @ theta_sample
        probabilities = 1 / (1 + np.exp(-logits))

        # Select the action with the highest probability
        best_idx = np.argmax(probabilities)
        sampled = {
            "theta": theta_sample,
            "estimated_reward": probabilities[best_idx],
        }
        return actions[best_idx], sampled

    def update(self, feature_vector, reward):
        x = np.asarray(feature_vector)
        sigmoid = 1 / (1 + np.exp(-x @ self.mu))

        # 2) aggiorno la precisione diagonale
        diag_old = np.diag(self.P).copy()  # s_i^(old)
        diag_new = diag_old + x**2 * sigmoid * (1 - sigmoid)
        self.P[np.diag_indices(self.feature_dim)] = diag_new

        # 3) gradiente (solo likelihood, perché gradiente di prior si annulla)
        grad = (sigmoid - reward) * x

        # 4) un passo di Newton coord-by-coord con la precisione aggiornata
        self.mu = self.mu - grad / diag_new

        params = {
            "mu": self.mu,
            "P": self.P.copy(),
        }
        return params
