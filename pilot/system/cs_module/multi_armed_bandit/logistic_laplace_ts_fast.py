import numpy as np

# ------------------------------------------------------------------------
# Logistic Thompson Sampling with Laplace approximation for Bayesian logistic regression
# ------------------------------------------------------------------------
# Model:
#   r_t ∈ {0,1}
#   r_t | x_t, θ ~ Bernoulli(σ(x_t^T θ))
#   σ(z) = 1 / (1 + exp(-z))   (logistic / sigmoid function)
#
# Prior over parameters:
#   θ ~ N(0, σ_θ^2 I)
#
# The posterior p(θ | D_t) is approximated as a Gaussian N(μ, P^{-1})
# using a diagonal Laplace approximation (Chapelle & Li, NIPS 2011, Algorithm 3).
#
# At each round:
#   1. Sample θ̃ ~ N(μ, P^{-1})  (Thompson sampling)
#   2. Select action a_t maximizing σ(x_t^T θ̃)
#   3. Observe binary reward r_t ∈ {0,1}
#   4. Update (μ, P) using one Newton step on the log-posterior
# ------------------------------------------------------------------------


class LogisticLaplaceTS:
    def __init__(
        self,
        feature_dim,
        discount=1,
        theta_prior_var=0.25,  # prior variance σ²_θ ≈ 0.25 (σ≈0.5)
    ):
        self.feature_dim = feature_dim

        # Prior precision λ₀ = 1 / σ²_θ
        theta_prior_prec = 1.0 / float(theta_prior_var)

        # Posterior mean μ (initially prior mean = 0)
        self.mu = np.zeros(feature_dim)

        # Posterior precision diagonal (inverse of variance)
        # Using diagonal approximation: P ≈ diag(Pd)
        self.Pd = np.ones(feature_dim) * theta_prior_prec

        # Store prior precision for regularization / forgetting floor
        self._P0_diag = np.ones(feature_dim) * theta_prior_prec

        # Discount factor λ ∈ (0,1]; λ=1 disables forgetting
        self.discount = float(discount)

        self.initial_parameters = self._initial_parameters()

    def _initial_parameters(self):
        return {
            "mu": self.mu.copy(),
            "Pd": self.Pd.copy(),
        }

    @staticmethod
    def _sigmoid(z):
        # Logistic link σ(z) = 1 / (1 + e^{-z})
        # Clipped for numerical stability.
        z = np.clip(z, -15.0, 15.0)
        return 1.0 / (1.0 + np.exp(-z))

    # -------------------------------------------------------------
    # 1. Action selection step (Thompson Sampling)
    # -------------------------------------------------------------
    def select_action(self, actions, feature_vectors):
        feature_vectors = np.array(feature_vectors)

        # Sample a parameter vector θ̃ ~ N(μ, P^{-1})
        # Here we use a diagonal covariance approximation:
        theta_sample = self.mu + np.random.normal(size=self.feature_dim) / np.sqrt(self.Pd)

        # Compute predicted reward probabilities under θ̃
        logits = feature_vectors @ theta_sample
        probabilities = 1 / (1 + np.exp(-logits))

        # Select the action with highest sampled probability
        best_idx = np.argmax(probabilities)
        sampled = {
            "theta": theta_sample,
            "estimated_reward": probabilities[best_idx],
        }
        return actions[best_idx], feature_vectors[best_idx].tolist(), sampled

    # -------------------------------------------------------------
    # 2. Posterior update via Laplace / online Newton step
    # -------------------------------------------------------------
    def update(self, feature_vector, reward):
        x = np.asarray(feature_vector)
        r = float(reward)  # binary reward {0,1}

        # -------------------- Forgetting (discount) --------------------
        # Apply exponential decay to precision before incorporating new data.
        # Ensures old information fades over time if λ < 1.
        if 0 < self.discount < 1:
            self.Pd = self.Pd * self.discount
            self.Pd = np.maximum(self.Pd, self._P0_diag)

        # -------------------- Compute posterior update --------------------
        # Predicted logit and probability under current posterior mean μ:
        z = x @ self.mu
        s = self._sigmoid(z)

        # Gradient of Bernoulli log-likelihood:
        # ∇_θ log p(r|x,θ) = (r - σ(x^T θ)) x
        # Here we compute the negative gradient (for a Newton-like update):
        grad = (s - r) * x

        # Curvature (second derivative / Hessian diagonal term)
        # For Bernoulli-logistic model: w = σ(1−σ)
        w = s * (1.0 - s)

        # Update precision (Laplace approximation):
        # P_d ← P_d + w * x²
        # This corresponds to accumulating curvature information.
        self.Pd += (x**2) * w

        # Newton update on the mean (diagonal case):
        # μ ← μ − P_d^{-1} * grad
        self.mu -= grad / self.Pd

        params = {
            "mu": self.mu.copy(),
            "Pd": self.Pd.copy(),
        }
        return params
