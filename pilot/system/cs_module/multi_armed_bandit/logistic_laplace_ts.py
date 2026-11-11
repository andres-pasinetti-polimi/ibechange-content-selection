import numpy as np

# ------------------------------------------------------------------------
# Logistic Thompson Sampling with Laplace approximation for Bayesian logistic regression
# ------------------------------------------------------------------------
# Supports:
#   • Diagonal Laplace (fast):   P ≈ diag(Pd)                 (include_full_hessian=False) (Chapelle & Li, NIPS 2011, Algorithm 3).
#   • Full Laplace (accurate):   P is full precision matrix   (include_full_hessian=True)
#
# Model:
#   r_t ∈ {0,1}
#   r_t | x_t, θ ~ Bernoulli(σ(x_t^T θ))
#   σ(z) = 1 / (1 + exp(-z))   (logistic / sigmoid function)
#
# Prior (isotropic)):
#   θ ~ N(0, σ_θ^2 I)
#
# Posterior approximation:
#   p(θ | D_t) ≈ N(μ_t, P_t^{-1})      with P_t diagonal here (Laplace, diagonal Hessian)
#
#
# Inner-step ingredients at θ:
#   grad_lik  = (σ - r) x
#   grad_prior= P_prior (θ - μ_prior)            # vector (diag) or matrix (full)
#   w         = σ(1-σ)                            # logistic curvature scalar
#   H(θ)      = P_prior + w * (x x^T)            # diag or full
#   θ ← θ - H(θ)^{-1} grad
# After inner loop:
#   μ ← θ_final
#   P ← P_prior + w_final * (x x^T)              # accumulate curvature ONCE
# ------------------------------------------------------------------------


class LogisticLaplaceTS:
    def __init__(
        self,
        feature_dim,
        discount=1,
        theta_prior_var=0.25,  # prior variance σ²_θ (e.g. 0.25 ⇒ σ ≈ 0.5)
        n_steps=1,  # number of Newton steps per sample
        include_full_hessian=False,  # False: diagonal, True: full matrix
        w_clip=(1e-8, 0.25),          # clip for w=σ(1−σ) to stabilize curvature
    ):
        self.d = feature_dim
        self.discount = float(discount)  # Discount factor λ ∈ (0,1]; λ=1 disables forgetting
        self.n_steps = int(n_steps)
        self.use_full = bool(include_full_hessian)
        self.w_clip = w_clip

        # Prior precision λ₀ = 1 / σ²_θ
        self._lambda0 = 1.0 / float(theta_prior_var)

        # Posterior mean μ
        self.mu = np.zeros(self.d)

        if self.use_full:
            # Full precision matrix P
            self.P = np.eye(self.d, dtype=float) * self._lambda0
        else:
            # Diagonal precision vector Pd
            self.Pd = np.ones(self.d, dtype=float) * self._lambda0

        self.initial_parameters = self._initial_parameters()

    def _initial_parameters(self):
        if self.use_full:
            return {"mu": self.mu.copy(), "P": self.P.copy()}
        else:
            return {"mu": self.mu.copy(), "Pd": self.Pd.copy()}

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
        X = np.array(feature_vectors)

        if self.use_full:
            # Sample θ̃ ~ N(μ, P^{-1}) using Cholesky of precision:
            # If P = L L^T (L lower-tri), then y = solve(L, z), z~N(0,I) ⇒ y ~ N(0, P^{-1})
            L = np.linalg.cholesky(self.P)  # P = L @ L.T  (L lower-triangular)
            z = np.random.normal(size=self.d)  # z ~ N(0, I)
            y = np.linalg.solve(L, z)  # L y = z  ⇒ y = L^{-1} z, so y ~ N(0, P^{-1})
            theta_sample = self.mu + y  # shift by mean ⇒ θ ~ N(μ, P^{-1})
        else:
            theta_sample = self.mu + np.random.normal(size=self.d) / np.sqrt(self.Pd)

        # Compute predicted reward probabilities under θ̃
        probabilities = self._sigmoid(X @ theta_sample)

        # Select the action with highest sampled probability
        best_idx = int(np.argmax(probabilities))
        sampled = {
            "theta": theta_sample,
            "estimated_reward": float(probabilities[best_idx]),
        }
        return actions[best_idx], X[best_idx].tolist(), sampled

    # -------------------------------------------------------------
    # 2. Posterior update (multi-setp Newton, diagonal or full)
    # -------------------------------------------------------------
    def update(self, feature_vector, reward):
        x = np.asarray(feature_vector)
        r = float(reward)  # binary reward {0,1}

        # (A) Forgetting BEFORE using current posterior as prior
        if 0 < self.discount < 1:
            if self.use_full:
                self.P *= self.discount
                # Floor the diagonal to λ0 to keep SPD and avoid over-forgetting
                diag = np.diag(self.P)
                np.fill_diagonal(self.P, np.maximum(diag, self._lambda0))
            else:
                self.Pd = np.maximum(self.Pd * self.discount, self._lambda0)

        # Snapshot PRIOR for this sample and initialize inner-loop iterate
        mu_prior = self.mu.copy()
        if self.use_full:
            P_prior = self.P.copy()  # (d x d)
        else:
            Pd_prior = self.Pd.copy()  # (d,)

        # ---------- Inner Newton-like iterations on θ ----------
        theta = mu_prior.copy()
        x2 = x * x
        for _ in range(self.n_steps):
            # -------------------- Compute posterior update --------------------
            # Predicted logit and probability under current posterior mean μ:
            z = x @ theta
            s = self._sigmoid(z)
            w = s * (1.0 - s)  # curvature for logistic
            if self.w_clip is not None:
                lo, hi = self.w_clip
                w = float(np.clip(w, lo, hi))

            # Negative gradient of log-posterior (log-likelihood + log-prior)
            # grad_lik  = (σ - r) x
            # grad_prior= P'_prior * (θ - μ_prior)
            if self.use_full:
                grad = (s - r) * x + P_prior @ (theta - mu_prior)

                # H = P_prior + w * (x x^T)
                # Solve H Δ = grad  ⇒ θ ← θ − η Δ
                H = P_prior + (w * np.outer(x, x))
                delta = np.linalg.solve(H, grad)
                theta = theta - delta
            else:
                # θ = θ − P_d^{-1} * grad
                grad = (s - r) * x + Pd_prior * (theta - mu_prior)

                # Diagonal negative Hessian: H_diag(θ) = Pd_prior + w * x^2
                H_diag = Pd_prior + (x2) * w
                delta = grad / H_diag
                theta = theta - delta

        # ---- after inner loop: update posterior mean and precision ----
        self.mu = theta
        s_final = self._sigmoid(x @ theta)
        w_final = s_final * (1 - s_final)
        if self.w_clip is not None:
            lo, hi = self.w_clip
            w_final = float(np.clip(w_final, lo, hi))

        if self.use_full:
            # P_new = P_prior + w_final * (x x^T)
            self.P = P_prior + (w_final * np.outer(x, x))
        else:
            # Pd_new = Pd_prior + w_final * x^2
            self.Pd = Pd_prior + w_final * x2

        # Return a copy of current posterior params (diag or full)
        if self.use_full:
            return {"mu": self.mu.copy(), "P": self.P.copy()}
        else:
            return {"mu": self.mu.copy(), "Pd": self.Pd.copy()}
