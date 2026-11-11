import numpy as np

# ------------------------------------------------------------------------
# Logistic Thompson Sampling with Laplace approximation (context + item bias)
# ------------------------------------------------------------------------
# Model:
#   r_t ∈ {0,1}
#   r_t | x_t, rec_t, θ, b ~ Bernoulli(σ(x_t^T θ + b[rec_t]))
#   σ(z) = 1 / (1 + exp(-z))
#
# Priors (independent):
#   θ ~ N(0, σ_θ^2 I)
#   b_i ~ N(0, σ_b^2)    for each recommendation i
#
# Posterior approximations:
#   p(θ | D_t) ≈ N(μ_θ,t, P_θ,t^{-1})
#   p(b_i | D_t) ≈ N(μ_bi,t, P_bi,t^{-1})
#
# For θ we support:
#   • Diagonal Laplace (fast):   P_θ ≈ diag(Pd_θ)                 (include_full_hessian=False)
#   • Full Laplace (accurate):   P_θ is full precision matrix     (include_full_hessian=True)
#
# For each b_i we always use scalar (diagonal) curvature.
#
# Inner-step ingredients at (θ, b_i):
#   z         = x^T θ + b_i
#   σ         = 1 / (1 + exp(-z))
#   grad_lik_θ= (σ - r) x
#   grad_lik_b= (σ - r)
#   grad_prior_θ = P_θ,prior (θ - μ_θ,prior)
#   grad_prior_b = P_bi,prior (b_i - μ_bi,prior)
#   w         = σ (1 − σ)
#   H_θ(θ)    = P_θ,prior + w * (x x^T)      # diag or full
#   H_b(b_i)  = P_bi,prior + w              # scalar
#   θ   ← θ   − H_θ(θ)^{-1} grad_θ
#   b_i ← b_i − H_b(b_i)^{-1} grad_b
#
# After inner loop:
#   μ_θ   ← θ_final
#   μ_bi  ← b_i,final
#   P_θ   ← P_θ,prior  + w_final * (x x^T)
#   P_bi  ← P_bi,prior + w_final
# ------------------------------------------------------------------------


class LogisticLaplaceTS_ItemBias:
    def __init__(
        self,
        feature_dim,
        discount=1.0,
        theta_prior_var=0.25,  # prior variance σ²_θ
        bias_prior_var=0.08,  # prior variance σ²_b
        n_steps=1,  # number of Newton steps per sample
        include_full_hessian=True,  # False: diagonal, True: full matrix (for θ only)
        w_clip=(1e-8, 0.25),  # clip for w=σ(1−σ) to stabilize curvature
    ):
        self.d = feature_dim
        self.discount = float(discount)
        self.n_steps = int(n_steps)
        self.use_full = bool(include_full_hessian)
        self.w_clip = w_clip

        # Prior precisions
        #   λ_θ0 = 1 / σ²_θ
        #   λ_b0 = 1 / σ²_b
        self._lambda_theta0 = 1.0 / float(theta_prior_var)
        self._lambda_bias0 = 1.0 / float(bias_prior_var)

        # ---- Posterior mean for θ ----
        self.mu = np.zeros(self.d, dtype=float)

        # ---- Precision for θ (full or diagonal) ----
        if self.use_full:
            self.P = np.eye(self.d, dtype=float) * self._lambda_theta0  # full precision matrix
        else:
            self.Pd = np.ones(self.d, dtype=float) * self._lambda_theta0  # diagonal precision vector

        # ---- Per-item biases: μ_b[rec_id], P_b[rec_id] ----
        self.mu_b = {}  # rec_id -> scalar mean
        self.P_b = {}  # rec_id -> scalar precision

        self.initial_parameters = self._initial_parameters()

    # --------------------------------------------------------------------
    # Helpers
    # --------------------------------------------------------------------
    def _initial_parameters(self):
        if self.use_full:
            theta_state = {"mu": self.mu.copy(), "P": self.P.copy()}
        else:
            theta_state = {"mu": self.mu.copy(), "Pd": self.Pd.copy()}

        return {
            **theta_state,
            "lambda_theta0": self._lambda_theta0,
            "lambda_bias0": self._lambda_bias0,
        }

    @staticmethod
    def _sigmoid(z):
        # Logistic link σ(z) = 1 / (1 + e^{-z})
        # Clipped for numerical stability.
        z = np.clip(z, -15.0, 15.0)
        return 1.0 / (1.0 + np.exp(-z))

    def _ensure_item(self, rec_id):
        rec_id = str(rec_id)
        if rec_id not in self.mu_b:
            self.mu_b[rec_id] = 0.0
            self.P_b[rec_id] = self._lambda_bias0
        return rec_id

    # -------------------------------------------------------------
    # 1. Action selection step (Thompson Sampling)
    # -------------------------------------------------------------
    def select_action(self, actions, feature_vectors, mc_propensity_n: int = 0, log_candidates: bool = True):
        actions = [str(a) for a in actions]
        if len(set(actions)) != len(actions):
            raise ValueError("Duplicate rec_id in actions; propensity and TS selection would be ambiguous.")
        if len(actions) != len(feature_vectors):
            raise ValueError("actions and feature_vectors must have same length")

        X = np.array(feature_vectors, dtype=float)

        # Sample θ̃ ~ N(μ, P^{-1})
        # --- prepare sampling for theta ---
        if self.use_full:
            P = 0.5 * (self.P + self.P.T)
            try:
                L = np.linalg.cholesky(P)
            except np.linalg.LinAlgError:
                L = np.linalg.cholesky(P + 1e-9 * np.eye(self.d))

            def sample_theta():
                z = np.random.normal(size=self.d)
                y = np.linalg.solve(L, z)  # N(0, P^{-1})
                return self.mu + y
        else:

            def sample_theta():
                return self.mu + np.random.normal(size=self.d) / np.sqrt(self.Pd)

        # --- single TS draw for actual action ---
        theta_sample = sample_theta()

        best_idx = 0
        best_p = -1.0
        per_action = {}

        for idx, rec_id in enumerate(actions):
            rec_id = self._ensure_item(rec_id)

            # Sample item bias b̃_i ~ N(μ_bi, P_bi^{-1})
            mu_bi = self.mu_b[rec_id]
            P_bi = self.P_b[rec_id]
            b_sample = mu_bi + np.random.normal() / np.sqrt(P_bi)

            z_logit = float(X[idx] @ theta_sample + b_sample)
            p = self._sigmoid(z_logit)

            if p > best_p:
                best_p = p
                best_idx = idx

            per_action[rec_id] = {
                "sampled_logit": z_logit,
                "sampled_prob": float(p),
                "sampled_bias": float(b_sample),
            }

        chosen_rec = self._ensure_item(actions[best_idx])

        sampled = {
            "estimated_reward": float(best_p),
            "per_item": per_action,
            "meta": {
                "bandit": "LogisticLaplaceTS_ItemBias",
                "use_full": self.use_full,
                "n_actions": len(actions),
            },
        }

        if log_candidates:
            sampled["candidates"] = [{"rec_id": str(actions[i]), "fv": X[i].tolist()} for i in range(len(actions))]

        # --- MC propensity ---
        if mc_propensity_n and mc_propensity_n > 0:
            counts = {str(rid): 0 for rid in actions}

            for _ in range(int(mc_propensity_n)):
                # sample theta
                th = sample_theta()

                best_r = None
                best_val = -1e18
                for i, rid in enumerate(actions):
                    rid = str(rid)
                    self._ensure_item(rid)
                    b = self.mu_b[rid] + np.random.normal() / np.sqrt(self.P_b[rid])
                    val = float(X[i] @ th + b)  # compare logits
                    if val > best_val:
                        best_val = val
                        best_r = rid
                counts[best_r] += 1

            sampled["mc_propensity"] = {
                "n": int(mc_propensity_n),
                "p_chosen": counts[chosen_rec] / float(mc_propensity_n),
                "p_by_item": {rid: c / float(mc_propensity_n) for rid, c in counts.items()},
            }
            sampled["p_log"] = sampled["mc_propensity"]["p_chosen"]

        return chosen_rec, X[best_idx].tolist(), sampled

    # -------------------------------------------------------------
    # 2. Posterior update (multi-step Newton, θ + item bias)
    # -------------------------------------------------------------
    def update(self, rec_id, feature_vector, reward):
        """
        Update both:
          - θ (shared context weights)
          - b[rec_id] (bias of the rated recommendation)

        rec_id        : item identifier (string)
        feature_vector: context vector x (shape (d,))
        reward        : 0/1
        """
        rec_id = self._ensure_item(rec_id)

        x = np.asarray(feature_vector, dtype=float)
        r = float(reward)

        # (A) Forgetting BEFORE using current posterior as prior
        if 0 < self.discount < 1:
            if self.use_full:
                self.P *= self.discount
                # Floor the diagonal to λ_θ0 to keep SPD and avoid over-forgetting
                diag = np.diag(self.P)
                np.fill_diagonal(self.P, np.maximum(diag, self._lambda_theta0))
            else:
                self.Pd = np.maximum(self.Pd * self.discount, self._lambda_theta0)

            # Bias precision forgets too, but stays >= prior precision
            self.P_b[rec_id] = max(self.P_b[rec_id] * self.discount, self._lambda_bias0)

        # Snapshot PRIOR for this sample and initialize inner-loop iterates
        mu_prior = self.mu.copy()
        if self.use_full:
            P_prior = self.P.copy()  # (d x d)
        else:
            Pd_prior = self.Pd.copy()  # (d,)

        mu_b_prior = float(self.mu_b[rec_id])
        P_b_prior = float(self.P_b[rec_id])

        theta = mu_prior.copy()
        b = mu_b_prior
        x2 = x * x

        # ---------- Inner Newton-like iterations on (θ, b_rec) ----------
        for _ in range(self.n_steps):
            # Predicted logit and probability under current (θ, b)
            z = float(x @ theta + b)
            s = self._sigmoid(z)
            w = s * (1.0 - s)  # curvature for logistic
            if self.w_clip is not None:
                lo, hi = self.w_clip
                w = float(np.clip(w, lo, hi))

            # --- θ block ---
            if self.use_full:
                # grad_θ = (σ - r) x + P_prior (θ - μ_prior)
                grad_theta = (s - r) * x + P_prior @ (theta - mu_prior)

                # H_θ = P_prior + w * (x x^T)
                H_theta = P_prior + (w * np.outer(x, x))
                delta_theta = np.linalg.solve(H_theta, grad_theta)
                theta = theta - delta_theta
            else:
                grad_theta = (s - r) * x + Pd_prior * (theta - mu_prior)

                # Diagonal negative Hessian: H_diag(θ) = Pd_prior + w * x^2
                H_diag = Pd_prior + (x2) * w
                delta_theta = grad_theta / H_diag
                theta = theta - delta_theta

            # --- bias block (scalar) ---
            # grad_b = (σ - r) + P_b_prior (b - μ_b_prior)
            grad_b = (s - r) + P_b_prior * (b - mu_b_prior)
            H_b = P_b_prior + w
            delta_b = grad_b / H_b
            b = b - delta_b

        # ---- after inner loop: update posterior means ----
        self.mu = theta
        self.mu_b[rec_id] = b

        # curvature at final point
        z_final = float(x @ theta + b)
        s_final = self._sigmoid(z_final)
        w_final = s_final * (1.0 - s_final)
        if self.w_clip is not None:
            lo, hi = self.w_clip
            w_final = float(np.clip(w_final, lo, hi))

        # ---- update precisions using PRIOR + final curvature ----
        if self.use_full:
            # P_new = P_prior + w_final * (x x^T)
            self.P = P_prior + (w_final * np.outer(x, x))
        else:
            # Pd_new = Pd_prior + w_final * x^2
            self.Pd = Pd_prior + w_final * x2

        self.P_b[rec_id] = P_b_prior + w_final

        # Return a copy of current posterior params (diag or full)
        bias_state = {
            "rec_id": str(rec_id),
            "mu_b": float(self.mu_b[rec_id]),
            "P_b": float(self.P_b[rec_id]),
        }
        if self.use_full:
            return {
                "mu": self.mu.copy(),
                "P": self.P.copy(),
                "item_bias": bias_state,
            }
        else:
            return {
                "mu": self.mu.copy(),
                "Pd": self.Pd.copy(),
                "item_bias": bias_state,
            }
