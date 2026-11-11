# https://papers.nips.cc/paper_files/paper/2011/file/e53a0a2978c28872a4505bdb51db06dc-Paper.pdf

import numpy as np


class BernoulliBetaTS:
    def __init__(self, alpha_0=1.0, beta_0=1.0, seed: int | None = None):
        self.alpha_0 = float(alpha_0)
        self.beta_0 = float(beta_0)
        self.initial_parameters = self._initial_parameters()

        # Store parameters dynamically as actions are encountered
        self.alpha = {}
        self.beta = {}

        # Optional RNG for reproducibility
        self._rng = np.random.default_rng(seed) if seed is not None else None

    def _initial_parameters(self):
        """Store initial parameters for the first action."""
        return {
            "alpha_0": self.alpha_0,
            "beta_0": self.beta_0,
        }

    def _initialize_action(self, action):
        """Initialize parameters for a new action if not already present."""
        action = str(action)
        if action not in self.alpha:
            self.alpha[action] = self.alpha_0
            self.beta[action] = self.beta_0
        return action

    def _beta_sample(self, a: float, b: float) -> float:
        if self._rng is None:
            return float(np.random.beta(a, b))
        # numpy Generator has beta()
        return float(self._rng.beta(a, b))

    def select_action(self, actions, mc_propensity_n: int = 0, log_candidates: bool = True):
        """Sample from posterior and select the action with the highest sampled value"""

        actions = [str(a) for a in actions]
        if len(set(actions)) != len(actions):
            raise ValueError("Duplicate action in actions; propensity and TS selection would be ambiguous.")
        if not actions:
            raise ValueError("actions must be non-empty")

        # Ensure parameters exist
        actions = [self._initialize_action(a) for a in actions]

        per_item = {}
        best_action = actions[0]
        best_theta = -1.0

        for a in actions:
            # Sample theta from Beta(α, β)
            th = self._beta_sample(self.alpha[a], self.beta[a])
            per_item[a] = {
                "sampled_theta": th,
                "alpha": float(self.alpha[a]),
                "beta": float(self.beta[a]),
            }

            if th > best_theta:
                best_theta = th
                best_action = a

        sampled = {
            "estimated_reward": float(best_theta),
            "chosen_action": str(best_action),
            "per_item": per_item,
            "meta": {
                "bandit": "BernoulliBetaTS",
                "n_actions": len(actions),
            },
        }

        if log_candidates:
            sampled["candidates"] = [{"id": str(a)} for a in actions]

        # --- MC propensity: P(argmax TS sample == chosen_action) ---
        if mc_propensity_n and mc_propensity_n > 0:
            counts = {str(a): 0 for a in actions}
            for _ in range(int(mc_propensity_n)):
                best_a = None
                best_val = -1.0
                for a in actions:
                    th = self._beta_sample(self.alpha[a], self.beta[a])
                    if th > best_val:
                        best_val = th
                        best_a = a
                counts[str(best_a)] += 1

            p_by_item = {a: c / float(mc_propensity_n) for a, c in counts.items()}
            p_chosen = p_by_item[str(best_action)]

            sampled["mc_propensity"] = {
                "n": int(mc_propensity_n),
                "p_chosen": float(p_chosen),
                "p_by_item": p_by_item,
            }
            sampled["p_log"] = float(p_chosen)

        return best_action, sampled

    def update(self, action, reward):
        action = self._initialize_action(action)  # Ensure the action is registered
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
