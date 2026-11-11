import numpy as np
from cs_module.config import REWARD_TYPE


class RecommendationOptimalBandit:
    def __init__(self, intervention_pref, recommendation_pref):
        self.intervention_pref = np.asarray(intervention_pref, dtype=float)
        # normalize keys to str, values to float
        self.recommendation_pref = {str(k): float(v) for k, v in recommendation_pref.items()}
        self.initial_parameters = self._initial_parameters()

    def _initial_parameters(self):
        return {
            "intervention_pref": self.intervention_pref.tolist(),
            "recommendation_pref": dict(self.recommendation_pref),
        }

    def select_action(self, actions, feature_vectors, **kwargs):
        X = np.asarray(feature_vectors, dtype=float)
        int_ratings = self.intervention_pref @ X.T  # shape: (n_action_groups,)

        rec_ids = []
        scores = []

        # Build rec_ids and scores in the *same order*
        for i, rec_list in enumerate(actions):
            base = float(int_ratings[i])
            for rec_id in rec_list:
                rid = str(rec_id)
                rec_ids.append(rid)
                scores.append(base + self.recommendation_pref.get(rid, 0.0))

        if not scores:
            raise ValueError("No candidate recommendations provided")

        if REWARD_TYPE == "thumbs":
            expected_rewards = [1.0 / (1.0 + np.exp(-s)) for s in scores]
        elif REWARD_TYPE == "float":
            expected_rewards = scores
        else:
            raise ValueError(f"Unknown REWARD_TYPE={REWARD_TYPE!r}")

        best_idx = int(np.argmax(expected_rewards))
        best_rec = rec_ids[best_idx]
        sampled = {"estimated_reward": float(expected_rewards[best_idx]), "action": best_rec}
        return best_rec, sampled

    def update(self, action, *args, **kwargs):
        return {"action": str(action)}