import numpy as np
from cs_module.config import REWARD_TYPE


class RecommendationOptimalBandit:
    def __init__(self, intervention_pref, recommendation_pref):
        self.intervention_pref = intervention_pref
        self.recommendation_pref = recommendation_pref
        self.initial_parameters = self._initial_parameters()

    def _initial_parameters(self):
        return {
            "intervention_pref": self.intervention_pref.copy(),
            "recommendation_pref": self.recommendation_pref.copy(),
        }

    def select_action(self, actions, feature_vectors):
        feature_vectors = np.array(feature_vectors)
        rec_ids = [rec_id for rec_ids in actions for rec_id in rec_ids]
        int_ratings = self.intervention_pref @ feature_vectors.T

        # Assume additive effects of recommendations
        int_rec_ratings = {
            rec_id: int_ratings[i] + self.recommendation_pref[rec_id]
            for i, rec_ids in enumerate(actions)
            for rec_id in rec_ids
        }

        expected_rewards = []
        if REWARD_TYPE == "thumbs":
            expected_rewards = [1 / (1 + np.exp(-rating)) for rating in int_rec_ratings.values()]
        elif REWARD_TYPE == "float":
            expected_rewards = list(int_rec_ratings.values())

        # Select action with the highest sampled reward
        best_idx = np.argmax(expected_rewards)
        best_rec = rec_ids[best_idx]
        sampled = {"estimated_reward": expected_rewards[best_idx], "action": best_rec}
        return best_rec, sampled

    def update(self, action):
        params = {"action": action}
        return params
