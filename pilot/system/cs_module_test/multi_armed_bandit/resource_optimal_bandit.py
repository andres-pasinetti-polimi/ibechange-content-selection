import numpy as np


class ResourceOptimalBandit:
    def __init__(self, resource_pref):
        self.resource_pref = resource_pref
        self.initial_parameters = self._initial_parameters()

    def _initial_parameters(self):
        return {
            "resource_pref": self.resource_pref.copy(),
        }

    def select_action(self, actions):
        expected_rewards = [1 / (1 + np.exp(-self.resource_pref[action])) for action in actions]
        best_idx = np.argmax(expected_rewards)
        best_action = actions[best_idx]
        sampled = {"estimated_reward": expected_rewards[best_idx], "action": best_action}
        return best_action, sampled

    def update(self, action):
        params = {"action": action}
        return params
