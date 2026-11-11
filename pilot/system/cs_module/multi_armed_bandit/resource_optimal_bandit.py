import numpy as np


class ResourceOptimalBandit:
    def __init__(self, resource_pref):
        # normalize keys to str, values to float
        self.resource_pref = {str(k): float(v) for k, v in resource_pref.items()}
        self.initial_parameters = self._initial_parameters()

    def _initial_parameters(self):
        return {"resource_pref": dict(self.resource_pref)}

    def select_action(self, actions, **kwargs):
        if not actions:
            raise ValueError("actions must be non-empty")

        actions_s = [str(a) for a in actions]

        # default 0.0 if missing to avoid crash
        expected_rewards = [
            1.0 / (1.0 + np.exp(-self.resource_pref.get(a, 0.0))) for a in actions_s
        ]

        best_idx = int(np.argmax(expected_rewards))
        best_action = actions_s[best_idx]
        sampled = {"estimated_reward": float(expected_rewards[best_idx]), "action": best_action}
        return best_action, sampled

    def update(self, action, *args, **kwargs):
        return {"action": str(action)}