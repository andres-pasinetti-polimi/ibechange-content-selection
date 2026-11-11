import random
import numpy as np


class RandomBandit:
    def __init__(self):
        self.initial_parameters = {}

    def select_action(self, actions):
        expected_rewards = [random.uniform(0, 1) for _ in actions]
        best_idx = np.argmax(expected_rewards)
        best_action = actions[best_idx]
        sampled = {
            "action": best_action,
            "estimated_reward": expected_rewards[best_idx],
        }
        return best_action, sampled

    def update(self, reward):
        params = {}
        return params
