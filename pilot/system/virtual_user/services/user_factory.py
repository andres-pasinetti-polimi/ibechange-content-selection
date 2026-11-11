import random
import numpy as np
from copy import deepcopy
from virtual_user.sim_user import SimUser
from virtual_user.config import NUMERIC_FEATURES_MIN_MAX, PERSONAL_DATA_CATEGORICAL_FEATURES
from virtual_user import config
import os
import json
import uuid
from virtual_user.utils.deterministic_rng import rng_from_key, deterministic_uuid


class UserFactory:
    def __init__(
        self, time_handler, content_manager, num_weeks_per_user, world_seed: int, env_name: str, world_id: int
    ):
        self.time_handler = time_handler
        self.num_weeks_per_user = num_weeks_per_user
        self.preferences = self.load_user_preferences()
        self.config = self.default_config()

        self.content_manager = content_manager

        self.world_seed = int(world_seed)
        self.env_name = str(env_name)
        self.world_id = int(world_id)

    def load_user_preferences(self):
        folder = os.path.join(os.path.dirname(__file__), "..", "user_preferences")
        filenames = ["res_preferences.json", "int_theta.json", "rec_preferences.json"]
        loaded_files = []

        for name in filenames:
            path = os.path.join(folder, name)
            with open(path, "r") as file:
                loaded_files.append(json.load(file))

        return loaded_files

    def default_config(self):
        base_open = getattr(config, "OPEN_PROBABILITY", 1.0)
        base_rate = getattr(config, "RATE_PROBABILITY", 1.0)

        def randint(rng, lo, hi_inclusive):
            return int(rng.integers(lo, hi_inclusive + 1))

        def choice(rng, items):
            return items[int(rng.integers(0, len(items)))]

        def beta(rng, a, b):
            return float(rng.beta(a, b))

        def uniform(rng, lo, hi):
            return float(rng.uniform(lo, hi))

        def normal_int_clipped(rng, mu, sigma, lo, hi):
            return int(np.clip(rng.normal(mu, sigma), lo, hi))

        return {
            "userAge": lambda rng: randint(
                rng, NUMERIC_FEATURES_MIN_MAX["userAge"][0], NUMERIC_FEATURES_MIN_MAX["userAge"][1]
            ),
            "gender": lambda rng: choice(rng, PERSONAL_DATA_CATEGORICAL_FEATURES["gender"]),
            "recruitmentCenter": lambda rng: choice(rng, PERSONAL_DATA_CATEGORICAL_FEATURES["recruitmentCenter"]),
            "education": lambda rng: choice(rng, PERSONAL_DATA_CATEGORICAL_FEATURES["education"]),
            "open_probability": base_open,
            "rate_probability": base_rate,
            "pillar_retain_probability": lambda rng: beta(rng, 8.0, 2.0),
            "mission_retain_probability": lambda rng: beta(rng, 7.0, 3.0),
            "mission_achieve_probability": lambda rng: uniform(rng, 0.0, 1.0),
            "preferences": lambda rng: deepcopy(self.preferences),
            "height": lambda rng: normal_int_clipped(rng, 170, 10, 140, 200),
            "weight": lambda rng: normal_int_clipped(rng, 70, 15, 45, 150),
            # IMPORTANT: enrolmentDate should be deterministic for the cohort,
            # not "now" which could differ depending on execution.
            # We'll set it in sample_user_profile() using cohort_key.
            "enrolmentDate": lambda rng: None,
            "wearable": lambda rng: choice(rng, ["yes", "no"]),
            "voiceRecording": lambda rng: choice(rng, ["yes", "no"]),
            "occupation": lambda rng: choice(rng, ["employed", "unemployed", "student", "retired", "other"]),
            "digitalLiteracy": lambda rng: choice(rng, ["low", "moderate", "high"]),
            "level": 0,
        }

    def sample_user_profile(self, rng, cohort_key: str, user_index: int):
        profile = {}
        for key, sampler in self.config.items():
            profile[key] = sampler(rng) if callable(sampler) else sampler

        # deterministic enrolmentDate: use cohort_key (e.g., entrance time)
        profile["enrolmentDate"] = str(cohort_key)
        return profile

    def generate_users(self, count, cohort_key: str):
        """
        cohort_key should uniquely identify *this intake batch* within the world.
        Good default: the ENTRANCE_TIME ISO string (rounded).
        """
        users = {}
        for i in range(count):
            # user-specific RNG, independent of call order elsewhere
            rng = rng_from_key(
                "env",
                self.env_name,
                "world_id",
                str(self.world_id),
                "world_seed",
                str(self.world_seed),
                "cohort",
                str(cohort_key),
                "user_index",
                str(i),
            )

            user_id = deterministic_uuid(
                "env",
                self.env_name,
                "world_id",
                str(self.world_id),
                "world_seed",
                str(self.world_seed),
                "cohort",
                str(cohort_key),
                "user_index",
                str(i),
            )

            profile = self.sample_user_profile(rng=rng, cohort_key=cohort_key, user_index=i)

            user = SimUser(
                user_id=user_id,
                time_handler=self.time_handler,
                personal_data=profile,  # goes into cs_module.User.personal_data
                num_weeks_per_user=self.num_weeks_per_user,
                content_manager=self.content_manager,
                profile=profile,  # keep a copy if you still want it
            )
            users[user_id] = user
        return users
