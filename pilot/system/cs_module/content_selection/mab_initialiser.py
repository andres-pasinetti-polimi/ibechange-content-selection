# content_selection/mab_initialiser.py

import logging
import json
import os

from cs_module.multi_armed_bandit.logistic_laplace_ts import LogisticLaplaceTS
from cs_module.multi_armed_bandit.bernoulli_beta_ts import BernoulliBetaTS
from cs_module.multi_armed_bandit.random_bandit import RandomBandit
from cs_module.multi_armed_bandit.recommendation_optimal_bandit import RecommendationOptimalBandit
from cs_module.multi_armed_bandit.resource_optimal_bandit import ResourceOptimalBandit
from cs_module.multi_armed_bandit.logistic_laplace_ts_itembias import LogisticLaplaceTS_ItemBias

from cs_module.utils.encoding import get_dim_intervention_feature_vector

from cs_module.config import INTERVENTION_MAB_CONFIG, RECOMMENDATION_MAB_CONFIG, RESOURCE_MAB_CONFIG

logger = logging.getLogger(__name__)


class MABInitialiser:
    def __init__(self, data_storage, version: str = ""):
        """
        version: table name suffix, e.g. "" for old bandits (existing tables)
                 or "_new" for the new-rating-version bandits.
        """
        self.data_storage = data_storage
        self._version = version

        self.resource_mab = self._init_resource_mab()
        self.intervention_mab = self._init_intervention_mab()
        self.recommendation_mab = self._init_recommendation_mab()

    def _init_resource_mab(self):
        t = RESOURCE_MAB_CONFIG["type"]
        logger.info(f"Initializing resource MAB (version='{self._version}') with mode: {t}...")

        if t == "BernoulliBetaTS":
            kwargs = {}
            if "alpha_0" in RESOURCE_MAB_CONFIG:
                kwargs["alpha_0"] = RESOURCE_MAB_CONFIG["alpha_0"]
            if "beta_0" in RESOURCE_MAB_CONFIG:
                kwargs["beta_0"] = RESOURCE_MAB_CONFIG["beta_0"]
            mab = BernoulliBetaTS(**kwargs)

        elif t == "ResourceOptimalBandit":
            path = os.path.join(os.path.dirname(__file__), "..", "user_preferences", "res_preferences.json")
            with open(path) as f:
                resource_pref = json.load(f)
            mab = ResourceOptimalBandit(resource_pref=resource_pref)

        elif t == "RandomBandit":
            mab = RandomBandit()

        else:
            raise ValueError(f"Unknown resource_mab_mode: {t}")

        self.data_storage.initialize_bandit(
            table=f"resource_mab_runs{self._version}", bandit_type=t, initial_params=mab.initial_parameters
        )
        return mab

    def _init_intervention_mab(self):
        t = INTERVENTION_MAB_CONFIG["type"]
        logger.info(f"Initializing intervention MAB (version='{self._version}') with mode: {t}...")

        if t == "LogisticLaplaceTS":
            dim = get_dim_intervention_feature_vector(include_bias=True)

            # Start from config and strip the 'type' key
            cfg = dict(INTERVENTION_MAB_CONFIG)
            cfg.pop("type", None)

            # Only keep keys that LogisticLaplaceTS actually accepts
            allowed_keys = {"discount", "theta_prior_var", "n_steps", "include_full_hessian", "w_clip"}
            kwargs = {k: v for k, v in cfg.items() if k in allowed_keys}
            kwargs["feature_dim"] = dim

            mab = LogisticLaplaceTS(**kwargs)

        elif t == "LogisticLaplaceTS_ItemBias":
            dim = get_dim_intervention_feature_vector(include_bias=True)

            cfg = dict(INTERVENTION_MAB_CONFIG)
            cfg.pop("type", None)

            # LogisticLaplaceTS_ItemBias has the same core args, plus bias_prior_var
            allowed_keys = {
                "discount",
                "theta_prior_var",
                "bias_prior_var",
                "n_steps",
                "include_full_hessian",
                "w_clip",
            }
            kwargs = {k: v for k, v in cfg.items() if k in allowed_keys}
            kwargs["feature_dim"] = dim

            mab = LogisticLaplaceTS_ItemBias(**kwargs)

        elif t == "None":
            return None

        else:
            raise ValueError(f"Unknown intervention_mab_mode: {t}")

        self.data_storage.initialize_bandit(
            table=f"intervention_mab_runs{self._version}", bandit_type=t, initial_params=mab.initial_parameters
        )
        return mab

    def _init_recommendation_mab(self):
        t = RECOMMENDATION_MAB_CONFIG["type"]
        logger.info(f"Initializing recommendation MAB (version='{self._version}') with mode: {t}...")

        if t == "BernoulliBetaTS":
            kwargs = {}
            if "alpha_0" in RECOMMENDATION_MAB_CONFIG:
                kwargs["alpha_0"] = RECOMMENDATION_MAB_CONFIG["alpha_0"]
            if "beta_0" in RECOMMENDATION_MAB_CONFIG:
                kwargs["beta_0"] = RECOMMENDATION_MAB_CONFIG["beta_0"]
            mab = BernoulliBetaTS(**kwargs)

        elif t == "RecommendationOptimalBandit":
            int_path = os.path.join(os.path.dirname(__file__), "..", "user_preferences", "int_theta.json")
            with open(int_path) as f:
                intervention_pref = json.load(f)

            rec_path = os.path.join(os.path.dirname(__file__), "..", "user_preferences", "rec_preferences.json")
            with open(rec_path) as f:
                recommendation_pref = json.load(f)

            mab = RecommendationOptimalBandit(
                intervention_pref=intervention_pref, recommendation_pref=recommendation_pref
            )

        elif t == "RandomBandit":
            mab = RandomBandit()

        elif t == "None":
            return None

        else:
            raise ValueError(f"Unknown recommendation_mab_mode: {t}")

        self.data_storage.initialize_bandit(
            table=f"recommendation_mab_runs{self._version}", bandit_type=t, initial_params=mab.initial_parameters
        )
        return mab
