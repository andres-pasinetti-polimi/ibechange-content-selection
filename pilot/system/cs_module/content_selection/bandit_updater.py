from cs_module.config import (
    REWARD_TYPE,
    RECOMMENDATION_MAB_CONFIG,
    INTERVENTION_MAB_CONFIG,
    RESOURCE_MAB_CONFIG,
)

from cs_module.utils.logging_utils import get_logger, ctx

# Maps rating_version to the corresponding DB table name suffix
_TABLE_SUFFIX = {"old": "", "new": "_new"}


class BanditUpdater:
    def __init__(
        self,
        data_storage,
        time_handler,
        intervention_mabs: dict,
        recommendation_mabs: dict,
        resource_mabs: dict,
    ):
        """
        *_mabs: {"old": <mab instance>, "new": <mab instance>}
        """
        self.data_storage = data_storage
        self.time_handler = time_handler
        self.intervention_mabs = intervention_mabs
        self.recommendation_mabs = recommendation_mabs
        self.resource_mabs = resource_mabs
        self.log = get_logger(self)
        self.log.info("BU001 BanditUpdater initialised")
        self._buf_intervention_updates = {"old": [], "new": []}
        self._buf_rec_updates = {"old": [], "new": []}
        self._buf_res_updates = {"old": [], "new": []}
        self._buf_feature_audit = []

    # ---------- public API ----------
    def flush(self):
        for version, suffix in _TABLE_SUFFIX.items():
            if self._buf_intervention_updates[version]:
                self.data_storage.add_intervention_mab_updates(
                    self._buf_intervention_updates[version],
                    table=f"intervention_mab_updates{suffix}",
                )
                self._buf_intervention_updates[version].clear()

            if self._buf_rec_updates[version]:
                self.data_storage.add_mab_updates(
                    f"recommendation_mab_updates{suffix}",
                    self._buf_rec_updates[version],
                )
                self._buf_rec_updates[version].clear()

            if self._buf_res_updates[version]:
                self.data_storage.add_mab_updates(
                    f"resource_mab_updates{suffix}",
                    self._buf_res_updates[version],
                )
                self._buf_res_updates[version].clear()

        if self._buf_feature_audit:
            self.data_storage.add_feature_audit_many(self._buf_feature_audit)
            self._buf_feature_audit.clear()

    def update_after_recommendation_rating(
        self,
        user,
        user_id,
        rec_id: str,
        mission_id: str,
        feature_vector,
        reward,
        process_id,
        rating_ts,
    ):
        version = user.rating_version
        fv_used = self._update_intervention_bandit_if_applicable(
            rec_id=rec_id,
            feature_vector=feature_vector,
            reward=reward,
            user_id=user_id,
            process_id=process_id,
            rating_ts=rating_ts,
            version=version,
        )

        self._update_recommendation_bandit(
            rec_id=rec_id,
            reward=reward,
            user=user,
            process_id=process_id,
            rating_ts=rating_ts,
            version=version,
        )

        return fv_used is not None

    def update_after_resource_rating(
        self,
        user_id,
        res_id: str,
        reward,
        process_id,
        rating_ts,
        version: str = "old",
    ):
        bandit_type = RESOURCE_MAB_CONFIG["type"]
        mab = self.resource_mabs[version]

        if bandit_type == "BernoulliBetaTS":
            params = mab.update(res_id, reward)
        elif bandit_type in ["RandomBandit", "ResourceOptimalBandit"]:
            params = mab.update(reward)
        else:
            raise ValueError(f"Unknown resource_mab_mode: {bandit_type}")

        timestamp = rating_ts
        processed_at = self.time_handler.wall_now_iso()
        update = {
            "user_id": user_id,
            "timestamp": timestamp,
            "processed_at": processed_at,
            "process_id": process_id,
            "reward": reward,
            "params": params,
        }
        self._buf_res_updates[version].append(update)

        self.log.info("BU300 resource MAB updated (%s)", ctx(user_id=user_id, pid=process_id, rec_id=res_id))

    def compute_reward(self, rating):
        if REWARD_TYPE == "thumbs":
            return 1 if rating == "liked" else 0
        if REWARD_TYPE == "float":
            return float(rating)
        raise ValueError(f"Unknown REWARD_TYPE: {REWARD_TYPE}")

    # ---------- private helpers ----------

    def _update_intervention_bandit_if_applicable(
        self,
        rec_id: str,
        feature_vector,
        reward,
        user_id,
        process_id,
        rating_ts,
        version: str,
    ):
        bandit_type = INTERVENTION_MAB_CONFIG["type"]
        mab = self.intervention_mabs.get(version)

        if feature_vector is None or bandit_type == "None" or mab is None:
            self.log.debug(
                "BU110 skip intervention update (%s)",
                ctx(user_id=user_id, pid=process_id, rec_id=rec_id, extra=f"type={bandit_type} fv=None_or_disabled"),
            )
            return None

        if bandit_type == "LogisticLaplaceTS":
            params = mab.update(feature_vector, reward)
            self.log.info("BU111 intervention CMAB updated (%s)", ctx(user_id=user_id, pid=process_id, rec_id=rec_id))
        elif bandit_type == "LogisticLaplaceTS_ItemBias":
            params = mab.update(rec_id, feature_vector, reward)
        else:
            raise ValueError(f"Unknown intervention_mab_mode: {bandit_type}")

        params = params or {}
        if isinstance(params, dict):
            params["rec_id"] = str(rec_id)

        timestamp = rating_ts
        processed_at = self.time_handler.wall_now_iso()

        update = {
            "user_id": user_id,
            "timestamp": timestamp,
            "processed_at": processed_at,
            "process_id": process_id,
            "rec_id": rec_id,
            "feature_vector": feature_vector,
            "reward": reward,
            "params": params,
        }

        self._buf_intervention_updates[version].append(update)

        return feature_vector

    def _update_recommendation_bandit(self, rec_id, reward, user, process_id, rating_ts, version: str):
        bandit_type = RECOMMENDATION_MAB_CONFIG["type"]
        mab = self.recommendation_mabs.get(version)

        if bandit_type == "BernoulliBetaTS":
            params = mab.update(rec_id, reward)
        elif bandit_type in ["RandomBandit", "RecommendationOptimalBandit"]:
            params = mab.update(rec_id)
        elif bandit_type == "None":
            return
        else:
            raise ValueError(f"Unknown recommendation_mab_mode: {bandit_type}")

        timestamp = rating_ts
        processed_at = self.time_handler.wall_now_iso()
        update = {
            "user_id": user.user_id,
            "timestamp": timestamp,
            "processed_at": processed_at,
            "process_id": process_id,
            "reward": reward,
            "params": params,
        }

        self._buf_rec_updates[version].append(update)

        self.log.info("BU210 recommendation MAB updated (%s)", ctx(user_id=user.user_id, pid=process_id, rec_id=rec_id))
