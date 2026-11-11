import logging
from datetime import datetime, timedelta
from cs_module.config import REWARD_TYPE, RECOMMENDATION_MAB_CONFIG, INTERVENTION_MAB_CONFIG, RESOURCE_MAB_CONFIG
from cs_module.utils.encoding import get_intervention_feature_vector, get_recommendation_feature_vector
from cs_module.utils.feedback_handler import get_opened_recommendations, get_rated_recommendations, get_rated_resources
from cs_module.utils.get_pillar import get_pillar

logger = logging.getLogger(__name__)


class MABUpdater:
    def __init__(
        self,
        time_handler,
        resource_mab,
        intervention_mab,
        recommendation_mab,
        user_manager,
        recommendations,
        missions,
        resources,
        data_storage,
    ):
        self.time_handler = time_handler
        self.resource_mab = resource_mab
        self.intervention_mab = intervention_mab
        self.recommendation_mab = recommendation_mab
        self.user_manager = user_manager
        self.recommendations = recommendations
        self.missions = missions
        self.resources = resources
        self.data_storage = data_storage

    def update_all(self, feedback):
        logger.info("Updating MABs...")

        for user_id, user_feedback in feedback.items():
            if user_id not in self.user_manager.get_all_users():
                logger.warning(f"User {user_id} not found. Skipping.")
                continue

            logger.info(f"Updating MABs with feedback from {user_id}")
            for event in user_feedback["events"]:
                # If mission was prescribed --> do not update MABs
                # No need to check for user level
                mission_id = event["properties"].get("mission_id")

                # It is possible during testing that we didn't consider old selected missions
                if self.user_manager.get_user(user_id).selected_missions_and_contents:
                    user_selected_missions = self.user_manager.get_user(user_id).selected_missions_and_contents
                    matches = [m for m in user_selected_missions if m["mission"] == mission_id]
                    if matches:
                        # If same mission is selected multiple times, take the latest one (which may have been prescribed)
                        latest_match = max(matches, key=lambda m: datetime.fromisoformat(m["selection_timestamp"]))
                        is_prescribed = latest_match["prescribed"]
                        if not is_prescribed:
                            logger.info(f"Updating MABs with event: {event}")
                            self._process_opened_recommendation(user_id, event)
                            self._process_rated_recommendation(user_id, event)
                            self._process_rated_resource(user_id, event)

    def _process_opened_recommendation(self, user_id, event):
        if not get_opened_recommendations([event]):
            return

        rec_id = event["properties"]["content_id"]
        if rec_id not in self.recommendations:
            logger.warning(f"Unknown recommendation ID {rec_id}")
            return

        user = self.user_manager.get_user(user_id)
        timestamp = datetime.fromisoformat(event["timestamp"])
        intervention = self.recommendations[rec_id]["intervention_type"]
        user.track_opened_recommendations(timestamp, event["process_id"], rec_id, intervention)

    def _process_rated_recommendation(self, user_id, event):
        if not get_rated_recommendations([event]):
            return

        user = self.user_manager.get_user(user_id)
        personal_data = user.get_personal_data()
        hhs = user.get_hhs()

        rec_id = event["properties"]["content_id"]
        mission_id = event["properties"]["mission_id"]
        rating = event["properties"]["rating"]

        if rec_id not in self.recommendations or mission_id not in self.missions:
            logger.warning(f"Unknown recommendation or mission ID: {rec_id}, {mission_id}")
            return

        reward = self._compute_reward(rating)
        mission = self.missions[mission_id]
        intervention = self.recommendations[rec_id]["intervention_type"]

        event_time = datetime.fromisoformat(event["timestamp"])
        # Filter and parse selection timestamps for the target mission
        valid_selection_times = [
            datetime.fromisoformat(m["selection_timestamp"])
            for m in user.selected_missions_and_contents
            if m["mission"] == mission_id and datetime.fromisoformat(m["selection_timestamp"]) < event_time
        ]

        # Get the latest (max) one if any exist
        selection_time = max(valid_selection_times) if valid_selection_times else None

        time_window_past_week = (selection_time - timedelta(weeks=1), selection_time)
        time_window_planned = (selection_time, event_time - timedelta(seconds=1))

        feature_vector = get_intervention_feature_vector(
            personal_data,
            hhs,
            user.get_num_intervention_days(),
            get_pillar(mission_id),
            mission["weekly_frequency"],
            user.get_total_frequency(time_window_past_week),
            user.get_total_frequency(time_window_planned),
            intervention,
            user.get_intervention_frequency(intervention, time_window_past_week),
            user.get_intervention_frequency(intervention, time_window_planned),
            user.get_recommendation_frequency(rec_id, time_window_past_week),
            user.get_recommendation_frequency(rec_id, time_window_planned),
        )
        t = INTERVENTION_MAB_CONFIG["type"]

        if t == "HybridLinearGaussianInverseGammaTS":
            params = self.intervention_mab.update(rec_id, list(feature_vector) + [1], reward)
            logger.info(f"Updated intervention MAB for recommendation {rec_id}.")
        elif t in ["LogisticLaplaceTS", "GlobalLinearGaussianInverseGammaTS"]:
            params = self.intervention_mab.update(feature_vector, reward)
            logger.info(f"Updated intervention MAB for recommendation {rec_id}.")
        else:
            raise ValueError(f"Unknown intervention_mab_mode: {t}")

        timestamp = self.time_handler.now.isoformat()
        process_id = event["process_id"]
        update = {
            "user_id": user_id,
            "timestamp": timestamp,
            "process_id": process_id,
            "reward": reward,
            "params": params,
        }
        self.data_storage.add_mab_update(table="intervention_mab_updates", update=update)

        self._update_recommendation_mab(rec_id, user, reward, process_id)

    def _update_recommendation_mab(self, rec_id, user, reward, process_id):
        t = RECOMMENDATION_MAB_CONFIG["type"]
        if t in ["RandomBandit", "RecommendationOptimalBandit"]:
            params = self.recommendation_mab.update(rec_id)
        elif t in ["BernoulliBetaTS", "GaussianInverseGammaTS"]:
            params = self.recommendation_mab.update(rec_id, reward)
            logger.info(f"Updated recommendation MAB for {rec_id}.")
        else:
            raise ValueError(f"Unknown recommendation_mab_mode: {t}")

        timestamp = self.time_handler.now.isoformat()
        update = {
            "user_id": user.user_id,
            "timestamp": timestamp,
            "process_id": process_id,
            "reward": reward,
            "params": params,
        }
        self.data_storage.add_mab_update(table="recommendation_mab_updates", update=update)

    def _process_rated_resource(self, user_id, event):
        if not get_rated_resources([event]):
            return

        res_id = event["properties"]["content_id"]
        if res_id not in self.resources:
            logger.warning(f"Unknown resource ID {res_id}")
            return

        reward = self._compute_reward(event["properties"]["rating"])
        t = RESOURCE_MAB_CONFIG["type"]
        if t in ["RandomBandit", "ResourceOptimalBandit"]:
            params = self.resource_mab.update(reward)
        elif t in ["BernoulliBetaTS", "GaussianInverseGammaTS"]:
            params = self.resource_mab.update(res_id, reward)
        timestamp = self.time_handler.now.isoformat()
        process_id = event["process_id"]
        update = {
            "user_id": user_id,
            "timestamp": timestamp,
            "process_id": process_id,
            "reward": reward,
            "params": params,
        }
        self.data_storage.add_mab_update(table="resource_mab_updates", update=update)

        logger.info(f"Updated resource MAB for {res_id}.")

    def _compute_reward(self, rating):
        if REWARD_TYPE == "thumbs":
            return 1 if rating == "liked" else 0
        elif REWARD_TYPE == "float":
            return float(rating)
        raise ValueError(f"Unknown REWARD_TYPE: {REWARD_TYPE}")
