from datetime import datetime, timedelta
from cs_module.config import MIN_NUM_REC_PER_MISSION, MAX_NUM_REC_PER_MISSION
from .user_manager import UserManager
from .feature_builders import get_mission_id_to_intervention_mapping
from .selector import select_recommendation, select_resource
from .frequency_updaters import update_frequency_offsets


class RecommendationEngine:
    def __init__(
        self,
        user_manager: UserManager,
        missions,
        resources,
        recommendations,
        resource_mab,
        intervention_mab,
        recommendation_mab,
        data_storage,
        time_handler,
    ):
        self.user_manager = user_manager
        self.missions = missions
        self.resources = resources
        self.recommendations = recommendations
        self.resource_mab = resource_mab
        self.intervention_mab = intervention_mab
        self.recommendation_mab = recommendation_mab
        self.data_storage = data_storage
        self.time_handler = time_handler
        self.selection_id = {}

    def get_selection_id(self, user_id):
        if user_id not in self.selection_id:
            self.selection_id[user_id] = {"plan_count": 0, "content_count": 0}
        return self.selection_id[user_id]

    def get_recommendations_to_send(self, user_id):
        user = self.user_manager.get_user(user_id)
        # ONE MISSION AT A TIME FOR PILOT STUDY
        user_missions = [m for m in user.selected_missions_and_contents if m.get("plan_required")]
        selection_timestamp = datetime.fromisoformat(user_missions[0]["selection_timestamp"])
        past_week_time_window = (selection_timestamp - timedelta(weeks=1), selection_timestamp)

        avail = {m["mission"]: user.get_available_recommendations(m["mission"]) for m in user_missions}
        selected_recs = []
        freq_off_total = 0
        intv_off = {}
        rec_off = {}
        mission_selected = {}

        # Assuming equal distribution of recommendations across missions
        slots = {m["mission"]: MAX_NUM_REC_PER_MISSION // len(user_missions) for m in user_missions}

        if user_id not in self.selection_id:
            self.selection_id[user_id] = {"plan_count": 0, "content_count": 0}

        for m_id, num in slots.items():
            count = 0
            for _ in range(num):
                feat_map = get_mission_id_to_intervention_mapping(
                    self.user_manager,
                    user_id,
                    self.missions,
                    self.recommendations,
                    avail,
                    freq_off_total,
                    intv_off,
                    rec_off,
                )
                sel = select_recommendation(
                    feat_map[m_id],
                    self.intervention_mab,
                    self.recommendation_mab,
                    rec_off,
                    past_week_time_window,
                    self.data_storage,
                    self.time_handler,
                    user,
                    select_anyway=count < MIN_NUM_REC_PER_MISSION,
                    selection_id=self.selection_id[user_id],
                )
                if sel is None:
                    continue
                selected_recs.append({"id": sel, "type": "recommendation", "mission_id": m_id})
                update_frequency_offsets(sel, m_id, mission_selected, intv_off, rec_off, avail, self.recommendations)
                freq_off_total += 1
                count += 1
                self.selection_id[user_id]["content_count"] += 1
                if not avail[m_id]:
                    break
        self.selection_id[user_id]["plan_count"] += 1  # Increment selection ID for the user
        return selected_recs

    def get_resources_to_send(self, user_id):
        if user_id not in self.selection_id:
            self.selection_id[user_id] = {"plan_count": 0, "content_count": 0}

        user = self.user_manager.get_user(user_id)
        self.selection_id[user_id]["content_count"] == 0
        selected_resources = select_resource(
            user, self.resource_mab, self.data_storage, self.time_handler, selection_id=self.selection_id[user_id]
        )
        self.selection_id[user_id]["content_count"] = len(selected_resources)
        return selected_resources
