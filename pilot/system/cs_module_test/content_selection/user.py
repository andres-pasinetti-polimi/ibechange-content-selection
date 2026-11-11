from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any, Optional
from datetime import datetime
from cs_module.utils.recommendation_history_tracker import RecommendationHistoryTracker
from cs_module.services.time_handler import TimeHandler
from cs_module.config import *
from cs_module.utils.min_max_norm import *


@dataclass
class User:
    user_id: str
    time_handler: TimeHandler
    intervention_start_date: Optional[datetime] = field(default=None)
    personal_data: Dict[str, Any] = field(default_factory=dict)
    health_habit_assessment: Dict[str, Any] = field(default_factory=dict)

    active: bool = True
    missions_started: bool = False

    new_plan_required: bool = field(default=False)

    selected_missions_and_contents: List[Dict] = field(default_factory=list)
    current_recommendation_plan: Dict[str, Any] = field(default_factory=dict)

    intervention_open_history: Dict[str, List[Any]] = field(default_factory=dict)
    recommendation_open_history: Dict[str, List[Any]] = field(default_factory=dict)
    global_open_history: Dict[str, List[Any]] = field(default_factory=dict)

    opened_rec_tracker: RecommendationHistoryTracker = field(default_factory=RecommendationHistoryTracker)

    disliked_rec_to_remaining_days: Dict[str, int] = field(default_factory=dict)
    past_week_rec_to_frequency: Dict[str, int] = field(default_factory=dict)

    received_resources: List[str] = field(default_factory=list)

    def add_received_resource(self, resource_id: str):
        """Add a resource to the list of received resources."""
        if resource_id not in self.received_resources:
            self.received_resources.append(resource_id)

    def get_received_resources(self) -> List[str]:
        """Get the list of received resources."""
        return self.received_resources

    def disable(self):
        self.active = False

    def update_escalation_level(self, level: int):
        self.personal_data["level"] = level

    def get_personal_data(self):
        return self.personal_data

    def update_health_habit_assessment(self, hhs: Dict[str, Any]):
        for key, value in hhs.items():
            if key == "components":
                # Ambiguous key without knowing the parent habit — skip or raise error
                continue

            self.health_habit_assessment[key] = value

            # If there's a "components" key, attach it as well
            if "components" in hhs:
                self.health_habit_assessment[f"{key}_components"] = hhs["components"]

    def get_hhs(self):
        return self.health_habit_assessment

    def get_num_intervention_days(self):
        """Get the number of days since the user started using the app."""
        if self.intervention_start_date:
            return int((self.time_handler.now - self.intervention_start_date).days)
        return None

    def set_missions_plan_to_false(self, missions: list):
        """Set the mission plan to false for a given mission ID."""
        for mission in self.selected_missions_and_contents:
            if mission["mission"] in missions:
                mission["plan_required"] = False
                break

    def update_missions_and_contents(self, missions_and_contents: Dict[str, Any]):
        for mission in missions_and_contents.get("new_missions", []):
            mission["plan_required"] = True
            self.selected_missions_and_contents.append(mission)

            if not self.missions_started:
                self.missions_started = True
                self.intervention_start_date = self.time_handler.string_to_iso(mission["selection_timestamp"])

        # self.mission_global_freq: List = []
        # self.intervention_to_mission_freq: Dict[str, int] = field(default_factory=dict)
        # self.recommendation_to_mission_freq: Dict[str, int] = field(default_factory=dict)

        self.new_plan_required = True

    def get_available_recommendations(self, mission_id: str) -> List[str]:
        """Get available contents for a given mission."""
        for mission in self.selected_missions_and_contents:
            if mission["mission"] == mission_id:
                return mission["recommendations"]
        return []

    def save_recommendation_plan(self, recommendation_plan: Dict[str, Any]):
        """Save the computed weekly recommendation plan to the dataclass instance."""
        self.current_recommendation_plan = recommendation_plan
        self.new_plan_required = False

    def update_rec_plan_to_position(self):
        """Update the recommendation plan to position mapping."""
        rec_plan_to_position_temp = {}
        self.rec_plan_to_position = {}

        contents = self.current_recommendation_plan.get("contents", [])  # Extract contents

        for content in contents:
            if content["type"] == "recommendation":  # Check if the content is a recommendation
                rec_plan_to_position_temp[content["id"]] = rec_plan_to_position_temp.get(content["id"], 0) + 1
                self.rec_plan_to_position[(content["send_timestamp"], content["id"])] = rec_plan_to_position_temp[
                    content["id"]
                ]

    def update_rec_plan_to_frequency(self):
        """Update the recommendation plan to frequency mapping."""
        self.rec_plan_to_frequency = {}

        contents = self.current_recommendation_plan.get("contents", [])  # Extract contents

        for content in contents:
            if content["type"] == "recommendation":  # Check if the content is a recommendation
                self.rec_plan_to_frequency[content["id"]] = self.rec_plan_to_frequency.get(content["id"], 0) + 1

    def get_sample_feedback_position(self, sample_feedback: Tuple[str, str]) -> int:
        """Get the position of the sample feedback in the recommendation plan."""
        return self.rec_plan_to_position[sample_feedback]

    def get_sample_feedback_frequency(self, sample_feedback: str) -> int:
        """Get the frequency of the sample feedback in the recommendation plan."""
        return self.rec_plan_to_frequency[sample_feedback]

    def update_recommendation_open_history(self, availability_feedback: Any):
        """Update the recommendation history with the given feedback."""
        for feedback in availability_feedback:
            if feedback["type"] == "recommendation":
                if feedback["content_id"] not in self.recommendation_open_history:
                    self.recommendation_open_history[feedback["content_id"]] = []
                self.recommendation_open_history[feedback["content_id"]].append(feedback["sent_timestamp"])

    def get_recommendation_sliding_frequency(self, content_id, timestamp, sliding_window=7) -> int:
        """Get the sliding frequency of the recommendation."""
        rec_history = self.recommendation_open_history.get(content_id, [])

        filtered_history = [
            rec_timestamp for rec_timestamp in rec_history if (timestamp - rec_timestamp).days <= sliding_window
        ]

        # Here we just return the length of the filtered history
        sliding_frequency = len(filtered_history)
        return sliding_frequency

    def track_opened_recommendations(self, open_timestamp, process_id, rec_id, intervention_type):
        self.opened_rec_tracker.add_recommendation(open_timestamp, process_id, rec_id, intervention_type)

    def get_total_frequency(self, time_window=None):
        """Get the global frequency of the user."""
        total_frequency = self.opened_rec_tracker.get_count(
            time_window=time_window, rec_id=None, intervention_type=None
        )
        return total_frequency

    def get_recommendation_frequency(self, rec_id, time_window=None):
        """Get the frequency of the recommendation."""
        frequency = self.opened_rec_tracker.get_count(time_window=time_window, rec_id=rec_id, intervention_type=None)
        return frequency

    def get_intervention_frequency(self, intervention_type, time_window=None):
        """Get the frequency of the intervention.

        If intervention_type is a list with more than 2 elements, return the average frequency across types.
        """
        if not intervention_type:
            return 0
        total = sum(
            self.opened_rec_tracker.get_count(time_window=time_window, rec_id=None, intervention_type=itype)
            for itype in intervention_type
        )
        return total / len(intervention_type)
