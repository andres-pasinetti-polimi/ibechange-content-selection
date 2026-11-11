# virtual_user/sim_user.py

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from cs_module.content_selection.user import User as CoreUser
from virtual_user.services.content_manager import ContentManager
from virtual_user.services.user_mission_manager import UserMissionManager
from virtual_user.services.feedback_manager import FeedbackManager
from virtual_user.config import PILLARS
from virtual_user.utils.deterministic_rng import rng_from_key


class SimUser(CoreUser):
    """
    Simulator-facing user.

    - Inherits all production behaviour from cs_module.User
    - Adds only what the simulator needs: missions per week, HHS simulation,
      weekly plans, etc.
    """

    def __init__(
        self,
        user_id: str,
        time_handler,
        personal_data: Dict[str, Any],
        num_weeks_per_user: int,
        content_manager: ContentManager,
        profile: Optional[Dict[str, Any]] = None,
    ):
        # Let the dataclass set up its fields with defaults.
        # We only pass explicit args we actually want to override.
        super().__init__(user_id=user_id, time_handler=time_handler, personal_data=personal_data or {})
        # __post_init__ is not called automatically because we override __init__
        self.__post_init__()  # sets self.log
        self.user_rng = rng_from_key("user", self.user_id)

        # ----- simulator-specific attributes -----
        self.profile: Dict[str, Any] = profile or dict(personal_data or {})
        self.num_weeks_per_user: int = num_weeks_per_user
        self.content_manager: ContentManager = content_manager
        self.user_mission_manager = UserMissionManager(
            user=self,
            missions=self.content_manager.missions,
        )
        # temporal fields for missions & questionnaires
        if self.intervention_start_date is None:
            self.intervention_start_date = self.time_handler.now

        self.new_mission_expiration: Optional[datetime] = None
        self.hhs_questionnaire_times: List[datetime] = [
            self.intervention_start_date + timedelta(weeks=4 * i) for i in range(0, 3)
        ]

        # weekly planning state
        self.weekly_recommendation_plan: Dict[str, Any] = {}
        self.current_missions: List[str] = []
        self.weekly_available_recommendations: Dict[str, List[str]] = {}
        self.weekly_available_resources: Dict[str, List[str]] = {}
        self.current_rated_contents: List[Dict[str, Any]] = []

        # HHS simulation events (keeps the OLD simulator API shape)
        self._sim_hhs_events: List[Dict[str, Any]] = []

        # feedback manager for event simulation
        self.feedback_manager: FeedbackManager = FeedbackManager(
            user=self,
            time_handler=self.time_handler,
            num_weeks_per_user=self.num_weeks_per_user,
        )

    # ------------------------------------------------------------------
    # Convenience properties (used by the simulator previously)
    # ------------------------------------------------------------------
    @property
    def intervention_day(self) -> int:
        if not self.intervention_start_date:
            return 0
        return (self.time_handler.now - self.intervention_start_date).days

    @property
    def intervention_week(self) -> int:
        return self.intervention_day // 7

    # ------------------------------------------------------------------
    # Resource storage – map simulator naming to production storage
    # ------------------------------------------------------------------
    @property
    def stored_resources(self) -> List[str]:
        # What simulator used to call stored_resources == received_resources in prod
        return self.get_received_resources()

    def add_stored_resource(self, resource_id: str):
        self.add_received_resource(resource_id)

    # ------------------------------------------------------------------
    # HHS simulation (keeps the *old* simulator return shape)
    # ------------------------------------------------------------------
    def simulate_health_habit_assessment(self):
        """
        Simulator-style HHS:

        - Every 4 weeks (based on intervention_start_date).
        - Returns a list of events:
          [
            { "hhs": {pillar: score}, "assessment_timestamp": <iso> },
            ...
          ]

        Internally, also keeps self.health_habit_assessment as a dict
        (production style) so the rest of cs_module logic keeps working.
        """
        if not self.intervention_start_date:
            return {}

        now = self.time_handler.now
        now_rounded = now.replace(second=0, microsecond=0)

        # trigger only at the scheduled questionnaire times (rounded)
        if not any(t.replace(second=0, microsecond=0) == now_rounded for t in self.hhs_questionnaire_times):
            return {}

        # first assessment: create baseline scores
        if not self._sim_hhs_events:
            self.health_habit_assessment = {}
            base_rng = rng_from_key("hhs_baseline", self.user_id)

            for pillar in PILLARS:
                score = int(base_rng.integers(0, 101))
                self.health_habit_assessment[pillar] = float(score)
                self._sim_hhs_events.append(
                    {
                        "hhs": {pillar: score},
                        "assessment_timestamp": now,
                    }
                )
        else:
            # subsequent assessments: drift each pillar slightly
            # drift branch
            assessment_idx = len(self._sim_hhs_events)  # 1 for 2nd assessment, 2 for 3rd, ...
            for evt in self._sim_hhs_events:
                pillar = next(iter(evt["hhs"].keys()))
                old = evt["hhs"][pillar]
                drift_rng = rng_from_key("hhs_drift", self.user_id, "k", str(assessment_idx), "pillar", pillar)
                delta = int(drift_rng.integers(-10, 11))
                new = max(0, min(100, old + delta))
                evt["hhs"][pillar] = new
                evt["assessment_timestamp"] = now
                self.health_habit_assessment[pillar] = float(new)

        return self._sim_hhs_events

    # ------------------------------------------------------------------
    # Mission selection (wraps production fields but keeps simulator API)
    # ------------------------------------------------------------------
    def select_new_missions(self):
        """
        Called by the simulator once per week for each active user.

        Returns the JSON shape that your /updates endpoint expects under
        "new_missions_and_contents", but internally uses the production
        selected_missions_and_contents field.
        """
        # respect weekly cadence and intervention horizon
        if self.intervention_week >= self.num_weeks_per_user:
            return None

        if self.new_mission_expiration is not None and self.time_handler.now < self.new_mission_expiration:
            # still within the current mission week → nothing new
            return None

        # delegate to mission manager (simulator logic)
        user_new_missions = self.user_mission_manager.select_new_missions()
        if not user_new_missions:
            return None

        # content availability given current stored/received resources
        available_recs, available_res = self.content_manager.get_available_recommendations_and_resources(
            user_new_missions,
            self.stored_resources,  # maps to received_resources
        )

        update_ts = self.time_handler.now.replace(microsecond=0)
        self.new_mission_expiration = update_ts + timedelta(days=7)

        self.current_missions = user_new_missions
        self.weekly_available_recommendations = available_recs
        self.weekly_available_resources = available_res

        new_missions = []
        for mission_id in user_new_missions:
            mission_dict = {
                "mission": mission_id,
                "recommendations": available_recs.get(mission_id, []),
                "resources": available_res.get(mission_id, []),
                "prescribed": False,
                "selection_timestamp": update_ts.isoformat(),
                "finish_timestamp": None,
            }

            # This updates selected_missions_and_contents and captures context
            self.update_missions_and_contents(mission_dict)
            new_missions.append(mission_dict)

        # this structure matches what your old simulator returned
        new_missions_and_contents = {
            "update_timestamp": update_ts.isoformat(),
            "new_missions": new_missions,
        }

        # keep it around in case you want to inspect it later
        self.selected_contents = {
            "mission_start_time": update_ts.isoformat(),
            "missions": new_missions,
        }

        return new_missions_and_contents

    # ------------------------------------------------------------------
    # Weekly recommendation plan helpers
    # ------------------------------------------------------------------
    def update_recommendation_plan(self, plan: Dict[str, Any]):
        """Sim-side: store the weekly plan coming from /recommendation_plans."""
        self.weekly_recommendation_plan = plan

    def get_contents_to_rate(self) -> List[Dict[str, Any]]:
        """
        Same logic as your old simulator:

        - Keep only the last occurrence of each content_id in the weekly plan
        - Exclude contents already rated in this week
        """
        plans = self.weekly_recommendation_plan.get("plans", [])
        last_contents = []

        for i, c1 in enumerate(plans):
            is_last = True
            for j in range(i + 1, len(plans)):
                c2 = plans[j]
                if c2["content_id"] == c1["content_id"]:
                    is_last = False
                    break
            if is_last:
                last_contents.append(c1)

        contents_to_rate = []
        for content in last_contents:
            if content not in self.current_rated_contents:
                contents_to_rate.append(content)

        return contents_to_rate

    # ------------------------------------------------------------------
    # Feedback simulation (wraps FeedbackManager)
    # ------------------------------------------------------------------
    def simulate_user_feedback(self):
        """
        Thin wrapper: simulator calls this; we delegate to FeedbackManager.
        """
        if not self.active:
            return []

        if not self.feedback_manager:
            return []

        return self.feedback_manager.simulate_feedback()

    def _compute_safe_selection_timestamp(self):
        # 1. Last scheduled content in current plan
        plans = self.weekly_recommendation_plan.get("plans", [])
        if plans:
            last_scheduled_ts = max(self.time_handler.parse_client_ts(p["scheduled_for"]) for p in plans)
        else:
            last_scheduled_ts = self.time_handler.now

        # 2. Mission expiration (may be None on week 0)
        exp = self.new_mission_expiration or self.time_handler.now

        # 3. Selection must occur after BOTH
        safe_ts = max(last_scheduled_ts, exp) + timedelta(seconds=1)
        return safe_ts
