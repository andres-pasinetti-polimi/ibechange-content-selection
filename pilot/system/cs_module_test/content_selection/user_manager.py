# content_selection/user_manager.py
import logging
from cs_module.content_selection.user import User
from datetime import datetime

logger = logging.getLogger(__name__)


class UserManager:
    def __init__(self, time_handler):
        self.time_handler = time_handler
        self.users = {}
        self.active_user_ids = []

    def get_all_users(self):
        return self.users

    def get_user(self, user_id):
        return self.users.get(user_id)

    def add_users(self, new_users):
        if not new_users:
            return

        for user_id, data in new_users.items():
            self.users[user_id] = User(
                user_id=user_id,
                time_handler=self.time_handler,
                personal_data=data,
                intervention_start_date=data["enrolmentDate"],
            )
            self.active_user_ids.append(user_id)

    def get_active_user_ids(self):
        return self.active_user_ids

    def disable_users(self, disabled_users):
        for user_id in disabled_users:
            if user_id in self.active_user_ids:
                self.active_user_ids.remove(user_id)
                self.users[user_id].disable()

    def update_escalation_levels(self, escalation_levels):
        for user_id, level_list in escalation_levels.items():
            user = self.get_user(user_id)
            if not user:
                logging.warning(f"User {user_id} not found for escalation level update.")
                continue

            for level in level_list:
                user.update_escalation_level(level["level"])

    def update_health_habits(self, assessments):
        for user_id, hhs_list in assessments.items():
            user = self.get_user(user_id)
            if not user:
                logging.warning(f"User {user_id} not found for health habit update.")
                continue

            # HHS are in ascending order by timestamp, so we start updating from the oldest
            for hhs in hhs_list:
                user.update_health_habit_assessment(hhs["hhs"])

    def update_missions_and_contents(self, new_missions_and_contents):
        for user_id, missions_and_contents in new_missions_and_contents.items():
            user = self.get_user(user_id)
            if user:
                user.update_missions_and_contents(missions_and_contents)
            else:
                logging.warning(f"User {user_id} not found for mission update.")

    def save_recommendation_plans(self, recommendation_plans):
        for user_id, plan in recommendation_plans.items():
            user = self.get_user(user_id)
            if user:
                user.save_recommendation_plan(plan)
            else:
                logging.warning(f"User {user_id} not found for saving weekly plan.")
