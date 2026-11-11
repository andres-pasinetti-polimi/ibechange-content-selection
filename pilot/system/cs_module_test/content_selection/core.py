import logging
from cs_module.content_selection.engine import RecommendationEngine
from cs_module.content_selection.mab_initialiser import MABInitializer
from cs_module.content_selection.user_manager import UserManager
from cs_module.content_selection.mab_updater import MABUpdater
from cs_module.utils.data_storage_old import DataStorage
from cs_module.utils.encoding import *
from cs_module.utils.feedback_handler import *
import os
import json
import numpy as np
from datetime import datetime, timedelta


class ContentSelection:
    def __init__(
        self,
        time_handler,
    ):
        logging.info("Initializing ContentSelection...")
        self.time_handler = time_handler
        self.missions = {}
        self.recommendations = {}
        self.resources = {}
        self.intervention_running = True
        self.user_manager = UserManager(self.time_handler)
        self.data_storage = DataStorage()
        mab_initializer = MABInitializer(self.data_storage)
        self.resource_mab = mab_initializer.resource_mab
        self.intervention_mab = mab_initializer.intervention_mab
        self.recommendation_mab = mab_initializer.recommendation_mab

        self.mab_updater = None
        self.recommendation_engine = None

        self.selection_id = 0  # Initialize selection_id

    def initialise_missions(self, missions):
        self.missions = {rec["mission_id"]: rec for rec in missions}

    def initialise_recommendations(self, recommendations):
        self.recommendations = {rec["rec_id"]: rec for rec in recommendations}

    def initialise_resources(self, resources):
        self.resources = {rec["rec_id"]: rec for rec in resources}

    def convert_ndarrays_to_lists(self, obj):
        """Recursively convert all ndarrays in a structure to lists."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self.convert_ndarrays_to_lists(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_ndarrays_to_lists(item) for item in obj]
        return obj

    def save_output(self, output, filename):
        path = f"outputs/{filename}.json"
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Convert any ndarray objects to lists
        output = self.convert_ndarrays_to_lists(output)

        try:
            with open(path, "w") as f:
                json.dump(output, f, indent=4)
            print(f"Output saved to {path}")
            logging.info(f"Output saved to {path}")
        except (IOError, OSError) as e:
            logging.error(f"Error saving output to {path}: {e}")

    # Hourly
    def update(self, updates, mode):
        logging.info("Processing updates with mode: %s", mode)
        if self.mab_updater is None:
            self.mab_updater = MABUpdater(
                time_handler=self.time_handler,
                resource_mab=self.resource_mab,
                intervention_mab=self.intervention_mab,
                recommendation_mab=self.recommendation_mab,
                user_manager=self.user_manager,
                recommendations=self.recommendations,
                missions=self.missions,
                resources=self.resources,
                data_storage=self.data_storage,
            )

        if "new_users" in updates:
            logging.info(f"New users: {updates['new_users']}")
            new_users = updates["new_users"]
            self.data_storage.add_users(new_users)
            self.user_manager.add_users(new_users)

        if "health_habit_assessments" in updates:
            logging.info(f"Health Habit Assessments: {updates['health_habit_assessments']}")
            health_habit_assessments = updates["health_habit_assessments"]
            self.data_storage.add_health_habit_assessments(health_habit_assessments)
            self.user_manager.update_health_habits(health_habit_assessments)

        if "escalation_level" in updates:
            logging.info(f"Escalation levels: {updates['escalation_level']}")
            escalation_levels = updates["escalation_level"]
            self.data_storage.add_escalation_levels(escalation_levels)
            self.user_manager.update_escalation_levels(escalation_levels)

        if mode == "intervention":
            if "user_feedback" in updates:
                logging.info(f"User feedback: {updates['user_feedback']}")
                user_feedback = updates["user_feedback"]
                for user_id in user_feedback.keys():
                    if user_id not in self.user_manager.get_all_users():
                        user_feedback.pop(user_id)
                        logging.warning(f"User {user_id} not found. Skipping feedback.")

                self.data_storage.add_user_feedback(user_feedback)
                self.mab_updater.update_all(user_feedback)

        if "disabled_users" in updates:
            logging.info(f"Disabled users: {updates['disabled_users']}")
            disabled_users = updates["disabled_users"]
            self.data_storage.add_disabled_users(disabled_users)
            self.user_manager.disable_users(disabled_users)

        if mode == "intervention":
            if "new_missions_and_contents" in updates:
                logging.info(f"New missions and contents: {updates['new_missions_and_contents']}")
                new_missions_and_contents = updates["new_missions_and_contents"]
                self.data_storage.add_new_missions_and_contents(new_missions_and_contents)
                self.user_manager.update_missions_and_contents(new_missions_and_contents)

    def get_selected_contents(self):
        if self.recommendation_engine is None:
            self.recommendation_engine = RecommendationEngine(
                self.user_manager,
                self.missions,
                self.resources,
                self.recommendations,
                self.resource_mab,
                self.intervention_mab,
                self.recommendation_mab,
                self.data_storage,
                self.time_handler,
            )

        selected_contents = {}
        mission_ids = {}
        for user_id, user in self.user_manager.get_all_users().items():
            if user.new_plan_required:
                missions = [m for m in user.selected_missions_and_contents if m.get("plan_required")]
                # PILOT STUDY WILL HAVE ONE MISSION AT A TIME
                if missions:
                    mission_ids[user_id] = [m["mission"] for m in missions]
                    mission = missions[0]
                else:
                    logging.warning(f"No new missions found for user {user_id}. Skipping content selection.")
                    continue
                logging.info(f"Selecting contents for user {user_id}...")
                selected_contents[user_id] = {}
                selected_resources = self.recommendation_engine.get_resources_to_send(user_id)
                selected_recommendations = self.recommendation_engine.get_recommendations_to_send(user_id)
                selected_contents[user_id]["contents"] = selected_resources + selected_recommendations

                start_time = datetime.fromisoformat(mission["selection_timestamp"])
                end_time = start_time + timedelta(days=7)

                selected_contents[user_id]["mission_start_time"] = mission["selection_timestamp"]
                selected_contents[user_id]["mission_end_time"] = end_time.isoformat()

                selection_id = self.recommendation_engine.get_selection_id(user_id)
                selected_contents[user_id]["plan_count"] = selection_id["plan_count"]
                user.new_plan_required = False
                user.set_missions_plan_to_false([mission["mission"]])

        if selected_contents:
            timestamp = self.time_handler.now.isoformat()
            selected_contents_and_timestamps = {
                "timestamp": timestamp,
                "selected_contents": selected_contents,
                "mission_ids": mission_ids,
            }
            self.data_storage.add_selected_contents(selected_contents_and_timestamps)

        return selected_contents

    def save_recommendation_plans(self, recommendation_plans):
        self.data_storage.add_recommendation_plans(recommendation_plans)
        self.user_manager.save_recommendation_plans(recommendation_plans)
        return True
