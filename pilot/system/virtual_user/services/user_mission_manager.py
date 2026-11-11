from virtual_user.config import PILLARS
from cs_module.utils.get_pillar import get_pillar
from virtual_user import config
from virtual_user.utils.deterministic_rng import rng_from_key

MISSION_SELECTION_MODE = getattr(config, "MISSION_SELECTION_MODE", "random")


class UserMissionManager:
    def __init__(self, user, missions):
        self.user = user
        self.missions = missions  # All missions available

    def _rng_for_week(self, week: int, prev_mission_id: str | None):
        return rng_from_key(
            "mission_select",
            self.user.user_id,
            "week",
            str(week),
            "prev",
            str(prev_mission_id),
        )

    @staticmethod
    def _choice_rng(rng, items):
        return items[int(rng.integers(0, len(items)))]

    def select_new_missions(self, mode: str = MISSION_SELECTION_MODE):
        week = int(self.user.intervention_week)
        user_new_missions = []
        if self.user.current_missions:  # If the user has missions from the previous week
            for previous_mission_id in self.user.current_missions:
                rng = self._rng_for_week(week, previous_mission_id)
                if mode == "random":  # Select a random pillar then a random mission
                    new_pillar = self._choice_rng(rng, PILLARS)
                    possible_missions = [
                        mission_id for mission_id in self.missions if get_pillar(mission_id) == new_pillar
                    ]
                    user_new_missions.append(self._choice_rng(rng, possible_missions) if possible_missions else None)

                elif mode == "fixed":
                    # Keep previous pillar and mission
                    user_new_missions.append(previous_mission_id)

                elif mode == "user_keep_pillar":
                    keep = float(rng.uniform(0.0, 1.0)) < float(self.user.profile["mission_retain_probability"])
                    if keep:
                        user_new_missions.append(previous_mission_id)  # Retain the mission
                    else:  # change mission
                        new_missions = [
                            mission_id
                            for mission_id in self.missions
                            if get_pillar(mission_id) == get_pillar(previous_mission_id)
                            and mission_id != previous_mission_id
                        ]
                        if new_missions:
                            user_new_missions.append(
                                self._choice_rng(rng, new_missions)
                            )  # Choose a new mission from the same pillar
                        else:
                            user_new_missions.append(None)  # No missions left in the pillar

                elif mode == "user_specific":
                    retain_pillar = float(rng.uniform(0.0, 1.0)) < float(self.user.profile["pillar_retain_probability"])
                    if retain_pillar:
                        retain_mission = float(rng.uniform(0.0, 1.0)) < float(
                            self.user.profile["mission_retain_probability"]
                        )
                        if retain_mission:
                            user_new_missions.append(previous_mission_id)  # Retain the mission
                        else:
                            new_missions = [
                                mission_id
                                for mission_id in self.missions
                                if get_pillar(mission_id) == get_pillar(previous_mission_id)
                                and mission_id != previous_mission_id
                            ]
                            if new_missions:
                                user_new_missions.append(
                                    self._choice_rng(rng, new_missions)
                                )  # Choose a new mission from the same pillar
                            else:
                                user_new_missions.append(None)  # No missions left in the pillar
                    else:  # change pillar uniformly
                        other_pillars = [p for p in PILLARS if p != get_pillar(previous_mission_id)]
                        new_pillar = self._choice_rng(rng, other_pillars)
                        new_missions = [
                            mission_id for mission_id in self.missions if get_pillar(mission_id) == new_pillar
                        ]
                        if new_missions:
                            user_new_missions.append(
                                self._choice_rng(rng, new_missions)
                            )  # Choose a new mission from the same pillar
                        else:
                            user_new_missions.append(None)  # No missions left in the pillar

        else:
            rng = self._rng_for_week(week, None)
            keys = list(self.missions.keys())
            user_new_missions.append(keys[int(rng.integers(0, len(keys)))])

        return user_new_missions
