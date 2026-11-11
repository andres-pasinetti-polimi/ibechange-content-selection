# content_selection/user_manager.py
import logging
import os
from datetime import datetime, timezone
from cs_module.content_selection.user import User
from cs_module.utils.logging_utils import get_logger, ctx

_APP_RELEASE_DATE_RAW = os.getenv("CS_APP_RELEASE_DATE", "")
try:
    _APP_RELEASE_DT = (
        datetime.fromisoformat(_APP_RELEASE_DATE_RAW.replace("Z", "+00:00"))
        if _APP_RELEASE_DATE_RAW
        else None
    )
except ValueError:
    _APP_RELEASE_DT = None
    logging.warning("CS_APP_RELEASE_DATE invalid, falling back to 'old' for all new users: %r", _APP_RELEASE_DATE_RAW)


class UserManager:
    def __init__(self, time_handler):
        self.time_handler = time_handler
        self.users = {}
        self.active_user_ids = []
        self.log = get_logger(self)  # per-class logger
        self.log.info("UM001 UserManager initialised")

    def get_all_users(self):
        return self.users

    def get_user(self, user_id):
        return self.users.get(user_id)

    def add_users(self, new_users):
        if not new_users:
            return

        for user_id, data in new_users.items():
            enrol_raw = data.get("enrolmentDate")  # normalized ISO string from core
            try:
                enrol_dt = self.time_handler.parse_client_ts(enrol_raw, mute_naive_warning=True)
            except Exception as e:
                # Should be rare now, but keep a safe fallback and a log
                enrol_dt = self.time_handler.now
                self.log.warning(
                    "UM101 bad enrolmentDate; fallback to now (%s)",
                    ctx(user_id=user_id, extra=f"raw={enrol_raw!r} err={e} now={enrol_dt.isoformat()}"),
                )

            # Assign version based on enrolment date vs app release date.
            # Falls back to "old" if CS_APP_RELEASE_DATE is not configured.
            # Guard: an enrolment date in the future is bad EUT data and cannot
            # mean "enrolled after release" — force 'old' so it does not pollute
            # the shared (global) new bandit. A real app update still upgrades the
            # user to 'new' via the explicit rating_version signal.
            if enrol_dt > self.time_handler.now:
                self.log.warning(
                    "UM103 future enrolmentDate; forcing rating_version='old' (%s)",
                    ctx(user_id=user_id, extra=f"enrol={enrol_dt.isoformat()}"),
                )
                rating_version = "old"
            elif _APP_RELEASE_DT is not None and enrol_dt >= _APP_RELEASE_DT:
                rating_version = "new"
            else:
                rating_version = "old"

            # Create/refresh in-memory user
            self.users[user_id] = User(
                user_id=user_id,
                time_handler=self.time_handler,
                personal_data=data,  # keep ISO string in dict
                intervention_start_date=enrol_dt,  # datetime for computations
                rating_version=rating_version,
            )
            if user_id not in self.active_user_ids:
                self.log.debug("UM102 user added (%s)", ctx(user_id=user_id))
                self.active_user_ids.append(user_id)

    def get_active_user_ids(self):
        return self.active_user_ids

    def disable_users(self, disabled_users):
        for user_id in disabled_users:
            if user_id in self.active_user_ids:
                self.active_user_ids.remove(user_id)
                self.users[user_id].disable()
                self.log.info("UM201 user disabled (%s)", ctx(user_id=user_id))
            else:
                self.log.warning("UM202 disable requested for non-active user (%s)", ctx(user_id=user_id))

    def update_escalation_levels(self, escalation_levels):
        for user_id, level_list in escalation_levels.items():
            user = self.get_user(user_id)
            if not user:
                self.log.warning("UM300 user not found for escalation update (%s)", ctx(user_id=user_id))
                continue

            for level in level_list:
                user.update_escalation_level(level["level"])
            self.log.debug("UM301 escalation updated (%s)", ctx(user_id=user_id, extra=f"n={len(level_list)}"))

    def update_health_habits(self, assessments):
        for user_id, hhs_list in assessments.items():
            user = self.get_user(user_id)
            if not user:
                self.log.warning("UM310 user not found for HHS update (%s)", ctx(user_id=user_id))
                continue

            # HHS are in ascending order by timestamp, so we start updating from the oldest
            for hhs in hhs_list:
                user.update_health_habit_assessment(hhs["hhs"])

    def apply_mission_selected(self, user_id: str, mission: dict) -> None:
        user = self.get_user(user_id)
        if not user:
            self.log.warning("UM320 user not found for mission selection (%s)", ctx(user_id=user_id))
            return

        user.update_missions_and_contents(mission)

    def upgrade_user_version(self, user_id: str) -> None:
        user = self.get_user(user_id)
        if not user:
            self.log.warning("UM400 user not found for version upgrade (%s)", ctx(user_id=user_id))
            return
        if user.rating_version != "new":
            user.rating_version = "new"
            self.log.info("UM401 user upgraded to new rating version (%s)", ctx(user_id=user_id))

    def apply_mission_accomplished(self, user_id: str, mission_id: str, score: float | int, ts) -> None:
        user = self.get_user(user_id)
        if not user:
            self.log.warning("UM330 user not found for mission accomplished (%s)", ctx(user_id=user_id))
            return

        user.set_previous_mission_score(score)
        user.mark_mission_finished(mission_id, ts)
