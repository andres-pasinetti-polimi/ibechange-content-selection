import time
import numpy as np
from datetime import timedelta
from cs_module.content_selection.engine import RecommendationEngine
from cs_module.content_selection.mab_initialiser import MABInitialiser
from cs_module.content_selection.user_manager import UserManager
from cs_module.content_selection.notification_feedback_handler import NotificationFeedbackHandler
from cs_module.content_selection.bandit_updater import BanditUpdater
from cs_module.utils.data_storage import DataStorage
from cs_module.utils.logging_utils import pretty, get_logger, ctx
import os
from collections import defaultdict

backend = os.getenv("STORAGE_BACKEND", "postgres").lower()


PRIORITY = {
    # same-timestamp causal tie-breakers
    "user_created": 0,
    "version_upgrade": 1,
    "hha": 2,
    "escalation": 3,
    "select": 4,
    "sent": 5,
    "opened": 6,
    "rated": 7,
    "accomplish": 8,
    "disabled": 9,
}


class ContentSelection:
    def __init__(
        self,
        time_handler,
    ):
        self.window_end = None
        self.log = get_logger(self)
        self.log.info("CS001 ContentSelection initialised")
        self.time_handler = time_handler
        self.missions = {}
        self.recommendations = {}
        self.resources = {}
        self.user_manager = UserManager(self.time_handler)
        self.data_storage = DataStorage()

        old_mabs = MABInitialiser(self.data_storage, version="")
        new_mabs = MABInitialiser(self.data_storage, version="_new")

        self.intervention_mabs = {"old": old_mabs.intervention_mab, "new": new_mabs.intervention_mab}
        self.recommendation_mabs = {"old": old_mabs.recommendation_mab, "new": new_mabs.recommendation_mab}
        self.resource_mabs = {"old": old_mabs.resource_mab, "new": new_mabs.resource_mab}

        self.bandit_updater = BanditUpdater(
            data_storage=self.data_storage,
            time_handler=self.time_handler,
            intervention_mabs=self.intervention_mabs,
            recommendation_mabs=self.recommendation_mabs,
            resource_mabs=self.resource_mabs,
        )
        self.notification_feedback_handler = None

        self.recommendation_engine = None

        self._seen_omi_plan_keys = set()  # (user_id, plan_key)
        self._computed_plan_keys = set()  # (user_id, plan_key)
        self._upgraded_user_ids = self.data_storage.get_upgraded_user_ids()

    def _reset_write_buffers(self):
        # buffers for this tick
        self._buf_users = {}  # uid -> user dict (latest wins)
        self._buf_hha = defaultdict(list)  # uid -> [rows]
        self._buf_escalation = defaultdict(list)  # uid -> [rows]
        self._buf_disabled = {}  # uid -> row dict
        self._buf_missions = defaultdict(lambda: {"update_timestamp": None, "new_missions": []})
        self._buf_feedback = defaultdict(list)  # uid -> [events]
        self._buf_version_upgrades = {}  # uid -> {"version": ..., "timestamp": ...}
        # optional: if you later batch audit/samples explicitly
        # self._buf_feature_audit = []
        # self._buf_samples = defaultdict(list)

    def _flush_write_buffers(self):
        # Order matters because of FKs: users first
        if self._buf_users:
            self.data_storage.add_users(self._buf_users)

        if self._buf_disabled:
            self.data_storage.add_disabled_users(self._buf_disabled)

        if self._buf_hha:
            self.data_storage.add_health_habit_assessments(dict(self._buf_hha))

        if self._buf_escalation:
            self.data_storage.add_escalation_levels(dict(self._buf_escalation))

        # missions structure matches DataStorage.add_new_missions_and_contents
        missions_payload = {}
        for uid, d in self._buf_missions.items():
            if d["new_missions"]:
                missions_payload[uid] = d
        if missions_payload:
            self.data_storage.add_new_missions_and_contents(missions_payload)

        if self._buf_feedback:
            feedback_payload = {uid: {"events": evs} for uid, evs in self._buf_feedback.items() if evs}
            if feedback_payload:
                self.data_storage.add_user_feedback(feedback_payload)

        if self._buf_version_upgrades:
            self.data_storage.add_user_version_upgrades(self._buf_version_upgrades)

        self._reset_write_buffers()

    def _public_selected_contents(self, sc: dict) -> dict:
        if not sc:
            return {}

        out = dict(sc)  # shallow copy
        out.pop("_mission_start_dt", None)  # remove datetime cache

        # if contents might contain numpy arrays or other non-JSON types:
        out["contents"] = self.convert_ndarrays_to_lists(out.get("contents", []))

        return out

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

    # Hourly
    def update(self, updates, is_learning, is_intervention, window_end):
        self.window_end = window_end
        t0 = time.time()
        self.log.info("CS010 tick: intervention=%s learning=%s", is_intervention, is_learning)
        if self.notification_feedback_handler is None:
            self.notification_feedback_handler = NotificationFeedbackHandler(
                time_handler=self.time_handler,
                user_manager=self.user_manager,
                recommendations=self.recommendations,
                missions=self.missions,
                resources=self.resources,
                bandit_updater=self.bandit_updater,
            )

        # self.log.debug("CS011 raw updates:\n%s", pretty(updates))

        self._reset_write_buffers()

        # 1) Build per-user stream
        t_build_start = time.time()
        stream_by_user = self._build_event_stream_by_user(updates, self.time_handler)
        t_build_end = time.time()

        # 2) Apply per-user streams (NO DB WRITES inside)
        t_apply_start = time.time()
        for uid, stream in stream_by_user.items():
            self.log.debug("CS012 stream for %s:\n%s", uid, pretty(stream))
            self._apply_stream(uid, stream, is_learning=is_learning)

        # flush once per tick
        self._flush_write_buffers()
        self.bandit_updater.flush()
        t_apply_end = time.time()

        # 3) Plan contents (if intervention enabled)
        t_plan_start = time.time()
        if is_intervention:
            self._select_contents()

        t_plan_end = time.time()
        t_total = time.time() - t0
        self.log.info(
            "PERF CS update: total=%.3fs (build=%.3fs, apply=%.3fs, plan=%.3fs)",
            t_total,
            t_build_end - t_build_start,
            t_apply_end - t_apply_start,
            t_plan_end - t_plan_start,
        )

    def get_selected_contents(self, start_time, end_time):
        selected_contents = {}
        for user_id, user in self.user_manager.get_all_users().items():
            if not user.selected_contents:
                continue

            mission_start = user.selected_contents.get("_mission_start_dt")
            if mission_start is None:
                mission_start = self.time_handler.parse_client_ts(
                    user.selected_contents["mission_start_time"], mute_naive_warning=True
                )
                user.selected_contents["_mission_start_dt"] = mission_start

            if start_time and mission_start < start_time:
                continue
            if end_time and mission_start >= end_time:
                continue

            # ✅ return JSON-safe payload
            selected_contents[user_id] = self._public_selected_contents(user.selected_contents)

        return selected_contents

    def get_omi_fetched_contents(self, omi_fetched_selected_contents):
        """
        Ingest plans OMI confirms it fetched from us
        - Flips plan_required=False for the matching selection window
        - Restores user.selected_contents in memory
        - Persists into DataStorage.add_selected_contents if not already seen
        Idempotent across restarts.
        """
        applied = 0

        # 1) Normalize to {user_id: [plan, ...]} and sort by mission_start_time
        sorted_omi = {}
        for user_id, plans in (omi_fetched_selected_contents or {}).items():
            if isinstance(plans, list):
                lst = [p for p in plans if p]
            elif plans:
                lst = [plans]
            else:
                lst = []
            try:
                for p in lst:
                    if isinstance(p, dict) and "_mission_start_dt" not in p:
                        p["_mission_start_dt"] = self.time_handler.parse_client_ts(p["mission_start_time"])
                lst.sort(key=lambda p: p.get("_mission_start_dt"))

            except Exception as e:
                self.log.warning("CS305 could not sort OMI plans for %s: %s", user_id, e)
            sorted_omi[user_id] = lst

        for user_id, plans in (sorted_omi or {}).items():
            user = self.user_manager.get_user(user_id)
            if not user:
                self.log.warning("CS301 skip OMI fetched: unknown (%s)", ctx(user_id))
                continue
            for plan in plans:
                # Build a stable idempotency key
                key = (user_id, self._plan_key(plan))
                if key in self._seen_omi_plan_keys or key in self._computed_plan_keys:
                    continue  # already ingested in this process

                self.log.info("CS300 ingest OMI fetched (%s)", ctx(user_id))
                self.log.debug("CS301 OMI payload %s\n%s", user_id, pretty(plan))

                # Normalize times
                try:
                    st = plan.get("_mission_start_dt")
                    if st is None:
                        st = self.time_handler.parse_client_ts(plan["mission_start_time"])
                        plan["_mission_start_dt"] = st

                    raw_end = plan.get("mission_end_time")
                    plan["mission_start_time"] = plan["mission_start_time"]
                    plan["mission_end_time"] = raw_end
                except Exception as e:
                    self.log.error("CS309 Bad OMI contents times for user %s: %s; skipping contents", user_id, e)
                    continue

                # Flip the specific mission selection's plan_required to False
                user.set_mission_plan_to_false(st)

                # Restore in-memory selected_contents (what the app expects to serve on /selected_contents)
                pid = plan.get("plan_id") or self._plan_key(plan)
                user.selected_contents = {
                    "mission_start_time": plan["mission_start_time"],
                    "mission_end_time": plan["mission_end_time"],
                    "plan_id": pid,
                    "contents": plan.get("contents") or plan.get("plans") or [],  # tolerate either field name
                }

                # Derive mission_id used in your historical payload (needed by DataStorage schema)
                # Prefer an explicit field if present; else try first rec with mission_id
                mission_id = plan.get("mission_id")
                if not mission_id:
                    for itm in user.selected_contents["contents"] or []:
                        mid = itm.get("mission_id")
                        if mid:
                            mission_id = mid
                            break
                if not mission_id:
                    self.log.warning("CS306 missing mission_id in OMI payload (%s) — persisting anyway", ctx(user_id))

                logical_ts = self.time_handler.utc_iso(st)
                # Persist to DataStorage only once (idempotent per (user, plan_key))
                sc_payload = {
                    "timestamp": logical_ts,
                    "selected_contents": {user_id: user.selected_contents},
                }
                if mission_id:
                    # add_selected_contents writes mission_id[user_id] straight into the
                    # scalar mission_id column — pass the scalar, NOT a dict (was causing
                    # CS308 "can't adapt type 'dict'" on the OMI-bootstrap path).
                    sc_payload["mission_id"] = {user_id: mission_id}

                try:
                    self.data_storage.add_selected_contents(sc_payload)
                except Exception as e:
                    self.log.error("CS308 persist OMI failed (%s): %s", ctx(user_id), e)
                    continue

                # Mark as seen for this process
                self._seen_omi_plan_keys.add(key)
                applied += 1

        return {"status": "ok", "applied": applied}

    def save_recommendation_plans(self, recommendation_plans):
        applied = self.data_storage.add_recommendation_plans(recommendation_plans)
        return {"applied": int(applied)}

    def _normalize_new_users(self, new_users: dict) -> None:
        """Ensure enrolmentDate is a valid ISO string; on invalid, log and use time_handler.now."""
        for uid, data in new_users.items():
            raw = data.get("enrolmentDate")
            try:
                # Treat missing/empty/sentinel as invalid
                if raw in (None, "", "Invalid Date"):
                    raise ValueError("missing or invalid enrolmentDate")
                # Try to parse to confirm it's valid ISO (but keep raw for fidelity)
                raw = self.time_handler.utc_iso(self.time_handler.parse_client_ts(raw, mute_naive_warning=True))
            except Exception as e:
                # Replace with full-precision now()
                dt = self.time_handler.now
                raw = self.time_handler.utc_iso(dt)
                self.log.warning(
                    "CS020 enrolmentDate invalid (%s): %r (%s) → using now=%s", uid, data.get("enrolmentDate"), e, raw
                )
            # Always keep as string exactly as received (or replaced)
            data["enrolmentDate"] = raw

    def _parse_ts(self, th, raw, fallback=None):
        try:
            return th.parse_client_ts(raw, mute_naive_warning=True)
        except Exception:
            return fallback or th.now

    def _build_event_stream_by_user(self, updates, th):
        per_user = {}

        def add(uid, kind, payload, ts):
            # Attach parsed timestamp to payload to avoid reparsing later
            if isinstance(payload, dict):
                payload["_ts_dt"] = ts
            per_user.setdefault(uid, []).append({"ts": ts, "kind": kind, "payload": payload})

        new_users = updates.get("new_users") or {}
        known = self.user_manager.get_all_users()
        known_or_new = set(known) | set(new_users)

        # Helper to log once per section
        def skip_unknown(section, uid):
            self.log.warning("CS040 skip %s: unknown user (%s)", section, uid)

        # Health Habit Assessments
        for uid, items in (updates.get("health_habit_assessments") or {}).items():
            if uid not in known_or_new:
                skip_unknown("HHA", uid)
                continue
            for row in items:
                add(uid, "hha", row, self._parse_ts(th, row.get("assessment_timestamp")))

        # Escalation Levels
        for uid, items in (updates.get("escalation_level") or {}).items():
            if uid not in known_or_new:
                skip_unknown("escalation", uid)
                continue
            for row in items:
                add(uid, "escalation", row, self._parse_ts(th, row.get("update_timestamp")))

        # Disabled Users
        for uid, row in (updates.get("disabled_users") or {}).items():
            if uid not in known_or_new:
                skip_unknown("disabled flag", uid)
                continue
            ts = self._parse_ts(th, (row or {}).get("date_disabled"))
            add(uid, "disabled", row, ts)

        # Mission Selections
        for uid, nm in (updates.get("new_missions_and_contents") or {}).items():
            if uid not in known_or_new:
                skip_unknown("mission selection", uid)
                continue

            cts = nm.get("update_timestamp")
            for m in nm.get("new_missions") or []:
                m2 = dict(m)  # shallow copy so we can annotate
                if cts:
                    m2["_container_update_timestamp"] = cts
                add(uid, "select", m2, self._parse_ts(th, m.get("selection_timestamp")))

        # Rating Version Upgrades
        for uid, rv in (updates.get("rating_version") or {}).items():
            if uid not in known_or_new:
                skip_unknown("rating_version", uid)
                continue
            if (rv or {}).get("version") == "new":
                ts = self._parse_ts(th, rv.get("timestamp"))
                add(uid, "version_upgrade", {}, ts)

        # Feedback Events (sent, opened, rated, accomplish)
        for uid, fb in (updates.get("user_feedback") or {}).items():
            if uid not in known_or_new:
                skip_unknown("feedback", uid)
                continue
            for ev in fb.get("events") or []:
                ts = self._parse_ts(th, ev.get("timestamp"))
                ev["process_id"] = self._normalize_process_id(ev.get("process_id"))
                name = ev.get("event_name")
                kind = {
                    "notification_sent": "sent",
                    "notification_opened": "opened",
                    "notification_rated": "rated",
                    "mission_accomplished": "accomplish",
                }.get(name)
                if kind:
                    add(uid, kind, ev, ts)

        # Inject user_created for truly new users
        for uid, data in new_users.items():
            evs = per_user.get(uid, [])
            earliest = min((e["ts"] for e in evs), default=th.now)
            user_ts = earliest - timedelta(microseconds=1)
            evs.append({"ts": user_ts, "kind": "user_created", "payload": data})
            per_user[uid] = evs

        # Final sort by timestamp, priority, and process_id tie
        for uid, evs in per_user.items():
            for e in evs:
                e["_tie"] = self._normalize_process_id((e["payload"] or {}).get("process_id")) or 0
            evs.sort(key=lambda e: (e["ts"], PRIORITY.get(e["kind"], 99), e["_tie"]))
            for e in evs:
                e.pop("_tie", None)

        return per_user

    def _apply_stream(self, uid, stream, is_learning):
        for e in stream:
            kind, p = e["kind"], e["payload"]
            if kind == "user_created":
                self._normalize_new_users({uid: p})
                self._buf_users[uid] = p
                self.user_manager.add_users({uid: p})
                if uid in self._upgraded_user_ids:
                    self.user_manager.upgrade_user_version(uid)
                continue

            # For any other kind, user must already exist in memory; else skip
            user = self.user_manager.get_user(uid)
            if not user:
                self.log.warning("CS041 skip %s: unknown user (%s)", kind, uid)
                continue

            elif kind == "version_upgrade":
                self.user_manager.upgrade_user_version(uid)
                self._buf_version_upgrades[uid] = {"version": "new", "timestamp": e["ts"].isoformat()}
                self._upgraded_user_ids.add(uid)

            elif kind == "hha":
                self._buf_hha[uid].append(p)
                self.user_manager.update_health_habits({uid: [p]})

            elif kind == "escalation":
                self._buf_escalation[uid].append(p)
                self.user_manager.update_escalation_levels({uid: [p]})

            elif kind == "disabled":
                self._buf_disabled[uid] = p
                self.user_manager.disable_users([uid])

            elif kind == "select":
                upd_ts = (
                    p.get("_container_update_timestamp")
                    or p.get("selection_timestamp")
                    or self.time_handler.now.isoformat()
                )
                d = self._buf_missions[uid]
                d["update_timestamp"] = d["update_timestamp"] or upd_ts
                p_db = {k: v for k, v in p.items() if not k.startswith("_")}
                d["new_missions"].append(p_db)
                self.user_manager.apply_mission_selected(uid, p)

            elif kind == "sent":
                # robust plan_required flip at event timestamp + trackers
                self.notification_feedback_handler.handle_notification_sent(uid, p)
                self._buf_feedback[uid].append(p)

            elif kind == "opened":
                # optional analytics; no learning
                self._buf_feedback[uid].append(p)
                pass

            elif kind == "rated":
                if is_learning:
                    mid = (p.get("properties") or {}).get("mission_id")
                    ts = e["ts"]  # already parsed in stream builder
                    user = self.user_manager.get_user(uid)
                    snap = user.mission_snapshot_at(mid, ts) if user else None
                    if snap is None:
                        # we ignore ratings when the rated mission isn't the last-selected
                        # mission at/before ts (or there was no selection yet). This avoids cross-
                        # mission leakage.
                        self.log.warning(
                            "CS210 skip rating: its mission is not latest selection (%s)",
                            ctx(user_id=uid, mission_id=mid, extra=f"ts={ts.isoformat()}"),
                        )
                    elif snap.get("prescribed", False):
                        self.log.info("CS211 prescribed rating ignored (%s)", ctx(uid, mid))
                    else:
                        ctype = (p.get("properties") or {}).get("content_type")
                        if ctype == "recommendation":
                            self.notification_feedback_handler.handle_recommendation_rating(uid, p)
                        elif ctype == "resource":
                            self.notification_feedback_handler.handle_resource_rating(uid, p)
                        else:
                            self.log.warning(
                                "CS213 rated event with unknown content_type (%s): %r",
                                ctx(user_id=uid, mission_id=mid),
                                ctype,
                            )
                self._buf_feedback[uid].append(p)

            elif kind == "accomplish":
                ts = e["ts"]
                self.user_manager.apply_mission_accomplished(
                    uid, p["properties"]["mission_id"], p["properties"].get("score"), ts
                )
                self._buf_feedback[uid].append(p)

    def _select_contents(self):
        window_end = self.window_end or self.time_handler.now
        window_end_iso = self.time_handler.utc_iso(window_end)
        if self.recommendation_engine is None:
            self.recommendation_engine = RecommendationEngine(
                self.user_manager,
                self.missions,
                self.resources,
                self.recommendations,
                self.resource_mabs,
                self.intervention_mabs,
                self.recommendation_mabs,
                self.data_storage,
                self.time_handler,
            )

        selected_contents = {}
        user_to_mission_id = {}
        for user_id, user in self.user_manager.get_all_users().items():
            new_missions = user.get_open_missions()
            if not new_missions:
                continue
            # PILOT STUDY WILL HAVE ONE MISSION AT A TIME
            # pick the most recent selection, can happen during normal run if user changes inside hour but also if something breaks
            # IMAGINE MY MODULE CRASHES, IN THE GAP TO FIX IT USERS CAN CHANGE MISSIONS,
            # WHEN WE GO BACK TO LIVE WE PLAN ONLY FOR THE MOST RECENT ONE
            mission = max(new_missions, key=lambda m: m["_selection_dt"])
            user_to_mission_id[user_id] = mission["mission"]

            # CLEAR only if other missions have been selected before this one:
            older_missions = [m for m in new_missions if m["mission"] != mission["mission"]]
            for older_m in older_missions:
                user.set_mission_plan_to_false(older_m.get("_selection_dt") or older_m["selection_timestamp"])

            selected_contents[user_id] = {}
            open_missions = new_missions
            selected_resources = self.recommendation_engine.get_resources_to_send(
                user_id, window_end, open_missions=open_missions
            )

            selected_recommendations = self.recommendation_engine.get_recommendations_to_send(user_id, window_end)
            selected_contents[user_id]["contents"] = selected_resources + selected_recommendations

            mission_start_dt = mission["_selection_dt"]

            selected_contents[user_id]["mission_start_time"] = mission["selection_timestamp"]
            selected_contents[user_id]["_mission_start_dt"] = mission["_selection_dt"]
            selected_contents[user_id]["mission_end_time"] = None

            plan_id = self.recommendation_engine.get_current_plan_id(user_id, window_end)
            selected_contents[user_id]["plan_id"] = str(plan_id)
            self.log.info(
                "CS120 planned (%s)",
                ctx(
                    user_id, mission_id=mission["mission"], extra=f"items={len(selected_contents[user_id]['contents'])}"
                ),
            )
            self.log.debug("CS121 plan detail %s\n%s", user_id, pretty(selected_contents[user_id]))

            user.set_mission_plan_to_false(mission_start_dt)
            user.selected_contents = selected_contents[user_id]
            self._computed_plan_keys.add((user_id, f"pid:{plan_id}"))
            self.recommendation_engine.rotate_plan_id(user_id, window_end)

        # after selecting for all users
        self.recommendation_engine.flush()

        if selected_contents:
            timestamp = window_end_iso
            selected_contents_and_timestamps = {
                "timestamp": timestamp,
                "selected_contents": selected_contents,
                "mission_id": user_to_mission_id,
            }
            self.data_storage.add_selected_contents(selected_contents_and_timestamps)

    def _normalize_process_id(self, pid):
        """Normalize process_id to int (handles str, float, None)."""
        if pid is None:
            return 0
        try:
            # Convert float like 123.0 → 123, or "123.0" → 123
            return int(float(pid))
        except (ValueError, TypeError):
            return 0

    def _plan_key(self, plan: dict) -> str:
        pid = (plan or {}).get("plan_id")
        if pid:
            return f"pid:{pid}"

        # fallback
        st_dt = self.time_handler.parse_client_ts(plan["mission_start_time"], mute_naive_warning=True)
        en_raw = plan.get("mission_end_time")
        en_dt = self.time_handler.parse_client_ts(en_raw, mute_naive_warning=True) if en_raw else None

        st = self.time_handler.utc_iso(st_dt)
        en = self.time_handler.utc_iso(en_dt) if en_dt else "None"
        return f"tw:{st}->{en}"
