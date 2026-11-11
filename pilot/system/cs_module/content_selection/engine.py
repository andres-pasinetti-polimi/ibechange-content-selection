from cs_module.config import (
    MIN_NUM_REC_PER_MISSION,
    MAX_NUM_REC_PER_MISSION,
    MAX_SAME_REC_SENT_PER_MISSION,
    MAX_DISTINCT_REC_PER_MISSION,
    MAX_BASE_HISTORY,
)
from .user_manager import UserManager
from .feature_builders import get_feature_vec_to_rec_ids
from .selector import select_recommendation, select_resource
from cs_module.utils.encoding import get_intervention_encoding
from cs_module.utils.logging_utils import get_logger, ctx
from cs_module.config import RECOMMENDATION_MAB_CONFIG
import uuid

USE_SIM_SPACE_FOR_OPTIMAL = RECOMMENDATION_MAB_CONFIG["type"] == "RecommendationOptimalBandit"


_TABLE_SUFFIX = {"old": "", "new": "_new"}


class RecommendationEngine:
    def __init__(
        self,
        user_manager: UserManager,
        missions,
        resources,
        recommendations,
        resource_mabs: dict,
        intervention_mabs: dict,
        recommendation_mabs: dict,
        data_storage,
        time_handler,
    ):
        """
        *_mabs: {"old": <mab instance>, "new": <mab instance>}
        """
        self.user_manager = user_manager
        self.missions = missions
        self.resources = resources
        self.recommendations = recommendations
        self.resource_mabs = resource_mabs
        self.intervention_mabs = intervention_mabs
        self.recommendation_mabs = recommendation_mabs
        self.data_storage = data_storage
        self.time_handler = time_handler
        self.plan_state = {}
        self.log = get_logger(self)
        self._buf_rec_samples = {"old": [], "new": []}
        self._buf_int_samples = {"old": [], "new": []}
        self._buf_res_samples = {"old": [], "new": []}

    # ---- sinks used by selector.py ----
    def buffer_rec_sample(self, record: dict, version: str = "old") -> None:
        self._buf_rec_samples[version].append(record)

    def buffer_int_sample(self, record: dict, version: str = "old") -> None:
        self._buf_int_samples[version].append(record)

    def buffer_res_sample(self, record: dict, version: str = "old") -> None:
        self._buf_res_samples[version].append(record)

    def flush(self):
        for version, suffix in _TABLE_SUFFIX.items():
            if self._buf_rec_samples[version]:
                self.data_storage.add_mab_samples(
                    f"recommendation_mab_samples{suffix}", self._buf_rec_samples[version]
                )
                self._buf_rec_samples[version].clear()
            if self._buf_int_samples[version]:
                self.data_storage.add_intervention_mab_samples(
                    self._buf_int_samples[version],
                    table=f"intervention_mab_samples{suffix}",
                )
                self._buf_int_samples[version].clear()
            if self._buf_res_samples[version]:
                self.data_storage.add_mab_samples(
                    f"resource_mab_samples{suffix}", self._buf_res_samples[version]
                )
                self._buf_res_samples[version].clear()

    def rotate_plan_id(self, user_id, window_end):
        self.plan_state[user_id] = {
            "plan_id": str(uuid.uuid4()),
            "slot_idx": 0,
            "window_end": self.time_handler.utc_iso(window_end),
        }

    def get_current_plan_id(self, user_id, window_end):
        if user_id not in self.plan_state:
            self.plan_state[user_id] = {
                "plan_id": str(uuid.uuid4()),
                "slot_idx": 0,
                "window_end": self.time_handler.utc_iso(window_end),
            }
        else:
            # important: keep window_end aligned with current tick
            self.plan_state[user_id]["window_end"] = self.time_handler.utc_iso(window_end)
        return self.plan_state[user_id]["plan_id"]

    def get_recommendations_to_send(self, user_id, window_end):
        user = self.user_manager.get_user(user_id)
        if user is None:
            self.log.warning("RE000 unknown user (%s)", ctx(user_id))
            return []

        version = user.rating_version

        # === Single-mission assumption (pilot) ===
        open_missions = user.get_open_missions()
        if not open_missions:
            self.log.info("RE100 no open missions (%s)", ctx(user_id))
            return []

        # If multiple, pick the last selected (most recent); log it so we can inspect upstream.
        if len(open_missions) > 1:
            self.log.warning("RE101 multiple open missions (%s)", ctx(user_id, extra=f"count={len(open_missions)}"))
            open_missions.sort(key=lambda m: m["_selection_dt"])

        mission = open_missions[-1]
        mission_id = mission["mission"]

        # Anchor time = selection timestamp of THIS mission (frozen context)
        sel_ts = mission["_selection_dt"]

        # Freeze context at selection time (and store if missing)
        context_snapshot = user.get_context_snapshot(mission_selection_ts=sel_ts)
        if context_snapshot is None:
            self.log.warning(
                "RE102 missing context snapshot — capturing (%s)",
                ctx(user_id, mission_id, extra=f"sel_ts={sel_ts.isoformat()}"),
            )
            context_snapshot = user.capture_context_snapshot(sel_ts)
            user.context_snapshots.append(context_snapshot)

        # Available recommendations for this mission, evaluated at sel_ts (seasonal filters, one-off constraints)
        avail = user.get_available_recommendations(mission_id, ref_ts=sel_ts)
        # Early exit if menu empty
        if not avail:
            self.log.info("RE103 no available recs at selection (%s)", ctx(user_id, mission_id))
            return []

        if user_id not in self.plan_state:
            self.plan_state[user_id] = {
                "plan_id": str(uuid.uuid4()),
                "slot_idx": 0,
                "window_end": self.time_handler.utc_iso(window_end),
            }
        else:
            self.plan_state[user_id]["window_end"] = self.time_handler.utc_iso(window_end)

        plan_state = self.plan_state[user_id]

        recent_exposure_history = user.get_recent_exposure_history(up_to_ts=sel_ts, max_events=MAX_BASE_HISTORY)

        selected_recs = []
        picked_count_by_rec = {}  # rec_id -> count (single mission)
        count = 0
        # Single-mission FV builder expects {mission_id: [rec_ids]}

        while avail and count < MAX_NUM_REC_PER_MISSION:
            fv_map = get_feature_vec_to_rec_ids(
                user=user,
                missions=self.missions,
                recommendations=self.recommendations,
                mission_id_to_avail_rec_ids={mission_id: list(avail)},
                recent_exposure_history=recent_exposure_history,
                selected_recs=selected_recs,
                prompted=False,
                forced_sel_ts=sel_ts,
                personal_data_override=context_snapshot["personal_data"],
                hhs_override=context_snapshot["hhs"],
                num_intervention_days_override=context_snapshot["num_intervention_days"],
                use_sim_space=USE_SIM_SPACE_FOR_OPTIMAL,
            )

            if not fv_map:
                self.log.warning(
                    "RE110 FV map empty — skipping slot (%s)", ctx(user_id, mission_id, extra="no recs defined?")
                )
                break

            # Force-pick until we have MIN_NUM_REC_PER_MISSION actually selected
            select_anyway = sum(1 for r in selected_recs if r["mission_id"] == mission_id) < MIN_NUM_REC_PER_MISSION

            sel_rec_id, sel_fv = select_recommendation(
                feature_vec_to_rec_ids=fv_map,
                intervention_mab=self.intervention_mabs.get(version),
                recommendation_mab=self.recommendation_mabs.get(version),
                data_storage=self.data_storage,
                time_handler=self.time_handler,
                user=user,
                select_anyway=select_anyway,
                plan_state=plan_state,
                sample_sink=lambda record, v=version: self.buffer_rec_sample(record, version=v),
                intervention_sample_sink=lambda record, v=version: self.buffer_int_sample(record, version=v),
            )
            count += 1

            if sel_rec_id is None:
                self.log.debug("RE111 selector returned None (%s)", ctx(user_id, mission_id))
                continue

            selected_recs.append({"id": sel_rec_id, "type": "recommendation", "mission_id": mission_id})

            chosen_mix = get_intervention_encoding(self.recommendations[sel_rec_id]["intervention_type"], keep_all=True)
            recent_exposure_history.append(
                {
                    "rec_id": sel_rec_id,
                    "intervention_mix": chosen_mix,
                }
            )

            # Caps: same-rec per mission
            picked_count_by_rec[sel_rec_id] = picked_count_by_rec.get(sel_rec_id, 0) + 1
            if picked_count_by_rec[sel_rec_id] >= MAX_SAME_REC_SENT_PER_MISSION:
                if sel_rec_id in avail:
                    avail.remove(sel_rec_id)

            # Caps: max distinct → restrict menu to chosen ones
            if len(picked_count_by_rec) >= MAX_DISTINCT_REC_PER_MISSION:
                used = set(picked_count_by_rec.keys())
                avail[:] = [r for r in avail if r in used]

            # One-off / mutual exclusion pruning (e.g., SRc52, SRc100/SRc101)
            avail = user.update_avail_recommendations({mission_id: avail}, sel_rec_id).get(mission_id, [])

            if not avail:
                break

        return selected_recs

    def get_resources_to_send(self, user_id, window_end, open_missions=None):
        if user_id not in self.plan_state:  # first plan for this user
            self.plan_state[user_id] = {
                "plan_id": str(uuid.uuid4()),
                "slot_idx": 0,
                "window_end": self.time_handler.utc_iso(window_end),
            }
        else:
            self.plan_state[user_id]["window_end"] = self.time_handler.utc_iso(window_end)

        user = self.user_manager.get_user(user_id)
        if user is None:
            self.log.warning("RE000 unknown user (%s)", ctx(user_id))
            return []

        version = user.rating_version
        missions = open_missions if open_missions is not None else user.get_open_missions()
        selected_resources = select_resource(
            user,
            missions,
            self.resource_mabs.get(version),
            self.data_storage,
            self.time_handler,
            plan_state=self.plan_state[user_id],
            sample_sink=lambda record, v=version: self.buffer_res_sample(record, version=v),
        )

        return selected_resources

    def _enforce_caps_and_prune(
        self,
        sel_rec_id,
        mission_id,
        mission_to_selected_rec_to_count,
        mission_to_available_recs,
    ):
        """
        Track how many times we've already scheduled each rec_id *within that mission*
        this planning cycle, and mutate mission_to_available_recs to respect:
        - MAX_SAME_REC_SENT_PER_MISSION
        - MAX_DISTINCT_REC_PER_MISSION
        Returns the updated dicts.
        """
        # bump count for this rec in this mission
        mission_to_selected_rec_to_count.setdefault(mission_id, {})
        mission_to_selected_rec_to_count[mission_id][sel_rec_id] = (
            mission_to_selected_rec_to_count[mission_id].get(sel_rec_id, 0) + 1
        )

        # cap 1: same-rec repetition
        if mission_to_selected_rec_to_count[mission_id][sel_rec_id] >= MAX_SAME_REC_SENT_PER_MISSION:
            if sel_rec_id in mission_to_available_recs[mission_id]:
                mission_to_available_recs[mission_id].remove(sel_rec_id)

        # cap 2: too many distinct recs already picked -> freeze menu to only chosen ones
        if len(mission_to_selected_rec_to_count[mission_id]) >= MAX_DISTINCT_REC_PER_MISSION:
            curr = set(mission_to_available_recs[mission_id])
            used = set(mission_to_selected_rec_to_count[mission_id].keys())
            mission_to_available_recs[mission_id] = list(curr & used)

        return mission_to_selected_rec_to_count, mission_to_available_recs

    def _to_float_array_or_none(self, fv):
        if fv is None:
            return None
        try:
            return [float(x) for x in fv]
        except Exception:
            return None
