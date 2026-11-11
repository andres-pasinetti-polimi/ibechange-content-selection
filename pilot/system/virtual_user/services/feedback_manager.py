from datetime import timedelta
import numpy as np
from cs_module.utils.encoding import get_intervention_encoding
from cs_module.content_selection.feature_builders import get_feature_vec_to_rec_ids
from cs_module.utils.logging_utils import pretty, get_logger, ctx
from virtual_user.utils.contents import load_json_files
from virtual_user.config import (
    REWARD_TYPE,
    INTERVENTION_TYPES,
)
import os
from virtual_user.utils.deterministic_rng import rng_from_key
import logging


class FeedbackManager:
    process_count = 0

    def __init__(self, user, time_handler, num_weeks_per_user):
        self.time_handler = time_handler
        self.user = user
        self.num_weeks_per_user = num_weeks_per_user
        self.missions, self.recommendations, self.resources = load_json_files("as_dict")
        self.log = get_logger(self)
        self.world_seed = int(os.getenv("WORLD_SEED", "0"))
        self.env_name = str(os.getenv("ENV_NAME", "baseline_v1"))

    def _hour_index(self, ts):
        base = self.time_handler.start_time or ts
        return int((ts - base).total_seconds() // 3600)

    def _rng_open(self, week: int, hour_idx: int, k: int):
        return rng_from_key(
            "open", self.env_name, str(self.world_seed), self.user.user_id, str(week), str(hour_idx), str(k)
        )

    def _rng_rate(self, week: int, hour_idx: int, k: int):
        return rng_from_key(
            "rate", self.env_name, str(self.world_seed), self.user.user_id, str(week), str(hour_idx), str(k)
        )

    def _rng_thumb(self, week: int, slot_idx_1based: int, is_eom: bool, content_id: str | None = None):
        # For EOM we MUST include content_id, otherwise all EOM items share the same draw.
        if is_eom:
            if content_id is None:
                raise ValueError("EOM thumb RNG requires content_id for independence.")
            return rng_from_key(
                "thumb",
                self.env_name,
                str(self.world_seed),
                self.user.user_id,
                str(week),
                str(int(slot_idx_1based)),
                "eom",
                str(content_id),
            )
        else:
            # Keep MOM coupling by slot only
            return rng_from_key(
                "thumb",
                self.env_name,
                str(self.world_seed),
                self.user.user_id,
                str(week),
                str(int(slot_idx_1based)),
                "mom",
            )

    def _rng_resource_thumb(self, week: int, resource_id: str, is_eom: bool):
        return rng_from_key(
            "res_thumb",
            self.env_name,
            str(self.world_seed),
            self.user.user_id,
            str(week),
            str(resource_id),
            "eom" if is_eom else "mom",
        )

    def _rng_mission_score(self, week: int, mission_id: str):
        return rng_from_key(
            "mission_score", self.env_name, str(self.world_seed), self.user.user_id, str(week), str(mission_id)
        )

    def simulate_feedback(self):
        events = []

        if self.user.weekly_recommendation_plan.get("plans") and self.user.intervention_week < self.num_weeks_per_user:
            plans = self.user.weekly_recommendation_plan.get("plans", [])
            # define slots as the plan order (1-based)
            indexed_plans = []
            for i, p in enumerate(plans, start=1):
                indexed_plans.append((i, p))

            # Step 1: Get contents sent in the past hour
            hour_items = []
            window_end = self.time_handler.now
            window_start = window_end - timedelta(hours=1)
            for slot_idx, content in indexed_plans:
                ts = self.time_handler.parse_client_ts(content["scheduled_for"])
                if window_start <= ts < window_end:
                    hour_items.append((slot_idx, content))

            # Keep plan order inside the hour (slot order only)
            hour_items.sort(key=lambda x: x[0])

            # Step 2: simulate events for those items
            week = int(self.user.intervention_week)
            for slot_idx, content in hour_items:
                pid = FeedbackManager.process_count

                # SENT
                events.append(
                    {
                        "process_id": pid,
                        "timestamp": content["scheduled_for"],
                        "event_name": "notification_sent",
                        "properties": {
                            "content_id": content["content_id"],
                            "content_type": content["type"],
                            "mission_id": content["mission_id"],
                            "is_end_misison": False,
                        },
                    }
                )
                self.user._last_pid_by_mission_rec = getattr(self.user, "_last_pid_by_mission_rec", {})
                self.user._last_pid_by_mission_rec[(content["mission_id"], content["content_id"])] = pid

                if content["type"] == "recommendation":
                    self.user.track_sent_recommendations(
                        self.time_handler.parse_client_ts(content["scheduled_for"]),
                        pid,
                        content["content_id"],
                        self.recommendations[content["content_id"]]["intervention_type"],
                        content["mission_id"],
                    )

                # OPEN (coupled by global slot
                open_rng = rng_from_key(
                    "open",
                    self.env_name,
                    str(self.world_seed),
                    self.user.user_id,
                    str(week),
                    str(slot_idx),
                )
                u_open = float(open_rng.uniform(0.0, 1.0))

                if u_open < float(self.user.profile["open_probability"]):
                    open_timestamp = content["scheduled_for"]
                    events.append(
                        {
                            "process_id": pid,
                            "timestamp": open_timestamp,
                            "event_name": "notification_opened",
                            "properties": {
                                "content_id": content["content_id"],
                                "content_type": content["type"],
                                "mission_id": content["mission_id"],
                                "is_end_misison": False,
                            },
                        }
                    )

                    # RATE (coupled by global slot)
                    rate_rng = rng_from_key(
                        "rate",
                        self.env_name,
                        str(self.world_seed),
                        self.user.user_id,
                        str(week),
                        str(slot_idx),
                    )
                    u_rate = float(rate_rng.uniform(0.0, 1.0))

                    if u_rate < float(self.user.profile["rate_probability"]):
                        events.append(
                            {
                                "process_id": pid,
                                "timestamp": content["scheduled_for"],
                                "event_name": "notification_rated",
                                "properties": {
                                    "content_id": content["content_id"],
                                    "content_type": content["type"],
                                    "mission_id": content["mission_id"],
                                    "is_end_misison": False,
                                    "rating": self._get_rec_rating(
                                        rec_id=content["content_id"],
                                        mission_id=content["mission_id"],
                                        ts=content["scheduled_for"],
                                        prompted=False,
                                        process_id=pid,
                                    )
                                    if content["type"] == "recommendation"
                                    else self.get_resource_rating(content["content_id"], prompted=False),
                                },
                            }
                        )
                        self.user.current_rated_contents.append(content)
                        self.user.track_rating(
                            self.time_handler.parse_client_ts(content["scheduled_for"]),
                            content["content_id"],
                            is_end_misison=False,
                        )
                FeedbackManager.process_count += 1

        # FIX IF MORE MISSIONS ACCOMPLISHED AT DIFFERENT TIMES
        # --- End-of-mission block ---
        if (
            (self.user.new_mission_expiration is not None)
            and (self.time_handler.now >= self.user.new_mission_expiration)
            and (self.user.intervention_day % 7 == 0)
            and (self.user.intervention_week < self.num_weeks_per_user)
        ):
            for mission_id in self.user.current_missions:
                # Prompt user only for unrated items that have been SENT at least once
                for content in self.user.get_contents_to_rate():
                    if content["type"] != "recommendation":
                        continue
                    pid = self._last_pid_for(content["mission_id"], content["content_id"])
                    if pid is None:
                        # Never sent → no EOM for this content (contract requirement)
                        logging.info(
                            "EOM: skipping (%s,%s) because it was never sent (no last pid).",
                            content["mission_id"],
                            content["content_id"],
                        )
                        continue

                    # Use the prompt time (now) for EOM events
                    eom_ts = self.time_handler.utc_iso(self.time_handler.now - timedelta(seconds=1))

                    # EOM opened (optional; include if your pipeline expects it)
                    events.append(
                        {
                            "process_id": pid,
                            "timestamp": eom_ts,
                            "event_name": "notification_opened",
                            "properties": {
                                "content_id": content["content_id"],
                                "content_type": content["type"],
                                "mission_id": content["mission_id"],
                                "is_end_misison": True,  # <-- FIXED
                            },
                        }
                    )

                    # EOM rated
                    events.append(
                        {
                            "process_id": pid,
                            "timestamp": eom_ts,
                            "event_name": "notification_rated",
                            "properties": {
                                "content_id": content["content_id"],
                                "content_type": content["type"],
                                "mission_id": content["mission_id"],
                                "is_end_misison": True,
                                "rating": self._get_rec_rating(
                                    rec_id=content["content_id"],
                                    mission_id=content["mission_id"],
                                    ts=content["scheduled_for"],
                                    prompted=True,
                                    process_id=pid,
                                )
                                if content["type"] == "recommendation"
                                else self.get_resource_rating(content["content_id"], prompted=True),
                            },
                        }
                    )

                    # Track rating as EOM (prompted=True)
                    self.user.track_rating(
                        self.time_handler.parse_client_ts(eom_ts),
                        content["content_id"],
                        is_end_misison=True,
                    )

                    # DO NOT increment process_count here: we reused an existing pid

                # Emit mission_accomplished after the EOM ratings
                # Emit mission_accomplished after the EOM ratings
                p_ach = float(self.user.profile.get("mission_achieve_probability", 0.7))

                # interpret p_ach as the mean success level and sample a score around it
                # (Beta gives [0,1] support with that mean)
                alpha = max(1e-3, 10.0 * p_ach)
                beta = max(1e-3, 10.0 * (1.0 - p_ach))
                week = int(self.user.intervention_week)
                score_rng = self._rng_mission_score(week=week, mission_id=mission_id)
                score = float(score_rng.beta(alpha, beta))

                events.append(
                    {
                        "process_id": FeedbackManager.process_count,
                        "timestamp": self.time_handler.utc_iso(self.time_handler.now),
                        "event_name": "mission_accomplished",
                        "properties": {
                            "mission_id": mission_id,
                            "score": score,
                        },
                    }
                )
                FeedbackManager.process_count += 1

        return events

    def _get_rec_rating(self, rec_id, mission_id, ts, prompted, process_id):
        res_preferences, int_theta, rec_preferences = self.user.profile["preferences"]
        ts = self.time_handler.parse_client_ts(ts)
        is_eom = prompted

        # --- mission snapshot to get sel_ts/finish_ts ---
        snap = self.user.mission_snapshot_at(mission_id, ts)
        if snap is None:
            # This happens when the rated mission isn't the user's latest mission
            # selection at/before ts (or there was no selection yet). We skip the update
            # to avoid attributing learning to the wrong mission window.
            self.log.warning(
                "FM202 skip rating: mission not latest selection (%s)",
                ctx(user_id=self.user.user_id, mission_id=mission_id, extra=f"ts={ts.isoformat()}"),
            )
            return

        sel_ts = snap["selection_timestamp"]
        finish_ts = snap.get("finish_timestamp") or ts

        # --- build sent sequence (may append synthetic last slot if EOM) ---
        sent_seq = self._replay_sent_sequence(
            user=self.user,
            mission_id=mission_id,
            sel_ts=sel_ts,
            end_ts=finish_ts,
            is_eom=is_eom,
            process_id=process_id,
            eom_rec_id=rec_id,
            eom_ts=finish_ts,
        )
        if not sent_seq:
            self.log.warning(
                "FM203 no sent_seq; skip update (%s)",
                ctx(user_id=self.user.user_id, mission_id=mission_id, extra=f"is_eom={is_eom}"),
            )
            return

        # --- find which slot we're rating ---
        if is_eom:
            # by construction, EOM slot is the LAST element in sent_seq
            slot_idx_1based = sent_seq[-1]["slot_idx_1based"]
        else:
            # "in the moment" rating: match by process_id
            slot_idx_1based = None
            for ev in sent_seq:
                if self.user._normalize_process_id(ev["process_id"]) == process_id:
                    slot_idx_1based = ev["slot_idx_1based"]
                    break
            if slot_idx_1based is None:
                self.log.warning(
                    "FM204 cannot align rating to sent sequence (%s)",
                    ctx(user_id=self.user.user_id, mission_id=mission_id, pid=process_id),
                )
                return

        context_snapshot = self.user.get_context_snapshot(mission_selection_ts=sel_ts)
        if context_snapshot is None:
            self.log.warning(
                "FM205 missing context snapshot — capturing (%s)",
                ctx(user_id=self.user.user_id, mission_id=mission_id, extra=f"sel_ts={sel_ts.isoformat()}"),
            )
            context_snapshot = self.user.capture_context_snapshot(sel_ts)
            self.user.context_snapshots.append(context_snapshot)

        # --- recompute FV as-of that slot ---
        fv, debug_ctx = self._recompute_slot_feature_vector(
            user=self.user,
            mission_id=mission_id,
            rec_id=rec_id,
            sent_seq=sent_seq,
            slot_idx_1based=slot_idx_1based,
            prompted=is_eom,
            sel_ts=sel_ts,
            context_snapshot=context_snapshot,
        )

        if fv is None:
            self.log.warning(
                "FM206 FV recompute failed (%s)", ctx(user_id=self.user.user_id, mission_id=mission_id, pid=process_id)
            )
            return

        if len(int_theta) != len(fv):
            logging.warning(f"Mismatch: int_theta({len(int_theta)}), fv({len(fv)})")

        int_score = np.array(int_theta) @ np.array(fv)
        rec_bias = rec_preferences[rec_id]

        # Assume additive effects of recommendations
        preference_score = int_score + rec_bias

        if REWARD_TYPE == "thumbs":
            prob = 1 / (1 + np.exp(-preference_score))

            week = int(self.user.intervention_week)

            # slot index used to define "which draw" (coupled across policies)
            if prompted:
                # EOM: by construction we rated the final synthetic slot
                slot_idx = sent_seq[-1]["slot_idx_1based"]
            else:
                slot_idx = slot_idx_1based

            thumb_rng = self._rng_thumb(
                week=week,
                slot_idx_1based=int(slot_idx),
                is_eom=bool(prompted),
                content_id=str(rec_id) if prompted else None,  # <-- key fix for EOM independence
            )
            u = float(thumb_rng.uniform(0.0, 1.0))
            rating = "liked" if u < float(prob) else "disliked"

        elif REWARD_TYPE == "float":
            rating = preference_score

        return rating

    def _recompute_slot_feature_vector(
        self, user, mission_id, rec_id, sent_seq, slot_idx_1based, prompted, sel_ts, context_snapshot
    ):
        # 1. Build "selected_recs" up to this slot (mirrors planner's selected_recs)
        # slot_idx_1based is 1-based.
        planned_so_far = self._planned_prefix_for_slot(
            sent_seq=sent_seq, slot_idx_1based=slot_idx_1based, mission_id=mission_id
        )

        # 2. Build the rolling recent_exposure_history *as of this slot*
        recent_exposure_history_for_slot = self._rebuild_recent_history_for_slot(
            user=user,
            sel_ts=sel_ts,
            sent_seq=sent_seq,
            slot_idx_1based=slot_idx_1based,
        )

        # 3. availability view at selection time
        avail = {mission_id: list(user.get_available_recommendations(mission_id, ref_ts=sel_ts))}

        # 4. pull the frozen snapshot for THIS mission selection
        # Extract frozen fields with sane fallbacks
        frozen_personal_data = context_snapshot.get("personal_data", user.get_personal_data())
        frozen_hhs = context_snapshot.get("hhs", user.get_hhs())
        frozen_num_days = context_snapshot.get("num_intervention_days", user.get_num_intervention_days())

        # 5. reuse the SAME builder as planning, but pass overrides + forced sel_ts
        fv_to_rec_ids = get_feature_vec_to_rec_ids(
            user=user,
            missions=self.missions,
            recommendations=self.recommendations,
            mission_id_to_avail_rec_ids=avail,
            selected_recs=planned_so_far,
            recent_exposure_history=recent_exposure_history_for_slot,
            prompted=prompted,
            forced_sel_ts=sel_ts,
            personal_data_override=frozen_personal_data,
            hhs_override=frozen_hhs,
            num_intervention_days_override=frozen_num_days,
            use_sim_space=True,
        )
        fv = self._lookup_fv_for_rec(fv_to_rec_ids, rec_id)

        # build debug_ctx (include snapshot)
        debug_ctx = {
            "planned_so_far": planned_so_far,
            "recent_exposure_history_for_slot": recent_exposure_history_for_slot,
            "available_recs_at_sel_ts": avail.get(mission_id, []),
            "context_snapshot": context_snapshot,
            "sent_seq": [
                {
                    "sent_ts": ev["sent_ts"].isoformat(),
                    "process_id": ev["process_id"],
                    "rec_id": ev["rec_id"],
                    "slot_idx_1based": ev["slot_idx_1based"],
                    "synthetic_eom": ev.get("synthetic_eom", False),
                }
                for ev in sent_seq
            ],
        }

        return fv, debug_ctx

    def get_resource_rating(self, rec_id, prompted: bool = False):
        res_preferences, int_theta, rec_preferences = self.user.profile["preferences"]

        if REWARD_TYPE == "thumbs":
            prob = 1 / (1 + np.exp(-res_preferences[rec_id]))
            week = int(self.user.intervention_week)
            r = self._rng_resource_thumb(week=week, resource_id=str(rec_id), is_eom=bool(prompted))
            u = float(r.uniform(0.0, 1.0))
            return "liked" if u < float(prob) else "disliked"

        elif REWARD_TYPE == "float":
            return res_preferences[rec_id]

    def _scheduled_offsets_before(self, current_content, include_current=False):
        """Return (total_freq_offset, intv_to_freq_offset: dict, rec_to_freq_offset: dict)
        counting ONLY items scheduled before `current_content` in the same weekly plan.
        """
        plans = self.user.weekly_recommendation_plan.get("plans", [])
        total = 0
        intv_to = {}  # type -> float (mixture-weighted sum)
        rec_to = {}  # rec_id -> int

        for c in plans:
            cur_ts = self.time_handler.parse_client_ts(current_content["scheduled_for"])
            ts = self.time_handler.parse_client_ts(c["scheduled_for"])

            if ts > cur_ts or (ts == cur_ts and not include_current):
                break  # reached current or future item

            if c["type"] != "recommendation":
                continue

            rid = c["content_id"]
            mix = get_intervention_encoding(self.recommendations[rid]["intervention_type"])
            # mixture-weighted type counters
            for t, w in zip(INTERVENTION_TYPES, mix):
                if w:
                    intv_to[t] = intv_to.get(t, 0.0) + w

            # per-rec counter
            rec_to[rid] = rec_to.get(rid, 0) + 1

            # global total (counts recommendations only, as in CS)
            total += 1

        return total, intv_to, rec_to

    def _last_pid_for(self, mission_id, rec_id):
        # primary cache
        if hasattr(self.user, "_last_pid_by_mission_rec"):
            if (mission_id, rec_id) in self.user._last_pid_by_mission_rec:
                return self.user._last_pid_by_mission_rec[(mission_id, rec_id)]
        return None

    def _replay_sent_sequence(
        self,
        user,
        mission_id,
        sel_ts,
        end_ts=None,
        is_eom=False,
        process_id=None,
        eom_rec_id=None,
        eom_ts=None,
    ):
        """
        Build the chronological sequence of *delivered* recs for this mission window.

        If is_eom == False:
            return only actual sent notifications (from sent_rec_tracker),
            each with slot_idx_1based.

        If is_eom == True:
            - return all actual sent notifications (slots 1..N),
            - then append a synthetic "EOM slot" as the LAST slot (N+1) for eom_rec_id,
            timestamped at eom_ts (usually finish_ts),
            even if that rec was originally shown way earlier.

            This makes the EOM prompt act like "final slot after everything",
            which matches fatigue/pressure at rating time.
        """

        if end_ts is None:
            end_ts = sel_ts + timedelta(days=7)  # default legacy behavior

        seq = []
        for row in user.sent_rec_tracker.history:
            ts = row["timestamp"]
            mid = row["mission_id"]
            if mid != mission_id:
                continue
            if not (sel_ts <= ts <= end_ts):
                continue
            if row.get("prompted"):
                # not necessary since prompted are always after end_ts...
                # (mission is accomplished, then users are prompted to rate)
                continue

            seq.append(
                {
                    "sent_ts": ts,
                    "process_id": self.user._normalize_process_id(row["process_id"]),
                    "rec_id": row["rec_id"],
                }
            )
        seq.sort(key=lambda x: x["sent_ts"])

        if is_eom and eom_rec_id is not None:
            # append synthetic final slot AFTER all real sends
            # (note: we do NOT try to reuse the original process_id here,
            #  because multiple EOM prompts could reuse old pid; we don't care.
            #  And we don't need pid to compute features.)
            seq.append(
                {
                    "sent_ts": eom_ts or end_ts,
                    "process_id": process_id,  # not needed for fv reconstruction in EOM case
                    "rec_id": eom_rec_id,
                    "synthetic_eom": True,
                }
            )

        for i, ev in enumerate(seq, start=1):
            ev["slot_idx_1based"] = i
        return seq

    def _lookup_fv_for_rec(self, fv_to_rec_ids, rec_id):
        for key_fv, rec_ids in fv_to_rec_ids.items():
            if rec_id in rec_ids:
                try:
                    return [float(x) for x in key_fv]
                except Exception:
                    return None
        return None

    def _planned_prefix_for_slot(self, sent_seq, slot_idx_1based, mission_id):
        """
        Recreate what had already been 'scheduled' (i.e. sent before this slot)
        at the time this slot was chosen.

        We DO NOT inject the prompted/EOM item here. Even for prompted ratings,
        `planned_so_far` is just the stuff that actually went out in earlier slots.
        """
        planned_so_far = []
        for i in range(1, slot_idx_1based):
            prior = sent_seq[i - 1]["rec_id"]
            planned_so_far.append({"id": prior, "type": "recommendation", "mission_id": mission_id})

        return planned_so_far

    def _rebuild_recent_history_for_slot(self, user, sel_ts, sent_seq, slot_idx_1based):
        """
        Recreate the 'recent_exposure_history' array as it would have looked
        at the time slot_idx_1based was chosen during planning.

        1. Start with whatever the user had BEFORE this mission started
           (i.e. up_to_ts = sel_ts).
        2. Append the mixes for each earlier slot in this mission window
           (slots < slot_idx_1based), in chronological order.
        """

        # 1. history before this mission selection
        base_hist = user.get_recent_exposure_history(up_to_ts=sel_ts)
        # Copy so we don't mutate original
        hist = list(base_hist)

        # 2. replay earlier slots in THIS mission
        # sent_seq is already chronological and has slot_idx_1based set
        for ev in sent_seq:
            if ev["slot_idx_1based"] >= slot_idx_1based:
                break  # we only include strictly earlier slots
            past_rec_id = ev["rec_id"]

            # get its mix the same way planner did
            try:
                intervention_type = self.recommendations[past_rec_id]["intervention_type"]
            except KeyError:
                # if somehow rec disappeared from catalog, skip it
                continue

            past_mix = get_intervention_encoding(intervention_type, keep_all=True)

            hist.append(
                {
                    "rec_id": past_rec_id,
                    "intervention_mix": past_mix,
                }
            )

        return hist
