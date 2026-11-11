from cs_module.utils.feedback_handler import (
    get_sent_recommendations,
    get_rated_recommendations,
    get_rated_resources,
)
from cs_module.utils.logging_utils import pretty, get_logger, ctx
from cs_module.content_selection.feature_builders import get_feature_vec_to_rec_ids
from datetime import timedelta
from cs_module.utils.encoding import get_intervention_encoding
from cs_module.config import MAX_BASE_HISTORY


class NotificationFeedbackHandler:
    def __init__(
        self,
        time_handler,
        user_manager,
        recommendations,
        missions,
        resources,
        bandit_updater,
    ):
        self.time_handler = time_handler
        self.user_manager = user_manager
        self.recommendations = recommendations
        self.missions = missions
        self.resources = resources
        self.bandit_updater = bandit_updater
        self.log = get_logger(self)

    # --------------- public ---------------
    def handle_notification_sent(self, user_id, event):
        if event.get("event_name") != "notification_sent":
            return

        # In case we are restarting the module and we don't know if
        # a plan has been already provided, we turn off the "plan required" flag
        # whenever we encounter a sent_notification event

        user = self.user_manager.get_user(user_id)
        mission_id = event.get("properties", {}).get("mission_id")
        if not mission_id:
            self.log.warning("NF100 sent event without mission_id (%s)", ctx(user_id=user_id))
            return  # nothing we can do

        sent_ts = self.time_handler.parse_client_ts(event["timestamp"])
        # Find the selection that was active when this send happened
        mission_snap = user.mission_snapshot_at(mission_id, sent_ts)
        if mission_snap is None:
            # Helpful debug: show the most recent selections we know about (processing-time state),
            # and keep event-time logic only inside mission_snapshot_at.
            recent = []
            for m in user.selected_missions_and_contents[-5:]:
                recent.append((m.get("mission"), m.get("selection_timestamp"), m.get("plan_required")))
            self.log.warning(
                "NF101 skip sent: mission not latest selection (%s) recent_selections=%s",
                ctx(user_id=user_id, mission_id=mission_id, extra=f"ts={sent_ts.isoformat()}"),
                recent,
            )
            return

        # prescribed branch: just bookkeeping, no learning
        if mission_snap.get("prescribed", False):
            self._record_prescribed_send(user, event, mission_id, sent_ts)
            self.log.info("NF110 prescribed send recorded (%s)", ctx(user_id=user_id, mission_id=mission_id))
            return

        # non-prescribed branch: tracking
        ctype = event.get("properties", {}).get("content_type")
        if ctype == "recommendation":
            self._handle_sent_recommendation(user_id, event)

        elif ctype == "resource":
            # for the above reason, record that resource has been already sent
            res_id = event["properties"].get("content_id")
            if res_id:
                user.add_received_resource(res_id)

    def handle_recommendation_rating(self, user_id, event):
        if not get_rated_recommendations([event]):
            return
        self.log.info(
            "NF200 rating received (%s)",
            ctx(
                user_id=user_id,
                mission_id=(event.get("properties") or {}).get("mission_id"),
                rec_id=(event.get("properties") or {}).get("content_id"),
            ),
        )
        self.log.debug("NF201 rating payload %s\n%s", user_id, pretty(event))

        user = self.user_manager.get_user(user_id)
        rating = event["properties"]["rating"]
        reward = self.bandit_updater.compute_reward(rating)
        process_id = self._normalize_process_id(event["process_id"])
        ts = self.time_handler.parse_client_ts(event["timestamp"])

        rec_id = event["properties"]["content_id"]
        mission_id = event["properties"]["mission_id"]
        is_eom = self._is_eom(event)

        # --- mission snapshot to get sel_ts/finish_ts ---
        snap = user.mission_snapshot_at(mission_id, ts)
        if snap is None:
            # This happens when the rated mission isn't the user's latest mission
            # selection at/before ts (or there was no selection yet). We skip the update
            # to avoid attributing learning to the wrong mission window.
            self.log.warning(
                "NF202 skip rating: mission not latest selection (%s)",
                ctx(user_id=user_id, mission_id=mission_id, extra=f"ts={ts.isoformat()}"),
            )
            return

        sel_ts = snap["selection_timestamp"]
        finish_ts = snap.get("finish_timestamp") or ts

        # --- build sent sequence (may append synthetic last slot if EOM) ---
        sent_seq = self._replay_sent_sequence(
            user=user,
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
                "NF203 no sent_seq; skip update (%s)",
                ctx(user_id=user_id, mission_id=mission_id, extra=f"is_eom={is_eom}"),
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
                if self._normalize_process_id(ev["process_id"]) == process_id:
                    slot_idx_1based = ev["slot_idx_1based"]
                    break
            if slot_idx_1based is None:
                self.log.warning(
                    "NF204 cannot align rating to sent sequence (%s)",
                    ctx(user_id=user_id, mission_id=mission_id, pid=process_id),
                )
                return

        context_snapshot = user.get_context_snapshot(mission_selection_ts=sel_ts)
        if context_snapshot is None:
            self.log.warning(
                "NF205 missing context snapshot — capturing (%s)",
                ctx(user_id=user_id, mission_id=mission_id, extra=f"sel_ts={sel_ts.isoformat()}"),
            )
            context_snapshot = user.capture_context_snapshot(sel_ts)
            user.context_snapshots.append(context_snapshot)

        # --- recompute FV as-of that slot ---
        fv, debug_ctx = self._recompute_slot_feature_vector(
            user=user,
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
                "NF206 FV recompute failed (%s)",
                ctx(
                    user_id=user_id,
                    mission_id=mission_id,
                    pid=process_id,
                    rec_id=rec_id,
                    extra=f"is_eom={is_eom}",
                ),
            )
            # debug_ctx already includes avail and sent_seq
            self.log.debug(
                "NF206 debug context (%s)",
                ctx(
                    user_id=user_id,
                    mission_id=mission_id,
                    rec_id=rec_id,
                    extra=f"avail={debug_ctx['available_recs_at_sel_ts']} sent_len={len(debug_ctx['sent_seq'])}",
                ),
            )
            return

        # track rating for engagement metrics etc
        user.track_rating(ts, rec_id, is_eom)

        # update bandits
        self.bandit_updater.update_after_recommendation_rating(
            user=user,
            user_id=user_id,
            rec_id=rec_id,
            mission_id=mission_id,
            feature_vector=fv,
            reward=reward,
            process_id=process_id,
            rating_ts=ts,
        )

        # audit row
        audit_row = {
            "user_id": user_id,
            "mission_id": mission_id,
            "rec_id": rec_id,
            # delivery alignment
            "process_id": process_id,
            "slot_idx": slot_idx_1based,
            "prompted": is_eom,
            # timing
            "sel_ts": sel_ts,
            "finish_ts": finish_ts,
            "rating_ts": ts,
            # bandit target
            "reward": reward,
            "fv": fv,
            # snapshot of planning context at this slot
            "planned_so_far": debug_ctx["planned_so_far"],
            "recent_exposure_history_for_slot": debug_ctx["recent_exposure_history_for_slot"],
            "available_recs_at_sel_ts": debug_ctx["available_recs_at_sel_ts"],
            "sent_seq": debug_ctx["sent_seq"],
            "context_snapshot": context_snapshot,
        }

        self.bandit_updater._buf_feature_audit.append(audit_row)

        if is_eom:
            # Treat the prompted recap as if we "delivered" it now,
            # so future fatigue / frequency stats include it.
            try:
                intervention_type = self.recommendations[rec_id]["intervention_type"]
            except KeyError:
                intervention_type = None  # shouldn't happen, but be defensive

            user.track_sent_recommendations(
                timestamp=ts,
                process_id=process_id,
                rec_id=rec_id,
                intervention_type=intervention_type,
                mission_id=mission_id,
                prompted=True,
            )

    def handle_resource_rating(self, user_id, event):
        if not get_rated_resources([event]):
            return
        self.log.info("NF300 resource rating received (%s)", ctx(user_id=user_id))
        self.log.debug("NF301 resource rating payload %s\n%s", user_id, pretty(event))

        res_id = event["properties"]["content_id"]
        if res_id not in self.resources:
            self.log.warning("NF302 unknown resource id (%s)", ctx(user_id=user_id, extra=f"res={res_id}"))
            return

        props = event.get("properties") or {}
        mission_id = props.get("mission_id")
        user = self.user_manager.get_user(user_id)
        ts = self.time_handler.parse_client_ts(event["timestamp"])
        if mission_id:
            snap = user.mission_snapshot_at(mission_id, ts)
            if snap is None:
                self.log.warning(
                    "NF304 resource rating mission not latest selection (skip update) (%s)",
                    ctx(
                        user_id=user_id,
                        mission_id=mission_id,
                        extra=f"ts={ts.isoformat()} res={res_id}",
                    ),
                )
                return
            else:
                # Optional: pillar mismatch debug
                mission_meta = self.missions.get(mission_id, {})
                mission_pillar = mission_meta.get("pillar")
                res_pillar = self.resources[res_id].get("pillar")

                if mission_pillar and res_pillar and mission_pillar != res_pillar:
                    self.log.warning(
                        "NF305 resource rating pillar mismatch (skip update) (%s)",
                        ctx(
                            user_id=user_id,
                            mission_id=mission_id,
                            extra=f"res={res_id} mission_pillar={mission_pillar} res_pillar={res_pillar}",
                        ),
                    )
                    return

        if self._is_eom(event):
            self.log.info("NF303 EOM resource feedback (%s)", ctx(user_id=user_id, extra=f"res={res_id}"))

        reward = self.bandit_updater.compute_reward(event["properties"]["rating"])
        process_id = self._normalize_process_id(event["process_id"])

        self.bandit_updater.update_after_resource_rating(
            user_id=user_id, res_id=res_id, reward=reward, process_id=process_id, rating_ts=ts,
            version=user.rating_version,
        )

    def _record_prescribed_send(self, user, event, mission_id, sent_ts):
        ctype = (event.get("properties") or {}).get("content_type")
        if ctype == "recommendation":
            rec_id = event["properties"].get("content_id")
            if rec_id in self.recommendations:
                intervention = self.recommendations[rec_id]["intervention_type"]
                pid_n = self._normalize_process_id(event["process_id"])
                user.track_sent_recommendations(sent_ts, pid_n, rec_id, intervention, mission_id)
                if rec_id == "SRc52" and "SRc52" not in user.only_one_rec_already:
                    user.only_one_rec_already.append("SRc52")

        elif ctype == "resource":
            res_id = event["properties"].get("content_id")
            if res_id:
                user.add_received_resource(res_id)

    def _handle_sent_recommendation(self, user_id, event):
        if not get_sent_recommendations([event]):
            return

        rec_id = event["properties"]["content_id"]
        if rec_id not in self.recommendations:
            self.log.warning("NF130 unknown rec id (%s)", ctx(user_id=user_id, rec_id=rec_id))
            return
        user = self.user_manager.get_user(user_id)
        # SRc52 appears only once in the intervention
        if rec_id == "SRc52" and "SRc52" not in user.only_one_rec_already:
            user.only_one_rec_already.append("SRc52")
        mission_id = event["properties"].get("mission_id")
        process_id = self._normalize_process_id(event["process_id"])

        ts = self.time_handler.parse_client_ts(event["timestamp"])
        intervention_type = self.recommendations[rec_id]["intervention_type"]
        user.track_sent_recommendations(
            timestamp=ts,
            process_id=process_id,
            rec_id=rec_id,
            intervention_type=intervention_type,
            mission_id=mission_id,
        )

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
                    "process_id": self._normalize_process_id(row["process_id"]),
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

    def _normalize_process_id(self, pid):
        """Normalize process_id to int (handles str, float, None)."""
        if pid is None:
            return 0
        try:
            # Convert float like 123.0 → 123, or "123.0" → 123
            return int(float(pid))
        except (ValueError, TypeError):
            return 0

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
        avail_list = list(user.get_available_recommendations(mission_id, ref_ts=sel_ts, enforce_rules=False))
        avail = {mission_id: avail_list}

        # 4. pull the frozen snapshot for THIS mission selection
        # Extract frozen fields with sane fallbacks
        frozen_personal_data = context_snapshot.get("personal_data", user.get_personal_data())
        frozen_hhs = context_snapshot.get("hhs", user.get_hhs())
        frozen_num_days = context_snapshot.get("num_intervention_days", user.get_num_intervention_days())

        # 4.5 EARLY CHECK: rec must be available for this mission at sel_ts
        if rec_id not in avail_list:
            # This is exactly the ERc41/NM37 type of error.
            self.log.warning(
                "NF207 rec not available at sel_ts — likely off-mission/invalid historical event (%s)",
                ctx(
                    user_id=user.user_id,
                    mission_id=mission_id,
                    rec_id=rec_id,
                    extra=f"sel_ts={sel_ts.isoformat()} prompted={prompted} avail={avail_list}",
                ),
            )

            debug_ctx = {
                "planned_so_far": planned_so_far,
                "recent_exposure_history_for_slot": recent_exposure_history_for_slot,
                "available_recs_at_sel_ts": avail_list,
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
            return None, debug_ctx

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
            use_sim_space=False,
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

    def _is_eom(self, event):
        props = event.get("properties") or {}
        return bool(props.get("is_end_mission") or props.get("is_end_misison"))

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
        base_hist = user.get_recent_exposure_history(up_to_ts=sel_ts, max_events=MAX_BASE_HISTORY)
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
