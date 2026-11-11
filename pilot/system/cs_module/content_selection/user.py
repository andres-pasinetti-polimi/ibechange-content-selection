from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any, Optional
from datetime import datetime, timedelta
from cs_module.utils.recommendation_history_tracker import RecommendationHistoryTracker
from cs_module.services.time_handler import TimeHandler
from cs_module.config import CAP_NUM_REC, CAP_SAME_REC, PILLARS
from cs_module.utils.encoding import get_intervention_encoding
from cs_module.utils.logging_utils import get_logger, ctx
from copy import deepcopy
from bisect import bisect_right


def _to_float_or_none(x):
    try:
        return float(x)
    except Exception:
        return None


@dataclass
class User:
    user_id: str
    time_handler: TimeHandler
    intervention_start_date: Optional[datetime] = field(default=None)
    personal_data: Dict[str, Any] = field(default_factory=dict)
    health_habit_assessment: Dict[str, Any] = field(default_factory=dict)
    active: bool = True
    missions_started: bool = False
    selected_missions_and_contents: List[Dict] = field(default_factory=list)
    selected_contents: Dict[str, Any] = field(default_factory=dict)
    sent_rec_tracker: RecommendationHistoryTracker = field(default_factory=RecommendationHistoryTracker)
    disliked_rec_to_remaining_days: Dict[str, int] = field(default_factory=dict)
    past_week_rec_to_frequency: Dict[str, int] = field(default_factory=dict)
    received_resources: List[str] = field(default_factory=list)
    only_one_rec_already: List[str] = field(default_factory=list)
    rating_history: List[Tuple[datetime, str, bool]] = field(default_factory=list)
    previous_mission_score: float = field(default=0.0)
    context_snapshots: List[Any] = field(default_factory=list)
    rating_version: str = field(default="old")

    # ----- logging -----
    def __post_init__(self):
        # Per-instance logger named like "package.module.User"
        self.log = get_logger(self)

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
        if not isinstance(hhs, dict):
            self.log.warning("US200 HHS payload not a dict (%s)", ctx(user_id=self.user_id))
            return

        # 1) Main pillar scores
        for key, value in hhs.items():
            if key in ("components", "emotional_distress"):
                continue
            if key in PILLARS:
                fv = _to_float_or_none(value)
                if fv is None:
                    self.log.warning(
                        "US201 pillar non-numeric; skip (%s)", ctx(user_id=self.user_id, extra=f"{key}={value!r}")
                    )
                else:
                    self.health_habit_assessment[key] = fv
            else:
                self.log.warning(
                    "US202 unknown HHS key; ignore (%s)", ctx(user_id=self.user_id, extra=f"{key}={value!r}")
                )

        # 2) Components (nutrition or emotional_wellbeing)
        if "components" in hhs:
            target_pillar = None
            if "nutrition" in hhs:
                target_pillar = "nutrition"
            elif "emotional_wellbeing" in hhs:
                target_pillar = "emotional_wellbeing"
            else:
                self.log.warning("US203 components missing pillar; skip (%s)", ctx(user_id=self.user_id))
            if target_pillar:
                comp_key = f"{target_pillar}_components"
                if comp_key not in self.health_habit_assessment or not isinstance(
                    self.health_habit_assessment.get(comp_key), dict
                ):
                    self.health_habit_assessment[comp_key] = {}
                for cname, cval in (hhs["components"] or {}).items():
                    fv = _to_float_or_none(cval)
                    if fv is None:
                        self.log.warning(
                            "US204 non-numeric component; skip (%s)",
                            ctx(user_id=self.user_id, extra=f"{target_pillar}.{cname}={cval!r}"),
                        )
                        continue
                    # overwrite or add
                    self.health_habit_assessment[comp_key][cname] = fv

        # 3) Bi-weekly emotional_distress → update emotional_wellbeing_components
        if "emotional_distress" in hhs:
            fv = _to_float_or_none(hhs["emotional_distress"])
            if fv is None:
                self.log.warning(
                    "US205 emotional_distress non-numeric; skip (%s)",
                    ctx(user_id=self.user_id, extra=f"value={hhs['emotional_distress']!r}"),
                )
            else:
                ew_key = "emotional_wellbeing_components"
                if ew_key not in self.health_habit_assessment or not isinstance(
                    self.health_habit_assessment.get(ew_key), dict
                ):
                    self.health_habit_assessment[ew_key] = {}
                self.health_habit_assessment[ew_key]["emotional_distress"] = fv

    def get_hhs(self):
        return self.health_habit_assessment

    def get_num_intervention_days(self):
        """Get the number of days since the user started using the app."""
        if self.intervention_start_date:
            return int((self.time_handler.now - self.intervention_start_date).days)
        return None

    def set_mission_plan_to_false(self, mission_start_time, tol_seconds=1.0):
        if not mission_start_time:
            return

        try:
            if isinstance(mission_start_time, datetime):
                target_dt = mission_start_time
            else:
                target_dt = self.time_handler.parse_client_ts(mission_start_time, mute_naive_warning=True)
        except Exception:
            self.log.warning(
                "US110 cannot parse mission_start_time (%s)",
                ctx(user_id=self.user_id, extra=f"val={mission_start_time!r}"),
            )
            return

        tol = timedelta(seconds=tol_seconds)

        # 🟡 Log current missions before flipping
        self.log.info(
            "US111 plan_required False attempt (%s)",
            ctx(user_id=self.user_id, extra=f"start={target_dt.isoformat()} tol={tol_seconds}s"),
        )
        for m in self.selected_missions_and_contents:
            sel_dt = m.get("_selection_dt")
            if sel_dt is None:
                try:
                    sel_dt = self.time_handler.parse_client_ts(m.get("selection_timestamp"), mute_naive_warning=True)
                    m["_selection_dt"] = sel_dt
                except Exception:
                    continue
                
            self.log.debug(
                "US112 mission slot (%s)",
                ctx(
                    user_id=self.user_id,
                    mission_id=m.get("mission"),
                    extra=f"sel={sel_dt.isoformat() if sel_dt else None} req={m.get('plan_required')}",
                ),
            )

        # 🔵 Flip if timestamps match exactly
        matched = False
        for m in self.selected_missions_and_contents:
            sel_dt = m.get("_selection_dt")
            if sel_dt is None:
                try:
                    sel_dt = self.time_handler.parse_client_ts(m.get("selection_timestamp"))
                except Exception:
                    continue
                m["_selection_dt"] = sel_dt
            if sel_dt and abs(sel_dt - target_dt) <= tol:
                m["plan_required"] = False
                matched = True
                self.log.info(
                    "US113 plan_required flipped (%s)", ctx(user_id=self.user_id, mission_id=m.get("mission"))
                )
                break

        if not matched:
            self.log.warning(
                "US114 no mission matched start_time (%s)",
                ctx(user_id=self.user_id, extra=f"start={target_dt.isoformat()}"),
            )

    def update_missions_and_contents(self, mission: Dict[str, Any]):
        mission["plan_required"] = True

        # Cache parsed datetimes (do this ONCE)
        try:
            sel_dt = self.time_handler.parse_client_ts(mission.get("selection_timestamp"), mute_naive_warning=True)
        except Exception:
            sel_dt = None
        mission["_selection_dt"] = sel_dt

        raw_end = mission.get("mission_end_time")
        if raw_end:
            try:
                mission["_end_dt"] = self.time_handler.parse_client_ts(raw_end, mute_naive_warning=True)
            except Exception:
                mission["_end_dt"] = None
        else:
            mission["_end_dt"] = None

        # finish timestamp cache if present
        raw_finish = mission.get("finish_timestamp")
        if raw_finish:
            try:
                mission["_finish_dt"] = self.time_handler.parse_client_ts(raw_finish, mute_naive_warning=True)
            except Exception:
                mission["_finish_dt"] = None
        else:
            mission["_finish_dt"] = None

        self.selected_missions_and_contents.append(mission)

        # Context snapshot uses sel_dt (cached)
        if sel_dt is None:
            # fall back safely
            sel_dt = self.time_handler.now
        context_snapshot = self.capture_context_snapshot(sel_dt)
        self.context_snapshots.append(context_snapshot)

        if not self.missions_started:
            self.missions_started = True
            self.intervention_start_date = sel_dt

    def get_context_snapshot(self, mission_selection_ts):
        for snap in self.context_snapshots:
            if snap["mission_selection_ts"] == mission_selection_ts:
                return snap

        self.log.warning(
            "US120 missing context snapshot (%s)", ctx(user_id=self.user_id, extra=f"sel_ts={mission_selection_ts}")
        )
        return None

    def get_open_missions(self):
        new_missions = []
        now_dt = self.time_handler.now

        for mission in self.selected_missions_and_contents:
            if not mission.get("plan_required"):
                continue

            sel_dt = mission.get("_selection_dt")
            if sel_dt is None:
                try:
                    sel_dt = self.time_handler.parse_client_ts(
                        mission.get("selection_timestamp"), mute_naive_warning=True
                    )
                except Exception:
                    continue
                mission["_selection_dt"] = sel_dt

            end_dt = mission.get("_end_dt")
            if end_dt is None:
                raw_end = mission.get("mission_end_time")
                if raw_end:
                    try:
                        end_dt = self.time_handler.parse_client_ts(raw_end, mute_naive_warning=True)
                    except Exception:
                        end_dt = None
                if end_dt is None:
                    end_dt = sel_dt + timedelta(days=7)
                mission["_end_dt"] = end_dt

            if end_dt < now_dt:
                mission["plan_required"] = False
                mission["finish_timestamp"] = mission.get("mission_end_time") or end_dt.isoformat()
                mission["_finish_dt"] = end_dt
                self.log.info(
                    "US130 auto-closed expired mission (%s)",
                    ctx(
                        user_id=self.user_id,
                        mission_id=mission.get("mission"),
                        extra=f"end={mission.get('mission_end_time')} now={now_dt.isoformat()}",
                    ),
                )
                continue

            new_missions.append(mission)

        return new_missions

    def get_open_missions_at(self) -> list[dict]:
        return self.get_open_missions()

    def get_open_missions_now(self) -> list[dict]:
        return self.get_open_missions()

    def is_winter_at(self, ref_ts) -> bool:
        return ref_ts.month in (12, 1, 2)

    def is_spring_at(self, ref_ts) -> bool:
        return ref_ts.month in (3, 4, 5)

    def get_available_recommendations(self, mission_id: str, ref_ts=None, enforce_rules: bool = True) -> List[str]:
        if ref_ts is None:
            ref_ts = getattr(self.time_handler, "now", None) or datetime.now()
        for m in self.selected_missions_and_contents:
            if m.get("mission") != mission_id:
                continue
            recommendations = list(m.get("recommendations", []))
            # Season is always referring to ref_ts time (selection time), so it's ok to always consider this constraints
            if "ERc65" in recommendations and not self.is_winter_at(ref_ts):
                recommendations.remove("ERc65")
            if "ERc66" in recommendations and not self.is_winter_at(ref_ts):
                recommendations.remove("ERc66")
            if "ERc110" in recommendations and not self.is_spring_at(ref_ts):
                recommendations.remove("ERc110")

            if enforce_rules:
                for rec_id in self.only_one_rec_already:
                    if rec_id in recommendations:
                        recommendations.remove(rec_id)
            return recommendations
        return []

    def track_sent_recommendations(self, timestamp, process_id, rec_id, intervention_type, mission_id, prompted=False):
        self.sent_rec_tracker.add_recommendation(
            timestamp=timestamp,
            process_id=process_id,
            rec_id=rec_id,
            intervention_type=intervention_type,
            mission_id=mission_id,
            prompted=prompted,
        )

    def get_total_frequency(self, time_window=None):
        """Past intensity of ANY rec, normalized to [0,1] against weekly cap."""
        count = self.sent_rec_tracker.get_count(time_window=time_window, rec_id=None)
        norm = count / float(CAP_NUM_REC)
        return min(1.0, norm)

    def get_recommendation_frequency(self, rec_id, time_window=None):
        """Past intensity of THIS rec_id, normalized to [0,1] against per-item cap."""
        count = self.sent_rec_tracker.get_count(time_window=time_window, rec_id=rec_id)
        norm = count / float(CAP_SAME_REC)
        return min(1.0, norm)

    def get_intervention_frequency(self, intervention_type, time_window=None):
        """
        Mixture-weighted past-week intervention frequency for THIS item:
        """
        if not intervention_type:
            return 0.0
        curr_mix = get_intervention_encoding(intervention_type, keep_all=True)  # len 8
        recent_sends = []
        start, end = time_window if time_window else (None, None)
        for row in self.sent_rec_tracker.history:
            ts = row["timestamp"]
            if start and ts < start:
                continue
            if end and ts >= end:
                break  # history is sorted asc, so we can break
            recent_sends.append(row["intervention_mix"])
        if not recent_sends:
            return 0.0

        raw_load = 0.0
        for past_mix in recent_sends:
            # histogram intersection between current and past mixtures, in [0,1]
            overlap = sum(min(a, b) for a, b in zip(curr_mix, past_mix))
            raw_load += overlap  # still in [0, #recent_sends]

        # 2) normalize to [0,1]
        # worst case is: every past send was exactly the same mix -> overlap = 1 every time
        # so max_possible = len(recent_sends)
        max_possible = len(recent_sends)
        burden = raw_load / max_possible

        return burden

    """
    SRc52: This recommendation must appear only once during the intervention but is mandatory.
    SRc100, SRc101: Only one of these two recommendations can be presented for the same mission.
    """

    def update_avail_recommendations(self, mission_to_recommendations, sel_rec_id):
        for mission, recs in mission_to_recommendations.items():
            # SRc52 appears only once in the intervention
            if sel_rec_id == "SRc52" and sel_rec_id in recs and "SRc52" not in self.only_one_rec_already:
                self.only_one_rec_already.append("SRc52")
                recs.remove("SRc52")

            # SRc100 and SRc101 are mutually exclusive
            if sel_rec_id in {"SRc100", "SRc101"} and sel_rec_id in recs:
                other = "SRc101" if sel_rec_id == "SRc100" else "SRc100"
                if other in recs:
                    recs.remove(other)

            mission_to_recommendations[mission] = recs

        return mission_to_recommendations

    # called from your updater when a rating arrives
    def track_rating(self, timestamp: datetime, rec_id: str, is_end_misison: bool):
        self.rating_history.append((timestamp, rec_id, bool(is_end_misison)))

    def get_engagement_rate(self, time_window=None) -> float:
        """
        ER^{past} = (# voluntary ratings last week) / (# items sent last week).
        - voluntary = not prompted (is_end_misision == False)
        - denominator = items SENT (from sent_rec_tracker) in the same window
        Returns 0.0 if denominator is 0.
        """
        # denominator: # sent
        sent = self.sent_rec_tracker.get_count(time_window=time_window, rec_id=None, exclude_prompted=True)

        if sent == 0:  # FIX
            return 0.0

        # numerator: # voluntary ratings (exclude end-of-week prompted)
        vol = sum(
            1
            for ts, rid, is_end_misision in self.rating_history
            if (time_window is None or (time_window[0] <= ts < time_window[1])) and not is_end_misision
        )
        return max(0.0, min(1.0, vol / sent))

    def set_previous_mission_score(self, score):
        self.previous_mission_score = score

    def get_previous_mission_score(self):
        return self.previous_mission_score

    def mission_snapshot_at(self, mission_id: str, ts: datetime) -> Optional[dict]:
        """
        Return the snapshot for mission_id IFF it is the *last* mission selected
        at or before ts (globally across all missions). Otherwise return None.

        Returns None in three common cases:
        1) The user has no mission selection at/before ts.
        2) There is a selection at/before ts, but its mission_id != requested mission_id
            (i.e., another mission was selected more recently).
        3) Timestamps are malformed and we couldn't parse them.
        """

        def _as_dt(raw):
            if not raw:
                return None
            if isinstance(raw, datetime):
                return raw
            try:
                return self.time_handler.parse_client_ts(raw, mute_naive_warning=True)
            except Exception as e:
                self.log.warning(
                    "US140 bad timestamp in mission_snapshot_at (%s)",
                    ctx(user_id=self.user_id, mission_id=mission_id, extra=f"raw={raw!r} err={e}"),
                )
                return None

        # Find globally last selection at/before ts (across all missions)
        last_record, last_sel_dt = None, None
        for record in self.selected_missions_and_contents:
            sel_dt = record.get("_selection_dt")
            if sel_dt is None:
                sel_dt = _as_dt(record.get("selection_timestamp"))
                record["_selection_dt"] = sel_dt
            if sel_dt and sel_dt <= ts and (last_sel_dt is None or sel_dt > last_sel_dt):
                last_record, last_sel_dt = record, sel_dt

        if last_record is None:
            self.log.debug(
                "US141 no selection at/before ts (%s)",
                ctx(user_id=self.user_id, mission_id=mission_id, extra=f"ts={ts.isoformat()}"),
            )
            return None

        if last_record.get("mission") != mission_id:
            self.log.debug(
                "US142 other mission is latest (%s)",
                ctx(
                    user_id=self.user_id,
                    mission_id=mission_id,
                    extra=f"latest={last_record.get('mission')} sel={last_sel_dt.isoformat() if last_sel_dt else None} ts={ts.isoformat()}",
                ),
            )
            return None

        finish_dt = last_record.get("_finish_dt")
        if finish_dt is None:
            finish_dt = _as_dt(last_record.get("finish_timestamp"))
            last_record["_finish_dt"] = finish_dt

        return {
            "prescribed": bool(last_record.get("prescribed", False)),
            "record": last_record,
            "selection_timestamp": last_sel_dt,
            "finish_timestamp": finish_dt,
        }

    def _normalize_process_id(self, pid):
        """Normalize process_id to int (handles str, float, None)."""
        if pid is None:
            return 0
        try:
            # Convert float like 123.0 → 123, or "123.0" → 123
            return int(float(pid))
        except (ValueError, TypeError):
            return 0

    def mark_mission_finished(self, mission_id: str, finished_ts):
        # normalize ts
        if isinstance(finished_ts, datetime):
            fdt = finished_ts
        else:
            try:
                fdt = self.time_handler.parse_client_ts(finished_ts, mute_naive_warning=True)
            except Exception:
                fdt = self.time_handler.now

        # latest selection at/before finish time
        best, best_sel = None, None
        for m in self.selected_missions_and_contents:
            if m.get("mission") != mission_id:
                continue

            sel_dt = m.get("_selection_dt")
            if sel_dt is None:
                try:
                    sel_dt = self.time_handler.parse_client_ts(m.get("selection_timestamp"), mute_naive_warning=True)
                except Exception:
                    continue
                m["_selection_dt"] = sel_dt

            if sel_dt <= fdt and (best_sel is None or sel_dt > best_sel):
                best, best_sel = m, sel_dt

        if not best:
            return

        # Keep original server data (including any finish_timestamp),
        best["finish_timestamp"] = fdt.isoformat()  # store trusted finish for auditing
        best["_finish_dt"] = fdt
        best["plan_required"] = False
        self.log.info("US150 mission terminated (%s)", ctx(user_id=self.user_id, mission_id=mission_id))

        curr_start_iso = self.selected_contents.get("mission_start_time")
        if curr_start_iso and best_sel:
            try:
                curr_start_dt = self.time_handler.parse_client_ts(curr_start_iso, mute_naive_warning=True)
            except Exception:
                curr_start_dt = None

            if curr_start_dt and abs(curr_start_dt - best_sel) <= timedelta(seconds=1):
                self.selected_contents["mission_end_time"] = fdt.isoformat()
                self.log.info("US151 updated selected_contents window for OMI (%s)", ctx(user_id=self.user_id))

    def is_new_plan_required(self):
        # get_open_missions() will also auto-close expired missions
        return bool(self.get_open_missions())

    def get_recent_exposure_history(self, up_to_ts=None, exclude_prompted=False, max_events=None):
        """
        Return past exposure history up to `up_to_ts` (inclusive),
        optionally skipping prompted events and limiting to the
        last `max_events` items.

        Each element: {"rec_id": ..., "intervention_mix": [...]}
        """
        hist = self.sent_rec_tracker.history

        # 1) restrict by time using bisect, since hist is sorted by timestamp
        if up_to_ts is not None:
            # timestamps list just for this bisect; complexity is O(n), same as old loop
            timestamps = [row["timestamp"] for row in hist]
            hi = bisect_right(timestamps, up_to_ts)
            candidate = hist[:hi]
        else:
            candidate = hist

        # 2) optionally filter out prompted
        if exclude_prompted:
            candidate = [row for row in candidate if not row.get("prompted")]

        # 3) keep only the last max_events if requested
        if max_events is not None and len(candidate) > max_events:
            candidate = candidate[-max_events:]

        # 4) project to the compact dicts used by the rest of the code
        seq = [
            {
                "rec_id": row["rec_id"],
                "intervention_mix": row["intervention_mix"],
            }
            for row in candidate
        ]
        return seq

    def capture_context_snapshot(self, mission_selection_ts):
        # personal_data, hhs, etc. are mutable dicts, so deepcopy.
        context_snapshot = {
            "mission_selection_ts": mission_selection_ts,
            "personal_data": deepcopy(self.personal_data),
            "hhs": deepcopy(self.health_habit_assessment),
            "num_intervention_days": (
                int((mission_selection_ts - self.intervention_start_date).days)
                if self.intervention_start_date
                else None
            ),
        }
        return context_snapshot
