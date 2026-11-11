import logging
from datetime import timedelta
from cs_module.utils.encoding import get_intervention_feature_vector, get_intervention_encoding
from cs_module.utils.get_pillar import get_pillar
from cs_module.config import CAP_NUM_REC, CAP_SAME_REC, MAX_BASE_HISTORY


def get_feature_vec_to_rec_ids(
    user,
    missions,
    recommendations,
    mission_id_to_avail_rec_ids,
    selected_recs,
    recent_exposure_history,
    prompted=False,
    forced_sel_ts=None,  # anchor the mission window when provided
    personal_data_override=None,
    hhs_override=None,
    num_intervention_days_override=None,
    use_sim_space=False,
):
    th = user.time_handler

    personal_data = personal_data_override if personal_data_override is not None else user.get_personal_data()
    hhs = hhs_override if hhs_override is not None else user.get_hhs()
    num_intervention_days = (
        num_intervention_days_override
        if num_intervention_days_override is not None
        else user.get_num_intervention_days()
    )

    # ASSUMING ONE MISSION AT A TIME FOR PILOT STUDY

    # ---- Anchor the selection timestamp ----
    if forced_sel_ts is not None:
        sel_ts = forced_sel_ts
    else:
        # Fallback for live selection (pilot assumes 1 mission active)
        user_missions = user.get_open_missions()
        if not user_missions:
            return {}
        sel_ts = th.parse_client_ts(user_missions[0]["selection_timestamp"])

    past_week_window = (sel_ts - timedelta(weeks=1), sel_ts)
    er_past = user.get_engagement_rate(time_window=past_week_window)
    prev_mission_score = user.get_previous_mission_score()

    total_frequency_past_week = user.get_total_frequency(past_week_window)
    total_sched_count, rec_to_sched_count, scheduled_mixes = _scheduled_stats(
        selected_recs=selected_recs, recommendations=recommendations
    )
    total_frequency_scheduled = _tf_sched_norm(total_count=total_sched_count)  # [0,1], same for all arms

    feature_vec_to_rec = {}
    for mission_id, avail_rec_ids in mission_id_to_avail_rec_ids.items():
        for rec_id in avail_rec_ids:
            rec = recommendations.get(rec_id)
            if rec is None:
                logging.warning("FB001 rec_id %r not in catalog — skipping", rec_id)
                continue
            mission = missions[mission_id]
            pillar = get_pillar(rec_id)
            mission_frequency = mission["weekly_frequency"]
            intervention_type = rec["intervention_type"]
            curr_mix = get_intervention_encoding(intervention_type, keep_all=True)

            rec_pressure_short, it_pressure_short = _get_pressures(
                rec_id=rec_id,
                curr_mix=curr_mix,
                recent_exposure_history=recent_exposure_history,
                pressure_history_length=5,
                alpha_rec=0.50,
                alpha_it=0.75,
            )

            rec_pressure_long, it_pressure_long = _get_pressures(
                rec_id=rec_id,
                curr_mix=curr_mix,
                recent_exposure_history=recent_exposure_history,
                pressure_history_length=MAX_BASE_HISTORY,
                alpha_rec=0.95,
                alpha_it=0.95,
            )
            intervention_frequency_past_week = user.get_intervention_frequency(intervention_type, past_week_window)
            intervention_frequency_scheduled = _hist_intersection_burden(curr_mix=curr_mix, past_mixes=scheduled_mixes)
            recommendation_frequency_past_week = user.get_recommendation_frequency(rec_id, past_week_window)
            recommendation_frequency_scheduled = _rf_sched_norm(rec_id=rec_id, rec_to_count=rec_to_sched_count)

            fv = get_intervention_feature_vector(
                personal_data=personal_data,
                hhs=hhs,
                num_intervention_days=num_intervention_days,
                pillar=pillar,
                mission_frequency=mission_frequency,
                total_frequency_past_week=total_frequency_past_week,
                total_frequency_scheduled=total_frequency_scheduled,
                intervention=intervention_type,
                rec_pressure_short=rec_pressure_short,
                rec_pressure_long=rec_pressure_long,
                it_pressure_short=it_pressure_short,
                it_pressure_long=it_pressure_long,
                intervention_frequency_past_week=intervention_frequency_past_week,
                intervention_frequency_scheduled=intervention_frequency_scheduled,
                recommendation_frequency_past_week=recommendation_frequency_past_week,
                recommendation_frequency_scheduled=recommendation_frequency_scheduled,
                er_past_value=er_past,
                prompted=prompted,
                prev_mission_score=prev_mission_score,
                use_sim_space=use_sim_space
            )
            feature_vec_to_rec.setdefault(fv, []).append(rec_id)

    return feature_vec_to_rec


def _scheduled_stats(selected_recs, recommendations):
    """
    Turn the in-progress plan (selected_recs) into:
      - total_sched_count (number of recs so far in this plan)
      - rec_to_sched_count[rec_id] = how many times we've already scheduled this exact rec
      - scheduled_mixes = list of intervention mixtures (each is a prob. dist that sums to 1)
    Only considers items of type 'recommendation'.
    """
    total_sched_count = 0
    rec_to_sched_count = {}
    scheduled_mixes = []

    for entry in selected_recs:
        if entry.get("type") != "recommendation":
            continue
        rid = entry["id"]
        total_sched_count += 1
        rec_to_sched_count[rid] = rec_to_sched_count.get(rid, 0) + 1
        mix = get_intervention_encoding(recommendations[rid]["intervention_type"], keep_all=True)
        scheduled_mixes.append(mix)

    return total_sched_count, rec_to_sched_count, scheduled_mixes


def _tf_sched_norm(total_count):
    """Scheduled total-frequency TF_sched in [0,1]."""
    if CAP_NUM_REC <= 0:
        return 0.0
    return min(1.0, total_count / float(CAP_NUM_REC))


def _hist_intersection_burden(curr_mix, past_mixes):
    """
    Histogram-intersection style density match:
    average_j min(curr_mix[j], past_mix[j]) averaged across past_mixes.
    Returns [0,1].
    """
    if not past_mixes:
        return 0.0

    acc = 0.0
    for pmix in past_mixes:
        acc += sum(min(a, b) for a, b in zip(curr_mix, pmix))
    return acc / len(past_mixes)


def _rf_sched_norm(rec_id, rec_to_count):
    """Scheduled same-item repetition RF_sched in [0,1]."""
    if CAP_SAME_REC <= 0:
        return 0.0
    c = rec_to_count.get(rec_id, 0)
    return min(1.0, c / float(CAP_SAME_REC))


def _get_pressures(rec_id, curr_mix, recent_exposure_history, pressure_history_length=14, alpha_rec=0.8, alpha_it=0.8):
    geom_max_rec = sum(alpha_rec**i for i in range(pressure_history_length))
    geom_max_it = sum(alpha_it**i for i in range(pressure_history_length))

    REC_pressure_raw = 0.0
    IT_pressure_raw = 0.0
    decay_rec = 1.0
    decay_it = 1.0

    for past in reversed(recent_exposure_history[-pressure_history_length:]):
        # same-item pressure
        past_rec_id = past["rec_id"]
        if past_rec_id == rec_id:
            REC_pressure_raw += decay_rec
        decay_rec *= alpha_rec

        # intervention-style pressure using distribution overlap
        past_mix = past["intervention_mix"]  # distribution (list of floats summing to 1)
        if len(curr_mix) != len(past_mix):
            raise ValueError(f"Dimension mismatch: curr_mix={len(curr_mix)}, past_mix={len(past_mix)} for rec {rec_id}")
        sim_it = sum(min(a, b) for a, b in zip(curr_mix, past_mix))
        IT_pressure_raw += decay_it * sim_it
        decay_it *= alpha_it

    rec_pressure_norm = min(REC_pressure_raw / geom_max_rec, 1.0)
    it_pressure_norm = min(IT_pressure_raw / geom_max_it, 1.0)

    return rec_pressure_norm, it_pressure_norm
