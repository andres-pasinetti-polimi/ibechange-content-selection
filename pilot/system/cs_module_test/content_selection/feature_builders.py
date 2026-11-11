from datetime import timedelta, datetime
from cs_module.utils.encoding import get_intervention_feature_vector
from cs_module.utils.get_pillar import get_pillar


def get_mission_id_to_intervention_mapping(
    user_manager,
    user_id,
    missions,
    recommendations,
    mission_id_to_avail_rec_ids,
    total_frequency_offset,
    intervention_to_freq_offset,
    recommendation_to_freq_offset,
):
    user = user_manager.get_user(user_id)
    personal_data = user.get_personal_data()
    num_intervention_days = user.get_num_intervention_days()
    hhs = user.get_hhs()
    user_missions = [m for m in user.selected_missions_and_contents if m.get("plan_required")]

    mission_to_features = {m["mission"]: {} for m in user_missions}
    # ASSUMING ONE MISSION AT A TIME FOR PILOT STUDY
    sel_ts_str = user_missions[0]["selection_timestamp"]
    sel_ts = datetime.fromisoformat(sel_ts_str)
    time_window = (sel_ts - timedelta(weeks=1), sel_ts)

    for mission_id, avail_rec_ids in mission_id_to_avail_rec_ids.items():
        for rec_id in avail_rec_ids:
            rec = recommendations[rec_id]
            mission = missions[mission_id]
            intervention_type = rec["intervention_type"]
            for intervention in intervention_type:
                intervention_to_freq_offset.setdefault(intervention, 0)
                intervention_frequency_planned = intervention_to_freq_offset[intervention]
            intervention_frequency_planned /= len(intervention_type)

            fv = get_intervention_feature_vector(
                personal_data,
                hhs,
                num_intervention_days,
                get_pillar(rec_id),
                mission_frequency=mission["weekly_frequency"],
                total_frequency_past_week=user.get_total_frequency(time_window),
                total_frequency_planned=total_frequency_offset,
                intervention=intervention_type,
                intervention_frequency_past_week=user.get_intervention_frequency(intervention_type, time_window),
                intervention_frequency_planned=intervention_frequency_planned,
                recommendation_frequency_past_week=user.get_recommendation_frequency(rec_id, time_window),
                recommendation_frequency_planned=recommendation_to_freq_offset.get(rec_id, 0),
            )
            mission_to_features[mission_id].setdefault(fv, []).append(rec_id)

    return mission_to_features
