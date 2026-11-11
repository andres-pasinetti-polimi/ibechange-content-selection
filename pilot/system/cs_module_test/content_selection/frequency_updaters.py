from cs_module.config import MAX_REC_REPETITIONS, MAX_DISTINCT_REC_PER_MISSION


def update_frequency_offsets(
    rec_id,
    mission_id,
    mission_selected,
    intv_freq_off,
    rec_freq_off,
    avail_rec_ids,
    recommendations,
):
    # count selection
    mission_selected.setdefault(mission_id, {})
    mission_selected[mission_id][rec_id] = mission_selected[mission_id].get(rec_id, 0) + 1
    # update offsets
    intv = recommendations[rec_id]["intervention_type"]
    for intervention in intv:
        intv_freq_off[intervention] = intv_freq_off.get(intervention, 0) + 1

    rec_freq_off[rec_id] = rec_freq_off.get(rec_id, 0) + 1

    # enforce repetition limits
    if mission_selected[mission_id][rec_id] >= MAX_REC_REPETITIONS:
        avail_rec_ids[mission_id].remove(rec_id)

    if len(mission_selected[mission_id]) >= MAX_DISTINCT_REC_PER_MISSION:
        avail_rec_ids[mission_id] = list(mission_selected[mission_id].keys())
