from cs_module.utils.encoding import get_recommendation_feature_vector
from cs_module.config import (
    RECOMMENDATION_MAB_CONFIG,
    INTERVENTION_MAB_CONFIG,
)


def select_recommendation(
    feature_map,
    intervention_mab,
    recommendation_mab,
    rec_freq_off,
    past_week_time_window,
    data_storage,
    time_handler,
    user,
    select_anyway,
    selection_id,
):
    if not feature_map:
        return None

    cfg = INTERVENTION_MAB_CONFIG["type"]

    if cfg == "None":
        return handle_recommendation_mab_only(
            user, feature_map, recommendation_mab, data_storage, time_handler, select_anyway, selection_id
        )

    return handle_intervention_mab(
        cfg,
        feature_map,
        intervention_mab,
        recommendation_mab,
        user,
        rec_freq_off,
        past_week_time_window,
        data_storage,
        time_handler,
        select_anyway,
        selection_id,
    )


def handle_recommendation_mab_only(
    user, feature_map, recommendation_mab, data_storage, time_handler, select_anyway, selection_id
):
    rec_cfg = RECOMMENDATION_MAB_CONFIG["type"]
    rec_ids = [rid for fv in feature_map for rid in feature_map[fv]]

    if rec_cfg == "RecommendationOptimalBandit":
        sel, sampled = recommendation_mab.select_action([feature_map[fv] for fv in feature_map], list(feature_map))
    elif rec_cfg == "RandomBandit":
        sel, sampled = recommendation_mab.select_action(rec_ids)
    else:
        raise ValueError(f"Unknown recommendation_mab_mode: {rec_cfg}")

    if not select_anyway and sampled["estimated_reward"] <= 0.5:
        return None
    data_storage.add_mab_sample(
        table="recommendation_mab_samples",
        record={
            "user_id": user.user_id,
            "plan_count": selection_id["plan_count"],
            "content_count": selection_id["content_count"],
            "timestamp": time_handler.now.isoformat(),
            "sample": sampled,
        },
    )
    return sel


def handle_intervention_mab(
    cfg,
    feature_map,
    intervention_mab,
    recommendation_mab,
    user,
    rec_freq_off,
    past_week_time_window,
    data_storage,
    time_handler,
    select_anyway,
    selection_id,
):
    fvs = list(feature_map.keys())

    if cfg == "HybridLinearGaussianInverseGammaTS":
        rec_ids = [rid for fv in fvs for rid in feature_map[fv]]
        ext_vecs = [tuple(fv + (1,)) for fv in fvs for _ in feature_map[fv]]
        sel, sampled = intervention_mab.select_action(rec_ids, ext_vecs)
    elif cfg in ["LogisticLaplaceTS", "GlobalLinearGaussianInverseGammaTS"]:
        selected_rec_ids, sampled = intervention_mab.select_action([feature_map[fv] for fv in fvs], fvs)
        if not select_anyway and sampled["estimated_reward"] <= 0.5:
            return None

        sel = select_from_recommendation_mab(
            user,
            selected_rec_ids,
            rec_freq_off,
            past_week_time_window,
            recommendation_mab,
            data_storage,
            time_handler,
            selection_id,
        )
    else:
        raise ValueError(f"Unknown intervention_mab_mode: {cfg}")

    data_storage.add_mab_sample(
        table="intervention_mab_samples",
        record={
            "user_id": user.user_id,
            "plan_count": selection_id["plan_count"],
            "content_count": selection_id["content_count"],
            "timestamp": time_handler.now.isoformat(),
            "sample": sampled,
        },
    )
    return sel


def select_from_recommendation_mab(
    user, rec_ids, rec_freq_off, past_week_time_window, recommendation_mab, data_storage, time_handler, selection_id
):
    rec_cfg = RECOMMENDATION_MAB_CONFIG["type"]

    if rec_cfg == "LinearGaussianInverseGammaTS":
        fvs = [
            get_recommendation_feature_vector(
                user.get_recommendation_frequency(r, past_week_time_window) + rec_freq_off.get(r, 0)
            )
            for r in rec_ids
        ]
        sel, sampled = recommendation_mab.select_action(rec_ids, fvs)
    elif rec_cfg in ["BernoulliBetaTS", "GaussianInverseGammaTS"]:
        sel, sampled = recommendation_mab.select_action(rec_ids)
    else:
        raise ValueError(f"Unknown recommendation_mab_mode: {rec_cfg}")

    data_storage.add_mab_sample(
        table="recommendation_mab_samples",
        record={
            "user_id": user.user_id,
            "plan_count": selection_id["plan_count"],
            "content_count": selection_id["content_count"],
            "timestamp": time_handler.now.isoformat(),
            "sample": sampled,
        },
    )
    return sel


def select_resource(user, resource_mab, data_storage, time_handler, selection_id):
    selected_resources = []
    content_count = 0
    for mission in user.selected_missions_and_contents:
        rids = [r for r in mission.get("resources", []) if r not in user.get_received_resources()]

        if not rids:
            continue

        sel, sampled = resource_mab.select_action(rids)
        data_storage.add_mab_sample(
            table="resource_mab_samples",
            record={
                "user_id": user.user_id,
                "plan_count": selection_id["plan_count"],
                "content_count": content_count,
                "timestamp": time_handler.now.isoformat(),
                "sample": sampled,
            },
        )
        selected_resources.append({"id": sel, "type": "resource", "mission": mission["mission"]})
        user.add_received_resource(sel)  # Assuming it will receive this resource
        content_count += 1

    return selected_resources
