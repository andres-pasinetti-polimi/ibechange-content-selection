import random
import json
import os
from cs_module.utils.encoding import get_intervention_feature_vector_labels
from virtual_user.utils.contents import load_json_files
from virtual_user.config import REC_PREFERENCE_RANGE, INTERVENTION_MAB_CONFIG
from datetime import datetime


def generate_rec_preferences():
    missions, recommendations, resources = load_json_files("as_dict")
    rec_preference_vector = {}
    for rec_id in list(recommendations.keys()):
        rec_preference_vector[rec_id] = random.uniform(REC_PREFERENCE_RANGE[0], REC_PREFERENCE_RANGE[1])

    return rec_preference_vector


def generate_custom_preferences(overrides=None):
    # Save to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    int_mab_type = INTERVENTION_MAB_CONFIG["type"]
    save_dir = f"virtual_user/outputs/{int_mab_type}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    overrides = overrides or {}
    intervention_labels = get_intervention_feature_vector_labels()
    int_preference_vector = []

    for label in intervention_labels:
        if label in overrides:
            int_preference_vector.append(overrides[label])
        else:
            int_preference_vector.append(random.uniform(-1, 1))

    with open(f"{save_dir}/intervention_preference.json", "w") as f:
        json.dump(int_preference_vector, f, indent=4)

    rec_preference_vector = generate_rec_preferences()
    with open(f"{save_dir}/recommendation_preference.json", "w") as f:
        json.dump(rec_preference_vector, f, indent=4)

    return int_preference_vector, rec_preference_vector
