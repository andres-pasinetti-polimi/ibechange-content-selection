from itertools import product
from cs_module.config import (
    PILLARS,
    INTERVENTION_TYPES,
    MAX_REC_REPETITIONS,
    LEAVE_OUT_VARS,
    PERSONAL_DATA_FEATURES,
    PERSONAL_DATA_CATEGORICAL_FEATURES,
    CATEGORICAL_TO_NUMERIC,
    CATEGORICAL_TO_NUMERIC_EXPLICIT,
    NUMERIC_FEATURES_MIN_MAX,
    FREQUENCY_FEATURE_DEGREES,
    INTERVENTION_MAB_FEATURES,
    RECOMMENDATION_MAB_FEATURES,
    MAX_NUM_REC_PER_MISSION,
)
from .min_max_norm import min_max_norm


def get_personal_data_dimension():
    dim = 0
    for feature in PERSONAL_DATA_FEATURES:
        # feature is categorical
        if feature in PERSONAL_DATA_CATEGORICAL_FEATURES:
            # feature encoded as number
            if feature in CATEGORICAL_TO_NUMERIC:
                dim += 1

            # feature encoded as one-hot
            else:
                present = PERSONAL_DATA_CATEGORICAL_FEATURES[feature]
                dim += len(present)
                if feature in LEAVE_OUT_VARS:
                    # subtract only the leave-out values that are actually present (against typos)
                    dim -= sum(1 for v in LEAVE_OUT_VARS[feature] if v in present)

        # feature is numeric
        else:
            dim += 1
    return dim


BASE_DIMENSIONS = {
    "D": get_personal_data_dimension(),
    "H": len(PILLARS),
    "ND": 1,
    "P": len(PILLARS) - len(LEAVE_OUT_VARS.get("pillar", [])),
    "MF": FREQUENCY_FEATURE_DEGREES["MF"],
    "TF": 2,
    "I": len(INTERVENTION_TYPES),  # some contents have no intervention type associated...
    "IF": 2,
    "RF": 2,
}


INTERACTION_PAIRS = {
    "D_H": ("D", "H"),
    "D_P": ("D", "P"),
    "D_I": ("D", "I"),
    "D_MF": ("D", "MF"),
    "D_TF": ("D", "TF"),
    "D_IF": ("D", "IF"),
    "D_RF": ("D", "RF"),
    "P_I": ("P", "I"),
    "P_MF": ("P", "MF"),
    "P_TF": ("P", "TF"),
    "P_IF": ("P", "IF"),
    "P_RF": ("P", "RF"),
    "I_IF": ("I", "IF"),
    "I_RF": ("I", "RF"),
}


# --- Feature Labels ---
def get_personal_data_labels():
    labels = []

    for feature in PERSONAL_DATA_FEATURES:
        # feature is categorical
        if feature in PERSONAL_DATA_CATEGORICAL_FEATURES:
            # feature to be encoded as number
            if feature in CATEGORICAL_TO_NUMERIC:
                labels.append(feature)

            # feature encoded as one-hot
            else:
                values = [
                    val
                    for val in PERSONAL_DATA_CATEGORICAL_FEATURES[feature]
                    if val not in LEAVE_OUT_VARS.get(feature, [])
                ]
                labels.extend(values)

        # feature is numeric
        else:
            labels.append(feature)

    return labels


BASE_LABELS = {
    "D": get_personal_data_labels(),
    "H": [pillar for pillar in PILLARS],
    "ND": "num_intervention_days",
    "P": [f"pillar_{pillar}" for pillar in PILLARS if pillar not in LEAVE_OUT_VARS.get("pillar", [])],
    "MF": "mission_frequency",
    "TF": ["total_frequency_past_weel", "total_frequency_planned"],
    "I": [f"intervention_{intervention}" for intervention in INTERVENTION_TYPES],
    "IF": ["intervention_frequency_past_week", "intervention_frequency_planned"],
    "RF": ["recommendation_frequency_past_week", "recommendation_frequency_planned"],
}


def get_intervention_feature_vector_labels():
    labels = ["bias"]
    for key, base_label in BASE_LABELS:
        if key in INTERVENTION_MAB_FEATURES:
            labels.extend(base_label)

    for inter_key, (a, b) in INTERACTION_PAIRS.items():
        if INTERVENTION_MAB_FEATURES.get(inter_key):
            inter_labels = [f"{label_a}_{label_b}" for label_a in BASE_LABELS[a] for label_b in BASE_LABELS[b]]
            labels.extend(inter_labels)

    return labels


# --- Feature Dimensions ---
def get_dim_intervention_feature_vector(include_bias=True):
    dim = 1 if include_bias else 0

    # 1. Base features
    for key, base_dim in BASE_DIMENSIONS.items():
        if INTERVENTION_MAB_FEATURES.get(key):
            dim += base_dim

    # 2. Interactions
    for inter_key, (a, b) in INTERACTION_PAIRS.items():
        if INTERVENTION_MAB_FEATURES.get(inter_key):
            dim += BASE_DIMENSIONS[a] * BASE_DIMENSIONS[b]

    return dim


def get_dim_recommendation_feature_vector(include_bias=True):
    dim = 1 if include_bias else 0
    if RECOMMENDATION_MAB_FEATURES["RF"]:
        dim += FREQUENCY_FEATURE_DEGREES["RF"]
    return dim


# --- Feature Encoding ---
def one_hot_encode(value, all_values):
    encoding = [1 if value == v else 0 for v in all_values]
    return encoding


def encode_frequency(value, degree, min_val, max_val):
    return [(value**d - min_val**d) / (max_val**d - min_val**d) for d in range(1, degree + 1)]


def get_personal_data_encoding(personal_data):
    encoding = []

    for feature in PERSONAL_DATA_FEATURES:
        # feature is categorical
        if feature in PERSONAL_DATA_CATEGORICAL_FEATURES:
            values = [
                val for val in PERSONAL_DATA_CATEGORICAL_FEATURES[feature] if val not in LEAVE_OUT_VARS.get(feature, [])
            ]
            # feature to be encoded as number
            if feature in CATEGORICAL_TO_NUMERIC:  # e.g., education
                if feature not in personal_data or personal_data[feature] is None:
                    # Default to 0.5 for missing categorical features
                    encoded_value = 0.5
                elif feature in CATEGORICAL_TO_NUMERIC_EXPLICIT:
                    # Explicit mapping for categorical to numeric
                    encoded_value = CATEGORICAL_TO_NUMERIC_EXPLICIT[feature].get(personal_data.get(feature), 0.5)
                else:
                    if personal_data[feature] not in values:
                        encoded_value = 0.5  # Default for unexpected or leave-out values (e.g., "other" in education)
                    else:
                        encoded_value = values.index(personal_data[feature]) / (len(values) - 1)

            # feature encoded as one-hot
            else:
                encoded_value = one_hot_encode(
                    personal_data[feature],
                    values,
                )

        # feature is numeric
        else:
            if feature not in personal_data or personal_data[feature] is None:
                # Default to 0.5 for missing numeric features
                encoded_value = 0.5
            else:
                encoded_value = personal_data[feature]
                if feature in NUMERIC_FEATURES_MIN_MAX:
                    # Normalize numeric features
                    min_val, max_val = NUMERIC_FEATURES_MIN_MAX[feature]
                    encoded_value = min_max_norm(encoded_value, min_val, max_val)

        if isinstance(encoded_value, list):
            encoding.extend(encoded_value)
        else:
            encoding.append(encoded_value)

    return encoding


def get_hhs_encoding(hhs):
    """Generate and store the health habit assessment encoding.

    Each pillar is normalized by dividing by 100.
    If a value is None or missing, it defaults to 50.
    """

    hhs_encoding = []
    for pillar in PILLARS:
        value = hhs.get(pillar, 50)
        hhs_encoding.append(value / 100)

    return hhs_encoding


def get_num_intervention_days_encoding(num_intervention_days):
    """Generate and store the number of intervention days encoding."""
    # Normalize the number of intervention days to a range of 0 to 1 over a period of 93 days (3 months)
    return [num_intervention_days / 12 * 7]  # 12 weeks


def get_pillar_encoding(pillar):
    """Generate and store the pillar encoding."""
    if pillar not in PILLARS:
        raise ValueError(f"Invalid pillar: {pillar}. Must be one of {PILLARS}.")
    return one_hot_encode(pillar, [p for p in PILLARS if p not in LEAVE_OUT_VARS.get("pillar", [])])


def get_mission_frequency_encoding(mission_frequency):
    """Generate and store the mission frequency encoding."""
    return encode_frequency(
        mission_frequency,
        FREQUENCY_FEATURE_DEGREES["MF"],
        min_val=1,  # Minimum frequency is 1 (once per week)
        max_val=7,  # Maximum frequency is 7 (once per day)
    )


def get_total_frequency_encoding(total_frequency):
    """Generate and store the total frequency encoding."""
    return encode_frequency(
        total_frequency,
        FREQUENCY_FEATURE_DEGREES["TF"],
        min_val=0,  # Minimum frequency is 0 (no missions)
        max_val=MAX_NUM_REC_PER_MISSION,
    )


def get_intervention_encoding(intervention):
    """Generate and store the intervention encoding."""
    if not all(i in INTERVENTION_TYPES for i in intervention):
        raise ValueError(f"Invalid intervention type {intervention}. Must be one of {INTERVENTION_TYPES}.")
    return [1 if i in intervention else 0 for i in INTERVENTION_TYPES]


def get_intervention_frequency_encoding(intervention_frequency):
    """Generate and store the intervention frequency encoding."""
    return encode_frequency(
        intervention_frequency,
        FREQUENCY_FEATURE_DEGREES["IF"],
        min_val=0,  # Minimum frequency is 0 (no interventions)
        max_val=MAX_NUM_REC_PER_MISSION,
    )


def get_recommendation_frequency_encoding(recommendation_frequency):
    """Generate and store the recommendation frequency encoding."""
    return encode_frequency(
        recommendation_frequency,
        FREQUENCY_FEATURE_DEGREES["RF"],
        min_val=0,  # Minimum frequency is 0 (no recommendations)
        max_val=MAX_REC_REPETITIONS,
    )


def get_encodings(
    personal_data,
    hhs,
    num_intervention_days,
    pillar,
    mission_frequency,
    total_frequency_past_week,
    total_frequency_planned,
    intervention,
    intervention_frequency_past_week,
    intervention_frequency_planned,
    recommendation_frequency_past_week,
    recommendation_frequency_planned,
):
    D = get_personal_data_encoding(personal_data)
    H = get_hhs_encoding(hhs)
    ND = get_num_intervention_days_encoding(num_intervention_days)
    P = get_pillar_encoding(pillar)
    MF = get_mission_frequency_encoding(mission_frequency)
    TF = get_total_frequency_encoding(total_frequency_past_week) + get_total_frequency_encoding(total_frequency_planned)
    I = get_intervention_encoding(intervention)
    IF = get_intervention_frequency_encoding(intervention_frequency_past_week) + get_intervention_frequency_encoding(
        intervention_frequency_planned
    )
    RF = get_recommendation_frequency_encoding(
        recommendation_frequency_past_week
    ) + get_recommendation_frequency_encoding(recommendation_frequency_planned)

    return D, H, ND, P, MF, TF, I, IF, RF


def get_intervention_feature_vector(
    personal_data,
    hhs,
    num_intervention_days,
    pillar,
    mission_frequency,
    total_frequency_past_week,
    total_frequency_planned,
    intervention,
    intervention_frequency_past_week,
    intervention_frequency_planned,
    recommendation_frequency_past_week,
    recommendation_frequency_planned,
):
    # --- Base Features ---
    D, H, ND, P, MF, TF, I, IF, RF = get_encodings(
        personal_data,
        hhs,
        num_intervention_days,
        pillar,
        mission_frequency,
        total_frequency_past_week,
        total_frequency_planned,
        intervention,
        intervention_frequency_past_week,
        intervention_frequency_planned,
        recommendation_frequency_past_week,
        recommendation_frequency_planned,
    )

    # 1. Bias
    feature_vector = [1]

    # 2. Base parts
    base_parts = {
        "D": D,
        "H": H,
        "ND": ND,
        "P": P,
        "MF": MF,
        "TF": TF,
        "I": I,
        "IF": IF,
        "RF": RF,
    }

    for key in BASE_DIMENSIONS:
        if INTERVENTION_MAB_FEATURES.get(key):
            feature_vector.extend(base_parts[key])

    # 3. Interactions
    interaction_parts = {
        "D_H": (D, H),
        "D_P": (D, P),
        "D_I": (D, I),
        "D_MF": (D, MF),
        "D_TF": (D, TF),
        "D_IF": (D, IF),
        "D_RF": (D, RF),
        "P_I": (P, I),
        "P_MF": (P, MF),
        "P_TF": (P, TF),
        "P_IF": (P, IF),
        "P_RF": (P, RF),
        "I_IF": (I, IF),
        "I_RF": (I, RF),
    }

    interactions = []
    for key, (a, b) in interaction_parts.items():
        if INTERVENTION_MAB_FEATURES.get(key):
            interactions += [x * y for x, y in product(a, b)]

    feature_vector.extend(interactions)
    return tuple(feature_vector)


def get_recommendation_feature_vector(recommendation_frequency=None):
    # 1. Bias
    feature_vector = [1]

    # 2. Recommendation Frequency Feature
    if RECOMMENDATION_MAB_FEATURES.get("RF"):
        if recommendation_frequency is None:
            raise ValueError("recommendation_frequency must be provided when RF feature is enabled.")
        encoded_rf = get_recommendation_frequency_encoding(recommendation_frequency)
        feature_vector.extend(encoded_rf)

    return tuple(feature_vector)
