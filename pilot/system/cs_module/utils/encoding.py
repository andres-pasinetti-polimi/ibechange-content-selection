# --- Unified feature schema -----------------------------------------------
from cs_module.config import (
    PILLARS,
    INTERVENTION_TYPES,
    PERSONAL_DATA_FEATURES,
    PERSONAL_DATA_CATEGORICAL_FEATURES,
    NUMERIC_FEATURES_MIN_MAX,
    FREQUENCY_FEATURE_DEGREES,
    INTERVENTION_MAB_FEATURES,
    TRUE_INTERVENTION_MAB_FEATURES,
    RECOMMENDATION_MAB_FEATURES,
)
from .min_max_norm import min_max_norm


# ------------------------------
# Explicit modeling choices
# ------------------------------

# Gender: one-hot with dropped baseline
GENDER_BASELINE = "female"  # change if your dataset has a different dominant class

# RecruitmentCenter: exclude certain centers from one-hot
RECRUITMENTCENTER_EXCLUDE = {"IEO"}

# Education: ordinal mapping (0..1) for known levels; 'other' and missing handled via a flag
EDUCATION_ORDER = [
    "no-education",
    "primary",
    "secondary",
    "vocational",
    "university",
    "postgraduate",
]
EDUCATION_VALUES = {level: i / (len(EDUCATION_ORDER) - 1) for i, level in enumerate(EDUCATION_ORDER)}
EDUCATION_NEUTRAL = 0.5  # placeholder for 'other' / missing
EDUCATION_OTHER_LABELS = {"other"}  # strings considered "other-like"

# Pillars / intervention types: local exclusions
PILLAR_EXCLUDE = {"smoking"}
INTERVENTION_TYPES_EXCLUDE = {"Education"}

# --- canonical order specs ---------------------------------

MAIN_BLOCK_ORDER = [
    "D",
    "H",
    "HC",
    "HT",
    "EW",
    "ND",
    "P",
    "MF",
    "REC_PRESS_SHORT",
    "REC_PRESS_LONG",
    "IT_PRESS_SHORT",
    "IT_PRESS_LONG",
    "TF_past",
    "TF_sched",
    "IT",
    "NIT",
    "IF_past",
    "IF_sched",
    "RF_past",
    "RF_sched",
    "ER",
    "PR",
    "MS",
]

INTERACTION_ORDER = [
    "ND_x_PR",  # time-in-program × prompted
    "AGE_x_IT",
    "EW_x_IT",  # emotional wellbeing × intervention type
    "MF_x_TF_sched",
    "MF_x_RF_sched",
    "MF_x_IF_sched",
    "NIT_x_IF_sched",
    "AGE_x_RF_sched",
    "AGE_x_REC_PRESS_LONG",
    "MF_x_REC_PRESS_SHORT",
    "MF_x_IT_PRESS_RES",
    "HC_x_RF_sched",
    "HC_x_IT",
]


# ========== Helpers (small, reused) ==========
def _is_nan(x):
    return isinstance(x, float) and x != x


def _clip01(v):
    try:
        return max(0.0, min(1.0, float(v)))
    except Exception:
        return 0.0


def _one_hot_encode(value, vocab):
    return [1 if value == v else 0 for v in vocab]


def _encode_frequency(value, degree, min_val, max_val):
    v = max(min_val, min(max_val, value))  # clip
    return [(v**d - min_val**d) / (max_val**d - min_val**d) for d in range(1, degree + 1)]


def _center(x):  # for interaction-only centering of [0,1] vars
    return x - 0.5


# ------------------------------
# One-hot vocabularies (after local exclusions/baselines)
# ------------------------------


def _gender_vocab():
    base = PERSONAL_DATA_CATEGORICAL_FEATURES["gender"]
    return [v for v in base if v != GENDER_BASELINE]


def _recruitment_vocab():
    base = PERSONAL_DATA_CATEGORICAL_FEATURES["recruitmentCenter"]
    return [v for v in base if v not in RECRUITMENTCENTER_EXCLUDE]


def _pillar_vocab():
    return [p for p in PILLARS if p not in PILLAR_EXCLUDE]


def _intervention_vocab():
    return [t for t in INTERVENTION_TYPES if t not in INTERVENTION_TYPES_EXCLUDE]


# ========== Personal data and atomic encoders ==========
def get_personal_data_dimension():
    dim = 0
    for feature in PERSONAL_DATA_FEATURES:
        if feature == "gender":
            dim += len(_gender_vocab())
        elif feature == "recruitmentCenter":
            dim += len(_recruitment_vocab())
        elif feature == "education":
            dim += 2  # education_ordinal + education_other flag
        else:
            dim += 1  # numeric (e.g., userAge)
    return dim


def get_personal_data_labels():
    labels = []
    for feature in PERSONAL_DATA_FEATURES:
        if feature == "gender":
            labels.extend(_gender_vocab())
        elif feature == "recruitmentCenter":
            labels.extend(_recruitment_vocab())
        elif feature == "education":
            labels.append("education")  # ordinal scalar
            labels.append("education_other")  # binary flag
        else:
            labels.append(feature)  # numeric
    return labels


def get_personal_data_encoding(personal_data):
    enc = []
    for feature in PERSONAL_DATA_FEATURES:
        if feature == "gender":
            vocab = _gender_vocab()
            raw = personal_data.get("gender")
            vals = [0] * len(vocab) if raw not in vocab else _one_hot_encode(raw, vocab)
            enc.extend(vals)

        elif feature == "recruitmentCenter":
            vocab = _recruitment_vocab()
            raw = personal_data.get("recruitmentCenter")
            vals = [0] * len(vocab) if raw not in vocab else _one_hot_encode(raw, vocab)
            enc.extend(vals)

        elif feature == "education":
            raw = personal_data.get("education")
            # Determine if this sample is 'other-like'
            is_other_like = (
                raw is None or _is_nan(raw) or str(raw).lower() in EDUCATION_OTHER_LABELS or raw not in EDUCATION_VALUES
            )
            # Ordinal value: known levels map; other-like → neutral
            val = EDUCATION_VALUES.get(raw, EDUCATION_NEUTRAL)
            enc.append(float(val))
            enc.append(1.0 if is_other_like else 0.0)

        else:  # numeric (e.g., userAge)
            raw = personal_data.get(feature)
            if raw is None or _is_nan(raw):
                val = 0.5
            else:
                val = float(raw)
                if feature in NUMERIC_FEATURES_MIN_MAX:
                    a, b = NUMERIC_FEATURES_MIN_MAX[feature]
                    val = min_max_norm(val, a, b)
            enc.append(float(val))

    return enc


# ========== Core base blocks (order = output order) ==========


BASE_LABELS = {
    "D": get_personal_data_labels(),  # demographics
    "H": [p for p in PILLARS],  # (unused if flag False)
    "HC": ["hhs_current"],  # current pillar HHS in [0,1]
    "HT": ["hhs_total"],
    "EW": ["emotional_wellbeing"],
    "ND": ["num_intervention_days"],
    "P": [f"pillar_{p}" for p in _pillar_vocab()],
    "MF": ["mission_frequency"],
    "IT": [f"IT_{t}" for t in _intervention_vocab()],
    "NIT": ["num_int_types"],
    "REC_PRESS_SHORT": ["rec_pressure_short"],
    "REC_PRESS_LONG": ["rec_pressure_long"],
    "IT_PRESS_SHORT": ["it_pressure_short"],
    "IT_PRESS_LONG": ["it_pressure_long"],
    "TF_past": ["TF_past"],
    "TF_sched": ["TF_sched"],
    "IF_past": ["IF_past"],
    "IF_sched": ["IF_sched"],
    "RF_past": ["RF_past"],
    "RF_sched": ["RF_sched"],
    "ER": ["engagement_rate_past"],
    "PR": ["prompted"],
    "MS": ["prev_mission_score"],
}

BASE_DIMENSIONS = {k: len(v) for k, v in BASE_LABELS.items()}


# ========== Atomic encoders for each block ==========


def get_hhs_encoding(hhs):
    return [hhs.get(p, 50) / 100.0 for p in PILLARS]


def get_total_hhs_encoding(hhs):
    hhs_norm = [hhs.get(p, 50) / 100.0 for p in PILLARS]
    return [sum(hhs_norm) / len(hhs_norm)]


def get_emotional_wellbeing_encoding(hhs):
    return [hhs.get("emotional_wellbeing", 50) / 100.0]


def get_hhs_current_encoding(hhs, pillar):
    return [hhs.get(pillar, 50) / 100.0]


def get_num_intervention_days_encoding(days):
    if days is None:
        return [0.0]
    v = days / (12 * 7)
    return [max(0.0, min(1.0, v))]


def get_pillar_encoding(pillar):
    vocab = _pillar_vocab()
    if pillar not in PILLARS:
        raise ValueError(f"Invalid pillar: {pillar}")
    return _one_hot_encode(pillar, vocab)


def get_mission_frequency_encoding(mf):
    return _encode_frequency(mf, FREQUENCY_FEATURE_DEGREES["MF"], min_val=1, max_val=7)


def _get_frequency_encoding(frequency, block_key):
    # returns a list
    clip = _clip01(frequency)
    freq_encoding = _encode_frequency(clip, FREQUENCY_FEATURE_DEGREES[block_key], min_val=0, max_val=1)
    return freq_encoding


def get_intervention_encoding(intervention, keep_all=False):
    intervention = list(dict.fromkeys(intervention or []))  # dedupe, keep order

    if not all(i in INTERVENTION_TYPES for i in intervention):
        raise ValueError(f"Invalid intervention types: {intervention}")

    if keep_all:
        tags = INTERVENTION_TYPES
    else:
        tags = _intervention_vocab()
    if len(intervention) == 0:
        return [0.0] * len(tags)

    intervention_encoding = [1.0 / len(intervention) if t in intervention else 0.0 for t in tags]
    return intervention_encoding


def get_num_int_types_encoding(intervention):
    k = len(intervention or [])
    return [k / float(len(INTERVENTION_TYPES))]


def get_engagement_rate_encoding(er_value):
    return [_clip01(er_value)]


def get_prompted_encoding(prompted):
    return [1.0 if prompted else 0.0]


# ========== Grab positions used for interactions ==========


def _get_age_centered(D):
    age_idx = BASE_LABELS["D"].index("userAge")
    return _center(D[age_idx])


def _get_hhs_current_centered(H, pillar):
    idx = PILLARS.index(pillar)
    return _center(H[idx])


# ========== Public: labels & dims ==========
def get_intervention_feature_vector_labels(use_sim_space=False):
    labs = ["bias"]

    # Map block → frequency family key used in FREQUENCY_FEATURE_DEGREES
    freq_family = {
        "MF": "MF",
        "TF_past": "TF",
        "TF_sched": "TF",
        "IF_past": "IF",
        "IF_sched": "IF",
        "RF_past": "RF",
        "RF_sched": "RF",
    }

    def _labels_for_block(block_key):
        # Base (already dynamic for D/P/IT via *_vocab() in BASE_LABELS)
        base = list(BASE_LABELS[block_key])

        # Expand polynomial frequency blocks (degree ≥ 1)
        if block_key in freq_family:
            deg = int(FREQUENCY_FEATURE_DEGREES[freq_family[block_key]])
            # single name in BASE_LABELS → expand with suffixes when deg > 1
            if deg <= 1:
                return base
            assert len(base) == 1, f"Expected single label for freq block {block_key}"
            stem = base[0]
            return [f"{stem}_{d}" for d in range(1, deg + 1)]

        return base

    # Main-effect blocks, in canonical order
    if use_sim_space:
        feature_flags = TRUE_INTERVENTION_MAB_FEATURES
    else:
        feature_flags = INTERVENTION_MAB_FEATURES

    for key in MAIN_BLOCK_ORDER:
        if feature_flags.get(key, False):
            labs.extend(_labels_for_block(key))

    # Curated interactions
    for inter_key in INTERACTION_ORDER:
        if not feature_flags.get(inter_key, False):
            continue
        if inter_key in ("AGE_x_IT", "HC_x_IT", "EW_x_IT"):
            labs.extend([f"{inter_key}_{t}" for t in _intervention_vocab()])
        else:
            labs.append(inter_key)

    return labs


def get_dim_intervention_feature_vector(include_bias=True, use_sim_space=False):
    if use_sim_space:
        feature_flags = TRUE_INTERVENTION_MAB_FEATURES
    else:
        feature_flags = INTERVENTION_MAB_FEATURES

    dim = 1 if include_bias else 0
    for key in BASE_LABELS:
        if feature_flags.get(key, False):
            dim += BASE_DIMENSIONS[key]

    # custom interactions
    if feature_flags.get("MF_x_TF_sched", False):
        dim += 1
    if feature_flags.get("MF_x_RF_sched", False):
        dim += 1
    if feature_flags.get("MF_x_IF_sched", False):
        dim += 1
    if feature_flags.get("NIT_x_IF_sched", False):
        dim += 1
    if feature_flags.get("AGE_x_RF_sched", False):
        dim += 1
    if feature_flags.get("AGE_x_IT", False):
        dim += len(_intervention_vocab())
    if feature_flags.get("AGE_x_REC_PRESS_LONG", False):
        dim += 1
    if feature_flags.get("MF_x_REC_PRESS_SHORT", False):
        dim += 1
    if feature_flags.get("MF_x_IT_PRESS_RES", False):
        dim += 1
    if feature_flags.get("HC_x_RF_sched", False):
        dim += 1
    if feature_flags.get("HC_x_IT", False):
        dim += len(_intervention_vocab())
    if feature_flags.get("ND_x_PR", False):
        dim += 1
    if feature_flags.get("EW_x_IT", False):
        dim += len(_intervention_vocab())

    return dim


# --- internal block builders ---------------------------------
def _build_main_blocks(
    personal_data,
    hhs,
    num_intervention_days,
    pillar,
    mission_frequency,
    total_frequency_past_week,
    total_frequency_scheduled,
    intervention,
    rec_pressure_short,
    rec_pressure_long,
    it_pressure_short,
    it_pressure_long,
    intervention_frequency_past_week,
    intervention_frequency_scheduled,
    recommendation_frequency_past_week,
    recommendation_frequency_scheduled,
    er_past_value,
    prompted,
    prev_mission_score,
):
    """
    Build ALL base feature blocks (whether or not they're enabled),
    return dict[str, list[float]].
    """
    D = get_personal_data_encoding(personal_data)
    H = get_hhs_encoding(hhs)
    HC = get_hhs_current_encoding(hhs, pillar)
    HT = get_total_hhs_encoding(hhs)
    EW = get_emotional_wellbeing_encoding(hhs)
    ND = get_num_intervention_days_encoding(num_intervention_days)
    P = get_pillar_encoding(pillar)
    MF = get_mission_frequency_encoding(mission_frequency)

    TF_past = _get_frequency_encoding(total_frequency_past_week, block_key="TF")
    TF_sched = _get_frequency_encoding(total_frequency_scheduled, block_key="TF")

    IT_vec = get_intervention_encoding(intervention)
    NIT = get_num_int_types_encoding(intervention)

    REC_PRESS_SHORT = [_clip01(rec_pressure_short)]
    REC_PRESS_LONG = [_clip01(rec_pressure_long)]
    IT_PRESS_SHORT = [_clip01(it_pressure_short)]
    IT_PRESS_LONG = [_clip01(it_pressure_long)]

    IF_past = _get_frequency_encoding(intervention_frequency_past_week, block_key="IF")
    IF_sched = _get_frequency_encoding(intervention_frequency_scheduled, block_key="IF")

    RF_past = _get_frequency_encoding(recommendation_frequency_past_week, block_key="RF")
    RF_sched = _get_frequency_encoding(recommendation_frequency_scheduled, block_key="RF")

    ER = get_engagement_rate_encoding(er_past_value)
    PR = get_prompted_encoding(prompted)
    MS = [_clip01(prev_mission_score)]

    return {
        "D": D,
        "H": H,
        "HC": HC,
        "HT": HT,
        "EW": EW,
        "ND": ND,
        "P": P,
        "MF": MF,
        "REC_PRESS_SHORT": REC_PRESS_SHORT,
        "REC_PRESS_LONG": REC_PRESS_LONG,
        "IT_PRESS_SHORT": IT_PRESS_SHORT,
        "IT_PRESS_LONG": IT_PRESS_LONG,
        "TF_past": TF_past,
        "TF_sched": TF_sched,
        "IT": IT_vec,
        "NIT": NIT,
        "IF_past": IF_past,
        "IF_sched": IF_sched,
        "RF_past": RF_past,
        "RF_sched": RF_sched,
        "ER": ER,
        "PR": PR,
        "MS": MS,
    }


def _build_interactions(blocks, pillar):
    """
    Build ALL curated interaction features (whether or not they're enabled),
    return dict[str, list[float]].

    We'll index degree-1 entries directly (MF[0], etc.), which is safe
    under FREQUENCY_FEATURE_DEGREES[...] == 1.
    """
    MF = blocks["MF"]
    TF_sched = blocks["TF_sched"]
    RF_sched = blocks["RF_sched"]
    IF_sched = blocks["IF_sched"]
    NIT = blocks["NIT"]
    D = blocks["D"]
    H = blocks["H"]
    EW_block = blocks["EW"]
    IT_vec = blocks["IT"]
    REC_PRESS_SHORT = blocks["REC_PRESS_SHORT"]
    REC_PRESS_LONG = blocks["REC_PRESS_LONG"]
    IT_PRESS_SHORT = blocks["IT_PRESS_SHORT"]
    IT_PRESS_LONG = blocks["IT_PRESS_LONG"]

    ND = blocks["ND"]
    PR = blocks["PR"]
    NDc = _center(ND[0])

    MFc = _center(MF[0])
    TFc_sched = _center(TF_sched[0])
    RFc_sched = _center(RF_sched[0])
    IFc_sched = _center(IF_sched[0])
    NITc = _center(NIT[0])
    age_c = _get_age_centered(D)
    hhs_c = _get_hhs_current_centered(H, pillar)
    ew_c = _center(EW_block[0])  # emotional wellbeing in [0,1] → centered

    # residualized long: keep only the "beyond-immediate" part
    it_long_only = max(0.0, IT_PRESS_LONG[0] - IT_PRESS_SHORT[0])
    it_long_only_c = _center(it_long_only)
    rec_short_c = _center(REC_PRESS_SHORT[0])
    rec_long_c = _center(REC_PRESS_LONG[0])

    return {
        "MF_x_TF_sched": [MFc * TFc_sched],
        "MF_x_RF_sched": [MFc * RFc_sched],
        "MF_x_IF_sched": [MFc * IFc_sched],
        "NIT_x_IF_sched": [NITc * IFc_sched],
        "AGE_x_RF_sched": [age_c * RFc_sched],
        "AGE_x_IT": [age_c * w for w in IT_vec],
        "AGE_x_REC_PRESS_LONG": [age_c * rec_long_c],
        "MF_x_REC_PRESS_SHORT": [MFc * rec_short_c],
        "MF_x_IT_PRESS_RES": [MFc * it_long_only_c],
        "HC_x_RF_sched": [hhs_c * RFc_sched],
        "HC_x_IT": [hhs_c * w for w in IT_vec],
        "ND_x_PR": [NDc * PR[0]],  # extra slop only when prompted
        "EW_x_IT": [ew_c * w for w in IT_vec],
    }


def _flatten_feature_vector(blocks, interactions, feature_flags):
    """
    Deterministically flatten into [bias, ...] using:
    1. MAIN_BLOCK_ORDER
    2. INTERACTION_ORDER

    We guard each block/interaction with feature_flags[<key>].
    """
    fv = [1.0]  # bias first

    # main-effect blocks
    for block_name in MAIN_BLOCK_ORDER:
        if feature_flags.get(block_name, False):
            fv.extend(blocks[block_name])

    # curated interaction terms
    for inter_name in INTERACTION_ORDER:
        if feature_flags.get(inter_name, False):
            fv.extend(interactions[inter_name])

    return tuple(fv)


def get_intervention_feature_vector(
    personal_data,
    hhs,
    num_intervention_days,
    pillar,
    mission_frequency,
    total_frequency_past_week,
    total_frequency_scheduled,
    intervention,
    rec_pressure_short,
    rec_pressure_long,
    it_pressure_short,
    it_pressure_long,
    intervention_frequency_past_week,
    intervention_frequency_scheduled,
    recommendation_frequency_past_week,
    recommendation_frequency_scheduled,
    er_past_value=None,
    prompted=False,
    prev_mission_score=0.0,
    use_sim_space=False,
):
    # 1. Build block dict
    blocks = _build_main_blocks(
        personal_data=personal_data,
        hhs=hhs,
        num_intervention_days=num_intervention_days,
        pillar=pillar,
        mission_frequency=mission_frequency,
        total_frequency_past_week=total_frequency_past_week,
        total_frequency_scheduled=total_frequency_scheduled,
        intervention=intervention,
        rec_pressure_short=rec_pressure_short,
        rec_pressure_long=rec_pressure_long,
        it_pressure_short=it_pressure_short,
        it_pressure_long=it_pressure_long,
        intervention_frequency_past_week=intervention_frequency_past_week,
        intervention_frequency_scheduled=intervention_frequency_scheduled,
        recommendation_frequency_past_week=recommendation_frequency_past_week,
        recommendation_frequency_scheduled=recommendation_frequency_scheduled,
        er_past_value=er_past_value,
        prompted=prompted,
        prev_mission_score=prev_mission_score,
    )

    # 2. Build curated interactions
    interactions = _build_interactions(blocks, pillar)

    # 3. Flatten deterministically under the feature flags
    if use_sim_space:
        fv = _flatten_feature_vector(blocks, interactions, TRUE_INTERVENTION_MAB_FEATURES)
    else:
        fv = _flatten_feature_vector(blocks, interactions, INTERVENTION_MAB_FEATURES)
    return fv


def get_recommendation_feature_vector(recommendation_frequency=None):
    fv = [1]
    if RECOMMENDATION_MAB_FEATURES.get("RF", False):
        if recommendation_frequency is None:
            raise ValueError("recommendation_frequency must be provided when RF feature is enabled.")
        fv.extend(_get_frequency_encoding(frequency=recommendation_frequency, block_key="RF"))
    return tuple(fv)
