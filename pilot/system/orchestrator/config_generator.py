import os
import json
import random
from cs_module.utils import encoding as enc

# orchestrator/config_generator.py
from cs_module.utils.encoding import (
    get_intervention_feature_vector_labels,  # ordered labels incl. bias
)

NUM_WEEKLY_USERS = 900

# ============================================================
# Core simulation grid: env → (open_prob, rate_prob, bias_std)
# ============================================================

CORE_SCENARIO_GRID = {
    # core-ref family: OPEN=0.80, RATE=0.60
    "core-ref": {
        "open_prob": 0.80,
        "rate_prob": 0.60,
        "bias_std": 0.50,  # b_a ~ N(0, 0.5^2)
    },
    "core-ref-nobias": {
        "open_prob": 0.80,
        "rate_prob": 0.60,
        "bias_std": 0.00,  # b_a = 0
    },
    "core-ref-highbias": {
        "open_prob": 0.80,
        "rate_prob": 0.60,
        "bias_std": 1.00,  # b_a ~ N(0, 1.0^2)
    },
    # core-lowfb: OPEN=0.40, RATE=0.30
    "core-lowfb": {
        "open_prob": 0.40,
        "rate_prob": 0.30,
        "bias_std": 0.50,
    },
    "core-lowfb-nobias": {
        "open_prob": 0.40,
        "rate_prob": 0.30,
        "bias_std": 0.00,
    },
    # core-highfb: OPEN=0.95, RATE=0.85
    "core-highfb": {
        "open_prob": 0.95,
        "rate_prob": 0.85,
        "bias_std": 0.50,
    },
    "core-highfb-nobias": {
        "open_prob": 0.95,
        "rate_prob": 0.85,
        "bias_std": 0.00,
    },
    "core-highfb-highbias": {
        "open_prob": 0.95,
        "rate_prob": 0.85,
        "bias_std": 1.00,
    },
    "core-topfb-highbias": {
        "open_prob": 1.00,
        "rate_prob": 1.00,
        "bias_std": 1.00,
    },
    "core-topfb-bias05": {
        "open_prob": 1.00,
        "rate_prob": 1.00,
        "bias_std": 0.50,  # b_a ~ N(0, 0.5^2)  (reference)
    },
    "core-topfb-bias10": {
        "open_prob": 1.00,
        "rate_prob": 1.00,
        "bias_std": 1.00,  # b_a ~ N(0, 1.0^2)  (stress)
    },
}

# Fallback for anything not in the core grid (e.g., ad-hoc sims)
DEFAULT_OPEN_PROB = 1.0
DEFAULT_RATE_PROB = 1.0
DEFAULT_BIAS_STD = 0.50  # if you want a “generic” biased world


def _env_params():
    """Return (open_prob, rate_prob, bias_std) for current ENV_NAME."""
    env_name = os.getenv("ENV_NAME")
    cfg = CORE_SCENARIO_GRID.get(env_name, {})
    open_prob = cfg.get("open_prob", DEFAULT_OPEN_PROB)
    rate_prob = cfg.get("rate_prob", DEFAULT_RATE_PROB)
    bias_std = cfg.get("bias_std", DEFAULT_BIAS_STD)
    return open_prob, rate_prob, bias_std


# ---------------------------
# Feature profile definitions
# ---------------------------

# "small" = lean bandit space
SMALL_INTERVENTION_FEATURE_FLAGS = {
    # === Core user-level features ===
    "D": True,  # Demographics
    "H": True,
    "ND": True,
    "P": True,
    # === Mission-level & scheduling ===
    "MF": True,
    "REC_PRESS_SHORT": True,
    "REC_PRESS_LONG": True,
    "IT_PRESS_SHORT": True,
    "IT_PRESS_LONG": True,
    "TF_sched": True,
    # === Intervention-style descriptors ===
    "IT": True,
    "NIT": True,
    # === Behavioral state ===
    "PR": True,
    # === Interactions ===
    "ND_x_PR": True,
    "AGE_x_IT": False,
    "EW_x_IT": False,
}

# "full" = richer simulator space (more interactions ON)
FULL_INTERVENTION_FEATURE_FLAGS = {
    **SMALL_INTERVENTION_FEATURE_FLAGS,
    "AGE_x_IT": True,
    "EW_x_IT": True,
}

# Profile → (bandit_feature_flags, true_feature_flags)
FEATURE_PROFILES: dict[str, tuple[dict, dict]] = {
    "small": (SMALL_INTERVENTION_FEATURE_FLAGS, FULL_INTERVENTION_FEATURE_FLAGS),
    "full": (FULL_INTERVENTION_FEATURE_FLAGS, FULL_INTERVENTION_FEATURE_FLAGS),
}

# ------------------------------------------------------
# Structured–moderate preference ranges (sim feature θ*)
# ------------------------------------------------------

# Default range for most features (moderate magnitude)
DEFAULT_RANGE = (-0.5, 0.5)

# Per-label overrides (by exact label)
RANGE_OVERRIDES: dict[str, tuple[float, float]] = {
    "bias": (0.0, 0.0),
    # Frequencies (fatigue on any “too frequent” thing)
    "REC_PRESS_SHORT": (-0.6, 0.2),
    "REC_PRESS_LONG": (-0.8, 0.0),
    "IT_PRESS_SHORT": (-0.6, 0.2),
    "IT_PRESS_LONG": (-0.8, 0.0),
    "TF_sched": (-0.6, 0.2),
    # Mission frequency slightly positive (habit scaffolding)
    "MF": (0.0, 0.5),
    "ND_x_PR": (-0.25, 0.25),
}

# Prefix overrides for interaction families
PREFIX_OVERRIDES: list[tuple[str, tuple[float, float]]] = [
    ("AGE_x_IT_", (-0.25, 0.25)),  # age × IT interactions
    ("EW_x_IT_", (-0.25, 0.25)),  # EW × IT interactions
]


def _range_for_label(label: str):
    if label in RANGE_OVERRIDES:
        return RANGE_OVERRIDES[label]
    for pref, rng in PREFIX_OVERRIDES:
        if label.startswith(pref):
            return rng
    return DEFAULT_RANGE


# ============================================================
# TRUE intervention preferences in simulator feature space
# ============================================================


def generate_int_preferences(seed: int | None = None, rng: random.Random | None = None):
    """
    Generate structured–moderate TRUE intervention preferences θ*.

    - Uses label-aware ranges so some features have directional priors
      (e.g. fatigue on frequency, neutral bias).
    - Always works in the *simulator* feature space (use_sim_space=True),
      with FULL_INTERVENTION_FEATURE_FLAGS, independent of whatever the
      current config.TRUE_INTERVENTION_MAB_FEATURES says.
    """
    if rng is None:
        rng = random.Random(seed) if seed is not None else random

    # --- force encoder into FULL sim space while we get labels ---
    old_flags = enc.TRUE_INTERVENTION_MAB_FEATURES
    try:
        enc.TRUE_INTERVENTION_MAB_FEATURES = FULL_INTERVENTION_FEATURE_FLAGS
        labels = get_intervention_feature_vector_labels(use_sim_space=True)
    finally:
        enc.TRUE_INTERVENTION_MAB_FEATURES = old_flags

    prefs_dict: dict[str, float] = {}
    theta: list[float] = []

    for lab in labels:
        lo, hi = _range_for_label(lab)
        w = lo if lo == hi else rng.uniform(lo, hi)
        prefs_dict[lab] = w
        theta.append(w)

    return prefs_dict, theta


# Path for generated config.py when no explicit output_path is given
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "generated_configs")


def load_json_files():
    folder = os.path.join(os.path.dirname(__file__), "contents")
    filenames = ["missions.json", "recommendations.json", "resources.json"]
    loaded_files = []

    for name in filenames:
        path = os.path.join(folder, name)
        with open(path, "r") as file:
            loaded_files.append(json.load(file))

    return tuple(loaded_files)


def generate_preferences(seed: int, output_dir: str | None):
    """
    Generate a full 'world' preference bundle:

      - rec_preferences (per recommendation) → this is where b_a lives
      - res_preferences (per resource)
      - int_preferences (per-feature TRUE intervention weights)
      - theta (ordered vector matching sim-space FV)

    All driven by a single RNG seeded with `seed`.

    Bias regime is controlled by the current ENV_NAME via CORE_SCENARIO_GRID:
      - bias_std > 0  → b_a ~ Normal(0, bias_std^2) per recommendation
      - bias_std == 0 → b_a ≡ 0 (nobias)
    """
    rng = random.Random(seed)

    if output_dir is None:
        output_dir = os.path.join(OUT_PATH, "user_preferences")

    os.makedirs(output_dir, exist_ok=True)

    missions, recommendations, resources = load_json_files()
    _, _, bias_std = _env_params()

    # --- rec preferences (recommendation bias b_a) ---
    if bias_std > 0:
        rec_preferences = {rec_id: rng.gauss(0.0, bias_std) for rec_id in recommendations.keys()}
        # centre to zero mean per world (keeps global calibration)
        avg = sum(rec_preferences.values()) / len(rec_preferences)
        rec_preferences = {k: v - avg for k, v in rec_preferences.items()}
    else:
        # nobias regime: all b_a = 0
        rec_preferences = {rec_id: 0.0 for rec_id in recommendations.keys()}

    # --- res preferences (keep neutral for now) ---
    res_preferences = {res_id: 0.0 for res_id in resources.keys()}

    # --- int preferences (TRUE θ* in sim space, structured–moderate) ---
    int_preferences, theta = generate_int_preferences(seed=seed)

    # Persist CURRENT preferences (world-local)
    with open(os.path.join(output_dir, "rec_preferences.json"), "w") as f:
        json.dump(rec_preferences, f, indent=4)

    with open(os.path.join(output_dir, "res_preferences.json"), "w") as f:
        json.dump(res_preferences, f, indent=4)

    with open(os.path.join(output_dir, "int_theta.json"), "w") as f:
        json.dump(theta, f, indent=4)

    # Also store labeled dict for analysis:
    with open(os.path.join(output_dir, "int_preferences.json"), "w") as f:
        json.dump(int_preferences, f, indent=4)

    return int_preferences, theta


def generate_config(
    prefs,
    theta,
    mab_type: str,
    output_path: str | None,
    feature_profile: str | None = None,
):
    """
    Generates a fresh config.py with randomized PREFERENCES and matching θ*.

    Arguments
    ---------
    prefs : dict
        Labeled simulator preferences (from generate_int_preferences).
    theta : list[float]
        Ordered vector of the same prefs (typically for offline analysis).
    mab_type : {"Random","ContextBase","TwoStageBase","TwoStageFull","Oracle"}
        High-level bandit regime (matches paper naming).
    output_path : str or None
        Where to write config.py. If None, writes to OUT_PATH/config.py.
    feature_profile : {"small","full"} or None
        Which intervention feature profile the *learning bandit* uses.
        If None, chosen automatically from mab_type.
    """
    if output_path is None:
        CONFIG_PATH = os.path.join(OUT_PATH, "config.py")
    else:
        CONFIG_PATH = output_path

    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)

    # --- choose MAB algorithm types (paper → implementation) ---
    lts_extra = ""
    if mab_type == "Random":
        # No contextual learning; recommendation/resource bandits are uniform
        intervention_mab_type = "None"
        recommendation_mab_type = "RandomBandit"
        resource_mab_type = "RandomBandit"

    elif mab_type in ["ContextBase", "ContextFull"]:
        # 1-stage contextual bandit, base feature set; no rec-level bandit
        intervention_mab_type = "LogisticLaplaceTS"
        recommendation_mab_type = "RandomBandit"
        resource_mab_type = "BernoulliBetaTS"
        lts_extra = """
        "include_full_hessian": True,
        "n_steps": 1,
        "theta_prior_var": 0.25,
        "discount": 1.0,
        "w_clip": [1e-8, 0.25],
        """

    elif mab_type in ["TwoStageBase", "TwoStageFull"]:
        # 2-stage: contextual (base features) + rec-level bandit
        intervention_mab_type = "LogisticLaplaceTS"
        recommendation_mab_type = "BernoulliBetaTS"
        resource_mab_type = "BernoulliBetaTS"
        lts_extra = """
        "include_full_hessian": True,
        "n_steps": 1,
        "theta_prior_var": 0.25,
        "discount": 1.0,
        "w_clip": [1e-8, 0.25],
        """

    elif mab_type in ["HybridBase", "HybridBase_bvar008", "HybridFull", "HybridFull_bvar008"]:
        # 1-stage contextual with per-item bias
        intervention_mab_type = "LogisticLaplaceTS_ItemBias"
        recommendation_mab_type = "None"
        resource_mab_type = "BernoulliBetaTS"
        lts_extra = """
        "include_full_hessian": True,
        "n_steps": 1,
        "theta_prior_var": 0.25,
        "bias_prior_var": 0.08,
        "discount": 1.0,
        "w_clip": [1e-8, 0.25],
        """

    elif mab_type in ["HybridBase_bvar025", "HybridFull_bvar025"]:
        # 1-stage contextual with per-item bias
        intervention_mab_type = "LogisticLaplaceTS_ItemBias"
        recommendation_mab_type = "None"
        resource_mab_type = "BernoulliBetaTS"
        lts_extra = """
        "include_full_hessian": True,
        "n_steps": 1,
        "theta_prior_var": 0.25,
        "bias_prior_var": 0.25,
        "discount": 1.0,
        "w_clip": [1e-8, 0.25],
        """

    elif mab_type == "Oracle":
        # Oracle non-learning bandits that use (θ*, b)
        intervention_mab_type = "None"
        recommendation_mab_type = "RecommendationOptimalBandit"
        resource_mab_type = "ResourceOptimalBandit"

    else:
        raise ValueError(f"Unknown mab_type={mab_type!r}")

    # -------------- choose feature profile --------------

    default_profile_for_mab = {
        "Random": "small",
        "ContextBase": "small",
        "TwoStageBase": "small",
        "HybridBase": "small",
        "HybridBase_bvar025": "small",
        "ContextFull": "full",
        "TwoStageFull": "full",
        "HybridFull": "full",
        "HybridFull_bvar025": "full",
        "Oracle": "full",
    }

    if feature_profile is None:
        feature_profile = default_profile_for_mab.get(mab_type, "small")

    if feature_profile not in FEATURE_PROFILES:
        raise ValueError(f"Unknown feature_profile={feature_profile!r}. Available: {list(FEATURE_PROFILES.keys())}")

    bandit_flags, true_flags = FEATURE_PROFILES[feature_profile]

    # Core grid params (feedback density + bias)
    open_prob, rate_prob, _ = _env_params()

    # compose the new config.py content
    content = f"""from datetime import datetime, timedelta, timezone

# NOTE: auto-generated by config_generator.generate_config
INTERVENTION_FEATURE_PROFILE = {feature_profile!r}
MAB_TYPE = {mab_type!r}

# --- Time ---
ENTRANCE_TIMES = [datetime(2025, 9, 1, 9, 0, 0, tzinfo=timezone.utc)]
NUM_WEEKLY_USERS = {NUM_WEEKLY_USERS}

# --- Features ---
PERSONAL_DATA_FEATURES = ["gender", "userAge", "education", "recruitmentCenter"]
PERSONAL_DATA_CATEGORICAL_FEATURES = {{
    "gender": ["female", "male", "decline", "other"],
    "recruitmentCenter": ["IEO", "ICO", "UMFCD", "UNIPA"],
    "education": ["no-education", "primary", "secondary", "vocational", "university", "postgraduate", "other"],
}}
NUMERIC_FEATURES_MIN_MAX = {{"userAge": [45, 80]}}
PILLARS = ["smoking", "alcohol", "nutrition", "physical_activity", "emotional_wellbeing"]
PILLARS_WITH_COMPONENTS = {{"nutrition", "emotional_wellbeing"}}
INTERVENTION_TYPES = [
    "Education",
    "Enablement",
    "Environmental Restructuring",
    "Incentivisation",
    "Modelling",
    "Training",
    "Persuasion",
    "Restrictions",
]

# --- User Preferences ---
REWARD_TYPE = "thumbs"  # "float" or "thumbs"
LOGISTIC_STEEPNESS = 1

# Core grid: feedback density (global knobs the simulator should read)
OPEN_PROBABILITY = {open_prob}
RATE_PROBABILITY = {rate_prob}

# Recommendation bias is encoded in rec_preferences.json (b_a) per env
PREFERENCES = {prefs!r}
MISSION_SELECTION_MODE = "user_specific"  # "random", "fixed", "user_keep_pillar", "user_specific"

# --- MAB ---
RESOURCE_MAB_CONFIG = {{
    "type": "{resource_mab_type}",
}}
INTERVENTION_MAB_CONFIG = {{
    "type": "{intervention_mab_type}",{lts_extra}
}}
RECOMMENDATION_MAB_CONFIG = {{
    "type": "{recommendation_mab_type}",
}}

# --- Weekly caps ---
MAX_SAME_REC_SENT_PER_MISSION = 3
MAX_NUM_REC_PER_MISSION = 14
MIN_NUM_REC_PER_MISSION = 3
MAX_DISTINCT_REC_PER_MISSION = 6

CAP_SAME_REC = MAX_SAME_REC_SENT_PER_MISSION + 1
CAP_NUM_REC = MAX_NUM_REC_PER_MISSION + 1

MAX_BASE_HISTORY = 28  # must be >= max pressure_history_length (currently 28)

# --- Frequency featurization (keep all degree-1) ---
FREQUENCY_FEATURE_DEGREES = {{"MF": 1, "TF": 1, "IF": 1, "RF": 1}}

# ================================
# Feature flags (single logic)
# ================================
# The bandit sees INTERVENTION_MAB_FEATURES; the simulator uses
# TRUE_INTERVENTION_MAB_FEATURES in sim space when use_sim_space=True.

INTERVENTION_MAB_FEATURES = {bandit_flags!r}
TRUE_INTERVENTION_MAB_FEATURES = {true_flags!r}

RECOMMENDATION_MAB_FEATURES = {{"RF": False}}
"""

    with open(CONFIG_PATH, "w") as f:
        f.write(content)

    return CONFIG_PATH
