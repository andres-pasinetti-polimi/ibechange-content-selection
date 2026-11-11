# --- Features ---
PERSONAL_DATA_FEATURES = ["gender", "userAge", "education", "recruitmentCenter"]
PERSONAL_DATA_CATEGORICAL_FEATURES = {
    "gender": ["female", "male", "decline", "other"],
    "recruitmentCenter": ["IEO", "ICO", "UMFCD", "UNIPA"],
    "education": ["no-education", "primary", "secondary", "vocational", "university", "postgraduate", "other"],
}
NUMERIC_FEATURES_MIN_MAX = {"userAge": [45, 80]}
PILLARS = ["smoking", "alcohol", "nutrition", "physical_activity", "emotional_wellbeing"]
PILLARS_WITH_COMPONENTS = {"nutrition", "emotional_wellbeing"}
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


# --- Rewards / bandits ---
REWARD_TYPE = "thumbs"  # "thumbs" or "float"

RESOURCE_MAB_CONFIG = {"type": "BernoulliBetaTS"}
INTERVENTION_MAB_CONFIG = {
    "type": "LogisticLaplaceTS_ItemBias",
    # ---- optional tuning knobs (all optional) ----
    "include_full_hessian": True,  # True -> full-Hessian Laplace (slower, more accurate)
    "n_steps": 1,  # Newton steps per update
    "theta_prior_var": 0.25,  # prior variance σ²_θ
    "bias_prior_var": 0.25,
    "discount": 1.0,  # 0<λ≤1; λ<1 adds exponential forgetting
    "w_clip": [1e-8, 0.25],  # clip for logistic curvature σ(1−σ)
}

RECOMMENDATION_MAB_CONFIG = {"type": "None"}

# --- Weekly caps ---
MAX_SAME_REC_SENT_PER_MISSION = 3
MAX_NUM_REC_PER_MISSION = 14
MIN_NUM_REC_PER_MISSION = 3
MAX_DISTINCT_REC_PER_MISSION = 6

# caps are increased by one to allow EOM overflow to reach 1.0
CAP_SAME_REC = MAX_SAME_REC_SENT_PER_MISSION + 1
CAP_NUM_REC = MAX_NUM_REC_PER_MISSION + 1

MAX_BASE_HISTORY = 28  # must be >= max pressure_history_length (currently 14)

# --- Frequency featurization (keep all degree-1) ---
FREQUENCY_FEATURE_DEGREES = {"MF": 1, "TF": 1, "IF": 1, "RF": 1}

# ================================
# Feature flags (single logic)
# ================================
# Base blocks (main effects)
INTERVENTION_MAB_FEATURES = {
    # === Core user-level features ===
    "D": True,  # Demographics (gender, age, education, recruitmentCenter)
    "H": True,  # Vector of all 5 pillar scores (broad health profile)
    "P": True,  # Pillar one-hot encoding (leave-one-out; mission domain)
    "ND": True,  # Days since start of intervention (normalized)
    # === Mission-level and scheduling context ===
    "MF": True,  # Mission frequency (intended weekly intensity)
    "REC_PRESS_SHORT": True,  # same-item recent (short horizon)
    "REC_PRESS_LONG": True,  # same-item cumulative (long horizon)
    "IT_PRESS_SHORT": True,  # style overlap recent (short horizon)
    "IT_PRESS_LONG": True,  # style overlap cumulative (long horizon)
    "TF_sched": True,  # Total frequency already scheduled (how full the plan is)
    "IF_past": False,  # Intervention-type frequency past week — disabled
    "IF_sched": False,  # Intervention-type frequency already scheduled — disabled
    "RF_past": False,  # This-item repetition frequency past week — disabled
    "TF_past": False,  # Total frequency past week (all recs; overall load) — disabled for simplicity
    "RF_sched": False,  # This-item repetition frequency in schedule — disabled
    # === Intervention-style descriptors ===
    "IT": True,  # Intervention type mixture vector (8-dim; normalized BCT taxonomy)
    "NIT": True,  # Number of distinct intervention types active in the plan (diversity metric)
    # === Behavioral state ===
    "ER": False,  # Past-week voluntary engagement rate — currently off (unstable early in pilot)
    "PR": True,  # Prompted flag (True for end-of-mission or forced feedback)
    "MS": False,  # Previous mission score (average mission-level success; not yet used)
    # === Interactions (Base) ===
    "ND_x_PR": True,  # Time-in-intervention × prompted flag → fatigue to prompts over weeks
    # === Interactions (Full) ===
    "AGE_x_IT": False,  # Age-centered × intervention style → age-specific style preference
    "EW_x_IT": False,  # 
    # --- Optional (currently off) ---
    "HC": False,  # Current pillar score (specific to this mission's domain)
    "HT": False,  # Total habit score (sum of all pillars; overall wellbeing)
    "EW": False,  # Emotional wellbeing pillar score    
    "HC_x_IT": False,  # Pillar severity × intervention style → style preference by problem severity
    "AGE_x_REC_PRESS_LONG": False,  # Age-centered × repetition pressure → tolerance for repetition by age
    "MF_x_REC_PRESS_SHORT": False,  # Mission frequency × recommendation pressure → overload penalty for intense plans
    "MF_x_TF_sched": False,  # Mission intensity × plan fullness → diminishing returns (not implemented)    
    "MF_x_IT_PRESS_RES": False,  # Mission frequency × style pressure → overload penalty for intense plans
    "MF_x_RF_sched": False,  # Mission intensity × same-item repetition in plan
    "MF_x_IF_sched": False,  # Mission intensity × same-style repetition in plan
    "AGE_x_RF_sched": False,  # Age × same-item repetition → spam tolerance by age (alternative to REC_PRESS)
    "HC_x_RF_sched": False,  # Pillar severity × same-item repetition → need-tolerance coupling (not implemented)
    "NIT_x_IF_sched": False,  # Diversity × planned style repetition → balance novelty vs overload (not implemented)
    # --- Additional interesting ideas (NOT IMPLEMENTED) ---
    "ER_x_IT": False,  # Engagement rate × style → tailor style to motivation (requires ER main effect)
    "HC_x_REC_PRESS": False,  # Pillar severity × repetition pressure → tolerance for nagging when struggling
    "ND_x_IT_PRESS": False,  # Time-in-intervention × style pressure → late-stage burnout sensitivity
    "AGE_x_IT_PRESS": False,  # Age × style pressure → age-specific fatigue to similar-style recs
    "PR_x_P": False,  # Prompted × pillar → some pillars’ prompts may be less intrusive
}
TRUE_INTERVENTION_MAB_FEATURES = INTERVENTION_MAB_FEATURES
RECOMMENDATION_MAB_FEATURES = {"RF": False}
