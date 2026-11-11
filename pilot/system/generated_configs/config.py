from datetime import datetime, timedelta, timezone

    
# --- Time ---
ENTRANCE_TIMES = [datetime(2025, 9, 1, 9, 0, 0, tzinfo=timezone.utc)] # + timedelta(weeks=i) for i in range(3)]
NUM_WEEKLY_USERS = 90

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



# --- User Preferences ---
REWARD_TYPE = "thumbs"  # "float" or "thumbs"
LOGISTIC_STEEPNESS = 1

OPEN_PROBABILITY = 1
RATE_PROBABILITY = 1

REC_PREFERENCE_RANGE = (-0.0, 0.0)
PREFERENCES = {'bias': -0.14589063171112326, 'male': -0.16713736247239008, 'decline': -0.0966611396653052, 'other': 0.041498123177556745, 'userAge': 0.2717103814276176, 'education': -0.1471154354661276, 'education_other': 0.34688367691363453, 'ICO': -0.3878691185147869, 'UMFCD': -0.22951248593539886, 'UNIPA': -0.4003512957581743, 'hhs_current': -0.38731522437663, 'hhs_total': 0.2789830633248863, 'emotional_wellbeing': 0.2272893273241281, 'num_intervention_days': -0.3151540704207608, 'pillar_alcohol': -0.310830477125259, 'pillar_nutrition': -0.08334466315694777, 'pillar_physical_activity': 0.24331745051730236, 'pillar_emotional_wellbeing': 0.31574817734517113, 'mission_frequency': 0.24870041853544256, 'rec_pressure_short': 0.0919162974349711, 'rec_pressure_long': -0.3535288433766184, 'it_pressure_short': -0.10158057661208786, 'it_pressure_long': -0.3063615519946822, 'TF_sched': 0.027601213532174373, 'IT_Enablement': 0.0683682052708332, 'IT_Environmental Restructuring': -0.29792326786690504, 'IT_Incentivisation': -0.24984940611124173, 'IT_Modelling': 0.2816629431621529, 'IT_Training': -0.4699125450678382, 'IT_Persuasion': 0.3031564611556872, 'IT_Restrictions': 0.3912001133503552, 'num_int_types': 0.44932279597603964, 'prompted': -0.1168536967341044, 'MF_x_TF_sched': 0.05260640312853315, 'AGE_x_IT_Enablement': 0.08305669061545284, 'AGE_x_IT_Environmental Restructuring': 0.1336419023410741, 'AGE_x_IT_Incentivisation': 0.4769766742667433, 'AGE_x_IT_Modelling': 0.18663013508128168, 'AGE_x_IT_Training': -0.20059647488816068, 'AGE_x_IT_Persuasion': 0.3600107928189751, 'AGE_x_IT_Restrictions': -0.015927828623549245, 'AGE_x_REC_PRESS_LONG': 0.10136397324864044, 'MF_x_REC_PRESS_SHORT': 0.22683438386052823, 'HC_x_IT_Enablement': -0.49762710848836056, 'HC_x_IT_Environmental Restructuring': 0.270456483632056, 'HC_x_IT_Incentivisation': 0.16193756635335077, 'HC_x_IT_Modelling': -0.00812745710381968, 'HC_x_IT_Training': 0.02363969106242103, 'HC_x_IT_Persuasion': -0.0394665509543789, 'HC_x_IT_Restrictions': -0.3065635801691421, 'ND_x_PR': 0.029547916441340027}
MISSION_SELECTION_MODE = "user_specific"  # "random", "fixed", "user_keep_pillar", "user_specific"


# --- MAB ---
RESOURCE_MAB_CONFIG = { 
    "type": "RandomBandit", 
}
INTERVENTION_MAB_CONFIG = {
    "type": "None",
}

RECOMMENDATION_MAB_CONFIG = { 
    "type": "RandomBandit", 
}



# --- Weekly caps ---
MAX_SAME_REC_SENT_PER_MISSION = 3
MAX_NUM_REC_PER_MISSION = 10
MIN_NUM_REC_PER_MISSION = 3
MAX_DISTINCT_REC_PER_MISSION = 4

# caps are increased by one to allow EOM overflow to reach 1.0
CAP_SAME_REC = MAX_SAME_REC_SENT_PER_MISSION + 1
CAP_NUM_REC = MAX_NUM_REC_PER_MISSION + 1

MAX_BASE_HISTORY = 14  # must be >= max pressure_history_length (currently 14)

# --- Frequency featurization (keep all degree-1) ---
FREQUENCY_FEATURE_DEGREES = {"MF": 1, "TF": 1, "IF": 1, "RF": 1}

# ================================
# Feature flags (single logic)
# ================================
# Base blocks (main effects)
INTERVENTION_MAB_FEATURES = {
    # === Core user-level features ===
    "D": True,  # Demographics (gender, age, education, recruitmentCenter)
    "H": False,  # Vector of all 5 pillar scores (broad health profile)
    "HC": True,  # Current pillar score (specific to this mission's domain)
    "HT": True,  # Total habit score (sum of all pillars; overall wellbeing)
    "EW": True,  # Emotional wellbeing pillar score
    "ND": True,  # Days since start of intervention (normalized)
    "P": True,  # Pillar one-hot encoding (leave-one-out; mission domain)

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
    "PR": True,  # Prompted flag (True for end-of-mission or forced feedback)    
    "ER": False,  # Past-week voluntary engagement rate — currently off (unstable early in pilot)
    "MS": False,  # Previous mission score (average mission-level success; not yet used)

    # === Interactions (on) ===
    "AGE_x_IT": False,  # Age-centered × intervention style → age-specific style preference
    "AGE_x_REC_PRESS_LONG": False,  # Age-centered × repetition pressure → tolerance for repetition by age
    "MF_x_TF_sched": False,  # Mission intensity × plan fullness → diminishing returns (not implemented)
    "MF_x_REC_PRESS_SHORT": False,  # Mission frequency × recommendation pressure → overload penalty for intense plans
    "ND_x_PR": False,  # Time-in-intervention × prompted flag → fatigue to prompts over weeks
    "HC_x_IT": False,  # Pillar severity × intervention style → style preference by problem severity

    # --- Optional / experimental (currently off) ---
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


TRUE_INTERVENTION_MAB_FEATURES = {
    **INTERVENTION_MAB_FEATURES,

    # === Interactions (on) ===
    "AGE_x_IT": True,  # Age-centered × intervention style → age-specific style preference
    "AGE_x_REC_PRESS_LONG": True,  # Age-centered × repetition pressure → tolerance for repetition by age
    "MF_x_TF_sched": True,  # Mission intensity × plan fullness → diminishing returns (not implemented)
    "MF_x_REC_PRESS_SHORT": True,  # Mission frequency × recommendation pressure → overload penalty for intense plans
    "ND_x_PR": True,  # Time-in-intervention × prompted flag → fatigue to prompts over weeks
    "HC_x_IT": True,  # Pillar severity × intervention style → style preference by problem severity

}


RECOMMENDATION_MAB_FEATURES = {"RF": False}
