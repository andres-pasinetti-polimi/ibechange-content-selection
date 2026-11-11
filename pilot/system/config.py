# --- Time ---
USE_REAL_TIME = True


# --- Features ---
PERSONAL_DATA_FEATURES = ["gender", "userAge", "education", "recruitmentCenter"]
PERSONAL_DATA_CATEGORICAL_FEATURES = {
    "gender": ["female", "male", "decline", "other"],
    "recruitmentCenter": ["IEO", "ICO", "UMFCD", "UNIPA"],
    "education": ["no-education", "primary", "secondary", "vocational", "university", "postgraduate", "other"],
}
CATEGORICAL_TO_NUMERIC = ["education", "gender"]
CATEGORICAL_TO_NUMERIC_EXPLICIT = {
    "gender": {"female": 0, "decline": 0.5, "other": 0.5, "male": 1},
}
NUMERIC_FEATURES_MIN_MAX = {"userAge": [45, 80]}

PILLARS = ["smoking", "alcohol", "nutrition", "physical_activity", "emotional_wellbeing"]
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

# no INTERVENTION since there are None
# education we want it to be an integer feature --> "other" is middle value
LEAVE_OUT_VARS = {
    "recruitmentCenter": ["IEO"],
    "education": ["other"],
    "pillar": ["smoking"],  # BECAUSE IN PILOT ONE PILLAR AT A TIME
}


# --- User Preferences ---
REWARD_TYPE = "thumbs"  # "float" or "thumbs"

# --- MAB ---
RESOURCE_MAB_CONFIG = {
    "type": "BernoulliBetaTS",
}
INTERVENTION_MAB_CONFIG = {
    "type": "LogisticLaplaceTS",
}
RECOMMENDATION_MAB_CONFIG = {
    "type": "BernoulliBetaTS",
}


# --- MAB / Frequency Settings ---
MAX_REC_REPETITIONS = 7
MAX_NUM_REC_PER_MISSION = 3 * 7
MIN_NUM_REC_PER_MISSION = 3
MAX_DISTINCT_REC_PER_MISSION = 5


# --- Features / Encoding ---
FREQUENCY_FEATURE_DEGREES = {"MF": 1, "TF": 1, "IF": 1, "RF": 1}
INTERVENTION_MAB_FEATURES = {
    "D": True,
    "H": True,
    "ND": True,
    "P": True,
    "MF": True,
    "TF": True,
    "I": True,
    "IF": True,
    "RF": True,
    "D_H": False,
    "D_P": False,
    "D_I": False,
    "D_IF": False,
    "D_MF": False,
    "D_TF": False,
    "P_I": False,
    "P_MF": False,
    "P_TF": False,
    "P_IF": False,
    "I_IF": False,
}
RECOMMENDATION_MAB_FEATURES = {"RF": False}
