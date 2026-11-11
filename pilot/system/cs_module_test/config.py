from datetime import datetime, timedelta


# --- Time ---
USE_REAL_TIME = True
ENTRANCE_TIMES = [datetime(2025, 5, 19, 9, 0, 0) + timedelta(weeks=i) for i in range(9)]


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
LOGISTIC_STEEPNESS = 1
OPEN_RATE_PROBABILITY = 0.3
REC_PREFERENCE_RANGE = (-0.2, 0.2)
PREFERENCES = {
    "bias": -0.8015780238038086,
    "AGE": 0.5476374649428868,
    "SEX_male": 0.46855868331423167,
    "COUNTRY_ITA": -0.9385983080961877,
    "COUNTRY_ROM": -0.10656280173232702,
    "FORMAL_EDUCATION_LEVEL": 0.3728362085971162,
    "HHS_smoking": -0.9397315308954601,
    "HHS_alcohol": 0.8385647068032274,
    "HHS_nutrition": 0.9244849730208384,
    "HHS_physical activity": 0.44508554417768,
    "HHS_emotional wellbeing": -0.8429229206963924,
    "ND1": -0.9296705341236443,
    "PILLAR_smoking": -0.2814933703575262,
    "PILLAR_alcohol": -0.9412449844860271,
    "PILLAR_nutrition": -0.30424454543132096,
    "PILLAR_physical activity": -0.980071517374068,
    "MF1": 0.9486470256819357,
    "TF1": -0.1809933009311373,
    "INTERVENTION_education": -0.8589647770436253,
    "INTERVENTION_persuasion": 0.7868701836957206,
    "INTERVENTION_incentivisation": -0.5840439199919687,
    "INTERVENTION_training": -0.5904184034613,
    "INTERVENTION_environmental restructuring": 0.3475182910576682,
    "INTERVENTION_enablement": 0.8765245363250962,
    "IF1": -0.8768118787707626,
    "RF1": -0.9928154327477295,
}
MISSION_SELECTION_MODE = "random"  # "random" or "fixed"


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
