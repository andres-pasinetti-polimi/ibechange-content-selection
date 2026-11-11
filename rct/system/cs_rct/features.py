"""The state s and the baseline features psi of one decision.

Table "Feature blocks" of rct/paper/main.tex. The state has eight entries in the usability
study and is extended by the Mission's class and frequency in the trial. Since the composition
design (implementation_plan_composition_2026-11-04.md, Phase 2) s no longer multiplies the
treatment terms of the model: it is logged with each decision (column state) for the analyses and
for moderators still to be decided. The count of the service's deliveries in the three preceding
days is kept in the context and not in s or psi, because the lag terms of the model replace it. The
baseline takes s without its intercept and without the four entries built from the marks
(MARK_NAMES), the share of days the Mission implies, the class and frequency indicators, the
pillar of the open Mission (four indicators, Smoking the reference), the
sociodemographic block A (age, sex, education, recruitment centre), and the latest
HHS score of the pillar of the open Mission with an indicator of its absence. The HHS is repeated
over the programme, so it varies within participant and the per-participant level
alpha_i does not absorb it. The HAPA and COM-B entries of block C come from the
Step-3 messages, which carry no user_uuid, so they are absent.

The entries in MARK_NAMES are left out of the baseline since 11 November 2026
(rct/design/lag_terms_decision_2026-11-10.md): the deliveries of the three preceding days can change
them, so in the row of day t they are outcomes of the compositions that enter the lag terms, and
conditioning on them biases the lag coefficients. They stay in s. The versions fitted before that
date used the baseline with them (baseline_names(with_marks=True)).

Every entry is computed from what the service had received at the decision.
Numeric entries are scaled to be of order one, since the prior on theta has the
same scale tau on every coefficient.
"""

from datetime import timedelta

from cs_rct import design
from cs_rct.catalogue import DAILY, NO_COUNT
from cs_rct.history import (earlier_marked_share, latest_health_score, marked_days, scheduled_slot,
                            sent_count, start_day)

USABILITY_NAMES = [
    "intercept",
    "deficit",                   # q / l, 0 on a Mission that states no count
    "scheduled_today",
    "marked_yesterday",          # as entered by the time of the decision
    "episode_marked_share",      # share of the episode's earlier days marked so far
    "participant_marked_share",  # over earlier episodes, 0 when there is none
    "episode_day",               # days elapsed in the episode, over W
    "programme_day",             # days since enrolment, over 84
]
TRIAL_NAMES = ["class_a", "frequency_daily", "frequency_no_count"]
# Entries of s built from the marks, not in the baseline (see the module docstring).
MARK_NAMES = ["deficit", "marked_yesterday", "episode_marked_share", "participant_marked_share"]


def state_names(study=None):
    study = study or design.STUDY
    return USABILITY_NAMES + TRIAL_NAMES if study == "trial" else list(USABILITY_NAMES)


def baseline_names(study=None, with_marks=False):
    """The names of psi; with_marks=True gives the baseline of the versions fitted before
    11 November 2026, which also had MARK_NAMES."""
    state = [name for name in state_names(study)[1:] if with_marks or name not in MARK_NAMES]
    return (["intercept"] + state + [name for name in TRIAL_NAMES if name not in state]
            + ["expected_share"] + PILLAR_NAMES + SOCIODEMOGRAPHIC_NAMES + ["hhs_pillar", "hhs_missing"])


PILLAR_NAMES = ["pillar_alcohol", "pillar_nutrition", "pillar_physical_activity", "pillar_emotional_wellbeing"]
SOCIODEMOGRAPHIC_NAMES = ["age", "sex_male", "education", "centre_ieo", "centre_umfcd"]

# Ordinal education, from the values of the pilot's new_user messages (users.csv of the
# 20 August 2026 snapshot), over 6. An unknown value is read as secondary.
EDUCATION_LEVEL = {"no-education": 0, "primary": 1, "secondary": 2, "vocational": 3,
                   "university-no-degree": 3.5, "university": 4, "master-degree": 5, "postgraduate": 6}
EDUCATION_SCALE = 6

# pillar_id of the HHS messages by the pillar of the Mission catalogue.
PILLAR_ID = {"Smoking": "SO", "Alcohol": "AO", "Nutrition": "NO", "Physical_activity": "PO",
             "Emotional_Wellbeing": "EO"}
# The range of the pillar's HHS score is not stated in the EUT document (its example
# reads 34.9). PROVISIONAL scale.
HHS_SCALE = 100

# Scales of the counts.
RECENT_DAYS = 3
PROGRAMME_SCALE = 84


def compute(participant, episode, mission, day, now, study=None, previous=()):
    """Return (s, psi, context): two lists of floats and a dict of the raw quantities, for logging.

    previous: the deliveries the service handed out to the participant on earlier days (store.py),
    from which planned_last_3_days is counted."""
    t0 = start_day(episode)
    episode_day = (day - t0).days
    days_left = design.WINDOW_DAYS - episode_day  # l, today included
    marked = marked_days(participant, episode, now)
    marked_before_today = {d for d in marked if t0 <= d < day}

    if mission is not None and mission.states_count:
        # q counts every mark of the episode received so far, whatever day it names.
        still_to_mark = max(mission.count - len(marked), 0)
        deficit = still_to_mark / days_left if days_left > 0 else 0.0
    else:
        still_to_mark, deficit = None, 0.0

    slot = scheduled_slot(participant, episode, day, now)
    yesterday = day - timedelta(days=1)
    earlier_share = earlier_marked_share(participant, episode, now)
    # planned_last_3_days counts the deliveries of the service's own plans in the three preceding
    # days, less those a revision withdrew before they were sent, and not the notifications EUT
    # records as sent: the plans are known exactly at every decision, whereas the sent events can
    # arrive late or not at all (none arrived for the test sends of 29 October 2026), and a count of
    # delivered notifications mixes the randomised draws with delivery failures that need not be
    # random (rct_design_record_2026-10-27.md, Section 1.2; debate_2026-10-26/D_sceptic.md, 1.4).
    # The count recorded by EUT is kept in the context. EUT produces notification_sent when it
    # receives a plan, not when the phone receives the notification (Anna Domenech, 2 November 2026),
    # so it counts the deliveries EUT accepted; only notification_opened follows a delivery.
    recent = sum(1 for d in previous
                 if day - timedelta(days=RECENT_DAYS) <= d["decision_day"] <= yesterday
                 and d.get("content_id") is not None and not d.get("cancelled"))
    recorded = sent_count(participant, day - timedelta(days=RECENT_DAYS), yesterday, now)
    enrolment = participant.enrolment_date or t0
    hhs = latest_health_score(participant, PILLAR_ID.get(mission.pillar), now) if mission is not None else None
    programme_day = (day - enrolment).days

    values = {
        "intercept": 1.0,
        "deficit": deficit,
        "scheduled_today": float(slot is not None),
        "marked_yesterday": float(yesterday in marked and yesterday >= t0),
        "episode_marked_share": len(marked_before_today) / episode_day if episode_day > 0 else 0.0,
        "participant_marked_share": earlier_share if earlier_share is not None else 0.0,
        "episode_day": episode_day / design.WINDOW_DAYS,
        "programme_day": programme_day / PROGRAMME_SCALE,
        "class_a": float(mission is not None and mission.mission_class == "A"),
        "frequency_daily": float(mission is not None and mission.frequency == DAILY),
        "frequency_no_count": float(mission is not None and mission.frequency in NO_COUNT),
        "expected_share": mission.expected_share if mission is not None else 0.0,
        "pillar_alcohol": float(mission is not None and mission.pillar == "Alcohol"),
        "pillar_nutrition": float(mission is not None and mission.pillar == "Nutrition"),
        "pillar_physical_activity": float(mission is not None and mission.pillar == "Physical_activity"),
        "pillar_emotional_wellbeing": float(mission is not None and mission.pillar == "Emotional_Wellbeing"),
        "age": (participant.age - 60) / 10 if participant.age is not None else 0.0,
        "sex_male": float(participant.sex == "male"),
        "education": EDUCATION_LEVEL.get(participant.education, 2) / EDUCATION_SCALE,
        # Centres of the pilot; ICO and any other centre are the reference.
        "centre_ieo": float(participant.recruitment_center == "IEO"),
        "centre_umfcd": float(participant.recruitment_center == "UMFCD"),
        "hhs_pillar": hhs / HHS_SCALE if hhs is not None else 0.0,
        "hhs_missing": float(hhs is None),
    }
    s = [values[name] for name in state_names(study)]
    psi = [values[name] for name in baseline_names(study)]
    context = {
        "mission_process_id": episode["mission_process_id"],
        "mission_id": episode["mission_id"],
        "episode_start_day": t0.isoformat(),
        "episode_day": episode_day,
        "marks_received": len(marked),
        "still_to_mark": still_to_mark,
        "days_left": days_left,
        "scheduled_at": slot.isoformat() if slot is not None else None,
        "planned_last_3_days": recent,
        "sent_last_3_days": recorded,  # notification_sent events of EUT received by the decision
        "earlier_marked_share": earlier_share,
        "programme_day": programme_day,
        "enrolment_date": enrolment.isoformat(),
        "age": participant.age,
        "recruitment_center": participant.recruitment_center,
        "hhs_pillar": hhs,
    }
    return s, psi, context
