"""Constants of the design, each with the place it comes from.

The daily decision is the composition (R, Q) of the day, R recommendations and Q reminders
(rct/design/rct_design_record_2026-10-27.md, Sections 1 and 12). Values marked PROVISIONAL are not
fixed by the design record and are set here so that the service can run; each must be fixed before
the trial, and the value in force is written into every decision row (parameters column), so that a
later change is visible in the logs.

STUDY selects the length of the programme and the state entries logged with each decision: the
trial adds the Mission's class and frequency. The model has no per-participant random effect on the
treatment terms in either study (design record, Section 1.2).
"""

import os

STUDY = os.getenv("CS_STUDY", "usability")  # "usability" or "trial"
assert STUDY in ("usability", "trial"), STUDY

# Section "Study constants".
WINDOW_DAYS = 7               # W, the Mission window
REPLACEMENT_WINDOW_HOURS = 24  # rho, a Mission can be replaced only within 24 hours of selection
PROGRAMME_DAYS = 84 if STUDY == "trial" else 30  # T

# An episode neither completed, replaced nor closed expires at t0 + W + delta
# (Section "Fixed rules"). PROVISIONAL: the paper does not give delta.
EXPIRY_MARGIN_DAYS = 7

PRIOR_SCALE_TAU = 0.05  # prior standard deviation of each coefficient of the treatment and lag terms

# Working model (implementation_plan_composition_2026-11-04.md, Phase 2). PROVISIONAL values.
NOISE_VARIANCE = 0.25               # sigma^2; 0.25 is the largest variance of a binary outcome
INTERCEPT_VARIANCE = 0.04           # sigma_alpha^2 of the per-participant level alpha_i (sd 0.2)
BASELINE_PRIOR_VARIANCE = 1.0       # prior variance of the baseline coefficients, kept wide

# The admissible compositions (design record, Section 1.1): 2 <= R + Q <= 4, R <= 2, R <= 1 in
# Emotional Wellbeing, Q <= 4.
DAILY_MINIMUM = 2
DAILY_MAXIMUM = 4
MAX_RECOMMENDATIONS = 2
MAX_RECOMMENDATIONS_BY_PILLAR = {"Emotional_Wellbeing": 1}
MAX_REMINDERS = 4

# The policy (design record, Sections 1.2 and 12). "uniform": every admissible composition has the
# same probability. "conservative": a share EXPLORE_SHARE is spread uniformly over the admissible
# compositions and the rest goes to DEFAULT_COMPOSITION, unless some compositions have a posterior
# probability above BEAT_THRESHOLD of a higher value than the default; then the rest goes to Thompson
# sampling among those compositions and the default. The value of a composition is its effect on the
# day plus LAG_DAYS times its effect on one later day. EXPLORE_SHARE, BEAT_THRESHOLD and
# DEFAULT_COMPOSITION are PROVISIONAL, to be discussed with the supervisor.
POLICY = os.getenv("CS_POLICY", "conservative")
assert POLICY in ("uniform", "conservative"), POLICY
EXPLORE_SHARE = 0.3
BEAT_THRESHOLD = 0.9
DEFAULT_COMPOSITION = (1, 1)
THOMPSON_DRAWS = 500
# The lag terms of the model: the sums of R and of Q over the LAG_DAYS preceding days, each day
# centred on its expectation under the probabilities of that day.
LAG_DAYS = 3

WEEKLY_CAP = 28        # the daily maximum times seven; proposed to ICO, to be set by them

# OPEN DECISION: the daily outcome the model learns on (rct/design/composition_model_2026-10-15.md,
# Section 8).
#   "mark":                the mark of the day, entered by the end of the episode.
#   "mark_minus_backfill": the mark of the day, less 1 if a mark for the day before was
#                          entered on that day after the decision of that day. A delivery cannot
#                          change the behaviour of the day before, so this removes the part of the
#                          effect that acts only on recording, if that part is the same for both
#                          days. Marks entered before the decision are not counted (8 November
#                          2026; history.backfilled_days explains why the cut is the decision time
#                          and not the time of the first delivery).
DAILY_OUTCOME = "mark"

# Section "Content rule".
# A reminder text sent on a day is not sent again that day nor the next, so that at least one day
# passes before it returns (decided 21 October 2026; it was four days, set for three reminders a day).
REMINDER_NO_REPEAT_DAYS = 2

# Recommendations (content.choose_recommendation): among codes not yet sent, the probability of the
# class the clinical team prioritises. PROVISIONAL, the clinical team's to set
# (rct_design_record_2026-10-27.md, Section 2.1); a value below 1 keeps the other class in the logs,
# so that matched against unmatched content can be compared after the trial.
BARRIER_PRIORITY_PROBABILITY = 0.7   # codes whose COM-B sub-domain is a barrier flagged in Step 3
HAPA_PRIORITY_PROBABILITY = 0.7      # reflective motivation, pre-intenders and intenders
HAPA_PILLARS = ("Smoking", "Alcohol")

# The value of notification_rated.rating that means "disliked". With the exit button removed,
# a rating is 1 (negative) or 2 (positive): comment of EUT and reply, 14 October 2026, in
# iBeChange-modules-architecture.docx.
DISLIKE_RATING = 1

# OPEN DECISION: when deliveries start for a newly selected Mission.
#   "selection_day": on the day of the selection itself, at the first run after it,
#                    up to LAST_DECISION_HOUR; a Mission selected later starts the next day.
#   "next_day":      from the morning of the day after the selection; the day of the
#                    selection carries nothing and its row says so.
# Still to be decided; "selection_day" is a provisional default. The value in force is
# written into every decision row.
FIRST_DELIVERY_DAY = os.getenv("CS_FIRST_DELIVERY_DAY", "selection_day")
assert FIRST_DELIVERY_DAY in ("selection_day", "next_day"), FIRST_DELIVERY_DAY

# Latest hour of a run that decides a day (used with "selection_day"). PROVISIONAL.
LAST_DECISION_HOUR = 20

# Time zones. The EUT messages without an offset carry UTC (measured on INT, 20 October 2026:
# their created_at equals the time of reception in UTC); the notification messages carry the
# participant's offset. The service stores every time in UTC and reads each participant's
# history in the participant's local time, from the recruitment centre, so that the decision
# day, the last decision hour and the day a mark refers to are the participant's.
CENTRE_TIMEZONE = {"ICO": "Europe/Madrid", "IEO": "Europe/Rome", "UNIPA": "Europe/Rome",
                   "UMFCD": "Europe/Bucharest"}
DEFAULT_TIMEZONE = os.getenv("CS_DEFAULT_TIMEZONE", "Europe/Madrid")  # unknown or missing centre

# Share of participants randomised to the fixed-probability group within the arm
# (proposal_rct_module_architecture_2026-09-25.md, Section 4.2). The evaluation design
# is an open decision: 0 runs every participant on the learned rule, 0.5 is the 1:1
# split of the proposal. The fixed group draws its composition from the starting
# probabilities of the policy (the conservative rule with a flat posterior, or the uniform
# rule), and its days do not enter the posterior.
FIXED_GROUP_SHARE = float(os.getenv("CS_FIXED_GROUP_SHARE", "0"))


def utc_now():
    """The current time in UTC, naive, the clock of every time the service stores."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).replace(tzinfo=None)


def parameters():
    """The values in force, written into every decision row."""
    return {
        "study": STUDY,
        "window_days": WINDOW_DAYS,
        "expiry_margin_days": EXPIRY_MARGIN_DAYS,
        "tau": PRIOR_SCALE_TAU,
        "noise_variance": NOISE_VARIANCE,
        "intercept_variance": INTERCEPT_VARIANCE,
        "baseline_prior_variance": BASELINE_PRIOR_VARIANCE,
        "daily_minimum": DAILY_MINIMUM,
        "daily_maximum": DAILY_MAXIMUM,
        "max_recommendations": MAX_RECOMMENDATIONS,
        "max_recommendations_by_pillar": MAX_RECOMMENDATIONS_BY_PILLAR,
        "max_reminders": MAX_REMINDERS,
        "policy": POLICY,
        "explore_share": EXPLORE_SHARE,
        "beat_threshold": BEAT_THRESHOLD,
        "default_composition": list(DEFAULT_COMPOSITION),
        "thompson_draws": THOMPSON_DRAWS,
        "lag_days": LAG_DAYS,
        "weekly_cap": WEEKLY_CAP,
        "daily_outcome": DAILY_OUTCOME,
        "reminder_no_repeat_days": REMINDER_NO_REPEAT_DAYS,
        "barrier_priority_probability": BARRIER_PRIORITY_PROBABILITY,
        "hapa_priority_probability": HAPA_PRIORITY_PROBABILITY,
        "dislike_rating": DISLIKE_RATING,
        "first_delivery_day": FIRST_DELIVERY_DAY,
        "last_decision_hour": LAST_DECISION_HOUR,
        "fixed_group_share": FIXED_GROUP_SHARE,
    }
