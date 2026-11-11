"""Shared data pipeline + utilities for the iBeChange pilot analysis compendium.

Every analysis script imports from here so that data loading, the
scheduled-vs-sent matching pipeline, mission-episode reconstruction and the
statistical helpers are defined exactly once (they reproduce the pipelines
validated interactively on the 2026_07_06 snapshot).
"""
from __future__ import annotations
import json
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Paths & snapshot selection
# ----------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent
PILOT_ROOT = BASE.parent.parent               # pilot/ (results_analysis/ now holds only code)
DATA_ROOT = PILOT_ROOT / "data"
SNAPSHOT = DATA_ROOT / "cs_snapshots" / "2026_08_20_11_34"    # main snapshot (update at DB freeze)
BASELINE = DATA_ROOT / "cs_snapshots" / "2026_06_23_13_40"    # old-prod baseline (plan->mission map)
EXPORT = DATA_ROOT / "eut_exports" / "complete"                # EUT's final partner export
FIG_DIR = BASE / "paper" / "figures"
RES_DIR = BASE / "results"
NUMBERS = RES_DIR / "numbers.json"

APP_RELEASE = pd.Timestamp("2026-06-17T13:56:00Z")
PEAK_HOURS = [8, 9, 13, 14]

# ----------------------------------------------------------------------------
# Plot style (dataviz reference palette, light surface, print-oriented)
# ----------------------------------------------------------------------------
C = {  # categorical slots, fixed order
    "blue": "#2a78d6", "aqua": "#1baf7a", "yellow": "#eda100", "green": "#008300",
    "violet": "#4a3aa7", "red": "#e34948", "magenta": "#e87ba4", "orange": "#eb6834",
}
SERIES = [C["blue"], C["aqua"], C["yellow"], C["green"], C["violet"], C["red"]]
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, INK2 = "#0b0b0b", "#52514e"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#d8d7d2", "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#eceae5", "grid.linewidth": 0.7,
    "axes.axisbelow": True,
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
    "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "figure.dpi": 150, "savefig.bbox": "tight",
})


def savefig(fig, name: str):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.pdf")
    fig.savefig(FIG_DIR / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"  [fig] {name}")


# ----------------------------------------------------------------------------
# Numbers registry -> consumed by make_numbers.py into paper/numbers.tex
# ----------------------------------------------------------------------------
def reg(**kwargs):
    """Register formatted strings for LaTeX macros, e.g. reg(sentRate='53.6\\%')."""
    RES_DIR.mkdir(parents=True, exist_ok=True)
    cur = json.loads(NUMBERS.read_text()) if NUMBERS.exists() else {}
    for k, v in kwargs.items():
        assert k.isalpha(), f"macro name must be alphabetic: {k}"
        cur[k] = str(v)
    cur["_meta"] = _meta()
    cur["dataSnapshot"] = SNAPSHOT.name.replace("_", "\\_")
    NUMBERS.write_text(json.dumps(cur, indent=1, ensure_ascii=False))


INTERVALS = RES_DIR / "intervals.json"


def save_interval(key, **fields):
    """Record the unrounded estimate and intervals of one contrast in
    results/intervals.json, for forest_rows.py. The macros carry the rounded
    strings; this file carries the numbers a figure is drawn from."""
    RES_DIR.mkdir(parents=True, exist_ok=True)
    cur = json.loads(INTERVALS.read_text()) if INTERVALS.exists() else {}
    cur[key] = fields
    INTERVALS.write_text(json.dumps(cur, indent=1, ensure_ascii=False))


def _meta() -> dict:
    """Provenance stamped into the registry: which snapshot produced it, when,
    and at which revision. check_document.py asserts it against common.SNAPSHOT
    and statistical_tests.S, so a document can no longer name a snapshot its
    numbers did not come from."""
    import subprocess
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=str(BASE), capture_output=True, text=True,
                             timeout=10).stdout.strip() or "unknown"
    except Exception:
        rev = "unknown"
    return {"snapshot": SNAPSHOT.name, "baseline": BASELINE.name,
            "written": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "git_rev": rev}


def calibration():
    """The content-simulator calibration (results/calibration_content.json)."""
    return json.loads((RES_DIR / "calibration_content.json").read_text())


def draw_world(rng, items=(), users_=(), cal=None):
    """Draw one world from the calibration posterior.

    Returns (alpha, b_item, ell_user) with the effects SAMPLED as N(map, sd),
    not plugged in at their posterior means: shrinkage estimators under-disperse
    by construction, and the dispersion of item quality is what determines how
    much value the selection problem contains. Items or users absent from the
    calibration (never rated in the pilot) are drawn from the estimated prior.

    Draw once per replicate and hold fixed within it: the world is a property
    of the replicate, not of the decision.
    """
    cal = cal or calibration()
    si, su = cal["prior_var_item"] ** 0.5, cal["prior_var_user"] ** 0.5
    b = {k: v + rng.normal() * cal["b_item_sd"][k] for k, v in cal["b_item"].items()}
    l = {k: v + rng.normal() * cal["ell_user_sd"][k] for k, v in cal["ell_user"].items()}
    for it in items:
        b.setdefault(it, float(rng.normal() * si))
    for us in users_:
        l.setdefault(us, float(rng.normal() * su))
    return cal["alpha"], b, l


def pct(x, d=1):
    return f"{100*x:.{d}f}\\%"


# Participant accounts excluded from the analysis at the request of the platform provider
# (accounts created with errors or by mistake, with no events). The identifiers are
# pseudonymous participant identifiers and are not distributed with the code: the file
# data/excluded_user_ids.txt holds one UUID per line. Applied to the analysis loaders
# (users, feedback, missions), not to feature_audit and bandit_updates, which are the
# history of the deployed system.
_EXCLUDED_FILE = DATA_ROOT / "excluded_user_ids.txt"
EUT_EXCLUDED = frozenset(
    line.strip() for line in (_EXCLUDED_FILE.read_text().splitlines() if _EXCLUDED_FILE.exists() else [])
    if line.strip() and not line.startswith("#"))


# ----------------------------------------------------------------------------
# Loaders (cached)
# ----------------------------------------------------------------------------
@lru_cache(maxsize=None)
def users() -> pd.DataFrame:
    u = _with_lost_user(pd.read_csv(SNAPSHOT / "users.csv"), "users.csv")
    # 2026-07-20: 5 internal test accounts are labeled recruitmentcenter='test'
    # in the July exports (they were labeled 'IEO' in earlier exports). They have
    # zero feedback events but inflated every all-enrolled denominator (cohort
    # size, non-responder counts, demographic screens). Excluded everywhere.
    u = u[u["recruitmentcenter"] != "test"].reset_index(drop=True)
    u = u[~u["user_id"].astype(str).isin(EUT_EXCLUDED)].reset_index(drop=True)
    u["userage"] = pd.to_numeric(u["userage"], errors="coerce")
    u["enrol"] = pd.to_datetime(u["enrolmentdate"], format="ISO8601", utc=True, errors="coerce")
    # Two enrolment dates of the export are out of range. The creation date of the
    # account (`date_creation` of user_data.json) replaces enrolmentDate for every
    # participant; the two dates agree to the day for all the other participants.
    _created = {r["user_uuid"]: pd.Timestamp(r["date_creation"]["$date"]).normalize()
                for r in json.load(open(EXPORT / "ibechange-recommender.user_data.json"))}
    u["enrol"] = u["user_id"].map(_created).fillna(u["enrol"])
    edu = {"no-education": 0, "primary": 1, "secondary": 2, "vocational": 3,
           "university": 4, "postgraduate": 5, "other": np.nan}
    u["edu"] = u["education"].map(edu)
    return u


# Events with process_id = 0. On notification events the value marks a notification
# that the platform could not send (25 notification_sent rows of 11-12 February 2026,
# when the app was down), so these rows are never counted as sends. On mission events
# the value also occurs on real closures, which are kept.
#
# One participant account is present in the database up to the migration of 24 June
# 2026 and absent afterwards, while the export of the platform still holds it. Its rows
# are read from the last snapshot before the migration. The prefix of its identifier is
# not distributed with the code: data/recovered_user_prefix.txt holds it.
_RECOVERED_FILE = DATA_ROOT / "recovered_user_prefix.txt"
LOST_USER = _RECOVERED_FILE.read_text().strip() if _RECOVERED_FILE.exists() else None


def _lost_user_rows(table: str, user_col: str = "user_id") -> pd.DataFrame:
    """The rows of the recovered account, read from the last snapshot before the migration.

    Notification events with process_id = 0 are left out, as everywhere else; mission
    events with process_id = 0 are kept."""
    if LOST_USER is None:
        return pd.DataFrame()
    old_ = pd.read_csv(BASELINE / table)
    rows = old_[old_[user_col].astype(str).str.startswith(LOST_USER)].copy()
    if table == "user_feedback.csv":
        never_sent = (rows["process_id"].astype(str).isin(["0", "0.0"])
                      & rows["event_name"].astype(str).str.startswith("notification"))
        rows = rows[~never_sent]
    return rows


def _with_lost_user(df: pd.DataFrame, table: str, user_col: str = "user_id"):
    extra = _lost_user_rows(table, user_col)
    return pd.concat([df, extra], ignore_index=True) if len(extra) else df

OUTAGE_START = pd.Timestamp("2026-02-11T00:00:00Z")
OUTAGE_END = pd.Timestamp("2026-02-13T00:00:00Z")


@lru_cache(maxsize=None)
def feedback() -> pd.DataFrame:
    fb = _with_lost_user(pd.read_csv(SNAPSHOT / "user_feedback.csv"),
                         "user_feedback.csv")
    fb["ts"] = pd.to_datetime(fb["timestamp"], format="ISO8601", utc=True)
    props = fb["properties"].apply(json.loads)
    fb["content_id"] = props.apply(lambda d: d.get("content_id"))
    fb["rating"] = props.apply(lambda d: d.get("rating"))
    fb["score"] = props.apply(lambda d: d.get("score"))
    fb["mission_prop"] = props.apply(lambda d: d.get("mission_id"))
    # `eom` marks events produced by the mission-close review flow rather than
    # by engaging with a push notification (the backend key is typo'd upstream).
    # It applies to BOTH ratings and opens, and the opens are the part that is
    # easy to get wrong:
    #
    #   eom RATING  = the participant rated the item from the end-of-mission
    #                 review list (about 30% of all ratings);
    #   eom OPEN    = the item was PRESENTED in that review list. It is not an
    #                 act of reading the notification. 342 of the 344 eom opens
    #                 are followed by an eom rating, so presentation and rating
    #                 are nearly the same event.
    #
    # RULE, and the reason this comment is long: whenever you ask whether an
    # item "was opened" as a CAUSE or a PRECONDITION of something that happens
    # at mission close, count in-flow opens only (`~eom`). An eom open is a
    # CONSEQUENCE of the review, so including it conditions on the outcome.
    # Concretely, on the delayed-feedback eligibility question this inverts the
    # answer: 47.7% vs 2.2% with in-flow opens, 5.2% vs 87.5% with all opens.
    # See a02_delivery.py (df* macros) and the compendium's known-issues
    # section. This mistake has been made three times in this project.
    fb["eom"] = props.apply(lambda d: bool(d.get("is_end_misison")))
    fb["kind"] = fb["content_id"].map(content_kind)
    fb = fb[~fb["user_id"].astype(str).isin(EUT_EXCLUDED)].reset_index(drop=True)
    return fb


def content_kind(content_id) -> str | None:
    """'recommendation' (id <letter>Rc<n>) or 'resource' (id <letter>Rs<n>).

    The kind comes from the content id and never from the event's
    content_type, which the app mislabels in 38 of the 1,183 ratings of the
    20 Aug snapshot (31 resources sent as "recommendation", 7 recommendations
    as "resource"). Ratings of resources update the resource bandit, not the
    model of the rating, so every analysis of the rating of an item uses
    recommendation_ratings() below. Until 27 Oct 2026 the compendium counted
    the 107 resource ratings as item ratings (1,187 ratings and 59 dislikes
    instead of 1,080 and 51).
    """
    m = re.fullmatch(r"[A-Z]R([cs])\d+", str(content_id))
    if m is None:
        return None
    return "recommendation" if m.group(1) == "c" else "resource"


def recommendation_ratings(fb: pd.DataFrame) -> pd.DataFrame:
    """Rating events of recommendations; resource ratings are left out."""
    return fb[(fb["event_name"] == "notification_rated")
              & (fb["kind"] == "recommendation")]


def resource_ratings(fb: pd.DataFrame) -> pd.DataFrame:
    """Rating events of resources, reported separately from item ratings."""
    return fb[(fb["event_name"] == "notification_rated")
              & (fb["kind"] == "resource")]


def closure_times(nm: pd.DataFrame, fb: pd.DataFrame) -> pd.Series:
    """Time of the mission_accomplished event of each mission start (row of nm).

    What the event records (verified on the 20 Aug 2026 snapshot): the app had a
    single "complete or change mission" button, and closing the current mission
    with it was required before selecting another (947 of 948 starts that are
    not the participant's last are closed before the next selection). The event
    therefore records that the participant closed the mission instance, whether
    by completing it or by changing it; no field tells the two apart, and there
    is no automatic closure at expiry. Each start has at most one event.

    Matching: an event belongs to the latest start of the same participant and
    mission selected at or before it. Returns NaT for starts never closed.
    """
    ev = fb[fb["event_name"] == "mission_accomplished"]
    starts: dict = {}
    for i, r in nm.iterrows():
        starts.setdefault((r["user_id"], r["mission_id"]), []).append((r["sel"], i))
    close = pd.Series(pd.NaT, index=nm.index, dtype="datetime64[ns, UTC]")
    for r in ev.itertuples():
        cands = [(s, i) for s, i in starts.get((r.user_id, r.mission_prop), [])
                 if pd.notna(s) and s <= r.ts]
        if cands:
            i = max(cands, key=lambda c: c[0])[1]
            # keep the first event if a start ever had two (none in the data)
            if pd.isna(close[i]):
                close[i] = r.ts
    return close


@lru_cache(maxsize=None)
def missions() -> pd.DataFrame:
    nm = _with_lost_user(pd.read_csv(SNAPSHOT / "new_missions_and_contents.csv"),
                         "new_missions_and_contents.csv")
    nm["sel"] = pd.to_datetime(nm["selection_timestamp"], format="ISO8601", utc=True)
    # The database field, kept for reference only. It is NOT the closure
    # indicator: it is missing for 20 starts that have a closure event (17 after
    # the 24 June 2026 migration, 3 of the recovered account), and where both exist they agree
    # to the second.
    nm["fin_db"] = pd.to_datetime(nm["finish_timestamp"], format="ISO8601", utc=True, errors="coerce")
    nm = nm[~nm["user_id"].astype(str).isin(EUT_EXCLUDED)].reset_index(drop=True)
    # Closure = the mission_accomplished event matched to this start (see
    # closure_times). `fin` is its time and `closed` says whether it exists;
    # every script that reads `fin` as "the mission was closed" gets this.
    nm["fin"] = closure_times(nm, feedback())
    nm["closed"] = nm["fin"].notna()
    nm["dur_h"] = (nm["fin"] - nm["sel"]).dt.total_seconds() / 3600
    return nm


# ----------------------------------------------------------------------------
# Cap missions (added 6 Oct 2026)
# ----------------------------------------------------------------------------
# Missions that cap a behaviour: "Reduce intake of X to N occasions/servings
# (or less) this week". All are Nutrition missions, and each has N steps in the
# deployed catalog, i.e. as many calendar boxes as the weekly allowance. No app
# text explained what a tick meant on these missions. With N boxes equal to the
# allowance, a tick most plausibly marks one allowed serving consumed (deduced
# from the box count, not documented); any number of ticks from 0 to N is then
# compatible with staying within the cap, exceeding the cap cannot be recorded,
# and the closure score carries no information on adherence for them. Analyses
# of the closure score or of "all tasks ticked" are reported on all missions
# (primary) and with these missions excluded (sensitivity, macros *NoCap).
#
# Rule: the mission text, from the canonical clinical workbook (the deployed
# catalog JSON carries no text), matches CAP_MISSION_RE, and the allowance N in
# the text equals num_steps in the deployed catalog (EUT export mission.json).
# On the 273 deployed missions the rule selects 40 (NM40-NM92, with gaps), and
# N == num_steps holds for every text that matches; the assertion below keeps it so.
MISSION_WORKBOOK = (PILOT_ROOT / "intervention_structure" / "pilot_intervention_structure"
                    / "STEP 5. Intervention (and Stratification)" / "5.2. Missions"
                    / "5.2. Missions_finished.xlsx")
CAP_MISSION_RE = (r"^Reduce intake of .* to (one|two|three|four|five|six|seven) "
                  r"(occasions?|servings?)( or less)? this week\.?$")
_NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7}


@lru_cache(maxsize=None)
def cap_missions() -> frozenset:
    """Mission ids of the cap missions (see the comment above)."""
    import re
    import openpyxl
    steps = {r["mission_id"]: r["num_steps"]
             for r in json.load(open(EXPORT / "ibechange-recommender.mission.json"))}
    wb = openpyxl.load_workbook(MISSION_WORKBOOK, read_only=True, data_only=True)
    text = {}
    for ws in wb.worksheets:
        rows = list(ws.iter_rows(values_only=True))
        head = [str(x).strip() if x else None for x in rows[0]]
        ci, mi = head.index("Code"), head.index("Mission")
        for r in rows[1:]:
            r = tuple(r) + (None,) * len(head)
            if r[ci] and r[mi]:
                text[str(r[ci]).strip()] = " ".join(str(r[mi]).split())
    wb.close()
    pat = re.compile(CAP_MISSION_RE)
    cap = set()
    for mid, k in steps.items():
        m = pat.match(text.get(mid, ""))
        if m:
            assert _NUMBER_WORDS[m.group(1)] == k, (mid, m.group(1), k)
            cap.add(mid)
    return frozenset(cap)


# Missions that ask for a set number of actions in the week ("Scheduled count", e.g. "Eat
# one serving of varied fresh fruits three times during this week"), on which a tick records
# an action done. The forms are those of the sheet "Pilot missions by form" of
# rct/design/mission_forms_calendar_rules.xlsx (205 of the 273 deployed missions; the other
# forms are weekly cap, daily state and event-contingent, on which a tick or its absence does
# not record completion unambiguously). Used for the dose contrast of D4.5 Section 7.1
# (decided 9 November 2026).
MISSION_FORMS = PILOT_ROOT.parent / "rct" / "design" / "mission_forms_calendar_rules.xlsx"


@lru_cache(maxsize=None)
def scheduled_count_missions() -> frozenset:
    """Mission ids of the pilot missions of form "Scheduled count"."""
    import openpyxl
    wb = openpyxl.load_workbook(MISSION_FORMS, read_only=True, data_only=True)
    rows = list(wb["Pilot missions by form"].iter_rows(values_only=True))
    wb.close()
    head = list(rows[0])
    code, form = head.index("Code"), head.index("Form")
    ids = frozenset(r[code] for r in rows[1:] if r[code] and r[form] == "Scheduled count")
    assert len(ids) == 205, len(ids)
    return ids


@lru_cache(maxsize=None)
def health_assessments() -> pd.DataFrame:
    """Health-habit assessments, with the recovered account restored and the timestamp parsed.

    Read through this rather than from the CSV directly, so that the restored
    participant reaches every analysis that uses the clinical outcome.
    """
    h = _with_lost_user(pd.read_csv(SNAPSHOT / "health_habit_assessments.csv"),
                        "health_habit_assessments.csv")
    h["ts"] = pd.to_datetime(h["assessment_timestamp"], format="ISO8601",
                             utc=True, errors="coerce")
    return h.reset_index(drop=True)


@lru_cache(maxsize=None)
def feature_audit() -> pd.DataFrame:
    fa = _with_lost_user(pd.read_csv(SNAPSHOT / "feature_audit.csv"),
                         "feature_audit.csv")
    fa["fv_arr"] = fa["fv"].apply(lambda s: np.array([float(v) for v in s.strip("{}").split(",")]))
    return fa


@lru_cache(maxsize=None)
def upgrades() -> pd.DataFrame:
    uv = pd.read_csv(SNAPSHOT / "user_version_upgrades.csv")
    uv["uts"] = pd.to_datetime(uv["timestamp"], format="ISO8601", utc=True)
    return uv


@lru_cache(maxsize=None)
def bandit_updates(new: bool = False) -> pd.DataFrame:
    t = "intervention_mab_updates_new" if new else "intervention_mab_updates"
    d = pd.read_csv(SNAPSHOT / f"{t}.csv")
    d["pts"] = pd.to_datetime(d["processed_at"], format="ISO8601", utc=True)
    d["ets"] = pd.to_datetime(d["timestamp"], format="ISO8601", utc=True)
    return d.sort_values("pts").reset_index(drop=True)


@lru_cache(maxsize=None)
def accomplishments() -> pd.DataFrame:
    """mission_accomplished events with episode duration (selection -> event)."""
    fb = feedback()
    ma = fb[fb["event_name"] == "mission_accomplished"].copy()
    ma["score"] = pd.to_numeric(ma["score"], errors="coerce")
    ma["mission_id"] = ma["mission_prop"]
    ma = ma.dropna(subset=["score"]).sort_values("ts")
    sel_map: dict = {}
    for _, r in missions().iterrows():
        sel_map.setdefault((r["user_id"], r["mission_id"]), []).append(r["sel"])
    for k in sel_map:
        sel_map[k] = sorted(sel_map[k])

    def dur(row):
        sels = [s for s in sel_map.get((row["user_id"], row["mission_id"]), []) if s <= row["ts"]]
        return (row["ts"] - sels[-1]).total_seconds() / 3600 if sels else np.nan

    ma["dur_h"] = ma.apply(dur, axis=1)
    return ma


# ----------------------------------------------------------------------------
# Delivery pipeline: OMI-scheduled slots matched to EUT sent events
# ----------------------------------------------------------------------------
@lru_cache(maxsize=None)
def delivery() -> pd.DataFrame:
    """One row per OMI-scheduled slot with columns:
    user_id, content_id, sched, sent_ok, mission_id (where known),
    ep_sel, ep_fin (episode driving the plan), post_finish (suppressed)."""
    plans = _with_lost_user(pd.read_csv(SNAPSHOT / "recommendation_plans.csv"),
                            "recommendation_plans.csv")[["plan_id", "user_id"]]
    slots = pd.read_csv(SNAPSHOT / "plan_contents.csv")
    # plan_contents carries no user_id, so the restored participant's slots are
    # taken by plan id from the pre-migration snapshot
    lost_plans = set(_lost_user_rows("recommendation_plans.csv")["plan_id"])
    if lost_plans:
        old_slots = pd.read_csv(BASELINE / "plan_contents.csv")
        slots = pd.concat([slots, old_slots[old_slots["plan_id"].isin(lost_plans)]],
                          ignore_index=True)
    pc = slots.merge(plans, on="plan_id", how="left")
    pc["sched"] = pd.to_datetime(pc["scheduled_for"], format="ISO8601", utc=True)
    pc = pc[pc["source"] == "omi"].reset_index(drop=True)
    pc = pc[~pc["user_id"].astype(str).isin(EUT_EXCLUDED)].reset_index(drop=True)

    fb = feedback()
    # Censoring guard: slots scheduled after the last feedback event of ANY
    # type are unobservable in this export (their sent events cannot have been
    # ingested yet) and would count as spurious misses. The 2026-08-01 export
    # e.g. has feedback until Jul 29 21:08 but slots scheduled through Aug 1.
    horizon = fb["ts"].max()
    pc = pc[pc["sched"] <= horizon].reset_index(drop=True)
    sent = fb[fb["event_name"] == "notification_sent"]
    sent_idx: dict = {}
    for _, r in sent.iterrows():
        sent_idx.setdefault((r["user_id"], r["content_id"]), []).append(r["ts"])

    used, ok = set(), []
    for _, r in pc.iterrows():
        cands = sent_idx.get((r["user_id"], r["content_id"]), [])
        best, bd = None, None
        for i, t in enumerate(cands):
            if (r["user_id"], r["content_id"], i) in used:
                continue
            d = abs(t - r["sched"])
            if d <= pd.Timedelta("30min") and (bd is None or d < bd):
                best, bd = i, d
        if best is not None:
            used.add((r["user_id"], r["content_id"], best))
            ok.append(True)
        else:
            ok.append(False)
    pc["sent_ok"] = ok

    # plan -> mission mapping. The 2026-06-24 dual-rating migration truncated
    # selected_contents in prod (454 -> 15 rows), so: BASELINE (old-prod export,
    # plans up to 2026-06-23) + SNAPSHOT (live table, post-migration plans only).
    # Without the SNAPSHOT part, post-fix slots have no mission mapping and drop
    # out of active_slots(), starving the post-fix delivery verification.
    sc = (pd.concat([pd.read_csv(BASELINE / "selected_contents.csv"),
                     pd.read_csv(SNAPSHOT / "selected_contents.csv")])
          .dropna(subset=["mission_id"]).drop_duplicates("plan_id")[["plan_id", "mission_id"]])
    pc = pc.merge(sc, on="plan_id", how="left")
    pc["first_sched"] = pc.groupby("plan_id")["sched"].transform("min")

    nm = missions()
    key = pc[["plan_id", "user_id", "mission_id", "first_sched"]].drop_duplicates("plan_id")

    def episode(row):
        if pd.isna(row["mission_id"]):
            return pd.Series({"ep_sel": pd.NaT, "ep_fin": pd.NaT})
        c = nm[(nm["user_id"] == row["user_id"]) & (nm["mission_id"] == row["mission_id"])
               & (nm["sel"] <= row["first_sched"] + pd.Timedelta("1h"))]
        if len(c) == 0:
            return pd.Series({"ep_sel": pd.NaT, "ep_fin": pd.NaT})
        r = c.sort_values("sel").iloc[-1]
        return pd.Series({"ep_sel": r["sel"], "ep_fin": r["fin"]})

    key[["ep_sel", "ep_fin"]] = key.apply(episode, axis=1)
    pc = pc.merge(key[["plan_id", "ep_sel", "ep_fin"]], on="plan_id", how="left")
    pc["post_finish"] = pc["ep_fin"].notna() & (pc["sched"] > pc["ep_fin"])
    pc["hour"] = pc["sched"].dt.hour
    pc["peak"] = pc["hour"].isin(PEAK_HOURS)
    return pc


def active_slots() -> pd.DataFrame:
    d = delivery()
    return d[d["mission_id"].notna() & ~d["post_finish"]].copy()


# ----------------------------------------------------------------------------
# Stats helpers
# ----------------------------------------------------------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def logloss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def fit_ridge_logistic(X, y, lam=10.0):
    Xb = np.hstack([np.ones((len(X), 1)), X])
    w = np.zeros(Xb.shape[1])
    for _ in range(60):
        p = sigmoid(Xb @ w)
        W = np.maximum(p * (1 - p), 1e-6)
        g = Xb.T @ (y - p) - lam * np.r_[0, w[1:]]
        H = (Xb * W[:, None]).T @ Xb + lam * np.diag(np.r_[0, np.ones(len(w) - 1)])
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        w += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return w


def predict_lr(w, X):
    return sigmoid(np.hstack([np.ones((len(X), 1)), X]) @ w)


def eb_rate(train_ent, train_y, test_ent, prior=20.0):
    gm = train_y.mean()
    d = {}
    for e in set(train_ent):
        m = train_ent == e
        d[e] = (train_y[m].sum() + prior * gm) / (m.sum() + prior)
    return np.array([d.get(e, gm) for e in test_ent]), d, gm


def eb_kappa(ent, y, m=None, lo=0.5, hi=1000.0):
    """Weight kappa of eb_rate estimated by empirical Bayes (5 Nov 2026).

    The like share of each entity (item or participant) is taken to follow a
    Beta distribution with mean m and weight kappa, Beta(kappa*m, kappa*(1-m)),
    so that the likes of an entity with n ratings follow a beta-binomial
    distribution. kappa maximises the beta-binomial likelihood of the like
    counts of all entities, with m fixed to the share of likes of y unless it
    is given. The search is bounded to [lo, hi]: when the entities do not
    differ more than chance allows, the likelihood keeps rising with kappa and
    the estimate stops at hi, which shrinks every entity almost to m.
    """
    from scipy.optimize import minimize_scalar
    from scipy.special import betaln
    ent, y = np.asarray(ent), np.asarray(y, dtype=float)
    if m is None:
        m = y.mean()
    _, inv = np.unique(ent, return_inverse=True)
    s = np.bincount(inv, weights=y)
    n = np.bincount(inv).astype(float)

    def nll(logk):
        k = np.exp(logk)
        a, b = k * m, k * (1 - m)
        return -np.sum(betaln(s + a, n - s + b) - betaln(a, b))

    r = minimize_scalar(nll, bounds=(np.log(lo), np.log(hi)), method="bounded",
                        options={"xatol": 1e-6})
    return float(np.exp(r.x))


def eb_logit(train_ent, train_y, test_ent, prior=20.0):
    r, d, gm = eb_rate(train_ent, train_y, test_ent, prior)
    return np.log(r / (1 - r)), d, gm


def user_fold(uid, k=5):
    return int(hashlib.md5(str(uid).encode()).hexdigest(), 16) % k


def auc(p, y):
    from scipy.stats import mannwhitneyu
    y = np.asarray(y)
    if y.sum() in (0, len(y)):
        return np.nan
    u, _ = mannwhitneyu(p[y == 1], p[y == 0])
    return u / ((y == 1).sum() * (y == 0).sum())


# ---- cross-validated AUC: mean of the per-fold AUCs (since 5 October 2026) ----
# Every cross-validated AUC of the paper is the mean over the five folds of the
# AUC computed inside each fold. Pooling the held-out predictions of all folds
# into one AUC also compares predictions from different folds, whose models
# were fitted on different training sets; under folds over participants a
# predictor that is constant within each fold scores 0.33 pooled and exactly
# 0.50 as a per-fold mean.
CV_BOOT_DRAWS = 2000
CV_BOOT_SEED = 20261002


def rank_auc(p, y):
    """Mann-Whitney AUC with mid-ranks for ties (same value as auc()); nan when
    y has one class only."""
    from scipy.stats import rankdata
    y = np.asarray(y)
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    return (rankdata(p)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def fold_aucs(p, y, folds, rows=None, k=5):
    """AUC inside each fold, as an array of length k. `rows` selects (with
    repetition) the records of a bootstrap draw. A fold with one class only
    (no dislike in a draw) gives nan."""
    if rows is None:
        rows = np.arange(len(y))
    p_, y_, f_ = p[rows], y[rows], folds[rows]
    return np.array([rank_auc(p_[f_ == f], y_[f_ == f]) for f in range(k)])


def fold_mean_auc(p, y, folds, k=5):
    """The cross-validated AUC: mean of the per-fold AUCs."""
    return float(np.mean(fold_aucs(p, y, folds, k=k)))


def participant_draws(groups, n=CV_BOOT_DRAWS, seed=CV_BOOT_SEED):
    """Row indices of `n` bootstrap draws that resample participants with
    replacement (all records of a drawn participant enter the draw)."""
    rng = np.random.default_rng(seed)
    people = np.unique(groups)
    rows = [np.where(groups == u)[0] for u in people]
    return [np.concatenate([rows[j] for j in rng.integers(0, len(people), len(people))])
            for _ in range(n)]


def fold_auc_draws(p, y, folds, draws, k=5):
    """Per-fold AUCs of the held-out predictions in every draw (models not
    refitted): an array of shape (draws, k), nan where a fold has one class."""
    return np.array([fold_aucs(p, y, folds, r, k) for r in draws])


def cv_auc_interval(per_fold, per_fold_ref=None, ref_value=None):
    """Bootstrap limits of a per-fold-mean AUC, or of a difference (per_fold
    minus per_fold_ref, or minus the constant ref_value). The per_fold arrays
    come from fold_auc_draws.

    Primary rule: a draw in which a fold has no dislike (nan) is dropped.
    Sensitivity check: the same draws averaged over the folds available.
    Returns the 90% and 95% limits, the number of draws kept, the 90% limits
    of the sensitivity check and the kept draws themselves ("draws", not
    JSON-serialisable)."""
    def stat(agg):
        a = agg(per_fold, axis=1)
        if per_fold_ref is not None:
            a = a - agg(per_fold_ref, axis=1)
        if ref_value is not None:
            a = a - ref_value
        return a
    d = stat(np.mean)
    d = d[~np.isnan(d)]
    alt = stat(np.nanmean)
    alt = alt[~np.isnan(alt)]
    return dict(ci90=[float(v) for v in np.percentile(d, [5, 95])],
                ci95=[float(v) for v in np.percentile(d, [2.5, 97.5])],
                n_draws_kept=int(len(d)),
                ci90_available_folds=[float(v) for v in np.percentile(alt, [5, 95])],
                draws=d)


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return (c - h, c + h)
