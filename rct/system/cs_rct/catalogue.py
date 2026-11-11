"""The catalogue the service reads: Missions, reminders, recommendations and resources.

Missions and reminders are CSV files in system/catalogue/, generated from the clinical
workbooks (scripts/build_mission_catalogue.py, scripts/build_reminder_catalogue.py), and so are
the IDs of the pop-up texts of recommendations and resources (catalogue/push_texts.csv,
scripts/build_push_texts.py), whose text EUT sends. Recommendations and
resources reach the service from VU through POST /recommendations and POST /resources. Recommendations are
written by the clinical team and reach the service from VU through
POST /recommendations, in the format of GET /internal/recommendations
(rct/design/iBeChange-modules-architecture.docx): {content_id: {"missions": [...], ...}}.
"""

import csv
import hashlib
import json
import re
from datetime import time
from dataclasses import dataclass, field
from pathlib import Path

CATALOGUE_FOLDER = Path(__file__).parent.parent / "catalogue"

# Frequencies of the catalogue that state no count of days (paper, Section "Episodes
# and the daily outcome"). "daily" is a condition held every day; the other two are
# acts whose occasion the participant does not choose.
DAILY = "daily"
NO_COUNT = ("non-schedulable", "when-possible")
# Frequencies that state a count of days k_m. The k_m column of a capped Mission holds
# its cap of servings instead, and its frequency is one of NO_COUNT.
COUNT_FREQUENCIES = ("once a week", "twice a week", "three times a week", "four times a week",
                     "five times a week", "six times a week")


@dataclass(frozen=True)
class Mission:
    mission_id: str
    pillar: str
    objectives: tuple
    mission_class: str        # "A": marking required to complete; "B": not required
    frequency: str            # e.g. "three times a week", "daily", "non-schedulable"
    count: int | None         # k_m where the frequency states a count of days, else None
    cap: int | None           # the weekly cap of servings of a capped Mission, else None
    # Share of days the Mission implies when met exactly: k_m / W, 1 on a daily one, and
    # 0 on a non-schedulable or when-possible one, whose frequency indicator carries the level.
    expected_share: float
    schedulable: bool
    reminder_pools: tuple     # pools of day reminders (Objective, or group in Emotional Wellbeing)

    @property
    def states_count(self):
        return self.count is not None


@dataclass(frozen=True)
class Reminder:
    reminder_id: str
    kind: str                 # "day" or "hour"
    pools: tuple              # for a day reminder, the pools it belongs to
    missions: tuple           # for an hour reminder, the schedulable Missions it serves
    bct: str                  # the CACTIS BCT code, logged as the reminder's class
    texts: dict               # language code -> text
    latest_time: object = None  # datetime.time after which the reminder is not sent, or None


@dataclass
class Catalogue:
    missions: dict = field(default_factory=dict)            # mission_id -> Mission
    day_reminders: dict = field(default_factory=dict)       # pool -> [Reminder]
    hour_reminder_of: dict = field(default_factory=dict)    # mission_id -> Reminder
    reminders: dict = field(default_factory=dict)           # reminder_id -> Reminder
    recommendations_of: dict = field(default_factory=dict)  # mission_id -> [content_id]
    recommendation_code: dict = field(default_factory=dict)  # content_id -> code shared by its variations
    recommendation_subdomains: dict = field(default_factory=dict)  # content_id -> frozenset of COM-B sub-domains
    resources_of: dict = field(default_factory=dict)        # mission_id -> [content_id]
    push_texts: list = field(default_factory=list)          # IDs of the pop-up texts
    # What was read: {file name or "recommendations"/"resources": SHA-256}, logged with every
    # decision (decisions.catalogue_version).
    version: dict = field(default_factory=dict)

    def day_reminder_pool(self, mission_id):
        """Every day reminder that can be sent on the Mission, without repetitions."""
        mission = self.missions.get(mission_id)
        if mission is None:
            return []
        seen, pool = set(), []
        for name in mission.reminder_pools:
            for reminder in self.day_reminders.get(name, []):
                if reminder.reminder_id not in seen:
                    seen.add(reminder.reminder_id)
                    pool.append(reminder)
        return pool


def load(recommendations=None, resources=None, folder=CATALOGUE_FOLDER):
    """Read the CSV files, and the recommendations and resources if given in the VU format."""
    catalogue = Catalogue()
    pools_of = {row["mission_id"]: tuple(row["pools"].split()) for row in read_csv(folder / "reminder_pools.csv")}

    for row in read_csv(folder / "missions.csv"):
        frequency = row["frequency"]
        count = int(row["k_m"]) if frequency in COUNT_FREQUENCIES else None
        cap = int(row["k_m"]) if row["weekly_cap"] == "yes" and row["k_m"] != "" else None
        if frequency == DAILY:
            expected_share = 1.0
        elif count is not None:
            expected_share = count / 7
        else:
            expected_share = 0.0
        catalogue.missions[row["code"]] = Mission(
            mission_id=row["code"],
            pillar=row["pillar"],
            objectives=tuple(o.strip() for o in row["objectives"].split(";") if o.strip()),
            mission_class=row["class"],
            frequency=frequency,
            count=count,
            cap=cap,
            expected_share=expected_share,
            schedulable=row["schedulable"] == "yes",
            reminder_pools=pools_of.get(row["code"], ()),
        )

    if (folder / "push_texts.csv").exists():
        catalogue.push_texts = [int(row["push_text_id"]) for row in read_csv(folder / "push_texts.csv")]

    for row in read_csv(folder / "reminders.csv"):
        reminder = Reminder(
            reminder_id=row["reminder_id"],
            kind=row["kind"],
            pools=tuple(row["pools"].split()),
            missions=tuple(row["missions"].split()),
            bct=row["bct"],
            texts={language: row[f"text_{language}"] for language in ("es", "en", "it", "ro")},
            latest_time=time.fromisoformat(row["latest_time"]) if row.get("latest_time") else None,
        )
        catalogue.reminders[reminder.reminder_id] = reminder
        if reminder.kind == "day":
            for pool in reminder.pools:
                catalogue.day_reminders.setdefault(pool, []).append(reminder)
        else:
            for mission_id in reminder.missions:
                catalogue.hour_reminder_of[mission_id] = reminder


    if recommendations:
        set_recommendations(catalogue, recommendations)
    if resources:
        catalogue.resources_of = by_mission(resources)
    for name in ("missions.csv", "reminder_pools.csv", "reminders.csv", "push_texts.csv"):
        if (folder / name).exists():
            catalogue.version[name] = hashlib.sha256((folder / name).read_bytes()).hexdigest()
    for name, body in (("recommendations", recommendations), ("resources", resources)):
        if body:
            catalogue.version[name] = hashlib.sha256(
                json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()
    return catalogue


def set_recommendations(catalogue, recommendations):
    """Index the recommendations, given as {content_id: {"missions": [...], "com_b": ..., ...}}, by
    Mission, with the code each shares with its variations and its COM-B sub-domains.

    The code is the attribute "code" if present, else the id less a final lowercase letter (the
    variation letter of the clinical workbook: SRc1a and SRc1b are variations of SRc1); how VU
    will send the variations is still to be agreed with EUT. "com_b" is a name or a list of names
    separated by commas, semicolons or slashes, as in the workbook ("Psychological Capability,
    Physical Opportunity")."""
    catalogue.recommendations_of = by_mission(recommendations)
    for content_id, attributes in recommendations.items():
        catalogue.recommendation_code[content_id] = attributes.get("code") or VARIATION_SUFFIX.sub("", content_id)
        # "com_b" in the EUT document, "comb" (a list) in what EUT serves on INT.
        catalogue.recommendation_subdomains[content_id] = subdomain_names(attributes.get("com_b") or attributes.get("comb"))


VARIATION_SUFFIX = re.compile(r"(?<=\d)[a-z]$")
COMB_SUBDOMAINS = ("physical capability", "psychological capability", "physical opportunity",
                   "social opportunity", "reflective motivation", "automatic motivation")
# Spellings found in the clinical workbook and in the Step 3 questionnaires.
SPELLING = {"oportunity": "opportunity", "reflecive": "reflective", "reflecitve": "reflective"}


def subdomain_names(value):
    """The COM-B sub-domains named by a "com_b" attribute or a Step 3 tag, in lowercase with spaces:
    "Psychological Capability, Physical Oportunity" and "psychological_capability" are both read.
    Names that are not COM-B sub-domains ("maintenance_self_efficacy", "test") are left out."""
    names = value if isinstance(value, list) else re.split(r"[,;/]", value or "")
    found = set()
    for name in names:
        name = " ".join((name or "").lower().replace("_", " ").split())
        for wrong, right in SPELLING.items():
            name = name.replace(wrong, right)
        # "psychologica_capability": Smoking 3d in the questionnaire of 3 November 2026
        if name in ("sychological capability", "psychologica capability"):
            name = "psychological capability"
        if name in COMB_SUBDOMAINS:
            found.add(name)
    return frozenset(found)


def hapa_stage_name(value):
    """"pre-intender", "intender" or "actor" from a Step 3 tag ("pre-intender, motivational",
    "Intender"), or None."""
    first = re.split(r"[,;]", (value or "").lower())[0].strip().replace(" ", "")
    if first in ("pre-intender", "preintender"):
        return "pre-intender"
    return first if first in ("intender", "actor") else None


MISSION_CODE = re.compile(r"([A-Z]+M)\s*0*(\d+)")


def mission_code(value):
    """The Mission code of the catalogue (missions.csv) for a code written with leading zeros or
    spaces: "SM01" and " SM1 " give "SM1". The resources workbook of ICO of 3 November 2026 writes
    "SM01 - SM49" while the Missions workbook writes SM1; the recommendations workbook of 28 October
    2026 writes "EM 31" for EM31. Other values are returned stripped."""
    text = str(value).strip()
    match = MISSION_CODE.fullmatch(text)
    return f"{match.group(1)}{int(match.group(2))}" if match else text


def by_mission(items):
    """{mission_id: [content_id, ...]} from {content_id: {"missions": [...], ...}}, with the Mission
    codes normalised by mission_code; a content listed twice for the same Mission counts once."""
    out = {}
    for content_id, attributes in sorted(items.items()):
        for mission_id in dict.fromkeys(mission_code(m) for m in attributes.get("missions", [])):
            out.setdefault(mission_id, []).append(content_id)
    return out


def read_csv(path):
    with open(path, newline="") as file:
        return list(csv.DictReader(file))
