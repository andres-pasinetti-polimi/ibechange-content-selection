"""Build catalogue/missions.csv from the ICO Missions workbook.

The module reads two fields of each Mission (rct/paper/main.tex, "Episodes and
the daily outcome"): the class the app gives it, A where marking the days is
required to complete it and B where it is not, and the frequency it asks for,
which is either a count of days ("daily" counting as seven) or one of the
words the workbook uses for the Missions that state none ("non-schedulable",
"when possible"). On a Mission with a weekly cap, k_m holds the cap stated in
its text (servings or drinks a week) instead of a count of days. It also
copies whether the participant can plan the Mission's tasks in the calendar
(column Schedulable), which decides whether a reminder can be sent at a declared
hour. The
daily outcome is the calendar mark itself, so nothing else about a Mission is
needed to read its days.

Columns are read by their header, because the sheets do not carry the same
columns in the same order.

The CSV is generated, never edited by hand.

Run from the system/ folder:
    python -m scripts.build_mission_catalogue
"""

import csv
import re
from pathlib import Path

import openpyxl

SYSTEM_FOLDER = Path(__file__).parent.parent
WORKBOOK = SYSTEM_FOLDER.parent / "intervention_structure" / "RCT_Missions_v5_16Oct2026.xlsx"
OUTPUT = SYSTEM_FOLDER / "catalogue" / "missions.csv"

# Days per week, for the frequencies that state a count.
COUNT_OF_FREQUENCY = {
    "once a week": 1,
    "twice a week": 2,
    "three times a week": 3,
    "four times a week": 4,
    "five times a week": 5,
    "six times a week": 6,
    "daily": 7,
}

# The workbook spells the same frequency in more than one way.
SPELLING_OF_FREQUENCY = {
    "no-schedulable": "non-schedulable",
    "when possible": "when-possible",
}

# The header a field carries, in the order the sheets use.
HEADERS = {
    "code": ["Code"],
    "objectives": ["Objective_associated"],
    "mission_class": ["Mission_type", "Missions_type"],
    "frequency": ["Frequency"],
    "weekly_cap": ["Weekly_cap"],
    "schedulable": ["Schedulable", "schedulable"],
    "mission_text": ["Mission"],
}


def normalise_frequency(text):
    frequency = " ".join(str(text).split()).lower()
    return SPELLING_OF_FREQUENCY.get(frequency, frequency)


def cap_from_text(mission_text):
    """Return N from "... máximo N ..." or "... máximo de N ...", N - 1 from "menos de N ...", or None."""
    match = re.search(r"m[aá]ximo (?:de )?(\d+)", mission_text)
    if match:
        return int(match.group(1))
    match = re.search(r"menos de (\d+)", mission_text)
    return int(match.group(1)) - 1 if match else None


def cell(value):
    """Return the text of a workbook cell, "" for an empty one."""
    return "" if value is None else " ".join(str(value).split())


def column_index(headers, names):
    for name in names:
        if name in headers:
            return headers.index(name)
    raise KeyError(f"none of {names} in {headers}")


def read_workbook():
    workbook = openpyxl.load_workbook(WORKBOOK, read_only=True, data_only=True)
    missions = []
    for sheet in workbook.worksheets:
        rows = list(sheet.iter_rows(values_only=True))
        headers = [" ".join(str(cell).split()) if cell else "" for cell in rows[0]]
        where = {field: column_index(headers, names) for field, names in HEADERS.items()}
        for row in rows[1:]:
            if not any(row):
                continue
            code = row[where["code"]]
            if not code:
                continue
            frequency = normalise_frequency(cell(row[where["frequency"]]))
            text = cell(row[where["mission_text"]])
            capped = cell(row[where["weekly_cap"]]).lower() == "yes"
            missions.append(
                {
                    "code": str(code).strip(),
                    "pillar": sheet.title,
                    "objectives": ";".join(
                        part.strip() for part in cell(row[where["objectives"]]).split(",") if part.strip()),
                    "mission_class": cell(row[where["mission_class"]]),
                    "frequency": frequency,
                    # Days the Mission asks for, from the frequency; on a capped
                    # Mission, the cap its text states.
                    "k_m": cap_from_text(text) if capped else COUNT_OF_FREQUENCY.get(frequency),
                    "weekly_cap": capped,
                    "schedulable": cell(row[where["schedulable"]]).lower() == "true",
                    "mission_text": text,
                }
            )
    workbook.close()
    return missions


def main():
    rows = [
        {
            "code": mission["code"],
            "pillar": mission["pillar"],
            "objectives": mission["objectives"],
            "class": mission["mission_class"],
            "frequency": mission["frequency"],
            "k_m": "" if mission["k_m"] is None else mission["k_m"],
            "weekly_cap": "yes" if mission["weekly_cap"] else "no",
            "schedulable": "yes" if mission["schedulable"] else "no",
            "mission_text": mission["mission_text"],
        }
        for mission in read_workbook()
    ]
    OUTPUT.parent.mkdir(exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} Missions to {OUTPUT}")


if __name__ == "__main__":
    main()
