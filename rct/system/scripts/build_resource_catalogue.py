"""Build catalogue/resources_workbook.json from the ICO resources workbook, in the format of
POST /resources ({content_id: {"missions": [...], ...}}).

The file stands in for GET /internal/recommendations?resources=true while EUT has not loaded the
RCT resources (the endpoint returned an empty list on 3 November 2026). Once EUT serves them, the
Cronicle job "catalogue" replaces this catalogue with theirs.

The column Mission_associated is expanded as follows:
- "All" gives every Mission of the sheet's pillar in catalogue/missions.csv;
- "SM01 - SM49" and "PM1-PM93" give every Mission of the catalogue in the range;
- single codes are kept; codes are normalised as in catalogue.mission_code (SM01 -> SM1);
- an empty cell gives no Missions, and the service never sends that resource.

One entry per resource code, read from its Spanish row; the attributes other than "missions" are
copied for reference and the service does not use them.

Run from the system/ folder:
    python -m scripts.build_resource_catalogue
"""

import csv
import json
import re
from pathlib import Path

import openpyxl

from cs_rct.catalogue import mission_code

SYSTEM_FOLDER = Path(__file__).parent.parent
WORKBOOK = (SYSTEM_FOLDER.parent / "intervention_structure" / "resources_v1_03Nov2026"
            / "RCT_Resources_v1_03Nov2026.xlsx")
MISSIONS = SYSTEM_FOLDER / "catalogue" / "missions.csv"
OUTPUT = SYSTEM_FOLDER / "catalogue" / "resources_workbook.json"

# Sheet of the workbook -> pillar of catalogue/missions.csv.
PILLAR_OF_SHEET = {"Smoking": "Smoking", "Alcohol": "Alcohol", "Nutrition": "Nutrition",
                   "Physical_activity": "Physical_activity", "Emotional_wellbeing": "Emotional_Wellbeing"}
RANGE = re.compile(r"([A-Z]+M)0*(\d+)\s*-\s*([A-Z]+M)0*(\d+)")


def expand(cell, pillar, missions_of_pillar, known):
    """The Missions named by a Mission_associated cell, in catalogue order, and the parts of the
    cell that name no Mission of the catalogue."""
    if cell is None or not str(cell).strip():
        return [], []
    if str(cell).strip().lower() == "all":
        return list(missions_of_pillar[pillar]), []
    found, unread = [], []
    for part in str(cell).split(","):
        part = part.strip()
        if not part:
            continue
        match = RANGE.fullmatch(part)
        if match and match.group(1) == match.group(3):
            prefix, first, last = match.group(1), int(match.group(2)), int(match.group(4))
            codes = [f"{prefix}{n}" for n in range(first, last + 1)]
        else:
            codes = [mission_code(part)]
        for code in codes:
            if code in known:
                found.append(code)
            else:
                unread.append(code)
    return list(dict.fromkeys(found)), unread


def build():
    with open(MISSIONS, newline="") as file:
        missions = list(csv.DictReader(file))
    known = {row["code"] for row in missions}
    missions_of_pillar = {}
    for row in missions:
        missions_of_pillar.setdefault(row["pillar"], []).append(row["code"])

    workbook = openpyxl.load_workbook(WORKBOOK, read_only=True)
    resources, problems = {}, []
    for sheet in workbook:
        rows = list(sheet.iter_rows(values_only=True))
        header = [str(c).strip() if c else "" for c in rows[0]]
        column = {name: header.index(name) for name in header if name}
        pillar = PILLAR_OF_SHEET[sheet.title]
        for row in rows[1:]:
            code = row[column["Code"]]
            if not code or row[column["Language"]] != "SPA":
                continue
            code = str(code).strip()
            found, unread = expand(row[column["Mission_associated"]], pillar, missions_of_pillar, known)
            if unread:
                problems.append(f"{code}: not in the Missions catalogue: {unread}")
            if not found:
                problems.append(f"{code}: no Mission")
            resources[code] = {
                "missions": found,
                "type": "resource",
                "pillar": pillar,
                "title": row[column["Resource_title"]],
                "resource_type": row[column["Resource_type"]],
                "com_b": row[column["COM-B_Subdomain"]],
                "cactis_code": row[column["Cactis_code"]],
            }
    OUTPUT.write_text(json.dumps(resources, ensure_ascii=False, indent=1, default=str) + "\n")
    print(f"{len(resources)} resources written to {OUTPUT.relative_to(SYSTEM_FOLDER)}")
    for line in problems:
        print(line)
    return resources


if __name__ == "__main__":
    build()
