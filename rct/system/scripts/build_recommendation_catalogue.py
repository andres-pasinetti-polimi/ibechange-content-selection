"""Build catalogue/recommendations_workbook.json from the ICO recommendations workbook (version 2,
28 October 2026), in the format of POST /recommendations ({content_id: {"missions": [...], ...}}).

The file stands in for GET /internal/recommendations while that endpoint does not serve the RCT
recommendations (it returned HTTP 422 on 3 November 2026 and 401 on 4 November 2026). Once VU serves
them, the Cronicle job replaces this catalogue with theirs (POST /recommendations?source=vu).

Rules, as in scripts/build_resource_catalogue.py:
- the content id is the code followed by the variation letter (SRc1a), or the code alone for a
  recommendation without variations, as the ids EUT uses for the RCT (NRc15a, 2 November 2026);
- rows without a code or without a text (the section rows of the workbook, such as "Frutas") are skipped;
- the workbook writes the title, Mission_associated, the COM-B sub-domain, the target and the CACTIS code on
  the first variation of a code only, and the following variations of the same code take them from it;
- Mission_associated "All" gives every Mission of the sheet's pillar in catalogue/missions.csv, ranges
  such as "SM50 - SM62" give every Mission in the range, codes are normalised (SM01 -> SM1);
- rows with a text and no code are not in the catalogue and are reported: in version 2, 21 rows of
  Physical activity below the table (rows 433-448, and the variations b to d of rows 459-461 and
  465-467), with no code, title or Mission;
- a content id written twice is kept once, and the script reports whether the two rows differ: in
  version 2, PRc21a and PRc22a are written again below the table (rows 458 and 464, with no title or
  Mission and with other texts), and the first rows are kept.

Run from the system/ folder:
    python -m scripts.build_recommendation_catalogue
"""

import csv
import json
from pathlib import Path

import openpyxl

from scripts.build_resource_catalogue import PILLAR_OF_SHEET, expand

SYSTEM_FOLDER = Path(__file__).parent.parent
WORKBOOK = SYSTEM_FOLDER.parent / "intervention_structure" / "RCT_Recommendations_v2_28Oct2026.xlsx"
MISSIONS = SYSTEM_FOLDER / "catalogue" / "missions.csv"
OUTPUT = SYSTEM_FOLDER / "catalogue" / "recommendations_workbook.json"


def build():
    with open(MISSIONS, newline="") as file:
        missions = list(csv.DictReader(file))
    known = {row["code"] for row in missions}
    missions_of_pillar = {}
    for row in missions:
        missions_of_pillar.setdefault(row["pillar"], []).append(row["code"])

    workbook = openpyxl.load_workbook(WORKBOOK, read_only=True)
    recommendations, problems = {}, []
    for sheet in workbook:
        rows = list(sheet.iter_rows(values_only=True))
        header = [str(c).strip() if c else "" for c in rows[0]]
        column = {name: header.index(name) for name in header if name}
        pillar = PILLAR_OF_SHEET[sheet.title]
        shared_columns = ["Recommendation_title", "Mission_associated", "COM-B Subdomain", "Target_BC_factor",
                          "Cactis_Code"]
        first_row_of = {}  # code -> the values of shared_columns on its first row
        for row in rows[1:]:
            code, text = row[column["Code"]], row[column["Recommendation"]]
            if not text or not str(text).strip():
                continue
            if not code:
                problems.append(f"{sheet.title}: a text with no code, not in the catalogue: {str(text)[:60]!r}")
                continue
            code = str(code).strip()
            row = list(row)
            if code in first_row_of:
                for name in shared_columns:
                    if row[column[name]] is None or not str(row[column[name]]).strip():
                        row[column[name]] = first_row_of[code][name]
            else:
                first_row_of[code] = {name: row[column[name]] for name in shared_columns}
            variation = str(row[column["Variation"]] or "").strip()
            content_id = code + variation
            found, unread = expand(row[column["Mission_associated"]], pillar, missions_of_pillar, known)
            if unread:
                problems.append(f"{content_id}: not in the Missions catalogue: {unread}")
            if not found:
                problems.append(f"{content_id}: no Mission")
            item = {
                "missions": found,
                "code": code,
                "type": "recommendation",
                "pillar": pillar,
                "title": row[column["Recommendation_title"]],
                "text": str(text).strip(),
                "com_b": row[column["COM-B Subdomain"]],
                "target": row[column["Target_BC_factor"]],
                "cactis_code": row[column["Cactis_Code"]],
            }
            if content_id in recommendations:
                same = recommendations[content_id] == item
                problems.append(f"{content_id}: written twice, {'same row' if same else 'rows differ'}; the first is kept")
                continue
            recommendations[content_id] = item
    OUTPUT.write_text(json.dumps(recommendations, ensure_ascii=False, indent=1, default=str) + "\n")
    print(f"{len(recommendations)} recommendations written to {OUTPUT.relative_to(SYSTEM_FOLDER)}")
    for line in problems:
        print(line)
    return recommendations


if __name__ == "__main__":
    build()
