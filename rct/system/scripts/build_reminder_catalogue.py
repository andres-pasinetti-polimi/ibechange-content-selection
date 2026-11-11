"""Write catalogue/reminders.csv from the reminders of version 3 (27 October 2026).

The service sends reminder ids that EUT resolves to texts, so its catalogue must hold the same reminders
as EUT. EUT loaded version 3 (GET /internal/reminders on INT, 2 November 2026: 251 reminders, ids, English
texts and Missions equal to RCT_reminders_v3_27Oct2026.xlsx). The source here is the CSV written with that
Excel by ../reminders/build_reminders_v3.py; the script checks that ids and English texts match the Excel.

Columns, as cs_rct/catalogue.py reads them: reminder_id, kind ("day" or "hour"), pools (the pool of a day
reminder: Objective, or group in Emotional Wellbeing), missions (the Missions of an hour reminder), bct,
the texts in four languages and latest_time. No text of version 3 names a time of day, so latest_time is
empty for every reminder. The pools of the Missions (catalogue/reminder_pools.csv) are those of version 2,
which version 3 keeps (22 pools, the same Missions).

Run from the system/ folder:
    python -m scripts.build_reminder_catalogue
"""

import csv
import re
from pathlib import Path

import openpyxl

SYSTEM = Path(__file__).parent.parent
SOURCE = SYSTEM.parent / "reminders" / "v3_output" / "reminders_v3.csv"
EXCEL = SYSTEM.parent / "intervention_structure" / "RCT_reminders_v3_27Oct2026.xlsx"
OUTPUT = SYSTEM / "catalogue" / "reminders.csv"
POOLS = SYSTEM / "catalogue" / "reminder_pools.csv"


def expand(missions):
    """'SM14-15 SM22' -> ['SM14', 'SM15', 'SM22']"""
    codes = []
    for token in missions.split():
        match = re.fullmatch(r"([A-Z]+)(\d+)(?:-(\d+))?", token)
        assert match, token
        first, last = int(match.group(2)), int(match.group(3) or match.group(2))
        codes += [f"{match.group(1)}{number}" for number in range(first, last + 1)]
    return codes


def main():
    rows = list(csv.DictReader(SOURCE.open(newline="", encoding="utf-8")))
    excel = {}
    for sheet in openpyxl.load_workbook(EXCEL, read_only=True).worksheets:
        header = None
        for values in sheet.iter_rows(values_only=True):
            if header is None:
                header = list(values)
                continue
            row = dict(zip(header, values))
            if row.get("ID"):
                excel[row["ID"]] = str(row["English"]).strip()
    assert {row["reminder_id"] for row in rows} == set(excel), "ids differ from the Excel"
    different = [row["reminder_id"] for row in rows if row["text_en"].strip() != excel[row["reminder_id"]]]
    assert not different, f"English texts differ from the Excel: {different}"

    pools = {pool for row in csv.DictReader(POOLS.open(newline="")) for pool in row["pools"].split()}
    output = []
    for row in rows:
        day = row["kind"] == "day"
        assert not day or row["pool"] in pools, row["reminder_id"]
        output.append({"reminder_id": row["reminder_id"], "kind": row["kind"],
                       "pools": row["pool"] if day else "",
                       "missions": "" if day else " ".join(expand(row["missions"])),
                       "bct": row["code"], "text_es": row["text_es"], "text_en": row["text_en"],
                       "text_it": row["text_it"], "text_ro": row["text_ro"], "latest_time": ""})
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    days = sum(row["kind"] == "day" for row in output)
    print(f"Wrote {len(output)} reminders ({days} day, {len(output) - days} hour) to {OUTPUT}")


if __name__ == "__main__":
    main()
