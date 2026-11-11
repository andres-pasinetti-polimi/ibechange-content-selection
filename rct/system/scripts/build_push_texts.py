"""Write catalogue/push_texts.csv from the workbook of the pop-up texts of recommendations and resources.

The workbook (../intervention_structure/notification_popup_messages_5_languages_*.xlsx, sheet
"Messages") holds 42 texts, each with an integer ID, a nudge type and its text in English, Italian,
Spanish, Romanian and Catalan. EUT sends the text whose ID the plan gives as push_text_id
(EUT, 26 October 2026). Its sheet "Guidance" asks to rotate the texts and never to send the same ID
twice in a row, which content.choose_push_text applies.

Run from the system/ folder:
    python -m scripts.build_push_texts
"""

import csv
from pathlib import Path

import openpyxl

SYSTEM = Path(__file__).parent.parent
WORKBOOK = next((SYSTEM.parent / "intervention_structure").glob("notification_popup_messages_5_languages_*.xlsx"))
OUTPUT = SYSTEM / "catalogue" / "push_texts.csv"
COLUMNS = {"English": "text_en", "Italian": "text_it", "Spanish": "text_es", "Romanian": "text_ro",
           "Catalan": "text_ca"}


def main():
    sheet = openpyxl.load_workbook(WORKBOOK, data_only=True)["Messages"]
    header, *rows = list(sheet.iter_rows(values_only=True))
    index = {name: header.index(name) for name in ("ID", "Nudge type", *COLUMNS)}
    texts = []
    for row in rows:
        if row[index["ID"]] is None:
            continue
        record = {"push_text_id": int(row[index["ID"]]), "nudge_type": row[index["Nudge type"]]}
        record.update({column: row[index[name]] for name, column in COLUMNS.items()})
        texts.append(record)
    ids = [text["push_text_id"] for text in texts]
    assert len(ids) == len(set(ids)), "duplicate IDs in the workbook"
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(texts[0]))
        writer.writeheader()
        writer.writerows(texts)
    print(f"Wrote {len(texts)} push texts to {OUTPUT}")


if __name__ == "__main__":
    main()
