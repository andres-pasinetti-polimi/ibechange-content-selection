"""Write catalogue/step3_options.csv from the Step 3 questionnaires of EUT.

The questionnaires (../intervention_structure/step3_questionnaires/*.json, sent by EUT for the
psychological_assessments message) list, for every question, its answer options; the field
data.answer of an option (data.data in the version of 29 September 2026) carries the clinical tag
of the answer: a HAPA stage ("pre-intender", "intender", "actor") in the first question of Smoking
and Alcohol, and one or more COM-B sub-domains ("social_opportunity,reflective_motivation") in the
others. The messages carry the same tags in their answers; this table is the fallback for answers
that arrive without one, which in the messages received up to 29 October 2026 were those of
Smoking and Alcohol.

One row per option: option_id (the value the answer carries), questionnaire, pillar_id,
question_code, tag (as written), hapa_stage and subdomains (normalised, space separated by ";").

Run from the system/ folder:
    python -m scripts.build_step3_options
"""

import csv
import json
from pathlib import Path

from cs_rct.catalogue import hapa_stage_name, subdomain_names

SYSTEM = Path(__file__).parent.parent
FOLDER = SYSTEM.parent / "intervention_structure" / "step3_questionnaires"
OUTPUT = SYSTEM / "catalogue" / "step3_options.csv"
# The pillar of a questionnaire, from its code (metadata.pillar is missing in three of them).
PILLAR_OF_CODE = {"smoking": "SO", "alcohol": "AO", "nutrition": "NO", "physical-activity": "PO",
                  "psychosocial": "EO"}


def main():
    rows = []
    for path in sorted(FOLDER.glob("*.json")):
        questionnaire = json.loads(path.read_text())
        code = questionnaire["code"]
        pillar = next(pillar for key, pillar in PILLAR_OF_CODE.items() if key in code)
        for section in questionnaire["sections"]:
            for question in section["questions"]:
                for option in (question.get("constraint") or {}).get("values") or []:
                    data = option.get("data") or {}
                    tag = data.get("answer") or data.get("data") or ""
                    rows.append({"option_id": option["value"], "questionnaire": code, "pillar_id": pillar,
                                 "question_code": question["code"], "tag": tag,
                                 "hapa_stage": hapa_stage_name(tag) or "",
                                 "subdomains": ";".join(sorted(subdomain_names(tag)))})
    # An option id may appear in more than one questionnaire: in the version of 30 October 2026,
    # 6 ids are shared by the parallel questions of Nutrition fruits-vegetables and
    # legumes-whole-grains-nuts. The service reads an option by its id, so such an id must carry
    # the same tag and stage everywhere.
    by_id = {}
    for row in rows:
        by_id.setdefault(row["option_id"], set()).add((row["tag"], row["hapa_stage"]))
    conflicting = sorted(item for item, tags in by_id.items() if len(tags) > 1)
    assert not conflicting, f"option ids with different tags: {conflicting}"
    repeated = sum(1 for item in by_id if sum(row["option_id"] == item for row in rows) > 1)
    if repeated:
        print(f"Option ids that appear more than once, with the same tag: {repeated}")
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tagged = sum(1 for row in rows if row["tag"])
    print(f"Wrote {len(rows)} options ({tagged} tagged) of {len(list(FOLDER.glob('*.json')))} questionnaires "
          f"to {OUTPUT}")


if __name__ == "__main__":
    main()
