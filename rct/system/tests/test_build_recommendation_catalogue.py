"""catalogue/recommendations_workbook.json, built from the ICO workbook of 28 October 2026."""

import json

from cs_rct import catalogue
import pytest

from scripts.build_recommendation_catalogue import OUTPUT, WORKBOOK


# The clinical workbook and the catalogue built from it are not distributed with the public code.
@pytest.mark.skipif(not WORKBOOK.exists(), reason="clinical workbook not available")
def test_the_committed_catalogue_matches_the_workbook_rules():
    items = json.loads(OUTPUT.read_text())
    missions = catalogue.load().missions
    assert len(items) == 2142
    # Variations take the Missions of the first row of their code.
    assert items["SRc7b"]["missions"] == items["SRc7a"]["missions"] and items["SRc7b"]["missions"]
    assert items["SRc7e"]["com_b"] == "Psychological Capability"
    # "EM 31" read as EM31, every Mission named is in the catalogue.
    assert "EM31" in items["ERc122a"]["missions"]
    assert all(m in missions for item in items.values() for m in item["missions"])
    # The only recommendation with no Mission, a question to ICO.
    assert sorted(k for k, v in items.items() if not v["missions"]) == [f"PRc56{v}" for v in "abcde"]
    # The rows below the table do not replace the first PRc21a.
    assert items["PRc21a"]["title"] == "Encuentra tu ritmo"
