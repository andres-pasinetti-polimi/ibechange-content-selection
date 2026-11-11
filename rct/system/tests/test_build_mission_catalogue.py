from scripts.build_mission_catalogue import (
    COUNT_OF_FREQUENCY,
    cap_from_text,
    normalise_frequency,
    read_workbook,
)
from scripts.build_mission_catalogue import WORKBOOK

import pytest

# The clinical workbook is not distributed with the public code.
needs_workbook = pytest.mark.skipif(not WORKBOOK.exists(), reason="clinical workbook not available")


def test_frequency_spellings_are_merged():
    assert normalise_frequency("Daily ") == "daily"
    assert normalise_frequency("No-schedulable") == "non-schedulable"
    assert normalise_frequency("when possible") == "when-possible"


def test_cap_is_read_from_the_text():
    assert cap_from_text("Bebe como máximo 4 bebidas con alcohol a la semana") == 4
    assert cap_from_text("Limita la carne roja a un máximo de 3 porciones") == 3


@needs_workbook
def test_every_class_a_mission_states_a_count():
    # Completing a class A Mission requires its count of marks, so the count has
    # to be there for every one of them.
    for mission in read_workbook():
        if mission["mission_class"] == "A":
            assert mission["frequency"] in COUNT_OF_FREQUENCY, mission["code"]
            assert mission["k_m"], mission["code"]


@needs_workbook
def test_a_mission_states_a_count_or_a_frequency_word():
    words = {"daily", "non-schedulable", "when-possible"}
    for mission in read_workbook():
        assert mission["frequency"] in COUNT_OF_FREQUENCY or mission["frequency"] in words, (
            mission["code"], mission["frequency"])


@needs_workbook
def test_a_capped_mission_states_its_cap():
    for mission in read_workbook():
        if mission["weekly_cap"]:
            assert mission["k_m"] is not None, mission["code"]
