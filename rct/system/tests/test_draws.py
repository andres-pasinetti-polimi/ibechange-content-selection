"""The table draws (schema.sql): every delivery carries the draws behind it, with the probability of
each admissible option, so that the probability of a plan can be computed after the trial whatever
the policy that decided it (rct_logging_specification.md, 1.10)."""

import math
from datetime import datetime, timedelta

import pytest

from cs_rct import catalogue as catalogue_module
from cs_rct import composition, model
from cs_rct.decide import decide
from tests.helpers import START, episode, load_catalogue, participant, scheduled

PRIOR = model.prior()


def catalogue(with_recommendations=True):
    cat = load_catalogue(items_per_mission=0)
    if with_recommendations:
        catalogue_module.set_recommendations(cat, {
            f"R-{m}-{j}{v}": {"missions": [m], "com_b": "Reflective Motivation" if j % 2 else "Physical Opportunity"}
            for m in cat.missions for j in range(4) for v in "ab"})
        cat.resources_of = {m: [f"RS-{m}-{j}" for j in range(2)] for m in cat.missions}
    return cat


def plans(cat, missions=("PM3", "SM1", "EM1"), seeds=range(30), days=3):
    """Plans of consecutive days, each reading the plans of the earlier days."""
    for mission in missions:
        for seed in seeds:
            day = START.date() + timedelta(days=1)
            slot = datetime.combine(day, datetime.min.time()) + timedelta(hours=18)
            person = participant([episode(mission_id=mission)],
                                 scheduled_tasks=[scheduled(slot)] if seed % 3 == 0 else [])
            previous = []
            for k in range(days):
                d = day + timedelta(days=k)
                now = datetime.combine(d, datetime.min.time()) + timedelta(hours=1)
                result = decide(person, cat, PRIOR, 1, "learned", d, now, seed * 10 + k, list(previous), "usability")
                if result is None:
                    continue
                yield result
                previous += [{**x, "decision_day": d} for x in result[1]]


def chosen_probability(draws):
    return math.prod(x["options"][x["chosen"]] for x in draws)


def test_every_draw_is_a_distribution_with_its_chosen_option():
    for row, rows in plans(catalogue()):
        for draw in row["draws"] + [x for d in rows for x in d["draws"]]:
            assert sum(draw["options"].values()) == pytest.approx(1.0)
            assert draw["chosen"] in draw["options"]
            assert not draw["forced"] or draw["options"] == {draw["chosen"]: 1.0}


def test_draws_give_the_logged_probabilities():
    for row, rows in plans(catalogue()):
        [day_draw] = row["draws"]
        assert day_draw["kind"] == "composition" and day_draw["options"] == row["composition_probabilities"]
        assert day_draw["chosen"] == composition.key((row["composition_r"], row["composition_q"]))
        for delivery in rows:
            content = [x for x in delivery["draws"] if x["kind"] not in ("push_text", "order")]
            assert chosen_probability(content) == pytest.approx(delivery["content_probability"])
            push = [x for x in delivery["draws"] if x["kind"] == "push_text"]
            assert chosen_probability(push) == pytest.approx(delivery["push_text_probability"] or 1.0)


def test_without_recommendations_every_composition_has_no_recommendation():
    for row, rows in plans(catalogue(with_recommendations=False), seeds=range(5)):
        if row["composition_r"] is not None:
            assert row["composition_r"] == 0
        assert {d["format"] for d in rows} <= {"reminder"}


def test_a_composition_not_drawn_is_forced():
    marked = [row for row, _ in plans(catalogue(), missions=("PM3",), seeds=range(3), days=1)]
    assert all(not row["draws"][0]["forced"] for row in marked if row["status"] == "decided")


def test_the_same_seed_gives_the_same_draws():
    first = [(row["draws"], [d["draws"] for d in rows]) for row, rows in plans(catalogue(), seeds=range(5))]
    second = [(row["draws"], [d["draws"] for d in rows]) for row, rows in plans(catalogue(), seeds=range(5))]
    assert first == second


def test_catalogue_version_names_every_file_read():
    cat = catalogue_module.load({"R1": {"missions": ["PM3"]}}, {"RS1": {"missions": ["PM3"]}})
    assert {"missions.csv", "reminder_pools.csv", "reminders.csv", "recommendations", "resources"} <= set(cat.version)
    other = catalogue_module.load({"R1": {"missions": ["PM4"]}})
    assert other.version["recommendations"] != cat.version["recommendations"]
