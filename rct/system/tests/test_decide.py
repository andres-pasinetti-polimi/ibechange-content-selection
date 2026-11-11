from datetime import datetime, timedelta

import pytest

from cs_rct import composition
from cs_rct import decide as decide_module
from cs_rct import design, model
from cs_rct.decide import decide
from tests.helpers import START, episode, force_composition, load_catalogue, mark, participant, scheduled

CATALOGUE = load_catalogue()
DAY2 = START.date() + timedelta(days=1)
MORNING = datetime.combine(DAY2, datetime.min.time()) + timedelta(hours=1)
PRIOR = model.prior()


def run(person, posterior=PRIOR, group="learned", day=DAY2, now=MORNING, previous=(), seed=7):
    return decide(person, CATALOGUE, posterior, 1, group, day, now, seed, list(previous), "usability")


def test_no_open_episode_gives_no_row():
    assert run(participant([])) is None


def test_decision_point_with_scheduled_task_sends_the_hour_reminder(monkeypatch):
    force_composition(monkeypatch, (0, 3))
    slot = datetime.combine(DAY2, datetime.min.time()) + timedelta(hours=18)
    person = participant([episode(mission_id="PM3")], scheduled_tasks=[scheduled(slot)])
    row, deliveries = run(person)
    assert row["status"] == "decided" and (row["composition_r"], row["composition_q"]) == (0, 3)
    assert row["composition_probabilities"]["0,3"] == 1.0 and row["state"]["scheduled_today"] == 1.0
    assert [d["role"] for d in deliveries] == ["reminder"] * 3
    # The reminder at the declared hour goes last; the other deliveries are in a drawn order.
    hour = deliveries[-1]
    assert hour["reminder_kind"] == "hour" and hour["declared_at"] == slot and hour["content_id"] == "PRh1"
    assert [d["reminder_kind"] for d in deliveries[:-1]] == ["day", "day"]


def test_same_seed_gives_the_same_draws_and_content():
    person = participant([episode(mission_id="PM3")])
    first, second = run(person, seed=123), run(person, seed=123)
    assert first[0]["composition_probabilities"] == second[0]["composition_probabilities"]
    assert (first[0]["composition_r"], first[0]["composition_q"]) == (second[0]["composition_r"], second[0]["composition_q"])
    assert [d["content_id"] for d in first[1]] == [d["content_id"] for d in second[1]]


def test_count_reached_leaves_two_recommendations_and_no_choice():
    marks = [mark(datetime.combine(START.date(), datetime.min.time()), START + timedelta(hours=3))]
    person = participant([episode(mission_id="PM1")], marks)  # once a week
    row, deliveries = run(person)
    assert row["status"] == "no_choice" and row["blocked_by"] == "count_reached"
    assert row["composition_probabilities"] == {"2,0": 1.0} and row["policy_rule"] is None
    assert set(row["removed_compositions"].values()) == {"count_reached"}
    assert [d["format"] for d in deliveries] == ["recommendation", "recommendation"]


def test_task_already_marked_today_leaves_two_recommendations():
    marks = [mark(datetime.combine(DAY2, datetime.min.time()), MORNING - timedelta(minutes=30))]
    person = participant([episode(mission_id="PM3")], marks)
    row, deliveries = run(person)
    assert row["status"] == "no_choice" and row["blocked_by"] == "task_marked"
    assert [d["role"] for d in deliveries] == ["recommendation", "recommendation"]


def test_a_marked_task_in_emotional_wellbeing_is_below_the_minimum():
    marks = [mark(datetime.combine(DAY2, datetime.min.time()), MORNING - timedelta(minutes=30))]
    person = participant([episode(mission_id="EM1")], marks)
    row, deliveries = run(person)
    assert row["status"] == "below_minimum" and row["composition_r"] is None
    assert [d["format"] for d in deliveries] == ["recommendation"]


def test_after_the_window_nothing_is_sent():
    day = START.date() + timedelta(days=design.WINDOW_DAYS)
    now = datetime.combine(day, datetime.min.time()) + timedelta(hours=1)
    row, deliveries = run(participant([episode(mission_id="PM3")]), day=day, now=now)
    assert row["status"] == "past_window" and row["composition_r"] is None
    assert row["composition_probabilities"] is None and deliveries == []


def test_weekly_cap_removes_the_compositions_of_four():
    previous = [{"decision_day": DAY2 - timedelta(days=1 + i % 6), "content_id": f"X{i}", "format": "reminder",
                 "position": i} for i in range(design.WEEKLY_CAP - 3)]
    row, _ = run(participant([episode(mission_id="PM3")]), previous=previous)
    assert row["removed_compositions"] == {"2,2": "weekly_cap", "1,3": "weekly_cap", "0,4": "weekly_cap"}
    assert row["status"] == "decided" and len(row["composition_probabilities"]) == 6


def test_without_recommendations_only_reminders_are_admissible():
    empty = load_catalogue(items_per_mission=0)
    row, deliveries = decide(participant([episode(mission_id="PM3")]), empty, PRIOR, 1, "learned", DAY2, MORNING,
                             7, [], "usability")
    assert set(row["composition_probabilities"]) == {"0,2", "0,3", "0,4"}
    assert set(row["removed_compositions"].values()) == {"no_recommendation"}
    # The default (1,1) is not admissible: its share goes to (0,2), the fallback.
    assert row["composition_probabilities"]["0,2"] == pytest.approx(1 - design.EXPLORE_SHARE + design.EXPLORE_SHARE / 3)
    assert {d["format"] for d in deliveries} == {"reminder"}


def test_one_recommendation_left_removes_the_compositions_with_two():
    one_item = load_catalogue(items_per_mission=1)
    row, _ = decide(participant([episode(mission_id="PM3")]), one_item, PRIOR, 1, "learned", DAY2, MORNING, 7, [],
                    "usability")
    assert {k for k, rule in row["removed_compositions"].items()} == {"2,0", "2,1", "2,2"}
    assert set(row["removed_compositions"].values()) == {"recommendations_available"}


def test_the_conservative_rule_starts_at_the_default():
    row, _ = run(participant([episode(mission_id="PM3")]))
    probabilities = row["composition_probabilities"]
    assert row["policy_rule"] == "conservative" and len(probabilities) == 9
    assert probabilities["1,1"] == pytest.approx(1 - design.EXPLORE_SHARE + design.EXPLORE_SHARE / 9)
    assert sum(probabilities.values()) == pytest.approx(1.0)


def test_emotional_wellbeing_admits_six_compositions():
    row, _ = run(participant([episode(mission_id="EM1")]))
    assert set(row["composition_probabilities"]) == {"1,1", "0,2", "1,2", "0,3", "1,3", "0,4"}


def test_fixed_group_draws_from_the_starting_probabilities():
    row, _ = run(participant([episode(mission_id="PM3")]), group="fixed")
    assert row["policy_rule"] == "fixed"
    assert row["composition_probabilities"]["1,1"] == pytest.approx(1 - design.EXPLORE_SHARE + design.EXPLORE_SHARE / 9)


def test_missing_model_falls_back_to_the_starting_probabilities():
    row, _ = run(participant([episode(mission_id="PM3")]), posterior=None)
    assert row["fallback"] and row["policy_rule"] == "fallback"
    assert row["composition_probabilities"]["1,1"] == pytest.approx(1 - design.EXPLORE_SHARE + design.EXPLORE_SHARE / 9)


def test_uniform_policy(monkeypatch):
    monkeypatch.setattr(design, "POLICY", "uniform")
    row, _ = run(participant([episode(mission_id="PM3")]), posterior=None)
    assert row["policy_rule"] == "uniform" and not row["fallback"]
    assert list(row["composition_probabilities"].values()) == [pytest.approx(1 / 9)] * 9


def test_capped_mission_states_no_count():
    row, deliveries = run(participant([episode(mission_id="NM28")]))
    assert row["status"] == "decided" and row["state"]["deficit"] == 0.0


def test_first_delivery_day_option(monkeypatch):
    monkeypatch.setattr(design, "FIRST_DELIVERY_DAY", "selection_day")
    person = participant([episode(mission_id="PM3")])
    selection_day = START.date()
    later = START + timedelta(hours=1)
    assert run(person, day=selection_day, now=later) is not None
    monkeypatch.setattr(design, "FIRST_DELIVERY_DAY", "next_day")
    assert run(person, day=selection_day, now=later) is None
    assert run(person)[0]["status"] == "decided"  # the morning after
    from cs_rct.decide import closing_status
    assert closing_status(person, selection_day, later + timedelta(days=1)) == "selection_day"


def test_baseline_reads_the_latest_hhs_of_the_mission_pillar():
    def assessment(pillar, score, at):
        return {"user_uuid": "u1", "pillar_id": pillar, "score": score, "colour": None, "trend": None,
                "subpillars": {}, "assessed_at": at, "received_at": at}
    person = participant([episode(mission_id="PM3")])
    person.health = [assessment("PO", 40.0, START - timedelta(days=3)), assessment("PO", 55.0, START),
                     assessment("NO", 90.0, START), assessment("PO", 70.0, MORNING + timedelta(hours=1))]
    row, _ = run(person)
    assert row["baseline"]["hhs_pillar"] == 0.55 and row["baseline"]["hhs_missing"] == 0.0
    person.health = []
    row, _ = run(person)
    assert row["baseline"]["hhs_missing"] == 1.0


def test_the_first_plan_of_a_mission_carries_a_resource_never_repeated(monkeypatch):
    force_composition(monkeypatch, (1, 1))
    from cs_rct import catalogue as catalogue_module
    cat = load_catalogue()
    cat.resources_of = catalogue_module.by_mission({"RS1": {"missions": ["PM3"]}})
    person = participant([episode(mission_id="PM3")])
    first = decide(person, cat, PRIOR, 1, "learned", DAY2, MORNING, 7, [], "usability")[1]
    [resource] = [d for d in first if d["role"] == "resource"]
    assert resource["content_id"] == "RS1" and resource["content_probability"] == 1.0
    assert sorted(d["role"] for d in first) == ["reminder", "resource"]
    # The next day of the same Mission: the recommendation again.
    nxt = decide(person, cat, PRIOR, 1, "learned", DAY2 + timedelta(days=1), MORNING + timedelta(days=1), 7,
                 first, "usability")[1]
    assert "recommendation" in [d["role"] for d in nxt] and "resource" not in [d["role"] for d in nxt]
    # A new Mission whose only resource was already received: the recommendation.
    second = participant([episode(2, mission_id="PM3", selected_at=START + timedelta(days=8))])
    later = decide(second, cat, PRIOR, 1, "learned", DAY2 + timedelta(days=8), MORNING + timedelta(days=8), 7,
                   first, "usability")[1]
    assert later[0]["role"] == "recommendation"


def test_a_reminder_with_a_latest_time_is_due_by_it():
    from cs_rct.decide import earliest
    from cs_rct.catalogue import Reminder
    from datetime import time
    evening = Reminder("NRd4", "day", ("NO1",), (), "3215", {}, time(19, 0))
    plain = Reminder("NRd1", "day", ("NO1",), (), "3144", {}, None)
    task = datetime.combine(DAY2, datetime.min.time()) + timedelta(hours=18)
    assert earliest(None, DAY2, evening) == datetime.combine(DAY2, time(19, 0))
    assert earliest(task, DAY2, evening) == task
    assert earliest(task + timedelta(hours=2), DAY2, evening) == datetime.combine(DAY2, time(19, 0))
    assert earliest(None, DAY2, plain) is None


def test_latest_time_is_the_limit_of_the_reminder_text_only():
    # No reminder of version 3 has a latest time, so one is given to PRd5 of PO1, the pool of PM3.
    import copy
    from dataclasses import replace
    from datetime import time
    cat = copy.deepcopy(CATALOGUE)
    limited = replace(cat.reminders["PRd5"], latest_time=time(19, 0))
    cat.reminders["PRd5"] = limited
    cat.day_reminders = {pool: [limited if r.reminder_id == "PRd5" else r for r in reminders]
                         for pool, reminders in cat.day_reminders.items()}
    slot = datetime.combine(DAY2, datetime.min.time()) + timedelta(hours=18)
    person = participant([episode(mission_id="PM3")], scheduled_tasks=[scheduled(slot)])
    seen_limit = False
    for seed in range(200):
        _, deliveries = decide(person, cat, PRIOR, 1, "learned", DAY2, MORNING, seed, [], "usability")
        for d in deliveries:
            assert d["task_scheduled_for"] == slot
            reminder = cat.reminders.get(d["content_id"])
            if d["format"] == "reminder" and reminder.kind == "day" and reminder.latest_time is not None:
                assert d["latest_time"] == datetime.combine(DAY2, reminder.latest_time)
                seen_limit = True
            else:
                assert d["latest_time"] is None
    assert seen_limit


def test_every_admissible_composition_is_drawn_and_delivered(monkeypatch):
    monkeypatch.setattr(design, "POLICY", "uniform")
    for mission, expected in (("PM3", 9), ("EM1", 6)):
        person = participant([episode(mission_id=mission)])
        seen = set()
        for seed in range(300):
            row, deliveries = run(person, seed=seed)
            chosen = (row["composition_r"], row["composition_q"])
            seen.add(chosen)
            formats = [d["format"] for d in deliveries]
            assert (formats.count("recommendation") + formats.count("resource"), formats.count("reminder")) == chosen
        assert len(seen) == expected and seen == set(composition.compositions(CATALOGUE.missions[mission].pillar))


def test_a_recommendation_is_not_repeated_within_the_day(monkeypatch):
    force_composition(monkeypatch, (2, 0))
    one_item = load_catalogue(items_per_mission=1)
    person = participant([episode(mission_id="PM3")])
    for seed in range(50):
        _, deliveries = decide(person, one_item, PRIOR, 1, "learned", DAY2, MORNING, seed, [], "usability")
        assert [d["format"] for d in deliveries].count("recommendation") == 1


def test_a_reminder_text_waits_at_least_one_day(monkeypatch):
    force_composition(monkeypatch, (0, 4))
    person = participant([episode(mission_id="PM3")])
    pool = [r.reminder_id for r in CATALOGUE.day_reminder_pool("PM3")]

    def sent(texts, days_ago):
        return [{"decision_day": DAY2 - timedelta(days=days_ago), "content_id": text, "format": "reminder",
                 "mission_process_id": 1, "position": i} for i, text in enumerate(texts)]

    for seed in range(30):
        # Four texts sent yesterday, the most a day carries, leave four of the eight, enough for the day:
        # none of the four is drawn.
        _, deliveries = run(person, seed=seed, previous=sent(pool[:4], 1))
        assert not {d["content_id"] for d in deliveries} & set(pool[:4])
    # Texts sent two days ago may return.
    seen = set()
    for seed in range(30):
        _, deliveries = run(person, seed=seed, previous=sent(pool, 2))
        seen |= {d["content_id"] for d in deliveries}
    assert seen & set(pool)


def test_each_recommendation_carries_a_push_text_never_the_previous_one(monkeypatch):
    force_composition(monkeypatch, (2, 2))
    assert len(CATALOGUE.push_texts) == 42
    person = participant([episode(mission_id="PM3")])
    yesterday = [{"decision_day": DAY2 - timedelta(days=1), "content_id": "R-PM3-0", "format": "recommendation",
                  "mission_process_id": 1, "position": 0, "push_text_id": 7}]
    for seed in range(50):
        _, deliveries = run(person, seed=seed, previous=yesterday)
        shown = [d["push_text_id"] for d in deliveries if d["format"] == "recommendation"]
        assert shown and None not in shown and 7 not in shown and len(set(shown)) == len(shown)
        # The push texts are drawn before the order, in the order the recommendations were drawn.
        probabilities = sorted(d["push_text_probability"] for d in deliveries if d["format"] == "recommendation")
        assert probabilities == sorted(1 / (42 - 1 - i) for i in range(len(probabilities)))
        assert all(d["push_text_id"] is None for d in deliveries if d["format"] == "reminder")


def test_priority_follows_the_hapa_stage_in_smoking_and_the_barrier_elsewhere():
    from cs_rct import catalogue
    cat = catalogue.load()
    catalogue.set_recommendations(cat, {
        "SRc1a": {"missions": ["SM1"], "com_b": "Reflective Motivation"},
        "SRc1b": {"missions": ["SM1"], "com_b": "Reflective Motivation"},
        "SRc2": {"missions": ["SM1"], "com_b": "Physical Oportunity"},
        "PRc1": {"missions": ["PM3"], "com_b": "psychological_capability, physical_opportunity"},
        "PRc2": {"missions": ["PM3"], "com_b": "Social Opportunity"},
    })
    assert cat.recommendation_code["SRc1b"] == "SRc1" and cat.recommendation_code["SRc2"] == "SRc2"
    person = participant([episode(mission_id="SM1")])

    def answered(pillar_id, barriers, stage, day, subpillar=None):
        moment = MORNING - timedelta(days=day)
        return {"pillar_id": pillar_id, "subpillar": subpillar, "barriers_reported": barriers,
                "barriers_from_answers": [], "hapa_stage": stage, "assessed_at": moment, "received_at": moment}

    person.step3 = [answered("SO", ["physical opportunity"], "pre-intender", 3)]
    assert decide_module.priority(cat, cat.missions["SM1"], person, MORNING) == ({"SRc1a", "SRc1b"}, 0.7)
    person.step3.append(answered("SO", ["physical opportunity"], "actor", 1))
    assert decide_module.priority(cat, cat.missions["SM1"], person, MORNING) == ({"SRc2"}, 0.7)
    # A questionnaire received after the decision is not used.
    assert decide_module.priority(cat, cat.missions["SM1"], person, MORNING - timedelta(days=2)) == (
        {"SRc1a", "SRc1b"}, 0.7)
    person.step3 = [answered("PO", ["physical opportunity"], None, 1)]
    assert decide_module.priority(cat, cat.missions["PM3"], person, MORNING) == ({"PRc1"}, 0.7)
    assert decide_module.priority(cat, cat.missions["SM1"], person, MORNING) == (None, None)


def test_a_recommendation_logs_its_class(monkeypatch):
    force_composition(monkeypatch, (2, 0))
    person = participant([episode(mission_id="PM3")])
    _, deliveries = run(person)
    rules = [d["content_rule"] for d in deliveries if d["format"] == "recommendation"]
    assert rules and all(rule["class"] == "no_priority" for rule in rules)


def test_recent_deliveries_count_the_service_plans_not_the_sent_events():
    person = participant([episode(mission_id="PM3")])
    two_days_ago = DAY2 - timedelta(days=2)
    previous = [{"decision_day": two_days_ago, "format": "reminder", "content_id": "PRd1", "position": 0},
                {"decision_day": two_days_ago, "format": "reminder", "content_id": "PRd2", "position": 1},
                {"decision_day": two_days_ago, "format": "reminder", "content_id": "PRd3", "position": 2,
                 "cancelled": True},
                {"decision_day": DAY2 - timedelta(days=5), "format": "reminder", "content_id": "PRd4", "position": 0}]
    row, _ = run(person, previous=previous)
    assert "recent_deliveries" not in row["state"]
    assert row["context"]["planned_last_3_days"] == 2 and row["context"]["sent_last_3_days"] == 0


def test_the_order_of_the_day_is_drawn_uniformly_and_logged(monkeypatch):
    force_composition(monkeypatch, (2, 2))
    person = participant([episode(mission_id="PM3")])
    orders = set()
    for seed in range(60):
        _, deliveries = run(person, seed=seed)
        [order] = [x for d in deliveries for x in d["draws"] if x["kind"] == "order"]
        assert len(order["options"]) == 24 and all(p == 1 / 24 for p in order["options"].values())
        assert order["chosen"] == ">".join(str(d["content_id"]) for d in deliveries)
        assert [d["position"] for d in deliveries] == list(range(len(deliveries)))
        orders.add(tuple(d["format"] for d in deliveries))
    # Recommendations and reminders are interleaved in more than one way.
    assert len(orders) > 1
