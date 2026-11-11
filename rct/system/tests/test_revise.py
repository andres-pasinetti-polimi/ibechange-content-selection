from datetime import datetime, timedelta

from cs_rct import model
from cs_rct.decide import decide
from cs_rct.revise import revise
from tests.helpers import START, episode, load_catalogue, mark, participant

CATALOGUE = load_catalogue()
DAY = START.date() + timedelta(days=1)
MORNING = datetime.combine(DAY, datetime.min.time()) + timedelta(hours=1)
NOON = MORNING + timedelta(hours=11)
PRIOR = model.prior()


def planned_day(person, seed=5):
    """The day's decision with ids on the deliveries, as store.py returns them."""
    row, deliveries = decide(person, CATALOGUE, PRIOR, 1, "learned", DAY, MORNING, seed, [], "usability")
    row["id"] = 1
    for index, delivery in enumerate(deliveries, start=1):
        delivery.update(id=index, decision_id=1, cancelled=False)
    return row, deliveries


def test_nothing_changes_without_an_event():
    person = participant([episode(mission_id="PM3")])
    row, deliveries = planned_day(person)
    assert revise(person, CATALOGUE, row, deliveries, deliveries, DAY, NOON, 9) is None


def test_a_mark_today_cancels_the_reminders_and_keeps_the_recommendation():
    person = participant([episode(mission_id="PM3")])
    row, deliveries = planned_day(person)
    person.marks.append(mark(datetime.combine(DAY, datetime.min.time()) + timedelta(hours=9), NOON - timedelta(hours=1)))
    reason, cancelled, new_rows = revise(person, CATALOGUE, row, deliveries, deliveries, DAY, NOON, 9)
    reminders = [d["id"] for d in deliveries if d["format"] == "reminder"]
    assert reason == "task_marked" and cancelled == reminders and new_rows == []


def test_reaching_the_count_cancels_the_reminders():
    person = participant([episode(mission_id="PM1")])  # once a week
    row, deliveries = planned_day(person)
    person.marks.append(mark(datetime.combine(START.date(), datetime.min.time()), NOON - timedelta(hours=1)))
    reason, cancelled, _ = revise(person, CATALOGUE, row, deliveries, deliveries, DAY, NOON, 9)
    assert reason == "count_reached" and len(cancelled) == sum(d["format"] == "reminder" for d in deliveries)


def test_a_replaced_mission_gets_the_same_composition_of_the_new_one():
    changed = START + timedelta(hours=26)  # within the day of the plan, before noon
    person = participant([episode(1, mission_id="PM3", outcome="changed", resolved_at=changed),
                          episode(2, mission_id="NM1", selected_at=changed + timedelta(minutes=5))])
    old = participant([episode(1, mission_id="PM3")])
    row, deliveries = planned_day(old)
    reason, cancelled, new_rows = revise(person, CATALOGUE, row, deliveries, deliveries, DAY, NOON, 9)
    assert reason == "mission_changed" and cancelled == [d["id"] for d in deliveries]
    assert [d["role"] for d in new_rows] == [d["role"] for d in deliveries]
    assert {d["mission_id"] for d in new_rows} == {"NM1"} and {d["mission_process_id"] for d in new_rows} == {2}
    assert min(d["position"] for d in new_rows) > max(d["position"] for d in deliveries)
    assert not {d["content_id"] for d in new_rows} & {d["content_id"] for d in deliveries}


def test_a_closed_mission_without_a_new_one_cancels_everything():
    closed = START + timedelta(hours=26)
    person = participant([episode(1, mission_id="PM3", outcome="not_achieved", resolved_at=closed)])
    row, deliveries = planned_day(participant([episode(1, mission_id="PM3")]))
    reason, cancelled, new_rows = revise(person, CATALOGUE, row, deliveries, deliveries, DAY, NOON, 9)
    assert reason == "mission_ended" and len(cancelled) == len(deliveries) and new_rows == []


def test_cancelled_deliveries_are_not_revised_again():
    person = participant([episode(mission_id="PM3")])
    row, deliveries = planned_day(person)
    person.marks.append(mark(datetime.combine(DAY, datetime.min.time()), NOON - timedelta(hours=1)))
    for d in deliveries:
        d["cancelled"] = d["format"] == "reminder"
    assert revise(person, CATALOGUE, row, deliveries, deliveries, DAY, NOON, 9) is None


def test_a_new_mission_in_emotional_wellbeing_keeps_one_recommendation(monkeypatch):
    from tests.helpers import force_composition
    force_composition(monkeypatch, (2, 2))
    changed = START + timedelta(hours=26)
    person = participant([episode(1, mission_id="PM3", outcome="changed", resolved_at=changed),
                          episode(2, mission_id="EM1", selected_at=changed + timedelta(minutes=5))])
    row, deliveries = planned_day(participant([episode(1, mission_id="PM3")]))
    assert [d["format"] for d in deliveries].count("recommendation") == 2
    reason, _, new_rows = revise(person, CATALOGUE, row, deliveries, deliveries, DAY, NOON, 9)
    formats = [d["format"] for d in new_rows]
    assert reason == "mission_changed" and formats.count("recommendation") == 1 and formats.count("reminder") == 3
