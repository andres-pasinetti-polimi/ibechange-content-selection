from datetime import datetime, timedelta

from cs_rct import design
from cs_rct.history import daily_outcomes, earlier_marked_share, open_episode, resolution
from tests.helpers import START, episode, mark, participant

DAY1 = START.date()


def test_episode_is_open_until_its_outcome():
    resolved = START + timedelta(days=3)
    person = participant([episode(outcome="achieved", resolved_at=resolved)])
    assert open_episode(person, START + timedelta(days=1)) is not None
    assert open_episode(person, resolved + timedelta(minutes=1)) is None
    assert resolution(person.episodes[0], resolved)[0] == "resolved"


def test_outcome_not_yet_received_leaves_the_episode_open():
    resolved = START + timedelta(days=3)
    row = episode(outcome="achieved", resolved_at=resolved)
    row["resolution_received_at"] = resolved + timedelta(hours=5)
    person = participant([row])
    assert open_episode(person, resolved + timedelta(hours=1)) is not None


def test_episode_without_outcome_expires_after_window_and_margin():
    person = participant([episode()])
    expiry = datetime.combine(DAY1 + timedelta(days=design.WINDOW_DAYS + design.EXPIRY_MARGIN_DAYS),
                              datetime.min.time())
    assert open_episode(person, expiry - timedelta(seconds=1)) is not None
    assert resolution(person.episodes[0], expiry) == ("expired", expiry, 0)


def test_replaced_episode_is_withdrawn_and_keeps_its_marks():
    replaced = START + timedelta(hours=3)
    marks = [mark(datetime.combine(DAY1, datetime.min.time()), START + timedelta(hours=2)),
             # Entered after the replacement: not read.
             mark(datetime.combine(DAY1 + timedelta(days=1), datetime.min.time()), replaced + timedelta(hours=1))]
    person = participant([episode(outcome="changed", resolved_at=replaced)], marks=marks)
    assert resolution(person.episodes[0], START + timedelta(days=1))[0] == "withdrawn"
    assert daily_outcomes(person, person.episodes[0], [DAY1, DAY1 + timedelta(days=1)],
                          START + timedelta(days=2)) == {DAY1: 1, DAY1 + timedelta(days=1): 0}


def test_daily_outcome_is_the_mark_of_the_day_entered_by_the_end():
    end = START + timedelta(days=4, hours=12)
    marks = [
        mark(datetime.combine(DAY1, datetime.min.time()), START + timedelta(hours=10)),
        # Day 3 marked at the completion itself: read as the day it names.
        mark(datetime.combine(DAY1 + timedelta(days=2), datetime.min.time()), end),
        # Day 4 marked after the end: not read.
        mark(datetime.combine(DAY1 + timedelta(days=3), datetime.min.time()), end + timedelta(minutes=5)),
    ]
    person = participant([episode(outcome="achieved", resolved_at=end)], marks)
    days = [DAY1 + timedelta(days=i) for i in range(4)]
    outcomes = daily_outcomes(person, person.episodes[0], days, end + timedelta(days=1))
    assert outcomes == {days[0]: 1, days[1]: 0, days[2]: 1, days[3]: 0}


def test_mark_minus_backfill_subtracts_a_mark_for_the_day_before_entered_that_day(monkeypatch):
    monkeypatch.setattr(design, "DAILY_OUTCOME", "mark_minus_backfill")
    end = START + timedelta(days=4, hours=12)
    marks = [
        # Day 2 marked on day 2, day 1 marked on day 2: day 2 reads 1 - 1.
        mark(datetime.combine(DAY1 + timedelta(days=1), datetime.min.time()), START + timedelta(days=1, hours=10)),
        mark(datetime.combine(DAY1, datetime.min.time()), START + timedelta(days=1, hours=11)),
        # Day 3 marked on day 4 without a mark of day 4: day 4 reads 0 - 1.
        mark(datetime.combine(DAY1 + timedelta(days=2), datetime.min.time()), START + timedelta(days=3, hours=1)),
    ]
    person = participant([episode(outcome="achieved", resolved_at=end)], marks)
    days = [DAY1 + timedelta(days=i) for i in range(4)]
    outcomes = daily_outcomes(person, person.episodes[0], days, end + timedelta(days=1))
    assert outcomes == {days[0]: 1, days[1]: 0, days[2]: 1, days[3]: -1}


def test_earlier_share_counts_finished_episodes_only():
    first_end = START + timedelta(days=6, hours=12)
    episodes = [episode(1, outcome="not_achieved", resolved_at=first_end),
                episode(2, selected_at=START + timedelta(days=7))]
    marks = [mark(datetime.combine(DAY1 + timedelta(days=i), datetime.min.time()), START + timedelta(days=i, hours=12))
             for i in range(2)]
    person = participant(episodes, marks)
    assert earlier_marked_share(person, person.episodes[1], START + timedelta(days=8)) == 2 / 7
    assert earlier_marked_share(person, person.episodes[0], START + timedelta(days=8)) is None


def test_mark_minus_backfill_counts_only_marks_entered_after_the_decision_of_the_day(monkeypatch):
    monkeypatch.setattr(design, "DAILY_OUTCOME", "mark_minus_backfill")
    end = START + timedelta(days=4, hours=12)
    day2 = DAY1 + timedelta(days=1)
    day3 = DAY1 + timedelta(days=2)
    decided = {day2: datetime.combine(day2, datetime.min.time()) + timedelta(hours=3),
               day3: datetime.combine(day3, datetime.min.time()) + timedelta(hours=3)}
    marks = [
        # Day 1 marked on day 2 at 02:00, before the decision of day 2 at 03:00: not subtracted.
        mark(datetime.combine(DAY1, datetime.min.time()), datetime.combine(day2, datetime.min.time()) + timedelta(hours=2)),
        # Day 2 marked on day 3 at 10:00, after the decision of day 3: subtracted.
        mark(datetime.combine(day2, datetime.min.time()), datetime.combine(day3, datetime.min.time()) + timedelta(hours=10)),
    ]
    person = participant([episode(outcome="achieved", resolved_at=end)], marks)
    days = [DAY1, day2, day3]
    outcomes = daily_outcomes(person, person.episodes[0], days, end + timedelta(days=1), decided)
    assert outcomes == {DAY1: 1, day2: 1, day3: -1}
    # Without decision times every mark for the day before counts.
    assert daily_outcomes(person, person.episodes[0], days, end + timedelta(days=1))[day2] == 0
