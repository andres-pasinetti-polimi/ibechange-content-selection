"""Times are stored in UTC and read in the participant's local time, from the recruitment centre."""

from datetime import date, datetime

import pytest

from cs_rct import cycle, design
from cs_rct.api import from_utc, with_offset
from cs_rct.history import build_participants, marked_days, open_episode
from tests.helpers import load_catalogue

CATALOGUE = load_catalogue()


@pytest.fixture(autouse=True)
def real_zones(monkeypatch):
    monkeypatch.setattr(design, "CENTRE_TIMEZONE", {"ICO": "Europe/Madrid", "UMFCD": "Europe/Bucharest"})
    monkeypatch.setattr(design, "DEFAULT_TIMEZONE", "Europe/Madrid")


def tables(centre, selected_at, marks=()):
    """One participant of `centre` with PM3 selected at `selected_at` (UTC)."""
    return {
        "users": [{"user_uuid": "u1", "language": "ro", "enrolment_date": date(2027, 1, 1), "age": 60,
                   "recruitment_center": centre, "disabled_at": None}],
        "episodes": [{"mission_process_id": 1, "user_uuid": "u1", "mission_id": "PM3", "selected_at": selected_at,
                      "selection_received_at": selected_at, "outcome": None, "resolved_at": None,
                      "resolution_received_at": None}],
        "marks": [{"user_uuid": "u1", "mission_process_id": 1, "mission_id": "PM3", "step": None, "total_steps": None,
                   "reference_time": moment, "entered_at": moment, "process_id": None, "received_at": moment}
                  for moment in marks],
    }


def run_at(now, participant_tables):
    return cycle.run_cycle(now, build_participants(participant_tables), CATALOGUE, [], {}, None, {}, {},
                           draw_seed=lambda: 1, ran_yesterday=False)


def test_the_decision_day_is_the_participants_local_day():
    # 22:30 UTC on 4 January is 00:30 on 5 January in Bucharest (UTC+2).
    result = run_at(datetime(2027, 1, 4, 23, 0), tables("UMFCD", datetime(2027, 1, 4, 22, 30)))
    [(row, _)] = result.decisions
    assert row["decision_day"] == date(2027, 1, 5) and row["timezone"] == "Europe/Bucharest"
    assert row["decided_at"] == datetime(2027, 1, 4, 23, 0)  # the log keeps UTC


def test_a_mark_refers_to_the_participants_local_day():
    participant = build_participants(tables("UMFCD", datetime(2027, 1, 4, 8, 0),
                                            marks=[datetime(2027, 1, 4, 23, 30)]))["u1"]
    assert marked_days(participant, participant.episodes[0], datetime(2027, 1, 6)) == {date(2027, 1, 5)}


def test_the_last_decision_hour_is_local():
    # Madrid is UTC+1 in January: 19:30 UTC is 20:30, still within the last decision hour (20);
    # 20:30 UTC is 21:30, after it.
    assert run_at(datetime(2027, 1, 4, 19, 30), tables("ICO", datetime(2027, 1, 4, 19, 0))).decisions
    assert not run_at(datetime(2027, 1, 4, 20, 30), tables("ICO", datetime(2027, 1, 4, 19, 0))).decisions


def test_the_episode_is_open_from_the_local_selection_time():
    participant = build_participants(tables("UMFCD", datetime(2027, 1, 4, 22, 30)))["u1"]
    assert participant.episodes[0]["selected_at"] == datetime(2027, 1, 5, 0, 30)
    assert open_episode(participant, datetime(2027, 1, 5, 0, 31)) is not None


def test_api_times_carry_the_participants_offset():
    assert with_offset(datetime(2027, 1, 5, 18, 0), "Europe/Bucharest") == "2027-01-05T18:00:00+02:00"
    assert from_utc(datetime(2027, 7, 5, 16, 0), "Europe/Madrid") == "2027-07-05T18:00:00+02:00"
