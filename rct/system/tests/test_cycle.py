from collections import Counter

from cs_rct import replay, simulate
from cs_rct.history import build_participants


def run_simulation():
    return simulate.simulate(participants=6, days=20, enrolment_days=3, scenario="common", effect=0.1, seed=3)


def test_one_row_per_participant_day_and_closing_rows():
    decision_rows, latest, world = run_simulation()
    keys = Counter((row["user_uuid"], row["decision_day"]) for row in decision_rows)
    assert max(keys.values()) == 1
    statuses = Counter(row["status"] for row in decision_rows)
    assert statuses["decided"] > 0 and statuses["no_open_episode"] > 0
    assert latest.version > 1 and latest.episodes


def test_replay_reproduces_every_probability_and_every_version():
    decision_rows, latest, world = run_simulation()
    checked, mismatches = replay.check_probabilities(decision_rows, world.model_versions)
    assert checked == sum(row["status"] == "decided" for row in decision_rows) and mismatches == []
    participants = build_participants(world.tables)
    end = max(row["decided_at"] for row in decision_rows)
    checked, mismatches = replay.check_versions(decision_rows, world.model_versions, participants, end)
    assert checked == latest.version and mismatches == []


def test_baseline_leaves_out_the_entries_built_from_the_marks():
    from cs_rct import features
    for study in ("usability", "trial"):
        assert not set(features.MARK_NAMES) & set(features.baseline_names(study))
        assert set(features.MARK_NAMES) <= set(features.state_names(study))
        assert set(features.MARK_NAMES) <= set(features.baseline_names(study, with_marks=True))


def test_replay_refits_a_version_written_before_11_november_with_the_baseline_it_used():
    from cs_rct import design, features, model, update
    decision_rows, latest, world = run_simulation()
    participants = build_participants(world.tables)
    end = max(row["decided_at"] for row in decision_rows)
    # Rows logged before 11 November 2026 carried the mark entries in their baseline.
    for row in decision_rows:
        if row.get("baseline") and row.get("state"):
            row["baseline"].update({name: row["state"][name] for name in features.MARK_NAMES})
    old = features.baseline_names(design.STUDY, with_marks=True)
    versions = {}
    for version, (posterior, episodes) in world.model_versions.items():
        rows = [row for row in decision_rows if row["mission_process_id"] in set(episodes)]
        points, _ = update.training_points(rows, participants, end, design.STUDY, lag_rows=decision_rows,
                                           baseline=old)
        refitted = model.fit(points)  # baseline None, as in a version written before the change
        versions[version] = (refitted, episodes)
    checked, mismatches = replay.check_versions(decision_rows, versions, participants, end)
    assert checked == latest.version and mismatches == []
