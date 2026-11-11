"""The replay check (rct/paper/main.tex, Section "Design choices and their rationale", item 4).

Since the posterior is shared, the logged sequence of probabilities cannot be
checked by eye, so it is checked exactly:

1. the probabilities of the admissible compositions of every decision day are
   recomputed from the rule the row names, the parameters it logged, its seed and the
   model version it names (composition.probabilities);
2. every model version is refitted from the decision rows of the episodes it lists
   and compared with the stored posterior mean.

Rows and model versions of the scheme of 14 October 2026 (no composition logged, a
posterior with theta_mean) are not checked.

A refit can differ from the stored version only if the derived tables changed after
the version was written, for example because a mark arrived late or the parsing
rules changed; the check reports it and does not decide which is right.

Run from the system/ folder:
    python -m cs_rct.replay
"""

import logging

import numpy as np

from cs_rct import composition, database, design, features, model, store, update
from cs_rct.history import build_participants

log = logging.getLogger("replay")

TOLERANCE = 1e-9


def check_probabilities(decision_rows, versions):
    """Return (number checked, list of mismatches)."""
    checked, mismatches = 0, []
    for row in decision_rows:
        if row["status"] != "decided" or not row.get("composition_probabilities"):
            continue
        logged = row["composition_probabilities"]
        admissible = [composition.parse(key) for key in logged]
        parameters = row["parameters"]
        rule = row["policy_rule"]
        if rule == "fixed":
            replayed = composition.starting_probabilities(parameters["policy"], admissible,
                                                          parameters["explore_share"],
                                                          tuple(parameters["default_composition"]))
        else:
            posterior = None
            if rule == "conservative":
                posterior = versions[row["model_version"]][0]
            replayed, _ = composition.probabilities(parameters["policy"], admissible, posterior, row["rng_seed"],
                                                    parameters)
        checked += 1
        difference = max(abs(logged[key] - replayed.get(key, 0.0)) for key in logged)
        if set(replayed) != set(logged) or difference > TOLERANCE:
            mismatches.append({"decision_id": row.get("id"), "logged": logged, "replayed": replayed})
    return checked, mismatches


def check_versions(decision_rows, versions, participants, now):
    """Return (number checked, list of mismatches)."""
    checked, mismatches = 0, []
    for version, (posterior, episodes) in sorted(versions.items()):
        if posterior is None:
            continue
        rows = [row for row in decision_rows if row["mission_process_id"] in set(episodes)]
        baseline = posterior.baseline or features.baseline_names(design.STUDY, with_marks=True)
        points, included = update.training_points(rows, participants, now, design.STUDY, lag_rows=decision_rows,
                                                  baseline=baseline)
        refitted = model.fit(points, baseline)
        checked += 1
        difference = float(np.max(np.abs(np.array(refitted.mean) - np.array(posterior.mean))))
        if included != sorted(episodes) or difference > TOLERANCE:
            mismatches.append({"version": version, "largest_difference": difference,
                               "episodes_missing": sorted(set(episodes) - set(included))})
    return checked, mismatches


def run_replay(db):
    """Check every logged probability and model version, store the result, and return a summary."""
    now = design.utc_now()
    decision_rows = store.load_decisions(db)
    versions = store.load_model_versions(db)
    participants = build_participants(store.load_tables(db))

    probabilities_checked, probability_mismatches = check_probabilities(decision_rows, versions)
    versions_checked, version_mismatches = check_versions(decision_rows, versions, participants, now)
    db.execute(
        "INSERT INTO replay_checks (checked_at, decisions_checked, probability_mismatches, versions_checked, "
        "version_mismatches, details) VALUES (%s, %s, %s, %s, %s, %s)",
        (now, probabilities_checked, len(probability_mismatches), versions_checked, len(version_mismatches),
         store.as_value({"probabilities": probability_mismatches[:100], "versions": version_mismatches[:100]})),
    )
    summary = {"decisions_checked": probabilities_checked, "probability_mismatches": len(probability_mismatches),
               "versions_checked": versions_checked, "version_mismatches": len(version_mismatches)}
    log.info("Replay check: %s", summary)
    return summary


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    db = database.connect()
    database.create_tables(db)
    run_replay(db)


if __name__ == "__main__":
    main()
