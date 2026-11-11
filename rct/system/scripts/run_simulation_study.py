"""The simulation study of the paper (rct/paper/main.tex, Section "Simulation study"), recorded.

Runs every scenario under each policy (uniform, conservative) on the service's own cycle (cs_rct/simulate.py)
and writes the summaries, with the configuration of the design in force, to
simulation_results/simulation_study_<study>_<date>.json.

Run from the system/ folder:
    python -m scripts.run_simulation_study --participants 10 --days 30 --replications 100
"""

import argparse
import json
from datetime import date
from pathlib import Path

from cs_rct import design, simulate

SCENARIOS = ["null", "common", "heterogeneous", "decaying"]
POLICIES = ["uniform", "conservative"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--participants", type=int, default=10)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--enrolment-days", type=int, default=7)
    parser.add_argument("--effect", type=float, default=0.05)
    parser.add_argument("--replications", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    arguments = parser.parse_args()
    design.PROGRAMME_DAYS = arguments.days

    results = []
    for scenario in SCENARIOS:
        effect = 0.0 if scenario == "null" else arguments.effect
        for policy in POLICIES:
            design.POLICY = policy
            replications = [simulate.simulate(arguments.participants, arguments.days, arguments.enrolment_days,
                                              scenario, effect, arguments.seed + r)
                            for r in range(arguments.replications)]
            summary = simulate.summarise(replications, effect, scenario)
            results.append(summary)
            print(scenario, policy, summary["one_more_notification"], flush=True)

    output = (Path(__file__).parent.parent / "simulation_results"
              / f"simulation_study_{design.STUDY}_{date.today().isoformat()}.json")
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps({"arguments": vars(arguments), "parameters": design.parameters(),
                                  "results": results}, indent=1))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
