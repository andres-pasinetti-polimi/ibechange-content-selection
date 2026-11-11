# iBeChange content selection

This repository holds the code of the content selection (CS) module developed by Politecnico di Milano (POLIMI) in Task 4.5 of the iBeChange project (Horizon Europe, grant agreement No 101136840). It is the code of deliverable D4.6, Reinforcement Learning Approach, and implements the techniques described in deliverable D4.5, Reinforcement Learning Model. The release that accompanies D4.6 is the tag `d4.6-v1.0`.

The code covers the two phases of the project in which the CS module learns from the data of the participants:

- the pilot study (iBC/PS), in which the CS module selected the recommendations and resources of each participant from December 2025 to August 2026;
- the randomised controlled trial (iBC/CT), for which the CS service decides each day how many recommendations and reminders a participant receives and which ones.

## Layout

| Folder | Content |
|---|---|
| `pilot/system/cs_module/` | CS module of the pilot study: REST API (`cs_api.py`), selection of the content (`content_selection/`), Thompson sampling policies (`multi_armed_bandit/`), storage and updates (`utils/`) |
| `pilot/system/airflow/` | Airflow DAGs that ran the CS module in the integration and production environments of the platform (`cs_orchestrator_int.py`, `cs_orchestrator_prod.py`) |
| `pilot/system/` (other folders) | Test environment of the pilot: virtual users (`virtual_user/`), a time service (`time_service/`), a stand-in for the opportune moment identification module (`omi_module/`), the orchestrator of simulated studies (`orchestrator/`) and the module variant used in it (`cs_module_test/`) |
| `pilot/results_analysis/pilot_compendium/` | Off-policy evaluation of the pilot logs used in D4.5 (`s05_offpolicy.py`, `s05c_offpolicy_absolute.py`, `s05d_offpolicy_rated_selection.py`, `s06_sensitivity.py`, shared loaders in `common.py`) |
| `rct/system/cs_rct/` | CS service of the RCT: message consumer, REST API, hourly cycle, decision rule, model, replay and report |
| `rct/system/scripts/` | Cronicle jobs (`cronicle/`), scripts that build the catalogue from the clinical workbooks (`build_*.py`), deployment check (`server_check.py`) and simulation study (`run_simulation_study.py`) |
| `rct/system/tests/` | Unit tests of the RCT service |
| `rct/system/catalogue/` | Synthetic catalogue (see below) |

## What is not included

- **Clinical content.** The Missions, recommendations, resources, reminders and pop-up texts of the trial were written by the clinical partners and are not distributed. `rct/system/catalogue/` holds a synthetic catalogue in the same format: 17 Missions with the structural columns of the trial catalogue (Pillar, Objective, class, frequency, count of days, weekly cap, schedulable), 79 reminders, 91 recommendations, 11 resources and 42 pop-up texts, with every text replaced by a placeholder. The content files of the pilot test environment (`virtual_user/intervention_contents/`, `orchestrator/contents/`) are not included either, so the simulated pilot studies do not run without them.
- **Data.** No participant data, database snapshot or platform export is included. The off-policy scripts read them from `pilot/data/`, which is not distributed; the identifiers of the participant accounts excluded from the analysis are read from `pilot/data/excluded_user_ids.txt`.
- **Deployment settings.** Addresses, credentials and tokens of the servers of the platform are read from environment variables and are not part of the code.

## Running the tests of the RCT service

The RCT service requires Python 3.13. The tests that use a database run on the PostgreSQL container of `rct/system/docker-compose.yml` and are skipped when it is not running.

```
cd rct/system
pip install -r requirements.txt
docker compose up -d postgres
python -m pytest
```

With the synthetic catalogue, 118 tests pass and 4 are skipped: the skipped tests check the catalogue scripts against the clinical workbooks, which are not distributed. With the clinical catalogue, all 122 tests pass.

The commands to run the service, the message consumer and the simulation study are listed in D4.6, Table 9.

## Licence

The code is released under the Apache License, Version 2.0 (see `LICENSE` and `NOTICE`).

## Funding

This project has received funding from the European Union's Horizon Europe research and innovation programme under the Grant Agreement Number 101136840.

Funded by the European Union. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Health and Digital Executive Agency (HADEA). Neither the European Union nor the granting authority can be held responsible for them.
