SET TIME ZONE 'UTC';

-- core
\copy (SELECT * FROM runs ORDER BY run_id ASC) TO '/tmp/runs.csv' WITH CSV HEADER;
\copy (SELECT * FROM users ORDER BY id ASC) TO '/tmp/users.csv' WITH CSV HEADER;
\copy (SELECT * FROM disabled_users) TO '/tmp/disabled_users.csv' WITH CSV HEADER;
\copy (SELECT * FROM escalation_levels ORDER BY id ASC) TO '/tmp/escalation_levels.csv' WITH CSV HEADER;
\copy (SELECT * FROM health_habit_assessments ORDER BY id ASC) TO '/tmp/health_habit_assessments.csv' WITH CSV HEADER;
\copy (SELECT * FROM new_missions_and_contents ORDER BY id ASC) TO '/tmp/new_missions_and_contents.csv' WITH CSV HEADER;
\copy (SELECT * FROM recommendation_plans ORDER BY id ASC) TO '/tmp/recommendation_plans.csv' WITH CSV HEADER;
\copy (SELECT * FROM plan_contents ORDER BY id ASC) TO '/tmp/plan_contents.csv' WITH CSV HEADER;
\copy (SELECT * FROM selected_contents ORDER BY id ASC) TO '/tmp/selected_contents.csv' WITH CSV HEADER;
\copy (SELECT * FROM user_feedback ORDER BY id ASC) TO '/tmp/user_feedback.csv' WITH CSV HEADER;
\copy (SELECT * FROM user_version_upgrades ORDER BY id ASC) TO '/tmp/user_version_upgrades.csv' WITH CSV HEADER;
\copy (SELECT * FROM feature_audit ORDER BY id ASC) TO '/tmp/feature_audit.csv' WITH CSV HEADER;

-- old bandits
\copy (SELECT * FROM intervention_mab_runs ORDER BY run_id ASC) TO '/tmp/intervention_mab_runs.csv' WITH CSV HEADER;
\copy (SELECT * FROM intervention_mab_updates ORDER BY id ASC) TO '/tmp/intervention_mab_updates.csv' WITH CSV HEADER;
\copy (SELECT * FROM intervention_mab_samples ORDER BY id ASC) TO '/tmp/intervention_mab_samples.csv' WITH CSV HEADER;
\copy (SELECT * FROM recommendation_mab_runs ORDER BY run_id ASC) TO '/tmp/recommendation_mab_runs.csv' WITH CSV HEADER;
\copy (SELECT * FROM recommendation_mab_updates ORDER BY id ASC) TO '/tmp/recommendation_mab_updates.csv' WITH CSV HEADER;
\copy (SELECT * FROM recommendation_mab_samples ORDER BY id ASC) TO '/tmp/recommendation_mab_samples.csv' WITH CSV HEADER;
\copy (SELECT * FROM resource_mab_runs ORDER BY run_id ASC) TO '/tmp/resource_mab_runs.csv' WITH CSV HEADER;
\copy (SELECT * FROM resource_mab_updates ORDER BY id ASC) TO '/tmp/resource_mab_updates.csv' WITH CSV HEADER;
\copy (SELECT * FROM resource_mab_samples ORDER BY id ASC) TO '/tmp/resource_mab_samples.csv' WITH CSV HEADER;

-- new bandits
\copy (SELECT * FROM intervention_mab_runs_new ORDER BY run_id ASC) TO '/tmp/intervention_mab_runs_new.csv' WITH CSV HEADER;
\copy (SELECT * FROM intervention_mab_updates_new ORDER BY id ASC) TO '/tmp/intervention_mab_updates_new.csv' WITH CSV HEADER;
\copy (SELECT * FROM intervention_mab_samples_new ORDER BY id ASC) TO '/tmp/intervention_mab_samples_new.csv' WITH CSV HEADER;
\copy (SELECT * FROM recommendation_mab_runs_new ORDER BY run_id ASC) TO '/tmp/recommendation_mab_runs_new.csv' WITH CSV HEADER;
\copy (SELECT * FROM recommendation_mab_updates_new ORDER BY id ASC) TO '/tmp/recommendation_mab_updates_new.csv' WITH CSV HEADER;
\copy (SELECT * FROM recommendation_mab_samples_new ORDER BY id ASC) TO '/tmp/recommendation_mab_samples_new.csv' WITH CSV HEADER;
\copy (SELECT * FROM resource_mab_runs_new ORDER BY run_id ASC) TO '/tmp/resource_mab_runs_new.csv' WITH CSV HEADER;
\copy (SELECT * FROM resource_mab_updates_new ORDER BY id ASC) TO '/tmp/resource_mab_updates_new.csv' WITH CSV HEADER;
\copy (SELECT * FROM resource_mab_samples_new ORDER BY id ASC) TO '/tmp/resource_mab_samples_new.csv' WITH CSV HEADER;
