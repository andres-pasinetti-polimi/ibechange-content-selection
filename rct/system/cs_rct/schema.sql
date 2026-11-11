-- Every message received from Artemis, stored exactly as it arrived.
--
-- Rows are only ever inserted, never updated or deleted, so the full history of
-- events can always be parsed again from this table. Duplicates delivered by the
-- broker are kept here and removed when the events are parsed.
--
-- message_id is the STOMP header set by Artemis, "<connection>,<message>". The
-- first part changes when a message is delivered again on a new connection, so
-- message_id alone does not identify duplicates.
CREATE TABLE IF NOT EXISTS raw_events (
    id          BIGSERIAL PRIMARY KEY,
    received_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    message_id  TEXT,
    destination TEXT,
    headers     JSONB NOT NULL,
    body        TEXT NOT NULL,
    -- Top-level key of the JSON body, e.g. 'mission_selected'.
    -- NULL when the body is not a JSON object with exactly one key.
    event_type  TEXT
);

CREATE INDEX IF NOT EXISTS raw_events_event_type_idx ON raw_events (event_type);

-- The tables below are derived from raw_events by rebuild_tables.py, which
-- empties and refills them on every run. Do not edit them by hand.
-- Timestamps are stored as sent by EUT, without a time zone.

CREATE TABLE IF NOT EXISTS users (
    user_uuid          TEXT PRIMARY KEY,
    personal_id        TEXT,
    language           TEXT,
    enrolment_date     DATE,
    age                INTEGER,
    recruitment_center TEXT,
    attributes         JSONB,       -- every attribute of the new_user message
    disabled_at        TIMESTAMP
);

-- One episode is one Mission selected by a participant, until its outcome.
CREATE TABLE IF NOT EXISTS episodes (
    mission_process_id BIGINT PRIMARY KEY,
    user_uuid          TEXT,
    mission_id         TEXT,
    selected_at        TIMESTAMP,
    outcome            TEXT,        -- 'achieved', 'not_achieved', 'changed', or NULL while open
    resolved_at        TIMESTAMP,
    -- When the service received the selection and the outcome (raw_events.received_at),
    -- so that a decision reads only what had arrived by then.
    selection_received_at  TIMESTAMP,
    resolution_received_at TIMESTAMP
);

-- Calendar marks (task_step_done messages).
CREATE TABLE IF NOT EXISTS marks (
    user_uuid          TEXT,
    mission_process_id BIGINT,
    mission_id         TEXT,
    step               TEXT,        -- as sent; a UUID in the September 2026 INT messages
    total_steps        INTEGER,
    reference_time     TIMESTAMP,   -- the day the mark refers to (finished_at)
    entered_at         TIMESTAMP,   -- when the participant entered it (created_at)
    process_id         BIGINT,
    received_at        TIMESTAMP    -- when the service received the message
);

-- Tasks planned at a day and hour (task_step_scheduled messages), one row per slot.
CREATE TABLE IF NOT EXISTS scheduled_tasks (
    user_uuid          TEXT,
    mission_process_id BIGINT,
    mission_id         TEXT,
    step               TEXT,        -- as sent; a UUID in the September 2026 INT messages
    total_steps        INTEGER,
    start_at           TIMESTAMP,
    end_at             TIMESTAMP,
    entered_at         TIMESTAMP,
    process_id         BIGINT,
    received_at        TIMESTAMP    -- when the service received the message
);

-- Notifications sent, opened, rated, saved and unsaved.
CREATE TABLE IF NOT EXISTS notification_events (
    user_uuid          TEXT,
    action             TEXT,        -- 'sent', 'opened', 'rated', 'saved', 'unsaved'
    content_id         TEXT,
    content_type       TEXT,        -- 'recommendation', 'reminder', ...
    mission_id         TEXT,
    mission_process_id BIGINT,
    rating             INTEGER,     -- only for 'rated': 1 negative, 2 positive
    created_at         TIMESTAMP,
    process_id         BIGINT,
    received_at        TIMESTAMP    -- when the service received the message
);

-- step was an INTEGER column before 16 October 2026; the derived tables are refilled on every
-- run, so changing the type on an existing database loses nothing.
ALTER TABLE marks ALTER COLUMN step TYPE TEXT;
ALTER TABLE scheduled_tasks ALTER COLUMN step TYPE TEXT;

-- HHS questionnaires answered (health_habit_assessments messages), one row per answer.
CREATE TABLE IF NOT EXISTS health_assessments (
    user_uuid          TEXT,
    pillar_id          TEXT,        -- 'SO', 'AO', 'NO', 'PO', 'EO'
    score              DOUBLE PRECISION,
    colour             TEXT,
    trend              DOUBLE PRECISION,  -- difference from the previous assessment
    subpillars         JSONB,       -- {name: score}
    assessed_at        TIMESTAMP,
    received_at        TIMESTAMP
);

-- Step 3 questionnaires answered (psychological_assessments messages), one row per answer.
CREATE TABLE IF NOT EXISTS psychological_assessments (
    user_uuid              TEXT,
    pillar_id              TEXT,   -- 'SO', 'AO', 'NO', 'PO', 'EO'
    subpillar              TEXT,
    barriers_reported      JSONB,  -- COM-B sub-domains of the field "barriers", computed by EUT
    barriers_from_answers  JSONB,  -- COM-B sub-domains of the tagged answers (catalogue/step3_options.csv)
    hapa_stage             TEXT,   -- 'pre-intender', 'intender', 'actor' or NULL
    answers                JSONB,  -- ids of the tagged options answered
    assessed_at            TIMESTAMP,
    received_at            TIMESTAMP
);

-- The tables below are written by the service and never rebuilt from raw_events:
-- they are the only record of what the module knew and drew at the moment it
-- decided, and every estimate after the trial is computed from them. Rows are
-- inserted, never updated (job_runs excepted, whose row is closed at the end of the run).

-- One row per participant-day of the arm (rct_logging_specification.md, 4.1),
-- including the days without a decision, with the reason in status:
--   'decided'             a decision point: two or more admissible compositions, one drawn
--   'no_choice'           one admissible composition (e.g. (2,0) once the task is marked), sent
--   'below_minimum'       no admissible composition (Emotional Wellbeing with the task marked):
--                         one recommendation alone if the Mission can supply one
--   'past_window'         the episode is past its W days: nothing sent, the Mission awaits closure
--   'no_admissible_draw', 'count_reached'
--                         rows of the scheme of 14 October 2026 (before 6 November 2026)
--   'unknown_mission'     the Mission is not in the catalogue: nothing sent
--   'no_open_episode', 'selected_after_last_decision_hour', 'selection_day', 'disabled'
--                         written by the closing pass for a day with no decision
-- A day absent from this table is a day the service did not run.
CREATE TABLE IF NOT EXISTS decisions (
    id                   BIGSERIAL PRIMARY KEY,
    decision_day         DATE NOT NULL,
    decided_at           TIMESTAMP NOT NULL,   -- service clock, the "now" every feature was read at
    user_uuid            TEXT NOT NULL,
    mission_process_id   BIGINT,               -- the open episode, NULL if none
    mission_id           TEXT,
    group_name           TEXT,                 -- 'learned' or 'fixed' (within-arm evaluation)
    status               TEXT NOT NULL,
    -- What the policy read: the treatment vector s, the baseline features psi and the raw
    -- quantities they come from, as computed at the decision and not reconstructed later.
    state                JSONB,
    baseline             JSONB,
    context              JSONB,
    -- The probability both draws were taken with (clipped), its unclipped value, and on
    -- fixed-group days the probability the posterior would have given (computed, not used).
    propensity           DOUBLE PRECISION,
    propensity_raw       DOUBLE PRECISION,
    propensity_learned   DOUBLE PRECISION,
    fallback             BOOLEAN,              -- model unavailable: fixed probability used
    admissible_draws     INTEGER,              -- n, 0 to 2
    draw_1               BOOLEAN,
    draw_2               BOOLEAN,
    blocked_by           TEXT,                 -- 'task_marked' or 'weekly_cap' when n < 2
    additional           INTEGER,              -- k, additional reminders sent
    model_version        INTEGER,
    parameters           JSONB,
    rng_seed             BIGINT,               -- seed of the generator of the draws and the content
    data_cutoff          BIGINT,               -- id of the last raw event read
    code_version         TEXT,
    UNIQUE (user_uuid, decision_day)
);

-- Every item the service hands to OMI, fixed by rule or drawn (logging spec 4.2, 4.4),
-- with the set it was drawn from, the excluded items with their rule, and its probability.
-- content_id is NULL when no item could be drawn (note gives why); such rows are not sent.
CREATE TABLE IF NOT EXISTS deliveries (
    id                    BIGSERIAL PRIMARY KEY,
    decision_id           BIGINT NOT NULL REFERENCES decisions (id),
    decision_day          DATE NOT NULL,
    user_uuid             TEXT NOT NULL,
    mission_process_id    BIGINT,
    mission_id            TEXT,
    position              INTEGER,             -- order within the day
    role                  TEXT,                -- 'recommendation', 'resource', 'reminder' (earlier: 'fixed_reminder', 'additional_reminder')
    format                TEXT,                -- 'recommendation' or 'reminder'
    content_id            TEXT,
    content_probability   DOUBLE PRECISION,
    candidates            JSONB,
    excluded              JSONB,
    note                  TEXT,                -- e.g. 'redraw', 'repeat_within_no_repeat_days', 'all_disliked'
    reminder_kind         TEXT,                -- 'hour' or 'day'
    bct                   TEXT,                -- the reminder's class
    text                  TEXT,                -- the reminder text in the participant's language
    language              TEXT,
    declared_at           TIMESTAMP            -- the declared hour, for the reminder of a scheduled task
);

-- The participant's time zone (IANA name) the decision day was read in; decided_at is in UTC,
-- and every time of the deliveries is local to this zone (20 October 2026).
ALTER TABLE decisions ADD COLUMN IF NOT EXISTS timezone TEXT;

-- The latest time a delivery can be sent (OMI's deadline, TU/e, 16 October 2026): on a day
-- whose task is scheduled, the scheduled time for every reminder; NULL otherwise.
ALTER TABLE deliveries ADD COLUMN IF NOT EXISTS deadline TIMESTAMP;

-- What the deadline is made of (20 October 2026): the scheduled time of the day's task, on every
-- delivery of the plan, and the latest time the reminder's own text allows.
ALTER TABLE deliveries ADD COLUMN IF NOT EXISTS task_scheduled_for TIMESTAMP;
ALTER TABLE deliveries ADD COLUMN IF NOT EXISTS latest_time TIMESTAMP;
-- The ID of the pop-up text of a recommendation or resource (catalogue/push_texts.csv), and the
-- probability with which it was drawn.
ALTER TABLE deliveries ADD COLUMN IF NOT EXISTS push_text_id INTEGER;
ALTER TABLE deliveries ADD COLUMN IF NOT EXISTS push_text_probability DOUBLE PRECISION;
-- Of a recommendation (29 October 2026): its class in the content rule ("prioritised", "other",
-- "no_priority", "variation"), the class probability and the ids of the prioritised class.
ALTER TABLE deliveries ADD COLUMN IF NOT EXISTS content_rule JSONB;

CREATE INDEX IF NOT EXISTS deliveries_user_day_idx ON deliveries (user_uuid, decision_day);

-- The revision of the day's plan a delivery was created in: 0 at the decision, then one more
-- for each revision (revise.py). Added on 16 October 2026.
ALTER TABLE deliveries ADD COLUMN IF NOT EXISTS revision INTEGER NOT NULL DEFAULT 0;

-- Revisions of a day's plan after events of the day (task marked, count reached, Mission
-- replaced or closed). The plan sent to OMI has plan_id "<decision_id>.<revision>".
CREATE TABLE IF NOT EXISTS plan_revisions (
    id                   BIGSERIAL PRIMARY KEY,
    decision_id          BIGINT NOT NULL REFERENCES decisions (id),
    revision             INTEGER NOT NULL,
    created_at           TIMESTAMP NOT NULL,
    reason               TEXT NOT NULL,       -- 'task_marked', 'count_reached', 'mission_changed', 'mission_ended'
    rng_seed             BIGINT,              -- seed of the content drawn for a replaced Mission
    UNIQUE (decision_id, revision)
);

-- Deliveries withdrawn by a revision; the delivery rows themselves are never changed.
CREATE TABLE IF NOT EXISTS delivery_cancellations (
    delivery_id          BIGINT PRIMARY KEY REFERENCES deliveries (id),
    revision             INTEGER NOT NULL,
    cancelled_at         TIMESTAMP NOT NULL,
    reason               TEXT NOT NULL
);

-- The posterior after each update, with the episodes it was fitted on, so that every
-- logged probability can be recomputed from it and every version refitted (replay check).
CREATE TABLE IF NOT EXISTS model_versions (
    version              INTEGER PRIMARY KEY,
    created_at           TIMESTAMP NOT NULL,
    episodes             JSONB NOT NULL,       -- mission_process_ids included
    fingerprint          TEXT NOT NULL,        -- hash of the decision points fitted on
    posterior            JSONB NOT NULL,
    parameters           JSONB,
    data_cutoff          BIGINT,
    code_version         TEXT
);

-- Group of each participant within the arm, drawn once.
CREATE TABLE IF NOT EXISTS participant_groups (
    user_uuid            TEXT PRIMARY KEY,
    group_name           TEXT NOT NULL,
    stratum              TEXT,
    assigned_at          TIMESTAMP NOT NULL,
    rng_seed             BIGINT
);

-- The placements OMI returns (POST /recommendation_plans), stored as received.
CREATE TABLE IF NOT EXISTS placements (
    id                   BIGSERIAL PRIMARY KEY,
    received_at          TIMESTAMP NOT NULL DEFAULT now(),
    delivery_id          BIGINT,
    user_uuid            TEXT,
    scheduled_for        TIMESTAMP,
    status               TEXT,                -- 'scheduled', 'deferred', 'dropped', as OMI states
    body                 JSONB NOT NULL
);

-- The recommendations catalogue received from VU (POST /recommendations), every version.
CREATE TABLE IF NOT EXISTS recommendation_catalogue (
    id                   BIGSERIAL PRIMARY KEY,
    received_at          TIMESTAMP NOT NULL DEFAULT now(),
    body                 JSONB NOT NULL
);

-- The resources catalogue received from VU (POST /resources), every version.
CREATE TABLE IF NOT EXISTS resource_catalogue (
    id                   BIGSERIAL PRIMARY KEY,
    received_at          TIMESTAMP NOT NULL DEFAULT now(),
    body                 JSONB NOT NULL
);

-- One row per run of a scheduled job (logging spec 9.1).
CREATE TABLE IF NOT EXISTS job_runs (
    id                   BIGSERIAL PRIMARY KEY,
    job                  TEXT NOT NULL,
    started_at           TIMESTAMP NOT NULL,
    finished_at          TIMESTAMP,
    status               TEXT,                -- 'ok' or 'failed'
    summary              JSONB,
    error                TEXT
);

-- Result of each replay check (paper, Section "Design choices", item 4).
CREATE TABLE IF NOT EXISTS replay_checks (
    id                   BIGSERIAL PRIMARY KEY,
    checked_at           TIMESTAMP NOT NULL,
    decisions_checked    INTEGER,
    probability_mismatches INTEGER,
    versions_checked     INTEGER,
    version_mismatches   INTEGER,
    details              JSONB
);

-- Every random draw of the service, and every choice it fixed by rule, with the probability of each
-- option and the option chosen (5 November 2026; rct_logging_specification.md, 1.10). The probability
-- of a plan under the policy that decided it is the product of the probabilities of the chosen
-- options of its rows, whatever the policy, so that later analyses need no column specific to one
-- algorithm. delivery_id is NULL for a draw of the day (the additional draws, and later the
-- composition); seq is the order of the draws of one delivery, or of the day. Options are the
-- admissible ones only; the items excluded by a rule are in deliveries.excluded.
--   kind: 'composition' (from 6 November 2026; 'additional_draw' and 'format' on earlier rows),
--         'resource', 'class', 'code', 'variation', 'reminder_text', 'push_text'
--   forced: the choice was fixed by a rule (one option, probability 1)
CREATE TABLE IF NOT EXISTS draws (
    id                   BIGSERIAL PRIMARY KEY,
    decision_id          BIGINT NOT NULL REFERENCES decisions (id),
    revision             INTEGER NOT NULL DEFAULT 0,
    delivery_id          BIGINT REFERENCES deliveries (id),
    seq                  INTEGER NOT NULL,
    kind                 TEXT NOT NULL,
    options              JSONB NOT NULL,
    chosen               TEXT,
    forced               BOOLEAN NOT NULL DEFAULT false
);

CREATE INDEX IF NOT EXISTS draws_decision_idx ON draws (decision_id);

-- The rule that gave the probability of the day's draws (decide.py), and the version of every
-- catalogue the decision read: SHA-256 of each CSV file of catalogue/ and of the recommendations
-- and resources received from VU (5 November 2026).
ALTER TABLE decisions ADD COLUMN IF NOT EXISTS policy_rule TEXT;
ALTER TABLE decisions ADD COLUMN IF NOT EXISTS catalogue_version JSONB;
ALTER TABLE model_versions ADD COLUMN IF NOT EXISTS catalogue_version JSONB;

-- The composition design (implementation_plan_composition_2026-11-04.md, Phase 1; 6 November 2026):
-- the composition drawn (R recommendations, a resource counting as one, and Q reminders), the
-- probability of each admissible composition ({"R,Q": p}, also a row of draws with kind
-- 'composition') and each composition removed by a rule of the day ({"R,Q": rule}). On these rows
-- propensity, propensity_raw, propensity_learned, admissible_draws, draw_1, draw_2 and additional
-- are NULL; the role of a delivery is 'recommendation', 'resource' or 'reminder'.
ALTER TABLE decisions ADD COLUMN IF NOT EXISTS composition_r INTEGER;
ALTER TABLE decisions ADD COLUMN IF NOT EXISTS composition_q INTEGER;
ALTER TABLE decisions ADD COLUMN IF NOT EXISTS composition_probabilities JSONB;
ALTER TABLE decisions ADD COLUMN IF NOT EXISTS removed_compositions JSONB;
