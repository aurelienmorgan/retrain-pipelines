
-- compiled from metaflow-service-2.4.11/services/migration_service/migration_files

/***************************************************
*                  1_create_tables                 *
***************************************************/
BEGIN;
DO $$
BEGIN
    RAISE NOTICE 'Create flows_v3 table';
    EXECUTE 'CREATE TABLE IF NOT EXISTS flows_v3 (
        flow_id VARCHAR(255) PRIMARY KEY,
        user_name VARCHAR(255),
        ts_epoch BIGINT NOT NULL,
        tags JSONB,
        system_tags JSONB
    )';

    RAISE NOTICE 'Create runs_v3 table';
    EXECUTE 'CREATE TABLE IF NOT EXISTS runs_v3 (
        flow_id VARCHAR(255) NOT NULL,
        run_number SERIAL NOT NULL,
        user_name VARCHAR(255),
        ts_epoch BIGINT NOT NULL,
        tags JSONB,
        system_tags JSONB,
        PRIMARY KEY (flow_id, run_number),
        FOREIGN KEY (flow_id) REFERENCES flows_v3 (flow_id)
    )';

    RAISE NOTICE 'Create steps_v3 table';
    EXECUTE 'CREATE TABLE IF NOT EXISTS steps_v3 (
        flow_id VARCHAR(255) NOT NULL,
        run_number BIGINT NOT NULL,
        step_name VARCHAR(255) NOT NULL,
        user_name VARCHAR(255),
        ts_epoch BIGINT NOT NULL,
        tags JSONB,
        system_tags JSONB,
        PRIMARY KEY (flow_id, run_number, step_name),
        FOREIGN KEY (flow_id, run_number)
            REFERENCES runs_v3 (flow_id, run_number)
    )';

    RAISE NOTICE 'Create tasks_v3 table';
    EXECUTE 'CREATE TABLE IF NOT EXISTS tasks_v3 (
        flow_id VARCHAR(255) NOT NULL,
        run_number BIGINT NOT NULL,
        step_name VARCHAR(255) NOT NULL,
        task_id BIGSERIAL PRIMARY KEY,
        user_name VARCHAR(255),
        ts_epoch BIGINT NOT NULL,
        tags JSONB,
        system_tags JSONB,
        FOREIGN KEY (flow_id, run_number, step_name)
            REFERENCES steps_v3 (flow_id, run_number, step_name)
    )';

    RAISE NOTICE 'Create metadata_v3 table';
    EXECUTE 'CREATE TABLE IF NOT EXISTS metadata_v3 (
        flow_id VARCHAR(255),
        run_number BIGINT NOT NULL,
        step_name VARCHAR(255) NOT NULL,
        task_id BIGINT NOT NULL,
        id BIGSERIAL NOT NULL,
        field_name VARCHAR(255) NOT NULL,
        value TEXT NOT NULL,
        type VARCHAR(255) NOT NULL,
        user_name VARCHAR(255),
        ts_epoch BIGINT NOT NULL,
        tags JSONB,
        system_tags JSONB,
        PRIMARY KEY (flow_id, run_number, step_name,
                     task_id, field_name)
    )';

    RAISE NOTICE 'Create artifact_v3 table';
    EXECUTE 'CREATE TABLE IF NOT EXISTS artifact_v3 (
        flow_id VARCHAR(255) NOT NULL,
        run_number BIGINT NOT NULL,
        step_name VARCHAR(255) NOT NULL,
        task_id BIGINT NOT NULL,
        name VARCHAR(255) NOT NULL,
        location VARCHAR(255) NOT NULL,
        ds_type VARCHAR(255) NOT NULL,
        sha VARCHAR(255),
        type VARCHAR(255),
        content_type VARCHAR(255),
        user_name VARCHAR(255),
        attempt_id SMALLINT NOT NULL,
        ts_epoch BIGINT NOT NULL,
        tags JSONB,
        system_tags JSONB,
        PRIMARY KEY (flow_id, run_number, step_name,
                    task_id, attempt_id, name)
    )';
END $$;
COMMIT;
/**************************************************/


/***************************************************
*          20200603104139_add_str_id_cols          *
***************************************************/
BEGIN;
DO $$
BEGIN
    RAISE NOTICE 'Alter table runs_v3';
    EXECUTE 'ALTER TABLE runs_v3
             ADD COLUMN run_id VARCHAR(255)';

    RAISE NOTICE 'Alter table runs_v3';
    EXECUTE 'ALTER TABLE runs_v3
             ADD COLUMN last_heartbeat_ts BIGINT';

    RAISE NOTICE 'Alter table runs_v3';
    EXECUTE 'ALTER TABLE runs_v3
             ADD CONSTRAINT runs_v3_flow_id_run_id_key UNIQUE (
                flow_id, run_id)';

    RAISE NOTICE 'Alter table steps_v3';
    EXECUTE 'ALTER TABLE steps_v3
             ADD COLUMN run_id VARCHAR(255)';

    RAISE NOTICE 'Alter table steps_v3';
    EXECUTE 'ALTER TABLE steps_v3
             ADD CONSTRAINT steps_v3_flow_id_run_id_step_name_key UNIQUE (
                flow_id, run_id, step_name)';

    RAISE NOTICE 'Alter table tasks_v3';
    EXECUTE 'ALTER TABLE tasks_v3
             ADD COLUMN run_id VARCHAR(255)';

    RAISE NOTICE 'Alter table tasks_v3';
    EXECUTE 'ALTER TABLE tasks_v3
             ADD COLUMN task_name VARCHAR(255)';

    RAISE NOTICE 'Alter table tasks_v3';
    EXECUTE 'ALTER TABLE tasks_v3
             ADD COLUMN last_heartbeat_ts BIGINT';

    RAISE NOTICE 'Alter table tasks_v3';
    EXECUTE 'ALTER TABLE tasks_v3
             ADD CONSTRAINT tasks_v3_flow_id_run_number_step_name_task_name_key UNIQUE (
                flow_id, run_number, step_name, task_name)';

    RAISE NOTICE 'Alter table metadata_v3';
    EXECUTE 'ALTER TABLE metadata_v3
             ADD COLUMN run_id VARCHAR(255)';

    RAISE NOTICE 'Alter table metadata_v3';
    EXECUTE 'ALTER TABLE metadata_v3
             ADD COLUMN task_name VARCHAR(255)';

    RAISE NOTICE 'Alter table artifact_v3';
    EXECUTE 'ALTER TABLE artifact_v3
             ADD COLUMN run_id VARCHAR(255)';

    RAISE NOTICE 'Alter table artifact_v3';
    EXECUTE 'ALTER TABLE artifact_v3
             ADD COLUMN task_name VARCHAR(255)';
END $$;
COMMIT;
/**************************************************/


/***************************************************
*    20201002000616_update_metadata_primary_key    *
***************************************************/
BEGIN;
DO $$
BEGIN
    RAISE NOTICE 'Alter table metadata_v3';
    EXECUTE 'ALTER TABLE metadata_v3
             ADD CONSTRAINT metadata_v3_primary_key UNIQUE (
                id,flow_id, run_number, step_name, task_id, field_name)';

    RAISE NOTICE 'Create index metadata_v3_akey';
    EXECUTE 'CREATE INDEX metadata_v3_akey ON metadata_v3(
                flow_id, run_number, step_name, task_id, field_name)';

    RAISE NOTICE 'Alter table metadata_v3';
    EXECUTE 'ALTER TABLE metadata_v3
             DROP CONSTRAINT metadata_v3_pkey';

    RAISE NOTICE 'Alter table metadata_v3';
    EXECUTE 'ALTER TABLE metadata_v3
             ADD PRIMARY KEY (
                id,flow_id, run_number, step_name, task_id, field_name)';

    RAISE NOTICE 'Alter table metadata_v3';
    EXECUTE 'ALTER TABLE metadata_v3
             DROP CONSTRAINT metadata_v3_primary_key';
END $$;
COMMIT;
/**************************************************/


/***************************************************
*   20210202145952_add_runs_idx_ts_epoch_flow_id   *
***************************************************/
BEGIN;
DO $$
BEGIN
    RAISE NOTICE 'Create index runs_v3_idx_ts_epoch';
    EXECUTE 'CREATE INDEX IF NOT EXISTS runs_v3_idx_ts_epoch ON runs_v3 (ts_epoch)';

    RAISE NOTICE 'Create index runs_v3_idx_gin_tags_combined';
    EXECUTE 'CREATE INDEX IF NOT EXISTS runs_v3_idx_gin_tags_combined
             ON runs_v3 USING gin ((tags || system_tags))';

    RAISE NOTICE 'Create index runs_v3_idx_flow_id_asc_ts_epoch_desc';
    EXECUTE 'CREATE INDEX IF NOT EXISTS runs_v3_idx_flow_id_asc_ts_epoch_desc
             ON runs_v3 (flow_id ASC, ts_epoch DESC)';

    RAISE NOTICE 'Create index runs_v3_idx_user_asc_ts_epoch_desc';
    EXECUTE 'CREATE INDEX IF NOT EXISTS "runs_v3_idx_user_asc_ts_epoch_desc"
        ON "runs_v3" (
            (CASE
                WHEN "system_tags" ? (''user:'' || "user_name")
                THEN "user_name"
                ELSE NULL
            END) ASC, "ts_epoch" DESC
        )';
END $$;
COMMIT;
/**************************************************/


/***************************************************
*         20211202100726_add_str_id_indices        *
***************************************************/
\echo Create indexes concurrently :
\echo - runs_v3_idx_str_ids_primary_key
\echo - steps_v3_idx_str_ids_primary_key
\echo - metadata_v3_idx_str_ids_primary_key
\echo - artifact_v3_idx_str_ids_primary_key

CREATE INDEX CONCURRENTLY IF NOT EXISTS runs_v3_idx_str_ids_primary_key
ON runs_v3 (flow_id, run_id)
WHERE
  run_id IS NOT NULL;

CREATE INDEX CONCURRENTLY IF NOT EXISTS steps_v3_idx_str_ids_primary_key
ON steps_v3 (flow_id, run_id, step_name)
WHERE
  run_id IS NOT NULL;

CREATE INDEX CONCURRENTLY IF NOT EXISTS metadata_v3_idx_str_ids_primary_key
ON metadata_v3 (
  id,
  flow_id,
  run_id,
  step_name,
  task_name,
  field_name
)
WHERE
  run_id IS NOT NULL
  AND task_name IS NOT NULL;

CREATE INDEX CONCURRENTLY IF NOT EXISTS artifact_v3_idx_str_ids_primary_key
ON artifact_v3 (
  flow_id,
  run_id,
  step_name,
  task_name,
  attempt_id,
  name
)
WHERE
  run_id IS NOT NULL
  AND task_name IS NOT NULL;
/**************************************************/


/***************************************************
*        20220503175500_add_run_epoch_index        *
***************************************************/
\echo 'Create index concurrently : runs_v3_idx_epoch_ts_desc'
CREATE INDEX CONCURRENTLY IF NOT EXISTS runs_v3_idx_epoch_ts_desc
  ON runs_v3 (ts_epoch DESC);
/**************************************************/


/***************************************************
*      20230118020300_drop_partial_indexes.sql     *
***************************************************/
\echo Create indexes concurrently :
\echo - runs_v3_idx_str_ids_primary_key_v2
\echo - steps_v3_idx_str_ids_primary_key_v2
\echo - tasks_v3_idx_flow_id_run_id_step_name_task_name_v2
\echo - metadata_v3_idx_str_ids_a_key
\echo - metadata_v3_idx_str_ids_a_key_with_task_id
\echo - artifact_v3_idx_str_ids_primary_key_v2
\echo - artifact_v3_idx_str_ids_primary_key_with_task_id

\echo Drop indexes concurrently : runs_v3_idx_str_ids_primary_key
\echo - steps_v3_idx_str_ids_primary_key
\echo - tasks_v3_idx_flow_id_run_id_step_name_task_name
\echo - metadata_v3_idx_str_ids_primary_key
\echo - artifact_v3_idx_str_ids_primary_key

CREATE INDEX CONCURRENTLY IF NOT EXISTS runs_v3_idx_str_ids_primary_key_v2
ON runs_v3 (flow_id, run_id);

DROP INDEX CONCURRENTLY IF EXISTS runs_v3_idx_str_ids_primary_key;

CREATE INDEX CONCURRENTLY IF NOT EXISTS steps_v3_idx_str_ids_primary_key_v2
ON steps_v3 (flow_id, run_id, step_name);

DROP INDEX CONCURRENTLY IF EXISTS steps_v3_idx_str_ids_primary_key;

CREATE INDEX CONCURRENTLY IF NOT EXISTS tasks_v3_idx_flow_id_run_id_step_name_task_name_v2
ON tasks_v3(flow_id, run_id, step_name, task_name);

DROP INDEX CONCURRENTLY IF EXISTS tasks_v3_idx_flow_id_run_id_step_name_task_name;

CREATE INDEX CONCURRENTLY IF NOT EXISTS metadata_v3_idx_str_ids_a_key
ON metadata_v3 (
  flow_id,
  run_id,
  step_name,
  task_name,
  field_name
);

CREATE INDEX CONCURRENTLY IF NOT EXISTS metadata_v3_idx_str_ids_a_key_with_task_id
ON metadata_v3 (
  flow_id,
  run_id,
  step_name,
  task_id,
  field_name
);

DROP INDEX CONCURRENTLY IF EXISTS metadata_v3_idx_str_ids_primary_key;

CREATE INDEX CONCURRENTLY IF NOT EXISTS artifact_v3_idx_str_ids_primary_key_v2
ON artifact_v3 (
  flow_id,
  run_id,
  step_name,
  task_name,
  attempt_id,
  name
);

CREATE INDEX CONCURRENTLY IF NOT EXISTS artifact_v3_idx_str_ids_primary_key_with_task_id
ON artifact_v3 (
  flow_id,
  run_id,
  step_name,
  task_id,
  attempt_id,
  name
);

DROP INDEX CONCURRENTLY IF EXISTS artifact_v3_idx_str_ids_primary_key;
/**************************************************/

