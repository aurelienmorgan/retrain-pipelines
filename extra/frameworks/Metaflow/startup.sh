
#!/bin/bash

pip install udocker
udocker --allow-root install
useradd -m user

##############################
#    metaflow-service-db-1   #
##############################
MF_ROOT=${MF_ROOT:-/data}
PGDATA_DIR=${PGDATA_DIR:-${MF_ROOT}/pgdata}
echo "Creating and launching 'metaflow-service-db'"
udocker --allow-root pull postgres:11
udocker --allow-root create --name=metaflow-service-db-1  postgres:11
nohup udocker --allow-root run \
    -p=5432:5432 \
    -v $PGDATA_DIR/:/var/lib/postgresql/data \
    -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=postgres \
    -e POSTGRES_DB=postgres metaflow-service-db-1 \
    postgres -c log_statement=none -c wal_level=minimal -c max_wal_senders=0 \
            -c synchronous_commit=on -c checkpoint_timeout=30s \
    >> $MF_ROOT/logs/udocker_db.log 2>&1 &
# Check if PostgreSQL is accessible
apt-get install -y postgresql-client
RETRIES=10
until psql -h localhost -p 5432 -U postgres -c '\q'; do
  RETRIES=$((RETRIES-1))
  if [ $RETRIES -le 0 ]; then
    echo "Failed to connect to PostgreSQL, exiting."
    exit 1
  fi
  echo "Retrying to connect to PostgreSQL..."
  sleep 4
done
echo "PostgreSQL started successfully."
TABLE_EXISTS=$(psql -h localhost -p 5432 -U postgres -d postgres -t \
                    -c "SELECT EXISTS (SELECT 1 FROM information_schema.tables \
                        WHERE table_name = 'flows_v3');")
if [ "$TABLE_EXISTS" = " f" ]; then
    curl -o ${MF_ROOT}/db_init.sql \
        https://github.com/aurelienmorgan/retrain-pipelines/extra/frameworks/Metaflow/db_init.sql
    psql -h localhost -p 5432 -U postgres -d postgres -f ${MF_ROOT}/db_init.sql
else
    # db preventive maintenance
    ALL_INDEXES=$(psql -h localhost -p 5432 -U postgres -At -c \
    "SELECT \
        i.relname AS index_name, \
        t.relname AS table_name \
    FROM \
        pg_class i \
    JOIN \
        pg_index idx ON i.oid = idx.indexrelid \
    JOIN \
        pg_class t ON t.oid = idx.indrelid \
    WHERE \
        i.relkind = 'i' \
        AND t.relname IN ('runs_v3', 'flows_v3', 'steps_v3', \
                          'tasks_v3', 'metadata_v3', 'artifact_v3') \
    ;")
    echo $ALL_INDEXES

    if [ -n "$ALL_INDEXES" ]; then
        while IFS=$'|' read -r index_name table_name ; do
            output=$(psql -h localhost -p 5432 -U postgres \
                    -c "REINDEX INDEX \"$index_name\";" 2>&1)
            if [ $? -ne 0 ]; then
                echo "$output"
            else
                echo "reindexed $table_name.$index_name"
            fi
        done <<< "$ALL_INDEXES"
    fi
    # Let PostgreSQL deal with catalog reindexing
    psql -h localhost -p 5432 -U postgres -c "REINDEX DATABASE postgres;"
    psql -h localhost -p 5432 -U postgres -c "VACUUM (VERBOSE, ANALYZE);"
fi
##############################

##############################
#  metaflow-service-metadata #
##############################
udocker --allow-root pull \
    aurelienmorgan/metaflow-service-metadata:2.4.11
udocker --allow-root create \
    --name=metaflow-service-metadata-1 \
    aurelienmorgan/metaflow-service-metadata:2.4.11
udocker --allow-root run \
    -p=8888:8888 \
    -e LOGLEVEL=DEBUG \
    -e MF_DEFAULT_DATASTORE=local \
    -e MF_DATASTORE_SYSROOT_LOCAL=$MF_ROOT/local_datastore/ \
    -e MF_METADATA_DB_HOST=0.0.0.0 \
    -e MF_METADATA_DB_PORT=5432 \
    -e MF_METADATA_DB_USER=postgres \
    -e MF_METADATA_DB_PSWD=postgres \
    -e MF_METADATA_DB_NAME=postgres \
    -e MF_METADATA_PORT=${MF_METADATA_PORT:-8888} \
    -e MF_METADATA_HOST=${MF_METADATA_HOST:-0.0.0.0} \
    metaflow-service-metadata-1 metadata_service \
    >> $MF_ROOT/logs/udocker_service.log 2>&1 &
sleep 5;
##############################

##############################
#     metaflow-service-ui    #
##############################
udocker --allow-root pull \
    aurelienmorgan/metaflow-service-ui_backend:2.4.11
udocker --allow-root create \
    --name=metaflow-service-ui_backend-1 \
    aurelienmorgan/metaflow-service-ui_backend:2.4.11
udocker --allow-root run \
    -p=8083:8083 \
    -v $MF_ROOT/local_datastore/:$MF_ROOT/local_datastore/ \
    -e LOGLEVEL=DEBUG \
    -e MF_DEFAULT_DATASTORE=local \
    -e MF_DATAFSTORE_SYSROOT_LOCAL=$MF_ROOT/local_datastore/ \
    -e MF_METADATA_DB_HOST=0.0.0.0 \
    -e MF_METADATA_DB_PORT=5432 \
    -e MF_METADATA_DB_USER=postgres \
    -e MF_METADATA_DB_PSWD=postgres \
    -e MF_METADATA_DB_NAME=postgres \
    -e MF_UI_METADATA_PORT=${MF_UI_METADATA_PORT:-8083} \
    -e MF_UI_METADATA_HOST=${MF_UI_METADATA_HOST:-0.0.0.0} \
    -e MF_METADATA_DB_POOL_MIN=1 \
    -e MF_METADATA_DB_POOL_MAX=10 \
    -e METAFLOW_S3_RETRY_COUNT=0 \
    -e AIOPG_ECHO=0 \
    -e UI_ENABLED=0 \
    -e PREFETCH_RUNS_SINCE=2592000 \
    -e PREFETCH_RUNS_LIMIT=1 \
    -e S3_NUM_WORKERS=2 \
    -e CACHE_ARTIFACT_MAX_ACTIONS=1 \
    -e CACHE_DAG_MAX_ACTIONS=1 \
    -e CACHE_LOG_MAX_ACTIONS=1 \
    -e CACHE_ARTIFACT_STORAGE_LIMIT=16000000 \
    -e CACHE_DAG_STORAGE_LIMIT=16000000 \
    -e WS_POSTPROCESS_CONCURRENCY_LIMIT=8 \
    -e FEATURE_PREFETCH_DISABLE=0 \
    -e FEATURE_CACHE_DISABLE=0 \
    -e FEATURE_S3_DISABLE=0 \
    -e FEATURE_REFINE_DISABLE=0 \
    -e FEATURE_WS_DISABLE=0 \
    -e FEATURE_HEARTBEAT_DISABLE=0 \
    -e FEATURE_DB_LISTEN_DISABLE=0 \
    -e FEATURE_ARTIFACT_SEARCH=1 \
    -e FEATURE_FOREACH_VAR_SEARCH=1 \
    -e FEATURE_ARTIFACT_TABLE=1 \
    -e CUSTOM_QUICKLINKS=$CUSTOM_QUICKLINKS \
    -e NOTIFICATIONS=$NOTIFICATIONS \
    -e GA_TRACKING_ID=none \
    -e PLUGINS=$PLUGINS \
    -e AWS_PROFILE=$AWS_PROFILE \
    metaflow-service-ui_backend-1 ui_backend_service \
    >> $MF_ROOT/logs/udocker_ui_backend.log 2>&1 &
#!sleep 20;
##############################

##############################
#     Nginx reverse proxy    #
##############################
apt-get update
apt-get install -y nginx
cat <<EOT > /etc/nginx/sites-available/default
server {
    listen 7860  default_server;
    listen [::]:7860  default_server;

    server_name _;

    location / {
        # Metaflow UI 3000
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_cache_bypass \$http_upgrade;
        proxy_read_timeout 86400;
        proxy_redirect off;
    }

    location /ui_backend_service/ {
        # Metaflow UI backend service
        rewrite /ui_backend_service/(.*) /\$1 break;
        proxy_pass http://localhost:8083;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_cache_bypass \$http_upgrade;
        proxy_read_timeout 86400;
        proxy_redirect off;
    }

    location /service/ {
        # Metaflow API/SDK service
        rewrite /service/(.*) /\$1 break;
        proxy_pass http://localhost:8888;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_cache_bypass \$http_upgrade;
        proxy_read_timeout 86400;
        proxy_redirect off;
    }
}
EOT
service nginx start
##############################

##############################
#         metaflow-ui        #
##############################
udocker --allow-root pull \
    aurelienmorgan/metaflow-ui:1.3.13
udocker --allow-root create \
    --name=metaflow-ui-1 \
    aurelienmorgan/metaflow-ui:1.3.13
udocker --allow-root run \
    -p 3000:3000 \
    -e METAFLOW_SERVICE=$UI_EXT_URL/ui_backend_service/ \
    metaflow-ui-1 \
    >> $MF_ROOT/logs/udocker_ui.log 2>&1 &
sleep 5;
##############################
