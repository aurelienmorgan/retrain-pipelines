#!/usr/bin/env bash
set -euo pipefail

MINIO_COMPOSE_FILE="extra/minio-local/docker-compose.yml"
MINIO_SERVICE="minio"
minio_started_by_me=0

POSTGRES_COMPOSE_FILE="extra/postgres-local/docker-compose.yml"
POSTGRES_SERVICE="postgres"
postgres_started_by_me=0

cleanup() {
  local exit_code=$?

  if [[ "$postgres_started_by_me" -eq 1 ]]; then
    docker compose \
      -f "$POSTGRES_COMPOSE_FILE" \
      down || true
  fi

  if [[ "$minio_started_by_me" -eq 1 ]]; then
    docker compose \
      -f "$MINIO_COMPOSE_FILE" \
      down || true
  fi

  exit "$exit_code"
}

trap cleanup EXIT

if docker compose \
  -f "$MINIO_COMPOSE_FILE" \
  ps -q "$MINIO_SERVICE" | grep -q .; then
  :
  # service already up ; do nothing
else
  # detached mode
  docker compose \
    -f "$MINIO_COMPOSE_FILE" \
    up -d

  minio_started_by_me=1
fi


####################################################################################################
## Grant ownership to Linux user "postgres" to database files
## (PG data dir ; required by the PostgreSQL container)
####################################################################################################
prepare_postgres_data_dir() {
    POSTGRES_BASE="${LOCAL_PGSQL_DATA:-${RP_ASSETS_CACHE:-$HOME/.cache/retrain-pipelines}}"

    # Remove trailing "/" separators, while preserving "/".
    while [ "$POSTGRES_BASE" != "/" ] &&
          [ "${POSTGRES_BASE%/}" != "$POSTGRES_BASE" ]; do
        POSTGRES_BASE=${POSTGRES_BASE%/}
    done

    # Avoid producing "//postgres-data" when POSTGRES_BASE="/".
    if [ "$POSTGRES_BASE" = "/" ]; then
        POSTGRES_DATA="/postgres-data"
    else
        POSTGRES_DATA="$POSTGRES_BASE/postgres-data"
    fi

    POSTGRES_DIR="$POSTGRES_DATA/18/docker"

    # Create the directory only when it does not already exist.
    if [ ! -d "$POSTGRES_DIR" ]; then
        mkdir -p "$POSTGRES_DIR" 2>/dev/null ||
            sudo mkdir -p "$POSTGRES_DIR" ||
            {
                printf 'ERROR: unable to create directory: %s\n' "$POSTGRES_DIR" >&2
                return 1
            }
    fi

    POSTGRES_UID="$(
        docker run --rm \
            --entrypoint sh \
            postgres:18.4-alpine3.24 \
            -c 'id -u postgres'
    )" || return 1

    POSTGRES_GID="$(
        docker run --rm \
            --entrypoint sh \
            postgres:18.4-alpine3.24 \
            -c 'id -g postgres'
    )" || return 1

    if [ -z "$POSTGRES_UID" ] || [ -z "$POSTGRES_GID" ]; then
        printf 'ERROR: unable to resolve PostgreSQL UID/GID\n' >&2
        return 1
    fi

    # Change ONLY the owner UID.
    # Preserve the existing group and all existing permission bits.
    if [ "$(stat -c '%u' "$POSTGRES_DIR")" != "$POSTGRES_UID" ]; then
        chown -R "$POSTGRES_UID" "$POSTGRES_DIR" 2>/dev/null ||
            sudo chown -R "$POSTGRES_UID" "$POSTGRES_DIR"
    fi

    # check whether expected vs actual match
    ACTUAL_OWNERSHIP="$(
        stat -c '%u:%g' "$POSTGRES_DIR" 2>/dev/null ||
        stat -f '%u:%g' "$POSTGRES_DIR" 2>/dev/null ||
        true
    )"
    ACTUAL_PERMISSIONS="$(
        stat -c '%A' "$POSTGRES_DIR" 2>/dev/null ||
        stat -f '%Sp' "$POSTGRES_DIR" 2>/dev/null ||
        true
    )"
    ACTUAL_GID="$(
        stat -c '%g' "$POSTGRES_DIR" 2>/dev/null ||
        stat -f '%g' "$POSTGRES_DIR" 2>/dev/null ||
        true
    )"

    EXPECTED_OWNERSHIP="${POSTGRES_UID}:${ACTUAL_GID}"

    printf 'PostgreSQL container user: %s:%s\n' \
        "$POSTGRES_UID" "$POSTGRES_GID"
    printf 'Expected host ownership:   %s\n' \
        "$EXPECTED_OWNERSHIP"
    printf 'Actual host ownership:     %s\n' \
        "$ACTUAL_OWNERSHIP"
    printf 'Existing permissions:      %s\n' \
        "$ACTUAL_PERMISSIONS"

    if [ "$ACTUAL_OWNERSHIP" != "$EXPECTED_OWNERSHIP" ]; then
        printf '%s\n' \
            'ERROR: Failed to set PostgreSQL UID ownership.' \
            "       Directory: $POSTGRES_DIR" \
            "       Expected:  $EXPECTED_OWNERSHIP" \
            "       Actual:    $ACTUAL_OWNERSHIP" \
            '' \
            'The filesystem does not preserve the PostgreSQL container UID.' >&2

        case "$POSTGRES_DIR" in
            /mnt/*)
                printf '%s\n' \
                    '' \
                    'This path is under /mnt/, which is typically a Windows-mounted' \
                    'filesystem in WSL. PostgreSQL data must be stored on the WSL' \
                    'Linux filesystem, for example:' \
                    "  \$HOME/.cache/retrain-pipelines/postgres-data" \
                    '' \
                    'You can set LOCAL_PGSQL_DATA for this.' >&2
                ;;
        esac

        return 1
    fi

    printf '%s\n' \
        'Ownership: correct' \
        'Permissions: unchanged'
}


####################################################################################################
## Start PostgreSQL only when the service is not already running
####################################################################################################

if (
    docker compose \
        -f "$POSTGRES_COMPOSE_FILE" \
        ps -q "$POSTGRES_SERVICE" | grep -q .
); then
    :
    # Service already up ; do nothing.
else
    # Prepare the directory only when PostgreSQL was not already running.
    prepare_postgres_data_dir || exit 1

    # --wait implies detached mode in Docker Compose.
    docker compose \
        -f "$POSTGRES_COMPOSE_FILE" \
        up \
        --wait \
        --wait-timeout 90 \
        "$POSTGRES_SERVICE" ||
        exit 1

    postgres_started_by_me=1
fi

# Wait until MinIO is ready.
until curl -sf \
  http://localhost:9000/minio/health/live \
  >/dev/null; do
  sleep 1
done

pytest \
  tests/unit \
  -n auto --dist loadgroup \
  --timeout=45 \
  --cov=pkg_src/retrain_pipelines/dag_engine \
  --cov-report=term-missing \
  --cov-config=pkg_src/pyproject.toml
