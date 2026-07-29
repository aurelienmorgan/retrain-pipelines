#!/usr/bin/env bash
set -euo pipefail

COMPOSE_FILE="extra/minio-local/docker-compose.yml"
SERVICE="minio"
started_by_me=0

cleanup() {
  if [[ "$started_by_me" -eq 1 ]]; then
    docker compose -f "$COMPOSE_FILE" down
  fi
}
trap cleanup EXIT

if docker compose -f "$COMPOSE_FILE" ps -q "$SERVICE" | grep -q .; then
  :
else
  docker compose -f "$COMPOSE_FILE" up -d
  started_by_me=1
fi

# wait until MinIO is ready
until curl -sf http://localhost:9000/minio/health/live >/dev/null; do
  sleep 1
done

pytest tests/unit --cov=pkg_src/retrain_pipelines/dag_engine --cov-report=term-missing --cov-config=pkg_src/pyproject.toml
