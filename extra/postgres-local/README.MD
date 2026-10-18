serves for PostgreSQL-related integration-testing.
for metadata, e.g. : RP_METADATASTORE_URL (and RP_METADATASTORE_ASYNC_URL).

Exposes:
  - PostgreSQL at postgresql://localhost:5432

Note:
  - To start the Docker service, no .env is per-se "required".
    The Compose file provides development defaults:
      user:     postgres
      password: postgres
      database: postgres
  - You can override these values through environment variables:
      POSTGRES_USER
      POSTGRES_PASSWORD
      POSTGRES_DB
      POSTGRES_EXTERNAL_PORT
  - The PostgreSQL data is persisted under:
      ```bash
      ${LOCAL_PGSQL_DATA:-${RP_ASSETS_CACHE:-$HOME/.cache/retrain-pipelines}}/postgres-data
      ```
  - This service is intended for local testing only. The default credentials
    are deliberately convenient and must not be used for an exposed or
    shared PostgreSQL instance.
  - PostgreSQL 18 uses /var/lib/postgresql as the container volume mount
    point.
  - To reset the local database completely:
      ```bash
      docker compose down
      rm -rf "${LOCAL_PGSQL_DATA:-${RP_ASSETS_CACHE:-$HOME/.cache/retrain-pipelines}}/postgres-data"
      ```

Using it from the host machine:
```bash
export PGHOST="localhost"
export PGPORT="5432"
export PGUSER="postgres"
export PGPASSWORD="postgres"
export PGDATABASE="postgres"

# connection url =>
export DATABASE_URL="postgresql://$PGUSER:$PGPASSWORD@$PGHOST:$PGPORT/$PGDATABASE"
```

Using in a Python client, with `psycopg` :
```python
import os
import psycopg

with psycopg.connect(os.environ["DATABASE_URL"]) as connection:
    with connection.cursor() as cursor:
        cursor.execute("SELECT 1")
        print(cursor.fetchone())
```
