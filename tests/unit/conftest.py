"""
Shared test configuration.

- ``postgres_session``: session-scoped autouse fixture that creates a
  worker-specific PostgreSQL test database (with PID naming)
  so xdist workers never race on DROP/CREATE/alembic.
- ``minio_session``: session-scoped autouse fixture that creates a worker-specific
  S3 bucket (with PID naming) so xdist workers never race
  on object deletion or bucket deletion during teardown.
- Registers xdist group markers so that ``pytest -n auto --dist loadgroup``
  keeps all tests from the same directory on the same worker.
- Captures truly-original global state once at collection time.
- Provides an autouse fixture that restores that state around every test,
  preventing RichLoggingController patches from leaking between tests
  and causing pytest teardown recursion.
- Provides shared test utilities (e.g., _SuppressLogger) for all test modules.
"""

import boto3
import builtins
import logging
import os
import sys

import psycopg2
import pytest

from retrain_pipelines.utils.s3_utils import ensure_s3_bucket


#######################################################################################
#     xdist worker identity ; evaluated once per worker process at import time        #
#######################################################################################

# ``PYTEST_XDIST_WORKER`` is set by pytest-xdist to e.g. "gw0", "gw1", "main".
_WORKER_ID = os.environ.get("PYTEST_XDIST_WORKER", "main")


#######################################################################################
#     xdist: group tests by directory so --dist loadgroup keeps DB state intact       #
#######################################################################################

GROUP_ROOT = os.path.dirname(os.path.abspath(__file__))


def pytest_collection_modifyitems(items):
    for item in items:
        test_file = os.path.abspath(os.fspath(item.fspath))
        relative_path = os.path.relpath(test_file, GROUP_ROOT)
        directory = os.path.dirname(relative_path) or "."

        # Stable group names on Linux and Windows.
        directory = directory.replace(os.sep, "/")

        item.add_marker(pytest.mark.xdist_group(name=f"unit:{directory}"))


#######################################################################################


#######################################################################################
#     custom logger  -  capture originals *before* any test runs                      #
#######################################################################################
_ORIG_PRINT = builtins.print
_ORIG_GETLOGGER = logging.getLogger
_ORIG_STDOUT_WRITE = sys.stdout.write
_ORIG_STDERR_WRITE = sys.stderr.write


@pytest.fixture(autouse=True)
def _restore_global_patches():
    """Restore every global patched by RichLoggingController before and after each test.

    RichLoggingController.activate() replaces builtins.print, sys.stdout.write,
    sys.stderr.write, and logging.getLogger with module-level wrappers.
    If a test fails mid-flight those replacements are left in place, which
    causes infinite recursion when pytest itself calls logging.getLogger()
    during teardown.
    The herein fixture guarantees a clean slate for every test.
    """
    builtins.print = _ORIG_PRINT
    logging.getLogger = _ORIG_GETLOGGER
    sys.stdout.write = _ORIG_STDOUT_WRITE
    sys.stderr.write = _ORIG_STDERR_WRITE

    import retrain_pipelines.dag_engine.rp_logging as _rplog

    _rplog._global_controller = None
    # Scrubbing "nb_* names" here =>
    # prevents hidden cross-test dependencies through _rplog.__dict__.
    for _name in ("nb_activate", "nb_deactivate", "nb_console_print"):
        _rplog.__dict__.pop(_name, None)

    yield

    builtins.print = _ORIG_PRINT
    logging.getLogger = _ORIG_GETLOGGER
    sys.stdout.write = _ORIG_STDOUT_WRITE
    sys.stderr.write = _ORIG_STDERR_WRITE
    _rplog._global_controller = None
    for _name in ("nb_activate", "nb_deactivate", "nb_console_print"):
        _rplog.__dict__.pop(_name, None)


@pytest.fixture
def suppress_logger():
    class _SuppressLogger:
        def __init__(self, module_name):
            self._module_logger = logging.getLogger(module_name)

        def __enter__(self):
            self._original_level = self._module_logger.level
            self._module_logger.setLevel(logging.CRITICAL)
            self._handler = logging.NullHandler()
            self._module_logger.addHandler(self._handler)
            return self

        def __exit__(self, *_):
            self._module_logger.removeHandler(self._handler)
            self._module_logger.setLevel(self._original_level)

    return _SuppressLogger


#######################################################################################


#######################################################################################
#     PostgreSQL session: to avoid xdist race conditions, per-worker fresh test DB    #
#     + alembic schema init (on first retrain-pipelines package import)               #
#######################################################################################

_PG_TEST_DB = f"postgres_integration_tests_{_WORKER_ID}"


def _admin_conn():
    """Open a psycopg2 connection to the default 'postgres' admin database."""
    return psycopg2.connect(
        host=os.environ.get("PGHOST", "localhost"),
        port=os.environ.get("PGPORT", "5432"),
        user=os.environ.get("PGUSER", "postgres"),
        password=os.environ.get("PGPASSWORD", "postgres"),
        dbname="postgres",
    )


def _terminate_test_db_connections(cur, dbname):
    cur.execute(
        "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
        "WHERE datname = %s AND pid <> pg_backend_pid()",
        (dbname,),
    )


@pytest.fixture(scope="session", autouse=True)
def postgres_session():
    """Session-scoped autouse fixture for PostgreSQL test database lifecycle.

    Creates a worker-specific database, runs alembic migrations against it,
    and drops it on teardown.  Mirrors ``minio_session`` in structure.

    Note
    ----
    Each xdist worker (``PYTEST_XDIST_WORKER``) gets its own database
    (``postgres_integration_tests_<worker_id>``) computed at module import time.
    Without per-worker naming all workers would race on DROP/CREATE and on
    the alembic ``INSERT INTO alembic_version`` step, causing
    ``UniqueViolation`` errors that poison every test in the affected session,
    including tests that never touch the database.
    """
    original_env = {
        "PGHOST": os.environ.get("PGHOST"),
        "PGPORT": os.environ.get("PGPORT"),
        "PGUSER": os.environ.get("PGUSER"),
        "PGPASSWORD": os.environ.get("PGPASSWORD"),
        "PGDATABASE": os.environ.get("PGDATABASE"),
    }

    os.environ["PGHOST"] = os.environ.get("PGHOST", "localhost")
    os.environ["PGPORT"] = os.environ.get("PGPORT", "5432")
    os.environ["PGUSER"] = os.environ.get("PGUSER", "postgres")
    os.environ["PGPASSWORD"] = os.environ.get("PGPASSWORD", "postgres")
    os.environ["PGDATABASE"] = _PG_TEST_DB

    conn = _admin_conn()
    conn.autocommit = True
    cur = conn.cursor()
    _terminate_test_db_connections(cur, _PG_TEST_DB)
    cur.execute(f'DROP DATABASE IF EXISTS "{_PG_TEST_DB}"')
    cur.execute(f'CREATE DATABASE "{_PG_TEST_DB}"')
    cur.close()
    conn.close()

    pg_url = (
        f"postgresql://{os.environ['PGUSER']}:{os.environ['PGPASSWORD']}"
        f"@{os.environ['PGHOST']}:{os.environ['PGPORT']}/{_PG_TEST_DB}"
    )
    _prev_url = os.environ.get("RP_METADATASTORE_URL")
    os.environ["RP_METADATASTORE_URL"] = pg_url

    import retrain_pipelines  # noqa: F401  # <= runs ``run_alembic_upgrade_once``
    #                                            on test worker DB

    if _prev_url is None:
        os.environ.pop("RP_METADATASTORE_URL", None)
    else:
        os.environ["RP_METADATASTORE_URL"] = _prev_url

    yield

    try:
        conn = _admin_conn()
        conn.autocommit = True
        cur = conn.cursor()
        _terminate_test_db_connections(cur, _PG_TEST_DB)
        cur.execute(f'DROP DATABASE IF EXISTS "{_PG_TEST_DB}"')
        cur.close()
        conn.close()
    except Exception:
        pass
    finally:
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


#######################################################################################


#######################################################################################
#     MinIO session: to avoid xdist race conditions, per-worker fresh local s3 bucket #
#     (with a test-worker-session lifespan)                                           #
#######################################################################################

MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "http://localhost:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
# S3 bucket names must be lowercase, numbers, and hyphens only, no underscores
# Worker IDs ("main", "gw0", "gw1", …) are already safe characters.
BUCKET_NAME = f"retrain-pipelines-integration-tests-{_WORKER_ID}"


@pytest.fixture(scope="session")
def bucket_name():
    """Return the worker-specific S3 bucket name used for this session."""
    return BUCKET_NAME


@pytest.fixture(scope="session", autouse=True)
def minio_session(bucket_name):
    """Session-scoped fixture that sets up a real MinIO environment.

    - Configures environment variables so that all boto3 clients in the test
    suite point to the local MinIO instance.
    - Creates the test-worker-specific bucket via ``s3_utils.ensure_s3_bucket``,
    - And force-deletes all objects and the bucket itself on teardown
      (even if tests failed).

    This fixture is automatically used for the whole test worker session
    (autouse=True).

    Note
    ----
    Each xdist worker (``PYTEST_XDIST_WORKER``) gets its own bucket
    (``retrain-pipelines-integration-tests-<worker_id>``). Without per-worker
    naming all workers share the same bucket and race during teardown: one
    worker's ``delete_objects`` / ``delete_bucket`` call can land while
    another worker's tests are still writing to the bucket, causing spurious
    ``NoSuchBucket`` / ``NoSuchKey`` errors and leaving the bucket undropped.
    """
    original_env = {
        "AWS_ENDPOINT_URL": os.environ.get("AWS_ENDPOINT_URL"),
        "AWS_ACCESS_KEY_ID": os.environ.get("AWS_ACCESS_KEY_ID"),
        "AWS_SECRET_ACCESS_KEY": os.environ.get("AWS_SECRET_ACCESS_KEY"),
    }

    os.environ["AWS_ENDPOINT_URL"] = MINIO_ENDPOINT
    os.environ["AWS_ACCESS_KEY_ID"] = MINIO_ACCESS_KEY
    os.environ["AWS_SECRET_ACCESS_KEY"] = MINIO_SECRET_KEY

    ensure_s3_bucket(f"s3://{bucket_name}/")

    s3 = boto3.client("s3")

    yield

    try:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket_name):
            if "Contents" in page:
                objects = [{"Key": obj["Key"]} for obj in page["Contents"]]
                s3.delete_objects(Bucket=bucket_name, Delete={"Objects": objects})
        s3.delete_bucket(Bucket=bucket_name)
    except Exception:
        pass
    finally:
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


#######################################################################################
