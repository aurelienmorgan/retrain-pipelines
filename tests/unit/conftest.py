"""
Shared test configuration.

- Captures truly-original global state once at collection time.
- Provides an autouse fixture that restores that state around every test,
  preventing RichLoggingController patches from leaking between tests
  and causing pytest teardown recursion.
- Provides shared test utilities (e.g., _SuppressLogger) for all test modules.
- ensures an ephemeral s3 bucket is created/disposed-of for each test session.
"""

import boto3
import builtins
import logging
import os
import sys

import pytest


# Correct import path for s3_utils
from retrain_pipelines.utils.s3_utils import ensure_s3_bucket


os.environ.setdefault("RP_METADATASTORE_URL", "sqlite:///:memory:")
os.environ.setdefault("RP_METADATASTORE_ASYNC_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("RP_WEB_SERVER_URL", "http://localhost:0")

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
    # --- PRE: restore before the test in case the previous one left a mess ---
    builtins.print = _ORIG_PRINT
    logging.getLogger = _ORIG_GETLOGGER
    sys.stdout.write = _ORIG_STDOUT_WRITE
    sys.stderr.write = _ORIG_STDERR_WRITE

    import retrain_pipelines.dag_engine.rp_logging as _rplog

    _rplog._global_controller = None

    yield

    # --- POST: always restore after the test --------------------------------
    builtins.print = _ORIG_PRINT
    logging.getLogger = _ORIG_GETLOGGER
    sys.stdout.write = _ORIG_STDOUT_WRITE
    sys.stderr.write = _ORIG_STDERR_WRITE
    _rplog._global_controller = None


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
#     create a local s3 bucket dedicated unit-test (with a test-session lifespan)     #
#######################################################################################

MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "http://localhost:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
# S3 bucket names must be lowercase, numbers, and hyphens only, no underscores
BUCKET_NAME = "retrain-pipelines-unit-tests"


@pytest.fixture(scope="session")
def bucket_name():
    """Return the S3 bucket name used for tests."""
    return BUCKET_NAME


@pytest.fixture(scope="session", autouse=True)
def minio_session(bucket_name):
    """
    Session‑scoped fixture that sets up a real MinIO environment.

    - Configures environment variables so that all boto3 clients in the test
      suite point to the local MinIO instance.
    - Creates the bucket `retrain-pipelines-unit-tests` using
      `s3_utils.ensure_s3_bucket`.
    - After all tests, force‑deletes all objects and the bucket itself,
      even if tests failed.

    This fixture is automatically used for the whole session (autouse=True).
    """
    original_env = {
        "AWS_ENDPOINT_URL": os.environ.get("AWS_ENDPOINT_URL"),
        "AWS_ACCESS_KEY_ID": os.environ.get("AWS_ACCESS_KEY_ID"),
        "AWS_SECRET_ACCESS_KEY": os.environ.get("AWS_SECRET_ACCESS_KEY"),
    }

    os.environ["AWS_ENDPOINT_URL"] = MINIO_ENDPOINT
    os.environ["AWS_ACCESS_KEY_ID"] = MINIO_ACCESS_KEY
    os.environ["AWS_SECRET_ACCESS_KEY"] = MINIO_SECRET_KEY

    # Use the project's utility to ensure the bucket exists
    ensure_s3_bucket(f"s3://{bucket_name}/")

    s3 = boto3.client("s3")

    yield

    # Force delete bucket and all objects
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
