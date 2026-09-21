"""
Parametrized local filesystem and S3 fixtures for the unit-test subtree.

Provides:
  ``metadata_root``: parametrized fixture providing a local tmp_path
    and an S3 MinIO path.
  ``assets_cache``: parametrized fixture that monkeypatches a local tmp_path
    and an S3 MinIO path to the RP_ASSETS_CACHE environment variable.
  ``artifacts_store_root``: parametrized fixture routing RP_ARTIFACTS_STORE
    to a local tmp_path or an S3 MinIO path, normalized with a trailing
    separator to match Config.get_artifacts_store_root() behavior.
  ``web_server_logs_root``: parametrized fixture routing RP_WEB_SERVER_LOGS
    to a local tmp_path or an S3 MinIO path, normalized with a trailing
    separator to match Config.get_web_server_logs_root() behavior.
  ``runtime_env``: parametrized fixture routing RP_ARTIFACTS_STORE_ROOT and
    RP_ASSETS_CACHE to a local tmp_path or an S3 MinIO path, and resets the
    DAG execution context variable for runtime tests.
"""

import os
import pytest

from retrain_pipelines.dag_engine.core.core import _dag_execution_context_var


@pytest.fixture(params=["local", "s3"])
def metadata_root(request, tmp_path, bucket_name):
    """Parametrized fixture providing a local tmp_path and an S3 MinIO path."""
    if request.param == "local":
        return os.path.join(tmp_path, "metadata")
    else:
        return f"s3://{bucket_name}/metadata"


@pytest.fixture(params=["local", "s3"])
def assets_cache(request, tmp_path, monkeypatch, bucket_name):
    """Parametrized fixture routing RP_ASSETS_CACHE to local or S3."""
    if request.param == "local":
        cache_root = os.path.join(tmp_path, ".cache")
    else:
        cache_root = f"s3://{bucket_name}/.cache/"
    monkeypatch.setenv("RP_ASSETS_CACHE", cache_root)
    yield cache_root


@pytest.fixture(params=["local", "s3"])
def artifacts_store_root(request, tmp_path, monkeypatch, bucket_name):
    """Parametrized fixture routing RP_ARTIFACTS_STORE to local or S3."""
    if request.param == "local":
        # Normalize to end with a path separator to match Config.get_artifacts_store_root()
        path = str(tmp_path / "artifacts") + os.sep
    else:
        path = f"s3://{bucket_name}/{tmp_path.name}/artifacts/"
    monkeypatch.setenv("RP_ARTIFACTS_STORE", path)
    yield path


@pytest.fixture(params=["local", "s3"])
def web_server_logs_root(request, tmp_path, monkeypatch, bucket_name):
    """Parametrized fixture routing RP_WEB_SERVER_LOGS to local or S3."""
    if request.param == "local":
        # Normalize to end with a path separator to match Config.get_web_server_logs_root()
        path = os.path.join(tmp_path, "web_server_logs").rstrip(os.sep) + os.sep
    else:
        path = f"s3://{bucket_name}/web_server_logs/"
    monkeypatch.setenv("RP_WEB_SERVER_LOGS", path)
    yield path


@pytest.fixture(params=["local", "s3"])
def runtime_env(request, tmp_path, monkeypatch, bucket_name):
    """Parametrized fixture routing RP_ARTIFACTS_STORE_ROOT and RP_ASSETS_CACHE to local or S3."""
    if request.param == "local":
        artifacts_root = os.path.join(tmp_path, "artifacts")
        cache_root = os.path.join(tmp_path, ".cache")
    else:
        artifacts_root = f"s3://{bucket_name}/artifacts"
        cache_root = f"s3://{bucket_name}/.cache"

    monkeypatch.setenv("RP_ARTIFACTS_STORE_ROOT", artifacts_root)
    monkeypatch.setenv("RP_ASSETS_CACHE", cache_root)

    _dag_execution_context_var.set(None)

    yield

    _dag_execution_context_var.set(None)
