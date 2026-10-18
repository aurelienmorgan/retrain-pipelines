"""
Unit and Integration tests for the `config` module.
"""

import os
import pytest

from retrain_pipelines.dag_engine.config import Config, NotSupportedError
from retrain_pipelines.utils.s3_utils import is_s3_path


@pytest.fixture(params=["local", "s3"])
def artifacts_root(request, tmp_path, monkeypatch, bucket_name):
    """Parametrized fixture routing RP_ARTIFACTS_STORE to local or S3."""
    if request.param == "local":
        path = str(tmp_path / "artifacts")
    else:
        path = f"s3://{bucket_name}/artifacts/"
    monkeypatch.setenv("RP_ARTIFACTS_STORE", path)
    yield path


@pytest.fixture(params=["local", "s3"])
def logs_root(request, tmp_path, monkeypatch, bucket_name):
    """Parametrized fixture routing RP_WEB_SERVER_LOGS to local or S3."""
    if request.param == "local":
        path = str(tmp_path / "logs")
    else:
        path = f"s3://{bucket_name}/logs/"
    monkeypatch.setenv("RP_WEB_SERVER_LOGS", path)
    yield path


class TestEnsureDir:
    """Tests for the `_ensure_dir` static method."""

    def test_creates_directory(self, tmp_path):
        """_ensure_dir should create the directory if it does not exist."""
        new_dir = tmp_path / "sub" / "dir"
        assert not new_dir.exists()
        Config._ensure_dir(str(new_dir))
        assert new_dir.exists()

    def test_handles_permission_error(self, monkeypatch):
        """_ensure_dir should raise PermissionError when os.makedirs raises it."""

        def mock_makedirs(path, exist_ok=False):
            raise PermissionError("mock permission denied")

        monkeypatch.setattr(os, "makedirs", mock_makedirs)
        with pytest.raises(PermissionError):
            Config._ensure_dir("/some/path")

    def test_handles_generic_exception(self, monkeypatch):
        """_ensure_dir should re-raise any other exception from os.makedirs."""

        def mock_makedirs(path, exist_ok=False):
            raise OSError("mock OS error")

        monkeypatch.setattr(os, "makedirs", mock_makedirs)
        with pytest.raises(OSError):
            Config._ensure_dir("/some/path")


class TestAssetsCacheRoot:
    """Tests for `get_assets_cache_root`."""

    def test_default(self, tmp_path, monkeypatch):
        """Default assets cache root should be under ~/.cache/retrain-pipelines/."""
        monkeypatch.delenv("RP_ASSETS_CACHE", raising=False)
        monkeypatch.setattr(
            os.path,
            "expanduser",
            lambda p: str(tmp_path / ".cache" / "retrain-pipelines"),
        )
        root = Config.get_assets_cache_root()
        expected = str(tmp_path / ".cache" / "retrain-pipelines") + os.sep
        assert root == expected
        assert (tmp_path / ".cache" / "retrain-pipelines").exists()

    def test_env_var(self, assets_cache):
        """Assets cache root should be taken from RP_ASSETS_CACHE env var."""
        root = Config.get_assets_cache_root()
        if is_s3_path(assets_cache):
            assert root == assets_cache
        else:
            expected = assets_cache + os.sep
            assert root == expected
            assert os.path.exists(assets_cache)


class TestArtifactsStoreRoot:
    """Tests for `get_artifacts_store_root`."""

    def test_default(self, tmp_path, monkeypatch):
        """Default artifacts store should be under assets_cache/artifacts/."""
        cache_path = tmp_path / "cache"
        monkeypatch.setenv("RP_ASSETS_CACHE", str(cache_path))
        root = Config.get_artifacts_store_root()
        expected = str(cache_path / "artifacts") + os.sep
        assert root == expected
        assert (cache_path / "artifacts").exists()

    def test_env_var(self, artifacts_root):
        """Artifacts store root should be taken from RP_ARTIFACTS_STORE env var."""
        root = Config.get_artifacts_store_root()
        if is_s3_path(artifacts_root):
            assert root == artifacts_root
        else:
            expected = artifacts_root + os.sep
            assert root == expected
            assert os.path.exists(artifacts_root)


class TestWebServerLogsRoot:
    """Tests for `get_web_server_logs_root`."""

    def test_default(self, tmp_path, monkeypatch):
        """Default web server logs should be under assets_cache/logs/web_server/."""
        cache_path = tmp_path / "cache"
        monkeypatch.setenv("RP_ASSETS_CACHE", str(cache_path))
        root = Config.get_web_server_logs_root()
        expected = str(cache_path / "logs" / "web_server") + os.sep
        assert root == expected
        assert (cache_path / "logs" / "web_server").exists()

    def test_env_var(self, logs_root):
        """Web server logs root should be taken from RP_WEB_SERVER_LOGS env var."""
        root = Config.get_web_server_logs_root()
        if is_s3_path(logs_root):
            assert root == logs_root
        else:
            expected = logs_root + os.sep
            assert root == expected
            assert os.path.exists(logs_root)


class TestPorts:
    """Tests for port getters: `get_web_server_port`."""

    def test_web_server_port_default(self, monkeypatch):
        """Default web server port should be 5001."""
        monkeypatch.delenv("RP_WEB_SERVER_PORT", raising=False)
        assert Config.get_web_server_port() == 5001

    def test_web_server_port_env_var(self, monkeypatch):
        """Web server port should be taken from RP_WEB_SERVER_PORT env var."""
        monkeypatch.setenv("RP_WEB_SERVER_PORT", "8080")
        assert Config.get_web_server_port() == 8080


class TestWebServerURL:
    """Tests for `get_web_server_url`."""

    def test_default(self, monkeypatch):
        """Default web server URL should be http://localhost:5001/ (no trailing slash)."""
        monkeypatch.delenv("RP_WEB_SERVER_URL", raising=False)
        monkeypatch.delenv("RP_WEB_SERVER_PORT", raising=False)
        url = Config.get_web_server_url()
        assert url == "http://localhost:5001"

    def test_uses_port_env(self, monkeypatch):
        """When RP_WEB_SERVER_URL is not set, port from RP_WEB_SERVER_PORT should be used."""
        monkeypatch.delenv("RP_WEB_SERVER_URL", raising=False)
        monkeypatch.setenv("RP_WEB_SERVER_PORT", "8000")
        url = Config.get_web_server_url()
        assert url == "http://localhost:8000"

    def test_env_var_overrides(self, monkeypatch):
        """RP_WEB_SERVER_URL should override everything and strip trailing slash."""
        monkeypatch.setenv("RP_WEB_SERVER_URL", "https://example.com:8443/")
        url = Config.get_web_server_url()
        assert url == "https://example.com:8443"


class TestMetadataStoreURL:
    """Tests for `get_metadatastore_url`."""

    def test_default(self, assets_cache, monkeypatch):
        """Default metadata store URL should be sqlite:// with timeout and assets cache."""
        monkeypatch.delenv("RP_METADATASTORE_URL", raising=False)
        if is_s3_path(assets_cache):
            with pytest.raises(NotSupportedError):
                Config.get_metadatastore_url()
        else:
            url = Config.get_metadatastore_url()
            expected = (
                f"sqlite:///{assets_cache}{os.sep}local_metadatastore.db?timeout=10.0"
            )
            assert url == expected

    def test_env_var(self, tmp_path, monkeypatch):
        """Metadata store URL should be taken from RP_METADATASTORE_URL env var."""
        monkeypatch.setenv("RP_ASSETS_CACHE", str(tmp_path))
        custom_url = "postgresql://user:pass@host/db"
        monkeypatch.setenv("RP_METADATASTORE_URL", custom_url)
        assert Config.get_metadatastore_url() == custom_url

    def test_invalid_sqlite_parent_raises(self, tmp_path, monkeypatch):
        """Should raise FileNotFoundError if SQLite parent directory does not exist."""
        monkeypatch.setenv("RP_ASSETS_CACHE", str(tmp_path))
        monkeypatch.setenv(
            "RP_METADATASTORE_URL", "sqlite:///nonexistent_dir/db.sqlite"
        )
        with pytest.raises(FileNotFoundError):
            Config.get_metadatastore_url()


class TestAsyncMetadataStoreURL:
    """Tests for `get_metadatastore_async_url`."""

    def test_env_var_direct(self, tmp_path, monkeypatch):
        """If RP_METADATASTORE_ASYNC_URL is set, it should be returned as-is."""
        monkeypatch.setenv("RP_ASSETS_CACHE", str(tmp_path))
        async_url = "postgresql+asyncpg://user:pass@host/db"
        monkeypatch.setenv("RP_METADATASTORE_ASYNC_URL", async_url)
        assert Config.get_metadatastore_async_url() == async_url

    def test_derived_sqlite(self, tmp_path, monkeypatch):
        """Async URL derived from sqlite:// should become sqlite+aiosqlite:// and strip parameters."""
        cache_path = tmp_path / "cache"
        monkeypatch.setenv("RP_ASSETS_CACHE", str(cache_path))
        monkeypatch.delenv("RP_METADATASTORE_ASYNC_URL", raising=False)
        monkeypatch.setenv(
            "RP_METADATASTORE_URL", f"sqlite:///{cache_path}local.db?timeout=10.0"
        )
        async_url = Config.get_metadatastore_async_url()
        # The async derivation strips query params and replaces scheme
        expected = f"sqlite+aiosqlite:///{cache_path}local.db"
        assert async_url == expected

    def test_derived_postgresql(self, tmp_path, monkeypatch):
        """Async URL derived from postgresql:// should become postgresql+asyncpg://."""
        monkeypatch.setenv("RP_ASSETS_CACHE", str(tmp_path))
        monkeypatch.delenv("RP_METADATASTORE_ASYNC_URL", raising=False)
        monkeypatch.setenv("RP_METADATASTORE_URL", "postgresql://user:pass@host/db")
        async_url = Config.get_metadatastore_async_url()
        assert async_url == "postgresql+asyncpg://user:pass@host/db"

    def test_derived_postgres(self, tmp_path, monkeypatch):
        """Async URL derived from postgres:// should become postgresql+asyncpg://."""
        monkeypatch.setenv("RP_ASSETS_CACHE", str(tmp_path))
        monkeypatch.delenv("RP_METADATASTORE_ASYNC_URL", raising=False)
        monkeypatch.setenv("RP_METADATASTORE_URL", "postgres://user:pass@host/db")
        async_url = Config.get_metadatastore_async_url()
        assert async_url == "postgresql+asyncpg://user:pass@host/db"

    def test_unsupported_scheme_raises(self, tmp_path, monkeypatch):
        """An unsupported database scheme should raise NotSupportedError."""
        monkeypatch.setenv("RP_ASSETS_CACHE", str(tmp_path))
        monkeypatch.delenv("RP_METADATASTORE_ASYNC_URL", raising=False)
        monkeypatch.setenv("RP_METADATASTORE_URL", "mysql://user:pass@host/db")
        with pytest.raises(NotSupportedError) as exc_info:
            Config.get_metadatastore_async_url()
        assert "mysql" in str(exc_info.value)

    def test_s3_cache_no_url_raises(self, assets_cache, monkeypatch):
        """Should raise NotSupportedError if cache is S3 and no async metadatastore URL is set."""
        if not is_s3_path(assets_cache):
            pytest.skip(
                "NotSupportedError is specific to S3 cache without explicit URL"
            )
        monkeypatch.delenv("RP_METADATASTORE_ASYNC_URL", raising=False)
        monkeypatch.delenv("RP_METADATASTORE_URL", raising=False)
        with pytest.raises(NotSupportedError):
            Config.get_metadatastore_async_url()
