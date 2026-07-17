"""
Unit tests for the `config` module.
"""

import os
import pytest

from retrain_pipelines.dag_engine.config import Config, NotSupportedError


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

    def test_env_var(self, tmp_path, monkeypatch):
        """Assets cache root should be taken from RP_ASSETS_CACHE env var."""
        custom_path = tmp_path / "custom_cache"
        monkeypatch.setenv("RP_ASSETS_CACHE", str(custom_path))
        root = Config.get_assets_cache_root()
        expected = str(custom_path) + os.sep
        assert root == expected
        assert custom_path.exists()


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

    def test_env_var(self, tmp_path, monkeypatch):
        """Artifacts store root should be taken from RP_ARTIFACTS_STORE env var."""
        custom_path = tmp_path / "custom_artifacts"
        monkeypatch.setenv("RP_ARTIFACTS_STORE", str(custom_path))
        root = Config.get_artifacts_store_root()
        expected = str(custom_path) + os.sep
        assert root == expected
        assert custom_path.exists()


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

    def test_env_var(self, tmp_path, monkeypatch):
        """Web server logs root should be taken from RP_WEB_SERVER_LOGS env var."""
        custom_path = tmp_path / "custom_logs"
        monkeypatch.setenv("RP_WEB_SERVER_LOGS", str(custom_path))
        root = Config.get_web_server_logs_root()
        expected = str(custom_path) + os.sep
        assert root == expected
        assert custom_path.exists()


class TestPorts:
    """Tests for port getters: `get_web_server_port` and `get_grpc_server_port`."""

    def test_web_server_port_default(self, monkeypatch):
        """Default web server port should be 5001."""
        monkeypatch.delenv("RP_WEB_SERVER_PORT", raising=False)
        assert Config.get_web_server_port() == 5001

    def test_web_server_port_env_var(self, monkeypatch):
        """Web server port should be taken from RP_WEB_SERVER_PORT env var."""
        monkeypatch.setenv("RP_WEB_SERVER_PORT", "8080")
        assert Config.get_web_server_port() == 8080

    def test_grpc_server_port_default(self, monkeypatch):
        """Default gRPC server port should be 50051."""
        monkeypatch.delenv("RP_GRPC_SERVER_PORT", raising=False)
        assert Config.get_grpc_server_port() == 50051

    def test_grpc_server_port_env_var(self, monkeypatch):
        """gRPC server port should be taken from RP_GRPC_SERVER_PORT env var."""
        monkeypatch.setenv("RP_GRPC_SERVER_PORT", "50052")
        assert Config.get_grpc_server_port() == 50052


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

    def test_default(self, tmp_path, monkeypatch):
        """Default metadata store URL should be sqlite:// with timeout and assets cache."""
        cache_path = tmp_path / "cache"
        monkeypatch.setenv("RP_ASSETS_CACHE", str(cache_path))
        monkeypatch.delenv("RP_METADATASTORE_URL", raising=False)
        url = Config.get_metadatastore_url()
        # get_assets_cache_root returns path with trailing separator, so we include it
        expected = f"sqlite:///{cache_path}{os.sep}local_metadatastore.db?timeout=10.0"
        assert url == expected

    def test_env_var(self, monkeypatch):
        """Metadata store URL should be taken from RP_METADATASTORE_URL env var."""
        custom_url = "postgresql://user:pass@host/db"
        monkeypatch.setenv("RP_METADATASTORE_URL", custom_url)
        assert Config.get_metadatastore_url() == custom_url


class TestAsyncMetadataStoreURL:
    """Tests for `get_metadatastore_async_url`."""

    def test_env_var_direct(self, monkeypatch):
        """If RP_METADATASTORE_ASYNC_URL is set, it should be returned as-is."""
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

    def test_derived_postgresql(self, monkeypatch):
        """Async URL derived from postgresql:// should become postgresql+asyncpg://."""
        monkeypatch.delenv("RP_METADATASTORE_ASYNC_URL", raising=False)
        monkeypatch.setenv("RP_METADATASTORE_URL", "postgresql://user:pass@host/db")
        async_url = Config.get_metadatastore_async_url()
        assert async_url == "postgresql+asyncpg://user:pass@host/db"

    def test_derived_postgres(self, monkeypatch):
        """Async URL derived from postgres:// should become postgresql+asyncpg://."""
        monkeypatch.delenv("RP_METADATASTORE_ASYNC_URL", raising=False)
        monkeypatch.setenv("RP_METADATASTORE_URL", "postgres://user:pass@host/db")
        async_url = Config.get_metadatastore_async_url()
        assert async_url == "postgresql+asyncpg://user:pass@host/db"

    def test_unsupported_scheme_raises(self, monkeypatch):
        """An unsupported database scheme should raise NotSupportedError."""
        monkeypatch.delenv("RP_METADATASTORE_ASYNC_URL", raising=False)
        monkeypatch.setenv("RP_METADATASTORE_URL", "mysql://user:pass@host/db")
        with pytest.raises(NotSupportedError) as exc_info:
            Config.get_metadatastore_async_url()
        assert "mysql" in str(exc_info.value)
