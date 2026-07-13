"""
Unit tests for retrain_pipelines.dag_engine.stores.params_store.
"""

import os
import re
import platform
from unittest.mock import MagicMock


from retrain_pipelines.dag_engine.stores import params_store
from retrain_pipelines.dag_engine.stores.commons import DISK_REF_KEY, metadata_root


class TestTempDirId:
    """Tests for temp_dir_id generation."""

    def test_format_and_uniqueness(self):
        """Verify the generated ID matches the expected timestamp+hex format."""
        tid = params_store.temp_dir_id()

        # Format: 17 digits (YYYYMMDDHHMMSSmmm) + '_' + 6 hex chars
        assert re.match(r"^\d{17}_[a-f0-9]{6}$", tid)


class TestParamsSubdirPath:
    """Tests for _params_subdir_path resolution."""

    def test_absolute_path_construction(self, tmp_path, monkeypatch):
        """Ensure it correctly joins metadata_root with dir_id and subdir."""
        monkeypatch.setenv("RP_ASSETS_CACHE", str(tmp_path))
        path = params_store._params_subdir_path("123", "defaults")
        expected = os.path.join(metadata_root(), "123", "params", "defaults")

        assert path == expected


class TestParamDiskPath:
    """Tests for param_disk_path resolution."""

    def test_relative_path_construction(self):
        """Ensure it returns a relative path suitable for DB disk_ref storage."""
        path = params_store.param_disk_path("123", "defaults", "my_param")

        assert path == os.path.join("123", "params", "defaults", "my_param.pkl")


class TestLinkParamsDefaultsToExec:
    """Tests for linking temp param defaults to exec_id defaults."""

    def _force_non_wsl(self, monkeypatch):
        """Helper to force is_wsl() to return False by hiding WSL markers.

        This avoids triggering the wslpath subprocess call in is_windows_path,
        allowing us to test the platform.system() fallback branch deterministically
        without mocking any retrain_pipelines imports.
        """
        original_exists = os.path.exists

        def fake_exists(p):
            if p in ("/proc/version", "/etc/os-release"):
                return False

            return original_exists(p)

        monkeypatch.setattr(os.path, "exists", fake_exists)

        # Clear lru_cache to ensure is_windows_path re-evaluates with mocked env
        if hasattr(params_store.is_windows_path, "cache_clear"):
            params_store.is_windows_path.cache_clear()

    def test_skips_if_src_not_exists(self, tmp_path, monkeypatch):
        """If the source defaults dir doesn't exist, no linking should occur."""
        monkeypatch.setenv("RP_ASSETS_CACHE", str(tmp_path))
        self._force_non_wsl(monkeypatch)

        params_store.link_params_defaults_to_exec("temp1", 1)
        dst = params_store._params_subdir_path(1, "defaults")

        assert not os.path.exists(dst)

    def test_posix_symlink(self, tmp_path, monkeypatch):
        """On POSIX, should use os.symlink for linking directories."""
        monkeypatch.setenv("RP_ASSETS_CACHE", str(tmp_path))
        self._force_non_wsl(monkeypatch)
        # Force platform.system to return Linux to trigger POSIX branch
        monkeypatch.setattr(platform, "system", lambda: "Linux")

        temp_id = "temp1"
        exec_id = 1
        src = params_store._params_subdir_path(temp_id, "defaults")
        os.makedirs(src)

        params_store.link_params_defaults_to_exec(temp_id, exec_id)

        dst = params_store._params_subdir_path(exec_id, "defaults")

        assert os.path.islink(dst)
        assert os.readlink(dst) == src

    def test_windows_junction(self, monkeypatch):
        """On Windows/WSL DrvFs, should use cmd.exe mklink /J.

        We use a /mnt/ path format so wsl_to_windows_path succeeds naturally
        without mocking retrain_pipelines imports.
        """
        # Must use /mnt/ path so wsl_to_windows_path doesn't raise ValueError
        monkeypatch.setenv("RP_ASSETS_CACHE", "/mnt/c/fake_cache")

        # Compute before patching os.path.exists so metadata_root() is stable
        _dst = params_store._params_subdir_path(1, "defaults")

        def fake_exists(p):
            if p in ("/proc/version", "/etc/os-release", _dst):
                # _dst must appear absent; otherwise the early-return guard
                # (os.path.exists(dst) or os.path.islink(dst)) fires and
                # link_params_defaults_to_exec returns before reaching
                # is_windows_path() / subprocess.run().
                return False

            return True  # src and all other paths look present

        monkeypatch.setattr(os.path, "exists", fake_exists)
        # os.path.islink is the second clause of the early-return guard;
        # patch it so it never fires on any test machine.
        monkeypatch.setattr(os.path, "islink", lambda p: False)

        # Force platform.system to return Windows to trigger Windows branch
        monkeypatch.setattr(platform, "system", lambda: "Windows")

        if hasattr(params_store.is_windows_path, "cache_clear"):
            params_store.is_windows_path.cache_clear()

        mock_cmd = MagicMock()

        def fake_run(cmd, *args, **kwargs):
            if cmd[0] == "cmd.exe":
                mock_cmd(cmd)
                return MagicMock()
            raise RuntimeError(f"Unexpected subprocess call: {cmd}")

        monkeypatch.setattr("subprocess.run", fake_run)
        # Mock os.makedirs since we don't want to actually create dirs in /mnt/c
        monkeypatch.setattr(os, "makedirs", lambda p, exist_ok=False: None)

        params_store.link_params_defaults_to_exec("temp1", 1)

        mock_cmd.assert_called_once()
        args = mock_cmd.call_args[0][0]

        assert args[:4] == ["cmd.exe", "/c", "mklink", "/J"]
        # Verify paths were converted correctly by wsl_to_windows_path
        assert args[4] == "C:\\fake_cache\\metadata\\1\\params\\defaults"
        assert args[5] == "C:\\fake_cache\\metadata\\temp1\\params\\defaults"

    def test_windows_junction_skips_if_exists(self, tmp_path, monkeypatch):
        """If dst already exists, the function should return early without linking."""
        monkeypatch.setenv("RP_ASSETS_CACHE", str(tmp_path))
        self._force_non_wsl(monkeypatch)
        monkeypatch.setattr(platform, "system", lambda: "Linux")

        temp_id = "temp1"
        exec_id = 1

        # Create src so the outer if passes
        src = params_store._params_subdir_path(temp_id, "defaults")
        os.makedirs(src)

        # Create dst so the guard triggers
        dst = params_store._params_subdir_path(exec_id, "defaults")
        os.makedirs(dst)

        # Spy on symlink creation
        mock_symlink = MagicMock()
        monkeypatch.setattr(os, "symlink", mock_symlink)

        params_store.link_params_defaults_to_exec(temp_id, exec_id)

        # Ensure no link was created
        mock_symlink.assert_not_called()


class TestValueToStorable:
    """Tests for value_to_storable serialization logic."""

    def test_json_safe_value(self, tmp_path, monkeypatch):
        """Natively JSON-serializable values should be returned as-is."""
        monkeypatch.setenv("RP_ASSETS_CACHE", str(tmp_path))
        obj = {"a": 1, "b": [2, 3]}
        res = params_store.value_to_storable("1", "defaults", "p1", obj)

        assert res == obj

    def test_cloudpickle_fallback(self, tmp_path, monkeypatch):
        """Non-JSON-serializable objects should be cloudpickled to disk."""
        monkeypatch.setenv("RP_ASSETS_CACHE", str(tmp_path))
        obj = object()
        res = params_store.value_to_storable("1", "defaults", "p1", obj)

        assert "__sha__" in res
        assert DISK_REF_KEY in res

        rel_path = res[DISK_REF_KEY]
        assert rel_path == os.path.join("1", "params", "defaults", "p1.pkl")

        abs_path = os.path.join(metadata_root(), rel_path)
        assert os.path.exists(abs_path)


class TestAttrRefFromParamStorable:
    """Tests for attr_ref_from_param_storable dict construction."""

    def test_disk_ref_storable(self):
        """Disk-ref sentinel dicts should extract sha and disk_ref."""
        storable = {"__sha__": "abc", DISK_REF_KEY: "some/path.pkl"}
        res = params_store.attr_ref_from_param_storable(storable, None)

        assert res == {"sha": "abc", "disk_ref": "some/path.pkl", "inline": None}

    def test_inline_storable(self, tmp_path, monkeypatch):
        """Inline JSON-safe values should compute SHA on the resolved object."""
        monkeypatch.setenv("RP_ASSETS_CACHE", str(tmp_path))
        storable = 42
        res = params_store.attr_ref_from_param_storable(storable, 42)

        assert res["disk_ref"] is None
        assert res["inline"] == 42
        assert "sha" in res
