"""
Unit tests for retrain_pipelines.dag_engine.stores.params_store.
"""

import os
import re
import platform
import boto3
import cloudpickle
import pytest
from unittest.mock import MagicMock

from retrain_pipelines.dag_engine.stores import params_store
from retrain_pipelines.dag_engine.stores.commons import DISK_REF_KEY, metadata_root
from retrain_pipelines.utils.file_utils import read_binary_file
from retrain_pipelines.utils.s3_utils import (
    is_s3_path,
    parse_s3_uri,
    s3_prefix_has_objects,
)


class TestTempDirId:
    """Tests for temp_dir_id generation."""

    def test_format_and_uniqueness(self):
        """Verify the generated ID matches the expected timestamp+hex format."""
        tid = params_store.temp_dir_id()

        # Format: 17 digits (YYYYMMDDHHMMSSmmm) + '_' + 6 hex chars
        assert re.match(r"^\d{17}_[a-f0-9]{6}$", tid)


class TestParamsSubdirPath:
    """Tests for _params_subdir_path resolution."""

    def test_absolute_path_construction(self, assets_cache):
        """Ensure it correctly joins metadata_root with dir_id and subdir."""
        path = params_store._params_subdir_path("123", "defaults")
        if is_s3_path(metadata_root()):
            # S3 paths always use '/' and do not have trailing slashes in build_path
            expected = metadata_root().rstrip("/") + "/123/params/defaults"
        else:
            expected = os.path.join(metadata_root(), "123", "params", "defaults")

        assert path == expected


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

    def test_skips_if_src_not_exists(self, assets_cache, monkeypatch):
        """If the source defaults dir doesn't exist, no linking should occur."""
        self._force_non_wsl(monkeypatch)

        params_store.link_params_defaults_to_exec("temp1", 1)
        dst = params_store._params_subdir_path(1, "defaults")

        if is_s3_path(metadata_root()):
            # For S3, verify no marker object was created under the dst prefix.
            bucket, prefix = parse_s3_uri(dst)
            assert not s3_prefix_has_objects(bucket, prefix)
        else:
            assert not os.path.exists(dst)

    def test_link_creation(self, assets_cache, monkeypatch):
        """Covers local symlink and S3 marker object creation branches."""
        self._force_non_wsl(monkeypatch)
        # Force platform.system to return Linux to trigger POSIX branch on local.
        # Harmless for S3 as the S3 branch returns early.
        monkeypatch.setattr(platform, "system", lambda: "Linux")

        temp_id = "temp1"
        exec_id = 1

        if is_s3_path(metadata_root()):
            # S3 marker creation logic
            s3 = boto3.client("s3")
            bucket, meta_prefix = parse_s3_uri(metadata_root())
            src_prefix = f"{meta_prefix}{temp_id}/params/defaults/"
            # Put a dummy object in src_prefix so the real s3_prefix_has_objects()
            # returns True and the marker object creation branch is exercised.
            s3.put_object(Bucket=bucket, Key=f"{src_prefix}dummy.pkl", Body=b"")

            params_store.link_params_defaults_to_exec(temp_id, exec_id)

            # Verify the zero-byte marker object was created at the expected key.
            readable_src_prefix = src_prefix.replace("/", "／")
            expected_key = (
                f"{meta_prefix}{exec_id}/params/defaults/{readable_src_prefix}"
            )

            try:
                resp = s3.head_object(Bucket=bucket, Key=expected_key)
                assert resp["ContentLength"] == 0
            finally:
                s3.delete_object(Bucket=bucket, Key=expected_key)
                s3.delete_object(Bucket=bucket, Key=f"{src_prefix}dummy.pkl")
        else:
            # Local symlink creation logic
            src = params_store._params_subdir_path(temp_id, "defaults")
            os.makedirs(src)

            params_store.link_params_defaults_to_exec(temp_id, exec_id)

            dst = params_store._params_subdir_path(exec_id, "defaults")
            assert os.path.islink(dst)
            assert os.readlink(dst) == src

    def test_windows_junction(self, monkeypatch):
        """On Windows/WSL DrvFs, should use cmd.exe mklink /J.

        We mock the WSL path detection helpers to simulate a DrvFs mount
        deterministically without relying on real OS state.
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

        # Simulate WSL DrvFs mount deterministically
        monkeypatch.setattr(params_store, "is_windows_path", lambda p: True)
        monkeypatch.setattr(params_store, "is_wsl_mount_path", lambda p: True)
        monkeypatch.setattr(
            params_store,
            "wsl_to_windows_path",
            lambda p: p.replace("/mnt/c", "C:\\").replace("/", "\\"),
        )

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
        # Verify paths were converted correctly by mocked wsl_to_windows_path
        assert args[4] == "C:\\\\fake_cache\\metadata\\1\\params\\defaults"
        assert args[5] == "C:\\\\fake_cache\\metadata\\temp1\\params\\defaults"

    def test_windows_junction_skips_if_exists(self, assets_cache, monkeypatch):
        """If dst already exists, the function should return early without linking."""
        if is_s3_path(metadata_root()):
            pytest.skip("Windows junctions are only applicable to local filesystems")

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

    def test_json_safe_value(self, assets_cache):
        """Natively JSON-serializable values should be returned as-is."""
        obj = {"a": 1, "b": [2, 3]}
        res = params_store.value_to_storable("1", "defaults", "p1", obj)

        assert res == obj

    def test_cloudpickle_fallback(self, assets_cache):
        """Non-JSON-serializable objects should be cloudpickled to disk."""
        obj = object()
        res = params_store.value_to_storable("1", "defaults", "p1", obj)

        assert "__sha__" in res
        assert DISK_REF_KEY in res

        rel_path = res[DISK_REF_KEY]
        if is_s3_path(metadata_root()):
            assert rel_path == "1/params/defaults/p1.pkl"
        else:
            assert rel_path == os.path.join("1", "params", "defaults", "p1.pkl")

        # Verify the artifact was written to the configured backend.
        data = read_binary_file(metadata_root(), [rel_path])
        assert cloudpickle.loads(data) is not None


class TestAttrRefFromParamStorable:
    """Tests for attr_ref_from_param_storable dict construction."""

    def test_disk_ref_storable(self):
        """Disk-ref sentinel dicts should extract sha and disk_ref."""
        storable = {
            "__eTAG__": "etag1",
            "__sha__": "abc",
            DISK_REF_KEY: "some/path.pkl",
        }
        res = params_store.attr_ref_from_param_storable(storable)

        assert res == {
            "eTAG": "etag1",
            "sha": "abc",
            "disk_ref": "some/path.pkl",
            "inline": None,
        }

    def test_inline_storable(self, assets_cache):
        """Inline JSON-safe values should compute SHA on the resolved object."""
        storable = 42
        res = params_store.attr_ref_from_param_storable(storable)

        assert res["disk_ref"] is None
        assert res["inline"] == 42
        assert res["sha"] is None
        assert "eTAG" in res
