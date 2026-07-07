"""Unit tests for retrain_pipelines.dag_engine.context_store."""

import cloudpickle
import re
from datetime import date, datetime, timezone
from unittest.mock import patch

import pytest
from pydantic import BaseModel

import retrain_pipelines.dag_engine.context_store as context_store
from retrain_pipelines.dag_engine.context_store import (
    _DISK_REF_KEY,
    _metadata_root,
    _params_subdir_path,
    _try_json_serialize,
    is_disk_ref,
    make_disk_ref,
    param_disk_path,
    temp_dir_id,
    value_to_storable,
)

_MODULE = "retrain_pipelines.dag_engine.context_store"


# ---------------------------------------------------------------------------
# _metadata_root
# ---------------------------------------------------------------------------


class TestMetadataRoot:
    def test_returns_cache_metadata_subdir(self, monkeypatch, tmp_path):
        monkeypatch.setenv("RP_ASSETS_CACHE", str(tmp_path))
        assert _metadata_root() == str(tmp_path / "metadata")


# ---------------------------------------------------------------------------
# temp_dir_id
# ---------------------------------------------------------------------------


class TestTempDirId:
    def test_format(self):
        tid = temp_dir_id()
        assert re.fullmatch(r"\d{17}_[0-9a-f]{6}", tid), f"Unexpected format: {tid}"

    def test_uniqueness(self):
        assert temp_dir_id() != temp_dir_id()


# ---------------------------------------------------------------------------
# _params_subdir_path
# ---------------------------------------------------------------------------


class TestParamsDirPath:
    def test_absolute_path_structure(self, monkeypatch, tmp_path):
        monkeypatch.setenv("RP_ASSETS_CACHE", str(tmp_path))
        result = _params_subdir_path("myid", "defaults")
        assert result == str(tmp_path / "metadata" / "myid" / "params" / "defaults")

    def test_int_dir_id(self, monkeypatch, tmp_path):
        monkeypatch.setenv("RP_ASSETS_CACHE", str(tmp_path))
        result = _params_subdir_path(42, "overrides")
        assert result.endswith("42/params/overrides")


# ---------------------------------------------------------------------------
# param_disk_path
# ---------------------------------------------------------------------------


class TestParamDiskPath:
    def test_relative_path_structure(self):
        result = param_disk_path("myid", "defaults", "my_param")
        assert result == "myid/params/defaults/my_param.pkl"

    def test_int_dir_id(self):
        result = param_disk_path(99, "overrides", "p")
        assert result == "99/params/overrides/p.pkl"


# ---------------------------------------------------------------------------
# link_params_defaults_to_exec
# ---------------------------------------------------------------------------


class TestLinkParamsDefaultsToExec:
    def test_no_link_when_src_absent(self, monkeypatch, tmp_path):
        """When the temp defaults dir doesn't exist, no linking occurs.

        The guard ``if os.path.exists(src)`` is the early-exit branch:
        when no param default required disk serialization, the temp dir
        was never created, so no link is needed and we must not attempt one.
        """
        monkeypatch.setenv("RP_ASSETS_CACHE", str(tmp_path))
        tid = temp_dir_id()
        # src directory was never created => os.path.exists returns False for real
        with patch(f"{_MODULE}.os.symlink") as mock_sym:
            context_store.link_params_defaults_to_exec(tid, 1)
        mock_sym.assert_not_called()

    def test_posix_symlink(self, monkeypatch, tmp_path):
        """When src exists and path is POSIX, os.symlink is called.

        ``os.path.exists`` is mocked to return True so we enter the linking
        block without having to create the actual temp directory on disk.
        ``is_windows_path`` is patched at its bound name in the module
        (top-level import) — patching the source module would have no effect.
        """
        monkeypatch.setenv("RP_ASSETS_CACHE", str(tmp_path))
        tid = temp_dir_id()
        src = _params_subdir_path(tid, "defaults")

        with (
            patch(f"{_MODULE}.os.path.exists", return_value=True),
            patch(f"{_MODULE}.os.makedirs"),
            patch(f"{_MODULE}.is_windows_path", return_value=False),
            patch(f"{_MODULE}.os.symlink") as mock_sym,
        ):
            context_store.link_params_defaults_to_exec(tid, 42)

        dst = _params_subdir_path(42, "defaults")
        mock_sym.assert_called_once_with(src, dst)

    def test_windows_junction(self, monkeypatch, tmp_path):
        """When src exists and path is Windows, mklink /J junction is used.

        ``wsl_to_windows_path`` is patched with a simple prefix transform so
        we can assert the correct Windows-style paths are forwarded to
        ``cmd.exe`` without needing a real WSL environment.
        ``subprocess.run`` is patched to prevent any real shell invocation.
        """
        monkeypatch.setenv("RP_ASSETS_CACHE", str(tmp_path))
        tid = temp_dir_id()

        with (
            patch(f"{_MODULE}.os.path.exists", return_value=True),
            patch(f"{_MODULE}.os.makedirs"),
            patch(f"{_MODULE}.is_windows_path", return_value=True),
            patch(f"{_MODULE}.wsl_to_windows_path", side_effect=lambda p: f"W:{p}"),
            patch(f"{_MODULE}.subprocess.run") as mock_run,
        ):
            context_store.link_params_defaults_to_exec(tid, 42)

        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args[:4] == ["cmd.exe", "/c", "mklink", "/J"]


# ---------------------------------------------------------------------------
# is_disk_ref
# ---------------------------------------------------------------------------


class TestIsDiskRef:
    def test_true_for_disk_ref_dict(self):
        assert is_disk_ref({_DISK_REF_KEY: "some/path.pkl"}) is True

    def test_false_for_plain_dict(self):
        assert is_disk_ref({"other_key": "val"}) is False

    def test_false_for_non_dict(self):
        assert is_disk_ref("string") is False
        assert is_disk_ref(42) is False
        assert is_disk_ref(None) is False


# ---------------------------------------------------------------------------
# make_disk_ref
# ---------------------------------------------------------------------------


class TestMakeDiskRef:
    def test_returns_sentinel_dict(self):
        ref = make_disk_ref("some/path.pkl")
        assert ref == {_DISK_REF_KEY: "some/path.pkl"}


# ---------------------------------------------------------------------------
# load_from_disk
# ---------------------------------------------------------------------------


class TestLoadFromDisk:
    def test_deserializes_cloudpickle_artifact(self, monkeypatch, tmp_path):
        """Verify that load_from_disk correctly reads and unpickles a file."""
        monkeypatch.setenv("RP_ASSETS_CACHE", str(tmp_path))
        rel_path = "myid/params/defaults/my_param.pkl"
        abs_path = tmp_path / "metadata" / rel_path
        abs_path.parent.mkdir(parents=True, exist_ok=True)

        class Custom:
            def __init__(self, val=42):
                self.val = val

            def __eq__(self, other):
                return isinstance(other, Custom) and self.val == other.val

        with open(abs_path, "wb") as fh:
            cloudpickle.dump({"complex": Custom(99)}, fh)

        result = context_store.load_from_disk(rel_path)
        assert result == {"complex": Custom(99)}


# ---------------------------------------------------------------------------
# resolve_storable
# ---------------------------------------------------------------------------


class TestResolveStorable:
    def test_resolves_disk_ref(self, monkeypatch, tmp_path):
        """When given a disk_ref envelope, it loads and returns the original object."""
        monkeypatch.setenv("RP_ASSETS_CACHE", str(tmp_path))

        class Custom:
            def __init__(self, v):
                self.v = v

        obj = Custom(42)
        # value_to_storable creates the physical file and returns the envelope dict
        envelope = value_to_storable("tid", "defaults", "p", obj)

        resolved = context_store.resolve_storable(envelope)
        assert isinstance(resolved, Custom)
        assert resolved.v == 42

    def test_returns_non_envelope_unchanged(self):
        """When given a plain value or non-disk-ref dict, it returns it as-is."""
        assert context_store.resolve_storable(42) == 42
        assert context_store.resolve_storable("hello") == "hello"
        assert context_store.resolve_storable({"other": "dict"}) == {"other": "dict"}
        assert context_store.resolve_storable(None) is None


# ---------------------------------------------------------------------------
# _try_json_serialize
# ---------------------------------------------------------------------------


class TestTryJsonSerialize:
    def test_none(self):
        assert _try_json_serialize(None) is None

    def test_primitives(self):
        assert _try_json_serialize(1) == 1
        assert _try_json_serialize(3.14) == 3.14
        assert _try_json_serialize("s") == "s"
        assert _try_json_serialize(True) is True

    def test_datetime(self):
        dt = datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc)
        assert _try_json_serialize(dt) == dt.isoformat()

    def test_date(self):
        d = date(2024, 1, 15)
        assert _try_json_serialize(d) == "2024-01-15"

    def test_pydantic_model(self):
        class M(BaseModel):
            x: int = 7

        assert _try_json_serialize(M()) == {"x": 7}

    def test_dict_recursive(self):
        assert _try_json_serialize({"a": [1, 2]}) == {"a": [1, 2]}

    def test_list(self):
        assert _try_json_serialize([1, "two"]) == [1, "two"]

    def test_tuple_and_set(self):
        # tuples and sets are both coerced to list; single-element set avoids
        # ordering ambiguity in the assertion.
        assert _try_json_serialize((1, 2)) == [1, 2]
        assert _try_json_serialize({99}) == [99]

    def test_unknown_type_raises(self):
        class Custom:
            pass

        with pytest.raises(TypeError):
            _try_json_serialize(Custom())


# ---------------------------------------------------------------------------
# value_to_storable
# ---------------------------------------------------------------------------


class TestValueToStorable:
    def test_json_safe_returned_inline(self):
        assert value_to_storable("tid", "defaults", "p", 42) == 42
        assert value_to_storable("tid", "defaults", "p", "hello") == "hello"
        assert value_to_storable("tid", "defaults", "p", None) is None

    def test_non_serializable_written_to_disk(self, monkeypatch, tmp_path):
        monkeypatch.setenv("RP_ASSETS_CACHE", str(tmp_path))

        class Custom:
            pass

        obj = Custom()
        result = value_to_storable("tid", "defaults", "p", obj)

        assert _DISK_REF_KEY in result
        assert "__sha__" in result
        assert (tmp_path / "metadata" / result[_DISK_REF_KEY]).exists()

    def test_sha_is_consistent(self, monkeypatch, tmp_path):
        """Equal Python objects produce the same SHA regardless of disk artifact."""
        monkeypatch.setenv("RP_ASSETS_CACHE", str(tmp_path))

        # Custom is not JSON-serializable => value_to_storable writes to disk
        # and returns a SHA envelope.  Two calls on instances with identical
        # state must yield the same __sha__ since SHA is computed on the
        # Python object (cloudpickle.dumps(obj)), not on the disk artifact.
        class Custom:
            def __init__(self, v):
                self.v = v

        r1 = value_to_storable("tid1", "defaults", "p1", Custom(42))
        r2 = value_to_storable("tid2", "defaults", "p2", Custom(42))
        assert r1["__sha__"] == r2["__sha__"]

    def test_disk_ref_path_is_relative(self, monkeypatch, tmp_path):
        """disk_ref path stored in DB must be relative to _metadata_root().

        Storing absolute paths would couple the DB to a specific machine's
        filesystem layout.  The path must therefore not begin with the
        RP_ASSETS_CACHE root.
        """
        monkeypatch.setenv("RP_ASSETS_CACHE", str(tmp_path))

        class Custom:
            pass

        result = value_to_storable("tid", "defaults", "p", Custom())
        assert not result[_DISK_REF_KEY].startswith(str(tmp_path))
