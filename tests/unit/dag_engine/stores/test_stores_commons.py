"""Unit tests for retrain_pipelines.dag_engine.stores.commons."""

import cloudpickle
from datetime import date, datetime, timezone

import pytest
from pydantic import BaseModel

import retrain_pipelines.dag_engine.stores.commons as commons
from retrain_pipelines.dag_engine.stores.commons import (
    DISK_REF_KEY,
    metadata_root,
    try_json_serialize,
    is_disk_ref,
    make_disk_ref,
)
from retrain_pipelines.dag_engine.stores.params_store import value_to_storable

_MODULE = "retrain_pipelines.dag_engine.stores.commons"


# ---------------------------------------------------------------------------
# metadata_root
# ---------------------------------------------------------------------------


class TestMetadataRoot:
    def test_returns_cache_metadata_subdir(self, monkeypatch, tmp_path):
        monkeypatch.setenv("RP_ASSETS_CACHE", str(tmp_path))

        assert metadata_root() == str(tmp_path / "metadata")


# ---------------------------------------------------------------------------
# is_disk_ref
# ---------------------------------------------------------------------------


class TestIsDiskRef:
    def test_true_for_disk_ref_dict(self):
        assert is_disk_ref({DISK_REF_KEY: "some/path.pkl"}) is True

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
        assert ref == {DISK_REF_KEY: "some/path.pkl"}


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

        result = commons.load_from_disk(rel_path)

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
        # value_to_storable (params_store) creates the physical file and returns the envelope
        envelope = value_to_storable("tid", "defaults", "p", obj)

        resolved = commons.resolve_storable(envelope)

        assert isinstance(resolved, Custom)
        assert resolved.v == 42

    def test_returns_non_envelope_unchanged(self):
        """When given a plain value or non-disk-ref dict, it returns it as-is."""
        assert commons.resolve_storable(42) == 42
        assert commons.resolve_storable("hello") == "hello"
        assert commons.resolve_storable({"other": "dict"}) == {"other": "dict"}
        assert commons.resolve_storable(None) is None


# ---------------------------------------------------------------------------
# try_json_serialize
# ---------------------------------------------------------------------------


class TestTryJsonSerialize:
    def test_none(self):
        assert try_json_serialize(None) is None

    def test_primitives(self):
        assert try_json_serialize(1) == 1
        assert try_json_serialize(3.14) == 3.14
        assert try_json_serialize("s") == "s"
        assert try_json_serialize(True) is True

    def test_datetime(self):
        dt = datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc)

        assert try_json_serialize(dt) == dt.isoformat()

    def test_date(self):
        d = date(2024, 1, 15)

        assert try_json_serialize(d) == "2024-01-15"

    def test_pydantic_model(self):
        class M(BaseModel):
            x: int = 7

        assert try_json_serialize(M()) == {"x": 7}

    def test_dict_recursive(self):
        assert try_json_serialize({"a": [1, 2]}) == {"a": [1, 2]}

    def test_list(self):
        assert try_json_serialize([1, "two"]) == [1, "two"]

    def test_tuple_and_set(self):
        # tuples and sets are both coerced to list; single-element set avoids
        # ordering ambiguity in the assertion.
        assert try_json_serialize((1, 2)) == [1, 2]
        assert try_json_serialize({99}) == [99]

    def test_unknown_type_raises(self):
        class Custom:
            pass

        with pytest.raises(TypeError):
            try_json_serialize(Custom())
