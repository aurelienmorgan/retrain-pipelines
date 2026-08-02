import os
from typing import Any
import pytest

from retrain_pipelines.dag_engine.stores.contexts_store import (
    _CONTEXT_EXCLUDE_ATTRS,
    context_attr_disk_path,
    _serialize_attr,
    snapshot_context_etags,
    compute_context_diff,
)
from retrain_pipelines.dag_engine.stores.commons import compute_sha


class MockContext:
    """Duck-typed execution context matching DagExecutionContext interface for testing."""

    def __init__(
        self, params: dict[str, Any], attr_refs: dict[str, dict] | None = None
    ):
        self._params = params
        self._attr_refs = attr_refs or {}


@pytest.fixture(autouse=True)
def _setup_cache_env(tmp_path):
    """Isolate disk artifacts by routing RP_ASSETS_CACHE to a temporary directory."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    os.environ["RP_ASSETS_CACHE"] = str(cache_dir)
    yield


def test_context_attr_disk_path():
    path = context_attr_disk_path(10, 20, "model_weights")

    assert path == os.path.join("10", "20", "model_weights.pkl")


class TestSerializeAttr:
    def test_json_serializable_value(self):
        exec_id, task_id, attr_name = 1, 1, "config"
        value = {"lr": 0.01, "epochs": 10}
        current_etag = "etag-1"
        row, ref = _serialize_attr(exec_id, task_id, attr_name, value, current_etag)

        assert ref["eTAG"] == current_etag
        assert "sha" not in ref  # sha is absent in ref for JSON-serializable
        assert ref["disk_ref"] is None
        assert ref["inline"] == value
        assert row["task_id"] == task_id
        assert row["attr_name"] == attr_name
        assert row["eTAG"] == current_etag
        assert row["sha"] is None  # sha is None in row for JSON-serializable
        assert row["disk_ref"] is None
        assert row["inline_val"] == value

    def test_non_json_serializable_value(self, tmp_path):
        exec_id, task_id, attr_name = 5, 10, "lambda_func"

        def value(x: int) -> int:
            return x**2

        current_etag = "etag-2"
        current_sha = compute_sha(value)
        row, ref = _serialize_attr(exec_id, task_id, attr_name, value, current_etag)

        assert ref["eTAG"] == current_etag
        assert ref["sha"] == current_sha
        assert ref["inline"] is None
        assert ref["disk_ref"] == os.path.join("5", "10", "lambda_func.pkl")
        assert row["disk_ref"] == ref["disk_ref"]
        assert row["inline_val"] is None
        assert row["sha"] == current_sha
        assert row["eTAG"] == current_etag

        abs_path = os.path.join(tmp_path, "cache", "metadata", ref["disk_ref"])
        assert os.path.exists(abs_path)


class TestSnapshotContextEtags:
    def test_excludes_and_none(self):
        ctx = MockContext(
            {"keep": "val", "skip": "skip_val", "none_val": None},
            {"keep": {"eTAG": "etag-keep"}, "skip": {"eTAG": "etag-skip"}},
        )
        exclude = frozenset(["skip"])
        etags = snapshot_context_etags(ctx, exclude)

        assert "skip" not in etags
        assert "none_val" not in etags
        assert etags["keep"] == "etag-keep"

    def test_generates_new_etag(self):
        ctx = MockContext({"new": "val"}, {})
        etags = snapshot_context_etags(ctx, frozenset())

        assert "new" in etags
        assert isinstance(etags["new"], str)
        assert len(etags["new"]) > 0


class TestComputeContextDiff:
    def test_new_attr_serializes(self):
        ctx = MockContext({"new": [1, 2, 3]}, {})
        rows = compute_context_diff(1, 1, ctx, {}, frozenset())

        assert len(rows) == 1
        assert rows[0]["attr_name"] == "new"
        assert "new" in ctx._attr_refs
        assert rows[0]["inline_val"] == [1, 2, 3]
        assert rows[0]["eTAG"] is not None
        assert rows[0]["sha"] is None
        assert ctx._attr_refs["new"]["eTAG"] == rows[0]["eTAG"]

    def test_modified_attr_serializes(self):
        _, new_val = "old", "updated"
        old_etag = "etag-old"
        new_etag = "etag-new"
        ctx = MockContext({"mod": new_val}, {"mod": {"eTAG": new_etag}})
        entry_etags = {"mod": old_etag}
        rows = compute_context_diff(1, 1, ctx, entry_etags, frozenset())

        assert len(rows) == 1
        assert rows[0]["eTAG"] == new_etag
        assert rows[0]["sha"] is None  # "updated" is JSON-serializable
        assert rows[0]["inline_val"] == new_val
        assert ctx._attr_refs["mod"]["eTAG"] == new_etag

    def test_unchanged_attr_carries_forward_ref(self):
        val = "stable"
        etag = "etag-stable"
        ref = {"eTAG": etag, "disk_ref": None, "inline": val}
        ctx = MockContext({"keep": val}, {"keep": ref})
        entry_etags = {"keep": etag}
        rows = compute_context_diff(1, 1, ctx, entry_etags, frozenset())

        assert len(rows) == 1
        assert rows[0]["eTAG"] == etag
        assert rows[0]["sha"] is None  # ref.get("sha") evaluates to None
        assert rows[0]["disk_ref"] is None
        assert rows[0]["inline_val"] == val

    def test_unchanged_attr_missing_ref_serializes(self):
        val = "edge_case"
        etag = "etag-edge"
        ctx = MockContext({"edge": val}, {})
        entry_etags = {"edge": etag}
        rows = compute_context_diff(1, 1, ctx, entry_etags, frozenset())

        assert len(rows) == 1
        assert rows[0]["inline_val"] == val
        assert rows[0]["eTAG"] is not None
        assert "edge" in ctx._attr_refs
        assert ctx._attr_refs["edge"]["eTAG"] == rows[0]["eTAG"]

    def test_filters_excluded_and_none(self):
        ctx = MockContext({"username": "admin", "val": None, "keep": True})
        rows = compute_context_diff(1, 1, ctx, {}, _CONTEXT_EXCLUDE_ATTRS)

        assert len(rows) == 1
        assert rows[0]["attr_name"] == "keep"
        assert rows[0]["inline_val"] is True
