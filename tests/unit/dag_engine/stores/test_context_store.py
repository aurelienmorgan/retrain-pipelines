import os
from typing import Any
import pytest

from retrain_pipelines.dag_engine.stores.context_store import (
    _CONTEXT_EXCLUDE_ATTRS,
    context_attr_disk_path,
    _serialize_attr,
    snapshot_context_shas,
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
        current_sha = compute_sha(value)
        row, ref = _serialize_attr(exec_id, task_id, attr_name, value, current_sha)

        assert ref["sha"] == current_sha
        assert ref["disk_ref"] is None
        assert ref["inline"] == value
        assert row["task_id"] == task_id
        assert row["attr_name"] == attr_name
        assert row["sha"] == current_sha
        assert row["disk_ref"] is None
        assert row["inline_val"] == value

    def test_non_json_serializable_value(self, tmp_path):
        exec_id, task_id, attr_name = 5, 10, "lambda_func"

        def value(x: int) -> int:
            return x**2

        current_sha = compute_sha(value)
        row, ref = _serialize_attr(exec_id, task_id, attr_name, value, current_sha)

        assert ref["sha"] == current_sha
        assert ref["inline"] is None
        assert ref["disk_ref"] == os.path.join("5", "10", "lambda_func.pkl")
        assert row["disk_ref"] == ref["disk_ref"]
        assert row["inline_val"] is None

        abs_path = os.path.join(tmp_path, "cache", "metadata", ref["disk_ref"])
        assert os.path.exists(abs_path)


class TestSnapshotContextShas:
    def test_computes_shas_for_new_attrs(self):
        ctx = MockContext({"a": 1, "b": "test"})
        res = snapshot_context_shas(ctx, frozenset())

        assert res["a"] == compute_sha(1)
        assert res["b"] == compute_sha("test")

    def test_skips_excluded_and_none(self):
        ctx = MockContext(
            {"exec_id": 1, "pipeline_name": "p", "username": "u", "c": None, "d": 42}
        )
        res = snapshot_context_shas(ctx, _CONTEXT_EXCLUDE_ATTRS)

        assert "exec_id" not in res
        assert "pipeline_name" not in res
        assert "c" not in res
        assert "d" in res

    def test_reuses_tracked_sha(self):
        tracked_sha = "precomputed_stable_sha"
        ctx = MockContext({"x": 99}, {"x": {"sha": tracked_sha}})
        res = snapshot_context_shas(ctx, frozenset())

        assert res["x"] == tracked_sha


class TestComputeContextDiff:
    def test_new_attr_serializes(self):
        ctx = MockContext({"new": [1, 2, 3]}, {})
        rows = compute_context_diff(1, 1, ctx, {}, frozenset())

        assert len(rows) == 1
        assert rows[0]["attr_name"] == "new"
        assert "new" in ctx._attr_refs
        assert rows[0]["inline_val"] == [1, 2, 3]

    def test_modified_attr_serializes(self):
        old_val, new_val = "old", "updated"
        ctx = MockContext({"mod": new_val}, {"mod": {"sha": compute_sha(old_val)}})
        entry_shas = {"mod": compute_sha(old_val)}
        rows = compute_context_diff(1, 1, ctx, entry_shas, frozenset())

        assert len(rows) == 1
        assert rows[0]["sha"] == compute_sha(new_val)
        assert ctx._attr_refs["mod"]["sha"] == compute_sha(new_val)

    def test_unchanged_attr_carries_forward_ref(self):
        val = "stable"
        ref = {"sha": compute_sha(val), "disk_ref": None, "inline": val}
        ctx = MockContext({"keep": val}, {"keep": ref})
        entry_shas = {"keep": compute_sha(val)}
        rows = compute_context_diff(1, 1, ctx, entry_shas, frozenset())

        assert len(rows) == 1
        assert rows[0]["sha"] == compute_sha(val)
        assert rows[0]["disk_ref"] is None
        assert rows[0]["inline_val"] == val

    def test_unchanged_attr_missing_ref_serializes(self):
        val = "edge_case"
        ctx = MockContext({"edge": val}, {})
        entry_shas = {"edge": compute_sha(val)}
        rows = compute_context_diff(1, 1, ctx, entry_shas, frozenset())

        assert len(rows) == 1
        assert rows[0]["inline_val"] == val
        assert "edge" in ctx._attr_refs

    def test_filters_excluded_and_none(self):
        ctx = MockContext({"username": "admin", "val": None, "keep": True})
        rows = compute_context_diff(1, 1, ctx, {}, _CONTEXT_EXCLUDE_ATTRS)

        assert len(rows) == 1
        assert rows[0]["attr_name"] == "keep"
