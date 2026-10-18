"""Unit tests for retrain_pipelines.dag_engine.sdk.stores

Exercises AttrsDiff, ExecutionParams, TaskExitContext, and TaskExitPayload.
Achieves 100% line coverage of stores.py by testing all classes and branches.

Disk artifacts are written using the project's file_utils to both local
filesystem and S3 (MinIO) to verify load_from_disk() across storage
backends, following the exact same local vs s3 mechanism used across the
project's other test suites.
"""

import os
import pickle

import pytest

from retrain_pipelines.dag_engine.sdk.stores import (
    AttrsDiff,
    ExecutionParams,
    TaskExitContext,
    TaskExitPayload,
)


# ---------------------------------------------------------------------------
# AttrsDiff
# ---------------------------------------------------------------------------


class TestAttrsDiff:
    def test_attrsdiff_repr(self):
        d = AttrsDiff(
            only_in_self=["a"],
            modified=["b"],
            only_in_other=["c"],
        )

        assert repr(d) == (
            "AttrsDiff(only_in_self=['a'], modified=['b'], only_in_other=['c'])"
        )


# ---------------------------------------------------------------------------
# ExecutionParams
# ---------------------------------------------------------------------------


class TestExecutionParams:
    """Cover ExecutionParams mapping, lazy resolution, and SHA-based comparison."""

    def test_init_stores_raw(self):
        raw = {"a": {"default": 1}}
        params = ExecutionParams(raw, "/tmp/meta")
        assert params._raw is raw

    def test_active_storable_override_vs_default(self):
        raw = {"k": {"default": "def_val", "override": "ovr_val"}}
        params = ExecutionParams(raw, "/tmp/meta")
        assert params._active_storable("k") == "ovr_val"
        del raw["k"]["override"]
        assert params._active_storable("k") == "def_val"

    def test_getitem_resolves_storable(self):
        raw = {"x": {"default": 99}}
        params = ExecutionParams(raw, "/tmp/meta")
        assert params["x"] == 99

    def test_getitem_resolves_disk_ref(self, metadata_root, write_disk_artifact):
        """Covers branch where active storable is a disk-ref."""
        write_disk_artifact(metadata_root, "val.pkl", "disk_val")
        disk_ref = {"__disk_ref__": "val.pkl", "__sha__": "abc"}
        raw = {"x": {"default": disk_ref}}
        params = ExecutionParams(raw, metadata_root)
        assert params["x"] == "disk_val"

    def test_contains_true(self):
        assert "a" in ExecutionParams({"a": {}}, "/tmp/meta")

    def test_contains_false(self):
        assert "b" not in ExecutionParams({"a": {}}, "/tmp/meta")

    def test_iter(self):
        params = ExecutionParams({"a": {}, "b": {}}, "/tmp/meta")
        assert list(params) == ["a", "b"]

    def test_len(self):
        assert len(ExecutionParams({"a": {}, "b": {}}, "/tmp/meta")) == 2

    def test_repr(self):
        params = ExecutionParams({"x": {}, "y": {}}, "/tmp/meta")
        assert repr(params) == "ExecutionParams(params=['x', 'y'])"

    def test_keys(self):
        params = ExecutionParams({"k1": {}, "k2": {}}, "/tmp/meta")
        assert list(params.keys()) == ["k1", "k2"]

    def test_description_returns_string(self):
        params = ExecutionParams(
            {"p": {"description": "a dummy param", "default": 1}}, "/tmp/meta"
        )
        assert params.description("p") == "a dummy param"

    def test_description_missing_key_raises(self):
        params = ExecutionParams({"p": {"description": "x"}}, "/tmp/meta")
        with pytest.raises(KeyError):
            params.description("missing")

    def test_default_native_value(self):
        params = ExecutionParams(
            {"p": {"description": "d", "default": 42}}, "/tmp/meta"
        )
        assert params.default("p") == 42

    def test_default_resolves_disk_ref(self, metadata_root, write_disk_artifact):
        """Covers branch where default is a disk-ref; load_from_disk reads a real file."""
        write_disk_artifact(metadata_root, "some/path.pkl", "unpickled_value")
        disk_ref = {"__sha__": "abc", "__disk_ref__": "some/path.pkl"}
        params = ExecutionParams(
            {"p": {"description": "d", "default": disk_ref}},
            metadata_root,
        )
        result = params.default("p")
        assert result == "unpickled_value"

    def test_default_no_default_key_returns_none(self):
        params = ExecutionParams({"p": {"description": "d"}}, "/tmp/meta")
        assert params.default("p") is None

    def test_default_ignores_override(self):
        """default() always returns the default value, never the override."""
        params = ExecutionParams(
            {"p": {"description": "d", "default": "orig", "override": "overridden"}},
            "/tmp/meta",
        )
        assert params.default("p") == "orig"

    def test_param_equals_native_match(self):
        p1 = ExecutionParams({"m": {"default": 1}}, "/tmp/meta")
        p2 = ExecutionParams({"m": {"default": 1}}, "/tmp/meta")
        assert p1.param_equals("m", p2) is True

    def test_param_equals_native_mismatch(self):
        p1 = ExecutionParams({"m": {"default": 1}}, "/tmp/meta")
        p2 = ExecutionParams({"m": {"default": 2}}, "/tmp/meta")
        assert p1.param_equals("m", p2) is False

    def test_param_equals_disk_ref_match(self):
        disk_a = {"__sha__": "hash1", "__disk_ref__": True}
        disk_b = {"__sha__": "hash1", "__disk_ref__": True}
        p1 = ExecutionParams({"m": {"default": disk_a}}, "/tmp/meta")
        p2 = ExecutionParams({"m": {"default": disk_b}}, "/tmp/meta")
        assert p1.param_equals("m", p2) is True

    def test_param_equals_disk_ref_mismatch(self):
        disk_a = {"__sha__": "hash1", "__disk_ref__": True}
        disk_b = {"__sha__": "hash2", "__disk_ref__": True}
        p1 = ExecutionParams({"m": {"default": disk_a}}, "/tmp/meta")
        p2 = ExecutionParams({"m": {"default": disk_b}}, "/tmp/meta")
        assert p1.param_equals("m", p2) is False

    def test_param_equals_mixed_storage_mismatch(self):
        """Covers branch where one param is disk-pickled and the other is inline."""
        disk_a = {"__sha__": "hash1", "__disk_ref__": True}
        p1 = ExecutionParams({"m": {"default": disk_a}}, "/tmp/meta")
        p2 = ExecutionParams({"m": {"default": 1}}, "/tmp/meta")
        assert p1.param_equals("m", p2) is False

    def test_diff(self):
        disk_ref = {"__disk_ref__": "path", "__sha__": "abc123"}
        p1 = ExecutionParams(
            {
                "only1": {"default": 1},
                "same": {"default": 2},
                "mod": {"default": 3},
                "disk": {"default": disk_ref},
            },
            "/tmp",
        )
        p2 = ExecutionParams(
            {
                "same": {"default": 2},
                "mod": {"default": 4},
                "only2": {"default": 5},
                "disk": {"default": disk_ref},
            },
            "/tmp",
        )
        diff = p1.diff(p2)

        assert diff.only_in_self == ["only1"]
        assert diff.modified == ["mod"]
        assert diff.only_in_other == ["only2"]

    def test_diff_identical(self):
        params = {"foo": {"default": 1}}
        lhs = ExecutionParams(params, "/tmp")
        rhs = ExecutionParams(params, "/tmp")
        diff = lhs.diff(rhs)
        assert diff.only_in_self == []
        assert diff.modified == []
        assert diff.only_in_other == []

    def test_diff_mixed_storage_mismatch(self):
        """Covers branch in diff() where one param is disk and the other is inline."""
        disk_a = {"__sha__": "hash1", "__disk_ref__": True}
        p1 = ExecutionParams({"mod": {"default": disk_a}}, "/tmp")
        p2 = ExecutionParams({"mod": {"default": 1}}, "/tmp")
        diff = p1.diff(p2)
        assert diff.modified == ["mod"]


# ---------------------------------------------------------------------------
# TaskExitContext
# ---------------------------------------------------------------------------


class TestTaskExitContext:
    def test_getitem_inline(self, make_row):
        rows = [make_row("a", inline_val="v1")]
        ctx = TaskExitContext(rows, "/tmp")
        assert ctx["a"] == "v1"

    def test_getitem_disk_ref(self, metadata_root, make_row, write_disk_artifact):
        """Covers branch where attr is a disk-ref; load_from_disk reads a real file."""
        write_disk_artifact(metadata_root, "path.pkl", "unpickled")
        rows = [make_row("a", disk_ref="path.pkl")]
        ctx = TaskExitContext(rows, metadata_root)
        val = ctx["a"]
        assert val == "unpickled"

    def test_getitem_disk_ref_sets_retrain_pipeline_type_env(
        self, metadata_root, make_row, write_disk_artifact, monkeypatch
    ):
        """Covers branch where retrain_pipeline_type attr is present with a
        non-None inline value and the env var was not previously set.

        The context manager sets the env var for the duration of the disk
        load, then removes it (sentinel path) on exit.
        """
        monkeypatch.delenv("retrain_pipeline_type", raising=False)
        write_disk_artifact(metadata_root, "path.pkl", "unpickled")
        rows = [
            make_row("retrain_pipeline_type", inline_val="test_pipeline"),
            make_row("a", disk_ref="path.pkl"),
        ]
        ctx = TaskExitContext(rows, metadata_root)
        assert ctx["a"] == "unpickled"
        assert "retrain_pipeline_type" not in os.environ

    def test_getitem_disk_ref_restores_retrain_pipeline_type_env(
        self, metadata_root, make_row, write_disk_artifact, monkeypatch
    ):
        """Covers branch where retrain_pipeline_type attr is present with a
        non-None inline value and the env var was already set.

        The context manager restores the original value on exit.
        """
        monkeypatch.setenv("retrain_pipeline_type", "original_value")
        write_disk_artifact(metadata_root, "path.pkl", "unpickled")
        rows = [
            make_row("retrain_pipeline_type", inline_val="test_pipeline"),
            make_row("a", disk_ref="path.pkl"),
        ]
        ctx = TaskExitContext(rows, metadata_root)
        assert ctx["a"] == "unpickled"
        assert os.environ["retrain_pipeline_type"] == "original_value"

    def test_getitem_disk_ref_retrain_pipeline_type_none(
        self, metadata_root, make_row, write_disk_artifact, monkeypatch
    ):
        """Covers branch where retrain_pipeline_type attr is present but its
        inline value is None; the context manager falls through to a plain
        yield without touching the env var.
        """
        monkeypatch.delenv("retrain_pipeline_type", raising=False)
        write_disk_artifact(metadata_root, "path.pkl", "unpickled")
        rows = [
            make_row("retrain_pipeline_type", inline_val=None),
            make_row("a", disk_ref="path.pkl"),
        ]
        ctx = TaskExitContext(rows, metadata_root)
        assert ctx["a"] == "unpickled"
        assert "retrain_pipeline_type" not in os.environ

    def test_get_found_and_not_found(self, make_row):
        rows = [make_row("a", inline_val="v1")]
        ctx = TaskExitContext(rows, "/tmp")
        assert ctx.get("a") == "v1"
        assert ctx.get("missing", "default") == "default"

    def test_contains(self, make_row):
        rows = [make_row("a")]
        ctx = TaskExitContext(rows, "/tmp")
        assert "a" in ctx
        assert "b" not in ctx

    def test_iter_and_len_and_keys(self, make_row):
        rows = [make_row("a"), make_row("b")]
        ctx = TaskExitContext(rows, "/tmp")
        assert list(ctx) == ["a", "b"]
        assert len(ctx) == 2
        assert list(ctx.keys()) == ["a", "b"]

    def test_repr(self, make_row):
        rows = [make_row("a")]
        ctx = TaskExitContext(rows, "/tmp")
        assert repr(ctx) == "TaskExitContext(attrs=['a'])"

    def test_attr_equals(self, make_row):
        r1 = make_row("a", inline_val="v1", sha=None)
        r2 = make_row("a", inline_val="v1", sha=None)
        r3 = make_row("a", inline_val="v2", sha=None)

        ctx1 = TaskExitContext([r1], "/tmp")
        ctx2 = TaskExitContext([r2], "/tmp")
        ctx3 = TaskExitContext([r3], "/tmp")
        ctx4 = TaskExitContext([], "/tmp")

        assert ctx1.attr_equals("a", ctx2) is True
        assert ctx1.attr_equals("a", ctx3) is False
        assert ctx1.attr_equals("missing", ctx4) is True  # both missing
        assert ctx1.attr_equals("a", ctx4) is False  # one missing

    def test_attr_equals_disk_ref(self, make_row):
        """Covers branch in attr_equals where one or both rows are disk-pickled."""
        r1 = make_row("a", disk_ref="d1", sha="h1")
        r2 = make_row("a", disk_ref="d1", sha="h1")
        r3 = make_row("a", disk_ref="d1", sha="h2")

        ctx1 = TaskExitContext([r1], "/tmp")
        ctx2 = TaskExitContext([r2], "/tmp")
        ctx3 = TaskExitContext([r3], "/tmp")

        assert ctx1.attr_equals("a", ctx2) is True
        assert ctx1.attr_equals("a", ctx3) is False

    @pytest.mark.parametrize(
        "case_id, self_rows, other_rows, expected_only_in_self, expected_modified, "
        "expected_only_in_other",
        [
            (
                "all_inline",
                [
                    ("only1", None, "v1", None),
                    ("mod", None, "v1", None),
                    ("same", None, "v1", None),
                ],
                [
                    ("only2", None, "v2", None),
                    ("mod", None, "v2", None),
                    ("same", None, "v1", None),
                ],
                ["only1"],
                ["mod"],
                ["only2"],
            ),
            (
                "all_disk_ref",
                [
                    ("only1", "d1", None, "h1"),
                    ("mod", "d2", None, "h1"),
                    ("same", "d3", None, "h1"),
                ],
                [
                    ("only2", "d4", None, "h2"),
                    ("mod", "d2", None, "h2"),
                    ("same", "d3", None, "h1"),
                ],
                ["only1"],
                ["mod"],
                ["only2"],
            ),
            (
                "mixed_inline_and_disk",
                [
                    ("only1", None, "v1", None),
                    ("mod", "d1", None, "h1"),
                    ("same", None, "v1", None),
                    ("mixed_mod", "d2", None, "h1"),
                ],
                [
                    ("only2", "d3", None, "h2"),
                    ("mod", "d1", None, "h2"),
                    ("same", None, "v1", None),
                    ("mixed_mod", None, "v1", None),
                ],
                ["only1"],
                ["mixed_mod", "mod"],
                ["only2"],
            ),
            (
                "inline_vs_disk_mismatch",
                [("attr", None, "v1", None)],
                [("attr", "d1", None, "h1")],
                [],
                ["attr"],
                [],
            ),
        ],
    )
    def test_diff(
        self,
        make_row,
        case_id,
        self_rows,
        other_rows,
        expected_only_in_self,
        expected_modified,
        expected_only_in_other,
    ):
        r1 = [make_row(n, disk_ref=d, inline_val=v, sha=s) for n, d, v, s in self_rows]
        r2 = [make_row(n, disk_ref=d, inline_val=v, sha=s) for n, d, v, s in other_rows]

        ctx1 = TaskExitContext(r1, "/tmp")
        ctx2 = TaskExitContext(r2, "/tmp")

        diff = ctx1.diff(ctx2)
        assert diff.only_in_self == expected_only_in_self
        assert diff.only_in_other == expected_only_in_other
        assert diff.modified == expected_modified

    @pytest.mark.parametrize(
        "case_id, rows_config",
        [
            (
                "all_inline",
                [("a", None, "v1", None), ("b", None, "v2", None)],
            ),
            (
                "all_disk_ref",
                [("a", "d1", None, "h1"), ("b", "d2", None, "h2")],
            ),
            (
                "mixed_inline_and_disk",
                [("a", None, "v1", None), ("b", "d1", None, "h1")],
            ),
        ],
    )
    def test_diff_identical(self, make_row, case_id, rows_config):
        rows = [
            make_row(n, disk_ref=d, inline_val=v, sha=s) for n, d, v, s in rows_config
        ]

        ctx1 = TaskExitContext(rows, "/tmp")
        ctx2 = TaskExitContext(rows, "/tmp")

        diff = ctx1.diff(ctx2)

        assert diff.only_in_self == []
        assert diff.modified == []
        assert diff.only_in_other == []


# ---------------------------------------------------------------------------
# TaskExitPayload – construction & _load_row
# ---------------------------------------------------------------------------


class TestTaskExitPayloadLoadRow:
    """Cover __init__, _load_row, and the value property."""

    def test_init_stores_fields(self, make_row):
        rows = {"t": make_row(attr_name="t", inline_val=1)}
        payload = TaskExitPayload(rows, "/tmp/meta")
        assert payload._rows is rows
        assert payload._metadata_root == "/tmp/meta"

    def test_load_row_none(self):
        payload = TaskExitPayload({"t": None}, "/tmp/meta")
        assert payload._load_row(None) is None

    def test_load_row_inline(self, make_row):
        row = make_row(inline_val=42)
        payload = TaskExitPayload({"t": row}, "/tmp/meta")
        assert payload._load_row(row) == 42

    def test_load_row_disk_ref(self, metadata_root, make_row, write_disk_artifact):
        write_disk_artifact(metadata_root, "payload.pkl", "disk_value")
        row = make_row(disk_ref="payload.pkl")
        payload = TaskExitPayload({"t": row}, metadata_root)
        assert payload._load_row(row) == "disk_value"

    def test_value_single_entry_inline(self, make_row):
        row = make_row(inline_val="hello")
        payload = TaskExitPayload({"t": row}, "/tmp/meta")
        assert payload.value == "hello"

    def test_value_single_entry_disk(
        self, metadata_root, make_row, write_disk_artifact
    ):
        write_disk_artifact(metadata_root, "p.pkl", {"k": "v"})
        row = make_row(disk_ref="p.pkl")
        payload = TaskExitPayload({"t": row}, metadata_root)
        assert payload.value == {"k": "v"}

    def test_value_multiple_entries(self, make_row):
        rows = {
            "a": make_row(inline_val=1),
            "b": make_row(inline_val=2),
        }
        payload = TaskExitPayload(rows, "/tmp/meta")
        assert payload.value == {"a": 1, "b": 2}


# ---------------------------------------------------------------------------
# TaskExitPayload – payload_equals & __eq__
# ---------------------------------------------------------------------------


class TestTaskExitPayloadEquals:
    """Cover payload_equals() and __eq__()."""

    def test_payload_equals_not_task_exit_payload(self, make_row):
        payload = TaskExitPayload({"t": make_row(inline_val=1)}, "/tmp")
        assert payload.payload_equals("not_a_payload") is False

    def test_payload_equals_different_lengths(self, make_row):
        p1 = TaskExitPayload({"a": make_row(inline_val=1)}, "/tmp")
        p2 = TaskExitPayload(
            {
                "a": make_row(inline_val=1),
                "b": make_row(inline_val=2),
            },
            "/tmp",
        )
        assert p1.payload_equals(p2) is False

    def test_payload_equals_key_missing_in_other(self, make_row):
        p1 = TaskExitPayload({"a": make_row(inline_val=1)}, "/tmp")
        p2 = TaskExitPayload({"b": make_row(inline_val=1)}, "/tmp")
        assert p1.payload_equals(p2) is False

    def test_payload_equals_both_rows_none(self):
        p1 = TaskExitPayload({"a": None}, "/tmp")
        p2 = TaskExitPayload({"a": None}, "/tmp")
        assert p1.payload_equals(p2) is True

    def test_payload_equals_one_row_none(self, make_row):
        p1 = TaskExitPayload({"a": None}, "/tmp")
        p2 = TaskExitPayload({"a": make_row(inline_val=1)}, "/tmp")
        assert p1.payload_equals(p2) is False

    def test_payload_equals_mixed_storage(self, make_row):
        """One row disk-pickled, the other inline: different storage types."""
        p1 = TaskExitPayload({"a": make_row(disk_ref="d1", sha="h1")}, "/tmp")
        p2 = TaskExitPayload({"a": make_row(inline_val=1)}, "/tmp")
        assert p1.payload_equals(p2) is False

    def test_payload_equals_disk_sha_match(self, make_row):
        p1 = TaskExitPayload({"a": make_row(disk_ref="d1", sha="h1")}, "/tmp")
        p2 = TaskExitPayload({"a": make_row(disk_ref="d2", sha="h1")}, "/tmp")
        assert p1.payload_equals(p2) is True

    def test_payload_equals_disk_sha_mismatch(self, make_row):
        p1 = TaskExitPayload({"a": make_row(disk_ref="d1", sha="h1")}, "/tmp")
        p2 = TaskExitPayload({"a": make_row(disk_ref="d2", sha="h2")}, "/tmp")
        assert p1.payload_equals(p2) is False

    def test_payload_equals_inline_match(self, make_row):
        p1 = TaskExitPayload({"a": make_row(inline_val=1)}, "/tmp")
        p2 = TaskExitPayload({"a": make_row(inline_val=1)}, "/tmp")
        assert p1.payload_equals(p2) is True

    def test_payload_equals_inline_mismatch(self, make_row):
        p1 = TaskExitPayload({"a": make_row(inline_val=1)}, "/tmp")
        p2 = TaskExitPayload({"a": make_row(inline_val=2)}, "/tmp")
        assert p1.payload_equals(p2) is False

    def test_payload_equals_multiple_entries_all_match(self, make_row):
        rows1 = {
            "a": make_row(inline_val=1),
            "b": make_row(disk_ref="d1", sha="h1"),
        }
        rows2 = {
            "a": make_row(inline_val=1),
            "b": make_row(disk_ref="d2", sha="h1"),
        }
        p1 = TaskExitPayload(rows1, "/tmp")
        p2 = TaskExitPayload(rows2, "/tmp")
        assert p1.payload_equals(p2) is True

    def test_eq_with_task_exit_payload(self, make_row):
        rows = {"a": make_row(inline_val=1)}
        p1 = TaskExitPayload(rows, "/tmp")
        p2 = TaskExitPayload(rows, "/tmp")
        assert (p1 == p2) is True

    def test_eq_with_non_payload_single_entry(self, make_row):
        row = make_row(inline_val=42)
        payload = TaskExitPayload({"a": row}, "/tmp")
        assert (payload == 42) is True

    def test_eq_with_non_payload_multiple_entries(self, make_row):
        rows = {
            "a": make_row(inline_val=1),
            "b": make_row(inline_val=2),
        }
        payload = TaskExitPayload(rows, "/tmp")
        assert (payload == 42) is False


# ---------------------------------------------------------------------------
# TaskExitPayload – mapping protocol
# ---------------------------------------------------------------------------


class TestTaskExitPayloadMapping:
    """Cover __getitem__, __setitem__, __contains__, get, keys, values, items, copy."""

    def test_getitem_single_entry_key_is_task_name(self, make_row):
        row = make_row(attr_name="task_a", inline_val="value")
        payload = TaskExitPayload({"task_a": row}, "/tmp")
        assert payload["task_a"] == "value"

    def test_getitem_single_entry_delegates_to_wrapped_value(self, make_row):
        """Single entry, hashable key not in rows: delegates to wrapped value."""
        row = make_row(attr_name="task_a", inline_val={"x": 10, "y": 20})
        payload = TaskExitPayload({"task_a": row}, "/tmp")
        assert payload["x"] == 10

    def test_getitem_single_entry_unhashable_key(self, make_row):
        """Single entry, unhashable key: TypeError caught, delegates to value.

        The wrapped list does not support list-indexing, so the delegation
        itself raises TypeError — exercising the except/pass branch.
        """
        row = make_row(attr_name="task_a", inline_val=[10, 20, 30])
        payload = TaskExitPayload({"task_a": row}, "/tmp")
        with pytest.raises(TypeError):
            payload[[0, 1]]

    def test_getitem_multiple_entries(self, make_row):
        rows = {
            "a": make_row(inline_val=1),
            "b": make_row(inline_val=2),
        }
        payload = TaskExitPayload(rows, "/tmp")
        assert payload["a"] == 1
        assert payload["b"] == 2

    def test_getitem_multiple_entries_missing_key(self, make_row):
        """Multi-entry missing key triggers dict lookup raising KeyError."""
        rows = {
            "a": make_row(inline_val=1),
            "b": make_row(inline_val=2),
        }
        payload = TaskExitPayload(rows, "/tmp")
        with pytest.raises(KeyError):
            payload["missing"]

    def test_setitem_raises_typeerror(self, make_row):
        payload = TaskExitPayload({"a": make_row(inline_val=1)}, "/tmp")
        with pytest.raises(TypeError, match="immutable"):
            payload["a"] = 99

    def test_contains(self, make_row):
        rows = {"a": make_row(inline_val=1)}
        payload = TaskExitPayload(rows, "/tmp")
        assert "a" in payload
        assert "b" not in payload

    def test_get_found(self, make_row):
        payload = TaskExitPayload({"a": make_row(inline_val=1)}, "/tmp")
        assert payload.get("a") == 1

    def test_get_not_found_returns_default(self, make_row):
        payload = TaskExitPayload({"a": make_row(inline_val=1)}, "/tmp")
        assert payload.get("missing", "default") == "default"

    def test_keys(self, make_row):
        rows = {
            "a": make_row(inline_val=1),
            "b": make_row(inline_val=2),
        }
        payload = TaskExitPayload(rows, "/tmp")
        assert list(payload.keys()) == ["a", "b"]

    def test_values_single_entry(self, make_row):
        row = make_row(inline_val=42)
        payload = TaskExitPayload({"a": row}, "/tmp")
        assert list(payload.values()) == [42]

    def test_values_multiple_entries(self, make_row):
        rows = {
            "a": make_row(inline_val=1),
            "b": make_row(inline_val=2),
        }
        payload = TaskExitPayload(rows, "/tmp")
        assert list(payload.values()) == [1, 2]

    def test_items_single_entry(self, make_row):
        row = make_row(inline_val=42)
        payload = TaskExitPayload({"a": row}, "/tmp")
        assert list(payload.items()) == [("a", 42)]

    def test_items_multiple_entries(self, make_row):
        rows = {
            "a": make_row(inline_val=1),
            "b": make_row(inline_val=2),
        }
        payload = TaskExitPayload(rows, "/tmp")
        assert list(payload.items()) == [("a", 1), ("b", 2)]

    def test_copy(self, make_row):
        rows = {"a": make_row(inline_val=1)}
        payload = TaskExitPayload(rows, "/tmp")
        copied = payload.copy()
        assert isinstance(copied, TaskExitPayload)
        assert copied._rows is rows
        assert copied._metadata_root == "/tmp"


# ---------------------------------------------------------------------------
# TaskExitPayload – dunders: bool, hash, len, iter
# ---------------------------------------------------------------------------


class TestTaskExitPayloadDunders:
    """Cover __bool__, __hash__, __len__, __iter__."""

    def test_bool_true(self, make_row):
        payload = TaskExitPayload({"a": make_row(inline_val=1)}, "/tmp")
        assert bool(payload) is True

    def test_bool_false(self):
        payload = TaskExitPayload({}, "/tmp")
        assert bool(payload) is False

    def test_hash_single_entry(self, make_row):
        row = make_row(inline_val=42)
        payload = TaskExitPayload({"a": row}, "/tmp")
        assert hash(payload) == hash(42)

    def test_hash_multiple_entries(self, make_row):
        rows = {
            "a": make_row(inline_val=1),
            "b": make_row(inline_val=2),
        }
        payload = TaskExitPayload(rows, "/tmp")
        assert hash(payload) == hash(tuple(sorted(rows.keys())))

    def test_len_single_entry_with_len(self, make_row):
        row = make_row(inline_val=[1, 2, 3])
        payload = TaskExitPayload({"a": row}, "/tmp")
        assert len(payload) == 3

    def test_len_single_entry_without_len(self, make_row):
        row = make_row(inline_val=42)
        payload = TaskExitPayload({"a": row}, "/tmp")
        assert len(payload) == 1

    def test_len_multiple_entries(self, make_row):
        rows = {
            "a": make_row(inline_val=1),
            "b": make_row(inline_val=2),
        }
        payload = TaskExitPayload(rows, "/tmp")
        assert len(payload) == 2

    def test_iter_single_entry_iterable(self, make_row):
        row = make_row(inline_val=[1, 2, 3])
        payload = TaskExitPayload({"a": row}, "/tmp")
        assert list(payload) == [1, 2, 3]

    def test_iter_single_entry_str(self, make_row):
        """str is iterable but should be yielded as a single item."""
        row = make_row(inline_val="hello")
        payload = TaskExitPayload({"a": row}, "/tmp")
        assert list(payload) == ["hello"]

    def test_iter_single_entry_bytes(self, make_row):
        """bytes is iterable but should be yielded as a single item."""
        row = make_row(inline_val=b"data")
        payload = TaskExitPayload({"a": row}, "/tmp")
        assert list(payload) == [b"data"]

    def test_iter_single_entry_not_iterable(self, make_row):
        row = make_row(inline_val=42)
        payload = TaskExitPayload({"a": row}, "/tmp")
        assert list(payload) == [42]

    def test_iter_multiple_entries(self, make_row):
        rows = {
            "a": make_row(inline_val=1),
            "b": make_row(inline_val=2),
        }
        payload = TaskExitPayload(rows, "/tmp")
        assert list(payload) == ["a", "b"]


# ---------------------------------------------------------------------------
# TaskExitPayload – arithmetic
# ---------------------------------------------------------------------------


class TestTaskExitPayloadArithmetic:
    """Cover __add__, __radd__, __mul__, __rmul__."""

    def test_add_single_entry(self, make_row):
        row = make_row(inline_val=10)
        payload = TaskExitPayload({"a": row}, "/tmp")
        assert payload + 5 == 15

    def test_add_multiple_entries_not_implemented(self, make_row):
        rows = {
            "a": make_row(inline_val=1),
            "b": make_row(inline_val=2),
        }
        payload = TaskExitPayload(rows, "/tmp")
        with pytest.raises(TypeError):
            payload + 5

    def test_radd_single_entry(self, make_row):
        row = make_row(inline_val=10)
        payload = TaskExitPayload({"a": row}, "/tmp")
        assert 5 + payload == 15

    def test_radd_multiple_entries_not_implemented(self, make_row):
        rows = {
            "a": make_row(inline_val=1),
            "b": make_row(inline_val=2),
        }
        payload = TaskExitPayload(rows, "/tmp")
        with pytest.raises(TypeError):
            5 + payload

    def test_mul_single_entry(self, make_row):
        row = make_row(inline_val=10)
        payload = TaskExitPayload({"a": row}, "/tmp")
        assert payload * 3 == 30

    def test_mul_multiple_entries_not_implemented(self, make_row):
        rows = {
            "a": make_row(inline_val=1),
            "b": make_row(inline_val=2),
        }
        payload = TaskExitPayload(rows, "/tmp")
        with pytest.raises(TypeError):
            payload * 3

    def test_rmul_single_entry(self, make_row):
        row = make_row(inline_val=10)
        payload = TaskExitPayload({"a": row}, "/tmp")
        assert 3 * payload == 30

    def test_rmul_multiple_entries_not_implemented(self, make_row):
        rows = {
            "a": make_row(inline_val=1),
            "b": make_row(inline_val=2),
        }
        payload = TaskExitPayload(rows, "/tmp")
        with pytest.raises(TypeError):
            3 * payload


# ---------------------------------------------------------------------------
# TaskExitPayload – __getattr__
# ---------------------------------------------------------------------------


class TestTaskExitPayloadGetattr:
    """Cover __getattr__ branches."""

    def test_getattr_underscore_name_raises(self, make_row):
        payload = TaskExitPayload({"a": make_row(inline_val=1)}, "/tmp")
        with pytest.raises(AttributeError):
            payload._nonexistent

    def test_getattr_single_entry_row_none(self):
        payload = TaskExitPayload({"a": None}, "/tmp")
        with pytest.raises(AttributeError):
            payload.some_attr

    def test_getattr_single_entry_delegates(self, make_row):
        """Single entry: attribute access delegates to the loaded value."""
        row = make_row(inline_val="hello")
        payload = TaskExitPayload({"a": row}, "/tmp")
        assert payload.upper() == "HELLO"

    def test_getattr_multiple_entries_raises(self, make_row):
        rows = {
            "a": make_row(inline_val=1),
            "b": make_row(inline_val=2),
        }
        payload = TaskExitPayload(rows, "/tmp")
        with pytest.raises(AttributeError):
            payload.some_attr


# ---------------------------------------------------------------------------
# TaskExitPayload – __str__ & __repr__
# ---------------------------------------------------------------------------


class TestTaskExitPayloadStr:
    """Cover __str__ and __repr__."""

    def test_str_single_entry_inline(self, make_row):
        row = make_row(inline_val=42)
        payload = TaskExitPayload({"a": row}, "/tmp")
        assert str(payload) == "TaskExitPayload(42)"

    def test_str_single_entry_disk_ref(self, make_row):
        row = make_row(disk_ref="path.pkl", inline_val=None)
        payload = TaskExitPayload({"a": row}, "/tmp")
        assert str(payload) == "TaskExitPayload(path.pkl)"

    def test_str_single_entry_row_none(self):
        payload = TaskExitPayload({"a": None}, "/tmp")
        assert str(payload) == "TaskExitPayload(None)"

    def test_str_multiple_entries(self, make_row):
        rows = {
            "a": make_row(inline_val=1),
            "b": make_row(disk_ref="p.pkl", inline_val=None),
        }
        payload = TaskExitPayload(rows, "/tmp")
        assert str(payload) == "TaskExitPayload({'a': 1, 'b': 'p.pkl'})"

    def test_str_multiple_entries_with_none_row(self, make_row):
        rows = {"a": make_row(inline_val=1), "b": None}
        payload = TaskExitPayload(rows, "/tmp")
        assert str(payload) == "TaskExitPayload({'a': 1, 'b': None})"

    def test_repr_matches_str(self, make_row):
        row = make_row(inline_val=42)
        payload = TaskExitPayload({"a": row}, "/tmp")
        assert repr(payload) == str(payload)


# ---------------------------------------------------------------------------
# TaskExitPayload – pickling
# ---------------------------------------------------------------------------


class TestTaskExitPayloadPickle:
    """Cover __getstate__, __setstate__, __reduce__."""

    def test_getstate(self, make_row):
        rows = {"a": make_row(inline_val=1)}
        payload = TaskExitPayload(rows, "/tmp/meta")
        state = payload.__getstate__()
        assert state == {"_rows": rows, "_metadata_root": "/tmp/meta"}

    def test_setstate(self, make_row):
        rows = {"a": make_row(inline_val=1)}
        payload = TaskExitPayload.__new__(TaskExitPayload)
        payload.__setstate__({"_rows": rows, "_metadata_root": "/tmp/meta"})
        assert payload._rows is rows
        assert payload._metadata_root == "/tmp/meta"

    def test_reduce(self, make_row):
        rows = {"a": make_row(inline_val=1)}
        payload = TaskExitPayload(rows, "/tmp/meta")
        cls, args = payload.__reduce__()
        assert cls is TaskExitPayload
        assert args == (rows, "/tmp/meta")

    def test_pickle_roundtrip(self, make_row):
        rows = {"a": make_row(inline_val=42)}
        payload = TaskExitPayload(rows, "/tmp/meta")
        restored = pickle.loads(pickle.dumps(payload))
        assert isinstance(restored, TaskExitPayload)
        assert restored["a"] == 42
        assert restored._metadata_root == "/tmp/meta"
