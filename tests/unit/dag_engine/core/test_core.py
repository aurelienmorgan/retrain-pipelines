"""
Unit & Integration tests for retrain_pipelines.dag_engine.core.core
"""

import gc
import logging
import os
import pickle
import sys
import types
from datetime import date, datetime, timezone
from unittest.mock import MagicMock, patch
from uuid import UUID

import pytest
from pydantic import BaseModel

import retrain_pipelines.dag_engine.core.core as core_module
from retrain_pipelines.dag_engine.core.core import (
    DAG,
    DagExecutionContext,
    DagParam,
    DistributionNotSupportedError,
    MergeNotSupportedError,
    StreamToDb,
    TaskFuncException,
    TaskGroup,
    TaskGroupException,
    TaskMergeFuncException,
    TaskPayload,
    TaskType,
    UiCss,
    _dag_execution_context_var,
    _get_dag_params,
    ctx as dag_ctx,
    dag,
    parallel_task,
    task,
    taskgroup,
)
from retrain_pipelines.dag_engine.core import trace_buffer as trace_buffer_module
from retrain_pipelines.dag_engine.db.model import (
    TaskContextAttr,
    TaskGroup as TaskGroupORM,
    TaskPayloadAttr,
    TaskTrace,
)
from retrain_pipelines.dag_engine.stores import contexts_store as contexts_store_module
from retrain_pipelines.dag_engine.stores import payloads_store as payloads_store_module


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════


def _real_task(name: str, *, is_parallel=False, merge_func=None) -> TaskType:
    def func():
        pass

    func.__name__ = name
    func.__module__ = "__test__"
    return TaskType(func=func, is_parallel=is_parallel, merge_func=merge_func)


class _NoFuncDefClass:
    """Module-level class used to drive the no-FuncDef branch of _get_dag_params.

    inspect.getsource() returns the class source (ClassDef, no FunctionDef),
    so ast.walk yields no FunctionDef and _get_dag_params raises ValueError.
    """

    pass


# ══════════════════════════════════════════════════════════════════════════════
#  Exception constructors
# ══════════════════════════════════════════════════════════════════════════════


class TestExceptions:
    def test_task_func_exception(self):
        ex = TaskFuncException("bad task")
        assert str(ex) == "bad task"
        assert isinstance(ex, Exception)

    def test_task_merge_func_exception(self):
        ex = TaskMergeFuncException("bad merge")
        assert str(ex) == "bad merge"
        assert isinstance(ex, Exception)

    def test_task_group_exception(self):
        ex = TaskGroupException("bad group")
        assert str(ex) == "bad group"
        assert isinstance(ex, Exception)


# ══════════════════════════════════════════════════════════════════════════════
#  TaskPayload
# ══════════════════════════════════════════════════════════════════════════════


class TestTaskPayload:
    def test_getitem_exact_key(self):
        p = TaskPayload({"a": 1, "b": 2})
        assert p["a"] == 1 and p["b"] == 2

    def test_single_entry_delegates_inner_indexing(self):
        assert TaskPayload({"x": [10, 20, 30]})[1] == 20

    def test_single_entry_key_in_data_returned_directly(self):
        assert TaskPayload({"a": 42})["a"] == 42

    def test_single_entry_unhashable_key_raises_type_error_then_delegates(self):
        # key is unhashable => TypeError on `key in self._data` => delegates to inner value.
        # Inner value is an object whose __getitem__ accepts the unhashable key.
        class _UnhashableKeyMap:
            def __getitem__(self, key):
                return "found"

        p = TaskPayload({"x": _UnhashableKeyMap()})
        assert p[[1, 2]] == "found"

    def test_single_entry_hashable_key_not_in_data_delegates_to_inner(self):
        # hashable key not present in _data => falls through to inner value
        assert TaskPayload({"x": {0: "zero"}})[0] == "zero"

    def test_setitem(self):
        p = TaskPayload({})
        p["k"] = 99
        assert p["k"] == 99

    def test_contains(self):
        p = TaskPayload({"a": 1})
        assert "a" in p
        assert "z" not in p

    def test_get_with_and_without_default(self):
        p = TaskPayload({"a": 1})
        assert p.get("a") == 1
        assert p.get("z", 99) == 99

    def test_keys_values_items(self):
        p = TaskPayload({"a": 1, "b": 2})
        assert set(p.keys()) == {"a", "b"}
        assert set(p.values()) == {1, 2}
        assert set(p.items()) == {("a", 1), ("b", 2)}

    def test_copy_creates_independent_dict(self):
        p = TaskPayload({"a": [1, 2]})
        q = p.copy()
        q._data["a"] = [9]
        assert p["a"] == [1, 2]

    def test_bool_truthy_and_falsy(self):
        assert bool(TaskPayload({"a": 1}))
        assert not bool(TaskPayload({}))

    def test_eq_single_entry_compares_inner(self):
        assert TaskPayload({"x": 42}) == 42

    def test_eq_multi_entry_falls_back_to_object_eq(self):
        p = TaskPayload({"a": 1, "b": 2})
        assert not (p == 42)

    def test_hash_single_delegates_to_inner(self):
        assert hash(TaskPayload({"x": 5})) == hash(5)

    def test_hash_multi_entry(self):
        p = TaskPayload({"a": 1, "b": 2})
        assert isinstance(hash(p), int)

    def test_len_single_list(self):
        assert len(TaskPayload({"x": [1, 2, 3]})) == 3

    def test_len_single_non_sized(self):
        # single entry with no __len__ => returns 1
        assert len(TaskPayload({"x": 42})) == 1

    def test_len_multi_entry_counts_keys(self):
        assert len(TaskPayload({"a": 1, "b": 2})) == 2

    def test_iter_single_list(self):
        assert list(TaskPayload({"x": [10, 20]})) == [10, 20]

    def test_iter_single_non_iterable_wraps_in_list(self):
        assert list(TaskPayload({"x": 7})) == [7]

    def test_iter_single_string_wraps_in_list(self):
        # str is iterable but treated as scalar => wrapped
        assert list(TaskPayload({"x": "hi"})) == ["hi"]

    def test_iter_multi_entry_yields_keys(self):
        assert set(TaskPayload({"a": 1, "b": 2})) == {"a", "b"}

    def test_add_and_radd(self):
        p = TaskPayload({"x": 5})
        assert p + 3 == 8
        assert 3 + p == 8

    def test_mul_and_rmul(self):
        p = TaskPayload({"x": 5})
        assert p * 2 == 10
        assert 2 * p == 10

    def test_add_multi_entry_not_implemented(self):
        assert TaskPayload({"a": 1, "b": 2}).__add__(1) is NotImplemented

    def test_radd_multi_entry_not_implemented(self):
        assert TaskPayload({"a": 1, "b": 2}).__radd__(1) is NotImplemented

    def test_mul_multi_entry_not_implemented(self):
        assert TaskPayload({"a": 1, "b": 2}).__mul__(2) is NotImplemented

    def test_rmul_multi_entry_not_implemented(self):
        assert TaskPayload({"a": 1, "b": 2}).__rmul__(2) is NotImplemented

    def test_getattr_delegates_to_single_entry(self):
        class _Obj:
            foo = "bar"

        assert TaskPayload({"x": _Obj()}).foo == "bar"

    def test_getattr_raises_on_multi_entry(self):
        with pytest.raises(AttributeError):
            TaskPayload({"a": 1, "b": 2}).nonexistent

    def test_getattr_raises_when_data_missing(self):
        p = TaskPayload.__new__(TaskPayload)
        with pytest.raises(AttributeError):
            _ = p.something

    def test_getattr_raises_on_data_itself(self):
        p = TaskPayload.__new__(TaskPayload)
        with pytest.raises(AttributeError):
            _ = p._data

    def test_str_and_repr_contain_class_name(self):
        p = TaskPayload({"k": 7})
        assert "TaskPayload" in str(p)
        assert "TaskPayload" in repr(p)

    def test_pickle_roundtrip(self):
        p = TaskPayload({"a": [1, 2], "b": "hello"})
        q = pickle.loads(pickle.dumps(p))
        assert q["a"] == [1, 2]
        assert q["b"] == "hello"

    def test_getstate_setstate(self):
        p = TaskPayload({"z": 42})
        q = TaskPayload({})
        q.__setstate__(p.__getstate__())
        assert q["z"] == 42

    def test_reduce_callable(self):
        p = TaskPayload({"k": 1})
        ctor, (data,) = p.__reduce__()
        assert ctor is TaskPayload
        assert data == {"k": 1}


# ══════════════════════════════════════════════════════════════════════════════
#  DagExecutionContext
# ══════════════════════════════════════════════════════════════════════════════


class TestDagExecutionContext:
    def test_getattr_returns_known_param(self):
        assert DagExecutionContext({"exec_id": 7}).exec_id == 7

    def test_getattr_unknown_returns_none(self):
        assert DagExecutionContext({}).missing is None

    def test_private_attrs_not_proxied_through_params(self):
        ctx = DagExecutionContext({"_params": "should not reach"})
        assert isinstance(ctx._params, dict)

    def test_getattr_private_name_uses_object_getattribute(self):
        # name.startswith("_") => object.__getattribute__
        ctx = DagExecutionContext({})
        # _params is a real attribute, must be returned directly
        assert isinstance(ctx._params, dict)
        # _nonexistent private attr raises AttributeError
        with pytest.raises(AttributeError):
            _ = ctx._nonexistent_private

    def test_update_sets_and_tracks(self):
        ctx = DagExecutionContext({"a": 1})
        ctx.update(a=2, b=3)
        assert ctx.a == 2 and ctx.b == 3

    def test_get_updates_returns_copy(self):
        ctx = DagExecutionContext({})
        ctx.update(x=9)
        updates = ctx.get_updates()
        updates["x"] = 999
        assert ctx.get_updates()["x"] == 9  # mutation didn't affect context

    def test_concat_taskgroup_updates_applies_to_params(self):
        ctx = DagExecutionContext({"a": 1})
        # Pass attr_refs to cover the deep_update(self._attr_refs, attr_refs) branch.
        ctx.concat_taskgroup_updates(
            {"a": 99, "z": 0}, attr_refs={"p1": {"eTAG": "abc"}}
        )
        assert ctx.a == 99 and ctx.z == 0
        assert ctx._attr_refs["p1"] == {"eTAG": "abc"}

    def test_concat_taskgroup_updates_also_recorded_in_updates(self):
        ctx = DagExecutionContext({})
        ctx.concat_taskgroup_updates({"k": 5}, attr_refs={})
        assert ctx.get_updates()["k"] == 5

    def test_merge_parallel_updates_applies_to_params(self):
        ctx = DagExecutionContext({"a": 1})
        ctx.merge_parallel_updates({"a": 99, "z": 0})
        assert ctx.a == 99 and ctx.z == 0

    def test_merge_parallel_updates_recorded_in_updates(self):
        ctx = DagExecutionContext({})
        ctx.merge_parallel_updates({"k": 5})
        assert ctx.get_updates()["k"] == 5

    def test_merge_parallel_updates_deep_merges_nested(self):
        ctx = DagExecutionContext({"a": {"b": 1, "c": 2}})
        ctx.merge_parallel_updates({"a": {"b": 99}})
        assert ctx._params["a"]["b"] == 99
        assert ctx._params["a"]["c"] == 2

    def test_deep_update_nested_dict(self):
        d = {"a": {"b": 1, "c": 2}}
        DagExecutionContext.deep_update(d, {"a": {"b": 99}})
        assert d["a"]["b"] == 99
        assert d["a"]["c"] == 2  # untouched sibling

    def test_deep_update_overwrites_non_dict(self):
        d = {"a": 1}
        DagExecutionContext.deep_update(d, {"a": [1, 2]})
        assert d["a"] == [1, 2]

    def test_deep_update_adds_new_key(self):
        d = {}
        DagExecutionContext.deep_update(d, {"new": 42})
        assert d["new"] == 42

    def test_copy_is_deep(self):
        ctx = DagExecutionContext({"a": [1, 2]})
        ctx2 = ctx.copy()
        ctx2._params["a"].append(3)
        assert ctx._params["a"] == [1, 2]

    def test_dag_execution_context_get_attr_ref_updates(self):
        """Verify get_attr_ref_updates returns a shallow copy of the internal attribute references."""
        ctx = DagExecutionContext({})
        ctx._attr_refs = {"p1": {"eTAG": "abc"}}
        refs = ctx.get_attr_ref_updates()
        assert refs == {"p1": {"eTAG": "abc"}}

    def test_dag_execution_context_copy(self):
        """Verify that copy performs a deep copy of both parameters and attribute references."""
        ctx = DagExecutionContext({"a": 1})
        ctx._attr_refs = {"p1": {"eTAG": "abc"}}
        ctx2 = ctx.copy()
        assert ctx2._params == {"a": 1}
        assert ctx2._attr_refs == {"p1": {"eTAG": "abc"}}

    def test_dag_execution_context_init_attr_refs(self):
        """Verify _init_attr_refs_from_params resolves storables and populates attribute references.

        Calls the real attr_ref_from_param_storable against a params_json dict
        containing both an inline JSON-safe default and a disk-ref sentinel,
        covering both branches of the storable-type dispatch.
        """
        inline_storable = 42
        disk_ref_storable = {
            "__eTAG__": "abc123",
            "__sha__": "def456",
            "__disk_ref__": "tmp/params/defaults/p.pkl",
        }
        params_json = {
            "p_inline": {"description": "d", "default": inline_storable},
            "p_disk": {"description": "d", "default": disk_ref_storable},
        }
        ctx = DagExecutionContext({})
        ctx._init_attr_refs_from_params(params_json)
        # Inline storable => eTAG generated, no sha, no disk_ref, value kept inline.
        assert ctx._attr_refs["p_inline"]["inline"] == 42
        assert ctx._attr_refs["p_inline"]["sha"] is None
        assert ctx._attr_refs["p_inline"]["disk_ref"] is None
        # Disk-ref storable => eTAG/sha/disk_ref extracted from the sentinel.
        assert ctx._attr_refs["p_disk"]["disk_ref"] == "tmp/params/defaults/p.pkl"
        assert ctx._attr_refs["p_disk"]["sha"] == "def456"
        assert ctx._attr_refs["p_disk"]["inline"] is None


# ══════════════════════════════════════════════════════════════════════════════
#  UiCss
# ══════════════════════════════════════════════════════════════════════════════


class TestUiCss:
    def test_valid_six_char_hex(self):
        css = UiCss(background="#FF0000", color="#00FF00", border="#0000FF")
        assert css.background == "#FF0000"

    def test_valid_three_char_hex(self):
        assert UiCss(background="#f0f").background == "#f0f"

    def test_invalid_name_color_raises(self):
        with pytest.raises(Exception):
            UiCss(background="red")

    def test_missing_hash_raises(self):
        with pytest.raises(Exception):
            UiCss(color="FF0000")

    def test_all_none_is_valid(self):
        css = UiCss()
        assert css.background is None and css.color is None and css.border is None

    def test_to_dict_excludes_none_fields(self):
        d = UiCss(background="#abc").to_dict()
        assert "background" in d
        assert "color" not in d and "border" not in d

    def test_to_dict_includes_all_when_set(self):
        d = UiCss(background="#111", color="#222", border="#333").to_dict()
        assert len(d) == 3


# ══════════════════════════════════════════════════════════════════════════════
#  DagParam
# ══════════════════════════════════════════════════════════════════════════════


class TestDagParam:
    def test_default_is_none(self):
        assert DagParam(description="d").default is None

    def test_serializable_dict_basic(self, assets_cache):
        """int is JSON-native ; returned as-is by the real value_to_storable."""
        result = DagParam(description="d", default=42).to_serializable_dict(
            dir_id="t", param_name="p"
        )
        assert result == {"description": "d", "default": 42}

    def test_serialize_datetime(self, assets_cache):
        """datetime is JSON-native (ISO format) ; returned as-is."""
        ts = datetime(2024, 6, 1, tzinfo=timezone.utc)
        result = DagParam(description="d", default=ts).to_serializable_dict(
            dir_id="t", param_name="p"
        )
        assert result["description"] == "d"
        assert "2024-06-01" in str(result["default"])

    def test_serialize_date(self, assets_cache):
        """date is JSON-native (ISO format) ; returned as-is."""
        result = DagParam(
            description="d", default=date(2024, 6, 1)
        ).to_serializable_dict(dir_id="t", param_name="p")
        assert result["description"] == "d"
        assert "2024-06-01" in str(result["default"])

    def test_serialize_nested_dict(self, assets_cache):
        """dict with JSON-native values is returned as-is."""
        result = DagParam(description="d", default={"a": [1, 2]}).to_serializable_dict(
            dir_id="t", param_name="p"
        )
        assert result == {"description": "d", "default": {"a": [1, 2]}}

    def test_serialize_list(self, assets_cache):
        """list with JSON-native values is returned as-is."""
        result = DagParam(
            description="d", default=[1, "two", None]
        ).to_serializable_dict(dir_id="t", param_name="p")
        assert result == {"description": "d", "default": [1, "two", None]}

    def test_serialize_tuple(self, assets_cache):
        """tuple is JSON-native (converted to list) ; returned as-is."""
        result = DagParam(description="d", default=(1, 2)).to_serializable_dict(
            dir_id="t", param_name="p"
        )
        assert result["description"] == "d"
        assert list(result["default"]) == [1, 2]

    def test_serialize_set(self, assets_cache):
        """set is JSON-native (converted to list) ; returned as-is."""
        result = DagParam(description="d", default={99}).to_serializable_dict(
            dir_id="t", param_name="p"
        )
        assert result["description"] == "d"
        assert list(result["default"]) == [99]

    def test_serialize_none(self, assets_cache):
        """None is JSON-native ; returned as-is."""
        result = DagParam(description="d").to_serializable_dict(
            dir_id="t", param_name="p"
        )
        assert result == {"description": "d", "default": None}

    def test_serialize_bool_preserved(self, assets_cache):
        """bool is JSON-native ; returned as-is (not converted to int)."""
        result = DagParam(description="d", default=True).to_serializable_dict(
            dir_id="t", param_name="p"
        )
        assert result == {"description": "d", "default": True}

    def test_serialize_unknown_type_cloudpickled_to_disk(self, assets_cache):
        """Custom class instances are not JSON-native ; cloudpickled to disk.

        The returned default is a disk-ref sentinel dict containing
        __disk_ref__, __eTAG__, and __sha__.
        """

        class Custom:
            pass

        result = DagParam(description="d", default=Custom()).to_serializable_dict(
            dir_id="t", param_name="p"
        )
        assert result["description"] == "d"
        assert isinstance(result["default"], dict)
        assert "__disk_ref__" in result["default"]
        assert "__eTAG__" in result["default"]
        assert "__sha__" in result["default"]
        assert result["default"]["__disk_ref__"].endswith("p.pkl")

    def test_serialize_pydantic_model(self, assets_cache):
        """Pydantic models are JSON-native (model_dump) ; returned as-is."""

        class M(BaseModel):
            x: int = 1

        result = DagParam(description="d", default=M()).to_serializable_dict(
            dir_id="t", param_name="p"
        )
        assert result == {"description": "d", "default": {"x": 1}}


# ══════════════════════════════════════════════════════════════════════════════
#  _ContextProxy  (via module-level `ctx`)
# ══════════════════════════════════════════════════════════════════════════════


class TestContextProxy:
    def setup_method(self):
        _dag_execution_context_var.set(None)

    def test_getattr_raises_when_context_unset(self):
        with pytest.raises(RuntimeError, match="not set"):
            _ = dag_ctx.some_attr

    def test_setattr_raises_when_context_unset(self):
        with pytest.raises(RuntimeError, match="not set"):
            dag_ctx.x = 1

    def test_reads_from_active_context(self):
        ec = DagExecutionContext({"key": "val"})
        token = _dag_execution_context_var.set(ec)
        try:
            assert dag_ctx.key == "val"
        finally:
            _dag_execution_context_var.reset(token)

    def test_write_updates_params_and_updates_dict(self):
        ec = DagExecutionContext({})
        token = _dag_execution_context_var.set(ec)
        try:
            dag_ctx.new_key = 777
            assert ec._params["new_key"] == 777
            assert ec._updates["new_key"] == 777
        finally:
            _dag_execution_context_var.reset(token)

    def test_overwrites_existing_param(self):
        ec = DagExecutionContext({"a": 1})
        token = _dag_execution_context_var.set(ec)
        try:
            dag_ctx.a = 99
            assert ec._params["a"] == 99
        finally:
            _dag_execution_context_var.reset(token)

    def test_context_proxy_set_none_deletes_attr_ref(self):
        ec = DagExecutionContext({})
        ec._attr_refs["to_delete"] = {"eTAG": "old"}
        token = _dag_execution_context_var.set(ec)
        try:
            dag_ctx.to_delete = None
            assert "to_delete" not in ec._attr_refs
        finally:
            _dag_execution_context_var.reset(token)

    def test_context_proxy_set_none_no_existing_attr_ref_is_noop(self):
        """Covers the pop(name, None) no-op branch of _ContextProxy.__setattr__.

        When value is None and the attr is not tracked in _attr_refs,
        the pop returns the default (None) without raising.
        """
        ec = DagExecutionContext({})
        token = _dag_execution_context_var.set(ec)
        try:
            dag_ctx.never_set = None
            assert ec._params["never_set"] is None
            assert "never_set" not in ec._attr_refs
        finally:
            _dag_execution_context_var.reset(token)


# ══════════════════════════════════════════════════════════════════════════════
#  StreamToDb
# ══════════════════════════════════════════════════════════════════════════════


class TestStreamToDb:
    def _stream(self, task_id=1, is_err=False):
        orig = MagicMock()
        orig.write = MagicMock(return_value=None)
        orig.flush = MagicMock()
        orig.isatty = MagicMock(return_value=False)
        orig.fileno = MagicMock(return_value=3)
        return StreamToDb(orig, task_id, is_err), orig

    def test_write_returns_byte_count(self):
        s, orig = self._stream()
        assert s.write("hello") == 5
        orig.write.assert_called_once_with("hello")

    def test_write_empty_string_returns_none(self):
        s, _ = self._stream()
        assert s.write("") is None

    def test_flush_clears_line_buffer(self):
        s, orig = self._stream()
        s.write("buffered")
        s.flush()
        assert s.line_buffer == ""
        orig.flush.assert_called()

    def test_flush_empty_buffer_still_calls_original_flush(self):
        s, orig = self._stream()
        s.flush()
        orig.flush.assert_called()

    def test_isatty_delegates_to_original(self):
        s, orig = self._stream()
        orig.isatty.return_value = True
        assert s.isatty() is True

    def test_fileno_delegates_to_original(self):
        s, _ = self._stream()
        assert s.fileno() == 3

    def test_getattr_falls_through_to_original(self):
        s, orig = self._stream()
        orig.custom_attr = "delegated"
        assert s.custom_attr == "delegated"

    def test_close_does_not_close_original_stream(self):
        s, orig = self._stream()
        orig.close = MagicMock()
        s.close()
        orig.close.assert_not_called()
        # Explicitly call __del__ to ensure coverage of the GC finalizer.
        s.__del__()
        del s
        gc.collect()

    def test_is_err_flag_stored(self):
        s, _ = self._stream(is_err=True)
        assert s.is_err is True

    def test_task_id_stored(self):
        s, _ = self._stream(task_id=42)
        assert s.task_id == 42

    def test_successive_writes_accumulate_in_buffer(self):
        s, _ = self._stream()
        s.write("part1")
        s.write("part2")
        assert s.line_buffer == "part1part2"


# ══════════════════════════════════════════════════════════════════════════════
#  TaskType – construction, properties, operators
# ══════════════════════════════════════════════════════════════════════════════


class TestTaskTypeConstruction:
    def test_name_property_equals_function_name(self):
        t = _real_task("my_task")
        assert t.name == "my_task"

    def test_is_parallel_defaults_to_false(self):
        assert not _real_task("t").is_parallel

    def test_parallel_flag_set(self):
        t = _real_task("pt", is_parallel=True)
        assert t.is_parallel

    def test_log_property_returns_logger(self):
        assert isinstance(_real_task("t").log, logging.Logger)

    def test_parents_empty_on_new_task(self):
        assert _real_task("t").parents == []

    def test_children_empty_on_new_task(self):
        assert _real_task("t").children == []

    def test_task_group_none_on_new_task(self):
        assert _real_task("t").task_group is None

    def test_hash_is_stable(self):
        t = _real_task("t")
        assert hash(t) == hash(t)

    def test_eq_same_instance(self):
        t = _real_task("t")
        assert t == t

    def test_eq_different_instances_not_equal(self):
        assert _real_task("t1") != _real_task("t2")

    def test_eq_non_task_type_false(self):
        t = _real_task("t")
        assert (t == "string") is False

    def test_str_contains_name(self):
        t = _real_task("named_task")
        assert "named_task" in str(t)

    def test_str_contains_merge_func_name_when_set(self):
        def mfunc(x):
            pass

        t = _real_task("t", merge_func=mfunc)
        assert "mfunc" in str(t)

    def test_str_merge_func_none_shows_none(self):
        assert "None" in str(_real_task("t"))

    def test_repr_equals_str(self):
        t = _real_task("t")
        assert repr(t) == str(t)

    def test_rshift_wires_parent_and_child(self):
        a, b = _real_task("a"), _real_task("b")
        result = a >> b
        assert result is b
        assert b in a.children
        assert a in b.parents

    def test_rshift_chain_three(self):
        a, b, c = _real_task("a2"), _real_task("b2"), _real_task("c2")
        a >> b >> c
        assert b in a.children
        assert c in b.children
        assert a in b.parents
        assert b in c.parents

    def test_rshift_to_task_group_wires_all_elements(self):
        a = _real_task("src")
        b, c = _real_task("b3"), _real_task("c3")
        tg = TaskGroup("TG", b, c)
        result = a >> tg
        assert result is tg
        assert b in a.children
        assert c in a.children

    def test_rshift_to_task_group_with_nested_group_element(self):
        """_add_child when child TaskGroup contains a TaskGroup element."""
        parent = _real_task("parent")
        inner = _real_task("inner")
        tg = TaskGroup("TG2", inner)
        parent >> tg
        assert inner in parent.children

    def test_rshift_invalid_type_raises_type_error(self):
        with pytest.raises(TypeError):
            _real_task("t3") >> "not_a_task"

    def test_add_child_task_group_child_recursively_adds_elements(self):
        """TaskType._add_child with TaskGroup child."""
        parent = _real_task("parent_ac")
        child_a = _real_task("child_a")
        child_b = _real_task("child_b")
        tg_child = TaskGroup("TGChild", child_a, child_b)
        parent._add_child(tg_child)
        assert child_a in parent.children
        assert child_b in parent.children

    def test_reduce_returns_reconstruct_callable(self):
        t = _real_task("r")
        fn, args = t.__reduce__()
        assert callable(fn)
        assert isinstance(args, tuple) and len(args) == 2

    def test_reconstruct_task_by_name_returns_task_type(self):
        """_reconstruct_task_by_name happy path."""
        # Register a fake module with a TaskType attribute
        mod = types.ModuleType("_test_recon_mod")
        t = _real_task("recon_task")
        mod.recon_task = t
        sys.modules["_test_recon_mod"] = mod
        try:
            result = TaskType._reconstruct_task_by_name("recon_task", "_test_recon_mod")
            assert result is t
        finally:
            del sys.modules["_test_recon_mod"]

    def test_reconstruct_task_by_name_raises_on_non_task_type(self):
        """_reconstruct_task_by_name raises when attr is not a TaskType."""
        mod = types.ModuleType("_test_recon_bad_mod")
        mod.not_a_task = "just a string"
        sys.modules["_test_recon_bad_mod"] = mod
        try:
            with pytest.raises(ValueError, match="Expected"):
                TaskType._reconstruct_task_by_name("not_a_task", "_test_recon_bad_mod")
        finally:
            del sys.modules["_test_recon_bad_mod"]


# ══════════════════════════════════════════════════════════════════════════════
#  TaskType – _wrap_func and _wrap_merge_func via real DAO + real stores
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture()
def _dao_and_registry(isolated_db, isolated_dao, assets_cache):
    """Provide a clean per-test DB, a real DAO for assertions, and a real assets cache.

    The wrapped funcs in core.py create their own internal DAO via
    Config.get_metadatastore_url() ; the DAO yielded here is used only for
    post-call state assertions on the real DB.
    """
    yield isolated_dao, None


class TestTaskTypeWrapFunc:
    def _seed_execution_and_tasktype(self, dao, tasktype_uuid):
        """Seed a real Execution row and a matching TaskType row.

        Required because tasks.exec_id+tasktype_uuid is a composite FK into
        tasktypes ; the wrapped funcs in core.py call dao.add_task with
        self.tasktype_uuid, so the corresponding tasktypes row must already
        exist or the FK constraint fires.
        """
        exec_id = dao.add_execution(
            name="test_pipe",
            username="tester",
            _start_timestamp=datetime.now(timezone.utc),
            metadata_root="/tmp/meta",
            artifacts_store_root="/tmp/artifacts",
        )
        dao.add_tasktype(
            exec_id=exec_id,
            order=0,
            uuid=tasktype_uuid,
            name="seeded_tasktype",
            is_parallel=False,
            children=[],
        )
        return exec_id

    def test_wrap_func_success_returns_task_id_and_result(self, _dao_and_registry):
        """Verify successful task execution returns correct IDs and persists rows.

        Drives the real-DB success path of _wrap_func: add_task, _task_registry
        register/unregister, add_task_context_attrs (context diff non-empty),
        set_task_exit_payload, update_task(failed=False), dispose.
        """
        dao, _ = _dao_and_registry
        ec = DagExecutionContext({"exec_id": None, "shared_val": 0})
        token = _dag_execution_context_var.set(ec)
        try:

            def my_func():
                dag_ctx.shared_val = 1  # generates a context diff
                return 42

            t = TaskType(func=my_func, is_parallel=False)
            exec_id = self._seed_execution_and_tasktype(dao, t.tasktype_uuid)
            ec._params["exec_id"] = exec_id
            task_id, result = t.func(exec_id=exec_id)
            assert result == 42
            # Real Task row exists, marked not failed.
            task_row = dao.get_task(task_id)
            assert task_row is not None
            assert task_row.failed is False
            # Real TaskContextAttr row written for the changed attr.
            attrs = dao._get_entities(TaskContextAttr, task_id=task_id)
            assert any(a.attr_name == "shared_val" for a in attrs)
            # Real TaskPayloadAttr row written (42 is JSON-safe => inline_val).
            payload = dao._get_entities(TaskPayloadAttr, task_id=task_id)
            assert len(payload) == 1
            assert payload[0].inline_val == 42
        finally:
            _dag_execution_context_var.reset(token)

    def test_wrap_func_with_docstring(self, _dao_and_registry, monkeypatch):
        """Verifies the docstring formatting branch (func.__doc__ truthy) AND
        the except branch around compute_context_diff in _wrap_func's finally block.

        Drives the truthy branch of `if func.__doc__` (green docstring block in
        the framed log output). Then forces compute_context_diff to raise in
        its cloudpickle branch by mocking write_binary_file at its import site
        in contexts_store (the `from ...utils.file_utils import write_binary_file`
        binds the name into contexts_store's namespace, so the patch must target
        contexts_store, not file_utils). The mocked raise propagates up through
        compute_context_diff and is caught by the real except Exception handler
        in _wrap_func. The task still completes successfully (failed=False).

        The task sets a non-JSON-safe context attr so the cloudpickle branch of
        _serialize_attr (inside compute_context_diff) is the one that calls
        write_binary_file and triggers the mocked raise.

        Note: write_binary_file mocking is explicitly user-authorized for this
        test only, as the only honest way to force a disk-write failure on
        backends (local + S3) without depending on filesystem-specific quirks.
        """
        dao, _ = _dao_and_registry
        # Mock write_binary_file at its bound name in contexts_store.
        # _serialize_attr (called from compute_context_diff) calls this bound
        # name when serializing a non-JSON-safe context attr via cloudpickle.
        monkeypatch.setattr(
            contexts_store_module,
            "write_binary_file",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("disk write blocked")),
        )
        ec = DagExecutionContext({"exec_id": None})
        token = _dag_execution_context_var.set(ec)
        try:

            class _NonJsonSafe:
                pass

            def my_func():
                """My doc."""
                dag_ctx.non_json_safe = _NonJsonSafe()  # forces cloudpickle branch
                return "ok"

            t = TaskType(func=my_func, is_parallel=False)
            exec_id = self._seed_execution_and_tasktype(dao, t.tasktype_uuid)
            ec._params["exec_id"] = exec_id
            _, result = t.func(exec_id=exec_id)
            assert result == "ok"
            # Task still marked not failed despite the swallowed context-diff
            # serialization error ; the except branch in _wrap_func ran and
            # logged the error via logger.exception.
            tasks = dao.get_tasks_by_execution(exec_id)
            assert len(tasks) == 1
            assert tasks[0].failed is False
        finally:
            _dag_execution_context_var.reset(token)

    def test_wrap_func_payload_serialize_exception(
        self, _dao_and_registry, monkeypatch
    ):
        """Verify the except branch around set_task_exit_payload in _wrap_func.

        Forces serialize_task_payload to raise in its cloudpickle branch by
        mocking write_binary_file at its import site in payloads_store (same
        import-binding gotcha as contexts_store). The mocked raise propagates
        up through serialize_task_payload and is caught by the real
        except Exception handler in _wrap_func. The task still completes
        successfully (failed=False) and no payload row is written.

        The task returns a non-JSON-safe value so serialize_task_payload takes
        the cloudpickle branch (the only branch that calls write_binary_file).

        Note: write_binary_file mocking is explicitly user-authorized for this
        test only.
        """
        dao, _ = _dao_and_registry
        # Mock write_binary_file at its bound name in payloads_store.
        # serialize_task_payload's cloudpickle branch calls write_binary_file(
        # metadata_root(), [rel_path], raw_bytes) ; the mocked raise propagates
        # up and is caught by _wrap_func's except Exception handler around
        # set_task_exit_payload.
        monkeypatch.setattr(
            payloads_store_module,
            "write_binary_file",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("disk write blocked")),
        )
        ec = DagExecutionContext({"exec_id": None})
        token = _dag_execution_context_var.set(ec)
        try:

            class _NonJsonSafe:
                pass

            def my_func():
                return (
                    _NonJsonSafe()
                )  # forces cloudpickle branch in serialize_task_payload

            t = TaskType(func=my_func, is_parallel=False)
            exec_id = self._seed_execution_and_tasktype(dao, t.tasktype_uuid)
            ec._params["exec_id"] = exec_id
            task_id, result = t.func(exec_id=exec_id)
            assert result is not None
            # Task still marked not failed despite the swallowed payload
            # serialization error ; the except branch in _wrap_func ran and
            # logged the error via logger.exception.
            task_row = dao.get_task(task_id)
            assert task_row is not None
            assert task_row.failed is False
            # No payload row written: serialize_task_payload raised before
            # returning the row dict, so set_task_exit_payload was never
            # called with a valid row.
            payloads = dao._get_entities(TaskPayloadAttr, task_id=task_id)
            assert len(payloads) == 0
        finally:
            _dag_execution_context_var.reset(token)

    def test_wrap_func_failure_raises_task_func_exception(self, _dao_and_registry):
        """Task body raises ; _wrap_func wraps it in TaskFuncException."""
        dao, _ = _dao_and_registry
        ec = DagExecutionContext({"exec_id": None})
        token = _dag_execution_context_var.set(ec)
        try:

            def bad_func():
                raise ValueError("boom")

            t = TaskType(func=bad_func, is_parallel=False)
            exec_id = self._seed_execution_and_tasktype(dao, t.tasktype_uuid)
            ec._params["exec_id"] = exec_id
            with pytest.raises(TaskFuncException):
                t.func(exec_id=exec_id)
        finally:
            _dag_execution_context_var.reset(token)

    def test_wrap_func_marks_task_failed_on_exception(self, _dao_and_registry):
        """Task body raises ; _wrap_func finally-block calls update_task(failed=True)."""
        dao, _ = _dao_and_registry
        ec = DagExecutionContext({"exec_id": None})
        token = _dag_execution_context_var.set(ec)
        try:

            def bad_func():
                raise RuntimeError("oops")

            t = TaskType(func=bad_func, is_parallel=False)
            exec_id = self._seed_execution_and_tasktype(dao, t.tasktype_uuid)
            ec._params["exec_id"] = exec_id
            with pytest.raises(TaskFuncException):
                t.func(exec_id=exec_id)
            # Real Task row exists, marked failed.
            tasks = dao.get_tasks_by_execution(exec_id)
            assert len(tasks) == 1
            assert tasks[0].failed is True
        finally:
            _dag_execution_context_var.reset(token)

    def test_wrap_func_with_rank_in_signature(self, _dao_and_registry):
        """'rank' in func signature => kept in kwargs, not popped."""
        dao, _ = _dao_and_registry
        ec = DagExecutionContext({"exec_id": None})
        token = _dag_execution_context_var.set(ec)
        try:

            def func_with_rank(rank=None):
                return rank

            t = TaskType(func=func_with_rank, is_parallel=False)
            exec_id = self._seed_execution_and_tasktype(dao, t.tasktype_uuid)
            ec._params["exec_id"] = exec_id
            _, result = t.func(exec_id=exec_id, rank=7)
            assert result == 7
        finally:
            _dag_execution_context_var.reset(token)

    def test_wrap_func_with_task_id_in_signature_injects_id(self, _dao_and_registry):
        """'task_id' in func signature => DAO creates it, wrapper injects it."""
        dao, _ = _dao_and_registry
        ec = DagExecutionContext({"exec_id": None})
        token = _dag_execution_context_var.set(ec)
        try:

            def a_tasktype_func(task_id=None):
                return task_id

            t = TaskType(func=a_tasktype_func, is_parallel=False)
            exec_id = self._seed_execution_and_tasktype(dao, t.tasktype_uuid)
            ec._params["exec_id"] = exec_id
            returned_task_id, task_id = t.func(exec_id=exec_id, task_id=None)
            # id matches the DAO-returned id (real DB row's id).
            assert task_id == returned_task_id
            assert dao.get_task(task_id).id == task_id
        finally:
            _dag_execution_context_var.reset(token)

    def test_wrap_func_with_provided_task_id_skips_dao_add(self, _dao_and_registry):
        """Pre-supplied task_id (from merge wrapper) => DAO.add_task NOT called."""
        dao, _ = _dao_and_registry
        ec = DagExecutionContext({"exec_id": None})
        token = _dag_execution_context_var.set(ec)
        try:

            def plain():
                return "done"

            t = TaskType(func=plain, is_parallel=False)
            exec_id = self._seed_execution_and_tasktype(dao, t.tasktype_uuid)
            ec._params["exec_id"] = exec_id
            # Pre-seed a Task row with the task_id we'll pass.
            pre_task_id = dao.add_task(
                exec_id=exec_id,
                tasktype_uuid=t.tasktype_uuid,
                rank=None,
                _start_timestamp=datetime.now(timezone.utc),
            )
            t.func(exec_id=exec_id, task_id=pre_task_id)
            # Only one Task row exists (the pre-seeded one) ; no new row added.
            tasks = dao.get_tasks_by_execution(exec_id)
            assert len(tasks) == 1
            assert tasks[0].id == pre_task_id
        finally:
            _dag_execution_context_var.reset(token)

    def test_wrap_merge_func_success(self, _dao_and_registry):
        dao, _ = _dao_and_registry
        ec = DagExecutionContext({"exec_id": None})
        token = _dag_execution_context_var.set(ec)
        try:

            def merge(x):
                return x * 2

            def main_func():
                return 1

            t = TaskType(func=main_func, is_parallel=False, merge_func=merge)
            exec_id = self._seed_execution_and_tasktype(dao, t.tasktype_uuid)
            ec._params["exec_id"] = exec_id
            task_id, result = t.merge_func(10, exec_id=exec_id, rank=None)
            assert result == 20
            # _wrap_merge_func success path does NOT call update_task(failed=...)
            # (only the except branch does). The Task row exists ; we assert
            # the row was created.
            assert dao.get_task(task_id) is not None
        finally:
            _dag_execution_context_var.reset(token)

    def test_wrap_merge_func_failure_raises_merge_exception(self, _dao_and_registry):
        dao, _ = _dao_and_registry
        ec = DagExecutionContext({"exec_id": None})
        token = _dag_execution_context_var.set(ec)
        try:

            def bad_merge(x):
                raise ValueError("merge fail")

            def main_func():
                return 1

            t = TaskType(func=main_func, is_parallel=False, merge_func=bad_merge)
            exec_id = self._seed_execution_and_tasktype(dao, t.tasktype_uuid)
            ec._params["exec_id"] = exec_id
            with pytest.raises(TaskMergeFuncException):
                t.merge_func(0, exec_id=exec_id, rank=None)
        finally:
            _dag_execution_context_var.reset(token)

    def test_wrap_merge_func_marks_task_failed(self, _dao_and_registry):
        dao, _ = _dao_and_registry
        ec = DagExecutionContext({"exec_id": None})
        token = _dag_execution_context_var.set(ec)
        try:

            def bad_merge(x):
                raise RuntimeError("m")

            def main_func():
                return 1

            t = TaskType(func=main_func, is_parallel=False, merge_func=bad_merge)
            exec_id = self._seed_execution_and_tasktype(dao, t.tasktype_uuid)
            ec._params["exec_id"] = exec_id
            with pytest.raises(TaskMergeFuncException):
                t.merge_func(0, exec_id=exec_id, rank=None)
            # Real Task row exists, marked failed.
            tasks = dao.get_tasks_by_execution(exec_id)
            assert len(tasks) == 1
            assert tasks[0].failed is True
        finally:
            _dag_execution_context_var.reset(token)

    def test_wrap_merge_then_func_propagates_pending_entry_etags(
        self, _dao_and_registry
    ):
        """merge_func stashes its entry-etags snapshot on the active context
        so the subsequent func wrapper reuses it instead of re-snapshotting.

        Exercises the two branches that work as a logical pair:
          * the merge-func finally-block branch that propagates entry etags
            onto a non-None active context for the next wrapper to detect.
          * the func-entry branch that detects a stashed pending-etags value
            and consumes (deletes) it rather than taking a fresh snapshot.
        """
        dao, _ = _dao_and_registry
        ec = DagExecutionContext({"exec_id": None})
        token = _dag_execution_context_var.set(ec)
        try:

            def merge(x):
                return x * 2

            def main_func():
                return "main"

            t = TaskType(func=main_func, is_parallel=False, merge_func=merge)
            exec_id = self._seed_execution_and_tasktype(dao, t.tasktype_uuid)
            ec._params["exec_id"] = exec_id

            merge_task_id, merge_result = t.merge_func(5, exec_id=exec_id, rank=None)
            assert merge_result == 10
            # merge_func finally-block propagated entry etags onto the context.
            assert vars(ec).get("_pending_merge_entry_etags") == {}

            func_task_id, result = t.func(exec_id=exec_id, task_id=merge_task_id)
            assert result == "main"
            # func-entry branch consumed and deleted the stash.
            assert "_pending_merge_entry_etags" not in vars(ec)
        finally:
            _dag_execution_context_var.reset(token)


# ══════════════════════════════════════════════════════════════════════════════
#  _capture_and_stream_trace – notebook-mode and pipe_reader branches
# ══════════════════════════════════════════════════════════════════════════════


class TestCaptureAndStreamTrace:
    def _seed_execution_and_tasktype(self, dao, tasktype_uuid):
        """Seed a real Execution row and a matching TaskType row.

        Required because tasks.exec_id+tasktype_uuid is a composite FK into
        tasktypes ; the wrapped funcs in core.py call dao.add_task with
        self.tasktype_uuid, so the corresponding tasktypes row must already
        exist or the FK constraint fires.
        """
        exec_id = dao.add_execution(
            name="test_pipe",
            username="tester",
            _start_timestamp=datetime.now(timezone.utc),
            metadata_root="/tmp/meta",
            artifacts_store_root="/tmp/artifacts",
        )
        dao.add_tasktype(
            exec_id=exec_id,
            order=0,
            uuid=tasktype_uuid,
            name="seeded_tasktype",
            is_parallel=False,
            children=[],
        )
        return exec_id

    def test_ipython_import_error_sets_non_notebook_mode(self, _dao_and_registry):
        """exception during get_ipython() => _notebook_mode = False.

        The task still completes and traces are written via the non-notebook
        pipe_reader path. We write directly to fd 1 (bypassing sys.stdout)
        because pipe_reader reads from the OS-level pipe that fd 1 is dup'd to,
        not from sys.stdout.
        """
        dao, _ = _dao_and_registry
        ec = DagExecutionContext({"exec_id": None})
        token = _dag_execution_context_var.set(ec)
        try:

            def my_func():
                os.write(1, b"hello from task\n")
                return "ok"

            # Simulate IPython import raising an exception
            with patch.dict("sys.modules", {"IPython": None}):
                t = TaskType(func=my_func, is_parallel=False)
                exec_id = self._seed_execution_and_tasktype(dao, t.tasktype_uuid)
                ec._params["exec_id"] = exec_id
                _, result = t.func(exec_id=exec_id)
            assert result == "ok"
            # Flush the global trace buffer to ensure the async writer thread
            # has persisted all queued traces before we query.
            trace_buffer_module.get_trace_buffer().flush()
            # Real TaskTrace rows written via the non-notebook pipe_reader path.
            task_id = dao.get_tasks_by_execution(exec_id)[0].id
            traces = dao._get_entities(TaskTrace, task_id=task_id)
            assert len(traces) > 0
        finally:
            _dag_execution_context_var.reset(token)

    def test_notebook_mode_patches_write_methods(self, _dao_and_registry):
        """notebook mode wraps capturer_out/err.write with _nb_write.

        The fake IPython module makes the local `from IPython import get_ipython`
        inside _capture_and_stream_trace return a kernel-configured shell,
        flipping _notebook_mode = True. The _OnceRecursiveStdout stand-in
        drives the inner (recursive) branch of _nb_write.
        """
        dao, _ = _dao_and_registry
        ec = DagExecutionContext({"exec_id": None})
        token = _dag_execution_context_var.set(ec)
        try:

            def my_func():
                print("notebook output")
                return "nb_ok"

            ipython_mock = MagicMock()
            ipython_mock.config = {"IPKernelApp": True}

            # Injecting a mock IPython module into sys.modules ensures the local
            # import inside _capture_and_stream_trace succeeds and triggers the
            # notebook-specific stream patching logic.
            fake_ipython_module = types.ModuleType("IPython")
            fake_ipython_module.get_ipython = lambda: ipython_mock

            # _OnceRecursiveStdout is used as sys.stdout so that the first write
            # call coming from StreamToDb.original_stream.write() re-enters
            # capturer_out.write (= _nb_write) with is_top=False, covering
            # the inner/recursive branch of _nb_write. _done guards
            # against unbounded recursion: only one recursive call is triggered.
            class _OnceRecursiveStdout:
                def __init__(self):
                    self._done = False

                def write(self, data):
                    if not self._done:
                        self._done = True
                        sys.stdout.write(data)  # re-enters _nb_write with is_top=False

                def flush(self):
                    pass

                def fileno(self):
                    return sys.__stdout__.fileno()

                def isatty(self):
                    return False

            t = TaskType(func=my_func, is_parallel=False)
            exec_id = self._seed_execution_and_tasktype(dao, t.tasktype_uuid)
            ec._params["exec_id"] = exec_id
            recursive_stdout = _OnceRecursiveStdout()
            orig_stdout = sys.stdout
            try:
                sys.stdout = recursive_stdout
                with patch.dict(sys.modules, {"IPython": fake_ipython_module}):
                    _, result = t.func(exec_id=exec_id)
            finally:
                sys.stdout = orig_stdout
            assert result == "nb_ok"
        finally:
            _dag_execution_context_var.reset(token)

    def test_pipe_reader_processes_newline_delimited_output(
        self, _dao_and_registry, monkeypatch
    ):
        """pipe_reader splits on newlines and add_trace raises ;
        the exception is silently swallowed and the task still returns normally.

        Drives the except Exception: pass swallow branch in pipe_reader via a
        real TraceBuffer subclass whose add_trace raises. The subclass is
        swapped in by monkeypatching the trace_buffer module's global singleton.
        """

        class _RaisingTraceBuffer(trace_buffer_module.TraceBuffer):
            def add_trace(self, *args, **kwargs):
                raise RuntimeError("trace swallowed")

        raising_buffer = _RaisingTraceBuffer()
        monkeypatch.setattr(trace_buffer_module, "_trace_buffer", raising_buffer)

        dao, _ = _dao_and_registry
        ec = DagExecutionContext({"exec_id": None})
        token = _dag_execution_context_var.set(ec)
        try:

            def my_func():
                # Write multiple lines to trigger pipe_reader line-splitting logic.
                # Writing directly to file descriptor 1 bypasses the patched sys.stdout
                # and feeds data directly into the OS-level pipe.
                os.write(1, b"line1\nline2\n")
                return "multiline"

            t = TaskType(func=my_func, is_parallel=False)
            exec_id = self._seed_execution_and_tasktype(dao, t.tasktype_uuid)
            ec._params["exec_id"] = exec_id
            _, result = t.func(exec_id=exec_id)
            assert result == "multiline"
        finally:
            _dag_execution_context_var.reset(token)

    def test_pipe_reader_processes_newline_delimited_output_success(
        self, _dao_and_registry
    ):
        """pipe_reader splits on newlines and adds traces to buffer without raising.

        Covers the non-notebook console echo branch immediately following add_trace.
        """
        dao, _ = _dao_and_registry
        ec = DagExecutionContext({"exec_id": None})
        token = _dag_execution_context_var.set(ec)
        try:

            def my_func():
                # Write multiple lines to trigger pipe_reader line-splitting logic.
                os.write(1, b"line1\nline2\n")
                return "multiline_ok"

            t = TaskType(func=my_func, is_parallel=False)
            exec_id = self._seed_execution_and_tasktype(dao, t.tasktype_uuid)
            ec._params["exec_id"] = exec_id
            _, result = t.func(exec_id=exec_id)
            assert result == "multiline_ok"
            # Flush the global trace buffer to ensure the async writer thread
            # has persisted all queued traces before we query.
            trace_buffer_module.get_trace_buffer().flush()
            # Real TaskTrace rows written for each line.
            task_id = dao.get_tasks_by_execution(exec_id)[0].id
            traces = dao._get_entities(TaskTrace, task_id=task_id)
            assert len(traces) >= 2
        finally:
            _dag_execution_context_var.reset(token)

    def test_pipe_reader_flushes_partial_line_at_eof(self, _dao_and_registry):
        """pipe_reader flushes remaining buffer (no trailing newline)."""
        dao, _ = _dao_and_registry
        ec = DagExecutionContext({"exec_id": None})
        token = _dag_execution_context_var.set(ec)
        try:

            def my_func():
                # Write without trailing newline => pipe_reader sees partial buffer at close.
                # Writing directly to file descriptor 1 bypasses the patched sys.stdout
                # and feeds data directly into the OS-level pipe.
                os.write(1, b"no newline at end")
                return "partial"

            t = TaskType(func=my_func, is_parallel=False)
            exec_id = self._seed_execution_and_tasktype(dao, t.tasktype_uuid)
            ec._params["exec_id"] = exec_id
            _, result = t.func(exec_id=exec_id)
            assert result == "partial"
            # Flush the global trace buffer to ensure the async writer thread
            # has persisted all queued traces before we query.
            trace_buffer_module.get_trace_buffer().flush()
            # Real TaskTrace row written for the partial line.
            task_id = dao.get_tasks_by_execution(exec_id)[0].id
            traces = dao._get_entities(TaskTrace, task_id=task_id)
            assert len(traces) >= 1
        finally:
            _dag_execution_context_var.reset(token)


# ══════════════════════════════════════════════════════════════════════════════
#  TaskGroup
# ══════════════════════════════════════════════════════════════════════════════


class TestTaskGroup:
    def test_positional_name_and_elements(self):
        t1, t2 = _real_task("t1"), _real_task("t2")
        tg = TaskGroup("MyGroup", t1, t2)
        assert tg.name == "MyGroup"
        assert len(tg.elements) == 2

    def test_keyword_name(self):
        t = _real_task("t")
        tg = TaskGroup(t, name="KwGroup")
        assert tg.name == "KwGroup"

    def test_missing_name_raises_type_error(self):
        with pytest.raises(TypeError):
            TaskGroup()

    def test_non_string_name_raises_type_error(self):
        with pytest.raises(TypeError):
            TaskGroup(123, _real_task("t"))

    def test_empty_elements_raises_task_group_exception(self):
        """Covers the empty-elements guard branch of __init__.

        A TaskGroup declared with a name but no constituent elements
        (e.g. when @taskgroup returns None or an empty tuple) is
        rejected at construction time.
        """
        with pytest.raises(TaskGroupException, match="resolves to no TaskType"):
            TaskGroup("EmptyTG")

    def test_cycle_raises_task_group_exception(self):
        """Covers the cycle-detection branch of _has_cycle inside __init__.

        A TaskGroup that transitively contains itself (through a nested
        TaskGroup element whose elements loop back) is rejected at
        construction time to prevent infinite recursion during DAG
        traversal.
        """
        t = _real_task("cycle_task")
        inner = TaskGroup("CycleInner", t)
        outer = TaskGroup("CycleOuter", inner)
        # Forge a cycle: inner now references outer, creating
        # inner -> outer -> inner.
        inner.elements.append(outer)
        with pytest.raises(TaskGroupException, match="contains a cycle"):
            TaskGroup("CycleTrigger", inner)

    def test_pydantic_validation_failure_raises_task_group_exception(self):
        """pydantic validation error => TaskGroupException."""
        # Pass an invalid ui_css value to trigger pydantic validation inside super().__init__
        with pytest.raises(TaskGroupException, match="Error instanciating"):
            TaskGroup("BadCss", _real_task("t"), ui_css="not_a_ui_css_instance")

    def test_each_uuid_is_unique(self):
        t = _real_task("t")
        tg1 = TaskGroup("G1", t)
        tg2 = TaskGroup("G2", t)
        assert tg1.uuid != tg2.uuid
        assert hash(tg1) != hash(tg2)

    def test_equality_is_by_uuid(self):
        t = _real_task("t")
        tg = TaskGroup("G", t)
        assert tg == tg
        assert tg != TaskGroup("H", t)

    def test_eq_non_task_group_false(self):
        t = _real_task("t")
        assert TaskGroup("G", t) != "string"

    def test_str_contains_group_name(self):
        t = _real_task("t")
        assert "MyTG" in str(TaskGroup("MyTG", t))

    def test_repr_contains_TaskGroup(self):
        t = _real_task("t")
        assert "TaskGroup" in repr(TaskGroup("G", t))

    def test_elements_get_task_group_reference(self):
        t = _real_task("t")
        tg = TaskGroup("G", t)
        assert t._task_group is tg

    def test_log_property_returns_logger(self):
        t = _real_task("t")
        assert isinstance(TaskGroup("G", t).log, logging.Logger)

    def test_task_group_property_none_at_top_level(self):
        t = _real_task("t")
        assert TaskGroup("G", t).task_group is None

    def test_parents_property_empty(self):
        t = _real_task("t")
        assert TaskGroup("G", t).parents == []

    def test_taskgroup_name_positional_with_docstring_and_ui_css(self):
        """Combined positional name + docstring + ui_css keyword construction.

        Covers the full path through __init__ where name is positional and
        docstring/ui_css arrive as keyword arguments alongside the elements.
        """
        t = _real_task("t")
        css = UiCss(background="#aabbcc")
        tg = TaskGroup("FullTG", t, docstring="A group", ui_css=css)
        assert tg.name == "FullTG"
        assert tg.docstring == "A group"
        assert tg.ui_css is css

    def test_rshift_to_task_type_wires_elements(self):
        a, b = _real_task("tg_a"), _real_task("tg_b")
        c = _real_task("tg_c")
        tg = TaskGroup("AB", a, b)
        result = tg >> c
        assert result is c
        assert a in c.parents
        assert b in c.parents

    def test_rshift_to_task_type_parallel_raises(self):
        a = _real_task("a")
        tg = TaskGroup("G", a)
        p = _real_task("p", is_parallel=True)
        with pytest.raises(DistributionNotSupportedError):
            tg >> p

    def test_rshift_to_merging_task_raises(self):
        def mf(x):
            pass

        a = _real_task("a")
        tg = TaskGroup("G", a)
        m = _real_task("m", merge_func=mf)
        with pytest.raises(MergeNotSupportedError):
            tg >> m

    def test_rshift_to_task_group_wires_cross(self):
        a = _real_task("tg2_a")
        b = _real_task("tg2_b")
        tg1 = TaskGroup("TG1", a)
        tg2 = TaskGroup("TG2", b)
        result = tg1 >> tg2
        assert result is tg2
        assert b in a.children

    def test_rshift_invalid_type_raises(self):
        t = _real_task("t")
        tg = TaskGroup("G", t)
        with pytest.raises(TypeError):
            tg >> 42

    def test_add_child_nested_task_group_element_recurses(self):
        """TaskGroup._add_child: outer element is TaskGroup => recurse into it."""
        inner_a = _real_task("ia")
        inner_tg = TaskGroup("InnerTG", inner_a)
        outer_tg = TaskGroup("OuterTG", inner_tg)
        target = _real_task("target")
        outer_tg >> target
        assert inner_a in target.parents

    def test_add_child_with_task_group_child_wires_sub_children(self):
        """_add_child with a TaskGroup child ; element is TaskType."""
        elem_a = _real_task("tg_elem_a")
        elem_b = _real_task("tg_elem_b")
        tg_parent = TaskGroup("TGParent", elem_a)
        tg_child = TaskGroup("TGChild", elem_b)
        tg_parent._add_child(tg_child)
        assert elem_b in elem_a.children
        assert elem_a in elem_b.parents

    def test_add_child_with_task_group_child_and_group_element_recurses(self):
        """_add_child with TaskGroup child ; outer element is also TaskGroup."""
        inner_elem = _real_task("inner_elem")
        inner_tg = TaskGroup("InnerTG2", inner_elem)
        outer_tg = TaskGroup("OuterTG2", inner_tg)
        child_elem = _real_task("child_elem")
        tg_child = TaskGroup("ChildTG", child_elem)
        outer_tg._add_child(tg_child)
        assert child_elem in inner_elem.children


# ══════════════════════════════════════════════════════════════════════════════
#  DAG._find_root_tasks
# ══════════════════════════════════════════════════════════════════════════════


class TestDagFindRootTasks:
    def test_single_root_upstream_of_leaf(self):
        a = _real_task("A")
        b = _real_task("B")
        a >> b
        assert DAG._find_root_tasks(b) == [a]

    def test_task_with_no_parents_is_its_own_root(self):
        a = _real_task("A")
        assert DAG._find_root_tasks(a) == [a]

    def test_long_chain_finds_head(self):
        a = _real_task("A")
        b = _real_task("B")
        c = _real_task("C")
        d = _real_task("D")
        a >> b >> c >> d
        assert DAG._find_root_tasks(d) == [a]

    def test_multiple_roots(self):
        a = _real_task("A")
        b = _real_task("B")
        c = _real_task("C")
        a >> c
        b >> c
        roots = DAG._find_root_tasks(c)
        assert set(roots) == {a, b}

    def test_task_group_anchor_traverses_elements(self):
        """When anchor is a TaskGroup, roots are derived from elements' parents.

        Uses real TaskType instances so isinstance() checks inside
        _find_root_tasks work correctly (MagicMock __class__ override is
        insufficient for isinstance).
        """
        root = _real_task("Root")
        leaf = _real_task("Leaf")
        root >> leaf
        tg = TaskGroup("AnchorTG", leaf)
        roots = DAG._find_root_tasks(tg)
        assert root in roots

    def test_task_group_mid_traversal_extends_parents(self):
        """TaskGroup encountered mid-stack => its parents are pushed onto stack.

        Manually injecting a TaskGroup into a task's _parents list places it in
        the traversal stack, hitting the isinstance(current, TaskGroup) branch.
        """
        inner = _real_task("rt_inner")
        # A TaskGroup wrapping inner; root_tg.parents is empty.
        root_tg = TaskGroup("RootTG_traverse", inner)
        # leaf's only declared parent is root_tg (a TaskGroup).
        # Traversal: pop leaf (TaskType) -> stack.extend([root_tg])
        #            pop root_tg (TaskGroup) -> stack.extend(root_tg.parents=[])
        leaf = _real_task("rt_leaf_traverse")
        leaf._parents.append(root_tg)
        roots = DAG._find_root_tasks(leaf)
        # leaf has a parent so it is not a root; root_tg is never added to all_tasks.
        assert isinstance(roots, list)

    def test_drop_unanchored_taskgroups_cleans_phantom_reference(self):
        """Covers _drop_unanchored_taskgroups: phantom detection."""
        # Anchor task reachable from the DAG entry point.
        anchor = _real_task("phantom_anchor")

        # Intermediate TaskGroup wrapping the anchor. This creates the
        # two-level _task_group chain (anchor -> outer_tg -> phantom_tg)
        # that makes _chain yield twice when walking the anchor's chain.
        outer_tg = TaskGroup("OuterTG", anchor)

        # Inner TaskGroup with its own task element; its _task_group
        # will be reassigned to the phantom group below.
        inner_elem = _real_task("inner_elem")
        inner_tg = TaskGroup("InnerPhantomTG", inner_elem)

        # Task not reachable from DAG roots -- makes the phantom group
        # contain an element outside all_tasks.
        unreachable = _real_task("unreachable")

        # phantom_tg's __init__ reassigns _task_group on each of its
        # direct elements (outer_tg, inner_tg, unreachable) to itself.
        # anchor is nested inside outer_tg, not a direct element, so
        # anchor._task_group stays outer_tg.
        phantom_tg = TaskGroup("PhantomTG", outer_tg, inner_tg, unreachable)
        assert anchor._task_group is outer_tg
        assert outer_tg._task_group is phantom_tg
        assert inner_tg._task_group is phantom_tg

        # _find_root_tasks walks anchor._task_group up to outer_tg, then
        # outer_tg._task_group up to phantom_tg, detects that phantom_tg
        # contains unreachable (not in all_tasks), and drops it.
        #
        # The _chain generator walks anchor's chain (outer_tg then
        # phantom_tg) to compute which tasks reference the dropped group,
        # advancing past the first yield because outer_tg != phantom_tg.
        # _clean then nullifies _task_group on every direct element of
        # phantom_tg whose uuid matches, recursing into TaskGroup elements.
        roots = DAG._find_root_tasks(anchor)

        assert anchor in roots
        # _task_group refs cleaned by _clean on direct elements with uuid match:
        assert outer_tg._task_group is None
        assert inner_tg._task_group is None
        assert unreachable._task_group is None
        # _clean recursed into outer_tg.elements=[anchor]:
        #   anchor._task_group is outer_tg, uuid mismatch with phantom_tg -> skipped.
        assert anchor._task_group is outer_tg
        # _clean recursed into inner_tg.elements=[inner_elem]:
        #   inner_elem._task_group is inner_tg, uuid mismatch with phantom_tg -> skipped.
        assert inner_elem._task_group is inner_tg


# ══════════════════════════════════════════════════════════════════════════════
#  _get_dag_params
# ══════════════════════════════════════════════════════════════════════════════


class TestGetDagParams:
    def test_extracts_dag_params(self):
        def func_with_params():
            alpha = DagParam(description="alpha desc", default=1)  # noqa: F841
            beta = DagParam(description="beta desc")  # noqa: F841
            return None

        params = _get_dag_params(func_with_params)
        assert "alpha" in params
        assert params["alpha"].default == 1
        assert "beta" in params
        assert params["beta"].default is None

    def test_skips_non_dagparam_assignments(self):
        def func_mixed():
            x = 5  # noqa: F841
            p = DagParam(description="p")  # noqa: F841
            return None

        params = _get_dag_params(func_mixed)
        assert "p" in params
        assert "x" not in params

    def test_skips_non_assign_statements(self):
        """Statements that are not ast.Assign (e.g. if-statements) are silently skipped.

        Covers the branch where a statement in the function body is not an
        ast.Assign node and is therefore ignored by the DagParam-extraction pass.
        """

        def func_with_control_flow():
            if True:
                pass
            p = DagParam(description="still parsed")  # noqa: F841
            return None

        params = _get_dag_params(func_with_control_flow)
        assert "p" in params

    def test_no_func_node_raises_value_error(self):
        """no function definition found in AST => ValueError.

        Passing a module-level class (not a function) to _get_dag_params:
        inspect.getsource returns the class source, ast.walk yields ClassDef
        nodes (no FunctionDef), so func_node stays None and the raise fires.
        """
        with pytest.raises(ValueError, match="Could not find function definition"):
            _get_dag_params(_NoFuncDefClass)

    def test_stmt_exec_exception_is_silently_skipped(self):
        """statement that raises during exec => silently skipped, parsing continues."""

        def func_with_failing_stmt():
            result = undefined_name_xyz  # noqa: F841, F821 ; intentionally undefined
            p = DagParam(description="still parsed")  # noqa: F841
            return None

        # Should not raise; the undefined name statement fails silently,
        # and the DagParam assignment is still parsed.
        params = _get_dag_params(func_with_failing_stmt)
        assert "p" in params

    def test_literal_eval_failure_falls_back_to_eval_with_context(self):
        """literal_eval fails for non-literal => falls back to eval."""

        def func_with_computed_default():
            _computed = [1, 2, 3]
            p = DagParam(description="uses list", default=_computed)  # noqa: F841
            return None

        params = _get_dag_params(func_with_computed_default)
        assert "p" in params
        assert params["p"].default == [1, 2, 3]

    def test_both_literal_eval_and_eval_fail_skips_param(self):
        """both literal_eval and eval fail => keyword skipped, param has no default."""

        def func_with_unevaluable_default():
            p = DagParam(description="bad default", default=__totally_undefined_xyz__)  # noqa: F841, F821
            return None

        # param is still extracted but without the un-evaluable default keyword
        params = _get_dag_params(func_with_unevaluable_default)
        # DagParam is created with only the successfully-parsed keywords;
        # `default` was skipped => falls back to DagParam default of None.
        assert "p" in params
        assert params["p"].default is None


# ══════════════════════════════════════════════════════════════════════════════
#  @dag decorator
# ══════════════════════════════════════════════════════════════════════════════


class TestDagDecorator:
    def test_no_parens_returns_dag_instance(self):
        @task
        def t1():
            return 1

        @dag
        def my_pipeline():
            return t1

        assert isinstance(my_pipeline, DAG)
        assert len(my_pipeline.roots) == 1

    def test_with_parens_returns_dag_instance(self):
        @task
        def t2():
            return 2

        @dag()
        def pipeline2():
            return t2

        assert isinstance(pipeline2, DAG)

    def test_docstring_stored(self):
        @task
        def t3():
            return 3

        @dag
        def pipeline3():
            """My pipeline doc."""
            return t3

        assert pipeline3.docstring == "My pipeline doc."

    def test_ui_css_stored(self):
        @task
        def t4():
            return 4

        css = UiCss(background="#123456")

        @dag(ui_css=css)
        def pipeline4():
            return t4

        assert pipeline4.ui_css is css

    def test_dag_param_extracted(self):
        @task
        def t5():
            return 5

        @dag
        def pipeline5():
            my_param = DagParam(description="a param", default=7)  # noqa: F841
            return t5

        assert "my_param" in pipeline5.params
        assert pipeline5.params["my_param"].default == 7


# ══════════════════════════════════════════════════════════════════════════════
#  @task decorator – inner wrapper callable
# ══════════════════════════════════════════════════════════════════════════════


class TestTaskDecoratorWrapper:
    def test_task_no_parens_returns_task_type(self):
        @task
        def plain():
            pass

        assert isinstance(plain, TaskType)
        assert plain.is_parallel is False

    def test_task_with_parens_returns_task_type(self):
        @task()
        def plain2():
            pass

        assert isinstance(plain2, TaskType)

    def test_task_with_merge_func(self):
        def mf(x):
            return x

        @task(merge_func=mf)
        def with_merge():
            pass

        assert with_merge.merge_func is not None

    def test_task_with_ui_css(self):
        css = UiCss(color="#ffffff")

        @task(ui_css=css)
        def styled():
            pass

        assert styled.ui_css is css


# ══════════════════════════════════════════════════════════════════════════════
#  @parallel_task decorator
# ══════════════════════════════════════════════════════════════════════════════


class TestParallelTaskDecorator:
    def test_parallel_task_no_parens(self):
        @parallel_task
        def par():
            pass

        assert isinstance(par, TaskType)
        assert par.is_parallel is True

    def test_parallel_task_with_parens(self):
        @parallel_task()
        def par2():
            pass

        assert par2.is_parallel is True

    def test_parallel_task_with_ui_css(self):
        css = UiCss(border="#000000")

        @parallel_task(ui_css=css)
        def par3():
            pass

        assert par3.ui_css is css


# ══════════════════════════════════════════════════════════════════════════════
#  @taskgroup decorator
# ══════════════════════════════════════════════════════════════════════════════


class TestTaskgroupDecorator:
    def test_taskgroup_no_parens_returns_task_group(self):
        @task
        def tgd_a():
            return 1

        @task
        def tgd_b():
            return 2

        @taskgroup
        def my_tg():
            return tgd_a, tgd_b

        assert isinstance(my_tg, TaskGroup)
        assert my_tg.name == "my_tg"
        assert len(my_tg.elements) == 2

    def test_taskgroup_with_parens(self):
        @task
        def tgd_c():
            return 3

        @taskgroup()
        def my_tg2():
            return (tgd_c,)

        assert isinstance(my_tg2, TaskGroup)

    def test_taskgroup_with_ui_css(self):
        @task
        def tgd_d():
            return 4

        css = UiCss(background="#112233")

        @taskgroup(ui_css=css)
        def my_tg3():
            return (tgd_d,)

        assert my_tg3.ui_css is css

    def test_taskgroup_with_docstring(self):
        @task
        def tgd_e():
            return 5

        @taskgroup
        def my_tg4():
            """Group doc."""
            return (tgd_e,)

        assert my_tg4.docstring == "Group doc."

    def test_taskgroup_construction_failure_raises_task_group_exception(self):
        with pytest.raises(TaskGroupException):

            @taskgroup
            def bad_tg():
                raise ValueError("construction failed")

    def test_taskgroup_decorator_scalar_return_raises(self):
        """taskgroup decorator with a function returning a non-list/tuple scalar
        => `*tasks if isinstance(tasks, (list, tuple)) else tasks` evaluates
        as `*(tasks if cond else tasks)` ; with `tasks=42` the `else` branch
        yields `42`, then `*42` raises TypeError at the call site (splat target
        must be iterable).

        The splat TypeError fires outside the decorator's try/except (which
        only wraps `tasks = f()`), so it propagates as TypeError to the caller.
        """
        with pytest.raises(TypeError):

            @taskgroup
            def bad_tg_scalar():
                return 42


# ══════════════════════════════════════════════════════════════════════════════
#  DAG.to_elements_lists
# ══════════════════════════════════════════════════════════════════════════════


class TestDagToElementsLists:
    def _simple_dag(self):
        @task
        def root_t():
            return 1

        @task
        def leaf_t(x):
            return x

        root_t >> leaf_t

        @dag
        def d():
            return leaf_t

        return d

    def test_returns_two_lists(self):
        tl, gl = self._simple_dag().to_elements_lists()
        assert isinstance(tl, list) and isinstance(gl, list)

    def test_all_tasks_present(self):
        tl, _ = self._simple_dag().to_elements_lists()
        names = {t["name"] for t in tl}
        assert "root_t" in names and "leaf_t" in names

    def test_serializable_flag_converts_uuids_to_str(self):
        tl, _ = self._simple_dag().to_elements_lists(serializable=True)
        for entry in tl:
            assert isinstance(entry["uuid"], str)

    def test_non_serializable_flag_keeps_uuids_as_uuid_objects(self):
        """serializable=False (default) => uuids are UUID objects, not strings.

        Sibling to test_serializable_flag_converts_uuids_to_str ; covers the
        other branch of the ternary that returns the raw UUID object.
        """
        tl, _ = self._simple_dag().to_elements_lists()
        for entry in tl:
            assert isinstance(entry["uuid"], UUID)

    def test_task_group_included_in_groups_list(self):
        @task
        def ta():
            return 1

        @task
        def tb():
            return 2

        # Nesting TaskGroups ensures the traversal logic correctly recursively
        # processes inner groups and surfaces them in the elements list.
        inner_tg2 = TaskGroup("InnerTG2", tb)
        inner_tg = TaskGroup("InnerTG", ta, inner_tg2)
        outer_tg = TaskGroup("OuterTG", inner_tg)

        @task
        def tc(x, y):
            return x

        outer_tg >> tc

        @dag
        def d_with_tg():
            return tc

        _, gl = d_with_tg.to_elements_lists()
        assert "InnerTG" in {g["name"] for g in gl}

    def test_taskgroup_elements_contains_inner_tg_uuid_string(self):
        """TaskGroup dict's elements field includes inner TaskGroup uuids as strings.

        Covers the else branch of the elements list comprehension:
        `str(e.tasktype_uuid) if isinstance(e, TaskType) else str(e.uuid)`
        where `e` is an inner TaskGroup.
        """

        @task
        def ta():
            return 1

        @task
        def tb():
            return 2

        inner_tg = TaskGroup("InnerTG", ta, tb)
        outer_tg = TaskGroup("OuterTG", inner_tg)

        @task
        def tc(x, y):
            return x

        outer_tg >> tc

        @dag
        def d_with_tg():
            return tc

        _, gl = d_with_tg.to_elements_lists()
        outer_g = next(g for g in gl if g["name"] == "OuterTG")
        assert len(outer_g["elements"]) == 1
        # The single element is the inner TaskGroup's uuid as a string.
        assert isinstance(outer_g["elements"][0], str)
        assert outer_g["elements"][0] == str(inner_tg.uuid)

    def test_task_with_ui_css_entry_has_css(self):
        @task(ui_css=UiCss(background="#aabbcc"))
        def styled_t():
            return 1

        @dag
        def d_css():
            return styled_t

        tl, _ = d_css.to_elements_lists()
        entry = next(e for e in tl if e["name"] == "styled_t")
        assert entry["ui_css"] is not None

    def test_parallel_task_flagged_in_list(self):
        @parallel_task
        def par_t():
            return []

        @dag
        def d_par():
            return par_t

        tl, _ = d_par.to_elements_lists()
        assert tl[0]["is_parallel"] is True

    def test_children_listed_as_uuid_strings(self):
        tl, _ = self._simple_dag().to_elements_lists()
        root_entry = next(e for e in tl if e["name"] == "root_t")
        assert all(isinstance(c, str) for c in root_entry["children"])

    def test_to_elements_lists_merge_func(self):
        """Verify that tasks with a merge_func are correctly serialized with their merge metadata."""

        @task
        def root_t():
            return 1

        @task(merge_func=lambda x: x)
        def merge_t(x):
            return x

        root_t >> merge_t

        @dag
        def d():
            return merge_t

        tl, gl = d.to_elements_lists()
        assert tl[1]["merge_func"] is not None


# ══════════════════════════════════════════════════════════════════════════════
#  DAG.help()
# ══════════════════════════════════════════════════════════════════════════════


class TestDagHelp:
    def test_help_no_params_message(self):
        @task
        def ht1():
            return 1

        @dag
        def d_help():
            return ht1

        assert "No parameters defined" in d_help.help()

    def test_help_with_docstring_included(self):
        @task
        def ht2():
            return 1

        @dag
        def d_help2():
            """Pipeline docstring."""
            return ht2

        assert "Pipeline docstring" in d_help2.help()

    def test_help_with_param_and_default(self):
        @task
        def ht3():
            return 1

        @dag
        def d_help3():
            threshold = DagParam(description="Threshold value", default=0.5)  # noqa: F841
            return ht3

        text = d_help3.help()
        assert "threshold" in text
        assert "0.5" in text

    def test_help_with_param_no_default_omits_default(self):
        @task
        def ht4():
            return 1

        @dag
        def d_help4():
            alpha = DagParam(description="Alpha")  # noqa: F841
            return ht4

        text = d_help4.help()
        assert "alpha" in text
        assert "default" not in text


# ══════════════════════════════════════════════════════════════════════════════
#  DAG.mark_complete
# ══════════════════════════════════════════════════════════════════════════════


class TestDagMarkComplete:
    def test_mark_complete_sets_end_timestamp(self, isolated_db, isolated_dao):
        """mark_complete writes _end_timestamp on the real Execution row."""
        dao = isolated_dao
        exec_id = dao.add_execution(
            name="test_pipe",
            username="tester",
            _start_timestamp=datetime.now(timezone.utc),
            metadata_root="/tmp/meta",
            artifacts_store_root="/tmp/artifacts",
        )
        ec = DagExecutionContext({"exec_id": exec_id})
        token = _dag_execution_context_var.set(ec)
        try:
            DAG.mark_complete(exec_id=exec_id)
            # Real Execution row has _end_timestamp set.
            execution = dao.get_execution(exec_id)
            assert execution._end_timestamp is not None
        finally:
            _dag_execution_context_var.reset(token)

    def test_mark_complete_asserts_context_set(self):
        _dag_execution_context_var.set(None)
        with pytest.raises(AssertionError):
            DAG.mark_complete(exec_id=1)


# ══════════════════════════════════════════════════════════════════════════════
#  DAG.init()
# ══════════════════════════════════════════════════════════════════════════════


class TestDagInit:
    def _make_dag(self):
        @task
        def init_t():
            return 1

        @dag
        def d_init():
            return init_t

        return d_init

    def _make_dag_with_taskgroup(self):
        @task
        def init_ta():
            return 1

        @task
        def init_tb():
            return 2

        tg = TaskGroup("InitTG", init_ta, init_tb)

        @task
        def init_tc(x, y):
            return x

        tg >> init_tc

        @dag
        def d_init_tg():
            return init_tc

        return d_init_tg

    def test_init_creates_execution_record(
        self, isolated_db, isolated_dao, artifacts_store_root
    ):
        """DAG.init() creates a real Execution row and updates the context.

        in_notebook() is patched to return True (notebook branch). The real
        DAO writes a real Execution row ; the real link_params_defaults_to_exec
        runs against the artifacts_store_root fixture.
        """
        d_init = self._make_dag()
        dao = isolated_dao
        ec = DagExecutionContext({"exec_id": None})
        token = _dag_execution_context_var.set(ec)
        try:
            with (
                patch.object(core_module, "in_notebook", return_value=True),
                patch.dict(sys.modules, {"__main__": types.ModuleType("__main__")}),
            ):
                d_init.init()
            exec_id = ec._params["exec_id"]
            assert exec_id is not None
            assert dao.get_execution(exec_id) is not None
        finally:
            _dag_execution_context_var.reset(token)

    def test_init_non_notebook_path_uses_caller_filename(
        self, isolated_db, isolated_dao, artifacts_store_root
    ):
        """non-notebook path derives pipeline_name from caller filename.

        in_notebook() is patched to return False. inspect.currentframe() is
        patched to return a fake frame whose f_back.f_back.f_back points at a
        caller file named 'my_pipeline.py', so pipeline_name is derived from
        its basename. We patch currentframe (rather than relying on the real
        caller frame) because pytest's internal _callers module sits in the
        real frame chain and would pollute the derived name.
        """
        d_init = self._make_dag()
        dao = isolated_dao
        ec = DagExecutionContext({"exec_id": None})
        token = _dag_execution_context_var.set(ec)

        fake_frame = MagicMock()
        fake_frame.f_code.co_filename = "/some/path/my_pipeline.py"

        try:
            with (
                patch.object(core_module, "in_notebook", return_value=False),
                patch.object(
                    core_module.inspect,
                    "currentframe",
                    return_value=MagicMock(
                        f_back=MagicMock(f_back=MagicMock(f_back=fake_frame))
                    ),
                ),
            ):
                d_init.init()
            exec_id = ec._params["exec_id"]
            assert exec_id is not None
            execution = dao.get_execution(exec_id)
            assert execution.name == "my_pipeline"
        finally:
            _dag_execution_context_var.reset(token)

    def test_init_non_notebook_retraining_pipeline_filename_uses_dirname(
        self, isolated_db, isolated_dao, artifacts_store_root
    ):
        """caller file is named 'retraining_pipeline' => use dirname.

        in_notebook() is patched to return False. inspect.currentframe() is
        patched to return a fake frame whose f_back.f_back.f_back points at a
        caller file named 'retraining_pipeline.py' under '/some/pipeline_dir/'.
        """
        d_init = self._make_dag()
        dao = isolated_dao
        ec = DagExecutionContext({"exec_id": None})
        token = _dag_execution_context_var.set(ec)

        fake_frame = MagicMock()
        fake_frame.f_code.co_filename = "/some/pipeline_dir/retraining_pipeline.py"

        try:
            with (
                patch.object(core_module, "in_notebook", return_value=False),
                patch.object(
                    core_module.inspect,
                    "currentframe",
                    return_value=MagicMock(
                        f_back=MagicMock(f_back=MagicMock(f_back=fake_frame))
                    ),
                ),
            ):
                d_init.init()
            exec_id = ec._params["exec_id"]
            execution = dao.get_execution(exec_id)
            assert execution.name == "pipeline_dir"
        finally:
            _dag_execution_context_var.reset(token)

    def test_init_adds_taskgroup_to_dao(
        self, isolated_db, isolated_dao, artifacts_store_root
    ):
        """dao.add_taskgroup called when DAG contains a taskgroup.

        in_notebook() is patched to return True. Real TaskGroup rows are
        written to the real DB.
        """
        d_init_tg = self._make_dag_with_taskgroup()
        dao = isolated_dao
        ec = DagExecutionContext({"exec_id": None})
        token = _dag_execution_context_var.set(ec)
        try:
            with (
                patch.object(core_module, "in_notebook", return_value=True),
                patch.dict(sys.modules, {"__main__": types.ModuleType("__main__")}),
            ):
                d_init_tg.init()
            exec_id = ec._params["exec_id"]
            # Real TaskGroup rows written.
            taskgroups = dao._get_entities(TaskGroupORM, exec_id=exec_id)
            assert len(taskgroups) >= 1
            assert any(tg.name == "InitTG" for tg in taskgroups)
        finally:
            _dag_execution_context_var.reset(token)

    def test_init_notebook_pipeline_name_from_main_file(
        self, isolated_db, isolated_dao, artifacts_store_root
    ):
        """notebook mode ; __main__.__file__ is a non-ipykernel path.

        in_notebook() is patched to return True. __main__ is faked to a module
        with __file__ pointing at a non-ipykernel .py file. The notebook branch
        checks __main__.__file__ BEFORE the sys.modules scan, so the derived
        pipeline_name is the __file__ basename.
        """
        d_init = self._make_dag()
        dao = isolated_dao
        ec = DagExecutionContext({"exec_id": None})
        token = _dag_execution_context_var.set(ec)

        fake_main = types.ModuleType("__main__")
        fake_main.__file__ = "/some/path/my_pipeline.py"

        try:
            with (
                patch.object(core_module, "in_notebook", return_value=True),
                patch.dict(sys.modules, {"__main__": fake_main}),
            ):
                d_init.init()
            exec_id = ec._params["exec_id"]
            execution = dao.get_execution(exec_id)
            assert execution.name == "my_pipeline"
        finally:
            _dag_execution_context_var.reset(token)

    def test_init_notebook_pipeline_name_from_sys_modules_scan(
        self, isolated_db, isolated_dao, artifacts_store_root
    ):
        """notebook mode ; __main__ has no __file__ => scan sys.modules for DAG owner.

        in_notebook() is patched to return True. __main__ is faked to a module
        with no __file__. A fake owner module is placed in sys.modules with the
        DAG instance as an attribute, so the scan finds it.
        """
        d_init = self._make_dag()
        dao = isolated_dao
        ec = DagExecutionContext({"exec_id": None})
        token = _dag_execution_context_var.set(ec)

        fake_main = types.ModuleType("__main__")
        fake_owner_mod = types.ModuleType("my_owner_pipeline")
        fake_owner_mod.the_dag = d_init

        try:
            with (
                patch.object(core_module, "in_notebook", return_value=True),
                patch.dict(
                    sys.modules,
                    {"__main__": fake_main, "my_owner_pipeline": fake_owner_mod},
                ),
            ):
                d_init.init()
            exec_id = ec._params["exec_id"]
            execution = dao.get_execution(exec_id)
            assert execution.name == "my_owner_pipeline"
        finally:
            _dag_execution_context_var.reset(token)

    def test_init_notebook_pipeline_name_falls_back_to_cwd(
        self, isolated_db, isolated_dao, artifacts_store_root
    ):
        """notebook mode ; no owner found => 'retraining_pipeline' => os.getcwd() basename.

        in_notebook() is patched to return True. __main__ has no __file__, no
        module owns the dag => falls back to 'retraining_pipeline' =>
        os.getcwd() basename. os.getcwd() is patched to return a known path.

        An integer entry in sys.modules (non-None, non-dunder-named) causes
        vars(42) to raise TypeError inside the pipeline-name scan, exercising
        the except Exception: pass handler.
        """
        d_init = self._make_dag()
        dao = isolated_dao
        ec = DagExecutionContext({"exec_id": None})
        token = _dag_execution_context_var.set(ec)

        fake_main = types.ModuleType("__main__")

        try:
            with (
                patch.object(core_module, "in_notebook", return_value=True),
                patch.object(
                    core_module.os, "getcwd", return_value="/some/path/my_cwd_pipeline"
                ),
                # 42 is not None and not __dunder__-named, so vars(42) raises TypeError,
                # hitting the except Exception: pass block.
                patch.dict(sys.modules, {"__main__": fake_main, "bad_non_module": 42}),
            ):
                d_init.init()
            exec_id = ec._params["exec_id"]
            execution = dao.get_execution(exec_id)
            assert execution.name == "my_cwd_pipeline"
        finally:
            _dag_execution_context_var.reset(token)
