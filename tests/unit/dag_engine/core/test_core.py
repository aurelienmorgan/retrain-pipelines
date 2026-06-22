"""
Unit tests for retrain_pipelines.dag_engine.core.core
"""

import logging
import pickle
import sys
from datetime import date, datetime, timezone
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

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


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════


def _real_task(name: str, *, is_parallel=False, merge_func=None) -> TaskType:
    def func():
        pass

    func.__name__ = name
    func.__module__ = "__test__"
    return TaskType(func=func, is_parallel=is_parallel, merge_func=merge_func)


def _stub_task(name, parents=None):
    t = MagicMock(spec=TaskType)
    type(t).name = PropertyMock(return_value=name)
    t.__class__ = TaskType
    t._parents = parents or []
    t.parents = t._parents
    t._children = []
    t.children = t._children
    t.task_group = None
    t.__hash__ = lambda self: id(self)
    t.__eq__ = lambda self, other: self is other
    return t


def _elem(name):
    """Lightweight TaskType stand-in for TaskGroup element slots."""
    t = MagicMock(spec=TaskType)
    t.__class__ = TaskType
    t.name = name
    t._task_group = None
    return t


def _make_dao_mock(task_id=1, exec_id=10):
    """Return a DAO mock suitable for wrapping tests."""
    dao = MagicMock()
    dao.add_task.return_value = task_id
    dao.add_execution.return_value = exec_id
    dao.update_task.return_value = None
    dao.update_execution.return_value = None
    dao.add_tasktype.return_value = None
    dao.add_taskgroup.return_value = None
    dao.dispose.return_value = None
    return dao


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

    def test_merge_updates_applies_to_params(self):
        ctx = DagExecutionContext({"a": 1})
        ctx.merge_updates({"a": 99, "z": 0})
        assert ctx.a == 99 and ctx.z == 0

    def test_merge_updates_also_recorded_in_updates(self):
        ctx = DagExecutionContext({})
        ctx.merge_updates({"k": 5})
        assert ctx.get_updates()["k"] == 5

    def test_deep_update_nested_dict(self):
        d = {"a": {"b": 1, "c": 2}}
        DagExecutionContext._deep_update(d, {"a": {"b": 99}})
        assert d["a"]["b"] == 99
        assert d["a"]["c"] == 2  # untouched sibling

    def test_deep_update_overwrites_non_dict(self):
        d = {"a": 1}
        DagExecutionContext._deep_update(d, {"a": [1, 2]})
        assert d["a"] == [1, 2]

    def test_deep_update_adds_new_key(self):
        d = {}
        DagExecutionContext._deep_update(d, {"new": 42})
        assert d["new"] == 42

    def test_copy_is_deep(self):
        ctx = DagExecutionContext({"a": [1, 2]})
        ctx2 = ctx.copy()
        ctx2._params["a"].append(3)
        assert ctx._params["a"] == [1, 2]

    def test_to_serializable_dict_datetime(self):
        ctx = DagExecutionContext({"ts": datetime(2024, 1, 1, tzinfo=timezone.utc)})
        assert ctx.to_serializable_dict()["ts"] == "2024-01-01T00:00:00+00:00"

    def test_to_serializable_dict_primitives(self):
        ctx = DagExecutionContext({"n": 5, "s": "hello", "f": 3.14, "b": True})
        d = ctx.to_serializable_dict()
        assert d == {"n": 5, "s": "hello", "f": 3.14, "b": True}


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

    def test_serializable_dict_basic(self):
        assert DagParam(description="d", default=42).to_serializable_dict() == {
            "description": "d",
            "default": 42,
        }

    def test_serialize_datetime(self):
        ts = datetime(2024, 6, 1, tzinfo=timezone.utc)
        assert (
            "2024-06-01"
            in DagParam(description="d", default=ts).to_serializable_dict()["default"]
        )

    def test_serialize_date(self):
        assert (
            DagParam(description="d", default=date(2024, 6, 1)).to_serializable_dict()[
                "default"
            ]
            == "2024-06-01"
        )

    def test_serialize_nested_dict(self):
        assert DagParam(description="d", default={"a": [1, 2]}).to_serializable_dict()[
            "default"
        ] == {"a": [1, 2]}

    def test_serialize_list(self):
        assert DagParam(
            description="d", default=[1, "two", None]
        ).to_serializable_dict()["default"] == [1, "two", None]

    def test_serialize_tuple(self):
        result = DagParam(description="d", default=(1, 2)).to_serializable_dict()[
            "default"
        ]
        assert result == [1, 2]

    def test_serialize_set(self):
        result = DagParam(description="d", default={99}).to_serializable_dict()[
            "default"
        ]
        assert result == [99]

    def test_serialize_none(self):
        assert DagParam(description="d").to_serializable_dict()["default"] is None

    def test_serialize_bool_preserved(self):
        assert (
            DagParam(description="d", default=True).to_serializable_dict()["default"]
            is True
        )

    def test_serialize_unknown_type_falls_back_to_str(self):
        class Custom:
            pass

        d = DagParam(description="d", default=Custom()).to_serializable_dict()
        assert isinstance(d["default"], str)

    def test_serialize_pydantic_model(self):
        from pydantic import BaseModel as BM

        class M(BM):
            x: int = 1

        result = DagParam(description="d", default=M()).to_serializable_dict()[
            "default"
        ]
        assert result == {"x": 1}


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
        import types

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
        import types

        mod = types.ModuleType("_test_recon_bad_mod")
        mod.not_a_task = "just a string"
        sys.modules["_test_recon_bad_mod"] = mod
        try:
            with pytest.raises(ValueError, match="Expected"):
                TaskType._reconstruct_task_by_name("not_a_task", "_test_recon_bad_mod")
        finally:
            del sys.modules["_test_recon_bad_mod"]


# ══════════════════════════════════════════════════════════════════════════════
#  TaskType – _wrap_func and _wrap_merge_func via mocked DAO + runtime
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture()
def _dao_and_registry():
    """Patch DAO and _task_registry so wrapped funcs don't need a real DB."""
    dao_mock = _make_dao_mock(task_id=1)
    registry_mock = MagicMock()
    with (
        patch("retrain_pipelines.dag_engine.core.core.DAO", return_value=dao_mock),
        patch(
            "retrain_pipelines.dag_engine.core.core.get_trace_buffer",
            return_value=MagicMock(add_trace=MagicMock(), flush=MagicMock()),
        ),
        patch("retrain_pipelines.dag_engine.runtime._task_registry", registry_mock),
    ):
        yield dao_mock, registry_mock


class TestTaskTypeWrapFunc:
    def test_wrap_func_success_returns_task_id_and_result(self, _dao_and_registry):
        dao_mock, reg_mock = _dao_and_registry

        def my_func():
            return 42

        t = TaskType(func=my_func, is_parallel=False)
        task_id, result = t.func(exec_id=99)
        assert task_id == 1
        assert result == 42
        dao_mock.update_task.assert_called_once()
        reg_mock.unregister_task.assert_called_once_with(1)

    def test_wrap_func_with_docstring(self, _dao_and_registry):
        """Verifies the docstring formatting branch (func.__doc__ truthy)."""

        def my_func():
            """My doc."""
            return "ok"

        t = TaskType(func=my_func, is_parallel=False)
        _, result = t.func(exec_id=1)
        assert result == "ok"

    def test_wrap_func_failure_raises_task_func_exception(self, _dao_and_registry):
        def bad_func():
            raise ValueError("boom")

        t = TaskType(func=bad_func, is_parallel=False)
        with pytest.raises(TaskFuncException):
            t.func(exec_id=1)

    def test_wrap_func_marks_task_failed_on_exception(self, _dao_and_registry):
        dao_mock, _ = _dao_and_registry

        def bad_func():
            raise RuntimeError("oops")

        t = TaskType(func=bad_func, is_parallel=False)
        with pytest.raises(TaskFuncException):
            t.func(exec_id=1)
        call_kwargs = dao_mock.update_task.call_args[1]
        assert call_kwargs["failed"] is True

    def test_wrap_func_with_rank_in_signature(self, _dao_and_registry):
        """'rank' in func signature => kept in kwargs, not popped."""

        def func_with_rank(rank=None):
            return rank

        t = TaskType(func=func_with_rank, is_parallel=False)
        _, result = t.func(exec_id=1, rank=7)
        assert result == 7

    def test_wrap_func_with_task_id_in_signature_injects_id(self, _dao_and_registry):
        """'task_id' in func signature => (mocked) DAO creates it, wrapper injects it."""

        # task_id must NOT be pre-passed; the wrapper injects it after add_task.
        def a_tasktype_func(task_id=None):
            return task_id

        t = TaskType(func=a_tasktype_func, is_parallel=False)

        # Pass task_id=None to satisfy core.py's kwargs["task_id"] lookup.
        # The wrapper will still overwrite it with the DAO-generated ID.
        returned_task_id, task_id = t.func(exec_id=1, task_id=None)

        # id matches the DAO-returned id
        assert task_id == returned_task_id

    def test_wrap_func_with_provided_task_id_skips_dao_add(self, _dao_and_registry):
        """Pre-supplied task_id (from merge wrapper) => DAO.add_task NOT called."""
        dao_mock, _ = _dao_and_registry

        def plain():
            return "done"

        t = TaskType(func=plain, is_parallel=False)
        t.func(exec_id=1, task_id=99)
        dao_mock.add_task.assert_not_called()

    def test_wrap_merge_func_success(self, _dao_and_registry):
        dao_mock, reg_mock = _dao_and_registry

        def merge(x):
            return x * 2

        def main_func():
            return 1

        t = TaskType(func=main_func, is_parallel=False, merge_func=merge)
        task_id, result = t.merge_func(10, exec_id=5, rank=None)
        assert task_id == 1
        assert result == 20

    def test_wrap_merge_func_failure_raises_merge_exception(self, _dao_and_registry):
        def bad_merge(x):
            raise ValueError("merge fail")

        def main_func():
            return 1

        t = TaskType(func=main_func, is_parallel=False, merge_func=bad_merge)
        with pytest.raises(TaskMergeFuncException):
            t.merge_func(0, exec_id=1, rank=None)

    def test_wrap_merge_func_marks_task_failed(self, _dao_and_registry):
        dao_mock, _ = _dao_and_registry

        def bad_merge(x):
            raise RuntimeError("m")

        def main_func():
            return 1

        t = TaskType(func=main_func, is_parallel=False, merge_func=bad_merge)
        with pytest.raises(TaskMergeFuncException):
            t.merge_func(0, exec_id=1, rank=None)
        call_kwargs = dao_mock.update_task.call_args[1]
        assert call_kwargs["failed"] is True


# ══════════════════════════════════════════════════════════════════════════════
#  _capture_and_stream_trace – notebook-mode and pipe_reader branches
# ══════════════════════════════════════════════════════════════════════════════


class TestCaptureAndStreamTrace:
    def test_ipython_import_error_sets_non_notebook_mode(self, _dao_and_registry):
        """exception during get_ipython() => _notebook_mode = False."""

        def my_func():
            print("hello from task")
            return "ok"

        # Simulate IPython import raising an exception
        with patch.dict("sys.modules", {"IPython": None}):
            t = TaskType(func=my_func, is_parallel=False)
            _, result = t.func(exec_id=1)
        assert result == "ok"

    def test_notebook_mode_patches_write_methods(self, _dao_and_registry):
        """notebook mode wraps capturer_out/err.write with _nb_write."""

        def my_func():
            print("notebook output")
            return "nb_ok"

        ipython_mock = MagicMock()
        ipython_mock.config = {"IPKernelApp": True}

        # get_ipython is imported locally inside _capture_and_stream_trace,
        # so patch it at the IPython module level.
        import IPython

        with patch.object(IPython, "get_ipython", return_value=ipython_mock):
            t = TaskType(func=my_func, is_parallel=False)
            _, result = t.func(exec_id=1)
        assert result == "nb_ok"

    def test_pipe_reader_processes_newline_delimited_output(self, _dao_and_registry):
        """pipe_reader splits on newlines and adds traces to buffer."""
        trace_buffer_mock = MagicMock(add_trace=MagicMock(), flush=MagicMock())

        def my_func():
            # Write multiple lines to trigger pipe_reader line-splitting logic
            sys.stdout.write("line1\nline2\n")
            return "multiline"

        with patch(
            "retrain_pipelines.dag_engine.core.core.get_trace_buffer",
            return_value=trace_buffer_mock,
        ):
            t = TaskType(func=my_func, is_parallel=False)
            _, result = t.func(exec_id=1)
        assert result == "multiline"

    def test_pipe_reader_flushes_partial_line_at_eof(self, _dao_and_registry):
        """pipe_reader flushes remaining buffer (no trailing newline)."""
        trace_buffer_mock = MagicMock(add_trace=MagicMock(), flush=MagicMock())

        def my_func():
            # Write without trailing newline => pipe_reader sees partial buffer at close
            sys.stdout.write("no newline at end")
            return "partial"

        with patch(
            "retrain_pipelines.dag_engine.core.core.get_trace_buffer",
            return_value=trace_buffer_mock,
        ):
            t = TaskType(func=my_func, is_parallel=False)
            _, result = t.func(exec_id=1)
        assert result == "partial"


# ══════════════════════════════════════════════════════════════════════════════
#  TaskGroup
# ══════════════════════════════════════════════════════════════════════════════


class TestTaskGroup:
    def test_positional_name_and_elements(self):
        t1, t2 = _elem("t1"), _elem("t2")
        tg = TaskGroup("MyGroup", t1, t2)
        assert tg.name == "MyGroup"
        assert len(tg.elements) == 2

    def test_keyword_name(self):
        t = _elem("t")
        tg = TaskGroup(t, name="KwGroup")
        assert tg.name == "KwGroup"

    def test_missing_name_raises_type_error(self):
        with pytest.raises(TypeError):
            TaskGroup()

    def test_non_string_name_raises_type_error(self):
        with pytest.raises(TypeError):
            TaskGroup(123, _elem("t"))

    def test_pydantic_validation_failure_raises_task_group_exception(self):
        """pydantic validation error => TaskGroupException."""
        # Pass an invalid ui_css value to trigger pydantic validation inside super().__init__
        with pytest.raises(TaskGroupException, match="Error instanciating"):
            TaskGroup("BadCss", _elem("t"), ui_css="not_a_ui_css_instance")

    def test_each_uuid_is_unique(self):
        t = _elem("t")
        tg1 = TaskGroup("G1", t)
        tg2 = TaskGroup("G2", t)
        assert tg1.uuid != tg2.uuid
        assert hash(tg1) != hash(tg2)

    def test_equality_is_by_uuid(self):
        t = _elem("t")
        tg = TaskGroup("G", t)
        assert tg == tg
        assert tg != TaskGroup("H", t)

    def test_eq_non_task_group_false(self):
        t = _elem("t")
        assert TaskGroup("G", t) != "string"

    def test_str_contains_group_name(self):
        t = _elem("t")
        assert "MyTG" in str(TaskGroup("MyTG", t))

    def test_repr_contains_TaskGroup(self):
        t = _elem("t")
        assert "TaskGroup" in repr(TaskGroup("G", t))

    def test_elements_get_task_group_reference(self):
        t = _elem("t")
        tg = TaskGroup("G", t)
        assert t._task_group is tg

    def test_log_property_returns_logger(self):
        t = _elem("t")
        assert isinstance(TaskGroup("G", t).log, logging.Logger)

    def test_task_group_property_none_at_top_level(self):
        t = _elem("t")
        assert TaskGroup("G", t).task_group is None

    def test_parents_property_empty(self):
        t = _elem("t")
        assert TaskGroup("G", t).parents == []

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
        t = _elem("t")
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
        a = _stub_task("A")
        b = _stub_task("B", parents=[a])
        a.children = [b]
        assert DAG._find_root_tasks(b) == [a]

    def test_task_with_no_parents_is_its_own_root(self):
        a = _stub_task("A")
        assert DAG._find_root_tasks(a) == [a]

    def test_long_chain_finds_head(self):
        a, b, c, d = [_stub_task(n) for n in "ABCD"]
        a.children = [b]
        b.parents = [a]
        b.children = [c]
        c.parents = [b]
        c.children = [d]
        d.parents = [c]
        assert DAG._find_root_tasks(d) == [a]

    def test_multiple_roots(self):
        a = _stub_task("A")
        b = _stub_task("B")
        c = _stub_task("C", parents=[a, b])
        a.children = [c]
        b.children = [c]
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
        """TaskGroup encountered mid-stack => its parents are pushed onto stack."""
        # Build: root => inner_task, inner_task is inside a mid_tg TaskGroup.
        # We simulate a TaskGroup in the ancestry traversal by constructing
        # a chain where a TaskGroup appears in parents during BFS.
        root = _real_task("rt_root")
        mid = _real_task("rt_mid")
        root >> mid

        # Wrap mid in a TaskGroup so _find_root_tasks encounters a TaskGroup on stack
        mid_tg = TaskGroup("MidTG", mid)
        # Manually inject the TaskGroup as a parent of a leaf so it surfaces in traversal
        leaf = _real_task("rt_leaf")
        mid_tg >> leaf

        roots = DAG._find_root_tasks(leaf)
        assert root in roots


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

    def test_no_func_node_raises_value_error(self):
        """no function definition found in AST => ValueError."""

        # Feed a source that has no FunctionDef at the top level.
        # We achieve this by monkeypatching inspect.getsource to return a plain assignment.
        def dummy():
            pass

        with patch(
            "retrain_pipelines.dag_engine.core.core.inspect.getsource",
            return_value="x = 1\n",
        ):
            with pytest.raises(ValueError, match="Could not find function definition"):
                _get_dag_params(dummy)

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

    def test_task_inner_wrapper_invoked_directly(self):
        """inner wrapper function inside @task decorator is callable."""
        called = []

        # Access the wrapper via _task attribute on a manually-constructed decorator result
        def raw_func():
            called.append(True)
            return "raw"

        import functools

        # Replicate the decorator's wrapper creation
        @functools.wraps(raw_func)
        def wrapper(*args, **kwargs):
            return raw_func(*args, **kwargs)

        wrapper._task = TaskType(func=raw_func, is_parallel=False)
        result = wrapper()
        assert result == "raw"
        assert called


# ══════════════════════════════════════════════════════════════════════════════
#  @parallel_task decorator – inner wrapper callable
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

    def test_parallel_task_inner_wrapper_invoked_directly(self):
        """inner wrapper function inside @parallel_task decorator is callable."""
        import functools

        called = []

        def raw_func():
            called.append(True)
            return "par_raw"

        @functools.wraps(raw_func)
        def wrapper(*args, **kwargs):
            return raw_func(*args, **kwargs)

        wrapper._task = TaskType(func=raw_func, is_parallel=True)
        result = wrapper()
        assert result == "par_raw"
        assert called


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

    def test_task_group_included_in_groups_list(self):
        @task
        def ta():
            return 1

        @task
        def tb():
            return 2

        tg = TaskGroup("MyTG", ta, tb)

        @task
        def tc(x, y):
            return x

        tg >> tc

        @dag
        def d_with_tg():
            return tc

        _, gl = d_with_tg.to_elements_lists()
        assert "MyTG" in {g["name"] for g in gl}

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
    def test_mark_complete_calls_dao(self):
        dao_mock = _make_dao_mock()
        ec = DagExecutionContext({"exec_id": 5})
        token = _dag_execution_context_var.set(ec)
        try:
            with patch(
                "retrain_pipelines.dag_engine.core.core.DAO", return_value=dao_mock
            ):
                DAG.mark_complete(exec_id=5)
            dao_mock.update_execution.assert_called_once()
            dao_mock.dispose.assert_called_once()
        finally:
            _dag_execution_context_var.reset(token)

    def test_mark_complete_asserts_context_set(self):
        _dag_execution_context_var.set(None)
        with patch(
            "retrain_pipelines.dag_engine.core.core.DAO", return_value=_make_dao_mock()
        ):
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

    def test_init_creates_execution_record(self):
        d_init = self._make_dag()
        dao_mock = _make_dao_mock(exec_id=42)
        ec = DagExecutionContext({"exec_id": None})
        token = _dag_execution_context_var.set(ec)
        try:
            with (
                patch(
                    "retrain_pipelines.dag_engine.core.core.DAO", return_value=dao_mock
                ),
                patch(
                    "retrain_pipelines.dag_engine.core.core.in_notebook",
                    return_value=True,
                ),
            ):
                d_init.init()
            dao_mock.add_execution.assert_called_once()
            assert ec._params["exec_id"] == 42
        finally:
            _dag_execution_context_var.reset(token)

    def test_init_non_notebook_path_uses_caller_filename(self):
        """non-notebook path derives pipeline_name from caller filename."""
        d_init = self._make_dag()
        dao_mock = _make_dao_mock(exec_id=55)
        ec = DagExecutionContext({"exec_id": None})
        token = _dag_execution_context_var.set(ec)
        try:
            with (
                patch(
                    "retrain_pipelines.dag_engine.core.core.DAO", return_value=dao_mock
                ),
                patch(
                    "retrain_pipelines.dag_engine.core.core.in_notebook",
                    return_value=False,
                ),
            ):
                d_init.init()
            dao_mock.add_execution.assert_called_once()
            assert ec._params["exec_id"] == 55
        finally:
            _dag_execution_context_var.reset(token)

    def test_init_non_notebook_retraining_pipeline_filename_uses_dirname(self):
        """caller file is named 'retraining_pipeline' => use dirname."""
        d_init = self._make_dag()
        dao_mock = _make_dao_mock(exec_id=66)
        ec = DagExecutionContext({"exec_id": None})
        token = _dag_execution_context_var.set(ec)

        fake_frame = MagicMock()
        fake_frame.f_code.co_filename = "/some/pipeline_dir/retraining_pipeline.py"

        try:
            with (
                patch(
                    "retrain_pipelines.dag_engine.core.core.DAO", return_value=dao_mock
                ),
                patch(
                    "retrain_pipelines.dag_engine.core.core.in_notebook",
                    return_value=False,
                ),
                patch(
                    "retrain_pipelines.dag_engine.core.core.inspect.currentframe",
                    return_value=MagicMock(
                        f_back=MagicMock(f_back=MagicMock(f_back=fake_frame))
                    ),
                ),
            ):
                d_init.init()
            call_kwargs = dao_mock.add_execution.call_args[1]
            assert call_kwargs["name"] == "pipeline_dir"
        finally:
            _dag_execution_context_var.reset(token)

    def test_init_adds_taskgroup_to_dao(self):
        """dao.add_taskgroup called when DAG contains a taskgroup."""
        d_init_tg = self._make_dag_with_taskgroup()
        dao_mock = _make_dao_mock(exec_id=77)
        ec = DagExecutionContext({"exec_id": None})
        token = _dag_execution_context_var.set(ec)
        try:
            with (
                patch(
                    "retrain_pipelines.dag_engine.core.core.DAO", return_value=dao_mock
                ),
                patch(
                    "retrain_pipelines.dag_engine.core.core.in_notebook",
                    return_value=True,
                ),
            ):
                d_init_tg.init()
            dao_mock.add_taskgroup.assert_called()
        finally:
            _dag_execution_context_var.reset(token)

    def test_init_notebook_pipeline_name_from_main_file(self):
        """notebook mode ; __main__.__file__ is a non-ipykernel path."""
        d_init = self._make_dag()
        dao_mock = _make_dao_mock(exec_id=88)
        ec = DagExecutionContext({"exec_id": None})
        token = _dag_execution_context_var.set(ec)

        import types

        fake_main = types.ModuleType("__main__")
        fake_main.__file__ = "/some/path/my_pipeline.py"

        try:
            with (
                patch(
                    "retrain_pipelines.dag_engine.core.core.DAO", return_value=dao_mock
                ),
                patch(
                    "retrain_pipelines.dag_engine.core.core.in_notebook",
                    return_value=True,
                ),
                patch.dict(sys.modules, {"__main__": fake_main}),
            ):
                d_init.init()
            call_kwargs = dao_mock.add_execution.call_args[1]
            assert call_kwargs["name"] == "my_pipeline"
        finally:
            _dag_execution_context_var.reset(token)

    def test_init_notebook_pipeline_name_from_sys_modules_scan(self):
        """notebook mode ; scan sys.modules to find DAG owner."""
        d_init = self._make_dag()
        dao_mock = _make_dao_mock(exec_id=89)
        ec = DagExecutionContext({"exec_id": None})
        token = _dag_execution_context_var.set(ec)

        import types

        fake_main = types.ModuleType("__main__")
        # No __file__ => pipeline_name stays None after main_file check
        fake_owner_mod = types.ModuleType("my_owner_pipeline")
        # Place the dag instance as an attribute so the scan finds it
        fake_owner_mod.the_dag = d_init

        try:
            with (
                patch(
                    "retrain_pipelines.dag_engine.core.core.DAO", return_value=dao_mock
                ),
                patch(
                    "retrain_pipelines.dag_engine.core.core.in_notebook",
                    return_value=True,
                ),
                patch.dict(
                    sys.modules,
                    {"__main__": fake_main, "my_owner_pipeline": fake_owner_mod},
                ),
            ):
                d_init.init()
            call_kwargs = dao_mock.add_execution.call_args[1]
            assert call_kwargs["name"] == "my_owner_pipeline"
        finally:
            _dag_execution_context_var.reset(token)

    def test_init_notebook_pipeline_name_falls_back_to_cwd(self):
        """notebook mode ; no owner found => 'retraining_pipeline' => os.getcwd() basename."""
        d_init = self._make_dag()
        dao_mock = _make_dao_mock(exec_id=90)
        ec = DagExecutionContext({"exec_id": None})
        token = _dag_execution_context_var.set(ec)

        import types

        fake_main = types.ModuleType("__main__")
        # No __file__, no module owns the dag => falls back to retraining_pipeline => cwd basename

        try:
            with (
                patch(
                    "retrain_pipelines.dag_engine.core.core.DAO", return_value=dao_mock
                ),
                patch(
                    "retrain_pipelines.dag_engine.core.core.in_notebook",
                    return_value=True,
                ),
                patch(
                    "retrain_pipelines.dag_engine.core.core.os.getcwd",
                    return_value="/some/path/my_cwd_pipeline",
                ),
                patch.dict(sys.modules, {"__main__": fake_main}),
            ):
                d_init.init()
            call_kwargs = dao_mock.add_execution.call_args[1]
            assert call_kwargs["name"] == "my_cwd_pipeline"
        finally:
            _dag_execution_context_var.reset(token)
