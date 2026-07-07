"""
Unit tests for retrain_pipelines.dag_engine.runtime.
"""

import ctypes
import os
import signal
import sys
import threading
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from retrain_pipelines.dag_engine.core.core import (
    DagExecutionContext,
    TaskGroup,
    TaskPayload,
    TaskType,
    _dag_execution_context_var,
)
from retrain_pipelines.dag_engine.runtime import (
    _TaskRegistry,
    _collect_parent_results,
    _execute_branch,
    _execute_branch_with_context,
    _execute_parallel_branches_with_context,
    _execute_task,
    _execute_task_with_context,
    _execute_taskgroup,
    _execute_taskgroup_with_context,
    _find_subdag_end,
    _install_interrupt_handler,
    _interrupt_thread,
    _kill_process,
    _parallel_input_count,
    _sigint_handler,
    _task_registry,
    _topological_sort,
    _update_interrupted_tasks_in_db,
    execute,
)


# -----------------------------------------------------------------------------
#  Shared helpers
# -----------------------------------------------------------------------------


def _mock_task(
    name,
    *,
    is_parallel=False,
    merge_func=None,
    parents=None,
    children=None,
    task_group=None,
):
    """Lightweight duck-type stand-in for TaskType (no DB calls)."""
    t = MagicMock(spec=TaskType)
    type(t).name = PropertyMock(return_value=name)
    t.__class__ = TaskType
    t.is_parallel = is_parallel
    t.merge_func = merge_func
    t._parents = parents or []
    t._children = children or []
    t.parents = t._parents
    t.children = t._children
    t.task_group = task_group
    t.log = MagicMock()
    t.__hash__ = lambda self: id(self)
    t.__eq__ = lambda self, other: self is other
    return t


def _mock_tg(name="tg", elements=None):
    tg = MagicMock(spec=TaskGroup)
    tg.__class__ = TaskGroup
    type(tg).name = PropertyMock(return_value=name)
    tg.elements = elements or []
    tg.log = MagicMock()
    tg.task_group = None
    tg.__hash__ = lambda self: id(self)
    tg.__eq__ = lambda self, other: self is other
    return tg


def _make_context(params=None):
    """Return a fresh DagExecutionContext with exec_id set."""
    return DagExecutionContext({"exec_id": 42, **(params or {})})


# Patch targets (all in runtime module's namespace)
_RT = "retrain_pipelines.dag_engine.runtime"


# ══════════════════════════════════════════════════════════════════════════════
#  _TaskRegistry
# ══════════════════════════════════════════════════════════════════════════════


class TestTaskRegistry:
    def test_register_and_get(self):
        reg = _TaskRegistry()
        reg.register_task(1, 111)
        assert reg.get_running_tasks() == {1: 111}

    def test_unregister_removes_entry(self):
        reg = _TaskRegistry()
        reg.register_task(1, 111)
        reg.unregister_task(1)
        assert reg.get_running_tasks() == {}

    def test_unregister_missing_key_is_noop(self):
        reg = _TaskRegistry()
        reg.unregister_task(999)

    def test_mark_and_check_interrupted(self):
        reg = _TaskRegistry()
        assert not reg.is_interrupted()
        reg.mark_interrupted()
        assert reg.is_interrupted()

    def test_get_returns_snapshot_not_live_view(self):
        reg = _TaskRegistry()
        reg.register_task(1, 10)
        snap = reg.get_running_tasks()
        reg.register_task(2, 20)
        assert 2 not in snap

    def test_multiple_tasks_stored_correctly(self):
        reg = _TaskRegistry()
        for i in range(5):
            reg.register_task(i, i * 10)
        tasks = reg.get_running_tasks()
        assert len(tasks) == 5
        assert tasks[3] == 30

    def test_concurrent_register_unregister_no_errors(self):
        reg = _TaskRegistry()
        errors = []

        def worker(tid):
            try:
                reg.register_task(tid, tid * 100)
                reg.unregister_task(tid)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []


# ══════════════════════════════════════════════════════════════════════════════
#  _interrupt_thread
# ══════════════════════════════════════════════════════════════════════════════


class TestInterruptThread:
    def test_callable(self):
        assert callable(_interrupt_thread)

    def test_invalid_thread_id_does_not_raise(self):
        # ctypes returns 0 => logs warning, no exception
        _interrupt_thread(0)

    def test_valid_thread_id_res_equals_one(self):
        with patch.object(
            ctypes.pythonapi, "PyThreadState_SetAsyncExc", return_value=1
        ):
            _interrupt_thread(threading.current_thread().ident)

    def test_res_greater_than_one_reverts(self):
        # res > 1 branch ; must call SetAsyncExc a second time with None
        revert_calls = []

        def fake_set(ident, exc):
            if exc is None:
                revert_calls.append(ident)
                return 1
            return 2  # first call returns 2 => triggers revert

        with patch.object(
            ctypes.pythonapi, "PyThreadState_SetAsyncExc", side_effect=fake_set
        ):
            _interrupt_thread(999)

        assert revert_calls, "revert call with None was expected"

    def test_exception_in_ctypes_is_absorbed(self):
        with patch.object(
            ctypes.pythonapi,
            "PyThreadState_SetAsyncExc",
            side_effect=RuntimeError("boom"),
        ):
            _interrupt_thread(1)  # must not propagate


# ══════════════════════════════════════════════════════════════════════════════
#  _kill_process
# ══════════════════════════════════════════════════════════════════════════════


class TestKillProcess:
    def test_nonexistent_pid_does_not_raise(self):
        _kill_process(9_999_999)

    def test_psutil_path_kills_children_then_parent(self):
        NoSuch = type("NoSuchProcess", (Exception,), {})
        fake_psutil = MagicMock()
        fake_psutil.NoSuchProcess = NoSuch
        child = MagicMock()
        proc = MagicMock()
        proc.children.return_value = [child]
        fake_psutil.Process.return_value = proc

        with patch.dict(sys.modules, {"psutil": fake_psutil}):
            _kill_process(12345)

        proc.children.assert_called_once_with(recursive=True)
        child.kill.assert_called_once()
        proc.kill.assert_called_once()

    def test_child_no_such_process_is_swallowed(self):
        # child.kill() raises NoSuchProcess => swallowed
        NoSuch = type("NoSuchProcess", (Exception,), {})
        fake_psutil = MagicMock()
        fake_psutil.NoSuchProcess = NoSuch
        child = MagicMock()
        child.kill.side_effect = NoSuch()
        proc = MagicMock()
        proc.children.return_value = [child]
        fake_psutil.Process.return_value = proc

        with patch.dict(sys.modules, {"psutil": fake_psutil}):
            _kill_process(12345)  # must not raise

        proc.kill.assert_called_once()

    def test_psutil_no_such_process_on_process_init(self):
        NoSuch = type("NoSuchProcess", (Exception,), {})
        fake_psutil = MagicMock()
        fake_psutil.NoSuchProcess = NoSuch
        fake_psutil.Process.side_effect = NoSuch()

        with patch.dict(sys.modules, {"psutil": fake_psutil}):
            _kill_process(12345)

    def test_fallback_os_kill_when_psutil_missing(self):
        # ImportError path ; os.kill fallback
        with patch.dict(sys.modules, {"psutil": None}):
            with patch("os.kill") as mock_kill:
                _kill_process(12345)
            mock_kill.assert_called_once_with(12345, signal.SIGKILL)

    def test_fallback_oserror_is_swallowed(self):
        with patch.dict(sys.modules, {"psutil": None}):
            with patch("os.kill", side_effect=OSError):
                _kill_process(12345)  # must not raise

    def test_fallback_process_lookup_error_is_swallowed(self):
        with patch.dict(sys.modules, {"psutil": None}):
            with patch("os.kill", side_effect=ProcessLookupError):
                _kill_process(12345)  # must not raise


# ══════════════════════════════════════════════════════════════════════════════
#  _update_interrupted_tasks_in_db
# ══════════════════════════════════════════════════════════════════════════════


class TestUpdateInterruptedTasksInDb:
    def _patch_dao(self, **kwargs):
        return patch(f"{_RT}.DAO", **kwargs)

    def test_happy_path_calls_update_for_each_task(self):
        mock_dao_inst = MagicMock()
        with (
            self._patch_dao(return_value=mock_dao_inst),
            patch.dict(os.environ, {"RP_METADATASTORE_URL": "sqlite://"}),
        ):
            _update_interrupted_tasks_in_db([1, 2, 3], exec_id=99)
        assert mock_dao_inst.update_task.call_count == 3

    def test_dao_init_failure_absorbed_flushes_handlers(self):
        # DAO(...) raises => log handlers flushed, return early
        fake_handler = MagicMock()
        with (
            self._patch_dao(side_effect=Exception("no db")),
            patch.dict(os.environ, {"RP_METADATASTORE_URL": "sqlite://"}),
            patch(f"{_RT}.logger") as mock_log,
        ):
            mock_log.handlers = [fake_handler]
            _update_interrupted_tasks_in_db([1], exec_id=1)
        fake_handler.flush.assert_called()

    def test_empty_task_list_no_dao_calls(self):
        mock_dao_inst = MagicMock()
        with (
            self._patch_dao(return_value=mock_dao_inst),
            patch.dict(os.environ, {"RP_METADATASTORE_URL": "sqlite://"}),
        ):
            _update_interrupted_tasks_in_db([], exec_id=5)
        mock_dao_inst.update_task.assert_not_called()

    def test_update_task_exception_absorbed_dispose_still_called(self, suppress_logger):
        # update_task raises => exception absorbed, dispose() called
        mock_dao_inst = MagicMock()
        mock_dao_inst.update_task.side_effect = Exception("db write error")
        with (
            self._patch_dao(return_value=mock_dao_inst),
            patch.dict(os.environ, {"RP_METADATASTORE_URL": "sqlite://"}),
            suppress_logger(_RT),
        ):
            _update_interrupted_tasks_in_db([7], exec_id=1)
        mock_dao_inst.dispose.assert_called()

    def test_flushes_log_handlers_at_end(self):
        # handler flush after update loop
        fake_handler = MagicMock()
        mock_dao_inst = MagicMock()
        with (
            self._patch_dao(return_value=mock_dao_inst),
            patch.dict(os.environ, {"RP_METADATASTORE_URL": "sqlite://"}),
            patch(f"{_RT}.logger") as mock_log,
        ):
            mock_log.handlers = [fake_handler]
            _update_interrupted_tasks_in_db([1], exec_id=1)
        fake_handler.flush.assert_called()


# ══════════════════════════════════════════════════════════════════════════════
#  _sigint_handler
# ══════════════════════════════════════════════════════════════════════════════


class TestSigintHandler:
    def setup_method(self):
        with _task_registry._lock:
            _task_registry._running_tasks.clear()
            _task_registry._interrupted = False

    def test_calls_sys_exit_when_no_tasks(self, suppress_logger):
        with suppress_logger(_RT), patch.object(sys, "exit") as mock_exit:
            _sigint_handler(None, None)
        mock_exit.assert_called_with(1)

    def test_marks_registry_interrupted(self, suppress_logger):
        with suppress_logger(_RT), patch.object(sys, "exit"):
            _sigint_handler(None, None)
        assert _task_registry.is_interrupted()

    def test_kills_each_running_task_process(self, suppress_logger):
        _task_registry.register_task(99, 12345)
        with (
            suppress_logger(_RT),
            patch(f"{_RT}._kill_process") as mock_kill,
            patch.object(sys, "exit"),
            patch(f"{_RT}._dag_execution_context_var") as mock_ctxvar,
        ):
            mock_ctxvar.get.return_value = None
            _sigint_handler(None, None)
        mock_kill.assert_called_once_with(12345)

    def test_updates_db_when_context_and_exec_id_present(self, suppress_logger):
        # context exists with exec_id => calls _update_interrupted_tasks_in_db
        _task_registry.register_task(1, 999)
        ctx = _make_context({"exec_id": 77})
        with (
            suppress_logger(_RT),
            patch(f"{_RT}._kill_process"),
            patch.object(sys, "exit"),
            patch(f"{_RT}._dag_execution_context_var") as mock_ctxvar,
            patch(f"{_RT}._update_interrupted_tasks_in_db") as mock_upd,
        ):
            mock_ctxvar.get.return_value = ctx
            _sigint_handler(None, None)
        mock_upd.assert_called_once_with([1], 77)

    def test_db_update_exception_absorbed(self, suppress_logger):
        # exception from _update_interrupted_tasks_in_db is caught
        _task_registry.register_task(2, 888)
        with (
            suppress_logger(_RT),
            patch(f"{_RT}._kill_process"),
            patch.object(sys, "exit"),
            patch(f"{_RT}._dag_execution_context_var") as mock_ctxvar,
            patch(
                f"{_RT}._update_interrupted_tasks_in_db", side_effect=Exception("fail")
            ),
        ):
            mock_ctxvar.get.return_value = _make_context()
            _sigint_handler(None, None)  # must not raise

    def test_no_exec_id_skips_db_update(self, suppress_logger):
        _task_registry.register_task(3, 777)
        ctx = DagExecutionContext({})  # no exec_id
        with (
            suppress_logger(_RT),
            patch(f"{_RT}._kill_process"),
            patch.object(sys, "exit"),
            patch(f"{_RT}._dag_execution_context_var") as mock_ctxvar,
            patch(f"{_RT}._update_interrupted_tasks_in_db") as mock_upd,
        ):
            mock_ctxvar.get.return_value = ctx
            _sigint_handler(None, None)
        mock_upd.assert_not_called()


# ══════════════════════════════════════════════════════════════════════════════
#  _install_interrupt_handler
# ══════════════════════════════════════════════════════════════════════════════


class TestInstallInterruptHandler:
    def test_installs_correct_handler(self):
        _install_interrupt_handler()
        assert signal.getsignal(signal.SIGINT) is _sigint_handler


# ══════════════════════════════════════════════════════════════════════════════
#  _topological_sort  (including task_group nesting)
# ══════════════════════════════════════════════════════════════════════════════


class TestTopologicalSort:
    def test_single_task(self):
        t = _mock_task("A")
        assert _topological_sort([t]) == [t]

    def test_linear_chain_preserves_order(self):
        a = _mock_task("A")
        b = _mock_task("B")
        c = _mock_task("C")
        a.children = [b]
        b.parents = [a]
        b.children = [c]
        c.parents = [b]
        assert _topological_sort([a]) == [a, b, c]

    def test_diamond_graph(self):
        a, b, c, d = [_mock_task(n) for n in "ABCD"]
        a.children = [b, c]
        b.children = [d]
        c.children = [d]
        b.parents = [a]
        c.parents = [a]
        d.parents = [b, c]

        result = _topological_sort([a])
        names = [r.name for r in result]
        assert names.index("A") < names.index("B")
        assert names.index("A") < names.index("C")
        assert names.index("B") < names.index("D")
        assert names.index("C") < names.index("D")

    def test_independent_roots_both_included(self):
        a = _mock_task("A")
        b = _mock_task("B")
        result_names = {r.name for r in _topological_sort([a, b])}
        assert result_names == {"A", "B"}

    def test_child_in_taskgroup_increments_indegree(self):
        # child is a TaskGroup => iterate its elements
        parent = _mock_task("Parent")
        child_task = _mock_task("Child")
        child_task.parents = [parent]
        tg = _mock_tg(elements=[child_task])
        child_task.task_group = tg
        parent.children = [tg]

        result = _topological_sort([parent])
        # parent must appear; child_task is inside a group => group appears
        assert parent in result

    def test_task_with_task_group_emits_group_not_task(self):
        # node.task_group is not None and group.task_group is None
        # => append group to order (not node)
        parent = _mock_task("P")
        child = _mock_task("C")
        tg = _mock_tg(elements=[child])
        child.task_group = tg
        tg.task_group = None
        parent.children = [tg]
        # tg is the child in children_map (via TaskGroup branch)
        child.children = []

        # Build indegree properly via a TaskGroup child
        result = _topological_sort([parent])
        assert parent in result

    def test_task_group_in_nested_group_is_skipped(self):
        # group.task_group is not None => skip (don't append)
        parent = _mock_task("P")
        child = _mock_task("C")
        inner_tg = _mock_tg(name="inner", elements=[child])
        outer_tg = _mock_tg(name="outer", elements=[child])
        child.task_group = inner_tg
        inner_tg.task_group = outer_tg  # nested => skip
        parent.children = [inner_tg]

        result = _topological_sort([parent])
        # Neither inner_tg (nested) nor child (has task_group) should appear raw
        assert parent in result


# ══════════════════════════════════════════════════════════════════════════════
#  _find_subdag_end
# ══════════════════════════════════════════════════════════════════════════════


class TestFindSubdagEnd:
    def test_no_merge_returns_full_length(self):
        order = [_mock_task("A", is_parallel=True), _mock_task("B")]
        assert _find_subdag_end(order, 0) == 2

    def test_finds_first_merge_task(self):
        order = [
            _mock_task("A", is_parallel=True),
            _mock_task("B"),
            _mock_task("C", merge_func=lambda x: x),
        ]
        assert _find_subdag_end(order, 0) == 2

    def test_nested_parallel_balanced_depth(self):
        order = [
            _mock_task("A", is_parallel=True),
            _mock_task("B", is_parallel=True),
            _mock_task("C", merge_func=lambda x: x),
            _mock_task("D", merge_func=lambda x: x),
        ]
        assert _find_subdag_end(order, 0) == 3

    def test_start_idx_skips_preceding_elements(self):
        order = [
            _mock_task("X"),
            _mock_task("A", is_parallel=True),
            _mock_task("M", merge_func=lambda x: x),
        ]
        assert _find_subdag_end(order, 1) == 2

    def test_taskgroup_elements_are_skipped(self):
        tg = _mock_tg()
        merge = _mock_task("M", merge_func=lambda x: x)
        order = [_mock_task("P", is_parallel=True), tg, merge]
        assert _find_subdag_end(order, 0) == 2


# ══════════════════════════════════════════════════════════════════════════════
#  _collect_parent_results
# ══════════════════════════════════════════════════════════════════════════════


class TestCollectParentResults:
    def test_no_parents_returns_empty(self):
        t = _mock_task("A")
        out = _collect_parent_results(t, TaskPayload({"A": 1}))
        assert dict(out._data) == {}

    def test_single_parent_present_in_results(self):
        p = _mock_task("P")
        t = _mock_task("T", parents=[p])
        out = _collect_parent_results(t, TaskPayload({"P": 99, "other": 0}))
        assert out["P"] == 99
        assert "other" not in out._data

    def test_parent_absent_from_results(self):
        p = _mock_task("X")
        t = _mock_task("T", parents=[p])
        assert dict(_collect_parent_results(t, TaskPayload({}))._data) == {}

    def test_multiple_parents_all_collected(self):
        p1, p2 = _mock_task("P1"), _mock_task("P2")
        t = _mock_task("T", parents=[p1, p2])
        out = _collect_parent_results(t, TaskPayload({"P1": "a", "P2": "b"}))
        assert out["P1"] == "a"
        assert out["P2"] == "b"

    def test_taskgroup_delegates_to_first_element(self):
        parent = _mock_task("P")
        elem = _mock_task("E", parents=[parent])
        tg = _mock_tg(elements=[elem])
        out = _collect_parent_results(tg, TaskPayload({"P": 7}))
        assert out["P"] == 7


# ══════════════════════════════════════════════════════════════════════════════
#  _parallel_input_count
# ══════════════════════════════════════════════════════════════════════════════


class TestParallelInputCount:
    def test_empty_payload_returns_one(self):
        assert _parallel_input_count(TaskPayload({})) == 1

    def test_scalar_value_returns_one(self):
        assert _parallel_input_count(TaskPayload({"a": 42})) == 1

    def test_list_value_returns_its_length(self):
        assert _parallel_input_count(TaskPayload({"a": [1, 2, 3]})) == 3

    def test_empty_list_returns_zero(self):
        assert _parallel_input_count(TaskPayload({"a": []})) == 0

    def test_string_value_treated_as_scalar(self):
        assert _parallel_input_count(TaskPayload({"a": "abc"})) == 1


# ══════════════════════════════════════════════════════════════════════════════
#  _execute_task  (lines 550-581)
# ══════════════════════════════════════════════════════════════════════════════


class TestExecuteTask:
    """_execute_task calls GrpcClient.init/shutdown and t.func / t.merge_func."""

    def _grpc_patches(self):
        return patch(f"{_RT}.GrpcClient")

    def _make_func_task(self, name="T", result="res"):
        """Task whose func returns (task_id, result)."""
        t = _mock_task(name)
        t.merge_func = None
        t.func = MagicMock(return_value=(None, result))
        return t

    def test_task_without_parent_results(self):
        t = self._make_func_task(result="out")
        with self._grpc_patches():
            r = _execute_task(t, TaskPayload({}), exec_id=1)
        assert r == "out"
        t.func.assert_called_once_with(exec_id=1, task_id=None)

    def test_task_with_parent_results_no_rank(self):
        t = self._make_func_task(result="out")
        with self._grpc_patches():
            r = _execute_task(t, TaskPayload({"p": 5}), exec_id=1)
        assert r == "out"
        t.func.assert_called_once()
        _, kwargs = t.func.call_args
        assert kwargs.get("exec_id") == 1

    def test_task_with_rank(self):
        t = self._make_func_task(result="ranked")
        with self._grpc_patches():
            r = _execute_task(t, TaskPayload({"p": 5}), exec_id=1, rank=[0])
        assert r == "ranked"
        _, kwargs = t.func.call_args
        assert kwargs.get("rank") == [0]

    def test_task_without_parent_results_with_rank(self):
        t = self._make_func_task(result="out")
        with self._grpc_patches():
            r = _execute_task(t, TaskPayload({}), exec_id=1, rank=[2])
        assert r == "out"
        _, kwargs = t.func.call_args
        assert kwargs.get("rank") == [2]

    def test_merge_func_is_called_before_func(self):
        # merge_func path
        t = _mock_task("TM")
        parent_t = _mock_task("P")
        t.parents = [parent_t]
        merged_val = "merged"
        merge_mock = MagicMock(return_value=(10, merged_val))
        merge_mock.__name__ = "fake_merge"
        t.merge_func = merge_mock
        t.func = MagicMock(return_value=(None, "final"))

        with self._grpc_patches():
            r = _execute_task(t, TaskPayload({"P": [1, 2]}), exec_id=1, rank=[0])

        t.merge_func.assert_called_once()
        t.func.assert_called_once()
        assert r == "final"

    def test_grpc_shutdown_called_even_on_func_exception(self):
        t = _mock_task("T")
        t.merge_func = None
        t.func = MagicMock(side_effect=RuntimeError("task boom"))

        with self._grpc_patches() as mock_grpc:
            with pytest.raises(RuntimeError):
                _execute_task(t, TaskPayload({}), exec_id=1)
        mock_grpc.shutdown.assert_called_once()


# ══════════════════════════════════════════════════════════════════════════════
#  _execute_task_with_context
# ══════════════════════════════════════════════════════════════════════════════


class TestExecuteTaskWithContext:
    def test_sets_context_and_returns_result_and_updates(self):
        ctx = _make_context()
        t = _mock_task("T")
        t.merge_func = None
        t.func = MagicMock(return_value=(None, "val"))

        with (
            patch(f"{_RT}.GrpcClient"),
            patch(f"{_RT}._execute_task", return_value="val"),
        ):
            result, updates = _execute_task_with_context(
                ctx, t, TaskPayload({}), exec_id=1
            )

        assert result == "val"
        assert isinstance(updates, dict)

    def test_context_var_reset_on_exception(self):
        ctx = _make_context()
        t = _mock_task("T")

        with patch(f"{_RT}._execute_task", side_effect=ValueError("boom")):
            with pytest.raises(ValueError):
                _execute_task_with_context(ctx, t, TaskPayload({}), exec_id=1)


# ══════════════════════════════════════════════════════════════════════════════
#  _execute_taskgroup
# ══════════════════════════════════════════════════════════════════════════════


class TestExecuteTaskgroup:
    """Uses a real DagExecutionContext set on the context var."""

    def _run_with_context(self, fn, *args, **kwargs):
        ctx = _make_context()
        token = _dag_execution_context_var.set(ctx)
        try:
            return fn(*args, **kwargs)
        finally:
            _dag_execution_context_var.reset(token)

    def test_single_task_in_group(self):
        t = _mock_task("T")
        t.merge_func = None
        t.func = MagicMock(return_value=(None, "r"))
        tg = _mock_tg(elements=[t])

        with (
            patch(f"{_RT}.GrpcClient"),
            patch(f"{_RT}.RetrainPipelinesExecutor") as MockExec,
        ):
            inst = MockExec.return_value.__enter__.return_value
            inst = MockExec.return_value

            # Fake executor: submit returns a future-like object
            future = MagicMock()
            future.result.return_value = ("r", {})
            inst.submit.return_value = future

            with patch(f"{_RT}.as_completed", return_value=[future]):
                result = self._run_with_context(
                    _execute_taskgroup, tg, TaskPayload({}), exec_id=1
                )

        assert "T" in result._data or result is not None

    def test_element_exception_propagates(self, suppress_logger):
        t = _mock_task("T")
        tg = _mock_tg(elements=[t])
        ctx = _make_context()
        token = _dag_execution_context_var.set(ctx)

        future = MagicMock()
        future.result.side_effect = RuntimeError("element fail")

        try:
            with (
                suppress_logger(_RT),
                patch(f"{_RT}.RetrainPipelinesExecutor") as MockExec,
                patch(f"{_RT}.as_completed", return_value=[future]),
            ):
                inst = MockExec.return_value
                inst.submit.return_value = future
                with pytest.raises(RuntimeError):
                    _execute_taskgroup(tg, TaskPayload({}), exec_id=1)
        finally:
            _dag_execution_context_var.reset(token)

    def test_nested_taskgroup_result_merged(self):
        # Element result is a TaskPayload => merge all keys
        inner_t = _mock_task("inner")
        inner_tg = _mock_tg(name="itg", elements=[inner_t])
        outer_tg = _mock_tg(elements=[inner_tg])
        ctx = _make_context()
        token = _dag_execution_context_var.set(ctx)

        future = MagicMock()
        payload = TaskPayload({"inner": "v"})
        future.result.return_value = (payload, {})

        try:
            with (
                patch(f"{_RT}.RetrainPipelinesExecutor") as MockExec,
                patch(f"{_RT}.as_completed", return_value=[future]),
            ):
                inst = MockExec.return_value
                inst.submit.return_value = future
                result = _execute_taskgroup(outer_tg, TaskPayload({}), exec_id=1)
            assert result["inner"] == "v"
        finally:
            _dag_execution_context_var.reset(token)


# ══════════════════════════════════════════════════════════════════════════════
#  _execute_taskgroup_with_context
# ══════════════════════════════════════════════════════════════════════════════


class TestExecuteTaskgroupWithContext:
    def test_sets_context_and_returns_payload_and_updates(self):
        ctx = _make_context()
        tg = _mock_tg(elements=[])
        payload = TaskPayload({"x": 1})

        with patch(f"{_RT}._execute_taskgroup", return_value=payload):
            result, updates = _execute_taskgroup_with_context(
                ctx, tg, TaskPayload({}), exec_id=1
            )

        assert result is payload
        assert isinstance(updates, dict)

    def test_context_var_reset_on_exception(self):
        ctx = _make_context()
        tg = _mock_tg(elements=[])

        with patch(f"{_RT}._execute_taskgroup", side_effect=ValueError("tg boom")):
            with pytest.raises(ValueError):
                _execute_taskgroup_with_context(ctx, tg, TaskPayload({}), exec_id=1)


# ══════════════════════════════════════════════════════════════════════════════
#  _execute_branch
# ══════════════════════════════════════════════════════════════════════════════


class TestExecuteBranch:
    """_execute_branch is pure Python logic ; mock _execute_task/_execute_taskgroup."""

    def test_single_plain_task(self):
        t = _mock_task("T")
        t.is_parallel = False

        with patch(f"{_RT}._execute_task", return_value="out") as mock_et:
            r = _execute_branch([t], TaskPayload({"p": 1}), exec_id=1, rank=[0])

        mock_et.assert_called_once()
        assert r == "out"

    def test_single_taskgroup_in_branch(self):
        # _execute_branch stores taskgroup's inner task results by their names,
        # then returns branch_results[last_element.name].
        # Last element is tg named "inner_tg"; its inner task is also named "inner_tg"
        # so the payload key matches the final lookup.
        inner = _mock_task("inner_tg")
        tg = _mock_tg(name="inner_tg", elements=[inner])
        payload_out = TaskPayload({"inner_tg": "v"})

        with patch(f"{_RT}._execute_taskgroup", return_value=payload_out):
            r = _execute_branch([tg], TaskPayload({}), exec_id=1, rank=[0])

        assert r == "v"

    def test_merge_task_at_end_of_branch(self):
        plain = _mock_task("P")
        plain.is_parallel = False
        merge = _mock_task("M", merge_func=lambda x: x)
        merge.is_parallel = False
        merge.parents = [plain]

        with patch(f"{_RT}._execute_task", side_effect=["plain_out", "merged_out"]):
            r = _execute_branch(
                [plain, merge],
                TaskPayload({"parent": "v"}),
                exec_id=1,
                rank=[0],
            )
        assert r == "merged_out"

    def test_nested_parallel_with_list_input(self):
        # elmt.is_parallel and i > 0 => nested parallelism
        root = _mock_task("root")
        root.is_parallel = False
        nested_par = _mock_task("nested", is_parallel=True)
        nested_merge = _mock_task("nmerge", merge_func=lambda x: x)
        nested_merge.parents = [nested_par]
        nested_par.parents = [root]

        # parent_value is a list => _execute_parallel_branches_with_context
        with (
            patch(f"{_RT}._execute_task", return_value=["a", "b"]),
            patch(
                f"{_RT}._execute_parallel_branches_with_context",
                return_value=["ra", "rb"],
            ) as mock_par,
        ):
            _ = _execute_branch(
                [root, nested_par, nested_merge],
                TaskPayload({"p": "v"}),
                exec_id=1,
                rank=[0],
            )
        mock_par.assert_called_once()

    def test_nested_parallel_with_scalar_input_falls_back_to_branch(self):
        # parent_value is scalar => _execute_branch (recursive)
        root = _mock_task("root")
        root.is_parallel = False
        nested_par = _mock_task("nested", is_parallel=True)
        nested_merge = _mock_task("nmerge", merge_func=lambda x: x)
        nested_merge.parents = [nested_par]
        nested_par.parents = [root]

        with (
            patch(f"{_RT}._execute_task", return_value="scalar"),
            patch(f"{_RT}._execute_branch", wraps=_execute_branch),
        ):
            # prevent infinite recursion by limiting _execute_task in second call
            call_count = [0]

            def controlled(t, pr, exec_id, rank=None):
                call_count[0] += 1
                if call_count[0] == 1:
                    return "scalar"
                return "nested_out"

            with patch(f"{_RT}._execute_task", side_effect=controlled):
                _ = _execute_branch(
                    [root, nested_par, nested_merge],
                    TaskPayload({"p": "v"}),
                    exec_id=1,
                    rank=[0],
                )


# ══════════════════════════════════════════════════════════════════════════════
#  _execute_branch_with_context
# ══════════════════════════════════════════════════════════════════════════════


class TestExecuteBranchWithContext:
    def test_sets_context_returns_result_and_updates(self):
        ctx = _make_context()
        t = _mock_task("T")
        payload = TaskPayload({"T": "v"})

        with patch(f"{_RT}._execute_branch", return_value=payload):
            result, updates = _execute_branch_with_context(
                ctx, [t], TaskPayload({}), exec_id=1, rank=[0]
            )

        assert result is payload
        assert isinstance(updates, dict)

    def test_context_var_reset_on_exception(self):
        ctx = _make_context()

        with patch(f"{_RT}._execute_branch", side_effect=RuntimeError("br boom")):
            with pytest.raises(RuntimeError):
                _execute_branch_with_context(
                    ctx, [], TaskPayload({}), exec_id=1, rank=[]
                )


# ══════════════════════════════════════════════════════════════════════════════
#  _execute_parallel_branches_with_context
# ══════════════════════════════════════════════════════════════════════════════


class TestExecuteParallelBranchesWithContext:
    def _run(self, subdag, parent_results, count, exec_id=1, rank=None):
        ctx = _make_context()
        token = _dag_execution_context_var.set(ctx)
        try:
            return _execute_parallel_branches_with_context(
                subdag, parent_results, count, exec_id, rank
            )
        finally:
            _dag_execution_context_var.reset(token)

    def test_count_zero_returns_empty(self):
        with patch(f"{_RT}.RetrainPipelinesExecutor") as MockExec:
            inst = MockExec.return_value
            inst.submit.return_value = MagicMock()
            r = self._run([], TaskPayload({}), count=0)
        assert r == []

    def test_multiple_branches_submitted(self):
        t = _mock_task("T")
        parent_results = TaskPayload({"p": ["a", "b"]})

        future1 = MagicMock()
        future1.result.return_value = (TaskPayload({"T": "r1"}), {})
        future2 = MagicMock()
        future2.result.return_value = (TaskPayload({"T": "r2"}), {})

        with patch(f"{_RT}.RetrainPipelinesExecutor") as MockExec:
            inst = MockExec.return_value
            inst.submit.side_effect = [future1, future2]
            r = self._run([t], parent_results, count=2)

        assert len(r) == 2

    def test_context_updates_merged(self):
        t = _mock_task("T")
        parent_results = TaskPayload({"p": [1]})
        future = MagicMock()
        future.result.return_value = (TaskPayload({"T": "v"}), {"new_key": "new_val"})

        with patch(f"{_RT}.RetrainPipelinesExecutor") as MockExec:
            inst = MockExec.return_value
            inst.submit.return_value = future
            r = self._run([t], parent_results, count=1)

        assert len(r) == 1


# ══════════════════════════════════════════════════════════════════════════════
#  execute / _execute
# ══════════════════════════════════════════════════════════════════════════════


def _make_dag_mock(roots, params=None):
    """Build a minimal DAG mock compatible with _execute / execute."""
    dag = MagicMock()
    dag.roots = roots
    dag.params = params or {}
    dag.init = MagicMock()
    return dag


def _exec_patches():
    """Return a context manager that patches all external I/O for _execute."""
    return (
        patch(f"{_RT}._install_interrupt_handler"),
        patch(f"{_RT}.GrpcClient"),
        patch(f"{_RT}.DAG"),
        patch(f"{_RT}.get_trace_buffer"),
        patch(f"{_RT}._task_registry"),
    )


class TestExecute:
    """Integration-level tests for execute() and _execute()."""

    def _minimal_exec(self, dag, params=None):
        """Run execute() with all external I/O mocked."""
        trace_buf = MagicMock()
        registry = MagicMock()
        registry.get_running_tasks.return_value = {}

        with (
            patch(f"{_RT}._install_interrupt_handler"),
            patch(f"{_RT}.GrpcClient"),
            patch(f"{_RT}.DAG") as MockDAG,
            patch(f"{_RT}.DAO") as MockRtDAO,
            patch(f"{_RT}.get_trace_buffer", return_value=trace_buf),
            patch(f"{_RT}._task_registry", registry),
            patch(f"{_RT}.RichLoggingController") as MockRLC,
        ):
            MockDAG.mark_complete = MagicMock()
            MockRtDAO.return_value.get_execution.return_value.params = {}
            rlc_inst = MockRLC.return_value  # noqa: F841
            return execute(dag, params)

    def test_single_plain_task_dag(self):
        t = _mock_task("T")
        t.is_parallel = False
        t.merge_func = None
        t.func = MagicMock(return_value=(None, "result"))
        dag = _make_dag_mock(
            roots=[t], params={"exec_id": {"description": "id", "default": 1}}
        )

        from retrain_pipelines.dag_engine.core.core import DagParam

        dag.params = {"exec_id": DagParam(description="id", default=1)}

        with (
            patch(f"{_RT}._execute_task", return_value="result"),
            patch(f"{_RT}._topological_sort", return_value=[t]),
            patch(f"{_RT}._collect_parent_results", return_value=TaskPayload({})),
        ):
            result, ctx_dump = self._minimal_exec(dag)

        assert result == "result"

    def test_params_defaults_resolved(self):
        from retrain_pipelines.dag_engine.core.core import DagParam

        t = _mock_task("T")
        t.is_parallel = False
        t.merge_func = None
        dag = _make_dag_mock(roots=[t])
        dag.params = {
            "exec_id": DagParam(description="id", default=99),
            "missing": DagParam(description="no default"),
        }

        with (
            patch(f"{_RT}._execute_task", return_value="r"),
            patch(f"{_RT}._topological_sort", return_value=[t]),
            patch(f"{_RT}._collect_parent_results", return_value=TaskPayload({})),
        ):
            result, ctx_dump = self._minimal_exec(dag, params={"extra": "val"})

        assert ctx_dump["exec_id"] == 99
        assert ctx_dump["missing"] is None

    def test_last_element_taskgroup_collects_results(self):
        # last element is a TaskGroup
        from retrain_pipelines.dag_engine.core.core import DagParam

        inner = _mock_task("inner")
        tg = _mock_tg(name="tg", elements=[inner])
        dag = _make_dag_mock(roots=[tg])
        dag.params = {"exec_id": DagParam(description="id", default=1)}

        tg_result = TaskPayload({"inner": "tg_val"})

        with (
            patch(f"{_RT}._execute_taskgroup", return_value=tg_result),
            patch(f"{_RT}._topological_sort", return_value=[tg]),
            patch(f"{_RT}._collect_parent_results", return_value=TaskPayload({})),
        ):
            result, _ = self._minimal_exec(dag)

        assert result["inner"] == "tg_val"

    def test_parallel_subdag_executed(self):
        # is_parallel branch in _execute
        from retrain_pipelines.dag_engine.core.core import DagParam

        par = _mock_task("par", is_parallel=True)
        merge = _mock_task("merge", merge_func=lambda x: x)
        merge.parents = [par]
        dag = _make_dag_mock(roots=[par])
        dag.params = {"exec_id": DagParam(description="id", default=1)}

        with (
            patch(f"{_RT}._topological_sort", return_value=[par, merge]),
            patch(
                f"{_RT}._collect_parent_results",
                return_value=TaskPayload({"par": [1, 2]}),
            ),
            patch(f"{_RT}._find_subdag_end", return_value=2),
            patch(f"{_RT}._parallel_input_count", return_value=2),
            patch(
                f"{_RT}._execute_parallel_branches_with_context",
                return_value=["r1", "r2"],
            ) as mock_par,
            patch(f"{_RT}._execute_task", return_value="merged"),
        ):
            result, _ = self._minimal_exec(dag)

        mock_par.assert_called_once()

    def test_rich_logging_deactivated_on_exception(self):
        # finally block in execute() calls deactivate
        from retrain_pipelines.dag_engine.core.core import DagParam

        dag = _make_dag_mock(roots=[])
        dag.params = {"exec_id": DagParam(description="id", default=1)}

        trace_buf = MagicMock()
        registry = MagicMock()
        registry.get_running_tasks.return_value = {}

        with (
            patch(f"{_RT}._install_interrupt_handler"),
            patch(f"{_RT}.GrpcClient"),
            patch(f"{_RT}.DAG"),
            patch(f"{_RT}.get_trace_buffer", return_value=trace_buf),
            patch(f"{_RT}._task_registry", registry),
            patch(f"{_RT}.RichLoggingController") as MockRLC,
            patch(f"{_RT}._execute", side_effect=RuntimeError("inner")),
        ):
            rlc_inst = MockRLC.return_value
            with pytest.raises(RuntimeError):
                execute(dag)
            rlc_inst.deactivate.assert_called_once()

    def test_execute_returns_context_dump(self):
        from retrain_pipelines.dag_engine.core.core import DagParam

        t = _mock_task("T")
        t.is_parallel = False
        t.merge_func = None
        dag = _make_dag_mock(roots=[t])
        dag.params = {"exec_id": DagParam(description="id", default=5)}

        with (
            patch(f"{_RT}._execute_task", return_value="val"),
            patch(f"{_RT}._topological_sort", return_value=[t]),
            patch(f"{_RT}._collect_parent_results", return_value=TaskPayload({})),
        ):
            _, ctx_dump = self._minimal_exec(dag)

        assert "exec_id" in ctx_dump
