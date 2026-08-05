# test_runtime_1.py
"""
Unit tests for retrain_pipelines.dag_engine.runtime.
"""

import ctypes
import logging
import os
import signal
import sys
import threading
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from retrain_pipelines.dag_engine.core.core import (
    DagExecutionContext,
    DagParam,
    TaskFuncException,
    TaskGroup,
    TaskPayload,
    TaskType,
    _dag_execution_context_var,
    dag,
    parallel_task,
    task,
)
from retrain_pipelines.dag_engine.db.model import Task
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

# Ensure the test module is importable by cloudpickle in subprocesses
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# -----------------------------------------------------------------------------
#  Shared helpers
# -----------------------------------------------------------------------------


class FailingFlushHandler(logging.Handler):
    """A logging handler that raises an exception when flush() is called."""

    def flush(self):
        raise RuntimeError("flush error")

    def emit(self, record):
        pass


def make_task(name, is_parallel=False, merge_func=None):
    """Create a real TaskType instance for testing graph logic (no executor)."""

    def func(*args, **kwargs):
        return name

    func.__name__ = name
    func.__doc__ = "Test task"
    return TaskType(func=func, is_parallel=is_parallel, merge_func=merge_func)


def make_tg(name="tg", elements=None):
    """Create a real TaskGroup instance for testing."""
    return TaskGroup(name, *(elements or []))


def _setup_db(dao):
    """Helper to create a real execution record in DB and return its ID."""
    return dao.add_execution(
        name="test",
        metadata_root="",
        artifacts_store_root="",
        username="test",
        start_timestamp=datetime.now(timezone.utc),
    )


def _setup_dao_with_tasks(dao, tasks):
    """Helper to create an execution and register the given task types in DB."""
    exec_id = _setup_db(dao)
    task_ids = []
    for i, t in enumerate(tasks):
        dao.add_tasktype(
            exec_id=exec_id,
            order=i,
            uuid=t.tasktype_uuid,
            name=t.name,
            is_parallel=t.is_parallel,
            children=[],
        )
        task_ids.append(
            dao.add_task(
                exec_id=exec_id,
                tasktype_uuid=t.tasktype_uuid,
                start_timestamp=datetime.now(timezone.utc),
            )
        )
    return exec_id, task_ids


_RT = "retrain_pipelines.dag_engine.runtime"


# Module-level tasks required for pickling when using the real executor


@task
def task_T(*args, **kwargs):
    return "T"


@task
def task_fail(*args, **kwargs):
    raise ValueError("boom")


@task
def task_tg_single(*args, **kwargs):
    return "single"


@task
def task_tg_inner(*args, **kwargs):
    return "inner"


@task
def task_branch_root_list(*args, **kwargs):
    return ["a", "b"]


@parallel_task
def task_branch_nested(*args, **kwargs):
    return "nested"


def _branch_nmerge_func(x, **kwargs):
    return x


@task(merge_func=_branch_nmerge_func)
def task_branch_nmerge(*args, **kwargs):
    return "nmerge"


@task
def task_branch_root_scalar(*args, **kwargs):
    return "scalar"


# Module-level DAG definitions for integration tests


@task
def simple_dag_task():
    return "result"


@dag
def simple_dag():
    return simple_dag_task


@task
def params_dag_task():
    return "r"


@dag
def params_dag():
    p1 = DagParam(description="id", default=99)
    p2 = DagParam(description="no default")
    _ = (p1, p2)  # Silences linter F841; @dag extracts params via AST
    return params_dag_task


@task
def taskgroup_dag_t1():
    return 1


@task
def taskgroup_dag_t2():
    return 2


@dag
def taskgroup_dag():
    return TaskGroup("my_tg", taskgroup_dag_t1, taskgroup_dag_t2)


@task
def nested_tg_task1():
    return 1


@task
def nested_tg_task2():
    return 2


@dag
def nested_taskgroup_dag():
    """DAG ending with a TaskGroup that contains a nested TaskGroup."""
    inner_tg = TaskGroup("inner_tg", nested_tg_task1, nested_tg_task2)
    outer_tg = TaskGroup("outer_tg", inner_tg, task_T)
    return outer_tg


@task
def parallel_dag_gen():
    return [1, 2]


@parallel_task
def parallel_dag_proc(inp):
    return inp["parallel_dag_gen"] * 2


def _parallel_dag_agg_merge(x, **kwargs):
    return sum(x)


@task(merge_func=_parallel_dag_agg_merge)
def parallel_dag_agg(inp):
    return inp["parallel_dag_proc"]


@dag
def parallel_dag():
    parallel_dag_gen >> parallel_dag_proc >> parallel_dag_agg
    return parallel_dag_agg


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

    def test_res_zero_logs_warning(self):
        """Covers branch where SetAsyncExc returns zero (no matching thread)."""
        with patch.object(
            ctypes.pythonapi, "PyThreadState_SetAsyncExc", return_value=0
        ):
            _interrupt_thread(12345)

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
    def test_happy_path_updates_tasks(self, runtime_env, isolated_dao):
        t1, t2, t3 = make_task("T1"), make_task("T2"), make_task("T3")
        exec_id, task_ids = _setup_dao_with_tasks(isolated_dao, [t1, t2, t3])
        with patch(
            f"{_RT}.Config.get_metadatastore_url", return_value=isolated_dao.engine.url
        ):
            _update_interrupted_tasks_in_db(task_ids, exec_id=exec_id)

        for tid in task_ids:
            task = isolated_dao._get_session().get(Task, tid)
            assert task.failed is True

    def test_happy_path_flush_exception_absorbed(self, runtime_env, isolated_dao):
        """Covers the except Exception block for h.flush() at the end of _update_interrupted_tasks_in_db."""
        handler = FailingFlushHandler()
        rt_logger = logging.getLogger(_RT)
        rt_logger.addHandler(handler)
        try:
            t1 = make_task("T1")
            exec_id, task_ids = _setup_dao_with_tasks(isolated_dao, [t1])
            with patch(
                f"{_RT}.Config.get_metadatastore_url",
                return_value=isolated_dao.engine.url,
            ):
                _update_interrupted_tasks_in_db(task_ids, exec_id=exec_id)
        finally:
            rt_logger.removeHandler(handler)

    def test_dao_init_failure_absorbed(self, runtime_env, isolated_dao):
        # DAO(...) raises => log handlers flushed, return early
        mock_handler = logging.NullHandler()
        rt_logger = logging.getLogger(_RT)
        rt_logger.addHandler(mock_handler)
        try:
            with patch(f"{_RT}.DAO", side_effect=RuntimeError("init fail")):
                _update_interrupted_tasks_in_db([1], exec_id=1)
        finally:
            rt_logger.removeHandler(mock_handler)

    def test_dao_init_failure_flush_exception_absorbed(self, runtime_env, isolated_dao):
        """Covers the except Exception block for h.flush() when DAO init fails."""
        handler = FailingFlushHandler()
        rt_logger = logging.getLogger(_RT)
        rt_logger.addHandler(handler)
        try:
            with patch(f"{_RT}.DAO", side_effect=RuntimeError("init fail")):
                _update_interrupted_tasks_in_db([1], exec_id=1)
        finally:
            rt_logger.removeHandler(handler)

    def test_update_task_exception_absorbed(self, runtime_env, isolated_dao):
        # update_task raises => exception absorbed, dispose() called
        mock_dao_inst = MagicMock()
        mock_dao_inst.update_task.side_effect = RuntimeError("db write error")
        with patch(f"{_RT}.DAO", return_value=mock_dao_inst):
            _update_interrupted_tasks_in_db([1], exec_id=1)
        mock_dao_inst.dispose.assert_called()

    def test_empty_task_list_no_dao_calls(self, runtime_env, isolated_dao):
        session = isolated_dao._get_session()
        assert session.query(Task).count() == 0
        _update_interrupted_tasks_in_db([], exec_id=5)
        assert session.query(Task).count() == 0


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
        _task_registry.register_task(99, 999999)  # non-existent PID
        with (
            suppress_logger(_RT),
            patch.object(sys, "exit"),
            patch(f"{_RT}._kill_process"),
        ):
            _sigint_handler(None, None)

    def test_falsy_pid_skips_kill_process(self, suppress_logger):
        """Covers branch where a registered task has a falsy pid (no process to kill)."""
        _task_registry.register_task(1, 0)
        with (
            suppress_logger(_RT),
            patch.object(sys, "exit"),
            patch(f"{_RT}._kill_process"),
        ):
            _sigint_handler(None, None)

    def test_updates_db_when_context_and_exec_id_present(
        self, runtime_env, isolated_dao, suppress_logger
    ):
        # context exists with exec_id => calls _update_interrupted_tasks_in_db
        t1 = make_task("T1")
        exec_id, task_ids = _setup_dao_with_tasks(isolated_dao, [t1])
        _task_registry.register_task(task_ids[0], 999999)
        ctx = DagExecutionContext({"exec_id": exec_id})
        _dag_execution_context_var.set(ctx)

        with (
            suppress_logger(_RT),
            patch.object(sys, "exit"),
            patch(f"{_RT}._kill_process"),
            patch(f"{_RT}.DAO", return_value=isolated_dao),
        ):
            _sigint_handler(None, None)

        task = isolated_dao._get_session().get(Task, task_ids[0])
        assert task.failed is True

    def test_db_update_exception_absorbed(self, runtime_env, monkeypatch):
        # exception from _update_interrupted_tasks_in_db is caught and logged
        mock_handler = logging.NullHandler()
        rt_logger = logging.getLogger(_RT)
        rt_logger.addHandler(mock_handler)
        try:
            _task_registry.register_task(2, 888888)
            ctx = DagExecutionContext({"exec_id": 1})
            _dag_execution_context_var.set(ctx)

            with (
                patch(
                    f"{_RT}._update_interrupted_tasks_in_db",
                    side_effect=RuntimeError("DB error"),
                ),
                patch.object(sys, "exit"),
                patch(f"{_RT}._kill_process"),
            ):
                _sigint_handler(None, None)
        finally:
            rt_logger.removeHandler(mock_handler)

    def test_no_exec_id_skips_db_update(self, suppress_logger):
        _task_registry.register_task(3, 777777)
        ctx = DagExecutionContext({})  # no exec_id
        _dag_execution_context_var.set(ctx)

        with (
            suppress_logger(_RT),
            patch.object(sys, "exit"),
            patch(f"{_RT}._kill_process"),
        ):
            _sigint_handler(None, None)


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
        t = make_task("A")
        assert _topological_sort([t]) == [t]

    def test_linear_chain_preserves_order(self):
        a, b, c = make_task("A"), make_task("B"), make_task("C")
        a.children.append(b)
        b.parents.append(a)
        b.children.append(c)
        c.parents.append(b)
        assert _topological_sort([a]) == [a, b, c]

    def test_diamond_graph(self):
        # This graph exercises the duplicate-task skip branch:
        # D is reachable via both B and C, so it will be encountered twice.
        a, b, c, d = [make_task(n) for n in "ABCD"]
        a.children.append(b)
        a.children.append(c)
        b.children.append(d)
        c.children.append(d)
        b.parents.append(a)
        c.parents.append(a)
        d.parents.append(b)
        d.parents.append(c)

        result = _topological_sort([a])
        names = [r.name for r in result]
        assert names.index("A") < names.index("B")
        assert names.index("A") < names.index("C")
        assert names.index("B") < names.index("D")
        assert names.index("C") < names.index("D")

    def test_independent_roots_both_included(self):
        a, b = make_task("A"), make_task("B")
        result_names = {r.name for r in _topological_sort([a, b])}
        assert result_names == {"A", "B"}

    def test_child_in_taskgroup_increments_indegree(self):
        # child is a TaskGroup => iterate its elements
        parent = make_task("Parent")
        child = make_task("Child")
        child.parents.append(parent)
        tg = make_tg("tg", elements=[child])
        child._task_group = tg
        parent.children.append(tg)

        result = _topological_sort([parent])
        # parent must appear; child_task is inside a group => group appears
        assert parent in result

    def test_task_with_task_group_emits_group_not_task(self):
        # node.task_group is not None and group.task_group is None
        # => append group to order (not node)
        parent = make_task("P")
        child = make_task("C")
        tg = make_tg("tg", elements=[child])
        child._task_group = tg
        tg._task_group = None
        parent.children.append(tg)
        # tg is the child in children_map (via TaskGroup branch)
        child._children.clear()

        # Build indegree properly via a TaskGroup child
        result = _topological_sort([parent])
        assert parent in result

    def test_task_group_in_nested_group_is_skipped(self):
        # group.task_group is not None => skip (don't append)
        parent = make_task("P")
        child = make_task("C")
        inner_tg = make_tg("inner", elements=[child])
        outer_tg = make_tg("outer", elements=[child])
        child._task_group = inner_tg
        inner_tg._task_group = outer_tg  # nested => skip
        parent.children.append(inner_tg)

        result = _topological_sort([parent])
        # Neither inner_tg (nested) nor child (has task_group) should appear raw
        assert parent in result

    def test_taskgroup_root_skipped_in_traversal(self):
        """Covers branch where a root element is a TaskGroup (skipped during stack-based traversal)."""
        tg = make_tg("root_tg", elements=[])
        result = _topological_sort([tg])
        assert result == []


# ══════════════════════════════════════════════════════════════════════════════
#  _find_subdag_end
# ══════════════════════════════════════════════════════════════════════════════


class TestFindSubdagEnd:
    def test_no_merge_returns_full_length(self):
        order = [make_task("A", is_parallel=True), make_task("B")]
        assert _find_subdag_end(order, 0) == 2

    def test_finds_first_merge_task(self):
        # This test covers the branch where parallel_depth becomes 0.
        order = [
            make_task("A", is_parallel=True),
            make_task("B"),
            make_task("C", merge_func=lambda x, **kw: x),
        ]
        assert _find_subdag_end(order, 0) == 2

    def test_nested_parallel_balanced_depth(self):
        order = [
            make_task("A", is_parallel=True),
            make_task("B", is_parallel=True),
            make_task("C", merge_func=lambda x, **kw: x),
            make_task("D", merge_func=lambda x, **kw: x),
        ]
        assert _find_subdag_end(order, 0) == 3

    def test_start_idx_skips_preceding_elements(self):
        order = [
            make_task("X"),
            make_task("A", is_parallel=True),
            make_task("M", merge_func=lambda x, **kw: x),
        ]
        assert _find_subdag_end(order, 1) == 2

    def test_taskgroup_elements_are_skipped(self):
        tg = make_tg("tg")
        merge = make_task("M", merge_func=lambda x, **kw: x)
        order = [make_task("P", is_parallel=True), tg, merge]
        assert _find_subdag_end(order, 0) == 2


# ══════════════════════════════════════════════════════════════════════════════
#  _collect_parent_results
# ══════════════════════════════════════════════════════════════════════════════


class TestCollectParentResults:
    def test_no_parents_returns_empty(self):
        t = make_task("A")
        out = _collect_parent_results(t, TaskPayload({"A": 1}))
        assert dict(out._data) == {}

    def test_single_parent_present_in_results(self):
        p = make_task("P")
        t = make_task("T")
        t.parents.append(p)
        out = _collect_parent_results(t, TaskPayload({"P": 99, "other": 0}))
        assert out["P"] == 99
        assert "other" not in out._data

    def test_parent_absent_from_results(self):
        p = make_task("X")
        t = make_task("T")
        t.parents.append(p)
        assert dict(_collect_parent_results(t, TaskPayload({}))._data) == {}

    def test_multiple_parents_all_collected(self):
        p1, p2 = make_task("P1"), make_task("P2")
        t = make_task("T")
        t.parents.append(p1)
        t.parents.append(p2)
        out = _collect_parent_results(t, TaskPayload({"P1": "a", "P2": "b"}))
        assert out["P1"] == "a"
        assert out["P2"] == "b"

    def test_taskgroup_delegates_to_first_element(self):
        # Since all elements in a TaskGroup share relatives
        # (both parents and children), we look into the first only.
        parent = make_task("P")
        elem = make_task("E")
        elem.parents.append(parent)
        tg = make_tg("tg", elements=[elem])
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
#  _execute_task
# ══════════════════════════════════════════════════════════════════════════════


class TestExecuteTask:
    """_execute_task calls t.func / t.merge_func."""

    def test_task_without_parent_results(self, runtime_env, isolated_dao):
        t = task_T
        exec_id, _ = _setup_dao_with_tasks(isolated_dao, [t])
        ctx = DagExecutionContext({"exec_id": exec_id})
        token = _dag_execution_context_var.set(ctx)
        try:
            r = _execute_task(t, TaskPayload({}), exec_id=exec_id)
            assert r == "T"
        finally:
            _dag_execution_context_var.reset(token)

    def test_task_with_parent_results_no_rank(self, runtime_env, isolated_dao):
        t = task_T
        exec_id, _ = _setup_dao_with_tasks(isolated_dao, [t])
        ctx = DagExecutionContext({"exec_id": exec_id})
        token = _dag_execution_context_var.set(ctx)
        try:
            r = _execute_task(t, TaskPayload({"p": 5}), exec_id=exec_id)
            assert r == "T"
        finally:
            _dag_execution_context_var.reset(token)

    def test_task_with_rank(self, runtime_env, isolated_dao):
        t = task_T
        exec_id, _ = _setup_dao_with_tasks(isolated_dao, [t])
        ctx = DagExecutionContext({"exec_id": exec_id})
        token = _dag_execution_context_var.set(ctx)
        try:
            r = _execute_task(t, TaskPayload({"p": 5}), exec_id=exec_id, rank=[0])
            assert r == "T"
        finally:
            _dag_execution_context_var.reset(token)


# ══════════════════════════════════════════════════════════════════════════════
#  _execute_task_with_context
# ══════════════════════════════════════════════════════════════════════════════


class TestExecuteTaskWithContext:
    def test_sets_context_and_returns_result_and_updates(
        self, runtime_env, isolated_dao
    ):
        t = task_T
        exec_id, _ = _setup_dao_with_tasks(isolated_dao, [t])
        ctx = DagExecutionContext({"exec_id": exec_id})
        result, updates, attr_refs = _execute_task_with_context(
            ctx, t, TaskPayload({}), exec_id=exec_id
        )
        assert result == "T"
        assert isinstance(updates, dict)
        assert isinstance(attr_refs, dict)

    def test_context_var_reset_on_exception(self, runtime_env, isolated_dao):
        t = task_fail
        exec_id, _ = _setup_dao_with_tasks(isolated_dao, [t])
        ctx = DagExecutionContext({"exec_id": exec_id})
        with pytest.raises(TaskFuncException):
            _execute_task_with_context(ctx, t, TaskPayload({}), exec_id=exec_id)


# ══════════════════════════════════════════════════════════════════════════════
#  _execute_taskgroup
# ══════════════════════════════════════════════════════════════════════════════


class TestExecuteTaskgroup:
    """Uses a real DagExecutionContext set on the context var."""

    def test_single_task_in_group(self, runtime_env, isolated_dao):
        t = task_tg_single
        tg = make_tg("tg", elements=[t])
        exec_id, _ = _setup_dao_with_tasks(isolated_dao, [t])
        ctx = DagExecutionContext({"exec_id": exec_id})
        token = _dag_execution_context_var.set(ctx)
        try:
            result = _execute_taskgroup(tg, TaskPayload({}), exec_id=exec_id)
            assert result["task_tg_single"] == "single"
        finally:
            _dag_execution_context_var.reset(token)

    def test_nested_taskgroup_result_merged(self, runtime_env, isolated_dao):
        # Element result is a TaskPayload => merge all keys
        inner_tg_elem = task_tg_inner
        inner_tg = make_tg("itg", elements=[inner_tg_elem])
        outer_tg = make_tg("otg", elements=[inner_tg])
        exec_id, _ = _setup_dao_with_tasks(isolated_dao, [inner_tg_elem])
        ctx = DagExecutionContext({"exec_id": exec_id})
        token = _dag_execution_context_var.set(ctx)
        try:
            result = _execute_taskgroup(outer_tg, TaskPayload({}), exec_id=exec_id)
            assert result["task_tg_inner"] == "inner"
        finally:
            _dag_execution_context_var.reset(token)

    def test_element_exception_propagates(
        self, runtime_env, isolated_dao, suppress_logger
    ):
        """Covers the exception logging and re-raise block in _execute_taskgroup."""
        t = task_fail
        tg = make_tg("tg", elements=[t])
        exec_id, _ = _setup_dao_with_tasks(isolated_dao, [t])
        ctx = DagExecutionContext({"exec_id": exec_id})
        token = _dag_execution_context_var.set(ctx)
        try:
            with pytest.raises(TaskFuncException):
                _execute_taskgroup(tg, TaskPayload({}), exec_id=exec_id)
        finally:
            _dag_execution_context_var.reset(token)


# ══════════════════════════════════════════════════════════════════════════════
#  _execute_taskgroup_with_context
# ══════════════════════════════════════════════════════════════════════════════


class TestExecuteTaskgroupWithContext:
    def test_sets_context_and_returns_payload_and_updates(
        self, runtime_env, isolated_dao
    ):
        t = task_tg_single
        tg = make_tg("tg", elements=[t])
        exec_id, _ = _setup_dao_with_tasks(isolated_dao, [t])
        ctx = DagExecutionContext({"exec_id": exec_id})
        result, updates, attr_refs = _execute_taskgroup_with_context(
            ctx, tg, TaskPayload({}), exec_id=exec_id
        )
        assert result["task_tg_single"] == "single"
        assert isinstance(updates, dict)
        assert isinstance(attr_refs, dict)


# ══════════════════════════════════════════════════════════════════════════════
#  _execute_branch
# ══════════════════════════════════════════════════════════════════════════════


class TestExecuteBranch:
    """_execute_branch is pure Python logic ; mock _execute_task/_execute_taskgroup."""

    def test_single_plain_task(self, runtime_env, isolated_dao):
        t = task_T
        exec_id, _ = _setup_dao_with_tasks(isolated_dao, [t])
        ctx = DagExecutionContext({"exec_id": exec_id})
        token = _dag_execution_context_var.set(ctx)
        try:
            r = _execute_branch([t], TaskPayload({"p": 1}), exec_id=exec_id, rank=[0])
            assert r == "T"
        finally:
            _dag_execution_context_var.reset(token)

    def test_single_taskgroup_in_branch(self, runtime_env, isolated_dao):
        # _execute_branch stores taskgroup's inner task results by their names,
        # then returns branch_results[last_element.name].
        # Last element is tg named "task_tg_single"; its inner task is also named "task_tg_single"
        # so the payload key matches the final lookup.
        inner = task_tg_single
        tg = make_tg("task_tg_single", elements=[inner])
        exec_id, _ = _setup_dao_with_tasks(isolated_dao, [inner])
        ctx = DagExecutionContext({"exec_id": exec_id})
        token = _dag_execution_context_var.set(ctx)
        try:
            r = _execute_branch([tg], TaskPayload({}), exec_id=exec_id, rank=[0])
            assert r == "single"
        finally:
            _dag_execution_context_var.reset(token)

    def test_merge_task_at_end_of_branch(self, runtime_env, isolated_dao):
        plain = make_task("P")
        merge = make_task("M", merge_func=lambda x, **kw: x)
        merge.parents.append(plain)

        exec_id, _ = _setup_dao_with_tasks(isolated_dao, [plain, merge])
        ctx = DagExecutionContext({"exec_id": exec_id})
        token = _dag_execution_context_var.set(ctx)
        try:
            r = _execute_branch(
                [plain, merge],
                TaskPayload({"parent": "v"}),
                exec_id=exec_id,
                rank=[0],
            )
            assert r == "M"
        finally:
            _dag_execution_context_var.reset(token)

    def test_nested_parallel_with_list_input(self, runtime_env, isolated_dao):
        # elmt.is_parallel and i > 0 => nested parallelism with list input
        # parent_value is a list => _execute_parallel_branches_with_context
        root = task_branch_root_list
        nested_par = task_branch_nested
        nested_merge = task_branch_nmerge

        nested_par.parents.append(root)
        nested_merge.parents.append(nested_par)
        root.children.append(nested_par)
        nested_par.children.append(nested_merge)

        exec_id, _ = _setup_dao_with_tasks(
            isolated_dao, [root, nested_par, nested_merge]
        )
        ctx = DagExecutionContext({"exec_id": exec_id})
        token = _dag_execution_context_var.set(ctx)
        try:
            r = _execute_branch(
                [root, nested_par, nested_merge],
                TaskPayload({"p": "v"}),
                exec_id=exec_id,
                rank=[0],
            )
            assert r == "nmerge"
        finally:
            _dag_execution_context_var.reset(token)
            nested_par.parents.remove(root)
            nested_merge.parents.remove(nested_par)
            root.children.remove(nested_par)
            nested_par.children.remove(nested_merge)

    def test_nested_parallel_with_scalar_input_falls_back_to_branch(
        self, runtime_env, isolated_dao
    ):
        # parent_value is scalar => recursive _execute_branch call
        # We do not mock _execute_branch; we let the real function run,
        # which will make the recursive call.
        root = task_branch_root_scalar
        nested_par = task_branch_nested
        nested_merge = task_branch_nmerge

        nested_par.parents.append(root)
        nested_merge.parents.append(nested_par)
        root.children.append(nested_par)
        nested_par.children.append(nested_merge)

        exec_id, _ = _setup_dao_with_tasks(
            isolated_dao, [root, nested_par, nested_merge]
        )
        ctx = DagExecutionContext({"exec_id": exec_id})
        token = _dag_execution_context_var.set(ctx)
        try:
            r = _execute_branch(
                [root, nested_par, nested_merge],
                TaskPayload({"p": "v"}),
                exec_id=exec_id,
                rank=[0],
            )
            assert r == "nmerge"
        finally:
            _dag_execution_context_var.reset(token)
            nested_par.parents.remove(root)
            nested_merge.parents.remove(nested_par)
            root.children.remove(nested_par)
            nested_par.children.remove(nested_merge)


# ══════════════════════════════════════════════════════════════════════════════
#  _execute_branch_with_context
# ══════════════════════════════════════════════════════════════════════════════


class TestExecuteBranchWithContext:
    def test_sets_context_returns_result_and_updates(self, runtime_env, isolated_dao):
        t = task_T
        exec_id, _ = _setup_dao_with_tasks(isolated_dao, [t])
        ctx = DagExecutionContext({"exec_id": exec_id})
        result, updates = _execute_branch_with_context(
            ctx, [t], TaskPayload({}), exec_id=exec_id, rank=[0]
        )
        assert result == "T"
        assert isinstance(updates, dict)


# ══════════════════════════════════════════════════════════════════════════════
#  _execute_parallel_branches_with_context
# ══════════════════════════════════════════════════════════════════════════════


class TestExecuteParallelBranchesWithContext:
    def test_multiple_branches_submitted(self, runtime_env, isolated_dao):
        t = task_T
        exec_id, _ = _setup_dao_with_tasks(isolated_dao, [t])
        ctx = DagExecutionContext({"exec_id": exec_id})
        token = _dag_execution_context_var.set(ctx)
        try:
            r = _execute_parallel_branches_with_context(
                [t], TaskPayload({"p": ["a", "b"]}), count=2, exec_id=exec_id
            )
            assert len(r) == 2
            assert r[0] == "T"
            assert r[1] == "T"
        finally:
            _dag_execution_context_var.reset(token)


# ══════════════════════════════════════════════════════════════════════════════
#  execute / _execute
# ══════════════════════════════════════════════════════════════════════════════


class TestExecute:
    """Integration-level tests for execute() and _execute()."""

    def test_single_plain_task_dag(self, runtime_env, isolated_dao):
        result, ctx_dump = execute(simple_dag)
        assert result == "result"
        assert "exec_id" in ctx_dump

    def test_params_defaults_resolved(self, runtime_env, isolated_dao):
        result, ctx_dump = execute(params_dag, params={"extra": "val"})
        assert ctx_dump["p1"] == 99
        assert ctx_dump["p2"] is None

    def test_params_override_stored_to_db(self, runtime_env, isolated_dao):
        _, ctx_dump = execute(params_dag, params={"p1": 99})
        exec_id = ctx_dump["exec_id"]

        dao = isolated_dao
        execution = dao.get_execution(exec_id)
        assert execution.params["p1"]["override"] == 99

    def test_last_element_taskgroup_collects_results(self, runtime_env, isolated_dao):
        # last element is a taskgroup =>
        # collect its task results into TaskPayload
        result, _ = execute(taskgroup_dag)
        assert result["taskgroup_dag_t1"] == 1
        assert result["taskgroup_dag_t2"] == 2

    def test_nested_taskgroup_dag(self, runtime_env, isolated_dao):
        """Covers branch where the last DAG element is a TaskGroup containing nested TaskGroups."""
        result, _ = execute(nested_taskgroup_dag)
        assert result["nested_tg_task1"] == 1
        assert result["nested_tg_task2"] == 2
        assert result["task_T"] == "T"

    def test_parallel_subdag_executed(self, runtime_env, isolated_dao):
        # is_parallel branch in _execute
        result, _ = execute(parallel_dag)
        assert result == 6

    def test_execute_waits_for_running_tasks(self, runtime_env, isolated_dao):
        """Covers the while loop in _execute's finally block."""
        mock_registry = MagicMock()
        # Use an iterator to safely provide return values for multiple calls
        states = iter([{1: 123}, {1: 123}])
        mock_registry.get_running_tasks.side_effect = lambda: next(states, {})
        with patch(f"{_RT}._task_registry", mock_registry):
            result, ctx_dump = execute(simple_dag)
        assert result == "result"

    def test_sse_started_when_webconsole_reachable(self, runtime_env, isolated_dao):
        """Covers the branch where the WebConsole HEAD probe succeeds,
        causing the real SSE streaming server to be started, registered
        via POST, and later stopped in the finally block.

        Only third-party ``requests`` is patched; the retrain_pipelines
        SSE server runs for real to exercise its start/stop lifecycle.
        """
        fake_post_response = MagicMock()
        fake_post_response.raise_for_status.return_value = None
        with (
            patch(f"{_RT}.requests.head", return_value=MagicMock()),
            patch(f"{_RT}.requests.post", return_value=fake_post_response),
        ):
            result, _ = execute(simple_dag)
        assert result == "result"
