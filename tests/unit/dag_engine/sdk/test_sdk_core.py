"""Unit tests for retrain_pipelines.dag_engine.sdk.core

Uses file-based SQLite (via the ``isolated_async_dao`` fixture) so that
``AsyncDAO`` instances created internally by the SDK share the same
database. No DAO mocking or patching is used ; all DB interactions are real.

The ``metadata_root`` fixture (parametrized for local and S3 backends)
is used throughout so that disk-artifact resolution is exercised against
both storage backends.
"""

import json
from datetime import datetime
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session

from retrain_pipelines.dag_engine.db.model import (
    Execution as ExecutionDbModel,
    Task as TaskDbModel,
    TaskType as TaskTypeDbModel,
    TaskGroup as TaskGroupDbModel,
    TaskContextAttr as TaskContextAttrDbModel,
    TaskPayloadAttr as TaskPayloadAttrDbModel,
)
from retrain_pipelines.dag_engine.sdk import core
from retrain_pipelines.dag_engine.sdk.core import (
    Execution,
    ExecutionParams,
    Task,
    TaskExitContext,
    TaskExitPayload,
    TaskGroup,
    TaskGroupRank,
    TaskGroupRanks,
    shutdown_async_sdk,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
# IMPORTANT: We use SQLAlchemy ORM (`Session.add()`) for all seed inserts.
# `disable_all_dao_listeners` dynamically disables all DAO event listeners,
# making ORM inserts safe. Raw SQL `text()` queries must be avoided as
# they bypass SQLAlchemy's `TypeDecorator` logic. This causes `JSON` and
# `Uuid` columns to be serialized as plain strings without the exact
# formatting `AsyncDAO` expects in its `WHERE` clauses, leading to silent
# filter mismatches and KeyErrors in the SDK.
# ---------------------------------------------------------------------------

_NOW = datetime(2024, 6, 1, 12, 0, 0)
_LATER = datetime(2024, 6, 1, 13, 0, 0)


def _make_execution(metadata_root, **kwargs):
    """Return a minimal valid Execution instance."""
    defaults = dict(
        id=1,
        name="pipe",
        metadata_root=metadata_root,
        artifacts_store_root="/tmp/artifacts",
        username="username",
        start_timestamp=_NOW,
        end_timestamp=None,
        success=True,
    )
    defaults.update(kwargs)
    return Execution(**defaults)


def _make_task(metadata_root, **kwargs):
    """Return a minimal valid Task instance."""
    defaults = dict(
        id=10,
        exec_id=7,
        name="my_task",
        metadata_root=metadata_root,
        start_timestamp=_NOW,
        end_timestamp=None,
        success=True,
    )
    defaults.update(kwargs)
    return Task(**defaults)


def _seed_execution(engine, metadata_root, params=None, end_ts=_LATER):
    """Insert an execution row via ORM. Returns the execution id."""
    with Session(engine) as session:
        execution = ExecutionDbModel(
            name="pipe",
            username="user",
            start_timestamp=_NOW,
            end_timestamp=end_ts,
            metadata_root=metadata_root,
            artifacts_store_root="/tmp/artifacts",
            params=json.loads(params) if isinstance(params, str) else params,
        )
        session.add(execution)
        session.commit()
        session.refresh(execution)
        return execution.id


def _seed_execution_with_task(
    engine, metadata_root, task_name="step", failed=False, params=None
):
    """Seed an execution, tasktype, and task via ORM inserts.

    Returns (exec_id, task_id).
    """
    exec_id = _seed_execution(engine, metadata_root, params=params)
    tt_uuid = uuid4()
    failed_val = failed if failed is not None else None
    end_ts = _LATER if failed is not None else None

    with Session(engine) as session:
        tasktype = TaskTypeDbModel(
            uuid=tt_uuid,
            exec_id=exec_id,
            order=0,
            name=task_name,
            is_parallel=False,
            children=[],
        )
        session.add(tasktype)

        task = TaskDbModel(
            tasktype_uuid=tt_uuid,
            exec_id=exec_id,
            start_timestamp=_NOW,
            end_timestamp=end_ts,
            failed=failed_val,
        )
        session.add(task)
        session.commit()
        task_id = task.id

    return exec_id, task_id


def _seed_context_attr(
    engine,
    task_id,
    attr_name,
    inline_val=None,
    disk_ref=None,
    sha=None,
    eTAG="dummy_eTAG",
):
    """Insert a task_context_attrs row via ORM."""
    with Session(engine) as session:
        attr = TaskContextAttrDbModel(
            task_id=task_id,
            attr_name=attr_name,
            eTAG=eTAG,
            sha=sha,
            disk_ref=disk_ref,
            inline_val=inline_val,
        )
        session.add(attr)
        session.commit()


def _seed_task_payload(engine, task_id, inline_val=None, disk_ref=None, sha=None):
    """Insert a task_payload_attrs row via ORM."""
    with Session(engine) as session:
        payload = TaskPayloadAttrDbModel(
            task_id=task_id, sha=sha, disk_ref=disk_ref, inline_val=inline_val
        )
        session.add(payload)
        session.commit()


def _seed_taskgroup_with_task(
    engine, metadata_root, task_name="tg_task", failed=False, rank="UNSET"
):
    """Seed an execution, taskgroup, tasktype, and task via ORM inserts.

    Returns (exec_id, task_id, tg_uuid, tt_uuid).
    """
    exec_id = _seed_execution(engine, metadata_root)
    tt_uuid = uuid4()
    tg_uuid = uuid4()
    if rank == "UNSET":
        rank = [1]
    failed_val = failed if failed is not None else None
    end_ts = _LATER if failed is not None else None

    with Session(engine) as session:
        taskgroup = TaskGroupDbModel(
            uuid=tg_uuid,
            exec_id=exec_id,
            order=0,
            name="tg",
            elements=[str(tt_uuid)],
        )
        session.add(taskgroup)

        tasktype = TaskTypeDbModel(
            uuid=tt_uuid,
            exec_id=exec_id,
            order=0,
            name=task_name,
            is_parallel=True,
            children=[],
            taskgroup_uuid=tg_uuid,
        )
        session.add(tasktype)

        task = TaskDbModel(
            tasktype_uuid=tt_uuid,
            exec_id=exec_id,
            start_timestamp=_NOW,
            end_timestamp=end_ts,
            failed=failed_val,
            rank=rank,
        )
        session.add(task)
        session.commit()
        task_id = task.id

    return exec_id, task_id, str(tg_uuid), str(tt_uuid)


# ---------------------------------------------------------------------------
# _run_async
# ---------------------------------------------------------------------------


class TestRunAsync:
    """Cover the module-level _run_async helper and event loop lifecycle."""

    def test_shutdown_async_sdk_stops_and_joins(self):
        """Covers shutdown_async_sdk() when loop is running and thread is alive."""
        core._start_loop()
        loop = core._loop
        thread = core._loop_thread

        assert loop is not None
        assert thread is not None
        assert loop.is_running()
        assert thread.is_alive()

        shutdown_async_sdk()

        assert not loop.is_running()
        assert not thread.is_alive()

        # Reset global state so other tests can start it cleanly if needed
        core._loop = None
        core._loop_thread = None

    def test_shutdown_async_sdk_when_loop_not_started(self):
        """Covers branch where loop was never started (idempotent no-op).

        Ensures shutdown_async_sdk does not raise when _loop and
        _loop_thread are both None, covering the short-circuit
        condition branches.
        """
        # Ensure clean state - if loop was started, stop it first
        if core._loop is not None and core._loop.is_running():
            shutdown_async_sdk()
        core._loop = None
        core._loop_thread = None

        # Should be a safe no-op
        shutdown_async_sdk()
        assert core._loop is None
        assert core._loop_thread is None


# ---------------------------------------------------------------------------
# Execution model
# ---------------------------------------------------------------------------


class TestExecutionModel:
    """Cover Execution model construction and simple methods."""

    def test_fields_set_correctly(self, metadata_root):
        exec_ = _make_execution(metadata_root, end_timestamp=_LATER)

        assert exec_.id == 1
        assert exec_.name == "pipe"
        assert exec_.start_timestamp == _NOW
        assert exec_.end_timestamp == _LATER
        assert exec_.success is True

    def test_completed_true_when_end_timestamp_set(self, metadata_root):
        exec_ = _make_execution(metadata_root, end_timestamp=_LATER)
        assert exec_.completed() is True

    def test_completed_false_when_end_timestamp_none(self, metadata_root):
        exec_ = _make_execution(metadata_root, end_timestamp=None)
        assert exec_.completed() is False

    def test_elements_iterator_not_implemented(self, metadata_root):
        """Covers the NotImplementedError guard in elements_iterator."""
        exec_ = _make_execution(metadata_root)
        with pytest.raises(NotImplementedError):
            exec_.elements_iterator()


class TestExecutionGetLatest:
    """Cover Execution.get_latest()."""

    def test_get_latest_success(self, isolated_async_dao, metadata_root):
        """Covers branch where the latest execution is found."""
        exec_id, _ = _seed_execution_with_task(
            isolated_async_dao, metadata_root, task_name="step", failed=False
        )
        exec_ = Execution.get_latest("pipe")
        assert exec_.id == exec_id
        assert exec_.name == "pipe"

    def test_get_latest_success_only(self, isolated_async_dao, metadata_root):
        """Covers branch where success_only=True filters to successful executions."""
        # Seed a failed execution first
        _seed_execution_with_task(
            isolated_async_dao, metadata_root, task_name="step", failed=True
        )
        # Seed a successful execution
        exec_id, _ = _seed_execution_with_task(
            isolated_async_dao, metadata_root, task_name="step", failed=False
        )
        exec_ = Execution.get_latest("pipe", success_only=True)
        assert exec_.id == exec_id
        assert exec_.success is True

    def test_get_latest_not_found(self, isolated_async_dao, metadata_root):
        """Covers branch where no execution matches the name, raising KeyError."""
        _seed_execution_with_task(
            isolated_async_dao, metadata_root, task_name="step", failed=False
        )
        with pytest.raises(KeyError, match="No execution found with name=missing_pipe"):
            Execution.get_latest("missing_pipe")


class TestExecutionGetTasksWithName:
    """Cover Execution.get_tasks_with_name()."""

    def test_get_tasks_with_name_returns_tasks(self, isolated_async_dao, metadata_root):
        """Covers branch where tasks are returned and failed=False => success=True."""
        exec_id, _ = _seed_execution_with_task(
            isolated_async_dao, metadata_root, task_name="preprocess", failed=False
        )

        exec_ = _make_execution(metadata_root, id=exec_id)
        tasks = exec_.get_tasks_with_name("preprocess")

        assert len(tasks) == 1
        assert tasks[0].name == "preprocess"
        assert tasks[0].success is True

    def test_get_tasks_with_name_failed_none(self, isolated_async_dao, metadata_root):
        """Covers branch where failed=None => success defaults to True."""
        exec_id, _ = _seed_execution_with_task(
            isolated_async_dao, metadata_root, task_name="train", failed=None
        )

        exec_ = _make_execution(metadata_root, id=exec_id)
        tasks = exec_.get_tasks_with_name("train")

        assert tasks[0].success is True

    def test_get_tasks_with_name_empty(self, isolated_async_dao, metadata_root):
        """Covers branch where no tasks match the given name."""
        exec_id, _ = _seed_execution_with_task(
            isolated_async_dao, metadata_root, task_name="preprocess", failed=False
        )

        exec_ = _make_execution(metadata_root, id=exec_id)
        tasks = exec_.get_tasks_with_name("nonexistent")

        assert tasks == []


class TestExecutionGetTaskById:
    """Cover Execution.get_by_id() and Execution.get_task_by_id().

    Exercises dao.engine.dispose() and the KeyError branch of get_task_by_id.
    """

    def test_get_by_id_success(self, isolated_async_dao, metadata_root):
        """Covers branch where execution is found and task is retrieved."""
        exec_id, task_id = _seed_execution_with_task(
            isolated_async_dao, metadata_root, task_name="task_name", failed=False
        )

        exec_ = Execution.get_by_id(exec_id)
        assert exec_.id == exec_id
        assert exec_.success is True

        task = exec_.get_task_by_id(task_id)
        assert task.id == task_id
        assert task.name == "task_name"
        assert task.success is True

    def test_get_by_id_raises_keyerror(self, isolated_async_dao):
        """Covers branch where execution is not found."""
        with pytest.raises(KeyError, match="No execution found with id=99"):
            Execution.get_by_id(99)

    def test_get_task_by_id_keyerror(self, isolated_async_dao, metadata_root):
        """Covers branch where task is not found."""
        exec_id, _ = _seed_execution_with_task(
            isolated_async_dao, metadata_root, task_name="task_name", failed=False
        )

        exec_ = Execution.get_by_id(exec_id)
        with pytest.raises(KeyError, match="No task found with id=999"):
            exec_.get_task_by_id(999)


class TestExecutionGetParams:
    """Cover Execution.get_params()."""

    def test_get_params_returns_instance(self, isolated_async_dao, metadata_root):
        params_json = json.dumps({"a": {"default": 1}})
        exec_id = _seed_execution(isolated_async_dao, metadata_root, params=params_json)

        exec_ = _make_execution(metadata_root, id=exec_id)
        params = exec_.get_params()

        assert isinstance(params, ExecutionParams)
        assert len(params) == 1
        assert params["a"] == 1

    def test_get_params_empty_dict(self, isolated_async_dao, metadata_root):
        exec_id = _seed_execution(isolated_async_dao, metadata_root, params=None)

        exec_ = _make_execution(metadata_root, id=exec_id)
        params = exec_.get_params()

        assert len(params) == 0


class TestExecutionGetTaskgroupsWithName:
    """Cover Execution.get_taskgroups_with_name()."""

    def test_get_taskgroups_with_name_success(self, isolated_async_dao, metadata_root):
        """Covers branch where taskgroup is found and ranks are returned."""
        exec_id, _, tg_uuid, _ = _seed_taskgroup_with_task(
            isolated_async_dao, metadata_root
        )
        exec_ = _make_execution(metadata_root, id=exec_id)
        tgr = exec_.get_taskgroups_with_name("tg")

        assert isinstance(tgr, TaskGroupRanks)
        assert tgr.taskgroup.uuid == tg_uuid
        assert tgr.ranks == [[1]]

    def test_get_taskgroups_with_name_not_found(
        self, isolated_async_dao, metadata_root
    ):
        """Covers branch where taskgroup is not found, triggering the except TypeError block.

        The DAO returns None for a missing taskgroup, which causes tuple
        unpacking to raise TypeError. This test verifies that the TypeError
        is caught and None is returned gracefully.
        """
        exec_id, _ = _seed_execution_with_task(
            isolated_async_dao, metadata_root, task_name="step", failed=False
        )
        exec_ = _make_execution(metadata_root, id=exec_id)
        tgr = exec_.get_taskgroups_with_name("missing_tg")

        assert tgr is None


class TestExecutionGetTaskgroupForRank:
    """Cover Execution.get_taskgroup_for_rank()."""

    def test_get_taskgroup_for_rank_success(self, isolated_async_dao, metadata_root):
        """Covers branch where taskgroup and rank are found."""
        exec_id, _, _, _ = _seed_taskgroup_with_task(
            isolated_async_dao, metadata_root, rank=[1, 2]
        )
        exec_ = _make_execution(metadata_root, id=exec_id)
        tgr = exec_.get_taskgroup_for_rank("tg", [1, 2])

        assert isinstance(tgr, TaskGroupRank)
        assert tgr.rank == [1, 2]

    def test_get_taskgroup_for_rank_rank_not_found(
        self, isolated_async_dao, metadata_root
    ):
        """Covers branch where taskgroup is found but rank is not."""
        exec_id, _, _, _ = _seed_taskgroup_with_task(
            isolated_async_dao, metadata_root, rank=[1, 2]
        )
        exec_ = _make_execution(metadata_root, id=exec_id)
        tgr = exec_.get_taskgroup_for_rank("tg", [9, 9])

        assert tgr is None

    def test_get_taskgroup_for_rank_empty_rank_success(
        self, isolated_async_dao, metadata_root
    ):
        """Covers branch where requested rank is empty and taskgroup has NULL ranks.

        Verifies the path where target_rank is None and ranks equals [None],
        satisfying the rank_exists check.
        """
        exec_id, _, _, _ = _seed_taskgroup_with_task(
            isolated_async_dao, metadata_root, rank=None
        )
        exec_ = _make_execution(metadata_root, id=exec_id)
        tgr = exec_.get_taskgroup_for_rank("tg", [])

        assert isinstance(tgr, TaskGroupRank)
        assert tgr.rank is None

    def test_get_taskgroup_for_rank_empty_rank_not_found(
        self, isolated_async_dao, metadata_root
    ):
        """Covers branch where requested rank is empty but taskgroup has non-None ranks.

        When target_rank is None (empty input) but the taskgroup's ranks
        list contains non-None entries, rank_exists evaluates to False
        and None is returned with a warning.
        """
        exec_id, _, _, _ = _seed_taskgroup_with_task(
            isolated_async_dao, metadata_root, rank=[1, 2]
        )
        exec_ = _make_execution(metadata_root, id=exec_id)
        tgr = exec_.get_taskgroup_for_rank("tg", [])

        assert tgr is None

    def test_get_taskgroup_for_rank_not_found(self, isolated_async_dao, metadata_root):
        """Covers branch where taskgroup is not found, returning None gracefully.

        Requires the suggested `try...except TypeError` fix in `sdk.core` to
        handle the DAO returning `None` for a missing taskgroup, allowing the
        `if not tg:` block to execute and return `None`.
        """
        exec_id, _ = _seed_execution_with_task(
            isolated_async_dao, metadata_root, task_name="step", failed=False
        )
        exec_ = _make_execution(metadata_root, id=exec_id)
        tgr = exec_.get_taskgroup_for_rank("missing_tg", [1])
        assert tgr is None


class TestExecutionGetAttrs:
    """Cover Execution.get_attrs()."""

    def test_get_attrs_success(self, isolated_async_dao, metadata_root):
        """Covers branch where execution attrs are found."""
        exec_id, task_id = _seed_execution_with_task(
            isolated_async_dao, metadata_root, task_name="step", failed=False
        )
        _seed_context_attr(isolated_async_dao, task_id, "my_attr", inline_val="val")

        exec_ = _make_execution(metadata_root, id=exec_id)
        ctx = exec_.get_attrs()

        assert isinstance(ctx, TaskExitContext)
        assert ctx["my_attr"] == "val"

    def test_get_attrs_empty(self, isolated_async_dao, metadata_root):
        """Covers branch where execution completed but no context attrs exist."""
        exec_id, task_id = _seed_execution_with_task(
            isolated_async_dao, metadata_root, task_name="step", failed=False
        )
        exec_ = _make_execution(metadata_root, id=exec_id)
        ctx = exec_.get_attrs()

        assert isinstance(ctx, TaskExitContext)
        assert len(ctx) == 0

    def test_get_attrs_none(self, isolated_async_dao, metadata_root):
        """Covers branch where execution did not complete successfully (None).

        Seeds an execution with end_timestamp=None (still running), which
        causes the DAO to return None from get_execution_latest_context_attrs,
        triggering the warning-and-return-None path.
        """
        exec_id = _seed_execution(isolated_async_dao, metadata_root, end_ts=None)
        exec_ = _make_execution(
            metadata_root, id=exec_id, end_timestamp=None, success=False
        )
        ctx = exec_.get_attrs()

        assert ctx is None


# ---------------------------------------------------------------------------
# Task model
# ---------------------------------------------------------------------------


class TestTaskModel:
    """Cover Task model construction and field defaults."""

    def test_task_fields(self, metadata_root):
        task = _make_task(metadata_root)
        assert task.id == 10
        assert task.name == "my_task"
        assert task.start_timestamp == _NOW
        assert task.end_timestamp is None
        assert task.success is True

    def test_task_success_none(self, metadata_root):
        task = _make_task(metadata_root, success=None)
        assert task.success is None

    def test_task_with_end_timestamp(self, metadata_root):
        task = _make_task(metadata_root, end_timestamp=_LATER)
        assert task.end_timestamp == _LATER

    def test_task_failed_true_success_false(self, metadata_root):
        task = _make_task(metadata_root, success=False)
        assert task.success is False


class TestTaskGetExitContext:
    """Cover Task.get_exit_context()."""

    def test_get_exit_context_with_attrs(self, isolated_async_dao, metadata_root):
        """Covers branch where context attrs exist for the task."""
        exec_id, task_id = _seed_execution_with_task(
            isolated_async_dao, metadata_root, task_name="step", failed=False
        )
        _seed_context_attr(isolated_async_dao, task_id, "my_attr", inline_val="val")

        exec_ = Execution.get_by_id(exec_id)
        task = exec_.get_task_by_id(task_id)
        ctx = task.get_exit_context()

        assert isinstance(ctx, TaskExitContext)
        assert ctx["my_attr"] == "val"

    def test_get_exit_context_empty(self, isolated_async_dao, metadata_root):
        """Covers branch where no context attrs exist for the task."""
        exec_id, task_id = _seed_execution_with_task(
            isolated_async_dao, metadata_root, task_name="step", failed=False
        )

        exec_ = Execution.get_by_id(exec_id)
        task = exec_.get_task_by_id(task_id)
        ctx = task.get_exit_context()

        assert isinstance(ctx, TaskExitContext)
        assert len(ctx) == 0


class TestTaskGetExitPayload:
    """Cover Task.get_exit_payload()."""

    def test_get_exit_payload_none(self, isolated_async_dao, metadata_root):
        """Covers branch where payload is not found (None)."""
        exec_id, task_id = _seed_execution_with_task(
            isolated_async_dao, metadata_root, task_name="my_task", failed=False
        )

        exec_ = Execution.get_by_id(exec_id)
        task = exec_.get_task_by_id(task_id)
        payload = task.get_exit_payload()

        assert isinstance(payload, TaskExitPayload)
        assert payload["my_task"] is None


# ---------------------------------------------------------------------------
# TaskGroup model
# ---------------------------------------------------------------------------


class TestTaskGroup:
    """Cover TaskGroup equality."""

    def test_eq(self, metadata_root):
        tg1 = TaskGroup(
            exec_id=1, metadata_root=metadata_root, uuid="123", name="tg", elements=[]
        )
        tg2 = TaskGroup(
            exec_id=1, metadata_root=metadata_root, uuid="123", name="tg", elements=[]
        )
        tg3 = TaskGroup(
            exec_id=1, metadata_root=metadata_root, uuid="456", name="tg", elements=[]
        )

        assert tg1 == tg2
        assert tg1 != tg3
        assert tg1 != "not_a_tg"


class TestTaskGroupRanks:
    """Cover TaskGroupRanks methods."""

    def test_parse_ranks_strings(self, metadata_root):
        tgr = TaskGroupRanks(
            exec_id=1,
            metadata_root=metadata_root,
            uuid="123",
            name="tg",
            elements=[],
            ranks=["[1, 2]", "[3, 4]"],
        )
        assert tgr.ranks == [[1, 2], [3, 4]]

    def test_parse_ranks_lists(self, metadata_root):
        tgr = TaskGroupRanks(
            exec_id=1,
            metadata_root=metadata_root,
            uuid="123",
            name="tg",
            elements=[],
            ranks=[[1, 2], [3, 4]],
        )
        assert tgr.ranks == [[1, 2], [3, 4]]

    def test_parse_ranks_empty_string(self, metadata_root):
        """Covers branch where a rank string is empty (e.g. '[]'), yielding an empty list."""
        tgr = TaskGroupRanks(
            exec_id=1,
            metadata_root=metadata_root,
            uuid="123",
            name="tg",
            elements=[],
            ranks=["[]"],
        )
        assert tgr.ranks == [[]]

    def test_parse_ranks_not_list(self, metadata_root):
        """Covers branch where ranks input is not a list (e.g., None)."""
        assert TaskGroupRanks.parse_ranks(None) is None

    def test_iter(self, metadata_root):
        tgr = TaskGroupRanks(
            exec_id=1,
            metadata_root=metadata_root,
            uuid="123",
            name="tg",
            elements=[],
            ranks=[[1, 2]],
        )
        ranks = [tgr_rank.rank for tgr_rank in tgr]
        assert ranks == [[1, 2]]

    def test_iter_empty(self, metadata_root):
        """Covers branch where ranks is an empty list, yielding a single None rank."""
        tgr = TaskGroupRanks(
            exec_id=1,
            metadata_root=metadata_root,
            uuid="123",
            name="tg",
            elements=[],
            ranks=[],
        )
        ranks = [tgr_rank.rank for tgr_rank in tgr]
        assert ranks == [None]

    def test_getitem(self, metadata_root):
        tgr = TaskGroupRanks(
            exec_id=1,
            metadata_root=metadata_root,
            uuid="123",
            name="tg",
            elements=[],
            ranks=[[1, 2], [3, 4]],
        )
        assert tgr[0].rank == [1, 2]
        assert isinstance(tgr[0:1], list)
        assert tgr[0:1][0].rank == [1, 2]

    def test_getitem_empty(self, metadata_root):
        tgr = TaskGroupRanks(
            exec_id=1,
            metadata_root=metadata_root,
            uuid="123",
            name="tg",
            elements=[],
            ranks=[],
        )
        assert tgr[0].rank is None

    def test_len(self, metadata_root):
        tgr = TaskGroupRanks(
            exec_id=1,
            metadata_root=metadata_root,
            uuid="123",
            name="tg",
            elements=[],
            ranks=[[1, 2], [3, 4]],
        )
        assert len(tgr) == 2

    def test_get_rank(self, metadata_root):
        tgr = TaskGroupRanks(
            exec_id=1,
            metadata_root=metadata_root,
            uuid="123",
            name="tg",
            elements=[],
            ranks=[[1, 2], [3, 4]],
        )
        assert tgr.get_rank([1, 2]).rank == [1, 2]
        assert tgr.get_rank([9, 9]) is None

    def test_get_rank_empty(self, metadata_root):
        tgr = TaskGroupRanks(
            exec_id=1,
            metadata_root=metadata_root,
            uuid="123",
            name="tg",
            elements=[],
            ranks=[],
        )
        assert tgr.get_rank([1, 2]).rank is None


class TestTaskGroupRank:
    """Cover TaskGroupRank methods."""

    def test_parse_rank_string(self, metadata_root):
        tg = TaskGroup(
            exec_id=1, metadata_root=metadata_root, uuid="123", name="tg", elements=[]
        )
        tgr = TaskGroupRank(taskgroup=tg, rank="[1, 2]")
        assert tgr.rank == [1, 2]

    def test_parse_rank_list(self, metadata_root):
        tg = TaskGroup(
            exec_id=1, metadata_root=metadata_root, uuid="123", name="tg", elements=[]
        )
        tgr = TaskGroupRank(taskgroup=tg, rank=[1, 2])
        assert tgr.rank == [1, 2]

    def test_parse_rank_empty_string(self, metadata_root):
        """Covers branch where rank string is empty (e.g. '[]'), yielding an empty list."""
        tg = TaskGroup(
            exec_id=1, metadata_root=metadata_root, uuid="123", name="tg", elements=[]
        )
        tgr = TaskGroupRank(taskgroup=tg, rank="[]")
        assert tgr.rank == []

    def test_parse_rank_not_string_or_list(self, metadata_root):
        """Covers branch where rank is not a string or list (e.g., None)."""
        tg = TaskGroup(
            exec_id=1, metadata_root=metadata_root, uuid="123", name="tg", elements=[]
        )
        tgr = TaskGroupRank(taskgroup=tg, rank=None)
        assert tgr.rank is None

    def test_eq(self, metadata_root):
        tg1 = TaskGroup(
            exec_id=1, metadata_root=metadata_root, uuid="123", name="tg", elements=[]
        )
        tg2 = TaskGroup(
            exec_id=1, metadata_root=metadata_root, uuid="123", name="tg", elements=[]
        )
        tgr1 = TaskGroupRank(taskgroup=tg1, rank=[1, 2])
        tgr2 = TaskGroupRank(taskgroup=tg2, rank=[1, 2])
        tgr3 = TaskGroupRank(taskgroup=tg2, rank=[3, 4])

        assert tgr1 == tgr2
        assert tgr1 != tgr3
        assert tgr1 != "not_a_tgr"

    def test_get_exit_context_empty(self, isolated_async_dao, metadata_root):
        """Covers branch where no tasks are found for the taskgroup."""
        exec_id, _ = _seed_execution_with_task(
            isolated_async_dao, metadata_root, task_name="step", failed=False
        )
        tg = TaskGroup(
            exec_id=exec_id,
            metadata_root=metadata_root,
            uuid="123",
            name="missing_tg",
            elements=[],
        )
        tgr = TaskGroupRank(taskgroup=tg, rank=[1])

        ctx = tgr.get_exit_context()
        assert len(ctx) == 0

    def test_get_exit_context_with_attrs(self, isolated_async_dao, metadata_root):
        """Covers success path where member tasks are found and their context attrs are merged.

        Seeds a real taskgroup with one member task that has a context attr,
        then uses the SDK's own ``get_taskgroups_with_name`` and ``get_rank``
        methods to fetch the ``TaskGroupRank`` instance. This ensures the
        object is perfectly aligned with the DB state and tests the full SDK
        flow end-to-end.
        """
        exec_id, task_id, _, _ = _seed_taskgroup_with_task(
            isolated_async_dao, metadata_root, task_name="tg_task", rank=[1, 2]
        )
        _seed_context_attr(
            isolated_async_dao, task_id, "merged_attr", inline_val="val1"
        )

        exec_ = _make_execution(metadata_root, id=exec_id)
        tgr_list = exec_.get_taskgroups_with_name("tg")
        assert tgr_list is not None
        tgr = tgr_list.get_rank([1, 2])
        assert tgr is not None

        ctx = tgr.get_exit_context()
        assert isinstance(ctx, TaskExitContext)
        assert ctx["merged_attr"] == "val1"

    def test_get_exit_payload_empty(self, isolated_async_dao, metadata_root):
        """Covers branch where no tasks are found for the taskgroup."""
        exec_id, _ = _seed_execution_with_task(
            isolated_async_dao, metadata_root, task_name="step", failed=False
        )
        tg = TaskGroup(
            exec_id=exec_id,
            metadata_root=metadata_root,
            uuid="123",
            name="missing_tg",
            elements=[],
        )
        tgr = TaskGroupRank(taskgroup=tg, rank=[1])

        payload = tgr.get_exit_payload()
        assert bool(payload) is False

    def test_get_exit_payload_with_data(self, isolated_async_dao, metadata_root):
        """Covers success path where member tasks and their payloads are found.

        Seeds a real taskgroup with one member task that has an inline
        payload, then uses the SDK's own ``get_taskgroups_with_name`` and
        ``get_rank`` methods to fetch the ``TaskGroupRank`` instance. This
        ensures the object is perfectly aligned with the DB state and tests
        the full SDK flow end-to-end.
        """
        exec_id, task_id, _, _ = _seed_taskgroup_with_task(
            isolated_async_dao, metadata_root, task_name="tg_task", rank=[1, 2]
        )
        _seed_task_payload(isolated_async_dao, task_id, inline_val=42)

        exec_ = _make_execution(metadata_root, id=exec_id)
        tgr_list = exec_.get_taskgroups_with_name("tg")
        assert tgr_list is not None
        tgr = tgr_list.get_rank([1, 2])
        assert tgr is not None

        payload = tgr.get_exit_payload()
        assert isinstance(payload, TaskExitPayload)
        assert payload["tg_task"] == 42
