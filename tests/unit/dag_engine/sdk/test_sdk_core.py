"""Unit & Integration tests for retrain_pipelines.dag_engine.sdk.core

Uses file-based SQLite (via the ``isolated_async_dao`` fixture) so that
``AsyncDAO`` instances created internally by the SDK share the same
database. No DAO mocking or patching is used ; all DB interactions are real.

The ``metadata_root`` fixture (parametrized for local and S3 backends)
is used throughout so that disk-artifact resolution is exercised against
both storage backends.
"""

import json
from datetime import datetime
from uuid import UUID, uuid4

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
from retrain_pipelines.dag_engine.sdk.core import (
    Execution,
    ExecutionParams,
    RankNotFoundException,
    Task,
    TaskExitContext,
    TaskExitPayload,
    TaskGroup,
    TaskGroupRank,
    TaskGroupRanks,
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
    engine, metadata_root, task_name="step", failed=False, params=None, rank=None
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
            rank=rank,
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


def _seed_nested_taskgroups(engine, metadata_root, rank=None):
    """Seed an execution with a nested taskgroup (outer contains inner).

    ``outer_tg.elements`` references the inner taskgroup UUID rather than a
    TaskType UUID, exercising the DAO's recursive de-nesting. The leaf task
    hangs off the inner taskgroup and carries both a context attr and a
    payload so that ``TaskGroupRank.get_exit_context`` and
    ``get_exit_payload`` can be tested against actual nested resolution.

    Returns (exec_id, task_id, outer_tg_uuid, inner_tg_uuid).
    """
    exec_id = _seed_execution(engine, metadata_root)
    inner_tt_uuid = uuid4()
    inner_tg_uuid = uuid4()
    outer_tg_uuid = uuid4()
    failed_val = None
    end_ts = None

    with Session(engine) as session:
        inner_taskgroup = TaskGroupDbModel(
            uuid=inner_tg_uuid,
            exec_id=exec_id,
            order=0,
            name="inner_tg",
            elements=[str(inner_tt_uuid)],
        )
        session.add(inner_taskgroup)

        outer_taskgroup = TaskGroupDbModel(
            uuid=outer_tg_uuid,
            exec_id=exec_id,
            order=1,
            name="outer_tg",
            elements=[str(inner_tg_uuid)],
        )
        session.add(outer_taskgroup)

        leaf_tasktype = TaskTypeDbModel(
            uuid=inner_tt_uuid,
            exec_id=exec_id,
            order=0,
            name="leaf_task",
            is_parallel=True,
            children=[],
            taskgroup_uuid=inner_tg_uuid,
        )
        session.add(leaf_tasktype)

        leaf_task = TaskDbModel(
            tasktype_uuid=inner_tt_uuid,
            exec_id=exec_id,
            start_timestamp=_NOW,
            end_timestamp=end_ts,
            failed=failed_val,
            rank=rank,
        )
        session.add(leaf_task)
        session.commit()
        task_id = leaf_task.id

        session.add(
            TaskContextAttrDbModel(
                task_id=task_id,
                attr_name="nested_attr",
                eTAG="dummy_eTAG",
                sha=None,
                disk_ref=None,
                inline_val="nested_val",
            )
        )
        session.add(
            TaskPayloadAttrDbModel(
                task_id=task_id,
                sha=None,
                disk_ref=None,
                inline_val=99,
            )
        )
        session.commit()

    return exec_id, task_id, str(outer_tg_uuid), str(inner_tg_uuid)


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


class TestExecutionGetTasksWithUuid:
    """Cover Execution.get_tasks_with_uuid().

    Placed right after the name-based selector: both are the task-listing
    entry points of Execution, ahead of the single-task lookups built on
    the same row-to-Task mapping.
    """

    def test_get_tasks_with_uuid_returns_tasks(self, isolated_async_dao, metadata_root):
        """Covers the str auto-cast and the row-to-Task mapping."""
        exec_id, task_id = _seed_execution_with_task(
            isolated_async_dao, metadata_root, task_name="step", failed=False
        )
        with Session(isolated_async_dao) as session:
            tt_uuid = (
                session.query(TaskTypeDbModel).filter_by(exec_id=exec_id).one().uuid
            )

        exec_ = _make_execution(metadata_root, id=exec_id)
        tasks = exec_.get_tasks_with_uuid(str(tt_uuid))

        assert len(tasks) == 1
        assert tasks[0].id == task_id
        assert tasks[0].name == "step"
        assert tasks[0].success is True

    def test_get_tasks_with_uuid_empty(self, isolated_async_dao, metadata_root):
        """Covers the early return when no task matches the given UUID."""
        exec_id, _ = _seed_execution_with_task(
            isolated_async_dao, metadata_root, task_name="step", failed=False
        )
        exec_ = _make_execution(metadata_root, id=exec_id)

        assert exec_.get_tasks_with_uuid(uuid4()) == []


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


class TestExecutionGetTask:
    """Cover Execution.get_task()."""

    def test_get_task_rejects_both_selectors(self, metadata_root):
        """Covers the mutually-exclusive selector guard, ahead of any DB call."""
        exec_ = _make_execution(metadata_root)
        with pytest.raises(
            AttributeError,
            match=r"Provide either 'tasktype_name' or 'tasktype_uuid', not both\.",
        ):
            exec_.get_task(tasktype_name="step", tasktype_uuid=uuid4())

    def test_get_task_requires_a_selector(self, metadata_root):
        """Covers the missing-selector guard, ahead of any DB call."""
        exec_ = _make_execution(metadata_root)
        with pytest.raises(
            AttributeError,
            match=r"Provide either 'tasktype_name' or 'tasktype_uuid'\.",
        ):
            exec_.get_task()

    def test_get_task_success_default_rank(self, isolated_async_dao, metadata_root):
        """Covers branch where the task is found on the main DAG spine (rank=None)."""
        exec_id, task_id = _seed_execution_with_task(
            isolated_async_dao, metadata_root, task_name="step", failed=False
        )
        exec_ = _make_execution(metadata_root, id=exec_id)
        task = exec_.get_task("step")

        assert task.id == task_id
        assert task.name == "step"
        assert task.rank is None
        assert task.success is True

    def test_get_task_success_with_rank(self, isolated_async_dao, metadata_root):
        """Covers branch where a non-empty rank matches a seeded ranked task."""
        exec_id, task_id = _seed_execution_with_task(
            isolated_async_dao,
            metadata_root,
            task_name="step",
            failed=False,
            rank=[1, 2],
        )
        exec_ = _make_execution(metadata_root, id=exec_id)
        task = exec_.get_task("step", rank=[1, 2])

        assert task.id == task_id
        assert task.rank == [1, 2]

    def test_get_task_empty_rank_normalizes_to_none(
        self, isolated_async_dao, metadata_root
    ):
        """Covers branch where an empty rank list is normalized to None."""
        exec_id, task_id = _seed_execution_with_task(
            isolated_async_dao, metadata_root, task_name="step", failed=False
        )
        exec_ = _make_execution(metadata_root, id=exec_id)
        task = exec_.get_task("step", rank=[])

        assert task.id == task_id
        assert task.rank is None

    def test_get_task_success_with_uuid(self, isolated_async_dao, metadata_root):
        """Covers the uuid selector, which also labels failures with the UUID."""
        exec_id, task_id = _seed_execution_with_task(
            isolated_async_dao, metadata_root, task_name="step", failed=False
        )
        with Session(isolated_async_dao) as session:
            tt_uuid = (
                session.query(TaskTypeDbModel).filter_by(exec_id=exec_id).one().uuid
            )

        exec_ = _make_execution(metadata_root, id=exec_id)
        task = exec_.get_task(tasktype_uuid=tt_uuid)

        assert task.id == task_id
        assert task.name == "step"

    def test_get_task_rank_not_found(self, isolated_async_dao, metadata_root):
        """Covers branch where no task matches the requested rank, raising RankNotFoundException."""
        exec_id, _ = _seed_execution_with_task(
            isolated_async_dao, metadata_root, task_name="step", failed=False
        )
        exec_ = _make_execution(metadata_root, id=exec_id)
        with pytest.raises(
            RankNotFoundException,
            match=r"No task 'step' found at rank \[9, 9\].*Available ranks: \[None\]",
        ):
            exec_.get_task("step", rank=[9, 9])

    def test_get_task_name_not_found(self, isolated_async_dao, metadata_root):
        """Covers branch where no task with the given name exists, raising RankNotFoundException."""
        exec_id, _ = _seed_execution_with_task(
            isolated_async_dao, metadata_root, task_name="step", failed=False
        )
        exec_ = _make_execution(metadata_root, id=exec_id)
        with pytest.raises(
            RankNotFoundException,
            match=r"No task 'missing' found in execution 'pipe' \[id \d+\]\.",
        ):
            exec_.get_task("missing")


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


class TestExecutionGetTaskgroupsWithUuid:
    """Cover Execution.get_taskgroups_with_uuid().

    Mirrors the name-based listing right above: same DAO contract, same
    TypeError-on-missing-group handling, uuid selector.
    """

    def test_get_taskgroups_with_uuid_success(self, isolated_async_dao, metadata_root):
        """Covers the UUID input and the TaskGroupRanks construction."""
        exec_id, _, tg_uuid, _ = _seed_taskgroup_with_task(
            isolated_async_dao, metadata_root
        )
        exec_ = _make_execution(metadata_root, id=exec_id)
        tgr = exec_.get_taskgroups_with_uuid(UUID(tg_uuid))

        assert isinstance(tgr, TaskGroupRanks)
        assert tgr.taskgroup.uuid == tg_uuid
        assert tgr.ranks == [[1]]

    def test_get_taskgroups_with_uuid_not_found(
        self, isolated_async_dao, metadata_root
    ):
        """Covers the str auto-cast and the TypeError catch on a missing group."""
        exec_id, _ = _seed_execution_with_task(
            isolated_async_dao, metadata_root, task_name="step", failed=False
        )
        exec_ = _make_execution(metadata_root, id=exec_id)

        assert exec_.get_taskgroups_with_uuid(str(uuid4())) is None


class TestExecutionGetTaskgroupForRank:
    """Cover Execution.get_taskgroup()."""

    def test_get_taskgroup_rejects_both_selectors(self, metadata_root):
        """Covers the mutually-exclusive selector guard, ahead of any DB call."""
        exec_ = _make_execution(metadata_root)
        with pytest.raises(
            AttributeError,
            match=r"Provide either 'taskgroup_name' or 'taskgroup_uuid', not both\.",
        ):
            exec_.get_taskgroup(taskgroup_name="tg", taskgroup_uuid=uuid4())

    def test_get_taskgroup_requires_a_selector(self, metadata_root):
        """Covers the missing-selector guard, ahead of any DB call."""
        exec_ = _make_execution(metadata_root)
        with pytest.raises(
            AttributeError,
            match=r"Provide either 'taskgroup_name' or 'taskgroup_uuid'\.",
        ):
            exec_.get_taskgroup()

    def test_get_taskgroup_success(self, isolated_async_dao, metadata_root):
        """Covers branch where taskgroup and rank are found."""
        exec_id, _, _, _ = _seed_taskgroup_with_task(
            isolated_async_dao, metadata_root, rank=[1, 2]
        )
        exec_ = _make_execution(metadata_root, id=exec_id)
        tgr = exec_.get_taskgroup("tg", rank=[1, 2])

        assert isinstance(tgr, TaskGroupRank)
        assert tgr.rank == [1, 2]

    def test_get_taskgroup_rank_not_found(self, isolated_async_dao, metadata_root):
        """Covers branch where taskgroup is found but rank is not, raising RankNotFoundException."""
        exec_id, _, _, _ = _seed_taskgroup_with_task(
            isolated_async_dao, metadata_root, rank=[1, 2]
        )
        exec_ = _make_execution(metadata_root, id=exec_id)
        with pytest.raises(
            RankNotFoundException,
            match=r"No taskgroup 'tg' found at rank \[9, 9\].*"
            r"Available ranks: \[\[1, 2\]\]",
        ):
            exec_.get_taskgroup("tg", rank=[9, 9])

    def test_get_taskgroup_empty_rank_success(self, isolated_async_dao, metadata_root):
        """Covers branch where requested rank is empty and taskgroup has NULL ranks.

        Verifies the path where target_rank is None and ranks equals [None],
        satisfying the rank_exists check.
        """
        exec_id, _, _, _ = _seed_taskgroup_with_task(
            isolated_async_dao, metadata_root, rank=None
        )
        exec_ = _make_execution(metadata_root, id=exec_id)
        tgr = exec_.get_taskgroup("tg", rank=[])

        assert isinstance(tgr, TaskGroupRank)
        assert tgr.rank is None

    def test_get_taskgroup_empty_rank_not_found(
        self, isolated_async_dao, metadata_root
    ):
        """Covers branch where empty rank input doesn't match non-None available ranks, raising RankNotFoundException."""
        exec_id, _, _, _ = _seed_taskgroup_with_task(
            isolated_async_dao, metadata_root, rank=[1, 2]
        )
        exec_ = _make_execution(metadata_root, id=exec_id)
        with pytest.raises(
            RankNotFoundException,
            match=r"No taskgroup 'tg' found at rank \[\].*"
            r"Available ranks: \[\[1, 2\]\]",
        ):
            exec_.get_taskgroup("tg", rank=[])

    def test_get_taskgroup_not_found(self, isolated_async_dao, metadata_root):
        """Covers branch where taskgroup is not found, raising RankNotFoundException."""
        exec_id, _ = _seed_execution_with_task(
            isolated_async_dao, metadata_root, task_name="step", failed=False
        )
        exec_ = _make_execution(metadata_root, id=exec_id)
        with pytest.raises(
            RankNotFoundException,
            match=r"No taskgroup 'missing_tg' found in execution 'pipe' \[id \d+\]\.",
        ):
            exec_.get_taskgroup("missing_tg", rank=[1])

    def test_get_taskgroup_success_with_uuid(self, isolated_async_dao, metadata_root):
        """Covers the uuid selector of the same rank-resolution logic.

        Kept in this class rather than a parallel one: get_taskgroup is a
        single method whose uuid branch only changes how the group is
        fetched and labelled, not how the rank is matched.
        """
        exec_id, _, tg_uuid, _ = _seed_taskgroup_with_task(
            isolated_async_dao, metadata_root, rank=[1, 2]
        )
        exec_ = _make_execution(metadata_root, id=exec_id)
        tgr = exec_.get_taskgroup(taskgroup_uuid=tg_uuid, rank=[1, 2])

        assert isinstance(tgr, TaskGroupRank)
        assert tgr.taskgroup.uuid == tg_uuid
        assert tgr.rank == [1, 2]

    def test_get_taskgroup_with_uuid_not_found(self, isolated_async_dao, metadata_root):
        """Covers the uuid-labelled failure when the group itself is missing."""
        exec_id, _ = _seed_execution_with_task(
            isolated_async_dao, metadata_root, task_name="step", failed=False
        )
        exec_ = _make_execution(metadata_root, id=exec_id)
        missing = uuid4()
        with pytest.raises(
            RankNotFoundException,
            match=rf"No taskgroup '{missing}' found in execution 'pipe' \[id \d+\]\.",
        ):
            exec_.get_taskgroup(taskgroup_uuid=missing, rank=[1])


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
        """Covers branch where ranks input is not a list (e.g., None).

        Reached through the class then through an instance, so both binding
        modes of the validator descriptor, which the SDK installs,
        are exercised.
        """
        assert TaskGroupRanks._parse_ranks(None) is None

        tgr = TaskGroupRanks(
            exec_id=1,
            metadata_root=metadata_root,
            uuid="123",
            name="tg",
            elements=[],
            ranks=[[1]],
        )
        assert tgr._parse_ranks(None) is None

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

    # ------------------------------------------------------------------
    # Nested taskgroups: ``outer_tg.elements`` references an inner
    # taskgroup UUID rather than a TaskType UUID. The DAO's
    # ``get_taskgroup_nested_tasks`` must BFS-resolve the chain so the
    # SDK's ``get_exit_context`` / ``get_exit_payload`` see the leaf task.
    #
    # The SDK's public ``get_taskgroups_with_name`` cannot reach a purely
    # nested outer taskgroup (its INNER JOIN on TaskType yields no rows),
    # so these tests construct ``TaskGroup`` / ``TaskGroupRank`` directly
    # using the real seeded ``exec_id`` and ``outer_tg_uuid`` -- the same
    # pattern as ``test_get_exit_context_empty`` above. The taskgroup name
    # still resolves in the DB, exercising the DAO's recursive BFS for
    # real.
    # ------------------------------------------------------------------

    def test_get_exit_context_nested_rank_none(self, isolated_async_dao, metadata_root):
        """Nested taskgroup, rank=None: BFS must reach the inner leaf task."""
        exec_id, _, outer_tg_uuid, _ = _seed_nested_taskgroups(
            isolated_async_dao, metadata_root, rank=None
        )
        tg = TaskGroup(
            exec_id=exec_id,
            metadata_root=metadata_root,
            uuid=outer_tg_uuid,
            name="outer_tg",
            elements=[],
        )
        tgr = TaskGroupRank(taskgroup=tg, rank=None)

        ctx = tgr.get_exit_context()
        assert isinstance(ctx, TaskExitContext)
        assert ctx["nested_attr"] == "nested_val"

    def test_get_exit_context_nested_rank_not_none(
        self, isolated_async_dao, metadata_root
    ):
        """Nested taskgroup, rank=[1, 2]: rank-filtered BFS reaches the leaf."""
        exec_id, _, outer_tg_uuid, _ = _seed_nested_taskgroups(
            isolated_async_dao, metadata_root, rank=[1, 2]
        )
        tg = TaskGroup(
            exec_id=exec_id,
            metadata_root=metadata_root,
            uuid=outer_tg_uuid,
            name="outer_tg",
            elements=[],
        )
        tgr = TaskGroupRank(taskgroup=tg, rank=[1, 2])

        ctx = tgr.get_exit_context()
        assert isinstance(ctx, TaskExitContext)
        assert ctx["nested_attr"] == "nested_val"

    def test_get_exit_payload_nested_rank_none(self, isolated_async_dao, metadata_root):
        """Nested taskgroup, rank=None: BFS reaches the leaf task's payload."""
        exec_id, _, outer_tg_uuid, _ = _seed_nested_taskgroups(
            isolated_async_dao, metadata_root, rank=None
        )
        tg = TaskGroup(
            exec_id=exec_id,
            metadata_root=metadata_root,
            uuid=outer_tg_uuid,
            name="outer_tg",
            elements=[],
        )
        tgr = TaskGroupRank(taskgroup=tg, rank=None)

        payload = tgr.get_exit_payload()
        assert isinstance(payload, TaskExitPayload)
        assert payload["leaf_task"] == 99

    def test_get_exit_payload_nested_rank_not_none(
        self, isolated_async_dao, metadata_root
    ):
        """Nested taskgroup, rank=[1, 2]: rank-filtered BFS reaches the leaf payload."""
        exec_id, _, outer_tg_uuid, _ = _seed_nested_taskgroups(
            isolated_async_dao, metadata_root, rank=[1, 2]
        )
        tg = TaskGroup(
            exec_id=exec_id,
            metadata_root=metadata_root,
            uuid=outer_tg_uuid,
            name="outer_tg",
            elements=[],
        )
        tgr = TaskGroupRank(taskgroup=tg, rank=[1, 2])

        payload = tgr.get_exit_payload()
        assert isinstance(payload, TaskExitPayload)
        assert payload["leaf_task"] == 99


class TestTaskGroupRankGetElement:
    """Cover TaskGroupRank.get_element().

    Split from TestTaskGroupRank because element resolution dispatches on
    what the DAO returns (a task row vs a nested taskgroup), whereas the
    rest of that class covers rank parsing and the merged exit views. It
    is the last method of the last SDK class, hence the last block here.
    """

    def test_get_element_returns_task(self, isolated_async_dao, metadata_root):
        """Covers the task branch, with a UUID object as input."""
        exec_id, task_id, _, tt_uuid = _seed_taskgroup_with_task(
            isolated_async_dao, metadata_root, task_name="tg_task", rank=[1, 2]
        )
        exec_ = _make_execution(metadata_root, id=exec_id)
        tgr = exec_.get_taskgroups_with_name("tg").get_rank([1, 2])

        element = tgr.get_element(UUID(tt_uuid))

        assert isinstance(element, Task)
        assert element.id == task_id
        assert element.rank == [1, 2]

    def test_get_element_returns_nested_taskgroup(
        self, isolated_async_dao, metadata_root
    ):
        """Covers the nested taskgroup branch, with the str auto-cast.

        The outer group is purely nested, so it is built directly from the
        seeded ids -- same pattern as the nested exit-view tests above.
        """
        exec_id, _, outer_tg_uuid, inner_tg_uuid = _seed_nested_taskgroups(
            isolated_async_dao, metadata_root, rank=[1, 2]
        )
        tg = TaskGroup(
            exec_id=exec_id,
            metadata_root=metadata_root,
            uuid=outer_tg_uuid,
            name="outer_tg",
            elements=[inner_tg_uuid],
        )
        tgr = TaskGroupRank(taskgroup=tg, rank=[1, 2])

        element = tgr.get_element(inner_tg_uuid)

        assert isinstance(element, TaskGroupRank)
        assert element.taskgroup.uuid == inner_tg_uuid
        assert element.rank == [1, 2]

    def test_get_element_rejects_non_first_degree_uuid(self, metadata_root):
        """Covers the membership guard, which fires before any DB round-trip."""
        tg = TaskGroup(
            exec_id=1,
            metadata_root=metadata_root,
            uuid="123",
            name="tg",
            elements=[],
        )
        tgr = TaskGroupRank(taskgroup=tg, rank=[1])

        with pytest.raises(ValueError, match="is not a 1st-degree element"):
            tgr.get_element(uuid4())

    def test_get_element_without_instance_at_rank(
        self, isolated_async_dao, metadata_root
    ):
        """Covers the unresolved-element branch, where the DAO returns None."""
        exec_id, _, tg_uuid, tt_uuid = _seed_taskgroup_with_task(
            isolated_async_dao, metadata_root, task_name="tg_task", rank=[1, 2]
        )
        with Session(isolated_async_dao) as session:
            session.query(TaskDbModel).filter_by(exec_id=exec_id).delete()
            session.commit()

        tg = TaskGroup(
            exec_id=exec_id,
            metadata_root=metadata_root,
            uuid=tg_uuid,
            name="tg",
            elements=[tt_uuid],
        )
        tgr = TaskGroupRank(taskgroup=tg, rank=[1, 2])

        with pytest.raises(
            RankNotFoundException,
            match=rf"No element '{tt_uuid}' found at rank \[1, 2\] "
            rf"in execution '{exec_id}'\.",
        ):
            tgr.get_element(tt_uuid)
