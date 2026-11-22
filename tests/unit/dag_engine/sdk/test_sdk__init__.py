"""Unit tests for retrain_pipelines.dag_engine.sdk.__init__"""

from datetime import datetime, timedelta
from typing import ClassVar
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session

from retrain_pipelines.dag_engine.db.model import (
    Execution as ExecutionDbModel,
    Task as TaskDbModel,
    TaskType as TaskTypeDbModel,
)
import retrain_pipelines.dag_engine.sdk as sdk_pkg
from retrain_pipelines.dag_engine.sdk import (
    BaseModelExt,
    _CleanDirMeta,
    ExecutionsIterator,
    Execution,
    run_async,
    shutdown_async_sdk,
)
from retrain_pipelines.dag_engine.sdk.core import Task


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_exec(engine, exec_id, name="pipe", failed=None, start=None):
    """Insert an execution (and optionally a task) via ORM.

    If ``failed`` is True or False, a corresponding task is inserted so that
    the execution's success status can be computed by the DAO. Uses naive
    datetime objects to ensure consistent string comparison in SQLite.
    """
    if start is None:
        start = datetime(2024, 1, 1, 12, 0, 0) - timedelta(hours=exec_id)

    end_ts = start + timedelta(hours=1) if failed is not None else None

    with Session(engine) as session:
        execution = ExecutionDbModel(
            id=exec_id,
            name=name,
            username="user",
            start_timestamp=start,
            end_timestamp=end_ts,
            metadata_root="/tmp/meta",
            artifacts_store_root="/tmp/artifacts",
        )
        session.add(execution)

        if failed is not None:
            tt_uuid = uuid4()
            tasktype = TaskTypeDbModel(
                uuid=tt_uuid,
                exec_id=exec_id,
                order=0,
                name="step",
                is_parallel=False,
                children=[],
            )
            session.add(tasktype)

            task = TaskDbModel(
                tasktype_uuid=tt_uuid,
                exec_id=exec_id,
                start_timestamp=start,
                end_timestamp=end_ts,
                failed=failed,
            )
            session.add(task)

        session.commit()


# ---------------------------------------------------------------------------
# run_async
# ---------------------------------------------------------------------------


class TestRunAsync:
    """Cover the package-level permanent background event loop and ``run_async``.

    No mocking or patching is used: a real coroutine is dispatched
    against the real background loop, and the real ``shutdown_async_sdk`` stops it.
    """

    def test_run_async_returns_coroutine_result(self):
        """``run_async`` schedules the coroutine on the background loop and
        blocks on the resulting future until it completes."""

        async def _coro():
            return 42

        # We are NOT inside an async context here, so there is no running loop.
        result = run_async(_coro())
        assert result == 42

    def test_shutdown_async_sdk_stops_and_joins(self):
        """Covers ``shutdown_async_sdk`` when the loop is running and the
        daemon thread is alive."""
        sdk_pkg._start_loop()
        loop = sdk_pkg._loop
        thread = sdk_pkg._loop_thread

        assert loop is not None
        assert thread is not None
        assert loop.is_running()
        assert thread.is_alive()

        shutdown_async_sdk()

        assert not loop.is_running()
        assert not thread.is_alive()

        # Reset global state so other tests can start it cleanly if needed
        sdk_pkg._loop = None
        sdk_pkg._loop_thread = None

    def test_shutdown_async_sdk_when_loop_not_started(self):
        """Covers branch where the loop was never started (idempotent no-op).

        Ensures ``shutdown_async_sdk`` does not raise when ``_loop`` and
        ``_loop_thread`` are both ``None``, covering the short-circuit
        condition branches.
        """
        # Ensure clean state - if loop was started, stop it first
        if sdk_pkg._loop is not None and sdk_pkg._loop.is_running():
            shutdown_async_sdk()
        sdk_pkg._loop = None
        sdk_pkg._loop_thread = None

        # Should be a safe no-op
        shutdown_async_sdk()
        assert sdk_pkg._loop is None
        assert sdk_pkg._loop_thread is None


# ---------------------------------------------------------------------------
# pydantic BaseModelExt: custom dir & validator descriptors
# ---------------------------------------------------------------------------


class _RefusingClassAccess:
    """Descriptor refusing class-level introspection.

    ``getattr(cls, name, None)`` only swallows ``AttributeError``, so a
    descriptor raising anything else is the one attribute shape that
    reaches the metaclass's tolerant guard while walking ``cls.__dict__``.
    """

    def __get__(self, obj, objtype=None):
        if obj is None:
            raise RuntimeError("class-level introspection refused")
        return "instance-bound"


class _IntrospectionProbe(BaseModelExt):
    """Model declaring an attribute that refuses class-level introspection."""

    refuses: ClassVar[_RefusingClassAccess] = _RefusingClassAccess()
    value: int = 0


class _LockedNameMeta(_CleanDirMeta):
    """Metaclass owning a read-only attribute named ``swappable``.

    ``type.__setattr__`` honours data descriptors found on the metaclass,
    so a setter-less property makes any assignment to that name fail with
    ``AttributeError`` -- which is what the validator-swap guard tolerates.
    """

    @property
    def swappable(cls):
        return None


class _ValidatorSwapProbe(BaseModelExt, metaclass=_LockedNameMeta):
    """Model whose public classmethod name is locked by its own metaclass.

    A plain classmethod (not a pydantic validator) is used on purpose:
    pydantic's own decorator collection never assigns to this name, so
    only the SDK's descriptor swap meets the locked attribute.
    """

    value: int = 0

    @classmethod
    def swappable(cls):
        return None


class TestBaseModelExtDir:
    """Cover the metaclass __dir__, the model __dir__ and the validator swap."""

    def test_class_dir_keeps_callables_and_fields_only(self):
        """Covers _CleanDirMeta.__dir__ dropping non-callable class data."""
        names = dir(Execution)

        assert "get_task" in names
        assert "id" in names
        assert "model_config" not in names

    def test_class_dir_skips_attributes_refusing_introspection(self):
        """Covers the tolerant guard for attributes raising on class access."""
        names = dir(_IntrospectionProbe)

        assert "refuses" in names
        assert "value" in names

    def test_instance_dir_excludes_inherited_attributes(self, metadata_root):
        """Covers BaseModelExt.__dir__ on cache miss then cache hit.

        Execution and Task share BaseModelExt as direct parent, so the
        second call reuses the frozenset built by the first one.
        """
        exec_ = Execution(
            id=1,
            name="pipe",
            metadata_root=metadata_root,
            artifacts_store_root="/tmp/artifacts",
            username="username",
            start_timestamp=datetime(2024, 6, 1, 12, 0, 0),
            success=True,
        )
        task = Task(
            id=10,
            name="my_task",
            start_timestamp=datetime(2024, 6, 1, 12, 0, 0),
            exec_id=7,
            metadata_root=metadata_root,
        )

        exec_names = dir(exec_)
        task_names = dir(task)

        assert "get_task" in exec_names
        assert "model_dump" not in exec_names
        assert "get_exit_context" in task_names
        assert "model_dump" not in task_names

    def test_validator_swap_tolerates_locked_attribute_name(self):
        """Covers the guard around the validator descriptor replacement.

        The metaclass property wins over the class dict entry, so the
        swap is refused and class construction must still succeed.
        """
        assert _ValidatorSwapProbe(value="3").value == 3
        assert _ValidatorSwapProbe.swappable is None


# ---------------------------------------------------------------------------
# ExecutionsIterator – construction / defaults
# ---------------------------------------------------------------------------


class TestExecutionsIteratorConstruction:
    def test_defaults(self):
        it = ExecutionsIterator(pipeline_name="my_pipe")
        assert it.pipeline_name == "my_pipe"
        assert it.success_only is False
        assert it.page_size == 10

    def test_custom_params(self):
        it = ExecutionsIterator(pipeline_name="p", success_only=True, page_size=3)
        assert it.success_only is True
        assert it.page_size == 3


# ---------------------------------------------------------------------------
# ExecutionsIterator._previous  (via async __anext__ to stay in async context)
# ---------------------------------------------------------------------------


class TestPreviousAsync:
    """Drive _previous() directly via __anext__ (async path)."""

    @pytest.mark.asyncio
    async def test_empty_dao_returns_none(self, isolated_async_dao):
        it = ExecutionsIterator(pipeline_name="pipe")
        result = await it._previous()

        assert result is None
        assert it._buffer == []
        assert it._index == 0

    @pytest.mark.asyncio
    async def test_first_page_populates_buffer(self, isolated_async_dao):
        _seed_exec(isolated_async_dao, 1)
        _seed_exec(isolated_async_dao, 2)
        _seed_exec(isolated_async_dao, 3)
        # each iteration returns at most a full page of size 3
        it = ExecutionsIterator(pipeline_name="pipe", page_size=3)

        exec0 = await it._previous()

        assert exec0 is not None
        assert exec0.id == 1
        # Buffer should hold all three ; index advanced to 1
        assert len(it._buffer) == 3
        assert it._index == 1
        # _before_datetime (for next page elements retrieval filtering)
        # is expected to be set to last row's start - 1 ms
        expected_bdt = it._buffer[-1].start_timestamp - timedelta(milliseconds=1)
        assert it._before_datetime == expected_bdt

    @pytest.mark.asyncio
    async def test_anext_stop_async_iteration(self, isolated_async_dao):
        it = ExecutionsIterator(pipeline_name="pipe")
        with pytest.raises(StopAsyncIteration):
            await it.__anext__()

    @pytest.mark.asyncio
    async def test_aiter_returns_self(self):
        it = ExecutionsIterator(pipeline_name="pipe")
        assert it.__aiter__() is it

    @pytest.mark.asyncio
    async def test_success_only_passes_status_to_dao(self, isolated_async_dao):
        _seed_exec(isolated_async_dao, 1, failed=False)
        _seed_exec(isolated_async_dao, 2, failed=True)
        it = ExecutionsIterator(pipeline_name="pipe", success_only=True, page_size=10)

        result = await it._previous()

        assert result is not None
        assert result.id == 1
        assert await it._previous() is None

    @pytest.mark.asyncio
    async def test_not_success_only_passes_none_status_to_dao(self, isolated_async_dao):
        _seed_exec(isolated_async_dao, 1, failed=False)
        _seed_exec(isolated_async_dao, 2, failed=True)
        it = ExecutionsIterator(pipeline_name="pipe", success_only=False, page_size=10)

        e1 = await it._previous()
        e2 = await it._previous()

        assert {e1.id, e2.id} == {1, 2}
        assert await it._previous() is None


# ---------------------------------------------------------------------------
# _previous – page-buffer round-trip validation (engine-level query counting)
# ---------------------------------------------------------------------------


class TestPreviousAsyncDAOCallCount:
    """Validate that ``_previous()`` minimizes database round-trips.

    Uses SQLAlchemy's ``before_cursor_execute`` engine event (via the
    ``query_counter`` fixture) to count real SQL statements issued against
    the file-based SQLite database.
    No mocking or patching of the
    ``AsyncDAO`` symbol is required—the DAO, ORM, connection pool, and
    SQL are all exercised end-to-end.

    Covers:
      * buffer not exhausted => zero additional queries on next call
      * buffer exhausted     => new page fetched from the database
    """

    @pytest.mark.asyncio
    async def test_second_call_uses_buffer(self, isolated_async_dao, query_counter):
        """When the buffer still holds unread rows the second call
        must not issue any SQL at all."""
        _seed_exec(isolated_async_dao, 1)
        _seed_exec(isolated_async_dao, 2)
        _seed_exec(isolated_async_dao, 3)
        it = ExecutionsIterator(pipeline_name="pipe", page_size=3)

        await it._previous()  # fills buffer (3 rows)
        query_counter.reset()
        exec1 = await it._previous()  # served from buffer

        assert exec1.id == 2
        assert query_counter.count == 0

    @pytest.mark.asyncio
    async def test_second_page_fetched_when_buffer_exhausted(
        self, isolated_async_dao, query_counter
    ):
        """When the buffer is exhausted the next call must hit the
        database again to fetch a fresh page."""
        _seed_exec(isolated_async_dao, 1)
        _seed_exec(isolated_async_dao, 2)
        it = ExecutionsIterator(pipeline_name="pipe", page_size=1)

        await it._previous()  # page 1 (1 row)
        query_counter.reset()
        e2 = await it._previous()  # buffer exhausted => page 2

        assert e2.id == 2
        assert query_counter.count > 0


# ---------------------------------------------------------------------------
# ExecutionsIterator.previous  (sync public wrapper)
# ---------------------------------------------------------------------------


class TestPreviousSync:
    """previous() is a sync wrapper; it must work outside an event loop."""

    def test_previous_returns_execution(self, isolated_async_dao):
        _seed_exec(isolated_async_dao, 7)
        it = ExecutionsIterator(pipeline_name="pipe")

        result = it.previous()

        assert result is not None
        assert result.id == 7

    def test_previous_returns_none_when_exhausted(self, isolated_async_dao):
        it = ExecutionsIterator(pipeline_name="pipe")

        result = it.previous()

        assert result is None


# ---------------------------------------------------------------------------
# ExecutionsIterator.length
# ---------------------------------------------------------------------------


class TestLength:
    def test_length_all(self, isolated_async_dao):
        for i in range(1, 6):
            _seed_exec(isolated_async_dao, i)
        it = ExecutionsIterator(pipeline_name="pipe", success_only=False)

        n = it.length()

        assert n == 5

    def test_length_success_only(self, isolated_async_dao):
        _seed_exec(isolated_async_dao, 1, failed=False)
        _seed_exec(isolated_async_dao, 2, failed=False)
        _seed_exec(isolated_async_dao, 3, failed=True)
        it = ExecutionsIterator(pipeline_name="pipe", success_only=True)

        n = it.length()

        assert n == 2


# ---------------------------------------------------------------------------
# ExecutionsIterator sync iterator protocol
# ---------------------------------------------------------------------------


class TestSyncIteratorProtocol:
    def test_iter_returns_self(self):
        it = ExecutionsIterator(pipeline_name="pipe")
        assert iter(it) is it

    def test_next_returns_execution(self, isolated_async_dao):
        _seed_exec(isolated_async_dao, 99)
        it = ExecutionsIterator(pipeline_name="pipe")

        result = next(it)

        assert result.id == 99

    def test_next_raises_stop_iteration(self, isolated_async_dao):
        it = ExecutionsIterator(pipeline_name="pipe")
        with pytest.raises(StopIteration):
            next(it)

    def test_full_sync_iteration(self, isolated_async_dao):
        _seed_exec(isolated_async_dao, 10)
        _seed_exec(isolated_async_dao, 9)
        _seed_exec(isolated_async_dao, 8)
        it = ExecutionsIterator(pipeline_name="pipe", page_size=3)

        collected = list(it)

        assert [e.id for e in collected] == [8, 9, 10]


# ---------------------------------------------------------------------------
# ExecutionsIterator async iteration protocol (full loop)
# ---------------------------------------------------------------------------


class TestAsyncIteratorProtocol:
    @pytest.mark.asyncio
    async def test_full_async_iteration(self, isolated_async_dao):
        _seed_exec(isolated_async_dao, 5)
        _seed_exec(isolated_async_dao, 4)
        _seed_exec(isolated_async_dao, 3)
        it = ExecutionsIterator(pipeline_name="pipe", page_size=3)

        collected = []
        async for exec_ in it:
            collected.append(exec_.id)

        assert collected == [3, 4, 5]
