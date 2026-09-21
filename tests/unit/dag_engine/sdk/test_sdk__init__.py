"""Unit tests for retrain_pipelines.dag_engine.sdk.__init__"""

import asyncio
import concurrent.futures
import importlib
from datetime import datetime, timedelta
from unittest.mock import patch
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session

from retrain_pipelines.dag_engine.db.model import (
    Execution as ExecutionDbModel,
    Task as TaskDbModel,
    TaskType as TaskTypeDbModel,
)
from retrain_pipelines.dag_engine.sdk import ExecutionsIterator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MODULE = "retrain_pipelines.dag_engine.sdk"  # module under test (as importable path)


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
# _run_async
# ---------------------------------------------------------------------------


class TestRunAsync:
    """Tests for the module-level _run_async helper."""

    def _import_run_async(self):
        # Import fresh each time; patch must be applied before import in some
        # cases, but here we just reach into the module namespace directly.
        mod = importlib.import_module(_MODULE)
        return mod._run_async

    # ------------------------------------------------------------------
    # Path 1: no running event loop => asyncio.run()
    # ------------------------------------------------------------------

    def test_no_running_loop_uses_asyncio_run(self):
        """When there is no running loop, asyncio.run() should be called."""
        _run_async = self._import_run_async()

        async def _coro():
            return 42

        # We are NOT inside an async context here, so there is no running loop.
        result = _run_async(_coro())
        assert result == 42

    # ------------------------------------------------------------------
    # Path 2: running loop + in_notebook=True => ThreadPoolExecutor
    # ------------------------------------------------------------------

    def test_running_loop_in_notebook_uses_thread_executor(self):
        """Inside a running loop + notebook env, a worker thread must be used."""
        _run_async = self._import_run_async()

        async def _coro():
            return "notebook_result"

        async def _inner():
            # Patch in_notebook at the point of use inside the sdk module
            with patch(f"{_MODULE}.in_notebook", return_value=True):
                return _run_async(_coro())

        result = asyncio.run(_inner())
        assert result == "notebook_result"

    # ------------------------------------------------------------------
    # Path 3: running loop + in_notebook=False => loop.run_until_complete()
    # ------------------------------------------------------------------

    def test_running_loop_not_in_notebook_uses_run_until_complete(self):
        """Inside a running loop (non-notebook), _run_async delegates to
        loop.run_until_complete().

        Strategy
        --------
        Thread A creates dedicated_loop and drives _inner() via the *original*
        run_until_complete ; this makes get_running_loop() return dedicated_loop
        inside _inner, satisfying the "running loop" precondition.

        dedicated_loop.run_until_complete is replaced with a spy *before* the
        loop starts.  When _run_async calls the spy, the spy dispatches the
        coroutine to Thread B (which has no running loop) via asyncio.run(),
        avoiding all re-entrancy errors, while still recording the call so we
        can assert the right branch was taken.
        """
        _run_async = self._import_run_async()

        ruc_calls = []

        def _thread_main():
            dedicated_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(dedicated_loop)

            async def _coro():
                return "loop_result"

            original_ruc = dedicated_loop.run_until_complete

            def _spy_ruc(coro_or_fut):
                ruc_calls.append(True)
                # Dispatch to a sibling thread that owns no running loop.
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as p:
                    return p.submit(asyncio.run, coro_or_fut).result()

            dedicated_loop.run_until_complete = _spy_ruc

            async def _inner():
                with patch(f"{_MODULE}.in_notebook", return_value=False):
                    return _run_async(_coro())

            result = original_ruc(_inner())
            dedicated_loop.close()
            return result

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(_thread_main).result()

        assert result == "loop_result"
        assert len(ruc_calls) == 1


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
