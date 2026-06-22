"""Unit tests for retrain_pipelines.dag_engine.sdk.__init__"""

import asyncio
import concurrent.futures
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MODULE = "retrain_pipelines.dag_engine.sdk"  # module under test (as importable path)


def _make_exec_ext(
    id: int,
    name: str = "pipe",
    start: datetime | None = None,
    end: datetime | None = None,
    success: bool = True,
):
    """Build a lightweight stand-in for an ORM ExecutionExt row."""
    if start is None:
        start = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc) - timedelta(
            hours=id
        )
    return SimpleNamespace(
        id=id,
        name=name,
        start_timestamp=start,
        end_timestamp=end,
        success=success,
    )


def _make_dao_mock(execs_pages, count=0):
    """Return an AsyncDAO mock whose get_executions_ext cycles through *execs_pages*
    (a list of lists) on successive calls, and whose get_executions_count returns
    *count*.
    """
    dao = MagicMock()
    # Each call to get_executions_ext returns the next page
    dao.get_executions_ext = AsyncMock(side_effect=execs_pages)
    dao.get_executions_count = AsyncMock(return_value=count)
    return dao


# ---------------------------------------------------------------------------
# _run_async
# ---------------------------------------------------------------------------


class TestRunAsync:
    """Tests for the module-level _run_async helper."""

    def _import_run_async(self):
        # Import fresh each time; patch must be applied before import in some
        # cases, but here we just reach into the module namespace directly.
        import importlib

        mod = importlib.import_module(_MODULE)
        return mod._run_async

    # ------------------------------------------------------------------
    # Path 1: no running event loop → asyncio.run()
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
        from retrain_pipelines.dag_engine.sdk import ExecutionsIterator

        it = ExecutionsIterator(exec_name="my_pipe")
        assert it.exec_name == "my_pipe"
        assert it.success_only is False
        assert it.page_size == 10

    def test_custom_params(self):
        from retrain_pipelines.dag_engine.sdk import ExecutionsIterator

        it = ExecutionsIterator(exec_name="p", success_only=True, page_size=3)
        assert it.success_only is True
        assert it.page_size == 3


# ---------------------------------------------------------------------------
# ExecutionsIterator._previous  (via async __anext__ to stay in async context)
# ---------------------------------------------------------------------------


class TestPreviousAsync:
    """Drive _previous() directly via __anext__ (async path)."""

    # ------------------------------------------------------------------
    # DAO returns empty list => None => StopAsyncIteration
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_empty_dao_returns_none(self):
        from retrain_pipelines.dag_engine.sdk import ExecutionsIterator

        dao_mock = _make_dao_mock(execs_pages=[[]])
        it = ExecutionsIterator(exec_name="pipe")

        with patch(f"{_MODULE}.AsyncDAO", return_value=dao_mock):
            with patch.dict("os.environ", {"RP_METADATASTORE_ASYNC_URL": "fake://url"}):
                result = await it._previous()

        assert result is None
        assert it._buffer == []
        assert it._index == 0

    # ------------------------------------------------------------------
    # DAO returns rows => buffer filled, _before_datetime updated
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_first_page_populates_buffer(self):
        from retrain_pipelines.dag_engine.sdk import ExecutionsIterator

        rows = [_make_exec_ext(i) for i in range(1, 4)]  # ids 1,2,3
        dao_mock = _make_dao_mock(execs_pages=[rows, []])  # page1 then empty
        it = ExecutionsIterator(exec_name="pipe", page_size=3)

        with patch(f"{_MODULE}.AsyncDAO", return_value=dao_mock):
            with patch.dict("os.environ", {"RP_METADATASTORE_ASYNC_URL": "fake://url"}):
                exec0 = await it._previous()

        assert exec0 is not None
        assert exec0.id == 1
        # Buffer should hold all three; index advanced to 1
        assert len(it._buffer) == 3
        assert it._index == 1
        # _before_datetime must be set to last row's start - 1 ms
        expected_bdt = rows[-1].start_timestamp - timedelta(milliseconds=1)
        assert it._before_datetime == expected_bdt

    # ------------------------------------------------------------------
    # Buffer not exhausted => subsequent call serves from cache, no new DAO call
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_second_call_uses_buffer(self):
        from retrain_pipelines.dag_engine.sdk import ExecutionsIterator

        rows = [_make_exec_ext(i) for i in range(1, 4)]
        dao_mock = _make_dao_mock(execs_pages=[rows])
        it = ExecutionsIterator(exec_name="pipe", page_size=3)

        with patch(f"{_MODULE}.AsyncDAO", return_value=dao_mock):
            with patch.dict("os.environ", {"RP_METADATASTORE_ASYNC_URL": "fake://url"}):
                await it._previous()  # fills buffer, index=1
                exec1 = await it._previous()  # served from buffer, index=2

        assert exec1.id == 2
        # DAO should only have been called once
        assert dao_mock.get_executions_ext.call_count == 1

    # ------------------------------------------------------------------
    # Buffer exhausted => fetch next page
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_second_page_fetched_when_buffer_exhausted(self):
        from retrain_pipelines.dag_engine.sdk import ExecutionsIterator

        page1 = [_make_exec_ext(1)]
        page2 = [_make_exec_ext(2)]
        dao_mock = _make_dao_mock(execs_pages=[page1, page2, []])
        it = ExecutionsIterator(exec_name="pipe", page_size=1)

        with patch(f"{_MODULE}.AsyncDAO", return_value=dao_mock):
            with patch.dict("os.environ", {"RP_METADATASTORE_ASYNC_URL": "fake://url"}):
                e1 = await it._previous()  # page1 loaded, e1 returned
                e2 = await it._previous()  # page1 exhausted => page2 loaded

        assert e1.id == 1
        assert e2.id == 2
        assert dao_mock.get_executions_ext.call_count == 2

    # ------------------------------------------------------------------
    # __anext__ raises StopAsyncIteration when DAO exhausted
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_anext_stop_async_iteration(self):
        from retrain_pipelines.dag_engine.sdk import ExecutionsIterator

        dao_mock = _make_dao_mock(execs_pages=[[]])
        it = ExecutionsIterator(exec_name="pipe")

        with patch(f"{_MODULE}.AsyncDAO", return_value=dao_mock):
            with patch.dict("os.environ", {"RP_METADATASTORE_ASYNC_URL": "fake://url"}):
                with pytest.raises(StopAsyncIteration):
                    await it.__anext__()

    # ------------------------------------------------------------------
    # __anext__ returns execution when available
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_anext_returns_execution(self):
        from retrain_pipelines.dag_engine.sdk import ExecutionsIterator

        rows = [_make_exec_ext(10)]
        dao_mock = _make_dao_mock(execs_pages=[rows])
        it = ExecutionsIterator(exec_name="pipe")

        with patch(f"{_MODULE}.AsyncDAO", return_value=dao_mock):
            with patch.dict("os.environ", {"RP_METADATASTORE_ASYNC_URL": "fake://url"}):
                result = await it.__anext__()

        assert result.id == 10

    # ------------------------------------------------------------------
    # __aiter__ returns self
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_aiter_returns_self(self):
        from retrain_pipelines.dag_engine.sdk import ExecutionsIterator

        it = ExecutionsIterator(exec_name="pipe")
        assert it.__aiter__() is it

    # ------------------------------------------------------------------
    # success_only=True passes "success" status to DAO
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_success_only_passes_status_to_dao(self):
        from retrain_pipelines.dag_engine.sdk import ExecutionsIterator

        dao_mock = _make_dao_mock(execs_pages=[[]])
        it = ExecutionsIterator(exec_name="pipe", success_only=True)

        with patch(f"{_MODULE}.AsyncDAO", return_value=dao_mock):
            with patch.dict("os.environ", {"RP_METADATASTORE_ASYNC_URL": "fake://url"}):
                await it._previous()

        call_kwargs = dao_mock.get_executions_ext.call_args
        assert call_kwargs.kwargs.get("execs_status") == "success"

    # ------------------------------------------------------------------
    # success_only=False passes None status to DAO
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_not_success_only_passes_none_status_to_dao(self):
        from retrain_pipelines.dag_engine.sdk import ExecutionsIterator

        dao_mock = _make_dao_mock(execs_pages=[[]])
        it = ExecutionsIterator(exec_name="pipe", success_only=False)

        with patch(f"{_MODULE}.AsyncDAO", return_value=dao_mock):
            with patch.dict("os.environ", {"RP_METADATASTORE_ASYNC_URL": "fake://url"}):
                await it._previous()

        call_kwargs = dao_mock.get_executions_ext.call_args
        assert call_kwargs.kwargs.get("execs_status") is None


# ---------------------------------------------------------------------------
# ExecutionsIterator.previous  (sync public wrapper)
# ---------------------------------------------------------------------------


class TestPreviousSync:
    """previous() is a sync wrapper; it must work outside an event loop."""

    def test_previous_returns_execution(self):
        from retrain_pipelines.dag_engine.sdk import ExecutionsIterator

        rows = [_make_exec_ext(7)]
        dao_mock = _make_dao_mock(execs_pages=[rows])
        it = ExecutionsIterator(exec_name="pipe")

        with patch(f"{_MODULE}.AsyncDAO", return_value=dao_mock):
            with patch.dict("os.environ", {"RP_METADATASTORE_ASYNC_URL": "fake://url"}):
                result = it.previous()

        assert result is not None
        assert result.id == 7

    def test_previous_returns_none_when_exhausted(self):
        from retrain_pipelines.dag_engine.sdk import ExecutionsIterator

        dao_mock = _make_dao_mock(execs_pages=[[]])
        it = ExecutionsIterator(exec_name="pipe")

        with patch(f"{_MODULE}.AsyncDAO", return_value=dao_mock):
            with patch.dict("os.environ", {"RP_METADATASTORE_ASYNC_URL": "fake://url"}):
                result = it.previous()

        assert result is None


# ---------------------------------------------------------------------------
# ExecutionsIterator.length
# ---------------------------------------------------------------------------


class TestLength:
    def test_length_all(self):
        from retrain_pipelines.dag_engine.sdk import ExecutionsIterator

        dao_mock = _make_dao_mock(execs_pages=[], count=5)
        it = ExecutionsIterator(exec_name="pipe", success_only=False)

        with patch(f"{_MODULE}.AsyncDAO", return_value=dao_mock):
            with patch.dict("os.environ", {"RP_METADATASTORE_ASYNC_URL": "fake://url"}):
                n = it.length()

        assert n == 5
        dao_mock.get_executions_count.assert_awaited_once_with(
            pipeline_name="pipe", execs_status=None
        )

    def test_length_success_only(self):
        from retrain_pipelines.dag_engine.sdk import ExecutionsIterator

        dao_mock = _make_dao_mock(execs_pages=[], count=3)
        it = ExecutionsIterator(exec_name="pipe", success_only=True)

        with patch(f"{_MODULE}.AsyncDAO", return_value=dao_mock):
            with patch.dict("os.environ", {"RP_METADATASTORE_ASYNC_URL": "fake://url"}):
                n = it.length()

        assert n == 3
        dao_mock.get_executions_count.assert_awaited_once_with(
            pipeline_name="pipe", execs_status="success"
        )


# ---------------------------------------------------------------------------
# ExecutionsIterator sync iterator protocol
# ---------------------------------------------------------------------------


class TestSyncIteratorProtocol:
    def test_iter_returns_self(self):
        from retrain_pipelines.dag_engine.sdk import ExecutionsIterator

        it = ExecutionsIterator(exec_name="pipe")
        assert iter(it) is it

    def test_next_returns_execution(self):
        from retrain_pipelines.dag_engine.sdk import ExecutionsIterator

        rows = [_make_exec_ext(99)]
        dao_mock = _make_dao_mock(execs_pages=[rows])
        it = ExecutionsIterator(exec_name="pipe")

        with patch(f"{_MODULE}.AsyncDAO", return_value=dao_mock):
            with patch.dict("os.environ", {"RP_METADATASTORE_ASYNC_URL": "fake://url"}):
                result = next(it)

        assert result.id == 99

    def test_next_raises_stop_iteration(self):
        from retrain_pipelines.dag_engine.sdk import ExecutionsIterator

        dao_mock = _make_dao_mock(execs_pages=[[]])
        it = ExecutionsIterator(exec_name="pipe")

        with patch(f"{_MODULE}.AsyncDAO", return_value=dao_mock):
            with patch.dict("os.environ", {"RP_METADATASTORE_ASYNC_URL": "fake://url"}):
                with pytest.raises(StopIteration):
                    next(it)

    def test_full_sync_iteration(self):
        """for-loop over iterator yields all executions in order."""
        from retrain_pipelines.dag_engine.sdk import ExecutionsIterator

        rows = [_make_exec_ext(i) for i in [10, 9, 8]]
        dao_mock = _make_dao_mock(execs_pages=[rows, []])
        it = ExecutionsIterator(exec_name="pipe", page_size=3)

        with patch(f"{_MODULE}.AsyncDAO", return_value=dao_mock):
            with patch.dict("os.environ", {"RP_METADATASTORE_ASYNC_URL": "fake://url"}):
                collected = list(it)

        assert [e.id for e in collected] == [10, 9, 8]


# ---------------------------------------------------------------------------
# ExecutionsIterator async iteration protocol (full loop)
# ---------------------------------------------------------------------------


class TestAsyncIteratorProtocol:
    @pytest.mark.asyncio
    async def test_full_async_iteration(self):
        from retrain_pipelines.dag_engine.sdk import ExecutionsIterator

        rows = [_make_exec_ext(i) for i in [5, 4, 3]]
        dao_mock = _make_dao_mock(execs_pages=[rows, []])
        it = ExecutionsIterator(exec_name="pipe", page_size=3)

        collected = []
        with patch(f"{_MODULE}.AsyncDAO", return_value=dao_mock):
            with patch.dict("os.environ", {"RP_METADATASTORE_ASYNC_URL": "fake://url"}):
                async for exec_ in it:
                    collected.append(exec_.id)

        assert collected == [5, 4, 3]
