"""Unit tests for retrain_pipelines.dag_engine.sdk.core

Note
-----------
- AsyncDAO and in_notebook are patched via their "bound names"
  inside the module under test (retrain_pipelines.dag_engine.sdk.core).
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2024, 6, 1, 12, 0, 0)
_LATER = datetime(2024, 6, 1, 13, 0, 0)

_MODULE = "retrain_pipelines.dag_engine.sdk.core"


def _make_execution(**kwargs):
    """Return a minimal valid Execution instance."""
    from retrain_pipelines.dag_engine.sdk.core import Execution

    defaults = dict(
        id=1,
        name="pipe",
        start_timestamp=_NOW,
        end_timestamp=None,
        success=True,
    )
    defaults.update(kwargs)
    return Execution(**defaults)


def _make_task(**kwargs):
    from retrain_pipelines.dag_engine.sdk.core import Task

    defaults = dict(
        id=10,
        name="my_task",
        start_timestamp=_NOW,
        end_timestamp=None,
        success=True,
    )
    defaults.update(kwargs)
    return Task(**defaults)


# ---------------------------------------------------------------------------
# _run_async ; three branches
# ---------------------------------------------------------------------------


class TestRunAsync:
    """Cover every branch of the module-level _run_async helper."""

    # Branch 1: no running event loop => asyncio.run()
    def test_no_running_loop(self):
        """_run_async uses asyncio.run() when there is no running loop."""
        from retrain_pipelines.dag_engine.sdk.core import _run_async

        async def _coro():
            return 42

        assert _run_async(_coro()) == 42

    # Branch 2: running loop + notebook => ThreadPoolExecutor path
    def test_running_loop_in_notebook(self):
        """Inside a running loop in a notebook, work is sent to a fresh thread."""
        from retrain_pipelines.dag_engine.sdk.core import _run_async

        async def _coro():
            return "notebook_result"

        async def _runner():
            with patch(f"{_MODULE}.in_notebook", return_value=True):
                return _run_async(_coro())

        result = asyncio.run(_runner())
        assert result == "notebook_result"

    # Branch 3: running loop + NOT notebook => loop.run_until_complete()
    #
    # The real call path is:
    #   asyncio.get_event_loop().run_until_complete(coro)
    #
    # Calling that for real inside an already-running loop raises
    # "This event loop is already running".  We therefore mock
    # asyncio.get_event_loop() so that run_until_complete() is a
    # plain MagicMock that captures the coroutine and returns our
    # sentinel ; covering the line without hitting the asyncio guard.
    # The coroutine object is explicitly closed afterwards to prevent
    # the "coroutine was never awaited" RuntimeWarning.
    def test_running_loop_not_notebook(self):
        """Inside a running loop outside a notebook, loop.run_until_complete is called."""
        from retrain_pipelines.dag_engine.sdk.core import _run_async

        async def _coro():
            return "loop_result"

        coro = _coro()
        captured = {}

        fake_loop = MagicMock()

        def _fake_run_until_complete(c):
            captured["coro"] = c
            return "loop_result"

        fake_loop.run_until_complete.side_effect = _fake_run_until_complete

        async def _runner():
            with (
                patch(f"{_MODULE}.in_notebook", return_value=False),
                patch(f"{_MODULE}.asyncio.get_event_loop", return_value=fake_loop),
            ):
                return _run_async(coro)

        result = asyncio.run(_runner())

        # Close the coroutine that was handed to the mock to silence
        # the "coroutine was never awaited" RuntimeWarning.
        captured["coro"].close()

        assert result == "loop_result"
        fake_loop.run_until_complete.assert_called_once()


# ---------------------------------------------------------------------------
# Execution model
# ---------------------------------------------------------------------------


class TestExecutionModel:
    def test_fields_set_correctly(self):
        exec_ = _make_execution(end_timestamp=_LATER)
        assert exec_.id == 1
        assert exec_.name == "pipe"
        assert exec_.start_timestamp == _NOW
        assert exec_.end_timestamp == _LATER
        assert exec_.success is True

    # completed()
    def test_completed_true_when_end_timestamp_set(self):
        exec_ = _make_execution(end_timestamp=_LATER)
        assert exec_.completed() is True

    def test_completed_false_when_end_timestamp_none(self):
        exec_ = _make_execution(end_timestamp=None)
        assert exec_.completed() is False

    # get_attr() ; attribute present in context_dump
    def test_get_attr_found(self):
        exec_ = _make_execution()

        full_exec_mock = MagicMock()
        full_exec_mock.context_dump = {"model_version_blessed": "v3"}

        async_dao_instance = MagicMock()
        async_dao_instance.get_execution = AsyncMock(return_value=full_exec_mock)

        with (
            patch(f"{_MODULE}.AsyncDAO", return_value=async_dao_instance),
            patch.dict(
                "os.environ",
                {"RP_METADATASTORE_ASYNC_URL": "sqlite+aiosqlite:///:memory:"},
            ),
        ):
            result = exec_.get_attr("model_version_blessed")

        assert result == "v3"

    # get_attr() ; attribute absent from context_dump
    def test_get_attr_key_missing(self):
        exec_ = _make_execution()

        full_exec_mock = MagicMock()
        full_exec_mock.context_dump = {"other_key": "other_value"}

        async_dao_instance = MagicMock()
        async_dao_instance.get_execution = AsyncMock(return_value=full_exec_mock)

        with (
            patch(f"{_MODULE}.AsyncDAO", return_value=async_dao_instance),
            patch.dict(
                "os.environ",
                {"RP_METADATASTORE_ASYNC_URL": "sqlite+aiosqlite:///:memory:"},
            ),
        ):
            result = exec_.get_attr("model_version_blessed")

        assert result is None

    # get_attr() ; context_dump is None / falsy
    def test_get_attr_no_context_dump(self):
        exec_ = _make_execution()

        full_exec_mock = MagicMock()
        full_exec_mock.context_dump = None

        async_dao_instance = MagicMock()
        async_dao_instance.get_execution = AsyncMock(return_value=full_exec_mock)

        with (
            patch(f"{_MODULE}.AsyncDAO", return_value=async_dao_instance),
            patch.dict(
                "os.environ",
                {"RP_METADATASTORE_ASYNC_URL": "sqlite+aiosqlite:///:memory:"},
            ),
        ):
            result = exec_.get_attr("anything")

        assert result is None

    # get_tasks_with_name() ; tasks returned, failed=False => success=True
    def test_get_tasks_with_name_returns_tasks(self):
        exec_ = _make_execution()

        orm_task = MagicMock()
        orm_task.id = 99
        orm_task.start_timestamp = _NOW
        orm_task.end_timestamp = _LATER
        orm_task.failed = False  # success = not False => True

        async_dao_instance = MagicMock()
        async_dao_instance.get_execution_tasks_with_name = AsyncMock(
            return_value=[orm_task]
        )

        with (
            patch(f"{_MODULE}.AsyncDAO", return_value=async_dao_instance),
            patch.dict(
                "os.environ",
                {"RP_METADATASTORE_ASYNC_URL": "sqlite+aiosqlite:///:memory:"},
            ),
        ):
            tasks = exec_.get_tasks_with_name("preprocess")

        assert len(tasks) == 1
        assert tasks[0].id == 99
        assert tasks[0].name == "preprocess"
        assert tasks[0].success is True

    # get_tasks_with_name() ; failed=None => success defaults to True
    def test_get_tasks_with_name_failed_none(self):
        exec_ = _make_execution()

        orm_task = MagicMock()
        orm_task.id = 7
        orm_task.start_timestamp = _NOW
        orm_task.end_timestamp = None
        orm_task.failed = None  # triggers the `else True` branch of the ternary

        async_dao_instance = MagicMock()
        async_dao_instance.get_execution_tasks_with_name = AsyncMock(
            return_value=[orm_task]
        )

        with (
            patch(f"{_MODULE}.AsyncDAO", return_value=async_dao_instance),
            patch.dict(
                "os.environ",
                {"RP_METADATASTORE_ASYNC_URL": "sqlite+aiosqlite:///:memory:"},
            ),
        ):
            tasks = exec_.get_tasks_with_name("train")

        assert tasks[0].success is True

    # get_tasks_with_name() ; empty list from DAO
    def test_get_tasks_with_name_empty(self):
        exec_ = _make_execution()

        async_dao_instance = MagicMock()
        async_dao_instance.get_execution_tasks_with_name = AsyncMock(return_value=[])

        with (
            patch(f"{_MODULE}.AsyncDAO", return_value=async_dao_instance),
            patch.dict(
                "os.environ",
                {"RP_METADATASTORE_ASYNC_URL": "sqlite+aiosqlite:///:memory:"},
            ),
        ):
            tasks = exec_.get_tasks_with_name("nonexistent")

        assert tasks == []

    # get_tasks_with_name() ; None from DAO => early-return []
    def test_get_tasks_with_name_none_from_dao(self):
        exec_ = _make_execution()

        async_dao_instance = MagicMock()
        async_dao_instance.get_execution_tasks_with_name = AsyncMock(return_value=None)

        with (
            patch(f"{_MODULE}.AsyncDAO", return_value=async_dao_instance),
            patch.dict(
                "os.environ",
                {"RP_METADATASTORE_ASYNC_URL": "sqlite+aiosqlite:///:memory:"},
            ),
        ):
            tasks = exec_.get_tasks_with_name("nonexistent")

        assert tasks == []

    # elements_iterator() ; raises NotImplementedError
    def test_elements_iterator_not_implemented(self):
        exec_ = _make_execution()
        with pytest.raises(NotImplementedError):
            exec_.elements_iterator()


# ---------------------------------------------------------------------------
# Task model
# ---------------------------------------------------------------------------


class TestTaskModel:
    def test_task_fields(self):
        task = _make_task()
        assert task.id == 10
        assert task.name == "my_task"
        assert task.start_timestamp == _NOW
        assert task.end_timestamp is None
        assert task.success is True

    def test_task_success_none(self):
        task = _make_task(success=None)
        assert task.success is None

    def test_task_with_end_timestamp(self):
        task = _make_task(end_timestamp=_LATER)
        assert task.end_timestamp == _LATER
