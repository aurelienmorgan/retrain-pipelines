"""Unit tests for retrain_pipelines.dag_engine.sdk.core

Note
-----------
- AsyncDAO and in_notebook are patched via their "bound names"
  inside the module being tested.
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from retrain_pipelines.dag_engine.sdk.core import (
    _run_async,
    Execution,
    ExecutionParams,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


_NOW = datetime(2024, 6, 1, 12, 0, 0)
_LATER = datetime(2024, 6, 1, 13, 0, 0)

_MODULE = "retrain_pipelines.dag_engine.sdk.core"


def _make_execution(**kwargs):
    """Return a minimal valid Execution instance."""
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

        async def _coro():
            return 42

        assert _run_async(_coro()) == 42

    # Branch 2: running loop + notebook => ThreadPoolExecutor path
    def test_running_loop_in_notebook(self):
        """Inside a running loop in a notebook, work is sent to a fresh thread."""

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
# ExecutionParams
# ---------------------------------------------------------------------------


class TestExecutionParams:
    """Cover ExecutionParams mapping, lazy resolution, and object-SHA-based comparison."""

    def test_init_stores_raw(self):
        raw = {"a": {"default": 1}}
        params = ExecutionParams(raw)
        assert params._raw is raw

    def test_active_storable_override_vs_default(self):
        raw = {"k": {"default": "def_val", "override": "ovr_val"}}
        params = ExecutionParams(raw)
        assert params._active_storable("k") == "ovr_val"
        del raw["k"]["override"]
        assert params._active_storable("k") == "def_val"

    def test_getitem_resolves_storable(self):
        raw = {"x": {"default": 99}}
        params = ExecutionParams(raw)
        with patch(f"{_MODULE}.resolve_storable", return_value="resolved") as mock_res:
            assert params["x"] == "resolved"
            mock_res.assert_called_once_with(99)

    def test_contains_true(self):
        assert "a" in ExecutionParams({"a": {}})

    def test_contains_false(self):
        assert "b" not in ExecutionParams({"a": {}})

    def test_iter(self):
        params = ExecutionParams({"a": {}, "b": {}})
        assert list(params) == ["a", "b"]

    def test_len(self):
        assert len(ExecutionParams({"a": {}, "b": {}})) == 2

    def test_repr(self):
        params = ExecutionParams({"x": {}, "y": {}})
        assert repr(params) == "ExecutionParams(params=['x', 'y'])"

    def test_keys(self):
        params = ExecutionParams({"k1": {}, "k2": {}})
        assert list(params.keys()) == ["k1", "k2"]

    def test_description_returns_string(self):
        params = ExecutionParams({"p": {"description": "a dummy param", "default": 1}})
        assert params.description("p") == "a dummy param"

    def test_description_missing_key_raises(self):
        params = ExecutionParams({"p": {"description": "x"}})
        with pytest.raises(KeyError):
            params.description("missing")

    def test_default_native_value(self):
        params = ExecutionParams({"p": {"description": "d", "default": 42}})
        with patch(
            f"{_MODULE}.resolve_storable",
            side_effect=lambda x: x,
        ):
            assert params.default("p") == 42

    def test_default_resolves_disk_ref(self):
        disk_ref = {"__sha__": "abc", "__disk_ref__": "some/path.pkl"}
        params = ExecutionParams({"p": {"description": "d", "default": disk_ref}})
        with patch(
            f"{_MODULE}.resolve_storable",
            return_value="unpickled_value",
        ) as mock_res:
            result = params.default("p")
        mock_res.assert_called_once_with(disk_ref)
        assert result == "unpickled_value"

    def test_default_no_default_key_returns_none(self):
        params = ExecutionParams({"p": {"description": "d"}})
        with patch(
            f"{_MODULE}.resolve_storable",
            side_effect=lambda x: x,
        ):
            assert params.default("p") is None

    def test_default_ignores_override(self):
        """default() always returns the default value, never the override."""
        params = ExecutionParams(
            {"p": {"description": "d", "default": "orig", "override": "overridden"}}
        )
        with patch(
            f"{_MODULE}.resolve_storable",
            side_effect=lambda x: x,
        ):
            assert params.default("p") == "orig"

    def test_param_equals_native_match(self):
        p1 = ExecutionParams({"m": {"default": 1}})
        p2 = ExecutionParams({"m": {"default": 1}})
        assert p1.param_equals("m", p2) is True

    def test_param_equals_native_mismatch(self):
        p1 = ExecutionParams({"m": {"default": 1}})
        p2 = ExecutionParams({"m": {"default": 2}})
        assert p1.param_equals("m", p2) is False

    def test_param_equals_disk_ref_match(self):
        disk_a = {"__sha__": "hash1", "__disk_ref__": True}
        disk_b = {"__sha__": "hash1", "__disk_ref__": True}
        p1 = ExecutionParams({"m": {"default": disk_a}})
        p2 = ExecutionParams({"m": {"default": disk_b}})
        assert p1.param_equals("m", p2) is True

    def test_param_equals_disk_ref_mismatch(self):
        disk_a = {"__sha__": "hash1", "__disk_ref__": True}
        disk_b = {"__sha__": "hash2", "__disk_ref__": True}
        p1 = ExecutionParams({"m": {"default": disk_a}})
        p2 = ExecutionParams({"m": {"default": disk_b}})
        assert p1.param_equals("m", p2) is False


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


class TestExecutionGetById:
    """Cover Execution.getById() classmethod & dao.engine.dispose()."""

    def test_get_by_id_success(self):
        mock_ext = MagicMock()
        # Use configure_mock so as to set mocked-object "name" attribute (and not mock name)
        mock_ext.configure_mock(
            id=42,
            name="pipe_v2",
            start_timestamp=_NOW,
            end_timestamp=_LATER,
            success=True,
        )
        mock_dao = MagicMock()
        mock_dao.get_execution_ext = AsyncMock(return_value=mock_ext)
        # engine.dispose() is awaited in the finally block; must be AsyncMock.
        mock_dao.engine = MagicMock()
        mock_dao.engine.dispose = AsyncMock()
        with (
            patch(f"{_MODULE}.AsyncDAO", return_value=mock_dao),
            patch.dict(
                "os.environ",
                {"RP_METADATASTORE_ASYNC_URL": "sqlite+aiosqlite:///:memory:"},
            ),
        ):
            exec_ = Execution.getById(42)
        assert exec_.id == 42
        assert exec_.success is True
        mock_dao.engine.dispose.assert_awaited_once()

    def test_get_by_id_raises_keyerror(self):
        mock_dao = MagicMock()
        mock_dao.get_execution_ext = AsyncMock(return_value=None)
        # engine.dispose() is awaited in the finally block; must be AsyncMock.
        mock_dao.engine = MagicMock()
        mock_dao.engine.dispose = AsyncMock()
        with (
            patch(f"{_MODULE}.AsyncDAO", return_value=mock_dao),
            patch.dict(
                "os.environ",
                {"RP_METADATASTORE_ASYNC_URL": "sqlite+aiosqlite:///:memory:"},
            ),
        ):
            with pytest.raises(KeyError, match="No execution found with id=99"):
                Execution.getById(99)
        mock_dao.engine.dispose.assert_awaited_once()


class TestExecutionGetParams:
    """Cover Execution.getParams()."""

    def test_get_params_returns_instance(self):
        mock_full = MagicMock()
        mock_full.params = {"a": {"default": 1}}
        mock_dao = MagicMock()
        mock_dao.get_execution = AsyncMock(return_value=mock_full)
        mock_dao.engine = MagicMock()
        exec_ = _make_execution()
        with (
            patch(f"{_MODULE}.AsyncDAO", return_value=mock_dao),
            patch.dict(
                "os.environ",
                {"RP_METADATASTORE_ASYNC_URL": "sqlite+aiosqlite:///:memory:"},
            ),
        ):
            params = exec_.getParams()
        assert isinstance(params, ExecutionParams)
        assert len(params) == 1

    def test_get_params_empty_dict(self):
        mock_dao = MagicMock()
        mock_dao.get_execution = AsyncMock(return_value=MagicMock(params=None))
        mock_dao.engine = MagicMock()
        exec_ = _make_execution()
        with (
            patch(f"{_MODULE}.AsyncDAO", return_value=mock_dao),
            patch.dict(
                "os.environ",
                {"RP_METADATASTORE_ASYNC_URL": "sqlite+aiosqlite:///:memory:"},
            ),
        ):
            params = exec_.getParams()
        assert len(params) == 0


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

    def test_task_failed_true_success_false(self):
        task = _make_task(success=False)
        assert task.success is False
