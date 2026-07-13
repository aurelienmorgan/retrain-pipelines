"""Unit tests for retrain_pipelines.dag_engine.sdk.core

Note
-----------
- AsyncDAO and in_notebook are patched via their "bound names"
  inside the module being tested.
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
import threading

from retrain_pipelines.dag_engine.sdk import core
from retrain_pipelines.dag_engine.sdk.core import (
    AttrsDiff,
    Execution,
    ExecutionParams,
    TaskExitContext,
    shutdown_async_sdk,
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
        metadata_root="/tmp/meta",
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
        metadata_root="/tmp/meta",
        start_timestamp=_NOW,
        end_timestamp=None,
        success=True,
    )
    defaults.update(kwargs)
    return Task(**defaults)


# ---------------------------------------------------------------------------
# _run_async
# ---------------------------------------------------------------------------


class TestRunAsync:
    """Cover the module-level _run_async helper."""

    def test_shutdown_async_sdk_stops_and_joins(self):
        """Covers shutdown_async_sdk() happy path."""

        loop = MagicMock()
        loop.is_running.return_value = True

        thread = MagicMock(spec=threading.Thread)
        thread.is_alive.return_value = True

        with (
            patch.object(core, "_loop", loop),
            patch.object(core, "_loop_thread", thread),
        ):
            shutdown_async_sdk()

        loop.call_soon_threadsafe.assert_called_once_with(loop.stop)
        thread.join.assert_called_once_with(timeout=2.0)


# ---------------------------------------------------------------------------
# ExecutionParams
# ---------------------------------------------------------------------------


class TestExecutionParams:
    """Cover ExecutionParams mapping, lazy resolution, and object-SHA-based comparison."""

    def test_init_stores_raw(self):
        raw = {"a": {"default": 1}}
        params = ExecutionParams(raw, "/tmp/meta")
        assert params._raw is raw

    def test_active_storable_override_vs_default(self):
        raw = {"k": {"default": "def_val", "override": "ovr_val"}}
        params = ExecutionParams(raw, "/tmp/meta")
        assert params._active_storable("k") == "ovr_val"
        del raw["k"]["override"]
        assert params._active_storable("k") == "def_val"

    def test_getitem_resolves_storable(self):
        raw = {"x": {"default": 99}}
        params = ExecutionParams(raw, "/tmp/meta")
        assert params["x"] == 99

    def test_contains_true(self):
        assert "a" in ExecutionParams({"a": {}}, "/tmp/meta")

    def test_contains_false(self):
        assert "b" not in ExecutionParams({"a": {}}, "/tmp/meta")

    def test_iter(self):
        params = ExecutionParams({"a": {}, "b": {}}, "/tmp/meta")
        assert list(params) == ["a", "b"]

    def test_len(self):
        assert len(ExecutionParams({"a": {}, "b": {}}, "/tmp/meta")) == 2

    def test_repr(self):
        params = ExecutionParams({"x": {}, "y": {}}, "/tmp/meta")
        assert repr(params) == "ExecutionParams(params=['x', 'y'])"

    def test_keys(self):
        params = ExecutionParams({"k1": {}, "k2": {}}, "/tmp/meta")
        assert list(params.keys()) == ["k1", "k2"]

    def test_description_returns_string(self):
        params = ExecutionParams(
            {"p": {"description": "a dummy param", "default": 1}}, "/tmp/meta"
        )
        assert params.description("p") == "a dummy param"

    def test_description_missing_key_raises(self):
        params = ExecutionParams({"p": {"description": "x"}}, "/tmp/meta")
        with pytest.raises(KeyError):
            params.description("missing")

    def test_default_native_value(self):
        params = ExecutionParams(
            {"p": {"description": "d", "default": 42}}, "/tmp/meta"
        )
        assert params.default("p") == 42

    def test_default_resolves_disk_ref(self):
        disk_ref = {"__sha__": "abc", "__disk_ref__": "some/path.pkl"}
        params = ExecutionParams(
            {"p": {"description": "d", "default": disk_ref}}, "/tmp/meta"
        )
        with patch.object(
            ExecutionParams, "_resolve", return_value="unpickled_value"
        ) as mock_resolve:
            result = params.default("p")
        mock_resolve.assert_called_once_with(disk_ref)
        assert result == "unpickled_value"

    def test_default_no_default_key_returns_none(self):
        params = ExecutionParams({"p": {"description": "d"}}, "/tmp/meta")
        assert params.default("p") is None

    def test_default_ignores_override(self):
        """default() always returns the default value, never the override."""
        params = ExecutionParams(
            {"p": {"description": "d", "default": "orig", "override": "overridden"}},
            "/tmp/meta",
        )
        assert params.default("p") == "orig"

    def test_param_equals_native_match(self):
        p1 = ExecutionParams({"m": {"default": 1}}, "/tmp/meta")
        p2 = ExecutionParams({"m": {"default": 1}}, "/tmp/meta")
        assert p1.param_equals("m", p2) is True

    def test_param_equals_native_mismatch(self):
        p1 = ExecutionParams({"m": {"default": 1}}, "/tmp/meta")
        p2 = ExecutionParams({"m": {"default": 2}}, "/tmp/meta")
        assert p1.param_equals("m", p2) is False

    def test_param_equals_disk_ref_match(self):
        disk_a = {"__sha__": "hash1", "__disk_ref__": True}
        disk_b = {"__sha__": "hash1", "__disk_ref__": True}
        p1 = ExecutionParams({"m": {"default": disk_a}}, "/tmp/meta")
        p2 = ExecutionParams({"m": {"default": disk_b}}, "/tmp/meta")
        assert p1.param_equals("m", p2) is True

    def test_param_equals_disk_ref_mismatch(self):
        disk_a = {"__sha__": "hash1", "__disk_ref__": True}
        disk_b = {"__sha__": "hash2", "__disk_ref__": True}
        p1 = ExecutionParams({"m": {"default": disk_a}}, "/tmp/meta")
        p2 = ExecutionParams({"m": {"default": disk_b}}, "/tmp/meta")
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


class TestExecutionGetTasksWithName:
    """Cover Execution.get_tasks_with_name() classmethod."""

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


class TestExecutionGetTaskById:
    """Cover Execution.get_by_id() classmethod & dao.engine.dispose()."""

    """Cover Execution.get_by_id() classmethod & dao.engine.dispose()."""

    def test_get_by_id_success(self):
        mock_ext = MagicMock()
        # Use configure_mock so as to set mocked-object "name" attribute (and not mock name)
        mock_ext.configure_mock(
            id=42,
            name="pipe_v2",
            start_timestamp=_NOW,
            end_timestamp=_LATER,
            success=True,
            metadata_root="/tmp/meta",
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
            exec_ = Execution.get_by_id(42)
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
                Execution.get_by_id(99)
        mock_dao.engine.dispose.assert_awaited_once()


class TestAttrsDiff:
    def test_attrsdiff_repr(self):
        d = AttrsDiff(
            only_in_self=["a"],
            modified=["b"],
            only_in_other=["c"],
        )

        assert repr(d) == (
            "AttrsDiff(only_in_self=['a'], modified=['b'], only_in_other=['c'])"
        )


class TestExecutionGetParams:
    """Cover Execution.get_params()."""

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
            params = exec_.get_params()

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
            params = exec_.get_params()

        assert len(params) == 0

    def test_params_getattr_passthrough(self):
        params = ExecutionParams({}, "/tmp")

        obj = object()

        with patch(f"{_MODULE}.is_disk_ref", return_value=False):
            assert params._resolve(obj) is obj

    def test_diff(self):
        p1 = ExecutionParams(
            {
                "only1": {"default": 1},
                "same": {"default": 2},
                "mod": {"default": 3},
            },
            "/tmp",
        )
        p2 = ExecutionParams(
            {
                "same": {"default": 2},
                "mod": {"default": 4},
                "only2": {"default": 5},
            },
            "/tmp",
        )
        diff = p1.diff(p2)

        assert diff.only_in_self == ["only1"]
        assert diff.modified == ["mod"]
        assert diff.only_in_other == ["only2"]

    def test_diff_identical(self):
        params = {"foo": {"default": 1}}

        lhs = ExecutionParams(params, "/tmp")
        rhs = ExecutionParams(params, "/tmp")

        diff = lhs.diff(rhs)

        assert diff.only_in_self == []
        assert diff.modified == []
        assert diff.only_in_other == []


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


class TestTaskExitContext:
    def _make_row(self, attr_name, disk_ref=None, inline_val=None, sha="hash"):
        row = MagicMock()
        row.attr_name = attr_name
        row.disk_ref = disk_ref
        row.inline_val = inline_val
        row.sha = sha
        return row

    def test_getitem_inline(self):
        rows = [self._make_row("a", inline_val="v1")]
        ctx = TaskExitContext(rows, "/tmp")
        assert ctx["a"] == "v1"

    def test_getitem_disk_ref(self):
        rows = [self._make_row("a", disk_ref="path.pkl")]
        ctx = TaskExitContext(rows, "/tmp")
        # Patch at the source module where it's defined/imported
        with patch(f"{_MODULE}.load_from_disk", return_value="unpickled") as mock_load:
            val = ctx["a"]
            mock_load.assert_called_once_with("/tmp/path.pkl")
            assert val == "unpickled"

    def test_get_found_and_not_found(self):
        rows = [self._make_row("a", inline_val="v1")]
        ctx = TaskExitContext(rows, "/tmp")
        assert ctx.get("a") == "v1"
        assert ctx.get("missing", "default") == "default"

    def test_contains(self):
        rows = [self._make_row("a")]
        ctx = TaskExitContext(rows, "/tmp")
        assert "a" in ctx
        assert "b" not in ctx

    def test_iter_and_len_and_keys(self):
        rows = [self._make_row("a"), self._make_row("b")]
        ctx = TaskExitContext(rows, "/tmp")
        assert list(ctx) == ["a", "b"]
        assert len(ctx) == 2
        assert list(ctx.keys()) == ["a", "b"]

    def test_repr(self):
        rows = [self._make_row("a")]
        ctx = TaskExitContext(rows, "/tmp")
        assert repr(ctx) == "TaskExitContext(attrs=['a'])"

    def test_attr_equals(self):
        r1 = self._make_row("a", sha="h1")
        r2 = self._make_row("a", sha="h1")
        r3 = self._make_row("a", sha="h2")

        ctx1 = TaskExitContext([r1], "/tmp")
        ctx2 = TaskExitContext([r2], "/tmp")
        ctx3 = TaskExitContext([r3], "/tmp")
        ctx4 = TaskExitContext([], "/tmp")

        assert ctx1.attr_equals("a", ctx2) is True
        assert ctx1.attr_equals("a", ctx3) is False
        assert ctx1.attr_equals("missing", ctx4) is True  # both missing
        assert ctx1.attr_equals("a", ctx4) is False  # one missing

    def test_diff(self):
        r1 = self._make_row("only1", sha="h1")
        r2 = self._make_row("mod", sha="h1")
        r3 = self._make_row("same", sha="h1")

        r4 = self._make_row("only2", sha="h2")
        r5 = self._make_row("mod", sha="h2")
        r6 = self._make_row("same", sha="h1")

        ctx1 = TaskExitContext([r1, r2, r3], "/tmp")
        ctx2 = TaskExitContext([r4, r5, r6], "/tmp")

        diff = ctx1.diff(ctx2)
        assert diff.only_in_self == ["only1"]
        assert diff.only_in_other == ["only2"]
        assert diff.modified == ["mod"]

    def test_diff_identical(self):
        r1 = self._make_row("a", sha="h1")
        r2 = self._make_row("a", sha="h1")
        r3 = self._make_row("a", sha="h2")
        context = [r1, r2, r3]

        ctx1 = TaskExitContext(context, "/tmp")
        ctx2 = TaskExitContext(context, "/tmp")

        diff = ctx1.diff(ctx2)

        assert diff.only_in_self == []
        assert diff.modified == []
        assert diff.only_in_other == []


class TestTaskGetExitContext:
    """Cover Task.get_exit_context()."""

    def test_get_exit_context_success(self):
        task = _make_task(metadata_root="/tmp/meta")
        mock_row = MagicMock()
        mock_row.attr_name = "my_attr"
        mock_row.disk_ref = None
        mock_row.inline_val = "val"
        mock_row.sha = "hash1"

        mock_dao = MagicMock()
        mock_dao.get_task_context_attrs = AsyncMock(return_value=[mock_row])
        mock_dao.engine = MagicMock()
        mock_dao.engine.dispose = AsyncMock()

        with (
            patch(f"{_MODULE}.AsyncDAO", return_value=mock_dao),
            patch.dict(
                "os.environ",
                {"RP_METADATASTORE_ASYNC_URL": "sqlite+aiosqlite:///:memory:"},
            ),
        ):
            ctx = task.get_exit_context()

        assert isinstance(ctx, TaskExitContext)
        assert ctx["my_attr"] == "val"

    def test_get_exit_context_empty(self):
        task = _make_task(metadata_root="/tmp/meta")
        mock_dao = MagicMock()
        mock_dao.get_task_context_attrs = AsyncMock(return_value=None)
        mock_dao.engine = MagicMock()
        mock_dao.engine.dispose = AsyncMock()

        with (
            patch(f"{_MODULE}.AsyncDAO", return_value=mock_dao),
            patch.dict(
                "os.environ",
                {"RP_METADATASTORE_ASYNC_URL": "sqlite+aiosqlite:///:memory:"},
            ),
        ):
            ctx = task.get_exit_context()
        assert len(ctx) == 0
