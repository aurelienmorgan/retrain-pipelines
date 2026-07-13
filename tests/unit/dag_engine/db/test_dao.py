"""
Unit tests for retrain_pipelines.dag_engine.db.dao.
"""

import logging
import threading
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from unittest.mock import AsyncMock

from retrain_pipelines.dag_engine.db.dao import _truncate_to_millis
from retrain_pipelines.dag_engine.db.model import (
    Execution,
    ExecutionExt,
    Task,
    TaskContextAttr,
    TaskExt,
    TaskGroup,
    TaskTrace,
    TaskType,
)

_NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def _null_session():
    """Return an AsyncMock session whose execute yields first()=None,
    scalar_one_or_none()=None, and fetchall()=[]."""
    mock_result = MagicMock()
    mock_result.first.return_value = None
    mock_result.scalar_one_or_none.return_value = None
    mock_result.fetchall.return_value = []

    mock_session = AsyncMock()
    mock_session.execute = AsyncMock(return_value=mock_result)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    return mock_session


# ==============================================================================
# _truncate_to_millis
# ==============================================================================


class TestTruncateToMillis:
    def test_truncates_sub_millisecond_precision(self):
        dt = datetime(2024, 1, 1, 0, 0, 0, 123456, tzinfo=timezone.utc)
        assert _truncate_to_millis(dt).microsecond == 123000

    def test_already_millisecond_precision_unchanged(self):
        dt = datetime(2024, 1, 1, 0, 0, 0, 500000, tzinfo=timezone.utc)
        assert _truncate_to_millis(dt).microsecond == 500000

    def test_none_passthrough(self):
        assert _truncate_to_millis(None) is None

    def test_non_datetime_passthrough(self):
        assert _truncate_to_millis("2024-01-01") == "2024-01-01"


# ==============================================================================
# DAOBase – sync, shared in-memory SQLite
# ==============================================================================


class TestDAOExecution:
    def test_add_and_get_roundtrip(self, sync_dao):
        with patch("requests.post"):
            exec_id = sync_dao.add_execution(
                name="pipe_exec",
                username="user",
                _start_timestamp=_NOW,
                metadata_root="/tmp/meta",
            )
        assert isinstance(exec_id, int)
        ex = sync_dao.get_execution(exec_id)
        assert ex.name == "pipe_exec" and ex.username == "user"

    @pytest.mark.asyncio
    async def test_asyncdao_add_execution_wrapper_hits_not_null_constraint(
        self, async_dao
    ):
        from sqlalchemy.exc import IntegrityError

        # add_execution() passes no kwargs; Execution.name is NOT NULL,
        # so the wrapper's _add_entity call raises here.
        with pytest.raises(IntegrityError):
            await async_dao.add_execution()

    @pytest.mark.asyncio
    async def test_asyncdao_get_execution_wrapper(self, async_dao):
        exec_id = await async_dao._async_add_entity(
            Execution,
            name="wrapper_pipe",
            username="u",
            _start_timestamp=_NOW,
            metadata_root="/tmp/meta",
        )
        result = await async_dao.get_execution(exec_id)
        assert result is not None and result.id == exec_id

    def test_get_missing_returns_none(self, sync_dao):
        assert sync_dao.get_execution(999999) is None

    def test_update_end_timestamp(self, sync_dao):
        with patch("requests.post"):
            exec_id = sync_dao.add_execution(
                name="pipe_update",
                username="u",
                _start_timestamp=_NOW,
                metadata_root="/tmp/meta",
            )
            sync_dao.update_execution(exec_id, _end_timestamp=_NOW + timedelta(hours=1))
        ex = sync_dao.get_execution(exec_id)
        assert ex._end_timestamp is not None

    def test_update_missing_returns_none(self, sync_dao):
        with patch("requests.post"):
            result = sync_dao.update_execution(999999, name="x")
        assert result is None

    def test_get_executions_by_name_filters_correctly(self, sync_dao):
        with patch("requests.post"):
            for _ in range(3):
                sync_dao.add_execution(
                    name="named_pipe",
                    username="u",
                    _start_timestamp=_NOW,
                    metadata_root="/tmp/meta",
                )
            sync_dao.add_execution(
                name="other_pipe",
                username="u",
                _start_timestamp=_NOW,
                metadata_root="/tmp/meta",
            )
        results = sync_dao.get_executions("named_pipe")
        assert len(results) >= 3
        assert all(r.name == "named_pipe" for r in results)

    def test_get_executions_unknown_name_empty(self, sync_dao):
        assert sync_dao.get_executions("__no_such_pipe__") == []


class TestDAOTaskType:
    def _exec_id(self, dao):
        with patch("requests.post"):
            return dao.add_execution(
                name="p", username="u", _start_timestamp=_NOW, metadata_root="/tmp/meta"
            )

    def test_add_tasktype_returns_uuid_value(self, sync_dao):
        exec_id = self._exec_id(sync_dao)
        with patch("requests.post"):
            uuid_val = sync_dao.add_tasktype(
                uuid=uuid4(),
                exec_id=exec_id,
                order=0,
                name="step",
                is_parallel=False,
                children=[],
            )
        assert uuid_val is not None


# ==============================================================================
# TestDAOTask uses isolated_dao (NullPool, file SQLite)
# ==============================================================================


class TestDAOTask:
    def _seed(self, dao):
        tt_uuid = uuid4()
        with patch("requests.post"):
            exec_id = dao.add_execution(
                name="p", username="u", _start_timestamp=_NOW, metadata_root="/tmp/meta"
            )
            dao.add_tasktype(
                uuid=tt_uuid,
                exec_id=exec_id,
                order=0,
                name="step",
                is_parallel=False,
                children=[],
            )
        return exec_id, tt_uuid

    def test_add_and_get_task(self, isolated_dao):
        exec_id, tt_uuid = self._seed(isolated_dao)
        with patch("requests.post"):
            task_id = isolated_dao.add_task(
                tasktype_uuid=tt_uuid, exec_id=exec_id, _start_timestamp=_NOW
            )
        assert isinstance(task_id, int)
        t = isolated_dao.get_task(task_id)
        assert t.exec_id == exec_id

    def test_get_task_missing_returns_none(self, isolated_dao):
        assert isolated_dao.get_task(999999) is None

    def test_update_task_end_and_failed(self, isolated_dao):
        exec_id, tt_uuid = self._seed(isolated_dao)
        with patch("requests.post"):
            task_id = isolated_dao.add_task(
                tasktype_uuid=tt_uuid, exec_id=exec_id, _start_timestamp=_NOW
            )
            isolated_dao.update_task(
                task_id, _end_timestamp=_NOW + timedelta(minutes=5), failed=False
            )
        t = isolated_dao.get_task(task_id)
        assert t._end_timestamp is not None and t.failed is False

    def test_get_tasks_by_execution(self, isolated_dao):
        exec_id, tt_uuid = self._seed(isolated_dao)
        with patch("requests.post"):
            for _ in range(4):
                isolated_dao.add_task(
                    tasktype_uuid=tt_uuid, exec_id=exec_id, _start_timestamp=_NOW
                )
        tasks = isolated_dao.get_tasks_by_execution(exec_id)
        assert len(tasks) >= 4

    def test_get_tasks_by_execution_empty(self, isolated_dao):
        assert isolated_dao.get_tasks_by_execution(999999) == []


# ==============================================================================
# DAOTaskGroup
# ==============================================================================


class TestDAOTaskGroup:
    def test_add_taskgroup(self, sync_dao):
        with patch("requests.post"):
            exec_id = sync_dao.add_execution(
                name="p", username="u", _start_timestamp=_NOW, metadata_root="/tmp/meta"
            )
            tg_uuid = sync_dao.add_taskgroup(
                uuid=uuid4(),
                exec_id=exec_id,
                order=0,
                name="grp",
                elements=[],
            )
        assert tg_uuid is not None


# ==============================================================================
# DAOTaskTrace
# ==============================================================================


class TestDAOTaskTrace:
    def _seed_task(self, dao):
        tt_uuid = uuid4()
        with patch("requests.post"):
            exec_id = dao.add_execution(
                name="p",
                username="u",
                _start_timestamp=_NOW,
                metadata_root="/tmp/meta",
            )
            dao.add_tasktype(
                uuid=tt_uuid,
                exec_id=exec_id,
                order=0,
                name="step",
                is_parallel=False,
                children=[],
            )
        return dao.add_task(
            tasktype_uuid=tt_uuid, exec_id=exec_id, _start_timestamp=_NOW
        )

    def test_add_single_trace(self, sync_dao):
        task_id = self._seed_task(sync_dao)
        with patch("requests.post"):
            trace_id = sync_dao.add_task_trace(
                task_id=task_id,
                timestamp=_NOW,
                microsec=0,
                microsec_idx=1,
                content="log line",
                is_err=False,
            )
        assert isinstance(trace_id, int)

    def test_batch_add_traces(self, sync_dao):
        task_id = self._seed_task(sync_dao)
        rows = [
            dict(
                task_id=task_id,
                timestamp=_NOW,
                microsec=i,
                microsec_idx=1,
                content=f"line {i}",
                is_err=False,
            )
            for i in range(10)
        ]
        with patch("requests.post"):
            sync_dao.batch_add_task_traces(items=rows)


# ==============================================================================
# DAO - concurrent stress – isolated_dao (NullPool)
# ==============================================================================


class TestDAOConcurrentWrites:
    def test_concurrent_add_executions(self, isolated_dao):
        n_threads = 12
        results, errors = [], []

        def worker():
            try:
                with patch("requests.post"):
                    eid = isolated_dao.add_execution(
                        name="concurrent_pipe",
                        username="u",
                        _start_timestamp=_NOW,
                        metadata_root="/tmp/meta",
                    )
                results.append(eid)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent writes raised: {errors}"
        assert len(results) == n_threads and len(set(results)) == n_threads

    def test_concurrent_add_tasks_same_execution(self, isolated_dao):
        tt_uuid = uuid4()
        with patch("requests.post"):
            exec_id = isolated_dao.add_execution(
                name="stress",
                username="u",
                _start_timestamp=_NOW,
                metadata_root="/tmp/meta",
            )
            isolated_dao.add_tasktype(
                uuid=tt_uuid,
                exec_id=exec_id,
                order=0,
                name="s",
                is_parallel=True,
                children=[],
            )

        n_threads = 16
        results, errors = [], []

        def worker():
            try:
                with patch("requests.post"):
                    tid = isolated_dao.add_task(
                        tasktype_uuid=tt_uuid, exec_id=exec_id, _start_timestamp=_NOW
                    )
                results.append(tid)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent task inserts raised: {errors}"
        assert len(results) == n_threads and len(set(results)) == n_threads

    def test_concurrent_batch_trace_inserts(self, isolated_dao):
        tt_uuid = uuid4()
        with patch("requests.post"):
            exec_id = isolated_dao.add_execution(
                name="trace_stress",
                username="u",
                _start_timestamp=_NOW,
                metadata_root="/tmp/meta",
            )
            isolated_dao.add_tasktype(
                uuid=tt_uuid,
                exec_id=exec_id,
                order=0,
                name="s",
                is_parallel=False,
                children=[],
            )
            task_id = isolated_dao.add_task(
                tasktype_uuid=tt_uuid, exec_id=exec_id, _start_timestamp=_NOW
            )

        n_threads = 8
        errors = []

        def worker(idx):
            rows = [
                dict(
                    task_id=task_id,
                    timestamp=_NOW,
                    microsec=idx * 10 + i,
                    microsec_idx=i,
                    content=f"t{idx}-{i}",
                    is_err=False,
                )
                for i in range(5)
            ]
            try:
                with patch("requests.post"):
                    isolated_dao.batch_add_task_traces(items=rows)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent batch inserts raised: {errors}"

    def test_concurrent_read_write_mix(self, isolated_dao):
        tt_uuid = uuid4()
        with patch("requests.post"):
            exec_id = isolated_dao.add_execution(
                name="rw_mix",
                username="u",
                _start_timestamp=_NOW,
                metadata_root="/tmp/meta",
            )
            isolated_dao.add_tasktype(
                uuid=tt_uuid,
                exec_id=exec_id,
                order=0,
                name="s",
                is_parallel=False,
                children=[],
            )

        write_ids, read_results, errors = [], [], []

        def writer():
            try:
                with patch("requests.post"):
                    tid = isolated_dao.add_task(
                        tasktype_uuid=tt_uuid, exec_id=exec_id, _start_timestamp=_NOW
                    )
                write_ids.append(tid)
            except Exception as exc:
                errors.append(exc)

        def reader():
            try:
                result = isolated_dao.get_execution(exec_id)
                read_results.append(result is not None)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=writer) for _ in range(8)] + [
            threading.Thread(target=reader) for _ in range(8)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Mixed read/write raised: {errors}"
        assert len(read_results) == 8

    def test_update_entity_raises_after_max_attempts(self, isolated_dao):
        with patch("requests.post"):
            exec_id = isolated_dao.add_execution(
                name="upd_fail",
                username="u",
                _start_timestamp=_NOW,
                metadata_root="/tmp/meta",
            )

        with (
            patch.object(
                isolated_dao,
                "_sync_update_entity",
                side_effect=Exception("persistent update lock"),
            ),
            patch("time.sleep"),
            patch("requests.post"),
        ):
            with pytest.raises(Exception, match="persistent update lock"):
                isolated_dao.update_execution(exec_id, username="x")


# ==============================================================================
# DAO - dispose
# ==============================================================================


class TestDAODispose:
    def test_dispose_idempotent(self, sync_dao):
        from sqlalchemy.pool import StaticPool

        with patch("sqlalchemy.pool.NullPool", StaticPool), patch("requests.post"):
            from retrain_pipelines.dag_engine.db.dao import DAO as _DAO

            dao = _DAO(db_url="sqlite:///:memory:")
        dao.dispose()

    def test_dispose_without_session_attr(self, tmp_path):
        from sqlalchemy.pool import StaticPool

        with patch("sqlalchemy.pool.NullPool", StaticPool), patch("requests.post"):
            from retrain_pipelines.dag_engine.db.dao import DAO as _DAO

            dao = _DAO(db_url=f"sqlite:///{tmp_path / 'disp.db'}")
        del dao.Session
        dao.dispose()

    def test_dispose_session_remove_raises(self, tmp_path):
        from sqlalchemy.pool import StaticPool

        with patch("sqlalchemy.pool.NullPool", StaticPool), patch("requests.post"):
            from retrain_pipelines.dag_engine.db.dao import DAO as _DAO

            dao = _DAO(db_url=f"sqlite:///{tmp_path / 'disp2.db'}")

        session_mock = MagicMock()
        session_mock.remove.side_effect = RuntimeError("boom")
        dao.Session = session_mock
        dao.dispose()

    def test_dispose_engine_dispose_raises(self, tmp_path):
        from sqlalchemy.pool import StaticPool

        with patch("sqlalchemy.pool.NullPool", StaticPool), patch("requests.post"):
            from retrain_pipelines.dag_engine.db.dao import DAO as _DAO

            dao = _DAO(db_url=f"sqlite:///{tmp_path / 'disp3.db'}")

        engine_mock = MagicMock()
        engine_mock.dispose.side_effect = RuntimeError("boom")
        engine_mock.url = dao.engine.url
        del dao.Session
        dao.engine = engine_mock
        dao.dispose()


# ==============================================================================
# DAOBase non-sqlite (QueuePool) construction path
# ==============================================================================


class TestDAOBaseNonSqlite:
    def test_non_sqlite_engine_uses_queuepool(self):
        from sqlalchemy import QueuePool
        from retrain_pipelines.dag_engine.db.dao import DAO as _DAO

        fake_engine = MagicMock()
        fake_engine.url = MagicMock()
        fake_engine.url.__str__ = lambda s: "postgresql://x"

        with (
            patch(
                "retrain_pipelines.dag_engine.db.dao.create_engine",
                return_value=fake_engine,
            ) as mock_ce,
            patch("retrain_pipelines.dag_engine.db.dao.Base.metadata.create_all"),
            patch(
                "retrain_pipelines.dag_engine.db.dao.sessionmaker",
                return_value=MagicMock(),
            ),
            patch(
                "retrain_pipelines.dag_engine.db.dao.scoped_session",
                return_value=MagicMock(),
            ),
            patch("requests.post"),
        ):
            _DAO(db_url="postgresql://user:pass@localhost/db")

        _, kwargs = mock_ce.call_args
        assert kwargs.get("poolclass") is QueuePool


# ==============================================================================
# DAO - Retry / backoff paths
# ==============================================================================


class TestRetryPaths:
    def test_add_entity_retries_then_succeeds(self, isolated_dao, capture_log):
        """Exercise retry loop and assert on programmatically captured logs."""
        call_count = {"n": 0}
        original = isolated_dao._sync_add_entity

        def flaky(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise Exception("simulated lock")
            return original(*args, **kwargs)

        # Bypass caplog's flaky handler attachment. Capture directly to a buffer.
        with capture_log("retrain_pipelines.dag_engine.db.dao") as log_capture:
            with (
                patch.object(isolated_dao, "_sync_add_entity", side_effect=flaky),
                patch("requests.post"),
                patch("time.sleep"),
            ):
                isolated_dao.add_execution(
                    name="retry_pipe",
                    username="u",
                    _start_timestamp=_NOW,
                    metadata_root="/tmp/meta",
                )

        assert call_count["n"] == 3

        # Exact assertion as intended
        caplog_text = log_capture.getvalue()
        # assert any("simulated lock" in msg for msg in caplog_text)
        assert "[blink]_sync_add_entity - 3 -" in caplog_text, (
            f"Expected log not found. Captured: {caplog_text}"
        )

    def test_add_entity_raises_after_max_attempts(self, isolated_dao):
        with (
            patch.object(
                isolated_dao,
                "_sync_add_entity",
                side_effect=Exception("persistent lock"),
            ),
            patch("time.sleep"),
            patch("requests.post"),
        ):
            with pytest.raises(Exception, match="persistent lock"):
                isolated_dao.add_execution(
                    name="fail_pipe",
                    username="u",
                    _start_timestamp=_NOW,
                    metadata_root="/tmp/meta",
                )

    def test_batch_add_entities_retries_then_succeeds(self, isolated_dao):
        call_count = {"n": 0}
        original = isolated_dao._sync_batch_add_entities

        def flaky(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] < 2:
                raise Exception("simulated lock")
            return original(*args, **kwargs)

        with patch("requests.post"):
            exec_id = isolated_dao.add_execution(
                name="batch_retry",
                username="u",
                _start_timestamp=_NOW,
                metadata_root="/tmp/meta",
            )
            tt_uuid = uuid4()
            isolated_dao.add_tasktype(
                uuid=tt_uuid,
                exec_id=exec_id,
                order=0,
                name="s",
                is_parallel=False,
                children=[],
            )
            task_id = isolated_dao.add_task(
                tasktype_uuid=tt_uuid, exec_id=exec_id, _start_timestamp=_NOW
            )

        rows = [
            dict(
                task_id=task_id,
                timestamp=_NOW,
                microsec=i,
                microsec_idx=1,
                content=f"c{i}",
                is_err=False,
            )
            for i in range(3)
        ]

        with (
            patch.object(isolated_dao, "_sync_batch_add_entities", side_effect=flaky),
            patch("time.sleep"),
            patch("requests.post"),
        ):
            isolated_dao.batch_add_task_traces(items=rows)

        assert call_count["n"] == 2

    def test_batch_add_entities_raises_after_max_attempts(self, isolated_dao):
        with patch("requests.post"):
            exec_id = isolated_dao.add_execution(
                name="batch_fail",
                username="u",
                _start_timestamp=_NOW,
                metadata_root="/tmp/meta",
            )
            tt_uuid = uuid4()
            isolated_dao.add_tasktype(
                uuid=tt_uuid,
                exec_id=exec_id,
                order=0,
                name="s",
                is_parallel=False,
                children=[],
            )
            task_id = isolated_dao.add_task(
                tasktype_uuid=tt_uuid, exec_id=exec_id, _start_timestamp=_NOW
            )

        rows = [
            dict(
                task_id=task_id,
                timestamp=_NOW,
                microsec=0,
                microsec_idx=1,
                content="x",
                is_err=False,
            )
        ]

        with (
            patch.object(
                isolated_dao,
                "_sync_batch_add_entities",
                side_effect=Exception("persistent"),
            ),
            patch("time.sleep"),
            patch("requests.post"),
        ):
            with pytest.raises(Exception, match="persistent"):
                isolated_dao.batch_add_task_traces(items=rows)

    def test_update_entity_retries_then_succeeds(self, isolated_dao):
        with patch("requests.post"):
            exec_id = isolated_dao.add_execution(
                name="upd_retry",
                username="u",
                _start_timestamp=_NOW,
                metadata_root="/tmp/meta",
            )

        call_count = {"n": 0}
        original = isolated_dao._sync_update_entity

        def flaky(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] < 2:
                raise Exception("lock")
            return original(*args, **kwargs)

        with (
            patch.object(isolated_dao, "_sync_update_entity", side_effect=flaky),
            patch("time.sleep"),
            patch("requests.post"),
        ):
            result = isolated_dao.update_execution(exec_id, username="new_u")

        assert call_count["n"] == 2
        assert result is not None

    def test_sync_update_entity_not_found_returns_none(self, isolated_dao):
        result = isolated_dao._sync_update_entity(Execution, 999999, username="x")
        assert result is None

    def test_sync_update_entity_rollback_on_exception(self, isolated_dao):
        with patch("requests.post"):
            exec_id = isolated_dao.add_execution(
                name="rb_pipe",
                username="u",
                _start_timestamp=_NOW,
                metadata_root="/tmp/meta",
            )

        session = isolated_dao.Session()
        session.rollback_called = False
        original_rollback = session.rollback

        def tracking_rollback():
            session.rollback_called = True
            original_adopted = original_rollback()
            return original_adopted

        session.rollback = tracking_rollback

        with (
            patch.object(isolated_dao, "_get_session", return_value=session),
            patch("requests.post"),
        ):
            original_get = session.get

            def get_then_raise(cls, pk):
                entity = original_get(cls, pk)
                if entity is not None:
                    raise RuntimeError("forced")
                return entity

            session.get = get_then_raise
            with pytest.raises(RuntimeError, match="forced"):
                isolated_dao._sync_update_entity(Execution, exec_id, username="x")

        assert session.rollback_called

    def test_sync_add_entity_rollback_on_exception(self, isolated_dao):
        session = isolated_dao.Session()
        session.rollback_called = False
        original_rollback = session.rollback

        def tracking_rollback():
            session.rollback_called = True
            original_rollback()

        session.rollback = tracking_rollback

        with (
            patch.object(isolated_dao, "_get_session", return_value=session),
            patch("requests.post"),
        ):
            session.add = MagicMock(side_effect=RuntimeError("forced add"))
            with pytest.raises(RuntimeError, match="forced add"):
                isolated_dao._sync_add_entity(
                    Execution,
                    name="x",
                    username="u",
                    _start_timestamp=_NOW,
                    metadata_root="/tmp/meta",
                )

        assert session.rollback_called

    def test_sync_batch_add_entities_rollback_on_exception(self, isolated_dao):
        session = isolated_dao.Session()
        session.rollback_called = False
        original_rollback = session.rollback

        def tracking_rollback():
            session.rollback_called = True
            original_rollback()

        session.rollback = tracking_rollback

        with (
            patch.object(isolated_dao, "_get_session", return_value=session),
            patch("requests.post"),
        ):
            session.add_all = MagicMock(side_effect=RuntimeError("forced batch"))
            with pytest.raises(RuntimeError, match="forced batch"):
                isolated_dao._sync_batch_add_entities(
                    TaskTrace,
                    items=[
                        dict(
                            task_id=1,
                            timestamp=_NOW,
                            microsec=0,
                            microsec_idx=1,
                            content="x",
                            is_err=False,
                        )
                    ],
                )

        assert session.rollback_called


# ==============================================================================
# Event listeners (sync side-effects)
# ==============================================================================


class TestEventListeners:
    def test_after_insert_execution_posts_to_endpoint(self):
        from retrain_pipelines.dag_engine.db.dao import after_insert_execution_listener

        target = MagicMock(spec=Execution)
        target.__table__ = Execution.__table__
        for col in Execution.__table__.columns:
            setattr(target, col.name, None)
        target.name = "p"
        target._start_timestamp = _NOW

        with patch("requests.post") as mock_post:
            after_insert_execution_listener(None, None, target)
        mock_post.assert_called_once()

    def test_after_insert_execution_handles_connection_error(self):
        from retrain_pipelines.dag_engine.db.dao import after_insert_execution_listener
        import requests

        target = MagicMock(spec=Execution)
        target.__table__ = Execution.__table__
        for col in Execution.__table__.columns:
            setattr(target, col.name, None)

        with patch("requests.post", side_effect=requests.exceptions.ConnectionError):
            after_insert_execution_listener(None, None, target)

    def test_after_insert_execution_handles_generic_exception(self):
        from retrain_pipelines.dag_engine.db.dao import after_insert_execution_listener

        target = MagicMock(spec=Execution)
        target.__table__ = Execution.__table__
        for col in Execution.__table__.columns:
            setattr(target, col.name, None)

        with patch("requests.post", side_effect=RuntimeError("unexpected")):
            after_insert_execution_listener(None, None, target)

    def test_after_end_timestamp_change_posts_when_new_value(self, isolated_dao):
        from retrain_pipelines.dag_engine.db.dao import after_end_timestamp_change

        target = MagicMock(spec=Execution)
        target.__table__ = Execution.__table__
        for col in Execution.__table__.columns:
            setattr(target, col.name, None)
        target.tasks = []

        with patch("requests.post") as mock_post:
            after_end_timestamp_change(target, _NOW + timedelta(hours=1), None, None)
        mock_post.assert_called_once()

    def test_after_end_timestamp_change_skips_when_none(self):
        from retrain_pipelines.dag_engine.db.dao import after_end_timestamp_change

        with patch("requests.post") as mock_post:
            after_end_timestamp_change(MagicMock(), None, None, None)
        mock_post.assert_not_called()

    def test_after_end_timestamp_change_skips_when_unchanged(self):
        from retrain_pipelines.dag_engine.db.dao import after_end_timestamp_change

        with patch("requests.post") as mock_post:
            after_end_timestamp_change(MagicMock(), _NOW, _NOW, None)
        mock_post.assert_not_called()

    def test_after_end_timestamp_change_connection_error(self):
        from retrain_pipelines.dag_engine.db.dao import after_end_timestamp_change
        import requests

        target = MagicMock(spec=Execution)
        target.__table__ = Execution.__table__
        for col in Execution.__table__.columns:
            setattr(target, col.name, None)
        target.tasks = []

        with patch("requests.post", side_effect=requests.exceptions.ConnectionError):
            after_end_timestamp_change(target, _NOW + timedelta(hours=1), None, None)

    def test_after_end_timestamp_change_generic_exception(self):
        from retrain_pipelines.dag_engine.db.dao import after_end_timestamp_change

        target = MagicMock(spec=Execution)
        target.__table__ = Execution.__table__
        for col in Execution.__table__.columns:
            setattr(target, col.name, None)
        target.tasks = []

        with patch("requests.post", side_effect=RuntimeError("unexpected")):
            after_end_timestamp_change(target, _NOW + timedelta(hours=1), None, None)

    def test_after_insert_task_listener_posts_when_tasktype_found(self, isolated_dao):
        from retrain_pipelines.dag_engine.db.dao import after_insert_task_listener

        with patch("requests.post"):
            exec_id = isolated_dao.add_execution(
                name="p", username="u", _start_timestamp=_NOW, metadata_root="/tmp/meta"
            )
            tt_uuid = uuid4()
            isolated_dao.add_tasktype(
                uuid=tt_uuid,
                exec_id=exec_id,
                order=0,
                name="step",
                is_parallel=False,
                children=[],
            )

        target = MagicMock(spec=Task)
        target.__table__ = Task.__table__
        for col in Task.__table__.columns:
            setattr(target, col.name, None)
        target.exec_id = exec_id
        target.tasktype_uuid = tt_uuid

        conn_mock = MagicMock()
        conn_mock.engine = isolated_dao.engine

        with patch("requests.post") as mock_post:
            after_insert_task_listener(None, conn_mock, target)
        mock_post.assert_called_once()

    def test_after_insert_task_listener_tasktype_not_found_logs_warning(
        self, isolated_dao
    ):
        from retrain_pipelines.dag_engine.db.dao import after_insert_task_listener

        with patch("requests.post"):
            exec_id = isolated_dao.add_execution(
                name="p", username="u", _start_timestamp=_NOW, metadata_root="/tmp/meta"
            )

        target = MagicMock(spec=Task)
        target.__table__ = Task.__table__
        for col in Task.__table__.columns:
            setattr(target, col.name, None)
        target.exec_id = exec_id
        target.tasktype_uuid = uuid4()

        conn_mock = MagicMock()
        conn_mock.engine = isolated_dao.engine

        with patch("requests.post"):
            after_insert_task_listener(None, conn_mock, target)

    def test_after_insert_task_listener_session_get_raises(self, isolated_dao):
        from retrain_pipelines.dag_engine.db.dao import after_insert_task_listener

        target = MagicMock(spec=Task)
        target.__table__ = Task.__table__
        for col in Task.__table__.columns:
            setattr(target, col.name, None)
        target.exec_id = 1
        target.tasktype_uuid = uuid4()

        broken_session = MagicMock()
        broken_session.get.side_effect = RuntimeError("db exploded")
        broken_session.__enter__ = MagicMock(return_value=broken_session)
        broken_session.__exit__ = MagicMock(return_value=False)
        broken_scoped = MagicMock(return_value=broken_session)

        conn_mock = MagicMock()
        conn_mock.engine = isolated_dao.engine

        with (
            patch(
                "retrain_pipelines.dag_engine.db.dao.scoped_session",
                return_value=broken_scoped,
            ),
            patch(
                "retrain_pipelines.dag_engine.db.dao.sessionmaker",
                return_value=MagicMock(),
            ),
            patch("requests.post"),
        ):
            after_insert_task_listener(None, conn_mock, target)

    def test_after_insert_task_listener_connection_error(self, isolated_dao):
        from retrain_pipelines.dag_engine.db.dao import after_insert_task_listener
        import requests

        target = MagicMock(spec=Task)
        target.__table__ = Task.__table__
        for col in Task.__table__.columns:
            setattr(target, col.name, None)
        target.exec_id = 9999
        target.tasktype_uuid = uuid4()

        conn_mock = MagicMock()
        conn_mock.engine = isolated_dao.engine

        with patch("requests.post", side_effect=requests.exceptions.ConnectionError):
            after_insert_task_listener(None, conn_mock, target)

    def test_after_insert_task_listener_generic_post_exception(self, isolated_dao):
        from retrain_pipelines.dag_engine.db.dao import after_insert_task_listener

        target = MagicMock(spec=Task)
        target.__table__ = Task.__table__
        for col in Task.__table__.columns:
            setattr(target, col.name, None)
        target.exec_id = 9999
        target.tasktype_uuid = uuid4()

        conn_mock = MagicMock()
        conn_mock.engine = isolated_dao.engine

        with patch("requests.post", side_effect=RuntimeError("unexpected")):
            after_insert_task_listener(None, conn_mock, target)

    def test_after_task_update_posts_when_end_timestamp_set(self, isolated_dao):
        from retrain_pipelines.dag_engine.db.dao import after_task_update

        with patch("requests.post"):
            exec_id = isolated_dao.add_execution(
                name="p", username="u", _start_timestamp=_NOW, metadata_root="/tmp/meta"
            )
            tt_uuid = uuid4()
            isolated_dao.add_tasktype(
                uuid=tt_uuid,
                exec_id=exec_id,
                order=0,
                name="step",
                is_parallel=False,
                children=[],
            )
            task_id = isolated_dao.add_task(
                tasktype_uuid=tt_uuid, exec_id=exec_id, _start_timestamp=_NOW
            )

        session = isolated_dao.Session()
        task_orm = session.get(Task, task_id)

        with session.no_autoflush:
            task_orm._end_timestamp = _NOW + timedelta(minutes=1)
            task_orm.failed = False
            conn_mock = MagicMock()
            conn_mock.engine = isolated_dao.engine

            with patch("requests.post") as mock_post:
                after_task_update(None, conn_mock, task_orm)
            mock_post.assert_called_once()
        session.close()

    def test_after_task_update_skips_when_no_end_timestamp(self, isolated_dao):
        from retrain_pipelines.dag_engine.db.dao import after_task_update

        target = MagicMock(spec=Task)
        target._end_timestamp = None

        with patch("requests.post") as mock_post:
            after_task_update(None, None, target)
        mock_post.assert_not_called()

    def test_after_task_update_skips_when_no_active_session(self, isolated_dao):
        from retrain_pipelines.dag_engine.db.dao import after_task_update

        target = MagicMock(spec=Task)
        target.__table__ = Task.__table__
        for col in Task.__table__.columns:
            setattr(target, col.name, None)
        target._end_timestamp = _NOW

        with (
            patch(
                "retrain_pipelines.dag_engine.db.dao.object_session", return_value=None
            ),
            patch("requests.post") as mock_post,
        ):
            after_task_update(None, None, target)
        mock_post.assert_not_called()

    def test_after_task_update_session_get_raises(self, isolated_dao, capture_log):
        from retrain_pipelines.dag_engine.db.dao import after_task_update

        target = MagicMock(spec=Task)
        target.__table__ = Task.__table__
        for col in Task.__table__.columns:
            setattr(target, col.name, None)
        target._end_timestamp = _NOW
        target.exec_id = 1
        target.tasktype_uuid = uuid4()

        mock_session = MagicMock()
        mock_session.get.side_effect = RuntimeError("db exploded")

        with capture_log(
            "retrain_pipelines.dag_engine.db.dao", level=logging.ERROR
        ) as captured:
            with (
                patch(
                    "retrain_pipelines.dag_engine.db.dao.object_session",
                    return_value=mock_session,
                ),
                patch("requests.post"),
            ):
                after_task_update(None, None, target)

        assert any(
            "Worker thread crashed" in msg for msg in captured.getvalue().splitlines()
        )

    def test_after_task_update_connection_error(self, isolated_dao):
        import requests
        from retrain_pipelines.dag_engine.db.dao import after_task_update

        with patch("requests.post"):
            exec_id = isolated_dao.add_execution(
                name="p", username="u", _start_timestamp=_NOW, metadata_root="/tmp/meta"
            )
            tt_uuid = uuid4()
            isolated_dao.add_tasktype(
                uuid=tt_uuid,
                exec_id=exec_id,
                order=0,
                name="step",
                is_parallel=False,
                children=[],
            )
            task_id = isolated_dao.add_task(
                tasktype_uuid=tt_uuid, exec_id=exec_id, _start_timestamp=_NOW
            )

        session = isolated_dao.Session()
        task_orm = session.get(Task, task_id)

        with session.no_autoflush:
            task_orm._end_timestamp = _NOW + timedelta(minutes=1)
            task_orm.failed = False
            conn_mock = MagicMock()
            conn_mock.engine = isolated_dao.engine

            with patch(
                "requests.post", side_effect=requests.exceptions.ConnectionError
            ):
                after_task_update(None, conn_mock, task_orm)
        session.close()

    def test_after_task_update_generic_post_exception(self, isolated_dao):
        from retrain_pipelines.dag_engine.db.dao import after_task_update

        with patch("requests.post"):
            exec_id = isolated_dao.add_execution(
                name="p", username="u", _start_timestamp=_NOW, metadata_root="/tmp/meta"
            )
            tt_uuid = uuid4()
            isolated_dao.add_tasktype(
                uuid=tt_uuid,
                exec_id=exec_id,
                order=0,
                name="step",
                is_parallel=False,
                children=[],
            )
            task_id = isolated_dao.add_task(
                tasktype_uuid=tt_uuid, exec_id=exec_id, _start_timestamp=_NOW
            )

        session = isolated_dao.Session()
        task_orm = session.get(Task, task_id)

        with session.no_autoflush:
            task_orm._end_timestamp = _NOW + timedelta(minutes=1)
            task_orm.failed = False
            conn_mock = MagicMock()
            conn_mock.engine = isolated_dao.engine

            with patch("requests.post", side_effect=RuntimeError("unexpected")):
                after_task_update(None, conn_mock, task_orm)
        session.close()

    def test_after_insert_task_trace_grpc_sends_when_initiated(self):
        from retrain_pipelines.dag_engine.db.dao import after_insert_task_trace_listener

        target = MagicMock(spec=TaskTrace)
        target.id = 1
        target.task_id = 1
        target.timestamp = _NOW
        target.microsec = 0
        target.microsec_idx = 1
        target.content = "msg"
        target.is_err = False

        stub_mock = MagicMock()
        with (
            patch(
                "retrain_pipelines.dag_engine.db.dao.GrpcClient.initiated",
                return_value=True,
            ),
            patch(
                "retrain_pipelines.dag_engine.db.dao.GrpcClient.stub",
                return_value=stub_mock,
            ),
        ):
            after_insert_task_trace_listener(None, None, target)
        stub_mock.SendTrace.assert_called_once()

    def test_after_insert_task_trace_skips_when_not_initiated(self):
        from retrain_pipelines.dag_engine.db.dao import after_insert_task_trace_listener

        target = MagicMock(spec=TaskTrace)
        target.timestamp = _NOW

        with patch(
            "retrain_pipelines.dag_engine.db.dao.GrpcClient.initiated",
            return_value=False,
        ):
            after_insert_task_trace_listener(None, None, target)

    def test_after_insert_task_trace_handles_grpc_error(self):
        import grpc
        from retrain_pipelines.dag_engine.db.dao import after_insert_task_trace_listener

        target = MagicMock(spec=TaskTrace)
        target.id = 1
        target.task_id = 1
        target.timestamp = _NOW
        target.microsec = 0
        target.microsec_idx = 1
        target.content = "msg"
        target.is_err = False

        stub_mock = MagicMock()
        rpc_error = MagicMock()
        rpc_error.code.return_value = grpc.StatusCode.UNAVAILABLE
        rpc_error.details.return_value = "unavailable"
        stub_mock.SendTrace.side_effect = rpc_error

        with (
            patch(
                "retrain_pipelines.dag_engine.db.dao.GrpcClient.initiated",
                return_value=True,
            ),
            patch(
                "retrain_pipelines.dag_engine.db.dao.GrpcClient.stub",
                return_value=stub_mock,
            ),
        ):
            after_insert_task_trace_listener(None, None, target)

    def test_after_insert_task_trace_handles_generic_exception(self):
        from retrain_pipelines.dag_engine.db.dao import after_insert_task_trace_listener

        target = MagicMock(spec=TaskTrace)
        target.id = 1
        target.task_id = 1
        target.timestamp = _NOW
        target.microsec = 0
        target.microsec_idx = 1
        target.content = "msg"
        target.is_err = False

        stub_mock = MagicMock()
        stub_mock.SendTrace.side_effect = RuntimeError("unexpected")

        with (
            patch(
                "retrain_pipelines.dag_engine.db.dao.GrpcClient.initiated",
                return_value=True,
            ),
            patch(
                "retrain_pipelines.dag_engine.db.dao.GrpcClient.stub",
                return_value=stub_mock,
            ),
        ):
            after_insert_task_trace_listener(None, None, target)

    def test_after_insert_task_trace_grpc_error_logs_details(self, capture_log):
        import grpc
        from retrain_pipelines.dag_engine.db.dao import after_insert_task_trace_listener

        target = MagicMock(spec=TaskTrace)
        target.id = 1
        target.task_id = 1
        target.timestamp = _NOW
        target.microsec = 0
        target.microsec_idx = 1
        target.content = "boom_msg"
        target.is_err = True

        class FakeRpcError(grpc.RpcError):
            def code(self):
                return grpc.StatusCode.UNAVAILABLE

            def details(self):
                return "server down"

        stub_mock = MagicMock()
        stub_mock.SendTrace.side_effect = FakeRpcError()

        with capture_log(
            "retrain_pipelines.dag_engine.db.dao", level=logging.ERROR
        ) as captured:
            with (
                patch(
                    "retrain_pipelines.dag_engine.db.dao.GrpcClient.initiated",
                    return_value=True,
                ),
                patch(
                    "retrain_pipelines.dag_engine.db.dao.GrpcClient.stub",
                    return_value=stub_mock,
                ),
            ):
                after_insert_task_trace_listener(None, None, target)

        assert any(
            "boom_msg" in line and "server down" in line
            for line in captured.getvalue().splitlines()
        )


# ==============================================================================
# AsyncDAO
# ==============================================================================


class TestAsyncDAO:
    @pytest.mark.asyncio
    async def test_async_add_and_get_entity(self, async_dao):
        exec_id = await async_dao._async_add_entity(
            Execution,
            name="async_pipe",
            username="u",
            _start_timestamp=_NOW,
            metadata_root="/tmp/meta",
        )
        assert isinstance(exec_id, int)
        result = await async_dao._async_get_entity(Execution, id=exec_id)
        assert result is not None and result.name == "async_pipe"

    @pytest.mark.asyncio
    async def test_async_get_entity_missing_returns_none(self, async_dao):
        result = await async_dao._async_get_entity(Execution, id=999999)
        assert result is None

    @pytest.mark.asyncio
    async def test_async_get_entities(self, async_dao):
        for _ in range(3):
            await async_dao._async_add_entity(
                Execution,
                name="multi_pipe",
                username="u",
                _start_timestamp=_NOW,
                metadata_root="/tmp/meta",
            )
        results = await async_dao._async_get_entities(Execution, name="multi_pipe")
        assert len(results) >= 3

    @pytest.mark.asyncio
    async def test_async_update_entity(self, async_dao):
        exec_id = await async_dao._async_add_entity(
            Execution,
            name="upd_pipe",
            username="u",
            _start_timestamp=_NOW,
            metadata_root="/tmp/meta",
        )
        updated = await async_dao._async_update_entity(
            Execution, exec_id, username="updated_user"
        )
        assert updated is not None and updated.username == "updated_user"

    @pytest.mark.asyncio
    async def test_async_update_entity_not_found(self, async_dao, _null_session):
        """scalar_one_or_none()=None => return None."""
        with patch.object(async_dao, "_get_session", return_value=_null_session):
            result = await async_dao._async_update_entity(Execution, entity_id=99999)
        assert result is None

    @pytest.mark.asyncio
    async def test_async_batch_add_entities(self, async_dao):
        from sqlalchemy import text

        exec_id = await async_dao._async_add_entity(
            Execution,
            name="batch_pipe",
            username="u",
            _start_timestamp=_NOW,
            metadata_root="/tmp/meta",
        )
        tt_uuid = uuid4()
        async with async_dao.engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO tasktypes (uuid, exec_id, \"order\", name, is_parallel, children) VALUES (:uuid, :eid, 0, 's', 0, '[]')"
                ),
                {"uuid": tt_uuid.hex, "eid": exec_id},
            )
            await conn.execute(
                text(
                    "INSERT INTO tasks (tasktype_uuid, exec_id, start_timestamp) VALUES (:tu, :eid, :s)"
                ),
                {"tu": tt_uuid.hex, "eid": exec_id, "s": _NOW.isoformat()},
            )
            row = await conn.execute(text("SELECT last_insert_rowid()"))
            task_id = row.scalar()

        rows = [
            dict(
                task_id=task_id,
                timestamp=_NOW,
                microsec=i,
                microsec_idx=1,
                content=f"c{i}",
                is_err=False,
            )
            for i in range(4)
        ]
        await async_dao._async_batch_add_entities(TaskTrace, items=rows)

    @pytest.mark.asyncio
    async def test_add_entity_dispatches_to_async(self, async_dao):
        result = await async_dao._add_entity(
            Execution,
            name="dispatch_pipe",
            username="u",
            _start_timestamp=_NOW,
            metadata_root="/tmp/meta",
        )
        assert isinstance(result, int)

    @pytest.mark.asyncio
    async def test_batch_add_entities_dispatches_to_async(self, async_dao):
        from sqlalchemy import text

        exec_id = await async_dao._async_add_entity(
            Execution,
            name="disp_batch",
            username="u",
            _start_timestamp=_NOW,
            metadata_root="/tmp/meta",
        )
        tt_uuid = uuid4()
        async with async_dao.engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO tasktypes (uuid, exec_id, \"order\", name, is_parallel, children) VALUES (:uuid, :eid, 0, 's', 0, '[]')"
                ),
                {"uuid": tt_uuid.hex, "eid": exec_id},
            )
            await conn.execute(
                text(
                    "INSERT INTO tasks (tasktype_uuid, exec_id, start_timestamp) VALUES (:tu, :eid, :s)"
                ),
                {"tu": tt_uuid.hex, "eid": exec_id, "s": _NOW.isoformat()},
            )
            row = await conn.execute(text("SELECT last_insert_rowid()"))
            task_id = row.scalar()

        rows = [
            dict(
                task_id=task_id,
                timestamp=_NOW,
                microsec=0,
                microsec_idx=1,
                content="x",
                is_err=False,
            )
        ]
        await async_dao._batch_add_entities(TaskTrace, items=rows)

    @pytest.mark.asyncio
    async def test_update_entity_dispatches_to_async(self, async_dao):
        exec_id = await async_dao._async_add_entity(
            Execution,
            name="disp_update",
            username="u",
            _start_timestamp=_NOW,
            metadata_root="/tmp/meta",
        )
        result = await async_dao._update_entity(Execution, exec_id, username="new_u")
        assert result is not None and result.username == "new_u"

    @pytest.mark.asyncio
    async def test_get_entity_dispatches_to_async(self, async_dao):
        exec_id = await async_dao._async_add_entity(
            Execution,
            name="disp_get",
            username="u",
            _start_timestamp=_NOW,
            metadata_root="/tmp/meta",
        )
        result = await async_dao._get_entity(Execution, id=exec_id)
        assert result is not None

    @pytest.mark.asyncio
    async def test_get_entities_dispatches_to_async(self, async_dao):
        for _ in range(2):
            await async_dao._async_add_entity(
                Execution,
                name="disp_gets",
                username="u",
                _start_timestamp=_NOW,
                metadata_root="/tmp/meta",
            )
        results = await async_dao._get_entities(Execution, name="disp_gets")
        assert len(results) >= 2


# ==============================================================================
# AsyncDAOExecutionExt
# ==============================================================================


class TestAsyncDAOExecutionExt:
    @pytest.mark.asyncio
    async def test_get_execution_ext_found(self, async_dao):
        """full get_execution_ext happy path."""
        exec_id = await async_dao._async_add_entity(
            Execution,
            name="ext_pipe",
            username="u",
            _start_timestamp=_NOW,
            metadata_root="/tmp/meta",
        )
        result = await async_dao.get_execution_ext(exec_id)
        assert result is not None
        assert result.id == exec_id
        # No failed tasks => success is True
        assert result.success is True

    @pytest.mark.asyncio
    async def test_get_execution_ext_not_found(self, async_dao, _null_session):
        """result.first()=None => return None."""
        with patch.object(async_dao, "_get_session", return_value=_null_session):
            result = await async_dao.get_execution_ext(99999)
        assert result is None


# ==============================================================================
# AsyncDAOExecutionsCount
# ==============================================================================


class TestAsyncDAOExecutionsCount:
    @pytest.mark.asyncio
    async def test_get_executions_count_success(self, seeded_async_dao):
        async_dao, _ = seeded_async_dao
        count = await async_dao.get_executions_count("cnt_pipe", execs_status="success")
        assert isinstance(count, int)

    @pytest.mark.asyncio
    async def test_get_executions_count_failure(self, seeded_async_dao):
        async_dao, _ = seeded_async_dao
        count = await async_dao.get_executions_count("cnt_pipe", execs_status="failure")
        assert isinstance(count, int)

    @pytest.mark.asyncio
    async def test_get_executions_count_invalid_status(self, seeded_async_dao):
        async_dao, _ = seeded_async_dao
        count = await async_dao.get_executions_count("cnt_pipe", execs_status="bogus")
        assert isinstance(count, int)

    @pytest.mark.asyncio
    async def test_get_executions_count_returns_zero_for_unknown_pipeline(
        self, seeded_async_dao
    ):
        async_dao, _ = seeded_async_dao
        count = await async_dao.get_executions_count("__nonexistent_pipe__")
        assert count == 0

    @pytest.mark.asyncio
    async def test_get_executions_count_returns_scalar(self, async_dao):
        """return result.scalar() is reached and returns an int."""
        await async_dao._async_add_entity(
            Execution,
            name="scalar_pipe",
            username="u",
            _start_timestamp=_NOW,
            metadata_root="/tmp/meta",
        )
        count = await async_dao.get_executions_count("scalar_pipe")
        assert count == 1


# ==============================================================================
# AsyncDAOExecutionNames
# ==============================================================================


class TestAsyncDAOExecutionNames:
    @pytest.mark.asyncio
    async def test_get_distinct_execution_names_unsorted(self, seeded_async_dao):
        async_dao, _ = seeded_async_dao
        names = await async_dao.get_distinct_execution_names()
        assert "cnt_pipe" in names

    @pytest.mark.asyncio
    async def test_get_distinct_execution_names_sorted(self, seeded_async_dao):
        async_dao, _ = seeded_async_dao
        names = await async_dao.get_distinct_execution_names(sorted=True)
        assert names == sorted(names)


# ==============================================================================
# AsyncDAOExecutionUsernames
# ==============================================================================


class TestAsyncDAOExecutionUsernames:
    @pytest.mark.asyncio
    async def test_get_distinct_execution_usernames_unsorted(self, seeded_async_dao):
        async_dao, _ = seeded_async_dao
        users = await async_dao.get_distinct_execution_usernames()
        assert isinstance(users, list)

    @pytest.mark.asyncio
    async def test_get_distinct_execution_usernames_sorted(self, seeded_async_dao):
        async_dao, _ = seeded_async_dao
        users = await async_dao.get_distinct_execution_usernames(sorted=True)
        assert users == sorted(users)


# ==============================================================================
# AsyncDAOExecutionsExt
# ==============================================================================


class TestAsyncDAOExecutionsExt:
    @pytest.mark.asyncio
    async def test_get_executions_ext_no_filter(self, seeded_async_dao):
        async_dao, _ = seeded_async_dao
        exts = await async_dao.get_executions_ext()
        assert len(exts) >= 2

    @pytest.mark.asyncio
    async def test_get_executions_ext_filter_name(self, seeded_async_dao):
        async_dao, _ = seeded_async_dao
        exts = await async_dao.get_executions_ext(pipeline_name="cnt_pipe")
        assert all(e.name == "cnt_pipe" for e in exts)

    @pytest.mark.asyncio
    async def test_get_executions_ext_filter_username(self, seeded_async_dao):
        async_dao, _ = seeded_async_dao
        exts = await async_dao.get_executions_ext(username="alice")
        assert all(e.username == "alice" for e in exts)

    @pytest.mark.asyncio
    async def test_get_executions_ext_filter_before_datetime(self, seeded_async_dao):
        async_dao, _ = seeded_async_dao
        exts = await async_dao.get_executions_ext(
            before_datetime=_NOW + timedelta(seconds=1)
        )
        assert isinstance(exts, list)

    @pytest.mark.asyncio
    async def test_get_executions_ext_filter_success(self, seeded_async_dao):
        async_dao, _ = seeded_async_dao
        exts = await async_dao.get_executions_ext(execs_status="success")
        assert all(e.success is True for e in exts)

    @pytest.mark.asyncio
    async def test_get_executions_ext_filter_failure(self, seeded_async_dao):
        async_dao, _ = seeded_async_dao
        exts = await async_dao.get_executions_ext(execs_status="failure")
        assert all(e.success is False for e in exts)

    @pytest.mark.asyncio
    async def test_get_executions_ext_limit(self, seeded_async_dao):
        async_dao, _ = seeded_async_dao
        exts = await async_dao.get_executions_ext(n=1)
        assert len(exts) == 1

    @pytest.mark.asyncio
    async def test_get_executions_ext_descending(self, seeded_async_dao):
        async_dao, _ = seeded_async_dao
        exts = await async_dao.get_executions_ext(
            pipeline_name="cnt_pipe", descending=True
        )
        if len(exts) >= 2:
            assert exts[0]._start_timestamp >= exts[-1]._start_timestamp

    @pytest.mark.asyncio
    async def test_get_executions_ext_ascending_explicit(self, seeded_async_dao):
        async_dao, _ = seeded_async_dao
        exts = await async_dao.get_executions_ext(
            pipeline_name="cnt_pipe", descending=False
        )
        if len(exts) >= 2:
            assert exts[0]._start_timestamp <= exts[-1]._start_timestamp

    @pytest.mark.asyncio
    async def test_get_executions_ext_all_filters_combined(self, seeded_async_dao):
        async_dao, _ = seeded_async_dao
        exts = await async_dao.get_executions_ext(
            pipeline_name="cnt_pipe",
            username="alice",
            before_datetime=_NOW + timedelta(hours=2),
            execs_status="failure",
            n=5,
            descending=False,
        )
        assert isinstance(exts, list)

    @pytest.mark.asyncio
    async def test_get_executions_ext_builds_execution_ext_objects(self, async_dao):
        await async_dao._async_add_entity(
            Execution,
            name="ext_loop_pipe",
            username="u",
            _start_timestamp=_NOW,
            metadata_root="/tmp/meta",
        )
        exts = await async_dao.get_executions_ext(pipeline_name="ext_loop_pipe")
        assert len(exts) >= 1
        for e in exts:
            assert isinstance(e, ExecutionExt)
            assert isinstance(e.success, bool)


# ==============================================================================
# AsyncDAOExecutionTasksList
# ==============================================================================


class TestAsyncDAOExecutionTasksList:
    @pytest.mark.asyncio
    async def test_get_execution_tasks_list_returns_task_exts(self, seeded_async_dao):
        async_dao, (exec_id_with_tasks, _) = seeded_async_dao
        result = await async_dao.get_execution_tasks_list(exec_id_with_tasks)
        assert result is not None and len(result) >= 1
        assert result[0].name is not None

    @pytest.mark.asyncio
    async def test_get_execution_tasks_list_empty_returns_none(self, async_dao):
        exec_id = await async_dao._async_add_entity(
            Execution,
            name="empty_tasks_pipe",
            username="u",
            _start_timestamp=_NOW,
            metadata_root="/tmp/meta",
        )
        result = await async_dao.get_execution_tasks_list(exec_id)
        assert result is None

    @pytest.mark.asyncio
    async def test_get_execution_tasks_list_merge_func_none(self, async_dao):
        from sqlalchemy import text

        exec_id = await async_dao._async_add_entity(
            Execution,
            name="mf_pipe",
            username="u",
            _start_timestamp=_NOW,
            metadata_root="/tmp/meta",
        )
        tt_uuid = uuid4()
        async with async_dao.engine.begin() as conn:
            await conn.execute(
                text(
                    'INSERT INTO tasktypes (uuid, exec_id, "order", name, is_parallel, children) '
                    + "VALUES (:uuid, :eid, 0, 'mf_step', 0, '[]')"
                ),
                {"uuid": tt_uuid.hex, "eid": exec_id},
            )
            await conn.execute(
                text(
                    "INSERT INTO tasks (tasktype_uuid, exec_id, start_timestamp) "
                    + "VALUES (:tu, :eid, :s)"
                ),
                {"tu": tt_uuid.hex, "eid": exec_id, "s": _NOW.isoformat()},
            )

        result = await async_dao.get_execution_tasks_list(exec_id)
        assert result is not None
        assert result[0].merge_func is None

    @pytest.mark.asyncio
    async def test_get_execution_tasks_list_merge_func_truthy(self, async_dao):
        from sqlalchemy import text

        exec_id = await async_dao._async_add_entity(
            Execution,
            name="mf_pipe",
            username="u",
            _start_timestamp=_NOW,
            metadata_root="/tmp/meta",
        )
        tt_uuid = uuid4()
        async with async_dao.engine.begin() as conn:
            await conn.execute(
                text(
                    'INSERT INTO tasktypes (uuid, exec_id, "order", name, is_parallel, children, merge_func) '
                    + "VALUES (:uuid, :eid, 0, 'mf_step', 0, '[]', '{\"name\": \"custom_merge\"}')"
                ),
                {"uuid": tt_uuid.hex, "eid": exec_id},
            )
            await conn.execute(
                text(
                    "INSERT INTO tasks (tasktype_uuid, exec_id, start_timestamp) VALUES (:tu, :eid, :s)"
                ),
                {"tu": tt_uuid.hex, "eid": exec_id, "s": _NOW.isoformat()},
            )

        result = await async_dao.get_execution_tasks_list(exec_id)
        assert result is not None
        assert result[0].merge_func == "custom_merge"


# ==============================================================================
# AsyncDAOExecutionTaskTypes
# ==============================================================================


class TestAsyncDAOExecutionTaskTypes:
    @pytest.mark.asyncio
    async def test_get_execution_tasktypes_list_builds_tasktype_objects(
        self, async_dao
    ):
        from sqlalchemy import text

        exec_id = await async_dao._async_add_entity(
            Execution,
            name="tasktypes_loop_pipe",
            username="u",
            _start_timestamp=_NOW,
            metadata_root="/tmp/meta",
        )
        async with async_dao.engine.begin() as conn:
            await conn.execute(
                text(
                    'INSERT INTO tasktypes (uuid, exec_id, "order", name, '
                    "is_parallel, children) VALUES "
                    "(:uuid, :eid, 0, 'loop_step', 0, '[]')"
                ),
                {"uuid": uuid4().hex, "eid": exec_id},
            )
        async_dao.get_execution_tasktypes_list.cache_clear()
        result = await async_dao.get_execution_tasktypes_list(exec_id)
        assert result is not None and len(result) == 1
        for tt in result:
            assert isinstance(tt, TaskType)

    @pytest.mark.asyncio
    async def test_get_execution_tasktypes_list_returns_none_when_empty(
        self, async_dao
    ):
        """execution with no TaskTypes => return None."""
        exec_id = await async_dao._async_add_entity(
            Execution,
            name="tt_empty_pipe",
            username="u",
            _start_timestamp=_NOW,
            metadata_root="/tmp/meta",
        )
        async_dao.get_execution_tasktypes_list.cache_clear()
        result = await async_dao.get_execution_tasktypes_list(exec_id)
        assert result is None


# ==============================================================================
# AsyncDAOExecutionTaskGroups
# ==============================================================================


class TestAsyncDAOExecutionTaskGroups:
    @pytest.mark.asyncio
    async def test_get_execution_taskgroups_list_builds_taskgroup_objects(
        self, async_dao
    ):
        from sqlalchemy import text

        exec_id = await async_dao._async_add_entity(
            Execution,
            name="taskgroups_loop_pipe",
            username="u",
            _start_timestamp=_NOW,
            metadata_root="/tmp/meta",
        )
        async with async_dao.engine.begin() as conn:
            await conn.execute(
                text(
                    'INSERT INTO taskgroups (uuid, exec_id, "order", name, '
                    "elements) VALUES "
                    "(:uuid, :eid, 0, 'loop_group', '[]')"
                ),
                {"uuid": uuid4().hex, "eid": exec_id},
            )
        async_dao.get_execution_taskgroups_list.cache_clear()
        result = await async_dao.get_execution_taskgroups_list(exec_id)
        assert result is not None and len(result) == 1
        for tg in result:
            assert isinstance(tg, TaskGroup)

    @pytest.mark.asyncio
    async def test_get_execution_taskgroups_list_returns_none_when_empty(
        self, async_dao
    ):
        """execution with no TaskGroups => return None."""
        exec_id = await async_dao._async_add_entity(
            Execution,
            name="tg_empty_pipe",
            username="u",
            _start_timestamp=_NOW,
            metadata_root="/tmp/meta",
        )
        async_dao.get_execution_taskgroups_list.cache_clear()
        result = await async_dao.get_execution_taskgroups_list(exec_id)
        assert result is None


# ==============================================================================
# AsyncDAOExecutionTasksWithName
# ==============================================================================


class TestAsyncDAOExecutionTasksWithName:
    @pytest.mark.asyncio
    async def test_get_execution_tasks_with_name_returns_list(self, seeded_async_dao):
        """tasks_list assignment and return."""
        dao, (exec_id, _) = seeded_async_dao
        result = await dao.get_execution_tasks_with_name(exec_id, "seeded_step")
        assert isinstance(result, list)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_get_execution_tasks_with_name_returns_empty_list(self, async_dao):
        """empty result still returns a list (not None)."""
        exec_id = await async_dao._async_add_entity(
            Execution,
            name="tasklist_pipe",
            username="u",
            _start_timestamp=_NOW,
            metadata_root="/tmp/meta",
        )
        result = await async_dao.get_execution_tasks_with_name(exec_id, "no_such_step")
        assert result == []


# ==============================================================================
# AsyncDAOExecutionInfo
# ==============================================================================


class TestAsyncDAOExecutionInfo:
    @pytest.mark.asyncio
    async def test_get_execution_info_returns_expected_dict_shape(self, async_dao):
        exec_id = await async_dao._async_add_entity(
            Execution,
            name="info_pipe",
            username="u",
            docstring="info docstring",
            _start_timestamp=_NOW,
            metadata_root="/tmp/meta",
        )
        async_dao.get_execution_info.cache_clear()
        info = await async_dao.get_execution_info(exec_id)
        assert info is not None
        assert set(info.keys()) == {"name", "username", "start_timestamp", "docstring"}
        assert isinstance(info["start_timestamp"], str)

    @pytest.mark.asyncio
    async def test_get_execution_info_not_found(self, async_dao, _null_session):
        """result.first()=None => return None."""
        async_dao.get_execution_info.cache_clear()
        with patch.object(async_dao, "_get_session", return_value=_null_session):
            result = await async_dao.get_execution_info(99999)
        assert result is None


# ==============================================================================
# AsyncDAOExecutionNumber
# ==============================================================================


class TestAsyncDAOExecutionNumber:
    @pytest.mark.asyncio
    async def test_get_execution_number_returns_dict(self, async_dao):
        exec_id = await async_dao._async_add_entity(
            Execution,
            name="exec_number_pipe",
            username="u",
            _start_timestamp=_NOW,
            metadata_root="/tmp/meta",
        )
        info = await async_dao.get_execution_number(exec_id)
        assert info is not None
        assert set(info.keys()) == {"name", "number", "count", "completed", "failed"}
        assert info["name"] == "exec_number_pipe"
        assert info["count"] == 1

    @pytest.mark.asyncio
    async def test_get_execution_number_not_found(self, async_dao, _null_session):
        """Line 807: row is None => return None."""
        with patch.object(async_dao, "_get_session", return_value=_null_session):
            result = await async_dao.get_execution_number(99999)
        assert result is None


# ==============================================================================
# AsyncDAOTaskgroupsHierarchy
# ==============================================================================


class TestAsyncDAOTaskgroupsHierarchy:
    @pytest.mark.asyncio
    async def test_get_taskgroups_hierarchy_with_ui_css(self, async_dao):
        from sqlalchemy import text

        tg_uuid = uuid4()
        async with async_dao.engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO executions (name, username, start_timestamp, metadata_root) "
                    + "VALUES ('tg_pipe', 'bob', :s, '/tmp/meta')"
                ),
                {"s": _NOW.isoformat()},
            )
            exec_id = (await conn.execute(text("SELECT last_insert_rowid()"))).scalar()
            await conn.execute(
                text(
                    'INSERT INTO taskgroups (uuid, exec_id, "order", name, elements, ui_css) '
                    + "VALUES (:uuid, :eid, 0, 'parent_grp', '[]', '{\"color\": \"red\"}')"
                ),
                {"uuid": tg_uuid.hex, "eid": exec_id},
            )

        async_dao.get_taskgroups_hierarchy.cache_clear()
        result = await async_dao.get_taskgroups_hierarchy(tg_uuid)
        assert result is not None
        assert result[0]["ui_css"] == {"color": "red"}

    @pytest.mark.asyncio
    async def test_get_taskgroups_hierarchy_builds_entries(
        self, taskgroup_seeded_async_dao
    ):
        async_dao, (_, parent_uuid, child_uuid) = taskgroup_seeded_async_dao
        async_dao.get_taskgroups_hierarchy.cache_clear()
        result = await async_dao.get_taskgroups_hierarchy(child_uuid)
        assert result is not None
        for entry in result:
            assert set(entry.keys()) == {"uuid", "name", "ui_css"}
            assert isinstance(entry["uuid"], str)

    @pytest.mark.asyncio
    async def test_get_taskgroups_hierarchy_unknown_uuid_warns_and_returns_none(
        self, async_dao, capture_log
    ):
        unknown = uuid4()
        async_dao.get_taskgroups_hierarchy.cache_clear()

        with capture_log(
            "retrain_pipelines.dag_engine.db.dao", level=logging.WARNING
        ) as captured:
            result = await async_dao.get_taskgroups_hierarchy(unknown)

        assert result is None
        assert any(str(unknown) in msg for msg in captured.getvalue().splitlines())

    @pytest.mark.asyncio
    async def test_get_taskgroups_hierarchy_not_found(self, async_dao, _null_session):
        """fetchall()=[] => logger.warning + return None."""
        async_dao.get_taskgroups_hierarchy.cache_clear()
        with patch.object(async_dao, "_get_session", return_value=_null_session):
            result = await async_dao.get_taskgroups_hierarchy(uuid4())
        assert result is None


# ==============================================================================
# AsyncDAOTaskTypeDocstring
# ==============================================================================


class TestAsyncDAOTaskTypeDocstring:
    @pytest.mark.asyncio
    async def test_get_tasktype_docstring_returns_value_when_set(self, async_dao):
        from sqlalchemy import text

        exec_id = await async_dao._async_add_entity(
            Execution,
            name="docstr_pipe",
            username="u",
            _start_timestamp=_NOW,
            metadata_root="/tmp/meta",
        )
        tt_uuid = uuid4()
        async with async_dao.engine.begin() as conn:
            await conn.execute(
                text(
                    'INSERT INTO tasktypes (uuid, exec_id, "order", name, '
                    "is_parallel, children, docstring) VALUES "
                    "(:uuid, :eid, 0, 'docstr_step', 0, '[]', 'hello docstring')"
                ),
                {"uuid": tt_uuid.hex, "eid": exec_id},
            )

        async_dao.get_tasktype_docstring.cache_clear()
        result = await async_dao.get_tasktype_docstring(str(tt_uuid))
        assert result == "hello docstring"

    @pytest.mark.asyncio
    async def test_get_tasktype_docstring_not_found(self, async_dao, _null_session):
        """result.first()=None => return None."""
        async_dao.get_tasktype_docstring.cache_clear()
        with patch.object(async_dao, "_get_session", return_value=_null_session):
            result = await async_dao.get_tasktype_docstring(str(uuid4()))
        assert result is None


# ==============================================================================
# AsyncDAOTaskTraces
# ==============================================================================


class TestAsyncDAOTaskTraces:
    @pytest.mark.asyncio
    async def test_get_task_traces_found(self, seeded_async_dao):
        async_dao, (exec_id_with_tasks, _) = seeded_async_dao
        tasks = await async_dao.get_execution_tasks_list(exec_id_with_tasks)
        assert tasks
        task_id = tasks[0].id

        await async_dao._async_add_entity(
            TaskTrace,
            task_id=task_id,
            timestamp=_NOW,
            microsec=0,
            microsec_idx=1,
            content="trace_msg",
            is_err=False,
        )

        result = await async_dao.get_task_traces(task_id)
        assert result is not None and len(result) >= 1
        assert result[0]["content"] == "trace_msg"

    @pytest.mark.asyncio
    async def test_get_task_traces_empty_for_existing_task_returns_none(
        self, seeded_async_dao
    ):
        async_dao, (exec_id_with_tasks, _) = seeded_async_dao
        tasks = await async_dao.get_execution_tasks_list(exec_id_with_tasks)
        assert tasks
        task_id = tasks[0].id
        result = await async_dao.get_task_traces(task_id)
        assert result is None


# ==============================================================================
# DAOTaskContextAttrs
# ==============================================================================


class TestDAOTaskContextAttrs:
    """add_task_context_attrs – method was never called."""

    def _seed(self, dao):
        """Return a task_id in a fresh isolated database."""
        tt_uuid = uuid4()
        with patch("requests.post"):
            exec_id = dao.add_execution(
                name="ctx_pipe",
                username="u",
                _start_timestamp=_NOW,
                metadata_root="/tmp/meta",
            )
            dao.add_tasktype(
                uuid=tt_uuid,
                exec_id=exec_id,
                order=0,
                name="ctx_step",
                is_parallel=False,
                children=[],
            )
        return dao.add_task(
            tasktype_uuid=tt_uuid, exec_id=exec_id, _start_timestamp=_NOW
        )

    def test_add_task_context_attrs_with_rows(self, isolated_dao):
        """non-empty rows triggers the batch insert."""
        task_id = self._seed(isolated_dao)
        rows = [
            {
                "task_id": task_id,
                "attr_name": "x",
                "sha": "deadbeef",
                "disk_ref": "some/path.pkl",
                "inline_val": None,
            }
        ]
        # Must not raise; inserts one TaskContextAttr row.
        isolated_dao.add_task_context_attrs(rows)

    def test_add_task_context_attrs_empty_rows_is_noop(self, isolated_dao):
        """(False branch): empty list must skip the batch insert."""
        isolated_dao.add_task_context_attrs([])


# ==============================================================================
# AsyncDAOTaskContextAttrs
# ==============================================================================


class TestAsyncDAOTaskContextAttrs:
    """Dedicated class for async-DAO lines that were unreachable due to
    seeded-fixture failures (conftest metadata_root issue) or missing tests."""

    @pytest.mark.asyncio
    async def test_get_task_context_attrs_returns_empty_list(self, async_dao):
        """method body – no rows => empty list."""
        result = await async_dao.get_task_context_attrs(task_id=99999)
        assert result == []

    @pytest.mark.asyncio
    async def test_get_task_context_attrs_returns_rows(self, seeded_async_dao):
        """method body – with a real task_id and inserted attr."""
        from sqlalchemy import text

        dao, (exec_id, _) = seeded_async_dao
        tasks = await dao.get_execution_tasks_list(exec_id)
        task_id = tasks[0].id

        async with dao.engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO task_context_attrs"
                    " (task_id, attr_name, sha, disk_ref, inline_val)"
                    " VALUES (:tid, 'out', 'abc123', NULL, NULL)"
                ),
                {"tid": task_id},
            )

        result = await dao.get_task_context_attrs(task_id=task_id)
        assert len(result) == 1
        assert isinstance(result[0], TaskContextAttr)
        assert result[0].attr_name == "out"


# ==============================================================================
# AsyncDAOTaskExt
# ==============================================================================


class TestAsyncDAOTaskExt:
    @pytest.mark.asyncio
    async def test_get_task_ext_found(self, seeded_async_dao):
        """found => return TaskExt with name populated."""
        dao, (exec_id, _) = seeded_async_dao
        tasks = await dao.get_execution_tasks_list(exec_id)
        task_id = tasks[0].id
        result = await dao.get_task_ext(task_id)
        assert result is not None
        assert isinstance(result, TaskExt)
        assert result.name == "seeded_step"

    @pytest.mark.asyncio
    async def test_get_task_ext_not_found(self, async_dao, _null_session):
        """result.first()=None => return None."""
        with patch.object(async_dao, "_get_session", return_value=_null_session):
            result = await async_dao.get_task_ext(task_id=99999)
        assert result is None
