"""
Unit & Integration tests for retrain_pipelines.dag_engine.core.trace_buffer.
"""

import queue
import threading
import time
from datetime import datetime, timezone
from unittest.mock import patch
from uuid import uuid4

from retrain_pipelines.dag_engine.core import trace_buffer as _tb_mod
from retrain_pipelines.dag_engine.core.trace_buffer import (
    TraceBuffer,
    get_trace_buffer,
)
from retrain_pipelines.dag_engine.db.model import (
    TaskTrace,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_trace(task_id=1, content="hello", microsec=0, is_err=False):
    return dict(
        task_id=task_id,
        content=content,
        timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        microsec=microsec,
        is_err=is_err,
    )


def _seed_task(dao, task_id_out=None):
    """Seed Execution + TaskType + Task rows so TaskTrace.task_id FK is satisfied.

    Returns the Task.id (real DB auto-increment integer).
    """
    exec_id = dao.add_execution(
        name="test_pipe",
        username="tester",
        _start_timestamp=datetime.now(timezone.utc),
        metadata_root="/tmp/meta",
        artifacts_store_root="/tmp/artifacts",
    )
    tt_uuid = uuid4()
    dao.add_tasktype(
        exec_id=exec_id,
        order=0,
        uuid=tt_uuid,
        name="seeded_tasktype",
        is_parallel=False,
        children=[],
    )
    task_id = dao.add_task(
        exec_id=exec_id,
        tasktype_uuid=tt_uuid,
        rank=None,
        _start_timestamp=datetime.now(timezone.utc),
    )
    return task_id


def _make_buffer(isolated_db, isolated_dao, flush_interval=0.05, batch_size=10):
    """Return a TraceBuffer whose DAO factory hits the real test DB.

    The wrapped funcs in trace_buffer.py call `DAO(Config.get_metadatastore_url())`
    via the `_dao_factory` lambda ; isolated_db sets RP_METADATASTORE_URL so the
    real DAO hits the clean per-test DB. The `isolated_dao` is yielded for
    post-call state assertions on the same DB.
    """
    buf = TraceBuffer(flush_interval=flush_interval, batch_size=batch_size)
    return buf, isolated_dao


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


class TestInit:
    def test_defaults(self):
        buf = TraceBuffer()
        assert buf._flush_interval == 0.5
        assert buf._batch_size == 2
        assert isinstance(buf._queue, queue.Queue)
        assert buf._writer_thread is None
        assert not buf._started
        assert isinstance(buf._stop_event, threading.Event)
        assert isinstance(buf._lock, type(threading.Lock()))
        assert isinstance(buf._writing_lock, type(threading.Lock()))

    def test_custom_params(self):
        buf = TraceBuffer(flush_interval=1.5, batch_size=50)
        assert buf._flush_interval == 1.5
        assert buf._batch_size == 50


# ---------------------------------------------------------------------------
# start
# ---------------------------------------------------------------------------


class TestStart:
    def test_start_spawns_daemon_thread(self, isolated_db, isolated_dao):
        buf, _ = _make_buffer(isolated_db, isolated_dao)
        try:
            buf.start()
            assert buf._writer_thread is not None
            assert buf._writer_thread.is_alive()
            assert buf._writer_thread.daemon
            assert buf._writer_thread.name == "TraceWriter"
            assert buf._started
        finally:
            buf.stop()

    def test_start_is_idempotent(self, isolated_db, isolated_dao):
        buf, _ = _make_buffer(isolated_db, isolated_dao)
        try:
            buf.start()
            first_thread = buf._writer_thread
            buf.start()  # second call – must not replace thread
            assert buf._writer_thread is first_thread
        finally:
            buf.stop()

    def test_restart_after_stop(self, isolated_db, isolated_dao):
        buf, _ = _make_buffer(isolated_db, isolated_dao)
        buf.start()
        buf.stop()
        buf.start()
        try:
            assert buf._started
            assert buf._writer_thread.is_alive()
        finally:
            buf.stop()

    def test_start_registers_atexit(self, isolated_db, isolated_dao):
        buf, _ = _make_buffer(isolated_db, isolated_dao)
        with patch("atexit.register") as mock_reg:
            buf.start()
            mock_reg.assert_called_once_with(buf.stop)
        buf.stop()


# ---------------------------------------------------------------------------
# add_trace
# ---------------------------------------------------------------------------


class TestAddTrace:
    def test_enqueues_dict(self, isolated_db, isolated_dao):
        buf, _ = _make_buffer(isolated_db, isolated_dao)
        ts = datetime(2024, 6, 1, 10, 0, 0)
        buf.add_trace(
            task_id=7, content="msg\n", timestamp=ts, microsec=42, is_err=True
        )
        item = buf._queue.get_nowait()
        assert item == {
            "task_id": 7,
            "content": "msg\n",
            "timestamp": ts,
            "microsec": 42,
            "is_err": True,
        }

    def test_enqueues_is_err_default_false(self, isolated_db, isolated_dao):
        buf, _ = _make_buffer(isolated_db, isolated_dao)
        ts = datetime(2024, 6, 1)
        buf.add_trace(task_id=1, content="x", timestamp=ts, microsec=0)
        item = buf._queue.get_nowait()
        assert item["is_err"] is False

    def test_non_blocking(self, isolated_db, isolated_dao):
        """add_trace must return immediately even with a very large queue."""
        buf, _ = _make_buffer(isolated_db, isolated_dao, batch_size=1_000_000)
        ts = datetime.now(timezone.utc)
        start = time.monotonic()
        for i in range(500):
            buf.add_trace(1, str(i), ts, i)
        elapsed = time.monotonic() - start
        assert elapsed < 1.0


# ---------------------------------------------------------------------------
# _flush_batch
# ---------------------------------------------------------------------------


class TestFlushBatch:
    def test_microsec_idx_counter_same_key(self, isolated_db, isolated_dao):
        buf, dao = _make_buffer(isolated_db, isolated_dao)
        task_id = _seed_task(dao)
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        batch = [_make_trace(task_id=task_id, microsec=0) for _ in range(3)]
        for t in batch:
            t["timestamp"] = ts
        buf._flush_batch(batch)
        assert batch[0]["microsec_idx"] == 0
        assert batch[1]["microsec_idx"] == 1
        assert batch[2]["microsec_idx"] == 2
        # Real TaskTrace rows written.
        rows = dao._get_entities(TaskTrace, task_id=task_id)
        assert len(rows) == 3

    def test_microsec_idx_counter_different_keys(self, isolated_db, isolated_dao):
        buf, dao = _make_buffer(isolated_db, isolated_dao)
        task_id = _seed_task(dao)
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        b0 = _make_trace(task_id=task_id, microsec=0)
        b0["timestamp"] = ts
        b1 = _make_trace(task_id=task_id, microsec=1)
        b1["timestamp"] = ts
        b2 = _make_trace(task_id=task_id, microsec=0)
        b2["timestamp"] = ts
        buf._flush_batch([b0, b1, b2])
        assert b0["microsec_idx"] == 0
        assert b1["microsec_idx"] == 0  # different key
        assert b2["microsec_idx"] == 1  # same key as b0
        rows = dao._get_entities(TaskTrace, task_id=task_id)
        assert len(rows) == 3

    def test_calls_dao_batch_add(self, isolated_db, isolated_dao):
        """Real DB: batch_add_task_traces writes rows to the real TaskTrace table."""
        buf, dao = _make_buffer(isolated_db, isolated_dao)
        task_id = _seed_task(dao)
        batch = [_make_trace(task_id=task_id)]
        buf._flush_batch(batch)
        rows = dao._get_entities(TaskTrace, task_id=task_id)
        assert len(rows) == 1
        assert rows[0].content == "hello"

    def test_exception_in_dao_is_swallowed(self, isolated_db, isolated_dao):
        """Forcing DAO to raise via a non-existent task_id (FK violation) is
        caught by _flush_batch's except Exception and logged, not propagated.

        Uses a task_id that has no matching Task row, so the real
        batch_add_task_traces raises IntegrityError on the FK constraint.
        """
        buf, dao = _make_buffer(isolated_db, isolated_dao)
        # task_id=999999 has no matching Task row => FK violation => swallowed.
        buf._flush_batch([_make_trace(task_id=999999)])
        # No rows written for the bogus task_id.
        assert dao._get_entities(TaskTrace, task_id=999999) == []

    def test_dao_factory_exception_swallowed(self, isolated_db, isolated_dao):
        """Forcing _dao_factory to raise is swallowed by _flush_batch."""
        buf, _ = _make_buffer(isolated_db, isolated_dao)
        buf._dao_factory = lambda: (_ for _ in ()).throw(ConnectionError("no db"))
        # must not propagate
        buf._flush_batch([_make_trace()])


# ---------------------------------------------------------------------------
# flush (public synchronous)
# ---------------------------------------------------------------------------


class TestFlush:
    def test_flush_drains_all_pending(self, isolated_db, isolated_dao):
        buf, dao = _make_buffer(isolated_db, isolated_dao)
        task_id = _seed_task(dao)
        ts = datetime.now(timezone.utc)
        for i in range(5):
            buf.add_trace(task_id, f"line{i}", ts, i)
        buf.flush()
        assert buf._queue.empty()
        # Real TaskTrace rows written.
        rows = dao._get_entities(TaskTrace, task_id=task_id)
        assert len(rows) == 5

    def test_flush_empty_queue_is_noop(self, isolated_db, isolated_dao):
        buf, dao = _make_buffer(isolated_db, isolated_dao)
        buf.flush()  # must not raise
        # No rows written.
        assert dao._get_entities(TaskTrace) == []

    def test_flush_acquires_writing_lock(self, isolated_db, isolated_dao):
        """flush must serialise with the writer via _writing_lock."""
        buf, dao = _make_buffer(isolated_db, isolated_dao)
        task_id = _seed_task(dao)
        acquired_inside = []

        orig_flush_batch = buf._flush_batch

        def spy_flush(batch):
            acquired_inside.append(buf._writing_lock.locked())
            orig_flush_batch(batch)

        buf._flush_batch = spy_flush
        buf.add_trace(task_id, "x", datetime.now(timezone.utc), 0)
        buf.flush()
        assert acquired_inside == [True]


# ---------------------------------------------------------------------------
# stop
# ---------------------------------------------------------------------------


class TestStop:
    def test_stop_joins_thread(self, isolated_db, isolated_dao):
        buf, _ = _make_buffer(isolated_db, isolated_dao)
        buf.start()
        thread = buf._writer_thread
        buf.stop()
        assert not thread.is_alive()
        assert not buf._started

    def test_stop_flushes_remaining(self, isolated_db, isolated_dao):
        buf, dao = _make_buffer(isolated_db, isolated_dao)
        buf.start()
        task_id = _seed_task(dao)
        ts = datetime.now(timezone.utc)
        buf.add_trace(task_id, "last", ts, 0)
        buf.stop()
        rows = dao._get_entities(TaskTrace, task_id=task_id)
        assert len(rows) >= 1

    def test_stop_is_idempotent(self, isolated_db, isolated_dao):
        buf, _ = _make_buffer(isolated_db, isolated_dao)
        buf.start()
        buf.stop()
        buf.stop()  # must not raise

    def test_stop_before_start_is_noop(self, isolated_db, isolated_dao):
        buf, _ = _make_buffer(isolated_db, isolated_dao)
        buf.stop()  # must not raise


# ---------------------------------------------------------------------------
# _writer_loop integration (live thread, short intervals)
# ---------------------------------------------------------------------------


class TestWriterLoop:
    def test_writer_loop_flushes_batch(self, isolated_db, isolated_dao):
        buf, dao = _make_buffer(
            isolated_db, isolated_dao, flush_interval=0.05, batch_size=10
        )
        buf.start()
        task_id = _seed_task(dao)
        ts = datetime.now(timezone.utc)
        for i in range(3):
            buf.add_trace(task_id, f"t{i}", ts, i)
        # allow background thread to drain
        deadline = time.monotonic() + 2.0
        while buf._queue.qsize() > 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        buf.stop()
        rows = dao._get_entities(TaskTrace, task_id=task_id)
        assert len(rows) == 3

    def test_writer_loop_exception_does_not_kill_thread(
        self, isolated_db, isolated_dao
    ):
        """First batch uses a bogus task_id (FK violation) => exception swallowed
        by _writer_loop's except Exception ; thread stays alive for subsequent batches.

        The FK violation triggers 5 retries with exponential backoff inside
        _sync_batch_add_entities (~1.5s total) ; the writer thread blocks on
        _flush_batch until all retries exhaust, then the except handler runs
        and the loop continues to the next batch. We sleep 2.0s to let the
        bad batch's retries exhaust AND the good batch write.
        """
        buf, dao = _make_buffer(
            isolated_db, isolated_dao, flush_interval=0.05, batch_size=2
        )
        buf.start()
        task_id = _seed_task(dao)
        ts = datetime.now(timezone.utc)
        for i in range(2):
            buf.add_trace(999999, f"bad{i}", ts, i)
        for i in range(2):
            buf.add_trace(task_id, f"good{i}", ts, i)
        time.sleep(2.0)
        assert buf._writer_thread.is_alive()
        buf.stop()
        # Good traces were written despite the earlier exception.
        rows = dao._get_entities(TaskTrace, task_id=task_id)
        assert len(rows) == 2

    def test_writer_respects_batch_size(self, isolated_db, isolated_dao):
        """Writer must not exceed batch_size per _flush_batch call."""
        buf, dao = _make_buffer(
            isolated_db, isolated_dao, flush_interval=0.5, batch_size=2
        )
        task_id = _seed_task(dao)
        received_sizes = []
        orig = buf._flush_batch

        def spy(batch):
            received_sizes.append(len(batch))
            orig(batch)

        buf._flush_batch = spy
        buf.start()
        ts = datetime.now(timezone.utc)
        for i in range(6):
            buf.add_trace(task_id, f"t{i}", ts, i)
        time.sleep(0.3)
        buf.stop()
        assert all(s <= 2 for s in received_sizes)


# ---------------------------------------------------------------------------
# get_trace_buffer module-level singleton
# ---------------------------------------------------------------------------


class TestGetTraceBuffer:
    def test_returns_trace_buffer_instance(self, isolated_db, isolated_dao):

        with patch.object(_tb_mod, "_trace_buffer", None):
            with patch.object(_tb_mod, "_buffer_lock", threading.Lock()):
                instance = get_trace_buffer()
                try:
                    assert isinstance(instance, TraceBuffer)
                finally:
                    instance.stop()

    def test_singleton_same_object(self, isolated_db, isolated_dao):

        with patch.object(_tb_mod, "_trace_buffer", None):
            with patch.object(_tb_mod, "_buffer_lock", threading.Lock()):
                a = get_trace_buffer()
                b = get_trace_buffer()
                try:
                    assert a is b
                finally:
                    a.stop()

    def test_buffer_is_started(self, isolated_db, isolated_dao):

        with patch.object(_tb_mod, "_trace_buffer", None):
            with patch.object(_tb_mod, "_buffer_lock", threading.Lock()):
                buf = get_trace_buffer()
                try:
                    assert buf._started
                    assert buf._writer_thread is not None
                    assert buf._writer_thread.is_alive()
                finally:
                    buf.stop()
