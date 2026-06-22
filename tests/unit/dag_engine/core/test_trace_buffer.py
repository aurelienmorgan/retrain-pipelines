"""
Unit tests for retrain_pipelines.dag_engine.core.trace_buffer.
"""

import queue
import threading
import time
from datetime import datetime
from unittest.mock import MagicMock, patch


from retrain_pipelines.dag_engine.core.trace_buffer import (
    TraceBuffer,
    get_trace_buffer,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_trace(task_id=1, content="hello", microsec=0, is_err=False):
    return dict(
        task_id=task_id,
        content=content,
        timestamp=datetime(2024, 1, 1, 12, 0, 0),
        microsec=microsec,
        is_err=is_err,
    )


def _make_buffer(flush_interval=0.05, batch_size=10) -> TraceBuffer:
    """Return a TraceBuffer whose DAO factory is fully mocked."""
    buf = TraceBuffer(flush_interval=flush_interval, batch_size=batch_size)
    mock_dao = MagicMock()
    buf._dao_factory = lambda: mock_dao
    return buf, mock_dao


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
    def test_start_spawns_daemon_thread(self):
        buf, _ = _make_buffer()
        try:
            buf.start()
            assert buf._writer_thread is not None
            assert buf._writer_thread.is_alive()
            assert buf._writer_thread.daemon
            assert buf._writer_thread.name == "TraceWriter"
            assert buf._started
        finally:
            buf.stop()

    def test_start_is_idempotent(self):
        buf, _ = _make_buffer()
        try:
            buf.start()
            first_thread = buf._writer_thread
            buf.start()  # second call – must not replace thread
            assert buf._writer_thread is first_thread
        finally:
            buf.stop()

    def test_restart_after_stop(self):
        buf, _ = _make_buffer()
        buf.start()
        buf.stop()
        buf.start()
        try:
            assert buf._started
            assert buf._writer_thread.is_alive()
        finally:
            buf.stop()

    def test_start_registers_atexit(self):
        buf, _ = _make_buffer()
        with patch("atexit.register") as mock_reg:
            buf.start()
            mock_reg.assert_called_once_with(buf.stop)
        buf.stop()


# ---------------------------------------------------------------------------
# add_trace
# ---------------------------------------------------------------------------


class TestAddTrace:
    def test_enqueues_dict(self):
        buf, _ = _make_buffer()
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

    def test_enqueues_is_err_default_false(self):
        buf, _ = _make_buffer()
        ts = datetime(2024, 6, 1)
        buf.add_trace(task_id=1, content="x", timestamp=ts, microsec=0)
        item = buf._queue.get_nowait()
        assert item["is_err"] is False

    def test_non_blocking(self):
        """add_trace must return immediately even with a very large queue."""
        buf, _ = _make_buffer(batch_size=1_000_000)
        ts = datetime.now()
        start = time.monotonic()
        for i in range(500):
            buf.add_trace(1, str(i), ts, i)
        elapsed = time.monotonic() - start
        assert elapsed < 1.0


# ---------------------------------------------------------------------------
# _flush_batch
# ---------------------------------------------------------------------------


class TestFlushBatch:
    def test_microsec_idx_counter_same_key(self):
        buf, mock_dao = _make_buffer()
        ts = datetime(2024, 1, 1)
        batch = [_make_trace(microsec=0) for _ in range(3)]
        for t in batch:
            t["timestamp"] = ts
        buf._flush_batch(batch)
        assert batch[0]["microsec_idx"] == 0
        assert batch[1]["microsec_idx"] == 1
        assert batch[2]["microsec_idx"] == 2

    def test_microsec_idx_counter_different_keys(self):
        buf, mock_dao = _make_buffer()
        ts = datetime(2024, 1, 1)
        b0 = _make_trace(microsec=0)
        b0["timestamp"] = ts
        b1 = _make_trace(microsec=1)
        b1["timestamp"] = ts
        b2 = _make_trace(microsec=0)
        b2["timestamp"] = ts
        buf._flush_batch([b0, b1, b2])
        assert b0["microsec_idx"] == 0
        assert b1["microsec_idx"] == 0  # different key
        assert b2["microsec_idx"] == 1  # same key as b0

    def test_calls_dao_batch_add(self):
        buf, mock_dao = _make_buffer()
        batch = [_make_trace()]
        buf._flush_batch(batch)
        mock_dao.batch_add_task_traces.assert_called_once_with(traces_batch=batch)

    def test_calls_dao_dispose(self):
        buf, mock_dao = _make_buffer()
        buf._flush_batch([_make_trace()])
        mock_dao.dispose.assert_called_once()

    def test_exception_in_dao_is_swallowed(self):
        buf, mock_dao = _make_buffer()
        mock_dao.batch_add_task_traces.side_effect = RuntimeError("db down")
        # must not propagate
        buf._flush_batch([_make_trace()])

    def test_dispose_called_even_on_exception(self):
        buf, mock_dao = _make_buffer()
        mock_dao.batch_add_task_traces.side_effect = RuntimeError("db down")
        buf._flush_batch([_make_trace()])
        mock_dao.dispose.assert_called_once()

    def test_dao_factory_exception_swallowed(self):
        buf, _ = _make_buffer()
        buf._dao_factory = lambda: (_ for _ in ()).throw(ConnectionError("no db"))
        # must not propagate
        buf._flush_batch([_make_trace()])


# ---------------------------------------------------------------------------
# flush (public synchronous)
# ---------------------------------------------------------------------------


class TestFlush:
    def test_flush_drains_all_pending(self):
        buf, mock_dao = _make_buffer()
        ts = datetime.now()
        for i in range(5):
            buf.add_trace(1, f"line{i}", ts, i)
        buf.flush()
        assert buf._queue.empty()
        calls = mock_dao.batch_add_task_traces.call_args_list
        # all 5 traces delivered in one or more batches
        total = sum(len(c.kwargs["traces_batch"]) for c in calls)
        assert total == 5

    def test_flush_empty_queue_is_noop(self):
        buf, mock_dao = _make_buffer()
        buf.flush()  # must not raise
        mock_dao.batch_add_task_traces.assert_not_called()

    def test_flush_acquires_writing_lock(self):
        """flush must serialise with the writer via _writing_lock."""
        buf, _ = _make_buffer()
        acquired_inside = []

        orig_flush_batch = buf._flush_batch

        def spy_flush(batch):
            acquired_inside.append(buf._writing_lock.locked())
            orig_flush_batch(batch)

        buf._flush_batch = spy_flush
        buf.add_trace(1, "x", datetime.now(), 0)
        buf.flush()
        assert acquired_inside == [True]


# ---------------------------------------------------------------------------
# stop
# ---------------------------------------------------------------------------


class TestStop:
    def test_stop_joins_thread(self):
        buf, _ = _make_buffer()
        buf.start()
        thread = buf._writer_thread
        buf.stop()
        assert not thread.is_alive()
        assert not buf._started

    def test_stop_flushes_remaining(self):
        buf, mock_dao = _make_buffer()
        buf.start()
        ts = datetime.now()
        buf.add_trace(1, "last", ts, 0)
        buf.stop()
        total = sum(
            len(c.kwargs["traces_batch"])
            for c in mock_dao.batch_add_task_traces.call_args_list
        )
        assert total >= 1

    def test_stop_is_idempotent(self):
        buf, _ = _make_buffer()
        buf.start()
        buf.stop()
        buf.stop()  # must not raise

    def test_stop_before_start_is_noop(self):
        buf, _ = _make_buffer()
        buf.stop()  # must not raise


# ---------------------------------------------------------------------------
# _writer_loop integration (live thread, short intervals)
# ---------------------------------------------------------------------------


class TestWriterLoop:
    def test_writer_loop_flushes_batch(self):
        buf, mock_dao = _make_buffer(flush_interval=0.05, batch_size=10)
        buf.start()
        ts = datetime.now()
        for i in range(3):
            buf.add_trace(1, f"t{i}", ts, i)
        # allow background thread to drain
        deadline = time.monotonic() + 2.0
        while buf._queue.qsize() > 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        buf.stop()
        total = sum(
            len(c.kwargs["traces_batch"])
            for c in mock_dao.batch_add_task_traces.call_args_list
        )
        assert total == 3

    def test_writer_loop_exception_does_not_kill_thread(self):
        buf, mock_dao = _make_buffer(flush_interval=0.05, batch_size=2)
        # first call raises; subsequent calls succeed
        mock_dao.batch_add_task_traces.side_effect = [RuntimeError("boom"), None, None]
        buf.start()
        ts = datetime.now()
        for i in range(4):
            buf.add_trace(1, f"t{i}", ts, i)
        time.sleep(0.3)
        assert buf._writer_thread.is_alive()
        buf.stop()

    def test_writer_respects_batch_size(self):
        """Writer must not exceed batch_size per _flush_batch call."""
        buf, mock_dao = _make_buffer(flush_interval=0.5, batch_size=2)
        received_sizes = []
        orig = buf._flush_batch

        def spy(batch):
            received_sizes.append(len(batch))
            orig(batch)

        buf._flush_batch = spy
        buf.start()
        ts = datetime.now()
        for i in range(6):
            buf.add_trace(1, f"t{i}", ts, i)
        time.sleep(0.3)
        buf.stop()
        assert all(s <= 2 for s in received_sizes)


# ---------------------------------------------------------------------------
# get_trace_buffer module-level singleton
# ---------------------------------------------------------------------------


class TestGetTraceBuffer:
    def test_returns_trace_buffer_instance(self):
        import retrain_pipelines.dag_engine.core.trace_buffer as _tb_mod

        with patch.object(_tb_mod, "_trace_buffer", None):
            with patch.object(_tb_mod, "_buffer_lock", threading.Lock()):
                instance = get_trace_buffer()
                try:
                    assert isinstance(instance, TraceBuffer)
                finally:
                    instance.stop()

    def test_singleton_same_object(self):
        import retrain_pipelines.dag_engine.core.trace_buffer as _tb_mod

        with patch.object(_tb_mod, "_trace_buffer", None):
            with patch.object(_tb_mod, "_buffer_lock", threading.Lock()):
                a = get_trace_buffer()
                b = get_trace_buffer()
                try:
                    assert a is b
                finally:
                    a.stop()

    def test_buffer_is_started(self):
        import retrain_pipelines.dag_engine.core.trace_buffer as _tb_mod

        with patch.object(_tb_mod, "_trace_buffer", None):
            with patch.object(_tb_mod, "_buffer_lock", threading.Lock()):
                buf = get_trace_buffer()
                try:
                    assert buf._started
                    assert buf._writer_thread is not None
                    assert buf._writer_thread.is_alive()
                finally:
                    buf.stop()
