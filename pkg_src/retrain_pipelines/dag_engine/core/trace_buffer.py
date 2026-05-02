
import os
import queue
import atexit
import logging
import threading

from typing import Optional
from collections import defaultdict
from datetime import datetime, timezone

from ..db.dao import DAO


logger = logging.getLogger(__name__)


class TraceBuffer:
    """Thread-safe buffer for task traces with background writer."""

    def __init__(
        self, flush_interval: float = 0.5, batch_size: int = 2
    ):
        """
        Params:
            flush_interval (float):
                Seconds between flushes
            batch_size (int):
                Max traces to batch in one write
        """
        self._queue = queue.Queue()
        self._dao_factory = \
            lambda: DAO(os.environ["RP_METADATASTORE_URL"])
        self._flush_interval = flush_interval
        self._batch_size = batch_size
        self._stop_event = threading.Event()
        self._writer_thread = None
        self._started = False
        self._lock = threading.Lock()
        self._writing_lock = threading.Lock()

    def start(self):
        """Start (or restart) the background writer thread."""
        with self._lock:
            if self._writer_thread and self._writer_thread.is_alive():
                return
            self._stop_event.clear()
            self._started = True
            self._writer_thread = threading.Thread(
                target=self._writer_loop,
                daemon=True,
                name="TraceWriter"
            )
            self._writer_thread.start()
            atexit.register(self.stop)

    def add_trace(self, task_id: int, content: str,
                  timestamp: datetime, microsec: int,
                  is_err: bool = False):
        """Add a trace entry to the buffer (non-blocking).
        
        Each entry represents one complete line (with \n preserved).
        """
        # logger.error(f"TraceBuffer.add_trace({task_id}, {content})")
        trace_dict = {
            "task_id": task_id,
            "content": content,
            "timestamp": timestamp,
            "microsec": microsec,
            "is_err": is_err
        }
        self._queue.put(trace_dict)

    def _writer_loop(self):
        """Background thread that writes traces to DB in batches."""
        while not self._stop_event.is_set():
            batch = []
            deadline = self._flush_interval

            try:
                while len(batch) < self._batch_size:
                    try:
                        trace = self._queue.get(timeout=deadline)
                        batch.append(trace)
                        deadline = 0.01
                    except queue.Empty:
                        break

                if batch:
                    with self._writing_lock:
                        self._flush_batch(batch)

            except Exception as e:
                logger.error(f"Error in trace writer loop: {e}",
                             exc_info=True)

    def _flush_batch(self, batch):
        """Write a batch of traces to the database."""
        dao = None
        try:
            counters = defaultdict(int)
            for trace in batch:
                key = (trace["timestamp"], trace["microsec"])
                trace["microsec_idx"] = counters[key]
                counters[key] += 1

            dao = self._dao_factory()
            dao.batch_add_task_traces(traces_batch=batch)
        except Exception as e:
            logger.error(f"Failed to write trace batch: {e}")
        finally:
            if dao:
                dao.dispose()

    def flush(self):
        """Flush all pending traces immediately."""
        batch = []
        try:
            while True:
                batch.append(self._queue.get_nowait())
        except queue.Empty:
            pass

        with self._writing_lock:
            if batch:
                self._flush_batch(batch)

    def stop(self, timeout=5.0):
        """Stop the background writer and flush remaining traces."""
        with self._lock:
            if not self._started:
                return

            self._stop_event.set()

            if self._writer_thread and self._writer_thread.is_alive():
                self._writer_thread.join(timeout=timeout)

            self.flush()
            self._started = False


_trace_buffer: Optional[TraceBuffer] = None
_buffer_lock = threading.Lock()


def get_trace_buffer() -> TraceBuffer:
    """Get or create the global trace buffer."""
    global _trace_buffer

    with _buffer_lock:
        if _trace_buffer is None:
            _trace_buffer = TraceBuffer()
        _trace_buffer.start()
        return _trace_buffer

