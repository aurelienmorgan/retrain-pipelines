"""
Unit tests for retrain_pipelines.dag_engine.sse_streaming_server.server.

Executes the real Uvicorn server and multiprocessing bridge in background threads
to achieve 100% real execution coverage.
"""

import asyncio
import time

import pytest
import httpx

from retrain_pipelines.dag_engine.sse_streaming_server import server
from retrain_pipelines.dag_engine.sse_streaming_server.model import TraceData


@pytest.fixture(autouse=True)
def ensure_server_stopped():
    """Automatically stop the SSE server after each test to avoid cross-test pollution."""
    yield
    if server._server is not None:
        server.stop()


@pytest.fixture
def sse_server():
    """Start the SSE server, yield its URL, and guarantee stop afterwards."""
    url = server.start()
    yield url
    server.stop()


class TestSSEServer:
    def test_start_stop_lifecycle(self, sse_server):
        """Covers start(), stop(), was_started(), _run_server(), and _bridge_loop() empty queue path."""
        assert server.was_started() is True
        assert sse_server.startswith("http://127.0.0.1:")
        assert server._server is not None
        assert server._server_thread is not None
        assert server._bridge_thread is not None
        assert server._event_loop is not None
        assert not server._event_loop.is_closed()
        assert server._sse_server_url is not None

        # Let the bridge loop run briefly to hit the empty queue path
        time.sleep(0.2)

        server.stop()
        assert server.was_started() is False

    def test_publish_and_consume_trace(self, sse_server):
        """Covers _tasktraces_endpoint, event_stream(), _dispatch_trace(), publish_trace(),
        and the GeneratorExit branch of event_stream()'s except clause (triggered when
        the HTTP client closes the stream and Starlette closes the generator)."""
        url = sse_server

        trace = TraceData(
            id=1,
            task_id=100,
            timestamp=1717238400000,
            microsec=123,
            microsec_idx=1,
            content="Test trace message",
            is_err=False,
        )

        # Use a generous timeout for the whole operation
        timeout = httpx.Timeout(connect=5.0, read=30.0, write=5.0, pool=5.0)
        with httpx.stream("GET", f"{url}/tasktraces", timeout=timeout) as resp:
            # Wait for subscriber registration
            t_start = time.time()
            while len(server._sse_subscribers) == 0 and time.time() - t_start < 5:
                time.sleep(0.05)
            assert len(server._sse_subscribers) == 1, "Subscriber was not registered"

            # Publish trace to the multiprocessing queue
            server.publish_trace(trace)

            # Give the bridge thread a moment to dispatch the trace
            time.sleep(0.2)

            # Read lines with a timeout – if nothing received after 15s, fail.
            received = False
            line_timeout = 15.0
            line_start = time.time()
            for line in resp.iter_lines():
                if time.time() - line_start > line_timeout:
                    break
                if line.startswith("data: "):
                    assert trace.model_dump_json() in line
                    received = True
                    break

            assert received, "Did not receive trace data from SSE stream (timeout)"

        # After closing the stream, the event_stream() generator should clean up
        t_start = time.time()
        while len(server._sse_subscribers) > 0 and time.time() - t_start < 5:
            time.sleep(0.05)
        assert len(server._sse_subscribers) == 0, "Subscriber was not cleaned up"

    def test_publish_trace_when_server_down(self):
        """Covers the branch where publish_trace is called when the server
        is not running."""
        # Ensure server is stopped
        if server._server is not None:
            server.stop()

        trace = TraceData(
            id=1,
            task_id=100,
            timestamp=1717238400000,
            microsec=123,
            microsec_idx=1,
            content="Test trace message",
            is_err=False,
        )
        # Should return early without putting on the queue
        server.publish_trace(trace)

    def test_stop_drains_pending_traces(self, sse_server):
        """Covers the branch in stop()'s drain loop where get_nowait() succeeds
        and the trace is dispatched to the event loop via call_soon_threadsafe.

        Achieves this without mocking by signalling the bridge thread to exit
        (via _stop_bridge.set()) and waiting for it to terminate, so a subsequent
        publish_trace() call leaves the item on the multiprocessing queue for
        stop()'s drain loop to retrieve. The bridge's blocking get(timeout=0.1)
        would otherwise race with stop()'s get_nowait() and drain the queue first.
        """
        trace = TraceData(
            id=1,
            task_id=100,
            timestamp=1717238400000,
            microsec=123,
            microsec_idx=1,
            content="Pending trace",
            is_err=False,
        )

        # Signal the bridge thread to exit and wait for it to terminate,
        # so the multiprocessing queue is no longer being drained
        # concurrently by the bridge's blocking get(timeout=0.1).
        server._stop_bridge.set()
        server._bridge_thread.join(timeout=2.0)
        assert not server._bridge_thread.is_alive()

        # With the bridge gone, publish_trace() leaves the item on the
        # multiprocessing queue. stop()'s drain loop then retrieves it
        # via get_nowait() and dispatches it to the event loop.
        server.publish_trace(trace)

        # multiprocessing.Queue.put_nowait() is asynchronous: the item is
        # placed on an internal buffer and flushed to the underlying pipe
        # by a background feeder thread. Wait for that flush so stop()'s
        # get_nowait() can actually retrieve the item ; otherwise the
        # feeder thread may not have flushed yet, get_nowait() raises
        # "Empty", the drain loop breaks early, and the dispatch branch
        # is never reached.
        t_flush_start = time.time()
        while not server._mp_queue._reader.poll() and time.time() - t_flush_start < 2.0:
            time.sleep(0.02)

        server.stop()

        assert server._server is None
        assert server._bridge_thread is None

    def test_stop_with_idle_subscriber(self, sse_server):
        """Covers stop() with an active subscriber whose asyncio.Queue is empty
        (generator suspended at await q.get()).

        Exercises the branches in stop() that require at least one active
        subscriber: the _check_drained if-branch (queue empty -> set event),
        the _subscribers_drained.wait(timeout=5.0) call, the sentinel-dispatch
        loop over _sse_subscribers, and the sentinel break path in
        event_stream() (trace_data is None -> break).

        Uses a real HTTP subscriber so that the event_stream() generator is
        started by Starlette and suspends at await q.get() ; no traces are
        published, so the queue stays empty until stop() dispatches the sentinel.
        """
        url = sse_server

        timeout = httpx.Timeout(connect=5.0, read=30.0, write=5.0, pool=5.0)
        with httpx.stream("GET", f"{url}/tasktraces", timeout=timeout) as resp:
            # Wait for subscriber registration. At this point the
            # event_stream() generator has been started by Starlette and
            # is suspended at await q.get() because the queue is empty.
            t_start = time.time()
            while len(server._sse_subscribers) == 0 and time.time() - t_start < 5:
                time.sleep(0.05)
            assert len(server._sse_subscribers) == 1

            # Call stop() with the subscriber active and its queue empty.
            # _check_drained takes the if-branch (queue empty -> set event).
            # The sentinel is dispatched; the generator pulls None and breaks.
            server.stop()

            # Drain any remaining bytes so httpx can close cleanly after
            # the server has shut down.
            try:
                for _ in resp.iter_lines():
                    break
            except Exception:
                pass

        assert server._server is None
        assert server._sse_subscribers == []

    def test_stop_with_nonempty_subscriber_queue(self, sse_server):
        """Covers the _check_drained elif-branch (rescheduling via call_later)
        when stop() is called with a subscriber whose asyncio.Queue is non-empty.

        Registers a standalone subscriber queue (no consuming generator) with a
        pending trace, simulating a back-pressured subscriber whose generator is
        suspended at a yield and cannot drain its queue. _check_drained reschedules
        via call_later until the 5-second wait times out, after which stop()
        proceeds to the sentinel-dispatch loop.

        This is the only way to deterministically keep a subscriber's queue
        non-empty during stop(): a real HTTP subscriber's generator would
        immediately pull any available trace from the queue, leaving it empty
        unless TCP-level back-pressure suspends the generator at a yield, which
        is timing-dependent and unreliable.
        """
        loop = server._event_loop
        assert loop is not None

        trace = TraceData(
            id=1,
            task_id=100,
            timestamp=1717238400000,
            microsec=123,
            microsec_idx=1,
            content="orphan trace",
            is_err=False,
        )

        # Create the subscriber queue inside the event loop and pre-load a
        # trace, simulating a back-pressured subscriber whose generator is
        # suspended at a yield and cannot drain its queue.
        async def _setup_orphan_subscriber():
            orphan_q = asyncio.Queue()
            server._sse_subscribers.append(orphan_q)
            orphan_q.put_nowait(trace)
            return orphan_q

        future = asyncio.run_coroutine_threadsafe(_setup_orphan_subscriber(), loop)
        orphan_q = future.result(timeout=5.0)
        assert orphan_q.qsize() == 1
        assert len(server._sse_subscribers) == 1

        # Call stop() with the subscriber's queue non-empty. _check_drained
        # takes the elif-branch (reschedule via call_later) until the
        # 5-second wait times out.
        server.stop()

        assert server._server is None
        assert server._sse_subscribers == []

    def test_run_server_cancellation(self, sse_server):
        """Covers the branch in _run_server where loop.run_until_complete
        raises CancelledError, exercising the except handler.

        Without mocking: cancels the serve() task in the event loop via
        asyncio.all_tasks, causing CancelledError to propagate out of serve()
        (modern Uvicorn does not catch BaseException) and out of
        run_until_complete, where _run_server's except handler suppresses it.
        Subsequent stop() performs graceful cleanup of remaining state.
        """
        loop = server._event_loop
        assert loop is not None

        # Schedule cancellation of all tasks in the event loop. The serve()
        # task is the main one; CancelledError propagates out of serve()
        # and out of run_until_complete, where _run_server's except handler
        # suppresses it.
        def _cancel_serve_task():
            for task in asyncio.all_tasks(loop):
                task.cancel()

        loop.call_soon_threadsafe(_cancel_serve_task)

        # Wait for the server thread to exit via the CancelledError handler.
        if server._server_thread is not None:
            server._server_thread.join(timeout=5.0)

        # stop() performs graceful cleanup of remaining state (bridge thread,
        # event loop reference) even though serve() exited via cancellation
        # rather than should_exit.
        server.stop()

        assert server._server is None
        assert server._server_thread is None
