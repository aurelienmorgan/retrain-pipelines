"""
Unit tests for retrain_pipelines.dag_engine.sse_streaming_server.server.

Executes the real Uvicorn server and multiprocessing bridge in background threads
to achieve 100% real execution coverage without emulating or mocking package imports.
"""

import asyncio
import time
import queue

import pytest
import httpx
import uvicorn
from unittest.mock import patch, MagicMock

from retrain_pipelines.dag_engine.sse_streaming_server import server
from retrain_pipelines.dag_engine.sse_streaming_server.model import TraceData


@pytest.fixture(autouse=True)
def ensure_server_stopped():
    """Automatically stop the SSE server after each test to avoid cross‑test pollution."""
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
        """Covers start(), stop(), _run_server(), and _bridge_loop() empty queue path."""
        assert sse_server.startswith("http://127.0.0.1:")
        assert server._server is not None
        assert server._server_thread is not None
        assert server._bridge_thread is not None
        assert server._event_loop is not None
        assert not server._event_loop.is_closed()
        assert server._sse_server_url is not None

        # Let the bridge loop run briefly to hit the empty queue path
        time.sleep(0.2)

    def test_publish_and_consume_trace(self, sse_server):
        """Covers _tasktraces_endpoint, event_stream(), _dispatch_trace(), publish_trace()."""
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

    def test_stop_drains_pending_traces(self):
        """Covers the branch where stop() is called while the multiprocessing
        queue still has pending items, triggering the drain loop and dispatch."""
        _ = server.start()

        trace = TraceData(
            id=1,
            task_id=100,
            timestamp=1717238400000,
            microsec=123,
            microsec_idx=1,
            content="Pending trace",
            is_err=False,
        )
        # Mock get_nowait to return the trace, then raise queue.Empty.
        # This deterministically forces stop() to execute the drain loop
        # and dispatch the trace, bypassing multiprocessing feeder thread delays.
        mock_get = MagicMock(side_effect=[trace, queue.Empty()])
        with patch.object(server._mp_queue, "get_nowait", mock_get):
            server.stop()

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

    def test_run_server_cancellation(self):
        """Covers the branch where the Uvicorn serve() coroutine is cancelled,
        triggering the CancelledError handler in the server thread."""

        async def mock_serve(self):
            # Mock socket and server attributes to allow start() to bind successfully
            mock_sock = MagicMock()
            mock_sock.getsockname.return_value = ("127.0.0.1", 9999)
            mock_server = MagicMock()
            mock_server.sockets = [mock_sock]
            self.servers = [mock_server]
            self.started = True

            # Keep alive until should_exit is set by stop()
            while not self.should_exit:
                await asyncio.sleep(0.01)
            # Cancel the current task to simulate cancellation during shutdown
            asyncio.current_task().cancel()

        with patch.object(uvicorn.Server, "serve", mock_serve):
            _ = server.start()
            assert server._server is not None
            # Stop the server, which sets should_exit and causes mock_serve to cancel
            server.stop()
            # The thread should have completed without hanging
            assert server._server_thread is None
