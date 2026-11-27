"""SSE-streaming server for task traces.

Runs in the DAG-engine process.
Worker processes (forked children) call publish_trace() ;
the bridge thread relays each TraceData to every active
SSE subscriber queue inside the Uvicorn event loop.

Topology
--------
worker process  ==put==>  multiprocessing.Queue  ==bridge thread==>
    asyncio.Queue (per subscriber)  ==yield==>  GET /tasktraces (WebConsole)

Why two threads ?
-----------------
Two background threads are required because the server sits at the boundary
between two incompatible concurrency models:

* **SseServer thread** - runs Uvicorn inside a dedicated ``asyncio`` event
  loop. That loop must never block: it is simultaneously streaming chunks
  to every connected WebConsole subscriber.

* **SseBridge thread** - ``multiprocessing.Queue.get()`` is a *blocking*
  call and cannot be awaited inside the event loop without stalling all SSE
  connections.
  The bridge thread blocks on the queue, then hands each ``TraceData``
  to the event loop via ``loop.call_soon_threadsafe()``,
  the only thread-safe entry-point into asyncio.

Merging the two roles into one thread is impossible : blocking on the
multiprocessing queue inside the event loop would freeze SSE delivery, while
running the async server outside an event loop is not supported by Uvicorn.
"""

import asyncio
import logging
import multiprocessing
import queue as _queue
import threading
import time
from collections.abc import AsyncGenerator

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import StreamingResponse
from starlette.routing import Route

from .model import TraceData

logger = logging.getLogger(__name__)

# Created at import time so the reference is inherited by forked child processes.
_mp_queue: multiprocessing.Queue = multiprocessing.Queue()

_server: uvicorn.Server | None = None
_server_thread: threading.Thread | None = None
_bridge_thread: threading.Thread | None = None
_stop_bridge = threading.Event()
_event_loop: asyncio.AbstractEventLoop | None = None
_sse_server_url: str | None = None

# Accessed exclusively from the Uvicorn event loop thread ; no lock needed.
_sse_subscribers: list[asyncio.Queue[TraceData | None]] = []


# ---------------------------------------------------------------------------
# SSE endpoint
# ---------------------------------------------------------------------------


async def _tasktraces_endpoint(request: Request) -> StreamingResponse:
    """SSE endpoint: streams TraceData JSON events to the WebConsole."""
    q: asyncio.Queue = asyncio.Queue()
    _sse_subscribers.append(q)

    async def event_stream() -> AsyncGenerator[str, None]:
        try:
            while True:
                trace_data: TraceData | None = await q.get()
                if trace_data is None:
                    # sentinel received from ``stop()``
                    break
                yield f"data: {trace_data.model_dump_json()}\n\n"
        except (asyncio.CancelledError, GeneratorExit):
            pass
        finally:
            if q in _sse_subscribers:
                _sse_subscribers.remove(q)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Bridge: multiprocessing.Queue => asyncio subscriber queues
# ---------------------------------------------------------------------------


def _dispatch_trace(trace_data: TraceData) -> None:
    """Push trace to all SSE subscribers (runs inside the event loop thread)."""
    for q in list(_sse_subscribers):
        q.put_nowait(trace_data)


def _bridge_loop() -> None:
    """Background thread: drains the multiprocessing.Queue and dispatches."""
    while not _stop_bridge.is_set():
        try:
            trace_data: TraceData = _mp_queue.get(timeout=0.1)
        except Exception:
            # Empty / timeout - loop and re-check _stop_bridge
            continue

        if _event_loop is not None and not _event_loop.is_closed():
            _event_loop.call_soon_threadsafe(_dispatch_trace, trace_data)


# ---------------------------------------------------------------------------
# Public methods
# ---------------------------------------------------------------------------


def start() -> str:
    """Start the SSE streaming server.

    Binds to an OS-assigned port (port 0), starts Uvicorn in a background
    thread, and starts the bridge thread.

    Returns
    -------
    str
        Reachable base URL of the SSE server (e.g. ``'http://127.0.0.1:PORT'``).
    """
    global _server, _server_thread, _bridge_thread, _event_loop, _sse_server_url

    app = Starlette(routes=[Route("/tasktraces", _tasktraces_endpoint)])
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning", ws="wsproto")
    _server = uvicorn.Server(config)

    loop_ready = threading.Event()

    def _run_server() -> None:
        global _event_loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        _event_loop = loop
        loop_ready.set()
        try:
            loop.run_until_complete(_server.serve())
        except asyncio.CancelledError:
            # Expected during shutdown when should_exit is set
            pass

    _server_thread = threading.Thread(target=_run_server, daemon=True, name="SseServer")
    _server_thread.start()

    # Wait until the event loop is assigned, then until the server has bound.
    loop_ready.wait()
    while not _server.started:
        time.sleep(0.01)

    port = _server.servers[0].sockets[0].getsockname()[1]
    _sse_server_url = f"http://127.0.0.1:{port}"

    _stop_bridge.clear()
    _bridge_thread = threading.Thread(target=_bridge_loop, daemon=True, name="SseBridge")
    _bridge_thread.start()

    logger.info(f"SSE streaming server started at {_sse_server_url}")
    return _sse_server_url


def was_started():
    return _server is not None


def stop() -> None:
    """Shut down the SSE streaming server and clean up all resources."""
    global _server, _server_thread, _bridge_thread, _sse_server_url, _event_loop

    # 1. Drain any pending items from the multiprocessing queue.
    # Note: multiprocessing.Queue.empty() is unreliable across platforms,
    # so we use a non-blocking get loop to process remaining traces
    # before shutting down the bridge thread.
    timeout = 5.0
    t_start = time.time()
    while time.time() - t_start < timeout:
        try:
            trace_data = _mp_queue.get_nowait()
        except _queue.Empty:
            # Queue is fully drained
            break

        if _event_loop is not None and not _event_loop.is_closed():
            _event_loop.call_soon_threadsafe(_dispatch_trace, trace_data)

    # 2. Wait for all scheduled _dispatch_trace calls to execute in the
    #    event loop AND for event_stream() generators to consume every
    #    trace from their subscriber asyncio.Queues. Without this,
    #    should_exit cancels generators that still have undelivered
    #    traces in their queues, and the WebConsole consumer sees
    #    "server closed stream" before the last traces arrive.
    _subscribers_drained = threading.Event()

    def _check_drained():
        if not _sse_subscribers or all(q.empty() for q in _sse_subscribers):
            _subscribers_drained.set()
        elif _event_loop is not None and not _event_loop.is_closed():
            _event_loop.call_later(0.05, _check_drained)

    if _event_loop is not None and not _event_loop.is_closed() and _sse_subscribers:
        _event_loop.call_soon_threadsafe(_check_drained)
        _subscribers_drained.wait(timeout=5.0)

    # 3. Send a sentinel to each subscriber so event_stream() exits
    #    cleanly, allowing Starlette to flush the StreamingResponse
    #    normally rather than cancelling the generator mid-stream.
    if _event_loop is not None and not _event_loop.is_closed():
        for q in list(_sse_subscribers):
            _event_loop.call_soon_threadsafe(q.put_nowait, None)

    # 4. Stop the bridge thread
    _stop_bridge.set()

    # 5. Shut down the Uvicorn server gracefully
    if _server is not None:
        _server.should_exit = True
        if _server_thread is not None:
            _server_thread.join(timeout=2.0)
        _server = None
        _server_thread = None

    if _bridge_thread is not None:
        _bridge_thread.join(timeout=2.0)
        _bridge_thread = None

    # 6. Clear subscriber queues
    _sse_subscribers.clear()

    # 7. Clean up event loop reference
    _event_loop = None

    _sse_server_url = None
    logger.info("SSE streaming server stopped")


def publish_trace(trace_data: TraceData) -> None:
    """Put a TraceData onto the shared multiprocessing queue.

    Safe to call from forked worker processes: the queue's underlying pipe
    is inherited across fork and the feeder thread is re-initialised on
    first use in the child.

    Parameters
    ----------
    trace_data : TraceData
        The trace to publish to the subscribed WebConsole.
    """
    if _server is None:
        return
    _mp_queue.put_nowait(trace_data)
