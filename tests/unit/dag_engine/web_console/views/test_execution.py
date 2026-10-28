"""
Unit and Integration tests for retrain_pipelines.dag_engine.web_console.views.execution.

- Routes are exercised end-to-end through a real WebConsole uvicorn
  instance, started per test via the production ``webconsole_start``
  entrypoint and hit over real TCP with ``httpx.AsyncClient``.
  ``_consume_sse_traces`` is exercised against the real
  ``sse_streaming_server`` (success path) and against deliberately
  broken raw socket servers (error branches).
- Database state is provided by the shared parametrized fixtures
  exposed by dag_engine/conftest.py (isolated_db, seeded_async_dao,
  artifacts_store_root).
"""

import asyncio
import logging
import socket
import threading
import weakref
from datetime import datetime, timezone
from uuid import uuid4
from sqlalchemy.ext.asyncio import AsyncEngine


import httpx
import pytest
import pytest_asyncio

from retrain_pipelines.dag_engine.db.dao import AsyncDAO
from retrain_pipelines.dag_engine.db.model import (
    Execution,
    TaskGroup,
    TaskTrace,
    TaskType,
)
from retrain_pipelines.dag_engine.sse_streaming_server import server as sse_server_mod
from retrain_pipelines.dag_engine.sse_streaming_server.model import TraceData
from retrain_pipelines.dag_engine.web_console.main import (
    webconsole_shutdown,
    webconsole_start,
)
from retrain_pipelines.dag_engine.web_console.utils import ClientInfo
from retrain_pipelines.dag_engine.web_console.utils.execution import (
    events as execution_events,
)
from retrain_pipelines.dag_engine.web_console.views import execution as exec_module
from retrain_pipelines.dag_engine.web_console.views.execution import (
    get_execution_dag_elements_lists,
    get_execution_elements_lists,
)
from retrain_pipelines.utils.file_utils import write_text_file


# =============================================================================
# Module-local helpers
# =============================================================================


def _serve_one_shot_raw_response(raw_bytes: bytes) -> str:
    """Start a one-shot TCP server that accepts one HTTP request, then
    writes the given raw bytes verbatim to the client and closes.

    Returns the base URL (``http://127.0.0.1:<port>``) for the client to
    connect to. The server accepts exactly one connection, sends the
    payload, and exits.

    Reused by every test that needs a response the real
    ``sse_streaming_server`` cannot produce: malformed HTTP status
    lines (to trigger ``httpx.RemoteProtocolError``) and SSE frames with
    invalid JSON (to trigger the ``json.loads`` exception handler).
    """
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]

    def serve_once():
        conn = None
        try:
            conn, _ = listener.accept()
            conn.settimeout(5.0)
            # Drain the request line + headers (until the blank line).
            buf = b""
            while b"\r\n\r\n" not in buf:
                chunk = conn.recv(4096)
                if not chunk:
                    return
                buf += chunk
            conn.sendall(raw_bytes)
        except OSError:
            pass
        finally:
            if conn is not None:
                try:
                    conn.close()
                except OSError:
                    pass
            listener.close()

    threading.Thread(target=serve_once, daemon=True).start()
    return f"http://127.0.0.1:{port}"


async def _wait_for_sse_subscriber(timeout: float = 5.0) -> None:
    """Wait until the SSE streaming server has at least one connected
    subscriber (i.e., ``_consume_sse_traces`` has finished its HTTP
    handshake with ``/tasktraces`` and the server has registered the
    per-client queue in ``_sse_subscribers``).

    Without this synchronization, traces published before the consumer
    connects are silently dropped (the server's ``_dispatch_trace`` has
    no queue to put them on).
    """
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while not sse_server_mod._sse_subscribers:
        if loop.time() > deadline:
            raise AssertionError(
                f"SSE consumer did not connect to the SSE server within {timeout}s"
            )
        await asyncio.sleep(0.05)


def _pick_free_port() -> int:
    """Return an OS-assigned free TCP port on the loopback interface.

    The socket is closed before the port is returned, so the caller
    (``webconsole_start``) re-binds via ``acquire_server_lock`` with
    ``SO_REUSEADDR``. A brief race window exists in which another
    process could claim the port, but that is extremely unlikely on a
    test worker and is surfaced immediately as an ``OSError`` from
    ``acquire_server_lock``.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# =============================================================================
# Module-local fixtures
# =============================================================================


_tracked_engines: weakref.WeakSet = weakref.WeakSet()


@pytest_asyncio.fixture(autouse=True)
async def _dispose_lingering_async_engines():
    """Dispose any ``AsyncEngine`` instances left undissposed after each
    test.

    The issue
    ---------
    Every route handler in ``views.execution`` constructs an ``AsyncDAO``
    as a local variable to query the metadata store, then returns the
    response. The ``AsyncDAO.__init__`` call to ``create_async_engine``
    creates a connection pool that holds open aiosqlite connections. Each
    aiosqlite connection owns a ``_connection_worker_thread`` bound to
    the event loop of the test that created it. When the route handler
    returns, the ``AsyncDAO`` local goes out of scope, but the engine's
    pool is never disposed (the sync ``DAOBase.dispose`` swallows the
    coroutine returned by ``AsyncEngine.dispose`` in a bare ``except``).

    The sync ``DAO`` uses ``NullPool`` for sqlite, which prevents this
    problem entirely. The async ``AsyncDAO`` on the other hand doesn't.
    This asymmetry is in the ``dao`` module, which must not be altered
    for this test module alone.

    Without disposal, the lingering worker threads keep the test's event
    loop alive after pytest finishes. The loop's ``atexit`` handler
    blocks, and the process never exits.

    The solution
    ------------
    Engines are tracked via a module-level ``WeakSet``. The fixture
    temporarily wraps ``AsyncDAO.__init__`` so that each newly created
    engine is registered in the set. After each test, every tracked
    engine is disposed and the set is cleared.

    This is O(1) per engine (no full ``gc.get_objects()`` scan) and
    avoids the ``FutureWarning`` spam that ``gc.collect()`` triggers via
    third-party ``__del__`` finalizers (notably torch's).
    """
    _tracked_engines.clear()
    _orig_init = AsyncDAO.__init__

    def _tracking_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        if hasattr(self, "engine") and isinstance(self.engine, AsyncEngine):
            _tracked_engines.add(self.engine)

    AsyncDAO.__init__ = _tracking_init
    try:
        yield
    finally:
        AsyncDAO.__init__ = _orig_init
        for engine in list(_tracked_engines):
            try:
                await engine.dispose()
            except Exception:
                pass
        _tracked_engines.clear()


@pytest_asyncio.fixture
async def webconsole_server(isolated_db):
    """Real WebConsole uvicorn instance on an OS-assigned port.

    Uses the production ``webconsole_start`` / ``webconsole_shutdown``
    entrypoints so routing, exception handlers, static mounts, and
    uvicorn middleware run exactly as in production. Each test gets a
    fresh server, sharing the per-test database (transitively via
    ``isolated_db`` which monkeypatches ``RP_METADATASTORE_URL``).

    Yields an ``httpx.AsyncClient`` bound to the live server URL. The
    server is shut down in teardown so the next test can claim a new
    port (``webconsole_start`` forbids two instances in one process).
    """
    port = _pick_free_port()
    webconsole_start(port=port)
    base_url = f"http://127.0.0.1:{port}"

    async with httpx.AsyncClient(base_url=base_url, timeout=10.0) as client:
        # Poll until uvicorn is actually accepting connections, so the
        # first test request never hits a ConnectionRefusedError.
        loop = asyncio.get_event_loop()
        deadline = loop.time() + 10.0
        while True:
            try:
                probe = await client.get("/execution?id=0")
                if probe.status_code < 500:
                    break
            except (httpx.ConnectError, httpx.ReadError):
                pass
            if loop.time() > deadline:
                webconsole_shutdown()
                raise AssertionError(
                    f"WebConsole did not become ready on port {port} within 10s"
                )
            await asyncio.sleep(0.05)
        yield client
    webconsole_shutdown()


@pytest.fixture
def sse_server():
    """Real ``sse_streaming_server`` started for the duration of one
    test.

    Yields the reachable base URL (e.g. ``http://127.0.0.1:PORT``). The
    server is stopped in teardown to guarantee no global state leaks
    between tests.
    """
    url = sse_server_mod.start()
    yield url
    # Capture references before stop() clears them (stop() sets both
    # to None after joining).
    loop = sse_server_mod._event_loop
    thread = sse_server_mod._server_thread
    sse_server_mod.stop()
    # stop() only joins _server_thread with timeout=1.0. Uvicorn's
    # shutdown() waits for SSE connections to close, but
    # event_stream() blocks on q.get() forever, so shutdown() never
    # completes and the thread is still alive after the join returns.
    # The thread is daemon=True, but the asyncio event loop running
    # inside it has pending tasks (the event_stream() generators)
    # that prevent clean loop shutdown. Force-stop the loop so
    # run_until_complete() unblocks and the thread exits.
    if (
        thread is not None
        and thread.is_alive()
        and loop is not None
        and not loop.is_closed()
    ):
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=2.0)


@pytest.fixture
def task_trace_subscriber():
    """Register a real ``asyncio.Queue`` against
    ``execution_events.task_trace_subscribers`` so any trace fanned out
    by ``_consume_sse_traces`` lands in this queue.

    Yields the queue. Teardown removes the subscriber entry to keep the
    global subscriber list clean across tests.
    """
    queue: asyncio.Queue = asyncio.Queue()
    client_info = ClientInfo(ip="127.0.0.1", port=0, url="/test")
    execution_events.task_trace_subscribers.append((queue, client_info))
    yield queue
    try:
        execution_events.task_trace_subscribers.remove((queue, client_info))
    except ValueError:
        pass


@pytest.fixture(autouse=True)
def _reset_events_global_state():
    """Reset ``execution_events`` module-level globals (subscriber lists,
    server-loop reference) after each test so no cross-test pollution
    leaks through ``multiplexed_event_generator``'s globals.
    """
    yield
    execution_events.reset_for_restart()


@pytest.fixture(scope="session", autouse=True)
def _cleanup_sse_server_at_exit():
    """Session-scoped cleanup: close the SSE server's multiprocessing.Queue
    feeder thread after all tests complete.

    ``multiprocessing.Queue`` spawns a non-daemon feeder thread on the
    first ``put_nowait()``. That thread only exits when ``close()`` +
    ``join_thread()`` are called on the queue. ``sse_server_mod.stop()``
    never calls those, so the feeder thread survives the test suite and
    blocks process exit (it is non-daemon).

    Also force-stops the SSE server's event loop if the server thread is
    still alive (Uvicorn's ``shutdown()`` waits for SSE connections that
    never close because ``event_stream()`` blocks on ``q.get()`` forever).
    """
    yield
    # Force-stop the server's event loop so the SseServer thread exits.
    if (
        sse_server_mod._server_thread is not None
        and sse_server_mod._server_thread.is_alive()
        and sse_server_mod._event_loop is not None
        and not sse_server_mod._event_loop.is_closed()
    ):
        sse_server_mod._event_loop.call_soon_threadsafe(sse_server_mod._event_loop.stop)
        sse_server_mod._server_thread.join(timeout=2.0)
    # Close the multiprocessing.Queue so its non-daemon feeder thread exits.
    sse_server_mod._mp_queue.close()
    sse_server_mod._mp_queue.join_thread()


# =============================================================================
# Tests for get_execution_dag_elements_lists
# =============================================================================


class TestGetExecutionDagElementsLists:
    """Cover all branches of the helper that fetches TaskTypes + TaskGroups
    for DAG rendering, serializing them to JSON-safe dicts."""

    @pytest.mark.asyncio
    async def test_no_tasktypes_returns_none(self, isolated_db):
        """Covers the early-return branch where the execution has no
        TaskTypes (e.g. the execution_id does not exist), which makes the
        falsy-tasktypes guard fire and return ``(None, None)`` before any
        serialization runs.
        """
        result = await get_execution_dag_elements_lists(999999)
        assert result == (None, None)

    @pytest.mark.asyncio
    async def test_tasktypes_without_taskgroups(self, seeded_async_dao):
        """Covers the success branch where ``taskgroups_list`` is None/empty,
        exercising the falsy side of the taskgroups ternary and the
        ``taskgroup_uuid is None`` to empty-string coercion.
        """
        _dao, (exec_id, _tt_uuid) = seeded_async_dao
        result = await get_execution_dag_elements_lists(exec_id)
        assert result[0] is not None
        assert result[1] == []
        tasktype_dict = result[0][0]
        assert isinstance(tasktype_dict["uuid"], str)
        assert tasktype_dict["taskgroup_uuid"] == ""

    @pytest.mark.asyncio
    async def test_tasktypes_with_taskgroups(self, seeded_async_dao):
        """Covers the truthy side of the taskgroups ternary by adding a
        TaskGroup inline on the seeded execution, exercising the taskgroup
        uuid str-cast loop.
        """
        dao, (exec_id, _tt_uuid) = seeded_async_dao
        async with dao._get_session() as session:
            session.add(
                TaskGroup(
                    uuid=uuid4(),
                    exec_id=exec_id,
                    order=0,
                    name="inline_grp",
                    elements=[],
                )
            )
            await session.commit()
        result = await get_execution_dag_elements_lists(exec_id)
        assert result[1] != []
        assert isinstance(result[1][0]["uuid"], str)

    @pytest.mark.asyncio
    async def test_tasktype_with_taskgroup_uuid_set(self, seeded_async_dao):
        """Covers the truthy branch of the per-tasktype taskgroup_uuid
        coercion (``str(uuid)`` when not None) by adding a TaskType that
        references an existing TaskGroup on the seeded execution.
        """
        dao, (exec_id, _tt_uuid) = seeded_async_dao
        tg_uuid = uuid4()
        async with dao._get_session() as session:
            session.add(
                TaskGroup(
                    uuid=tg_uuid,
                    exec_id=exec_id,
                    order=0,
                    name="grp_for_tt",
                    elements=[str(tg_uuid)],
                )
            )
            session.add(
                TaskType(
                    uuid=uuid4(),
                    exec_id=exec_id,
                    order=1,
                    name="tt_with_tg",
                    is_parallel=False,
                    children=[],
                    taskgroup_uuid=tg_uuid,
                )
            )
            await session.commit()
        result = await get_execution_dag_elements_lists(exec_id)
        tt_with_tg = next(tt for tt in result[0] if tt["name"] == "tt_with_tg")
        assert tt_with_tg["taskgroup_uuid"] == str(tg_uuid)


# =============================================================================
# Tests for get_execution_elements_lists
# =============================================================================


class TestGetExecutionElementsLists:
    """Cover the helper that fetches Tasks + TaskGroups for the Gantt chart
    rendering, returning the raw model objects (no serialization)."""

    @pytest.mark.asyncio
    async def test_no_tasks_returns_none(self, isolated_db):
        """Covers the branch where the execution has no Tasks (DAO returns
        None) - the helper transparently returns ``(None, None)`` since it
        has no transformation logic.
        """
        tasks, taskgroups = await get_execution_elements_lists(999999)
        assert tasks is None
        assert taskgroups is None

    @pytest.mark.asyncio
    async def test_with_tasks(self, seeded_async_dao):
        """Covers the success branch where the DAO returns a non-empty
        Tasks list for the seeded execution.
        """
        _dao, (exec_id, _tt_uuid) = seeded_async_dao
        tasks, taskgroups = await get_execution_elements_lists(exec_id)
        assert tasks is not None
        assert len(tasks) >= 1


# =============================================================================
# Tests for _consume_sse_traces
# =============================================================================


class TestConsumeSseTraces:
    """Cover the SSE consumer background task end-to-end.

    No mocking of httpx or ``sse_streaming_server``: the real server is
    started, real traces are published through it, and real subscribers
    receive them. Two branches (``httpx.RemoteProtocolError`` handler and
    ``json.loads`` exception handler) require malformed responses that the
    real server cannot produce, so they reuse the
    ``_serve_one_shot_raw_response`` helper.
    """

    @pytest.mark.asyncio
    async def test_real_server_relays_trace_to_subscriber(
        self, isolated_db, sse_server, task_trace_subscriber
    ):
        """Covers the branch where a valid ``data: <json>`` line is
        received from the real SSE streaming server, parsed via
        ``json.loads``, and fanned out to all registered
        ``task_trace_subscribers`` via ``q.put_nowait``. Also covers
        the ``continue`` branch for non-data lines (the empty SSE
        separator lines between events).
        """
        consumer = asyncio.create_task(exec_module._consume_sse_traces(sse_server))
        await _wait_for_sse_subscriber()
        trace = TraceData(
            id=1,
            task_id=1,
            timestamp=1600000000000,
            microsec=0,
            microsec_idx=1,
            content="hello",
            is_err=False,
        )
        sse_server_mod.publish_trace(trace)
        received = await asyncio.wait_for(task_trace_subscriber.get(), timeout=5.0)
        assert received["content"] == "hello"
        assert received["task_id"] == 1
        consumer.cancel()
        try:
            await consumer
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_invalid_json_skipped_valid_relayed(
        self, isolated_db, task_trace_subscriber
    ):
        """Covers the ``json.loads`` exception handler (``except Exception:
        continue``) inside the main SSE loop.

        Sends a valid HTTP/1.1 chunked ``text/event-stream`` response
        with two SSE frames: one with invalid JSON (triggers the
        except+continue branch) and one with valid JSON (triggers the
        ``q.put_nowait`` fan-out). Asserts only the valid trace is
        delivered to the subscriber.
        """
        invalid_frame = b"data: {invalid json}\n\n"
        valid_frame = b'data: {"k":"v"}\n\n'
        body = (
            f"{len(invalid_frame):x}\r\n".encode()
            + invalid_frame
            + b"\r\n"
            + f"{len(valid_frame):x}\r\n".encode()
            + valid_frame
            + b"\r\n"
            + b"0\r\n\r\n"
        )
        raw_bytes = (
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: text/event-stream\r\n"
            b"Transfer-Encoding: chunked\r\n"
            b"\r\n" + body
        )
        url = _serve_one_shot_raw_response(raw_bytes)
        await exec_module._consume_sse_traces(url)
        received = await asyncio.wait_for(task_trace_subscriber.get(), timeout=5.0)
        assert received == {"k": "v"}
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(task_trace_subscriber.get(), timeout=0.5)

    @pytest.mark.asyncio
    async def test_remote_protocol_error_logged_at_debug(
        self, isolated_db, capture_log
    ):
        """Covers the ``except httpx.RemoteProtocolError`` branch by
        feeding an invalid HTTP status line, which makes httpx (via
        httpcore) raise ``RemoteProtocolError`` when parsing the
        response. The debug-level log line is then emitted.
        """
        url = _serve_one_shot_raw_response(b"INVALID STATUS LINE\r\n\r\n")
        with capture_log(exec_module.logger.name, level=logging.DEBUG) as captured:
            await exec_module._consume_sse_traces(url)
        assert "server closed stream" in captured.getvalue()

    @pytest.mark.asyncio
    async def test_generic_exception_logged_at_warning(self, isolated_db, capture_log):
        """Covers the ``except Exception`` branch by pointing the
        consumer at a non-listening port, which raises
        ``httpx.ConnectError`` (a non-``RemoteProtocolError``
        ``Exception`` subclass) that the generic handler logs at
        warning level.
        """
        with capture_log(exec_module.logger.name, level=logging.WARNING) as captured:
            await exec_module._consume_sse_traces("http://127.0.0.1:1")
        assert "disconnected unexpectedly" in captured.getvalue()


# =============================================================================
# Tests for register() routes (real WebConsole uvicorn instance)
# =============================================================================


class TestRegisterRoutes:
    """Exercise each route registered by ``register()`` through a real
    WebConsole uvicorn instance, hit over real TCP with
    ``httpx.AsyncClient``. Routing, request parsing, exception handlers,
    and response serialization all run exactly as in production.

    Each test depends on the parametrized ``isolated_db`` (sqlite +
    postgres) for a clean database backend per test, transitively
    through ``webconsole_server``.
    """

    # ----- /register_sse_subscriber -----
    @pytest.mark.asyncio
    async def test_register_sse_subscriber_missing_url(self, webconsole_server):
        """Covers the branch where the request body lacks an
        ``sse_server_url`` field, returning a 400 JSONResponse.
        """
        response = await webconsole_server.post("/register_sse_subscriber", json={})
        assert response.status_code == 400
        assert response.json() == {"error": "missing sse_server_url"}

    @pytest.mark.asyncio
    async def test_register_sse_subscriber_success(self, webconsole_server):
        """Covers the branch where a valid ``sse_server_url`` is provided,
        spawning the SSE consumer background task and returning 200.

        Uses a non-listening URL so the spawned consumer exits
        immediately with ``httpx.ConnectError``. The actual SSE relay
        is tested end-to-end by
        ``test_real_server_relays_trace_to_subscriber``, which calls
        ``_consume_sse_traces`` directly against the real SSE server.
        The background task is automatically cancelled when the server
        shuts down in fixture teardown (``asyncio.run`` cancels pending
        tasks before closing the loop).
        """
        response = await webconsole_server.post(
            "/register_sse_subscriber",
            json={"sse_server_url": "http://127.0.0.1:1"},
        )
        assert response.status_code == 200
        assert response.json() == {"ok": True}

    # ----- /dag_rendering -----
    @pytest.mark.asyncio
    async def test_dag_rendering_invalid_id(self, webconsole_server):
        """Covers the branch where the ``id`` query param can't be parsed
        as int, returning a ``Div`` with an error message (HTTP 200, HTML
        body).
        """
        response = await webconsole_server.get("/dag_rendering?id=notanumber")
        assert response.status_code == 200
        assert "Invalid execution ID" in response.text

    @pytest.mark.asyncio
    async def test_dag_rendering_no_tasktypes(self, webconsole_server):
        """Covers the branch where the execution_id is valid as int but
        no TaskTypes exist for it (e.g. execution doesn't exist),
        returning a ``Div`` error message via the tasktypes-list is None
        guard.
        """
        response = await webconsole_server.get("/dag_rendering?id=999999")
        assert response.status_code == 200
        assert "Invalid execution ID" in response.text

    @pytest.mark.asyncio
    async def test_dag_rendering_success_with_taskgroups(
        self, webconsole_server, seeded_async_dao
    ):
        """Covers the success branch where TaskTypes AND TaskGroups are
        present, exercising the ``taskgroups_list or []`` truthy path and
        the full Jinja2 SVG template rendering + JS file inlining.
        """
        dao, (exec_id, _tt_uuid) = seeded_async_dao
        # Inline-seed a TaskGroup so the taskgroups_list is truthy.
        async with dao._get_session() as session:
            session.add(
                TaskGroup(
                    uuid=uuid4(),
                    exec_id=exec_id,
                    order=0,
                    name="render_grp",
                    elements=[],
                )
            )
            await session.commit()
        response = await webconsole_server.get(f"/dag_rendering?id={exec_id}")
        assert response.status_code == 200
        # The rendered SVG contains inlined <script> blocks from the
        # static JS files concatenated by the route handler.
        assert "<svg" in response.text
        assert "<script>" in response.text

    # ----- /exec_current_progress -----
    @pytest.mark.asyncio
    async def test_exec_current_progress_invalid_id(self, webconsole_server):
        """Covers the branch where the ``id`` query param can't be
        parsed as int, returning a ``Div`` error message.
        """
        response = await webconsole_server.get("/exec_current_progress?id=notanumber")
        assert response.status_code == 200
        assert "Invalid execution ID" in response.text

    @pytest.mark.asyncio
    async def test_exec_current_progress_no_tasks(self, webconsole_server):
        """Covers the branch where execution_id is valid but no Tasks
        exist (tasks_list is None), returning a ``Div`` error message.
        """
        response = await webconsole_server.get("/exec_current_progress?id=999999")
        assert response.status_code == 200
        assert "Invalid execution ID" in response.text

    @pytest.mark.asyncio
    async def test_exec_current_progress_success(
        self, webconsole_server, seeded_async_dao
    ):
        """Covers the success branch where ``draw_chart`` is called with
        real Tasks and TaskGroups, returning the Gantt chart script as a
        string.
        """
        _dao, (exec_id, _tt_uuid) = seeded_async_dao
        response = await webconsole_server.get(f"/exec_current_progress?id={exec_id}")
        assert response.status_code == 200
        # draw_chart returns a <script>...</script> string.
        assert "script" in response.text.lower()

    # ----- /execution_info -----
    @pytest.mark.asyncio
    async def test_execution_info_invalid_id(self, webconsole_server):
        """Covers the branch where the ``id`` query param can't be
        parsed as int, returning a 500 plain-text ``Response``.
        """
        response = await webconsole_server.get("/execution_info?id=notanumber")
        assert response.status_code == 500
        assert "Invalid execution ID" in response.text

    @pytest.mark.asyncio
    async def test_execution_info_success(self, webconsole_server, seeded_async_dao):
        """Covers the success branch where the DAO returns the execution
        info dict containing name, username, start_timestamp, and
        docstring.
        """
        _dao, (exec_id, _tt_uuid) = seeded_async_dao
        response = await webconsole_server.get(f"/execution_info?id={exec_id}")
        assert response.status_code == 200
        body = response.json()
        assert body["name"] == "cnt_pipe"
        assert body["username"] == "alice"
        assert "start_timestamp" in body

    # ----- /execution_number -----
    @pytest.mark.asyncio
    async def test_execution_number_success(self, webconsole_server, seeded_async_dao):
        """Covers the route's delegation to
        ``events.execution_number()`` with a valid execution_id. The DAO
        returns the living-counts dict for the execution's pipeline name.
        """
        _dao, (exec_id, _tt_uuid) = seeded_async_dao
        response = await webconsole_server.get(f"/execution_number?id={exec_id}")
        assert response.status_code == 200
        body = response.json()
        assert body["name"] == "cnt_pipe"
        assert body["count"] >= 1

    # ----- /tasktype_docstring -----
    @pytest.mark.asyncio
    async def test_tasktype_docstring_invalid_uuid(self, webconsole_server):
        """Covers the branch where the ``uuid`` query param can't be
        parsed as a UUID, returning a 500 plain-text ``Response``.
        """
        response = await webconsole_server.get("/tasktype_docstring?uuid=not-a-uuid")
        assert response.status_code == 500
        assert "Invalid tasktype UUID" in response.text

    @pytest.mark.asyncio
    async def test_tasktype_docstring_success(
        self, webconsole_server, seeded_async_dao
    ):
        """Covers the success branch where the DAO returns the
        tasktype's docstring (``None`` for the seeded TaskType, which
        has no docstring set).
        """
        _dao, (exec_id, tt_uuid) = seeded_async_dao
        response = await webconsole_server.get(f"/tasktype_docstring?uuid={tt_uuid}")
        assert response.status_code == 200

    # ----- /pipeline-card -----
    @pytest.mark.asyncio
    async def test_pipeline_card_invalid_exec_id(self, webconsole_server):
        """Covers the branch where ``exec_id`` can't be parsed as int,
        raising ``HTTPException(500)``.
        """
        response = await webconsole_server.get("/pipeline-card?exec_id=notanumber")
        assert response.status_code == 500

    @pytest.mark.asyncio
    async def test_pipeline_card_execution_not_found(self, webconsole_server):
        """Covers the branch where ``dao.get_execution`` returns
        ``None``, triggering ``AttributeError`` on ``.name`` access and
        raising ``HTTPException(500)``.
        """
        response = await webconsole_server.get("/pipeline-card?exec_id=999999")
        assert response.status_code == 500

    @pytest.mark.asyncio
    async def test_pipeline_card_file_not_found(
        self, webconsole_server, seeded_async_dao, artifacts_store_root
    ):
        """Covers the branch where the execution exists but the
        ``pipeline_card.html`` artifact is absent at the
        ``artifacts_store_root``, raising ``HTTPException(404)``.

        The execution row is repointed to the parametrized
        ``artifacts_store_root`` so the test runs against both local FS
        and S3 MinIO backends.
        """
        dao, (exec_id, _tt_uuid) = seeded_async_dao
        await dao._async_update_entity(
            Execution, exec_id, artifacts_store_root=artifacts_store_root
        )
        response = await webconsole_server.get(f"/pipeline-card?exec_id={exec_id}")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_pipeline_card_success(
        self, webconsole_server, seeded_async_dao, artifacts_store_root
    ):
        """Covers the success branch where ``pipeline_card.html`` exists
        at the execution's ``artifacts_store_root``, returned as
        ``HTMLResponse`` 200.

        Runs against both local filesystem and S3 MinIO backends via
        the parametrized ``artifacts_store_root`` fixture.
        """
        dao, (exec_id, _tt_uuid) = seeded_async_dao
        # The seeded execution hard-codes artifacts_store_root to a
        # fixed path; repoint it to the parametrized fixture so the
        # test exercises both local FS and S3 MinIO backends.
        await dao._async_update_entity(
            Execution, exec_id, artifacts_store_root=artifacts_store_root
        )
        write_text_file(
            artifacts_store_root,
            ["cnt_pipe", str(exec_id), "pipeline_card.html"],
            "<html>card</html>",
        )
        response = await webconsole_server.get(f"/pipeline-card?exec_id={exec_id}")
        assert response.status_code == 200
        assert response.text == "<html>card</html>"

    # ----- /task_traces -----
    @pytest.mark.asyncio
    async def test_task_traces_invalid_task_id(self, webconsole_server):
        """Covers the branch where ``task_id`` can't be parsed as int,
        returning a 500 plain-text ``Response``.
        """
        response = await webconsole_server.get("/task_traces?task_id=notanumber")
        assert response.status_code == 500
        assert "Invalid task ID" in response.text

    @pytest.mark.asyncio
    async def test_task_traces_none(self, webconsole_server):
        """Covers the branch where the DAO returns ``None`` for traces
        (no traces for this task_id), and the route returns
        ``JSONResponse(None)``.
        """
        response = await webconsole_server.get("/task_traces?task_id=999999")
        assert response.status_code == 200
        assert response.json() is None

    @pytest.mark.asyncio
    async def test_task_traces_with_traces(self, webconsole_server, seeded_async_dao):
        """Covers the success branch where the DAO returns traces and the
        route converts each trace's timestamp to int milliseconds.
        """
        dao, (exec_id, _tt_uuid) = seeded_async_dao
        # The seeded execution has one Task; fetch its id and add a trace.
        tasks_list = await dao.get_execution_tasks_list(exec_id)
        assert tasks_list is not None
        task_id = tasks_list[0].id
        async with dao._get_session() as session:
            session.add(
                TaskTrace(
                    task_id=task_id,
                    timestamp=datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
                    content="trace line",
                    is_err=False,
                )
            )
            await session.commit()
        response = await webconsole_server.get(f"/task_traces?task_id={task_id}")
        assert response.status_code == 200
        body = response.json()
        assert len(body) == 1
        assert body[0]["content"] == "trace line"
        assert isinstance(body[0]["timestamp"], int)

    # ----- /execution_events -----
    @pytest.mark.asyncio
    async def test_execution_events_returns_streaming_response(self, webconsole_server):
        """Cover the route that wraps ``multiplexed_event_generator``
        in a ``StreamingResponse``.

        The route is hit via the real WebConsole uvicorn instance using
        an httpx streaming GET. Opening the response (which exercises
        every line of the route handler: ``ClientInfo`` construction +
        ``StreamingResponse`` instantiation) and reading the status code
        + ``Content-Type`` header is sufficient ; the body iterator is
        deliberately not consumed because it would block indefinitely
        waiting for SSE events that never arrive in this test context.
        Closing the stream cancels the generator on the server side.
        """
        async with webconsole_server.stream("GET", "/execution_events") as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")

    # ----- /execution (home) -----
    @pytest.mark.asyncio
    async def test_home_invalid_id(self, webconsole_server):
        """Covers the branch where the ``id`` query param can't be
        parsed as int, returning ``page_layout`` with empty content
        (HTTP 200, HTML).
        """
        response = await webconsole_server.get("/execution?id=notanumber")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_home_success(self, webconsole_server, seeded_async_dao):
        """Covers the success branch where a valid execution_id is
        provided, returning the full ``page_layout`` with the execution
        timeline, Gantt chart placeholder, and DAG renderer placeholder.
        """
        _dao, (exec_id, _tt_uuid) = seeded_async_dao
        response = await webconsole_server.get(f"/execution?id={exec_id}")
        assert response.status_code == 200
        # The rendered page includes distinctive markers from the template.
        assert "retrain-pipelines" in response.text
        assert "Execution DAG" in response.text
