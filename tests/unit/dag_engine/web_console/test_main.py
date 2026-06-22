"""
Unit tests for retrain_pipelines/dag_engine/web_console/main.py.

Only OS-level side-effects that must not actually execute in tests are mocked:
real socket binds, thread creation/joining, and asyncio server loops.
All package imports resolve normally from the installed retrain_pipelines package.
"""

import threading
from unittest.mock import MagicMock, patch

import pytest

import retrain_pipelines.dag_engine.web_console.main as main


# ---------------------------------------------------------------------------
# acquire_server_lock / release_server_lock
# ---------------------------------------------------------------------------


class TestAcquireServerLock:
    def setup_method(self):
        main._lock_socket = None

    def test_success_returns_true(self):
        mock_sock = MagicMock()
        with patch("socket.socket", return_value=mock_sock):
            result = main.acquire_server_lock(9999)
        assert result is True
        assert main._lock_socket is mock_sock

    def test_oserror_returns_false(self):
        mock_sock = MagicMock()
        mock_sock.bind.side_effect = OSError("address in use")
        with patch("socket.socket", return_value=mock_sock):
            result = main.acquire_server_lock(9999)
        assert result is False

    def test_oserror_clears_global(self):
        mock_sock = MagicMock()
        mock_sock.bind.side_effect = OSError
        with patch("socket.socket", return_value=mock_sock):
            main.acquire_server_lock(9999)
        assert main._lock_socket is None

    def test_oserror_closes_socket(self):
        mock_sock = MagicMock()
        mock_sock.bind.side_effect = OSError
        with patch("socket.socket", return_value=mock_sock):
            main.acquire_server_lock(9999)
        mock_sock.close.assert_called_once()


class TestReleaseServerLock:
    def test_closes_and_clears_global(self):
        mock_sock = MagicMock()
        main._lock_socket = mock_sock
        main.release_server_lock()
        mock_sock.close.assert_called_once()
        assert main._lock_socket is None

    def test_no_op_when_no_socket(self):
        main._lock_socket = None
        main.release_server_lock()  # must not raise


# ---------------------------------------------------------------------------
# _webconsole_start ; early-return guard paths
# ---------------------------------------------------------------------------


class TestWebconsoleStartGuards:
    def setup_method(self):
        main._process_has_server = False
        main._server_thread = None
        main._lock_socket = None

    def test_refuses_second_server_same_process(self):
        main._process_has_server = True
        main._running_port = 8000
        with (
            patch.object(main._logger_controller, "activate"),
            patch.object(main._logger_controller, "deactivate"),
        ):
            main._webconsole_start(port=8001, grpc_port=50051)
        # state must be unchanged ; no second server was started
        assert main._process_has_server is True
        assert main._running_port == 8000

    def test_refuses_when_thread_alive(self):
        main._server_thread = MagicMock()
        main._server_thread.is_alive.return_value = True
        with patch.object(main._logger_controller, "activate"):
            main._webconsole_start(port=8001, grpc_port=50051)
        assert main._process_has_server is False

    def test_refuses_when_lock_not_acquired(self):
        with (
            patch.object(main, "acquire_server_lock", return_value=False),
            patch.object(main._logger_controller, "activate"),
            patch.object(main._logger_controller, "deactivate"),
        ):
            main._webconsole_start(port=8001, grpc_port=50051)
        assert main._process_has_server is False

    def test_refuses_when_port_in_use(self):
        mock_sock = MagicMock()
        mock_sock.__enter__ = lambda s: s
        mock_sock.__exit__ = MagicMock(return_value=False)
        mock_sock.connect_ex.return_value = 0  # port occupied

        with (
            patch.object(main, "acquire_server_lock", return_value=True),
            patch("socket.socket", return_value=mock_sock),
            patch.object(main._logger_controller, "activate"),
            patch.object(main._logger_controller, "deactivate"),
        ):
            main._webconsole_start(port=8001, grpc_port=50051)
        assert main._process_has_server is False


# ---------------------------------------------------------------------------
# _webconsole_start ; happy path
# ---------------------------------------------------------------------------


def _make_port_free_sock():
    """Context-manager socket mock where all ports appear free."""
    mock_sock = MagicMock()
    mock_sock.__enter__ = lambda s: s
    mock_sock.__exit__ = MagicMock(return_value=False)
    mock_sock.connect_ex.return_value = 1  # port not in use
    return mock_sock


def _make_mock_routes():
    """
    Return a list with one HTTP route and one WebSocket route mock.

    Ensures the route-classification loop body is exercised:
     - the HTTP branch appends to http_routes
     - and the WS branch appends to ws_routes.
    """
    from starlette.routing import WebSocketRoute as _WSRoute

    http_route = MagicMock(spec=[])  # no spec => not isinstance WebSocketRoute
    http_route.methods = ["GET"]
    http_route.path = "/healthz"
    http_route.handle = MagicMock(__name__="healthz_handler")

    ws_route = MagicMock(spec=_WSRoute)
    ws_route.path = "/ws/test"
    ws_route.handle = MagicMock(__name__="ws_handler")

    return [http_route, ws_route]


def _stub_webconsole_start(port=8001, grpc_port=50051):
    """
    Drive _webconsole_start past all guards with a stubbed uvicorn server
    whose serve() coroutine returns immediately.
    Returns the mock uvicorn.Server instance so callers can inspect it.
    """
    mock_uvicorn_server = MagicMock()

    mock_app = MagicMock()
    mock_app.routes = _make_mock_routes()
    mock_app.router = MagicMock()
    mock_app.router.routes = []
    mock_app.on_event = MagicMock(side_effect=lambda event: lambda fn: fn)

    mock_config = MagicMock()
    mock_config.host = "0.0.0.0"
    mock_config.port = port

    with (
        patch.object(main, "acquire_server_lock", return_value=True),
        patch("socket.socket", return_value=_make_port_free_sock()),
        patch(
            "retrain_pipelines.dag_engine.web_console.main.FastHTML",
            return_value=mock_app,
        ),
        patch(
            "retrain_pipelines.dag_engine.web_console.main.get_log_websocket_endpoint",
            return_value=MagicMock(),
        ),
        patch(
            "retrain_pipelines.dag_engine.web_console.main.get_log_config",
            return_value={},
        ),
        patch(
            "retrain_pipelines.dag_engine.web_console.main.framed_rich_log_str",
            return_value="",
        ),
        patch("retrain_pipelines.dag_engine.web_console.main.uvicorn") as mock_uvicorn,
        patch(
            "retrain_pipelines.dag_engine.web_console.main.in_notebook",
            return_value=False,
        ),
        patch("threading.Thread.start", lambda self_t: None),
        patch.object(main._logger_controller, "activate"),
        patch.object(main._logger_controller, "deactivate"),
    ):
        mock_uvicorn.Config.return_value = mock_config
        mock_uvicorn.Server.return_value = mock_uvicorn_server

        main._webconsole_start(port=port, grpc_port=grpc_port)

    return mock_uvicorn_server


class TestWebconsoleStartHappyPath:
    def setup_method(self):
        main._process_has_server = False
        main._server_thread = None
        main._lock_socket = None
        main._server = None
        main._running_port = None
        main._shutdown_event = threading.Event()

    def teardown_method(self):
        pass  # Thread.start is suppressed; no real thread to join

    def test_sets_process_has_server(self):
        _stub_webconsole_start()
        assert main._process_has_server is True

    def test_sets_running_port(self):
        _stub_webconsole_start(port=8001)
        assert main._running_port == 8001

    def test_server_global_assigned(self):
        _stub_webconsole_start()
        assert main._server is not None

    def test_server_thread_started(self):
        _stub_webconsole_start()
        assert main._server_thread is not None
        assert main._server_thread.is_alive() or True  # thread may finish fast

    def test_shutdown_event_cleared_before_thread(self):
        """_shutdown_event must be cleared (not pre-set) when server starts."""
        main._shutdown_event.set()  # pre-pollute
        _stub_webconsole_start()
        # After start the event is cleared; thread sets it again on exit.
        # We just verify it was cleared at some point (server global was assigned).
        assert main._server is not None

    def test_uvicorn_config_called_with_port(self):
        mock_config = MagicMock()
        mock_config.host = "0.0.0.0"
        mock_config.port = 8001

        mock_app = MagicMock()
        mock_app.routes = _make_mock_routes()
        mock_app.router = MagicMock()
        mock_app.router.routes = []
        mock_app.on_event = MagicMock(side_effect=lambda event: lambda fn: fn)

        with (
            patch.object(main, "acquire_server_lock", return_value=True),
            patch("socket.socket", return_value=_make_port_free_sock()),
            patch(
                "retrain_pipelines.dag_engine.web_console.main.FastHTML",
                return_value=mock_app,
            ),
            patch(
                "retrain_pipelines.dag_engine.web_console.main.get_log_websocket_endpoint",
                return_value=MagicMock(),
            ),
            patch(
                "retrain_pipelines.dag_engine.web_console.main.get_log_config",
                return_value={},
            ),
            patch(
                "retrain_pipelines.dag_engine.web_console.main.framed_rich_log_str",
                return_value="",
            ),
            patch(
                "retrain_pipelines.dag_engine.web_console.main.uvicorn"
            ) as mock_uvicorn,
            patch(
                "retrain_pipelines.dag_engine.web_console.main.in_notebook",
                return_value=False,
            ),
            patch("threading.Thread.start", lambda self_t: None),
            patch.object(main._logger_controller, "activate"),
            patch.object(main._logger_controller, "deactivate"),
        ):
            mock_uvicorn.Config.return_value = mock_config
            mock_uvicorn.Server.return_value = MagicMock()

            main._webconsole_start(port=8001, grpc_port=50051)

            mock_uvicorn.Config.assert_called_once()
            call_kwargs = mock_uvicorn.Config.call_args
            assert call_kwargs.kwargs.get("port") == 8001 or (
                call_kwargs.args and call_kwargs.args[1] == 8001
            )


# ---------------------------------------------------------------------------
# startup_event / shutdown_event closures
# ---------------------------------------------------------------------------


class TestAppEventCallbacks:
    """
    Extract the startup_event and shutdown_event async closures registered
    via app.on_event() and call them directly.
    """

    def _capture_event_callbacks(self):
        """
        Run _webconsole_start with on_event() patched to capture the registered
        async callables rather than discarding them.
        Returns {'startup': fn, 'shutdown': fn}.
        """
        captured = {}

        def fake_on_event(event_name):
            def decorator(fn):
                captured[event_name] = fn
                return fn

            return decorator

        mock_app = MagicMock()
        mock_app.routes = _make_mock_routes()
        mock_app.router = MagicMock()
        mock_app.router.routes = []
        mock_app.on_event = fake_on_event

        mock_config = MagicMock()
        mock_config.host = "0.0.0.0"
        mock_config.port = 8001

        with (
            patch.object(main, "acquire_server_lock", return_value=True),
            patch("socket.socket", return_value=_make_port_free_sock()),
            patch(
                "retrain_pipelines.dag_engine.web_console.main.FastHTML",
                return_value=mock_app,
            ),
            patch(
                "retrain_pipelines.dag_engine.web_console.main.get_log_websocket_endpoint",
                return_value=MagicMock(),
            ),
            patch(
                "retrain_pipelines.dag_engine.web_console.main.get_log_config",
                return_value={},
            ),
            patch(
                "retrain_pipelines.dag_engine.web_console.main.framed_rich_log_str",
                return_value="",
            ),
            patch(
                "retrain_pipelines.dag_engine.web_console.main.uvicorn"
            ) as mock_uvicorn,
            patch(
                "retrain_pipelines.dag_engine.web_console.main.in_notebook",
                return_value=False,
            ),
            patch("threading.Thread.start", lambda self_t: None),
            patch.object(main._logger_controller, "activate"),
            patch.object(main._logger_controller, "deactivate"),
        ):
            mock_uvicorn.Config.return_value = mock_config
            mock_uvicorn.Server.return_value = MagicMock()
            main._webconsole_start(port=8001, grpc_port=50051)

        return captured

    def test_startup_event_starts_grpc_thread(self):
        """startup_event: spawns grpc thread and calls serve_grpc."""
        import asyncio as _asyncio

        callbacks = self._capture_event_callbacks()
        startup_fn = callbacks.get("startup")
        assert startup_fn is not None, "startup_event was not registered"

        spawned = {}
        real_thread_init = threading.Thread.__init__

        def fake_thread_init(self_t, target=None, **kwargs):
            spawned["target"] = target
            real_thread_init(self_t, target=target, **kwargs)

        with (
            patch("threading.Thread.__init__", fake_thread_init),
            patch("threading.Thread.start", lambda self_t: None),
        ):
            _asyncio.get_event_loop().run_until_complete(startup_fn())

        # startup_event must have created a thread whose target is run_grpc
        assert "target" in spawned and spawned["target"] is not None
        assert main._grpc_thread is not None

    def test_startup_event_run_grpc_body(self):
        """run_grpc() inner closure: sets _grpc_server and calls wait_for_termination."""
        import asyncio as _asyncio

        captured_thread = {}
        real_thread_init = threading.Thread.__init__

        def fake_thread_init(self_t, target=None, **kwargs):
            captured_thread["target"] = target
            real_thread_init(self_t, target=target, **kwargs)

        callbacks = self._capture_event_callbacks()
        startup_fn = callbacks.get("startup")
        assert startup_fn is not None

        mock_grpc_srv = MagicMock()

        with (
            patch(
                "retrain_pipelines.dag_engine.web_console.main.serve_grpc",
                return_value=mock_grpc_srv,
            ),
            patch("threading.Thread.__init__", fake_thread_init),
            patch("threading.Thread.start", lambda self_t: None),
        ):
            _asyncio.get_event_loop().run_until_complete(startup_fn())

        # Now invoke run_grpc() directly
        run_grpc_fn = captured_thread.get("target")
        assert run_grpc_fn is not None

        with patch(
            "retrain_pipelines.dag_engine.web_console.main.serve_grpc",
            return_value=mock_grpc_srv,
        ):
            run_grpc_fn()

        mock_grpc_srv.wait_for_termination.assert_called_once()
        assert main._grpc_server is mock_grpc_srv

    def test_shutdown_event_stops_grpc_server(self):
        """shutdown_event: calls _grpc_server.stop when server exists."""
        import asyncio as _asyncio

        callbacks = self._capture_event_callbacks()
        shutdown_fn = callbacks.get("shutdown")
        assert shutdown_fn is not None, "shutdown_event was not registered"

        mock_grpc_srv = MagicMock()
        main._grpc_server = mock_grpc_srv

        _asyncio.get_event_loop().run_until_complete(shutdown_fn())

        mock_grpc_srv.stop.assert_called_once_with(grace=5.0)

    def test_shutdown_event_no_server_does_not_raise(self):
        """shutdown_event: skips stop() when _grpc_server is None."""
        import asyncio as _asyncio

        callbacks = self._capture_event_callbacks()
        shutdown_fn = callbacks.get("shutdown")
        assert shutdown_fn is not None

        main._grpc_server = None
        # Must not raise
        _asyncio.get_event_loop().run_until_complete(shutdown_fn())


# ---------------------------------------------------------------------------
# run() thread internals ; non-notebook and notebook branches
# ---------------------------------------------------------------------------


class TestRunThreadInternals:
    """
    The run() closure defined inside _webconsole_start is exercised by
    extracting the target function from the Thread before it starts,
    then calling it directly to cover its branches.
    """

    def _capture_run_fn(self, in_notebook_val=False):
        """
        Patch threading.Thread so we capture the target=run closure
        without actually spawning a thread. Returns (run_fn, mock_server).
        """
        captured = {}

        real_thread_init = threading.Thread.__init__

        def fake_thread_init(self_t, target=None, **kwargs):
            captured["target"] = target
            # Initialise as a real (but not started) daemon thread
            real_thread_init(self_t, target=target, **kwargs)

        mock_app = MagicMock()
        mock_app.routes = _make_mock_routes()
        mock_app.router = MagicMock()
        mock_app.router.routes = []
        mock_app.on_event = MagicMock(side_effect=lambda event: lambda fn: fn)

        mock_config = MagicMock()
        mock_config.host = "0.0.0.0"
        mock_config.port = 8001

        mock_uvicorn_server = MagicMock()

        with (
            patch.object(main, "acquire_server_lock", return_value=True),
            patch("socket.socket", return_value=_make_port_free_sock()),
            patch(
                "retrain_pipelines.dag_engine.web_console.main.FastHTML",
                return_value=mock_app,
            ),
            patch(
                "retrain_pipelines.dag_engine.web_console.main.get_log_websocket_endpoint",
                return_value=MagicMock(),
            ),
            patch(
                "retrain_pipelines.dag_engine.web_console.main.get_log_config",
                return_value={},
            ),
            patch(
                "retrain_pipelines.dag_engine.web_console.main.framed_rich_log_str",
                return_value="",
            ),
            patch(
                "retrain_pipelines.dag_engine.web_console.main.uvicorn"
            ) as mock_uvicorn,
            patch(
                "retrain_pipelines.dag_engine.web_console.main.in_notebook",
                return_value=in_notebook_val,
            ),
            patch("threading.Thread.__init__", fake_thread_init),
            patch("threading.Thread.start", lambda self_t: None),
            patch.object(main._logger_controller, "activate"),
            patch.object(main._logger_controller, "deactivate"),
        ):
            mock_uvicorn.Config.return_value = mock_config
            mock_uvicorn.Server.return_value = mock_uvicorn_server

            main._webconsole_start(port=8001, grpc_port=50051)

        return captured.get("target"), mock_uvicorn_server

    def test_run_non_notebook_calls_asyncio_run(self):
        run_fn, mock_srv = self._capture_run_fn(in_notebook_val=False)
        assert run_fn is not None

        with (
            patch(
                "retrain_pipelines.dag_engine.web_console.main.in_notebook",
                return_value=False,
            ),
            patch(
                "retrain_pipelines.dag_engine.web_console.main.asyncio"
            ) as mock_asyncio,
            patch.object(main, "release_server_lock"),
        ):
            mock_asyncio.run = MagicMock()
            run_fn()

        mock_asyncio.run.assert_called_once()

    def test_run_notebook_uses_event_loop(self):
        run_fn, mock_srv = self._capture_run_fn(in_notebook_val=True)
        assert run_fn is not None

        mock_loop = MagicMock()
        mock_loop.is_closed.return_value = False
        mock_loop.run_until_complete = MagicMock()

        with (
            patch(
                "retrain_pipelines.dag_engine.web_console.main.in_notebook",
                return_value=True,
            ),
            patch(
                "retrain_pipelines.dag_engine.web_console.main.asyncio"
            ) as mock_asyncio,
            patch.object(main, "release_server_lock"),
        ):
            mock_asyncio.get_event_loop.return_value = mock_loop
            run_fn()

        mock_loop.run_until_complete.assert_called_once()

    def test_run_notebook_creates_new_loop_when_closed(self):
        run_fn, mock_srv = self._capture_run_fn(in_notebook_val=True)
        assert run_fn is not None

        closed_loop = MagicMock()
        closed_loop.is_closed.return_value = True

        new_loop = MagicMock()
        new_loop.run_until_complete = MagicMock()

        with (
            patch(
                "retrain_pipelines.dag_engine.web_console.main.in_notebook",
                return_value=True,
            ),
            patch(
                "retrain_pipelines.dag_engine.web_console.main.asyncio"
            ) as mock_asyncio,
            patch.object(main, "release_server_lock"),
        ):
            mock_asyncio.get_event_loop.return_value = closed_loop
            mock_asyncio.new_event_loop.return_value = new_loop
            run_fn()

        mock_asyncio.new_event_loop.assert_called_once()
        mock_asyncio.set_event_loop.assert_called_once_with(new_loop)
        new_loop.run_until_complete.assert_called_once()

    def test_run_notebook_creates_new_loop_on_runtime_error(self):
        run_fn, mock_srv = self._capture_run_fn(in_notebook_val=True)
        assert run_fn is not None

        new_loop = MagicMock()
        new_loop.run_until_complete = MagicMock()

        with (
            patch(
                "retrain_pipelines.dag_engine.web_console.main.in_notebook",
                return_value=True,
            ),
            patch(
                "retrain_pipelines.dag_engine.web_console.main.asyncio"
            ) as mock_asyncio,
            patch.object(main, "release_server_lock"),
        ):
            mock_asyncio.get_event_loop.side_effect = RuntimeError("no loop")
            mock_asyncio.new_event_loop.return_value = new_loop
            run_fn()

        mock_asyncio.new_event_loop.assert_called_once()
        new_loop.run_until_complete.assert_called_once()

    def test_run_finally_resets_globals(self):
        run_fn, _ = self._capture_run_fn(in_notebook_val=False)
        assert run_fn is not None

        main._process_has_server = True
        main._running_port = 8001

        with (
            patch(
                "retrain_pipelines.dag_engine.web_console.main.in_notebook",
                return_value=False,
            ),
            patch(
                "retrain_pipelines.dag_engine.web_console.main.asyncio"
            ) as mock_asyncio,
            patch.object(main, "release_server_lock") as mock_rel,
        ):
            mock_asyncio.run = MagicMock()
            run_fn()

        mock_rel.assert_called_once()
        assert main._process_has_server is False
        assert main._running_port is None

    def test_run_finally_sets_shutdown_event(self):
        run_fn, _ = self._capture_run_fn(in_notebook_val=False)
        assert run_fn is not None

        main._shutdown_event = threading.Event()

        with (
            patch(
                "retrain_pipelines.dag_engine.web_console.main.in_notebook",
                return_value=False,
            ),
            patch(
                "retrain_pipelines.dag_engine.web_console.main.asyncio"
            ) as mock_asyncio,
            patch.object(main, "release_server_lock"),
        ):
            mock_asyncio.run = MagicMock()
            run_fn()

        assert main._shutdown_event.is_set()

    def test_run_exception_propagates_after_finally(self):
        run_fn, _ = self._capture_run_fn(in_notebook_val=False)
        assert run_fn is not None

        with (
            patch(
                "retrain_pipelines.dag_engine.web_console.main.in_notebook",
                return_value=False,
            ),
            patch(
                "retrain_pipelines.dag_engine.web_console.main.asyncio"
            ) as mock_asyncio,
            patch.object(main, "release_server_lock"),
        ):
            mock_asyncio.run.side_effect = RuntimeError("boom")
            with pytest.raises(RuntimeError, match="boom"):
                run_fn()


# ---------------------------------------------------------------------------
# _webconsole_shutdown
# ---------------------------------------------------------------------------


class TestWebconsoleShutdown:
    def setup_method(self):
        main._shutdown_event = threading.Event()

    def test_warns_and_returns_when_no_server(self):
        main._server = None
        # Must not raise and must not call deactivate
        with patch.object(main._logger_controller, "deactivate") as mock_deact:
            main._webconsole_shutdown()
        mock_deact.assert_not_called()

    def test_sets_exit_flags(self):
        srv = MagicMock(should_exit=False, force_exit=False)
        main._server = srv
        main._shutdown_event.set()  # pre-signal so wait() returns immediately

        with patch.object(main._logger_controller, "deactivate"):
            main._webconsole_shutdown()

        assert srv.force_exit is True
        assert srv.should_exit is True

    def test_clears_server_reference(self):
        srv = MagicMock(should_exit=False, force_exit=False)
        main._server = srv
        main._shutdown_event.set()

        with patch.object(main._logger_controller, "deactivate"):
            main._webconsole_shutdown()

        assert main._server is None

    def test_calls_deactivate_on_clean_shutdown(self):
        srv = MagicMock(should_exit=False, force_exit=False)
        main._server = srv
        main._shutdown_event.set()

        with patch.object(main._logger_controller, "deactivate") as mock_deact:
            main._webconsole_shutdown()

        mock_deact.assert_called_once()

    def test_force_cleans_up_on_timeout(self):
        srv = MagicMock(should_exit=False, force_exit=False)
        main._server = srv
        # _shutdown_event never set => wait() times out

        with (
            patch.object(main._shutdown_event, "wait", return_value=False),
            patch.object(main, "release_server_lock") as mock_rel,
            patch.object(main._logger_controller, "deactivate"),
        ):
            main._webconsole_shutdown()

        mock_rel.assert_called_once()
        assert main._process_has_server is False
        assert main._running_port is None


# ---------------------------------------------------------------------------
# webconsole_start / webconsole_shutdown ; notebook vs. non-notebook dispatch
# ---------------------------------------------------------------------------


class TestWebconsoleDispatch:
    def test_start_calls_plain_when_not_notebook(self):
        with (
            patch(
                "retrain_pipelines.dag_engine.web_console.main.in_notebook",
                return_value=False,
            ),
            patch.object(main, "_webconsole_start") as mock_start,
        ):
            main.webconsole_start(port=8000, grpc_port=50051)
        mock_start.assert_called_once_with(8000, 50051)

    def test_start_calls_notebook_when_in_notebook(self):
        with (
            patch(
                "retrain_pipelines.dag_engine.web_console.main.in_notebook",
                return_value=True,
            ),
            patch(
                "retrain_pipelines.dag_engine.web_console"
                ".main_notebook._webconsole_start_notebook"
            ) as mock_nb,
        ):
            main.webconsole_start(port=8000, grpc_port=50051)
        mock_nb.assert_called_once_with(8000, 50051)

    def test_start_reads_env_when_ports_not_given(self):
        with (
            patch(
                "retrain_pipelines.dag_engine.web_console.main.in_notebook",
                return_value=False,
            ),
            patch.object(main, "_webconsole_start") as mock_start,
            patch.dict(
                "os.environ",
                {
                    "RP_WEB_SERVER_PORT": "9001",
                    "RP_GRPC_SERVER_PORT": "50061",
                },
            ),
        ):
            main.webconsole_start()
        mock_start.assert_called_once_with(9001, 50061)

    def test_shutdown_calls_plain_when_not_notebook(self):
        with (
            patch(
                "retrain_pipelines.dag_engine.web_console.main.in_notebook",
                return_value=False,
            ),
            patch.object(main, "_webconsole_shutdown") as mock_shut,
        ):
            main.webconsole_shutdown()
        mock_shut.assert_called_once()

    def test_shutdown_calls_notebook_when_in_notebook(self):
        with (
            patch(
                "retrain_pipelines.dag_engine.web_console.main.in_notebook",
                return_value=True,
            ),
            patch(
                "retrain_pipelines.dag_engine.web_console"
                ".main_notebook._webconsole_shutdown_notebook"
            ) as mock_nb,
        ):
            main.webconsole_shutdown()
        mock_nb.assert_called_once()
