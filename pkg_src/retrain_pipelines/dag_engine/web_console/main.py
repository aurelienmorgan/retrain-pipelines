import asyncio
import logging
import socket
import sys
import threading
import time
import traceback
from collections.abc import Callable

import uvicorn
from fasthtml import FastHTML
from starlette.exceptions import HTTPException
from starlette.routing import WebSocketRoute

from ...utils import in_notebook
from ...utils.rich_logging import framed_rich_log_str
from ..config import Config
from ..rp_logging import RichLoggingController
from . import api
from .grpc_server import serve_grpc
from .main_cli_utility import webconsole_shutdown_cli, webconsole_start_cli  # noqa: F401
from .utils.server_logs import get_log_config, get_log_websocket_endpoint
from .views import execution, home, server
from .views.error_pages import error_page

logger = logging.getLogger()


_server = None
_server_thread = None
_server_loop = None
_lock_socket = None
_process_has_server = False
_running_port = None
_shutdown_event = threading.Event()

_grpc_server = None
_grpc_thread = None

_logger_controller = RichLoggingController()


def acquire_server_lock(port: int) -> bool:
    """Try to acquire exclusive lock for this port.

    Returns True if successful.
    """
    global _lock_socket

    try:
        # create a socket bound to localhost:port ;
        # fails if another process already has the port
        _lock_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        _lock_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        _lock_socket.bind(("127.0.0.1", port))
        return True
    except OSError:
        # port already in use
        if _lock_socket:
            _lock_socket.close()
            _lock_socket = None
        return False


def release_server_lock():
    """Release the lock."""
    global _lock_socket
    if _lock_socket:
        try:
            _lock_socket.close()
        finally:
            _lock_socket = None


def _webconsole_start(port: int, grpc_port: int):
    """Start a webconsole instance on the calling process.

    We disallow several webconsole instances
    (be it on different uvicorn ports) on a single process
    for teminal logs readability.
    """
    global _server, _server_thread, _process_has_server, _running_port, _shutdown_event

    _logger_controller.activate()

    # ############## #
    # port singleton #
    # ############## #
    # Check if the herein process already has any server running
    if _process_has_server:
        logger.error(
            "\N{CROSS MARK} Server already running "
            + f"on port {_running_port} in this process. "
            + f"Cannot start another server on port {port} "
            + "in the same process."
        )
        _logger_controller.deactivate()
        return

    # Check if server thread is still alive
    # (shouldn't happen with flag above, but safety check)
    if _server_thread and _server_thread.is_alive():
        logger.warning("\N{WHITE HEAVY CHECK MARK} Server already running " + "in this process.")
        return

    # Try to acquire lock
    # (checks OTHER processes on this specific port)
    if not acquire_server_lock(port):
        logger.warning(
            "\N{WHITE HEAVY CHECK MARK} Server already running "
            + f"on port {port} in another process."
        )
        _logger_controller.deactivate()
        return
    # check ports availability
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as _s:
        _s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        ports_list = [port, grpc_port]
        for verify_port in ports_list:
            if _s.connect_ex(("127.0.0.1", verify_port)) == 0:
                logger.warning(
                    f"\N{CROSS MARK} Can't start a WebConsole instance on ports "
                    f"{ports_list}, port {verify_port} is not available."
                )
                _logger_controller.deactivate()
                return
    ##################

    logger.info("\N{ROCKET} Starting server...")
    logger.info("logs going to " + f"{Config.get_web_server_logs_root()}")

    # ########## #
    # app server #
    # ########## #
    # declare error pages handler
    def make_handler(code):
        return lambda req, exc: error_page(code, req, exc)

    exception_handlers: dict[int | type[HTTPException], Callable] = {
        code: make_handler(code) for code in range(400, 600)
    }
    exception_handlers[HTTPException] = lambda req, exc: error_page(exc.status_code, req, exc)

    # Create FastHTML app and route
    app = FastHTML(exception_handlers=exception_handlers, exts="ws")
    ##############

    # ########### #
    # gRPC server #
    # ########### #
    @app.on_event("startup")
    async def startup_event():
        global _grpc_thread, _grpc_server

        # Start gRPC server in background
        def run_grpc():
            global _grpc_server
            _grpc_server = serve_grpc(grpc_port=grpc_port)
            _grpc_server.wait_for_termination()

        _grpc_thread = threading.Thread(target=run_grpc, daemon=True)
        _grpc_thread.start()

        server_url = f"http://{display_host}:{grpc_port}"
        logger.info(f"\N{GLOWING STAR} gRPC server thread started at {server_url}")

    @app.on_event("shutdown")
    async def shutdown_event():
        global _grpc_server

        if _grpc_server:
            _grpc_server.stop(grace=5.0)
        logger.info("\N{WHITE HEAVY CHECK MARK} gRPC server stopped")

    ###############

    # ################### #
    # routes registration #
    # ################### #
    rt = app.route
    for view in [api, home, server, execution]:
        view.register(app, rt)

    # Add the server-logs WebSocket route
    web_socket_endpoint_route = "/web_server/stream_logs"
    app.router.routes.append(
        WebSocketRoute(
            web_socket_endpoint_route, get_log_websocket_endpoint(web_socket_endpoint_route)
        )
    )

    http_routes = []
    ws_routes = []

    for route in getattr(app, "routes", []):
        if isinstance(route, WebSocketRoute):
            methods = ["WS"]
        else:
            methods = route.methods

        path = route.path
        handle_name = getattr(route.handle, "__name__", str(route.handle))

        if "WS" in methods:
            ws_routes.append((path, handle_name))
        else:
            http_routes.append((methods, path, handle_name))

    logger.info("\n--- Registered HTTP Routes ---")
    registered_routes_str = ""
    for methods, path, handle in http_routes:
        registered_routes_str += "\n" + f"{methods} {path} -> {handle}"
    logger.info(framed_rich_log_str(registered_routes_str[1:]))
    logger.info("--- Registered WebSocket Endpoints ---")
    registered_routes_str = ""
    for path, handle in ws_routes:
        registered_routes_str += "\n" + f"['ws'] {path} -> {handle}"
    logger.info(framed_rich_log_str(registered_routes_str[1:]))

    #######################

    # Create the ASGI server (uvicorn.Server)
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        access_log=True,
        proxy_headers=True,
        forwarded_allow_ips="*",
        reload=False,
        log_config=get_log_config(),
        ws="wsproto",
        # ws_ping_interval=None,
        # ws_ping_timeout=None,
        timeout_keep_alive=30,
        backlog=4_096,
        h11_max_incomplete_event_size=1_048_576,
    )
    _server = uvicorn.Server(config)

    # Reset shutdown event for new server
    _shutdown_event.clear()

    def run():
        try:
            logger.info(f"SERVER THREAD STARTING (in_notebook={in_notebook()})")
            if in_notebook():
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_closed():
                        raise RuntimeError("Loop is closed")
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                loop.run_until_complete(_server.serve())
            else:
                asyncio.run(_server.serve())

            logger.info("SERVER EXITED")
        except Exception as e:
            logger.error(f"CRASHED: {e}\n{traceback.format_exc()}")
            raise
        finally:
            release_server_lock()
            global _process_has_server, _running_port
            _process_has_server = False
            _running_port = None
            _shutdown_event.set()

    _server_thread = threading.Thread(target=run, daemon=True)
    _server_thread.start()

    # Mark that this process now has a server and track the port
    _process_has_server = True
    _running_port = port

    display_host = config.host
    if display_host == "0.0.0.0":
        display_host = "localhost"
    server_url = f"http://{display_host}:{config.port}"
    logger.info(f"\N{GLOWING STAR} Server started at {server_url}")


def _webconsole_shutdown():
    global _server, _server_thread, _process_has_server, _running_port, _shutdown_event

    if not _server or not hasattr(_server, "should_exit"):
        logger.warning("\N{WARNING SIGN} Server is not running " + "in this process.")
        return

    logger.info("\N{OCTAGONAL SIGN} Shutting down server...")
    _server.force_exit = True
    _server.should_exit = True

    # Wait for shutdown to complete (using the event)
    timeout = 60
    shutdown_complete = _shutdown_event.wait(timeout=timeout)

    if shutdown_complete:
        logger.info("\N{WHITE HEAVY CHECK MARK} Server is down")
    else:
        logger.warning(
            f"\N{WARNING SIGN} Server did not shut down cleanly within {timeout} seconds"
        )
        # Force cleanup anyway
        release_server_lock()
        _process_has_server = False
        _running_port = None

    _logger_controller.deactivate()

    _server = None


def webconsole_start(port=None, grpc_port=None):
    """Start a WebConsole instance on the calling process.

    We disallow several webconsole instances
    (be it on different uvicorn ports) on a single process
    for teminal logs readability.
    """
    if port is None:
        port = Config.get_web_server_port()

    if grpc_port is None:
        grpc_port = Config.get_grpc_server_port()

    if in_notebook():
        from .main_notebook import _webconsole_start_notebook

        _webconsole_start_notebook(port, grpc_port)
    else:
        _webconsole_start(port, grpc_port)


def webconsole_shutdown():
    if in_notebook():
        from .main_notebook import _webconsole_shutdown_notebook

        _webconsole_shutdown_notebook()
    else:
        _webconsole_shutdown()


# ----------------------------------------------------------------------------------------


if __name__ == "__main__":  # pragma: no cover
    # override port via command line: python main.py 5002 50052
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            logger.error(f"\N{CROSS MARK} Invalid port: {sys.argv[1]}")
            sys.exit(1)
        if len(sys.argv) > 2:
            try:
                grpc_port = int(sys.argv[2])
            except ValueError:
                logger.error(f"\N{CROSS MARK} Invalid port: {sys.argv[1]}")
                sys.exit(1)

    webconsole_start(port=port, grpc_port=grpc_port)
    logger.info(
        f"\N{GLOBE WITH MERIDIANS} Visit http://localhost:{port}/web_server to view server logs"
    )

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("\nReceived shutdown signal...")
        webconsole_shutdown()
        time.sleep(0.5)
