import os
import time
import json
import uvicorn
import asyncio
import logging
import threading

from datetime import datetime
from collections import deque

from fasthtml.common import fast_app

from .utils.server_logs import get_log_config
from .views import home, server, ui


_server = None
_server_thread = None


def start_server_once():
    global _server, _server_thread

    if _server_thread and _server_thread.is_alive():
        print("✅ Server already running.")
        return

    logging.getLogger().info("\N{rocket} Starting server...")
    logging.getLogger().info(f"logs going to {os.environ['RP_WEB_SERVER_LOGS']}")

    # Create FastHTML app and route
    app, rt = fast_app(exts='ws')
    # app = FastHtml(exts='ws')
    # rt = app.route

    for view in [home, server, ui]:
        view.register(app, rt)

    # Create the ASGI server (uvicorn.Server)
    config = uvicorn.Config(
        app, host="0.0.0.0", port=5001,
        log_level="info", access_log=True, log_config=get_log_config()
    )
    _server = uvicorn.Server(config)

    def run():
        asyncio.run(_server.serve())

    _server_thread = threading.Thread(target=run, daemon=True)
    _server_thread.start()


def shutdown_server():
    global _server
    if _server and hasattr(_server, "should_exit"):
        print("🛑 Shutting down server...")
        _server.should_exit = True
    else:
        print("⚠️ Server is not running.")


if __name__ == "__main__":
    start_server_once()
    print("🌐 Visit http://localhost:5001/web_server to view server logs")
    time.sleep(30)  # Let it run longer to see logs

    shutdown_server()
    time.sleep(3)  # Wait to confirm shutdown

