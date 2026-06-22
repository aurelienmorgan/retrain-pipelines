"""Shared fixtures for web_console unit tests."""

import threading
import pytest


@pytest.fixture(autouse=True)
def reset_main_globals():
    """Restore all module-level globals in main.py after each test."""
    import retrain_pipelines.dag_engine.web_console.main as main

    yield

    main._server = None
    main._server_thread = None
    main._server_loop = None
    main._lock_socket = None
    main._process_has_server = False
    main._running_port = None
    main._shutdown_event = threading.Event()
    main._grpc_server = None
    main._grpc_thread = None
