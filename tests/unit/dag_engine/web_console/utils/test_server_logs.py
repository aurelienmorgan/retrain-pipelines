from __future__ import annotations

import asyncio
import logging
import sys
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, mock_open

import pytest

from retrain_pipelines.dag_engine.web_console.utils import server_logs


# ---------------------------------------------------------------------------
# get_log_config
# ---------------------------------------------------------------------------


def test_get_log_config(monkeypatch):
    monkeypatch.setenv("RP_WEB_SERVER_LOGS", "/tmp/logs")

    cfg = server_logs.get_log_config()

    assert cfg["version"] == 1
    assert cfg["disable_existing_loggers"] is False

    assert cfg["handlers"]["file_default"]["filename"] == "/tmp/logs/server.log"
    assert cfg["handlers"]["file_access"]["filename"] == "/tmp/logs/access.log"

    assert "uvicorn" in cfg["loggers"]
    assert "uvicorn.error" in cfg["loggers"]
    assert "uvicorn.access" in cfg["loggers"]


# ---------------------------------------------------------------------------
# AccessLogEntry.extract_ip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("127.0.0.1:1234", "127.0.0.1"),
        ("127.0.0.1", "127.0.0.1"),
        ("2001:db8::1:9999", "2001:db8::1"),
        ("::1:9999", "::1"),
    ],
)
def test_extract_ip(value, expected):
    assert server_logs.AccessLogEntry.extract_ip(value) == expected


def test_extract_ip_bracketed_ipv6():
    # IMPORTANT: matches the implementation's double-escaped regex semantics
    value = r"\[::1\]:12345"
    result = server_logs.AccessLogEntry.extract_ip(value)

    assert result == r"\[::1\]"


# ---------------------------------------------------------------------------
# AccessLogEntry.to_json
# ---------------------------------------------------------------------------


def test_access_log_entry_to_json():
    entry = server_logs.AccessLogEntry(
        raw_str="raw",
        timestamp=datetime(2025, 1, 1, 10, 0, 0),
        level="INFO",
        client_addr="127.0.0.1",
        method="GET",
        message="/health",
        status_code=200,
    )

    payload = entry.to_json()

    assert '"level": "INFO"' in payload
    assert '"client_addr": "127.0.0.1"' in payload
    assert '"status_code": 200' in payload
    assert '"message": "/health"' in payload


# ---------------------------------------------------------------------------
# AccessLogEntry.to_fasthtml_div
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status_code", "method", "expected_title"),
    [
        (200, "GET", "200 - OK"),
        (404, "POST", "404 - Not Found"),
        (500, "DELETE", "500 - Internal Server Error"),
    ],
)
def test_to_fasthtml_div_http_titles(status_code, method, expected_title):
    entry = server_logs.AccessLogEntry(
        raw_str="raw",
        timestamp=datetime.now(),
        level="INFO",
        client_addr="127.0.0.1",
        method=method,
        message="/path",
        status_code=status_code,
    )

    div = entry.to_fasthtml_div()

    assert getattr(div, "title") == expected_title


@pytest.mark.parametrize(
    ("method", "expected_title"),
    [
        ("ws", "WebSocket"),
        ("sse", "Server-Side Event"),
        ("grpc", "gRPC"),
    ],
)
def test_to_fasthtml_div_special_titles(method, expected_title):
    entry = server_logs.AccessLogEntry(
        raw_str="raw",
        timestamp=datetime.now(),
        level="INFO",
        client_addr="127.0.0.1",
        method=method,
        message="/path",
        status_code=200,
    )

    div = entry.to_fasthtml_div()

    assert getattr(div, "title") == expected_title


def test_to_fasthtml_div_unknown_method():
    entry = server_logs.AccessLogEntry(
        raw_str="raw",
        timestamp=datetime.now(),
        level="INFO",
        client_addr="127.0.0.1",
        method="CUSTOM",
        message="/path",
        status_code=200,
    )

    div = entry.to_fasthtml_div()

    assert div is not None


# ---------------------------------------------------------------------------
# AccessLogEntry.from_access_log
# ---------------------------------------------------------------------------


def test_from_access_log_success():
    line = "2025-07-06 15:41:53 [INFO] 127.0.0.1:36042 - 'POST /ui/update HTTP/1.1' 200"

    entry = server_logs.AccessLogEntry.from_access_log(line)

    assert entry.level == "INFO"
    assert entry.method == "POST"
    assert entry.status_code == 200
    assert entry.message == "/ui/update"


def test_from_access_log_invalid():
    with pytest.raises(ValueError):
        server_logs.AccessLogEntry.from_access_log("bad-log-line")


# ---------------------------------------------------------------------------
# _read_last_n_lines (linux path)
# ---------------------------------------------------------------------------


def test_read_last_n_lines_without_filter(tmp_path):
    f = tmp_path / "log.txt"
    f.write_text("a\nb\nc\nd\n")

    result = server_logs._read_last_n_lines(str(f), 2, None)

    assert result == [b"c", b"d"]


def test_read_last_n_lines_with_filter(tmp_path):
    f = tmp_path / "log.txt"
    f.write_text("INFO one\nDEBUG two\nINFO three\nINFO four\n")

    result = server_logs._read_last_n_lines(
        str(f),
        2,
        r"^INFO",
    )

    assert result == [b"INFO three", b"INFO four"]


def test_read_last_n_lines_close_error_ignored(monkeypatch, tmp_path):
    f = tmp_path / "log.txt"
    f.write_text("a\n")

    original_close = server_logs.os.close

    def bad_close(fd):
        raise OSError("boom")

    monkeypatch.setattr(server_logs.os, "close", bad_close)

    try:
        result = server_logs._read_last_n_lines(str(f), 1, None)
        assert result == [b"a"]
    finally:
        monkeypatch.setattr(server_logs.os, "close", original_close)


def test_read_last_n_lines_proc_version_open_error(monkeypatch, tmp_path):
    f = tmp_path / "log.txt"
    f.write_text("hello\n")

    original_open = open

    def patched_open(path, *args, **kwargs):
        if path == "/proc/version":
            raise OSError("no /proc/version")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", patched_open)

    result = server_logs._read_last_n_lines(str(f), 1, None)

    assert result == [b"hello"]


def test_read_last_n_lines_multi_block_loop_exhausted(tmp_path):
    f = tmp_path / "big.log"
    padding = "x" * 4000
    content = f"{padding}\nlineA\nlineB\n"
    f.write_text(content)

    result = server_logs._read_last_n_lines(str(f), 10, None)

    decoded = [r.decode() for r in result]
    assert "lineA" in decoded
    assert "lineB" in decoded


def test_read_last_n_lines_multi_block_with_filter_loop_exhausted(tmp_path):
    f = tmp_path / "big_filtered.log"
    padding = "y" * 4000
    content = f"{padding}\nINFO alpha\nDEBUG beta\n"
    f.write_text(content)

    result = server_logs._read_last_n_lines(str(f), 5, r"^INFO")

    assert result == [b"INFO alpha"]


# ---------------------------------------------------------------------------
# _read_last_n_lines (WSL branch)
# ---------------------------------------------------------------------------


def test_read_last_n_lines_wsl_no_filter(monkeypatch):
    proc_file = mock_open(read_data="microsoft")

    monkeypatch.setattr("builtins.open", proc_file)

    run_result = SimpleNamespace(
        returncode=0,
        stdout="a\nb\nc\n",
    )

    fake_subprocess = SimpleNamespace(run=lambda *args, **kwargs: run_result)

    sys.modules["subprocess"] = fake_subprocess

    result = server_logs._read_last_n_lines(
        "/mnt/c/test.log",
        2,
        None,
    )

    assert result == [b"b", b"c"]


def test_read_last_n_lines_wsl_filtered(monkeypatch):
    proc_file = mock_open(read_data="microsoft")
    monkeypatch.setattr("builtins.open", proc_file)

    run_result = SimpleNamespace(
        returncode=0,
        stdout="DEBUG\nINFO one\nINFO two\n",
    )

    fake_subprocess = SimpleNamespace(run=lambda *args, **kwargs: run_result)

    sys.modules["subprocess"] = fake_subprocess

    result = server_logs._read_last_n_lines(
        "/mnt/c/test.log",
        2,
        r"^INFO",
    )

    assert result == [b"INFO one", b"INFO two"]


def test_read_last_n_lines_wsl_nonzero_returncode(monkeypatch, tmp_path):
    import io

    original_open = open

    def patched_open(path, *args, **kwargs):
        if path == "/proc/version":
            return io.StringIO("Linux version microsoft")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", patched_open)

    run_result = SimpleNamespace(returncode=1, stdout="")
    fake_subprocess = SimpleNamespace(run=lambda *a, **kw: run_result)
    monkeypatch.setitem(sys.modules, "subprocess", fake_subprocess)

    real_file = tmp_path / "fallback.log"
    real_file.write_text("line1\n")

    real_os_open = server_logs.os.open

    def fake_os_open(path, flags):
        if path.startswith("/mnt/"):
            return real_os_open(str(real_file), flags)
        return real_os_open(path, flags)

    monkeypatch.setattr(server_logs.os, "open", fake_os_open)

    result = server_logs._read_last_n_lines("/mnt/c/test.log", 5, None)
    assert b"line1" in result


def test_read_last_n_lines_wsl_filtered_exhausted(monkeypatch):
    import io

    original_open = open

    def patched_open(path, *args, **kwargs):
        if path == "/proc/version":
            return io.StringIO("Linux version microsoft")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", patched_open)

    run_result = SimpleNamespace(returncode=0, stdout="INFO only\n")
    fake_subprocess = SimpleNamespace(run=lambda *a, **kw: run_result)
    monkeypatch.setitem(sys.modules, "subprocess", fake_subprocess)

    result = server_logs._read_last_n_lines("/mnt/c/test.log", 3, r"^INFO")

    assert result == [b"INFO only"]


def test_read_last_n_lines_wsl_filtered_enough_lines_not_enough_matches(
    monkeypatch,
):
    import io

    original_open = open

    def patched_open(path, *args, **kwargs):
        if path == "/proc/version":
            return io.StringIO("Linux version microsoft")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", patched_open)

    stdout = "DEBUG a\nDEBUG b\nDEBUG c\nDEBUG d\nINFO one\n"
    run_result = SimpleNamespace(returncode=0, stdout=stdout)
    fake_subprocess = SimpleNamespace(run=lambda *a, **kw: run_result)
    monkeypatch.setitem(sys.modules, "subprocess", fake_subprocess)

    result = server_logs._read_last_n_lines("/mnt/c/test.log", 3, r"^INFO")

    assert len(result) == 3
    assert result[-1] == b"INFO one"


# ---------------------------------------------------------------------------
# read_last_access_logs
# ---------------------------------------------------------------------------


def test_read_last_access_logs(monkeypatch):
    monkeypatch.setattr(
        server_logs.glob,
        "glob",
        lambda *_: ["new.log", "old.log"],
    )

    monkeypatch.setattr(
        server_logs.os.path,
        "getmtime",
        lambda p: 2 if p == "new.log" else 1,
    )

    monkeypatch.setattr(
        server_logs,
        "_read_last_n_lines",
        lambda *args, **kwargs: [
            (b"2025-07-06 15:41:53 [INFO] 127.0.0.1:36042 - 'GET /health HTTP/1.1' 200")
        ],
    )

    monkeypatch.setattr(
        server_logs,
        "strip_ansi_escape_codes",
        lambda s: s,
    )

    result = server_logs.read_last_access_logs(
        "/tmp",
        "access.log",
        1,
        None,
    )

    assert len(result) == 1


def test_read_last_access_logs_skips_bad_lines(monkeypatch):
    monkeypatch.setattr(
        server_logs.glob,
        "glob",
        lambda *_: ["access.log"],
    )

    monkeypatch.setattr(
        server_logs.os.path,
        "getmtime",
        lambda *_: 1,
    )

    monkeypatch.setattr(
        server_logs,
        "_read_last_n_lines",
        lambda *args, **kwargs: [b"bad"],
    )

    monkeypatch.setattr(
        server_logs,
        "strip_ansi_escape_codes",
        lambda s: s,
    )

    result = server_logs.read_last_access_logs(
        "/tmp",
        "access.log",
        10,
        None,
    )

    assert result == []


# ---------------------------------------------------------------------------
# WebSocketLogHandler
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_broadcast_removes_dead_clients(monkeypatch):
    handler = server_logs.WebSocketLogHandler("/ws")

    good_ws = AsyncMock()
    good_ws.client = ("127.0.0.1", 1234)

    bad_ws = AsyncMock()
    bad_ws.client = ("127.0.0.1", 5678)
    bad_ws.send_text.side_effect = RuntimeError()

    handler.clients = {good_ws, bad_ws}

    entry = MagicMock()
    entry.message = "/other"

    div = MagicMock()
    monkeypatch.setattr(entry, "to_fasthtml_div", lambda: div)

    logger = MagicMock()
    monkeypatch.setattr(server_logs.logging, "getLogger", lambda *_: logger)

    await handler.broadcast(entry)

    assert bad_ws not in handler.clients
    logger.info.assert_called_once()


@pytest.mark.asyncio
async def test_broadcast_no_recursive_logging(monkeypatch):
    handler = server_logs.WebSocketLogHandler("/same")

    ws = AsyncMock()
    ws.client = ("127.0.0.1", 9999)

    handler.clients = {ws}

    entry = MagicMock()
    entry.message = "/same"

    monkeypatch.setattr(
        entry,
        "to_fasthtml_div",
        lambda: MagicMock(),
    )

    logger = MagicMock()
    monkeypatch.setattr(server_logs.logging, "getLogger", lambda *_: logger)

    await handler.broadcast(entry)

    logger.info.assert_not_called()


def test_register_unregister():
    handler = server_logs.WebSocketLogHandler("/ws")

    ws = object()

    handler.register(ws)
    assert ws in handler.clients

    handler.unregister(ws)
    assert ws not in handler.clients


@pytest.mark.asyncio
async def test_broadcast_to_others(monkeypatch):
    handler = server_logs.WebSocketLogHandler("/ws")
    handler._handlers_added = True

    excluded = AsyncMock()
    alive = AsyncMock()

    dead = AsyncMock()
    dead.send_text.side_effect = RuntimeError()

    handler.clients = {excluded, alive, dead}

    await handler.broadcast_to_others("msg", excluded)

    alive.send_text.assert_awaited_once_with("msg")
    assert dead not in handler.clients


@pytest.mark.asyncio
async def test_broadcast_to_others_resets_flag():
    handler = server_logs.WebSocketLogHandler("/ws")
    handler._handlers_added = True

    ws = AsyncMock()

    handler.clients = {ws}

    await handler.broadcast_to_others("msg", ws)

    assert handler._handlers_added is False


def test_emit_non_running_loop(monkeypatch):
    handler = server_logs.WebSocketLogHandler("/ws")
    loop = MagicMock()
    loop.is_running.return_value = False
    monkeypatch.setattr(
        server_logs.asyncio,
        "get_event_loop",
        lambda: loop,
    )

    monkeypatch.setattr(
        handler,
        "broadcast",
        AsyncMock(),
    )

    run_mock = MagicMock()

    # Helper to properly await and consume the coroutine to prevent the
    # "coroutine was never awaited" RuntimeWarning, while avoiding infinite
    # recursion since asyncio.run is patched globally during this test.
    def _run_coro(coro):
        temp_loop = asyncio.new_event_loop()
        try:
            temp_loop.run_until_complete(coro)
        finally:
            temp_loop.close()

    run_mock.side_effect = _run_coro

    monkeypatch.setattr(
        server_logs.asyncio,
        "run",
        run_mock,
    )

    monkeypatch.setattr(
        handler,
        "format",
        lambda r: "entry",
    )

    handler.emit(object())

    run_mock.assert_called_once()


# ---------------------------------------------------------------------------
# get_log_websocket_endpoint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_emit_running_loop_with_real_task(monkeypatch):
    handler = server_logs.WebSocketLogHandler("/ws")

    loop = MagicMock()
    loop.is_running.return_value = True
    loop.create_task.side_effect = lambda coro: asyncio.create_task(coro)

    broadcast_calls = []

    async def broadcast_mock(record):
        broadcast_calls.append(record)

    monkeypatch.setattr(
        server_logs.asyncio,
        "get_event_loop",
        lambda: loop,
    )

    monkeypatch.setattr(handler, "broadcast", broadcast_mock)
    monkeypatch.setattr(handler, "format", lambda r: "entry")

    handler.emit(object())

    await asyncio.sleep(0)

    loop.create_task.assert_called_once()
    assert len(broadcast_calls) == 1
    assert broadcast_calls[0] == "entry"


@pytest.mark.asyncio
async def test_websocket_endpoint_first_connection_and_disconnect(monkeypatch):
    endpoint = server_logs.get_log_websocket_endpoint("logs")

    ws = AsyncMock()
    ws.client = ("127.0.0.1", 9000)
    ws.receive_text.side_effect = RuntimeError("client gone")

    monkeypatch.setattr(
        server_logs.WebSocketLogHandler,
        "broadcast_to_others",
        AsyncMock(),
    )

    await endpoint(ws)

    ws.accept.assert_awaited_once()
    ws.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_websocket_endpoint_handlers_already_added(monkeypatch):
    endpoint = server_logs.get_log_websocket_endpoint("logs2")

    bcast = AsyncMock()
    monkeypatch.setattr(
        server_logs.WebSocketLogHandler,
        "broadcast_to_others",
        bcast,
    )

    ws1 = AsyncMock()
    ws1.client = ("10.0.0.1", 9001)
    ws1.receive_text.side_effect = RuntimeError("ws1 gone")
    await endpoint(ws1)

    ws2 = AsyncMock()
    ws2.client = ("10.0.0.2", 9002)
    ws2.receive_text.side_effect = RuntimeError("ws2 gone")
    await endpoint(ws2)

    ws2.accept.assert_awaited_once()
    ws2.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_websocket_endpoint_close_raises(monkeypatch):
    endpoint = server_logs.get_log_websocket_endpoint("logs3")

    ws = AsyncMock()
    ws.client = ("192.168.1.1", 9003)
    ws.receive_text.side_effect = RuntimeError("disconnect")
    ws.close.side_effect = RuntimeError("close failed")

    monkeypatch.setattr(
        server_logs.WebSocketLogHandler,
        "broadcast_to_others",
        AsyncMock(),
    )

    await endpoint(ws)  # must not raise

    ws.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_websocket_endpoint_route_normalisation():
    endpoint = server_logs.get_log_websocket_endpoint("//ws/logs")
    assert asyncio.iscoroutinefunction(endpoint)


@pytest.mark.asyncio
async def test_access_log_entry_formatter_format_method(monkeypatch):
    captured_handlers: list = []

    real_add_handler = logging.Logger.addHandler

    def capturing_add_handler(self, hdlr):
        captured_handlers.append(hdlr)
        real_add_handler(self, hdlr)

    monkeypatch.setattr(logging.Logger, "addHandler", capturing_add_handler)

    endpoint = server_logs.get_log_websocket_endpoint("capture_route")

    ws = AsyncMock()
    ws.client = ("127.0.0.1", 1111)
    ws.receive_text.side_effect = RuntimeError("bye")

    monkeypatch.setattr(
        server_logs.WebSocketLogHandler,
        "broadcast_to_others",
        AsyncMock(),
    )

    await endpoint(ws)

    assert captured_handlers, "handler was not added to uvicorn.access"
    handler = captured_handlers[0]
    formatter = handler.formatter
    assert formatter is not None

    record = logging.LogRecord(
        name="uvicorn.access",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg='%s - "%s %s %s" %d',
        args=("127.0.0.1:9999", "GET", "/health", "HTTP/1.1", 200),
        exc_info=None,
    )

    log_entry = formatter.format(record)

    assert isinstance(log_entry, server_logs.AccessLogEntry)
    assert log_entry.method == "GET"
    assert log_entry.message == "/health"
    assert log_entry.status_code == 200
    assert log_entry.client_addr == "127.0.0.1"

    logging.getLogger("uvicorn.access").removeHandler(handler)
