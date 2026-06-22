import os
from unittest.mock import MagicMock, patch
from queue import Empty

import pytest
from rich.text import Text

from retrain_pipelines.dag_engine import nb_console_print as nb_module

_in_subprocess = nb_module._in_subprocess
_OrigStdoutWrapper = nb_module._OrigStdoutWrapper
_NotebookLogRelay = nb_module._NotebookLogRelay
_build_pool_initializer = nb_module._build_pool_initializer
nb_activate = nb_module.nb_activate
nb_deactivate = nb_module.nb_deactivate
nb_console_print = nb_module.nb_console_print


@pytest.fixture(autouse=True)
def reset_module_state():
    """Ensure global state is reset before and after each test."""
    nb_module._main_pid = None
    nb_module._notebook_relay = None
    yield
    nb_module._main_pid = None
    if nb_module._notebook_relay is not None:
        # Defensive check: only call stop() if the thread was actually started
        if nb_module._notebook_relay._thread is not None:
            nb_module._notebook_relay.stop()
        nb_module._notebook_relay = None


def test_in_subprocess():
    # Case 1: _main_pid is None
    nb_module._main_pid = None
    assert _in_subprocess() is False

    # Case 2: _main_pid matches current process
    nb_module._main_pid = os.getpid()
    assert _in_subprocess() is False

    # Case 3: _main_pid differs from current process
    nb_module._main_pid = -1
    assert _in_subprocess() is True


def test_orig_stdout_wrapper():
    written = []

    def mock_write(s):
        written.append(s)

    wrapper = _OrigStdoutWrapper(mock_write)
    assert wrapper.write("hello") == 5
    assert written == ["hello"]

    wrapper.flush()  # Should not raise
    assert wrapper.isatty() is True


def test_notebook_log_relay_normal_flow():
    written = []

    def mock_write(s):
        written.append(s)

    relay = _NotebookLogRelay(mock_write)
    relay.start()
    relay.publish(Text("test message"))
    relay.stop()

    # Give it a moment to process if needed, though stop() joins the thread
    assert any("test message" in w for w in written)


def test_notebook_log_relay_publish_exception():
    relay = _NotebookLogRelay(lambda s: None)
    relay.start()  # Must start the thread before calling stop()
    relay._queue.put_nowait = MagicMock(side_effect=Exception("queue full"))

    # Should not raise
    relay.publish(Text("test"))
    relay.stop()


def test_notebook_log_relay_stop_exception():
    relay = _NotebookLogRelay(lambda s: None)
    relay.start()
    relay._queue.put_nowait = MagicMock(side_effect=Exception("fail"))

    # Should not raise
    relay.stop()


def test_notebook_log_relay_first_loop_exception():
    relay = _NotebookLogRelay(lambda s: None)

    call_count = 0

    def mock_get(timeout):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise Exception("get failed")
        raise Empty()

    relay._queue.get = mock_get

    is_set_calls = [False, True]

    def mock_is_set():
        return is_set_calls.pop(0) if is_set_calls else True

    relay._stop_evt.is_set = mock_is_set

    # Should handle exception and exit cleanly without hanging
    relay._run()


def test_notebook_log_relay_stop_via_none():
    relay = _NotebookLogRelay(lambda s: None)
    relay._queue.put_nowait(None)

    with patch.object(nb_module, "Console") as mock_console_class:
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console
        relay._run()
        mock_console.print.assert_not_called()


def test_notebook_log_relay_drain_loop_print():
    relay = _NotebookLogRelay(lambda s: None)
    relay._stop_evt.set()  # Skip the first loop

    call_count = 0

    def mock_get_nowait():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return Text("drain item")
        raise Empty()

    relay._queue.get_nowait = mock_get_nowait

    with patch.object(nb_module, "Console") as mock_console_class:
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console
        relay._run()
        mock_console.print.assert_called_once()


def test_notebook_log_relay_drain_loop_break_on_falsy():
    relay = _NotebookLogRelay(lambda s: None)
    relay._stop_evt.set()

    call_count = 0

    def mock_get_nowait():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return None  # Falsy item should break the loop
        raise Empty()

    relay._queue.get_nowait = mock_get_nowait

    with patch.object(nb_module, "Console") as mock_console_class:
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console
        relay._run()
        mock_console.print.assert_not_called()


def test_build_pool_initializer_without_relay():
    nb_module._notebook_relay = None
    nb_module._main_pid = None

    init_fn, initargs = _build_pool_initializer()

    assert initargs == (None, None, None)


def test_build_pool_initializer_with_relay():
    nb_module._main_pid = 999
    nb_module._notebook_relay = _NotebookLogRelay(lambda s: None)
    nb_module._notebook_relay.start()

    # Save reference to clean up the background thread later
    old_relay = nb_module._notebook_relay

    init_fn, initargs = _build_pool_initializer()

    assert initargs[0] == 999
    assert initargs[1] is old_relay._queue
    assert initargs[2] is old_relay._orig_write

    # Test _worker_init with relay_queue is not None
    mock_queue = MagicMock()
    mock_orig_write = MagicMock()
    init_fn(123, mock_queue, mock_orig_write)

    assert nb_module._main_pid == 123
    assert nb_module._notebook_relay._queue is mock_queue
    assert nb_module._notebook_relay._thread is None

    # Clean up the old running thread to prevent test leakage
    old_relay.stop()


def test_worker_init_module_not_found():
    nb_module._main_pid = None
    nb_module._notebook_relay = None
    init_fn, _ = _build_pool_initializer()

    with patch("sys.modules") as mock_sys_modules:
        mock_sys_modules.values.return_value = [MagicMock()]
        init_fn(123, None, None)

    # Should return early without modifying state
    assert nb_module._main_pid is None


def test_worker_init_relay_queue_none():
    nb_module._main_pid = None
    nb_module._notebook_relay = None

    init_fn, _ = _build_pool_initializer()

    # sys.modules will naturally contain nb_module, so mod will be found.
    init_fn(123, None, None)

    assert nb_module._main_pid == 123
    assert nb_module._notebook_relay is None  # Unchanged because relay_queue is None


def test_nb_activate():
    nb_module._main_pid = None
    nb_module._notebook_relay = None

    nb_activate(lambda s: None)

    assert nb_module._main_pid == os.getpid()
    assert nb_module._notebook_relay is not None

    old_relay = nb_module._notebook_relay
    nb_activate(lambda s: None)

    # Should not recreate the relay
    assert nb_module._notebook_relay is old_relay


def test_nb_deactivate():
    nb_module._notebook_relay = None
    nb_deactivate()  # Should not raise

    nb_module._notebook_relay = _NotebookLogRelay(lambda s: None)
    nb_module._notebook_relay.start()

    nb_deactivate()

    assert nb_module._notebook_relay is None


def test_nb_console_print_not_in_subprocess():
    nb_module._main_pid = None
    nb_module._notebook_relay = None
    console = MagicMock()
    msg = Text("hello")

    nb_console_print(console, msg)
    console.print.assert_called_once_with(msg, sep="")


def test_nb_console_print_in_subprocess_with_relay():
    nb_module._main_pid = -1
    nb_module._notebook_relay = _NotebookLogRelay(lambda s: None)
    nb_module._notebook_relay.start()  # Must start the thread
    console = MagicMock()
    msg = Text("hello")

    with patch.object(nb_module._notebook_relay, "publish") as mock_publish:
        nb_console_print(console, msg)
        mock_publish.assert_called_once_with(msg)


def test_nb_console_print_in_subprocess_no_relay():
    nb_module._main_pid = -1
    nb_module._notebook_relay = None
    console = MagicMock()
    msg = Text("hello")

    nb_console_print(console, msg)
    console.print.assert_not_called()
