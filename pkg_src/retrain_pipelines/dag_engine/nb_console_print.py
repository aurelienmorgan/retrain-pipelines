
import os
import threading
import multiprocessing

from typing import Optional

from rich.text import Text
from rich.console import Console


# ---- Notebook subprocess log relay ----
#
# Sub-DAG tasks and taskgroup tasks run in child processes.
# In notebook mode, calling console.print() from a child process triggers:
#   [IPKernelApp] WARNING | WARNING: attempted to send message from fork
# because IPython's ZMQ socket can only be used from the main process.
#
# Fix: children publish their rich.text.Text object to a multiprocessing
# Queue. The main process relay thread drains the queue and calls
# console.print() itself — the only process allowed to touch ZMQ.
#
# On Linux (fork default) children inherit _main_pid and _notebook_relay
# automatically. On macOS/Windows (spawn default) wire the queue via the
# pool initializer — see _build_pool_initializer() below.


_main_pid: Optional[int] = None
_notebook_relay: Optional["_NotebookLogRelay"] = None


def _in_subprocess() -> bool:
    """True when called from a child process (not the process that called
    RichLoggingController.activate())."""
    return _main_pid is not None and os.getpid() != _main_pid


class _OrigStdoutWrapper:
    """Thin file-like wrapper around a bare write callable.

    Lets Console accept an unwrapped original stdout.write function
    as its file= argument without triggering the patched write path.
    """
    def __init__(self, write_fn):
        self.write = write_fn
    def flush(self): pass
    def isatty(self): return True


class _NotebookLogRelay:
    """Relay rich.text.Text log messages from child processes to the
    main notebook cell output.

    Data flow:
        child  → publish(Text) → queue → listener thread → console.print()
                                                              ↓
                                                     IPython OutStream (ZMQ) ✓
                                                     notebook cell output    ✓

    The listener thread and console.print() only ever run in the main process.
    """

    def __init__(self, orig_stdout_write):
        self._orig_write = orig_stdout_write
        self._queue: multiprocessing.Queue = multiprocessing.Queue()
        self._stop_evt = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="nb-log-relay", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_evt.set()
        try:
            self._queue.put_nowait(None)
        except Exception:
            pass
        self._thread.join(timeout=3.0)

    def publish(self, rich_text: Text) -> None:
        """Called from a child process. Never raises."""
        try:
            self._queue.put_nowait(rich_text)
        except Exception:
            pass

    def _run(self) -> None:
        file = _OrigStdoutWrapper(self._orig_write)
        while not self._stop_evt.is_set():
            try:
                item = self._queue.get(timeout=0.05)
                if item is None:
                    break
                Console(
                    file=file,
                    soft_wrap=False,
                    width=1_000_000,
                    force_terminal=True,
                    legacy_windows=False,
                ).print(item, sep="")
            except Exception:
                pass
        # drain any items that arrived after the stop signal
        while True:
            try:
                item = self._queue.get_nowait()
                if not item:
                    break
                Console(
                    file=file,
                    soft_wrap=False,
                    width=1_000_000,
                    force_terminal=True,
                    legacy_windows=False,
                ).print(item, sep="")
            except Exception:
                break


def _build_pool_initializer():
    """Return (initializer, initargs) for spawn-based ProcessPoolExecutors.

    On macOS / Windows, where multiprocessing defaults to 'spawn', child
    processes do not inherit module globals. Pass the returned tuple to
    ProcessPoolExecutor so each worker has _main_pid and _notebook_relay set:

        init_fn, initargs = _build_pool_initializer()
        executor = ProcessPoolExecutor(
            initializer=init_fn, initargs=initargs
        )

    On Linux (fork default) this is unnecessary but harmless.
    """
    queue = _notebook_relay._queue if _notebook_relay else None
    orig_write = _notebook_relay._orig_write if _notebook_relay else None

    def _worker_init(main_pid, relay_queue, relay_orig_write):
        import sys as _sys
        mod = None
        for _mod in list(_sys.modules.values()):
            if getattr(_mod, "_NotebookLogRelay", None) is _NotebookLogRelay:
                mod = _mod
                break
        if mod is None:
            return
        mod._main_pid = main_pid
        if relay_queue is not None:
            relay = _NotebookLogRelay.__new__(_NotebookLogRelay)
            relay._orig_write = relay_orig_write
            relay._queue = relay_queue
            relay._stop_evt = threading.Event()
            relay._thread = None    # no listener needed in worker
            mod._notebook_relay = relay

    return _worker_init, (_main_pid, queue, orig_write)


def nb_activate(orig_stdout_write) -> None:
    """Record main PID and start the relay so subprocess log messages
    reach the notebook cell output safely."""
    global _main_pid, _notebook_relay
    _main_pid = os.getpid()
    if _notebook_relay is None:
        _notebook_relay = _NotebookLogRelay(orig_stdout_write)
        _notebook_relay.start()


def nb_deactivate() -> None:
    """Stop the relay before restoring stdout so the final drain still
    reaches the notebook cell output."""
    global _notebook_relay
    if _notebook_relay is not None:
        _notebook_relay.stop()
        _notebook_relay = None


def nb_console_print(console, rich_formatted_message) -> None:
    if _in_subprocess():
        # child process: do NOT touch ZMQ from here.
        # publish the Text object to the main process relay,
        # which will call console.print() on our behalf.
        if _notebook_relay is not None:
            _notebook_relay.publish(rich_formatted_message)
    else:
        # let Notebook render in cell output via its own mechanism
        # (we later "short" the "_custom_write" delegation
        #  to "original")
        console.print(rich_formatted_message, sep="")

