
import os
import sys
import time
import atexit
import socket
import threading


# --- WebConsole served via the CLI utility ----------------------------------------------
"""
CLI daemon design — rationale and trade-offs
=============================================

Goal
----
Two entry-points, ``webconsole_start`` and ``webconsole_shutdown``, that behave
naturally from any interactive terminal:
  - one instance per terminal (a second start from the same tab is refused)
  - different terminals may each run their own instance, on different ports
  - two terminals may NOT share the same port (the port is a unique resource)
  - closing the terminal automatically terminates the instance it owns
  - ``webconsole_shutdown`` only affects the instance owned by the calling terminal


Daemon strategy — why single-fork with parent-death monitor, not double-fork
-----------------------------------------------------------------------------
The classic Unix daemon pattern uses a double-fork + ``os.setsid()`` to create a
new session with no controlling terminal. ``os.setsid()`` is precisely what
makes the process immune to SIGHUP. When the terminal closes, the kernel sends
SIGHUP to every process in the session, but a process that called ``os.setsid()``
has left the original session and will never receive it.

A single-fork keeping the child in the original session was tried, with a SIGHUP
handler. However SIGHUP on terminal closure is only delivered to the *foreground*
process group ; a background child after fork never receives it.

The current approach uses a single fork and a lightweight monitor thread.
The shell PID is captured via ``os.getppid()`` *before* forking. This is
critical: after the fork the parent (entry-point script) exits immediately, so
``os.getppid()`` in the child would return 1 (init) from the very first poll,
triggering instant shutdown. The child inherits the captured shell PID and the
monitor thread calls ``os.kill(shell_pid, 0)`` every 2 seconds ; a no-op that
merely checks the process is still alive. When the terminal closes and the shell
dies, that call raises ``ProcessLookupError``; the monitor calls
``webconsole_shutdown()`` and exits.


Terminal identity ; why pts device, not session/pgrp/ppid
----------------------------------------------------------
Several approaches were tried for keying the pid file to the terminal:

  - ``os.getsid(0)`` (session ID): under WSL and many Linux terminal emulators,
    all tabs of the same application share a session. Useless for per-tab identity.

  - ``os.getpgrp()`` (process group): each command invocation spawns in its own
    process group. Two successive ``webconsole_start`` calls from the same tab
    produce different pgrp values → different pid files → singleton guard never
    fires. Does not work.

  - ``os.getppid()`` (parent PID, i.e. the shell): stable within a tab and distinct
    across tabs in most cases, but not guaranteed cross-platform ; a terminal
    multiplexer or init-style launcher may share a parent across tabs.

  - ``os.ttyname(fd)`` on the pts device: the pts (pseudo-terminal slave) node,
    e.g. ``/dev/pts/3``, is allocated by the kernel per open terminal tab and is
    guaranteed unique across all tabs on Linux, WSL, and macOS. It is the only
    identifier that is both stable within a tab and distinct across tabs on all
    platforms.

    Caveat: entry-point scripts launched by pip/setuptools redirect ``sys.stdin``,
    so ``os.ttyname(sys.stdin.fileno())`` raises ``OSError``. ``sys.stdout`` and
    ``sys.stderr`` remain connected to the terminal. The implementation therefore
    tries stdout first, then stderr, then stdin, and falls back to ``os.getppid()``
    only if all three are redirected (e.g. fully non-interactive invocation).


Port availability — why no port-scoped pid file
------------------------------------------------
An earlier revision maintained a second pid file keyed on the port to prevent two
terminals from starting on the same port. This is unnecessary: ``acquire_server_lock``
inside ``_webconsole_start`` attempts to bind the port and fails immediately if
anything (another webconsole instance or any other process) already holds it.
The CLI layer adds an explicit ``connect_ex`` probe ahead of the start attempt so
the error message is printed to the terminal before the logger is active.  Two
pid files tracking the same fact would be redundant and error-prone.


Pid file format
---------------
Each pid file contains a single line: ``{pid}:{port}``.
This lets ``webconsole_shutdown_cli`` recover both the PID to signal and the port
to report in the confirmation message, without needing a separate lookup.
"""


def _pid_file():
    for fd in (sys.stdout.fileno(), sys.stderr.fileno(),
               sys.stdin.fileno()):
        try:
            tty = os.ttyname(fd).replace('/', '_')
            return f"/tmp/webconsole{tty}.pid"
        except OSError:
            pass
    return f"/tmp/webconsole_{os.getppid()}.pid"


def _daemonize(port):
    """Fork into background and monitor the shell for terminal closure.

    The shell PID is captured before forking. After the fork, the parent
    (entry-point script) exits immediately — so inside the child,
    os.getppid() would return 1 (init), not the shell. Capturing it
    beforehand gives the child a stable reference to the shell process.

    A background thread polls os.kill(shell_pid, 0) every two seconds.
    That call does nothing but raises ProcessLookupError when the process
    no longer exists. When the shell dies (terminal closed), the monitor
    catches that error and triggers a clean shutdown.
    """
    from . import main as _main

    # Capture the shell PID before forking ; after the fork the parent
    # (entry-point script) exits immediately, so os.getppid() in the
    # child would change at once. We watch the shell instead.
    shell_pid = os.getppid()

    pid = os.fork()
    if pid > 0:
        sys.exit(0)

    pid_file = _pid_file()
    with open(pid_file, "w") as f:
        f.write(f"{os.getpid()}:{port}")

    atexit.register(
        lambda: os.unlink(pid_file) if os.path.exists(pid_file) else None
    )

    # Monitor parent death: when the shell exits (terminal closed),
    # shell_pid disappears — kill(shell_pid, 0) raises ProcessLookupError.
    def _watch_parent():
        while True:
            time.sleep(2)
            try:
                os.kill(shell_pid, 0)
            except ProcessLookupError:
                _main.webconsole_shutdown()
                os._exit(0)
    threading.Thread(target=_watch_parent, daemon=True).start()

    sys.stdout.flush()
    sys.stderr.flush()
    with open(os.devnull, 'r') as f:
        os.dup2(f.fileno(), sys.stdin.fileno())
    with open(os.devnull, 'a+') as f:
        os.dup2(f.fileno(), sys.stdout.fileno())
        os.dup2(f.fileno(), sys.stderr.fileno())

    _main.webconsole_start(port=port)

    try:
        while True:
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        _main.webconsole_shutdown()


def webconsole_start_cli():
    import argparse
    parser = argparse.ArgumentParser(prog="webconsole_start")
    parser.add_argument("-p", "--port", type=int, default=None)
    parser.add_argument("-gp", "--grpc-port", type=int, default=None,
                        dest="grpc_port")
    parser.add_argument("-f", "--foreground", dest="foreground",
                        action="store_true",
                        help="Run in foreground (persistent, not a daemon)")
    args = parser.parse_args()

    from . import main as _main

    port = args.port or int(os.environ.get("RP_WEB_SERVER_PORT"))
    grpc_port = args.port or int(os.environ.get("RP_GRPC_SERVER_PORT"))

    pid_file = _pid_file()
    if os.path.exists(pid_file):
        with open(pid_file) as f:
            existing_pid, existing_port = f.read().strip().split(":")
        try:
            os.kill(int(existing_pid), 0)
            print(
                f"\N{cross mark} An instance of the WebConsole is already "
                f"running on this terminal (on port {existing_port}).")
            return
        except ProcessLookupError:
            os.unlink(pid_file)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as _s:
        _s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        ports_list = [port, grpc_port]
        for verify_port in ports_list:
            if _s.connect_ex(('127.0.0.1', verify_port)) == 0:
                print(
                    f"\N{cross mark} Can't start a WebConsole instance on ports "
                    f"{ports_list}, port {verify_port} is not available.")
                return

    if args.foreground:
        with open(pid_file, "w") as f:
            f.write(f"{os.getpid()}:{port}")
        atexit.register(
            lambda: os.unlink(pid_file) if os.path.exists(pid_file) else None
        )
        _main.webconsole_start(port=port, grpc_port=grpc_port)
        print(
            f"\N{white heavy check mark} WebConsole successfully started "
            f"on port {port}.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            _main.webconsole_shutdown()
    else:
        print(
            f"\N{white heavy check mark} WebConsole successfully started "
            f"on port {port}.")
        _daemonize(port)


def webconsole_shutdown_cli():
    import signal

    pid_file = _pid_file()

    if not os.path.exists(pid_file):
        print(
            f"\N{warning sign} No WebConsole instance running "
            "from this terminal.")
        return

    with open(pid_file) as f:
        pid, port = f.read().strip().split(":")

    try:
        os.kill(int(pid), signal.SIGTERM)
        print(f"\N{white heavy check mark} WebConsole on port {port} is down.")
    except ProcessLookupError:
        print(f"\N{warning sign} No process found for pid={pid}")

    try:
        os.unlink(pid_file)
    except FileNotFoundError:
        pass

