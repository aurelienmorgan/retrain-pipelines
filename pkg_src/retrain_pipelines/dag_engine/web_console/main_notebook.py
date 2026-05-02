
import os
import re
import sys
import time
import logging
import logging.handlers
import threading

from functools import lru_cache

from IPython.display import clear_output, display, HTML


logger = logging.getLogger()


# --- WebConsole served inside a Notebook ------------------------------------------------


def _ngrok_start(port):
    """Start an ngrok tunnel on *port* and return its public URL.

    Token resolution order (both Colab and Kaggle):
      1. ``NGROK_AUTHTOKEN`` already in the environment  →  use it as-is.
      2. Platform secrets store (Colab / Kaggle)  →  set and use it.
      3. Prompt the user; if left empty, attempt connection anyway so
         ngrok's own error message is shown (``PyngrokNgrokError``).
    Results:
        - tunnel_url (str) or None on failure
    """
    from pyngrok import ngrok
    from pyngrok.exception import PyngrokNgrokError
    logging.getLogger("pyngrok").setLevel(logging.WARNING)

    if not os.environ.get("NGROK_AUTHTOKEN"):
        env = _notebook_env()
        token = None

        if env == 'colab':
            try:
                from google.colab import userdata
                from google.colab.userdata import SecretNotFoundError
                token = userdata.get("NGROK_AUTHTOKEN")
            except (SecretNotFoundError, Exception):
                pass
        elif env == 'kaggle':
            try:
                from kaggle_secrets import UserSecretsClient
                token = UserSecretsClient().get_secret("NGROK_AUTHTOKEN")
            except Exception:
                pass

        if token:
            os.environ["NGROK_AUTHTOKEN"] = token
        else:
            logger.info(
                  f"{env.capitalize()} tip: add NGROK_AUTHTOKEN " +
                  f"to {env.capitalize()} Secrets " +
                  "to skip this prompt on kernel restart.")
            token = input(
                "NGROK_AUTHTOKEN (leave empty for more info): "
            ).strip()
            if token:
                os.environ["NGROK_AUTHTOKEN"] = token

    try:
        tunnel = ngrok.connect(port, bind_tls=True)
    except PyngrokNgrokError as e:
        logger.warning("[white on #9a2bab]"
            "[#c39c1a on black]ngrok[/] is a free, "
            "widely-used tunneling tool "
            "that exposes your notebook's network\n" +
            "so the WebConsole is reachable from your browser.\n" +
            "A free account takes ~1 minute to create, "
            "lasts forever, and will serve you\nwell beyond " +
            "[#c39c1a on black]retrain-pipelines[/]."
        "[/]")
        msg = e.ngrok_error.encode("raw_unicode_escape") \
                  .decode("unicode_escape") \
                  .replace("\r", "").strip()
        msg = re.sub(
            r'(https?://\S+)',
            r'[#4d0066 on #FFD700 link=\1]\1[/]',
            msg
        )
        logger.warning(f"[bold]{msg}[/]")
        return None
    return tunnel.public_url

def _ngrok_stop():
    """Kill all active ngrok tunnels."""
    from pyngrok import ngrok
    ngrok.kill()


@lru_cache
def _notebook_env():
    """Return 'colab', 'kaggle', or 'local'."""
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return "kaggle"
    try:
        import google.colab
        return "colab"
    except ImportError:
        pass
    return "local"


def _wait_for_server(url, timeout=30, interval=0.25):
    """Poll *url* until a connection succeeds or *timeout* seconds elapse.

    Returns True as soon as the server responds (any HTTP status counts).
    """
    import urllib.request, urllib.error
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except urllib.error.HTTPError:
            return True   # server up, error page is fine
        except Exception:
            pass
        time.sleep(interval)
    return False


def _webconsole_start_notebook(port: int, grpc_port: int):
    """ Starts a Notebook-friendly WebConsole instance

    also :
      - streams console terminal to file
      - shows 3 links table for local logging files
      - displays embedded iFrame on the homepage
 
    on Google Colab and Kaggle, uses an ngrok tunnel for external access.
 
    Params:
        - port (int):
            port to bind the web server on
        - grpc_port (int):
            port to bind to the grpc server
    """
    from . import main as _main
    from .utils.executions.events import \
        reset_for_restart as execs_reset_for_restart
    from .utils.execution.events import \
        reset_for_restart as exec_reset_for_restart
    execs_reset_for_restart()
    exec_reset_for_restart()

    log_dir = os.environ.get("RP_WEB_SERVER_LOGS", "")
    access_log_path = os.path.join(log_dir, "access.log") \
                      if log_dir else None
    error_log_path  = os.path.join(log_dir, "error.log") \
                      if log_dir else None
    live_log_path   = os.path.join(log_dir, "live_console.log") \
                      if log_dir else None



    import re
    # strip :
    #  - ANSI escapes
    #  - Rich markup tags
    #  - emoji
    # replace :
    #  - frame-drawing │ (U+2502) -> '|'
    #  - other frame-drawing (U+2500-257F) -> '-'
    _ANSI_RE = re.compile(
        r'\x1b\[[0-9;]*[mAKHF]|\[[^\[\]\']*\]' +
        r'|[\u2500-\u257f]|[\U00010000-\U0010FFFF]'
    )

    def _plain(m):
        if m.group() == '│': return '|'
        if '─' <= m.group() <= '╿': return '-'
        return ''

    class _PlainFormatter(logging.Formatter):
        def format(self, record):
            return _ANSI_RE.sub(_plain, record.getMessage())

    if live_log_path:
        os.makedirs(os.path.dirname(live_log_path), exist_ok=True)

        class _FlushingRotatingFileHandler(
            logging.handlers.TimedRotatingFileHandler
        ):
            def emit(self, record):
                super().emit(record)
                self.flush()

        _live_handler = _FlushingRotatingFileHandler(
            live_log_path, when="midnight",
            backupCount=7, encoding="utf-8"
        )
        _live_handler.setLevel(logging.DEBUG)
        _live_handler.setFormatter(_PlainFormatter())
        logging.getLogger().addHandler(_live_handler)



    # ##   ##   ##   ##   ##   DEMO   -   DELETE   ##   ##   ##   ##   ## #
    access_log_path = access_log_path.replace("jupyter_notebooks/job_hunt/AWS/fresh_start/", "")
    error_log_path  = error_log_path.replace("jupyter_notebooks/job_hunt/AWS/fresh_start/", "")
    live_log_path   = live_log_path.replace("jupyter_notebooks/job_hunt/AWS/fresh_start/", "")
    # ##   ##   ##   ##   ##   DEMO   -   DELETE   ##   ##   ##   ##   ## #



    # start server (_logger_controller.activate() runs inside here)
    _main._webconsole_start(port=port, grpc_port=grpc_port)

    # wait for uvicorn dictConfig to finish (it wipes handlers)
    time.sleep(1.5)

    _SERVER_TERMINATED_MSG = (
        "   ######   ######   ######   " +
        "server thread terminated  " +
        "   ######   ######   ######\n"
    )
    _SERVER_FAILED_MSG = (
        "   ######   ######   ######   " +
        "server failed to start  " +
        "   ######   ######   ######\n"
    )

    # poll until first heartbeat so the iframe never shows a blank page
    if not _wait_for_server(f"http://localhost:{port}/"):
        logger.error(_SERVER_FAILED_MSG)
        return

    # resolve iframe URL; start ngrok tunnel when on Colab or Kaggle
    if _notebook_env() in ('colab', 'kaggle'):
        display(HTML("⏳ spinning up ngrok tunnel, please wait..."))
        iframe_url = _ngrok_start(port)
        if iframe_url is None:
            # authentication failed
            # (feedback to user dealt with at this point already)
            _main.webconsole_shutdown()
            return
        display(HTML(
            f'<a href="{iframe_url}" target="_blank">{iframe_url}</a>')
        )
    else:
        iframe_url = f"http://localhost:{port}/"

    # poll proxy until up so the iframe never shows a blank page
    if not _wait_for_server(iframe_url):
        logger.error(_SERVER_FAILED_MSG)
        return

    clear_output(wait=True)

    # table + iframe after server is confirmed up
    logs = [
        ("access.log",       access_log_path),
        ("error.log",        error_log_path),
        ("live_console.log", live_log_path),
    ]
    rows = "".join(
        f"<tr><td>{name}</td>" + \
        "<td style=\"white-space:nowrap\">" + \
        "tail -n 50 -f \"<a href='" + \
        f"file://{os.path.abspath(p)}" + \
        f"' target='_blank'>{p}</a>\"</td></tr>"
        for name, p in logs if p
    )
    display(HTML(f"""
    <table>
      <thead><tr><th>Log</th><th>Path</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
    <iframe src="{iframe_url}" width="100%" height="620"
            style="margin-top:10px;border:1px solid #ccc;display:block;"></iframe>
    """))

    # ── NOW: deactivate rich logging, gate file handler to server thread only ──
    _main._logger_controller.deactivate()

    if live_log_path:
        class _ServerThreadFilter(logging.Filter):
            def filter(self, record):
                return threading.current_thread() is _main._server_thread

        _live_handler.addFilter(_ServerThreadFilter())

        # Thread-aware wrapper: only server thread writes go to file,
        # all other cells keep writing to real stdout/stderr
        _orig_stdout = sys.stdout
        _orig_stderr = sys.stderr

        class _ServerThreadStream:
            def __init__(self, original):
                self._original = original
            def write(self, data):
                if threading.current_thread() is _main._server_thread:
                    try:
                        _live_handler.stream.write(data)
                        _live_handler.stream.flush()
                    except (ValueError, AttributeError):
                        pass
                else:
                    self._original.write(data)
            def flush(self):
                self._original.flush()
                try:
                    _live_handler.stream.flush()
                except (ValueError, AttributeError):
                    pass
            def __getattr__(self, name):
                return getattr(self._original, name)

        sys.stdout = _ServerThreadStream(_orig_stdout)
        sys.stderr = _ServerThreadStream(_orig_stderr)

        def _write_termination_line():
            try:
                _live_handler.stream.write(_SERVER_TERMINATED_MSG)
                _live_handler.stream.flush()
            except (ValueError, AttributeError):
                pass
 
        def _cleanup_on_shutdown():
            if _main._server_thread:
                _main._server_thread.join()
            if _main._grpc_thread:
                _main._grpc_thread.join()
            _write_termination_line()
            logging.getLogger().removeHandler(_live_handler)
            _live_handler.close()
            sys.stdout = _orig_stdout
            sys.stderr = _orig_stderr

        import atexit
        atexit.register(_write_termination_line)

        threading.Thread(target=_cleanup_on_shutdown, daemon=True).start()


def _webconsole_shutdown_notebook():
    from . import main as _main

    if (
        not _main._server or
        not hasattr(_main._server, "should_exit")
    ):
        display(HTML(
            "<b>\N{warning sign} Server is not running " +
            "in this process.</b>"))
        return

    display(HTML(
        "<b>\N{octagonal sign} Shutting down server...</b>"))

    _main._server.force_exit = True
    _main._server.should_exit = True
    _main._server = None

    # signal SSE generators to exit cleanly
    # before killing the thread,
    # so their finally-blocks fire
    # and subscriber lists get cleared
    from .utils.executions.events import \
        notify_server_shutdown as execs_notify_server_shutdown
    from .utils.execution.events import \
        notify_server_shutdown as exec_notify_server_shutdown
    execs_notify_server_shutdown()
    exec_notify_server_shutdown()

    # give a chance to the thread to interrupt
    # but most likely wont due to client web-browser
    # holding open SSE conection(s)
    _main._server_thread.join(timeout=10)

    if _main._server_thread.is_alive():
        import ctypes
        ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_ulong(_main._server_thread.ident),
            ctypes.py_object(SystemExit)
        )
        _main._server_thread.join(timeout=5)

    # uvicorn's shutdown_event never fires
    # in the force-kill path above,
    # so the gRPC server must be stopped explicitly
    if _main._grpc_server:
        _main._grpc_server.stop(grace=5.0)

    if _main._grpc_thread and _main._grpc_thread.is_alive():
        _main._grpc_thread.join(timeout=5)
        if _main._grpc_thread.is_alive():
            import ctypes
            ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_ulong(_main._grpc_thread.ident),
                ctypes.py_object(SystemExit)
            )
            _main._grpc_thread.join(timeout=5)
    _main._grpc_server = None

    _main.release_server_lock()
    _main._process_has_server = False
    _main._running_port = None

    if _notebook_env() in ('colab', 'kaggle'):
        _ngrok_stop()

    _main._logger_controller.deactivate()

    display(HTML("<b>\N{white heavy check mark} Server is down.</b>"))

