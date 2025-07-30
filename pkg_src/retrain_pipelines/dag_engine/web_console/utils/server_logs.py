
import os
import glob
import json
import regex
import asyncio
import logging
import tzlocal

from http import HTTPStatus
from datetime import datetime
from typing import Optional, List
from fasthtml.common import Div, Span
from pydantic import BaseModel, validator
from uvicorn.logging import AccessFormatter

from ....utils import strip_ansi_escape_codes


server_tz = tzlocal.get_localzone()


# ---- Standard daily journalisation ----


def get_log_config():
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                # Removed %(name)s below
                "fmt": "%(asctime)s [%(levelname)s]: %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "access": {
                "()": "uvicorn.logging.AccessFormatter",
                "fmt": (
                    "%(asctime)s [%(levelname)s] %(client_addr)s - "
                    "'%(request_line)s' %(status_code)s"
                ),
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "file_default": {
                "class": "logging.handlers.TimedRotatingFileHandler",
                "filename": os.path.join(
                    os.environ["RP_WEB_SERVER_LOGS"],
                    "server.log"
                ),
                "when": "midnight",
                "backupCount": 7,
                "formatter": "default",
                "encoding": "utf-8",
            },
            "file_access": {
                "class": "logging.handlers.TimedRotatingFileHandler",
                "filename": os.path.join(
                    os.environ["RP_WEB_SERVER_LOGS"],
                    "access.log"
                ),
                "when": "midnight",
                "backupCount": 7,
                "formatter": "access",
                "encoding": "utf-8",
            },
        },
        "loggers": {
            "uvicorn": {
                "handlers": ["file_default"],
                "level": "INFO",
                "propagate": False,
            },
            "uvicorn.error": {
                "handlers": ["file_default"],
                "level": "INFO",
                "propagate": False,
            },
            "uvicorn.access": {
                "handlers": ["file_access"],
                "level": "INFO",
                "propagate": False,
            },
        },
    }


# ---- initial state for ui ----


class AccessLogEntry(BaseModel):
    raw_str: str
    timestamp: datetime
    level: str
    client_addr: str
    method: str
    message: str
    status_code: int
    thread: Optional[int] = None
    threadName: Optional[str] = None
    process: Optional[int] = None
    processName: Optional[str] = None

    @validator('client_addr', pre=True)
    def extract_ip(cls, v):
        # Handles IPv4 and IPv6, with or without port
        # IPv6 addresses may be in [::1]:12345 format
        # IPv4: 127.0.0.1:12345 or just 127.0.0.1
        if v.startswith('['):  # IPv6 with port, e.g. [::1]:12345
            match = regex.match(r'^\[([^\]]+)\](?::\d+)?$', v)
            if match:
                return match.group(1)
        else:
            # IPv4 or IPv6 without brackets
            return v.split(':')[0]
        return v  # fallback, should not happen

    def to_json(self) -> str:
        log_entry = {
            "timestamp": self.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            "level": self.level,
            "client_addr": self.client_addr,
            "method": self.method,
            "status_code": self.status_code,
            "message": self.message,
            "thread": self.thread,
            "threadName": self.threadName,
            "process": self.process,
            "processName": self.processName,
        }
        return json.dumps(log_entry)

    def to_fasthtml_div(self) -> Div:
        """
        Results:
            - (Div):
                the DOM element
                representing that log entry
                in the WebServer Logs page list.
        """
        status_color = (
           # #28a745
           (40, 167, 69) if 200 <= self.status_code < 300 else
           # #ffc107
           (255, 193, 7) if 400 <= self.status_code < 500 else
           # #dc3545
           (220, 53, 69) if 500 <= self.status_code < 600 else
           # #333, HTTP 3xx (not-modified, redirects, etc.)
           (51, 51, 51)
        )
        def _rgb_to_rgba(rgb_tuple, alpha=1):
            color_str = (
                f"rgba({rgb_tuple[0]},{rgb_tuple[1]},{rgb_tuple[2]},{alpha})"
            )
            return color_str

        method_icon = {
            'GET':     "\N{DOWNWARDS BLACK ARROW}", # fetching data
            'POST':    "\N{UPWARDS BLACK ARROW}",   # sending data
            'PUT':     "\N{PENCIL}",                # updating data
            'DELETE':  "\N{WASTEBASKET}",           # deleting data
            'PATCH':   "\N{ADHESIVE BANDAGE}",      # patching data
            'HEAD':    "\N{BRAIN}",                 # "head"
            'OPTIONS': "\N{GEAR}",                  # options/config
            # websocket, persistent, bidirectional connection
            'ws': "\N{ELECTRIC PLUG}",
            # server-side event, one-way, server-to-client streaming
            'sse':       "\N{SATELLITE ANTENNA}",
        }.get(self.method, "\N{TWISTED RIGHTWARDS ARROWS}")

        timestamp_local = self.timestamp.replace(tzinfo=server_tz)

        div = Div(
            Div(
                # Glass shine overlay
                style=(
                    "position: absolute; top: 0; left: 0; right: 0; "
                    "height: 40%; "
                    "background: linear-gradient(135deg, "
                        "rgba(255,255,255,0.3) 0%, "
                        "rgba(255,255,255,0.1) 50%, transparent 100%); "
                    "pointer-events: none; border-radius: 6px 6px 0 0; "
                    "transition: opacity 0.2s ease;"
                ),
                id="glass-overlay"
            ),

            Div(
                Span(
                    method_icon,
                    style=(
                        "min-width: 22px; display: inline-block; "
                        "text-align: center;"
                    )
                ),
                Span(
                    f"\u00A0{self.method}",
                    style=(
                        f"color: {_rgb_to_rgba(status_color)}; "
                        "font-weight: bold; "
                        "margin-right: 10px; "
                        "min-width: 70px; "
                        "text-shadow: 0 0 12px #FFD700, 0 0 2px #FFD700;"
                    )
                ),
                Span(
                    (
                        timestamp_local.strftime('%Y-%m-%d %H:%M:%S') +
                        " " + timestamp_local.strftime('%Z')
                    ),
                    style=(
                        "color: #ccc; font-size: 0.9em; "
                        "margin-right: 10px; min-width: 170px;"
                    )
                ),
                style=(
                    "display: flex; align-items: baseline; "
                    "position: relative; z-index: 1;"
                )
            ),
            Div(
                Span(
                    self.client_addr,
                    style=(
                        "padding-left: 10px; margin-right: 10px; "
                        "border-left: 3px solid #eee; "
                        "min-width: 120px; display: inline-block; "
                        "text-align: center; vertical-align: middle;"
                    )
                ),
               Span(self.message,
                    style=(
                        "padding-left: 10px; border-left: 3px solid #eee; "
                        "display: inline-block; vertical-align: middle;"
                    )
               ),
               style="position: relative; z-index: 1;"
            ),
            style=(
                f"--status-color-normal: {_rgb_to_rgba(status_color, .45)}; "
                f"--status-color-hover: {_rgb_to_rgba(status_color, .65)}; "
                f"background: var(--status-color-normal); "
                "padding-top: 1px; padding-bottom: 1px; padding-left: 12px; "
                "padding-right: 12px; margin-bottom: 4px; "
                "border-radius: 6px; "
                "box-shadow: 0 2px 4px rgba(0,0,0,0.1), "
                    "0 8px 16px rgba(0,0,0,0.05), "
                    "inset 0 1px 0 rgba(255,255,255,0.4), "
                    "inset 0 -1px 0 rgba(0,0,0,0.1); "
                f"border-left: 4px solid {_rgb_to_rgba(status_color)}; "
                "display: flex; align-items: flex-start; "
                "backdrop-filter: blur(10px); "
                "-webkit-backdrop-filter: blur(10px); position: relative; "
                "overflow: hidden; "
                "transition: transform 0.2s ease, margin 0.2s ease; "
                "transform-origin: center center; "
            ),
            raw_str=self.raw_str,
            cls="log-entry",
            title="WebSocket" if self.method == "ws" else \
                  "Server-Side Event" if self.method == "sse" else \
                  f"{self.status_code} - {HTTPStatus(self.status_code).phrase}"
        )

        return div

    @classmethod
    def from_access_log(cls, log: str):
        """
        Parse a classic access log string like:
        2025-07-06 15:41:53 [INFO] 127.0.0.1:36042 - 'POST /ui/update HTTP/1.1' 200 OK
        """
        # Regex to parse the log string
        pattern = regex.compile(
            r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) "
            r"\[(?P<level>[A-Z]+)\] "
            r"(?P<client_addr>\d{1,3}(?:\.\d{1,3}){3}:\d+) - "
            r"'(?P<method>[A-Za-z]+) (?P<message>.+?) HTTP/[\d\.]+' "
            r"(?P<status_code>\d{3})"
        )
        match = regex.match(pattern, log)
        if not match:
            raise ValueError(
                f"Log string `{log}` does not match expected format"
            )

        timestamp = datetime.strptime(
            match.group('timestamp'),
            '%Y-%m-%d %H:%M:%S'
        )
        level = match.group('level')
        client_addr = match.group('client_addr')
        method = match.group('method')
        status_code = match.group('status_code')
        message = match.group('message')

        # Fill in missing fields with None or suitable defaults
        return cls(
            raw_str=log,
            timestamp=timestamp,
            level=level,
            client_addr=client_addr,
            method=method,
            status_code=status_code,
            message=message,
            thread=None,
            threadName=None,
            process=None,
            processName=None
        )


def _read_last_n_lines(
    filename: str,
    n: int,
    regex_filter: str | None
) -> list[bytes]:
    matches = []
    pattern = regex.compile(regex_filter) if regex_filter else None

    try:
        with open('/proc/version', 'r') as f:
            is_wsl = 'microsoft' in f.read().lower()
    except Exception:
        is_wsl = False

    if is_wsl and filename.startswith('/mnt/'):
        import subprocess

        win_path = regex.sub(r'^/mnt/([a-z])/', r'\1:\\', filename).replace('/', '\\')
        # Read backwards in batches until enough matches
        total_lines = 0
        batch = 4096
        while True:
            line_count = n + total_lines
            result = subprocess.run([
                'powershell.exe', '-Command',
                f'Get-Content "{win_path}" | Select-Object -Last {line_count}'
            ], capture_output=True, text=True)
            if result.returncode != 0:
                break
            lines = result.stdout.strip().split('\n')
            lines = [line for line in lines if line.strip()]
            if regex_filter:
                filtered = [line for line in reversed(lines) if pattern.match(line)]
                if len(filtered) >= n:
                    return [l.encode('utf-8') for l in reversed(filtered[:n])]
                # Not enough matches, read more lines next time
            else:
                return [line.encode('utf-8') for line in lines[-n:]]
            if len(lines) < line_count:
                # Reached start of file
                return [l.encode('utf-8') for l in reversed(filtered)]
            total_lines += batch

    fd = os.open(filename, os.O_RDONLY)
    try:
        with os.fdopen(fd, 'rb') as f:
            f.seek(0, os.SEEK_END)
            filesize = f.tell()
            blocksize = 4096
            buffer = b''
            while filesize > 0 and len(matches) < n:
                read_size = min(blocksize, filesize)
                filesize -= read_size
                f.seek(filesize)
                buffer = f.read(read_size) + buffer
                lines = buffer.splitlines()
                # Only scan new lines added in this block
                for line in reversed(lines):
                    if regex_filter:
                        if pattern.match(line.decode('utf-8', 'replace')):
                            matches.append(line)
                            if len(matches) == n:
                                return list(reversed(matches))
                    else:
                        matches.append(line)
                        if len(matches) == n:
                            return list(reversed(matches))
                buffer = b'\n'.join(lines)
            return list(reversed(matches))
    finally:
        if fd >= 0:
            try:
                os.close(fd)
            except Exception:
                pass


def read_last_access_logs(
    log_dir: str,
    base_filename: str,
    n: int,
    regex_filter: str | None
) -> List[str]:
    """
    Read the last n access log lines
    from current and rotated log files,
    parse each into AccessLogEntry,
    and return a list of html div elements.
    """
    # Find all log files
    log_files = glob.glob(f"{log_dir}/{base_filename}*")
    log_files.sort(key=os.path.getmtime, reverse=True)

    lines = []
    lines_needed = n

    lines = []
    lines_needed = n
    for log_file in log_files:
        file_lines = _read_last_n_lines(log_file, lines_needed, regex_filter)
        lines = file_lines + lines
        lines_needed = n - len(lines)
        if lines_needed <= 0:
            break

    last_lines = lines[-n:]
    log_entries = []
    for line in last_lines:
        try:
            log_line = strip_ansi_escape_codes(
                line.decode('utf-8', errors='replace').strip()
            )
            if log_line:
                entry = AccessLogEntry.from_access_log(log_line)
                log_entries.append(str(entry.to_fasthtml_div()))
        except Exception as ex:
            print(ex)
            continue

    return log_entries


# ---- websocket streaming for ui ----


class WebSocketLogHandler(logging.Handler):
    def __init__(self, socket_endpoint_route: str):
        super().__init__()
        self.clients = set()
        self.socket_endpoint_route = socket_endpoint_route

    def emit(self, record):
        log_entry = self.format(record)
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(self.broadcast(log_entry))
        else:
            asyncio.run(self.broadcast(log_entry))

    async def broadcast(self, access_log_entry: AccessLogEntry):
        div = access_log_entry.to_fasthtml_div()

        dead = []
        for ws in self.clients:
            try:
                await ws.send_text(str(div))

                if self.socket_endpoint_route != access_log_entry.message:
                    # to avoid infinite loop, we only broadcast on
                    # "not the websocket logging itslef"
                    logger = logging.getLogger("uvicorn.access")
                    client_addr = f"{ws.client[0]}:{ws.client[1]}"
                    method = "ws"
                    path = self.socket_endpoint_route
                    http_version = "0.0"
                    status_code = 200
                    logger.info(
                        '%s - "%s %s %s" %d',
                        client_addr,
                        method,
                        path,
                        http_version,
                        status_code
                    )
            except Exception as ex:
                print(ex)
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)

    def register(self, websocket):
        self.clients.add(websocket)

    def unregister(self, websocket):
        self.clients.discard(websocket)

    async def broadcast_to_others(self, message, exclude_ws):
        dead = []
        for ws in self.clients:
            if ws != exclude_ws:
                try:
                    await ws.send_text(message)
                except Exception:
                    dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)
        if len(self.clients) == 1:
            # if only "exclude_ws" remains (the one being shut-down)
            delattr(self, '_handlers_added')

def get_log_websocket_endpoint(
    route: str
):
    """
    Results:
        - the async websocket_endpoint method
    """

    ws_log_handler = WebSocketLogHandler(route)

    async def websocket_endpoint(websocket):
        await websocket.accept()

        # Only add handlers once
        if not hasattr(ws_log_handler, '_handlers_added'):
            class AccessLogEntryFormatter(AccessFormatter):
                def format(self, record):
                    # Let AccessFormatter do its work first to populate client_addr
                    formatted_msg = super().format(record)
                    client_addr, method, message, http_version, status_code \
                        = record.args
                    time_str = self.formatTime(record).split(",")[0]
                    log_entry = AccessLogEntry(
                        raw_str=(
                            time_str +
                            f" [{record.levelname}] " +
                            formatted_msg.replace('"', "'") +
                            " " +
                            HTTPStatus(status_code).phrase
                        ),
                        timestamp=time_str,
                        level=record.levelname,
                        client_addr=client_addr,
                        method=method,
                        status_code=status_code,
                        message=message,
                        thread=record.thread,
                        threadName=record.threadName,
                        process=record.process,
                        processName=record.processName,
                    )
                    return log_entry

            ws_log_handler.setFormatter(AccessLogEntryFormatter())
            logging.getLogger("uvicorn.access").addHandler(ws_log_handler)
            ws_log_handler._handlers_added = True

        ws_log_handler.register(websocket)

        try:
            while True:
                await websocket.receive_text()
        except:
            # in case e.g. more than 1 client
            # or 1 webbrowser tab is opened at once
            # on the page that streams this
            await ws_log_handler.broadcast_to_others(
                f"WebSocket client {websocket.client} disconnected",
                websocket
            )
        finally:
            ws_log_handler.unregister(websocket)

    return websocket_endpoint

