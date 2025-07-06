
import os
import re
import json
import asyncio
import logging

from datetime import datetime
from typing import Optional, List
from fasthtml.common import Div, Span
from pydantic import BaseModel, validator

from ....utils import strip_ansi_escape_codes

# ---- Standard daily journalisation ----


def get_log_config():
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(asctime)s [%(levelname)s]: %(message)s",  # ← Removed %(name)s
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "access": {
                "()": "uvicorn.logging.AccessFormatter",
                "fmt": "%(asctime)s [%(levelname)s] %(client_addr)s - '%(request_line)s' %(status_code)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "file_default": {
                "class": "logging.handlers.TimedRotatingFileHandler",
                "filename": os.path.join(os.environ["RP_WEB_SERVER_LOGS"], "server.log"),
                "when": "midnight",
                "backupCount": 7,
                "formatter": "default",
                "encoding": "utf-8",
            },
            "file_access": {
                "class": "logging.handlers.TimedRotatingFileHandler",
                "filename": os.path.join(os.environ["RP_WEB_SERVER_LOGS"], "access.log"),
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
    timestamp: datetime
    level: str
    client_addr: str
    method: str
    message: str
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
            match = re.match(r'^\[([^\]]+)\](?::\d+)?$', v)
            if match:
                return match.group(1)
        else:
            # IPv4 or IPv6 without brackets
            return v.split(':')[0]
        return v  # fallback, should not happen

    def to_json(self) -> str:
        log_entry = {
            "timestamp": self.timestamp.strftime('%Y-%m-%d %H:%M:%S,%f')[:-3],  # mimic logging default
            "level": self.level,
            "client_addr": self.client_addr,
            "method": self.method,
            "message": self.message,
            "thread": self.thread,
            "threadName": self.threadName,
            "process": self.process,
            "processName": self.processName,
        }
        return json.dumps(log_entry)

    def to_fasthtml_div(self) -> Div:
        level_color = {
            'INFO': '#28a745',
            'WARNING': '#ffc107', 
            'ERROR': '#dc3545',
            'DEBUG': '#6c757d'
        }.get(self.level, "#333")

        level_icon = {
            'INFO': 'ℹ️',
            'WARNING': '⚠️',
            'ERROR': '❌',
            'DEBUG': '🔍'
        }.get(self.level, "📝")
        div = Div(
            Div(
                Span(f"{level_icon} {self.level}",
                     style=f"color: {level_color}; font-weight: bold; margin-right: 10px;"),
                Span(self.timestamp.strftime('%Y-%m-%d %H:%M:%S,%f')[:-3],
                     style="color: #666; font-size: 0.9em; margin-right: 10px;")
            ),
            Div(self.message,
                style="margin-top: 5px; padding-left: 10px; border-left: 3px solid #eee;"),
            style="background: white; padding: 12px; margin-bottom: 8px; border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); border-left: 4px solid " + level_color
        )
        return div

    @classmethod
    def from_access_log(cls, log: str):
        """
        Parse a classic access log string like:
        2025-07-06 15:41:53 [INFO] 127.0.0.1:36042 - 'POST /ui/update HTTP/1.1' 200 OK
        """
        # Regex to parse the log string
        pattern = (
            r'^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) '
            r'\[(?P<level>[A-Z]+)\] '
            r'(?P<client_addr>[^\s]+) - '
            r'\'(?P<method>[A-Z]+) (?P<path>[^\s]+) [^\'"]+\' '
            r'(?P<status>\d{3}) (?P<status_msg>.+)$'
        )
        match = re.match(pattern, log)
        if not match:
            raise ValueError("Log string does not match expected format")

        timestamp = datetime.strptime(match.group('timestamp'), '%Y-%m-%d %H:%M:%S')
        level = match.group('level')
        client_addr = match.group('client_addr')
        method = match.group('method')
        path = match.group('path')
        status = match.group('status')
        status_msg = match.group('status_msg')
        message = f"{method} {path} {status} {status_msg}"

        # Fill in missing fields with None or suitable defaults
        return cls(
            timestamp=timestamp,
            level=level,
            client_addr=client_addr,
            method=method,
            message=message,
            thread=None,
            threadName=None,
            process=None,
            processName=None
        )


def read_last_access_logs(filename: str, n: int) -> List[str]:
    """
    Read the last n access log lines from a file,
    parse each into AccessLogEntry,
    and return a list of html div elements.
    """
    log_entries = []
    with open(filename, 'rb') as f:
        f.seek(0, 2)
        filesize = f.tell()
        size = 1024
        data = b''
        while filesize > 0 and data.count(b'\n') <= n:
            read_size = min(size, filesize)
            filesize -= read_size
            f.seek(filesize)
            data = f.read(read_size) + data
        lines = data.splitlines()[-n:]
        for line in lines:
            try:
                log_line = strip_ansi_escape_codes(
                    line.decode('utf-8', errors='replace').strip()
                )
                if log_line:  # skip empty lines
                    entry = AccessLogEntry.from_access_log(log_line)
                    log_entries.append(str(entry.to_fasthtml_div()))
            except Exception as ex:
                print(ex)
                continue

    return log_entries


# ---- websocket streaming for ui ----


class WebSocketLogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.clients = set()

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
            delattr(ws_log_handler, '_handlers_added')

ws_log_handler = WebSocketLogHandler()

async def websocket_endpoint(websocket):
    await websocket.accept()

    # Only add handlers once
    if not hasattr(ws_log_handler, '_handlers_added'):
        import json
        from uvicorn.logging import AccessFormatter

        class JSONFormatter(AccessFormatter):
            def format(self, record):
                # Let AccessFormatter do its work first to populate client_addr
                formatted_msg = super().format(record)
                client_addr, method, message, http_version, status_code \
                    = record.args
                # print(record.__dict__)
                log_entry = AccessLogEntry(
                    timestamp=self.formatTime(record),
                    level=record.levelname,
                    client_addr=client_addr,
                    method=method,
                    message=message,
                    thread=record.thread,
                    threadName=record.threadName,
                    process=record.process,
                    processName=record.processName,
                )
                return log_entry #.to_json()

        ws_log_handler.setFormatter(JSONFormatter())
        logging.getLogger("uvicorn.access").addHandler(ws_log_handler)
        ws_log_handler._handlers_added = True

    ws_log_handler.register(websocket)

    try:
        while True:
            await websocket.receive_text()
    except:
        # in case e.g. more than 1 webbrowser tab is opened
        # on the page that streams this
        await ws_log_handler.broadcast_to_others(
            f"WebSocket client {websocket.client} disconnected",
            websocket
        )
    finally:
        ws_log_handler.unregister(websocket)

