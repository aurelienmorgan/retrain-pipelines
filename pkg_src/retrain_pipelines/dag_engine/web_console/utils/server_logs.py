
import os
import asyncio
import logging


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


# ---- websocket streaming for ui ----


class WebSocketLogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.clients = set()

    def emit(self, record):
        log_entry = self.format(record)
        asyncio.run(self.broadcast(log_entry))

    async def broadcast(self, message):
        dead = []
        for ws in self.clients:
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)

    def register(self, websocket):
        self.clients.add(websocket)

    def unregister(self, websocket):
        self.clients.discard(websocket)

