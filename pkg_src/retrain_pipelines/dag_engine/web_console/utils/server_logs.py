
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
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(self.broadcast(log_entry))
        else:
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
                log_entry = {
                    "timestamp": self.formatTime(record),
                    "level": record.levelname,
                    "client_addr": client_addr,
                    "method": method,
                    "message": message,
                    "thread": record.thread,
                    "threadName": record.threadName,
                    "process": record.process,
                    "processName": record.processName,
                }
                return json.dumps(log_entry)
        
        ws_log_handler.setFormatter(JSONFormatter(
            "%(asctime)s [%(levelname)s] %(client_addr)s: %(message)s"
        ))
        logging.getLogger("uvicorn.access").addHandler(ws_log_handler)
        ws_log_handler._handlers_added = True

    ws_log_handler.register(websocket)

    try:
        while True:
            await websocket.receive_text()
    except:
        # in case more than 1 webbrowser tab is opened
        # on the streaming page
        await ws_log_handler.broadcast_to_others(
            f"WebSocket client {websocket.client} disconnected",
            websocket
        )
    finally:
        ws_log_handler.unregister(websocket)


