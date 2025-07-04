
from fasthtml.common import Response, WebSocket

from ..utils.cookies import set_ui_state
# from ..utils.server_logs import WebSocketLogHandler


def register(app, rt, prefix=""):
    @rt(f"{prefix}/ui/update", methods=["POST"])
    async def update_ui(req):
        data = await req.json()
        view = data.get("view")
        key = data.get("key")
        value = str(data.get("value")).lower()

        if view and key:
            headers = {"Content-Type": "text/plain"}
            resp = Response("", headers=headers)
            set_ui_state(resp, view, key, value)
            return resp

        headers = {"Content-Type": "text/plain"}
        return Response("Missing key or view", status=400, headers=headers)


    @app.ws(f"{prefix}/ws/logs")
    async def log_socket(websocket: WebSocket):
        await websocket.accept()

        ws_log_handler = WebSocketLogHandler()
        ws_log_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        ))
        logging.getLogger().addHandler(ws_log_handler)

        ws_log_handler.register(websocket)

        try:
            while True:
                await websocket.receive_text()  # Keep connection open (you can ignore input)
        except Exception:
            pass
        finally:
            ws_log_handler.unregister(websocket)

