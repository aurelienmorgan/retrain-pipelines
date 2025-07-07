
import os

from fasthtml.common import Response, WebSocket, Div

from ..utils.cookies import set_ui_state
from ..utils.server_logs import WebSocketLogHandler, \
    read_last_access_logs


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
        return Response("Missing key or view",
                        status=400, headers=headers)


    @rt(f"{prefix}/web_server/load_logs", methods=["POST"])
    async def get_log_entries(request):
        # Retrieve "count" from form data (POST)
        form = await request.form()
        count = form.get("count", 50)
        try:
            count = int(count)
        except (ValueError, TypeError):
            count = 50  # if conversion fails

        log_entries = read_last_access_logs(
            os.path.join(os.environ["RP_WEB_SERVER_LOGS"],
                         "access.log"),
            count
        )
        return log_entries


    @app.ws(f"{prefix}/ws/test")
    async def on_message(message: str, send):
        # print(f"msg type: {type(message)}, msg: {message}")
        await send(Div('Hello ' + message, id='notifications'))
        await send(Div('Goodbye ' + message, id='notifications'))

