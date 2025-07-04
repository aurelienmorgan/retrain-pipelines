
from datetime import datetime

from fasthtml.common import *

from ..utils.cookies import get_ui_state, set_ui_state


#     !! TODO  -  DELETE !!    #
from collections import deque
_server_logs = deque(maxlen=100)  
################################


def register(app, rt, prefix=""):
    @rt(f"{prefix}/web_server")
    def server_dashboard(req):
        # Get stored UI state or defaults
        lines = get_ui_state(req, "server_dashboard", "lines", "100")
        autoscroll = get_ui_state(req, "server_dashboard", "autoscroll", "true") == "true"

        return Titled(
            "Server Logs",
            Div(
                H1("🖥️ Web Server Logs", style="color: #333; margin-bottom: 20px;"),
                Div(
                    Button("🔄 Refresh Logs", 
                           hx_get="/web_server/logs", 
                           hx_target="#log-container",
                           style="margin-bottom: 15px; padding: 8px 16px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer;"),
                    Div(
                        P("📊 Server Status: ", 
                          Span("🟢 Running", style="color: green; font-weight: bold;"),
                          style="margin-bottom: 10px;"),
                        P(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                          style="margin-bottom: 10px;"),
                        P(f"📈 Total Logs: {len(_server_logs)}", 
                          style="margin-bottom: 20px;"),
                        style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 20px;"
                    )
                ),
                Div(
                    Div(
                        Div(
                            H3("📋 Recent Logs", style="color: #333; margin: 0; white-space: nowrap; font-size: 16px; line-height: 1.5;"),
                            style="display: flex; align-items: baseline;"
                        ),
                        Div(
                            Span("show last", style="margin-right: 5px; white-space: nowrap; font-size: 14px; align-self: baseline;"),
                            Select(
                                *[Option(str(v), selected=(str(v) == lines)) for v in [50, 100, 1000]],
                                value="100",
                                id="lines",
                                style="""
                                    margin-right: 15px;
                                    font-size: 13px;
                                    line-height: 1.2;
                                    padding: 2px 24px 2px 4px;
                                    width: fit-content;
                                    white-space: nowrap;
                                    box-sizing: content-box;
                                """
                            ),
                            Input(
                                type="checkbox",
                                checked=autoscroll,
                                id="autoscroll",
                                style="margin: 0 5px 0 0; align-self: center; transform: translateY(-7px);"
                            ),
                            Span("autoscroll", style="white-space: nowrap; font-size: 14px; align-self: baseline;"),
                            style="display: flex; align-items: baseline;"
                        ),
                        Script("""
                            function updateUI(view, key, value) {
                                fetch('/ui/update', {
                                    method: 'POST',
                                    headers: {'Content-Type': 'application/json'},
                                    body: JSON.stringify({ view, key, value })
                                });
                            }

                            document.getElementById("autoscroll").addEventListener("change", e => {
                                updateUI("server_dashboard", "autoscroll", e.target.checked);
                            });

                            document.getElementById("lines").addEventListener("change", e => {
                                updateUI("server_dashboard", "lines", e.target.value);
                            });
                        """),
                        style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 15px;"
                    ),
                    Div(id="log-container", hx_get="/web_server/logs", hx_trigger="load, every 5s"),
                    style="background: #f8f9fa; padding: 15px; border-radius: 8px;"
                )
            ),
            Script("""
            document.body.addEventListener('htmx:afterSettle', function(evt) {
                var logContainer = document.getElementById('log-items');
                if (logContainer) {
                    logContainer.scrollTop = logContainer.scrollHeight;
                }
            });
            """)
        )


    @rt(f"{prefix}/web_server/logs")
    def get_logs():
        if not _server_logs:
            return Div(
                P("📝 No logs available yet...", 
                  style="text-align: center; color: #666; padding: 20px;")
            )
        
        log_items = []
        for log in list(_server_logs):  # Show newest last
            level_color = {
                'INFO': '#28a745',
                'WARNING': '#ffc107', 
                'ERROR': '#dc3545',
                'DEBUG': '#6c757d'
            }.get(log['level'], '#333')
            
            level_icon = {
                'INFO': 'ℹ️',
                'WARNING': '⚠️',
                'ERROR': '❌',
                'DEBUG': '🔍'
            }.get(log['level'], '📝')
            
            log_items.append(
                Div(
                    Div(
                        Span(f"{level_icon} {log['level']}", 
                             style=f"color: {level_color}; font-weight: bold; margin-right: 10px;"),
                        Span(log['timestamp'], 
                             style="color: #666; font-size: 0.9em; margin-right: 10px;"),
                        Span(f"[{log['module']}]", 
                             style="color: #007bff; font-size: 0.8em; margin-right: 10px;")
                    ),
                    Div(log['message'], 
                        style="margin-top: 5px; padding-left: 10px; border-left: 3px solid #eee;"),
                    style="background: white; padding: 12px; margin-bottom: 8px; border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); border-left: 4px solid " + level_color
                )
            )
        
        return Div(
            Div(*log_items, id="log-items", style="max-height: 600px; overflow-y: auto;")
        )

    @rt(f"{prefix}/web_server/status")
    def web_server_status():
        # TODO, will even probably move
        return "web server status"





























































