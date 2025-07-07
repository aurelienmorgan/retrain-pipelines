
from datetime import datetime

from fasthtml.common import *

from .page_template import page_layout
from ..utils.cookies import get_ui_state, set_ui_state


def register(app, rt, prefix=""):
    @rt(f"{prefix}/web_server")
    def server_dashboard(req):
        # Get stored UI state or defaults
        lines = get_ui_state(req, "server_dashboard", "lines", "100")
        autoscroll = get_ui_state(req, "server_dashboard", "autoscroll", "true") == "true"

        return page_layout(title="WebServer Logs", content=Titled(
            "Server Logs",
            Div(
                H1("🖥️ Web Server Logs", style="color: #333; margin-bottom: 20px;"),
                Div(
                    Button("🔄 Refresh Logs", 
                           hx_get=f"{prefix}/web_server/stream_logs", 
                           hx_target="#log-container",
                           style="margin-bottom: 15px; padding: 8px 16px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer;"),
                    Div(
                        P("📊 Server Status: ", 
                          Span("🟢 Running", style="color: green; font-weight: bold;"),
                          style="margin-bottom: 10px;"),
                        P(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                          style="margin-bottom: 10px;"),
                        P(f"📈 Total Logs: {0}", 
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
                    Div(id="log-container", style="max-height: 600px; overflow-y: auto;"),
                    style="background: #f8f9fa; padding: 15px; border-radius: 8px;"
                )
            ),
            Script("""
                /* *********************************************
                * append log item on WebSocket message receive *
                ********************************************* */
                const logContainer = document.getElementById("log-container");

                let ws;

                function connectWebSocket() {
                  ws = new WebSocket(`ws://${location.host}/{prefix}ws/stream_logs`);

                  ws.onmessage = (event) => {
                    const message = document.createElement("div");
                    message.innerHTML = event.data;
                    logContainer.appendChild(message);
                    var autoScrollCheckbox = document.getElementById('autoscroll');
                    if (logContainer && autoScrollCheckbox && autoScrollCheckbox.checked) {
                        logContainer.scrollTop = logContainer.scrollHeight;
                    }
                  };

                  ws.onopen = () => {
                    console.log("WebSocket connected.");
                  };

                  ws.onclose = () => {
                    console.log("WebSocket disconnected.");
                  };

                  ws.onerror = (err) => {
                    console.error("WebSocket error:", err);
                  };
                }

                // Initial connection
                connectWebSocket();

                // Close WebSocket when tab/window is closed
                window.addEventListener('beforeunload', () => {
                  setTimeout(() => ws.close(), 1000);
                });

                // // Handle visibility changes - close when hidden, reconnect when visible
                // document.addEventListener('visibilitychange', () => {
                //  if (document.hidden) {
                //    console.log("WebSocket closing on hidden event.");
                //    setTimeout(() => ws.close(), 1000);
                //  } else {
                //    if (ws.readyState === WebSocket.CLOSED) {
                //      console.log("WebSocket connecting on unhidden event.");
                //      connectWebSocket();
                //    }
                //  }
                // });
            """.replace("{prefix}", prefix+"/" if prefix > "" else "")
            ),
            Script("""
                /* ******************************************
                * Cold start of logs list at page load time *
                ****************************************** */
                function loadLogs() {
                    // Get how many log-entries to return from selection list
                    const linesSelect = document.getElementById('lines');
                    const count = linesSelect ? parseInt(linesSelect.value, 10) : 0;

                    // form data for the html POST
                    const formData = new FormData();
                    formData.append('count', count);

                    fetch('/{prefix}web_server/load_logs', {
                        method: 'POST',
                        headers: { "HX-Request": "true" },
                        body: formData
                    })
                    .then(response => response.text())
                    .then(html => {
                        const logContainer = document.getElementById("log-container");
                        logContainer.innerHTML = html;
                        var autoScrollCheckbox = document.getElementById('autoscroll');
                        if (logContainer && autoScrollCheckbox && autoScrollCheckbox.checked) {
                            logContainer.scrollTop = logContainer.scrollHeight;
                        }
                    });
                }

                // Assign to DOMContentLoaded event
                window.addEventListener('DOMContentLoaded', loadLogs);

                // Assign to selection list change event
                document.addEventListener('DOMContentLoaded', function() {
                    const linesSelect = document.getElementById('lines');
                    if (linesSelect) {
                        linesSelect.addEventListener('change', function() {
                            // Clear log-container content
                            const logContainer = document.getElementById('log-container');
                            if (logContainer) {
                                logContainer.innerHTML = '';
                                // Once cleared, call loadLogs
                                // (Since .innerHTML = '' is synchronous, we can call immediately)
                                loadLogs();
                            }
                        });
                    }
                });
            """.replace("{prefix}", prefix+"/" if prefix > "" else "")
            )
        ))


    @rt(f"{prefix}/web_server/status", methods=["HEAD"])
    def web_server_status():
        # TODO, will even probably move
        return "web server status"





























































