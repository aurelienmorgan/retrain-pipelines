
import os

from datetime import datetime, timezone
from email.utils import formatdate, \
    parsedate_to_datetime
from fasthtml.common import Div, H1, P, Code, \
    Script, \
    Request, Response, FileResponse

from .. import APP_STATIC_DIR
from ..utils.executions import get_executions_before
from .page_template import page_layout

def register(app, rt, prefix=""):
    @rt("/favicon.ico")
    def favicon():
        favicon_fullname = os.path.join(
            APP_STATIC_DIR, "retrain-pipelines.ico")
        return FileResponse(favicon_fullname)


    @rt("/{fname:path}.{ext:static}")
    async def get(request: Request, fname:str, ext:str):
        """Serves static files, allows for webbrowser-caching."""
        file_fullname = os.path.join(APP_STATIC_DIR, f"{fname}.{ext}")
        stat = os.stat(file_fullname)
        last_modified = formatdate(stat.st_mtime, usegmt=True)
        # Check If-Modified-Since header
        if_modified_since = request.headers.get("if-modified-since")
        if if_modified_since:
            since_dt = parsedate_to_datetime(if_modified_since)
            file_dt = datetime.utcfromtimestamp(stat.st_mtime) \
                        .replace(tzinfo=timezone.utc) \
                        .replace(microsecond=0)
            if file_dt <= since_dt:
                return Response(status_code=304)
        headers = {"Last-Modified": last_modified}
        return FileResponse(file_fullname, headers=headers)


    @rt(f"{prefix}/load_executions", methods=["POST"])
    async def get_execution_entries(request):
        # Retrieve "count" from form data (POST)
        form = await request.form()
        print(form)
        before_datetime = \
            datetime.strptime(
                form.get("before_datetime")[:33],
                "%a %b %d %Y %H:%M:%S GMT%z"
            )
        print(request)
        execution_entries = await get_executions_before(
            before_datetime=before_datetime, n=50
        )

        return execution_entries


    @rt(f"{prefix}/")
    def home():
        content = (
            H1("Placeholder", style="color: white;"),
            P(Code("retrain-pipelines"), " executions!",
              style="color: white;")
        )

        return page_layout(current_page="/", title="retrain-pipelines", content=\
            Div(# page content
                Div(# Actual list
                    id="executions-container",
                    style=(
                        "max-height: 600px; overflow-y: auto; padding: 8px 16px 4px 16px; "
                        "background: linear-gradient(135deg, "
                            "rgba(255,255,255,0.05) 0%, "
                            "rgba(248,249,250,0.05) 100%); "
                        "border: 1px solid rgba(222,226,230,0.6); "
                        "border-radius: 8px; "
                        "box-shadow: inset 0 2px 4px rgba(0,0,0,0.05), "
                            "0 1px 3px rgba(0,0,0,0.1); "
                    )
                ),
                Script("""// Cold start of executions list at page load time
                    function loadExecs() {
                        const server_status_circle = document.getElementById('status-circle');
                        server_status_circle.classList.add('spinning');

                        const execContainer = document.getElementById("executions-container");
                        execContainer.innerHTML = '';

                        // form data for the html POST
                        const formData = new FormData();
                        formData.append('before_datetime', new Date());

                        fetch('/{prefix}load_executions', {
                            method: 'POST',
                            headers: { "HX-Request": "true" },
                            body: formData
                        })
                        .then(response => response.text())
                        .then(html => {
                            execContainer.insertAdjacentHTML('beforeend', html);

                            server_status_circle.classList.remove('spinning');
                        });
                    }

                    // Assign to DOMContentLoaded event
                    window.addEventListener('DOMContentLoaded', loadExecs);
                """.replace("{prefix}", prefix+"/" if prefix > "" else "")
                ),
                style=(
                    "background: rgba(248, 249, 250, 0.3); padding: 8px 16px 4px 16px; "
                    "border-radius: 12px; "
                    "box-shadow: 0 4px 12px rgba(0,0,0,0.1), "
                        "inset 0 1px 0 rgba(255,255,255,0.6); "
                    "border: 1px solid rgba(222,226,230,0.4);"
                )
            )
        )


    @rt(f"{prefix}/a_page_in_error", methods=["GET"])
    def throw_error():
        raise Exception("DEBUG");

