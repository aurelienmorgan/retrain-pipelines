
import os
import asyncio

from typing import Optional, List
from fasthtml.common import H1, H2, Div, P, \
    Link, Script, Style, \
    Request, Response, JSONResponse
from jinja2 import Environment, FileSystemLoader

from ...db.dao import AsyncDAO
from .page_template import page_layout
from ....utils import get_text_pixel_width


async def get_execution_elements_lists(
    execution_id: int
) -> Optional[List[str]]:
    """Can be None, e.g. if no execution with that id exists."""
    dao = AsyncDAO(
        db_url=os.environ["RP_METADATASTORE_ASYNC_URL"]
    )

    execution_tasktypes_list, execution_taskgroups_list = await asyncio.gather(
        dao.get_execution_tasktypes_list(execution_id, serializable=True),
        dao.get_execution_taskgroups_list(execution_id, serializable=True)
    )
    # print(f"execution_tasktypes_list : {execution_tasktypes_list}")
    # print(f"execution_taskgroups_list : {execution_taskgroups_list}")

    return execution_tasktypes_list, execution_taskgroups_list


def register(app, rt, prefix=""):
    @rt(f"{prefix}/dag_rendering", methods=["GET"])
    async def dag_rendering(request: Request):
        execution_id = request.query_params.get("id")
        try:
            execution_id = int(execution_id)
        except (TypeError, ValueError):
            return Div(P(f"Invalid execution ID {execution_id}"))

        tasktypes_list, taskgroups_list = \
            await get_execution_elements_lists(execution_id)
        if tasktypes_list is None:
            return Div(P(f"Invalid execution ID {execution_id}"))

        template_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "utils", "execution")
        env = Environment(loader=FileSystemLoader(template_dir))
        env.globals['get_text_pixel_width'] = get_text_pixel_width
        template = env.get_template("svg_template.html")
        rendering_content = template.render(
            nodes=tasktypes_list,
            taskgroups=taskgroups_list or []
        )

        return rendering_content


    @rt(f"{prefix}/execution_details", methods=["GET"])
    async def execution_details(request: Request):
        execution_id = request.query_params.get("id")
        try:
            execution_id = int(execution_id)
        except (TypeError, ValueError):
            return Response(
                f"Invalid execution ID {execution_id}", 500)


        return JSONResponse({})


    @rt(f"{prefix}/execution_number", methods=["GET"])
    async def execution_number(request: Request):
        execution_id = request.query_params.get("id")
        try:
            execution_id = int(execution_id)
        except (TypeError, ValueError):
            return Response(
                f"Invalid execution ID {execution_id}", 500)


        return Response(str(0))


    @rt(f"{prefix}/execution", methods=["GET"])
    def home(request: Request):
        execution_id = request.query_params.get("id")
        try:
            execution_id = int(execution_id)
        except (TypeError, ValueError):
            return page_layout(
                title="retrain-pipelines",
                content=""
            )

        return page_layout(title="retrain-pipelines", \
            content=Div(# page content
                H1(# title
                    style="padding-left: 30px;",
                    cls="shiny-gold-text",
                    id="title"
                ),
                H2(# subtitle
                    Div(
                        "flow run # ",
                        Div(
                            id="execution-number"
                        ),
                        f", run_id: {execution_id} -\u00A0",
                        Div(
                            id="utc-start-date-time-str"
                        ),
                        style=(
                            "display: flex; align-items: center; "
                            "flex-wrap: nowrap; white-space: nowrap;"
                        )
                    ),
                    style=(
                        "background-color: #FFFFCC40; padding-left: 3px; "
                        "text-align: left; color: #FFEA66;"
                    ),
                    id="subtitle"
                ),
                ## TODO, add stuff here
                Script(f"""// async get execution-details
                    (function(){{
                        document.addEventListener("DOMContentLoaded", function() {{
                            // title & start-date-time
                            fetch("{prefix}/execution_details?id={execution_id}", {{
                                method: 'GET',
                                headers: {{ "HX-Request": "true" }}
                            }})
                            .then(function(resp) {{
                                if (!resp.ok) {{
                                  return resp.text().then(text => {{
                                    throw new Error(text || resp.statusText || 'Unknown error');
                                  }});
                                }}
                                return resp.json();
                            }})
                            .then(function(execution_details) {{
                                // inform title & subtitle
                                document.getElementById("title").innerText = execution_details.title;
                                // (flow run # 4, run_id: 101 - Friday Apr 11 2025 11:57:29 PM UTC)
                                document.getElementById("utc-start-date-time-str").innerText =
                                    execution_details.start_timestamp;

                            }}).catch(error => {{
                                console.error("Error fetching {prefix}/execution_details:", error);
                            }});

                            // execution-number
                            fetch("{prefix}/execution_number?id={execution_id}", {{
                                method: 'GET',
                                headers: {{ "HX-Request": "true" }}
                            }})
                            .then(function(resp) {{
                                return resp.text();
                            }})
                            .then(function(executionNumber) {{
                                document.getElementById("execution-number").innerText = executionNumber;

                            }}).catch(error => {{
                                console.error("Error fetching {prefix}/execution_details:", error);
                            }});

                        }});
                    }})();
                """),
                Style("""
.shiny-gold-text {
    font-size: 3em;
    font-weight: bold;
    color: #FFD700; /* Base gold color */
    text-shadow: 
        0 0 5px rgba(255, 215, 0, 0.7),  /* Inner glow */
        0 0 10px rgba(255, 215, 0, 0.5), /* Outer glow */
        0 0 15px rgba(255, 215, 0, 0.3); /* Soft outer glow */
    background: linear-gradient(45deg, #FFD700, #FFEA66);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
                """),
                Div(# DAG renderer
                    P("Loading DAG..."),
                    hx_get=f"{prefix}/dag_rendering?id={execution_id}",
                    hx_trigger="load",
                    hx_swap="outerHTML"
                ),
                Link(
                    rel="stylesheet",
                    href=f"{prefix}/svg_dag.css"
                )
            )
        )

