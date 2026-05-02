
import os
import logging
import asyncio

from typing import List, Tuple
from fasthtml.common import H1, H2, Div, P, \
    Span, A, Link, Script, Style, Button, \
    Table, Colgroup, Col, Thead, Tr, Th, Tbody, \
    Request, Response, JSONResponse, \
    StreamingResponse, HTMLResponse, HTTPException
from jinja2 import Environment, FileSystemLoader


from ...db.dao import AsyncDAO
from ...db.model import TaskExt, TaskGroup

from ..utils import ClientInfo
from ..utils.execution.events import \
    multiplexed_event_generator, execution_number
from ..utils.execution.gantt_chart import draw_chart

from .page_template import page_layout
from ....utils import get_text_pixel_width


async def get_execution_dag_elements_lists(
    execution_id: int
) -> Tuple[List[dict], List[dict]]:
    """Tuple of topologically-sorted serializable lists

    of constituting elements, respectively :
        - all TaskTypes (exhaustively)
        - and all TaskGroups.

    Can be None, e.g. if no execution with that id exists.

    Params:
        - execution_id (int)

    Results:
        - (List[TaskTypes])
        - (List[TaskGroups])
    """
    dao = AsyncDAO(
        db_url=os.environ["RP_METADATASTORE_ASYNC_URL"]
    )

    execution_tasktypes_list, execution_taskgroups_list = await asyncio.gather(
        dao.get_execution_tasktypes_list(execution_id),
        dao.get_execution_taskgroups_list(execution_id)
    )
    # print(f"execution_tasktypes_list : {execution_tasktypes_list}")
    # print(f"execution_taskgroups_list : {execution_taskgroups_list}")
    if not execution_tasktypes_list:
        return None, None


    # turn into serializables
    execution_tasktypes_list = \
        [tasktypes.__dict__ for tasktypes in execution_tasktypes_list]
    for tasktype_dict in execution_tasktypes_list:
        tasktype_dict["uuid"] = str(tasktype_dict["uuid"])
        tasktype_dict["taskgroup_uuid"] = str(tasktype_dict["taskgroup_uuid"]) \
                                          if tasktype_dict["taskgroup_uuid"] else ""

    execution_taskgroups_list = \
        [taskgroup.__dict__ for taskgroup in execution_taskgroups_list] \
        if execution_taskgroups_list else []
    for taskgroup_dict in execution_taskgroups_list:
        taskgroup_dict["uuid"] = str(taskgroup_dict["uuid"])

    return execution_tasktypes_list, execution_taskgroups_list


async def get_execution_elements_lists(
    execution_id: int
) -> Tuple[List[TaskExt], List[TaskGroup]]:
    """Tuple of topologically-sorted lists of model objects.

    of constituting elements, respectively :
        - all Tasks (exhaustively)
        - and all TaskGroups.

    Can be None, e.g. if no execution with that id exists.

    Params:
        - execution_id (int)

    Results:
        - (List[TaskExt])
        - (List[TaskGroup])
    """
    dao = AsyncDAO(
        db_url=os.environ["RP_METADATASTORE_ASYNC_URL"]
    )

    execution_tasks_list, execution_taskgroups_list = await asyncio.gather(
        dao.get_execution_tasks_list(execution_id),
        dao.get_execution_taskgroups_list(execution_id)
    )
    # print(f"execution_tasks_list : {execution_tasks_list}")
    # print(f"execution_taskgroups_list : {execution_taskgroups_list}")

    return execution_tasks_list, execution_taskgroups_list


def register(app, rt, prefix=""):
    @rt(f"{prefix}/dag_rendering", methods=["GET"])
    async def dag_rendering(request: Request):
        execution_id = request.query_params.get("id")
        try:
            execution_id = int(execution_id)
        except (TypeError, ValueError):
            return Div(P(f"Invalid execution ID {execution_id}"))

        tasktypes_list, taskgroups_list = \
            await get_execution_dag_elements_lists(execution_id)
        if tasktypes_list is None:
            return Div(P(f"Invalid execution ID {execution_id}"))

        template_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "utils", "execution")
        env = Environment(loader=FileSystemLoader(template_dir))
        env.globals['get_text_pixel_width'] = get_text_pixel_width
        template = env.get_template("svg_template.html")
        rendering_content = template.render(
            id_prefix="00000",
            nodes=tasktypes_list,
            taskgroups=taskgroups_list or []
        )

        return rendering_content


    @rt(f"{prefix}/exec_current_progress", methods=["GET"])
    async def exec_current_progress(request: Request):
        """Progress of the execution.

        May be the final one if the execution is
        already completed.
        """
        execution_id = request.query_params.get("id")
        try:
            execution_id = int(execution_id)
        except (TypeError, ValueError):
            return Div(P(f"Invalid execution ID {execution_id}"))

        tasks_list, taskgroups_list = \
            await get_execution_elements_lists(execution_id)
        if tasks_list is None:
            return Div(P(f"Invalid execution ID {execution_id}"))

        chart = draw_chart(execution_id, tasks_list, taskgroups_list)
        return chart


    @rt(f"{prefix}/execution_info", methods=["GET"])
    async def execution_info(request: Request):
        execution_id = request.query_params.get("id")
        try:
            execution_id = int(execution_id)
        except (TypeError, ValueError):
            return Response(
                f"Invalid execution ID {execution_id}", 500)

        dao = AsyncDAO(
            db_url=os.environ["RP_METADATASTORE_ASYNC_URL"]
        )
        execution_info = await dao.get_execution_info(execution_id)

        return JSONResponse(execution_info)


    @rt(f"{prefix}/execution_number", methods=["GET"])
    async def get_execution_number(request: Request):
        """Which execution of that pipeline (by name) it is,

        (first, second, etc.).
        """
        execution_id = request.query_params.get("id")
        execution_number_response = await execution_number(execution_id)

        return execution_number_response


    @rt(f"{prefix}/tasktype_docstring", methods=["GET"])
    async def tasktype_docstring(request: Request):
        tasktype_uuid = request.query_params.get("uuid")
        try:
            tasktype_uuid = str(tasktype_uuid)
        except (TypeError, ValueError):
            return Response(
                f"Invalid tasktype UUID {tasktype_uuid}", 500)

        dao = AsyncDAO(
            db_url=os.environ["RP_METADATASTORE_ASYNC_URL"]
        )
        tasktype_docstring = \
            await dao.get_tasktype_docstring(tasktype_uuid)

        return JSONResponse(tasktype_docstring)


    @rt(f"{prefix}/pipeline-card", methods=["GET"])
    async def pipeline_card(request: Request):
        exec_id = request.query_params.get("exec_id")
        try:
            execution_id = int(exec_id)
        except (TypeError, ValueError):
            raise HTTPException(
                status_code=500,
                detail=f"exec_id '{exec_id}'"
            )

        dao = AsyncDAO(
            db_url=os.environ["RP_METADATASTORE_ASYNC_URL"]
        )
        try:
            execution_name = (
                await dao.get_execution(execution_id)).name
        except AttributeError as ex:
            raise HTTPException(
                status_code=500,
                detail=f"exec_id '{exec_id}' not valid."
            )

        filename = os.path.join(
            os.environ["RP_ARTIFACTS_STORE"],
            execution_name, str(execution_id),
            "pipeline_card.html"
        )
        if not os.path.exists(filename):
            uvicorn_logger = logging.getLogger("uvicorn")
            client_info = ClientInfo(
                ip=request.client.host,
                port=request.client.port,
                url=request.url.path
            )
            uvicorn_logger.info(
                f"{client_info['ip']}:{client_info['port']}" +
                f"{client_info['url']} pipeline-card not found "+
                filename
            )
            raise HTTPException(
                status_code=404,
                detail="pipeline-card not found"
            )

        with open(filename, 'r', encoding='utf-8') as f:
            html_content = f.read()

        return HTMLResponse(
            content=html_content, status_code=200)


    @rt(f"{prefix}/task_traces", methods=["GET"])
    async def get_task_traces(request: Request):
        task_id = request.query_params.get("task_id")
        try:
            task_id = int(task_id)
        except (TypeError, ValueError):
            return Response(
                f"Invalid task ID {task_id}", 500)

        dao = AsyncDAO(
            db_url=os.environ["RP_METADATASTORE_ASYNC_URL"]
        )
        task_traces = await dao.get_task_traces(task_id)
        if task_traces:
            for task_trace in task_traces:
                task_trace["timestamp"] = int(
                    task_trace["timestamp"].timestamp() * 1_000)

        return JSONResponse(task_traces)


    @rt(f"{prefix}/execution_events", methods=["GET"])
    async def sse_execution_events(request: Request):
        client_info = ClientInfo(
            ip=request.client.host,
            port=request.client.port,
            url=request.url.path
        )
        return StreamingResponse(
            multiplexed_event_generator(client_info=client_info),
            media_type="text/event-stream"
        )


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
                H1(# execution-name
                    style="padding-left: 30px; margin: 40px 0;",
                    cls="shiny-gold-text",
                    id="execution-name"
                ),
                Div(# DAG docstring
                    "",
                    Div(
                        cls="content"
                    ),
                    Div(
                        "Show more",
                        id="dag-docstring-show-more"
                    ),
                    id="dag-docstring"
                ),
                H2(# subtitle
                    Div(
                        "execution # ",
                        Div(
                            id="execution-number",
                            title=f"exec_id:\u00A0{execution_id}",
                        ),
                        "/",
                        Div(
                            id="executions-count"
                        ),
                        "\u00A0-\u00A0",
                        Div(
                            id="utc-start-date-time-str"
                        ),
                        style=(
                            "display: flex; align-items: center; "
                            "flex-wrap: nowrap; white-space: nowrap;"
                        )
                    ),
                    style=(
                        "padding-left: 10px; margin: 20px 0; "
                        "background-color: #FFFFCC40; "
                        "text-align: left; color: #FFEA66;"
                    ),
                    id="subtitle"
                ),
                Div(# pipeline-card button
                    A(
                        Div(
                            Div(
                                "pipeline-card"
                            ),
                            Div(
                                "\U0001F5D7", # &#x1f5d7;
                                style=(
                                    "display: inline-block; "
                                    "vertical-align: top; "
                                    "margin-top: -5px;"
                                )
                            ),
                            style="""
                                font-size: 16px;
                                font-weight: 700;
                                line-height: 1;
                                display: inline-flex;
                                width: 100%;
                                justify-content: center;
                                align-items: center;
                                gap: 6px;
                                background: linear-gradient(
                                    145deg,
                                    #d4a9f2 0%,
                                    #c28be8 20%,
                                    #a55ccc 40%,
                                    #8a3fb8 50%,
                                    #a55ccc 60%,
                                    #c28be8 80%,
                                    #d4a9f2 100%
                                );
                                -webkit-background-clip: text;
                                background-clip: text;
                                -webkit-text-fill-color: transparent;
                                filter:
                                    drop-shadow(0 1px 1px rgba(77,0,102,0.4))
                                    drop-shadow(0 2px 4px rgba(77,0,102,0.3))
                                    drop-shadow(0 0 10px rgba(77,0,102,0.2))
                                    drop-shadow(0 4px 8px rgba(77,0,102,0.25))
                                    drop-shadow(3px 3px 6px rgba(77,0,102,0.4));
                            """
                        ),
                        tabindex="0",
                        cls=["pill-button"],
                        style=(
                            "text-decoration: none; "
                            "margin-right: 6em;"
                        ),
                        href=f"/pipeline-card?exec_id={execution_id}",
                        _onkeydown="""
                            if(
                                event.ctrlKey && (
                                    event.key===' '||event.key==='Spacebar'||event.key==='Enter'
                                )
                            ){
                                event.preventDefault();
                                window.open(this.href, '_blank', 'noopener,noreferrer');
                            } else if(
                                event.key===' '||event.key==='Spacebar'||event.key==='Enter'
                            ){
                                event.preventDefault();
                                this.click();
                            }
                        """
                    ),
                    Style(""" /* pill-button */
                        .pill-button {
                            display: inline-flex;
                            align-items: center;
                            justify-content: center;
                            padding: 8px 24px;
                            border-radius: 12px;
                            cursor: pointer;
                            position: relative;
                            backdrop-filter: blur(14px) saturate(160%);
                            -webkit-backdrop-filter: blur(14px) saturate(160%);
                            transition: all 0.15s ease-in-out;

                            background:
                                radial-gradient(
                                    ellipse at 30% 30%,
                                    rgba(255,255,255,0.25),
                                    rgba(255,255,255,0.08) 25%,
                                    rgba(255,255,255,0.0) 45%
                                ),
                                radial-gradient(
                                    ellipse at 70% 70%,
                                    rgba(77,0,102,0.9),
                                    rgba(77,0,102,0.5) 30%,
                                    rgba(77,0,102,0.2) 55%,
                                    rgba(77,0,102,0.0) 75%
                                ),
                                linear-gradient(
                                    180deg,
                                    rgba(255,255,255,0.12) 0%,
                                    rgba(255,255,255,0.04) 35%,
                                    rgba(0,0,0,0.2) 100%
                                );
                            box-shadow:
                                inset 0 2px 2px rgba(255,255,255,0.25),
                                inset 0 -6px 12px rgba(77,0,102,0.55),
                                inset 0 0 18px rgba(255,255,255,0.18),
                                0 6px 14px rgba(77,0,102,0.45),
                                0 12px 28px rgba(77,0,102,0.35),
                                0 0 24px rgba(77,0,102,0.35);
                        }

                        .pill-button:hover {
                            background:
                                radial-gradient(
                                    ellipse at 25% 25%,
                                    rgba(255,255,255,0.35),
                                    rgba(255,255,255,0.12) 25%,
                                    rgba(255,255,255,0.0) 45%
                                ),
                                radial-gradient(
                                    ellipse at 75% 75%,
                                    rgba(77,0,102,0.95),
                                    rgba(77,0,102,0.6) 30%,
                                    rgba(77,0,102,0.25) 55%,
                                    rgba(77,0,102,0.0) 75%
                                ),
                                linear-gradient(
                                    180deg,
                                    rgba(255,255,255,0.18) 0%,
                                    rgba(255,255,255,0.06) 35%,
                                    rgba(0,0,0,0.22) 100%
                                );
                            box-shadow:
                                inset 0 2px 4px rgba(255,255,255,0.35),
                                inset 0 -6px 14px rgba(77,0,102,0.65),
                                inset 0 0 20px rgba(255,255,255,0.25),
                                0 8px 18px rgba(77,0,102,0.5),
                                0 14px 32px rgba(77,0,102,0.45),
                                0 0 28px rgba(77,0,102,0.4);
                        }

                        .pill-button:focus {
                            outline: none;
                            background:
                                radial-gradient(
                                    ellipse at 25% 25%,
                                    rgba(255,255,255,0.38),
                                    rgba(255,255,255,0.15) 25%,
                                    rgba(255,255,255,0.0) 45%
                                ),
                                radial-gradient(
                                    ellipse at 75% 75%,
                                    rgba(77,0,102,0.97),
                                    rgba(77,0,102,0.63) 30%,
                                    rgba(77,0,102,0.28) 55%,
                                    rgba(77,0,102,0.0) 75%
                                ),
                                linear-gradient(
                                    180deg,
                                    rgba(255,255,255,0.2) 0%,
                                    rgba(255,255,255,0.08) 35%,
                                    rgba(0,0,0,0.23) 100%
                                );
                            box-shadow:
                                inset 0 3px 5px rgba(255,255,255,0.37),
                                inset 0 -6px 16px rgba(77,0,102,0.67),
                                inset 0 0 22px rgba(255,255,255,0.27),
                                0 10px 20px rgba(77,0,102,0.53),
                                0 16px 30px rgba(77,0,102,0.48),
                                0 0 30px rgba(77,0,102,0.45);
                        }

                        .pill-button:hover:focus {
                            background:
                                radial-gradient(
                                    ellipse at 20% 20%,
                                    rgba(255,255,255,0.45),
                                    rgba(255,255,255,0.18) 25%,
                                    rgba(255,255,255,0.0) 45%
                                ),
                                radial-gradient(
                                    ellipse at 80% 80%,
                                    rgba(77,0,102,1),
                                    rgba(77,0,102,0.7) 30%,
                                    rgba(77,0,102,0.35) 55%,
                                    rgba(77,0,102,0.0) 75%
                                ),
                                linear-gradient(
                                    180deg,
                                    rgba(255,255,255,0.22) 0%,
                                    rgba(255,255,255,0.1) 35%,
                                    rgba(0,0,0,0.25) 100%
                                );
                            box-shadow:
                                inset 0 4px 6px rgba(255,255,255,0.45),
                                inset 0 -8px 18px rgba(77,0,102,0.72),
                                inset 0 0 26px rgba(255,255,255,0.32),
                                0 12px 24px rgba(77,0,102,0.58),
                                0 18px 36px rgba(77,0,102,0.52),
                                0 0 36px rgba(77,0,102,0.5);
                        }

                        .pill-button:active {
                            background:
                                radial-gradient(
                                    ellipse at 20% 20%,
                                    rgba(255,255,255,0.85),
                                    rgba(255,255,255,0.5) 25%,
                                    rgba(255,255,255,0.0) 45%
                                ),
                                radial-gradient(
                                    ellipse at 80% 80%,
                                    rgba(255,255,255,0.85),
                                    rgba(255,255,255,0.5) 25%,
                                    rgba(255,255,255,0.0) 45%
                                ),
                                radial-gradient(
                                    ellipse at 80% 80%,
                                    rgba(77,0,102,1),
                                    rgba(77,0,102,0.9) 30%,
                                    rgba(77,0,102,0.65) 55%,
                                    rgba(77,0,102,0.0) 75%
                                ),
                                linear-gradient(
                                    180deg,
                                    rgba(255,255,255,0.65) 0%,
                                    rgba(255,255,255,0.35) 35%,
                                    rgba(0,0,0,0.45) 100%
                                ) !important;
                            box-shadow:
                                inset 0 10px 20px rgba(255,255,255,0.85) !important,
                                inset 0 -22px 40px rgba(77,0,102,0.95) !important,
                                inset 0 0 60px rgba(255,255,255,0.6) !important,
                                0 28px 56px rgba(77,0,102,0.8) !important,
                                0 36px 72px rgba(77,0,102,0.75) !important,
                                0 0 72px rgba(77,0,102,0.7) !important;
                            transform: translateY(1px) !important;
                        }
                    """),
                    id="pipeline-card",
                    style=(
                        "width: 100%; "
                        "text-align-last: right;"
                    )
                ),
                Script(f"""// async get execution-info (name, etc.)
                    function updateExecutionNumber(executionNumberJson) {{
                        // Update the executions counters (incl. tooltip)
                        //console.log("updateExecutionNumber ENTER", executionNumberJson);
                        document.getElementById("execution-number").innerText = executionNumberJson.number;
                        const executionsCount = document.getElementById("executions-count");
                        executionsCount.innerText = executionNumberJson.count;
                        executionsCount.title =
                            `${{executionNumberJson.count - executionNumberJson.completed}} ongoing\n` +
                            `${{executionNumberJson.failed}} failed`;
                    }}

                    (function(){{
                        function formatUtcTimestamp(isoString) {{
                            // expected to receive UTC formatted date time string
                            // includes year if different than current.
                            if (!isoString) return null;
                            if (!isoString.endsWith('Z')) {{
                                isoString += 'Z';
                            }}
                            const date = new Date(isoString);
                            const options = {{
                                weekday: 'short',
                                year: 'numeric',
                                month: 'short',
                                day: '2-digit',
                                hour: '2-digit',
                                minute: '2-digit',
                                second: '2-digit',
                                hour12: true,
                                timeZone: 'UTC'
                            }};
                            const parts = new Intl.DateTimeFormat('en-US', options).formatToParts(date);
                            const partMap = {{}};
                            parts.forEach(part => {{
                                partMap[part.type] = part.value;
                            }});
                            const currentYear = new Date().getUTCFullYear();
                            const formattedString = 
                                `${{partMap.weekday}} ${{partMap.month}} ${{partMap.day}}` +
                                (parseInt(partMap.year) !== currentYear ? ` ${{partMap.year}}` : '') +
                                `, ${{partMap.hour}}:${{partMap.minute}}:${{partMap.second}} ${{partMap.dayPeriod}} UTC`;
                            return formattedString
                        }}

                        document.addEventListener("DOMContentLoaded", function() {{
                            // execution-name & username & start-date-time
                            fetch("{prefix}/execution_info?id={execution_id}", {{
                                method: 'GET',
                                headers: {{ "HX-Request": "true",
                                            "Cache-Control": "no-cache" }},
                                cache: 'no-cache'
                            }})
                            .then(function(resp) {{
                                if (!resp.ok) {{
                                    return resp.text().then(text => {{
                                        throw new Error(text || resp.statusText || 'Unknown error');
                                    }});
                                }}
                                return resp.json();
                            }})
                            .then(function(execution_info) {{
                                // inform execution-name & subtitle
                                const executionName = document.getElementById("execution-name");
                                executionName.innerText = execution_info.name;
                                executionName.title = execution_info.username;
                                if (execution_info.docstring) {{
                                    const dagDocstring = document.getElementById("dag-docstring");
                                    const dagDocstringContentDiv =
                                        document.querySelector("#dag-docstring .content");
                                    dagDocstringContentDiv.innerText =
                                        (execution_info.docstring || "").trim();
                                    dagDocstring.classList.add('showing');
                                    checkDocstringOverflow();
                                }}
                                // (flow run # 4, run_id: 101 - Friday Apr 11 2025 11:57:29 PM UTC)
                                document.getElementById("utc-start-date-time-str").innerText =
                                    formatUtcTimestamp(execution_info.start_timestamp);

                            }}).catch(error => {{
                                console.error("Error fetching {prefix}/execution_info:", error);
                            }});

                            // execution-number
                            fetch("{prefix}/execution_number?id={execution_id}", {{
                                method: 'GET',
                                headers: {{ "HX-Request": "true",
                                            "Cache-Control": "no-cache" }},
                                cache: 'no-cache'
                            }})
                            .then(function(resp) {{
                                if (!resp.ok) {{
                                  return resp.text().then(text => {{
                                    throw new Error(text || resp.statusText || 'Unknown error');
                                  }});
                                }}
                                return resp.json();
                            }})
                            .then(function(executionNumberJson) {{
                                updateExecutionNumber(executionNumberJson);
                            }}).catch(error => {{
                                console.error("Error fetching {prefix}/execution_number:", error);
                            }});

                        }});
                    }})();
                """),
                Script("""// DAG docstring 'show-more'
                    function checkDocstringOverflow() {
                        const dagDocstring = document.getElementById('dag-docstring');
                        const dagDocstringContentDiv = document.querySelector("#dag-docstring .content");
                        const showMore = document.getElementById('dag-docstring-show-more');

                        // Calculating two lines of text height
                        const lineHeight = parseFloat(getComputedStyle(dagDocstringContentDiv).lineHeight);
                        const maxTwoLinesHeight = lineHeight * 2;
                        //console.log("checkDocstringOverflow", lineHeight, maxTwoLinesHeight, dagDocstringContentDiv.scrollHeight);

                        const overflows =
                            dagDocstringContentDiv.scrollHeight > maxTwoLinesHeight + 2;
                        if (overflows) {
                            if (showMore.style.display != "block") {
                                showMore.style.display = "block";
                                dagDocstring.classList.add("collapsed");
                            }
                        } else {
                            showMore.style.display = "none";
                        }
                    }

                    window.addEventListener('resize', checkDocstringOverflow);
                    checkDocstringOverflow();

                    const dagDocstring = document.getElementById('dag-docstring');
                    const showMoreDagDocstringBtn = document.getElementById('dag-docstring-show-more');

                    showMoreDagDocstringBtn.addEventListener('click', () => {
                        if (dagDocstring.classList.contains('collapsed')) {
                            // Expand
                            dagDocstring.classList.remove('collapsed');
                            dagDocstring.classList.add('expanded');
                            showMoreDagDocstringBtn.innerText = 'Show less';
                        } else {
                            // Collapse
                            dagDocstring.scrollTop = 0;
                            dagDocstring.classList.remove('expanded');
                            dagDocstring.classList.add('collapsed');
                            showMoreDagDocstringBtn.innerText = 'Show more';
                        }
                    });
                """),
                Script(f"""// SSE events : retraining-pipeline execution events
                    let executionEventsSource;

                    // add some logging convenience to event-source
                    EventSource.prototype._listenerStore = [];
                    const origAddEventListener =
                        EventSource.prototype.addEventListener;
                    EventSource.prototype.addEventListener =
                        function(type, listener, options) {{
                            this._listenerStore.push({{type, listener, options}});
                            origAddEventListener.call(this, type, listener, options);
                        }};
                    function listEventSourceListeners(es) {{
                        return es._listenerStore;
                    }}

                    // actually register source of Execution events
                    function registerExecEventsSrc() {{
                        executionEventsSource = new EventSource(
                            `{prefix}/execution_events`
                        );

                        executionEventsSource.onerror = (err) => {{
                            console.error('SSE error:', err);
                        }};
                        // Force close EventSource when leaving the page
                        window.addEventListener('pagehide', () => {{
                            executionEventsSource.close();
                        }});

                        /* ************************
                        * "new execution started" *
                        ************************ */
                        // update executions count label
                        executionEventsSource.addEventListener('newExecution', (event) => {{
                            let payload;
                            try {{
                                // Parse the server data, assuming it's JSON
                                payload = JSON.parse(event.data);
                            }} catch (e) {{
                                console.error('Error parsing SSE message data:',
                                              event.data);
                                console.error(e);
                                return;
                            }}
                            //console.log("executionEventsSource 'newExecution'", payload);
                            if (
                                payload.name ===
                                    document.getElementById("execution-name").innerText
                            ) {{
                                executionNumberJson = payload;
                                executionNumberJson.number =
                                    document.getElementById("execution-number").innerText;
                                updateExecutionNumber(executionNumberJson);
                            }}
                        }});
                        /* ********************* */

                        /* *********************
                        * "an execution ended" *
                        ********************** */
                        // update executions count label and tooltip
                        executionEventsSource.addEventListener('executionEnded', (event) => {{
                            let payload;
                            try {{
                                // Parse the server data, assuming it's JSON
                                payload = JSON.parse(event.data);
                            }} catch (e) {{
                                console.error('Error parsing SSE message data:',
                                              event.data);
                                console.error(e);
                                return;
                            }}
                            //console.log("executionEventsSource 'executionEnded'", payload);
                            if (
                                payload.name ===
                                    document.getElementById("execution-name").innerText
                            ) {{
                                executionNumberJson = payload;
                                executionNumberJson.number =
                                    document.getElementById("execution-number").innerText;
                                updateExecutionNumber(executionNumberJson);
                            }}
                        }});
                        /* ******************* */
                    }}
                    registerExecEventsSrc();
                """),
                Script(f"""// re-register SSE source on window history.back()
                    window.addEventListener('pageshow', function(event) {{
                        if (event.persisted) {{ // page reloaded from bfcache
                            // inject up-to-date Gantt chart
                            const targetDiv = document.getElementById("gantt-script-placeholder");
                            fetch("{prefix}/exec_current_progress?id={execution_id}",
                                  {{
                                        method: 'GET',
                                        headers: {{ "HX-Request": "true",
                                                    "Cache-Control": "no-cache" }},
                                        cache: 'no-cache'
                                  }}
                                ).then(response => response.text())
                                 .then(html =>
                                {{
                                    targetDiv.innerHTML = "";
                                    const script = document.createElement("script");
                                    const inlineCode = html.replace(/<script[\s\S]*?>|<\/script>/gi, '');
                                    script.textContent = inlineCode;
                                    targetDiv.appendChild(script);
                                }}
                            );

                            registerExecEventsSrc();
                            registerTaskEvents();
                        }}
                    }});
                """),
                Script(f"""// handling & recovering from server-loss
                    const statusCircle = document.getElementById('status-circle');
                    let previousClasses =
                        Array.from(statusCircle.classList); // remember initial classes
                    function onClassChange(mutationsList) {{
                        for (let mutation of mutationsList) {{
                            if (
                                mutation.type === 'attributes' &&
                                mutation.attributeName === 'class'
                            ) {{
                                const newClasses = Array.from(statusCircle.classList);

                                if (
                                    newClasses.includes('disconnected') &&
                                    !previousClasses.includes('disconnected')
                                ) {{
                                    console.log("disconnected");
                                    executionEventsSource.close();
                                }} else if (
                                    newClasses.includes('connected') &&
                                    !previousClasses.includes('connected')
                                ) {{
                                    console.log("reconnected");

                                    // inject up-to-date Gantt chart
                                    const targetDiv = document.getElementById("gantt-script-placeholder");
                                    fetch("{prefix}/exec_current_progress?id={execution_id}",
                                          {{
                                                method: 'GET',
                                                headers: {{ "HX-Request": "true",
                                                            "Cache-Control": "no-cache" }},
                                                cache: 'no-cache'
                                          }}
                                        ).then(response => response.text())
                                         .then(html =>
                                        {{
                                            targetDiv.innerHTML = "";
                                            const script = document.createElement("script");
                                            const inlineCode = html.replace(
                                                /<script[\s\S]*?>|<\/script>/gi,
                                                ""
                                            );
                                            script.textContent = inlineCode;
                                            targetDiv.appendChild(script);
                                        }}
                                    );

                                    // inject up-to-date task traces
                                    const existingModal = document.getElementById('detailsModal');
                                        const visible = existingModal && !!(
                                            existingModal.offsetWidth ||
                                            existingModal.offsetHeight ||
                                            existingModal.getClientRects().length
                                        );
                                    if (visible) {{
                                        const tracesContainer =
                                            existingModal.querySelector(".traces-log-container");
                                        if (tracesContainer) {{
                                            // case "log-traces" TAB
                                            // has been initialized/loaded before
                                            // => needs refreshing
                                            //console.log("needs refreshing");
                                            const container = existingModal.querySelector("#tab-traces");
                                            const taskId =
                                                existingModal.querySelector("#modal-task-id").textContent;
                                            const tracesContainer =
                                                container.querySelector(".traces-log-container");

                                            const wasAutoscroll =
                                                tracesContainer.classList.contains("autoscroll");
                                            // index of the first log-trace in the viewport
                                            let firstVisibleIndex = 0;
                                            if (!wasAutoscroll) {{
                                                const lines =
                                                    tracesContainer.querySelectorAll(".trace-line");
                                                const visibleRect = tracesContainer.getBoundingClientRect();
                                                if (visibleRect.height > 0) {{
                                                    const scrollTop = tracesContainer.scrollTop;
                                                    for (let i = 0; i < lines.length; i++) {{
                                                        if (lines[i].offsetTop >= scrollTop) {{
                                                            firstVisibleIndex = i;
                                                            break;
                                                        }}
                                                    }}
                                                }} else {{
                                                    // case "traces tab not showing"
                                                    const execId =
                                                        existingModal.querySelector(
                                                            "#modal-exec-id").textContent;
                                                    const taskId = existingModal.querySelector(
                                                        "#modal-task-id").textContent;
                                                    const tasktypeUuid =
                                                        existingModal.querySelector(
                                                            "#modal-tasktype-uuid").textContent;
                                                    //console.log("(execId, tasktypeUuid, taskId)", "  -  ",
                                                    //            execId, "  -  ", tasktypeUuid, "  -  ", taskId);
                                                    const entry =
                                                        getTaskTracesCookieEntry(
                                                            execId, tasktypeUuid, taskId);
                                                    //console.log("taskTracesCookieEntry", entry);
                                                    firstVisibleIndex = entry.viewPortTopLineIndex;
                                                }}
                                            }}

                                            // clean start
                                            existingModal.querySelectorAll(
                                                    '#tab-traces > *:not(.traces-log-toolbar)'
                                                ).forEach(el => el.remove());

                                            loadTracesTabContent(
                                                container, taskId
                                            ).then(({{ tracesContainer }}) => {{
                                                if (tracesContainer) {{
                                                    // scoll even if another TAB is showing
                                                    waitForVisible(
                                                        "#tab-traces .traces-log-container"
                                                    ).then(() => {{
                                                        if (wasAutoscroll) {{
                                                            tracesContainer.scrollTop =
                                                                tracesContainer.scrollHeight;
                                                        }} else {{
                                                            //console.log(
                                                            //    "firstVisibleIndex", firstVisibleIndex);
                                                            const lines =
                                                                tracesContainer.querySelectorAll(
                                                                    ".trace-line");
                                                            const target = lines[firstVisibleIndex];
                                                            if (target) {{
                                                                tracesContainer.scrollTop =
                                                                    target.offsetTop -
                                                                        tracesContainer.offsetTop;
                                                            }}
                                                        }}
                                                    }}).catch(err => {{
                                                        console.warn(err);
                                                    }});

                                                }}
                                            }}).catch(err => {{
                                                console.error('loadTracesTabContent failed:', err);
                                            }});
                                        }}
                                    }}

                                    registerExecEventsSrc();
                                    registerTaskEvents();
                                    document.dispatchEvent(
                                        new Event("DOMContentLoaded",
                                                  {{ bubbles: true, cancelable: true }}));
                                }}

                                // Update previousClasses for next mutation check
                                previousClasses = newClasses;
                            }}
                        }}
                    }}

                    const observer = new MutationObserver(onClassChange);
                    observer.observe(statusCircle, {{ attributes: true }});
                """),
                Style(""" /* header and body */
                    .shiny-gold-text {
                        font-size: 3em; font-weight: bold; min-height: 50.5px;
                        color: #FFD700; /* Base gold color */
                        text-shadow: 
                            0 0 5px rgba(255, 215, 0, 0.7),  /* Inner glow */
                            0 0 10px rgba(255, 215, 0, 0.5), /* Outer glow */
                            0 0 15px rgba(255, 215, 0, 0.3); /* Soft outer glow */
                        background: linear-gradient(45deg, #FFD700, #FFEA66);
                        -webkit-background-clip: text;
                        -webkit-text-fill-color: transparent;
                    }

                    .body-execution { /* page content container */
                        padding-top: 3.5rem;
                        padding-bottom: 2.5rem;
                    }
                """),

                Script("""// savePageAsSingleFile
                    async function savePageAsSingleFile() {
                        try {
                            // Clone the entire document
                            const docClone = document.cloneNode(true);

                            // === INLINE CSS ===
                            const styles = Array.from(document.styleSheets);
                            const inlinedStyles = [];

                            for (const sheet of styles) {
                                try {
                                    const rules = Array.from(sheet.cssRules || sheet.rules || []);
                                    const css = rules.map(rule => rule.cssText).join('\\n');
                                    if (css) {
                                        inlinedStyles.push(css);
                                    }
                                } catch (e) {
                                    console.warn('Could not access stylesheet:', sheet.href || 'inline stylesheet', 'Error:', e.message);
                                }
                            }

                            // Remove existing style and link elements from clone
                            const oldStyles = docClone.querySelectorAll('style, link[rel="stylesheet"]');
                            oldStyles.forEach(el => el.remove());

                            // Add new consolidated style element
                            if (inlinedStyles.length > 0) {
                                const styleElement = docClone.createElement('style');
                                styleElement.textContent = inlinedStyles.join('\\n\\n');
                                docClone.head.appendChild(styleElement);
                            }

                            // === INLINE EXTERNAL JAVASCRIPT ===
                            const externalScripts = docClone.querySelectorAll('script[src]');

                            for (const script of externalScripts) {
                                try {
                                    const src = script.src;

                                    const response = await fetch(src);
                                    if (!response.ok) {
                                        console.warn(`Failed to fetch script: ${src} - Status: ${response.status}`);
                                        continue;
                                    }

                                    const jsContent = await response.text();

                                    // Create new inline script with the fetched content
                                    const inlineScript = docClone.createElement('script');

                                    // Preserve script attributes (except src)
                                    for (const attr of script.attributes) {
                                        if (attr.name !== 'src') {
                                            inlineScript.setAttribute(attr.name, attr.value);
                                        }
                                    }

                                    inlineScript.textContent = jsContent;

                                    // Replace the external script with inline version
                                    script.parentNode.replaceChild(inlineScript, script);

                                } catch (e) {
                                    console.warn(`Could not inline script ${script.src}:`, e.message);
                                }
                            }

                            // === INLINE IMAGES (fetch to base64) ===
                            const images = Array.from(document.querySelectorAll('img[src]'));
                            const cloneImages = Array.from(docClone.querySelectorAll('img[src]'));

                            for (let i = 0; i < images.length; i++) {
                                const originalImg = images[i];
                                const cloneImg = cloneImages[i];

                                if (!cloneImg || !originalImg.src) continue;

                                try {
                                    const response = await fetch(originalImg.src);
                                    if (!response.ok) {
                                        console.warn(`Failed to fetch image: ${originalImg.src} - Status: ${response.status}`);
                                        continue;
                                    }

                                    const blob = await response.blob();
                                    const base64 = await new Promise((resolve, reject) => {
                                        const reader = new FileReader();
                                        reader.onloadend = () => resolve(reader.result);
                                        reader.onerror = reject;
                                        reader.readAsDataURL(blob);
                                    });

                                    cloneImg.src = base64;
                                    console.log('Successfully inlined image:', originalImg.src);

                                } catch (e) {
                                    console.warn('Could not inline image:', originalImg.src, '- Error:', e.message);
                                }
                            }

                            // Get the final HTML
                            const doctype = '<!DOCTYPE html>';
                            const html = docClone.documentElement.outerHTML;
                            const fullHtml = doctype + '\\n' + html;

                            // Create blob and download
                            const blob = new Blob([fullHtml], { type: 'text/html' });
                            const url = URL.createObjectURL(blob);
                            const a = document.createElement('a');
                            a.href = url;
                            a.download = `saved-page-${Date.now()}.html`;
                            document.body.appendChild(a);
                            a.click();
                            document.body.removeChild(a);
                            URL.revokeObjectURL(url);

                            alert('Page saved successfully!');

                        } catch (error) {
                            console.error('Error saving page:', error);
                            alert('Error saving page. Check console for details.');
                        }
                    }
                """),
                # Button("Save Page", onclick="savePageAsSingleFile()"),

                ## TODO, add stuff here (pipeline-card sections)

                H1(# timeline
                    "Execution Timeline",
                    style="""
                        color: #6082B6;
                        padding: 2rem 0 3rem;
                        margin-bottom: 0;
                        background-color: rgba(0, 0, 0, 0.03);
                        border-bottom: 1px solid rgba(0, 0, 0, 0.125);
                        text-align: center;
                    """
                ),
                Script("const interBarsSpacing = 2;     /* in px */"),
                Script(src="/collapsible_grouped_table.js"),
                Script(src="/gantt-timeline-renderer.js"),
                Script(src="/gantt-events.js"),
                Style(""" /* collapsible table */
                    .gantt-table {
                        border-collapse: collapse;
                        table-layout: fixed;
                        width: calc(100% - 10px); /* left+right margin */
                        margin: 6px 5px 4px 5px; /* extra room on top (vs. bottom) for reflexion effect */
                        border-radius: 12px;
                        font-size: 14px; font-weight: bold; letter-spacing: 0.375px;
                        font-family: Robotto, Arial, sans-serif;
                        overflow: hidden;
                        box-sizing: border-box; /* ensure padding affects size correctly */
                    }

                    .gantt-table thead {
                        display: none;
                    }

                    .gantt-table th,
                    .gantt-table td {
                        padding: 4px 0;         /* important */
                        white-space: nowrap;
                        border: 1px solid #ddd;
                        position: relative;
                    }
                    .gantt-table tr {
                        height: 36px;
                    }

                    /* Remove border on top row cells */
                    tr:first-child td, tr:first-child th {
                      border-top: none;
                    }
                    /* Remove border on bottom row cells */
                    tr:last-child td, tr:last-child th {
                      border-bottom: none;
                    }
                    /* Remove border on first column cells */
                    td:first-child, th:first-child {
                      border-left: none;
                    }
                    /* Remove border on last column cells */
                    td:last-child, th:last-child {
                      border-right: none;
                    }

                    .gantt-table #task-col {
                        /* width computed dynamically */
                    }
                    .gantt-table #timeline-col {
                        width: auto;
                    }

                    .gantt-table tr td:first-child {
                        padding-left: calc(5px + var(--indent-level) * 5px);
                        position: relative;
                    }
                    .gantt-table tr td:last-child { /* timeline col */
                        padding-left: 3px;
                        padding-right:
                            calc(3px + 3px + var(--max-visible-level, 0) * 5px);
                    }

                    .gantt-table tr.hidden {
                        display: none;
                    }

                    .group-header {
                        cursor: pointer;
                    }

                    .element-name {/* shaped labels */
                        align-items: center;
                        justify-content: center;
                        display: flex;
                    }

                    .left-nesting-bar {
                        position: absolute;
                        top: 0;
                        bottom: 0;
                        width: 3px;           /* important */
                        z-index: 0;
                    }

                    .right-nesting-bar {
                        position: absolute;
                        top: 0;
                        bottom: 0;
                        width: 3px;           /* important */
                        z-index: 0;
                        right: 0;
                    }

                    .top-nesting-bar {
                        position: absolute;
                        height: 3px;           /* important */
                        z-index: 0;
                        top: 0;
                    }

                    .bottom-nesting-bar {
                        position: absolute;
                        height: 3px;           /* important */
                        z-index: 0;
                        bottom: 0;
                    }
                """),
                Style(""" /* timelines */
                .gantt-timeline-cell {
                    position: relative;
                    min-height: 50px;
                    vertical-align: middle;
                }

                .gantt-timeline-container {
                    position: relative;
                    width: 100%;
                    height: 25px;
                    line-height: 1em;
                    background: #e8e8e8;
                    border-radius: 4px;
                    overflow: visible;
                }

                .gantt-timeline-bar {
                    position: absolute;
                    height: 100%;
                    top: 0;
                    border-radius: 4px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: #4d0066;
                    font-size: 11px;
                    box-shadow:
                        0 2px 4px rgba(0,0,0,0.2),
                        0 8px 16px rgba(0,0,0,0.1),
                        inset 0 1px 0 rgba(255,255,255,0.4),
                        inset 0 -1px 0 rgba(0,0,0,0.2);
                    transition: box-shadow 0.3s ease, filter 0.3s ease,
                                border 0.3s ease, transform 0.3s ease;

                    /* GOLD BAR */
                    background: linear-gradient(
                        135deg,
                        #ffd700 0%,     /* bright gold */
                        #d4af37 60%,    /* mid warm gold */
                        #a67c00 100%    /* deep gold */
                    );

                    position: relative;
                    overflow: hidden;
                    border: 1px solid rgba(255,255,255,0.2);
                }

                /* Ice crystal texture overlay */
                .gantt-timeline-bar::after {
                    content: '';
                    position: absolute;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    background:
                        linear-gradient(180deg, rgba(255,255,255,0.3) 0%,
                                                transparent 50%, rgba(0,0,0,0.1) 100%),
                        radial-gradient(
                            circle at 20% 30%, rgba(255,255,255,0.4) 0%, transparent 40%),
                        radial-gradient(
                            circle at 80% 70%, rgba(255,255,255,0.3) 0%, transparent 40%),
                        radial-gradient(
                            circle at 50% 50%, rgba(255,255,255,0.15) 0%, transparent 60%);
                    pointer-events: none;
                    z-index: 1;
                }

                /* PURPLE STARTUP SHINE */
                .gantt-timeline-bar:not(.ongoing)::before {
                    content: '';
                    position: absolute;
                    top: -50%;
                    left: -100%;
                    width: 80%;
                    height: 200%;

                    background: linear-gradient(
                        90deg,
                        transparent,
                        rgba(122,0,179,0.6), /* #7a00b3 @ 0.6 */
                        rgba(77,0,102,0.8),  /* #4d0066 @ 0.8 */
                        rgba(122,0,179,0.6),
                        transparent
                    );

                    transform: skewX(-25deg);
                    pointer-events: none;
                    z-index: 100;
                    animation: timeline-bar-shine-start 0.5s ease-out 0.1s;
                    animation-fill-mode: forwards;
                }

                /* Hover shine trigger */
                tr:hover .gantt-timeline-bar .gantt-timeline-bar-hover-shine {
                    animation: timeline-bar-shine-hover 0.8s ease-out;
                    animation-fill-mode: forwards;
                }

                .gantt-timeline-bar:hover {
                    box-shadow:
                        0 4px 12px rgba(0,0,0,0.3),
                        0 12px 32px rgba(0,0,0,0.15),
                        inset 0 2px 0 rgba(255,255,255,0.6),
                        inset 0 -2px 0 rgba(0,0,0,0.3),
                        inset 0 0 20px rgba(255,255,255,0.2);
                    filter: brightness(1.15) saturate(1.1);
                    border: 1px solid rgba(255,255,255,0.4);
                    transform: translateY(-1px);
                }

                .gantt-timeline-bar.ongoing {
                    background: linear-gradient(135deg, #00b3a3 0%, #006b66 100%);
                    animation: timeline-bar-pulse 2s ease-in-out infinite;
                }

                .gantt-timeline-bar.failed {
                    background: linear-gradient(135deg, #cc3333 0%, #800020 100%);
                }

                /* PURPLE HOVER SHINE */
                .gantt-timeline-bar-hover-shine {
                    content: '';
                    position: absolute;
                    top: -50%;
                    left: -100%;
                    width: 80%;
                    height: 200%;
                    background: linear-gradient(
                        90deg,
                        transparent,
                        rgba(122,0,179,0.6),
                        rgba(77,0,102,0.8),
                        rgba(122,0,179,0.6),
                        transparent
                    );
                    transform: skewX(-25deg);
                    pointer-events: none;
                    z-index: 100;
                    opacity: 0;
                }

                /* Animations */
                @keyframes timeline-bar-pulse {
                    0%, 100% {
                        opacity: 1;
                        box-shadow: 
                            0 2px 4px rgba(0,0,0,0.2),
                            0 8px 16px rgba(0,0,0,0.1),
                            inset 0 1px 0 rgba(255,255,255,0.4),
                            inset 0 -1px 0 rgba(0,0,0,0.2);
                    }
                    50% {
                        opacity: 0.85;
                        box-shadow: 
                            0 2px 8px rgba(0,107,102,0.4),
                            0 8px 20px rgba(0,179,163,0.3),
                            inset 0 1px 0 rgba(255,255,255,0.5),
                            inset 0 -1px 0 rgba(0,0,0,0.2);
                    }
                }

                @keyframes timeline-bar-shine-start {
                    0% {
                        left: -100%;
                        opacity: 1;
                    }
                    100% {
                        left: 150%;
                        opacity: 1;
                    }
                }

                @keyframes timeline-bar-shine-hover {
                    0% {
                        left: -100%;
                        opacity: 1;
                    }
                    100% {
                        left: 150%;
                        opacity: 1;
                    }
                }
                """),
                Style(""" /* gantt-events */
                    .task {
                        cursor: pointer;
                    }
                """),
                Div(# Gantt diagram
                    Div(# collapse/expand all
                        Span(
                            "collapse all",
                            style=(
                                "padding-left: 2px; display: inline-block; "
                                "text-align: center; "
                                "width: 106px; flex: 0 0 106px; "
                                "cursor: pointer; user-select: none;"
                            ),
                            _onmousedown=(
                                "this.style.transform='translateY(1px)'; "
                                "this.style.textShadow='0 1px 0 rgba(255, 255, 255, 0.5)'"
                            ),
                            _onmouseup="this.style.transform=''; this.style.textShadow=''",
                            _onmouseleave="this.style.transform=''; this.style.textShadow=''",
                            _onclick=f"collapseAll('gantt-{execution_id}');"
                        ),
                        Span("|", style="flex: 0 0 auto;"),
                        Span(
                            "expand all",
                            style=(
                                "padding-right: 2px; display: inline-block; "
                                "text-align: center; "
                                "width: 95px; flex: 0 0 95px; "
                                "cursor: pointer; user-select: none;"
                            ),
                            _onmousedown=(
                                "this.style.transform='translateY(1px)'; "
                                "this.style.textShadow='0 1px 0 rgba(255, 255, 255, 0.5)'"
                            ),
                            _onmouseup="this.style.transform=''; this.style.textShadow=''",
                            _onmouseleave="this.style.transform=''; this.style.textShadow=''",
                            _onclick=f"expandAll('gantt-{execution_id}');"
                        ),
                        cls="glass-engraved",
                        style=(
                            "display: flex; align-items: flex-start; "
                            "margin-left: auto; width: 250px; "
                            "justify-content: space-between; "
                            "align-items: baseline; "
                            "margin-bottom: 4px; padding: 0px 6px; "
                            "background: linear-gradient(135deg, "
                                "rgba(77,0,102,0.2) 0%, "
                                "rgba(77,0,102,0.7) 100%); "
                            "border: 1px solid rgba(222,226,230,0.5); "
                            "border-radius: 6px; "
                            "box-shadow: 0 2px 8px rgba(77,0,102,0.3), "
                                "inset 0 1px 0 rgba(77,0,102,0.95);"
                        )
                    ),
                    Div(
                        Table(
                            Colgroup(
                                Col(id="task-col"),
                                Col(id="timeline-col"),
                            ),
                            Thead(
                                Tr(
                                    Th("task"),
                                    Th("timeline")
                                )
                            ),
                            Tbody(id="data-tbody"),
                            cls="gantt-table",
                            id=f"gantt-{execution_id}"
                        ),
                        style="""
                            border-radius: 12px;
                            overflow: hidden;
                            box-sizing: border-box;

                            background: rgba(255, 255, 255, 0.2);
                            backdrop-filter: blur(8px);
                            border: 1px solid rgba(255, 255, 255, 0.3);
                            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1),
                                        inset 0 1px 0 rgba(255, 255, 255, 0.6);
                        """
                    ),
                    style="""
                        width: 90vw;
                        display: block;
                        margin-left: auto;
                        margin-right: auto;
                        padding: 8px 12px 8px 12px; border-radius: 12px;
                        background: rgba(248, 249, 250, 0.3);
                        box-shadow: 0 4px 12px rgba(0,0,0,0.1),
                                    inset 0 1px 0 rgba(255,255,255,0.6);
                        border: 1px solid rgba(222,226,230,0.4);
                    """
                ),
                Div(# init Gantt diagram data (loaded async) and timeline renderer
                    Script("// Placeholder script (shall be replaced async)."),
                    id="gantt-script-placeholder",
                    hx_get=f"{prefix}/exec_current_progress?id={execution_id}",
                    hx_trigger="load",
                    hx_swap="innerHTML",
                    hx_headers='{"Cache-Control": "no-cache"}'
                ),
                Script(f"""// SSE events : retraining-pipeline task events
                    // additional event listeners to existing executionEventsSource
                    function registerTaskEvents() {{
                        /* *****************
                        * "a task started" *
                        ***************** */
                        executionEventsSource.addEventListener('newTask', (event) => {{
                            let payload;
                            try {{
                                // Parse the server data, assuming it's JSON
                                payload = JSON.parse(event.data);
                            }} catch (e) {{
                                console.error('Error parsing SSE message data:', event.data);
                                console.error(e);
                                return;
                            }}

                            if (payload.exec_id == {execution_id}) {{
                                //console.log("executionEventsSource 'newTask'", payload);

                                function checkAndInsert() {{
                                    /* ***************************************
                                    * Ensure the init table state is         *
                                    * loaded before handling new events.     *
                                    *                                        *
                                    * BEWARE :                               *
                                    *   This will lead to                    *
                                    *   treating events twice                *
                                    *   if they occur in the middle of the   *
                                    *   initial response.                    *
                                    *   (which we can handle and,            *
                                    *    if we don't wait, the gantt object  *
                                    *    may not exists yet here)            *
                                    *************************************** */
                                    if (window.execGanttTimelineObj) {{
                                        ganttInsert('execGanttTimelineObj', payload, interBarsSpacing);
                                    }} else {{
                                        requestAnimationFrame(checkAndInsert);
                                    }}
                                }}
                                requestAnimationFrame(checkAndInsert);

                            }}
                        }});
                        /* ************** */

                        /* **********************
                        * "a task trace logged" *
                        ********************** */
                        executionEventsSource.addEventListener("taskTrace", (event) => {{
                            let payload;
                            try {{
                                // Parse the server data, assuming it's JSON
                                payload = JSON.parse(event.data);
                            }} catch (e) {{
                                console.error('Error parsing SSE message data:', event.data);
                                console.error(e);
                                return;
                            }}
                            //console.log("taskTrace", payload);
                            function checkAndUpdate() {{
                                /* *************************************
                                * Ensure the task details script is    *
                                * loaded before handling new events.   *
                                *                                      *
                                * BEWARE :                             *
                                *   This will lead to                  *
                                *   treating events twice              *
                                *   if they occur in the middle of the *
                                *   initial response.                  *
                                *   (which we can handle)              *
                                ************************************* */
                                if (insertTraceToTable) {{
                                    insertTraceToTable(payload);
                                }} else {{
                                    requestAnimationFrame(checkAndUpdate);
                                }}
                            }}
                            requestAnimationFrame(checkAndUpdate);
                        }});
                        /* ******************* */

                        /* ***************
                        * "a task ended" *
                        *************** */
                        executionEventsSource.addEventListener('taskEnded', (event) => {{
                            let payload;
                            try {{
                                // Parse the server data, assuming it's JSON
                                payload = JSON.parse(event.data);
                            }} catch (e) {{
                                console.error('Error parsing SSE message data:', event.data);
                                console.error(e);
                                return;
                            }}

                            if (payload.exec_id == {execution_id}) {{
                                //console.log("executionEventsSource 'taskEnded'", payload);

                                function checkAndUpdate() {{
                                    /* ***************************************
                                    * Ensure the init table state is         *
                                    * loaded before handling new events.     *
                                    *                                        *
                                    * BEWARE :                               *
                                    *   This will lead to                    *
                                    *   treating events twice                *
                                    *   if they occur in the middle of the   *
                                    *   initial response.                    *
                                    *   (which we can handle and,            *
                                    *    if we don't wait, the gantt object  *
                                    *    may not exists yet here)            *
                                    *************************************** */
                                    if (window.execGanttTimelineObj) {{
                                        ganttUpdate('execGanttTimelineObj', payload, interBarsSpacing);
                                    }} else {{
                                        requestAnimationFrame(checkAndUpdate);
                                    }}
                                }}
                                requestAnimationFrame(checkAndUpdate);

                            }}
                        }});
                        /* ************ */

                    }}
                    registerTaskEvents();
                    console.log("event-source listeners", listEventSourceListeners(executionEventsSource));
                """),

                Script(f"""// AnsiUp dependency (ascii to html)
                    import {{ AnsiUp }} from '{prefix}/ansi_up.js';
                    window.AnsiUp = AnsiUp;
                """, type="module"),
                Script(src="/task_details_modal.js"),
                Link(rel="stylesheet", href="/task_details_modal.css"),

                H1(# DAG
                    "Execution DAG",
                    style="""
                        color: #6082B6;
                        padding: 2rem 0 3rem;
                        margin-bottom: 0;
                        background-color: rgba(0, 0, 0, 0.03);
                        border-bottom: 1px solid rgba(0, 0, 0, 0.125);
                        text-align: center;
                    """
                ),
                Style(""" /* DAG */
                    #dag-docstring {
                        margin: 0 20px;
                        padding: 12px 12px 12px 16px;
                        min-height: 56px;
                        position: relative;
                        display: flex;
                        flex-direction: column;
                        align-items: flex-end;
                    }
                    #dag-docstring.showing {
                        background: linear-gradient(135deg, rgba(255, 255, 255, 0.1) 0%,
                                                    rgba(248, 249, 250, 0.1) 100%);
                        border: 1px solid rgba(222, 226, 230, 0.5);
                        border-radius: 10px;
                        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08),
                                    inset 0 1px 0 rgba(255, 255, 255, 0.8);
                        color: white;
                    }
                    #dag-docstring.collapsed {
                        height: 56px; /* min-height*/
                        overflow: hidden;
                    }
                    #dag-docstring.expanded {
                        max-height: 150px;
                        overflow-y: auto;
                    }

                    #dag-docstring .content {
                        flex: 1 1 auto;
                        min-width: 0;
                        width: 100%;
                        text-align: justify;
                    }

                    #dag-docstring-show-more {
                        position: sticky;
                        bottom: 0;
                        transform: translateY(10px);
                        background: rgba(0, 0, 0, 0.5);
                        color: white;
                        padding: 2px 8px;
                        border-radius: 5px;
                        cursor: pointer;
                        font-size: 13px;
                        white-space: nowrap; /* prevents wrapping */
                        width: auto;
                        max-width: fit-content;
                        z-index: 10;
                    }
                """),
                Div(# DAG renderer
                    # herein div will be async-dropped and
                    # tags from svg_template async-inserted to DOM here
                    P("\u00A0 Loading DAG...", style="color: white;"),
                    hx_get=f"{prefix}/dag_rendering?id={execution_id}",
                    hx_trigger="load",
                    hx_swap="outerHTML",
                    id="dag-anchor"
                ),
                Script(""" // hide pipeline-card button on no eponyme DAG task
                    function hasNoPipelineCard(svg) {
                        const texts = svg.querySelectorAll("g.node text.label");
                        return Array.from(
                                texts).every(t => t.textContent.trim() !== "pipeline_card"
                        );
                    }

                    function hidePipelineCard() {
                        const pipelineCard = document.getElementById("pipeline-card");
                        if (pipelineCard) {
                            pipelineCard.style.display = "none";
                        }
                    }

                    (function checkDagOnce() {
                        const anchor = document.getElementById("dag-anchor");
                        const svg = document.getElementById("dag");
                        //console.log("Initial check:", { anchor: !!anchor, svg: !!svg });

                        if (svg && hasNoPipelineCard(svg)) {
                            console.log("SVG ready, no pipeline_card");
                            hidePipelineCard();
                            return;
                        }

                        if (!anchor) {
                            // No anchor
                            hidePipelineCard();
                            return;
                        }

                        // Watch anchor's PARENT for anchor child removal + SVG creation
                        const parent = anchor.parentElement;
                        const observer = new MutationObserver(function(mutations) {
                            const newSvg = document.getElementById('dag-00000');
                            if (newSvg && hasNoPipelineCard(newSvg)) {
                                console.log(
                                    "SVG appeared after anchor vanished,",
                                    "no pipeline_card"
                                );
                                hidePipelineCard();
                                observer.disconnect();
                            }
                        });

                        observer.observe(parent, {
                            childList: true,
                            subtree: true
                        });
                    })();
                """),
                Link(rel="stylesheet", href=f"{prefix}/svg_dag.css")
            ),
            body_cls=["body-execution"]
        )

