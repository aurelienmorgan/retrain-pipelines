
import os
import asyncio

from typing import Optional, List
from fasthtml.common import H1, H2, Div, P, \
    Link, Script, Style, \
    Request, Response, JSONResponse, \
    StreamingResponse
from jinja2 import Environment, FileSystemLoader

from ...db.dao import AsyncDAO
from ..utils import ClientInfo
from .page_template import page_layout
from ....utils import get_text_pixel_width
from ..utils.execution.events import \
    new_exec_subscribers, exec_end_subscribers, \
    multiplexed_event_generator, execution_number


async def get_execution_elements_lists(
    execution_id: int
) -> Optional[List[str]]:
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
        execution_id = request.query_params.get("id")
        execution_number_response = await execution_number(execution_id)

        return execution_number_response


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
                            id="execution-number"
                        ),
                        "/",
                        Div(
                            id="executions-count"
                        ),
                        f", exec_id: {execution_id} -\u00A0",
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
                Script(f"""// async get execution-info
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
                            .then(function(execution_info) {{
                                // inform execution-name & subtitle
                                const executionName = document.getElementById("execution-name");
                                executionName.innerText = execution_info.name;
                                executionName.title = execution_info.username;
                                if (execution_info.docstring) {{
                                    const dagDocstring = document.getElementById("dag-docstring");
                                    const dagDocstringContentDiv = document.querySelector("#dag-docstring .content");
                                    dagDocstringContentDiv.innerText = execution_info.docstring;
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

                        // Calculating two lines of text height (safer: use computed styles)
                        const lineHeight = parseFloat(getComputedStyle(dagDocstringContentDiv).lineHeight);
                        const maxTwoLinesHeight = lineHeight * 2;
                        //console.log("checkDocstringOverflow", lineHeight, maxTwoLinesHeight, dagDocstringContentDiv.scrollHeight);

                        if (dagDocstringContentDiv.scrollHeight > maxTwoLinesHeight + 2) {
                            showMore.style.display = 'block';
                            dagDocstring.classList.add('collapsed');
                        } else {
                            showMore.style.display = 'none';
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
                    let executionEventSource;
                    function registerExecEventSrc() {{
                        executionEventSource = new EventSource(
                            `{prefix}/execution_events`
                        );

                        executionEventSource.onerror = (err) => {{
                            console.error('SSE error:', err);
                        }};
                        // Force close EventSource when leaving the page
                        window.addEventListener('pagehide', () => {{
                            executionEventSource.close();
                        }});

                        /* ************************
                        * "new execution started" *
                        ************************ */
                        // update executions count label
                        executionEventSource.addEventListener('newExecution', (event) => {{
                            let payload;
                            try {{
                                // Parse the server data, assuming it's JSON
                                payload = JSON.parse(event.data);
                            }} catch (e) {{
                                console.error('Error parsing SSE message data:', event.data);
                                console.error(e);
                                return;
                            }}
                            //console.log("executionEventSource 'newExecution'", payload);
                            if (payload.name === document.getElementById("execution-name").innerText) {{
                                executionNumberJson = payload;
                                executionNumberJson.number = document.getElementById("execution-number").innerText;
                                updateExecutionNumber(executionNumberJson);
                            }}
                        }});

                        /* *********************
                        * "an execution ended" *
                        ********************** */
                        // update executions count label and tooltip
                        executionEventSource.addEventListener('executionEnded', (event) => {{
                            let payload;
                            try {{
                                // Parse the server data, assuming it's JSON
                                payload = JSON.parse(event.data);
                            }} catch (e) {{
                                console.error('Error parsing SSE message data:', event.data);
                                console.error(e);
                                return;
                            }}
                            //console.log("executionEventSource 'executionEnded'", payload);
                            if (payload.name === document.getElementById("execution-name").innerText) {{
                                executionNumberJson = payload;
                                executionNumberJson.number = document.getElementById("execution-number").innerText;
                                updateExecutionNumber(executionNumberJson);
                            }}
                        }});

                    }}
                    registerExecEventSrc();
                """),
                Script("""// re-register SSE source on window history.back()
                    window.addEventListener('pageshow', function(event) {
                        if (event.persisted) {
                            registerExecEventSrc();
                        }
                    });
                """),
                Script("""// handling & recovering from server-loss
                    const statusCircle = document.getElementById('status-circle');
                    let previousClasses =
                        Array.from(statusCircle.classList); // remember initial classes
                    function onClassChange(mutationsList) {
                        for (let mutation of mutationsList) {
                            if (
                                mutation.type === 'attributes' &&
                                mutation.attributeName === 'class'
                            ) {
                                const newClasses = Array.from(statusCircle.classList);

                                if (
                                    newClasses.includes('disconnected') &&
                                    !previousClasses.includes('disconnected')
                                ) {
                                    console.log("disconnected");
                                    executionEventSource.close();
                                } else if (
                                    newClasses.includes('connected') &&
                                    !previousClasses.includes('connected')
                                ) {
                                    console.log("reconnected");
                                    registerExecEventSrc();
                                    document.dispatchEvent(
                                        new Event("DOMContentLoaded", { bubbles: true, cancelable: true }));
                                }

                                // Update previousClasses for next mutation check
                                previousClasses = newClasses;
                            }
                        }
                    }

                    const observer = new MutationObserver(onClassChange);
                    observer.observe(statusCircle, { attributes: true });
                """),
                Style("""
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

                    .body-execution {
                        padding-top: 3.5rem;
                        padding-bottom: 2.5rem;
                    }

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
                        background: linear-gradient(135deg, rgba(255, 255, 255, 0.1) 0%, rgba(248, 249, 250, 0.1) 100%);
                        border: 1px solid rgba(222, 226, 230, 0.5);
                        border-radius: 10px;
                        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08), inset 0 1px 0 rgba(255, 255, 255, 0.8);
                        color: white;
                    }
                    #dag-docstring.collapsed {
                        height: 56px; /* min-height*/
                        overflow: hidden;
                    }
                    #dag-docstring.expanded {
                        height: 150px;
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

                ## TODO, add stuff here (live-streamed Grantt diagram of tasks, etc.)

                H1(
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
                Div(# DAG renderer
                    P("\u00A0 Loading DAG...", style="color: white;"),
                    hx_get=f"{prefix}/dag_rendering?id={execution_id}",
                    hx_trigger="load",
                    hx_swap="outerHTML"
                ),
                Link(
                    rel="stylesheet",
                    href=f"{prefix}/svg_dag.css"
                )
            ),
            body_cls=["body-execution"]
        )

