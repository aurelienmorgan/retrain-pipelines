
import os
import logging

from typing import Optional, Union
from datetime import datetime, timezone
from email.utils import formatdate, \
    parsedate_to_datetime
from fasthtml.common import Div, H1, H3, P, \
    Span, Code, Input, Script, Style, \
    Request, Response, FileResponse, JSONResponse, \
    StreamingResponse

from .. import APP_STATIC_DIR
from .page_template import page_layout

from ..utils.executions import get_users, \
    get_pipeline_names, get_executions
from ..utils.executions.events import \
    new_exec_subscribers, new_exec_event_generator
from ...db.model import Execution
from ..views.api import rt_api

def AutoCompleteSelect(
    options_url: str,
    id: str,
    placeholder: Optional[str] = "",
    js_callback: Optional[str] = "",
    style: Optional[str] = ""
) -> Div:
    """A DOM element of type auto-complete combobox

    Initializes from api endpoint
    that returns list of dropdown item strings.

    Params:
        - options_url (str):
            the url to the endpoint
            serving the list of dropdown values
        - id (str):
            the id of the DOM element
        - placeholder (optional, str):
            the placeholder of the textfield
        - js_callback (optional, str):
            the callback event listener js code
        - style (optional, str):
            custom css
    Results:
        - (Div)
    """
    input_id = f"{id}-input"
    dropdown_id = f"{id}-dropdown"

    id = id.replace("-", "_")
    just_sel_flag = f"_justSelected_{id}"

    return Div(
        Div(
            Input(
                id=input_id,
                type="text",
                placeholder=placeholder,
                value="",
                autocomplete="off",
                spellcheck="false",
                cls="combo-input",
                _onfocus=f"window.g_onFocus_{id}()",
                _oninput=f"window.g_filterDropdown_{id}(this.value)",
                _onkeydown=f"window.g_dropdownKey_{id}(event)",
                _onkeyup=f"if(event.key === 'Enter'){{ {js_callback} }}",
                _onblur=f"window.g_delayedHideDropdown_{id}()",
            ),
            Div(# container of .combo-option items
                id=dropdown_id,
                tabindex="-1", # so it doesn't get focusable when with scrollbars
                cls="combo-dropdown"
            ),
            cls="combo-root"
        ),
        Script(f"""// dropdown, autocomplete, validate input
            (function(){{
                var state = {{
                    active: -1,
                    blurTimer: null,
                }};
                window.{just_sel_flag} = false;

                window.g_onFocus_{id} = function() {{
                    if(window.{just_sel_flag}) {{
                        window.{just_sel_flag} = false;
                        return;
                    }}
                    var dd = document.getElementById('{dropdown_id}');
                    if(dd) dd.classList.add('open');
                    state.active = -1;
                    if(state.blurTimer) {{
                        clearTimeout(state.blurTimer); state.blurTimer = null;
                    }}
                }};
                window.g_delayedHideDropdown_{id} = function() {{
                    state.blurTimer = setTimeout(function() {{
                        var dd = document.getElementById('{dropdown_id}');
                        // prevent blur on dropdown scrollbar drag
                        if (!dd.contains(document.activeElement)) {{
                            if(dd) dd.classList.remove('open');
                            state.active = -1;
                            if(dd) {{
                                for(let c of dd.children) {{
                                    c.classList.remove('keyboard-active');
                                }}
                            }}
                        }}
                    }}, 110);
                }};
                window.g_onMouseSelect_{id} = function(val) {{
                    window.{just_sel_flag} = true;
                    window.g_selectDropdownOption_{id}(val, true);
                }};
                window.g_filterDropdown_{id} = function(substr) {{
                    var dd = document.getElementById('{dropdown_id}');
                    if(!dd) return;
                    substr = substr.toLowerCase();
                    var found = 0;
                    var opts = dd.children;
                    for(let el of opts) {{
                        el.style.display = el.textContent.toLowerCase()
                                            .includes(substr)
                                            ? '' : 'none';
                        if (el.style.display === "") found += 1;
                        el.classList.remove('keyboard-active');
                    }}
                    dd.classList.add('open');
                    state.active = -1;
                }};
                window.g_selectDropdownOption_{id} = function(val, giveFocus) {{
                    var input = document.getElementById('{input_id}');
                    var dd = document.getElementById('{dropdown_id}');
                    if(input) {{
                        input.value = val;
                        input.dispatchEvent(new Event('input'));
                        if(giveFocus) {{
                            setTimeout(function() {{
                                input.focus();
                                var len = input.value.length;
                                input.setSelectionRange(len, len);
                                // fire 'onkeyup' event (trigger 'js_callback')
                                input.dispatchEvent(
                                    new KeyboardEvent("keyup", {{
                                      key: "Enter", code: "Enter",
                                      keyCode: 13, which: 13,
                                      bubbles: true, cancelable: true
                                    }})
                                );
                            }}, 0);
                        }}
                    }}
                    if(dd) dd.classList.remove('open');
                    state.active = -1;
                }};
                window.g_dropdownKey_{id} = function(ev) {{
                    var input = document.getElementById('{input_id}');
                    var dd = document.getElementById('{dropdown_id}');
                    var wasOpen = dd && dd.classList.contains('open');
                    var opts = dd ? dd.children : [];
                    var visible = [];
                    for(let i=0; i < opts.length; ++i)
                        if(opts[i].style.display !== "none") visible.push(opts[i]);
                    if(ev.key == 'ArrowDown') {{
                        if (!wasOpen) {{
                            if(dd) dd.classList.add('open');
                            // focus the first visible option
                            if (visible.length>0) {{
                                for(let c of visible) {{
                                    c.classList.remove('keyboard-active');
                                }}
                                visible[0].classList.add('keyboard-active');
                                visible[0].scrollIntoView({{behavior: 'smooth', block: 'center'}});
                                state.active = 0;
                            }}
                            ev.preventDefault();
                            return;
                        }}
                    }}
                    if(!wasOpen || visible.length == 0) return;

                    let active = state.active;

                    if(ev.key == 'ArrowDown') {{
                        ev.preventDefault();
                        if (active < visible.length-1) active += 1;
                        else active = 0;
                        highlight();
                    }} else if(ev.key == 'ArrowUp') {{
                        ev.preventDefault();
                        if (active > 0) active -= 1;
                        else active = visible.length-1;
                        highlight();
                    }} else if(ev.key == 'Enter') {{
                        if (active >=0 && active < visible.length) {{
                            ev.preventDefault();
                            window.g_selectDropdownOption_{id}(
                                visible[active].textContent, true
                            );
                            state.active = -1;
                            if(dd) dd.classList.remove('open');
                        }}
                    }} else if(ev.key == 'Escape') {{
                        ev.preventDefault();
                        if(dd) dd.classList.remove('open');
                        state.active = -1;
                    }}

                    function highlight() {{
                        for(const v of visible) v.classList.remove('keyboard-active');
                        if(active >= 0 && active < visible.length)
                            visible[active].classList.add('keyboard-active');
                        state.active = active;
                        let el = visible[active];
                        if(el && typeof el.scrollIntoView === "function")
                            el.scrollIntoView({{behavior: 'smooth', block: 'center'}});
                    }}
                }};
                window.g_hoverDropdownOption_{id} = function(idx) {{
                    var dd = document.getElementById('{dropdown_id}');
                    if(!dd) return;
                    for(let c of dd.children) c.classList.remove('keyboard-active');
                    var el = document.getElementById('{dropdown_id}_opt_' + idx);
                    if(el) {{
                        el.classList.add('keyboard-active');
                    }}
                    state.active = -1;
                }};
                window.g_unhoverDropdownOption_{id} = function(idx) {{
                    var el = document.getElementById('{dropdown_id}_opt_' + idx);
                    if(el) el.classList.remove('keyboard-active');
                }};
            }})();
        """),
        Script(f"""// 3-states management
            // (draft, applied & avaialble, applied & not-available)
            (function(){{
                var input = document.getElementById('{input_id}');
                var dd = document.getElementById('{dropdown_id}');
                var selected = false;

                // Helper function to set a cookie
                function setCookie(name, value) {{
                    const expires = new Date(Date.now() + 50*365*24*60*60*1000).toUTCString();
                        document.cookie = `${{name}}=${{value}}; expires=${{expires}}; path=/`;
                }}
                const COOKIE_PREFIX = "executions_dashboard:";

                const resizeObserver = new ResizeObserver((entries) => {{
                    entries.forEach(entry => {{
                        option = entry.target;
                        // if entry label is too long to fit, ellipsis is applied
                        // we add the whole value in contextual popup
                        if (option.scrollWidth > option.clientWidth)
                            option.title = option.textContent.trim();
                    }});
                }});

                function renderOptions(list, term) {{
                    dd.innerHTML = "";
                    list.forEach(function(opt, i) {{
                        if (!term || opt.toLowerCase().includes(term.toLowerCase())) {{
                            var item = document.createElement('div');
                            item.textContent = opt;
                            item.className = 'combo-option';
                            item.id = '{dropdown_id}_opt_' + i;
                            item.onmousedown = function() {{
                                window.g_onMouseSelect_{id}(opt);
                                selected = true;
                                input.classList.remove('combo-input-unselected');
                                input.classList.remove('combo-input-selected-red');
                                const cookieKey = COOKIE_PREFIX + '{id.replace("_", "-")}';
                                setCookie(cookieKey, input.value);
                            }};
                            item.onmouseover = function() {{
                                window.g_hoverDropdownOption_{id}(i);
                            }};
                            item.onmouseout = function() {{
                                window.g_unhoverDropdownOption_{id}(i);
                            }};
                            resizeObserver.observe(item);
                            dd.appendChild(item);
                        }}
                    }});
                }}

                input.addEventListener('input', function() {{
                    input.classList.remove('combo-input-selected-red');
                    input.classList.add('combo-input-unselected');
                    var term = input.value;
                    selected = false;
                    var opts = window["_options_{id}"] || [];
                    renderOptions(opts, term);

                    input.title = input.value;
                }});

                input.addEventListener('keydown', function(e) {{
                    if (e.key === 'Enter') {{
                        selected = true;

                        input.classList.remove('combo-input-unselected');
                        var val = input.value.trim();
                        var opts = window["_options_{id}"] || [];
                        if (val > "" && opts.indexOf(val) === -1) {{
                            input.classList.add('combo-input-selected-red');
                        }} else {{
                            input.classList.remove('combo-input-selected-red');
                            const cookieKey = COOKIE_PREFIX + '{id.replace("_", "-")}';
                            setCookie(cookieKey, val);
                        }}
                    }}
                }});

                dd.addEventListener('mouseup', function(e) {{
                    // to prevent focus going to dropdown
                    // on user scrollbar mouse drag
                    input.focus();
                }});

                // arrays with content-change listener
                function createObservableArray(arr, onChange) {{
                    return new Proxy(arr, {{
                        set(target, property, value) {{
                            // Ignore length property changes to avoid too noisy logs
                            const res = Reflect.set(target, property, value);
                            if (property !== 'length') {{
                                onChange();
                            }}
                            return res;
                        }},
                        deleteProperty(target, property) {{
                            const res = Reflect.deleteProperty(target, property);
                            onChange();
                            return res;
                        }}
                    }});
                }}
                document.addEventListener("DOMContentLoaded", function() {{
                    fetch("{options_url}", {{
                        method: 'GET',
                        headers: {{ "HX-Request": "true" }}
                    }})
                    .then(function(resp) {{ return resp.json(); }})
                    .then(function(list) {{
                        // intiale value from cookie
                        const cookieKey = COOKIE_PREFIX + '{id.replace("_", "-")}';
                        const cookies = document.cookie.split("; ");
                        input.value = "";
                        for (const cookie of cookies) {{
                            const [key, val] = cookie.split("=");
                            if (key === cookieKey) {{
                                input.value = decodeURIComponent(val||"");
                            }}
                        }}
                        input.title = input.value;
                        // cascade to dropdown behavior
                        window["_options_{id}"] = createObservableArray(
                            list,
                            // re-render on options-list change
                            () => renderOptions(window["_options_{id}"], input.value)
                        );
                        renderOptions(list, input.value);

                        selected = true;
                    }});
                }});

            }})();
        """),
        id=id.replace("_", "-"),
        style=style
    )

def FilterElement(
    label: str,
    *elements: Union[Div, Script, Style],
    label_shadow_color: Optional[str] = None
) -> Div:
    """Element with overlaying label on top left corner."""
    return Div(
            Span(
                label,
                style=("""
                  position: absolute;
                  top: -1em;
                  left: -0.2em;
                  pointer-events: none;
                  color: white;
                  text-shadow:
                    -1px -1px 0 var(--label-shadow-color),
                     1px -1px 0 var(--label-shadow-color),
                    -1px  1px 0 var(--label-shadow-color),
                     1px  1px 0 var(--label-shadow-color),
                     0   -1px 0 var(--label-shadow-color),
                    -1px  0   0 var(--label-shadow-color),
                     1px  0   0 var(--label-shadow-color),
                     0    1px 0 var(--label-shadow-color);
                    z-index: 999; 
                """)
            ),
            elements,
            style=(
                "position: relative; "
                f"--label-shadow-color: {label_shadow_color};"
            )
        )

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
        headers = {
            "Last-Modified": last_modified,
            "Cache-Control": "public" # "no-cache" to force
                                      # browser revalidation
                                      # on each request
        }

        # Check If-Modified-Since header
        if_modified_since = request.headers.get("if-modified-since")
        if if_modified_since:
            since_dt = parsedate_to_datetime(if_modified_since)
            file_dt = datetime.utcfromtimestamp(stat.st_mtime) \
                        .replace(tzinfo=timezone.utc) \
                        .replace(microsecond=0)
            if file_dt <= since_dt:
                return Response(status_code=304, headers=headers)

        return FileResponse(file_fullname, headers=headers)


    @rt(f"{prefix}/distinct_pipeline_names", methods=["GET"])
    async def get_distinct_pipeline_names():
        pipeline_names = await get_pipeline_names()
        return JSONResponse(pipeline_names)


    @rt(f"{prefix}/distinct_users", methods=["GET"])
    async def get_distinct_users():
        users = await get_users()
        return JSONResponse(users)


    @rt_api(
        rt, url= f"{prefix}/api/v1/new_execution_event",
        methods=["POST"],
        schema={
            "requestBody": {
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "integer"},
                                "name": {"type": "string"},
                                "username": {"type": "string"},
                                "start_timestamp": {
                                    "type": "string",
                                    "format": "date-time"
                                }
                            },
                            "required": ["id", "name", "username",
                                         "start_timestamp"]
                        }
                    }
                }
            },
            "responses": {
                "200": {"description": "OK"},
                "422": {"description": "Invalid input"}
            }
        })
    async def post_new_execution_event(
        request: Request
    ):
        """DAG-engine notifies of a new pipeline execution."""
        data = await request.json()

        # validate posted data
        try:
            execution = Execution(data)
        except (KeyError, ValueError, TypeError) as e:
            logging.getLogger().warn(e)
            return Response(status_code=422,
                            content=f"Invalid input: {str(e)}")

        # dispatch 'new Execution' event
        for q, _ in new_exec_subscribers:
            await q.put(data)

        return Response(status_code=200)


    @rt(f"{prefix}/new_pipeline_exec_event", methods=["GET"])
    async def get_new_pipeline_exec_event(request: Request):
        client_info = {
            "ip": request.client.host,
            "port": request.client.port,
            "url": request.url.path
        }
        return StreamingResponse(
            new_exec_event_generator(client_info=client_info),
            media_type="text/event-stream")


    @rt(f"{prefix}/load_executions", methods=["POST"])
    async def get_execution_entries(
        request: Request
    ):
        # Retrieves params from form data (POST)
        form = await request.form()
        print(form)

        before_datetime_str = form.get("before_datetime")
        if before_datetime_str:
            try:
                before_datetime = datetime.strptime(
                    before_datetime_str[:33],
                    "%a %b %d %Y %H:%M:%S GMT%z"
                )
                # Convert to UTC (timezone used by the DAG engine)
                before_datetime = \
                    before_datetime.astimezone(timezone.utc)
            except Exception as e:
                print(e)
                before_datetime = None
        else:
            before_datetime = None

        pipeline_name = form.get("pipeline_name") or None
        username = form.get("username") or None
        n = form.get("n") or None

        execution_entries = await get_executions(
            pipeline_name=pipeline_name,
            username=username,
            before_datetime=before_datetime,
            n=n,
            descending=True
        )

        return execution_entries


    @rt(f"{prefix}/")
    def home():
        content = (
            H1("Placeholder", style="color: white;"),
            P(Code("retrain-pipelines"), " executions!",
              style="color: white;")
        )

        return page_layout(current_page="/", title="retrain-pipelines", \
            content=Div(# page content
                Div(# params panel
                    Div(
                        Div(
                            H3(
                                "Latest pipeline executions",
                                style=(
                                    "color: white; margin: 0; white-space: nowrap; "
                                    "font-size: 16px; line-height: 1.5;"
                                )
                            ),
                            style="display: flex; align-items: baseline;"
                        ),
                        Div(
                            FilterElement(# pipeline-name filter
                                "pipeline",
                                Div(
                                    AutoCompleteSelect(
                                        options_url=f"{prefix}/distinct_pipeline_names",
                                        id="pipeline-name-autocomplete",
                                        placeholder="select or type...",
                                        js_callback="loadExecs();",
                                        style=(
                                            "margin-right: 4px; "
                                            "scrollbar-width: thin; "
                                            "scrollbar-color: #4d0066 #FFFFCC20;"
                                        )
                                    ),
                                    style="min-width: 50px; min-height: 18px;"
                                ),
                                label_shadow_color="rgba(77, 0, 102, .7)"
                            ),
                            FilterElement(# username filter
                                "user",
                                Div(
                                    AutoCompleteSelect(
                                        options_url=f"{prefix}/distinct_users",
                                        id="pipeline-user-autocomplete",
                                        placeholder="select or type...",
                                        js_callback="loadExecs();",
                                        style=(
                                            "margin-right: 4px; "
                                            "scrollbar-width: thin; "
                                            "scrollbar-color: #4d0066 #FFFFCC20;"
                                        )
                                    ),
                                    style="min-width: 50px; min-height: 18px;"
                                ),
                                label_shadow_color="rgba(77, 0, 102, .7)"
                            ),
                            Style("""
                                #pipeline-name-autocomplete-input.combo-input {
                                    min-width: 130px; width: 130px;
                                }
                                #pipeline-user-autocomplete-input.combo-input {
                                    min-width: 100px; width: 100px;
                                }
                            """),
                            Script(f"""// SSE events : new retraining-pipeline execution
                                const newExecEventSource = new EventSource(
                                    `{prefix}/new_pipeline_exec_event`
                                );

                                newExecEventSource.onerror = (err) => {{
                                    console.error('SSE error:', err);
                                }};
                                // Force close EventSource when leaving the page
                                window.addEventListener('pagehide', () => {{
                                    newExecEventSource.close();
                                }});

                                // Listen for incoming messages from the server
                                newExecEventSource.onmessage = (event) => {{
                                    let payload;
                                    try {{
                                        // Parse the server data, assuming it's JSON
                                        payload = JSON.parse(event.data);
                                    }} catch (e) {{
                                        console.error('Error parsing SSE message data:', e);
                                        return;
                                    }}

                                    /* ****************************************************
                                    * Add "newExecutionElement" to "executions-container" *
                                    **************************************************** */
                                    const execContainer = document.getElementById("executions-container");
                                    const newExecutionStart = new Date(payload['start_timestamp']);
                                    const template = document.createElement('template');
                                    template.innerHTML = payload['html'].trim();
                                    const newExecutionElement = template.content.firstElementChild;

                                    // Find where to insert the newExecutionElement
                                    // assuming async from different seeders may occur
                                    let inserted = false;
                                    const children = execContainer.getElementsByClassName('execution');
                                    for (let i = 0; i < children.length; ++i) {{
                                        const existingDiv = children[i];
                                        const existingStartTS = existingDiv.dataset.startTimestamp;
                                        if (!existingStartTS) continue;
                                        const existingStart = new Date(existingStartTS);

                                        // Descending: insert before the first older item
                                        if (newExecutionStart > existingStart) {{
                                            execContainer.insertBefore(newExecutionElement, existingDiv);
                                            inserted = true;
                                            break;
                                        }}
                                    }}
                                    // If not inserted anywhere
                                    // (all items are newer or container is empty),
                                    // append at the end
                                    if (!inserted) {{
                                        execContainer.appendChild(newExecutionElement);
                                    }}
                                    /* ************************************************* */

                                    /* ***********************************************
                                    * Keep autocomplete comboboxes dropdowns in sync *
                                    *********************************************** */
                                    // Helper to add and keep list sorted without duplicates
                                    function addAndSortUnique(arr, newItem) {{
                                        if (!newItem) return arr;
                                        // Only add if not present
                                        if (!arr.includes(newItem)) {{
                                            arr.push(newItem);
                                            arr.sort((a, b) => a.localeCompare(b));
                                        }}
                                        return arr;
                                    }}
                                    window["_options_pipeline_name_autocomplete"] = 
                                        addAndSortUnique(
                                            window["_options_pipeline_name_autocomplete"] || [],
                                            payload.name
                                        );
                                    window["_options_pipeline_user_autocomplete"] = 
                                        addAndSortUnique(
                                            window["_options_pipeline_user_autocomplete"] || [],
                                            payload.username
                                        );
                                    /* ************************************************* */
                                }};
                            """),
                            Script(f"""// SSE events : retraining-pipeline execution ended
                                const execEndEventSource = new EventSource(
                                    `{prefix}/pipeline_exec_end_event`
                                );

                                execEndEventSource.onerror = (err) => {{
                                    console.error('SSE error:', err);
                                }};
                                // Force close EventSource when leaving the page
                                window.addEventListener('pagehide', () => {{
                                    execEndEventSource.close();
                                }});

                                // Listen for incoming messages from the server
                                execEndEventSource.onmessage = (event) => {{
                                    let payload;
                                    try {{
                                        // Parse the server data, assuming it's JSON
                                        payload = JSON.parse(event.data);
                                    }} catch (e) {{
                                        console.error('Error parsing SSE message data:', e);
                                        return;
                                    }}
console.log(payload);
                                }}
                            """),
                            FilterElement(# datetime filter
                                "before",
                                Div(
                                    id="pipeline-before-datetime", # picker container
                                    callback="loadExecs();",
                                    style=(
                                        "min-width: 60px; min-height: 18px; "
                                        "--shadow-color: rgba(77, 0, 102, .3);"
                                    )
                                ),
                                Script(
                                    """
                                        import { attachDateTimePicker } from './datetime-picker.js';
                                        attachDateTimePicker(
                                            'pipeline-before-datetime',
                                            {COOKIE_PREFIX: 'executions_dashboard:'}
                                        );
                                    """,
                                    type="module"
                                ),
                                label_shadow_color="rgba(77, 0, 102, .7)"
                            ),
                            id="params_panel",
                            style=(
                                "position: relative;"
                                "display: flex; "
                                "align-items: baseline; "
                                "padding: 12px 12px 12px 16px; "
                                "background: linear-gradient(135deg, "
                                    "rgba(255,255,255,0.1) 0%, "
                                    "rgba(248,249,250,0.1) 100%); "
                                "border: 1px solid rgba(222,226,230,0.5); "
                                "border-radius: 10px; "
                                "box-shadow: 0 2px 8px rgba(0,0,0,0.08), "
                                    "inset 0 1px 0 rgba(255,255,255,0.8); "
                                "color: white; "
                                "margin-left: auto;"
                            )
                        ),
                        style=(
                            "display: flex; align-items: baseline; margin-bottom: 8px;"
                        )
                    )
                ),
                Div(# Actual list
                    id="executions-container",
                    style=(
                        "height: calc(100vh - 200px); " # window height minus header & footer
                        "overflow-y: auto; padding: 8px 16px 4px 16px; "
                        "background: linear-gradient(135deg, "
                            "rgba(255,255,255,0.05) 0%, "
                            "rgba(248,249,250,0.05) 100%); "
                        "border: 1px solid rgba(222,226,230,0.6); "
                        "border-radius: 8px; "
                        "box-shadow: inset 0 2px 4px rgba(0,0,0,0.05), "
                            "0 1px 3px rgba(0,0,0,0.1); "
                    )
                ),
                Style("""
                    .execution {
                      display: flex;
                      justify-content: space-between;
                      align-items: center;
                      width: 100%;
                    }
                """),
                Script("""// Cold start of executions list at page load time
                    function loadExecs() {
                        const server_status_circle = document.getElementById('status-circle');
                        server_status_circle.classList.add('spinning');

                        const execContainer = document.getElementById("executions-container");
                        execContainer.innerHTML = '';

                        // retrieve last validated comboboxes value from cookie
                        const cookies = document.cookie.split("; ");
                        const COOKIE_PREFIX = "executions_dashboard:";

                        pipeline_name = "";
                        const pipeline_name_cookieKey = COOKIE_PREFIX + 'pipeline-name-autocomplete';
                        for (const cookie of cookies) {{
                            const [key, val] = cookie.split("=");
                            if (key === pipeline_name_cookieKey) {{
                                pipeline_name = decodeURIComponent(val||"");
                            }}
                        }}

                        username = "";
                        const username_cookieKey = COOKIE_PREFIX + 'pipeline-user-autocomplete';
                        for (const cookie of cookies) {{
                            const [key, val] = cookie.split("=");
                            if (key === username_cookieKey) {{
                                username = decodeURIComponent(val||"");
                            }}
                        }}

                        // retireve last validated datetime value from hidden input
                        before_datetime_str = document.getElementById(
                            "pipeline-before-datetime-selected").value;

                        // form data for the html POST
                        const formData = new FormData();
                        if (pipeline_name != "")
                            formData.append('pipeline_name', pipeline_name);
                        if (username != "")
                            formData.append('username', username);
                        if (before_datetime_str != "")
                            formData.append('before_datetime',
                                            new Date(before_datetime_str));
                        const batch_executions_count = 75;
                        formData.append('n', batch_executions_count);

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

