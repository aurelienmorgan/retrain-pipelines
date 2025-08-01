
import os
import logging

from typing import Optional, Union, List
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
    get_pipeline_names, get_executions_ext
from ..utils.executions.events import \
    ClientInfo, ExecutionEnd, \
    new_exec_subscribers, new_exec_event_generator, \
    exec_end_subscribers, exec_end_event_generator
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

def MultiStatesToggler(
    options: List[Div],
    id: str,
    style: Optional[str]
) -> Div:
    # Add "bandit-toggle-label" to each option's class attribute
    options_with_cls = []
    for opt in options:
        print(opt.attrs)
        classes = (opt.attrs.get("class", "")
                   if hasattr(opt, "attrs") else opt.cls or "")
        class_list = set(classes.split()) if classes else set()
        class_list.add("bandit-toggle-label")
        new_opt = Div(
            *opt.children,
            cls=" ".join(class_list),
            **{k: v for k, v in (opt.attrs.items()
               if hasattr(opt, "attrs") else {}) if k != "class"}
        )
        options_with_cls.append(new_opt)

    return Div(
        Div(
            *options_with_cls,
            id="banditLabels",
            cls="bandit-toggle-labels"
        ),
        Style(f"""
            {style}
            .bandit-toggle-container {{
                display: inline-block;
                position: relative;
                cursor: pointer;
                user-select: none;
                font-family: 'Roboto', sans-serif;
                font-weight: bold;
                font-size: 8pt;
                letter-spacing: 2px;
                font-style: italic;
                line-height: 1.2;
                height: 1.2em;
                overflow: hidden;
                padding: 0 10px;
                background: rgba(255, 255, 255, 0.15);
                backdrop-filter: blur(8px);
                -webkit-backdrop-filter: blur(8px);
                border: 1px solid rgba(255, 255, 255, 0.3);
                border-radius: 10px;
                box-shadow: 0 4px 10px rgba(255, 255, 255, 0.2);
            }}
            .bandit-toggle-container:focus {{
                outline: 1px solid #4d0066;
            }}
            .bandit-toggle-labels {{
                position: relative;
                transition: transform 0.6s
                  cubic-bezier(0.68, -0.55, 0.265, 1.55);
            }}
            .bandit-toggle-label {{
                height: 1em;
                display: flex;
                align-items: center;
                justify-content: center;
                white-space: nowrap;
                color: #4d0066;

                text-shadow:
                    -0.7px 0 0 rgba(255, 255, 255, 0.6),
                    0.7px 0 0 rgba(255, 255, 255, 0.6),
                    0 -0.7px 0 rgba(255, 255, 255, 0.6),
                    0 0.7px 0 rgba(255, 255, 255, 0.6),
                    -0.7px -0.7px 0 rgba(255, 255, 255, 0.6),
                    0.7px -0.7px 0 rgba(255, 255, 255, 0.6),
                    -0.7px 0.7px 0 rgba(255, 255, 255, 0.6),
                    0.7px 0.7px 0 rgba(255, 255, 255, 0.6),
                    2px 2px 6px white;
            }}
            /*
             .all {{ color: #1976d2; }}
             .success {{ color: #30a030; }}
             .failure {{ color: #c52323; }}
            */
        """),
        Script(f"""// behavior
            (function(){{
                const container = document.getElementById('{id}');
                const labels = container.getElementsByClassName('bandit-toggle-labels')[0];

                const labelDivs = container.getElementsByClassName('bandit-toggle-label');
                window["_states_{id}"] = Array.from(labelDivs).map(div => {{
                    // array of dicts of form :
                    // {{ text: 'the text content',
                    //    class: 'the secondary class(es)'
                    //    data-*: 'the dataset entry value'
                    // }}

                    // Convert dataset DOMStringMap to a plain object
                    const dataset = {{}};
                    for (const [key, value] of Object.entries(div.dataset)) {{
                        dataset[key] = value;
                    }}

                    return {{
                        text: div.textContent.trim(),
                        class: Array.from(div.classList)
                                .find(c => c !== 'bandit-toggle-label'),
                        dataset: dataset
                    }};
                }});

                 let currentIndex = -1;
                 let isAnimating = false;

                function addNextLabel() {{
                    // put last active label back at the bottom of the list
                    const nextIndex = (currentIndex + 1) % window["_states_{id}"].length;
                    const nextState = window["_states_{id}"][nextIndex];
                    const newLabel = document.createElement('div');
                    newLabel.className = `bandit-toggle-label ${{nextState.class}}`;
                    newLabel.textContent = nextState.text;

                    // Add all dataset entries as data-* attributes
                    for (const [key, value] of Object.entries(nextState.dataset || {{}})) {{
                        // Convert camelCase key to dash-case data attribute name
                        const dataAttrName = 'data-' + key.replace(/[A-Z]/g, m => '-' + m.toLowerCase());
                        newLabel.setAttribute(dataAttrName, value);
                    }}

                    labels.appendChild(newLabel);
                }}

                function triggerStateChange() {{
                    if (isAnimating) return;
                    isAnimating = true;

                    addNextLabel();

                    labels.style.transition =
                        'transform 0.6s cubic-bezier(0.68, -0.55, 0.265, 1.55)';
                    const height = container.offsetHeight;
                    labels.style.transform = `translateY(-${{height}}px)`;

                    setTimeout(() => {{
                        labels.style.transition = 'none';
                        labels.removeChild(labels.firstElementChild);
                        labels.style.transform = 'translateY(0px)';
                        currentIndex = (currentIndex + 1) % window["_states_{id}"].length;
                        isAnimating = false;
                    }}, 600);
                }}

                container.addEventListener('click', (event) => {{
                    container.focus();
                    triggerStateChange();
                }});

                container.addEventListener('keydown', (event) => {{
                    if (event.code === 'Space' || event.key === ' ') {{
                        event.preventDefault();
                        triggerStateChange();
                    }}
                }});

            }})();
        """),
        id=id,
        tabindex="0",
        cls="bandit-toggle-container"
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
        },
        category="Executions")
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
        client_info = ClientInfo(
            ip=request.client.host,
            port=request.client.port,
            url=request.url.path
        )
        return StreamingResponse(
            new_exec_event_generator(client_info=client_info),
            media_type="text/event-stream")
        users = await get_users()
        return JSONResponse(users)


    @rt_api(
        rt, url= f"{prefix}/api/v1/execution_end_event",
        methods=["POST"],
        schema={
            "requestBody": {
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "integer"},
                                "end_timestamp": {
                                    "type": "string",
                                    "format": "date-time"
                                },
                                "success": {"type": "boolean"},
                            },
                            "required": ["id", "end_timestamp",
                                         "success"]
                        }
                    }
                }
            },
            "responses": {
                "200": {"description": "OK"},
                "422": {"description": "Invalid input"}
            }
        },
        category="Executions")
    async def post_execution_ended_event(
        request: Request
    ):
        """DAG-engine notifies of a pipeline execution end."""
        data = await request.json()

        # validate posted data
        try:
            execution_end = ExecutionEnd(data)
        except (KeyError, ValueError, TypeError) as e:
            logging.getLogger().warn(e)
            return Response(status_code=422,
                            content=f"Invalid input: {str(e)}")

        # dispatch 'new Execution' event
        for q, _ in exec_end_subscribers:
            await q.put(data)

        return Response(status_code=200)


    @rt(f"{prefix}/pipeline_exec_end_event", methods=["GET"])
    async def get_pipeline_exec_end_event(request: Request):
        client_info = ClientInfo(
            ip=request.client.host,
            port=request.client.port,
            url=request.url.path
        )
        return StreamingResponse(
            exec_end_event_generator(client_info=client_info),
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

        execution_entries = await get_executions_ext(
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
                            MultiStatesToggler(# executions status filter
                                options=[
                                    Div("All Statuses", cls="all"),
                                    Div("Successes", **{"data-execs-status": "success"}, cls="success"),
                                    Div("Failures", **{"data-execs-status": "failure"}, cls="failure")
                                ],
                                id="status-bandit-toggle",
                                style="""
                                    #status-bandit-toggle {
                                        position: absolute;
                                        bottom: 2px;
                                        right: 3px;
                                    }
                                """
                            ),
                            Script("""// MultiStatesToggler cookies
                                const container = document.getElementById('status-bandit-toggle');
                                const labels = container.getElementsByClassName('bandit-toggle-labels')[0];

                                // Helper function to set a cookie
                                function setCookie(name, label) {
                                    if (!label.dataset ||
                                        Object.keys(label.dataset).length === 0
                                    ) {
                                        // If dataset is empty, remove the cookie
                                        // by setting its expiry in the past.
                                        document.cookie = `${name}=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;`;
                                    } else {
                                        const expires = new Date(Date.now() + 50*365*24*60*60*1000).toUTCString();
                                        document.cookie = `${name}=${label.dataset.execsStatus}; expires=${expires}; path=/`;
                                    }
                                }

                                const COOKIE_PREFIX = "executions_dashboard:";
                                const cookieKey = COOKIE_PREFIX + 'status-bandit-toggle';

                                container.addEventListener('click', (event) => {
                                    setCookie(cookieKey, labels.children[1]);
                                });

                                container.addEventListener('keydown', (event) => {
                                    if (event.code === 'Space' || event.key === ' ') {
                                        setCookie(cookieKey, labels.children[1]);
                                    }
                                });

                                document.addEventListener("DOMContentLoaded", function() {
                                    // intiale value from cookie
                                    const cookies = document.cookie.split("; ");
                                    for (const cookie of cookies) {
                                        const [key, val] = cookie.split("=");
                                        if (key === cookieKey) {
                                            const execsStatus = decodeURIComponent(val||"");

                                            // removing from top of labels
                                            for (let i = 0; i < labels.childNodes.length; ) {
                                                const child = labels.childNodes[i];
                                                if (child.nodeType !== 3 && (!child.dataset ||
                                                    Object.keys(child.dataset).length === 0 ||
                                                    child.dataset.execsStatus != execsStatus)
                                                ) {
                                                    labels.removeChild(child);
                                                    // do not increment i because childNodes collection updates
                                                } else if (
                                                    child.dataset &&
                                                    Object.keys(child.dataset).length > 0 &&
                                                    child.dataset.execsStatus === execsStatus
                                                ) {
                                                    break;
                                                } else {
                                                    i++;  // only increment if no removal happened
                                                }
                                            }

                                            // appending at bottom of labels
                                            // and moving states up the states list
                                            // so currentIndex points to the right entry
                                            for (let i = 0; i < window["_states_status-bandit-toggle"].length; i++) {
                                                const state = window["_states_status-bandit-toggle"][i];
                                                if (
                                                    Object.keys(state.dataset).length > 0 &&
                                                    state.dataset.execsStatus === execsStatus
                                                ) {
                                                    // Remove all states before the current one (index i)
                                                    const before = window["_states_status-bandit-toggle"].splice(0, i);
                                                    // Append those removed states to the end to keep their order
                                                    window["_states_status-bandit-toggle"].push(...before);
                                                    break;
                                                } else {
                                                    const newLabel = document.createElement('div');
                                                    newLabel.className = `bandit-toggle-label ${state.class}`;
                                                    newLabel.textContent = state.text;

                                                    // Add all dataset entries as data-* attributes
                                                    for (const [key, value] of Object.entries(state.dataset || {})) {
                                                        // Convert camelCase key to dash-case data attribute name
                                                        const dataAttrName = 
                                                            'data-' + key.replace(/[A-Z]/g, m => '-' + m.toLowerCase());
                                                        newLabel.setAttribute(dataAttrName, value);
                                                    }

                                                    labels.appendChild(newLabel);
                                                }
                                            }
                                            // console.log("labels", labels);
                                        }
                                    }
                                });
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
                                    const newExecutionStart = new Date(payload.start_timestamp);
                                    const template = document.createElement('template');
                                    template.innerHTML = payload.html.trim();
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

function formatTimeDelta(start, end) {{
    const startDate = new Date(start);
    const endDate = new Date(end);

    const diffMs = endDate - startDate;
    if (isNaN(diffMs)) {{
        throw new Error("Invalid date(s) provided");
    }}

    const totalSeconds = Math.floor(diffMs / 1000);
    const milliseconds = diffMs % 1000;
    const seconds = totalSeconds % 60;
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const hours = Math.floor(totalSeconds / 3600);

    return `${{hours}}:${{String(minutes).padStart(2, '0')}}:${{String(seconds).padStart(2, '0')}}.${{String(milliseconds).padStart(6, '0')}}`;
}}

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
const executionElement = document.getElementById(payload.id);
if (executionElement) {{
    //console.log(executionElement);
    const endTimestampElement = executionElement.querySelector('.end_timestamp');
    console.log(endTimestampElement);
    console.log(executionElement.dataset.startTimestamp);
    console.log(payload.end_timestamp);
    endTimestampElement.innerText = formatTimeDelta(
        executionElement.dataset.startTimestamp, payload.end_timestamp
    );
    endTimestampElement.classList.add(payload.success ? 'success' : 'failure');
}}
                                }}
                            """),
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
                                "margin-left: auto; "
                                "height: 48px;"
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
                    const execContainer = document.getElementById("executions-container");
                    function loadExecs() {
                        const server_status_circle = document.getElementById('status-circle');
                        server_status_circle.classList.add('spinning');

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

