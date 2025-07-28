
import os

from typing import Optional, Union

from datetime import datetime, timezone
from email.utils import formatdate, \
    parsedate_to_datetime
from fasthtml.common import Div, H1, H3, P, \
    Span, Code, Input, Script, Style, \
    Request, Response, FileResponse, JSONResponse

from .. import APP_STATIC_DIR
from ..utils.executions import get_users, \
    get_pipeline_names, get_executions_before
from .page_template import page_layout


def AutoCompleteSelect(
    options_url: str,
    id: str,
    placeholder: Optional[str] = "",
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
                        window["_options_{id}"] = list;
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


    @rt(f"{prefix}/load_executions", methods=["POST"])
    async def get_execution_entries(
        request: Request
    ):
        # Retrieves params from form data (POST)
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

