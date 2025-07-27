
import os
import json

from datetime import datetime

from fasthtml.common import Response, \
    Div, H3, Span, Select, Input, Option, \
    Script

from .page_template import page_layout
from ..utils.cookies import get_ui_state
from ..utils.server_logs import read_last_access_logs


def register(app, rt, prefix=""):
    @rt(f"{prefix}/web_server/heartbeat", methods=["GET"])
    async def heartbeat(request):
        return Response(status_code=200)


    @rt(f"{prefix}/web_server/load_logs", methods=["POST"])
    async def get_log_entries(request):
        # Retrieve "count" from form data (POST)
        form = await request.form()
        count = form.get("count", 50)
        try:
            count = int(count)
        except (ValueError, TypeError):
            count = 50  # if conversion fails
        regex_filter = form.get("regex_filter", None)

        log_entries = read_last_access_logs(
            os.environ["RP_WEB_SERVER_LOGS"],
            "access.log",
            n=count,
            regex_filter=regex_filter
        )
        return log_entries


    @rt(f"{prefix}/web_server", methods=["GET"])
    def server_dashboard(req):
        # Get stored UI state or defaults
        lines = get_ui_state(
            req, "server_dashboard", "lines", "100"
        )
        autoscroll = (
                get_ui_state(
                    req, "server_dashboard", "autoscroll", "true"
                )
            )== "true"
        try:
            logic_filter, regex_filter = json.loads(get_ui_state(
                        req, "server_dashboard", "filter_values", '["",""]'
            ))
        except Exception as e:
            print(e)
            logic_filter, regex_filter = "", ""

        return page_layout(title="WebServer Logs", content=\
            Div(# page content
                Div(# params panel
                    Div(
                        Div(
                            H3(
                                "\N{CLIPBOARD} Recent Logs",
                                style=(
                                    "color: white; margin: 0; white-space: nowrap; "
                                    "font-size: 16px; line-height: 1.5;"
                                )
                            ),
                            style="display: flex; align-items: baseline;"
                        ),
                        Div(
                            Span(
                                "show last",
                                style=(
                                    "margin-right: 5px; white-space: nowrap; "
                                    "font-size: 14px; align-self: baseline;"
                                )
                            ),
                            Select(
                                *[Option(str(v), selected=(str(v) == lines))
                                  for v in [50, 100, 1000]],
                                value="50",
                                id="lines",
                                style=(
                                    "margin-right: 5px; font-size: 13px; "
                                    # "line-height: 1.2; padding: 2px 6px 2px 4px; "
                                    "width: fit-content; white-space: nowrap; "
                                    "box-sizing: content-box; border-radius: 6px; "
                                    "background: linear-gradient(135deg, "
                                        "rgba(255,255,255,0.8) 0%, "
                                        "rgba(230,240,255,0.6) 100%); "
                                    "border: 1px solid rgba(180,200,230,0.5); "
                                    "box-shadow: 0 1px 3px rgba(0,0,0,0.06), "
                                        "inset 0 1px 0 rgba(255,255,255,0.7); "
                                    "backdrop-filter: blur(1.5px); "
                                    "transition: box-shadow 0.15s, border 0.15s; "
                                    "text-align: center; text-align-last: center; "
                                    "outline: none; color: #4d0066;"
                                )
                            ),
                            Input(
                                type="checkbox",
                                checked=autoscroll,
                                id="autoscroll",
                                cls="gcheckbox",
                            ),
                            Span(
                                "autoscroll",
                                style=(
                                    "white-space: nowrap; font-size: 14px; "
                                    "align-self: baseline;"
                                )
                            ),
                            Input(
                                id="expand-textfield",
                                type="text",
                                placeholder="boolean logic filter...",
                                value=logic_filter,
                                spellcheck="false",
                                style=(
                                    "height: 18px; min-width: 0; width: 0; "
                                    "transition: width 0.4s ease, padding 0.4s ease, "
                                    "   opacity 0.4s ease; "
                                    "transform-origin: right; box-sizing: border-box; "
                                    "margin-left: 5px; margin-right: 8px; "
                                    "padding: 0 6px; border: 1px solid rgba(180,200,230,0.5); "
                                    "border-radius: 6px; font-size: 13px; color: #4d0066; "
                                    "background: linear-gradient(135deg, "
                                        "rgba(230,240,255,0.7) 0%, "
                                        "rgba(200,220,255,0.6) 100%); "
                                    "box-shadow: 0 1px 3px rgba(0,0,0,0.06), "
                                        "inset 0 1px 0 rgba(255,255,255,0.7); "
                                    "backdrop-filter: blur(1.5px); outline: none; opacity: 0;"
                                ),
                                _oninput="""
                                    this.style.fontStyle = 'italic';
                                    this.style.color = 'black';
                                """,
                                _onkeydown=("""
                                    if (event.key === 'Enter') {
                                        try {
                                            if (!this.value) {
                                                document.getElementById('regex-filter').value = "";
                                                return;
                                            }
                                            const regex_pattern = parseToRegex(this.value);
                                            document.getElementById('regex-filter').value = regex_pattern;
                                            this.style.fontStyle = 'normal';
                                            this.style.color = '#4d0066';
                                            let errorDiv = document.getElementById('regex-error-tooltip');
                                            if (errorDiv) errorDiv.remove();
                                        } catch (error) {
                                            document.getElementById('regex-filter').value = "";
                                            this.style.fontStyle = 'italic';
                                            this.style.color = 'red';
                                            // Format error tooltip
                                            let errorDiv = document.getElementById('regex-error-tooltip');
                                            if (!errorDiv) {
                                                errorDiv = document.createElement('div');
                                                errorDiv.id = 'regex-error-tooltip';
                                                errorDiv.style.position = 'absolute';
                                                errorDiv.style.background = 'rgba(220, 50, 47, 0.75)';
                                                errorDiv.style.color = '#fff1f1';  // soft off-white text
                                                errorDiv.style.padding = '10px 18px';
                                                errorDiv.style.borderRadius = '10px';
                                                errorDiv.style.fontSize = '13.5px';
                                                errorDiv.style.zIndex = '1000';
                                                errorDiv.style.transition = 'opacity 1s ease';
                                                errorDiv.style.opacity = '1';
                                                errorDiv.style.boxShadow = '0 1px 6px rgba(90,24,44,0.3)';
                                                errorDiv.style.border = '1.5px solid rgba(220,50,47,0.85)';
                                                errorDiv.style.backdropFilter = 'blur(3px) saturate(1.4)';
                                                errorDiv.style.letterSpacing = '0.01em';
                                                errorDiv.style.fontWeight = '500';
                                                errorDiv.style.textShadow = '0 1px 2px rgba(90,24,44,0.3)';
                                                var rect = this.getBoundingClientRect();
                                                errorDiv.style.left = rect.left + 'px';
                                                errorDiv.style.top = (rect.bottom + window.scrollY) + 'px';
                                                document.body.appendChild(errorDiv);
                                            }
                                            errorDiv.textContent = error.message || error.toString();
                                            errorDiv.style.opacity = '1';
                                            setTimeout(() => {
                                              errorDiv.style.opacity = '0';
                                              setTimeout(() => {
                                                if (errorDiv.parentNode) {
                                                    errorDiv.parentNode.removeChild(errorDiv);
                                                }
                                              }, 1000);
                                            }, 6000);
                                        }
                                    }
                                """)
                            ),
                            Input(
                                id="regex-filter",
                                type="hidden",
                                value=regex_filter
                            ),
                            Script("""// Caret at end position at load time
                                setTimeout(function() {
                                    const input = document.getElementById('expand-textfield');
                                    if (input) {
                                        input.selectionStart = input.selectionEnd = input.value.length;
                                    }
                                }, 100);
                            """),
                            Script("""// textfield value as tooltip
                                // (for cases with overflowing-ly long values)
                                function updateTitle() {
                                    if (!expand_textfield_input.value ||
                                        expand_textfield_input.value.trim() === ""
                                    ) {
                                        expand_textfield_input.title = "chained combo of AND(), OR() and NOT() operators over double-quoted substrings";
                                    } else {
                                        expand_textfield_input.title = expand_textfield_input.value;
                                    }
                                }
                                const expand_textfield_input = document.getElementById("expand-textfield");
                                // Set title on page load
                                updateTitle();
                                // Update title on input
                                expand_textfield_input.addEventListener("input", updateTitle);
                            """),
                            Script("""// converts boolean-logic string into regex patter
                                // e.g. `OR(AND(NOT("cool"), NOT ("man")), AND("really", "not"))`
                                //      gives `^(?:(?!.*cool)(?!.*man)|(?=.*really)(?=.*not)).*$`
                                // e.g. `AND(NOT("hello Kitty"), "cat")`
                                //      gives `^(?!.*hello Kitty)(?=.*cat).*$`
                                // e.g. `OR("Mario", "Luigi")`
                                //      gives `^(?:(?=.*Mario)|(?=.*Luigi)).*$`
                                function parseToRegex(input) {
                                    function escapeRegex(str) {
                                        return str.replace(/[.*+?^${}()|[\]\\\]/g, '\\\$&');
                                    }

                                    function stripSpacesOutsideQuotes(str) {
                                        let out = '';
                                        let inQuotes = false;
                                        for (let i = 0; i < str.length; ++i) {
                                            if (str[i] === '"') inQuotes = !inQuotes;
                                            if (!inQuotes && /\s/.test(str[i])) continue;
                                            out += str[i];
                                        }
                                        return out;
                                    }

                                    function parse(expr) {
                                        expr = stripSpacesOutsideQuotes(expr);
                                        let i = 0, parenDepth = 0;
                                        function parseRec() {
                                            let tokens = [];
                                            let expectArg = false;
                                            while (i < expr.length) {
                                                if (expr[i] === '"') {
                                                    let j = i + 1;
                                                    while (j < expr.length && expr[j] !== '"') j++;
                                                    if (j >= expr.length) throw new Error(
                                                        'Unclosed quoted string at position ' + i);
                                                    tokens.push(expr.slice(i + 1, j));
                                                    i = j + 1;
                                                    expectArg = false;
                                                } else if (expr.startsWith('AND(', i)) {
                                                    parenDepth++;
                                                    i += 4;
                                                    let args = parseRec();
                                                    if (!args.length) throw new Error(
                                                        'Operator "AND" requires at least one argument');
                                                    tokens.push({ op: 'AND', args });
                                                    expectArg = false;
                                                } else if (expr.startsWith('OR(', i)) {
                                                    parenDepth++;
                                                    i += 3;
                                                    let args = parseRec();
                                                    if (!args.length) throw new Error(
                                                        'Operator "OR" requires at least one argument');
                                                    tokens.push({ op: 'OR', args });
                                                    expectArg = false;
                                                } else if (expr.startsWith('NOT(', i)) {
                                                    parenDepth++;
                                                    i += 4;
                                                    let args = parseRec();
                                                    if (args.length !== 1) throw new Error(
                                                        'Operator "NOT" requires exactly one argument');
                                                    tokens.push({ op: 'NOT', args });
                                                    expectArg = false;
                                                } else if (expr[i] === ',') {
                                                    if (expectArg) throw new Error(
                                                        'Missing argument before comma at position ' + i);
                                                    i++;
                                                    expectArg = true;
                                                } else if (expr[i] === ')') {
                                                    parenDepth--;
                                                    if (parenDepth < 0) throw new Error(
                                                        'Unmatched closing parenthesis at position ' + i);
                                                    i++;
                                                    return tokens;
                                                } else if (expr[i] === '(') {
                                                    throw new Error(
                                                        'Unexpected "(" at position ' + i);
                                                } else {
                                                    i++;
                                                }
                                            }
                                            if (parenDepth > 0) throw new Error(
                                                'Unmatched opening parenthesis');
                                            return tokens;
                                        }
                                        const result = parseRec();
                                        if (parenDepth > 0) throw new Error('Unmatched opening parenthesis');
                                        if (parenDepth < 0) throw new Error('Unmatched closing parenthesis');
                                        if (i !== expr.length) throw new Error('Unexpected characters after parsing at position ' + i);
                                        return result;
                                    }

                                    function toRegex(node) {
                                        if (typeof node === 'string') {
                                            if (!node.length) throw new Error(
                                                'Empty quoted string is not allowed');
                                            return `(?=.*${escapeRegex(node)})`;
                                        }
                                        if (!node.args || !Array.isArray(node.args)) {
                                            throw new Error('Malformed node');
                                        }
                                        if (node.op === 'AND') {
                                            if (node.args.length === 0) throw new Error(
                                                'Operator "AND" requires at least one argument');
                                            const notArgs = node.args.filter(a => a && a.op === 'NOT');
                                            const posArgs = node.args.filter(a => !a || a.op !== 'NOT');
                                            let parts = [];
                                            for (const na of notArgs) {
                                                let inner = na.args[0];
                                                if (typeof inner === 'string') {
                                                    parts.push(`(?!.*${escapeRegex(inner)})`);
                                                } else {
                                                    parts.push(`(?!.*${toRegex(inner)})`);
                                                }
                                            }
                                            for (const pa of posArgs) {
                                                parts.push(toRegex(pa));
                                            }
                                            return parts.join('');
                                        }
                                        if (node.op === 'OR') {
                                            if (node.args.length === 0) throw new Error(
                                                'Operator "OR" requires at least one argument');
                                            return `(?:${node.args.map(arg => toRegex(arg)).join('|')})`;
                                        }
                                        if (node.op === 'NOT') {
                                            if (node.args.length !== 1) throw new Error(
                                                'Operator "NOT" requires exactly one argument');
                                            let inner = node.args[0];
                                            if (typeof inner === 'string') {
                                                return `(?!.*${escapeRegex(inner)})`;
                                            } else {
                                                return `(?!.*${toRegex(inner)})`;
                                            }
                                        }
                                        throw new Error('Unknown operator: ' + node.op);
                                    }

                                    // --- Handle root as array or node ---
                                    const parsed = parse(input);
                                    let pattern;
                                    if (parsed.length === 1 && parsed[0] && parsed[0].op) {
                                        pattern = toRegex(parsed[0]);
                                    } else if (parsed.length > 1) {
                                        pattern = toRegex({ op: 'AND', args: parsed });
                                    } else if (parsed.length === 1 && typeof parsed[0] === 'string') {
                                        pattern = toRegex(parsed[0]);
                                    } else {
                                        throw new Error('Empty expression');
                                    }
                                    return `^${pattern}.*$`;
                                }
                            """),
                            Div(
                                "+",
                                id="expand-button",
                                style=(
                                    "position: absolute; "
                                    "bottom: 2px; "
                                    "right: 3px; "
                                    "width: 12px; "
                                    "height: 12px; "
                                    "line-height: 12px; "
                                    "text-align: center; "
                                    "font-size: 16px; "
                                    "font-weight: 800; "
                                    "color: rgba(77, 0, 102, 0.5); /* #4d0066 */ "
                                    "background: rgba(230, 240, 255, 0.6); "
                                    "border-radius: 50%; "
                                    "border: 1px solid rgba(255, 255, 255, 0.25); "
                                    "box-shadow: "
                                    "    0 1px 3px rgba(0, 0, 0, 0.1), "
                                    "    inset 0 1px 1px rgba(255, 255, 255, 0.4); "
                                    "backdrop-filter: blur(5px); "
                                    "-webkit-backdrop-filter: blur(5px); "
                                    "cursor: pointer; "
                                    "user-select: none; "
                                    "transition: background 0.3s ease, color 0.3s ease; "
                                    "z-index: 9999;"
                                ),
                                _onmouseover=(
                                    "this.style.background='rgba(255,255,255,0.95)'; "
                                    "this.style.color='rgba(77, 0, 102, 0.7)';"
                                ),
                                _onmouseout=(
                                    "this.style.background='rgba(230,240,255,0.6)'; "
                                    "this.style.color='rgba(77, 0, 102, 0.5)';"
                                ),
                                _onclick="expandTextfield();",
                                _onkeydown=(
                                    "if(event.key === ' ' || event.key === 'Spacebar') { "
                                        "event.preventDefault(); this.click(); }"
                                ),
                            ),
                            Script("""// expandTextfield
                                function expandTextfield(fromResize = false) {
                                    const input = document.getElementById('expand-textfield');
                                    const expandButton = document.getElementById('expand-button');

                                    if (fromResize) {
                                        // Temporarily disable transition for resize
                                        const originalTransition = input.style.transition;
                                        input.style.transition = 'none';

                                        if (input.style.width !== '0px' && input.style.width) {
                                            const paramsPanel = document.getElementById('params_panel');
                                            const parentOfParams = paramsPanel.parentElement;
                                            const availableWidth = Math.max(
                                                1,
                                                parentOfParams.offsetWidth - 380
                                            );
                                            input.style.width = availableWidth + 'px';
                                        }

                                        // Restore transition after a frame
                                        setTimeout(() => {
                                            input.style.transition = originalTransition;
                                        }, 0);
                                        return;
                                    }

                                    if (input.style.width === '0px' || !input.style.width) {
                                        const paramsPanel = document.getElementById('params_panel');
                                        const parentOfParams = paramsPanel.parentElement;
                                        const availableWidth = Math.max(
                                            1,
                                            parentOfParams.offsetWidth - 380
                                        );
                                        input.style.width = availableWidth + 'px';
                                        input.style.padding = '4px 8px';
                                        input.style.opacity = '1';
                                        input.focus();
                                        expandButton.textContent = '-';
                                        if (typeof loadLogs === "function" && input.value.trim() > "") {
                                            // if the function has already been page-loaded
                                            // (otherwise delegated to DOMContentLoaded further down)
                                            loadLogs();
                                        }
                                    } else {
                                        input.style.width = '0';
                                        input.style.padding = '0';
                                        input.style.opacity = '0';
                                        expandButton.textContent = '+';
                                        loadLogs();
                                    }
                                    updateExpandCookie();
                                }

                                window.addEventListener('resize', function() {
                                    expandTextfield(true);
                                });
                            """),
                            id="params_panel",
                            style=(
                                "position: relative;"
                                "display: flex; "
                                "align-items: baseline; "
                                "padding: 12px 16px; "
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
                        Script("""// Memorize the user-selected params as cookies
                            // Helper function to set a cookie
                            function setCookie(name, value, days = 365) {
                                // 50 years in the future
                                const expires = new Date(Date.now() + 50*365*24*60*60*1000).toUTCString();
                                    document.cookie = `${name}=${value}; expires=${expires}; path=/`;
                            }
                            const COOKIE_PREFIX = "server_dashboard:";

                            /* ******************************
                            * For the checkbox (autoscroll) *
                            ****************************** */
                            const autoScrollCheckbox = document.getElementById("autoscroll");
                            autoScrollCheckbox.addEventListener("change", e => {
                                setCookie(COOKIE_PREFIX + e.target.id, e.target.checked);

                                // Only get the log container and scroll if checked
                                if (e.target.checked) {
                                    const logContainer = document.getElementById("log-container");
                                    if (logContainer) {
                                        logContainer.scrollTop = logContainer.scrollHeight;
                                    }
                                }
                            });

                            /* *******************************
                            * For the selection-list (lines) *
                            ******************************* */
                            document.getElementById("lines").addEventListener("change", e => {
                                setCookie(COOKIE_PREFIX + e.target.id, e.target.value);
                            });

                            /* ****************************************
                            * For the hidden textfield (regex-filter) *
                            **************************************** */
                            // update 1 cookie for both textfields
                            const regexFilter = document.getElementById("regex-filter");
                            const observer = new MutationObserver(() => {
                                const regexValue = document.getElementById("regex-filter").value.trim();
                                var logicValue = "";
                                if (regexValue > "") {
                                    logicValue = document.getElementById("expand-textfield").value;
                                }
                                const tupleValue = JSON.stringify([logicValue, regexValue]);
                                setCookie(COOKIE_PREFIX + "filter_values", tupleValue);
                                loadLogs();
                            });
                            observer.observe(regexFilter, { attributes: true, attributeFilter: ['value'] });

                            /* *************************************
                            * For the textfield (expand-textfield) *
                            ________________________________________
                            |           expansion state            |
                            _____________________________________ */
                            // Function that will be called on expansion event
                            const textfield = document.getElementById("expand-textfield")
                            function updateExpandCookie() {{
                                const expandButton = document.getElementById('expand-button');
                                const isExpanded = expandButton.textContent === '-';
                                setCookie(COOKIE_PREFIX + textfield.id, isExpanded);
                            }}
                            // Set textfield component its width at load time
                            const cookies = document.cookie.split("; ");
                            for (const cookie of cookies) {{
                                const [key, val] = cookie.split("=");
                                if (key === COOKIE_PREFIX + "expand-textfield") {
                                    if (val === "true") {
                                        if (textfield.value.trim() > "") expandTextfield();
                                    }
                                }
                            }}
                        """),
                        style=(
                            "display: flex; align-items: baseline; margin-bottom: 8px;"
                        )
                    )
                ),
                Div(# List header
                    Span(
                        "method",
                        style=(
                            "padding-left: 20px; display: inline-block; "
                            "text-align: center; width: 110px; flex: 0 0 110px;"
                        )
                    ),
                    Span("|", style="flex: 0 0 auto;"),
                    Span(
                        "timestamp",
                        style=(
                            "display: inline-block; "
                            "text-align: center; width: 192px; flex: 0 0 192px;"
                        )
                    ),
                    Span("|", style="flex: 0 0 auto;"),
                    Span(
                        "client ip",
                        style=(
                            "display: inline-block; "
                            "text-align: center; width: 124px; flex: 0 0 124px;"
                        )
                    ),
                    Span("|", style="flex: 0 0 auto;"),
                    Span(
                        "url",
                        style=(
                            "display: inline-block; text-align: center; "
                            "flex: 1 1 0; min-width: 0;"
                        )
                    ),
                    cls="glass-engraved",
                    style=(
                        "display: flex; align-items: flex-start; "
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
                Div(# Actual list
                    id="log-container",
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
                Script("""// Capping the total items count
                    (function() {
                        // Helper to get the max lines from the selection list
                        function getMaxLines() {
                            var default_fallback = 50;
                            var linesSelect = document.getElementById('lines');
                            if (!linesSelect) return default_fallback;
                            return parseInt(linesSelect.value, 10) || default_fallback;
                        }

                        var container = document.getElementById('log-container');
                        if (!container) return;

                        // Function to trim excess log items
                        function trimLogItems() {
                            var maxLines = getMaxLines();
                            var items = container.querySelectorAll('.log-entry');
                            while (items.length > maxLines) {
                                container.removeChild(items[0]);
                                items = container.querySelectorAll('.log-entry');
                            }
                        }

                        // Observe for new children in the log-container
                        var observer = new MutationObserver(function(mutationsList) {
                            for (var mutation of mutationsList) {
                                if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
                                    trimLogItems();
                                }
                            }
                        });
                        observer.observe(container, { childList: true });
                    })();
                """),
                Script("""// Mouseover wavy effect on log-entries
                    (function() {
                        const clamp = (v, min, max) => Math.max(min, Math.min(max, v));

                        const resetStyles = (entries) => {
                            entries.forEach(e => {
                                // Reset only transform, margin, zIndex — keep padding & background & overlay untouched
                                e.style.transform = '';
                                e.style.margin = '';
                                e.style.zIndex = '1';

                                // Restore glass overlay opacity
                                const overlay = e.querySelector('#glass-overlay');
                                if (overlay) overlay.style.opacity = '1';

                                // Restore background color to normal state
                                e.style.background = e.style.getPropertyValue('--status-color-normal');
                            });
                        };

                        const applyWaveEffect = (entry, entries, i) => {
                            const clamp = (v, min, max) => Math.max(min, Math.min(max, v));
                            const rect = entry.getBoundingClientRect();

                            const scaleFocusedBase = 1.05;
                            const scaleNeighborMaxBase = 1.03;
                            const marginTopMax = 3;

                            // Get container width and clamp scales
                            // so scaled width does not exceed container width
                            const container = document.getElementById('log-container');
                            if (!container) return;

                            const containerWidth = container.clientWidth;
                            const entryWidth = entry.offsetWidth;

                            // Clamp scales to prevent overflow:
                            const scaleFocused = Math.min(scaleFocusedBase, containerWidth / entryWidth);
                            const scaleNeighborMax = Math.min(scaleNeighborMaxBase, containerWidth / entryWidth);

                            // Store original margin-bottom on all entries if not stored yet
                            entries.forEach(e => {
                                if (e._originalMarginBottom === undefined) {
                                    const style = window.getComputedStyle(e);
                                    e._originalMarginBottom = style.marginBottom;
                                }
                            });

                            // Focused entry styles
                            entry.style.transition = 'transform 0.2s ease, margin-top 0.2s ease, background 0.3s ease';
                            entry.style.transform = `scale(${scaleFocused})`;
                            entry.style.marginTop = '4px';
                            entry.style.marginBottom = entry._originalMarginBottom; // restore original margin-bottom
                            entry.style.zIndex = '10';

                            const overlay = entry.querySelector('#glass-overlay');
                            if (overlay) {
                                overlay.style.transition = 'opacity 0.3s ease';
                                overlay.style.opacity = '0';
                            }
                            entry.style.background = entry.style.getPropertyValue('--status-color-hover');

                            const mouseMoveHandler = (ev) => {
                                const y = ev.clientY;
                                const relativeY = (y - rect.top) / rect.height;

                                const before = entries[i - 1];
                                const after = entries[i + 1];

                                if (before && relativeY <= 0.66) {
                                    const intensity = 1 - clamp(relativeY / 0.66, 0, 1);
                                    before.style.transition = 'transform 0.2s ease, margin-top 0.2s ease, background 0.3s ease';

                                    // Clamp neighbor scale as well:
                                    const neighborScale = 1 + (scaleNeighborMax - 1) * intensity;
                                    before.style.transform = `scale(${neighborScale})`;
                                    before.style.marginTop = `${marginTopMax * intensity}px`;
                                    before.style.marginBottom = before._originalMarginBottom; // original margin-bottom restored
                                    before.style.zIndex = '5';

                                    const beforeOverlay = before.querySelector('#glass-overlay');
                                    if (beforeOverlay) {
                                        beforeOverlay.style.transition = 'opacity 0.3s ease';
                                        beforeOverlay.style.opacity = `${1 - intensity}`;
                                    }

                                    before.style.background = before.style.getPropertyValue('--status-color-hover');
                                } else if (before) {
                                    before.style.transform = '';
                                    before.style.marginTop = '';
                                    before.style.marginBottom = before._originalMarginBottom;
                                    before.style.zIndex = '1';

                                    const beforeOverlay = before.querySelector('#glass-overlay');
                                    if (beforeOverlay) beforeOverlay.style.opacity = '1';

                                    before.style.background = before.style.getPropertyValue('--status-color-normal');
                                }

                                if (after && relativeY >= 0.33) {
                                    const intensity = clamp((relativeY - 0.33) / 0.66, 0, 1);
                                    after.style.transition = 'transform 0.2s ease, margin-top 0.2s ease, background 0.3s ease';

                                    // Clamp neighbor scale as well:
                                    const neighborScale = 1 + (scaleNeighborMax - 1) * intensity;
                                    after.style.transform = `scale(${neighborScale})`;
                                    after.style.marginTop = `${marginTopMax * intensity}px`;
                                    after.style.marginBottom = after._originalMarginBottom; // original margin-bottom restored
                                    after.style.zIndex = '5';

                                    const afterOverlay = after.querySelector('#glass-overlay');
                                    if (afterOverlay) {
                                        afterOverlay.style.transition = 'opacity 0.3s ease';
                                        afterOverlay.style.opacity = `${1 - intensity}`;
                                    }

                                    after.style.background = after.style.getPropertyValue('--status-color-hover');
                                } else if (after) {
                                    after.style.transform = '';
                                    after.style.marginTop = '';
                                    after.style.marginBottom = after._originalMarginBottom;
                                    after.style.zIndex = '1';

                                    const afterOverlay = after.querySelector('#glass-overlay');
                                    if (afterOverlay) afterOverlay.style.opacity = '1';

                                    after.style.background = after.style.getPropertyValue('--status-color-normal');
                                }
                            };

                            const cleanup = () => {
                                entries.forEach(e => {
                                    e.style.transform = '';
                                    e.style.marginTop = '';
                                    e.style.marginBottom = e._originalMarginBottom;
                                    e.style.zIndex = '1';

                                    const overlay = e.querySelector('#glass-overlay');
                                    if (overlay) overlay.style.opacity = '1';

                                    e.style.background = e.style.getPropertyValue('--status-color-normal');
                                });
                                document.removeEventListener('mousemove', mouseMoveHandler);

                                if (overlay) overlay.style.opacity = '1';
                                entry.style.background = entry.style.getPropertyValue('--status-color-normal');
                            };

                            document.addEventListener('mousemove', mouseMoveHandler);
                            entry.addEventListener('mouseleave', cleanup, { once: true });
                        };

                        const enhanceEntry = (entry) => {
                            if (entry.dataset.enhanced === "1") return;
                            entry.dataset.enhanced = "1";

                            entry.addEventListener('mouseenter', () => {
                                const entries = Array.from(document.querySelectorAll('.log-entry'));
                                const i = entries.indexOf(entry);
                                if (i !== -1) {
                                    applyWaveEffect(entry, entries, i);
                                }
                            });
                        };

                        const observeContainer = () => {
                            const container = document.getElementById('log-container');
                            if (!container) {
                                requestAnimationFrame(observeContainer);
                                return;
                            }

                            container.querySelectorAll('.log-entry').forEach(enhanceEntry);

                            const observer = new MutationObserver(mutations => {
                                mutations.forEach(mutation => {
                                    mutation.addedNodes.forEach(node => {
                                        if (!(node instanceof HTMLElement)) return;

                                        if (node.classList.contains('log-entry')) {
                                            enhanceEntry(node);
                                        }

                                        node.querySelectorAll?.('.log-entry').forEach(enhanceEntry);
                                    });
                                });
                            });

                            observer.observe(container, { childList: true, subtree: true });
                        };

                        observeContainer();
                    })();
                """),
                Script("""// Append log item on WebSocket message receive
                    const logContainer = document.getElementById("log-container");
                    let ws;

                    async function connectWebSocket() {
                        let ws;

                        const ws_url = `ws://${location.host}/{prefix}web_server/stream_logs`;
                        // start and allow for restart on connection lost
                        while (true) {
                            try {
                                ws = new WebSocket(ws_url);
                                // Wait for the connection to open
                                await new Promise((resolve, reject) => {
                                    ws.onopen = () => {
                                        console.log("WebSocket connected.");
                                        loadLogs();
                                        resolve();
                                    };
                                    ws.onerror = (err) => {
                                        console.error("WebSocket error:", err);
                                        reject(err);
                                    };
                                });
                                // Connection established, exit loop
                                break;
                            } catch (e) {
                                // Connection failed, wait before retrying
                                await new Promise(res => setTimeout(res, 3000));
                            }
                        }

                        ws.onmessage = (event) => {
                            if (document.getElementById('expand-button').textContent === '-') {
                                // apply regex filter
                                const regex_filter_str = document.getElementById('regex-filter').value;
                                if (regex_filter_str > "") {
                                    // extract "raw log" string from received div element
                                    const parser = new DOMParser();
                                    const log_entry = parser.parseFromString(
                                            event.data,
                                            "text/html"
                                        ).querySelector('.log-entry');
                                    if (log_entry) {
                                        const raw_str = log_entry.getAttribute('raw-str');
                                        // check against regex-filter for elligibility
                                        const regex = new RegExp(regex_filter_str);
                                        if (!regex.test(raw_str)) return;
                                    }
                                }
                            }
                            logContainer.insertAdjacentHTML('beforeend', event.data);
                            var autoScrollCheckbox = document.getElementById('autoscroll');
                            if (logContainer && autoScrollCheckbox && autoScrollCheckbox.checked) {
                                logContainer.scrollTop = logContainer.scrollHeight;
                            }
                        };

                        ws.onclose = () => {
                            console.log("WebSocket disconnected.");
                            // attempt to reconnect
                            // from that very webpage if not itself closed
                            // (i.e. in case of a server restart)
                            connectWebSocket(ws_url);
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
                    //    if (document.hidden) {
                    //        console.log("WebSocket closing on hidden event.");
                    //        setTimeout(() => ws.close(), 1000);
                    //    } else {
                    //        if (ws.readyState === WebSocket.CLOSED) {
                    //            console.log("WebSocket connecting on unhidden event.");
                    //            connectWebSocket();
                    //        }
                    //    }
                    // });
                """.replace("{prefix}", prefix+"/" if prefix > "" else "")
                ),
                Script("""// Cold start of logs list at page load time
                    function loadLogs() {
                        const server_status_circle = document.getElementById('status-circle');
                        server_status_circle.classList.add('spinning');

                        // Get how many log-entries to return from selection list
                        const linesSelect = document.getElementById('lines');
                        const count = linesSelect ? parseInt(linesSelect.value, 10) : 0;

                        // get the regex for log-entry filtering
                        const expandButton = document.getElementById('expand-button');
                        const isExpanded = expandButton.textContent === '-';
                        var regexValue = "";
                        if (isExpanded) {
                            const regexFilterInput = document.getElementById('regex-filter');
                            regexValue = regexFilterInput ? regexFilterInput.value : "";
                        }

                        const logContainer = document.getElementById("log-container");
                        logContainer.innerHTML = '';

                        // form data for the html POST
                        const formData = new FormData();
                        formData.append('count', count);
                        if (regexValue > "") formData.append('regex_filter', regexValue);

                        fetch('/{prefix}web_server/load_logs', {
                            method: 'POST',
                            headers: { "HX-Request": "true" },
                            body: formData
                        })
                        .then(response => response.text())
                        .then(html => {
                            logContainer.insertAdjacentHTML('afterbegin', html);
                            var autoScrollCheckbox = document.getElementById('autoscroll');
                            if (logContainer && autoScrollCheckbox && autoScrollCheckbox.checked) {
                                logContainer.scrollTop = logContainer.scrollHeight;
                            }

                            server_status_circle.classList.remove('spinning');
                        });
                    }

                    // Note that 'loadLogs' is first triggered
                    // not straight at page load
                    // but once the log-streaming websocket get connected
                    // so the so loadLogs event is streamed

                    // Assign to selection list change event
                    document.addEventListener('DOMContentLoaded', function() {
                        const linesSelect = document.getElementById('lines');
                        if (linesSelect) {
                            linesSelect.addEventListener('change', function() {
                                // Clear log-container content
                                const logContainer = document.getElementById('log-container');
                                if (logContainer) {
                                    loadLogs();
                                }
                            });
                        }
                    });
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

