
from datetime import datetime

from fasthtml.common import *

from retrain_pipelines import __version__


def header(current_page="/"):
    nav_items = [
        ("Home", "/"),
        ("Not-Found", "/not-exists"),
        ("Error", "/a_page_in_error"),
        ("Logs", "/web_server"),
        ("SSE API", "/api/docs")
    ]
    nav_links = []
    for title, path in nav_items:
        is_current = current_page == path
        link_style = (
            "color: white; margin-left: 16px; text-decoration: none;"
            + (" font-weight: bold; text-decoration: underline;"
               if is_current
               else "")
        )
        nav_links.append(
            A(title, href=path, style=link_style)
        )

    # Left: image + nav, in normal document flow (not fixed, not floating)
    left = Div(
        A(
            Img(
                src="pkg_logo_small.png", alt="retrain-pipelines",
                style=(
                    "height:40px; vertical-align: middle; "
                    "border: 2px solid black; "
                    "box-shadow: 0 0 0 1px #FFD700; "
                    "border-radius: 3px;"
                )
            ),
            href="/",
            style="display: inline-block; vertical-align: middle;"
        ),
        *nav_links,
        style=(
            "position: absolute; top: 12px; left: 16px; "
            "z-index: 200; display: flex; align-items: center;"
        )
    )

    # Right: fixed green circle
    right = Div(
        Div(
            "",  # Status indicator
            id="status-circle",
            cls="connected"
        ),
        Style("""
            @keyframes pulse {
                0% { transform: scale(1); opacity: 1; }
                50% { transform: scale(1.5); opacity: 0.6; }
                100% { transform: scale(1); opacity: 1; }
            }

            #status-circle {
                width: 32px;
                height: 32px;
                border-radius: 50%;
                transition: background 0.3s ease;
                box-shadow:
                    inset -2px -2px 5px rgba(0,0,0,0.2),
                    inset 2px 2px 5px rgba(255,255,255,0.1),
                    0 2px 6px rgba(0,0,0,0.2);
            }

            .connected {
                background: radial-gradient(circle at 30% 30%, #3cff76, #27c93f);
            }

            .disconnected {
                background: radial-gradient(circle at 30% 30%, #ff8888, #ff4444);
            }

            .pulsing {
                animation: pulse 1.2s infinite;
            }

            @keyframes swirl {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            #status-circle.swirling {
                animation: swirl 1s linear infinite;
            }

            @keyframes dashSpin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            #status-circle.spinning {
                border: 2px dashed #ffa500;
                animation: dashSpin 1s linear infinite;
            }

            @keyframes glowPulseOrange {
                0%, 100% { box-shadow: 0 0 8px #ffd700; }
                50% { box-shadow: 0 0 20px #ffcc00; }
            }
            #status-circle.orangePulsing {
                background: radial-gradient(circle at 30% 30%, #ffe066, #ffae00);
                animation: glowPulseOrange 1s ease-in-out infinite;
            }

            /* squishy ball */
            @keyframes jelly {
                  0%, 100% { transform: scale(1, 1); }
                  25% { transform: scale(1.2, 0.8); }
                  50% { transform: scale(0.8, 1.2); }
                  75% { transform: scale(1.1, 0.9); }
            }
            .squishing {
                transition: background 0.3s ease;
                box-shadow:
                  inset -2px -2px 5px rgba(0,0,0,0.2),
                  inset 2px 2px 5px rgba(255,255,255,0.1),
                  0 2px 6px rgba(0,0,0,0.2);

                animation: jelly 0.8s infinite;
            }

            /* orange orbiting ball */
            .orbited::before {
                content: '';
                position: absolute;
                width: 8px;
                height: 8px;
                background: #ffae00;
                border-radius: 50%;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%) rotate(0deg) translateX(20px);
                animation: orbit 1.2s linear infinite;
            }
            @keyframes orbit {
                0% { transform: translate(-50%, -50%) rotate(0deg) translateX(20px); }
                100% { transform: translate(-50%, -50%) rotate(360deg) translateX(20px); }
            }
            .orbited {
                position: relative;
            }

            /* pulsar bursting */
            .bursting::after {
                content: '';
                position: absolute;
                top: 50%;
                left: 50%;
                width: 100%;
                height: 100%;
                border: 2px solid rgba(255, 94, 0, 0.85);
                border-radius: 50%;
                transform: translate(-50%, -50%) scale(1);
                animation: violentPing 0.8s cubic-bezier(0.2, 0.6, 0.4, 1) infinite;
                box-shadow: 0 0 20px rgba(255, 94, 0, 0.6);
            }
            @keyframes violentPing {
                0% {
                    transform: translate(-50%, -50%) scale(0.6);
                    opacity: 1;
                }
                70% {
                    transform: translate(-50%, -50%) scale(2.2);
                    opacity: 0.4;
                }
                100% {
                    transform: translate(-50%, -50%) scale(2.5);
                    opacity: 0;
                }
            }
            .bursting {
                position: relative;
                box-shadow:
                    0 0 12px rgba(255, 102, 0, 0.6),
                    inset 0 0 6px rgba(255, 200, 0, 0.4);
            }
        """),
        Script("""
            document.addEventListener('htmx:sendError', function(event) {
                if (event.detail.elt.id === 'status-circle-container') {
                    const circle = document.getElementById('status-circle');
                    circle.classList.remove('connected');
                    circle.classList.add('disconnected', 'pulsing');

                    const container = event.detail.elt;
                    container.setAttribute('hx-trigger', 'load, every 2s');
                    htmx.process(container);
                }
            });

            document.addEventListener('htmx:afterRequest', function(event) {
                if (event.detail.elt.id === 'status-circle-container' && event.detail.xhr.status === 200) {
                    const circle = document.getElementById('status-circle');
                    circle.classList.remove('disconnected', 'pulsing');
                    circle.classList.add('connected');

                    const container = event.detail.elt;
                    container.setAttribute('hx-trigger', 'load, every 5s');
                    htmx.process(container);
                }
            });
        """),
        id="status-circle-container",
        style=(
            "position: fixed; top: 12px; right: 24px; z-index: 200; "
            "display: flex; align-items: center;"
        ),
        hx_get="/web_server/heartbeat",
        hx_trigger="load, every 5s",
        hx_swap="none"
    )

    return (left, right)


def footer():
    """
    Creates a consistent footer for all pages.
    Returns:
        A Footer component with copyright and links.
    """
    current_year = datetime.now().year
    return Footer(
        Hr(),
        P(
            Small(
                Span(
                    f"\u00A0 2023-{current_year} \u00A0 - \u00A0 ",
                    A(
                        Code(
                            Span(
                                f"retrain-pipelines {__version__}",
                                style=(
                                    "font-family: 'Lobster'; "
                                    "font-size: 14px; "
                                    "font-weight: normal; "
                                    "letter-spacing: 3px;"
                                ),
                                cls="shiny-silver-text"
                            ),
                        ),
                        href="https://github.com/aurelienmorgan/retrain-pipelines",
                        target="_blank",
                        style="text-decoration: none;"
                    ),
                    style="color: white;"
                )
            ),
            A(
                "page top\u00A0↑\u00A0\u00A0\u00A0",
                href="#pageTop",
                id="pageTopLink",
                style=(
                    "color: #6082B6; text-decoration: none; "
                    "display: none;"
                )
            ),
            Script("""// Show/Hide to-window-top hyperlink
                function togglePageTopLink() {
                    const link = document.getElementById("pageTopLink");
                    if (!link) return;
                    if (document.documentElement.scrollHeight > window.innerHeight) {
                        link.style.display = "inline";
                    } else {
                        link.style.display = "none";
                    }
                }

                // Initial checks for scrollability
                window.addEventListener('resize', togglePageTopLink);
                window.addEventListener('DOMContentLoaded', togglePageTopLink);
                window.addEventListener('load', togglePageTopLink);

                // Watch for DOM/content changes
                const linkToggleObserver = new MutationObserver(togglePageTopLink);
                linkToggleObserver.observe(document.body, {
                    childList: true,
                    subtree: true,
                    characterData: true
                });
            """),
            style=(
                "margin-top: 8px; margin-bottom: 8px; "
                "display: flex; justify-content: space-between; "
                "align-items: center;"
            )
        ),
        cls="footer"
    )


page_template_css = Style("""
/* checkboxes */
input.gcheckbox {
    appearance: none;
    -webkit-appearance: none;
    -moz-appearance: none;
    width: 18px;
    height: 18px;
    border-radius: 6px;
    background: linear-gradient(135deg, rgba(255,255,255,0.8), rgba(230,240,255,0.6));
    border: 1px solid rgba(180,200,230,0.5);
    box-shadow: 0 1px 3px rgba(0,0,0,0.1), inset 0 1px 0 rgba(255,255,255,0.7);
    backdrop-filter: blur(1.5px);
    cursor: pointer;
    display: inline-block;
    position: relative;
    margin: 0 2px 0 2px; align-self: center;
}
input.gcheckbox::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    text-align: center;
    font-size: 14px;
    font-weight: bold;
    color: #4d0066;
    pointer-events: none;
    line-height: 18px;
}
input.gcheckbox:checked::before {
  content: "✓";
}

/* table headers */
.glass-engraved {
    font-size: 1.1em;
    color: rgba(154,102,179,0.3);
    letter-spacing: 0.1em;
    text-shadow:
        0.05em 0.05em 0.1em rgba(0,0,0,0.6),
        -0.05em -0.05em 0.05em rgba(255,255,255,0.8),
        0 0.1em 0.3em rgba(0,0,0,0.3);
    opacity: 0.3;
}

/* comboboxes */
.combo-root {
    position: relative;
    display: flex;
    align-items: baseline;
    margin-right: 8px;
}
.combo-input {
    // flex: 1 1 60px;
    height: 18px;
    transition: width 0.4s ease, padding 0.4s ease,
       opacity 0.4s ease;
    transform-origin: right;
    box-sizing: border-box;
    margin-left: 5px; margin-right: 8px;
    padding: 0 6px; border: 1px solid rgba(180,200,230,0.5);
    border-radius: 6px;
    font-size: 13px; color: #4d0066;
    background: linear-gradient(135deg,
        rgba(230,240,255,0.7) 0%,
        rgba(200,220,255,0.6) 100%);
    box-shadow: 0 1px 3px rgba(0,0,0,0.06),
        inset 0 1px 0 rgba(255,255,255,0.7);
    backdrop-filter: blur(1.5px);
    outline: none;
}
.combo-input:focus {
    border: 1.5px solid #4d0066;
    box-shadow: 0 2.5px 10px rgba(80, 0, 140, 0.10);
}
.combo-input-unselected {
    font-style: italic; color: #222 !important;
}
.combo-input-selected-red {
    font-style: italic; color: red !important;
}
.combo-dropdown {
    position: absolute;
    left: 6px;
    right: 9px;
    top: 100%;
    z-index: 40;
    background: linear-gradient(135deg,
        rgba(230,240,255,0.7) 40%,
        rgba(200,220,255,0.6) 100%);
    border: 1px solid rgba(180,200,230,0.45);
    border-radius: 4px;
    box-shadow: 0 2px 14px rgba(60,30,102,0.12);
    padding: 2px 0;
    backdrop-filter: blur(2px);
    max-height: 180px;
    overflow-y: auto;
    display: none;
    font-family: 'Roboto', sans-serif;
    font-size: 13px;
    color: #4d0066;
}
.combo-dropdown.open {
    display: block;
}
.combo-option {
    font-size: 13px;
    color: #4d0066;
    padding: 4px 8px;
    cursor: pointer;
    background: transparent;
    border: none;
    transition: background 0.1s ease;
    line-height: 16px;
    display: block;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    direction: rtl;
    text-align: left;
    margin: 1px 0;
}
.combo-option * {
    direction: ltr;
}
.combo-option:nth-child(odd) {
    background: linear-gradient(135deg,
        rgba(180,140,220,0.4),
        rgba(200,160,230,0.3));
    border: 1px solid rgba(112, 51, 152, 0.2);
    box-shadow: 0 1px 2px rgba(0,0,0,0.05),
        inset 0 1px 0 rgba(255,255,255,0.7);
}
.combo-option:nth-child(even) {
    background: linear-gradient(135deg,
        rgba(160,100,200,0.5),
        rgba(190,140,220,0.4));
    border: 1px solid rgba(112, 51, 152, 0.3);
    box-shadow: 0 1px 2px rgba(0,0,0,0.05),
        inset 0 1px 0 rgba(255,255,255,0.6);
}
.combo-option.selected,
.combo-option.keyboard-active {
    background: linear-gradient(135deg,
        rgba(120,60,160,0.8),
        rgba(150,90,190,0.7)) !important;
    border: 1px solid rgba(77, 0, 102, 0.7);
    box-shadow: 0 3px 12px rgba(77,0,102,0.3),
        inset 0 1px 0 rgba(255,255,255,0.8);
    color: white;
}
/* A less-glassy alternative
.combo-option:nth-child(odd) {
    background: linear-gradient(135deg,
        rgba(255,255,255,0.4),
        rgba(245,240,255,0.3));
}
.combo-option:nth-child(even) {
    background: linear-gradient(135deg,
        rgba(220,200,255,0.5),
        rgba(200,180,240,0.4));
}
.combo-option.selected,
.combo-option.keyboard-active {
    background: linear-gradient(135deg,
        rgba(163,101,203,0.75),
        rgba(196,168,213,0.65)) !important;
    border: 1px solid rgba(112, 51, 152, 0.7);
    box-shadow: 0 2px 8px rgba(77,0,102,0.3),
        inset 0 1px 0 rgba(255,255,255,0.8);
}
*/

.shiny-silver-text {
    font-weight: bold;
    font-size: 14px;
    background: linear-gradient(
        90deg, #ffffff, #f0c6d0, 
        #c0c0c0, #f0c6d0, #ffffff
    );
    background-size: 200%;
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    text-shadow: 
        1px 1px 3px rgba(0, 0, 0, 0.4),
        0 0 8px rgba(255, 255, 255, 0.7);
    animation: shiny 2s linear infinite;
    padding: 0 4px;
    white-space: nowrap;
}

@keyframes shiny {
    0% {
        background-position: 200%;
    }
    100% {
        background-position: -200%;
    }
}

.footer {
    position: fixed;
    left: 4px;
    bottom: 0px;
    right: 4px;
    width: 100%;
    background: linear-gradient(to top, 
        rgba(77, 0, 102, 1) 0%, 
        rgba(77, 0, 102, 0.6) 50%, 
        rgba(77, 0, 102, 0.6) 50%, 
        rgba(77, 0, 102, 0) 100%);
    z-index: 100;
}
""")


def page_layout(title, content, current_page="/"):
    return Html(
        Head(
            Title(title or "retrain-pipelines"),
            Meta(
                name="viewport",
                content="width=device-width, initial-scale=1.0"
            ),
            Meta(
                name="description",
                content=f"{title} - WebConsole built with FastHTML"
            ),
            Link(
                rel="stylesheet",
                href="https://fonts.googleapis.com/css2?family=Lobster&display=swap"
            ),
            Link(
                rel="stylesheet",
                href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap"
            ),
            Link(
                rel="stylesheet",
                href="/html_body.css"
            ),
            Script(src="https://cdn.tailwindcss.com"),
            Script(src="https://unpkg.com/htmx.org@1.9.2"),
            page_template_css
        ),
        Body(
            header(current_page),
            Main(
                Br(),
                # Script("""//htmlx debugging
                    # htmx.logger = function (elt, event, data) {
                    # console.log("HTMX event:", event, data);
                    # }
                # """),
                Div(content, cls="container mx-auto px-4 py-8"),
                cls="min-h-screen"
            ),
            footer()
        ),
        id="pageTop"
    )

