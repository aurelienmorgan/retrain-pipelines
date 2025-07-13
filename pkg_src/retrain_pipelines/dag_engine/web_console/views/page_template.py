
from datetime import datetime

from fasthtml.common import *

from retrain_pipelines import __version__


def header(current_page="/"):
    nav_items = [
        ("Home", "/"),
        ("About", "/about"),
        ("Logs", "/web_server")
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
                style="height:40px; vertical-align: middle;"
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
            "",  # Just the status indicator
            id="status-circle",
            style=(
                "width: 32px; height: 32px; background: #27c93f; "
                "border-radius: 50%;"
            )
        ),
        Script("""
            document.addEventListener('htmx:sendError', function(event) {
                if (event.detail.elt.id === 'status-circle-container') {
                    document.getElementById('status-circle').style.background = '#ff4444';
                    const container = event.detail.elt;
                    container.setAttribute('hx-trigger', 'load, every 2s');
                    htmx.process(container);
                }
            });

            document.addEventListener('htmx:afterRequest', function(event) {
                if (event.detail.elt.id === 'status-circle-container' && event.detail.xhr.status === 200) {
                    document.getElementById('status-circle').style.background = '#27c93f';
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
                style="color: #6082B6; text-decoration: none;"
            ),
            style=(
                "margin-top: 8px; margin-bottom: 8px; "
                "display: flex; justify-content: space-between; "
                "align-items: center;"
            )
        ),
        cls="footer"
    )


page_template_css = Style("""
html {
    scroll-behavior: smooth;
}

body {
    background-color: #4d0066;
    font-family: 'Roboto', sans-serif;
    font-size: 0.95rem;
}

/* Webkit browsers (Chrome, Safari, Edge) */
::-webkit-scrollbar {
    height: 12px;
    width: 12px;
}
::-webkit-scrollbar-track {
    background: #f1f1f1;
    border: 2px solid #FFFFCC;
}
::-webkit-scrollbar-thumb {
    background-color: #4d0066;
    border: 2px solid #FFFFCC80;
}
/* Firefox */
html {
    scrollbar-width: thin;
    scrollbar-color: #4d0066 #FFFFCC80;
}

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
        )
    )

