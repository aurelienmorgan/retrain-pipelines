
from datetime import datetime

from fasthtml.common import *

from retrain_pipelines import __version__


def header(current_page="/"):
    nav_items = [
        ("Home", "/"),
        ("About", "/about"),
        ("Contact", "/contact")
    ]
    nav_links = []
    for title, path in nav_items:
        is_current = current_page == path
        link_style = (
            "color: white; margin-left: 16px; text-decoration: none;"
            + (" font-weight: bold; text-decoration: underline;" if is_current else "")
        )
        nav_links.append(
            A(title, href=path, style=link_style)
        )

    # Left: image + nav, in normal document flow (not fixed, not floating)
    left = Div(
        A(
            Img(src="logo.png", alt="Logo", style="height:40px; vertical-align: middle;"),
            href="/",
            style="display: inline-block; vertical-align: middle;"
        ),
        *nav_links,
        style="position: absolute; top: 12px; left: 16px; z-index: 200; display: flex; align-items: center;"
    )

    # Right: fixed green circle
    right = Div(
        Div(
            "",  # Just the circle
            style="width: 32px; height: 32px; background: #27c93f; border-radius: 50%;"
        ),
        style=(
            "position: fixed; top: 12px; right: 24px; z-index: 200; "
            "display: flex; align-items: center;"
        )
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
                    " \u00A0 - \u00A0 Open\u00A0Source\u00A0FTW\u00A0!",
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
                "display: flex; justify-content: space-between; align-items: center;"
            )
        ),
        cls="footer"
    )


footer_css = Style("""
html {
    scroll-behavior: smooth;
}

body {
    background-color: #4d0066;
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
            Meta(name="viewport", content="width=device-width, initial-scale=1.0"),
            Meta(name="description", content=f"{title} - MyWebsite built with FastHTML"),
            Script(src="https://cdn.tailwindcss.com"),
            footer_css
        ),
        Br(),
        Body(
            header(current_page),
            Main(
                Div(content, cls="container mx-auto px-4 py-8"),
                cls="min-h-screen"
            ),
            footer()
        )
    )

