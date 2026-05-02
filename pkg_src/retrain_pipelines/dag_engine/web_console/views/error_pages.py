

from fasthtml.common import \
    Request, to_xml, Style, Div, Br, \
    HTMLResponse, HTTPException

from .page_template import page_layout


def error_page(
    status_code: int,
    req: Request | None = None,
    exc: Exception | None = None
):
    """Custom error page template for status_code.
    
    Params:
        - status_code (int):
            HTTP status code (400-599)
        - req:
            Starlette Request object (unused)
        - exc:
            Exception that triggered error (unused)
        
    Results:
        - FastHTML error page
    """
    if exc is None:
        detail = "An error occurred"
    elif isinstance(exc, HTTPException):
        detail = exc.detail
    elif hasattr(exc, 'detail'):
        detail = exc.detail
    else:
        detail = str(exc) if str(exc) else exc.__class__.__name__
    
    result = page_layout(
        current_page="/", 
        title="retrain-pipelines",
        content=Div(
            Div(
                str(status_code), Br(), str(detail),
                style="""
                    width: 100%;
                    font-size: clamp(1rem, 12vw, 20rem);
                    font-weight: 700;
                    text-align: center;
                    line-height: 1;

                    overflow-wrap: break-word;
                    word-wrap: break-word;
                    word-break: break-all;
                    white-space: normal;

                    background: linear-gradient(
                        145deg,
                        #ffffff 0%,
                        #f8e8ff 20%,
                        #d8b8f0 40%,
                        #c8a0e8 50%,
                        #d8b8f0 60%,
                        #f8e8ff 80%,
                        #ffffff 100%
                    );
                    -webkit-background-clip: text;
                    background-clip: text;
                    -webkit-text-fill-color: transparent;
                    filter: drop-shadow(0 1px 1px rgba(255,255,255,1))
                           drop-shadow(0 2px 4px rgba(210,180,240,0.7))
                           drop-shadow(0 0 12px rgba(200,160,240,0.5))
                           drop-shadow(0 4px 8px rgba(150,100,200,0.3))
                           drop-shadow(-0.05px -0.05px 0 #4d0066)
                           drop-shadow(0.05px -0.05px 0 #4d0066)
                           drop-shadow(-0.05px 0.05px 0 #4d0066)
                           drop-shadow(0.05px 0.05px 0 #4d0066)
                           drop-shadow(3px 3px 6px #FFEA66);

                    position: relative;
                """
            ),
            Style(""" /* body */
                .body-error-page { /* page content container */
                    padding-top: 7rem;
                    padding-bottom: 2.5rem;
                }
            """),
        ),
        body_cls=["body-error-page"]
    )

    return HTMLResponse(to_xml(result), status_code=status_code)

