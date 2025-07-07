
import os

from fasthtml.common import H1, P, FileResponse

from .. import APP_STATIC_DIR
from .page_template import page_layout

def register(app, rt, prefix=""):


    @rt("/favicon.ico")
    def favicon():
        return FileResponse(os.path.join(
            APP_STATIC_DIR, "retrain-pipelines.ico")
        )


    @rt("/{fname:path}.{ext:static}")
    async def get(fname:str, ext:str): 
        file_fullname = os.path.join(APP_STATIC_DIR, f"{fname}.{ext}")
        return FileResponse(file_fullname)


    @rt(f"{prefix}/")
    def hello():
        content = (
            H1("Hello"),
            P("Hello, World!")
        )

        return page_layout(
            title="Hello",
            content=content,
            current_page="/"  # Home link in the header
        )

