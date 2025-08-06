
import os

from typing import Optional, List
from fasthtml.common import Div, P, Link, \
    Request
from jinja2 import Environment, FileSystemLoader

from .page_template import page_layout

from ...db.dao import AsyncDAO


async def get_execution_nodes_list(
    execution_id: int
) -> Optional[List[str]]:
    """Can be None, e.g. if no execution with that id exists."""
    dao = AsyncDAO(
        db_url=os.environ["RP_METADATASTORE_ASYNC_URL"]
    )
    return await dao.get_execution_nodes_list(execution_id)


def register(app, rt, prefix=""):
    @rt(f"{prefix}/dag_rendering", methods=["GET"])
    async def dag_rendering(request: Request):
        execution_id = request.query_params.get("id")
        try:
            execution_id = int(execution_id)
        except (TypeError, ValueError):
            return Div(P("Invalid execution ID"))

        nodes = await get_execution_nodes_list(execution_id)
        if nodes is None:
            return Div(P("Invalid execution ID"))

        template_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "utils", "execution")
        env = Environment(loader=FileSystemLoader(template_dir))
        template = env.get_template("svg_template.html")
        rendering_content = template.render(nodes=nodes)

        return rendering_content


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
                Div(# DAG renderer
                    P("Loading DAG..."),
                    hx_get=f"{prefix}/dag_rendering?id={execution_id}",
                    hx_trigger="load",
                    hx_swap="outerHTML"
                ),
                Link(
                    rel="stylesheet",
                    href=f"{prefix}/svg_dag.css"
                )
            )
        )

