
import os
import string
import random
import asyncio

from typing import Optional
from jinja2 import Environment, FileSystemLoader

from concurrent.futures import ThreadPoolExecutor

from .core import DAG
from .db.dao import AsyncDAO
from ..utils import get_text_pixel_width, in_notebook


def render_svg(dag: DAG, filename="dag.html"):
    """Renders the DAG for visualization

    at SVG format as a portable html file.
    """
    static_dir = os.path.join(
        os.path.dirname(__file__), "web_console", "static")
    with open(os.path.join(static_dir, "html_body.css"),
              'r', encoding='utf-8') as f:
        html_body_css = f.read()

    with open(filename, 'w', encoding='utf-8') as file:
        file.write(
            "<html>" +
            "<head>" +
            "<style>\n" +
            html_body_css +
            "\n" +
            "</style>\n" +
            "</head>" +
            "<body>\n" +
            dag_svg(dag) +
            "</body></html>"
        )

def dag_svg(
    dag: Optional[DAG] = None,
    execution_id: Optional[int] = None
) -> str:
    """Renders the DAG for visualization

    at SVG format str prepended with css DOM tag.

    Params:
        - dag (DAG):
            the dag to generate
            an svg rendering for
        - execution_id (int):
            the id of an executuion,
            the dag of which to generate
            an svg rendering for
            (ignored if dag is provided)
    """
    template_dir = os.path.join(
        os.path.dirname(__file__), 
        "web_console", "utils", "execution")
    env = Environment(loader=FileSystemLoader(template_dir))
    env.globals['get_text_pixel_width'] = get_text_pixel_width
    template = env.get_template("svg_template.html")

    if dag:
        tasktypes_list, taskgroups_list = \
            dag.to_elements_lists(serializable=True)
    else: # if execution_id

        async def _fetch_execution_elements(exec_id: int):
            dao = AsyncDAO(
                db_url=os.environ["RP_METADATASTORE_ASYNC_URL"]
            )
            return await asyncio.gather(
                dao.get_execution_tasktypes_list(exec_id),
                dao.get_execution_taskgroups_list(exec_id),
            )

        if in_notebook():
            # Jupyter (IPython kernel) already
            # starts and runs an event loop internally.
            # Run the coroutine in a dedicated worker thread
            # with its own fresh event loop
            # so that the kernel's running loop is left untouched.
            with ThreadPoolExecutor(
                max_workers=1
            ) as pool:
                future = pool.submit(
                    asyncio.run,
                    _fetch_execution_elements(execution_id)
                )
                tasktypes_list, taskgroups_list = future.result()
        else:
            # create or get loop in this thread
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            tasktypes_list, taskgroups_list = \
                loop.run_until_complete(
                    _fetch_execution_elements(execution_id)
                )

        if tasktypes_list is None:
            return f"Invalid execution ID {execution_id}"

        # turn into serializables
        tasktypes_list = \
            [tasktypes.__dict__ for tasktypes in tasktypes_list]
        for tasktype_dict in tasktypes_list:
            tasktype_dict["uuid"] = str(tasktype_dict["uuid"])
            tasktype_dict["taskgroup_uuid"] = \
                str(tasktype_dict["taskgroup_uuid"]) \
                if tasktype_dict["taskgroup_uuid"] else ""

        taskgroups_list = \
            [taskgroup.__dict__ for taskgroup in taskgroups_list] \
            if taskgroups_list else []
        for taskgroup_dict in taskgroups_list:
            taskgroup_dict["uuid"] = str(taskgroup_dict["uuid"])

    # print(f"execution_tasktypes_list : {tasktypes_list}")
    # print(f"execution_taskgroups_list : {taskgroups_list}")

    # using unique DOM ids, so several instances
    # can coexist on a webpage (e.g. in notebooks)
    id_prefix = ''.join(random.choices(
                            string.ascii_letters + string.digits,
                            k=5)
    )
    # print(f"id_prefix : {id_prefix}")
    rendering_content = template.render(
        id_prefix=id_prefix,
        nodes=tasktypes_list,
        taskgroups=taskgroups_list or []
    )

    static_dir = os.path.join(
        os.path.dirname(__file__), "web_console", "static")
    with open(os.path.join(static_dir, "svg_dag.css"),
              'r', encoding='utf-8') as f:
        svg_dag_css = f.read()

    return (
            "<style>\n" +
            svg_dag_css +
            "\n" +
            "</style>\n" +
            rendering_content +
            "\n"
        )

