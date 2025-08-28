
import os
import logging

from jinja2 import Environment, FileSystemLoader

from .core import DAG

from ..utils import get_text_pixel_width


def render_svg(dag: DAG, filename="dag.html"):
    """Renders the DAG for visualization

    at SVG format as a portable html file.
    """
    template_dir = os.path.join(
        os.path.dirname(__file__), 
        "web_console", "utils", "execution")
    env = Environment(loader=FileSystemLoader(template_dir))
    env.globals['get_text_pixel_width'] = get_text_pixel_width
    template = env.get_template("svg_template.html")

    tasktypes_list, taskgroups_list = \
        dag.to_elements_lists(serializable=True)
    # print(f"execution_tasktypes_list : {tasktypes_list}")
    # print(f"execution_taskgroups_list : {taskgroups_list}")

    rendering_content = template.render(
        nodes=tasktypes_list,
        taskgroups=taskgroups_list or []
    )

    static_dir = os.path.join(
        os.path.dirname(__file__), "web_console", "static")
    with open(os.path.join(static_dir, "html_body.css"),
              'r', encoding='utf-8') as f:
        html_body_css = f.read()
    with open(os.path.join(static_dir, "svg_dag.css"),
              'r', encoding='utf-8') as f:
        svg_dag_css = f.read()

    with open(filename, 'w', encoding='utf-8') as file:
        file.write(
            "<html>" +
            "<head>" +
            "<style>\n" +
            html_body_css +
            "\n" +
            svg_dag_css +
            "\n</style>" +
            "</head>" +
            "<body>\n" +
            rendering_content +
            "\n</body></html>"
        )

