
import os
import logging

from jinja2 import Environment, FileSystemLoader

from .core import DAG


def render_svg(dag: DAG, filename="dag.html"):
    """Renders the DAG for visualization

    at SVG format as a portable html file.
    """
    template_dir = os.path.join(
        os.path.dirname(__file__), 
        "web_console", "utils", "execution")
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template("svg_template.html")
    rendering_content = template.render(nodes=dag.to_nodes_list())

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

