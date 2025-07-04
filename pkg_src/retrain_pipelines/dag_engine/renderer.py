import logging

import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from .runtime import find_root_tasks


# ---- SVG Rendering ----


def render_svg(task, filename="dag.svg"):
    """Renders the DAG as an SVG file for visualization.

    Parallel tasks are colored differently.
    """
    import xml.etree.ElementTree as ET

    def create_svg_element(width, height, zoom_factor=2.0):
        svg = ET.Element(
            "svg", width=str(width * zoom_factor), height=str(height * zoom_factor), viewBox=f"0 0 {width} {height}"
        )
        style = ET.SubElement(svg, "style")
        style.text = ".task { font: bold 14px sans-serif; }"
        return svg

    def add_task_node(svg, task, x, y):
        color = "#b3e6ff" if not task.is_parallel else "#ffe6b3"  # Blue for normal, orange for parallel
        rect = ET.SubElement(svg, "rect", x=str(x), y=str(y), width="160", height="40", fill=color, stroke="#333")
        text = ET.SubElement(svg, "text", x=str(x + 10), y=str(y + 25), class_="task")
        text.text = task.name
        return rect

    def add_edge(svg, x1, y1, x2, y2):
        line = ET.SubElement(
            svg,
            "line",
            x1=str(x1),
            y1=str(y1),
            x2=str(x2),
            y2=str(y2),
            stroke="#333",
            attrib={"marker-end": "url(#arrow)"},
        )
        return line

    def add_arrow_def(svg):
        defs = ET.SubElement(svg, "defs")
        marker = ET.SubElement(
            defs, "marker", id="arrow", markerWidth="10", markerHeight="10", refX="10", refY="5", orient="auto"
        )
        polygon = ET.SubElement(marker, "polygon", points="0,0 10,5 0,10", fill="#333")  # noqa: F841

    def layout_tasks(task, x, y, level, positions):
        if task in positions:
            return positions[task]
        positions[task] = (x, y)
        max_x = x
        max_y = y
        for i, child in enumerate(task.children):
            child_x = x + 200
            child_y = y + 80 * i
            cx, cy = layout_tasks(child, child_x, child_y, level + 1, positions)
            max_x = max(max_x, cx)
            max_y = max(max_y, cy)
        return (max_x, max_y)

    roots = find_root_tasks(task)
    positions = {}
    max_x, max_y = 0, 0
    for i, root in enumerate(roots):
        mx, my = layout_tasks(root, 50, 50 + i * 100, 0, positions)
        max_x = max(max_x, mx)
        max_y = max(max_y, my)

    svg_width = max_x + 200  # Add some padding
    svg_height = max_y + 100  # Add some padding
    svg = create_svg_element(svg_width, svg_height, zoom_factor=2.0)
    add_arrow_def(svg)

    for t, (x, y) in positions.items():
        add_task_node(svg, t, x, y)
        for child in t.children:
            cx, cy = positions[child]
            add_edge(svg, x + 160, y + 20, cx, cy + 20)

    tree = ET.ElementTree(svg)
    tree.write(filename, encoding="utf-8", xml_declaration=True)
    print(f"DAG rendered to {filename}")


# ---- networkx Rendering ----


def render_networkx(task, filename="dag.png"):
    """Renders the DAG using NetworkX."""

    G = nx.DiGraph()

    def add_nodes_edges(task):
        G.add_node(task.id, label=task.name)
        for child in task.children:
            add_nodes_edges(child)
            G.add_edge(task.id, child.id)

    roots = find_root_tasks(task)
    for root in roots:
        add_nodes_edges(root)

    pos = nx.spring_layout(G)
    labels = nx.get_node_attributes(G, "label")
    nx.draw(
        G,
        pos,
        labels=labels,
        with_labels=True,
        node_size=3000,
        node_color="skyblue",
        font_size=10,
        font_color="black",
        font_weight="bold",
        arrowsize=20,
    )
    dag_json = nx.node_link_data(G)
    logging.getLogger(__name__).info(dag_json)
    plt.savefig(filename)
    plt.close()
    print(f"DAG rendered to {filename}")


# ---- plotly Rendering ----



def render_plotly(task, filename="dag.html"):
    """Renders the DAG using Plotly with arrowheads and nodes on top."""
    fig = go.Figure()
    added_nodes = set()
    edges = []

    def add_nodes(task):
        if task.id not in added_nodes:
            added_nodes.add(task.id)
            fig.add_trace(
                go.Scatter(
                    x=[0],  # Placeholder x value, will be updated later
                    y=[0],  # Placeholder y value, will be updated later
                    mode="markers+text",
                    text=task.name,
                    textposition="bottom center",
                    marker=dict(size=20, color="skyblue"),
                    name=task.id,
                    zorder=2,  # Higher zorder for nodes
                )
            )
        for child in task.children:
            edges.append((task.id, child.id))
            add_nodes(child)

    def layout_nodes():
        pos = nx.spring_layout(nx.DiGraph(edges))
        for trace in fig.data:
            node_id = trace.name
            if node_id in pos:
                trace.x = [pos[node_id][0]]
                trace.y = [pos[node_id][1]]
        return pos

    roots = find_root_tasks(task)
    for root in roots:
        add_nodes(root)

    pos = layout_nodes()

    # Add edges as lines with lower zorder
    for parent_id, child_id in edges:
        parent_x, parent_y = pos[parent_id]
        child_x, child_y = pos[child_id]
        fig.add_trace(
            go.Scatter(
                x=[parent_x, child_x],
                y=[parent_y, child_y],
                mode="lines",
                line=dict(color="gray", width=2),
                showlegend=False,
                zorder=1,  # Lower zorder for edges
            )
        )

        # Add arrow annotation (always on top of traces)
        fig.add_annotation(
            x=child_x,
            y=child_y,
            ax=parent_x,
            ay=parent_y,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="gray",
            standoff=5,
        )

    fig.update_layout(
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=20, r=20, t=20, b=20),
        hovermode="closest",
    )
    fig.write_html(filename)
    print(f"DAG rendered to {filename}")

