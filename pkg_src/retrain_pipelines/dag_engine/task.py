import os
import concurrent.futures
import functools
import logging
import uuid
from collections import defaultdict, deque

from .db.dao import DAO


# ---- Core Task and Execution Infrastructure ----


class Task:
    """Represents a node in the DAG."""

    def __init__(self, func, is_parallel=False, merge_func=None):
        self.log = logging.getLogger()
        self.log.info(f"Hello, [red on white]{func.__name__}[/red on white]")

        self.func = self._wrap_func(func)  # Wrap the function with logging
        self.is_parallel = is_parallel  # Should this task run in parallel for a list of inputs?
        self.merge_func = merge_func  # Optional function to merge parallel results
        self.id = str(uuid.uuid4())  # Unique ID for graph rendering
        self.parents = []  # List of parent Task objects
        self.children = []  # List of child Task objects
        self.exec_id = None  # Execution ID for the task
        self.task_id = None

    def _wrap_func(self, func):
        """Wrap the function with logging."""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if (docstring:=func.__doc__):
                self.log.info(f"[bold green]{docstring}[/bold green]")
            print(f"args: {args}")
            print(f"kwargs: {kwargs}")
            index = kwargs.pop("index", None)
            dao = DAO(os.environ["RP_METADATASTORE_URL"])
            self.task_id = dao.add_task(self.exec_id)

            if index is not None:
                self.log.info(
                    f"Executing task: {func.__name__}, ID: {self.id}, Index: {index}, Inputs: {args}, {kwargs}, TaskId: {self.task_id}"
                )
            else:
                self.log.info(
                    f"Executing task: {func.__name__}, ID: {self.id}, Inputs: {args}, {kwargs}, TaskId: {self.task_id}"
                )

            result = func(*args, **kwargs)

            self.log.info(f"Completed task: {func.__name__}, ID: {self.id}, Index: {index}, Result: {result}")

            return result

        return wrapper

    def __rshift__(self, other):
        """Operator overloading for '>>' to connect tasks in the DAG.

        Allows chaining: a >> b >> c or a >> TaskGroup(b, c).
        """
        # execution_id cascading
        dao = DAO(os.environ["RP_METADATASTORE_URL"])

        if self.exec_id is None:
            self.exec_id = dao.add_execution()
            self.log.error(self.func.__name__ + " has no exec_id")
            for child in self.children:
                self._cascade_exec_id(self.exec_id, child)
        if other.exec_id is None:
            other.exec_id = self.exec_id
            if isinstance(other, Task):
                for child in other.children:
                    self._cascade_exec_id(self.exec_id, child)
            if isinstance(other, TaskGroup):
                for element in other.elements:
                    self._cascade_exec_id(self.exec_id, element)

        # actual chaining
        if isinstance(other, Task):
            self.children.append(other)
            other.parents.append(self)
            return other
        elif isinstance(other, TaskGroup):
            for element in other.elements:
                self._add_child(element)
            return other
        else:
            raise TypeError("The right-hand side of '>>' must be a Task object, or a TaskGroup object.")

    def _cascade_exec_id(self, exec_id, element):
        if element.exec_id is None:
            element.exec_id = exec_id
            if isinstance(element, TaskGroup):
                for sub_element in element.elements:
                    self._cascade_exec_id(exec_id, sub_element)

    def _add_child(self, child):
        """Recursively add children to the task."""
        if isinstance(child, Task):
            self.children.append(child)
            child.parents.append(self)
            if child.exec_id is None:
                child.exec_id = self.exec_id
        elif isinstance(child, TaskGroup):
            for child_element in child.elements:
                self._add_child(child_element)

    # def __call__(self, *args, **kwargs):
        # return self.func(*args, **kwargs)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, Task):
            return self.id == other.id
        return False


class MergeNotSupportedError(Exception):
    """Raised when attempting to chain a taskgroup
    to a parallel-merging task,
    which is not supported (yet?)."""
    pass


class TaskGroup:
    """Represents an ordered group of tasks that
    can be treated as a single entity in the DAG.

    Note: the downward task in the task receives inputs
    from the tasks in the group in the order of appearance in that group.
    i.e. Taskgroup(task1, task2, task3) >> task4
    will call task4.func(result_1, result_2, result_3)
    where result_1, result_2, result_3 are task1's result
    and task2's result and task3's result respectively and in that order.
    """

    def __init__(self, *elements):
        self.log = logging.getLogger()
        self.elements = elements
        self.id = str(uuid.uuid4())  # Unique ID for graph rendering
        self.parents = []  # List of level-1 parent Task objects
        self.children = []  # List of level-1 child Task objects
        self.exec_id = None  # Execution ID for the task group

    def __rshift__(self, other):
        """Operator overloading for '>>' to connect tasks in the DAG.

        Allows chaining: a >> b >> c or a >> TaskGroup(b, c).
        """
        # execution_id cascading
        dao = DAO(os.environ["RP_METADATASTORE_URL"])

        if self.exec_id is None:
            self.exec_id = dao.add_execution()
            self.log.error(self.func.__name__ + " has no exec_id")
            for child in self.children:
                self._cascade_exec_id(self.exec_id, child)
        if other.exec_id is None:
            other.exec_id = self.exec_id
            if isinstance(other, Task):
                for child in other.children:
                    self._cascade_exec_id(self.exec_id, child)
            if isinstance(other, TaskGroup):
                for element in other.elements:
                    self._cascade_exec_id(self.exec_id, element)

        # actual chaining
        if isinstance(other, Task):
            if other.merge_func:
                # Note: TODO, implement support for that someday (or not)
                raise MergeNotSupportedError(
                    "merging tasks can only have 1 parent")
            self._add_child(other)
            return other
        elif isinstance(other, TaskGroup):
            for element in other.elements:
                self._add_child(element)
            return other
        else:
            raise TypeError(
                "The right-hand side of '>>' must be a Task object, or a TaskGroup object.")

    def _cascade_exec_id(self, exec_id, element):
        if element.exec_id is None:
            element.exec_id = exec_id
            if isinstance(element, TaskGroup):
                for sub_element in element.elements:
                    self._cascade_exec_id(exec_id, sub_element)

    def _add_child(self, child):
        """Recursively add children to the task or task group."""
        if isinstance(child, Task):
            for element in self.elements:
                if isinstance(element, Task):
                    element.children.append(child)
                    child.parents.append(element)
                elif isinstance(element, TaskGroup):
                    element._add_child(child)
        elif isinstance(child, TaskGroup):
            for element in self.elements:
                if isinstance(element, Task):
                    for sub_child in child.elements:
                        element.children.append(sub_child)
                        sub_child.parents.append(element)
                elif isinstance(element, TaskGroup):
                    element._add_child(child)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, TaskGroup):
            return self.id == other.id
        return False


# ---- Decorators for Task Declaration ----


def task(func=None, *, merge_func=None):
    """Decorator for regular (non-parallel) tasks.

    Optionally takes a merge_func for merging results from parallel tasks.
    """

    def decorator(f):
        t = Task(f, is_parallel=False, merge_func=merge_func)

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)

        wrapper._task = t
        return t  # Return the Task object instead of the wrapper

    return decorator(func) if func else decorator


def parallel_task(func=None, *, merge_func=None):
    """Decorator for parallel tasks.

    Optionally takes a merge_func for merging results after parallel execution.
    """

    def decorator(f):
        t = Task(f, is_parallel=True, merge_func=merge_func)

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)

        wrapper._task = t
        return t  # Return the Task object instead of the wrapper

    return decorator(func) if func else decorator


# ---- DAG Traversal and Execution Utilities ----


def get_all_tasks(task, seen=None) -> list[Task]:
    """Recursively collect all tasks reachable from the given task.

    Used for graph traversal and rendering.
    """
    if seen is None:
        seen = set()
    if task in seen:
        return []
    seen.add(task)
    tasks = [task] if isinstance(task, Task) else []
    for child in task.children:
        if isinstance(child, TaskGroup):
            for t in child.tasks:
                tasks += get_all_tasks(t, seen)
        else:
            tasks += get_all_tasks(child, seen)
    return tasks


def find_root_tasks(task) -> list[Task]:
    """Find all root tasks in the DAG starting from the given task."""
    all_tasks = set()
    stack = [task]
    while stack:
        current = stack.pop()
        if current not in all_tasks:
            all_tasks.add(current)
            stack.extend(current.parents)
    return [t for t in all_tasks if not t.parents]


def topological_sort(tasks) -> list[Task]:
    """Standard Kahn's algorithm for topological sorting of the DAG.

    Ensures tasks are executed in dependency order.
    """
    all_tasks = set()
    for t in tasks:
        all_tasks.update(get_all_tasks(t))
    print(f"all_tasks {[t.func.__name__ for t in all_tasks]}")
    in_degree = defaultdict(int)
    for t in all_tasks:
        for child in t.children:
            if isinstance(child, TaskGroup):
                for task in child.tasks:
                    in_degree[task] += 1
            else:
                in_degree[child] += 1

    queue = deque([t for t in all_tasks if in_degree[t] == 0])
    order = []
    while queue:
        t = queue.popleft()
        order.append(t)
        for child in t.children:
            if isinstance(child, TaskGroup):
                for task in child.tasks:
                    in_degree[task] -= 1
                    if in_degree[task] == 0:
                        queue.append(task)
            else:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

    return order


def _rich_log_execution_id_with_timestamp(execution_id: int):
    """Force the timestamp to show on execution start log with rich logger."""
    log = logging.getLogger()
    for handler in log.handlers:
        # Check if handler is RichHandler (or subclass)
        if hasattr(handler, "_log_render"):
            handler._log_render.omit_repeated_times = False
            handler._log_render.last_log_time = None  # Force timestamp on next log
            log.info(f"Execution ID: {execution_id}")
            handler._log_render.omit_repeated_times = True


def execute(task: Task, input_data=None):
    """From the start, executes the DAG that contains the given task.

    Handles parallel and nested parallel tasks, and merges results as needed.
    """
    _rich_log_execution_id_with_timestamp(task.exec_id)

    roots = find_root_tasks(task)
    print(f"Root tasks: {[t.func.__name__ for t in roots]}")

    order = topological_sort(roots)
    print(f"Topological order: {[t.func.__name__ for t in order]}")

    results = {}
    for t in order:

        t.log.info(f"Executing task: [rgb(0,255,255) on #af00ff]{t.func.__name__}[/rgb(0,255,255) on #af00ff]")
        # print(f"Children: {[p.func.__name__ for p in t.children]}")
        # print(f"Parents: {[p.func.__name__ for p in t.parents]}")
        parent_results = {}
        for p in t.parents:
            # print(f" parent {p.func.__name__}, results : {results[p.func.__name__]}")
            parent_results.update({p.func.__name__: results[p.func.__name__]})
        print(f"parent_results: {parent_results}")
        if t.is_parallel:
            # For parallel tasks, run func for each item in the "parent_results"
            # (all values shall be iterables of the same length)
            print(f"parent_results : {parent_results}")
            print(f"parent_results.values() : {parent_results.values()}")
            with concurrent.futures.ThreadPoolExecutor() as executor:
                keys = list(parent_results.keys())
                values = zip(*parent_results.values())
                futures = [
                    executor.submit(t.func, dict(zip(keys, row)), index=i)
                    for i, row in enumerate(values)
                ]
                sub_results = [f.result() for f in futures]
            # Merge results if a merge_func is provided
            t.log.info(f"3. {t.func.__name__}, [rgb(0,255,255) on #af00ff]{t.id}[/rgb(0,255,255) on #af00ff]")
            result = t.merge_func(sub_results) if t.merge_func else sub_results
        else:
            # For regular tasks, pass parent results as arguments
            t.log.info(f"4. {t.func.__name__}, [rgb(0,255,255) on #af00ff]{t.id}[/rgb(0,255,255) on #af00ff]")
            # merging tasks can only have 1 parent
            parent_results = \
                {t.parents[0].func.__name__: t.merge_func(list(parent_results.values())[0])} if t.merge_func \
                else parent_results
            if t.merge_func:
                print(
                    f"`{t.func.__name__}` merged {t.merge_func.__name__}(parent_results) : " +
                    str(parent_results)
                )
            result = t.func(parent_results) if parent_results else t.func()

        results.update({t.func.__name__: result})

    # Return the result of the last task in topological order (the sink)
    return results #[order[-1]]


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
        text.text = task.func.__name__
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

import networkx as nx
import matplotlib.pyplot as plt


def render_networkx(task, filename="dag.png"):
    """Renders the DAG using NetworkX."""
    logger = logging.getLogger("matplotlib")
    logger.setLevel(logging.INFO)

    logger = logging.getLogger("PIL.PngImagePlugin")
    logger.setLevel(logging.INFO)

    G = nx.DiGraph()

    def add_nodes_edges(task):
        G.add_node(task.id, label=task.func.__name__)
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

import plotly.graph_objects as go


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
                    text=task.func.__name__,
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

