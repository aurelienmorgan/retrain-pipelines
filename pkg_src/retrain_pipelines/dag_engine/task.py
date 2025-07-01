import os
import uuid
import logging
import functools
import concurrent.futures

from collections import defaultdict, deque
from typing import Callable, List, Optional, Union, \
    Dict, Any
from pydantic import BaseModel, Field, PrivateAttr, \
    model_validator

from .db.dao import DAO
from ..utils.rich_logging import framed_rich_log_str


# ---- Core Task and Execution Infrastructure ----


class Task(BaseModel):
    """Represents a node in the DAG.

    Note: The class is instantiated at DAG declaration time.
          Attributes related to DAG execution
          (such as `exec_id` and `task_id` and `rank`)
          are populated at execution time.
    """
    func: Callable
    is_parallel: bool = False
    merge_func: Optional[Callable] = None
    id: str = Field(default_factory=lambda: str(uuid.uuid4()),
                    description="Unique ID for graph rendering")
    exec_id: Optional[int] = None
    task_id: Optional[int] = None

    # in case of a task part of a parallel lane (can be nested)
    rank: Optional[List[int]] = None

    # Private attributes (not validated or included in .dict())
    _log: logging.Logger = PrivateAttr(default_factory=logging.getLogger)
    _parents: List["Task"] = PrivateAttr(default_factory=list)
    _children: List["Task"] = PrivateAttr(default_factory=list)
    # in case of a task part of a taskgroup
    _task_group: Optional["TaskGroup"] = None


    def __init__(self, **data):
        super().__init__(**data)
        self._log.info(
            "[red on white]" +
            f"{self.func.__name__}" +
            f"{f' ([#6f00d6 on white]parallel[/])' if self.is_parallel else f' ([#2d97ba on white]merge[/])' if self.merge_func else ''}" +
            "[/]")
        self.func = self._wrap_func(self.func)


    @property
    def log(self) -> logging.Logger:
        """Publicly expose the logger."""
        return self._log


    @property
    def parents(self) -> List["Task"]:
        return self._parents


    @property
    def children(self) -> List["Task"]:
        return self._children


    @property
    def task_group(self) -> "TaskGroup":
        return self._task_group


    def _wrap_func(self, func):
        """Wrap the function with logging."""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            index = kwargs.pop("index", None)
            dao = DAO(os.environ["RP_METADATASTORE_URL"])
            self.task_id = dao.add_task(self.exec_id)

            self.log.info(
                framed_rich_log_str(
                    f"\N{wrench} Executing Task "
                    f"[#D2691E]`{func.__name__}[{self.task_id}]"
                    f"{f'[{index}]' if index is not None else ''}`[/]:\n"
                    f"Inputs :\n"
                    f"  \N{BULLET} Positional: {args}\n"
                    f"  \N{BULLET} Keyword   : {kwargs}" +
                    (f"\n[bold green]{docstring}[/]" if (docstring:=func.__doc__) else ""),
                    border_color="#FFFFE0"
                )
            )

            result = func(*args, **kwargs)

            self.log.info(
                f"\n\N{WHITE HEAVY CHECK MARK} Completed Task [#D2691E]`{func.__name__}[{self.task_id}]{f'[{index}]' if index is not None else ''}`[/]:\n"
                f"Results :\n" +
                f"  \N{BULLET} {result} {f'({type(result).__name__})' if result is not None else ''}\n"
            )

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
            self.log.info(f"[bold red]{self.func.__name__} has no exec_id[/]")
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

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, Task):
            return self.id == other.id
        return False

    def __str__(self):
        return (f"Task({self.func.__name__!r}, is_parallel={self.is_parallel}, "
                f"merge_func={{self.merge_func.__name__!r if self.merge_func else None}}, "
                f"id={self.id!r}, exec_id={self.exec_id!r}, task_id={self.task_id!r}, "
                f"rank={self.rank!r})")


    def __repr__(self):
        return f"{self.__str__()}"


class MergeNotSupportedError(Exception):
    """Raised when attempting to chain a taskgroup
    to a parallel-merging task,
    which is not supported (yet?)."""
    pass


class TaskGroup(BaseModel):
    """Represents an ordered group of tasks that
    can be treated as a single entity in the DAG.

    Note: the downward task in the task receives inputs
    from the tasks in the group in the order of appearance in that group.
    i.e. Taskgroup(task1, task2, task3) >> task4
    will call task4.func(result_1, result_2, result_3)
    where result_1, result_2, result_3 are task1's result
    and task2's result and task3's result respectively and in that order.
    """
    name: str = Field(..., description="Name of the task group (required)")
    elements: List[Union["Task", "TaskGroup"]] = Field(
        default_factory=list,
        description="List of consituent items: tasks and/or taskgroups"
    )
    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique ID for graph rendering"
    )
    exec_id: Optional[str] = None


    # Private attributes (not validated or included in .dict())
    _log: logging.Logger = PrivateAttr(default_factory=logging.getLogger)
    _parents: List["Task"] = PrivateAttr(default_factory=list)
    _children: List["Task"] = PrivateAttr(default_factory=list)
    # in case of a taskgroup itself part of a taskgroup
    _task_group: Optional["TaskGroup"] = None


    def __init__(self, *args, **kwargs):
        name = kwargs.pop("name", None)

        # If there are positional args, use them as the elements list
        if name is None:
            if not args:
                raise TypeError("Missing required 'name' argument.")
            name, *elements = args
        else:
            elements = list(args)

        if not isinstance(name, str):
            raise TypeError("The 'name' must be a string.")

        # Assign values to kwargs
        kwargs["name"] = name
        kwargs["elements"] = elements
        super().__init__(**kwargs)

        for elmt in self.elements:
            elmt._task_group = self

        logging.getLogger().info(
            f"[red on white]TaskGroup {self}[/red on white]"
        )


    @property
    def task_group(self) -> "TaskGroup":
        return self._task_group


    def __rshift__(self, other):
        """Operator overloading for '>>' to connect tasks in the DAG.

        Allows chaining: a >> b >> c or a >> TaskGroup(b, c).
        """
        # execution_id cascading
        dao = DAO(os.environ["RP_METADATASTORE_URL"])

        if self.exec_id is None:
            # SHOULD NEVER OCCUR !
            # (since in the most extreme case
            #  of that group opening the DAG,
            #  the "start" node cascaded to it)
            self.exec_id = dao.add_execution()
            self.log.info(f"[bold red]{self.func.__name__} has no exec_id[/]")
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

    def _get_elements_names(self):
        return [
            elmt.name if isinstance(elmt, TaskGroup)
            else elmt.func.__name__
            for elmt in self.elements
        ]

    def __str__(self):
        return f"TaskGroup({self.name}{self._get_elements_names()})"

    def __repr__(self):
        return self.__str__()


# ---- Decorators for Task Declaration ----


def task(func=None, *, merge_func=None):
    """Decorator for regular (non-parallel) tasks.

    Optionally takes a merge_func for merging results from parallel tasks.
    """

    def decorator(f):
        t = Task(func=f, is_parallel=False, merge_func=merge_func)

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)

        wrapper._task = t
        return t  # Return the Task object instead of the wrapper

    return decorator(func) if func else decorator


def parallel_task(func=None):
    """Decorator for parallel tasks."""

    def decorator(f):
        t = Task(func=f, is_parallel=True, merge_func=None)

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)

        wrapper._task = t
        return t  # Return the Task object instead of the wrapper

    return decorator(func) if func else decorator


def taskgroup(func=None, *, merge_func=None):
    """Decorator for task groups."""
    def decorator(f):
        tasks = f()

        if isinstance(tasks, (list, tuple)):
            tg = TaskGroup(func.__name__, *tasks, merge_func=merge_func)
        else:
            tg = TaskGroup(func.__name__, tasks, merge_func=merge_func)

        return tg

    return decorator(func)


# ---- DAG Traversal and Execution Utilities ----


# Type for task input/output data
TaskData = Dict[str, Any]

class TaskPayload:
    """
    Class for tasks data exchanges. One's output and other's input.

    Behaves similarly to a dict, with `task.func.__name` as key.

    For instance, with DAG
        A >> B
    `A` returns a TaskPayload object and `def B(x)` can access
    the result of `A`'s `return` statement via either :
        `x["A"]` or `x.get("A")`
    Note that, in cases (like in the above toy example)
    where `B` has only 1 direct parent (above, `A`), then we have
    the below equivalences, which allows for the following shorthand :
        `x["A"] == x.get("A") == x`
    """
    def __init__(self, data: Dict[str, Any]):
        self._data = data

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def copy(self):
        return TaskPayload(self._data.copy())

    def __bool__(self):
        return bool(self._data)

    def __eq__(self, other):
        if len(self._data) == 1:
            return list(self._data.values())[0] == other
        return super().__eq__(other)

    def __hash__(self):
        if len(self._data) == 1:
            return hash(list(self._data.values())[0])
        return hash(tuple(sorted(self._data.items())))

    def __len__(self):
        if len(self._data) == 1:
            value = list(self._data.values())[0]
            return len(value) if hasattr(value, '__len__') else 1
        return len(self._data)

    def __iter__(self):
        if len(self._data) == 1:
            value = list(self._data.values())[0]
            if (
                hasattr(value, '__iter__') and
                not isinstance(value, (str, bytes))
            ):
                return iter(value)
            return iter([value])
        return iter(self._data)

    def __add__(self, other):
        if len(self._data) == 1:
            return list(self._data.values())[0] + other
        return NotImplemented

    def __radd__(self, other):
        if len(self._data) == 1:
            return other + list(self._data.values())[0]
        return NotImplemented

    def __mul__(self, other):
        if len(self._data) == 1:
            return list(self._data.values())[0] * other
        return NotImplemented

    def __rmul__(self, other):
        if len(self._data) == 1:
            return other * list(self._data.values())[0]
        return NotImplemented

    def __getattr__(self, name):
        if len(self._data) == 1:
            value = list(self._data.values())[0]
            return getattr(value, name)
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __str__(self):
        if len(self._data) == 1:
            return f"TaskPayload(list(self._data.values())[0])"
        return f"TaskPayload(self._data)"

    def __repr__(self):
        return self.__str__()


def _find_root_tasks(task) -> list[Task]:
    """Find all root tasks in the DAG starting from the given task."""
    all_tasks = set()
    stack = [task]
    while stack:
        current = stack.pop()
        if current not in all_tasks:
            all_tasks.add(current)
            stack.extend(current.parents)
    return [t for t in all_tasks if not t.parents]


def _get_all_tasks(task, seen=None) -> list[Task]:
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
                tasks += _get_all_tasks(t, seen)
        else:
            tasks += _get_all_tasks(child, seen)
    return tasks


def _topological_sort(tasks) -> list[Task]:
    """Standard Kahn's algorithm for topological sorting of the DAG.

    Ensures tasks are executed in dependency order.
    """
    all_tasks = set()
    for t in tasks:
        all_tasks.update(_get_all_tasks(t))
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

#################   TMP   #######################


# def _topological_sort2(tasks) -> list:
    # sorted_tasks = _topological_sort(tasks)

    # order = []
    # for t in reversed(sorted_tasks):
        # if not (taskgroup:= t.task_group):
            # order.insert(0, t)
        # elif not (taskgroup.task_group):
            # # keeping first-level task-groups only
            # # print(f"taskgroup : {taskgroup}")
            # order.insert(0, taskgroup)

    # # drop duplicates
    # ordered_set = list(dict.fromkeys(order))
    # return ordered_set


def _topological_sort2(tasks) -> list:
    sorted_tasks = _topological_sort(tasks)

    order = []
    for t in sorted_tasks:
        if not (taskgroup:= t.task_group):
            order.append(t)
        elif not (taskgroup.task_group):
            # keeping first-level task-groups only
            # print(f"taskgroup : {taskgroup}")
            order.append(taskgroup)

    # drop duplicates
    ordered_set = list(dict.fromkeys(order))
    return ordered_set


#################   TMP   #######################

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


def _find_subdag_end(task_order, start_idx):
    """Find where the parallel subdag ends, e.g.
    at the next merge task, accounting for potential nesting."""
    parallel_depth = 0
    for i in range(start_idx, len(task_order)):
        if task_order[i].is_parallel:
            parallel_depth += 1
        elif task_order[i].merge_func:
            parallel_depth -= 1
            if parallel_depth == 0:
                return i

    return len(task_order)


def _collect_parent_results(t, results):
    """
    Collects results from parent tasks for a given task.

    Params:
        - t (Task):
            The current task.
        - results (TaskPayload):
            The dictionary of previously computed results.

    Resuts:
        - TaskPayload:
            A payload containing
            the results from all available parent tasks.
    """
    parent_results = TaskPayload({})
    for p in t.parents:
        if p.func.__name__ in results:
            parent_results[p.func.__name__] = results[p.func.__name__]
    return parent_results


def _parallel_input_count(parent_results):
    """
    Determines how many parallel executions are needed
    based on the parent results.

    Params:
        - parent_results (TaskPayload):
            The payload of inputs from parent tasks.

    Results:
        - int:
            The number of parallel branches to execute.
    """
    if not parent_results:
        return 1
    first = list(parent_results.values())[0]
    return len(first) if isinstance(first, list) else 1


def _run_parallel_branches(subdag_tasks, parent_results, count, branch_index=None):
    """
    Executes a subdag in parallel across multiple input branches.

    Params:
        - subdag_tasks (List[Task]):
            The tasks forming the parallel subdag.
        - parent_results (TaskPayload):
            Input values for the branches.
        - count (int):
            Number of parallel executions to perform.
        - branch_index (List[int], optional):
            Index path for nested branches.

    Results:
        - List[Any]:
            Results from each parallel branch.
    """
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for idx in range(count):
            branch_input = TaskPayload({
                k: (v[idx] if isinstance(v, list) else v)
                for k, v in parent_results.items()
            })
            futures.append(executor.submit(
                _execute_branch,
                subdag_tasks,
                branch_input,
                (branch_index or []) + [idx]
            ))
        return [f.result() for f in futures]


def _run_regular_task(t, parent_results, index=None):
    """
    Executes a single non-parallel task,
    optionally applying a merge function.

    Params:
        - t (Task):
            The task to execute.
        - parent_results (TaskPayload):
            Input values from parent tasks.
        - index (List[int], optional):
            Branch index for logging
            and function context.

    Results:
        - Any:
            The result produced by the task.
    """
    if t.merge_func:
        merged = t.merge_func(list(parent_results.values())[0])
        parent_results = TaskPayload({t.parents[0].func.__name__: merged})

        t.log.info(
            f"[#FFFFE0]`{t.func.__name__}{index if index else ''} merged "
            f" {t.merge_func.__name__}(parent_results) :\n"
            f"Inputs :\n"
            f"  \N{BULLET} {str(parent_results)}[/]"
        )

    t.log.info(f"parent_results: {parent_results}")

    if parent_results._data:
        return t.func(parent_results, index=index) \
               if index \
               else t.func(parent_results)
    else:
        return t.func(index=index) if index else t.func()


def execute(task: Task, input_data: TaskPayload = None):
    """From the start, executes the DAG that contains the given task.

    Handles task-groups, parallel and nested parallel tasks,
    and merges results as needed.
    """
    _rich_log_execution_id_with_timestamp(task.exec_id)

    roots = _find_root_tasks(task)
    print(f"Root tasks: {[t.func.__name__ for t in roots]}")

    order = _topological_sort(roots)
    print(f"Topological order:   {[t.func.__name__ for t in order]}")
    print(f"Topological order 2: {[(e.func.__name__ if isinstance(e, Task) else e.name) for e in _topological_sort2(roots)]}")

    # raise Exception("DEBUG")

    results = TaskPayload({})
    i = 0

    while i < len(order):
        t = order[i]

        t.log.info("Executing task: " +
                   f"[rgb(0,255,255) on #af00ff]{t.func.__name__}[/]")

        parent_results = _collect_parent_results(t, results)

        if t.is_parallel:
            t.log.info(f"parent_results: {parent_results}")

            # Find the end of this parallel subdag
            subdag_end = _find_subdag_end(order, i)
            subdag_tasks = order[i:subdag_end]
            t.log.info("subdag_tasks [red on yellow]{}[/]".format(
                [t.func.__name__ for t in subdag_tasks]))

            # Execute parallel subdag for each input
            input_count = _parallel_input_count(parent_results)
            sub_results = _run_parallel_branches(
                subdag_tasks, parent_results, input_count)

            # Store results for the last task in subdag
            results[subdag_tasks[-1].func.__name__] = sub_results

            # Skip to after the subdag
            i = subdag_end
            continue
        else:
            result = _run_regular_task(t, parent_results)
            results[t.func.__name__] = result

        i += 1

    return results[order[-1].func.__name__]


def _execute_branch(branch_tasks, branch_input: TaskPayload, branch_index):
    """Execute a single parallel branch with recursive subdag handling."""
    branch_results = branch_input.copy()
    i = 0

    while i < len(branch_tasks):
        t = branch_tasks[i]
        t.rank = branch_index

        parent_results = _collect_parent_results(t, branch_results)

        if t.is_parallel and i > 0:
            t.log.info(f"parent_results: {parent_results}")

            # Find nested subdag end
            nested_subdag_end = _find_subdag_end(branch_tasks, i)
            nested_subdag_tasks = branch_tasks[i:nested_subdag_end]
            t.log.info("nested_subdag_tasks [red on yellow]{}{}[/]".format(
                branch_index, [t.func.__name__ for t in nested_subdag_tasks]))

            # Execute nested parallel subdag
            parent_value = list(parent_results.values())[0]
            if isinstance(parent_value, list):
                nested_results = _run_parallel_branches(
                    nested_subdag_tasks, parent_results,
                    len(parent_value), branch_index)
            else:
                nested_results = _execute_branch(
                    nested_subdag_tasks, parent_results, branch_index)

            branch_results[nested_subdag_tasks[-1].func.__name__] = nested_results
            i = nested_subdag_end
            continue
        else:
            # Regular task
            t.log.info("Executing task: " +
                       f"[rgb(0,255,255) on #af00ff]{t.func.__name__}{branch_index}[/]")
            result = _run_regular_task(t, parent_results, branch_index)
            branch_results[t.func.__name__] = result

        i += 1

    return branch_results[branch_tasks[-1].func.__name__]


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

    roots = _find_root_tasks(task)
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

    roots = _find_root_tasks(task)
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

    roots = _find_root_tasks(task)
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

