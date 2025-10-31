
import os
import re
import sys
import getpass
import inspect
import logging
import textwrap
import functools
import concurrent.futures

from uuid import UUID, uuid4
from collections import deque
from datetime import datetime, timezone

from typing import Callable, List, Optional, Union, \
    Dict, Any, Tuple
from pydantic import BaseModel, Field, PrivateAttr, \
    field_validator

from .db.dao import DAO
from ..utils.rich_logging import framed_rich_log_str


# ---- Core Task and Execution Infrastructure ----


class TaskFuncException(Exception):
    """Exception raised when a task function fails."""
    def __init__(self, message):
        super().__init__(message)

class TaskMergeFuncException(Exception):
    """Exception raised when a task merge-function fails."""
    def __init__(self, message):
        super().__init__

class TaskGroupException(Exception):
    """Exception raised when declaring a TaskGroup."""
    def __init__(self, message):
        super().__init__(message)


class TaskType(BaseModel):
    """Represents a node in the DAG.

    Note: The class is instantiated at DAG declaration time.
          Attributes related to DAG execution.
          `task_id`, `exec_id` and `rank` are passed
          at runtime but not stored.
    Note: at runtime, rank serves to track sub-DAG
          split line depth and indices.
          In case of a task part of a parallel lane,
          indice n corresponds to the rank to which the task belongs
          after parallel task n in the chain of nested parallelism.
    """
    func: Callable
    is_parallel: bool = False
    merge_func: Optional[Callable] = Field(default=None)

    docstring: Optional[str] = Field(default=None)
    ui_css: Optional["UiCss"] = Field(default=None)

    tasktype_uuid: UUID = Field(default_factory=uuid4,
                                description="Unique ID for graph rendering")

    # Private attributes (not validated or included in .dict())
    _log: logging.Logger = PrivateAttr(default_factory=logging.getLogger)
    _parents: List["TaskType"] = PrivateAttr(default_factory=list)
    _children: List["TaskType"] = PrivateAttr(default_factory=list)
    # in case of a task part of a taskgroup
    _task_group: Optional["TaskGroup"] = PrivateAttr(default=None)


    def __init__(self, **data):
        super().__init__(**data)
        self._log.info(
            "[red on white]" +
            f"{self.name}" +
            f"{f' ([#6f00d6 on white]parallel[/])' if self.is_parallel else f' ([#2d97ba on white]merge[/])' if self.merge_func else ''}" +
            "[/]")
        self.func = self._wrap_func(self.func)
        if self.merge_func is not None:
            self.merge_func = self._wrap_merge_func(self.merge_func)


    @property
    def name(self) -> str:
        """Publicly expose the logger."""
        return self.func.__name__

    @property
    def log(self) -> logging.Logger:
        """Publicly expose the logger."""
        return self._log


    @property
    def parents(self) -> List["TaskType"]:
        return self._parents


    @property
    def children(self) -> List["TaskType"]:
        return self._children


    @property
    def task_group(self) -> "TaskGroup":
        return self._task_group


    def _wrap_merge_func(self, merge_func):
        """Wrap the function

        with logging and Exception handling.

        Returns: task_id for use by the main func wrapper
        """

        @functools.wraps(merge_func)
        def wrapper(*args, **kwargs):
            rank = kwargs.pop("rank", None)
            exec_id = kwargs.pop("exec_id", None)
            dao = DAO(os.environ["RP_METADATASTORE_URL"])
            task_id = dao.add_task(
                exec_id=exec_id,
                tasktype_uuid=self.tasktype_uuid,
                rank=rank,
                start_timestamp=datetime.now(timezone.utc)
            )

            self.log.info(
                framed_rich_log_str(
                    f"\N{wrench} Executing Merge "
                    f"[#D2691E]`{self.merge_func.__name__}`[/] of task "
                    f"[#D2691E]`{self.name}[{task_id}]"
                    f"{f'[{rank}]' if rank is not None else ''}`[/]:\n"
                    f"Inputs :\n"
                    f"  \N{BULLET} Positional: {args}\n"
                    f"  \N{BULLET} Keyword   : {kwargs}",
                    border_color="#FFFFE0"
                )
            )

            try:
                result = merge_func(*args, **kwargs)
            except Exception as ex:
                end_timestamp = datetime.now(timezone.utc)
                dao.update_task(
                    id=task_id,
                    end_timestamp=end_timestamp,
                    failed=True
                )
                dao.update_execution(
                    id=exec_id,
                    end_timestamp=end_timestamp
                )
                raise TaskMergeFuncException(
                        f"merge `{merge_func.__name__}` " +
                        f"of task `{self.name}` failed"
                    ) from ex

            return task_id, result

        return wrapper


    def _wrap_func(self, func):
        """Wrap the function

        with logging and Exception handling.

        Returns: tuple of (task_id, result)
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            rank = kwargs.pop("rank", None)
            exec_id = kwargs.pop("exec_id", None)
            task_id = kwargs.pop("task_id", None)  # Get task_id from merge_func if it exists
            dao = DAO(os.environ["RP_METADATASTORE_URL"])

            if task_id is None:
                # Only create task_id if merge_func didn't create it
                task_id = dao.add_task(
                    exec_id=exec_id,
                    tasktype_uuid=self.tasktype_uuid,
                    rank=rank,
                    start_timestamp=datetime.now(timezone.utc)
                )

            docstring = '\n'.join(
                f'[bold green]{line}[/]'
                for line in filter(None, [
                    func.__doc__.splitlines()[0],
                    *textwrap.dedent(
                            '\n'.join(func.__doc__.splitlines()[1:])
                        ).splitlines()
                ])) \
                if func.__doc__ else None

            self.log.info(
                framed_rich_log_str(
                    f"\N{wrench} Executing Task "
                    f"[#D2691E]`{self.name}[{task_id}]"
                    f"{f'[{rank}]' if rank is not None else ''}`[/]:\n"
                    f"Inputs :\n"
                    f"  \N{BULLET} Positional: {args}\n"
                    f"  \N{BULLET} Keyword   : {kwargs}" +
                    (f"\n[bold green]{docstring}[/]" if docstring else ""),
                    border_color="#FFFFE0"
                )
            )

            task_failed = False
            try:
                result = func(*args, **kwargs)
                self.log.info(
                    f"\n\N{WHITE HEAVY CHECK MARK} Completed Task [#D2691E]`{self.name}[{task_id}]{f'[{rank}]' if rank is not None else ''}`[/]:\n"
                    f"Results :\n" +
                    f"  \N{BULLET} {result} {f'({type(result).__name__})' if result is not None else ''}\n"
                )
            except Exception as ex:
                task_failed = True
                raise TaskFuncException(
                    f"task `{self.name}` failed") from ex
            finally:
                end_timestamp = datetime.now(timezone.utc)
                dao.update_task(
                    id=task_id,
                    end_timestamp=end_timestamp,
                    failed=task_failed
                )
                if task_failed or (len(self.children) == 0):
                    dao.update_execution(
                        id=exec_id,
                        end_timestamp=end_timestamp
                    )

            return task_id, result

        return wrapper


    def __rshift__(self, other):
        """Operator overloading for '>>' to connect tasks in the DAG.

        Allows chaining: a >> b >> c or a >> TaskGroup(b, c).
        """
        if isinstance(other, TaskType):
            self.children.append(other)
            other.parents.append(self)
            return other
        elif isinstance(other, TaskGroup):
            for element in other.elements:
                self._add_child(element)
            return other
        else:
            raise TypeError(
                "The right-hand side of '>>' must be a TaskType object, or a TaskGroup object.")


    def _add_child(self, child):
        """Recursively add children to the task."""
        if isinstance(child, TaskType):
            self.children.append(child)
            child.parents.append(self)
        elif isinstance(child, TaskGroup):
            for child_element in child.elements:
                self._add_child(child_element)

    def __hash__(self):
        return hash(self.tasktype_uuid)

    def __eq__(self, other):
        if isinstance(other, TaskType):
            return self.tasktype_uuid == other.tasktype_uuid
        return False

    def __str__(self):
        merge_name = self.merge_func.__name__ if self.merge_func else None
        return (f"Task({self.name!r}, is_parallel={self.is_parallel}, "
                f"merge_func={merge_name!r}, "
                f"tasktype_uuid={self.tasktype_uuid!r})")

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
    docstring: Optional[str] = Field(default=None)
    ui_css: Optional["UiCss"] = Field(default=None)

    elements: List[Union["TaskType", "TaskGroup"]] = Field(
        default_factory=list,
        description="List of consituent items: tasks and/or taskgroups"
    )
    uuid: UUID = Field(default_factory=uuid4,
                       description="Unique ID for graph rendering")
    exec_id: Optional[int] = Field(default = None)

    # Private attributes (not validated or included in .dict())
    _log: logging.Logger = PrivateAttr(default_factory=logging.getLogger)
    _parents: List["TaskType"] = PrivateAttr(default_factory=list)
    _children: List["TaskType"] = PrivateAttr(default_factory=list)
    # in case of a taskgroup itself part of a taskgroup
    _task_group: Optional["TaskGroup"] = PrivateAttr(default = None)


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
        try:
            super().__init__(**kwargs)
        except Exception as ex:
            # typical error include pydantic validation
            raise TaskGroupException(
                    f"Error instanciating taskgroup `{name}`"
                ) from ex

        for elmt in self.elements:
            elmt._task_group = self

        logging.getLogger().info(
            f"[red on white]{self}[/red on white]"
        )


    @property
    def task_group(self) -> "TaskGroup":
        return self._task_group


    @property
    def log(self) -> logging.Logger:
        """Publicly expose the logger."""
        return self._log


    def __rshift__(self, other):
        """Operator overloading for '>>' to connect tasks in the DAG.

        Allows chaining: a >> b >> c or a >> TaskGroup(b, c).
        """
        if isinstance(other, TaskType):
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
                "The right-hand side of '>>' must be a TaskType object, or a TaskGroup object.")

    def _add_child(self, child):
        """Recursively add children to the task or task group."""
        if isinstance(child, TaskType):
            for element in self.elements:
                if isinstance(element, TaskType):
                    element.children.append(child)
                    child.parents.append(element)
                elif isinstance(element, TaskGroup):
                    element._add_child(child)
        elif isinstance(child, TaskGroup):
            for element in self.elements:
                if isinstance(element, TaskType):
                    for sub_child in child.elements:
                        element.children.append(sub_child)
                        sub_child.parents.append(element)
                elif isinstance(element, TaskGroup):
                    element._add_child(child)

    def __hash__(self):
        return hash(self.uuid)

    def __eq__(self, other):
        if isinstance(other, TaskGroup):
            return self.uuid == other.uuid
        return False

    def _get_elements_names(self):
        return [elmt.name for elmt in self.elements]

    def __str__(self):
        return f"{self.name}{self._get_elements_names()}"

    def __repr__(self):
        return f"TaskGroup({self.__str__()})"


class DAG(BaseModel):
    """Represents a DAG.

    Note: The class is instantiated at DAG declaration time.
          Attributes related to DAG execution
          (such as `exec_id` and `exec_params`)
          are populated at execution time.
    """
    roots: List["TaskType"] = Field(...)
    docstring: Optional[str] = Field(default=None)
    ui_css: Optional["UiCss"] = Field(default=None)

    exec_id: Optional[int] = Field(default=None)
    exec_params: Optional[Dict[str, Any]] = Field(default_factory=dict)


    def __init__(self, *args, **kwargs):
        task_anchor = kwargs.pop('task_anchor')
        roots = DAG._find_root_tasks(task_anchor)
        super().__init__(*args, roots=roots, **kwargs)


    def init(self):
        """Shall be called programmatically 
        by the DAG execution routine.
        """
        # Get the calling frame (2 levels up)
        frame = inspect.currentframe()
        caller_frame = frame.f_back.f_back
        caller_module_name = os.path.basename(
                caller_frame.f_code.co_filename
            ).split(".")[-2]

        dao = DAO(os.environ["RP_METADATASTORE_URL"])
        self.exec_id = dao.add_execution(
            name=caller_module_name,
            docstring=self.docstring,
            username=getpass.getuser(),
            ui_css=self.ui_css.__dict__ if self.ui_css else None,
            start_timestamp=datetime.now(timezone.utc)
        )
        tasktypes_list, taskgroups_list = self.to_elements_lists()
        for i, tasktype in enumerate(tasktypes_list):
            dao.add_tasktype(exec_id=self.exec_id, order=i,
                             **tasktype)
        for i, taskgroup in enumerate(taskgroups_list):
            dao.add_taskgroup(exec_id=self.exec_id, order=i,
                             **taskgroup)


    def to_elements_lists(
        self, serializable: bool = False
    ) -> (List[Dict[str, Any]], List[Dict[str, Any]]):
        """
        Returns a human-readable, (optionally serializable)
        DAG structure using breadth-first traversal.
        Two topologically-ordered lists

        Example output (not all TaskType & TaskGroup
        attributes shown):
        [
            {"uuid": 1, "name": "Root", "children": [2, 3]},
            {"uuid": 2, "name": "Child A", "children": [4]},
            {"uuid": 3, "name": "Child B", "children": []},
            {"uuid": 4, "name": "Leaf", "children": []}
        ],
        [
            {"uuid": 1, "name": "taskgroup A", "elements": [2, 3]}
        ]
        """
        visited = set()
        tasktypes_list = []
        queue = deque(self.roots)

        taskgroups_dict = {}

        while queue:
            # Process all nodes at current level
            current_level_size = len(queue)
            for _ in range(current_level_size):
                task = queue.popleft()

                if task.tasktype_uuid in visited:
                    continue
                visited.add(task.tasktype_uuid)

                tasktypes_list.append(
                    {
                        "uuid": str(task.tasktype_uuid) if serializable else task.tasktype_uuid,
                        "name": task.name,
                        "docstring": task.func.__doc__,
                        "ui_css": task.ui_css.to_dict() if task.ui_css else None,
                        "is_parallel": task.is_parallel,
                        "merge_func": (
                            {"name": task.merge_func.__name__,
                             "docstring": task.merge_func.__doc__}
                            if task.merge_func is not None
                            else None
                        ),
                        "taskgroup_uuid": (
                            str(task.task_group.uuid) if serializable
                            else task.task_group.uuid
                        ) if task.task_group else None,
                        "children": [str(c.tasktype_uuid) for c in task.children]
                    }
                )

                if task.task_group and task.task_group.uuid not in taskgroups_dict:
                    taskgroups_dict[task.task_group.uuid] = {
                        "uuid": str(task.task_group.uuid) if serializable else task.task_group.uuid,
                        "name": task.task_group.name,
                        "docstring": task.task_group.docstring,
                        "ui_css": task.task_group.ui_css.to_dict() if task.task_group.ui_css else None,
                        "elements": [
                            str(e.tasktype_uuid) if isinstance(e, TaskType)
                            else str(e.uuid) # inner TaskGroup
                            for e in task.task_group.elements
                        ]
                    }

                # Enqueue children to process in next level
                for child in task.children:
                    if child.tasktype_uuid not in visited:
                        queue.append(child)

        return tasktypes_list, list(taskgroups_dict.values())


    @staticmethod
    def _find_root_tasks(task: TaskType) -> list[TaskType]:
        """Find all root tasks in the DAG starting from the given task."""
        all_tasks = set()
        stack = [task]
        while stack:
            current = stack.pop()
            if current not in all_tasks:
                all_tasks.add(current)
                stack.extend(current.parents)
        return [t for t in all_tasks if not t.parents]


# ---- Decorators for Task Declaration ----


def task(func=None, *, merge_func=None, ui_css=None):
    """Decorator for regular (non-parallel) tasks.

    Optionally takes a merge_func for merging results from parallel tasks.
    """

    def decorator(f):
        t = TaskType(
            func=f,
            is_parallel=False,
            merge_func=merge_func,
            docstring=f.__doc__,
            ui_css=ui_css
        )

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)

        wrapper._task = t
        return t  # Return the TaskType object instead of the wrapper

    return decorator(func) if func else decorator


def parallel_task(func=None, ui_css=None):
    """Decorator for parallel tasks."""

    def decorator(f):
        t = TaskType(
            func=f,
            is_parallel=True,
            merge_func=None,
            docstring=f.__doc__,
            ui_css=ui_css
        )

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)

        wrapper._task = t
        return t  # Return the TaskType object instead of the wrapper

    return decorator(func) if func else decorator


def taskgroup(func=None, *, ui_css=None):
    """Decorator for task groups."""
    def decorator(f):
        try:
            tasks = f()
        except Exception as ex:
            # REMARK : we're at DAG-construct time here
            #          (before any execution has even been requested)
            ## retrieve taskgroup name from stacktrace =>
            tb = sys.exc_info()[2]
            while tb.tb_next:
                tb = tb.tb_next
            taskgroup_name = tb.tb_frame.f_code.co_name
            raise TaskGroupException(
                    f"taskgroup `{taskgroup_name}` failed to construct"
                ) from ex

        tg = TaskGroup(
            name=f.__name__,
            docstring=f.__doc__,
            ui_css=ui_css,
            *tasks if isinstance(tasks, (list, tuple)) else tasks
        )

        return tg

    if func is None:
        # Called as @dag(...) with optional arguments
        return decorator
    else:
        # Called as @dag without parentheses
        return decorator(func)


def dag(func=None, *, ui_css=None):
    """Decorator factory for a DAG."""
    def decorator(f):
        task_anchor = f()
        pipeline = DAG(
            task_anchor=task_anchor,
            docstring=f.__doc__,
            ui_css=ui_css
        )
        return pipeline

    if func is None:
        # Called as @dag(...) with optional arguments
        return decorator
    else:
        # Called as @dag without parentheses
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
        return f"TaskPayload({self._data})"

    def __repr__(self):
        return self.__str__()


class UiCss(BaseModel):
    background: Optional[str] = Field(
        default=None,
        description=(
            "The color of the background of the record "
            "in the WebConsole's 'executions-list' page. "
            "Must be a hex color code starting with #."
        )
    )
    color: Optional[str] = Field(
        default=None,
        description=(
            "The font color of the record "
            "in the WebConsole's 'executions-list' page. "
            "Must be a hex color code starting with #."
        )
    )
    border: Optional[str] = Field(
        default=None,
        description=(
            "The color of the border of the record "
            "in the WebConsole's 'executions-list' page. "
            "Must be a hex color code starting with #."
        )
    )

    @field_validator("background", "color", "border")
    def must_be_hex_color(cls, v, info):
        if (
            v is not None and
            not re.match(r'^#([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$', v)
        ):
            raise ValueError(
                f"{cls.__name__}.{info.field_name} got invalid value {v!r}:" +
                " must be a hex color code starting with #."
            )
        return v

    def to_dict(self) -> dict:
        # Filter out None values before serializing
        data = {k: v for k, v in self.dict().items() if v is not None}
        return data

