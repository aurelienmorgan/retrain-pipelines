
import os
import sys
import uuid
import getpass
import inspect
import logging
import textwrap
import functools
import concurrent.futures

from datetime import datetime

from typing import Callable, List, Optional, Union, \
    Dict, Any
from pydantic import BaseModel, Field, PrivateAttr, \
    model_validator

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

    rank: Optional[List[int]] = Field(
        default=None,
        description=(
            "In case of a task part of a parallel lane (can be nested): "
            "indice n corresponds to the rank to which the task belongs "
            "after parallel task n in the chain of nested parallelism."
        )
    )

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
    def parents(self) -> List["Task"]:
        return self._parents


    @property
    def children(self) -> List["Task"]:
        return self._children


    @property
    def task_group(self) -> "TaskGroup":
        return self._task_group


    def _wrap_merge_func(self, merge_func):
        """Wrap the function

        with logging and Exception handling."""

        @functools.wraps(merge_func)
        def wrapper(*args, **kwargs):
            dao = DAO(os.environ["RP_METADATASTORE_URL"])
            self.task_id = dao.add_task(
                exec_id=self.exec_id,
                start_timestamp=datetime.now()
            )

            self.log.info(
                framed_rich_log_str(
                    f"\N{wrench} Executing Merge "
                    f"[#D2691E]`{self.merge_func.__name__}`[/] of task "
                    f"[#D2691E]`{self.func.__name__}[{self.task_id}]`[/]:\n"
                    f"Inputs :\n"
                    f"  \N{BULLET} Positional: {args}\n"
                    f"  \N{BULLET} Keyword   : {kwargs}",
                    border_color="#FFFFE0"
                )
            )

            try:
                result = merge_func(*args, **kwargs)
            except Exception as ex:
                end_timestamp = datetime.now()
                dao.update_task(
                    id=self.task_id,
                    end_timestamp=end_timestamp,
                    failed=True
                )
                dao.update_execution(
                    id=self.exec_id,
                    end_timestamp=end_timestamp
                )
                raise TaskMergeFuncException(
                        f"merge `{merge_func.__name__}` " +
                        f"of task `{self.name}` failed"
                    ) from ex

            return result

        return wrapper


    def _wrap_func(self, func):
        """Wrap the function

        with logging and Exception handling."""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            rank = kwargs.pop("rank", None)
            dao = DAO(os.environ["RP_METADATASTORE_URL"])

            if self.merge_func is None :
                self.task_id = dao.add_task(
                    exec_id=self.exec_id,
                    start_timestamp=datetime.now()
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
                    f"[#D2691E]`{self.name}[{self.task_id}]"
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
                    f"\n\N{WHITE HEAVY CHECK MARK} Completed Task [#D2691E]`{self.name}[{self.task_id}]{f'[{rank}]' if rank is not None else ''}`[/]:\n"
                    f"Results :\n" +
                    f"  \N{BULLET} {result} {f'({type(result).__name__})' if result is not None else ''}\n"
                )
            except Exception as ex:
                task_failed = True
                raise TaskFuncException(
                    f"task `{self.name}` failed") from ex
            finally:
                end_timestamp = datetime.now()
                dao.update_task(
                    id=self.task_id,
                    end_timestamp=end_timestamp,
                    failed=task_failed
                )
                if task_failed or (len(self.children) == 0):
                    dao.update_execution(
                        id=self.exec_id,
                        end_timestamp=end_timestamp
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
            # Get the calling frame (1 level up)
            frame = inspect.currentframe()
            caller_frame = frame.f_back
            coller_module_name = os.path.basename(
                    caller_frame.f_code.co_filename
                ).split(".")[-2]
            self.exec_id = dao.add_execution(
                name=coller_module_name,
                username=getpass.getuser(),
                start_timestamp=datetime.now()
            )
            self.log.info(f"[bold red]{self.name} has no exec_id[/]")
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
        return (f"Task({self.name!r}, is_parallel={self.is_parallel}, "
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
        # execution_id cascading
        dao = DAO(os.environ["RP_METADATASTORE_URL"])

        if self.exec_id is None:
            # SHOULD NEVER OCCUR !
            # (since in the most extreme case
            #  of that group opening the DAG,
            #  the "start" node cascaded to it)
            self.exec_id = dao.add_execution()
            self.log.info(f"[bold red]{self.name} has no exec_id[/]")
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
        return [elmt.name for elmt in self.elements]

    def __str__(self):
        return f"{self.name}{self._get_elements_names()}"

    def __repr__(self):
        return f"TaskGroup({self.__str__()})"


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


def taskgroup(func):
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

        if isinstance(tasks, (list, tuple)):
            tg = TaskGroup(f.__name__, *tasks)
        else:
            tg = TaskGroup(f.__name__, tasks)

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
        return f"TaskPayload({self._data})"

    def __repr__(self):
        return self.__str__()

