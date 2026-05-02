
import os
import re
import io
import sys
import ast
import copy
import getpass
import inspect
import logging
import textwrap
import functools
import threading
import traceback
import contextvars
import concurrent.futures

from uuid import UUID, uuid4
from collections import deque
from contextlib import contextmanager
from datetime import datetime, timezone, date

from rich.markup import escape
from rich.console import Console
from rich.logging import RichHandler

from typing import Callable, List, Optional, Union, \
    Dict, Any, Tuple

from pydantic import BaseModel, Field, PrivateAttr, \
    field_validator, ValidationInfo

from ..db.dao import DAO
from .trace_buffer import get_trace_buffer
from ...utils import in_notebook
from ...utils.rich_logging import framed_rich_log_str


logger = logging.getLogger(__name__)


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


tasktype_taskgroup_console = Console(
    width=None, force_terminal=True, legacy_windows=False)
class RawRichHandler(RichHandler):
    def emit(self, record):
        msg = self.format(record)
        tasktype_taskgroup_console.print(
            msg, soft_wrap=False)
tasktype_taskgroup_rich_handler = \
    RawRichHandler(show_time=False, 
                   show_level=False, show_path=False)
tasktype_taskgroup_rich_handler.setFormatter(
    logging.Formatter("%(message)s"))

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
        self._log = logger

        # add custom-rich handler
        # (since package-logger doesn't hold one when we're here)
        self._log.addHandler(tasktype_taskgroup_rich_handler)
        self._log.setLevel(logging.INFO)
        self._log.info(
            "[red on white]" +
            f"{self.name}" +
            f"{f' ([#6f00d6 on white]parallel[/])' if self.is_parallel else f' ([#2d97ba on white]merge[/])' if self.merge_func else ''}" +
            "[/]")
        self._log.removeHandler(self._log.handlers[-1])
        self._log.setLevel(logging.NOTSET)

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


    @staticmethod
    @contextmanager
    def _capture_and_stream_trace(task_id):
        """Capture stdout/stderr/logging AND C-level output, streaming to DB."""

        # Save original file descriptors
        old_stdout_fd = os.dup(1)
        old_stderr_fd = os.dup(2)

        # Create pipes for capturing C-level output
        stdout_pipe_read, stdout_pipe_write = os.pipe()
        stderr_pipe_read, stderr_pipe_write = os.pipe()

        # Redirect file descriptors at OS level
        os.dup2(stdout_pipe_write, 1)
        os.dup2(stderr_pipe_write, 2)

        # Redirect Python streams
        old_out = sys.stdout
        old_err = sys.stderr
        capturer_out = StreamToDb(old_out, task_id, is_err=False)
        capturer_err = StreamToDb(old_err, task_id, is_err=True)
        sys.stdout = capturer_out
        sys.stderr = capturer_err

        ##############################################################
        # In notebook environments (e.g. Google Colab / Jupyter),
        # original_stream is ipykernel.iostream.OutStream which routes
        # "writes" via ZMQ rather than the OS-level file descriptor,
        # so the (below) "pipe_reader" threads never receive the data.
        # Detect this and patch the StreamToDb instances so their
        # write() additionally pushes data into the pipes, letting
        # the existing pipe_reader threads capture notebook output
        # exactly as they already do for CLI via the FD dup2.
        try:
            from IPython import get_ipython
            _notebook_mode = (
                get_ipython() is not None and
                'IPKernelApp' in get_ipython().config
            )
        except:
            _notebook_mode = False

        if _notebook_mode:
            _orig_out_write = capturer_out.write
            _orig_err_write = capturer_err.write

            # threading.local tracks re-entrant calls per thread.
            # For pure "sys.stdout.write" calls,
            # on the outer/top-level call, rp_logging intercepts
            # and writes back formatted ANSI through a recursive call.
            # We push to pipe only on that inner (recursive) call so
            # the DB receives the ANSI-formatted string, not the raw one.
            # If no inner call occurs (rp_logging inactive), we fall
            # back to pushing raw at the end of the top-level call.
            _nb_tl = threading.local()

            def _nb_write(data, pipe_fd, orig):
                is_top = not getattr(_nb_tl, 'active', False)
                if is_top:
                    _nb_tl.active = True
                    _nb_tl.inner_pushed = False
                try:
                    result = orig(data)
                finally:
                    if is_top:
                        _nb_tl.active = False

                if data:
                    encoded = (data.encode('utf-8')
                               if isinstance(data, str) else data)
                    if not is_top:
                        # Inner/recursive call: rp_logging is writing
                        # back the formatted ANSI output - push to pipe.
                        os.write(pipe_fd, encoded)
                        _nb_tl.inner_pushed = True
                    elif not getattr(_nb_tl, 'inner_pushed', False):
                        # Top-level call with no inner push: rp_logging
                        # was not active, push raw as fallback.
                        os.write(pipe_fd, encoded)
                return result

            capturer_out.write = lambda data, _fd=stdout_pipe_write, _o=\
                                    _orig_out_write: _nb_write(data, _fd, _o)
            capturer_err.write = lambda data, _fd=stderr_pipe_write, _o=\
                                    _orig_err_write: _nb_write(data, _fd, _o)
        ##############################################################

        # Thread to read from pipes and write to trace buffer
        def pipe_reader(pipe_fd, is_err):
            trace_buffer = get_trace_buffer()
            line_buffer = ""
            try:
                while True:
                    data = os.read(pipe_fd, 1024).decode('utf-8', errors='replace')
                    if not data:
                        break
                    line_buffer += data
                    while '\n' in line_buffer:
                        line, line_buffer = line_buffer.split('\n', 1)
                        trace_timestamp = datetime.now(timezone.utc)
                        trace_microsec = trace_timestamp.microsecond%1000
                        trace_buffer.add_trace(
                            task_id=task_id,
                            content=line + '\n',
                            timestamp=trace_timestamp,
                            microsec=trace_microsec,
                            is_err=is_err
                        )
                        if not _notebook_mode:
                            # echo to console
                            orig_fd = old_stderr_fd if is_err else old_stdout_fd
                            os.write(orig_fd, (line + '\n').encode())
                if line_buffer:
                    trace_timestamp = datetime.now(timezone.utc)
                    trace_buffer.add_trace(
                        task_id=task_id,
                        content=line_buffer,
                        timestamp=trace_timestamp,
                        microsec=trace_timestamp.microsecond%1000,
                        is_err=is_err
                    )
            except Exception as ex:
                pass

        stdout_thread = threading.Thread(
            target=pipe_reader, args=(stdout_pipe_read, False), daemon=True)
        stderr_thread = threading.Thread(
            target=pipe_reader, args=(stderr_pipe_read, True), daemon=True)
        stdout_thread.start()
        stderr_thread.start()

        try:
            yield
        finally:
            # Restore everything
            sys.stdout = old_out
            sys.stderr = old_err
            os.dup2(old_stdout_fd, 1)
            os.dup2(old_stderr_fd, 2)
            os.close(stdout_pipe_write)
            os.close(stderr_pipe_write)
            stdout_thread.join(timeout=1)
            stderr_thread.join(timeout=1)
            os.close(stdout_pipe_read)
            os.close(stderr_pipe_read)
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)
            capturer_out.trace_buffer.flush()


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
















            # Register task with registry for interrupt handling
            from ..runtime import _task_registry
            import os as _os
            pid = _os.getpid()
            _task_registry.register_task(task_id, pid)
















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
                with TaskType._capture_and_stream_trace(task_id):
                    try:
                        result = merge_func(*args, **kwargs)
                    except Exception:
                        traceback.print_exc()
                        sys.stderr.flush()
                        raise
            except Exception as ex:
                end_timestamp = datetime.now(timezone.utc)
                dao.update_task(
                    id=task_id,
                    end_timestamp=end_timestamp,
                    failed=True
                )
                raise TaskMergeFuncException(
                        f"merge `{merge_func.__name__}` " +
                        f"of task `{self.name}` failed"
                    ) from ex
            finally:
                dao.dispose()
                # Unregister task from registry
                _task_registry.unregister_task(task_id)

            return task_id, result

        return wrapper


    def _wrap_func(self, func):
        """Wrap the function

        with logging and Exception handling.

        Returns: tuple of (task_id, result)
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            """ rank and task_id are optional named arguments."""
            exec_id = kwargs.pop("exec_id")
            arg_names = list(inspect.signature(func).parameters.keys())
            if not "rank" in arg_names:
                rank = kwargs.pop("rank", None)
            else:
                rank = kwargs["rank"]
            # Get task_id from merge_func
            # if it exists
            if not "task_id" in arg_names:
                task_id = kwargs.pop("task_id", None)
            else:
                task_id = kwargs["task_id"]

            dao = DAO(os.environ["RP_METADATASTORE_URL"])
            if task_id is None:
                # Only create task_id if merge_func didn't create it
                task_id = dao.add_task(
                    exec_id=exec_id,
                    tasktype_uuid=self.tasktype_uuid,
                    rank=rank,
                    start_timestamp=datetime.now(timezone.utc)
                )
                if "task_id" in arg_names:
                    kwargs["task_id"] = task_id


















            # Register task with registry for interrupt handling
            from ..runtime import _task_registry
            import os as _os
            pid = _os.getpid()
            _task_registry.register_task(task_id, pid)




















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
                    f"  \N{BULLET} Positional: {escape(str(args))}\n"
                    f"  \N{BULLET} Keyword   : {escape(str(kwargs))}" +
                    (f"\n[bold green]{docstring}[/]" if docstring else ""),
                    border_color="#FFFFE0"
                )
            )

            task_failed = False
            try:
                with TaskType._capture_and_stream_trace(task_id):
                    try:
                        result = func(*args, **kwargs)
                    except Exception:
                        traceback.print_exc()
                        sys.stderr.flush()
                        raise
                self.log.info(
                    f"\n\N{WHITE HEAVY CHECK MARK} Completed Task [#D2691E]`{self.name}[{task_id}]{f'[{rank}]' if rank is not None else ''}`[/]:\n"
                    f"Results :\n" +
                    f"  \N{BULLET} {escape(str(result))} {f'({type(result).__name__})' if result is not None else ''}\n"
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
                dao.dispose()
                # Unregister task from registry
                _task_registry.unregister_task(task_id)

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


    @staticmethod
    def _reconstruct_task_by_name(func_name: str, func_module: str) -> "TaskType":
        """Reconstruct a TaskType by importing it from its module."""
        import importlib
        
        module = importlib.import_module(func_module)
        task = getattr(module, func_name)
        
        # Should already be a TaskType due to @task decorator
        if isinstance(task, TaskType):
            return task
        
        # This shouldn't happen, but handle it gracefully
        raise ValueError(
            f"Expected {func_name} in {func_module} to be a TaskType, "
            f"but got {type(task).__name__}"
        )

    def __reduce__(self):
        """Enable pickling by referencing the module-level TaskType."""
        return (
            TaskType._reconstruct_task_by_name,
            (self.func.__name__, self.func.__module__)
        )


class DistributionNotSupportedError(Exception):
    """Raised when attempting to start a distributed sub-DAG
    (parallel_task) off several direct predecessors.
    A parallel-task takes a the payload from a single predecessor,
    which pauyload is expected to be an iterator."""
    pass


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

        self._log = logger

        for elmt in self.elements:
            elmt._task_group = self

        # add custom-rich handler
        # (since package-logger doesn't hold one when we're here)
        self._log.addHandler(tasktype_taskgroup_rich_handler)
        self._log.setLevel(logging.INFO)
        self._log.info(
            f"[red on white]" +
            self.name +
            str([f"[green]{elemt_name}[/green]"
                 for elemt_name in self._get_elements_names()]) +
            "[/red on white]"
        )
        self._log.removeHandler(self._log.handlers[-1])
        self._log.setLevel(logging.NOTSET)


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
            if other.is_parallel:
                raise DistributionNotSupportedError(
                    "parallel-tasks can only have 1 parent. " +
                    f"You tried to chain taskgroup '{self.name}' " +
                    f"before parallel-task '{other.name}' here.")
            if other.merge_func:
                # Note: TODO, implement support for that someday (or not)
                raise MergeNotSupportedError(
                    "merging-tasks can only have 1 parent. " +
                    f"You tried to chain taskgroup '{self.name}' " +
                    f"before merging-task '{other.name}' here.")

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

    params: Dict[str, "DagParam"] = Field(default_factory=dict)


    def __init__(self, *args, **kwargs):
        task_anchor = kwargs.pop('task_anchor')
        roots = DAG._find_root_tasks(task_anchor)
        super().__init__(*args, roots=roots, **kwargs)


    def init(self):
        """Shall be called programmatically 
        by the DAG execution routine.
        """
        if not in_notebook():
            # Get the calling frame (3 levels up)
            frame = inspect.currentframe()
            caller_frame = frame.f_back.f_back.f_back
            if (
                os.path.basename(caller_frame.f_code.co_filename).split(".")[-2]
                    != "retraining_pipeline"
            ):
                pipeline_name = os.path.basename(
                    caller_frame.f_code.co_filename).split(".")[-2]
            else:
                pipeline_name = os.path.basename(
                    os.path.dirname(caller_frame.f_code.co_filename))
        else:
            # note that notebooks cells do get frame filenames
            # like /tmp/ipykernel_5124/2977828399.py
            # so we don't rely on that to get the pipeline-name
            # from python module name
            pipeline_name = None
            # when a launcher script is executed
            # as __main__ from a notebook cell,
            # __main__.__file__ is the script path.
            # => that filename IS the pipeline name.
            main_mod = sys.modules.get("__main__")
            main_file = getattr(main_mod, "__file__", None)
            if (main_file and
                    not re.search(r'ipykernel_\d+/\d+\.py$', main_file)):
                pipeline_name = os.path.basename(main_file).rsplit(".", 1)[0]
            # otherwise (plain notebook cell),
            # we scan for the module that owns this
            # DAG instance, (skipping all dunder-named
            # execution namespaces)
            if pipeline_name is None:
                for mod_name, mod in list(sys.modules.items()):
                    if (mod is None or
                            (
                                mod_name.startswith("__") and
                                mod_name.endswith("__")
                            )
                    ):
                        continue
                    try:
                        for attr in vars(mod).values():
                            if attr is self:
                                pipeline_name = mod_name.split(".")[-1]
                                break
                    except Exception:
                        pass
                    if pipeline_name:
                        break

            if pipeline_name is None:
                pipeline_name = "retraining_pipeline"

            if pipeline_name == "retraining_pipeline":
                pipeline_name = os.path.basename(os.getcwd())

        username = getpass.getuser()

        dao = DAO(os.environ["RP_METADATASTORE_URL"])
        exec_id = dao.add_execution(
            name=pipeline_name,
            docstring=self.docstring,
            params={
                name: param.to_serializable_dict()
                for name, param in self.params.items()
            },
            username=username,
            ui_css=self.ui_css.__dict__ if self.ui_css else None,
            start_timestamp=datetime.now(timezone.utc)
        )
        tasktypes_list, taskgroups_list = self.to_elements_lists()
        for i, tasktype in enumerate(tasktypes_list):
            dao.add_tasktype(exec_id=exec_id, order=i,
                             **tasktype)
        for i, taskgroup in enumerate(taskgroups_list):
            dao.add_taskgroup(exec_id=exec_id, order=i,
                             **taskgroup)
        dao.dispose()

        # Update context
        _dag_execution_context_var.get().update(
            pipeline_name=pipeline_name, exec_id=exec_id,
            username=username,
            params_definitions=self.params
        )


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
        """Find all root tasks in the DAG starting from the given task,

        often the DAG tail."""

        # Handle case where DAG ends with a TaskGroup
        stack = list(task.elements) if isinstance(task, TaskGroup) else [task]

        all_tasks = set()
        while stack:
            current = stack.pop()
            if current not in all_tasks:
                all_tasks.add(current)
                stack.extend(current.parents)
        return [t for t in all_tasks if not t.parents]


    def help(self) -> str:
        """Generate help text for DAG parameters."""
        lines = []

        if self.docstring:
            lines.append(textwrap.dedent(self.docstring))
            lines.append("")

        if not self.params:
            lines.append("No parameters defined for this DAG.")
            return "\n".join(lines)

        lines.append("DAG Parameters definitions:")
        for name, param in self.params.items():
            default = f" [default: {param.default}]" if param.default is not None else ""
            lines.append(f"  - {name}: {param.description}{default}")
        return "\n".join(lines)


    @staticmethod
    def mark_complete(exec_id: int):
        """Mark the execution as complete.
        
        Shall be called programmatically by the DAG execution routine
        when the entire DAG execution finishes.
        """
        dao = DAO(os.environ["RP_METADATASTORE_URL"])
        context = _dag_execution_context_var.get()
        dao.update_execution(
            id=exec_id,
            end_timestamp=datetime.now(timezone.utc),
            context_dump=context.to_serializable_dict()  # Only serialize for DB
        )
        dao.dispose()


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


def _get_dag_params(dag_func: Callable):
    """Executes the function so params get instanciated."""
    # Get source and parse it
    source = inspect.getsource(dag_func)
    tree = ast.parse(source)

    # Extract just the function body lines
    func_node = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_node = node
            break

    if not func_node:
        raise ValueError("Could not find function definition")

    # Execute statements line by line to build up locals
    func_globals = dag_func.__globals__.copy()
    func_locals = {}

    for stmt in func_node.body:
        # Skip return statements
        if isinstance(stmt, ast.Return):
            continue
        # Execute each statement to build up locals
        try:
            stmt_module = ast.Module(body=[stmt], type_ignores=[])
            code = compile(stmt_module, '<string>', 'exec')
            exec(code, func_globals, func_locals)
        except:
            # If a statement fails, continue anyway
            pass

    params_dict = {}
    # Combine globals and locals
    eval_context = {**func_globals, **func_locals}
    # Now parse DagParam assignments
    for stmt in func_node.body:
        if isinstance(stmt, ast.Assign):
            # Check if this is a DagParam assignment
            if isinstance(stmt.value, ast.Call):
                # Check if it's calling DagParam
                call_name = None
                if isinstance(stmt.value.func, ast.Name):
                    call_name = stmt.value.func.id

                if call_name == 'DagParam':
                    # Get the variable name being assigned
                    for target in stmt.targets:
                        if isinstance(target, ast.Name):
                            param_name = target.id
                            # Extract the arguments to DagParam()
                            param_kwargs = {}
                            for keyword in stmt.value.keywords:
                                arg_name = keyword.arg
                                try:
                                    # First try literal_eval
                                    value = ast.literal_eval(keyword.value)
                                    param_kwargs[arg_name] = value
                                except:
                                    # If that fails, eval with combined context
                                    try:
                                        expr = ast.Expression(body=keyword.value)
                                        code = compile(expr, '<string>', 'eval')
                                        value = eval(code, eval_context)
                                        param_kwargs[arg_name] = value
                                    except:
                                        # If all else fails, skip
                                        continue

                            params_dict[param_name] = DagParam(**param_kwargs)

    return params_dict


def dag(func=None, *, ui_css=None):
    """Decorator factory for a DAG."""
    def decorator(f):
        params_dict = _get_dag_params(f)

        task_anchor = f()
        pipeline = DAG(
            task_anchor=task_anchor,
            docstring=f.__doc__,
            ui_css=ui_css,
            params=params_dict
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
        # If single entry, try to delegate to the wrapped value
        if len(self._data) == 1:
            # First check if key is in _data (task name lookup)
            try:
                if key in self._data:
                    return self._data[key]
            except TypeError:
                # key is unhashable, so it can't be a task name
                #  => Delegate to the wrapped value
                pass

            # If not a task name,
            #  => delegate to wrapped value
            value = list(self._data.values())[0]
            return value[key]

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
        # Prevent recursion during pickling/unpickling
        if name == '_data':
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '_data'")
        
        # Use object.__getattribute__ to avoid recursion
        try:
            _data = object.__getattribute__(self, '_data')
        except AttributeError:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'")
        
        if len(_data) == 1:
            value = list(_data.values())[0]
            return getattr(value, name)
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __str__(self):
        return f"TaskPayload({self._data})"

    def __repr__(self):
        return self.__str__()

    # METHODS FOR PICKLING
    def __getstate__(self):
        """Return state for pickling."""
        return {'_data': self._data}
    
    def __setstate__(self, state):
        """Restore state from pickling."""
        self._data = state['_data']
    
    def __reduce__(self):
        """Enable pickling."""
        return (TaskPayload, (self._data,))

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
    @classmethod
    def must_be_hex_color(
        cls, v: Optional[str],
        info: ValidationInfo
    ) -> Optional[str]:
        if (
            v is not None and
            not re.match(r'^#([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$', v)
        ):
            raise ValueError(
                f"{cls.__name__}.{info.field_name} got invalid value {v!r}: "
                "must be a hex color code starting with #."
            )
        return v


    def to_dict(self) -> dict:
        # Filter out None values before serializing
        data = {k: v for k, v in self.dict().items() if v is not None}
        return data


class DagParam(BaseModel):
    """Represents a parameter for DAG execution."""
    description: str
    default: Any = None

    def to_serializable_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dict representation of this DagParam."""
        return {"description": self.description,
                "default": DagParam._serialize(self.default)}

    @staticmethod
    def _serialize(obj: Any) -> Any:
        """
        Convert obj into JSON-serializable primitives.
        """
        if obj is None:
            return None
        if isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, BaseModel):
            # convert model to plain dict first,
            # then serialize recursively
            return DagParam._serialize(obj.dict())
        if isinstance(obj, dict):
            return {k: DagParam._serialize(v)
                    for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [DagParam._serialize(v) for v in obj]
        # Safe fallback
        return str(obj)


# only to be called in the runtime module.
_dag_execution_context_var = \
    contextvars.ContextVar("dag_execution_context", default=None)

class DagExecutionContext:
    """Container for DAG execution parameters accessible within tasks."""
    def __init__(self, params: Dict[str, Any]):
        self._params = params
        self._updates = {}  # Track updates made in this context

    def __getattr__(self, name: str) -> Any:
        if name.startswith('_'):
            return object.__getattribute__(self, name)
        return self._params.get(name)

    @staticmethod
    def _deep_update(target: dict, updates: dict):
        for k, v in updates.items():
            if k in target and isinstance(target[k], dict) and isinstance(v, dict):
                DagExecutionContext._deep_update(target[k], v)  # recurse
            else:
                target[k] = v

    def update(self, **kwargs):
        """Update context parameters."""
        DagExecutionContext._deep_update(self._params, kwargs)
        DagExecutionContext._deep_update(self._updates, kwargs)
    
    def get_updates(self) -> Dict[str, Any]:
        """Get all updates made in this context."""
        return self._updates.copy()
    
    def merge_updates(self, updates: Dict[str, Any]):
        """Merge updates from child context (child wins on conflicts)."""
        DagExecutionContext._deep_update(self._params, updates)
        DagExecutionContext._deep_update(self._updates, updates)
    
    def copy(self):
        """Create a deep copy of this context for child tasks runs."""
        new_ctx = DagExecutionContext(copy.deepcopy(self._params))
        return new_ctx
    
    def to_serializable_dict(self) -> Dict[str, Any]:
        """Convert to serializable format ONLY for database storage."""
        return {k: DagParam._serialize(v)
                for k, v in self._params.items()}

class _ContextProxy:
    """Proxy that automatically calls .get() on the ContextVar"""
    def __getattr__(self, name: str):
        ctx = _dag_execution_context_var.get()
        if ctx is None:
            raise RuntimeError(
                "DAG execution context not set. " +
                "Are you calling this outside of a DAG execution?")
        return getattr(ctx, name)

    def __setattr__(self, name: str, value: Any):
        ctx = _dag_execution_context_var.get()
        if ctx is None:
            raise RuntimeError(
                "DAG execution context not set. " +
                "Are you calling this outside of a DAG execution?")
        ctx._params[name] = value
        ctx._updates[name] = value  # Track the updates

ctx = _ContextProxy()


# task-traces
class StreamToDb(io.TextIOBase):
    def __init__(self, original_stream, task_id, is_err):
        super().__init__()
        self.original_stream = original_stream
        self.task_id = task_id
        self.is_err = is_err
        self.trace_buffer = get_trace_buffer()
        self.line_buffer = ""

    def write(self, data):
        if not data:
            return

        ## DEBUG
        # self.original_stream.write("StreamToDb.write\n")
        # self.original_stream.write("'")
        # self.original_stream.write(data)
        # self.original_stream.write("'\n")

        self.original_stream.write(data)
        self.original_stream.flush()

        self.line_buffer += data

        return len(data)

    def flush(self):
        if self.line_buffer:
            # Process line_buffer to DB before clearing
            pass
        self.line_buffer = ""
        self.original_stream.flush()

    def isatty(self):
        return self.original_stream.isatty()

    def close(self):
        """
        Note: Rich (and many libraries) call close()
              on streams during context managers,
              but we need the original stream
              to stay alive.
        """
        self.flush()
        pass  # Don't close original_stream
              # let OS manage it on exit

    def fileno(self):
        return self.original_stream.fileno()

    def __getattr__(self, name):
        return getattr(self.original_stream, name)

    def __del__(self):
        """Close wrapped stream on GC"""
        self.flush()

