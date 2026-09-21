"""
Permanent background event loop for synchronous SDK methods.

This module provides a single, long-lived asyncio event loop running in a
dedicated daemon thread. All asynchronous operations are scheduled on this
loop via run_coroutine_threadsafe and block the calling thread until
completion.

WHY THIS DESIGN:
- Many async database drivers (e.g., asyncpg) perform deferred cleanup of SSL
  transports after a coroutine has finished. If the event loop is closed
  immediately (as asyncio.run() does), these callbacks will fire on a closed
  loop, causing "Event loop is closed" RuntimeErrors and SSL transport
  failures. By keeping a single permanent loop alive for the entire process
  lifetime, we allow all pending cleanup callbacks to complete safely,
  eliminating those errors.
- The loop runs in a background thread, isolated from any pre-existing event
  loop (e.g., in Jupyter notebooks) and from the main thread. This isolation
  ensures that the SDK never interferes with other async code, and other
  async code never interferes with the SDK.

This design handles all execution environments without special-case logic:
- Scripts and command-line tools: The loop starts on first use and persists
  until process exit.
- Jupyter/IPython notebooks: The background thread runs independently of the
  kernel's own event loop, so no conflicts arise.
- Interactive Python sessions: Same as scripts; the loop remains available
  for the duration of the session.
- Environments with an existing running event loop: The SDK loop is
  independent; calls from any thread are safe and do not affect or depend
  on any pre-existing loop.

The loop is daemonized so it does not prevent process termination. For
long-lived applications, an atexit handler provides a best-effort graceful
shutdown.
"""

import asyncio
import atexit
import logging
import threading
from collections.abc import Coroutine
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator

from ..config import Config
from ..db.dao import AsyncDAO
from .stores import ExecutionParams, TaskExitContext, TaskExitPayload

logger = logging.getLogger(__name__)

# /// event loop deamon ////////////////////////////////////////////////////////////////////////////


# Module-level loop and thread (lazily initialised)
_loop: asyncio.AbstractEventLoop | None = None
_loop_thread: threading.Thread | None = None
_loop_lock = threading.Lock()


def _start_loop() -> None:
    """Start the permanent event loop in a daemon thread."""
    global _loop, _loop_thread
    with _loop_lock:
        if _loop is not None:
            return
        _loop = asyncio.new_event_loop()
        _loop_thread = threading.Thread(
            target=_run_loop_forever, args=(_loop,), daemon=True, name="AsyncSDKEventLoop"
        )
        _loop_thread.start()


def _run_loop_forever(loop: asyncio.AbstractEventLoop) -> None:
    asyncio.set_event_loop(loop)
    loop.run_forever()


def _run_async(coro: Coroutine) -> Any:
    """Run a coroutine on the permanent background event loop, blocking until done."""
    _start_loop()
    assert _loop is not None  # type narrowing for mypy

    future = asyncio.run_coroutine_threadsafe(coro, _loop)

    return future.result()


def shutdown_async_sdk():
    global _loop, _loop_thread
    if _loop is not None and _loop.is_running():
        _loop.call_soon_threadsafe(_loop.stop)
    if _loop_thread is not None and _loop_thread.is_alive():
        _loop_thread.join(timeout=2.0)  # wait briefly


atexit.register(shutdown_async_sdk)


# //////////////////////////////////////////////////////////////////////////////////////////////////


# /// core SDK classes /////////////////////////////////////////////////////////////////////////////


class Execution(BaseModel):
    """SDK representation of a pipeline execution.

    Provides access to execution metadata and context attributes.

    Attributes
    ----------
        id: Unique execution identifier
        name: Pipeline name
        start_timestamp: When execution started
        end_timestamp: When execution ended (None if still running)
        success: Whether execution completed successfully
        metadata_root: Absolute metadata root recorded at execution time
        artifacts_store_root: Absolute artifacts store root recorded at execution time
    """

    id: int = Field(..., description="Unique execution identifier")
    name: str = Field(..., description="Pipeline name")
    docstring: str | None = Field(None, description="Pipeline description")
    metadata_root: str = Field(
        ...,
        description=(
            "Absolute path to {Config.get_assets_cache_root()}/metadata/ as it was on the machine "
            "and at the time this execution ran. Used by SDK read methods to resolve "
            "disk artifacts independently of the current ``Config.get_assets_cache_root()`` value."
        ),
    )
    artifacts_store_root: str = Field(
        ...,
        description=(
            "Absolute path to {Config.get_artifacts_store_root()} as it was on the machine "
            "and at the time this execution ran. Used by SDK read methods to resolve "
            "disk artifacts independently of the current ``Config.get_artifacts_store_root()`` "
            "value."
        ),
    )
    username: str = Field(..., description="Execution user")
    start_timestamp: datetime = Field(..., description="Execution start time (UTC)")
    end_timestamp: datetime | None = Field(
        None, description="Execution end time (UTC), None if still running (not live-synched)"
    )
    success: bool = Field(..., description="Whether execution completed successfully")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        id: int,
        name: str,
        metadata_root: str,
        artifacts_store_root: str,
        username: str,
        start_timestamp: datetime,
        success: bool,
        docstring: str | None = None,
        end_timestamp: datetime | None = None,
        **data: Any,
    ) -> None:
        super().__init__(
            id=id,
            name=name,
            metadata_root=metadata_root,
            artifacts_store_root=artifacts_store_root,
            username=username,
            start_timestamp=start_timestamp,
            success=success,
            docstring=docstring,
            end_timestamp=end_timestamp,
            **data,
        )

    @classmethod
    def get_latest(cls, name: str, success_only: bool = False) -> "Execution":
        """Fetch an Execution by its name.

        Parameters
        ----------
        name : str
            name of the execution pipeline to iterate over
        success_only : bool
            if True, only iterates over successful executions

        Returns
        -------
        Execution

        Raises
        ------
        KeyError
            If no execution with the given name exists.
        """
        from . import ExecutionsIterator  # avoid circular import

        latest_execution = ExecutionsIterator(name, success_only, page_size=1).previous()

        if latest_execution is not None:
            return latest_execution
        else:
            raise KeyError(f"No execution found with name={name}")

    @classmethod
    def get_by_id(cls, id: int) -> "Execution":
        """Fetch an Execution by its id.

        Parameters
        ----------
        id : int
            Execution id.

        Returns
        -------
        Execution

        Raises
        ------
        KeyError
            If no execution with the given id exists.
        """

        async def _fetch():
            dao = AsyncDAO(db_url=Config.get_metadatastore_async_url())
            try:
                return await dao.get_execution_ext(id)
            finally:
                await dao.engine.dispose()

        row = _run_async(_fetch())
        if row is None:
            raise KeyError(f"No execution found with id={id}")
        return cls(
            id=row.id,
            name=row.name,
            docstring=row.docstring,
            metadata_root=row.metadata_root,
            artifacts_store_root=row.artifacts_store_root,
            username=row.username,
            start_timestamp=row.start_timestamp,
            end_timestamp=row.end_timestamp,
            success=row.success,
        )

    def completed(self) -> bool:
        """Check if execution has completed.

        Returns
        -------
        bool
            True if execution has finished (successfully or not).
        """
        return self.end_timestamp is not None

    def get_tasks_with_name(self, task_type_name: str) -> list["Task"]:
        """Return tasks by name.

        Returns
        -------
        list[Task]
            Naive unordered list of task instances.
        """

        async def _get_tasks_async():
            dao = AsyncDAO(db_url=Config.get_metadatastore_async_url())
            tasks_orm = await dao.get_execution_tasks_with_name(
                execution_id=self.id, task_type_name=task_type_name
            )
            if not tasks_orm:
                return []
            return [
                Task(
                    id=t.id,
                    name=task_type_name,
                    rank=t.rank,
                    start_timestamp=t.start_timestamp,
                    end_timestamp=t.end_timestamp,
                    success=not t.failed if t.failed is not None else True,
                    # private attrs
                    exec_id=self.id,
                    metadata_root=self.metadata_root,
                )
                for t in tasks_orm
            ]

        return _run_async(_get_tasks_async())

    def get_task_by_id(self, id: int) -> "Task":
        """Fetch a Task by its id.

        Parameters
        ----------
        id : int
            Task id.

        Returns
        -------
        Task

        Raises
        ------
        KeyError
            If no task with the given id exists.

        Examples
        --------
        >>> task = Execution.get_by_id(id=my_exec_id_int).get_task_by_id(id=42)
        >>> ctx  = task.get_exit_context()
        >>> print(ctx["added_entry"])
        """

        async def _fetch():
            dao = AsyncDAO(db_url=Config.get_metadatastore_async_url())
            return await dao.get_task_ext(id)

        row = _run_async(_fetch())
        if row is None:
            raise KeyError(f"No task found with id={id}")
        return Task(
            id=row.id,
            name=row.name,
            rank=row.rank,
            start_timestamp=row.start_timestamp,
            end_timestamp=row.end_timestamp,
            success=not row.failed if row.failed is not None else True,
            # private attrs
            exec_id=self.id,
            metadata_root=self.metadata_root,
        )

    def get_taskgroups_with_name(self, taskgroup_name: str) -> Optional["TaskGroupRanks"]:
        """Return ranks of a taskgroup by its name.

        Parameters
        ----------
        taskgroup_name : str
            Name of the taskgroup.

        Returns
        -------
        TaskGroupRanks
            Ranks of matching taskgroup, if exists.

        Examples
        --------
        >>> tg = Execution.get_by_id(id=my_exec_id_int).get_taskgroup_for_rank("my_taskgroup", [0])
        >>> payload = tg.get_exit_payload()
        """

        async def _fetch():
            dao = AsyncDAO(db_url=Config.get_metadatastore_async_url())
            try:
                return await dao.get_execution_taskgroup_ranks(
                    execution_id=self.id,
                    taskgroup_name=taskgroup_name,
                )
            finally:
                await dao.engine.dispose()

        tg = None
        try:
            tg, ranks = _run_async(_fetch())
        except TypeError as err:
            logger.warning(err.__str__())

        if not tg:
            logger.warning(
                f"There's no `{taskgroup_name}` taskgroup in execution `{self.name}` [id {self.id}]"
            )
            return None

        return TaskGroupRanks(
            uuid=str(tg.uuid),
            name=tg.name,
            docstring=tg.docstring,
            elements=tg.elements,
            ranks=ranks if len(ranks) > 0 else [],
            # private attrs
            exec_id=self.id,
            metadata_root=self.metadata_root,
        )

    def get_taskgroup_for_rank(
        self, taskgroup_name: str, rank: list[int]
    ) -> Optional["TaskGroupRank"]:
        """Return taskgroup by name for a given execution rank.

        Parameters
        ----------
        taskgroup_name : str
            Name of the taskgroup.
        rank : list[int]
            execution rank of the DAG sub-branch if applicable

        Returns
        -------
        TaskGroup
            Matching taskgroup, if exists.

        Examples
        --------
        >>> tg = Execution.get_by_id(id=356).get_taskgroup_for_rank(<a_valid_taskgroup_name>, [0])
        >>> payload = tg.get_exit_payload()
        """

        async def _fetch():
            dao = AsyncDAO(db_url=Config.get_metadatastore_async_url())
            try:
                return await dao.get_execution_taskgroup_ranks(
                    execution_id=self.id,
                    taskgroup_name=taskgroup_name,
                )
            finally:
                await dao.engine.dispose()

        tg = None
        try:
            tg, ranks = _run_async(_fetch())
        except TypeError as err:
            logger.warning(err.__str__())

        if not tg:
            logger.warning(
                f"There's no {taskgroup_name} taskgroup in the execution {self.name} [id {self.id}]"
            )
            return None

        # Normalize rank for comparison: treat empty list as None
        target_rank = rank if (rank and len(rank) > 0) else None

        # Check if the requested rank exists in the fetched ranks list.
        # `ranks` is a list of lists (e.g. [[1, 2], [3, 4]]) or [None].
        if target_rank is None:
            rank_exists = ranks == [None] or len(ranks) == 0
        else:
            rank_exists = target_rank in ranks

        if not rank_exists:
            logger.warning(
                f"There's no such rank as {rank or []} "
                f"for taskgroup {taskgroup_name} "
                f"in the execution {self.name} with id {self.id}"
            )
            return None

        return TaskGroupRank(
            taskgroup=TaskGroup(
                uuid=str(tg.uuid),
                name=tg.name,
                docstring=tg.docstring,
                elements=tg.elements,
                # private attrs
                exec_id=self.id,
                metadata_root=self.metadata_root,
            ),
            rank=rank if rank and len(rank) > 0 else None,
        )

    def get_params(self) -> "ExecutionParams":
        """Return a lazy view of the resolved DAG params for this execution.

        The DB is queried once to fetch the params JSON ;
        individual param values (defaults or overrides)
        are deserialized from disk only when first accessed by key,
        using the execution's metadata_root.

        Returns
        -------
        ExecutionParams
            Lazy mapping of param name => resolved value.

        Examples
        --------
        >>> exec_params = Execution.get_by_id(id=356).get_params()
        >>> for param_name in exec_params:
        ...     print(f"{param_name}, {exec_params[param_name]}")
        param1 param1_value
        param2 param2_value
        """

        async def _fetch():
            dao = AsyncDAO(db_url=Config.get_metadatastore_async_url())
            full_exec = await dao.get_execution(self.id)
            return full_exec.params

        params_json = _run_async(_fetch()) or {}
        return ExecutionParams(params_json, self.metadata_root)

    def get_attrs(self) -> TaskExitContext | None:
        """Return a lazy view of the execution-level context attrs at execution completion.

        Returns the exit context of the last non-parallel task (highest task_id
        among rank IS NULL tasks) of this execution.

        Equivalent to:

            >>> last_task = Execution.get_by_id(id=356).get_tasks_with_name("end")[0]
            >>> last_task.get_exit_context()

        Note:
        -----
        If the last element of a DAG is a non-merged sub-DAG (parallel branches
        left hanging / in limbo), you can use ``get_tasks_with_name`` and
        ``get_exit_context`` on those to access the non-merged final context states
        of those different branches.

        Parameters
        ----------
        None

        Returns
        -------
        TaskExitContext or None
            Lazy mapping of attr_name => value, identical to the last
            non-parallel task's exit context.
            Deserialization from disk occurs only when individual attrs are accessed.
            None if the execution did not complete successfully.

        Examples
        --------
        >>> attrs = Execution.get_by_id(id=356).get_attrs()
        >>> for name in attrs:
        ...     print(name, attrs[name])
        """

        async def _fetch():
            dao = AsyncDAO(db_url=Config.get_metadatastore_async_url())
            try:
                return await dao.get_execution_latest_context_attrs(self.id)
            finally:
                await dao.engine.dispose()

        rows = _run_async(_fetch())
        if rows is None:
            logger.warning(
                f"get_attrs: execution {self.id} did not complete successfully ; "
                "no execution-level context available."
            )
            return None
        return TaskExitContext(rows, self.metadata_root)

    def elements_iterator(self):
        """Get iterator over execution elements.

        Iterates through tasks, taskgroups, and sub-DAGs in topological order.

        Notes
        -----
        "next" on a sub-DAG returns the element following its closing merge task.
        """
        # TODO: Implement iteration over tasks, taskgroups, and sub-DAGs
        raise NotImplementedError()


class Task(BaseModel):
    """SDK representation of a pipeline task instance.

    Provides access to task metadata.

    Attributes
    ----------
        id : int
            Unique task identifier
        name : str
            TaskType name
        rank : list[int]
            execution rank of the DAG sub-branch (if applicable).
        start_timestamp : datetime
            When task started
        end_timestamp : datetime
            When task ended (None if still running)
        success : bool
            Whether task completed successfully
        exec_id : int
            Parent execution id.
        metadata_root : str
            Absolute metadata root from the parent execution
    """

    id: int = Field(..., description="Unique task identifier")
    name: str = Field(..., description="Task type name")
    rank: list[int] | None = Field(
        None, description="Execution rank of the task instance's DAG sub-branch (if applicable)"
    )
    start_timestamp: datetime = Field(..., description="Task start time (UTC)")
    end_timestamp: datetime | None = Field(
        None, description="Task end time (UTC), None if still running (not live-synched)"
    )
    success: bool | None = Field(None, description="Whether task completed successfully")

    _exec_id: int = PrivateAttr()
    # Absolute path to the metadata root from the parent execution.
    # Propagated from Execution.metadata_root ; used to resolve disk artifacts.
    _metadata_root: str = PrivateAttr()

    def __init__(
        self,
        id: int,
        name: str,
        start_timestamp: datetime,
        exec_id: int,
        metadata_root: str,
        rank: list[int] | None = None,
        end_timestamp: datetime | None = None,
        success: bool | None = None,
        **data: Any,
    ) -> None:
        super().__init__(
            id=id,
            name=name,
            start_timestamp=start_timestamp,
            rank=rank,
            end_timestamp=end_timestamp,
            success=success,
            **data,
        )
        self._exec_id = exec_id
        self._metadata_root = metadata_root

    def get_exit_context(self) -> TaskExitContext:
        """Return a lazy view of this task's exit context.

        The DB is queried once (O(1): ``SELECT ... WHERE task_id = ?``) to
        fetch the full attr index. Individual attribute values are deserialized
        from disk only when accessed by key, using the execution's own
        metadata_root stored in DB.

        Returns
        -------
        TaskExitContext
            Mapping of attr_name => value at this task's exit.

        Raises
        ------
        ValueError
            If metadata_root is None (task was not obtained via Execution.get_task_by_id
            or Execution.get_tasks_with_name).

        Examples
        --------
        >>> ctx = Execution.get_by_id(id=356).get_task_by_id(id=42).get_exit_context()
        >>> print(ctx["added_entry"])
        >>> for attr_name in ctx:
        ...     print(attr_name, ctx[attr_name])
        """

        async def _fetch():
            dao = AsyncDAO(db_url=Config.get_metadatastore_async_url())
            return await dao.get_task_context_attrs(self.id)

        rows = _run_async(_fetch())
        return TaskExitContext(rows or [], self._metadata_root)

    def get_exit_payload(self) -> TaskExitPayload:
        """Return a lazy view of this task's exit payload.

        The DB is queried once (O(1): ``SELECT ... WHERE task_id = ?``) to fetch
        the payload row. The value is deserialized from disk only when accessed,
        using the execution's own metadata_root stored in DB.

        Returns
        -------
        TaskExitPayload
            Lazy wrapper around the serialized task return value.
            Behaves identically to ``core.TaskPayload``.

        Examples
        --------
        >>> payload = Execution.get_by_id(id=356).get_tasks_with_name("merge")[0] \
        ...                 .get_exit_payload()
        >>> print(payload.value)
        """

        async def _fetch():
            dao = AsyncDAO(db_url=Config.get_metadatastore_async_url())
            try:
                return await dao.get_task_exit_payload(self.id)
            finally:
                await dao.engine.dispose()

        row = _run_async(_fetch())
        return TaskExitPayload({self.name: row}, self._metadata_root)


class TaskGroup(BaseModel):
    """SDK representation of a taskgroup.

    Mirrors ``dao.model.TaskGroup``: holds the 1st-degree element UUIDs
    (tasks or nested taskgroups) declared in the taskgroup definition.
    De-nesting across nested taskgroups occurs in ``get_exit_context`` and
    ``get_exit_payload``, not here.

    Attributes
    ----------
    uuid : str
        Taskgroup UUID (string with dashes, as stored in DB).
    name : str
        Taskgroup name.
    docstring : str or None
        Taskgroup docstring.
    ui_css : dict or None
        UI styling metadata.
    elements : list[str]
        1st-degree element UUIDs (TaskType or nested TaskGroup UUIDs).
    exec_id : int
        Parent execution id.
    metadata_root : str
        Absolute metadata root propagated from the parent Execution.
    """

    uuid: str = Field(..., description="Taskgroup UUID")
    name: str = Field(..., description="Taskgroup name")
    docstring: str | None = Field(None, description="Taskgroup docstring")
    ui_css: dict | None = Field(None, description="UI styling metadata")
    elements: list[str] = Field(
        ..., description="1st-degree element UUIDs (tasks or nested taskgroups)"
    )

    _exec_id: int = PrivateAttr()
    # Absolute path to the metadata root from the parent execution.
    # Propagated from Execution.metadata_root ; used to resolve disk artifacts.
    _metadata_root: str = PrivateAttr()

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        uuid: str,
        name: str,
        elements: list[str],
        exec_id: int,
        metadata_root: str,
        docstring: str | None = None,
        ui_css: dict | None = None,
        **data: Any,
    ) -> None:
        super().__init__(
            uuid=uuid,
            name=name,
            elements=elements,
            docstring=docstring,
            ui_css=ui_css,
            **data,
        )
        self._exec_id = exec_id
        self._metadata_root = metadata_root

    def __eq__(self, other: object) -> bool:
        """Explicit equality based on public model fields, ignoring private attrs."""
        if not isinstance(other, TaskGroup):
            return NotImplemented
        return self.model_dump() == other.model_dump()


class TaskGroupRanks(BaseModel):
    """SDK representation of an taskgroup and its execution ranks.

    Behaves as an iterable container yielding `TaskGroupRank` instances.

    Note
    ----
    Say we have a taskgroup inside a 2-levels deep sub-DAG
    (first sub DAG has for instance 2 exec-time branches and second has 3).
    Then we have multiple exec-time "instances" of that taskgroup,
    1 per execution-rank, (i.e. in our example,
    ranks=[[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]]).

    Attributes
    ----------
    taskgroup : TaskGroup
        The taskgroup definition and metadata.
    ranks : list[list[int]]
        list of ranks for that taskgroup in its execution.
    """

    taskgroup: TaskGroup
    ranks: list[list[int]] = Field(
        ..., description="list of ranks for that taskgroup in its execution"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("ranks", mode="before")
    @classmethod
    def parse_ranks(cls, v):
        """Convert db-stored ranks (strings or lists) into list[list[int]]."""
        if isinstance(v, list):
            parsed_ranks = []
            for item in v:
                if isinstance(item, str):
                    # If the DB returns a string like "[1, 2]"
                    clean_str = item.strip("[]")
                    if clean_str:
                        parsed_ranks.append(list(map(int, clean_str.split(","))))
                    else:
                        parsed_ranks.append([])
                elif isinstance(item, list):
                    # If it's already a list of ints
                    parsed_ranks.append(list(map(int, item)))
            return parsed_ranks
        return v

    def __init__(
        self,
        uuid: str,
        name: str,
        elements: list[str],
        exec_id: int,
        metadata_root: str,
        ranks: list[list[int]] | None = None,
        docstring: str | None = None,
        ui_css: dict | None = None,
        **data: Any,
    ) -> None:
        taskgroup_instance = TaskGroup(
            uuid=uuid,
            name=name,
            elements=elements,
            exec_id=exec_id,
            metadata_root=metadata_root,
            docstring=docstring,
            ui_css=ui_css,
            **data,
        )
        super().__init__(taskgroup=taskgroup_instance, ranks=ranks if ranks is not None else [])

    def __iter__(self):
        """Allow iteration over ranks, yielding TaskGroupRank objects."""
        if len(self.ranks) == 0:
            yield TaskGroupRank(taskgroup=self.taskgroup, rank=None)

        for r in self.ranks:
            yield TaskGroupRank(taskgroup=self.taskgroup, rank=r)

    def __getitem__(self, index):
        """Allow indexing/slicing, returning TaskGroupRank objects."""
        if isinstance(index, slice):
            return [TaskGroupRank(taskgroup=self.taskgroup, rank=r) for r in self.ranks[index]]
        return TaskGroupRank(
            taskgroup=self.taskgroup,
            rank=self.ranks[index] if len(self.ranks) > 0 else None,
        )

    def __len__(self):
        """Return the number of ranks."""
        return len(self.ranks)

    def get_rank(self, rank: list[int]) -> Optional["TaskGroupRank"]:
        """Return the TaskGroupRank for a specific rank list, if it exists.

        Parameters
        ----------
        rank : list[int]
            The specific rank (e.g., [1, 2]) to look up.

        Returns
        -------
        TaskGroupRank | None
            The matching TaskGroupRank object, or None if not found.

        Examples
        --------
        >>> tgr = TaskGroupRanks(
        ...     exec_id=1,
        ...     metadata_root="/path",
        ...     uuid="...",
        ...     name="my_group",
        ...     elements=[],
        ...     ranks=[[0], [1]]
        ... )
        >>> taskgroup_rank = tgr.get_rank([0])
        >>> print(taskgroup_rank.rank)
        [0]
        >>> print(taskgroup_rank.taskgroup.name)
        my_group

        >>> missing_rank = tgr.get_rank([9, 9])
        >>> print(missing_rank)
        None
        """
        if len(self.ranks) == 0:
            return TaskGroupRank(taskgroup=self.taskgroup, rank=None)

        for r in self.ranks:
            if r == rank:
                return TaskGroupRank(taskgroup=self.taskgroup, rank=r)
        return None


class TaskGroupRank(BaseModel):
    """SDK representation of a single taskgroup and a single execution rank.

    Attributes
    ----------
    taskgroup : TaskGroup
        The taskgroup definition and metadata.
    rank : list[int]
        A single rank list of ints (e.g., [1, 2]) for this taskgroup.
        Execution rank of the DAG sub-branch (if applicable, None otherwise).
    """

    taskgroup: TaskGroup
    rank: list[int] | None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        taskgroup: TaskGroup,
        rank: list[int] | None = None,
        **data: Any,
    ) -> None:
        super().__init__(taskgroup=taskgroup, rank=rank, **data)

    @field_validator("rank", mode="before")
    @classmethod
    def parse_rank(cls, v):
        """Convert db-stored rank (string or list) into list[int]."""
        if isinstance(v, str):
            # If the DB returns a string like "[1, 2]"
            clean_str = v.strip("[]")
            if clean_str:
                return list(map(int, clean_str.split(",")))
            return []
        elif isinstance(v, list):
            return list(map(int, v))
        return v

    def __eq__(self, other: object) -> bool:
        """Explicit equality checking both taskgroup and rank."""
        if not isinstance(other, TaskGroupRank):
            return NotImplemented
        return self.taskgroup == other.taskgroup and self.rank == other.rank

    def get_exit_context(self) -> "TaskExitContext":
        """Return the merged lazy exit context across all (recursively de-nested) member tasks.

        De-nesting (BFS expansion of nested taskgroups) is performed here.
        For attrs present in multiple member tasks, last-writer-wins.

        Returns
        -------
        TaskExitContext
            Merged mapping of attr_name => value across all member tasks.

        Examples
        --------
        >>> tg = Execution.get_by_id(id=my_exec_id_int).get_taskgroup_for_rank("my_taskgroup", [0])
        >>> ctx = tg.get_exit_context()
        >>> print(ctx["some_attr"])
        """

        async def _fetch():
            dao = AsyncDAO(db_url=Config.get_metadatastore_async_url())
            try:
                tasks = await dao.get_taskgroup_nested_tasks(
                    execution_id=self.taskgroup._exec_id,
                    taskgroup_name=self.taskgroup.name,
                    rank=self.rank,
                )
                if not tasks:
                    return {}
                task_ids = [task_ext.id for task_ext in tasks]
                return await dao.get_task_context_attrs_bulk(task_ids)
            finally:
                await dao.engine.dispose()

        rows_by_task = _run_async(_fetch()) or {}
        # Flatten across all member tasks; last writer wins for shared attr names.
        merged: dict[str, Any] = {}
        for rows in rows_by_task.values():
            for row in rows:
                merged[row.attr_name] = row
        return TaskExitContext(list(merged.values()), self.taskgroup._metadata_root)

    def get_exit_payload(self) -> TaskExitPayload:
        """Return a lazy view of this taskgroup's exit payload.

        Fetches all member task payload rows in a single bulk query.
        The value is deserialized into a dict only when accessed,
        using the execution's own metadata_root.

        Returns
        -------
        TaskExitPayload
            Lazy wrapper around the serialized taskgroup return values.
            Behaves identically to ``core.TaskPayload``, mapping task_name =>
            deserialized return value for all member tasks.
            Tasks with no recorded payload (e.g. failed) map to None.

        Examples
        --------
        >>> tg = Execution.get_by_id(id=my_exec_id_int).get_taskgroup_for_rank("my_taskgroup", [0])
        >>> payload = tg.get_exit_payload()
        >>> print(payload["snake_head_A1"])
        """

        async def _fetch():
            dao = AsyncDAO(db_url=Config.get_metadatastore_async_url())
            try:
                tasks = await dao.get_taskgroup_nested_tasks(
                    execution_id=self.taskgroup._exec_id,
                    taskgroup_name=self.taskgroup.name,
                    rank=self.rank,
                )
                if not tasks:
                    return [], {}
                task_ids = [task_ext.id for task_ext in tasks]
                payload_rows = await dao.get_task_exit_payloads_bulk(task_ids)
                return tasks, payload_rows
            finally:
                await dao.engine.dispose()

        tasks, payload_rows = _run_async(_fetch())
        rows = {task_ext.name: payload_rows.get(task_ext.id) for task_ext in tasks}
        return TaskExitPayload(rows, self.taskgroup._metadata_root)


# //////////////////////////////////////////////////////////////////////////////////////////////////
