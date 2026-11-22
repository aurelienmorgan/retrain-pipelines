"""SDK representation of pipeline executions, tasks and taskgroups.

All async DB access in this module runs on the package-level permanent
event loop exposed as ``sdk.run_async`` (see ``sdk/__init__.py``).
"""

import logging
from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import ConfigDict, Field, PrivateAttr, field_validator

from ..config import Config
from ..db.dao import AsyncDAO
from ..db.model import TaskExt
from . import BaseModelExt, run_async
from .stores import ExecutionParams, TaskExitContext, TaskExitPayload

logger = logging.getLogger(__name__)


# /// core SDK classes /////////////////////////////////////////////////////////////////////////////


class RankNotFoundException(Exception):
    """Raised when no task exists at the given rank for this execution."""

    pass


class Execution(BaseModelExt):
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

        row = run_async(_fetch())
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

    def get_tasks_with_name(self, tasktype_name: str) -> list["Task"]:
        """Return tasks by name.

        Returns
        -------
        list[Task]
            Naive unordered list of task instances.
        """

        async def _get_tasks_async():
            dao = AsyncDAO(db_url=Config.get_metadatastore_async_url())
            tasks_orm = await dao.get_execution_tasks_with_name(
                execution_id=self.id, tasktype_name=tasktype_name
            )
            if not tasks_orm:
                return []
            return [
                Task(
                    id=t.id,
                    name=tasktype_name,
                    rank=t.rank,
                    start_timestamp=t.start_timestamp,
                    end_timestamp=t.end_timestamp,
                    success=not t.failed if t.failed is not None else True,
                    docstring=t.docstring,
                    is_parallel=t.is_parallel,
                    # private attrs
                    exec_id=self.id,
                    metadata_root=self.metadata_root,
                )
                for t in tasks_orm
            ]

        return run_async(_get_tasks_async())

    def get_tasks_with_uuid(self, tasktype_uuid: UUID | str) -> list["Task"]:
        """Return tasks by tasktype uuid.

        Parameters
        ----------
        tasktype_uuid : UUID or str
            UUID of the TaskType. A ``str`` is auto-cast to ``UUID``.

        Returns
        -------
        list[Task]
            Naive unordered list of task instances.
        """
        uuid_value = tasktype_uuid if isinstance(tasktype_uuid, UUID) else UUID(str(tasktype_uuid))

        async def _get_tasks_async():
            dao = AsyncDAO(db_url=Config.get_metadatastore_async_url())
            rows = await dao.get_execution_tasks_with_uuid(
                execution_id=self.id, tasktype_uuid=uuid_value
            )
            if not rows:
                return []
            return [
                Task(
                    id=t.id,
                    name=t.name,
                    rank=t.rank,
                    start_timestamp=t.start_timestamp,
                    end_timestamp=t.end_timestamp,
                    success=not t.failed if t.failed is not None else True,
                    docstring=t.docstring,
                    is_parallel=t.is_parallel,
                    # private attrs
                    exec_id=self.id,
                    metadata_root=self.metadata_root,
                )
                for t in rows
            ]

        return run_async(_get_tasks_async())

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

        row = run_async(_fetch())
        if row is None:
            raise KeyError(f"No task found with id={id}")
        return Task(
            id=row.id,
            name=row.name,
            rank=row.rank,
            start_timestamp=row.start_timestamp,
            end_timestamp=row.end_timestamp,
            success=not row.failed if row.failed is not None else True,
            docstring=row.docstring,
            is_parallel=row.is_parallel,
            # private attrs
            exec_id=self.id,
            metadata_root=self.metadata_root,
        )

    def get_task(
        self,
        tasktype_name: str | None = None,
        tasktype_uuid: UUID | str | None = None,
        rank: list[int] | None = None,
    ) -> "Task":
        """Fetch the task at the given rank for this execution.

        Provide either ``tasktype_name`` (positional) or ``tasktype_uuid`` (keyword).

        Parameters
        ----------
        tasktype_name : str
            Name of the TaskType (positional first argument).
        tasktype_uuid : UUID or str
            UUID of the TaskType (keyword argument).
        rank : list[int], optional
            Execution rank of the DAG sub-branch to filter on.
            None (default) targets the main DAG spine.

        Returns
        -------
        Task
            The task instance.

        Raises
        ------
        AttributeError
            If both or neither of ``tasktype_name`` and ``tasktype_uuid`` are provided.
        RankNotFoundException
            If no task exists at the given rank for this execution.

        Examples
        --------
        >>> execution = Execution.get_by_id(id=356)
        >>> task = execution.get_task("my_task")
        >>> task = execution.get_task("my_task", rank=[0, 1])
        >>> task = execution.get_task(tasktype_uuid="12345678-1234-1234-1234-123456789abc")
        >>> task = execution.get_task(tasktype_uuid=tt_uuid, rank=[0, 1])
        """
        if tasktype_name is not None and tasktype_uuid is not None:
            raise AttributeError("Provide either 'tasktype_name' or 'tasktype_uuid', not both.")
        if tasktype_name is None and tasktype_uuid is None:
            raise AttributeError("Provide either 'tasktype_name' or 'tasktype_uuid'.")

        if tasktype_name is not None:
            tasks = self.get_tasks_with_name(tasktype_name)
            tasktype_label = tasktype_name
        else:
            assert tasktype_uuid is not None  # type narrowing for mypy
            tasks = self.get_tasks_with_uuid(tasktype_uuid)
            tasktype_label = str(tasktype_uuid)

        # No tasks of this tasktype at all -> distinct from "rank not found".
        if not tasks:
            raise RankNotFoundException(
                f"No task '{tasktype_label}' found in execution '{self.name}' [id {self.id}]."
            )

        # Normalize rank: empty list and None both target the main DAG spine.
        target_rank = rank if (rank and len(rank) > 0) else None
        matching = [t for t in tasks if t.rank == target_rank]
        if not matching:
            # Available ranks among the tasks of this tasktype (None included).
            available = sorted({t.rank for t in tasks}, key=lambda r: (r is None, r))
            raise RankNotFoundException(
                f"No task '{tasktype_label}' found at rank {rank} "
                f"in execution '{self.name}' [id {self.id}]. "
                f"Available ranks: {available}"
            )
        return matching[0]

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
            tg, ranks = run_async(_fetch())
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

    def get_taskgroups_with_uuid(self, taskgroup_uuid: UUID | str) -> Optional["TaskGroupRanks"]:
        """Return ranks of a taskgroup by its tasktype uuid.

        Parameters
        ----------
        taskgroup_uuid : UUID or str
            UUID of the taskgroup. A ``str`` is auto-cast to ``UUID``.

        Returns
        -------
        TaskGroupRanks
            Ranks of matching taskgroup, if exists.
        """
        uuid_value = (
            taskgroup_uuid if isinstance(taskgroup_uuid, UUID) else UUID(str(taskgroup_uuid))
        )

        async def _fetch():
            dao = AsyncDAO(db_url=Config.get_metadatastore_async_url())
            try:
                return await dao.get_execution_taskgroup_ranks_with_uuid(
                    execution_id=self.id,
                    taskgroup_uuid=uuid_value,
                )
            finally:
                await dao.engine.dispose()

        tg = None
        try:
            tg, ranks = run_async(_fetch())
        except TypeError as err:
            logger.warning(err.__str__())

        if not tg:
            logger.warning(
                f"There's no `{taskgroup_uuid}` taskgroup in execution `{self.name}` [id {self.id}]"
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

    def get_taskgroup(
        self,
        taskgroup_name: str | None = None,
        taskgroup_uuid: UUID | str | None = None,
        rank: list[int] | None = None,
    ) -> "TaskGroupRank":
        """Fetch the taskgroup at the given rank for this execution.

        Provide either ``taskgroup_name`` (positional) or ``taskgroup_uuid`` (keyword).

        Parameters
        ----------
        taskgroup_name : str
            Name of the taskgroup (positional first argument).
        taskgroup_uuid : UUID or str
            UUID of the taskgroup (keyword argument).
        rank : list[int], optional
            Execution rank of the DAG sub-branch to filter on.
            None (default) targets the main DAG spine.

        Returns
        -------
        TaskGroupRank
            The taskgroup-rank instance.

        Raises
        ------
        AttributeError
            If both or neither of ``taskgroup_name`` and ``taskgroup_uuid`` are provided.
        RankNotFoundException
            If no taskgroup exists at the given rank for this execution.

        Examples
        --------
        >>> execution = Execution.get_by_id(id=356)
        >>> tg = execution.get_taskgroup("my_taskgroup")
        >>> tg = execution.get_taskgroup("my_taskgroup", rank=[0, 1])
        >>> tg = execution.get_taskgroup(taskgroup_uuid="12345678-1234-1234-1234-123456789abc")
        >>> tg = execution.get_taskgroup(taskgroup_uuid=tg_uuid, rank=[0, 1])
        """
        if taskgroup_name is not None and taskgroup_uuid is not None:
            raise AttributeError("Provide either 'taskgroup_name' or 'taskgroup_uuid', not both.")
        if taskgroup_name is None and taskgroup_uuid is None:
            raise AttributeError("Provide either 'taskgroup_name' or 'taskgroup_uuid'.")

        if taskgroup_name is not None:
            taskgroup_label = taskgroup_name
        else:
            uuid_value = (
                taskgroup_uuid if isinstance(taskgroup_uuid, UUID) else UUID(str(taskgroup_uuid))
            )
            taskgroup_label = str(taskgroup_uuid)

        async def _fetch():
            dao = AsyncDAO(db_url=Config.get_metadatastore_async_url())
            try:
                if taskgroup_name is not None:
                    return await dao.get_execution_taskgroup_ranks(
                        execution_id=self.id,
                        taskgroup_name=taskgroup_name,
                    )
                else:
                    return await dao.get_execution_taskgroup_ranks_with_uuid(
                        execution_id=self.id,
                        taskgroup_uuid=uuid_value,
                    )
            finally:
                await dao.engine.dispose()

        tg = None
        try:
            tg, ranks = run_async(_fetch())
        except TypeError as err:
            logger.warning(err.__str__())

        if not tg:
            raise RankNotFoundException(
                f"No taskgroup '{taskgroup_label}' found in execution '{self.name}' [id {self.id}]."
            )

        # Normalize rank for comparison: treat empty list as None
        target_rank = rank if (rank and len(rank) > 0) else None

        # Check if the requested rank exists in the fetched ranks list.
        # `ranks` is a list of lists (e.g. [[1, 2], [3, 4]]) or [None].
        if target_rank is None:
            rank_exists = ranks == [None] or len(ranks) == 0
        else:
            rank_exists = target_rank in ranks

        if not rank_exists:
            raise RankNotFoundException(
                f"No taskgroup '{taskgroup_label}' found at rank {rank} "
                f"in execution '{self.name}' [id {self.id}]. "
                f"Available ranks: {ranks}"
            )

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

        params_json = run_async(_fetch()) or {}
        return ExecutionParams(params_json, self.metadata_root)

    def get_attrs(self) -> TaskExitContext | None:
        """Return a lazy view of the execution-level context attrs at execution completion.

        Returns the exit context of the last non-parallel task (highest task_id
        among rank IS NULL tasks) of this execution.

        Equivalent to:

            >>> last_task = Execution.get_by_id(id=<a_valid_execution_id>).get_task("end")
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

        rows = run_async(_fetch())
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


class Task(BaseModelExt):
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
        docstring : str or None
            TaskType docstring.
        is_parallel : bool or None
            Whether the task is part of a parallel branch.
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
    docstring: str | None = Field(None, description="TaskType docstring")
    is_parallel: bool | None = Field(
        None, description="Whether the task is part of a parallel branch"
    )

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
        docstring: str | None = None,
        is_parallel: bool | None = None,
        **data: Any,
    ) -> None:
        super().__init__(
            id=id,
            name=name,
            start_timestamp=start_timestamp,
            rank=rank,
            end_timestamp=end_timestamp,
            success=success,
            docstring=docstring,
            is_parallel=is_parallel,
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

        rows = run_async(_fetch())
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

        row = run_async(_fetch())
        return TaskExitPayload({self.name: row}, self._metadata_root)


class TaskGroup(BaseModelExt):
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


class TaskGroupRanks(BaseModelExt):
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
    def _parse_ranks(cls, v):
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


class TaskGroupRank(BaseModelExt):
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
    def _parse_rank(cls, v):
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

        rows_by_task = run_async(_fetch()) or {}
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

        tasks, payload_rows = run_async(_fetch())
        rows = {task_ext.name: payload_rows.get(task_ext.id) for task_ext in tasks}
        return TaskExitPayload(rows, self.taskgroup._metadata_root)

    def get_element(self, element_uuid: UUID | str) -> "Task | TaskGroupRank":
        """Resolve an element of this taskgroup at the given rank.

        Since taskgroups can be nested, the element is either a TaskType
        (returns a :class:`Task`) or a nested TaskGroup (returns a
        :class:`TaskGroupRank`).

        Parameters
        ----------
        element_uuid : UUID or str
            UUID of the element (TaskType or nested TaskGroup) to resolve.
            A ``str`` is auto-cast to ``UUID``. Must be a 1st-degree element
            of this taskgroup (i.e. present in ``self.taskgroup.elements``).

        Returns
        -------
        Task or TaskGroupRank
            The resolved element instance.

        Raises
        ------
        ValueError
            If ``element_uuid`` is not a 1st-degree element of this taskgroup.
        RankNotFoundException
            If the resolved element has no instance at the given rank.

        Examples
        --------
        >>> tg = Execution.get_by_id(id=356).get_taskgroup("my_taskgroup")
        >>> elem = tg.get_element("12345678-1234-1234-1234-123456789abc")
        """
        uuid_value = element_uuid if isinstance(element_uuid, UUID) else UUID(str(element_uuid))
        # elements are stored as strings with dashes (same str(tg.uuid) form).
        if str(uuid_value) not in self.taskgroup.elements:
            raise ValueError(
                f"Element '{uuid_value}' is not a 1st-degree element of "
                f"taskgroup '{self.taskgroup.name}' [uuid {self.taskgroup.uuid}]."
            )

        async def _resolve():
            dao = AsyncDAO(db_url=Config.get_metadatastore_async_url())
            try:
                return await dao.get_taskgroup_element(
                    execution_id=self.taskgroup._exec_id,
                    taskgroup_name=self.taskgroup.name,
                    element_uuid=uuid_value,
                    rank=self.rank,
                )
            finally:
                await dao.engine.dispose()

        result = run_async(_resolve())
        if result is None:
            raise RankNotFoundException(
                f"No element '{uuid_value}' found at rank {self.rank} "
                f"in execution '{self.taskgroup._exec_id}'."
            )

        if isinstance(result, TaskExt):
            return Task(
                id=result.id,
                name=result.name,
                rank=result.rank,
                start_timestamp=result.start_timestamp,
                end_timestamp=result.end_timestamp,
                success=not result.failed if result.failed is not None else True,
                docstring=result.docstring,
                is_parallel=result.is_parallel,
                # private attrs
                exec_id=self.taskgroup._exec_id,
                metadata_root=self.taskgroup._metadata_root,
            )

        # result is a TaskGroup ORM object.
        return TaskGroupRank(
            taskgroup=TaskGroup(
                uuid=str(result.uuid),
                name=result.name,
                docstring=result.docstring,
                elements=result.elements,
                # private attrs
                exec_id=self.taskgroup._exec_id,
                metadata_root=self.taskgroup._metadata_root,
            ),
            rank=self.rank,
        )


# //////////////////////////////////////////////////////////////////////////////////////////////////
