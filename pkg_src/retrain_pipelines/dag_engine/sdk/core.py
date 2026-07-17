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
import hashlib
import os
import threading
from collections.abc import Coroutine
from datetime import datetime
from typing import Any

import cloudpickle
from pydantic import BaseModel, ConfigDict, Field

from ..config import Config
from ..db.dao import AsyncDAO
from ..stores.commons import DISK_REF_KEY, is_disk_ref, load_from_disk

# /// event loop deamon //////////////////////////////////////////////////////////


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


# ////////////////////////////////////////////////////////////////////////////////


class AttrsDiff:
    """Summary of differences between two attribute snapshots (params or exit context).

    Produced by :meth:`ExecutionParams.diff` and :meth:`TaskExitContext.diff`.
    No deserialization ever occurs.

    Attributes
    ----------
    only_in_self : list[str]
        Attr names present in the left-hand context but absent from the other.
    modified : list[str]
        Attr names present in both contexts whose stored SHA values differ.
    only_in_other : list[str]
        Attr names present in the other context but absent from the left-hand one.
    """

    def __init__(
        self,
        only_in_self: list[str],
        modified: list[str],
        only_in_other: list[str],
    ) -> None:
        self.only_in_self = only_in_self
        self.modified = modified
        self.only_in_other = only_in_other

    def __repr__(self) -> str:
        return (
            f"AttrsDiff("
            f"only_in_self={self.only_in_self}, "
            f"modified={self.modified}, "
            f"only_in_other={self.only_in_other})"
        )


class ExecutionParams:
    """Lazy mapping of DAG param names to their resolved values.

    Values (defaults or execution-time overrides) are deserialized
    from disk only when individually accessed, never all at once.
    Disk paths are resolved against the execution's own metadata_root
    as stored in DB, not the current ``Config.get_assets_cache_root()``.

    For disk-pickled params, value equality can be tested via ``param_equals()``
    using the stored SHA (computed on the Python object) ;
    no deserialization required.

    Parameters
    ----------
    params_json : dict
        Raw ``executions.params`` JSON as stored in DB.
    metadata_root : str
        Absolute path to the metadata root that was active when this
        execution ran (executions.metadata_root column).
    """

    def __init__(self, params_json: dict, metadata_root: str) -> None:
        self._raw = params_json
        self._metadata_root = metadata_root

    def _active_storable(self, key: str) -> Any:
        """Return the active storable (override if present, else default) for key."""
        param_def = self._raw[key]
        return param_def.get("override", param_def.get("default"))

    def _resolve(self, storable: Any) -> Any:
        """Resolve storable using this execution's metadata_root."""
        if is_disk_ref(storable):
            return load_from_disk(os.path.join(self._metadata_root, storable[DISK_REF_KEY]))
        return storable

    def __getitem__(self, key: str) -> Any:
        return self._resolve(self._active_storable(key))

    def __contains__(self, key: object) -> bool:
        return key in self._raw

    def __iter__(self):
        return iter(self._raw)

    def __len__(self) -> int:
        return len(self._raw)

    def __repr__(self) -> str:
        return f"ExecutionParams(params={list(self._raw.keys())})"

    def keys(self):
        return self._raw.keys()

    def description(self, key: str) -> str:
        """Return the description string for a param.

        Parameters
        ----------
        key : str
            Param name.

        Returns
        -------
        str
        """
        return self._raw[key]["description"]

    def default(self, key: str) -> Any:
        """Return the resolved default value for a param.

        The default is always returned regardless of whether an override
        is present. Disk-pickled defaults are deserialized on access.

        Parameters
        ----------
        key : str
            Param name.

        Returns
        -------
        Any
        """
        return self._resolve(self._raw[key].get("default"))

    def param_equals(self, key: str, other: "ExecutionParams") -> bool:
        """Return True if key holds the same value in both ExecutionParams instances.

        Comparison is always SHA-based ; no deserialization from disk ever occurs.
        SHAs are computed via sha256(cloudpickle.dumps(obj)).hexdigest() on
        the Python object itself, matching what ``value_to_storable`` stores
        for disk-pickled params.
        Disk-pickled params read their pre-computed SHA straight from DB ;
        native params have their SHA computed on the fly from the inline value.

        Parameters
        ----------
        key : str
            Param name to compare.
        other : ExecutionParams
            The other ExecutionParams instance to compare against.

        Returns
        -------
        bool
            True if both instances carry the same value for key.

        Raises
        ------
        KeyError
            If key is absent from either instance.

        Examples
        --------
        >>> params_set_a = Execution.getById(id=341).getParams()
        >>> params_set_b = Execution.getById(id=342).getParams()
        >>> # SHA comparison, no unpickling
        >>> params_set_a.param_equals("a_param_name", params_set_b)
        True
        """
        a = self._active_storable(key)
        b = other._active_storable(key)

        def _sha(storable: Any) -> str:
            # Disk-pickled: SHA was pre-computed on the Python object at store time.
            if isinstance(storable, dict) and DISK_REF_KEY in storable:
                return storable["__sha__"]

            # Native: compute SHA on the Python object the same way value_to_storable does.
            return hashlib.sha256(cloudpickle.dumps(storable)).hexdigest()

        return _sha(a) == _sha(b)

    def diff(self, other: "ExecutionParams") -> "AttrsDiff":
        """Return a SHA-based diff between this param set and another.

        No deserialization occurs: only param names and stored SHA values
        are compared. The active storable (override if present, else default)
        is used for each param on both sides.

        Parameters
        ----------
        other : ExecutionParams
            The param set to compare against.

        Returns
        -------
        AttrsDiff

        Examples
        --------
        >>> a = Execution.get_by_id(id=1).get_params()
        >>> b = Execution.get_by_id(id=2).get_params()
        >>> d = a.diff(b)
        >>> d.only_in_self   # params declared only in execution 1
        ['legacy_flag']
        >>> d.modified        # params present in both but with a different value
        ['dummy_param_1']
        >>> d.only_in_other  # params declared only in execution 2
        []
        """

        def _sha(storable: Any) -> str:
            if isinstance(storable, dict) and DISK_REF_KEY in storable:
                return storable["__sha__"]
            return hashlib.sha256(cloudpickle.dumps(storable)).hexdigest()

        self_keys = set(self._raw)
        other_keys = set(other._raw)
        return AttrsDiff(
            only_in_self=sorted(self_keys - other_keys),
            modified=sorted(
                k
                for k in self_keys & other_keys
                if _sha(self._active_storable(k)) != _sha(other._active_storable(k))
            ),
            only_in_other=sorted(other_keys - self_keys),
        )


class TaskExitContext:
    """Lazy view of a task's exit context.

    Attributes are deserialized from disk (or read from inline storage)
    only when individually accessed. The full attribute index is built
    from a single ``SELECT * FROM task_context_attrs WHERE task_id = ?``
    query ; O(1) regardless of DAG depth or history.

    Disk paths are resolved against the execution's own metadata_root
    as stored in DB, not the current ``Config.get_assets_cache_root()``.

    Parameters
    ----------
    rows : list
        TaskContextAttr ORM rows for this task, as returned by
        ``AsyncDAO.get_task_context_attrs(task_id)``.
    metadata_root : str
        Absolute path to the metadata root that was active when this
        execution ran (executions.metadata_root column).
    """

    def __init__(self, rows: list, metadata_root: str) -> None:
        self._index: dict[str, Any] = {row.attr_name: row for row in rows}
        self._metadata_root = metadata_root

    def __getitem__(self, key: str) -> Any:
        row = self._index[key]
        if row.disk_ref is not None:
            return load_from_disk(os.path.join(self._metadata_root, row.disk_ref))
        return row.inline_val

    def get(self, key: str, default: Any = None) -> Any:
        """Return the value for key, or default if absent."""
        if key not in self._index:
            return default
        return self[key]

    def __contains__(self, key: object) -> bool:
        return key in self._index

    def __iter__(self):
        return iter(self._index)

    def __len__(self) -> int:
        return len(self._index)

    def __repr__(self) -> str:
        return f"TaskExitContext(attrs={list(self._index.keys())})"

    def keys(self):
        return self._index.keys()

    def attr_equals(self, key: str, other: "TaskExitContext") -> bool:
        """Return True if key holds the same value in both TaskExitContext instances.

        Comparison is SHA-based; no deserialization from disk ever occurs.

        Parameters
        ----------
        key : str
            Attr name to compare.
        other : TaskExitContext
            The other TaskExitContext instance to compare against.

        Returns
        -------
        bool
            True if both instances carry the same value for key.

        Raises
        ------
        KeyError
            If key is absent from either instance.

        Examples
        --------
        >>> ctx_a = Execution.get_by_id(id=1).get_task_by_id(id=10).get_exit_context()
        >>> ctx_b = Execution.get_by_id(id=2).get_task_by_id(id=20).get_exit_context()
        >>> ctx_a.attr_equals("model_version", ctx_b)
        True
        """
        a_row = self._index.get(key)
        b_row = other._index.get(key)
        if a_row is None and b_row is None:
            return True
        if a_row is None or b_row is None:
            return False
        return a_row.sha == b_row.sha

    def diff(self, other: "TaskExitContext") -> AttrsDiff:
        """Return a SHA-based diff between this context and another.

        No deserialization occurs: only attr names and stored SHA values
        are compared.

        Parameters
        ----------
        other : TaskExitContext
            The context to compare against.

        Returns
        -------
        AttrsDiff

        Examples
        --------
        >>> a = Execution.get_by_id(id=1).get_task_by_id(id=10).get_exit_context()
        >>> b = Execution.get_by_id(id=2).get_task_by_id(id=20).get_exit_context()
        >>> d = a.diff(b)
        >>> d.only_in_self   # attrs added or kept only in execution 1's task
        ['model_accuracy']
        >>> d.modified        # attrs present in both but with a different value
        ['pipeline_version']
        >>> d.only_in_other  # attrs present only in execution 2's task
        ['legacy_flag']
        """
        self_keys = set(self._index)
        other_keys = set(other._index)
        return AttrsDiff(
            only_in_self=sorted(self_keys - other_keys),
            modified=sorted(
                k for k in self_keys & other_keys if self._index[k].sha != other._index[k].sha
            ),
            only_in_other=sorted(other_keys - self_keys),
        )


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
    """

    id: int = Field(..., description="Unique execution identifier")
    name: str = Field(..., description="Pipeline name")
    start_timestamp: datetime = Field(..., description="Execution start time (UTC)")
    end_timestamp: datetime | None = Field(
        None, description="Execution end time (UTC), None if still running (not live-synched)"
    )
    success: bool = Field(..., description="Whether execution completed successfully")
    metadata_root: str = Field(
        None,
        description=(
            "Absolute path to {Config.get_assets_cache_root()}/metadata/ as it was on the machine "
            "and at the time this execution ran. Used by SDK read methods to resolve "
            "disk artifacts independently of the current ``Config.get_assets_cache_root()`` value."
        ),
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

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
            start_timestamp=row.start_timestamp,
            end_timestamp=row.end_timestamp,
            success=row.success,
            metadata_root=row.metadata_root,
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
                    start_timestamp=t.start_timestamp,
                    end_timestamp=t.end_timestamp,
                    success=not t.failed if t.failed is not None else True,
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
        >>> task = Execution.get_by_id(id=356).get_task_by_id(id=42)
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
            start_timestamp=row.start_timestamp,
            end_timestamp=row.end_timestamp,
            success=not row.failed if row.failed is not None else True,
            metadata_root=self.metadata_root,
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
        id: Unique task identifier
        name: TaskType name
        start_timestamp: When task started
        end_timestamp: When task ended (None if still running)
        success: Whether task completed successfully
        metadata_root: Absolute metadata root from the parent execution
    """

    id: int = Field(..., description="Unique task identifier")
    name: str = Field(..., description="Task type name")
    start_timestamp: datetime = Field(..., description="Task start time (UTC)")
    end_timestamp: datetime | None = Field(
        None, description="Task end time (UTC), None if still running (not live-synched)"
    )
    success: bool | None = Field(None, description="Whether task completed successfully")
    metadata_root: str = Field(
        None,
        description=(
            "Absolute path to the metadata root from the parent execution. "
            "Propagated from Execution.metadata_root; used to resolve disk artifacts."
        ),
    )

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
        return TaskExitContext(rows or [], self.metadata_root)
