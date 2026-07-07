import asyncio
import concurrent.futures
import hashlib
import os
from datetime import datetime
from typing import Any

import cloudpickle
from pydantic import BaseModel, ConfigDict, Field

from ...utils import in_notebook
from ..context_store import _DISK_REF_KEY, resolve_storable
from ..db.dao import AsyncDAO


def _run_async(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    if in_notebook():
        # Jupyter (IPython kernel) already
        # starts and runs an event loop internally.
        # Run the coroutine in a dedicated worker thread
        # with its own fresh event loop
        # so that the kernel's running loop is left untouched.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()

    return asyncio.get_event_loop().run_until_complete(coro)


class ExecutionParams:
    """Lazy mapping of DAG param names to their resolved values.

    Values (defaults or execution-time overrides) are deserialized
    from disk only when individually accessed, never all at once.

    For disk-pickled params, value equality can be tested via ``param_equals()``
    using the stored SHA (computed on the Python object) ; no deserialization required.

    Parameters
    ----------
    params_json : dict
        Raw ``executions.params`` JSON as stored in DB.
    """

    def __init__(self, params_json: dict) -> None:
        self._raw = params_json

    def _active_storable(self, key: str) -> Any:
        """Return the active storable (override if present, else default) for key."""
        param_def = self._raw[key]
        return param_def.get("override", param_def.get("default"))

    def __getitem__(self, key: str) -> Any:
        return resolve_storable(self._active_storable(key))

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
        return resolve_storable(self._raw[key].get("default"))

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
            if isinstance(storable, dict) and _DISK_REF_KEY in storable:
                return storable["__sha__"]
            # Native: compute SHA on the Python object the same way value_to_storable does.
            return hashlib.sha256(cloudpickle.dumps(storable)).hexdigest()

        return _sha(a) == _sha(b)


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
    """

    id: int = Field(..., description="Unique execution identifier")
    name: str = Field(..., description="Pipeline name")
    start_timestamp: datetime = Field(..., description="Execution start time (UTC)")
    end_timestamp: datetime | None = Field(
        None, description="Execution end time (UTC), " + "None if still running (not live-synched)"
    )
    success: bool = Field(..., description="Whether execution completed successfully")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def getById(cls, id: int) -> "Execution":
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
            dao = AsyncDAO(db_url=os.environ["RP_METADATASTORE_ASYNC_URL"])
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
        )

    def completed(self) -> bool:
        """Check if execution has completed.

        Returns
        -------
        bool
        true if execution has finished
        (successfully or not).
        """
        return self.end_timestamp is not None

    def get_attr(self, attr_name: str) -> Any:
        """Get attribute from execution context_dump.

        Fetches custom attributes stored in the execution's context,
        such as model versions, parameters, or other metadata.

        Parameters
        ----------
        attr_name : str
            name of the attribute to retrieve

        Returns
        -------
        Any
            Value of the attribute or None if not found.

        Examples
        --------
        >>> version = await execution.get_attr("model_version_blessed")
        """

        async def _get_attr_async():
            dao = AsyncDAO(db_url=os.environ["RP_METADATASTORE_ASYNC_URL"])
            full_exec = await dao.get_execution(self.id)
            if full_exec.context_dump:
                # print(f"_get_attr_async - {full_exec.context_dump.keys()}")
                return (
                    full_exec.context_dump[attr_name]
                    if attr_name in full_exec.context_dump
                    else None
                )
            return None

        return _run_async(_get_attr_async())

    def get_tasks_with_name(self, task_type_name: str) -> list["Task"]:
        """Retrurn tasks by name.

        Returns
        -------
        List[Task]
        naive unordered list of task instances.
        """

        async def _get_tasks_async():
            dao = AsyncDAO(db_url=os.environ["RP_METADATASTORE_ASYNC_URL"])
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
                )
                for t in tasks_orm
            ]

        return _run_async(_get_tasks_async())

    def getParams(self) -> "ExecutionParams":
        """Return a lazy view of the resolved DAG params for this execution.

        The DB is queried once to fetch the params JSON; individual param
        values (defaults or overrides) are deserialized from disk only when
        first accessed by key.

        Returns
        -------
        ExecutionParams
            Lazy mapping of param name => resolved value.

        Examples
        --------
        >>> exec_params = Execution.getById(id=356).getParams()
        >>> for param_name in exec_params:
        >>>    print(f"{param_name}, {exec_params[param_name]}")
        param1 param1_value
        param2 param2_value
        """

        async def _fetch():
            dao = AsyncDAO(db_url=os.environ["RP_METADATASTORE_ASYNC_URL"])
            full_exec = await dao.get_execution(self.id)
            return full_exec.params

        params_json = _run_async(_fetch()) or {}
        return ExecutionParams(params_json)

    def elements_iterator(self):
        """Get iterator over execution elements.

        Iterates through tasks, taskgroups,
        and sub-DAGs in topological order.

        Returns
        -------
        Iterator over execution elements.

        Notes
        -----
        "next" on a sub-DAG
        returns the element following
        its closing merge task.
        """
        # TODO: Implement iteration over tasks, taskgroups, and sub-DAGs
        raise NotImplementedError()


class Task(BaseModel):
    """SDK representation of a pipeline task instance.

    Provides access to task metadata.

    Attributes
    ----------
        id: Unique task identifier
        task_type_name: TaskType name
        start_timestamp: When task started
        end_timestamp: When task ended (None if still running)
        success: Whether task completed successfully
    """

    id: int = Field(..., description="Unique task identifier")
    name: str = Field(..., description="Pipeline name")
    start_timestamp: datetime = Field(..., description="Task start time (UTC)")
    end_timestamp: datetime | None = Field(
        None, description="Task end time (UTC), " + "None if still running (not live-synched)"
    )
    success: bool | None = Field(None, description="Whether task completed successfully")
