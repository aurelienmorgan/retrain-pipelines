
import os
import asyncio
import concurrent.futures

from datetime import datetime
from pydantic import BaseModel, Field
from typing import Any, Optional, List, \
    ClassVar

from ..db.dao import AsyncDAO
from ...utils import in_notebook


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
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=1
        ) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()

    return asyncio.get_event_loop().run_until_complete(coro)


class Execution(BaseModel):
    """SDK representation of a pipeline execution.

    Provides access to execution metadata and context attributes.

    Attributes:
        id: Unique execution identifier
        name: Pipeline name
        start_timestamp: When execution started
        end_timestamp: When execution ended (None if still running)
        success: Whether execution completed successfully
    """
    id: int = Field(
        ..., description="Unique execution identifier")
    name: str = Field(..., description="Pipeline name")
    start_timestamp: datetime = Field(
        ..., description="Execution start time (UTC)")
    end_timestamp: Optional[datetime] = Field(
        None, description="Execution end time (UTC), " +
                          "None if still running (not live-synched)")
    success: bool = Field(
        ..., description="Whether execution completed successfully")

    class Config:
        arbitrary_types_allowed = True

    def completed(self) -> bool:
        """Check if execution has completed.

        Results:
            - (bool):
                true if execution has finished
                (successfully or not).
        """
        return self.end_timestamp is not None

    def get_attr(self, attr_name: str) -> Any:
        """Get attribute from execution context_dump.

        Fetches custom attributes stored in the execution's context,
        such as model versions, parameters, or other metadata.

        Params:
            - attr_name (str):
                name of the attribute to retrieve

        Results:
            Value of the attribute or None if not found.

        Example:
            version = await execution.get_attr("model_version_blessed")
        """
        async def _get_attr_async():
            dao = AsyncDAO(
                db_url=os.environ["RP_METADATASTORE_ASYNC_URL"]
            )
            full_exec = await dao.get_execution(self.id)
            if full_exec.context_dump:
                # print(f"_get_attr_async - {full_exec.context_dump.keys()}")
                return full_exec.context_dump[attr_name] \
                       if attr_name in full_exec.context_dump \
                       else None
            return None

        return _run_async(_get_attr_async())

    def get_tasks_with_name(self, task_type_name: str) -> List["Task"]:
        """
        Results:
            - (List[Task]):
                naive unordered list of task instances.
        """
        async def _get_tasks_async():
            dao = AsyncDAO(
                db_url=os.environ["RP_METADATASTORE_ASYNC_URL"]
            )
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
                    success=not t.failed if t.failed is not None \
                            else True
                )
                for t in tasks_orm
            ]

        return _run_async(_get_tasks_async())

    def elements_iterator(self):
        """Get iterator over execution elements.

        Iterates through tasks, taskgroups,
        and sub-DAGs in topological order.
        NOTE : "next" on a sub-DAG
               returns the element following
               its closing merge task.

        Results:
            Iterator over execution elements.
        """
        # TODO: Implement iteration over tasks, taskgroups, and sub-DAGs
        raise NotImplementedError()


class Task(BaseModel):
    """SDK representation of a pipeline task instance.

    Provides access to task metadata.

    Attributes:
        id: Unique task identifier
        task_type_name: TaskType name
        start_timestamp: When task started
        end_timestamp: When task ended (None if still running)
        success: Whether task completed successfully
    """
    id: int = Field(
        ..., description="Unique task identifier")
    name: str = Field(..., description="Pipeline name")
    start_timestamp: datetime = Field(
        ..., description="Task start time (UTC)")
    end_timestamp: Optional[datetime] = Field(
        None, description="Task end time (UTC), " +
                          "None if still running (not live-synched)")
    success: Optional[bool] = Field(
        None, description="Whether task completed successfully")

