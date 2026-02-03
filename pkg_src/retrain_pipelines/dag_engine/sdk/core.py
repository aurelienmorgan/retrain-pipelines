
import os
import asyncio

from datetime import datetime
from pydantic import BaseModel, Field
from typing import Any, Optional, ClassVar

from ..db.dao import AsyncDAO


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
                          "None if still running (not live synched)")
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
                return full_exec.context_dump[attr_name]
            return None

        return asyncio.run(_get_attr_async())

        asyncio.run(_get_attr_async())

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

