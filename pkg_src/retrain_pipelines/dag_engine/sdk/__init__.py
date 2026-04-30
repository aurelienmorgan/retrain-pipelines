
import os
import sys
import asyncio
import logging
import concurrent.futures

from typing import Optional, List
from datetime import datetime, timedelta
from pydantic import BaseModel, Field, PrivateAttr

from .core import Execution
from ..db.dao import AsyncDAO
from ...utils import in_notebook


logger = logging.getLogger(__name__)


def _run_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    if in_notebook():
        """
        Jupyter (IPython kernel) already
        starts and runs an event loop internally.
        Run the coroutine in a dedicated worker thread
        with its own fresh event loop
        so that the kernel's running loop is left untouched:
            - loop.run_until_complete()
            - asyncio.run()
            - other loop management calls
        """
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=1
        ) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()

    return loop.run_until_complete(coro)


class ExecutionsIterator(BaseModel):
    """Reversed paged iterator over executions.

    Fetches executions in pages in reverse chronological order
    (newest first) without loading all executions into memory.

    Params:
        - exec_name (str):
            name of the execution pipeline to iterate over
        - success_only (bool):
            if True, only iterates over successful executions
        - page_size (int):
            number of executions to fetch per page (default: 10)
    """
    exec_name: str = Field(
        ..., description="Name of the execution pipeline")
    success_only: Optional[bool] = Field(
        False, description="Filter for successful executions only")
    page_size: int = Field(
        10, description="Number of executions to fetch per page")

    _before_datetime: Optional[datetime] = PrivateAttr(
        default=None                        # tracks position in iteration
    )
    _buffer: List[Execution] = \
        PrivateAttr(default_factory=list)   # buffer for current page
    _index: int = \
        PrivateAttr(default=0)              # current position in buffer

    class Config:
        arbitrary_types_allowed = True

    async def _previous(self) -> Optional[Execution]:
        """Get the previous (older) execution in the sequence.

        Results:
            - (Execution):
                previous Execution object in line,
                none if no more executions exist

        Note:
            Timestamps are stored at millisecond precision.
            We assume no two executions of the same pipeline
            start within the same millisecond
            (at page boundaries).
        """
        if not self._buffer or self._index >= len(self._buffer):
            dao = AsyncDAO(
                db_url=os.environ["RP_METADATASTORE_ASYNC_URL"]
            )
            execs = await dao.get_executions_ext(
                pipeline_name=self.exec_name,
                execs_status="success" if self.success_only else None,
                before_datetime=self._before_datetime,
                n=self.page_size,
                descending=True
            )
            if not execs:
                self._buffer = []
                self._index = 0
                return None

            self._buffer = [
                Execution(
                    id=exec_ext.id,
                    name=exec_ext.name,
                    start_timestamp=exec_ext.start_timestamp,
                    end_timestamp=exec_ext.end_timestamp,
                    success=exec_ext.success
                ) for exec_ext in execs
            ]

            # Update position marker for next page fetch
            # ensure strictly older results next time
            # (we assume here that no 2 executions
            #  of the same pipeline started
            #  within the same millisecond at page boundaries)
            # (recall that timestamps are purposely
            #  stored at millisecond [not microsecond] precision)
            last_exec = self._buffer[-1]
            self._before_datetime = \
                last_exec.start_timestamp - timedelta(milliseconds=1)

            self._index = 0

        execution = self._buffer[self._index]
        self._index += 1
        return execution

    def previous(self) -> Optional[Execution]:
        """Get the previous (older) execution in the sequence.

        Results:
            - (Execution):
                previous Execution object in line,
                none if no more executions exist

        Note:
            Timestamps are stored at millisecond precision.
            We assume no two executions of the same pipeline
            start within the same millisecond
            (at page boundaries).
        """
        logger.debug(f"{self}  - previous")
        return _run_async(self._previous())

    def length(self) -> int:
        """Get total count of executions matching the criteria.

        Implementation using SQL COUNT without fetching all rows.

        Results:
            - (int):
                total number of matching executions.
        """
        dao = AsyncDAO(
            db_url=os.environ["RP_METADATASTORE_ASYNC_URL"]
        )
        logger.debug(f"{self}  - length")
        return _run_async(
            dao.get_executions_count(
                pipeline_name=self.exec_name,
                execs_status="success" if self.success_only else None
            )
        )

    # async iterator
    def __aiter__(self):
        return self

    async def __anext__(self):
        execution = await self._previous()
        if execution is None:
            raise StopAsyncIteration
        return execution

    # sync iterator
    def __iter__(self):
        return self

    def __next__(self):
        execution = self.previous()
        if execution is None:
            raise StopIteration
        return execution

