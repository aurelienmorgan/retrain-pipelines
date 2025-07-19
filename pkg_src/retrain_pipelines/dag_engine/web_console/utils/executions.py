
import os

from typing import List
from datetime import datetime

from fasthtml.common import *

from ...db.dao import AsyncDAO
from ...db.model import Execution


async def get_users() -> List[str]:
    dao = AsyncDAO(
        db_url=os.environ["RP_METADATASTORE_ASYNC_URL"]
    )
    return await dao.get_distinct_execution_usernames()


async def get_pipeline_names() -> List[str]:
    dao = AsyncDAO(
        db_url=os.environ["RP_METADATASTORE_ASYNC_URL"]
    )
    return await dao.get_distinct_execution_names()


async def get_executions_before(
    before_datetime: datetime,
    n: int
): # -> List[Execution]:
    """Lists Execution records from a given start time.

    Params:
        - before_datetime (datetime):
            time from which to start listing
        - n (int):
            number of Executions to retrieve

    Results:
        List[Execution]
    """
    print(type(before_datetime))
    print(before_datetime)

    dao = AsyncDAO(
        db_url=os.environ["RP_METADATASTORE_ASYNC_URL"]
    )
    executions = await dao.get_executions_before(
        before_datetime, n
    )
    print(n, len(executions))

    return executions

