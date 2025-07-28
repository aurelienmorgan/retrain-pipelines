
import os
import tzlocal

from typing import List
from datetime import datetime

from fasthtml.common import *

from ...db.dao import AsyncDAO
from ...db.model import Execution


server_tz = tzlocal.get_localzone()


async def get_users() -> List[str]:
    dao = AsyncDAO(
        db_url=os.environ["RP_METADATASTORE_ASYNC_URL"]
    )
    return await dao.get_distinct_execution_usernames(
        sorted=True)


async def get_pipeline_names() -> List[str]:
    dao = AsyncDAO(
        db_url=os.environ["RP_METADATASTORE_ASYNC_URL"]
    )
    return await dao.get_distinct_execution_names(
        sorted=True)


async def get_executions(
    pipeline_name: Optional[str] = None,
    username: Optional[str] = None,
    before_datetime: Optional[datetime] = None,
    n: Optional[int] = None,
    descending: Optional[bool] = False
):
    """Lists Execution records from a given start time.

    Params:
        - pipeline_name (str):
            the only retraining pipeline to consider
            (if mentioned)
        - username (str):
            the user having lunched the executions
            to consider (if mentioned)
        - before_datetime (datetime):
            UTC time from which to start listing
        - n (int):
            number of Executions to retrieve
        - descending (bool):
            sorting order, wheter latest comes first
            or last

    Results:
        List[Execution]
    """
    dao = AsyncDAO(
        db_url=os.environ["RP_METADATASTORE_ASYNC_URL"]
    )
    executions = await dao.get_executions(
        pipeline_name=pipeline_name, username=username,
        before_datetime=before_datetime, n=n,
        descending=descending
    )
    print("executions.get_executions ", n, len(executions))

    dom_executions = []
    for execution in executions:
        dom_executions.append(
            Div(
                f"{execution.name} [{execution.id}] - {execution.start_timestamp.astimezone(server_tz).strftime('%Y-%m-%d %H:%M:%S')}"
            )
        )

    return dom_executions

