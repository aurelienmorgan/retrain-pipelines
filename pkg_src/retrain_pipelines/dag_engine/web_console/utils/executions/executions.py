
import os
import tzlocal

from datetime import datetime
from typing import List, Optional, Union

from fasthtml.common import Div, A, Span

from ....db.dao import AsyncDAO
from ....db.model import Execution, ExecutionExt


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


def execution_to_html(execution_ext: Union[Execution, ExecutionExt]) -> Div:
    localized_start_timestamp = \
        execution_ext.start_timestamp.astimezone(server_tz)
    localized_start_timestamp_str = \
        localized_start_timestamp.strftime('%Y-%m-%d %H:%M:%S')
    return Div(
        Div(
            Span(f"{execution_ext.name} "),
            A(
                f"[{execution_ext.id}]",
                href=f"/execution?id={execution_ext.id}",
                target="_self"
            ),
            Span(f" - {localized_start_timestamp_str}")
        ),
        Div(
            (execution_ext.end_timestamp - execution_ext.start_timestamp) \
                if execution_ext.end_timestamp else "",
            cls="end_timestamp" + ((
                    ", success" if execution_ext.success else ", failure"
                ) if execution_ext.end_timestamp else "")
        ),
        **{
            'data-pipeline-name': execution_ext.name,
            'data-username': execution_ext.username,
            'data-start-timestamp': execution_ext.start_timestamp,
            'data-success': (
                str(execution_ext.success)
                if hasattr(execution_ext, "success") else ""
            )
        },
        cls="execution",
        id=str(execution_ext.id)
    ).__html__()


async def get_executions_ext(
    pipeline_name: Optional[str] = None,
    username: Optional[str] = None,
    before_datetime: Optional[datetime] = None,
    execs_status: Optional[datetime] = None,
    n: Optional[int] = None,
    descending: Optional[bool] = False
) -> List[str]:
    """Lists Execution records from a given start time.

    Returns a styled DOM element's html.

    Params:
        - pipeline_name (str):
            the only retraining pipeline to consider
            (if mentioned)
        - username (str):
            the user having lunched the executions
            to consider (if mentioned)
        - before_datetime (datetime):
            UTC time from which to start listing
        - execs_status str):
            any (None)/success/failure
        - n (int):
            number of Executions to retrieve
        - descending (bool):
            sorting order, wheter latest comes first
            or last

    Results:
        List[str]
    """
    dao = AsyncDAO(
        db_url=os.environ["RP_METADATASTORE_ASYNC_URL"]
    )
    executions_ext = await dao.get_executions_ext(
        pipeline_name=pipeline_name, username=username,
        before_datetime=before_datetime,
        execs_status=execs_status, n=n,
        descending=descending
    )
    print("executions.get_executions_ext ", n, len(executions_ext))

    dom_executions = []
    for execution_ext in executions_ext:
        dom_executions.append(execution_to_html(execution_ext))

    return dom_executions

