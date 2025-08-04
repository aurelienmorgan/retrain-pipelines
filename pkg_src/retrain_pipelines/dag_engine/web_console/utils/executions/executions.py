
import os
import tzlocal

from datetime import datetime
from typing import List, Optional, Union

from fasthtml.common import Div, A, Span, \
    Style

from ....db.dao import AsyncDAO
from ....db.model import Execution, ExecutionExt
from .....utils import hex_to_rgba


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

    exec_background = \
        execution_ext.ui_css["background"] \
        if (execution_ext.ui_css and "background" in execution_ext.ui_css) \
        else "#4d0066" # "#ffffff"

    exec_color = execution_ext.ui_css["background"] \
                 if (execution_ext.ui_css and "background" in execution_ext.ui_css) \
                 else "#4d0066" # "#ffffff"
    # status_color = (
        # (
            # "#28a745"
            # if execution_ext.success else
            # "#dc3545"
        # ) if execution_ext.end_timestamp else
        # None
    # )
    # print(execution_ext.end_timestamp, execution_ext.success, status_color)

    return Div(

        Div(
            # Glass shine overlay
            style=(
                "position: absolute; top: 0; left: 0; right: 0; "
                "height: 40%; "
                "background: linear-gradient(135deg, "
                    "rgba(255,255,255,0.3) 0%, "
                    "rgba(255,255,255,0.1) 50%, transparent 100%); "
                "pointer-events: none; border-radius: 2px 2px 0 0; "
                "transition: opacity 0.2s ease;"
            ),
            id="glass-overlay"
        ),

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
                    " success" if execution_ext.success else " failure"
                ) if execution_ext.end_timestamp else "")
        ),
        Style(f"""
            #_{execution_ext.id}.execution {{
                --exec_background-normal: {hex_to_rgba(exec_background, .45)};
                --exec_background-hover: {hex_to_rgba(exec_background, .65)};
            }}
        """),

        # Style(f"""
            # #_{execution_ext.id}.execution {{
                # display: flex;
                # justify-content: space-between;
                # align-items: center;
                # width: 100%;
                # position: relative;
                # --status-color-normal: {hex_to_rgba(exec_background, .45)};
                # --status-color-hover: {hex_to_rgba(exec_background, .65)};
                # background: {
                    # f"linear-gradient(120deg, var(--status-color-normal) 0%, var(--status-color-normal) 40%, {hex_to_rgba(status_color, 0.5)} 70%, {hex_to_rgba(status_color, 0)} 100%)"
                    # if status_color else "var(--status-color-normal)"
                # };
                # padding: 8px 20px;
                # margin-bottom: 4px;
                # border-radius: 8px;
                # box-shadow: 0 2px 4px rgba(0,0,0,0.1),
                    # 0 8px 16px rgba(0,0,0,0.05),
                    # inset 0 1px 0 rgba(255,255,255,0.4),
                    # inset 0 -1px 0 rgba(0,0,0,0.1);
                # backdrop-filter: blur(10px);
                # -webkit-backdrop-filter: blur(10px);
                # overflow: hidden;
                # transition: all 0.3s ease;
                # transform-origin: center center;
            # }}

            # #_{execution_ext.id}.execution:hover {{
                # {"background: linear-gradient(120deg, var(--status-color-hover) 0%, var(--status-color-hover) 40%, " + hex_to_rgba(status_color, 0.7) + " 70%, " + hex_to_rgba(status_color, 0) + " 100%);" if status_color else ""}
                # transform: translateY(-1px);
            # }}
        # """),

        # Style(f"""
            # #_{execution_ext.id}.execution {{
                # display: flex;
                # justify-content: space-between;
                # align-items: center;
                # width: 100%;
                # position: relative;
                # --status-color-normal: {hex_to_rgba(exec_background, .45)};
                # --status-color-hover: {hex_to_rgba(exec_background, .65)};
                # background: {
                    # f"radial-gradient(ellipse at 100% 50%, {hex_to_rgba(status_color, 0.6)} 0%, {hex_to_rgba(status_color, 0.3)} 30%, transparent 60%), var(--status-color-normal)"
                    # if status_color else "var(--status-color-normal)"
                # };
                # padding: 8px 20px;
                # margin-bottom: 4px;
                # border-radius: 8px;
                # box-shadow: 0 2px 4px rgba(0,0,0,0.1),
                    # 0 8px 16px rgba(0,0,0,0.05),
                    # inset 0 1px 0 rgba(255,255,255,0.4),
                    # inset 0 -1px 0 rgba(0,0,0,0.1);
                # backdrop-filter: blur(10px);
                # -webkit-backdrop-filter: blur(10px);
                # overflow: hidden;
                # transition: all 0.3s ease;
                # transform-origin: center center;
            # }}

            # #_{execution_ext.id}.execution:hover {{
                # {"background: radial-gradient(ellipse at 100% 50%, " + hex_to_rgba(status_color, 0.8) + " 0%, " + hex_to_rgba(status_color, 0.5) + " 30%, transparent 60%), var(--status-color-hover);" if status_color else ""}
                # transform: translateY(-1px);
            # }}
        # """),

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
        id=f"_{execution_ext.id}",
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

    dom_executions = []
    for execution_ext in executions_ext:
        dom_executions.append(execution_to_html(execution_ext))

    return dom_executions

