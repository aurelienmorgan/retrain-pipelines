from datetime import datetime
from typing import cast

import tzlocal
from fasthtml.common import A, B, Div, Span, Style

from .....utils import hex_to_rgba
from ....config import Config
from ....core import UiCss
from ....db.dao import AsyncDAO
from ....db.model import Execution, ExecutionExt

server_tz = tzlocal.get_localzone()


async def get_users() -> list[str]:
    dao = AsyncDAO(db_url=Config.get_metadatastore_async_url())
    return await dao.get_distinct_execution_usernames(sorted=True)


async def get_pipeline_names() -> list[str]:
    dao = AsyncDAO(db_url=Config.get_metadatastore_async_url())
    return await dao.get_distinct_execution_names(sorted=True)


def execution_to_html(execution_ext: Execution | ExecutionExt) -> Div:
    localized_start_timestamp = execution_ext.start_timestamp.astimezone(server_tz)
    localized_start_timestamp_str = localized_start_timestamp.strftime("%Y-%m-%d %H:%M:%S")

    ui_css = UiCss(**execution_ext.ui_css) if execution_ext.ui_css else UiCss()
    exec_background = ui_css.background or "#4d0066"
    exec_color = ui_css.color or "#fff"
    exec_border = ui_css.border or None  # "#FFD700"

    return Div(
        A(
            Div(
                # Glass shine overlay
                style=(
                    "position: absolute; top: 0; left: 0; right: 0; "
                    "height: 40%; "
                    "background: linear-gradient(135deg, "
                    "rgba(255,255,255,0.3) 0%, "
                    "rgba(255,255,255,0.1) 50%, transparent 100%); "
                    "pointer-events: none; border-radius: 6px 6px 0 0; "
                    "transition: opacity 0.2s ease;"
                ),
                id="glass-overlay",
            ),
            Div(
                B(Span(f"{execution_ext.name} ")),
                Span(f"\u00a0- {localized_start_timestamp_str}"),
                cls=["execution-body"],
            ),
            Div(
                f"{int((d := execution_ext.end_timestamp - execution_ext.start_timestamp).total_seconds() // 3600)}:{int((d.total_seconds() % 3600) // 60):02}:{int(d.total_seconds() % 60):02}.{(d.microseconds // 1000):03}"  # noqa: E501
                if execution_ext.end_timestamp
                else "",
                cls="end_timestamp"
                + (
                    (" success" if cast(ExecutionExt, execution_ext).success else " failure")
                    if execution_ext.end_timestamp
                    else ""
                ),
            ),
            href=f"/execution?id={execution_ext.id}",
            target="_self",
            style="color: inherit; text-decoration: none; display: flex; width: 100%;",
        ),
        Style(f"""
                #_{execution_ext.id}.execution {{
                    --background-normal: {hex_to_rgba(exec_background, 0.35)};
                    --background-hover: {hex_to_rgba(exec_background, 0.75)};
                    --color-normal: {hex_to_rgba(exec_color, 0.65)};
                    --color-hover: {hex_to_rgba(exec_color, 1)};
                    --border-normal: {hex_to_rgba(exec_border, 0.45) if exec_border else " "};
                    --border-hover: {hex_to_rgba(exec_border, 0.85) if exec_border else " "};
                }}
            """),
        **{
            "data-pipeline-name": execution_ext.name,
            "data-username": execution_ext.username,
            "data-start-timestamp": execution_ext.start_timestamp,
            "data-success": (
                str(execution_ext.success) if hasattr(execution_ext, "success") else ""
            ),
        },
        cls=["execution", "wavy-list-item", "wavy-list-item-body"],
        id=f"_{execution_ext.id}",
    ).__html__()


async def get_executions_ext(
    pipeline_name: str | None = None,
    username: str | None = None,
    before_datetime: datetime | None = None,
    execs_status: str | None = None,
    n: int | None = None,
    descending: bool | None = False,
) -> list[str]:
    """List Execution records from a given start time.

    Returns a styled DOM element's html.

    Parameters
    ----------
    pipeline_name : str
        the only retraining pipeline to consider
        (if mentioned)
    username : str
        the user having lunched the executions
        to consider (if mentioned)
    before_datetime : datetime
        UTC time from which to start listing
    execs_status : str
        any (None)/success/failure
    n : int
        number of Executions to retrieve
    descending : bool
        sorting order, wheter latest comes first
        or last

    Returns
    -------
    List[str]
    """
    dao = AsyncDAO(db_url=Config.get_metadatastore_async_url())
    executions_ext = await dao.get_executions_ext(
        pipeline_name=pipeline_name,
        username=username,
        before_datetime=before_datetime,
        execs_status=execs_status,
        n=n,
        descending=descending,
    )

    dom_executions = []
    for execution_ext in executions_ext:
        dom_executions.append(execution_to_html(execution_ext))

    return dom_executions
