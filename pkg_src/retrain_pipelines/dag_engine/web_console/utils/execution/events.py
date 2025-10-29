
import os
import json
import copy
import asyncio
import logging

from uuid import UUID
from functools import lru_cache

from typing import Union
from fasthtml.common import Response, JSONResponse

from ....db.model import Execution, ExecutionExt
from ....db.dao import AsyncDAO
from .. import ClientInfo
from .gantt_chart import fill_defaults, Style, \
    GroupTypes


# Global lists of subscriber queues
new_exec_subscribers = []
exec_end_subscribers = []

new_task_subscribers = []
task_end_subscribers = []


uvicorn_logger = logging.getLogger("uvicorn.access")


async def execution_number(
    execution_id: int
) -> Union[Response, JSONResponse]:
    """ Living counts - DO NOT USE CACHE """
    dao = AsyncDAO(
        db_url=os.environ["RP_METADATASTORE_ASYNC_URL"]
    )
    execution_number_dict = await dao.get_execution_number(execution_id)
    if execution_number_dict is None:
        return Response(
            f"Invalid execution ID {execution_id}", 500)
    # print(f"execution_number_dict : {execution_number_dict}")

    return JSONResponse(execution_number_dict)


async def taskgroups_hierarchy(
    taskgroup_uuid: UUID
) -> Union[Response, JSONResponse]:
    dao = AsyncDAO(
        db_url=os.environ["RP_METADATASTORE_ASYNC_URL"]
    )
    taskgroups_list = await dao.get_taskgroups_hierarchy(taskgroup_uuid)
    if taskgroups_list is None:
        return Response(
            f"Invalid TaskType UUID {str(taskgroup_uuid)}", 500)
    # print(f"taskgroups_list : {taskgroups_list}")

    return JSONResponse(taskgroups_list)


async def augment_new_task(task_ext_dict: dict) -> dict:
    # if task_ext is the head of a distributed sub-DAG split-line
    if task_ext_dict["is_parallel"]:
        parallel_lines_style = Style(task_ext_dict["ui_css"],
                                     labelUnderlay="#4d0066")
        fill_defaults(parallel_lines_style,
                      GroupTypes.PARALLEL_LINES)
        parallel_line_style = Style(task_ext_dict["ui_css"],
                                    labelUnderlay="#4d0066")
        fill_defaults(parallel_line_style,
                      GroupTypes.PARALLEL_LINE)
        task_ext_dict["parent_ui_css"] = {
            "parallel_lines": parallel_lines_style,
            "parallel_line": parallel_line_style
        }

    # task_ext styling, fill defaults
    task_ext_style = Style(
        task_ext_dict["ui_css"],
        labelUnderlay="#4d0066" # will be overridden
                                # for taskgroup tasks
    )
    fill_defaults(task_ext_style, GroupTypes.NONE)
    task_ext_dict["ui_css"] = task_ext_style

    # if task_ext belongs to a TaskGroup
    taskgroup_uuid = task_ext_dict["taskgroup_uuid"]
    if taskgroup_uuid:
        taskgroups_hierarchy_response = \
            await taskgroups_hierarchy(UUID(taskgroup_uuid))
        if taskgroups_hierarchy_response.status_code == 200:
            taskgroups_hierarchy_list = \
                json.loads(
                    taskgroups_hierarchy_response.body.decode("utf-8")
                )
            for taskgroup_dict in taskgroups_hierarchy_list:
                # taskgroup styling, fill defaults
                taskgroup_style = Style(taskgroup_dict["ui_css"])
                fill_defaults(taskgroup_style, GroupTypes.TASKGROUP)
                taskgroup_dict["ui_css"] = taskgroup_style
            # task styling, adapt labelUnderlay color
            task_ext_dict["ui_css"]["labelUnderlay"] = \
                taskgroups_hierarchy_list[0]["ui_css"]["background"]
        else:
            logging.getLogger().warn(
                taskgroups_hierarchy_response.body.decode("utf-8"))
            taskgroups_hierarchy_list = []
    else:
        taskgroups_hierarchy_list = []
    task_ext_dict["taskgroups_hierarchy"] =  taskgroups_hierarchy_list


async def multiplexed_event_generator(client_info: ClientInfo):
    queues = {
        "newExecution": asyncio.Queue(),
        "executionEnded": asyncio.Queue(),

        "newTask": asyncio.Queue(),
        "taskEnded": asyncio.Queue()
    }

    new_exec_subscribers.append((queues["newExecution"], client_info))
    exec_end_subscribers.append((queues["executionEnded"], client_info))
    print(f"execution subscribers [{len(new_exec_subscribers)}] : {new_exec_subscribers}")

    new_task_subscribers.append((queues["newTask"], client_info))
    task_end_subscribers.append((queues["taskEnded"], client_info))
    print(f"task subscribers [{len(new_task_subscribers)}] : {task_end_subscribers}")

    try:
        # Initial get asyncio tasks
        get_tasks = {
            key: asyncio.create_task(q.get())
            for key, q in queues.items()
        }
        while True:
            # Wait for any queue
            done, _ = await asyncio.wait(get_tasks.values(),
                                         return_when=asyncio.FIRST_COMPLETED)
            for finished in done:
                # Identify which asyncio queue/task finished
                key = next(k for k, v in get_tasks.items() if v == finished)

                execution_id = finished.result()["id"]
                if key in ["newExecution", "executionEnded"]:
                    execution_number_response = await execution_number(execution_id)
                    data = execution_number_response.body.decode("utf-8")

                elif key in ["newTask", "taskEnded"]:
                    # dispatches a TaskExt object dict, augmented with :
                    #   - "parent_ui_css" key for parallel tasks
                    #     parents styling (distributed sub-DAG & split-line)
                    #   - filled-in default styling
                    #   - "taskgroups_hierarchy_list" key with ordered list
                    #     of taskgroup nesting
                    data = copy.copy(finished.result())
                    await augment_new_task(data)
                    data = json.dumps(data)
                    # print(f"{key} - {data}")

                else:
                    raise Exception(f"handling of SSE event '{key}' not implemented.")

                event_type = key

                # Replace only the finished asyncio task
                get_tasks[key] = asyncio.create_task(queues[key].get())

                uvicorn_logger.info(
                    '%s - "%s %s %s" %d',
                    f"{client_info['ip']}:{client_info['port']}",
                    "sse", f"{client_info['url']} {{ {event_type} }}", '0.0', 200
                )
                yield (
                    f"event: {event_type}\n"
                    f"data: {data}\n\n"
                )
    except asyncio.CancelledError:
        for task in get_tasks.values():
            task.cancel()
        await asyncio.gather(*get_tasks.values(), return_exceptions=True)
        # re-raise the CancelledError so it propagates
        # (ensuring the asyncio task is cancelled)
        raise
    finally:
        new_exec_subscribers.remove((queues["newExecution"], client_info))
        exec_end_subscribers.remove((queues["executionEnded"], client_info))
        print(f"execution subscribers [{len(new_exec_subscribers)}] : {new_exec_subscribers}")
        new_task_subscribers.remove((queues["newTask"], client_info))
        task_end_subscribers.remove((queues["taskEnded"], client_info))
        print(f"execution subscribers [{len(new_exec_subscribers)}] : {new_exec_subscribers}")

