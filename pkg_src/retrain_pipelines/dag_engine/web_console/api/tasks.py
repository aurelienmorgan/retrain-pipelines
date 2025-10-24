
import logging

from fasthtml.common import Request, Response

from .open_api import rt_api
from ...db.model import TaskExt
from ..utils.execution import events as execution_events
from ..utils.execution.gantt_chart import fill_defaults, \
    Style, GroupTypes


def register(app, rt, prefix=""):
    @rt_api(
        rt, url= f"{prefix}/api/v1/new_task_event",
        methods=["POST"],
        schema={
            "requestBody": {
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "integer"},
                                "exec_id": {"type": "integer"},
                                "tasktype_uuid": {
                                    "type": "string",
                                    "format": "uuid"
                                },
                                "name": {"type": "string"},
                                "is_parallel": {
                                    "type": "boolean",
                                    "default": "false"
                                },
                                "merge_func": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "docstring": {"type": "string"}
                                    },
                                    "required": ["name", "docstring"]
                                },
                                "docstring": {"type": "string"},
                                "ui_css": {
                                    "type": "object",
                                    "properties": {
                                        "color": {
                                              "type": "string",
                                              "pattern": "^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$"
                                        },
                                        "background": {
                                              "type": "string",
                                              "pattern": "^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$"
                                        },
                                        "border": {
                                              "type": "string",
                                              "pattern": "^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$"
                                        },
                                        "labelUnderlay": {
                                              "type": "string",
                                              "pattern": "^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$"
                                        }
                                    },
                                    "required": ["color", "background", "border"]
                                },
                                "rank": {
                                    "type": "array",
                                    "items": {"type": "integer"},
                                    "default": ""
                                },
                                "taskgroup_uuid": {
                                    "type": "string",
                                    "format": "uuid",
                                    "default": ""
                                },
                                "start_timestamp": {
                                    "type": "string",
                                    "format": "date-time"
                                },
                            },
                            "required": ["id", "exec_id", "tasktype_uuid",
                                         "name", "is_parallel", "ui_css",
                                         "start_timestamp"]
                        }
                    }
                }
            },
            "responses": {
                "200": {"description": "OK"},
                "422": {"description": "Invalid input"}
            }
        },
        category="Tasks"
    )
    async def post_new_task_event(
        request: Request
    ):
        """DAG-engine notifies of a new pipeline execution task."""
        data = await request.json()

        # validate posted data
        try:
            task_ext = TaskExt(data)
        except (KeyError, ValueError, TypeError) as e:
            logging.getLogger().warn(e)
            return Response(status_code=422,
                            content=f"Invalid input: {str(e)}")

        # styling, fill defaults
        task_ext_style = Style(
            task_ext.ui_css,
            labelUnderlay="#4d0066" # will be overridden
                                    # for non-standalone tasks
        )
        fill_defaults(task_ext_style, GroupTypes.NONE)
        task_ext.ui_css = task_ext_style

        # make payload serializable
        task_ext_dict = task_ext.__dict__
        del task_ext_dict["_sa_instance_state"]
        task_ext_dict = {
            (k[1:] if k.startswith('_') else k): v
            for k, v in task_ext.__dict__.items()
        }

        # dispatch 'new Task' event
        # to all subscribers on all pages that relate to it
        for q, _ in execution_events.new_task_subscribers:
            # print(f"dispatch 'new Task' event  -  {task_ext_dict}")
            await q.put(task_ext_dict)

        return Response(status_code=200)

