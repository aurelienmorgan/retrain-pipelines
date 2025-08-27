
from fasthtml.common import Request, Response

from .open_api import rt_api
from ...db.model import Execution, ExecutionExt
from ....utils import hex_to_rgba

from ..utils.executions import events as executions_events
from ..utils.execution import events as execution_events


def register(app, rt, prefix=""):
    @rt_api(
        rt, url= f"{prefix}/api/v1/new_execution_event",
        methods=["POST"],
        schema={
            "requestBody": {
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "integer"},
                                "name": {"type": "string"},
                                "username": {"type": "string"},
                                "start_timestamp": {
                                    "type": "string",
                                    "format": "date-time"
                                }
                            },
                            "required": ["id", "name", "username",
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
        category="Executions"
    )
    async def post_new_execution_event(
        request: Request
    ):
        """DAG-engine notifies of a new pipeline execution."""
        data = await request.json()

        # validate posted data
        try:
            execution = Execution(data)
        except (KeyError, ValueError, TypeError) as e:
            logging.getLogger().warn(e)
            return Response(status_code=422,
                            content=f"Invalid input: {str(e)}")

        # dispatch 'new Execution' event
        # to all subscribers on all pages that relate to it
        for q, _ in executions_events.new_exec_subscribers:
            await q.put(data)
        for q, _ in execution_events.new_exec_subscribers:
            await q.put(data)

        return Response(status_code=200)


    @rt_api(
        rt, url= f"{prefix}/api/v1/execution_end_event",
        methods=["POST"],
        schema={
            "requestBody": {
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "integer"},
                                "name": {"type": "string"},
                                "username": {"type": "string"},
                                "start_timestamp": {
                                    "type": "string",
                                    "format": "date-time"
                                },
                                "end_timestamp": {
                                    "type": "string",
                                    "format": "date-time"
                                },
                                "success": {"type": "boolean"}
                            },
                            "required": ["id", "name", "username",
                                         "start_timestamp",
                                         "end_timestamp", "success"]
                        }
                    }
                }
            },
            "responses": {
                "200": {"description": "OK"},
                "422": {"description": "Invalid input"}
            }
        },
        category="Executions"
    )
    async def post_execution_ended_event(
        request: Request
    ):
        """DAG-engine notifies of a pipeline execution end."""
        data = await request.json()

        # validate posted data
        try:
            execution_ext = ExecutionExt(**data)
        except (KeyError, ValueError, TypeError) as e:
            logging.getLogger().warn(e)
            return Response(status_code=422,
                            content=f"Invalid input: {str(e)}")

        # dispatch 'Execution ended' event
        # to all subscribers on all pages that relate to it
        for q, _ in executions_events.exec_end_subscribers:
            await q.put(data)
        for q, _ in execution_events.exec_end_subscribers:
            await q.put(data)

        return Response(status_code=200)

