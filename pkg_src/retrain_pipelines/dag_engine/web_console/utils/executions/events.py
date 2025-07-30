
import json
import asyncio
import logging
import tzlocal

from typing import TypedDict
from datetime import datetime

from ....db.model import Execution
from .executions import execution_to_html
from retrain_pipelines.utils import parse_datetime


# Global lists of subscriber queues
new_exec_subscribers = []
exec_end_subscribers = []

uvicorn_logger = logging.getLogger("uvicorn.access")
server_tz = tzlocal.get_localzone()


class ClientInfo(TypedDict):
    ip: str
    port: int
    url: str

async def new_exec_event_generator(client_info: ClientInfo):
    queue = asyncio.Queue()
    new_exec_subscribers.append((queue, client_info))
    print(f"new_exec_subscribers : {new_exec_subscribers}")
    try:
        while True:
            data = await queue.get()

            execution = Execution(data)
            data["html"] = execution_to_html(execution)

            uvicorn_logger.info(
                '%s - "%s %s %s" %d',
                f"{client_info['ip']}:{client_info['port']}",
                "sse", client_info['url'], '0.0', 200
            )
            yield f"data: {json.dumps(data)}\n\n"
    except asyncio.CancelledError:
        new_exec_subscribers.remove((queue, client_info))
        print(f"new_exec_subscribers : {new_exec_subscribers}")
        raise


class ExecutionEnd(TypedDict):
    id: int
    end_timestamp: datetime
    success: bool


async def exec_end_event_generator(client_info: ClientInfo):
    queue = asyncio.Queue()
    exec_end_subscribers.append((queue, client_info))
    print(f"exec_end_subscribers : {exec_end_subscribers}")
    try:
        while True:
            data = await queue.get()
            print(f"exec_end_event_generator - {data}")
            execution_end = ExecutionEnd(data)

            uvicorn_logger.info(
                '%s - "%s %s %s" %d',
                f"{client_info['ip']}:{client_info['port']}",
                "sse", client_info['url'], '0.0', 200
            )
            print(f"exec_end_event_generator - YIELDING : {execution_end}")
            yield f"data: {json.dumps(execution_end)}\n\n"
    except asyncio.CancelledError:
        exec_end_subscribers.remove((queue, client_info))
        print(f"exec_end_subscribers : {exec_end_subscribers}")
        raise

