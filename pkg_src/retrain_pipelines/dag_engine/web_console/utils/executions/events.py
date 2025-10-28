
import json
import copy
import asyncio
import logging
import tzlocal

from datetime import datetime

from ....db.model import Execution, ExecutionExt
from .. import ClientInfo
from .executions import execution_to_html


# Global lists of subscriber queues
new_exec_subscribers = []
exec_end_subscribers = []

uvicorn_logger = logging.getLogger("uvicorn.access")


async def multiplexed_event_generator(client_info: ClientInfo):
    queues = {
        "newExecution": asyncio.Queue(),
        "executionEnded": asyncio.Queue()
    }

    new_exec_subscribers.append((queues["newExecution"], client_info))
    exec_end_subscribers.append((queues["executionEnded"], client_info))
    print(f"executions subscribers [{len(new_exec_subscribers)}] : {new_exec_subscribers}")

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
                data = copy.copy(finished.result())
                if key == "newExecution":
                    event_type = "newExecution"
                    execution = Execution(data)
                    data["html"] = execution_to_html(execution)
                else:
                    event_type = "executionEnded"
                    execution_ext = ExecutionExt(**data)
                    data = execution_ext.to_dict()
                    data["html"] = execution_to_html(execution_ext)

                # Replace only the finished asyncio task
                get_tasks[key] = asyncio.create_task(queues[key].get())

                uvicorn_logger.info(
                    '%s - "%s %s %s" %d',
                    f"{client_info['ip']}:{client_info['port']}",
                    "sse", f"{client_info['url']} {{ {event_type} }}", '0.0', 200
                )
                yield (
                    f"event: {event_type}\n"
                    f"data: {json.dumps(data)}\n\n"
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
        print(f"executions subscribers [{len(new_exec_subscribers)}] : {new_exec_subscribers}")

