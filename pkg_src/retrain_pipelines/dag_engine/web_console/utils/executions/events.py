
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


###### clean shutdown on hard thread kill ######################################

# Sentinel pushed into every subscriber queue on server shutdown,
# so each generator exits cleanly and its finally-block fires.
_SHUTDOWN = object()

# The asyncio event loop the server runs on.
# Set by the first generator that starts; cleared on reset.
_server_loop = None


def notify_server_shutdown():
    """Push the shutdown sentinel into every active subscriber queue.

    Called from the main (notebook) thread before the server thread
    is killed, giving generators time to exit and run their finally-blocks
    (which remove them from the subscriber lists).
    """
    global _server_loop
    if not (_server_loop is None or _server_loop.is_closed()):
        for queue, _ in (
            list(new_exec_subscribers) +
            list(exec_end_subscribers)
        ):
            try:
                _server_loop.call_soon_threadsafe(
                    queue.put_nowait, _SHUTDOWN)
            except RuntimeError:
                pass
    else:
        # No live loop — just wipe the lists directly as a safety net.
        pass
    new_exec_subscribers.clear()
    exec_end_subscribers.clear()


def reset_for_restart():
    """Wipe any stale subscriber state before a new server start.

    Safety net for the case where notify_server_shutdown() didn't manage
    to drain everything (e.g. hard kernel restart).
    """
    global _server_loop
    _server_loop = None
    new_exec_subscribers.clear()
    exec_end_subscribers.clear()


################################################################################


async def multiplexed_event_generator(client_info: ClientInfo):
    global new_exec_subscribers, exec_end_subscribers, \
           _server_loop

    if _server_loop is None:
        _server_loop = asyncio.get_event_loop()

    queues = {
        "newExecution": asyncio.Queue(),
        "executionEnded": asyncio.Queue()
    }

    new_exec_subscribers.append((queues["newExecution"], client_info))
    exec_end_subscribers.append((queues["executionEnded"], client_info))
    print(f"executions subscribers [{len(new_exec_subscribers)}] : " +
          f"{new_exec_subscribers}")

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
                result = finished.result()

                # Server shutdown signal — exit cleanly so finally runs.
                if result is _SHUTDOWN:
                    for task in get_tasks.values():
                        task.cancel()
                    await asyncio.gather(*get_tasks.values(),
                                         return_exceptions=True)
                    return

                data = copy.copy(result)
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
        try:
            new_exec_subscribers.remove((queues["newExecution"], client_info))
        except Exception:
            pass
        try:
            exec_end_subscribers.remove((queues["executionEnded"], client_info))
        except Exception:
            pass
        print(f"executions subscribers [{len(new_exec_subscribers)}] : " +
              f"{new_exec_subscribers}")

