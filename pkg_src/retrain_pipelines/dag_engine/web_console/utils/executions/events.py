
import json
import logging
import asyncio


# Global list of subscriber queues
new_exec_subscribers = []

uvicorn_logger = logging.getLogger("uvicorn.access")

async def new_exec_event_generator(client_info: str):
    queue = asyncio.Queue()
    new_exec_subscribers.append((queue, client_info))
    print(f"new_exec_subscribers : {new_exec_subscribers}")
    try:
        while True:
            data = await queue.get()
            uvicorn_logger.info(
                '%s - "%s %s %s" %d',
                f"{client_info['ip']}:{client_info['port']}",
                "sse", client_info['url'], '0.0', 200
            )
            print(f"YIELDING : {data}")
            yield f"data: {json.dumps(data)}\n\n"
    except asyncio.CancelledError:
        new_exec_subscribers.remove((queue, client_info))
        raise

