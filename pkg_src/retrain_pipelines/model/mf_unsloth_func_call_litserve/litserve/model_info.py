
import re
import time

import docker

import requests


def endpoint_started(
    container_name: str,
    port: int = 8000,
    timeout: int = 10*60
) -> bool:
    """
    Whether or not endpoint service
    started successfully.

    Params:
        - container_name (str):
        - port (int):
        - timeout (int):
            In seconds.
            Note that too short of a value may
            infer with model (and adapter[s])
            weight loading in mem.

    Results:
        - (bool)
    """
    start_time = time.time()
    print("##### local docker container log tail BEGIN :")

    container = docker.from_env().containers.list(
        all=True, filters={"name": container_name})[0]

    pattern = r"^INFO:\s*Application startup complete\.$"

    logs_generator = container.logs(timestamps=False, tail="all",
                                    stream=True)

    for log_line in logs_generator:
        log_line = log_line.decode('utf-8').strip()
        print(log_line, flush=True)

        if bool(re.match(pattern, log_line)):
            print("##### local docker container log tail END.",
                  flush=True)
            return True

        if time.time() - start_time > timeout:
            print("##### local docker container log tail END." +
                  f"\t-\t {timeout}s TIMEOUT REACHED !",
                  flush=True)
            return False

    print("##### local docker container log tail END." +
          f"\t-\t CONTAINER {container.id[:12]} DIED !",
          flush=True)
    return False


def endpoint_is_ready(
    port: int = 8000
) -> bool:
    """

    Params:
        - port (int):

    Results:
        - (bool)
    """

    # LitServe management API
    api_url = f"http://localhost:{port}/status"

    response = requests.get(api_url)

    return 200 == response.status_code and \
           "ok" == response.text.lower()
