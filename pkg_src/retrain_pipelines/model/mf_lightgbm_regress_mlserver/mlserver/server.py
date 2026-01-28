
import time
import requests

from typing import Final


def await_server_ready(
    retries: int = 10,
    delay: int = 2,
    port: int = 9080
) -> bool:
    """
    Block until the MLServer HTTP process
    reports itself as ready.

    Checking the MLServer V2 health endpoint:
        GET /v2/health/ready

    Semantics:
        - HTTP 200 → MLServer event loop is up
          and accepting requests
        - Any other status / exception
          → server not ready yet

    Params:
        - retries (int):
            number of attempts before giving up.
        - delay (int):
            delay (seconds) between attempts.
        - port (int):
            server HTTP port.

    Results:
        - (bool):
            true if the server became ready
            within the retry window.
            false otherwise.
    """
    url: Final[str] = f"http://localhost:{port}/v2/health/ready"

    for _ in range(retries):
        try:
            response: requests.Response = requests.get(url)
            if response.status_code == 200:
                return True
        except requests.RequestException:
            pass

        time.sleep(delay)

    print(f"Server did not become ready after {retries*delay} seconds.")

    return False

