import time

import requests


def await_server_ready(retries: int = 10, delay: int = 2, port: int = 9080) -> bool:
    """Await server readiness.

    Parameters
    ----------
    retries : int
        number of attempts before giving up.
    delay : int
        delay (seconds) between attempts.
    port : int
        server HTTP port.

    Returns
    -------
    bool
        true if the server became ready
        within the retry window.
        false otherwise.
    """
    url = f"http://localhost:{port}/ping"
    for _ in range(retries):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                print("Server is ready!")
                return True
        except requests.RequestException as e:
            print(f"Request failed: {e}")

        print(f"Retrying in {delay} seconds...")
        time.sleep(delay)

    print(f"Server did not become ready after {retries * delay} seconds.")

    return False
