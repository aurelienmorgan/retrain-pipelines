from typing import Final

import requests


def endpoint_still_starting(
    model_name: str,
    port: int = 9080,
) -> bool:
    """
    Check whether a model endpoint is still starting.

    MLServer has NO worker lifecycle and
    NO partial readiness states.
    Therefore, "still starting" is defined strictly
    as:
        model_ready == False

    Implementation:
        - HTTP 200 → NOT starting
        - any other status / exception
          → still starting

    Parameters
    ----------
    model_name : str
        name of the model
    port : int
        server HTTP port

    Returns
    -------
    bool
        true if the model is not yet ready,
        false if it is ready.
    """
    url: Final[str] = f"http://localhost:{port}/v2/models/{model_name}/ready"

    try:
        response: requests.Response = requests.get(url)
        return response.status_code != 200
    except requests.RequestException:
        return True


def endpoint_is_ready(
    model_name: str,
    port: int = 9080,
) -> bool:
    """Check whether a model endpoint is ready.

    To accept inference requests.

    This function directly maps to:
        GET /v2/models/{model_name}/ready

    Semantics:
        - HTTP 200 → inference will succeed
        - anything else → inference must
                          NOT be attempted

    Parameters
    ----------
    model_name : str
        name of the model
    port : int
        server HTTP port

    Returns
    -------
    bool
        true if the model is ready,
        false otherwise.
    """
    url: Final[str] = f"http://localhost:{port}/v2/models/{model_name}/ready"

    try:
        response: requests.Response = requests.get(url)
        return response.status_code == 200
    except requests.RequestException:
        return False
