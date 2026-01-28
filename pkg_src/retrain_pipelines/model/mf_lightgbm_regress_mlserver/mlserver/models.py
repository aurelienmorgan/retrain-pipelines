
import requests

from typing import Final


def server_has_model(
    model_name: str,
    port: int = 9080,
) -> bool:
    """
    Determine whether MLServer is aware of a model.

    IMPORTANT SEMANTICS (READ CAREFULLY):
    MLServer does NOT provide:
        - a guaranteed model registry endpoint
        - a model listing API
        - a "loaded vs not loaded" distinction
    The ONLY stable, spec-backed way to ask
    whether a model exists is to probe the model
    readiness endpoint:
        GET /v2/models/{model_name}/ready

    Status code interpretation:
        - 200 → model exists AND is ready
        - 503 → model exists BUT is still loading
        - 404 → model does NOT exist

    Params:
        - model_name (str):
            name of the model as configured
            in MLServer.
        - port (int):
            server HTTP port.

    Results:
        - bool
            true if model exists
            (ready or loading)
            false if model is unknown
            or unreachable.
    """
    url: Final[str] = \
        f"http://localhost:{port}/v2/models/{model_name}/ready"

    try:
        response: requests.Response = requests.get(url)
        return response.status_code in (200, 503)
    except requests.RequestException:
        return False

