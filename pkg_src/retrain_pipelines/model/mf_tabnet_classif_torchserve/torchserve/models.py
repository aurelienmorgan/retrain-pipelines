import requests
from pydantic import BaseModel


class Model(BaseModel):
    modelName: str
    modelUrl: str


class ModelsResponse(BaseModel):
    models: list[Model]


def fetch_and_parse_models() -> ModelsResponse:
    # TorchServe management API
    url = "http://localhost:9081/models"
    response = requests.get(url)

    if response.status_code == 200:
        # Parse & validate response
        response_data = response.json()
        return ModelsResponse.parse_obj(response_data)
    else:
        response.raise_for_status()
        raise RuntimeError("type-checker, unreachable")


def server_has_model(model_name: str) -> bool:
    """Test whether the model has been loaded.

    (i.e. if it ever attempted to be).
    """
    try:
        models_response = fetch_and_parse_models()
        return any([model_name == model.modelName for model in models_response.models])
    except Exception as e:
        print(f"An error occurred: {e}")

    return False
