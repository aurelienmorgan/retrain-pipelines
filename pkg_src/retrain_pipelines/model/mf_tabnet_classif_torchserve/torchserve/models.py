
import requests

from pydantic import BaseModel, parse_obj_as
from typing import List, Optional


class Model(BaseModel):
    modelName: str
    modelUrl: str

class ModelsResponse(BaseModel):
    models: List[Model]


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


def server_has_model(
    model_name: str
) -> bool:
    """
    Whether or not the model has been loaded
    (ever attempted to be).
    """

    try:
        models_response = fetch_and_parse_models()
        return any([model_name == model.modelName
                    for model in models_response.models])
    except Exception as e:
        print(f"An error occurred: {e}")

    return False

