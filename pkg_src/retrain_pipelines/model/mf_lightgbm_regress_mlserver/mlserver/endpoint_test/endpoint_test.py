
import json
import logging
import requests

from typing import List, Union, \
    Annotated
from pydantic import BaseModel, \
    validator, RootModel

logger = logging.getLogger(__name__)


# Request Model
class InputTensor(BaseModel):
    """
    Single MLServer V2 input tensor.
    """
    name: str
    shape: List[int]
    datatype: str
    data: List[Union[str, int, float]]


class InferenceRequest(BaseModel):
    """
    MLServer V2 batch inference request.
    Root-level list of InputTensor objects.
    """
    inputs: List[InputTensor]

    @validator("inputs")
    def non_empty_inputs(cls, v):
        if not v:
            raise ValueError(
                "Inference request must contain at least one input tensor.")
        return v


# Response Model
class InferenceResponse(RootModel):
    """
    Single-target regression response.
    Returns one float per input row.
    """
    root: Annotated[list[float], ...]


def parse_endpoint_response(
    model_name: str,
    raw_inference_request_items: dict,  # already V2 protocol payload
    host: str = "localhost",
    port: int = 9080,
):
    # Validate request payload
    try:
        request_data = InferenceRequest.parse_obj(
                            raw_inference_request_items)
    except ValueError as e:
        logger.info(f"Request validation error: {e}")
        return None

    json_payload = json.dumps(request_data.dict())
    logger.info(f"inference request payload:: {json_payload}")

    url = f"http://{host}:{port}/v2/models/{model_name}/infer"
    headers = {"Content-Type": "application/json"}

    response = requests.post(url, headers=headers, data=json_payload)
    logger.info(f"inference raw response: {response.json()}")

    # Validate and parse the response
    parsed_response = None
    if response.status_code == 200:
        try:
            resp_json = response.json()
            outputs = resp_json.get("outputs")
            if not outputs or "data" not in outputs[0]:
                raise ValueError(
                        "Missing 'outputs' or 'data' in response.")

            data = outputs[0]["data"]
            # flatten in case of [N,1]
            flattened = [float(x[0]) if isinstance(x, list) else float(x)
                         for x in data]
            parsed_response = \
                InferenceResponse.model_validate(flattened).root
            logger.debug(
                f"validated & parsed response: {parsed_response}")
        except ValueError as e:
            logger.warning(f"Validation Error: {e}")
        except Exception as e:
            logger.warning(f"Parsing Error: {e}")
    else:
        print(f"logger : {logger}")
        logger.warning(
            f"Request failed with status code: {response.status_code}")
        logger.warning(
            f"Response Body: {response.text}")

    return parsed_response

