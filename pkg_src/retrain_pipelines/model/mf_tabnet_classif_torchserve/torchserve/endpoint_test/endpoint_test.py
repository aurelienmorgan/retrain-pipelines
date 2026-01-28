
import numpy as np

import requests
import json
import sys

from typing import List, Union, \
    Annotated
from pydantic import BaseModel, \
    validator, RootModel

from retrain_pipelines.utils.docker import \
        print_container_log_tail


# Request Model
class RawFeaturesRecord(BaseModel):
    # purposely very permissive,
    # for adaptability to all tabular use-cases
    data: List[Union[str, int, float]]


class InferenceRequest(RootModel):
    root: Annotated[List[RawFeaturesRecord], ...]


# Response Model
class InferenceResponse(RootModel):
    """
    SINGLE TARGET-VARIABLE CLASSIFICATION PROBLEM
    response is a SINGLE string (i.e. a class label).
    note that we accept batch requests (per single http-request)
    so responses shall be lists of at least size 1
    """
    root: Annotated[list[str], ...]


def parse_endpoint_response(
    model_name: str,
    raw_inference_request_items: list,
    port: int = 9080
):
    # Create the request payload using Pydantic models
    request_data = InferenceRequest.model_validate([
                        RawFeaturesRecord(data=item)
                        for item in raw_inference_request_items])
    json_payload = request_data.json()
    print("inference request payload:", json_payload)

    url = f"http://localhost:{port}/predictions/{model_name}"
    headers = {'Content-Type': 'application/json'}

    response = requests.post(url, headers=headers, data=json_payload)

    # Validate and parse the response
    parsed_response = None
    if response.status_code == 200:
        try:
            parsed_response = \
                InferenceResponse.model_validate_json(response.text).root
            print(f"Response: {parsed_response}")
        except ValueError as e:
            print(f"Validation Error: {e}")
    else:
        print(f"Request failed with status code: {response.status_code}")
        print(f"Response Body: {response.text}")

        print_container_log_tail(
            container_name=model_name,
            tail_length=20
        )

    return parsed_response

