
import numpy as np

import requests
import json
import sys

import docker

from retrain_pipelines.utils.docker import \
        print_container_log_tail


from pydantic import BaseModel, conlist, validator
from typing import List, Union


# Request models
class RawFeaturesRecord(BaseModel):
    # purposely very permissive,
    # for adaptability to all tabular use-cases
    data: List[Union[str, int, float]]

    class Config:
        # only way to avoid all list items
        # be converted to string type.
        smart_union = True

class InferenceRequest(BaseModel):
    __root__: List[RawFeaturesRecord]


# Response Model
class InferenceResponse(BaseModel):
    # SINGLE TARGET-VARIABLE CLASSIFICATION PROBLEM
    # response is a SINGLE string (i.e. a class label).
    # note that we accept batch requests (per single http-request)
    # so responses shall be lists of at least size 1
    __root__: conlist(str, min_items=1)


def parse_endpoint_response(
    model_name: str,
    raw_inference_request_items: list
):
    # Create the request payload using Pydantic models
    request_data = InferenceRequest(__root__=[
        RawFeaturesRecord(data=item)
        for item in raw_inference_request_items])
    json_payload = request_data.json()
    print("inference request payload:", json_payload)

    url = f"http://localhost:9080/predictions/{model_name}"
    headers = {'Content-Type': 'application/json'}

    response = requests.post(url, headers=headers, data=json_payload)

    # Validate and parse the response
    parsed_response = None
    if response.status_code == 200:
        try:
            parsed_response = \
                InferenceResponse.parse_raw(response.text).__root__
            print(f"Response: {parsed_response}")
        except ValueError as e:
            print(f"Validation Error: {e}")
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

