
import requests

from pydantic import BaseModel, Field
from typing import List, Optional


class Worker(BaseModel):
    id: str
    startTime: str
    status: str
    memoryUsage: int
    pid: int
    gpu: bool
    gpuUsage: str

class ModelInfo(BaseModel):
    modelName: str
    modelVersion: str
    modelUrl: str
    runtime: str
    minWorkers: int
    maxWorkers: int
    batchSize: int
    maxBatchDelay: int
    loadedAtStartup: bool
    workers: List[Worker]


def fetch_model_info(
    model_name: str,
    port: int
) -> List[ModelInfo]:
    # TorchServe management API
    api_url = f"http://localhost:{port}/models/{model_name}"

    response = requests.get(api_url)
    response.raise_for_status()

    data = response.json()

    # Validate & parse response
    model_info = [ModelInfo(**model_info)
                  for model_info in data][0]
    # print(model_info)

    return model_info


def endpoint_still_starting(
    model_name: str,
    port: int = 9081
) -> bool:
    """
    all model workers still have "STARTING" status.
    """
    """
    Side note :
    if we move too fast, the worker never has time to
    move aways from its starting idle state
    @see TorchServe warmup (loading/unloading) policy.
    """

    model_info = fetch_model_info(model_name, port)

    # print(f"endpoint_still_starting ? => {model_info.workers}")
    return all([worker.status.lower()
                in ["starting", "unloading"]
                for worker in model_info.workers])


def endpoint_is_ready(
    model_name: str,
    port: int = 9081
) -> bool:
    """
    At least model worker has "READY" status.
    """

    model_info = fetch_model_info(model_name, port)

    # Print the parsed model info
    is_ready = any(["ready" == worker.status.lower()
                    for worker in model_info.workers])
    print(f"endpoint_is_ready ? => {model_info.workers}")

    return is_ready

