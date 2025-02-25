
import os
import yaml
import time


class Config:
    """
    Example yaml file for ingestion :

    ```yaml
    port: 8765

    max_seq_length: 2048
    max_new_token: 400

    base_model:
        repo_id: unsloth/Qwen2.5-1.5B
        revision: 8951671def651bbedbcdea3751f46cf35e78dfa9

    adapters:
        - name: func_caller
          path: $PWD/serving_artifacts/UnslothFuncCallFlow/1575/Unsloth/sft_model
        - name: func_caller_wip
          repo_id: retrain-pipelines/function_caller_wip
          revision: 588c46432037cec34a866f94ccd1ddeebc321c37
    ```
    """
    print(f"Start Time : {time.strftime('%H:%M:%S')}")

    with open("litserve_serverconfig.yaml", "r") as file:
        yaml_config = yaml.safe_load(file)

    BASE_MODEL_PATH = (
        os.path.realpath(os.path.expanduser(BASE_MODEL_PATH))
        if (BASE_MODEL_PATH:=yaml_config["base_model"].get("path")) is not None
        else None)
    if BASE_MODEL_PATH is not None:
        print(f"BASE_MODEL_PATH : {BASE_MODEL_PATH}")
    else:
        BASE_MODEL_REPO_ID = yaml_config["base_model"].get("repo_id") \
                             or "unsloth/Qwen2.5-1.5B"
        BASE_MODEL_REVISION = yaml_config["base_model"].get("revision") \
                             or None
        print(f"BASE_MODEL_REPO_ID : {BASE_MODEL_REPO_ID}, " +
              f"BASE_MODEL_REVISION : {BASE_MODEL_REVISION}")
        HF_HUB_CACHE = os.path.realpath(os.path.expanduser(
            os.getenv(
                "HF_HUB_CACHE",
                os.path.join(os.getenv("HF_HOME",
                                       "~/.cache/huggingface"),
                             "hub")
            )))
        print(f"HF_HUB_CACHE : {HF_HUB_CACHE}")

    adapters = {}
    if (yaml_adapters:=yaml_config.get("adapters")) is not None:
        for adapter in yaml_adapters:
            adapter_path = adapter.get("path")
            if (adapter_path) is not None:
                adapter_path = os.path.realpath(os.path.expanduser(
                    adapter_path))
                adapters[adapter["name"]] = {
                    "path": adapter_path}
            else:
                adapters[adapter["name"]] = {
                    "repo_id": adapter["repo_id"],
                    "revision": adapter.get("revision")
                }
    print(f"config_adapters ({len(adapters)}):\n\t{adapters}")

    MAX_SEQ_LENGTH = (
        int(MAX_SEQ_LENGTH)
        if (MAX_SEQ_LENGTH:=yaml_config.get("max_seq_length")) is not None
        else None)
    MAX_NEW_TOKENS = (
        int(MAX_NEW_TOKENS)
        if (MAX_NEW_TOKENS:=yaml_config.get("max_new_token")) is not None
        else None)

    PORT = (
        int(PORT)
        if (PORT:=yaml_config.get("port")) is not None
        else 8000)
    print(f"PORT : {PORT}")

