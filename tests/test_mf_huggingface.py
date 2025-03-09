
import os
import json

from huggingface_hub import HfApi, repo_exists
"""
Top-leavel env for pytest has
dependency with 'huggingface_hub' !
"""
from retrain_pipelines.utils.hf_utils import \
        create_repo_if_not_exists
from retrain_pipelines.utils.pytest_utils import \
        get_venv
from retrain_pipelines.utils import \
        as_env_var
from retrain_pipelines.local_launcher import \
        retrain_pipelines_local


def delete_repo_safe_if_exists(
    hf_api: HfApi,
    repo_id: str,
    repo_type: str = "model",
    hf_token: str = os.getenv("HF_TOKEN", None)
) -> None:
    """
    Params:
        - hf_api (HfApi):
            An instanciated FH api object.
        - repo_id (str):
        - repo_type (str):
        - hf_token (str):
    """
    api = HfApi()
    try:
        if repo_exists(repo_id=repo_id,
                       repo_type=repo_type,
                       token=hf_token):
            api.delete_repo(repo_id=repo_id,
                            repo_type=repo_type)
            print(f"Deleted {repo_type} repo : {repo_id}")
    except Exception as e:
        print(f"Error handling repository {repo_id}: {e}")


##################################################
#        Metaflow retraining pipelines for       #
#            Hugging Face integration            #
##################################################

def test_mf_unsloth_func_call_litserve():

    # assumes the "requirements.txt" from the subdir
    # of the herein "sample pipeline"
    # are installed in an env named "metaflow_unsloth_venv"
    # (would it be through conda or venv)
    env = get_venv(virtual_env_name="metaflow_unsloth_venv")

    hf_api = HfApi()
    hf_token = os.getenv("HF_TOKEN", None)
    potential_repo_owners = ["retrain-pipelines",
                             hf_api.whoami()["name"]]

    pytest_dataset_repo_id = None
    pytest_dataset_repo_shortname = "unsloth_litserve_ds_pytest"
    for owner in potential_repo_owners:
        potential_repo_id = f"{owner}/{pytest_dataset_repo_shortname}"
        if create_repo_if_not_exists(
            hf_api=hf_api,
            repo_id=potential_repo_id,
            repo_type="dataset",
            hf_token=hf_token
        ):
            pytest_dataset_repo_id = potential_repo_id
            break
    assert pytest_dataset_repo_id, f"Failed to create dataset repo " + \
                                   f"\"{pytest_dataset_repo_shortname}\"."
    print(f"Using dataset repo : {pytest_dataset_repo_id}")

    pytest_model_repo_id = None
    pytest_model_repo_shortname = "unsloth_litserve_pytest"
    for owner in potential_repo_owners:
        potential_repo_id = f"{owner}/{pytest_model_repo_shortname}"
        if create_repo_if_not_exists(
            hf_api=hf_api,
            repo_id=potential_repo_id,
            repo_type="model",
            hf_token=hf_token
        ):
            pytest_model_repo_id = potential_repo_id
            break
    assert pytest_model_repo_id, f"Failed to create model repo " + \
                                 f"\"{pytest_model_repo_shortname}\"."
    print(f"Using model repo : {pytest_model_repo_id}")

    # actual pipeline run starts here

    env["HF_TOKEN"] = os.getenv("HF_TOKEN", None)

    cpt_training_args = {
        "records_cap": 32, # limit on training records used
        "max_steps": 1,
        "warmup_steps": 0}
    as_env_var(cpt_training_args,
               "cpt_training_args",
               env=env)

    sft_training_args = {
        "records_cap": 32, # limit on training records used
        "max_steps": 1,
        "warmup_steps": 0}
    as_env_var(sft_training_args,
               "sft_training_args",
               env=env)

    command = [
        os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "sample_pipelines", "Unsloth_Qwen_FuncCall",
            "retraining_pipeline.py"
        ), "run",
        "--dataset_repo_id", pytest_dataset_repo_id, \
        "--cpt_training_args", "{cpt_training_args}",
        "--sft_training_args", "{sft_training_args}",
        "--model_repo_id", pytest_model_repo_id \
    ]

    success = retrain_pipelines_local(
        command = " ".join(command),
        env=env
    )

    if pytest_dataset_repo_id:
        delete_repo_safe_if_exists(hf_api=hf_api,
                                   repo_id=pytest_dataset_repo_id,
                                   repo_type="dataset",
                                   hf_token=hf_token)
    if pytest_model_repo_id:
        delete_repo_safe_if_exists(hf_api=hf_api,
                                   repo_id=pytest_model_repo_id,
                                   repo_type="model",
                                   hf_token=hf_token)

    assert success, "retraining pipeline failed."

