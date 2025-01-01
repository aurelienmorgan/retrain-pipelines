
import os
from datetime import datetime

from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError

from retrain_pipelines import __version__
from retrain_pipelines.utils.hf_utils import \
    local_repo_folder_to_hub


def push_model_version_to_hub(
    repo_id: str,
    model_version_blessed: bool,
    version_label: str,
    timestamp_str: str,
    model_dir: str,
    model_readme_content: str,
    hf_token: str = os.getenv("HF_TOKEN", None)
) -> str:
    """
    Loads locally-serialized model safetensor
    and tokenizer.
    Includes `retrain-pipelines` README.

    Uploaded model version superseeds entirely
    any existing version (any previous file
    not anymore present is excluded from
    new remote model snapshot).

    Params:
        - repo_id (str):
            Path to the HuggingFace model version
            (is created if needed and if authorized).
        - model_version_blessed (bool):
            Whether the model version is blessed ;
            dictates the branch on which to
            publish it on the HF hub.
        - version_label (str):
            value associated to the version
            to be published on the HF hub.
        - timestamp_str (str):
            value associated to the version
            to be published on the HF hub
        - model_dir (str):
            Path to the serialized
            new version to be pushed.
        - model_readme_content (str):
            The full content (yaml header + body)
            of the 'README.md' to be pushed
            alongside the datafiles.
        - hf_token (Optional, str):
            "create on namespace" permission required.

    Results:
        - (str):
            commit_hash on the HF hub
            for the new model version
    """

    with open(os.path.join(model_dir, "README.md"),
              "w") as f:
        f.write(model_readme_content)

    commit_message = \
        f"v{version_label} - {timestamp_str} - " + \
        f"retrain-pipelines v{__version__} - "+ \
        "Upload model and tokenizer with README."
    print(commit_message)

    branch_name=(
        "main" if model_version_blessed
        else "retrain-pipelines_not-blessed"
    )

    model_version_commit_hash = \
        local_repo_folder_to_hub(
            repo_id=repo_id,
            branch_name=branch_name,
            local_folder=model_dir,
            commit_message=commit_message,
            repo_type="model",
            hf_token=hf_token
        )

    return model_version_commit_hash


def current_blessed_model_version_dict(
    repo_id: str,
    hf_token: str = os.getenv("HF_TOKEN", None)
) -> dict:
    """
    None if no prior model version
    exists on the HF Hub.

    Params:
        - repo_id (str):
            Path to the HuggingFace model.
        - hf_token (Optional, str):
            "create on namespace" permission required.

    Results:
        - (dict):
            - mf_run_id (str)
            - commit_hash (str)
            - version_label (str)
            - commit_datetime (datetime)
            - perf_metrics (dict)
    """

    try:
        model_info = HfApi().repo_info(
            repo_id=repo_id,
            revision="main",
            token=hf_token
        )
    except RepositoryNotFoundError as err:
        print(f"repo {repo_id} not found.\n" +
              "If you are trying to access a " +
              "private or gated repo, " +
              "make sure you are authenticated " +
              "and your credentials allow it.",
              file=sys.stderr)
        print(err, file=sys.stderr)
        return None

    if model_info:
        model_version_card_data = \
            model_info.cardData
        commit_datetime = datetime.strptime(
            model_version_card_data["timestamp"],
            "%Y%m%d_%H%M%S%f_%Z")

        eval_results_dict = {
                m['type']: m['value']
                for m in model_info \
                            .model_index[0]['results'][0]['metrics']
            }

        return {
            "mf_run_id": model_version_card_data["mf_run_id"],
            "commit_hash": model_info.sha,
            "version_label": model_version_card_data["version"],
            "commit_datetime": commit_datetime,
            "perf_metrics": eval_results_dict
        }

    return None

