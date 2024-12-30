
import os

from retrain_pipelines import __version__
from retrain_pipelines.utils.hf_utils import \
    local_repo_folder_to_hub


def push_model_version_to_hub(
    repo_id: str,
    version_label: str,
    timestamp_str: str,
    model_dir: str,
    model_readme_content: str,
    hf_token: str = None,
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

    model_version_commit_hash = \
        local_repo_folder_to_hub(
            repo_id=repo_id,
            local_folder=model_dir,
            commit_message=commit_message,
            repo_type="model",
            hf_token=hf_token
        )

    return model_version_commit_hash

