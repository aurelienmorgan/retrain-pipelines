
import os
import json

from ast import literal_eval
from datetime import datetime

from jinja2 import Environment, FileSystemLoader

from datasets import DatasetDict
from huggingface_hub import whoami

from retrain_pipelines import __version__

from retrain_pipelines.utils.hf_utils import \
        get_latest_README_commit, get_arxiv_codes, \
        get_license_label, get_pretty_name


def _model_readme_params(
    base_model_dict: dict,
    training_dataset_dict: dict,
    version_label: str,
    commit_datetime: datetime,
    mf_flow_name: str,
    mf_run_id: str,
) -> dict:
    """
    Populates the params dict to be used
    to customize the model jinja template.

    Built on metadata from the base model.

    Params:
        - base_model_dict (dict)
        - training_dataset_dict (dict):
            - repo_id
            - version_label
            - commit_hash
            - commit_datetime
        - version_label (str):
            typical `retrain-pipelines`
            version label are of format "major.minor"
        - commit_datetime (datetime):
            timestamp for the new model version.
        - mf_flow_name (str)
        - mf_run_id (str)

    Results:
        - (dict)
    """

    pretty_name = "retrain-pipelines Function Caller"

    base_model_commit_hash, base_model_commit_datetime = \
        get_latest_README_commit(
            repo_id=base_model_dict["repo_id"],
            target_commit_hash=base_model_dict["commit_hash"],
            repo_type="model"
        )

    base_model_pretty_name = get_pretty_name(
        repo_id=base_model_dict["repo_id"],
        repo_type="model",
        commit_hash=base_model_commit_hash
    )

    base_model_arxiv_codes = get_arxiv_codes(
        repo_id=base_model_dict["repo_id"],
        repo_type="model",
        commit_hash=base_model_commit_hash
    )

    base_model_license_label = get_license_label(
        repo_id=base_model_dict["repo_id"],
        repo_type="model",
        commit_hash=base_model_commit_hash
    )
    if not base_model_license_label:
        base_model_license_label = "unknown"

    return {
            "new_version_label": version_label,
            "commit_datetime": commit_datetime,

            "pretty_name": pretty_name,

            "dataset_repo_id": \
                training_dataset_dict["repo_id"],
            "dataset_version_label": \
                training_dataset_dict["version_label"],
            "dataset_commit_hash": \
                training_dataset_dict["commit_hash"],
            "dataset_commit_datetime": \
                training_dataset_dict["commit_datetime"],

            "base_model_repo_id": base_model_dict["repo_id"],
            "base_model_pretty_name": base_model_pretty_name,
            "base_model_commit_hash": base_model_commit_hash,
            "base_model_commit_datetime": base_model_commit_datetime,
            "base_model_arxiv_codes": base_model_arxiv_codes,
            "base_model_license_label": base_model_license_label,

            "__version__": __version__,
            "run_user": whoami()["name"],
            "mf_flow_name": mf_flow_name,
            "mf_run_id": mf_run_id
        }
    

def get_model_readme_content(
    template_folder: str,

    base_model_dict: dict,
    training_dataset_dict: dict,

    version_label: str,
    commit_datetime: datetime,

    mf_flow_name: str,
    mf_run_id: str,
) -> str:
    """

    Note: the only known way to programmatically
    add "arxiv codes" to a repo on the HF Hub
    is by including an hyperplink
    (to HF 'papers' or to arxiv. In any event,
     the paper in question must be 'uploaded'
     on the hub for this to work)
    in the body of the README.

    Params:
        - template_folder (str)
        - base_model_dict (dict)
            - repo_id
            - commit_hash
        - training_dataset_dict (dict)
            - repo_id
            - commit_hash
            - commit_datetime
        - version_label (str):
            typical `retrain-pipelines`
            version label are of format "major.minor"
        - commit_datetime (datetime):
            timestamp for the new dataset version.
        - mf_flow_name (str)
        - mf_run_id (str)

    Results:
        - (str)
    """

    params = _model_readme_params(
        base_model_dict=base_model_dict,
        training_dataset_dict=training_dataset_dict,
        version_label=version_label,
        commit_datetime=commit_datetime,
        mf_flow_name=mf_flow_name,
        mf_run_id=mf_run_id
    )

    env = Environment(loader=FileSystemLoader(template_folder))
    template = env.get_template("model_readme_template.md")
    readme_content = template.render(params)

    return readme_content

