
import os
import json

from ast import literal_eval

from jinja2 import Environment, FileSystemLoader

from datasets import DatasetDict
from huggingface_hub import whoami

from retrain_pipelines import __version__

from retrain_pipelines.dataset.hf_utils import \
        get_latest_README_commit, \
        get_arxiv_codes, get_license_label, \
        get_pretty_name,


def _model_readme_params(
    version_label: str,
    utc_timestamp_str: str,
    mf_flow_name: str,
    mf_run_id: str,
    engine:str = "cpu"
) -> dict:
    """
    Populates the params dict to be used
    to customize the model jinja template.

    Built on metadata from the base model.

    Params:
        - version_label (str):
            typical `retrain-pipelines`
            version label are of format "major.minor"
        - utc_timestamp_str (str):
            timestampt for the new dataset version.
        - mf_flow_name (str)
        - mf_run_id (str)
        - engine (str):
            Polars engine (can be "cpu", "gpu"..)

    Results:
        - (dict)
    """

    pretty_name = "retrain-pipelines Function Caller"

    base_commit_hash, base_commit_utc_date_str = \
        get_latest_README_commit(
            repo_id=hf_dataset_dict["repo_id"],
            target_commit_hash=hf_dataset_dict["commit_hash"]
        )

    base_pretty_name = get_pretty_name(
        repo_id=hf_dataset_dict["repo_id"],
        commit_hash=base_commit_hash
    )

    base_arxiv_codes = get_arxiv_codes(
        repo_id=hf_dataset_dict["repo_id"],
        commit_hash=base_commit_hash
    )

    base_license_label = get_license_label(
        repo_id=hf_dataset_dict["repo_id"],
        commit_hash=base_commit_hash
    )
    if not base_license_label:
        base_license_label = "unknown"

    return {
            "new_version_label": version_label,
            "utc_timestamp": utc_timestamp_str,

            "pretty_name": pretty_name,

            "main_repo_id": \
                hf_dataset_dict["repo_id"],
            "enrich_repo_id": \
                hf_enrich_dataset_dict["repo_id"],

            "main_commit_hash": main_commit_hash,
            "enrich_commit_hash": enrich_commit_hash,

            "main_commit_utc_date_str": \
                main_commit_utc_date_str,
            "enrich_commit_utc_date_str": \
                enrich_commit_utc_date_str,

            "main_pretty_name": main_pretty_name,
            "enrich_pretty_name": enrich_pretty_name,

            "size_category": size_category,

            "main_arxiv_codes": main_arxiv_codes,
            "enrich_arxiv_codes": enrich_arxiv_codes,

            "main_license_label": main_license_label,
            "enrich_license_label": enrich_license_label,

            "main_format_description" : main_format_description,

            "__version__": __version__,
            "run_user": whoami()["name"],
            "mf_flow_name": mf_flow_name,
            "mf_run_id": mf_run_id
        }
    

def get_dataset_readme_content(
    template_folder: str,
    hf_dataset_dict: dict,
    hf_enrich_dataset_dict: dict,
    dataset_dict: DatasetDict,
    version_label: str,
    utc_timestamp_str: str,
    mf_flow_name: str,
    mf_run_id: str,
    engine:str = "cpu"
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
        - hf_dataset_dict (dict):
            - repo_id
            - commit_hash
            - commit_utc_date_str
            - lazy_df
        - hf_enrich_dataset_dict (dict)
            - repo_id
            - commit_hash
            - commit_utc_date_str
        - dataset_dict (DatasetDict):
            the dataset version to be pushed
            to the HF hub.
        - version_label (str):
            typical `retrain-pipelines`
            version label are of format "major.minor"
        - utc_timestamp_str (str):
            timestampt for the new dataset version.
        - mf_flow_name (str)
        - mf_run_id (str)
        - engine (str):
            Polars engine (can be "cpu", gpu"..)

    Results:
        - (str)
    """

    params = _dataset_readme_params(
        hf_dataset_dict=hf_dataset_dict,
        hf_enrich_dataset_dict=hf_enrich_dataset_dict,
        dataset_dict=dataset_dict,
        version_label=version_label,
        utc_timestamp_str=utc_timestamp_str,
        mf_flow_name=mf_flow_name,
        mf_run_id=mf_run_id
    )

    env = Environment(loader=FileSystemLoader(template_folder))
    template = env.get_template("dataset_readme_template.md")
    readme_content = template.render(params)

    return readme_content

