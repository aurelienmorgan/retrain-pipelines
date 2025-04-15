
import os
import json

from ast import literal_eval
from datetime import datetime

from urllib.parse import quote as url_encode
from jinja2 import Environment, FileSystemLoader

from datasets import DatasetDict
from huggingface_hub import whoami

from retrain_pipelines import __version__
from retrain_pipelines.dataset.hf_utils import \
        get_size_category, dataset_dict_to_config_str
from retrain_pipelines.utils.hf_utils import \
        get_latest_README_commit, get_arxiv_codes, \
        get_license_label, get_pretty_name


def _dataset_readme_params(
    hf_dataset_dict: dict,
    hf_enrich_dataset_dict: dict,
    dataset_dict: DatasetDict,
    augmentation_rate: float,
    enrichment_rate: float,
    version_label: str,
    commit_datetime: datetime,
    mf_flow_name: str,
    mf_run_id: str,
    engine:str = "cpu"
) -> dict:
    """
    Populates the params dict to be used
    to customize the dataset jinja template.

    Built on metadata from the source datasets.

    Params:
        - hf_dataset_dict (dict):
            - repo_id
            - commit_hash
            - commit_datetime
            - lazy_df
        - hf_enrich_dataset_dict (dict)
            - repo_id
            - commit_hash
            - commit_datetime
        - dataset_dict (DatasetDict):
            the dataset version to be pushed
            to the HF hub.
        - augmentation_rate (float):
            - proportion of the original dataset
              that has been augmented.
        - enrichment_rate (float):
            - proportion of the original dataset
              that has been added from
              the enrichment dataset.
        - version_label (str):
            typical `retrain-pipelines`
            version label are of format "major.minor"
        - commit_datetime (datetime):
            timestamp for the new dataset version.
        - mf_flow_name (str)
        - mf_run_id (str)
        - engine (str):
            Polars engine (can be "cpu", "gpu"..)

    Results:
        - (dict)
    """

    pretty_name = "retrain-pipelines Function Calling"

    records_count = \
        dataset_dict["supervised_finetuning"]["train"].num_rows + \
        dataset_dict["supervised_finetuning"]["validation"].num_rows
    size_category = get_size_category(records_count)

    main_commit_hash, main_commit_datetime = \
        get_latest_README_commit(
            repo_id=hf_dataset_dict["repo_id"],
            target_commit_hash=hf_dataset_dict["commit_hash"],
            repo_type="dataset"
        )
    enrich_commit_hash, enrich_commit_datetime = \
        get_latest_README_commit(
            repo_id=hf_enrich_dataset_dict["repo_id"],
            target_commit_hash=\
                hf_enrich_dataset_dict["commit_hash"],
            repo_type="dataset"
        )

    main_pretty_name = get_pretty_name(
        repo_id=hf_dataset_dict["repo_id"],
        repo_type="dataset",
        commit_hash=main_commit_hash
    )
    enrich_pretty_name = get_pretty_name(
        repo_id=hf_enrich_dataset_dict["repo_id"],
        repo_type="dataset",
        commit_hash=enrich_commit_hash
    )

    main_arxiv_codes = get_arxiv_codes(
        repo_id=hf_dataset_dict["repo_id"],
        repo_type="dataset",
        commit_hash=main_commit_hash
    )
    enrich_arxiv_codes = get_arxiv_codes(
        repo_id=hf_enrich_dataset_dict["repo_id"],
        repo_type="dataset",
        commit_hash=enrich_commit_hash
    )

    main_license_label = get_license_label(
        repo_id=hf_dataset_dict["repo_id"],
        repo_type="dataset",
        commit_hash=main_commit_hash
    )
    enrich_license_label = get_license_label(
        repo_id=hf_enrich_dataset_dict["repo_id"],
        repo_type="dataset",
        commit_hash=enrich_commit_hash
    )
    license_label = \
        (main_license_label or enrich_license_label) \
        if main_license_label == enrich_license_label or \
           not (main_license_label and enrich_license_label) \
        else None
    if not license_label:
        license_label = "unknown"

    first_record_df = \
        hf_dataset_dict["lazy_df"].limit(1) \
        .collect(engine=engine)[0]
    tool_0_0 = literal_eval(first_record_df["tools"][0])[0]
    main_format_description = "attributes : \n"
    def _build_keys(d, parent='', output_str=''):
        for k, v in d.items():
            new_key = f"{parent}.{k}" if parent else k
            output_str += f" - {new_key}\n"
            if isinstance(v, dict):
                output_str = _build_keys(v, new_key, output_str)
        return output_str
    main_format_description = _build_keys(
        tool_0_0, output_str=main_format_description)
    main_format_description += "\none example : \n"
    main_format_description += json.dumps(tool_0_0, indent=4)

    return {
            "configs": dataset_dict_to_config_str(dataset_dict),
            "new_version_label": version_label,
            "commit_datetime": commit_datetime,

            "pretty_name": pretty_name,

            "main_repo_id": \
                hf_dataset_dict["repo_id"],
            "enrich_repo_id": \
                hf_enrich_dataset_dict["repo_id"],

            "main_version_label": hf_dataset_dict["version_label"],
            "main_commit_hash": main_commit_hash,
            "enrich_version_label": hf_enrich_dataset_dict["version_label"],
            "enrich_commit_hash": enrich_commit_hash,

            "main_commit_datetime": \
                main_commit_datetime,
            "enrich_commit_datetime": \
                enrich_commit_datetime,

            "main_pretty_name": main_pretty_name,
            "enrich_pretty_name": enrich_pretty_name,

            "augmentation_rate": augmentation_rate,
            "enrichment_rate": enrichment_rate,

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
    augmentation_rate: float,
    enrichment_rate: float,
    version_label: str,
    commit_datetime: datetime,
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
            - commit_datetime
            - lazy_df
        - hf_enrich_dataset_dict (dict)
            - repo_id
            - commit_hash
            - commit_datetime
        - dataset_dict (DatasetDict):
            the dataset version to be pushed
            to the HF hub.
        - augmentation_rate (float):
            - proportion of the original dataset
              that has been augmented.
        - enrichment_rate (float):
            - proportion of the original dataset
              that has been added from
              the enrichment dataset.
        - version_label (str):
            typical `retrain-pipelines`
            version label are of format "major.minor"
        - commit_datetime (datetime):
            timestamp for the new dataset version.
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
        augmentation_rate=augmentation_rate,
        enrichment_rate=enrichment_rate,
        version_label=version_label,
        commit_datetime=commit_datetime,
        mf_flow_name=mf_flow_name,
        mf_run_id=mf_run_id,
        engine=engine
    )

    env = Environment(loader=FileSystemLoader(template_folder))
    env.filters['urlencode'] = url_encode
    template = env.get_template("dataset_readme_template.md")
    readme_content = template.render(params)

    return readme_content

