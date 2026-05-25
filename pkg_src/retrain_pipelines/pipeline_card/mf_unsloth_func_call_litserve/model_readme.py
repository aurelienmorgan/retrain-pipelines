import textwrap
from datetime import datetime
from urllib.parse import quote as url_encode

from huggingface_hub import whoami
from jinja2 import Environment, FileSystemLoader

from retrain_pipelines import __version__
from retrain_pipelines.utils.hf_utils import (
    get_arxiv_codes,
    get_latest_README_commit,
    get_license_label,
    get_pretty_name,
)


def _model_readme_params(
    model_repo_id: str,
    base_model_dict: dict,
    training_dataset_dict: dict,
    version_label: str,
    commit_datetime: datetime,
    perf_metrics: dict,
    pipeline_name: str,
    exec_id: str,
) -> dict:
    """Populate the params dict to be used.

    To customize the model jinja template.

    Built on metadata from the base model.

    Parameters
    ----------
    model_repo_id : str
        id of the target repository
        on the Hugging Face Hub.
    base_model_dict : dict
        required keys :
            - repo_id (str)
            - commit_hash (str)
    training_dataset_dict : dict
        required keys :
            - repo_id (str)
            - commit_hash (str)
    version_label : str
        typical `retrain-pipelines`
        version label are of format "major.minor"
    commit_datetime : datetime
        timestamp for the new model version.
    perf_metrics : dict
        metric_name/metric_value as
        key/value pairs.
    pipeline_name : str
        name of the pipeline the execution of which
        lead to generating the herein model repo README.
    exec_id : str
        execution id.

    Returns
    -------
    dict
    """
    pretty_name = "retrain-pipelines Function Caller"

    base_model_commit_hash, base_model_commit_datetime = get_latest_README_commit(
        repo_id=base_model_dict["repo_id"],
        target_commit_hash=base_model_dict["commit_hash"],
        repo_type="model",
    )

    base_model_pretty_name = get_pretty_name(
        repo_id=base_model_dict["repo_id"], repo_type="model", commit_hash=base_model_commit_hash
    )

    base_model_arxiv_codes = get_arxiv_codes(
        repo_id=base_model_dict["repo_id"], repo_type="model", commit_hash=base_model_commit_hash
    )

    base_model_license_label = get_license_label(
        repo_id=base_model_dict["repo_id"], repo_type="model", commit_hash=base_model_commit_hash
    )
    if not base_model_license_label:
        base_model_license_label = "unknown"

    dataset_pretty_name = get_pretty_name(
        repo_id=training_dataset_dict["repo_id"],
        repo_type="dataset",
        commit_hash=training_dataset_dict["commit_hash"],
    )

    perf_metrics_yaml = textwrap.indent(
        "  metrics:\n"
        + "\n".join([
            f"    - type: {key}\n      value: {value}" for key, value in perf_metrics.items()
        ]),
        "  ",
    )

    main_usage_snippet = (
        textwrap
        .dedent("""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from torch import device, cuda

        repo_id = "{model_repo_id}"
        revision = "<model_revision_commit_hash>"
        model = AutoModelForCausalLM.from_pretrained(
            repo_id, revision=revision, torch_dtype="auto", device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(
            repo_id, revision=revision, torch_dtype="auto", device_map="auto")

        device = device("cuda" if cuda.is_available() else "cpu")
        def generate_tool_calls_list(query, max_new_tokens=400) -> str:
            formatted_query = tokenizer.chat_template.format(query, "")
            inputs = tokenizer(formatted_query, return_tensors="pt").input_ids.to(device)
            outputs = model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=False)
            generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            return generated_text[len(formatted_query):].strip()

        generate_tool_calls_list("Is 49 a perfect square ?")
    """)
        .strip()
        .format(model_repo_id=model_repo_id)
    )

    return {
        "model_repo_id": model_repo_id,
        "new_version_label": version_label,
        "commit_datetime": commit_datetime,
        "pretty_name": pretty_name,
        "dataset_pretty_name": dataset_pretty_name,
        "dataset_repo_id": training_dataset_dict["repo_id"],
        "dataset_version_label": training_dataset_dict["version_label"],
        "dataset_commit_hash": training_dataset_dict["commit_hash"],
        "dataset_commit_datetime": training_dataset_dict["commit_datetime"],
        "base_model_repo_id": base_model_dict["repo_id"],
        "base_model_pretty_name": base_model_pretty_name,
        "base_model_version_label": base_model_dict["version_label"],
        "base_model_commit_hash": base_model_commit_hash,
        "base_model_commit_datetime": base_model_commit_datetime,
        "base_model_arxiv_codes": base_model_arxiv_codes,
        "base_model_license_label": base_model_license_label,
        "perf_metrics": perf_metrics_yaml,
        "main_usage_snippet": main_usage_snippet,
        "__version__": __version__,
        "run_user": whoami()["name"],
        "pipeline_name": pipeline_name,
        "exec_id": exec_id,
    }


def get_model_readme_content(
    template_folder: str,
    model_repo_id: str,
    base_model_dict: dict,
    training_dataset_dict: dict,
    version_label: str,
    commit_datetime: datetime,
    perf_metrics: dict,
    pipeline_name: str,
    exec_id: str,
) -> str:
    """Generate markdown for model card README.

    Note: the only known way to programmatically
    add "arxiv codes" to a repo on the HF Hub
    is by including an hyperplink
    (to HF 'papers' or to arxiv. In any event,
     the paper in question must be 'uploaded'
     on the hub for this to work)
    in the body of the README.

    Parameters
    ----------
    template_folder : str
        full path to the directory holding
        the model repo REAME template.
    model_repo_id : str
        id of the target repository
        on the Hugging Face Hub.
    base_model_dict : dict
        - repo_id
        - commit_hash
    training_dataset_dict : dict
        - repo_id
        - commit_hash
        - commit_datetime
    version_label : str
        typical `retrain-pipelines`
        version label are of format "major.minor"
    commit_datetime : datetime
        timestamp for the new dataset version.
    perf_metrics : dict
        metric_name/metric_value as
        key/value pairs.
    pipeline_name : str
        name of the pipeline the execution of which
        lead to generating the herein model repo README.
    exec_id : str
        execution id.

    Returns
    -------
    str
    """
    params = _model_readme_params(
        model_repo_id=model_repo_id,
        base_model_dict=base_model_dict,
        training_dataset_dict=training_dataset_dict,
        version_label=version_label,
        commit_datetime=commit_datetime,
        perf_metrics=perf_metrics,
        pipeline_name=pipeline_name,
        exec_id=exec_id,
    )

    env = Environment(loader=FileSystemLoader(template_folder))
    env.filters["urlencode"] = url_encode
    template = env.get_template("model_readme_template.md")
    readme_content = template.render(params)

    return readme_content
