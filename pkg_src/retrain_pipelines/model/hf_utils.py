
import os
import sys
import numpy as np
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError

from retrain_pipelines import __version__
from retrain_pipelines.utils.hf_utils import \
    local_repo_folder_to_hub, get_commit_created_at


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

    Note: we look into the "main" branch of the model repo
    because it's the `retrain-pipelines` branch
    for blessed model versions.

    Note : this method here is permissive to
    non-`retrain-pipelines` trained prior model versions.
    As long as they have reported eval results.

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

    hf_api = HfApi()

    try:
        model_info = hf_api.repo_info(
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

    if (
        model_info and
        model_info.model_index and
        "results" in model_info.model_index[0] and
        "metrics" in model_info.model_index[0]["results"][0]
    ):
        if "timestamp" in model_info.cardData:
            # sign of a `retrain-pipelines` model card
            commit_datetime = datetime.strptime(
                model_info.cardData["timestamp"],
                "%Y%m%d_%H%M%S%f_%Z")
        else:
            commit_datetime = get_commit_created_at(
                hf_api=api, repo_id=repo_id,
                revision="main",
                repo_type="model",
                hf_token=hf_token
            )

        eval_results_dict = {
                m['type']: m['value']
                for m in model_info \
                            .model_index[0]["results"][0]["metrics"]
            }

        return {
            "mf_run_id": model_info.cardData["mf_run_id"],
            "commit_hash": model_info.sha,
            "version_label": model_info.cardData["version"],
            "commit_datetime": commit_datetime,
            "perf_metrics": eval_results_dict
        }

    return None


def _mpl_scientific_format_func(value, tick_number):
    """Must be declared outside calling function
    for returned objects to be 'pickelable'.
    plt.FuncFormatter(lambda x, _: f"{x:.2e}")
    scientific notation with two decimal places"""
    return f"{value:.2e}"

def plot_log_history(
    log_history: list,
    title: str,
    sliding_window_size: int = 10
) -> Figure:
    """
    Dual y-axis plot for training logs
    with no validation datapoints.
    Tripple y-axis otherwise
    (training loss, validation loss and learning rate).

    Note: for the training loss curve, a smoothed version
    is overlayed on top of partially transparent
    actual measurements.

    Params:
        - log_history (list):
            as provided by
            `transformers.Trainer.state.log_history`
        - title (str):
            the figure title.
        - sliding_window_size (int):
            size of the sliding window used
            for training loss curve smoothing.

    Results:
        - (Figure)
    """

    training_data = [d for d in log_history
                     if all(k in d for k
                            in ["loss", "epoch", "step",
                                "learning_rate"])]
    validation_data = [(log["epoch"], log["eval_loss"])
                       for log in log_history
                       if "eval_loss" in log]

    loss_values = [d["loss"] for d in training_data]
    epochs = [d["epoch"] for d in training_data]
    learning_rates = [d["learning_rate"] for d in training_data]
    validation_epochs, validation_losses = \
        zip(*validation_data) if validation_data else ([], [])

    moving_avg = [
        np.mean(
            loss_values[max(0, i-(sliding_window_size-1)):i+1])
        for i in range(len(loss_values))]

    fig, ax1 = plt.subplots(figsize=(7, 3.45))

    ax1.plot(epochs, loss_values, "c-", linewidth=0.5, alpha=0.4)
    ax1.plot(epochs, moving_avg, "c-", linewidth=2, alpha=1.0)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss", color="c")
    ax1.tick_params(axis="y", labelcolor="c")
    ax1.set_xticks(np.arange(0, max(epochs) + 1))
    ax1.grid(color="lightgrey", linestyle="-", linewidth=0.5)

    if validation_data:
        ax2 = ax1.twinx()
        ax2.plot(validation_epochs, validation_losses, 'b--',
                 linewidth=1, alpha=.8)
        ax2.set_ylabel("Validation Loss", color="b")
        ax2.tick_params(axis="y", labelcolor="b")

        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("outward", 60))
        ax3.plot(epochs, learning_rates, "m--",
                 linewidth=1.5, alpha=1.0)
        ax3.set_ylabel("Learning Rate", color="m")
        ax3.tick_params(axis="y", labelcolor="m")
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(
            _mpl_scientific_format_func))

    else:
        ax2 = ax1.twinx()
        ax2.plot(epochs, learning_rates, "m--",
                 linewidth=1.5, alpha=1.0)
        ax2.set_ylabel("Learning Rate", color="m")
        ax2.tick_params(axis="y", labelcolor="m")
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(
            _mpl_scientific_format_func))

    ax1.set_title(title)
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    if "ax3" in locals():
        ax3.spines["top"].set_visible(False)

    fig.tight_layout()
    plt.close(fig)

    return fig

