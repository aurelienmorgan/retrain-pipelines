
import os
import time
from datetime import datetime

import pandas as pd

from dataclasses import dataclass, field
from typing import List, TypedDict

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from huggingface_hub import list_repo_refs, \
    list_repo_commits, HfApi


class EvalResults(TypedDict):
    """
    Type definition for
    evaluation results dictionary.
    """


class ModelVersion(TypedDict):
    """
    Type definition for a single
    model version entry.
    """
    branch_name: str
    commit_hash: str
    commit_datetime: datetime
    version_label: str
    eval_results: EvalResults


@dataclass
class ModelVersionsHistory:
    """
    Class representing the history of model versions
    for a Hugging Face repository.
    """
    repo_id: str
    history: List[ModelVersion] = field(default_factory=list)


def get_model_versions_history(
    repo_id: str,
    hf_token: str = os.getenv("HF_TOKEN", None),
    verbose: bool = False
) -> ModelVersionsHistory:
    """
    We look into all branches,
    revisions with a README and a "version" tag
    (signature of `retrain-pipelines`
     model versions)
    and assume they also all have
    an eval_results card_data yaml section.

    Params:
        - repo_id (str):
            Path to the HuggingFace model version.
        - hf_token (Optional, str):
            "create on namespace" permission required.
        - verbose (bool)

    Results:
        - (ModelVersionsHistory)
    """

    start = time.time()

    refs = list_repo_refs(
        repo_id=repo_id,
        repo_type="model",
        token=hf_token
    )

    api = HfApi()
    model_versions = {}
    for branch in refs.branches:
        branch_commits = list_repo_commits(
                repo_id=repo_id,
                revision=branch.name,
                repo_type="model",
                token=hf_token
            )
        for branch_commit in branch_commits:
            model_info = api.model_info(
                repo_id=repo_id,
                revision=branch_commit.commit_id,
                token=hf_token
            )
            revision_card_data = model_info.card_data
            if (
                revision_card_data and
                "version" in revision_card_data
            ):
                eval_results_dict = {
                    m['type']: m['value']
                    for m in model_info.model_index[0] \
                                ['results'][0]['metrics']
                }
                if branch_commit.created_at not in model_versions:
                    model_versions[branch_commit.created_at] = {
                        "branch_name": branch.name,
                        "commit_hash": branch_commit.commit_id,
                        "commit_datetime": branch_commit.created_at,
                        "version_label": revision_card_data["version"],
                        "eval_results": eval_results_dict
                    }
                # print(model_versions[branch_commit.created_at])

    sorted_dict = dict(sorted(model_versions.items(),
                              key=lambda x: x[0],
                              reverse=True))
    model_versions_history = list(sorted_dict.values())

    if verbose:
        print(f"{time.time() - start:.2f} seconds")

    return ModelVersionsHistory(
        repo_id=repo_id,
        history=model_versions_history
    )


def model_versions_history_html_table(
    model_versions_history: ModelVersionsHistory
) -> str:
    """
    turns a 'model_versions_history'
    into a pretty-print formatted table
    for notebook rendering via the
    `display(HTML(..))` command.
    Includes hyperlinks to model versions
    on the HuggingFace hub.

    Params:
        - model_versions_history (ModelVersionsHistory):

    Results:
        - (str)
    """

    df = pd.DataFrame(model_versions_history.history)

    df['commit_hash'] = df['commit_hash'].apply(
        lambda commit_hash:
            f"<a href=\"https://hf.co/" +
                      model_versions_history.repo_id +
                      f"/blob/{commit_hash}/README.md\" " +
                "target=\"_blank\">" +
            f"{commit_hash[:7]}</a>"
    )

    return df.to_html(escape=False, index=False)


def plot_model_versions_history(
    model_versions_history: ModelVersionsHistory,
    main_metric_name: str
) -> None:
    """
    Plot 2 metrics in dual y-axis :
      - the one specified as "main metric"
      - plus the first other available
        (if any).

    Params:
        - model_versions_history (dict):
            - model_repo_id (str)
            - model_versions_history (ModelVersionsHistory)
        - main_metric_name (str)
            The metric to be plotted on the left y-axis.
            This shall be the one used during retraining
            to promote a newly-retrained model version
            to "blessed" status
            (or not, depending on eval results of
             that model version compared to
             its best predecessor).
    """

    eval_metrics = list(
        model_versions_history.history[0]['eval_results'].keys())

    fig, ax1 = plt.subplots(figsize=(11, 4))

    versions = ["v"+entry['version_label']
                for entry in model_versions_history.history] \
               [::-1]
    main_metric_vals = [
        entry['eval_results'][main_metric_name]
        for entry in model_versions_history.history][::-1]

    line1, = ax1.plot(versions, main_metric_vals,
                      marker='', linestyle='-',
                      label=main_metric_name, 
                      color='lightsalmon', zorder=10)

    branch_colors = [
        '#98c998' if entry['branch_name'] == 'main'
        else None for entry in model_versions_history.history][::-1]
    for i, (x, y) in enumerate(zip(versions,
                                   main_metric_vals)):
        if branch_colors[i]:
            ax1.scatter(x, y, color=branch_colors[i],
                        zorder=11)

    ax1.set_xlabel('Version Label')
    ax1.set_ylabel(main_metric_name, color='salmon',
                   fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='salmon')
    ax1.set_xticks(versions)
    for i, label in enumerate(ax1.get_xticklabels()):
        if [entry['branch_name']
        for entry in model_versions_history.history][::-1][i] == 'main':
            label.set_color('#98c998')
            label.set_fontweight('bold')
        else:
            label.set_color('#707070')

    ax1.axhline(y=max(main_metric_vals), color='#98c998',
                linestyle='-', linewidth=1)

    if len(eval_metrics) > 1:
        other_metric_name = next(
            m for m in eval_metrics if m != main_metric_name)
        other_metric_vals = [
            entry['eval_results'][other_metric_name]
            for entry in model_versions_history.history][::-1]

        other_min, other_max = \
            min(other_metric_vals), max(other_metric_vals)
        main_min, main_max = \
            min(main_metric_vals), max(main_metric_vals)
        other_metric_norm = [
            (
                (x - other_min) / (other_max - other_min) *
                (main_max - main_min) + main_min
            ) if (other_max - other_min) > 0
            else 0
            for x in other_metric_vals
        ]

        # plotting on same "ax1" instance
        # so they share the same zorder system
        # (not the case with twinx() axes),
        # so we can put the first plot on top
        line2, = ax1.plot(versions, other_metric_norm,
                          marker='', linestyle='-',
                          linewidth=0.7, color='skyblue',
                          label=other_metric_name,
                          zorder=5)

        ax2 = ax1.twinx()
        ax2.set_ylim(other_min, other_max)
        darker_skyblue = (0.4, 0.7, 0.8)
        ax2.set_ylabel(other_metric_name,
                       color=darker_skyblue)
        ax2.tick_params(axis='y',
                        labelcolor=darker_skyblue)
        plt.legend([line1, line2],
                   [main_metric_name, other_metric_name],
                   loc='upper left')
    else:
        plt.legend([line1], [main_metric_name],
                   loc='upper left')

    blessed_marker = Line2D(
        [0], [0], marker='o', color='#98c998',
        markeredgewidth=0, markerfacecolor='#98c998',
        linestyle='None', markersize=7)
    handles, labels = ax1.get_legend_handles_labels()
    handles.insert(1, blessed_marker)
    labels.insert(1, 'blessed')
    legend = plt.legend(handles=handles, labels=labels,
                        loc='upper left')

    for text in legend.get_texts():
        if text.get_text() == main_metric_name:
            text.set_fontweight('bold')
            text.set_fontstyle('italic')

    ax1.grid(True, linestyle=':', alpha=0.5)
    plt.title(model_versions_history.repo_id +
              "  -  Eval by Version")
    plt.show()

