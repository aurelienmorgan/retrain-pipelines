"""Provider of convenience methods for after-pipeline-execution investigation.

The purpose of the HyperTuneCrossValInspector
is to provide convenience methods
for investigation after pipeline run,
for special focus on
"hyperparameter tuning cross validation"
intermediary training tasks.

In short : it is intended for
some "after the facts" deep-diving
and further analysis
of logged training history.
"""

import math
from typing import cast

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ...frameworks import local_metaflow as metaflow
from ...frameworks.local_metaflow.client.core import DataArtifact  # type: ignore[import-not-found]
from ..hp_cv_inspector import (
    TrainingJobHistory,
    _calculate_fold_metrics,
    _plot_average,
    _plot_fold,
)

MF_FLOW_NAME = "LightGbmHpCvWandbFlow"


def _plot_validation_rmse_analysis(
    folds_workers_history: list,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes]:
    """Generate figure and axes array plotting validation RMSE metrics across folds.

    Given a history object.
    Also lays the average across all folds.

    Parameters
    ----------
    folds_workers_history : list[dict]
        A list where each element is a dictionary representing
        the history of validation RMSE values
        for different Dask workers in a cross-validation fold.
        Each dictionary contains worker IDs as keys
        and their validation RMSE values as lists.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes]
        fig is a matplotlib figure object and
        axes is the array of axes objects
    """
    num_folds = len(folds_workers_history)
    # +1 below for the extra "average across folds" subplot
    ncols = int(math.ceil(math.sqrt(num_folds + 1)))
    nrows = int(math.ceil((num_folds + 1) / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 4 * nrows))
    axes = axes.flatten()

    all_mean_rmse = []
    all_median_rmse = []
    all_std_rmse = []

    legend_handles: list[matplotlib.artist.Artist] = []
    legend_labels: list[str] = []

    for i, (training_job_id, fold_history) in enumerate(folds_workers_history):
        epochs, mean_rmse, std_rmse, median_rmse = _calculate_fold_metrics(fold_history)
        all_mean_rmse.append(mean_rmse)
        all_std_rmse.append(std_rmse)
        all_median_rmse.append(median_rmse)

        _plot_fold(
            axes[i],
            epochs,
            mean_rmse,
            std_rmse,
            median_rmse,
            i,
            TrainingJobHistory(
                task_id=int(training_job_id),
                rank="N/A",
                fold_workers_history=fold_history,
            ),
            legend_handles,
            legend_labels,
        )

    # Plot the average metrics across all folds in the last subplot
    avg_mean_rmse = np.mean(all_mean_rmse, axis=0)
    avg_std_rmse = np.mean(all_std_rmse, axis=0)
    avg_median_rmse = np.mean(all_median_rmse, axis=0)

    avg_ax = axes[-1]
    _plot_average(avg_ax, epochs, avg_mean_rmse, avg_std_rmse, avg_median_rmse)
    # border thicness of the subplot's axes
    axes[-1].spines["top"].set_linewidth(2)
    axes[-1].spines["right"].set_linewidth(2)
    axes[-1].spines["bottom"].set_linewidth(2)
    axes[-1].spines["left"].set_linewidth(2)

    # Hide any unused subplots
    for j in range(num_folds, len(axes) - 1):
        fig.delaxes(axes[j])

    # Add the figure legend
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),  # Adjust for legend
        ncol=3,
    )

    plt.tight_layout(rect=(0, 0.1, 1, 0.9))  # Adjust layout for title & legend

    return fig, axes


def plot_run_cv_history(mf_run_id: int = -1, best_cv: bool = False):
    """Plot RMSE history for the best or worst set of HP in a pipeline execution.

    Plot RMSE history for best/worst performing (depending)
    set of hyperparameters for a given pipeline execution.

    Parameters
    ----------
    mf_run_id : int
        the id of the Metaflow flow run to consider.
        If ommitted, the last flow run is considered.
        throws MetaflowInvalidPathspec or MetaflowNotFound
    best_cv : bool
        whether to consider the best-performing
        set of hyperparameters among all
        those covered during hyperparameters tuning.
        If False, the less-well performing
        set of hp is considered
    """
    if -1 == mf_run_id:
        mf_flow_run = metaflow.Flow(MF_FLOW_NAME).latest_run
        assert mf_flow_run is not None
    else:
        mf_flow_run = metaflow.Run(f"{MF_FLOW_NAME}/{mf_run_id}")
        if mf_flow_run is None:
            raise ValueError(f"No runs found for flow {MF_FLOW_NAME}/{mf_run_id}")

    # pipeline task which holds the "hp_perfs_list"
    # consolidated artifact
    hp_perfs_list_task = [
        step.task for step in mf_flow_run.steps() if step.id == "best_hp" and step.task is not None
    ][0]

    hp_perfs_df = (
        pd
        .DataFrame(cast(DataArtifact, hp_perfs_list_task["hp_perfs_list"]).data)
        .sort_values(by="rmse", ascending=True)
        .reset_index(drop=True)
    )
    cv_task_id = hp_perfs_df.iloc[0 if best_cv else -1, 0]

    folds_workers_history: list[tuple[str, dict]] = []
    for step in mf_flow_run.steps():
        if step.id == "training_job":
            for training_job_task in step.tasks():
                if cv_task_id == cast(DataArtifact, training_job_task["cross_validation_id"]).data:
                    folds_workers_history.insert(
                        0,
                        (
                            training_job_task.id,
                            cast(DataArtifact, training_job_task["fold_workers_history"]).data,
                        ),
                    )

    fig, axes = _plot_validation_rmse_analysis(folds_workers_history)

    fig.suptitle(
        "Validation RMSE Analysis Across Folds " + f"for cv_task {cv_task_id}", fontsize=16, y=0.98
    )  # Adjust y for title

    fig.text(
        0.5,
        0.92,
        cast(DataArtifact, training_job_task["fold_hp_params"]).data,
        ha="center",
        fontsize=12,
    )

    plt.tight_layout(rect=(0, 0.1, 1, 0.9))  # Adjust layout for title & legend

    plt.show()


def plot_run_all_cv_tasks(mf_run_id: int = -1):
    """Generate and plot validation RMSE metrics charts.

    For each training_job tasks in a given pipeline run
    (i.e. all sets of hyperparameter values
     combinaitions covered during hyperparameter tuning
     for that run)

    Parameters
    ----------
    mf_run_id : int
        the id of the Metaflow flow run to consider.
        If ommitted, the last flow run is considered.
        throws MetaflowInvalidPathspec or MetaflowNotFound
    """
    if -1 == mf_run_id:
        mf_flow_run = metaflow.Flow(MF_FLOW_NAME).latest_run
        assert mf_flow_run is not None
    else:
        mf_flow_run = metaflow.Run(f"{MF_FLOW_NAME}/{mf_run_id}")
        if mf_flow_run is None:
            raise ValueError(f"No runs found for flow {MF_FLOW_NAME}/{mf_run_id}")

    ####################################
    # Collect all validation histories #
    #       for all Dask workers       #
    ####################################
    hp_perfs_histories_dict: dict[str, list[tuple[str, dict]]] = {}
    hp_dict: dict[str, str] = {}
    for step in mf_flow_run.steps():
        if step.id == "training_job":
            for training_job_task in step.tasks():
                cross_validation_id = cast(
                    DataArtifact,
                    training_job_task["cross_validation_id"],
                ).data
                if cross_validation_id not in hp_perfs_histories_dict:
                    hp_perfs_histories_dict[cross_validation_id] = []

                if cross_validation_id not in hp_dict:
                    hp_dict[cross_validation_id] = cast(
                        DataArtifact,
                        training_job_task["fold_hp_params"],
                    ).data
                hp_perfs_histories_dict[cross_validation_id].insert(
                    0,
                    (
                        training_job_task.id,
                        cast(DataArtifact, training_job_task["fold_workers_history"]).data,
                    ),
                )
    print(f"run_id {mf_flow_run.id} ; " + f"{len(hp_perfs_histories_dict)} training_job tasks")
    ####################################

    ####################################
    #          Actual plotting         #
    ####################################
    for cv_task_id in sorted(hp_perfs_histories_dict.keys(), reverse=False):
        folds_workers_history = hp_perfs_histories_dict[cv_task_id]
        # Generate the plot for the given cv_task_id
        fig, axes = _plot_validation_rmse_analysis(folds_workers_history)

        fig.suptitle(
            "Validation RMSE Analysis Across Folds " + f"for cross_validation {cv_task_id}",
            fontsize=16,
            y=0.98,
        )

        fig.text(0.5, 0.92, hp_dict[cv_task_id], ha="center", fontsize=12)

        plt.tight_layout(rect=(0, 0.1, 1, 0.9))  # Adjust layout for title & legend
        plt.show()
    ####################################
