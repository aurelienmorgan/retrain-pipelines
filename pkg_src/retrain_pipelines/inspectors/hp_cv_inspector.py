"""
Provider of convenience methods for after-pipeline-execution investigation.

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

import logging
import math
from dataclasses import dataclass

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import get_execution

logger = logging.getLogger(__name__)

PIPELINE_NAME = "LightGBM_hp_cv_WandB"


@dataclass
class TrainingJobHistory:
    """Bundle for a single training_job task history.

    Groups together the task-id, the hyperparameter
    tuning rank, and the per-fold Dask workers validation
    RMSE history, so callers downstream (plotting helpers)
    can read every relevant piece from a single object.

    Attributes
    ----------
    task_id : int
        task-id of the "training_job" task.
    rank : str
        Rank of the training_job task within the
        hyperparameter tuning search, as a string.
        Surfaced in the fold subplot title.
    fold_workers_history : dict
        Per-fold Dask workers validation RMSE history.
        Keys are fold indices, values are dictionaries
        holding "Validation_rmse" lists.
    """

    task_id: int
    rank: str
    fold_workers_history: dict


def _calculate_fold_metrics(fold_history: dict) -> tuple[list, list, list, list]:
    """Calculate the mean, standard deviation, and median of RMSE values.

    For each epoch across different folds.

    Parameters
    ----------
    fold_history : dict
        A dictionary where keys are fold indices and
        values are dictionaries containing "Validation_rmse",
        a list of RMSE values (one per epoch).

    Returns
    -------
    tuple
        A tuple containing:
        - list[int]:
            List of epoch numbers
        - list[float]:
            List of mean RMSE values
        - list[float]:
            List of standard deviation
            of RMSE values
        - list[float]:
            List of median RMSE values
    """
    epochs = list(range(1, len(next(iter(fold_history.values()))["Validation_rmse"]) + 1))
    all_rmse_values: dict[int, list[float]] = {epoch: [] for epoch in epochs}

    for results in fold_history.values():
        validation_rmse = results["Validation_rmse"]
        for epoch, rmse in zip(epochs, validation_rmse, strict=False):
            all_rmse_values[epoch].append(rmse)

    rmse_arrays = {epoch: np.array(rmse_list) for epoch, rmse_list in all_rmse_values.items()}

    mean_rmse = []
    std_rmse = []
    median_rmse = []

    for epoch in epochs:
        rmse_array = rmse_arrays[epoch]
        mean_rmse.append(np.mean(rmse_array))
        std_rmse.append(np.std(rmse_array))
        median_rmse.append(np.median(rmse_array))

    return epochs, mean_rmse, std_rmse, median_rmse


def _plot_fold(
    ax: matplotlib.axes.Axes,
    epochs: list,
    mean_rmse: list,
    std_rmse: list,
    median_rmse: list,
    fold_idx: int,
    training_job_history: TrainingJobHistory,
    legend_handles: list,
    legend_labels: list,
):
    """
    Plot the validation RMSE metrics for a specific fold.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object on which to plot.
    epochs : list of int
        List of epoch numbers.
    mean_rmse : list of float
        List of mean RMSE values for each epoch.
    std_rmse : list of float
        List of standard deviation of RMSE values
        for each epoch.
    median_rmse : list[float]
        List of median RMSE values, one per epoch
    fold_idx : int
        Index of the cross-validation fold being plotted.
    training_job_history : TrainingJobHistory
        Bundle holding the retraining pipeline "training_job"
        task-id, its hyper-parameter tuning rank, and its
        per-fold Dask workers validation RMSE history.
        The rank is surfaced in the subplot title.
    legend_handles : list[matplotlib.lines.Line2D]
        by-ref param.
    legend_labels : list[str]
        by-ref param.
    """
    fold_history = training_job_history.fold_workers_history
    for worker_id, results in fold_history.items():
        validation_rmse = results["Validation_rmse"]
        # Gray and translucent lines for workers
        (line,) = ax.plot(epochs, validation_rmse, color="#4A4A4A", alpha=0.5)

        # Add text annotation next to the line
        ax.annotate(
            f"Worker {worker_id}",
            xy=(epochs[-1], validation_rmse[-1]),
            xytext=(epochs[-1] + 1, validation_rmse[-1]),
            textcoords="data",
            color=line.get_color(),
            fontsize=9,
            va="center",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.3", alpha=0.6, edgecolor="none", facecolor="white"),
        )

    (line_mean,) = ax.plot(epochs, mean_rmse, "k--", label="Mean RMSE")
    fill_std = ax.fill_between(
        epochs,
        np.array(mean_rmse) - np.array(std_rmse),
        np.array(mean_rmse) + np.array(std_rmse),
        color="#5F9FFF",
        alpha=0.15,
        label="±1 Std Dev",
    )
    (line_median,) = ax.plot(epochs, median_rmse, "r--", label="Median RMSE")

    # Add annotations for mean and median values at the last epoch
    last_epoch = epochs[-1]
    last_mean_value = mean_rmse[-1]
    ax.annotate(
        f"{last_mean_value:.2f}",
        xy=(last_epoch, last_mean_value),
        xytext=(last_epoch + 1, last_mean_value),
        textcoords="data",
        color=line_mean.get_color(),
        fontsize=9,
        va="center",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.3", edgecolor="none", facecolor="white", alpha=0.6),
    )

    last_median_value = median_rmse[-1]
    ax.annotate(
        f"{last_median_value:.2f}",
        xy=(last_epoch, last_median_value),
        xytext=(last_epoch + 1, last_median_value),
        textcoords="data",
        color=line_median.get_color(),
        fontsize=9,
        va="center",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.3", alpha=0.6, edgecolor="none", facecolor="white"),
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation RMSE")
    ax.set_title(
        f"Fold {fold_idx + 1} - training_job {training_job_history.task_id} "
        f"(rank {training_job_history.rank})"
    )
    ax.grid(which="both", linestyle=":", linewidth=0.5, color="blue")

    # Add handles and labels for legend
    legend_handles.append(line_mean)
    legend_handles.append(fill_std)
    legend_handles.append(line_median)

    if "Mean RMSE" not in legend_labels:
        legend_labels.append("Mean RMSE")
    if "±1 Std Dev" not in legend_labels:
        legend_labels.append("±1 Std Dev")
    if "Median RMSE" not in legend_labels:
        legend_labels.append("Median RMSE")


def _plot_average(
    ax: matplotlib.axes.Axes,
    epochs: list,
    avg_mean_rmse: list,
    avg_std_rmse: list,
    avg_median_rmse: list,
):
    """Plot the average validation RMSE metrics across all folds.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object on which to plot.
    epochs : list[int]
        List of epoch numbers.
    avg_mean_rmse : list[float]
        List of average mean RMSE values across folds
        for each epoch.
    avg_std_rmse : list[float]
        List of average standard deviation of RMSE values
        across folds for each epoch.
    avg_median_rmse : list of float
        List of average median RMSE values across folds
        for each epoch.
    """
    ax.plot(epochs, avg_mean_rmse, "k--", label="Mean RMSE")
    avg_mean_rmse_arr = np.array(avg_mean_rmse)
    avg_std_rmse_arr = np.array(avg_std_rmse)  # (np.array(avg_std_rmse),)
    ax.fill_between(
        epochs,
        avg_mean_rmse_arr - avg_std_rmse_arr,
        avg_mean_rmse_arr + avg_std_rmse_arr,
        color="#5F9FFF",
        alpha=0.15,
        label="±1 Std Dev",
    )
    ax.plot(epochs, avg_median_rmse, "r--", label="Median RMSE")

    # Add annotations for mean and median values
    # at the last epoch
    last_epoch = epochs[-1]
    last_avg_mean_value = avg_mean_rmse[-1]
    ax.annotate(
        f"{last_avg_mean_value:.2f}",
        xy=(last_epoch, last_avg_mean_value),
        xytext=(last_epoch + 1, last_avg_mean_value),
        textcoords="data",
        color="k",
        fontsize=9,
        va="center",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.3", alpha=0.6, edgecolor="none", facecolor="white"),
    )

    last_avg_median_value = avg_median_rmse[-1]
    ax.annotate(
        f"{last_avg_median_value:.2f}",
        xy=(last_epoch, last_avg_median_value),
        xytext=(last_epoch + 1, last_avg_median_value),
        textcoords="data",
        color="r",
        fontsize=9,
        va="center",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.3", alpha=0.6, edgecolor="none", facecolor="white"),
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation RMSE")
    ax.set_title("Average Across Folds")
    ax.grid(which="both", linestyle=":", linewidth=0.5, color="blue")


def _plot_validation_rmse_analysis(
    folds_workers_history: list[TrainingJobHistory],
) -> tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes]:
    """Generate figure and axes array plotting validation RMSE metrics across folds.

    Given a history object.
    Also lays the average across all folds.

    Parameters
    ----------
    folds_workers_history : list[TrainingJobHistory]
        A list where each element bundles the
        retraining pipeline "training_job" task-id,
        its hyperparameter tuning rank, and its
        per-fold Dask workers validation RMSE history.

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

    for i, training_job_history in enumerate(folds_workers_history):
        fold_history = training_job_history.fold_workers_history
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
            training_job_history,
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


def plot_validation_rmse_analysis(folds_workers_history: list[TrainingJobHistory]):
    """Plot validation RMSE metrics across folds.

    Given a history object.
    Also lays the average across all folds.

    Parameters
    ----------
    folds_workers_history : list[TrainingJobHistory]
        A list where each element bundles the retraining pipeline
        "training_job" task-id, its hyperparameter tuning rank,
        and its per-fold Dask workers validation RMSE history.
    """
    fig, axes = _plot_validation_rmse_analysis(folds_workers_history)

    fig.suptitle("Validation RMSE Analysis Across Folds", fontsize=16, y=0.98)  # Adjust y for title

    plt.show()


def plot_exec_cv_history(exec_id: int = -1, best_cv: bool = True) -> None:
    """Plot RMSE history for the best or worst set of HP in a pipeline execution.

    Plot RMSE history for best/worst performing (depending)
    set of hyperparameters for a given pipeline execution.

    Parameters
    ----------
    exec_id : int
        the id of the retrainin-pipeline execution to consider.
        If ommitted, the last execution is considered.
    best_cv : bool
        whether to consider the best-performing
        set of hyperparameters among all
        those covered during hyperparameters tuning.
        If False, the less-well performing
        set of hp is considered
    """
    execution = get_execution(PIPELINE_NAME, exec_id)
    if PIPELINE_NAME != execution.name:
        raise ValueError(f"Only available for the {PIPELINE_NAME} retraining pipeline.")

    # pipeline task which holds the "hp_perfs_list"
    # consolidated artifact
    hp_perfs_list_tasks = execution.get_tasks_with_name("best_hp")
    if len(hp_perfs_list_tasks) == 0:
        logger.warning(
            f"Execution of the retraining-pipeline '{execution.name}' "
            f"with id {execution.id} didn't go through a 'best_hp' task."
        )
        return

    hp_perfs_df = (
        pd
        .DataFrame(hp_perfs_list_tasks[0].get_exit_context()["hp_perfs_list"])
        .sort_values(by="rmse", ascending=True)
        .reset_index(drop=True)
    )
    cv_task_id = hp_perfs_df.iloc[0 if best_cv else -1, 0]
    cv_task_rank = hp_perfs_df.iloc[0 if best_cv else -1, 1]

    folds_workers_history: list[TrainingJobHistory] = []
    for training_job_task in execution.get_tasks_with_name("training_job"):
        if (
            str(cv_task_rank)
            == list(training_job_task.get_exit_context()["cross_validation"].keys())[0]
        ):
            training_job_task_context = training_job_task.get_exit_context()
            training_job_task_rank_context = training_job_task_context["training_job"][
                str(training_job_task.rank)
            ]
            folds_workers_history.insert(
                0,
                TrainingJobHistory(
                    task_id=int(training_job_task.id),
                    rank=str(training_job_task.rank),
                    fold_workers_history=training_job_task_rank_context["fold_workers_history"],
                ),
            )

    fig, axes = _plot_validation_rmse_analysis(folds_workers_history)

    fig.suptitle(
        f"Validation RMSE Analysis Across Folds for cv_task {cv_task_id} (rank {cv_task_rank})",
        fontsize=16,
        y=0.98,
    )  # Adjust y for title

    fig.text(
        0.5,
        0.92,
        hp_perfs_list_tasks[0].get_exit_context()["cross_validation"][str(cv_task_rank)]["hp"],
        ha="center",
        fontsize=12,
    )

    plt.tight_layout(rect=(0, 0.1, 1, 0.9))  # Adjust layout for title & legend

    plt.show()


def plot_exec_all_cv_tasks(exec_id: int = -1) -> None:
    """Generate and plot validation RMSE metrics charts.

    For each training_job tasks in a given pipeline run
    (i.e. all sets of hyperparameter values
     combinaitions covered during hyperparameter tuning
     for that run)

    Parameters
    ----------
    exec_id : int
        the id of the retrainin-pipeline execution to consider.
        If ommitted, the last execution is considered.
    """
    execution = get_execution(PIPELINE_NAME, exec_id)
    if PIPELINE_NAME != execution.name:
        raise ValueError(f"Only available for the {PIPELINE_NAME} retraining pipeline.")

    ####################################
    # Collect all validation histories #
    #       for all Dask workers       #
    ####################################
    hp_perfs_histories_dict: dict[str, list[TrainingJobHistory]] = {}
    cross_validation_ranks_dict: dict[str, str] = {}
    hp_dict: dict[str, str] = {}
    for training_job_task in execution.get_tasks_with_name("training_job"):
        cross_validation = training_job_task.get_exit_context()["cross_validation"]
        cross_validation_rank = list(cross_validation.keys())[0]
        cross_validation_dict = list(cross_validation.values())[0]
        cross_validation_id = cross_validation_dict["task_id"]

        if cross_validation_id not in cross_validation_ranks_dict:
            cross_validation_ranks_dict[cross_validation_id] = str(cross_validation_rank)

        if cross_validation_id not in hp_perfs_histories_dict:
            hp_perfs_histories_dict[cross_validation_id] = []

        if cross_validation_id not in hp_dict:
            hp_dict[cross_validation_id] = cross_validation_dict["hp"]

        training_job_task_context = training_job_task.get_exit_context()
        training_job_task_rank_context = training_job_task_context["training_job"][
            str(training_job_task.rank)
        ]

        hp_perfs_histories_dict[cross_validation_id].insert(
            0,
            TrainingJobHistory(
                task_id=int(training_job_task.id),
                rank=str(training_job_task.rank),
                fold_workers_history=training_job_task_rank_context["fold_workers_history"],
            ),
        )

    print(f"run_id {execution.id} ; " + f"{len(hp_perfs_histories_dict)} training_job tasks")
    ####################################

    ####################################
    #          Actual plotting         #
    ####################################
    for cv_task_id in sorted(hp_perfs_histories_dict.keys(), reverse=False):
        cv_task_rank = cross_validation_ranks_dict[cv_task_id]
        folds_workers_history = hp_perfs_histories_dict[cv_task_id]
        # Generate the plot for the given cv_task_id
        fig, axes = _plot_validation_rmse_analysis(folds_workers_history)

        fig.suptitle(
            "Validation RMSE Analysis Across Folds for cross_validation "
            + f"{cv_task_id} (rank {cv_task_rank})",
            fontsize=16,
            y=0.98,
        )

        fig.text(0.5, 0.92, hp_dict[cv_task_id], ha="center", fontsize=12)

        plt.tight_layout(rect=(0, 0.1, 1, 0.9))  # Adjust layout for title & legend
        plt.show()
    ####################################
