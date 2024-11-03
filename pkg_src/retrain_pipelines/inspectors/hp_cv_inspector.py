
"""
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

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

import math

import metaflow

MF_FLOW_NAME = "LightGbmHpCvWandbFlow"


def _calculate_fold_metrics(
    fold_history: dict
) -> (list, list, list, list):
    """
    Calculate the mean, standard deviation, and median of RMSE values
    for each epoch across different folds.

    Params:
        - fold_history (dict):
            A dictionary where keys are fold indices and
            values are dictionaries containing "Validation_rmse",
            a list of RMSE values (one per epoch).

    Results:
        - tuple:
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
    epochs = list(
        range(1, len(next(iter(fold_history.values())
             )["Validation_rmse"]) + 1))
    all_rmse_values = {epoch: [] for epoch in epochs}

    for results in fold_history.values():
        validation_rmse = results["Validation_rmse"]
        for epoch, rmse in zip(epochs, validation_rmse):
            all_rmse_values[epoch].append(rmse)

    all_rmse_values = {epoch: np.array(rmse_list)
                       for epoch, rmse_list in all_rmse_values.items()}

    mean_rmse = []
    std_rmse = []
    median_rmse = []

    for epoch in epochs:
        rmse_array = all_rmse_values[epoch]
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
    fold_history: dict,
    fold_idx: int,
    training_job_id: str,
    legend_handles: list,
    legend_labels: list
):
    """
    Plot the validation RMSE metrics for a specific fold.

    Params:
        - ax (matplotlib.axes.Axes):
            The axes object on which to plot.
        - epochs (list of int):
            List of epoch numbers.
        - mean_rmse (list of float):
            List of mean RMSE values for each epoch.
        - std_rmse (list of float):
            List of standard deviation of RMSE values
            for each epoch.
        - median_rmse (list[float]):
            List of median RMSE values, one per epoch
        - fold_history (dict):
            A dictionary where keys are Dask worker IDs
            and values are dictionaries containing
            "Validation_rmse", a list of RMSE values
            for each epoch.
        - fold_idx (int):
            Index of the cross-validation fold being plotted.
        - training_job_id (str):
            Metaflow id for the "training_job" task covered
        - legend_handles (list[matplotlib.lines.Line2D]):
        - legend_labels (list[str]):
    """
    for worker_id, results in fold_history.items():
        validation_rmse = results["Validation_rmse"]
        # Gray and translucent lines for workers
        line, = ax.plot(epochs, validation_rmse,
                color='#4A4A4A', alpha=0.5)

        # Add text annotation next to the line
        ax.annotate(f'Worker {worker_id}', 
                    xy=(epochs[-1], validation_rmse[-1]), 
                    xytext=(epochs[-1] + 1, validation_rmse[-1]), 
                    textcoords='data',
                    color=line.get_color(),
                    fontsize=9,
                    va='center',
                    ha='left',
                    bbox=dict(boxstyle="round,pad=0.3", alpha=0.6,
                              edgecolor='none', facecolor='white'))

    line_mean, = ax.plot(epochs, mean_rmse,
                         'k--', label='Mean RMSE')
    fill_std = ax.fill_between(
        epochs, np.array(mean_rmse) - np.array(std_rmse),
        np.array(mean_rmse) + np.array(std_rmse),
        color='#5F9FFF', alpha=0.15, label='±1 Std Dev')
    line_median, = ax.plot(epochs, median_rmse,
                           'r--', label='Median RMSE')

    # Add annotations for mean and median values at the last epoch
    last_epoch = epochs[-1]
    last_mean_value = mean_rmse[-1]
    ax.annotate(f'{last_mean_value:.2f}', 
                xy=(last_epoch, last_mean_value), 
                xytext=(last_epoch + 1, last_mean_value), 
                textcoords='data',
                color=line_mean.get_color(),
                fontsize=9,
                va='center',
                ha='left',
                bbox=dict(boxstyle="round,pad=0.3",
                edgecolor='none', facecolor='white',
                alpha=0.6))

    last_median_value = median_rmse[-1]
    ax.annotate(f'{last_median_value:.2f}', 
                xy=(last_epoch, last_median_value), 
                xytext=(last_epoch + 1, last_median_value), 
                textcoords='data',
                color=line_median.get_color(),
                fontsize=9,
                va='center',
                ha='left',
                bbox=dict(boxstyle="round,pad=0.3", alpha=0.6,
                          edgecolor='none', facecolor='white'))
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation RMSE')
    ax.set_title(f"Fold {fold_idx + 1} - "+
                 f"training_job {training_job_id}")
    ax.grid(which='both', linestyle=':', linewidth=0.5,
            color='blue')
    
    # Add handles and labels for legend
    legend_handles.append(line_mean)
    legend_handles.append(fill_std)
    legend_handles.append(line_median)

    if 'Mean RMSE' not in legend_labels:
        legend_labels.append('Mean RMSE')
    if '±1 Std Dev' not in legend_labels:
        legend_labels.append('±1 Std Dev')
    if 'Median RMSE' not in legend_labels:
        legend_labels.append('Median RMSE')


def _plot_average(
    ax: matplotlib.axes.Axes,
    epochs: list,
    avg_mean_rmse: list,
    avg_std_rmse: list,
    avg_median_rmse: list
):
    """
    Plot the average validation RMSE metrics
    across all folds.

    Params:
        - ax (matplotlib.axes.Axes):  
            The axes object on which to plot.
        - epochs (list[int]):
            List of epoch numbers.
        - avg_mean_rmse (list[float]):
            List of average mean RMSE values across folds
            for each epoch.
        - avg_std_rmse (list[float]):
            List of average standard deviation of RMSE values
            across folds for each epoch.
        - avg_median_rmse (list of float):
            List of average median RMSE values across folds
            for each epoch.
    """

    ax.plot(epochs, avg_mean_rmse, 'k--',
            label='Mean RMSE')
    ax.fill_between(epochs,
                    avg_mean_rmse - avg_std_rmse,
                    avg_mean_rmse + avg_std_rmse,
                    color='#5F9FFF', alpha=0.15,
                    label='±1 Std Dev')
    ax.plot(epochs, avg_median_rmse, 'r--', label='Median RMSE')

    # Add annotations for mean and median values
    # at the last epoch
    last_epoch = epochs[-1]
    last_avg_mean_value = avg_mean_rmse[-1]
    ax.annotate(f'{last_avg_mean_value:.2f}', 
                xy=(last_epoch, last_avg_mean_value), 
                xytext=(last_epoch + 1, last_avg_mean_value), 
                textcoords='data',
                color='k',
                fontsize=9,
                va='center',
                ha='left',
                bbox=dict(boxstyle="round,pad=0.3", alpha=0.6,
                          edgecolor='none', facecolor='white'))

    last_avg_median_value = avg_median_rmse[-1]
    ax.annotate(f'{last_avg_median_value:.2f}', 
                xy=(last_epoch, last_avg_median_value), 
                xytext=(last_epoch + 1, last_avg_median_value), 
                textcoords='data',
                color='r',
                fontsize=9,
                va='center',
                ha='left',
                bbox=dict(boxstyle="round,pad=0.3", alpha=0.6,
                          edgecolor='none', facecolor='white'))

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation RMSE')
    ax.set_title('Average Across Folds')
    ax.grid(which='both', linestyle=':', linewidth=0.5, color='blue')


def _plot_validation_rmse_analysis(
    folds_workers_history: list
) -> (
    matplotlib.figure.Figure,
    matplotlib.axes._axes.Axes
) :
    """
    Given a history object,
    generates a figure and axes array
    plotting validation RMSE metrics
    across different folds
    and the average across all folds.

    Params:
        - folds_workers_history (list[dict]):
            A list where each element is a dictionary representing
            the history of validation RMSE values
            for different Dask workers in a cross-validation fold.
            Each dictionary contains worker IDs as keys
            and their validation RMSE values as lists.

    Results:
        tuple(
            matplotlib.figure.Figure,
            matplotlib.axes._axes.Axes
        ):
            fig is a matplotlib figure object and
            axes is the array of axes objects
    """
    num_folds = len(folds_workers_history)
    # +1 below for the extra "average across folds" subplot
    ncols = int(math.ceil(math.sqrt(num_folds + 1)))
    nrows = int(math.ceil((num_folds + 1) / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(15, 4 * nrows))
    axes = axes.flatten()

    all_mean_rmse = []
    all_median_rmse = []
    all_std_rmse = []

    legend_handles = []
    legend_labels = []

    for i, (training_job_id, fold_history) \
    in enumerate(folds_workers_history):
        epochs, mean_rmse, std_rmse, median_rmse = \
            _calculate_fold_metrics(fold_history)
        all_mean_rmse.append(mean_rmse)
        all_std_rmse.append(std_rmse)
        all_median_rmse.append(median_rmse)

        _plot_fold(axes[i], epochs, mean_rmse,
                   std_rmse, median_rmse, fold_history,
                   i, training_job_id, legend_handles,
                   legend_labels)

    # Plot the average metrics across all folds in the last subplot
    avg_mean_rmse = np.mean(all_mean_rmse, axis=0)
    avg_std_rmse = np.mean(all_std_rmse, axis=0)
    avg_median_rmse = np.mean(all_median_rmse, axis=0)

    avg_ax = axes[-1]
    _plot_average(avg_ax, epochs, avg_mean_rmse,
                  avg_std_rmse, avg_median_rmse)
    # border thicness of the subplot's axes
    axes[-1].spines['top'].set_linewidth(2)
    axes[-1].spines['right'].set_linewidth(2)
    axes[-1].spines['bottom'].set_linewidth(2)
    axes[-1].spines['left'].set_linewidth(2)

    # Hide any unused subplots
    for j in range(num_folds, len(axes) - 1):
        fig.delaxes(axes[j])

    # Add the figure legend
    fig.legend(legend_handles, legend_labels,
               loc='upper center',
               bbox_to_anchor=(0.5, 0.92), # Adjust for legend
               ncol=3)

    plt.tight_layout(
        rect=[0, 0.1, 1, 0.9])  # Adjust layout for title & legend

    return fig, axes


def plot_validation_rmse_analysis(
    folds_workers_history: list
):
    """
    Given a history object, plot validation RMSE metrics
    across different folds and the average across all folds.

    Params:
        - folds_workers_history (list[dict]):
            A list where each element is a dictionary representing
            the history of validation RMSE values
            for different Dask workers in a cross-validation fold.
            Each dictionary contains worker IDs as keys
            and their validation RMSE values as lists.
    """

    fig, axes = _plot_validation_rmse_analysis(
        folds_workers_history)

    fig.suptitle('Validation RMSE Analysis Across Folds',
                 fontsize=16, y=0.98)  # Adjust y for title

    plt.show()

def plot_run_cv_history(
    mf_run_id: int = -1,
    best_cv: bool = False
):
    """
    Plots RMSE history for best/worst performing (depending)
    set of hyperparameters for a given pipeline execution.

    Params:
        - mf_run_id (int):
            the id of the Metaflow flow run to consider.
            If ommitted, the last flow run is considered.
            throws MetaflowInvalidPathspec or MetaflowNotFound
        - best_cv (bool):
            whether to consider the best-performing
            set of hyperparameters among all
            those covered during hyperparameters tuning.
            If False, the less-well performing
            set of hp is considered
    """

    if -1 == mf_run_id:
        mf_flow_run = metaflow.Flow(MF_FLOW_NAME).latest_run
    else:
        mf_flow_run = metaflow.Run(f"{MF_FLOW_NAME}/{mf_run_id}")

    # pipeline task which holds the "hp_perfs_list"
    # consolidated artifact
    hp_perfs_list_task = [
            step.task
            for step in mf_flow_run.steps()
            if step.id == 'best_hp'
        ][0]

    hp_perfs_df = pd.DataFrame(
        hp_perfs_list_task['hp_perfs_list'].data) \
            .sort_values(by='rmse', ascending=True) \
            .reset_index(drop=True)
    cv_task_id = hp_perfs_df.iloc[0 if best_cv else -1, 0]

    folds_workers_history = []
    for step in mf_flow_run.steps():
        if step.id == 'training_job':
            for training_job_task in step.tasks():
                if (
                    cv_task_id ==
                    training_job_task['cross_validation_id'].data
                ):
                    folds_workers_history.insert(
                        0,
                        (
                            training_job_task.id,
                            training_job_task[
                                'fold_workers_history'].data
                        )
                    )

    fig, axes = _plot_validation_rmse_analysis(
                    folds_workers_history)

    fig.suptitle(
        f"Validation RMSE Analysis Across Folds "+
        f"for cv_task {cv_task_id}",
        fontsize=16, y=0.98)  # Adjust y for title

    fig.text(0.5, 0.92, training_job_task['fold_hp_params'].data,
             ha="center", fontsize=12)

    plt.tight_layout(
        rect=[0, 0.1, 1, 0.9]) # Adjust layout for title & legend

    plt.show()


def plot_run_all_cv_tasks(
    mf_run_id: int = -1
):
    """
    Generate and plots of validation RMSE metrics
    for each training_job tasks in a given pipeline run
    (i.e. all sets of hyperparameter values
     combinaitions covered during hyperparameter tuning
     for that run)

    Params:
        - mf_run_id (int):
            the id of the Metaflow flow run to consider.
            If ommitted, the last flow run is considered.
            throws MetaflowInvalidPathspec or MetaflowNotFound
    """

    if -1 == mf_run_id:
        mf_flow_run = metaflow.Flow(MF_FLOW_NAME).latest_run
    else:
        mf_flow_run = metaflow.Run(f"{MF_FLOW_NAME}/{mf_run_id}")

    ####################################
    # Collect all validation histories #
    #       for all Dask workers       #
    ####################################
    hp_perfs_histories_dict = {}
    hp_dict = {}
    for step in mf_flow_run.steps():
        if step.id == 'training_job':
            for training_job_task in step.tasks():

                cross_validation_id = \
                    training_job_task['cross_validation_id'].data
                if not cross_validation_id in hp_perfs_histories_dict:
                    hp_perfs_histories_dict[cross_validation_id] = []

                if not cross_validation_id in hp_dict:
                    hp_dict[cross_validation_id] = \
                        training_job_task['fold_hp_params'].data
                hp_perfs_histories_dict[cross_validation_id].insert(
                        0,
                        (
                            training_job_task.id,
                            training_job_task['fold_workers_history'].data
                        )
                    )
    print(f"run_id {mf_flow_run.id} ; "+
          f"{len(hp_perfs_histories_dict)} training_job tasks")
    ####################################

    ####################################
    #          Actual plotting         #
    ####################################
    for cv_task_id in sorted(hp_perfs_histories_dict.keys(),
                             reverse=False):
        folds_workers_history = \
            hp_perfs_histories_dict[cv_task_id]
        # Generate the plot for the given cv_task_id
        fig, axes = _plot_validation_rmse_analysis(
                        folds_workers_history)

        fig.suptitle(f"Validation RMSE Analysis Across Folds "+
                     f"for cross_validation {cv_task_id}",
                     fontsize=16, y=0.98)

        fig.text(0.5, 0.92, hp_dict[cv_task_id],
                 ha="center", fontsize=12)

        plt.tight_layout(
            rect=[0, 0.1, 1, 0.9])  # Adjust layout for title & legend
        plt.show()
    ####################################

