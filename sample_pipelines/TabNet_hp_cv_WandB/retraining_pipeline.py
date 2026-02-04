
import os
import re
import sys
import json
import time
import shutil
import logging
import traceback

import importlib.util

from typing import List
from textwrap import dedent

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, \
                                    train_test_split
from sklearn.preprocessing import StandardScaler, \
                                  OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, \
                            recall_score, f1_score, \
                            confusion_matrix


from retrain_pipelines import __version__
from retrain_pipelines.dataset import features_desc, \
                                      features_distri_plot
from retrain_pipelines.dataset.features_dependencies import \
    dataset_to_heatmap_fig
from retrain_pipelines.dag_engine.core import \
    TaskPayload, task, taskgroup, parallel_task, \
    dag, DagParam, ctx, UiCss

from retrain_pipelines.utils import flatten_dict, \
    dict_dict_list_get_all_combinations, \
    create_requirements

from retrain_pipelines.dag_engine.sdk import \
    ExecutionsIterator


import wandb

from pytorch_tabnet.tab_model import TabNetClassifier
from torch.nn import CrossEntropyLoss


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@task
def start() -> TaskPayload:
    logger.info(f"{ctx.pipeline_name} - {ctx.exec_id}")
    logging.getLogger("retrain_pipelines").setLevel(logging.INFO)

    # inputs validation
    assert os.path.exists(os.path.realpath(ctx.data_file_fullname))
    ctx.buckets_param = json.loads(ctx.buckets_param)
    ctx.pipeline_hp_grid = json.loads(ctx.pipeline_hp_grid)
    ctx.cv_folds = int(ctx.cv_folds)
    assert ctx.wandb_run_mode in ['disabled', 'offline', 'online']

    # the WandB local folder with which the server is async.
    artifacts_root_dir = os.environ["RP_ARTIFACTS_STORE"]
    os.environ["WANDB_DIR"] = artifacts_root_dir
    ctx.wandb_run_dir = \
        os.path.join(artifacts_root_dir,
                     ctx.pipeline_name, str(ctx.exec_id))
    # logger.info(f"wandb_run_dir : {ctx.wandb_run_dir}")
    if not os.path.exists(ctx.wandb_run_dir):
        os.makedirs(ctx.wandb_run_dir)
    if "disabled" != ctx.wandb_run_mode:
        _ = wandb.login(
            host="https://api.wandb.ai"
        )

    data = pd.read_csv(ctx.data_file_fullname)
    ctx.data = data

    ctx.model_version_blessed = False
    ctx.current_blessed_exec = None

    ctx.retrain_pipelines = f"retrain-pipelines {__version__}"
    ctx.retrain_pipeline_type = os.environ["retrain_pipeline_type"]

    ctx.serving_artifacts_local_folder = os.path.realpath(os.path.join(
        os.path.dirname(__file__), "..", "..", "serving_artifacts",
        ctx.pipeline_name, str(ctx.exec_id)
    ))

    if not os.path.exists(ctx.serving_artifacts_local_folder):
        os.makedirs(ctx.serving_artifacts_local_folder)

    return data


@task
def eda(payload: TaskPayload) -> None:
    """
    exploratory data analysis.
    """

    data = payload

    ############################
    #    features and label    #
    #       basic counts       #
    ############################
    column_types = features_desc(data)
    ctx.features_desc = column_types
    ############################

    ############################
    #    features and label    #
    # respective distributions #
    ############################
    fig = features_distri_plot(data)
    ctx.data_distri_fig = fig
    ############################

    ############################
    #    features and label    #
    #       relationships      #
    ############################
    fig, _ = dataset_to_heatmap_fig(data)
    ctx.data_heatmap_fig = fig
    ############################

    return None


@task
def preprocess_data(payload: TaskPayload) -> TaskPayload:
    """
    feature engineering
    """
    # Load preprocessing module (can be user-tailored)
    from retrain_pipelines.utils import get_preprocess_data_fct
    print(f"ctx.preprocess_artifacts_path : {ctx.preprocess_artifacts_path}")
    preprocess_data_fct = \
        get_preprocess_data_fct(ctx.preprocess_artifacts_path)
    #from retrain_pipelines.model import preprocess_data_fct

    scaler = StandardScaler()
    encoder = OneHotEncoder(
        sparse_output=False,
        drop=None # drop='first' for linear models, @see the doc
    )
    buckets = ctx.buckets_param.copy()

    ctx.X = ctx.data.iloc[:, :-1]
    ctx.y = ctx.data.iloc[:, -1]

    grouped_features = []
    X_transformed = preprocess_data_fct(
        ctx.X, scaler, encoder, buckets, grouped_features,
        is_training=True, local_path=ctx.serving_artifacts_local_folder
    )
    ctx.scaler = scaler                     # <= to artifact store
    ctx.encoder = encoder                   # <= to artifact store
    ctx.buckets = buckets                   # <= to artifact store
    ctx.grouped_features = grouped_features # <= to artifact store
                                            # (and coming model init)

    return X_transformed


@taskgroup
def data() -> tuple:
    return eda, preprocess_data


@task
def split_data(payload: TaskPayload) -> None:
    """
    hold-out dataset for eval
    on the final overall retrained model version.
    """
    X_transformed = payload["preprocess_data"]

    ctx.X_train, ctx.X_test, ctx.y_train, ctx.y_test = \
        train_test_split(
            X_transformed, ctx.y,
            test_size=0.2, random_state=42
        )
    logger.info(f"train set : {len(ctx.X_train):,}, " +
                f"holdout set : {len(ctx.X_test):,}")

    return None


@task
def hyper_tuning(_):
    """
    Hyperparameters tuning
    """

    # Generate all combinations of parameters
    all_hp_params = \
        dict_dict_list_get_all_combinations(
            ctx.pipeline_hp_grid)

    for i, hp_params in enumerate(all_hp_params):
        logger.info(
            f"hp_params[{i}] : " +
            json.dumps(hp_params, indent=4, default=str))

    # For each combination of hyperparameters,
    # use cross validation to evaluate
    return all_hp_params


@parallel_task
def cross_validation(
    payload: TaskPayload, rank: List[int], task_id: int
) -> TaskPayload:
    logger.info(task_id)
    hp_params = payload["hyper_tuning"]
    logger.info(
        f"hp_params : " +
            json.dumps(hp_params, indent=4, default=str)
    )
    logger.info(
        f"{ctx.cv_folds} cross-validation folds"
    )

    strat_kf = StratifiedKFold(n_splits=ctx.cv_folds,
                               shuffle=True,
                               random_state=1121218)
    all_fold_splits = \
        list(strat_kf.split(ctx.X_train, ctx.y_train))

    #################################################
    # Inject DAG execution-context                  #
    # for the herein distributed sub-DAG split line #
    #################################################
    if not ctx.cross_validation:
        ctx.cross_validation = {}
    ctx.cross_validation[str(rank)] = {
        "task_id": task_id,
        "hp": hp_params
    }
    #################################################

    return all_fold_splits

@parallel_task
def training_job(
    payload: TaskPayload, rank: List[int], task_id: int
) -> None:
    from retrain_pipelines.model import PrintLR, WandbCallback

    parent_cv_task_info = ctx.cross_validation[str(rank[:-1])]
    fold_hp_params = parent_cv_task_info['hp']
    logger.info(
        f"cross_validation : {parent_cv_task_info['task_id']}\n" +
        json.dumps(fold_hp_params, indent=4, default=str)
    )
    logger.info(f"CV fold #{rank[-1]+1}/{ctx.cv_folds}")

    if "disabled" != ctx.wandb_run_mode:
        func_name = sys._getframe().f_code.co_name
        parallel_run = wandb.init(
            project=ctx.pipeline_name,
            group=str(ctx.exec_id),
            id=str(task_id),
            name=str(task_id),
            mode=ctx.wandb_run_mode,
            #entity='organization', # default being the entity named
                                    # same as your 'login'
            notes="first attempt",
            tags=["baseline", "dev"],
            job_type=str(parent_cv_task_info["task_id"]),
            # sync_tensorboard=True,
            dir=ctx.wandb_run_dir,  # custom logger directory
                                    # (internals ; local, for online synch)
            config={
                **fold_hp_params,
                **dict(
                    rp_id=ctx.exec_id,
                    rp='_'.join(["cv_task", str(parent_cv_task_info["task_id"])]),
                    rp_task='_'.join([func_name,
                                      str(parent_cv_task_info["task_id"]),
                                      str(task_id)])
                )
            },
            settings=wandb.Settings(
                # saving the env for reproducibility
                # under high workload, saving hits 15s timeout,
                # (even in offline mode)
                # here we disable it
                _save_requirements=False
            ),
            resume=False
        )


    (train_idx, val_idx) = payload
    logger.info(f"task {task_id} - " +
                f"train set : {len(train_idx)}, " +
                f"valid set : {len(val_idx)}")

    X_train, y_train = ctx.X_train.iloc[train_idx], \
                       ctx.y_train.iloc[train_idx]
    X_val, y_val = ctx.X_train.iloc[val_idx], \
                   ctx.y_train.iloc[val_idx]

    # Add eval_set and eval_names to track evaluation results
    eval_set = [(ctx.X_train.values, ctx.y_train),
                (ctx.X_test.values, ctx.y_test)]
    eval_names = ['Training', 'Validation']

    tabnet_clf = TabNetClassifier(
        **fold_hp_params.get('model', {}),
        grouped_features=ctx.grouped_features
    )
    tabnet_clf.fit(
        X_train=X_train.values,
        y_train=y_train,
        loss_fn=CrossEntropyLoss(),
        eval_set=eval_set,
        eval_name=eval_names,
        eval_metric=['accuracy'],
        **fold_hp_params.get('trainer', {}),
        augmentations=None,
        drop_last=False,
        callbacks=(
            [PrintLR()] +
            [WandbCallback()] if 'disabled' != ctx.wandb_run_mode
            else []
        )
    )

    pred_labels = tabnet_clf.predict(X_val.values)
    fold_accuracy = accuracy_score(y_val, pred_labels)

    print(f"Fold finished with accuracy: {fold_accuracy:.5f}.")

    #################################################################

    if "disabled" != ctx.wandb_run_mode:
        parallel_run.log(dict(hp_cv_acc=fold_accuracy))
        # recall that fold_hp_params is a dict of dicts
        cv_input_flattened_dict = \
            flatten_dict(fold_hp_params, callable_to_name=True)
        hp_table = wandb.Table(
            data=[list(cv_input_flattened_dict.values())+[fold_accuracy]],
            columns=list(cv_input_flattened_dict.keys())+['accuracy']
        )
        wandb.log({'Sweep_table': hp_table})
        parallel_run.log(dict(val_accuracy=fold_accuracy))
        parallel_run.finish()

    return fold_accuracy


def avg_scalars(
    scalars_list: List[float]
):
    return float(np.mean(scalars_list))

@task(merge_func=avg_scalars)
def cross_validation_agg(
    payload: TaskPayload, rank: List[int], task_id: int
):
    """
    Aggregates the performance metrics values
    obtained during cross-validation trainings
    for the concerned set of hyperparameter values.
    """

    hp_accuracy = payload*1
    hp_dict = ctx.cross_validation[str(rank)]["hp"]
    logger.info(
        f"hp values {json.dumps(hp_dict, indent=4, default=str)}\n"
        f"lead to an rmse of {hp_accuracy}."
    )

    if not ctx.cross_validation_agg:
        ctx.cross_validation_agg = {}
    ctx.cross_validation_agg[str(rank)] = {
        "task_id": task_id,
        "hp_accuracy": hp_accuracy, # <= to artifact store
    }

    return {
        "rp:cv_task": ctx.cross_validation[str(rank)]["task_id"],
        "rp:rank": str(rank),
        **{**flatten_dict(hp_dict, callable_to_name=True),
           "accuracy": hp_accuracy}
    }


def best_cv_perfs_rank(
    cv_perf_dicts_list: dict
):
    """
    Returns the rank of the
    best-performing set of hyperparameters.
    """
    print(pd.DataFrame(cv_perf_dicts_list)
            .sort_values(by="accuracy", ascending=True)
            .to_string(index=False)
    )

    hp_accuracies = [cv_perf_dict["accuracy"] for cv_perf_dict in cv_perf_dicts_list]

    best_cv_agg_task_idx = np.argmin(hp_accuracies)
    print(f"best_cv_agg_task_idx : {best_cv_agg_task_idx}")
    print(f"best_cv_task_id : "+
          f"{cv_perf_dicts_list[best_cv_agg_task_idx]['rp:cv_task']}")

    best_cv_rank = \
        cv_perf_dicts_list[best_cv_agg_task_idx]["rp:rank"]

    ctx.hp_perfs_list = [
        {k: v for k, v in d.items() if k != "rank"}
        for d in cv_perf_dicts_list
    ]

    return best_cv_rank

@task(merge_func=best_cv_perfs_rank)
def best_hp(
    payload: TaskPayload
):
    """
    Checks performance metric values
    from the different sets of hyperparameter values
    (perf evaluated during hp-tuning)
    and selects the best such set
    for training on the whole training-set next.
    """

    best_cv_rank = payload["cross_validation_agg"]
    print(f"best_cv_rank : {best_cv_rank}")

    # find best hyperparams
    ctx.hyperparameters = ctx.cross_validation[best_cv_rank]["hp"]
    print("hyperparameters : " +
          json.dumps(ctx.hyperparameters, indent=4, default=str)
    )

    return None


@task
def train_model(_, task_id: int):
    """
    A new TabNet model is fitted.
    """
    from retrain_pipelines.model import PrintLR, WandbCallback
    from retrain_pipelines.pipeline_card import plot_masks_to_dict

    model = TabNetClassifier(
        **ctx.hyperparameters.get('model', {}),
        grouped_features=ctx.grouped_features
    )

    # Add eval_set and eval_names to track evaluation results
    eval_set = [(ctx.X_train.values, ctx.y_train),
                (ctx.X_test.values, ctx.y_test)]
    eval_names = ['Training', 'Validation']

    if "disabled" != ctx.wandb_run_mode:
        wandb.join()
        func_name = sys._getframe().f_code.co_name
        training_run = wandb.init(
            project=ctx.pipeline_name,
            group=str(ctx.exec_id),
            id=str(task_id),
            name=str(task_id),
            mode=ctx.wandb_run_mode,
            #entity='organization', # default being the entity
                                    # named same as your 'login'
            notes="first attempt",
            tags=["baseline", "dev"],
            job_type=str(task_id),
            dir=ctx.wandb_run_dir, # custom log directory
                                   # (internals ; local, for online synch)
            config= {
                **ctx.hyperparameters,
                **dict(
                    rp_id=ctx.exec_id,
                    rp='_'.join([func_name, str(task_id)]),
                    rp_task='_'.join([func_name, str(task_id)])
                )
            },
            resume=False,
            settings=wandb.Settings(
                # saving the env for reproducibility
                # under high workload, saving hits 15s timeout,
                # (even in offline mode)
                # here we disable it
                _save_requirements=False
            ),
            # save training code as artifacts
            save_code=True
        )
        # @see https://docs.wandb.ai/ref/python/run#log_code
        wandb_code_root = os.path.realpath(
            os.path.dirname(os.path.abspath(__file__)))
        print(f"wandb log code from : '{wandb_code_root}'")
        training_run.log_code(
            root=wandb_code_root,
            include_fn=lambda path: (
                path.endswith(".py") or
                path.endswith(".html")
            )
        )
        # retrieve WandB URL for this flow run
        # print("WandB UI url : " + wandb.Api().client.app_url)
        # print("WandB entity : " + wandb.Api().viewer._attrs['entity'])
        # print("WandB username : " + wandb.Api().viewer._attrs['username'])
        print("WandB project : " + training_run.project)
        ctx.wandb_project_ui_url = \
            "{wandb_ui_url}{wandb_org}/{wandb_proj}/workspace" \
                .format(
                    wandb_ui_url=wandb.Api().client.app_url,
                    wandb_org=wandb.Api().viewer._attrs['entity'],
                    wandb_proj=training_run.project
                )
        print(ctx.wandb_project_ui_url)
        # retrieve run_id for WandB workspace filtering
        # (needed for cases where pipeline execution is
        #  resumed from a posterior step)
        ctx.wandb_filter_run_id = ctx.exec_id

    model.fit(
        X_train=ctx.X_train.values,
        y_train=ctx.y_train,
        loss_fn=CrossEntropyLoss(),
        eval_set=eval_set, eval_name=eval_names,
        eval_metric=['accuracy'],
        **ctx.hyperparameters.get('trainer', {}),
        augmentations=None,
        drop_last=False,
        callbacks=(
            [PrintLR()] +
            [WandbCallback()] if 'disabled' != ctx.wandb_run_mode
            else []
        )
    )

    if "disabled" != ctx.wandb_run_mode:
        training_run.finish()

    ctx.model = model
    ctx.predictions = model.predict(ctx.X_test.values)

    # training plot
    epochs = range(1, len(model.history['Training_accuracy']) + 1)
    train_accuracy = model.history['Training_accuracy']
    valid_accuracy = model.history['Validation_accuracy']
    learning_rates = model.history['lr']

    # training history
    fig, ax1 = plt.subplots()
    # Plotting train_accuracy and valid_accuracy on the primary y-axis
    ax1.plot(epochs, train_accuracy, '-', label='Training Accuracy',
             color='#4B0082', zorder=2)
    ax1.plot(epochs, valid_accuracy, '-', label='Validation Accuracy',
             color='#008080', zorder=2)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.tick_params(axis='y', labelcolor='black')
    # secondary y-axis for learning rates
    ax2 = ax1.twinx()
    ax2.plot(epochs, learning_rates, 'r-', linewidth=0.4,
             label='Learning Rate', zorder=1)
    ax2.set_ylabel('Learning Rate', color="#d93838")
    ax2.tick_params(axis='y', labelcolor="#d93838")
    # legends (making sure both are above plots' zorder)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    first_legend = plt.legend(handles1, labels1, loc='lower left')
    ax2.add_artist(first_legend)
    ax2.legend(handles2, labels2, loc='upper left')
    # Show grid and plot
    plt.grid(axis='y', linestyle='--', linewidth=0.5,
             color='lightgrey') # horizontal gridlines
    plt.title('Training and Validation Accuracy with Learning Rates')
    fig.tight_layout()
    ctx.training_plt_fig = fig                          # <= to artifact store
    plt.close()

    # model attention layers (as set by 'n_steps') masks
    target_class_figs = plot_masks_to_dict(
        model,
        X_transformed=ctx.X_test,
        grouped_features=ctx.grouped_features,
        raw_feature_names=ctx.X.columns.tolist(),
        y=ctx.y_test
    )
    ctx.target_class_figs = target_class_figs           # <= to artifact store

    return None


@task
def evaluate_model(_, task_id: int):
    #############################################
    # overall (over whole test set) performance #
    #############################################
    accuracy = accuracy_score(ctx.y_test, ctx.predictions)
    precision = precision_score(ctx.y_test, ctx.predictions,
                                average='weighted')
    recall = recall_score(ctx.y_test, ctx.predictions,
                          average='weighted')
    f1 = f1_score(ctx.y_test, ctx.predictions,
                  average='weighted')

    ctx.classes_weighted_metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

    ctx.conf_matrix = confusion_matrix(ctx.y_test, ctx.predictions)
    #############################################

    if "disabled" != ctx.wandb_run_mode:
        func_name = sys._getframe().f_code.co_name
        wandb_flow_run = wandb.init(
            project=ctx.pipeline_name,
            group=str(ctx.exec_id),
            id=str(ctx.exec_id),
            name=str(ctx.exec_id),
            mode=ctx.wandb_run_mode,
            #entity='organization', # default being
                                    # the entity named same as your 'login'
            notes="first attempt",
            tags=["baseline", "dev"],
            job_type=str(ctx.exec_id),
            dir=ctx.wandb_run_dir, # custom log directory
                                   # (internals ; local, for online synch)
            config= {
                **ctx.hyperparameters,
                **dict(
                    rp_id=ctx.exec_id,
                    rp='_'.join([func_name, str(task_id)]),
                    rp_task='_'.join([func_name, str(task_id)])
                )
            },
            settings=wandb.Settings(
                # saving the env for reproducibility
                # under high workload, saving hits 15s timeout,
                # (even in offline mode)
                # here we disable it
                _save_requirements=False
            ),
            resume=True
        )

        wandb_flow_run.log(dict(metrics=ctx.classes_weighted_metrics,
                                # for charts
                                hp_cv_acc= \
                                    ctx.classes_weighted_metrics['accuracy']
        ))

        wandb_flow_run.finish()
        wandb.join()

    #############################################
    #             sliced performance            #
    #############################################
    # as categ_features at this stage are One-Hot encoded,
    # we shall retrieve the column names (from the encoder)
    encoder_file = os.path.join(ctx.serving_artifacts_local_folder,
                                'encoder_params.json')
    with open(encoder_file, "r") as json_file:
        encoder_dict = json.load(json_file)
    if encoder_dict:
        # we choose to report on model performance
        # per slice of the first categorical feature
        first_categ_feature_sliced_metrics = {}
        first_categorical_feature = list(encoder_dict.keys())[0]
        # actual sliced perf computation
        for feature_categ in encoder_dict[first_categorical_feature]:
            feature_column_name = \
                '_'.join([first_categorical_feature, feature_categ])
            # Get test-records indices for this category
            slice_filter = \
                ctx.X_test[feature_column_name].values == 1

            # Calculate metrics for this category
            sliced_y_test = ctx.y_test[slice_filter]
            sliced_predictions = ctx.predictions[slice_filter]

            accuracy = accuracy_score(sliced_y_test, sliced_predictions)
            precision = precision_score(sliced_y_test, sliced_predictions,
                                        average='weighted')
            recall = recall_score(sliced_y_test, sliced_predictions,
                                  average='weighted')
            f1 = f1_score(sliced_y_test, sliced_predictions,
                          average='weighted')

            # Store results in dictionary
            first_categ_feature_sliced_metrics[feature_categ] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }

        ctx.first_categorical_feature = \
                first_categorical_feature
        ctx.first_categ_feature_sliced_metrics = \
                first_categ_feature_sliced_metrics
        print(pd.DataFrame.from_dict(
                    first_categ_feature_sliced_metrics, orient='index'
                ).sort_index(ascending=True))
    else:
        ctx.first_categorical_feature = None
        ctx.first_categ_feature_sliced_metrics = None
    #############################################

    return None


@task
def model_version_blessing(_):
    """
    Compare new model version against best predecessor.
    """

    current_blessed_accuracy = 0
    # find latest blessed model version (from a previous flow-run)
    for execution in ExecutionsIterator(
        exec_name=ctx.pipeline_name, success_only=True,
        page_size=10
    ):
        # print(f"execution {execution.id} - {execution.get_attr('model_version_blessed')}")
        if execution.get_attr("model_version_blessed"):
            ctx.current_blessed_exec = execution
            current_blessed_accuracy = float(
                execution.get_attr("classes_weighted_metrics")["accuracy"])
            ctx.model_version_blessed = \
                ctx.classes_weighted_metrics['accuracy'] >= current_blessed_accuracy
            print("new : " + str(ctx.classes_weighted_metrics['accuracy']) +
                  " - previous best : " + str(current_blessed_accuracy) +
                  " - model_version_blessing : " +
                      str(ctx.model_version_blessed))
            break

    # ctx.model_version_blessed = False ### DEBUG - DELETE ###

    # case 'no prior blessed run'
    if current_blessed_accuracy == 0:
        print("case 'no prior blessed model version found => blessing.'")
        ctx.model_version_blessed = True

    return None


@task
def infra_validator(_):
    """
    If the trained model version is blessed, validate serving.
    """
    """
    Note that we could embark the whole task dependencies
    into the local TorchServe server.
    That's actually unnecessary and lengthens
    the docker image build time.
    But we could doi so for generalizability purpose
    of the sample pipeline.
    In the herein case, an empty "requirements.txt"
    actually suffices since all necessary dependencies
    are already included in the base image.
    Uncomment the dependency code snippet if you e.g.
    update the preprocessing module with external lib
    since hard-coding a list it here would be unwise.
    """

    # tracking below using a 3-states var
    # -1 for not applicable, and
    # 0/1 bool for failure/success otherwize
    ctx.local_serve_is_ready = -1

    if ctx.model_version_blessed:
        # serialize model as artifact
        model_file = os.path.join(
            ctx.serving_artifacts_local_folder, "model")
        ctx.model.save_model(model_file)

        preprocess_module_dir = \
            os.path.dirname(
                importlib.util.find_spec(
                    f"retrain_pipelines.model.{os.getenv('retrain_pipeline_type')}"
                ).origin)
        # save Dockerfile.torchserve as artifact
        shutil.copy(
            os.path.join(
                preprocess_module_dir,
                'Dockerfile.torchserve'),
            os.path.join(ctx.serving_artifacts_local_folder,
                         'Dockerfile.torchserve'))
        # save TabNet TorchServe handler class as artifact
        shutil.copy(
            os.path.join(
                preprocess_module_dir,
                'torchserve_tabnet_handler.py'),
            os.path.join(ctx.serving_artifacts_local_folder,
                         'torchserve_tabnet_handler.py'))
        # save dependencies as artifact
        open(
            f"{ctx.serving_artifacts_local_folder}/requirements.txt",
            "w").close() # just create an empty one here
        # create_requirements(ctx.serving_artifacts_local_folder,
                            # exclude=[
                                # "matplotlib", "pillow", # version conflict
                                                        # # quick fix
                                # "torch", # already present
                                         # # in the torchserve
                                         # # Docker base image
                            # ]
        # )

        os.environ["no_proxy"] = "localhost,127.0.0.1,0.0.0.0"

        ############################################
        #  actually deploy the prediction service  #
        ############################################
        start_time = time.time()
        from retrain_pipelines.utils.docker import \
            build_and_run_docker, print_container_log_tail, \
            cleanup_docker
        from retrain_pipelines.model.torchserve import \
            await_server_ready, server_has_model, \
            endpoint_still_starting, endpoint_is_ready

        serving_container = build_and_run_docker(
            image_name="tabnet_serve", image_tag="1.0",
            build_path=ctx.serving_artifacts_local_folder,
            dockerfile="Dockerfile.torchserve",
            ports_publish_dict={
                '8080/tcp': 9080, '8081/tcp': 9081,
                '8082/tcp': 9082}
        )
        if not serving_container:
            print("failed spinning the TorchServe container",
                  file=sys.stderr)
            ctx.local_serve_is_ready = 0
            try:
                cleanup_docker(container_name="tabnet_serve",
                               image_name="tabnet_serve:1.0")
            except Exception as cleanup_ex:
                # fail silently
                pass
        elif not await_server_ready():
            print("TorchServe server failed to get ready",
                  file=sys.stderr)
            ctx.local_serve_is_ready = 0
        elif not server_has_model("tabnet_model"):
            print("TorchServe server didn't publish the model",
                  file=sys.stderr)
            ctx.local_serve_is_ready = 0
        else:
            print("Awaiting endpoint launch..")
            timeout = 10*60
            start_time = time.time()
            while endpoint_still_starting("tabnet_model"):
                if time.time() - start_time > timeout:
                    print(
                        f"The endpoint 'tabnet_model' did not start " +
                        f"within {timeout} seconds.")
                    print_container_log_tail("tabnet_serve", 50)
                    ctx.local_serve_is_ready = 0
                    break
                time.sleep(4)
            elapsed_time = time.time() - start_time
            print(f"Elapsed time: {elapsed_time:.3f} seconds")

            # health check on the spun endpoint
            ctx.local_serve_is_ready = int(
                endpoint_is_ready("tabnet_model")
            )
        elapsed_time = time.time() - start_time
        print(f"deploy_local -   Elapsed time: {elapsed_time:.2f} seconds")
        ############################################

        if ctx.local_serve_is_ready == 1:
            from pydantic import ValidationError

            from retrain_pipelines.model import endpoint_test

            start_time = time.time()
            raw_inference_request_items = [
                ["value1", 1, 2, 3, 4],
                ["value2", 5, 6, 7, 8]
            ]
            try:
                endpoint_response = \
                    endpoint_test.parse_endpoint_response(
                        "tabnet_model", raw_inference_request_items)
                # Compare the length of request and response items
                # (response type is already validated
                #  by enforced pydantic model)
                assert len(endpoint_response) == \
                        len(raw_inference_request_items), \
                       "Validation Error: The number of " \
                       "response items does not match " \
                       "the number of request items."
                ctx.local_serve_is_ready = 1
                elapsed_time = time.time() - start_time
                print(f"inference test -   "
                      f"Elapsed time: {elapsed_time:.2f} seconds")
            except ValidationError as vEx:
                print(vEx.errors(), file=sys.stderr)
                print(vEx, file=sys.stderr)
                traceback.print_tb(vEx.__traceback__, file=sys.stderr)
                ctx.local_serve_is_ready = 0
            except Exception as ex:
                print(ex, file=sys.stderr)
                traceback.print_tb(ex.__traceback__, file=sys.stderr)
                ctx.local_serve_is_ready = 0
                pass
        try:
            cleanup_docker(container_name="tabnet_serve",
                           image_name="tabnet_serve:1.0")
        except Exception as cleanup_ex:
            # fail silently
            pass


@task
def pipeline_card(_, task_id: int):
    import datetime

    ###########################
    # user-provided artifacts #
    ###########################
    # note that user can provide either
    # 'pipeline_card.py' or 'template.html'
    # or both when specifying custom
    # 'pipeline_card_artifacts_path'
    if "template.html" in os.listdir(ctx.pipeline_card_artifacts_path):
        template_dir = ctx.pipeline_card_artifacts_path
    else:
        template_dir = os.path.dirname(
            importlib.util.find_spec(
                f"retrain_pipelines.pipeline_card."+
                f"{os.getenv('retrain_pipeline_type')}"
            ).origin)
    logger.debug(f"template_dir : {template_dir}")
    ###########################
    if "pipeline_card.py" in os.listdir(ctx.pipeline_card_artifacts_path):
        from retrain_pipelines.utils import get_get_html
        get_html = \
            get_get_html(ctx.pipeline_card_artifacts_path)
    else:
        from retrain_pipelines.pipeline_card import get_html
    from retrain_pipelines.dag_engine.renderer import dag_svg
    ###########################

    ###########################
    ##   html "custom" card  ##
    ###########################
    classes_prior_prob = ctx.data['target'].value_counts(normalize=True)
    classes_prior_prob = re.sub(
            r'\s+', ', ',
            classes_prior_prob.sort_index().map(
                lambda x: f"{x*100:.1f}%").to_frame() \
                                .reset_index() \
                                .to_string(index=False,header=False)
        )

    dt = datetime.datetime.now(tz=datetime.timezone.utc)
    formatted_dt = dt.strftime("%A %b %d %Y %I:%M:%S %p %Z")
    task_obj_python_cmd = f"sdk.Task({task_id})"
    executions_count = ExecutionsIterator(
        exec_name=ctx.pipeline_name).length()

    params={
        'template_dir': template_dir,
        'title': ctx.pipeline_name,
        "subtitle": f"(Pipeline execution # {executions_count}," + \
                    f" exec_id: {str(ctx.exec_id)}  -  {formatted_dt})",
        'model_version_blessed': ctx.model_version_blessed,
        'current_blessed_run': ctx.current_blessed_exec,
        'local_serve_is_ready': ctx.local_serve_is_ready,
        'records_count': ctx.data.shape[0],
        'classes_prior_prob': classes_prior_prob,
        'features_desc': ctx.features_desc,
        'data_distri_fig': ctx.data_distri_fig,
        'data_heatmap_fig': ctx.data_heatmap_fig,
        'training_plt_fig': ctx.training_plt_fig,
        'target_class_figs': ctx.target_class_figs,
        'classes_weighted_metrics_dict': ctx.classes_weighted_metrics,
        'slice_feature_name': ctx.first_categorical_feature,
        'sliced_perf_metrics_dict': \
            ctx.first_categ_feature_sliced_metrics,
        'buckets_dict': ctx.buckets_param,
        'hyperparameters_dict': flatten_dict(ctx.hyperparameters,
                                             callable_to_name=True),
        'wandb_project_ui_url': (
            ctx.wandb_project_ui_url
            if 'disabled' != ctx.wandb_run_mode else None),
        'wandb_filter_run_id': (
            ctx.wandb_filter_run_id
            if 'disabled' != ctx.wandb_run_mode else None),
        'wandb_need_sync_dir': (
            "" if "offline" != ctx.wandb_run_mode
            else ctx.wandb_run_dir +
                 os.path.sep.join(['', 'wandb', 'offline-*'])
        ),
        'pipeline_hp_grid_dict': flatten_dict(ctx.pipeline_hp_grid,
                                              callable_to_name=True),
        'cv_folds': ctx.cv_folds,
        'hp_perfs_list': ctx.hp_perfs_list,
        'task_obj_python_cmd': task_obj_python_cmd,
        'dag_svg': dag_svg(execution_id=ctx.exec_id)
    }
    html = get_html(params)

    filename = os.path.join(
        os.environ["RP_ARTIFACTS_STORE"],
        ctx.pipeline_name, str(ctx.exec_id),
        "pipeline_card.html"
    )
    with open(filename, "w", encoding="utf-8") as file:
        file.write(html)
    logger.debug(
        "pipeline_card - " +
        f"[bold]pipeline_card_file_fullname : {filename}[/]")

    ctx.pipeline_card_file_fullname = filename
    ###########################

    return None


@task
def deploy(_):
    """
    placeholder for the serving SDK deploy call
    (on the target production platform).
    consider including the portable pipelione-card itself !
    """

    if ctx.model_version_blessed and (ctx.local_serve_is_ready == 1):
        pass # your code here

    return None


@task
def load_test(_):
    """
    placeholder
    """

    if ctx.model_version_blessed and (ctx.local_serve_is_ready == 1):
        pass # your code here

    return None


@task
def end(payload: TaskPayload):
    # ctx.wandb_flow_run.finish()
    # wandb.join()
    pass

    if "offline" == ctx.wandb_run_mode:
        logger.info(
            "Note that the herein flow run was not live-synced to WandB.\n"+
            "Execute the following WandB CLI command locally to sync it\n"+
            "so logger-data shows there for this flow run :\n"+
            "wandb sync {wandb_need_sync_dir}".format(
                wandb_need_sync_dir=ctx.wandb_run_dir +
                    os.path.sep.join(['', 'wandb', 'offline-*'])
            )
        )


@dag(ui_css=UiCss(color="#0E7600", background="#FDFF83", border="#CBFFC4"))
def retrain_pipeline():
    """
    Retraining pipeline with hyperparameters tuning with cross validation. 1 target variable, classification model from tabular data.
    PyTorch TabNet architexture. Model-version blessing. Inference pipeline served over TorchServe.
    """

    #--- flow parameters -------------------------------------------------------


    RETRAIN_PIPELINE_TYPE = "mf_tabnet_classif_torchserve"
    # best way to share the config across subprocesses
    os.environ["retrain_pipeline_type"] = RETRAIN_PIPELINE_TYPE


    data_file_fullname = DagParam(
        description="Path to the input data file",
        default=os.path.realpath(
            os.path.join(os.path.dirname(__file__), "..", "data",
                         "synthetic_classif_tab_data_4classes.csv"))
    )
    buckets_param = DagParam(
        description="Bucketization to be applied " + \
                    "on raw numerical feature(s). " + \
                    "dict of optional pairs of  " + \
                    "feature_name/buckets_count.",
        default="{}"
    )
    pipeline_hp_grid = DagParam(
        description="TabNet model hyperparameters domain " + \
                    "(for both the model and the trainer)",
        default=dedent("""        {
            "trainer": {
                "max_epochs": [100],
                "patience": [10],
                "batch_size": [1024],
                "virtual_batch_size": [512]
            },
            "model": {
                "n_d": [64],
                "n_a": [64],
                "n_steps": [5],
                "gamma":[1.5],
                "n_independent":[2],
                "n_shared":[2],
                "lambda_sparse":[0.0001],
                "momentum":[0.3],
                "clip_value":[2.0],
                "optimizer_fn":["torch.optim.Adam"],
                "optimizer_params":[{"lr": 0.02}],
                "scheduler_params":[{"gamma": 0.80,
                                     "step_size": 20}],
                "scheduler_fn":["torch.optim.lr_scheduler.StepLR"],
                "epsilon":[0.000000000000001]
            }
        }""")
    )
    cv_folds = DagParam(
        description="(int) how many Cross Validation folds shall be used.",
        default=2
    )
    wandb_run_mode = DagParam(
        description="WandB mode for the flow-run " + \
                    "indicating whether to sync it to the wandb server " + \
                    "can be either 'disabled', 'offline', or 'online'.",
        default="online"
    )

    # TorchServe artifacts location
    # (i.e. dir hosting preprocessing.py)
    default_preprocess_module_dir = \
        os.path.dirname(
            importlib.util.find_spec(
                f"retrain_pipelines.model."
                f"{RETRAIN_PIPELINE_TYPE}"
            ).origin)
    preprocess_artifacts_path = DagParam(
        description="TorchServe artifacts location " + \
                    "(i.e. dir hosting your custom 'preprocessing.py'" + \
                    " file), if different from default",
        default=default_preprocess_module_dir
    )
    # TODO  -  convert from class method to TBD
    # @staticmethod
    # def _get_default_preprocess_artifacts_path() -> str:
        # return TabNetHpCvWandbFlow.default_preprocess_module_dir
    # @staticmethod
    # def copy_default_preprocess_module(
        # target_dir: str,
        # exists_ok: bool = False
    # ) -> None:
        # os.makedirs(target_dir, exist_ok=True)
        # if (
            # not exists_ok and
            # os.path.exists(os.path.join(target_dir, "preprocessing.py"))
        # ):
            # print("File already exists. Skipping copy.")
        # else:
            # filefullname = os.path.join(
                    # TabNetHpCvWandbFlow._get_default_preprocess_artifacts_path(),
                    # "preprocessing.py"
                # )
            # shutil.copy(filefullname, target_dir)
            # print(filefullname)

    default_pipeline_card_module_dir = \
        os.path.dirname(
            importlib.util.find_spec(
                f"retrain_pipelines.pipeline_card."+
                f"{RETRAIN_PIPELINE_TYPE}"
            ).origin)
    pipeline_card_artifacts_path = DagParam(
        description="pipeline_card artifacts location " + \
                    "(i.e. dir hosting your custom 'pipeline_card.py'" + \
                    " and/or 'template.html' file)," + \
                    " if different from default",
        default=default_pipeline_card_module_dir
    )

    # TODO  -  convert from class method to TBD
    # @staticmethod
    # def _get_default_pipeline_card_module_dir() -> str:
        # return TabNetHpCvWandbFlow.default_pipeline_card_module_dir
    # @staticmethod
    # def copy_default_pipeline_card_module(
        # target_dir: str,
        # exists_ok: bool = False
    # ) -> None:
        # os.makedirs(target_dir, exist_ok=True)
        # if (
            # not exists_ok and
            # os.path.exists(os.path.join(target_dir, "pipeline_card.py"))
        # ):
            # print("File already exists. Skipping copy.")
        # else:
            # filefullname = os.path.join(
                    # TabNetHpCvWandbFlow._get_default_pipeline_card_module_dir(),
                    # "pipeline_card.py"
                # )
            # shutil.copy(filefullname, target_dir)
            # print(filefullname)
    # TODO  -  convert from class method to TBD
    # @staticmethod
    # def copy_default_pipeline_card_html_template(
        # target_dir: str,
        # exists_ok: bool = False
    # ) -> None:
        # os.makedirs(target_dir, exist_ok=True)
        # if (
            # not exists_ok and
            # os.path.exists(os.path.join(target_dir, "template.html"))
        # ):
            # print("File already exists. Skipping copy.")
        # else:
            # filefullname = os.path.join(
                    # TabNetHpCvWandbFlow._get_default_pipeline_card_module_dir(),
                    # "template.html")
            # shutil.copy(filefullname, target_dir)
            # print(filefullname)

    del RETRAIN_PIPELINE_TYPE

    #---------------------------------------------------------------------------


    return start >> data >> split_data >> hyper_tuning \
            >> cross_validation >> training_job \
            >> cross_validation_agg >> best_hp \
            >> train_model >> evaluate_model \
            >> model_version_blessing >> infra_validator \
            >> pipeline_card >> deploy >> load_test \
            >> end


















































