
import os
import sys

import re
import json
import time
import shutil
import logging
import warnings
import itertools
import importlib.util

from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, \
                                    KFold
from sklearn.metrics import root_mean_squared_error, \
                            mean_absolute_error, r2_score
from sklearn.preprocessing import OrdinalEncoder

import wandb
from wandb.integration.lightgbm import log_summary

import lightgbm as lgb


from retrain_pipelines import __version__
from retrain_pipelines.dag_engine.core import \
    TaskPayload, task, taskgroup, parallel_task, \
    dag, DagParam, ctx, UiCss

from retrain_pipelines.dataset import features_desc, \
    features_distri_plot
from retrain_pipelines.dataset.features_dependencies import \
    dataset_to_heatmap_fig
from retrain_pipelines.utils import create_requirements

from retrain_pipelines.dag_engine.sdk import \
    ExecutionsIterator

from retrain_pipelines.utils import grant_read_access, \
    tmp_os_environ


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@task
def start() -> TaskPayload:
    logger.info(f"{ctx.pipeline_name} - {ctx.exec_id}")
    logging.getLogger("retrain_pipelines").setLevel(logging.INFO)

    # inputs validation
    assert os.path.exists(os.path.realpath(ctx.data_file_fullname))
    ctx.cv_folds = int(ctx.cv_folds)
    ctx.dask_partitions = int(ctx.dask_partitions)
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


    # preemptive dask/distributed/lightgbm hard-reset
    # from potential prior execution on the same
    # python interpreter as the current one
    from retrain_pipelines.model import nuke_dask_cpp
    nuke_dask_cpp()


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

    encoder = OrdinalEncoder()
    buckets = ctx.buckets_param.copy()

    data = payload
    ctx.X = data.iloc[:, :-1]
    ctx.y = data.iloc[:, -1]

    X_transformed = preprocess_data_fct(
        ctx.X, encoder, buckets,
        is_training=True,
        local_path=ctx.serving_artifacts_local_folder
    )
    ctx.encoder = encoder # <= to artifact store
    ctx.buckets = buckets # <= to artifact store

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
            X_transformed, ctx.y, test_size=0.2, random_state=42
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
    all_hp_params = [
        dict(zip(ctx.pipeline_hp_grid.keys(), v))
        for v in itertools.product(*ctx.pipeline_hp_grid.values())
    ]
    for i, hp_params in enumerate(all_hp_params):
        logger.info(f"hp_params[{i}] : {hp_params}")

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
        f"hp_params : {hp_params},\n" +
        f"{ctx.cv_folds} cross-validation folds."
    )

    # recall that the respective proportion
    # of datapoints between train & valid sets
    # depends on n_splits
    kf = KFold(n_splits=ctx.cv_folds,
               shuffle=True,
               random_state=1121218)
    # list of one such pair per fold
    all_fold_splits = list(kf.split(X=ctx.X_train))

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
    from retrain_pipelines.model import dask_regressor_fit

    parent_cv_task_info = ctx.cross_validation[str(rank[:-1])]
    fold_hp_params = parent_cv_task_info['hp']
    logger.info(f"cross_validation : {parent_cv_task_info['task_id']}" +
             f" - {fold_hp_params}")
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
                _save_requirements=False,
                # console="wrap",  # or "off" to disable console output entirely
                # quiet=True,
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

    old_filters = warnings.filters[:]
    old_level = logging.root.level
    warnings.simplefilter("ignore")
    logging.getLogger("bokeh.server.util").setLevel(logging.ERROR)
    fold_lgb_reg, fold_history, fold_workers_history = \
        dask_regressor_fit(
            X_train, y_train, X_val, y_val,
            npartitions=ctx.dask_partitions,
            hp_dict=fold_hp_params
        )
    warnings.filters[:] = old_filters
    logging.getLogger("bokeh.server.util").setLevel(old_level)

    if not ctx.training_job:
        ctx.training_job = {}
    ctx.training_job[str(rank)] = {
        "task_id": task_id,
        "fold_workers_history": fold_workers_history, # <= to artifact store
        "fold_lgb_reg": fold_lgb_reg                  # <= to artifact store
    }

    # rmse
    preds = fold_lgb_reg.predict(X_val)
    fold_rmse = root_mean_squared_error(y_val, preds)
    logger.info(f"Fold finished with rmse: {fold_rmse:.5f}.\n"+
                f"(on last distributed epoch :"+
                f"{fold_history['Validation']['rmse'][-1]})")

    if "disabled" != ctx.wandb_run_mode:
        for step in range(len(fold_history['Training']['rmse'])):
            wandb.log({
                'Step': step + 1,  # +1 to start the step count from 1
                'Training_rmse': fold_history['Training']['rmse'][step],
                'Validation_rmse': fold_history['Validation']['rmse'][step],
                'Training_l2': fold_history['Training']['l2'][step],
                'Validation_l2': fold_history['Validation']['l2'][step]
            })

        log_summary(
            model=fold_lgb_reg.booster_,
            feature_importance=True,
            save_model_checkpoint=False
        )

        parallel_run.log(dict(hp_cv_rmse=fold_rmse))
        hp_table = wandb.Table(
            data=[list(fold_hp_params.values())+
                  [fold_rmse]],
            columns=list(fold_hp_params.keys())+\
                    ['rmse']
        )
        wandb.log({'Sweep_table': hp_table})
        parallel_run.finish()

    return fold_rmse


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

    hp_rmse = payload*1
    hp_dict = ctx.cross_validation[str(rank)]["hp"]
    logger.info(
        f"hp values {hp_dict} "
        f"lead to an rmse of {hp_rmse}."
    )

    if not ctx.cross_validation_agg:
        ctx.cross_validation_agg = {}
    ctx.cross_validation_agg[str(rank)] = {
        "task_id": task_id,
        "hp_rmse": hp_rmse, # <= to artifact store
    }

    return {
        "rp:cv_task": ctx.cross_validation[str(rank)]["task_id"],
        "rp:rank": str(rank),
        **{**hp_dict,
           "rmse": hp_rmse}
    }


def best_cv_perfs_rank(
    cv_perf_dicts_list: dict
):
    """
    Returns the rank of the
    best-performing set of hyperparameters.
    """
    print(pd.DataFrame(cv_perf_dicts_list)
            .sort_values(by="rmse", ascending=True)
            .to_string(index=False)
    )

    hp_rmses = [cv_perf_dict["rmse"] for cv_perf_dict in cv_perf_dicts_list]

    best_cv_agg_task_idx = np.argmin(hp_rmses)
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
    print(f"hyperparameters : {ctx.hyperparameters}")

    return None


@task
def train_model(_, task_id: int):
    """
    A new LightGBM model is fitted.
    """
    from retrain_pipelines.model import dask_regressor_fit

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
        # print("WandB project : " + training_run.project)
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


    model, history, workers_history = \
        dask_regressor_fit(
            ctx.X_train, ctx.y_train,
            ctx.X_test, ctx.y_test,
            npartitions = ctx.dask_partitions,
            hp_dict=ctx.hyperparameters
        )

    if not ctx.train_model:
        ctx.train_model = {}
    ctx.train_model = {
        "task_id": task_id,
        "workers_history": workers_history,             # <= to artifact store
        "history": history                              # <= to artifact store
    }

    if "disabled" != ctx.wandb_run_mode:
        for step in range(len(history['Training']['rmse'])):
            wandb.log({
                'Step': step + 1,  # +1 to start the step count from 1
                'Training_rmse': history['Training']['rmse'][step],
                'Validation_rmse': history['Validation']['rmse'][step],
                'Training_l2': history['Training']['l2'][step],
                'Validation_l2': history['Validation']['l2'][step]
            })

        log_summary(
            model=model.booster_,
            feature_importance=True,
            save_model_checkpoint=False
        )
        training_run.finish()

    ctx.train_model.update({
        "model": model,                                 # <= to artifact store
        "predictions": model.predict(ctx.X_test)        # <= to artifact store
    })

    # --- Generate training plots ---
    num_rounds = model.get_params()['n_estimators']
    train_loss = history['Training']['l2']
    valid_loss = history['Validation']['l2']
    epochs = np.arange(1, num_rounds + 1)

    # training history
    fig = plt.figure()
    plt.plot(epochs, train_loss, label='Training Loss',
             color='#4B0082')
    plt.plot(epochs, valid_loss, label='Validation Loss',
             color='#008080')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(axis='y', linestyle='--', linewidth=0.5,
             color='lightgrey') # horizontal gridlines
    ctx.training_plt_fig = fig                          # <= to artifact store
    plt.close()

    # feature importance
    fig, ax = plt.subplots()#figsize=(10, 6))
    lgb.plot_importance(model, ax=ax, max_num_features=10,
                        grid=False, ignore_zero=False,
                        color='purple', zorder=3)
    ax.grid(axis='x', linestyle='--', color='lightgrey',
            zorder=2) # vertical gridlines
    # clear x-axis label and ticks labels
    ax.set_xlabel('')
    for label in ax.get_xticklabels(): label.set_visible(False)
    # Remove labels inside the bars
    for text in ax.texts: text.set_visible(False)
    # Adjust y-axis tick labels
    for label in ax.get_yticklabels():
        label.set_rotation(60)
        label.set_verticalalignment('center')
    ax.set_title("Feature Importance")
    fig.tight_layout()
    ctx.features_plt_fig = fig                          # <= to artifact store
    plt.close()

    return None


@task
def evaluate_model(_, task_id: int):
    #############################################
    # overall (over whole test set) performance #
    #############################################
    rmse = root_mean_squared_error(ctx.y_test, ctx.train_model["predictions"])
    mae  = mean_absolute_error(ctx.y_test, ctx.train_model["predictions"])
    r2 = r2_score(ctx.y_test, ctx.train_model["predictions"])

    ctx.metrics = {
        "rmse": rmse,
        "mae ": mae,
        "r2": r2
    }

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

        wandb_flow_run.log(dict(metrics=ctx.metrics,
                                # and, for charts
                                hp_cv_rmse=ctx.metrics['rmse']
        ))

        wandb_flow_run.finish()
        wandb.join()
    #############################################

    #############################################
    #             sliced performance            #
    #############################################
    if "feature_names_in_" in ctx.encoder.__dict__:
        categ_features_arr = \
            ctx.encoder.__dict__['feature_names_in_']
        first_categorical_feature = \
                        categ_features_arr[0]
        first_categ_feature__labels = \
            ctx.encoder.__dict__['categories_'][0]

        first_categ_feature_sliced_metrics = {}
        # actual sliced perf computation
        for feature_categ in ctx.X_test[first_categorical_feature].unique():
            # Get test-records indices for this category
            slice_filter = \
                ctx.X_test[first_categorical_feature].values == feature_categ

            # Calculate metrics for this category
            sliced_y_test = ctx.y_test[slice_filter]
            sliced_predictions = ctx.train_model["predictions"][slice_filter]

            rmse = root_mean_squared_error(sliced_y_test, sliced_predictions)
            mae  = mean_absolute_error(sliced_y_test, sliced_predictions)
            r2 = r2_score(sliced_y_test, sliced_predictions)

            # Store results in dictionary
            first_categ_feature__label = \
                first_categ_feature__labels[int(feature_categ)]
            first_categ_feature_sliced_metrics[first_categ_feature__label] = {
                "rmse": rmse,
                "mae ": mae,
                "r2": r2
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

    current_blessed_rmse = sys.maxsize
    # find latest blessed model version (from a previous flow-run)
    for execution in ExecutionsIterator(
        exec_name=ctx.pipeline_name, success_only=True,
        page_size=10
    ):
        # print(f"execution {execution.id} - {execution.get_attr('model_version_blessed')}")
        if execution.get_attr("model_version_blessed"):
            ctx.current_blessed_exec = execution
            current_blessed_rmse = \
                float(execution.get_attr("metrics")["rmse"])
            ctx.model_version_blessed = \
                ctx.metrics['rmse'] <= current_blessed_rmse
            print("new : " + str(ctx.metrics['rmse']) +
                  " - previous best : " + str(current_blessed_rmse) +
                  " - model_version_blessing : " +
                      str(ctx.model_version_blessed))
            break

    # ctx.model_version_blessed = False ### DEBUG - DELETE ###

    # case 'no prior blessed run'
    if current_blessed_rmse == sys.maxsize:
        print("case 'no prior blessed model version found => blessing.'")
        ctx.model_version_blessed = True

    return None


@task
def infra_validator(_):
    """
    If the trained model version is blessed, validate serving.
    """
    """
    Note that we embark the whole pipeline dependencies
    into the local MLServer server.
    That's actually unnecessary and lengthens
    the docker image build time.
    But we keep doing so for generalizability purpose
    of the sample pipeline.
    In the herein case, only below dependencies are
    strictly required for the inference service :
      - mlserver==1.3.5
      - mlserver-lightgbm==1.3.5
      - lightgbm==4.3.0
      - numpy==1.26.4
      - uvloop==0.17.0
    """
    # tracking below using a 3-states var
    # -1 for not applicable, and
    # 0/1 bool for failure/success otherwise
    ctx.local_serve_is_ready = -1

    if ctx.model_version_blessed:
        # serialize model version
        model_file = os.path.join(ctx.serving_artifacts_local_folder,
                                  "model.txt")
        ctx.train_model["model"].booster_.save_model(model_file)

        preprocess_module_dir = \
            os.path.dirname(
                importlib.util.find_spec(
                    f"retrain_pipelines.model.{os.getenv('retrain_pipeline_type')}"
                ).origin)
        # save Dockerfile.mlserver as artifact
        shutil.copy(
            os.path.join(
                preprocess_module_dir,
                'Dockerfile.mlserver'),
            os.path.join(ctx.serving_artifacts_local_folder,
                         'Dockerfile.mlserver'))
        # save LightGBM Regressor MLServer handler class as artifact
        shutil.copy(
            os.path.join(
                preprocess_module_dir,
                'mlserver_lightgbm_reg_handler.py'),
            os.path.join(ctx.serving_artifacts_local_folder,
                         'mlserver_lightgbm_reg_handler.py'))
        # save MLServer settings as artifact
        shutil.copy(
            os.path.join(
                preprocess_module_dir,
                'settings.json'),
            os.path.join(ctx.serving_artifacts_local_folder,
                         'settings.json'))
        shutil.copy(
            os.path.join(
                preprocess_module_dir,
                'model-settings.json'),
            os.path.join(ctx.serving_artifacts_local_folder,
                         'model-settings.json'))
        # save dependencies as artifact
        create_requirements(ctx.serving_artifacts_local_folder,
                            exclude=[
                                "retrain-pipelines",
                                "tritonclient", "pydantic.*"
                            ]
        )

        os.environ["no_proxy"] = "localhost,127.0.0.1,0.0.0.0"


        ############################################
        #  actually deploy the prediction service  #
        ############################################
        start_time = time.time()
        from retrain_pipelines.utils.docker import \
            build_and_run_docker, print_container_log_tail, \
            cleanup_docker
        from retrain_pipelines.model.mlserver import \
            await_server_ready, server_has_model, \
            endpoint_still_starting, endpoint_is_ready

        serving_container = build_and_run_docker(
            image_name="lgbm_reg_serve", image_tag="1.0",
            build_path=ctx.serving_artifacts_local_folder,
            dockerfile="Dockerfile.mlserver",
            ports_publish_dict={'8080/tcp': 9080}
        )
        if not serving_container:
            logger.warning("failed spinning the MLServer container")
            ctx.local_serve_is_ready = 0
            try:
                cleanup_docker(container_name="lgbm_reg_serve",
                               image_name="lgbm_reg_serve:1.0")
            except Exception as cleanup_ex:
                # fail silently
                pass
        elif not await_server_ready():
            print("MLServer server failed to get ready",
                  file=sys.stderr)
            print_container_log_tail("lgbm_reg_serve", 50)
            ctx.local_serve_is_ready = 0
        elif not server_has_model("lightgbm-model"):
            print("MLServer server didn't publish the model",
                  file=sys.stderr)
            ctx.local_serve_is_ready = 0
        else:
            print("Awaiting endpoint launch..")
            timeout = 10*60
            start_time = time.time()
            while endpoint_still_starting("lightgbm-model"):
                if time.time() - start_time > timeout:
                    print(
                        f"The endpoint 'lightgbm-model' did not start " +
                        f"within {timeout} seconds.")
                    print_container_log_tail("lgbm_reg_serve", 50)
                    ctx.local_serve_is_ready = 0
                    break
                time.sleep(4)
            elapsed_time = time.time() - start_time
            print(f"Elapsed time: {elapsed_time:.3f} seconds")

            # health check on the spun endpoint
            ctx.local_serve_is_ready = int(
                endpoint_is_ready("lightgbm-model")
            )
        elapsed_time = time.time() - start_time
        print(f"deploy_local -   Elapsed time: {elapsed_time:.2f} seconds")
        ############################################

        if ctx.local_serve_is_ready == 1:
            from pydantic import ValidationError
            import traceback

            from retrain_pipelines.model import endpoint_test

            start_time = time.time()
            raw_inference_request_items = {
                "inputs": [
                    {
                        "name": "categ_feature0",
                        "shape": [3],
                        "datatype": "BYTES",
                        "data": ["value1", "value1", "value1"]
                    },
                    {
                        "name": "num_feature1",
                        "shape": [3],
                        "datatype": "FP32",
                        "data": [0.1, 0.1, 0.1]
                    },
                    {
                        "name": "num_feature2",
                        "shape": [3],
                        "datatype": "FP32",
                        "data": [0.1, 0.1, 0.1]
                    },
                    {
                        "name": "num_feature3",
                        "shape": [3],
                        "datatype": "FP32",
                        "data": [3.1, 3.1, 3.1]
                    },
                    {
                        "name": "num_feature4",
                        "shape": [3],
                        "datatype": "FP32",
                        "data": [0.2, 0.2, 0.2]
                    }
                ]
            }
            try:
                endpoint_response = \
                    endpoint_test.parse_endpoint_response(
                        "lightgbm-model", raw_inference_request_items)
                # Compare the length of request and response items
                # (response type is already validated
                #  by enforced pydantic model)
                assert endpoint_response and \
                       len(endpoint_response) == \
                            len(raw_inference_request_items
                                    ["inputs"][0]["data"]), \
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
            cleanup_docker(container_name="lgbm_reg_serve",
                           image_name="lgbm_reg_serve:1.0")
        except Exception as cleanup_ex:
            # fail silently
            pass
    else:
        logger.info("skipped")

    return None


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
        'features_desc': ctx.features_desc,
        'data_distri_fig': ctx.data_distri_fig,
        'data_heatmap_fig': ctx.data_heatmap_fig,
        'training_plt_fig': ctx.training_plt_fig,
        'features_plt_fig': ctx.features_plt_fig,
        'perf_metrics_dict': ctx.metrics,
        'slice_feature_name': ctx.first_categorical_feature,
        'sliced_perf_metrics_dict': \
            ctx.first_categ_feature_sliced_metrics,
        'buckets_dict': ctx.buckets_param,
        'hyperparameters_dict': ctx.hyperparameters,
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
        'pipeline_hp_grid_dict': ctx.pipeline_hp_grid,
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


@dag(ui_css=UiCss(color="#087292", background="#FFC47A", border="#FFD700"))
def retrain_pipeline():
    """
    Retraining pipeline with Dask-distributed hyperparameters tuning & cross validation. 1 target variable, regression model from tabular data.  LightGBM architexture. Model-version blessing. Inference pipeline served over MLServer.
    """

    #--- flow parameters -------------------------------------------------------


    RETRAIN_PIPELINE_TYPE = "mf_lightgbm_regress_mlserver"
    # best way to share the config across subprocesses
    os.environ["retrain_pipeline_type"] = RETRAIN_PIPELINE_TYPE

    data_file_fullname = DagParam(
        description="Path to the input data file",
        default=os.path.realpath(
            os.path.join(os.path.dirname(__file__), "..", "data",
                         "synthetic_classif_tab_data_continuous.csv"))
    )
    buckets_param = DagParam(
        description="Bucketization to be applied " + \
             "on raw numerical feature(s). " + \
             "dict of optional pairs of  " + \
             "feature_name/buckets_count.",
        default={}
    )
    pipeline_hp_grid = DagParam(
        description="LightGBM model hyperparameters domain",
        default={
            "boosting_type": ["gbdt"],
            "num_leaves": [20, 30],
            "learning_rate": [0.01, 0.1],
            "n_estimators": [50, 100]
        }
    )
    cv_folds = DagParam(
        description="(int) how many Cross Validation folds shall be used.",
        default=3
    )
    dask_partitions = DagParam(
        description="(int) how many Dask partitions shall be used "+\
                    "per training_job.",
        default=6
    )
    wandb_run_mode = DagParam(
        description="WandB mode for the flow-run "+\
                    "indicating whether to sync it to the wandb server "+\
                    "can be either 'disabled', 'offline', or 'online'.",
        default="online"
    )

    default_preprocess_module_dir = \
        os.path.dirname(
            importlib.util.find_spec(
                f"retrain_pipelines.model."
                f"{RETRAIN_PIPELINE_TYPE}"
            ).origin)
    preprocess_artifacts_path = DagParam(
        description="MLserver artifacts location " + \
                    "(i.e. dir hosting your custom 'preprocessing.py'" + \
                    " file), if different from default",
        default=default_preprocess_module_dir
    )
    # TODO  -  convert from class method to TBD
    # @staticmethod
    # def _get_default_preprocess_artifacts_path() -> str:
        # return LightGbmHpCvWandbFlow.default_preprocess_module_dir
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
                    # LightGbmHpCvWandbFlow._get_default_preprocess_artifacts_path(),
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
    # def _get_default_pipeline_card_module_dir() -> str:
        # return LightGbmHpCvWandbFlow.default_pipeline_card_module_dir
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
                    # LightGbmHpCvWandbFlow._get_default_pipeline_card_module_dir(),
                    # "pipeline_card.py"
                # )
            # shutil.copy(filefullname, target_dir)
            # print(filefullname)
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
                    # LightGbmHpCvWandbFlow._get_default_pipeline_card_module_dir(),
                    # "template.html")
            # shutil.copy(filefullname, target_dir)
            # print(filefullname)

    del RETRAIN_PIPELINE_TYPE


    #---------------------------------------------------------------------------


    return start >> data >> split_data >> hyper_tuning >> \
           cross_validation >> training_job >> cross_validation_agg >> \
           best_hp >> train_model >> evaluate_model >> \
           model_version_blessing >> infra_validator >> \
           pipeline_card >> deploy >> load_test >> end

