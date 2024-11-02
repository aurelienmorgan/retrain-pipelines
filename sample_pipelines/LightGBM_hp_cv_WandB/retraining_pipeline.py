
import os
import sys

import numpy as np
import pandas as pd

import json
import time
import shutil
import joblib
import logging
import itertools
import importlib.util

from textwrap import dedent
from io import StringIO

import matplotlib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, \
                                    KFold
from sklearn.metrics import root_mean_squared_error, \
                            mean_absolute_error, r2_score
from sklearn.preprocessing import OrdinalEncoder

from metaflow import FlowSpec, step, Parameter, JSONType, \
    IncludeFile, current, metaflow_config as mf_config, \
    retry, Flow, Task, card
from metaflow.current import Current
from metaflow.cards import Image, Table, Markdown, Artifact

import wandb
from wandb.lightgbm import log_summary

import tempo
from tempo.serve.metadata import ModelDataArg, ModelDataArgs

import lightgbm as lgb

from retrain_pipelines import __version__
from retrain_pipelines.dataset import features_desc, \
        features_distri_plot
from retrain_pipelines.dataset.features_dependencies import \
        dataset_to_heatmap_fig
from retrain_pipelines.utils import grant_read_access, \
        tmp_os_environ


class LightGbmHpCvWandbFlow(FlowSpec):
    """
    Training pipeline
    """

    #--- flow parameters -------------------------------------------------------

    RETRAIN_PIPELINE_TYPE = "mf_lightgbm_regress_tempo"
    # best way to share the config across subprocesses
    os.environ["retrain_pipeline_type"] = RETRAIN_PIPELINE_TYPE

    data_file = IncludeFile(
        "data_file",
        is_text=True,
        help="Path to the input data file",
        default=os.path.realpath(
            os.path.join(os.path.dirname(__file__), "..", "data",
                         "synthetic_classif_tab_data_continuous.csv"))
    )

    # bucketization to be applied
    # on raw numerical feature(s)
    buckets_param = Parameter(
        "buckets_param",
        help="Bucketization to be applied "+\
             "on raw numerical feature(s). "+\
             "dict of optional pairs of  "+\
             "feature_name/buckets_count.",
        type=JSONType,
        default=dedent("""{}""")
    )

    # Tune hyperparameters of the model
    pipeline_hp_grid = Parameter(
        "pipeline_hp_grid",
        help="LightGBM model hyperparameters domain",
        type=JSONType,
        default=dedent("""{  
            "boosting_type": ["gbdt"],
            "num_leaves": [20, 30],
            "learning_rate": [0.01, 0.1],
            "n_estimators": [50, 100]
        }""")
    )

    # Cross-Validation folds
    cv_folds = Parameter(
        "cv_folds",
        help="(int) how many Cross Validation folds shall be used.",
        default=3
    )

    # Dask partitions per training_job
    dask_partitions = Parameter(
        "dask_partitions",
        help="(int) how many Dask partitions shall be used "+\
             "per training_job.",
        default=6
    )

    # WandB mode for the flow-run
    # indicating whether to sync it to the wandb server
    # can be either 'disabled', 'offline', or 'online'
    wandb_run_mode = Parameter(
        "wandb_run_mode",
        type=str,
        default="online",
        help="WandB mode for the flow-run "+\
             "indicating whether to sync it to the wandb server "+\
             "can be either 'disabled', 'offline', or 'online'."
    )

    # Tempo (MLserver SDK) artifacts location
    # (i.e. dir hosting preprocessing.py)
    default_preprocess_module_dir = \
        os.path.dirname(
            importlib.util.find_spec(
                f"retrain_pipelines.model."
                f"{RETRAIN_PIPELINE_TYPE}"
            ).origin)
    preprocess_artifacts_path = Parameter(
        "preprocess_artifacts_path",
        type=str,
        default=default_preprocess_module_dir,
        help="Tempo [MLserver SDK] artifacts location "+\
             "(i.e. dir hosting your custom 'preprocessing.py'"+\
             " file), if different from default"
    )
    @staticmethod
    def _get_default_preprocess_artifacts_path() -> str:
        return LightGbmHpCvWandbFlow.default_preprocess_module_dir
    @staticmethod
    def copy_default_preprocess_module(
        target_dir: str,
        exists_ok: bool = False
    ) -> None:
        os.makedirs(target_dir, exist_ok=True)
        if (
            not exists_ok and
            os.path.exists(os.path.join(target_dir, "preprocessing.py"))
        ):
            print("File already exists. Skipping copy.")
        else:
            filefullname = os.path.join(
                    LightGbmHpCvWandbFlow._get_default_preprocess_artifacts_path(),
                    "preprocessing.py"
                )
            shutil.copy(filefullname, target_dir)
            print(filefullname)

    # pipeline_card artifacts location
    # (i.e. dir hosting pipeline_card.py and/or template.html)
    default_pipeline_card_module_dir = \
        os.path.dirname(
            importlib.util.find_spec(
                f"retrain_pipelines.pipeline_card."+
                f"{RETRAIN_PIPELINE_TYPE}"
            ).origin)
    pipeline_card_artifacts_path = Parameter(
        "pipeline_card_artifacts_path",
        type=str,
        default=default_pipeline_card_module_dir,
        help="pipeline_card artifacts location "+\
             "(i.e. dir hosting your custom 'pipeline_card.py'"+\
             " and/or 'template.html' file)," +\
             " if different from default"
    )
    @staticmethod
    def _get_default_pipeline_card_module_dir() -> str:
        return LightGbmHpCvWandbFlow.default_pipeline_card_module_dir
    @staticmethod
    def copy_default_pipeline_card_module(
        target_dir: str,
        exists_ok: bool = False
    ) -> None:
        os.makedirs(target_dir, exist_ok=True)
        if (
            not exists_ok and
            os.path.exists(os.path.join(target_dir, "pipeline_card.py"))
        ):
            print("File already exists. Skipping copy.")
        else:
            filefullname = os.path.join(
                    LightGbmHpCvWandbFlow._get_default_pipeline_card_module_dir(),
                    "pipeline_card.py"
                )
            shutil.copy(filefullname, target_dir)
            print(filefullname)
    @staticmethod
    def copy_default_pipeline_card_html_template(
        target_dir: str,
        exists_ok: bool = False
    ) -> None:
        os.makedirs(target_dir, exist_ok=True)
        if (
            not exists_ok and
            os.path.exists(os.path.join(target_dir, "template.html"))
        ):
            print("File already exists. Skipping copy.")
        else:
            filefullname = os.path.join(
                    LightGbmHpCvWandbFlow._get_default_pipeline_card_module_dir(),
                    "template.html")
            shutil.copy(filefullname, target_dir)
            print(filefullname)

    del RETRAIN_PIPELINE_TYPE

    #---------------------------------------------------------------------------

    @step
    def start(self):
        print(f"{current.flow_name} - {current.run_id}")

        if "disabled" != self.wandb_run_mode:
            _ = wandb.login(
                host="https://api.wandb.ai"
            )

        # the WandB local folder with which the server is async.
        # 'local' == mf_config.DEFAULT_DATASTORE:
        wandb_run_dir = os.path.join(
            mf_config.DATASTORE_SYSROOT_LOCAL,
            mf_config.DATASTORE_LOCAL_DIR
        )
        if not os.path.isdir(wandb_run_dir):
            wandb_run_dir = os.getcwd() # the wandb default
        self.wandb_run_dir = \
            os.path.join(wandb_run_dir, current.flow_name, str(current.run_id))
        print(self.wandb_run_dir)

        if not os.path.exists(self.wandb_run_dir):
            os.makedirs(self.wandb_run_dir)

        self.data = pd.read_csv(StringIO(self.data_file))

        self.model_version_blessed = False
        self.current_blessed_run = None
        current.run.remove_tag("model_version_blessed")

        self.retrain_pipelines = f"retrain-pipelines {__version__}"
        self.retrain_pipeline_type = os.environ["retrain_pipeline_type"]

        self.serving_artifacts_local_folder = os.path.realpath(os.path.join(
                os.path.dirname(__file__),
                '..', '..', 'serving_artifacts',
                os.path.sep.join(current.run.path_components)
        ))

        if not os.path.exists(self.serving_artifacts_local_folder):
            os.makedirs(self.serving_artifacts_local_folder)

        self.next(self.eda)


    @step
    def eda(self):
        """
        exploratory data analysis.
        """

        ############################
        #    features and label    #
        #       basic counts       #
        ############################
        column_types = features_desc(self.data)
        self.features_desc = column_types
        ############################

        ############################
        #    features and label    #
        # respective distributions #
        ############################
        fig = features_distri_plot(self.data)
        self.data_distri_fig = fig
        ############################

        ############################
        #    features and label    #
        #       relationships      #
        ############################
        fig, _ = dataset_to_heatmap_fig(self.data)
        self.data_heatmap_fig = fig
        ############################

        self.next(self.preprocess_data)


    @step
    def preprocess_data(self):
        """
        feature engineering
        """
        # Load preprocessing module (can be user-tailored)
        from retrain_pipelines.utils import get_preprocess_data_fct
        preprocess_data_fct = \
            get_preprocess_data_fct(self.preprocess_artifacts_path)
        #from retrain_pipelines.model import preprocess_data_fct

        encoder = OrdinalEncoder()
        buckets = self.buckets_param.copy()

        self.X = self.data.iloc[:, :-1]
        self.y = self.data.iloc[:, -1]

        self.X_transformed = preprocess_data_fct(
            self.X, encoder, buckets,
            is_training=True, local_path=self.serving_artifacts_local_folder
        )
        self.encoder = encoder # <= to artifact store
        self.buckets = buckets # <= to artifact store

        self.next(self.split_data)


    @step
    def split_data(self):
        """
        hold-out dataset for eval
        on the final overall retrained model version.
        """

        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(
                self.X_transformed, self.y, test_size=0.2, random_state=42
            )
        print(f"train set : {len(self.X_train):,}, "+
              f"holdout set : {len(self.X_test):,}")

        self.next(self.hyper_tuning)


    @step
    def hyper_tuning(self):
        """
        Hyperparameters tuning
        """

        # Generate all combinations of parameters
        self.all_hp_params = [
            dict(zip(self.pipeline_hp_grid.keys(), v))
            for v in itertools.product(*self.pipeline_hp_grid.values())
        ]

        # For each combination of hyperparameters,
        # use cross validation to evaluate
        self.next(self.cross_validation, foreach='all_hp_params')


    @step
    def cross_validation(self):
        print(f"{current.task_id} : {self.input}")

        # recall that the respective proportion
        # of datapoints between train & valid sets
        # depends on n_splits
        kf = KFold(n_splits=self.cv_folds,
                   shuffle=True,
                   random_state=1121218)
        # list of one such pair per fold
        self.all_fold_splits = list(kf.split(X=self.X_train))

        self.next(self.training_job, foreach='all_fold_splits')


    @retry(times=2)
    @step
    def training_job(self):
        from retrain_pipelines.model import dask_regressor_fit

        def _parent_cv_task_info(
            training_job_task: LightGbmHpCvWandbFlow,
            current_: Current
        ) -> (int, dict) :
            """
            for a given training job, retrieve info pertaining
            to the parent "cross_validation" task :
                    - its task_id
                    - the set of hyperparameter values
                      it has been assigned.

            Params:
                - training_job_task (LightGbmHpCvWandbFlow)
                - current_ (Current)

            Results:
                - (int)
                - (dict)
            """

            current_task = Task(current_.pathspec)

            # retrieve list of ids for the "cross_validation" tasks
            # of the current flow run
            cross_validation_ids = [
                task.id for task in current_task.parent.parent[
                                        'cross_validation'].tasks()]
            cross_validation_ids.sort()

            # retrieve predecessors info (nested foreach)
            # for the current task
            foreach_level = 0
            cross_validation_foreach_frame = \
                training_job_task._foreach_stack[foreach_level]
            cross_validation_id = \
                cross_validation_ids[cross_validation_foreach_frame.index]
            print(f"cross_validation {cross_validation_id}")
            cross_validation_input = \
                training_job_task.foreach_stack()[foreach_level][-1]
            print(f"hyperparameter values : {cross_validation_input}")

            foreach_level = 1
            training_job_foreach_frame = \
                training_job_task._foreach_stack[foreach_level]
            training_job_index = training_job_foreach_frame.index
            print(f"CV fold #{training_job_index+1}/"+
                  f"{training_job_foreach_frame.num_splits}")

            return cross_validation_id, cross_validation_input

        self.cross_validation_id, self.fold_hp_params = \
            _parent_cv_task_info(self, current)

        ###################
        # actual training #
        ###################

        if "disabled" != self.wandb_run_mode:
            parallel_run = wandb.init(
                project=current.flow_name,
                group=str(current.run_id),
                id=str(current.task_id),
                name=str(current.task_id),
                mode=self.wandb_run_mode,
                #entity='organization', # default being the entity named
                                        # same as your 'login'
                notes="first attempt",
                tags=["baseline", "dev"],
                job_type=str(self.cross_validation_id),
                # sync_tensorboard=True,
                dir=self.wandb_run_dir, # custom log directory
                                        # (internals ; local, for online synch)
                config= {
                    **self.fold_hp_params,
                    **dict(
                        mf_id=current.run_id,
                        mf='_'.join(["cv_task", self.cross_validation_id]),
                        mf_task='_'.join([current.step_name,
                                          self.cross_validation_id,
                                          current.task_id])
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


        (train_idx, val_idx) = self.input
        print(f"train set : {len(train_idx)}, valid set : {len(val_idx)}")

        X_train, y_train = self.X_train.iloc[train_idx], \
                           self.y_train.iloc[train_idx]
        X_val, y_val = self.X_train.iloc[val_idx], \
                       self.y_train.iloc[val_idx]
        fold_lgb_reg, fold_history, fold_workers_history = \
            dask_regressor_fit(
                X_train, y_train, X_val, y_val,
                npartitions = self.dask_partitions,
                hp_dict=self.fold_hp_params
            )
        self.fold_workers_history = fold_workers_history # <= to artifact store
        self.fold_lgb_reg = fold_lgb_reg                 # <= to artifact store

        # rmse
        preds = fold_lgb_reg.predict(X_val)
        fold_rmse = root_mean_squared_error(y_val, preds)
        self.fold_rmse = fold_rmse
        print(f"Fold finished with rmse: {self.fold_rmse:.5f}.\n"+
              f"(on last distributed epoch :"+
              f"{fold_history['Validation']['rmse'][-1]})")

        #################################################################

        if "disabled" != self.wandb_run_mode:
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

            parallel_run.log(dict(hp_cv_rmse=self.fold_rmse))
            hp_table = wandb.Table(
                data=[list(self.fold_hp_params.values())+
                      [self.fold_rmse]],
                columns=list(self.fold_hp_params.keys())+\
                        ['rmse']
            )
            wandb.log({'Sweep_table': hp_table})
            parallel_run.finish()

        self.next(self.cross_validation_agg)


    @step
    def cross_validation_agg(self, inputs):
        """
        Aggregates the performance metrics values
        obtained during cross-validation trainings
        for the concerned set of hyperparameter values.
        """

        self.merge_artifacts(inputs, exclude=
            ['fold_hp_params', 'fold_rmse', 'fold_lgb_reg',
             'fold_history', 'fold_workers_history'])

        hp_cv_rmses = []

        for training_job_input in inputs:
            # Get per metrics from previous steps
            hp_cv_rmses.append(training_job_input.fold_rmse)

        self.hp_rmse = float(np.mean(hp_cv_rmses))
        print(f"hp_cv_rmses : {hp_cv_rmses},\nmean : {self.hp_rmse}, "+
              f"std : {float(np.std(hp_cv_rmses))}, "+
              f"median : {float(np.median(hp_cv_rmses))}")

        self.hp_dict = self.foreach_stack()[0][-1]

        print(f"hp values {self.hp_dict} lead to an rmse of {self.hp_rmse}")

        self.next(self.best_hp)


    @step
    def best_hp(self, inputs):
        """
        Checks performance metric values
        from the different sets of hyperparameter values
        (perf evaluated during hp-tuning)
        and selects the best such set
        for training on the whole training-set next.
        """

        self.merge_artifacts(inputs, exclude=['cross_validation_id',
                                              'hp_rmse', 'hp_dict'])

        # organize respective hp results
        self.hp_perfs_list = [
            {'mf:cv_task': input.cross_validation_id,
             **{**input.hp_dict,
                "rmse": input.hp_rmse}
            } for input in inputs
        ]
        print(pd.DataFrame(self.hp_perfs_list
                          ).sort_values(by='rmse',ascending=False
                          ).to_string(index=False))

        # get perf metrics from previous steps
        hp_rmses = [input.hp_rmse for input in inputs]
        print(f"hp_accuracies : {hp_rmses}")

        best_cv_agg_task_idx = np.argmin(hp_rmses)
        print(f"best_cv_agg_task_idx : {best_cv_agg_task_idx}")
        print(f"best_cv_agg_task_idx : "+
              f"{inputs[best_cv_agg_task_idx].cross_validation_id}")

        # find best hyperparams
        self.hyperparameters = inputs[best_cv_agg_task_idx].hp_dict
        print(f"hyperparameters : {self.hyperparameters}")

        self.next(self.train_model)


    @step
    def train_model(self):
        """
        A new LightGBM model is fitted.
        """

        from retrain_pipelines.model import dask_regressor_fit

        if "disabled" != self.wandb_run_mode:
            wandb.join()
            training_run = wandb.init(
                project=current.flow_name,
                group=str(current.run_id),
                id=str(current.task_id),
                name=str(current.task_id),
                mode=self.wandb_run_mode,
                #entity='organization', # default being the entity
                                        # named same as your 'login'
                notes="first attempt",
                tags=["baseline", "dev"],
                job_type=str(current.task_id),
                dir=self.wandb_run_dir, # custom log directory
                                        # (internals ; local, for online synch)
                config= {
                    **self.hyperparameters,
                    **dict(
                        mf_id=current.run_id,
                        mf='_'.join([current.step_name, current.task_id]),
                        mf_task='_'.join([current.step_name, current.task_id])
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
            self.wandb_project_ui_url = \
                "{wandb_ui_url}{wandb_org}/{wandb_proj}/workspace" \
                    .format(
                        wandb_ui_url=wandb.Api().client.app_url,
                        wandb_org=wandb.Api().viewer._attrs['entity'],
                        wandb_proj=training_run.project_name()
                    )
            print(self.wandb_project_ui_url)
            # retrieve run_id for WandB workspace filtering
            # (needed for cases where Metaflow flow run is
            #  resumed from a posterior step)
            self.wandb_filter_run_id = current.run_id

        model, history, workers_history = \
            dask_regressor_fit(
                self.X_train, self.y_train,
                self.X_test, self.y_test,
                npartitions = self.dask_partitions,
                hp_dict=self.hyperparameters
            )
        self.workers_history = workers_history # <= to artifact store
        self.history = history                 # <= to artifact store

        if "disabled" != self.wandb_run_mode:
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

        self.model = model
        self.predictions = model.predict(self.X_test)

        # Generate training plot
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
        self.training_plt_fig = fig
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
        self.features_plt_fig = fig
        plt.close()

        self.next(self.evaluate_model)


    @step
    def evaluate_model(self):
        #############################################
        # overall (over whole test set) performance #
        #############################################
        rmse = root_mean_squared_error(self.y_test, self.predictions)
        mae  = mean_absolute_error(self.y_test, self.predictions)
        r2 = r2_score(self.y_test, self.predictions)

        self.metrics = {
            "rmse": rmse,
            "mae ": mae,
            "r2": r2
        }
        #############################################

        if "disabled" != self.wandb_run_mode:
            wandb_flow_run = wandb.init(
                project=current.flow_name,
                group=str(current.run_id),
                id=str(current.run_id),
                name=str(current.run_id),
                mode=self.wandb_run_mode,
                #entity='organization', # default being
                                        # the entity named same as your 'login'
                notes="first attempt",
                tags=["baseline", "dev"],
                job_type=str(current.run_id),
                dir=self.wandb_run_dir, # custom log directory
                                        # (internals ; local, for online synch)
                config= {
                    **self.hyperparameters,
                    **dict(
                        mf_id=current.run_id,
                        mf='_'.join([current.step_name, current.task_id]),
                        mf_task='_'.join([current.step_name, current.task_id])
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

            wandb_flow_run.log(dict(metrics=self.metrics,
                                    # and, for charts
                                    hp_cv_rmse=self.metrics['rmse']
            ))

            wandb_flow_run.finish()
            wandb.join()

        #############################################
        #             sliced performance            #
        #############################################
        if "feature_names_in_" in self.encoder.__dict__:
            categ_features_arr = \
                self.encoder.__dict__['feature_names_in_']
            first_categorical_feature = \
                            categ_features_arr[0]
            first_categ_feature__labels = \
                self.encoder.__dict__['categories_'][0]

            first_categ_feature_sliced_metrics = {}
            # actual sliced perf computation
            for feature_categ in self.X_test[first_categorical_feature].unique():
                # Get test-records indices for this category
                slice_filter = \
                    self.X_test[first_categorical_feature].values == feature_categ

                # Calculate metrics for this category
                sliced_y_test = self.y_test[slice_filter]
                sliced_predictions = self.predictions[slice_filter]

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
            self.first_categorical_feature = \
                first_categorical_feature
            self.first_categ_feature_sliced_metrics = \
                first_categ_feature_sliced_metrics
            print(pd.DataFrame.from_dict(
                        first_categ_feature_sliced_metrics, orient='index'
                    ).sort_index(ascending=True))
        else:
            self.first_categorical_feature = None
            self.first_categ_feature_sliced_metrics = None
        #############################################

        self.next(self.model_version_blessing)


    @step
    def model_version_blessing(self):
        """
        Compare new model version against best predecessor.
        """

        current_blessed_rmse = sys.maxsize
        # find latest blessed model version (from a previous flow-run)
        for run in Flow(self.__class__.__name__):
            if (
                run.successful and
                'model_version_blessed' in run.data and
                run.data.model_version_blessed
            ):
                self.current_blessed_run = run
                current_blessed_rmse = \
                    self.current_blessed_run.data.metrics['rmse']
                self.model_version_blessed = \
                    self.metrics['rmse'] <= current_blessed_rmse
                print("new : " + str(self.metrics['rmse']) +
                      " - previous best : " + str(current_blessed_rmse) +
                      " - model_version_blessing : " +
                          str(self.model_version_blessed))
                break
        # case 'no prior blessed run'
        if current_blessed_rmse == sys.maxsize:
            print("case 'no prior blessed model version found => blessing.'")
            self.model_version_blessed = True

        # self.model_version_blessed = True ### DEBUG - DELETE ###
        if self.model_version_blessed:
            current.run.add_tags(['model_version_blessed'])

        self.next(self.infra_validator)


    @step
    def infra_validator(self):
        """
        If the trained model version is blessed, validate serving.
        """
        from retrain_pipelines.model import get_tempo_artifact, \
                tempo_wait_ready, tempo_predict

        # tracking below using a 3-states var
        # -1 for not applicable, and
        # 0/1 bool for failure/success otherwise
        self.local_serve_is_ready = -1

        if self.model_version_blessed:
            from tempo.serve.loader import save
            from retrain_pipelines.utils import system_has_conda, \
                    is_conda_env, venv_as_conda

            # serialize model version
            model_file = os.path.join(self.serving_artifacts_local_folder,
                                      'model.joblib')
            joblib.dump(self.model, model_file)

            if system_has_conda():
                # get name of virtual environment
                conda_need_delete = False
                if is_conda_env():
                    conda_env = os.path.basename(sys.prefix)
                else:
                    conda_env = "_"+os.path.basename(sys.prefix)
                    # create the (temp) conda env
                    # for Seldon/MLServer to use
                    venv_as_conda(sys.prefix, conda_env)
                    conda_need_delete = True

                os.environ["no_proxy"] = "localhost,127.0.0.1,0.0.0.0"

                # inference signature
                # must be declared explicitely for "from BYTES" conversion
                # of string attributes of payload on http inference request
                tempo_input_args = []
                for column in self.X.columns:
                    if np.issubdtype(self.X[column].dtype, np.number):
                        ty = np.ndarray
                    else:
                        ty = str
                    tempo_input_args.append(ModelDataArg(ty=ty, name=column))
                tempo_inputs = ModelDataArgs(args=tempo_input_args)
                tempo_outputs = ModelDataArgs(args=[
                    ModelDataArg(ty=float, name="predicted_value")
                ])

                # wrap the model version for Seldon
                tempo_model = get_tempo_artifact(
                    conda_env=conda_env,
                    local_folder=self.serving_artifacts_local_folder,
                    description="Blessed model version",
                    inputs=tempo_inputs,
                    outputs=tempo_outputs,
                    verbosity=logging.DEBUG
                )

                ############################################
                #       pack the python environment        #
                ############################################
                print("!!!   Packing env can take a while, hang in there.   !!!")
                save(tempo_model)  ## environment.tar.gz
                grant_read_access(os.path.join(
                    self.serving_artifacts_local_folder, 'environment.tar.gz'))
                # print(os.system("ls -lh " +
                #     os.path.join(serving_artifacts_local_folder, 'environment.tar.gz')))
                if conda_need_delete:
                    shutil.rmtree(
                        os.path.join(
                            os.environ['CONDA_PREFIX'], 'envs', conda_env
                        ), ignore_errors=True)
                ############################################

                with tmp_os_environ({'REQUESTS_CA_BUNDLE': ''}):
                    ############################################
                    # temporarily annihilate env var           #
                    #           REQUESTS_CA_BUNDLE             #
                    # due to interferences in their respective #
                    # interactions with python lib `requests`  #
                    # between `wandb` & `seldon tempo`         #
                    ############################################


                    ############################################
                    #  actually deploy the prediction service  #
                    ############################################
                    start_time = time.time()
                    from tempo import deploy_local
                    remote_model = deploy_local(tempo_model)
                    self.local_serve_is_ready = \
                        int(tempo_wait_ready(remote_model, timeout=10*60))
                    elapsed_time = time.time() - start_time
                    print(f"deploy_local -   Elapsed time: {elapsed_time:.2f} seconds")
                    ############################################

                    if self.local_serve_is_ready == 1:
                        from tempo.protocols.v2 import V2Protocol

                        inference_request = V2Protocol().to_protocol_request(
                            categ_feature0='value1',
                            num_feature1=np.array([0.1], dtype=np.float32),
                            num_feature2=np.array([0.1], dtype=np.float32),
                            num_feature3=np.array([3.1], dtype=np.float32),
                            num_feature4=np.array([0.2], dtype=np.float32)
                        )
                        print(f"inference_request : {inference_request}")
                        try:
                            prediction = tempo_predict(remote_model, inference_request)
                            print(json.dumps(prediction, indent=4))
                        except Exception as ex:
                            import traceback
                            print('inference_request : ', end='', file=sys.stderr)
                            print(inference_request, file=sys.stderr)
                            print(ex, file=sys.stderr)
                            traceback.print_tb(ex.__traceback__, file=sys.stderr)
                            self.local_serve_is_ready = 0

                        finally:
                            remote_model.undeploy()
            else:
                # No conda
                self.local_serve_is_ready = 0
                print(
                    "Error: 'conda' not installed or not found in PATH. "+
                    "Seldon/MLServer however requires it. " +
                    "You may consider 'miniforge'. ",
                    file=sys.stderr, flush=True)

        self.next(self.pipeline_card)


    @card(id='default')
    @card(type='html', id='custom')
    @step
    def pipeline_card(self):
        import datetime

        ###########################
        # user-provided artifacts #
        ###########################
        # note that user can provide either
        # 'pipeline_card.py' or 'template.html'
        # or both when specifying custom
        # 'pipeline_card_artifacts_path'
        if "template.html" in os.listdir(self.pipeline_card_artifacts_path):
            template_dir = self.pipeline_card_artifacts_path
        else:
            template_dir = os.path.dirname(
                importlib.util.find_spec(
                    f"retrain_pipelines.pipeline_card."+
                    f"{os.getenv('retrain_pipeline_type')}"
                ).origin)
        ###########################
        if "pipeline_card.py" in os.listdir(self.pipeline_card_artifacts_path):
            from retrain_pipelines.utils import get_get_html
            get_html = \
                get_get_html(self.pipeline_card_artifacts_path)
        else:
            from retrain_pipelines.pipeline_card import get_html
        from retrain_pipelines.pipeline_card.helpers import mf_dag_svg
        ###########################

        ########################
        ##   "default" card   ##
        ########################
        self.metadata = {
            "name": "LightGBM Model",
            "version": "1.0",
            "retrain_pipelines": f"retrain-pipelines {__version__}",
            "retrain_pipeline_type": os.environ["retrain_pipeline_type"],
            "description": "A LightGBM model retrained using Dask",
            "authors": [current.username],
            "tags": ["regression", "lightgbm"],
            "license": "MIT License",
            "model_details": {
                "library": "LightGBM",
                "version": lgb.__version__
            },
            "metrics": self.metrics,
            "data_sources": [
                {"name": "Training Data", "path": self.data_file}
            ],
            "preprocessing": [
                {
                    "name": "Normalization",
                    "description": "Standard scaling of features"
                }
            ],
            "references": [
                {
                    "title": "LightGBM Documentation",
                    "link": "https://lightgbm.readthedocs.io/"
                }
            ]
        }

        training_image = Image.from_matplotlib(self.training_plt_fig)
        feature_imp_image = Image.from_matplotlib(self.features_plt_fig)

        current.card['default'].append(Markdown(
            "model_version_blessed : **%s**" % str(self.model_version_blessed)))
        current.card['default'].append(Artifact(
            {"model_version_blessed": self.model_version_blessed}))

        current.card['default'].append(training_image)
        current.card['default'].append(feature_imp_image)

        rows = []
        rows.append([float(value) for value in self.metrics.values()])
        current.card['default'].append(Table(rows,
                                             headers=list(self.metrics.keys())))
        ########################

        ########################
        ## html "custom" card ##
        ########################
        dt = datetime.datetime.now(tz=datetime.timezone.utc)
        formatted_dt = dt.strftime("%A %b %d %Y %I:%M:%S %p %Z")
        task_obj_python_cmd = f"metaflow.Task(\"{current.pathspec}\", " + \
                                              f"attempt={str(current.retry_count)})"
        params={
            'template_dir': template_dir,
            'title': f"{current.flow_name}",
            "subtitle": f"(flow run # {len(list(current.run.parent.runs()))}," + \
                        f" run_id: {str(current.run.id)}  -  {formatted_dt})",
            'model_version_blessed': self.model_version_blessed,
            'current_blessed_run': self.current_blessed_run,
            'local_serve_is_ready': self.local_serve_is_ready,
            'records_count': self.data.shape[0],
            'features_desc': self.features_desc,
            'data_distri_fig': self.data_distri_fig,
            'data_heatmap_fig': self.data_heatmap_fig,
            'training_plt_fig': self.training_plt_fig,
            'features_plt_fig': self.features_plt_fig,
            'perf_metrics_dict': self.metrics,
            'slice_feature_name': self.first_categorical_feature,
            'sliced_perf_metrics_dict': \
                self.first_categ_feature_sliced_metrics,
            'buckets_dict': self.buckets_param,
            'hyperparameters_dict': self.hyperparameters,
            'wandb_project_ui_url': (
                self.wandb_project_ui_url
                if 'disabled' != self.wandb_run_mode else None),
            'wandb_filter_run_id': (
                self.wandb_filter_run_id
                if 'disabled' != self.wandb_run_mode else None),
            'wandb_need_sync_dir': (
                "" if "offline" != self.wandb_run_mode
                else self.wandb_run_dir +
                     os.path.sep.join(['', 'wandb', 'offline-*'])
            ),
            'pipeline_hp_grid_dict': self.pipeline_hp_grid,
            'cv_folds': self.cv_folds,
            'hp_perfs_list': self.hp_perfs_list,
            'task_obj_python_cmd': task_obj_python_cmd,
            'dag_svg': mf_dag_svg(self)
        }
        self.html = get_html(params)
        ########################
        current
        ########################

        self.next(self.deploy)


    @step
    def deploy(self):
        """
        placeholder for the serving SDK deploy call
        (on the target production platform).
        Include any artifact you want, consider including the model card !
        """

        if self.model_version_blessed and (self.local_serve_is_ready == 1):
            pass # your code here

        self.next(self.load_test)


    @step
    def load_test(self):
        """
        placeholder
        """

        if self.model_version_blessed and (self.local_serve_is_ready == 1):
            pass # your code here

        self.next(self.end)


    @step
    def end(self):
        # self.wandb_flow_run.finish()
        # wandb.join()
        pass

        if "offline" == self.wandb_run_mode:
            print(
                "Note that the herein flow run was not live-synced to WandB.\n"+
                "Execute the following WandB CLI command locally to sync it\n"+
                "so log-data shows there for this flow run :\n"+
                "wandb sync {wandb_need_sync_dir}".format(
                    wandb_need_sync_dir=self.wandb_run_dir +
                        os.path.sep.join(['', 'wandb', 'offline-*'])
                )
            )


if __name__ == "__main__":
    LightGbmHpCvWandbFlow()
