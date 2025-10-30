
import os
import sys

import json
import time
import shutil
import logging
import traceback
import subprocess
import importlib.util
from textwrap import dedent
from io import StringIO

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, \
                                    train_test_split
from sklearn.metrics import accuracy_score, precision_score, \
                            recall_score, f1_score, \
                            confusion_matrix
from sklearn.preprocessing import StandardScaler, \
                                  OneHotEncoder

from metaflow import FlowSpec, step, Parameter, JSONType, \
    IncludeFile, current, metaflow_config as mf_config, \
    resources, Flow, Task, card
from metaflow.current import Current
from metaflow.cards import Image, Table, Markdown, Artifact

import wandb

from pytorch_tabnet.tab_model import TabNetClassifier
from torch.nn import CrossEntropyLoss

from retrain_pipelines import __version__
from retrain_pipelines.dataset import features_desc, \
                                      features_distri_plot
from retrain_pipelines.dataset.features_dependencies import \
        dataset_to_heatmap_fig
from retrain_pipelines.utils import flatten_dict, \
        dict_dict_list_get_all_combinations, \
        create_requirements


logging.getLogger().setLevel(logging.INFO)


class TabNetHpCvWandbFlow(FlowSpec):
    """
    Training pipeline
    """

    #--- flow parameters -------------------------------------------------------

    RETRAIN_PIPELINE_TYPE = "mf_tabnet_classif_torchserve"
    # best way to share the config across subprocesses
    os.environ["retrain_pipeline_type"] = RETRAIN_PIPELINE_TYPE

    data_file = IncludeFile(
        "data_file",
        is_text=True,
        help="Path to the input data file",
        default=os.path.realpath(
            os.path.join(os.path.dirname(__file__), "..", "data",
                         "synthetic_classif_tab_data_4classes.csv"))
    )

    buckets_param = Parameter(
        "buckets_param",
        help="Bucketization to be applied "+\
             "on raw numerical feature(s). "+\
             "dict of optional pairs of  "+\
             "feature_name/buckets_count.",
        type=JSONType,
        default=dedent("""{}""")
    )

    pipeline_hp_grid = Parameter(
        "pipeline_hp_grid",
        help="TabNet model hyperparameters domain " + \
             "(for both the model and the trainer)",
        type=JSONType,
        default=dedent("""{
            "trainer": {
                "max_epochs": [300],
                "patience": [30],
                "batch_size": [1024],
                "virtual_batch_size": [256]
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
        }""").replace("'", '"').strip('"')
    )

    cv_folds = Parameter(
        "cv_folds",
        help="(int) how many Cross Validation folds shall be used.",
        default=3
    )

    wandb_run_mode = Parameter(
        "wandb_run_mode",
        type=str,
        default="online",
        help="WandB mode for the flow-run "+\
             "indicating whether to sync it to the wandb server "+\
             "can be either 'disabled', 'offline', or 'online'."
    )

    # TorchServe artifacts location
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
        return TabNetHpCvWandbFlow.default_preprocess_module_dir
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
                    TabNetHpCvWandbFlow._get_default_preprocess_artifacts_path(),
                    "preprocessing.py"
                )
            shutil.copy(filefullname, target_dir)
            print(filefullname)

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
        return TabNetHpCvWandbFlow.default_pipeline_card_module_dir
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
                    TabNetHpCvWandbFlow._get_default_pipeline_card_module_dir(),
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
                    TabNetHpCvWandbFlow._get_default_pipeline_card_module_dir(),
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
            os.path.join(
                wandb_run_dir, current.flow_name, str(current.run_id))
        print(self.wandb_run_dir)

        if not os.path.exists(self.wandb_run_dir):
            os.makedirs(self.wandb_run_dir)

        self.data = pd.read_csv(StringIO(self.data_file))

        self.model_version_blessed = False
        self.current_blessed_run = None
        current.run.remove_tag("model_version_blessed")

        self.retrain_pipelines = f"retrain-pipelines {__version__}"
        self.retrain_pipeline_type = os.environ["retrain_pipeline_type"]

        self.serving_artifacts_local_folder = \
            os.path.realpath(os.path.join(
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

        scaler = StandardScaler()
        encoder = OneHotEncoder(
            sparse_output=False,
            drop=None # drop='first' for linear models, @see the doc
        )
        buckets = self.buckets_param.copy()

        self.X = self.data.iloc[:, :-1]
        self.y = self.data.iloc[:, -1]

        grouped_features = []
        self.X_transformed = preprocess_data_fct(
            self.X, scaler, encoder, buckets, grouped_features,
            is_training=True, local_path=self.serving_artifacts_local_folder
        )
        self.scaler = scaler                     # <= to artifact store
        self.encoder = encoder                   # <= to artifact store
        self.buckets = buckets                   # <= to artifact store
        self.grouped_features = grouped_features # <= to artifact store
                                                 # (and coming model init)

        self.next(self.split_data)


    @step
    def split_data(self):
        """
        hold-out dataset for eval
        on the final overall retrained model version.
        """

        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(
                self.X_transformed, self.y,
                test_size=0.2, random_state=42
            )

        self.next(self.hyper_tuning)


    @step
    def hyper_tuning(self):
        """
        Hyperparameters tuning
        """

        # Generate all combinations of parameters
        self.all_hp_params = \
            dict_dict_list_get_all_combinations(
                self.pipeline_hp_grid)

        # For each combination of hyperparameters,
        # use cross validation to evaluate
        self.next(self.cross_validation, foreach='all_hp_params')


    @step
    def cross_validation(self):
        print(str(current.task_id) + ': ' + str(self.input))

        strat_kf = StratifiedKFold(n_splits=self.cv_folds,
                                   shuffle=True,
                                   random_state=1121218)
        self.all_fold_splits = \
            list(strat_kf.split(self.X_train, self.y_train))

        self.next(self.training_job, foreach='all_fold_splits')


    @resources(cpu=3) # restrict or parallel models trainings
                      # fight unefficiently for cores,
                      # canibalizing one-another
                      # (total bloodbath..)
    @step
    def training_job(self):
        from retrain_pipelines.model import PrintLR, WandbCallback

        def _parent_cv_task_info(
            training_job_task: TabNetHpCvWandbFlow,
            current_: Current
        ) -> (int, dict) :
            """
            for a given training job, retrieve info pertaining
            to the parent "cross_validation" task :
                    - its task_id
                    - the set of hyperparameter values
                      it has been assigned.

            Params:
                - training_job_task (TabNetHpCvWandbFlow)
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
                #entity='organization', # default being the entity
                                        # named same as your 'login'
                notes="first attempt",
                tags=["baseline", "dev"],
                job_type=str(self.cross_validation_id),
                # sync_tensorboard=True,
                dir=self.wandb_run_dir, # custom log directory
                                        # (internals ; local,
                                        #  for online synch)
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

        (train_idx, test_idx) = self.input

        X_train, X_val = (self.X_train.iloc[train_idx],
                          self.X_train.iloc[test_idx])
        y_train, y_val = (self.y_train.iloc[train_idx],
                          self.y_train.iloc[test_idx])

        # Add eval_set and eval_names to track evaluation results
        eval_set = [(self.X_train.values, self.y_train),
                    (self.X_test.values, self.y_test)]
        eval_names = ['Training', 'Validation']

        tabnet_clf = TabNetClassifier(
            **self.fold_hp_params.get('model', {}),
            grouped_features=self.grouped_features
        )
        tabnet_clf.fit(
            X_train=X_train.values,
            y_train=y_train,
            loss_fn=CrossEntropyLoss(),
            eval_set=eval_set,
            eval_name=eval_names,
            eval_metric=['accuracy'],
            **self.fold_hp_params.get('trainer', {}),
            augmentations=None,
            drop_last=False,
            callbacks=(
                [PrintLR()] +
                [WandbCallback()] if 'disabled' != self.wandb_run_mode
                else []
            )
        )

        pred_labels = tabnet_clf.predict(X_val.values)
        self.accuracy = accuracy_score(y_val, pred_labels)

        print(f"Fold finished with accuracy: {self.accuracy:.5f}.")

        #################################################################

        if "disabled" != self.wandb_run_mode:
            parallel_run.log(dict(hp_cv_acc=self.accuracy))
            # recall that self.fold_hp_params is a dict of dicts
            cv_input_flattened_dict = \
                flatten_dict(self.fold_hp_params, callable_to_name=True)
            hp_table = wandb.Table(
                data=[list(cv_input_flattened_dict.values())+[self.accuracy]],
                columns=list(cv_input_flattened_dict.keys())+['accuracy']
            )
            wandb.log({'Sweep_table': hp_table})
            parallel_run.log(dict(val_accuracy=self.accuracy))
            parallel_run.finish()

        self.next(self.cross_validation_agg)


    @step
    def cross_validation_agg(self, inputs):
        """
        Aggregates the performance metrics values
        obtained during cross-validation trainings
        for the concerned set of hyperparameter values.
        """

        self.merge_artifacts(inputs,
                             exclude=['fold_hp_params', 'accuracy'])

        hp_cv_accuracies = []

        for training_job_input in inputs:
            hp_cv_accuracies.append(training_job_input.accuracy)

        self.hp_accuracy = float(np.mean(hp_cv_accuracies))

        self.hp_dict_dict = self.foreach_stack()[0][-1]

        print(f"hp values {self.hp_dict_dict} lead "+
              f"to an accuracy of {self.hp_accuracy}")

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

        self.merge_artifacts(
            inputs,
            exclude=['cross_validation_id',
                     'hp_accuracy', 'hp_dict_dict'])

        # organize respective hp results
        self.hp_perfs_list = [
            {'mf:cv_task': input.cross_validation_id,
             **{**flatten_dict(input.hp_dict_dict,
                               callable_to_name=True),
                'accuracy': input.hp_accuracy}
            } for input in inputs
        ]
        print(pd.DataFrame(self.hp_perfs_list
                          ).sort_values(
                                by='accuracy', ascending=False))

        # get perf metrics from previous steps
        hp_accuracies = [input.hp_accuracy for input in inputs]
        print(f"hp_accuracies : {hp_accuracies}")

        best_cv_agg_task_idx = np.argmax(hp_accuracies)
        print(f"best_cv_agg_task_idx : {best_cv_agg_task_idx}")
        print(f"best_cv_agg_task_idx : "+
              f"{inputs[best_cv_agg_task_idx].cross_validation_id}")

        # find best hyperparams
        self.hyperparameters = inputs[best_cv_agg_task_idx].hp_dict_dict
        print(f"hyperparameters : {self.hyperparameters}")

        self.next(self.train_model)


    @step
    def train_model(self):
        """
        A new TabNet model is fitted.
        """
        from retrain_pipelines.model import PrintLR, WandbCallback
        from retrain_pipelines.pipeline_card import plot_masks_to_dict

        model = TabNetClassifier(
            **self.hyperparameters.get('model', {}),
            grouped_features=self.grouped_features
        )

        # Add eval_set and eval_names to track evaluation results
        eval_set = [(self.X_train.values, self.y_train),
                    (self.X_test.values, self.y_test)]
        eval_names = ['Training', 'Validation']

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
            # print("WandB UI url : " + wandb.Api().client.app_url)
            # print("WandB entity : " + wandb.Api().viewer._attrs['entity'])
            # print("WandB username : " + wandb.Api().viewer._attrs['username'])
            # print("WandB project : " + training_run.project_name())
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

        model.fit(
            X_train=self.X_train.values,
            y_train=self.y_train,
            loss_fn=CrossEntropyLoss(),
            eval_set=eval_set, eval_name=eval_names,
            eval_metric=['accuracy'],
            **self.hyperparameters.get('trainer', {}),
            augmentations=None,
            drop_last=False,
            callbacks=(
                [PrintLR()] +
                [WandbCallback()] if 'disabled' != self.wandb_run_mode
                else []
            )
        )

        if "disabled" != self.wandb_run_mode:
            training_run.finish()

        self.model = model
        self.predictions = model.predict(self.X_test.values)

        # training plot
        epochs = range(1, len(model.history['Training_accuracy']) + 1)
        train_accuracy = model.history['Training_accuracy']
        valid_accuracy = model.history['Validation_accuracy']
        learning_rates = model.history['lr']

        # training history
        fig, ax1 = plt.subplots()
        # Plotting train_accuracy and valid_accuracy on the primary y-axis
        ax1.plot(epochs, train_accuracy, 'b-', label='Training Accuracy',
                 color='#4B0082', zorder=2)
        ax1.plot(epochs, valid_accuracy, 'g-', label='Validation Accuracy',
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
        self.training_plt_fig = fig
        plt.close()

        # model attention layers (as set by 'n_steps') masks
        target_class_figs = plot_masks_to_dict(
            model,
            X_transformed=self.X_test,
            grouped_features=self.grouped_features,
            raw_feature_names=self.X.columns.tolist(),
            y=self.y_test
        )
        self.target_class_figs = target_class_figs

        self.next(self.evaluate_model)


    @step
    def evaluate_model(self):
        #############################################
        # overall (over whole test set) performance #
        #############################################
        accuracy = accuracy_score(self.y_test, self.predictions)
        precision = precision_score(self.y_test, self.predictions,
                                    average='weighted')
        recall = recall_score(self.y_test, self.predictions,
                              average='weighted')
        f1 = f1_score(self.y_test, self.predictions,
                      average='weighted')

        self.classes_weighted_metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

        self.conf_matrix = confusion_matrix(self.y_test, self.predictions)
        #############################################

        if "disabled" != self.wandb_run_mode:
            wandb_flow_run = wandb.init(
                project=current.flow_name,
                group=str(current.run_id),
                id=str(current.run_id),
                name=str(current.run_id),
                mode=self.wandb_run_mode,
                #entity='organization', # default being the entity
                                        # named same as your WandB 'login'
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

            wandb_flow_run.log(dict(metrics=self.classes_weighted_metrics,
                                    # for charts
                                    hp_cv_acc= \
                                        self.classes_weighted_metrics['accuracy']
            ))

            wandb_flow_run.finish()
            wandb.join()

        #############################################
        #             sliced performance            #
        #############################################
        # as categ_features at this stage are One-Hot encoded,
        # we shall retrieve the column names (from the encoder)
        encoder_file = os.path.join(self.serving_artifacts_local_folder,
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
                    self.X_test[feature_column_name].values == 1

                # Calculate metrics for this category
                sliced_y_test = self.y_test[slice_filter]
                sliced_predictions = self.predictions[slice_filter]

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

        current_blessed_accuracy = 0
        # find latest blessed model version (from a previous flow-run)
        for run in Flow(self.__class__.__name__):
            if (
                run.successful and
                'model_version_blessed' in run.data and
                run.data.model_version_blessed
            ):
                self.current_blessed_run = run
                current_blessed_accuracy = \
                    self.current_blessed_run.data.classes_weighted_metrics['accuracy']
                print("new : " + str(self.classes_weighted_metrics['accuracy']) +
                      " - previous best : " + str(current_blessed_accuracy) +
                      " - model_version_blessing : " +
                          str(self.classes_weighted_metrics['accuracy'] >=
                                self.current_blessed_run.data.classes_weighted_metrics[
                                    'accuracy']))
                self.model_version_blessed = (
                    self.classes_weighted_metrics['accuracy'] >=
                        current_blessed_accuracy
                )
                break
        # case 'no prior blessed run'
        if current_blessed_accuracy == 0:
            print("case 'no prior blessed model version found => blessing.'")
            self.model_version_blessed = True

        # self.model_version_blessed = False ### DEBUG - DELETE ###
        if self.model_version_blessed:
            current.run.add_tags(['model_version_blessed'])

        self.next(self.infra_validator)


    @resources(cpu=max(1, os.cpu_count()-2))
    @step
    def infra_validator(self):
        """
        If the trained model version is blessed, validate serving.
        """
        """
        Note that using isolated virtual env (using @conda task decorator)
        is advisable to not embark the whole pipeline dependencies
        into the local TorchServe server.
        We don't for educational purpose, keep things "simple" to grasp
        as well as to avoid forcing conda (for instance miniconda) as
        a virtual environment management mean to the user.
        """

        # tracking below using a 3-states var
        # -1 for not applicable, and
        # 0/1 bool for failure/success otherwize
        self.local_serve_is_ready = -1

        if self.model_version_blessed:
            # serialize model as artifact
            model_file = os.path.join(
                self.serving_artifacts_local_folder, 'model')
            self.model.save_model(model_file)

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
                os.path.join(self.serving_artifacts_local_folder,
                             'Dockerfile.torchserve'))
            # save TabNet TorchServe handler class as artifact
            shutil.copy(
                os.path.join(
                    preprocess_module_dir,
                    'torchserve_tabnet_handler.py'),
                os.path.join(self.serving_artifacts_local_folder,
                             'torchserve_tabnet_handler.py'))
            # save dependencies as artifact
            create_requirements(self.serving_artifacts_local_folder,
                                exclude=[
                                    "torch", # already present
                                             # in the torchserve
                                             # Docker base image
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
            from retrain_pipelines.model.torchserve import \
                await_server_ready, server_has_model, \
                endpoint_still_starting, endpoint_is_ready

            serving_container = build_and_run_docker(
                image_name="tabnet_serve", image_tag="1.0",
                build_path=self.serving_artifacts_local_folder,
                dockerfile="Dockerfile.torchserve",
                ports_publish_dict={
                    '8080/tcp': 9080, '8081/tcp': 9081,
                    '8082/tcp': 9082}
            )
            if not serving_container:
                print("failed spinning the TorchServe container",
                      file=sys.stderr)
                self.local_serve_is_ready = 0
                try:
                    cleanup_docker(container_name="tabnet_serve",
                                   image_name="tabnet_serve:1.0")
                except Exception as cleanup_ex:
                    # fail silently
                    pass
            elif not await_server_ready():
                print("TorchServe server failed to get ready",
                      file=sys.stderr)
                self.local_serve_is_ready = 0
            elif not server_has_model("tabnet_model"):
                print("TorchServe server didn't publish the model",
                      file=sys.stderr)
                self.local_serve_is_ready = 0
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
                        self.local_serve_is_ready = 0
                        break
                    time.sleep(4)
                elapsed_time = time.time() - start_time
                print(f"Elapsed time: {elapsed_time:.3f} seconds")

                # health check on the spun endpoint
                self.local_serve_is_ready = int(
                    endpoint_is_ready("tabnet_model")
                )
            elapsed_time = time.time() - start_time
            print(f"deploy_local -   Elapsed time: {elapsed_time:.2f} seconds")
            ############################################

            if self.local_serve_is_ready == 1:
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
                    self.local_serve_is_ready = 1
                    elapsed_time = time.time() - start_time
                    print(f"inference test -   "
                          f"Elapsed time: {elapsed_time:.2f} seconds")
                except ValidationError as vEx:
                    print(vEx.errors(), file=sys.stderr)
                    print(vEx, file=sys.stderr)
                    traceback.print_tb(vEx.__traceback__, file=sys.stderr)
                    self.local_serve_is_ready = 0
                except Exception as ex:
                    print(ex, file=sys.stderr)
                    traceback.print_tb(ex.__traceback__, file=sys.stderr)
                    self.local_serve_is_ready = 0
                    pass
            try:
                cleanup_docker(container_name="tabnet_serve",
                               image_name="tabnet_serve:1.0")
            except Exception as cleanup_ex:
                # fail silently
                pass

        self.next(self.pipeline_card)


    @card(id='default')
    @card(type='html', id='custom')
    @step
    def pipeline_card(self):
        import re
        import datetime
        import importlib.metadata

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


        ###########################
        ##     "default" card    ##
        ###########################
        self.metadata = {
            "name": "TabNet Model",
            "version": "1.0",
            "retrain_pipelines": f"retrain-pipelines {__version__}",
            "retrain_pipeline_type": os.environ["retrain_pipeline_type"],
            "description": "A PyTorch TabNet model retrained",
            "authors": [current.username],
            "tags": ["classification", "tabnet"],
            "license": "MIT License",
            "model_details": {
                "library": "PyTorch TabNet",
                "version": importlib.metadata.version("pytorch-tabnet")
            },
            "classes_weighted_metrics": self.classes_weighted_metrics,
            "data_sources": [
                {"name": "Training Data", "path": self.data_file}
            ],
            "preprocessing": [
                {
                    "name": "Normalization",
                    "description": "Standard scaling of numerical features"
                },
                {
                    "name": "One-Hot Encoding",
                    "description": "Transforming categorical features " + \
                                   "into binary columns"
                }
            ],
            "references": [
                {
                    "title": "TabNet Documentation",
                    "link": "https://cloud.google.com/blog/products/ai-machine-learning/ml-model-tabnet-is-easy-to-use-on-cloud-ai-platform"
                },
                {
                    "title": "TabNet Architecture",
                    "link": "https://github.com/google-research/google-research/blob/master/tabnet/tabnet_model.py"
                },
                {
                    "title": "PyTorch TabNet Documentation",
                    "link": "https://dreamquark-ai.github.io/tabnet/generated_docs/README.html#source-code"
                },
                {
                    "title": "PyTorch TabNet PyPi Project",
                    "link": "https://pypi.org/project/pytorch-tabnet/"
                }
            ]
        }

        training_image = Image.from_matplotlib(self.training_plt_fig)
        target_class_images = {
            target_class: Image.from_matplotlib(fig)
                          if fig is not None else None
            for target_class, fig in self.target_class_figs.items()
        }

        current.card['default'].append(Markdown(
            "model_version_blessed : **%s**" % str(self.model_version_blessed)))
        current.card['default'].append(Artifact(
            {"model_version_blessed": self.model_version_blessed}))

        current.card['default'].append(training_image)
        for target_class, fig in target_class_images.items():
            current.card['default'].append(
                fig if fig is not None
                else Markdown(
                    "<center><font size=4>**no true positive in test-set " +
                    f"for class '{target_class}'**</font></center>"
                )
            )

        rows = []
        rows.append([float(value)
                     for value in self.classes_weighted_metrics.values()])
        current.card['default'].append(
            Table(rows, headers=list(self.classes_weighted_metrics.keys())))
        ###########################

        ###########################
        ##   html "custom" card  ##
        ###########################
        classes_prior_prob = self.data['target'].value_counts(normalize=True)
        classes_prior_prob = re.sub(
                r'\s+', ', ',
                classes_prior_prob.sort_index().map(
                    lambda x: f"{x*100:.1f}%").to_frame() \
                                    .reset_index() \
                                    .to_string(index=False,header=False)
            )

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
            'classes_prior_prob': classes_prior_prob,
            'features_desc': self.features_desc,
            'data_distri_fig': self.data_distri_fig,
            'data_heatmap_fig': self.data_heatmap_fig,
            'training_plt_fig': self.training_plt_fig,
            'target_class_figs': self.target_class_figs,
            'classes_weighted_metrics_dict': self.classes_weighted_metrics,
            'slice_feature_name': self.first_categorical_feature,
            'sliced_perf_metrics_dict': \
                self.first_categ_feature_sliced_metrics,
            'buckets_dict': self.buckets_param,
            'hyperparameters_dict': flatten_dict(self.hyperparameters,
                                                 callable_to_name=True),
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
            'pipeline_hp_grid_dict': flatten_dict(self.pipeline_hp_grid,
                                                  callable_to_name=True),
            'cv_folds': self.cv_folds,
            'hp_perfs_list': self.hp_perfs_list,
            'task_obj_python_cmd': task_obj_python_cmd,
            'dag_svg': mf_dag_svg(self)
        }
        self.html = get_html(params)
        ###########################
        current
        ###########################

        self.next(self.deploy)


    @step
    def deploy(self):
        """
        placeholder for the serving SDK deploy call
        (on the target production platform).
        consider including the portable pipelione-card itself !
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
    TabNetHpCvWandbFlow()

