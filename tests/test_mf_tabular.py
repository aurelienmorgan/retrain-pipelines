
import os
import sys

import json
import shutil
import tempfile
from textwrap import dedent

from retrain_pipelines.dataset import DatasetType, \
        pseudo_random_generate
from retrain_pipelines.utils.pytest_utils import \
        get_venv
from retrain_pipelines.utils import \
        as_env_var
from retrain_pipelines.local_launcher import \
        retrain_pipelines_local


##################################################
# Metaflow retraining pipelines for tabular data #
##################################################

def test_mf_lightgbm_regress_tempo():
    data = pseudo_random_generate(
                DatasetType.TABULAR_REGRESSION,
                num_samples = 1_500)
    temp_dir = tempfile.mkdtemp()
    data_file_path = \
        f"{temp_dir}/synthetic_classif_tab_data_continuous.csv"
    data.to_csv(data_file_path, index=False)

    # assumes the "requirements.txt" from the subdir
    # of the herein "sample pipeline"
    # are installed in an env named "metaflow_lightgbm"
    # (would it be through conda or venv)
    env = get_venv(virtual_env_name="metaflow_lightgbm")

    pipeline_hp_grid = {
        "boosting_type": ["gbdt"],
        "num_leaves": [10],
        "learning_rate": [0.01],
        "n_estimators": [2],
    }
    as_env_var(pipeline_hp_grid,
               "pipeline_hp_grid",
               env=env)

    command = [
        os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "sample_pipelines", "LightGBM_hp_cv_WandB",
            "retraining_pipeline.py"
        ), "run",
        "--data_file", data_file_path,
        "--buckets_param", '{"num_feature1": 100, "num_feature2": 50}',
        "--pipeline_hp_grid", "{pipeline_hp_grid}",
        "--cv_folds", "2",
        "--wandb_run_mode", "offline"
    ]

    success = retrain_pipelines_local(
        command = " ".join(command),
        env=env
    )

    shutil.rmtree(temp_dir)

    assert success, "retraining pipeline failed."


def test_mf_tabnet_classif_torchserve():
    data = pseudo_random_generate(
                DatasetType.TABULAR_CLASSIFICATION, num_samples = 1_500)
    temp_dir = tempfile.mkdtemp()
    data_file_path = \
        f"{temp_dir}/synthetic_classif_tab_data_4classes.csv"
    data.to_csv(data_file_path, index=False)

    # assumes the "requirements.txt" from the subdir
    # of the herein "sample pipeline"
    # are installed in an env named "metaflow_pytorch_1"
    # (would it be through conda or venv)
    env = get_venv(virtual_env_name="metaflow_pytorch_1")

    pipeline_hp_grid = {
        "trainer": {
            "max_epochs":[100],
            "patience":[10],
            "batch_size":[1024],
            "virtual_batch_size":[256],
        },
        "model": {
            "n_d":[64],
            "n_a":[64],
            "n_steps":[5],
            "gamma":[1.5],
            "n_independent":[2],
            "n_shared":[2],
            "lambda_sparse":[1e-4],
            "momentum":[0.3],
            "clip_value":[2.],
            "optimizer_fn":["torch.optim.Adam"],
            "optimizer_params":[dict(lr=2e-2), dict(lr=0.1)],
            "scheduler_params":[{"gamma": 0.80,
                                "step_size": 20}],
            "scheduler_fn":["torch.optim.lr_scheduler.StepLR"],
            "epsilon":[1e-15]
        }
    }
    as_env_var(pipeline_hp_grid,
               "pipeline_hp_grid",
               env=env)

    command = [
        os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "sample_pipelines", "TabNet_hp_cv_WandB",
            "retraining_pipeline.py"
        ), "run",
        "--data_file", data_file_path,
        "--buckets_param", '{"num_feature1": 100, "num_feature2": 50}',
        "--pipeline_hp_grid", "{pipeline_hp_grid}",
        "--cv_folds", "2",
        "--wandb_run_mode", "offline"
    ]

    success = retrain_pipelines_local(
        command = " ".join(command),
        env=env
    )

    shutil.rmtree(temp_dir)

    assert success, "retraining pipeline failed."

