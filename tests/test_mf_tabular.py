
import os
import sys
import platform

import json
import shutil
import tempfile
from textwrap import dedent

import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(
            os.path.dirname(__file__), '../')))

from retrain_pipelines.dataset import DatasetType, \
        pseudo_random_generate
from retrain_pipelines.utils import find_env_python
from retrain_pipelines.local_launcher import \
        retrain_pipelines_local

abort_cmd = (
    '("Command+." to abort)' if platform.system() == "Darwin"
    else '("Ctrl+C" to abort)'
)


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

    # assumes the "requirement.txt" from the subdir
    # of the herein "sample pipeline"
    # are installed in an env named "metaflow_lightgbm"
    # (would it be through conda or venv)
    virtual_env_name = "metaflow_lightgbm"
    print(f"Looking for virtual environment '{virtual_env_name}' " +
          f"{abort_cmd}.. ", end="")
    python_path = find_env_python(virtual_env_name)
    assert python_path, \
           "Virtual environment for this test is missing."
    print("Found.")

    env = os.environ.copy()
    ############################################
    #    replace default python bin in PATH    #
    ############################################
    # drop existing python bin directory from PATH
    path_dirs = env['PATH'].split(os.pathsep)
    path_dirs = [d for d in path_dirs
                 if not os.path.exists(os.path.join(d, 'python'))]
    # prepend current environment python bin directory to PATH
    new_path = os.pathsep.join([os.path.dirname(python_path)] +
                               path_dirs)
    env['PATH'] = new_path
    ############################################

    pipeline_hp_grid = {
        "boosting_type": ["gbdt"],
        "num_leaves": [10],
        "learning_rate": [0.01],
        "n_estimators": [2],
    }
    env['pipeline_hp_grid'] = str(json.dumps(pipeline_hp_grid))

    command = [
        os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "sample_pipelines", "LightGBM_hp_cv_WandB",
            "retraining_pipeline.py"
        ), "run",
        "--data_file", data_file_path,
        "--buckets_param", '{"num_feature1": 100, "num_feature2": 50}',
        "--pipeline_hp_grid", "${pipeline_hp_grid}",
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

    # assumes the "requirement.txt" from the subdir
    # of the herein "sample pipeline"
    # are installed in an env named "metaflow_pytorch_1"
    # (would it be through conda or venv)
    virtual_env_name = "metaflow_pytorch_1"
    print(f"Looking for virtual environment '{virtual_env_name}' " +
          f"{abort_cmd}.. ", end="")
    python_path = find_env_python(virtual_env_name)
    assert python_path, \
           "Virtual environment for this test is missing."
    print("Found.")

    env = os.environ.copy()
    ############################################
    #    replace default python bin in PATH    #
    ############################################
    # drop existing python bin directory from PATH
    path_dirs = env['PATH'].split(os.pathsep)
    path_dirs = [d for d in path_dirs
                 if not os.path.exists(os.path.join(d, 'python'))]
    # prepend current environment python bin directory to PATH
    new_path = os.pathsep.join([os.path.dirname(python_path)] +
                               path_dirs)
    env['PATH'] = new_path
    ############################################

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
    env['pipeline_hp_grid'] = str(
            json.dumps(dedent(
                """{pipeline_hp_grid}""".format(
                    pipeline_hp_grid=pipeline_hp_grid)))
        ).replace("'", '"').strip('"')

    command = [
        os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "sample_pipelines", "TabNet_hp_cv_WandB",
            "retraining_pipeline.py"
        ), "run",
        "--data_file", data_file_path,
        "--buckets_param", '{"num_feature1": 100, "num_feature2": 50}',
        "--pipeline_hp_grid", "${pipeline_hp_grid}",
        "--cv_folds", "2",
        "--wandb_run_mode", "offline"
    ]

    success = retrain_pipelines_local(
        command = " ".join(command),
        env=env
    )

    shutil.rmtree(temp_dir)

    assert success, "retraining pipeline failed."

