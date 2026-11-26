import logging
import os

import wandb

from ..dag_engine.config import Config
from ..dag_engine.sdk.core import Execution, Task
from ..utils.wsl_utils import is_windows_path, is_wsl, windows_to_wsl_path
from .utils import _open_explorer, get_execution

logger = logging.getLogger(__name__)
if not any(type(h) is logging.StreamHandler for h in logger.handlers):
    logger.addHandler(logging.StreamHandler())
# logger.setLevel(logging.INFO)


os.environ["REQUESTS_CA_BUNDLE"] = "/etc/ssl/certs/ca-certificates.crt"


def _get_execution_source_code(
    train_model_task: Task,  # type: ignore[name-defined]
    api: wandb.apis.public.api.Api,
) -> list:
    """Retrieve pipeline execution source code from the WandB server.

    Parameters
    ----------
    train_model_task : Task
        the "train_model" task of the retraining-pipeline
        execution to consider.
    api : wandb.apis.public.api.Api
        The `Api` instance used
        to retrieve the Files.

    Returns
    -------
    list[wandb.apis.public.files.File]
    """
    # recall that wandb_run_id = task_id
    # and that source code is logged
    # during the 'train_model' pipeline task.

    #############################
    # retrieve the wandb entity #
    #        for the flow       #
    #############################
    train_model_context = train_model_task.get_exit_context()
    wandb_project_ui_url = train_model_context["wandb_project_ui_url"]
    wandb_entity, wandb_project = wandb_project_ui_url.rsplit("/", 3)[-3:-1]

    rp_train_model_task_id = train_model_task.id
    #############################

    artifact_files = None
    #############################
    #   Retrieve the WandB run  #
    #############################
    wandb_run = None
    api_path = f"{wandb_entity}/{wandb_project}/{rp_train_model_task_id}"
    # print(api_path)
    try:
        wandb_run = api.run(api_path)
    except wandb.errors.CommError as cErr:
        if cErr.message.startswith("Could not find run "):
            logger.warning(f"Couldn't find run {rp_train_model_task_id} on WandB.")
            if "offline" == train_model_context["wandb_run_mode"]:
                print("Are you sure it's been synched yet ?")
            artifact_files = []
        else:
            raise (cErr)
    #############################

    if wandb_run is not None:
        for artifact in wandb_run.logged_artifacts():
            if artifact.name.startswith("source-"):
                print(f"{artifact.source_version} ; {artifact.updated_at}")
                artifact_files = artifact.files()

    return artifact_files


def get_execution_source_code(
    pipeline_name: str | None = None,
    exec_id: int = -1,
) -> list:
    """Retrieve source code from WandB remote.

    Parameters
    ----------
    pipeline_name : str
        Name of the retraining-pipeline.
        When ``exec_id`` (below) is omitted,
        this param here is mandatory.
    exec_id : int
        the id of the pipeline execution to consider.
        If omitted, the last execution is considered.

    Returns
    -------
    list[wandb.apis.public.files.File]
    """
    execution = get_execution(pipeline_name, exec_id)

    train_model_tasks_list = execution.get_tasks_with_name("train_model")
    train_model_task: Task
    if train_model_tasks_list:
        train_model_task = train_model_tasks_list[0]
    else:
        raise ValueError(f"{execution.name} [{execution.id}] didn't get to the 'train_model' task.")

    if "disabled" == train_model_task.get_exit_context()["wandb_run_mode"]:
        raise Exception(
            "WandB logging was disabled for retaining-pipeline execution "
            f"{execution.name} [{execution.id}]."
        )

    if exec_id != train_model_task.get_exit_context()["wandb_filter_run_id"]:
        # case "retraining-pipeline execution was resumed
        #       from a posterior task"
        execution = Execution.get_by_id(train_model_task.get_exit_context()["wandb_filter_run_id"])
        train_model_task = execution.get_tasks_with_name("train_model")[0]

    artifact_files = None
    try:
        os.environ["WANDB_DIR"] = Config.get_assets_cache_root()
        # Make sure to have the `WANDB_API_KEY`
        # environement variable set adequately.
        # It can be through a `secret`.
        _ = wandb.login(host="https://api.wandb.ai")
        api = wandb.Api()

        logger.debug(f"execution.id : {execution.id} - {execution.start_timestamp}")
        artifact_files = _get_execution_source_code(train_model_task, api)

    finally:
        try:
            wandb.finish()
        except Exception:
            # fail silently
            pass

    return artifact_files


def explore_source_code(root: str = ".", pipeline_name: str | None = None, exec_id: int = -1):
    """Download local copy of WandB-saved source code files.

    In the local target directory.

    Then, opens a file explorer on that directory
    (so long as we're not on an headless OS, like
     via CLI on a server without Desktop Environment).

    Parameters
    ----------
    root : str
        target local directory.
    pipeline_name : str
        Name of the retraining-pipeline.
        When ``exec_id`` (below) is omitted,
        this param here is mandatory.
    exec_id : int
        the id of the pipeline execution to consider.
        If omitted, the last execution is considered.
    """
    ##########################
    #          task          #
    ##########################
    execution = get_execution(pipeline_name, exec_id)

    train_model_tasks_list = execution.get_tasks_with_name("train_model")
    train_model_task: Task
    if train_model_tasks_list:
        train_model_task = train_model_tasks_list[0]
    else:
        raise ValueError(f"{execution.name} [{execution.id}] didn't get to the 'train_model' task.")

    if "disabled" == train_model_task.get_exit_context()["wandb_run_mode"]:
        raise Exception(
            "WandB logging was disabled for retaining-pipeline execution "
            f"{execution.name} [{execution.id}]."
        )

    wandb_filter_run_id = train_model_task.get_exit_context()["wandb_filter_run_id"]
    if exec_id != wandb_filter_run_id:
        # case "Metaflow flow run was resumed
        #       from aposterior step"
        execution = Execution.get_by_id(wandb_filter_run_id)
        train_model_task = execution.get_tasks_with_name("train_model")[0]
    ##########################

    ##########################
    # local target directory #
    ##########################
    # handle possible "WSL" case
    if is_wsl() and is_windows_path(root):
        root = windows_to_wsl_path(root)
    root = os.path.join(root, f"{execution.name}_{wandb_filter_run_id}")
    print(os.path.realpath(root))
    if not os.path.exists(root):
        os.makedirs(root)
    ##########################

    artifact_files = None
    try:
        os.environ["WANDB_DIR"] = Config.get_assets_cache_root()
        # Make sure to have the `WANDB_API_KEY`
        # environement variable set adequately.
        # It can be through a `secret`.
        _ = wandb.login(host="https://api.wandb.ai")
        api = wandb.Api()

        artifact_files = _get_execution_source_code(train_model_task, api)

        for source_code_artifact in artifact_files:
            source_code_artifact.download(root, exist_ok=True)

        _open_explorer(root)

    finally:
        try:
            wandb.finish()
        except Exception:
            # fail silently
            pass
