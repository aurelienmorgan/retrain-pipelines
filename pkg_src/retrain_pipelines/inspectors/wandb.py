
import os

import wandb

from ..frameworks import local_metaflow as metaflow
from ..frameworks.local_metaflow.exception import MetaflowNotFound
from .utils import _get_mf_run, _open_explorer, _is_wsl, \
                   _windows_to_wsl_path, _is_windows_path

os.environ['REQUESTS_CA_BUNDLE'] = \
    "/etc/ssl/certs/ca-certificates.crt"


def _get_execution_source_code(
    train_model_task: metaflow.client.core.Task,
    api: wandb.apis.internal.Api,
) -> list:
    """

    Params:
        - train_model_task (metaflow.client.core.Task):
            the "train_model" task of
            the Metaflow flow run to consider.
        - api (wandb.apis.internal.Api):
            The `Api` instance used
            to retrieve the Files.

    Results:
        - (list[wandb.apis.public.files.File])
    """
    # recall that wandb_run_id = mf_task_id
    # and that source code is logged
    # during the 'train_model' pipeline task.

    #############################
    # retrieve the wandb entity #
    #        for the flow       #
    #############################
    wandb_project_ui_url = \
        train_model_task['wandb_project_ui_url'].data
    wandb_project = train_model_task.path_components[0]
    url_splits = wandb_project_ui_url.split('/')
    wandb_entity = url_splits[url_splits.index(wandb_project) - 1]

    mf_train_model_task_id = train_model_task.id
    #############################

    artifact_files = None
    #############################
    #   Retrieve the WandB run  #
    #############################
    wandb_run = None
    try:
        wandb_run = api.run(
            f"{wandb_entity}/{wandb_project}/{mf_train_model_task_id}")
    except wandb.errors.CommError as cErr:
        if cErr.message.startswith("Could not find run "):
            print(f"Couldn't find run {mf_train_model_task_id} "+
                   "on WandB.")
            if "offline" == train_model_task['wandb_run_mode'].data:
                print("Are you sure it's been synched yet ?")
            artifact_files = []
        else: raise(cErr)
    #############################

    if wandb_run is not None:
        for artifact in wandb_run.logged_artifacts():
            if artifact.name.startswith("source-"):
                print(f"{artifact.source_version} ; "+
                      f"{artifact.updated_at}")
                artifact_files = artifact.files()

    return artifact_files


def get_execution_source_code(
    mf_flow_name: str = None,
    mf_run_id: int = -1
) -> list:
    """

    Params:
        - mf_flow_name (str):
            Name of the Metaflow flowspec.
            When "mf_run_id" (below) is omitted,
            this param here is mandatory.
        - mf_run_id (int):
            the id of the Metaflow flow run
            to consider.
            If ommitted, the last flow run
            is considered.
            throws MetaflowInvalidPathspec
            or MetaflowNotFound

    Results:
        - (list[wandb.apis.public.files.File])
    """

    mf_flow_run = _get_mf_run(mf_flow_name, mf_run_id)

    train_model_task = [
        step.task for step in mf_flow_run.steps()
        if step.id == 'train_model']
    if train_model_task:
        train_model_task = train_model_task[0]
    else:
        raise MetaflowNotFound(
                f"{mf_flow_run.parent.id} run {mf_flow_run.id}"+
                  " didn't get to the 'train_model' step.")

    if "disabled" == train_model_task['wandb_run_mode'].data:
        raise Exception(
                "WandB logging was disabled "+
                "for this pipeline run")

    if mf_run_id != train_model_task.data.wandb_filter_run_id:
        # case "Metaflow flow run was resumed
        #       from aposterior step"
        mf_flow_run = _get_mf_run(
            mf_flow_name,
            train_model_task.data.wandb_filter_run_id
        )
        train_model_task = [
            step.task for step in mf_flow_run.steps()
            if step.id == 'train_model'][0]

    artifact_files = None
    try:
        os.environ['WANDB_DIR'] = \
            train_model_task.metadata_dict.get("ds-root")
        # Make sure to have the `WANDB_API_KEY`
        # environement variable set adequately.
        # It can be through a `secret`.
        _ = wandb.login(
            host="https://api.wandb.ai"
        )
        api = wandb.Api()

        artifact_files = _get_execution_source_code(
            train_model_task, api
        )

    finally:
        try:
            wandb.finish()
        except Exception:
            # fail silently
            pass

    return artifact_files


def explore_source_code(
    root: str = ".",
    mf_flow_name: str = None,
    mf_run_id: int = -1
):
    """
    Downloads local copy of WandB-saved
    source code files in the local target directory.
    Then, opens a file explorer on that directory
    (so long as we're not on an headless OS, like
     via CLI on a server without Desktop Environment).

    Params:
        - root (str):
            target local directory.
        - mf_flow_name (str):
            Name of the Metaflow flowspec.
            When "mf_run_id" (below) is omitted,
            this param here is mandatory.
        - mf_run_id (int):
            the id of the Metaflow flow run
            to consider.
            If ommitted, the last flow run
            is considered.
            throws MetaflowInvalidPathspec
            or MetaflowNotFound
    """

    ##########################
    #      Metaflow task     #
    ##########################
    mf_flow_run = _get_mf_run(mf_flow_name, mf_run_id)
    # for cases when either call argument is missing =>
    mf_flow_name = mf_flow_run.parent.id
    mf_run_id = mf_flow_run.id

    train_model_task = [
        step.task for step in mf_flow_run.steps()
        if step.id == 'train_model'][0]

    if "disabled" == train_model_task['wandb_run_mode'].data:
        raise Exception(
                "WandB logging was disabled "+
                "for this pipeline run")

    if mf_run_id != train_model_task.data.wandb_filter_run_id:
        # case "Metaflow flow run was resumed
        #       from aposterior step"
        mf_flow_run = _get_mf_run(
            mf_flow_name,
            train_model_task.data.wandb_filter_run_id
        )
        train_model_task = [
            step.task for step in mf_flow_run.steps()
            if step.id == 'train_model'][0]
    ##########################

    ##########################
    # local target directory #
    ##########################
    # handle possible "WSL" case
    if _is_wsl() and _is_windows_path(root):
        root = _windows_to_wsl_path(root)
    root = os.path.join(
        root,
        f"{mf_flow_name}_{train_model_task.data.wandb_filter_run_id}"
    )
    print(os.path.realpath(root))
    if not os.path.exists(root):
        os.makedirs(root)
    ##########################

    artifact_files = None
    try:
        os.environ['WANDB_DIR'] = \
            train_model_task.metadata_dict.get("ds-root")
        # Make sure to have the `WANDB_API_KEY`
        # environement variable set adequately.
        # It can be through a `secret`.
        _ = wandb.login(
            host="https://api.wandb.ai"
        )
        api = wandb.Api()

        artifact_files = _get_execution_source_code(
            train_model_task, api
        )

        for source_code_artifact in artifact_files:
            source_code_artifact.download(root,
                                          exist_ok=True)

        _open_explorer(root)

    finally:
        try:
            wandb.finish()
        except Exception:
            # fail silently
            pass

