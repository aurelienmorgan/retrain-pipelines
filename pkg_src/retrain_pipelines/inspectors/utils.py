# utils_1.py
import logging
import os
import platform
import shutil
import subprocess
import tempfile
import types
import webbrowser
from datetime import timedelta

import boto3
import requests

from ..dag_engine.config import Config
from ..dag_engine.sdk import ExecutionsIterator
from ..dag_engine.sdk.core import Execution
from ..utils.file_utils import file_exists
from ..utils.s3_utils import is_s3_path, parse_s3_uri
from ..utils.wsl_utils import (
    is_windows_path,
    is_wsl,
    windows_to_wsl_path,
    wsl_to_windows_path,
)

logger = logging.getLogger(__name__)
if not any(type(h) is logging.StreamHandler for h in logger.handlers):
    logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


class NoDesktopEnvException(Exception):
    """Exception raised when no desktop environment (GUI) is detected."""

    pass


def _assert_linux_is_desktop_environment() -> None:
    """Assert whether or not the host OS has a GUI.

    Raises
    ------
        NoDesktopEnvException: If no desktop environment variables are found.
    """
    # Check common environment variables related to Linux DE presence
    de_vars = ["XDG_SESSION_TYPE", "DESKTOP_SESSION", "DISPLAY"]

    for var in de_vars:
        if os.getenv(var):
            # If any variable is found, we have a desktop environment
            return

    # If the loop completes without returning, no DE was found
    raise NoDesktopEnvException("No desktop environment detected. This operation requires a GUI.")


def _open_explorer(path: str):
    """Open OS default file explorer on given directory path.

    Supports MacOS, native Linux and WSL.
    """
    path = os.path.abspath(os.path.expanduser(path))
    system = platform.system()

    try:
        if system == "Darwin":  # macOS
            subprocess.run(["open", path])
        elif system == "Linux":  # Native Linux or WSL
            # Check if running in WSL
            if "microsoft" in platform.uname().release.lower():
                # WSL: Open using Windows Explorer
                subprocess.run(["explorer.exe", wsl_to_windows_path(path)])
            else:
                _assert_linux_is_desktop_environment()  # raises on failure
                # Native Linux with DE: Open with xdg-open
                subprocess.run(["xdg-open", path])
        else:
            raise NotImplementedError(f"Unsupported system: {system}")
    except Exception as e:
        logger.warning(f"Failed to open explorer on {system}: {e}")


def _webbrowser_open(file_fullnames: list, verbose: bool = False) -> bool:
    """Open local file in OS default web-browser.

    Supports MacOS, native Linux and WSL.

    Parameters
    ----------
    file_fullnames : list[str]
        path to the file(s), the last of which will actually be opened.
    verbose : bool
        whether or not local tmp file path shall be logged.

    Returns
    -------
    bool
        success/failure

    Raises
    ------
        NoDesktopEnvException: If on Linux with no desktop environment.
        ValueError: If on WSL and Windows %TEMP% is not accessible.

    Notes
    -----
    - Opens the last entry of the provided list of file_fullnames.
    For prior entries, it copies them to the same root directory
    (for offline hyperlinking ease).
    - Assumes all files are reachable on the calling machine at this point
    (shall have been ensured, e.g. when call on paths retrieved by
    ``_pipeline_card_paths``).
    """
    system = platform.system()
    on_wsl = system == "Linux" and is_wsl()

    if system == "Linux" and not on_wsl:
        # Linux, no desktop, raises on failure ;
        # headless native Linux (server distro, for instance)
        _assert_linux_is_desktop_environment()

    if on_wsl:
        # WSL
        import ntpath

        win_tmp_dir = subprocess.check_output("cmd.exe /c echo %TEMP%", shell=True).decode().strip()
        wsl_tmp_dir = windows_to_wsl_path(win_tmp_dir)
        if not os.path.isdir(wsl_tmp_dir):
            raise ValueError(
                f"Windows %TEMP% ('{win_tmp_dir}') is not accessible via WSL. "
                "Ensure the target drive is mounted "
                "(e.g., verify '[automount] enabled=true' in /etc/wsl.conf)."
            )
        tmp_dir = win_tmp_dir
        path_module: types.ModuleType = ntpath
    else:
        # MacOS or native Linux
        tmp_dir = tempfile.gettempdir()
        path_module = os.path

    # pipeline_card
    tmp_file_basename = (
        f"{path_module.splitext(path_module.basename(file_fullnames[-1]))[0]}"
        f"__{path_module.basename(path_module.dirname(file_fullnames[-1]))}"
        f"{path_module.splitext(path_module.basename(file_fullnames[-1]))[1]}"
    )
    tmp_file_path = path_module.join(tmp_dir, tmp_file_basename)

    if on_wsl:
        target_local_path = windows_to_wsl_path(tmp_file_path)
    else:
        target_local_path = tmp_file_path

    src = file_fullnames[-1]
    if is_s3_path(src):
        bucket, key = parse_s3_uri(src)
        if key.endswith("/"):
            key = key.rstrip("/")
        boto3.client("s3").download_file(bucket, key, target_local_path)
    else:
        src_local = windows_to_wsl_path(src) if on_wsl and is_windows_path(src) else src
        shutil.copy2(src_local, target_local_path)

    if verbose:
        logger.info(target_local_path)

    # potential "prior_blessed" pipeline_card(s)
    for i, prior_file in enumerate(file_fullnames[:-1]):
        _tmp_file_basename = ("_" * (i + 1)) + tmp_file_basename
        _tmp_file_path = path_module.join(tmp_dir, _tmp_file_basename)

        if on_wsl:
            _target_local_path = windows_to_wsl_path(_tmp_file_path)
        else:
            _target_local_path = _tmp_file_path

        if is_s3_path(prior_file):
            _bucket, _key = parse_s3_uri(prior_file)
            if _key.endswith("/"):
                _key = _key.rstrip("/")
            boto3.client("s3").download_file(_bucket, _key, _target_local_path)
        else:
            _src_local = (
                windows_to_wsl_path(prior_file)
                if on_wsl and is_windows_path(prior_file)
                else prior_file
            )
            shutil.copy2(_src_local, _target_local_path)

        if verbose:
            logger.info(_target_local_path)

    # actual webbrowser open
    if on_wsl:
        _ = webbrowser.open(f"file:///{tmp_file_path}", new=2)
    else:
        _ = webbrowser.open(f"file://{tmp_file_path}", new=2)

    return True


def _pipeline_card_path(execution: Execution) -> str | None:
    """Retrieve the path to the pipeline-card file for that execution."""
    pipeline_card_tasks = execution.get_tasks_with_name("pipeline_card")
    if len(pipeline_card_tasks) == 0:
        logger.warning(
            f"Execution of the retraining-pipeline '{execution.name}' "
            f"with id {execution.id} didn't go through a 'pipeline_card' task."
        )
        return None
    pipeline_card_task = pipeline_card_tasks[0]
    logger.debug(f"pipeline_card_task : {pipeline_card_task}")

    pipeline_card_context = pipeline_card_task.get_exit_context()
    if "pipeline_card_file_fullname" not in pipeline_card_context:
        logger.warning(
            f"The exit context of the 'pipeline_card' task "
            f"of '{execution.name}' [id={execution.id}] retraining pipeline execution "
            "doesn't have a 'pipeline_card_file_fullname' entry."
        )
        return None

    return pipeline_card_context["pipeline_card_file_fullname"]


def _exec_model_version_blessed(execution: Execution) -> bool:
    # if pipeline_card relates to a model version
    # that was not blessed, we retrieve
    # both this and "last blessed" pipeline_cards
    exec_model_version_blessed = False
    exec_blessing_tasks = execution.get_tasks_with_name("model_version_blessing")
    if len(exec_blessing_tasks) > 0:
        exec_blessing_context = exec_blessing_tasks[0].get_exit_context()
        exec_model_version_blessed = "model_version_blessed" in exec_blessing_context and bool(
            exec_blessing_context["model_version_blessed"]
        )

    return exec_model_version_blessed


def _pipeline_card_paths(execution: Execution) -> list[str]:
    """Retrieve the pipeline-card file path (and optionally of the previous blessed).

    Only retrieve paths that can be reached on the calling machine.

    Parameters
    ----------
    execution : Execution
        retraining pipeline execution to consider.

    Returns
    -------
    list[str]
        path to the pipeline-card in the execution's artifact store
        (accompagnied by the "last blessed" card if this one is "not blessed").
    """
    exec_pipeline_card_paths = []
    execution_pipeline_card_path = _pipeline_card_path(execution)
    if execution_pipeline_card_path is None:
        return []

    exec_pipeline_card_paths.append(execution_pipeline_card_path)

    if not file_exists(exec_pipeline_card_paths[0]):
        logger.warning(
            f"Couldn't reach '{exec_pipeline_card_paths[0]}',"
            " do you have access setup for the artifact-store "
            f"of the '{execution.name}' [id={execution.id}] "
            "retraining pipeline execution ?"
        )
        return []

    if not _exec_model_version_blessed(execution):
        # the execution's model version wasn't blessed,
        # so we also retrieve the pipeline-card of
        # its best performing predecessor (if any prior was blessed)

        # find latest prior blessed model version
        # (from a previous flow-run)
        for previous_execution in ExecutionsIterator(
            execution.name,
            success_only=True,
            _before_datetime=(execution.start_timestamp - timedelta(milliseconds=1)),
        ):
            if _exec_model_version_blessed(previous_execution):
                previous_execution_pipeline_card_path = _pipeline_card_path(previous_execution)
                if previous_execution_pipeline_card_path is None:
                    continue

                if not file_exists(previous_execution_pipeline_card_path):
                    logger.warning(
                        f"Couldn't reach '{exec_pipeline_card_paths[1]}', "
                        "of the latest blessed model version. Skipping."
                    )
                    logger.warning(
                        "Did you forget to setup the access "
                        f"for the artifact-store of the "
                        f"'{previous_execution.name}' "
                        f"[id={previous_execution.id}] "
                        "retraining pipeline execution ?"
                    )
                else:
                    exec_pipeline_card_paths.insert(0, previous_execution_pipeline_card_path)
                    break

    return exec_pipeline_card_paths


def get_execution(
    pipeline_name: str | None = None,
    exec_id: int = -1,
) -> Execution:
    """Return an execution by its id or pipeline name.

    Convenience method.

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
    Execution

    Raises
    ------
    KeyError
        for non-existing ``exec_id`` entry or
        non-existing ``pipeline_name`` entry
    """
    if exec_id == -1:
        if pipeline_name is None:
            raise KeyError("pipeline_name is required when exec_id is omitted")
        execution = ExecutionsIterator(pipeline_name, page_size=1).previous()
        if execution is None:
            raise KeyError(f"No execution found with pipeline name '{pipeline_name}'")
    else:
        execution = Execution.get_by_id(exec_id)
        if execution is None:
            raise KeyError(f"No execution found with id '{exec_id}'")

    return execution


def open_pipeline_card_file(
    pipeline_name: str | None = None,
    exec_id: int = -1,
    verbose: bool = False,
):
    """Open the portable html pipeline-card.

    For a given pipeline execution, into the
    default web browser of the requester.

    Opens the file from the execution's artifacts-store.
    Bypasses the need for a running WebConsole.

    Parameters
    ----------
    pipeline_name : str
        Name of the retraining-pipeline.
        When ``exec_id`` (below) is omitted,
        this param here is mandatory.
    exec_id : int
        the id of the pipeline execution to consider.
        If omitted, the last execution is considered.
    verbose : bool
        whether the execution shall be verbose.

    Raises
    ------
    KeyError
        non-existing ``pipeline_name`` or ``exec_id`` entry,
        as raised by ``get_execution``.

    Notes
    -----
    if the execution's model version wasn't blessed,
    we also ensure the latest blessed execution's
    pipeline-card is accessible in order for the
    "go to latest blessed" embedded hyperlink
    to go through.
    """
    execution = get_execution(pipeline_name, exec_id)

    pipeline_card_paths = _pipeline_card_paths(execution)

    if verbose:
        logger.info(pipeline_card_paths)

    if len(pipeline_card_paths) == 0:
        return

    if not _webbrowser_open(pipeline_card_paths, verbose=verbose):
        logger.warning("Do you not want to call 'browse_pipeline_card' instead ?")

    return


def browse_pipeline_card(
    pipeline_name: str | None = None,
    exec_id: int = -1,
    verbose: bool = False,
):
    """Open the custom/html pipeline-card.

    For a given pipeline execution, into the
    default web browser of the requester.

    Opens the file from the WebConsole server.

    Parameters
    ----------
    pipeline_name : str
        Name of the retraining-pipeline.
        When ``exec_id`` (below) is omitted,
        this param here is mandatory.
    exec_id : int
        the id of the pipeline execution to consider.
        If omitted, the last execution is considered.
    verbose : bool
        whether the execution shall be verbose.

    Raises
    ------
    AttributeError
        non-existing ``pipeline_name``
    KeyError
        non-exesting ``exec_id``
    """
    # Check whether WebConsole is reachable.
    # The HEAD probe is intentionally fire-and-forget on failure.
    try:
        requests.head(Config.get_web_server_url(), timeout=0.5)
    except requests.exceptions.RequestException:
        logger.warning(
            f"WebConsole not reachable at '{Config.get_web_server_url()}'. "
            "Did you mean to set different values for "
            "RP_WEB_SERVER_URL and/or RP_WEB_SERVER_PORT ?\n"
            "Or maybe did you intend to use `open_pipeline_card_file` "
            "instead ?"
        )
        return

    # input validation
    execution = get_execution(pipeline_name, exec_id)

    if verbose:
        logger.info(f"{execution.name} [{execution.id}]")

    _ = webbrowser.open(Config.get_web_server_url() + f"/pipeline-card?exec_id={execution.id}")

    return
