import logging
import os
import platform
import shutil
import subprocess
import sys
import webbrowser
from collections.abc import Iterable
from typing import cast
from urllib.parse import urlparse

from ..frameworks import local_metaflow as metaflow
from ..frameworks.local_metaflow import metaflow_config as _mf_config  # type: ignore[attr-defined]
from ..frameworks.local_metaflow.exception import MetaflowNotFound  # type: ignore[import-not-found]
from ..utils.wsl_utils import is_wsl, windows_to_wsl_path, wsl_to_windows_path

logger = logging.getLogger()


def _get_mf_run_by_id(mf_run_id: int) -> metaflow.Run:
    """Retrieve Flow instance.

    Note : Metaflow doesn’t provide
           a direct function to get
           a run by ID across all flows.
           So iterate until we find it..
    """
    mf = metaflow.Metaflow()
    for mf_flow in mf.flows:
        try:
            run = metaflow.Run(f"{mf_flow.id}/{mf_run_id}")
            return run
        except MetaflowNotFound:
            # Run ID not found in this flow,
            # continue to next flow
            continue

    raise MetaflowNotFound(f"Run with id {mf_run_id} doesn't exist.")


def _get_mf_run(
    mf_flow_name: str | None = None,
    mf_run_id: int = -1,
) -> metaflow.Run:
    """Retrieve Flow instance.

    Parameters
    ----------
    mf_flow_name : str
        Name of the Metaflow flowspec.
        When "mf_run_id" (below) is omitted,
        this param here is mandatory.
    mf_run_id : int
        the id of the Metaflow flow run
        to consider.
        If omitted, the last flow run
        is considered.
        If both "mf_flow_name" and "mf_run_id"
        are specified, they indeed must
        be compatible.
        throws MetaflowInvalidPathspec
        or MetaflowNotFound

    Returns
    -------
    metaflow.Run
    """
    if mf_flow_name is None:
        if -1 == mf_run_id:
            raise ValueError("either `mf_flow_name` or " + "`mf_run_id` have to be specified.")
        else:
            mf_flow_run = _get_mf_run_by_id(mf_run_id)
    else:
        if -1 == mf_run_id:
            _latest = metaflow.Flow(mf_flow_name).latest_run
            if _latest is None:
                raise MetaflowNotFound(f"No runs found for flow {mf_flow_name}")
            mf_flow_run = _latest
        else:
            mf_flow_run = metaflow.Run(f"{mf_flow_name}/{mf_run_id}")

    return mf_flow_run


def _pipeline_card_task(
    mf_flow_name: str | None = None,
    mf_run_id: int = -1,
) -> metaflow.Task | None:
    """Retrieve pipeline-card Task instance.

    Parameters
    ----------
    mf_flow_name : str
        Name of the Metaflow flowspec.
        When "mf_run_id" (below) is omitted,
        this param here is mandatory.
    mf_run_id : int
        the id of the Metaflow flow run
        to consider.
        If omitted, the last flow run
        is considered.
        If both "mf_flow_name" and "mf_run_id"
        are specified, they indeed must
        be compatible.
        throws MetaflowInvalidPathspec
        or MetaflowNotFound

    Returns
    -------
    metaflow.Task
    """
    mf_flow_run = _get_mf_run(mf_flow_name, mf_run_id)

    pipeline_card_task = [step.task for step in mf_flow_run.steps() if step.id == "pipeline_card"]
    if not pipeline_card_task:
        if mf_run_id == -1:
            assert mf_flow_run.parent is not None
            print(
                f"{mf_flow_run.parent.id} run {mf_flow_run.id}"
                + " didn't get to the 'pipeline_card' step.",
                file=sys.stderr,
            )
            return None
        else:
            assert mf_flow_run.parent is not None
            raise MetaflowNotFound(
                f"{mf_flow_run.parent.id} run {mf_flow_run.id}"
                + " didn't get to the 'pipeline_card' step."
            )

    return pipeline_card_task[0]


def _local_pipeline_card_path(mf_flow_name: str | None = None, mf_run_id: int = -1) -> list:
    """Retrieve pipeline-card html file path.

    Parameters
    ----------
    mf_flow_name : str
        Name of the Metaflow flowspec.
        When "mf_run_id" (below) is omitted,
        this param here is mandatory.
    mf_run_id : int
        the id of the Metaflow flow run
        to consider.
        If omitted, the last flow run
        is considered.
        If both "mf_flow_name" and "mf_run_id"
        are specified, they indeed must
        be compatible.
        throws MetaflowInvalidPathspec
        or MetaflowNotFound

    Returns
    -------
    list[str]
        path to the custom/html card
        in the local datastore
        (accompagnied with "last blessed" card
         if this one is "not blessed").
    """
    pipeline_card_task = _pipeline_card_task(mf_flow_name, mf_run_id)
    if pipeline_card_task is None:
        return []

    try:
        custom_card = metaflow.cards.get_cards(pipeline_card_task, id="custom", type="html")[0]
    except IndexError:
        logger.warn(f"No custom/html card exists for {pipeline_card_task}")
        return []

    assert pipeline_card_task.parent is not None
    mf_flow_run = pipeline_card_task.parent.parent
    assert mf_flow_run is not None

    latest_prior_blessed_custom_card_fullname = None
    if (
        "model_version_blessed" in mf_flow_run.data  # type: ignore[attr-defined]
        and not mf_flow_run.data.model_version_blessed  # type: ignore[attr-defined]
    ):
        # if pipeline_card relates to a model version
        # that was not blessed, we retrieve
        # both this and "last blessed" pipeline_cards
        mf_flow_name = mf_flow_run._object["flow_id"]
        mf_run_id = mf_flow_run._object["run_number"]
        # find latest prior blessed model version
        # (from a previous flow-run)
        for latest_prior_blessed_run in cast(Iterable[metaflow.Run], metaflow.Flow(mf_flow_name)):
            if (
                int(latest_prior_blessed_run.id) < mf_run_id
                and latest_prior_blessed_run.successful
                and "model_version_blessed" in latest_prior_blessed_run.data  # type: ignore[operator]
                and latest_prior_blessed_run.data.model_version_blessed  # type: ignore[union-attr]
            ):
                blessed_pipeline_card_task = [
                    step.task
                    for step in latest_prior_blessed_run.steps()
                    if step.id == "pipeline_card"
                ][0]
                assert blessed_pipeline_card_task is not None
                latest_prior_blessed_custom_card = metaflow.cards.get_cards(
                    blessed_pipeline_card_task, id="custom", type="html"
                )[0]
                latest_prior_blessed_custom_card_fullname = os.path.realpath(
                    os.path.join(
                        blessed_pipeline_card_task.metadata_dict.get("ds-root") or "",
                        _mf_config.CARD_SUFFIX,
                        latest_prior_blessed_custom_card.path,
                    )
                )
                break

    custom_card_fullname = os.path.realpath(
        os.path.join(
            pipeline_card_task.metadata_dict.get("ds-root") or "",
            _mf_config.CARD_SUFFIX,
            custom_card.path,
        )
    )

    return (
        []
        if latest_prior_blessed_custom_card_fullname is None
        else [latest_prior_blessed_custom_card_fullname]
    ) + [custom_card_fullname]


def _is_desktop_environment() -> bool:
    """Wheter or not the host (Linux) OS has GUI."""
    # Check common environment variables related to DE presence
    de_vars = ["XDG_SESSION_TYPE", "DESKTOP_SESSION", "DISPLAY"]
    for var in de_vars:
        if os.getenv(var):
            return True

    return False


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
            elif _is_desktop_environment():
                # Native Linux with DE: Open with xdg-open
                subprocess.run(["xdg-open", path])
            else:
                print("No Desktop Environment detected. " + "Cannot open file explorer.")
        else:
            raise NotImplementedError(f"Unsupported system: {system}")
    except Exception as e:
        print(f"Failed to open explorer on {system}: {e}")


def _webbrowser_open(file_fullnames: list) -> bool:
    """Open local file in OS default web-browser.

    Supports MacOS, native Linux and WSL.

    Note that it does so for the last entry
    of the provided list of file_fullnames.
    For Prior entries, it copies them to
    the same root directory
    (for offline hyperlinking ease).

    Parameters
    ----------
    file_fullnames : list[str]
        path to the file.

    Returns
    -------
    bool
        success/failure
    """
    system = platform.system()

    if system == "Linux":
        # WSL
        if is_wsl():
            import ntpath

            # pipeline_card
            temp_dir = (
                subprocess.check_output("cmd.exe /c echo %TEMP%", shell=True).decode().strip()
            )
            tmp_file_path = ntpath.join(temp_dir, os.path.basename(file_fullnames[-1]))
            shutil.copy2(file_fullnames[-1], windows_to_wsl_path(tmp_file_path))
            # potential "latest prior_blessed" pipeline_card
            if len(file_fullnames) > 1:
                _tmp_file_path = ntpath.join(temp_dir, "_" + os.path.basename(file_fullnames[0]))
                shutil.copy2(file_fullnames[0], windows_to_wsl_path(_tmp_file_path))
            # actual webbrowser open
            _ = webbrowser.open(f"file:///{tmp_file_path}", new=2)
            return True
        elif not _is_desktop_environment():
            # headless native Linux
            # (server distro, for instance)
            print(
                "No Desktop Environment detected. " + "Cannot open host-local file on web-browser."
            )
            return False

    if len(file_fullnames) > 1:
        parent_dir = os.path.dirname(file_fullnames[-1])
        for file in file_fullnames[:-1]:
            shutil.copy2(file, os.path.join(parent_dir, "_" + os.path.basename(file)))

    # All other cases
    _ = webbrowser.open(f"file://{file_fullnames[-1]}")

    return True


def browse_local_pipeline_card(
    mf_flow_name: str | None = None, mf_run_id: int = -1, verbose: bool = False
):
    """Open the custom/html pipeline card.

    For a given pipeline execution, into the
    default web browser of the requester.

    Parameters
    ----------
    mf_flow_name : str
        Name of the Metaflow flowspec.
        When "mf_run_id" (below) is omitted,
        this param here is mandatory.
    mf_run_id : int
        the id of the Metaflow flow run
        to consider.
        If omitted, the last flow run
        is considered.
        throws MetaflowInvalidPathspec
        or MetaflowNotFound
    verbose : bool
        whether the execution shall be verbose.
    """
    local_pipeline_card_paths = _local_pipeline_card_path(mf_flow_name, mf_run_id=mf_run_id)
    if local_pipeline_card_paths and len(local_pipeline_card_paths) > 0:
        if verbose:
            print(local_pipeline_card_paths)
        if not local_pipeline_card_paths:
            return
        if not _webbrowser_open(local_pipeline_card_paths):
            logger.warn("Do you not want to call " + "'browse_pipeline_card' instead ?")


def browse_pipeline_card(
    mf_backend_service_url: str,
    mf_flow_name: str | None = None,
    mf_run_id: int = -1,
    verbose: bool = False,
):
    """Open the custom/html pipeline card.

    For a given flow run into the
    default web browser of the requester.

    Parameters
    ----------
    mf_backend_service_url : str
        URL of the Metaflow Metadata service
        (among other things, its the service
         in charge of serving cards to the Metaflow UI).
    mf_flow_name : str
        Name of the Metaflow flowspec.
        When "mf_run_id" (below) is omitted,
        this param here is mandatory.
    mf_run_id : int
        the id of the Metaflow flow run
        to consider.
        If omitted, the last flow run
        is considered.
        throws MetaflowInvalidPathspec
        or MetaflowNotFound
    verbose : bool
        whether the execution shall be verbose.
    """
    try:
        result = urlparse(mf_backend_service_url)
        assert all([result.scheme, result.netloc]), ValueError(
            f"Invalid URL: '{mf_backend_service_url}'"
        )
    except ValueError:
        raise ValueError(f"Invalid URL: '{mf_backend_service_url}'") from None

    pipeline_card_task = _pipeline_card_task(mf_flow_name, mf_run_id)
    if pipeline_card_task is None:
        return

    custom_card = None
    try:
        custom_card = metaflow.cards.get_cards(pipeline_card_task, id="custom", type="html")[0]
    except IndexError:
        logger.warn(f"No custom/html card exists for {pipeline_card_task}")
        return

    mf_flow_name = pipeline_card_task.path_components[0]
    mf_run_id = int(pipeline_card_task.path_components[1])
    mf_task_id = pipeline_card_task.path_components[3]
    pipeline_card_hash = custom_card.hash

    pipeline_card_url = (
        f"{mf_backend_service_url}/flows/{mf_flow_name}/runs/{mf_run_id}/steps/"
        f"pipeline_card/tasks/{mf_task_id}/cards/{pipeline_card_hash}"
    )
    if verbose:
        print(pipeline_card_url)
    _ = webbrowser.open(pipeline_card_url)
