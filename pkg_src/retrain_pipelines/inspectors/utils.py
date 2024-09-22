
import os
import sys
import shutil
import platform
import webbrowser
import subprocess

from ..frameworks import local_metaflow as metaflow
from ..frameworks.local_metaflow.exception import MetaflowNotFound


def _get_mf_run_by_id(
    mf_run_id: int
) -> metaflow.Run:
    """
    
    """
    """
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

    raise MetaflowNotFound(
            f"Run with id {mf_run_id} doesn't exist.")


def _get_mf_run(
    mf_flow_name: str = None,
    mf_run_id: int = -1
):
    """

    Params:
        - mf_flow_name (str):
            Name of the Metaflow flowspec.
            When "mf_run_id" (below) is omitted,
            this param here is mandatory.
        - mf_run_id (int):
            the id of the Metaflow flow run
            to consider.
            If omitted, the last flow run
            is considered.
            If both "mf_flow_name" and "mf_run_id"
            are specified, they indeed must
            be compatible.
            throws MetaflowInvalidPathspec
            or MetaflowNotFound
    """

    if mf_flow_name is None:
        if -1 == mf_run_id:
            raise ValueError(
                    "either `mf_flow_name` or " +
                    "`mf_run_id` have to be specified.")
        else:
            mf_flow_run = _get_mf_run_by_id(mf_run_id)
    else:
        if -1 == mf_run_id:
            mf_flow_run = \
                metaflow.Flow(mf_flow_name).latest_run
        else:
            mf_flow_run = \
                metaflow.Run(f"{mf_flow_name}/{mf_run_id}")

    return mf_flow_run


def _local_pipeline_card_path(
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
            If omitted, the last flow run
            is considered.
            If both "mf_flow_name" and "mf_run_id"
            are specified, they indeed must
            be compatible.
            throws MetaflowInvalidPathspec
            or MetaflowNotFound

    Results:
        - (list[str])
            path to the custom/html card
            in the local datastore
            (accompagnied with "last blessed" card
             if this one is "not blessed").
    """

    mf_flow_run = _get_mf_run(mf_flow_name, mf_run_id)

    pipeline_card_task = [
        step.task for step in mf_flow_run.steps()
        if step.id == 'pipeline_card']
    if not pipeline_card_task:
        if mf_run_id == -1:
            print(f"{mf_flow_run.parent.id} run {mf_flow_run.id}"+
                  " didn't get to the 'pipeline_card' step.",
                  file=sys.stderr)
            return []
        else:
            raise MetaflowNotFound(
                f"{mf_flow_run.parent.id} run {mf_flow_run.id}"+
                  " didn't get to the 'pipeline_card' step.")
    pipeline_card_task = pipeline_card_task[0]

    custom_card = metaflow.cards.get_cards(
        pipeline_card_task, id='custom', type='html')[0]
    # display(custom_card)

    latest_prior_blessed_custom_card_fullname = None
    if (
        "model_version_blessed" in mf_flow_run.data
        and not mf_flow_run.data.model_version_blessed
    ):
        # if pipeline_card relates to a model version
        # that was not blessed, we retrieve
        # both this and "last blessed" pipeline_cards
        mf_flow_name = mf_flow_run._object['flow_id']
        mf_run_id = mf_flow_run._object['run_number']
        # find latest prior blessed model version
        # (from a previous flow-run)
        for latest_prior_blessed_run in metaflow.Flow(mf_flow_name):
            if (
                int(latest_prior_blessed_run.id) < mf_run_id and
                latest_prior_blessed_run.successful and
                'model_version_blessed' \
                    in latest_prior_blessed_run.data and
                latest_prior_blessed_run.data.model_version_blessed
            ):
                blessed_pipeline_card_task = [
                    step.task
                    for step in latest_prior_blessed_run.steps()
                    if step.id == 'pipeline_card'][0]
                latest_prior_blessed_custom_card = metaflow.cards.get_cards(
                    blessed_pipeline_card_task, id='custom', type='html')[0]
                latest_prior_blessed_custom_card_fullname = \
                    os.path.realpath(os.path.join(
                        blessed_pipeline_card_task.metadata_dict.get(
                            "ds-root", None),
                        metaflow.metaflow_config.CARD_SUFFIX,
                        latest_prior_blessed_custom_card.path)
                    )
                break

    custom_card_fullname = os.path.realpath(os.path.join(
        pipeline_card_task.metadata_dict.get("ds-root", None),
        metaflow.metaflow_config.CARD_SUFFIX, custom_card.path)
    )

    return (
                [] if latest_prior_blessed_custom_card_fullname is None
                else [latest_prior_blessed_custom_card_fullname]
           ) + [custom_card_fullname]


def _is_wsl():
    if os.path.exists('/proc/version'):
        with open('/proc/version', 'r') as f:
            version_info = f.read().lower()
            if 'microsoft' in version_info:
                # print(version_info)
                return True
    if os.path.exists('/etc/os-release'):
        with open('/etc/os-release', 'r') as f:
            os_info = f.read().lower()
            if 'WSL' in os_info or 'Microsoft' in os_info:
                # print(os_info)
                return True
    return False


def _windows_to_wsl_path(windows_path):
    """
    # wslpath command to convert Windows path to WSL path
    and does so even if the directory does not exist
    """
    wsl_path = subprocess.check_output(
        ['wslpath', windows_path]).decode().strip()
    return wsl_path


def _wsl_to_windows_path(wsl_path):
    """
    Convert a WSL path to a Windows-style path.

    Args:
        wsl_path (str): The WSL path to convert.

    Returns:
        str: The converted Windows-style path.
    """
    if not wsl_path.startswith('/mnt/'):
        raise ValueError("The provided path does not appear to " +
                         "be a WSL mount path.")

    # Remove the '/mnt/' prefix and replace '/' with '\\'
    windows_path = wsl_path[5:]  # Remove '/mnt/'
    windows_path = windows_path.replace('/', '\\')

    # Convert the drive letter to uppercase
    drive_letter = windows_path[0].upper()
    windows_path = f"{drive_letter}:{windows_path[1:]}"
    
    return windows_path


def _is_windows_path(path_dir):
    if _is_wsl():
        try:
            # Try to convert the path to a Windows path
            result = subprocess.run(
                ['wslpath', '-u', path_dir], 
                capture_output=True, text=True, check=True)
            windows_path = result.stdout.strip()
            # Check if the conversion resulted in a change
            return windows_path != path_dir
        except subprocess.CalledProcessError:
            # likely not a valid Windows path
            return False
    else:
        # If not in WSL, we can't reliably determine the path type
        # assume it's Linux if on Linux OS
        return platform.system().lower() not in ["linux", "darwin"]


def _is_desktop_environment() -> bool:
    """
    wheter or not the host (Linux) OS has GUI
    """
    # Check common environment variables related to DE presence
    de_vars = ["XDG_SESSION_TYPE", "DESKTOP_SESSION", "DISPLAY"]
    for var in de_vars:
        if os.getenv(var): return True
    return False


def _open_explorer(path: str):
    """
    Open OS default file explorer
    on given directory path.
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
                subprocess.run(["explorer.exe", _wsl_to_windows_path(path)])
            elif _is_desktop_environment():
                # Native Linux with DE: Open with xdg-open
                subprocess.run(["xdg-open", path])
            else:
                print("No Desktop Environment detected. "+
                      "Cannot open file explorer.")
        else:
            raise NotImplementedError(f"Unsupported system: {system}")
    except Exception as e:
        print(f"Failed to open explorer on {system}: {e}")


def _webbrowser_open(file_fullnames: list):
    """
    Open local file in OS default web-browser.
    Supports MacOS, native Linux and WSL.

    Note that it does so for the last entry
    of the provided list of file_fullnames.
    For Prior entries, it copies them to
    the same root directory
    (for offline hyperlinking ease).
    """

    system = platform.system()

    if system == "Linux":
        # WSL
        if _is_wsl():
            import ntpath
            # pipeline_card
            temp_dir = subprocess.check_output(
                    "cmd.exe /c echo %TEMP%",
                    shell=True
                ).decode().strip()
            tmp_file_path = ntpath.join(
                temp_dir,
                os.path.basename(file_fullnames[-1])
            )
            shutil.copy2(file_fullnames[-1],
                         _windows_to_wsl_path(tmp_file_path))
            # potential "latest prior_blessed" pipeline_card
            if len(file_fullnames) > 1:
                _tmp_file_path = ntpath.join(
                    temp_dir,
                    "_"+os.path.basename(file_fullnames[0])
                )
                shutil.copy2(file_fullnames[0],
                             _windows_to_wsl_path(_tmp_file_path))
            # actual webbrowser open
            _ = webbrowser.open(f'file:///{tmp_file_path}',
                                new=2)
            return
        elif not _is_desktop_environment():
            # headless native Linux
            # (server distro, for instance)
            print("No Desktop Environment detected. "+
                  "Cannot open file explorer.")
            return

    if len(file_fullnames) > 1:
        parent_dir = os.path.dirname(file_fullnames[-1])
        for file in file_fullnames[:-1]:
            shutil.copy2(
                file,
                os.path.join(parent_dir, "_"+os.path.basename(file))
            )

    # All other cases
    _ = webbrowser.open(f'file://{file_fullnames[-1]}')


def browse_local_pipeline_card(
    mf_flow_name: str = None,
    mf_run_id: int = -1,
    verbose:bool = False
):
    """
    opens the custom/html pipeline card
    for a given flow run into the
    default web browser of the requester.

    Params:
        - mf_flow_name (str):
            Name of the Metaflow flowspec.
            When "mf_run_id" (below) is omitted,
            this param here is mandatory.
        - mf_run_id (int):
            the id of the Metaflow flow run
            to consider.
            If omitted, the last flow run
            is considered.
            throws MetaflowInvalidPathspec
            or MetaflowNotFound
    """

    local_pipeline_card_paths = _local_pipeline_card_path(
        mf_flow_name, mf_run_id=mf_run_id)
    if verbose: print(local_pipeline_card_paths)
    if not local_pipeline_card_paths: return
    _webbrowser_open(local_pipeline_card_paths)

