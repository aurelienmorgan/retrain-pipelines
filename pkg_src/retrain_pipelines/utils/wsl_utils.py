import logging
import os
import re
import subprocess
from functools import lru_cache

# Matches all valid Windows native path formats:
# 1. Drive paths (strictly single letter A-Z): C:\, D:/, c:\, etc.
# 2. UNC paths: \\server\share, //server/share
# 3. Long device paths: \\?\C:\, \\.\COM1
_WIN_PATH_REGEX = re.compile(r"^([a-zA-Z]:[\\/]|[\\/]{2})")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def is_wsl():
    if os.path.exists("/proc/version"):
        with open("/proc/version") as f:
            version_info = f.read().lower()
            if "microsoft" in version_info:
                return True
    if os.path.exists("/etc/os-release"):
        with open("/etc/os-release") as f:
            os_info = f.read().lower()
            if "WSL" in os_info or "Microsoft" in os_info:
                return True
    return False


@lru_cache(maxsize=128)
def is_windows_path(path_dir) -> bool:
    r"""Check whether the path is a Windows native path.

    (eg. C:\... or \\server\share).
    """
    if not path_dir:
        return False

    path_str = str(path_dir)
    # Syntactic check based on Microsoft Windows path specifications.
    return bool(_WIN_PATH_REGEX.match(path_str))


def is_wsl_mount_path(path_dir) -> bool:
    """Check whether the path WSL path is one a Windows filesystem mount.

    (DrvFs, i.e. path under /mnt/)
    """
    return is_wsl() and path_dir.startswith("/mnt/")


def windows_to_wsl_path(windows_path):
    """Convert Windows path to WSL path.

    Wslpath command.
    (even does so if the directory does not exist).
    """
    wsl_path = subprocess.check_output(["wslpath", windows_path]).decode().strip()
    return wsl_path


def wsl_to_windows_path(wsl_path):
    """
    Convert a WSL path to a Windows-style path.

    Args:
        wsl_path (str): The WSL path to convert.

    Returns
    -------
        str: The converted Windows-style path.
    """
    if not wsl_path.startswith("/mnt/"):
        raise ValueError(
            f"The provided path does not appear to be a WSL mount path : {wsl_path}",
        )

    # Remove the '/mnt/' prefix and replace '/' with '\\'
    windows_path = wsl_path[5:]  # Remove '/mnt/'
    windows_path = windows_path.replace("/", "\\")

    # Convert the drive letter to uppercase
    drive_letter = windows_path[0].upper()
    windows_path = f"{drive_letter}:{windows_path[1:]}"

    return windows_path
