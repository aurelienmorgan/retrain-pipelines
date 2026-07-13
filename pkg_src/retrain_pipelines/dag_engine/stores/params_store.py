"""
Disk serialization utilities for DAG execution parameters.

Artifacts layout under $RP_ASSETS_CACHE/metadata/ (a.k.a. _metadata_root()):
  <temp_dir_id>/params/defaults/<param_name>.pkl  - cloudpickled DagParam default values
                                                    written before exec_id is known;
                                                    temp_dir_id is a timestamp+uuid string.
  <exec_id>/params/defaults/                      - directory link => <temp_dir_id>/params/defaults/
                                                    created once exec_id is available
                                                    (os.symlink on POSIX; mklink /J junction
                                                     on WSL over a Windows DrvFs mount).
  <exec_id>/params/overrides/<param_name>.pkl     - cloudpickled execution-time override values
                                                    written after exec_id is known.

Values stored in DB use one of two formats:
  <json_safe_value>                                         - for natively serializable values
  {"__sha__": "<sha256hex>", "__disk_ref__": "<rel_path>"}  - for cloudpickled values
SHA is computed on the raw pickle bytes (sha256(cloudpickle.dumps(obj))).

_attr_refs entries (held in DagExecutionContext._attr_refs) use:
  {"sha": "<sha256hex>", "disk_ref": "<rel_path> | None", "inline": <value> | None}
"""

import hashlib
import os
import subprocess
import uuid
from datetime import datetime
from typing import Any

import cloudpickle

from ...utils.wsl_utils import is_windows_path, wsl_to_windows_path
from .commons import (
    DISK_REF_KEY,
    compute_sha,
    is_disk_ref,
    metadata_root,
    try_json_serialize,
)


def temp_dir_id() -> str:
    """Generate a unique temp directory name for use before exec_id is known.

    Format: <YYYYMMDDHHMMSSmmm>_<6-char hex>
            (millisecond timestamp + random suffix).
    """
    return datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3] + "_" + uuid.uuid4().hex[:6]


def _params_subdir_path(dir_id: int | str, subdir: str) -> str:
    """Absolute path to $RP_ASSETS_CACHE/metadata/<dir_id>/params/<subdir>/."""
    return os.path.join(metadata_root(), str(dir_id), "params", subdir)


def param_disk_path(dir_id: int | str, subdir: str, param_name: str) -> str:
    """Path for a param cloudpickle artifact, relative to metadata_root().

    The relative form is what gets stored in DB disk_ref dicts,
    avoiding redundant repetition of the metadata_root() prefix.

    Parameters
    ----------
    dir_id : int | str
        Either a numeric exec_id or a temp_dir_id string (used before exec_id is known).
    subdir : str
        Sub-directory under params/ (e.g. ``"defaults"`` or ``"overrides"``).
    param_name : str
        Parameter name; used as the artifact filename stem.
    """
    return os.path.join(str(dir_id), "params", subdir, f"{param_name}.pkl")


def link_params_defaults_to_exec(temp_id: str, exec_id: int) -> None:
    """Link metadata/<exec_id>/params/defaults => metadata/<temp_id>/params/defaults.

    Uses OS-appropriate linking:
    - POSIX (native Linux, macOS): os.symlink
    - WSL on a Windows filesystem mount (DrvFs, i.e. path under /mnt/):
      os.symlink is unreliable on DrvFs; a Windows directory junction
      (cmd.exe mklink /J) is used instead.

    Called in DAG.init() once exec_id is returned by dao.add_execution(),
    so that canonical exec_id-based disk access resolves correctly.
    """
    src = _params_subdir_path(temp_id, "defaults")
    if os.path.exists(src):
        # if any param has a default value that requires disk cloudpickling
        dst = _params_subdir_path(exec_id, "defaults")
        # Guard: do not relink if dst already exists (symlink, junction, or dir)
        # from a prior exec with same exec_id
        # (possibly maybe from an old installation using the same cache location).
        if os.path.exists(dst) or os.path.islink(dst):
            return

        os.makedirs(os.path.dirname(dst), exist_ok=True)

        if is_windows_path(src):
            # Windows filesystem (native or WSL DrvFs mount) ; use a directory junction.
            subprocess.run(
                [
                    "cmd.exe",
                    "/c",
                    "mklink",
                    "/J",
                    wsl_to_windows_path(dst),
                    wsl_to_windows_path(src),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            os.symlink(src, dst)


def value_to_storable(dir_id: int | str, subdir: str, param_name: str, obj: Any) -> Any:
    """Return a DB-storable representation of obj.

    Natively JSON-serializable values are returned as-is.
    Everything else is cloudpickled to disk; the returned dict contains
    ``__disk_ref__`` (relative path) and ``__sha__`` (sha256 of the pickle
    bytes) so that change detection requires no deserialization.

    Parameters
    ----------
    dir_id : int | str
        Either a numeric exec_id or a temp_dir_id string (used before exec_id is known).
    subdir : str
        Sub-directory under params/ (e.g. ``"defaults"`` or ``"overrides"``).
    param_name : str
        Parameter name; used to derive the disk artifact filename.
    obj : Any
        Value to serialize.
    """
    try:
        return try_json_serialize(obj)
    except TypeError:
        raw_bytes = cloudpickle.dumps(obj)
        sha = hashlib.sha256(raw_bytes).hexdigest()
        rel_path = param_disk_path(dir_id, subdir, param_name)
        abs_path = os.path.join(metadata_root(), rel_path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "wb") as fh:
            fh.write(raw_bytes)
        return {"__sha__": sha, DISK_REF_KEY: rel_path}


def attr_ref_from_param_storable(storable: Any, resolved_value: Any) -> dict:
    """Build an _attr_ref dict from a param's active storable (from executions.params JSON).

    Parameters
    ----------
    storable : Any
        The raw storable as read from executions.params
        (disk-ref sentinel dict, or inline JSON-safe value).
    resolved_value : Any
        The deserialized Python object (result of resolve_storable(storable)).
        Used to compute SHA for inline values.

    Returns
    -------
    dict
        {"sha": str, "disk_ref": str | None, "inline": Any}
    """
    if is_disk_ref(storable):
        return {"sha": storable["__sha__"], "disk_ref": storable[DISK_REF_KEY], "inline": None}
    # Inline JSON-safe param: SHA computed on the resolved Python object.
    return {"sha": compute_sha(resolved_value), "disk_ref": None, "inline": storable}
