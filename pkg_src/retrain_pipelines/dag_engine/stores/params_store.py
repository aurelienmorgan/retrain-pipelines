"""
Serialization utilities for DAG execution parameters.

Artifacts layout under {Config.get_assets_cache_root()}/metadata/ (a.k.a. _metadata_root()):
  <temp_dir_id>/params/defaults/<param_name>.pkl  - cloudpickled DagParam default values
                                                    written before exec_id is known;
                                                    temp_dir_id is a timestamp+uuid string.
  <exec_id>/params/defaults/                      - for local: directory link =>
                                                    ``<temp_dir_id>/params/defaults/``
                                                    (os.symlink on POSIX; mklink /J junction
                                                     on WSL over a Windows DrvFs mount).
                                                    for S3: zero-byte marker object whose key
                                                    name is the src_prefix, pointing navigators
                                                    to the original objects at temp_dir_id.
  <exec_id>/params/overrides/<param_name>.pkl     - cloudpickled execution-time override values
                                                    written after exec_id is known.

Values stored in DB use one of two formats:
  <json_safe_value>                                   - for natively
                                                      serializable values
  {"__eTAG__": "<uuid_hex>", "__sha__": "<sha256hex>", "__disk_ref__": "<rel_path>"}
                                                      - for cloudpickled values
- eTAG is generated at serialization time (uuid4().hex)
    used internally for task-to-task change detection.
- SHA is a content-based digest computed at serialization time via compute_sha()
    used by the SDK for value-equality comparisons.

_attr_refs entries (held in DagExecutionContext._attr_refs) use:
  disk : {"eTAG": "<uuid_hex>", "sha": "<sha256hex>", "disk_ref": "<rel_path>", "inline": None}
  inline: {"eTAG": "<uuid_hex>", "disk_ref": None, "inline": <value>}
"""

import os
import subprocess
import uuid
from datetime import datetime
from typing import Any

import cloudpickle

from ...utils.file_utils import write_binary_file
from ...utils.s3_utils import (
    create_s3_prefix_symlink,
    is_s3_path,
    parse_s3_uri,
    s3_prefix_has_objects,
)
from ...utils.wsl_utils import (
    is_windows_path,
    is_wsl_mount_path,
    wsl_to_windows_path,
)
from .commons import (
    DISK_REF_KEY,
    compute_sha,
    generate_etag,
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
    """Absolute path to {Config.get_assets_cache_root()}/metadata/<dir_id>/params/<subdir>/."""
    return os.path.join(metadata_root(), str(dir_id), "params", subdir)


def link_params_defaults_to_exec(temp_id: str, exec_id: int) -> None:
    """Link metadata/<exec_id>/params/defaults => metadata/<temp_id>/params/defaults.

    For local paths, uses OS-appropriate linking:
    - POSIX (native Linux, macOS): os.symlink
    - WSL on a Windows filesystem mount (DrvFs, i.e. path under /mnt/):
      os.symlink is unreliable on DrvFs; a Windows directory junction
      (cmd.exe mklink /J) is used instead.

    For S3: places a zero-byte marker object at the exec_id prefix whose key
    name is the src_prefix, so the original defaults remain at their temp
    location and are navigable from exec_id without duplication.

    Called in DAG.init() once exec_id is returned by dao.add_execution(),
    so that canonical exec_id-based access resolves correctly.
    """
    if is_s3_path(metadata_root()):
        # Guard: if no defaults were cloudpickled (all inline), nothing was written
        # under the temp prefix => skip entirely,
        # (same as the below local ``os.path.exists(src)`` guard).
        bucket, meta_prefix = parse_s3_uri(metadata_root())
        src_prefix = f"{meta_prefix}{temp_id}/params/defaults/"
        if s3_prefix_has_objects(bucket, src_prefix):
            create_s3_prefix_symlink(
                bucket,
                src_prefix=src_prefix,
                dst_prefix=f"{meta_prefix}{exec_id}/params/defaults/",
            )
        return

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

        if is_windows_path(src) and is_wsl_mount_path(src):
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
    ``__disk_ref__`` (relative path), ``__eTAG__`` (uuid generated at write
    time, for internal change detection) and ``__sha__`` (content-based digest
    via compute_sha(), for SDK value-equality comparisons).

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
        # Relative path for the param cloudpickle artifact (relative to metadata_root()).
        # The relative form is what gets stored in DB disk_ref dicts,
        # avoiding redundant repetition of the metadata_root() prefix.
        sep = "/" if is_s3_path(metadata_root()) else os.sep
        rel_path = sep.join([str(dir_id), "params", subdir, f"{param_name}.pkl"])
        write_binary_file(metadata_root(), [rel_path], raw_bytes)
        return {
            "__eTAG__": generate_etag(),
            "__sha__": compute_sha(obj),
            DISK_REF_KEY: rel_path,
        }


def attr_ref_from_param_storable(storable: Any) -> dict:
    """Build an _attr_ref dict from a param's active storable (from executions.params JSON).

    Parameters
    ----------
    storable : Any
        The raw storable as read from executions.params
        (disk-ref sentinel dict, or inline JSON-safe value).

    Returns
    -------
    dict
        Disk-pickled: {"eTAG": str, "sha": str, "disk_ref": str, "inline": None}
        Inline      : {"eTAG": str, "sha": None, "disk_ref": None, "inline": Any}
        eTAG is used internally for task-to-task change detection.
        sha (disk only) is used by the SDK for value-equality comparisons;
        inline values are compared directly by the SDK without SHA involvement.
    """
    if is_disk_ref(storable):
        return {
            "eTAG": storable["__eTAG__"],
            "sha": storable["__sha__"],
            "disk_ref": storable[DISK_REF_KEY],
            "inline": None,
        }
    # Inline JSON-safe param: no SHA ; SDK compares by value directly.
    return {
        "eTAG": generate_etag(),
        "sha": None,
        "disk_ref": None,
        "inline": storable,
    }
