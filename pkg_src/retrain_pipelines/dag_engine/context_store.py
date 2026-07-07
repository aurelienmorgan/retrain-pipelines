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
The SHA is stored only for disk-pickled values, where it allows change detection
from DB alone without deserializing from disk.
SHA is computed on the Python object (via cloudpickle.dumps(obj)),
independently of the bytes actually written to disk.
"""

import hashlib
import os
import subprocess
import uuid
from datetime import date, datetime
from typing import Any

import cloudpickle
from pydantic import BaseModel

from ..utils.wsl_utils import is_windows_path, wsl_to_windows_path

_DISK_REF_KEY = "__disk_ref__"


def _metadata_root() -> str:
    return os.path.join(os.environ["RP_ASSETS_CACHE"], "metadata")


def temp_dir_id() -> str:
    """Generate a unique temp directory name for use before exec_id is known.

    Format: <YYYYMMDDHHMMSSmmm>_<6-char hex>
            (millisecond timestamp + random suffix).
    """
    return datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3] + "_" + uuid.uuid4().hex[:6]


def _params_subdir_path(dir_id: int | str, subdir: str) -> str:
    """Absolute path to $RP_ASSETS_CACHE/metadata/<dir_id>/params/<subdir>/."""
    return os.path.join(_metadata_root(), str(dir_id), "params", subdir)


def param_disk_path(dir_id: int | str, subdir: str, param_name: str) -> str:
    """Path for a param cloudpickle artifact, relative to _metadata_root().

    The relative form is what gets stored in DB disk_ref dicts,
    avoiding redundant repetition of the _metadata_root() prefix.

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
        # if any param has a default value that requires disk cloudlickling
        dst = _params_subdir_path(exec_id, "defaults")
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


def load_from_disk(rel_path: str) -> Any:
    """Deserialize a cloudpickle artifact from rel_path (relative to _metadata_root())."""
    with open(os.path.join(_metadata_root(), rel_path), "rb") as fh:
        return cloudpickle.load(fh)


def is_disk_ref(obj: Any) -> bool:
    """Return True if obj is a SHA-envelope dict pointing to a disk artifact."""
    return isinstance(obj, dict) and _DISK_REF_KEY in obj


def make_disk_ref(path: str) -> dict:
    """Return a disk-reference sentinel dict pointing to path."""
    return {_DISK_REF_KEY: path}


def resolve_storable(obj: Any) -> Any:
    """Resolve a SHA-envelope dict to its original value.

    Handles both inline envelopes (``__value__``) and disk envelopes
    (``__disk_ref__``). Returns obj unchanged if it is not an envelope.
    """
    if is_disk_ref(obj):
        return load_from_disk(obj[_DISK_REF_KEY])
    return obj


def _try_json_serialize(obj: Any) -> Any:
    """Attempt to produce a JSON-safe representation.

    Raises TypeError for objects that cannot be natively serialized,
    rather than falling back to str().
    """
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, BaseModel):
        return _try_json_serialize(obj.model_dump(mode="python"))
    if isinstance(obj, dict):
        return {k: _try_json_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_try_json_serialize(v) for v in obj]
    raise TypeError(f"Cannot JSON-serialize {type(obj).__name__}")


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
        return _try_json_serialize(obj)
    except TypeError:
        raw_bytes = cloudpickle.dumps(obj)
        sha = hashlib.sha256(raw_bytes).hexdigest()
        rel_path = param_disk_path(dir_id, subdir, param_name)
        abs_path = os.path.join(_metadata_root(), rel_path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "wb") as fh:
            fh.write(raw_bytes)
        return {"__sha__": sha, _DISK_REF_KEY: rel_path}
