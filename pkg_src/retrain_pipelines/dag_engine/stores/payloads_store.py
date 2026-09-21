"""
Serialization utilities for DAG task exit payloads.

Artifacts layout under {Config.get_assets_cache_root()}/metadata/ (a.k.a. metadata_root()):
  <exec_id>/<task_id>/payload.pkl  - cloudpickled payload value written at
                                     task func exit when the value is not
                                     JSON-serializable ; JSON-safe values are
                                     stored inline in DB instead.
"""

import os
from typing import Any

import cloudpickle

from ...utils.file_utils import write_binary_file
from ...utils.s3_utils import is_s3_path
from .commons import compute_sha, metadata_root, try_json_serialize


def serialize_task_payload(exec_id: int, task_id: int, value: Any) -> dict:
    """Serialize a task exit payload ; return a row dict ready for DB insert.

    Tries JSON-safe inline first ; falls back to cloudpickle on disk.
    None values are stored as inline null.

    Parameters
    ----------
    exec_id : int
        Execution id.
    task_id : int
        Task id, used to derive the disk artifact path when pickling is needed.
    value : Any
        The raw value returned by the task function.

    Returns
    -------
    dict
        Row ready for insert into task_exit_payloads:
        {task_id, sha, disk_ref, inline_val}.
        sha is None for inline payloads.
    """
    if value is None:
        return {"task_id": task_id, "sha": None, "disk_ref": None, "inline_val": None}

    try:
        json_val = try_json_serialize(value)
        return {"task_id": task_id, "sha": None, "disk_ref": None, "inline_val": json_val}
    except TypeError:
        raw_bytes = cloudpickle.dumps(value)
        sha = compute_sha(value)
        # Relative path for the task exit-payload cloudpickle artifact
        # (relative to metadata_root()).
        # Only written for payloads whose value is not JSON-serializable.
        sep = "/" if is_s3_path(metadata_root()) else os.sep
        rel_path = sep.join([str(exec_id), str(task_id), "payload.pkl"])
        write_binary_file(metadata_root(), [rel_path], raw_bytes)
        return {"task_id": task_id, "sha": sha, "disk_ref": rel_path, "inline_val": None}
