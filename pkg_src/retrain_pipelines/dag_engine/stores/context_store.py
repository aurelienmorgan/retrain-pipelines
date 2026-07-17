"""
Disk serialization utilities for DAG task execution context.

Artifacts layout under {Config.get_assets_cache_root()}/metadata/ (a.k.a. metadata_root()):
  <exec_id>/<task_id>/<attr_name>.pkl  - cloudpickled context attribute values written at
                                         task func exit for attrs whose value is not
                                         JSON-serializable; JSON-safe attrs are stored
                                         inline in DB instead.

Only attrs whose value differs from the task-entry snapshot are written to new disk
artifacts ; unchanged attrs carry forward the existing disk_ref / inline_val from
_attr_refs unchanged ; no new file, same path as previous task's row.
Attrs with None values are ignored (equivalent to deleted entries).
"""

import os
from typing import Any

from .commons import compute_sha, metadata_root, try_json_serialize

# Context attrs injected by dag.init() that must never be serialized as serialized user context
# (since they each already are available as other db metadata fields).
_CONTEXT_EXCLUDE_ATTRS: frozenset = frozenset({"exec_id", "pipeline_name", "username"})


def context_attr_disk_path(exec_id: int, task_id: int, attr_name: str) -> str:
    """Relative path for a context attr cloudpickle artifact (relative to metadata_root()).

    Only written for attrs whose value is not JSON-serializable.

    Parameters
    ----------
    exec_id : int
        Execution id.
    task_id : int
        Task id (defines the artifact subdirectory).
    attr_name : str
        Attribute name (used as the artifact filename stem).
    """
    return os.path.join(str(exec_id), str(task_id), f"{attr_name}.pkl")


def _serialize_attr(
    exec_id: int, task_id: int, attr_name: str, value: Any, current_sha: str
) -> tuple[dict, dict]:
    """Serialize value ; return (row_dict, new_ref).

    Tries JSON-safe inline first ; falls back to cloudpickle on disk.
    Uses the pre-computed current_sha for both branches to avoid
    double-serialization.

    Parameters
    ----------
    exec_id, task_id : int
        Used to derive the disk artifact path when pickling is needed.
    attr_name : str
        Attribute name (artifact filename stem when pickling).
    value : Any
        Current Python value (must not be None; callers filter None out).
    current_sha : str
        Pre-computed sha256(cloudpickle.dumps(value)).

    Returns
    -------
    tuple[dict, dict]
        (row ready for bulk-insert into task_context_attrs, updated _attr_ref)
    """
    try:
        json_val = try_json_serialize(value)
        ref = {"sha": current_sha, "disk_ref": None, "inline": json_val}
        row = {
            "task_id": task_id,
            "attr_name": attr_name,
            "sha": current_sha,
            "disk_ref": None,
            "inline_val": json_val,
        }
    except TypeError:
        import cloudpickle

        raw_bytes = cloudpickle.dumps(value)
        rel_path = context_attr_disk_path(exec_id, task_id, attr_name)
        abs_path = os.path.join(metadata_root(), rel_path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "wb") as fh:
            fh.write(raw_bytes)
        ref = {"sha": current_sha, "disk_ref": rel_path, "inline": None}
        row = {
            "task_id": task_id,
            "attr_name": attr_name,
            "sha": current_sha,
            "disk_ref": rel_path,
            "inline_val": None,
        }
    return row, ref


def snapshot_context_shas(context: Any, exclude: frozenset) -> dict:
    """Return {attr_name: sha} for all non-excluded, non-None attrs in context._params.

    For attrs already tracked in _attr_refs the stored SHA is reused directly ;
    this is the stable SHA computed at the time the value was last written, which
    avoids false-positive change detection caused by cloudpickle non-determinism
    across repeated calls on the same object.
    A fresh SHA is computed only for attrs not yet tracked (new this task).

    Parameters
    ----------
    context : DagExecutionContext
        The current execution context (duck-typed to avoid circular import).
    exclude : frozenset[str]
        Attr names to skip (internal DAG attrs, see _CONTEXT_EXCLUDE_ATTRS).

    Returns
    -------
    dict[str, str]
    """
    result = {}
    for attr_name, value in context._params.items():
        if attr_name in exclude or value is None:
            continue
        ref = context._attr_refs.get(attr_name)
        result[attr_name] = ref["sha"] if ref is not None else compute_sha(value)
    return result


def compute_context_diff(
    exec_id: int,
    task_id: int,
    context: Any,
    entry_shas: dict,
    exclude: frozenset,
) -> list:
    """Snapshot all surviving non-None context attrs at task exit.

    For each attr in the exit context (skipping excluded and None-valued attrs):
      - SHA is computed fresh from the current Python value and compared against
        the entry SHA (which for tracked attrs is the stable stored SHA from
        _attr_refs, preventing false-positive re-serialization).
      - New or modified attrs: serialized via JSON-safe inline when possible,
        cloudpickle to disk otherwise. context._attr_refs updated in-place.
      - Unchanged attrs: existing ref carried forward as-is ; no new file written,
        disk_ref and inline_val are identical to the previous task's row.
      - If no ref exists for an unchanged attr (edge case), the value is serialized
        now so that disk_ref / inline_val are never both absent.

    Parameters
    ----------
    exec_id : int
        Execution id.
    task_id : int
        Task id (disk artifact subdirectory for non-JSON-safe new/modified attrs).
    context : DagExecutionContext
        The execution context at task exit (duck-typed to avoid circular import).
    entry_shas : dict[str, str]
        Snapshot of {attr_name: sha} taken at task entry via snapshot_context_shas().
    exclude : frozenset[str]
        Attr names to skip.

    Returns
    -------
    list[dict]
        Rows ready for bulk-insert into task_context_attrs.
        Each dict: {task_id, attr_name, sha, disk_ref, inline_val}.
    """
    rows = []

    for attr_name, value in context._params.items():
        if attr_name in exclude or value is None:
            continue

        current_sha = compute_sha(value)
        entry_sha = entry_shas.get(attr_name)

        if entry_sha is None or current_sha != entry_sha:
            # New or modified: serialize.
            row, ref = _serialize_attr(exec_id, task_id, attr_name, value, current_sha)
            context._attr_refs[attr_name] = ref
            rows.append(row)
        else:
            # Unchanged: carry forward the existing ref ; no new file, same disk path.
            ref = context._attr_refs.get(attr_name)

            if ref is not None:
                rows.append({
                    "task_id": task_id,
                    "attr_name": attr_name,
                    "sha": current_sha,
                    "disk_ref": ref["disk_ref"],
                    "inline_val": ref["inline"],
                })
            else:
                # ref missing (e.g. type mismatch between DB round-trip and Python object
                # caused entry_sha to be computed on a different representation).
                # Serialize now to establish the ref; will be stable from next task onward.
                row, ref = _serialize_attr(exec_id, task_id, attr_name, value, current_sha)
                context._attr_refs[attr_name] = ref
                rows.append(row)

    return rows
