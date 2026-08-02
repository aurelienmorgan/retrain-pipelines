"""
Serialization utilities for DAG task execution context.

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

from ...utils.file_utils import write_binary_file
from .commons import compute_sha, generate_etag, metadata_root, try_json_serialize

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
    exec_id: int, task_id: int, attr_name: str, value: Any, etag: str
) -> tuple[dict, dict]:
    """Serialize value ; return (row_dict, new_ref).

    Tries JSON-safe inline first ; falls back to cloudpickle on disk.
    Disk-pickled values additionally compute a content-based SHA via
    compute_sha() for SDK value-equality comparisons.
    Inline values carry no SHA. The SDK compares them directly by value.

    Parameters
    ----------
    exec_id, task_id : int
        Used to derive the disk artifact path when pickling is needed.
    attr_name : str
        Attribute name (artifact filename stem when pickling).
    value : Any
        Current Python value (must not be None; callers filter None out).
    etag : str
        eTAG generated at serialization time for this attr ; used for
        internal change detection, not SDK comparisons.

    Returns
    -------
    tuple[dict, dict]
        (row ready for bulk-insert into task_context_attrs, updated _attr_ref)
    """
    try:
        json_val = try_json_serialize(value)
        ref = {"eTAG": etag, "disk_ref": None, "inline": json_val}
        row = {
            "task_id": task_id,
            "attr_name": attr_name,
            "eTAG": etag,
            "sha": None,
            "disk_ref": None,
            "inline_val": json_val,
        }
    except TypeError:
        import cloudpickle

        raw_bytes = cloudpickle.dumps(value)
        sha = compute_sha(value)
        rel_path = context_attr_disk_path(exec_id, task_id, attr_name)
        write_binary_file(metadata_root(), [rel_path], raw_bytes)
        ref = {"eTAG": etag, "sha": sha, "disk_ref": rel_path, "inline": None}
        row = {
            "task_id": task_id,
            "attr_name": attr_name,
            "eTAG": etag,
            "sha": sha,
            "disk_ref": rel_path,
            "inline_val": None,
        }
    return row, ref


def snapshot_context_etags(context: Any, exclude: frozenset) -> dict:
    """Return {attr_name: eTAG} for all non-excluded, non-None attrs in context._params.

    For attrs already tracked in _attr_refs the stored eTAG is reused directly ;
    the eTAG is assigned once at value-assignment time .
    A fresh eTAG is generated only for attrs not yet tracked
    (created in sub-DAGs, merged / deep-updated).

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
        result[attr_name] = ref["eTAG"] if ref is not None else generate_etag()
    return result


def compute_context_diff(
    exec_id: int,
    task_id: int,
    context: Any,
    entry_etags: dict,
    exclude: frozenset,
) -> list:
    """Snapshot all surviving non-None context attrs at task exit.

    For each attr in the exit context (skipping excluded and None-valued attrs):
      - eTAG is read from context._attr_refs (assigned at value-assignment time)
        and compared against the entry eTAG.
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
    entry_etags : dict[str, str]
        Snapshot of {attr_name: eTAG} taken at task entry via snapshot_context_etags().
    exclude : frozenset[str]
        Attr names to skip.

    Returns
    -------
    list[dict]
        Rows ready for bulk-insert into task_context_attrs.
        Each dict: {task_id, attr_name, eTAG, sha, disk_ref, inline_val}.
        sha is None for inline attrs ; non-null only for disk-pickled attrs.
    """
    rows = []

    for attr_name, value in context._params.items():
        if attr_name in exclude or value is None:
            continue

        ref = context._attr_refs.get(attr_name)
        current_etag = ref["eTAG"] if ref is not None else None
        entry_etag = entry_etags.get(attr_name)

        if entry_etag is None or current_etag != entry_etag:
            # New or modified: ensure eTAG is set.
            if current_etag is None:
                current_etag = generate_etag()
            row, ref = _serialize_attr(exec_id, task_id, attr_name, value, current_etag)
            context._attr_refs[attr_name] = ref
            rows.append(row)
        else:
            # Unchanged: carry forward the existing ref ; no new file, same disk path.
            ref = context._attr_refs[attr_name]
            rows.append({
                "task_id": task_id,
                "attr_name": attr_name,
                "eTAG": current_etag,
                "sha": ref.get("sha"),  # None for inline attrs
                "disk_ref": ref["disk_ref"],
                "inline_val": ref["inline"],
            })

    return rows
