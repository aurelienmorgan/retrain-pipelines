"""Shared serialization primitives used by both params_store and context_store."""

import hashlib
import os
from datetime import date, datetime
from typing import Any

import cloudpickle
from pydantic import BaseModel

from ..config import Config

DISK_REF_KEY = "__disk_ref__"


def metadata_root() -> str:
    return os.path.join(Config.get_assets_cache_root(), "metadata")


def load_from_disk(rel_path: str) -> Any:
    """Deserialize a cloudpickle artifact from rel_path (relative to metadata_root())."""
    with open(os.path.join(metadata_root(), rel_path), "rb") as fh:
        return cloudpickle.load(fh)


def is_disk_ref(obj: Any) -> bool:
    """Return True if obj is a SHA-envelope dict pointing to a disk artifact."""
    return isinstance(obj, dict) and DISK_REF_KEY in obj


def make_disk_ref(path: str) -> dict:
    """Return a disk-reference sentinel dict pointing to path."""
    return {DISK_REF_KEY: path}


def resolve_storable(obj: Any) -> Any:
    """Resolve a disk-ref sentinel dict to its original value.

    Returns obj unchanged if it is not a disk-ref sentinel.
    """
    if is_disk_ref(obj):
        return load_from_disk(obj[DISK_REF_KEY])
    return obj


def try_json_serialize(obj: Any) -> Any:
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
        return try_json_serialize(obj.model_dump(mode="python"))
    if isinstance(obj, dict):
        return {k: try_json_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [try_json_serialize(v) for v in obj]
    raise TypeError(f"Cannot JSON-serialize {type(obj).__name__}")


def compute_sha(obj: Any) -> str:
    """SHA-256 of cloudpickle.dumps(obj).

    Consistent with value_to_storable (params_store)
    and _serialize_attr (context_store).
    """
    return hashlib.sha256(cloudpickle.dumps(obj)).hexdigest()
