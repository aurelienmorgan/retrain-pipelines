"""Shared serialization primitives used by stores [e.g. params_store, contexts_store...]."""

import hashlib
import math
import uuid
from datetime import date, datetime
from typing import Any

import cloudpickle
from pydantic import BaseModel

from ...utils.file_utils import build_path, read_binary_file
from ..config import Config

DISK_REF_KEY = "__disk_ref__"


def metadata_root() -> str:
    return build_path(Config.get_assets_cache_root(), ("metadata",))


def load_from_disk(metadata_root: str, rel_path: str) -> Any:
    """Deserialize a cloudpickle artifact from rel_path (relative to metadata_root())."""
    return cloudpickle.loads(read_binary_file(metadata_root, [rel_path]))


def is_disk_ref(obj: Any) -> bool:
    """Return True if obj is an eTAG-envelope dict pointing to a disk artifact."""
    return isinstance(obj, dict) and DISK_REF_KEY in obj


def make_disk_ref(rel_path: str) -> dict:
    """Return a disk-reference sentinel dict pointing to path."""
    return {DISK_REF_KEY: rel_path}


def resolve_storable(metadata_root: str, obj: Any) -> Any:
    """Resolve a disk-ref sentinel dict to its original value.

    Returns obj unchanged if it is not a disk-ref sentinel.
    """
    if is_disk_ref(obj):
        return load_from_disk(metadata_root, obj[DISK_REF_KEY])
    return obj


def try_json_serialize(obj: Any) -> Any:
    """Attempt to produce a JSON-safe representation.

    Raises TypeError for objects that cannot be natively serialized,
    rather than falling back to str().
    """
    if obj is None:
        return None
    if isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, float):
        # NaN / ±Inf are not valid JSON tokens
        if math.isfinite(obj):
            return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, BaseModel):
        return try_json_serialize(obj.model_dump(mode="python"))
    if isinstance(obj, dict):
        return {k: try_json_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [try_json_serialize(v) for v in obj]
    raise TypeError(f"Cannot JSON-serialize {type(obj).__name__} -  {obj!r}")


def generate_etag() -> str:
    """Generate a unique eTAG for a value-assignment event.

    eTAGs are assigned once at serialization time and never derived
    from object content, making internal change-detection fully deterministic.
    Used exclusively by the DAG engine for task-to-task diff tracking.
    """
    return uuid.uuid4().hex


def compute_sha(obj: Any) -> str:
    """Compute a stable content-based SHA-256 for disk-pickled values.

    Used by the SDK for lazy values comparisons (without de-serialization).

    Called exactly once per value at disk-serialization time. The result is
    stored in DB and never recomputed, so cloudpickle non-determinism (which
    caused the same object, e.g. a matplotlib Figure, to hash differently on
    repeated calls) does not affect stored comparisons.

    Only called for non-JSON-serializable values (inline values are compared
    directly by the SDK without SHA involvement).

    Strategy (first match wins):
      - numpy ndarray           : SHA-256 of raw bytes + dtype + shape string.
      - pandas DataFrame/Series : SHA-256 via pd.util.hash_pandas_object.
      - matplotlib Figure       : SHA-256 of a PNG render at 72 dpi. Content-stable
                                  across serialization calls for the same figure.
      - Fallback                : SHA-256 of cloudpickle bytes.
                                  Warning :
                                  Non-deterministic across processes/versions
                                  but stable once stored.

    Parameters
    ----------
    obj : Any
        A non-JSON-serializable Python object.

    Returns
    -------
    str
        64-character lowercase hex SHA-256 digest.
    """
    module = getattr(type(obj), "__module__", "") or ""

    # numpy ndarray
    if "numpy" in module:
        try:
            import numpy as np

            if isinstance(obj, np.ndarray):
                h = hashlib.sha256()
                h.update(obj.tobytes())
                h.update(str(obj.dtype).encode())
                h.update(str(obj.shape).encode())
                return h.hexdigest()
        except Exception:
            pass

    # pandas DataFrame / Series
    if "pandas" in module:
        try:
            import pandas as pd

            if isinstance(obj, (pd.DataFrame, pd.Series)):
                return hashlib.sha256(
                    pd.util.hash_pandas_object(obj, index=True).values.tobytes()
                ).hexdigest()
        except Exception:
            pass

    # matplotlib Figure
    if "matplotlib" in module:
        try:
            import io as _io

            buf = _io.BytesIO()
            obj.savefig(buf, format="png", dpi=72)
            return hashlib.sha256(buf.getvalue()).hexdigest()
        except Exception:
            pass

    # Fallback: cloudpickle (non-deterministic for some types
    # in some edge-circumstances, but harmless since this is
    # computed once at write time and stored)
    # This may lead to edge-cases of false negative in the SDK
    # when comparing equal objects if they were serialized
    # with different revision of cloudpickle (which changed
    # how it pickles them), for instance.
    return hashlib.sha256(cloudpickle.dumps(obj)).hexdigest()
