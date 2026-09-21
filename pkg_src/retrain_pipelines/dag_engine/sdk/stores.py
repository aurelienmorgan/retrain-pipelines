import contextlib
import os
from typing import Any

from ..core.core import TaskPayload
from ..stores.commons import DISK_REF_KEY, is_disk_ref, load_from_disk


class AttrsDiff:
    """Summary of differences between two attribute snapshots (params or exit context).

    Produced by :meth:`ExecutionParams.diff` and :meth:`TaskExitContext.diff`.
    No deserialization ever occurs.

    Attributes
    ----------
    only_in_self : list[str]
        Attr names present in the left-hand context but absent from the other.
    modified : list[str]
        Attr names present in both contexts whose stored SHA values differ.
    only_in_other : list[str]
        Attr names present in the other context but absent from the left-hand one.
    """

    def __init__(
        self,
        only_in_self: list[str],
        modified: list[str],
        only_in_other: list[str],
    ) -> None:
        self.only_in_self = only_in_self
        self.modified = modified
        self.only_in_other = only_in_other

    def __repr__(self) -> str:
        return (
            f"AttrsDiff("
            f"only_in_self={self.only_in_self}, "
            f"modified={self.modified}, "
            f"only_in_other={self.only_in_other})"
        )


class ExecutionParams:
    """Lazy mapping of DAG param names to their resolved values.

    Values (defaults or execution-time overrides) are deserialized
    from disk only when individually accessed, never all at once.
    Disk paths are resolved against the execution's own metadata_root
    as stored in DB (not the current ``Config.get_assets_cache_root()``).

    For disk-pickled params, value equality can be tested via ``param_equals()``
    using the stored SHA (assigned at write time) ; no deserialization required.

    Parameters
    ----------
    params_json : dict
        Raw ``executions.params`` JSON as stored in DB.
    metadata_root : str
        Absolute path to the metadata root that was active when this
        execution ran (executions.metadata_root column).
    """

    def __init__(self, params_json: dict, metadata_root: str) -> None:
        self._raw = params_json
        self._metadata_root = metadata_root

    def _active_storable(self, key: str) -> Any:
        """Return the active storable (override if present, else default) for key."""
        param_def = self._raw[key]
        return param_def.get("override", param_def.get("default"))

    def _resolve(self, storable: Any) -> Any:
        """Resolve storable using this execution's metadata_root."""
        if is_disk_ref(storable):
            return load_from_disk(self._metadata_root, storable[DISK_REF_KEY])
        return storable

    def __getitem__(self, key: str) -> Any:
        return self._resolve(self._active_storable(key))

    def __contains__(self, key: object) -> bool:
        return key in self._raw

    def __iter__(self):
        return iter(self._raw)

    def __len__(self) -> int:
        return len(self._raw)

    def __repr__(self) -> str:
        return f"ExecutionParams(params={list(self._raw.keys())})"

    def keys(self):
        return self._raw.keys()

    def description(self, key: str) -> str:
        """Return the description string for a param.

        Parameters
        ----------
        key : str
            Param name.

        Returns
        -------
        str
        """
        return self._raw[key]["description"]

    def default(self, key: str) -> Any:
        """Return the resolved default value for a param.

        The default is always returned regardless of whether an override
        is present. Disk-pickled defaults are deserialized on access.

        Parameters
        ----------
        key : str
            Param name.

        Returns
        -------
        Any
        """
        return self._resolve(self._raw[key].get("default"))

    def param_equals(self, key: str, other: "ExecutionParams") -> bool:
        """Return True if key holds the same value in both ExecutionParams instances.

        Comparison is SHA-based for disk-pickled params and direct value equality
        for inline (JSON-safe) params ; no deserialization from disk ever occurs.
        At execution time, SHAs are computed once at serialization time
        via compute_sha() and stored in DB.

        Parameters
        ----------
        key : str
            Param name to compare.
        other : ExecutionParams
            The other ExecutionParams instance to compare against.

        Returns
        -------
        bool
            True if both instances carry the same value for key.

        Raises
        ------
        KeyError
            If key is absent from either instance.

        Examples
        --------
        >>> params_set_a = Execution.getById(id=341).getParams()
        >>> params_set_b = Execution.getById(id=342).getParams()
        >>> # SHA comparison, no unpickling
        >>> params_set_a.param_equals("a_param_name", params_set_b)
        True
        """
        a = self._active_storable(key)
        b = other._active_storable(key)

        def _is_disk(storable: Any) -> bool:
            return isinstance(storable, dict) and DISK_REF_KEY in storable

        a_disk, b_disk = _is_disk(a), _is_disk(b)
        if a_disk != b_disk:
            return False  # one disk-pickled, one inline: different storage types
        if a_disk:
            return a["__sha__"] == b["__sha__"]
        return a == b  # both inline: compare values directly

    def diff(self, other: "ExecutionParams") -> "AttrsDiff":
        """Return a SHA-based diff between this param set and another.

        No deserialization occurs: for entries common to both sets,
        disk-pickled params are compared by their stored SHA
        (computed once at serialization time via compute_sha()).
        Inline params are compared directly by value.
        The active storable (override if present, else default) is used for each
        param on both sides.

        Parameters
        ----------
        other : ExecutionParams
            The param set to compare against.

        Returns
        -------
        AttrsDiff

        Examples
        --------
        >>> a = Execution.get_by_id(id=1).get_params()
        >>> b = Execution.get_by_id(id=2).get_params()
        >>> d = a.diff(b)
        >>> d.only_in_self   # params declared only in execution 1
        ['legacy_flag']
        >>> d.modified       # params present in both but with a different value
        ['dummy_param_1']
        >>> d.only_in_other  # params declared only in execution 2
        []
        """

        def _is_disk(storable: Any) -> bool:
            return isinstance(storable, dict) and DISK_REF_KEY in storable

        def _differs(s: Any, o: Any) -> bool:
            s_disk, o_disk = _is_disk(s), _is_disk(o)
            if s_disk != o_disk:
                return True  # one disk-pickled, one inline: different storage types
            if s_disk:
                return s["__sha__"] != o["__sha__"]
            return s != o  # both inline: compare values directly

        self_keys = set(self._raw)
        other_keys = set(other._raw)
        return AttrsDiff(
            only_in_self=sorted(self_keys - other_keys),
            modified=sorted(
                k
                for k in self_keys & other_keys
                if _differs(self._active_storable(k), other._active_storable(k))
            ),
            only_in_other=sorted(other_keys - self_keys),
        )


class TaskExitContext:
    """Lazy view of a task's exit context.

    Attributes are deserialized from disk (or read from inline storage)
    only when individually accessed. The full attribute index is built
    from a single ``SELECT * FROM task_context_attrs WHERE task_id = ?``
    query ; O(1) regardless of DAG depth or history.

    Disk paths are resolved against the execution's own metadata_root
    as stored in DB (not the current ``Config.get_assets_cache_root()``).

    Parameters
    ----------
    rows : list
        TaskContextAttr ORM rows for this task, as returned by
        ``AsyncDAO.get_task_context_attrs(task_id)``.
    metadata_root : str
        Absolute path to the metadata root that was active when this
        execution ran (executions.metadata_root column).
    """

    def __init__(self, rows: list, metadata_root: str) -> None:
        self._index: dict[str, Any] = {row.attr_name: row for row in rows}
        self._metadata_root = metadata_root

    @contextlib.contextmanager
    def _temp_set_retrain_pipeline_type_env_var(self):
        """Temporarily set retrain_pipeline_type env var, restoring prior state on exit.

        Note
        ----
        This is to allow for the unpickling of artifacts serialized
        from a custom-loaded retrain-pipelines modules (i.e. execution-time
        computed module qualified name).
        """
        if "retrain_pipeline_type" in self._index:
            retrain_pipeline_type = self._index["retrain_pipeline_type"].inline_val

            if retrain_pipeline_type is not None:
                # Use a unique sentinel object to distinguish between "env var not set"
                # and "env var set to an empty string"
                sentinel = object()
                old_retrain_pipeline_type = os.environ.get("retrain_pipeline_type", sentinel)
                os.environ["retrain_pipeline_type"] = retrain_pipeline_type
                try:
                    yield
                finally:
                    if old_retrain_pipeline_type is sentinel:
                        # The env var didn't exist before,
                        # so remove it to restore the initial state
                        os.environ.pop("retrain_pipeline_type", None)
                    else:
                        # The env var existed, restore its original value
                        os.environ["retrain_pipeline_type"] = old_retrain_pipeline_type
                return

        yield

    def __getitem__(self, key: str) -> Any:
        row = self._index[key]
        if row.disk_ref is not None:
            with self._temp_set_retrain_pipeline_type_env_var():
                return load_from_disk(self._metadata_root, row.disk_ref)
        return row.inline_val

    def get(self, key: str, default: Any = None) -> Any:
        """Return the value for key, or default if absent."""
        if key not in self._index:
            return default
        return self[key]

    def __contains__(self, key: object) -> bool:
        return key in self._index

    def __iter__(self):
        return iter(self._index)

    def __len__(self) -> int:
        return len(self._index)

    def __repr__(self) -> str:
        return f"TaskExitContext(attrs={list(self._index.keys())})"

    def keys(self):
        return self._index.keys()

    def attr_equals(self, key: str, other: "TaskExitContext") -> bool:
        """Return True if key holds the same value in both TaskExitContext instances.

        Comparison is SHA-based for disk-pickled attrs and direct value equality
        for inline attrs ; no deserialization from disk ever occurs.
        At execution time, SHAs are computed once at serialization time
        via compute_sha() and stored in DB.

        Parameters
        ----------
        key : str
            Attr name to compare.
        other : TaskExitContext
            The other TaskExitContext instance to compare against.

        Returns
        -------
        bool
            True if both instances carry the same value for key.

        Raises
        ------
        KeyError
            If key is absent from either instance.

        Examples
        --------
        >>> ctx_a = Execution.get_by_id(id=1).get_task_by_id(id=10).get_exit_context()
        >>> ctx_b = Execution.get_by_id(id=2).get_task_by_id(id=20).get_exit_context()
        >>> ctx_a.attr_equals("model_version", ctx_b)
        True
        """
        a_row = self._index.get(key)
        b_row = other._index.get(key)
        if a_row is None and b_row is None:
            return True
        if a_row is None or b_row is None:
            return False

        # Disk-pickled: compare stored SHA.
        if a_row.disk_ref is not None or b_row.disk_ref is not None:
            return a_row.sha == b_row.sha
        # Both inline: compare values directly.
        return a_row.inline_val == b_row.inline_val

    def diff(self, other: "TaskExitContext") -> AttrsDiff:
        """Return a SHA-based diff between this context and another.

        No deserialization occurs: for entries common to both sets,
        disk-pickled attrs are compared by their stored
        SHA (computed once at serialization time via compute_sha()) ;
        inline attrs are compared directly by value.

        Parameters
        ----------
        other : TaskExitContext
            The context to compare against.

        Returns
        -------
        AttrsDiff

        Examples
        --------
        >>> a = Execution.get_by_id(id=1).get_task_by_id(id=10).get_exit_context()
        >>> b = Execution.get_by_id(id=2).get_task_by_id(id=20).get_exit_context()
        >>> d = a.diff(b)
        >>> d.only_in_self   # attrs added or kept only in execution 1's task
        ['model_accuracy']
        >>> d.modified        # attrs present in both but with a different value
        ['pipeline_version']
        >>> d.only_in_other  # attrs present only in execution 2's task
        ['legacy_flag']
        """

        def _differs(a: Any, b: Any) -> bool:
            if a.disk_ref is not None or b.disk_ref is not None:
                return a.sha != b.sha
            return a.inline_val != b.inline_val

        self_keys = set(self._index)
        other_keys = set(other._index)
        return AttrsDiff(
            only_in_self=sorted(self_keys - other_keys),
            modified=sorted(
                k for k in self_keys & other_keys if _differs(self._index[k], other._index[k])
            ),
            only_in_other=sorted(other_keys - self_keys),
        )


class TaskExitPayload(TaskPayload):
    """Lazy view of a task's (or taskgroup's) exit payload.

    Behaves identically to ``core.TaskPayload``, but deserializes the
    payload data from disk (or reads from inline storage) only when explicitly
    accessed (e.g., via ``.value``, iteration, or key access).
    Disk paths are resolved against the execution's own metadata_root
    as stored in DB (not the current ``Config.get_assets_cache_root()``).

    Like ``dag_engine.core.TaskPayload``, it behaves similarly to a dict,
    with task names as keys.
    In cases where there is only 1 entry (e.g., fetching the payload
    of a single task), the following shorthand equivalences hold:
        `payload["task_name"] == payload.get("task_name") == payload`

    Parameters
    ----------
    rows : dict[str, Any]
        A mapping of task_name => TaskPayloadAttr ORM row.
    metadata_root : str
        Absolute path to the metadata root that was active when this
        execution ran.
    """

    def __init__(self, rows: dict[str, Any], metadata_root: str) -> None:
        self._rows = rows
        self._metadata_root = metadata_root
        # inherited, unused
        # self._data = None  # type: ignore[assignment]

    def _load_row(self, row: Any) -> Any:
        if row is None:
            return None
        if row.disk_ref is not None:
            return load_from_disk(self._metadata_root, row.disk_ref)
        return row.inline_val

    @property
    def value(self) -> Any:
        """Deserialize and return the payload value.

        Returns the underlying dictionary, or the single value if only
        one entry exists.
        """
        if len(self._rows) == 1:
            return self._load_row(list(self._rows.values())[0])
        return {k: self._load_row(v) for k, v in self._rows.items()}

    def payload_equals(self, other: "TaskExitPayload") -> bool:
        """Return True if both payloads hold the same value.

        Comparison is SHA-based for disk-pickled payloads and direct value
        equality for inline payloads ; no deserialization from disk ever occurs.

        Parameters
        ----------
        other : TaskExitPayload
            The other payload to compare against.

        Returns
        -------
        bool
        """
        if not isinstance(other, TaskExitPayload):
            return False
        if len(self._rows) != len(other._rows):
            return False
        for k, row in self._rows.items():
            if k not in other._rows:
                return False
            o_row = other._rows[k]
            if row is None and o_row is None:
                continue
            if row is None or o_row is None:
                return False
            a_disk = row.disk_ref is not None
            b_disk = o_row.disk_ref is not None
            if a_disk != b_disk:
                return False
            if a_disk:
                if row.sha != o_row.sha:
                    return False
            else:
                if row.inline_val != o_row.inline_val:
                    return False
        return True

    def __getitem__(self, key: str) -> Any:
        if len(self._rows) == 1:
            try:
                if key in self._rows:
                    return self._load_row(self._rows[key])
            except TypeError:
                pass
            value = self._load_row(list(self._rows.values())[0])
            return value[key]
        return self._load_row(self._rows[key])

    def __setitem__(self, key: str, value: Any) -> None:
        raise TypeError("TaskExitPayload is immutable")

    def __contains__(self, key: object) -> bool:
        return key in self._rows

    def get(self, key: str, default: Any = None) -> Any:
        if key in self._rows:
            return self[key]
        return default

    def keys(self):
        return self._rows.keys()

    def values(self):
        if len(self._rows) == 1:
            yield self._load_row(list(self._rows.values())[0])
        else:
            for v in self._rows.values():
                yield self._load_row(v)

    def items(self):
        if len(self._rows) == 1:
            k, v = list(self._rows.items())[0]
            yield k, self._load_row(v)
        else:
            for k, v in self._rows.items():
                yield k, self._load_row(v)

    def copy(self):
        return TaskExitPayload(self._rows, self._metadata_root)

    def __bool__(self):
        return bool(self._rows)

    def __eq__(self, other):
        if isinstance(other, TaskExitPayload):
            return self.payload_equals(other)
        if len(self._rows) == 1:
            return self._load_row(list(self._rows.values())[0]) == other
        return False

    def __hash__(self):
        if len(self._rows) == 1:
            return hash(self._load_row(list(self._rows.values())[0]))
        return hash(tuple(sorted(self._rows.keys())))

    def __len__(self):
        if len(self._rows) == 1:
            value = self._load_row(list(self._rows.values())[0])
            return len(value) if hasattr(value, "__len__") else 1
        return len(self._rows)

    def __iter__(self):
        if len(self._rows) == 1:
            value = self._load_row(list(self._rows.values())[0])
            if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
                return iter(value)
            return iter([value])
        return iter(self._rows)

    def __add__(self, other):
        if len(self._rows) == 1:
            return self._load_row(list(self._rows.values())[0]) + other
        return NotImplemented

    def __radd__(self, other):
        if len(self._rows) == 1:
            return other + self._load_row(list(self._rows.values())[0])
        return NotImplemented

    def __mul__(self, other):
        if len(self._rows) == 1:
            return self._load_row(list(self._rows.values())[0]) * other
        return NotImplemented

    def __rmul__(self, other):
        if len(self._rows) == 1:
            return other * self._load_row(list(self._rows.values())[0])
        return NotImplemented

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        if len(self._rows) == 1:
            row = list(self._rows.values())[0]
            if row is None:
                raise AttributeError(
                    f"'{self.__class__.__name__}' object has no attribute '{name}'"
                )
            loaded = self._load_row(row)
            return getattr(loaded, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __str__(self):
        if len(self._rows) == 1:
            row = list(self._rows.values())[0]
            val = row.inline_val or row.disk_ref if row is not None else None
            return f"TaskExitPayload({val})"
        val_dict = {
            k: (v.inline_val or v.disk_ref if v is not None else None)
            for k, v in self._rows.items()
        }
        return f"TaskExitPayload({val_dict})"

    def __repr__(self):
        return self.__str__()

    def __getstate__(self):
        return {"_rows": self._rows, "_metadata_root": self._metadata_root}

    def __setstate__(self, state):
        self._rows = state["_rows"]
        self._metadata_root = state["_metadata_root"]
        # inherited, unused
        # self._data = None  # type: ignore[assignment]

    def __reduce__(self):
        return (TaskExitPayload, (self._rows, self._metadata_root))
