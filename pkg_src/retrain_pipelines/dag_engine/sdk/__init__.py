import asyncio
import atexit
import logging
import threading
from collections.abc import Coroutine
from datetime import datetime, timedelta
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from pydantic._internal._model_construction import ModelMetaclass

from ..config import Config
from ..db.dao import AsyncDAO

logger = logging.getLogger(__name__)


"""
Permanent background event loop for synchronous SDK methods.

This module provides a single, long-lived asyncio event loop running in a
dedicated daemon thread. All asynchronous operations are scheduled on this
loop via run_coroutine_threadsafe and block the calling thread until
completion.

WHY THIS DESIGN:
- Many async database drivers (e.g., asyncpg) perform deferred cleanup of SSL
  transports after a coroutine has finished. If the event loop is closed
  immediately (as asyncio.run() does), these callbacks will fire on a closed
  loop, causing "Event loop is closed" RuntimeErrors and SSL transport
  failures. By keeping a single permanent loop alive for the entire process
  lifetime, we allow all pending cleanup callbacks to complete safely,
  eliminating those errors.
- The loop runs in a background thread, isolated from any pre-existing event
  loop (e.g., in Jupyter notebooks) and from the main thread. This isolation
  ensures that the SDK never interferes with other async code, and other
  async code never interferes with the SDK.

This design handles all execution environments without special-case logic:
- Scripts and command-line tools: The loop starts on first use and persists
  until process exit.
- Jupyter/IPython notebooks: The background thread runs independently of the
  kernel's own event loop, so no conflicts arise.
- Interactive Python sessions: Same as scripts; the loop remains available
  for the duration of the session.
- Environments with an existing running event loop: The SDK loop is
  independent; calls from any thread are safe and do not affect or depend
  on any pre-existing loop.

The loop is daemonized so it does not prevent process termination. For
long-lived applications, an atexit handler provides a best-effort graceful
shutdown.
"""


# /// event loop deamon ////////////////////////////////////////////////////////////////////////////


# Module-level loop and thread (lazily initialised)
_loop: asyncio.AbstractEventLoop | None = None
_loop_thread: threading.Thread | None = None
_loop_lock = threading.Lock()


def _start_loop() -> None:
    """Start the permanent event loop in a daemon thread."""
    global _loop, _loop_thread
    with _loop_lock:
        if _loop is not None:
            return
        _loop = asyncio.new_event_loop()
        _loop_thread = threading.Thread(
            target=_run_loop_forever, args=(_loop,), daemon=True, name="AsyncSDKEventLoop"
        )
        _loop_thread.start()


def _run_loop_forever(loop: asyncio.AbstractEventLoop) -> None:
    asyncio.set_event_loop(loop)
    loop.run_forever()


def run_async(coro: Coroutine) -> Any:
    """Run a coroutine on the permanent background event loop, blocking until done."""
    _start_loop()
    assert _loop is not None  # type narrowing for mypy

    future = asyncio.run_coroutine_threadsafe(coro, _loop)

    return future.result()


def shutdown_async_sdk():
    global _loop, _loop_thread
    if _loop is not None and _loop.is_running():
        _loop.call_soon_threadsafe(_loop.stop)
    if _loop_thread is not None and _loop_thread.is_alive():
        _loop_thread.join(timeout=2.0)  # wait briefly


atexit.register(shutdown_async_sdk)


# //////////////////////////////////////////////////////////////////////////////////////////////////


# /// pydantic BaseModel, custom dir & help methods ////////////////////////////////////////////////


_PARENT_ATTRS_CACHE: dict[type, frozenset[str]] = {}


def _make_vd(func):
    class _VD:
        """Non-data descriptor for pydantic validator methods.

        Binds to the instance when accessed via an instance, and to
        the class when accessed via the class. This makes ``help()``
        describe the method as belonging to our class rather
        than to pydantic's metaclass.
        """

        def __get__(self, obj, objtype=None):
            from types import MethodType

            return MethodType(func, obj if obj is not None else objtype)

    _VD.__doc__ = func.__doc__
    return _VD()


class _CleanDirMeta(ModelMetaclass):
    """Metaclass that filters ``dir(cls)`` to callable attrs and field annotations.

    Defined directly on the class.

    This makes ``help(cls)`` drop the "Data and other attributes
    defined here" section (``model_config``, ``__annotations__``,
    ``__pydantic_*``, ``__signature__``, etc.) and all "inherited
    from" sections, keeping only "Methods defined here",
    "Static methods defined here", and declared fields.
    """

    def __dir__(cls):
        result = set()
        for name in cls.__dict__:
            try:
                obj = getattr(cls, name, None)
            except Exception:
                continue
            if callable(obj):
                result.add(name)
        ann = getattr(cls, "__annotations__", ())
        result.update(ann)
        return sorted(result)


class BaseModelExt(BaseModel, metaclass=_CleanDirMeta):
    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs):
        """Replace pydantic validator with instance-binding descriptors.

        ``@field_validator`` / ``@model_validator`` wrap the function
        in a ``classmethod``. When accessed via an instance,
        ``help()`` reads ``bound_method.__self__.__class__``, which is
        the metaclass, because ``__self__`` is the class itself.
        Replacing the classmethod with a descriptor that binds to the
        instance makes ``__self__`` the instance, so ``help()`` produces
        "method of <ClassName> instance".

        Note:
        -----
        Pydantic's internal validator dispatch is unaffected. It uses
        the reference stored in ``__pydantic_decorators__``, not the
        ``cls.__dict__`` entry.
        """
        super().__pydantic_init_subclass__(**kwargs)
        for name, obj in list(cls.__dict__.items()):
            if (
                not name.startswith("_")
                and isinstance(obj, classmethod)
                and hasattr(obj, "__wrapped__")
            ):
                try:
                    type.__setattr__(cls, name, _make_vd(obj.__func__))
                except (AttributeError, TypeError):
                    pass

    def __dir__(self):
        """Attributes introduced by this class only.

        Caches ``frozenset(dir(super))`` per parent class globally.
        ``dir()`` is C-implemented and walks the parent's entire MRO
        in one call, so every inherited attribute (from BaseModel,
        object, everything upstream) is collected without a
        Python-level loop. Each MRO base's ``__annotations__`` are
        merged in to catch pydantic fields, which ``dir()`` does not
        expose on the class itself.

        Lazy; zero cost at import for classes never introspected. The
        per-parent frozenset is built exactly once per process and
        shared across every subclass of that parent.

        Returns
        -------
        list[str]
            Attribute names introduced by ``type(self)`` itself.
        """
        cls = type(self)
        parent = cls.__mro__[1]
        parent_attrs = _PARENT_ATTRS_CACHE.get(parent)
        if parent_attrs is None:
            s = set()
            for base in cls.__mro__[1:]:
                s.update(base.__dict__)
                s.update(getattr(base, "__annotations__", ()))
            parent_attrs = frozenset(s)
            _PARENT_ATTRS_CACHE[parent] = parent_attrs
        out = []
        seen = set()
        all_names = (*cls.__dict__, *getattr(cls, "__annotations__", ()))
        for name in all_names:
            in_parent = name in parent_attrs
            if in_parent or name in seen:
                continue
            seen.add(name)
            out.append(name)
        return out


# //////////////////////////////////////////////////////////////////////////////////////////////////


from .core import Execution  # noqa: E402  (after BaseModelExt/run_async to avoid circular import)


class ExecutionsIterator(BaseModelExt):
    """Reversed paged iterator over executions.

    Fetches executions in pages in reverse chronological order
    (newest first) without loading all executions into memory.

    Parameters
    ----------
    pipeline_name : str
        name of the execution pipeline to iterate over
    success_only : bool
        if True, only iterates over successful executions
    page_size : int
        number of executions to fetch per page (default: 10)
    """

    pipeline_name: str = Field(..., description="Name of the execution pipeline")
    success_only: bool | None = Field(False, description="Filter for successful executions only")
    page_size: int = Field(10, description="Number of executions to fetch per page")

    _before_datetime: datetime | None = PrivateAttr(
        default=None  # tracks position in iteration
    )
    _buffer: list[Execution] = PrivateAttr(default_factory=list)  # buffer for current page
    _index: int = PrivateAttr(default=0)  # current position in buffer

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        pipeline_name: str,
        success_only: bool | None = False,
        page_size: int = 10,
        _before_datetime: datetime | None = None,
        **kwargs,
    ) -> None:
        """Initialize the iterator.

        Accepts both positional and named arguments,
        in any combination.

        Parameters
        ----------
        pipeline_name : str
            name of the execution pipeline to iterate over
        success_only : bool, optional
            if True, only iterates over successful executions
            (default: False)
        page_size : int, optional
            number of executions to fetch per page (default: 10)
        _before_datetime : datetime, optional
            internal marker for iteration position (default: None)
        **kwargs
            additional keyword fields forwarded to pydantic's
            BaseModel initializer (useful for subclasses)

        Examples
        --------
        >>> # positional
        >>> ExecutionsIterator("my_pipeline")
        >>> ExecutionsIterator("my_pipeline", True, 20)
        >>> # named
        >>> ExecutionsIterator(pipeline_name="my_pipeline", page_size=20)
        >>> # mixed
        >>> ExecutionsIterator("my_pipeline", page_size=20)
        """
        super().__init__(
            pipeline_name=str(pipeline_name),
            success_only=bool(success_only),
            page_size=int(page_size),
            **kwargs,
        )
        # Pydantic resets PrivateAttrs to their defaults after super().__init__,
        # so we must explicitly set it here to preserve the passed value.
        self._before_datetime = _before_datetime

    async def _previous(self) -> Execution | None:
        """Get the previous (older) execution in the sequence.

        Returns
        -------
        Execution, optional
        previous Execution object in line,
        none if no more executions exist

        Notes
        -----
        Timestamps are stored at millisecond precision.
        We assume no two executions of the same pipeline
        start within the same millisecond
        (at page boundaries).
        """
        if not self._buffer or self._index >= len(self._buffer):
            dao = AsyncDAO(db_url=Config.get_metadatastore_async_url())
            execs = await dao.get_executions_ext(
                pipeline_name=self.pipeline_name,
                execs_status="success" if self.success_only else None,
                before_datetime=self._before_datetime,
                n=self.page_size,
                descending=True,
            )
            if not execs:
                self._buffer = []
                self._index = 0
                return None

            self._buffer = [
                Execution(
                    id=exec_ext.id,
                    name=exec_ext.name,
                    docstring=exec_ext.docstring,
                    metadata_root=exec_ext.metadata_root,
                    artifacts_store_root=exec_ext.artifacts_store_root,
                    username=exec_ext.username,
                    start_timestamp=exec_ext.start_timestamp,
                    end_timestamp=exec_ext.end_timestamp,
                    success=exec_ext.success,
                )
                for exec_ext in execs
            ]

            # Update position marker for next page fetch
            # ensure strictly older results next time
            # (we assume here that no 2 executions
            #  of the same pipeline started
            #  within the same millisecond at page boundaries)
            # (recall that timestamps are purposely
            #  stored at millisecond [not microsecond] precision)
            last_exec = self._buffer[-1]
            self._before_datetime = last_exec.start_timestamp - timedelta(milliseconds=1)

            self._index = 0

        execution = self._buffer[self._index]
        self._index += 1
        return execution

    def previous(self) -> Execution | None:
        """Get the previous (older) execution in the sequence.

        Returns
        -------
        Execution, optional
        previous Execution object in line,
        none if no more executions exist

        Notes
        -----
        Timestamps are stored at millisecond precision.
        We assume no two executions of the same pipeline
        start within the same millisecond
        (at page boundaries).
        """
        logger.debug(f"{self}  - previous")
        return run_async(self._previous())

    def length(self) -> int:
        """Get total count of executions matching the criteria.

        Implementation using SQL COUNT without fetching all rows.

        Returns
        -------
        int
            total number of matching executions.
        """
        dao = AsyncDAO(db_url=Config.get_metadatastore_async_url())
        logger.debug(f"{self}  - length")
        return run_async(
            dao.get_executions_count(
                pipeline_name=self.pipeline_name,
                execs_status="success" if self.success_only else None,
            )
        )

    # async iterator
    def __aiter__(self):
        return self

    async def __anext__(self):
        execution = await self._previous()
        if execution is None:
            raise StopAsyncIteration
        return execution

    # sync iterator
    def __iter__(self):
        return self

    def __next__(self):
        execution = self.previous()
        if execution is None:
            raise StopIteration
        return execution
