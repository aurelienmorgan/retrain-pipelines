
import os
import sys
import time
import logging
import builtins
import threading
import cloudpickle
import multiprocessing
from typing import Callable, Optional, Dict, Any
from concurrent.futures import (
    Executor, 
    Future, 
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed
)


def _cloudpickle_wrapper(pickled_fn_and_args):
    """Unpickle and execute function with args."""
    fn, args, kwargs = cloudpickle.loads(pickled_fn_and_args)
    try:
        return fn(*args, **kwargs)
    finally:
        # Explicitly dispose all SQLAlchemy engines before the worker
        # process exits, so no pooled connections are left open
        # (and i.e. SQLite WAL locks are cleanly released
        #  if running on local SQLite db backend).
        # try:
            # import gc
            # from sqlalchemy.engine import Engine
            # for obj in gc.get_objects():
                # if isinstance(obj, Engine):
                    # try:
                        # obj.dispose()
                    # except Exception:
                        # pass
        # except Exception:
            # pass
        pass


class CloudpickleProcessPoolExecutor(ProcessPoolExecutor):
    """ProcessPoolExecutor that uses cloudpickle for serialization."""

    def __init__(
        self,
        max_workers: Optional[int] = None,
        *,
        mp_context: Optional[multiprocessing.context.BaseContext] = None
    ):
        super().__init__(
            max_workers=max_workers,
            # initializer=None,
            mp_context=mp_context
        )

    def submit(self, fn, *args, **kwargs):
        pickled = cloudpickle.dumps((fn, args, kwargs))
        return super().submit(_cloudpickle_wrapper, pickled)


class HybridPoolExecutor(Executor):
    """Hybrid executor: thread-based scheduling with process-based execution.
    
    Uses threads for coordination and scheduling, but executes each task
    in a separate process for isolation (clean C-level stdout/stderr traces).
    
    This provides:
    - Easy context/state sharing (thread benefit)
    - Process isolation for traces (process benefit)
    - Proper future cancellation support
    """

    def __init__(self, max_workers: Optional[int] = None):
        """Initialize hybrid executor.

        Params:
            max_workers: Maximum number of worker processes (default: CPU count - 2)
        """
        self._max_workers = max(1, (max_workers or os.cpu_count() - 2))
        self._process_pool = CloudpickleProcessPoolExecutor(
            max_workers=self._max_workers,
            mp_context=multiprocessing.get_context("fork")
        )
        self._thread_pool = ThreadPoolExecutor(max_workers=self._max_workers * 2)  # More threads for coordination

        self._shutdown_lock = threading.Lock()
        self._shutdown = False

        # Track all futures: thread_future -> (process_future, thread)
        self._active_futures: Dict[Future, Dict[str, Any]] = {}
        self._futures_lock = threading.Lock()

        self._logger = logging.getLogger(__name__)

    def submit(self, fn: Callable, *args, **kwargs) -> Future:
        """Submit a callable to be executed.

        The callable will be coordinated by a thread but executed in a process.

        Params:
            fn: Callable to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Results:
            Future representing the pending execution
        """
        with self._shutdown_lock:
            if self._shutdown:
                raise RuntimeError("Cannot schedule new futures after shutdown")

        # Create a thread future (what we return to caller)
        thread_future = self._thread_pool.submit(
            self._coordinate_task,
            fn,
            args,
            kwargs
        )

        return thread_future

    def _coordinate_task(self, fn: Callable, args: tuple, kwargs: dict) -> Any:
        """Thread coordinator that submits task to process pool and monitors it.

        This runs in a thread and coordinates the actual process execution.
        """
        # Submit to process pool
        process_future = self._process_pool.submit(fn, *args, **kwargs)

        # Track this coordination
        current_future = None
        with self._futures_lock:
            # Get the current thread's future (the one we're executing in)
            current_thread = threading.current_thread()
            for tf, info in self._active_futures.items():
                if info.get('thread') == current_thread:
                    current_future = tf
                    info['process_future'] = process_future
                    break

            if current_future is None:
                # Fallback: just track the process future
                self._active_futures[process_future] = {
                    'process_future': process_future,
                    'thread': current_thread
                }

        # Wait for process to complete, polling for cancellation
        try:
            while not process_future.done():
                # Check if our thread future was cancelled
                if current_future and current_future.cancelled():
                    self._logger.debug("Thread future cancelled, cancelling process future")
                    process_future.cancel()
                    raise TimeoutError("Task cancelled")

                # Check if process future was cancelled
                if process_future.cancelled():
                    raise TimeoutError("Process future cancelled")

                time.sleep(0.01)  # Small polling interval

            # Return result or raise exception
            return process_future.result()

        finally:
            # Clean up tracking
            with self._futures_lock:
                if current_future in self._active_futures:
                    del self._active_futures[current_future]
                if process_future in self._active_futures:
                    del self._active_futures[process_future]

    def shutdown(self, wait: bool = True, cancel_futures: bool = False):
        """Shutdown the executor.

        Params:
            - wait (bool):
                if true, wait for all pending futures to complete
            - cancel_futures (bool):
                if true, cancel all pending futures
        """
        with self._shutdown_lock:
            if self._shutdown:
                return
            self._shutdown = True

        if cancel_futures:
            self._logger.debug("Cancelling all active futures")
            with self._futures_lock:
                for thread_future, info in list(self._active_futures.items()):
                    if not thread_future.done():
                        thread_future.cancel()

                    process_future = info.get('process_future')
                    if process_future and not process_future.done():
                        process_future.cancel()

        # Shutdown pools in order: process pool first, then thread pool
        self._logger.debug("Shutting down process pool")
        self._process_pool.shutdown(wait=wait, cancel_futures=cancel_futures)

        self._logger.debug("Shutting down thread pool")
        self._thread_pool.shutdown(wait=wait, cancel_futures=cancel_futures)

        self._logger.debug("Hybrid executor shutdown complete")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown(wait=True)
        return False

    def cancel_future(self, future: Future) -> bool:
        """Attempt to cancel a specific future.

        Params:
            future: The future to cancel

        Results:
            True if cancellation was successful
        """
        with self._futures_lock:
            if future in self._active_futures:
                info = self._active_futures[future]
                process_future = info.get('process_future')

                # Cancel both thread and process futures
                cancelled = future.cancel()
                if process_future:
                    cancelled = process_future.cancel() or cancelled

                return cancelled

        return future.cancel()


# Convenience function for backward compatibility
def get_hybrid_executor(max_workers: Optional[int] = None) -> HybridPoolExecutor:
    """Create and return a HybridPoolExecutor instance.

    Params:
        max_workers: Maximum number of worker processes

    Results:
        HybridPoolExecutor instance
    """
    return HybridPoolExecutor(max_workers=max_workers)

