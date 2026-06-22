"""
Unit tests for retrain_pipelines.dag_engine.hybrid_pool_executor
"""

import os
import threading
import time
from concurrent.futures import Future

import cloudpickle
import pytest

from retrain_pipelines.dag_engine.hybrid_pool_executor import (
    CloudpickleProcessPoolExecutor,
    HybridPoolExecutor,
    _cloudpickle_wrapper,
    get_hybrid_executor,
)


# ══════════════════════════════════════════════════════════════════════════════
#  _cloudpickle_wrapper
# ══════════════════════════════════════════════════════════════════════════════


class TestCloudpickleWrapper:
    def test_returns_scalar_result(self):
        packed = cloudpickle.dumps((lambda x, y: x + y, (2, 3), {}))
        assert _cloudpickle_wrapper(packed) == 5

    def test_passes_kwargs(self):
        packed = cloudpickle.dumps((lambda a, b=10: a * b, (3,), {"b": 7}))
        assert _cloudpickle_wrapper(packed) == 21

    def test_no_args(self):
        packed = cloudpickle.dumps((lambda: 99, (), {}))
        assert _cloudpickle_wrapper(packed) == 99

    def test_propagates_exception(self):
        def boom():
            raise ValueError("oops")

        packed = cloudpickle.dumps((boom, (), {}))
        with pytest.raises(ValueError, match="oops"):
            _cloudpickle_wrapper(packed)


# ══════════════════════════════════════════════════════════════════════════════
#  CloudpickleProcessPoolExecutor
# ══════════════════════════════════════════════════════════════════════════════


class TestCloudpickleProcessPoolExecutor:
    def test_submit_simple_lambda(self):
        with CloudpickleProcessPoolExecutor(max_workers=1) as ex:
            assert ex.submit(lambda: 42).result(timeout=10) == 42

    def test_submit_with_positional_args(self):
        with CloudpickleProcessPoolExecutor(max_workers=1) as ex:
            assert ex.submit(lambda a, b: a - b, 10, 3).result(timeout=10) == 7

    def test_submit_with_kwargs(self):
        with CloudpickleProcessPoolExecutor(max_workers=1) as ex:
            assert ex.submit(lambda x=1, y=2: x + y, x=5, y=6).result(timeout=10) == 11

    def test_exception_from_child_propagates(self):
        def raise_val():
            raise ValueError("child error")

        with CloudpickleProcessPoolExecutor(max_workers=1) as ex:
            with pytest.raises(ValueError, match="child error"):
                ex.submit(raise_val).result(timeout=10)


# ══════════════════════════════════════════════════════════════════════════════
#  HybridPoolExecutor
# ══════════════════════════════════════════════════════════════════════════════


class TestHybridPoolExecutor:
    def test_submit_returns_correct_result(self):
        with HybridPoolExecutor(max_workers=2) as ex:
            assert ex.submit(lambda: 42).result(timeout=15) == 42

    def test_submit_with_positional_args(self):
        with HybridPoolExecutor(max_workers=2) as ex:
            assert ex.submit(lambda a, b: a * b, 3, 4).result(timeout=15) == 12

    def test_multiple_futures_independent_results(self):
        # cloudpickle bakes the default-arg value in at serialisation time,
        # so each lambda captures its own i.
        with HybridPoolExecutor(max_workers=2) as ex:
            futs = [ex.submit(lambda n=i: n * 2) for i in range(4)]
            results = sorted(f.result(timeout=15) for f in futs)
        assert results == [0, 2, 4, 6]

    def test_exception_propagates(self):
        def bad(x):
            raise RuntimeError(f"fail-{x}")

        with HybridPoolExecutor(max_workers=2) as ex:
            with pytest.raises(RuntimeError, match="fail-z"):
                ex.submit(bad, "z").result(timeout=15)

    def test_submit_after_shutdown_raises(self):
        ex = HybridPoolExecutor(max_workers=1)
        ex.shutdown(wait=True)
        with pytest.raises(RuntimeError, match="shutdown"):
            ex.submit(lambda: 1)

    def test_context_manager_executes_and_returns(self):
        # Tasks run in a child process – test via return value, not side-effects.
        with HybridPoolExecutor(max_workers=1) as ex:
            result = ex.submit(lambda: 99).result(timeout=15)
        assert result == 99

    def test_shutdown_idempotent(self):
        ex = HybridPoolExecutor(max_workers=1)
        ex.shutdown(wait=True)
        ex.shutdown(wait=True)  # must not raise

    def test_max_workers_explicit_value(self):
        ex = HybridPoolExecutor(max_workers=3)
        assert ex._max_workers == 3
        ex.shutdown(wait=False, cancel_futures=True)

    def test_max_workers_zero_uses_cpu_formula(self):
        # 0 is falsy; formula falls back to max(1, cpu_count - 2)
        ex = HybridPoolExecutor(max_workers=0)
        expected = max(1, (os.cpu_count() or 1) - 2)
        assert ex._max_workers == expected
        ex.shutdown(wait=False, cancel_futures=True)

    def test_max_workers_none_uses_cpu_formula(self):
        ex = HybridPoolExecutor(max_workers=None)
        expected = max(1, (os.cpu_count() or 1) - 2)
        assert ex._max_workers == expected
        ex.shutdown(wait=False, cancel_futures=True)

    def test_thread_pool_double_worker_count(self):
        ex = HybridPoolExecutor(max_workers=3)
        assert ex._thread_pool._max_workers == 6
        ex.shutdown(wait=False, cancel_futures=True)

    def test_cancel_untracked_future_returns_true(self):
        # Future.cancel() on a never-started Future returns True.
        ex = HybridPoolExecutor(max_workers=1)
        orphan = Future()
        assert ex.cancel_future(orphan) is True
        ex.shutdown(wait=False, cancel_futures=True)

    def test_get_hybrid_executor_factory(self):
        ex = get_hybrid_executor(max_workers=2)
        assert isinstance(ex, HybridPoolExecutor)
        ex.shutdown(wait=False, cancel_futures=True)

    # ----------------------------------------------------------------------
    #  _coordinate_task tracking / cancellation polling
    # ----------------------------------------------------------------------

    def test_active_futures_tracked_then_cleaned_up(self):
        # _active_futures starts empty, so _coordinate_task never finds
        # a "thread"-matching entry: current_future stays None and the
        # fallback registers the process_future under its own key.
        # While the task is in flight, that key must be present ;
        # once it completes, the finally block must remove it.
        with HybridPoolExecutor(max_workers=1) as ex:
            thread_future = ex.submit(lambda: time.sleep(0.2) or "done")

            # Poll until the coordinator has registered the fallback entry
            deadline = time.monotonic() + 5
            seen_fallback = False
            while time.monotonic() < deadline:
                with ex._futures_lock:
                    if ex._active_futures:
                        seen_fallback = True
                        break
                time.sleep(0.01)

            assert seen_fallback
            assert thread_future.result(timeout=15) == "done"

            # After completion, tracking dict must be cleaned up.
            with ex._futures_lock:
                assert ex._active_futures == {}

    def test_fallback_tracking_when_current_future_not_found(self):
        # If, at the moment _coordinate_task runs, no entry in
        # _active_futures has "thread" matching the running thread
        # (current_future stays None), the coordinator falls back to
        # tracking the process_future directly under its own key,
        # and removes it again in the finally block.
        with HybridPoolExecutor(max_workers=1) as ex:
            with ex._futures_lock:
                ex._active_futures.clear()

            thread_future = ex.submit(lambda: "fallback-ok")
            assert thread_future.result(timeout=15) == "fallback-ok"

            with ex._futures_lock:
                assert ex._active_futures == {}

    def test_cancel_future_tracked_entry_via_real_submission(self):
        # _coordinate_task always registers under the fallback path,
        # keyed by process_future, with process_future also
        # stored in info["process_future"]. cancel_future on that key
        # calls future.cancel() (the process_future, as RUNNING ->
        # returns False) and info["process_future"].cancel() (same
        # object, also False) so the overall result is False, but the
        # tracked branch is exercised.
        with HybridPoolExecutor(max_workers=1) as ex:
            thread_future = ex.submit(lambda: "ok")

            deadline = time.monotonic() + 5
            tracked_key = None
            while time.monotonic() < deadline:
                with ex._futures_lock:
                    if ex._active_futures:
                        tracked_key = next(iter(ex._active_futures))
                        break
                time.sleep(0.01)
            assert tracked_key is not None

            ex.cancel_future(tracked_key)  # exercises tracked branch, 240-244

            assert thread_future.result(timeout=15) == "ok"

    def test_cancel_future_tracked_entry_cancels_both(self):
        # Synthetic entry (not yet submitted to either pool, both futures
        # still PENDING): cancel_future must cancel both the thread_future
        # and the tracked process_future and return True.
        with HybridPoolExecutor(max_workers=1) as ex:
            thread_future = Future()
            process_future = Future()
            with ex._futures_lock:
                ex._active_futures[thread_future] = {
                    "process_future": process_future,
                    "thread": threading.current_thread(),
                }

            result = ex.cancel_future(thread_future)

            assert result is True
            assert thread_future.cancelled()
            assert process_future.cancelled()

    def test_cancel_future_tracked_entry_no_process_future(self):
        # Tracked entry whose process_future is still None (not yet
        # attached by _coordinate_task): cancel_future must still cancel
        # the thread_future and return True without erroring on the
        # missing process_future.
        with HybridPoolExecutor(max_workers=1) as ex:
            thread_future = Future()
            with ex._futures_lock:
                ex._active_futures[thread_future] = {
                    "process_future": None,
                    "thread": threading.current_thread(),
                }

            result = ex.cancel_future(thread_future)

            assert result is True
            assert thread_future.cancelled()

    # ----------------------------------------------------------------------
    #  shutdown(cancel_futures=True) active-futures branch
    # ----------------------------------------------------------------------

    def test_shutdown_cancel_futures_cancels_active_entries(self):
        # Synthetic entry (both futures still PENDING): shutdown's
        # cancel_futures loop must cancel the thread_future and the
        # tracked process_future.
        ex = HybridPoolExecutor(max_workers=1)

        thread_future = Future()
        process_future = Future()
        with ex._futures_lock:
            ex._active_futures[thread_future] = {
                "process_future": process_future,
                "thread": threading.current_thread(),
            }

        ex.shutdown(wait=True, cancel_futures=True)

        assert thread_future.cancelled()
        assert process_future.cancelled()
