import asyncio
import pytest

import retrain_pipelines.dag_engine.web_console.utils.executions.events as events_module
from retrain_pipelines.dag_engine.web_console.utils.executions.events import (
    reset_for_restart,
    multiplexed_event_generator,
    _SHUTDOWN,
)


# ==============================================================================
# TESTS: reset_for_restart & notify_server_shutdown
# ==============================================================================


def test_reset_for_restart():
    """Test that reset_for_restart clears state and sets _server_loop to None."""
    # Set up stale state
    events_module._server_loop = asyncio.new_event_loop()
    events_module.new_exec_subscribers.append((asyncio.Queue(), {"ip": "1.1.1.1"}))
    events_module.exec_end_subscribers.append((asyncio.Queue(), {"ip": "1.1.1.1"}))

    # Execute function
    reset_for_restart()

    # Verify state is wiped
    assert events_module._server_loop is None
    assert len(events_module.new_exec_subscribers) == 0
    assert len(events_module.exec_end_subscribers) == 0

    # Clean up the dummy loop
    events_module._server_loop = None


def test_notify_server_shutdown_valid_loop():
    """Test notify_server_shutdown with a valid, open event loop."""
    reset_for_restart()
    loop = asyncio.new_event_loop()
    events_module._server_loop = loop
    q = asyncio.Queue()
    client_info = {"ip": "127.0.0.1", "port": 8080, "url": "/test"}
    events_module.new_exec_subscribers.append((q, client_info))
    events_module.exec_end_subscribers.append((q, client_info))

    # Execute function
    events_module.notify_server_shutdown()

    # Verify shutdown signal was scheduled and executed
    async def check():
        val1 = await q.get()
        val2 = await q.get()
        assert val1 is events_module._SHUTDOWN
        assert val2 is events_module._SHUTDOWN

    loop.run_until_complete(check())
    loop.close()

    # Verify lists are cleared at the end of the function
    assert len(events_module.new_exec_subscribers) == 0
    assert len(events_module.exec_end_subscribers) == 0


def test_notify_server_shutdown_runtime_error():
    """Test notify_server_shutdown catches RuntimeError during call_soon_threadsafe."""
    reset_for_restart()
    loop = asyncio.new_event_loop()
    events_module._server_loop = loop
    q = asyncio.Queue()
    client_info = {"ip": "127.0.0.1", "port": 8080, "url": "/test"}
    events_module.new_exec_subscribers.append((q, client_info))

    # Mock put_nowait to raise RuntimeError to cover the except block
    original_put = q.put_nowait

    def mock_put_nowait(item):
        raise RuntimeError("Simulated thread-safe put error")

    q.put_nowait = mock_put_nowait

    # Should not raise, just pass
    events_module.notify_server_shutdown()

    # Verify lists are still cleared despite the error
    assert len(events_module.new_exec_subscribers) == 0

    # Restore and cleanup
    q.put_nowait = original_put
    loop.close()


def test_notify_server_shutdown_no_loop():
    """Test notify_server_shutdown when _server_loop is None or closed."""
    reset_for_restart()  # Ensures _server_loop is None
    q = asyncio.Queue()
    client_info = {"ip": "127.0.0.1", "port": 8080, "url": "/test"}
    events_module.new_exec_subscribers.append((q, client_info))

    # Execute function
    events_module.notify_server_shutdown()

    # Verify lists are cleared via the safety net
    assert len(events_module.new_exec_subscribers) == 0
    assert len(events_module.exec_end_subscribers) == 0


# ==============================================================================
# TESTS: multiplexed_event_generator
# ==============================================================================


@pytest.mark.asyncio
async def test_multiplexed_event_generator_new_execution():
    """Test the generator yielding a newExecution event."""
    reset_for_restart()
    client_info = {"ip": "127.0.0.1", "port": 8080, "url": "/test"}
    gen = multiplexed_event_generator(client_info)

    async def feeder():
        # Yield control to let generator register queues
        await asyncio.sleep(0.01)
        new_q = next(
            q for q, info in events_module.new_exec_subscribers if info == client_info
        )
        # Payload must satisfy Execution.__init__ (requires id & start_timestamp)
        new_q.put_nowait(
            {"id": 1, "name": "test_exec", "start_timestamp": "2023-01-01T00:00:00"}
        )

    feeder_task = asyncio.create_task(feeder())

    # Collect the first yield
    result = await gen.__anext__()

    assert "event: newExecution\n" in result
    assert "data: " in result
    assert '"html":' in result

    # Clean shutdown
    new_q = next(
        q for q, info in events_module.new_exec_subscribers if info == client_info
    )
    end_q = next(
        q for q, info in events_module.exec_end_subscribers if info == client_info
    )
    new_q.put_nowait(_SHUTDOWN)
    end_q.put_nowait(_SHUTDOWN)

    with pytest.raises(StopAsyncIteration):
        await gen.__anext__()

    await feeder_task

    # Verify cleanup in finally block
    assert not any(
        info == client_info for q, info in events_module.new_exec_subscribers
    )
    assert not any(
        info == client_info for q, info in events_module.exec_end_subscribers
    )


@pytest.mark.asyncio
async def test_multiplexed_event_generator_execution_ended():
    """Test the generator yielding an executionEnded event."""
    reset_for_restart()
    client_info = {"ip": "127.0.0.1", "port": 8080, "url": "/test"}
    gen = multiplexed_event_generator(client_info)

    async def feeder():
        await asyncio.sleep(0.01)
        end_q = next(
            q for q, info in events_module.exec_end_subscribers if info == client_info
        )
        # Payload must satisfy ExecutionExt(**data) & .to_dict()
        end_q.put_nowait(
            {
                "id": 2,
                "start_timestamp": "2023-01-01T00:00:00",
                "end_timestamp": "2023-01-01T00:01:00",
            }
        )

    feeder_task = asyncio.create_task(feeder())

    # Collect the first yield
    result = await gen.__anext__()

    assert "event: executionEnded\n" in result
    assert "data: " in result
    assert '"html":' in result

    # Clean shutdown
    new_q = next(
        q for q, info in events_module.new_exec_subscribers if info == client_info
    )
    end_q = next(
        q for q, info in events_module.exec_end_subscribers if info == client_info
    )
    new_q.put_nowait(_SHUTDOWN)
    end_q.put_nowait(_SHUTDOWN)

    with pytest.raises(StopAsyncIteration):
        await gen.__anext__()

    await feeder_task

    # Verify cleanup in finally block
    assert not any(
        info == client_info for q, info in events_module.new_exec_subscribers
    )
    assert not any(
        info == client_info for q, info in events_module.exec_end_subscribers
    )


@pytest.mark.asyncio
async def test_multiplexed_event_generator_cancelled():
    """Test the generator handling asyncio.CancelledError and cleaning up."""
    reset_for_restart()
    client_info = {"ip": "127.0.0.1", "port": 8080, "url": "/test"}
    gen = multiplexed_event_generator(client_info)

    # Launch the generator and let it register queues & reach asyncio.wait
    gen_task = asyncio.create_task(gen.__anext__())

    # Yield control once. This is enough for the generator to append to globals
    # and suspend at `await asyncio.wait(...)`. Avoids busy-wait stalls.
    await asyncio.sleep(0.01)

    # Cancel the task to trigger the except asyncio.CancelledError block
    gen_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await gen_task

    # Verify cleanup in finally block occurred despite cancellation
    assert not any(
        info == client_info for q, info in events_module.new_exec_subscribers
    )
    assert not any(
        info == client_info for q, info in events_module.exec_end_subscribers
    )


@pytest.mark.asyncio
async def test_multiplexed_event_generator_finally_exception():
    """Test the generator's finally block gracefully handling removal exceptions."""
    reset_for_restart()
    client_info = {"ip": "127.0.0.1", "port": 8080, "url": "/test"}
    gen = multiplexed_event_generator(client_info)

    async def feeder_and_corrupt():
        await asyncio.sleep(0.01)
        # Corrupt the global lists to force .remove() to raise ValueError
        # This ensures the `except Exception: pass` blocks in the finally clause are covered
        events_module.new_exec_subscribers.clear()
        events_module.exec_end_subscribers.clear()

    feeder_task = asyncio.create_task(feeder_and_corrupt())
    anext_task = asyncio.create_task(gen.__anext__())

    await asyncio.sleep(0.02)  # Let it register and the feeder to corrupt the lists

    # Cancel to trigger the finally block
    anext_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await anext_task

    await feeder_task

    # The test passes if no unhandled exceptions bubbled up from the finally block
    assert len(events_module.new_exec_subscribers) == 0
    assert len(events_module.exec_end_subscribers) == 0
