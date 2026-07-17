import asyncio
import json
import os
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from retrain_pipelines.dag_engine.web_console.utils import ClientInfo
from retrain_pipelines.dag_engine.web_console.utils.execution import events
from fasthtml.common import JSONResponse, Response


@pytest.fixture(autouse=True)
def reset_events_state():
    """Ensure a clean state for global variables before and after each test."""
    yield
    events._server_loop = None
    events.new_exec_subscribers.clear()
    events.exec_end_subscribers.clear()
    events.new_task_subscribers.clear()
    events.task_end_subscribers.clear()
    events.task_trace_subscribers.clear()


def test_notify_server_shutdown_active_loop():
    mock_loop = MagicMock()
    mock_loop.is_closed.return_value = False
    events._server_loop = mock_loop

    q1 = MagicMock()
    q2 = MagicMock()
    client_info = ClientInfo(ip="127.0.0.1", port=8000, url="/")
    events.new_exec_subscribers.append((q1, client_info))
    events.exec_end_subscribers.append((q2, client_info))

    events.notify_server_shutdown()

    assert mock_loop.call_soon_threadsafe.call_count == 2
    assert len(events.new_exec_subscribers) == 0
    assert len(events.exec_end_subscribers) == 0


def test_notify_server_shutdown_runtime_error():
    mock_loop = MagicMock()
    mock_loop.is_closed.return_value = False
    mock_loop.call_soon_threadsafe.side_effect = RuntimeError("Loop is closed")
    events._server_loop = mock_loop

    q1 = MagicMock()
    client_info = ClientInfo(ip="127.0.0.1", port=8000, url="/")
    events.new_exec_subscribers.append((q1, client_info))

    events.notify_server_shutdown()

    mock_loop.call_soon_threadsafe.assert_called_once()
    assert len(events.new_exec_subscribers) == 0


def test_notify_server_shutdown_closed_loop():
    mock_loop = MagicMock()
    mock_loop.is_closed.return_value = True
    events._server_loop = mock_loop

    client_info = ClientInfo(ip="127.0.0.1", port=8000, url="/")
    events.new_exec_subscribers.append((MagicMock(), client_info))

    events.notify_server_shutdown()

    mock_loop.call_soon_threadsafe.assert_not_called()
    assert len(events.new_exec_subscribers) == 0


def test_notify_server_shutdown_none_loop():
    events._server_loop = None
    client_info = ClientInfo(ip="127.0.0.1", port=8000, url="/")
    events.new_exec_subscribers.append((MagicMock(), client_info))

    events.notify_server_shutdown()

    assert len(events.new_exec_subscribers) == 0


def test_reset_for_restart():
    events._server_loop = MagicMock()
    client_info = ClientInfo(ip="127.0.0.1", port=8000, url="/")
    events.new_exec_subscribers.append((MagicMock(), client_info))
    events.exec_end_subscribers.append((MagicMock(), client_info))
    events.new_task_subscribers.append((MagicMock(), client_info))
    events.task_end_subscribers.append((MagicMock(), client_info))
    events.task_trace_subscribers.append((MagicMock(), client_info))

    events.reset_for_restart()

    assert events._server_loop is None
    assert len(events.new_exec_subscribers) == 0
    assert len(events.exec_end_subscribers) == 0
    assert len(events.new_task_subscribers) == 0
    assert len(events.task_end_subscribers) == 0
    assert len(events.task_trace_subscribers) == 0


@pytest.mark.asyncio
async def test_execution_number_success():
    mock_dao = AsyncMock()
    mock_dao.get_execution_number.return_value = {"name": "test", "count": 1}

    with (
        patch.dict(os.environ, {"RP_METADATASTORE_ASYNC_URL": "sqlite:///:memory:"}),
        patch.object(events, "AsyncDAO", return_value=mock_dao),
    ):
        response = await events.execution_number(1)

        assert isinstance(response, JSONResponse)
        assert json.loads(response.body) == {"name": "test", "count": 1}
        mock_dao.get_execution_number.assert_called_once_with(1)


@pytest.mark.asyncio
async def test_execution_number_failure():
    mock_dao = AsyncMock()
    mock_dao.get_execution_number.return_value = None

    with (
        patch.dict(os.environ, {"RP_METADATASTORE_ASYNC_URL": "sqlite:///:memory:"}),
        patch.object(events, "AsyncDAO", return_value=mock_dao),
    ):
        response = await events.execution_number(999)

        assert isinstance(response, Response)
        assert response.status_code == 500
        assert b"Invalid execution ID 999" in response.body


@pytest.mark.asyncio
async def test_taskgroups_hierarchy_success():
    mock_dao = AsyncMock()
    mock_dao.get_taskgroups_hierarchy.return_value = [{"uuid": "123", "name": "tg1"}]
    test_uuid = uuid4()

    with (
        patch.dict(os.environ, {"RP_METADATASTORE_ASYNC_URL": "sqlite:///:memory:"}),
        patch.object(events, "AsyncDAO", return_value=mock_dao),
    ):
        response = await events.taskgroups_hierarchy(test_uuid)

        assert isinstance(response, JSONResponse)
        assert json.loads(response.body) == [{"uuid": "123", "name": "tg1"}]
        mock_dao.get_taskgroups_hierarchy.assert_called_once_with(test_uuid)


@pytest.mark.asyncio
async def test_taskgroups_hierarchy_failure():
    mock_dao = AsyncMock()
    mock_dao.get_taskgroups_hierarchy.return_value = None
    test_uuid = uuid4()

    with (
        patch.dict(os.environ, {"RP_METADATASTORE_ASYNC_URL": "sqlite:///:memory:"}),
        patch.object(events, "AsyncDAO", return_value=mock_dao),
    ):
        response = await events.taskgroups_hierarchy(test_uuid)

        assert isinstance(response, Response)
        assert response.status_code == 500
        assert b"Invalid TaskType UUID" in response.body


@pytest.mark.asyncio
async def test_augment_new_task_with_parallel_and_taskgroup():
    task_ext_dict = {
        "is_parallel": True,
        "ui_css": {"color": "#fff"},
        "taskgroup_uuid": str(uuid4()),
    }

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.body = json.dumps(
        [{"uuid": str(uuid4()), "ui_css": {"background": "#000"}}]
    ).encode("utf-8")

    with (
        patch.object(events, "taskgroups_hierarchy", return_value=mock_response),
        patch.object(events, "CustomStyleDict") as mock_style,
        patch.object(events, "fill_defaults"),
    ):
        mock_style.return_value = {
            "color": "#fff",
            "background": "#000",
            "border": "#000",
            "labelUnderlay": "#4d0066",
        }

        await events.augment_new_task(task_ext_dict)

        assert "parent_ui_css" in task_ext_dict
        assert "parallel_lines" in task_ext_dict["parent_ui_css"]
        assert "parallel_line" in task_ext_dict["parent_ui_css"]
        assert "taskgroups_hierarchy" in task_ext_dict
        assert len(task_ext_dict["taskgroups_hierarchy"]) == 1
        assert task_ext_dict["ui_css"]["labelUnderlay"] == "#000"


@pytest.mark.asyncio
async def test_augment_new_task_taskgroup_failure():
    task_ext_dict = {
        "is_parallel": False,
        "ui_css": {"color": "#fff"},
        "taskgroup_uuid": str(uuid4()),
    }

    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.body = b"Not found"

    with (
        patch.object(events, "taskgroups_hierarchy", return_value=mock_response),
        patch.object(events, "CustomStyleDict") as mock_style,
        patch.object(events, "fill_defaults"),
        patch("logging.getLogger") as mock_logger,
    ):
        mock_style.return_value = {
            "color": "#fff",
            "background": "#000",
            "border": "#000",
            "labelUnderlay": "#4d0066",
        }
        mock_logger_instance = MagicMock()
        mock_logger.return_value = mock_logger_instance

        await events.augment_new_task(task_ext_dict)

        mock_logger_instance.warn.assert_called_once_with("Not found")
        assert task_ext_dict["taskgroups_hierarchy"] == []


@pytest.mark.asyncio
async def test_augment_new_task_no_taskgroup():
    task_ext_dict = {
        "is_parallel": False,
        "ui_css": {"color": "#fff"},
        "taskgroup_uuid": None,
    }

    with (
        patch.object(events, "CustomStyleDict") as mock_style,
        patch.object(events, "fill_defaults"),
    ):
        mock_style.return_value = {
            "color": "#fff",
            "background": "#000",
            "border": "#000",
            "labelUnderlay": "#4d0066",
        }

        await events.augment_new_task(task_ext_dict)

        assert "parent_ui_css" not in task_ext_dict
        assert task_ext_dict["taskgroups_hierarchy"] == []


@pytest.mark.asyncio
async def test_multiplexed_event_generator_execution_real():
    client_info = ClientInfo(ip="127.0.0.1", port=8000, url="/sse")

    with (
        patch.object(events, "execution_number") as mock_exec_num,
        patch.object(events, "augment_new_task", new_callable=AsyncMock),
        patch("json.dumps", return_value='{"data": 1}'),
    ):
        mock_exec_num.return_value = MagicMock(body=b'{"name":"test","count":1}')

        gen = events.multiplexed_event_generator(client_info)

        async def run_gen():
            async for _ in gen:
                pass

        task = asyncio.create_task(run_gen())
        await asyncio.sleep(0.05)

        q = None
        for queue, info in events.new_exec_subscribers:
            if info == client_info:
                q = queue
                break

        assert q is not None
        await q.put({"id": 1})

        await asyncio.sleep(0.05)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert len(events.new_exec_subscribers) == 0
        assert len(events.exec_end_subscribers) == 0
        assert len(events.new_task_subscribers) == 0
        assert len(events.task_end_subscribers) == 0
        assert len(events.task_trace_subscribers) == 0


@pytest.mark.asyncio
async def test_multiplexed_event_generator_new_task_real():
    client_info = ClientInfo(ip="127.0.0.1", port=8000, url="/sse")

    with (
        patch.object(events, "execution_number", new_callable=AsyncMock),
        patch.object(
            events, "augment_new_task", new_callable=AsyncMock
        ) as mock_augment,
        patch("json.dumps", return_value='{"data": 1}'),
    ):
        gen = events.multiplexed_event_generator(client_info)

        async def run_gen():
            async for _ in gen:
                pass

        task = asyncio.create_task(run_gen())
        await asyncio.sleep(0.05)

        # Retrieve the queue for the newTask subscriber
        q_new_task = None
        for queue, info in events.new_task_subscribers:
            if info == client_info:
                q_new_task = queue
                break
        assert q_new_task is not None

        # Also retrieve queues for taskEnd and taskTrace to allow manual removal later
        q_task_end = None
        for queue, info in events.task_end_subscribers:
            if info == client_info:
                q_task_end = queue
                break
        assert q_task_end is not None

        q_task_trace = None
        for queue, info in events.task_trace_subscribers:
            if info == client_info:
                q_task_trace = queue
                break
        assert q_task_trace is not None

        # Put an event into the newTask queue ; this will be processed and yield
        await q_new_task.put({"name": "task1"})

        # Manually remove the subscribers before cancellation so that the
        # finally block's remove() calls raise ValueError, exercising the
        # except Exception handlers for new_task, task_end, and task_trace.
        events.new_task_subscribers.remove((q_new_task, client_info))
        events.task_end_subscribers.remove((q_task_end, client_info))
        events.task_trace_subscribers.remove((q_task_trace, client_info))

        await asyncio.sleep(0.05)

        mock_augment.assert_called_once()

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # The generator's finally block will attempt to remove the subscribers again,
        # but they are already gone, so the except blocks are triggered.
        # The lists are now empty because the finally block also clears them
        # (the remove fails, but the except swallows it; the lists were already
        # cleared by our manual removal).
        assert len(events.new_task_subscribers) == 0
        assert len(events.task_end_subscribers) == 0
        assert len(events.task_trace_subscribers) == 0


@pytest.mark.asyncio
async def test_multiplexed_event_generator_task_trace_real():
    client_info = ClientInfo(ip="127.0.0.1", port=8000, url="/sse")

    with (
        patch.object(events, "execution_number", new_callable=AsyncMock),
        patch.object(events, "augment_new_task", new_callable=AsyncMock),
        patch("json.dumps", return_value='{"data": 1}'),
    ):
        gen = events.multiplexed_event_generator(client_info)

        async def run_gen():
            async for _ in gen:
                pass

        task = asyncio.create_task(run_gen())
        await asyncio.sleep(0.05)

        q = None
        for queue, info in events.task_trace_subscribers:
            if info == client_info:
                q = queue
                break

        assert q is not None
        await q.put({"trace": "data"})

        await asyncio.sleep(0.05)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert len(events.task_trace_subscribers) == 0
