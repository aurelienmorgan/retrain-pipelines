import asyncio
import json
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from retrain_pipelines.dag_engine.web_console.utils.execution import (
    events as execution_events,
)
from retrain_pipelines.dag_engine.web_console.api import tasks


class FakeRouter:
    """Captures handlers registered via `@rt(url, methods=...)`."""

    def __init__(self):
        self.handlers = {}

    def __call__(self, url, methods=None):
        def decorator(func):
            self.handlers[url] = func
            return func

        return decorator


class FakeRequest:
    """Minimal stand-in for a FastHTML/Starlette Request exposing `.json()`."""

    def __init__(self, payload):
        self._payload = payload

    async def json(self):
        return self._payload


@pytest.fixture(autouse=True)
def clean_subscribers():
    """Ensure subscriber lists don't leak state between tests."""
    execution_events.new_task_subscribers.clear()
    execution_events.task_end_subscribers.clear()
    yield
    execution_events.new_task_subscribers.clear()
    execution_events.task_end_subscribers.clear()


@pytest.fixture
def handlers():
    rt = FakeRouter()
    tasks.register(app=None, rt=rt, prefix="")
    return rt.handlers


def _valid_new_task_payload():
    return {
        "id": 1,
        "exec_id": 10,
        "tasktype_uuid": str(uuid4()),
        "name": "my_task",
        "is_parallel": False,
        "ui_css": {
            "color": "#ffffff",
            "background": "#000000",
            "border": "#111111",
        },
        "rank": [0],
        "taskgroup_uuid": None,
        "start_timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _valid_task_end_payload():
    payload = _valid_new_task_payload()
    payload["end_timestamp"] = datetime.now(timezone.utc).isoformat()
    payload["failed"] = False
    return payload


# ---------------------------------------------------------------------------
# register - route registration
# ---------------------------------------------------------------------------


def test_register_registers_both_routes():
    rt = FakeRouter()
    tasks.register(app=None, rt=rt, prefix="")

    assert "/api/v1/new_task_event" in rt.handlers
    assert "/api/v1/task_end_event" in rt.handlers


def test_register_respects_prefix():
    rt = FakeRouter()
    tasks.register(app=None, rt=rt, prefix="/console")

    assert "/console/api/v1/new_task_event" in rt.handlers
    assert "/console/api/v1/task_end_event" in rt.handlers


def test_register_sets_schema_and_category_in_route_schemas():
    from retrain_pipelines.dag_engine.web_console.api.open_api import _route_schemas

    rt = FakeRouter()
    tasks.register(app=None, rt=rt, prefix="")

    entry = _route_schemas["/api/v1/new_task_event"]
    assert entry["category"] == "Tasks"
    assert "requestBody" in entry["schema"]
    assert entry["schema"]["responses"]["200"] == {"description": "OK"}

    entry2 = _route_schemas["/api/v1/task_end_event"]
    assert entry2["category"] == "Tasks"
    assert "requestBody" in entry2["schema"]


# ---------------------------------------------------------------------------
# post_new_task_event
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_post_new_task_event_valid_payload_returns_200(handlers):
    handler = handlers["/api/v1/new_task_event"]
    request = FakeRequest(_valid_new_task_payload())

    response = await handler(request)

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_post_new_task_event_dispatches_to_subscribers(handlers):
    handler = handlers["/api/v1/new_task_event"]
    request = FakeRequest(_valid_new_task_payload())

    queue = asyncio.Queue()
    execution_events.new_task_subscribers.append((queue, {"ip": "127.0.0.1"}))

    await handler(request)

    assert queue.qsize() == 1
    dispatched = queue.get_nowait()
    assert dispatched["id"] == 1
    assert dispatched["name"] == "my_task"
    assert "_sa_instance_state" not in dispatched
    # underscore-prefixed attrs (e.g. _start_timestamp) get unprefixed
    assert "start_timestamp" in dispatched
    assert "_start_timestamp" not in dispatched


@pytest.mark.asyncio
async def test_post_new_task_event_does_not_dispatch_to_task_end_subscribers(handlers):
    handler = handlers["/api/v1/new_task_event"]
    request = FakeRequest(_valid_new_task_payload())

    new_task_queue = asyncio.Queue()
    task_end_queue = asyncio.Queue()
    execution_events.new_task_subscribers.append((new_task_queue, {"ip": "127.0.0.1"}))
    execution_events.task_end_subscribers.append((task_end_queue, {"ip": "127.0.0.1"}))

    await handler(request)

    assert new_task_queue.qsize() == 1
    assert task_end_queue.qsize() == 0


@pytest.mark.asyncio
async def test_post_new_task_event_unknown_field_returns_422(handlers):
    handler = handlers["/api/v1/new_task_event"]
    payload = _valid_new_task_payload()
    payload["totally_unexpected_field"] = "boom"
    request = FakeRequest(payload)

    response = await handler(request)

    assert response.status_code == 422
    body = response.body.decode("utf-8")
    assert "Invalid input" in body


@pytest.mark.asyncio
async def test_post_new_task_event_unknown_field_does_not_dispatch(handlers):
    handler = handlers["/api/v1/new_task_event"]
    payload = _valid_new_task_payload()
    payload["totally_unexpected_field"] = "boom"
    request = FakeRequest(payload)

    queue = asyncio.Queue()
    execution_events.new_task_subscribers.append((queue, {"ip": "127.0.0.1"}))

    await handler(request)

    assert queue.qsize() == 0


@pytest.mark.asyncio
async def test_post_new_task_event_no_subscribers(handlers):
    handler = handlers["/api/v1/new_task_event"]
    request = FakeRequest(_valid_new_task_payload())

    response = await handler(request)

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_post_new_task_event_payload_is_json_serializable_after_dispatch(
    handlers,
):
    handler = handlers["/api/v1/new_task_event"]
    request = FakeRequest(_valid_new_task_payload())

    queue = asyncio.Queue()
    execution_events.new_task_subscribers.append((queue, {"ip": "127.0.0.1"}))

    await handler(request)

    dispatched = queue.get_nowait()
    json.dumps(dispatched, default=str)


# ---------------------------------------------------------------------------
# post_task_end_event
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_post_task_end_event_valid_payload_returns_200(handlers):
    handler = handlers["/api/v1/task_end_event"]
    request = FakeRequest(_valid_task_end_payload())

    response = await handler(request)

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_post_task_end_event_dispatches_to_subscribers(handlers):
    handler = handlers["/api/v1/task_end_event"]
    request = FakeRequest(_valid_task_end_payload())

    queue = asyncio.Queue()
    execution_events.task_end_subscribers.append((queue, {"ip": "127.0.0.1"}))

    await handler(request)

    assert queue.qsize() == 1
    dispatched = queue.get_nowait()
    assert dispatched["id"] == 1
    assert dispatched["failed"] is False
    assert "_sa_instance_state" not in dispatched
    assert "end_timestamp" in dispatched
    assert "_end_timestamp" not in dispatched


@pytest.mark.asyncio
async def test_post_task_end_event_does_not_dispatch_to_new_task_subscribers(handlers):
    handler = handlers["/api/v1/task_end_event"]
    request = FakeRequest(_valid_task_end_payload())

    new_task_queue = asyncio.Queue()
    task_end_queue = asyncio.Queue()
    execution_events.new_task_subscribers.append((new_task_queue, {"ip": "127.0.0.1"}))
    execution_events.task_end_subscribers.append((task_end_queue, {"ip": "127.0.0.1"}))

    await handler(request)

    assert task_end_queue.qsize() == 1
    assert new_task_queue.qsize() == 0


@pytest.mark.asyncio
async def test_post_task_end_event_unknown_field_returns_422(handlers):
    handler = handlers["/api/v1/task_end_event"]
    payload = _valid_task_end_payload()
    payload["totally_unexpected_field"] = "boom"
    request = FakeRequest(payload)

    response = await handler(request)

    assert response.status_code == 422
    body = response.body.decode("utf-8")
    assert "Invalid input" in body


@pytest.mark.asyncio
async def test_post_task_end_event_unknown_field_does_not_dispatch(handlers):
    handler = handlers["/api/v1/task_end_event"]
    payload = _valid_task_end_payload()
    payload["totally_unexpected_field"] = "boom"
    request = FakeRequest(payload)

    queue = asyncio.Queue()
    execution_events.task_end_subscribers.append((queue, {"ip": "127.0.0.1"}))

    await handler(request)

    assert queue.qsize() == 0


@pytest.mark.asyncio
async def test_post_task_end_event_no_subscribers(handlers):
    handler = handlers["/api/v1/task_end_event"]
    request = FakeRequest(_valid_task_end_payload())

    response = await handler(request)

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_post_task_end_event_payload_is_json_serializable_after_dispatch(
    handlers,
):
    handler = handlers["/api/v1/task_end_event"]
    request = FakeRequest(_valid_task_end_payload())

    queue = asyncio.Queue()
    execution_events.task_end_subscribers.append((queue, {"ip": "127.0.0.1"}))

    await handler(request)

    dispatched = queue.get_nowait()
    json.dumps(dispatched, default=str)
