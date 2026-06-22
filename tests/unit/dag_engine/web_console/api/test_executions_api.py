import asyncio
import json
from datetime import datetime, timezone

import pytest

from retrain_pipelines.dag_engine.web_console.api import executions
from retrain_pipelines.dag_engine.web_console.utils.execution import (
    events as execution_events,
)
from retrain_pipelines.dag_engine.web_console.utils.executions import (
    events as executions_events,
)


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
    execution_events.new_exec_subscribers.clear()
    execution_events.exec_end_subscribers.clear()
    executions_events.new_exec_subscribers.clear()
    executions_events.exec_end_subscribers.clear()
    yield
    execution_events.new_exec_subscribers.clear()
    execution_events.exec_end_subscribers.clear()
    executions_events.new_exec_subscribers.clear()
    executions_events.exec_end_subscribers.clear()


@pytest.fixture
def handlers():
    rt = FakeRouter()
    executions.register(app=None, rt=rt, prefix="")
    return rt.handlers


def _valid_new_execution_payload():
    return {
        "id": 1,
        "name": "my_pipeline",
        "username": "alice",
        "start_timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _valid_execution_end_payload():
    payload = _valid_new_execution_payload()
    payload["end_timestamp"] = datetime.now(timezone.utc).isoformat()
    payload["success"] = True
    return payload


# ---------------------------------------------------------------------------
# register - route registration
# ---------------------------------------------------------------------------


def test_register_registers_both_routes():
    rt = FakeRouter()
    executions.register(app=None, rt=rt, prefix="")

    assert "/api/v1/new_execution_event" in rt.handlers
    assert "/api/v1/execution_end_event" in rt.handlers


def test_register_respects_prefix():
    rt = FakeRouter()
    executions.register(app=None, rt=rt, prefix="/console")

    assert "/console/api/v1/new_execution_event" in rt.handlers
    assert "/console/api/v1/execution_end_event" in rt.handlers


def test_register_sets_schema_and_category_in_route_schemas():
    from retrain_pipelines.dag_engine.web_console.api.open_api import _route_schemas

    rt = FakeRouter()
    executions.register(app=None, rt=rt, prefix="")

    entry = _route_schemas["/api/v1/new_execution_event"]
    assert entry["category"] == "Executions"
    assert "requestBody" in entry["schema"]
    assert entry["schema"]["responses"]["200"] == {"description": "OK"}

    entry2 = _route_schemas["/api/v1/execution_end_event"]
    assert entry2["category"] == "Executions"
    assert "requestBody" in entry2["schema"]


# ---------------------------------------------------------------------------
# post_new_execution_event
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_post_new_execution_event_valid_payload_returns_200(handlers):
    handler = handlers["/api/v1/new_execution_event"]
    request = FakeRequest(_valid_new_execution_payload())

    response = await handler(request)

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_post_new_execution_event_dispatches_to_both_subscriber_lists(handlers):
    handler = handlers["/api/v1/new_execution_event"]
    payload = _valid_new_execution_payload()
    request = FakeRequest(payload)

    executions_queue = asyncio.Queue()
    execution_queue = asyncio.Queue()
    executions_events.new_exec_subscribers.append(
        (executions_queue, {"ip": "127.0.0.1"})
    )
    execution_events.new_exec_subscribers.append((execution_queue, {"ip": "127.0.0.1"}))

    await handler(request)

    assert executions_queue.qsize() == 1
    assert execution_queue.qsize() == 1
    assert executions_queue.get_nowait() == payload
    assert execution_queue.get_nowait() == payload


@pytest.mark.asyncio
async def test_post_new_execution_event_does_not_dispatch_to_end_subscribers(handlers):
    handler = handlers["/api/v1/new_execution_event"]
    request = FakeRequest(_valid_new_execution_payload())

    new_queue = asyncio.Queue()
    end_queue = asyncio.Queue()
    executions_events.new_exec_subscribers.append((new_queue, {"ip": "127.0.0.1"}))
    executions_events.exec_end_subscribers.append((end_queue, {"ip": "127.0.0.1"}))

    await handler(request)

    assert new_queue.qsize() == 1
    assert end_queue.qsize() == 0


@pytest.mark.asyncio
async def test_post_new_execution_event_missing_id_returns_422(handlers):
    handler = handlers["/api/v1/new_execution_event"]
    payload = _valid_new_execution_payload()
    del payload["id"]
    request = FakeRequest(payload)

    response = await handler(request)

    assert response.status_code == 422
    body = response.body.decode("utf-8")
    assert "Invalid input" in body


@pytest.mark.asyncio
async def test_post_new_execution_event_invalid_timestamp_returns_422(handlers):
    handler = handlers["/api/v1/new_execution_event"]
    payload = _valid_new_execution_payload()
    payload["start_timestamp"] = "not-a-timestamp"
    request = FakeRequest(payload)

    response = await handler(request)

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_post_new_execution_event_invalid_payload_does_not_dispatch(handlers):
    handler = handlers["/api/v1/new_execution_event"]
    payload = _valid_new_execution_payload()
    del payload["id"]
    request = FakeRequest(payload)

    executions_queue = asyncio.Queue()
    execution_queue = asyncio.Queue()
    executions_events.new_exec_subscribers.append(
        (executions_queue, {"ip": "127.0.0.1"})
    )
    execution_events.new_exec_subscribers.append((execution_queue, {"ip": "127.0.0.1"}))

    await handler(request)

    assert executions_queue.qsize() == 0
    assert execution_queue.qsize() == 0


@pytest.mark.asyncio
async def test_post_new_execution_event_no_subscribers(handlers):
    handler = handlers["/api/v1/new_execution_event"]
    request = FakeRequest(_valid_new_execution_payload())

    response = await handler(request)

    assert response.status_code == 200


# ---------------------------------------------------------------------------
# post_execution_ended_event
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_post_execution_ended_event_valid_payload_returns_200(handlers):
    handler = handlers["/api/v1/execution_end_event"]
    request = FakeRequest(_valid_execution_end_payload())

    response = await handler(request)

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_post_execution_ended_event_dispatches_to_both_subscriber_lists(handlers):
    handler = handlers["/api/v1/execution_end_event"]
    payload = _valid_execution_end_payload()
    request = FakeRequest(payload)

    executions_queue = asyncio.Queue()
    execution_queue = asyncio.Queue()
    executions_events.exec_end_subscribers.append(
        (executions_queue, {"ip": "127.0.0.1"})
    )
    execution_events.exec_end_subscribers.append((execution_queue, {"ip": "127.0.0.1"}))

    await handler(request)

    assert executions_queue.qsize() == 1
    assert execution_queue.qsize() == 1
    assert executions_queue.get_nowait() == payload
    assert execution_queue.get_nowait() == payload


@pytest.mark.asyncio
async def test_post_execution_ended_event_does_not_dispatch_to_new_subscribers(
    handlers,
):
    handler = handlers["/api/v1/execution_end_event"]
    request = FakeRequest(_valid_execution_end_payload())

    new_queue = asyncio.Queue()
    end_queue = asyncio.Queue()
    executions_events.new_exec_subscribers.append((new_queue, {"ip": "127.0.0.1"}))
    executions_events.exec_end_subscribers.append((end_queue, {"ip": "127.0.0.1"}))

    await handler(request)

    assert end_queue.qsize() == 1
    assert new_queue.qsize() == 0


@pytest.mark.asyncio
async def test_post_execution_ended_event_invalid_timestamp_returns_422(handlers):
    handler = handlers["/api/v1/execution_end_event"]
    payload = _valid_execution_end_payload()
    payload["end_timestamp"] = "not-a-timestamp"
    request = FakeRequest(payload)

    response = await handler(request)

    assert response.status_code == 422
    body = response.body.decode("utf-8")
    assert "Invalid input" in body


@pytest.mark.asyncio
async def test_post_execution_ended_event_unknown_field_returns_422(handlers):
    handler = handlers["/api/v1/execution_end_event"]
    payload = _valid_execution_end_payload()
    payload["totally_unexpected_field"] = "boom"
    request = FakeRequest(payload)

    response = await handler(request)

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_post_execution_ended_event_invalid_payload_does_not_dispatch(handlers):
    handler = handlers["/api/v1/execution_end_event"]
    payload = _valid_execution_end_payload()
    payload["end_timestamp"] = "not-a-timestamp"
    request = FakeRequest(payload)

    executions_queue = asyncio.Queue()
    execution_queue = asyncio.Queue()
    executions_events.exec_end_subscribers.append(
        (executions_queue, {"ip": "127.0.0.1"})
    )
    execution_events.exec_end_subscribers.append((execution_queue, {"ip": "127.0.0.1"}))

    await handler(request)

    assert executions_queue.qsize() == 0
    assert execution_queue.qsize() == 0


@pytest.mark.asyncio
async def test_post_execution_ended_event_no_subscribers(handlers):
    handler = handlers["/api/v1/execution_end_event"]
    request = FakeRequest(_valid_execution_end_payload())

    response = await handler(request)

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_post_execution_ended_event_dispatched_payload_is_json_serializable(
    handlers,
):
    handler = handlers["/api/v1/execution_end_event"]
    payload = _valid_execution_end_payload()
    request = FakeRequest(payload)

    queue = asyncio.Queue()
    execution_events.exec_end_subscribers.append((queue, {"ip": "127.0.0.1"}))

    await handler(request)

    dispatched = queue.get_nowait()
    json.dumps(dispatched, default=str)
