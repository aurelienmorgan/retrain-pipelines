import pytest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

from retrain_pipelines.dag_engine.web_console.utils.executions.executions import (
    get_users,
    get_pipeline_names,
    execution_to_html,
    get_executions_ext,
)
from retrain_pipelines.dag_engine.db.dao import AsyncDAO


def create_execution_ext(**kwargs):
    """Helper to create a mock execution object with default attributes."""
    defaults = {
        "id": "exec-1",
        "name": "test-pipeline",
        "start_timestamp": datetime(2023, 10, 1, 12, 0, 0, tzinfo=timezone.utc),
        "end_timestamp": datetime(2023, 10, 1, 13, 1, 1, 500000, tzinfo=timezone.utc),
        "username": "test-user",
        "success": True,
        "ui_css": {"background": "#ff0000", "color": "#ffffff", "border": "#000000"},
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


@pytest.mark.asyncio
async def test_get_users(monkeypatch):
    monkeypatch.setenv("RP_METADATASTORE_ASYNC_URL", "sqlite+aiosqlite:///:memory:")

    mock_get = AsyncMock(return_value=["user1", "user2"])
    # Monkeypatch the method on the imported class, avoiding module-level import mocking
    monkeypatch.setattr(AsyncDAO, "get_distinct_execution_usernames", mock_get)

    result = await get_users()

    assert result == ["user1", "user2"]
    mock_get.assert_called_once_with(sorted=True)


@pytest.mark.asyncio
async def test_get_pipeline_names(monkeypatch):
    monkeypatch.setenv("RP_METADATASTORE_ASYNC_URL", "sqlite+aiosqlite:///:memory:")

    mock_get = AsyncMock(return_value=["pipeline1", "pipeline2"])
    monkeypatch.setattr(AsyncDAO, "get_distinct_execution_names", mock_get)

    result = await get_pipeline_names()

    assert result == ["pipeline1", "pipeline2"]
    mock_get.assert_called_once_with(sorted=True)


def test_execution_to_html_success():
    exec_ext = create_execution_ext(
        id="exec-123",
        name="success-pipeline",
        success=True,
    )

    html = execution_to_html(exec_ext)

    assert isinstance(html, str)
    assert "success-pipeline" in html
    assert "success" in html
    assert "failure" not in html
    assert "1:01:01.500" in html
    assert "exec-123" in html
    assert "data-pipeline-name" in html
    assert "data-username" in html
    assert "data-start-timestamp" in html
    assert "data-success" in html
    # Check that colors are processed correctly (hex to rgba)
    assert "255,0,0" in html
    assert "255,255,255" in html
    assert "0,0,0" in html


def test_execution_to_html_failure():
    start = datetime(2023, 10, 1, 12, 0, 0, tzinfo=timezone.utc)
    end = datetime(2023, 10, 1, 12, 0, 5, 0, tzinfo=timezone.utc)

    exec_ext = create_execution_ext(
        id="exec-124",
        name="fail-pipeline",
        start_timestamp=start,
        end_timestamp=end,
        success=False,
        ui_css=None,  # Tests default ui_css fallback
    )

    html = execution_to_html(exec_ext)

    assert isinstance(html, str)
    assert "fail-pipeline" in html
    assert "failure" in html
    assert " success " not in html  # Check for class, not data-success attribute
    assert "0:00:05.000" in html
    # Verify default colors are used when ui_css is None (hex is converted to rgba)
    assert "77,0,102" in html
    assert "255,255,255" in html
    # Verify border is empty when ui_css is None
    assert "--border-normal:  ;" in html


def test_execution_to_html_running():
    start = datetime(2023, 10, 1, 12, 0, 0, tzinfo=timezone.utc)

    exec_ext = create_execution_ext(
        id="exec-125",
        name="running-pipeline",
        start_timestamp=start,
        end_timestamp=None,
        success=False,  # Should be ignored since end_timestamp is None
        ui_css={},  # Tests truthy but empty ui_css dict
    )

    html = execution_to_html(exec_ext)

    assert isinstance(html, str)
    assert "running-pipeline" in html
    # When end_timestamp is None, success/failure classes are not added
    assert " success " not in html
    assert " failure " not in html
    # Empty ui_css dict means UiCss(**{}) is called, falling back to defaults
    assert "77,0,102" in html
    assert "--border-normal:  ;" in html


def test_execution_to_html_no_success_attr():
    start = datetime(2023, 10, 1, 12, 0, 0, tzinfo=timezone.utc)

    class ExecutionNoSuccess:
        id = "exec-126"
        name = "no-success-pipeline"
        start_timestamp = start
        end_timestamp = None
        username = "test-user"
        ui_css = None

    html = execution_to_html(ExecutionNoSuccess())

    assert isinstance(html, str)
    assert "no-success-pipeline" in html
    # Verify it handles missing 'success' attribute gracefully
    assert "data-success" in html


@pytest.mark.asyncio
async def test_get_executions_ext(monkeypatch):
    monkeypatch.setenv("RP_METADATASTORE_ASYNC_URL", "sqlite+aiosqlite:///:memory:")

    start = datetime(2023, 10, 1, 12, 0, 0, tzinfo=timezone.utc)
    end = datetime(2023, 10, 1, 12, 0, 10, tzinfo=timezone.utc)

    mock_exec = create_execution_ext(
        id="exec-200",
        name="pipeline-200",
        start_timestamp=start,
        end_timestamp=end,
        username="user-200",
        success=True,
        ui_css=None,
    )

    mock_get = AsyncMock(return_value=[mock_exec])
    monkeypatch.setattr(AsyncDAO, "get_executions_ext", mock_get)

    before_dt = datetime(2023, 10, 2, tzinfo=timezone.utc)

    result = await get_executions_ext(
        pipeline_name="p1",
        username="u1",
        before_datetime=before_dt,
        execs_status="success",
        n=10,
        descending=True,
    )

    assert isinstance(result, list)
    assert len(result) == 1
    assert "pipeline-200" in result[0]

    mock_get.assert_called_once_with(
        pipeline_name="p1",
        username="u1",
        before_datetime=before_dt,
        execs_status="success",
        n=10,
        descending=True,
    )


@pytest.mark.asyncio
async def test_get_executions_ext_defaults(monkeypatch):
    monkeypatch.setenv("RP_METADATASTORE_ASYNC_URL", "sqlite+aiosqlite:///:memory:")

    mock_get = AsyncMock(return_value=[])
    monkeypatch.setattr(AsyncDAO, "get_executions_ext", mock_get)

    result = await get_executions_ext()

    assert result == []

    # Verify all default parameters are passed correctly
    mock_get.assert_called_once_with(
        pipeline_name=None,
        username=None,
        before_datetime=None,
        execs_status=None,
        n=None,
        descending=False,
    )
