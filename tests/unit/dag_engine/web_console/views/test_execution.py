import os
import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock, mock_open
from uuid import uuid4

from fasthtml.common import (
    Response,
    JSONResponse,
    HTMLResponse,
    HTTPException,
    StreamingResponse,
)

from retrain_pipelines.dag_engine.web_console.views import execution as exec_module
from retrain_pipelines.dag_engine.web_console.views.execution import (
    get_execution_dag_elements_lists,
    get_execution_elements_lists,
    register,
)


@pytest.fixture
def mock_env():
    """Mock required environment variables."""
    with patch.dict(
        os.environ,
        {
            "RP_METADATASTORE_ASYNC_URL": "sqlite+aiosqlite:///:memory:",
            "RP_ARTIFACTS_STORE": "/tmp/artifacts",
        },
    ):
        yield


@pytest.fixture
def mock_request():
    """Create a mock FastHTML Request object."""
    req = MagicMock()
    req.query_params = MagicMock()
    req.query_params.get = MagicMock(return_value=None)
    req.client = MagicMock()
    req.client.host = "127.0.0.1"
    req.client.port = 8000
    req.url = MagicMock()
    req.url.path = "/test"
    return req


@pytest.fixture
def route_capturer():
    """Capture routes registered by the `register` function for isolated testing."""
    routes = {}

    def mock_rt(path, methods=None):
        def decorator(func):
            routes[path] = func
            return func

        return decorator

    return routes, mock_rt


# =============================================================================
# Tests for get_execution_dag_elements_lists
# =============================================================================
class TestGetExecutionDagElementsLists:
    @pytest.mark.asyncio
    async def test_success(self, mock_env):
        mock_dao = AsyncMock()
        mock_dao.get_execution_tasktypes_list.return_value = [
            MagicMock(uuid=uuid4(), taskgroup_uuid=uuid4())
        ]
        mock_dao.get_execution_taskgroups_list.return_value = [MagicMock(uuid=uuid4())]

        with patch.object(exec_module, "AsyncDAO", return_value=mock_dao):
            result = await get_execution_dag_elements_lists(1)

        assert result is not None
        assert len(result) == 2
        assert isinstance(result[0][0], dict)
        assert "uuid" in result[0][0]
        assert isinstance(result[0][0]["uuid"], str)

    @pytest.mark.asyncio
    async def test_none_tasktypes(self, mock_env):
        mock_dao = AsyncMock()
        mock_dao.get_execution_tasktypes_list.return_value = None
        mock_dao.get_execution_taskgroups_list.return_value = []

        with patch.object(exec_module, "AsyncDAO", return_value=mock_dao):
            result = await get_execution_dag_elements_lists(1)

        assert result == (None, None)

    @pytest.mark.asyncio
    async def test_empty_tasktypes(self, mock_env):
        mock_dao = AsyncMock()
        mock_dao.get_execution_tasktypes_list.return_value = []
        mock_dao.get_execution_taskgroups_list.return_value = []

        with patch.object(exec_module, "AsyncDAO", return_value=mock_dao):
            result = await get_execution_dag_elements_lists(1)

        assert result == (None, None)

    @pytest.mark.asyncio
    async def test_none_taskgroups(self, mock_env):
        mock_dao = AsyncMock()
        mock_dao.get_execution_tasktypes_list.return_value = [
            MagicMock(uuid=uuid4(), taskgroup_uuid=None)
        ]
        mock_dao.get_execution_taskgroups_list.return_value = None

        with patch.object(exec_module, "AsyncDAO", return_value=mock_dao):
            result = await get_execution_dag_elements_lists(1)

        assert result is not None
        assert len(result) == 2
        assert result[1] == []


# =============================================================================
# Tests for get_execution_elements_lists
# =============================================================================
class TestGetExecutionElementsLists:
    @pytest.mark.asyncio
    async def test_success(self, mock_env):
        mock_dao = AsyncMock()
        mock_dao.get_execution_tasks_list.return_value = [MagicMock()]
        mock_dao.get_execution_taskgroups_list.return_value = [MagicMock()]

        with patch.object(exec_module, "AsyncDAO", return_value=mock_dao):
            result = await get_execution_elements_lists(1)

        assert result is not None
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_none_tasks(self, mock_env):
        mock_dao = AsyncMock()
        mock_dao.get_execution_tasks_list.return_value = None
        mock_dao.get_execution_taskgroups_list.return_value = []

        with patch.object(exec_module, "AsyncDAO", return_value=mock_dao):
            result = await get_execution_elements_lists(1)

        assert result == (None, [])


# =============================================================================
# Tests for register routes
# =============================================================================
class TestRegisterRoutes:
    @pytest.mark.asyncio
    async def test_dag_rendering_invalid_id(self, route_capturer, mock_request):
        routes, mock_rt = route_capturer
        mock_request.query_params.get.return_value = "invalid"

        register(MagicMock(), mock_rt)
        response = await routes["/dag_rendering"](mock_request)
        assert type(response).__name__ == "FT"
        assert response.tag == "div"

    @pytest.mark.asyncio
    async def test_dag_rendering_none_tasktypes(
        self, route_capturer, mock_request, mock_env
    ):
        routes, mock_rt = route_capturer
        mock_request.query_params.get.return_value = "1"

        with patch.object(
            exec_module, "get_execution_dag_elements_lists", return_value=(None, None)
        ):
            register(MagicMock(), mock_rt)
            response = await routes["/dag_rendering"](mock_request)
            assert type(response).__name__ == "FT"
            assert response.tag == "div"

    @pytest.mark.asyncio
    async def test_dag_rendering_success(self, route_capturer, mock_request, mock_env):
        routes, mock_rt = route_capturer
        mock_request.query_params.get.return_value = "1"

        mock_tasktypes = [{"uuid": "123", "taskgroup_uuid": "456"}]
        mock_taskgroups = [{"uuid": "456"}]

        with (
            patch.object(
                exec_module,
                "get_execution_dag_elements_lists",
                return_value=(mock_tasktypes, mock_taskgroups),
            ),
            patch.object(exec_module, "Environment") as mock_env_cls,
            patch.object(exec_module, "FileSystemLoader"),
            patch.object(exec_module, "get_text_pixel_width"),
        ):
            mock_template = MagicMock()
            mock_template.render.return_value = "<svg>test</svg>"
            mock_env_instance = MagicMock()
            mock_env_instance.get_template.return_value = mock_template
            mock_env_cls.return_value = mock_env_instance

            register(MagicMock(), mock_rt)
            response = await routes["/dag_rendering"](mock_request)
            assert response == "<svg>test</svg>"

    @pytest.mark.asyncio
    async def test_exec_current_progress_invalid_id(self, route_capturer, mock_request):
        routes, mock_rt = route_capturer
        mock_request.query_params.get.return_value = "invalid"
        register(MagicMock(), mock_rt)
        response = await routes["/exec_current_progress"](mock_request)
        assert type(response).__name__ == "FT"
        assert response.tag == "div"

    @pytest.mark.asyncio
    async def test_exec_current_progress_none_tasks(
        self, route_capturer, mock_request, mock_env
    ):
        routes, mock_rt = route_capturer
        mock_request.query_params.get.return_value = "1"
        with patch.object(
            exec_module, "get_execution_elements_lists", return_value=(None, None)
        ):
            register(MagicMock(), mock_rt)
            response = await routes["/exec_current_progress"](mock_request)
            assert type(response).__name__ == "FT"
            assert response.tag == "div"

    @pytest.mark.asyncio
    async def test_exec_current_progress_success(
        self, route_capturer, mock_request, mock_env
    ):
        routes, mock_rt = route_capturer
        mock_request.query_params.get.return_value = "1"
        mock_tasks = [MagicMock()]
        mock_taskgroups = [MagicMock()]

        with (
            patch.object(
                exec_module,
                "get_execution_elements_lists",
                return_value=(mock_tasks, mock_taskgroups),
            ),
            patch.object(
                exec_module, "draw_chart", return_value="chart_script"
            ) as mock_draw,
        ):
            register(MagicMock(), mock_rt)
            response = await routes["/exec_current_progress"](mock_request)
            assert response == "chart_script"
            mock_draw.assert_called_once_with(1, mock_tasks, mock_taskgroups)

    @pytest.mark.asyncio
    async def test_execution_info_invalid_id(self, route_capturer, mock_request):
        routes, mock_rt = route_capturer
        mock_request.query_params.get.return_value = "invalid"
        register(MagicMock(), mock_rt)
        response = await routes["/execution_info"](mock_request)
        assert isinstance(response, Response)
        assert response.status_code == 500

    @pytest.mark.asyncio
    async def test_execution_info_success(self, route_capturer, mock_request, mock_env):
        routes, mock_rt = route_capturer
        mock_request.query_params.get.return_value = "1"
        mock_dao = AsyncMock()
        mock_dao.get_execution_info.return_value = {"name": "test"}

        with patch.object(exec_module, "AsyncDAO", return_value=mock_dao):
            register(MagicMock(), mock_rt)
            response = await routes["/execution_info"](mock_request)
            assert isinstance(response, JSONResponse)

    @pytest.mark.asyncio
    async def test_get_execution_number(self, route_capturer, mock_request, mock_env):
        routes, mock_rt = route_capturer
        mock_request.query_params.get.return_value = "1"
        mock_response = MagicMock()

        with patch.object(
            exec_module, "execution_number", return_value=mock_response
        ) as mock_exec_num:
            register(MagicMock(), mock_rt)
            response = await routes["/execution_number"](mock_request)
            assert response == mock_response
            mock_exec_num.assert_called_once_with("1")

    @pytest.mark.asyncio
    async def test_tasktype_docstring_success(
        self, route_capturer, mock_request, mock_env
    ):
        routes, mock_rt = route_capturer
        mock_request.query_params.get.return_value = "valid-uuid"
        mock_dao = AsyncMock()
        mock_dao.get_tasktype_docstring.return_value = "docstring"

        with patch.object(exec_module, "AsyncDAO", return_value=mock_dao):
            register(MagicMock(), mock_rt)
            response = await routes["/tasktype_docstring"](mock_request)
            assert isinstance(response, JSONResponse)

    @pytest.mark.asyncio
    async def test_pipeline_card_invalid_exec_id(self, route_capturer, mock_request):
        routes, mock_rt = route_capturer
        mock_request.query_params.get.return_value = "invalid"
        register(MagicMock(), mock_rt)
        with pytest.raises(HTTPException) as exc_info:
            await routes["/pipeline-card"](mock_request)
        assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_pipeline_card_not_found_attr(
        self, route_capturer, mock_request, mock_env
    ):
        routes, mock_rt = route_capturer
        mock_request.query_params.get.return_value = "1"
        mock_dao = AsyncMock()
        mock_dao.get_execution.return_value = None  # Triggers AttributeError on .name

        with patch.object(exec_module, "AsyncDAO", return_value=mock_dao):
            register(MagicMock(), mock_rt)
            with pytest.raises(HTTPException) as exc_info:
                await routes["/pipeline-card"](mock_request)
            assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_pipeline_card_file_not_found(
        self, route_capturer, mock_request, mock_env
    ):
        routes, mock_rt = route_capturer
        mock_request.query_params.get.return_value = "1"
        mock_dao = AsyncMock()
        mock_exec = MagicMock()
        mock_exec.name = "test_pipeline"
        mock_dao.get_execution.return_value = mock_exec

        with (
            patch.object(exec_module, "AsyncDAO", return_value=mock_dao),
            patch("os.path.exists", return_value=False),
            patch.object(exec_module.logging, "getLogger"),
        ):
            register(MagicMock(), mock_rt)
            with pytest.raises(HTTPException) as exc_info:
                await routes["/pipeline-card"](mock_request)
            assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_pipeline_card_success(self, route_capturer, mock_request, mock_env):
        routes, mock_rt = route_capturer
        mock_request.query_params.get.return_value = "1"
        mock_dao = AsyncMock()
        mock_exec = MagicMock()
        mock_exec.name = "test_pipeline"
        mock_dao.get_execution.return_value = mock_exec

        with (
            patch.object(exec_module, "AsyncDAO", return_value=mock_dao),
            patch("os.path.exists", return_value=True),
            patch("builtins.open", mock_open(read_data="<html>test</html>")),
        ):
            register(MagicMock(), mock_rt)
            response = await routes["/pipeline-card"](mock_request)
            assert isinstance(response, HTMLResponse)
            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_get_task_traces_invalid_task_id(self, route_capturer, mock_request):
        routes, mock_rt = route_capturer
        mock_request.query_params.get.return_value = "invalid"
        register(MagicMock(), mock_rt)
        response = await routes["/task_traces"](mock_request)
        assert isinstance(response, Response)
        assert response.status_code == 500

    @pytest.mark.asyncio
    async def test_get_task_traces_none(self, route_capturer, mock_request, mock_env):
        routes, mock_rt = route_capturer
        mock_request.query_params.get.return_value = "1"
        mock_dao = AsyncMock()
        mock_dao.get_task_traces.return_value = None

        with patch.object(exec_module, "AsyncDAO", return_value=mock_dao):
            register(MagicMock(), mock_rt)
            response = await routes["/task_traces"](mock_request)
            assert isinstance(response, JSONResponse)

    @pytest.mark.asyncio
    async def test_get_task_traces_success(
        self, route_capturer, mock_request, mock_env
    ):
        routes, mock_rt = route_capturer
        mock_request.query_params.get.return_value = "1"
        mock_dao = AsyncMock()
        mock_dt = MagicMock()
        mock_dt.timestamp.return_value = 1600000000.123
        mock_dao.get_task_traces.return_value = [
            {"timestamp": mock_dt, "other": "data"}
        ]

        with patch.object(exec_module, "AsyncDAO", return_value=mock_dao):
            register(MagicMock(), mock_rt)
            response = await routes["/task_traces"](mock_request)
            assert isinstance(response, JSONResponse)

            body = json.loads(response.body.decode("utf-8"))
            assert body[0]["timestamp"] == 1600000000123

    @pytest.mark.asyncio
    async def test_sse_execution_events(self, route_capturer, mock_request):
        routes, mock_rt = route_capturer
        mock_request.client.host = "127.0.0.1"
        mock_request.client.port = 8000
        mock_request.url.path = "/events"

        with patch.object(
            exec_module, "multiplexed_event_generator", return_value="gen"
        ) as mock_gen:
            register(MagicMock(), mock_rt)
            response = await routes["/execution_events"](mock_request)
            assert isinstance(response, StreamingResponse)
            mock_gen.assert_called_once()

    def test_home_invalid_id(self, route_capturer, mock_request):
        routes, mock_rt = route_capturer
        mock_request.query_params.get.return_value = "invalid"
        with patch.object(
            exec_module, "page_layout", return_value="layout"
        ) as mock_layout:
            register(MagicMock(), mock_rt)
            response = routes["/execution"](mock_request)
            mock_layout.assert_called_once_with(title="retrain-pipelines", content="")
            assert response == "layout"

    def test_home_success(self, route_capturer, mock_request):
        routes, mock_rt = route_capturer
        mock_request.query_params.get.return_value = "1"
        with patch.object(
            exec_module, "page_layout", return_value="layout"
        ) as mock_layout:
            register(MagicMock(), mock_rt)
            response = routes["/execution"](mock_request)
            assert response == "layout"
            assert mock_layout.call_args.kwargs["title"] == "retrain-pipelines"
