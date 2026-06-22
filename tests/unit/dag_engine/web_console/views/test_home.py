"""Unit tests for dag_engine.web_console.views.home."""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from fasthtml.common import Div

from retrain_pipelines.dag_engine.web_console.views.home import (
    AutoCompleteSelect,
    FilterElement,
    MultiStatesToggler,
    register,
)
import retrain_pipelines.dag_engine.web_console.views.home as home_module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_routes(prefix=""):
    """Call register() and return {path: handler} via a capturing rt decorator."""
    routes = {}

    def rt(path, methods=None):
        def decorator(fn):
            routes[path] = fn
            return fn

        return decorator

    register(MagicMock(), rt, prefix=prefix)
    return routes


def _make_request(*, headers=None, client_host="127.0.0.1", client_port=9999, path="/"):
    req = MagicMock()
    req.headers = headers or {}
    req.client.host = client_host
    req.client.port = client_port
    req.url.path = path
    return req


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# AutoCompleteSelect
# ---------------------------------------------------------------------------


class TestAutoCompleteSelect:
    def test_returns_div(self):
        result = AutoCompleteSelect(options_url="/opts", id="sel")
        assert result is not None

    def test_default_optional_params_do_not_raise(self):
        # placeholder, js_callback, style all default to ""
        AutoCompleteSelect(options_url="/opts", id="sel")

    def test_hyphenated_id_accepted(self):
        AutoCompleteSelect(options_url="/opts", id="my-sel")

    def test_input_and_dropdown_ids_in_output(self):
        result = AutoCompleteSelect(options_url="/opts", id="abc-def")
        html = result.__html__()
        assert "abc-def-input" in html
        assert "abc-def-dropdown" in html

    def test_options_url_present_in_script(self):
        result = AutoCompleteSelect(options_url="/my/special/url", id="s")
        assert "/my/special/url" in result.__html__()

    def test_js_callback_in_script(self):
        result = AutoCompleteSelect(options_url="/u", id="s", js_callback="doIt();")
        assert "doIt();" in result.__html__()

    def test_custom_style_propagated(self):
        result = AutoCompleteSelect(options_url="/u", id="s", style="color:red;")
        assert "color:red;" in result.__html__()


# ---------------------------------------------------------------------------
# FilterElement
# ---------------------------------------------------------------------------


class TestFilterElement:
    def test_returns_div(self):
        assert FilterElement("lbl", Div()) is not None

    def test_label_text_in_output(self):
        result = FilterElement("pipeline", Div())
        assert "pipeline" in result.__html__()

    def test_label_shadow_color_none(self):
        FilterElement("lbl", Div(), label_shadow_color=None)

    def test_label_shadow_color_in_style(self):
        result = FilterElement("lbl", Div(), label_shadow_color="red")
        assert "red" in result.__html__()

    def test_multiple_child_elements(self):
        FilterElement("lbl", Div(), Div(), Div())


# ---------------------------------------------------------------------------
# MultiStatesToggler
# ---------------------------------------------------------------------------


def _opts():
    return [
        Div("All", cls="all"),
        Div("Success", cls="success", **{"data-execs-status": "success"}),
        Div("Failure", cls="failure", **{"data-execs-status": "failure"}),
    ]


class TestMultiStatesToggler:
    def test_returns_div(self):
        assert MultiStatesToggler(options=_opts(), id="tog") is not None

    def test_bandit_toggle_label_class_added_to_all_options(self):
        result = MultiStatesToggler(options=_opts(), id="tog")
        # First child is the labels Div; each option inside must have the class
        assert "bandit-toggle-label" in result.__html__()

    def test_id_on_container(self):
        result = MultiStatesToggler(options=_opts(), id="my-tog")
        assert "my-tog" in result.__html__()

    def test_js_callback_in_script(self):
        result = MultiStatesToggler(options=_opts(), id="t", js_callback="cb();")
        assert "cb();" in result.__html__()

    def test_custom_style_in_output(self):
        result = MultiStatesToggler(options=_opts(), id="t", style="#t{color:red;}")
        assert "#t{color:red;}" in result.__html__()

    def test_opt_with_no_extra_attrs(self):
        # Branch: opt.attrs is empty ; class comes from opt.cls fallback
        plain = Div("plain")
        MultiStatesToggler(options=[plain], id="t")


# ---------------------------------------------------------------------------
# register() ; route registration
# ---------------------------------------------------------------------------


class TestRegisterRoutes:
    def test_all_routes_present_no_prefix(self):
        routes = _collect_routes()
        for path in (
            "/favicon.ico",
            "/{fname:path}.{ext:static}",
            "/distinct_pipeline_names",
            "/distinct_users",
            "/executions_events",
            "/load_executions",
            "/",
        ):
            assert path in routes, f"missing route: {path}"

    def test_all_routes_present_with_prefix(self):
        routes = _collect_routes(prefix="/pfx")
        for path in (
            "/pfx/distinct_pipeline_names",
            "/pfx/distinct_users",
            "/pfx/executions_events",
            "/pfx/load_executions",
            "/pfx/",
        ):
            assert path in routes, f"missing route: {path}"


# ---------------------------------------------------------------------------
# favicon
# ---------------------------------------------------------------------------


class TestFaviconRoute:
    def test_returns_file_response_for_ico(self):
        from fasthtml.common import FileResponse

        handler = _collect_routes()["/favicon.ico"]
        result = handler()
        assert isinstance(result, FileResponse)
        assert result.path.endswith("retrain-pipelines.ico")


# ---------------------------------------------------------------------------
# static file serving
# ---------------------------------------------------------------------------


class TestStaticFileRoute:
    def _handler(self):
        return _collect_routes()["/{fname:path}.{ext:static}"]

    def _stat(self, mtime):
        s = MagicMock()
        s.st_mtime = mtime
        s.st_mode = 0o100644  # regular file ; stat.S_ISREG() requires a real int
        return s

    def test_no_if_modified_since_returns_file_response(self):
        mtime = datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp()
        req = _make_request(headers={})
        mock_fr = MagicMock()
        with (
            patch.object(home_module.os, "stat", return_value=self._stat(mtime)),
            patch.object(home_module, "FileResponse", return_value=mock_fr) as fr_cls,
        ):
            result = _run(self._handler()(req, fname="app", ext="js"))
        fr_cls.assert_called_once()
        assert result is mock_fr

    def test_304_when_file_not_modified(self):
        mtime = datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp()
        req = _make_request(
            headers={"if-modified-since": "Mon, 01 Jan 2024 00:00:00 GMT"}
        )
        # FileResponse is never reached on 304 ; no need to patch it
        with patch.object(home_module.os, "stat", return_value=self._stat(mtime)):
            result = _run(self._handler()(req, fname="app", ext="js"))
        assert result.status_code == 304

    def test_200_when_file_is_newer(self):
        mtime = datetime(2025, 6, 1, tzinfo=timezone.utc).timestamp()
        req = _make_request(
            headers={"if-modified-since": "Mon, 01 Jan 2024 00:00:00 GMT"}
        )
        mock_fr = MagicMock()
        with (
            patch.object(home_module.os, "stat", return_value=self._stat(mtime)),
            patch.object(home_module, "FileResponse", return_value=mock_fr) as fr_cls,
        ):
            result = _run(self._handler()(req, fname="app", ext="js"))
        fr_cls.assert_called_once()
        assert result is mock_fr


# ---------------------------------------------------------------------------
# distinct_pipeline_names
# ---------------------------------------------------------------------------


class TestDistinctPipelineNamesRoute:
    def test_returns_json_with_names(self):
        from fasthtml.common import JSONResponse

        handler = _collect_routes()["/distinct_pipeline_names"]
        with patch.object(
            home_module, "get_pipeline_names", new=AsyncMock(return_value=["a", "b"])
        ):
            result = _run(handler())
        assert isinstance(result, JSONResponse)
        assert result.body == b'["a","b"]'


# ---------------------------------------------------------------------------
# distinct_users
# ---------------------------------------------------------------------------


class TestDistinctUsersRoute:
    def test_returns_json_with_users(self):
        from fasthtml.common import JSONResponse

        handler = _collect_routes()["/distinct_users"]
        with patch.object(
            home_module, "get_users", new=AsyncMock(return_value=["alice"])
        ):
            result = _run(handler())
        assert isinstance(result, JSONResponse)
        assert b"alice" in result.body


# ---------------------------------------------------------------------------
# executions_events (SSE)
# ---------------------------------------------------------------------------


class TestSseExecutionsEventsRoute:
    def test_returns_streaming_response(self):
        from fasthtml.common import StreamingResponse

        handler = _collect_routes()["/executions_events"]

        async def _gen(**_):
            return
            yield

        req = _make_request()
        with patch.object(home_module, "multiplexed_event_generator", side_effect=_gen):
            result = _run(handler(req))
        assert isinstance(result, StreamingResponse)
        assert result.media_type == "text/event-stream"


# ---------------------------------------------------------------------------
# load_executions
# ---------------------------------------------------------------------------


class TestLoadExecutionsRoute:
    def _handler(self):
        return _collect_routes()["/load_executions"]

    def _run_with_form(self, form_data, mock_return=None):
        req = _make_request()
        req.form = AsyncMock(return_value=form_data)
        mock_fn = AsyncMock(return_value=mock_return or [])
        with patch.object(home_module, "get_executions_ext", new=mock_fn):
            result = _run(self._handler()(req))
        return result, mock_fn

    def test_no_filters(self):
        result, _ = self._run_with_form({})
        assert result == []

    def test_valid_before_datetime_parsed_and_utc(self):
        _, mock_fn = self._run_with_form(
            {"before_datetime": "Mon Jan 01 2024 12:00:00 GMT+0000"}
        )
        _, kwargs = mock_fn.call_args
        assert kwargs["before_datetime"].tzinfo == timezone.utc

    def test_invalid_before_datetime_becomes_none(self):
        _, mock_fn = self._run_with_form({"before_datetime": "not-a-date"})
        _, kwargs = mock_fn.call_args
        assert kwargs["before_datetime"] is None

    def test_empty_strings_become_none(self):
        _, mock_fn = self._run_with_form(
            {"pipeline_name": "", "username": "", "execs_status": "", "n": ""}
        )
        _, kwargs = mock_fn.call_args
        assert kwargs["pipeline_name"] is None
        assert kwargs["username"] is None
        assert kwargs["execs_status"] is None
        assert kwargs["n"] is None

    def test_all_filters_forwarded(self):
        _, mock_fn = self._run_with_form(
            {
                "before_datetime": "Mon Jan 01 2024 12:00:00 GMT+0000",
                "pipeline_name": "my_pipe",
                "username": "alice",
                "execs_status": "success",
                "n": "10",
            }
        )
        mock_fn.assert_awaited_once()
        _, kwargs = mock_fn.call_args
        assert kwargs["pipeline_name"] == "my_pipe"
        assert kwargs["username"] == "alice"
        assert kwargs["execs_status"] == "success"
        assert kwargs["n"] == "10"
        assert kwargs["descending"] is True


# ---------------------------------------------------------------------------
# home page
# ---------------------------------------------------------------------------


class TestHomeRoute:
    def test_calls_page_layout_and_returns_result(self):
        handler = _collect_routes()["/"]
        mock_layout = MagicMock(return_value="<html/>")
        with patch.object(home_module, "page_layout", mock_layout):
            result = handler()
        mock_layout.assert_called_once()
        assert result == "<html/>"

    def test_home_with_prefix(self):
        handler = _collect_routes(prefix="/pfx")["/pfx/"]
        mock_layout = MagicMock(return_value="<html/>")
        with patch.object(home_module, "page_layout", mock_layout):
            result = handler()
        assert result == "<html/>"
