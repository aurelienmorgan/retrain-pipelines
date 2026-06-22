from types import SimpleNamespace

import pytest

from retrain_pipelines.dag_engine.web_console.api.open_api import (
    _route_schemas,
    register,
    rt_api,
)


class FakeApp:
    """Minimal stand-in for the FastHTML app exposing a `routes` list."""

    def __init__(self, routes):
        self.routes = routes


class FakeRoute:
    """Minimal stand-in for a FastHTML route object."""

    def __init__(self, path=None, rule=None, methods=None, endpoint=None):
        if rule is not None:
            self.rule = rule
        if path is not None:
            self.path = path
        self.methods = methods if methods is not None else ["GET"]
        self.endpoint = endpoint


class FakeRouter:
    """Captures handlers registered via `@rt(url, methods=...)`."""

    def __init__(self):
        self.handlers = {}

    def __call__(self, url, methods=None):
        def decorator(func):
            self.handlers[url] = func
            return func

        return decorator


@pytest.fixture(autouse=True)
def clean_route_schemas():
    """Ensure `_route_schemas` module-level state doesn't leak between tests."""
    _route_schemas.clear()
    yield
    _route_schemas.clear()


# ---------------------------------------------------------------------------
# rt_api
# ---------------------------------------------------------------------------


def test_rt_api_registers_route_without_schema_or_category():
    rt = FakeRouter()

    @rt_api(rt, "/api/v1/foo", methods=["GET"])
    def handler():
        return "ok"

    assert "/api/v1/foo" not in _route_schemas
    assert rt.handlers["/api/v1/foo"] is handler
    assert handler() == "ok"


def test_rt_api_registers_schema_and_category():
    rt = FakeRouter()
    schema = {"summary": "Get foo"}

    @rt_api(rt, "/api/v1/foo", methods=["GET"], schema=schema, category="Foo")
    def handler():
        return "ok"

    assert _route_schemas["/api/v1/foo"] == {"schema": schema, "category": "Foo"}
    assert rt.handlers["/api/v1/foo"] is handler


def test_rt_api_registers_category_only_no_schema():
    rt = FakeRouter()

    @rt_api(rt, "/api/v1/bar", methods=["POST"], category="Bar")
    def handler():
        return "ok"

    assert _route_schemas["/api/v1/bar"] == {"schema": {}, "category": "Bar"}


def test_rt_api_registers_schema_only_no_category():
    rt = FakeRouter()
    schema = {"description": "desc"}

    @rt_api(rt, "/api/v1/baz", methods=["GET"], schema=schema)
    def handler():
        return "ok"

    assert _route_schemas["/api/v1/baz"] == {"schema": schema, "category": None}


# ---------------------------------------------------------------------------
# register / openapi_spec
# ---------------------------------------------------------------------------


def _get_openapi_spec(app, prefix=""):
    rt = FakeRouter()
    register(app, rt, prefix=prefix)
    return rt.handlers[f"{prefix}/api/openapi.json"]()


def test_openapi_spec_top_level_structure():
    app = FakeApp(routes=[])
    spec = _get_openapi_spec(app)

    assert spec["openapi"] == "3.0.0"
    assert spec["info"]["title"] == "retrain-pipelines - WebConsole API"
    assert "version" in spec["info"]
    assert spec["paths"] == {}


def test_openapi_spec_ignores_routes_outside_prefix():
    routes = [FakeRoute(path="/other/path", methods=["GET"])]
    app = FakeApp(routes=routes)
    spec = _get_openapi_spec(app)

    assert spec["paths"] == {}


def test_openapi_spec_includes_routes_under_prefix_using_path_attr():
    def handler():
        return "ok"

    routes = [FakeRoute(path="/api/v1/items", methods=["GET"], endpoint=handler)]
    app = FakeApp(routes=routes)
    spec = _get_openapi_spec(app)

    assert "/api/v1/items" in spec["paths"]
    assert "get" in spec["paths"]["/api/v1/items"]


def test_openapi_spec_uses_rule_attr_when_present():
    def handler():
        return "ok"

    routes = [
        FakeRoute(
            rule="/api/v1/items", path="/ignored", methods=["GET"], endpoint=handler
        )
    ]
    app = FakeApp(routes=routes)
    spec = _get_openapi_spec(app)

    assert "/api/v1/items" in spec["paths"]
    assert "/ignored" not in spec["paths"]


def test_openapi_spec_removes_head_method():
    def handler():
        return "ok"

    routes = [
        FakeRoute(path="/api/v1/items", methods=["GET", "HEAD"], endpoint=handler)
    ]
    app = FakeApp(routes=routes)
    spec = _get_openapi_spec(app)

    assert "get" in spec["paths"]["/api/v1/items"]
    assert "head" not in spec["paths"]["/api/v1/items"]


def test_openapi_spec_multiple_methods_for_same_route():
    def handler():
        return "ok"

    routes = [
        FakeRoute(path="/api/v1/items", methods=["GET", "POST"], endpoint=handler)
    ]
    app = FakeApp(routes=routes)
    spec = _get_openapi_spec(app)

    assert set(spec["paths"]["/api/v1/items"].keys()) == {"get", "post"}


def test_openapi_spec_uses_registered_schema_for_route():
    def handler():
        return "ok"

    schema = {"summary": "List items", "responses": {"200": {"description": "ok"}}}
    _route_schemas["/api/v1/items"] = {"schema": schema, "category": None}

    routes = [FakeRoute(path="/api/v1/items", methods=["GET"], endpoint=handler)]
    app = FakeApp(routes=routes)
    spec = _get_openapi_spec(app)

    operation = spec["paths"]["/api/v1/items"]["get"]
    assert operation["summary"] == "List items"
    assert operation["responses"] == {"200": {"description": "ok"}}


def test_openapi_spec_schema_copy_does_not_mutate_route_schemas():
    def handler():
        return "ok"

    schema = {"summary": "List items"}
    _route_schemas["/api/v1/items"] = {"schema": schema, "category": None}

    routes = [FakeRoute(path="/api/v1/items", methods=["GET"], endpoint=handler)]
    app = FakeApp(routes=routes)
    _get_openapi_spec(app)

    # Original schema dict in _route_schemas must remain untouched.
    assert _route_schemas["/api/v1/items"]["schema"] == {"summary": "List items"}
    assert "description" not in _route_schemas["/api/v1/items"]["schema"]
    assert "tags" not in _route_schemas["/api/v1/items"]["schema"]


def test_openapi_spec_adds_tags_when_category_present():
    def handler():
        return "ok"

    _route_schemas["/api/v1/items"] = {"schema": {}, "category": "Items"}

    routes = [FakeRoute(path="/api/v1/items", methods=["GET"], endpoint=handler)]
    app = FakeApp(routes=routes)
    spec = _get_openapi_spec(app)

    assert spec["paths"]["/api/v1/items"]["get"]["tags"] == ["Items"]


def test_openapi_spec_no_tags_when_category_absent():
    def handler():
        return "ok"

    routes = [FakeRoute(path="/api/v1/items", methods=["GET"], endpoint=handler)]
    app = FakeApp(routes=routes)
    spec = _get_openapi_spec(app)

    assert "tags" not in spec["paths"]["/api/v1/items"]["get"]


def test_openapi_spec_no_tags_when_category_is_none_in_schema_entry():
    def handler():
        return "ok"

    _route_schemas["/api/v1/items"] = {"schema": {}, "category": None}

    routes = [FakeRoute(path="/api/v1/items", methods=["GET"], endpoint=handler)]
    app = FakeApp(routes=routes)
    spec = _get_openapi_spec(app)

    assert "tags" not in spec["paths"]["/api/v1/items"]["get"]


def test_openapi_spec_uses_docstring_for_description_when_missing():
    def handler():
        """Handler docstring description."""
        return "ok"

    routes = [FakeRoute(path="/api/v1/items", methods=["GET"], endpoint=handler)]
    app = FakeApp(routes=routes)
    spec = _get_openapi_spec(app)

    assert (
        spec["paths"]["/api/v1/items"]["get"]["description"]
        == "Handler docstring description."
    )


def test_openapi_spec_does_not_overwrite_existing_description_with_docstring():
    def handler():
        """Handler docstring description."""
        return "ok"

    _route_schemas["/api/v1/items"] = {
        "schema": {"description": "Schema description"},
        "category": None,
    }

    routes = [FakeRoute(path="/api/v1/items", methods=["GET"], endpoint=handler)]
    app = FakeApp(routes=routes)
    spec = _get_openapi_spec(app)

    assert spec["paths"]["/api/v1/items"]["get"]["description"] == "Schema description"


def test_openapi_spec_no_description_when_handler_has_no_docstring():
    def handler():
        return "ok"

    routes = [FakeRoute(path="/api/v1/items", methods=["GET"], endpoint=handler)]
    app = FakeApp(routes=routes)
    spec = _get_openapi_spec(app)

    assert "description" not in spec["paths"]["/api/v1/items"]["get"]


def test_openapi_spec_no_description_when_handler_is_none():
    routes = [FakeRoute(path="/api/v1/items", methods=["GET"], endpoint=None)]
    app = FakeApp(routes=routes)
    spec = _get_openapi_spec(app)

    assert "description" not in spec["paths"]["/api/v1/items"]["get"]


def test_openapi_spec_defaults_methods_to_get_when_route_has_no_methods_attr():
    def handler():
        return "ok"

    # SimpleNamespace has no `methods` attribute => getattr default ["GET"] applies.
    route = SimpleNamespace(path="/api/v1/items", endpoint=handler)
    app = FakeApp(routes=[route])
    spec = _get_openapi_spec(app)

    assert "get" in spec["paths"]["/api/v1/items"]


def test_openapi_spec_respects_prefix():
    def handler():
        return "ok"

    routes = [
        FakeRoute(path="/console/api/v1/items", methods=["GET"], endpoint=handler)
    ]
    app = FakeApp(routes=routes)
    spec = _get_openapi_spec(app, prefix="/console")

    assert "/console/api/v1/items" in spec["paths"]


def test_openapi_spec_prefix_excludes_non_matching_routes():
    def handler():
        return "ok"

    routes = [FakeRoute(path="/api/v1/items", methods=["GET"], endpoint=handler)]
    app = FakeApp(routes=routes)
    spec = _get_openapi_spec(app, prefix="/console")

    assert spec["paths"] == {}


def test_openapi_spec_route_with_no_path_or_rule_is_skipped():
    routes = [SimpleNamespace(methods=["GET"], endpoint=None)]
    app = FakeApp(routes=routes)
    spec = _get_openapi_spec(app)

    assert spec["paths"] == {}


def test_openapi_spec_app_without_routes_attr():
    app = SimpleNamespace()  # no `routes` attribute at all
    spec = _get_openapi_spec(app)

    assert spec["paths"] == {}


# ---------------------------------------------------------------------------
# register / swagger_ui
# ---------------------------------------------------------------------------


def test_swagger_ui_returns_html_with_expected_elements():
    app = FakeApp(routes=[])
    rt = FakeRouter()
    register(app, rt, prefix="")

    html = rt.handlers["/api/docs"]()
    rendered = str(html)

    assert "Swagger UI" in rendered
    assert "swagger-ui-dist/swagger-ui.css" in rendered
    assert "swagger-ui-dist/swagger-ui-bundle.js" in rendered
    assert "/api/openapi.json" in rendered
    assert 'id="swagger-ui"' in rendered or "swagger-ui" in rendered


def test_swagger_ui_route_registered_under_prefix():
    app = FakeApp(routes=[])
    rt = FakeRouter()
    register(app, rt, prefix="/console")

    assert "/console/api/docs" in rt.handlers


def test_register_registers_openapi_and_docs_routes():
    app = FakeApp(routes=[])
    rt = FakeRouter()
    register(app, rt, prefix="")

    assert "/api/openapi.json" in rt.handlers
    assert "/api/docs" in rt.handlers
