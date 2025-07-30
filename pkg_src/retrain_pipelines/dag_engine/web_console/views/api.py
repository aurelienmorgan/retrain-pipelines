
from fasthtml.common import Html, Head, Title, \
    Link, Body, Div, Script

from retrain_pipelines import __version__


_route_schemas = {}


def rt_api(rt, url, methods=None, schema=None, category=None):
    def decorator(func):
        if schema or category:
            _route_schemas[url] = {
                "schema": schema or {},
                "category": category
            }
        return rt(url, methods=methods)(func)
    return decorator


def register(app, rt, prefix=""):
    @rt(f"{prefix}/api/openapi.json")
    def openapi_spec():
        paths = {}

        for route in getattr(app, "routes", []):
            route_path = getattr(route, "rule", None) or getattr(route, "path", None)
            # Only include routes under the API prefix
            if route_path and route_path.startswith(f"{prefix}/api/v1/"):
                paths[route_path] = {}
                handler = getattr(route, "endpoint", None)

                methods = getattr(route, "methods", ["GET"])
                if "HEAD" in methods: methods.remove("HEAD") # we don't document those

                for method in methods:
                    method_lower = method.lower()

                    operation = {}
                    category = None

                    if route_path in _route_schemas:
                        entry = _route_schemas[route_path]
                        operation = entry.get("schema", {}).copy()
                        category = entry.get("category")

                    # # Add defaults if not in schema
                    # if "summary" not in operation:
                        # operation["summary"] = f"{method} {route_path}"
                    # if "responses" not in operation:
                        # operation["responses"] = {
                            # "200": {"description": "Successful response"},
                            # "422": {"description": "Validation error"}
                        # }

                    # Use docstring for description if not already in schema
                    if handler and handler.__doc__ and "description" not in operation:
                        operation["description"] = handler.__doc__.strip()

                    # Add tags to group endpoints in UI
                    if category:
                        operation["tags"] = [category]

                    paths[route_path][method_lower] = operation

        spec = {
            "openapi": "3.0.0",
            "info": {
                "title": "retrain-pipelines - WebConsole API",
                "version": __version__
            },
            "paths": paths
        }
        return spec


    @rt(f"{prefix}/api/docs")
    def swagger_ui():
        """return an HTML page embedding Swagger UI

        configured to fetch the OpenAPI spec from /api/openapi.json
        """
        return Html(
            Head(
                Title("Swagger UI"),
                Link(
                    rel="stylesheet",
                    href="https://cdn.jsdelivr.net/npm/swagger-ui-dist/swagger-ui.css"
                )
            ),
            Body(
                Div(id="swagger-ui"),
                Script(src="https://cdn.jsdelivr.net/npm/swagger-ui-dist/swagger-ui-bundle.js"),
                Script("""\
                  const ui = SwaggerUIBundle({
                    url: '/api/openapi.json',
                    dom_id: '#swagger-ui',
                  });
                """)
            )
        )

