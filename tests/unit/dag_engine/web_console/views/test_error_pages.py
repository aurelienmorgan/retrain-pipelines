"""
Unit tests for error_pages.error_page.
"""

from fasthtml.common import HTTPException
import pytest

from retrain_pipelines.dag_engine.web_console.views.error_pages import error_page


class TestErrorPageNoException:
    """Branch: exc is None => default detail 'An error occurred'."""

    def test_uses_default_detail(self) -> None:
        response = error_page(404)
        assert response.status_code == 404
        body = response.body.decode()
        assert "404" in body
        assert "An error occurred" in body


class TestErrorPageWithHTTPException:
    """Branch: exc is HTTPException => use exc.detail."""

    def test_uses_http_exception_detail(self) -> None:
        detail = "Invalid input"
        exc = HTTPException(status_code=400, detail=detail)
        response = error_page(400, exc=exc)
        assert response.status_code == 400
        body = response.body.decode()
        assert "400" in body
        assert detail in body


class TestErrorPageWithDetailAttr:
    """
    Branch: exc is not HTTPException but has 'detail' attribute.
    This covers custom exceptions that mimic HTTPException's shape.
    """

    def test_uses_custom_detail_attr(self) -> None:
        class CustomError(Exception):
            def __init__(self, detail: str) -> None:
                self.detail = detail

        detail = "Database timeout"
        exc = CustomError(detail)
        response = error_page(500, exc=exc)
        assert response.status_code == 500
        body = response.body.decode()
        assert "500" in body
        assert detail in body


class TestErrorPageWithGenericException:
    """
    Branch: exc is a plain Exception with a non-empty string representation.
    Falls back to str(exc).
    """

    def test_uses_str_exception(self) -> None:
        detail = "Access denied"
        exc = Exception(detail)
        response = error_page(403, exc=exc)
        assert response.status_code == 403
        body = response.body.decode()
        assert "403" in body
        assert detail in body


class TestErrorPageWithEmptyStrException:
    """
    Branch: str(exc) is empty => fallback to exception class name.
    This ensures we never show an empty error message.
    """

    def test_fallback_to_class_name_when_str_empty(self) -> None:
        class EmptyStrException(Exception):
            def __str__(self) -> str:
                return ""

        exc = EmptyStrException()
        response = error_page(418, exc=exc)
        assert response.status_code == 418
        body = response.body.decode()
        assert "418" in body
        # The class name is used instead of an empty string.
        assert "EmptyStrException" in body


class TestErrorPageContentAndStatus:
    """
    Verifies that the rendered HTML contains the expected CSS classes
    and inline styles, regardless of the exception type.
    """

    def test_contains_required_css_and_styles(self) -> None:
        response = error_page(500)
        body = response.body.decode()

        # The body class is set via body_cls.
        assert 'class="body-error-page"' in body

        # The gradient style is present (from the inline style block).
        assert "background: linear-gradient" in body

        # The status code appears as visible text.
        assert "500" in body


class TestErrorPageStatusCodes:
    """
    Parameterized test for various HTTP status codes.
    Ensures the status code is correctly passed to the response.
    """

    @pytest.mark.parametrize("code", [400, 401, 403, 404, 500, 502, 503])
    def test_response_status_matches_input(self, code: int) -> None:
        response = error_page(code)
        assert response.status_code == code
        body = response.body.decode()
        assert str(code) in body
