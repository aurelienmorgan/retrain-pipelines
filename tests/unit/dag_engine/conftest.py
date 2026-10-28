"""
Shared fixtures for the dag_engine integration-test subtree.

Expose parametrized storage (filesystem and S3) and database (SQLite and
PostgreSQL) fixtures so they are discoverable by pytest for all tests in this
subtree as if they were defined directly in the herein top-level conftest.py.

Also exposes ``LogCapture`` / ``capture_log``, a generic log-capture utility
moved up from the db subtree so it is available to every test in the
dag_engine subtree (db, web_console, stores, etc.).
"""

import io
import logging
import pytest

from conftest_database import (  # noqa: F401
    _disable_dao_listeners_session,
    isolated_async_dao,
    isolated_dao,
    isolated_db,
    metadatastore_url,
    seeded_async_dao,
    taskgroup_seeded_async_dao,
)
from conftest_storage import (  # noqa: F401
    artifacts_store_root,
    assets_cache,
    metadata_root,
    runtime_env,
    web_server_logs_root,
)


# ==============================================================================
# Log capture fixture
# ==============================================================================


class LogCapture:
    """
    Context manager that captures log messages from a specified logger.

    ``__enter__`` returns ``self``. ``getvalue()`` is readable both inside
    the ``with`` block (live buffer) and after it exits (snapshot taken just
    before the buffer is closed). ``level`` controls the minimum severity
    captured (default: ``logging.DEBUG``).

    Usage:
        with LogCapture("retrain_pipelines.dag_engine.db.dao",
                        level=logging.ERROR) as capture:
            # code that logs
        assert "expected substring" in capture.getvalue()
    """

    def __init__(self, module_name, level=logging.DEBUG):
        self._module_name = module_name
        self._level = level
        self._logger = None
        self._original_level = None
        self._handler = None
        self._buffer = None
        self._snapshot = None

    def getvalue(self):
        """Return captured log text (live inside the block, snapshot after)."""
        if self._snapshot is not None:
            return self._snapshot
        return self._buffer.getvalue() if self._buffer else ""

    def __enter__(self):
        self._buffer = io.StringIO()
        self._logger = logging.getLogger(self._module_name)
        self._original_level = self._logger.level

        self._handler = logging.StreamHandler(self._buffer)
        self._handler.setFormatter(logging.Formatter("%(message)s"))
        self._handler.setLevel(self._level)

        self._logger.addHandler(self._handler)
        self._logger.setLevel(self._level)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._handler and self._logger:
            self._logger.removeHandler(self._handler)
        if self._logger and self._original_level is not None:
            self._logger.setLevel(self._original_level)
        self._snapshot = self._buffer.getvalue()
        self._buffer.close()


@pytest.fixture
def capture_log():
    """
    Fixture that returns a LogCapture context manager factory.

    Accepts an optional ``level`` argument (default: ``logging.DEBUG``)
    to set the minimum severity captured. ``getvalue()`` remains readable
    after the ``with`` block via an internal snapshot.

    Usage:
        def test_something(capture_log):
            with capture_log("retrain_pipelines.dag_engine.db.dao",
                             level=logging.ERROR) as captured:
                # code that logs
            assert "expected substring" in captured.getvalue()
    """
    return LogCapture
