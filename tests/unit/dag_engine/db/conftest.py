"""
DAO fixtures scoped to the db integration-test subtree.

NOTE: ``isolated_dao``, ``isolated_async_dao``, ``seeded_async_dao``,
``taskgroup_seeded_async_dao``, ``_disable_dao_listeners_session``,
``isolated_db``, and ``metadatastore_url`` are all inherited from parent
conftests via normal pytest fixture discovery.

Provides:
  ``sync_dao``: session-scoped DAO parametrised on ``["sqlite", "postgresql"]``.
    Uses ``DAO(db_url=url)`` for both backends ; no treatment difference.
    SQLite: session-scoped file in ``tmp_path_factory``.
    PostgreSQL: URL from the PG* env vars set by ``postgres_session``.
    ``_disable_dao_listeners_session`` requested explicitly so no
    ``patch("requests.post")`` is needed in any test.

  ``async_dao``: function-scoped ``AsyncDAO`` via ``isolated_db``.
    Schema already exists (``isolated_db`` called ``create_all``).
    Uses ``Config.get_metadatastore_async_url()`` ; URL derived from the
    monkeypatched ``RP_METADATASTORE_URL``, no explicit async URL.

  ``silence_dao_logs``, ``LogCapture``, ``capture_log``: logging utilities.
"""

import io
import logging
import os

import pytest
import pytest_asyncio

from retrain_pipelines.dag_engine.config import Config
from retrain_pipelines.dag_engine.db.dao import AsyncDAO, DAO


def _pg_url() -> str:
    return (
        f"postgresql://{os.environ['PGUSER']}:{os.environ['PGPASSWORD']}"
        f"@{os.environ['PGHOST']}:{os.environ['PGPORT']}"
        f"/{os.environ['PGDATABASE']}"
    )


@pytest.fixture(scope="session", params=["sqlite", "postgresql"])
def sync_dao(
    request, postgres_session, _disable_dao_listeners_session, tmp_path_factory
):
    """Session-scoped DAO parametrised on SQLite and PostgreSQL.

    Same construction logic for both backends: ``DAO(db_url=url)``.
    ``DAO.__init__`` calls ``Base.metadata.create_all`` (idempotent).
    Data accumulates within the session (tests use ``>= N`` assertions).
    """
    if request.param == "sqlite":
        db_path = tmp_path_factory.mktemp("sync_db") / "sync.db"
        url = f"sqlite:///{db_path}"
    else:
        url = _pg_url()

    dao = DAO(db_url=url)
    yield dao
    dao.dispose()


@pytest_asyncio.fixture
async def async_dao(isolated_db):
    """Function-scoped AsyncDAO on a clean per-test database.

    Schema is present (``isolated_db`` called ``create_all``).
    Async URL derived by ``Config.get_metadatastore_async_url()`` from the
    monkeypatched ``RP_METADATASTORE_URL``.
    """
    dao = AsyncDAO(db_url=Config.get_metadatastore_async_url())
    yield dao
    await dao.engine.dispose()


@pytest.fixture(autouse=True)
def silence_dao_logs():
    """Suppress DAO logger globally.

    Tests can override with caplog but this doesn't work reliably
    with our ``RichLoggingController`` so, favor custom log capturing when needed.
    """
    logging.getLogger("retrain_pipelines.dag_engine.db.dao").setLevel(logging.CRITICAL)
    yield


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
