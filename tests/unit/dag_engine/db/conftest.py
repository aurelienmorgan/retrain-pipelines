"""
DAO fixtures scoped to the db integration-test subtree.

NOTE: ``isolated_dao``, ``isolated_async_dao``, ``seeded_async_dao``,
``taskgroup_seeded_async_dao``, ``_disable_dao_listeners_session``,
``isolated_db``, and ``metadatastore_url`` are all inherited from parent
conftests via normal pytest fixture discovery.

``LogCapture`` / ``capture_log`` have been moved up to the dag_engine-level
conftest so they are available to every subtree (db, web_console, stores,
etc.), not just db/.

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

  ``silence_dao_logs``: autouse fixture that suppresses DAO logger.
"""

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
