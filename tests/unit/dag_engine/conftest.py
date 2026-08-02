"""
Shared fixtures for the dag_engine unit-test subtree.

Provides:
  ``disable_http_listener``: Deregisters the HTTP notification listener
    to prevent it from racing with file-based SQLite or making real HTTP
    calls. Moved here from db/conftest.py so both db and runtime tests
    can recycle it.

  ``isolated_dao``: function-scoped DAO backed by a temporary file-based
    SQLite database (NullPool). Unlike the session-scoped ``sync_dao``
    (which uses StaticPool so a single in-memory connection is shared by
    all sessions), this fixture lets each SQLAlchemy session obtain its
    own NullPool connection.

    That isolation is required for tests that insert Tasks:
    ``after_insert_task_listener`` opens a second scoped_session on the
    same engine inside the outer session's active ``BEGIN IMMEDIATE``
    transaction. Even with NullPool, SQLite serializes the inner
    ``BEGIN`` against the outer write lock; the listener's
    ``session.close()`` then leaves the outer cursor broken, silently
    preventing ``commit()`` from persisting the row.

    The fixture therefore temporarily removes
    ``after_insert_task_listener`` from SQLAlchemy's event registry via
    ``event.remove`` / ``event.listen`` (patching the module attribute
    would not suffice ; SQLAlchemy holds its own reference to the
    original callable). The listener's HTTP side-effect is already
    suppressed per-test by ``patch("requests.post")``.

    It is shared with runtime tests.
    Additionally sets ``RP_METADATASTORE_URL`` and creates the
    schema so that runtime.py can connect to the same isolated DB.

  ``runtime_env``: Isolated environment variables (``RP_ARTIFACTS_STORE_ROOT``,
    ``RP_ASSETS_CACHE``) and resets the DAG execution context variable
    for runtime tests. NOT autouse; tests must explicitly request it.
    Relies on isolated_dao for DB setup if DB is needed.
"""

import pytest
from sqlalchemy import create_engine, event

from retrain_pipelines.dag_engine.core.core import _dag_execution_context_var
from retrain_pipelines.dag_engine.db.dao import DAO, after_insert_task_listener
from retrain_pipelines.dag_engine.db.model import Base, Task


@pytest.fixture
def disable_http_listener():
    """Deregister the HTTP notification listener to prevent it
    from racing with file-based SQLite or making real HTTP calls.
    """
    event.remove(Task, "after_insert", after_insert_task_listener)
    yield
    event.listen(Task, "after_insert", after_insert_task_listener)


@pytest.fixture
def isolated_dao(tmp_path, monkeypatch, disable_http_listener):
    """Function-scoped DAO on a fresh file-based SQLite (NullPool)."""
    db_path = tmp_path / "test.db"
    url = f"sqlite:///{db_path}"
    monkeypatch.setenv("RP_METADATASTORE_URL", url)

    engine = create_engine(url)
    Base.metadata.create_all(engine)

    dao = DAO(db_url=url)
    yield dao
    dao.dispose()
    engine.dispose()


@pytest.fixture
def runtime_env(tmp_path, monkeypatch):
    """Isolated environment variables for runtime tests.

    NOT autouse; tests must explicitly request it.
    Relies on isolated_dao for DB setup if DB is needed.
    """
    monkeypatch.setenv("RP_ARTIFACTS_STORE_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("RP_ASSETS_CACHE", str(tmp_path / "metadata"))

    _dag_execution_context_var.set(None)

    yield

    _dag_execution_context_var.set(None)
