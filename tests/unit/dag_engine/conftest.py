"""
Shared fixtures for the dag_engine unit-test subtree.

Provides:
  ``disable_all_dao_listeners``: Dynamically retrieves all functions from the
    ``dao`` module, checks if they are registered as event listeners on
    any ORM model or mapped attribute, and removes them. This prevents
    any DAO listener from racing with file-based SQLite or making real
    HTTP calls, without needing to hardcode listener names.

  ``isolated_dao``: function-scoped DAO backed by a temporary file-based
    SQLite database (NullPool). Unlike the session-scoped ``sync_dao``
    (which uses StaticPool so a single in-memory connection is shared by
    all sessions), this fixture lets each SQLAlchemy session obtain its
    own NullPool connection.

    It is shared with runtime tests.
    Additionally sets ``RP_METADATASTORE_URL`` and creates the
    schema so that runtime.py can connect to the same isolated DB.

  Also imports parametrized filesystem and S3 fixtures from
  ``conftest_storage`` so they are discoverable by pytest.
"""

import inspect
import pytest
from sqlalchemy import create_engine, event

from retrain_pipelines.dag_engine.db import dao
from retrain_pipelines.dag_engine.db.dao import DAO
from retrain_pipelines.dag_engine.db.model import (
    Base,
    Task,
    Execution,
    TaskType,
    TaskContextAttr,
    TaskPayloadAttr,
    TaskGroup,
    TaskTrace,
)

# Expose parametrized filesystem and S3 fixtures so they are discoverable
# by pytest as if they were defined directly in conftest.py.
from conftest_storage import (  # noqa: F401
    metadata_root,
    artifacts_store_root,
    assets_cache,
    web_server_logs_root,
    runtime_env,
)


def _retrieve_and_disable_all_listeners():
    """Retrieve all listeners from the dao module and disable them."""
    listener_funcs = [
        obj
        for name, obj in inspect.getmembers(dao, inspect.isfunction)
        if obj.__module__ == dao.__name__
    ]
    targets = [
        Execution,
        Task,
        TaskType,
        TaskContextAttr,
        TaskPayloadAttr,
        TaskGroup,
        TaskTrace,
    ]
    attrs = []
    for target in targets:
        if hasattr(target, "__mapper__"):
            for attr_name in target.__mapper__.attrs.keys():
                attr = getattr(target, attr_name, None)
                if attr is not None and hasattr(attr, "dispatch"):
                    attrs.append(attr)
    all_targets = targets + attrs
    events = [
        "after_insert",
        "before_insert",
        "after_update",
        "before_update",
        "after_delete",
        "before_delete",
        "set",
    ]
    disabled = []
    for target in all_targets:
        for evt in events:
            for fn in listener_funcs:
                try:
                    if event.contains(target, evt, fn):
                        event.remove(target, evt, fn)
                        disabled.append((target, evt, fn))
                except Exception:
                    pass
    return disabled


@pytest.fixture
def disable_all_dao_listeners():
    """Deregister all DAO listeners to prevent them from racing with
    file-based SQLite or making real HTTP calls.

    Retrieves all functions from the ``dao`` module and disables any that
    are registered as event listeners on the ORM models or their mapped
    attributes. This ensures no HTTP requests or side effects occur during
    tests.
    """
    disabled = _retrieve_and_disable_all_listeners()
    yield
    for target, evt, fn in disabled:
        if not event.contains(target, evt, fn):
            event.listen(target, evt, fn)


@pytest.fixture
def isolated_dao(tmp_path, monkeypatch, disable_all_dao_listeners):
    """Function-scoped DAO on a fresh file-based SQLite (NullPool)."""
    db_path = tmp_path / "test.db"
    url = f"sqlite:///{db_path}"
    monkeypatch.setenv("RP_METADATASTORE_URL", url)

    engine = create_engine(url)
    Base.metadata.create_all(engine)

    dao_instance = DAO(db_url=url)
    yield dao_instance
    dao_instance.dispose()
    engine.dispose()
