"""
Database fixtures for the dag_engine integration-test subtree.

Note:
All fixtures that request ``isolated_db`` as a parameter are defined here
(where ``isolated_db`` is also defined), so no file ever imports a name
and then shadows it as a function parameter in the same scope.

Provides:
  ``_disable_dao_listeners_session``: session-scoped, **not** autouse. Removes
    every DAO SQLAlchemy event listener once, for the whole session. Any
    fixture that touches the database requests this explicitly so that no
    HTTP side-effects (``requests.post``) can fire from ORM operations.
    Listener functions remain directly callable ; ``TestEventListeners`` tests
    that invoke them explicitly are unaffected.

  ``metadatastore_url``: function-scoped, parametrised on
    ``["sqlite", "postgresql"]``. Mirrors ``conftest_storage.artifacts_store_root``:
    monkeypatches ``RP_METADATASTORE_URL`` (sync URL only - Config derives the
    async counterpart automatically) and yields the sync URL.
    sqlite      : fresh file in ``tmp_path`` per test.
    postgresql  : URL built from the PG* env vars set by ``postgres_session``.

  ``isolated_db``: function-scoped, built on ``metadatastore_url`` and
    ``_disable_dao_listeners_session``. Calls ``Base.metadata.create_all`` then
    ``table.delete()`` on every table - the same two-step for both backends,
    no treatment difference.  The first call is idempotent (safe for PostgreSQL
    where alembic may already own the schema) ; the second guarantees a clean
    slate before every test.  Yields the sync URL.

  ``isolated_dao``: function-scoped DAO on a clean per-test database.

  ``isolated_async_dao``: function-scoped sync Engine for ORM seeding with
    ``RP_METADATASTORE_URL`` already wired; SDK code that internally creates
    ``AsyncDAO`` via ``Config.get_metadatastore_async_url()`` sees the same DB.

  ``seeded_async_dao``: function-scoped; seeds Executions, TaskTypes, Tasks.
    Shared across the whole dag_engine subtree.
    Yields ``(AsyncDAO, (exec_id_with_tasks, tt_uuid))``.

  ``taskgroup_seeded_async_dao``: function-scoped; seeds Execution + two
    TaskGroups for hierarchy tests.
    Yields ``(AsyncDAO, (exec_id, parent_tg_uuid, child_tg_uuid))``.
"""

import inspect
import os
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import create_engine, event

from retrain_pipelines.dag_engine.config import Config
from retrain_pipelines.dag_engine.db import dao as _dao_module
from retrain_pipelines.dag_engine.db.dao import AsyncDAO, DAO
from retrain_pipelines.dag_engine.db.model import Base
from retrain_pipelines.dag_engine.db.model import (
    Execution,
    Task,
    TaskContextAttr,
    TaskGroup,
    TaskPayloadAttr,
    TaskTrace,
    TaskType,
)

_NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Listener disabling
# ---------------------------------------------------------------------------


def _remove_all_dao_listeners():
    listener_funcs = [
        fn
        for _, fn in inspect.getmembers(_dao_module, inspect.isfunction)
        if fn.__module__ == _dao_module.__name__
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
    for t in targets:
        if hasattr(t, "__mapper__"):
            for attr_name in t.__mapper__.attrs.keys():
                attr = getattr(t, attr_name, None)
                if attr is not None and hasattr(attr, "dispatch"):
                    attrs.append(attr)

    evt_names = [
        "after_insert",
        "before_insert",
        "after_update",
        "before_update",
        "after_delete",
        "before_delete",
        "set",
    ]
    for target in targets + attrs:
        for evt_name in evt_names:
            for fn in listener_funcs:
                try:
                    if event.contains(target, evt_name, fn):
                        event.remove(target, evt_name, fn)
                except Exception:
                    pass


@pytest.fixture(scope="session")
def _disable_dao_listeners_session():
    """Disable all DAO SQLAlchemy event listeners for the test session.

    Not autouse: requested explicitly by ``isolated_db`` and ``sync_dao``.
    Listener functions remain directly callable for ``TestEventListeners``.
    """
    _remove_all_dao_listeners()
    yield


# ---------------------------------------------------------------------------
# metadatastore_url
# ---------------------------------------------------------------------------


def _pg_url() -> str:
    return (
        f"postgresql://{os.environ['PGUSER']}:{os.environ['PGPASSWORD']}"
        f"@{os.environ['PGHOST']}:{os.environ['PGPORT']}"
        f"/{os.environ['PGDATABASE']}"
    )


@pytest.fixture(params=["sqlite", "postgresql"])
def metadatastore_url(request, tmp_path, monkeypatch):
    """Parametrised fixture: SQLite (fresh file) or PostgreSQL sync URL.

    Mirrors ``conftest_storage.artifacts_store_root``: monkeypatches
    ``RP_METADATASTORE_URL`` (sync only) and yields the URL.
    ``Config.get_metadatastore_async_url()`` derives the async counterpart.
    """
    if request.param == "sqlite":
        url = f"sqlite:///{tmp_path / 'test.db'}"
    else:
        url = _pg_url()
    monkeypatch.setenv("RP_METADATASTORE_URL", url)
    yield url


# ---------------------------------------------------------------------------
# isolated_db
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_db(metadatastore_url, _disable_dao_listeners_session):
    """Function-scoped clean database state — same logic for both backends.

    1. ``Base.metadata.create_all``: idempotent; creates any missing tables.
    2. Row deletion on every table in reverse FK order: clean slate per test.
    """
    engine = create_engine(metadatastore_url)
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        for table in reversed(Base.metadata.sorted_tables):
            conn.execute(table.delete())
    engine.dispose()
    yield metadatastore_url


# ---------------------------------------------------------------------------
# isolated_dao / isolated_async_dao
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_dao(isolated_db):
    """Function-scoped DAO on a clean per-test database (SQLite or PostgreSQL)."""
    dao_instance = DAO(db_url=Config.get_metadatastore_url())
    yield dao_instance
    dao_instance.dispose()


@pytest.fixture
def isolated_async_dao(isolated_db):
    """Function-scoped sync Engine for ORM seeding with env vars wired to the test DB."""
    engine = create_engine(Config.get_metadatastore_url())
    yield engine
    engine.dispose()


# ---------------------------------------------------------------------------
# seeded_async_dao / taskgroup_seeded_async_dao
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def seeded_async_dao(isolated_db):
    """Function-scoped seeded AsyncDAO — shared across the dag_engine subtree.

    Seeds: two completed Executions (name="cnt_pipe", username="alice") — one
    with a failed Task, one without — and one in-progress Execution with a
    TaskType (name="seeded_step") and one Task.

    Yields ``(AsyncDAO, (exec_id_with_tasks, tt_uuid))``.
    """
    dao = AsyncDAO(db_url=Config.get_metadatastore_async_url())
    tt_uuid_f = uuid4()
    tt_uuid = uuid4()
    _end = _NOW + timedelta(hours=1)

    await dao._async_add_entity(
        Execution,
        name="cnt_pipe",
        username="alice",
        _start_timestamp=_NOW,
        _end_timestamp=_end,
        metadata_root="/tmp/meta",
        artifacts_store_root="/tmp/artifacts",
    )

    exec_id_failed = await dao._async_add_entity(
        Execution,
        name="cnt_pipe",
        username="alice",
        _start_timestamp=_NOW + timedelta(seconds=1),
        _end_timestamp=_end,
        metadata_root="/tmp/meta",
        artifacts_store_root="/tmp/artifacts",
    )
    async with dao._get_session() as session:
        session.add(
            TaskType(
                uuid=tt_uuid_f,
                exec_id=exec_id_failed,
                order=0,
                name="fail_step",
                is_parallel=False,
                children=[],
            )
        )
        await session.flush()
        await session.commit()
    await dao._async_add_entity(
        Task,
        tasktype_uuid=tt_uuid_f,
        exec_id=exec_id_failed,
        _start_timestamp=_NOW + timedelta(seconds=1),
        _end_timestamp=_end,
        failed=True,
    )

    exec_id_with_tasks = await dao._async_add_entity(
        Execution,
        name="cnt_pipe",
        username="alice",
        _start_timestamp=_NOW + timedelta(seconds=2),
        metadata_root="/tmp/meta",
        artifacts_store_root="/tmp/artifacts",
    )
    async with dao._get_session() as session:
        session.add(
            TaskType(
                uuid=tt_uuid,
                exec_id=exec_id_with_tasks,
                order=0,
                name="seeded_step",
                is_parallel=False,
                children=[],
            )
        )
        await session.flush()
        await session.commit()
    await dao._async_add_entity(
        Task,
        tasktype_uuid=tt_uuid,
        exec_id=exec_id_with_tasks,
        _start_timestamp=_NOW + timedelta(seconds=2),
    )

    yield dao, (exec_id_with_tasks, tt_uuid)
    await dao.engine.dispose()


@pytest_asyncio.fixture
async def taskgroup_seeded_async_dao(isolated_db):
    """Function-scoped seeded AsyncDAO for TaskGroup hierarchy tests.

    Seeds one Execution, one parent TaskGroup (elements=[str(parent_uuid)]),
    and one child TaskGroup (elements=[]).

    Yields ``(AsyncDAO, (exec_id, parent_tg_uuid, child_tg_uuid))``.
    """
    dao = AsyncDAO(db_url=Config.get_metadatastore_async_url())
    parent_uuid = uuid4()
    child_uuid = uuid4()

    exec_id = await dao._async_add_entity(
        Execution,
        name="tg_pipe",
        username="bob",
        _start_timestamp=_NOW,
        metadata_root="/tmp/meta",
        artifacts_store_root="/tmp/artifacts",
    )
    async with dao._get_session() as session:
        session.add(
            TaskGroup(
                uuid=parent_uuid,
                exec_id=exec_id,
                order=0,
                name="parent_grp",
                elements=[str(parent_uuid)],
            )
        )
        session.add(
            TaskGroup(
                uuid=child_uuid,
                exec_id=exec_id,
                order=1,
                name="child_grp",
                elements=[],
            )
        )
        await session.flush()
        await session.commit()

    yield dao, (exec_id, parent_uuid, child_uuid)
    await dao.engine.dispose()
