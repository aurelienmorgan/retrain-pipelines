"""
DAO fixtures scoped to the db unit-test subtree.

NOTE: ``isolated_dao`` and ``disable_http_listener`` can be found in
the parent ``dag_engine/conftest.py``. They're shared with
runtime tests (they are available in this subtree via normal
pytest fixture inheritance).

Provides:
  ``sync_dao``: Session-scoped DAO backed by a shared in-memory SQLite.
    StaticPool is substituted for NullPool only during DAO construction so
    that the single in-memory database is visible to all threads. The patch
    is not held open beyond construction; doing so would corrupt NullPool
    isolation for any ``isolated_dao`` created in the same session.

  ``async_dao``: function-scoped AsyncDAO backed by a fresh in-memory
    aiosqlite database with all ORM tables created on startup.

  ``seeded_async_dao``: function-scoped fixture that seeds the
    ``async_dao`` database with a set of Executions, TaskTypes, and Tasks
    suitable for exercising all AsyncDAO query methods. All inserts are
    done via raw SQL so that no ORM event listeners fire during fixture
    setup: setting ``_end_timestamp`` through the ORM on an
    async-session-attached ``Execution`` fires ``after_end_timestamp_change``,
    which attempts a synchronous lazy-load of ``target.tasks`` outside a
    greenlet context, raising ``MissingGreenlet``.
    Yields ``(async_dao, (exec_id_with_tasks, tt_uuid))`` where
    ``exec_id_with_tasks`` is an Execution that has an associated
    TaskType (name="seeded_step") and one Task, and ``tt_uuid`` is
    that TaskType's UUID.

  ``taskgroup_seeded_async_dao``: function-scoped fixture that seeds the
    ``async_dao`` database with an Execution, a TaskGroup, and a nested
    child TaskGroup whose ``elements`` list references the parent uuid,
    suitable for exercising ``get_taskgroups_hierarchy`` and
    ``get_execution_taskgroups_list`` found-path branches.
    Yields ``(async_dao, (exec_id, parent_tg_uuid, child_tg_uuid))``.

NOTE: ``sync_dao`` must NOT wrap its yield inside the
NullPool=>StaticPool patch context manager. Doing so would keep the
patch active for the entire session, causing every subsequently-created
``isolated_dao`` engine to also use StaticPool ; a single shared
connection ; which defeats NullPool isolation and causes concurrent
threads to collide on the same session. The patch is therefore scoped
only to the DAO constructor call.
"""

import io
import logging
import os
from datetime import datetime, timedelta, timezone
from unittest.mock import patch
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.pool import StaticPool

from retrain_pipelines.dag_engine.db.dao import DAO
from retrain_pipelines.dag_engine.db.model import Base


_NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(scope="session")
def sync_dao():
    """Session-scoped DAO backed by a shared in-memory SQLite.

    StaticPool is substituted for NullPool only during DAO construction so
    that the single in-memory database is visible to all threads. The patch
    is not held open beyond construction; doing so would corrupt NullPool
    isolation for any ``isolated_dao`` created in the same session.
    """
    with patch("sqlalchemy.pool.NullPool", StaticPool), patch("requests.post"):
        dao = DAO(db_url=os.environ["RP_METADATASTORE_URL"])  # sqlite in-mem
    yield dao
    dao.dispose()


@pytest_asyncio.fixture
async def async_dao():
    """Function-scoped AsyncDAO on a fresh in-memory aiosqlite database.

    Tables are created via the ORM metadata before yielding so every test
    gets a clean schema with no pre-existing rows.
    The AsyncDAO instance
    is then wired to the same engine so all operations share the migrated
    schema.
    A second in-memory URL is passed to ``AsyncDAO.__init__`` to
    satisfy the constructor; that engine is immediately discarded and
    replaced with the pre-migrated one.
    """
    from retrain_pipelines.dag_engine.db.dao import AsyncDAO
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_async_engine(
        os.environ["RP_METADATASTORE_ASYNC_URL"],  # sqlite in-mem
        future=True,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    dao = AsyncDAO(db_url=os.environ["RP_METADATASTORE_ASYNC_URL"])  # sqlite in-mem
    await dao.engine.dispose()
    dao.engine = engine
    dao.session_factory = sessionmaker(
        bind=engine,
        expire_on_commit=False,
        class_=AsyncSession,
    )

    yield dao
    await engine.dispose()


@pytest_asyncio.fixture
async def seeded_async_dao(async_dao):
    """Function-scoped fixture: seeded AsyncDAO database.

    Seeds:
      - Two Executions named "cnt_pipe" (username="alice"): one completed
        successfully (no failed tasks), one completed with a failed Task.
      - One Execution with a TaskType (name="seeded_step") and one Task,
        used by task-list / tasktype / info / number / trace tests.

    Yields:
      (async_dao, (exec_id_with_tasks, tt_uuid))
    """
    tt_uuid_f = uuid4()
    tt_uuid = uuid4()
    _end = (_NOW + timedelta(hours=1)).isoformat()

    async with async_dao.engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO executions (name, username, start_timestamp, end_timestamp,"
                " metadata_root, artifacts_store_root)"
                "VALUES (:n, :u, :s, :e, '/tmp/meta', '/tmp/artifacts')"
            ),
            {"n": "cnt_pipe", "u": "alice", "s": _NOW.isoformat(), "e": _end},
        )

        await conn.execute(
            text(
                "INSERT INTO executions (name, username, start_timestamp, end_timestamp,"
                " metadata_root, artifacts_store_root) "
                "VALUES (:n, :u, :s, :e, '/tmp/meta', '/tmp/artifacts')"
            ),
            {
                "n": "cnt_pipe",
                "u": "alice",
                "s": (_NOW + timedelta(seconds=1)).isoformat(),
                "e": _end,
            },
        )
        row = await conn.execute(text("SELECT last_insert_rowid()"))
        exec_id_failed = row.scalar()

        await conn.execute(
            text(
                "INSERT INTO tasktypes"
                ' (uuid, exec_id, "order", name, is_parallel, children) '
                "VALUES (:uuid, :eid, 0, 'fail_step', 0, '[]')"
            ),
            {"uuid": tt_uuid_f.hex, "eid": exec_id_failed},
        )

        await conn.execute(
            text(
                "INSERT INTO tasks"
                " (tasktype_uuid, exec_id, start_timestamp, end_timestamp, failed) "
                "VALUES (:tu, :eid, :s, :e, 1)"
            ),
            {
                "tu": tt_uuid_f.hex,
                "eid": exec_id_failed,
                "s": (_NOW + timedelta(seconds=1)).isoformat(),
                "e": _end,
            },
        )

        await conn.execute(
            text(
                "INSERT INTO executions (name, username, start_timestamp,"
                " metadata_root, artifacts_store_root) "
                "VALUES (:n, :u, :s, '/tmp/meta', '/tmp/artifacts')"
            ),
            {
                "n": "cnt_pipe",
                "u": "alice",
                "s": (_NOW + timedelta(seconds=2)).isoformat(),
            },
        )
        row = await conn.execute(text("SELECT last_insert_rowid()"))
        exec_id_with_tasks = row.scalar()

        await conn.execute(
            text(
                "INSERT INTO tasktypes"
                ' (uuid, exec_id, "order", name, is_parallel, children) '
                "VALUES (:uuid, :eid, 0, 'seeded_step', 0, '[]')"
            ),
            {"uuid": tt_uuid.hex, "eid": exec_id_with_tasks},
        )

        await conn.execute(
            text(
                "INSERT INTO tasks (tasktype_uuid, exec_id, start_timestamp) "
                "VALUES (:tu, :eid, :s)"
            ),
            {
                "tu": tt_uuid.hex,
                "eid": exec_id_with_tasks,
                "s": (_NOW + timedelta(seconds=2)).isoformat(),
            },
        )

    yield async_dao, (exec_id_with_tasks, tt_uuid)


@pytest_asyncio.fixture
async def taskgroup_seeded_async_dao(async_dao):
    """Function-scoped fixture: AsyncDAO seeded with TaskGroups for hierarchy tests.

    Seeds:
      - One Execution.
      - One parent TaskGroup (order=0, elements=[]).
      - One child TaskGroup (order=1) whose ``elements`` list contains the
        parent's UUID (hex, no dashes), establishing the parent-chain used
        by the recursive CTE in ``get_taskgroups_hierarchy``.

    Yields:
      (async_dao, (exec_id, parent_tg_uuid, child_tg_uuid))
    """
    import json

    parent_uuid = uuid4()
    child_uuid = uuid4()

    async with async_dao.engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO executions (name, username, start_timestamp,"
                " metadata_root, artifacts_store_root) "
                "VALUES (:n, :u, :s, '/tmp/meta', '/tmp/artifacts')"
            ),
            {"n": "tg_pipe", "u": "bob", "s": _NOW.isoformat()},
        )
        row = await conn.execute(text("SELECT last_insert_rowid()"))
        exec_id = row.scalar()

        # parent's elements list references child so the recursive CTE walks down
        await conn.execute(
            text(
                "INSERT INTO taskgroups"
                ' (uuid, exec_id, "order", name, elements) '
                "VALUES (:uuid, :eid, 0, 'parent_grp', :elems)"
            ),
            {
                "uuid": parent_uuid.hex,
                "eid": exec_id,
                "elems": json.dumps([parent_uuid.hex]),
            },
        )

        await conn.execute(
            text(
                "INSERT INTO taskgroups"
                ' (uuid, exec_id, "order", name, elements) '
                "VALUES (:uuid, :eid, 1, 'child_grp', :elems)"
            ),
            {"uuid": child_uuid.hex, "eid": exec_id, "elems": json.dumps([])},
        )

    yield async_dao, (exec_id, parent_uuid, child_uuid)


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
