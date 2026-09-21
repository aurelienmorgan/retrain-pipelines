"""
SDK specific fixtures for the unit-test subtree.

Provides:
  ``isolated_async_dao``: function-scoped fixture that creates a
    file-based SQLite database and sets both ``RP_METADATASTORE_URL``
    and ``RP_METADATASTORE_ASYNC_URL`` to point to it. Used by SDK
    tests where the SDK creates its own ``AsyncDAO`` instances
    internally via ``Config.get_metadatastore_async_url()``.
    Yields a sync engine for ORM seeding (DAO listeners are disabled
    via ``disable_all_dao_listeners``).
  ``query_counter``: function-scoped fixture that attaches a
    ``before_cursor_execute`` listener to the ``sqlalchemy.Engine``
    class so that every engine—including those created internally by
    ``AsyncDAO`` via ``create_async_engine``—is observed.  Yields a
    lightweight counter with a ``reset()`` method.  No mocking or
    patching of application code is required.
  ``make_row``: Helper to fabricate ORM-like namespace rows.
  ``write_disk_artifact``: Helper to write cloudpickle artifacts using
    project file_utils transparently across local and S3 backends.
"""

import cloudpickle
import pytest
from sqlalchemy import Engine, create_engine, event
from types import SimpleNamespace

from retrain_pipelines.dag_engine.db.model import Base
from retrain_pipelines.utils.file_utils import write_binary_file


@pytest.fixture
def isolated_async_dao(tmp_path, monkeypatch, disable_all_dao_listeners):
    """File-based SQLite with both sync and async env vars set.

    Used by SDK tests where the SDK creates its own ``AsyncDAO`` instances
    internally via ``Config.get_metadatastore_async_url()``. Yields a sync
    engine for ORM seeding. ``disable_all_dao_listeners`` dynamically disables
    all DAO event listeners, allowing direct ORM inserts without triggering
    HTTP requests.
    """
    db_path = tmp_path / "async_test.db"
    url = f"sqlite:///{db_path}"
    async_url = f"sqlite+aiosqlite:///{db_path}"

    monkeypatch.setenv("RP_METADATASTORE_URL", url)
    monkeypatch.setenv("RP_METADATASTORE_ASYNC_URL", async_url)

    engine = create_engine(url)
    Base.metadata.create_all(engine)

    yield engine

    engine.dispose()


@pytest.fixture
def query_counter():
    """Count SQL statements executed on any SQLAlchemy engine.

    Attaches a class-level ``before_cursor_execute`` listener on
    ``sqlalchemy.Engine`` so that every engine (including those created
    internally by ``AsyncDAO`` via ``create_async_engine``) is observed.
    No mocking or patching of application code is required ; the counter
    observes real database round-trips at the driver level.

    Call ``counter.reset()`` after seeding to zero the count before
    exercising the code under test.
    """

    class _Counter:
        def __init__(self):
            self.count = 0

        def reset(self):
            self.count = 0

    counter = _Counter()

    def _handler(*_args, **_kwargs):
        counter.count += 1

    event.listen(Engine, "before_cursor_execute", _handler)
    try:
        yield counter
    finally:
        event.remove(Engine, "before_cursor_execute", _handler)


# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def make_row():
    """Return a lightweight namespace mimicking a TaskContextAttr/PayloadAttr ORM row."""

    def _make_row(attr_name=None, disk_ref=None, inline_val=None, sha="hash"):
        return SimpleNamespace(
            attr_name=attr_name,
            disk_ref=disk_ref,
            inline_val=inline_val,
            sha=sha,
        )

    return _make_row


@pytest.fixture
def write_disk_artifact():
    """Write a cloudpickle artifact using project file_utils (handles local/S3)."""

    def _write(metadata_root, rel_path, value):
        data = cloudpickle.dumps(value)
        write_binary_file(metadata_root, [rel_path], data)

    return _write
