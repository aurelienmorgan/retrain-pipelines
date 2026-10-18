"""
SDK-specific fixtures for the integration-test subtree.

NOTE: ``isolated_async_dao`` from the parent ``dag_engine/conftest.py``
is available to all dag_engine subtests (including this sdk subtree)
via normal pytest fixture inheritance.

Provides:
  ``query_counter``: function-scoped; attaches a ``before_cursor_execute``
    listener to ``sqlalchemy.Engine`` so every engine - including those
    created internally by ``AsyncDAO`` - is observed.  Yields a lightweight
    counter with a ``reset()`` method.
  ``make_row``: helper to fabricate ORM-like namespace rows.
  ``write_disk_artifact``: helper to write cloudpickle artifacts via
    project file_utils, transparently across local and S3 backends.
"""

import cloudpickle
import pytest
from sqlalchemy import Engine, event
from types import SimpleNamespace

from retrain_pipelines.utils.file_utils import write_binary_file


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
