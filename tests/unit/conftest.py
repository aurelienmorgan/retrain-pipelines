"""
Shared test configuration.

- Captures truly-original global state once at collection time.
- Provides an autouse fixture that restores that state around every test,
  preventing RichLoggingController patches from leaking between tests
  and causing pytest teardown recursion.
- Provides shared test utilities (e.g., _SuppressLogger) for all test modules.
"""

import builtins
import logging
import os
import sys

import pytest


os.environ.setdefault("RP_METADATASTORE_URL", "sqlite:///:memory:")
os.environ.setdefault("RP_METADATASTORE_ASYNC_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("RP_WEB_SERVER_URL", "http://localhost:0")

# --- capture originals *before* any test runs --------------------------------
_ORIG_PRINT = builtins.print
_ORIG_GETLOGGER = logging.getLogger
_ORIG_STDOUT_WRITE = sys.stdout.write
_ORIG_STDERR_WRITE = sys.stderr.write


@pytest.fixture(autouse=True)
def _restore_global_patches():
    """Restore every global patched by RichLoggingController before and after each test.

    RichLoggingController.activate() replaces builtins.print, sys.stdout.write,
    sys.stderr.write, and logging.getLogger with module-level wrappers.
    If a test fails mid-flight those replacements are left in place, which
    causes infinite recursion when pytest itself calls logging.getLogger()
    during teardown.
    The herein fixture guarantees a clean slate for every test.
    """
    # --- PRE: restore before the test in case the previous one left a mess ---
    builtins.print = _ORIG_PRINT
    logging.getLogger = _ORIG_GETLOGGER
    sys.stdout.write = _ORIG_STDOUT_WRITE
    sys.stderr.write = _ORIG_STDERR_WRITE

    import retrain_pipelines.dag_engine.rp_logging as _rplog

    _rplog._global_controller = None

    yield

    # --- POST: always restore after the test --------------------------------
    builtins.print = _ORIG_PRINT
    logging.getLogger = _ORIG_GETLOGGER
    sys.stdout.write = _ORIG_STDOUT_WRITE
    sys.stderr.write = _ORIG_STDERR_WRITE
    _rplog._global_controller = None


@pytest.fixture
def suppress_logger():
    class _SuppressLogger:
        def __init__(self, module_name):
            self._module_logger = logging.getLogger(module_name)

        def __enter__(self):
            self._original_level = self._module_logger.level
            self._module_logger.setLevel(logging.CRITICAL)
            self._handler = logging.NullHandler()
            self._module_logger.addHandler(self._handler)
            return self

        def __exit__(self, *_):
            self._module_logger.removeHandler(self._handler)
            self._module_logger.setLevel(self._original_level)

    return _SuppressLogger
