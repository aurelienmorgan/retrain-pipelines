from unittest.mock import MagicMock, patch

import pytest

from retrain_pipelines.dag_engine import (
    alembic_capture,
    run_alembic_upgrade_once,
)
import retrain_pipelines.dag_engine as dag_engine_module


@pytest.fixture(autouse=True)
def _reset_alembic_upgraded():
    """Ensure the module-level upgrade flag doesn't leak across tests."""
    original = dag_engine_module._alembic_upgraded
    dag_engine_module._alembic_upgraded = False
    yield
    dag_engine_module._alembic_upgraded = original


@pytest.fixture
def mock_rich_controller():
    with patch.object(dag_engine_module, "RichLoggingController") as mock_cls:
        instance = MagicMock()
        mock_cls.return_value = instance
        yield instance


class TestAlembicCapture:
    def test_non_notebook_success(self, mock_rich_controller):
        """command.upgrade runs successfully outside a notebook."""
        with (
            patch.object(dag_engine_module, "in_notebook", return_value=False),
            patch.object(dag_engine_module.command, "upgrade") as mock_upgrade,
        ):
            cfg = MagicMock()
            with alembic_capture(cfg) as capture:
                pass

            mock_upgrade.assert_called_once_with(cfg, "head")
        mock_rich_controller.activate.assert_called_once()
        mock_rich_controller.deactivate.assert_called_once()
        assert capture.getvalue() == ""

    def test_non_notebook_exception(self, mock_rich_controller, capsys):
        """command.upgrade raises; exception is printed, not propagated."""
        with (
            patch.object(dag_engine_module, "in_notebook", return_value=False),
            patch.object(
                dag_engine_module.command, "upgrade", side_effect=RuntimeError("boom")
            ),
        ):
            cfg = MagicMock()
            with alembic_capture(cfg):
                pass

        captured = capsys.readouterr()
        assert "boom" in captured.out
        mock_rich_controller.deactivate.assert_called_once()

    def test_notebook_success(self, mock_rich_controller):
        """command.upgrade runs successfully inside a notebook."""
        mock_capture_output = MagicMock()
        mock_capture_output.return_value.__enter__ = MagicMock()
        mock_capture_output.return_value.__exit__ = MagicMock(return_value=False)

        with (
            patch.object(dag_engine_module, "in_notebook", return_value=True),
            patch.object(dag_engine_module.command, "upgrade") as mock_upgrade,
            patch("IPython.utils.io.capture_output", mock_capture_output),
        ):
            cfg = MagicMock()
            with alembic_capture(cfg):
                pass

            mock_upgrade.assert_called_once_with(cfg, "head")
        mock_rich_controller.deactivate.assert_called_once()


class TestRunAlembicUpgradeOnce:
    def test_upgrade_runs_and_sets_flag_non_notebook(self, mock_rich_controller):
        """First call performs upgrade, detects 'Running upgrade', prints output."""
        fake_alembic_capture = contextmanager_yielding("Running upgrade 1 -> 2\n")

        with (
            patch.object(dag_engine_module, "in_notebook", return_value=False),
            patch.object(dag_engine_module, "alembic_capture", fake_alembic_capture),
            patch.object(dag_engine_module, "Config") as mock_config,
            patch("builtins.print") as mock_print,
        ):
            run_alembic_upgrade_once()

        mock_config.return_value.set_main_option.assert_called_once_with(
            "sqlalchemy.url", dag_engine_module.os.environ["RP_METADATASTORE_URL"]
        )
        mock_print.assert_called_once_with("Running upgrade 1 -> 2\n")
        assert dag_engine_module._alembic_upgraded is True

    def test_no_upgrade_needed(self, mock_rich_controller):
        """Output without 'Running upgrade' triggers no print/logging."""
        fake_alembic_capture = contextmanager_yielding("Nothing to do\n")

        with (
            patch.object(dag_engine_module, "in_notebook", return_value=False),
            patch.object(dag_engine_module, "alembic_capture", fake_alembic_capture),
            patch.object(dag_engine_module, "Config"),
            patch("builtins.print") as mock_print,
        ):
            run_alembic_upgrade_once()

        mock_print.assert_not_called()
        assert dag_engine_module._alembic_upgraded is True

    def test_upgrade_runs_in_notebook(self, mock_rich_controller):
        """In-notebook branch logs raw output via RichLoggingController."""
        fake_alembic_capture = contextmanager_yielding("Running upgrade 1 -> 2\n")

        with (
            patch.object(dag_engine_module, "in_notebook", return_value=True),
            patch.object(dag_engine_module, "alembic_capture", fake_alembic_capture),
            patch.object(dag_engine_module, "Config"),
            patch.object(dag_engine_module.logging, "getLogger") as mock_get_logger,
        ):
            run_alembic_upgrade_once()

        mock_get_logger.return_value.info.assert_called_once_with(
            "Running upgrade 1 -> 2\n"
        )
        assert mock_rich_controller.activate.call_count == 1
        assert mock_rich_controller.deactivate.call_count == 1

    def test_idempotent_second_call_skips(self, mock_rich_controller):
        """Second call is a no-op because the global flag is already set."""
        dag_engine_module._alembic_upgraded = True

        with (
            patch.object(dag_engine_module, "alembic_capture") as mock_capture,
            patch.object(dag_engine_module, "Config") as mock_config,
        ):
            run_alembic_upgrade_once()

        mock_capture.assert_not_called()
        mock_config.assert_not_called()


def contextmanager_yielding(value):
    """Build a context-manager function that yields an object whose
    .getvalue() returns `value`, for use as a drop-in alembic_capture stub."""
    from contextlib import contextmanager

    @contextmanager
    def _cm(cfg):
        stub = MagicMock()
        stub.getvalue.return_value = value
        yield stub

    return _cm
