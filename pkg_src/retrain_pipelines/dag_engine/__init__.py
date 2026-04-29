
import os
import sys
import logging

from contextlib import contextmanager, \
    redirect_stdout
from io import StringIO
from IPython import get_ipython

from alembic import command
from alembic.config import Config

from .rp_logging import RichLoggingController

from ..utils import in_notebook


_alembic_upgraded = False


@contextmanager
def alembic_capture(alembic_cfg):
    """Custom context manager:
    Rich activate, alembic logger to StringIO,
    stdout redirect, run upgrade."""
    logger_controller = RichLoggingController()
    logger_controller.activate()

    stdout_capture = StringIO()

    def _set_alembic_logger_to_stringio(stringio):
        logger = logging.getLogger("alembic")
        logger.handlers = []  # Remove any existing handlers
        handler = logging.StreamHandler(stringio)
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    try:
        if in_notebook():
            from IPython.utils.io import capture_output
            _set_alembic_logger_to_stringio(stdout_capture)
            with redirect_stdout(stdout_capture), capture_output(display=True):
                command.upgrade(alembic_cfg, "head")
        else:
            with redirect_stdout(stdout_capture):
                command.upgrade(alembic_cfg, "head")
    except Exception as ex:
        print(ex)
    finally:
        logger_controller.deactivate()

    yield stdout_capture


def run_alembic_upgrade_once():
    global _alembic_upgraded

    if not _alembic_upgraded:
        file_dir = os.path.dirname(os.path.abspath(__file__))
        alembic_ini_path = os.path.join(
            file_dir, "db", "alembic", "alembic.ini")
        # @see https://stackoverflow.com/questions/78780118/
        alembic_cfg = Config(alembic_ini_path,
                             attributes={"configure_logger": False})
        db_url = os.environ["RP_METADATASTORE_URL"]
        alembic_cfg.set_main_option(
            "sqlalchemy.url",
            db_url
        )

        # not logging anything if there's no upgrade
        # => capturing stream and displaying if elligible.
        stdout_capture = StringIO()
        with alembic_capture(alembic_cfg) as stdout_capture:
            stdout_output = stdout_capture.getvalue()

        if "Running upgrade" in stdout_output:
            if in_notebook():
                # tried to redirect notebook stdout to rich handler
                # but couldn't succeed so, formatting raw text =>
                logger_controller = RichLoggingController()
                logger_controller.activate()
                logging.getLogger().info(stdout_output)
                logger_controller.deactivate()
            else:
                print(stdout_output)


        _alembic_upgraded = True


if __name__ == "__main__":
    from retrain_pipelines import config

    run_alembic_upgrade_once()

