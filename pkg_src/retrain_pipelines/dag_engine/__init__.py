import os
import sys
import logging

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from IPython import get_ipython

from alembic import command
from alembic.config import Config


_alembic_upgraded = False


def _set_alembic_logger_to_stringio(stringio):
    logger = logging.getLogger('alembic')
    logger.handlers = []  # Remove any existing handlers
    handler = logging.StreamHandler(stringio)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def run_alembic_upgrade_once():
    global _alembic_upgraded
    if not _alembic_upgraded:
        file_dir = os.path.dirname(os.path.abspath(__file__))
        alembic_ini_path = os.path.join(file_dir, "alembic", "alembic.ini")
        # @see https://stackoverflow.com/questions/78780118/
        alembic_cfg = Config(alembic_ini_path, attributes={"configure_logger": False})
        db_url = f"sqlite:///{os.environ['RP_ASSETS_CACHE']}local_metadatastore.db"
        alembic_cfg.set_main_option(
            "sqlalchemy.url",
            db_url
        )

        stdout_capture = StringIO()
        in_jupyter = get_ipython() is not None and hasattr(sys, 'ps1')
        if in_jupyter:
            from IPython.utils.io import capture_output
            _set_alembic_logger_to_stringio(stdout_capture)
            with redirect_stdout(stdout_capture), capture_output(display=True):
                    command.upgrade(alembic_cfg, "head")
        else:
            with redirect_stdout(stdout_capture):
                command.upgrade(alembic_cfg, "head")

        stdout_output = stdout_capture.getvalue().strip()
        if "Running upgrade" in stdout_output:
            if in_jupyter:
                # tried to redirect notebook stdout to rich handler
                # but couldn't succeed so, formatting raw text =>
                logging.getLogger().info(stdout_output)
            else:
                print(stdout_output)

        _alembic_upgraded = True


if __name__ == "__main__":
    from retrain_pipelines import config

    run_alembic_upgrade_once()

