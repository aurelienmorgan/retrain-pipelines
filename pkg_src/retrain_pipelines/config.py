import os
import logging

from rich.logging import RichHandler
from rich.text import Text


# RP_ASSETS_CACHE
# lib root dir for default locations
os.environ["RP_ASSETS_CACHE"] = (
    os.environ.get(
        "RP_ASSETS_CACHE",
        os.path.expanduser("~/.cache/retrain-pipelines/")
    ).rstrip(os.sep) + os.sep
)
try:
    os.makedirs(os.environ["RP_ASSETS_CACHE"], exist_ok=True)
except PermissionError as e:
    print(f"Permission denied: {e}")
except Exception as e:
    print(f"Error creating cache directory: {e}")


# RP_METADATASTORE_URL
# We instruct SQLite to wait for a lock to be released
# before raising a "db locked" error on conccurency issue.
# Setting a timeout in the URL
os.environ["RP_METADATASTORE_URL"] = (
    os.environ.get(
        "RP_METADATASTORE_URL",
        f"sqlite:///{os.environ['RP_ASSETS_CACHE']}local_metadatastore.db?timeout=10.0"
    )
)
# RP_METADATASTORE_ASYNC_URL
# For the WebConsole connection pool
os.environ["RP_METADATASTORE_ASYNC_URL"] = (
    os.environ.get(
        "RP_METADATASTORE_ASYNC_URL",
        f"sqlite+aiosqlite:///{os.environ['RP_ASSETS_CACHE']}local_metadatastore.db"
    )
)


# RP_ARTIFACTS_STORE
# non-model, non-dataset, and non-metadata artifacts
# (Training code, scripts, and configuration files,
#  Data validation reports, charts, and documentation,
#  Feature engineering outputs, etc.)
os.environ["RP_ARTIFACTS_STORE"] = (
    os.environ.get(
        "RP_ARTIFACTS_STORE",
        os.path.join(os.path.expanduser(os.environ["RP_ASSETS_CACHE"]),
                     "artifacts")
    ).rstrip(os.sep) + os.sep
)
try:
    os.makedirs(os.environ["RP_ARTIFACTS_STORE"], exist_ok=True)
except PermissionError as e:
    print(f"Permission denied: {e}")
except Exception as e:
    print(f"Error creating cache directory: {e}")


# RP_WEB_SERVER_URL
os.environ["RP_WEB_SERVER_URL"] = (
    os.environ.get(
        "RP_WEB_SERVER_URL", "http://localhost:5001/"
    ).rstrip("/")
)


# RP_WEB_SERVER_LOGS
# for 'local' installs with web UI
os.environ["RP_WEB_SERVER_LOGS"] = (
    os.environ.get(
        "RP_WEB_SERVER_LOGS",
        os.path.join(os.path.expanduser(os.environ["RP_ASSETS_CACHE"]),
                     "logs", "web_server")
    ).rstrip(os.sep) + os.sep
)
try:
    os.makedirs(os.environ["RP_WEB_SERVER_LOGS"], exist_ok=True)
except PermissionError as e:
    print(f"Permission denied: {e}")
except Exception as e:
    print(f"Error creating cache directory: {e}")


################################################################

# https://rich.readthedocs.io/en/stable/markup.html

class CustomRichHandler(RichHandler):
    def get_level_text(self, record: logging.LogRecord) -> Text:
        # Override to hide level text for INFO logs
        if record.levelno == logging.INFO:
            return Text("")  # Return empty text to hide level
        return super().get_level_text(record)


FORMAT = "%(message)s"  # Basic log message format
DATEFMT = "%H:%M:%S"


# Configure the root logger to use RichHandler
logging.basicConfig(
    level="NOTSET",     # Log all levels
    format=FORMAT,
    datefmt=DATEFMT,
    handlers=[CustomRichHandler(markup=True)],
)
# for overly noisy dependencies
logging.getLogger("matplotlib").setLevel(logging.ERROR)
logging.getLogger("PIL.PngImagePlugin").setLevel(logging.ERROR)
logging.getLogger("graphviz").setLevel(logging.ERROR)
logging.getLogger("python_multipart").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger("tzlocal").setLevel(logging.ERROR)
logging.getLogger("aiosqlite").setLevel(logging.ERROR)

################################################################

