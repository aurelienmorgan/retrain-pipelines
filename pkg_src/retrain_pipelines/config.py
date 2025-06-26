import os
import logging

from rich.logging import RichHandler
from rich.text import Text


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


################################################################

