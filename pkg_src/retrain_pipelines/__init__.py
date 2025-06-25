
# import sys
# sys.path.append('..')
from .utils import animate_wave
from .__version__ import __version__

import os
from IPython import get_ipython

if get_ipython() is not None:
    from . import local_launcher


################################################################


import logging
import time

from rich.live import Live
from rich.logging import LogRender, RichHandler
from rich.text import Text

# https://rich.readthedocs.io/en/stable/markup.html


class CustomRichHandler(RichHandler):
    def get_level_text(self, record: logging.LogRecord) -> Text:
        # Override to hide level text for INFO logs
        if record.levelno == logging.INFO:
            return Text("")  # Return empty text to hide level
        return super().get_level_text(record)


FORMAT = "%(message)s"  # Define a basic log message format
DATEFMT = "%H:%M:%S"


# Configure the root logger to use RichHandler
logging.basicConfig(
    level="NOTSET",  # Log all levels
    format=FORMAT,  # Use simple message format
    datefmt=DATEFMT,  # Time format (optional)
    handlers=[CustomRichHandler(markup=True)],
)


################################################################


if not os.getenv("retrain_pipeline_type", None):
    # only if not Metaflow pipeline launch (TODO: DELETE if statement)
    animate_wave(f"retrain-pipelines {__version__}",
                 wave_length=6, delay=0.01, loops=2)

