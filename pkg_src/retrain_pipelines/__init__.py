
# import sys
# sys.path.append('..')

import os
import time
import logging

from IPython import get_ipython

import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from . import config        # <<== !! first ever import of the lib !!

from .utils import animate_wave
from .__version__ import __version__

if get_ipython() is not None:
    from . import legacy_launcher                                   ## <<== LEGACY  -  DELETE

from .dag_engine import run_alembic_upgrade_once


################################################################


if (
    not os.getenv("RP_LAUNCHER_SUPPRESS_WAVE", None) and
    not os.getenv("RP_LAUNCHER_SUBPROCESS", None) and
    # only if not pipeline launch (LEGACY  -  DELETE below cond)
    not os.getenv("retrain_pipeline_type", None)
):
    animate_wave(f"retrain-pipelines {__version__}",
                 wave_length=6, delay=0.01, loops=2)
    cli_wave_showed = True
else:
    cli_wave_showed = False


################################################################


logging.getLogger().debug(f"cache root directory : {os.environ['RP_ASSETS_CACHE']}")
if not bool(os.getenv("ALEMBIC_REV_AUTOGEN", False)):
    run_alembic_upgrade_once()

