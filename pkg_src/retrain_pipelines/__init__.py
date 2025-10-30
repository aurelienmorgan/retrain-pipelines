
# import sys
# sys.path.append('..')

import os
import time
import logging

from IPython import get_ipython

from . import config        # <<== !! first ever import of the lib !!

from .utils import animate_wave
from .__version__ import __version__

if get_ipython() is not None:
    from . import legacy_launcher

from .dag_engine import run_alembic_upgrade_once


################################################################


if not os.getenv("retrain_pipeline_type", None):
    # only if not Metaflow pipeline launch (TODO: DELETE if statement)
    animate_wave(f"retrain-pipelines {__version__}",
                 wave_length=6, delay=0.01, loops=2)
    cli_wave_showed = True                                  ##  LEGACY DELETE ##
else:                                                       ##  LEGACY DELETE ##
    cli_wave_showed = False                                 ##  LEGACY DELETE ##


################################################################


logging.getLogger().debug(f"cache root directory : {os.environ['RP_ASSETS_CACHE']}")
if not bool(os.getenv("ALEMBIC_REV_AUTOGEN", False)):
    run_alembic_upgrade_once()

