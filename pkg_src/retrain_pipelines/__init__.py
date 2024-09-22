
# import sys
# sys.path.append('..')

from .utils import animate_wave
from .__version__ import __version__

import os
from IPython import get_ipython

if get_ipython() is not None:
    from . import local_launcher

if not os.getenv("retrain_pipeline_type", None):
    animate_wave(f"retrain-pipelines {__version__}",
                 duration=0.5, fps=100)
    cli_wave_showed = True
else:
    cli_wave_showed = False

