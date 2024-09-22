
"""
The purpose of the inspectors
is to provide convenience methods
for investigation after pipeline run.

In short : they are intended for
some "after the facts" analysis.
"""

from .utils import browse_local_pipeline_card

from .wandb import get_execution_source_code, \
                   explore_source_code

from .. import cli_wave_showed

if not cli_wave_showed:
    from ..utils import animate_wave
    from ..__version__ import __version__

    animate_wave(f"retrain-pipelines {__version__}",
                 duration=0.5, fps=100)
