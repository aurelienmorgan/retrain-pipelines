"""
Provider of convenience methods for investigation after pipeline execution.

In short : they are intended for some "after the facts" analysis.
"""

from .. import cli_wave_showed
from .utils import browse_local_pipeline_card as browse_local_pipeline_card
from .utils import browse_pipeline_card as browse_pipeline_card
from .wandb import explore_source_code as explore_source_code
from .wandb import get_execution_source_code as get_execution_source_code

__all__ = [
    "browse_local_pipeline_card",
    "browse_pipeline_card",
    "explore_source_code",
    "get_execution_source_code",
]

if not cli_wave_showed:
    from ..__version__ import __version__ as __version__
    from ..utils import animate_wave as animate_wave

    animate_wave(f"retrain-pipelines {__version__}")
