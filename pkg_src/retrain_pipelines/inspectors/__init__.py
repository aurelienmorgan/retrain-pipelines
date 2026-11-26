"""
Provider of convenience methods for investigation after pipeline execution.

In short : they are intended for some "after the facts" analysis.
"""

from .utils import browse_pipeline_card as browse_pipeline_card
from .utils import open_pipeline_card_file as open_pipeline_card_file

__all__ = [
    "open_pipeline_card_file",
    "browse_pipeline_card",
]
