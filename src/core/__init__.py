"""
Core functionality for SmartMirror.
"""

from .realtime_pipeline import RealTimePipeline
from .dataset_builder import ClassificationDatasetBuilder

__all__ = [
    'RealTimePipeline',
    'ClassificationDatasetBuilder',
] 