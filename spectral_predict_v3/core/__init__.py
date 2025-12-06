"""Core backend module - wraps v1's proven functionality."""

from .types import SpectralDataset, LoadResult, MergeResult
from .engine import Engine
from . import io_utils
from . import model_io

__all__ = [
    'SpectralDataset',
    'LoadResult',
    'MergeResult',
    'Engine',
    'io_utils',
    'model_io',
]
