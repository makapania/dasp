"""Analysis engine - Core spectral analysis functionality.

This layer wraps the existing src/spectral_predict/ modules with a clean API.
"""
from .api import EngineAPI

__all__ = ["EngineAPI"]
