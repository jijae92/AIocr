"""
Model interfaces and implementations.

This module provides both the base interface (BaseOCRModel) and
re-exports the actual engine implementations for convenience.
"""

from .base_model import BaseOCRModel

# Re-export engine implementations for backward compatibility
try:
    from ..engines.donut_engine import DonutEngine
    from ..engines.trocr_onnx import TrOCRONNXEngine
    from ..engines.tatr_tables import TATREngine
    from ..engines.pix2tex_math import Pix2TexEngine
except ImportError:
    # Engines not available yet
    pass

__all__ = [
    "BaseOCRModel",
    "DonutEngine",
    "TrOCRONNXEngine",
    "TATREngine",
    "Pix2TexEngine",
]
