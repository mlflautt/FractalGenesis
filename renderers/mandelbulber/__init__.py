"""
Mandelbulber renderer interface module.
Provides programmatic access to Mandelbulber 3D fractal rendering.
"""

from .renderer import MandelbulberRenderer
from .parameters import MandelbulberParameters
from .templates import ParameterTemplates

__all__ = ['MandelbulberRenderer', 'MandelbulberParameters', 'ParameterTemplates']
