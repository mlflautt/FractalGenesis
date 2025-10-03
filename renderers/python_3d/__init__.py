"""
Python 3D Fractal Renderer
=========================

High-performance Python-based 3D fractal rendering system with:
- Numba JIT compilation for speed
- Multiple fractal types (Mandelbulb, Julia sets, extensible)
- Advanced materials and lighting
- Animation capabilities
- Complete parameter control

This replaces Mandelbulber for better integration and control.
"""

from .fractal_animator import FractalRenderer, FractalParams, FractalPresets

__all__ = ['FractalRenderer', 'FractalParams', 'FractalPresets']