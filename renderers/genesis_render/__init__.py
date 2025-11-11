"""
GenesisRender - Professional 3D Fractal Rendering System
=======================================================

Complete fractal rendering system with:
- 12+ fractal types
- Advanced lighting and materials
- Smooth animation system
- High-performance JIT rendering
"""

from .genesis_render import GenesisRender, GenesisRenderParams
from .fractal_types import FRACTAL_TYPES, FRACTAL_PRESETS
from .animation_system import AnimationSequence, AnimationTemplates

__all__ = ['GenesisRender', 'GenesisRenderParams', 'FRACTAL_TYPES', 'FRACTAL_PRESETS', 'AnimationSequence', 'AnimationTemplates']
