"""
Animation Package
=================

Advanced animation system for FractalGenesis with:
- Procedural generators (orbits, formula evolution, colors)
- Animation presets & templates
- Keyframe interpolation
- Audio-reactive features

Usage:
    from animation import AnimationController, OrbitGenerator
    from animation.presets import AnimationPresets
"""

from .animation_controller import (
    AnimationController,
    AnimationKeyframe,
    EasingType,
    MandelbulbMorph,
    EasingFunctions,
    CatmullRomSpline
)

from .procedural_generators import (
    OrbitGenerator,
    OrbitConfig,
    OrbitPattern,
    FormulaEvolutionGenerator,
    ColorAnimationGenerator
)

from .presets import (
    AnimationPreset,
    AnimationPresets,
    PresetApplicator,
    save_preset,
    load_preset
)

__all__ = [
    # Core Animation
    'AnimationController',
    'AnimationKeyframe',
    'EasingType',
    'MandelbulbMorph',
    'EasingFunctions',
    'CatmullRomSpline',
    
    # Procedural Generators
    'OrbitGenerator',
    'OrbitConfig',
    'OrbitPattern',
    'FormulaEvolutionGenerator',
    'ColorAnimationGenerator',
    
    # Presets
    'AnimationPreset',
    'AnimationPresets',
    'PresetApplicator',
    'save_preset',
    'load_preset'
]

__version__ = "1.0.0"
