"""
Fractal Animation Evolution System

This package provides a complete system for creating and evolving fractal animations
using evolutionary algorithms and Mandelbulber integration.

Main Components:
- AnimationParameters: Define keyframe-based fractal animations
- AnimationRenderer: Render frame sequences and create videos
- AnimationEvolutionEngine: Evolve animations using genetic algorithms
- AnimationTemplates: Pre-configured animation templates

Example Usage:
    from FractalAnimator import AnimationTemplates, AnimationRenderer, AnimationEvolutionEngine
    
    # Create animation from template
    animation = AnimationTemplates.orbital_mandelbulb(duration=10.0)
    
    # Set up renderer and evolution
    renderer = AnimationRenderer()
    evolution = AnimationEvolutionEngine(renderer)
    
    # Evolve animations
    population = evolution.evolve_animations([animation])
"""

from .animation_parameters import AnimationParameters, CameraPath, InterpolationType, Keyframe
from .animation_renderer import AnimationRenderer, AnimationRenderResult
from .animation_evolution import AnimationEvolutionEngine, AnimationIndividual, AnimationFitness
from .animation_templates import AnimationTemplates

__version__ = "1.0.0"

__all__ = [
    'AnimationParameters',
    'CameraPath', 
    'InterpolationType',
    'Keyframe',
    'AnimationRenderer',
    'AnimationRenderResult',
    'AnimationEvolutionEngine',
    'AnimationIndividual',
    'AnimationFitness',
    'AnimationTemplates'
]