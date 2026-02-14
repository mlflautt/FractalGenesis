#!/usr/bin/env python3
"""
Animation Presets & Templates
==============================

Pre-built animation configurations for common use cases:
- Cinematic camera movements
- Formula exploration sequences
- Color transitions
- Music visualization templates

Usage:
    from animation.presets import AnimationPresets
    
    # Apply cinematic preset
    preset = AnimationPresets.cinematic_orbit()
    anim.apply_preset(preset)
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

from animation_controller import AnimationController, EasingType
from renderers.python_3d.fractal_animator import FractalParams, FractalPresets
from animation.procedural_generators import (
    OrbitGenerator, OrbitConfig, OrbitPattern,
    FormulaEvolutionGenerator, ColorAnimationGenerator
)


@dataclass
class AnimationPreset:
    """Represents a pre-built animation configuration"""
    name: str
    description: str
    category: str
    duration: float  # seconds
    keyframe_count: int
    generator_config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'description': self.description,
            'category': self.category,
            'duration': self.duration,
            'keyframe_count': self.keyframe_count,
            'generator_config': self.generator_config
        }


class AnimationPresets:
    """Library of pre-built animation presets"""
    
    @staticmethod
    def cinematic_orbit() -> AnimationPreset:
        """
        Cinematic slow orbit around fractal.
        
        Professional camera movement for showcasing fractal structures.
        Smooth spherical orbit with subtle height variation.
        """
        return AnimationPreset(
            name="Cinematic Orbit",
            description="Slow, smooth orbit around fractal center with professional camera movement",
            category="camera",
            duration=20.0,
            keyframe_count=600,
            generator_config={
                'type': 'orbit',
                'pattern': 'spherical',
                'radius': 3.0,
                'height_variation': 0.3,
                'speed': 1.0,
                'easing': 'ease_in_out'
            }
        )
    
    @staticmethod
    def dramatic_zoom() -> AnimationPreset:
        """
        Dramatic slow zoom into fractal detail.
        
        Starts wide, smoothly pushes in to reveal intricate details.
        Great for showcasing complexity.
        """
        return AnimationPreset(
            name="Dramatic Zoom",
            description="Slow push-in revealing fractal details",
            category="camera",
            duration=15.0,
            keyframe_count=450,
            generator_config={
                'type': 'camera_move',
                'start_distance': 5.0,
                'end_distance': 1.2,
                'easing': 'exponential'
            }
        )
    
    @staticmethod
    def power_exploration() -> AnimationPreset:
        """
        Explore different power values.
        
        Cycles through powers 2-16 to show how structure changes.
        Educational and visually interesting.
        """
        return AnimationPreset(
            name="Power Exploration",
            description="Smoothly transition between different power values",
            category="formula",
            duration=12.0,
            keyframe_count=360,
            generator_config={
                'type': 'formula_evolution',
                'evolution_type': 'power_sweep',
                'start_power': 2.0,
                'end_power': 16.0,
                'easing': 'ease_in_out'
            }
        )
    
    @staticmethod
    def formula_morph() -> AnimationPreset:
        """
        Morph between two fractal configurations.
        
        Smoothly blends all parameters between start and end states.
        """
        return AnimationPreset(
            name="Formula Morph",
            description="Morph between two fractal parameter sets",
            category="formula",
            duration=10.0,
            keyframe_count=300,
            generator_config={
                'type': 'formula_evolution',
                'evolution_type': 'morph',
                'easing': 'ease_in_out'
            }
        )
    
    @staticmethod
    def psychedelic_color() -> AnimationPreset:
        """
        Psychedelic color cycling.
        
        Full spectrum hue rotation with intensity pulsing.
        Great for music visualization.
        """
        return AnimationPreset(
            name="Psychedelic Colors",
            description="Full spectrum color cycling with intensity modulation",
            category="color",
            duration=8.0,
            keyframe_count=240,
            generator_config={
                'type': 'color',
                'color_type': 'hue_rotation',
                'rotations': 2.0,
                'intensity_pulse': True
            }
        )
    
    @staticmethod
    def seasonal_transition() -> AnimationPreset:
        """
        Transition through seasonal color palettes.
        
        Warm (spring) → Cool (summer) → Warm (autumn) → Cool (winter)
        """
        return AnimationPreset(
            name="Seasonal Transition",
            description="Cycle through seasonal color palettes",
            category="color",
            duration=16.0,
            keyframe_count=480,
            generator_config={
                'type': 'color',
                'color_type': 'palette_cycle',
                'palettes': ['spring', 'summer', 'autumn', 'winter'],
                'hold_time': 3.0,
                'transition_time': 1.0
            }
        )
    
    @staticmethod
    def torus_knot() -> AnimationPreset:
        """
        Complex torus knot orbit.
        
        Beautiful mathematical path showing fractal from all angles.
        """
        return AnimationPreset(
            name="Torus Knot",
            description="Complex torus knot camera path",
            category="camera",
            duration=18.0,
            keyframe_count=540,
            generator_config={
                'type': 'orbit',
                'pattern': 'toroidal',
                'radius': 3.5,
                'speed': 1.0,
                'easing': 'ease_in_out'
            }
        )
    
    @staticmethod
    def fly_through() -> AnimationPreset:
        """
        Fly through fractal interior.
        
        Starts outside, moves through surface, explores interior.
        Warning: Requires fractal with interior space.
        """
        return AnimationPreset(
            name="Interior Fly-Through",
            description="Camera flies through fractal interior",
            category="camera",
            duration=25.0,
            keyframe_count=750,
            generator_config={
                'type': 'camera_move',
                'path': 'through_center',
                'start_distance': 4.0,
                'end_distance': 0.0,
                'spiral_inward': True,
                'easing': 'ease_in_out'
            }
        )
    
    @staticmethod
    def detail_reveal() -> AnimationPreset:
        """
        Gradually increase iteration count to reveal detail.
        
        Starts low-res, progressively increases quality.
        """
        return AnimationPreset(
            name="Detail Reveal",
            description="Progressively increase iteration count to reveal fine details",
            category="formula",
            duration=10.0,
            keyframe_count=300,
            generator_config={
                'type': 'formula_evolution',
                'evolution_type': 'iteration_ramp',
                'start_iter': 50,
                'end_iter': 250,
                'easing': 'ease_in_out'
            }
        )
    
    @staticmethod
    def chaotic_dance() -> AnimationPreset:
        """
        Random walk orbit with erratic movements.
        
        Unpredictable camera path for chaotic, energetic feel.
        """
        return AnimationPreset(
            name="Chaotic Dance",
            description="Unpredictable random walk camera movement",
            category="camera",
            duration=12.0,
            keyframe_count=360,
            generator_config={
                'type': 'orbit',
                'pattern': 'random_walk',
                'radius': 3.0,
                'speed': 2.0,
                'easing': 'linear'
            }
        )
    
    @staticmethod
    def helix_ascent() -> AnimationPreset:
        """
        Helical path moving upward.
        
        Spirals around fractal while gaining altitude.
        """
        return AnimationPreset(
            name="Helix Ascent",
            description="Spiral upward around fractal",
            category="camera",
            duration=15.0,
            keyframe_count=450,
            generator_config={
                'type': 'orbit',
                'pattern': 'helix',
                'radius': 2.5,
                'height_variation': 2.0,
                'speed': 1.5,
                'easing': 'ease_in_out'
            }
        )
    
    @staticmethod
    def music_visualizer_template() -> AnimationPreset:
        """
        Template for audio-reactive animation.
        
        Sets up parameters ready for audio input.
        """
        return AnimationPreset(
            name="Music Visualizer",
            description="Audio-reactive template for music synchronization",
            category="audio",
            duration=30.0,
            keyframe_count=900,
            generator_config={
                'type': 'audio_reactive',
                'audio_mappings': {
                    'bass': {'param': 'camera_distance', 'range': [2.0, 4.0]},
                    'mids': {'param': 'power', 'range': [6.0, 12.0]},
                    'highs': {'param': 'rotation_speed', 'range': [0.0, 2.0]},
                    'onset': {'param': 'color_shift', 'trigger': True}
                }
            }
        )
    
    @classmethod
    def get_all_presets(cls) -> List[AnimationPreset]:
        """Get all available presets"""
        return [
            cls.cinematic_orbit(),
            cls.dramatic_zoom(),
            cls.power_exploration(),
            cls.formula_morph(),
            cls.psychedelic_color(),
            cls.seasonal_transition(),
            cls.torus_knot(),
            cls.fly_through(),
            cls.detail_reveal(),
            cls.chaotic_dance(),
            cls.helix_ascent(),
            cls.music_visualizer_template()
        ]
    
    @classmethod
    def get_by_category(cls, category: str) -> List[AnimationPreset]:
        """Get presets filtered by category"""
        return [p for p in cls.get_all_presets() if p.category == category]
    
    @classmethod
    def get_preset(cls, name: str) -> Optional[AnimationPreset]:
        """Get preset by name"""
        for preset in cls.get_all_presets():
            if preset.name.lower() == name.lower():
                return preset
        return None


class PresetApplicator:
    """
    Applies presets to animation controllers.
    
    Converts preset configurations into actual keyframes
    and applies them to the animation.
    """
    
    @staticmethod
    def apply_preset(
        controller: AnimationController,
        preset: AnimationPreset,
        base_params: Optional[FractalParams] = None
    ) -> AnimationController:
        """
        Apply a preset to an animation controller.
        
        Args:
            controller: The animation controller to modify
            preset: The preset to apply
            base_params: Optional base parameters to start from
            
        Returns:
            Modified controller with preset applied
        """
        config = preset.generator_config
        gen_type = config.get('type')
        
        if gen_type == 'orbit':
            PresetApplicator._apply_orbit_preset(controller, config, base_params)
        
        elif gen_type == 'formula_evolution':
            PresetApplicator._apply_formula_preset(controller, config, base_params)
        
        elif gen_type == 'color':
            PresetApplicator._apply_color_preset(controller, config, base_params)
        
        elif gen_type == 'camera_move':
            PresetApplicator._apply_camera_preset(controller, config, base_params)
        
        return controller
    
    @staticmethod
    def _apply_orbit_preset(
        controller: AnimationController,
        config: Dict[str, Any],
        base_params: Optional[FractalParams]
    ):
        """Apply an orbit preset"""
        pattern = OrbitPattern(config.get('pattern', 'spherical'))
        
        orbit_config = OrbitConfig(
            pattern=pattern,
            radius=config.get('radius', 3.0),
            height_variation=config.get('height_variation', 0.0),
            speed=config.get('speed', 1.0)
        )
        
        generator = OrbitGenerator(orbit_config)
        frames = config.get('frames', 120)
        
        keyframes = generator.generate(frames=frames)
        
        # Apply base params to all keyframes if provided
        if base_params:
            for kf in keyframes:
                # Merge base params with generated
                kf.params.power = base_params.power
                kf.params.iterations = base_params.iterations
                kf.params.color_palette = base_params.color_palette
                kf.params.base_color = base_params.base_color
        
        # Add to controller
        controller.keyframes = keyframes
    
    @staticmethod
    def _apply_formula_preset(
        controller: AnimationController,
        config: Dict[str, Any],
        base_params: Optional[FractalParams]
    ):
        """Apply a formula evolution preset"""
        evolution_type = config.get('evolution_type')
        frames = config.get('frames', 120)
        
        if evolution_type == 'power_sweep':
            start_power = config.get('start_power', 2.0)
            end_power = config.get('end_power', 16.0)
            
            keyframes = FormulaEvolutionGenerator.generate_power_sweep(
                start_power=start_power,
                end_power=end_power,
                frames=frames
            )
        
        elif evolution_type == 'iteration_ramp':
            start_iter = config.get('start_iter', 50)
            end_iter = config.get('end_iter', 200)
            
            keyframes = FormulaEvolutionGenerator.generate_iteration_ramp(
                start_iter=start_iter,
                end_iter=end_iter,
                frames=frames
            )
        
        elif evolution_type == 'morph':
            # For morph, need start and end params
            if base_params:
                # Create a slightly modified version
                end_params = FractalParams()
                end_params.power = base_params.power * 1.5
                end_params.camera_pos = (
                    base_params.camera_pos[0] + 1.0,
                    base_params.camera_pos[1],
                    base_params.camera_pos[2]
                )
                
                keyframes = FormulaEvolutionGenerator.generate_formula_morph(
                    params1=base_params,
                    params2=end_params,
                    frames=frames
                )
            else:
                # Default morph
                params1 = FractalPresets.get_preset("classic_mandelbulb")
                params2 = FractalPresets.get_preset("high_power_mandelbulb")
                keyframes = FormulaEvolutionGenerator.generate_formula_morph(
                    params1=params1,
                    params2=params2,
                    frames=frames
                )
        
        else:
            return
        
        controller.keyframes = keyframes
    
    @staticmethod
    def _apply_color_preset(
        controller: AnimationController,
        config: Dict[str, Any],
        base_params: Optional[FractalParams]
    ):
        """Apply a color animation preset"""
        color_type = config.get('color_type')
        
        if color_type == 'hue_rotation':
            rotations = config.get('rotations', 2.0)
            frames = config.get('frames', 120)
            
            keyframes = ColorAnimationGenerator.generate_hue_rotation(
                rotations=rotations,
                frames=frames
            )
        
        elif color_type == 'palette_cycle':
            # For palette cycling, would need palette library
            # Simplified version uses hue rotation
            keyframes = ColorAnimationGenerator.generate_hue_rotation(
                rotations=1.0,
                frames=120
            )
        
        else:
            return
        
        controller.keyframes = keyframes
    
    @staticmethod
    def _apply_camera_preset(
        controller: AnimationController,
        config: Dict[str, Any],
        base_params: Optional[FractalParams]
    ):
        """Apply a camera movement preset"""
        # Implementation for camera movement presets
        # Would generate keyframes with camera position changes
        pass


def save_preset(preset: AnimationPreset, filepath: str):
    """Save preset to JSON file"""
    data = preset.to_dict()
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_preset(filepath: str) -> AnimationPreset:
    """Load preset from JSON file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return AnimationPreset(
        name=data['name'],
        description=data['description'],
        category=data['category'],
        duration=data['duration'],
        keyframe_count=data['keyframe_count'],
        generator_config=data.get('generator_config', {})
    )


def demo_presets():
    """Demonstrate preset system"""
    print("Animation Presets Demo")
    print("=" * 60)
    
    # Show all presets
    print("\nAvailable Presets:")
    print("-" * 60)
    
    presets = AnimationPresets.get_all_presets()
    
    # Group by category
    categories = {}
    for preset in presets:
        if preset.category not in categories:
            categories[preset.category] = []
        categories[preset.category].append(preset)
    
    for category, cat_presets in sorted(categories.items()):
        print(f"\n{category.upper()} ({len(cat_presets)} presets):")
        for preset in cat_presets:
            print(f"  • {preset.name}")
            print(f"    {preset.description}")
            print(f"    Duration: {preset.duration}s, Keyframes: {preset.keyframe_count}")
    
    # Demo applying a preset
    print("\n" + "=" * 60)
    print("Applying Preset Demo:")
    print("-" * 60)
    
    from animation_controller import AnimationController
    
    controller = AnimationController()
    preset = AnimationPresets.cinematic_orbit()
    
    print(f"\nApplying: {preset.name}")
    applicator = PresetApplicator()
    applicator.apply_preset(controller, preset)
    
    print(f"Generated {len(controller.keyframes)} keyframes")
    print(f"Animation duration: {preset.duration}s")
    
    # Show first and last keyframe
    if controller.keyframes:
        first = controller.keyframes[0]
        last = controller.keyframes[-1]
        print(f"\nFirst keyframe (t={first.time}):")
        print(f"  Camera: {first.params.camera_pos}")
        print(f"\nLast keyframe (t={last.time}):")
        print(f"  Camera: {last.params.camera_pos}")
    
    print("\nDemo complete!")


if __name__ == "__main__":
    demo_presets()
