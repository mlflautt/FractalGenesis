"""
Animation Templates and Presets

This module provides pre-configured animation templates for common fractal animation types,
including orbital motion, zoom sequences, color morphing, and complex transformations.
These serve as starting points for evolution and manual animation creation.
"""

import random
import math
from typing import Dict, List
from .animation_parameters import AnimationParameters, CameraPath, InterpolationType
from renderers.mandelbulber.templates import ParameterTemplates


class AnimationTemplates:
    """
    Collection of pre-configured animation templates.
    
    These templates provide good starting points for different animation types
    and can be used to seed evolution or create manual animations.
    """
    
    @staticmethod
    def orbital_mandelbulb(duration: float = 10.0, revolutions: float = 1.0) -> AnimationParameters:
        """
        Create an orbital animation around a Mandelbulb.
        
        Args:
            duration: Animation duration in seconds
            revolutions: Number of complete orbits
            
        Returns:
            Animation parameters for orbital motion
        """
        params = AnimationParameters()
        params.duration_seconds = duration
        params.fps = 30
        
        # Set up orbital camera path
        params.camera_path = CameraPath(
            path_type="orbit",
            orbit_radius=5.0,
            orbit_height=1.0,
            orbit_speed=revolutions,
            focus_point=(0.0, 0.0, 0.0)
        )
        
        # Use classic mandelbulb as base
        base_template = ParameterTemplates.classic_mandelbulb()
        
        # Create keyframes with slight parameter variations
        start_params = base_template.copy()
        end_params = base_template.copy()
        end_params.fractal.power = base_template.fractal.power + 1.0
        end_params.material.surface_color_r = min(1.0, base_template.material.surface_color_r + 0.2)
        
        params.add_keyframe(0.0, start_params, InterpolationType.EASE_IN_OUT)
        params.add_keyframe(1.0, end_params, InterpolationType.EASE_IN_OUT)
        
        return params
    
    @staticmethod
    def zoom_into_fractal(duration: float = 8.0, zoom_factor: float = 0.8) -> AnimationParameters:
        """
        Create a zoom-in animation towards the fractal.
        
        Args:
            duration: Animation duration in seconds
            zoom_factor: How much to zoom in (0.0 = no zoom, 1.0 = complete zoom)
            
        Returns:
            Animation parameters for zoom motion
        """
        params = AnimationParameters()
        params.duration_seconds = duration
        params.fps = 30
        
        # Set up zoom camera path
        params.camera_path = CameraPath(
            path_type="zoom",
            orbit_radius=8.0,
            zoom_factor=zoom_factor,
            focus_point=(0.0, 0.0, 0.0)
        )
        
        # Use high power mandelbulb for interesting details when zooming
        base_template = ParameterTemplates.high_power_mandelbulb()
        
        # Start with lower iterations, increase as we zoom in for more detail
        start_params = base_template.copy()
        start_params.fractal.iterations = 150
        
        end_params = base_template.copy() 
        end_params.fractal.iterations = 300
        end_params.render.detail_level = 1.5
        
        params.add_keyframe(0.0, start_params, InterpolationType.EASE_IN_OUT)
        params.add_keyframe(1.0, end_params, InterpolationType.EASE_IN_OUT)
        
        return params
    
    @staticmethod
    def color_morph_sequence(duration: float = 12.0) -> AnimationParameters:
        """
        Create an animation that morphs through different color schemes.
        
        Args:
            duration: Animation duration in seconds
            
        Returns:
            Animation parameters with color morphing
        """
        params = AnimationParameters()
        params.duration_seconds = duration
        params.fps = 30
        
        # Gentle orbital motion
        params.camera_path = CameraPath(
            path_type="orbit",
            orbit_radius=4.0,
            orbit_height=0.5,
            orbit_speed=0.5,
            focus_point=(0.0, 0.0, 0.0)
        )
        
        # Create color progression through different templates
        templates = [
            ParameterTemplates.classic_mandelbulb(),    # Blue/purple
            ParameterTemplates.golden_mandelbulb(),     # Gold
            ParameterTemplates.high_power_mandelbulb(), # Purple/magenta
        ]
        
        # Add intermediate color variations
        for i, template in enumerate(templates):
            time = i / max(1, len(templates) - 1)
            params.add_keyframe(time, template, InterpolationType.EASE_IN_OUT)
        
        # Add a mid-point with unique coloring
        if len(templates) >= 2:
            mid_template = templates[0].copy()
            mid_template.material.surface_color_r = 0.2
            mid_template.material.surface_color_g = 0.9
            mid_template.material.surface_color_b = 0.3
            params.add_keyframe(0.5, mid_template, InterpolationType.EASE_IN_OUT)
        
        return params
    
    @staticmethod
    def power_evolution_sequence(duration: float = 15.0) -> AnimationParameters:
        """
        Create an animation that evolves fractal power over time.
        
        Args:
            duration: Animation duration in seconds
            
        Returns:
            Animation parameters with power evolution
        """
        params = AnimationParameters()
        params.duration_seconds = duration
        params.fps = 30
        
        # Spiral camera path to show the evolution from different angles
        params.camera_path = CameraPath(
            path_type="spiral",
            orbit_radius=6.0,
            orbit_height=2.0,
            orbit_speed=1.5,
            focus_point=(0.0, 0.0, 0.0)
        )
        
        base_template = ParameterTemplates.classic_mandelbulb()
        
        # Create keyframes with different powers
        powers = [2.0, 4.0, 8.0, 12.0, 16.0, 8.0]  # Return to middle for smooth loop
        
        for i, power in enumerate(powers):
            time = i / max(1, len(powers) - 1)
            template = base_template.copy()
            template.fractal.power = power
            
            # Adjust iterations based on power (higher power needs more iterations)
            template.fractal.iterations = min(400, int(150 + power * 8))
            
            # Slight color variation based on power
            color_factor = power / 16.0
            template.material.surface_color_r = 0.3 + color_factor * 0.5
            template.material.surface_color_b = 0.8 - color_factor * 0.3
            
            params.add_keyframe(time, template, InterpolationType.CUBIC)
        
        return params
    
    @staticmethod
    def julia_transformation_sequence(duration: float = 10.0) -> AnimationParameters:
        """
        Create an animation transitioning from Mandelbulb to Julia set.
        
        Args:
            duration: Animation duration in seconds
            
        Returns:
            Animation parameters for Julia transformation
        """
        params = AnimationParameters()
        params.duration_seconds = duration
        params.fps = 30
        
        # Orbital motion to show the transformation
        params.camera_path = CameraPath(
            path_type="orbit",
            orbit_radius=4.5,
            orbit_height=1.5,
            orbit_speed=1.2,
            focus_point=(0.0, 0.0, 0.0)
        )
        
        # Start with regular Mandelbulb
        start_template = ParameterTemplates.classic_mandelbulb()
        start_template.fractal.julia_mode = False
        
        # Transition to Julia mode
        mid_template = start_template.copy()
        mid_template.fractal.julia_mode = True
        mid_template.fractal.julia_c_x = 0.3
        mid_template.fractal.julia_c_y = 0.5
        mid_template.fractal.julia_c_z = 0.0
        
        # End with different Julia parameters
        end_template = mid_template.copy()
        end_template.fractal.julia_c_x = -0.2
        end_template.fractal.julia_c_y = 0.8
        end_template.fractal.julia_c_z = 0.1
        
        # Color changes to emphasize transformation
        end_template.material.surface_color_r = 0.9
        end_template.material.surface_color_g = 0.4
        end_template.material.surface_color_b = 0.1
        
        params.add_keyframe(0.0, start_template, InterpolationType.EASE_IN_OUT)
        params.add_keyframe(0.4, mid_template, InterpolationType.EASE_IN_OUT) 
        params.add_keyframe(1.0, end_template, InterpolationType.EASE_IN_OUT)
        
        return params
    
    @staticmethod
    def multi_formula_evolution(duration: float = 20.0) -> AnimationParameters:
        """
        Create an animation cycling through different fractal formulas.
        
        Args:
            duration: Animation duration in seconds
            
        Returns:
            Animation parameters with formula changes
        """
        params = AnimationParameters()
        params.duration_seconds = duration
        params.fps = 24  # Slightly lower fps for longer sequence
        
        # Complex camera path combining orbit and zoom
        params.camera_path = CameraPath(
            path_type="orbit",
            orbit_radius=5.5,
            orbit_height=2.0,
            orbit_speed=2.0,
            focus_point=(0.0, 0.0, 0.0)
        )
        
        # Use different templates representing different formulas
        templates_sequence = [
            ("mandelbulb", ParameterTemplates.classic_mandelbulb()),
            ("golden_mandelbulb", ParameterTemplates.golden_mandelbulb()),
            ("mandelbox", ParameterTemplates.classic_mandelbox()),
            ("menger_sponge", ParameterTemplates.menger_sponge()),
            ("high_power", ParameterTemplates.high_power_mandelbulb()),
            ("julia", ParameterTemplates.julia_mandelbulb())
        ]
        
        for i, (name, template) in enumerate(templates_sequence):
            time = i / max(1, len(templates_sequence) - 1)
            params.add_keyframe(time, template, InterpolationType.EASE_IN_OUT)
        
        return params
    
    @staticmethod
    def lighting_showcase(duration: float = 8.0) -> AnimationParameters:
        """
        Create an animation showcasing different lighting effects.
        
        Args:
            duration: Animation duration in seconds
            
        Returns:
            Animation parameters with lighting variations
        """
        params = AnimationParameters()
        params.duration_seconds = duration
        params.fps = 30
        
        # Static camera to focus on lighting changes
        params.camera_path = CameraPath(
            path_type="orbit",
            orbit_radius=4.0,
            orbit_height=0.0,
            orbit_speed=0.3,  # Very slow orbit
            focus_point=(0.0, 0.0, 0.0)
        )
        
        base_template = ParameterTemplates.golden_mandelbulb()
        
        # Create different lighting scenarios
        lighting_configs = [
            {"alpha": -45, "beta": 30, "intensity": 1.0, "r": 1.0, "g": 1.0, "b": 1.0},
            {"alpha": 90, "beta": 45, "intensity": 1.5, "r": 1.0, "g": 0.8, "b": 0.6},
            {"alpha": -120, "beta": 60, "intensity": 0.8, "r": 0.8, "g": 0.9, "b": 1.0},
            {"alpha": 0, "beta": 90, "intensity": 1.2, "r": 1.0, "g": 0.6, "b": 0.8},
        ]
        
        for i, lighting in enumerate(lighting_configs):
            time = i / max(1, len(lighting_configs) - 1)
            template = base_template.copy()
            
            template.lighting.main_light_alpha = lighting["alpha"]
            template.lighting.main_light_beta = lighting["beta"] 
            template.lighting.main_light_intensity = lighting["intensity"]
            template.lighting.main_light_color_r = lighting["r"]
            template.lighting.main_light_color_g = lighting["g"]
            template.lighting.main_light_color_b = lighting["b"]
            
            params.add_keyframe(time, template, InterpolationType.EASE_IN_OUT)
        
        return params
    
    @staticmethod
    def dramatic_zoom_sequence(duration: float = 6.0) -> AnimationParameters:
        """
        Create a dramatic zoom sequence with parameter changes.
        
        Args:
            duration: Animation duration in seconds
            
        Returns:
            Animation parameters for dramatic zoom
        """
        params = AnimationParameters()
        params.duration_seconds = duration
        params.fps = 60  # Higher fps for smooth zoom
        
        # Aggressive zoom path
        params.camera_path = CameraPath(
            path_type="zoom",
            orbit_radius=10.0,
            zoom_factor=0.95,  # Very aggressive zoom
            focus_point=(0.2, 0.1, 0.0)  # Slightly off-center
        )
        
        # Start with low detail, ramp up dramatically
        start_template = ParameterTemplates.classic_mandelbulb()
        start_template.fractal.iterations = 100
        start_template.render.detail_level = 0.5
        
        mid_template = start_template.copy()
        mid_template.fractal.iterations = 250
        mid_template.render.detail_level = 1.2
        mid_template.fractal.power = 10.0
        
        end_template = mid_template.copy()
        end_template.fractal.iterations = 400
        end_template.render.detail_level = 2.0
        end_template.fractal.power = 12.0
        
        # Color intensification
        end_template.material.surface_color_r = 1.0
        end_template.material.surface_color_g = 0.2
        end_template.material.surface_color_b = 0.1
        end_template.lighting.main_light_intensity = 2.0
        
        params.add_keyframe(0.0, start_template, InterpolationType.EASE_IN_OUT)
        params.add_keyframe(0.6, mid_template, InterpolationType.EASE_IN_OUT)
        params.add_keyframe(1.0, end_template, InterpolationType.CUBIC)
        
        return params
    
    @staticmethod
    def get_all_templates() -> Dict[str, AnimationParameters]:
        """
        Get all available animation templates.
        
        Returns:
            Dictionary mapping template names to animation parameters
        """
        return {
            'orbital_mandelbulb': AnimationTemplates.orbital_mandelbulb(),
            'zoom_into_fractal': AnimationTemplates.zoom_into_fractal(),
            'color_morph_sequence': AnimationTemplates.color_morph_sequence(),
            'power_evolution_sequence': AnimationTemplates.power_evolution_sequence(),
            'julia_transformation_sequence': AnimationTemplates.julia_transformation_sequence(),
            'multi_formula_evolution': AnimationTemplates.multi_formula_evolution(),
            'lighting_showcase': AnimationTemplates.lighting_showcase(),
            'dramatic_zoom_sequence': AnimationTemplates.dramatic_zoom_sequence(),
        }
    
    @staticmethod
    def create_random_variation(base_template: AnimationParameters, 
                               variation_strength: float = 0.3) -> AnimationParameters:
        """
        Create a random variation of an existing template.
        
        Args:
            base_template: Template to create variation from
            variation_strength: Strength of variation (0.0 = no change, 1.0 = major change)
            
        Returns:
            Varied animation parameters
        """
        variation = base_template.copy()
        
        # Vary duration
        if random.random() < variation_strength:
            variation.duration_seconds *= random.uniform(0.7, 1.3)
        
        # Vary camera path parameters
        if random.random() < variation_strength:
            variation.camera_path.orbit_radius *= random.uniform(0.6, 1.4)
            variation.camera_path.orbit_speed *= random.uniform(0.5, 1.5)
            variation.camera_path.orbit_height += random.uniform(-1, 1) * variation_strength
        
        # Vary keyframe parameters
        for keyframe in variation.keyframes:
            if random.random() < variation_strength * 0.7:
                keyframe.parameters.mutate(
                    mutation_rate=variation_strength * 0.5,
                    mutation_strength=variation_strength * 0.3
                )
        
        return variation
    
    @staticmethod
    def create_seed_population(population_size: int = 10) -> List[AnimationParameters]:
        """
        Create a diverse seed population for evolution.
        
        Args:
            population_size: Number of individuals in population
            
        Returns:
            List of animation parameters for initial population
        """
        templates = list(AnimationTemplates.get_all_templates().values())
        population = []
        
        # Include all base templates
        for template in templates:
            if len(population) < population_size:
                population.append(template)
        
        # Fill remaining spots with variations
        while len(population) < population_size:
            base = random.choice(templates)
            variation = AnimationTemplates.create_random_variation(
                base,
                variation_strength=random.uniform(0.2, 0.5)
            )
            population.append(variation)
        
        return population[:population_size]
    
    @staticmethod
    def create_preview_template() -> AnimationParameters:
        """
        Create a template optimized for fast preview generation.
        
        Returns:
            Animation parameters optimized for speed
        """
        params = AnimationTemplates.orbital_mandelbulb(duration=3.0)
        params.fps = 15  # Lower fps
        
        # Optimize all keyframes for speed
        for keyframe in params.keyframes:
            keyframe.parameters.render.image_width = 400
            keyframe.parameters.render.image_height = 300
            keyframe.parameters.render.detail_level = 0.5
            keyframe.parameters.render.quality = 0.7
            keyframe.parameters.fractal.iterations = 80
        
        return params
    
    @staticmethod
    def create_high_quality_template() -> AnimationParameters:
        """
        Create a template optimized for high quality output.
        
        Returns:
            Animation parameters optimized for quality
        """
        params = AnimationTemplates.power_evolution_sequence(duration=12.0)
        params.fps = 60
        params.render_width = 2560
        params.render_height = 1440
        params.render_quality = 1.5
        
        # Enhance all keyframes for quality
        for keyframe in params.keyframes:
            keyframe.parameters.render.image_width = 2560
            keyframe.parameters.render.image_height = 1440
            keyframe.parameters.render.detail_level = 1.5
            keyframe.parameters.render.quality = 1.2
            keyframe.parameters.fractal.iterations = min(500, keyframe.parameters.fractal.iterations * 2)
        
        return params