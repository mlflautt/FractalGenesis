"""
Mandelbulber Parameter Templates

This module provides pre-configured parameter templates for various fractal types,
making it easy to start with known good configurations for the genetic algorithm.
"""

from .parameters import MandelbulberParameters
from typing import Dict, List


class ParameterTemplates:
    """
    Collection of pre-configured fractal parameter templates.
    
    These templates provide good starting points for different fractal types
    and can be used to seed the initial population of the genetic algorithm.
    """
    
    @staticmethod
    def classic_mandelbulb() -> MandelbulberParameters:
        """Classic Mandelbulb with power 8"""
        params = MandelbulberParameters()
        
        # Camera setup for classic view
        params.camera.camera_x = 0.0
        params.camera.camera_y = 0.0
        params.camera.camera_z = -3.0
        params.camera.target_x = 0.0
        params.camera.target_y = 0.0
        params.camera.target_z = 0.0
        
        # Classic Mandelbulb fractal
        params.fractal.formula_name = "mandelbulb"
        params.fractal.power = 8.0
        params.fractal.bailout = 10.0
        params.fractal.iterations = 250
        
        # Nice blue/purple coloring
        params.material.surface_color_r = 0.3
        params.material.surface_color_g = 0.2
        params.material.surface_color_b = 0.8
        
        # Good lighting setup
        params.lighting.main_light_alpha = -45.0
        params.lighting.main_light_beta = 30.0
        params.lighting.main_light_intensity = 1.2
        
        return params
    
    @staticmethod
    def golden_mandelbulb() -> MandelbulberParameters:
        """Golden colored Mandelbulb with interesting camera angle"""
        params = MandelbulberParameters()
        
        params.camera.camera_x = -2.5
        params.camera.camera_y = -1.0
        params.camera.camera_z = -2.0
        params.camera.target_x = 0.0
        params.camera.target_y = 0.0
        params.camera.target_z = 0.0
        
        params.fractal.formula_name = "mandelbulb"
        params.fractal.power = 9.0
        params.fractal.bailout = 8.0
        params.fractal.iterations = 300
        
        # Golden coloring
        params.material.surface_color_r = 1.0
        params.material.surface_color_g = 0.8
        params.material.surface_color_b = 0.2
        params.material.specular_r = 1.0
        params.material.specular_g = 1.0
        params.material.specular_b = 0.8
        
        params.lighting.main_light_alpha = -60.0
        params.lighting.main_light_beta = 45.0
        params.lighting.main_light_intensity = 1.5
        
        return params
    
    @staticmethod
    def classic_mandelbox() -> MandelbulberParameters:
        """Classic Mandelbox fractal"""
        params = MandelbulberParameters()
        
        params.camera.camera_x = 0.0
        params.camera.camera_y = 0.0
        params.camera.camera_z = -4.0
        
        params.fractal.formula_name = "mandelbox"
        params.fractal.power = 2.0  # Not used for mandelbox, but keep for consistency
        params.fractal.bailout = 100.0
        params.fractal.iterations = 200
        
        # Add mandelbox-specific parameters
        params.custom_parameters = {
            'mandelbox_scale': -2.0,
            'mandelbox_fold': 1.0
        }
        
        # Red/orange coloring
        params.material.surface_color_r = 0.9
        params.material.surface_color_g = 0.3
        params.material.surface_color_b = 0.1
        
        return params
    
    @staticmethod
    def menger_sponge() -> MandelbulberParameters:
        """Menger Sponge fractal"""
        params = MandelbulberParameters()
        
        params.camera.camera_x = 3.0
        params.camera.camera_y = 2.0
        params.camera.camera_z = -3.0
        params.camera.target_x = 0.0
        params.camera.target_y = 0.0  
        params.camera.target_z = 0.0
        
        params.fractal.formula_name = "menger_sponge"
        params.fractal.bailout = 10.0
        params.fractal.iterations = 15  # IFS fractals need fewer iterations
        
        # Green coloring
        params.material.surface_color_r = 0.2
        params.material.surface_color_g = 0.8
        params.material.surface_color_b = 0.3
        
        return params
    
    @staticmethod
    def high_power_mandelbulb() -> MandelbulberParameters:
        """High power Mandelbulb for complex shapes"""
        params = MandelbulberParameters()
        
        params.camera.camera_x = 0.0
        params.camera.camera_y = 0.0
        params.camera.camera_z = -2.5
        
        params.fractal.formula_name = "mandelbulb"
        params.fractal.power = 16.0
        params.fractal.bailout = 15.0
        params.fractal.iterations = 200
        
        # Purple/magenta coloring
        params.material.surface_color_r = 0.8
        params.material.surface_color_g = 0.2
        params.material.surface_color_b = 0.9
        
        params.lighting.main_light_alpha = 0.0
        params.lighting.main_light_beta = 60.0
        
        return params
    
    @staticmethod
    def julia_mandelbulb() -> MandelbulberParameters:
        """Mandelbulb in Julia mode"""
        params = MandelbulberParameters()
        
        params.camera.camera_x = 0.0
        params.camera.camera_y = 0.0
        params.camera.camera_z = -2.0
        
        params.fractal.formula_name = "mandelbulb"
        params.fractal.power = 8.0
        params.fractal.julia_mode = True
        params.fractal.julia_c_x = 0.1
        params.fractal.julia_c_y = 0.2
        params.fractal.julia_c_z = 0.0
        params.fractal.bailout = 10.0
        params.fractal.iterations = 300
        
        # Cyan coloring
        params.material.surface_color_r = 0.0
        params.material.surface_color_g = 0.8
        params.material.surface_color_b = 0.9
        
        return params
    
    @staticmethod
    def get_all_templates() -> Dict[str, MandelbulberParameters]:
        """
        Get all available parameter templates.
        
        Returns:
            Dictionary mapping template names to parameter objects
        """
        return {
            'classic_mandelbulb': ParameterTemplates.classic_mandelbulb(),
            'golden_mandelbulb': ParameterTemplates.golden_mandelbulb(),
            'classic_mandelbox': ParameterTemplates.classic_mandelbox(),
            'menger_sponge': ParameterTemplates.menger_sponge(),
            'high_power_mandelbulb': ParameterTemplates.high_power_mandelbulb(),
            'julia_mandelbulb': ParameterTemplates.julia_mandelbulb(),
        }
    
    @staticmethod
    def get_seed_population(population_size: int = 10) -> List[MandelbulberParameters]:
        """
        Generate a diverse seed population for the genetic algorithm.
        
        Args:
            population_size: Number of individuals in the population
            
        Returns:
            List of parameter sets for initial population
        """
        templates = list(ParameterTemplates.get_all_templates().values())
        population = []
        
        # Start with all templates
        for template in templates:
            if len(population) < population_size:
                population.append(template.copy())
        
        # Fill remaining spots with variations of templates
        while len(population) < population_size:
            base_template = templates[len(population) % len(templates)].copy()
            # Add some variation
            base_template.mutate(mutation_rate=0.3, mutation_strength=0.2)
            population.append(base_template)
        
        return population[:population_size]
    
    @staticmethod
    def create_thumbnail_template() -> MandelbulberParameters:
        """
        Create a template optimized for fast thumbnail generation.
        Lower quality settings for quick preview renders.
        """
        params = ParameterTemplates.classic_mandelbulb()
        
        # Lower quality for speed
        params.render.image_width = 256
        params.render.image_height = 256
        params.render.detail_level = 0.5
        params.render.quality = 0.8
        params.fractal.iterations = 100
        
        return params
