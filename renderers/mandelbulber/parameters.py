"""
Mandelbulber Parameters Management

This module handles the complex parameter structure for Mandelbulber fractals,
including camera settings, fractal formulas, materials, lighting, and hybrid configurations.
Based on Mandelbulber manual v0.9.1 specifications.
"""

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import random
import numpy as np


@dataclass
class CameraParameters:
    """Camera and view parameters"""
    camera_x: float = 0.0
    camera_y: float = 0.0  
    camera_z: float = -3.0
    target_x: float = 0.0
    target_y: float = 0.0
    target_z: float = 0.0
    camera_up_x: float = 0.0
    camera_up_y: float = 1.0
    camera_up_z: float = 0.0
    flight_rotation_x: float = 0.0
    flight_rotation_y: float = 0.0
    flight_rotation_z: float = 0.0
    fov: float = 1.0
    perspective_type: int = 0  # 0=three_point, 1=fish_eye, 2=equirectangular


@dataclass
class RenderParameters:
    """Rendering quality and algorithm parameters"""
    image_width: int = 800
    image_height: int = 600
    detail_level: float = 1.0
    raymarching_step_multiplier: float = 1.0
    de_threshold: float = 0.01
    quality: float = 1.0
    frames_per_keyframe: int = 30
    ssao_enabled: bool = False
    hdr_enabled: bool = False
    

@dataclass
class FractalParameters:
    """Core fractal formula parameters"""
    formula_name: str = "mandelbulb"  # mandelbulb, mandelbox, menger_sponge, etc.
    power: float = 8.0
    bailout: float = 10.0
    iterations: int = 250
    julia_mode: bool = False
    julia_c_x: float = 0.0
    julia_c_y: float = 0.0
    julia_c_z: float = 0.0
    # Transform parameters (varies by formula)
    transforms: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MaterialParameters:
    """Surface material and coloring parameters"""
    surface_color_r: float = 0.5
    surface_color_g: float = 0.5  
    surface_color_b: float = 0.5
    specular_r: float = 1.0
    specular_g: float = 1.0
    specular_b: float = 1.0
    specular_width: float = 1.0
    roughness: float = 0.1
    reflectance: float = 0.0
    transparency: float = 0.0
    luminosity_r: float = 0.0
    luminosity_g: float = 0.0
    luminosity_b: float = 0.0
    transparency_interior_color_r: float = 1.0
    transparency_interior_color_g: float = 1.0
    transparency_interior_color_b: float = 1.0


@dataclass
class LightingParameters:
    """Lighting setup parameters"""
    main_light_alpha: float = -45.0
    main_light_beta: float = 45.0
    main_light_intensity: float = 1.0
    main_light_visibility: float = 1.0
    main_light_color_r: float = 1.0
    main_light_color_g: float = 1.0
    main_light_color_b: float = 1.0
    ambient_light_intensity: float = 1.0
    shadows_enabled: bool = True
    penetrating_lights: bool = False


class MandelbulberParameters:
    """
    Complete Mandelbulber parameter set for generating .fract files.
    
    This class manages all parameters needed to define a complete fractal scene
    including camera position, fractal formulas, materials, and lighting.
    """
    
    def __init__(self):
        self.camera = CameraParameters()
        self.render = RenderParameters()
        self.fractal = FractalParameters()
        self.material = MaterialParameters()
        self.lighting = LightingParameters()
        self.hybrid_fractals: List[FractalParameters] = []
        self.custom_parameters: Dict[str, Any] = {}
    
    def to_fract_file(self) -> str:
        """
        Generate a complete .fract file content string.
        
        Returns:
            str: Complete .fract file content
        """
        lines = []
        lines.append("# Mandelbulber settings file")
        lines.append("# version 2.32")
        lines.append("")
        
        # Camera parameters
        lines.append("[camera]")
        lines.append(f"camera {self.camera.camera_x} {self.camera.camera_y} {self.camera.camera_z};")
        lines.append(f"target {self.camera.target_x} {self.camera.target_y} {self.camera.target_z};")
        lines.append(f"camera_up {self.camera.camera_up_x} {self.camera.camera_up_y} {self.camera.camera_up_z};")
        lines.append(f"flight_rotation {self.camera.flight_rotation_x} {self.camera.flight_rotation_y} {self.camera.flight_rotation_z};")
        lines.append(f"fov {self.camera.fov};")
        lines.append(f"perspective_type {self.camera.perspective_type};")
        lines.append("")
        
        # Image parameters
        lines.append("[image]")
        lines.append(f"image_width {self.render.image_width};")
        lines.append(f"image_height {self.render.image_height};")
        lines.append("")
        
        # Fractal parameters
        lines.append("[fractal]")
        lines.append(f"formula_1 {self.fractal.formula_name};")
        lines.append(f"power_1 {self.fractal.power};")
        lines.append(f"bailout_1 {self.fractal.bailout};")
        lines.append(f"iterations_1 {self.fractal.iterations};")
        
        if self.fractal.julia_mode:
            lines.append(f"julia_mode_1 true;")
            lines.append(f"julia_c_1 {self.fractal.julia_c_x} {self.fractal.julia_c_y} {self.fractal.julia_c_z};")
        
        # Add hybrid fractals if any
        for i, hybrid in enumerate(self.hybrid_fractals, start=2):
            lines.append(f"formula_{i} {hybrid.formula_name};")
            lines.append(f"power_{i} {hybrid.power};")
            lines.append(f"bailout_{i} {hybrid.bailout};")
            lines.append(f"iterations_{i} {hybrid.iterations};")
        
        lines.append("")
        
        # Rendering parameters
        lines.append("[rendering]")
        lines.append(f"detail_level {self.render.detail_level};")
        lines.append(f"raymarching_step_multiplier {self.render.raymarching_step_multiplier};")
        lines.append(f"DE_threshold {self.render.de_threshold};")
        lines.append(f"quality {self.render.quality};")
        lines.append(f"SSAO_enabled {str(self.render.ssao_enabled).lower()};")
        lines.append("")
        
        # Material parameters
        lines.append("[material]")
        lines.append(f"surface_color_1 {self.material.surface_color_r} {self.material.surface_color_g} {self.material.surface_color_b};")
        lines.append(f"specular_1 {self.material.specular_r} {self.material.specular_g} {self.material.specular_b};")
        lines.append(f"specular_width_1 {self.material.specular_width};")
        lines.append(f"roughness_1 {self.material.roughness};")
        lines.append(f"reflectance_1 {self.material.reflectance};")
        lines.append(f"transparency_of_surface_1 {self.material.transparency};")
        lines.append("")
        
        # Lighting parameters
        lines.append("[lights]")
        lines.append(f"main_light_alpha {self.lighting.main_light_alpha};")
        lines.append(f"main_light_beta {self.lighting.main_light_beta};")
        lines.append(f"main_light_intensity {self.lighting.main_light_intensity};")
        lines.append(f"main_light_colour {self.lighting.main_light_color_r} {self.lighting.main_light_color_g} {self.lighting.main_light_color_b};")
        lines.append(f"ambient_light_intensity {self.lighting.ambient_light_intensity};")
        lines.append(f"shadows_enabled {str(self.lighting.shadows_enabled).lower()};")
        lines.append("")
        
        # Custom parameters
        if self.custom_parameters:
            lines.append("[custom]")
            for key, value in self.custom_parameters.items():
                lines.append(f"{key} {value};")
            lines.append("")
        
        return "\n".join(lines)
    
    def mutate(self, mutation_rate: float = 0.1, mutation_strength: float = 0.1):
        """
        Apply random mutations to parameters for genetic algorithm.
        
        Args:
            mutation_rate: Probability of mutating each parameter
            mutation_strength: Relative strength of mutations
        """
        if random.random() < mutation_rate:
            # Mutate camera position
            self.camera.camera_x += random.gauss(0, mutation_strength)
            self.camera.camera_y += random.gauss(0, mutation_strength)
            self.camera.camera_z += random.gauss(0, mutation_strength)
            
        if random.random() < mutation_rate:
            # Mutate target position
            self.camera.target_x += random.gauss(0, mutation_strength)
            self.camera.target_y += random.gauss(0, mutation_strength)
            self.camera.target_z += random.gauss(0, mutation_strength)
            
        if random.random() < mutation_rate:
            # Mutate fractal power
            self.fractal.power += random.gauss(0, mutation_strength * 2)
            self.fractal.power = max(0.1, min(20.0, self.fractal.power))
            
        if random.random() < mutation_rate:
            # Mutate colors
            self.material.surface_color_r = max(0, min(1, self.material.surface_color_r + random.gauss(0, mutation_strength)))
            self.material.surface_color_g = max(0, min(1, self.material.surface_color_g + random.gauss(0, mutation_strength)))  
            self.material.surface_color_b = max(0, min(1, self.material.surface_color_b + random.gauss(0, mutation_strength)))
            
        if random.random() < mutation_rate:
            # Mutate lighting
            self.lighting.main_light_alpha += random.gauss(0, mutation_strength * 30)
            self.lighting.main_light_beta += random.gauss(0, mutation_strength * 30)
            self.lighting.main_light_alpha = max(-180, min(180, self.lighting.main_light_alpha))
            self.lighting.main_light_beta = max(-90, min(90, self.lighting.main_light_beta))
    
    def crossover(self, other: 'MandelbulberParameters') -> 'MandelbulberParameters':
        """
        Create a new parameter set by crossing over with another set.
        
        Args:
            other: Another parameter set to crossover with
            
        Returns:
            New parameter set combining features from both parents
        """
        child = MandelbulberParameters()
        
        # Camera crossover
        if random.random() < 0.5:
            child.camera = copy.deepcopy(self.camera)
        else:
            child.camera = copy.deepcopy(other.camera)
            
        # Fractal crossover
        child.fractal.formula_name = random.choice([self.fractal.formula_name, other.fractal.formula_name])
        child.fractal.power = random.choice([self.fractal.power, other.fractal.power])
        child.fractal.bailout = random.choice([self.fractal.bailout, other.fractal.bailout])
        child.fractal.iterations = random.choice([self.fractal.iterations, other.fractal.iterations])
        
        # Material crossover - blend colors
        child.material.surface_color_r = (self.material.surface_color_r + other.material.surface_color_r) / 2
        child.material.surface_color_g = (self.material.surface_color_g + other.material.surface_color_g) / 2
        child.material.surface_color_b = (self.material.surface_color_b + other.material.surface_color_b) / 2
        
        # Lighting crossover
        if random.random() < 0.5:
            child.lighting = copy.deepcopy(self.lighting)
        else:
            child.lighting = copy.deepcopy(other.lighting)
            
        return child
    
    def copy(self) -> 'MandelbulberParameters':
        """Create a deep copy of this parameter set"""
        return copy.deepcopy(self)
    
    def randomize(self):
        """Generate completely random parameters"""
        # Random camera position
        self.camera.camera_x = random.uniform(-5, 5)
        self.camera.camera_y = random.uniform(-5, 5)
        self.camera.camera_z = random.uniform(-5, 5)
        
        # Random target
        self.camera.target_x = random.uniform(-1, 1)
        self.camera.target_y = random.uniform(-1, 1)
        self.camera.target_z = random.uniform(-1, 1)
        
        # Random fractal parameters
        self.fractal.formula_name = random.choice(['mandelbulb', 'mandelbox', 'menger_sponge'])
        self.fractal.power = random.uniform(2, 12)
        self.fractal.bailout = random.uniform(5, 20)
        self.fractal.iterations = random.randint(50, 300)
        
        # Random colors
        self.material.surface_color_r = random.random()
        self.material.surface_color_g = random.random()
        self.material.surface_color_b = random.random()
        
        # Random lighting
        self.lighting.main_light_alpha = random.uniform(-180, 180)
        self.lighting.main_light_beta = random.uniform(-90, 90)
        self.lighting.main_light_intensity = random.uniform(0.5, 2.0)
