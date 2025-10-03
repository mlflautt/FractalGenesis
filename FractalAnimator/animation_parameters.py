"""
Animation Parameters for Fractal Evolution

This module defines the parameter structures for creating animated fractals,
including keyframe interpolation, camera paths, and time-based parameter evolution.
Designed for evolutionary optimization of fractal animations.
"""

import copy
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable, Any
from enum import Enum
import numpy as np

from renderers.mandelbulber.parameters import MandelbulberParameters


class InterpolationType(Enum):
    """Types of interpolation between keyframes"""
    LINEAR = "linear"
    CUBIC = "cubic"
    HERMITE = "hermite"
    CATMULL_ROM = "catmull_rom"
    EASE_IN_OUT = "ease_in_out"
    CIRCULAR = "circular"
    SPIRAL = "spiral"


@dataclass
class Keyframe:
    """A keyframe containing fractal parameters at a specific time"""
    time: float  # Normalized time from 0.0 to 1.0
    parameters: MandelbulberParameters
    interpolation_type: InterpolationType = InterpolationType.LINEAR
    
    def copy(self) -> 'Keyframe':
        """Create a deep copy of the keyframe"""
        return Keyframe(
            time=self.time,
            parameters=copy.deepcopy(self.parameters),
            interpolation_type=self.interpolation_type
        )


@dataclass
class CameraPath:
    """Defines camera movement path for animation"""
    path_type: str = "orbit"  # orbit, linear, spiral, zoom, custom
    orbit_radius: float = 5.0
    orbit_height: float = 0.0
    orbit_speed: float = 1.0  # Revolutions per animation
    zoom_factor: float = 1.0
    focus_point: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    custom_points: List[Tuple[float, float, float]] = field(default_factory=list)
    
    def calculate_camera_position(self, time: float) -> Tuple[float, float, float]:
        """Calculate camera position at given time (0.0 to 1.0)"""
        if self.path_type == "orbit":
            angle = time * 2 * math.pi * self.orbit_speed
            x = self.focus_point[0] + self.orbit_radius * math.cos(angle)
            y = self.focus_point[1] + self.orbit_height * math.sin(time * 2 * math.pi)
            z = self.focus_point[2] + self.orbit_radius * math.sin(angle)
            return (x, y, z)
            
        elif self.path_type == "linear":
            if len(self.custom_points) >= 2:
                start = self.custom_points[0]
                end = self.custom_points[1]
                return (
                    start[0] + time * (end[0] - start[0]),
                    start[1] + time * (end[1] - start[1]),
                    start[2] + time * (end[2] - start[2])
                )
        
        elif self.path_type == "spiral":
            angle = time * 2 * math.pi * self.orbit_speed
            radius = self.orbit_radius * (1 - time * 0.8)  # Spiral inward
            x = self.focus_point[0] + radius * math.cos(angle)
            y = self.focus_point[1] + self.orbit_height * time
            z = self.focus_point[2] + radius * math.sin(angle)
            return (x, y, z)
            
        elif self.path_type == "zoom":
            # Zoom into focus point
            distance = self.orbit_radius * (1 - time * self.zoom_factor)
            return (
                self.focus_point[0],
                self.focus_point[1],
                self.focus_point[2] - distance
            )
        
        # Default: static position
        return self.focus_point


@dataclass 
class AnimationParameters:
    """Complete parameters for a fractal animation"""
    
    # Basic animation properties
    duration_seconds: float = 10.0
    fps: int = 30
    keyframes: List[Keyframe] = field(default_factory=list)
    
    # Camera animation
    camera_path: CameraPath = field(default_factory=CameraPath)
    camera_target_follows_path: bool = False
    
    # Parameter evolution over time
    parameter_curves: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)
    
    # Evolution fitness weights
    fitness_weights: Dict[str, float] = field(default_factory=lambda: {
        'visual_continuity': 0.3,
        'motion_smoothness': 0.2,
        'aesthetic_appeal': 0.2,
        'complexity_variation': 0.15,
        'color_harmony': 0.15
    })
    
    # Render settings
    render_width: int = 1920
    render_height: int = 1080
    render_quality: float = 1.0
    
    def __post_init__(self):
        """Initialize with default keyframes if none provided"""
        if not self.keyframes:
            self._create_default_keyframes()
    
    def _create_default_keyframes(self):
        """Create default start and end keyframes"""
        from renderers.mandelbulber.templates import ParameterTemplates
        
        start_params = ParameterTemplates.classic_mandelbulb()
        end_params = ParameterTemplates.golden_mandelbulb()
        
        self.keyframes = [
            Keyframe(time=0.0, parameters=start_params),
            Keyframe(time=1.0, parameters=end_params)
        ]
    
    @property
    def total_frames(self) -> int:
        """Calculate total number of frames"""
        return int(self.duration_seconds * self.fps)
    
    def add_keyframe(self, time: float, parameters: MandelbulberParameters, 
                    interpolation: InterpolationType = InterpolationType.LINEAR):
        """Add a keyframe at the specified time"""
        keyframe = Keyframe(time=time, parameters=parameters, interpolation_type=interpolation)
        self.keyframes.append(keyframe)
        self.keyframes.sort(key=lambda k: k.time)
    
    def remove_keyframe(self, index: int):
        """Remove keyframe at index"""
        if 0 <= index < len(self.keyframes):
            self.keyframes.pop(index)
    
    def get_parameters_at_time(self, time: float) -> MandelbulberParameters:
        """
        Get interpolated parameters at specific time.
        
        Args:
            time: Time from 0.0 to 1.0
            
        Returns:
            Interpolated MandelbulberParameters
        """
        time = max(0.0, min(1.0, time))  # Clamp to valid range
        
        # Find surrounding keyframes
        before_kf = None
        after_kf = None
        
        for kf in self.keyframes:
            if kf.time <= time:
                before_kf = kf
            elif kf.time > time and after_kf is None:
                after_kf = kf
                break
        
        # Handle edge cases
        if before_kf is None:
            return self.keyframes[0].parameters.copy() if self.keyframes else MandelbulberParameters()
        if after_kf is None:
            return before_kf.parameters.copy()
        if before_kf.time == after_kf.time:
            return before_kf.parameters.copy()
        
        # Calculate interpolation factor
        alpha = (time - before_kf.time) / (after_kf.time - before_kf.time)
        
        # Apply interpolation curve
        alpha = self._apply_interpolation_curve(alpha, before_kf.interpolation_type)
        
        # Interpolate parameters
        result = self._interpolate_parameters(before_kf.parameters, after_kf.parameters, alpha)
        
        # Apply camera path if enabled
        if self.camera_path:
            camera_pos = self.camera_path.calculate_camera_position(time)
            result.camera.camera_x = camera_pos[0]
            result.camera.camera_y = camera_pos[1] 
            result.camera.camera_z = camera_pos[2]
            
            if not self.camera_target_follows_path:
                result.camera.target_x = self.camera_path.focus_point[0]
                result.camera.target_y = self.camera_path.focus_point[1]
                result.camera.target_z = self.camera_path.focus_point[2]
        
        # Apply parameter curves
        for param_name, curve_points in self.parameter_curves.items():
            if len(curve_points) >= 2:
                value = self._interpolate_curve(time, curve_points)
                self._set_parameter_by_name(result, param_name, value)
        
        return result
    
    def _apply_interpolation_curve(self, alpha: float, interp_type: InterpolationType) -> float:
        """Apply interpolation curve to alpha value"""
        if interp_type == InterpolationType.LINEAR:
            return alpha
        elif interp_type == InterpolationType.EASE_IN_OUT:
            return alpha * alpha * (3.0 - 2.0 * alpha)
        elif interp_type == InterpolationType.CUBIC:
            return alpha * alpha * alpha
        elif interp_type == InterpolationType.CIRCULAR:
            return 1.0 - math.sqrt(1.0 - alpha * alpha)
        else:
            return alpha  # Default to linear
    
    def _interpolate_parameters(self, p1: MandelbulberParameters, p2: MandelbulberParameters, 
                              alpha: float) -> MandelbulberParameters:
        """Interpolate between two parameter sets"""
        result = p1.copy()
        
        # Interpolate camera parameters
        result.camera.camera_x = self._lerp(p1.camera.camera_x, p2.camera.camera_x, alpha)
        result.camera.camera_y = self._lerp(p1.camera.camera_y, p2.camera.camera_y, alpha)
        result.camera.camera_z = self._lerp(p1.camera.camera_z, p2.camera.camera_z, alpha)
        result.camera.target_x = self._lerp(p1.camera.target_x, p2.camera.target_x, alpha)
        result.camera.target_y = self._lerp(p1.camera.target_y, p2.camera.target_y, alpha)
        result.camera.target_z = self._lerp(p1.camera.target_z, p2.camera.target_z, alpha)
        result.camera.fov = self._lerp(p1.camera.fov, p2.camera.fov, alpha)
        
        # Interpolate fractal parameters
        result.fractal.power = self._lerp(p1.fractal.power, p2.fractal.power, alpha)
        result.fractal.bailout = self._lerp(p1.fractal.bailout, p2.fractal.bailout, alpha)
        # Note: iterations should probably be integer, so we round
        result.fractal.iterations = int(self._lerp(p1.fractal.iterations, p2.fractal.iterations, alpha))
        result.fractal.julia_c_x = self._lerp(p1.fractal.julia_c_x, p2.fractal.julia_c_x, alpha)
        result.fractal.julia_c_y = self._lerp(p1.fractal.julia_c_y, p2.fractal.julia_c_y, alpha)
        result.fractal.julia_c_z = self._lerp(p1.fractal.julia_c_z, p2.fractal.julia_c_z, alpha)
        
        # Interpolate material parameters
        result.material.surface_color_r = self._lerp(p1.material.surface_color_r, p2.material.surface_color_r, alpha)
        result.material.surface_color_g = self._lerp(p1.material.surface_color_g, p2.material.surface_color_g, alpha)
        result.material.surface_color_b = self._lerp(p1.material.surface_color_b, p2.material.surface_color_b, alpha)
        result.material.specular_r = self._lerp(p1.material.specular_r, p2.material.specular_r, alpha)
        result.material.specular_g = self._lerp(p1.material.specular_g, p2.material.specular_g, alpha)
        result.material.specular_b = self._lerp(p1.material.specular_b, p2.material.specular_b, alpha)
        result.material.specular_width = self._lerp(p1.material.specular_width, p2.material.specular_width, alpha)
        result.material.roughness = self._lerp(p1.material.roughness, p2.material.roughness, alpha)
        result.material.reflectance = self._lerp(p1.material.reflectance, p2.material.reflectance, alpha)
        result.material.transparency = self._lerp(p1.material.transparency, p2.material.transparency, alpha)
        
        # Interpolate lighting parameters
        result.lighting.main_light_alpha = self._lerp(p1.lighting.main_light_alpha, p2.lighting.main_light_alpha, alpha)
        result.lighting.main_light_beta = self._lerp(p1.lighting.main_light_beta, p2.lighting.main_light_beta, alpha)
        result.lighting.main_light_intensity = self._lerp(p1.lighting.main_light_intensity, p2.lighting.main_light_intensity, alpha)
        result.lighting.main_light_color_r = self._lerp(p1.lighting.main_light_color_r, p2.lighting.main_light_color_r, alpha)
        result.lighting.main_light_color_g = self._lerp(p1.lighting.main_light_color_g, p2.lighting.main_light_color_g, alpha)
        result.lighting.main_light_color_b = self._lerp(p1.lighting.main_light_color_b, p2.lighting.main_light_color_b, alpha)
        result.lighting.ambient_light_intensity = self._lerp(p1.lighting.ambient_light_intensity, p2.lighting.ambient_light_intensity, alpha)
        
        return result
    
    def _lerp(self, a: float, b: float, alpha: float) -> float:
        """Linear interpolation between two values"""
        return a + alpha * (b - a)
    
    def _interpolate_curve(self, time: float, curve_points: List[Tuple[float, float]]) -> float:
        """Interpolate value from curve points"""
        curve_points = sorted(curve_points, key=lambda p: p[0])
        
        for i in range(len(curve_points) - 1):
            t1, v1 = curve_points[i]
            t2, v2 = curve_points[i + 1]
            
            if t1 <= time <= t2:
                if t1 == t2:
                    return v1
                alpha = (time - t1) / (t2 - t1)
                return self._lerp(v1, v2, alpha)
        
        # Outside curve range - return nearest value
        if time < curve_points[0][0]:
            return curve_points[0][1]
        else:
            return curve_points[-1][1]
    
    def _set_parameter_by_name(self, params: MandelbulberParameters, name: str, value: float):
        """Set parameter value by string name"""
        if name == "fractal.power":
            params.fractal.power = value
        elif name == "fractal.bailout":
            params.fractal.bailout = value
        elif name == "material.surface_color_r":
            params.material.surface_color_r = value
        elif name == "material.surface_color_g":
            params.material.surface_color_g = value
        elif name == "material.surface_color_b":
            params.material.surface_color_b = value
        elif name == "lighting.main_light_intensity":
            params.lighting.main_light_intensity = value
        # Add more parameter paths as needed
    
    def mutate(self, mutation_rate: float = 0.1, mutation_strength: float = 0.1):
        """
        Apply genetic mutations to animation parameters.
        
        Args:
            mutation_rate: Probability of mutating each parameter
            mutation_strength: Relative strength of mutations
        """
        # Mutate animation duration
        if random.random() < mutation_rate:
            self.duration_seconds *= random.uniform(0.8, 1.2)
            self.duration_seconds = max(1.0, min(60.0, self.duration_seconds))
        
        # Mutate camera path
        if random.random() < mutation_rate:
            self.camera_path.orbit_radius *= random.uniform(0.7, 1.3)
            self.camera_path.orbit_speed *= random.uniform(0.5, 2.0)
            self.camera_path.orbit_height += random.gauss(0, mutation_strength * 2)
        
        # Mutate keyframes
        for keyframe in self.keyframes:
            if random.random() < mutation_rate:
                keyframe.parameters.mutate(mutation_rate * 0.5, mutation_strength)
        
        # Add or remove keyframes occasionally
        if random.random() < mutation_rate * 0.3:
            if len(self.keyframes) < 8:  # Max 8 keyframes
                self._add_random_keyframe()
        
        if random.random() < mutation_rate * 0.2:
            if len(self.keyframes) > 2:  # Min 2 keyframes
                self._remove_random_keyframe()
    
    def _add_random_keyframe(self):
        """Add a random keyframe at a random time"""
        from ..renderers.mandelbulber.templates import ParameterTemplates
        
        time = random.uniform(0.1, 0.9)
        # Create parameters by interpolating between existing keyframes
        params = self.get_parameters_at_time(time)
        params.mutate(mutation_rate=0.5, mutation_strength=0.2)
        
        self.add_keyframe(time, params, random.choice(list(InterpolationType)))
    
    def _remove_random_keyframe(self):
        """Remove a random keyframe (but keep first and last)"""
        if len(self.keyframes) > 2:
            # Don't remove first or last keyframe
            removable_indices = list(range(1, len(self.keyframes) - 1))
            if removable_indices:
                self.remove_keyframe(random.choice(removable_indices))
    
    def crossover(self, other: 'AnimationParameters') -> 'AnimationParameters':
        """
        Create new animation by crossing over with another animation.
        
        Args:
            other: Another animation to crossover with
            
        Returns:
            New animation combining features from both parents
        """
        child = AnimationParameters()
        
        # Basic properties crossover
        child.duration_seconds = random.choice([self.duration_seconds, other.duration_seconds])
        child.fps = random.choice([self.fps, other.fps])
        
        # Camera path crossover
        if random.random() < 0.5:
            child.camera_path = copy.deepcopy(self.camera_path)
        else:
            child.camera_path = copy.deepcopy(other.camera_path)
        
        # Keyframes crossover - blend keyframes from both parents
        all_keyframes = []
        
        # Add keyframes from both parents with some probability
        for kf in self.keyframes:
            if random.random() < 0.7:  # 70% chance to inherit keyframe
                all_keyframes.append(kf.copy())
        
        for kf in other.keyframes:
            if random.random() < 0.7:
                # Check if time is not too close to existing keyframe
                too_close = any(abs(existing.time - kf.time) < 0.1 for existing in all_keyframes)
                if not too_close:
                    all_keyframes.append(kf.copy())
        
        # Sort and limit keyframes
        all_keyframes.sort(key=lambda k: k.time)
        child.keyframes = all_keyframes[:8]  # Max 8 keyframes
        
        # Ensure we have at least start and end keyframes
        if not any(kf.time == 0.0 for kf in child.keyframes):
            child.keyframes.insert(0, self.keyframes[0].copy())
        if not any(kf.time == 1.0 for kf in child.keyframes):
            child.keyframes.append(self.keyframes[-1].copy())
        
        # Fitness weights blend
        for key in self.fitness_weights:
            if key in other.fitness_weights:
                child.fitness_weights[key] = (self.fitness_weights[key] + other.fitness_weights[key]) / 2
        
        return child
    
    def copy(self) -> 'AnimationParameters':
        """Create a deep copy of the animation parameters"""
        return copy.deepcopy(self)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert animation parameters to dictionary for serialization"""
        return {
            'duration_seconds': self.duration_seconds,
            'fps': self.fps,
            'keyframes': [
                {
                    'time': kf.time,
                    'parameters': kf.parameters.to_fract_file(),  # Serialize as .fract content
                    'interpolation_type': kf.interpolation_type.value
                } for kf in self.keyframes
            ],
            'camera_path': {
                'path_type': self.camera_path.path_type,
                'orbit_radius': self.camera_path.orbit_radius,
                'orbit_height': self.camera_path.orbit_height,
                'orbit_speed': self.camera_path.orbit_speed,
                'zoom_factor': self.camera_path.zoom_factor,
                'focus_point': self.camera_path.focus_point,
                'custom_points': self.camera_path.custom_points
            },
            'parameter_curves': self.parameter_curves,
            'fitness_weights': self.fitness_weights,
            'render_width': self.render_width,
            'render_height': self.render_height,
            'render_quality': self.render_quality
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnimationParameters':
        """Create animation parameters from dictionary"""
        # This would need implementation to parse .fract files back to parameters
        # For now, we'll create a basic implementation
        anim = cls()
        anim.duration_seconds = data.get('duration_seconds', 10.0)
        anim.fps = data.get('fps', 30)
        # TODO: Implement full deserialization
        return anim