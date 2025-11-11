#!/usr/bin/env python3
"""
GenesisRender - Smooth Animation System
======================================

Advanced animation system with:
- Camera movement with easing functions
- Smooth zoom controls
- Parameter sliding with interpolation
- Keyframe system
- Multiple easing types
- Path-based camera movement
- Synchronized parameter changes
"""

import numpy as np
import math
from typing import List, Dict, Tuple, Callable, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import json

@dataclass 
class CameraKeyframe:
    """Camera position and orientation keyframe"""
    time: float  # 0.0 to 1.0
    position: Tuple[float, float, float]
    target: Tuple[float, float, float] 
    up: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    fov: float = 45.0

@dataclass
class ParameterKeyframe:
    """Fractal parameter keyframe"""
    time: float
    fractal_type: str = "mandelbulb"
    power: float = 8.0
    iterations: int = 100
    bailout: float = 2.0
    julia_c: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    
    # Mandelbox specific
    folding_limit: float = 1.0
    folding_value: float = 2.0
    scale: float = -1.5
    
    # Coloring
    color_palette: str = "warm"
    color_intensity: float = 1.0
    
    # Materials  
    metallic: float = 0.0
    roughness: float = 0.1

@dataclass
class LightingKeyframe:
    """Lighting keyframe"""
    time: float
    light1_pos: Tuple[float, float, float] = (2.0, 2.0, -2.0)
    light1_intensity: float = 1.0
    light1_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    
    light2_pos: Tuple[float, float, float] = (-1.0, 1.0, -1.0)
    light2_intensity: float = 0.5
    light2_color: Tuple[float, float, float] = (0.8, 0.9, 1.0)
    
    ambient: float = 0.15
    ao_strength: float = 0.3

class EasingFunctions:
    """Collection of easing functions for smooth animation"""
    
    @staticmethod
    def linear(t: float) -> float:
        """Linear interpolation (no easing)"""
        return t
    
    @staticmethod
    def ease_in_quad(t: float) -> float:
        """Quadratic ease-in"""
        return t * t
    
    @staticmethod
    def ease_out_quad(t: float) -> float:
        """Quadratic ease-out"""
        return 1.0 - (1.0 - t) * (1.0 - t)
    
    @staticmethod
    def ease_in_out_quad(t: float) -> float:
        """Quadratic ease-in-out"""
        return 2.0 * t * t if t < 0.5 else 1.0 - pow(-2.0 * t + 2.0, 2.0) / 2.0
    
    @staticmethod
    def ease_in_cubic(t: float) -> float:
        """Cubic ease-in"""
        return t * t * t
    
    @staticmethod
    def ease_out_cubic(t: float) -> float:
        """Cubic ease-out"""
        return 1.0 - pow(1.0 - t, 3.0)
    
    @staticmethod
    def ease_in_out_cubic(t: float) -> float:
        """Cubic ease-in-out"""
        return 4.0 * t * t * t if t < 0.5 else 1.0 - pow(-2.0 * t + 2.0, 3.0) / 2.0
    
    @staticmethod
    def ease_in_out_sine(t: float) -> float:
        """Sinusoidal ease-in-out"""
        return -(math.cos(math.pi * t) - 1.0) / 2.0
    
    @staticmethod
    def ease_out_bounce(t: float) -> float:
        """Bounce ease-out"""
        n1 = 7.5625
        d1 = 2.75
        
        if t < 1.0 / d1:
            return n1 * t * t
        elif t < 2.0 / d1:
            t -= 1.5 / d1
            return n1 * t * t + 0.75
        elif t < 2.5 / d1:
            t -= 2.25 / d1
            return n1 * t * t + 0.9375
        else:
            t -= 2.625 / d1
            return n1 * t * t + 0.984375
    
    @staticmethod
    def ease_in_out_elastic(t: float) -> float:
        """Elastic ease-in-out"""
        c5 = (2.0 * math.pi) / 4.5
        
        if t == 0.0:
            return 0.0
        elif t == 1.0:
            return 1.0
        elif t < 0.5:
            return -(pow(2.0, 20.0 * t - 10.0) * math.sin((20.0 * t - 11.125) * c5)) / 2.0
        else:
            return (pow(2.0, -20.0 * t + 10.0) * math.sin((20.0 * t - 11.125) * c5)) / 2.0 + 1.0

class CameraController:
    """Advanced camera animation controller"""
    
    def __init__(self):
        self.keyframes: List[CameraKeyframe] = []
        self.easing_function: Callable[[float], float] = EasingFunctions.ease_in_out_cubic
        
    def add_keyframe(self, keyframe: CameraKeyframe):
        """Add a camera keyframe"""
        self.keyframes.append(keyframe)
        # Keep keyframes sorted by time
        self.keyframes.sort(key=lambda k: k.time)
    
    def set_easing(self, easing_name: str):
        """Set easing function by name"""
        easing_map = {
            'linear': EasingFunctions.linear,
            'ease_in_quad': EasingFunctions.ease_in_quad,
            'ease_out_quad': EasingFunctions.ease_out_quad,
            'ease_in_out_quad': EasingFunctions.ease_in_out_quad,
            'ease_in_cubic': EasingFunctions.ease_in_cubic,
            'ease_out_cubic': EasingFunctions.ease_out_cubic,
            'ease_in_out_cubic': EasingFunctions.ease_in_out_cubic,
            'ease_in_out_sine': EasingFunctions.ease_in_out_sine,
            'ease_out_bounce': EasingFunctions.ease_out_bounce,
            'ease_in_out_elastic': EasingFunctions.ease_in_out_elastic,
        }
        
        if easing_name in easing_map:
            self.easing_function = easing_map[easing_name]
        else:
            print(f"Warning: Unknown easing function '{easing_name}', using cubic")
            self.easing_function = EasingFunctions.ease_in_out_cubic
    
    def interpolate_camera(self, t: float) -> CameraKeyframe:
        """Interpolate camera position at time t (0.0 to 1.0)"""
        if not self.keyframes:
            return CameraKeyframe(0.0, (0.0, 0.0, -3.0), (0.0, 0.0, 0.0))
        
        if len(self.keyframes) == 1:
            return self.keyframes[0]
        
        # Find surrounding keyframes
        if t <= self.keyframes[0].time:
            return self.keyframes[0]
        if t >= self.keyframes[-1].time:
            return self.keyframes[-1]
        
        # Find the two keyframes to interpolate between
        for i in range(len(self.keyframes) - 1):
            if self.keyframes[i].time <= t <= self.keyframes[i + 1].time:
                k1, k2 = self.keyframes[i], self.keyframes[i + 1]
                
                # Calculate local t between these keyframes
                local_t = (t - k1.time) / (k2.time - k1.time)
                eased_t = self.easing_function(local_t)
                
                # Interpolate all parameters
                return CameraKeyframe(
                    time=t,
                    position=self._lerp_tuple(k1.position, k2.position, eased_t),
                    target=self._lerp_tuple(k1.target, k2.target, eased_t),
                    up=self._lerp_tuple(k1.up, k2.up, eased_t),
                    fov=self._lerp_float(k1.fov, k2.fov, eased_t)
                )
        
        return self.keyframes[-1]
    
    def _lerp_float(self, a: float, b: float, t: float) -> float:
        """Linear interpolation for floats"""
        return a + (b - a) * t
    
    def _lerp_tuple(self, a: Tuple[float, float, float], 
                   b: Tuple[float, float, float], t: float) -> Tuple[float, float, float]:
        """Linear interpolation for 3D tuples"""
        return (
            a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            a[2] + (b[2] - a[2]) * t
        )
    
    def create_orbit_animation(self, center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                             radius: float = 3.0, height_variation: float = 1.0,
                             num_keyframes: int = 8) -> None:
        """Create a smooth orbital camera movement"""
        self.keyframes.clear()
        
        for i in range(num_keyframes):
            t = i / (num_keyframes - 1)
            angle = t * 2.0 * math.pi
            
            # Calculate position with smooth height variation
            height = height_variation * math.sin(angle * 2.0)  # Double frequency for figure-8 height
            pos = (
                center[0] + radius * math.cos(angle),
                center[1] + height,
                center[2] + radius * math.sin(angle)
            )
            
            keyframe = CameraKeyframe(
                time=t,
                position=pos,
                target=center,
                up=(0.0, 1.0, 0.0),
                fov=45.0
            )
            self.add_keyframe(keyframe)
    
    def create_zoom_animation(self, start_pos: Tuple[float, float, float],
                            target: Tuple[float, float, float],
                            zoom_factor: float = 10.0) -> None:
        """Create smooth zoom-in animation"""
        self.keyframes.clear()
        
        # Calculate direction vector
        direction = np.array(target) - np.array(start_pos)
        direction = direction / np.linalg.norm(direction)
        
        # Create keyframes moving towards target
        num_keyframes = 5
        for i in range(num_keyframes):
            t = i / (num_keyframes - 1)
            
            # Non-linear zoom - faster at the end
            zoom_t = t * t  # Quadratic zoom
            distance = np.linalg.norm(direction) * (1.0 - zoom_t * (1.0 - 1.0/zoom_factor))
            
            pos = np.array(start_pos) + direction * (1.0 - distance)
            
            keyframe = CameraKeyframe(
                time=t,
                position=tuple(pos),
                target=target,
                fov=45.0 - zoom_t * 20.0  # Reduce FOV for more dramatic zoom
            )
            self.add_keyframe(keyframe)

class ParameterController:
    """Animation controller for fractal parameters"""
    
    def __init__(self):
        self.keyframes: List[ParameterKeyframe] = []
        self.easing_function: Callable[[float], float] = EasingFunctions.ease_in_out_cubic
    
    def add_keyframe(self, keyframe: ParameterKeyframe):
        """Add parameter keyframe"""
        self.keyframes.append(keyframe)
        self.keyframes.sort(key=lambda k: k.time)
    
    def set_easing(self, easing_name: str):
        """Set easing function"""
        # Use same easing map as CameraController
        CameraController.set_easing(self, easing_name)
    
    def interpolate_parameters(self, t: float) -> ParameterKeyframe:
        """Interpolate fractal parameters at time t"""
        if not self.keyframes:
            return ParameterKeyframe(0.0)
        
        if len(self.keyframes) == 1:
            return self.keyframes[0]
        
        # Handle edge cases
        if t <= self.keyframes[0].time:
            return self.keyframes[0]
        if t >= self.keyframes[-1].time:
            return self.keyframes[-1]
        
        # Find surrounding keyframes
        for i in range(len(self.keyframes) - 1):
            if self.keyframes[i].time <= t <= self.keyframes[i + 1].time:
                k1, k2 = self.keyframes[i], self.keyframes[i + 1]
                
                local_t = (t - k1.time) / (k2.time - k1.time)
                eased_t = self.easing_function(local_t)
                
                # Interpolate numeric parameters
                return ParameterKeyframe(
                    time=t,
                    fractal_type=k1.fractal_type if eased_t < 0.5 else k2.fractal_type,
                    power=k1.power + (k2.power - k1.power) * eased_t,
                    iterations=int(k1.iterations + (k2.iterations - k1.iterations) * eased_t),
                    bailout=k1.bailout + (k2.bailout - k1.bailout) * eased_t,
                    julia_c=self._lerp_tuple(k1.julia_c, k2.julia_c, eased_t),
                    folding_limit=k1.folding_limit + (k2.folding_limit - k1.folding_limit) * eased_t,
                    folding_value=k1.folding_value + (k2.folding_value - k1.folding_value) * eased_t,
                    scale=k1.scale + (k2.scale - k1.scale) * eased_t,
                    color_palette=k1.color_palette if eased_t < 0.5 else k2.color_palette,
                    color_intensity=k1.color_intensity + (k2.color_intensity - k1.color_intensity) * eased_t,
                    metallic=k1.metallic + (k2.metallic - k1.metallic) * eased_t,
                    roughness=k1.roughness + (k2.roughness - k1.roughness) * eased_t
                )
        
        return self.keyframes[-1]
    
    def _lerp_tuple(self, a: Tuple[float, float, float], 
                   b: Tuple[float, float, float], t: float) -> Tuple[float, float, float]:
        """Linear interpolation for 3D tuples"""
        return (
            a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            a[2] + (b[2] - a[2]) * t
        )
    
    def create_power_morph(self, start_power: float, end_power: float,
                          fractal_type: str = "mandelbulb") -> None:
        """Create smooth power morphing animation"""
        self.keyframes.clear()
        
        self.add_keyframe(ParameterKeyframe(
            time=0.0, fractal_type=fractal_type, power=start_power,
            color_palette="fire"
        ))
        
        # Add intermediate keyframe for smooth transition
        mid_power = (start_power + end_power) / 2.0
        self.add_keyframe(ParameterKeyframe(
            time=0.5, fractal_type=fractal_type, power=mid_power,
            color_palette="rainbow"
        ))
        
        self.add_keyframe(ParameterKeyframe(
            time=1.0, fractal_type=fractal_type, power=end_power,
            color_palette="ice"
        ))

class AnimationSequence:
    """Complete animation sequence combining camera, parameters, and lighting"""
    
    def __init__(self):
        self.camera_controller = CameraController()
        self.parameter_controller = ParameterController()
        self.lighting_keyframes: List[LightingKeyframe] = []
        self.duration_seconds: float = 10.0
        self.fps: float = 30.0
        
    def set_duration(self, seconds: float, fps: float = 30.0):
        """Set animation duration and framerate"""
        self.duration_seconds = seconds
        self.fps = fps
    
    def get_frame_count(self) -> int:
        """Calculate total frame count"""
        return int(self.duration_seconds * self.fps)
    
    def get_frame_state(self, frame_number: int) -> Tuple[CameraKeyframe, ParameterKeyframe, Optional[LightingKeyframe]]:
        """Get camera, parameters, and lighting state for specific frame"""
        t = frame_number / max(1, self.get_frame_count() - 1)
        t = max(0.0, min(1.0, t))  # Clamp to valid range
        
        camera_state = self.camera_controller.interpolate_camera(t)
        param_state = self.parameter_controller.interpolate_parameters(t)
        
        # Simple lighting interpolation (could be expanded)
        lighting_state = None
        if self.lighting_keyframes:
            if len(self.lighting_keyframes) == 1:
                lighting_state = self.lighting_keyframes[0]
            else:
                # Find closest lighting keyframe (simplified)
                closest_keyframe = min(self.lighting_keyframes, key=lambda k: abs(k.time - t))
                lighting_state = closest_keyframe
        
        return camera_state, param_state, lighting_state
    
    def save_sequence(self, filename: str):
        """Save animation sequence to JSON file"""
        data = {
            'duration_seconds': self.duration_seconds,
            'fps': self.fps,
            'camera_keyframes': [asdict(k) for k in self.camera_controller.keyframes],
            'parameter_keyframes': [asdict(k) for k in self.parameter_controller.keyframes],
            'lighting_keyframes': [asdict(k) for k in self.lighting_keyframes]
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_sequence(self, filename: str):
        """Load animation sequence from JSON file"""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        self.duration_seconds = data['duration_seconds']
        self.fps = data['fps']
        
        # Load camera keyframes
        self.camera_controller.keyframes = [
            CameraKeyframe(**k) for k in data['camera_keyframes']
        ]
        
        # Load parameter keyframes  
        self.parameter_controller.keyframes = [
            ParameterKeyframe(**k) for k in data['parameter_keyframes']
        ]
        
        # Load lighting keyframes
        self.lighting_keyframes = [
            LightingKeyframe(**k) for k in data['lighting_keyframes']
        ]

# Predefined animation templates
class AnimationTemplates:
    """Collection of pre-made animation sequences"""
    
    @staticmethod
    def create_exploration_flyby() -> AnimationSequence:
        """Create a dramatic flyby exploration animation"""
        seq = AnimationSequence()
        seq.set_duration(8.0, 30.0)
        
        # Camera movement - dramatic flyby
        seq.camera_controller.create_orbit_animation(
            center=(0.0, 0.0, 0.0),
            radius=4.0,
            height_variation=2.0,
            num_keyframes=6
        )
        seq.camera_controller.set_easing('ease_in_out_cubic')
        
        # Parameter animation - power morph
        seq.parameter_controller.create_power_morph(2.0, 12.0, "mandelbulb")
        seq.parameter_controller.set_easing('ease_in_out_sine')
        
        # Add lighting keyframe
        seq.lighting_keyframes.append(LightingKeyframe(
            time=0.0,
            light1_intensity=1.2,
            light1_color=(1.0, 0.9, 0.8),
            ambient=0.1
        ))
        
        return seq
    
    @staticmethod
    def create_zoom_and_morph() -> AnimationSequence:
        """Create zoom-in with fractal morphing"""
        seq = AnimationSequence() 
        seq.set_duration(6.0, 30.0)
        
        # Zoom animation
        seq.camera_controller.create_zoom_animation(
            start_pos=(0.0, 0.0, -5.0),
            target=(0.0, 0.0, 0.0),
            zoom_factor=8.0
        )
        seq.camera_controller.set_easing('ease_in_cubic')
        
        # Complex parameter morphing
        seq.parameter_controller.add_keyframe(ParameterKeyframe(
            time=0.0, fractal_type="mandelbulb", power=8.0,
            color_palette="warm", metallic=0.0
        ))
        seq.parameter_controller.add_keyframe(ParameterKeyframe(
            time=0.3, fractal_type="mandelbulb", power=4.0,
            color_palette="cool", metallic=0.3
        ))
        seq.parameter_controller.add_keyframe(ParameterKeyframe(
            time=0.7, fractal_type="julia_3d", power=6.0,
            julia_c=(-0.2, 0.1, 0.0), color_palette="rainbow", metallic=0.7
        ))
        seq.parameter_controller.add_keyframe(ParameterKeyframe(
            time=1.0, fractal_type="julia_3d", power=10.0,
            julia_c=(-0.1, 0.2, 0.1), color_palette="fire", metallic=1.0
        ))
        
        return seq
    
    @staticmethod
    def create_fractal_showcase() -> AnimationSequence:
        """Showcase different fractal types"""
        seq = AnimationSequence()
        seq.set_duration(12.0, 30.0)
        
        # Static camera focused on center
        seq.camera_controller.add_keyframe(CameraKeyframe(
            time=0.0, position=(0.0, 0.0, -3.0), target=(0.0, 0.0, 0.0)
        ))
        seq.camera_controller.add_keyframe(CameraKeyframe(
            time=1.0, position=(0.0, 0.0, -3.0), target=(0.0, 0.0, 0.0)
        ))
        
        # Cycle through different fractal types
        fractal_types = ["mandelbulb", "julia_3d", "mandelbox", "burning_ship_3d", "menger_sponge"]
        palettes = ["warm", "cool", "rainbow", "fire", "ice"]
        
        for i, (ftype, palette) in enumerate(zip(fractal_types, palettes)):
            t = i / (len(fractal_types) - 1)
            
            # Set fractal-specific parameters
            if ftype == "julia_3d":
                julia_c = (-0.2 + i*0.1, 0.1 - i*0.05, i*0.05)
            else:
                julia_c = (0.0, 0.0, 0.0)
            
            seq.parameter_controller.add_keyframe(ParameterKeyframe(
                time=t,
                fractal_type=ftype,
                power=8.0 - i*0.5,  # Vary power slightly
                julia_c=julia_c,
                color_palette=palette,
                metallic=i * 0.2
            ))
        
        return seq