#!/usr/bin/env python3
"""
Procedural Animation Generators
================================

Generate fractal animations procedurally using mathematical patterns:
- Camera orbit generators (spherical, toroidal, spiral)
- Formula evolution generators (power sweeps, parameter morphs)
- Color animation generators (palette cycling, transitions)

Usage:
    from animation.procedural_generators import OrbitGenerator
    
    # Generate spherical orbit
    generator = OrbitGenerator(pattern="spherical")
    keyframes = generator.generate(center=(0, 0, 0), radius=3.0, frames=120)
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import random

from renderers.python_3d.fractal_animator import FractalParams
from animation_controller import AnimationKeyframe, EasingType


class OrbitPattern(Enum):
    """Camera orbit patterns"""
    SPHERICAL = "spherical"          # Orbit on sphere surface
    TOROIDAL = "toroidal"            # Torus knot paths
    LISSAJOUS = "lissajous"          # Lissajous curves in 3D
    SPIRAL = "spiral"                # Spiral in/out
    FIGURE_EIGHT = "figure_eight"    # Figure-8 paths
    RANDOM_WALK = "random_walk"      # Perlin noise guided
    ELLIPTICAL = "elliptical"        # Elliptical orbits
    HELIX = "helix"                  # Helical paths


@dataclass
class OrbitConfig:
    """Configuration for orbit generation"""
    pattern: OrbitPattern = OrbitPattern.SPHERICAL
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius: float = 3.0
    height_variation: float = 0.0  # For non-planar orbits
    speed: float = 1.0  # Orbits per animation
    direction: str = "clockwise"  # or "counterclockwise"
    tilt: float = 0.0  # Axis tilt in radians


class OrbitGenerator:
    """
    Generate camera orbit keyframes procedurally.
    
    Creates smooth camera paths that orbit around fractal center,
    avoiding surface clipping and providing cinematic motion.
    """
    
    def __init__(self, config: Optional[OrbitConfig] = None):
        self.config = config or OrbitConfig()
    
    def generate(self, frames: int = 120) -> List[AnimationKeyframe]:
        """
        Generate orbit keyframes.
        
        Args:
            frames: Number of keyframes to generate
            
        Returns:
            List of AnimationKeyframe objects
        """
        pattern_method = getattr(self, f"_generate_{self.config.pattern.value}")
        return pattern_method(frames)
    
    def _generate_spherical(self, frames: int) -> List[AnimationKeyframe]:
        """Generate spherical orbit (standard orbital path)"""
        keyframes = []
        cx, cy, cz = self.config.center
        r = self.config.radius
        
        for i in range(frames):
            t = i / (frames - 1) if frames > 1 else 0.0
            angle = t * 2.0 * np.pi * self.config.speed
            
            if self.config.direction == "counterclockwise":
                angle = -angle
            
            # Apply tilt rotation
            tilt = self.config.tilt
            
            # Spherical coordinates with height variation
            theta = angle  # Azimuth
            phi = np.pi / 2 + self.config.height_variation * np.sin(angle * 2)
            
            # Convert to Cartesian
            x = cx + r * np.sin(phi) * np.cos(theta)
            y = cy + r * np.sin(phi) * np.sin(theta)
            z = cz + r * np.cos(phi)
            
            # Apply tilt
            if tilt != 0:
                y_rot = y * np.cos(tilt) - z * np.sin(tilt)
                z_rot = y * np.sin(tilt) + z * np.cos(tilt)
                y, z = y_rot, z_rot
            
            params = FractalParams()
            params.camera_pos = (float(x), float(y), float(z))
            params.target = self.config.center
            
            keyframes.append(AnimationKeyframe(
                time=t,
                params=params,
                easing=EasingType.LINEAR
            ))
        
        return keyframes
    
    def _generate_toroidal(self, frames: int) -> List[AnimationKeyframe]:
        """Generate toroidal orbit (knot-like paths)"""
        keyframes = []
        cx, cy, cz = self.config.center
        R = self.config.radius  # Major radius
        r = R * 0.3  # Minor radius
        
        # Torus knot parameters
        p, q = 2, 3  # Co-prime integers for knot
        
        for i in range(frames):
            t = i / (frames - 1) if frames > 1 else 0.0
            angle = t * 2.0 * np.pi * self.config.speed
            
            # Torus knot equations
            theta = angle * p
            phi = angle * q
            
            x = cx + (R + r * np.cos(phi)) * np.cos(theta)
            y = cy + (R + r * np.cos(phi)) * np.sin(theta)
            z = cz + r * np.sin(phi)
            
            params = FractalParams()
            params.camera_pos = (float(x), float(y), float(z))
            params.target = self.config.center
            
            keyframes.append(AnimationKeyframe(
                time=t,
                params=params,
                easing=EasingType.EASE_IN_OUT
            ))
        
        return keyframes
    
    def _generate_lissajous(self, frames: int) -> List[AnimationKeyframe]:
        """Generate Lissajous curve in 3D"""
        keyframes = []
        cx, cy, cz = self.config.center
        r = self.config.radius
        
        # Lissajous frequencies
        a, b, c = 3, 2, 1
        delta_xy = np.pi / 4
        delta_xz = np.pi / 2
        
        for i in range(frames):
            t = i / (frames - 1) if frames > 1 else 0.0
            angle = t * 2.0 * np.pi * self.config.speed
            
            x = cx + r * np.sin(a * angle + delta_xy)
            y = cy + r * np.sin(b * angle)
            z = cz + r * np.sin(c * angle + delta_xz) * 0.5
            
            params = FractalParams()
            params.camera_pos = (float(x), float(y), float(z))
            params.target = self.config.center
            
            keyframes.append(AnimationKeyframe(
                time=t,
                params=params,
                easing=EasingType.EASE_IN_OUT
            ))
        
        return keyframes
    
    def _generate_spiral(self, frames: int) -> List[AnimationKeyframe]:
        """Generate spiral orbit (move in/out while rotating)"""
        keyframes = []
        cx, cy, cz = self.config.center
        r_min = self.config.radius * 0.5
        r_max = self.config.radius * 1.5
        
        for i in range(frames):
            t = i / (frames - 1) if frames > 1 else 0.0
            angle = t * 2.0 * np.pi * self.config.speed * 2  # Double rotation
            
            # Radius varies from min to max and back
            radius = r_min + (r_max - r_min) * (0.5 + 0.5 * np.sin(t * np.pi))
            
            x = cx + radius * np.cos(angle)
            y = cy + radius * np.sin(angle)
            z = cz + self.config.height_variation * np.sin(t * 2 * np.pi)
            
            params = FractalParams()
            params.camera_pos = (float(x), float(y), float(z))
            params.target = self.config.center
            
            keyframes.append(AnimationKeyframe(
                time=t,
                params=params,
                easing=EasingType.EASE_IN_OUT
            ))
        
        return keyframes
    
    def _generate_figure_eight(self, frames: int) -> List[AnimationKeyframe]:
        """Generate figure-8 orbit (lemniscate)"""
        keyframes = []
        cx, cy, cz = self.config.center
        r = self.config.radius
        
        for i in range(frames):
            t = i / (frames - 1) if frames > 1 else 0.0
            angle = t * 2.0 * np.pi * self.config.speed
            
            # Lemniscate of Bernoulli
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            denom = 1 + sin_a * sin_a
            
            x = cx + r * cos_a / denom
            y = cy + r * sin_a * cos_a / denom
            z = cz + self.config.height_variation * np.sin(angle * 2)
            
            params = FractalParams()
            params.camera_pos = (float(x), float(y), float(z))
            params.target = self.config.center
            
            keyframes.append(AnimationKeyframe(
                time=t,
                params=params,
                easing=EasingType.EASE_IN_OUT
            ))
        
        return keyframes
    
    def _generate_random_walk(self, frames: int) -> List[AnimationKeyframe]:
        """Generate Perlin noise guided random walk"""
        keyframes = []
        cx, cy, cz = self.config.center
        r = self.config.radius
        
        # Simple pseudo-random walk (could use actual Perlin noise)
        np.random.seed(42)  # Reproducible
        
        positions = [(cx + r, cy, cz)]  # Start position
        
        for i in range(1, frames):
            t = i / (frames - 1)
            
            # Random spherical offset
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0, np.pi)
            
            # Smooth by averaging with previous
            dx = r * np.sin(phi) * np.cos(theta) * 0.3
            dy = r * np.sin(phi) * np.sin(theta) * 0.3
            dz = r * np.cos(phi) * 0.3
            
            prev_x, prev_y, prev_z = positions[-1]
            new_x = cx + (prev_x - cx) * 0.9 + dx
            new_y = cy + (prev_y - cy) * 0.9 + dy
            new_z = cz + (prev_z - cz) * 0.9 + dz
            
            # Normalize to maintain radius
            dist = np.sqrt((new_x - cx)**2 + (new_y - cy)**2 + (new_z - cz)**2)
            scale = r / dist if dist > 0 else 1
            new_x = cx + (new_x - cx) * scale
            new_y = cy + (new_y - cy) * scale
            new_z = cz + (new_z - cz) * scale
            
            positions.append((new_x, new_y, new_z))
        
        # Convert to keyframes
        for i, (x, y, z) in enumerate(positions):
            t = i / (frames - 1) if frames > 1 else 0.0
            
            params = FractalParams()
            params.camera_pos = (float(x), float(y), float(z))
            params.target = self.config.center
            
            keyframes.append(AnimationKeyframe(
                time=t,
                params=params,
                easing=EasingType.LINEAR
            ))
        
        return keyframes
    
    def _generate_elliptical(self, frames: int) -> List[AnimationKeyframe]:
        """Generate elliptical orbit"""
        keyframes = []
        cx, cy, cz = self.config.center
        a = self.config.radius  # Semi-major axis
        b = a * 0.6  # Semi-minor axis
        
        for i in range(frames):
            t = i / (frames - 1) if frames > 1 else 0.0
            angle = t * 2.0 * np.pi * self.config.speed
            
            x = cx + a * np.cos(angle)
            y = cy + b * np.sin(angle)
            z = cz + self.config.height_variation * np.sin(angle * 2)
            
            params = FractalParams()
            params.camera_pos = (float(x), float(y), float(z))
            params.target = self.config.center
            
            keyframes.append(AnimationKeyframe(
                time=t,
                params=params,
                easing=EasingType.EASE_IN_OUT
            ))
        
        return keyframes
    
    def _generate_helix(self, frames: int) -> List[AnimationKeyframe]:
        """Generate helical path"""
        keyframes = []
        cx, cy, cz = self.config.center
        r = self.config.radius
        
        height_range = self.config.height_variation * 2
        
        for i in range(frames):
            t = i / (frames - 1) if frames > 1 else 0.0
            angle = t * 2.0 * np.pi * self.config.speed * 3  # Multiple turns
            
            x = cx + r * np.cos(angle)
            y = cy + r * np.sin(angle)
            z = cz + height_range * (t - 0.5)  # Move up then down
            
            params = FractalParams()
            params.camera_pos = (float(x), float(y), float(z))
            params.target = self.config.center
            
            keyframes.append(AnimationKeyframe(
                time=t,
                params=params,
                easing=EasingType.LINEAR
            ))
        
        return keyframes


class FormulaEvolutionGenerator:
    """
    Generate procedural formula parameter evolution.
    
    Creates smooth transitions between formula states:
    - Power sweeps
    - Iteration ramp-ups
    - Formula morphing
    - Chaos-to-order transitions
    """
    
    @staticmethod
    def generate_power_sweep(
        start_power: float,
        end_power: float,
        frames: int = 120,
        easing: EasingType = EasingType.EASE_IN_OUT
    ) -> List[AnimationKeyframe]:
        """
        Generate power sweep animation.
        
        Smoothly transitions power from start to end value,
        useful for revealing different fractal structures.
        """
        keyframes = []
        
        for i in range(frames):
            t = i / (frames - 1) if frames > 1 else 0.0
            
            # Apply easing
            if easing == EasingType.EASE_IN_OUT:
                t = t * t * (3 - 2 * t)
            elif easing == EasingType.CUBIC:
                t = t * t * (3 - 2 * t)
            
            # Log-space interpolation for smooth visual evolution
            log_start = np.log(max(start_power, 0.1))
            log_end = np.log(max(end_power, 0.1))
            log_current = log_start + (log_end - log_start) * t
            power = np.exp(log_current)
            
            params = FractalParams()
            params.power = float(power)
            
            keyframes.append(AnimationKeyframe(
                time=t,
                params=params,
                easing=easing
            ))
        
        return keyframes
    
    @staticmethod
    def generate_iteration_ramp(
        start_iter: int = 50,
        end_iter: int = 200,
        frames: int = 120
    ) -> List[AnimationKeyframe]:
        """Generate iteration count ramp-up (increasing detail)"""
        keyframes = []
        
        for i in range(frames):
            t = i / (frames - 1) if frames > 1 else 0.0
            
            # Exponential ramp for perceptually smooth increase
            iterations = int(start_iter + (end_iter - start_iter) * (t ** 0.5))
            
            params = FractalParams()
            params.iterations = iterations
            
            keyframes.append(AnimationKeyframe(
                time=t,
                params=params,
                easing=EasingType.EASE_IN_OUT
            ))
        
        return keyframes
    
    @staticmethod
    def generate_formula_morph(
        params1: FractalParams,
        params2: FractalParams,
        frames: int = 120
    ) -> List[AnimationKeyframe]:
        """
        Generate smooth morph between two formula parameter sets.
        
        Uses smart interpolation for fractal-aware transitions.
        """
        keyframes = []
        
        from animation_controller import MandelbulbMorph
        morph_helper = MandelbulbMorph()
        
        for i in range(frames):
            t = i / (frames - 1) if frames > 1 else 0.0
            
            # Use smart interpolation
            params = FractalParams()
            
            # Power (log-space)
            params.power = morph_helper.smooth_power_transition(
                params1.power, params2.power, t
            )
            
            # Camera (spherical)
            params.camera_pos = morph_helper.spherical_camera_interpolation(
                params1.camera_pos, params2.camera_pos, t
            )
            params.target = (
                params1.target[0] + (params2.target[0] - params1.target[0]) * t,
                params1.target[1] + (params2.target[1] - params1.target[1]) * t,
                params1.target[2] + (params2.target[2] - params1.target[2]) * t
            )
            
            # Color (HSV space)
            params.base_color = morph_helper.color_harmony_transition(
                params1.base_color, params2.base_color, t
            )
            
            # Linear interpolation for others
            params.iterations = int(params1.iterations + (params2.iterations - params1.iterations) * t)
            params.fov = params1.fov + (params2.fov - params1.fov) * t
            params.metallic = params1.metallic + (params2.metallic - params1.metallic) * t
            params.roughness = params1.roughness + (params2.roughness - params1.roughness) * t
            
            keyframes.append(AnimationKeyframe(
                time=t,
                params=params,
                easing=EasingType.EASE_IN_OUT
            ))
        
        return keyframes


class ColorAnimationGenerator:
    """Generate color and palette animations"""
    
    @staticmethod
    def generate_palette_cycle(
        palettes: List[str],
        frames_per_palette: int = 60,
        transition_frames: int = 30
    ) -> List[AnimationKeyframe]:
        """
        Generate palette cycling animation.
        
        Cycles through multiple color palettes with smooth transitions.
        """
        keyframes = []
        
        for i, palette in enumerate(palettes):
            # Hold on palette
            for j in range(frames_per_palette):
                t = len(keyframes) / (len(palettes) * frames_per_palette)
                
                params = FractalParams()
                params.color_palette = palette
                
                keyframes.append(AnimationKeyframe(
                    time=t,
                    params=params,
                    easing=EasingType.LINEAR
                ))
            
            # Transition to next (except for last)
            if i < len(palettes) - 1:
                next_palette = palettes[i + 1]
                # Could add color interpolation here
        
        # Renormalize times
        total_frames = len(keyframes)
        for i, kf in enumerate(keyframes):
            kf.time = i / (total_frames - 1) if total_frames > 1 else 0.0
        
        return keyframes
    
    @staticmethod
    def generate_hue_rotation(
        start_hue: float = 0.0,
        rotations: float = 2.0,
        frames: int = 120
    ) -> List[AnimationKeyframe]:
        """
        Generate continuous hue rotation animation.
        
        Rotates through full color spectrum for psychedelic effect.
        """
        keyframes = []
        
        for i in range(frames):
            t = i / (frames - 1) if frames > 1 else 0.0
            
            # Calculate hue
            hue = (start_hue + t * rotations) % 1.0
            
            # Convert HSV to RGB
            import matplotlib.colors as mcolors
            rgb = mcolors.hsv_to_rgb([hue, 0.8, 0.9])
            
            params = FractalParams()
            params.base_color = (float(rgb[0]), float(rgb[1]), float(rgb[2]))
            params.color_intensity = 1.0 + 0.5 * np.sin(t * 2 * np.pi)
            
            keyframes.append(AnimationKeyframe(
                time=t,
                params=params,
                easing=EasingType.LINEAR
            ))
        
        return keyframes


def demo_procedural_generators():
    """Demonstrate procedural generators"""
    print("Procedural Animation Generators Demo")
    print("=" * 60)
    
    # Orbit generators
    print("\n1. Orbit Patterns:")
    for pattern in OrbitPattern:
        config = OrbitConfig(pattern=pattern, radius=3.0)
        generator = OrbitGenerator(config)
        keyframes = generator.generate(frames=60)
        print(f"  {pattern.value}: {len(keyframes)} keyframes generated")
    
    # Formula evolution
    print("\n2. Formula Evolution:")
    
    # Power sweep
    power_keyframes = FormulaEvolutionGenerator.generate_power_sweep(
        start_power=2.0, end_power=16.0, frames=60
    )
    print(f"  Power sweep: {len(power_keyframes)} keyframes")
    print(f"    Start power: {power_keyframes[0].params.power:.2f}")
    print(f"    End power: {power_keyframes[-1].params.power:.2f}")
    
    # Iteration ramp
    iter_keyframes = FormulaEvolutionGenerator.generate_iteration_ramp(
        start_iter=50, end_iter=200, frames=60
    )
    print(f"  Iteration ramp: {len(iter_keyframes)} keyframes")
    print(f"    Start iterations: {iter_keyframes[0].params.iterations}")
    print(f"    End iterations: {iter_keyframes[-1].params.iterations}")
    
    # Color animations
    print("\n3. Color Animations:")
    
    hue_keyframes = ColorAnimationGenerator.generate_hue_rotation(
        rotations=2.0, frames=60
    )
    print(f"  Hue rotation: {len(hue_keyframes)} keyframes")
    
    print("\nDemo complete!")


if __name__ == "__main__":
    demo_procedural_generators()
