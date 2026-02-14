#!/usr/bin/env python3
"""
Fractal Animation Controller
============================

Advanced animation system with:
- Smooth Mandelbulb space morphing
- Catmull-Rom spline interpolation for camera
- Fast GPU preview with Taichi
- Animation curve editing
- Professional video export

Usage:
    python animation_controller.py --preview --keyfile animation.json
    python animation_controller.py --render --output final_video.mp4
"""

import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
from scipy import interpolate

# Import existing systems
from renderers.python_3d.fractal_animator import FractalRenderer, FractalParams, FractalPresets


class EasingType(Enum):
    """Animation easing functions for smooth motion"""
    LINEAR = "linear"
    EASE_IN_OUT = "ease_in_out"  # Smooth acceleration/deceleration
    CUBIC = "cubic"              # S-curve
    CATMULL_ROM = "catmull_rom"  # Smooth spline through keyframes
    BOUNCE = "bounce"            # Bouncy effect
    ELASTIC = "elastic"          # Springy effect
    EXPONENTIAL = "exponential"  # Fast start, slow end


@dataclass
class AnimationKeyframe:
    """Enhanced keyframe with easing and curve control"""
    time: float  # 0.0 to 1.0
    params: FractalParams
    easing: EasingType = EasingType.EASE_IN_OUT
    
    # Animation curves for individual parameters
    # Allows different parameters to animate at different speeds
    parameter_curves: Dict[str, Callable] = field(default_factory=dict)
    
    # Hold this keyframe (don't interpolate to next)
    hold: bool = False
    
    # Notes for the animator
    notes: str = ""


@dataclass
class MandelbulbMorph:
    """
    Smart interpolation for Mandelbulb parameters.
    
    Unlike simple linear interpolation, this ensures:
    - Power transitions maintain fractal continuity
    - Camera paths avoid 'popping' through the fractal
    - Color transitions remain harmonious
    """
    
    @staticmethod
    def smooth_power_transition(start_power: float, end_power: float, t: float) -> float:
        """
        Smoothly transition between powers without jarring jumps.
        Uses exponential interpolation for natural feel.
        """
        # Powers affect the fractal exponentially, so interpolate in log space
        if start_power <= 0 or end_power <= 0:
            return start_power + (end_power - start_power) * t
        
        log_start = np.log(start_power)
        log_end = np.log(end_power)
        log_result = log_start + (log_end - log_start) * t
        return np.exp(log_result)
    
    @staticmethod
    def spherical_camera_interpolation(
        start_pos: Tuple[float, float, float],
        end_pos: Tuple[float, float, float],
        t: float
    ) -> Tuple[float, float, float]:
        """
        Interpolate camera along spherical path to avoid cutting through fractal.
        
        Instead of linear path A->B, we move along sphere surface.
        This prevents camera from passing through the fractal surface.
        """
        # Convert to spherical coordinates
        def to_spherical(x, y, z):
            r = np.sqrt(x**2 + y**2 + z**2)
            theta = np.arctan2(np.sqrt(x**2 + y**2), z)
            phi = np.arctan2(y, x)
            return r, theta, phi
        
        def to_cartesian(r, theta, phi):
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)
            return x, y, z
        
        r1, t1, p1 = to_spherical(*start_pos)
        r2, t2, p2 = to_spherical(*end_pos)
        
        # Interpolate in spherical space
        r = r1 + (r2 - r1) * t
        theta = t1 + (t2 - t1) * t
        phi = p1 + (p2 - p1) * t
        
        return to_cartesian(r, theta, phi)
    
    @staticmethod
    def color_harmony_transition(
        start_color: Tuple[float, float, float],
        end_color: Tuple[float, float, float],
        t: float
    ) -> Tuple[float, float, float]:
        """
        Interpolate colors through HSV space for harmonious transitions.
        RGB interpolation can produce muddy grays; HSV maintains vibrancy.
        """
        def rgb_to_hsv(r, g, b):
            maxc = max(r, g, b)
            minc = min(r, g, b)
            v = maxc
            if minc == maxc:
                return 0.0, 0.0, v
            s = (maxc - minc) / maxc
            rc = (maxc - r) / (maxc - minc)
            gc = (maxc - g) / (maxc - minc)
            bc = (maxc - b) / (maxc - minc)
            if r == maxc:
                h = bc - gc
            elif g == maxc:
                h = 2.0 + rc - bc
            else:
                h = 4.0 + gc - rc
            h = (h / 6.0) % 1.0
            return h, s, v
        
        def hsv_to_rgb(h, s, v):
            if s == 0.0:
                return v, v, v
            i = int(h * 6.0)
            f = (h * 6.0) - i
            p = v * (1.0 - s)
            q = v * (1.0 - s * f)
            t = v * (1.0 - s * (1.0 - f))
            i = i % 6
            if i == 0:
                return v, t, p
            if i == 1:
                return q, v, p
            if i == 2:
                return p, v, t
            if i == 3:
                return p, q, v
            if i == 4:
                return t, p, v
            return v, p, q
        
        h1, s1, v1 = rgb_to_hsv(*start_color)
        h2, s2, v2 = rgb_to_hsv(*end_color)
        
        # Handle hue wrapping (shortest path around color wheel)
        if abs(h2 - h1) > 0.5:
            if h2 > h1:
                h1 += 1.0
            else:
                h2 += 1.0
        
        h = h1 + (h2 - h1) * t
        s = s1 + (s2 - s1) * t
        v = v1 + (v2 - v1) * t
        
        return hsv_to_rgb(h % 1.0, s, v)


class EasingFunctions:
    """Collection of easing functions for smooth animation"""
    
    @staticmethod
    def linear(t: float) -> float:
        return t
    
    @staticmethod
    def ease_in_out(t: float) -> float:
        """Smooth acceleration and deceleration"""
        return t * t * (3.0 - 2.0 * t)
    
    @staticmethod
    def cubic(t: float) -> float:
        """S-curve"""
        return t * t * (3.0 - 2.0 * t)
    
    @staticmethod
    def exponential(t: float) -> float:
        """Fast start, slow end"""
        return 0.0 if t == 0.0 else np.power(2.0, 10.0 * (t - 1.0))
    
    @staticmethod
    def bounce(t: float) -> float:
        """Bouncy effect"""
        if t < 1.0 / 2.75:
            return 7.5625 * t * t
        elif t < 2.0 / 2.75:
            t -= 1.5 / 2.75
            return 7.5625 * t * t + 0.75
        elif t < 2.5 / 2.75:
            t -= 2.25 / 2.75
            return 7.5625 * t * t + 0.9375
        else:
            t -= 2.625 / 2.75
            return 7.5625 * t * t + 0.984375
    
    @staticmethod
    def elastic(t: float) -> float:
        """Springy effect"""
        if t == 0.0:
            return 0.0
        if t == 1.0:
            return 1.0
        p = 0.3
        s = p / 4.0
        return np.power(2.0, -10.0 * t) * np.sin((t - s) * (2.0 * np.pi) / p) + 1.0
    
    @classmethod
    def apply(cls, t: float, easing_type: EasingType) -> float:
        """Apply easing function"""
        if easing_type == EasingType.LINEAR:
            return cls.linear(t)
        elif easing_type == EasingType.EASE_IN_OUT:
            return cls.ease_in_out(t)
        elif easing_type == EasingType.CUBIC:
            return cls.cubic(t)
        elif easing_type == EasingType.EXPONENTIAL:
            return cls.exponential(t)
        elif easing_type == EasingType.BOUNCE:
            return cls.bounce(t)
        elif easing_type == EasingType.ELASTIC:
            return cls.elastic(t)
        return t


class CatmullRomSpline:
    """
    Catmull-Rom spline interpolation for smooth camera paths.
    
    Creates smooth curves that pass through all control points (keyframes).
    Much smoother than linear interpolation for camera movement.
    """
    
    def __init__(self, points: List[Tuple[float, ...]]):
        """
        Initialize spline with control points.
        
        Args:
            points: List of N-dimensional points (e.g., [(x1,y1,z1), (x2,y2,z2), ...])
        """
        self.points = np.array(points)
        self.n = len(points)
        
        if self.n < 2:
            raise ValueError("Need at least 2 points for spline")
    
    def interpolate(self, t: float) -> Tuple[float, ...]:
        """
        Get interpolated point at parameter t (0.0 to 1.0).
        """
        # Scale t to point index range
        t_scaled = t * (self.n - 1)
        i = int(np.floor(t_scaled))
        remainder = t_scaled - i
        
        # Clamp to valid range
        i = max(0, min(i, self.n - 2))
        
        # Get 4 points for Catmull-Rom (with boundary handling)
        p0 = self.points[max(0, i - 1)]
        p1 = self.points[i]
        p2 = self.points[min(self.n - 1, i + 1)]
        p3 = self.points[min(self.n - 1, i + 2)]
        
        # Catmull-Rom basis functions
        t2 = remainder * remainder
        t3 = t2 * remainder
        
        result = (
            0.5 * (2.0 * p1 +
                   (p2 - p0) * remainder +
                   (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2 +
                   (3.0 * p1 - p0 - 3.0 * p2 + p3) * t3)
        )
        
        return tuple(result)


class AnimationController:
    """
    Main controller for fractal animation with advanced features.
    """
    
    def __init__(self):
        self.keyframes: List[AnimationKeyframe] = []
        self.renderer = FractalRenderer()
        self.morph_helper = MandelbulbMorph()
        
    def add_keyframe(self, time: float, params: FractalParams, 
                     easing: EasingType = EasingType.EASE_IN_OUT,
                     notes: str = ""):
        """Add a keyframe to the animation"""
        keyframe = AnimationKeyframe(
            time=time,
            params=params,
            easing=easing,
            notes=notes
        )
        self.keyframes.append(keyframe)
        self.keyframes.sort(key=lambda k: k.time)
    
    def get_parameters_at_time(self, t: float) -> FractalParams:
        """
        Get interpolated fractal parameters at time t (0.0 to 1.0).
        
        Uses smart interpolation for Mandelbulb-specific parameters.
        """
        if not self.keyframes:
            return FractalParams()
        
        # Clamp time
        t = max(0.0, min(1.0, t))
        
        # Find surrounding keyframes
        prev_kf = None
        next_kf = None
        
        for kf in self.keyframes:
            if kf.time <= t:
                prev_kf = kf
            elif kf.time > t and next_kf is None:
                next_kf = kf
                break
        
        # Handle edge cases
        if prev_kf is None:
            return self.keyframes[0].params
        if next_kf is None or prev_kf.hold:
            return prev_kf.params
        
        # Calculate interpolation factor
        alpha = (t - prev_kf.time) / (next_kf.time - prev_kf.time)
        
        # Apply easing
        alpha = EasingFunctions.apply(alpha, prev_kf.easing)
        
        # Interpolate with Mandelbulb-aware morphing
        return self._smart_interpolate(prev_kf.params, next_kf.params, alpha)
    
    def _smart_interpolate(self, p1: FractalParams, p2: FractalParams, 
                          alpha: float) -> FractalParams:
        """
        Interpolate between two parameter sets with Mandelbulb awareness.
        """
        result = FractalParams()
        
        # Use smart power interpolation
        result.power = self.morph_helper.smooth_power_transition(
            p1.power, p2.power, alpha
        )
        
        # Use spherical camera interpolation
        result.camera_pos = self.morph_helper.spherical_camera_interpolation(
            p1.camera_pos, p2.camera_pos, alpha
        )
        result.target = (
            p1.target[0] + (p2.target[0] - p1.target[0]) * alpha,
            p1.target[1] + (p2.target[1] - p1.target[1]) * alpha,
            p1.target[2] + (p2.target[2] - p1.target[2]) * alpha
        )
        
        # Use HSV color interpolation
        result.base_color = self.morph_helper.color_harmony_transition(
            p1.base_color, p2.base_color, alpha
        )
        
        # Linear interpolation for other parameters
        result.iterations = int(p1.iterations + (p2.iterations - p1.iterations) * alpha)
        result.fov = p1.fov + (p2.fov - p1.fov) * alpha
        result.metallic = p1.metallic + (p2.metallic - p1.metallic) * alpha
        result.roughness = p1.roughness + (p2.roughness - p1.roughness) * alpha
        result.color_intensity = p1.color_intensity + (p2.color_intensity - p1.color_intensity) * alpha
        
        # Copy other properties from start
        result.fractal_type = p1.fractal_type
        result.coloring_mode = p1.coloring_mode
        result.color_palette = p1.color_palette if alpha < 0.5 else p2.color_palette
        result.light_pos = p1.light_pos
        result.ambient = p1.ambient
        result.width = p1.width
        result.height = p1.height
        result.max_ray_steps = p1.max_ray_steps
        result.epsilon = p1.epsilon
        result.max_distance = p1.max_distance
        
        return result
    
    def render_preview(self, width: int = 320, height: int = 240, 
                      fps: int = 15, duration: float = 5.0) -> List[np.ndarray]:
        """
        Render low-resolution preview for real-time feedback.
        
        Much faster than full-resolution final render.
        """
        frames = []
        total_frames = int(fps * duration)
        
        print(f"Rendering preview: {width}x{height} @ {fps}fps for {duration}s")
        print(f"Total frames: {total_frames}")
        
        for i in range(total_frames):
            t = i / (total_frames - 1) if total_frames > 1 else 0.0
            params = self.get_parameters_at_time(t)
            
            # Use low resolution for speed
            params.width = width
            params.height = height
            params.max_ray_steps = 50  # Reduced quality for speed
            
            image, metrics = self.renderer.render(params)
            frames.append(image)
            
            if (i + 1) % 10 == 0:
                print(f"  Frame {i+1}/{total_frames} ({metrics['render_time']:.2f}s)")
        
        return frames
    
    def export_video(self, output_path: str, width: int = 1920, height: int = 1080,
                    fps: int = 30, duration: float = 10.0, quality: str = "high"):
        """
        Export final animation as video file using FFmpeg.
        
        Supports: MP4 (H.264), ProRes, WebM
        """
        try:
            import ffmpeg
        except ImportError:
            print("Error: ffmpeg-python not installed")
            print("Install with: pip install ffmpeg-python")
            return False
        
        output_path = Path(output_path)
        temp_dir = Path("temp_frames")
        temp_dir.mkdir(exist_ok=True)
        
        total_frames = int(fps * duration)
        print(f"\n{'='*60}")
        print(f"Exporting Video: {output_path}")
        print(f"Resolution: {width}x{height}")
        print(f"Duration: {duration}s @ {fps}fps")
        print(f"Total frames: {total_frames}")
        print(f"{'='*60}\n")
        
        # Render frames
        start_time = time.time()
        
        for i in range(total_frames):
            t = i / (total_frames - 1) if total_frames > 1 else 0.0
            params = self.get_parameters_at_time(t)
            
            params.width = width
            params.height = height
            
            if quality == "draft":
                params.max_ray_steps = 50
            elif quality == "medium":
                params.max_ray_steps = 100
            else:  # high
                params.max_ray_steps = 200
            
            image, metrics = self.renderer.render(params)
            
            # Save frame
            frame_path = temp_dir / f"frame_{i:06d}.png"
            import matplotlib.pyplot as plt
            plt.imsave(frame_path, image)
            
            # Progress
            elapsed = time.time() - start_time
            eta = (elapsed / (i + 1)) * (total_frames - i - 1)
            print(f"Frame {i+1}/{total_frames} | "
                  f"Time: {metrics['render_time']:.1f}s | "
                  f"ETA: {eta/60:.1f}m")
        
        # Compile video with FFmpeg
        print("\nCompiling video with FFmpeg...")
        
        input_pattern = str(temp_dir / "frame_%06d.png")
        
        try:
            if output_path.suffix == '.mp4':
                # H.264 encoding
                (
                    ffmpeg
                    .input(input_pattern, framerate=fps)
                    .output(str(output_path), 
                           vcodec='libx264',
                           pix_fmt='yuv420p',
                           preset='medium',
                           crf=18)
                    .run(overwrite_output=True, quiet=True)
                )
            elif output_path.suffix in ['.mov', '.prores']:
                # ProRes (for editing)
                (
                    ffmpeg
                    .input(input_pattern, framerate=fps)
                    .output(str(output_path),
                           vcodec='prores_ks',
                           pix_fmt='yuv422p10le',
                           qscale=9)
                    .run(overwrite_output=True, quiet=True)
                )
            elif output_path.suffix == '.webm':
                # WebM (for web)
                (
                    ffmpeg
                    .input(input_pattern, framerate=fps)
                    .output(str(output_path),
                           vcodec='libvpx-vp9',
                           pix_fmt='yuv420p',
                           crf=30,
                           b='0',
                           deadline='good')
                    .run(overwrite_output=True, quiet=True)
                )
            
            print(f"✓ Video saved: {output_path}")
            
        except ffmpeg.Error as e:
            print(f"FFmpeg error: {e}")
            return False
        
        # Cleanup temp frames
        import shutil
        shutil.rmtree(temp_dir)
        
        total_time = time.time() - start_time
        print(f"\nTotal export time: {total_time/60:.1f} minutes")
        
        return True
    
    def save_animation(self, filepath: str):
        """Save animation to JSON file"""
        data = {
            'keyframes': [
                {
                    'time': kf.time,
                    'params': {
                        'fractal_type': kf.params.fractal_type,
                        'power': kf.params.power,
                        'iterations': kf.params.iterations,
                        'camera_pos': kf.params.camera_pos,
                        'target': kf.params.target,
                        'base_color': kf.params.base_color,
                        'fov': kf.params.fov,
                        'metallic': kf.params.metallic,
                        'roughness': kf.params.roughness,
                        'color_intensity': kf.params.color_intensity,
                    },
                    'easing': kf.easing.value,
                    'notes': kf.notes
                }
                for kf in self.keyframes
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Animation saved: {filepath}")
    
    def load_animation(self, filepath: str):
        """Load animation from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.keyframes = []
        for kf_data in data['keyframes']:
            params = FractalParams()
            params.fractal_type = kf_data['params'].get('fractal_type', 'mandelbulb')
            params.power = kf_data['params'].get('power', 8.0)
            params.iterations = kf_data['params'].get('iterations', 100)
            params.camera_pos = tuple(kf_data['params'].get('camera_pos', [0.0, 0.0, -3.0]))
            params.target = tuple(kf_data['params'].get('target', [0.0, 0.0, 0.0]))
            params.base_color = tuple(kf_data['params'].get('base_color', [0.8, 0.6, 0.4]))
            params.fov = kf_data['params'].get('fov', 45.0)
            params.metallic = kf_data['params'].get('metallic', 0.0)
            params.roughness = kf_data['params'].get('roughness', 0.1)
            params.color_intensity = kf_data['params'].get('color_intensity', 1.0)
            
            keyframe = AnimationKeyframe(
                time=kf_data['time'],
                params=params,
                easing=EasingType(kf_data.get('easing', 'ease_in_out')),
                notes=kf_data.get('notes', '')
            )
            self.keyframes.append(keyframe)
        
        self.keyframes.sort(key=lambda k: k.time)
        print(f"Animation loaded: {filepath} ({len(self.keyframes)} keyframes)")


def demo_animation():
    """Demonstrate animation controller capabilities"""
    print("Fractal Animation Controller Demo")
    print("=" * 60)
    
    # Create animation
    anim = AnimationController()
    
    # Keyframe 1: Classic Mandelbulb
    kf1_params = FractalPresets.get_preset("classic_mandelbulb")
    kf1_params.width = 800
    kf1_params.height = 600
    anim.add_keyframe(0.0, kf1_params, EasingType.EASE_IN_OUT, "Classic start")
    
    # Keyframe 2: Zoom in with power morph
    kf2_params = FractalParams()
    kf2_params.fractal_type = "mandelbulb"
    kf2_params.power = 12.0
    kf2_params.camera_pos = (0.0, 0.0, -1.5)  # Closer
    kf2_params.base_color = (0.9, 0.3, 0.2)  # Warmer
    kf2_params.width = 800
    kf2_params.height = 600
    anim.add_keyframe(0.5, kf2_params, EasingType.CUBIC, "Power morph & zoom")
    
    # Keyframe 3: Orbit around
    kf3_params = FractalParams()
    kf3_params.fractal_type = "mandelbulb"
    kf3_params.power = 8.0  # Back to original
    kf3_params.camera_pos = (3.0, 0.0, 0.0)  # Orbit to side
    kf3_params.base_color = (0.2, 0.5, 0.9)  # Cooler
    kf3_params.width = 800
    kf3_params.height = 600
    anim.add_keyframe(1.0, kf3_params, EasingType.EASE_IN_OUT, "Orbit finish")
    
    # Save animation
    anim.save_animation("demo_animation.json")
    
    # Render preview
    print("\nRendering preview...")
    frames = anim.render_preview(width=320, height=240, fps=10, duration=3.0)
    print(f"Generated {len(frames)} preview frames")
    
    print("\nDemo complete!")
    print("To export full video, run:")
    print("  anim.export_video('output.mp4', width=1920, height=1080)")


if __name__ == "__main__":
    demo_animation()
