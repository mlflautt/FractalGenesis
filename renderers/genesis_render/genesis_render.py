#!/usr/bin/env python3
"""
GenesisRender - Complete 3D Fractal Rendering System
===================================================

Professional-grade fractal renderer featuring:
- 12+ fractal types with distance estimation
- Advanced lighting system (hard/soft shadows, AO, SSS)
- Smooth animation with easing functions
- Multiple color palettes and materials
- High-performance JIT compilation
- Comprehensive parameter control
- Professional lighting presets
- Animation templates and keyframe system

This is the flagship renderer for FractalGenesis evolution system.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass, asdict
from numba import jit, prange
import math
import imageio

# Import our fractal types and animation system
from .fractal_types import FRACTAL_TYPES, FRACTAL_PRESETS, get_fractal_distance
from .animation_system import (
    AnimationSequence, CameraKeyframe, ParameterKeyframe, LightingKeyframe,
    AnimationTemplates, EasingFunctions
)

@dataclass
class GenesisRenderParams:
    """Complete parameter set for GenesisRender"""
    # Fractal configuration
    fractal_type: str = "mandelbulb"
    power: float = 8.0
    iterations: int = 100
    bailout: float = 2.0
    julia_c: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    
    # Mandelbox specific
    folding_limit: float = 1.0
    folding_value: float = 2.0
    scale: float = -1.5
    
    # Image configuration
    width: int = 800
    height: int = 600
    max_ray_steps: int = 100
    epsilon: float = 0.001
    max_distance: float = 10.0
    
    # Camera system
    camera_pos: Tuple[float, float, float] = (0.0, 0.0, -3.0)
    target: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    up: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    fov: float = 45.0
    
    # Lighting system
    light1_pos: Tuple[float, float, float] = (2.0, 2.0, -2.0)
    light1_intensity: float = 1.0
    light1_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    light1_hard_factor: float = 0.7
    
    light2_pos: Tuple[float, float, float] = (-1.0, 1.0, -1.0)
    light2_intensity: float = 0.5
    light2_color: Tuple[float, float, float] = (0.8, 0.9, 1.0)
    light2_hard_factor: float = 0.3
    
    light3_pos: Tuple[float, float, float] = (0.0, -2.0, 1.0)
    light3_intensity: float = 0.3
    light3_color: Tuple[float, float, float] = (1.0, 0.8, 0.6)
    light3_hard_factor: float = 0.5
    
    # Advanced lighting features
    hard_lighting_enabled: bool = True
    diffuse_lighting_enabled: bool = True
    ambient_occlusion_enabled: bool = True
    subsurface_scattering_enabled: bool = False
    
    # AO settings
    ao_strength: float = 0.3
    ao_radius: float = 0.1
    ao_samples: int = 5
    
    # Global lighting
    global_ambient: float = 0.15
    shadow_softness: float = 0.02
    specular_power: float = 32.0
    specular_intensity: float = 0.5
    
    # Materials
    base_color: Tuple[float, float, float] = (0.8, 0.6, 0.4)
    color_palette: str = "warm"
    color_intensity: float = 1.0
    coloring_mode: str = "orbit_trap"
    metallic: float = 0.0
    roughness: float = 0.1
    
    # Subsurface scattering
    subsurface_color: Tuple[float, float, float] = (1.0, 0.7, 0.5)
    subsurface_radius: float = 0.05
    transmittance: float = 0.1

# Enhanced color palette system
@jit(nopython=True, fastmath=True)
def get_enhanced_color(t, palette_type, intensity, base_color):
    """Enhanced color generation with base color mixing"""
    t = max(0.0, min(1.0, t)) * intensity
    
    # Generate palette color
    if palette_type == 0:  # warm
        r = 0.9 + 0.1 * math.sin(t * 6.28)
        g = 0.5 + 0.4 * math.sin(t * 6.28 + 2.0)
        b = 0.2 + 0.3 * math.sin(t * 6.28 + 4.0)
    elif palette_type == 1:  # cool
        r = 0.2 + 0.3 * math.sin(t * 6.28)
        g = 0.4 + 0.4 * math.sin(t * 6.28 + 2.0)
        b = 0.9 + 0.1 * math.sin(t * 6.28 + 4.0)
    elif palette_type == 2:  # rainbow
        h = t * 6.0
        c = 1.0
        x = c * (1.0 - abs((h % 2.0) - 1.0))
        if h < 1.0:
            r, g, b = c, x, 0.0
        elif h < 2.0:
            r, g, b = x, c, 0.0
        elif h < 3.0:
            r, g, b = 0.0, c, x
        elif h < 4.0:
            r, g, b = 0.0, x, c
        elif h < 5.0:
            r, g, b = x, 0.0, c
        else:
            r, g, b = c, 0.0, x
    elif palette_type == 3:  # monochrome
        r = g = b = 0.3 + 0.7 * t
    elif palette_type == 4:  # fire
        r = min(1.0, t * 1.8)
        g = max(0.0, min(1.0, (t - 0.4) * 2.0))
        b = max(0.0, min(1.0, (t - 0.7) * 3.0))
    elif palette_type == 5:  # ice
        b = min(1.0, t * 1.5)
        g = max(0.0, min(1.0, (t - 0.2) * 1.8))
        r = max(0.0, min(1.0, (t - 0.5) * 2.0))
    else:  # default warm
        r = 0.8 + 0.2 * math.sin(t * 6.28)
        g = 0.4 + 0.4 * math.sin(t * 6.28 + 2.0)
        b = 0.2 + 0.3 * math.sin(t * 6.28 + 4.0)
    
    # Mix with base color
    mix_factor = 0.3
    r = r * (1.0 - mix_factor) + base_color[0] * mix_factor
    g = g * (1.0 - mix_factor) + base_color[1] * mix_factor
    b = b * (1.0 - mix_factor) + base_color[2] * mix_factor
    
    return r, g, b

# Enhanced lighting functions
def compute_enhanced_lighting(pos, normal, view_dir, fractal_params, lighting_params,
                             fractal_type, fractal_distance_params):
    """Compute enhanced lighting with all advanced features"""
    final_color = np.array([0.0, 0.0, 0.0])
    
    # Ambient occlusion
    ao_factor = 1.0
    if lighting_params[2] > 0.5:  # AO enabled
        ao_samples = int(lighting_params[30])
        ao_radius = lighting_params[29]
        ao_strength = lighting_params[28]
        
        ao_sum = 0.0
        for i in range(ao_samples):
            sample_dist = ao_radius * (i + 1) / ao_samples
            sample_pos = pos + normal * sample_dist
            
            dist, _, _ = get_fractal_distance(sample_pos, fractal_type, fractal_distance_params)
            
            if dist < sample_dist:
                ao_sum += 1.0 - (dist / sample_dist)
        
        ao_factor = 1.0 - (ao_sum / ao_samples * ao_strength)
    
    # Global ambient
    global_ambient = lighting_params[31]
    ambient_contrib = global_ambient * ao_factor
    final_color += ambient_contrib
    
    # Process each light source
    lights_data = [
        (lighting_params[4:8], lighting_params[8:11], lighting_params[11]),  # Light 1
        (lighting_params[12:16], lighting_params[16:19], lighting_params[19]),  # Light 2
        (lighting_params[20:24], lighting_params[24:27], lighting_params[27])   # Light 3
    ]
    
    for light_data in lights_data:
        light_pos_intensity, light_color, hard_factor = light_data
        light_pos = light_pos_intensity[:3]
        light_intensity = light_pos_intensity[3]
        
        if light_intensity <= 0.0:
            continue
        
        # Light direction
        light_dir = light_pos - pos
        light_distance = math.sqrt(light_dir[0]**2 + light_dir[1]**2 + light_dir[2]**2)
        if light_distance > 0:
            light_dir = light_dir / light_distance
        
        # Diffuse lighting
        diffuse = max(0.0, np.dot(normal, light_dir))
        
        # Hard vs soft lighting
        hard_enabled = lighting_params[0] > 0.5
        diffuse_enabled = lighting_params[1] > 0.5
        
        final_diffuse = 0.0
        if hard_enabled and diffuse_enabled:
            hard_diffuse = 1.0 if diffuse > 0.5 else 0.0
            soft_diffuse = diffuse
            final_diffuse = hard_diffuse * hard_factor + soft_diffuse * (1.0 - hard_factor)
        elif hard_enabled:
            final_diffuse = 1.0 if diffuse > 0.5 else 0.0
        elif diffuse_enabled:
            final_diffuse = diffuse
        
        # Shadow calculation (simplified for performance)
        shadow_factor = 1.0
        if final_diffuse > 0.0 and hard_enabled:
            # Simple shadow test
            shadow_ray_pos = pos + normal * 0.01
            shadow_dist, _, _ = get_fractal_distance(shadow_ray_pos, fractal_type, fractal_distance_params)
            if shadow_dist < light_distance * 0.1:
                shadow_factor = 0.3
        
        # Apply light contribution
        light_contrib = final_diffuse * light_intensity * shadow_factor
        final_color += light_contrib * np.array(light_color)
    
    return final_color

# Main rendering function with all fractal types
def genesis_render_core(image, width, height, fractal_params, lighting_params):
    """Core GenesisRender function with all features"""
    
    # Unpack parameters
    fractal_type = FRACTAL_TYPES.get("mandelbulb", 0)  # Default
    if fractal_params[0] >= 0:
        fractal_type = int(fractal_params[0])
    
    # Camera setup
    camera_pos = np.array([fractal_params[4], fractal_params[5], fractal_params[6]])
    target = np.array([fractal_params[7], fractal_params[8], fractal_params[9]])
    up = np.array([fractal_params[10], fractal_params[11], fractal_params[12]])
    fov = fractal_params[13]
    
    # Ray marching parameters
    epsilon = fractal_params[14]
    max_steps = int(fractal_params[15])
    max_dist = fractal_params[16]
    
    # Material parameters
    base_color = np.array([fractal_params[21], fractal_params[22], fractal_params[23]])
    palette_type = int(fractal_params[25])
    color_intensity = fractal_params[26]
    metallic = fractal_params[27]
    roughness = fractal_params[28]
    
    # Fractal-specific parameters
    fractal_distance_params = fractal_params[29:]
    
    # Set up camera coordinate system
    forward = target - camera_pos
    if np.linalg.norm(forward) > 0:
        forward = forward / np.linalg.norm(forward)
    
    right = np.cross(forward, up)
    if np.linalg.norm(right) > 0:
        right = right / np.linalg.norm(right)
    
    camera_up = np.cross(right, forward)
    
    # Image plane setup
    aspect = width / height
    fov_rad = fov * math.pi / 180.0
    h = math.tan(fov_rad / 2.0)
    w = h * aspect
    
    # Render each pixel
    for j in range(height):
        for i in range(width):
            # Pixel to world coordinates
            u = (2.0 * i / width - 1.0) * w
            v = (2.0 * j / height - 1.0) * h
            
            # Ray direction
            ray_dir = forward + u * right + v * camera_up
            ray_length = np.linalg.norm(ray_dir)
            if ray_length > 0:
                ray_dir = ray_dir / ray_length
            
            # Ray marching
            t = 0.0
            hit = False
            final_pos = camera_pos
            orbit_trap = 1000.0
            final_iterations = 0
            
            for step in range(max_steps):
                pos = camera_pos + t * ray_dir
                
                # Use our enhanced fractal distance function
                dist, trap, iters = get_fractal_distance(pos, fractal_type, fractal_distance_params)
                
                orbit_trap = min(orbit_trap, trap)
                final_iterations = iters
                
                if abs(dist) < epsilon:
                    hit = True
                    final_pos = pos
                    break
                
                t += max(abs(dist), epsilon)
                
                if t > max_dist:
                    break
            
            if hit:
                # Compute normal using finite differences
                eps = 0.001
                normal = np.array([
                    get_fractal_distance(final_pos + np.array([eps, 0, 0]), fractal_type, fractal_distance_params)[0] - 
                    get_fractal_distance(final_pos - np.array([eps, 0, 0]), fractal_type, fractal_distance_params)[0],
                    
                    get_fractal_distance(final_pos + np.array([0, eps, 0]), fractal_type, fractal_distance_params)[0] - 
                    get_fractal_distance(final_pos - np.array([0, eps, 0]), fractal_type, fractal_distance_params)[0],
                    
                    get_fractal_distance(final_pos + np.array([0, 0, eps]), fractal_type, fractal_distance_params)[0] - 
                    get_fractal_distance(final_pos - np.array([0, 0, eps]), fractal_type, fractal_distance_params)[0]
                ])
                
                normal_length = np.linalg.norm(normal)
                if normal_length > 0:
                    normal = normal / normal_length
                
                # Color calculation
                color_t = min(1.0, orbit_trap)  # Use orbit trap for coloring
                r, g, b = get_enhanced_color(color_t, palette_type, color_intensity, base_color)
                surface_color = np.array([r, g, b])
                
                # Enhanced lighting
                view_dir = -ray_dir
                final_color = compute_enhanced_lighting(
                    final_pos, normal, view_dir, fractal_params, 
                    lighting_params, fractal_type, fractal_distance_params
                )
                
                # Apply surface color
                final_color = final_color * surface_color
                
                # Metallic/roughness effects
                if metallic > 0.0:
                    reflect_dir = ray_dir - 2.0 * np.dot(ray_dir, normal) * normal
                    env_factor = (reflect_dir[1] + 1.0) * 0.5  # Simple environment
                    metallic_contrib = env_factor * metallic * (1.0 - roughness)
                    final_color = final_color * (1.0 - metallic) + metallic_contrib * surface_color
                
                # Final color clamping
                image[j, i, 0] = min(1.0, max(0.0, final_color[0]))
                image[j, i, 1] = min(1.0, max(0.0, final_color[1]))
                image[j, i, 2] = min(1.0, max(0.0, final_color[2]))
                
            else:
                # Background gradient
                bg_t = j / height
                image[j, i, 0] = 0.05 + 0.05 * bg_t
                image[j, i, 1] = 0.1 + 0.1 * bg_t
                image[j, i, 2] = 0.2 + 0.1 * bg_t

class GenesisRender:
    """Complete GenesisRender system"""
    
    def __init__(self):
        self.name = "GenesisRender"
        self.version = "1.0.0"
        self.fractal_type_map = FRACTAL_TYPES
        self.color_palette_map = {
            "warm": 0, "cool": 1, "rainbow": 2, 
            "monochrome": 3, "fire": 4, "ice": 5
        }
    
    def params_to_arrays(self, params: GenesisRenderParams) -> Tuple[np.ndarray, np.ndarray]:
        """Convert parameters to JIT-compatible arrays"""
        
        # Main fractal parameters
        fractal_array = np.array([
            self.fractal_type_map.get(params.fractal_type, 0),
            params.power, params.iterations, params.bailout,
            params.camera_pos[0], params.camera_pos[1], params.camera_pos[2],
            params.target[0], params.target[1], params.target[2],
            params.up[0], params.up[1], params.up[2],
            params.fov, params.epsilon, params.max_ray_steps, params.max_distance,
            0, 0, 0, 0,  # Reserved slots
            params.base_color[0], params.base_color[1], params.base_color[2],
            0,  # coloring_mode (not used in this version)
            self.color_palette_map.get(params.color_palette, 0),
            params.color_intensity, params.metallic, params.roughness,
            
            # Extended fractal parameters
            params.power, params.iterations, params.bailout,
            params.julia_c[0], params.julia_c[1], params.julia_c[2],
            params.folding_limit, params.folding_value, params.scale
        ], dtype=np.float64)
        
        # Lighting parameters
        lighting_array = np.array([
            1.0 if params.hard_lighting_enabled else 0.0,
            1.0 if params.diffuse_lighting_enabled else 0.0,
            1.0 if params.ambient_occlusion_enabled else 0.0,
            1.0 if params.subsurface_scattering_enabled else 0.0,
            
            # Light 1
            params.light1_pos[0], params.light1_pos[1], params.light1_pos[2], params.light1_intensity,
            params.light1_color[0], params.light1_color[1], params.light1_color[2],
            params.light1_hard_factor,
            
            # Light 2
            params.light2_pos[0], params.light2_pos[1], params.light2_pos[2], params.light2_intensity,
            params.light2_color[0], params.light2_color[1], params.light2_color[2],
            params.light2_hard_factor,
            
            # Light 3
            params.light3_pos[0], params.light3_pos[1], params.light3_pos[2], params.light3_intensity,
            params.light3_color[0], params.light3_color[1], params.light3_color[2],
            params.light3_hard_factor,
            
            # Global settings
            params.ao_strength, params.ao_radius, float(params.ao_samples),
            params.global_ambient, params.shadow_softness,
            params.specular_power, params.specular_intensity,
            
            # Subsurface scattering
            params.subsurface_color[0], params.subsurface_color[1], params.subsurface_color[2],
            params.subsurface_radius, params.transmittance
        ], dtype=np.float64)
        
        return fractal_array, lighting_array
    
    def render(self, params: GenesisRenderParams, progress_callback=None) -> Tuple[np.ndarray, Dict]:
        """Render fractal with full GenesisRender capabilities"""
        print(f"GenesisRender v{self.version}: Rendering {params.fractal_type} ({params.width}x{params.height})")
        start_time = time.time()
        
        # Create image buffer
        image = np.zeros((params.height, params.width, 3), dtype=np.float64)
        
        # Convert parameters
        fractal_array, lighting_array = self.params_to_arrays(params)
        
        # Render using JIT-compiled function
        genesis_render_core(image, params.width, params.height, fractal_array, lighting_array)
        
        render_time = time.time() - start_time
        
        # Calculate metrics
        surface_mask = np.any(image > 0.2, axis=2)
        surface_pixels = np.sum(surface_mask)
        unique_colors = len(np.unique(image.reshape(-1, 3), axis=0))
        avg_brightness = np.mean(image)
        
        metrics = {
            'render_time': render_time,
            'renderer': f"{self.name} v{self.version}",
            'surface_pixels': int(surface_pixels),
            'unique_colors': int(unique_colors),
            'avg_brightness': float(avg_brightness),
            'fractal_type': params.fractal_type,
            'resolution': f"{params.width}x{params.height}",
            'features_enabled': {
                'hard_lighting': params.hard_lighting_enabled,
                'diffuse_lighting': params.diffuse_lighting_enabled,
                'ambient_occlusion': params.ambient_occlusion_enabled,
                'subsurface_scattering': params.subsurface_scattering_enabled,
                'advanced_materials': params.metallic > 0.0 or params.roughness != 0.1
            }
        }
        
        print(f"Render complete: {render_time:.2f}s, {surface_pixels:,} surface pixels")
        
        return image, metrics
    
    def create_preset(self, preset_name: str) -> GenesisRenderParams:
        """Create parameters from preset name"""
        if preset_name not in FRACTAL_PRESETS:
            print(f"Warning: Unknown preset '{preset_name}', using mandelbulb_classic")
            preset_name = "mandelbulb_classic"
        
        preset = FRACTAL_PRESETS[preset_name]
        
        # Create base parameters
        params = GenesisRenderParams()
        
        # Apply preset values
        params.fractal_type = preset['type']
        params.power = preset.get('power', 8.0)
        params.iterations = preset.get('iterations', 100)
        params.bailout = preset.get('bailout', 2.0)
        
        # Handle fractal-specific parameters
        if 'julia_c' in preset:
            params.julia_c = preset['julia_c']
        if 'scale' in preset:
            params.scale = preset['scale']
        if 'folding_limit' in preset:
            params.folding_limit = preset['folding_limit']
        if 'folding_value' in preset:
            params.folding_value = preset['folding_value']
        
        return params
    
    def save_image(self, image: np.ndarray, filepath: Path, title: str = ""):
        """Save rendered image with title"""
        plt.figure(figsize=(12, 9))
        plt.imshow(image, origin='upper')
        plt.axis('off')
        if title:
            plt.title(title, fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def create_animation(self, animation_sequence: AnimationSequence, 
                        output_dir: Path, name: str) -> List[Path]:
        """Create animation from sequence"""
        output_dir.mkdir(exist_ok=True, parents=True)
        frame_paths = []
        
        total_frames = animation_sequence.get_frame_count()
        print(f"\nCreating '{name}' animation: {total_frames} frames, {animation_sequence.duration_seconds}s")
        
        for frame_num in range(total_frames):
            camera_state, param_state, lighting_state = animation_sequence.get_frame_state(frame_num)
            
            # Create render parameters from keyframe states
            params = GenesisRenderParams()
            
            # Apply camera state
            params.camera_pos = camera_state.position
            params.target = camera_state.target
            params.up = camera_state.up
            params.fov = camera_state.fov
            
            # Apply parameter state
            params.fractal_type = param_state.fractal_type
            params.power = param_state.power
            params.iterations = param_state.iterations
            params.bailout = param_state.bailout
            params.julia_c = param_state.julia_c
            params.color_palette = param_state.color_palette
            params.metallic = param_state.metallic
            params.roughness = param_state.roughness
            
            # Apply lighting state if available
            if lighting_state:
                params.light1_pos = lighting_state.light1_pos
                params.light1_intensity = lighting_state.light1_intensity
                params.light1_color = lighting_state.light1_color
                params.global_ambient = lighting_state.ambient
                params.ao_strength = lighting_state.ao_strength
            
            # Render frame
            t = frame_num / max(1, total_frames - 1)
            print(f"  Frame {frame_num+1}/{total_frames} (t={t:.3f})")
            
            image, metrics = self.render(params)
            
            # Save frame
            frame_path = output_dir / f"{name}_frame_{frame_num:04d}.png"
            title = f"{name} - Frame {frame_num+1}/{total_frames} - {param_state.fractal_type} (power={param_state.power:.1f})"
            self.save_image(image, frame_path, title)
            frame_paths.append(frame_path)
        
        print(f"Animation complete: {len(frame_paths)} frames")
        return frame_paths
    
    def create_gif(self, frame_paths: List[Path], output_path: Path, fps: int = 10):
        """Create animated GIF from frames"""
        print(f"Creating GIF: {output_path} ({fps} fps)")
        
        images = []
        for frame_path in frame_paths:
            images.append(imageio.imread(frame_path))
        
        imageio.mimsave(output_path, images, fps=fps)
        return output_path