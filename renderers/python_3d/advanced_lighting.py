#!/usr/bin/env python3
"""
Advanced Lighting System for Python 3D Fractal Renderer
======================================================

This extends the base fractal renderer with advanced lighting features
inspired by Mandelbulb3D, including:
- Hard lighting with sharp shadows
- Diffuse lighting with soft gradients  
- Ambient occlusion for depth
- Multiple light sources
- Advanced material properties
- Subsurface scattering simulation
"""

import numpy as np
from numba import jit, prange
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import matplotlib.pyplot as plt

# Import base classes
from .fractal_animator import (
    FractalParams, FractalRenderer, FractalPresets,
    mandelbulb_de, julia_set_de, get_distance_and_info,
    get_color_from_palette
)

@dataclass
class AdvancedLightingParams:
    """Advanced lighting configuration parameters"""
    # Lighting types
    hard_lighting_enabled: bool = True
    diffuse_lighting_enabled: bool = True
    ambient_occlusion_enabled: bool = True
    subsurface_scattering_enabled: bool = False
    
    # Light sources (up to 3 lights)
    light1_pos: Tuple[float, float, float] = (2.0, 2.0, -2.0)
    light1_intensity: float = 1.0
    light1_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    light1_hard_factor: float = 0.7  # 0=soft, 1=hard
    
    light2_pos: Tuple[float, float, float] = (-1.0, 1.0, -1.0)
    light2_intensity: float = 0.5
    light2_color: Tuple[float, float, float] = (0.8, 0.9, 1.0)
    light2_hard_factor: float = 0.3
    
    light3_pos: Tuple[float, float, float] = (0.0, -2.0, 1.0)
    light3_intensity: float = 0.3
    light3_color: Tuple[float, float, float] = (1.0, 0.8, 0.6)
    light3_hard_factor: float = 0.5
    
    # Ambient occlusion
    ao_strength: float = 0.3
    ao_radius: float = 0.1
    ao_samples: int = 5
    
    # Global lighting
    global_ambient: float = 0.15
    shadow_softness: float = 0.02
    specular_power: float = 32.0
    specular_intensity: float = 0.5
    
    # Advanced material
    subsurface_color: Tuple[float, float, float] = (1.0, 0.7, 0.5)
    subsurface_radius: float = 0.05
    transmittance: float = 0.1

@dataclass 
class AdvancedFractalParams(FractalParams):
    """Extended fractal parameters with advanced lighting"""
    lighting: AdvancedLightingParams = None
    
    def __post_init__(self):
        if self.lighting is None:
            self.lighting = AdvancedLightingParams()

# JIT-compiled advanced lighting functions
@jit(nopython=True, fastmath=True)
def compute_ambient_occlusion(pos, normal, fractal_type, power, iterations, bailout, julia_c,
                             ao_radius, ao_samples, ao_strength):
    """Compute ambient occlusion at a surface point"""
    ao_factor = 0.0
    
    for i in range(ao_samples):
        # Sample points along the normal
        sample_dist = ao_radius * (i + 1) / ao_samples
        sample_pos = pos + normal * sample_dist
        
        # Get distance to fractal surface
        dist, _, _ = get_distance_and_info(
            sample_pos[0], sample_pos[1], sample_pos[2],
            fractal_type, power, iterations, bailout, julia_c
        )
        
        # If the sample point is inside the fractal, it's occluded
        if dist < sample_dist:
            ao_factor += 1.0 - (dist / sample_dist)
    
    ao_factor /= ao_samples
    return 1.0 - (ao_factor * ao_strength)

@jit(nopython=True, fastmath=True)
def compute_hard_shadow(pos, light_dir, light_distance, fractal_type, power, 
                       iterations, bailout, julia_c, shadow_softness):
    """Compute hard shadow with optional softness"""
    shadow_factor = 1.0
    t = shadow_softness
    
    # March towards light source
    for step in range(32):  # Max shadow ray steps
        if t > light_distance:
            break
            
        sample_pos = pos + light_dir * t
        dist, _, _ = get_distance_and_info(
            sample_pos[0], sample_pos[1], sample_pos[2],
            fractal_type, power, iterations, bailout, julia_c
        )
        
        if dist < 0.001:  # Hit surface, in shadow
            shadow_factor = 0.1  # Not completely black
            break
        
        # Soft shadow calculation
        if shadow_softness > 0.0:
            shadow_factor = min(shadow_factor, 8.0 * dist / t)
        
        t += max(dist, 0.01)
    
    return max(0.1, shadow_factor)

@jit(nopython=True, fastmath=True)
def compute_subsurface_scattering(pos, normal, view_dir, light_dir, 
                                subsurface_color, subsurface_radius,
                                transmittance, fractal_type, power,
                                iterations, bailout, julia_c):
    """Simulate subsurface scattering"""
    # Sample point slightly inside the surface
    inside_pos = pos - normal * subsurface_radius
    
    # Check if light can penetrate to this depth
    dist, _, _ = get_distance_and_info(
        inside_pos[0], inside_pos[1], inside_pos[2],
        fractal_type, power, iterations, bailout, julia_c
    )
    
    if dist < 0:  # Inside the fractal
        # Calculate scattering based on light and view directions
        scatter_dot = max(0.0, -np.dot(light_dir, view_dir))
        scatter_factor = pow(scatter_dot, 4.0) * transmittance
        
        return (subsurface_color[0] * scatter_factor,
                subsurface_color[1] * scatter_factor,
                subsurface_color[2] * scatter_factor)
    
    return (0.0, 0.0, 0.0)

@jit(nopython=True, parallel=True, fastmath=True)
def render_advanced_fractal(image, width, height, params_array, lighting_array):
    """Advanced fractal rendering with Mandelbulb3D-style lighting"""
    # Unpack basic parameters (same as before)
    fractal_type = int(params_array[0])
    power = params_array[1]
    iterations = int(params_array[2])
    bailout = params_array[3]
    
    # Camera
    cam_x, cam_y, cam_z = params_array[4], params_array[5], params_array[6]
    target_x, target_y, target_z = params_array[7], params_array[8], params_array[9]
    up_x, up_y, up_z = params_array[10], params_array[11], params_array[12]
    fov = params_array[13]
    
    # Rendering
    epsilon = params_array[14]
    max_steps = int(params_array[15])
    max_dist = params_array[16]
    
    # Basic material
    base_r, base_g, base_b = params_array[21], params_array[22], params_array[23]
    coloring_mode = int(params_array[24])
    palette_type = int(params_array[25])
    color_intensity = params_array[26]
    metallic = params_array[27]
    roughness = params_array[28]
    julia_c = (params_array[29], params_array[30], params_array[31])
    
    # Unpack advanced lighting parameters
    hard_enabled = lighting_array[0] > 0.5
    diffuse_enabled = lighting_array[1] > 0.5
    ao_enabled = lighting_array[2] > 0.5
    sss_enabled = lighting_array[3] > 0.5
    
    # Light 1
    light1_x, light1_y, light1_z = lighting_array[4], lighting_array[5], lighting_array[6]
    light1_intensity = lighting_array[7]
    light1_r, light1_g, light1_b = lighting_array[8], lighting_array[9], lighting_array[10]
    light1_hard_factor = lighting_array[11]
    
    # Light 2
    light2_x, light2_y, light2_z = lighting_array[12], lighting_array[13], lighting_array[14]
    light2_intensity = lighting_array[15]
    light2_r, light2_g, light2_b = lighting_array[16], lighting_array[17], lighting_array[18]
    light2_hard_factor = lighting_array[19]
    
    # Light 3
    light3_x, light3_y, light3_z = lighting_array[20], lighting_array[21], lighting_array[22]
    light3_intensity = lighting_array[23]
    light3_r, light3_g, light3_b = lighting_array[24], lighting_array[25], lighting_array[26]
    light3_hard_factor = lighting_array[27]
    
    # AO and global settings
    ao_strength = lighting_array[28]
    ao_radius = lighting_array[29]
    ao_samples = int(lighting_array[30])
    global_ambient = lighting_array[31]
    shadow_softness = lighting_array[32]
    specular_power = lighting_array[33]
    specular_intensity = lighting_array[34]
    
    # Subsurface scattering
    sss_r, sss_g, sss_b = lighting_array[35], lighting_array[36], lighting_array[37]
    sss_radius = lighting_array[38]
    transmittance = lighting_array[39]
    
    # Set up camera coordinate system
    camera_pos = np.array([cam_x, cam_y, cam_z])
    target = np.array([target_x, target_y, target_z])
    up = np.array([up_x, up_y, up_z])
    
    forward = target - camera_pos
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    camera_up = np.cross(right, forward)
    
    # Image plane setup
    aspect = width / height
    fov_rad = fov * np.pi / 180.0
    h = np.tan(fov_rad / 2.0)
    w = h * aspect
    
    # Normalize light directions
    light1_dir = np.array([light1_x, light1_y, light1_z])
    light1_dir = light1_dir / np.linalg.norm(light1_dir)
    
    light2_dir = np.array([light2_x, light2_y, light2_z])
    light2_dir = light2_dir / np.linalg.norm(light2_dir)
    
    light3_dir = np.array([light3_x, light3_y, light3_z])
    light3_dir = light3_dir / np.linalg.norm(light3_dir)
    
    # Render each pixel
    for j in prange(height):
        for i in range(width):
            # Pixel to world coordinates
            u = (2.0 * i / width - 1.0) * w
            v = (2.0 * j / height - 1.0) * h
            
            # Ray direction
            ray_dir = forward + u * right + v * camera_up
            ray_dir = ray_dir / np.linalg.norm(ray_dir)
            
            # Ray marching
            t = 0.0
            hit = False
            final_pos = camera_pos
            orbit_trap = 1000.0
            final_iterations = 0
            
            for step in range(max_steps):
                pos = camera_pos + t * ray_dir
                dist, trap, iters = get_distance_and_info(
                    pos[0], pos[1], pos[2], fractal_type, power, iterations, bailout, julia_c
                )
                
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
                # Compute normal (same as before)
                eps = 0.001
                normal = np.array([
                    get_distance_and_info(final_pos[0] + eps, final_pos[1], final_pos[2], 
                                        fractal_type, power, iterations, bailout, julia_c)[0] - 
                    get_distance_and_info(final_pos[0] - eps, final_pos[1], final_pos[2], 
                                        fractal_type, power, iterations, bailout, julia_c)[0],
                    
                    get_distance_and_info(final_pos[0], final_pos[1] + eps, final_pos[2], 
                                        fractal_type, power, iterations, bailout, julia_c)[0] - 
                    get_distance_and_info(final_pos[0], final_pos[1] - eps, final_pos[2], 
                                        fractal_type, power, iterations, bailout, julia_c)[0],
                                        
                    get_distance_and_info(final_pos[0], final_pos[1], final_pos[2] + eps, 
                                        fractal_type, power, iterations, bailout, julia_c)[0] - 
                    get_distance_and_info(final_pos[0], final_pos[1], final_pos[2] - eps, 
                                        fractal_type, power, iterations, bailout, julia_c)[0]
                ])
                
                normal_length = np.linalg.norm(normal)
                if normal_length > 0:
                    normal = normal / normal_length
                
                # Base color calculation
                color_t = 0.5
                if coloring_mode == 1:  # orbit_trap
                    color_t = min(1.0, orbit_trap)
                elif coloring_mode == 2:  # distance
                    color_t = min(1.0, t / max_dist)
                elif coloring_mode == 3:  # normal
                    color_t = (normal[0] + normal[1] + normal[2]) / 3.0 + 0.5
                elif coloring_mode == 4:  # iteration
                    color_t = final_iterations / iterations
                
                r, g, b = get_color_from_palette(color_t, palette_type, color_intensity)
                r = r * 0.7 + base_r * 0.3
                g = g * 0.7 + base_g * 0.3
                b = b * 0.7 + base_b * 0.3
                
                # Advanced lighting calculation
                final_r, final_g, final_b = 0.0, 0.0, 0.0
                
                # Ambient occlusion
                ao_factor = 1.0
                if ao_enabled:
                    ao_factor = compute_ambient_occlusion(
                        final_pos, normal, fractal_type, power, iterations, bailout, julia_c,
                        ao_radius, ao_samples, ao_strength
                    )
                
                # Global ambient
                ambient_contrib = global_ambient * ao_factor
                final_r += r * ambient_contrib
                final_g += g * ambient_contrib
                final_b += b * ambient_contrib
                
                # Light sources
                lights = [
                    (light1_dir, light1_intensity, (light1_r, light1_g, light1_b), light1_hard_factor),
                    (light2_dir, light2_intensity, (light2_r, light2_g, light2_b), light2_hard_factor),
                    (light3_dir, light3_intensity, (light3_r, light3_g, light3_b), light3_hard_factor)
                ]
                
                view_dir = -ray_dir
                
                for light_dir, intensity, light_color, hard_factor in lights:
                    if intensity <= 0.0:
                        continue
                    
                    # Basic diffuse
                    diffuse = max(0.0, np.dot(normal, light_dir))
                    
                    # Shadow calculation
                    shadow_factor = 1.0
                    if hard_enabled and diffuse > 0.0:
                        light_distance = 5.0  # Assume lights are far away
                        shadow_factor = compute_hard_shadow(
                            final_pos + normal * 0.01,  # Offset to avoid self-intersection
                            light_dir, light_distance, fractal_type, power,
                            iterations, bailout, julia_c, shadow_softness
                        )
                    
                    # Hard vs soft lighting blend
                    if hard_enabled and diffuse_enabled:
                        hard_diffuse = diffuse if diffuse > 0.5 else 0.0  # Sharp cutoff
                        soft_diffuse = diffuse  # Smooth falloff
                        final_diffuse = hard_diffuse * hard_factor + soft_diffuse * (1.0 - hard_factor)
                    elif hard_enabled:
                        final_diffuse = diffuse if diffuse > 0.5 else 0.0
                    elif diffuse_enabled:
                        final_diffuse = diffuse
                    else:
                        final_diffuse = 0.0
                    
                    # Apply lighting
                    light_contrib = final_diffuse * intensity * shadow_factor
                    
                    final_r += r * light_contrib * light_color[0]
                    final_g += g * light_contrib * light_color[1]
                    final_b += b * light_contrib * light_color[2]
                    
                    # Specular highlights
                    if specular_intensity > 0.0 and light_contrib > 0.0:
                        reflect_dir = ray_dir - 2.0 * np.dot(ray_dir, normal) * normal
                        spec_factor = max(0.0, np.dot(reflect_dir, light_dir))
                        spec_factor = pow(spec_factor, specular_power) * specular_intensity
                        
                        final_r += spec_factor * light_color[0] * intensity
                        final_g += spec_factor * light_color[1] * intensity
                        final_b += spec_factor * light_color[2] * intensity
                    
                    # Subsurface scattering
                    if sss_enabled and transmittance > 0.0:
                        sss_color = compute_subsurface_scattering(
                            final_pos, normal, view_dir, light_dir,
                            (sss_r, sss_g, sss_b), sss_radius, transmittance,
                            fractal_type, power, iterations, bailout, julia_c
                        )
                        
                        final_r += sss_color[0] * intensity
                        final_g += sss_color[1] * intensity
                        final_b += sss_color[2] * intensity
                
                # Metallic/roughness (simplified PBR)
                if metallic > 0.0:
                    reflect_dir = ray_dir - 2.0 * np.dot(ray_dir, normal) * normal
                    # Simple environment reflection (could be enhanced)
                    env_factor = (reflect_dir[1] + 1.0) * 0.5  # Simple sky gradient
                    metallic_contrib = env_factor * metallic * (1.0 - roughness)
                    
                    final_r = final_r * (1.0 - metallic) + metallic_contrib
                    final_g = final_g * (1.0 - metallic) + metallic_contrib * 0.9
                    final_b = final_b * (1.0 - metallic) + metallic_contrib * 0.8
                
                # Final clamping
                image[j, i, 0] = min(1.0, max(0.0, final_r))
                image[j, i, 1] = min(1.0, max(0.0, final_g))
                image[j, i, 2] = min(1.0, max(0.0, final_b))
                
            else:
                # Background gradient
                bg_t = j / height
                image[j, i, 0] = 0.05 + 0.05 * bg_t
                image[j, i, 1] = 0.1 + 0.1 * bg_t
                image[j, i, 2] = 0.2 + 0.1 * bg_t

class AdvancedFractalRenderer(FractalRenderer):
    """Enhanced fractal renderer with Mandelbulb3D-style advanced lighting"""
    
    def lighting_to_array(self, lighting: AdvancedLightingParams) -> np.ndarray:
        """Convert lighting parameters to array for JIT function"""
        return np.array([
            # Flags
            1.0 if lighting.hard_lighting_enabled else 0.0,
            1.0 if lighting.diffuse_lighting_enabled else 0.0,
            1.0 if lighting.ambient_occlusion_enabled else 0.0,
            1.0 if lighting.subsurface_scattering_enabled else 0.0,
            
            # Light 1
            lighting.light1_pos[0], lighting.light1_pos[1], lighting.light1_pos[2],
            lighting.light1_intensity,
            lighting.light1_color[0], lighting.light1_color[1], lighting.light1_color[2],
            lighting.light1_hard_factor,
            
            # Light 2
            lighting.light2_pos[0], lighting.light2_pos[1], lighting.light2_pos[2],
            lighting.light2_intensity,
            lighting.light2_color[0], lighting.light2_color[1], lighting.light2_color[2],
            lighting.light2_hard_factor,
            
            # Light 3
            lighting.light3_pos[0], lighting.light3_pos[1], lighting.light3_pos[2],
            lighting.light3_intensity,
            lighting.light3_color[0], lighting.light3_color[1], lighting.light3_color[2],
            lighting.light3_hard_factor,
            
            # AO and global
            lighting.ao_strength,
            lighting.ao_radius,
            float(lighting.ao_samples),
            lighting.global_ambient,
            lighting.shadow_softness,
            lighting.specular_power,
            lighting.specular_intensity,
            
            # Subsurface scattering
            lighting.subsurface_color[0], lighting.subsurface_color[1], lighting.subsurface_color[2],
            lighting.subsurface_radius,
            lighting.transmittance
        ], dtype=np.float64)
    
    def render_advanced(self, params: AdvancedFractalParams) -> Tuple[np.ndarray, Dict]:
        """Render fractal with advanced lighting"""
        print(f"Rendering {params.fractal_type} fractal with advanced lighting ({params.width}x{params.height})...")
        start_time = time.time()
        
        # Create image buffer
        image = np.zeros((params.height, params.width, 3), dtype=np.float64)
        
        # Convert parameters to arrays
        params_array = self.params_to_array(params)
        lighting_array = self.lighting_to_array(params.lighting)
        
        # Render with advanced lighting
        render_advanced_fractal(image, params.width, params.height, params_array, lighting_array)
        
        render_time = time.time() - start_time
        print(f"Advanced render completed in {render_time:.2f}s")
        
        # Calculate metrics
        surface_mask = np.any(image > 0.2, axis=2)
        surface_pixels = np.sum(surface_mask)
        unique_colors = len(np.unique(image.reshape(-1, 3), axis=0))
        avg_brightness = np.mean(image)
        
        metrics = {
            'render_time': render_time,
            'surface_pixels': surface_pixels,
            'unique_colors': unique_colors,
            'avg_brightness': avg_brightness,
            'fractal_type': params.fractal_type,
            'power': params.power,
            'resolution': f"{params.width}x{params.height}",
            'lighting_features': {
                'hard_lighting': params.lighting.hard_lighting_enabled,
                'diffuse_lighting': params.lighting.diffuse_lighting_enabled,
                'ambient_occlusion': params.lighting.ambient_occlusion_enabled,
                'subsurface_scattering': params.lighting.subsurface_scattering_enabled,
                'num_lights': sum([
                    1 if params.lighting.light1_intensity > 0 else 0,
                    1 if params.lighting.light2_intensity > 0 else 0,
                    1 if params.lighting.light3_intensity > 0 else 0
                ])
            }
        }
        
        return image, metrics

# Preset lighting configurations
class AdvancedLightingPresets:
    """Predefined advanced lighting configurations"""
    
    @staticmethod
    def mandelbulb3d_classic() -> AdvancedLightingParams:
        """Classic Mandelbulb3D-style lighting"""
        return AdvancedLightingParams(
            hard_lighting_enabled=True,
            diffuse_lighting_enabled=True,
            ambient_occlusion_enabled=True,
            
            light1_pos=(2.0, 2.0, -2.0),
            light1_intensity=1.0,
            light1_color=(1.0, 1.0, 0.95),
            light1_hard_factor=0.8,
            
            light2_pos=(-1.0, 1.0, -1.0),
            light2_intensity=0.4,
            light2_color=(0.7, 0.8, 1.0),
            light2_hard_factor=0.2,
            
            ao_strength=0.4,
            ao_radius=0.1,
            global_ambient=0.1,
            shadow_softness=0.02,
            specular_power=64.0,
            specular_intensity=0.3
        )
    
    @staticmethod
    def soft_artistic() -> AdvancedLightingParams:
        """Soft artistic lighting"""
        return AdvancedLightingParams(
            hard_lighting_enabled=False,
            diffuse_lighting_enabled=True,
            ambient_occlusion_enabled=True,
            subsurface_scattering_enabled=True,
            
            light1_pos=(1.0, 3.0, -1.0),
            light1_intensity=0.8,
            light1_color=(1.0, 0.95, 0.9),
            light1_hard_factor=0.0,
            
            light2_pos=(-2.0, 0.0, 1.0),
            light2_intensity=0.6,
            light2_color=(0.9, 0.9, 1.0),
            light2_hard_factor=0.0,
            
            ao_strength=0.25,
            global_ambient=0.2,
            specular_power=16.0,
            specular_intensity=0.1,
            
            subsurface_color=(1.0, 0.8, 0.6),
            transmittance=0.15
        )
    
    @staticmethod
    def dramatic_contrast() -> AdvancedLightingParams:
        """High contrast dramatic lighting"""
        return AdvancedLightingParams(
            hard_lighting_enabled=True,
            diffuse_lighting_enabled=False,
            ambient_occlusion_enabled=True,
            
            light1_pos=(3.0, 1.0, -1.0),
            light1_intensity=1.2,
            light1_color=(1.0, 0.9, 0.8),
            light1_hard_factor=1.0,
            
            light2_pos=(-1.0, -1.0, 2.0),
            light2_intensity=0.3,
            light2_color=(0.8, 0.9, 1.0),
            light2_hard_factor=0.8,
            
            ao_strength=0.6,
            global_ambient=0.05,
            shadow_softness=0.001,
            specular_power=128.0,
            specular_intensity=0.8
        )
    
    @staticmethod
    def three_point_studio() -> AdvancedLightingParams:
        """Professional three-point lighting setup"""
        return AdvancedLightingParams(
            hard_lighting_enabled=True,
            diffuse_lighting_enabled=True,
            ambient_occlusion_enabled=True,
            
            # Key light
            light1_pos=(2.0, 2.0, -2.0),
            light1_intensity=1.0,
            light1_color=(1.0, 0.98, 0.95),
            light1_hard_factor=0.7,
            
            # Fill light
            light2_pos=(-1.5, 1.0, -1.0),
            light2_intensity=0.5,
            light2_color=(0.95, 0.95, 1.0),
            light2_hard_factor=0.3,
            
            # Rim light
            light3_pos=(0.0, -1.0, 2.0),
            light3_intensity=0.6,
            light3_color=(1.0, 0.9, 0.8),
            light3_hard_factor=0.9,
            
            ao_strength=0.3,
            global_ambient=0.12,
            specular_power=32.0,
            specular_intensity=0.4
        )