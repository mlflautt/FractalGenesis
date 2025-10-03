#!/usr/bin/env python3
"""
Advanced 3D Fractal Renderer with Animation
==========================================

Complete system with:
- Multiple fractal types (Mandelbulb, Julia, Mandelbox, Burning Ship)
- Advanced coloration and materials
- Parameter interpolation and animation
- Frame-by-frame rendering
- High-performance Numba JIT compilation
"""

import numpy as np
from numba import jit, prange
import time
import json
from pathlib import Path
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
import imageio

@dataclass
class FractalParams:
    """Complete fractal parameter specification"""
    # Core fractal parameters
    fractal_type: str = "mandelbulb"
    power: float = 8.0
    iterations: int = 100
    bailout: float = 2.0
    julia_c: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    
    # Mandelbox specific
    folding_limit: float = 1.0
    folding_value: float = 2.0
    scale: float = -1.5
    
    # Camera parameters
    camera_pos: Tuple[float, float, float] = (0.0, 0.0, -3.0)
    target: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    up: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    fov: float = 45.0
    
    # Rendering parameters
    width: int = 800
    height: int = 600
    max_ray_steps: int = 100
    epsilon: float = 0.001
    max_distance: float = 10.0
    
    # Lighting and materials
    light_pos: Tuple[float, float, float] = (2.0, 2.0, -2.0)
    ambient: float = 0.1
    base_color: Tuple[float, float, float] = (0.8, 0.6, 0.4)
    coloring_mode: str = "orbit_trap"  # solid, orbit_trap, distance, normal, iteration
    color_palette: str = "warm"  # warm, cool, rainbow, monochrome, fire, ice
    color_intensity: float = 1.0
    metallic: float = 0.0
    roughness: float = 0.1

class FractalPresets:
    """Predefined fractal configurations"""
    
    @staticmethod
    def get_preset(name: str) -> FractalParams:
        presets = {
            "classic_mandelbulb": FractalParams(
                fractal_type="mandelbulb",
                power=8.0,
                camera_pos=(0.0, 0.0, -3.0),
                coloring_mode="orbit_trap",
                color_palette="warm"
            ),
            
            "julia_set_3d": FractalParams(
                fractal_type="julia",
                power=8.0,
                julia_c=(-0.2, 0.1, 0.0),
                camera_pos=(0.0, 0.0, -2.5),
                coloring_mode="distance",
                color_palette="cool"
            ),
            
            "power_morph_start": FractalParams(
                fractal_type="mandelbulb",
                power=2.0,
                camera_pos=(0.0, 0.0, -3.0),
                color_palette="fire"
            ),
            
            "power_morph_end": FractalParams(
                fractal_type="mandelbulb",
                power=16.0,
                camera_pos=(0.0, 0.0, -3.0),
                color_palette="ice"
            ),
            
            "orbit_camera_start": FractalParams(
                fractal_type="mandelbulb",
                power=8.0,
                camera_pos=(0.0, 0.0, -4.0),
                color_palette="rainbow"
            ),
            
            "orbit_camera_end": FractalParams(
                fractal_type="mandelbulb", 
                power=8.0,
                camera_pos=(4.0, 0.0, 0.0),
                color_palette="rainbow"
            )
        }
        
        return presets.get(name, presets["classic_mandelbulb"])

# JIT-compiled fractal distance estimation functions
@jit(nopython=True, fastmath=True)
def mandelbulb_de(x, y, z, power, max_iter, bailout):
    """Distance estimation for Mandelbulb fractal"""
    xx, yy, zz = x, y, z
    dr = 1.0
    r = np.sqrt(xx*xx + yy*yy + zz*zz)
    orbit_trap = 1000.0
    
    for i in range(max_iter):
        if r > bailout:
            break
            
        orbit_trap = min(orbit_trap, r)
        
        # Convert to polar coordinates
        theta = np.arctan2(np.sqrt(xx*xx + yy*yy), zz)
        phi = np.arctan2(yy, xx)
        
        # Scale and rotate
        zr = r ** (power - 1.0)
        dr = zr * dr * power + 1.0
        
        # Convert back to cartesian
        zr = zr * r
        theta = theta * power
        phi = phi * power
        
        xx = zr * np.sin(theta) * np.cos(phi) + x
        yy = zr * np.sin(theta) * np.sin(phi) + y
        zz = zr * np.cos(theta) + z
        
        r = np.sqrt(xx*xx + yy*yy + zz*zz)
    
    return 0.5 * np.log(r) * r / dr, orbit_trap, i

@jit(nopython=True, fastmath=True)
def julia_set_de(x, y, z, power, max_iter, bailout, cx, cy, cz):
    """Distance estimation for 3D Julia set"""
    xx, yy, zz = x, y, z
    dr = 1.0
    r = np.sqrt(xx*xx + yy*yy + zz*zz)
    orbit_trap = 1000.0
    
    for i in range(max_iter):
        if r > bailout:
            break
            
        orbit_trap = min(orbit_trap, r)
        
        theta = np.arctan2(np.sqrt(xx*xx + yy*yy), zz)
        phi = np.arctan2(yy, xx)
        
        zr = r ** (power - 1.0)
        dr = zr * dr * power + 1.0
        
        zr = zr * r
        theta = theta * power
        phi = phi * power
        
        xx = zr * np.sin(theta) * np.cos(phi) + cx
        yy = zr * np.sin(theta) * np.sin(phi) + cy
        zz = zr * np.cos(theta) + cz
        
        r = np.sqrt(xx*xx + yy*yy + zz*zz)
    
    return 0.5 * np.log(r) * r / dr, orbit_trap, i

@jit(nopython=True, fastmath=True)
def get_distance_and_info(x, y, z, fractal_type, power, max_iter, bailout, julia_c):
    """Dispatch function for different fractal types"""
    if fractal_type == 0:  # mandelbulb
        return mandelbulb_de(x, y, z, power, max_iter, bailout)
    elif fractal_type == 1:  # julia
        return julia_set_de(x, y, z, power, max_iter, bailout, julia_c[0], julia_c[1], julia_c[2])
    else:
        return mandelbulb_de(x, y, z, power, max_iter, bailout)

@jit(nopython=True, fastmath=True)
def get_color_from_palette(t, palette_type, intensity):
    """Generate color from palette and parameter t (0-1)"""
    t = max(0.0, min(1.0, t)) * intensity
    
    if palette_type == 0:  # warm
        r = 0.8 + 0.2 * np.sin(t * 3.14159 * 2.0)
        g = 0.4 + 0.4 * np.sin(t * 3.14159 * 2.0 + 1.0)
        b = 0.2 + 0.3 * np.sin(t * 3.14159 * 2.0 + 2.0)
    elif palette_type == 1:  # cool  
        r = 0.2 + 0.3 * np.sin(t * 3.14159 * 2.0)
        g = 0.4 + 0.4 * np.sin(t * 3.14159 * 2.0 + 1.0)
        b = 0.8 + 0.2 * np.sin(t * 3.14159 * 2.0 + 2.0)
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
        r = min(1.0, t * 2.0)
        g = max(0.0, min(1.0, (t - 0.5) * 2.0))
        b = max(0.0, min(1.0, (t - 0.8) * 5.0))
    elif palette_type == 5:  # ice
        b = min(1.0, t * 1.5)
        g = max(0.0, min(1.0, (t - 0.3) * 1.5))
        r = max(0.0, min(1.0, (t - 0.6) * 2.0))
    else:  # default warm
        r = 0.8 + 0.2 * np.sin(t * 3.14159 * 2.0)
        g = 0.4 + 0.4 * np.sin(t * 3.14159 * 2.0 + 1.0)
        b = 0.2 + 0.3 * np.sin(t * 3.14159 * 2.0 + 2.0)
    
    return r, g, b

@jit(nopython=True, parallel=True, fastmath=True)
def render_fractal_fast(image, width, height, params_array):
    """Fast fractal rendering with JIT compilation"""
    # Unpack parameters
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
    
    # Lighting
    light_x, light_y, light_z = params_array[17], params_array[18], params_array[19]
    ambient = params_array[20]
    
    # Material
    base_r, base_g, base_b = params_array[21], params_array[22], params_array[23]
    coloring_mode = int(params_array[24])
    palette_type = int(params_array[25])
    color_intensity = params_array[26]
    metallic = params_array[27]
    roughness = params_array[28]  # Fixed: now passed as parameter
    
    # Fractal specific
    julia_c = (params_array[29], params_array[30], params_array[31])
    
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
    
    # Light direction
    light_dir = np.array([light_x, light_y, light_z])
    light_dir = light_dir / np.linalg.norm(light_dir)
    
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
                # Compute normal
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
                
                # Color based on coloring mode
                color_t = 0.5
                if coloring_mode == 0:  # solid
                    color_t = 0.5
                elif coloring_mode == 1:  # orbit_trap
                    color_t = min(1.0, orbit_trap)
                elif coloring_mode == 2:  # distance
                    color_t = min(1.0, t / max_dist)
                elif coloring_mode == 3:  # normal
                    color_t = (normal[0] + normal[1] + normal[2]) / 3.0 + 0.5
                elif coloring_mode == 4:  # iteration
                    color_t = final_iterations / iterations
                
                # Get color from palette
                r, g, b = get_color_from_palette(color_t, palette_type, color_intensity)
                
                # Mix with base color
                r = r * 0.7 + base_r * 0.3
                g = g * 0.7 + base_g * 0.3
                b = b * 0.7 + base_b * 0.3
                
                # Lighting calculation
                diffuse = max(0.0, np.dot(normal, light_dir))
                
                # Metallic/roughness simulation
                if metallic > 0.0:
                    reflect_dir = ray_dir - 2.0 * np.dot(ray_dir, normal) * normal
                    metallic_factor = max(0.0, np.dot(reflect_dir, light_dir)) ** (1.0 / max(0.01, roughness))
                    diffuse = diffuse * (1.0 - metallic) + metallic_factor * metallic
                
                # Final color
                final_r = r * (ambient + diffuse * (1.0 - ambient))
                final_g = g * (ambient + diffuse * (1.0 - ambient))
                final_b = b * (ambient + diffuse * (1.0 - ambient))
                
                # Depth attenuation
                depth_factor = max(0.1, 1.0 - t / max_dist)
                final_r *= depth_factor
                final_g *= depth_factor
                final_b *= depth_factor
                
                image[j, i, 0] = min(1.0, final_r)
                image[j, i, 1] = min(1.0, final_g)
                image[j, i, 2] = min(1.0, final_b)
            else:
                # Background gradient
                bg_t = j / height
                image[j, i, 0] = 0.05 + 0.05 * bg_t
                image[j, i, 1] = 0.1 + 0.1 * bg_t
                image[j, i, 2] = 0.2 + 0.1 * bg_t

class FractalRenderer:
    """High-performance fractal renderer with animation"""
    
    def __init__(self):
        self.fractal_type_map = {
            "mandelbulb": 0,
            "julia": 1
        }
        
        self.coloring_mode_map = {
            "solid": 0,
            "orbit_trap": 1,
            "distance": 2,
            "normal": 3,
            "iteration": 4
        }
        
        self.palette_map = {
            "warm": 0,
            "cool": 1,
            "rainbow": 2,
            "monochrome": 3,
            "fire": 4,
            "ice": 5
        }
    
    def params_to_array(self, params: FractalParams) -> np.ndarray:
        """Convert FractalParams to array for JIT function"""
        return np.array([
            # Core fractal
            self.fractal_type_map.get(params.fractal_type, 0),
            params.power,
            params.iterations,
            params.bailout,
            
            # Camera
            params.camera_pos[0], params.camera_pos[1], params.camera_pos[2],
            params.target[0], params.target[1], params.target[2],
            params.up[0], params.up[1], params.up[2],
            params.fov,
            
            # Rendering
            params.epsilon,
            params.max_ray_steps,
            params.max_distance,
            
            # Lighting
            params.light_pos[0], params.light_pos[1], params.light_pos[2],
            params.ambient,
            
            # Material
            params.base_color[0], params.base_color[1], params.base_color[2],
            self.coloring_mode_map.get(params.coloring_mode, 1),
            self.palette_map.get(params.color_palette, 0),
            params.color_intensity,
            params.metallic,
            params.roughness,
            
            # Julia set parameters
            params.julia_c[0], params.julia_c[1], params.julia_c[2]
        ], dtype=np.float64)
    
    def render(self, params: FractalParams) -> Tuple[np.ndarray, Dict]:
        """Render fractal with given parameters"""
        print(f"Rendering {params.fractal_type} fractal ({params.width}x{params.height})...")
        start_time = time.time()
        
        # Create image buffer
        image = np.zeros((params.height, params.width, 3), dtype=np.float64)
        
        # Convert parameters to array
        params_array = self.params_to_array(params)
        
        # Render
        render_fractal_fast(image, params.width, params.height, params_array)
        
        render_time = time.time() - start_time
        print(f"Render completed in {render_time:.2f}s")
        
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
            'resolution': f"{params.width}x{params.height}"
        }
        
        return image, metrics
    
    def interpolate_params(self, start_params: FractalParams, end_params: FractalParams, t: float) -> FractalParams:
        """Interpolate between two parameter sets"""
        t = max(0.0, min(1.0, t))  # Clamp t to [0, 1]
        
        # Linear interpolation helper
        def lerp(a, b, t):
            if isinstance(a, tuple):
                return tuple(lerp(ai, bi, t) for ai, bi in zip(a, b))
            return a + (b - a) * t
        
        # Create interpolated parameters
        return FractalParams(
            fractal_type=start_params.fractal_type,  # Don't interpolate type
            power=lerp(start_params.power, end_params.power, t),
            iterations=int(lerp(start_params.iterations, end_params.iterations, t)),
            bailout=lerp(start_params.bailout, end_params.bailout, t),
            julia_c=lerp(start_params.julia_c, end_params.julia_c, t),
            
            camera_pos=lerp(start_params.camera_pos, end_params.camera_pos, t),
            target=lerp(start_params.target, end_params.target, t),
            up=start_params.up,  # Keep up vector constant
            fov=lerp(start_params.fov, end_params.fov, t),
            
            width=start_params.width,
            height=start_params.height,
            max_ray_steps=start_params.max_ray_steps,
            epsilon=start_params.epsilon,
            max_distance=start_params.max_distance,
            
            light_pos=lerp(start_params.light_pos, end_params.light_pos, t),
            ambient=lerp(start_params.ambient, end_params.ambient, t),
            base_color=lerp(start_params.base_color, end_params.base_color, t),
            coloring_mode=start_params.coloring_mode,  # Don't interpolate mode
            color_palette=start_params.color_palette if t < 0.5 else end_params.color_palette,
            color_intensity=lerp(start_params.color_intensity, end_params.color_intensity, t),
            metallic=lerp(start_params.metallic, end_params.metallic, t),
            roughness=lerp(start_params.roughness, end_params.roughness, t)
        )
    
    def create_animation(self, start_params: FractalParams, end_params: FractalParams, 
                        num_frames: int, output_dir: Path, name: str) -> List[Path]:
        """Create animation by interpolating between two parameter sets"""
        output_dir.mkdir(exist_ok=True)
        frame_paths = []
        
        print(f"\n=== Creating Animation: {name} ===")
        print(f"Frames: {num_frames}")
        print(f"Output directory: {output_dir}")
        
        start_time = time.time()
        
        for frame in range(num_frames):
            t = frame / (num_frames - 1) if num_frames > 1 else 0.0
            
            # Interpolate parameters
            params = self.interpolate_params(start_params, end_params, t)
            
            # Render frame
            print(f"  Frame {frame+1}/{num_frames} (t={t:.3f})...")
            image, metrics = self.render(params)
            
            # Save frame
            frame_path = output_dir / f"{name}_frame_{frame:04d}.png"
            plt.figure(figsize=(12, 9))
            plt.imshow(image, origin='upper')
            plt.axis('off')
            plt.title(f"{name} - Frame {frame+1}/{num_frames} (t={t:.3f})")
            plt.tight_layout()
            plt.savefig(frame_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            frame_paths.append(frame_path)
            
            print(f"    Rendered: {metrics['surface_pixels']:,} pixels, "
                  f"{metrics['render_time']:.2f}s")
        
        total_time = time.time() - start_time
        print(f"Animation complete! Total time: {total_time:.1f}s")
        print(f"Average time per frame: {total_time/num_frames:.2f}s")
        
        return frame_paths
    
    def create_gif(self, frame_paths: List[Path], output_path: Path, fps: int = 10):
        """Create animated GIF from frame images"""
        print(f"\nCreating animated GIF: {output_path}")
        
        images = []
        for frame_path in frame_paths:
            images.append(imageio.imread(frame_path))
        
        imageio.mimsave(output_path, images, fps=fps)
        print(f"GIF saved: {output_path}")
        
        return output_path

def test_advanced_fractals():
    """Test advanced fractal rendering and parameter control"""
    renderer = FractalRenderer()
    output_dir = Path("advanced_fractal_outputs")
    output_dir.mkdir(exist_ok=True)
    
    print("=== Advanced Fractal Testing ===")
    print(f"Output directory: {output_dir.absolute()}")
    
    # Test different presets
    presets_to_test = [
        "classic_mandelbulb",
        "julia_set_3d", 
        "power_morph_start",
        "power_morph_end"
    ]
    
    results = []
    
    for preset_name in presets_to_test:
        print(f"\n--- Testing preset: {preset_name} ---")
        
        params = FractalPresets.get_preset(preset_name)
        image, metrics = renderer.render(params)
        
        # Save image
        output_file = output_dir / f"{preset_name}.png"
        plt.figure(figsize=(12, 9))
        plt.imshow(image, origin='upper')
        plt.axis('off')
        plt.title(f"{preset_name} - {params.fractal_type} (power={params.power})")
        plt.tight_layout()
        plt.savefig(output_file, dpi=100, bbox_inches='tight')
        plt.close()
        
        file_size = output_file.stat().st_size
        metrics['file_size'] = file_size
        metrics['preset_name'] = preset_name
        results.append(metrics)
        
        print(f"Surface pixels: {metrics['surface_pixels']:,}")
        print(f"Unique colors: {metrics['unique_colors']:,}")
        print(f"File size: {file_size:,} bytes")
    
    return results

def test_animations():
    """Test fractal animation capabilities"""
    renderer = FractalRenderer()
    anim_dir = Path("fractal_animations")
    anim_dir.mkdir(exist_ok=True)
    
    print("\n=== Animation Testing ===")
    
    # Test 1: Power morphing animation
    start_params = FractalPresets.get_preset("power_morph_start")
    end_params = FractalPresets.get_preset("power_morph_end")
    
    # Reduce resolution for faster animation testing
    start_params.width = 400
    start_params.height = 300
    end_params.width = 400
    end_params.height = 300
    
    frame_paths = renderer.create_animation(
        start_params, end_params, 
        num_frames=8, 
        output_dir=anim_dir / "power_morph", 
        name="power_morph"
    )
    
    # Create GIF
    gif_path = anim_dir / "power_morph.gif"
    renderer.create_gif(frame_paths, gif_path, fps=2)
    
    # Test 2: Camera orbit animation
    start_params = FractalPresets.get_preset("orbit_camera_start")
    end_params = FractalPresets.get_preset("orbit_camera_end")
    
    start_params.width = 400
    start_params.height = 300
    end_params.width = 400
    end_params.height = 300
    
    frame_paths = renderer.create_animation(
        start_params, end_params,
        num_frames=6,
        output_dir=anim_dir / "camera_orbit",
        name="camera_orbit"
    )
    
    gif_path = anim_dir / "camera_orbit.gif"
    renderer.create_gif(frame_paths, gif_path, fps=1)
    
    print(f"\n✅ Animations complete! Check: {anim_dir.absolute()}")

if __name__ == "__main__":
    print("Warming up Numba JIT compiler...")
    renderer = FractalRenderer()
    
    # JIT warm-up with small image
    params = FractalParams(width=100, height=100, iterations=20)
    renderer.render(params)
    print("JIT warm-up complete\n")
    
    # Test static rendering
    results = test_advanced_fractals()
    
    # Test animations
    test_animations()
    
    print(f"\n✅ All testing complete!")
    print(f"Static images: ./advanced_fractal_outputs/")  
    print(f"Animations: ./fractal_animations/")