#!/usr/bin/env python3
"""
Advanced Numba JIT 3D Fractal Renderer
=====================================

Complete fractal rendering system with:
- Multiple fractal types (Mandelbulb, Julia sets, Mandelbox, etc.)
- Advanced coloration schemes
- Full parameter control and presets
- Frame-by-frame animation capabilities
- High-performance JIT compilation
"""

import numpy as np
from numba import jit, prange
import time
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
import imageio

@dataclass
class FractalParams:
    """Complete fractal parameter specification"""
    # Core fractal parameters
    fractal_type: str = "mandelbulb"  # mandelbulb, julia, mandelbox, burning_ship
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
    
    # Lighting
    light_pos: Tuple[float, float, float] = (2.0, 2.0, -2.0)
    light_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    ambient: float = 0.1
    
    # Material and coloring
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
            
            "mandelbox_classic": FractalParams(
                fractal_type="mandelbox",
                scale=-1.5,
                folding_limit=1.0,
                camera_pos=(0.0, 0.0, -4.0),
                coloring_mode="orbit_trap",
                color_palette="rainbow"
            ),
            
            "burning_ship_3d": FractalParams(
                fractal_type="burning_ship",
                power=2.0,
                camera_pos=(0.0, 0.0, -2.0),
                coloring_mode="iteration",
                color_palette="fire"
            ),
            
            "high_power_exotic": FractalParams(
                fractal_type="mandelbulb",
                power=16.0,
                iterations=150,
                camera_pos=(1.0, 1.0, -4.0),
                coloring_mode="normal",
                color_palette="rainbow"
            ),
            
            "metallic_study": FractalParams(
                fractal_type="mandelbulb",
                power=8.0,
                camera_pos=(0.0, 0.0, -2.5),
                coloring_mode="orbit_trap",
                color_palette="monochrome",
                metallic=0.8,
                roughness=0.05,
                base_color=(0.9, 0.9, 1.0)
            ),
            
            "ice_crystal": FractalParams(
                fractal_type="julia",
                power=6.0,
                julia_c=(0.0, 0.0, 0.8),
                camera_pos=(0.0, 0.0, -2.0),
                coloring_mode="distance",
                color_palette="ice",
                base_color=(0.8, 0.9, 1.0)
            )
        }
        
        if name not in presets:
            raise ValueError(f"Unknown preset '{name}'. Available: {list(presets.keys())}")
        return presets[name]
    
    @staticmethod
    def list_presets() -> List[str]:
        return ["classic_mandelbulb", "julia_set_3d", "mandelbox_classic", 
                "burning_ship_3d", "high_power_exotic", "metallic_study", "ice_crystal"]

# JIT-compiled fractal distance estimation functions
@jit(nopython=True, fastmath=True)
def mandelbulb_de(x, y, z, power, max_iter, bailout):
    """Distance estimation for Mandelbulb fractal"""
    xx, yy, zz = x, y, z
    dr = 1.0
    r = np.sqrt(xx*xx + yy*yy + zz*zz)
    orbit_trap = 1000.0  # Track minimum distance to origin during iteration
    
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
        
        # Same transformation as Mandelbulb but with constant c
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
def mandelbox_de(x, y, z, scale, folding_limit, folding_value, max_iter):
    """Distance estimation for Mandelbox fractal"""
    xx, yy, zz = x, y, z
    dr = 1.0
    orbit_trap = 1000.0
    
    for i in range(max_iter):
        orbit_trap = min(orbit_trap, np.sqrt(xx*xx + yy*yy + zz*zz))
        
        # Box folding
        if xx > folding_limit:
            xx = folding_value - xx
        elif xx < -folding_limit:
            xx = -folding_value - xx
        if yy > folding_limit:
            yy = folding_value - yy
        elif yy < -folding_limit:
            yy = -folding_value - yy
        if zz > folding_limit:
            zz = folding_value - zz
        elif zz < -folding_limit:
            zz = -folding_value - zz
        
        # Sphere folding
        r2 = xx*xx + yy*yy + zz*zz
        if r2 < 0.25:
            xx *= 4.0
            yy *= 4.0
            zz *= 4.0
            dr *= 4.0
        elif r2 < 1.0:
            temp = 1.0 / r2
            xx *= temp
            yy *= temp
            zz *= temp
            dr *= temp
        
        # Scale and translate
        xx = scale * xx + x
        yy = scale * yy + y
        zz = scale * zz + z
        dr = dr * abs(scale) + 1.0
        
        if np.sqrt(xx*xx + yy*yy + zz*zz) > 10.0:
            break
    
    return np.sqrt(xx*xx + yy*yy + zz*zz) / abs(dr), orbit_trap, i

@jit(nopython=True, fastmath=True)
def burning_ship_de(x, y, z, power, max_iter, bailout):
    """Distance estimation for 3D Burning Ship fractal"""
    xx, yy, zz = x, y, z
    dr = 1.0
    orbit_trap = 1000.0
    
    for i in range(max_iter):
        r = np.sqrt(xx*xx + yy*yy + zz*zz)
        if r > bailout:
            break
            
        orbit_trap = min(orbit_trap, r)
        
        # Burning ship uses absolute values
        xx = abs(xx)
        yy = abs(yy)
        zz = abs(zz)
        
        # Power iteration
        theta = np.arctan2(np.sqrt(xx*xx + yy*yy), zz)
        phi = np.arctan2(yy, xx)
        
        zr = r ** (power - 1.0)
        dr = zr * dr * power + 1.0
        
        zr = zr * r
        theta = theta * power
        phi = phi * power
        
        xx = zr * np.sin(theta) * np.cos(phi) + x
        yy = zr * np.sin(theta) * np.sin(phi) + y
        zz = zr * np.cos(theta) + z
    
    return 0.5 * np.log(r) * r / dr, orbit_trap, i

@jit(nopython=True, fastmath=True)
def get_distance_and_info(x, y, z, fractal_type, power, max_iter, bailout, 
                         julia_c, scale, folding_limit, folding_value):
    """Dispatch function for different fractal types"""
    if fractal_type == 0:  # mandelbulb
        return mandelbulb_de(x, y, z, power, max_iter, bailout)
    elif fractal_type == 1:  # julia
        return julia_set_de(x, y, z, power, max_iter, bailout, julia_c[0], julia_c[1], julia_c[2])
    elif fractal_type == 2:  # mandelbox
        return mandelbox_de(x, y, z, scale, folding_limit, folding_value, max_iter)
    elif fractal_type == 3:  # burning_ship
        return burning_ship_de(x, y, z, power, max_iter, bailout)
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
        # HSV to RGB conversion
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
def render_fractal_advanced(image, width, height, params_array):
    """Advanced fractal rendering with full parameter control"""
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
    
    # Fractal specific
    julia_c = (params_array[28], params_array[29], params_array[30])
    scale = params_array[31]
    folding_limit = params_array[32]
    folding_value = params_array[33]
    
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
                    pos[0], pos[1], pos[2], fractal_type, power, iterations, bailout,
                    julia_c, scale, folding_limit, folding_value
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
                                        fractal_type, power, iterations, bailout,
                                        julia_c, scale, folding_limit, folding_value)[0] - 
                    get_distance_and_info(final_pos[0] - eps, final_pos[1], final_pos[2], 
                                        fractal_type, power, iterations, bailout,
                                        julia_c, scale, folding_limit, folding_value)[0],
                    
                    get_distance_and_info(final_pos[0], final_pos[1] + eps, final_pos[2], 
                                        fractal_type, power, iterations, bailout,
                                        julia_c, scale, folding_limit, folding_value)[0] - 
                    get_distance_and_info(final_pos[0], final_pos[1] - eps, final_pos[2], 
                                        fractal_type, power, iterations, bailout,
                                        julia_c, scale, folding_limit, folding_value)[0],
                                        
                    get_distance_and_info(final_pos[0], final_pos[1], final_pos[2] + eps, 
                                        fractal_type, power, iterations, bailout,
                                        julia_c, scale, folding_limit, folding_value)[0] - 
                    get_distance_and_info(final_pos[0], final_pos[1], final_pos[2] - eps, 
                                        fractal_type, power, iterations, bailout,
                                        julia_c, scale, folding_limit, folding_value)[0]
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
                    # Simple metallic reflection
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

class AdvancedFractalRenderer:
    """Advanced fractal renderer with full parameter control"""
    
    def __init__(self):
        self.fractal_type_map = {
            "mandelbulb": 0,
            "julia": 1,
            "mandelbox": 2,
            "burning_ship": 3
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
            
            # Fractal specific
            params.julia_c[0], params.julia_c[1], params.julia_c[2],
            params.scale,
            params.folding_limit,
            params.folding_value
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
        render_fractal_advanced(image, params.width, params.height, params_array)
        
        render_time = time.time() - start_time
        print(f"Render completed in {render_time:.2f}s")
        
        # Calculate metrics
        surface_mask = np.any(image > 0.2, axis=2)  # Non-background pixels
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
    
    def save_image(self, image: np.ndarray, filepath: Path, title: str = ""):
        """Save rendered image"""
        plt.figure(figsize=(12, 9))
        plt.imshow(image, origin='upper')
        plt.axis('off')
        if title:
            plt.title(title, fontsize=14, pad=10)
        plt.tight_layout()
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        plt.close()
        
        return filepath.stat().st_size if filepath.exists() else 0

def test_preset_variations():
    """Test all preset variations and parameter controls"""
    renderer = AdvancedFractalRenderer()
    output_dir = Path("advanced_fractal_tests")
    output_dir.mkdir(exist_ok=True)
    
    print("=== Advanced Fractal Testing ===")
    print(f"Output directory: {output_dir.absolute()}")
    
    results = []
    
    # Test all presets
    for preset_name in FractalPresets.list_presets():
        print(f"\n--- Testing preset: {preset_name} ---")
        
        params = FractalPresets.get_preset(preset_name)
        image, metrics = renderer.render(params)
        
        # Save image
        output_file = output_dir / f"{preset_name}.png"
        file_size = renderer.save_image(image, output_file, 
                                      f"{preset_name} - {params.fractal_type}")
        
        metrics['file_size'] = file_size
        metrics['preset_name'] = preset_name
        results.append(metrics)
        
        print(f"Surface pixels: {metrics['surface_pixels']:,}")
        print(f"Unique colors: {metrics['unique_colors']:,}")
        print(f"File size: {file_size:,} bytes")
    
    # Test parameter variations
    print(f"\n--- Testing parameter variations ---")
    base_params = FractalPresets.get_preset("classic_mandelbulb")
    
    # Power variations
    for power in [4.0, 6.0, 12.0, 16.0]:
        test_params = FractalParams(**asdict(base_params))
        test_params.power = power
        
        image, metrics = renderer.render(test_params)
        output_file = output_dir / f"power_{power:.0f}.png"
        file_size = renderer.save_image(image, output_file, 
                                      f"Mandelbulb Power {power}")
        
        metrics['file_size'] = file_size
        metrics['test_type'] = f"power_{power}"
        results.append(metrics)
        
        print(f"Power {power}: {metrics['surface_pixels']:,} pixels, {file_size:,} bytes")
    
    # Color palette variations
    for palette in ["warm", "cool", "rainbow", "fire", "ice"]:
        test_params = FractalParams(**asdict(base_params))
        test_params.color_palette = palette
        
        image, metrics = renderer.render(test_params)
        output_file = output_dir / f"palette_{palette}.png"
        file_size = renderer.save_image(image, output_file, 
                                      f"Mandelbulb - {palette} palette")
        
        metrics['file_size'] = file_size
        metrics['test_type'] = f"palette_{palette}"
        results.append(metrics)
    
    # Camera angle variations
    camera_positions = [
        (0.0, 0.0, -2.0),   # Close
        (2.0, 2.0, -3.0),   # Angled
        (0.0, 3.0, -2.0),   # Above
        (-2.0, 0.0, -2.0)   # Side
    ]
    
    for i, cam_pos in enumerate(camera_positions):
        test_params = FractalParams(**asdict(base_params))
        test_params.camera_pos = cam_pos
        
        image, metrics = renderer.render(test_params)
        output_file = output_dir / f"camera_angle_{i+1}.png"
        file_size = renderer.save_image(image, output_file, 
                                      f"Camera Angle {i+1}")
        
        metrics['file_size'] = file_size
        metrics['test_type'] = f"camera_{i+1}"
        results.append(metrics)
    
    print(f"\n=== RESULTS SUMMARY ===")
    print(f"Total tests: {len(results)}")
    avg_render_time = np.mean([r['render_time'] for r in results])
    print(f"Average render time: {avg_render_time:.2f}s")
    
    surface_pixels = [r['surface_pixels'] for r in results]
    print(f"Surface pixel range: {min(surface_pixels):,} - {max(surface_pixels):,}")
    
    total_size = sum(r['file_size'] for r in results)
    print(f"Total output size: {total_size/1024/1024:.1f}MB")
    
    return results

if __name__ == "__main__":
    # Warm up JIT compiler
    print("Warming up Numba JIT compiler...")
    params = FractalParams(width=100, height=100, iterations=20)
    renderer = AdvancedFractalRenderer()
    renderer.render(params)
    print("JIT warm-up complete\n")
    
    # Run comprehensive tests
    results = test_preset_variations()
    print(f"\n✅ Advanced fractal testing complete!")
    print(f"Check output images in: ./advanced_fractal_tests/")