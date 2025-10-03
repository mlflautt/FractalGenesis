#!/usr/bin/env python3
"""
Numba JIT-Compiled 3D Mandelbulb Renderer
==========================================

Fast CPU ray marching using Numba's JIT compilation.
Good middle-ground between pure NumPy and GPU acceleration.
"""

import numpy as np
import time
from pathlib import Path

# Try to import numba, handle gracefully if not available
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
    print("Numba available - JIT acceleration enabled")
except ImportError:
    NUMBA_AVAILABLE = False
    print("Numba not available - falling back to pure NumPy")

if NUMBA_AVAILABLE:
    @jit(nopython=True, fastmath=True)
    def mandelbulb_de_numba(x, y, z, power, max_iter):
        """JIT-compiled distance estimation for Mandelbulb"""
        xx, yy, zz = x, y, z
        dr = 1.0
        r = np.sqrt(xx*xx + yy*yy + zz*zz)
        
        for i in range(max_iter):
            if r > 2.0:
                break
                
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
        
        return 0.5 * np.log(r) * r / dr

    @jit(nopython=True, fastmath=True)
    def ray_march_numba(ox, oy, oz, dx, dy, dz, power, max_iter):
        """JIT-compiled ray marching"""
        t = 0.0
        min_dist = 0.001
        max_steps = 100
        
        for step in range(max_steps):
            x = ox + t * dx
            y = oy + t * dy  
            z = oz + t * dz
            
            dist = mandelbulb_de_numba(x, y, z, power, max_iter)
            
            if abs(dist) < min_dist:
                return t, dist
                
            t += max(dist, min_dist)
            
            if t > 10.0:
                break
        
        return t, 1.0  # Miss

    @jit(nopython=True, parallel=True, fastmath=True)
    def render_mandelbulb_numba(image, width, height, camera_pos, target, up, fov, power, max_iter):
        """JIT-compiled parallel rendering"""
        
        # Set up camera coordinate system
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
        
        # Background color
        bg_r, bg_g, bg_b = 0.1, 0.1, 0.2
        
        # Render each pixel in parallel
        for j in prange(height):
            for i in range(width):
                # Convert pixel to NDC
                u = (2.0 * i / width - 1.0) * w
                v = (2.0 * j / height - 1.0) * h
                
                # Compute ray direction
                ray_dir = forward + u * right + v * camera_up
                ray_dir = ray_dir / np.linalg.norm(ray_dir)
                
                # Ray march
                t, dist = ray_march_numba(
                    camera_pos[0], camera_pos[1], camera_pos[2],
                    ray_dir[0], ray_dir[1], ray_dir[2],
                    power, max_iter
                )
                
                if abs(dist) < 0.01:  # Hit surface
                    # Compute surface normal (simplified)
                    pos_x = camera_pos[0] + t * ray_dir[0]
                    pos_y = camera_pos[1] + t * ray_dir[1]
                    pos_z = camera_pos[2] + t * ray_dir[2]
                    
                    # Simple lighting calculation
                    light_x, light_y, light_z = 1.0, 1.0, -1.0
                    light_norm = np.sqrt(light_x*light_x + light_y*light_y + light_z*light_z)
                    light_x, light_y, light_z = light_x/light_norm, light_y/light_norm, light_z/light_norm
                    
                    # Approximate normal via finite differences
                    eps = 0.001
                    normal_x = mandelbulb_de_numba(pos_x + eps, pos_y, pos_z, power, max_iter) - mandelbulb_de_numba(pos_x - eps, pos_y, pos_z, power, max_iter)
                    normal_y = mandelbulb_de_numba(pos_x, pos_y + eps, pos_z, power, max_iter) - mandelbulb_de_numba(pos_x, pos_y - eps, pos_z, power, max_iter)
                    normal_z = mandelbulb_de_numba(pos_x, pos_y, pos_z + eps, power, max_iter) - mandelbulb_de_numba(pos_x, pos_y, pos_z - eps, power, max_iter)
                    
                    normal_norm = np.sqrt(normal_x*normal_x + normal_y*normal_y + normal_z*normal_z)
                    if normal_norm > 0:
                        normal_x, normal_y, normal_z = normal_x/normal_norm, normal_y/normal_norm, normal_z/normal_norm
                    
                    # Diffuse lighting
                    diffuse = max(0.0, normal_x*light_x + normal_y*light_y + normal_z*light_z)
                    
                    # Base color with lighting
                    base_r, base_g, base_b = 0.8, 0.6, 0.4
                    ambient = 0.3
                    color_r = base_r * (ambient + diffuse * 0.7)
                    color_g = base_g * (ambient + diffuse * 0.7)  
                    color_b = base_b * (ambient + diffuse * 0.7)
                    
                    # Depth attenuation
                    depth_factor = max(0.2, 1.0 - t / 8.0)
                    color_r *= depth_factor
                    color_g *= depth_factor
                    color_b *= depth_factor
                    
                    image[j, i, 0] = color_r
                    image[j, i, 1] = color_g
                    image[j, i, 2] = color_b
                else:
                    # Background
                    image[j, i, 0] = bg_r
                    image[j, i, 1] = bg_g
                    image[j, i, 2] = bg_b

class NumbaMandelbulb:
    def __init__(self):
        if not NUMBA_AVAILABLE:
            raise RuntimeError("Numba not available")
            
    def render_image(self, width=800, height=600, power=8.0, max_iter=80,
                    camera_pos=np.array([0.0, 0.0, -3.0]), target=np.array([0.0, 0.0, 0.0]), 
                    up=np.array([0.0, 1.0, 0.0]), fov=45.0):
        """Render Mandelbulb image using Numba JIT compilation"""
        
        print(f"Rendering {width}x{height} image (power={power}) with Numba JIT...")
        start_time = time.time()
        
        # Create image buffer
        image = np.zeros((height, width, 3), dtype=np.float64)
        
        # Render using JIT-compiled function
        render_mandelbulb_numba(image, width, height, camera_pos, target, up, fov, power, max_iter)
        
        render_time = time.time() - start_time
        print(f"Numba render completed in {render_time:.2f}s")
        
        # Calculate surface pixels (non-background pixels)
        background = np.array([0.1, 0.1, 0.2])
        surface_mask = np.any(np.abs(image - background) > 0.01, axis=2)
        surface_pixels = np.sum(surface_mask)
        
        return image, surface_pixels, render_time

def test_numba_mandelbulb():
    """Test Numba Mandelbulb renderer with various parameters"""
    if not NUMBA_AVAILABLE:
        print("❌ Numba not available - skipping JIT tests")
        return []
        
    output_dir = Path("numba_mandelbulb_tests")
    output_dir.mkdir(exist_ok=True)
    
    test_configs = [
        {
            'name': 'numba_classic',
            'power': 8.0,
            'camera_pos': np.array([0.0, 0.0, -3.0]),
            'description': 'Classic Mandelbulb with Numba JIT'
        },
        {
            'name': 'numba_power5_artistic',
            'power': 5.0,
            'camera_pos': np.array([0.5, 1.0, -2.5]),
            'fov': 55.0,
            'description': 'Artistic Power 5 composition'
        },
        {
            'name': 'numba_power10_complex',
            'power': 10.0,
            'camera_pos': np.array([1.2, 0.8, -4.0]),
            'width': 1024,
            'height': 768,
            'description': 'Complex high-power fractal'
        },
        {
            'name': 'numba_detailed_study',
            'power': 8.0,
            'max_iter': 120,
            'camera_pos': np.array([0.0, 0.0, -2.0]),
            'fov': 35.0,
            'description': 'High-detail scientific study'
        },
        {
            'name': 'numba_extreme_closeup',
            'power': 8.0,
            'camera_pos': np.array([0.0, 0.0, -1.5]),
            'fov': 25.0,
            'max_iter': 150,
            'description': 'Extreme close-up with high iterations'
        }
    ]
    
    try:
        renderer = NumbaMandelbulb()
    except RuntimeError as e:
        print(f"❌ {e}")
        return []
    
    results = []
    
    print("=== Numba JIT Mandelbulb Testing ===")
    print(f"Output directory: {output_dir.absolute()}")
    
    # Import matplotlib here
    import matplotlib.pyplot as plt
    
    # Warm up JIT compiler with small render
    print("Warming up JIT compiler...")
    renderer.render_image(width=100, height=100, power=8.0, max_iter=20)
    print("JIT warm-up complete")
    
    for config in test_configs:
        print(f"\n--- Testing: {config['name']} ---")
        print(f"Description: {config['description']}")
        
        # Render image
        image, surface_pixels, render_time = renderer.render_image(
            width=config.get('width', 600),
            height=config.get('height', 450), 
            power=config['power'],
            max_iter=config.get('max_iter', 80),
            camera_pos=config['camera_pos'],
            target=config.get('target', np.array([0.0, 0.0, 0.0])),
            fov=config.get('fov', 45.0)
        )
        
        # Save image
        output_file = output_dir / f"{config['name']}.png"
        plt.figure(figsize=(12, 9))
        plt.imshow(image, origin='upper')
        plt.axis('off')
        plt.title(f"{config['description']} (Power: {config['power']})")
        plt.tight_layout()
        plt.savefig(output_file, dpi=100, bbox_inches='tight')
        plt.close()
        
        # Calculate metrics
        unique_colors = len(np.unique(image.reshape(-1, 3), axis=0))
        avg_brightness = np.mean(image)
        contrast = np.std(image)
        
        result = {
            'name': config['name'],
            'power': config['power'],
            'resolution': f"{config.get('width', 600)}x{config.get('height', 450)}",
            'surface_pixels': surface_pixels,
            'render_time': render_time,
            'unique_colors': unique_colors,
            'avg_brightness': avg_brightness,
            'contrast': contrast,
            'file_size': output_file.stat().st_size if output_file.exists() else 0
        }
        
        print(f"Surface pixels: {surface_pixels:,}")
        print(f"Render time: {render_time:.2f}s")
        print(f"Unique colors: {unique_colors:,}")
        print(f"File size: {result['file_size']:,} bytes")
        
        results.append(result)
    
    # Generate summary
    print(f"\n=== NUMBA SUMMARY ===")
    print(f"Tests completed: {len(results)}")
    if results:
        avg_render_time = np.mean([r['render_time'] for r in results])
        print(f"Average render time: {avg_render_time:.2f}s")
        
        surface_pixels = [r['surface_pixels'] for r in results]
        print(f"Surface pixel range: {min(surface_pixels):,} - {max(surface_pixels):,}")
        
        non_trivial = sum(1 for r in results if r['surface_pixels'] > 1000 and r['contrast'] > 0.1)
        print(f"Non-trivial outputs: {non_trivial}/{len(results)}")
    
    return results

if __name__ == "__main__":
    results = test_numba_mandelbulb()
    if results:
        print("\n✅ Numba Mandelbulb testing complete!")
        print(f"Check output images in: ./numba_mandelbulb_tests/")
    else:
        print("\n❌ Numba tests skipped (not available)")