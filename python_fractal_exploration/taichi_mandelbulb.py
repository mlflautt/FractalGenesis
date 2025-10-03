#!/usr/bin/env python3
"""
Taichi GPU-Accelerated 3D Mandelbulb Renderer
==============================================

High-performance GPU ray marching using Taichi's JIT compilation.
Should be significantly faster than pure NumPy approach.
"""

import numpy as np
import time
from pathlib import Path

# Try to import taichi, handle gracefully if not available
try:
    import taichi as ti
    TAICHI_AVAILABLE = True
    print("Taichi available - GPU acceleration enabled")
except ImportError:
    TAICHI_AVAILABLE = False
    print("Taichi not available - skipping GPU tests")

if TAICHI_AVAILABLE:
    # Initialize Taichi (try GPU first, fallback to CPU)
    try:
        ti.init(arch=ti.gpu, device_memory_fraction=0.8)
        print("Taichi initialized with GPU backend")
    except:
        try:
            ti.init(arch=ti.cpu)
            print("Taichi initialized with CPU backend")
        except:
            TAICHI_AVAILABLE = False
            print("Taichi initialization failed")

@ti.func
def mandelbulb_de(pos: ti.math.vec3, power: ti.f32, max_iter: ti.i32) -> ti.f32:
    """Distance estimation for Mandelbulb using Taichi"""
    x, y, z = pos.x, pos.y, pos.z
    xx, yy, zz = x, y, z
    dr = 1.0
    r = ti.sqrt(xx*xx + yy*yy + zz*zz)
    
    for i in range(max_iter):
        if r > 2.0:
            break
            
        # Convert to polar coordinates
        theta = ti.atan2(ti.sqrt(xx*xx + yy*yy), zz)
        phi = ti.atan2(yy, xx)
        
        # Scale and rotate
        zr = r ** (power - 1.0)
        dr = zr * dr * power + 1.0
        
        # Convert back to cartesian
        zr = zr * r
        theta = theta * power
        phi = phi * power
        
        xx = zr * ti.sin(theta) * ti.cos(phi) + x
        yy = zr * ti.sin(theta) * ti.sin(phi) + y
        zz = zr * ti.cos(theta) + z
        
        r = ti.sqrt(xx*xx + yy*yy + zz*zz)
    
    return 0.5 * ti.log(r) * r / dr

@ti.func  
def ray_march(ray_origin: ti.math.vec3, ray_dir: ti.math.vec3, 
              power: ti.f32, max_iter: ti.i32) -> ti.math.vec2:
    """Ray march to find Mandelbulb intersection"""
    t = 0.0
    min_dist = 0.001
    max_steps = 100
    
    for step in range(max_steps):
        pos = ray_origin + t * ray_dir
        dist = mandelbulb_de(pos, power, max_iter)
        
        if ti.abs(dist) < min_dist:
            return ti.math.vec2(t, dist)
            
        t += ti.max(dist, min_dist)
        
        if t > 10.0:
            break
    
    return ti.math.vec2(t, 1.0)  # Miss

@ti.func
def compute_normal(pos: ti.math.vec3, power: ti.f32, max_iter: ti.i32) -> ti.math.vec3:
    """Compute surface normal using finite differences"""
    eps = 0.001
    dx = ti.math.vec3(eps, 0.0, 0.0)
    dy = ti.math.vec3(0.0, eps, 0.0)  
    dz = ti.math.vec3(0.0, 0.0, eps)
    
    normal = ti.math.vec3(
        mandelbulb_de(pos + dx, power, max_iter) - mandelbulb_de(pos - dx, power, max_iter),
        mandelbulb_de(pos + dy, power, max_iter) - mandelbulb_de(pos - dy, power, max_iter),
        mandelbulb_de(pos + dz, power, max_iter) - mandelbulb_de(pos - dz, power, max_iter)
    )
    
    return ti.math.normalize(normal)

@ti.kernel
def render_mandelbulb(image: ti.template(), width: ti.i32, height: ti.i32,
                     camera_pos: ti.math.vec3, target: ti.math.vec3, up: ti.math.vec3,
                     fov: ti.f32, power: ti.f32, max_iter: ti.i32):
    """Render Mandelbulb to image buffer"""
    
    # Set up camera coordinate system
    forward = ti.math.normalize(target - camera_pos)
    right = ti.math.normalize(ti.math.cross(forward, up))
    camera_up = ti.math.cross(right, forward)
    
    # Image plane setup
    aspect = ti.cast(width, ti.f32) / ti.cast(height, ti.f32)
    fov_rad = fov * 3.14159265359 / 180.0
    h = ti.tan(fov_rad / 2.0)
    w = h * aspect
    
    # Render each pixel
    for i, j in ti.ndrange(width, height):
        # Convert pixel coordinates to normalized device coordinates
        u = (2.0 * ti.cast(i, ti.f32) / ti.cast(width, ti.f32) - 1.0) * w
        v = (2.0 * ti.cast(j, ti.f32) / ti.cast(height, ti.f32) - 1.0) * h
        
        # Compute ray direction
        ray_dir = ti.math.normalize(forward + u * right + v * camera_up)
        
        # Ray march
        result = ray_march(camera_pos, ray_dir, power, max_iter)
        t, dist = result.x, result.y
        
        # Set pixel color
        if ti.abs(dist) < 0.01:  # Hit surface
            # Compute position and normal
            pos = camera_pos + t * ray_dir
            normal = compute_normal(pos, power, max_iter)
            
            # Simple lighting
            light_dir = ti.math.normalize(ti.math.vec3(1.0, 1.0, -1.0))
            diffuse = ti.max(0.0, ti.math.dot(normal, light_dir))
            
            # Base color with lighting
            base_color = ti.math.vec3(0.8, 0.6, 0.4)
            ambient = 0.3
            color = base_color * (ambient + diffuse * 0.7)
            
            # Add depth attenuation  
            depth_factor = ti.max(0.2, 1.0 - t / 8.0)
            color = color * depth_factor
            
            image[i, j] = color
        else:
            # Background
            image[i, j] = ti.math.vec3(0.1, 0.1, 0.2)

class TaichiMandelbulb:
    def __init__(self):
        if not TAICHI_AVAILABLE:
            raise RuntimeError("Taichi not available")
            
    def render_image(self, width=800, height=600, power=8.0, max_iter=80,
                    camera_pos=np.array([0, 0, -3]), target=np.array([0, 0, 0]), 
                    up=np.array([0, 1, 0]), fov=45):
        """Render Mandelbulb image using Taichi GPU acceleration"""
        
        print(f"Rendering {width}x{height} image (power={power}) with Taichi...")
        start_time = time.time()
        
        # Create image buffer
        image = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))
        
        # Convert numpy arrays to Taichi vectors
        cam_pos = ti.math.vec3(camera_pos[0], camera_pos[1], camera_pos[2])
        cam_target = ti.math.vec3(target[0], target[1], target[2])
        cam_up = ti.math.vec3(up[0], up[1], up[2])
        
        # Render
        render_mandelbulb(image, width, height, cam_pos, cam_target, cam_up, 
                         fov, power, max_iter)
        
        # Convert back to numpy
        image_np = image.to_numpy()
        
        render_time = time.time() - start_time
        print(f"Taichi render completed in {render_time:.2f}s")
        
        # Calculate surface pixels (non-background pixels)
        background = np.array([0.1, 0.1, 0.2])
        surface_mask = np.any(np.abs(image_np - background) > 0.01, axis=2)
        surface_pixels = np.sum(surface_mask)
        
        return image_np, surface_pixels, render_time

def test_taichi_mandelbulb():
    """Test Taichi Mandelbulb renderer with various parameters"""
    if not TAICHI_AVAILABLE:
        print("❌ Taichi not available - skipping GPU tests")
        return []
        
    output_dir = Path("taichi_mandelbulb_tests")
    output_dir.mkdir(exist_ok=True)
    
    test_configs = [
        {
            'name': 'taichi_classic',
            'power': 8.0,
            'camera_pos': np.array([0, 0, -3]),
            'description': 'Classic Mandelbulb with Taichi GPU'
        },
        {
            'name': 'taichi_power6_hires',
            'power': 6.0,
            'width': 1024,
            'height': 768,
            'camera_pos': np.array([0, 0, -2.5]),
            'description': 'High-res Power 6 Mandelbulb'
        },
        {
            'name': 'taichi_power16_exotic',
            'power': 16.0,
            'camera_pos': np.array([0.8, 0.8, -3.5]),
            'fov': 50,
            'description': 'Exotic high-power fractal'
        },
        {
            'name': 'taichi_detailed_closeup',
            'power': 8.0,
            'width': 800,
            'height': 600,
            'max_iter': 120,
            'camera_pos': np.array([0, 0, -1.8]),
            'fov': 35,
            'description': 'High-detail close-up render'
        }
    ]
    
    try:
        renderer = TaichiMandelbulb()
    except RuntimeError as e:
        print(f"❌ {e}")
        return []
    
    results = []
    
    print("=== Taichi GPU Mandelbulb Testing ===")
    print(f"Output directory: {output_dir.absolute()}")
    
    # Import matplotlib here to avoid issues if not available
    import matplotlib.pyplot as plt
    
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
            target=config.get('target', np.array([0, 0, 0])),
            fov=config.get('fov', 45)
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
    print(f"\n=== TAICHI SUMMARY ===")
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
    results = test_taichi_mandelbulb()
    if results:
        print("\n✅ Taichi Mandelbulb testing complete!")
        print(f"Check output images in: ./taichi_mandelbulb_tests/")
    else:
        print("\n❌ Taichi tests skipped (not available)")