#!/usr/bin/env python3
"""
Pure NumPy/Matplotlib 3D Mandelbulb Renderer
===============================================

Classic distance estimation ray marching implementation using only NumPy and Matplotlib.
Good baseline to compare against more advanced options.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from pathlib import Path

class NumPyMandelbulb:
    def __init__(self, power=8.0, max_iter=100, bailout=2.0):
        self.power = power
        self.max_iter = max_iter
        self.bailout = bailout
        
    def mandelbulb_de(self, pos):
        """Distance estimation for 3D Mandelbulb using triplex algebra"""
        x, y, z = pos[..., 0], pos[..., 1], pos[..., 2]
        
        # Initialize
        xx, yy, zz = x.copy(), y.copy(), z.copy()
        dr = np.ones_like(x)
        r = np.sqrt(xx*xx + yy*yy + zz*zz)
        
        for i in range(self.max_iter):
            # Break if escaped
            mask = r < self.bailout
            if not np.any(mask):
                break
                
            # Convert to polar coordinates
            theta = np.arctan2(np.sqrt(xx*xx + yy*yy), zz)
            phi = np.arctan2(yy, xx)
            
            # Scale and rotate the point
            zr = r ** (self.power - 1.0)
            dr = zr * dr * self.power + 1.0
            
            # Convert back to cartesian coordinates
            zr = zr * r
            theta = theta * self.power
            phi = phi * self.power
            
            xx = zr * np.sin(theta) * np.cos(phi) + x
            yy = zr * np.sin(theta) * np.sin(phi) + y
            zz = zr * np.cos(theta) + z
            
            r = np.sqrt(xx*xx + yy*yy + zz*zz)
            
        return 0.5 * np.log(r) * r / dr
    
    def ray_march(self, ray_origin, ray_dir, max_steps=100, min_dist=0.001):
        """Ray march to find intersection with Mandelbulb surface"""
        t = np.zeros(ray_origin.shape[:-1])
        
        for step in range(max_steps):
            pos = ray_origin + t[..., np.newaxis] * ray_dir
            dist = self.mandelbulb_de(pos)
            
            t += np.maximum(dist, min_dist)
            
            # Check for hits (close enough to surface)
            hit_mask = np.abs(dist) < min_dist
            
            # Check for escapes (too far from origin)
            escape_mask = t > 10.0
            
            if np.all(hit_mask | escape_mask):
                break
        
        return t, dist
    
    def render_image(self, width=800, height=600, camera_pos=np.array([0, 0, -3]), 
                    target=np.array([0, 0, 0]), up=np.array([0, 1, 0]), fov=45):
        """Render 2D image of 3D Mandelbulb"""
        print(f"Rendering {width}x{height} image (power={self.power})...")
        start_time = time.time()
        
        # Set up camera coordinate system
        forward = target - camera_pos
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        
        # Create image plane coordinates
        aspect = width / height
        fov_rad = np.radians(fov)
        h = np.tan(fov_rad / 2)
        w = h * aspect
        
        # Generate rays for each pixel
        u = np.linspace(-w, w, width)
        v = np.linspace(-h, h, height)
        U, V = np.meshgrid(u, v)
        
        # Ray directions in world space
        ray_dirs = (forward[np.newaxis, np.newaxis, :] + 
                   U[:, :, np.newaxis] * right[np.newaxis, np.newaxis, :] + 
                   V[:, :, np.newaxis] * up[np.newaxis, np.newaxis, :])
        ray_dirs = ray_dirs / np.linalg.norm(ray_dirs, axis=2, keepdims=True)
        
        # Ray origins (all from camera position)
        ray_origins = np.broadcast_to(camera_pos, ray_dirs.shape)
        
        # Perform ray marching
        distances, final_dist = self.ray_march(ray_origins, ray_dirs)
        
        # Create image based on distance and surface detection
        image = np.zeros((height, width, 3))
        
        # Surface hit mask
        hit_mask = np.abs(final_dist) < 0.01
        
        # Color based on distance and surface normal approximation
        surface_color = np.array([0.8, 0.6, 0.4])  # Warm orange
        background_color = np.array([0.1, 0.1, 0.2])  # Dark blue
        
        # Apply coloring
        for i in range(3):
            image[:, :, i] = np.where(hit_mask, surface_color[i], background_color[i])
        
        # Add depth-based shading
        depth_factor = np.clip(1.0 - distances / 8.0, 0.2, 1.0)
        image = image * depth_factor[:, :, np.newaxis]
        
        render_time = time.time() - start_time
        print(f"Render completed in {render_time:.2f}s")
        
        return image, hit_mask.sum(), render_time

def test_numpy_mandelbulb():
    """Test NumPy Mandelbulb renderer with various parameters"""
    output_dir = Path("numpy_mandelbulb_tests")
    output_dir.mkdir(exist_ok=True)
    
    test_configs = [
        {
            'name': 'classic_power8',
            'power': 8.0,
            'camera_pos': np.array([0, 0, -3]),
            'description': 'Classic Mandelbulb power 8'
        },
        {
            'name': 'power6_closeup', 
            'power': 6.0,
            'camera_pos': np.array([0, 0, -2]),
            'fov': 30,
            'description': 'Power 6 Mandelbulb, closer view'
        },
        {
            'name': 'power12_wide',
            'power': 12.0,
            'camera_pos': np.array([0.5, 0.5, -4]),
            'fov': 60,
            'description': 'Power 12 Mandelbulb, angled wide view'
        },
        {
            'name': 'power4_sidview',
            'power': 4.0,
            'camera_pos': np.array([3, 0, 0]),
            'target': np.array([0, 0, 0]),
            'description': 'Power 4 Mandelbulb from side'
        },
        {
            'name': 'fractional_power',
            'power': 7.3,
            'camera_pos': np.array([0, 2, -2]),
            'description': 'Fractional power 7.3'
        }
    ]
    
    results = []
    
    print("=== NumPy Mandelbulb Testing ===")
    print(f"Output directory: {output_dir.absolute()}")
    
    for config in test_configs:
        print(f"\n--- Testing: {config['name']} ---")
        print(f"Description: {config['description']}")
        
        # Create renderer
        renderer = NumPyMandelbulb(
            power=config['power'], 
            max_iter=80,
            bailout=2.0
        )
        
        # Render image
        image, surface_pixels, render_time = renderer.render_image(
            width=600, height=450,
            camera_pos=config['camera_pos'],
            target=config.get('target', np.array([0, 0, 0])),
            fov=config.get('fov', 45)
        )
        
        # Save image
        output_file = output_dir / f"{config['name']}.png"
        plt.figure(figsize=(10, 7.5))
        plt.imshow(image, origin='upper')
        plt.axis('off')
        plt.title(f"{config['description']} (Power: {config['power']})")
        plt.tight_layout()
        plt.savefig(output_file, dpi=100, bbox_inches='tight')
        plt.close()
        
        # Calculate quality metrics
        unique_colors = len(np.unique(image.reshape(-1, 3), axis=0))
        avg_brightness = np.mean(image)
        contrast = np.std(image)
        
        result = {
            'name': config['name'],
            'power': config['power'],
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
    print(f"\n=== SUMMARY ===")
    print(f"Tests completed: {len(results)}")
    avg_render_time = np.mean([r['render_time'] for r in results])
    print(f"Average render time: {avg_render_time:.2f}s")
    
    surface_pixels = [r['surface_pixels'] for r in results]
    print(f"Surface pixel range: {min(surface_pixels):,} - {max(surface_pixels):,}")
    
    # Check for trivial outputs
    non_trivial = sum(1 for r in results if r['surface_pixels'] > 1000 and r['contrast'] > 0.1)
    print(f"Non-trivial outputs: {non_trivial}/{len(results)}")
    
    return results

if __name__ == "__main__":
    results = test_numpy_mandelbulb()
    print("\n✅ NumPy Mandelbulb testing complete!")
    print(f"Check output images in: ./numpy_mandelbulb_tests/")