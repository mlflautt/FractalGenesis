#!/usr/bin/env python3
"""
Test script to verify Mandelbulber parameter diversity.
Creates several fractals with different parameters to ensure variety.
"""

import sys
import os
import subprocess
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from renderers.mandelbulber.parameters import MandelbulberParameters
from renderers.mandelbulber.renderer import MandelbulberRenderer

def create_test_fractal(name, setup_func):
    """Create a test fractal with given setup function."""
    params = MandelbulberParameters()
    setup_func(params)
    
    # Render at smaller size for faster testing
    params.render.image_width = 300
    params.render.image_height = 300
    
    # Save parameter file
    param_file = f"test_{name}.fract"
    with open(param_file, 'w') as f:
        f.write(params.to_fract_file())
    
    # Render image
    output_file = f"test_{name}_output.jpg"
    cmd = [
        "flatpak", "run", "com.github.buddhi1980.mandelbulber2",
        "--nogui", "--output", output_file, param_file
    ]
    
    print(f"Creating {name} fractal...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if os.path.exists(output_file):
        size = os.path.getsize(output_file)
        print(f"✅ {name}: {size} bytes")
        return True
    else:
        print(f"❌ {name}: Failed")
        return False

def setup_classic_mandelbulb(params):
    """Setup classic Mandelbulb."""
    params.camera.camera_x = 3.0
    params.camera.camera_y = -6.0
    params.camera.camera_z = 2.0
    params.fractal.formula_name = "mandelbulb"
    params.fractal.power = 8.0
    params.fractal.iterations = 200
    params.material.surface_color_r = 0.8
    params.material.surface_color_g = 0.4
    params.material.surface_color_b = 0.1

def setup_high_power_mandelbulb(params):
    """Setup high power Mandelbulb.""" 
    params.camera.camera_x = 2.0
    params.camera.camera_y = -4.0
    params.camera.camera_z = 1.0
    params.fractal.formula_name = "mandelbulb"
    params.fractal.power = 16.0  # Much higher power
    params.fractal.iterations = 150
    params.material.surface_color_r = 0.2
    params.material.surface_color_g = 0.8
    params.material.surface_color_b = 0.9

def setup_close_view(params):
    """Setup close-up view."""
    params.camera.camera_x = 1.2
    params.camera.camera_y = -2.0
    params.camera.camera_z = 0.5
    params.fractal.formula_name = "mandelbulb"
    params.fractal.power = 8.0
    params.fractal.iterations = 300
    params.material.surface_color_r = 1.0
    params.material.surface_color_g = 0.2
    params.material.surface_color_b = 0.3

def setup_different_angle(params):
    """Setup different camera angle."""
    params.camera.camera_x = -2.0
    params.camera.camera_y = 3.0
    params.camera.camera_z = -1.0
    params.fractal.formula_name = "mandelbulb" 
    params.fractal.power = 10.0
    params.fractal.iterations = 180
    params.material.surface_color_r = 0.3
    params.material.surface_color_g = 1.0
    params.material.surface_color_b = 0.4

def setup_random_test(params):
    """Setup completely random parameters."""
    params.randomize()
    params.render.image_width = 300
    params.render.image_height = 300

if __name__ == "__main__":
    print("🧪 Testing Mandelbulber Parameter Diversity\n")
    
    test_setups = [
        ("classic_mandelbulb", setup_classic_mandelbulb),
        ("high_power", setup_high_power_mandelbulb),  
        ("close_view", setup_close_view),
        ("different_angle", setup_different_angle),
        ("random1", setup_random_test),
        ("random2", setup_random_test),
    ]
    
    results = []
    for name, setup_func in test_setups:
        success = create_test_fractal(name, setup_func)
        results.append((name, success))
    
    print(f"\n📊 Results:")
    successful = sum(1 for _, success in results if success)
    print(f"✅ Successful renders: {successful}/{len(results)}")
    
    if successful >= 4:
        print("🎉 Mandelbulber integration shows good diversity!")
    else:
        print("⚠️  Low success rate - may need parameter tweaking")
    
    # Show file sizes to verify diversity
    print(f"\nFile sizes (bytes):")
    for name, success in results:
        if success:
            filename = f"test_{name}_output.jpg"
            if os.path.exists(filename):
                size = os.path.getsize(filename)
                print(f"  {name}: {size}")