#!/usr/bin/env python3
"""
Test script to validate Mandelbulber parameter generation and rendering.
This tests the new modular parameter system.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from renderers.mandelbulber.parameters import MandelbulberParameters
from renderers.mandelbulber.renderer import MandelbulberRenderer

def test_parameter_generation():
    """Test generation of .fract file content."""
    print("=== Testing Parameter Generation ===")
    
    # Create default parameters
    params = MandelbulberParameters()
    
    # Customize some parameters for a good fractal
    params.camera.camera_x = 3.0
    params.camera.camera_y = -6.0
    params.camera.camera_z = 2.0
    params.fractal.formula_name = "mandelbulb"
    params.fractal.power = 8.0
    params.fractal.iterations = 200
    params.material.surface_color_r = 1.0
    params.material.surface_color_g = 0.5
    params.material.surface_color_b = 0.2
    
    # Generate .fract content
    fract_content = params.to_fract_file()
    
    print("Generated .fract content:")
    print("-" * 50)
    print(fract_content)
    print("-" * 50)
    
    # Save to file
    with open('test_generated.fract', 'w') as f:
        f.write(fract_content)
    
    print("✅ Parameter generation test completed")
    print("📁 Saved to test_generated.fract")
    return fract_content

def test_renderer_availability():
    """Test if the renderer can find Mandelbulber."""
    print("\n=== Testing Renderer Availability ===")
    
    try:
        renderer = MandelbulberRenderer()
        print(f"Mandelbulber path: {renderer.mandelbulber_path}")
        print(f"Using Flatpak: {renderer.using_flatpak}")
        print(f"Is available: {renderer.is_available}")
        
        if renderer.is_available:
            print("✅ Mandelbulber renderer is available")
        else:
            print("❌ Mandelbulber renderer is NOT available")
        
        return renderer
    except Exception as e:
        print(f"❌ Renderer initialization failed: {e}")
        return None

def test_single_render():
    """Test rendering a single fractal."""
    print("\n=== Testing Single Render ===")
    
    renderer = test_renderer_availability()
    if not renderer or not renderer.is_available:
        print("❌ Skipping render test - Mandelbulber not available")
        return
    
    # Create test parameters
    params = MandelbulberParameters()
    params.render.image_width = 400
    params.render.image_height = 400
    
    # Simple mandelbulb
    params.camera.camera_x = 2.5
    params.camera.camera_y = -4.0
    params.camera.camera_z = 1.5
    params.fractal.formula_name = "mandelbulb"
    params.fractal.power = 8.0
    params.fractal.iterations = 150
    
    # Bright colors for visibility
    params.material.surface_color_r = 0.8
    params.material.surface_color_g = 0.4
    params.material.surface_color_b = 0.1
    params.material.specular_r = 1.0
    params.material.specular_g = 1.0
    params.material.specular_b = 1.0
    
    try:
        output_path = renderer.render_single(params, "test_single_render.png")
        if output_path:
            print(f"✅ Single render successful: {output_path}")
            print(f"📁 File size: {output_path.stat().st_size} bytes")
        else:
            print("❌ Single render failed")
    except Exception as e:
        print(f"❌ Single render error: {e}")

def test_random_parameters():
    """Test random parameter generation and rendering."""
    print("\n=== Testing Random Parameters ===")
    
    renderer = test_renderer_availability()
    if not renderer or not renderer.is_available:
        print("❌ Skipping random test - Mandelbulber not available")
        return
    
    # Generate random parameters
    params = MandelbulberParameters()
    params.randomize()
    params.render.image_width = 300
    params.render.image_height = 300
    
    print(f"Random fractal: {params.fractal.formula_name}")
    print(f"Power: {params.fractal.power:.2f}")
    print(f"Camera: ({params.camera.camera_x:.2f}, {params.camera.camera_y:.2f}, {params.camera.camera_z:.2f})")
    
    try:
        output_path = renderer.render_single(params, "test_random_render.png")
        if output_path:
            print(f"✅ Random render successful: {output_path}")
        else:
            print("❌ Random render failed")
    except Exception as e:
        print(f"❌ Random render error: {e}")

if __name__ == "__main__":
    print("🧪 Testing Mandelbulber Integration\n")
    
    # Run tests
    test_parameter_generation()
    test_renderer_availability()  
    test_single_render()
    test_random_parameters()
    
    print("\n🏁 All tests completed!")