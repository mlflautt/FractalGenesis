#!/usr/bin/env python3
"""
Simple test for Mandelbulber lighting to verify proper fractal illumination.
"""

import subprocess
from pathlib import Path

def test_simple_render():
    """Test with minimal, known working parameters."""
    
    # Create simple settings file
    settings_content = """# Mandelbulber settings file
# version 2.33

image_width=512
image_height=512
formula_1=9
detail_level=1.0
N=250
power=8

# Camera
camera=3 -6 2
target=0 0 0
fov=53.13

# Enable all lights with high intensity
glow_enabled=1
glow_intensity=2.0
brightness=4.0
contrast=2.0
saturation=1.5

# Enable fake lights for illumination
fake_lights_enabled=1
fake_lights_intensity=5.0
fake_lights_visibility=5.0

# Enable ambient occlusion
ambient_occlusion_enabled=1
ambient_occlusion=2.0

# Material settings
mat1_shading=1
mat1_specular=8.0
mat1_surface_color=ff00 8000 4000
mat1_use_colors_from_palette=1
mat1_surface_gradient_enable=1

# Background
background_color_1=4000 8000 ff00
background_3_colors_enable=1

# Output
file_destination=test_lighting_simple
"""
    
    output_dir = Path("output/test_lighting")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    settings_file = output_dir / "simple_test.fract"
    with open(settings_file, 'w') as f:
        f.write(settings_content)
    
    output_file = output_dir / "simple_test.png"
    
    # Try to render with flatpak
    cmd = [
        "flatpak", "run", "com.github.buddhi1980.mandelbulber2",
        "--nogui",
        "--format", "png",
        "--res", "512x512",
        "--output", str(output_file),
        str(settings_file)
    ]
    
    print("Running command:", ' '.join(cmd))
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        print(f"Return code: {result.returncode}")
        print(f"Stdout: {result.stdout}")
        print(f"Stderr: {result.stderr}")
        
        if output_file.exists():
            print(f"✅ Success! Rendered: {output_file}")
            return True
        else:
            print("❌ Failed - no output file created")
            return False
            
    except Exception as e:
        print(f"❌ Error running command: {e}")
        return False

if __name__ == "__main__":
    test_simple_render()
