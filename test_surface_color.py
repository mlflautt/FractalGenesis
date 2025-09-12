#!/usr/bin/env python3
"""
Test with explicit surface color parameters
"""

import subprocess
from pathlib import Path

def create_surface_color_settings():
    """Create settings with explicit surface color."""
    settings_content = """# Mandelbulber settings file
# version 2.33
# Test explicit surface color

image_width=512
image_height=512
formula_1=2
N=250
power=8.0

# Camera
camera=3 -6 2
target=0 0 0
fov=53.13

# Lighting - maximum visibility
brightness=3.0
gamma=0.8
contrast=1.5
saturation=1.5

# Material with explicit colors
mat1_is_defined=true
mat1_surface_color=ffff 4000 0000
mat1_shading=1
mat1_specular=3.0
mat1_metallic=false
mat1_reflectance=1.5

# Strong auxiliary light
aux_light_enabled_1=true
aux_light_position_1=2 -2 -2
aux_light_intensity_1=3.0
aux_light_colour_1=ffff ffff ffff

# Strong fake lights
fake_lights_enabled=1
fake_lights_intensity=5.0
fake_lights_visibility=10.0

# Strong glow
glow_enabled=1
glow_intensity=2.0

# Basic parameters
detail_level=1.0
DE_factor=0.5
view_distance_max=20

# Output
file_destination=surface_color_test
save_image_format=0
"""
    
    output_dir = Path("output/mandelbulber_evolution")
    output_dir.mkdir(parents=True, exist_ok=True)
    settings_dir = output_dir / "settings"
    settings_dir.mkdir(parents=True, exist_ok=True)
    
    settings_file = settings_dir / "surface_color_test.fract"
    with open(settings_file, 'w') as f:
        f.write(settings_content)
    
    return str(settings_file)

def test_surface_color():
    """Test with explicit surface color."""
    print("🧪 Testing with explicit surface color...")
    settings_file = create_surface_color_settings()
    
    output_file = "output/mandelbulber_evolution/surface_color_test.png"
    
    cmd = [
        "flatpak", "run", "com.github.buddhi1980.mandelbulber2",
        "--nogui",
        "--format", "png", 
        "--res", "512x512",
        "--output", output_file,
        settings_file
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0 and Path(output_file).exists():
            print(f"✅ Surface color test successful! Output: {output_file}")
            print(f"📄 File size: {Path(output_file).stat().st_size} bytes")
            return True
        else:
            print(f"❌ Surface color test failed!")
            print(f"Return code: {result.returncode}")
            print(f"STDERR: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Exception occurred: {e}")
        return False

if __name__ == "__main__":
    test_surface_color()
