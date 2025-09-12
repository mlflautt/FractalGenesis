#!/usr/bin/env python3
"""
Test with formula_1=2 (default) instead of 103
"""

import subprocess
from pathlib import Path

def create_formula_2_settings():
    """Create settings using formula_1=2."""
    settings_content = """# Mandelbulber settings file
# version 2.33
# Using formula_1=2 instead of 103

image_width=512
image_height=512
formula_1=2

# Camera from working example
camera=3 -6 2
target=0 0 0
camera_distance_to_target=7
camera_rotation=26.565 -16.60154 0
fov=53.13

# Working auxiliary light configuration
aux_light_enabled_1=true
aux_light_position_1=3 -3 -3
aux_light_intensity_1=1.3
aux_light_colour_1=e800 9500 4200
aux_light_visibility=6.0256
aux_light_visibility_size=0.562341

# Material from working example
mat1_is_defined=true
mat1_metallic=false
mat1_reflectance=1
mat1_specular=2.27
mat1_specular_width=0.437
mat1_surface_color_gradient=0 9e8e5d 1999 4f2744 3999 ed4852 5999 380091 7999 fcbe3d
mat1_coloring_speed=2
mat1_fresnel_reflectance=true
mat1_use_colors_from_palette=1
mat1_surface_gradient_enable=1
mat1_shading=1

# Lighting from working example
brightness=1.45
gamma=0.42
DE_factor=0.7
view_distance_max=12
detail_level=1
N=250
power=8

# Fake lights
fake_lights_enabled=1
fake_lights_intensity=2.0
fake_lights_visibility=5.0

# Output
file_destination=formula_2_test
save_image_format=0
"""
    
    output_dir = Path("output/mandelbulber_evolution")
    output_dir.mkdir(parents=True, exist_ok=True)
    settings_dir = output_dir / "settings"
    settings_dir.mkdir(parents=True, exist_ok=True)
    
    settings_file = settings_dir / "formula_2_test.fract"
    with open(settings_file, 'w') as f:
        f.write(settings_content)
    
    return str(settings_file)

def test_formula_2():
    """Test with formula_1=2."""
    print("🧪 Testing with formula_1=2...")
    settings_file = create_formula_2_settings()
    
    output_file = "output/mandelbulber_evolution/formula_2_test.png"
    
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
            print(f"✅ Formula 2 test successful! Output: {output_file}")
            print(f"📄 File size: {Path(output_file).stat().st_size} bytes")
            return True
        else:
            print(f"❌ Formula 2 test failed!")
            print(f"Return code: {result.returncode}")
            print(f"STDERR: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Exception occurred: {e}")
        return False

if __name__ == "__main__":
    test_formula_2()
