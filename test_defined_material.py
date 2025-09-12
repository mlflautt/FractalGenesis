#!/usr/bin/env python3
"""
Test with mat1_is_defined=1 to fix the black silhouette issue
"""

import subprocess
from pathlib import Path

def create_defined_material_settings():
    """Create settings with mat1_is_defined=1."""
    settings_content = """# Mandelbulber settings file
# version 2.33
# Test with properly defined material

image_width=512
image_height=512
formula_1=2
N=250
power=8.0

# Camera
camera=3 -6 2
target=0 0 0
fov=53.13

# THE CRITICAL FIX: Enable material definition
mat1_is_defined=1

# Material settings
mat1_surface_color=ff00 8000 4000
mat1_shading=1
mat1_specular=2.27
mat1_metallic=false
mat1_reflectance=1.0

# Lighting
brightness=1.5
gamma=0.6
aux_light_enabled_1=true
aux_light_position_1=3 -3 -3
aux_light_intensity_1=1.5
aux_light_colour_1=ffff ffff ffff

# Output
file_destination=defined_material_test
save_image_format=0
"""
    
    output_dir = Path("output/mandelbulber_evolution")
    output_dir.mkdir(parents=True, exist_ok=True)
    settings_dir = output_dir / "settings"
    settings_dir.mkdir(parents=True, exist_ok=True)
    
    settings_file = settings_dir / "defined_material_test.fract"
    with open(settings_file, 'w') as f:
        f.write(settings_content)
    
    return str(settings_file)

def test_defined_material():
    """Test with mat1_is_defined=1."""
    print("🧪 Testing with mat1_is_defined=1...")
    settings_file = create_defined_material_settings()
    
    output_file = "output/mandelbulber_evolution/defined_material_test.png"
    
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
        
        print(f"Return code: {result.returncode}")
        print(f"STDERR: {result.stderr}")
        
        if result.returncode == 0 and Path(output_file).exists():
            print(f"✅ Defined material test successful! Output: {output_file}")
            print(f"📄 File size: {Path(output_file).stat().st_size} bytes")
            return True
        else:
            print(f"❌ Defined material test failed!")
            return False
            
    except Exception as e:
        print(f"❌ Exception occurred: {e}")
        return False

if __name__ == "__main__":
    test_defined_material()
