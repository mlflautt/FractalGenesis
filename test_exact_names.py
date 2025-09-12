#!/usr/bin/env python3
"""
Test using exact parameter names from --list
"""

import subprocess
from pathlib import Path

def create_exact_params_settings():
    """Create settings using exact parameter names and defaults."""
    settings_content = """# Mandelbulber settings file
# version 2.33
# Test with exact parameter names from --list

image_width=512
image_height=512

# Basic fractal settings (using defaults)
formula_1=2
N=250
power=8

# Camera (using defaults)
camera=3 -6 2
target=0 0 0
fov=53.13

# CRITICAL: Define material properly
mat1_is_defined=1
mat1_surface_color=ff00 8000 4000
mat1_shading=1
mat1_specular=2.27
mat1_surface_roughness=0.01
mat1_metallic=0
mat1_reflectance=1.0

# Use fake lights (they're in the defaults)
fake_lights_enabled=1
fake_lights_intensity=2.0
fake_lights_visibility=5.0
fake_lights_color=ffff ffff ffff

# Basic lighting
brightness=1.5
gamma=0.6
contrast=1.0

# Basic parameters
detail_level=1
DE_factor=1.0
view_distance_max=12
DE_thresh=0.01

# Output
file_destination=exact_params_test
save_image_format=0
"""
    
    output_dir = Path("output/mandelbulber_evolution")
    output_dir.mkdir(parents=True, exist_ok=True)
    settings_dir = output_dir / "settings"
    settings_dir.mkdir(parents=True, exist_ok=True)
    
    settings_file = settings_dir / "exact_params_test.fract"
    with open(settings_file, 'w') as f:
        f.write(settings_content)
    
    return str(settings_file)

def test_exact_params():
    """Test with exact parameter names."""
    print("🧪 Testing with exact parameter names from --list...")
    settings_file = create_exact_params_settings()
    
    output_file = "output/mandelbulber_evolution/exact_params_test.png"
    
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
        print("STDERR lines containing 'material':")
        for line in result.stderr.split('\n'):
            if 'material' in line.lower():
                print(f"  {line}")
        
        if result.returncode == 0 and Path(output_file).exists():
            print(f"✅ Exact params test successful! Output: {output_file}")
            print(f"📄 File size: {Path(output_file).stat().st_size} bytes")
            return True
        else:
            print(f"❌ Exact params test failed!")
            return False
            
    except Exception as e:
        print(f"❌ Exception occurred: {e}")
        return False

if __name__ == "__main__":
    test_exact_params()
