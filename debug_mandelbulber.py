#!/usr/bin/env python3
"""
Debug script to test Mandelbulber lighting and visibility.
Creates a minimal settings file with extreme glow to see if we can make anything visible.
"""

import os
import subprocess
from pathlib import Path

def create_test_settings():
    """Create minimal test settings with extreme visibility settings."""
    settings_content = """# Mandelbulber settings file
# version 2.33
# Minimal debug test for lighting

image_width=512
image_height=512
formula_1=9
power=8.0
detail_level=1.0
N=100

# Camera settings  
camera=3 -6 2
target=0 0 0
fov=60
camera_rotation=26.565 -16.60154 0

# Fractal settings
DE_thresh=0.01
DE_factor=1.0
view_distance_max=50
view_distance_min=1e-15

# Material 1 - bright and visible
mat1_is_defined=1
mat1_surface_color=ffff 0000 0000
mat1_shading=1
mat1_specular=10.0
mat1_surface_roughness=0.01
mat1_metallic=1.0
mat1_reflectance=1.0

# EXTREME glow for visibility testing  
glow_enabled=1
glow_intensity=10.0
glow_color_1=ffff ffff ffff
glow_color_2=ffff 0000 0000

# Extreme brightness
brightness=10.0
contrast=3.0
gamma=1.0
saturation=2.0

# Enable all lighting methods
fake_lights_enabled=1
fake_lights_intensity=5.0
fake_lights_color=ffff ffff ffff
fake_lights_visibility=10.0
fake_lights_thickness=5

all_lights_intensity=5.0

# Background - dark for contrast
background_3_colors_enable=0
background_color_1=0000 0000 0000

# Output
file_destination=debug_test
save_image_format=0
"""
    
    output_dir = Path("output/mandelbulber_evolution")
    output_dir.mkdir(parents=True, exist_ok=True)
    settings_dir = output_dir / "settings"
    settings_dir.mkdir(parents=True, exist_ok=True)
    
    settings_file = settings_dir / "debug_test.fract"
    with open(settings_file, 'w') as f:
        f.write(settings_content)
    
    return str(settings_file)

def test_mandelbulber():
    """Test basic Mandelbulber rendering with extreme settings."""
    print("🔍 Debug: Creating minimal test settings...")
    settings_file = create_test_settings()
    
    output_file = "output/mandelbulber_evolution/debug_test.png"
    
    # Test render
    cmd = [
        "flatpak", "run", "com.github.buddhi1980.mandelbulber2",
        "--nogui",
        "--format", "png", 
        "--res", "512x512",
        "--output", output_file,
        settings_file
    ]
    
    print(f"🚀 Debug: Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0 and Path(output_file).exists():
            print(f"✅ Debug: Render successful! Output: {output_file}")
            print(f"📄 Debug: File size: {Path(output_file).stat().st_size} bytes")
            return True
        else:
            print(f"❌ Debug: Render failed!")
            print(f"Return code: {result.returncode}")  
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Debug: Exception occurred: {e}")
        return False

if __name__ == "__main__":
    test_mandelbulber()
