#!/usr/bin/env python3
"""
Test simple Mandelbulb formula with the absolute minimum settings needed.
"""

import subprocess
from pathlib import Path

def create_simple_mandelbulb():
    """Create simple Mandelbulb settings."""
    settings_content = """# Mandelbulber settings file
# version 2.33
# Simple Mandelbulb test

image_width=512  
image_height=512
formula_1=9
power=8.0
N=100

# Simple camera
camera=3 -6 2
target=0 0 0
fov=60

# Basic distance estimation
DE_thresh=0.01
DE_factor=1.0
view_distance_max=50

# Material - keep it simple
mat1_is_defined=true
mat1_surface_color=ff00 8000 4000
mat1_shading=1

# Very basic lighting - just aux light
aux_light_enabled_1=true
aux_light_position_1=3 -3 -3
aux_light_intensity_1=1.0
aux_light_colour_1=ffff ffff ffff

# Basic brightness
brightness=2.0

# Output
file_destination=simple_mandelbulb
save_image_format=0
"""
    
    output_dir = Path("output/mandelbulber_evolution")
    output_dir.mkdir(parents=True, exist_ok=True)
    settings_dir = output_dir / "settings"  
    settings_dir.mkdir(parents=True, exist_ok=True)
    
    settings_file = settings_dir / "simple_mandelbulb.fract"
    with open(settings_file, 'w') as f:
        f.write(settings_content)
    
    return str(settings_file)

def test_simple():
    """Test simple Mandelbulb."""
    print("🔬 Testing simple Mandelbulb with minimal settings...")
    settings_file = create_simple_mandelbulb()
    
    output_file = "output/mandelbulber_evolution/simple_mandelbulb.png"
    
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
            print(f"✅ Simple test successful! Output: {output_file}")
            return True
        else:
            print(f"❌ Simple test failed!")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

if __name__ == "__main__":
    test_simple()
