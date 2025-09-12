#!/usr/bin/env python3
"""
Test with exact parameter names from a known working Mandelbulber example.
"""

import subprocess
from pathlib import Path

def create_exact_working_settings():
    """Create settings using exact parameter names from working example."""
    settings_content = """# Mandelbulber settings file
# version 2.33
# Using exact working parameters

image_width=512
image_height=512
formula_1=103

# Camera from working example
camera=5.557109284734602 1.484099428252363 -1.398417870763199
target=5.555285169825431 1.483126670556952 -1.398244365995301
camera_distance_to_target=0.002074549743998615
camera_rotation=118.0699536685941 4.797531081424072 0
fov=360

# Working auxiliary light configuration
aux_light_enabled_1=true
aux_light_position_1=5.532331132296927 1.456237487525709 -1.39693261395967
aux_light_intensity_1=0.0007466032
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

# Lighting from working example
brightness=1.45
gamma=0.42
DE_factor=0.7
view_distance_max=12

# Output
file_destination=exact_test
save_image_format=0
"""
    
    output_dir = Path("output/mandelbulber_evolution")
    output_dir.mkdir(parents=True, exist_ok=True)
    settings_dir = output_dir / "settings"
    settings_dir.mkdir(parents=True, exist_ok=True)
    
    settings_file = settings_dir / "exact_test.fract"
    with open(settings_file, 'w') as f:
        f.write(settings_content)
    
    return str(settings_file)

def test_exact_params():
    """Test with exact parameters from working example."""
    print("🧪 Testing with exact parameter names from working example...")
    settings_file = create_exact_working_settings()
    
    output_file = "output/mandelbulber_evolution/exact_test.png"
    
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
            print(f"✅ Exact params test successful! Output: {output_file}")
            print(f"📄 File size: {Path(output_file).stat().st_size} bytes")
            return True
        else:
            print(f"❌ Exact params test failed!")
            print(f"Return code: {result.returncode}")
            print(f"STDERR: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Exception occurred: {e}")
        return False

if __name__ == "__main__":
    test_exact_params()
