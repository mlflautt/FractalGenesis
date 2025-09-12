#!/usr/bin/env python3
"""
Absolutely minimal Mandelbulber settings to test basic rendering
"""

import subprocess
from pathlib import Path

def create_minimal_settings():
    """Create absolutely minimal settings."""
    # Just the absolute minimum needed
    settings_content = """# Mandelbulber settings file
# version 2.33
# Absolutely minimal test

image_width=512
image_height=512

file_destination=minimal_test
save_image_format=0
"""
    
    output_dir = Path("output/mandelbulber_evolution")
    output_dir.mkdir(parents=True, exist_ok=True)
    settings_dir = output_dir / "settings"
    settings_dir.mkdir(parents=True, exist_ok=True)
    
    settings_file = settings_dir / "minimal_test.fract"
    with open(settings_file, 'w') as f:
        f.write(settings_content)
    
    return str(settings_file)

def test_minimal():
    """Test with minimal settings, let Mandelbulber use all defaults."""
    print("🧪 Testing with absolutely minimal settings...")
    settings_file = create_minimal_settings()
    
    output_file = "output/mandelbulber_evolution/minimal_test.png"
    
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
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        
        if result.returncode == 0 and Path(output_file).exists():
            print(f"✅ Minimal test successful! Output: {output_file}")
            print(f"📄 File size: {Path(output_file).stat().st_size} bytes")
            return True
        else:
            print(f"❌ Minimal test failed!")
            return False
            
    except Exception as e:
        print(f"❌ Exception occurred: {e}")
        return False

if __name__ == "__main__":
    test_minimal()
