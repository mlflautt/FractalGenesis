#!/usr/bin/env python3
"""
Generate Sample Fractals for Viewing

Creates a variety of fractals in RESULTS_TO_VIEW folder for visual confirmation
that Mandelbulber integration is working and producing diverse results.
"""

import sys
import os
import time
from pathlib import Path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from renderers.mandelbulber.parameters import MandelbulberParameters
from renderers.mandelbulber.renderer import MandelbulberRenderer
from renderers.mandelbulber.templates import ParameterTemplates

def main():
    print("🎨 Generating Sample Fractals for Viewing")
    print("=" * 50)
    
    # Create output directory
    output_dir = Path("RESULTS_TO_VIEW")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize renderer
    renderer = MandelbulberRenderer(output_dir=str(output_dir))
    
    if not renderer.is_available:
        print("❌ Mandelbulber not available!")
        return
    
    print(f"📁 Output directory: {output_dir.absolute()}")
    print("🖼️  Generating 10 diverse fractal samples...")
    print()
    
    samples = []
    
    # Sample 1: Classic Blue Mandelbulb
    print("1️⃣ Classic Blue Mandelbulb...")
    params = MandelbulberParameters()
    params.camera.camera_x = 3.0
    params.camera.camera_y = -6.0  
    params.camera.camera_z = 2.0
    params.fractal.power = 8.0
    params.material.surface_color_r = 0.2
    params.material.surface_color_g = 0.4
    params.material.surface_color_b = 1.0
    params.render.image_width = 512
    params.render.image_height = 512
    samples.append(("01_classic_blue_mandelbulb", params))
    
    # Sample 2: Golden High-Power
    print("2️⃣ Golden High-Power Mandelbulb...")
    params = MandelbulberParameters() 
    params.camera.camera_x = 2.0
    params.camera.camera_y = -4.0
    params.camera.camera_z = 1.0
    params.fractal.power = 16.0
    params.material.surface_color_r = 1.0
    params.material.surface_color_g = 0.8
    params.material.surface_color_b = 0.2
    params.render.image_width = 512
    params.render.image_height = 512
    samples.append(("02_golden_highpower", params))
    
    # Sample 3: Close-up Red
    print("3️⃣ Close-up Red Mandelbulb...")
    params = MandelbulberParameters()
    params.camera.camera_x = 1.2
    params.camera.camera_y = -2.0
    params.camera.camera_z = 0.8
    params.fractal.power = 8.0
    params.material.surface_color_r = 1.0
    params.material.surface_color_g = 0.1
    params.material.surface_color_b = 0.1
    params.render.image_width = 512
    params.render.image_height = 512
    samples.append(("03_closeup_red", params))
    
    # Sample 4: Green Side View
    print("4️⃣ Green Side View...")
    params = MandelbulberParameters()
    params.camera.camera_x = -3.0
    params.camera.camera_y = 2.0
    params.camera.camera_z = -1.0
    params.fractal.power = 10.0
    params.material.surface_color_r = 0.1
    params.material.surface_color_g = 0.8
    params.material.surface_color_b = 0.3
    params.render.image_width = 512
    params.render.image_height = 512
    samples.append(("04_green_sideview", params))
    
    # Sample 5: Purple Power 12
    print("5️⃣ Purple Power 12...")
    params = MandelbulberParameters()
    params.camera.camera_x = 2.5
    params.camera.camera_y = -5.0
    params.camera.camera_z = 3.0
    params.fractal.power = 12.0
    params.material.surface_color_r = 0.8
    params.material.surface_color_g = 0.2
    params.material.surface_color_b = 0.9
    params.render.image_width = 512
    params.render.image_height = 512
    samples.append(("05_purple_power12", params))
    
    # Templates 6-8: Use built-in templates
    templates = ParameterTemplates.get_all_templates()
    template_names = list(templates.keys())[:3]
    
    for i, name in enumerate(template_names, 6):
        print(f"{i}️⃣ Template: {name}...")
        params = templates[name].copy()
        params.render.image_width = 512
        params.render.image_height = 512
        samples.append((f"{i:02d}_template_{name}", params))
    
    # Samples 9-10: Random variations
    for i in range(9, 11):
        print(f"{i}️⃣ Random Variation {i-8}...")
        params = MandelbulberParameters()
        params.randomize()
        params.render.image_width = 512
        params.render.image_height = 512
        samples.append((f"{i:02d}_random_var{i-8}", params))
    
    # Render all samples
    print("\n🎨 Rendering samples...")
    successful = 0
    
    for name, params in samples:
        print(f"  Rendering {name}...", end=" ")
        
        try:
            output_path = renderer.render_single(
                params,
                output_filename=f"{name}.png"
            )
            
            if output_path and output_path.exists():
                size = output_path.stat().st_size
                print(f"✅ {size} bytes")
                successful += 1
            else:
                print("❌ Failed")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Summary
    print(f"\n📊 RESULTS:")
    print(f"✅ Successfully rendered: {successful}/10 samples")
    print(f"📁 Location: {output_dir.absolute()}")
    print(f"🔍 View your fractals with:")
    print(f"    ls -la {output_dir}/")
    print(f"    # Or open the folder in your file manager")
    
    if successful >= 8:
        print("\n🎉 EXCELLENT! Mandelbulber integration is working perfectly!")
        print("   You should see diverse, colorful 3D fractals with different:")
        print("   - Camera angles and distances")
        print("   - Colors (blue, gold, red, green, purple)")  
        print("   - Fractal powers (8, 10, 12, 16)")
        print("   - Compositions and detail levels")
    else:
        print("\n⚠️  Some renders failed. Check the error messages above.")
    
    print(f"\n🖼️  All images are 512x512 pixels for easy viewing.")

if __name__ == "__main__":
    main()