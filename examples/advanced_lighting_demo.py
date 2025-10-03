#!/usr/bin/env python3
"""
Advanced Lighting Demo for Python 3D Fractal Renderer
====================================================

This demonstrates the new advanced lighting features inspired by Mandelbulb3D:
- Hard lighting with sharp shadows
- Soft diffuse lighting  
- Ambient occlusion
- Multiple light sources
- Subsurface scattering
- Professional lighting setups
"""

import sys
from pathlib import Path
import time
import matplotlib.pyplot as plt

# Add the project root to the path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from renderers.python_3d.advanced_lighting import (
    AdvancedFractalRenderer, AdvancedFractalParams, AdvancedLightingParams,
    AdvancedLightingPresets
)

def demo_lighting_comparison():
    """Compare different lighting setups on the same fractal"""
    print("=== Advanced Lighting Comparison Demo ===\n")
    
    renderer = AdvancedFractalRenderer()
    
    # Base fractal parameters
    base_params = {
        "fractal_type": "mandelbulb",
        "power": 8.0,
        "width": 600, "height": 600,
        "camera_pos": (0.0, 0.0, -3.0),
        "color_palette": "warm",
        "coloring_mode": "orbit_trap",
        "iterations": 100
    }
    
    # Different lighting setups to compare
    lighting_demos = [
        {
            "name": "Mandelbulb3D Classic",
            "lighting": AdvancedLightingPresets.mandelbulb3d_classic(),
            "description": "Classic hard+diffuse lighting with ambient occlusion"
        },
        {
            "name": "Soft Artistic",
            "lighting": AdvancedLightingPresets.soft_artistic(),
            "description": "Soft lighting with subsurface scattering"
        },
        {
            "name": "Dramatic Contrast", 
            "lighting": AdvancedLightingPresets.dramatic_contrast(),
            "description": "High contrast hard lighting with deep shadows"
        },
        {
            "name": "Three-Point Studio",
            "lighting": AdvancedLightingPresets.three_point_studio(),
            "description": "Professional key/fill/rim lighting setup"
        }
    ]
    
    output_dir = project_root / "output" / "advanced_lighting_demo"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    results = []
    
    for demo in lighting_demos:
        print(f"Rendering: {demo['name']}")
        print(f"  {demo['description']}")
        
        # Create parameters with advanced lighting
        params = AdvancedFractalParams(**base_params, lighting=demo["lighting"])
        
        start_time = time.time()
        image, metrics = renderer.render_advanced(params)
        total_time = time.time() - start_time
        
        # Save the result
        plt.figure(figsize=(10, 10))
        plt.imshow(image, origin='upper')
        plt.axis('off')
        plt.title(f"{demo['name']}\n{demo['description']}", fontsize=12, pad=20)
        
        output_file = output_dir / f"{demo['name'].lower().replace(' ', '_').replace('-', '_')}.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=100, bbox_inches='tight')
        plt.close()
        
        # Store results
        result = {
            "name": demo["name"],
            "description": demo["description"],
            "render_time": metrics["render_time"],
            "total_time": total_time,
            "surface_pixels": metrics["surface_pixels"],
            "unique_colors": metrics["unique_colors"],
            "lighting_features": metrics["lighting_features"],
            "file_path": str(output_file)
        }
        results.append(result)
        
        print(f"  ✅ Rendered in {metrics['render_time']:.2f}s")
        print(f"     Features: {list(f for f, enabled in metrics['lighting_features'].items() if enabled and f != 'num_lights')}")
        print(f"     Lights: {metrics['lighting_features']['num_lights']}")
        print(f"     Surface: {metrics['surface_pixels']:,} pixels")
        print(f"     Saved: {output_file.name}\n")
    
    return results

def demo_lighting_features():
    """Demonstrate individual lighting features"""
    print("=== Individual Lighting Features Demo ===\n")
    
    renderer = AdvancedFractalRenderer()
    
    base_params = {
        "fractal_type": "mandelbulb",
        "power": 8.0,
        "width": 500, "height": 500,
        "camera_pos": (0.0, 0.0, -3.0),
        "color_palette": "fire",
        "iterations": 80
    }
    
    # Individual feature tests
    feature_tests = [
        {
            "name": "Basic Diffuse Only",
            "lighting": AdvancedLightingParams(
                hard_lighting_enabled=False,
                diffuse_lighting_enabled=True,
                ambient_occlusion_enabled=False,
                global_ambient=0.2
            )
        },
        {
            "name": "Hard Lighting Only", 
            "lighting": AdvancedLightingParams(
                hard_lighting_enabled=True,
                diffuse_lighting_enabled=False,
                ambient_occlusion_enabled=False,
                light1_hard_factor=1.0,
                global_ambient=0.1
            )
        },
        {
            "name": "Ambient Occlusion Focus",
            "lighting": AdvancedLightingParams(
                hard_lighting_enabled=False,
                diffuse_lighting_enabled=True,
                ambient_occlusion_enabled=True,
                ao_strength=0.8,
                ao_samples=8,
                light1_intensity=0.6,
                global_ambient=0.3
            )
        },
        {
            "name": "Subsurface Scattering",
            "lighting": AdvancedLightingParams(
                hard_lighting_enabled=False,
                diffuse_lighting_enabled=True,
                ambient_occlusion_enabled=False,
                subsurface_scattering_enabled=True,
                subsurface_color=(1.0, 0.6, 0.3),
                transmittance=0.3,
                light1_intensity=0.8
            )
        }
    ]
    
    output_dir = project_root / "output" / "advanced_lighting_demo" / "features"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    for test in feature_tests:
        print(f"Testing: {test['name']}")
        
        params = AdvancedFractalParams(**base_params, lighting=test["lighting"])
        image, metrics = renderer.render_advanced(params)
        
        # Save result
        plt.figure(figsize=(8, 8))
        plt.imshow(image, origin='upper')
        plt.axis('off')
        plt.title(test['name'], fontsize=14, pad=15)
        
        output_file = output_dir / f"{test['name'].lower().replace(' ', '_')}.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Rendered in {metrics['render_time']:.2f}s")
        print(f"     Saved: {output_file.name}\n")

def demo_custom_lighting():
    """Create a custom lighting setup"""
    print("=== Custom Lighting Setup Demo ===\n")
    
    renderer = AdvancedFractalRenderer()
    
    # Create a custom dramatic lighting setup
    custom_lighting = AdvancedLightingParams(
        # Enable most features
        hard_lighting_enabled=True,
        diffuse_lighting_enabled=True,
        ambient_occlusion_enabled=True,
        subsurface_scattering_enabled=True,
        
        # Key light (warm, strong, from top-right)
        light1_pos=(3.0, 2.0, -1.0),
        light1_intensity=1.5,
        light1_color=(1.0, 0.9, 0.8),  # Warm white
        light1_hard_factor=0.8,
        
        # Fill light (cool, softer, from left)
        light2_pos=(-2.0, 1.0, -2.0),
        light2_intensity=0.4,
        light2_color=(0.8, 0.9, 1.0),  # Cool blue
        light2_hard_factor=0.2,
        
        # Rim light (strong, from behind)
        light3_pos=(0.0, -1.0, 3.0),
        light3_intensity=0.8,
        light3_color=(1.0, 0.8, 0.6),  # Orange rim
        light3_hard_factor=0.9,
        
        # Strong AO for depth
        ao_strength=0.5,
        ao_radius=0.15,
        ao_samples=6,
        
        # Low ambient for drama
        global_ambient=0.08,
        
        # Sharp shadows
        shadow_softness=0.01,
        
        # Strong specular
        specular_power=80.0,
        specular_intensity=0.6,
        
        # Subtle subsurface scattering
        subsurface_color=(1.0, 0.7, 0.4),
        subsurface_radius=0.08,
        transmittance=0.12
    )
    
    params = AdvancedFractalParams(
        fractal_type="mandelbulb",
        power=10.0,
        width=800, height=600,
        camera_pos=(0.0, 0.5, -3.5),
        target=(0.0, 0.0, 0.0),
        color_palette="rainbow",
        coloring_mode="orbit_trap",
        iterations=120,
        metallic=0.2,
        roughness=0.3,
        lighting=custom_lighting
    )
    
    print("Rendering custom dramatic lighting setup...")
    start_time = time.time()
    image, metrics = renderer.render_advanced(params)
    total_time = time.time() - start_time
    
    output_dir = project_root / "output" / "advanced_lighting_demo"
    output_file = output_dir / "custom_dramatic_lighting.png"
    
    # Create detailed visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Main render
    ax1.imshow(image, origin='upper')
    ax1.axis('off')
    ax1.set_title("Custom Dramatic Lighting\nMandelbulb with 3-Point Setup", fontsize=14, pad=20)
    
    # Lighting diagram (simplified)
    ax2.text(0.1, 0.9, "Lighting Setup:", fontsize=12, weight='bold', transform=ax2.transAxes)
    ax2.text(0.1, 0.8, "🔆 Key Light: Warm (top-right)", fontsize=10, transform=ax2.transAxes)
    ax2.text(0.1, 0.7, "💡 Fill Light: Cool (left)", fontsize=10, transform=ax2.transAxes)  
    ax2.text(0.1, 0.6, "✨ Rim Light: Orange (behind)", fontsize=10, transform=ax2.transAxes)
    
    ax2.text(0.1, 0.45, "Features Enabled:", fontsize=12, weight='bold', transform=ax2.transAxes)
    features = [
        f"Hard Lighting: {custom_lighting.hard_lighting_enabled}",
        f"Diffuse Lighting: {custom_lighting.diffuse_lighting_enabled}", 
        f"Ambient Occlusion: {custom_lighting.ambient_occlusion_enabled}",
        f"Subsurface Scattering: {custom_lighting.subsurface_scattering_enabled}",
        f"Shadow Softness: {custom_lighting.shadow_softness}",
        f"Specular Power: {custom_lighting.specular_power}",
    ]
    
    for i, feature in enumerate(features):
        ax2.text(0.1, 0.35 - i*0.05, feature, fontsize=9, transform=ax2.transAxes)
    
    ax2.text(0.1, 0.02, f"Render Time: {metrics['render_time']:.2f}s", 
             fontsize=10, weight='bold', transform=ax2.transAxes)
    
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Custom render complete!")
    print(f"   Render time: {metrics['render_time']:.2f}s")
    print(f"   Surface pixels: {metrics['surface_pixels']:,}")
    print(f"   Unique colors: {metrics['unique_colors']:,}")
    print(f"   Features: {metrics['lighting_features']}")
    print(f"   Saved: {output_file.name}")

def main():
    """Run the complete advanced lighting demo"""
    print("🌟 Advanced Lighting Demo - Mandelbulb3D-Style Features\\n")
    print("This demonstrates advanced lighting capabilities inspired by Mandelbulb3D\\n")
    
    try:
        # Run all demos
        print("Demo 1: Lighting Comparison")
        comparison_results = demo_lighting_comparison()
        
        print("Demo 2: Individual Features")
        demo_lighting_features()
        
        print("Demo 3: Custom Lighting")
        demo_custom_lighting()
        
        # Summary
        print("\\n" + "="*60)
        print("🎯 ADVANCED LIGHTING DEMO COMPLETE")
        print("="*60)
        
        print(f"✅ Lighting Comparison: {len(comparison_results)} setups rendered")
        avg_render_time = sum(r["render_time"] for r in comparison_results) / len(comparison_results)
        print(f"   Average render time: {avg_render_time:.2f}s per frame")
        
        print("✅ Individual Features: 4 feature tests completed")
        print("✅ Custom Lighting: Dramatic 3-point setup rendered")
        
        print(f"\\n📁 All outputs saved to: output/advanced_lighting_demo/")
        
        print("\\n🚀 Advanced Lighting Features Ready!")
        print("   ✨ Hard lighting with sharp shadows")
        print("   🌅 Soft diffuse lighting gradients") 
        print("   🕳️  Ambient occlusion for depth")
        print("   💡 Up to 3 configurable light sources")
        print("   🔮 Subsurface scattering simulation")
        print("   ⚡ High-performance JIT compilation")
        
        print("\\n🎨 Professional Lighting Presets:")
        print("   - Mandelbulb3D Classic")
        print("   - Soft Artistic")
        print("   - Dramatic Contrast") 
        print("   - Three-Point Studio")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()