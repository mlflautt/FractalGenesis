#!/usr/bin/env python3
"""
GenesisRender Comprehensive Showcase
===================================

Complete demonstration of GenesisRender capabilities:
- All fractal types with real renders
- Advanced lighting demonstrations
- Smooth animation examples
- Parameter control verification
- Performance benchmarking
- Production-quality output

This generates actual renders and animations to showcase the full power
of the GenesisRender system.
"""

import sys
from pathlib import Path
import time
import json

# Add project root for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# We need to create the __init__.py and import structure first
from renderers.genesis_render.genesis_render import GenesisRender, GenesisRenderParams
from renderers.genesis_render.fractal_types import FRACTAL_PRESETS
from renderers.genesis_render.animation_system import AnimationTemplates

def create_genesis_init_files():
    """Create necessary __init__.py files for the genesis_render module"""
    genesis_dir = project_root / "renderers" / "genesis_render"
    genesis_dir.mkdir(exist_ok=True, parents=True)
    
    # Create __init__.py
    init_file = genesis_dir / "__init__.py"
    if not init_file.exists():
        init_content = '''"""
GenesisRender - Professional 3D Fractal Rendering System
=======================================================

Complete fractal rendering system with:
- 12+ fractal types
- Advanced lighting and materials
- Smooth animation system
- High-performance JIT rendering
"""

from .genesis_render import GenesisRender, GenesisRenderParams
from .fractal_types import FRACTAL_TYPES, FRACTAL_PRESETS
from .animation_system import AnimationSequence, AnimationTemplates

__all__ = ['GenesisRender', 'GenesisRenderParams', 'FRACTAL_TYPES', 'FRACTAL_PRESETS', 'AnimationSequence', 'AnimationTemplates']
'''
        with open(init_file, 'w') as f:
            f.write(init_content)

def demo_all_fractal_types():
    """Demonstrate all available fractal types"""
    print("=== GenesisRender Fractal Types Showcase ===\n")
    
    renderer = GenesisRender()
    output_dir = project_root / "output" / "genesis_render_showcase" / "fractal_types"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Test all available presets
    fractal_results = []
    
    for preset_name in FRACTAL_PRESETS.keys():
        print(f"Rendering: {preset_name}")
        
        # Create parameters from preset
        params = renderer.create_preset(preset_name)
        params.width = 600
        params.height = 600
        params.iterations = 80  # Reduced for speed
        
        # Enhance with advanced lighting
        params.ambient_occlusion_enabled = True
        params.hard_lighting_enabled = True
        params.ao_strength = 0.4
        
        # Render
        start_time = time.time()
        image, metrics = renderer.render(params)
        total_time = time.time() - start_time
        
        # Save with descriptive title
        title = f"GenesisRender - {preset_name.replace('_', ' ').title()}"
        output_file = output_dir / f"{preset_name}.png"
        renderer.save_image(image, output_file, title)
        
        # Store results
        result = {
            'preset_name': preset_name,
            'fractal_type': params.fractal_type,
            'render_time': metrics['render_time'],
            'total_time': total_time,
            'surface_pixels': metrics['surface_pixels'],
            'unique_colors': metrics['unique_colors'],
            'file_path': str(output_file)
        }
        fractal_results.append(result)
        
        print(f"  ✅ {metrics['render_time']:.2f}s - {metrics['surface_pixels']:,} surface pixels")
        print(f"     Colors: {metrics['unique_colors']:,}, Saved: {output_file.name}")
    
    # Save results summary
    summary_file = output_dir / "fractal_types_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(fractal_results, f, indent=2)
    
    return fractal_results

def demo_lighting_showcase():
    """Demonstrate advanced lighting capabilities"""
    print("\n=== Advanced Lighting Showcase ===\n")
    
    renderer = GenesisRender()
    output_dir = project_root / "output" / "genesis_render_showcase" / "lighting"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    lighting_configs = [
        {
            "name": "Basic Lighting",
            "config": {
                "hard_lighting_enabled": False,
                "diffuse_lighting_enabled": True,
                "ambient_occlusion_enabled": False,
                "global_ambient": 0.3
            }
        },
        {
            "name": "Hard Shadows", 
            "config": {
                "hard_lighting_enabled": True,
                "diffuse_lighting_enabled": False,
                "ambient_occlusion_enabled": False,
                "light1_hard_factor": 1.0,
                "global_ambient": 0.1,
                "shadow_softness": 0.001
            }
        },
        {
            "name": "Ambient Occlusion",
            "config": {
                "hard_lighting_enabled": False,
                "diffuse_lighting_enabled": True,
                "ambient_occlusion_enabled": True,
                "ao_strength": 0.6,
                "ao_samples": 8,
                "global_ambient": 0.2
            }
        },
        {
            "name": "Dramatic Three-Point",
            "config": {
                "hard_lighting_enabled": True,
                "diffuse_lighting_enabled": True,
                "ambient_occlusion_enabled": True,
                "light1_intensity": 1.2,
                "light1_color": (1.0, 0.95, 0.9),
                "light2_intensity": 0.4,
                "light2_color": (0.8, 0.9, 1.0),
                "light3_intensity": 0.6,
                "light3_pos": (0.0, -1.0, 2.0),
                "ao_strength": 0.4,
                "global_ambient": 0.08
            }
        }
    ]
    
    lighting_results = []
    
    for config in lighting_configs:
        print(f"Testing: {config['name']}")
        
        # Base parameters
        params = renderer.create_preset("mandelbulb_classic")
        params.width = 600
        params.height = 600
        params.color_palette = "fire"
        
        # Apply lighting configuration
        for key, value in config['config'].items():
            setattr(params, key, value)
        
        # Render
        image, metrics = renderer.render(params)
        
        # Save
        title = f"GenesisRender - {config['name']} Lighting"
        output_file = output_dir / f"{config['name'].lower().replace(' ', '_')}.png"
        renderer.save_image(image, output_file, title)
        
        result = {
            'name': config['name'],
            'render_time': metrics['render_time'],
            'surface_pixels': metrics['surface_pixels'],
            'features_enabled': metrics['features_enabled'],
            'file_path': str(output_file)
        }
        lighting_results.append(result)
        
        print(f"  ✅ {metrics['render_time']:.2f}s - Features: {list(metrics['features_enabled'].keys())}")
    
    return lighting_results

def demo_material_showcase():
    """Demonstrate advanced materials"""
    print("\n=== Material Properties Showcase ===\n")
    
    renderer = GenesisRender()
    output_dir = project_root / "output" / "genesis_render_showcase" / "materials"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    material_configs = [
        {
            "name": "Matte Surface",
            "metallic": 0.0,
            "roughness": 0.8,
            "color_palette": "warm"
        },
        {
            "name": "Polished Metal",
            "metallic": 0.9,
            "roughness": 0.1,
            "color_palette": "cool"
        },
        {
            "name": "Brushed Metal",
            "metallic": 0.7,
            "roughness": 0.4,
            "color_palette": "monochrome"
        },
        {
            "name": "Glossy Ceramic",
            "metallic": 0.2,
            "roughness": 0.05,
            "color_palette": "rainbow"
        }
    ]
    
    material_results = []
    
    for config in material_configs:
        print(f"Rendering: {config['name']}")
        
        # Base parameters with enhanced lighting
        params = renderer.create_preset("mandelbulb_classic")
        params.width = 500
        params.height = 500
        params.metallic = config['metallic']
        params.roughness = config['roughness']
        params.color_palette = config['color_palette']
        
        # Enhanced lighting for material demonstration
        params.light1_intensity = 1.0
        params.light2_intensity = 0.6
        params.specular_intensity = 0.8
        params.ambient_occlusion_enabled = True
        
        # Render
        image, metrics = renderer.render(params)
        
        # Save
        title = f"GenesisRender - {config['name']} Material"
        output_file = output_dir / f"{config['name'].lower().replace(' ', '_')}.png"
        renderer.save_image(image, output_file, title)
        
        result = {
            'name': config['name'],
            'metallic': config['metallic'],
            'roughness': config['roughness'],
            'render_time': metrics['render_time'],
            'file_path': str(output_file)
        }
        material_results.append(result)
        
        print(f"  ✅ {metrics['render_time']:.2f}s - M:{config['metallic']:.1f} R:{config['roughness']:.1f}")
    
    return material_results

def demo_animation_showcase():
    """Create comprehensive animations"""
    print("\n=== Animation Showcase ===\n")
    
    renderer = GenesisRender()
    output_dir = project_root / "output" / "genesis_render_showcase" / "animations"
    
    # Animation 1: Exploration flyby
    print("Creating: Exploration Flyby Animation")
    flyby_seq = AnimationTemplates.create_exploration_flyby()
    flyby_seq.set_duration(5.0, 15.0)  # 5 seconds, 15 fps for manageable size
    
    flyby_frames = renderer.create_animation(flyby_seq, output_dir / "exploration_flyby", "exploration_flyby")
    flyby_gif = renderer.create_gif(flyby_frames, output_dir / "exploration_flyby.gif", fps=15)
    
    # Animation 2: Zoom and morph
    print("\nCreating: Zoom and Morph Animation")
    zoom_seq = AnimationTemplates.create_zoom_and_morph()
    zoom_seq.set_duration(4.0, 12.0)  # 4 seconds, 12 fps
    
    zoom_frames = renderer.create_animation(zoom_seq, output_dir / "zoom_and_morph", "zoom_and_morph")
    zoom_gif = renderer.create_gif(zoom_frames, output_dir / "zoom_and_morph.gif", fps=12)
    
    # Animation 3: Fractal showcase
    print("\nCreating: Fractal Types Showcase Animation")
    showcase_seq = AnimationTemplates.create_fractal_showcase()
    showcase_seq.set_duration(8.0, 10.0)  # 8 seconds, 10 fps
    
    showcase_frames = renderer.create_animation(showcase_seq, output_dir / "fractal_showcase", "fractal_showcase")
    showcase_gif = renderer.create_gif(showcase_frames, output_dir / "fractal_showcase.gif", fps=10)
    
    animation_results = [
        {
            'name': 'Exploration Flyby',
            'duration': 5.0,
            'fps': 15,
            'frames': len(flyby_frames),
            'gif_path': str(flyby_gif)
        },
        {
            'name': 'Zoom and Morph',
            'duration': 4.0,
            'fps': 12,
            'frames': len(zoom_frames),
            'gif_path': str(zoom_gif)
        },
        {
            'name': 'Fractal Showcase',
            'duration': 8.0,
            'fps': 10,
            'frames': len(showcase_frames),
            'gif_path': str(showcase_gif)
        }
    ]
    
    return animation_results

def demo_performance_benchmark():
    """Benchmark GenesisRender performance"""
    print("\n=== Performance Benchmark ===\n")
    
    renderer = GenesisRender()
    
    # Test different resolutions and complexities
    benchmark_configs = [
        {"resolution": (400, 300), "iterations": 50, "name": "Low Quality"},
        {"resolution": (600, 450), "iterations": 80, "name": "Medium Quality"},
        {"resolution": (800, 600), "iterations": 100, "name": "High Quality"},
        {"resolution": (1200, 900), "iterations": 120, "name": "Ultra Quality"},
    ]
    
    benchmark_results = []
    
    for config in benchmark_configs:
        print(f"Benchmarking: {config['name']} ({config['resolution'][0]}x{config['resolution'][1]})")
        
        # Create test parameters
        params = renderer.create_preset("mandelbulb_classic")
        params.width = config['resolution'][0]
        params.height = config['resolution'][1]
        params.iterations = config['iterations']
        params.ambient_occlusion_enabled = True
        params.hard_lighting_enabled = True
        
        # Warm-up render
        if config == benchmark_configs[0]:
            print("  JIT warm-up...")
            renderer.render(params)
        
        # Benchmark render
        start_time = time.time()
        image, metrics = renderer.render(params)
        total_time = time.time() - start_time
        
        # Calculate performance metrics
        pixels = config['resolution'][0] * config['resolution'][1]
        pixels_per_second = pixels / metrics['render_time']
        
        result = {
            'name': config['name'],
            'resolution': config['resolution'],
            'iterations': config['iterations'],
            'render_time': metrics['render_time'],
            'total_time': total_time,
            'pixels': pixels,
            'pixels_per_second': int(pixels_per_second),
            'surface_pixels': metrics['surface_pixels'],
            'unique_colors': metrics['unique_colors']
        }
        benchmark_results.append(result)
        
        print(f"  ✅ {metrics['render_time']:.2f}s ({pixels_per_second:,.0f} pixels/sec)")
        print(f"     Surface: {metrics['surface_pixels']:,}, Colors: {metrics['unique_colors']:,}")
    
    return benchmark_results

def create_summary_report(fractal_results, lighting_results, material_results, 
                         animation_results, benchmark_results):
    """Create comprehensive summary report"""
    print("\n" + "="*70)
    print("📊 GENESISRENDER SHOWCASE COMPLETE - SUMMARY REPORT")
    print("="*70)
    
    # Fractal types summary
    print(f"\n🎨 Fractal Types Rendered: {len(fractal_results)}")
    total_fractal_time = sum(r['render_time'] for r in fractal_results)
    avg_fractal_time = total_fractal_time / len(fractal_results)
    print(f"   Total render time: {total_fractal_time:.1f}s")
    print(f"   Average per fractal: {avg_fractal_time:.2f}s")
    
    # Top fractals by complexity
    complex_fractals = sorted(fractal_results, key=lambda x: x['unique_colors'], reverse=True)[:3]
    print(f"   Most complex fractals:")
    for i, fractal in enumerate(complex_fractals):
        print(f"     {i+1}. {fractal['preset_name']}: {fractal['unique_colors']:,} colors")
    
    # Lighting summary
    print(f"\n💡 Lighting Configurations: {len(lighting_results)}")
    lighting_time = sum(r['render_time'] for r in lighting_results)
    print(f"   Total render time: {lighting_time:.1f}s")
    fastest_lighting = min(lighting_results, key=lambda x: x['render_time'])
    print(f"   Fastest: {fastest_lighting['name']} ({fastest_lighting['render_time']:.2f}s)")
    
    # Material summary
    print(f"\n🔘 Material Variations: {len(material_results)}")
    material_time = sum(r['render_time'] for r in material_results)
    print(f"   Total render time: {material_time:.1f}s")
    
    # Animation summary
    print(f"\n🎬 Animations Created: {len(animation_results)}")
    total_frames = sum(r['frames'] for r in animation_results)
    total_duration = sum(r['duration'] for r in animation_results)
    print(f"   Total frames: {total_frames}")
    print(f"   Total duration: {total_duration:.1f}s")
    
    for anim in animation_results:
        print(f"     {anim['name']}: {anim['frames']} frames @ {anim['fps']} fps")
    
    # Performance summary
    print(f"\n⚡ Performance Benchmark:")
    for bench in benchmark_results:
        print(f"   {bench['name']}: {bench['pixels_per_second']:,} pixels/sec")
    
    best_performance = max(benchmark_results, key=lambda x: x['pixels_per_second'])
    print(f"   Peak throughput: {best_performance['pixels_per_second']:,} pixels/sec")
    
    # Overall statistics
    total_render_time = (total_fractal_time + lighting_time + material_time + 
                        sum(r['frames'] * 0.5 for r in animation_results))  # Estimate animation time
    total_images = len(fractal_results) + len(lighting_results) + len(material_results) + total_frames
    
    print(f"\n📈 Overall Statistics:")
    print(f"   Total images generated: {total_images}")
    print(f"   Estimated total render time: {total_render_time:.1f}s ({total_render_time/60:.1f} minutes)")
    print(f"   Average time per image: {total_render_time/total_images:.2f}s")
    
    # Output locations
    print(f"\n📁 Output Locations:")
    print(f"   Static images: output/genesis_render_showcase/")
    print(f"   Animations: output/genesis_render_showcase/animations/")
    
    print(f"\n🎯 GenesisRender Capabilities Demonstrated:")
    print(f"   ✅ {len(FRACTAL_PRESETS)} fractal types with distance estimation")
    print(f"   ✅ Advanced lighting (hard shadows, AO, multi-light)")
    print(f"   ✅ Material properties (metallic, roughness, PBR)")
    print(f"   ✅ Smooth animations with keyframe interpolation")
    print(f"   ✅ High-performance JIT compilation ({best_performance['pixels_per_second']:,} pixels/sec)")
    print(f"   ✅ Professional-quality output suitable for production")
    
    # Save complete report
    output_dir = project_root / "output" / "genesis_render_showcase"
    report_file = output_dir / "complete_showcase_report.json"
    
    complete_report = {
        'renderer_info': {
            'name': 'GenesisRender',
            'version': '1.0.0',
            'showcase_date': time.strftime('%Y-%m-%d %H:%M:%S')
        },
        'fractal_types': fractal_results,
        'lighting_tests': lighting_results,
        'material_tests': material_results,
        'animations': animation_results,
        'performance_benchmark': benchmark_results,
        'summary_stats': {
            'total_images': total_images,
            'total_render_time': total_render_time,
            'peak_performance_pixels_per_sec': best_performance['pixels_per_second'],
            'avg_time_per_image': total_render_time / total_images
        }
    }
    
    with open(report_file, 'w') as f:
        json.dump(complete_report, f, indent=2)
    
    print(f"\n📄 Complete report saved: {report_file}")
    
    return complete_report

def main():
    """Run the complete GenesisRender showcase"""
    print("🌟 GenesisRender Comprehensive Showcase")
    print("=====================================")
    print("Generating real renders and animations to demonstrate full capabilities\n")
    
    # Create necessary init files
    create_genesis_init_files()
    
    try:
        # Create output directory
        output_base = project_root / "output" / "genesis_render_showcase"
        output_base.mkdir(exist_ok=True, parents=True)
        
        # Run all demonstrations
        print("Starting comprehensive showcase...")
        showcase_start = time.time()
        
        # 1. Fractal types
        fractal_results = demo_all_fractal_types()
        
        # 2. Lighting showcase  
        lighting_results = demo_lighting_showcase()
        
        # 3. Material properties
        material_results = demo_material_showcase()
        
        # 4. Performance benchmark
        benchmark_results = demo_performance_benchmark()
        
        # 5. Animations (most time-consuming)
        animation_results = demo_animation_showcase()
        
        # 6. Create comprehensive report
        complete_report = create_summary_report(
            fractal_results, lighting_results, material_results,
            animation_results, benchmark_results
        )
        
        showcase_time = time.time() - showcase_start
        
        print(f"\n🎉 SHOWCASE COMPLETE!")
        print(f"Total time: {showcase_time:.1f}s ({showcase_time/60:.1f} minutes)")
        print(f"🌟 GenesisRender - Professional 3D Fractal Rendering System")
        print(f"   Ready for production use in FractalGenesis evolution pipeline!")
        
    except Exception as e:
        print(f"❌ Showcase failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()