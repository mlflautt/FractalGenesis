#!/usr/bin/env python3
"""
Python 3D Fractal Renderer Demo
==============================

Demonstrates the new Python-based 3D fractal rendering system integrated
into FractalGenesis as a replacement for Mandelbulber.

This shows:
- Basic rendering with different fractal types
- Animation creation with parameter interpolation
- Integration with the FractalGenesis architecture
- Performance and quality verification
"""

import sys
from pathlib import Path
import time

# Add the project root to the path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from renderers.python_3d import FractalRenderer, FractalParams, FractalPresets

def demo_basic_rendering():
    """Demonstrate basic fractal rendering capabilities"""
    print("=== Python 3D Fractal Renderer Demo ===\n")
    
    # Create renderer instance
    renderer = FractalRenderer()
    
    # Demo different fractal types and configurations
    demo_configs = [
        {
            "name": "Classic Mandelbulb",
            "params": FractalParams(
                fractal_type="mandelbulb",
                power=8.0,
                width=600, height=450,
                camera_pos=(0.0, 0.0, -3.0),
                color_palette="warm",
                coloring_mode="orbit_trap"
            )
        },
        {
            "name": "3D Julia Set",
            "params": FractalParams(
                fractal_type="julia",
                power=8.0,
                julia_c=(-0.2, 0.1, 0.0),
                width=600, height=450,
                camera_pos=(0.0, 0.0, -2.5),
                color_palette="cool",
                coloring_mode="distance"
            )
        },
        {
            "name": "High-Power Mandelbulb",
            "params": FractalParams(
                fractal_type="mandelbulb",
                power=12.0,
                width=600, height=450,
                camera_pos=(0.0, 0.0, -3.0),
                color_palette="fire",
                metallic=0.3,
                roughness=0.1
            )
        }
    ]
    
    output_dir = project_root / "output" / "python_3d_demo"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    results = []
    
    for config in demo_configs:
        print(f"Rendering: {config['name']}")
        start_time = time.time()
        
        # Render the fractal
        image, metrics = renderer.render(config["params"])
        
        # Save the result
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 9))
        plt.imshow(image, origin='upper')
        plt.axis('off')
        plt.title(f"{config['name']} - Python 3D Renderer")
        plt.tight_layout()
        
        output_file = output_dir / f"{config['name'].lower().replace(' ', '_')}.png"
        plt.savefig(output_file, dpi=100, bbox_inches='tight')
        plt.close()
        
        total_time = time.time() - start_time
        
        result = {
            "name": config["name"],
            "render_time": metrics["render_time"],
            "total_time": total_time,
            "surface_pixels": metrics["surface_pixels"],
            "unique_colors": metrics["unique_colors"],
            "file_path": str(output_file)
        }
        results.append(result)
        
        print(f"  ✅ Rendered in {metrics['render_time']:.2f}s")
        print(f"     Surface: {metrics['surface_pixels']:,} pixels")
        print(f"     Colors: {metrics['unique_colors']:,}")
        print(f"     Saved: {output_file.name}\n")
    
    return results

def demo_animation():
    """Demonstrate animation capabilities"""
    print("=== Animation Demo ===\n")
    
    renderer = FractalRenderer()
    
    # Create a power morphing animation
    start_params = FractalParams(
        fractal_type="mandelbulb",
        power=2.0,
        width=400, height=300,
        camera_pos=(0.0, 0.0, -3.0),
        color_palette="fire"
    )
    
    end_params = FractalParams(
        fractal_type="mandelbulb",
        power=12.0,
        width=400, height=300,
        camera_pos=(0.0, 0.0, -3.0),
        color_palette="ice"
    )
    
    output_dir = project_root / "output" / "python_3d_demo" / "animation"
    
    print("Creating power morphing animation (6 frames)...")
    start_time = time.time()
    
    frame_paths = renderer.create_animation(
        start_params, end_params,
        num_frames=6,
        output_dir=output_dir,
        name="power_morph_demo"
    )
    
    # Create GIF
    gif_path = output_dir.parent / "power_morph_demo.gif"
    renderer.create_gif(frame_paths, gif_path, fps=1)
    
    total_time = time.time() - start_time
    
    print(f"✅ Animation complete! Total time: {total_time:.1f}s")
    print(f"   Frames: {len(frame_paths)}")
    print(f"   GIF: {gif_path.name}")
    
    return frame_paths, gif_path

def demo_integration_example():
    """Show how this integrates with FractalGenesis evolution"""
    print("=== FractalGenesis Integration Example ===\n")
    
    # Simulate how this would work in evolution
    print("Simulating evolutionary parameter generation...")
    
    renderer = FractalRenderer()
    
    # Example: Generate random parameters as evolution would
    import random
    
    def random_fractal_params():
        """Generate random parameters for evolution simulation"""
        power = random.uniform(2.0, 16.0)
        distance = random.uniform(2.0, 5.0)
        angle = random.uniform(0, 6.28)  # 0 to 2π
        
        camera_x = distance * 0.3 * (random.random() - 0.5)
        camera_y = distance * 0.3 * (random.random() - 0.5)
        camera_z = -distance
        
        return FractalParams(
            fractal_type=random.choice(["mandelbulb", "julia"]),
            power=power,
            julia_c=(random.uniform(-0.3, 0.3), random.uniform(-0.3, 0.3), random.uniform(-0.3, 0.3)),
            width=300, height=300,  # Smaller for evolution speed
            camera_pos=(camera_x, camera_y, camera_z),
            color_palette=random.choice(["warm", "cool", "rainbow", "fire", "ice"]),
            iterations=random.randint(50, 120),
            metallic=random.uniform(0.0, 0.5),
            roughness=random.uniform(0.1, 0.8)
        )
    
    # Generate a "population" of fractals
    population_size = 5
    population_results = []
    
    print(f"Generating population of {population_size} fractals...")
    
    for i in range(population_size):
        params = random_fractal_params()
        
        print(f"  Individual {i+1}: {params.fractal_type}, power={params.power:.1f}")
        
        # This is how evolution would evaluate fitness
        image, metrics = renderer.render(params)
        
        # Calculate fitness (example: prefer diverse colors and good surface coverage)
        fitness = (metrics["unique_colors"] / 50000.0) + (metrics["surface_pixels"] / 90000.0)
        fitness = min(1.0, fitness)  # Normalize to 0-1
        
        population_results.append({
            "individual": i+1,
            "params": params,
            "metrics": metrics,
            "fitness": fitness
        })
        
        print(f"    Rendered in {metrics['render_time']:.2f}s, fitness: {fitness:.3f}")
    
    # Sort by fitness (evolution selection)
    population_results.sort(key=lambda x: x["fitness"], reverse=True)
    
    print(f"\n🏆 Evolution Results (sorted by fitness):")
    for i, result in enumerate(population_results):
        print(f"  Rank {i+1}: Individual {result['individual']}, fitness {result['fitness']:.3f}")
        print(f"           {result['params'].fractal_type}, power={result['params'].power:.1f}")
    
    return population_results

def main():
    """Run the complete demo"""
    print("🌟 Python 3D Fractal Renderer Integration Demo\n")
    print("This demonstrates the new Python-based system that replaces Mandelbulber\n")
    
    # Run all demos
    try:
        # Basic rendering
        basic_results = demo_basic_rendering()
        
        # Animation
        anim_frames, gif_path = demo_animation()
        
        # Evolution integration
        evolution_results = demo_integration_example()
        
        # Summary
        print("\n" + "="*60)
        print("🎯 DEMO COMPLETE - SUMMARY")
        print("="*60)
        
        print(f"✅ Basic Rendering: {len(basic_results)} fractals generated")
        avg_render_time = sum(r["render_time"] for r in basic_results) / len(basic_results)
        print(f"   Average render time: {avg_render_time:.2f}s per frame")
        
        print(f"✅ Animation: {len(anim_frames)} frames generated")
        print(f"   GIF created: {gif_path.name}")
        
        print(f"✅ Evolution Demo: {len(evolution_results)} individuals evaluated")
        best_fitness = evolution_results[0]["fitness"]
        print(f"   Best fitness achieved: {best_fitness:.3f}")
        
        print(f"\n📁 All outputs saved to: output/python_3d_demo/")
        
        print(f"\n🚀 The Python 3D renderer is ready for FractalGenesis integration!")
        print("   - High performance suitable for evolution populations")
        print("   - Complete parameter control for genetic algorithms")
        print("   - Native animation capabilities")
        print("   - No external CLI dependencies")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()