#!/usr/bin/env python3
"""
Basic Fractal Evolution Example

This example demonstrates the core FractalGenesis system:
1. Initialize a population of fractal genomes
2. Render them using Mandelbulber
3. Let user select preferred fractals
4. Evolve population based on preferences
5. Repeat for several generations

Run this to get started with fractal evolution!
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from shared.genome import FractalGenome, RendererType
from FractalExplorer.genetic_algorithm.evolution_engine import EvolutionEngine, EvolutionConfig
from FractalExplorer.ui.selection_interface import SelectionInterface
from renderers.mandelbulber.renderer import MandelbulberRenderer
from renderers.mandelbulber.templates import ParameterTemplates


def create_seed_genomes(count=6) -> list:
    """Create initial population from templates and random genomes."""
    genomes = []
    
    # Get template parameters and convert to genomes
    templates = ParameterTemplates.get_all_templates()
    
    for i, (name, template_params) in enumerate(templates.items()):
        if len(genomes) >= count:
            break
            
        genome = FractalGenome(RendererType.MANDELBULBER)
        
        # Convert template parameters to genome format
        # Camera
        genome.camera.position = (template_params.camera.camera_x,
                                 template_params.camera.camera_y,
                                 template_params.camera.camera_z)
        genome.camera.target = (template_params.camera.target_x,
                               template_params.camera.target_y,
                               template_params.camera.target_z)
        genome.camera.fov = template_params.camera.fov
        
        # Fractal
        genome.fractal.formula_type = template_params.fractal.formula_name
        genome.fractal.power = template_params.fractal.power
        genome.fractal.bailout = template_params.fractal.bailout
        genome.fractal.iterations = template_params.fractal.iterations
        
        # Color
        genome.color.base_color = (template_params.material.surface_color_r,
                                  template_params.material.surface_color_g,
                                  template_params.material.surface_color_b)
        
        # Lighting
        genome.lighting.main_light_direction = (template_params.lighting.main_light_alpha,
                                              template_params.lighting.main_light_beta)
        genome.lighting.main_light_intensity = template_params.lighting.main_light_intensity
        
        genomes.append(genome)
        print(f"Created genome from template: {name}")
    
    # Fill remaining spots with random genomes
    while len(genomes) < count:
        genome = FractalGenome(RendererType.MANDELBULBER)
        genome.randomize()
        genomes.append(genome)
        print(f"Created random genome {len(genomes)}")
    
    return genomes


def main():
    """Run the basic fractal evolution example."""
    print("🌀 FractalGenesis - Basic Evolution Example")
    print("=" * 50)
    
    # Configuration
    config = EvolutionConfig(
        population_size=12,
        elite_size=3,
        mutation_rate=0.2,
        crossover_rate=0.8,
        max_generations=10
    )
    
    print(f"Population size: {config.population_size}")
    print(f"Generations to run: {config.max_generations}")
    print()
    
    # Create directories
    output_dir = project_root / "renders"
    output_dir.mkdir(exist_ok=True)
    
    # Initialize components
    print("Initializing components...")
    
    # Evolution engine
    engine = EvolutionEngine(config, RendererType.MANDELBULBER)
    
    # Renderer (check if Mandelbulber is available)
    renderer = MandelbulberRenderer(output_dir=str(output_dir))
    if not renderer.is_available:
        print("⚠️  Mandelbulber not found! Please install it:")
        print("   sudo dnf install mandelbulber2  # Fedora")
        print("   sudo apt install mandelbulber2  # Ubuntu")
        print()
        print("For now, we'll simulate the evolution without rendering.")
        simulate_mode = True
    else:
        print("✅ Mandelbulber found and ready!")
        simulate_mode = False
    
    # User interface
    ui = SelectionInterface()
    
    # Create initial population
    print("Creating initial population from templates...")
    seed_genomes = create_seed_genomes(config.population_size)
    engine.initialize_population(seed_genomes)
    
    print(f"✅ Population initialized with {len(seed_genomes)} genomes")
    print()
    
    # Evolution loop
    try:
        for generation in range(config.max_generations):
            print(f"🧬 Generation {generation + 1}/{config.max_generations}")
            print("-" * 30)
            
            # Get candidates for user selection
            candidates = engine.get_candidates_for_user_selection(4)
            print(f"Selected {len(candidates)} candidates for evaluation")
            
            if simulate_mode:
                # Simulate user selection without rendering
                print("Simulating user selection (no rendering)...")
                selected_index = simulate_user_choice(candidates)
                print(f"Simulated selection: option {selected_index + 1}")
                
            else:
                # Render candidates
                print("Rendering candidate fractals...")
                image_paths = []
                
                for i, candidate in enumerate(candidates):
                    print(f"  Rendering candidate {i + 1}/4...", end=" ")
                    
                    try:
                        # Convert to Mandelbulber parameters
                        mandel_params = candidate.to_mandelbulber_parameters()
                        mandel_params.render.image_width = 400
                        mandel_params.render.image_height = 400
                        
                        # Render with thumbnail
                        render_path = renderer.render_single(
                            mandel_params,
                            output_filename=f"gen{generation+1}_candidate{i+1}.png",
                            thumbnail_size=(300, 300)
                        )
                        
                        image_paths.append(render_path)
                        print("✅")
                        
                    except Exception as e:
                        print(f"❌ Error: {e}")
                        image_paths.append(None)
                
                # Show to user for selection
                print("Showing candidates to user...")
                ui.load_fractal_images(image_paths)
                selected_index = ui.show_and_get_selection()
                
                if selected_index >= 0:
                    print(f"User selected: option {selected_index + 1}")
                else:
                    print("User skipped this generation")
            
            # Record selection and evolve
            engine.record_user_selection(candidates, selected_index)
            gen_stats = engine.evolve_generation()
            
            # Print generation statistics
            print(f"Population diversity: {gen_stats['diversity']:.3f}")
            print(f"Average fitness: {gen_stats['avg_fitness']:.3f}")
            print(f"Max fitness: {gen_stats['max_fitness']:.3f}")
            print()
            
            # Check for convergence
            if gen_stats['diversity'] < config.convergence_threshold:
                print("🎯 Population converged! Evolution complete.")
                break
    
    except KeyboardInterrupt:
        print("\n⏹️  Evolution interrupted by user")
    
    except Exception as e:
        print(f"\n❌ Error during evolution: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Show final results
        print("\n" + "=" * 50)
        print("🏆 Evolution Complete!")
        print("=" * 50)
        
        # Get best genomes
        best_genomes = engine.get_best_genomes(3)
        print(f"Top {len(best_genomes)} genomes by fitness:")
        
        for i, genome in enumerate(best_genomes, 1):
            print(f"{i}. Fitness: {genome.fitness:.3f}")
            print(f"   Formula: {genome.fractal.formula_type}")
            print(f"   Power: {genome.fractal.power:.2f}")
            print(f"   Color: RGB({genome.color.base_color[0]:.2f}, {genome.color.base_color[1]:.2f}, {genome.color.base_color[2]:.2f})")
            print()
        
        # Evolution summary
        summary = engine.get_evolution_summary()
        print("Evolution Summary:")
        print(f"- Total generations: {summary['current_generation']}")
        print(f"- User selections: {summary['total_user_selections']}")
        print(f"- Final diversity: {summary['population_diversity']:.3f}")
        
        if not simulate_mode:
            # Render statistics
            render_stats = renderer.get_render_statistics()
            print(f"- Images rendered: {render_stats['total_renders']}")
            print(f"- Disk usage: {render_stats['disk_usage_mb']:.1f} MB")
            print(f"- Output directory: {output_dir}")
        
        print("\nThank you for using FractalGenesis! 🌀")


def simulate_user_choice(candidates) -> int:
    """Simulate user choice for demonstration purposes."""
    import random
    
    # Simulate user preference based on some genome characteristics
    scores = []
    for candidate in candidates:
        # Simple scoring based on power and color
        score = 0
        
        # Prefer certain power ranges
        if 6 <= candidate.fractal.power <= 10:
            score += 2
        
        # Prefer more colorful fractals
        color_intensity = sum(candidate.color.base_color) / 3
        score += color_intensity
        
        # Add some randomness
        score += random.uniform(-0.5, 0.5)
        
        scores.append(score)
    
    # Select the highest scoring candidate
    best_index = scores.index(max(scores))
    
    # Sometimes skip (10% chance)
    if random.random() < 0.1:
        return -1
    
    return best_index


if __name__ == "__main__":
    main()
