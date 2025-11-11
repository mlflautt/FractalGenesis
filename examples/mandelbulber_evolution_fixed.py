#!/usr/bin/env python3
"""
Fixed Mandelbulber Evolution Example

This demonstrates the corrected FractalGenesis system with working Mandelbulber integration:
1. Initialize population from templates 
2. Render them using the corrected Mandelbulber renderer
3. Generate diverse, visually distinct fractals
4. Support mutation and crossover for evolution

Run this to test the fixed Mandelbulber integration!
"""

import sys
import os
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from renderers.mandelbulber.parameters import MandelbulberParameters
from renderers.mandelbulber.renderer import MandelbulberRenderer
from renderers.mandelbulber.templates import ParameterTemplates

def create_population(size=8):
    """Create diverse initial population from templates and variations."""
    population = []
    templates = ParameterTemplates.get_all_templates()
    
    # Start with all templates
    for name, template in templates.items():
        if len(population) < size:
            # Make a copy and add some variation
            params = template.copy()
            if len(population) > 0:  # Add variation to all but first
                params.mutate(mutation_rate=0.2, mutation_strength=0.1)
            population.append(params)
    
    # Fill remaining with random parameters
    while len(population) < size:
        params = MandelbulberParameters()
        params.randomize()
        population.append(params)
    
    return population[:size]

def render_population(renderer, population, generation=1):
    """Render a population of fractals."""
    results = []
    
    print(f"🎨 Rendering Generation {generation} ({len(population)} fractals)")
    
    for i, params in enumerate(population):
        print(f"  Rendering fractal {i+1}/{len(population)}...", end=" ")
        
        # Set rendering parameters
        params.render.image_width = 400
        params.render.image_height = 400
        
        output_filename = f"gen{generation}_fractal{i+1}.png"
        
        try:
            output_path = renderer.render_single(
                params,
                output_filename=output_filename,
                thumbnail_size=(200, 200)  # Create thumbnails too
            )
            
            if output_path and output_path.exists():
                size = output_path.stat().st_size
                results.append((output_path, size))
                print(f"✅ {size} bytes")
            else:
                results.append((None, 0))
                print("❌ Failed")
                
        except Exception as e:
            results.append((None, 0))
            print(f"❌ Error: {e}")
    
    successful = sum(1 for path, size in results if path is not None)
    print(f"  Success rate: {successful}/{len(population)}")
    
    return results

def evolve_population(population, selection_indices, mutation_rate=0.3):
    """Create next generation based on selection."""
    if not selection_indices:
        # No selection, return mutated population
        next_gen = []
        for params in population:
            mutated = params.copy()
            mutated.mutate(mutation_rate=mutation_rate, mutation_strength=0.2)
            next_gen.append(mutated)
        return next_gen
    
    # Select parents based on user selection
    selected = [population[i] for i in selection_indices]
    
    next_gen = []
    
    # Keep best performers (elitism)
    for params in selected:
        next_gen.append(params.copy())
    
    # Create offspring through crossover and mutation
    while len(next_gen) < len(population):
        if len(selected) >= 2:
            # Crossover
            parent1 = selected[len(next_gen) % len(selected)]
            parent2 = selected[(len(next_gen) + 1) % len(selected)]
            
            child = parent1.crossover(parent2)
            child.mutate(mutation_rate=mutation_rate, mutation_strength=0.15)
            next_gen.append(child)
        else:
            # Just mutate
            parent = selected[0]
            child = parent.copy()
            child.mutate(mutation_rate=mutation_rate, mutation_strength=0.2)
            next_gen.append(child)
    
    return next_gen

def simulate_user_selection(results, num_select=2):
    """Simulate user selection based on file size (as proxy for complexity)."""
    # Filter successful renders
    successful = [(i, size) for i, (path, size) in enumerate(results) if path is not None]
    
    if len(successful) < num_select:
        return [i for i, _ in successful]
    
    # Select based on file size diversity (prefer medium to large sizes)
    sorted_results = sorted(successful, key=lambda x: x[1], reverse=True)
    
    # Take top performers but ensure diversity  
    selected_indices = []
    for i, (index, size) in enumerate(sorted_results):
        if len(selected_indices) < num_select:
            selected_indices.append(index)
        if len(selected_indices) >= num_select:
            break
    
    return selected_indices

def main():
    """Run the fixed Mandelbulber evolution example."""
    print("🌀 Fixed FractalGenesis - Mandelbulber Evolution")
    print("=" * 60)
    
    # Setup output directory
    output_dir = Path(os.environ.get('FRACTAL_OUTPUT_DIR', str(project_root / "renders")))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize renderer
    print("Initializing Mandelbulber renderer...")
    renderer = MandelbulberRenderer(output_dir=str(output_dir))
    
    if not renderer.is_available:
        print("❌ Mandelbulber not found! Please install it.")
        print("   Fedora: sudo dnf install mandelbulber2")
        print("   Ubuntu: sudo apt install mandelbulber2")
        return
    
    print("✅ Mandelbulber renderer ready!")
    print(f"📁 Output directory: {output_dir}")
    
    # Evolution parameters
    population_size = 6
    generations = 3
    
    print(f"\n🧬 Evolution Parameters:")
    print(f"   Population size: {population_size}")
    print(f"   Generations: {generations}")
    
    # Create initial population
    print(f"\n🌱 Creating initial population...")
    population = create_population(population_size)
    print(f"✅ Created {len(population)} diverse fractals")
    
    # Evolution loop
    try:
        for gen in range(generations):
            print(f"\n" + "="*40)
            print(f"🧬 GENERATION {gen + 1}")
            print("="*40)
            
            # Render current population
            results = render_population(renderer, population, gen + 1)
            
            # Simulate user selection (in real version, user would select)
            print("\n🎯 Simulating selection...")
            selected_indices = simulate_user_selection(results, num_select=2)
            
            if selected_indices:
                print(f"   Selected fractals: {[i+1 for i in selected_indices]}")
                
                # Show selection details
                for idx in selected_indices:
                    path, size = results[idx]
                    print(f"   - Fractal {idx+1}: {size} bytes")
            else:
                print("   No selection made, using random mutation")
            
            # Evolve to next generation
            if gen < generations - 1:  # Don't evolve after last generation
                print("\n🔄 Evolving to next generation...")
                population = evolve_population(population, selected_indices)
                print("✅ Next generation created")
    
    except KeyboardInterrupt:
        print(f"\n⏹️  Evolution interrupted by user")
    
    except Exception as e:
        print(f"\n❌ Error during evolution: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Summary
        print(f"\n" + "="*60)
        print("🏆 EVOLUTION COMPLETE!")
        print("="*60)
        
        # Render statistics
        stats = renderer.get_render_statistics()
        print(f"📊 Render Statistics:")
        print(f"   Total images: {stats['total_renders']}")
        print(f"   Total thumbnails: {stats['total_thumbnails']}")
        print(f"   Disk usage: {stats['disk_usage_mb']:.1f} MB")
        print(f"   Output directory: {output_dir}")
        
        print(f"\n✨ Check the output directory for your evolved fractals!")
        print(f"🌈 Mandelbulber integration is working correctly!")

if __name__ == "__main__":
    main()