#!/usr/bin/env python3
"""
Flam3 Fractal Evolution Example

This script demonstrates the complete FractalGenesis pipeline using Flam3 
fractal flames instead of Mandelbulber. It generates fractals, presents them
to the user for selection, and evolves them over multiple generations.

Usage:
    python3 examples/flam3_evolution.py [--generations N] [--population N] [--render]
    
Options:
    --generations N  : Number of generations to evolve (default: 5)
    --population N   : Population size (default: 8)
    --render        : Actually render fractals (otherwise simulate)
    --quality N     : Render quality 1-100 (default: 50)
    --size N        : Image size in pixels (default: 512)
"""

import os
import sys
import argparse
import logging
import random
import time
from typing import List, Dict, Optional

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from renderers.flam3_renderer import Flam3Renderer
from genome.flam3_genome import Flam3Genome
from shared.genome import RendererType
from FractalExplorer.genetic_algorithm.evolution_engine import EvolutionEngine, EvolutionConfig
from FractalExplorer.ui.selection_interface import SelectionInterface


def setup_logging():
    """Configure logging for the example."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )


def simulate_user_selection(candidates: List[str], generation: int) -> int:
    """
    Simulate user selection for testing purposes.
    
    Args:
        candidates: List of rendered fractal image paths
        generation: Current generation number
        
    Returns:
        Index of selected fractal (0-3)
    """
    print(f"\n--- Generation {generation} Selection (Simulated) ---")
    for i, path in enumerate(candidates):
        print(f"  {i}: {os.path.basename(path)}")
    
    # Simulate preference evolution - gradually prefer higher quality
    weights = [0.4, 0.3, 0.2, 0.1]  # Prefer earlier options initially
    if generation > 2:
        weights = [0.1, 0.2, 0.3, 0.4]  # Later prefer evolved options
    
    selected = random.choices(range(len(candidates)), weights=weights[:len(candidates)])[0]
    print(f"  Simulated selection: {selected} ({os.path.basename(candidates[selected])})")
    return selected


def render_fractal_batch(renderer: Flam3Renderer, 
                        genomes: List[Flam3Genome], 
                        batch_name: str) -> List[str]:
    """
    Render a batch of Flam3 genomes.
    
    Args:
        renderer: Flam3 renderer instance
        genomes: List of Flam3Genome objects to render
        batch_name: Base name for the batch
        
    Returns:
        List of paths to rendered images
    """
    rendered_files = []
    
    for i, genome in enumerate(genomes):
        filename = f"{batch_name}_{i:02d}"
        try:
            xml_data = genome.to_xml()
            rendered_path = renderer.render_genome(xml_data, filename)
            rendered_files.append(rendered_path)
            print(f"  Rendered {filename}")
        except Exception as e:
            print(f"  Failed to render {filename}: {e}")
            # Create dummy path for simulation
            dummy_path = os.path.join(renderer.output_dir, f"{filename}.png")
            rendered_files.append(dummy_path)
    
    return rendered_files


def main():
    """Main evolution loop."""
    parser = argparse.ArgumentParser(description='Flam3 Fractal Evolution Example')
    parser.add_argument('--generations', type=int, default=5, 
                       help='Number of generations to evolve (default: 5)')
    parser.add_argument('--population', type=int, default=8,
                       help='Population size (default: 8)')
    parser.add_argument('--render', action='store_true',
                       help='Actually render fractals (otherwise simulate)')
    parser.add_argument('--quality', type=int, default=50,
                       help='Render quality 1-100 (default: 50)')
    parser.add_argument('--size', type=int, default=512,
                       help='Image size in pixels (default: 512)')
    parser.add_argument('--interactive', action='store_true',
                       help='Use interactive UI (if available)')
    
    args = parser.parse_args()
    
    setup_logging()
    
    print("=" * 60)
    print("FractalGenesis - Flam3 Fractal Evolution Example")
    print("=" * 60)
    print(f"Generations: {args.generations}")
    print(f"Population size: {args.population}")
    print(f"Rendering: {'Enabled' if args.render else 'Simulated'}")
    if args.render:
        print(f"Quality: {args.quality}, Size: {args.size}x{args.size}")
    print()
    
    # Initialize renderer
    output_dir = os.environ.get('FRACTAL_OUTPUT_DIR', 'output/flam3_evolution')
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    renderer = Flam3Renderer(
        output_dir=output_dir,
        quality=args.quality,
        size=args.size
    )
    print(f"✓ Flam3 renderer initialized (output: {output_dir})")
    
    # Initialize evolution engine
    config = EvolutionConfig(
        population_size=args.population,
        elite_size=max(1, args.population // 4),
        tournament_size=min(4, args.population // 2),
        mutation_rate=0.2,
        mutation_strength=0.15,
        crossover_rate=0.8,
        diversity_weight=0.3
    )
    
    engine = EvolutionEngine(config, RendererType.FRACTAL_FLAME)
    print(f"✓ Evolution engine configured")
    
    # Generate initial population
    print(f"\nGenerating initial population of {args.population} fractal flames...")
    initial_genomes = []
    
    for i in range(args.population):
        try:
            # Generate random Flam3 genome
            genome_xml = renderer.generate_random_genome(seed=random.randint(1, 10000))
            flam3_genome = Flam3Genome(genome_xml)
            fractal_genome = flam3_genome.to_fractal_genome()
            
            # Set initial fitness
            fractal_genome.fitness = 0.5
            initial_genomes.append(fractal_genome)
            
        except Exception as e:
            print(f"Failed to generate genome {i}: {e}")
            continue
    
    if not initial_genomes:
        print("ERROR: Failed to generate any initial genomes!")
        return 1
    
    engine.initialize_population(initial_genomes)
    print(f"✓ Initialized population with {len(initial_genomes)} genomes")
    
    # Evolution loop
    print(f"\nStarting evolution for {args.generations} generations...")
    print("-" * 60)
    
    selection_interface = None
    if args.interactive:
        try:
            selection_interface = SelectionInterface()
            print("✓ Interactive UI initialized")
        except Exception as e:
            print(f"Interactive UI not available: {e}")
            print("Falling back to simulation mode")
    
    evolution_stats = []
    
    for generation in range(args.generations):
        gen_start_time = time.time()
        print(f"\nGeneration {generation + 1}/{args.generations}")
        print("-" * 40)
        
        # Get current population
        current_population = engine.population.individuals
        
        # Convert to Flam3 genomes for rendering
        flam3_genomes = []
        for genome in current_population[:4]:  # Show top 4
            flam3_genome = Flam3Genome.from_fractal_genome(genome)
            flam3_genomes.append(flam3_genome)
        
        # Render candidates (or simulate)
        rendered_files = []
        if args.render:
            print("Rendering candidate fractals...")
            try:
                batch_name = f"gen{generation+1:02d}_candidates"
                rendered_files = render_fractal_batch(renderer, flam3_genomes, batch_name)
            except Exception as e:
                print(f"Rendering failed: {e}")
                print("Continuing in simulation mode")
                args.render = False
        
        if not rendered_files:
            # Simulate rendering for testing
            rendered_files = [f"simulated_fractal_{i}.png" for i in range(len(flam3_genomes))]
        
        # Get user selection
        selected_index = 0
        if selection_interface and args.interactive and rendered_files:
            try:
                # Use actual UI if available and files exist
                real_files = [f for f in rendered_files if os.path.exists(f)]
                if real_files:
                    selected_index = selection_interface.select_fractal(real_files)
                else:
                    selected_index = simulate_user_selection(rendered_files, generation + 1)
            except Exception as e:
                print(f"UI selection failed: {e}")
                selected_index = simulate_user_selection(rendered_files, generation + 1)
        else:
            # Simulated selection
            selected_index = simulate_user_selection(rendered_files, generation + 1)
        
        # Update fitness based on selection
        for i, genome in enumerate(current_population[:len(flam3_genomes)]):
            if i == selected_index:
                genome.fitness = 1.0  # Selected genome gets high fitness
            else:
                genome.fitness = max(0.1, (genome.fitness or 0.5) * 0.8)  # Others decay
        
        # Evolve to next generation
        try:
            stats = engine.evolve_generation()
            evolution_stats.append(stats)
            
            gen_time = time.time() - gen_start_time
            print(f"✓ Generation evolved in {gen_time:.1f}s")
            print(f"  Diversity: {stats['diversity']:.3f}")
            print(f"  Elite fitness: {max(g.fitness or 0 for g in engine.population.individuals):.3f}")
            
        except Exception as e:
            print(f"Evolution failed: {e}")
            break
    
    # Final statistics
    print("\n" + "="*60)
    print("EVOLUTION COMPLETE")
    print("="*60)
    
    if evolution_stats:
        final_stats = evolution_stats[-1]
        print(f"Final generation: {len(evolution_stats)}")
        print(f"Final diversity: {final_stats['diversity']:.3f}")
        print(f"Best fitness: {max(g.fitness or 0 for g in engine.population.individuals):.3f}")
        
        avg_time = sum(s['time_elapsed'] for s in evolution_stats) / len(evolution_stats)
        print(f"Average generation time: {avg_time:.2f}s")
        
        # Show diversity trend
        diversities = [s['diversity'] for s in evolution_stats]
        print(f"Diversity trend: {diversities[0]:.3f} -> {diversities[-1]:.3f}")
    
    # Render final elite if requested
    if args.render:
        print(f"\nRendering final elite fractals...")
        try:
            elite_genomes = sorted(engine.population.individuals, 
                                 key=lambda g: g.fitness or 0, reverse=True)[:4]
            
            elite_flam3 = [Flam3Genome.from_fractal_genome(g) for g in elite_genomes]
            final_renders = render_fractal_batch(renderer, elite_flam3, "final_elite")
            
            print(f"✓ Final elite rendered:")
            for i, path in enumerate(final_renders):
                if os.path.exists(path):
                    fitness = elite_genomes[i].fitness or 0
                    print(f"  {i+1}. {os.path.basename(path)} (fitness: {fitness:.3f})")
                    
        except Exception as e:
            print(f"Final rendering failed: {e}")
    
    print(f"\nAll output saved to: {os.path.abspath(output_dir)}")
    print("Evolution example completed successfully!")
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nEvolution interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        logging.exception("Full error details:")
        sys.exit(1)
