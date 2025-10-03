#!/usr/bin/env python3
"""
Evolve Fractal Animations Example

This script demonstrates how to use the fractal animation evolution system
to create and evolve beautiful animated fractals using evolutionary algorithms.

Usage:
    python3 examples/evolve_fractal_animations.py [OPTIONS]
    
Options:
    --mode          Evolution mode: 'quick', 'standard', 'quality' (default: standard)
    --generations   Number of generations to evolve (default: 20) 
    --population    Population size (default: 12)
    --output-dir    Output directory for animations (default: ./evolved_animations)
    --preview-only  Only generate previews, no full renders
    --template      Use specific template as seed: orbital, zoom, color_morph, etc.
    
Examples:
    # Quick evolution run for testing
    python3 examples/evolve_fractal_animations.py --mode quick --generations 5
    
    # High quality evolution with specific template
    python3 examples/evolve_fractal_animations.py --mode quality --template power_evolution --generations 15
    
    # Preview-only evolution for fast experimentation
    python3 examples/evolve_fractal_animations.py --preview-only --generations 10
"""

import sys
import os
import argparse
import logging
import time
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from FractalAnimator.animation_parameters import AnimationParameters
from FractalAnimator.animation_renderer import AnimationRenderer
from FractalAnimator.animation_evolution import AnimationEvolutionEngine, AnimationIndividual
from FractalAnimator.animation_templates import AnimationTemplates
from renderers.mandelbulber.renderer import MandelbulberRenderer


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_evolution_environment(mode: str, output_dir: Path, preview_only: bool):
    """Set up the evolution environment based on mode"""
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure based on mode
    if mode == 'quick':
        logger.info("Setting up QUICK evolution mode (fast, low quality)")
        mandelbulber_renderer = MandelbulberRenderer(
            output_dir=output_dir / "frames"
        )
        animation_renderer = AnimationRenderer(
            mandelbulber_renderer=mandelbulber_renderer,
            output_dir=output_dir,
            max_workers=2
        )
        
        evolution_config = {
            'population_size': 8,
            'elite_size': 2,
            'mutation_rate': 0.2,
            'crossover_rate': 0.8,
            'max_generations': 10
        }
        
    elif mode == 'standard':
        logger.info("Setting up STANDARD evolution mode (balanced)")
        mandelbulber_renderer = MandelbulberRenderer(
            output_dir=output_dir / "frames"
        )
        animation_renderer = AnimationRenderer(
            mandelbulber_renderer=mandelbulber_renderer,
            output_dir=output_dir,
            max_workers=4
        )
        
        evolution_config = {
            'population_size': 12,
            'elite_size': 3,
            'mutation_rate': 0.15,
            'crossover_rate': 0.8,
            'max_generations': 20
        }
        
    elif mode == 'quality':
        logger.info("Setting up QUALITY evolution mode (slow, high quality)")
        mandelbulber_renderer = MandelbulberRenderer(
            output_dir=output_dir / "frames"
        )
        animation_renderer = AnimationRenderer(
            mandelbulber_renderer=mandelbulber_renderer,
            output_dir=output_dir,
            max_workers=6
        )
        
        evolution_config = {
            'population_size': 16,
            'elite_size': 4,
            'mutation_rate': 0.12,
            'crossover_rate': 0.85,
            'max_generations': 30
        }
        
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # Store preview mode setting for later use
    if preview_only:
        logger.info("Preview-only mode enabled - using fast preview renders")
        evolution_config['preview_mode'] = True
    else:
        evolution_config['preview_mode'] = False
    
    return animation_renderer, evolution_config


def create_seed_population(template_name: str = None, population_size: int = 12):
    """Create initial population for evolution"""
    
    if template_name:
        logger.info(f"Creating seed population based on template: {template_name}")
        
        # Get the specific template
        all_templates = AnimationTemplates.get_all_templates()
        if template_name not in all_templates:
            logger.error(f"Template '{template_name}' not found. Available: {list(all_templates.keys())}")
            return None
        
        base_template = all_templates[template_name]
        
        # Create variations of the template
        seed_population = []
        seed_population.append(base_template)  # Include original
        
        for i in range(population_size - 1):
            variation_strength = 0.2 + (i / population_size) * 0.4  # Increasing variation
            variant = AnimationTemplates.create_random_variation(
                base_template,
                variation_strength=variation_strength
            )
            seed_population.append(variant)
            
    else:
        logger.info("Creating diverse seed population from all templates")
        seed_population = AnimationTemplates.create_seed_population(population_size)
    
    return seed_population


def progress_callback(generation: int, population: list):
    """Callback for tracking evolution progress"""
    fitness_scores = [ind.fitness_score for ind in population]
    best_fitness = max(fitness_scores) if fitness_scores else 0
    avg_fitness = sum(fitness_scores) / len(fitness_scores) if fitness_scores else 0
    
    logger.info(f"Generation {generation}: Best={best_fitness:.3f}, Avg={avg_fitness:.3f}")


def fitness_callback(individual: AnimationIndividual):
    """Callback for individual fitness evaluation"""
    logger.debug(f"Individual fitness: {individual.fitness.total_score:.3f}")


def render_final_animations(population: list, 
                          animation_renderer: AnimationRenderer,
                          output_dir: Path,
                          top_n: int = 3,
                          preview_only: bool = False):
    """Render the top animations from evolution"""
    
    # Sort population by fitness
    population.sort(key=lambda ind: ind.fitness_score, reverse=True)
    top_individuals = population[:top_n]
    
    logger.info(f"Rendering top {len(top_individuals)} animations...")
    
    results = []
    for i, individual in enumerate(top_individuals):
        logger.info(f"Rendering animation {i+1}/{len(top_individuals)} (fitness: {individual.fitness_score:.3f})")
        
        animation_name = f"evolved_animation_{i+1}_fitness_{individual.fitness_score:.3f}"
        
        if preview_only:
            result = animation_renderer.render_preview(
                individual.animation_params,
                max_frames=20
            )
        else:
            result = animation_renderer.render_animation(
                individual.animation_params,
                output_name=animation_name,
                video_format='mp4'
            )
        
        if result.success:
            logger.info(f"Successfully rendered: {result.animation_path}")
        else:
            logger.error(f"Failed to render animation {i+1}: {result.error_message}")
        
        results.append(result)
    
    return results


def save_evolution_summary(evolution_engine: AnimationEvolutionEngine,
                         results: list,
                         output_dir: Path,
                         start_time: float):
    """Save a summary of the evolution run"""
    
    end_time = time.time()
    total_time = end_time - start_time
    
    summary = {
        'evolution_summary': {
            'total_time_seconds': total_time,
            'total_time_formatted': f"{total_time/60:.1f} minutes",
            'successful_renders': len([r for r in results if r.success]),
            'failed_renders': len([r for r in results if not r.success]),
        },
        'best_individual_details': {},
        'animation_paths': [str(r.animation_path) for r in results if r.success]
    }
    
    # Get best individual details
    best_individual = evolution_engine.get_best_individual()
    if best_individual:
        summary['best_individual_details'] = {
            'fitness': best_individual.fitness.to_dict(),
            'animation_params': best_individual.animation_params.to_dict()
        }
    
    # Save evolution results
    results_path = output_dir / "evolution_results.json"
    evolution_engine.save_evolution_results(results_path)
    
    # Save summary
    summary_path = output_dir / "evolution_summary.json"
    import json
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Evolution summary saved to: {summary_path}")
    logger.info(f"Total evolution time: {total_time/60:.1f} minutes")
    
    if summary['evolution_summary']['successful_renders'] > 0:
        logger.info(f"Successfully created {summary['evolution_summary']['successful_renders']} animations")
        logger.info("Animation files:")
        for path in summary['animation_paths']:
            logger.info(f"  - {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evolve fractal animations using evolutionary algorithms",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--mode', 
        choices=['quick', 'standard', 'quality'],
        default='standard',
        help='Evolution mode (default: standard)'
    )
    
    parser.add_argument(
        '--generations',
        type=int,
        default=20,
        help='Number of generations to evolve (default: 20)'
    )
    
    parser.add_argument(
        '--population',
        type=int,
        default=12,
        help='Population size (default: 12)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('./evolved_animations'),
        help='Output directory (default: ./evolved_animations)'
    )
    
    parser.add_argument(
        '--preview-only',
        action='store_true',
        help='Only generate previews, no full renders'
    )
    
    parser.add_argument(
        '--template',
        choices=list(AnimationTemplates.get_all_templates().keys()),
        help='Use specific template as seed'
    )
    
    parser.add_argument(
        '--top-n',
        type=int,
        default=3,
        help='Number of top animations to render (default: 3)'
    )
    
    args = parser.parse_args()
    
    logger.info("=== Fractal Animation Evolution ===")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Generations: {args.generations}")
    logger.info(f"Population: {args.population}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Preview only: {args.preview_only}")
    if args.template:
        logger.info(f"Template: {args.template}")
    
    start_time = time.time()
    
    try:
        # Set up evolution environment
        animation_renderer, evolution_config = setup_evolution_environment(
            args.mode, args.output_dir, args.preview_only
        )
        
        # Override config with command line args
        evolution_config['population_size'] = args.population
        evolution_config['max_generations'] = args.generations
        
        # Extract preview_mode from config before creating engine
        preview_mode = evolution_config.pop('preview_mode', False)
        
        # Create evolution engine
        evolution_engine = AnimationEvolutionEngine(
            renderer=animation_renderer,
            **evolution_config
        )
        
        # Configure fitness evaluator for preview mode if needed
        if preview_mode:
            evolution_engine.fitness_evaluator.preview_mode = True
            evolution_engine.fitness_evaluator.max_preview_frames = 10
        
        # Set up progress callbacks
        evolution_engine.set_generation_callback(progress_callback)
        evolution_engine.set_fitness_callback(fitness_callback)
        
        # Create seed population
        seed_population = create_seed_population(args.template, args.population)
        if seed_population is None:
            return 1
        
        logger.info("Starting evolution...")
        
        # Run evolution
        final_population = evolution_engine.evolve_animations(
            seed_animations=seed_population,
            target_fitness=0.85,  # Stop if we reach high fitness
            early_stopping_patience=8
        )
        
        logger.info("Evolution complete! Rendering final animations...")
        
        # Render top animations
        render_results = render_final_animations(
            final_population,
            animation_renderer,
            args.output_dir,
            top_n=args.top_n,
            preview_only=args.preview_only
        )
        
        # Save evolution summary
        save_evolution_summary(
            evolution_engine,
            render_results,
            args.output_dir,
            start_time
        )
        
        logger.info("=== Evolution Complete ===")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Evolution interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Evolution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())