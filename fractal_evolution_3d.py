#!/usr/bin/env python3
"""
Working 3D Fractal Evolution Pipeline
=====================================

Automated 3D fractal generation with diversity verification.
Generates diverse Mandelbulb/Mandelbox fractals using genetic algorithms.

Usage:
    python3 fractal_evolution_3d.py --generations 5 --population 8
    python3 fractal_evolution_3d.py --renderer mandelbulber --generations 3
    python3 fractal_evolution_3d.py --verify-diversity
"""

import os
import sys
import argparse
import random
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
from dataclasses import dataclass

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from renderers.unified import create_renderer, RendererType, get_available_renderers
from shared.genome import FractalGenome, RendererType as GenomeRendererType
from FractalExplorer.genetic_algorithm.evolution_engine import EvolutionEngine, EvolutionConfig


@dataclass
class EvolutionResult:
    """Result from one generation of evolution"""
    generation: int
    genomes: List[FractalGenome]
    image_paths: List[str]
    diversity_score: float
    avg_fitness: float
    render_time: float


class FractalEvolution3D:
    """Complete 3D fractal evolution automation system"""
    
    def __init__(
        self,
        renderer_type: RendererType = RendererType.PYTHON_3D,
        output_dir: str = "output/evolution_3d",
        population_size: int = 8,
        mutation_rate: float = 0.25,
        mutation_strength: float = 0.15
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create renderer
        print(f"Initializing {renderer_type.value} renderer...")
        self.renderer = create_renderer(renderer_type)
        
        if not self.renderer or not self.renderer.is_available():
            raise RuntimeError(f"Renderer {renderer_type.value} not available")
        
        print(f"✓ Using {self.renderer.name}")
        
        # Setup evolution
        config = EvolutionConfig(
            population_size=population_size,
            elite_size=max(2, population_size // 4),
            tournament_size=min(4, population_size // 2),
            mutation_rate=mutation_rate,
            mutation_strength=mutation_strength,
            crossover_rate=0.8,
            diversity_weight=0.3
        )
        
        self.engine = EvolutionEngine(config, GenomeRendererType.MANDELBULBER)
        self.results: List[EvolutionResult] = []
        
    def initialize_population(self, seed_count: int = None):
        """Create initial diverse population"""
        count = seed_count or self.engine.config.population_size
        
        print(f"\nCreating initial population of {count} genomes...")
        genomes = []
        
        for i in range(count):
            # Generate random parameters using renderer
            params = self.renderer.generate_random_parameters()
            
            # Convert to genome
            genome = FractalGenome(GenomeRendererType.MANDELBULBER)
            genome.fractal.formula_type = params.get('fractal_type', 'mandelbulb')
            genome.fractal.power = params.get('power', 8.0)
            genome.fractal.iterations = int(params.get('iterations', 100))
            
            # Camera from renderer params
            cam_pos = params.get('camera_pos', (0.0, 0.0, -3.0))
            genome.camera.position = cam_pos
            genome.camera.target = params.get('target', (0.0, 0.0, 0.0))
            genome.camera.fov = params.get('fov', 45.0)
            
            # Color palette
            palette = params.get('color_palette', 'warm')
            genome.color.base_color = self._palette_to_color(palette)
            
            # Initial fitness
            genome.fitness = 0.5
            
            genomes.append(genome)
            print(f"  Genome {i+1}: {genome.fractal.formula_type}, power={genome.fractal.power:.1f}")
        
        self.engine.initialize_population(genomes)
        print(f"✓ Population initialized\n")
        
    def _palette_to_color(self, palette: str) -> Tuple[float, float, float]:
        """Convert palette name to RGB color"""
        colors = {
            'warm': (0.9, 0.5, 0.2),
            'cool': (0.2, 0.5, 0.9),
            'fire': (1.0, 0.3, 0.1),
            'ice': (0.2, 0.7, 1.0),
            'rainbow': (0.8, 0.6, 0.4),
            'monochrome': (0.5, 0.5, 0.5)
        }
        return colors.get(palette, (0.8, 0.6, 0.4))
    
    def _genome_to_params(self, genome: FractalGenome) -> Dict[str, Any]:
        """Convert genome to renderer parameters"""
        return {
            'fractal_type': genome.fractal.formula_type,
            'power': genome.fractal.power,
            'iterations': genome.fractal.iterations,
            'camera_pos': genome.camera.position,
            'target': genome.camera.target,
            'fov': genome.camera.fov,
            'color_palette': self._color_to_palette(genome.color.base_color),
            'width': 512,
            'height': 512
        }
    
    def _color_to_palette(self, color: Tuple[float, float, float]) -> str:
        """Guess palette from color"""
        r, g, b = color
        if r > 0.7 and g < 0.5 and b < 0.5:
            return 'fire'
        elif b > 0.7:
            return 'ice'
        elif r > 0.7 and g > 0.5:
            return 'warm'
        elif b > 0.6 and r < 0.4:
            return 'cool'
        return 'warm'
    
    def render_generation(
        self,
        generation: int,
        num_candidates: int = 4
    ) -> EvolutionResult:
        """Render one generation and return results"""
        
        print(f"Generation {generation + 1}")
        print("-" * 50)
        
        gen_start = time.time()
        
        # Get candidates for rendering
        candidates = self.engine.get_candidates_for_user_selection(num_candidates)
        
        # Render each candidate
        image_paths = []
        genomes_rendered = []
        
        for i, genome in enumerate(candidates):
            print(f"  Rendering candidate {i+1}/{len(candidates)}...", end=" ")
            
            params = self._genome_to_params(genome)
            output_path = self.output_dir / f"gen{generation:03d}_cand{i}.png"
            
            try:
                success = self.renderer.render_fractal(
                    params,
                    str(output_path),
                    width=400,
                    height=400
                )
                
                if success and output_path.exists():
                    image_paths.append(str(output_path))
                    genomes_rendered.append(genome)
                    print(f"✓ {output_path.name}")
                else:
                    print(f"✗ Failed")
                    
            except Exception as e:
                print(f"✗ Error: {e}")
        
        render_time = time.time() - gen_start
        
        # Calculate diversity
        diversity = self._calculate_diversity(genomes_rendered)
        
        # Simulated selection (in real use, user would select)
        # For automation, we use diversity-based fitness
        if genomes_rendered:
            # Boost fitness of most diverse genome
            selected_idx = self._select_diverse(genomes_rendered, diversity)
            self.engine.record_user_selection(candidates[:len(genomes_rendered)], selected_idx)
            
            # Evolve to next generation
            stats = self.engine.evolve_generation()
            
            result = EvolutionResult(
                generation=generation,
                genomes=genomes_rendered,
                image_paths=image_paths,
                diversity_score=diversity,
                avg_fitness=stats.get('avg_fitness', 0.0),
                render_time=render_time
            )
            
            self.results.append(result)
            
            print(f"  Diversity: {diversity:.3f}")
            print(f"  Render time: {render_time:.1f}s")
            print(f"  Selected: candidate {selected_idx + 1}")
            print()
            
            return result
        
        return None
    
    def _calculate_diversity(self, genomes: List[FractalGenome]) -> float:
        """Calculate diversity score for a set of genomes"""
        if len(genomes) < 2:
            return 0.0
        
        total_diversity = 0.0
        count = 0
        
        for i in range(len(genomes)):
            for j in range(i + 1, len(genomes)):
                diversity = genomes[i].calculate_diversity(genomes[j])
                total_diversity += diversity
                count += 1
        
        return total_diversity / count if count > 0 else 0.0
    
    def _select_diverse(
        self,
        genomes: List[FractalGenome],
        diversity_matrix: float
    ) -> int:
        """Select the most diverse genome (or random with diversity weighting)"""
        if len(genomes) < 2:
            return 0
        
        # Calculate average diversity for each genome
        avg_diversities = []
        for i in range(len(genomes)):
            div_sum = 0.0
            for j in range(len(genomes)):
                if i != j:
                    div_sum += genomes[i].calculate_diversity(genomes[j])
            avg_diversities.append(div_sum / (len(genomes) - 1))
        
        # Select genome with highest average diversity (with some randomness)
        weights = np.array(avg_diversities) + 0.1  # Ensure non-zero
        weights = weights / weights.sum()
        
        return np.random.choice(len(genomes), p=weights)
    
    def run_evolution(
        self,
        generations: int = 5,
        render_every: int = 1
    ) -> List[EvolutionResult]:
        """Run complete evolution process"""
        
        print(f"\n{'='*60}")
        print(f"Starting 3D Fractal Evolution")
        print(f"Generations: {generations}")
        print(f"Renderer: {self.renderer.name}")
        print(f"Output: {self.output_dir}")
        print(f"{'='*60}\n")
        
        # Initialize
        self.initialize_population()
        
        # Run generations
        for gen in range(generations):
            result = self.render_generation(gen)
            
            if result is None:
                print(f"  ⚠️  Generation {gen + 1} failed to render")
                continue
        
        # Final summary
        self._print_summary()
        
        return self.results
    
    def _print_summary(self):
        """Print evolution summary"""
        
        print(f"\n{'='*60}")
        print("Evolution Complete!")
        print(f"{'='*60}")
        
        if not self.results:
            print("No results generated")
            return
        
        # Statistics
        avg_diversity = np.mean([r.diversity_score for r in self.results])
        total_time = sum([r.render_time for r in self.results])
        total_images = sum([len(r.image_paths) for r in self.results])
        
        print(f"\nResults:")
        print(f"  Generations: {len(self.results)}")
        print(f"  Total images: {total_images}")
        print(f"  Average diversity: {avg_diversity:.3f}")
        print(f"  Total render time: {total_time:.1f}s")
        print(f"  Output directory: {self.output_dir}")
        
        # List best images
        print(f"\nGenerated fractals:")
        for result in self.results:
            print(f"  Gen {result.generation + 1}: {len(result.image_paths)} images")
        
        # Save metadata
        metadata = {
            'renderer': self.renderer.name,
            'generations': len(self.results),
            'total_images': total_images,
            'avg_diversity': float(avg_diversity),
            'total_time': total_time,
            'generation_results': [
                {
                    'generation': r.generation,
                    'diversity': float(r.diversity_score),
                    'images': r.image_paths,
                    'time': r.render_time
                }
                for r in self.results
            ]
        }
        
        meta_path = self.output_dir / "evolution_metadata.json"
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nMetadata saved: {meta_path}")


def main():
    parser = argparse.ArgumentParser(description='3D Fractal Evolution Pipeline')
    parser.add_argument('--generations', type=int, default=5, help='Number of generations')
    parser.add_argument('--population', type=int, default=8, help='Population size')
    parser.add_argument('--renderer', choices=['python3d', 'mandelbulber'], 
                       default='python3d', help='Renderer to use')
    parser.add_argument('--output', default='output/evolution_3d', help='Output directory')
    parser.add_argument('--mutation-rate', type=float, default=0.25, help='Mutation rate')
    
    args = parser.parse_args()
    
    # Select renderer
    renderer_type = RendererType.PYTHON_3D if args.renderer == 'python3d' else RendererType.MANDELBULBER
    
    try:
        # Create evolution system
        evolution = FractalEvolution3D(
            renderer_type=renderer_type,
            output_dir=args.output,
            population_size=args.population,
            mutation_rate=args.mutation_rate
        )
        
        # Run evolution
        results = evolution.run_evolution(generations=args.generations)
        
        print(f"\n✓ Evolution complete! Check {args.output}/ for results")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
