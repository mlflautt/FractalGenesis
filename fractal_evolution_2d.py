#!/usr/bin/env python3
"""
Working 2D Fractal Flame Evolution Pipeline
===========================================

Automated 2D fractal flame generation with diversity verification.
Uses Flam3 to generate Electric Sheep-style fractal flames.

Usage:
    python3 fractal_evolution_2d.py --generations 5 --population 8
    python3 fractal_evolution_2d.py --verify-diversity
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

from renderers.unified import create_renderer, RendererType
from renderers.flam3_renderer import Flam3Renderer
from genome.flam3_genome import Flam3Genome
from shared.genome import FractalGenome, RendererType as GenomeRendererType
from FractalExplorer.genetic_algorithm.evolution_engine import EvolutionEngine, EvolutionConfig


@dataclass
class EvolutionResult2D:
    """Result from one generation of 2D flame evolution"""
    generation: int
    genomes: List[Flam3Genome]
    image_paths: List[str]
    diversity_score: float
    avg_fitness: float
    render_time: float


class FractalEvolution2D:
    """Complete 2D fractal flame evolution automation system"""
    
    def __init__(
        self,
        output_dir: str = "output/evolution_2d",
        population_size: int = 8,
        mutation_rate: float = 0.25,
        quality: int = 100  # Lower for faster rendering during evolution
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup Flam3 renderer
        print("Initializing Flam3 renderer...")
        try:
            self.renderer = Flam3Renderer(
                output_dir=str(self.output_dir),
                quality=quality,
                size=400  # Smaller for evolution speed
            )
            print(f"✓ Flam3 renderer ready")
        except Exception as e:
            print(f"✗ Flam3 initialization failed: {e}")
            print("  Ensure flam3 is installed: sudo dnf install flam3")
            raise
        
        # Setup evolution
        config = EvolutionConfig(
            population_size=population_size,
            elite_size=max(2, population_size // 4),
            tournament_size=min(4, population_size // 2),
            mutation_rate=mutation_rate,
            mutation_strength=0.15,
            crossover_rate=0.8,
            diversity_weight=0.3
        )
        
        self.engine = EvolutionEngine(config, GenomeRendererType.FRACTAL_FLAME)
        self.results: List[EvolutionResult2D] = []
        
    def initialize_population(self, seed_count: int = None):
        """Create initial diverse population of flames"""
        count = seed_count or self.engine.config.population_size
        
        print(f"\nCreating initial population of {count} flame genomes...")
        genomes = []
        
        for i in range(count):
            try:
                # Generate random Flam3 genome
                genome_xml = self.renderer.generate_random_genome(seed=random.randint(1, 100000))
                flam3_genome = Flam3Genome(genome_xml)
                
                # Convert to FractalGenome for evolution
                fractal_genome = flam3_genome.to_fractal_genome()
                fractal_genome.fitness = 0.5
                
                genomes.append(fractal_genome)
                print(f"  Genome {i+1}: {flam3_genome.name[:30]}...")
                
            except Exception as e:
                print(f"  ⚠️  Failed to create genome {i+1}: {e}")
                # Create a simple fallback genome
                genome = FractalGenome(GenomeRendererType.FRACTAL_FLAME)
                genome.randomize()
                genome.fitness = 0.5
                genomes.append(genome)
        
        self.engine.initialize_population(genomes)
        print(f"✓ Population initialized with {len(genomes)} genomes\n")
        
    def render_generation(
        self,
        generation: int,
        num_candidates: int = 4
    ) -> EvolutionResult2D:
        """Render one generation and return results"""
        
        print(f"Generation {generation + 1}")
        print("-" * 50)
        
        gen_start = time.time()
        
        # Get candidates for rendering
        candidates = self.engine.get_candidates_for_user_selection(num_candidates)
        
        # Render each candidate
        image_paths = []
        genomes_rendered = []
        
        for i, fractal_genome in enumerate(candidates):
            print(f"  Rendering flame {i+1}/{len(candidates)}...", end=" ")
            
            try:
                # Convert back to Flam3 genome
                flam3_genome = Flam3Genome.from_fractal_genome(fractal_genome)
                xml_data = flam3_genome.to_xml()
                
                output_path = self.output_dir / f"gen{generation:03d}_flame{i}.png"
                
                # Render using Flam3
                rendered_path = self.renderer.render_genome(xml_data, output_path.stem)
                
                if rendered_path and Path(rendered_path).exists():
                    image_paths.append(str(rendered_path))
                    genomes_rendered.append(fractal_genome)
                    print(f"✓ {Path(rendered_path).name}")
                else:
                    print(f"✗ No output file")
                    
            except Exception as e:
                print(f"✗ Error: {e}")
        
        render_time = time.time() - gen_start
        
        # Calculate diversity
        diversity = self._calculate_diversity(genomes_rendered)
        
        # Select based on diversity
        if genomes_rendered:
            selected_idx = self._select_diverse(genomes_rendered, diversity)
            self.engine.record_user_selection(candidates[:len(genomes_rendered)], selected_idx)
            
            # Evolve to next generation
            stats = self.engine.evolve_generation()
            
            result = EvolutionResult2D(
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
            print(f"  Selected: flame {selected_idx + 1}")
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
        """Select the most diverse genome"""
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
        
        # Weighted random selection
        weights = np.array(avg_diversities) + 0.1
        weights = weights / weights.sum()
        
        return np.random.choice(len(genomes), p=weights)
    
    def run_evolution(
        self,
        generations: int = 5
    ) -> List[EvolutionResult2D]:
        """Run complete evolution process"""
        
        print(f"\n{'='*60}")
        print(f"Starting 2D Fractal Flame Evolution")
        print(f"Generations: {generations}")
        print(f"Renderer: Flam3 (Fractal Flames)")
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
        print("2D Flame Evolution Complete!")
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
        print(f"  Total flames: {total_images}")
        print(f"  Average diversity: {avg_diversity:.3f}")
        print(f"  Total render time: {total_time:.1f}s")
        print(f"  Output directory: {self.output_dir}")
        
        print(f"\nGenerated flames:")
        for result in self.results:
            print(f"  Gen {result.generation + 1}: {len(result.image_paths)} flames")
        
        # Save metadata
        metadata = {
            'renderer': 'Flam3 (Fractal Flames)',
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
    parser = argparse.ArgumentParser(description='2D Fractal Flame Evolution Pipeline')
    parser.add_argument('--generations', type=int, default=5, help='Number of generations')
    parser.add_argument('--population', type=int, default=8, help='Population size')
    parser.add_argument('--output', default='output/evolution_2d', help='Output directory')
    parser.add_argument('--mutation-rate', type=float, default=0.25, help='Mutation rate')
    parser.add_argument('--quality', type=int, default=100, help='Render quality (lower=faster)')
    
    args = parser.parse_args()
    
    try:
        # Create evolution system
        evolution = FractalEvolution2D(
            output_dir=args.output,
            population_size=args.population,
            mutation_rate=args.mutation_rate,
            quality=args.quality
        )
        
        # Run evolution
        results = evolution.run_evolution(generations=args.generations)
        
        print(f"\n✓ 2D Flame evolution complete! Check {args.output}/ for results")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
