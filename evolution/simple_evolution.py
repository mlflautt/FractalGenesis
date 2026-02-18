#!/usr/bin/env python3
"""
Simple Fractal Evolution Engine
================================

Model: minimax-m2.5 (opencode)
Created: 2026-02-16
Version: 1.0

User-guided evolution with optional AI assistance.
- User selects favorites from population
- System evolves next generation based on selections
- Can train AI after enough selections made

Usage:
    from evolution.simple_evolution import FractalEvolver
    
    evolver = FractalEvolver()
    
    # Generate initial population
    population = evolver.generate_population(size=8)
    
    # Render all
    evolver.render_population(population, "output/evolution")
    
    # User makes selections (ranks 1-3)
    evolver.record_selections([selected_ids])
    
    # Evolve next generation
    next_gen = evolver.evolve_next_generation()
"""

import random
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import uuid
import json
import numpy as np
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class FractalGenome:
    """A complete genome representing fractal parameters."""
    genome_id: str
    generation: int
    
    # Core fractal params
    fractal_type: str
    power: float
    iterations: int
    bailout: float
    
    # Mandelbox/IFS params
    scale: float
    min_r: float
    ifs_scale: float
    ifs_folds: int
    lambda_val: float
    
    # Julia constant
    julia_c: Tuple[float, float, float]
    
    # Camera
    camera_pos: Tuple[float, float, float]
    target: Tuple[float, float, float]
    fov: float
    
    # Appearance
    color_palette: str
    color_intensity: float
    coloring_mode: str
    
    # Render metadata
    image_path: Optional[str] = None
    fitness: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "genome_id": self.genome_id,
            "generation": self.generation,
            "fractal_type": self.fractal_type,
            "power": self.power,
            "iterations": self.iterations,
            "bailout": self.bailout,
            "scale": self.scale,
            "min_r": self.min_r,
            "ifs_scale": self.ifs_scale,
            "ifs_folds": self.ifs_folds,
            "lambda_val": self.lambda_val,
            "julia_c": list(self.julia_c),
            "camera_pos": list(self.camera_pos),
            "target": list(self.target),
            "fov": self.fov,
            "color_palette": self.color_palette,
            "color_intensity": self.color_intensity,
            "coloring_mode": self.coloring_mode,
            "image_path": self.image_path,
            "fitness": self.fitness
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "FractalGenome":
        data["julia_c"] = tuple(data.get("julia_c", [0,0,0]))
        data["camera_pos"] = tuple(data.get("camera_pos", [0,0,-3]))
        data["target"] = tuple(data.get("target", [0,0,0]))
        return cls(**data)


class FractalEvolver:
    """Main evolution engine for fractal parameters."""
    
    FRACTAL_TYPES = ["mandelbulb", "julia", "mandelbox", "burning_ship", "kifs", "tricorn"]
    COLOR_PALETTES = ["warm", "cool", "rainbow", "fire", "ice", "monochrome"]
    COLORING_MODES = ["orbit_trap", "distance", "normal", "iteration"]
    
    def __init__(self, renderer=None):
        if renderer:
            self.renderer = renderer
        else:
            from renderers.python_3d import FractalRenderer
            self.renderer = FractalRenderer()
        
        self.current_population: List[FractalGenome] = []
        self.generation = 0
        self.selection_history: List[Dict] = []
        
    def create_random_genome(self, generation: int = 0) -> FractalGenome:
        """Create a random fractal genome."""
        ft = random.choice(self.FRACTAL_TYPES)
        
        # Different param ranges for different types
        if ft == "mandelbulb":
            power = random.uniform(2, 16)
            scale = -1.5
            min_r = 0.5
        elif ft == "mandelbox":
            power = 8.0
            scale = random.choice([2.0, 2.5, 3.0, -1.5, -2.0])
            min_r = random.uniform(0.2, 0.8)
        elif ft == "julia":
            power = random.uniform(4, 12)
            scale = -1.5
            min_r = 0.5
        else:
            power = random.uniform(2, 12)
            scale = random.uniform(-2, 3)
            min_r = random.uniform(0.3, 0.7)
        
        return FractalGenome(
            genome_id=str(uuid.uuid4())[:8],
            generation=generation,
            fractal_type=ft,
            power=power,
            iterations=random.randint(50, 150),
            bailout=2.0,
            scale=scale,
            min_r=min_r,
            ifs_scale=random.uniform(2.0, 3.0),
            ifs_folds=random.randint(2, 5),
            lambda_val=random.uniform(0.5, 2.0),
            julia_c=(random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5), 0.0),
            camera_pos=(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-4, -2)),
            target=(0.0, 0.0, 0.0),
            fov=random.uniform(30, 60),
            color_palette=random.choice(self.COLOR_PALETTES),
            color_intensity=random.uniform(0.5, 2.0),
            coloring_mode=random.choice(self.COLORING_MODES)
        )
    
    def generate_population(self, size: int = 8) -> List[FractalGenome]:
        """Generate initial random population."""
        self.current_population = [
            self.create_random_genome(self.generation) 
            for _ in range(size)
        ]
        logger.info(f"Generated population of {size} genomes")
        return self.current_population
    
    def genome_to_params(self, genome: FractalGenome):
        """Convert genome to FractalParams for renderer."""
        from renderers.python_3d import FractalParams
        return FractalParams(
            fractal_type=genome.fractal_type,
            power=genome.power,
            iterations=genome.iterations,
            bailout=genome.bailout,
            scale=genome.scale,
            min_r=genome.min_r,
            ifs_scale=genome.ifs_scale,
            ifs_folds=genome.ifs_folds,
            lambda_val=genome.lambda_val,
            julia_c=genome.julia_c,
            camera_pos=genome.camera_pos,
            target=genome.target,
            fov=genome.fov,
            color_palette=genome.color_palette,
            color_intensity=genome.color_intensity,
            coloring_mode=genome.coloring_mode,
            width=400,
            height=400
        )
    
    def render_population(self, population: List[FractalGenome], 
                         output_dir: str = "output/evolution") -> List[str]:
        """Render entire population to images."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        image_paths = []
        for i, genome in enumerate(population):
            try:
                params = self.genome_to_params(genome)
                image, _ = self.renderer.render(params)
                
                # Save
                filename = f"gen{genome.generation:03d}_{genome.genome_id}.png"
                filepath = output_path / filename
                plt.imsave(filepath, image)
                
                genome.image_path = str(filepath)
                image_paths.append(str(filepath))
                logger.info(f"Rendered {i+1}/{len(population)}: {genome.genome_id}")
                
            except Exception as e:
                logger.error(f"Failed to render {genome.genome_id}: {e}")
                image_paths.append(None)
        
        return image_paths
    
    def record_selections(self, selected_ids: List[str], ranks: Dict[str, int] = None):
        """Record user's selections (which genomes they liked)."""
        ranks = ranks or {}
        for gid in selected_ids:
            for genome in self.current_population:
                if genome.genome_id == gid:
                    rank = ranks.get(gid, 1)
                    self.selection_history.append({
                        "genome_id": gid,
                        "generation": self.generation,
                        "rank": rank,
                        "genome": genome.to_dict()
                    })
                    break
        logger.info(f"Recorded {len(selected_ids)} selections")
    
    def mutate_genome(self, genome: FractalGenome, mutation_rate: float = 0.3) -> FractalGenome:
        """Create a mutated copy of a genome."""
        import copy
        new_genome = copy.deepcopy(genome)
        new_genome.genome_id = str(uuid.uuid4())[:8]
        new_genome.generation = self.generation + 1
        
        if random.random() < mutation_rate:
            new_genome.power = max(2, min(16, new_genome.power * random.uniform(0.7, 1.4)))
        
        if random.random() < mutation_rate:
            new_genome.scale = new_genome.scale * random.uniform(0.8, 1.2)
        
        if random.random() < mutation_rate:
            cam = new_genome.camera_pos
            new_genome.camera_pos = (
                cam[0] + random.uniform(-0.3, 0.3),
                cam[1] + random.uniform(-0.3, 0.3),
                max(-5, min(-1.5, cam[2] + random.uniform(-0.5, 0.5)))
            )
        
        if random.random() < mutation_rate:
            new_genome.color_palette = random.choice(self.COLOR_PALETTES)
        
        if random.random() < mutation_rate:
            new_genome.color_intensity = max(0.3, min(3.0, new_genome.color_intensity * random.uniform(0.8, 1.2)))
        
        if random.random() < mutation_rate * 0.5:
            new_genome.fractal_type = random.choice(self.FRACTAL_TYPES)
        
        return new_genome
    
    def crossover(self, parent1: FractalGenome, parent2: FractalGenome) -> FractalGenome:
        """Create offspring from two parents."""
        import copy
        child = copy.deepcopy(parent1)
        child.genome_id = str(uuid.uuid4())[:8]
        child.generation = self.generation + 1
        
        # Blend numeric parameters
        if random.random() < 0.5:
            child.power = (parent1.power + parent2.power) / 2
        if random.random() < 0.5:
            child.scale = (parent1.scale + parent2.scale) / 2
        if random.random() < 0.5:
            child.camera_pos = tuple(
                (a + b) / 2 for a, b in zip(parent1.camera_pos, parent2.camera_pos)
            )
        
        return child
    
    def evolve_next_generation(self, elite_count: int = 2) -> List[FractalGenome]:
        """Create next generation from selections."""
        if not self.selection_history:
            logger.warning("No selections recorded, generating random population")
            return self.generate_population(len(self.current_population))
        
        # Get selected genomes from history
        selected = [s["genome"] for s in self.selection_history[-10:] if s["rank"] <= 3]
        if not selected:
            selected = [self.selection_history[-1]["genome"]]
        
        # Convert to genome objects
        parents = [FractalGenome.from_dict(s) for s in selected[:3]]
        
        new_population = []
        
        # Keep elites (best from previous)
        best_parents = sorted(parents, key=lambda x: x.fitness, reverse=True)[:elite_count]
        for p in best_parents:
            p.generation = self.generation + 1
            p.genome_id = str(uuid.uuid4())[:8]
            new_population.append(p)
        
        # Create offspring
        target_size = len(self.current_population)
        while len(new_population) < target_size:
            if len(parents) >= 2:
                parent1, parent2 = random.sample(parents, 2)
                if random.random() < 0.7:
                    child = self.crossover(parent1, parent2)
                    child = self.mutate_genome(child, 0.4)
                else:
                    child = self.mutate_genome(parent1, 0.5)
            else:
                child = self.mutate_genome(parents[0], 0.5)
            
            new_population.append(child)
        
        self.generation += 1
        self.current_population = new_population
        logger.info(f"Evolved to generation {self.generation} with {len(new_population)} genomes")
        
        return new_population
    
    def save_population(self, output_dir: str = "output/evolution"):
        """Save current population to disk."""
        p = Path(output_dir)
        p.mkdir(parents=True, exist_ok=True)
        
        for genome in self.current_population:
            filename = p / f"genome_{genome.genome_id}.json"
            with open(filename, 'w') as f:
                json.dump(genome.to_dict(), f, indent=2)
        
        # Save generation info
        with open(p / "evolution_state.json", 'w') as f:
            json.dump({
                "generation": self.generation,
                "population_size": len(self.current_population),
                "total_selections": len(self.selection_history)
            }, f, indent=2)
        
        logger.info(f"Saved population to {output_dir}")
    
    def load_population(self, input_dir: str) -> bool:
        """Load population from disk."""
        p = Path(input_dir)
        if not p.exists():
            return False
        
        # Load generation state
        state_file = p / "evolution_state.json"
        if state_file.exists():
            with open(state_file) as f:
                state = json.load(f)
                self.generation = state.get("generation", 0)
        
        # Load genomes
        self.current_population = []
        for f in p.glob("genome_*.json"):
            with open(f) as fp:
                data = json.load(fp)
                self.current_population.append(FractalGenome.from_dict(data))
        
        logger.info(f"Loaded {len(self.current_population)} genomes")
        return True


# Animation evolution
class AnimationEvolver(FractalEvolver):
    """Evolution for animation parameters."""
    
    def create_random_genome(self, generation: int = 0) -> Dict:
        """Create random animation parameters."""
        return {
            "genome_id": str(uuid.uuid4())[:8],
            "generation": generation,
            "anim_type": random.choice(["power", "camera_orbit", "color_morph"]),
            "start_power": random.uniform(2, 16),
            "end_power": random.uniform(2, 16),
            "start_cam": (random.uniform(-2, 2), random.uniform(-2, 2), random.uniform(-4, -2)),
            "end_cam": (random.uniform(-2, 2), random.uniform(-2, 2), random.uniform(-4, -2)),
            "start_palette": random.choice(self.COLOR_PALETTES),
            "end_palette": random.choice(self.COLOR_PALETTES),
            "num_frames": random.choice([15, 30, 60]),
            "fps": random.choice([8, 10, 15]),
            "fractal_type": random.choice(self.FRACTAL_TYPES)
        }


def quick_evolution_demo():
    """Quick demo of the evolution system."""
    print("=" * 60)
    print("FRACTAL EVOLUTION DEMO")
    print("=" * 60)
    
    evolver = FractalEvolver()
    
    # Generate initial population
    print("\n1. Generating initial population...")
    population = evolver.generate_population(size=6)
    print(f"   Created {len(population)} genomes")
    
    # Render
    print("\n2. Rendering population...")
    image_paths = evolver.render_population(population, "output/evolution_demo")
    print(f"   Rendered to output/evolution_demo/")
    
    # Simulate user selection (in real app, user picks)
    selected_ids = [p.genome_id for p in population[:3]]
    print(f"\n3. User selections: {selected_ids}")
    evolver.record_selections(selected_ids, {gid: i+1 for i, gid in enumerate(selected_ids)})
    
    # Evolve
    print("\n4. Evolving next generation...")
    next_gen = evolver.evolve_next_generation()
    print(f"   Generation {evolver.generation}: {len(next_gen)} genomes")
    
    # Save
    evolver.save_population("output/evolution_demo")
    print("\n5. Saved to output/evolution_demo/")
    
    return evolver


if __name__ == "__main__":
    quick_evolution_demo()