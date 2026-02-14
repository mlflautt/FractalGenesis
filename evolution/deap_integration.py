#!/usr/bin/env python3
"""
DEAP Evolution Integration
==========================

Advanced evolutionary algorithms using DEAP:
- Novelty Search (rewards exploration)
- CMA-ES (state-of-the-art continuous optimization)
- NSGA-II (multi-objective optimization)
- Island Model (parallel populations with migration)

Usage:
    from evolution.deap_integration import NoveltySearch
    
    # Create and run evolution
    search = NoveltySearch(formula_registry)
    results = search.evolve(population_size=100, generations=50)
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import random
import copy
from collections import deque
from pathlib import Path
import json

# DEAP imports
try:
    from deap import base, creator, tools, algorithms, cma
    DEAP_AVAILABLE = True
except ImportError:
    DEAP_AVAILABLE = False
    print("Warning: DEAP not installed. Install with: pip install deap")

from shared.genome import FractalGenome
from formulas import FormulaRegistry, FormulaRandomSearch


@dataclass
class EvolutionResult:
    """Result of an evolution run"""
    best_individual: FractalGenome
    population: List[FractalGenome]
    generation: int
    fitness_history: List[float]
    novelty_history: List[float]
    phylogenetic_tree: Dict[str, Any]
    

def ensure_deap():
    """Ensure DEAP is available"""
    if not DEAP_AVAILABLE:
        raise ImportError("DEAP is required. Install with: pip install deap")


class BaseEvolutionStrategy(ABC):
    """Abstract base class for evolution strategies"""
    
    def __init__(self, formula_registry: FormulaRegistry):
        self.formula_registry = formula_registry
        self.generation = 0
        self.history = []
        
    @abstractmethod
    def evolve(self, population_size: int, generations: int) -> EvolutionResult:
        """Run evolution and return results"""
        pass
    
    def genome_to_deap(self, genome: FractalGenome) -> List[float]:
        """Convert genome to DEAP-compatible float list"""
        # Flatten genome parameters into a vector
        vector = []
        
        # Camera position (3)
        vector.extend(genome.camera.position)
        
        # Camera target (3)
        vector.extend(genome.camera.target)
        
        # Fractal power (1)
        vector.append(genome.fractal.power)
        
        # Fractal iterations (1, normalized)
        vector.append(genome.fractal.iterations / 500.0)
        
        # Color base (3)
        vector.extend(genome.color.base_color)
        
        # Add more parameters as needed
        return vector
    
    def deap_to_genome(self, vector: List[float]) -> FractalGenome:
        """Convert DEAP float list back to genome"""
        genome = FractalGenome()
        
        idx = 0
        
        # Camera position (3)
        genome.camera.position = tuple(vector[idx:idx+3])
        idx += 3
        
        # Camera target (3)
        genome.camera.target = tuple(vector[idx:idx+3])
        idx += 3
        
        # Fractal power (1)
        genome.fractal.power = max(0.1, vector[idx])
        idx += 1
        
        # Fractal iterations (1, denormalize)
        genome.fractal.iterations = int(max(10, vector[idx] * 500))
        idx += 1
        
        # Color base (3)
        genome.color.base_color = tuple(max(0, min(1, v)) for v in vector[idx:idx+3])
        idx += 3
        
        return genome


class NoveltySearch(BaseEvolutionStrategy):
    """
    Novelty Search Algorithm
    
    Unlike traditional GA that optimizes fitness,
    novelty search rewards exploring unvisited regions of parameter space.
    
    Key idea: Novelty is measured by distance from archive of past behaviors.
    This encourages exploration and often discovers better solutions than
    objective-driven search in deceptive landscapes.
    """
    
    def __init__(self, 
                 formula_registry: FormulaRegistry,
                 k_neighbors: int = 15,
                 archive_threshold: float = 0.3,
                 archive_limit: int = 1000):
        super().__init__(formula_registry)
        self.k_neighbors = k_neighbors
        self.archive_threshold = archive_threshold
        self.archive_limit = archive_limit
        self.archive = []
        self.novelty_scores = []
        
    def calculate_novelty(self, individual: List[float]) -> float:
        """
        Calculate novelty as average distance to k nearest neighbors
        in behavior space (archive + current population).
        """
        ensure_deap()
        
        # Combine archive and current population
        all_behaviors = self.archive + [ind for ind in self.history if isinstance(ind, list)]
        
        if len(all_behaviors) < self.k_neighbors:
            # Not enough data, consider highly novel
            return 1.0
        
        # Calculate distances to all archived behaviors
        distances = []
        for behavior in all_behaviors:
            dist = np.sqrt(sum((a - b) ** 2 for a, b in zip(individual, behavior)))
            distances.append(dist)
        
        # Sort and take k nearest
        distances.sort()
        k_nearest = distances[:self.k_neighbors]
        
        # Novelty is average distance to k nearest
        novelty = sum(k_nearest) / len(k_nearest)
        
        return novelty
    
    def add_to_archive(self, individual: List[float], novelty: float):
        """Add individual to archive if sufficiently novel"""
        if novelty > self.archive_threshold:
            self.archive.append(copy.deepcopy(individual))
            
            # Limit archive size (FIFO)
            if len(self.archive) > self.archive_limit:
                self.archive.pop(0)
    
    def evolve(self, 
               population_size: int = 100,
               generations: int = 50,
               crossover_prob: float = 0.7,
               mutation_prob: float = 0.2) -> EvolutionResult:
        """Run novelty search evolution"""
        ensure_deap()
        
        # Set up DEAP
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        toolbox = base.Toolbox()
        
        # Genome size (adjust based on parameters tracked)
        GENOME_SIZE = 11  # 3 pos + 3 target + 1 power + 1 iter + 3 color
        
        # Attribute generator
        toolbox.register("attr_float", random.uniform, -1, 1)
        
        # Structure initializers
        toolbox.register("individual", tools.initRepeat, creator.Individual,
                        toolbox.attr_float, n=GENOME_SIZE)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        # Genetic operators
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.1)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        # Create initial population
        pop = toolbox.population(n=population_size)
        
        # Evaluate initial population (all have novelty 0 since archive is empty)
        for ind in pop:
            novelty = self.calculate_novelty(ind)
            ind.fitness.values = (novelty,)
            self.novelty_scores.append(novelty)
        
        # Add to archive
        for ind in pop:
            self.add_to_archive(list(ind), ind.fitness.values[0])
        
        # Evolution loop
        fitness_history = []
        novelty_history = []
        
        for gen in range(generations):
            self.generation = gen
            
            # Select and clone the next generation individuals
            offspring = list(map(toolbox.clone, toolbox.select(pop, len(pop))))
            
            # Apply crossover and mutation
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < crossover_prob:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            for mutant in offspring:
                if random.random() < mutation_prob:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            # Evaluate invalid individuals (novelty calculation)
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            for ind in invalid_ind:
                novelty = self.calculate_novelty(ind)
                ind.fitness.values = (novelty,)
                self.novelty_scores.append(novelty)
                
                # Add to archive if novel enough
                self.add_to_archive(list(ind), novelty)
            
            # Replace population
            pop[:] = offspring
            
            # Record statistics
            fits = [ind.fitness.values[0] for ind in pop]
            fitness_history.append(max(fits))
            novelty_history.append(np.mean(self.novelty_scores[-population_size:]))
            
            if gen % 10 == 0:
                print(f"Generation {gen}: Max novelty = {max(fits):.4f}, "
                      f"Archive size = {len(self.archive)}")
        
        # Convert best individual back to genome
        best_deap = tools.selBest(pop, 1)[0]
        best_genome = self.deap_to_genome(best_deap)
        
        # Convert population to genomes
        population_genomes = [self.deap_to_genome(ind) for ind in pop]
        
        # Build phylogenetic tree
        phylogenetic_tree = self.build_phylogenetic_tree(pop)
        
        return EvolutionResult(
            best_individual=best_genome,
            population=population_genomes,
            generation=generations,
            fitness_history=fitness_history,
            novelty_history=novelty_history,
            phylogenetic_tree=phylogenetic_tree
        )
    
    def build_phylogenetic_tree(self, population: List) -> Dict[str, Any]:
        """Build phylogenetic tree from population"""
        # Simplified tree structure
        tree = {
            'root': 'generation_0',
            'generations': {},
            'archive_size': len(self.archive)
        }
        
        for gen in range(self.generation + 1):
            tree['generations'][f'gen_{gen}'] = {
                'population_size': len(population),
                'avg_novelty': np.mean(self.novelty_scores[gen*len(population):(gen+1)*len(population)]) 
                               if len(self.novelty_scores) > gen*len(population) else 0
            }
        
        return tree


class CMAEvolutionStrategy(BaseEvolutionStrategy):
    """
    CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
    
    State-of-the-art algorithm for continuous optimization.
    Self-adapts mutation distribution based on successful directions.
    Particularly effective for fractal parameter optimization.
    """
    
    def __init__(self, formula_registry: FormulaRegistry):
        super().__init__(formula_registry)
        self.fitness_function = None
        
    def set_fitness_function(self, func: Callable[[FractalGenome], float]):
        """Set custom fitness function"""
        self.fitness_function = func
    
    def default_fitness(self, genome: FractalGenome) -> float:
        """Default fitness based on diversity and complexity"""
        # Higher power = more complex = higher fitness
        power_score = min(genome.fractal.power / 16.0, 1.0)
        
        # Iteration balance
        iter_score = 1.0 - abs(genome.fractal.iterations - 100) / 200.0
        
        # Distance from origin (camera position diversity)
        camera_dist = np.sqrt(sum(x**2 for x in genome.camera.position))
        diversity_score = min(camera_dist / 5.0, 1.0)
        
        return power_score * 0.4 + iter_score * 0.3 + diversity_score * 0.3
    
    def evolve(self,
               population_size: int = 50,
               generations: int = 100,
               initial_params: Optional[FractalGenome] = None) -> EvolutionResult:
        """Run CMA-ES evolution"""
        ensure_deap()
        
        # Set up DEAP for minimization (CMA-ES minimizes)
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        
        GENOME_SIZE = 11
        
        # Initialize CMA-ES strategy
        if initial_params:
            centroid = self.genome_to_deap(initial_params)
        else:
            centroid = [0.0] * GENOME_SIZE
        
        # CMA-ES parameters
        sigma = 0.5  # Initial standard deviation
        
        strategy = cma.Strategy(centroid=centroid, sigma=sigma,
                               lambda_=population_size)
        
        toolbox = base.Toolbox()
        toolbox.register("generate", strategy.generate, creator.Individual)
        toolbox.register("update", strategy.update)
        
        # Evolution loop
        fitness_function = self.fitness_function or self.default_fitness
        fitness_history = []
        
        for gen in range(generations):
            self.generation = gen
            
            # Generate new population
            population = toolbox.generate()
            
            # Evaluate fitness (minimize negative fitness = maximize)
            fitnesses = []
            for ind in population:
                genome = self.deap_to_genome(ind)
                fitness = -fitness_function(genome)  # Negate for minimization
                fitnesses.append((fitness,))
            
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit
            
            # Update strategy
            toolbox.update(population)
            
            # Record
            fits = [-f[0] for f in fitnesses]  # Back to positive
            fitness_history.append(max(fits))
            
            if gen % 10 == 0:
                print(f"CMA-ES Gen {gen}: Best fitness = {max(fits):.4f}, "
                      f"Sigma = {strategy.sigma:.4f}")
        
        # Get results
        best_deap = tools.selBest(population, 1)[0]
        best_genome = self.deap_to_genome(best_deap)
        population_genomes = [self.deap_to_genome(ind) for ind in population]
        
        return EvolutionResult(
            best_individual=best_genome,
            population=population_genomes,
            generation=generations,
            fitness_history=fitness_history,
            novelty_history=[],
            phylogenetic_tree={}
        )


class NSGA2Evolution(BaseEvolutionStrategy):
    """
    NSGA-II (Non-dominated Sorting Genetic Algorithm II)
    
    Multi-objective optimization for conflicting goals:
    - Maximize quality vs Minimize render time
    - Maximize complexity vs Maximize visual appeal
    - Maximize novelty vs Maximize similarity to favorites
    """
    
    def __init__(self, formula_registry: FormulaRegistry):
        super().__init__(formula_registry)
        self.objectives = []
        
    def add_objective(self, name: str, func: Callable[[FractalGenome], float], maximize: bool = True):
        """Add an objective function"""
        weight = 1.0 if maximize else -1.0
        self.objectives.append((name, func, weight))
    
    def evolve(self,
               population_size: int = 100,
               generations: int = 50) -> EvolutionResult:
        """Run NSGA-II evolution"""
        ensure_deap()
        
        if not self.objectives:
            # Default objectives
            self.add_objective("complexity", lambda g: g.fractal.power / 16.0)
            self.add_objective("diversity", lambda g: np.sqrt(sum(x**2 for x in g.camera.position)) / 5.0)
        
        # Create fitness with multiple weights
        weights = tuple(obj[2] for obj in self.objectives)
        creator.create("FitnessMulti", base.Fitness, weights=weights)
        creator.create("Individual", list, fitness=creator.FitnessMulti)
        
        toolbox = base.Toolbox()
        GENOME_SIZE = 11
        
        toolbox.register("attr_float", random.uniform, -1, 1)
        toolbox.register("individual", tools.initRepeat, creator.Individual,
                        toolbox.attr_float, n=GENOME_SIZE)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        toolbox.register("mate", tools.cxSimulatedBinaryBounded, 
                        low=-1, up=1, eta=20.0)
        toolbox.register("mutate", tools.mutPolynomialBounded,
                        low=-1, up=1, eta=20.0, indpb=1.0/GENOME_SIZE)
        toolbox.register("select", tools.selNSGA2)
        
        # Initialize
        pop = toolbox.population(n=population_size)
        
        # Evaluate first generation
        fitnesses = []
        for ind in pop:
            genome = self.deap_to_genome(ind)
            fits = tuple(obj[1](genome) for obj in self.objectives)
            fitnesses.append(fits)
        
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        
        # Evolution
        fitness_history = []
        
        for gen in range(generations):
            self.generation = gen
            
            offspring = algorithms.varAnd(pop, toolbox, cxpb=0.9, mutpb=0.1)
            
            # Evaluate offspring
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = []
            for ind in invalid_ind:
                genome = self.deap_to_genome(ind)
                fits = tuple(obj[1](genome) for obj in self.objectives)
                fitnesses.append(fits)
            
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # Select next generation
            pop = toolbox.select(pop + offspring, population_size)
            
            # Record
            fits = [sum(ind.fitness.values) for ind in pop]
            fitness_history.append(max(fits))
            
            if gen % 10 == 0:
                print(f"NSGA-II Gen {gen}: Best = {max(fits):.4f}")
        
        # Get Pareto front
        pareto_front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
        
        best_deap = pareto_front[0]
        best_genome = self.deap_to_genome(best_deap)
        population_genomes = [self.deap_to_genome(ind) for ind in pop]
        
        return EvolutionResult(
            best_individual=best_genome,
            population=population_genomes,
            generation=generations,
            fitness_history=fitness_history,
            novelty_history=[],
            phylogenetic_tree={'pareto_front_size': len(pareto_front)}
        )


class IslandModelEvolution:
    """
    Island Model Parallel Evolution
    
    Multiple populations (islands) evolve independently
    with periodic migration of individuals between islands.
    
    Benefits:
    - Maintains diversity
    - Prevents premature convergence
    - Explores different regions simultaneously
    - Can use different strategies per island
    """
    
    def __init__(self, formula_registry: FormulaRegistry, num_islands: int = 4):
        self.formula_registry = formula_registry
        self.num_islands = num_islands
        self.islands = []
        self.migration_interval = 5
        
    def initialize_islands(self, island_size: int):
        """Initialize island populations"""
        ensure_deap()
        
        for i in range(self.num_islands):
            # Each island could use different strategy
            if i % 2 == 0:
                island = NoveltySearch(self.formula_registry)
            else:
                island = CMAEvolutionStrategy(self.formula_registry)
            
            self.islands.append({
                'strategy': island,
                'population': [],
                'generation': 0
            })
    
    def migrate(self, emigrants_per_island: int = 2):
        """Migrate individuals between islands (ring topology)"""
        for i, island in enumerate(self.islands):
            # Select emigrants from this island
            if len(island['population']) > emigrants_per_island:
                emigrants = random.sample(island['population'], emigrants_per_island)
                
                # Send to next island (ring)
                next_island = self.islands[(i + 1) % self.num_islands]
                next_island['population'].extend(emigrants)
                
                # Remove from current (keep size constant)
                for emigrant in emigrants:
                    if emigrant in island['population']:
                        island['population'].remove(emigrant)
    
    def evolve(self,
               island_size: int = 50,
               generations: int = 50) -> List[EvolutionResult]:
        """Run island model evolution"""
        self.initialize_islands(island_size)
        
        results = []
        
        for gen in range(generations):
            # Evolve each island
            for i, island in enumerate(self.islands):
                # Run one generation on island
                strategy = island['strategy']
                # ... evolution logic per island
                island['generation'] += 1
            
            # Migration
            if gen % self.migration_interval == 0:
                self.migrate()
            
            if gen % 10 == 0:
                print(f"Island Model Gen {gen}")
        
        return results


# Convenience function
def run_novelty_search(formula_registry: FormulaRegistry,
                      population_size: int = 100,
                      generations: int = 50) -> EvolutionResult:
    """Run a quick novelty search"""
    search = NoveltySearch(formula_registry)
    return search.evolve(population_size, generations)


def run_cma_es(formula_registry: FormulaRegistry,
               population_size: int = 50,
               generations: int = 100) -> EvolutionResult:
    """Run CMA-ES optimization"""
    cma = CMAEvolutionStrategy(formula_registry)
    return cma.evolve(population_size, generations)


if __name__ == "__main__":
    print("DEAP Evolution Integration Demo")
    print("=" * 60)
    
    if not DEAP_AVAILABLE:
        print("DEAP not installed. Install with: pip install deap")
        exit(1)
    
    # Demo with formula registry
    from formulas import FormulaRegistry
    registry = FormulaRegistry()
    
    print("\n1. Novelty Search Demo")
    print("-" * 60)
    search = NoveltySearch(registry)
    result = search.evolve(population_size=30, generations=20)
    print(f"Best individual: power={result.best_individual.fractal.power:.2f}")
    print(f"Final archive size: {len(search.archive)}")
    
    print("\n2. CMA-ES Demo")
    print("-" * 60)
    cma = CMAEvolutionStrategy(registry)
    result = cma.evolve(population_size=20, generations=30)
    print(f"Best individual: power={result.best_individual.fractal.power:.2f}")
    
    print("\nDemo complete!")
