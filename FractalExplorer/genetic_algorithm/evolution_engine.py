"""
Evolution Engine for Fractal Genetic Algorithm

This module implements the main evolutionary algorithm that drives fractal parameter evolution
based on user preferences. It uses tournament selection based on user choices rather than
numeric fitness scores, making it more intuitive and user-friendly.
"""

import random
import logging
import time
from typing import List, Optional, Dict, Any, Callable, Tuple
from dataclasses import dataclass, field
import numpy as np

from shared.genome import FractalGenome, RendererType
from .selection import SelectionStrategy, TournamentSelection
from .population import Population

logger = logging.getLogger(__name__)


@dataclass
class EvolutionConfig:
    """Configuration parameters for the evolution engine."""
    population_size: int = 20
    elite_size: int = 4  # Number of top individuals to preserve
    tournament_size: int = 4  # Size for tournament selection
    mutation_rate: float = 0.15
    mutation_strength: float = 0.1
    crossover_rate: float = 0.8
    diversity_weight: float = 0.2  # How much to weight diversity vs. fitness
    max_generations: int = 100
    convergence_threshold: float = 0.01  # Stop if population converges
    user_selection_batch_size: int = 4  # How many options to show user
    

class EvolutionEngine:
    """
    Main evolutionary algorithm engine.
    
    Manages the genetic algorithm process, including population evolution,
    selection based on user preferences, and diversity maintenance.
    """
    
    def __init__(self, 
                 config: EvolutionConfig = None,
                 renderer_type: RendererType = RendererType.MANDELBULBER):
        """
        Initialize the evolution engine.
        
        Args:
            config: Evolution configuration parameters
            renderer_type: Type of fractal renderer to target
        """
        self.config = config or EvolutionConfig()
        self.renderer_type = renderer_type
        
        # Core components
        self.population = Population(self.config.population_size)
        self.selection_strategy = TournamentSelection(self.config.tournament_size)
        
        # Evolution state
        self.current_generation = 0
        self.evolution_history: List[Dict[str, Any]] = []
        self.user_preferences: List[Tuple[List[str], int]] = []  # (genome_ids, chosen_index)
        
        # Statistics
        self.stats = {
            'total_user_selections': 0,
            'generation_times': [],
            'diversity_scores': [],
            'user_satisfaction_trend': []
        }
        
        # Callbacks
        self.on_generation_complete: Optional[Callable[[int, Dict[str, Any]], None]] = None
        self.on_user_selection_needed: Optional[Callable[[List[FractalGenome]], int]] = None
        
    def initialize_population(self, seed_genomes: Optional[List[FractalGenome]] = None):
        """
        Initialize the population with seed genomes or random individuals.
        
        Args:
            seed_genomes: Optional list of seed genomes to start with
        """
        self.population.initialize(seed_genomes, self.renderer_type)
        self.current_generation = 0
        
        logger.info(f"Initialized population with {len(self.population)} individuals")
    
    def evolve_generation(self) -> Dict[str, Any]:
        """
        Evolve one generation of the population.
        
        Returns:
            Dictionary with generation statistics and metrics
        """
        generation_start_time = time.time()
        
        # Select parents for next generation
        parents = self._select_parents()
        
        # Create offspring through crossover and mutation
        offspring = self._create_offspring(parents)
        
        # Create new population (elitism + offspring)
        new_population = self._create_next_generation(offspring)
        
        # Update population
        old_population = self.population.individuals.copy()
        self.population.individuals = new_population
        
        # Calculate generation statistics
        generation_stats = self._calculate_generation_stats(old_population)
        generation_stats['generation'] = self.current_generation + 1
        generation_stats['time_elapsed'] = time.time() - generation_start_time
        
        # Update evolution history
        self.evolution_history.append(generation_stats)
        self.current_generation += 1
        
        # Update internal statistics
        self.stats['generation_times'].append(generation_stats['time_elapsed'])
        self.stats['diversity_scores'].append(generation_stats['diversity'])
        
        logger.info(f\"Generation {self.current_generation} complete: \"\n                   f\"diversity={generation_stats['diversity']:.3f}, \"\n                   f\"time={generation_stats['time_elapsed']:.2f}s\")\n        \n        # Call generation complete callback\n        if self.on_generation_complete:\n            self.on_generation_complete(self.current_generation, generation_stats)\n        \n        return generation_stats\n    \n    def _select_parents(self) -> List[FractalGenome]:\n        \"\"\"Select parents for reproduction based on fitness and diversity.\"\"\"\n        # Get fitness scores (from user selections + AI predictions)\n        fitness_scores = [genome.fitness or 0.0 for genome in self.population.individuals]\n        \n        # Calculate diversity scores\n        diversity_scores = self._calculate_diversity_scores()\n        \n        # Combine fitness and diversity with weighting\n        combined_scores = []\n        for i in range(len(self.population.individuals)):\n            fitness = fitness_scores[i]\n            diversity = diversity_scores[i]\n            combined = (1 - self.config.diversity_weight) * fitness + self.config.diversity_weight * diversity\n            combined_scores.append(combined)\n        \n        # Select parents using tournament selection\n        num_parents = self.config.population_size - self.config.elite_size\n        parents = []\n        \n        for _ in range(num_parents):\n            parent = self.selection_strategy.select(self.population.individuals, combined_scores)\n            parents.append(parent)\n        \n        return parents\n    \n    def _create_offspring(self, parents: List[FractalGenome]) -> List[FractalGenome]:\n        \"\"\"Create offspring through crossover and mutation.\"\"\"\n        offspring = []\n        \n        # Create pairs of parents for crossover\n        for i in range(0, len(parents), 2):\n            parent1 = parents[i]\n            parent2 = parents[(i + 1) % len(parents)]  # Wrap around if odd number\n            \n            if random.random() < self.config.crossover_rate:\n                # Perform crossover\n                child1 = parent1.crossover(parent2)\n                child2 = parent2.crossover(parent1)\n            else:\n                # Copy parents directly\n                child1 = parent1.copy()\n                child2 = parent2.copy()\n            \n            # Apply mutation\n            if random.random() < self.config.mutation_rate:\n                child1.mutate(self.config.mutation_rate, self.config.mutation_strength)\n            \n            if random.random() < self.config.mutation_rate:\n                child2.mutate(self.config.mutation_rate, self.config.mutation_strength)\n            \n            offspring.extend([child1, child2])\n        \n        return offspring\n    \n    def _create_next_generation(self, offspring: List[FractalGenome]) -> List[FractalGenome]:\n        \"\"\"Create the next generation using elitism and offspring.\"\"\"\n        # Sort current population by fitness\n        current_sorted = sorted(self.population.individuals, \n                               key=lambda g: g.fitness or 0.0, reverse=True)\n        \n        # Select elite individuals\n        elite = current_sorted[:self.config.elite_size]\n        \n        # Select best offspring to fill remaining spots\n        remaining_spots = self.config.population_size - self.config.elite_size\n        selected_offspring = offspring[:remaining_spots]\n        \n        # Combine elite and offspring\n        next_generation = elite + selected_offspring\n        \n        return next_generation\n    \n    def _calculate_diversity_scores(self) -> List[float]:\n        \"\"\"Calculate diversity scores for all individuals in population.\"\"\"\n        diversity_scores = []\n        \n        for i, individual in enumerate(self.population.individuals):\n            # Calculate average distance to all other individuals\n            total_distance = 0.0\n            for j, other in enumerate(self.population.individuals):\n                if i != j:\n                    total_distance += individual.calculate_diversity(other)\n            \n            avg_distance = total_distance / (len(self.population.individuals) - 1)\n            diversity_scores.append(avg_distance)\n        \n        return diversity_scores\n    \n    def _calculate_generation_stats(self, old_population: List[FractalGenome]) -> Dict[str, Any]:\n        \"\"\"Calculate statistics for the current generation.\"\"\"\n        fitness_values = [g.fitness or 0.0 for g in self.population.individuals]\n        diversity_scores = self._calculate_diversity_scores()\n        \n        stats = {\n            'avg_fitness': np.mean(fitness_values),\n            'max_fitness': np.max(fitness_values),\n            'min_fitness': np.min(fitness_values),\n            'fitness_std': np.std(fitness_values),\n            'diversity': np.mean(diversity_scores),\n            'diversity_std': np.std(diversity_scores),\n            'population_size': len(self.population.individuals),\n            'unique_genomes': len(set(g.genome_id for g in self.population.individuals))\n        }\n        \n        return stats\n    \n    def get_candidates_for_user_selection(self, count: int = 4) -> List[FractalGenome]:\n        \"\"\"\n        Get candidate genomes for user selection.\n        \n        Selects a diverse set of individuals from the population,\n        balancing between high fitness and high diversity.\n        \n        Args:\n            count: Number of candidates to return\n            \n        Returns:\n            List of candidate genomes\n        \"\"\"\n        if len(self.population.individuals) < count:\n            return self.population.individuals.copy()\n        \n        candidates = []\n        remaining = self.population.individuals.copy()\n        \n        # First, select the highest fitness individual\n        if remaining:\n            best = max(remaining, key=lambda g: g.fitness or 0.0)\n            candidates.append(best)\n            remaining.remove(best)\n        \n        # Then select individuals that maximize diversity from already selected\n        while len(candidates) < count and remaining:\n            best_candidate = None\n            best_diversity = -1\n            \n            for candidate in remaining:\n                # Calculate minimum distance to already selected candidates\n                min_distance = min(candidate.calculate_diversity(selected) \n                                 for selected in candidates)\n                \n                if min_distance > best_diversity:\n                    best_diversity = min_distance\n                    best_candidate = candidate\n            \n            if best_candidate:\n                candidates.append(best_candidate)\n                remaining.remove(best_candidate)\n            else:\n                # Fallback: just take next available\n                candidates.append(remaining[0])\n                remaining.pop(0)\n        \n        return candidates\n    \n    def record_user_selection(self, candidates: List[FractalGenome], selected_index: int):\n        \"\"\"\n        Record a user selection for fitness evaluation.\n        \n        Args:\n            candidates: List of candidate genomes that were presented\n            selected_index: Index of selected genome (or -1 for skip)\n        \"\"\"\n        candidate_ids = [g.genome_id for g in candidates]\n        self.user_preferences.append((candidate_ids, selected_index))\n        self.stats['total_user_selections'] += 1\n        \n        # Update fitness based on selection\n        if 0 <= selected_index < len(candidates):\n            selected_genome = candidates[selected_index]\n            \n            # Increase fitness for selected genome\n            current_fitness = selected_genome.fitness or 0.0\n            selected_genome.fitness = current_fitness + 1.0\n            \n            # Slightly decrease fitness for non-selected candidates\n            for i, candidate in enumerate(candidates):\n                if i != selected_index:\n                    current_fitness = candidate.fitness or 0.0\n                    candidate.fitness = max(0.0, current_fitness - 0.1)\n        \n        logger.info(f\"Recorded user selection: {selected_index} out of {len(candidates)} candidates\")\n    \n    def run_evolution_cycle(self) -> bool:\n        \"\"\"\n        Run one complete evolution cycle (generation + user selection).\n        \n        Returns:\n            True if evolution should continue, False if stopping criteria met\n        \"\"\"\n        # Evolve one generation\n        gen_stats = self.evolve_generation()\n        \n        # Check stopping criteria\n        if self.current_generation >= self.config.max_generations:\n            logger.info(f\"Reached maximum generations ({self.config.max_generations})\")\n            return False\n        \n        if gen_stats['diversity'] < self.config.convergence_threshold:\n            logger.info(f\"Population converged (diversity={gen_stats['diversity']:.4f})\")\n            return False\n        \n        # Get candidates for user selection\n        candidates = self.get_candidates_for_user_selection()\n        \n        # Request user selection\n        if self.on_user_selection_needed:\n            selected_index = self.on_user_selection_needed(candidates)\n            self.record_user_selection(candidates, selected_index)\n        \n        return True\n    \n    def get_best_genomes(self, count: int = 5) -> List[FractalGenome]:\n        \"\"\"\n        Get the best genomes from current population.\n        \n        Args:\n            count: Number of top genomes to return\n            \n        Returns:\n            List of best genomes sorted by fitness\n        \"\"\"\n        sorted_population = sorted(self.population.individuals, \n                                 key=lambda g: g.fitness or 0.0, reverse=True)\n        return sorted_population[:count]\n    \n    def get_evolution_summary(self) -> Dict[str, Any]:\n        \"\"\"\n        Get a summary of the evolution process.\n        \n        Returns:\n            Dictionary with evolution statistics and history\n        \"\"\"\n        return {\n            'current_generation': self.current_generation,\n            'total_user_selections': self.stats['total_user_selections'],\n            'evolution_history': self.evolution_history,\n            'best_genomes': [g.to_dict() for g in self.get_best_genomes(3)],\n            'population_diversity': np.mean(self._calculate_diversity_scores()) if self.population.individuals else 0.0,\n            'config': {\n                'population_size': self.config.population_size,\n                'mutation_rate': self.config.mutation_rate,\n                'crossover_rate': self.config.crossover_rate\n            }\n        }"
