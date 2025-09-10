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
        
        logger.info(f"Generation {self.current_generation} complete: "
                   f"diversity={generation_stats['diversity']:.3f}, "
                   f"time={generation_stats['time_elapsed']:.2f}s")
        
        # Call generation complete callback
        if self.on_generation_complete:
            self.on_generation_complete(self.current_generation, generation_stats)
        
        return generation_stats
    
    def _select_parents(self) -> List[FractalGenome]:
        """Select parents for reproduction based on fitness and diversity."""
        # Get fitness scores (from user selections + AI predictions)
        fitness_scores = [genome.fitness or 0.0 for genome in self.population.individuals]
        
        # Calculate diversity scores
        diversity_scores = self._calculate_diversity_scores()
        
        # Combine fitness and diversity with weighting
        combined_scores = []
        for i in range(len(self.population.individuals)):
            fitness = fitness_scores[i]
            diversity = diversity_scores[i]
            combined = (1 - self.config.diversity_weight) * fitness + self.config.diversity_weight * diversity
            combined_scores.append(combined)
        
        # Select parents using tournament selection
        num_parents = self.config.population_size - self.config.elite_size
        parents = []
        
        for _ in range(num_parents):
            parent = self.selection_strategy.select(self.population.individuals, combined_scores)
            parents.append(parent)
        
        return parents
    
    def _create_offspring(self, parents: List[FractalGenome]) -> List[FractalGenome]:
        """Create offspring through crossover and mutation."""
        offspring = []
        
        # Create pairs of parents for crossover
        for i in range(0, len(parents), 2):
            parent1 = parents[i]
            parent2 = parents[(i + 1) % len(parents)]  # Wrap around if odd number
            
            if random.random() < self.config.crossover_rate:
                # Perform crossover
                child1 = parent1.crossover(parent2)
                child2 = parent2.crossover(parent1)
            else:
                # Copy parents directly
                child1 = parent1.copy()
                child2 = parent2.copy()
            
            # Apply mutation
            if random.random() < self.config.mutation_rate:
                child1.mutate(self.config.mutation_rate, self.config.mutation_strength)
            
            if random.random() < self.config.mutation_rate:
                child2.mutate(self.config.mutation_rate, self.config.mutation_strength)
            
            offspring.extend([child1, child2])
        
        return offspring
    
    def _create_next_generation(self, offspring: List[FractalGenome]) -> List[FractalGenome]:
        """Create the next generation using elitism and offspring."""
        # Sort current population by fitness
        current_sorted = sorted(self.population.individuals, 
                               key=lambda g: g.fitness or 0.0, reverse=True)
        
        # Select elite individuals
        elite = current_sorted[:self.config.elite_size]
        
        # Select best offspring to fill remaining spots
        remaining_spots = self.config.population_size - self.config.elite_size
        selected_offspring = offspring[:remaining_spots]
        
        # Combine elite and offspring
        next_generation = elite + selected_offspring
        
        return next_generation
    
    def _calculate_diversity_scores(self) -> List[float]:
        """Calculate diversity scores for all individuals in population."""
        diversity_scores = []
        
        for i, individual in enumerate(self.population.individuals):
            # Calculate average distance to all other individuals
            total_distance = 0.0
            for j, other in enumerate(self.population.individuals):
                if i != j:
                    total_distance += individual.calculate_diversity(other)
            
            avg_distance = total_distance / (len(self.population.individuals) - 1)
            diversity_scores.append(avg_distance)
        
        return diversity_scores
    
    def _calculate_generation_stats(self, old_population: List[FractalGenome]) -> Dict[str, Any]:
        """Calculate statistics for the current generation."""
        fitness_values = [g.fitness or 0.0 for g in self.population.individuals]
        diversity_scores = self._calculate_diversity_scores()
        
        stats = {
            'avg_fitness': np.mean(fitness_values),
            'max_fitness': np.max(fitness_values),
            'min_fitness': np.min(fitness_values),
            'fitness_std': np.std(fitness_values),
            'diversity': np.mean(diversity_scores),
            'diversity_std': np.std(diversity_scores),
            'population_size': len(self.population.individuals),
            'unique_genomes': len(set(g.genome_id for g in self.population.individuals))
        }
        
        return stats
    
    def get_candidates_for_user_selection(self, count: int = 4) -> List[FractalGenome]:
        """
        Get candidate genomes for user selection.
        
        Selects a diverse set of individuals from the population,
        balancing between high fitness and high diversity.
        
        Args:
            count: Number of candidates to return
            
        Returns:
            List of candidate genomes
        """
        if len(self.population.individuals) < count:
            return self.population.individuals.copy()
        
        candidates = []
        remaining = self.population.individuals.copy()
        
        # First, select the highest fitness individual
        if remaining:
            best = max(remaining, key=lambda g: g.fitness or 0.0)
            candidates.append(best)
            remaining.remove(best)
        
        # Then select individuals that maximize diversity from already selected
        while len(candidates) < count and remaining:
            best_candidate = None
            best_diversity = -1
            
            for candidate in remaining:
                # Calculate minimum distance to already selected candidates
                min_distance = min(candidate.calculate_diversity(selected) 
                                 for selected in candidates)
                
                if min_distance > best_diversity:
                    best_diversity = min_distance
                    best_candidate = candidate
            
            if best_candidate:
                candidates.append(best_candidate)
                remaining.remove(best_candidate)
            else:
                # Fallback: just take next available
                candidates.append(remaining[0])
                remaining.pop(0)
        
        return candidates
    
    def record_user_selection(self, candidates: List[FractalGenome], selected_index: int):
        """
        Record a user selection for fitness evaluation.
        
        Args:
            candidates: List of candidate genomes that were presented
            selected_index: Index of selected genome (or -1 for skip)
        """
        candidate_ids = [g.genome_id for g in candidates]
        self.user_preferences.append((candidate_ids, selected_index))
        self.stats['total_user_selections'] += 1
        
        # Update fitness based on selection
        if 0 <= selected_index < len(candidates):
            selected_genome = candidates[selected_index]
            
            # Increase fitness for selected genome
            current_fitness = selected_genome.fitness or 0.0
            selected_genome.fitness = current_fitness + 1.0
            
            # Slightly decrease fitness for non-selected candidates
            for i, candidate in enumerate(candidates):
                if i != selected_index:
                    current_fitness = candidate.fitness or 0.0
                    candidate.fitness = max(0.0, current_fitness - 0.1)
        
        logger.info(f"Recorded user selection: {selected_index} out of {len(candidates)} candidates")
    
    def get_best_genomes(self, count: int = 5) -> List[FractalGenome]:
        """
        Get the best genomes from current population.
        
        Args:
            count: Number of top genomes to return
            
        Returns:
            List of best genomes sorted by fitness
        """
        sorted_population = sorted(self.population.individuals, 
                                 key=lambda g: g.fitness or 0.0, reverse=True)
        return sorted_population[:count]
    
    def get_evolution_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the evolution process.
        
        Returns:
            Dictionary with evolution statistics and history
        """
        return {
            'current_generation': self.current_generation,
            'total_user_selections': self.stats['total_user_selections'],
            'evolution_history': self.evolution_history,
            'best_genomes': [g.to_dict() for g in self.get_best_genomes(3)],
            'population_diversity': np.mean(self._calculate_diversity_scores()) if self.population.individuals else 0.0,
            'config': {
                'population_size': self.config.population_size,
                'mutation_rate': self.config.mutation_rate,
                'crossover_rate': self.config.crossover_rate
            }
        }
